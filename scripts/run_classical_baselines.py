#!/usr/bin/env python3
"""Run reproducible molecular baselines on the frozen downstream splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import sklearn
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.data.downstream import DownstreamDataset, load_downstream_dataset, split_dataset
from gtpro.data.pretraining import SmilesProcessingError, tokenize_smiles
from gtpro.metrics import compute_metrics


MODELS = (
    "morgan_logistic_or_ridge",
    "morgan_random_forest",
    "graph_only",
    "smiles_only",
    "graph_smiles_no_alignment",
)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _eligible(dataset: DownstreamDataset, max_tokens: int = 200) -> tuple[DownstreamDataset, int]:
    selected_indices = []
    for index, smiles in enumerate(dataset.canonical_smiles):
        try:
            representable = len(tokenize_smiles(smiles)) <= max_tokens
        except SmilesProcessingError:
            representable = False
        if representable:
            selected_indices.append(index)
    selected = np.asarray(selected_indices, dtype=np.int64)
    dropped = len(dataset) - len(selected)
    if not dropped:
        return dataset, 0
    return DownstreamDataset(
        spec=dataset.spec, source_path=dataset.source_path,
        source_rows=dataset.source_rows[selected],
        identifiers=tuple(dataset.identifiers[index] for index in selected),
        smiles=tuple(dataset.smiles[index] for index in selected),
        canonical_smiles=tuple(dataset.canonical_smiles[index] for index in selected),
        targets=dataset.targets[selected], invalid_molecules=dataset.invalid_molecules,
    ), dropped


def _features(dataset: DownstreamDataset) -> tuple[sparse.csr_matrix, np.ndarray]:
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    morgan = np.zeros((len(dataset), 1024), dtype=np.float32)
    graph = np.zeros((len(dataset), 12), dtype=np.float32)
    for index, smiles in enumerate(dataset.canonical_smiles):
        molecule = Chem.MolFromSmiles(smiles)
        morgan[index] = generator.GetFingerprintAsNumPy(molecule)
        graph[index] = (
            Descriptors.MolWt(molecule), Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule), Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule), Lipinski.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcNumRings(molecule), molecule.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(molecule), Chem.GetFormalCharge(molecule),
            rdMolDescriptors.CalcNumAromaticRings(molecule), rdMolDescriptors.CalcNumAliphaticRings(molecule),
        )
    return sparse.csr_matrix(morgan), graph


def _representation(model_name, dataset, morgan, graph, train, validation, test):
    if model_name.startswith("morgan_"):
        return morgan[train], morgan[validation], morgan[test], 1024, None
    scaler = StandardScaler().fit(graph[train])
    graph_parts = [sparse.csr_matrix(scaler.transform(graph[indices])) for indices in (train, validation, test)]
    if model_name == "graph_only":
        return *graph_parts, graph.shape[1], scaler
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=2048, sublinear_tf=True)
    text_train = vectorizer.fit_transform([dataset.canonical_smiles[index] for index in train])
    text_validation = vectorizer.transform([dataset.canonical_smiles[index] for index in validation])
    text_test = vectorizer.transform([dataset.canonical_smiles[index] for index in test])
    if model_name == "smiles_only":
        return text_train, text_validation, text_test, text_train.shape[1], vectorizer
    return (
        sparse.hstack((graph_parts[0], text_train)).tocsr(),
        sparse.hstack((graph_parts[1], text_validation)).tocsr(),
        sparse.hstack((graph_parts[2], text_test)).tocsr(),
        graph.shape[1] + text_train.shape[1],
        {"scaler": scaler, "vectorizer": vectorizer},
    )


def _fit_target(model_name, task_type, features, labels, seed):
    if task_type == "regression":
        model = (
            RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
            if model_name == "morgan_random_forest" else Ridge(alpha=1.0, solver="lsqr")
        )
    elif model_name == "morgan_random_forest":
        model = RandomForestClassifier(
            n_estimators=50, random_state=seed, n_jobs=-1, class_weight="balanced"
        )
    else:
        model = LogisticRegression(
            max_iter=500, solver="liblinear", class_weight="balanced", random_state=seed
        )
    model.fit(features, labels)
    return model


def _predict(model, task_type, features):
    if task_type == "regression":
        return model.predict(features)
    classes = model.classes_.tolist()
    if 1 not in classes:
        return np.zeros(features.shape[0], dtype=np.float64)
    return model.predict_proba(features)[:, classes.index(1)]


def _train_models(model_name, task_type, train_x, targets, train_indices, seed):
    models = []
    for target_index in range(targets.shape[1]):
        labels = targets[train_indices, target_index]
        valid = np.isfinite(labels)
        if not valid.any():
            models.append({"constant": 0.0, "reason": "no_valid_training_labels"})
        elif task_type != "regression" and np.unique(labels[valid]).size < 2:
            models.append({"constant": float(labels[valid][0]), "reason": "single_training_class"})
        else:
            models.append(_fit_target(model_name, task_type, train_x[valid], labels[valid], seed + target_index))
    return models


def _predict_models(models, task_type, features):
    predictions = np.zeros((features.shape[0], len(models)), dtype=np.float64)
    for index, model in enumerate(models):
        predictions[:, index] = (
            model["constant"] if isinstance(model, dict) else _predict(model, task_type, features)
        )
    return predictions


def _parameter_count(models) -> tuple[int, str]:
    total = 0
    definition = "fitted scalar coefficients"
    for model in models:
        if isinstance(model, dict):
            total += 1
        elif hasattr(model, "coef_"):
            total += int(model.coef_.size + model.intercept_.size)
        elif hasattr(model, "estimators_"):
            definition = "total fitted decision-tree nodes"
            total += sum(estimator.tree_.node_count for estimator in model.estimators_)
    return total, definition


def _write_predictions(path, dataset, indices, targets, predictions):
    fields = ["dataset_index", "source_row", "smiles"]
    fields += [f"target_{name}" for name in dataset.spec.target_columns]
    fields += [f"prediction_{name}" for name in dataset.spec.target_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_position, index in enumerate(indices):
            row = {"dataset_index": int(index), "source_row": int(dataset.source_rows[index]),
                   "smiles": dataset.canonical_smiles[index]}
            for target_index, name in enumerate(dataset.spec.target_columns):
                target = targets[row_position, target_index]
                row[f"target_{name}"] = "" if not np.isfinite(target) else float(target)
                row[f"prediction_{name}"] = float(predictions[row_position, target_index])
            writer.writerow(row)


def run_one(dataset, split_name, seed, model_name, morgan, graph, output_root, long_dropped):
    run_dir = output_root / f"{dataset.spec.name}_{split_name}" / model_name / str(seed)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"Skipping completed baseline: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    clock = monotonic()
    split = split_dataset(dataset, method=split_name, fractions=(0.8, 0.1, 0.1), seed=seed)
    train_x, validation_x, test_x, feature_count, transformer = _representation(
        model_name, dataset, morgan, graph, split.train, split.validation, split.test
    )
    models = _train_models(model_name, dataset.spec.task_type, train_x, dataset.targets, split.train, seed)
    validation_predictions = _predict_models(models, dataset.spec.task_type, validation_x)
    test_predictions = _predict_models(models, dataset.spec.task_type, test_x)
    validation_targets = dataset.targets[split.validation]
    test_targets = dataset.targets[split.test]
    validation_metrics = compute_metrics(
        dataset.spec.task_type, validation_targets, validation_predictions, dataset.spec.target_columns
    )
    test_metrics = compute_metrics(
        dataset.spec.task_type, test_targets, test_predictions, dataset.spec.target_columns
    )
    parameters, parameter_definition = _parameter_count(models)
    config = {
        "dataset": dataset.spec.name, "split": split_name, "fractions": [0.8, 0.1, 0.1],
        "seed": seed, "model": model_name, "feature_count": feature_count,
        "long_smiles_dropped": long_dropped, "max_text_tokens": 200,
        "class_imbalance": "balanced for classification", "test_used_for_selection": False,
    }
    payload = {
        **config, "task_type": dataset.spec.task_type,
        "split_sizes": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "parameter_count": parameters, "parameter_count_definition": parameter_definition,
        "validation": validation_metrics, "test": test_metrics,
    }
    joblib.dump({"models": models, "transformer": transformer, "config": config}, run_dir / "best_checkpoint.joblib")
    _write_predictions(run_dir / "predictions.csv", dataset, split.test, test_targets, test_predictions)
    _atomic_text(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=False))
    _atomic_text(metrics_path, json.dumps(payload, indent=2) + "\n")
    ended = datetime.now(timezone.utc)
    environment = {
        "status": "success", "started_at": started.isoformat(), "ended_at": ended.isoformat(),
        "duration_seconds": round(monotonic() - clock, 6), "python": sys.version.replace("\n", " "),
        "platform": platform.platform(), "scikit_learn": sklearn.__version__, "device": "cpu",
    }
    _atomic_text(run_dir / "environment.json", json.dumps(environment, indent=2) + "\n")
    print(f"Completed {dataset.spec.name} {split_name} {model_name} seed={seed}: {test_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen classical baseline matrix")
    parser.add_argument("--datasets", nargs="+", choices=("bace", "tox21", "lipophilicity"),
                        default=("bace", "tox21", "lipophilicity"))
    parser.add_argument("--splits", nargs="+", choices=("random", "scaffold"), default=("random", "scaffold"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 52, 62))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs/reproduction")
    args = parser.parse_args()
    for dataset_name in args.datasets:
        dataset, long_dropped = _eligible(load_downstream_dataset(dataset_name))
        morgan, graph = _features(dataset)
        for split_name in args.splits:
            for model_name in args.models:
                for seed in args.seeds:
                    run_one(dataset, split_name, seed, model_name, morgan, graph, args.output_root,
                            long_dropped)


if __name__ == "__main__":
    main()

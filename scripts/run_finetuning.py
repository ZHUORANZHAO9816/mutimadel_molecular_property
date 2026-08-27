#!/usr/bin/env python3
"""Canonical config-driven GTpro downstream fine-tuning runner."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.config import DEFAULT_FINETUNE_CONFIG, ConfigError, load_finetune_config
from gtpro.data.downstream import DownstreamDataset, load_downstream_dataset, split_dataset
from gtpro.data.pretraining import tokenize_smiles
from gtpro.graph_trans.model.models import GROVEREmbedding
from gtpro.run_metadata import RunRecorder
from gtpro.training.finetuning import (
    MolecularPropertyPredictor,
    MolecularSubset,
    atomic_json,
    atomic_torch_save,
    class_positive_weights,
    collate_finetuning_samples,
    configure_encoder_mode,
    load_pretrained_encoders,
    metric_value,
    result_metrics,
    run_finetuning_epoch,
    write_predictions,
)
from gtpro.utils import get_device
from pretrain.seq_trans import K_BERT_WCL, set_random_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GTpro downstream molecular-property fine-tuning")
    parser.add_argument("--config", type=Path, default=DEFAULT_FINETUNE_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--encoder-mode", choices=("frozen", "partial", "full"))
    parser.add_argument("--pretrained-checkpoint", type=str)
    return parser


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return get_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device=cuda was requested, but CUDA is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ConfigError("device=mps was requested, but MPS is unavailable")
    return torch.device(requested)


def _grover_args(config: dict[str, object], device: torch.device) -> argparse.Namespace:
    return argparse.Namespace(
        hidden_size=config["hidden_size"], backbone=config["backbone"],
        embedding_output_type=config["embedding_output_type"], dropout=config["dropout"],
        activation=config["activation"], num_mt_block=config["num_mt_block"],
        num_attn_head=config["num_attn_head"], bias=config["bias"], cuda=device.type == "cuda",
        depth=config["depth"], dense=config["dense"], undirected=config["undirected"],
        bond_drop_rate=config["bond_drop_rate"], features_only=config["features_only"],
        no_cache=config["no_cache"],
    )


def _subset_dataset(dataset: DownstreamDataset, maximum: int | None, seed: int) -> DownstreamDataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    selected = np.sort(np.random.default_rng(seed).choice(len(dataset), maximum, replace=False))
    return DownstreamDataset(
        spec=dataset.spec, source_path=dataset.source_path,
        source_rows=dataset.source_rows[selected],
        identifiers=tuple(dataset.identifiers[index] for index in selected),
        smiles=tuple(dataset.smiles[index] for index in selected),
        canonical_smiles=tuple(dataset.canonical_smiles[index] for index in selected),
        targets=dataset.targets[selected], invalid_molecules=dataset.invalid_molecules,
    )


def _filter_text_length(dataset: DownstreamDataset, max_tokens: int) -> tuple[DownstreamDataset, int]:
    selected = np.asarray(
        [index for index, smiles in enumerate(dataset.canonical_smiles)
         if len(tokenize_smiles(smiles)) <= max_tokens],
        dtype=np.int64,
    )
    dropped = len(dataset) - len(selected)
    if dropped == 0:
        return dataset, 0
    return DownstreamDataset(
        spec=dataset.spec, source_path=dataset.source_path,
        source_rows=dataset.source_rows[selected],
        identifiers=tuple(dataset.identifiers[index] for index in selected),
        smiles=tuple(dataset.smiles[index] for index in selected),
        canonical_smiles=tuple(dataset.canonical_smiles[index] for index in selected),
        targets=dataset.targets[selected], invalid_molecules=dataset.invalid_molecules,
    ), dropped


def _loader(dataset, indices, max_tokens, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MolecularSubset(dataset, indices, max_tokens), batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, collate_fn=collate_finetuning_samples,
        drop_last=False,
    )


def _build_model(config, dataset, device):
    text_config = config["model"]["text"]
    graph_config = config["model"]["grover"]
    graph_args = _grover_args(graph_config, device)
    text = K_BERT_WCL(
        d_model=text_config["d_model"], n_layers=text_config["n_layers"],
        vocab_size=text_config["vocab_size"], maxlen=text_config["max_length"],
        d_k=text_config["d_k"], d_v=text_config["d_v"], n_heads=text_config["n_heads"],
        d_ff=text_config["d_ff"], global_label_dim=text_config["global_label_dim"],
        atom_label_dim=text_config["atom_label_dim"],
    )
    graph = GROVEREmbedding(graph_args)
    report = None
    checkpoint = config["model"]["pretrained_checkpoint"]
    if checkpoint is not None:
        report = load_pretrained_encoders(
            checkpoint, text_encoder=text, graph_encoder=graph,
            strict=config["model"]["strict_checkpoint"],
            max_mismatch_fraction=config["model"]["max_checkpoint_mismatch_fraction"],
            map_location=device,
        )
    model = MolecularPropertyPredictor(
        text, graph, graph_args, text_dim=text_config["d_model"],
        graph_dim=graph_config["hidden_size"], output_dim=len(dataset.spec.target_columns),
        hidden_dim=config["model"]["head_hidden_dim"], dropout=config["model"]["head_dropout"],
        representation=config["model"]["representation"],
    ).to(device)
    counts = configure_encoder_mode(
        model, config["model"]["encoder_mode"], config["model"]["partial_text_layers"]
    )
    return model, report, counts


def _strip_arrays(result):
    return {key: value for key, value in result.items() if key not in {"predictions", "targets", "indices"}}


def run_seed(base_config: dict[str, object], raw_dataset: DownstreamDataset, seed: int, device):
    config = deepcopy(base_config)
    config["seed"] = seed
    config["device"] = str(device)
    max_tokens = config["model"]["text"]["max_length"] - 1
    if config["model"]["representation"] in {"text", "joint"}:
        eligible_dataset, long_smiles_dropped = _filter_text_length(raw_dataset, max_tokens)
    else:
        eligible_dataset, long_smiles_dropped = raw_dataset, 0
    config["data"]["long_smiles_dropped"] = long_smiles_dropped
    dataset = _subset_dataset(eligible_dataset, config["data"]["max_samples"], seed)
    split = split_dataset(
        dataset, method=config["data"]["split"], fractions=config["data"]["fractions"], seed=seed
    )
    if any(len(indices) == 0 for indices in (split.train, split.validation, split.test)):
        raise ValueError(
            f"configured split produced an empty partition: train={len(split.train)}, "
            f"validation={len(split.validation)}, test={len(split.test)}"
        )

    with RunRecorder(config) as run:
        print(f"Run directory: {run.run_dir}")
        set_random_seed(seed)
        loaders = {
            "train": _loader(dataset, split.train, max_tokens, config["training"]["batch_size"], True, seed),
            "validation": _loader(dataset, split.validation, max_tokens, config["training"]["batch_size"], False, seed),
            "test": _loader(dataset, split.test, max_tokens, config["training"]["batch_size"], False, seed),
        }
        model, checkpoint_report, parameter_counts = _build_model(config, dataset, device)
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
        )
        is_classification = dataset.spec.task_type != "regression"
        weight_values, weight_warnings = (
            class_positive_weights(dataset.targets[split.train], config["training"]["class_imbalance"])
            if is_classification else (np.ones(len(dataset.spec.target_columns), dtype=np.float32), [])
        )
        positive_weights = torch.as_tensor(weight_values, device=device) if is_classification else None
        history = []
        best_score = None
        best_epoch = -1
        bad_epochs = 0
        selection_name = "roc_auc" if dataset.spec.task_type == "binary_classification" else (
            "macro_roc_auc" if dataset.spec.task_type == "multilabel_classification" else "rmse"
        )

        for epoch in range(config["training"]["epochs"]):
            train_result = run_finetuning_epoch(
                model=model, data_loader=loaders["train"], device=device,
                task_type=dataset.spec.task_type, optimizer=optimizer, training=True,
                positive_weights=positive_weights,
                gradient_clip_norm=config["training"]["gradient_clip_norm"],
            )
            validation_result = run_finetuning_epoch(
                model=model, data_loader=loaders["validation"], device=device,
                task_type=dataset.spec.task_type, optimizer=None, training=False,
                positive_weights=positive_weights, gradient_clip_norm=None,
            )
            validation_metrics = result_metrics(dataset, validation_result)
            value, maximize = metric_value(dataset.spec.task_type, validation_metrics)
            used_fallback = value is None
            candidate = float(validation_result["loss"] if used_fallback else value)
            candidate_maximize = False if used_fallback else maximize
            threshold = config["training"]["early_stopping_min_delta"]
            improved = best_score is None or (
                candidate > best_score + threshold if candidate_maximize
                else candidate < best_score - threshold
            )
            history.append({
                "epoch": epoch, "train": _strip_arrays(train_result),
                "validation": _strip_arrays(validation_result),
                "validation_metrics": validation_metrics, "selection_value": candidate,
                "selection_fallback_to_loss": used_fallback,
            })
            if improved:
                best_score, best_epoch, bad_epochs = candidate, epoch, 0
                checkpoint_payload = {
                    "epoch": epoch, "seed": seed, "config": config, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "selection_value": candidate,
                    "selection_metric": "validation_loss" if used_fallback else selection_name,
                }
                atomic_torch_save(run.checkpoint_dir / "best.pt", checkpoint_payload)
                if config["output"].get("formal_layout", False):
                    atomic_torch_save(run.run_dir / "best_checkpoint.pt", checkpoint_payload)
            else:
                bad_epochs += 1
            print(
                f"seed={seed} epoch={epoch + 1}: train_loss={train_result['loss']:.6f} "
                f"validation_loss={validation_result['loss']:.6f} {selection_name}={value}"
            )
            if bad_epochs >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping after epoch {epoch + 1}")
                break

        try:
            best_payload = torch.load(run.checkpoint_dir / "best.pt", map_location=device, weights_only=True)
        except TypeError:
            best_payload = torch.load(run.checkpoint_dir / "best.pt", map_location=device)
        model.load_state_dict(best_payload["model"], strict=True)
        final_results = {
            name: run_finetuning_epoch(
                model=model, data_loader=loaders[name], device=device,
                task_type=dataset.spec.task_type, optimizer=None, training=False,
                positive_weights=positive_weights, gradient_clip_norm=None,
            ) for name in ("validation", "test")
        }
        final_metrics = {name: result_metrics(dataset, result) for name, result in final_results.items()}
        write_predictions(run.run_dir / "predictions.csv", dataset=dataset, results=final_results)
        payload = {
            "dataset": dataset.spec.name, "task_type": dataset.spec.task_type, "seed": seed,
            "split": {"method": split.method, "fractions": split.fractions,
                      "sizes": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)}},
            "data_filtering": {"long_smiles_dropped": long_smiles_dropped, "max_text_tokens": max_tokens},
            "encoder_mode": config["model"]["encoder_mode"], "parameter_counts": parameter_counts,
            "representation": config["model"]["representation"],
            "checkpoint_load": checkpoint_report or {"source": None, "status": "random_initialization"},
            "class_imbalance": {"configured": config["training"]["class_imbalance"],
                                "effective_positive_weights": weight_values.tolist(), "warnings": weight_warnings},
            "best_epoch": best_epoch, "best_selection_value": best_score,
            "selection_metric": best_payload["selection_metric"], "history": history,
            "final": {name: {"epoch": _strip_arrays(final_results[name]), "metrics": final_metrics[name]}
                      for name in final_results},
        }
        atomic_json(run.run_dir / "metrics.json", payload)
        print(f"Completed {dataset.spec.name} seed {seed}; best epoch={best_epoch}; test={final_metrics['test']}")
        return run.run_dir


def main() -> None:
    args = build_parser().parse_args()
    config = load_finetune_config(
        args.config,
        {"seeds": args.seeds, "device": args.device, "training.epochs": args.epochs,
         "data.max_samples": args.max_samples, "model.encoder_mode": args.encoder_mode,
         "model.pretrained_checkpoint": args.pretrained_checkpoint},
    )
    device = _select_device(config["device"])
    raw_dataset = load_downstream_dataset(
        config["data"]["dataset"], config["data"]["root"],
        invalid_smiles=config["data"]["invalid_smiles"],
    )
    print(
        f"Loaded {len(raw_dataset)} valid {raw_dataset.spec.name} molecules "
        f"({len(raw_dataset.invalid_molecules)} invalid dropped); device={device}"
    )
    run_directories = [run_seed(config, raw_dataset, seed, device) for seed in config["seeds"]]
    print("Fine-tuning complete: " + ", ".join(str(path) for path in run_directories))


if __name__ == "__main__":
    main()

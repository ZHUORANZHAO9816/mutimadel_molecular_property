"""Reusable downstream fine-tuning data, model, epoch, and checkpoint logic."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from gtpro.data.downstream import DownstreamDataset
from gtpro.data.pretraining import process_smiles
from gtpro.graph_trans.data import mol2graph
from gtpro.metrics import compute_metrics
from gtpro.models.interfaces import encode_graph


@dataclass(frozen=True)
class FinetuningBatch:
    token_ids: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    smiles: tuple[str, ...]
    indices: torch.Tensor

    def to(self, device: torch.device | str) -> "FinetuningBatch":
        return FinetuningBatch(
            token_ids=self.token_ids.to(device=device, dtype=torch.long),
            targets=self.targets.to(device=device, dtype=torch.float32),
            target_mask=self.target_mask.to(device=device, dtype=torch.bool),
            smiles=self.smiles,
            indices=self.indices,
        )


class MolecularSubset(Dataset):
    """Pre-tokenized view over one downstream split."""

    def __init__(self, dataset: DownstreamDataset, indices: Sequence[int], max_tokens: int):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.token_ids = [
            process_smiles(dataset.canonical_smiles[int(index)], max_tokens).token_ids
            for index in self.indices
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return (
            self.token_ids[item],
            self.dataset.targets[index],
            self.dataset.canonical_smiles[index],
            index,
        )


def collate_finetuning_samples(samples) -> FinetuningBatch:
    if not samples:
        raise ValueError("cannot collate an empty fine-tuning batch")
    tokens = np.stack([np.asarray(sample[0], dtype=np.int64) for sample in samples])
    targets = np.stack([np.asarray(sample[1], dtype=np.float32) for sample in samples])
    return FinetuningBatch(
        token_ids=torch.from_numpy(tokens),
        targets=torch.from_numpy(np.nan_to_num(targets, nan=0.0)),
        target_mask=torch.from_numpy(np.isfinite(targets)),
        smiles=tuple(str(sample[2]) for sample in samples),
        indices=torch.as_tensor([sample[3] for sample in samples], dtype=torch.long),
    )


class MolecularPropertyPredictor(nn.Module):
    """Joint graph/text encoder with a task-specific prediction head."""

    def __init__(
        self,
        text_encoder: nn.Module,
        graph_encoder: nn.Module,
        graph_args: object,
        *,
        text_dim: int,
        graph_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        representation: str = "joint",
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.graph_args = graph_args
        if representation not in {"graph", "text", "joint"}:
            raise ValueError("representation must be graph, text, or joint")
        self.representation = representation
        input_dim = {"graph": graph_dim, "text": text_dim, "joint": text_dim + graph_dim}[
            representation
        ]
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, token_ids: torch.Tensor, smiles: Sequence[str]) -> torch.Tensor:
        graph_embedding = text_embedding = None
        if self.representation in {"graph", "joint"}:
            components = mol2graph(list(smiles), shared_dict={}, args=self.graph_args).get_components()
            graph_embedding = encode_graph(
                self.graph_encoder, components, token_ids.device
            ).molecule_embeddings
        if self.representation in {"text", "joint"}:
            text_embedding = self.text_encoder(token_ids).global_embedding
        if self.representation == "graph":
            features = graph_embedding
        elif self.representation == "text":
            features = text_embedding
        else:
            features = torch.cat((text_embedding, graph_embedding), dim=-1)
        return self.head(features)


def configure_encoder_mode(
    model: MolecularPropertyPredictor, mode: str, partial_text_layers: int
) -> dict[str, int]:
    for encoder in (model.text_encoder, model.graph_encoder):
        encoder.requires_grad_(False)
    if mode == "full":
        if model.representation in {"text", "joint"}:
            model.text_encoder.requires_grad_(True)
        if model.representation in {"graph", "joint"}:
            model.graph_encoder.requires_grad_(True)
    elif mode == "partial":
        if model.representation == "graph":
            raise ValueError("partial mode only unfreezes text layers and is invalid for graph-only")
        layers = getattr(model.text_encoder, "layers", None)
        if layers is None or len(layers) < partial_text_layers:
            raise ValueError(
                f"partial mode requested {partial_text_layers} text layers, but encoder has "
                f"{0 if layers is None else len(layers)}"
            )
        for layer in layers[-partial_text_layers:]:
            layer.requires_grad_(True)
        for name in ("fc", "classifier_global", "classifier_atom"):
            module = getattr(model.text_encoder, name, None)
            if module is not None:
                module.requires_grad_(True)
    elif mode != "frozen":
        raise ValueError("encoder mode must be frozen, partial, or full")
    model.head.requires_grad_(True)
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }


def _load_state_report(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
    strict: bool,
    max_mismatch_fraction: float,
) -> dict[str, object]:
    target = module.state_dict()
    unexpected = sorted(set(state).difference(target))
    missing = sorted(set(target).difference(state))
    shape_mismatches = sorted(
        key for key in set(target).intersection(state) if target[key].shape != state[key].shape
    )
    mismatched = set(unexpected) | set(missing) | set(shape_mismatches)
    fraction = len(mismatched) / max(1, len(target))
    report = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "mismatch_fraction": fraction,
    }
    if fraction > max_mismatch_fraction:
        raise ValueError(
            f"{name} checkpoint mismatch fraction {fraction:.3f} exceeds configured maximum "
            f"{max_mismatch_fraction:.3f}; report={report}"
        )
    if strict and mismatched:
        raise ValueError(f"strict {name} checkpoint loading failed; report={report}")
    compatible = {
        key: value for key, value in state.items() if key in target and key not in shape_mismatches
    }
    module.load_state_dict(compatible, strict=strict)
    return report


def load_pretrained_encoders(
    path: str | Path,
    *,
    text_encoder: nn.Module,
    graph_encoder: nn.Module,
    strict: bool,
    max_mismatch_fraction: float,
    map_location: torch.device | str,
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pretrained checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError("pretrained checkpoint must contain a 'models' mapping")
    states = payload["models"]
    if "text" not in states or "graph" not in states:
        raise ValueError("pretrained checkpoint must contain text and graph model states")
    return {
        "source": str(source.resolve()),
        "strict": strict,
        "text": _load_state_report(
            text_encoder,
            states["text"],
            name="text",
            strict=strict,
            max_mismatch_fraction=max_mismatch_fraction,
        ),
        "graph": _load_state_report(
            graph_encoder,
            states["graph"],
            name="graph",
            strict=strict,
            max_mismatch_fraction=max_mismatch_fraction,
        ),
    }


def class_positive_weights(targets: np.ndarray, mode: object) -> tuple[np.ndarray, list[str]]:
    width = targets.shape[1]
    if mode == "none":
        return np.ones(width, dtype=np.float32), []
    if isinstance(mode, list):
        if len(mode) != width:
            raise ValueError(f"configured class weights have length {len(mode)}, expected {width}")
        return np.asarray(mode, dtype=np.float32), []
    if mode != "auto":
        raise ValueError("class imbalance mode must be none, auto, or a list")
    weights = np.ones(width, dtype=np.float32)
    warnings: list[str] = []
    for index in range(width):
        values = targets[:, index]
        values = values[np.isfinite(values)]
        positives = int((values == 1).sum())
        negatives = int((values == 0).sum())
        if positives == 0 or negatives == 0:
            warnings.append(
                f"target {index} lacks both classes in training; positive weight defaults to 1"
            )
        else:
            weights[index] = negatives / positives
    return weights, warnings


def masked_task_loss(
    logits: torch.Tensor,
    batch: FinetuningBatch,
    task_type: str,
    positive_weights: torch.Tensor | None,
) -> tuple[torch.Tensor | None, int]:
    valid = batch.target_mask
    count = int(valid.sum().item())
    if count == 0:
        return None, 0
    if task_type in {"binary_classification", "multilabel_classification"}:
        elementwise = nn.functional.binary_cross_entropy_with_logits(
            logits, batch.targets, reduction="none", pos_weight=positive_weights
        )
    else:
        elementwise = nn.functional.mse_loss(logits, batch.targets, reduction="none")
    return elementwise[valid].mean(), count


def run_finetuning_epoch(
    *,
    model: MolecularPropertyPredictor,
    data_loader,
    device: torch.device,
    task_type: str,
    optimizer: torch.optim.Optimizer | None,
    training: bool,
    positive_weights: torch.Tensor | None,
    gradient_clip_norm: float | None,
) -> dict[str, object]:
    if training and optimizer is None:
        raise ValueError("training epoch requires an optimizer")
    model.train(training)
    # Frozen encoders must also keep deterministic evaluation-time dropout and
    # normalization behavior while the prediction head trains.
    if not any(parameter.requires_grad for parameter in model.text_encoder.parameters()):
        model.text_encoder.eval()
    if not any(parameter.requires_grad for parameter in model.graph_encoder.parameters()):
        model.graph_encoder.eval()
    total_weighted_loss = 0.0
    valid_labels = batches = skipped_batches = 0
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in data_loader:
            if not isinstance(raw_batch, FinetuningBatch):
                raise TypeError("fine-tuning loader must yield FinetuningBatch")
            batch = raw_batch.to(device)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch.token_ids, batch.smiles)
            loss, count = masked_task_loss(logits, batch, task_type, positive_weights)
            if loss is None:
                skipped_batches += 1
            elif training:
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        gradient_clip_norm,
                    )
                optimizer.step()
            if loss is not None:
                total_weighted_loss += float(loss.detach().cpu()) * count
                valid_labels += count
            outputs.append(logits.detach().cpu().numpy())
            targets.append(batch.targets.detach().cpu().numpy())
            masks.append(batch.target_mask.detach().cpu().numpy())
            indices.append(batch.indices.numpy())
            batches += 1
    if batches == 0:
        raise ValueError("fine-tuning data loader produced no batches")
    if valid_labels == 0:
        raise ValueError("fine-tuning split contains no valid labels")
    logits_array = np.concatenate(outputs)
    prediction_array = (
        1.0 / (1.0 + np.exp(-logits_array))
        if task_type in {"binary_classification", "multilabel_classification"}
        else logits_array
    )
    target_array = np.concatenate(targets)
    mask_array = np.concatenate(masks)
    target_array[~mask_array] = np.nan
    return {
        "loss": total_weighted_loss / valid_labels,
        "valid_labels": valid_labels,
        "batches": batches,
        "skipped_batches": skipped_batches,
        "predictions": prediction_array,
        "targets": target_array,
        "indices": np.concatenate(indices),
    }


def metric_value(task_type: str, metrics: Mapping[str, object]) -> tuple[float | None, bool]:
    if task_type == "binary_classification":
        return metrics.get("roc_auc"), True
    if task_type == "multilabel_classification":
        return metrics.get("macro_roc_auc"), True
    return metrics.get("rmse"), False


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_predictions(
    path: Path,
    *,
    dataset: DownstreamDataset,
    results: Mapping[str, Mapping[str, object]],
) -> None:
    target_names = dataset.spec.target_columns
    fields = ["split", "dataset_index", "source_row", "identifier", "smiles"]
    fields += [f"target_{name}" for name in target_names]
    fields += [f"prediction_{name}" for name in target_names]
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for split_name, result in results.items():
                for position, raw_index in enumerate(result["indices"]):
                    index = int(raw_index)
                    row = {
                        "split": split_name,
                        "dataset_index": index,
                        "source_row": int(dataset.source_rows[index]),
                        "identifier": dataset.identifiers[index] or "",
                        "smiles": dataset.canonical_smiles[index],
                    }
                    for target_index, target_name in enumerate(target_names):
                        target = float(result["targets"][position, target_index])
                        row[f"target_{target_name}"] = "" if not np.isfinite(target) else target
                        row[f"prediction_{target_name}"] = float(
                            result["predictions"][position, target_index]
                        )
                    writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def result_metrics(dataset: DownstreamDataset, result: Mapping[str, object]) -> dict[str, object]:
    return compute_metrics(
        dataset.spec.task_type,
        result["targets"],
        result["predictions"],
        dataset.spec.target_columns,
    )

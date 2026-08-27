"""Standardized GTpro pretraining epoch, metrics, and checkpoint state."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from gtpro.graph_trans.data import mol2graph
from gtpro.models.interfaces import PretrainingBatch, encode_graph
from pretrain.seq_trans import rdkit_functional_group_label_features_generator


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PretrainingModels:
    text: nn.Module
    graph: nn.Module
    alignment: nn.Module

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for model in (self.text, self.graph, self.alignment)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

    def state_dict(self) -> dict[str, Mapping[str, torch.Tensor]]:
        return {
            "text": self.text.state_dict(),
            "graph": self.graph.state_dict(),
            "alignment": self.alignment.state_dict(),
        }


@dataclass(frozen=True)
class EpochMetrics:
    split: str
    samples: int
    batches: int
    total_loss: float
    contrastive_loss: float
    atom_loss: float
    functional_group_loss: float
    molecule_loss: float
    gradient_norm: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collate_pretraining_samples(samples: Sequence[Sequence[object]]) -> PretrainingBatch:
    """Stack NumPy-backed samples efficiently and normalize legacy mask length."""

    if not samples:
        raise ValueError("cannot collate an empty pretraining batch")
    try:
        token_ids = np.stack([np.asarray(sample[0], dtype=np.int64) for sample in samples])
        global_labels = np.stack([np.asarray(sample[1], dtype=np.float32) for sample in samples])
        atom_labels = np.stack([np.asarray(sample[2], dtype=np.float32) for sample in samples])
        atom_mask = np.stack([np.asarray(sample[3], dtype=np.int8) for sample in samples])
        smiles = tuple(str(sample[4]) for sample in samples)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"malformed pretraining sample batch: {error}") from error

    if atom_labels.ndim != 3 or atom_mask.ndim != 2 or token_ids.ndim != 2:
        raise ValueError("pretraining tokens, atom labels, and atom masks have invalid ranks")
    if atom_labels.shape[1] == token_ids.shape[1]:
        atom_labels = atom_labels[:, 1:]
        atom_mask = atom_mask[:, 1:]
    if atom_labels.shape[1] != token_ids.shape[1] - 1:
        raise ValueError("atom labels must align to token positions excluding the global token")

    return PretrainingBatch(
        token_ids=torch.from_numpy(token_ids),
        global_labels=torch.from_numpy(global_labels),
        atom_labels=torch.from_numpy(atom_labels),
        atom_mask=torch.from_numpy(atom_mask),
        smiles=smiles,
    )


def split_pretraining_samples(
    samples: Sequence[Sequence[object]], validation_fraction: float, seed: int
) -> tuple[list[Sequence[object]], list[Sequence[object]]]:
    if len(samples) < 2:
        raise ValueError("pretraining train/validation split requires at least two samples")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    validation_size = min(len(samples) - 1, max(1, round(len(samples) * validation_fraction)))
    validation_indices = set(indices[:validation_size].tolist())
    train = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    validation = [sample for index, sample in enumerate(samples) if index in validation_indices]
    return train, validation


def _set_mode(models: PretrainingModels, training: bool) -> None:
    models.text.train(training)
    models.alignment.train(training)
    graph_is_trainable = any(parameter.requires_grad for parameter in models.graph.parameters())
    models.graph.train(training and graph_is_trainable)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_pretraining_epoch(
    *,
    models: PretrainingModels,
    data_loader: Iterable[PretrainingBatch],
    grover_args: object,
    device: torch.device,
    atom_loss: nn.Module,
    molecule_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    training: bool,
    mixed_precision: bool = False,
    gradient_clip_norm: float | None = None,
    molecule_loss_weight: float = 1.0,
    progress: Any = None,
) -> EpochMetrics:
    """Run one train or validation epoch and return per-sample mean losses."""

    if training and optimizer is None:
        raise ValueError("training epoch requires an optimizer")
    if mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is supported only on CUDA")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when configured")

    _set_mode(models, training)
    scaler = _grad_scaler(training and mixed_precision)
    sums = {name: 0.0 for name in ("total", "contrastive", "atom", "functional_group", "molecule")}
    samples_seen = batches_seen = 0
    last_gradient_norm: float | None = None
    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch_index, raw_batch in enumerate(data_loader):
            if not isinstance(raw_batch, PretrainingBatch):
                raise TypeError("pretraining data loader must yield PretrainingBatch")
            batch = raw_batch.to(device)
            batch_size = batch.token_ids.shape[0]
            if batch_size == 0:
                raise ValueError(f"empty batch encountered at index {batch_index}")
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            try:
                graph_components = mol2graph(
                    list(batch.smiles), shared_dict={}, args=grover_args
                ).get_components()
                graph_output = encode_graph(models.graph, graph_components, device)
            except Exception as error:
                raise ValueError(
                    f"failed to build/encode graph batch {batch_index} for SMILES {batch.smiles}: {error}"
                ) from error

            fg_array = np.asarray(
                rdkit_functional_group_label_features_generator(batch.smiles), dtype=np.float32
            )
            functional_groups = torch.from_numpy(fg_array).to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=training and mixed_precision,
            ):
                text_output = models.text(batch.token_ids)
                alignment_losses = models.alignment(
                    graph_output.atom_embeddings,
                    graph_output.molecule_embeddings,
                    text_output.global_embedding,
                    text_output.atom_tokens,
                    batch.atom_labels,
                    functional_groups,
                    text_output.all_tokens,
                    batch.atom_target_mask,
                    atom_loss,
                    return_loss=True,
                )
                molecule_value = molecule_loss(text_output.molecule_logits, batch.global_labels)
                total = alignment_losses.total + molecule_loss_weight * molecule_value

            if training:
                scaler.scale(total).backward()
                if gradient_clip_norm is not None:
                    assert optimizer is not None
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        models.trainable_parameters(), gradient_clip_norm
                    )
                    last_gradient_norm = float(norm.detach().cpu())
                assert optimizer is not None
                scaler.step(optimizer)
                scaler.update()

            values = {
                "total": total,
                "contrastive": alignment_losses.contrastive,
                "atom": alignment_losses.atom,
                "functional_group": alignment_losses.functional_group,
                "molecule": molecule_value,
            }
            for name, value in values.items():
                sums[name] += float(value.detach().cpu()) * batch_size
            samples_seen += batch_size
            batches_seen += 1
            if progress is not None:
                progress(
                    f"{('train' if training else 'validation')} batch {batch_index + 1}: "
                    f"total={float(total.detach().cpu()):.6f}"
                )

    if batches_seen == 0 or samples_seen == 0:
        raise ValueError("pretraining data loader produced no batches")
    return EpochMetrics(
        split="train" if training else "validation",
        samples=samples_seen,
        batches=batches_seen,
        total_loss=sums["total"] / samples_seen,
        contrastive_loss=sums["contrastive"] / samples_seen,
        atom_loss=sums["atom"] / samples_seen,
        functional_group_loss=sums["functional_group"] / samples_seen,
        molecule_loss=sums["molecule"] / samples_seen,
        gradient_norm=last_gradient_norm,
    )


def save_pretraining_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    seed: int,
    config: Mapping[str, object],
    models: PretrainingModels,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_validation_loss: float,
    history: Sequence[Mapping[str, object]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "seed": seed,
        "config": dict(config),
        "models": models.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_validation_loss": best_validation_loss,
        "history": list(history),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pretraining_checkpoint(
    path: str | Path,
    *,
    models: PretrainingModels,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    map_location: torch.device | str,
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pretraining checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch before weights_only support
        payload = torch.load(source, map_location=map_location)
    required = {
        "schema_version", "epoch", "seed", "config", "models", "optimizer",
        "scheduler", "best_validation_loss", "history",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(required.difference(payload if isinstance(payload, dict) else {}))
        raise ValueError(f"invalid pretraining checkpoint; missing keys: {missing}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported pretraining checkpoint schema: {payload['schema_version']}")
    model_states = payload["models"]
    if not isinstance(model_states, dict) or set(model_states) != {"text", "graph", "alignment"}:
        raise ValueError("pretraining checkpoint has invalid model state groups")
    models.text.load_state_dict(model_states["text"], strict=True)
    models.graph.load_state_dict(model_states["graph"], strict=True)
    models.alignment.load_state_dict(model_states["alignment"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload

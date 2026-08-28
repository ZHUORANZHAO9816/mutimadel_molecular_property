"""Validated YAML configuration loading for molecular-model runners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from gtpro.data.downstream import available_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN_CONFIG = PROJECT_ROOT / "configs" / "pretrain.yaml"
DEFAULT_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "finetune_bace_smoke.yaml"


class ConfigError(ValueError):
    """Raised when a GTpro configuration is missing or invalid."""


def _get(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def _set(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot override non-mapping configuration key: {part}")
        target = child
    target[parts[-1]] = value


def _resolve_project_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_pretrain_config(
    path: str | Path = DEFAULT_PRETRAIN_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, override, validate, and resolve a pretraining YAML config."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ConfigError(f"Pretraining config does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Pretraining config must contain a mapping: {config_path}")
    config = deepcopy(raw)

    for key, value in (overrides or {}).items():
        if value is not None:
            _set(config, key, value)

    required_positive_ints = (
        "model.text.d_model",
        "model.text.n_layers",
        "model.text.vocab_size",
        "model.text.max_length",
        "model.text.d_k",
        "model.text.d_v",
        "model.text.n_heads",
        "model.text.d_ff",
        "model.text.global_label_dim",
        "model.text.atom_label_dim",
        "model.grover.hidden_size",
        "model.grover.num_mt_block",
        "model.grover.num_attn_head",
        "model.grover.depth",
        "model.alignment.unimodal_depth",
        "model.alignment.multimodal_depth",
        "model.alignment.dim_head",
        "model.alignment.heads",
        "model.alignment.functional_group_dim",
        "training.batch_size",
        "training.epochs",
    )
    for key in required_positive_ints:
        value = _get(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer; got {value!r}")

    seed = _get(config, "seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError(f"seed must be a non-negative integer; got {seed!r}")

    learning_rate = _get(config, "training.learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ConfigError(f"training.learning_rate must be positive; got {learning_rate!r}")

    validation_fraction = _get(config, "data.validation_fraction")
    if not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 1:
        raise ConfigError(
            f"data.validation_fraction must be between 0 and 1; got {validation_fraction!r}"
        )

    max_samples = _get(config, "data.max_samples")
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 2
    ):
        raise ConfigError("data.max_samples must be null or an integer of at least 2")

    gradient_clip_norm = _get(config, "training.gradient_clip_norm")
    if gradient_clip_norm is not None and (
        not isinstance(gradient_clip_norm, (int, float)) or gradient_clip_norm <= 0
    ):
        raise ConfigError(
            f"training.gradient_clip_norm must be positive or null; got {gradient_clip_norm!r}"
        )

    mixed_precision = _get(config, "training.mixed_precision")
    if mixed_precision not in {True, False, "auto"}:
        raise ConfigError(
            f"training.mixed_precision must be true, false, or auto; got {mixed_precision!r}"
        )

    molecule_loss_weight = _get(config, "training.molecule_loss_weight")
    if not isinstance(molecule_loss_weight, (int, float)) or molecule_loss_weight < 0:
        raise ConfigError(
            f"training.molecule_loss_weight must be non-negative; got {molecule_loss_weight!r}"
        )

    for key in (
        "model.alignment.caption_loss_weight",
        "model.alignment.contrastive_loss_weight",
        "model.alignment.functional_group_loss_weight",
    ):
        value = _get(config, key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"{key} must be non-negative; got {value!r}")
    if not isinstance(_get(config, "model.alignment.use_cross_attention"), bool):
        raise ConfigError("model.alignment.use_cross_attention must be true or false")

    device = _get(config, "device")
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError(f"device must be auto, cpu, cuda, or mps; got {device!r}")

    _get(config, "data.path")
    _get(config, "output.root")
    _get(config, "model.grover.freeze")

    config["data"]["path"] = _resolve_project_path(config["data"]["path"])
    config["output"]["root"] = _resolve_project_path(config["output"]["root"])
    config["model"]["grover"]["checkpoint"] = _resolve_project_path(
        config["model"]["grover"].get("checkpoint")
    )
    config["training"]["resume_from"] = _resolve_project_path(
        config["training"].get("resume_from")
    )
    config["config_source"] = str(config_path)
    return config


def load_finetune_config(
    path: str | Path = DEFAULT_FINETUNE_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a downstream fine-tuning configuration."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ConfigError(f"Fine-tuning config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Fine-tuning config must contain a mapping: {config_path}")
    config = deepcopy(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            _set(config, key, value)

    seeds = _get(config, "seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
    ):
        raise ConfigError("seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("seeds must not contain duplicates")

    device = _get(config, "device")
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError(f"device must be auto, cpu, cuda, or mps; got {device!r}")
    dataset = _get(config, "data.dataset")
    supported_datasets = set(available_datasets())
    if dataset not in supported_datasets:
        choices = ", ".join(sorted(supported_datasets))
        raise ConfigError(f"data.dataset must be one of: {choices}")
    if _get(config, "data.split") not in {"random", "scaffold"}:
        raise ConfigError("data.split must be random or scaffold")
    if _get(config, "data.invalid_smiles") not in {"drop", "raise"}:
        raise ConfigError("data.invalid_smiles must be drop or raise")
    fractions = _get(config, "data.fractions")
    if not isinstance(fractions, list) or len(fractions) != 3 or any(
        not isinstance(value, (int, float)) or value < 0 for value in fractions
    ) or not np.isclose(sum(fractions), 1.0):
        raise ConfigError("data.fractions must contain three non-negative values summing to 1")
    max_samples = _get(config, "data.max_samples")
    if max_samples is not None and (
        not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 3
    ):
        raise ConfigError("data.max_samples must be null or an integer of at least 3")

    positive_ints = (
        "model.text.d_model", "model.text.n_layers", "model.text.vocab_size",
        "model.text.max_length", "model.text.d_k", "model.text.d_v",
        "model.text.n_heads", "model.text.d_ff", "model.text.global_label_dim",
        "model.text.atom_label_dim", "model.grover.hidden_size",
        "model.grover.num_mt_block", "model.grover.num_attn_head", "model.grover.depth",
        "model.head_hidden_dim", "model.partial_text_layers", "training.batch_size",
        "training.epochs", "training.early_stopping_patience",
    )
    for key in positive_ints:
        value = _get(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer; got {value!r}")
    if _get(config, "model.encoder_mode") not in {"frozen", "partial", "full"}:
        raise ConfigError("model.encoder_mode must be frozen, partial, or full")
    if _get(config, "model.representation") not in {"graph", "text", "joint"}:
        raise ConfigError("model.representation must be graph, text, or joint")
    mismatch_fraction = _get(config, "model.max_checkpoint_mismatch_fraction")
    if not isinstance(mismatch_fraction, (int, float)) or not 0 <= mismatch_fraction <= 1:
        raise ConfigError("model.max_checkpoint_mismatch_fraction must be between 0 and 1")
    for key in ("training.learning_rate", "training.gradient_clip_norm"):
        value = _get(config, key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{key} must be positive; got {value!r}")
    class_imbalance = _get(config, "training.class_imbalance")
    valid_weights = (
        isinstance(class_imbalance, list)
        and bool(class_imbalance)
        and all(isinstance(value, (int, float)) and value > 0 for value in class_imbalance)
    )
    if not (class_imbalance in {"none", "auto"} if isinstance(class_imbalance, str) else valid_weights):
        raise ConfigError("training.class_imbalance must be none, auto, or positive weights")

    strict_checkpoint = _get(config, "model.strict_checkpoint")
    if not isinstance(strict_checkpoint, bool):
        raise ConfigError("model.strict_checkpoint must be true or false")
    for key in ("model.head_dropout", "training.weight_decay", "training.early_stopping_min_delta"):
        value = _get(config, key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"{key} must be non-negative; got {value!r}")
    if _get(config, "model.head_dropout") >= 1:
        raise ConfigError("model.head_dropout must be less than 1")

    config["data"]["root"] = _resolve_project_path(_get(config, "data.root"))
    config["output"]["root"] = _resolve_project_path(_get(config, "output.root"))
    config["model"]["pretrained_checkpoint"] = _resolve_project_path(
        _get(config, "model.pretrained_checkpoint")
    )
    config["config_source"] = str(config_path)
    return config

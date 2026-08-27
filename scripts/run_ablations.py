#!/usr/bin/env python3
"""Run the frozen one-seed objective-ablation screening protocol."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.config import load_finetune_config
from gtpro.data.downstream import load_downstream_dataset
from scripts.run_finetuning import _select_device, run_seed


VARIANTS = {
    "full_model": {},
    "no_contrastive": {"--contrastive-loss-weight": "0"},
    "no_cross_attention": {"--no-cross-attention": None},
    "no_atom_objective": {"--caption-loss-weight": "0"},
    "no_functional_group_objective": {"--functional-group-loss-weight": "0"},
    "no_molecule_objective": {"--molecule-loss-weight": "0"},
}


def _pretrain(variant: str, options: dict[str, str | None]) -> Path:
    root = PROJECT_ROOT / "runs/ablation/pretrain" / variant
    checkpoint = root / "42/checkpoints/best.pt"
    if checkpoint.is_file():
        print(f"Skipping completed ablation pretraining: {variant}")
        return checkpoint
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts/run_pretraining.py"),
        "--config", str(PROJECT_ROOT / "configs/pretrain_reproduction_compact.yaml"),
        "--max-samples", "512", "--save-dir", str(root), "--run-id", "42",
    ]
    for flag, value in options.items():
        command.append(flag)
        if value is not None:
            command.append(value)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return checkpoint


def main() -> None:
    base = load_finetune_config(PROJECT_ROOT / "configs/finetune_reproduction_compact.yaml")
    base["seeds"] = [42]
    base["data"].update(dataset="bace", split="random")
    base["training"]["epochs"] = 2
    device = _select_device(base["device"])
    dataset = load_downstream_dataset("bace", base["data"]["root"])
    for variant, options in VARIANTS.items():
        checkpoint = _pretrain(variant, options)
        config = deepcopy(base)
        config["model"]["pretrained_checkpoint"] = str(checkpoint.resolve())
        config["model"]["representation"] = "joint"
        config["output"].update(
            root=str((PROJECT_ROOT / "runs/ablation/bace_random" / variant).resolve()),
            run_id="{seed}", formal_layout=True,
        )
        expected = Path(config["output"]["root"]) / "42/metrics.json"
        if not expected.is_file():
            run_seed(config, dataset, 42, device)
    full_checkpoint = PROJECT_ROOT / "runs/pretrain_reproduction_compact/compact_v1/checkpoints/best.pt"
    for representation in ("graph", "text"):
        config = deepcopy(base)
        config["model"]["pretrained_checkpoint"] = str(full_checkpoint.resolve())
        config["model"]["representation"] = representation
        variant = f"{representation}_only"
        config["output"].update(
            root=str((PROJECT_ROOT / "runs/ablation/bace_random" / variant).resolve()),
            run_id="{seed}", formal_layout=True,
        )
        expected = Path(config["output"]["root"]) / "42/metrics.json"
        if not expected.is_file():
            run_seed(config, dataset, 42, device)
    print("Completed eight one-seed ablation screening runs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen compact GTpro downstream reproduction matrix."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.config import load_finetune_config
from gtpro.data.downstream import load_downstream_dataset
from scripts.run_finetuning import _select_device, run_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen compact GTpro reproduction matrix")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/finetune_reproduction_compact.yaml"
    )
    parser.add_argument("--datasets", nargs="+", choices=("bace", "tox21", "lipophilicity"),
                        default=("bace", "tox21", "lipophilicity"))
    parser.add_argument("--splits", nargs="+", choices=("random", "scaffold"),
                        default=("random", "scaffold"))
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = load_finetune_config(args.config, {"seeds": args.seeds})
    device = _select_device(base["device"])
    completed = []
    for dataset_name in args.datasets:
        raw_dataset = load_downstream_dataset(dataset_name, base["data"]["root"], invalid_smiles="drop")
        for split_name in args.splits:
            config = deepcopy(base)
            config["data"]["dataset"] = dataset_name
            config["data"]["split"] = split_name
            config["training"]["class_imbalance"] = (
                "none" if raw_dataset.spec.task_type == "regression" else "auto"
            )
            output_root = PROJECT_ROOT / "runs/reproduction" / f"{dataset_name}_{split_name}" / "full_gtpro"
            config["output"].update(
                root=str(output_root.resolve()), run_id="{seed}", formal_layout=True
            )
            for seed in config["seeds"]:
                expected = output_root / str(seed) / "metrics.json"
                if expected.is_file():
                    print(f"Skipping completed run: {expected.parent}")
                    completed.append(expected.parent)
                    continue
                completed.append(run_seed(config, raw_dataset, seed, device))
    print(f"Completed or verified {len(completed)} compact GTpro runs")


if __name__ == "__main__":
    main()

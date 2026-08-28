#!/usr/bin/env python3
"""Run data preparation, graph-text pretraining, and five benchmark fine-tunes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DATASETS = ("bbbp", "bace", "sider", "clintox", "tox21")
DATA_PATHS = {
    "bbbp": Path("data/downstream/bbbp/raw/BBBP.csv"),
    "bace": Path("data/downstream/bace/raw/bace.csv"),
    "sider": Path("data/downstream/sider/raw/sider.csv"),
    "clintox": Path("data/downstream/clintox/raw/clintox.csv"),
    "tox21": Path("data/downstream/tox21/raw/tox21.csv"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=PAPER_DATASETS, default=list(PAPER_DATASETS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[10])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a 128-molecule CPU-friendly integration check for every selected dataset",
    )
    parser.add_argument(
        "--paper-scale", action="store_true",
        help="Use the large architecture and 50-epoch configurations from the paper settings",
    )
    parser.add_argument("--grover-checkpoint", type=Path)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--output-root", type=Path, default=Path("runs/full_pipeline"))
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Fail instead of downloading if a required tracked dataset file is missing",
    )
    return parser


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_datasets(selected: list[str], *, skip_download: bool) -> None:
    required = [PROJECT_ROOT / DATA_PATHS[name] for name in selected]
    required.append(PROJECT_ROOT / "data/pretrain_data/CHEMBL_smiles.csv")
    missing = [path for path in required if not path.is_file()]
    if not missing:
        print("Using the versioned paper datasets in data/.")
        return
    if skip_download:
        listing = "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in missing)
        raise SystemExit(f"Required dataset files are missing:\n{listing}")
    run([sys.executable, "scripts/download_paper_datasets.py", "--datasets", *selected])


def write_quick_chembl(source: Path, destination: Path, limit: int = 128) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None or "smiles" not in reader.fieldnames:
            raise ValueError(f"{source} does not contain a smiles column")
        rows = []
        for row in reader:
            smiles = str(row.get("smiles") or "").strip()
            if smiles:
                rows.append({"smiles": smiles})
            if len(rows) == limit:
                break
    with destination.open("w", encoding="utf-8", newline="") as destination_handle:
        writer = csv.DictWriter(destination_handle, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def collect_summary(
    pipeline_root: Path, datasets: list[str], seeds: list[int], checkpoint: Path
) -> dict[str, object]:
    results = []
    for dataset in datasets:
        for seed in seeds:
            metrics_path = pipeline_root / "finetune" / dataset / str(seed) / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_metrics = payload["final"]["test"]["metrics"]
            metric_name = "roc_auc" if dataset in {"bbbp", "bace"} else "macro_roc_auc"
            results.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "metric": metric_name,
                    "value": test_metrics.get(metric_name),
                    "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
                }
            )
    return {
        "pretrained_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "results": results,
    }


def write_summary(pipeline_root: Path, summary: dict[str, object]) -> None:
    json_path = pipeline_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Full pipeline summary",
        "",
        f"Pretrained checkpoint: `{summary['pretrained_checkpoint']}`",
        "",
        "| Dataset | Seed | Test metric | Value |",
        "|---|---:|---|---:|",
    ]
    for result in summary["results"]:
        value = result["value"]
        display = "N/A" if value is None else f"{value:.6f}"
        lines.append(
            f"| {result['dataset']} | {result['seed']} | {result['metric']} | {display} |"
        )
    (pipeline_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nFull workflow completed. Summary: {json_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.quick and args.paper_scale:
        raise SystemExit("--quick and --paper-scale cannot be used together")
    if args.paper_scale and args.grover_checkpoint is None:
        raise SystemExit("--paper-scale requires --grover-checkpoint")
    if any(seed < 0 for seed in args.seeds):
        raise SystemExit("--seeds must contain non-negative integers")
    for name, value in (
        ("--pretrain-epochs", args.pretrain_epochs),
        ("--finetune-epochs", args.finetune_epochs),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive")

    pipeline_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", pipeline_id) or pipeline_id in {".", ".."}:
        raise SystemExit("--run-id may contain only letters, digits, '.', '_', and '-'")
    pipeline_root = project_path(args.output_root) / pipeline_id
    if pipeline_root.exists():
        raise SystemExit(f"Output already exists; choose another --run-id: {pipeline_root}")
    pipeline_root.mkdir(parents=True)

    ensure_datasets(args.datasets, skip_download=args.skip_download)
    full_chembl = PROJECT_ROOT / "data/pretrain_data/CHEMBL_smiles.csv"
    if args.quick:
        input_csv = pipeline_root / "input/CHEMBL_smiles.csv"
        count = write_quick_chembl(full_chembl, input_csv)
        print(f"Quick mode selected {count} ChEMBL molecules.")
    else:
        input_csv = full_chembl

    prepared_dir = pipeline_root / "prepared_pretraining"
    run(
        [
            sys.executable, "scripts/prepare_pretrain_data.py",
            "--input", str(input_csv),
            "--output", str(prepared_dir),
            "--num-shards", "4",
        ]
    )
    prepared_prefix = prepared_dir / "CHEMBL_smiles"

    pretrain_config = (
        "configs/pretrain.yaml" if args.paper_scale
        else "configs/pretrain_reproduction_compact.yaml"
    )
    pretrain_root = pipeline_root / "pretrain"
    pretrain_command = [
        sys.executable, "scripts/run_pretraining.py",
        "--config", pretrain_config,
        "--data-path", str(prepared_prefix),
        "--save-dir", str(pretrain_root),
        "--run-id", "model",
        "--device", args.device,
    ]
    if args.pretrain_epochs is not None:
        pretrain_command.extend(["--epochs", str(args.pretrain_epochs)])
    if args.grover_checkpoint is not None:
        pretrain_command.extend(
            ["--grover-checkpoint", str(project_path(args.grover_checkpoint))]
        )
    run(pretrain_command)
    checkpoint = pretrain_root / "model/checkpoints/best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Pretraining completed without the expected checkpoint: {checkpoint}")

    finetune_config = (
        "configs/finetune_paper.yaml" if args.paper_scale
        else "configs/finetune_reproduction_compact.yaml"
    )
    for dataset in args.datasets:
        finetune_command = [
            sys.executable, "scripts/run_finetuning.py",
            "--config", finetune_config,
            "--dataset", dataset,
            "--seeds", *[str(seed) for seed in args.seeds],
            "--device", args.device,
            "--pretrained-checkpoint", str(checkpoint),
            "--output-root", str(pipeline_root / "finetune" / dataset),
            "--run-id", "{seed}",
        ]
        if args.finetune_epochs is not None:
            finetune_command.extend(["--epochs", str(args.finetune_epochs)])
        if args.quick:
            finetune_command.extend(["--max-samples", "128", "--epochs", "1"])
        run(finetune_command)

    summary = collect_summary(pipeline_root, args.datasets, args.seeds, checkpoint)
    write_summary(pipeline_root, summary)


if __name__ == "__main__":
    main()

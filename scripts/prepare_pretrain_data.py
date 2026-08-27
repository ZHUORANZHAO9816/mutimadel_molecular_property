#!/usr/bin/env python3
"""Canonical command-line entry point for GTpro pretraining data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.data.pretraining import (
    PreprocessingConfig,
    download_chembl_smiles,
    prepare_pretraining_data,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic, validated GTpro pretraining shards and reports.",
        epilog="Relative paths are resolved against the repository root.",
    )
    parser.add_argument(
        "--input", "--csv-path", dest="csv_path",
        help="Existing CSV with a smiles column; omit to download from ChEMBL",
    )
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir", default="data/pretrain_data",
        help="Output directory for shards and reports (default: data/pretrain_data)",
    )
    parser.add_argument(
        "--max-molecules", type=int, default=1000,
        help="Maximum unique molecules to download when --input is omitted",
    )
    parser.add_argument("--num-shards", type=int, default=4, help="Requested number of balanced shards")
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Filter canonical SMILES longer than this token count (default: 200)",
    )
    parser.add_argument(
        "--keep-duplicates", action="store_true",
        help="Retain duplicate canonical molecules (default: keep first row only)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Rewrite all shards instead of skipping files verified by checksum",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_molecules <= 0:
        raise SystemExit("--max-molecules must be positive")
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.csv_path:
        csv_path = _project_path(args.csv_path)
        if not csv_path.is_file():
            raise SystemExit(f"Input CSV does not exist: {csv_path}")
    else:
        csv_path = output_dir / "CHEMBL_smiles.csv"
        if csv_path.exists():
            print(f"Using existing CSV: {csv_path}")
        else:
            count = download_chembl_smiles(args.max_molecules, csv_path, progress=print)
            if count == 0:
                raise SystemExit("ChEMBL download returned no molecules")

    config = PreprocessingConfig(
        num_shards=args.num_shards,
        max_smiles_tokens=args.max_tokens,
        deduplicate=not args.keep_duplicates,
        resume=not args.no_resume,
    )
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    report = prepare_pretraining_data(csv_path, output_dir, config=config, progress=print)
    counts = report["counts"]
    print(
        f"Completed: {counts['final_samples']} samples, {counts['failed_rows']} failed rows, "
        f"{counts['duplicate_rows']} duplicate rows"
    )
    print(f"Reports: {output_dir / 'data_report.json'} and {output_dir / 'data_report.md'}")


if __name__ == "__main__":
    main()

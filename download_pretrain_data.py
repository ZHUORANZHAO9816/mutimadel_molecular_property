#!/usr/bin/env python3
"""Deprecated compatibility CLI for GTpro pretraining-data preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gtpro.data.pretraining import (
    PreprocessingConfig,
    download_chembl_smiles,
    prepare_pretraining_data,
)


def build_pretrain_data(csv_path: str, output_dir: str, n_splits: int = 4) -> dict[str, object]:
    """Compatibility wrapper for callers of the historical function."""

    return prepare_pretraining_data(
        csv_path,
        output_dir,
        config=PreprocessingConfig(num_shards=n_splits),
        progress=print,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated wrapper; use scripts/prepare_pretrain_data.py")
    parser.add_argument("--max_molecules", "--max-molecules", type=int, default=1000)
    parser.add_argument("--output_dir", "--output-dir", default="./data/pretrain_data")
    parser.add_argument("--csv_path", "--csv-path")
    parser.add_argument("--num_shards", "--num-shards", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_molecules <= 0 or args.num_shards <= 0:
        raise SystemExit("--max-molecules and --num-shards must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else output_dir / "CHEMBL_smiles.csv"
    if not csv_path.exists():
        count = download_chembl_smiles(args.max_molecules, csv_path, progress=print)
        if count == 0:
            raise SystemExit("ChEMBL download returned no molecules")
    build_pretrain_data(str(csv_path), str(output_dir), n_splits=args.num_shards)


if __name__ == "__main__":
    print(
        "DEPRECATED compatibility entry: use scripts/prepare_pretrain_data.py instead.",
        file=sys.stderr,
    )
    main()

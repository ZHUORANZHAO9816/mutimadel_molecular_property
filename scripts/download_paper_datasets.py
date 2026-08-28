#!/usr/bin/env python3
"""Download the paper's ChEMBL pretraining data and MoleculeNet benchmarks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MOLECULENET_DATASETS = {
    "bbbp": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        Path("data/downstream/bbbp/raw/BBBP.csv"),
    ),
    "bace": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        Path("data/downstream/bace/raw/bace.csv"),
    ),
    "sider": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        Path("data/downstream/sider/raw/sider.csv"),
    ),
    "clintox": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        Path("data/downstream/clintox/raw/clintox.csv"),
    ),
    "tox21": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        Path("data/downstream/tox21/raw/tox21.csv"),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def download_csv(url: str, destination: Path, *, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"Using existing dataset: {destination.relative_to(PROJECT_ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.content
    if url.endswith(".gz"):
        payload = gzip.decompress(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Downloaded {url} -> {destination.relative_to(PROJECT_ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(MOLECULENET_DATASETS),
        default=list(MOLECULENET_DATASETS),
        help="MoleculeNet datasets to download (default: all paper benchmarks)",
    )
    parser.add_argument(
        "--chembl-limit",
        type=int,
        default=12008,
        help="Number of ChEMBL molecules to download (default: 12008)",
    )
    parser.add_argument("--skip-chembl", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chembl_limit <= 0:
        raise SystemExit("--chembl-limit must be positive")

    for name in args.datasets:
        url, relative_path = MOLECULENET_DATASETS[name]
        download_csv(url, PROJECT_ROOT / relative_path, force=args.force)

    chembl_path = PROJECT_ROOT / "data/pretrain_data/CHEMBL_smiles.csv"
    if not args.skip_chembl:
        if chembl_path.is_file() and not args.force:
            print(f"Using existing dataset: {chembl_path.relative_to(PROJECT_ROOT)}")
        else:
            from gtpro.data.pretraining import download_chembl_smiles

            count = download_chembl_smiles(args.chembl_limit, chembl_path, progress=print)
            if count != args.chembl_limit:
                raise RuntimeError(
                    f"requested {args.chembl_limit} ChEMBL molecules, downloaded {count}"
                )

    records = []
    for name, (url, relative_path) in MOLECULENET_DATASETS.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            continue
        records.append(
            {
                "name": name,
                "source": url,
                "path": relative_path.as_posix(),
                "rows": csv_rows(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not args.skip_chembl:
        records.append(
            {
                "name": "chembl_pretraining",
                "source": "https://www.ebi.ac.uk/chembl/api/data/molecule",
                "path": chembl_path.relative_to(PROJECT_ROOT).as_posix(),
                "rows": csv_rows(chembl_path),
                "bytes": chembl_path.stat().st_size,
                "sha256": sha256(chembl_path),
            }
        )
    manifest_path = PROJECT_ROOT / "data" / "paper_datasets.json"
    manifest_path.write_text(
        json.dumps({"datasets": records}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

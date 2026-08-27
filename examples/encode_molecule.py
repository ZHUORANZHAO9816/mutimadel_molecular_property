#!/usr/bin/env python3
"""Encode one or more SMILES through the public GTpro API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro import GTproEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode molecules with a D2 GTpro checkpoint")
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "runs/pretrain_reproduction_compact/compact_v1/checkpoints/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--representation", choices=("graph", "text", "joint"), default="joint")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    encoder = GTproEncoder.from_pretrained(args.checkpoint, device=args.device)
    value = args.smiles[0] if len(args.smiles) == 1 else args.smiles
    embeddings = encoder.encode_smiles(
        value, representation=args.representation, batch_size=args.batch_size
    )
    print(f"shape={tuple(embeddings.shape)} dtype={embeddings.dtype} device={embeddings.device}")


if __name__ == "__main__":
    main()

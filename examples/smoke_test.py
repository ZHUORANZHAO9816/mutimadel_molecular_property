#!/usr/bin/env python3
"""Run the GTpro joint forward smoke test from any working directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [sys.executable, str(PROJECT_ROOT / "test_forward.py")]
    print(f"Running GTpro smoke test from {PROJECT_ROOT}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

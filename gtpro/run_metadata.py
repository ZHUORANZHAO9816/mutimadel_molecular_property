"""Run-directory and reproducibility metadata recording."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _git_value(*args: str, allow_empty: bool = False) -> str | None:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value or allow_empty else None


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class RunRecorder:
    """Create a run directory and finalize its environment manifest."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        command: list[str] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.started_at = started_at or _utc_now()
        seed = self.config["seed"]
        configured_run_id = self.config.get("output", {}).get("run_id")
        if configured_run_id is None:
            run_id = f"{self.started_at.strftime('%Y%m%dT%H%M%S.%fZ')}_seed{seed}"
        else:
            run_id = str(configured_run_id).format(seed=seed)
            if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id) or run_id in {".", ".."}:
                raise ValueError(f"output.run_id is not a safe directory name: {run_id!r}")
        self.run_dir = Path(self.config["output"]["root"]) / run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.command = list(command if command is not None else sys.argv)
        self._start_clock = monotonic()
        self._metadata: dict[str, Any] = {}

    def __enter__(self) -> "RunRecorder":
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        _atomic_text(
            self.run_dir / "config.yaml",
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True),
        )

        git_commit = _git_value("rev-parse", "HEAD")
        git_status = _git_value("status", "--porcelain", allow_empty=True)
        self._metadata = {
            "status": "running",
            "seed": self.config["seed"],
            "device": self.config["device"],
            "started_at": self.started_at.isoformat(),
            "ended_at": None,
            "duration_seconds": None,
            "command": self.command,
            "config_source": self.config.get("config_source"),
            "run_directory": str(self.run_dir.resolve()),
            "git": {
                "commit": git_commit,
                "dirty": bool(git_status) if git_status is not None else None,
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.replace("\n", " "),
                "pytorch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": bool(
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ),
            },
        }
        self._write_metadata()
        return self

    def _write_metadata(self) -> None:
        _atomic_text(
            self.run_dir / "environment.json",
            json.dumps(self._metadata, indent=2, sort_keys=True) + "\n",
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        ended_at = _utc_now()
        self._metadata["status"] = "success" if exc_type is None else "failed"
        self._metadata["ended_at"] = ended_at.isoformat()
        self._metadata["duration_seconds"] = round(monotonic() - self._start_clock, 6)
        if exc_type is not None:
            self._metadata["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }
        self._write_metadata()
        return False

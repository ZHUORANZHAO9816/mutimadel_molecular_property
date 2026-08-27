"""CLI location, help, and compatibility tests."""

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        ("scripts/prepare_pretrain_data.py", "pretraining shards"),
        ("scripts/run_pretraining.py", "Graph-Text Pretraining"),
        ("scripts/run_finetuning.py", "molecular-property fine-tuning"),
    ],
)
def test_entrypoint_help_works_outside_repository(
    tmp_path: Path, relative_path: str, expected: str
) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative_path), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert expected.lower() in result.stdout.lower()


def test_finetuning_missing_config_fails_explicitly(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_finetuning.py"),
            "--config",
            "configs/does-not-exist.yaml",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr.lower()


def test_relative_config_is_resolved_from_project_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_pretraining.py"),
            "--config",
            "configs/does-not-exist.yaml",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    expected = str(PROJECT_ROOT / "configs" / "does-not-exist.yaml")
    assert expected in result.stderr


def test_relative_data_input_is_resolved_from_project_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/prepare_pretrain_data.py"),
            "--input",
            "data/does-not-exist.csv",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    expected = str(PROJECT_ROOT / "data" / "does-not-exist.csv")
    assert expected in result.stderr

"""Tests for checked YAML configs and per-run reproducibility metadata."""

import json
from pathlib import Path

import pytest
import yaml

from gtpro.config import PROJECT_ROOT, ConfigError, load_finetune_config, load_pretrain_config
from gtpro.run_metadata import RunRecorder


@pytest.mark.parametrize("name", ["pretrain_smoke.yaml", "pretrain.yaml"])
def test_checked_pretrain_configs_are_valid_and_resolved(name: str) -> None:
    config = load_pretrain_config(PROJECT_ROOT / "configs" / name)

    assert Path(config["data"]["path"]).is_absolute()
    assert Path(config["output"]["root"]).is_absolute()
    assert config["training"]["batch_size"] > 0
    assert config["training"]["epochs"] > 0
    assert config["training"]["learning_rate"] > 0
    assert config["model"]["text"]["d_model"] > 0
    assert config["seed"] >= 0


def test_cli_style_overrides_replace_config_values(tmp_path: Path) -> None:
    config = load_pretrain_config(
        PROJECT_ROOT / "configs" / "pretrain_smoke.yaml",
        {
            "seed": 0,
            "training.batch_size": 2,
            "output.root": str(tmp_path / "runs"),
        },
    )

    assert config["seed"] == 0
    assert config["training"]["batch_size"] == 2
    assert config["output"]["root"] == str((tmp_path / "runs").resolve())


def test_invalid_config_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="batch_size"):
        load_pretrain_config(
            PROJECT_ROOT / "configs" / "pretrain_smoke.yaml",
            {"training.batch_size": 0},
        )


@pytest.mark.parametrize(
    "name",
    [
        "finetune_bace_smoke.yaml",
        "finetune_tox21_smoke.yaml",
        "finetune_lipophilicity_smoke.yaml",
    ],
)
def test_checked_finetune_configs_are_valid_and_resolved(name: str) -> None:
    config = load_finetune_config(PROJECT_ROOT / "configs" / name)
    assert Path(config["data"]["root"]).is_absolute()
    assert Path(config["output"]["root"]).is_absolute()
    assert config["seeds"]
    assert config["model"]["encoder_mode"] in {"frozen", "partial", "full"}


def test_run_recorder_writes_final_config_and_success_metadata(tmp_path: Path) -> None:
    config = load_pretrain_config(
        PROJECT_ROOT / "configs" / "pretrain_smoke.yaml",
        {"output.root": str(tmp_path / "runs")},
    )

    with RunRecorder(config, command=["gtpro-test", "--config", "smoke"]) as run:
        assert (run.run_dir / "config.yaml").is_file()
        running = json.loads((run.run_dir / "environment.json").read_text())
        assert running["status"] == "running"
        assert running["seed"] == config["seed"]
        assert running["started_at"]
        assert running["ended_at"] is None

    saved_config = yaml.safe_load((run.run_dir / "config.yaml").read_text())
    metadata = json.loads((run.run_dir / "environment.json").read_text())
    assert saved_config == config
    assert metadata["status"] == "success"
    assert metadata["ended_at"]
    assert metadata["duration_seconds"] >= 0
    assert metadata["command"] == ["gtpro-test", "--config", "smoke"]
    assert set(metadata["git"]) == {"commit", "dirty"}
    assert metadata["environment"]["python"]
    assert metadata["environment"]["pytorch"]


def test_run_recorder_marks_failed_runs(tmp_path: Path) -> None:
    config = load_pretrain_config(
        PROJECT_ROOT / "configs" / "pretrain_smoke.yaml",
        {"output.root": str(tmp_path / "runs")},
    )

    with pytest.raises(RuntimeError, match="expected failure"):
        with RunRecorder(config, command=["gtpro-test"]) as run:
            raise RuntimeError("expected failure")

    metadata = json.loads((run.run_dir / "environment.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error"] == {"type": "RuntimeError", "message": "expected failure"}
    assert metadata["ended_at"]

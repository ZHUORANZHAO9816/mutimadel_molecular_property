"""CPU-only command integration checks over tracked, bounded fixtures."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml

from gtpro.config import PROJECT_ROOT


def _run(script: str, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--config", str(config)],
        cwd=config.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_one_epoch_pretraining_command_on_cpu(tmp_path: Path) -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/pretrain_smoke.yaml").read_text())
    config["data"]["path"] = str(PROJECT_ROOT / "data/pretrain_data/gtpro_smoke")
    config["output"]["root"] = str(tmp_path / "pretrain-runs")
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = _run("scripts/run_pretraining.py", config_path)
    assert result.returncode == 0, result.stderr
    run = next((tmp_path / "pretrain-runs").iterdir())
    assert (run / "checkpoints/best.pt").is_file()
    assert (run / "checkpoints/last.pt").is_file()
    assert (run / "metrics.json").is_file()


def _write_downstream(root: Path, dataset: str) -> None:
    smiles = ["C", "CC", "CCC", "CCCC", "CO", "CCO", "CCN", "CN", "O", "N", "c1ccccc1", "C1CCCCC1"]
    if dataset == "bace":
        path = root / "bace/raw/bace.csv"
        fields = ["mol", "Class", "CID"]
        rows = ({"mol": smi, "Class": index % 2, "CID": f"b{index}"} for index, smi in enumerate(smiles))
    else:
        path = root / "lipophilicity/raw/Lipophilicity.csv"
        fields = ["smiles", "exp", "CMPD_CHEMBLID"]
        rows = ({"smiles": smi, "exp": index / 10, "CMPD_CHEMBLID": f"l{index}"} for index, smi in enumerate(smiles))
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finetune_config(tmp_path: Path, dataset: str) -> Path:
    source = "finetune_bace_smoke.yaml" if dataset == "bace" else "finetune_lipophilicity_smoke.yaml"
    config = yaml.safe_load((PROJECT_ROOT / f"configs/{source}").read_text())
    data_root = tmp_path / "downstream"
    _write_downstream(data_root, dataset)
    config["data"].update(root=str(data_root), fractions=[0.6, 0.2, 0.2], max_samples=None)
    config["training"]["epochs"] = 1
    config["output"]["root"] = str(tmp_path / f"{dataset}-runs")
    path = tmp_path / f"{dataset}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_bace_smoke_finetuning_command_on_cpu(tmp_path: Path) -> None:
    result = _run("scripts/run_finetuning.py", _finetune_config(tmp_path, "bace"))
    assert result.returncode == 0, result.stderr
    run = next((tmp_path / "bace-runs").iterdir())
    for relative in ("checkpoints/best.pt", "predictions.csv", "metrics.json", "config.yaml"):
        assert (run / relative).is_file()


def test_lipophilicity_smoke_finetuning_command_on_cpu(tmp_path: Path) -> None:
    result = _run("scripts/run_finetuning.py", _finetune_config(tmp_path, "lipophilicity"))
    assert result.returncode == 0, result.stderr
    run = next((tmp_path / "lipophilicity-runs").iterdir())
    for relative in ("checkpoints/best.pt", "predictions.csv", "metrics.json", "config.yaml"):
        assert (run / relative).is_file()

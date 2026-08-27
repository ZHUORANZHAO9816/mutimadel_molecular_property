"""Source and checkpoint compatibility tests for the bundled GROVER encoder."""

import ast
import inspect
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from gtpro.graph_trans.model.models import GROVEREmbedding
from gtpro.graph_trans.util.utils import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROVER_ROOT = (PROJECT_ROOT / "gtpro" / "graph_trans").resolve()


def _small_grover_args() -> Namespace:
    return Namespace(
        hidden_size=16,
        backbone="dualtrans",
        embedding_output_type="both",
        dropout=0.0,
        activation="PReLU",
        num_mt_block=1,
        num_attn_head=4,
        bias=False,
        cuda=False,
        depth=2,
        dense=False,
        undirected=False,
        bond_drop_rate=0,
        features_only=False,
        no_cache=True,
    )


def test_grover_embedding_is_loaded_from_bundled_package() -> None:
    source = Path(inspect.getfile(GROVEREmbedding)).resolve()
    print(f"GROVEREmbedding source: {source}")
    assert source.is_relative_to(GROVER_ROOT)


def test_project_python_does_not_import_external_grover() -> None:
    roots = [PROJECT_ROOT / "gtpro", PROJECT_ROOT / "pretrain", PROJECT_ROOT / "finetune"]
    files = [PROJECT_ROOT / "test_forward.py"]
    files.extend(path for root in roots for path in root.rglob("*.py"))

    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(module == "grover" or module.startswith("grover.") for module in modules):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert offenders == []


@pytest.mark.parametrize("prefix", ["", "grover."])
def test_checkpoint_loader_accepts_embedding_and_official_prefix(tmp_path: Path, prefix: str) -> None:
    args = _small_grover_args()
    source_model = GROVEREmbedding(args)
    source_state = source_model.state_dict()
    checkpoint = tmp_path / ("grover_prefixed.pt" if prefix else "embedding.pt")
    torch.save(
        {
            "args": args,
            "state_dict": {f"{prefix}{key}": value.clone() for key, value in source_state.items()},
        },
        checkpoint,
    )

    loaded_model = load_checkpoint(str(checkpoint), current_args=deepcopy(args), cuda=False)
    loaded_state = loaded_model.state_dict()

    assert loaded_state.keys() == source_state.keys()
    assert all(torch.equal(source_state[key], loaded_state[key]) for key in source_state)


def test_checkpoint_loader_rejects_unrelated_state_dict(tmp_path: Path) -> None:
    args = _small_grover_args()
    checkpoint = tmp_path / "unrelated.pt"
    torch.save({"args": args, "state_dict": {"head.weight": torch.ones(2, 2)}}, checkpoint)

    with pytest.raises(RuntimeError, match="no parameters compatible"):
        load_checkpoint(str(checkpoint), current_args=deepcopy(args), cuda=False)

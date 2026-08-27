from __future__ import annotations

import pytest
import torch
from torch import nn

from gtpro.training.finetuning import FinetuningBatch, load_pretrained_encoders, masked_task_loss


def _modules():
    return nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)), nn.Linear(2, 2)


def test_pretrained_encoder_checkpoint_exact_load_reports_no_mismatch(tmp_path):
    text, graph = _modules()
    path = tmp_path / "exact.pt"
    torch.save({"models": {"text": text.state_dict(), "graph": graph.state_dict()}}, path)
    report = load_pretrained_encoders(
        path, text_encoder=text, graph_encoder=graph, strict=True,
        max_mismatch_fraction=0.0, map_location="cpu",
    )
    assert report["text"]["missing_keys"] == []
    assert report["graph"]["unexpected_keys"] == []


def test_pretrained_encoder_strict_load_reports_missing_and_unexpected_keys(tmp_path):
    text, graph = _modules()
    text_state = dict(text.state_dict())
    text_state.pop("0.bias")
    text_state["unexpected"] = torch.zeros(1)
    path = tmp_path / "mismatch.pt"
    torch.save({"models": {"text": text_state, "graph": graph.state_dict()}}, path)
    with pytest.raises(ValueError, match="missing_keys.*unexpected_keys"):
        load_pretrained_encoders(
            path, text_encoder=text, graph_encoder=graph, strict=True,
            max_mismatch_fraction=1.0, map_location="cpu",
        )


def test_pretrained_encoder_large_mismatch_cannot_be_silently_ignored(tmp_path):
    text, graph = _modules()
    path = tmp_path / "large.pt"
    torch.save({"models": {"text": {}, "graph": graph.state_dict()}}, path)
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        load_pretrained_encoders(
            path, text_encoder=text, graph_encoder=graph, strict=False,
            max_mismatch_fraction=0.1, map_location="cpu",
        )


def test_multilabel_loss_ignores_missing_targets():
    batch = FinetuningBatch(
        token_ids=torch.ones((2, 3), dtype=torch.long),
        targets=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        target_mask=torch.tensor([[True, False], [True, True]]),
        smiles=("C", "CC"),
        indices=torch.tensor([0, 1]),
    )
    logits = torch.zeros((2, 2))
    loss, count = masked_task_loss(logits, batch, "multilabel_classification", torch.ones(2))
    assert count == 3
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())

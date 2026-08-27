from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from gtpro.training.pretraining import (
    PretrainingModels,
    load_pretraining_checkpoint,
    save_pretraining_checkpoint,
)


def _models() -> PretrainingModels:
    return PretrainingModels(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1))


def test_pretraining_checkpoint_restores_models_and_training_state(tmp_path):
    models = _models()
    optimizer = torch.optim.Adam(models.trainable_parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    loss = sum(parameter.square().sum() for parameter in models.trainable_parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = copy.deepcopy(models.state_dict())

    path = tmp_path / "checkpoint.pt"
    save_pretraining_checkpoint(
        path,
        epoch=2,
        seed=42,
        config={"training": {"epochs": 3}},
        models=models,
        optimizer=optimizer,
        scheduler=scheduler,
        best_validation_loss=0.25,
        history=[{"epoch": 2}],
    )
    with torch.no_grad():
        for parameter in models.trainable_parameters():
            parameter.add_(10)

    payload = load_pretraining_checkpoint(
        path,
        models=models,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location="cpu",
    )

    assert payload["epoch"] == 2
    assert payload["seed"] == 42
    assert payload["best_validation_loss"] == pytest.approx(0.25)
    assert scheduler.last_epoch == 1
    for group, state in expected.items():
        for name, tensor in state.items():
            assert torch.equal(models.state_dict()[group][name], tensor)


def test_pretraining_checkpoint_rejects_missing_state(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"epoch": 1}, path)
    models = _models()
    optimizer = torch.optim.Adam(models.trainable_parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    with pytest.raises(ValueError, match="missing keys"):
        load_pretraining_checkpoint(
            path,
            models=models,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )

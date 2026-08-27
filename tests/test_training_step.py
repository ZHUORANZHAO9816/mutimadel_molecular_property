from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from gtpro.models.interfaces import AlignmentLosses, TextEncoderOutput
from gtpro.training.pretraining import (
    PretrainingModels,
    collate_pretraining_samples,
    run_pretraining_epoch,
)


class TinyText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(47, 8)
        self.molecule = nn.Linear(8, 154)

    def forward(self, tokens: torch.Tensor) -> TextEncoderOutput:
        all_tokens = self.embedding(tokens)
        global_embedding = all_tokens[:, 0]
        return TextEncoderOutput(
            all_tokens=all_tokens,
            global_embedding=global_embedding,
            atom_tokens=all_tokens[:, 1:],
            molecule_logits=self.molecule(global_embedding),
        )


class TinyGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, components):
        atom_features = components[0]
        return atom_features[:, :8] * self.scale


class TinyAlignment(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, image_tokens, image, text, text_tokens, *args, **kwargs):
        graph_mean = image_tokens.mean()
        text_mean = text.mean()
        atom = (graph_mean + self.bias).square()
        functional_group = (text_mean + self.bias).square()
        contrastive = (graph_mean - text_mean).square()
        return AlignmentLosses(
            atom=atom,
            functional_group=functional_group,
            contrastive=contrastive,
            total=atom + functional_group + contrastive,
        )


def _sample(smiles: str = "CCO"):
    tokens = np.zeros(201, dtype=np.int64)
    tokens[:4] = [1, 2, 3, 4]
    global_labels = np.zeros(154, dtype=np.float32)
    atom_labels = np.zeros((200, 15), dtype=np.float32)
    atom_mask = np.zeros(200, dtype=np.int8)
    atom_mask[:3] = 1
    return tokens, global_labels, atom_labels, atom_mask, smiles


def _arguments() -> Namespace:
    return Namespace(bond_drop_rate=0.0, no_cache=True)


def test_training_epoch_handles_last_partial_batch_and_updates_parameters():
    models = PretrainingModels(TinyText(), TinyGraph(), TinyAlignment())
    optimizer = torch.optim.Adam(models.trainable_parameters(), lr=0.01)
    loader = DataLoader(
        [_sample("CCO"), _sample("CCN"), _sample("c1ccccc1")],
        batch_size=2,
        collate_fn=collate_pretraining_samples,
        drop_last=False,
    )
    before = models.text.embedding.weight.detach().clone()
    metrics = run_pretraining_epoch(
        models=models,
        data_loader=loader,
        grover_args=_arguments(),
        device=torch.device("cpu"),
        atom_loss=nn.BCEWithLogitsLoss(reduction="none"),
        molecule_loss=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        training=True,
        gradient_clip_norm=1.0,
    )
    assert metrics.samples == 3
    assert metrics.batches == 2
    assert metrics.gradient_norm is not None
    assert not torch.equal(before, models.text.embedding.weight)


def test_training_epoch_reports_empty_and_invalid_input():
    models = PretrainingModels(TinyText(), TinyGraph(), TinyAlignment())
    optimizer = torch.optim.Adam(models.trainable_parameters(), lr=0.01)
    common = dict(
        models=models,
        grover_args=_arguments(),
        device=torch.device("cpu"),
        atom_loss=nn.BCEWithLogitsLoss(reduction="none"),
        molecule_loss=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        training=True,
    )
    with pytest.raises(ValueError, match="produced no batches"):
        run_pretraining_epoch(data_loader=[], **common)

    invalid_loader = DataLoader(
        [_sample("not-a-smiles")], batch_size=1, collate_fn=collate_pretraining_samples
    )
    with pytest.raises(ValueError, match="failed to build/encode graph batch"):
        run_pretraining_epoch(data_loader=invalid_loader, **common)


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty pretraining batch"):
        collate_pretraining_samples([])

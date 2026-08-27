#!/usr/bin/env python3
"""Small CPU model smoke check suitable for local runs and CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.data.pretraining import process_smiles
from gtpro.graph_trans.model.models import GROVEREmbedding
from gtpro.training.pretraining import PretrainingModels, collate_pretraining_samples, run_pretraining_epoch
from pretrain.mutimodal_trans import CoCa
from pretrain.seq_trans import K_BERT_WCL, set_random_seed


def main() -> None:
    set_random_seed(7)
    graph_args = argparse.Namespace(
        hidden_size=32, backbone="dualtrans", embedding_output_type="both", dropout=0.0,
        activation="PReLU", num_mt_block=1, num_attn_head=4, bias=False, cuda=False,
        depth=3, dense=False, undirected=False, bond_drop_rate=0.0,
        features_only=False, no_cache=True,
    )
    text = K_BERT_WCL(32, 1, 47, 201, 8, 8, 4, 128, 154, 15)
    graph = GROVEREmbedding(graph_args)
    alignment = CoCa(
        dim=32, img_encoder=None, image_dim=32, num_tokens=15, sub_graph=85,
        unimodal_depth=1, multimodal_depth=1, dim_head=8, heads=4,
        caption_loss_weight=1.0, contrastive_loss_weight=1.0,
    )
    models = PretrainingModels(text, graph, alignment)
    samples = []
    for smiles in ("CCO", "CCN"):
        value = process_smiles(smiles)
        samples.append((value.token_ids, value.global_labels, value.atom_labels, value.atom_mask, value.canonical_smiles))
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_pretraining_samples)
    optimizer = torch.optim.Adam(models.trainable_parameters(), lr=1e-4)
    metrics = run_pretraining_epoch(
        models=models, data_loader=loader, grover_args=graph_args, device=torch.device("cpu"),
        atom_loss=torch.nn.BCEWithLogitsLoss(reduction="none"),
        molecule_loss=torch.nn.BCEWithLogitsLoss(), optimizer=optimizer, training=True,
        gradient_clip_norm=1.0,
    )
    print(f"GTpro CI smoke passed: samples={metrics.samples}, total_loss={metrics.total_loss:.6f}")


if __name__ == "__main__":
    main()

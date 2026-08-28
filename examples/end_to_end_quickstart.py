#!/usr/bin/env python3
"""Run a tiny CPU-only pretraining, fine-tuning, and evaluation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.data.downstream import DatasetSpec, load_downstream_csv
from gtpro.data.pretraining import process_smiles
from gtpro.graph_trans.model.models import GROVEREmbedding
from gtpro.training.finetuning import (
    MolecularPropertyPredictor,
    MolecularSubset,
    class_positive_weights,
    collate_finetuning_samples,
    configure_encoder_mode,
    result_metrics,
    run_finetuning_epoch,
)
from gtpro.training.pretraining import (
    PretrainingModels,
    collate_pretraining_samples,
    run_pretraining_epoch,
)
from pretrain.mutimodal_trans import CoCa
from pretrain.seq_trans import K_BERT_WCL, set_random_seed


def graph_arguments() -> argparse.Namespace:
    return argparse.Namespace(
        hidden_size=32,
        backbone="dualtrans",
        embedding_output_type="both",
        dropout=0.0,
        activation="PReLU",
        num_mt_block=1,
        num_attn_head=4,
        bias=False,
        cuda=False,
        depth=3,
        dense=False,
        undirected=False,
        bond_drop_rate=0.0,
        features_only=False,
        no_cache=True,
    )


def build_pretraining_models(graph_args: argparse.Namespace) -> PretrainingModels:
    text_encoder = K_BERT_WCL(
        d_model=32,
        n_layers=1,
        vocab_size=47,
        maxlen=201,
        d_k=8,
        d_v=8,
        n_heads=4,
        d_ff=128,
        global_label_dim=154,
        atom_label_dim=15,
    )
    graph_encoder = GROVEREmbedding(graph_args)
    alignment_model = CoCa(
        dim=32,
        img_encoder=None,
        image_dim=32,
        num_tokens=15,
        sub_graph=85,
        unimodal_depth=1,
        multimodal_depth=1,
        dim_head=8,
        heads=4,
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
    )
    return PretrainingModels(text_encoder, graph_encoder, alignment_model)


def run_tiny_pretraining(
    models: PretrainingModels,
    graph_args: argparse.Namespace,
    smiles: tuple[str, ...],
    device: torch.device,
) -> float:
    samples = []
    for value in map(process_smiles, smiles):
        samples.append(
            (
                value.token_ids,
                value.global_labels,
                value.atom_labels,
                value.atom_mask,
                value.canonical_smiles,
            )
        )
    loader = DataLoader(
        samples,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_pretraining_samples,
    )
    optimizer = Adam(models.trainable_parameters(), lr=1e-4)
    metrics = run_pretraining_epoch(
        models=models,
        data_loader=loader,
        grover_args=graph_args,
        device=device,
        atom_loss=torch.nn.BCEWithLogitsLoss(reduction="none"),
        molecule_loss=torch.nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        training=True,
        gradient_clip_norm=1.0,
    )
    return metrics.total_loss


def downstream_loader(dataset, indices: list[int]) -> DataLoader:
    return DataLoader(
        MolecularSubset(dataset, indices, max_tokens=200),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_finetuning_samples,
    )


def main() -> None:
    device = torch.device("cpu")
    set_random_seed(7)
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "downstream_smoke.csv"
    spec = DatasetSpec(
        name="quickstart",
        relative_path=str(fixture),
        smiles_column="smiles",
        target_columns=("binary_label",),
        task_type="binary_classification",
        native_missing_label="empty CSV field",
        id_column="id",
    )
    RDLogger.DisableLog("rdApp.error")
    try:
        dataset = load_downstream_csv(spec, fixture)
    finally:
        RDLogger.EnableLog("rdApp.error")

    graph_args = graph_arguments()
    models = build_pretraining_models(graph_args)
    pretraining_smiles = tuple(dataset.canonical_smiles[:8])
    pretraining_loss = run_tiny_pretraining(
        models, graph_args, pretraining_smiles, device
    )

    predictor = MolecularPropertyPredictor(
        models.text,
        models.graph,
        graph_args,
        text_dim=32,
        graph_dim=32,
        output_dim=1,
        hidden_dim=16,
        dropout=0.0,
        representation="joint",
    ).to(device)
    parameter_counts = configure_encoder_mode(predictor, "frozen", 0)
    optimizer = AdamW(
        [parameter for parameter in predictor.parameters() if parameter.requires_grad],
        lr=1e-3,
    )

    # Fixed, class-balanced partitions keep this small demonstration deterministic.
    train_indices = [0, 1, 2, 4, 7]
    validation_indices = [3, 5]
    test_indices = [6, 8]
    positive_weight_values, _ = class_positive_weights(
        dataset.targets[train_indices], "auto"
    )
    positive_weights = torch.as_tensor(positive_weight_values, device=device)

    train_result = None
    for _ in range(2):
        train_result = run_finetuning_epoch(
            model=predictor,
            data_loader=downstream_loader(dataset, train_indices),
            device=device,
            task_type=dataset.spec.task_type,
            optimizer=optimizer,
            training=True,
            positive_weights=positive_weights,
            gradient_clip_norm=1.0,
        )
    assert train_result is not None
    validation_result = run_finetuning_epoch(
        model=predictor,
        data_loader=downstream_loader(dataset, validation_indices),
        device=device,
        task_type=dataset.spec.task_type,
        optimizer=None,
        training=False,
        positive_weights=positive_weights,
        gradient_clip_norm=None,
    )
    test_result = run_finetuning_epoch(
        model=predictor,
        data_loader=downstream_loader(dataset, test_indices),
        device=device,
        task_type=dataset.spec.task_type,
        optimizer=None,
        training=False,
        positive_weights=positive_weights,
        gradient_clip_norm=None,
    )
    test_metrics = result_metrics(dataset, test_result)

    if not np.isfinite(pretraining_loss):
        raise RuntimeError("pretraining produced a non-finite loss")
    print("End-to-end quick start completed on CPU")
    print(f"  pretraining molecules: {len(pretraining_smiles)}")
    print(f"  pretraining loss:      {pretraining_loss:.4f}")
    print(f"  train/valid/test:      {len(train_indices)}/{len(validation_indices)}/{len(test_indices)}")
    print(f"  trainable parameters:  {parameter_counts['trainable']}")
    print(f"  fine-tuning loss:      {train_result['loss']:.4f}")
    print(f"  validation loss:       {validation_result['loss']:.4f}")
    print(f"  example test ROC-AUC:  {test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

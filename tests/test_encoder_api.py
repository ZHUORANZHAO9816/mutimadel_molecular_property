from __future__ import annotations

import argparse

import pytest
import torch

from gtpro import GTproEncoder
from gtpro.graph_trans.model.models import GROVEREmbedding
from pretrain.seq_trans import K_BERT_WCL


@pytest.fixture()
def encoder_checkpoint(tmp_path):
    text_config = {
        "d_model": 16, "n_layers": 1, "vocab_size": 47, "max_length": 201,
        "d_k": 4, "d_v": 4, "n_heads": 2, "d_ff": 32,
        "global_label_dim": 154, "atom_label_dim": 15,
    }
    graph_config = {
        "hidden_size": 16, "backbone": "dualtrans", "embedding_output_type": "both",
        "dropout": 0.0, "activation": "PReLU", "num_mt_block": 1,
        "num_attn_head": 2, "bias": False, "depth": 3, "dense": False,
        "undirected": False, "bond_drop_rate": 0.0, "features_only": False,
        "no_cache": True,
    }
    args = argparse.Namespace(**graph_config, cuda=False)
    text = K_BERT_WCL(
        text_config["d_model"], text_config["n_layers"], text_config["vocab_size"],
        text_config["max_length"], text_config["d_k"], text_config["d_v"],
        text_config["n_heads"], text_config["d_ff"], text_config["global_label_dim"],
        text_config["atom_label_dim"],
    )
    graph = GROVEREmbedding(args)
    path = tmp_path / "encoder.pt"
    torch.save({"config": {"model": {"text": text_config, "grover": graph_config}},
                "models": {"text": text.state_dict(), "graph": graph.state_dict()}}, path)
    return path


def test_public_encoder_single_batch_and_representations(encoder_checkpoint):
    encoder = GTproEncoder.from_pretrained(encoder_checkpoint, device="cpu")
    assert encoder.encode_smiles("CCO", representation="text").shape == (16,)
    assert encoder.encode_smiles(["CCO", "CCN", "CCC"], representation="graph", batch_size=2).shape == (3, 16)
    joint = encoder.encode_smiles(["CCO"], representation="joint")
    assert joint.shape == (1, 32)
    assert joint.dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_public_encoder_invalid_smiles_policies(encoder_checkpoint):
    encoder = GTproEncoder.from_pretrained(encoder_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="index 1"):
        encoder.encode_smiles(["CCO", "not-a-smiles"])
    output = encoder.encode_smiles(["CCO", "not-a-smiles", "CCN"], invalid_smiles="nan")
    assert output.shape == (3, 32)
    assert torch.isnan(output[1]).all()
    assert torch.isfinite(output[[0, 2]]).all()

"""Shape and mask contracts at GTpro model boundaries."""

from __future__ import annotations

import torch

from gtpro.models.interfaces import (
    PretrainingBatch,
    TextEncoderOutput,
    encode_graph,
    normalize_grover_atom_output,
)
from pretrain.seq_trans import K_BERT_WCL, get_attn_pad_mask


def test_text_encoder_output_and_padding_mask_shapes() -> None:
    model = K_BERT_WCL(
        d_model=16,
        n_layers=1,
        vocab_size=47,
        maxlen=6,
        d_k=4,
        d_v=4,
        n_heads=2,
        d_ff=32,
        global_label_dim=154,
        atom_label_dim=15,
    )
    token_ids = torch.tensor([[1, 3, 6, 0, 0, 0], [1, 3, 3, 6, 0, 0]])

    output = model(token_ids)
    padding_mask = get_attn_pad_mask(token_ids)

    assert isinstance(output, TextEncoderOutput)
    assert output.all_tokens.shape == (2, 6, 16)
    assert output.global_embedding.shape == (2, 16)
    assert output.atom_tokens.shape == (2, 5, 16)
    assert output.molecule_logits.shape == (2, 154)
    assert padding_mask.dtype == torch.bool
    assert padding_mask.shape == (2, 6, 6)
    assert padding_mask[0, 0].tolist() == [False, False, False, True, True, True]


def test_pretraining_batch_mask_semantics_and_device_move() -> None:
    batch = PretrainingBatch(
        token_ids=torch.tensor([[1, 3, 6, 0]]),
        global_labels=torch.zeros(1, 154),
        atom_labels=torch.zeros(1, 3, 15),
        atom_mask=torch.tensor([[1, 1, 0]]),
        smiles=("CO",),
    ).to("cpu")

    assert batch.padding_mask.tolist() == [[False, False, False, True]]
    assert batch.atom_target_mask.tolist() == [[True, True, False]]
    assert batch.token_ids.dtype == torch.long
    assert batch.atom_labels.dtype == torch.float32
    assert batch.atom_mask.dtype == torch.bool


class _DictGrover(torch.nn.Module):
    def __init__(self, atom_embeddings: torch.Tensor):
        super().__init__()
        self.register_buffer("atom_embeddings", atom_embeddings)

    def forward(self, components):
        return {"atom_from_atom": self.atom_embeddings}


def test_grover_schema_normalization_and_graph_pooling() -> None:
    atoms = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    scope = torch.tensor([[1, 2], [3, 3]])
    components = (torch.zeros(1), None, None, None, None, scope)

    output = encode_graph(_DictGrover(atoms), components, "cpu")

    assert output.atom_embeddings.shape == (6, 3)
    assert output.molecule_embeddings.shape == (2, 3)
    assert torch.equal(output.molecule_embeddings[0], atoms[1:3].mean(dim=0))
    assert torch.equal(output.molecule_embeddings[1], atoms[3:6].mean(dim=0))
    assert normalize_grover_atom_output((atoms, torch.zeros_like(atoms))) is atoms
    assert normalize_grover_atom_output(atoms) is atoms

"""
Quick test: verify that all three models (K_BERT_WCL + GROVER + CoCa)
can run a forward pass together.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pretrain'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from argparse import Namespace
from gtpro.utils import get_device
from gtpro.models.interfaces import encode_graph

device = get_device()
print(f"PyTorch: {torch.__version__}, device={device}")

# ---------- 1. Mock data ----------
B, S = 4, 201  # seq_len=201 for CoCa multi-modal layers
vocab_size = 47

# Random SMILES tokens
token_idx = torch.randint(1, vocab_size, (B, S), dtype=torch.long)

# Random atom labels (15-dim multi-label) - sized for [CLS] removed (S-1)
atom_labels = (torch.rand(B, S - 1, 15) > 0.5).float()
atom_mask = torch.ones(B, S - 1, 1)

# Random "functional group" labels (85-dim)
fg_feature = (torch.rand(B, 85) > 0.5).float()

# ---------- 2. Build SMILES BERT ----------
from seq_trans import K_BERT_WCL
bert = K_BERT_WCL(
    d_model=768, n_layers=6, vocab_size=vocab_size, maxlen=S,
    d_k=64, d_v=64, n_heads=12, d_ff=768*4,
    global_label_dim=154, atom_label_dim=15, use_atom=True, device=device
)
bert = bert.to(device)

print(f"BERT params: {sum(p.numel() for p in bert.parameters()):,}")

# ---------- 3. Build GROVER ----------
from gtpro.graph_trans.model.models import GROVEREmbedding

grover_args = Namespace(
    hidden_size=1200, backbone='dualtrans', embedding_output_type='atom',
    dropout=0.0, activation='PReLU', num_mt_block=1, num_attn_head=4,
    bias=False, cuda=(device != 'cpu'), depth=6, undirected=False, bond_drop_rate=0,
    dense=False, features_only=False, no_cache=True,
)

grover_emb = GROVEREmbedding(grover_args).to(device)
print(f"GROVER params: {sum(p.numel() for p in grover_emb.parameters()):,}")

# ---------- 4. Build CoCa ----------
from mutimodal_trans import CoCa as CoCaModel
coca = CoCaModel(
    dim=768, img_encoder=None, image_dim=1200,
    num_tokens=15, sub_graph=85,
    unimodal_depth=6, multimodal_depth=6,
    dim_head=64, heads=8,
    caption_loss_weight=1.0, contrastive_loss_weight=1.0,
).to(device)
print(f"CoCa params: {sum(p.numel() for p in coca.parameters()):,}")

# ---------- 5. Mock Graph Data ----------
from gtpro.graph_trans.data import mol2graph

# Create dummy molecule graph from SMILES strings
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'C1CCCCC1']

from gtpro.graph_trans.data.molgraph import MolCollator

batchgraph = mol2graph(smiles_list, shared_dict=[], args=grover_args).get_components()
graph_output = encode_graph(grover_emb, batchgraph, device)
graph_global = graph_output.molecule_embeddings
graph_atom = graph_output.atom_embeddings

print(f"Graph embeddings: atom={graph_atom.shape}, global={graph_global.shape}")

# ---------- 6. Run SMILES BERT ----------
token_idx = token_idx.to(device)
text_output = bert(token_idx)
print(
    f"Text embeddings: all={text_output.all_tokens.shape}, "
    f"global={text_output.global_embedding.shape}, atom={text_output.atom_tokens.shape}"
)

# ---------- 7. Run CoCa ----------
loss_criterion_atom = torch.nn.BCEWithLogitsLoss(reduction='none')

atom_labels = atom_labels.to(device)
atom_mask = atom_mask.to(device)
fg_feature = fg_feature.to(device)

losses = coca(
    graph_atom, graph_global, text_output.global_embedding, text_output.atom_tokens,
    atom_labels, fg_feature, text_output.all_tokens, atom_mask,
    loss_criterion_atom, return_loss=True
)

print(f"\n✅ Full forward pass successful! Loss = {losses.total.item():.4f}")
print("All three models work together correctly.")

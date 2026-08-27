# GTpro model boundaries and tensor contracts

Stage D1 documents the existing GTpro reproduction architecture without
claiming a new method. The supported runtime uses the bundled
`gtpro.graph_trans` GROVER implementation, the historical `K_BERT_WCL` SMILES
encoder, and the alignment/fusion `CoCa` module.

## Data flow

| Boundary | Input | Output |
|---|---|---|
| SMILES encoder | token IDs `[B, S]`, where ID 0 is `[PAD]` and position 0 is `[GLO]` | `TextEncoderOutput`: all tokens `[B, S, D_t]`, global embedding `[B, D_t]`, non-global tokens `[B, S-1, D_t]`, molecule-target logits `[B, 154]` |
| Graph encoder | GROVER graph component tuple | `GraphEncoderOutput`: atom embeddings `[1 + total_atoms, D_g]`, mean-pooled molecule embeddings `[B, D_g]`, atom scopes `[B, 2]` |
| Alignment/fusion | graph atom/global embeddings, text all/global/atom embeddings, atom/functional-group targets and masks | fused atom logits `[B, S-1, 15]` or pretraining loss components |
| Downstream prediction head | one fused or encoder-level molecule vector `[B, D]` | logits/predictions `[B, T]`, where `T` is the configured task count |

The leading GROVER atom row is its padding node; each `(start, count)` scope
points only to real atoms. Molecule embeddings are means over those real nodes.
All dict/tuple/tensor GROVER return variants are normalized centrally by
`normalize_grover_atom_output`; training loops do not branch on return type.

## Masks

- Text padding mask: Boolean `[B, S]` (expanded to `[B, S, S]` inside the SMILES
  encoder); `True` means a key position is padding and must not be attended to.
- Atom target mask: Boolean `[B, S-1]`; `True` means the non-global token is a
  real atom position and contributes to atom-label loss.
- Downstream target mask: Boolean `[B, T]`; `True` means the target is observed.
  It is derived from non-`NaN` targets and is especially important for Tox21.

`PretrainingBatch.to(device)` is the centralized move for text/target tensors.
`move_graph_components` performs the corresponding GROVER component move.
Model internals create positional/range tensors on the input tensor's device;
they do not call `.cuda()` or assume a particular accelerator.

## Source consolidation

`pretrain/seq_trans.py` is the retained historical SMILES implementation. The
former `seq_trans_fixed.py` differed only in comments and a local variable name
around the same device fix and had no imports, so D1 removed it after reference
and normalized-diff checks.

The old `pretrain/nt_xent.py` and `finetune/nt_xent.py` copies were unreferenced
and contained misspelled similarity-method bugs. They were removed after a
repository-wide reference search; `gtpro.nt_xent.NTXentLoss` is the sole retained
implementation if this loss is needed by an extension.

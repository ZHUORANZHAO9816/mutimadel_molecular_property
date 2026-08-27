"""Typed tensor boundaries shared by GTpro model and training code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


@dataclass(frozen=True)
class TextEncoderOutput:
    """SMILES encoder outputs.

    ``all_tokens`` has shape ``[batch, sequence, text_dim]`` including the
    global token. ``global_embedding`` is ``[batch, text_dim]`` and
    ``atom_tokens`` is ``[batch, sequence - 1, text_dim]``.
    """

    all_tokens: torch.Tensor
    global_embedding: torch.Tensor
    atom_tokens: torch.Tensor
    molecule_logits: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Preserve tuple-unpacking compatibility with the historical encoder.
        yield self.all_tokens
        yield self.global_embedding
        yield self.atom_tokens


@dataclass(frozen=True)
class GraphEncoderOutput:
    """Normalized GROVER output and molecule pooling metadata."""

    atom_embeddings: torch.Tensor
    molecule_embeddings: torch.Tensor
    atom_scope: torch.Tensor


@dataclass(frozen=True)
class AlignmentLosses:
    """Differentiable alignment/fusion loss components."""

    atom: torch.Tensor
    functional_group: torch.Tensor
    contrastive: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class PretrainingBatch:
    """Validated pretraining tensors before or after centralized device move."""

    token_ids: torch.Tensor
    global_labels: torch.Tensor
    atom_labels: torch.Tensor
    atom_mask: torch.Tensor
    smiles: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size, sequence_length = self.token_ids.shape
        expected_atom_length = sequence_length - 1
        if self.global_labels.shape[0] != batch_size:
            raise ValueError("global label batch dimension does not match token ids")
        if self.atom_labels.shape[:2] != (batch_size, expected_atom_length):
            raise ValueError(
                f"atom labels have shape {tuple(self.atom_labels.shape)}; expected batch/sequence "
                f"dimensions {(batch_size, expected_atom_length)}"
            )
        if self.atom_mask.shape != (batch_size, expected_atom_length):
            raise ValueError(
                f"atom mask has shape {tuple(self.atom_mask.shape)}; expected {(batch_size, expected_atom_length)}"
            )
        if len(self.smiles) != batch_size:
            raise ValueError("SMILES batch length does not match token ids")

    @property
    def padding_mask(self) -> torch.Tensor:
        """Boolean mask over text positions; True means padding/ignore."""

        return self.token_ids.eq(0)

    @property
    def atom_target_mask(self) -> torch.Tensor:
        """Boolean mask over non-global positions; True means supervise atom."""

        return self.atom_mask.bool()

    def to(self, device: torch.device | str) -> "PretrainingBatch":
        return PretrainingBatch(
            token_ids=self.token_ids.to(device=device, dtype=torch.long),
            global_labels=self.global_labels.to(device=device, dtype=torch.float32),
            atom_labels=self.atom_labels.to(device=device, dtype=torch.float32),
            atom_mask=self.atom_mask.to(device=device, dtype=torch.bool),
            smiles=self.smiles,
        )

    def __iter__(self) -> Iterator[object]:
        # Compatibility for historical callers while they migrate to fields.
        yield self.token_ids
        yield self.global_labels
        yield self.atom_labels
        yield self.atom_mask
        yield list(self.smiles)


def move_graph_components(
    components: Sequence[object], device: torch.device | str
) -> tuple[object, ...]:
    """Move every tensor component of a GROVER batch to one device."""

    return tuple(value.to(device) if torch.is_tensor(value) else value for value in components)


def normalize_grover_atom_output(output: object) -> torch.Tensor:
    """Select atom-from-atom embeddings from supported GROVER return schemas."""

    if isinstance(output, dict):
        if "atom_from_atom" not in output:
            raise ValueError("GROVER dict output is missing 'atom_from_atom'")
        atom_output = output["atom_from_atom"]
    elif isinstance(output, tuple):
        if not output:
            raise ValueError("GROVER tuple output is empty")
        atom_output = output[0]
    else:
        atom_output = output
    if not torch.is_tensor(atom_output) or atom_output.ndim != 2:
        raise TypeError("normalized GROVER atom output must be a rank-2 tensor")
    return atom_output


def encode_graph(
    model: torch.nn.Module,
    components: Sequence[object],
    device: torch.device | str,
) -> GraphEncoderOutput:
    """Run GROVER and mean-pool atom embeddings for each molecule."""

    moved = move_graph_components(components, device)
    if len(moved) < 6 or not torch.is_tensor(moved[5]):
        raise ValueError("GROVER components do not contain a tensor atom scope at index 5")
    atom_scope = moved[5]
    atom_embeddings = normalize_grover_atom_output(model(moved))
    molecule_embeddings = []
    for atom_start, atom_count in atom_scope.detach().cpu().tolist():
        if atom_count <= 0:
            raise ValueError("GROVER atom scope contains an empty molecule")
        molecule_embeddings.append(atom_embeddings.narrow(0, atom_start, atom_count).mean(dim=0))
    if not molecule_embeddings:
        raise ValueError("cannot pool an empty GROVER batch")
    return GraphEncoderOutput(
        atom_embeddings=atom_embeddings,
        molecule_embeddings=torch.stack(molecule_embeddings),
        atom_scope=atom_scope,
    )

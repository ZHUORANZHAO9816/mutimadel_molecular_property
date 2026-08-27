"""Public inference-only GTpro molecular encoder API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from rdkit import Chem
from torch import nn

from gtpro.data.pretraining import SmilesProcessingError, process_smiles
from gtpro.graph_trans.data import mol2graph
from gtpro.graph_trans.model.models import GROVEREmbedding
from gtpro.models.interfaces import encode_graph
from gtpro.utils import get_device
from pretrain.seq_trans import K_BERT_WCL


Representation = Literal["graph", "text", "joint"]
InvalidPolicy = Literal["raise", "nan"]


def _device(value: str | torch.device) -> torch.device:
    if str(value) == "auto":
        return get_device()
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if selected.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is unavailable")
    return selected


def _graph_args(config: dict[str, object], device: torch.device) -> argparse.Namespace:
    return argparse.Namespace(
        hidden_size=config["hidden_size"], backbone=config["backbone"],
        embedding_output_type=config["embedding_output_type"], dropout=config["dropout"],
        activation=config["activation"], num_mt_block=config["num_mt_block"],
        num_attn_head=config["num_attn_head"], bias=config["bias"], cuda=device.type == "cuda",
        depth=config["depth"], dense=config["dense"], undirected=config["undirected"],
        bond_drop_rate=config["bond_drop_rate"], features_only=config["features_only"],
        no_cache=config["no_cache"],
    )


class GTproEncoder(nn.Module):
    """Encode molecules with pretrained graph, text, or concatenated representations."""

    def __init__(self, text_encoder, graph_encoder, graph_args, *, device, text_dim, graph_dim):
        super().__init__()
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.graph_args = graph_args
        self.device = torch.device(device)
        self.text_dim = int(text_dim)
        self.graph_dim = int(graph_dim)
        self.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "auto",
        freeze: bool = True,
    ) -> "GTproEncoder":
        """Construct from a strict D2 pretraining checkpoint."""

        source = Path(checkpoint).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GTpro checkpoint does not exist: {source}")
        selected_device = _device(device)
        try:
            payload = torch.load(source, map_location=selected_device, weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location=selected_device)
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError("GTpro checkpoint is missing its effective config")
        if not isinstance(payload.get("models"), dict):
            raise ValueError("GTpro checkpoint is missing grouped model states")
        config = payload["config"]
        try:
            text_config = config["model"]["text"]
            graph_config = config["model"]["grover"]
            text_state = payload["models"]["text"]
            graph_state = payload["models"]["graph"]
        except KeyError as error:
            raise ValueError(f"GTpro checkpoint is missing required field: {error}") from error
        graph_args = _graph_args(graph_config, selected_device)
        text = K_BERT_WCL(
            d_model=text_config["d_model"], n_layers=text_config["n_layers"],
            vocab_size=text_config["vocab_size"], maxlen=text_config["max_length"],
            d_k=text_config["d_k"], d_v=text_config["d_v"], n_heads=text_config["n_heads"],
            d_ff=text_config["d_ff"], global_label_dim=text_config["global_label_dim"],
            atom_label_dim=text_config["atom_label_dim"],
        )
        graph = GROVEREmbedding(graph_args)
        try:
            text.load_state_dict(text_state, strict=True)
            graph.load_state_dict(graph_state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"GTpro checkpoint architecture mismatch: {error}") from error
        encoder = cls(
            text, graph, graph_args, device=selected_device,
            text_dim=text_config["d_model"], graph_dim=graph_config["hidden_size"],
        )
        if freeze:
            encoder.requires_grad_(False)
        encoder.eval()
        return encoder

    def embedding_dim(self, representation: Representation = "joint") -> int:
        if representation == "graph":
            return self.graph_dim
        if representation == "text":
            return self.text_dim
        if representation == "joint":
            return self.graph_dim + self.text_dim
        raise ValueError("representation must be graph, text, or joint")

    def encode_smiles(
        self,
        smiles: str | Sequence[str],
        *,
        representation: Representation = "joint",
        batch_size: int = 32,
        invalid_smiles: InvalidPolicy = "raise",
    ) -> torch.Tensor:
        """Return float32 embeddings, preserving list order under the `nan` policy."""

        if representation not in {"graph", "text", "joint"}:
            raise ValueError("representation must be graph, text, or joint")
        if invalid_smiles not in {"raise", "nan"}:
            raise ValueError("invalid_smiles must be raise or nan")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        single = isinstance(smiles, str)
        values = [smiles] if single else list(smiles)
        if not values:
            return torch.empty((0, self.embedding_dim(representation)), device=self.device)

        valid_indices: list[int] = []
        canonical: list[str] = []
        token_ids: list[np.ndarray] = []
        max_tokens = self.text_encoder.maxlen - 1
        for index, value in enumerate(values):
            if not isinstance(value, str):
                error = ValueError(f"SMILES at index {index} is not a string")
            else:
                molecule = Chem.MolFromSmiles(value)
                error = None if molecule is not None else ValueError(
                    f"invalid SMILES at index {index}: {value!r}"
                )
            if error is None:
                canonical_value = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                try:
                    if representation in {"text", "joint"}:
                        processed = process_smiles(canonical_value, max_tokens)
                        tokens = processed.token_ids
                        canonical_value = processed.canonical_smiles
                    else:
                        tokens = np.zeros(self.text_encoder.maxlen, dtype=np.int64)
                except SmilesProcessingError as processing_error:
                    error = ValueError(
                        f"SMILES at index {index} cannot be encoded: {processing_error.code}: "
                        f"{processing_error.detail}"
                    )
            if error is not None:
                if invalid_smiles == "raise":
                    raise error
                continue
            valid_indices.append(index)
            canonical.append(canonical_value)
            token_ids.append(tokens)

        output = torch.full(
            (len(values), self.embedding_dim(representation)), float("nan"),
            dtype=torch.float32, device=self.device,
        )
        self.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), batch_size):
                stop = start + batch_size
                batch_indices = valid_indices[start:stop]
                batch_smiles = canonical[start:stop]
                batch_tokens = torch.from_numpy(np.stack(token_ids[start:stop])).to(self.device)
                graph_embedding = text_embedding = None
                if representation in {"graph", "joint"}:
                    components = mol2graph(batch_smiles, shared_dict={}, args=self.graph_args).get_components()
                    graph_embedding = encode_graph(
                        self.graph_encoder, components, self.device
                    ).molecule_embeddings
                if representation in {"text", "joint"}:
                    text_embedding = self.text_encoder(batch_tokens).global_embedding
                if representation == "graph":
                    batch_output = graph_embedding
                elif representation == "text":
                    batch_output = text_embedding
                else:
                    batch_output = torch.cat((text_embedding, graph_embedding), dim=-1)
                output[torch.as_tensor(batch_indices, device=self.device)] = batch_output.float()
        return output[0] if single else output

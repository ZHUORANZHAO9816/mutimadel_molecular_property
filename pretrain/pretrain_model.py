"""GTpro graph-text pretraining CLI orchestration.

The reusable epoch and checkpoint implementation lives in
``gtpro.training.pretraining``; this compatibility module builds configured
models and data loaders only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gtpro.config import DEFAULT_PRETRAIN_CONFIG, ConfigError, load_pretrain_config
from gtpro.graph_trans.model.models import GROVEREmbedding
from gtpro.graph_trans.util.utils import load_checkpoint
from gtpro.run_metadata import RunRecorder
from gtpro.training.pretraining import (
    PretrainingModels,
    collate_pretraining_samples,
    load_pretraining_checkpoint,
    run_pretraining_epoch,
    save_pretraining_checkpoint,
    split_pretraining_samples,
)
from gtpro.utils import get_device
from pretrain.build_data import load_data_for_pretrain_1
from pretrain.mutimodal_trans import CoCa
from pretrain.seq_trans import K_BERT_WCL, set_random_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GTpro: Graph-Text Pretraining")
    parser.add_argument("--config", type=Path, default=DEFAULT_PRETRAIN_CONFIG)
    parser.add_argument("--data-path", "--data_path", dest="data_path", default=None)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grover-checkpoint", "--grover_checkpoint", dest="grover_checkpoint")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--caption-loss-weight", type=float, default=None)
    parser.add_argument("--contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--functional-group-loss-weight", type=float, default=None)
    parser.add_argument("--molecule-loss-weight", type=float, default=None)
    parser.add_argument("--no-cross-attention", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--d-model", "--d_model", dest="d_model", type=int, default=None)
    parser.add_argument("--n-layers", "--n_layers", dest="n_layers", type=int, default=None)
    parser.add_argument("--maxlen", type=int, default=None)
    parser.add_argument("--vocab-size", "--vocab_size", dest="vocab_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    return parser


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return get_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device=cuda was requested, but CUDA is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ConfigError("device=mps was requested, but MPS is unavailable")
    return torch.device(requested)


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data.path": args.data_path,
        "data.max_samples": args.max_samples,
        "output.root": args.save_dir,
        "output.run_id": args.run_id,
        "training.batch_size": args.batch_size,
        "training.epochs": args.epochs,
        "training.learning_rate": args.lr,
        "training.resume_from": args.resume_from,
        "training.gradient_clip_norm": args.gradient_clip_norm,
        "training.molecule_loss_weight": args.molecule_loss_weight,
        "model.alignment.caption_loss_weight": args.caption_loss_weight,
        "model.alignment.contrastive_loss_weight": args.contrastive_loss_weight,
        "model.alignment.functional_group_loss_weight": args.functional_group_loss_weight,
        "model.alignment.use_cross_attention": False if args.no_cross_attention else None,
        "model.grover.checkpoint": args.grover_checkpoint,
        "model.text.d_model": args.d_model,
        "model.text.n_layers": args.n_layers,
        "model.text.max_length": args.maxlen,
        "model.text.vocab_size": args.vocab_size,
        "seed": args.seed,
        "device": args.device,
    }


def _grover_args(config: dict[str, object], device: torch.device) -> argparse.Namespace:
    return argparse.Namespace(
        hidden_size=config["hidden_size"],
        backbone=config["backbone"],
        embedding_output_type=config["embedding_output_type"],
        dropout=config["dropout"],
        activation=config["activation"],
        num_mt_block=config["num_mt_block"],
        num_attn_head=config["num_attn_head"],
        bias=config["bias"],
        cuda=device.type == "cuda",
        depth=config["depth"],
        dense=config["dense"],
        undirected=config["undirected"],
        bond_drop_rate=config["bond_drop_rate"],
        features_only=config["features_only"],
        no_cache=config["no_cache"],
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    cli_args = build_parser().parse_args()
    config = load_pretrain_config(cli_args.config, _cli_overrides(cli_args))
    device = _select_device(config["device"])
    config["device"] = str(device)
    text_config = config["model"]["text"]
    grover_config = config["model"]["grover"]
    alignment_config = config["model"]["alignment"]
    training_config = config["training"]

    print(f"Using config: {config['config_source']}")
    print(f"Using device: {device}")
    run = RunRecorder(config)
    fixed_run_id = config.get("output", {}).get("run_id")
    environment_path = run.run_dir / "environment.json"
    if fixed_run_id is not None and environment_path.is_file():
        try:
            prior_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_environment = {}
        if (
            prior_environment.get("status") == "success"
            and (run.run_dir / "metrics.json").is_file()
            and (run.checkpoint_dir / "last.pt").is_file()
        ):
            print(f"Completed fixed run already exists; skipping: {run.run_dir}")
            return

    with run:
        print(f"Run directory: {run.run_dir}")
        set_random_seed(config["seed"])

        all_samples = load_data_for_pretrain_1(config["data"]["path"])
        max_samples = config["data"]["max_samples"]
        if max_samples is not None and max_samples < len(all_samples):
            sample_generator = torch.Generator().manual_seed(config["seed"])
            selected = torch.randperm(len(all_samples), generator=sample_generator)[:max_samples]
            all_samples = [all_samples[index] for index in selected.tolist()]
        train_samples, validation_samples = split_pretraining_samples(
            all_samples, config["data"]["validation_fraction"], config["seed"]
        )
        print(
            f"Loaded {len(all_samples)} samples: {len(train_samples)} train, "
            f"{len(validation_samples)} validation"
        )
        generator = torch.Generator().manual_seed(config["seed"])
        train_loader = DataLoader(
            train_samples,
            batch_size=training_config["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=collate_pretraining_samples,
            drop_last=False,
        )
        validation_loader = DataLoader(
            validation_samples,
            batch_size=training_config["batch_size"],
            shuffle=False,
            collate_fn=collate_pretraining_samples,
            drop_last=False,
        )

        text_model = K_BERT_WCL(
            d_model=text_config["d_model"],
            n_layers=text_config["n_layers"],
            vocab_size=text_config["vocab_size"],
            maxlen=text_config["max_length"],
            d_k=text_config["d_k"],
            d_v=text_config["d_v"],
            n_heads=text_config["n_heads"],
            d_ff=text_config["d_ff"],
            global_label_dim=text_config["global_label_dim"],
            atom_label_dim=text_config["atom_label_dim"],
        )
        graph_args = _grover_args(grover_config, device)
        graph_checkpoint = grover_config.get("checkpoint")
        if graph_checkpoint is not None:
            if not Path(graph_checkpoint).is_file():
                raise FileNotFoundError(f"GROVER checkpoint does not exist: {graph_checkpoint}")
            graph_model = load_checkpoint(graph_checkpoint, current_args=graph_args, logger=None)
            print(f"Loaded GROVER from {graph_checkpoint}")
        else:
            graph_model = GROVEREmbedding(graph_args)
            print("Initialized GROVER with random weights")
        if grover_config["freeze"]:
            graph_model.requires_grad_(False)

        alignment_model = CoCa(
            dim=text_config["d_model"],
            img_encoder=None,
            image_dim=grover_config["hidden_size"],
            num_tokens=text_config["atom_label_dim"],
            sub_graph=alignment_config["functional_group_dim"],
            unimodal_depth=alignment_config["unimodal_depth"],
            multimodal_depth=alignment_config["multimodal_depth"],
            dim_head=alignment_config["dim_head"],
            heads=alignment_config["heads"],
            caption_loss_weight=alignment_config["caption_loss_weight"],
            contrastive_loss_weight=alignment_config["contrastive_loss_weight"],
            functional_group_loss_weight=alignment_config["functional_group_loss_weight"],
            use_cross_attention=alignment_config["use_cross_attention"],
        )
        models = PretrainingModels(
            text=text_model.to(device),
            graph=graph_model.to(device),
            alignment=alignment_model.to(device),
        )
        optimizer = Adam(models.trainable_parameters(), lr=training_config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            # Keep a non-zero learning rate available for the first resumed
            # epoch even when the original smoke run contained one epoch.
            optimizer, T_max=max(2, training_config["epochs"])
        )
        atom_pos_weight = torch.tensor(
            [4.81, 1.0, 2.23, 53.49, 211.94, 0.49, 2.1, 1.13, 1.22, 1.93, 5.74, 15.42, 70.09, 61.47, 23.2],
            device=device,
        )
        atom_criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=atom_pos_weight)
        molecule_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        mixed_setting = training_config["mixed_precision"]
        mixed_precision = device.type == "cuda" and mixed_setting in {True, "auto"}
        print(f"Mixed precision enabled: {mixed_precision}")

        history: list[dict[str, object]] = []
        best_validation_loss = float("inf")
        start_epoch = 0
        resume_from = training_config.get("resume_from")
        if resume_from is not None:
            payload = load_pretraining_checkpoint(
                resume_from,
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            if payload["seed"] != config["seed"]:
                raise ValueError(
                    f"resume checkpoint seed {payload['seed']} does not match configured seed {config['seed']}"
                )
            start_epoch = int(payload["epoch"]) + 1
            best_validation_loss = float(payload["best_validation_loss"])
            history = list(payload["history"])
            print(f"Resuming after epoch {payload['epoch']} from {resume_from}")
        if start_epoch >= training_config["epochs"]:
            raise ValueError(
                f"resume checkpoint already reached epoch {start_epoch}; configured epochs={training_config['epochs']}"
            )

        for epoch in range(start_epoch, training_config["epochs"]):
            train_metrics = run_pretraining_epoch(
                models=models,
                data_loader=train_loader,
                grover_args=graph_args,
                device=device,
                atom_loss=atom_criterion,
                molecule_loss=molecule_criterion,
                optimizer=optimizer,
                training=True,
                mixed_precision=mixed_precision,
                gradient_clip_norm=training_config["gradient_clip_norm"],
                molecule_loss_weight=training_config["molecule_loss_weight"],
                progress=print,
            )
            validation_metrics = run_pretraining_epoch(
                models=models,
                data_loader=validation_loader,
                grover_args=graph_args,
                device=device,
                atom_loss=atom_criterion,
                molecule_loss=molecule_criterion,
                optimizer=None,
                training=False,
                mixed_precision=False,
                molecule_loss_weight=training_config["molecule_loss_weight"],
            )
            scheduler.step()
            epoch_record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
            }
            history.append(epoch_record)
            improved = validation_metrics.total_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_metrics.total_loss
            checkpoint_arguments = {
                "epoch": epoch,
                "seed": config["seed"],
                "config": config,
                "models": models,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "best_validation_loss": best_validation_loss,
                "history": history,
            }
            save_pretraining_checkpoint(run.checkpoint_dir / "last.pt", **checkpoint_arguments)
            if improved:
                save_pretraining_checkpoint(run.checkpoint_dir / "best.pt", **checkpoint_arguments)
            _atomic_json(
                run.run_dir / "metrics.json",
                {"best_validation_loss": best_validation_loss, "epochs": history},
            )
            print(
                f"epoch {epoch + 1}/{training_config['epochs']}: "
                f"train_total={train_metrics.total_loss:.6f}, "
                f"validation_total={validation_metrics.total_loss:.6f}, "
                f"contrastive={validation_metrics.contrastive_loss:.6f}, "
                f"atom={validation_metrics.atom_loss:.6f}, "
                f"functional_group={validation_metrics.functional_group_loss:.6f}, "
                f"molecule={validation_metrics.molecule_loss:.6f}"
            )

    print(f"Pretraining complete! Run saved to {run.run_dir}")


if __name__ == "__main__":
    print("DEPRECATED compatibility entry: use scripts/run_pretraining.py instead.", file=sys.stderr)
    main()

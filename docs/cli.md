# Command-line entry points

Stage B4 establishes stable command locations. All commands below can be
invoked with an absolute script path from any working directory. Relative data,
output, and config paths are resolved against the repository root rather than
the caller's current directory.

## Data preparation

```bash
python scripts/prepare_pretrain_data.py --help
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output-dir artifacts/pretrain \
  --num-shards 4
```

The command uses the importable `gtpro.data.pretraining` pipeline. It performs
deterministic canonicalization, validates text/graph atom alignment, records
row-level failures and canonical duplicates, writes shards atomically, and
skips existing shards only after input/config/checksum verification. It writes
`data_report.json` and `data_report.md` beside the shards. See
`docs/preprocessing.md` for policies and the five-array shard schema.

## Pretraining

```bash
python scripts/run_pretraining.py --help
python scripts/run_pretraining.py --config configs/pretrain_smoke.yaml
```

The command uses validated configuration, reusable train/validation epoch
logic, strict resumable checkpoints, and per-run metadata recording. The smoke
config is execution-only; the formal config requires stage-C prepared data.

## Fine-tuning

```bash
python scripts/run_finetuning.py --help
python scripts/run_finetuning.py --config configs/finetune_bace_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_tox21_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_lipophilicity_smoke.yaml
```

The unified D3 runner handles binary classification, masked multilabel
classification, and regression. It supports multiple seeds, frozen/partial/full
encoder training, early stopping, class weighting, strict pretraining
checkpoint audits, and complete per-seed artifacts. See `docs/finetuning.md`.
The checked smoke configs use random small encoders and are not benchmark runs.

## Forward smoke example

```bash
python examples/smoke_test.py
```

This runs the existing joint BERT/GROVER/CoCa forward check and propagates its
exit status. It is not a training or reproduction result.

## Compatibility entry points

`download_pretrain_data.py` and `pretrain/pretrain_model.py` remain executable
for existing commands. Direct use prints a deprecation notice pointing to the
stable scripts above. They will not be removed until the new paths have remained
compatible through the relevant later stages.

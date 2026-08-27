# Configuration and run records

Stage B3 makes YAML the primary source of pretraining parameters. Checked
configs are validated before model or data construction, command-line values are
explicit overrides, and every attempted training run receives a unique record.

## Checked configurations

- `configs/pretrain_smoke.yaml`: small CPU execution check using the two
  test-only shards. It uses 64-dimensional text and graph encoders, one text
  layer, shallow alignment blocks, batch size 4, and one epoch.
- `configs/pretrain.yaml`: production-size architecture and default training
  schedule. Its data prefix matches the C1 output
  `artifacts/pretrain/CHEMBL_smiles_*.npy`; the configuration is not itself a
  claim that formal pretraining has been run.

Both configs include:

| Section | Contents |
| --- | --- |
| top level | random seed and requested device |
| `data` | preprocessed shard path prefix |
| `model.text` | dimension, layers, vocabulary, sequence length, attention dimensions, heads, and label dimensions |
| `model.grover` | checkpoint, hidden size, graph architecture, attention/depth options, and freeze policy |
| `model.alignment` | CoCa depths, heads, dimensions, objective sizes, and loss weights |
| `training` | batch size, epochs, learning rate, clipping, mixed precision, molecule-loss weight, and resume checkpoint |
| `output` | root run directory |

Relative data, checkpoint, and output paths are resolved against the repository
root. The copied effective config contains resolved paths and the actual selected
device, so CLI overrides are not lost.

## Canonical runner

The B4 canonical entry accepts a config:

```bash
python scripts/run_pretraining.py --config configs/pretrain_smoke.yaml
```

The former flags remain as explicit config overrides. For example:

```bash
python scripts/run_pretraining.py \
  --config configs/pretrain_smoke.yaml \
  --seed 52 \
  --batch-size 2
```

Invalid positive values, an unsupported device name, or an unavailable requested
accelerator fail before model construction. A configured but missing GROVER
checkpoint also fails explicitly rather than silently switching to random
initialization.

The old `pretrain/pretrain_model.py` path remains as a deprecated compatibility
entry and points users to the canonical script.

## Run layout

Every invocation creates:

```text
runs/<experiment>/<UTC timestamp>_seed<seed>/
  config.yaml
  environment.json
  checkpoints/
```

`environment.json` is written with `status: running` before training and
atomically finalized with `success` or `failed`. It records:

- seed, selected device, command, and source config;
- UTC start/end timestamps and measured duration;
- Python, PyTorch, platform, machine, CUDA, and MPS summary;
- Git commit and dirty state;
- error type/message for a failed run.

The repository currently has no commit, so the verified run correctly records
`git.commit` as `null` and `git.dirty` as `true`. Once commits exist, the same
recorder captures the actual revision.

Run contents are ignored by Git. `runs/README.md` and `artifacts/README.md` are
the only tracked files in their respective convention directories.

## Standardized pretraining loop

The reusable epoch, collation, train/validation split, and checkpoint functions
live in `gtpro.training.pretraining`; the CLI is orchestration only. Losses are
reported as per-sample means for contrastive, atom, functional-group, molecule,
and total objectives. CPU runs reject mixed precision; CUDA runs can enable it
explicitly or with `auto`.

Each epoch atomically writes `last.pt`, updates `best.pt` on validation
improvement, and writes `metrics.json`. A checkpoint contains all three model
state groups, optimizer, scheduler, completed epoch, seed, effective config,
best validation loss, and history. Resume loading is strict and rejects an
incomplete or incompatible checkpoint.

## B3 verification run

The 2026-08-26 smoke invocation loaded 32 samples, completed eight batches, and
wrote a successful run record plus 324 KiB BERT and 897 KiB CoCa smoke
checkpoints. The printed accumulated loss is diagnostic smoke output and is not
a formal result.

The first invocation exposed hard-coded 1200/768/15 dimensions in the CoCa
projection and reshape path. Its manifest was correctly finalized as failed.
Those values were replaced with constructor dimensions, and the next invocation
passed. The ignored failed record is intentionally not presented as a successful
experiment.

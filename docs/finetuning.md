# Downstream fine-tuning

`scripts/run_finetuning.py` is the single config-driven runner for BACE binary
classification, Tox21 multilabel classification, and Lipophilicity regression.
It uses the graph and SMILES encoders jointly and adds a task-width prediction
head.

## Encoder modes and checkpoints

- `frozen` trains only the prediction head.
- `partial` trains the final configured SMILES Transformer layers and text
  projection/classifier modules plus the head; the graph encoder remains
  frozen.
- `full` trains both encoders and the head.

`model.pretrained_checkpoint` accepts a D2 pretraining checkpoint. The text and
graph architecture in the fine-tuning config must match that checkpoint.
`strict_checkpoint: true` rejects any missing, unexpected, or shape-mismatched
key. Non-strict loading still records every mismatch and always rejects a
mismatch fraction above `max_checkpoint_mismatch_fraction`; a large mismatch is
never silently ignored.

The checked smoke configs intentionally set the checkpoint to `null` and use
random initialization to test the runner quickly on CPU. Their metrics are not
pretrained-model or benchmark results.

## Tasks, losses, and selection

Classification uses masked binary cross entropy. Tox21 missing labels are
excluded from both loss and metrics. `class_imbalance` may be `none`, `auto`, or
a positive weight per target; resolved weights and warnings are written to
`metrics.json`. Regression uses mean squared error.

Early stopping selects ROC-AUC for BACE, macro ROC-AUC across valid Tox21 tasks,
and RMSE for Lipophilicity. If a small classification validation fold contains
only one class and the selection metric is unavailable, selection explicitly
falls back to validation loss and records that fact.

## Outputs

Each seed creates its own immutable run directory containing:

```text
config.yaml
environment.json
checkpoints/best.pt
metrics.json
predictions.csv
```

`predictions.csv` contains validation and test rows, source row identifiers,
canonical SMILES, labels (blank when missing), and one prediction per target.
The runner accepts multiple seeds with `--seeds 42 52 62`; every seed receives
an independent split, initialization, checkpoint, and run record.

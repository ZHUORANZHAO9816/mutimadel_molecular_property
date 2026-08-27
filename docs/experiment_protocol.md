# Frozen empirical reproduction protocol

This protocol distinguishes the original paper-scale architecture from the
CPU-feasible empirical reproduction shipped with this repository. No smoke
metric and no compact-model metric is presented as the paper's reported result.

## Data and preprocessing

- Pretraining input: the audited 12,008-row local ChEMBL-derived CSV with SHA-256
  `f06d01bc80fda05488881c8328465caaaec613987f16d0731c1eae442b12d560`.
- Canonical RDKit isomeric SMILES, first canonical occurrence retained,
  sequences above 200 tokens filtered, leaving 11,988 samples. Shard checksums
  are frozen in `artifacts/pretrain/data_report.json` and summarized in
  `docs/datasets.md`.
- Downstream files and schemas are those audited in
  `docs/downstream_datasets.md`. Invalid SMILES are dropped and reported;
  text/joint representations additionally filter canonical SMILES above 200
  tokens before splitting and record the count. Missing Tox21 labels remain
  missing and are masked in loss and metrics. Classical comparisons use this
  same text-eligible cohort so their split indices remain identical.

## Models and seeds

The empirical reproduction uses `configs/pretrain_reproduction_compact.yaml`:
a 64-dimensional, one-layer text encoder and 64-dimensional GROVER encoder,
trained for one CPU epoch on all 11,988 retained molecules. This is explicitly a
compact engineering reproduction, not the original 768/1200-dimensional paper
configuration. The frozen formal seeds are **42, 52, and 62**.

Every downstream model and classical baseline uses the identical split indices
for a given dataset, split method, and seed. Model parameter counts and
representation definitions are recorded to avoid treating differently sized
models as directly capacity-matched.

## Splits and metrics

- Random split: canonical-SMILES-grouped seeded 80/10/10 split, measuring the
  common IID setting without canonical duplicate leakage.
- Scaffold split: seeded Bemis-Murcko-grouped 80/10/10 split, measuring harder
  structural generalization. Random and scaffold values are never pooled.
- BACE: ROC-AUC and PR-AUC.
- Tox21: per-task ROC-AUC/PR-AUC over observed labels, macro averages, and valid
  task count.
- Lipophilicity: RMSE, MAE, and R2.

Single-class folds produce a documented unavailable (`null`) metric. They are
excluded from macro means rather than replaced with an invented value.

## Model selection and test isolation

The validation set alone selects checkpoints. BACE uses validation ROC-AUC,
Tox21 uses macro validation ROC-AUC, and Lipophilicity uses validation RMSE.
When a classification validation metric is unavailable, the runner records an
explicit fallback to validation loss. Early stopping uses patience two and zero
minimum delta for the compact protocol. The test split is evaluated exactly
once after restoring the best validation checkpoint and is never used for
hyperparameter selection.

## Environment and timing

Each run writes the selected CPU/GPU device, platform, architecture, Python and
PyTorch versions, CUDA/MPS availability, start/end timestamps, and wall-clock
duration to `environment.json`. GPU runs must additionally record the GPU model
and memory externally if PyTorch cannot expose them. The observed compact
pretraining run used CPU, Python 3.12.2, PyTorch 2.13.0, processed 10,789 train
and 1,199 validation molecules, and completed in approximately 105 seconds.

## Artifacts and aggregation

Formal run roots use `runs/reproduction/<dataset>_<split>/<model>/<seed>/` and
contain resolved configuration, environment, metrics, predictions, and a best
checkpoint. `scripts/summarize_results.py` is the sole supported aggregator;
README tables are generated from run records, never typed by hand. Lightweight
metrics/config/environment copies under `results/run_records` preserve public
traceability while model weights remain ignored and must be released through a
versioned external artifact host.

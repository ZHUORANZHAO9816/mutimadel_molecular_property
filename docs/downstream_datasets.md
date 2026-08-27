# Downstream dataset audit and interface

This C3 audit describes the files present locally on 2026-08-26. Raw benchmark
files remain ignored and are not redistributed by this repository. Their
presence and successful loading are engineering checks, not evidence that a
model result has been reproduced. Source licenses and redistribution terms
still require review before any public data release.

## Actual inventory

| Directory | Files found | Audit result |
|---|---|---|
| `bace` | `BACE_README`, `raw/bace.csv`, `raw/bace.npy` | Single-molecule classification and regression labels are available. |
| `biosnap` | `raw/all.csv`, `raw/sup_train_val.csv`, `raw/sup_test.csv` | Drug-pair binary classification; no dataset README is present. |
| `lipophilicity` | `Lipo_README`, `raw/Lipophilicity.csv`, `raw/lipo.npy` | Single-molecule regression is available. |
| `sider` | `SIDER_README`, `raw/sider.csv`, `raw/sider.npy` | Single-molecule 27-label classification is available. |
| `tox21` | `TOX21_README`, `raw/tox21.csv`, `raw/tox21.npy` | Single-molecule 12-label classification with missing labels is available. |
| `toxcast` | `TOXCAST_README` only | README claims a processed ToxCast collection, but no raw CSV or generated array is present. |
| `twosides` | `raw/Drug_META_DDIE.db`, `raw/reliable_negatives.csv`, `raw/twosides_interactions.csv` | Drug-pair binary interaction data plus a drug metadata database; no dataset README is present. |

`.DS_Store` files were also present under `data/downstream` and `bace`; they are
machine-local files already excluded by `.gitignore` and are not dataset assets.

## Supported single-molecule schemas

The supported interface reads the source CSV, canonicalizes each valid molecule
with RDKit, preserves source row numbers, converts targets to `float32`, and
normalizes missing labels to `NaN` with a Boolean validity mask.

| Loader name | Logical CSV rows | SMILES column | Target columns | Task type | Native missing-label encoding |
|---|---:|---|---|---|---|
| `bace` | 1,513 | `mol` | `Class` | Binary classification | None observed |
| `bace_regression` | 1,513 | `mol` | `pIC50` | Regression | None observed |
| `tox21` | 7,831 | `smiles` | 12 `NR-*`/`SR-*` assays | Multilabel classification | Empty CSV fields, normalized to `NaN` |
| `lipophilicity` | 4,200 | `smiles` | `exp` | Regression | None observed |
| `sider` | 1,427 | `smiles` | 27 MedDRA system-organ classes | Multilabel classification | None observed |

All five loader variants parsed every SMILES in their CSV during C3. No
canonical duplicate was found in these four underlying CSV files. The BACE CSV
contains 595 columns, mostly precomputed descriptors; the unified interface
deliberately selects only structure, identifier, and configured target columns.
BACE's `Model` column contains 203 `Train`, 45 `Valid`, and 1,265 `Test` rows,
but C3's generic random/scaffold APIs create new explicit splits and do not
silently reuse this source-specific partition.

Tox21 contains 16,026 missing target cells:

| Target | Missing | Target | Missing |
|---|---:|---|---:|
| `NR-AR` | 566 | `NR-AR-LBD` | 1,073 |
| `NR-AhR` | 1,282 | `NR-Aromatase` | 2,010 |
| `NR-ER` | 1,638 | `NR-ER-LBD` | 876 |
| `NR-PPAR-gamma` | 1,381 | `SR-ARE` | 1,999 |
| `SR-ATAD5` | 759 | `SR-HSE` | 1,364 |
| `SR-MMP` | 2,021 | `SR-p53` | 1,057 |

Classification targets are validated as 0, 1, or missing. Regression values are
numeric or missing. Invalid/empty structures can either be recorded and dropped
(`invalid_smiles="drop"`, the default) or cause an explicit error
(`invalid_smiles="raise"`).

## README and generated-file discrepancies

- `BACE_README` says 1,522 compounds, while the actual CSV has 1,513 rows. It
  names a lowercase `class` field, while the actual column is `Class`.
- `SIDER_README` claims 1,427 drugs and 27 labels, matching the CSV.
- `TOX21_README` describes approximately 8,000 compounds and 12 targets; the
  actual CSV has 7,831 rows and those 12 target columns.
- `Lipo_README` correctly identifies `smiles` and regression target `exp`; the
  CSV has 4,200 rows.
- `TOXCAST_README` describes data that are absent locally, so ToxCast is not
  loadable and is not presented as available.
- BioSNAP and TWOSIDES have no local README/provenance note. They are pairwise
  tasks and are outside the single-molecule loader introduced in C3.

The four legacy `.npy` files have object-array headers and therefore require
pickle-enabled loading. They were not treated as authoritative or unpickled for
this audit. Their safe header metadata are:

| File | Shape | CSV rows |
|---|---:|---:|
| `bace.npy` | `(4, 1513)` | 1,513 |
| `lipo.npy` | `(4, 4199)` | 4,200 |
| `sider.npy` | `(4, 1384)` | 1,427 |
| `tox21.npy` | `(4, 7825)` | 7,831 |

Except for BACE, the legacy array sample counts differ from their CSVs and no
generation metadata or checksum accompanies them. New workflows must load the
audited CSV interface rather than silently trusting those arrays.

BioSNAP has 83,040 logical rows in `all.csv`, exactly 66,432 train/validation
rows plus 16,608 test rows. Its columns are two drug IDs, two SMILES, and one
binary label (plus a saved index column). TWOSIDES' two interaction CSV files
are headerless even though their first records resemble headers: they contain
45,026 negative and 48,022 positive records. `Drug_META_DDIE.db` contains one
`drug` table with 3,535 rows and fields `id`, `name`, `interaction`, `smile`,
`target`, `enzyme`, `carrier`, and `transporter`.

## Unified Python interface

```python
from gtpro.data.downstream import load_downstream_dataset

dataset = load_downstream_dataset("tox21")
record = dataset[0]

record.canonical_smiles
record.targets       # shape: (12,)
record.target_mask   # False where the CSV label was missing
```

Registered names are `bace`, `bace_regression`, `tox21`, `lipophilicity`, and
`sider`. `DatasetSpec` and `load_downstream_csv` allow the same normalization to
be applied to an explicitly described CSV without adding an implicit schema.

## Random and scaffold splits

```python
from gtpro.data.downstream import split_dataset

random_split = split_dataset(dataset, method="random", seed=10)
scaffold_split = split_dataset(dataset, method="scaffold", seed=10)
```

Both return dataset-position arrays for train, validation, and test. Every
position appears exactly once. Random splitting shuffles canonical-SMILES groups
rather than individual rows, so equivalent SMILES cannot leak between splits.
Scaffold splitting groups RDKit Bemis–Murcko scaffolds, also keeping canonical
duplicates together. Group assignment is seeded and deterministic; requested
fractions are approximate because an indivisible scaffold group is never split.
No label stratification is claimed.

Using seed 10 and fractions 0.8/0.1/0.1 produced:

| Dataset | Random train/validation/test | Scaffold train/validation/test |
|---|---:|---:|
| BACE | 1,209 / 152 / 152 | 1,209 / 152 / 152 |
| Tox21 | 6,265 / 783 / 783 | 5,650 / 1,474 / 707 |
| Lipophilicity | 3,360 / 420 / 420 | 3,360 / 420 / 420 |

The larger Tox21 scaffold deviation is a consequence of keeping scaffold groups
indivisible. The C3 verification confirmed zero repeated indices, zero
canonical-SMILES overlap for both methods, and zero scaffold overlap for the
scaffold method.

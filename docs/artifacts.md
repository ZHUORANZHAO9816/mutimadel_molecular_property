# Data and artifact policy

This repository keeps source code, documentation, the paper's versioned source
CSVs, and explicitly documented test fixtures in Git. Generated shards, run
directories, logs, and model weights stay outside Git because they are
reproducible, machine-specific, or potentially misleading without complete
training provenance.

## What Git excludes

| Path or pattern | Reason | How to obtain or regenerate |
| --- | --- | --- |
| `checkpoints/*.pth`, `*.pt`, `*.ckpt` | Model weights are large binary outputs. The current files are smoke products, not pretrained releases. | Run the relevant training command locally. Formal weights must be downloaded from a versioned release location once one exists. |
| `runs/` | Per-run configuration, environment details, metrics, predictions, logs, and checkpoints are generated outputs. | Created by the standardized runners planned in stages B–D. |
| `artifacts/` | Processed data, reports, and other reproducible generated files may be large. | Created by data-preparation and result-summary scripts. |
| `data/pretrain_data/*.npy` | Generated ChEMBL shards are reproducible build outputs. | Use `scripts/prepare_pretrain_data.py` as described below. |
| Unlisted files under `data/downstream/**/raw/` | Historical arrays and unrelated datasets are excluded. | The five paper benchmark CSVs are explicit exceptions with provenance in `data/paper_datasets.json`. |
| caches, `.DS_Store`, logs, virtual environments | Machine-local or reproducible files. | Recreated automatically by the relevant tool. |
| `.claude/`, `.env*` | Personal permissions, local paths, and possible credentials must not be published. | Create locally; never copy into releases. `.env.example` may be tracked if it contains placeholders only. |

The ignore rules do not delete local files. Existing datasets and checkpoints
remain available to local smoke and baseline commands.

## Allowed smoke fixtures

The only generated pretraining binaries deliberately eligible for Git are:

```text
data/pretrain_data/gtpro_smoke_1.npy
data/pretrain_data/gtpro_smoke_2.npy
```

Both filenames and `data/pretrain_data/README.md` identify them as test-only
data. Each file contains 16 samples, for 32 total. Results produced from these
files must be labelled as smoke results and must not appear as formal model or
paper-reproduction metrics.

Future fixtures should live under `tests/fixtures/`, contain no more than a few
dozen molecules, document their origin and license, and include obvious edge
cases rather than a benchmark-quality sample. An explicit `.gitignore`
exception must be reviewed before adding another binary fixture.

## Current local regeneration paths

### Pretraining input and shards

The canonical CLI can download a bounded ChEMBL SMILES collection and generate
validated `.npy` shards:

```bash
python scripts/prepare_pretrain_data.py \
  --max-molecules 12008 \
  --output-dir artifacts/pretrain \
  --num-shards 4
```

It can also preprocess an existing CSV containing a `smiles` column:

```bash
python scripts/prepare_pretrain_data.py \
  --input /path/to/CHEMBL_smiles.csv \
  --output-dir artifacts/pretrain \
  --num-shards 4
```

The output directory contains machine-readable and human-readable reports with
the input hash, policies, row-level failures, duplicate records, shard counts,
and checksums. Generated data and reports remain ignored artifacts. The root
`download_pretrain_data.py` command remains only as a deprecated compatibility
wrapper.

### Smoke checkpoints

The current one-epoch baseline command is:

```bash
PYTHONPATH=. python pretrain/pretrain_model.py \
  --epochs 1 \
  --batch_size 2 \
  --data_path ./data/pretrain_data/gtpro_smoke
```

Outputs under `checkpoints/` are ignored. This command is scheduled for actual
baseline verification in A3; its presence here does not claim that the run has
passed in the current repository state.

### Downstream datasets

The source CSVs for BBBP, BACE, SIDER, ClinTox, and Tox21 are tracked under
`data/downstream/<dataset>/raw/`. Download URLs, hashes, and row counts live in
`data/paper_datasets.json`; source terms are summarized in
`data/DATA_LICENSES.md`. Historical arrays and datasets outside the five paper
benchmarks remain local-only.

## Formal checkpoint publication

No formal GTpro checkpoint download is currently advertised by this optimized
repository. The existing `model_bert0.pth` and `model_coca0.pth` are historical
smoke outputs and must not be renamed or presented as pretrained releases.

When a checkpoint has a traceable full training run, publish it through a
versioned GitHub Release, Hugging Face repository, or Zenodo record rather than
Git. A release entry must include:

- model and code version;
- immutable download URL and SHA-256 checksum;
- training configuration, data version, split policy, and seeds;
- software/hardware environment and training duration;
- metrics linked to machine-readable run outputs;
- license and upstream attribution;
- checkpoint loading compatibility notes.

Until those fields exist, users must train locally and all generated weights
remain artifacts rather than official pretrained models.

## Sensitive-file audit

The 2026-08-26 A2 audit searched text files for credential names, private-key
headers, passwords, tokens, and personal absolute paths.

- No embedded credential or private key was found.
- `.claude/settings.local.json` is a personal tool-permission file and is now
  ignored in full.
- `pretrain/build_pretrain.py` formerly contained an author-machine absolute
  path. C1 replaced that executable block with the importable
  `gtpro.data.pretraining` pipeline; the helpers remain only for provenance.
- Documentation mentions paths and terms such as “token” in explanatory text;
  those are not secrets.

Before any public release, run a dedicated history-aware secret scanner in
addition to the source-tree checks above.

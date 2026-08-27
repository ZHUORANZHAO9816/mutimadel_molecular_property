# GTpro current-state audit

Audit date: 2026-08-26

This document records the repository state before structural optimization. It is
an engineering audit, not evidence that the paper's reported results have been
reproduced. No uncertain or duplicate file was removed during this audit.

## Repository and runtime

- The supplied directory was not a Git repository. An empty Git repository was
  initialized on 2026-08-26; all project files are currently untracked.
- Observed runtime: Python 3.12.2, PyTorch 2.13.0, CPU
  (`torch.cuda.is_available() == False`).
- No `AGENTS.md`, packaging metadata, dependency lock file, test configuration,
  or CI workflow is present.
- `data/pretrain_data/CHEMBL_smiles.csv` has 12,008 data rows. All 12,008
  `smiles` values are non-empty and string-unique. This is not yet a canonical
  SMILES duplicate check.
- The two smoke shards are `gtpro_smoke_1.npy` and `gtpro_smoke_2.npy`.
- The two checkpoint files are smoke artifacts, not validated pretrained
  weights. They occupy about 726 MiB in total.

## Current architecture and data flow

1. `download_pretrain_data.py` optionally downloads ChEMBL SMILES and performs
   token, atom-label, and MACCS preprocessing into `.npy` shards.
2. `pretrain/build_data.py` loads numbered `.npy` shards and turns the five
   parallel arrays into per-molecule samples.
3. `pretrain/pretrain_model.py` is the current pretraining CLI. It constructs a
   SMILES Transformer (`K_BERT_WCL`), GROVER graph encoder, and CoCa-style
   graph-text alignment/fusion model, then calls the epoch loop in
   `pretrain/seq_trans.py`.
4. `pretrain/seq_trans.py` contains the SMILES encoder, batch collation,
   functional-group label generation, checkpoint helper, and one-epoch training
   loop. These responsibilities are not yet separated.
5. `pretrain/mutimodal_trans.py` contains the multimodal CoCa implementation.
6. `finetune/` contains helper modules only. There is no fine-tuning runner and
   no root `finetune.py`.
7. `gtpro/` is importable only while the repository root is on `sys.path`; it
   currently exposes utilities and a vendored GROVER-like tree, but no public
   encoder API.

## Import/source relationship

### GROVER

The active entry points import `grover.*`, not `gtpro.graph_trans.*`:

- `pretrain/pretrain_model.py` imports GROVER utilities and
  `grover.model.models.GROVEREmbedding`.
- `pretrain/seq_trans.py`, `pretrain/seq_trans_fixed.py`,
  `finetune/seq_trans.py`, and `test_forward.py` import `grover.data` or
  `grover.model`.
- Files under `gtpro/graph_trans/` also use absolute `grover.*` imports
  internally rather than package-relative imports.

In the audited environment, `grover` and `GROVEREmbedding` resolve to
`/opt/anaconda3/lib/python3.12/site-packages/grover`, while
`gtpro.graph_trans` resolves to this repository. Therefore the external package
is the current implementation used at runtime and the repository contains a
second, ambiguous implementation. Stage B2 must select and test one source.

### `pretrain` and `finetune`

- There are no package-style imports such as `from pretrain ...` or
  `from finetune ...`.
- `pretrain/pretrain_model.py` imports sibling modules as top-level
  `seq_trans`, `build_data`, and `mutimodal_trans` after inserting only the
  `pretrain/` directory into `sys.path`.
- From the repository root,
  `python pretrain/pretrain_model.py --help` currently fails at
  `from gtpro.utils import get_device` unless the repository root is supplied
  via `PYTHONPATH` or the project is installed.
- `test_forward.py` inserts both the repository root and `pretrain/` into
  `sys.path`, so it does not prove that the package is installable.

## Entry points and path/device assumptions

| File | Role | Current issue |
| --- | --- | --- |
| `download_pretrain_data.py` | ChEMBL download and preprocessing CLI | Uses working-directory-relative defaults; combines network and transformation logic. |
| `pretrain/pretrain_model.py` | Active pretraining CLI | Requires repository-root import help; parameters are mostly CLI defaults; output defaults to `./checkpoints`. |
| `pretrain/build_pretrain.py` | Legacy preprocessing helpers | Its former author-machine executable block is retired; the canonical pipeline is `gtpro.data.pretraining`. |
| `test_forward.py` | Forward smoke script | Mutates `sys.path` and uses the external `grover` installation. |
| `gtpro/graph_trans/data/dist_sampler.py` | Module-local sampler demo | Has an `if __name__ == "__main__"` block but is not a project training entry. |

README currently documents `python pretrain_model.py` and `python finetune.py`;
neither command exists at the documented root location.

Device-related assumptions found by source search:

- `K_BERT_WCL` and `Embedding` default their `device` argument to `"cuda"`,
  although the active CLI passes the detected device.
- Training helpers call `torch.cuda.empty_cache()` unconditionally and seed CUDA
  conditionally.
- vendored GROVER utilities and model paths contain direct `.cuda()` calls;
  their reachability under the future unified import source must be tested.
- No user-specific absolute path or explicit `PYTHONPATH` string occurs in
  Python source. The legacy author-machine block in `pretrain/build_pretrain.py`
  was removed after this initial audit.

## Duplicate and overlapping implementations

### `seq_trans.py` versus `seq_trans_fixed.py`

`pretrain/seq_trans.py` is the real implementation: it is imported by both
`pretrain/pretrain_model.py` and `test_forward.py`. Nothing imports
`seq_trans_fixed.py`.

After ignoring CRLF versus LF line endings, the only code difference is the
temporary variable used while moving graph tensors to the selected device
(`_dev` versus `device_`) plus reworded comments. There is no functional fix
exclusive to `seq_trans_fixed.py`. Stage D1 can therefore archive or remove the
duplicate after tests exist; this audit deliberately leaves both files intact.

### Pretraining data modules

- `pretrain/build_data.py` is not a builder. It only loads already generated,
  consecutively numbered `.npy` shards. `load_data_for_pretrain_2` partially
  overlaps its first loader and contains a duplicate variable initialization.
- `pretrain/build_pretrain.py` is the legacy transformation implementation. It
  defines tokenization, global features, atom labels, and shard writing, but its
  CLI is tied to an absolute author-machine path and fixed multiprocessing.
- `download_pretrain_data.py` reimplements the relevant tokenization, MACCS,
  atom-label, and shard-writing logic and adds ChEMBL API download, deduplication,
  length filtering, and the current no-CIP chirality fallback. It is the only
  currently usable data CLI, but duplicates the legacy implementation and
  silently drops preprocessing failures.

Stage C1 should extract one importable preprocessing implementation and keep
the CLIs thin; `build_data.py` should remain a loader or be renamed accordingly.

## Complete file inventory by category

The inventory excludes `.git/`, which was newly initialized during this audit.

### Source and executable scripts

```text
download_pretrain_data.py
test_forward.py
finetune/nt_xent.py
finetune/seq_trans.py
gtpro/__init__.py
gtpro/nt_xent.py
gtpro/utils.py
gtpro/models/__init__.py
gtpro/graph_trans/__init__.py
gtpro/graph_trans/data/__init__.py
gtpro/graph_trans/data/dist_sampler.py
gtpro/graph_trans/data/groverdataset.py
gtpro/graph_trans/data/moldataset.py
gtpro/graph_trans/data/molfeaturegenerator.py
gtpro/graph_trans/data/molgraph.py
gtpro/graph_trans/data/scaler.py
gtpro/graph_trans/data/task_labels.py
gtpro/graph_trans/data/torchvocab.py
gtpro/graph_trans/model/__init__.py
gtpro/graph_trans/model/layers.py
gtpro/graph_trans/model/models.py
gtpro/graph_trans/util/__init__.py
gtpro/graph_trans/util/metrics.py
gtpro/graph_trans/util/multi_gpu_wrapper.py
gtpro/graph_trans/util/nn_utils.py
gtpro/graph_trans/util/parsing.py
gtpro/graph_trans/util/scheduler.py
gtpro/graph_trans/util/utils.py
pretrain/build_data.py
pretrain/build_pretrain.py
pretrain/mutimodal_trans.py
pretrain/nt_xent.py
pretrain/pretrain_model.py
pretrain/seq_trans.py
pretrain/seq_trans_fixed.py
```

### Documentation and metadata

```text
CODEX_TODO.md
README.md
data/downstream/bace/BACE_README
data/downstream/lipophilicity/Lipo_README
data/downstream/sider/SIDER_README
data/downstream/tox21/TOX21_README
data/downstream/toxcast/TOXCAST_README
```

`docs/current_state.md` and `docs/progress.md` are engineering documents added
by A1 and are not part of the pre-audit inventory.

### Data

```text
data/pretrain_data/CHEMBL_smiles.csv
data/pretrain_data/gtpro_smoke_1.npy
data/pretrain_data/gtpro_smoke_2.npy
data/downstream/bace/raw/bace.csv
data/downstream/bace/raw/bace.npy
data/downstream/biosnap/raw/all.csv
data/downstream/biosnap/raw/sup_test.csv
data/downstream/biosnap/raw/sup_train_val.csv
data/downstream/lipophilicity/raw/Lipophilicity.csv
data/downstream/lipophilicity/raw/lipo.npy
data/downstream/sider/raw/sider.csv
data/downstream/sider/raw/sider.npy
data/downstream/tox21/raw/tox21.csv
data/downstream/tox21/raw/tox21.npy
data/downstream/twosides/raw/Drug_META_DDIE.db
data/downstream/twosides/raw/reliable_negatives.csv
data/downstream/twosides/raw/twosides_interactions.csv
```

The ToxCast directory currently contains only a README and no raw dataset.

### Checkpoints and other generated artifacts

```text
checkpoints/model_bert0.pth   # about 166 MiB, smoke artifact
checkpoints/model_coca0.pth   # about 560 MiB, smoke artifact
```

No run directory or training log file was found.

### Caches, system files, and private local configuration

```text
.DS_Store
data/.DS_Store
data/downstream/.DS_Store
data/downstream/bace/.DS_Store
gtpro/.DS_Store
pretrain/.DS_Store
__pycache__/download_pretrain_data.cpython-312.pyc
gtpro/__pycache__/__init__.cpython-312.pyc
gtpro/__pycache__/utils.cpython-312.pyc
pretrain/__pycache__/build_data.cpython-312.pyc
pretrain/__pycache__/build_pretrain.cpython-312.pyc
pretrain/__pycache__/mutimodal_trans.cpython-312.pyc
pretrain/__pycache__/seq_trans.cpython-312.pyc
.claude/settings.local.json
```

These are candidates for exclusion in A2. They were recorded but not deleted.

## Known compatibility changes already present

- `download_pretrain_data.py` supplies a fixed-length chirality label fallback
  for atoms without `_CIPCode`.
- `pretrain/seq_trans.py` accepts GROVER dictionary, tuple, or direct tensor
  output and moves graph tensors to the selected device before the forward pass.
- The training helper removes a leading atom-label position when labels include
  the global token and text atom embeddings do not.
- `gtpro/utils.py` selects CUDA, then Apple MPS, then CPU.

These observations describe source changes; A3 must independently run and
record the forward and one-epoch smoke baselines before refactoring.

## Immediate risks and next actions

1. A2 must prevent the 726 MiB smoke checkpoints, data copies, caches, system
   files, and local settings from entering the first commit.
2. A3 must record actual baseline success or failure before implementation
   changes.
3. B1/B2 must make imports installation-based and choose exactly one GROVER
   source.
4. C1 must consolidate preprocessing and replace silent failure handling with an
   auditable report.
5. D/E must establish tests before duplicate training modules are removed.

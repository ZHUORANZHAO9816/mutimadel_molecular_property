# Optimization progress

This log records engineering changes and actual verification results. Smoke
runs are never treated as paper reproduction results.

## 2026-08-28 — README aligned to paper datasets, training, and results

- Refined the homepage presentation after reviewing the official MolCLR,
  GraphMVP, GROVER, and Uni-Mol repositories. Added prominent authorship and
  venue metadata, compact paper links, contribution highlights, benchmark task
  counts, a proper ROC–AUC definition, and evidence-focused ablation analysis.
- Reframed the results around the evaluation question: what the metric means,
  what each benchmark measures, why multi-task datasets are not directly
  comparable by raw score, and what the consistent full-model improvement
  demonstrates.
- Replaced the compact reproduction and baseline tables in the README with the
  five final GTpro ROC-AUC values printed in paper Figure 3: BBBP 0.962, BACE
  0.881, SIDER 0.684, ClinTox 0.997, and Tox21 0.821.
- Added a paper-facing dataset section covering ChEMBL pretraining and the five
  MoleculeNet classification benchmarks, with links to the original sources.
- Added the original graph-text pretraining flow and paper-scale hyperparameter
  summary based on the maintained configuration and historical training code.
- Removed compact-result framing, resource-comparison tables, and checkpoint or
  release-availability limitations from the project homepage.

### Verification

- README paper results were checked against the labels printed in the retained
  high-resolution Figure 3 asset.
- Repository-local links and release hygiene were rechecked with
  `scripts/audit_release.py`.

## 2026-08-28 — README paper narrative and figures

- Expanded the README's paper narrative with Figure 1 (alignment motivation),
  Figure 3 (published pretraining/contrastive ablation), and Figure 6
  (atom/token attention analysis), in addition to the existing Figure 2
  architecture overview.
- Kept redundant Figure 4 and the training-loss-only Figure 5 out of the README
  to preserve a focused project homepage.
- Clearly separated publisher-reported paper results from the compact local
  reproduction tables, and added cautious interpretation rather than implying
  that an attention visualization is standalone quantitative evidence.
- Added per-file provenance, resolution, copyright, and license-scope metadata
  for all paper figures.

## 2026-08-27 — Paper architecture alignment

- Reframed README as the author-maintained GTpro paper implementation while
  retaining a precise distinction between the paper-scale method and the
  compact project-measured reproduction.
- Replaced the first-screen project sketch with the paper's high-resolution
  Figure 2 (3060 x 2240), which shows the dual encoders, contrastive alignment,
  cross-attention, Transformer block, and APP/FPG/GTM objectives.
- Added the figure number, DOI, publisher copyright, author-request attribution,
  asset provenance, and license-scope notice. The project-authored SVG remains
  in the architecture guide as a complementary implementation map.
- Visually inspected the downloaded publisher asset and reran the publication
  audit. It passed with 479 candidates, 116 trace records, and all 16 README
  local links valid.

## 2026-08-27 — Stage D2: standardized pretraining loop

### Changes

- Separated reusable collation, deterministic train/validation splitting,
  single-epoch execution, metrics, and checkpoint I/O into
  `gtpro.training.pretraining`.
- Added per-sample mean reporting for contrastive, atom, functional-group,
  molecule, and total losses; added configurable clipping and CUDA-only optional
  mixed precision.
- Added atomic `best.pt`/`last.pt` checkpoints with strict resume of the three
  model groups, optimizer, scheduler, epoch, seed, config, best value, and
  history.
- Replaced list-to-tensor construction with NumPy stacking and covered empty
  loaders, malformed batches, invalid molecules, and a final partial batch.

### Verification

Passed on 2026-08-27:

```bash
pytest -q
python scripts/run_pretraining.py --config configs/pretrain_smoke.yaml
python scripts/run_pretraining.py --config configs/pretrain_smoke.yaml \
  --epochs 2 --resume-from <successful-last.pt>
```

The full suite passed with 35 tests. The final recorded smoke run used 24 train
and 8 validation samples on CPU, completed six train batches, and wrote its
successful record to `runs/pretrain_smoke/20260827T035337.142749Z_seed10`.
Validation component losses were recorded in `metrics.json`; these are execution
diagnostics, not benchmark results. A separate two-epoch invocation resumed
strictly after epoch 0 and completed the next epoch.

## 2026-08-27 — Stage D3: unified downstream fine-tuning

### Changes

- Replaced the placeholder with one config-driven graph/text fine-tuning runner
  for BACE binary classification, Tox21 multilabel classification, and
  Lipophilicity regression.
- Added frozen, partial-text-unfreeze, and full encoder modes; masked missing
  multilabel targets; configurable and recorded positive-class weighting;
  validation-based early stopping; and independent multi-seed runs.
- Added strict D2 encoder-checkpoint loading with complete missing, unexpected,
  and shape-mismatch reporting. Even non-strict mode rejects mismatch fractions
  above the configured safety threshold.
- Each seed now writes `checkpoints/best.pt`, `predictions.csv`, `metrics.json`,
  the resolved `config.yaml`, and environment metadata.

### Verification

Passed on CPU on 2026-08-27:

```bash
python scripts/run_finetuning.py --config configs/finetune_bace_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_tox21_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_lipophilicity_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_bace_smoke.yaml \
  --seeds 1 2 --max-samples 16 --epochs 1
```

The three primary smoke records are respectively
`runs/finetune_bace_smoke/20260827T040616.902325Z_seed10`,
`runs/finetune_tox21_smoke/20260827T040631.299381Z_seed10`, and
`runs/finetune_lipophilicity_smoke/20260827T040636.088063Z_seed10`. Each used
44/10/10 sampled train/validation/test molecules and contains all required
artifacts. The checked configs deliberately use random small encoders, so these
diagnostic values are not benchmark or pretrained-model results.

## 2026-08-27 — Stage D4: downstream metrics

### Changes and verification

- Added ROC-AUC and PR-AUC for binary classification; per-task and macro metrics
  plus valid-task counts for masked multilabel classification; and RMSE, MAE,
  and R2 for regression.
- Unavailable metrics use JSON-safe `null` values and explanatory warnings.
  The Tox21 smoke test exercised six single-class test tasks without crashing,
  while macro scores used the six valid tasks only.
- Added known-answer, missing-label, constant-target, single-class, and shape
  validation tests.

Final D-stage verification passed:

```bash
pytest -q
python examples/smoke_test.py
python test_forward.py
```

The suite reported 47 passed tests. Both full graph/text/alignment forward
checks also completed on CPU. D1 through D4 are complete.

## 2026-08-27 — Stage E: CPU test system and CI definition

- Added bounded subprocess integration tests for one pretraining epoch and
  fixture-backed BACE and Lipophilicity fine-tuning. All write to pytest
  temporary directories and require neither full ChEMBL nor downstream raw
  files.
- Added `examples/ci_smoke_test.py`, which performs a small graph/text/alignment
  forward, backward, clipping, and optimizer step on CPU.
- Added `.github/workflows/ci.yml` with Python 3.11, editable
  `.[train,test]` installation, a static import check, `pytest -q`, and the CPU
  model smoke. It has no GPU or data-download step.
- Moved PyYAML into required runtime dependencies because both canonical runners
  import it. No CI badge was added because no remote GitHub run exists yet.

Local workflow-equivalent verification passed with 50 tests and the CI smoke
reported two processed samples. Stage E is complete locally; remote workflow
status remains an external publication check rather than an inferred success.

## 2026-08-27 — Stage F: compact empirical reproduction

- Froze `docs/experiment_protocol.md` with the data checksum, 11,988 retained
  pretraining molecules, seeds 42/52/62, leakage-safe random/scaffold splits,
  metrics, test isolation, early stopping, environment recording, and the
  explicit compact-vs-paper distinction.
- Trained a real 64-dimensional compact GTpro checkpoint for one epoch over all
  11,988 retained pretraining molecules. It used 10,789 train and 1,199
  validation samples and completed in about 105 seconds on CPU.
- Completed 18 GTpro downstream runs: three datasets × two split methods ×
  three seeds. Completed 90 classical/representation baseline runs covering
  Morgan linear/RF, graph descriptors, SMILES character features, and joint
  graph+SMILES without alignment. Identical dataset/split/seed cohorts were
  used, and total/trainable parameter definitions are reported.
- Completed eight seed-42 BACE ablation screens: full, no contrastive, no
  cross-attention, no atom objective, no functional-group objective, no
  molecule objective, graph-only, and text-only. Full and the key representation
  comparisons also have three-seed evidence in the main matrix.
- Generated 36 reproduction aggregates and eight ablation aggregates, including
  mean, sample standard deviation, valid-run count, parameter counts, Markdown
  tables, and two original SVG plots. The 116 contributing metrics/config/
  environment record sets were copied to `results/run_records` for traceability.

The first Tox21 attempt exposed one overlength downstream molecule. Its failed
local record was retained, and the frozen behavior now filters text/joint
molecules over 200 tokens before splitting while recording the count. No SIDER
extension was attempted because the frozen core matrix consumed the allocated
CPU run and ToxCast data are absent. All reported numbers are labeled
project-measured compact empirical results, never original-paper values.

## 2026-08-27 — Stage G: public molecular encoder API

- Exported `GTproEncoder` directly from `gtpro`, with strict reconstruction from
  a D2 checkpoint's recorded text/GROVER config and weights.
- Added string/list inputs, graph/text/joint float32 representations, bounded
  batching, CPU/CUDA/MPS selection, default model freezing, and explicit
  `raise` or order-preserving `nan` invalid-input policies.
- Added a public-only example and single/batch/invalid tests. The acceptance
  command encoded CCO with the compact checkpoint on CPU and returned joint
  shape `(128,)`, dtype `torch.float32`.

The API contract and compact/example dimensions are documented in
`docs/encoder_api.md`. Stage G is complete.

## 2026-08-27 — Stage H: documentation and publication metadata

- Rewrote README with the accurate paper/project identity, an original first-
  screen architecture SVG, implemented capabilities, CPU quick start, data and
  training commands, generated reproduction/ablation tables, resource scope,
  structure, limitations, provenance, citation, acknowledgements, and license
  scope.
- Added architecture and reproducibility guides, `CITATION.cff`, contribution
  policy, issue forms, changelog, MIT text for new contributions, and a detailed
  third-party/license scope audit.
- Verified the paper metadata against PubMed: Zhao et al., Journal of Molecular
  Graphics and Modelling 132 (2024) 108843, DOI
  `10.1016/j.jmgm.2024.108843`, PMID 39173218. Corrected the paper-listed source
  URL and retained the GROVER upstream URL/notice.

One H4 item is deliberately unresolved: the URL printed by the paper used the
repository name `multimodal_molecular_property` and returned 404. A later audit
found the accessible original repository under the near-identical name
`mutimadel_molecular_property`, but it and the supplied GTpro-derived snapshot
have no visible license. The root MIT license can cover new engineering
contributions and is compatible with bundled MIT GROVER, but it cannot
relicense the original files. `LICENSE_SCOPE.md`
records this legal blocker rather than inventing permission.

## 2026-08-27 — Stage I: local release verification and audit

- Created a fresh virtual environment under `/private/tmp`, installed the
  editable `.[test]` target, and corrected the declared base dependencies so
  import-time `einops` and `requests` requirements are explicit. The virtual
  environment reused the host's already installed scientific binary packages
  through `--system-site-packages`; build isolation and the editable wheel were
  nevertheless newly created and verified.
- Imported `gtpro` from outside the repository without `PYTHONPATH`; package
  metadata and `gtpro.__version__` both reported `0.1.0.dev0`.
- Ran the full suite in that environment: 52 tests passed. Ran the full CPU
  graph/text/alignment forward, the small forward/backward/optimizer CI smoke,
  one pretraining epoch, and BACE/Tox21/Lipophilicity smoke fine-tuning.
- Executed every README shell command. Full preprocessing re-audited all 12,008
  rows and checksum-verified/skipped the four existing shards. The compact
  pretraining command is now safely idempotent for its fixed completed run;
  reproduction, baseline, and ablation commands verified their completed
  records, and result summarization regenerated the public tables.
- Added `scripts/audit_release.py`. It passed across 477 publication candidates
  after the final documentation addition, with 116 complete trace records and
  16 valid README-local links; no candidate exceeded 50 MiB or matched the
  prohibited artifact/private-path checks. Copied result records now replace
  the workstation root with `${PROJECT_ROOT}`.
- Verified external README targets: the article/DOI, PubMed, GROVER, and
  corrected original repository links resolve. The paper printed `multimodal`
  while the accessible repository uses `mutimadel`; metadata now consistently
  uses the accessible URL. Added `docs/release_audit.md` and draft notes for
  v0.1.0, v0.5.0, and v1.0.0.
- Retired the obsolete absolute-path executable block in
  `pretrain/build_pretrain.py` and removed three legacy invalid-escape warnings.

Final local verification commands included:

```bash
python -m pip install -e ".[test]"
python -c "import gtpro"
pytest -q
python examples/smoke_test.py
python examples/ci_smoke_test.py
python scripts/run_pretraining.py --config configs/pretrain_smoke.yaml
python scripts/run_finetuning.py --config configs/finetune_bace_smoke.yaml
python scripts/audit_release.py
```

Three external release gates remain explicitly blocked rather than represented
as complete: the supplied GTpro-derived code needs license confirmation, no
remote GitHub Actions run exists, and no Git tags/releases or formal checkpoint
hosting target exists. Local release engineering and release-note drafts are
complete; publishing requires repository/hosting authority and the upstream
license resolution.

## 2026-08-26 — Stage A1: repository-state audit

### Changes

- Confirmed that the supplied directory was not a Git repository and initialized
  an empty repository with `git init`.
- Enumerated and categorized source, data, checkpoints, caches, documentation,
  system files, and local configuration in `docs/current_state.md`.
- Audited Python imports, runnable entry points, absolute paths, CUDA/device
  assumptions, duplicate `seq_trans` files, and overlapping preprocessing
  modules.
- Recorded known compatibility fixes without changing runtime code or deleting
  uncertain files.

### Baseline facts verified during the audit

- Python: 3.12.2
- PyTorch: 2.13.0
- Accelerator status: CUDA unavailable; observed execution target is CPU
- ChEMBL CSV: 12,008 rows, 12,008 non-empty string-unique SMILES
- Smoke data files present: `gtpro_smoke_1.npy`, `gtpro_smoke_2.npy`
- Smoke checkpoint files present: `model_bert0.pth`, `model_coca0.pth`
- Active external GROVER path:
  `/opt/anaconda3/lib/python3.12/site-packages/grover`
- Local GROVER-like path:
  `gtpro/graph_trans`

Forward and training baselines were intentionally not run here because they are
the separate A3 task.

### Verification

Passed on 2026-08-26:

```bash
git status --short
rg -n "from grover|import grover|gtpro\.graph_trans|cuda\(|\.cuda\(|PYTHONPATH|/Users/" . --glob '*.py'
```

`git status --short` completed successfully and showed the expected untracked
pre-first-commit files. The source search completed successfully and its GROVER
and direct-CUDA hits are categorized in `docs/current_state.md`; it found no
user-specific `/Users/...` path in Python source. A1 is complete.

### Remaining issues

- Repository hygiene is not yet configured, so all files other than `.git/` are
  untracked and large smoke checkpoints remain visible to Git.
- GROVER source selection is ambiguous.
- The active pretraining CLI requires `PYTHONPATH=.` (or equivalent repository
  root injection) in the current uninstalled layout.
- No supported fine-tuning entry point exists.

## 2026-08-26 — Stage A2: ignore and large-artifact policy

### Changes

- Added `.gitignore` rules for macOS/editor files, Python and test caches,
  virtual environments, logs, temporary files, local assistant/environment
  configuration, packaging output, run directories, generated artifacts, model
  checkpoints, full pretraining data, and downstream raw dataset copies.
- Kept the two small `gtpro_smoke_*.npy` files eligible for Git and added
  `data/pretrain_data/README.md`, which labels them as 16-sample test shards and
  prohibits presenting their outputs as formal results.
- Added `docs/artifacts.md` with the local regeneration commands, data and model
  storage policy, current smoke-checkpoint status, and requirements for a future
  formal checkpoint release.
- Kept ignored local data and checkpoints on disk; A2 did not delete user files.

### Sensitive-file audit

- Searched non-binary files for credential names, private-key headers,
  passwords, tokens, and personal absolute paths.
- No embedded credential or private key was found. A BACE README match was the
  chemistry term “secretase,” not a secret.
- `.claude/settings.local.json` contains machine-local tool permissions and is
  excluded by ignoring `.claude/`.
- `pretrain/build_pretrain.py` originally contained an author-machine absolute
  path. It was not a credential and was removed when C1 replaced the legacy
  executable block with the canonical preprocessing pipeline.

### Verification

Passed on 2026-08-26:

```bash
git status --short
find . -type f -size +50M -print
```

`git status --short` no longer lists caches, `.DS_Store`, `.claude/`, checkpoint
weights, the full ChEMBL CSV, or downstream raw data. The `find` inspection
still reports the two local smoke checkpoint files (about 560 MiB and 166 MiB)
and internal `.git` object files; these remain on disk but are not public
repository candidates.

Additional policy checks passed:

```bash
git check-ignore -v .claude/settings.local.json checkpoints/model_bert0.pth \
  checkpoints/model_coca0.pth data/pretrain_data/CHEMBL_smiles.csv \
  data/downstream/bace/raw/bace.csv
git ls-files --others --exclude-standard
```

All sensitive/large targets matched an explicit ignore rule. Among 49 current
public candidate files, none exceeds 50 MiB. The two smoke fixtures and their
README remain eligible for Git. A2 is complete.

### Remaining issues

- A3 still needs to run and record the forward and one-epoch smoke baselines.
- The repository has not made an initial commit; this checklist does not
  authorize committing on the user's behalf.
- Formal data provenance and deterministic regeneration remain C1–C3 work.

## 2026-08-26 — Stage A3: pre-refactor execution baseline

### Changes

- Ran the existing joint BERT/GROVER/CoCa forward smoke script without changing
  model code.
- Ran one pretraining epoch over the two retained smoke shards: 32 samples,
  batch size 2, 16 batches, CPU.
- Added `docs/baseline_run.md` with the environment, commands, concise output,
  timings, checkpoint classification, warning, and limitations.
- Kept the full logs out of README and made no paper-reproduction or model
  quality claim.

### Verification

Passed on 2026-08-26:

```bash
python test_forward.py
PYTHONPATH=. python pretrain/pretrain_model.py \
  --epochs 1 \
  --batch_size 2 \
  --data_path ./data/pretrain_data/gtpro_smoke
```

Forward smoke result: exit code 0; full joint forward completed on CPU. Measured
wall time was 5.17 seconds. Shape and parameter summaries are recorded in
`docs/baseline_run.md`.

Pretraining smoke result: exit code 0; all 16 batches and the single epoch
completed. The training loop reported about 9 seconds and measured total process
wall time was 14.25 seconds. It printed accumulated smoke loss `42.4049`, which
is explicitly not classified as a formal metric.

The run emitted the known slow tensor-construction warning at
`pretrain/seq_trans.py:298`. It updated the ignored smoke-only BERT and CoCa
checkpoint files. A3 is complete.

### Remaining issues

- Stage A is complete. B1 is the next task and must establish installable modern
  package metadata before GROVER source unification.
- The slow list-of-NumPy-arrays tensor construction remains assigned to D2.
- Smoke checkpoints remain non-release artifacts and are excluded from Git.

## 2026-08-26 — Stage B1: modern package configuration

### Changes

- Added `pyproject.toml` using setuptools with distribution name
  `gtpro-molecular`, pre-release version `0.1.0.dev0`, Python support
  `>=3.9,<3.13`, base runtime dependencies, and `train`, `legacy-pyg`, `test`,
  and `dev` optional groups.
- Corrected `gtpro.__version__` from the premature `1.0.0` value to
  `0.1.0.dev0`, matching installed package metadata.
- Added a pinned `environment.yml` based on the successful 2026-08-26 CPU
  baseline environment.
- Added `docs/environment.md` and updated the README installation section to
  distinguish the current validated environment from the historical paper
  environment and to avoid forcing an obsolete CUDA wheel index.
- Audited GROVER provenance and license. The authoritative Tencent AI Lab
  upstream repository uses the MIT license and includes Chemprop attribution.
  The installed `grover 0.1.0` came from `file:///tmp/grover` and omitted its
  license from distribution metadata, so it was documented but deliberately not
  encoded as a stable dependency before B2 chooses one implementation source.

### Verification

Passed on 2026-08-26:

```bash
python -m pip install -e .
python -c "import gtpro; print(gtpro.__file__)"
python -m build --no-isolation
python test_forward.py
```

The first sandboxed install attempt could not download isolated build
dependencies because network access was restricted. Re-running the same
required command with approved network access succeeded and installed
`gtpro-molecular-0.1.0.dev0` as an editable package.

The import check was run from `/private/tmp`, outside the repository working
directory. It resolved to this repository's `gtpro/__init__.py`; both
`gtpro.__version__` and installed distribution metadata reported
`0.1.0.dev0`. Source and wheel builds completed successfully, and the existing
joint forward smoke test continued to pass on CPU. B1 is complete.

### Remaining issues

- B2 must convert the repository's GROVER tree to package-relative imports,
  preserve the upstream MIT notice, and prove that runtime no longer resolves
  to an accidental site-packages copy.
- The locked environment reflects the current CPU baseline. GPU-specific
  PyTorch installation remains platform-dependent by design.
- Overall project licensing is still H4 work; the completed audit covers the
  GROVER upstream component only.

## 2026-08-26 — Stage B2: unified GROVER source

### Changes

- Selected the bundled `gtpro.graph_trans` implementation as the only GROVER
  source used by project Python code.
- Converted imports across the bundled data/model/utility tree, pretraining,
  fine-tuning helpers, and forward smoke script to package-relative or explicit
  `gtpro.graph_trans` imports.
- Restored the upstream structured dict output for
  `embedding_output_type="both"`; existing GTpro code explicitly selects
  `atom_from_atom`.
- Added `tests/test_grover_source.py` to verify the loaded source path, reject
  external GROVER imports, load direct and official-prefixed checkpoint schemas,
  and reject wholly unrelated state dictionaries.
- Updated the checkpoint loader for both prefix schemas and modern PyTorch's
  restricted `weights_only` default, explicitly allowlisting only
  `argparse.Namespace` metadata on supported versions.
- Added a packaged GROVER `NOTICE`, detailed source/compatibility documentation,
  and updated README/environment dependency guidance.

### Compatibility audit

The production-size bundled and previously active external encoders both had
107,143,232 parameters and 106 state-dict keys. Their key sets and tensor shapes
matched exactly: zero missing keys, zero extra keys, and zero shape mismatches.

### Verification

Passed on 2026-08-26:

```bash
python -c "import inspect; from gtpro.graph_trans.model.models import GROVEREmbedding; print(inspect.getfile(GROVEREmbedding))"
pytest -q
python test_forward.py
PYTHONPATH=. python pretrain/pretrain_model.py \
  --epochs 1 --batch_size 2 \
  --data_path ./data/pretrain_data/gtpro_smoke
```

The source resolved to the repository's
`gtpro/graph_trans/model/models.py`. All five tests passed. A full package walk
imported 19 bundled GROVER submodules and loaded zero top-level external
`grover` modules. The joint forward test passed with the baseline shapes.

The post-switch 32-sample CPU pretraining smoke run completed all 16 batches and
printed the same seeded accumulated diagnostic loss (`42.4049`) as A3. This is
recorded only as an execution compatibility check, not a model-quality result.

The first checkpoint test run exposed the new PyTorch safe-loading default and
failed before parameter comparison. After adding the restricted Namespace
allowlist, the test suite passed; the failure was not hidden or counted as a
successful run. B2 is complete.

### Remaining issues

- Three pre-existing invalid-escape `SyntaxWarning` messages remain in legacy
  docstrings/tokenizer regexes; they are not GROVER source-selection failures.
- Importing `multi_gpu_wrapper` reports that Horovod is unavailable, so this
  environment supports only the validated single-process CPU path.
- B3 is next and must establish versioned YAML configs plus run metadata/output
  directory conventions.

## 2026-08-26 — Stage B3: unified configs and run records

### Changes

- Added validated `configs/pretrain_smoke.yaml` and `configs/pretrain.yaml` with
  seed, device, data path, complete text/GROVER/alignment model dimensions,
  batch size, epochs, learning rate, and output root.
- Added `gtpro.config` for config loading, explicit CLI overrides, positive value
  validation, accelerator validation, and repository-relative path resolution.
- Added `gtpro.run_metadata.RunRecorder`, which creates a unique UTC/seed run
  directory, atomically copies the resolved config, and records environment,
  command, seed, selected device, Git commit/dirty state, timestamps, duration,
  status, and failure details.
- Converted the existing pretraining entry to use YAML as its primary parameter
  source while retaining old flags as explicit overrides until B4 replaces the
  entry point.
- Parameterized previously hard-coded CoCa projection and reshape dimensions so
  checked model sizes actually control graph/text alignment construction.
- Added tracked `runs/README.md` and `artifacts/README.md` policies while keeping
  all generated contents ignored.
- Added `docs/configuration.md`, config/run tests, and accurate README smoke
  instructions.

### Verification

Passed on 2026-08-26:

```bash
pytest -q
python test_forward.py
python pretrain/pretrain_model.py --config configs/pretrain_smoke.yaml
```

All 11 tests passed. They validate both checked configs, CLI-style overrides,
invalid-value rejection, success/failure metadata finalization, GROVER source,
and checkpoint compatibility. The production-size joint forward smoke remained
successful after CoCa dimension parameterization.

The successful config-driven training run used CPU, seed 10, 32 smoke samples,
batch size 4, 64-dimensional text/GROVER encoders, and eight batches. Its run
record contained resolved `config.yaml`, finalized `environment.json`, and small
ignored BERT/CoCa smoke checkpoints. The manifest recorded `status=success`,
start/end timestamps, a 0.952128-second recorded duration, and the current
repository state (`commit=null`, `dirty=true`). The duration and printed loss
are smoke diagnostics, not formal performance results.

### Failure found and fixed

The first real config-driven smoke run failed because CoCa still hard-coded
1200-dimensional graph normalization and 768/15-dimensional projections and
reshapes. Its run manifest correctly recorded `status=failed`, timestamps, and
the runtime error. The implementation was changed to use constructor-provided
`image_dim`, `dim`, and `num_tokens`; the subsequent run passed. The failed run
was not reclassified or reported as successful.

### Remaining issues

- The known slow tensor construction warning in `pretrain/seq_trans.py` remains
  assigned to D2.
- The formal config points to stage-C generated data and is not a claim that a
  full experiment has been run.
- B4 is next and must wrap the underlying functionality in canonical scripts and
  provide a fine-tuning placeholder pending stage D.

## 2026-08-26 — Stage B4: canonical command-line entry points

### Changes

- Added `scripts/prepare_pretrain_data.py` as the stable data command. It accepts
  an existing CSV or bounded ChEMBL download, configurable output directory and
  shard count, and currently delegates to the legacy implementation pending C1.
- Added `scripts/run_pretraining.py` as the stable wrapper around the validated
  B3 config/metadata runner. Modern hyphenated flags are supported while legacy
  underscore aliases remain compatible.
- Added `scripts/run_finetuning.py` as an honest D3 placeholder. Help succeeds;
  execution exits explicitly instead of pretending fine-tuning exists.
- Added `examples/smoke_test.py`, which runs the joint forward check and
  propagates its exit status.
- Defined repository-root resolution for relative config, data, and output
  paths, independent of the caller's working directory.
- Kept `download_pretrain_data.py` and `pretrain/pretrain_model.py` executable
  and added direct-use deprecation messages pointing to the stable commands.
- Added CLI tests and `docs/cli.md`; corrected README pretraining, data, and
  fine-tuning command claims.

### Verification

Passed on 2026-08-26:

```bash
python scripts/prepare_pretrain_data.py --help
python scripts/run_pretraining.py --help
python scripts/run_finetuning.py --help
python examples/smoke_test.py
pytest -q
```

All 17 tests passed. The three help commands exited successfully, and the smoke
example completed the production-size BERT/GROVER/CoCa forward path on CPU.

The smoke example and canonical pretraining command were also invoked by
absolute script path from `/private/tmp`. The pretraining command correctly
resolved `configs/pretrain_smoke.yaml`, the two data shards, and the output root
against this repository; it completed eight batches and finalized a successful
run manifest. The seeded accumulated loss matched the prior B3 small-config run
and remains diagnostic only.

Tests also confirm that nonexistent relative config/data paths are reported as
repository-root paths, the fine-tuning placeholder fails explicitly, and old
entry points emit their compatibility/deprecation messages. B4 and stage B are
complete.

### Remaining issues

- Stage C1 is next. The canonical data command is stable, but its delegated
  preprocessing implementation still lacks deterministic canonicalization,
  failure reports, atomic shard writes, and recovery.
- Fine-tuning remains intentionally unavailable until D3.
- Three legacy invalid-escape warnings and the slow tensor construction warning
  remain documented; neither was hidden by B4.

## 2026-08-26 — Stage C1: reproducible pretraining-data pipeline

### Changes

- Moved the supported ChEMBL download and preprocessing implementation into the
  importable `gtpro.data.pretraining` module. The canonical script now handles
  argument/path orchestration, and the historical root command is a deprecated
  compatibility wrapper over the same implementation.
- Added deterministic RDKit canonical isomeric SMILES processing. The first
  input row for each canonical molecule is retained by default, while every
  duplicate is reported with its row, raw value, canonical value, and first
  occurrence. The downloader also preserves first-seen ordering.
- Preserved the 15 historical atom-label targets and made CIP handling explicit
  for R, S, and no assigned CIP code. Added tests for all three cases.
- Added row-level failure diagnostics with CSV line number, raw value, stable
  reason code, and detail. Canonical strings over 200 tokens are explicitly
  filtered rather than truncated, preventing invalid SMILES fragments and
  graph/text misalignment.
- Validated token, 154-bit selected MACCS, `(200, 15)` atom-label, and atom-mask
  shapes for every sample. Atom token count and atom-mask sum must both match
  the RDKit graph node count; a test also compares this count with the bundled
  GROVER `MolGraph` representation.
- Added configurable balanced shard generation while preserving the legacy
  five-entry `.npy` schema and numbered-prefix naming expected by the current
  loader. Updated the formal pretraining config to the generated
  `artifacts/pretrain/CHEMBL_smiles` prefix.
- Shards and reports now use same-directory temporary files plus atomic
  replacement. Resume skips a shard only after the input hash, content config,
  expected sample count, stored checksum, and current file checksum agree.
- Added `data_report.json`, `data_report.md`, preprocessing documentation, and
  `tests/test_preprocessing.py`. Generated ChEMBL data and reports remain
  ignored artifacts.

### Verification

Passed on 2026-08-26:

```bash
pytest -q
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output artifacts/pretrain \
  --num-shards 4
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output artifacts/pretrain \
  --num-shards 4
```

All 21 tests passed. The suite covers R/S/no-CIP labels, invalid and empty
values, canonical duplicate removal, long-SMILES filtering, tensor/graph node
alignment, configurable shards, atomic-file cleanup, checksums, and verified
resume behavior. Three previously documented legacy invalid-escape warnings
remain and are unrelated to the C1 implementation.

The required full-data command processed all 12,008 CSV rows in about 18
seconds. It retained 11,988 samples, reported 20 `smiles_too_long` rows, found
zero canonical duplicates, and atomically wrote four equal 2,997-sample shards.
This is a data-generation result, not a training or paper-reproduction result.
Every generated shard was loaded and checked for the `(201,)`, `(154,)`,
`(200, 15)`, and `(200,)` per-sample layouts; the existing pretraining loader
also loaded all 11,988 samples successfully.

Repeating the exact acceptance command reprocessed and validated the input but
skipped all four existing shards after checksum verification. No temporary
files remained. The resulting report records these SHA-256 values:

- `CHEMBL_smiles_1.npy`: `738c8db6d04a2fe57b67c425769f589d90967e591c1f2f1f5a4705daea5202dc`
- `CHEMBL_smiles_2.npy`: `f0ccd475e36c1fdfed81df1a78df3c47cc1739f757762d3046fb4b030ac00035`
- `CHEMBL_smiles_3.npy`: `7fe93c41982ec82111f638e06b2a562af0a380f5b49ed7c71689d5f3dbcfbc5e`
- `CHEMBL_smiles_4.npy`: `e09a4901095513f610a90a36720b34c738b79742a262f56f37cdb2e57ae34f6b`

C1 is complete.

### Remaining issues

- C2 is next and must extend the generated reports with atom-count, SMILES
  length, and functional-group label distributions, then copy only stable
  summary statistics into `docs/datasets.md`.
- `pretrain/build_pretrain.py` remains as documented legacy provenance and still
  emits one of the known invalid-escape warnings. Supported workflows no longer
  import it; deletion or relocation still requires the later reference audit.
- A verified shard is skipped rather than incrementally streaming past its
  source rows; input preprocessing is intentionally repeated so row-level
  validation and reports cannot become stale.

## 2026-08-26 — Stage C2: pretraining dataset reports

### Changes

- Upgraded the generated report schema to version 2 and made count semantics
  explicit: total/empty rows, RDKit parse successes, canonicalized rows, full
  processing successes, failures, unique canonical SMILES, canonical
  duplicates, and final retained samples are reported separately.
- Corrected the reporting classification for valid but overlong molecules.
  They now count as parse and canonicalization successes while remaining
  explicit processing failures under the configured filter policy.
- Added basic distributions over final retained samples for RDKit atom count,
  canonical SMILES token length, canonical SMILES character length, and active
  molecule-level labels. Each includes count, min/max, mean, population standard
  deviation, and 5th/25th/50th/75th/95th percentiles.
- Added positive count and prevalence for every one of GTpro's 154 selected
  MACCS structural-key targets, plus the number of labels observed at least
  once. The report names these as structural-key targets rather than assigning
  unsupported human-readable functional-group descriptions.
- Expanded `data_report.md` with a readable distribution table, top-15 target
  prevalence table, and the existing shard/checksum table. The JSON retains all
  per-row failures, duplicate records, and all 154 per-target statistics.
- Added `docs/datasets.md` containing only the stable audit summary, source input
  hash, processing policy, aggregate distributions, top target prevalence, and
  shard checksums. No source CSV, generated shard, or row-level ChEMBL data was
  added to the tracked documentation.
- Extended preprocessing tests to verify report/file equality, distribution
  schema and values, functional-group dimensions, shard sample totals, and
  complete reporting for an empty input.

### Verification

Passed on 2026-08-26:

```bash
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output artifacts/pretrain \
  --num-shards 4
python -c "import json; d=json.load(open('artifacts/pretrain/data_report.json')); print(d)"
pytest -q
```

The full preparation command completed successfully and checksum-verified all
four existing shards. The exact checklist JSON command loaded and printed the
schema-version-2 report successfully. All 22 tests passed; three known legacy
invalid-escape warnings remain unchanged.

The audited local input has 12,008 rows, no empty values, 12,008 RDKit parse and
canonicalization successes, no canonical duplicates, 20 explicitly reported
`smiles_too_long` filters, and 11,988 final samples. The retained samples have a
median of 26 atoms, 43 canonical SMILES tokens, 45 canonical SMILES characters,
and 47.5 active selected MACCS targets. All 154 targets have at least one
positive sample. Each of the four unchanged shards contains 2,997 samples, and
their SHA-256 values match the C1 report.

C2 is complete. These are dataset audit statistics, not model training results
or evidence of paper-metric reproduction.

### Remaining issues

- C3 is next and must audit downstream datasets, implement a unified loader and
  random/scaffold splits, and test index and canonical-SMILES leakage.
- Generated reports and shards intentionally remain ignored under `artifacts/`;
  `docs/datasets.md` is the small tracked record.

## 2026-08-26 — Stage C3: downstream data audit and split interface

### Changes

- Enumerated all local downstream directories and files. BACE, Lipophilicity,
  SIDER, and Tox21 contain single-molecule CSVs; BioSNAP and TWOSIDES contain
  pairwise interaction data; ToxCast contains only a README and is not locally
  loadable.
- Compared every available dataset README with the actual files and recorded
  discrepancies. Notably, BACE documents 1,522 compounds while its CSV has
  1,513 and names `class` while the real column is `Class`; ToxCast's described
  raw data are absent. BioSNAP and TWOSIDES have no local README/provenance note.
- Audited CSV columns, logical row counts, task types, identifiers, label
  columns, and missing-label encoding. Tox21 has 16,026 missing cells across 12
  binary tasks; other supported CSV targets have no observed missing values.
- Safely inspected legacy NumPy headers without unpickling their object arrays.
  Lipophilicity, SIDER, and Tox21 `.npy` sample counts differ from their source
  CSVs and have no generation metadata, so they are not authoritative inputs to
  the new interface.
- Added `gtpro.data.downstream` with explicit `DatasetSpec` schemas and a unified
  `DownstreamDataset`. It supports BACE binary classification, BACE pIC50
  regression, Tox21 multilabel classification, Lipophilicity regression, and
  SIDER multilabel classification.
- The loader canonicalizes valid SMILES, preserves raw SMILES/identifier/source
  row, stores targets as `(samples, tasks)` `float32`, normalizes missing labels
  to `NaN`, exposes a Boolean target mask, validates classification values, and
  records or explicitly rejects invalid structures according to policy.
- Added deterministic random and Bemis–Murcko scaffold splitting. Random split
  operates on canonical-SMILES groups rather than individual rows; scaffold
  split keeps each scaffold group intact. Both therefore prevent equivalent
  canonical molecules from crossing splits.
- Added `tests/test_dataset_splits.py` covering registered schemas, missing-label
  masks, invalid SMILES diagnostics, canonical duplicates, deterministic random
  splits, scaffold grouping, complete index coverage, and leakage checks.
- Added `docs/downstream_datasets.md` with the full inventory, schemas, missing
  counts, README discrepancies, unsupported pairwise scope, interface examples,
  split behavior, and observed verification sizes. Corrected README and artifact
  documentation claims about downstream availability.

### Verification

Passed on 2026-08-26:

```bash
pytest -q tests/test_dataset_splits.py
python -m py_compile gtpro/data/downstream.py
pytest -q
```

All four focused dataset/split tests passed, followed by all 26 repository
tests. Three previously documented legacy invalid-escape warnings remain.

A separate full local audit loaded BACE classification and regression (1,513
rows each), Tox21 (7,831 rows and 12 targets), Lipophilicity (4,200 rows), and
SIDER (1,427 rows and 27 targets). RDKit accepted every source SMILES and no
canonical duplicate was found. The loader preserved all 16,026 Tox21 missing
labels as `NaN` plus a validity mask.

For BACE, Tox21, and Lipophilicity, seed-10 random and scaffold splits were
checked for complete index coverage, unique membership, and disjoint canonical
SMILES. Scaffold splits were additionally checked for disjoint Bemis–Murcko
scaffolds. All checks passed. Random sizes were respectively
1,209/152/152, 6,265/783/783, and 3,360/420/420. Scaffold sizes were
1,209/152/152, 5,650/1,474/707, and 3,360/420/420; the Tox21 deviation from the
requested ratio is reported because scaffold groups remain indivisible.

C3 is complete.

### Remaining issues

- C4 is next and must add a small, clearly test-only fixture covering valid,
  invalid, chiral, multi-atom, and missing-label cases without depending on the
  complete ChEMBL input.
- ToxCast is blocked on a source data file. BioSNAP and TWOSIDES require a
  separate pairwise dataset interface and provenance/license documentation;
  neither is silently treated as a single-molecule benchmark.
- The unified split API is deterministic and leakage-safe but does not claim
  label stratification. Scaffold-group indivisibility can produce noticeably
  imbalanced split sizes.

## 2026-08-26 — Stage C4: test-only molecular fixture

### Changes

- Added `tests/fixtures/downstream_smoke.csv`, a hand-constructed 11-row fixture
  covering valid and multi-atom molecules, equivalent canonical molecules, R/S
  stereochemistry, invalid and empty SMILES, binary/multilabel/regression
  targets, and missing labels.
- Added `tests/fixtures/README.md`, which explicitly prohibits treating the
  fixture or results derived from it as benchmark, ChEMBL, training, or
  scientific evaluation data.
- Extended downstream tests to load the committed fixture and verify its two
  rejected rows, canonical duplicate, three missing retained labels,
  stereochemistry, and multi-atom coverage.

### Verification

Passed on 2026-08-26:

```bash
pytest -q tests/test_dataset_splits.py tests/test_preprocessing.py
pytest -q
```

The focused suite passed all 10 tests, and the repository suite passed all 27
tests with the three known legacy warnings. The fixture has 11 data rows and is
eligible for Git; tests do not require the complete ChEMBL CSV. C4 and stage C
are complete.

### Remaining issues

- D1 is next and must establish model boundary structures, centralized device
  handling, mask semantics, and remove ambiguity among duplicate legacy files.

## 2026-08-26 — Stage D1: model boundaries and tensor contracts

### Changes

- Added typed `TextEncoderOutput`, `GraphEncoderOutput`, and `PretrainingBatch`
  boundaries plus centralized GROVER schema normalization, graph-component
  device movement, and molecule pooling in `gtpro.models.interfaces`.
- Changed the SMILES encoder to return the named structure while retaining tuple
  unpacking compatibility. Training and forward-smoke code now consume named
  fields and use one `encode_graph` path instead of branching over GROVER
  dict/tuple/tensor returns.
- Defined and tested padding-mask semantics (`True` means padded key) and atom
  target-mask semantics (`True` means a real supervised atom position). Removed
  `.data` mask construction and model-internal `.cuda()` calls; positional
  tensors follow their input device.
- Added `docs/model_interfaces.md` with graph, text, alignment/fusion, downstream
  head, shape, scope, and mask contracts.
- Added `tests/test_model_shapes.py` covering text output shapes, padding and
  atom masks, centralized device conversion, GROVER schema normalization, and
  atom-to-molecule pooling.
- Repository-wide reference search found no consumers of the three legacy
  `nt_xent.py` copies. Removed the buggy pretrain/finetune copies and retained
  `gtpro.nt_xent` as the sole implementation.
- Normalized diff and reference checks showed `seq_trans_fixed.py` differed from
  active `seq_trans.py` only in comments/local naming around the same device
  fix. Removed the unreferenced duplicate and declared `pretrain` as a
  compatibility package included by packaging metadata.

### Verification

Passed on 2026-08-26:

```bash
pytest -q tests/test_model_shapes.py tests/test_grover_source.py
python examples/smoke_test.py
python test_forward.py
pytest -q
```

The focused eight tests passed, both required forward commands completed with
the documented production shapes, and the full suite passed all 30 tests. A
source search found no remaining `.cuda()` calls in supported pretraining,
model-interface, or bundled GROVER code. D1 is complete.

### Remaining issues

- D2 is next and must separate the epoch runner, expose component losses,
  introduce validation/checkpoint recovery, and remove list-to-tensor warnings.

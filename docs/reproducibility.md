# Reproducibility guide

## Environment

Use Python 3.9–3.12. The measured 2026-08-27 run used macOS on CPU, Python
3.12.2, PyTorch 2.13.0, RDKit, NumPy, SciPy and scikit-learn versions captured
per run. Install and verify with:

```bash
python -m pip install -e ".[train,test]"
python -c "import gtpro; print(gtpro.__version__, gtpro.__file__)"
pytest -q
python examples/ci_smoke_test.py
```

See `docs/environment.md` for the locked environment and historical paper
environment. GPU PyTorch installation is platform-specific; requested but
unavailable CUDA/MPS devices fail explicitly.

## Data and commands

Prepare data and confirm the report before training:

```bash
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output-dir artifacts/pretrain --num-shards 4
python -c "import json; print(json.load(open('artifacts/pretrain/data_report.json'))['counts'])"
```

The frozen compact sequence is:

```bash
python scripts/run_pretraining.py --config configs/pretrain_reproduction_compact.yaml
python scripts/run_reproduction.py
python scripts/run_classical_baselines.py
python scripts/run_ablations.py
python scripts/summarize_results.py
```

Seeds are 42, 52 and 62. Splits are canonical-grouped random or Bemis-Murcko
scaffold 80/10/10. Rerunnable matrix scripts skip only directories with a
complete `metrics.json`; failed records remain auditable.

## Outputs and result provenance

Each formal downstream seed directory contains resolved `config.yaml`,
`environment.json`, `metrics.json`, `predictions.csv`, and
`best_checkpoint.*`. Pretraining additionally saves best/last state with model,
optimizer, scheduler, epoch, seed and history. Large local runs and weights are
ignored. The aggregator copies lightweight config/metrics/environment records
to `results/run_records`, then derives CSV, Markdown and SVG outputs.

Do not edit generated tables by hand. Re-run the aggregator and inspect the
source record named by the corresponding dataset/split/model/seed path.

## Differences from the paper

The historical paper environment used older Python/PyTorch/RDKit versions and a
768-dimensional text plus 1200-dimensional graph architecture. The measured
repository run uses a 64/64-dimensional, one-layer compact model, one
pretraining epoch, CPU, and the locally audited 11,988-molecule cohort. It does
not load an author-released GTpro checkpoint. Accordingly, numerical differences
may arise from capacity, pretraining duration/data, dependency versions,
initialization, split implementation, and missing original artifacts.

The compact GTpro result is weaker than multiple classical baselines. This is
reported directly, not hidden or described as state of the art. See
`docs/experiment_protocol.md` for the frozen selection and metric rules.

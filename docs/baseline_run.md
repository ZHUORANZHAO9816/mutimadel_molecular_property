# Pre-refactor smoke baseline

Run date: 2026-08-26 (Asia/Shanghai)

This page records the actual pre-refactor smoke checks required by stage A3.
These checks establish that the current code path executes; they are not paper
reproduction experiments, model-quality measurements, or formal training
results.

## Environment

| Item | Observed value |
| --- | --- |
| Operating system | macOS 26.5.2, arm64 |
| Python | 3.12.2, conda-forge build |
| PyTorch | 2.13.0 |
| NumPy | 1.26.4 |
| RDKit | 2024.03.5 |
| CUDA available | No |
| MPS available | No |
| Selected device | CPU |
| Active GROVER package | `/opt/anaconda3/lib/python3.12/site-packages/grover` |
| Git commit | None; the newly initialized repository has no commits |

This differs materially from the README's original-paper-oriented Python 3.7,
PyTorch 1.7.1, RDKit 2020.09.1, and CUDA installation instructions. The checks
below validate only the observed modern CPU environment.

## Forward smoke test

Command:

```bash
python test_forward.py
```

Result: **passed** with exit code 0.

Measured wall-clock time using `/usr/bin/time -p`: **5.17 seconds**.

Output summary:

```text
BERT parameters:   43,416,745
GROVER parameters: 107,143,232
CoCa parameters:   146,665,826
Graph atom embedding:   [20, 1200]
Graph global embedding: [4, 1200]
Text all embedding:     [4, 201, 768]
Text global embedding:  [4, 768]
Text atom embedding:    [4, 200, 768]
Full joint forward pass: successful
```

The script printed a loss of `2.8576`. The input tensors and model parameters
are randomly initialized and the script does not set a seed, so this number is
non-reproducible diagnostic output and must not be interpreted as a metric.

## One-epoch pretraining smoke test

Command:

```bash
PYTHONPATH=. python pretrain/pretrain_model.py \
  --epochs 1 \
  --batch_size 2 \
  --data_path ./data/pretrain_data/gtpro_smoke
```

Result: **passed** with exit code 0.

Observed run facts:

- loaded `gtpro_smoke_1.npy` and `gtpro_smoke_2.npy`;
- loaded 32 samples in total;
- used batch size 2, producing 16 batches;
- selected CPU;
- used configured seed 10;
- initialized GROVER with random weights and froze it;
- completed epoch 1 of 1;
- training loop reported an epoch duration of approximately 9 seconds;
- measured process wall-clock time was 14.25 seconds;
- the legacy loop printed accumulated pretraining loss `42.4049`.

The printed loss is the current loop's accumulated smoke value, not a
sample-normalized validation metric. It must not be copied into a reproduction
results table.

The run updated these ignored smoke artifacts:

| File | Size after run | Classification |
| --- | ---: | --- |
| `checkpoints/model_bert0.pth` | 173,700,825 bytes | smoke output, not a pretrained release |
| `checkpoints/model_coca0.pth` | 586,804,727 bytes | smoke output, not a pretrained release |

No GROVER checkpoint was saved by the current loop. Both files above remain
excluded by `.gitignore`.

## Warnings and limitations observed

- `pretrain/seq_trans.py:298` emitted a PyTorch warning that constructing a
  tensor directly from a list of NumPy arrays is very slow. D2 explicitly tracks
  this performance cleanup; A3 did not alter runtime code.
- The current pretraining command needs `PYTHONPATH=.` because the project is not
  installed and its import layout is inconsistent. B1/B4 track that issue.
- The runtime used the external `grover` package, not the vendored
  `gtpro.graph_trans` implementation. B2 must unify the source.
- The run has no validation split, best/last checkpoint distinction, resume
  state, environment manifest, or normalized component losses. These are D2
  requirements.
- CPU smoke timing on 32 samples is not an estimate of full ChEMBL training
  runtime.

## Acceptance conclusion

Both A3 acceptance commands completed successfully in the observed environment.
The repository therefore has a documented pre-refactor execution baseline, but
no claim of paper reproduction or pretrained model quality.

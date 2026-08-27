# Environments and installation

GTpro is being modernized as a reproducibility project. The original paper
environment is retained below as historical context, while package metadata and
the checked environment describe what this repository currently supports. A
successful smoke test is not evidence that the paper's numerical results have
been reproduced.

## Supported package installation

The modern package metadata supports Python 3.9 through 3.12. From the
repository root:

```bash
python -m pip install -e .
python -c "import gtpro; print(gtpro.__version__, gtpro.__file__)"
```

The base install contains dependencies needed by the reusable `gtpro` package,
including PyTorch, NumPy, RDKit, SciPy, scikit-learn, Descriptastorus, and tqdm.
Optional dependency groups are:

```bash
python -m pip install -e ".[train]"       # current training/data CLIs
python -m pip install -e ".[test]"        # test runner and coverage
python -m pip install -e ".[dev]"         # test plus build/lint tooling
python -m pip install -e ".[legacy-pyg]"  # legacy build_pretrain.py only
```

`torch-geometric` is isolated in `legacy-pyg` because only the legacy
`pretrain/build_pretrain.py` imports it. The active smoke preprocessing path
does not require it.

PyTorch is declared without a CUDA-specific wheel URL. CPU, CUDA, and Apple
Silicon wheel availability varies by operating system and driver/toolkit
version; GPU users should install the appropriate PyTorch build for their
platform first, then install GTpro. This repository does not force an obsolete
CUDA wheel index.

## Locked current verification environment

`environment.yml` pins the versions observed during the successful 2026-08-26
CPU baseline. To create it with conda or mamba:

```bash
conda env create -f environment.yml
conda activate gtpro-reproduction
```

The checked environment is:

| Component | Version/value |
| --- | --- |
| Platform | macOS 26.5.2, arm64 |
| Python | 3.12.2 |
| PyTorch | 2.13.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.13.1 |
| scikit-learn | 1.4.2 |
| RDKit | 2024.03.5 |
| Descriptastorus | 2.8.0 |
| Device used by baseline | CPU; CUDA and MPS unavailable |

Exact availability of future package builds is outside the repository's
control. `environment.yml` is the locked verification reference; compatible
ranges in `pyproject.toml` are the normal installation interface.

## Original paper environment

The original README instructed users to create a Python 3.7 environment with
PyTorch 1.7.1 + CUDA 11.0, torchvision 0.8.2, torch-geometric 1.6.3,
torch-sparse 0.6.9, torch-scatter 2.0.6, RDKit 2020.09.1.0, TensorBoard, and
optional NVIDIA Apex. Those versions are retained for historical comparison;
they are not the only supported installation path and are not installed by the
modern package metadata.

The original commands used legacy, version-specific wheel indexes. Do not copy
those indexes into a modern environment without first confirming that the
matching Python, CUDA, and operating-system wheels still exist.

## GROVER source and license audit

GTpro uses GROVER, the graph encoder from “Self-Supervised Graph Transformer on
Large-Scale Molecular Data.” Its authoritative source is the
[Tencent AI Lab GROVER repository](https://github.com/tencent-ailab/grover),
whose [LICENSE](https://github.com/tencent-ailab/grover/blob/main/LICENSE) is
MIT and includes attribution for incorporated Chemprop code.

The environment audited on 2026-08-26 contains `grover 0.1.0` under
`/opt/anaconda3/lib/python3.12/site-packages`. Its package metadata points to
the authoritative repository but its `direct_url.json` shows it was installed
from `file:///tmp/grover`, and the installed distribution metadata does not
contain a license file. The repository also contains a second GROVER-like tree
under `gtpro/graph_trans`.

B2 selected `gtpro.graph_trans` as the only project runtime source, converted
its internal imports to package-relative imports, preserved upstream
attribution in the installed package, and added source/checkpoint compatibility
tests. The separately installed external package is no longer a GTpro runtime
dependency and may remain installed without affecting project imports. See
`docs/grover_source.md` for the compatibility audit.

The license audit here applies to GROVER only. The overall repository license
and the compatibility of every incorporated upstream file remain H4 release
work and must be completed before public distribution.

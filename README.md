# Boosting the performance of molecular property prediction via graph-text alignment and multi-granularity representation enhancement

> Official implementation of GTpro, accompanying [Zhao et al. (2024)](https://doi.org/10.1016/j.jmgm.2024.108843).

## Overview

Molecules are commonly represented as either SMILES text or molecular graphs,
but embeddings learned independently from these two modalities are not
naturally aligned. GTpro aligns graph and text features through contrastive
learning, fuses them with cross-attention, and enriches the learned molecular
representation with atom-, functional-group-, and molecule-level pretraining
objectives. The resulting representation is transferred to downstream
molecular property prediction tasks.

The paper reports that GTpro outperforms the compared state-of-the-art methods
while using less pretraining data.

## Why graph-text alignment?

<p align="center">
  <img src="docs/assets/paper_figure_1_alignment_motivation.jpg"
       alt="Figure 1: Motivation for aligning molecular graph and SMILES representations"
       width="620">
</p>

The same molecule contains corresponding atom- and functional-group-level
information in both its graph and SMILES forms. However, independently learned
graph and text embeddings can remain separated in the representation space.
GTpro uses a contrastive objective to bring paired representations together and
cross-attention to exchange information between the two modalities.

*Figure 1 from Zhao et al. (2024). Source: [DOI
10.1016/j.jmgm.2024.108843](https://doi.org/10.1016/j.jmgm.2024.108843).
© 2024 Elsevier Inc.; reproduced here by author request.*

## Architecture

<p align="center">
  <img src="docs/assets/paper_figure_2_gtpro_architecture.jpg"
       alt="Figure 2: Overall architecture of GTpro"
       width="760">
</p>

GTpro contains a graph encoder, a SMILES Transformer, graph-text contrastive
alignment, and a multimodal cross-attention module. The pretraining objectives
operate at three molecular granularities:

- **APP — Atom Property Prediction:** learns local atom-level chemical
  information.
- **FGP — Functional Group Prediction:** injects functional-group-level domain
  knowledge.
- **GTM — Graph-Text Matching:** learns molecule-level correspondence between
  graph and SMILES views.
- **CON — Contrastive Learning:** aligns paired graph and text representations
  in the embedding space.

*Figure 2 from Zhao et al. (2024): (A) graph and SMILES encoders with
contrastive alignment and cross-attention; (B) the Transformer encoder block;
and (C) the multi-granularity pretraining objectives. Source: [Journal of
Molecular Graphics and Modelling, DOI
10.1016/j.jmgm.2024.108843](https://doi.org/10.1016/j.jmgm.2024.108843).
© 2024 Elsevier Inc.; reproduced here by author request.*

## Original datasets

### Pretraining data

GTpro is pretrained with molecules from
[ChEMBL](https://www.ebi.ac.uk/chembl/). Each molecule is converted into two
paired views: a molecular graph for the graph encoder and a tokenized SMILES
sequence for the text encoder. Atom properties, functional groups, and
graph-text pairs provide the supervision for the multi-granularity objectives.

### Downstream benchmarks

The paper's classification experiments use five public MoleculeNet benchmarks.
The original dataset files and standard benchmark definitions are available
through [MoleculeNet](https://github.com/deepchem/moleculenet).

| Dataset | Prediction task | Metric |
|---|---|---|
| BBBP | Blood-brain barrier penetration | ROC-AUC |
| BACE | BACE-1 inhibitor classification | ROC-AUC |
| SIDER | Drug side-effect classification | ROC-AUC |
| ClinTox | Clinical toxicity classification | ROC-AUC |
| Tox21 | Toxicity classification across biological targets | ROC-AUC |

## Training method

1. **Build paired molecular views.** Canonical SMILES strings are tokenized for
   the text branch and converted to atom-bond graphs for the graph branch.
2. **Encode each modality.** A pretrained GROVER encoder extracts graph
   representations, while a six-layer Transformer encodes the SMILES sequence.
3. **Apply multi-granularity pretraining.** APP, FGP, and GTM learn chemical
   information at the atom, functional-group, and molecule levels.
4. **Align and fuse graph and text.** The contrastive loss maximizes agreement
   between paired global representations, and cross-attention produces the
   joint multimodal representation.
5. **Fine-tune for molecular properties.** A task-specific prediction head is
   trained on each downstream dataset and evaluated with ROC-AUC for the
   classification benchmarks.

The paper-scale configuration represented by [`configs/pretrain.yaml`](configs/pretrain.yaml)
uses the following settings:

| Component | Setting |
|---|---|
| SMILES Transformer | 6 layers, 768 hidden dimensions, 12 attention heads |
| Maximum SMILES length | 201 tokens |
| Graph encoder | pretrained GROVER, frozen during graph-text pretraining |
| Cross-modal module | 6 unimodal layers, 6 multimodal layers, 8 attention heads |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Pretraining epochs | 50 |
| Random seed | 10 |

## Results reported in the paper

Figure 3 compares three settings: no pretraining, multi-granularity pretraining
with APP + FGP + GTM, and the complete objective with APP + FGP + GTM + CON.
The complete GTpro configuration achieves the best ROC-AUC on all five plotted
classification benchmarks.

<p align="center">
  <img src="docs/assets/paper_figure_3_pretraining_ablation.jpg"
       alt="Figure 3: Paper-reported pretraining and contrastive-learning ablation"
       width="680">
</p>

| Dataset | GTpro objective | ROC-AUC ↑ |
|---|---|---:|
| BBBP | APP + FGP + GTM + CON | **0.962** |
| BACE | APP + FGP + GTM + CON | **0.881** |
| SIDER | APP + FGP + GTM + CON | **0.684** |
| ClinTox | APP + FGP + GTM + CON | **0.997** |
| Tox21 | APP + FGP + GTM + CON | **0.821** |

*Values are the final-configuration ROC-AUC scores printed in Figure 3 of the
paper. © 2024 Elsevier Inc.; reproduced here by author request.*

### Representation analysis

Figure 6 visualizes how contrastive alignment changes the learned attention.
With the contrastive objective, the model places stronger weight on
corresponding chemically meaningful regions in the molecular graph and SMILES
sequence. The graph-text attention matrix further shows the learned interactions
between graph atoms and SMILES tokens.

<p align="center">
  <img src="docs/assets/paper_figure_6_attention_analysis.jpg"
       alt="Figure 6: Atom/token attention before and after contrastive alignment"
       width="820">
</p>

*Figure 6 from Zhao et al. (2024), showing the example without and with the
contrastive objective and its graph-text attention matrix. © 2024 Elsevier
Inc.; reproduced here by author request.*

## Installation and training

Python 3.9–3.12 is supported.

```bash
python -m pip install -e ".[train,test]"
pytest -q
```

Prepare ChEMBL molecules from a CSV containing a `smiles` column:

```bash
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output-dir artifacts/pretrain \
  --num-shards 4
```

The same command can retrieve molecules through the ChEMBL web service when
`--input` is omitted:

```bash
python scripts/prepare_pretrain_data.py \
  --output-dir artifacts/pretrain \
  --max-molecules 12008 \
  --num-shards 4
```

Run graph-text pretraining with the paper-scale configuration:

```bash
python scripts/run_pretraining.py \
  --config configs/pretrain.yaml \
  --grover-checkpoint path/to/grover_large.pt
```

See the [preprocessing guide](docs/preprocessing.md),
[fine-tuning guide](docs/finetuning.md), and
[configuration reference](docs/configuration.md) for the complete workflow.

## Repository structure

```text
configs/       paper-scale and development configurations
gtpro/         public API, data, metrics, models, and training code
pretrain/      graph-text pretraining and alignment modules
scripts/       preprocessing, pretraining, fine-tuning, and evaluation entry points
tests/         unit and integration tests
docs/          method, dataset, environment, and reproducibility documentation
examples/      forward and molecular encoding examples
```

## Citation

```bibtex
@article{zhao2024gtpro,
  title   = {Boosting the performance of molecular property prediction via graph-text alignment and multi-granularity representation enhancement},
  author  = {Zhao, Zhuoran and Zhou, Qing and Wu, Chengkai and Su, Renbin and Xiong, Weihong},
  journal = {Journal of Molecular Graphics and Modelling},
  volume  = {132},
  pages   = {108843},
  year    = {2024},
  doi     = {10.1016/j.jmgm.2024.108843}
}
```

Additional citation metadata are available in [`CITATION.cff`](CITATION.cff).

## License and acknowledgements

Repository-authored engineering additions are released under the
[`LICENSE`](LICENSE). Bundled GROVER code retains its upstream MIT notice; see
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) for file-level scope.

We acknowledge the GTpro authors, Tencent AI Lab and Chemprop contributors for
the GROVER implementation lineage, and the RDKit, PyTorch, scikit-learn, and
scientific Python communities.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

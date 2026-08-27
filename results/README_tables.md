# Generated empirical result tables

All values below are project-measured compact-model results generated from copied `metrics.json` records. They are not values reported by the original paper.

## Reproduction and baselines

| Dataset | Split | Model | Metric | Mean ± SD | n | Total / trainable parameters |
|---|---|---|---|---:|---:|---:|
| bace | random | full_gtpro | roc_auc | 0.6296 ± 0.0316 | 3 | 568778 / 8321 |
| bace | random | graph_only | roc_auc | 0.7229 ± 0.0260 | 3 | 13 / 13 |
| bace | random | graph_smiles_no_alignment | roc_auc | 0.8776 ± 0.0245 | 3 | 2061 / 2061 |
| bace | random | morgan_logistic_or_ridge | roc_auc | 0.9027 ± 0.0219 | 3 | 1025 / 1025 |
| bace | random | morgan_random_forest | roc_auc | 0.8852 ± 0.0215 | 3 | 28951 / 28951 |
| bace | random | smiles_only | roc_auc | 0.8694 ± 0.0223 | 3 | 2049 / 2049 |
| bace | scaffold | full_gtpro | roc_auc | 0.4700 ± 0.0907 | 3 | 568778 / 8321 |
| bace | scaffold | graph_only | roc_auc | 0.6775 ± 0.0064 | 3 | 13 / 13 |
| bace | scaffold | graph_smiles_no_alignment | roc_auc | 0.8128 ± 0.0202 | 3 | 2061 / 2061 |
| bace | scaffold | morgan_logistic_or_ridge | roc_auc | 0.8668 ± 0.0258 | 3 | 1025 / 1025 |
| bace | scaffold | morgan_random_forest | roc_auc | 0.8550 ± 0.0088 | 3 | 28591 / 28591 |
| bace | scaffold | smiles_only | roc_auc | 0.8176 ± 0.0302 | 3 | 2049 / 2049 |
| lipophilicity | random | full_gtpro | rmse | 1.1416 ± 0.0109 | 3 | 568778 / 8321 |
| lipophilicity | random | graph_only | rmse | 1.0219 ± 0.0037 | 3 | 13 / 13 |
| lipophilicity | random | graph_smiles_no_alignment | rmse | 0.7582 ± 0.0205 | 3 | 2061 / 2061 |
| lipophilicity | random | morgan_logistic_or_ridge | rmse | 0.9367 ± 0.0582 | 3 | 1025 / 1025 |
| lipophilicity | random | morgan_random_forest | rmse | 0.8397 ± 0.0168 | 3 | 203645 / 203645 |
| lipophilicity | random | smiles_only | rmse | 0.8301 ± 0.0159 | 3 | 2049 / 2049 |
| lipophilicity | scaffold | full_gtpro | rmse | 1.0976 ± 0.0441 | 3 | 568778 / 8321 |
| lipophilicity | scaffold | graph_only | rmse | 0.9814 ± 0.0190 | 3 | 13 / 13 |
| lipophilicity | scaffold | graph_smiles_no_alignment | rmse | 0.8216 ± 0.0326 | 3 | 2061 / 2061 |
| lipophilicity | scaffold | morgan_logistic_or_ridge | rmse | 1.0014 ± 0.0818 | 3 | 1025 / 1025 |
| lipophilicity | scaffold | morgan_random_forest | rmse | 0.8845 ± 0.0237 | 3 | 202694 / 202694 |
| lipophilicity | scaffold | smiles_only | rmse | 0.8915 ± 0.0559 | 3 | 2049 / 2049 |
| tox21 | random | full_gtpro | macro_roc_auc | 0.6906 ± 0.0182 | 3 | 569493 / 9036 |
| tox21 | random | graph_only | macro_roc_auc | 0.7645 ± 0.0158 | 3 | 156 / 156 |
| tox21 | random | graph_smiles_no_alignment | macro_roc_auc | 0.8185 ± 0.0040 | 3 | 24732 / 24732 |
| tox21 | random | morgan_logistic_or_ridge | macro_roc_auc | 0.7694 ± 0.0153 | 3 | 12300 / 12300 |
| tox21 | random | morgan_random_forest | macro_roc_auc | 0.8091 ± 0.0164 | 3 | 730277 / 730277 |
| tox21 | random | smiles_only | macro_roc_auc | 0.8054 ± 0.0052 | 3 | 24588 / 24588 |
| tox21 | scaffold | full_gtpro | macro_roc_auc | 0.6707 ± 0.0382 | 3 | 569493 / 9036 |
| tox21 | scaffold | graph_only | macro_roc_auc | 0.7840 ± 0.0153 | 3 | 156 / 156 |
| tox21 | scaffold | graph_smiles_no_alignment | macro_roc_auc | 0.8181 ± 0.0124 | 3 | 24732 / 24732 |
| tox21 | scaffold | morgan_logistic_or_ridge | macro_roc_auc | 0.7568 ± 0.0052 | 3 | 12300 / 12300 |
| tox21 | scaffold | morgan_random_forest | macro_roc_auc | 0.7996 ± 0.0090 | 3 | 645331 / 645331 |
| tox21 | scaffold | smiles_only | macro_roc_auc | 0.8035 ± 0.0145 | 3 | 24588 / 24588 |

## One-seed ablation screening

| Dataset | Split | Model | Metric | Mean ± SD | n | Total / trainable parameters |
|---|---|---|---|---:|---:|---:|
| bace | random | full_model | roc_auc | 0.6488 ± 0.0000 | 1 | 568778 / 8321 |
| bace | random | graph_only | roc_auc | 0.6413 ± 0.0000 | 1 | 564682 / 4225 |
| bace | random | no_atom_objective | roc_auc | 0.6538 ± 0.0000 | 1 | 568778 / 8321 |
| bace | random | no_contrastive | roc_auc | 0.6740 ± 0.0000 | 1 | 568778 / 8321 |
| bace | random | no_cross_attention | roc_auc | 0.5925 ± 0.0000 | 1 | 568778 / 8321 |
| bace | random | no_functional_group_objective | roc_auc | 0.6003 ± 0.0000 | 1 | 568778 / 8321 |
| bace | random | no_molecule_objective | roc_auc | 0.6384 ± 0.0000 | 1 | 568778 / 8321 |
| bace | random | text_only | roc_auc | 0.5955 ± 0.0000 | 1 | 564682 / 4225 |

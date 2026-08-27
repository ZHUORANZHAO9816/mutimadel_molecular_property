# Dataset audit summary

This document contains stable, small statistics copied from the generated C2
report. It does not redistribute the source CSV or derived shards, and it does
not claim to describe all of ChEMBL or exactly reconstruct the private data
selection used by the paper.

## Current pretraining input

The local `data/pretrain_data/CHEMBL_smiles.csv` inspected on 2026-08-26 has
SHA-256:

```text
f06d01bc80fda05488881c8328465caaaec613987f16d0731c1eae442b12d560
```

It was processed with RDKit canonical isomeric SMILES, first-row canonical
deduplication, a maximum of 200 SMILES tokens, and four balanced shards:

```bash
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output artifacts/pretrain \
  --num-shards 4
```

### Row counts

| Measure | Count |
|---|---:|
| CSV data rows | 12,008 |
| Empty SMILES | 0 |
| RDKit parse successes | 12,008 |
| Canonicalized rows | 12,008 |
| Full processing successes | 11,988 |
| Failed/filtered rows | 20 |
| Unique canonical SMILES | 12,008 |
| Canonical duplicate rows | 0 |
| Final retained samples | 11,988 |

All 20 failures have reason `smiles_too_long`: they parsed and canonicalized,
but exceeded the configured 200-token limit. They were filtered rather than
silently truncated. Their row-level details remain only in the ignored
machine-readable artifact.

### Retained-sample distributions

These statistics cover the 11,988 final retained samples. Percentiles use
NumPy's linear method.

| Measure | Min | P25 | Median | Mean | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atom count | 2 | 20 | 26 | 27.485235 | 32 | 47 | 114 |
| Canonical SMILES tokens | 2 | 32 | 43 | 46.350684 | 55 | 82 | 199 |
| Canonical SMILES characters | 2 | 34 | 45 | 51.004004 | 59 | 99 | 260 |
| Active selected MACCS targets | 5 | 38 | 47.5 | 48.290874 | 59 | 73 | 97 |

GTpro uses 154 selected MACCS structural keys as molecule-level targets. All
154 have at least one positive sample in this retained set. These are structural
keys used as functional-group supervision; their integer identifiers should not
be interpreted as human-readable functional-group names.

The most prevalent selected targets are:

| MACCS key | Output position | Positive samples | Prevalence |
|---:|---:|---:|---:|
| 165 | 153 | 11,794 | 98.3817% |
| 163 | 151 | 11,247 | 93.8188% |
| 164 | 152 | 10,867 | 90.6490% |
| 162 | 150 | 10,846 | 90.4738% |
| 161 | 149 | 10,578 | 88.2382% |
| 156 | 144 | 9,880 | 82.4157% |
| 158 | 146 | 9,871 | 82.3407% |
| 159 | 147 | 9,293 | 77.5192% |
| 137 | 125 | 9,196 | 76.7100% |
| 160 | 148 | 8,868 | 73.9740% |

The generated JSON report contains positive counts and prevalence for all 154
targets.

### Generated shards

The four ignored shards contain 2,997 samples each:

| File | Samples | SHA-256 |
|---|---:|---|
| `CHEMBL_smiles_1.npy` | 2,997 | `738c8db6d04a2fe57b67c425769f589d90967e591c1f2f1f5a4705daea5202dc` |
| `CHEMBL_smiles_2.npy` | 2,997 | `f0ccd475e36c1fdfed81df1a78df3c47cc1739f757762d3046fb4b030ac00035` |
| `CHEMBL_smiles_3.npy` | 2,997 | `7fe93c41982ec82111f638e06b2a562af0a380f5b49ed7c71689d5f3dbcfbc5e` |
| `CHEMBL_smiles_4.npy` | 2,997 | `e09a4901095513f610a90a36720b34c738b79742a262f56f37cdb2e57ae34f6b` |

The authoritative local details are regenerated into
`artifacts/pretrain/data_report.json` and `data_report.md`. Generated artifacts
remain excluded from Git according to `docs/artifacts.md`.

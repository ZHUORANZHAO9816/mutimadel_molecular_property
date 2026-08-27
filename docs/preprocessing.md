# Pretraining data preparation

Stage C1 replaces the supported legacy preprocessing path with importable
functions in `gtpro.data.pretraining`. The root `download_pretrain_data.py`
retains old flags as a deprecated compatibility wrapper; new code should import
the package module or use the canonical script.

## Canonical command

```bash
python scripts/prepare_pretrain_data.py \
  --input data/pretrain_data/CHEMBL_smiles.csv \
  --output artifacts/pretrain \
  --num-shards 4
```

Relative paths are resolved from the repository root. Omitting `--input`
downloads a bounded, first-seen-deduplicated ChEMBL CSV. Run `--help` for the
download limit, token limit, duplicate override, and resume controls.

## Deterministic policies

- RDKit produces canonical isomeric SMILES, preserving stereochemistry.
- The first row for each canonical SMILES is retained by default; later rows
  are reported as duplicates. `--keep-duplicates` explicitly changes this.
- Canonical SMILES longer than 200 tokenizer tokens are filtered and reported.
  They are never silently truncated because truncation could break molecular
  syntax and graph/text alignment. `--max-tokens` configures the threshold.
- Empty, invalid, overlong, tokenization, and alignment failures retain their
  CSV line number, raw value, stable reason code, and diagnostic detail.
- Input order and balanced shard partitioning are deterministic. The downloader
  also retains first-seen order rather than deduplicating through an unordered
  set.

## Validation and output schema

Every retained sample is reparsed from its canonical SMILES. The pipeline
requires the number of atom tokens and the sum of its atom mask to equal the
RDKit graph node count. It also validates all tensor dimensions before saving.

Each `CHEMBL_smiles_N.npy` file is compatible with the existing GTpro loader
and contains five entries:

| Index | Contents | Per-sample shape |
|---:|---|---|
| 0 | token ids including `[GLO]` | `(201,)` |
| 1 | selected MACCS targets | `(154,)` |
| 2 | atom targets aligned to non-global tokens | `(200, 15)` |
| 3 | atom-position mask | `(200,)` |
| 4 | canonical isomeric SMILES | scalar string |

The 15 atom targets preserve the historical GTpro selection. CIP handling is
explicit for R, S, and atoms with no assigned CIP code, so missing `_CIPCode`
properties no longer cause molecules to be silently dropped.

## Atomic writes and resume

Shards and both reports are written to temporary files in the destination
directory, flushed, and atomically replaced. A later run skips a shard only
when the input SHA-256 and content-affecting configuration match the previous
report, the expected sample count matches, and the current shard checksum is
verified. A missing, damaged, or mismatched shard is rewritten.

The output directory receives:

```text
CHEMBL_smiles_1.npy
...
data_report.json
data_report.md
```

The reports cover counts, policies, all failures and duplicates, shape
invariants, shard sample counts, checksums, and whether each shard was written
or skipped. They also summarize atom counts, canonical SMILES token and
character lengths, active selected MACCS targets per molecule, and positive
count/prevalence for each of the 154 targets. Distribution statistics use
NumPy's linear percentile convention and population standard deviation over
the final retained samples. A stable tracked summary is kept in
`docs/datasets.md` without committing generated ChEMBL derivatives.

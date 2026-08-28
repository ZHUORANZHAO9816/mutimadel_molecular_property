# Pretraining data

`CHEMBL_smiles.csv` contains 12,008 unique SMILES retrieved from the ChEMBL Data
Web Services. Its source, selection rule, license, row count, and SHA-256 hash
are recorded in `data/DATA_LICENSES.md` and `data/paper_datasets.json`.

`gtpro_smoke_1.npy` and `gtpro_smoke_2.npy` are test-only fixtures. Each shard
contains 16 molecules (32 in total) in the legacy five-array format:

1. token indices;
2. global labels;
3. atom labels;
4. atom masks;
5. SMILES strings.

They exist solely for CPU smoke tests and integration tests. They are not a
representative ChEMBL sample, a benchmark dataset, a formal training corpus, or
evidence of paper reproduction. Checkpoints and metrics produced from these
fixtures must always be labelled as smoke-test outputs.

Generated `.npy` shards remain ignored because they are reproducible build
artifacts. Create them with `scripts/prepare_pretrain_data.py`, or run the whole
training path with `scripts/run_full_pipeline.py`.

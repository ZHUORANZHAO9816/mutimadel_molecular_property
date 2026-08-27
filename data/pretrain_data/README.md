# GTpro smoke pretraining fixtures

This directory may contain local full pretraining data, but Git tracks only the
small files whose names begin with `gtpro_smoke_` and this README.

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

`CHEMBL_smiles.csv` and generated full-size shards are intentionally ignored.
See `docs/artifacts.md` for local regeneration and storage policy.

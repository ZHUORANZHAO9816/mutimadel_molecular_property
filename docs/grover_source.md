# GROVER source unification

Stage B2 selects the repository-maintained implementation under
`gtpro.graph_trans` as GTpro's only runtime GROVER source. Project code no
longer imports the top-level `grover` module, so an unrelated or locally patched
site-packages installation cannot be selected accidentally.

## Authoritative upstream and attribution

The bundled implementation is derived from the
[Tencent AI Lab GROVER repository](https://github.com/tencent-ailab/grover),
the implementation of “Self-Supervised Graph Transformer on Large-Scale
Molecular Data.” GROVER is MIT-licensed and includes attribution for
incorporated Chemprop code. A package-level `gtpro/graph_trans/NOTICE` records
the upstream source, copyrights, and full license link and is included in built
distributions.

Local engineering changes must not be represented as new GROVER research. The
current differences include modern NumPy compatibility, explicit readout type
handling, GTpro checkpoint loading, and package-relative imports.

## Import policy

All files below now use `gtpro.graph_trans` or relative imports:

- pretraining and fine-tuning helpers;
- the forward smoke script;
- GROVER data, model, and utility subpackages themselves.

The source test parses project Python imports and rejects `import grover` or
`from grover...`. It also prints and validates the actual
`GROVEREmbedding` source path.

Expected source:

```text
<repository>/gtpro/graph_trans/model/models.py
```

## Model and checkpoint compatibility audit

Before source switching, the production-size local and installed external
`GROVEREmbedding` classes were instantiated with the same GTpro arguments:

| Check | Result |
| --- | ---: |
| Parameters per model | 107,143,232 |
| State-dict keys per model | 106 |
| Missing local keys | 0 |
| Extra local keys | 0 |
| Shape mismatches | 0 |

This proves structural parameter compatibility for the audited encoder
configuration; it does not prove that an arbitrary checkpoint belongs to that
configuration.

Official GROVER task checkpoints commonly prefix encoder tensors with
`grover.`, while directly saved `GROVEREmbedding` state dicts do not. The local
loader now accepts both explicit schemas, reports how many encoder tensors were
loaded, and raises an error when none are compatible. Automated tests exercise
both schemas and the rejection path.

The bundled encoder returns the upstream dict schema for
`embedding_output_type="both"`. GTpro's training and smoke paths explicitly
select `atom_from_atom`, preserving the representation used by the successful
A3 baseline while keeping the GROVER API structured and traceable.

## Verification

The following checks passed on 2026-08-26:

```bash
python -c "import inspect; from gtpro.graph_trans.model.models import GROVEREmbedding; print(inspect.getfile(GROVEREmbedding))"
pytest -q
python test_forward.py
PYTHONPATH=. python pretrain/pretrain_model.py \
  --epochs 1 --batch_size 2 \
  --data_path ./data/pretrain_data/gtpro_smoke
```

The source command resolved to this repository's
`gtpro/graph_trans/model/models.py`. Five source/checkpoint tests passed. All 19
bundled GROVER submodules imported without loading any top-level `grover` module.
The joint forward test passed, and the 32-sample pretraining smoke run completed
all 16 batches with the same seeded accumulated diagnostic loss (`42.4049`) as
the A3 external-package baseline. That comparison is an execution compatibility
check, not a formal training result.

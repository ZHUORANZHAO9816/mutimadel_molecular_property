# Test-only molecular fixture

`downstream_smoke.csv` is a hand-constructed engineering fixture containing 11
rows. It is not ChEMBL, a benchmark subset, or suitable for training or model
evaluation.

The rows deliberately cover:

- ordinary valid and multi-atom SMILES;
- equivalent non-canonical SMILES (`CCO` and `OCC`);
- assigned R and S stereochemistry;
- an invalid SMILES and an empty SMILES;
- binary, multilabel, regression, and missing target values.

The simple molecules and synthetic labels are included only to test parsing,
masking, canonical grouping, and splitting behavior. No metric produced from
this file may be presented as a scientific result.

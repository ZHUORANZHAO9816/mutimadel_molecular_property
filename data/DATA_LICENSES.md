# Dataset sources and terms

The CSV files tracked for paper reproduction are unmodified downloads from
their public sources, except that `.csv.gz` downloads are decompressed without
changing their contents. File hashes and row counts are recorded in
[`paper_datasets.json`](paper_datasets.json).

## ChEMBL pretraining molecules

- Source: [ChEMBL Data Web Services](https://www.ebi.ac.uk/chembl/api/data/molecule)
- Selection: first 12,008 unique canonical SMILES returned in ascending ChEMBL
  identifier order by `scripts/download_paper_datasets.py`
- ChEMBL data license: [Creative Commons Attribution-ShareAlike 3.0
  Unported](https://creativecommons.org/licenses/by-sa/3.0/)
- Attribution: ChEMBL, European Molecular Biology Laboratory's European
  Bioinformatics Institute (EMBL-EBI)

The tracked `CHEMBL_smiles.csv` is therefore distributed under CC BY-SA 3.0,
not under the repository software license.

## MoleculeNet downstream benchmarks

BBBP, BACE, SIDER, ClinTox, and Tox21 are downloaded from the public DeepChem
MoleculeNet endpoints listed in `paper_datasets.json`. DeepChem provides these
copies for benchmark use, while each dataset retains the attribution and terms
of its original scientific source. Users redistributing or using a benchmark
outside research reproduction should review the corresponding source described
by the [MoleculeNet documentation](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html).

These dataset files are not relicensed by this repository's `LICENSE`.

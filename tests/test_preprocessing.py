"""Tests for deterministic and validated pretraining-data preparation."""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from rdkit import Chem

from gtpro.data.pretraining import (
    PreprocessingConfig,
    atom_labels,
    prepare_pretraining_data,
    process_smiles,
)
from gtpro.graph_trans.data.molgraph import MolGraph


def _cip_atom(smiles: str) -> Chem.Atom:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return next(atom for atom in molecule.GetAtoms() if atom.HasProp("_CIPCode"))


def test_atom_labels_preserve_r_s_and_missing_cip() -> None:
    r_atom = _cip_atom("F[C@H](Cl)Br")
    s_atom = _cip_atom("F[C@@H](Cl)Br")
    r_labels = atom_labels(r_atom)
    s_labels = atom_labels(s_atom)

    assert r_atom.GetProp("_CIPCode") != s_atom.GetProp("_CIPCode")
    assert r_labels[-3:-1] == [r_atom.GetProp("_CIPCode") == "R", r_atom.GetProp("_CIPCode") == "S"]
    assert s_labels[-3:-1] == [s_atom.GetProp("_CIPCode") == "R", s_atom.GetProp("_CIPCode") == "S"]
    assert r_labels[-1] and s_labels[-1]

    no_cip_atom = Chem.MolFromSmiles("CCO").GetAtomWithIdx(0)
    no_cip_labels = atom_labels(no_cip_atom)
    assert no_cip_labels[-3:-1] == [False, False]
    assert len(no_cip_labels) == 15


def test_token_atom_mask_and_grover_graph_nodes_align() -> None:
    sample = process_smiles("N[C@@H](C)C(=O)O")
    graph = MolGraph(sample.canonical_smiles, Namespace(bond_drop_rate=0))

    assert sample.token_ids.shape == (201,)
    assert sample.atom_labels.shape == (200, 15)
    assert sample.atom_mask.shape == (200,)
    assert sample.global_labels.shape == (154,)
    assert int(sample.atom_mask.sum()) == sample.atom_count == graph.n_atoms
    assert np.all(sample.atom_labels[sample.atom_mask == 0] == 2)


def _write_input(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        writer.writerow(["CCO"])
        writer.writerow(["OCC"])
        writer.writerow(["not-a-smiles"])
        writer.writerow([""])
        writer.writerow(["F[C@H](Cl)Br"])
        writer.writerow(["C" * 201])


def test_reports_failures_canonical_duplicates_and_long_filter(tmp_path: Path) -> None:
    input_csv = tmp_path / "molecules.csv"
    output_dir = tmp_path / "output"
    _write_input(input_csv)

    report = prepare_pretraining_data(
        input_csv,
        output_dir,
        PreprocessingConfig(num_shards=2),
    )

    assert report["counts"] == {
        "total_rows": 6,
        "empty_rows": 1,
        "parse_success_rows": 4,
        "processing_success_rows": 3,
        "canonicalized_rows": 4,
        "failed_rows": 3,
        "unique_canonical_smiles": 3,
        "duplicate_rows": 1,
        "final_samples": 2,
    }
    failures = {failure["reason"]: failure for failure in report["failures"]}
    assert failures["invalid_smiles"]["row"] == 4
    assert failures["invalid_smiles"]["raw"] == "not-a-smiles"
    assert failures["empty_smiles"]["row"] == 5
    assert failures["smiles_too_long"]["row"] == 7
    assert report["duplicates"] == [
        {"row": 3, "raw": "OCC", "canonical_smiles": "CCO", "first_row": 2}
    ]
    assert report["policy"]["duplicates"] == "drop after canonicalization; retain first input row"
    assert report["failure_reason_counts"] == {
        "empty_smiles": 1,
        "invalid_smiles": 1,
        "smiles_too_long": 1,
    }
    distributions = report["distributions"]
    assert distributions["atom_count"]["count"] == 2
    assert distributions["atom_count"]["min"] == 3
    assert distributions["atom_count"]["max"] == 4
    assert distributions["atom_count"]["mean"] == 3.5
    assert distributions["canonical_smiles_token_length"]["count"] == 2
    functional_groups = distributions["functional_group_labels"]
    assert functional_groups["dimension"] == 154
    assert len(functional_groups["labels"]) == 154
    assert all(0 <= label["positive_count"] <= 2 for label in functional_groups["labels"])
    assert distributions["active_functional_group_labels_per_molecule"]["count"] == 2
    assert (output_dir / "data_report.json").is_file()
    assert (output_dir / "data_report.md").is_file()
    assert json.loads((output_dir / "data_report.json").read_text(encoding="utf-8")) == report
    markdown = (output_dir / "data_report.md").read_text(encoding="utf-8")
    assert "## Distributions" in markdown
    assert "### Functional-group target summary" in markdown

    shard_paths = [output_dir / shard["file"] for shard in report["shards"]]
    assert len(shard_paths) == 2
    for path in shard_paths:
        loaded = np.load(path, allow_pickle=True)
        assert loaded.shape == (5,)
        assert len(loaded[0]) == 1
    assert sum(shard["samples"] for shard in report["shards"]) == report["counts"]["final_samples"]


def test_verified_shards_are_skipped_on_resume(tmp_path: Path) -> None:
    input_csv = tmp_path / "molecules.csv"
    output_dir = tmp_path / "output"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        writer.writerows(([smiles] for smiles in ("CC", "CO", "CN", "CF")))

    config = PreprocessingConfig(num_shards=2)
    first = prepare_pretraining_data(input_csv, output_dir, config)
    shard_paths = [output_dir / shard["file"] for shard in first["shards"]]
    mtimes = {path.name: path.stat().st_mtime_ns for path in shard_paths}
    second = prepare_pretraining_data(input_csv, output_dir, config)

    assert [shard["action"] for shard in first["shards"]] == ["written", "written"]
    assert [shard["action"] for shard in second["shards"]] == ["skipped", "skipped"]
    assert {path.name: path.stat().st_mtime_ns for path in shard_paths} == mtimes
    assert not list(output_dir.glob("*.tmp"))
    assert [shard["sha256"] for shard in second["shards"]] == [
        shard["sha256"] for shard in first["shards"]
    ]


def test_empty_input_still_generates_complete_distribution_report(tmp_path: Path) -> None:
    input_csv = tmp_path / "empty.csv"
    input_csv.write_text("smiles\n", encoding="utf-8")

    report = prepare_pretraining_data(input_csv, tmp_path / "output")

    assert report["counts"]["total_rows"] == 0
    assert report["distributions"]["atom_count"]["count"] == 0
    assert report["distributions"]["atom_count"]["median"] is None
    assert report["distributions"]["functional_group_labels"]["dimension"] == 154
    assert report["shards"] == []

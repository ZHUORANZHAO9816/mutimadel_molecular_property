"""Tests for unified downstream loading and leakage-safe splits."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gtpro.data.downstream import (
    DATASET_SPECS,
    DatasetSpec,
    load_downstream_csv,
    scaffold_for_smiles,
    split_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "downstream_smoke.csv"


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "smiles", "label_a", "label_b"])
        writer.writerows(rows)


def _multilabel_spec() -> DatasetSpec:
    return DatasetSpec(
        name="test_multilabel",
        relative_path="unused.csv",
        smiles_column="smiles",
        target_columns=("label_a", "label_b"),
        task_type="multilabel_classification",
        native_missing_label="empty field",
        id_column="id",
    )


def _assert_partition_and_canonical_isolation(dataset, split) -> None:
    parts = split.as_dict()
    combined = np.concatenate(list(parts.values()))
    assert len(combined) == len(dataset)
    assert len(set(combined.tolist())) == len(dataset)
    assert sorted(combined.tolist()) == list(range(len(dataset)))

    canonical_sets = {
        name: {dataset.canonical_smiles[index] for index in indices}
        for name, indices in parts.items()
    }
    assert canonical_sets["train"].isdisjoint(canonical_sets["validation"])
    assert canonical_sets["train"].isdisjoint(canonical_sets["test"])
    assert canonical_sets["validation"].isdisjoint(canonical_sets["test"])


def test_registered_task_schemas_are_explicit() -> None:
    assert DATASET_SPECS["bbbp"].task_type == "binary_classification"
    assert DATASET_SPECS["bbbp"].target_columns == ("p_np",)
    assert DATASET_SPECS["bace"].task_type == "binary_classification"
    assert DATASET_SPECS["bace"].smiles_column == "mol"
    assert DATASET_SPECS["bace"].target_columns == ("Class",)
    assert DATASET_SPECS["bace_regression"].task_type == "regression"
    assert DATASET_SPECS["bace_regression"].target_columns == ("pIC50",)
    assert DATASET_SPECS["tox21"].task_type == "multilabel_classification"
    assert len(DATASET_SPECS["tox21"].target_columns) == 12
    assert DATASET_SPECS["lipophilicity"].task_type == "regression"
    assert DATASET_SPECS["sider"].task_type == "multilabel_classification"
    assert len(DATASET_SPECS["sider"].target_columns) == 27
    assert DATASET_SPECS["clintox"].task_type == "multilabel_classification"
    assert DATASET_SPECS["clintox"].target_columns == ("FDA_APPROVED", "CT_TOX")


def test_committed_fixture_covers_required_edge_cases() -> None:
    spec = DatasetSpec(
        name="fixture",
        relative_path="unused.csv",
        smiles_column="smiles",
        target_columns=("task_a", "task_b"),
        task_type="multilabel_classification",
        native_missing_label="empty field",
        id_column="id",
    )
    dataset = load_downstream_csv(spec, SMOKE_FIXTURE)

    assert len(dataset) == 9
    assert len(dataset.invalid_molecules) == 2
    assert {entry.reason for entry in dataset.invalid_molecules} == {"empty_smiles", "invalid_smiles"}
    assert dataset.canonical_duplicate_rows == 1
    assert int((~dataset.target_mask).sum()) == 3
    assert any("@" in smiles for smiles in dataset.smiles)
    assert max(len(smiles) for smiles in dataset.smiles) > len("CCO")


def test_loader_normalizes_missing_labels_and_reports_invalid_smiles(tmp_path: Path) -> None:
    path = tmp_path / "molecules.csv"
    _write_csv(
        path,
        [
            ("a", "CCO", "1", ""),
            ("b", "OCC", "0", "1"),
            ("bad", "not-a-smiles", "1", "0"),
            ("chiral", "F[C@H](Cl)Br", "1", "0"),
        ],
    )

    dataset = load_downstream_csv(_multilabel_spec(), path)

    assert len(dataset) == 3
    assert dataset.targets.shape == (3, 2)
    assert dataset.targets.dtype == np.float32
    assert np.isnan(dataset.targets[0, 1])
    assert dataset.target_mask[0].tolist() == [True, False]
    assert dataset[0].identifier == "a"
    assert dataset[0].source_row == 2
    assert dataset.canonical_duplicate_rows == 1
    assert dataset.canonical_smiles[0] == dataset.canonical_smiles[1] == "CCO"
    assert dataset.invalid_molecules[0].source_row == 4
    assert dataset.invalid_molecules[0].reason == "invalid_smiles"


def test_random_split_has_no_index_or_canonical_smiles_leakage(tmp_path: Path) -> None:
    path = tmp_path / "molecules.csv"
    smiles = (
        "CCO", "OCC", "CCN", "CCCl", "CCBr", "CCF",
        "c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CC(=O)O", "CC#N", "COC",
    )
    _write_csv(path, [(str(index), value, str(index % 2), "") for index, value in enumerate(smiles)])
    dataset = load_downstream_csv(_multilabel_spec(), path)

    first = split_dataset(dataset, method="random", fractions=(0.6, 0.2, 0.2), seed=17)
    second = split_dataset(dataset, method="random", fractions=(0.6, 0.2, 0.2), seed=17)

    _assert_partition_and_canonical_isolation(dataset, first)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)


def test_scaffold_split_has_no_index_canonical_or_scaffold_leakage(tmp_path: Path) -> None:
    path = tmp_path / "molecules.csv"
    smiles = (
        "c1ccccc1", "Cc1ccccc1",
        "c1ccncc1", "Cc1ccncc1",
        "C1CCCCC1", "CC1CCCCC1",
        "C1CCCC1", "c1ccoc1",
        "CCO", "CCC",
    )
    _write_csv(path, [(str(index), value, str(index % 2), "1") for index, value in enumerate(smiles)])
    dataset = load_downstream_csv(_multilabel_spec(), path)
    split = split_dataset(dataset, method="scaffold", fractions=(0.6, 0.2, 0.2), seed=5)

    _assert_partition_and_canonical_isolation(dataset, split)
    scaffold_sets = {
        name: {scaffold_for_smiles(dataset.canonical_smiles[index]) for index in indices}
        for name, indices in split.as_dict().items()
    }
    assert scaffold_sets["train"].isdisjoint(scaffold_sets["validation"])
    assert scaffold_sets["train"].isdisjoint(scaffold_sets["test"])
    assert scaffold_sets["validation"].isdisjoint(scaffold_sets["test"])

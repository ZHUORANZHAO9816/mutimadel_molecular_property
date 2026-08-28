"""Unified loading and leakage-safe splitting for downstream molecule data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


TaskType = Literal["binary_classification", "multilabel_classification", "regression"]
SplitMethod = Literal["random", "scaffold"]

TOX21_TARGETS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)
SIDER_TARGETS = (
    "Hepatobiliary disorders",
    "Metabolism and nutrition disorders",
    "Product issues",
    "Eye disorders",
    "Investigations",
    "Musculoskeletal and connective tissue disorders",
    "Gastrointestinal disorders",
    "Social circumstances",
    "Immune system disorders",
    "Reproductive system and breast disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "General disorders and administration site conditions",
    "Endocrine disorders",
    "Surgical and medical procedures",
    "Vascular disorders",
    "Blood and lymphatic system disorders",
    "Skin and subcutaneous tissue disorders",
    "Congenital, familial and genetic disorders",
    "Infections and infestations",
    "Respiratory, thoracic and mediastinal disorders",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Ear and labyrinth disorders",
    "Cardiac disorders",
    "Nervous system disorders",
    "Injury, poisoning and procedural complications",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Schema and task semantics for one supported single-molecule dataset."""

    name: str
    relative_path: str
    smiles_column: str
    target_columns: tuple[str, ...]
    task_type: TaskType
    native_missing_label: str
    id_column: str | None = None


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "bbbp": DatasetSpec(
        name="bbbp",
        relative_path="bbbp/raw/BBBP.csv",
        smiles_column="smiles",
        target_columns=("p_np",),
        task_type="binary_classification",
        native_missing_label="none observed",
        id_column="name",
    ),
    "bace": DatasetSpec(
        name="bace",
        relative_path="bace/raw/bace.csv",
        smiles_column="mol",
        target_columns=("Class",),
        task_type="binary_classification",
        native_missing_label="none observed",
        id_column="CID",
    ),
    "bace_regression": DatasetSpec(
        name="bace_regression",
        relative_path="bace/raw/bace.csv",
        smiles_column="mol",
        target_columns=("pIC50",),
        task_type="regression",
        native_missing_label="none observed",
        id_column="CID",
    ),
    "tox21": DatasetSpec(
        name="tox21",
        relative_path="tox21/raw/tox21.csv",
        smiles_column="smiles",
        target_columns=TOX21_TARGETS,
        task_type="multilabel_classification",
        native_missing_label="empty CSV field parsed as NaN",
        id_column="mol_id",
    ),
    "lipophilicity": DatasetSpec(
        name="lipophilicity",
        relative_path="lipophilicity/raw/Lipophilicity.csv",
        smiles_column="smiles",
        target_columns=("exp",),
        task_type="regression",
        native_missing_label="none observed",
        id_column="CMPD_CHEMBLID",
    ),
    "sider": DatasetSpec(
        name="sider",
        relative_path="sider/raw/sider.csv",
        smiles_column="smiles",
        target_columns=SIDER_TARGETS,
        task_type="multilabel_classification",
        native_missing_label="none observed",
    ),
    "clintox": DatasetSpec(
        name="clintox",
        relative_path="clintox/raw/clintox.csv",
        smiles_column="smiles",
        target_columns=("FDA_APPROVED", "CT_TOX"),
        task_type="multilabel_classification",
        native_missing_label="none observed",
    ),
}


@dataclass(frozen=True)
class InvalidMolecule:
    source_row: int
    raw_smiles: str
    reason: str


@dataclass(frozen=True)
class MoleculeRecord:
    index: int
    source_row: int
    identifier: str | None
    smiles: str
    canonical_smiles: str
    targets: np.ndarray
    target_mask: np.ndarray


@dataclass(frozen=True)
class DownstreamDataset:
    """Normalized single-molecule dataset with NaN-based missing targets."""

    spec: DatasetSpec
    source_path: Path
    source_rows: np.ndarray
    identifiers: tuple[str | None, ...]
    smiles: tuple[str, ...]
    canonical_smiles: tuple[str, ...]
    targets: np.ndarray
    invalid_molecules: tuple[InvalidMolecule, ...]

    def __post_init__(self) -> None:
        size = len(self.smiles)
        if len(self.canonical_smiles) != size or len(self.identifiers) != size:
            raise ValueError("molecule metadata lengths do not match")
        if self.source_rows.shape != (size,):
            raise ValueError(f"source_rows has shape {self.source_rows.shape}, expected {(size,)}")
        if self.targets.shape != (size, len(self.spec.target_columns)):
            raise ValueError(
                f"targets has shape {self.targets.shape}, expected {(size, len(self.spec.target_columns))}"
            )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> MoleculeRecord:
        targets = self.targets[index]
        return MoleculeRecord(
            index=index,
            source_row=int(self.source_rows[index]),
            identifier=self.identifiers[index],
            smiles=self.smiles[index],
            canonical_smiles=self.canonical_smiles[index],
            targets=targets,
            target_mask=~np.isnan(targets),
        )

    @property
    def target_mask(self) -> np.ndarray:
        return ~np.isnan(self.targets)

    @property
    def canonical_duplicate_rows(self) -> int:
        return len(self.canonical_smiles) - len(set(self.canonical_smiles))


@dataclass(frozen=True)
class SplitIndices:
    method: SplitMethod
    seed: int
    fractions: tuple[float, float, float]
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def available_datasets() -> tuple[str, ...]:
    return tuple(DATASET_SPECS)


def _default_downstream_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "downstream"


def _parse_target(raw: str | None, column: str, source_row: int) -> float:
    value = "" if raw is None else str(raw).strip()
    if value.lower() in {"", "nan", "na", "n/a", "null", "none"}:
        return float("nan")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"row {source_row}: target {column!r} is not numeric: {value!r}") from error


def load_downstream_csv(
    spec: DatasetSpec,
    path: str | Path,
    *,
    invalid_smiles: Literal["drop", "raise"] = "drop",
) -> DownstreamDataset:
    """Load one schema-defined CSV and normalize missing labels to NaN."""

    if invalid_smiles not in {"drop", "raise"}:
        raise ValueError("invalid_smiles must be 'drop' or 'raise'")
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"downstream dataset does not exist: {source_path}")

    source_rows: list[int] = []
    identifiers: list[str | None] = []
    raw_smiles_values: list[str] = []
    canonical_values: list[str] = []
    target_values: list[list[float]] = []
    invalid: list[InvalidMolecule] = []

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"dataset has no CSV header: {source_path}")
        required = {spec.smiles_column, *spec.target_columns}
        if spec.id_column is not None:
            required.add(spec.id_column)
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"dataset {spec.name!r} is missing required columns: {missing}")

        for source_row, row in enumerate(reader, start=2):
            raw_smiles = str(row.get(spec.smiles_column) or "").strip()
            molecule = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
            if molecule is None:
                reason = "empty_smiles" if not raw_smiles else "invalid_smiles"
                entry = InvalidMolecule(source_row=source_row, raw_smiles=raw_smiles, reason=reason)
                if invalid_smiles == "raise":
                    raise ValueError(f"row {source_row}: {reason}: {raw_smiles!r}")
                invalid.append(entry)
                continue

            values = [_parse_target(row.get(column), column, source_row) for column in spec.target_columns]
            if spec.task_type in {"binary_classification", "multilabel_classification"}:
                invalid_values = [value for value in values if not np.isnan(value) and value not in {0.0, 1.0}]
                if invalid_values:
                    raise ValueError(
                        f"row {source_row}: classification targets must be 0, 1, or missing; found {invalid_values}"
                    )

            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            identifier_value = row.get(spec.id_column) if spec.id_column is not None else None
            identifier = str(identifier_value).strip() if identifier_value not in {None, ""} else None
            source_rows.append(source_row)
            identifiers.append(identifier)
            raw_smiles_values.append(raw_smiles)
            canonical_values.append(canonical)
            target_values.append(values)

    targets = np.asarray(target_values, dtype=np.float32)
    if not target_values:
        targets = np.empty((0, len(spec.target_columns)), dtype=np.float32)
    return DownstreamDataset(
        spec=spec,
        source_path=source_path,
        source_rows=np.asarray(source_rows, dtype=np.int64),
        identifiers=tuple(identifiers),
        smiles=tuple(raw_smiles_values),
        canonical_smiles=tuple(canonical_values),
        targets=targets,
        invalid_molecules=tuple(invalid),
    )


def load_downstream_dataset(
    name: str,
    root: str | Path | None = None,
    *,
    invalid_smiles: Literal["drop", "raise"] = "drop",
) -> DownstreamDataset:
    """Load a registered downstream dataset by name."""

    try:
        spec = DATASET_SPECS[name.lower()]
    except KeyError as error:
        choices = ", ".join(available_datasets())
        raise KeyError(f"unknown downstream dataset {name!r}; choose one of: {choices}") from error
    data_root = Path(root).expanduser().resolve() if root is not None else _default_downstream_root()
    return load_downstream_csv(spec, data_root / spec.relative_path, invalid_smiles=invalid_smiles)


def scaffold_for_smiles(smiles: str, include_chirality: bool = False) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"cannot generate scaffold for invalid SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=include_chirality)


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError("split fractions must contain train, validation, and test values")
    normalized = tuple(float(value) for value in fractions)
    if any(value < 0 for value in normalized) or not np.isclose(sum(normalized), 1.0):
        raise ValueError("split fractions must be non-negative and sum to 1")
    if sum(value > 0 for value in normalized) < 2:
        raise ValueError("at least two split fractions must be positive")
    return normalized  # type: ignore[return-value]


def _assign_groups(
    groups: list[list[int]],
    size: int,
    fractions: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(fractions, dtype=np.float64) * size
    counts = np.zeros(3, dtype=np.int64)
    assigned: list[list[int]] = [[], [], []]
    eligible = [index for index, fraction in enumerate(fractions) if fraction > 0]
    for group in groups:
        split_index = min(
            eligible,
            key=lambda index: (
                counts[index] / targets[index],
                -(targets[index] - counts[index]),
                index,
            ),
        )
        assigned[split_index].extend(group)
        counts[split_index] += len(group)
    return tuple(np.asarray(sorted(indices), dtype=np.int64) for indices in assigned)  # type: ignore[return-value]


def split_dataset(
    dataset: DownstreamDataset,
    *,
    method: SplitMethod = "random",
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    scaffold_include_chirality: bool = False,
) -> SplitIndices:
    """Split by canonical groups or Bemis-Murcko scaffold without leakage."""

    normalized_fractions = _validate_fractions(fractions)
    if method not in {"random", "scaffold"}:
        raise ValueError("method must be 'random' or 'scaffold'")

    grouped: dict[str, list[int]] = {}
    for index, canonical in enumerate(dataset.canonical_smiles):
        key = canonical if method == "random" else scaffold_for_smiles(canonical, scaffold_include_chirality)
        grouped.setdefault(key, []).append(index)

    rng = np.random.default_rng(seed)
    groups = list(grouped.values())
    if method == "random":
        rng.shuffle(groups)
    else:
        tie_breakers = rng.random(len(groups))
        groups = [
            group
            for _, _, group in sorted(
                ((-len(group), float(tie), group) for group, tie in zip(groups, tie_breakers)),
                key=lambda item: (item[0], item[1]),
            )
        ]

    train, validation, test = _assign_groups(groups, len(dataset), normalized_fractions)
    return SplitIndices(
        method=method,
        seed=seed,
        fractions=normalized_fractions,
        train=train,
        validation=validation,
        test=test,
    )

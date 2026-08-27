"""Deterministic, validated preprocessing for GTpro pretraining data.

The generated ``.npy`` files preserve the five-part layout used by the
original training loader: token ids, MACCS labels, atom labels, atom masks,
and canonical isomeric SMILES.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import MACCSkeys


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
VOCABULARY = (
    "[PAD]", "[GLO]", "c", "C", "(", ")", "O", "1", "2", "=", "N", "3",
    "n", "4", "[C@H]", "F", "[C@@H]", "-", "S", "/", "Cl", "[nH]", "s",
    "o", "5", "#", "[C@]", "[C@@]", "\\", "[O-]", "[N+]", "Br", "6", "P",
    "[n+]", "7", "I", "[S+]", "8", "[N-]", "[Si]", "B", "9", "[2H]",
    "[Se]", "[other_atom]", "[other_token]",
)
WORD_TO_INDEX = {token: index for index, token in enumerate(VOCABULARY)}
TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
UNBRACKETED_ATOMS = frozenset(
    {"B", "Br", "C", "Cl", "N", "O", "S", "P", "F", "I", "b", "c", "n", "o", "s", "p", "*"}
)
MACCS_SELECTED_INDICES = tuple(
    index for index in range(3, 166) if index not in {4, 5, 6, 7, 9, 10, 12, 31, 35}
)
ATOM_LABEL_SELECTED_INDICES = (1, 2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20, 21)
ATOM_LABEL_DIM = len(ATOM_LABEL_SELECTED_INDICES)
MASKED_ATOM_LABEL = (2,) * ATOM_LABEL_DIM
REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PreprocessingConfig:
    """Options that affect the content and partitioning of generated data."""

    num_shards: int = 4
    max_smiles_tokens: int = 200
    deduplicate: bool = True
    long_smiles_policy: str = "filter"
    resume: bool = True

    def validate(self) -> None:
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if self.max_smiles_tokens <= 0:
            raise ValueError("max_smiles_tokens must be positive")
        if self.long_smiles_policy != "filter":
            raise ValueError("long_smiles_policy currently supports only 'filter'")


@dataclass(frozen=True)
class ProcessedSample:
    token_ids: np.ndarray
    global_labels: np.ndarray
    atom_labels: np.ndarray
    atom_mask: np.ndarray
    canonical_smiles: str
    atom_count: int
    token_count: int


class SmilesProcessingError(ValueError):
    """An expected, reportable failure while processing one SMILES value."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        parsed: bool = False,
        canonical_smiles: str | None = None,
    ):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.parsed = parsed
        self.canonical_smiles = canonical_smiles


def _one_hot_unknown(value: object, allowable: Sequence[object]) -> list[bool]:
    if value not in allowable:
        value = allowable[-1]
    return [value == allowed for allowed in allowable]


def atom_labels(atom: Chem.Atom) -> list[bool]:
    """Return the original 15 GTpro atom targets, including safe CIP labels."""

    features = _one_hot_unknown(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    features += _one_hot_unknown(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other",
        ],
    )
    features += [atom.GetIsAromatic()]
    features += _one_hot_unknown(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    # Some atoms are potentially chiral but have no assigned CIP code. The
    # original code called GetProp unconditionally and dropped those molecules.
    cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
    features += [cip_code == "R", cip_code == "S", atom.HasProp("_ChiralityPossible")]
    return [bool(features[index]) for index in ATOM_LABEL_SELECTED_INDICES]


def tokenize_smiles(smiles: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise SmilesProcessingError(
            "tokenization_gap",
            "the tokenizer could not represent every character in the canonical SMILES",
            parsed=True,
            canonical_smiles=smiles,
        )
    return tokens


def _is_atom_token(token: str) -> bool:
    return token.startswith("[") or token in UNBRACKETED_ATOMS


def _token_index(token: str) -> int:
    if token in WORD_TO_INDEX:
        return WORD_TO_INDEX[token]
    fallback = "[other_atom]" if _is_atom_token(token) else "[other_token]"
    return WORD_TO_INDEX[fallback]


def process_smiles(smiles: str, max_smiles_tokens: int = 200) -> ProcessedSample:
    """Canonicalize one SMILES and build a fully validated training sample."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SmilesProcessingError("invalid_smiles", "RDKit could not parse the SMILES")

    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # defensive: canonical RDKit output should always parse
        raise SmilesProcessingError(
            "canonical_parse_failed",
            "RDKit could not reparse its canonical SMILES",
            parsed=True,
            canonical_smiles=canonical,
        )

    tokens = tokenize_smiles(canonical)
    if len(tokens) > max_smiles_tokens:
        raise SmilesProcessingError(
            "smiles_too_long",
            f"canonical SMILES has {len(tokens)} tokens; maximum is {max_smiles_tokens}; policy=filter",
            parsed=True,
            canonical_smiles=canonical,
        )

    real_atom_tokens = [token for token in tokens if _is_atom_token(token)]
    atom_count = molecule.GetNumAtoms()
    if len(real_atom_tokens) != atom_count:
        raise SmilesProcessingError(
            "atom_token_mismatch",
            f"tokenizer found {len(real_atom_tokens)} atom tokens but RDKit graph has {atom_count} nodes",
            parsed=True,
            canonical_smiles=canonical,
        )

    padded_tokens = tokens + ["[PAD]"] * (max_smiles_tokens - len(tokens))
    labels: list[Sequence[int | bool]] = []
    mask: list[int] = []
    atom_index = 0
    for token in padded_tokens:
        if _is_atom_token(token) and token != "[PAD]":
            labels.append(atom_labels(molecule.GetAtomWithIdx(atom_index)))
            mask.append(1)
            atom_index += 1
        else:
            labels.append(MASKED_ATOM_LABEL)
            mask.append(0)

    token_ids = np.asarray(
        [WORD_TO_INDEX["[GLO]"]] + [_token_index(token) for token in padded_tokens],
        dtype=np.int64,
    )
    atom_label_array = np.asarray(labels, dtype=np.int8)
    atom_mask = np.asarray(mask, dtype=np.int8)
    fingerprint = np.asarray(MACCSkeys.GenMACCSKeys(molecule), dtype=np.int8)
    global_labels = fingerprint[list(MACCS_SELECTED_INDICES)]

    expected_shapes = {
        "token_ids": (max_smiles_tokens + 1,),
        "atom_labels": (max_smiles_tokens, ATOM_LABEL_DIM),
        "atom_mask": (max_smiles_tokens,),
        "global_labels": (len(MACCS_SELECTED_INDICES),),
    }
    arrays = {
        "token_ids": token_ids,
        "atom_labels": atom_label_array,
        "atom_mask": atom_mask,
        "global_labels": global_labels,
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise SmilesProcessingError(
                "shape_validation_failed",
                f"{name} has shape {arrays[name].shape}, expected {expected}",
                parsed=True,
                canonical_smiles=canonical,
            )
    if atom_index != atom_count or int(atom_mask.sum()) != atom_count:
        raise SmilesProcessingError(
            "graph_alignment_failed",
            f"atom mask selects {int(atom_mask.sum())} positions but RDKit graph has {atom_count} nodes",
            parsed=True,
            canonical_smiles=canonical,
        )

    return ProcessedSample(
        token_ids=token_ids,
        global_labels=global_labels,
        atom_labels=atom_label_array,
        atom_mask=atom_mask,
        canonical_smiles=canonical,
        atom_count=atom_count,
        token_count=len(tokens),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_numpy(path: Path, samples: Sequence[ProcessedSample]) -> None:
    contents = np.empty(5, dtype=object)
    contents[0] = np.asarray([sample.token_ids for sample in samples], dtype=np.int64)
    contents[1] = np.asarray([sample.global_labels for sample in samples], dtype=np.int8)
    contents[2] = np.asarray([sample.atom_labels for sample in samples], dtype=np.int8)
    contents[3] = np.asarray([sample.atom_mask for sample in samples], dtype=np.int8)
    contents[4] = np.asarray([sample.canonical_smiles for sample in samples], dtype=object)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, contents, allow_pickle=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _config_fingerprint(config: PreprocessingConfig) -> str:
    content_config = asdict(config)
    content_config.pop("resume")
    encoded = json.dumps(content_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_previous_report(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _balanced_chunks(samples: Sequence[ProcessedSample], count: int) -> Iterable[Sequence[ProcessedSample]]:
    base, remainder = divmod(len(samples), count)
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        yield samples[start : start + size]
        start += size


def _numeric_distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    """Return a compact, JSON-safe distribution summary for integer values."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "standard_deviation": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    p05, p25, median, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 6),
        "standard_deviation": round(float(array.std()), 6),
        "p05": round(float(p05), 6),
        "p25": round(float(p25), 6),
        "median": round(float(median), 6),
        "p75": round(float(p75), 6),
        "p95": round(float(p95), 6),
    }


def _data_distributions(samples: Sequence[ProcessedSample]) -> dict[str, object]:
    atom_counts = [sample.atom_count for sample in samples]
    token_lengths = [sample.token_count for sample in samples]
    character_lengths = [len(sample.canonical_smiles) for sample in samples]
    if samples:
        label_matrix = np.stack([sample.global_labels for sample in samples]).astype(np.int64)
        positive_counts = label_matrix.sum(axis=0)
        active_per_molecule = label_matrix.sum(axis=1).tolist()
    else:
        positive_counts = np.zeros(len(MACCS_SELECTED_INDICES), dtype=np.int64)
        active_per_molecule = []

    labels = [
        {
            "position": position,
            "maccs_key": maccs_key,
            "positive_count": int(positive_counts[position]),
            "prevalence": (
                round(float(positive_counts[position]) / len(samples), 8) if samples else None
            ),
        }
        for position, maccs_key in enumerate(MACCS_SELECTED_INDICES)
    ]
    return {
        "population": "final retained samples after filtering and configured deduplication",
        "quantile_method": "NumPy linear percentile; population standard deviation",
        "atom_count": _numeric_distribution(atom_counts),
        "canonical_smiles_token_length": _numeric_distribution(token_lengths),
        "canonical_smiles_character_length": _numeric_distribution(character_lengths),
        "active_functional_group_labels_per_molecule": _numeric_distribution(active_per_molecule),
        "functional_group_labels": {
            "semantics": "154 selected MACCS structural-key targets used by GTpro",
            "dimension": len(MACCS_SELECTED_INDICES),
            "labels_with_any_positive": sum(label["positive_count"] > 0 for label in labels),
            "labels": labels,
        },
    }


def _report_markdown(report: dict[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, dict)
    policy = report["policy"]
    assert isinstance(policy, dict)
    shards = report["shards"]
    assert isinstance(shards, list)
    failures = report["failures"]
    assert isinstance(failures, list)
    distributions = report["distributions"]
    assert isinstance(distributions, dict)
    functional_groups = distributions["functional_group_labels"]
    assert isinstance(functional_groups, dict)
    functional_group_labels = functional_groups["labels"]
    assert isinstance(functional_group_labels, list)

    lines = [
        "# GTpro pretraining data report",
        "",
        f"- Input: `{report['input']}`",
        f"- Input SHA-256: `{report['input_sha256']}`",
        f"- Total data rows: {counts['total_rows']}",
        f"- Empty rows: {counts['empty_rows']}",
        f"- Parse-success rows: {counts['parse_success_rows']}",
        f"- Processing-success rows: {counts['processing_success_rows']}",
        f"- Failed rows: {counts['failed_rows']}",
        f"- Unique canonical SMILES: {counts['unique_canonical_smiles']}",
        f"- Duplicate rows: {counts['duplicate_rows']}",
        f"- Final samples: {counts['final_samples']}",
        "",
        "## Policy",
        "",
        f"- Duplicate policy: {policy['duplicates']}",
        f"- Long-SMILES policy: {policy['long_smiles']}",
        "- Canonicalization: RDKit canonical isomeric SMILES",
        "",
        "## Distributions",
        "",
        f"Population: {distributions['population']}.",
        "",
        "| Measure | Min | P25 | Median | Mean | P75 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Atom count", "atom_count"),
        ("Canonical SMILES tokens", "canonical_smiles_token_length"),
        ("Canonical SMILES characters", "canonical_smiles_character_length"),
        ("Active MACCS targets per molecule", "active_functional_group_labels_per_molecule"),
    ):
        values = distributions[key]
        lines.append(
            f"| {label} | {values['min']} | {values['p25']} | {values['median']} | "
            f"{values['mean']} | {values['p75']} | {values['p95']} | {values['max']} |"
        )

    top_labels = sorted(
        functional_group_labels,
        key=lambda label: (-label["positive_count"], label["maccs_key"]),
    )[:15]
    lines += [
        "",
        "### Functional-group target summary",
        "",
        f"GTpro uses {functional_groups['dimension']} selected MACCS structural-key targets; "
        f"{functional_groups['labels_with_any_positive']} have at least one positive sample.",
        "The complete per-label distribution is in `data_report.json`.",
        "",
        "| MACCS key | Output position | Positive samples | Prevalence |",
        "|---:|---:|---:|---:|",
    ]
    for label in top_labels:
        prevalence = label["prevalence"]
        prevalence_text = "n/a" if prevalence is None else f"{100 * prevalence:.4f}%"
        lines.append(
            f"| {label['maccs_key']} | {label['position']} | {label['positive_count']} | {prevalence_text} |"
        )
    lines += [
        "",
        "## Shards",
        "",
        "| File | Samples | SHA-256 | Action |",
        "|---|---:|---|---|",
    ]
    for shard in shards:
        lines.append(f"| `{shard['file']}` | {shard['samples']} | `{shard['sha256']}` | {shard['action']} |")
    if failures:
        lines += [
            "",
            "## Failed rows",
            "",
            "The machine-readable report contains every failed row, raw value, reason code, and detail.",
        ]
    lines.append("")
    return "\n".join(lines)


def prepare_pretraining_data(
    input_csv: str | Path,
    output_dir: str | Path,
    config: PreprocessingConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Process a CSV into deterministic shards and JSON/Markdown reports."""

    config = config or PreprocessingConfig()
    config.validate()
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    input_sha256 = _sha256(input_path)
    fingerprint = _config_fingerprint(config)
    previous = _load_previous_report(output_path / "data_report.json") if config.resume else None
    previous_shards: dict[str, dict[str, object]] = {}
    if (
        previous
        and previous.get("input_sha256") == input_sha256
        and previous.get("config_fingerprint") == fingerprint
    ):
        for shard in previous.get("shards", []):
            if isinstance(shard, dict) and isinstance(shard.get("file"), str):
                previous_shards[shard["file"]] = shard

    failures: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    samples: list[ProcessedSample] = []
    first_seen: dict[str, int] = {}
    total_rows = empty_rows = parse_success_rows = processing_success_rows = canonicalized_rows = 0

    def register_canonical(canonical: str, row_number: int, raw: str) -> bool:
        if canonical in first_seen:
            duplicates.append(
                {
                    "row": row_number,
                    "raw": raw,
                    "canonical_smiles": canonical,
                    "first_row": first_seen[canonical],
                }
            )
            return True
        first_seen[canonical] = row_number
        return False

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("input CSV has no header")
        columns = {column.strip().lower(): column for column in reader.fieldnames if column is not None}
        if "smiles" not in columns:
            raise ValueError(f"input CSV must contain a smiles column; found {reader.fieldnames}")
        smiles_column = columns["smiles"]

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            raw_value = row.get(smiles_column)
            raw = "" if raw_value is None else str(raw_value)
            smiles = raw.strip()
            if not smiles:
                empty_rows += 1
                failures.append(
                    {"row": row_number, "raw": raw, "reason": "empty_smiles", "detail": "SMILES value is empty"}
                )
                continue
            try:
                sample = process_smiles(smiles, config.max_smiles_tokens)
            except SmilesProcessingError as error:
                if error.parsed:
                    parse_success_rows += 1
                if error.canonical_smiles is not None:
                    canonicalized_rows += 1
                    register_canonical(error.canonical_smiles, row_number, raw)
                failures.append(
                    {"row": row_number, "raw": raw, "reason": error.code, "detail": error.detail}
                )
                continue
            except Exception as error:  # retain diagnostics instead of silently dropping rows
                failures.append(
                    {
                        "row": row_number,
                        "raw": raw,
                        "reason": "unexpected_processing_error",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            parse_success_rows += 1
            processing_success_rows += 1
            canonicalized_rows += 1
            is_duplicate = register_canonical(sample.canonical_smiles, row_number, raw)
            if is_duplicate and config.deduplicate:
                continue
            samples.append(sample)

            if progress is not None and total_rows % 500 == 0:
                progress(
                    f"Processed {total_rows} rows: {len(samples)} retained, "
                    f"{len(failures)} failed, {len(duplicates)} duplicates"
                )

    shard_reports: list[dict[str, object]] = []
    stem = input_path.stem
    for shard_number, shard_samples in enumerate(_balanced_chunks(samples, config.num_shards), start=1):
        if not shard_samples:
            continue
        filename = f"{stem}_{shard_number}.npy"
        shard_path = output_path / filename
        old = previous_shards.get(filename)
        can_skip = bool(
            old
            and old.get("samples") == len(shard_samples)
            and isinstance(old.get("sha256"), str)
            and shard_path.is_file()
            and _sha256(shard_path) == old["sha256"]
        )
        action = "skipped" if can_skip else "written"
        if not can_skip:
            _atomic_numpy(shard_path, shard_samples)
        checksum = _sha256(shard_path)
        shard_reports.append(
            {"file": filename, "samples": len(shard_samples), "sha256": checksum, "action": action}
        )
        if progress is not None:
            progress(f"{action.capitalize()} shard {shard_number}/{config.num_shards}: {filename}")

    failure_reason_counts: dict[str, int] = {}
    for failure in failures:
        reason = str(failure["reason"])
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input": str(input_path),
        "input_sha256": input_sha256,
        "config": asdict(config),
        "config_fingerprint": fingerprint,
        "policy": {
            "canonicalization": "RDKit canonical isomeric SMILES",
            "duplicates": "drop after canonicalization; retain first input row" if config.deduplicate else "retain",
            "long_smiles": f"filter when canonical token count exceeds {config.max_smiles_tokens}",
        },
        "validation": {
            "token_sequence_length": config.max_smiles_tokens + 1,
            "atom_label_sequence_length": config.max_smiles_tokens,
            "atom_label_dimension": ATOM_LABEL_DIM,
            "atom_mask_sum_equals_rdkit_graph_nodes": True,
        },
        "counts": {
            "total_rows": total_rows,
            "empty_rows": empty_rows,
            "parse_success_rows": parse_success_rows,
            "processing_success_rows": processing_success_rows,
            "canonicalized_rows": canonicalized_rows,
            "failed_rows": len(failures),
            "unique_canonical_smiles": len(first_seen),
            "duplicate_rows": len(duplicates),
            "final_samples": len(samples),
        },
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "distributions": _data_distributions(samples),
        "failures": failures,
        "duplicates": duplicates,
        "shards": shard_reports,
    }
    _atomic_json(output_path / "data_report.json", report)
    _atomic_text(output_path / "data_report.md", _report_markdown(report))
    return report


def download_chembl_smiles(
    max_molecules: int,
    output_csv: str | Path,
    progress: Callable[[str], None] | None = None,
    session: requests.Session | None = None,
) -> int:
    """Download a stable, first-seen-deduplicated ChEMBL SMILES CSV."""

    if max_molecules <= 0:
        raise ValueError("max_molecules must be positive")
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    smiles_values: list[str] = []
    seen: set[str] = set()
    offset = 0
    page_size = 100

    while len(smiles_values) < max_molecules:
        url = f"{CHEMBL_API}/molecule.json?limit={page_size}&offset={offset}&order=chembl_id"
        data = None
        for attempt in range(2):
            try:
                response = client.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException:
                if attempt == 0:
                    time.sleep(5)
                else:
                    raise
        molecules = data.get("molecules", []) if isinstance(data, dict) else []
        if not molecules:
            break
        for molecule in molecules:
            structures = molecule.get("molecule_structures")
            canonical = structures.get("canonical_smiles") if isinstance(structures, dict) else None
            if canonical and canonical not in seen:
                seen.add(canonical)
                smiles_values.append(canonical)
                if len(smiles_values) == max_molecules:
                    break
        offset += page_size
        if progress is not None:
            progress(f"Downloaded {len(smiles_values)}/{max_molecules} unique SMILES")
        if len(molecules) < page_size:
            break

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(["smiles"])
            writer.writerows((smiles,) for smiles in smiles_values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return len(smiles_values)

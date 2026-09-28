"""Build the hash-pinned categorical-diffusion training corpus.

The corpus is a deterministic view of the canonical parser sequence table and
the accepted five-fold homology/study split.  Fold 4 is validation-only.  The
builder deliberately carries no assay or provenance fields into model input.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARTIFACT = "categorical_diffusion_corpus_v1"
_ALPHABET = "".join(STANDARD_AMINO_ACIDS)
_MIN_LENGTH = 8
_MAX_LENGTH = 50
_FOLDS = 5
_HOLDOUT_FOLD = 4
_COMPONENT_WEIGHTING = "homology_component_equal_within_role_v1"
_ROLES = ("train", "validation")
_OUTPUT_FILES = frozenset({"corpus.jsonl", "summary.json", "manifest.json"})

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "alphabet",
        "min_length",
        "max_length",
        "folds",
        "holdout_fold",
        "component_weighting",
        "sequences_sha256",
        "assignments_sha256",
        "parser_top_manifest_sha256",
        "split_top_manifest_sha256",
        "split_independent_receipt_sha256",
        "expected_sequences",
        "expected_train_sequences",
        "expected_validation_sequences",
        "expected_homology_components",
        "expected_train_homology_components",
        "expected_validation_homology_components",
        "expected_union_components",
        "expected_train_union_components",
        "expected_validation_union_components",
        "expected_sequences_by_fold",
    }
)
_SEQUENCE_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_ASSIGNMENT_FIELDS = frozenset(
    {"schema_version", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)


@dataclass(frozen=True, slots=True)
class DiffusionCorpusConfig:
    """Validated, frozen policy and census for one corpus build."""

    artifact: str
    alphabet: str
    min_length: int
    max_length: int
    folds: int
    holdout_fold: int
    component_weighting: str
    sequences_sha256: str
    assignments_sha256: str
    parser_top_manifest_sha256: str
    split_top_manifest_sha256: str
    split_independent_receipt_sha256: str
    expected_sequences: int
    expected_train_sequences: int
    expected_validation_sequences: int
    expected_homology_components: int
    expected_train_homology_components: int
    expected_validation_homology_components: int
    expected_union_components: int
    expected_train_union_components: int
    expected_validation_union_components: int
    expected_sequences_by_fold: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DiffusionCorpusExecution:
    """Published paths and the principal sequence counts."""

    output_dir: Path
    corpus_path: Path
    summary_path: Path
    manifest_path: Path
    sequences: int
    train_sequences: int
    validation_sequences: int


@dataclass(frozen=True, slots=True)
class _InputSnapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _Assignment:
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    fold: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _inode(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_existing_symlink_chain(
    path: str | Path,
    *,
    label: str,
    include_leaf: bool = True,
) -> None:
    """Reject symlinks in every existing portion of an absolute path."""

    target = _absolute(path)
    candidates = list(reversed(target.parents))
    if include_leaf:
        candidates.append(target)
    for candidate in candidates:
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {candidate}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"{label} traverses a symbolic link: {candidate}")


def _input_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_snapshot(path: str | Path, *, label: str) -> _InputSnapshot:
    source = _absolute(path)
    _reject_existing_symlink_chain(source, label=label)
    try:
        descriptor = os.open(source, _input_open_flags())
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {source}")
        payload = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        try:
            named = os.lstat(source)
        except FileNotFoundError as error:
            raise ValueError(f"{label} changed while it was being read") from error
        _reject_existing_symlink_chain(source, label=label)
    finally:
        os.close(descriptor)
    fingerprint = _fingerprint(before)
    if (
        fingerprint != _fingerprint(after)
        or fingerprint != _fingerprint(named)
        or not stat.S_ISREG(named.st_mode)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was being read")
    return _InputSnapshot(
        path=source,
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=fingerprint,
    )


def _assert_snapshot_unchanged(snapshot: _InputSnapshot, *, label: str) -> None:
    try:
        current = _read_snapshot(snapshot.path, label=label)
    except ValueError as error:
        raise ValueError(f"{label} changed during the build") from error
    if current.fingerprint != snapshot.fingerprint or current.sha256 != snapshot.sha256:
        raise ValueError(f"{label} changed during the build")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _require_finite_json(value: object, *, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite JSON number")
    elif isinstance(value, Mapping):
        for item in value.values():
            _require_finite_json(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _require_finite_json(item, label=label)


def _parse_jsonl(
    payload: bytes,
    *,
    label: str,
    canonical: bool = False,
) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must use non-empty LF-delimited JSONL with one final LF")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload[:-1].split(b"\n"), start=1):
        if not raw_line:
            raise ValueError(f"{label} line {line_number} is blank")
        try:
            row = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{label} line {line_number} is not valid UTF-8 JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        _require_finite_json(row, label=f"{label} line {line_number}")
        if canonical and raw_line + b"\n" != _canonical_json_bytes(row):
            raise ValueError(f"{label} line {line_number} is not canonical compact JSON")
        rows.append(cast(dict[str, Any], row))
    return rows


def _require_exact_fields(row: Mapping[str, object], *, fields: frozenset[str], label: str) -> None:
    if set(row) != fields:
        raise ValueError(
            f"{label} schema mismatch: "
            f"missing={sorted(fields - set(row))}, extra={sorted(set(row) - fields)}"
        )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _config_from_payload(payload: bytes) -> DiffusionCorpusConfig:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("corpus config must use UTF-8 LF framing with one final LF")
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("corpus config is not valid UTF-8 TOML") from error
    if set(raw) != _CONFIG_FIELDS:
        raise ValueError(
            "corpus config schema mismatch: "
            f"missing={sorted(_CONFIG_FIELDS - set(raw))}, "
            f"extra={sorted(set(raw) - _CONFIG_FIELDS)}"
        )

    fixed_values: tuple[tuple[str, object], ...] = (
        ("schema_version", 1),
        ("artifact", _ARTIFACT),
        ("alphabet", _ALPHABET),
        ("min_length", _MIN_LENGTH),
        ("max_length", _MAX_LENGTH),
        ("folds", _FOLDS),
        ("holdout_fold", _HOLDOUT_FOLD),
        ("component_weighting", _COMPONENT_WEIGHTING),
    )
    for field, expected in fixed_values:
        value = raw[field]
        wrong_type = (
            type(value) is not int if type(expected) is int else type(value) is not type(expected)
        )
        if wrong_type or value != expected:
            raise ValueError(f"corpus config {field} must be {expected!r}")

    digest_fields = (
        "sequences_sha256",
        "assignments_sha256",
        "parser_top_manifest_sha256",
        "split_top_manifest_sha256",
        "split_independent_receipt_sha256",
    )
    digests = {
        field: _require_sha256(raw[field], label=f"corpus config {field}")
        for field in digest_fields
    }

    count_fields = (
        "expected_sequences",
        "expected_train_sequences",
        "expected_validation_sequences",
        "expected_homology_components",
        "expected_train_homology_components",
        "expected_validation_homology_components",
        "expected_union_components",
        "expected_train_union_components",
        "expected_validation_union_components",
    )
    counts = {
        field: _require_positive_int(raw[field], label=f"corpus config {field}")
        for field in count_fields
    }
    by_fold_raw = raw["expected_sequences_by_fold"]
    if not isinstance(by_fold_raw, list) or len(by_fold_raw) != _FOLDS:
        raise ValueError(f"corpus config expected_sequences_by_fold must have {_FOLDS} entries")
    by_fold = tuple(
        _require_positive_int(value, label=f"corpus config expected_sequences_by_fold[{fold}]")
        for fold, value in enumerate(by_fold_raw)
    )

    if sum(by_fold) != counts["expected_sequences"]:
        raise ValueError("corpus config fold sequence counts do not sum to expected_sequences")
    if by_fold[_HOLDOUT_FOLD] != counts["expected_validation_sequences"]:
        raise ValueError("corpus config holdout count differs from expected_validation_sequences")
    if sum(by_fold[:_HOLDOUT_FOLD]) != counts["expected_train_sequences"]:
        raise ValueError("corpus config fitting-fold counts differ from expected_train_sequences")
    if (
        counts["expected_train_sequences"] + counts["expected_validation_sequences"]
        != counts["expected_sequences"]
    ):
        raise ValueError("corpus config role sequence counts do not sum to expected_sequences")
    for component in ("homology", "union"):
        total = counts[f"expected_{component}_components"]
        train = counts[f"expected_train_{component}_components"]
        validation = counts[f"expected_validation_{component}_components"]
        if train + validation != total:
            raise ValueError(f"corpus config {component} role counts do not sum to its total")
        if total > counts["expected_sequences"]:
            raise ValueError(f"corpus config has more {component} components than sequences")
    if counts["expected_union_components"] > counts["expected_homology_components"]:
        raise ValueError("corpus config cannot have more union than homology components")
    for role in _ROLES:
        if (
            counts[f"expected_{role}_union_components"]
            > counts[f"expected_{role}_homology_components"]
        ):
            raise ValueError(f"corpus config has more {role} union than homology components")
        if counts[f"expected_{role}_homology_components"] > counts[f"expected_{role}_sequences"]:
            raise ValueError(f"corpus config has more {role} homology components than sequences")

    return DiffusionCorpusConfig(
        artifact=cast(str, raw["artifact"]),
        alphabet=cast(str, raw["alphabet"]),
        min_length=cast(int, raw["min_length"]),
        max_length=cast(int, raw["max_length"]),
        folds=cast(int, raw["folds"]),
        holdout_fold=cast(int, raw["holdout_fold"]),
        component_weighting=cast(str, raw["component_weighting"]),
        sequences_sha256=digests["sequences_sha256"],
        assignments_sha256=digests["assignments_sha256"],
        parser_top_manifest_sha256=digests["parser_top_manifest_sha256"],
        split_top_manifest_sha256=digests["split_top_manifest_sha256"],
        split_independent_receipt_sha256=digests["split_independent_receipt_sha256"],
        expected_sequences=counts["expected_sequences"],
        expected_train_sequences=counts["expected_train_sequences"],
        expected_validation_sequences=counts["expected_validation_sequences"],
        expected_homology_components=counts["expected_homology_components"],
        expected_train_homology_components=counts["expected_train_homology_components"],
        expected_validation_homology_components=counts["expected_validation_homology_components"],
        expected_union_components=counts["expected_union_components"],
        expected_train_union_components=counts["expected_train_union_components"],
        expected_validation_union_components=counts["expected_validation_union_components"],
        expected_sequences_by_fold=by_fold,
    )


def load_config(path: str | Path) -> DiffusionCorpusConfig:
    """Load and validate a categorical-diffusion corpus TOML contract."""

    return _config_from_payload(_read_snapshot(path, label="corpus config").payload)


def _validate_sequences(
    rows: Sequence[Mapping[str, object]], *, config: DiffusionCorpusConfig
) -> dict[str, str]:
    if len(rows) != config.expected_sequences:
        raise ValueError("sequence row count differs from the frozen config")
    sequences: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        label = f"sequence row {number}"
        _require_exact_fields(row, fields=_SEQUENCE_FIELDS, label=label)
        raw_id = row["sequence_id"]
        raw_sequence = row["sequence"]
        if not isinstance(raw_id, str) or not isinstance(raw_sequence, str):
            raise ValueError(f"{label} sequence_id and sequence must be strings")
        sequence = canonicalize_sequence(
            raw_sequence,
            min_length=config.min_length,
            max_length=config.max_length,
        )
        if sequence != raw_sequence:
            raise ValueError(f"{label} sequence is not canonical uppercase amino-acid text")
        sequence_id = _require_sha256(raw_id, label=f"{label} sequence_id")
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"{label} sequence_id does not match its canonical sequence")
        if sequence_id in sequences:
            raise ValueError(f"{label} repeats sequence_id")
        provenance = row["provenance"]
        if not isinstance(provenance, list) or not provenance:
            raise ValueError(f"{label} provenance must be a non-empty array")
        sequences[sequence_id] = sequence
    return sequences


def _component_id(value: object, *, label: str) -> str:
    return _require_sha256(value, label=label)


def _validate_assignments(
    rows: Sequence[Mapping[str, object]],
    *,
    sequences: Mapping[str, str],
    config: DiffusionCorpusConfig,
) -> dict[str, _Assignment]:
    if len(rows) != config.expected_sequences:
        raise ValueError("assignment row count differs from the frozen config")
    assignments: dict[str, _Assignment] = {}
    homology_owner: dict[str, tuple[str, int]] = {}
    union_fold: dict[str, int] = {}
    previous_sequence_id: str | None = None
    for number, row in enumerate(rows, start=1):
        label = f"assignment row {number}"
        _require_exact_fields(row, fields=_ASSIGNMENT_FIELDS, label=label)
        schema_version = row["schema_version"]
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError(f"{label} schema_version must be integer 1")
        sequence_id = _require_sha256(row["sequence_id"], label=f"{label} sequence_id")
        if previous_sequence_id is not None and sequence_id <= previous_sequence_id:
            raise ValueError(
                "accepted sequence assignments are not strictly ordered by sequence_id"
            )
        previous_sequence_id = sequence_id
        if sequence_id not in sequences:
            raise ValueError(f"{label} references an unknown sequence_id")
        if sequence_id in assignments:
            raise ValueError(f"{label} repeats sequence_id")
        homology_id = _component_id(
            row["homology_component_id"], label=f"{label} homology_component_id"
        )
        union_id = _component_id(row["union_component_id"], label=f"{label} union_component_id")
        fold = row["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < config.folds:
            raise ValueError(f"{label} fold must be an integer in 0..{config.folds - 1}")

        owner = (union_id, fold)
        previous_owner = homology_owner.setdefault(homology_id, owner)
        if previous_owner != owner:
            raise ValueError(f"{label} splits one homology component across union components/folds")
        previous_fold = union_fold.setdefault(union_id, fold)
        if previous_fold != fold:
            raise ValueError(f"{label} splits one union component across folds")
        assignments[sequence_id] = _Assignment(
            sequence_id=sequence_id,
            homology_component_id=homology_id,
            union_component_id=union_id,
            fold=fold,
        )

    if set(assignments) != set(sequences):
        missing = sorted(set(sequences) - set(assignments))
        extra = sorted(set(assignments) - set(sequences))
        raise ValueError(
            f"assignments must cover every sequence exactly once: missing={missing}, extra={extra}"
        )
    observed_folds = {assignment.fold for assignment in assignments.values()}
    if observed_folds != set(range(config.folds)):
        raise ValueError(f"assignments must contain every fold 0..{config.folds - 1}")
    return assignments


def _role(fold: int, *, config: DiffusionCorpusConfig) -> str:
    return "validation" if fold == config.holdout_fold else "train"


def _histogram(values: Sequence[int]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def _build_documents(
    *,
    sequences: Mapping[str, str],
    assignments: Mapping[str, _Assignment],
    config: DiffusionCorpusConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fold_counts = Counter(assignment.fold for assignment in assignments.values())
    role_counts = Counter(
        _role(assignment.fold, config=config) for assignment in assignments.values()
    )
    homology_by_role: dict[str, set[str]] = defaultdict(set)
    union_by_role: dict[str, set[str]] = defaultdict(set)
    component_sizes: Counter[tuple[str, str]] = Counter()
    for assignment in assignments.values():
        role = _role(assignment.fold, config=config)
        homology_by_role[role].add(assignment.homology_component_id)
        union_by_role[role].add(assignment.union_component_id)
        component_sizes[(role, assignment.homology_component_id)] += 1

    actual_fold_counts = tuple(fold_counts[fold] for fold in range(config.folds))
    if actual_fold_counts != config.expected_sequences_by_fold:
        raise ValueError("assignment fold counts differ from the frozen config")
    actual_counts = {
        "sequences": len(sequences),
        "train_sequences": role_counts["train"],
        "validation_sequences": role_counts["validation"],
    }
    expected_counts = {
        "sequences": config.expected_sequences,
        "train_sequences": config.expected_train_sequences,
        "validation_sequences": config.expected_validation_sequences,
    }
    if actual_counts != expected_counts:
        raise ValueError("assignment role counts differ from the frozen config")

    homology_counts = {
        "total": len(set().union(*homology_by_role.values())),
        "train": len(homology_by_role["train"]),
        "validation": len(homology_by_role["validation"]),
    }
    expected_homology_counts = {
        "total": config.expected_homology_components,
        "train": config.expected_train_homology_components,
        "validation": config.expected_validation_homology_components,
    }
    if homology_counts != expected_homology_counts:
        raise ValueError("homology component counts differ from the frozen config")
    union_counts = {
        "total": len(set().union(*union_by_role.values())),
        "train": len(union_by_role["train"]),
        "validation": len(union_by_role["validation"]),
    }
    expected_union_counts = {
        "total": config.expected_union_components,
        "train": config.expected_train_union_components,
        "validation": config.expected_validation_union_components,
    }
    if union_counts != expected_union_counts:
        raise ValueError("union component counts differ from the frozen config")

    corpus_rows: list[dict[str, object]] = []
    for sequence_id in sorted(sequences):
        assignment = assignments[sequence_id]
        role = _role(assignment.fold, config=config)
        component_count = homology_counts[role]
        component_size = component_sizes[(role, assignment.homology_component_id)]
        sampling_weight = 1.0 / (component_count * component_size)
        corpus_rows.append(
            {
                "schema_version": 1,
                "sequence_id": sequence_id,
                "sequence": sequences[sequence_id],
                "fold": assignment.fold,
                "role": role,
                "homology_component_id": assignment.homology_component_id,
                "homology_component_size": component_size,
                "union_component_id": assignment.union_component_id,
                "sampling_weight": sampling_weight,
            }
        )

    role_weight_sums: dict[str, float] = {}
    component_mass: dict[str, dict[str, float]] = {}
    for role in _ROLES:
        role_rows = [row for row in corpus_rows if row["role"] == role]
        role_sum = math.fsum(cast(float, row["sampling_weight"]) for row in role_rows)
        masses: dict[str, list[float]] = defaultdict(list)
        for row in role_rows:
            masses[cast(str, row["homology_component_id"])].append(
                cast(float, row["sampling_weight"])
            )
        observed_masses = [math.fsum(values) for values in masses.values()]
        expected_mass = 1.0 / homology_counts[role]
        if (
            not math.isclose(role_sum, 1.0, rel_tol=0.0, abs_tol=1e-15)
            or not observed_masses
            or any(
                not math.isfinite(value)
                or value <= 0.0
                or not math.isclose(value, expected_mass, rel_tol=0.0, abs_tol=1e-15)
                for value in observed_masses
            )
        ):
            raise AssertionError("component-equal sampling-weight invariant failed")
        role_weight_sums[role] = role_sum
        component_mass[role] = {
            "expected": expected_mass,
            "min": min(observed_masses),
            "max": max(observed_masses),
        }

    lengths_by_role: dict[str, list[int]] = {role: [] for role in _ROLES}
    for sequence_id, sequence in sequences.items():
        role = _role(assignments[sequence_id].fold, config=config)
        lengths_by_role[role].append(len(sequence))
    length_counts = {
        "total": _histogram([len(sequence) for sequence in sequences.values()]),
        "train": _histogram(lengths_by_role["train"]),
        "validation": _histogram(lengths_by_role["validation"]),
    }
    policy = {
        "alphabet": config.alphabet,
        "min_length": config.min_length,
        "max_length": config.max_length,
        "folds": config.folds,
        "holdout_fold": config.holdout_fold,
        "role_by_fold": {str(fold): _role(fold, config=config) for fold in range(config.folds)},
        "component_weighting": config.component_weighting,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact": config.artifact,
        "counts": actual_counts,
        "fold_counts": {str(fold): fold_counts[fold] for fold in range(config.folds)},
        "homology_component_counts": homology_counts,
        "union_component_counts": union_counts,
        "length_counts": length_counts,
        "sampling_weight_checks": {
            "formula": ("1/(homology_component_count_in_role*homology_component_size_in_role)"),
            "role_weight_sums": role_weight_sums,
            "homology_component_mass": component_mass,
        },
        "policy": policy,
    }
    return corpus_rows, summary


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row))
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())


def _write_compact_json(path: Path, document: object) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(document))
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _directory_open_flags())
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_output_inventory(
    directory_fd: int,
    *,
    expected: frozenset[str],
    phase: str,
) -> None:
    try:
        with os.scandir(directory_fd) as entries:
            names = frozenset(entry.name for entry in entries)
    except OSError as error:
        raise RuntimeError(f"cannot enumerate output directory {phase}") from error
    if names != expected:
        raise RuntimeError(
            f"output inventory mismatch {phase}: "
            f"missing={sorted(expected - names)}, extra={sorted(names - expected)}"
        )
    for name in names:
        try:
            observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(f"output entry {name!r} changed {phase}") from error
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"output entry {name!r} is not a regular file {phase}")


def _rollback_publication(
    *,
    output: Path,
    directory_fd: int | None,
    directory_inode: tuple[int, int] | None,
    published_inodes: Mapping[str, tuple[int, int]],
) -> None:
    """Remove only entries and the directory created by this invocation."""

    if directory_fd is not None:
        try:
            owns_directory_fd = (
                directory_inode is not None and _inode(os.fstat(directory_fd)) == directory_inode
            )
        except OSError:
            owns_directory_fd = False
        if owns_directory_fd:
            # A committed directory is 0555; rollback needs owner write and search.
            with contextlib.suppress(OSError):
                os.fchmod(directory_fd, 0o700)
        for name, expected_inode in (
            reversed(tuple(published_inodes.items())) if owns_directory_fd else ()
        ):
            try:
                observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                continue
            if _inode(observed) == expected_inode:
                with contextlib.suppress(OSError):
                    os.unlink(name, dir_fd=directory_fd)
        if owns_directory_fd:
            with contextlib.suppress(OSError):
                os.fsync(directory_fd)

    if directory_inode is None:
        return
    try:
        _reject_existing_symlink_chain(output, label="diffusion corpus output")
    except ValueError:
        return
    try:
        observed_directory = os.lstat(output)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(observed_directory.st_mode) and _inode(observed_directory) == directory_inode:
        # A foreign entry means the directory is no longer ours to remove.
        with contextlib.suppress(OSError):
            os.rmdir(output)
    with contextlib.suppress(OSError):
        _fsync_directory(output.parent)


def _publish_staged_bundle(
    *,
    staging: Path,
    output: Path,
    snapshots: Mapping[str, _InputSnapshot],
) -> None:
    """Commit hard-linked artifacts into an exclusively created directory."""

    directory_fd: int | None = None
    directory_inode: tuple[int, int] | None = None
    published_inodes: dict[str, tuple[int, int]] = {}
    committed = False
    try:
        _reject_existing_symlink_chain(
            output,
            label="diffusion corpus output",
            include_leaf=False,
        )
        try:
            os.mkdir(output)
        except FileExistsError as error:
            raise FileExistsError(
                f"diffusion corpus output appeared during build: {output}"
            ) from error
        directory_inode = _inode(os.lstat(output))
        directory_fd = os.open(output, _directory_open_flags())
        if _inode(os.fstat(directory_fd)) != directory_inode:
            raise RuntimeError("published output directory identity changed while opening it")
        _fsync_directory(output.parent)

        for name in ("corpus.jsonl", "summary.json"):
            source = staging / name
            source_inode = _inode(source.stat())
            os.link(source, name, dst_dir_fd=directory_fd, follow_symlinks=False)
            published_inodes[name] = source_inode
            published = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _inode(published) != source_inode:
                raise RuntimeError(f"published {name} inode differs from staging")
        _assert_output_inventory(
            directory_fd,
            expected=frozenset({"corpus.jsonl", "summary.json"}),
            phase="before manifest commit",
        )
        os.fsync(directory_fd)

        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)

        # Linking the manifest is the bundle commit marker and is always last.
        manifest_source = staging / "manifest.json"
        manifest_inode = _inode(manifest_source.stat())
        os.link(
            manifest_source,
            "manifest.json",
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published_inodes["manifest.json"] = manifest_inode
        published_manifest = os.stat("manifest.json", dir_fd=directory_fd, follow_symlinks=False)
        if _inode(published_manifest) != manifest_inode:
            raise RuntimeError("published manifest inode differs from staging")
        _assert_output_inventory(
            directory_fd,
            expected=_OUTPUT_FILES,
            phase="after manifest commit",
        )
        os.fsync(directory_fd)
        _fsync_directory(output.parent)

        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)
        observed_directory = os.lstat(output)
        if (
            not stat.S_ISDIR(observed_directory.st_mode)
            or _inode(observed_directory) != directory_inode
        ):
            raise RuntimeError("published output directory identity changed during commit")
        for name, expected_inode in published_inodes.items():
            observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _inode(observed) != expected_inode:
                raise RuntimeError(f"published {name} changed during commit")
            if stat.S_IMODE(observed.st_mode) != 0o444:
                raise RuntimeError(f"published {name} is not read-only")
        _assert_output_inventory(
            directory_fd,
            expected=_OUTPUT_FILES,
            phase="before directory chmod",
        )
        os.fchmod(directory_fd, 0o555)
        os.fsync(directory_fd)
        if stat.S_IMODE(os.fstat(directory_fd).st_mode) != 0o555:
            raise RuntimeError("published output directory is not read-only")
        _assert_output_inventory(
            directory_fd,
            expected=_OUTPUT_FILES,
            phase="after directory chmod",
        )
        _fsync_directory(output.parent)
        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)
        _reject_existing_symlink_chain(output, label="diffusion corpus output")
        _assert_output_inventory(
            directory_fd,
            expected=_OUTPUT_FILES,
            phase="at successful return",
        )
        committed = True
    finally:
        if not committed:
            _rollback_publication(
                output=output,
                directory_fd=directory_fd,
                directory_inode=directory_inode,
                published_inodes=published_inodes,
            )
        if directory_fd is not None:
            os.close(directory_fd)


def build_diffusion_corpus(
    *,
    sequences_path: str | Path,
    assignments_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> DiffusionCorpusExecution:
    """Validate, build, and transactionally publish a write-once corpus directory."""

    requested_output = _absolute(output_dir)
    _reject_existing_symlink_chain(
        requested_output,
        label="diffusion corpus output",
        include_leaf=False,
    )
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse diffusion corpus output: {requested_output}")
    output = requested_output

    snapshots = {
        "config": _read_snapshot(config_path, label="corpus config"),
        "sequences": _read_snapshot(sequences_path, label="canonical parser sequences"),
        "assignments": _read_snapshot(assignments_path, label="accepted sequence assignments"),
    }
    config = _config_from_payload(snapshots["config"].payload)
    if snapshots["sequences"].sha256 != config.sequences_sha256:
        raise ValueError("canonical parser sequences hash differs from the frozen config")
    if snapshots["assignments"].sha256 != config.assignments_sha256:
        raise ValueError("accepted sequence assignments hash differs from the frozen config")

    sequence_rows = _parse_jsonl(snapshots["sequences"].payload, label="canonical parser sequences")
    sequences = _validate_sequences(sequence_rows, config=config)
    assignment_rows = _parse_jsonl(
        snapshots["assignments"].payload,
        label="accepted sequence assignments",
        canonical=True,
    )
    assignments = _validate_assignments(assignment_rows, sequences=sequences, config=config)
    corpus_rows, summary = _build_documents(
        sequences=sequences,
        assignments=assignments,
        config=config,
    )

    for label, snapshot in snapshots.items():
        _assert_snapshot_unchanged(snapshot, label=label)

    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_chain(output, label="diffusion corpus output", include_leaf=False)
    if os.path.lexists(output):
        raise FileExistsError(f"diffusion corpus output appeared during build: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        corpus_path = staging / "corpus.jsonl"
        summary_path = staging / "summary.json"
        _write_jsonl(corpus_path, corpus_rows)
        _write_compact_json(summary_path, summary)

        manifest = {
            "schema_version": 1,
            "artifact": config.artifact,
            "config_sha256": snapshots["config"].sha256,
            "input": {
                "sequences": {"sha256": snapshots["sequences"].sha256},
                "assignments": {"sha256": snapshots["assignments"].sha256},
            },
            "provenance": {
                "parser_top_manifest_sha256": config.parser_top_manifest_sha256,
                "split_top_manifest_sha256": config.split_top_manifest_sha256,
                "split_independent_receipt_sha256": config.split_independent_receipt_sha256,
            },
            "artifacts": {
                "corpus": {"sha256": _sha256_bytes(corpus_path.read_bytes())},
                "summary": {"sha256": _sha256_bytes(summary_path.read_bytes())},
            },
            "counts": summary["counts"],
            "fold_counts": summary["fold_counts"],
            "homology_component_counts": summary["homology_component_counts"],
            "union_component_counts": summary["union_component_counts"],
            "length_counts": summary["length_counts"],
            "sampling_weight_checks": summary["sampling_weight_checks"],
            "policy": summary["policy"],
        }

        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)
        # The manifest is deliberately the final artifact created in staging.
        _write_compact_json(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)
        if os.path.lexists(output):
            raise FileExistsError(f"diffusion corpus output appeared during build: {output}")
        _publish_staged_bundle(staging=staging, output=output, snapshots=snapshots)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    with contextlib.suppress(OSError):
        _fsync_directory(output.parent)

    return DiffusionCorpusExecution(
        output_dir=output,
        corpus_path=output / "corpus.jsonl",
        summary_path=output / "summary.json",
        manifest_path=output / "manifest.json",
        sequences=config.expected_sequences,
        train_sequences=config.expected_train_sequences,
        validation_sequences=config.expected_validation_sequences,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        type=Path,
        required=True,
        help="hash-pinned canonical parser sequences JSONL",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="hash-pinned accepted sequence_assignments JSONL",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="strict categorical-diffusion corpus TOML contract",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to publish; existing paths are never reused",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_diffusion_corpus(
        sequences_path=args.sequences,
        assignments_path=args.assignments,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "sequences": result.sequences,
                "train_sequences": result.train_sequences,
                "validation_sequences": result.validation_sequences,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "DiffusionCorpusConfig",
    "DiffusionCorpusExecution",
    "build_diffusion_corpus",
    "build_parser",
    "load_config",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

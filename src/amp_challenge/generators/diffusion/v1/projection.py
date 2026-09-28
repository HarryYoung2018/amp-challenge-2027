"""Build leakage-safe outer-fold projections for native categorical diffusion v1.

The producer's only scientific inputs are the authenticated v1 contract, the
accepted 914-row trainer projection, and the sequence-free union assignments.
In particular, it has no corpus or fold-4 sequence input.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule
from amp_challenge.generators.diffusion.data import TrainingRow, load_training_projection
from amp_challenge.generators.diffusion.v1.contract import (
    NativeDiffusionV1Contract,
    load_unconditional_v1_contract,
)

ARTIFACT = "native_categorical_diffusion_v1_development_projection"
INDEPENDENT_RECEIPT_ARTIFACT = f"{ARTIFACT}_independent_verification"
STATUS = "passed"

TRAIN_FIELDS = ("sequence_id", "sequence", "sampling_weight")
SCORE_FIELDS = (
    "schema_version",
    "sequence_id",
    "sequence",
    "fold",
    "homology_component_id",
    "union_component_id",
    "sampling_weight",
)
ASSIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence_id",
        "homology_component_id",
        "union_component_id",
        "fold",
    }
)
CODE_MANIFEST_PATHS = (
    "cluster/slurm/build_native_diffusion_v1_development_projections_twins.sbatch",
    "configs/diffusion/unconditional_v1.toml",
    "pyproject.toml",
    "src/amp_challenge/__init__.py",
    "src/amp_challenge/constants.py",
    "src/amp_challenge/generators/__init__.py",
    "src/amp_challenge/generators/diffusion/__init__.py",
    "src/amp_challenge/generators/diffusion/categorical.py",
    "src/amp_challenge/generators/diffusion/data.py",
    "src/amp_challenge/generators/diffusion/v1/__init__.py",
    "src/amp_challenge/generators/diffusion/v1/contract.py",
    "src/amp_challenge/generators/diffusion/v1/projection.py",
    "src/amp_challenge/generators/diffusion/v1/verify_projection.py",
    "src/amp_challenge/sequences.py",
    "uv.lock",
)
FROZEN_INPUT_NAMES = (
    "sequence_assignments.jsonl",
    "training_projection.jsonl",
    "unconditional_v1.toml",
)
SERIALIZATION = (
    "utf8_canonical_json_sorted_keys_compact_separators_ensure_ascii_false_"
    "allow_nan_false_single_lf"
)
ROLE_WEIGHT_FORMULA = "1/(role_homology_component_count*role_homology_component_size)"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


@dataclass(frozen=True, slots=True)
class SequenceAssignment:
    """Sequence-free fold and component membership."""

    sequence_id: str
    homology_component_id: str
    union_component_id: str
    fold: int


@dataclass(frozen=True, slots=True)
class OuterScoreRow:
    """The evaluator's complete, least-privilege score record."""

    schema_version: int
    sequence_id: str
    sequence: str
    fold: int
    homology_component_id: str
    union_component_id: str
    sampling_weight: float


@dataclass(frozen=True, slots=True)
class OuterProjection:
    """One outer training projection and matching held-out score ledger."""

    fold: int
    train_folds: tuple[int, ...]
    train_rows: tuple[TrainingRow, ...]
    score_rows: tuple[OuterScoreRow, ...]
    train_homology_components: int
    score_homology_components: int
    train_union_components: int
    score_union_components: int
    train_sequence_ids_sha256: str
    score_sequence_ids_sha256: str
    selected_tokens_per_replicate: int
    selected_tokens_per_replicate_by_timestep_bin: tuple[int, ...]
    louco_minimum_union_components: int
    louco_minimum_homology_components: int
    louco_minimum_selected_tokens_per_replicate: int
    louco_minimum_selected_tokens_per_replicate_by_timestep_bin: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentProjectionSet:
    """The four deterministic outer projections derived from development only."""

    parent_contract_sha256: str
    development_sequences: int
    development_homology_components: int
    development_union_components: int
    outer_folds: tuple[OuterProjection, ...]


@dataclass(frozen=True, slots=True)
class ProjectionBundleExecution:
    """Content addresses for one successfully published projection bundle."""

    output_dir: Path
    manifest_sha256: str
    summary_sha256: str
    sha256sums_sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class _InputSnapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _ProjectionPolicy:
    config_sha256: str
    development_folds: tuple[int, ...]
    locked_holdout_fold: int
    training_projection_sha256: str
    union_assignments_sha256: str
    expected_sequences: int
    expected_development_sequences: int
    expected_locked_holdout_sequences: int
    expected_sequences_by_fold: tuple[int, ...]
    expected_development_homology_components: int
    expected_locked_holdout_homology_components: int
    expected_development_union_components: int
    expected_locked_holdout_union_components: int
    expected_outer_training_sequences: tuple[int, ...]
    expected_outer_score_sequences: tuple[int, ...]
    expected_outer_training_homology_components: tuple[int, ...]
    expected_outer_score_homology_components: tuple[int, ...]
    expected_outer_training_union_components: tuple[int, ...]
    expected_outer_score_union_components: tuple[int, ...]
    expected_score_selected_tokens: tuple[int, ...]
    expected_score_selected_tokens_by_bin: tuple[tuple[int, ...], ...]
    minimum_louco_union_components: tuple[int, ...]
    minimum_louco_homology_components: tuple[int, ...]
    minimum_louco_selected_tokens: tuple[int, ...]
    minimum_louco_selected_tokens_by_bin: tuple[tuple[int, ...], ...]
    total_levels: int
    cosine_offset: float
    selected_token_count_definition: str
    component_weighting: str
    outer_training_rule: str
    outer_score_rule: str
    calibration_crossfit: str
    calibration_exclusion_unit: str


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(
    path: str | os.PathLike[str],
    *,
    label: str,
    include_leaf: bool = True,
) -> None:
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


def _read_snapshot(path: str | os.PathLike[str], *, label: str) -> _InputSnapshot:
    source = _absolute(path)
    _reject_symlink_chain(source, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a non-symbolic regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            named = os.lstat(source)
        except OSError as error:
            raise ValueError(f"{label} changed while it was read") from error
        _reject_symlink_chain(source, label=label)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    fingerprint = _fingerprint(before)
    if (
        fingerprint != _fingerprint(after)
        or fingerprint != _fingerprint(named)
        or not stat.S_ISREG(named.st_mode)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was read")
    return _InputSnapshot(
        path=source,
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=fingerprint,
    )


def _assert_snapshot_unchanged(snapshot: _InputSnapshot, *, label: str) -> None:
    observed = _read_snapshot(snapshot.path, label=label)
    if observed.fingerprint != snapshot.fingerprint or observed.sha256 != snapshot.sha256:
        raise ValueError(f"{label} changed during projection construction")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_nonfinite(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label=label)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item, label=label)


def _parse_canonical_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty canonical LF-delimited JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload[:-1].split(b"\n"), start=1):
        if not raw_line:
            raise ValueError(f"{label} line {line_number} is blank")
        try:
            document = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from error
        if not isinstance(document, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        _reject_nonfinite(document, label=f"{label} line {line_number}")
        if _canonical_json_bytes(document) != raw_line + b"\n":
            raise ValueError(f"{label} line {line_number} is not canonical compact JSON")
        rows.append(cast(dict[str, Any], document))
    return tuple(rows)


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a table")
    return cast(Mapping[str, object], value)


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _integer_tuple(value: object, *, label: str, size: int) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple | list)
        or len(value) != size
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"{label} must contain exactly {size} positive integers")
    return tuple(cast(Sequence[int], value))


def _integer_matrix(
    value: object,
    *,
    label: str,
    rows: int,
    columns: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple | list) or len(value) != rows:
        raise ValueError(f"{label} must contain exactly {rows} rows")
    return tuple(
        _integer_tuple(row, label=f"{label}[{index}]", size=columns)
        for index, row in enumerate(value)
    )


def _policy_from_contract(contract: NativeDiffusionV1Contract) -> _ProjectionPolicy:
    document = _strict_mapping(contract.document, label="v1 contract")
    input_table = _strict_mapping(document.get("input"), label="input")
    development = _strict_mapping(document.get("development"), label="development")
    diffusion = _strict_mapping(document.get("diffusion"), label="diffusion")
    calibration = _strict_mapping(document.get("calibration"), label="calibration")
    folds = tuple(contract.development_folds)
    if folds != (0, 1, 2, 3) or contract.locked_holdout_fold != 4:
        raise ValueError("projection requires development folds 0..3 and locked fold 4")
    fold_count = len(folds)
    expected_sequences = _positive_int(input_table.get("expected_sequences"), label="sequences")
    expected_development = _positive_int(
        input_table.get("expected_development_sequences"), label="development sequences"
    )
    expected_locked = _positive_int(
        input_table.get("expected_locked_holdout_sequences"), label="locked sequences"
    )
    expected_by_fold = _integer_tuple(
        input_table.get("expected_sequences_by_fold"),
        label="sequences by fold",
        size=fold_count + 1,
    )
    if (
        sum(expected_by_fold) != expected_sequences
        or sum(expected_by_fold[:-1]) != expected_development
    ):
        raise ValueError("contract sequence censuses are inconsistent")
    if expected_by_fold[-1] != expected_locked:
        raise ValueError("contract locked-fold sequence census is inconsistent")

    expected_score = _integer_tuple(
        development.get("expected_outer_score_sequences"),
        label="outer score sequences",
        size=fold_count,
    )
    expected_train = _integer_tuple(
        development.get("expected_outer_training_sequences"),
        label="outer training sequences",
        size=fold_count,
    )
    if expected_score != expected_by_fold[:-1] or any(
        train != expected_development - score
        for train, score in zip(expected_train, expected_score, strict=True)
    ):
        raise ValueError("outer sequence censuses are inconsistent")

    selected_definition = development.get("selected_token_count_definition")
    if selected_definition != (
        "sum_levels_1_to_64_of_fixed_cosine_mask_count_per_sequence_per_replicate"
    ):
        raise ValueError("selected-token count definition changed")
    component_weighting = input_table.get("component_weighting")
    if component_weighting != (
        "homology_component_equal_recomputed_within_each_fit_and_score_role"
    ):
        raise ValueError("projection component weighting changed")
    outer_training_rule = development.get("outer_training_rule")
    outer_score_rule = development.get("outer_score_rule")
    calibration_crossfit = development.get("calibration_crossfit")
    calibration_exclusion_unit = development.get("calibration_exclusion_unit")
    if (
        outer_training_rule != "other_three_development_folds_only"
        or outer_score_rule != "held_out_development_fold_only"
        or calibration_crossfit != "leave_one_union_component_out_within_each_outer_score_fold"
        or calibration_exclusion_unit != "union_component_id"
    ):
        raise ValueError("outer-fold or LOUCO policy changed")
    levels = _positive_int(diffusion.get("levels"), label="diffusion levels")
    offset = diffusion.get("cosine_offset")
    if levels != 64 or type(offset) is not float or offset != 0.008:
        raise ValueError("projection requires the frozen 64-level cosine schedule")
    bins = calibration.get("timestep_bins")
    if tuple(cast(Sequence[object], bins or ())) != (
        "1_to_16",
        "17_to_32",
        "33_to_48",
        "49_to_64",
    ):
        raise ValueError("projection requires the frozen four timestep bins")

    return _ProjectionPolicy(
        config_sha256=_strict_sha256(contract.config_sha256, label="contract sha256"),
        development_folds=folds,
        locked_holdout_fold=contract.locked_holdout_fold,
        training_projection_sha256=_strict_sha256(
            input_table.get("training_projection_sha256"),
            label="training projection sha256",
        ),
        union_assignments_sha256=_strict_sha256(
            input_table.get("union_assignments_sha256"),
            label="union assignments sha256",
        ),
        expected_sequences=expected_sequences,
        expected_development_sequences=expected_development,
        expected_locked_holdout_sequences=expected_locked,
        expected_sequences_by_fold=expected_by_fold,
        expected_development_homology_components=_positive_int(
            input_table.get("expected_development_homology_components"),
            label="development homology components",
        ),
        expected_locked_holdout_homology_components=_positive_int(
            input_table.get("expected_locked_holdout_homology_components"),
            label="locked homology components",
        ),
        expected_development_union_components=_positive_int(
            input_table.get("expected_development_union_components"),
            label="development union components",
        ),
        expected_locked_holdout_union_components=_positive_int(
            input_table.get("expected_locked_holdout_union_components"),
            label="locked union components",
        ),
        expected_outer_training_sequences=expected_train,
        expected_outer_score_sequences=expected_score,
        expected_outer_training_homology_components=_integer_tuple(
            development.get("expected_outer_training_homology_components"),
            label="outer training homology components",
            size=fold_count,
        ),
        expected_outer_score_homology_components=_integer_tuple(
            development.get("expected_outer_score_homology_components"),
            label="outer score homology components",
            size=fold_count,
        ),
        expected_outer_training_union_components=_integer_tuple(
            development.get("expected_outer_training_union_components"),
            label="outer training union components",
            size=fold_count,
        ),
        expected_outer_score_union_components=_integer_tuple(
            development.get("expected_outer_score_union_components"),
            label="outer score union components",
            size=fold_count,
        ),
        expected_score_selected_tokens=_integer_tuple(
            development.get("expected_score_selected_tokens_per_replicate_by_fold"),
            label="score selected tokens",
            size=fold_count,
        ),
        expected_score_selected_tokens_by_bin=_integer_matrix(
            development.get("expected_score_selected_tokens_per_replicate_by_fold_bin"),
            label="score selected tokens by bin",
            rows=fold_count,
            columns=4,
        ),
        minimum_louco_union_components=_integer_tuple(
            development.get("minimum_louco_union_components_by_fold"),
            label="minimum LOUCO union components",
            size=fold_count,
        ),
        minimum_louco_homology_components=_integer_tuple(
            development.get("minimum_louco_homology_components_by_fold"),
            label="minimum LOUCO homology components",
            size=fold_count,
        ),
        minimum_louco_selected_tokens=_integer_tuple(
            development.get("minimum_louco_selected_tokens_per_replicate_by_fold"),
            label="minimum LOUCO selected tokens",
            size=fold_count,
        ),
        minimum_louco_selected_tokens_by_bin=_integer_matrix(
            development.get("minimum_louco_selected_tokens_per_replicate_by_fold_bin"),
            label="minimum LOUCO selected tokens by bin",
            rows=fold_count,
            columns=4,
        ),
        total_levels=levels,
        cosine_offset=offset,
        selected_token_count_definition=cast(str, selected_definition),
        component_weighting=cast(str, component_weighting),
        outer_training_rule=cast(str, outer_training_rule),
        outer_score_rule=cast(str, outer_score_rule),
        calibration_crossfit=cast(str, calibration_crossfit),
        calibration_exclusion_unit=cast(str, calibration_exclusion_unit),
    )


def _validate_assignments(
    payload: bytes,
    *,
    policy: _ProjectionPolicy,
) -> tuple[SequenceAssignment, ...]:
    documents = _parse_canonical_jsonl(payload, label="union assignments")
    if len(documents) != policy.expected_sequences:
        raise ValueError(
            f"union assignments expected {policy.expected_sequences} rows, got {len(documents)}"
        )
    rows: list[SequenceAssignment] = []
    for number, document in enumerate(documents, start=1):
        label = f"union assignment row {number}"
        if set(document) != ASSIGNMENT_FIELDS:
            raise ValueError(
                f"{label} schema mismatch: missing={sorted(ASSIGNMENT_FIELDS - set(document))}, "
                f"extra={sorted(set(document) - ASSIGNMENT_FIELDS)}"
            )
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError(f"{label} schema_version must be integer 1")
        fold = document["fold"]
        if type(fold) is not int or fold not in (
            *policy.development_folds,
            policy.locked_holdout_fold,
        ):
            raise ValueError(f"{label} fold must be an integer in 0..4")
        rows.append(
            SequenceAssignment(
                sequence_id=_strict_sha256(document["sequence_id"], label=f"{label} sequence_id"),
                homology_component_id=_strict_sha256(
                    document["homology_component_id"], label=f"{label} homology_component_id"
                ),
                union_component_id=_strict_sha256(
                    document["union_component_id"], label=f"{label} union_component_id"
                ),
                fold=fold,
            )
        )
    sequence_ids = tuple(row.sequence_id for row in rows)
    if sequence_ids != tuple(sorted(sequence_ids)) or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("union assignments must be strictly ordered by sequence_id")

    fold_counts = Counter(row.fold for row in rows)
    observed_by_fold = tuple(fold_counts[fold] for fold in (*policy.development_folds, 4))
    if observed_by_fold != policy.expected_sequences_by_fold:
        raise ValueError("union assignment fold census differs from the contract")

    homology_owner: dict[str, tuple[str, int]] = {}
    union_owner: dict[str, int] = {}
    for row in rows:
        owner = (row.union_component_id, row.fold)
        if homology_owner.setdefault(row.homology_component_id, owner) != owner:
            raise ValueError("a homology component crosses a union or fold boundary")
        if union_owner.setdefault(row.union_component_id, row.fold) != row.fold:
            raise ValueError("a union component crosses a fold boundary")
    return tuple(rows)


def _selected_token_counts(
    length: int,
    *,
    schedule: CosineMaskSchedule,
    total_levels: int,
) -> tuple[int, tuple[int, int, int, int]]:
    levels = np.arange(1, total_levels + 1, dtype=np.int64)
    lengths = np.full(total_levels, length, dtype=np.int64)
    counts = schedule.mask_counts(lengths, levels, total_levels=total_levels)
    bins = tuple(
        int(np.sum(counts[start : start + 16], dtype=np.int64)) for start in range(0, 64, 16)
    )
    return sum(bins), cast(tuple[int, int, int, int], bins)


def _sequence_ids_sha256(sequence_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sequence_id in sequence_ids:
        digest.update(sequence_id.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _role_weights(
    rows: Sequence[tuple[TrainingRow, SequenceAssignment]],
) -> tuple[dict[str, float], int, int]:
    component_sizes = Counter(assignment.homology_component_id for _, assignment in rows)
    homology_count = len(component_sizes)
    union_count = len({assignment.union_component_id for _, assignment in rows})
    if not rows or not homology_count or not union_count:
        raise ValueError("every outer role must contain rows and components")
    weights = {
        row.sequence_id: 1.0 / (homology_count * component_sizes[assignment.homology_component_id])
        for row, assignment in rows
    }
    if math.fsum(weights[row.sequence_id] for row, _ in rows).hex() != (1.0).hex():
        raise ValueError("role-local component-equal weights do not sum exactly to one")
    expected_mass = 1.0 / homology_count
    component_mass: dict[str, list[float]] = defaultdict(list)
    for row, assignment in rows:
        component_mass[assignment.homology_component_id].append(weights[row.sequence_id])
    if any(
        not math.isclose(math.fsum(values), expected_mass, rel_tol=0.0, abs_tol=1e-15)
        for values in component_mass.values()
    ):
        raise ValueError("role-local homology components do not have equal mass")
    return weights, homology_count, union_count


def _derive_from_rows(
    training_rows: Sequence[TrainingRow],
    assignments: Sequence[SequenceAssignment],
    *,
    policy: _ProjectionPolicy,
) -> DevelopmentProjectionSet:
    assignment_by_id = {row.sequence_id: row for row in assignments}
    development_assignments = tuple(
        row for row in assignments if row.fold in policy.development_folds
    )
    locked_assignments = tuple(row for row in assignments if row.fold == policy.locked_holdout_fold)
    training_ids = tuple(row.sequence_id for row in training_rows)
    development_ids = tuple(row.sequence_id for row in development_assignments)
    if training_ids != development_ids:
        missing = sorted(set(development_ids) - set(training_ids))
        extra = sorted(set(training_ids) - set(development_ids))
        raise ValueError(
            "training projection is not exactly folds 0..3: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if len(locked_assignments) != policy.expected_locked_holdout_sequences:
        raise ValueError("locked-fold assignment census differs from the contract")

    development_homology = {row.homology_component_id for row in development_assignments}
    locked_homology = {row.homology_component_id for row in locked_assignments}
    development_union = {row.union_component_id for row in development_assignments}
    locked_union = {row.union_component_id for row in locked_assignments}
    if development_homology & locked_homology or development_union & locked_union:
        raise ValueError("a component crosses the development/locked boundary")
    if (
        len(development_homology) != policy.expected_development_homology_components
        or len(locked_homology) != policy.expected_locked_holdout_homology_components
        or len(development_union) != policy.expected_development_union_components
        or len(locked_union) != policy.expected_locked_holdout_union_components
    ):
        raise ValueError("development or locked component census differs from the contract")

    source_component_sizes = Counter(
        assignment_by_id[row.sequence_id].homology_component_id for row in training_rows
    )
    for row in training_rows:
        assignment = assignment_by_id[row.sequence_id]
        expected = 1.0 / (
            policy.expected_development_homology_components
            * source_component_sizes[assignment.homology_component_id]
        )
        if not math.isclose(row.sampling_weight, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("accepted projection weight is not development-component-equal")
    if math.fsum(row.sampling_weight for row in training_rows).hex() != (1.0).hex():
        raise ValueError("accepted projection weights do not sum exactly to one")

    schedule = CosineMaskSchedule(policy.cosine_offset)
    count_cache = {
        length: _selected_token_counts(
            length,
            schedule=schedule,
            total_levels=policy.total_levels,
        )
        for length in sorted({len(row.sequence) for row in training_rows})
    }
    joined = tuple((row, assignment_by_id[row.sequence_id]) for row in training_rows)
    outer: list[OuterProjection] = []
    for index, fold in enumerate(policy.development_folds):
        train_source = tuple(item for item in joined if item[1].fold != fold)
        score_source = tuple(item for item in joined if item[1].fold == fold)
        if any(item[1].fold == policy.locked_holdout_fold for item in train_source + score_source):
            raise ValueError("fold-4 sequence reached an outer projection")
        train_weights, train_homology, train_union = _role_weights(train_source)
        score_weights, score_homology, score_union = _role_weights(score_source)
        observed = (
            len(train_source),
            len(score_source),
            train_homology,
            score_homology,
            train_union,
            score_union,
        )
        expected = (
            policy.expected_outer_training_sequences[index],
            policy.expected_outer_score_sequences[index],
            policy.expected_outer_training_homology_components[index],
            policy.expected_outer_score_homology_components[index],
            policy.expected_outer_training_union_components[index],
            policy.expected_outer_score_union_components[index],
        )
        if observed != expected:
            raise ValueError(f"outer fold {fold} census differs from the contract")

        train_rows_out = tuple(
            TrainingRow(row.sequence_id, row.sequence, train_weights[row.sequence_id])
            for row, _ in train_source
        )
        score_rows_out = tuple(
            OuterScoreRow(
                schema_version=1,
                sequence_id=row.sequence_id,
                sequence=row.sequence,
                fold=fold,
                homology_component_id=assignment.homology_component_id,
                union_component_id=assignment.union_component_id,
                sampling_weight=score_weights[row.sequence_id],
            )
            for row, assignment in score_source
        )

        selected_bins = tuple(
            sum(count_cache[len(row.sequence)][1][bin_index] for row, _ in score_source)
            for bin_index in range(4)
        )
        selected_total = sum(selected_bins)
        if (
            selected_total != policy.expected_score_selected_tokens[index]
            or selected_bins != policy.expected_score_selected_tokens_by_bin[index]
        ):
            raise ValueError(f"outer fold {fold} selected-token census differs from the contract")

        union_ids = sorted({assignment.union_component_id for _, assignment in score_source})
        if len(union_ids) < 2:
            raise ValueError(f"outer fold {fold} needs at least two union components for LOUCO")
        remaining_union: list[int] = []
        remaining_homology: list[int] = []
        remaining_total: list[int] = []
        remaining_bins: list[tuple[int, int, int, int]] = []
        for excluded_union in union_ids:
            remaining = tuple(
                item for item in score_source if item[1].union_component_id != excluded_union
            )
            remaining_union.append(len({item[1].union_component_id for item in remaining}))
            remaining_homology.append(len({item[1].homology_component_id for item in remaining}))
            bins_after_exclusion = cast(
                tuple[int, int, int, int],
                tuple(
                    sum(count_cache[len(row.sequence)][1][bin_index] for row, _ in remaining)
                    for bin_index in range(4)
                ),
            )
            remaining_bins.append(bins_after_exclusion)
            remaining_total.append(sum(bins_after_exclusion))
        louco_bins = cast(
            tuple[int, int, int, int],
            tuple(min(values[index] for values in remaining_bins) for index in range(4)),
        )
        louco = (
            min(remaining_union),
            min(remaining_homology),
            min(remaining_total),
            louco_bins,
        )
        expected_louco = (
            policy.minimum_louco_union_components[index],
            policy.minimum_louco_homology_components[index],
            policy.minimum_louco_selected_tokens[index],
            policy.minimum_louco_selected_tokens_by_bin[index],
        )
        if louco != expected_louco:
            raise ValueError(f"outer fold {fold} minimum LOUCO support differs from the contract")

        outer.append(
            OuterProjection(
                fold=fold,
                train_folds=tuple(item for item in policy.development_folds if item != fold),
                train_rows=train_rows_out,
                score_rows=score_rows_out,
                train_homology_components=train_homology,
                score_homology_components=score_homology,
                train_union_components=train_union,
                score_union_components=score_union,
                train_sequence_ids_sha256=_sequence_ids_sha256(
                    tuple(row.sequence_id for row in train_rows_out)
                ),
                score_sequence_ids_sha256=_sequence_ids_sha256(
                    tuple(row.sequence_id for row in score_rows_out)
                ),
                selected_tokens_per_replicate=selected_total,
                selected_tokens_per_replicate_by_timestep_bin=selected_bins,
                louco_minimum_union_components=louco[0],
                louco_minimum_homology_components=louco[1],
                louco_minimum_selected_tokens_per_replicate=louco[2],
                louco_minimum_selected_tokens_per_replicate_by_timestep_bin=louco[3],
            )
        )

    score_occurrences = Counter(
        row.sequence_id for projection in outer for row in projection.score_rows
    )
    train_occurrences = Counter(
        row.sequence_id for projection in outer for row in projection.train_rows
    )
    if set(score_occurrences) != set(training_ids) or set(score_occurrences.values()) != {1}:
        raise ValueError("every development sequence must be scored exactly once")
    if set(train_occurrences) != set(training_ids) or set(train_occurrences.values()) != {3}:
        raise ValueError("every development sequence must train exactly three outer fits")
    return DevelopmentProjectionSet(
        parent_contract_sha256=policy.config_sha256,
        development_sequences=len(training_rows),
        development_homology_components=len(development_homology),
        development_union_components=len(development_union),
        outer_folds=tuple(outer),
    )


def _load_projection_inputs(
    *,
    training_projection_path: str | os.PathLike[str],
    union_assignments_path: str | os.PathLike[str],
    contract: NativeDiffusionV1Contract,
) -> tuple[DevelopmentProjectionSet, _InputSnapshot, _InputSnapshot]:
    policy = _policy_from_contract(contract)
    training_snapshot = _read_snapshot(training_projection_path, label="training projection")
    if training_snapshot.sha256 != policy.training_projection_sha256:
        raise ValueError("training projection SHA-256 differs from the v1 contract")
    assignments_snapshot = _read_snapshot(union_assignments_path, label="union assignments")
    if assignments_snapshot.sha256 != policy.union_assignments_sha256:
        raise ValueError("union assignments SHA-256 differs from the v1 contract")
    distribution = load_training_projection(
        training_snapshot.path,
        expected_sha256=policy.training_projection_sha256,
        expected_rows=policy.expected_development_sequences,
    )
    assignments = _validate_assignments(assignments_snapshot.payload, policy=policy)
    _assert_snapshot_unchanged(training_snapshot, label="training projection")
    _assert_snapshot_unchanged(assignments_snapshot, label="union assignments")
    return (
        _derive_from_rows(distribution.rows, assignments, policy=policy),
        training_snapshot,
        assignments_snapshot,
    )


def derive_development_projections(
    *,
    training_projection_path: str | os.PathLike[str],
    union_assignments_path: str | os.PathLike[str],
    contract: NativeDiffusionV1Contract,
) -> DevelopmentProjectionSet:
    """Derive four outer projections without any fold-4 sequence source."""

    projections, _, _ = _load_projection_inputs(
        training_projection_path=training_projection_path,
        union_assignments_path=union_assignments_path,
        contract=contract,
    )
    return projections


def _training_row_document(row: TrainingRow) -> dict[str, object]:
    return {
        "sequence_id": row.sequence_id,
        "sequence": row.sequence,
        "sampling_weight": row.sampling_weight,
    }


def _score_row_document(row: OuterScoreRow) -> dict[str, object]:
    return {
        "schema_version": row.schema_version,
        "sequence_id": row.sequence_id,
        "sequence": row.sequence,
        "fold": row.fold,
        "homology_component_id": row.homology_component_id,
        "union_component_id": row.union_component_id,
        "sampling_weight": row.sampling_weight,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("projection JSONL cannot be empty")
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _summary_document(projections: DevelopmentProjectionSet) -> dict[str, object]:
    folds: list[dict[str, object]] = []
    for projection in projections.outer_folds:
        folds.append(
            {
                "fold": projection.fold,
                "train": {
                    "folds": list(projection.train_folds),
                    "rows": len(projection.train_rows),
                    "homology_components": projection.train_homology_components,
                    "union_components": projection.train_union_components,
                    "sequence_ids_sha256": projection.train_sequence_ids_sha256,
                    "sampling_weight_sum_hex": math.fsum(
                        row.sampling_weight for row in projection.train_rows
                    ).hex(),
                },
                "score": {
                    "rows": len(projection.score_rows),
                    "homology_components": projection.score_homology_components,
                    "union_components": projection.score_union_components,
                    "sequence_ids_sha256": projection.score_sequence_ids_sha256,
                    "sampling_weight_sum_hex": math.fsum(
                        row.sampling_weight for row in projection.score_rows
                    ).hex(),
                    "selected_tokens_per_replicate": (projection.selected_tokens_per_replicate),
                    "selected_tokens_per_replicate_by_timestep_bin": list(
                        projection.selected_tokens_per_replicate_by_timestep_bin
                    ),
                },
                "louco_minimum": {
                    "union_components": projection.louco_minimum_union_components,
                    "homology_components": projection.louco_minimum_homology_components,
                    "selected_tokens_per_replicate": (
                        projection.louco_minimum_selected_tokens_per_replicate
                    ),
                    "selected_tokens_per_replicate_by_timestep_bin": list(
                        projection.louco_minimum_selected_tokens_per_replicate_by_timestep_bin
                    ),
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": STATUS,
        "parent_contract_sha256": projections.parent_contract_sha256,
        "coverage": {
            "development_sequences": projections.development_sequences,
            "fold4_sequence_rows": 0,
            "score_occurrences_per_sequence": 1,
            "train_occurrences_per_sequence": 3,
        },
        "folds": folds,
    }


def _checksum_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    paths = tuple(sorted(entries))
    for path in paths:
        digest = entries[path]
        _strict_sha256(digest, label=f"checksum for {path}")
        pure = PurePosixPath(path)
        if (
            not path
            or pure.is_absolute()
            or path != pure.as_posix()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(f"unsafe checksum-manifest path: {path!r}")
    return "".join(f"{entries[path]}  {path}\n" for path in paths).encode("utf-8")


def _parse_code_manifest(payload: bytes) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("CODE_SHA256SUMS must be non-empty LF-terminated text")
    entries: dict[str, str] = {}
    for number, raw_line in enumerate(payload[:-1].split(b"\n"), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"CODE_SHA256SUMS line {number} is not UTF-8") from error
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"CODE_SHA256SUMS line {number} is malformed")
        digest = _strict_sha256(line[:64], label=f"CODE_SHA256SUMS line {number}")
        path = line[66:]
        if path in entries:
            raise ValueError(f"CODE_SHA256SUMS repeats path: {path}")
        entries[path] = digest
    canonical = _checksum_manifest_bytes(entries)
    if canonical != payload:
        raise ValueError("CODE_SHA256SUMS is not canonical or strictly C-sorted")
    if tuple(entries) != CODE_MANIFEST_PATHS:
        missing = sorted(set(CODE_MANIFEST_PATHS) - set(entries))
        extra = sorted(set(entries) - set(CODE_MANIFEST_PATHS))
        raise ValueError(f"CODE_SHA256SUMS path set mismatch: missing={missing}, extra={extra}")
    return entries


def _fold_payloads(projections: DevelopmentProjectionSet) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for projection in projections.outer_folds:
        payloads[f"folds/{projection.fold}/train.jsonl"] = _jsonl_bytes(
            tuple(_training_row_document(row) for row in projection.train_rows)
        )
        payloads[f"folds/{projection.fold}/score.jsonl"] = _jsonl_bytes(
            tuple(_score_row_document(row) for row in projection.score_rows)
        )
    return payloads


def _manifest_document(
    *,
    projections: DevelopmentProjectionSet,
    policy: _ProjectionPolicy,
    training_projection_sha256: str,
    union_assignments_sha256: str,
    git_commit: str,
    code_manifest_sha256: str,
    frozen_manifest_sha256: str,
    summary_sha256: str,
    fold_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    fold_artifacts: list[dict[str, object]] = []
    for projection in projections.outer_folds:
        train_name = f"folds/{projection.fold}/train.jsonl"
        score_name = f"folds/{projection.fold}/score.jsonl"
        fold_artifacts.append(
            {
                "fold": projection.fold,
                "train": {
                    "filename": train_name,
                    "role": "trainer_input",
                    "rows": len(projection.train_rows),
                    "fields": list(TRAIN_FIELDS),
                    "sha256": _sha256_bytes(fold_payloads[train_name]),
                },
                "score": {
                    "filename": score_name,
                    "role": "evaluator_score_ledger",
                    "rows": len(projection.score_rows),
                    "fields": list(SCORE_FIELDS),
                    "sha256": _sha256_bytes(fold_payloads[score_name]),
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": STATUS,
        "parent_contract_sha256": projections.parent_contract_sha256,
        "input": {
            "training_projection": {
                "rows": projections.development_sequences,
                "sha256": training_projection_sha256,
            },
            "union_assignments": {
                "rows": policy.expected_sequences,
                "sha256": union_assignments_sha256,
            },
        },
        "provenance": {
            "git_commit": git_commit,
            "code_sha256sums_sha256": code_manifest_sha256,
            "frozen_input_sha256sums_sha256": frozen_manifest_sha256,
        },
        "counts": {
            "development_sequences": projections.development_sequences,
            "development_homology_components": projections.development_homology_components,
            "development_union_components": projections.development_union_components,
            "outer_training_sequences": [len(item.train_rows) for item in projections.outer_folds],
            "outer_score_sequences": [len(item.score_rows) for item in projections.outer_folds],
            "outer_training_homology_components": [
                item.train_homology_components for item in projections.outer_folds
            ],
            "outer_score_homology_components": [
                item.score_homology_components for item in projections.outer_folds
            ],
            "outer_training_union_components": [
                item.train_union_components for item in projections.outer_folds
            ],
            "outer_score_union_components": [
                item.score_union_components for item in projections.outer_folds
            ],
        },
        "policies": {
            "development_folds": list(policy.development_folds),
            "locked_holdout_fold": policy.locked_holdout_fold,
            "outer_training_rule": policy.outer_training_rule,
            "outer_score_rule": policy.outer_score_rule,
            "row_order": "ascending_sequence_id",
            "trainer_visible_fields": list(TRAIN_FIELDS),
            "score_visible_fields": list(SCORE_FIELDS),
            "component_weighting": policy.component_weighting,
            "role_weight_formula": ROLE_WEIGHT_FORMULA,
            "calibration_crossfit": policy.calibration_crossfit,
            "calibration_exclusion_unit": policy.calibration_exclusion_unit,
            "selected_token_count_definition": policy.selected_token_count_definition,
            "fold4_assignment_metadata_input": True,
            "fold4_sequence_input": False,
            "serialization": SERIALIZATION,
        },
        "artifacts": {
            "summary": {
                "filename": "summary.json",
                "role": "semantic_summary",
                "sha256": summary_sha256,
            },
            "folds": fold_artifacts,
        },
    }


def _bundle_payloads(
    *,
    projections: DevelopmentProjectionSet,
    policy: _ProjectionPolicy,
    contract_sha256: str,
    training_projection_sha256: str,
    union_assignments_sha256: str,
    code_manifest: bytes,
    git_commit: str,
) -> dict[str, bytes]:
    fold_payloads = _fold_payloads(projections)
    summary_payload = _canonical_json_bytes(_summary_document(projections))
    frozen_payload = _checksum_manifest_bytes(
        {
            "unconditional_v1.toml": contract_sha256,
            "training_projection.jsonl": training_projection_sha256,
            "sequence_assignments.jsonl": union_assignments_sha256,
        }
    )
    payloads = {
        "CODE_SHA256SUMS": code_manifest,
        "FROZEN_INPUT_SHA256SUMS": frozen_payload,
        **fold_payloads,
        "summary.json": summary_payload,
    }
    manifest = _manifest_document(
        projections=projections,
        policy=policy,
        training_projection_sha256=training_projection_sha256,
        union_assignments_sha256=union_assignments_sha256,
        git_commit=git_commit,
        code_manifest_sha256=_sha256_bytes(code_manifest),
        frozen_manifest_sha256=_sha256_bytes(frozen_payload),
        summary_sha256=_sha256_bytes(summary_payload),
        fold_payloads=fold_payloads,
    )
    payloads["manifest.json"] = _canonical_json_bytes(manifest)
    payloads["SHA256SUMS"] = _checksum_manifest_bytes(
        {path: _sha256_bytes(payload) for path, payload in payloads.items()}
    )
    return payloads


def _write_read_only(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for atomic no-replace publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _remove_owned_tree(path: Path, *, expected_inode: tuple[int, int]) -> None:
    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(observed.st_mode) or _inode(observed) != expected_inode:
        return
    for directory, child_directories, _ in os.walk(path, topdown=True, followlinks=False):
        current = Path(directory)
        with contextlib.suppress(OSError):
            current.chmod(0o700)
        for child in child_directories:
            candidate = current / child
            if not candidate.is_symlink():
                with contextlib.suppress(OSError):
                    candidate.chmod(0o700)
    shutil.rmtree(path)


def _publish_with_commit_marker(
    *,
    staging: Path,
    output: Path,
    payloads: Mapping[str, bytes],
    snapshots: Sequence[tuple[str, _InputSnapshot]],
) -> tuple[int, int]:
    """Publish safely where the filesystem lacks ``RENAME_NOREPLACE``.

    The output directory is exclusively created, every immutable regular file
    except the top manifest is linked, and ``SHA256SUMS`` is linked last as the
    atomic commit marker. Consumers must reject a tree without that marker.
    """

    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise FileExistsError(
            f"projection output appeared during construction: {output}"
        ) from error
    output_inode = _inode(os.lstat(output))
    committed = False
    try:
        folds = output / "folds"
        folds.mkdir(mode=0o700)
        for fold in range(4):
            (folds / str(fold)).mkdir(mode=0o700)
        for relative in sorted(path for path in payloads if path != "SHA256SUMS"):
            source = staging / relative
            destination = output / relative
            os.link(source, destination, follow_symlinks=False)
            if _inode(os.lstat(source)) != _inode(os.lstat(destination)):
                raise RuntimeError(f"published projection link changed: {relative}")
        for fold in range(4):
            directory = folds / str(fold)
            directory.chmod(0o555)
            _fsync_directory(directory)
        folds.chmod(0o555)
        _fsync_directory(folds)
        for label, snapshot in snapshots:
            _assert_snapshot_unchanged(snapshot, label=label)

        # This single directory entry changes the tree from partial to committed.
        os.link(staging / "SHA256SUMS", output / "SHA256SUMS", follow_symlinks=False)
        output.chmod(0o555)
        _fsync_directory(output)
        observed_files = {
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        }
        if observed_files != set(payloads):
            raise RuntimeError("published projection tree inventory mismatch")
        for relative, payload in payloads.items():
            path = output / relative
            observed = os.lstat(path)
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o444
                or _sha256_bytes(path.read_bytes()) != _sha256_bytes(payload)
            ):
                raise RuntimeError(f"published projection artifact changed: {relative}")
        for label, snapshot in snapshots:
            _assert_snapshot_unchanged(snapshot, label=label)
        committed = True
        return output_inode
    finally:
        if not committed:
            _remove_owned_tree(output, expected_inode=output_inode)


def _publish_payloads(
    *,
    payloads: Mapping[str, bytes],
    output_dir: str | os.PathLike[str],
    snapshots: Sequence[tuple[str, _InputSnapshot]],
) -> Path:
    output = _absolute(output_dir)
    parent = output.parent
    _reject_symlink_chain(parent, label="projection output parent")
    if not parent.is_dir():
        raise ValueError("projection output parent must be an existing directory")
    _reject_symlink_chain(output, label="projection output")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse projection output: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=parent))
    staging_inode = _inode(os.lstat(staging))
    published = False
    published_inode = staging_inode
    try:
        for fold in range(4):
            (staging / "folds" / str(fold)).mkdir(parents=True, mode=0o700)
        expected_paths = {
            "CODE_SHA256SUMS",
            "FROZEN_INPUT_SHA256SUMS",
            "SHA256SUMS",
            "manifest.json",
            "summary.json",
            *(f"folds/{fold}/{role}.jsonl" for fold in range(4) for role in ("train", "score")),
        }
        if set(payloads) != expected_paths:
            raise RuntimeError("projection payload inventory differs from the 13-file contract")
        for relative in sorted(path for path in payloads if path != "SHA256SUMS"):
            _write_read_only(staging / relative, payloads[relative])
        # The top checksum manifest is always the final staged file.
        _write_read_only(staging / "SHA256SUMS", payloads["SHA256SUMS"])
        observed_files = {
            path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()
        }
        observed_directories = {
            path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_dir()
        }
        if observed_files != expected_paths or observed_directories != {
            "folds",
            "folds/0",
            "folds/1",
            "folds/2",
            "folds/3",
        }:
            raise RuntimeError("staged projection tree inventory mismatch")
        for relative, payload in payloads.items():
            path = staging / relative
            observed = os.lstat(path)
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o444
                or _sha256_bytes(path.read_bytes()) != _sha256_bytes(payload)
            ):
                raise RuntimeError(f"staged projection artifact changed: {relative}")
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
            _fsync_directory(directory)
        staging.chmod(0o555)
        _fsync_directory(staging)
        for label, snapshot in snapshots:
            _assert_snapshot_unchanged(snapshot, label=label)
        if os.path.lexists(output):
            raise FileExistsError(f"projection output appeared during construction: {output}")
        try:
            _rename_noreplace(staging, output)
            published = True
        except OSError as error:
            unsupported = {errno.EINVAL, errno.ENOSYS, getattr(errno, "EOPNOTSUPP", errno.ENOSYS)}
            if error.errno not in unsupported:
                raise
            published_inode = _publish_with_commit_marker(
                staging=staging,
                output=output,
                payloads=payloads,
                snapshots=snapshots,
            )
            published = True
            _remove_owned_tree(staging, expected_inode=staging_inode)
        _fsync_directory(parent)
        observed = os.lstat(output)
        if not stat.S_ISDIR(observed.st_mode) or _inode(observed) != published_inode:
            raise RuntimeError("published projection directory identity changed")
        for relative, payload in payloads.items():
            artifact = output / relative
            observed_artifact = os.lstat(artifact)
            if (
                not stat.S_ISREG(observed_artifact.st_mode)
                or stat.S_IMODE(observed_artifact.st_mode) != 0o444
                or observed_artifact.st_nlink != 1
                or _sha256_bytes(artifact.read_bytes()) != _sha256_bytes(payload)
            ):
                raise RuntimeError(f"final projection artifact changed: {relative}")
        for label, snapshot in snapshots:
            _assert_snapshot_unchanged(snapshot, label=label)
        return output
    except BaseException:
        candidate = output if published else staging
        expected_inode = published_inode if published else staging_inode
        _remove_owned_tree(candidate, expected_inode=expected_inode)
        with contextlib.suppress(OSError):
            _fsync_directory(parent)
        raise


def build_development_projection_bundle(
    *,
    contract_path: str | os.PathLike[str],
    training_projection_path: str | os.PathLike[str],
    union_assignments_path: str | os.PathLike[str],
    code_manifest_path: str | os.PathLike[str],
    git_commit: str,
    output_dir: str | os.PathLike[str],
) -> ProjectionBundleExecution:
    """Authenticate inputs and atomically publish the exact 13-file bundle."""

    if type(git_commit) is not str or _GIT_COMMIT_RE.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character digest")
    contract_snapshot = _read_snapshot(contract_path, label="v1 contract")
    contract = load_unconditional_v1_contract(contract_snapshot.path)
    policy = _policy_from_contract(contract)
    if contract_snapshot.sha256 != policy.config_sha256:
        raise ValueError("contract snapshot differs from the authenticated v1 contract")
    projections, training_snapshot, assignments_snapshot = _load_projection_inputs(
        training_projection_path=training_projection_path,
        union_assignments_path=union_assignments_path,
        contract=contract,
    )
    code_snapshot = _read_snapshot(code_manifest_path, label="CODE_SHA256SUMS")
    _parse_code_manifest(code_snapshot.payload)
    snapshots = (
        ("v1 contract", contract_snapshot),
        ("training projection", training_snapshot),
        ("union assignments", assignments_snapshot),
        ("CODE_SHA256SUMS", code_snapshot),
    )
    for label, snapshot in snapshots:
        _assert_snapshot_unchanged(snapshot, label=label)
    payloads = _bundle_payloads(
        projections=projections,
        policy=policy,
        contract_sha256=contract_snapshot.sha256,
        training_projection_sha256=training_snapshot.sha256,
        union_assignments_sha256=assignments_snapshot.sha256,
        code_manifest=code_snapshot.payload,
        git_commit=git_commit,
    )
    published = _publish_payloads(
        payloads=payloads,
        output_dir=output_dir,
        snapshots=snapshots,
    )
    return ProjectionBundleExecution(
        output_dir=published,
        manifest_sha256=_sha256_bytes(payloads["manifest.json"]),
        summary_sha256=_sha256_bytes(payloads["summary.json"]),
        sha256sums_sha256=_sha256_bytes(payloads["SHA256SUMS"]),
        rows=projections.development_sequences,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-projection", type=Path, required=True)
    parser.add_argument("--union-assignments", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = build_development_projection_bundle(
        contract_path=args.contract,
        training_projection_path=args.training_projection,
        union_assignments_path=args.union_assignments,
        code_manifest_path=args.code_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(
        _canonical_json_bytes(
            {
                "manifest_sha256": execution.manifest_sha256,
                "rows": execution.rows,
                "sha256sums_sha256": execution.sha256sums_sha256,
                "status": STATUS,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

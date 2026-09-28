"""Context-level Gate-1 benchmark on the accepted homology-and-study split.

This module is deliberately separate from :mod:`oracle_gate1`: historical
homology-only evidence remains immutable.  The v1 union benchmark consumes the
accepted parser-v7 sequence and endpoint-context artifacts, reuses the accepted
sequence-to-fold map without reassignment, and fits only transparent frozen
baselines.  Study and source-provenance metadata are grouping/audit inputs and
never enter a model feature vector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.models.oracle_baselines import (
    DescriptorLogisticOracle,
    HomologyKnnOracle,
    OracleInput,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import global_sequence_identity

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GramClass = Literal["positive", "negative"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_BENCHMARK = "gate1_context_activity_homology_study_union_v1"
_SPLIT_ARTIFACT = "homology_study_union_split"
_SPLIT_STATUS = "development_split_not_an_untouched_evaluation_panel"
_SPLIT_COMPONENT_POLICY = "full_sequence_homology_union_every_study_key_v1"
_SPLIT_HOMOLOGY_POLICY = "global_alignment_identity_single_link_v1"
_FOLD_POLICY = "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
_LABEL_POLICY = (
    "group every ledger row by assay_context_id before eligibility filtering; retain only "
    "fully bacterial_mic16-eligible contexts with unanimous label, source Gram, and "
    "canonical target"
)
_MODEL_FEATURES = ("sequence", "canonical_target", "gram")
_FORBIDDEN_MODEL_FEATURES = (
    "provenance_id",
    "source_record_id",
    "source_record_key",
    "study_keys",
    "study_status",
    "homology_component_id",
    "union_component_id",
    "fold",
)
_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_LOGICAL_CONFIG_PATH = "configs/benchmarks/oracle_gate1_union_v1.toml"
_EXECUTING_MODULE_PATH = "src/amp_challenge/benchmarks/oracle_gate1_union.py"
_FIXED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/oracle_gate1_union_v1_twins.sbatch",
        "cluster/validate_oracle_gate1_union_output.sh",
        _LOGICAL_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)
_INDEPENDENT_CHECKS = frozenset(
    {
        "artifacts_and_manifest_exact",
        "code_and_input_attestations_valid",
        "executing_source_bound_to_repository",
        "fold_assignment_recomputed",
        "frozen_inputs_config_pinned",
        "homology_and_study_union_recomputed",
        "input_twins_byte_identical",
        "json_and_jsonl_canonical_lf",
        "repository_commit_and_cleanliness_verified",
        "scratch_and_absolute_paths_absent",
        "strict_context_exclusions_recomputed",
        "top_manifests_valid",
        "twins_byte_identical",
    }
)

_SEQUENCE_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_ASSIGNMENT_FIELDS = frozenset(
    {"schema_version", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)
_LEDGER_FIELDS = frozenset(
    {
        "apparent_taxon_mentions",
        "assay_context_id",
        "assay_row_sha256",
        "blood_organism_mapping_version",
        "blood_organism_resolution",
        "canonical_target",
        "composite_language_marker",
        "composite_reason_codes",
        "configured_target_occurrences",
        "context_id",
        "eligible_tasks",
        "endpoint",
        "exclusion_codes",
        "expected_gram",
        "exposure_concentration",
        "gram_resolution",
        "input_line",
        "mapping_status",
        "mapping_version",
        "measurement",
        "measurement_status",
        "mic16_label",
        "mic16_reason",
        "normalized_organism",
        "normalized_target",
        "observation_id",
        "provenance_id",
        "resistance_status",
        "resolved_blood_organism",
        "schema_version",
        "sequence_id",
        "source_assay",
        "source_conditions",
        "source_field",
        "source_gram",
        "source_organism",
        "source_record_id",
        "source_record_key",
        "source_row_number",
        "source_target",
        "source_text",
        "strain_identifier_resolution",
        "strain_identifier_review_ids",
        "strain_level_eligible",
        "study_keys",
        "study_review_codes",
        "study_status",
        "target_domain",
    }
)
_SPLIT_TOP_ENTRIES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "split/audit.json",
        "split/components.jsonl",
        "split/grouping_edges.jsonl",
        "split/manifest.json",
        "split/sequence_assignments.jsonl",
    }
)
_CLASS_METRICS = (
    "gram_negative_mic16_negative",
    "gram_negative_mic16_positive",
    "gram_positive_mic16_negative",
    "gram_positive_mic16_positive",
)


@dataclass(frozen=True, slots=True)
class LogisticSettings:
    l2: float
    max_iterations: int
    tolerance: float
    prior_strength: float


@dataclass(frozen=True, slots=True)
class KnnSettings:
    neighbors: int
    similarity_power: float
    prior_strength: float
    minimum_weight: float


@dataclass(frozen=True, slots=True)
class Gate1UnionConfig:
    path: Path
    schema_version: int
    activity_threshold_um: float
    homology_identity_threshold: float
    folds: int
    seed: int
    bootstrap_replicates: int
    calibration_bins: int
    similarity_bin_edges: tuple[float, ...]
    sequences_sha256: str
    endpoint_context_ledger_sha256: str
    split_assignments_sha256: str
    split_components_sha256: str
    split_manifest_sha256: str
    split_audit_sha256: str
    split_top_manifest_sha256: str
    split_independent_receipt_sha256: str
    expected_sequences: int
    expected_ledger_rows: int
    expected_split_homology_components: int
    expected_split_union_components: int
    expected_examples: int
    expected_source_observations: int
    expected_modeled_sequences: int
    expected_positive_examples: int
    expected_negative_examples: int
    expected_examples_by_fold: tuple[int, ...]
    expected_positive_examples_by_fold: tuple[int, ...]
    expected_negative_examples_by_fold: tuple[int, ...]
    logistic: LogisticSettings
    knn: KnnSettings


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class SequenceAssignment:
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    fold: int


@dataclass(frozen=True, slots=True)
class ContextExample:
    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    canonical_target: str
    gram: GramClass
    label: int
    source_observation_ids: tuple[str, ...]
    fold: int
    homology_component_id: str
    union_component_id: str

    @property
    def model_input(self) -> OracleInput:
        """Return the complete and intentionally narrow model feature vector."""

        return OracleInput(
            sequence=self.sequence,
            strain=self.canonical_target,
            gram=self.gram,
        )


@dataclass(frozen=True, slots=True)
class ContextAuditRow:
    schema_version: int
    assay_context_id: str
    sequence_id: str
    endpoint: str
    source_observations: int
    eligible_source_observations: int
    status: str
    reason_codes: tuple[str, ...]
    example_id: str | None


@dataclass(frozen=True, slots=True)
class ContextDataset:
    examples: tuple[ContextExample, ...]
    audit_rows: tuple[ContextAuditRow, ...]
    census: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class OofPrediction:
    model: str
    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    canonical_target: str
    gram: GramClass
    label: int
    source_observations: int
    fold: int
    homology_component_id: str
    union_component_id: str
    max_train_identity: float
    probability: float


@dataclass(frozen=True, slots=True)
class Gate1UnionExecution:
    output_dir: Path
    context_audit_path: Path
    examples_path: Path
    folds_path: Path
    oof_path: Path
    metrics_path: Path
    split_receipt_path: Path
    manifest_path: Path
    top_manifest_path: Path
    examples: int
    source_observations: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_snapshot(path: str | Path, *, name: str) -> InputSnapshot:
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {requested}")
    source = requested.resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} is not a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    if _fingerprint(before) != _fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{name} changed while it was being read: {source}")
    return InputSnapshot(
        path=source,
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=_fingerprint(before),
    )


def _assert_snapshot_unchanged(snapshot: InputSnapshot, *, name: str) -> None:
    try:
        current = snapshot.path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} disappeared while the benchmark was running") from error
    if not stat.S_ISREG(current.st_mode) or _fingerprint(current) != snapshot.fingerprint:
        raise ValueError(f"{name} changed while the benchmark was running")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_json_object(payload: bytes, *, name: str) -> dict[str, object]:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{name} must use LF framing with exactly one final LF")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return cast(dict[str, object], value)


def _parse_jsonl(payload: bytes, *, name: str) -> list[dict[str, object]]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{name} must be non-empty LF-delimited JSONL with a final LF")
    rows: list[dict[str, object]] = []
    for line_number, raw in enumerate(payload[:-1].split(b"\n"), start=1):
        if not raw:
            raise ValueError(f"{name} line {line_number} is blank")
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{name} line {line_number} is not valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{name} line {line_number} must be a JSON object")
        rows.append(cast(dict[str, object], value))
    return rows


def _parse_sha256_manifest(payload: bytes, *, name: str) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{name} must be non-empty LF-delimited text with a final LF")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    entries: dict[str, str] = {}
    previous: str | None = None
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise ValueError(f"{name} line {line_number} is not a SHA-256 entry")
        digest, mode, filename = match.groups()
        if mode != " ":
            raise ValueError(f"{name} line {line_number} must use text-mode checksum syntax")
        pure = PurePosixPath(filename)
        if pure.is_absolute() or ".." in pure.parts or "\\" in filename:
            raise ValueError(f"{name} line {line_number} has an unsafe path")
        if filename in entries:
            raise ValueError(f"{name} repeats {filename!r}")
        if previous is not None and filename <= previous:
            raise ValueError(f"{name} paths must be strictly sorted")
        entries[filename] = digest
        previous = filename
    return entries


def _validate_code_manifest(
    manifest_snapshot: InputSnapshot,
    *,
    config_snapshot: InputSnapshot,
) -> tuple[dict[str, object], tuple[InputSnapshot, ...]]:
    """Bind the complete production code inventory without consulting Git state."""

    entries = _parse_sha256_manifest(manifest_snapshot.payload, name="code manifest")
    repository_root = Path(__file__).resolve(strict=True).parents[3]
    source_root = repository_root / "src" / "amp_challenge"
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("repository amp_challenge source root is missing or symbolic")
    source_paths: set[str] = set()
    for source in source_root.rglob("*.py"):
        if source.is_symlink():
            raise ValueError(f"repository Python source must not be symbolic: {source}")
        if source.is_file():
            source_paths.add(source.relative_to(repository_root).as_posix())
    expected_paths = set(_FIXED_CODE_PATHS) | source_paths
    if set(entries) != expected_paths:
        raise ValueError(
            "code manifest inventory mismatch: "
            f"missing={sorted(expected_paths - set(entries))}, "
            f"extra={sorted(set(entries) - expected_paths)}"
        )
    if _EXECUTING_MODULE_PATH not in entries:
        raise ValueError("code manifest omits the executing Gate-1 union module")

    repository_snapshots: list[InputSnapshot] = []
    for logical_path in sorted(expected_paths):
        if logical_path == _LOGICAL_CONFIG_PATH:
            actual_sha256 = config_snapshot.sha256
        else:
            snapshot = _read_snapshot(
                repository_root / logical_path,
                name=f"code inventory entry {logical_path}",
            )
            repository_snapshots.append(snapshot)
            actual_sha256 = snapshot.sha256
        if entries[logical_path] != actual_sha256:
            raise ValueError(f"code manifest checksum mismatch for {logical_path}")

    executing_sha256 = entries[_EXECUTING_MODULE_PATH]
    attestation = {
        "schema_version": 1,
        "code_manifest_sha256": manifest_snapshot.sha256,
        "inventory_entries": len(entries),
        "executing_module": {
            "logical_path": _EXECUTING_MODULE_PATH,
            "sha256": executing_sha256,
        },
        "logical_config": {
            "logical_path": _LOGICAL_CONFIG_PATH,
            "sha256": config_snapshot.sha256,
        },
        "launcher_git_cleanliness_policy": (
            "clean committed Git state is enforced by the production Slurm launcher; "
            "the core verifies bytes and intentionally runs no Git subprocess"
        ),
        "invariants": {
            "code_manifest_inventory_exact": True,
            "every_repository_code_entry_hash_verified": True,
            "executing_module_bound_to_manifest": True,
            "supplied_config_bound_to_logical_manifest_entry": True,
        },
    }
    return attestation, tuple(repository_snapshots)


def _require_exact_fields(
    row: Mapping[str, object], *, expected: frozenset[str] | set[str], name: str
) -> None:
    if set(row) != set(expected):
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(set(expected) - set(row))}, "
            f"extra={sorted(set(row) - set(expected))}"
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_number(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return number


def _require_string_list(
    value: object,
    *,
    field: str,
    allow_empty: bool = True,
    require_sorted: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a{' non-empty' if not allow_empty else ''} string array")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
        raise ValueError(f"{field} contains an invalid string")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    if require_sorted and tuple(sorted(result)) != result:
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _expect_hash(snapshot: InputSnapshot, expected: str, *, name: str) -> None:
    if snapshot.sha256 != expected:
        raise ValueError(
            f"{name} checksum differs from the frozen config: "
            f"expected {expected}, got {snapshot.sha256}"
        )


def _config_from_payload(path: Path, payload: bytes) -> Gate1UnionConfig:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("Gate-1 union config must use LF framing with one final LF")
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Gate-1 union config is not valid UTF-8 TOML") from error
    expected = {
        "schema_version",
        "activity_threshold_um",
        "homology_identity_threshold",
        "folds",
        "seed",
        "bootstrap_replicates",
        "calibration_bins",
        "similarity_bin_edges",
        "sequences_sha256",
        "endpoint_context_ledger_sha256",
        "split_assignments_sha256",
        "split_components_sha256",
        "split_manifest_sha256",
        "split_audit_sha256",
        "split_top_manifest_sha256",
        "split_independent_receipt_sha256",
        "expected_sequences",
        "expected_ledger_rows",
        "expected_split_homology_components",
        "expected_split_union_components",
        "expected_examples",
        "expected_source_observations",
        "expected_modeled_sequences",
        "expected_positive_examples",
        "expected_negative_examples",
        "expected_examples_by_fold",
        "expected_positive_examples_by_fold",
        "expected_negative_examples_by_fold",
        "descriptor_logistic",
        "homology_knn",
    }
    _require_exact_fields(raw, expected=expected, name="Gate-1 union config")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("Gate-1 union config schema_version must be 1")
    folds = _require_positive_int(raw["folds"], field="folds")
    if folds < 2:
        raise ValueError("folds must be at least two")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    threshold = _require_number(
        raw["homology_identity_threshold"], field="homology_identity_threshold", positive=True
    )
    if threshold > 1.0:
        raise ValueError("homology_identity_threshold cannot exceed one")
    bins_raw = raw["similarity_bin_edges"]
    if not isinstance(bins_raw, list):
        raise ValueError("similarity_bin_edges must be an array")
    bins = tuple(_require_number(value, field="similarity_bin_edges") for value in bins_raw)
    if len(bins) < 2 or bins[0] != 0.0 or bins[-1] < threshold:
        raise ValueError("similarity_bin_edges must start at zero and cover the threshold")
    if any(not 0.0 <= value <= 1.0 for value in bins) or any(
        right <= left for left, right in itertools.pairwise(bins)
    ):
        raise ValueError("similarity_bin_edges must be strictly increasing values in [0, 1]")
    fold_counts_raw = raw["expected_examples_by_fold"]
    positive_fold_counts_raw = raw["expected_positive_examples_by_fold"]
    negative_fold_counts_raw = raw["expected_negative_examples_by_fold"]

    def fold_count_array(value: object, *, field: str) -> tuple[int, ...]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be an integer array")
        result = tuple(_require_positive_int(item, field=field) for item in value)
        if len(result) != folds:
            raise ValueError(f"{field} length must equal folds")
        return result

    fold_counts = fold_count_array(fold_counts_raw, field="expected_examples_by_fold")
    positive_fold_counts = fold_count_array(
        positive_fold_counts_raw, field="expected_positive_examples_by_fold"
    )
    negative_fold_counts = fold_count_array(
        negative_fold_counts_raw, field="expected_negative_examples_by_fold"
    )
    expected_examples = _require_positive_int(raw["expected_examples"], field="expected_examples")
    expected_positive = _require_positive_int(
        raw["expected_positive_examples"], field="expected_positive_examples"
    )
    expected_negative = _require_positive_int(
        raw["expected_negative_examples"], field="expected_negative_examples"
    )
    if sum(fold_counts) != expected_examples:
        raise ValueError("expected_examples_by_fold does not sum to expected_examples")
    if expected_positive + expected_negative != expected_examples:
        raise ValueError("expected label counts do not sum to expected_examples")
    if sum(positive_fold_counts) != expected_positive:
        raise ValueError("expected_positive_examples_by_fold has the wrong total")
    if sum(negative_fold_counts) != expected_negative:
        raise ValueError("expected_negative_examples_by_fold has the wrong total")
    if any(
        positive + negative != total
        for positive, negative, total in zip(
            positive_fold_counts, negative_fold_counts, fold_counts, strict=True
        )
    ):
        raise ValueError("per-fold positive and negative counts do not sum to fold totals")
    bootstrap = _require_nonnegative_int(raw["bootstrap_replicates"], field="bootstrap_replicates")
    calibration_bins = _require_positive_int(raw["calibration_bins"], field="calibration_bins")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")
    logistic_raw = raw["descriptor_logistic"]
    knn_raw = raw["homology_knn"]
    if not isinstance(logistic_raw, dict) or not isinstance(knn_raw, dict):
        raise ValueError("model settings must be TOML tables")
    _require_exact_fields(
        logistic_raw,
        expected={"l2", "max_iterations", "tolerance", "prior_strength"},
        name="descriptor_logistic",
    )
    _require_exact_fields(
        knn_raw,
        expected={"neighbors", "similarity_power", "prior_strength", "minimum_weight"},
        name="homology_knn",
    )
    logistic = LogisticSettings(
        l2=_require_number(logistic_raw["l2"], field="descriptor_logistic.l2", positive=True),
        max_iterations=_require_positive_int(
            logistic_raw["max_iterations"], field="descriptor_logistic.max_iterations"
        ),
        tolerance=_require_number(
            logistic_raw["tolerance"], field="descriptor_logistic.tolerance", positive=True
        ),
        prior_strength=_require_number(
            logistic_raw["prior_strength"], field="descriptor_logistic.prior_strength"
        ),
    )
    if logistic.prior_strength < 0:
        raise ValueError("descriptor_logistic.prior_strength cannot be negative")
    knn = KnnSettings(
        neighbors=_require_positive_int(knn_raw["neighbors"], field="homology_knn.neighbors"),
        similarity_power=_require_number(
            knn_raw["similarity_power"], field="homology_knn.similarity_power", positive=True
        ),
        prior_strength=_require_number(
            knn_raw["prior_strength"], field="homology_knn.prior_strength"
        ),
        minimum_weight=_require_number(
            knn_raw["minimum_weight"], field="homology_knn.minimum_weight", positive=True
        ),
    )
    if knn.prior_strength < 0:
        raise ValueError("homology_knn.prior_strength cannot be negative")
    activity_threshold = _require_number(
        raw["activity_threshold_um"], field="activity_threshold_um", positive=True
    )
    if activity_threshold != 16.0:
        raise ValueError("the v1 context benchmark is defined only for MIC <= 16 uM")
    return Gate1UnionConfig(
        path=path,
        schema_version=1,
        activity_threshold_um=activity_threshold,
        homology_identity_threshold=threshold,
        folds=folds,
        seed=cast(int, seed),
        bootstrap_replicates=bootstrap,
        calibration_bins=calibration_bins,
        similarity_bin_edges=bins,
        sequences_sha256=_require_sha256(raw["sequences_sha256"], field="sequences_sha256"),
        endpoint_context_ledger_sha256=_require_sha256(
            raw["endpoint_context_ledger_sha256"], field="endpoint_context_ledger_sha256"
        ),
        split_assignments_sha256=_require_sha256(
            raw["split_assignments_sha256"], field="split_assignments_sha256"
        ),
        split_components_sha256=_require_sha256(
            raw["split_components_sha256"], field="split_components_sha256"
        ),
        split_manifest_sha256=_require_sha256(
            raw["split_manifest_sha256"], field="split_manifest_sha256"
        ),
        split_audit_sha256=_require_sha256(raw["split_audit_sha256"], field="split_audit_sha256"),
        split_top_manifest_sha256=_require_sha256(
            raw["split_top_manifest_sha256"], field="split_top_manifest_sha256"
        ),
        split_independent_receipt_sha256=_require_sha256(
            raw["split_independent_receipt_sha256"],
            field="split_independent_receipt_sha256",
        ),
        expected_sequences=_require_positive_int(
            raw["expected_sequences"], field="expected_sequences"
        ),
        expected_ledger_rows=_require_positive_int(
            raw["expected_ledger_rows"], field="expected_ledger_rows"
        ),
        expected_split_homology_components=_require_positive_int(
            raw["expected_split_homology_components"],
            field="expected_split_homology_components",
        ),
        expected_split_union_components=_require_positive_int(
            raw["expected_split_union_components"], field="expected_split_union_components"
        ),
        expected_examples=expected_examples,
        expected_source_observations=_require_positive_int(
            raw["expected_source_observations"], field="expected_source_observations"
        ),
        expected_modeled_sequences=_require_positive_int(
            raw["expected_modeled_sequences"], field="expected_modeled_sequences"
        ),
        expected_positive_examples=expected_positive,
        expected_negative_examples=expected_negative,
        expected_examples_by_fold=fold_counts,
        expected_positive_examples_by_fold=positive_fold_counts,
        expected_negative_examples_by_fold=negative_fold_counts,
        logistic=logistic,
        knn=knn,
    )


def load_config(path: str | Path) -> Gate1UnionConfig:
    snapshot = _read_snapshot(path, name="Gate-1 union config")
    return _config_from_payload(snapshot.path, snapshot.payload)


def _read_sequences(snapshot: InputSnapshot, *, config: Gate1UnionConfig) -> dict[str, str]:
    rows = _parse_jsonl(snapshot.payload, name="parser-v7 sequences")
    if len(rows) != config.expected_sequences:
        raise ValueError("parser-v7 sequence count differs from the frozen config")
    sequences: dict[str, str] = {}
    previous: str | None = None
    for index, row in enumerate(rows, start=1):
        name = f"parser-v7 sequence row {index}"
        _require_exact_fields(row, expected=_SEQUENCE_FIELDS, name=name)
        sequence = canonicalize_sequence(str(row["sequence"]))
        sequence_id = _require_sha256(row["sequence_id"], field=f"{name} sequence_id")
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"{name} sequence_id does not match sequence")
        if not isinstance(row["provenance"], list) or not row["provenance"]:
            raise ValueError(f"{name} provenance must be a non-empty array")
        if sequence_id in sequences:
            raise ValueError(f"{name} repeats sequence_id")
        if previous is not None and sequence_id <= previous:
            raise ValueError("parser-v7 sequences must be strictly sorted by sequence_id")
        sequences[sequence_id] = sequence
        previous = sequence_id
    return sequences


def _read_assignments(
    snapshot: InputSnapshot,
    *,
    config: Gate1UnionConfig,
    sequences: Mapping[str, str],
) -> dict[str, SequenceAssignment]:
    rows = _parse_jsonl(snapshot.payload, name="accepted split assignments")
    if len(rows) != config.expected_sequences:
        raise ValueError("accepted split assignment count differs from the sequence census")
    assignments: dict[str, SequenceAssignment] = {}
    previous: str | None = None
    homology_locations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    union_folds: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        name = f"accepted split assignment row {index}"
        _require_exact_fields(row, expected=_ASSIGNMENT_FIELDS, name=name)
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"{name} schema_version must be 1")
        sequence_id = _require_sha256(row["sequence_id"], field=f"{name} sequence_id")
        homology_id = _require_sha256(
            row["homology_component_id"], field=f"{name} homology_component_id"
        )
        union_id = _require_sha256(row["union_component_id"], field=f"{name} union_component_id")
        fold = row["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < config.folds:
            raise ValueError(f"{name} has an invalid fold")
        if sequence_id not in sequences:
            raise ValueError(f"{name} references an unknown sequence")
        if sequence_id in assignments:
            raise ValueError(f"{name} repeats sequence_id")
        if previous is not None and sequence_id <= previous:
            raise ValueError("accepted split assignments must be sorted by sequence_id")
        assignments[sequence_id] = SequenceAssignment(
            sequence_id=sequence_id,
            homology_component_id=homology_id,
            union_component_id=union_id,
            fold=fold,
        )
        homology_locations[homology_id].add((union_id, fold))
        union_folds[union_id].add(fold)
        previous = sequence_id
    if set(assignments) != set(sequences):
        raise ValueError("accepted split assignments do not exactly cover parser-v7 sequences")
    if any(len(locations) != 1 for locations in homology_locations.values()):
        raise ValueError("one homology component spans union components or folds")
    if any(len(folds) != 1 for folds in union_folds.values()):
        raise ValueError("one union component spans multiple folds")
    if len(homology_locations) != config.expected_split_homology_components:
        raise ValueError("accepted split homology-component census changed")
    if len(union_folds) != config.expected_split_union_components:
        raise ValueError("accepted split union-component census changed")
    if {assignment.fold for assignment in assignments.values()} != set(range(config.folds)):
        raise ValueError("accepted split does not contain every configured fold")
    return assignments


def _sequence_set_sha256(sequence_ids: Iterable[str]) -> str:
    payload = "".join(f"{sequence_id}\n" for sequence_id in sorted(sequence_ids)).encode("ascii")
    return _sha256_bytes(payload)


def _read_components(
    snapshot: InputSnapshot,
    *,
    config: Gate1UnionConfig,
    assignments: Mapping[str, SequenceAssignment],
) -> dict[str, dict[str, object]]:
    rows = _parse_jsonl(snapshot.payload, name="accepted split components")
    if len(rows) != config.expected_split_union_components:
        raise ValueError("accepted split component count differs from the frozen config")
    expected_fields = frozenset(
        {
            "schema_version",
            "union_component_id",
            "fold",
            "sequence_count",
            "sequence_ids_sha256",
            "grouping_edge_ids",
            "homology_component_count",
            "study_key_count",
            "balance_counts",
        }
    )
    members: dict[str, list[SequenceAssignment]] = defaultdict(list)
    for assignment in assignments.values():
        members[assignment.union_component_id].append(assignment)
    components: dict[str, dict[str, object]] = {}
    previous: str | None = None
    for index, row in enumerate(rows, start=1):
        name = f"accepted split component row {index}"
        _require_exact_fields(row, expected=expected_fields, name=name)
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"{name} schema_version must be 1")
        component_id = _require_sha256(
            row["union_component_id"], field=f"{name} union_component_id"
        )
        if component_id in components:
            raise ValueError(f"{name} repeats union_component_id")
        if previous is not None and component_id <= previous:
            raise ValueError("accepted split components must be sorted by union_component_id")
        component_members = members.get(component_id)
        if not component_members:
            raise ValueError(f"{name} has no sequence assignments")
        fold = row["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"{name} fold must be an integer")
        member_folds = {item.fold for item in component_members}
        if member_folds != {fold}:
            raise ValueError(f"{name} fold differs from sequence assignments")
        if row["sequence_count"] != len(component_members):
            raise ValueError(f"{name} sequence_count differs from sequence assignments")
        member_ids = [item.sequence_id for item in component_members]
        if row["sequence_ids_sha256"] != _sequence_set_sha256(member_ids):
            raise ValueError(f"{name} member checksum differs from sequence assignments")
        homology_count = len({item.homology_component_id for item in component_members})
        if row["homology_component_count"] != homology_count:
            raise ValueError(f"{name} homology count differs from sequence assignments")
        _require_nonnegative_int(row["study_key_count"], field=f"{name} study_key_count")
        grouping_edges = row["grouping_edge_ids"]
        if not isinstance(grouping_edges, list) or not grouping_edges:
            raise ValueError(f"{name} grouping_edge_ids must be a non-empty array")
        if any(_SHA256.fullmatch(str(item)) is None for item in grouping_edges):
            raise ValueError(f"{name} contains an invalid grouping edge ID")
        if cast(list[str], grouping_edges) != sorted(set(cast(list[str], grouping_edges))):
            raise ValueError(f"{name} grouping_edge_ids must be sorted and unique")
        balance = row["balance_counts"]
        if not isinstance(balance, dict) or set(balance) != {
            "component_count",
            "sequences",
            *_CLASS_METRICS,
        }:
            raise ValueError(f"{name} has an invalid balance_counts schema")
        for field, value in balance.items():
            _require_nonnegative_int(value, field=f"{name} balance_counts.{field}")
        if balance["component_count"] != 1 or balance["sequences"] != len(component_members):
            raise ValueError(f"{name} base balance counts differ from assignments")
        components[component_id] = row
        previous = component_id
    if set(components) != set(members):
        raise ValueError("component table and sequence assignments have different support")
    return components


def _validate_split_documents(
    *,
    config: Gate1UnionConfig,
    sequences_snapshot: InputSnapshot,
    ledger_snapshot: InputSnapshot,
    assignments_snapshot: InputSnapshot,
    components_snapshot: InputSnapshot,
    manifest_snapshot: InputSnapshot,
    audit_snapshot: InputSnapshot,
    top_snapshot: InputSnapshot,
    independent_receipt_snapshot: InputSnapshot,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    manifest = _parse_json_object(manifest_snapshot.payload, name="accepted split manifest")
    audit = _parse_json_object(audit_snapshot.payload, name="accepted split audit")
    receipt = _parse_json_object(
        independent_receipt_snapshot.payload, name="accepted split independent receipt"
    )
    top = _parse_sha256_manifest(top_snapshot.payload, name="accepted split top manifest")
    if set(top) != _SPLIT_TOP_ENTRIES:
        raise ValueError("accepted split top manifest has unexpected entries")
    expected_top_links = {
        "split/audit.json": audit_snapshot.sha256,
        "split/components.jsonl": components_snapshot.sha256,
        "split/manifest.json": manifest_snapshot.sha256,
        "split/sequence_assignments.jsonl": assignments_snapshot.sha256,
    }
    if any(top[name] != digest for name, digest in expected_top_links.items()):
        raise ValueError("accepted split top manifest does not bind supplied artifacts")

    manifest_fields = {
        "artifact",
        "artifacts",
        "config_sha256",
        "counts",
        "input",
        "policies",
        "provenance",
        "runtime",
        "schema_version",
        "status",
        "upstream",
    }
    _require_exact_fields(manifest, expected=manifest_fields, name="accepted split manifest")
    if (
        manifest["schema_version"] != 1
        or isinstance(manifest["schema_version"], bool)
        or manifest["artifact"] != _SPLIT_ARTIFACT
        or manifest["status"] != _SPLIT_STATUS
    ):
        raise ValueError("accepted split manifest identity/status is invalid")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "audit",
        "components",
        "grouping_edges",
        "sequence_assignments",
    }:
        raise ValueError("accepted split manifest artifact map is invalid")
    expected_artifact_links = {
        "audit": ("audit.json", audit_snapshot.sha256),
        "components": ("components.jsonl", components_snapshot.sha256),
        "sequence_assignments": ("sequence_assignments.jsonl", assignments_snapshot.sha256),
    }
    for name, (filename, digest) in expected_artifact_links.items():
        entry = artifacts[name]
        if not isinstance(entry, dict) or entry != {"filename": filename, "sha256": digest}:
            raise ValueError(f"accepted split manifest {name} artifact link is invalid")
    input_map = manifest["input"]
    if not isinstance(input_map, dict):
        raise ValueError("accepted split manifest input map is invalid")
    for name, expected_digest in (
        ("sequences", sequences_snapshot.sha256),
        ("endpoint_context_ledger", ledger_snapshot.sha256),
    ):
        entry = input_map.get(name)
        if not isinstance(entry, dict) or entry.get("sha256") != expected_digest:
            raise ValueError(f"accepted split manifest does not bind {name}")
    policies = manifest["policies"]
    if (
        not isinstance(policies, dict)
        or policies.get("component") != _SPLIT_COMPONENT_POLICY
        or policies.get("homology") != _SPLIT_HOMOLOGY_POLICY
        or "never a model feature" not in str(policies.get("study", ""))
    ):
        raise ValueError("accepted split policies are invalid")
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise ValueError("accepted split manifest counts are invalid")
    expected_counts = {
        "folds": config.folds,
        "unique_sequences": config.expected_sequences,
        "sequence_assignments": config.expected_sequences,
        "homology_components": config.expected_split_homology_components,
        "union_components": config.expected_split_union_components,
        "retained_balance_contexts": config.expected_examples,
    }
    if any(counts.get(name) != value for name, value in expected_counts.items()):
        raise ValueError("accepted split manifest census differs from the frozen config")

    audit_fields = {
        "artifact",
        "balance",
        "graph",
        "input",
        "invariants",
        "limitations",
        "schema_version",
        "status",
    }
    _require_exact_fields(audit, expected=audit_fields, name="accepted split audit")
    if (
        audit["schema_version"] != 1
        or audit["artifact"] != "homology_study_union_split_audit"
        or audit["status"] != _SPLIT_STATUS
    ):
        raise ValueError("accepted split audit identity/status is invalid")
    invariants = audit["invariants"]
    if (
        not isinstance(invariants, dict)
        or not invariants
        or not all(value is True for value in invariants.values())
    ):
        raise ValueError("accepted split audit invariants did not all pass")
    graph = audit["graph"]
    if (
        not isinstance(graph, dict)
        or graph.get("component_policy") != _SPLIT_COMPONENT_POLICY
        or graph.get("homology_algorithm") != _SPLIT_HOMOLOGY_POLICY
        or graph.get("identity_threshold") != config.homology_identity_threshold
        or graph.get("homology_components") != config.expected_split_homology_components
        or graph.get("union_components") != config.expected_split_union_components
    ):
        raise ValueError("accepted split audit graph contract is invalid")

    receipt_fields = {
        "artifact",
        "artifact_sha256",
        "balance",
        "checks",
        "code_manifest_sha256",
        "counts",
        "endpoint_context_top_manifest_sha256",
        "frozen_input_manifest_sha256",
        "git_commit",
        "graph",
        "normalized_top_manifest_sha256",
        "schema_version",
        "status",
        "top_manifest_sha256",
    }
    _require_exact_fields(
        receipt, expected=receipt_fields, name="accepted split independent receipt"
    )
    if (
        receipt["schema_version"] != 1
        or receipt["artifact"] != "homology_study_union_split_independent_verification"
        or receipt["status"] != "passed"
    ):
        raise ValueError("accepted split independent receipt identity/status is invalid")
    checks = receipt["checks"]
    if (
        not isinstance(checks, dict)
        or set(checks) != _INDEPENDENT_CHECKS
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("accepted split independent receipt checks did not all pass")
    receipt_hashes = receipt["artifact_sha256"]
    if not isinstance(receipt_hashes, dict) or receipt_hashes != {
        "audit": audit_snapshot.sha256,
        "components": components_snapshot.sha256,
        "grouping_edges": top["split/grouping_edges.jsonl"],
        "sequence_assignments": assignments_snapshot.sha256,
    }:
        raise ValueError("accepted split independent receipt artifact hashes are invalid")
    if receipt["top_manifest_sha256"] != top_snapshot.sha256:
        raise ValueError("accepted split independent receipt does not bind the top manifest")
    if receipt["code_manifest_sha256"] != top["CODE_SHA256SUMS"]:
        raise ValueError("accepted split receipt does not bind the top-level code manifest")
    if receipt["frozen_input_manifest_sha256"] != top["FROZEN_INPUT_SHA256SUMS"]:
        raise ValueError("accepted split receipt does not bind the frozen-input manifest")
    if receipt["counts"] != counts:
        raise ValueError("accepted split receipt and manifest censuses differ")
    if receipt["graph"] != graph:
        raise ValueError("accepted split receipt and audit graph records differ")
    audit_balance = audit["balance"]
    receipt_balance = receipt["balance"]
    if not isinstance(audit_balance, dict) or not isinstance(receipt_balance, dict):
        raise ValueError("accepted split balance records are invalid")
    for key in ("by_fold", "context_census", "totals"):
        if receipt_balance.get(key) != audit_balance.get(key):
            raise ValueError(f"accepted split receipt and audit differ for balance.{key}")
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or receipt["git_commit"] != provenance.get("git_commit"):
        raise ValueError("accepted split receipt and manifest Git identities differ")
    code_provenance = provenance.get("code_manifest") if isinstance(provenance, dict) else None
    if (
        not isinstance(code_provenance, dict)
        or code_provenance.get("sha256") != receipt["code_manifest_sha256"]
    ):
        raise ValueError("accepted split receipt and manifest code attestations differ")
    endpoint_top = input_map.get("endpoint_context_top_manifest")
    if (
        not isinstance(endpoint_top, dict)
        or endpoint_top.get("sha256") != receipt["endpoint_context_top_manifest_sha256"]
    ):
        raise ValueError("accepted split receipt and endpoint-context attestations differ")
    if (
        not isinstance(receipt["git_commit"], str)
        or _GIT_SHA1.fullmatch(cast(str, receipt["git_commit"])) is None
    ):
        raise ValueError("accepted split receipt Git identity is invalid")
    return manifest, audit, receipt


def _context_semantics(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["sequence_id"],
        row["endpoint"],
        row["context_id"],
        json.dumps(row["source_conditions"], sort_keys=True, separators=(",", ":")),
        json.dumps(row["exposure_concentration"], sort_keys=True, separators=(",", ":")),
    )


def _validate_and_group_ledger(
    rows: Sequence[Mapping[str, object]],
    *,
    sequences: Mapping[str, str],
    expected_rows: int,
) -> dict[str, list[Mapping[str, object]]]:
    if len(rows) != expected_rows:
        raise ValueError("endpoint-context ledger row count differs from the frozen config")
    contexts: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    observation_ids: set[str] = set()
    previous_observation_id: str | None = None
    for index, row in enumerate(rows, start=1):
        name = f"endpoint-context ledger row {index}"
        _require_exact_fields(row, expected=_LEDGER_FIELDS, name=name)
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"{name} schema_version must be 1")
        sequence_id = _require_sha256(row["sequence_id"], field=f"{name} sequence_id")
        if sequence_id not in sequences:
            raise ValueError(f"{name} references an unknown sequence")
        for field in (
            "observation_id",
            "assay_row_sha256",
            "assay_context_id",
            "context_id",
            "provenance_id",
        ):
            _require_sha256(row[field], field=f"{name} {field}")
        observation_id = cast(str, row["observation_id"])
        if observation_id in observation_ids:
            raise ValueError(f"{name} repeats observation_id")
        if previous_observation_id is not None and observation_id <= previous_observation_id:
            raise ValueError("endpoint-context ledger must be sorted by observation_id")
        observation_ids.add(observation_id)
        previous_observation_id = observation_id
        endpoint = row["endpoint"]
        if endpoint not in {"mic", "hc50", "hemolysis_percent"}:
            raise ValueError(f"{name} has an invalid endpoint")
        eligible_tasks = _require_string_list(row["eligible_tasks"], field=f"{name} eligible_tasks")
        _require_string_list(row["exclusion_codes"], field=f"{name} exclusion_codes")
        _require_string_list(
            row["study_keys"],
            field=f"{name} study_keys",
            allow_empty=False,
            require_sorted=False,
        )
        is_binary_eligible = "bacterial_mic16" in eligible_tasks
        source_gram = row["source_gram"]
        gram_specific_tasks = set(eligible_tasks) & {
            "gram_negative_mic16",
            "gram_positive_mic16",
        }
        expected_gram_task = f"gram_{source_gram}_mic16"
        if is_binary_eligible and (
            endpoint != "mic"
            or row["mapping_status"] != "mapped_single_supported_species"
            or row["gram_resolution"] != "concordant"
            or source_gram not in {"negative", "positive"}
            or row["expected_gram"] != source_gram
            or gram_specific_tasks != {expected_gram_task}
            or isinstance(row["mic16_label"], bool)
            or row["mic16_label"] not in {0, 1}
            or not isinstance(row["canonical_target"], str)
            or not cast(str, row["canonical_target"]).strip()
            or cast(str, row["canonical_target"]) != cast(str, row["canonical_target"]).strip()
        ):
            raise ValueError(f"{name} has inconsistent bacterial_mic16 eligibility")
        assay_context_id = cast(str, row["assay_context_id"])
        contexts[assay_context_id].append(row)

    for assay_context_id, members in contexts.items():
        semantic = _context_semantics(members[0])
        if any(_context_semantics(member) != semantic for member in members[1:]):
            raise ValueError(f"assay_context_id {assay_context_id} has inconsistent semantics")
    return dict(contexts)


def build_context_dataset(
    ledger_rows: Sequence[Mapping[str, object]],
    *,
    sequences: Mapping[str, str],
    assignments: Mapping[str, SequenceAssignment],
    expected_rows: int,
) -> ContextDataset:
    """Aggregate complete assay contexts before applying MIC16 eligibility."""

    contexts = _validate_and_group_ledger(
        ledger_rows,
        sequences=sequences,
        expected_rows=expected_rows,
    )
    examples: list[ContextExample] = []
    audits: list[ContextAuditRow] = []
    counts: Counter[str] = Counter()
    retained_sequences: set[str] = set()
    class_counts: Counter[str] = Counter()

    counts["ledger_raw_observations"] = len(ledger_rows)
    counts["ledger_assay_contexts"] = len(contexts)
    for assay_context_id in sorted(contexts):
        members = sorted(
            contexts[assay_context_id], key=lambda row: cast(str, row["observation_id"])
        )
        first = members[0]
        endpoint = cast(str, first["endpoint"])
        sequence_id = cast(str, first["sequence_id"])
        eligible = ["bacterial_mic16" in cast(list[str], row["eligible_tasks"]) for row in members]
        eligible_count = sum(eligible)
        reason_codes: tuple[str, ...] = ()
        example: ContextExample | None = None

        if endpoint != "mic":
            counts["non_mic_assay_contexts"] += 1
            counts["non_mic_raw_observations"] += len(members)
            status = "excluded_non_mic_endpoint"
        else:
            counts["mic_assay_contexts"] += 1
            counts["mic_raw_observations"] += len(members)
            counts["repeated_mic_assay_contexts"] += int(len(members) > 1)
            counts["eligible_raw_observations"] += eligible_count
            if eligible_count == 0:
                counts["all_ineligible_assay_contexts"] += 1
                counts["observations_in_all_ineligible_assay_contexts"] += len(members)
                status = "excluded_all_ineligible_context"
            elif eligible_count != len(members):
                counts["candidate_assay_contexts"] += 1
                counts["mixed_eligibility_assay_contexts"] += 1
                counts["observations_in_mixed_eligibility_assay_contexts"] += len(members)
                counts["eligible_observations_in_mixed_eligibility_assay_contexts"] += (
                    eligible_count
                )
                status = "excluded_mixed_eligibility_context"
                reason_codes = ("not_every_context_member_is_bacterial_mic16_eligible",)
            else:
                counts["candidate_assay_contexts"] += 1
                counts["fully_eligible_assay_contexts"] += 1
                labels = {cast(int, row["mic16_label"]) for row in members}
                grams = {cast(str, row["source_gram"]) for row in members}
                targets = {cast(str, row["canonical_target"]) for row in members}
                conflicts: list[str] = []
                if len(labels) != 1:
                    conflicts.append("mic16_label_disagreement")
                if len(grams) != 1:
                    conflicts.append("source_gram_disagreement")
                if len(targets) != 1:
                    conflicts.append("canonical_target_disagreement")
                if conflicts:
                    counts["conflicting_assay_contexts"] += 1
                    counts["eligible_label_conflict_contexts"] += int(
                        "mic16_label_disagreement" in conflicts
                    )
                    counts["eligible_gram_conflict_contexts"] += int(
                        "source_gram_disagreement" in conflicts
                    )
                    counts["eligible_target_conflict_contexts"] += int(
                        "canonical_target_disagreement" in conflicts
                    )
                    counts["observations_in_conflicting_assay_contexts"] += len(members)
                    status = "excluded_conflicting_context"
                    reason_codes = tuple(conflicts)
                else:
                    label = next(iter(labels))
                    gram = cast(GramClass, next(iter(grams)))
                    target = next(iter(targets))
                    assignment = assignments.get(sequence_id)
                    if assignment is None:
                        raise ValueError(
                            f"retained context {assay_context_id} has no accepted split assignment"
                        )
                    observation_ids = tuple(cast(str, row["observation_id"]) for row in members)
                    example = ContextExample(
                        example_id=assay_context_id,
                        assay_context_id=assay_context_id,
                        sequence_id=sequence_id,
                        sequence=sequences[sequence_id],
                        canonical_target=target,
                        gram=gram,
                        label=label,
                        source_observation_ids=observation_ids,
                        fold=assignment.fold,
                        homology_component_id=assignment.homology_component_id,
                        union_component_id=assignment.union_component_id,
                    )
                    examples.append(example)
                    retained_sequences.add(sequence_id)
                    counts["retained_assay_contexts"] += 1
                    counts["retained_source_observations"] += len(members)
                    metric = f"gram_{gram}_mic16_{'positive' if label else 'negative'}"
                    class_counts[metric] += 1
                    status = "included_fully_eligible_unanimous_context"
        audits.append(
            ContextAuditRow(
                schema_version=1,
                assay_context_id=assay_context_id,
                sequence_id=sequence_id,
                endpoint=endpoint,
                source_observations=len(members),
                eligible_source_observations=eligible_count,
                status=status,
                reason_codes=reason_codes,
                example_id=None if example is None else example.example_id,
            )
        )

    counts["retained_sequences"] = len(retained_sequences)
    census: dict[str, object] = {
        name: counts[name]
        for name in (
            "ledger_raw_observations",
            "ledger_assay_contexts",
            "mic_raw_observations",
            "mic_assay_contexts",
            "non_mic_raw_observations",
            "non_mic_assay_contexts",
            "repeated_mic_assay_contexts",
            "eligible_raw_observations",
            "candidate_assay_contexts",
            "all_ineligible_assay_contexts",
            "observations_in_all_ineligible_assay_contexts",
            "fully_eligible_assay_contexts",
            "mixed_eligibility_assay_contexts",
            "observations_in_mixed_eligibility_assay_contexts",
            "eligible_observations_in_mixed_eligibility_assay_contexts",
            "conflicting_assay_contexts",
            "eligible_label_conflict_contexts",
            "eligible_gram_conflict_contexts",
            "eligible_target_conflict_contexts",
            "observations_in_conflicting_assay_contexts",
            "retained_assay_contexts",
            "retained_source_observations",
            "retained_sequences",
        )
    }
    census["retained_class_counts"] = {metric: class_counts[metric] for metric in _CLASS_METRICS}
    return ContextDataset(
        examples=tuple(sorted(examples, key=lambda item: item.example_id)),
        audit_rows=tuple(audits),
        census=census,
    )


def _validate_dataset_contract(
    dataset: ContextDataset,
    *,
    config: Gate1UnionConfig,
    components: Mapping[str, Mapping[str, object]],
    split_audit: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    examples = dataset.examples
    if len(examples) != config.expected_examples:
        raise ValueError("retained context-example count differs from the frozen config")
    if sum(len(item.source_observation_ids) for item in examples) != (
        config.expected_source_observations
    ):
        raise ValueError("retained source-observation count differs from the frozen config")
    if len({item.sequence_id for item in examples}) != config.expected_modeled_sequences:
        raise ValueError("modeled sequence count differs from the frozen config")
    positives = sum(item.label for item in examples)
    if positives != config.expected_positive_examples:
        raise ValueError("positive context-example count differs from the frozen config")
    if len(examples) - positives != config.expected_negative_examples:
        raise ValueError("negative context-example count differs from the frozen config")
    if cast(int, dataset.census["eligible_target_conflict_contexts"]) != 0:
        raise ValueError("accepted split contains an unrecorded canonical-target conflict")

    balance = split_audit.get("balance")
    if not isinstance(balance, dict):
        raise ValueError("accepted split audit balance is invalid")
    accepted_census = balance.get("context_census")
    if not isinstance(accepted_census, dict):
        raise ValueError("accepted split audit context census is invalid")
    shared_census_fields = set(dataset.census) - {"eligible_target_conflict_contexts"}
    for field in shared_census_fields:
        if accepted_census.get(field) != dataset.census[field]:
            raise ValueError(f"Gate-1 context aggregation differs from split audit for {field}")

    by_fold: dict[str, dict[str, int]] = {}
    accepted_by_fold = balance.get("by_fold")
    if not isinstance(accepted_by_fold, dict):
        raise ValueError("accepted split audit fold balance is invalid")
    for fold in range(config.folds):
        selected = [item for item in examples if item.fold == fold]
        positive = sum(item.label for item in selected)
        negative = len(selected) - positive
        if len(selected) != config.expected_examples_by_fold[fold]:
            raise ValueError(f"fold {fold} context-example count differs from the frozen config")
        if positive != config.expected_positive_examples_by_fold[fold]:
            raise ValueError(f"fold {fold} positive count differs from the frozen config")
        if negative != config.expected_negative_examples_by_fold[fold]:
            raise ValueError(f"fold {fold} negative count differs from the frozen config")
        class_counts = {
            metric: sum(
                item.gram == metric.split("_")[1] and item.label == int(metric.endswith("positive"))
                for item in selected
            )
            for metric in _CLASS_METRICS
        }
        accepted_fold = accepted_by_fold.get(str(fold))
        if not isinstance(accepted_fold, dict) or any(
            accepted_fold.get(metric) != value for metric, value in class_counts.items()
        ):
            raise ValueError(f"fold {fold} Gram/label counts differ from the accepted split")
        by_fold[str(fold)] = {
            "examples": len(selected),
            "source_observations": sum(len(item.source_observation_ids) for item in selected),
            "sequences": len({item.sequence_id for item in selected}),
            "homology_components": len({item.homology_component_id for item in selected}),
            "union_components": len({item.union_component_id for item in selected}),
            "positives": positive,
            "negatives": negative,
            **class_counts,
        }

    examples_by_component: dict[str, list[ContextExample]] = defaultdict(list)
    for item in examples:
        examples_by_component[item.union_component_id].append(item)
    for component_id, component in components.items():
        balance_counts = component["balance_counts"]
        assert isinstance(balance_counts, dict)
        selected = examples_by_component.get(component_id, [])
        expected_class_counts = {
            metric: sum(
                item.gram == metric.split("_")[1] and item.label == int(metric.endswith("positive"))
                for item in selected
            )
            for metric in _CLASS_METRICS
        }
        if any(balance_counts[metric] != value for metric, value in expected_class_counts.items()):
            raise ValueError(
                f"component {component_id} context counts differ from the accepted split"
            )
    return by_fold


def _cross_fold_identity_audit(
    *,
    sequences: Mapping[str, str],
    assignments: Mapping[str, SequenceAssignment],
    modeled_sequence_ids: Iterable[str],
    identity_threshold: float,
) -> tuple[float, dict[str, float]]:
    """Recompute the strict identity boundary on the full parser sequence union."""

    ordered_ids = tuple(sorted(sequences))
    modeled = set(modeled_sequence_ids)
    maximum = 0.0
    modeled_maximum = {sequence_id: 0.0 for sequence_id in modeled}
    for left_index, left_id in enumerate(ordered_ids):
        left = sequences[left_id]
        left_fold = assignments[left_id].fold
        for right_id in ordered_ids[left_index + 1 :]:
            if assignments[right_id].fold == left_fold:
                continue
            right = sequences[right_id]
            length_upper_bound = min(len(left), len(right)) / max(len(left), len(right))
            if length_upper_bound < maximum and left_id not in modeled and right_id not in modeled:
                continue
            identity = global_sequence_identity(left, right)
            maximum = max(maximum, identity)
            if identity >= identity_threshold:
                raise ValueError(
                    "accepted split violates the strict cross-fold identity boundary: "
                    f"{left_id}/{right_id}={identity:.12g}"
                )
            if left_id in modeled and right_id in modeled:
                modeled_maximum[left_id] = max(modeled_maximum[left_id], identity)
                modeled_maximum[right_id] = max(modeled_maximum[right_id], identity)
    if any(value >= identity_threshold for value in modeled_maximum.values()):
        raise AssertionError("modeled max-training identity reaches the holdout threshold")
    return maximum, dict(sorted(modeled_maximum.items()))


def make_oof_predictions(
    examples: Iterable[ContextExample],
    *,
    config: Gate1UnionConfig,
    max_train_identity_by_sequence: Mapping[str, float],
) -> tuple[OofPrediction, ...]:
    """Fit the two frozen baselines and their untrained equal-probability mean."""

    items = tuple(sorted(examples, key=lambda item: item.example_id))
    output: list[OofPrediction] = []
    for fold in range(config.folds):
        training = tuple(item for item in items if item.fold != fold)
        testing = tuple(item for item in items if item.fold == fold)
        if not training or not testing:
            raise ValueError(f"fold {fold} has an empty train or test partition")
        training_inputs = tuple(item.model_input for item in training)
        testing_inputs = tuple(item.model_input for item in testing)
        labels = np.asarray([item.label for item in training], dtype=np.int64)
        models = (
            DescriptorLogisticOracle(
                l2=config.logistic.l2,
                max_iterations=config.logistic.max_iterations,
                tolerance=config.logistic.tolerance,
                prior_strength=config.logistic.prior_strength,
            ),
            HomologyKnnOracle(
                neighbors=config.knn.neighbors,
                similarity_power=config.knn.similarity_power,
                prior_strength=config.knn.prior_strength,
                minimum_weight=config.knn.minimum_weight,
            ),
        )
        probabilities: dict[str, FloatArray] = {}
        for model in models:
            model.fit(training_inputs, labels)
            probabilities[model.name] = model.predict_proba(testing_inputs)
        probabilities["equal_weight_ensemble"] = np.mean(
            np.stack(tuple(probabilities.values()), axis=0), axis=0
        )
        for index, item in enumerate(testing):
            maximum_identity = max_train_identity_by_sequence[item.sequence_id]
            if maximum_identity >= config.homology_identity_threshold:
                raise AssertionError("held-out example reaches the homology identity threshold")
            for model_name, model_probabilities in probabilities.items():
                probability = float(model_probabilities[index])
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{model_name} emitted an invalid probability")
                output.append(
                    OofPrediction(
                        model=model_name,
                        example_id=item.example_id,
                        assay_context_id=item.assay_context_id,
                        sequence_id=item.sequence_id,
                        sequence=item.sequence,
                        canonical_target=item.canonical_target,
                        gram=item.gram,
                        label=item.label,
                        source_observations=len(item.source_observation_ids),
                        fold=item.fold,
                        homology_component_id=item.homology_component_id,
                        union_component_id=item.union_component_id,
                        max_train_identity=maximum_identity,
                        probability=probability,
                    )
                )
    expected = len(items) * len(_MODELS)
    if len(output) != expected:
        raise AssertionError(f"expected {expected} OOF rows, created {len(output)}")
    return tuple(sorted(output, key=lambda item: (item.model, item.example_id)))


def _roc_auc(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = int(np.sum(labels))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    result = 0.0
    index = 0
    while index < labels.size:
        end = index + 1
        while end < labels.size and sorted_probabilities[end] == sorted_probabilities[index]:
            end += 1
        group = sorted_labels[index:end]
        new_positives = int(np.sum(group))
        true_positive += new_positives
        false_positive += len(group) - new_positives
        result += (new_positives / positives) * (true_positive / (true_positive + false_positive))
        index = end
    return float(result)


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    calibration_bins: int,
) -> dict[str, int | float | None]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or probability.ndim != 1 or y.size != probability.size or y.size == 0:
        raise ValueError("labels and probabilities must be equal, non-empty vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must contain only zero and one")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    prediction = probability >= 0.5
    positive = y == 1
    negative = ~positive
    sensitivity = float(np.mean(prediction[positive])) if np.any(positive) else None
    specificity = float(np.mean(~prediction[negative])) if np.any(negative) else None
    balanced_accuracy = (
        None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2.0
    )
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    bin_index = np.digitize(probability, edges[1:-1], right=False)
    calibration_error = 0.0
    for index in range(calibration_bins):
        selected = bin_index == index
        if np.any(selected):
            calibration_error += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(y[selected]))
            )
    return {
        "n": int(y.size),
        "positives": int(np.sum(y)),
        "negatives": int(y.size - np.sum(y)),
        "prevalence": float(np.mean(y)),
        "roc_auc": _roc_auc(y, probability),
        "average_precision": _average_precision(y, probability),
        "brier": float(np.mean(np.square(probability - y))),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "balanced_accuracy_at_0_5": balanced_accuracy,
        "sensitivity_at_0_5": sensitivity,
        "specificity_at_0_5": specificity,
        "ece_equal_width": calibration_error,
    }


def _metric_subset(
    predictions: Sequence[OofPrediction], *, calibration_bins: int
) -> dict[str, int | float | None]:
    return binary_metrics(
        [item.label for item in predictions],
        [item.probability for item in predictions],
        calibration_bins=calibration_bins,
    )


def _union_component_bootstrap_intervals(
    predictions: Sequence[OofPrediction],
    *,
    calibration_bins: int,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss")
    point = _metric_subset(predictions, calibration_bins=calibration_bins)
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    if replicates:
        by_component: dict[str, list[OofPrediction]] = defaultdict(list)
        for item in predictions:
            by_component[item.union_component_id].append(item)
        component_ids = tuple(sorted(by_component))
        generator = np.random.default_rng(seed)
        for _ in range(replicates):
            sampled = generator.choice(component_ids, size=len(component_ids), replace=True)
            rows = [row for component_id in sampled for row in by_component[str(component_id)]]
            metrics = _metric_subset(rows, calibration_bins=calibration_bins)
            for name in metric_names:
                value = metrics[name]
                if value is not None:
                    values[name].append(float(value))
    output: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        samples = np.asarray(values[name], dtype=np.float64)
        output[name] = {
            "point": cast(float | None, point[name]),
            "lower": None if samples.size == 0 else float(np.quantile(samples, 0.025)),
            "upper": None if samples.size == 0 else float(np.quantile(samples, 0.975)),
            "successful_replicates": int(samples.size),
        }
    return output


def summarize_oof_predictions(
    predictions: Iterable[OofPrediction], *, config: Gate1UnionConfig
) -> dict[str, object]:
    items = tuple(predictions)
    by_model: dict[str, list[OofPrediction]] = defaultdict(list)
    for item in items:
        by_model[item.model].append(item)
    if set(by_model) != set(_MODELS):
        raise ValueError("OOF predictions do not contain the frozen model set")
    output: dict[str, object] = {}
    for model_name in sorted(by_model):
        model_rows = tuple(sorted(by_model[model_name], key=lambda item: item.example_id))
        if len(model_rows) != config.expected_examples:
            raise ValueError(f"{model_name} does not cover every context example")
        by_similarity: dict[str, object] = {}
        for left, right in zip(
            config.similarity_bin_edges, config.similarity_bin_edges[1:], strict=False
        ):
            selected = [
                item
                for item in model_rows
                if left <= item.max_train_identity < right
                or (right == 1.0 and item.max_train_identity == 1.0)
            ]
            if selected:
                by_similarity[f"[{left:.2f},{right:.2f})"] = _metric_subset(
                    selected, calibration_bins=config.calibration_bins
                )
        output[model_name] = {
            "overall": _metric_subset(model_rows, calibration_bins=config.calibration_bins),
            "union_component_bootstrap_95ci": _union_component_bootstrap_intervals(
                model_rows,
                calibration_bins=config.calibration_bins,
                replicates=config.bootstrap_replicates,
                seed=config.seed,
            ),
            "by_fold": {
                str(fold): _metric_subset(
                    [item for item in model_rows if item.fold == fold],
                    calibration_bins=config.calibration_bins,
                )
                for fold in range(config.folds)
            },
            "by_gram": {
                gram: _metric_subset(
                    [item for item in model_rows if item.gram == gram],
                    calibration_bins=config.calibration_bins,
                )
                for gram in ("negative", "positive")
            },
            "by_canonical_target": {
                target: _metric_subset(
                    [item for item in model_rows if item.canonical_target == target],
                    calibration_bins=config.calibration_bins,
                )
                for target in sorted({item.canonical_target for item in model_rows})
            },
            "by_max_train_identity": by_similarity,
        }
    aligned = {
        name: {item.example_id: item.probability for item in by_model[name]} for name in _MODELS
    }
    if any(set(values) != set(aligned[_MODELS[0]]) for values in aligned.values()):
        raise ValueError("OOF model supports are not identical")
    example_ids = sorted(aligned[_MODELS[0]])
    left = np.asarray([aligned["descriptor_logistic"][item] for item in example_ids])
    right = np.asarray([aligned["homology_knn"][item] for item in example_ids])
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else None
    output["model_diversity"] = {
        "member_prediction_pearson": correlation,
        "members": ["descriptor_logistic", "homology_knn"],
        "ensemble_policy": "untrained_equal_probability_mean",
    }
    return output


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_ready(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_oof_csv(path: Path, predictions: Sequence[OofPrediction]) -> None:
    fieldnames = [
        "model",
        "example_id",
        "assay_context_id",
        "sequence_id",
        "sequence",
        "canonical_target",
        "gram",
        "label",
        "source_observations",
        "fold",
        "homology_component_id",
        "union_component_id",
        "max_train_identity",
        "probability",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in predictions:
            row = asdict(item)
            row["max_train_identity"] = f"{item.max_train_identity:.17g}"
            row["probability"] = f"{item.probability:.17g}"
            writer.writerow(row)


def _write_checksum_manifest(path: Path, entries: Mapping[str, str]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for filename in sorted(entries):
            handle.write(f"{entries[filename]}  {filename}\n")


def run_gate1_union_benchmark(
    *,
    sequences_path: str | Path,
    endpoint_context_ledger_path: str | Path,
    split_assignments_path: str | Path,
    split_components_path: str | Path,
    split_manifest_path: str | Path,
    split_audit_path: str | Path,
    split_top_manifest_path: str | Path,
    split_independent_receipt_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    git_commit: str,
    code_manifest_path: str | Path,
) -> Gate1UnionExecution:
    """Run deterministic OOF evaluation on the accepted, immutable union split."""

    if _GIT_SHA1.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character Git SHA")
    requested_output = Path(output_dir)
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse Gate-1 union output: {requested_output}")
    output = requested_output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse Gate-1 union output: {output}")

    snapshots = {
        "config": _read_snapshot(config_path, name="Gate-1 union config"),
        "sequences": _read_snapshot(sequences_path, name="parser-v7 sequences"),
        "ledger": _read_snapshot(endpoint_context_ledger_path, name="endpoint-context ledger"),
        "assignments": _read_snapshot(split_assignments_path, name="accepted split assignments"),
        "components": _read_snapshot(split_components_path, name="accepted split components"),
        "split_manifest": _read_snapshot(split_manifest_path, name="accepted split manifest"),
        "split_audit": _read_snapshot(split_audit_path, name="accepted split audit"),
        "split_top": _read_snapshot(split_top_manifest_path, name="accepted split top manifest"),
        "split_independent_receipt": _read_snapshot(
            split_independent_receipt_path,
            name="accepted split independent receipt",
        ),
        "code_manifest": _read_snapshot(code_manifest_path, name="code manifest"),
    }
    config = _config_from_payload(snapshots["config"].path, snapshots["config"].payload)
    code_attestation, code_snapshots = _validate_code_manifest(
        snapshots["code_manifest"], config_snapshot=snapshots["config"]
    )
    for snapshot_name, expected, label in (
        ("sequences", config.sequences_sha256, "parser-v7 sequences"),
        (
            "ledger",
            config.endpoint_context_ledger_sha256,
            "endpoint-context ledger",
        ),
        ("assignments", config.split_assignments_sha256, "accepted split assignments"),
        ("components", config.split_components_sha256, "accepted split components"),
        ("split_manifest", config.split_manifest_sha256, "accepted split manifest"),
        ("split_audit", config.split_audit_sha256, "accepted split audit"),
        ("split_top", config.split_top_manifest_sha256, "accepted split top manifest"),
        (
            "split_independent_receipt",
            config.split_independent_receipt_sha256,
            "accepted split independent receipt",
        ),
    ):
        _expect_hash(snapshots[snapshot_name], expected, name=label)

    split_manifest, split_audit, split_independent_receipt = _validate_split_documents(
        config=config,
        sequences_snapshot=snapshots["sequences"],
        ledger_snapshot=snapshots["ledger"],
        assignments_snapshot=snapshots["assignments"],
        components_snapshot=snapshots["components"],
        manifest_snapshot=snapshots["split_manifest"],
        audit_snapshot=snapshots["split_audit"],
        top_snapshot=snapshots["split_top"],
        independent_receipt_snapshot=snapshots["split_independent_receipt"],
    )
    sequences = _read_sequences(snapshots["sequences"], config=config)
    assignments = _read_assignments(snapshots["assignments"], config=config, sequences=sequences)
    components = _read_components(snapshots["components"], config=config, assignments=assignments)
    ledger_rows = _parse_jsonl(snapshots["ledger"].payload, name="endpoint-context ledger")
    dataset = build_context_dataset(
        ledger_rows,
        sequences=sequences,
        assignments=assignments,
        expected_rows=config.expected_ledger_rows,
    )
    fold_summary = _validate_dataset_contract(
        dataset,
        config=config,
        components=components,
        split_audit=split_audit,
    )
    maximum_cross_fold_identity, maximum_by_sequence = _cross_fold_identity_audit(
        sequences=sequences,
        assignments=assignments,
        modeled_sequence_ids=(item.sequence_id for item in dataset.examples),
        identity_threshold=config.homology_identity_threshold,
    )
    predictions = make_oof_predictions(
        dataset.examples,
        config=config,
        max_train_identity_by_sequence=maximum_by_sequence,
    )
    metrics = summarize_oof_predictions(predictions, config=config)

    for name, snapshot in snapshots.items():
        _assert_snapshot_unchanged(snapshot, name=name)
    for snapshot in code_snapshots:
        _assert_snapshot_unchanged(snapshot, name="code inventory")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    published = False
    try:
        paths = {
            "context_audit": staging / "context_audit.jsonl",
            "examples": staging / "examples.jsonl",
            "folds": staging / "folds.json",
            "oof": staging / "oof_predictions.csv",
            "metrics": staging / "metrics.json",
            "split_receipt": staging / "split_receipt.json",
            "manifest": staging / "manifest.json",
            "top_manifest": staging / "SHA256SUMS",
        }
        _write_jsonl(
            paths["context_audit"],
            (cast(dict[str, object], asdict(item)) for item in dataset.audit_rows),
        )
        _write_jsonl(
            paths["examples"],
            (
                {
                    "schema_version": 1,
                    "example_id": item.example_id,
                    "assay_context_id": item.assay_context_id,
                    "sequence_id": item.sequence_id,
                    "sequence": item.sequence,
                    "canonical_target": item.canonical_target,
                    "gram": item.gram,
                    "label": item.label,
                    "source_observations": len(item.source_observation_ids),
                    "fold": item.fold,
                    "homology_component_id": item.homology_component_id,
                    "union_component_id": item.union_component_id,
                }
                for item in dataset.examples
            ),
        )
        canonical_targets_by_fold = {
            str(fold): dict(
                sorted(
                    Counter(
                        item.canonical_target for item in dataset.examples if item.fold == fold
                    ).items()
                )
            )
            for fold in range(config.folds)
        }
        folds_document = {
            "schema_version": 1,
            "artifact": "gate1_context_union_fold_reuse",
            "assignment_policy": _FOLD_POLICY,
            "identity_threshold": config.homology_identity_threshold,
            "maximum_cross_fold_identity": maximum_cross_fold_identity,
            "folds": fold_summary,
            "canonical_targets_by_fold": canonical_targets_by_fold,
            "assignments": [
                {
                    "example_id": item.example_id,
                    "sequence_id": item.sequence_id,
                    "homology_component_id": item.homology_component_id,
                    "union_component_id": item.union_component_id,
                    "fold": item.fold,
                }
                for item in dataset.examples
            ],
        }
        _write_json(paths["folds"], folds_document)
        _write_oof_csv(paths["oof"], predictions)
        _write_json(paths["metrics"], metrics)

        input_hashes = {
            "config": snapshots["config"].sha256,
            "endpoint_context_ledger": snapshots["ledger"].sha256,
            "parser_v7_sequences": snapshots["sequences"].sha256,
            "split_assignments": snapshots["assignments"].sha256,
            "split_audit": snapshots["split_audit"].sha256,
            "split_components": snapshots["components"].sha256,
            "split_independent_receipt": snapshots["split_independent_receipt"].sha256,
            "split_manifest": snapshots["split_manifest"].sha256,
            "split_top_manifest": snapshots["split_top"].sha256,
            "code_manifest": snapshots["code_manifest"].sha256,
        }
        consumption_receipt = {
            "schema_version": 1,
            "artifact": "gate1_union_accepted_split_consumption_receipt",
            "status": "passed",
            "assignment_policy": _FOLD_POLICY,
            "input_sha256": input_hashes,
            "accepted_split": {
                "artifact": split_manifest["artifact"],
                "git_commit": split_independent_receipt["git_commit"],
                "top_manifest_sha256": snapshots["split_top"].sha256,
                "independent_receipt_sha256": snapshots["split_independent_receipt"].sha256,
            },
            "code_attestation": code_attestation,
            "census": {
                "parser_sequences": len(sequences),
                "ledger_rows": len(ledger_rows),
                "context_examples": len(dataset.examples),
                "source_observations": sum(
                    len(item.source_observation_ids) for item in dataset.examples
                ),
                "modeled_sequences": len({item.sequence_id for item in dataset.examples}),
                "homology_components": len(
                    {item.homology_component_id for item in assignments.values()}
                ),
                "union_components": len(components),
                "positives": sum(item.label for item in dataset.examples),
                "negatives": sum(1 - item.label for item in dataset.examples),
                "examples_by_fold": [
                    fold_summary[str(fold)]["examples"] for fold in range(config.folds)
                ],
                "positive_examples_by_fold": [
                    fold_summary[str(fold)]["positives"] for fold in range(config.folds)
                ],
                "negative_examples_by_fold": [
                    fold_summary[str(fold)]["negatives"] for fold in range(config.folds)
                ],
            },
            "identity": {
                "algorithm": _SPLIT_HOMOLOGY_POLICY,
                "threshold": config.homology_identity_threshold,
                "maximum_cross_fold_identity": maximum_cross_fold_identity,
            },
            "invariants": {
                "accepted_independent_verification_passed": True,
                "accepted_sequence_assignments_reused_exactly": True,
                "accepted_split_hash_chain_valid": True,
                "all_contexts_grouped_before_eligibility_filtering": True,
                "component_assignment_join_exact": True,
                "cross_fold_identity_strictly_below_threshold": (
                    maximum_cross_fold_identity < config.homology_identity_threshold
                ),
                "fold_label_census_exact": True,
                "model_feature_allowlist_exact": True,
            },
        }
        if not all(cast(dict[str, bool], consumption_receipt["invariants"]).values()):
            raise AssertionError("Gate-1 union split-consumption invariant failed")
        _write_json(paths["split_receipt"], consumption_receipt)

        label_summary = {
            "context_examples": len(dataset.examples),
            "source_observations": sum(
                len(item.source_observation_ids) for item in dataset.examples
            ),
            "modeled_sequences": len({item.sequence_id for item in dataset.examples}),
            "positive_examples": sum(item.label for item in dataset.examples),
            "negative_examples": sum(1 - item.label for item in dataset.examples),
            "activity_threshold_um": config.activity_threshold_um,
            "by_fold": fold_summary,
            "canonical_targets_by_fold": canonical_targets_by_fold,
            "context_census": dataset.census,
        }
        artifact_roles = {
            "context_audit": "all endpoint contexts and pre-filter aggregation decisions",
            "examples": "model-facing retained context examples without study/provenance features",
            "folds": "exact example join to accepted sequence assignments",
            "oof": "three-model context-level out-of-fold predictions",
            "metrics": "context metrics with union-component bootstrap intervals",
            "split_receipt": "path-free accepted split hash/census/invariant receipt",
        }
        manifest = {
            "schema_version": 1,
            "artifact": _BENCHMARK,
            "status": "development_evidence_not_an_untouched_evaluation_panel",
            "config_sha256": snapshots["config"].sha256,
            "git_commit": git_commit,
            "input_sha256": input_hashes,
            "accepted_split": consumption_receipt["accepted_split"],
            "code_attestation": code_attestation,
            "policies": {
                "label": _LABEL_POLICY,
                "fold_assignment": _FOLD_POLICY,
                "homology": _SPLIT_HOMOLOGY_POLICY,
                "bootstrap_unit": "union_component_id",
                "ensemble": "untrained_equal_probability_mean",
                "model_features": list(_MODEL_FEATURES),
                "forbidden_model_features": list(_FORBIDDEN_MODEL_FEATURES),
                "context_feature": "canonical_target",
            },
            "models": list(_MODELS),
            "label_summary": label_summary,
            "identity_audit": {
                "threshold": config.homology_identity_threshold,
                "maximum_cross_fold_identity": maximum_cross_fold_identity,
            },
            "runtime": {
                "numpy": np.__version__,
                "python": (
                    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                ),
            },
            "artifacts": {
                name: {
                    "filename": paths[name].name,
                    "sha256": _file_sha256(paths[name]),
                    "role": artifact_roles[name],
                }
                for name in artifact_roles
            },
        }
        _write_json(paths["manifest"], manifest)
        top_entries = {
            path.name: _file_sha256(path) for name, path in paths.items() if name != "top_manifest"
        }
        _write_checksum_manifest(paths["top_manifest"], top_entries)
        for name, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, name=name)
        for snapshot in code_snapshots:
            _assert_snapshot_unchanged(snapshot, name="code inventory")
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to replace Gate-1 union output: {output}")
        staging.rename(output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    return Gate1UnionExecution(
        output_dir=output,
        context_audit_path=output / "context_audit.jsonl",
        examples_path=output / "examples.jsonl",
        folds_path=output / "folds.json",
        oof_path=output / "oof_predictions.csv",
        metrics_path=output / "metrics.json",
        split_receipt_path=output / "split_receipt.json",
        manifest_path=output / "manifest.json",
        top_manifest_path=output / "SHA256SUMS",
        examples=len(dataset.examples),
        source_observations=sum(len(item.source_observation_ids) for item in dataset.examples),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--endpoint-context-ledger", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--split-components", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--split-top-manifest", type=Path, required=True)
    parser.add_argument("--split-independent-receipt", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/oracle_gate1_union_v1.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = run_gate1_union_benchmark(
        sequences_path=args.sequences,
        endpoint_context_ledger_path=args.endpoint_context_ledger,
        split_assignments_path=args.split_assignments,
        split_components_path=args.split_components,
        split_manifest_path=args.split_manifest,
        split_audit_path=args.split_audit,
        split_top_manifest_path=args.split_top_manifest,
        split_independent_receipt_path=args.split_independent_receipt,
        config_path=args.config,
        output_dir=args.output_dir,
        git_commit=args.git_commit,
        code_manifest_path=args.code_manifest,
    )
    print(
        json.dumps(
            {
                "examples": execution.examples,
                "source_observations": execution.source_observations,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

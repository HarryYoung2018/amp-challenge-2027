"""Calibrate immutable APEX members on the accepted context-level union folds.

The raw APEX MIC table is sequence indexed and may therefore be reused across
split policies.  Every fitted quantity in this module is new: each member is
calibrated on the complement of one accepted homology-and-study union fold,
and uncertainty intervals resample complete union components.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np

from amp_challenge.benchmarks.oracle_gate1_union import binary_metrics
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

GramClass = Literal["positive", "negative"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_ARTIFACT = "apex_context_activity_homology_study_union_v1"
_BASE_ARTIFACT = "gate1_context_activity_homology_study_union_v1"
_BASE_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_OUTPUT_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_ASSIGNMENT_POLICY = (
    "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
)
_CALIBRATION_POLICY = (
    "per_member_logistic_calibration_of_target_specific_log10_mic_signal_on_outer_train_folds"
)
_BOOTSTRAP_UNIT = "union_component_id"
_CALIBRATION_WEIGHTING = "context_equal_v1"
_BOOTSTRAP_WEIGHTING = "union_component_resample_context_multiplicity_v1"
_MODEL_FEATURES = ("raw_apex_member_mic", "canonical_target")
_FORBIDDEN_FEATURES = (
    "assay_context_id",
    "example_id",
    "fold",
    "homology_component_id",
    "union_component_id",
    "study_keys",
    "provenance_id",
    "source_record_id",
)
_LOGICAL_CONFIG_PATH = "configs/models/apex_union_oof_v1.toml"
_EXECUTING_MODULE_PATH = "src/amp_challenge/benchmarks/apex_union_oof.py"
_FIXED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/benchmark_apex_union_oof_v1_twins.sbatch",
        "cluster/validate_apex_union_oof_output.sh",
        _LOGICAL_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)
_BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_APEX_MEAN_MODEL = "apex_member_mean"
_SIMILARITY_STRATA = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8))
_SET_DIGEST_ENCODING = (
    "sorted unique lowercase identifier strings encoded as ASCII, joined by LF with a "
    "terminal LF; empty set is empty bytes"
)
_ASSIGNMENT_DIGEST_ENCODING = (
    "example_id, sequence_id, label, fold, homology_component_id, union_component_id, "
    "canonical_target encoded as tab-separated UTF-8 records sorted by example_id with "
    "a terminal LF"
)
_BOOTSTRAP_DRAW_ENCODING = (
    "zero-based replicate, sampled-position, union_component_id encoded as tab-separated "
    "ASCII records in generator order with a terminal LF"
)

_APEX_ENDPOINTS = (
    ("A. baumannii ATCC 19606", "apex_mic_um__a_baumannii_atcc_19606"),
    ("E. coli ATCC 11775", "apex_mic_um__e_coli_atcc_11775"),
    ("E. coli AIC221", "apex_mic_um__e_coli_aic221"),
    ("E. coli AIC222", "apex_mic_um__e_coli_aic222"),
    ("K. pneumoniae ATCC 13883", "apex_mic_um__k_pneumoniae_atcc_13883"),
    ("P. aeruginosa PA01", "apex_mic_um__p_aeruginosa_pa01"),
    ("P. aeruginosa PA14", "apex_mic_um__p_aeruginosa_pa14"),
    ("S. aureus ATCC 12600", "apex_mic_um__s_aureus_atcc_12600"),
    (
        "S. aureus (ATCC BAA-1556) - MRSA",
        "apex_mic_um__s_aureus_atcc_baa_1556_mrsa",
    ),
    (
        "vancomycin-resistant E. faecalis ATCC 700802",
        "apex_mic_um__vancomycin_resistant_e_faecalis_atcc_700802",
    ),
    (
        "vancomycin-resistant E. faecium ATCC 700221",
        "apex_mic_um__vancomycin_resistant_e_faecium_atcc_700221",
    ),
)
_APEX_ENDPOINT_COLUMNS = tuple(column for _, column in _APEX_ENDPOINTS)
_TARGET_ENDPOINTS = (
    ("acinetobacter_baumannii", ("apex_mic_um__a_baumannii_atcc_19606",)),
    (
        "escherichia_coli",
        (
            "apex_mic_um__e_coli_atcc_11775",
            "apex_mic_um__e_coli_aic221",
            "apex_mic_um__e_coli_aic222",
        ),
    ),
    ("klebsiella_pneumoniae", ("apex_mic_um__k_pneumoniae_atcc_13883",)),
    (
        "pseudomonas_aeruginosa",
        ("apex_mic_um__p_aeruginosa_pa01", "apex_mic_um__p_aeruginosa_pa14"),
    ),
    (
        "staphylococcus_aureus",
        (
            "apex_mic_um__s_aureus_atcc_12600",
            "apex_mic_um__s_aureus_atcc_baa_1556_mrsa",
        ),
    ),
    (
        "enterococcus_faecalis",
        ("apex_mic_um__vancomycin_resistant_e_faecalis_atcc_700802",),
    ),
    (
        "enterococcus_faecium",
        ("apex_mic_um__vancomycin_resistant_e_faecium_atcc_700221",),
    ),
)
_RAW_PREDICTION_SCHEMA = (
    "sequence_id",
    "sequence",
    "model_family",
    "model_version",
    "member_id",
    "member_checkpoint_sha256",
    "apex_member_mean_mic_um",
    *_APEX_ENDPOINT_COLUMNS,
)
_RECONCILIATION_SCHEMA = (
    "sequence_id",
    "sequence",
    "endpoint",
    "member_count",
    "introspected_mean_mic_um",
    "fidelity_mean_mic_um",
    "absolute_error",
    "relative_error",
    "allowed_error",
    "passed",
)
_EXAMPLE_FIELDS = frozenset(
    {
        "assay_context_id",
        "canonical_target",
        "example_id",
        "fold",
        "gram",
        "homology_component_id",
        "label",
        "schema_version",
        "sequence",
        "sequence_id",
        "source_observations",
        "union_component_id",
    }
)
_BASE_OOF_SCHEMA = (
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
)
_BASE_SEMANTIC_ENTRIES = frozenset(
    {
        "context_audit.jsonl",
        "examples.jsonl",
        "folds.json",
        "manifest.json",
        "metrics.json",
        "oof_predictions.csv",
        "split_receipt.json",
    }
)
_BASE_PUBLICATION_ENTRIES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "gate1/SHA256SUMS",
        *(f"gate1/{name}" for name in _BASE_SEMANTIC_ENTRIES),
    }
)
_BASE_TREE_FILES = _BASE_PUBLICATION_ENTRIES | {"SHA256SUMS"}
_APEX_TREE_FILES = frozenset(
    {
        "apex_member_predictions.csv",
        "apex_member_run_manifest.json",
        "apex_member_reconciliation.csv",
    }
)
_FROZEN_INPUT_ENTRIES = frozenset(
    {
        *(f"base/{name}" for name in _BASE_TREE_FILES),
        "base-independent-receipt.json",
        *(f"apex/{name}" for name in _APEX_TREE_FILES),
    }
)
_INDEPENDENT_CHECKS = frozenset(
    {
        "accepted_split_receipt_chain_valid",
        "all_semantic_artifacts_exact",
        "assignment_and_component_join_recomputed",
        "canonical_serialization_verified",
        "context_partition_recomputed",
        "cross_fold_identity_recomputed",
        "descriptor_logistic_oof_recomputed",
        "ensemble_mean_exact",
        "executing_source_bound_to_repository",
        "homology_knn_oof_recomputed",
        "input_twins_and_frozen_manifests_valid",
        "metrics_and_union_bootstrap_recomputed",
        "production_overlap_handshake_valid",
        "publication_top_manifests_valid",
        "repository_commit_and_cleanliness_verified",
        "scratch_and_absolute_paths_absent",
        "twins_byte_identical",
    }
)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    endpoints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemberSpec:
    member_id: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ApexUnionConfig:
    path: Path
    artifact: str
    activity_threshold_um: float
    folds: int
    homology_identity_threshold: float
    calibration_bins: int
    probability_clip_epsilon: float
    bootstrap_replicates: int
    bootstrap_seed: int
    calibration_weighting: str
    bootstrap_weighting: str
    expected_examples: int
    expected_source_observations: int
    expected_positive_examples: int
    expected_negative_examples: int
    expected_modeled_sequences: int
    expected_modeled_homology_components: int
    expected_modeled_union_components: int
    expected_examples_by_fold: tuple[int, ...]
    expected_positive_examples_by_fold: tuple[int, ...]
    expected_negative_examples_by_fold: tuple[int, ...]
    expected_output_rows: int
    expected_union_sequence_ids_sha256: str
    expected_base_publication_top_sha256: str
    expected_base_semantic_top_sha256: str
    expected_base_examples_sha256: str
    expected_base_folds_sha256: str
    expected_base_oof_sha256: str
    expected_base_manifest_sha256: str
    expected_base_split_receipt_sha256: str
    expected_base_independent_receipt_sha256: str
    expected_base_config_sha256: str
    expected_apex_predictions_sha256: str
    expected_apex_manifest_sha256: str
    expected_apex_reconciliation_sha256: str
    expected_apex_sequence_ids_sha256: str
    expected_apex_source_commit: str
    expected_apex_sequences: int
    expected_apex_extra_sequences: int
    expected_member_count: int
    expected_reconciliation_comparisons: int
    calibration_l2: float
    calibration_prior_strength: float
    calibration_max_iterations: int
    calibration_tolerance: float
    targets: tuple[TargetSpec, ...]
    members: tuple[MemberSpec, ...]

    @property
    def target_by_name(self) -> dict[str, TargetSpec]:
        return {item.name: item for item in self.targets}


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Example:
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


@dataclass(frozen=True, slots=True)
class RawApexPredictions:
    members: tuple[str, ...]
    member_hashes: Mapping[str, str]
    values: Mapping[str, Mapping[str, Mapping[str, float]]]
    sequences: Mapping[str, str]
    rows: int


@dataclass(frozen=True, slots=True)
class SupportedExample:
    example: Example
    target: TargetSpec
    signals: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Calibrator:
    heldout_fold: int
    excluded_folds: tuple[int, ...]
    training_folds: tuple[int, ...]
    member_id: str
    member_checkpoint_sha256: str
    weighting_policy: str
    training_examples: int
    training_positives: int
    training_negatives: int
    training_sequences: int
    training_union_components: int
    training_example_ids_sha256: str
    training_sequence_ids_sha256: str
    training_union_component_ids_sha256: str
    training_assignment_sha256: str
    iterations: int
    converged: bool
    signal_mean: float
    signal_scale: float
    intercept: float
    slope: float
    probability_clip_epsilon: float

    def predict(self, signal: float) -> float:
        logit = self.intercept + self.slope * ((signal - self.signal_mean) / self.signal_scale)
        return float(
            np.clip(
                _sigmoid(logit),
                self.probability_clip_epsilon,
                1.0 - self.probability_clip_epsilon,
            )
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    model: str
    member_id: str
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
    apex_endpoints: str
    activity_signal: float
    probability: float
    member_probability_std: float | None


@dataclass(frozen=True, slots=True)
class ApexUnionExecution:
    output_dir: Path
    predictions_path: Path
    calibrators_path: Path
    coverage_path: Path
    metrics_path: Path
    manifest_path: Path
    top_manifest_path: Path
    examples: int
    prediction_rows: int


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


def _read_snapshot(path: str | Path, *, name: str) -> Snapshot:
    unresolved = Path(path).absolute()
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be symbolic")
    source = unresolved.resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} is not a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    if _fingerprint(before) != _fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{name} changed while it was being read")
    return Snapshot(
        path=source,
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=_fingerprint(after),
    )


def _assert_snapshot_unchanged(snapshot: Snapshot, *, name: str) -> None:
    current = snapshot.path.stat()
    if (
        _fingerprint(current) != snapshot.fingerprint
        or _file_sha256(snapshot.path) != snapshot.sha256
    ):
        raise ValueError(f"{name} changed during execution")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json(payload: bytes, *, name: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not valid strict JSON") from error


def _parse_json_object(payload: bytes, *, name: str) -> dict[str, object]:
    value = _parse_json(payload, name=name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _parse_jsonl(payload: bytes, *, name: str) -> list[dict[str, object]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    if payload and not text.endswith("\n"):
        raise ValueError(f"{name} must end with LF")
    output: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{name} contains a blank line at {line_number}")
        value = _parse_json_object(line.encode("utf-8"), name=f"{name} line {line_number}")
        output.append(value)
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _parse_sha256_manifest(payload: bytes, *, name: str) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise ValueError(f"{name} must use canonical LF-terminated lines")
    entries: dict[str, str] = {}
    previous: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise ValueError(f"{name} line {line_number} is invalid")
        digest, mode, logical = match.groups()
        path = PurePosixPath(logical)
        if (
            mode != " "
            or path.is_absolute()
            or path.as_posix() != logical
            or "\\" in logical
            or any(part in {"", ".", ".."} for part in path.parts)
            or logical in entries
            or (previous is not None and logical <= previous)
        ):
            raise ValueError(f"{name} contains unsafe or duplicate path {logical!r}")
        entries[logical] = digest
        previous = logical
    if not entries:
        raise ValueError(f"{name} must contain at least one entry")
    return entries


def _require_exact_fields(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ from the required schema")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _int_tuple(value: object, *, field: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} integers")
    return tuple(_nonnegative_int(item, field=field) for item in value)


def _config_from_payload(source: Path, payload: bytes) -> ApexUnionConfig:
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("APEX union config is not valid UTF-8 TOML") from error
    expected = {
        "schema_version",
        "artifact",
        "activity_threshold_um",
        "folds",
        "homology_identity_threshold",
        "calibration_bins",
        "probability_clip_epsilon",
        "bootstrap_replicates",
        "bootstrap_seed",
        "calibration_weighting",
        "bootstrap_weighting",
        "expected_examples",
        "expected_source_observations",
        "expected_positive_examples",
        "expected_negative_examples",
        "expected_modeled_sequences",
        "expected_modeled_homology_components",
        "expected_modeled_union_components",
        "expected_examples_by_fold",
        "expected_positive_examples_by_fold",
        "expected_negative_examples_by_fold",
        "expected_output_rows",
        "expected_union_sequence_ids_sha256",
        "expected_base_publication_top_sha256",
        "expected_base_semantic_top_sha256",
        "expected_base_examples_sha256",
        "expected_base_folds_sha256",
        "expected_base_oof_sha256",
        "expected_base_manifest_sha256",
        "expected_base_split_receipt_sha256",
        "expected_base_independent_receipt_sha256",
        "expected_base_config_sha256",
        "expected_apex_predictions_sha256",
        "expected_apex_manifest_sha256",
        "expected_apex_reconciliation_sha256",
        "expected_apex_sequence_ids_sha256",
        "expected_apex_source_commit",
        "expected_apex_sequences",
        "expected_apex_extra_sequences",
        "expected_member_count",
        "expected_reconciliation_comparisons",
        "calibration",
        "target",
        "member",
    }
    if set(raw) != expected or raw.get("schema_version") != 1:
        raise ValueError("APEX union config fields or schema_version differ from v1")
    if raw["artifact"] != _ARTIFACT:
        raise ValueError("APEX union config artifact identity is invalid")
    if raw["calibration_weighting"] != _CALIBRATION_WEIGHTING:
        raise ValueError("APEX union config must use context_equal_v1 calibration weighting")
    if raw["bootstrap_weighting"] != _BOOTSTRAP_WEIGHTING:
        raise ValueError("APEX union config has the wrong union-component bootstrap weighting")
    folds = _positive_int(raw["folds"], field="folds")
    if folds != 5:
        raise ValueError("APEX union v1 requires exactly five accepted outer folds")
    arrays = {
        name: _int_tuple(raw[name], field=name, length=folds)
        for name in (
            "expected_examples_by_fold",
            "expected_positive_examples_by_fold",
            "expected_negative_examples_by_fold",
        )
    }
    expected_examples = _positive_int(raw["expected_examples"], field="expected_examples")
    expected_positives = _positive_int(
        raw["expected_positive_examples"], field="expected_positive_examples"
    )
    expected_negatives = _positive_int(
        raw["expected_negative_examples"], field="expected_negative_examples"
    )
    member_count = _positive_int(raw["expected_member_count"], field="expected_member_count")
    if member_count != 8:
        raise ValueError("APEX union v1 requires the frozen eight-member ensemble")
    output_rows = _positive_int(raw["expected_output_rows"], field="expected_output_rows")
    if (
        expected_positives + expected_negatives != expected_examples
        or sum(arrays["expected_examples_by_fold"]) != expected_examples
        or sum(arrays["expected_positive_examples_by_fold"]) != expected_positives
        or sum(arrays["expected_negative_examples_by_fold"]) != expected_negatives
        or any(
            arrays["expected_positive_examples_by_fold"][fold]
            + arrays["expected_negative_examples_by_fold"][fold]
            != arrays["expected_examples_by_fold"][fold]
            for fold in range(folds)
        )
        or output_rows != expected_examples * (member_count + 1)
    ):
        raise ValueError("APEX union config census or output-row arithmetic is inconsistent")

    calibration = raw["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "l2",
        "prior_strength",
        "max_iterations",
        "tolerance",
    }:
        raise ValueError("APEX union calibration table differs from v1")
    targets_raw = raw["target"]
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("APEX union config requires target tables")
    targets: list[TargetSpec] = []
    endpoint_owner: dict[str, str] = {}
    for index, item in enumerate(targets_raw):
        if not isinstance(item, dict) or set(item) != {"name", "endpoints"}:
            raise ValueError(f"target {index} has an invalid schema")
        name = item["name"]
        endpoints = item["endpoints"]
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(endpoints, list)
            or not endpoints
            or not all(
                isinstance(value, str) and value in _APEX_ENDPOINT_COLUMNS for value in endpoints
            )
        ):
            raise ValueError(f"target {index} is invalid")
        endpoint_tuple = tuple(cast(list[str], endpoints))
        if len(endpoint_tuple) != len(set(endpoint_tuple)):
            raise ValueError(f"target {name!r} repeats an endpoint")
        for endpoint in endpoint_tuple:
            if endpoint in endpoint_owner:
                raise ValueError(f"endpoint {endpoint!r} belongs to multiple targets")
            endpoint_owner[endpoint] = name
        targets.append(TargetSpec(name=name, endpoints=endpoint_tuple))
    if len({item.name for item in targets}) != len(targets):
        raise ValueError("APEX union target names must be unique")
    if tuple((item.name, item.endpoints) for item in targets) != _TARGET_ENDPOINTS:
        raise ValueError("APEX union config target-to-endpoint map differs from v1")

    members_raw = raw["member"]
    if not isinstance(members_raw, list) or len(members_raw) != member_count:
        raise ValueError("APEX union member tables differ from expected_member_count")
    members: list[MemberSpec] = []
    for index, item in enumerate(members_raw):
        if not isinstance(item, dict) or set(item) != {"member_id", "checkpoint_sha256"}:
            raise ValueError(f"member {index} has an invalid schema")
        member_id = item["member_id"]
        if not isinstance(member_id, str) or not member_id:
            raise ValueError(f"member {index} has an invalid ID")
        members.append(
            MemberSpec(
                member_id=member_id,
                checkpoint_sha256=_require_sha256(
                    item["checkpoint_sha256"], field=f"member {index} checkpoint"
                ),
            )
        )
    if len({item.member_id for item in members}) != member_count:
        raise ValueError("APEX union member IDs must be unique")

    hash_fields = (
        "expected_union_sequence_ids_sha256",
        "expected_base_publication_top_sha256",
        "expected_base_semantic_top_sha256",
        "expected_base_examples_sha256",
        "expected_base_folds_sha256",
        "expected_base_oof_sha256",
        "expected_base_manifest_sha256",
        "expected_base_split_receipt_sha256",
        "expected_base_independent_receipt_sha256",
        "expected_base_config_sha256",
        "expected_apex_predictions_sha256",
        "expected_apex_manifest_sha256",
        "expected_apex_reconciliation_sha256",
        "expected_apex_sequence_ids_sha256",
    )
    hashes = {name: _require_sha256(raw[name], field=name) for name in hash_fields}
    source_commit = raw["expected_apex_source_commit"]
    if not isinstance(source_commit, str) or _GIT_SHA1.fullmatch(source_commit) is None:
        raise ValueError("expected_apex_source_commit must be a lowercase full Git SHA-1")
    expected_sequences = _positive_int(
        raw["expected_apex_sequences"], field="expected_apex_sequences"
    )
    modeled_sequences = _positive_int(
        raw["expected_modeled_sequences"], field="expected_modeled_sequences"
    )
    extra_sequences = _nonnegative_int(
        raw["expected_apex_extra_sequences"], field="expected_apex_extra_sequences"
    )
    if expected_sequences != modeled_sequences + extra_sequences:
        raise ValueError("APEX sequence census does not imply complete union coverage")
    probability_clip_epsilon = _positive_float(
        raw["probability_clip_epsilon"], field="probability_clip_epsilon"
    )
    if probability_clip_epsilon >= 0.5:
        raise ValueError("probability_clip_epsilon must be strictly below 0.5")

    activity_threshold = _positive_float(
        raw["activity_threshold_um"], field="activity_threshold_um"
    )
    homology_threshold = _positive_float(
        raw["homology_identity_threshold"], field="homology_identity_threshold"
    )
    calibration_bins = _positive_int(raw["calibration_bins"], field="calibration_bins")
    bootstrap_replicates = _positive_int(raw["bootstrap_replicates"], field="bootstrap_replicates")
    bootstrap_seed = _nonnegative_int(raw["bootstrap_seed"], field="bootstrap_seed")
    if (
        activity_threshold != 16.0
        or homology_threshold != 0.8
        or calibration_bins < 2
        or bootstrap_replicates != 1000
        or bootstrap_seed != 42
    ):
        raise ValueError("APEX union v1 threshold, metric, or bootstrap constants differ")

    return ApexUnionConfig(
        path=source,
        artifact=_ARTIFACT,
        activity_threshold_um=activity_threshold,
        folds=folds,
        homology_identity_threshold=homology_threshold,
        calibration_bins=calibration_bins,
        probability_clip_epsilon=probability_clip_epsilon,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        calibration_weighting=_CALIBRATION_WEIGHTING,
        bootstrap_weighting=_BOOTSTRAP_WEIGHTING,
        expected_examples=expected_examples,
        expected_source_observations=_positive_int(
            raw["expected_source_observations"], field="expected_source_observations"
        ),
        expected_positive_examples=expected_positives,
        expected_negative_examples=expected_negatives,
        expected_modeled_sequences=modeled_sequences,
        expected_modeled_homology_components=_positive_int(
            raw["expected_modeled_homology_components"],
            field="expected_modeled_homology_components",
        ),
        expected_modeled_union_components=_positive_int(
            raw["expected_modeled_union_components"], field="expected_modeled_union_components"
        ),
        expected_examples_by_fold=arrays["expected_examples_by_fold"],
        expected_positive_examples_by_fold=arrays["expected_positive_examples_by_fold"],
        expected_negative_examples_by_fold=arrays["expected_negative_examples_by_fold"],
        expected_output_rows=output_rows,
        **hashes,
        expected_apex_source_commit=source_commit,
        expected_apex_sequences=expected_sequences,
        expected_apex_extra_sequences=extra_sequences,
        expected_member_count=member_count,
        expected_reconciliation_comparisons=_positive_int(
            raw["expected_reconciliation_comparisons"],
            field="expected_reconciliation_comparisons",
        ),
        calibration_l2=_positive_float(calibration["l2"], field="calibration.l2"),
        calibration_prior_strength=_nonnegative_float(
            calibration["prior_strength"], field="calibration.prior_strength"
        ),
        calibration_max_iterations=_positive_int(
            calibration["max_iterations"], field="calibration.max_iterations"
        ),
        calibration_tolerance=_positive_float(
            calibration["tolerance"], field="calibration.tolerance"
        ),
        targets=tuple(targets),
        members=tuple(members),
    )


def load_config(path: str | Path) -> ApexUnionConfig:
    """Load and strictly validate a union-specific APEX configuration."""

    snapshot = _read_snapshot(path, name="APEX union config")
    return _config_from_payload(snapshot.path, snapshot.payload)


def _sequence_set_sha256(values: Iterable[str]) -> str:
    items = sorted(set(values))
    payload = b"" if not items else ("\n".join(items) + "\n").encode("ascii")
    return _sha256_bytes(payload)


def _assignment_sha256(examples: Iterable[Example]) -> str:
    records = [
        "\t".join(
            (
                item.example_id,
                item.sequence_id,
                str(item.label),
                str(item.fold),
                item.homology_component_id,
                item.union_component_id,
                item.canonical_target,
            )
        )
        for item in sorted(examples, key=lambda value: value.example_id)
    ]
    return _sha256_bytes(("\n".join(records) + "\n").encode("utf-8"))


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _snapshot_exact_tree(
    root_value: str | Path, *, expected_files: frozenset[str], name: str
) -> tuple[Path, dict[str, Snapshot]]:
    unresolved = Path(root_value).absolute()
    if unresolved.is_symlink():
        raise ValueError(f"{name} root must not be symbolic")
    root = unresolved.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{name} root must be a directory")
    files: set[str] = set()
    directories: set[str] = set()
    for entry in root.rglob("*"):
        logical = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise ValueError(f"{name} contains symbolic entry {logical!r}")
        if entry.is_file():
            files.add(logical)
        elif entry.is_dir():
            directories.add(logical)
        else:
            raise ValueError(f"{name} contains non-file entry {logical!r}")
    expected_directories = {
        PurePosixPath(filename).parent.as_posix()
        for filename in expected_files
        if PurePosixPath(filename).parent.as_posix() != "."
    }
    if files != set(expected_files) or directories != expected_directories:
        raise ValueError(
            f"{name} inventory mismatch: missing={sorted(set(expected_files) - files)}, "
            f"extra={sorted(files - set(expected_files))}, "
            f"directory_difference={sorted(directories ^ expected_directories)}"
        )
    return root, {
        logical: _read_snapshot(root / logical, name=f"{name} {logical}")
        for logical in sorted(expected_files)
    }


def _validate_code_manifest(
    manifest: Snapshot, *, config_snapshot: Snapshot
) -> tuple[dict[str, object], tuple[Snapshot, ...]]:
    entries = _parse_sha256_manifest(manifest.payload, name="code manifest")
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
    verified: list[Snapshot] = []
    for logical in sorted(expected_paths):
        if logical == _LOGICAL_CONFIG_PATH:
            snapshot = config_snapshot
        else:
            snapshot = _read_snapshot(
                repository_root / logical, name=f"code inventory entry {logical}"
            )
            verified.append(snapshot)
        if entries[logical] != snapshot.sha256:
            raise ValueError(f"code manifest checksum mismatch for {logical}")
    return (
        {
            "schema_version": 1,
            "code_manifest_sha256": manifest.sha256,
            "inventory_entries": len(entries),
            "executing_module": {
                "logical_path": _EXECUTING_MODULE_PATH,
                "sha256": entries[_EXECUTING_MODULE_PATH],
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
        },
        tuple(verified),
    )


def _verify_frozen_input_manifest(manifest: Snapshot, *, snapshots: Mapping[str, Snapshot]) -> None:
    entries = _parse_sha256_manifest(manifest.payload, name="frozen input manifest")
    if set(entries) != set(_FROZEN_INPUT_ENTRIES):
        raise ValueError("frozen input manifest inventory differs from the required inputs")
    for logical, expected in entries.items():
        if snapshots[logical].sha256 != expected:
            raise ValueError(f"frozen input checksum mismatch for {logical}")


def _expect_hash(snapshot: Snapshot, expected: str, *, name: str) -> None:
    if snapshot.sha256 != expected:
        raise ValueError(f"{name} SHA-256 differs from the frozen config")


def _strict_csv_rows(payload: bytes, *, schema: Sequence[str], name: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    if not text.endswith("\n") or "\r" in text or text.startswith("\ufeff"):
        raise ValueError(f"{name} must be canonical LF-terminated UTF-8 without a BOM")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(schema):
        raise ValueError(f"{name} does not have the frozen schema")
    rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{name} is empty or has ragged rows")
    return cast(list[dict[str, str]], rows)


def _parse_int_text(value: str, *, field: str, minimum: int = 0) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"{field} must be a canonical non-negative integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} is below its minimum")
    return result


def _parse_float_text(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _fold_summary(examples: Sequence[Example], folds: int) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for fold in range(folds):
        selected = [item for item in examples if item.fold == fold]
        output[str(fold)] = {
            "examples": len(selected),
            "gram_negative_mic16_negative": sum(
                item.gram == "negative" and item.label == 0 for item in selected
            ),
            "gram_negative_mic16_positive": sum(
                item.gram == "negative" and item.label == 1 for item in selected
            ),
            "gram_positive_mic16_negative": sum(
                item.gram == "positive" and item.label == 0 for item in selected
            ),
            "gram_positive_mic16_positive": sum(
                item.gram == "positive" and item.label == 1 for item in selected
            ),
            "homology_components": len({item.homology_component_id for item in selected}),
            "negatives": sum(item.label == 0 for item in selected),
            "positives": sum(item.label == 1 for item in selected),
            "sequences": len({item.sequence_id for item in selected}),
            "source_observations": sum(item.source_observations for item in selected),
            "union_components": len({item.union_component_id for item in selected}),
        }
    return output


def _target_summary(examples: Sequence[Example], folds: int) -> dict[str, dict[str, int]]:
    return {
        str(fold): {
            target: sum(item.fold == fold and item.canonical_target == target for item in examples)
            for target in sorted({item.canonical_target for item in examples if item.fold == fold})
        }
        for fold in range(folds)
    }


def _read_examples(
    examples_snapshot: Snapshot,
    oof_snapshot: Snapshot,
    *,
    config: ApexUnionConfig,
) -> tuple[Example, ...]:
    documents = _parse_jsonl(examples_snapshot.payload, name="accepted union examples")
    canonical_payload = "".join(_canonical_json(item) + "\n" for item in documents).encode("utf-8")
    if canonical_payload != examples_snapshot.payload:
        raise ValueError("accepted union examples do not use canonical JSONL serialization")
    raw_by_id: dict[str, dict[str, object]] = {}
    previous_id: str | None = None
    for index, row in enumerate(documents):
        _require_exact_fields(row, set(_EXAMPLE_FIELDS), name=f"accepted example {index}")
        if row.get("schema_version") != 1:
            raise ValueError(f"accepted example {index} has the wrong schema version")
        example_id = _require_sha256(row.get("example_id"), field=f"example {index} ID")
        assay_context_id = _require_sha256(
            row.get("assay_context_id"), field=f"example {index} assay context"
        )
        if example_id != assay_context_id or example_id in raw_by_id:
            raise ValueError(f"accepted example {index} has a duplicate or mismatched ID")
        if previous_id is not None and example_id <= previous_id:
            raise ValueError("accepted union examples must be strictly sorted by example_id")
        previous_id = example_id
        sequence = row.get("sequence")
        if not isinstance(sequence, str) or canonicalize_sequence(sequence) != sequence:
            raise ValueError(f"accepted example {index} has a noncanonical sequence")
        sequence_id = _require_sha256(row.get("sequence_id"), field=f"example {index} sequence ID")
        if canonical_sequence_id(sequence) != sequence_id:
            raise ValueError(f"accepted example {index} sequence ID does not match its sequence")
        target = row.get("canonical_target")
        gram = row.get("gram")
        label = row.get("label")
        fold = row.get("fold")
        source_observations = row.get("source_observations")
        if not isinstance(target, str) or target not in config.target_by_name:
            raise ValueError(f"accepted example {index} has an unsupported canonical target")
        if gram not in {"positive", "negative"}:
            raise ValueError(f"accepted example {index} has an invalid Gram class")
        if not isinstance(label, int) or isinstance(label, bool) or label not in {0, 1}:
            raise ValueError(f"accepted example {index} has an invalid label")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(config.folds):
            raise ValueError(f"accepted example {index} has an invalid fold")
        _positive_int(source_observations, field=f"example {index} source_observations")
        _require_sha256(
            row.get("homology_component_id"), field=f"example {index} homology component"
        )
        _require_sha256(row.get("union_component_id"), field=f"example {index} union component")
        raw_by_id[example_id] = row

    oof_rows = _strict_csv_rows(
        oof_snapshot.payload,
        schema=_BASE_OOF_SCHEMA,
        name="accepted union OOF metadata",
    )
    identities: dict[str, float] = {}
    support_by_model: dict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(oof_rows, start=2):
        model = row["model"]
        example_id = row["example_id"]
        source = raw_by_id.get(example_id)
        if model not in _BASE_MODELS or source is None or example_id in support_by_model[model]:
            raise ValueError(f"accepted OOF row {row_number} has invalid model/support identity")
        support_by_model[model].add(example_id)
        expected_strings = {
            "assay_context_id": source["assay_context_id"],
            "sequence_id": source["sequence_id"],
            "sequence": source["sequence"],
            "canonical_target": source["canonical_target"],
            "gram": source["gram"],
            "homology_component_id": source["homology_component_id"],
            "union_component_id": source["union_component_id"],
        }
        if any(row[field] != value for field, value in expected_strings.items()):
            raise ValueError(f"accepted OOF row {row_number} metadata differs from examples")
        if (
            _parse_int_text(row["label"], field=f"OOF row {row_number} label") != source["label"]
            or _parse_int_text(row["source_observations"], field=f"OOF row {row_number} source")
            != source["source_observations"]
            or _parse_int_text(row["fold"], field=f"OOF row {row_number} fold") != source["fold"]
        ):
            raise ValueError(f"accepted OOF row {row_number} numeric metadata differs")
        identity = _parse_float_text(
            row["max_train_identity"], field=f"OOF row {row_number} max_train_identity"
        )
        probability = _parse_float_text(
            row["probability"], field=f"OOF row {row_number} probability"
        )
        if not 0 <= identity < config.homology_identity_threshold or not 0 <= probability <= 1:
            raise ValueError(f"accepted OOF row {row_number} has an invalid identity/probability")
        previous = identities.setdefault(example_id, identity)
        if previous != identity:
            raise ValueError("accepted base models disagree on max_train_identity metadata")
    expected_ids = set(raw_by_id)
    if set(support_by_model) != set(_BASE_MODELS) or any(
        ids != expected_ids for ids in support_by_model.values()
    ):
        raise ValueError("accepted OOF models do not share the complete example support")

    examples = tuple(
        Example(
            example_id=example_id,
            assay_context_id=cast(str, row["assay_context_id"]),
            sequence_id=cast(str, row["sequence_id"]),
            sequence=cast(str, row["sequence"]),
            canonical_target=cast(str, row["canonical_target"]),
            gram=cast(GramClass, row["gram"]),
            label=cast(int, row["label"]),
            source_observations=cast(int, row["source_observations"]),
            fold=cast(int, row["fold"]),
            homology_component_id=cast(str, row["homology_component_id"]),
            union_component_id=cast(str, row["union_component_id"]),
            max_train_identity=identities[example_id],
        )
        for example_id, row in raw_by_id.items()
    )
    _verify_example_census(examples, config=config)
    return examples


def _verify_example_census(examples: Sequence[Example], *, config: ApexUnionConfig) -> None:
    positives = sum(item.label for item in examples)
    sequence_ids = {item.sequence_id for item in examples}
    homology_ids = {item.homology_component_id for item in examples}
    union_ids = {item.union_component_id for item in examples}
    if (
        len(examples) != config.expected_examples
        or positives != config.expected_positive_examples
        or len(examples) - positives != config.expected_negative_examples
        or sum(item.source_observations for item in examples) != config.expected_source_observations
        or len(sequence_ids) != config.expected_modeled_sequences
        or len(homology_ids) != config.expected_modeled_homology_components
        or len(union_ids) != config.expected_modeled_union_components
        or _sequence_set_sha256(sequence_ids) != config.expected_union_sequence_ids_sha256
    ):
        raise ValueError("accepted modeled example census differs from the frozen config")
    by_fold = _fold_summary(examples, config.folds)
    if (
        tuple(by_fold[str(fold)]["examples"] for fold in range(config.folds))
        != config.expected_examples_by_fold
        or tuple(by_fold[str(fold)]["positives"] for fold in range(config.folds))
        != config.expected_positive_examples_by_fold
        or tuple(by_fold[str(fold)]["negatives"] for fold in range(config.folds))
        != config.expected_negative_examples_by_fold
    ):
        raise ValueError("accepted modeled fold census differs from the frozen config")
    sequence_assignments: dict[str, set[tuple[int, str, str, str]]] = defaultdict(set)
    homology_folds: dict[str, set[int]] = defaultdict(set)
    homology_union: dict[str, set[str]] = defaultdict(set)
    union_folds: dict[str, set[int]] = defaultdict(set)
    for item in examples:
        sequence_assignments[item.sequence_id].add(
            (
                item.fold,
                item.homology_component_id,
                item.union_component_id,
                item.sequence,
            )
        )
        homology_folds[item.homology_component_id].add(item.fold)
        homology_union[item.homology_component_id].add(item.union_component_id)
        union_folds[item.union_component_id].add(item.fold)
    if (
        any(len(values) != 1 for values in sequence_assignments.values())
        or any(len(values) != 1 for values in homology_folds.values())
        or any(len(values) != 1 for values in homology_union.values())
        or any(len(values) != 1 for values in union_folds.values())
    ):
        raise ValueError("accepted modeled sequence/component assignments leak across folds")
    for fold in range(config.folds):
        labels = {item.label for item in examples if item.fold == fold}
        complement = {item.label for item in examples if item.fold != fold}
        if labels != {0, 1} or complement != {0, 1}:
            raise ValueError(f"fold {fold} or its calibration complement lacks a label class")


def _verify_folds(
    snapshot: Snapshot, *, examples: Sequence[Example], config: ApexUnionConfig
) -> dict[str, object]:
    document = _parse_json_object(snapshot.payload, name="accepted union folds")
    _require_exact_fields(
        document,
        {
            "artifact",
            "assignment_policy",
            "assignments",
            "canonical_targets_by_fold",
            "folds",
            "identity_threshold",
            "maximum_cross_fold_identity",
            "schema_version",
        },
        name="accepted union folds",
    )
    if (
        document.get("schema_version") != 1
        or document.get("artifact") != "gate1_context_union_fold_reuse"
        or document.get("assignment_policy") != _ASSIGNMENT_POLICY
        or document.get("identity_threshold") != config.homology_identity_threshold
        or document.get("maximum_cross_fold_identity")
        != max(item.max_train_identity for item in examples)
    ):
        raise ValueError("accepted union folds identity or policy contract is invalid")
    assignments = document.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(examples):
        raise ValueError("accepted union fold assignments have the wrong census")
    expected = {
        item.example_id: {
            "example_id": item.example_id,
            "fold": item.fold,
            "homology_component_id": item.homology_component_id,
            "sequence_id": item.sequence_id,
            "union_component_id": item.union_component_id,
        }
        for item in examples
    }
    observed: dict[str, object] = {}
    for index, item in enumerate(assignments):
        if not isinstance(item, dict) or set(item) != {
            "example_id",
            "fold",
            "homology_component_id",
            "sequence_id",
            "union_component_id",
        }:
            raise ValueError(f"accepted union fold assignment {index} has an invalid schema")
        example_id = item.get("example_id")
        if not isinstance(example_id, str) or example_id in observed:
            raise ValueError(f"accepted union fold assignment {index} is duplicate/invalid")
        observed[example_id] = item
    if observed != expected:
        raise ValueError("accepted union fold assignments differ from examples")
    folds = _fold_summary(examples, config.folds)
    targets = _target_summary(examples, config.folds)
    if document.get("folds") != folds or document.get("canonical_targets_by_fold") != targets:
        raise ValueError("accepted union fold summaries differ from modeled examples")
    return document


def _verify_base_documents(
    snapshots: Mapping[str, Snapshot],
    independent_snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    folds_document: Mapping[str, object],
    config: ApexUnionConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    outer = _parse_sha256_manifest(snapshots["SHA256SUMS"].payload, name="publication top")
    semantic = _parse_sha256_manifest(
        snapshots["gate1/SHA256SUMS"].payload, name="Gate-1 semantic top"
    )
    if set(outer) != set(_BASE_PUBLICATION_ENTRIES) or set(semantic) != set(_BASE_SEMANTIC_ENTRIES):
        raise ValueError("accepted publication manifests have unexpected inventories")
    for logical, digest in outer.items():
        if snapshots[logical].sha256 != digest:
            raise ValueError(f"accepted publication checksum mismatch for {logical}")
    for logical, digest in semantic.items():
        if snapshots[f"gate1/{logical}"].sha256 != digest:
            raise ValueError(f"accepted semantic checksum mismatch for {logical}")
        if outer[f"gate1/{logical}"] != digest:
            raise ValueError(f"accepted top manifests disagree for {logical}")
    for logical in ("CODE_SHA256SUMS", "FROZEN_INPUT_SHA256SUMS"):
        _parse_sha256_manifest(snapshots[logical].payload, name=f"accepted {logical}")

    manifest = _parse_json_object(
        snapshots["gate1/manifest.json"].payload, name="accepted Gate-1 manifest"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact") != _BASE_ARTIFACT
        or manifest.get("status") != _BASE_STATUS
        or manifest.get("config_sha256") != config.expected_base_config_sha256
        or manifest.get("models") != list(_BASE_MODELS)
    ):
        raise ValueError("accepted Gate-1 manifest identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("accepted Gate-1 manifest artifacts table is missing")
    expected_artifact_entries = {
        "context_audit": "context_audit.jsonl",
        "examples": "examples.jsonl",
        "folds": "folds.json",
        "metrics": "metrics.json",
        "oof": "oof_predictions.csv",
        "split_receipt": "split_receipt.json",
    }
    if set(artifacts) != set(expected_artifact_entries):
        raise ValueError("accepted Gate-1 manifest artifact inventory is invalid")
    for key, filename in expected_artifact_entries.items():
        entry = artifacts[key]
        if (
            not isinstance(entry, dict)
            or entry.get("filename") != filename
            or entry.get("sha256") != semantic[filename]
        ):
            raise ValueError(f"accepted Gate-1 manifest artifact {key!r} is invalid")
    label_summary = manifest.get("label_summary")
    if not isinstance(label_summary, dict) or (
        label_summary.get("context_examples") != len(examples)
        or label_summary.get("positive_examples") != sum(item.label for item in examples)
        or label_summary.get("negative_examples") != sum(1 - item.label for item in examples)
        or label_summary.get("source_observations")
        != sum(item.source_observations for item in examples)
        or label_summary.get("modeled_sequences") != len({item.sequence_id for item in examples})
        or label_summary.get("by_fold") != folds_document["folds"]
        or label_summary.get("canonical_targets_by_fold")
        != folds_document["canonical_targets_by_fold"]
    ):
        raise ValueError("accepted Gate-1 manifest label summary is invalid")

    receipt = _parse_json_object(
        snapshots["gate1/split_receipt.json"].payload, name="accepted split receipt"
    )
    invariants = receipt.get("invariants")
    census = receipt.get("census")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact") != "gate1_union_accepted_split_consumption_receipt"
        or receipt.get("status") != "passed"
        or receipt.get("assignment_policy") != _ASSIGNMENT_POLICY
        or not isinstance(invariants, dict)
        or not invariants
        or any(value is not True for value in invariants.values())
        or not isinstance(census, dict)
        or census.get("context_examples") != config.expected_examples
        or census.get("source_observations") != config.expected_source_observations
        or census.get("modeled_sequences") != config.expected_modeled_sequences
        or census.get("homology_components") != 597
        or census.get("union_components") != 278
        or census.get("parser_sequences") != 1113
    ):
        raise ValueError("accepted split receipt contract is invalid")

    independent = _parse_json_object(
        independent_snapshot.payload, name="accepted independent verification receipt"
    )
    checks = independent.get("checks")
    handshake = independent.get("production_handshake")
    artifact_sha = independent.get("artifact_sha256")
    if (
        independent.get("schema_version") != 1
        or independent.get("artifact") != f"{_BASE_ARTIFACT}_independent_verification"
        or independent.get("status") != "passed"
        or independent.get("publication_top_manifest_sha256") != snapshots["SHA256SUMS"].sha256
        or independent.get("gate1_top_manifest_sha256") != snapshots["gate1/SHA256SUMS"].sha256
        or independent.get("config_sha256") != config.expected_base_config_sha256
        or not isinstance(checks, dict)
        or set(checks) != set(_INDEPENDENT_CHECKS)
        or any(value is not True for value in checks.values())
        or artifact_sha != semantic
        or not isinstance(handshake, dict)
        or handshake.get("bidirectional_acknowledgement") is not True
        or handshake.get("distinct_nodes") is not True
    ):
        raise ValueError("accepted independent verification receipt is invalid")
    for key in ("receipt_sha256", "acknowledgement_sha256"):
        values = handshake.get(key)
        if not isinstance(values, dict) or set(values) != {"0", "1"}:
            raise ValueError("accepted independent receipt handshake inventory is invalid")
        for value in values.values():
            _require_sha256(value, field=f"accepted handshake {key}")
    return manifest, receipt


def _verify_apex_manifest(
    snapshot: Snapshot,
    *,
    predictions_snapshot: Snapshot,
    reconciliation_snapshot: Snapshot,
    config: ApexUnionConfig,
) -> dict[str, object]:
    document = _parse_json_object(snapshot.payload, name="raw APEX manifest")
    endpoint_labels = [label for label, _ in _APEX_ENDPOINTS]
    endpoint_columns = [column for _, column in _APEX_ENDPOINTS]
    configured_columns = [endpoint for target in config.targets for endpoint in target.endpoints]
    configured_members = [asdict(item) for item in config.members]
    outputs = document.get("outputs")
    reconciliation = document.get("reconciliation")
    if (
        document.get("format_version") != 1
        or document.get("mode") != "apex_member_introspection"
        or document.get("source_commit") != config.expected_apex_source_commit
        or document.get("member_count") != config.expected_member_count
        or document.get("n_sequences") != config.expected_apex_sequences
        or document.get("endpoints") != endpoint_labels
        or configured_columns != endpoint_columns
        or document.get("members") != configured_members
        or not isinstance(outputs, dict)
        or set(outputs)
        != {
            "apex_member_predictions.csv",
            "apex_member_reconciliation.csv",
        }
        or outputs.get("apex_member_predictions.csv") != predictions_snapshot.sha256
        or outputs.get("apex_member_reconciliation.csv") != reconciliation_snapshot.sha256
        or not isinstance(reconciliation, dict)
        or reconciliation.get("all_passed") is not True
        or reconciliation.get("comparison_count") != config.expected_reconciliation_comparisons
    ):
        raise ValueError("raw APEX manifest contract is invalid")
    return document


def _read_apex_predictions(
    snapshot: Snapshot,
    *,
    manifest: Mapping[str, object],
    config: ApexUnionConfig,
) -> RawApexPredictions:
    rows = _strict_csv_rows(
        snapshot.payload, schema=_RAW_PREDICTION_SCHEMA, name="raw APEX member predictions"
    )
    members = tuple(item.member_id for item in config.members)
    member_hashes = {item.member_id: item.checkpoint_sha256 for item in config.members}
    endpoint_columns = tuple(column for _, column in _APEX_ENDPOINTS)
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    sequences: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        raw_sequence = row["sequence"]
        if canonicalize_sequence(raw_sequence) != raw_sequence:
            raise ValueError(f"raw APEX row {row_number} has a noncanonical sequence")
        sequence_id = row["sequence_id"]
        if sequence_id != canonical_sequence_id(raw_sequence):
            raise ValueError(f"raw APEX row {row_number} sequence ID mismatch")
        member_id = row["member_id"]
        if (
            row["model_family"] != "apex_pathogen_member"
            or row["model_version"] != config.expected_apex_source_commit
            or member_hashes.get(member_id) != row["member_checkpoint_sha256"]
            or member_id in values[sequence_id]
        ):
            raise ValueError(f"raw APEX row {row_number} has an invalid member identity")
        endpoint_values = {
            endpoint: _parse_float_text(row[endpoint], field=f"APEX row {row_number} {endpoint}")
            for endpoint in endpoint_columns
        }
        if any(value <= 0 for value in endpoint_values.values()):
            raise ValueError(f"raw APEX row {row_number} contains a non-positive MIC")
        broad_mean = _parse_float_text(
            row["apex_member_mean_mic_um"], field=f"APEX row {row_number} broad mean"
        )
        expected_broad_mean = math.fsum(endpoint_values.values()) / len(endpoint_values)
        if broad_mean <= 0 or broad_mean != expected_broad_mean:
            raise ValueError(f"raw APEX row {row_number} broad mean is inconsistent")
        previous_sequence = sequences.setdefault(sequence_id, raw_sequence)
        if previous_sequence != raw_sequence:
            raise ValueError(f"raw APEX row {row_number} has a sequence identity collision")
        values[sequence_id][member_id] = endpoint_values
    expected_members = set(members)
    if (
        len(values) != config.expected_apex_sequences
        or len(rows) != config.expected_apex_sequences * config.expected_member_count
        or manifest.get("n_sequences") != len(values)
        or any(set(items) != expected_members for items in values.values())
        or _sequence_set_sha256(values) != config.expected_apex_sequence_ids_sha256
    ):
        raise ValueError("raw APEX prediction grid is not the frozen rectangular inventory")
    return RawApexPredictions(
        members=members,
        member_hashes=member_hashes,
        values={sequence_id: dict(member_values) for sequence_id, member_values in values.items()},
        sequences=sequences,
        rows=len(rows),
    )


def _verify_apex_reconciliation(
    snapshot: Snapshot,
    *,
    manifest: Mapping[str, object],
    predictions: RawApexPredictions,
    config: ApexUnionConfig,
) -> None:
    rows = _strict_csv_rows(
        snapshot.payload, schema=_RECONCILIATION_SCHEMA, name="raw APEX reconciliation"
    )
    expected_endpoints = {label for label, _ in _APEX_ENDPOINTS} | {"broad_mean"}
    endpoint_columns = dict(_APEX_ENDPOINTS)
    summary = manifest.get("reconciliation")
    if not isinstance(summary, dict):
        raise ValueError("raw APEX manifest has no reconciliation summary")
    absolute_tolerance = _nonnegative_float(
        summary.get("absolute_tolerance"), field="APEX absolute tolerance"
    )
    relative_tolerance = _nonnegative_float(
        summary.get("relative_tolerance"), field="APEX relative tolerance"
    )
    seen: set[tuple[str, str]] = set()
    max_absolute_error = 0.0
    max_relative_error = 0.0
    for row_number, row in enumerate(rows, start=2):
        sequence_id = row["sequence_id"]
        sequence = row["sequence"]
        if (
            canonicalize_sequence(sequence) != sequence
            or canonical_sequence_id(sequence) != sequence_id
            or predictions.sequences.get(sequence_id) != sequence
        ):
            raise ValueError(f"APEX reconciliation row {row_number} sequence mismatch")
        endpoint = row["endpoint"]
        key = (sequence_id, endpoint)
        if endpoint not in expected_endpoints or key in seen:
            raise ValueError(f"APEX reconciliation row {row_number} endpoint mismatch")
        seen.add(key)
        member_count = _parse_int_text(
            row["member_count"], field=f"reconciliation row {row_number} member count", minimum=1
        )
        introspected = _parse_float_text(
            row["introspected_mean_mic_um"], field=f"reconciliation row {row_number} introspected"
        )
        fidelity = _parse_float_text(
            row["fidelity_mean_mic_um"], field=f"reconciliation row {row_number} fidelity"
        )
        absolute_error = _parse_float_text(
            row["absolute_error"], field=f"reconciliation row {row_number} absolute error"
        )
        relative_error = _parse_float_text(
            row["relative_error"], field=f"reconciliation row {row_number} relative error"
        )
        allowed_error = _parse_float_text(
            row["allowed_error"], field=f"reconciliation row {row_number} allowed error"
        )
        if endpoint == "broad_mean":
            member_values = [
                math.fsum(predictions.values[sequence_id][member].values()) / len(_APEX_ENDPOINTS)
                for member in predictions.members
            ]
        else:
            column = endpoint_columns[endpoint]
            member_values = [
                predictions.values[sequence_id][member][column] for member in predictions.members
            ]
        expected_introspected = math.fsum(member_values) / len(member_values)
        expected_absolute_error = abs(introspected - fidelity)
        expected_relative_error = expected_absolute_error / abs(fidelity)
        expected_allowed_error = absolute_tolerance + relative_tolerance * abs(fidelity)
        if (
            member_count != len(predictions.members)
            or introspected <= 0
            or fidelity <= 0
            or absolute_error < 0
            or relative_error < 0
            or allowed_error < 0
            or not math.isclose(introspected, expected_introspected, rel_tol=1e-15, abs_tol=1e-12)
            or absolute_error != expected_absolute_error
            or relative_error != expected_relative_error
            or allowed_error != expected_allowed_error
            or absolute_error > allowed_error
            or row["passed"] != "true"
        ):
            raise ValueError(f"APEX reconciliation row {row_number} did not reconcile")
        max_absolute_error = max(max_absolute_error, absolute_error)
        max_relative_error = max(max_relative_error, relative_error)
    expected_pairs = {
        (sequence_id, endpoint)
        for sequence_id in predictions.sequences
        for endpoint in expected_endpoints
    }
    if (
        seen != expected_pairs
        or len(rows) != config.expected_reconciliation_comparisons
        or summary.get("comparison_count") != len(rows)
        or summary.get("max_absolute_error") != max_absolute_error
        or summary.get("max_relative_error") != max_relative_error
        or summary.get("all_passed") is not True
    ):
        raise ValueError("APEX reconciliation coverage/summary is inconsistent")


def _build_activity_signals(
    examples: Sequence[Example],
    *,
    predictions: RawApexPredictions,
    config: ApexUnionConfig,
) -> tuple[SupportedExample, ...]:
    modeled_ids = {item.sequence_id for item in examples}
    raw_ids = set(predictions.sequences)
    missing = modeled_ids - raw_ids
    extra = raw_ids - modeled_ids
    if (
        missing
        or len(extra) != config.expected_apex_extra_sequences
        or len(modeled_ids & raw_ids) != config.expected_modeled_sequences
    ):
        raise ValueError("raw APEX sequences do not provide exact modeled-panel coverage")
    threshold_log = math.log10(config.activity_threshold_um)
    supported: list[SupportedExample] = []
    for item in examples:
        if predictions.sequences[item.sequence_id] != item.sequence:
            raise ValueError(f"APEX/base sequence bytes disagree for {item.sequence_id}")
        target = config.target_by_name.get(item.canonical_target)
        if target is None:
            raise ValueError(f"no exact APEX target map for {item.canonical_target!r}")
        signals: dict[str, float] = {}
        for member in predictions.members:
            endpoint_values = predictions.values[item.sequence_id][member]
            log_mean = math.fsum(
                math.log10(endpoint_values[name]) for name in target.endpoints
            ) / len(target.endpoints)
            signal = threshold_log - log_mean
            if not math.isfinite(signal):
                raise ValueError("APEX activity signal is non-finite")
            signals[member] = signal
        supported.append(SupportedExample(example=item, target=target, signals=signals))
    if len(supported) != config.expected_examples:
        raise ValueError("APEX signal construction lost a modeled context")
    return tuple(supported)


def _coverage_receipt(
    examples: Sequence[Example],
    *,
    predictions: RawApexPredictions,
    supported: Sequence[SupportedExample],
    config: ApexUnionConfig,
) -> dict[str, object]:
    modeled_ids = {item.sequence_id for item in examples}
    raw_ids = set(predictions.sequences)
    extra_ids = sorted(raw_ids - modeled_ids)
    fold_census = _fold_summary(examples, config.folds)
    target_by_fold = _target_summary(examples, config.folds)
    gram_target_census = {
        gram: {
            target: sum(item.gram == gram and item.canonical_target == target for item in examples)
            for target in sorted({item.canonical_target for item in examples})
        }
        for gram in ("negative", "positive")
    }
    return {
        "schema_version": 1,
        "artifact": "apex_union_sequence_coverage_receipt_v1",
        "status": "passed",
        "set_digest_encoding": _SET_DIGEST_ENCODING,
        "assignment_digest_encoding": _ASSIGNMENT_DIGEST_ENCODING,
        "base_modeled_panel": {
            "contexts": len(examples),
            "source_observations": sum(item.source_observations for item in examples),
            "positives": sum(item.label for item in examples),
            "negatives": sum(1 - item.label for item in examples),
            "sequences": len(modeled_ids),
            "homology_components": len({item.homology_component_id for item in examples}),
            "union_components": len({item.union_component_id for item in examples}),
            "sequence_ids_sha256": _sequence_set_sha256(modeled_ids),
            "assignment_sha256": _assignment_sha256(examples),
            "by_fold": fold_census,
            "canonical_targets_by_fold": target_by_fold,
            "gram_by_canonical_target": gram_target_census,
        },
        "raw_apex": {
            "rows": predictions.rows,
            "members": len(predictions.members),
            "sequences": len(raw_ids),
            "sequence_ids_sha256": _sequence_set_sha256(raw_ids),
        },
        "coverage": {
            "all_modeled_sequences_covered": modeled_ids <= raw_ids,
            "covered_sequences": len(modeled_ids & raw_ids),
            "covered_sequence_ids_sha256": _sequence_set_sha256(modeled_ids & raw_ids),
            "missing_sequences": 0,
            "missing_sequence_ids": [],
            "missing_sequence_ids_sha256": _sequence_set_sha256(set()),
            "extra_sequences": len(extra_ids),
            "extra_sequence_ids": extra_ids,
            "extra_sequence_ids_sha256": _sequence_set_sha256(extra_ids),
            "required_raw_member_joins": len(modeled_ids) * len(predictions.members),
        },
        "supported": {
            "contexts": len(supported),
            "sequences": len({item.example.sequence_id for item in supported}),
            "exact_canonical_target_mapping": True,
            "all_contexts_supported": len(supported) == len(examples),
            "calibration_weighting": config.calibration_weighting,
        },
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _fit_calibrator(
    rows: Sequence[SupportedExample],
    *,
    heldout_fold: int,
    excluded_folds: frozenset[int],
    member_id: str,
    config: ApexUnionConfig,
) -> Calibrator:
    if heldout_fold not in excluded_folds or not excluded_folds <= set(range(config.folds)):
        raise ValueError("calibrator excluded_folds must contain its held-out fold")
    checkpoint_by_member = {item.member_id: item.checkpoint_sha256 for item in config.members}
    if member_id not in checkpoint_by_member:
        raise ValueError("calibrator member is absent from the frozen member inventory")
    training = [item for item in rows if item.example.fold not in excluded_folds]
    if not training:
        raise ValueError("calibrator exclusion leaves no training contexts")
    labels = np.asarray([item.example.label for item in training], dtype=np.float64)
    signal = np.asarray([item.signals[member_id] for item in training], dtype=np.float64)
    if set(labels.tolist()) != {0.0, 1.0} or np.any(~np.isfinite(signal)):
        raise ValueError("calibration training complement must be finite and contain both classes")
    signal_mean = float(np.mean(signal))
    signal_scale = float(np.std(signal))
    if signal_scale < 1e-12:
        signal_scale = 1.0
    standardized = (signal - signal_mean) / signal_scale
    prior = float(
        (np.sum(labels) + 0.5 * config.calibration_prior_strength)
        / (len(labels) + config.calibration_prior_strength)
    )
    if not 0 < prior < 1:
        raise ValueError("calibration prior is degenerate")
    coefficient = np.asarray([math.log(prior / (1.0 - prior)), 0.0], dtype=np.float64)
    design = np.column_stack((np.ones(len(labels)), standardized))
    penalty = np.asarray([0.0, config.calibration_l2], dtype=np.float64)
    iterations = 0
    converged = False
    for iteration in range(1, config.calibration_max_iterations + 1):
        probabilities = np.asarray([_sigmoid(float(value)) for value in design @ coefficient])
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (probabilities - labels) / len(labels) + penalty * coefficient
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError("calibration produced a non-finite Newton step")
        coefficient -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficient)):
            raise ValueError("calibration produced non-finite coefficients")
        if float(np.max(np.abs(step))) <= config.calibration_tolerance:
            converged = True
            break
    if not converged:
        raise ValueError(f"fold {heldout_fold} member {member_id!r} calibration did not converge")
    training_examples = [item.example for item in training]
    return Calibrator(
        heldout_fold=heldout_fold,
        excluded_folds=tuple(sorted(excluded_folds)),
        training_folds=tuple(sorted({item.example.fold for item in training})),
        member_id=member_id,
        member_checkpoint_sha256=checkpoint_by_member[member_id],
        weighting_policy=config.calibration_weighting,
        training_examples=len(training),
        training_positives=int(np.sum(labels)),
        training_negatives=len(training) - int(np.sum(labels)),
        training_sequences=len({item.sequence_id for item in training_examples}),
        training_union_components=len({item.union_component_id for item in training_examples}),
        training_example_ids_sha256=_sequence_set_sha256(
            item.example_id for item in training_examples
        ),
        training_sequence_ids_sha256=_sequence_set_sha256(
            item.sequence_id for item in training_examples
        ),
        training_union_component_ids_sha256=_sequence_set_sha256(
            item.union_component_id for item in training_examples
        ),
        training_assignment_sha256=_assignment_sha256(training_examples),
        iterations=iterations,
        converged=True,
        signal_mean=signal_mean,
        signal_scale=signal_scale,
        intercept=float(coefficient[0]),
        slope=float(coefficient[1]),
        probability_clip_epsilon=config.probability_clip_epsilon,
    )


def _make_oof_predictions(
    supported: Sequence[SupportedExample],
    *,
    config: ApexUnionConfig,
) -> tuple[tuple[Prediction, ...], tuple[Calibrator, ...]]:
    calibrators: list[Calibrator] = []
    lookup: dict[tuple[int, str], Calibrator] = {}
    members = tuple(item.member_id for item in config.members)
    for fold in range(config.folds):
        for member in members:
            fitted = _fit_calibrator(
                supported,
                heldout_fold=fold,
                excluded_folds=frozenset({fold}),
                member_id=member,
                config=config,
            )
            calibrators.append(fitted)
            lookup[(fold, member)] = fitted
    rows: list[Prediction] = []
    for item in sorted(supported, key=lambda value: value.example.example_id):
        example = item.example
        probabilities: list[float] = []
        for member in members:
            probability = lookup[(example.fold, member)].predict(item.signals[member])
            probabilities.append(probability)
            rows.append(
                Prediction(
                    model=f"apex_member::{member}",
                    member_id=member,
                    **asdict(example),
                    apex_endpoints=";".join(item.target.endpoints),
                    activity_signal=item.signals[member],
                    probability=probability,
                    member_probability_std=None,
                )
            )
        rows.append(
            Prediction(
                model=_APEX_MEAN_MODEL,
                member_id="",
                **asdict(example),
                apex_endpoints=";".join(item.target.endpoints),
                activity_signal=math.fsum(item.signals.values()) / len(item.signals),
                probability=math.fsum(probabilities) / len(probabilities),
                member_probability_std=float(np.std(np.asarray(probabilities))),
            )
        )
    rows.sort(key=lambda item: (item.model, item.example_id))
    if (
        len(calibrators) != config.folds * config.expected_member_count
        or len(rows) != config.expected_output_rows
    ):
        raise ValueError("APEX union calibrator/prediction census is inconsistent")
    return tuple(rows), tuple(calibrators)


def _metric(rows: Sequence[Prediction], *, bins: int) -> dict[str, int | float | None]:
    if not rows:
        raise ValueError("metric subgroup must not be empty")
    return binary_metrics(
        [item.label for item in rows],
        [item.probability for item in rows],
        calibration_bins=bins,
    )


def _shared_union_component_draws(
    component_ids: Sequence[str], *, replicates: int, seed: int
) -> tuple[tuple[tuple[str, ...], ...], str]:
    components = tuple(sorted(component_ids))
    if not components or len(components) != len(set(components)):
        raise ValueError("bootstrap components must be non-empty and unique")
    generator = np.random.default_rng(seed)
    digest = hashlib.sha256()
    draws: list[tuple[str, ...]] = []
    for replicate in range(replicates):
        selected = tuple(
            str(item) for item in generator.choice(components, size=len(components), replace=True)
        )
        draws.append(selected)
        for position, component_id in enumerate(selected):
            digest.update(f"{replicate}\t{position}\t{component_id}\n".encode("ascii"))
    return tuple(draws), digest.hexdigest()


def _bootstrap_intervals(
    rows: Sequence[Prediction],
    *,
    draws: Sequence[Sequence[str]],
    bins: int,
) -> dict[str, dict[str, float | int]]:
    by_component: dict[str, list[Prediction]] = defaultdict(list)
    for item in rows:
        by_component[item.union_component_id].append(item)
    expected_components = set(by_component)
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for draw in draws:
        if set(draw) - expected_components:
            raise ValueError("shared bootstrap draw references an unknown union component")
        selected = [row for component_id in draw for row in by_component[component_id]]
        values = _metric(selected, bins=bins)
        for name in metric_names:
            value = values[name]
            if value is None:
                raise ValueError(f"union-component bootstrap produced undefined {name}")
            samples[name].append(float(value))
    point = _metric(rows, bins=bins)
    output: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        if len(samples[name]) != len(draws):
            raise ValueError("union-component bootstrap lost a deterministic replicate")
        values = np.asarray(samples[name], dtype=np.float64)
        point_value = point[name]
        if point_value is None:
            raise ValueError(f"overall {name} is undefined")
        output[name] = {
            "point": float(point_value),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "successful_replicates": len(samples[name]),
        }
    return output


def _summarize_metrics(
    predictions: Sequence[Prediction], *, config: ApexUnionConfig
) -> dict[str, object]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped[item.model].append(item)
    member_models = tuple(f"apex_member::{item.member_id}" for item in config.members)
    expected_models = set(member_models) | {_APEX_MEAN_MODEL}
    if set(grouped) != expected_models:
        raise ValueError("APEX union predictions do not contain the frozen model inventory")
    expected_example_ids = {item.example_id for item in grouped[_APEX_MEAN_MODEL]}
    if len(expected_example_ids) != config.expected_examples:
        raise ValueError("APEX mean predictions do not cover the modeled panel exactly once")
    for model, rows in grouped.items():
        if (
            len(rows) != config.expected_examples
            or {item.example_id for item in rows} != expected_example_ids
        ):
            raise ValueError(f"model {model!r} does not have exact context support")

    component_ids = sorted({item.union_component_id for item in grouped[_APEX_MEAN_MODEL]})
    if len(component_ids) != config.expected_modeled_union_components:
        raise ValueError("bootstrap support does not contain 213 modeled union components")
    draws, draw_sha256 = _shared_union_component_draws(
        component_ids,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    model_metrics: dict[str, object] = {}
    for model in sorted(grouped):
        rows = tuple(sorted(grouped[model], key=lambda item: item.example_id))
        by_identity: dict[str, object] = {}
        for left, right in _SIMILARITY_STRATA:
            selected = [item for item in rows if left <= item.max_train_identity < right]
            if selected:
                by_identity[f"[{left:.2f},{right:.2f})"] = _metric(
                    selected, bins=config.calibration_bins
                )
        model_metrics[model] = {
            "overall": _metric(rows, bins=config.calibration_bins),
            "by_fold": {
                str(fold): _metric(
                    [item for item in rows if item.fold == fold],
                    bins=config.calibration_bins,
                )
                for fold in range(config.folds)
            },
            "by_gram": {
                gram: _metric(
                    [item for item in rows if item.gram == gram],
                    bins=config.calibration_bins,
                )
                for gram in ("negative", "positive")
            },
            "by_canonical_target": {
                target: _metric(
                    [item for item in rows if item.canonical_target == target],
                    bins=config.calibration_bins,
                )
                for target in sorted({item.canonical_target for item in rows})
            },
            "by_max_train_identity": by_identity,
            "union_component_bootstrap_95ci": _bootstrap_intervals(
                rows,
                draws=draws,
                bins=config.calibration_bins,
            ),
        }

    aligned = {
        model: {item.example_id: item.probability for item in grouped[model]}
        for model in member_models
    }
    example_ids = sorted(expected_example_ids)
    matrix = np.asarray(
        [[aligned[model][example_id] for example_id in example_ids] for model in member_models],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.corrcoef(matrix)
    if correlations.shape != (len(member_models), len(member_models)) or not np.all(
        np.isfinite(correlations)
    ):
        raise ValueError("APEX member probability correlations are undefined")
    off_diagonal = correlations[np.triu_indices(len(member_models), 1)]
    return {
        "schema_version": 1,
        "artifact": "apex_union_oof_metrics_v1",
        "models": model_metrics,
        "bootstrap": {
            "unit": _BOOTSTRAP_UNIT,
            "weighting": config.bootstrap_weighting,
            "component_ids": len(component_ids),
            "component_ids_sha256": _sequence_set_sha256(component_ids),
            "replicates": config.bootstrap_replicates,
            "seed": config.bootstrap_seed,
            "draw_encoding": _BOOTSTRAP_DRAW_ENCODING,
            "draws_sha256": draw_sha256,
            "shared_across_models": True,
        },
        "model_diversity": {
            "members": list(member_models),
            "pairwise_probability_correlation_minimum": float(np.min(off_diagonal)),
            "pairwise_probability_correlation_median": float(np.median(off_diagonal)),
            "pairwise_probability_correlation_maximum": float(np.max(off_diagonal)),
        },
    }


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


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


_OUTPUT_PREDICTION_SCHEMA = (
    "model",
    "member_id",
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
    "apex_endpoints",
    "activity_signal",
    "probability",
    "member_probability_std",
)


def _write_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(_OUTPUT_PREDICTION_SCHEMA), lineterminator="\n"
        )
        writer.writeheader()
        for item in predictions:
            row = asdict(item)
            for field in ("max_train_identity", "activity_signal", "probability"):
                row[field] = format(cast(float, row[field]), ".17g")
            row["member_probability_std"] = (
                ""
                if item.member_probability_std is None
                else format(item.member_probability_std, ".17g")
            )
            writer.writerow(row)


def _write_checksum_manifest(path: Path, entries: Mapping[str, str]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for filename in sorted(entries):
            handle.write(f"{entries[filename]}  {filename}\n")


def run_apex_union_oof(
    *,
    base_run: str | Path,
    base_independent_receipt: str | Path,
    apex_root: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    frozen_input_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> ApexUnionExecution:
    """Fit strict fold-local APEX calibrators on the accepted union panel."""

    if _GIT_SHA1.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character Git SHA")
    requested_output = Path(output_dir).absolute()
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse APEX union output: {requested_output}")
    parent = requested_output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("APEX union output parent must be an existing non-symbolic directory")
    output = requested_output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse APEX union output: {output}")

    _, base_snapshots = _snapshot_exact_tree(
        base_run, expected_files=_BASE_TREE_FILES, name="accepted Gate-1 publication"
    )
    _, apex_snapshots = _snapshot_exact_tree(
        apex_root, expected_files=_APEX_TREE_FILES, name="raw APEX publication"
    )
    independent_snapshot = _read_snapshot(
        base_independent_receipt, name="accepted Gate-1 independent receipt"
    )
    config_snapshot = _read_snapshot(config_path, name="APEX union config")
    code_manifest_snapshot = _read_snapshot(code_manifest_path, name="code manifest")
    frozen_manifest_snapshot = _read_snapshot(
        frozen_input_manifest_path, name="frozen input manifest"
    )
    config = _config_from_payload(config_snapshot.path, config_snapshot.payload)
    code_attestation, repository_snapshots = _validate_code_manifest(
        code_manifest_snapshot, config_snapshot=config_snapshot
    )
    frozen_snapshots = {
        **{f"base/{logical}": snapshot for logical, snapshot in base_snapshots.items()},
        "base-independent-receipt.json": independent_snapshot,
        **{f"apex/{logical}": snapshot for logical, snapshot in apex_snapshots.items()},
    }
    _verify_frozen_input_manifest(frozen_manifest_snapshot, snapshots=frozen_snapshots)

    for snapshot, expected, name in (
        (
            base_snapshots["SHA256SUMS"],
            config.expected_base_publication_top_sha256,
            "accepted publication top",
        ),
        (
            base_snapshots["gate1/SHA256SUMS"],
            config.expected_base_semantic_top_sha256,
            "accepted semantic top",
        ),
        (
            base_snapshots["gate1/examples.jsonl"],
            config.expected_base_examples_sha256,
            "accepted examples",
        ),
        (
            base_snapshots["gate1/folds.json"],
            config.expected_base_folds_sha256,
            "accepted folds",
        ),
        (
            base_snapshots["gate1/oof_predictions.csv"],
            config.expected_base_oof_sha256,
            "accepted OOF metadata",
        ),
        (
            base_snapshots["gate1/manifest.json"],
            config.expected_base_manifest_sha256,
            "accepted manifest",
        ),
        (
            base_snapshots["gate1/split_receipt.json"],
            config.expected_base_split_receipt_sha256,
            "accepted split receipt",
        ),
        (
            independent_snapshot,
            config.expected_base_independent_receipt_sha256,
            "accepted independent receipt",
        ),
        (
            apex_snapshots["apex_member_predictions.csv"],
            config.expected_apex_predictions_sha256,
            "raw APEX predictions",
        ),
        (
            apex_snapshots["apex_member_run_manifest.json"],
            config.expected_apex_manifest_sha256,
            "raw APEX manifest",
        ),
        (
            apex_snapshots["apex_member_reconciliation.csv"],
            config.expected_apex_reconciliation_sha256,
            "raw APEX reconciliation",
        ),
    ):
        _expect_hash(snapshot, expected, name=name)

    examples = _read_examples(
        base_snapshots["gate1/examples.jsonl"],
        base_snapshots["gate1/oof_predictions.csv"],
        config=config,
    )
    folds_document = _verify_folds(
        base_snapshots["gate1/folds.json"], examples=examples, config=config
    )
    base_manifest, split_receipt = _verify_base_documents(
        base_snapshots,
        independent_snapshot,
        examples=examples,
        folds_document=folds_document,
        config=config,
    )
    apex_manifest = _verify_apex_manifest(
        apex_snapshots["apex_member_run_manifest.json"],
        predictions_snapshot=apex_snapshots["apex_member_predictions.csv"],
        reconciliation_snapshot=apex_snapshots["apex_member_reconciliation.csv"],
        config=config,
    )
    raw_predictions = _read_apex_predictions(
        apex_snapshots["apex_member_predictions.csv"],
        manifest=apex_manifest,
        config=config,
    )
    _verify_apex_reconciliation(
        apex_snapshots["apex_member_reconciliation.csv"],
        manifest=apex_manifest,
        predictions=raw_predictions,
        config=config,
    )
    supported = _build_activity_signals(examples, predictions=raw_predictions, config=config)
    coverage = _coverage_receipt(
        examples,
        predictions=raw_predictions,
        supported=supported,
        config=config,
    )
    prediction_rows, calibrators = _make_oof_predictions(supported, config=config)
    metrics = _summarize_metrics(prediction_rows, config=config)

    calibrator_document = {
        "schema_version": 1,
        "artifact": "apex_union_outer_fold_calibrators_v1",
        "status": "passed",
        "policy": _CALIBRATION_POLICY,
        "weighting": config.calibration_weighting,
        "activity_signal": (
            "log10(activity_threshold_um) minus arithmetic mean of target-endpoint log10 MIC"
        ),
        "census": {
            "outer_folds": config.folds,
            "members": config.expected_member_count,
            "calibrators": len(calibrators),
        },
        "calibrators": [asdict(item) for item in calibrators],
    }
    artifacts_for_manifest: dict[str, dict[str, str]] = {}
    staging = Path(tempfile.mkdtemp(prefix=f".{requested_output.name}.staging-", dir=str(parent)))
    published = False
    try:
        predictions_path = staging / "apex_union_oof_predictions.csv"
        calibrators_path = staging / "calibrators.json"
        coverage_path = staging / "sequence_coverage_receipt.json"
        metrics_path = staging / "metrics.json"
        manifest_path = staging / "manifest.json"
        top_path = staging / "SHA256SUMS"
        _write_predictions(predictions_path, prediction_rows)
        _write_json(calibrators_path, calibrator_document)
        _write_json(coverage_path, coverage)
        _write_json(metrics_path, metrics)
        for key, filename, role in (
            (
                "predictions",
                predictions_path.name,
                "nine-model context-level outer OOF predictions",
            ),
            ("calibrators", calibrators_path.name, "forty outer-fold member calibrators"),
            ("coverage", coverage_path.name, "strict sequence/context reconciliation receipt"),
            ("metrics", metrics_path.name, "context metrics and shared union-component bootstrap"),
        ):
            artifacts_for_manifest[key] = {
                "filename": filename,
                "role": role,
                "sha256": _file_sha256(staging / filename),
            }
        accepted_split = base_manifest.get("accepted_split")
        manifest_document = {
            "schema_version": 1,
            "artifact": _ARTIFACT,
            "status": _OUTPUT_STATUS,
            "git_commit": git_commit,
            "config_sha256": config_snapshot.sha256,
            "production_eligible": False,
            "production_eligibility_reason": (
                "upstream APEX training membership is unknown; calibrated development evidence "
                "must not be represented as an untouched evaluation"
            ),
            "upstream_training_independence": {
                "status": "unknown",
                "external_family_weight_policy": "zero_until_training_membership_is_resolved",
            },
            "assignment_policy": _ASSIGNMENT_POLICY,
            "calibration": {
                "policy": _CALIBRATION_POLICY,
                "weighting": config.calibration_weighting,
                "outer_calibrators": len(calibrators),
                "probability_clip_epsilon": config.probability_clip_epsilon,
                "label_feature_used_only_on_outer_training_complement": True,
                "forbidden_features": list(_FORBIDDEN_FEATURES),
                "model_features": list(_MODEL_FEATURES),
            },
            "bootstrap": cast(dict[str, object], metrics["bootstrap"]),
            "census": {
                "contexts": len(examples),
                "source_observations": sum(item.source_observations for item in examples),
                "positives": sum(item.label for item in examples),
                "negatives": sum(1 - item.label for item in examples),
                "modeled_sequences": len({item.sequence_id for item in examples}),
                "modeled_homology_components": len(
                    {item.homology_component_id for item in examples}
                ),
                "modeled_union_components": len({item.union_component_id for item in examples}),
                "prediction_rows": len(prediction_rows),
            },
            "models": [
                *(f"apex_member::{item.member_id}" for item in config.members),
                _APEX_MEAN_MODEL,
            ],
            "members": [asdict(item) for item in config.members],
            "accepted_gate1": {
                "artifact": _BASE_ARTIFACT,
                "publication_top_sha256": base_snapshots["SHA256SUMS"].sha256,
                "semantic_top_sha256": base_snapshots["gate1/SHA256SUMS"].sha256,
                "independent_receipt_sha256": independent_snapshot.sha256,
                "accepted_split": accepted_split,
                "split_consumption_receipt_sha256": base_snapshots[
                    "gate1/split_receipt.json"
                ].sha256,
                "all_split_census": split_receipt.get("census"),
            },
            "input_sha256": {
                "code_manifest": code_manifest_snapshot.sha256,
                "frozen_input_manifest": frozen_manifest_snapshot.sha256,
                "config": config_snapshot.sha256,
                "base_publication_top": base_snapshots["SHA256SUMS"].sha256,
                "base_semantic_top": base_snapshots["gate1/SHA256SUMS"].sha256,
                "base_examples": base_snapshots["gate1/examples.jsonl"].sha256,
                "base_folds": base_snapshots["gate1/folds.json"].sha256,
                "base_oof_metadata": base_snapshots["gate1/oof_predictions.csv"].sha256,
                "base_manifest": base_snapshots["gate1/manifest.json"].sha256,
                "base_split_receipt": base_snapshots["gate1/split_receipt.json"].sha256,
                "base_independent_receipt": independent_snapshot.sha256,
                "apex_predictions": apex_snapshots["apex_member_predictions.csv"].sha256,
                "apex_manifest": apex_snapshots["apex_member_run_manifest.json"].sha256,
                "apex_reconciliation": apex_snapshots["apex_member_reconciliation.csv"].sha256,
            },
            "code_attestation": code_attestation,
            "artifacts": artifacts_for_manifest,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        _write_json(manifest_path, manifest_document)
        top_entries = {
            filename: _file_sha256(staging / filename)
            for filename in (
                "apex_union_oof_predictions.csv",
                "calibrators.json",
                "manifest.json",
                "metrics.json",
                "sequence_coverage_receipt.json",
            )
        }
        _write_checksum_manifest(top_path, top_entries)
        if {item.name for item in staging.iterdir()} != {
            "SHA256SUMS",
            *top_entries,
        }:
            raise ValueError("staging output inventory is not the six-file contract")
        all_input_snapshots = (
            *base_snapshots.values(),
            *apex_snapshots.values(),
            independent_snapshot,
            config_snapshot,
            code_manifest_snapshot,
            frozen_manifest_snapshot,
            *repository_snapshots,
        )
        for index, snapshot in enumerate(all_input_snapshots):
            _assert_snapshot_unchanged(snapshot, name=f"input snapshot {index}")
        os.rename(staging, output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    return ApexUnionExecution(
        output_dir=output,
        predictions_path=output / "apex_union_oof_predictions.csv",
        calibrators_path=output / "calibrators.json",
        coverage_path=output / "sequence_coverage_receipt.json",
        metrics_path=output / "metrics.json",
        manifest_path=output / "manifest.json",
        top_manifest_path=output / "SHA256SUMS",
        examples=len(examples),
        prediction_rows=len(prediction_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate immutable APEX members on accepted union folds."
    )
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--base-independent-receipt", required=True, type=Path)
    parser.add_argument("--apex-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--frozen-input-manifest", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_apex_union_oof(
        base_run=args.base_run,
        base_independent_receipt=args.base_independent_receipt,
        apex_root=args.apex_root,
        config_path=args.config,
        code_manifest_path=args.code_manifest,
        frozen_input_manifest_path=args.frozen_input_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "examples": result.examples,
                "output_dir": str(result.output_dir),
                "prediction_rows": result.prediction_rows,
                "top_manifest_sha256": _file_sha256(result.top_manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

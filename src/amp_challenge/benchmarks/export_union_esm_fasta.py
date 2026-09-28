"""Export the exact accepted union Gate-1 sequence panel for ESM2 extraction.

The exporter is intentionally separate from the historical Gate-1 ESM path.
It validates the accepted context-level Gate-1 publication and independent
receipt, proves that all three stored OOF models carry identical metadata, and
then exports exactly the unique modeled sequences.  Stored model probabilities
are never consumed when constructing the panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MIN_LENGTH = 8
_MAX_LENGTH = 50

_BASE_ARTIFACT = "gate1_context_activity_homology_study_union_v1"
_BASE_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_BASE_FOLD_ARTIFACT = "gate1_context_union_fold_reuse"
_BASE_RECEIPT_ARTIFACT = "gate1_union_accepted_split_consumption_receipt"
_BASE_AUDIT_ARTIFACT = "gate1_context_activity_homology_study_union_v1_independent_verification"
_BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_FOLD_POLICY = "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
_LABEL_POLICY = (
    "group every ledger row by assay_context_id before eligibility filtering; retain only "
    "fully bacterial_mic16-eligible contexts with unanimous label, source Gram, and "
    "canonical target"
)
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
_SPLIT_RECEIPT_INVARIANTS = frozenset(
    {
        "accepted_independent_verification_passed",
        "accepted_sequence_assignments_reused_exactly",
        "accepted_split_hash_chain_valid",
        "all_contexts_grouped_before_eligibility_filtering",
        "component_assignment_join_exact",
        "cross_fold_identity_strictly_below_threshold",
        "fold_label_census_exact",
        "model_feature_allowlist_exact",
    }
)
_OUTPUT_ARTIFACT = "gate1_union_esm2_exact_panel_fasta_v1"
_COVERAGE_ARTIFACT = "gate1_union_esm2_exact_panel_coverage_v1"

_LOGICAL_CONFIG_PATH = "configs/models/esm2_union_v1.toml"
_EXECUTING_MODULE_PATH = "src/amp_challenge/benchmarks/export_union_esm_fasta.py"
_FIXED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/export_union_esm_fasta_v1_twins.sbatch",
        "cluster/validate_union_esm_fasta_output.sh",
        _LOGICAL_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)
_PUBLICATION_ENTRIES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "gate1/SHA256SUMS",
        "gate1/context_audit.jsonl",
        "gate1/examples.jsonl",
        "gate1/folds.json",
        "gate1/manifest.json",
        "gate1/metrics.json",
        "gate1/oof_predictions.csv",
        "gate1/split_receipt.json",
    }
)
_GATE1_ENTRIES = frozenset(
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
_EXAMPLE_FIELDS = frozenset(
    {
        "schema_version",
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
    }
)
_OOF_FIELDS = (
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
_FOLD_ASSIGNMENT_FIELDS = frozenset(
    {
        "example_id",
        "sequence_id",
        "homology_component_id",
        "union_component_id",
        "fold",
    }
)
_CLASS_METRICS = (
    "gram_negative_mic16_negative",
    "gram_negative_mic16_positive",
    "gram_positive_mic16_negative",
    "gram_positive_mic16_positive",
)
_INDEPENDENT_CHECKS = frozenset(
    {
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
        "accepted_split_receipt_chain_valid",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "gate1_publication_top_sha256",
        "gate1_top_sha256",
        "gate1_examples_sha256",
        "gate1_folds_sha256",
        "gate1_oof_sha256",
        "gate1_manifest_sha256",
        "gate1_split_receipt_sha256",
        "gate1_independent_receipt_sha256",
        "gate1_config_sha256",
        "gate1_code_manifest_sha256",
        "gate1_frozen_input_manifest_sha256",
        "gate1_git_commit",
        "expected_examples",
        "expected_source_observations",
        "expected_sequences",
        "expected_positive_examples",
        "expected_negative_examples",
        "expected_examples_by_fold",
        "expected_positive_examples_by_fold",
        "expected_negative_examples_by_fold",
        "folds",
        "homology_identity_threshold",
        "embedding_model",
        "representation_layer",
        "embedding_dimension",
        "embedding_batch_size",
        "embedding_seed",
        "model_checkpoint_sha256",
        "contact_regression_sha256",
        "environment_lock_sha256",
        "trust_manifest_sha256",
        "embedding_worker_sha256",
        "python_version",
        "torch_version",
        "fair_esm_version",
        "numpy_version",
        "cuda_runtime",
        "cudnn_version",
        "device_type",
        "historical_design_hint_status",
        "historical_design_hint_input_fasta_sha256",
        "historical_design_hint_input_fasta_manifest_sha256",
        "historical_design_hint_embedding_records",
        "historical_design_hint_extra_embedding_sequence_ids",
        "historical_design_hint_extra_embedding_sequence_ids_sha256",
        "historical_design_hint_missing_union_sequence_ids",
        "historical_design_hint_missing_union_sequence_ids_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class UnionEsmConfig:
    path: Path
    sha256: str
    gate1_publication_top_sha256: str
    gate1_top_sha256: str
    gate1_examples_sha256: str
    gate1_folds_sha256: str
    gate1_oof_sha256: str
    gate1_manifest_sha256: str
    gate1_split_receipt_sha256: str
    gate1_independent_receipt_sha256: str
    gate1_config_sha256: str
    gate1_code_manifest_sha256: str
    gate1_frozen_input_manifest_sha256: str
    gate1_git_commit: str
    expected_examples: int
    expected_source_observations: int
    expected_sequences: int
    expected_positive_examples: int
    expected_negative_examples: int
    expected_examples_by_fold: tuple[int, ...]
    expected_positive_examples_by_fold: tuple[int, ...]
    expected_negative_examples_by_fold: tuple[int, ...]
    folds: int
    homology_identity_threshold: float
    embedding_model: str
    representation_layer: int
    embedding_dimension: int
    embedding_batch_size: int
    embedding_seed: int
    model_checkpoint_sha256: str
    contact_regression_sha256: str
    environment_lock_sha256: str
    trust_manifest_sha256: str
    embedding_worker_sha256: str
    python_version: str
    torch_version: str
    fair_esm_version: str
    numpy_version: str
    cuda_runtime: str
    cudnn_version: int
    device_type: str
    historical_design_hint_status: str
    historical_design_hint_input_fasta_sha256: str
    historical_design_hint_input_fasta_manifest_sha256: str
    historical_design_hint_embedding_records: int
    historical_design_hint_extra_embedding_sequence_ids: tuple[str, ...]
    historical_design_hint_extra_embedding_sequence_ids_sha256: str
    historical_design_hint_missing_union_sequence_ids: tuple[str, ...]
    historical_design_hint_missing_union_sequence_ids_sha256: str


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    canonical_target: str
    gram: str
    label: int
    source_observations: int
    fold: int
    homology_component_id: str
    union_component_id: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_snapshot(path: str | Path, *, label: str) -> Snapshot:
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    source = requested.resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    payload = source.read_bytes()
    after = source.stat()
    if _fingerprint(before) != _fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while it was read")
    return Snapshot(source, payload, _sha256(payload), _fingerprint(after))


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    try:
        current = snapshot.path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} disappeared during export") from error
    if not stat.S_ISREG(current.st_mode) or _fingerprint(current) != snapshot.fingerprint:
        raise ValueError(f"{label} changed during export")


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_object(snapshot: Snapshot, *, label: str) -> dict[str, object]:
    payload = snapshot.payload
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must have exactly one final LF and no CR")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _jsonl(snapshot: Snapshot, *, label: str) -> list[dict[str, object]]:
    payload = snapshot.payload
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-delimited JSONL")
    rows: list[dict[str, object]] = []
    for number, raw in enumerate(payload[:-1].split(b"\n"), start=1):
        if not raw:
            raise ValueError(f"{label} row {number} is blank")
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{label} row {number} is not strict JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {number} must be an object")
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if raw != canonical:
            raise ValueError(f"{label} row {number} is not canonical JSONL")
        rows.append(cast(dict[str, object], value))
    return rows


def _manifest(snapshot: Snapshot, *, label: str) -> dict[str, str]:
    payload = snapshot.payload
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-delimited text")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None or match.group(2) != " ":
            raise ValueError(f"{label} row {number} is not a text-mode SHA-256 entry")
        digest, _, filename = match.groups()
        pure = PurePosixPath(filename)
        if pure.is_absolute() or ".." in pure.parts or "\\" in filename:
            raise ValueError(f"{label} row {number} has an unsafe path")
        if filename in result or (previous is not None and filename <= previous):
            raise ValueError(f"{label} paths must be unique and strictly sorted")
        result[filename] = digest
        previous = filename
    return result


def _exact_fields(value: Mapping[str, object], expected: Iterable[str], *, label: str) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(required - observed)}, "
            f"extra={sorted(observed - required)}"
        )


def _sha_field(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _count_vector(value: object, *, folds: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != folds
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(f"{label} must be a nonnegative integer vector of length {folds}")
    return tuple(cast(list[int], value))


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _sequence_id_pair(value: object, *, label: str) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{label} must be two sorted unique SHA-256 IDs")
    return cast(tuple[str, str], tuple(value))


def _load_config(snapshot: Snapshot) -> UnionEsmConfig:
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("union ESM config is not valid UTF-8 TOML") from error
    _exact_fields(raw, _CONFIG_FIELDS, label="union ESM config")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("union ESM config schema_version must be 1")
    folds = _positive_int(raw["folds"], label="folds")
    examples = _positive_int(raw["expected_examples"], label="expected_examples")
    positives = _positive_int(raw["expected_positive_examples"], label="expected_positive_examples")
    negatives = _positive_int(raw["expected_negative_examples"], label="expected_negative_examples")
    by_fold = _count_vector(raw["expected_examples_by_fold"], folds=folds, label="examples")
    positive_by_fold = _count_vector(
        raw["expected_positive_examples_by_fold"], folds=folds, label="positives"
    )
    negative_by_fold = _count_vector(
        raw["expected_negative_examples_by_fold"], folds=folds, label="negatives"
    )
    if examples != positives + negatives or sum(by_fold) != examples:
        raise ValueError("union ESM config example census is inconsistent")
    if sum(positive_by_fold) != positives or sum(negative_by_fold) != negatives:
        raise ValueError("union ESM config per-fold label census is inconsistent")
    if any(
        total != positive + negative
        for total, positive, negative in zip(
            by_fold, positive_by_fold, negative_by_fold, strict=True
        )
    ):
        raise ValueError("union ESM config fold totals are inconsistent")
    threshold = raw["homology_identity_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or float(threshold) != 0.8
    ):
        raise ValueError("union ESM homology_identity_threshold must be 0.8")
    git_commit = _string(raw["gate1_git_commit"], label="gate1_git_commit")
    if _GIT_RE.fullmatch(git_commit) is None:
        raise ValueError("gate1_git_commit must be a full lowercase Git SHA")
    design_hint_status = _string(
        raw["historical_design_hint_status"], label="historical_design_hint_status"
    )
    if design_hint_status != "unverified_historical_design_hint_not_evidence":
        raise ValueError("historical design hint must be explicitly marked unverified")
    historical_extra_ids = _sequence_id_pair(
        raw["historical_design_hint_extra_embedding_sequence_ids"],
        label="historical_design_hint_extra_embedding_sequence_ids",
    )
    if (
        _set_digest(historical_extra_ids)
        != raw["historical_design_hint_extra_embedding_sequence_ids_sha256"]
    ):
        raise ValueError("historical design-hint extra-embedding-ID digest mismatch")
    historical_missing_ids = _sequence_id_pair(
        raw["historical_design_hint_missing_union_sequence_ids"],
        label="historical_design_hint_missing_union_sequence_ids",
    )
    if (
        _set_digest(historical_missing_ids)
        != raw["historical_design_hint_missing_union_sequence_ids_sha256"]
    ):
        raise ValueError("historical design-hint missing-union-ID digest mismatch")
    hashes = {
        name: _sha_field(raw[name], label=name)
        for name in (
            "gate1_publication_top_sha256",
            "gate1_top_sha256",
            "gate1_examples_sha256",
            "gate1_folds_sha256",
            "gate1_oof_sha256",
            "gate1_manifest_sha256",
            "gate1_split_receipt_sha256",
            "gate1_independent_receipt_sha256",
            "gate1_config_sha256",
            "gate1_code_manifest_sha256",
            "gate1_frozen_input_manifest_sha256",
            "model_checkpoint_sha256",
            "contact_regression_sha256",
            "environment_lock_sha256",
            "trust_manifest_sha256",
            "embedding_worker_sha256",
            "historical_design_hint_input_fasta_sha256",
            "historical_design_hint_input_fasta_manifest_sha256",
            "historical_design_hint_extra_embedding_sequence_ids_sha256",
            "historical_design_hint_missing_union_sequence_ids_sha256",
        )
    }
    return UnionEsmConfig(
        path=snapshot.path,
        sha256=snapshot.sha256,
        gate1_publication_top_sha256=hashes["gate1_publication_top_sha256"],
        gate1_top_sha256=hashes["gate1_top_sha256"],
        gate1_examples_sha256=hashes["gate1_examples_sha256"],
        gate1_folds_sha256=hashes["gate1_folds_sha256"],
        gate1_oof_sha256=hashes["gate1_oof_sha256"],
        gate1_manifest_sha256=hashes["gate1_manifest_sha256"],
        gate1_split_receipt_sha256=hashes["gate1_split_receipt_sha256"],
        gate1_independent_receipt_sha256=hashes["gate1_independent_receipt_sha256"],
        gate1_config_sha256=hashes["gate1_config_sha256"],
        gate1_code_manifest_sha256=hashes["gate1_code_manifest_sha256"],
        gate1_frozen_input_manifest_sha256=hashes["gate1_frozen_input_manifest_sha256"],
        gate1_git_commit=git_commit,
        expected_examples=examples,
        expected_source_observations=_positive_int(
            raw["expected_source_observations"], label="expected_source_observations"
        ),
        expected_sequences=_positive_int(raw["expected_sequences"], label="expected_sequences"),
        expected_positive_examples=positives,
        expected_negative_examples=negatives,
        expected_examples_by_fold=by_fold,
        expected_positive_examples_by_fold=positive_by_fold,
        expected_negative_examples_by_fold=negative_by_fold,
        folds=folds,
        homology_identity_threshold=float(threshold),
        embedding_model=_string(raw["embedding_model"], label="embedding_model"),
        representation_layer=_positive_int(
            raw["representation_layer"], label="representation_layer"
        ),
        embedding_dimension=_positive_int(raw["embedding_dimension"], label="embedding_dimension"),
        embedding_batch_size=_positive_int(
            raw["embedding_batch_size"], label="embedding_batch_size"
        ),
        embedding_seed=_positive_int(raw["embedding_seed"], label="embedding_seed"),
        model_checkpoint_sha256=hashes["model_checkpoint_sha256"],
        contact_regression_sha256=hashes["contact_regression_sha256"],
        environment_lock_sha256=hashes["environment_lock_sha256"],
        trust_manifest_sha256=hashes["trust_manifest_sha256"],
        embedding_worker_sha256=hashes["embedding_worker_sha256"],
        python_version=_string(raw["python_version"], label="python_version"),
        torch_version=_string(raw["torch_version"], label="torch_version"),
        fair_esm_version=_string(raw["fair_esm_version"], label="fair_esm_version"),
        numpy_version=_string(raw["numpy_version"], label="numpy_version"),
        cuda_runtime=_string(raw["cuda_runtime"], label="cuda_runtime"),
        cudnn_version=_positive_int(raw["cudnn_version"], label="cudnn_version"),
        device_type=_string(raw["device_type"], label="device_type"),
        historical_design_hint_status=design_hint_status,
        historical_design_hint_input_fasta_sha256=hashes[
            "historical_design_hint_input_fasta_sha256"
        ],
        historical_design_hint_input_fasta_manifest_sha256=hashes[
            "historical_design_hint_input_fasta_manifest_sha256"
        ],
        historical_design_hint_embedding_records=_positive_int(
            raw["historical_design_hint_embedding_records"],
            label="historical_design_hint_embedding_records",
        ),
        historical_design_hint_extra_embedding_sequence_ids=historical_extra_ids,
        historical_design_hint_extra_embedding_sequence_ids_sha256=hashes[
            "historical_design_hint_extra_embedding_sequence_ids_sha256"
        ],
        historical_design_hint_missing_union_sequence_ids=historical_missing_ids,
        historical_design_hint_missing_union_sequence_ids_sha256=hashes[
            "historical_design_hint_missing_union_sequence_ids_sha256"
        ],
    )


def load_config(path: str | Path) -> UnionEsmConfig:
    """Load the exact union ESM extraction contract."""

    return _load_config(_read_snapshot(path, label="union ESM config"))


def _set_digest(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    payload = b"" if not ordered else ("\n".join(ordered) + "\n").encode("ascii")
    return _sha256(payload)


def _assignment_digest(examples: Sequence[Example]) -> str:
    payload = "".join(
        "\t".join(
            (
                item.example_id,
                item.assay_context_id,
                item.sequence_id,
                item.canonical_target,
                str(item.label),
                str(item.fold),
                item.homology_component_id,
                item.union_component_id,
            )
        )
        + "\n"
        for item in sorted(examples, key=lambda row: row.example_id)
    ).encode("utf-8")
    return _sha256(payload)


def _canonical_sequence(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} sequence must be a string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} sequence must be ASCII") from error
    if (
        value != value.strip().upper()
        or not _MIN_LENGTH <= len(value) <= _MAX_LENGTH
        or not set(value) <= _AMINO_ACIDS
    ):
        raise ValueError(f"{label} sequence is not canonical")
    return value


def _read_examples(snapshot: Snapshot, *, config: UnionEsmConfig) -> tuple[Example, ...]:
    rows = _jsonl(snapshot, label="Gate-1 examples")
    if len(rows) != config.expected_examples:
        raise ValueError("Gate-1 example count differs from the frozen config")
    examples: list[Example] = []
    previous: str | None = None
    sequence_metadata: dict[str, tuple[str, int, str, str]] = {}
    union_folds: dict[str, set[int]] = defaultdict(set)
    homology_locations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for number, row in enumerate(rows, start=1):
        label = f"Gate-1 example row {number}"
        _exact_fields(row, _EXAMPLE_FIELDS, label=label)
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"{label} schema_version must be 1")
        example_id = _sha_field(row["example_id"], label=f"{label} example_id")
        assay_context_id = _sha_field(row["assay_context_id"], label=f"{label} assay_context_id")
        if example_id != assay_context_id:
            raise ValueError(f"{label} example/context identity mismatch")
        if previous is not None and example_id <= previous:
            raise ValueError("Gate-1 examples must be strictly sorted by example_id")
        sequence = _canonical_sequence(row["sequence"], label=label)
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence_id")
        if sequence_id != _sha256(sequence.encode("ascii")):
            raise ValueError(f"{label} sequence_id mismatch")
        canonical_target = _string(row["canonical_target"], label=f"{label} target")
        gram = row["gram"]
        label_value = row["label"]
        fold = row["fold"]
        source_observations = row["source_observations"]
        if gram not in {"negative", "positive"}:
            raise ValueError(f"{label} has an invalid Gram class")
        if isinstance(label_value, bool) or label_value not in {0, 1}:
            raise ValueError(f"{label} has an invalid label")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < config.folds:
            raise ValueError(f"{label} has an invalid fold")
        if (
            isinstance(source_observations, bool)
            or not isinstance(source_observations, int)
            or source_observations < 1
        ):
            raise ValueError(f"{label} has an invalid source-observation count")
        homology_id = _sha_field(
            row["homology_component_id"], label=f"{label} homology_component_id"
        )
        union_id = _sha_field(row["union_component_id"], label=f"{label} union_component_id")
        metadata = (sequence, cast(int, fold), homology_id, union_id)
        prior = sequence_metadata.setdefault(sequence_id, metadata)
        if prior != metadata:
            raise ValueError("one modeled sequence has inconsistent sequence/fold components")
        union_folds[union_id].add(cast(int, fold))
        homology_locations[homology_id].add((union_id, cast(int, fold)))
        examples.append(
            Example(
                example_id=example_id,
                assay_context_id=assay_context_id,
                sequence_id=sequence_id,
                sequence=sequence,
                canonical_target=canonical_target,
                gram=cast(str, gram),
                label=cast(int, label_value),
                source_observations=cast(int, source_observations),
                fold=cast(int, fold),
                homology_component_id=homology_id,
                union_component_id=union_id,
            )
        )
        previous = example_id
    if any(len(folds) != 1 for folds in union_folds.values()):
        raise ValueError("one modeled union component spans multiple folds")
    if any(len(locations) != 1 for locations in homology_locations.values()):
        raise ValueError("one modeled homology component spans union components or folds")
    positives = sum(item.label for item in examples)
    if (
        sum(item.source_observations for item in examples) != config.expected_source_observations
        or len(sequence_metadata) != config.expected_sequences
        or positives != config.expected_positive_examples
        or len(examples) - positives != config.expected_negative_examples
    ):
        raise ValueError("Gate-1 example census differs from the frozen config")
    for fold in range(config.folds):
        selected = [item for item in examples if item.fold == fold]
        positive = sum(item.label for item in selected)
        if (
            len(selected) != config.expected_examples_by_fold[fold]
            or positive != config.expected_positive_examples_by_fold[fold]
            or len(selected) - positive != config.expected_negative_examples_by_fold[fold]
        ):
            raise ValueError(f"Gate-1 fold {fold} census differs from the frozen config")
    return tuple(examples)


def _metadata_tuple(item: Example) -> tuple[str, ...]:
    return (
        item.example_id,
        item.assay_context_id,
        item.sequence_id,
        item.sequence,
        item.canonical_target,
        item.gram,
        str(item.label),
        str(item.source_observations),
        str(item.fold),
        item.homology_component_id,
        item.union_component_id,
    )


def _verify_oof_metadata(
    snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    config: UnionEsmConfig,
) -> dict[str, float]:
    payload = snapshot.payload
    if (
        not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or b"\r" in payload
        or b"\x00" in payload
    ):
        raise ValueError("Gate-1 OOF must be LF-terminated UTF-8 CSV")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Gate-1 OOF must be UTF-8") from error
    expected = {item.example_id: item for item in examples}
    supports: dict[str, set[str]] = defaultdict(set)
    identities: dict[str, str] = {}
    previous: tuple[str, str] | None = None
    rows = 0
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(_OOF_FIELDS):
        raise ValueError("Gate-1 OOF CSV schema mismatch")
    for number, row in enumerate(reader, start=2):
        rows += 1
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Gate-1 OOF row {number} has the wrong column count")
        model = row["model"]
        example_id = row["example_id"]
        if model not in _BASE_MODELS or example_id not in expected:
            raise ValueError(f"Gate-1 OOF row {number} has unknown model/example")
        order_key = (model, example_id)
        if previous is not None and order_key <= previous:
            raise ValueError("Gate-1 OOF rows must be strictly sorted by model/example")
        if model in supports[example_id]:
            raise ValueError(f"Gate-1 OOF row {number} duplicates model/example")
        item = expected[example_id]
        observed = tuple(row[field] for field in _OOF_FIELDS[1:12])
        if observed != _metadata_tuple(item):
            raise ValueError(f"Gate-1 OOF row {number} metadata differs from examples.jsonl")
        identity_text = row["max_train_identity"]
        try:
            identity = float(identity_text)
        except ValueError as error:
            raise ValueError(f"Gate-1 OOF row {number} identity is invalid") from error
        if not math.isfinite(identity) or not 0 <= identity < config.homology_identity_threshold:
            raise ValueError(f"Gate-1 OOF row {number} violates the identity threshold")
        prior_identity = identities.setdefault(example_id, identity_text)
        if prior_identity != identity_text:
            raise ValueError("Gate-1 OOF model rows disagree on max_train_identity")
        # Probability is deliberately not parsed or retained.  Its accepted
        # bytes are hash-bound, but it cannot influence FASTA membership.
        supports[example_id].add(model)
        previous = order_key
    if rows != config.expected_examples * len(_BASE_MODELS):
        raise ValueError("Gate-1 OOF row census differs from the exact three-model panel")
    if set(supports) != set(expected) or any(
        set(models) != set(_BASE_MODELS) for models in supports.values()
    ):
        raise ValueError("Gate-1 OOF model support is incomplete")
    return {example_id: float(value) for example_id, value in identities.items()}


def _verify_folds(
    snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    identities: Mapping[str, float],
    config: UnionEsmConfig,
) -> None:
    document = _json_object(snapshot, label="Gate-1 folds")
    _exact_fields(
        document,
        {
            "schema_version",
            "artifact",
            "assignment_policy",
            "identity_threshold",
            "maximum_cross_fold_identity",
            "folds",
            "canonical_targets_by_fold",
            "assignments",
        },
        label="Gate-1 folds",
    )
    maximum = document["maximum_cross_fold_identity"]
    if (
        document["schema_version"] != 1
        or document["artifact"] != _BASE_FOLD_ARTIFACT
        or document["assignment_policy"] != _FOLD_POLICY
        or document["identity_threshold"] != config.homology_identity_threshold
        or isinstance(maximum, bool)
        or not isinstance(maximum, int | float)
        or not 0 <= float(maximum) < config.homology_identity_threshold
    ):
        raise ValueError("Gate-1 folds identity/policy contract mismatch")
    assignments = document["assignments"]
    if not isinstance(assignments, list) or len(assignments) != len(examples):
        raise ValueError("Gate-1 fold assignments do not cover every context")
    observed: dict[str, tuple[str, str, str, int]] = {}
    for number, raw in enumerate(assignments, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Gate-1 fold assignment {number} is not an object")
        _exact_fields(raw, _FOLD_ASSIGNMENT_FIELDS, label=f"Gate-1 fold assignment {number}")
        example_id = raw["example_id"]
        if not isinstance(example_id, str) or example_id in observed:
            raise ValueError(f"Gate-1 fold assignment {number} has an invalid example_id")
        observed[example_id] = (
            cast(str, raw["sequence_id"]),
            cast(str, raw["homology_component_id"]),
            cast(str, raw["union_component_id"]),
            cast(int, raw["fold"]),
        )
    expected = {
        item.example_id: (
            item.sequence_id,
            item.homology_component_id,
            item.union_component_id,
            item.fold,
        )
        for item in examples
    }
    if observed != expected or set(identities) != set(expected):
        raise ValueError("Gate-1 folds/OOF metadata do not match the context panel")
    fold_summary: dict[str, dict[str, int]] = {}
    targets: dict[str, dict[str, int]] = {}
    for fold in range(config.folds):
        selected = [item for item in examples if item.fold == fold]
        class_counts = {
            metric: sum(
                item.gram == metric.split("_")[1] and item.label == int(metric.endswith("positive"))
                for item in selected
            )
            for metric in _CLASS_METRICS
        }
        fold_summary[str(fold)] = {
            "examples": len(selected),
            "source_observations": sum(item.source_observations for item in selected),
            "sequences": len({item.sequence_id for item in selected}),
            "homology_components": len({item.homology_component_id for item in selected}),
            "union_components": len({item.union_component_id for item in selected}),
            "positives": sum(item.label for item in selected),
            "negatives": sum(1 - item.label for item in selected),
            **class_counts,
        }
        targets[str(fold)] = dict(
            sorted(Counter(item.canonical_target for item in selected).items())
        )
    if float(maximum) != max(identities.values()):
        raise ValueError("Gate-1 folds maximum identity differs from the OOF metadata")
    if document["folds"] != fold_summary or document["canonical_targets_by_fold"] != targets:
        raise ValueError("Gate-1 fold/target summary differs from reconstructed contexts")


def _artifact_hash(
    artifacts: object,
    *,
    key: str,
    filename: str,
    digest: str,
) -> None:
    if not isinstance(artifacts, dict):
        raise ValueError("Gate-1 manifest artifacts must be an object")
    item = artifacts.get(key)
    if (
        not isinstance(item, dict)
        or set(item) != {"filename", "sha256", "role"}
        or item.get("filename") != filename
        or item.get("sha256") != digest
        or not isinstance(item.get("role"), str)
        or not item.get("role")
    ):
        raise ValueError(f"Gate-1 manifest does not bind {filename}")


def _verify_base_documents(
    *,
    manifest_snapshot: Snapshot,
    split_receipt_snapshot: Snapshot,
    independent_receipt_snapshot: Snapshot,
    examples_snapshot: Snapshot,
    folds_snapshot: Snapshot,
    oof_snapshot: Snapshot,
    config: UnionEsmConfig,
) -> None:
    manifest = _json_object(manifest_snapshot, label="Gate-1 manifest")
    _exact_fields(
        manifest,
        {
            "schema_version",
            "artifact",
            "status",
            "config_sha256",
            "git_commit",
            "input_sha256",
            "accepted_split",
            "code_attestation",
            "policies",
            "models",
            "label_summary",
            "identity_audit",
            "runtime",
            "artifacts",
        },
        label="Gate-1 manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["artifact"] != _BASE_ARTIFACT
        or manifest["status"] != _BASE_STATUS
        or manifest["config_sha256"] != config.gate1_config_sha256
        or manifest["git_commit"] != config.gate1_git_commit
        or manifest["models"] != list(_BASE_MODELS)
    ):
        raise ValueError("Gate-1 manifest identity differs from the accepted benchmark")
    policies = manifest["policies"]
    if (
        not isinstance(policies, dict)
        or policies.get("label") != _LABEL_POLICY
        or policies.get("fold_assignment") != _FOLD_POLICY
        or policies.get("bootstrap_unit") != "union_component_id"
        or policies.get("ensemble") != "untrained_equal_probability_mean"
        or policies.get("model_features") != ["sequence", "canonical_target", "gram"]
        or policies.get("forbidden_model_features") != list(_FORBIDDEN_MODEL_FEATURES)
        or policies.get("context_feature") != "canonical_target"
    ):
        raise ValueError("Gate-1 manifest leakage/split policy mismatch")
    code = manifest["code_attestation"]
    if (
        not isinstance(code, dict)
        or code.get("code_manifest_sha256") != config.gate1_code_manifest_sha256
        or not isinstance(code.get("invariants"), dict)
        or not cast(dict[str, object], code["invariants"])
        or not all(value is True for value in cast(dict[str, object], code["invariants"]).values())
    ):
        raise ValueError("Gate-1 code attestation is invalid")
    _artifact_hash(
        manifest["artifacts"],
        key="examples",
        filename="examples.jsonl",
        digest=examples_snapshot.sha256,
    )
    _artifact_hash(
        manifest["artifacts"],
        key="folds",
        filename="folds.json",
        digest=folds_snapshot.sha256,
    )
    _artifact_hash(
        manifest["artifacts"],
        key="oof",
        filename="oof_predictions.csv",
        digest=oof_snapshot.sha256,
    )
    _artifact_hash(
        manifest["artifacts"],
        key="split_receipt",
        filename="split_receipt.json",
        digest=split_receipt_snapshot.sha256,
    )
    labels = manifest["label_summary"]
    if (
        not isinstance(labels, dict)
        or labels.get("context_examples") != config.expected_examples
        or labels.get("source_observations") != config.expected_source_observations
        or labels.get("modeled_sequences") != config.expected_sequences
        or labels.get("positive_examples") != config.expected_positive_examples
        or labels.get("negative_examples") != config.expected_negative_examples
    ):
        raise ValueError("Gate-1 manifest label census mismatch")

    receipt = _json_object(split_receipt_snapshot, label="Gate-1 split receipt")
    _exact_fields(
        receipt,
        {
            "schema_version",
            "artifact",
            "status",
            "assignment_policy",
            "input_sha256",
            "accepted_split",
            "code_attestation",
            "census",
            "identity",
            "invariants",
        },
        label="Gate-1 split receipt",
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact") != _BASE_RECEIPT_ARTIFACT
        or receipt.get("status") != "passed"
        or receipt.get("assignment_policy") != _FOLD_POLICY
        or receipt.get("input_sha256") != manifest["input_sha256"]
        or receipt.get("accepted_split") != manifest["accepted_split"]
        or receipt.get("code_attestation") != manifest["code_attestation"]
    ):
        raise ValueError("Gate-1 split receipt identity mismatch")
    census = receipt.get("census")
    if not isinstance(census, dict) or census.get("context_examples") != config.expected_examples:
        raise ValueError("Gate-1 split receipt census mismatch")
    expected_census = {
        "source_observations": config.expected_source_observations,
        "modeled_sequences": config.expected_sequences,
        "positives": config.expected_positive_examples,
        "negatives": config.expected_negative_examples,
        "examples_by_fold": list(config.expected_examples_by_fold),
        "positive_examples_by_fold": list(config.expected_positive_examples_by_fold),
        "negative_examples_by_fold": list(config.expected_negative_examples_by_fold),
    }
    if any(census.get(key) != value for key, value in expected_census.items()):
        raise ValueError("Gate-1 split receipt detailed census mismatch")
    invariants = receipt.get("invariants")
    if (
        not isinstance(invariants, dict)
        or set(invariants) != set(_SPLIT_RECEIPT_INVARIANTS)
        or not all(value is True for value in invariants.values())
    ):
        raise ValueError("Gate-1 split receipt invariants did not all pass")

    audit = _json_object(independent_receipt_snapshot, label="Gate-1 independent receipt")
    expected_audit_fields = {
        "schema_version",
        "artifact",
        "status",
        "checks",
        "git_commit",
        "config_sha256",
        "publication_top_manifest_sha256",
        "gate1_top_manifest_sha256",
        "code_manifest_sha256",
        "frozen_input_manifest_sha256",
        "input_sha256",
        "artifact_sha256",
        "census",
        "identity",
        "overall_metrics",
        "union_component_bootstrap_95ci",
        "production_handshake",
    }
    _exact_fields(audit, expected_audit_fields, label="Gate-1 independent receipt")
    if (
        audit["schema_version"] != 1
        or audit["artifact"] != _BASE_AUDIT_ARTIFACT
        or audit["status"] != "passed"
        or audit["git_commit"] != config.gate1_git_commit
        or audit["config_sha256"] != config.gate1_config_sha256
        or audit["publication_top_manifest_sha256"] != config.gate1_publication_top_sha256
        or audit["gate1_top_manifest_sha256"] != config.gate1_top_sha256
        or audit["code_manifest_sha256"] != config.gate1_code_manifest_sha256
        or audit["frozen_input_manifest_sha256"] != config.gate1_frozen_input_manifest_sha256
    ):
        raise ValueError("Gate-1 independent receipt identity/hash chain mismatch")
    checks = audit["checks"]
    if (
        not isinstance(checks, dict)
        or set(checks) != set(_INDEPENDENT_CHECKS)
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("Gate-1 independent receipt checks are not exact/passing")
    audit_artifacts = audit["artifact_sha256"]
    if not isinstance(audit_artifacts, dict) or any(
        audit_artifacts.get(filename) != digest
        for filename, digest in {
            "examples.jsonl": examples_snapshot.sha256,
            "folds.json": folds_snapshot.sha256,
            "oof_predictions.csv": oof_snapshot.sha256,
            "manifest.json": manifest_snapshot.sha256,
            "split_receipt.json": split_receipt_snapshot.sha256,
        }.items()
    ):
        raise ValueError("Gate-1 independent receipt artifact hashes mismatch")
    audit_census = audit["census"]
    if (
        not isinstance(audit_census, dict)
        or audit_census.get("context_examples") != config.expected_examples
        or audit_census.get("source_observations") != config.expected_source_observations
        or audit_census.get("modeled_sequences") != config.expected_sequences
        or audit_census.get("positives") != config.expected_positive_examples
        or audit_census.get("negatives") != config.expected_negative_examples
        or audit_census.get("examples_by_fold") != list(config.expected_examples_by_fold)
        or audit_census.get("positive_examples_by_fold")
        != list(config.expected_positive_examples_by_fold)
        or audit_census.get("negative_examples_by_fold")
        != list(config.expected_negative_examples_by_fold)
    ):
        raise ValueError("Gate-1 independent receipt census mismatch")


def _validate_input_manifests(
    *,
    publication_snapshot: Snapshot,
    gate1_top_snapshot: Snapshot,
    examples_snapshot: Snapshot,
    folds_snapshot: Snapshot,
    oof_snapshot: Snapshot,
    manifest_snapshot: Snapshot,
    split_receipt_snapshot: Snapshot,
    config: UnionEsmConfig,
) -> None:
    for snapshot, expected, label in (
        (publication_snapshot, config.gate1_publication_top_sha256, "publication top"),
        (gate1_top_snapshot, config.gate1_top_sha256, "Gate-1 top"),
        (examples_snapshot, config.gate1_examples_sha256, "Gate-1 examples"),
        (folds_snapshot, config.gate1_folds_sha256, "Gate-1 folds"),
        (oof_snapshot, config.gate1_oof_sha256, "Gate-1 OOF"),
        (manifest_snapshot, config.gate1_manifest_sha256, "Gate-1 manifest"),
        (split_receipt_snapshot, config.gate1_split_receipt_sha256, "Gate-1 split receipt"),
    ):
        if snapshot.sha256 != expected:
            raise ValueError(f"{label} checksum differs from the frozen config")
    publication = _manifest(publication_snapshot, label="Gate-1 publication top manifest")
    if set(publication) != set(_PUBLICATION_ENTRIES):
        raise ValueError("Gate-1 publication manifest inventory mismatch")
    expected_publication = {
        "CODE_SHA256SUMS": config.gate1_code_manifest_sha256,
        "FROZEN_INPUT_SHA256SUMS": config.gate1_frozen_input_manifest_sha256,
        "gate1/SHA256SUMS": gate1_top_snapshot.sha256,
        "gate1/examples.jsonl": examples_snapshot.sha256,
        "gate1/folds.json": folds_snapshot.sha256,
        "gate1/manifest.json": manifest_snapshot.sha256,
        "gate1/oof_predictions.csv": oof_snapshot.sha256,
        "gate1/split_receipt.json": split_receipt_snapshot.sha256,
    }
    if any(publication.get(name) != digest for name, digest in expected_publication.items()):
        raise ValueError("Gate-1 publication manifest does not bind supplied artifacts")
    gate1_top = _manifest(gate1_top_snapshot, label="Gate-1 semantic top manifest")
    if set(gate1_top) != set(_GATE1_ENTRIES):
        raise ValueError("Gate-1 semantic manifest inventory mismatch")
    expected_gate1 = {
        "examples.jsonl": examples_snapshot.sha256,
        "folds.json": folds_snapshot.sha256,
        "manifest.json": manifest_snapshot.sha256,
        "oof_predictions.csv": oof_snapshot.sha256,
        "split_receipt.json": split_receipt_snapshot.sha256,
    }
    if any(gate1_top.get(name) != digest for name, digest in expected_gate1.items()):
        raise ValueError("Gate-1 semantic manifest does not bind supplied artifacts")
    if any(publication[f"gate1/{name}"] != digest for name, digest in gate1_top.items()):
        raise ValueError("Gate-1 publication and semantic manifests disagree")


def _validate_code_manifest(
    snapshot: Snapshot,
    *,
    config_snapshot: Snapshot,
) -> tuple[dict[str, object], tuple[Snapshot, ...]]:
    entries = _manifest(snapshot, label="union ESM code manifest")
    repository = Path(__file__).resolve(strict=True).parents[3]
    source_root = repository / "src" / "amp_challenge"
    sources: set[str] = set()
    for path in source_root.rglob("*.py"):
        if path.is_symlink():
            raise ValueError("repository Python inventory contains a symbolic link")
        if path.is_file():
            sources.add(path.relative_to(repository).as_posix())
    expected = set(_FIXED_CODE_PATHS) | sources
    if set(entries) != expected:
        raise ValueError(
            "union ESM code manifest inventory mismatch: "
            f"missing={sorted(expected - set(entries))}, extra={sorted(set(entries) - expected)}"
        )
    snapshots: list[Snapshot] = []
    for logical_path in sorted(expected):
        if logical_path == _LOGICAL_CONFIG_PATH:
            digest = config_snapshot.sha256
        else:
            item = _read_snapshot(repository / logical_path, label=f"code entry {logical_path}")
            snapshots.append(item)
            digest = item.sha256
        if entries[logical_path] != digest:
            raise ValueError(f"union ESM code manifest checksum mismatch for {logical_path}")
    return (
        {
            "schema_version": 1,
            "code_manifest_sha256": snapshot.sha256,
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
                "the exporter verifies bytes and runs no Git subprocess"
            ),
            "invariants": {
                "code_manifest_inventory_exact": True,
                "every_repository_code_entry_hash_verified": True,
                "executing_module_bound_to_manifest": True,
                "supplied_config_bound_to_logical_manifest_entry": True,
            },
        },
        tuple(snapshots),
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _write_top_manifest(path: Path, entries: Mapping[str, str]) -> None:
    payload = "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("utf-8")
    _write_bytes(path, payload)


def export_union_esm_fasta(
    *,
    gate1_publication_top_manifest_path: str | Path,
    gate1_top_manifest_path: str | Path,
    gate1_examples_path: str | Path,
    gate1_folds_path: str | Path,
    gate1_oof_path: str | Path,
    gate1_manifest_path: str | Path,
    gate1_split_receipt_path: str | Path,
    gate1_independent_receipt_path: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Validate accepted union evidence and export its exact modeled panel."""

    if _GIT_RE.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase Git SHA")
    requested_output = Path(output_dir)
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse union ESM FASTA output: {requested_output}")
    output = requested_output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse union ESM FASTA output: {output}")

    snapshots = {
        "publication_top": _read_snapshot(
            gate1_publication_top_manifest_path, label="Gate-1 publication top manifest"
        ),
        "gate1_top": _read_snapshot(gate1_top_manifest_path, label="Gate-1 semantic top manifest"),
        "examples": _read_snapshot(gate1_examples_path, label="Gate-1 examples"),
        "folds": _read_snapshot(gate1_folds_path, label="Gate-1 folds"),
        "oof": _read_snapshot(gate1_oof_path, label="Gate-1 OOF predictions"),
        "manifest": _read_snapshot(gate1_manifest_path, label="Gate-1 manifest"),
        "split_receipt": _read_snapshot(gate1_split_receipt_path, label="Gate-1 split receipt"),
        "independent_receipt": _read_snapshot(
            gate1_independent_receipt_path, label="Gate-1 independent receipt"
        ),
        "config": _read_snapshot(config_path, label="union ESM config"),
        "code_manifest": _read_snapshot(code_manifest_path, label="union ESM code manifest"),
    }
    config = _load_config(snapshots["config"])
    if snapshots["independent_receipt"].sha256 != config.gate1_independent_receipt_sha256:
        raise ValueError("Gate-1 independent receipt checksum differs from the frozen config")
    _validate_input_manifests(
        publication_snapshot=snapshots["publication_top"],
        gate1_top_snapshot=snapshots["gate1_top"],
        examples_snapshot=snapshots["examples"],
        folds_snapshot=snapshots["folds"],
        oof_snapshot=snapshots["oof"],
        manifest_snapshot=snapshots["manifest"],
        split_receipt_snapshot=snapshots["split_receipt"],
        config=config,
    )
    _verify_base_documents(
        manifest_snapshot=snapshots["manifest"],
        split_receipt_snapshot=snapshots["split_receipt"],
        independent_receipt_snapshot=snapshots["independent_receipt"],
        examples_snapshot=snapshots["examples"],
        folds_snapshot=snapshots["folds"],
        oof_snapshot=snapshots["oof"],
        config=config,
    )
    examples = _read_examples(snapshots["examples"], config=config)
    identities = _verify_oof_metadata(snapshots["oof"], examples=examples, config=config)
    _verify_folds(snapshots["folds"], examples=examples, identities=identities, config=config)
    code_attestation, code_snapshots = _validate_code_manifest(
        snapshots["code_manifest"], config_snapshot=snapshots["config"]
    )

    sequences: dict[str, str] = {}
    for item in examples:
        previous = sequences.setdefault(item.sequence_id, item.sequence)
        if previous != item.sequence:  # pragma: no cover - SHA collision defense
            raise ValueError("one sequence_id maps to multiple canonical sequences")
    if len(sequences) != config.expected_sequences:
        raise AssertionError("validated context panel did not yield the expected sequence count")
    records = tuple(sorted(sequences.items()))
    fasta_payload = b"".join(
        f">sequence_id={sequence_id}\n{sequence}\n".encode("ascii")
        for sequence_id, sequence in records
    )
    sequence_ids = set(sequences)
    configured_design_hint_missing = set(config.historical_design_hint_missing_union_sequence_ids)
    if not configured_design_hint_missing <= sequence_ids:
        raise ValueError(
            "the accepted panel does not contain every configured design-hint union ID"
        )
    input_sha256 = {
        "gate1_publication_top_manifest": snapshots["publication_top"].sha256,
        "gate1_top_manifest": snapshots["gate1_top"].sha256,
        "gate1_examples": snapshots["examples"].sha256,
        "gate1_folds": snapshots["folds"].sha256,
        "gate1_oof": snapshots["oof"].sha256,
        "gate1_manifest": snapshots["manifest"].sha256,
        "gate1_split_receipt": snapshots["split_receipt"].sha256,
        "gate1_independent_receipt": snapshots["independent_receipt"].sha256,
        "config": snapshots["config"].sha256,
        "code_manifest": snapshots["code_manifest"].sha256,
    }
    historical_design_hint = {
        "status": config.historical_design_hint_status,
        "source_status": "preliminary_observation_not_consumed_or_verified_by_this_protocol",
        "configured_historical_input_fasta_sha256": (
            config.historical_design_hint_input_fasta_sha256
        ),
        "configured_historical_input_fasta_manifest_sha256": (
            config.historical_design_hint_input_fasta_manifest_sha256
        ),
        "configured_historical_embedding_records": (
            config.historical_design_hint_embedding_records
        ),
        "configured_old_cohort_extra_embedding_records": {
            "sequence_ids": list(config.historical_design_hint_extra_embedding_sequence_ids),
            "sequence_ids_sha256": (
                config.historical_design_hint_extra_embedding_sequence_ids_sha256
            ),
        },
        "configured_union_sequences_reported_missing_from_historical_embeddings": {
            "sequence_ids": list(config.historical_design_hint_missing_union_sequence_ids),
            "sequence_ids_sha256": (
                config.historical_design_hint_missing_union_sequence_ids_sha256
            ),
            "count": len(config.historical_design_hint_missing_union_sequence_ids),
            "all_present_in_exported_union_panel": True,
        },
        "all_union_sequences_exported_regardless_of_design_hint": True,
    }
    coverage = {
        "schema_version": 1,
        "artifact": _COVERAGE_ARTIFACT,
        "status": "passed",
        "input_sha256": input_sha256,
        "digest_encoding": (
            "sorted unique lowercase SHA-256 identifier strings encoded as ASCII, "
            "joined by LF with a terminal LF; empty set is empty bytes"
        ),
        "assignment_digest_encoding": (
            "example_id, assay_context_id, sequence_id, canonical_target, label, fold, "
            "homology_component_id, union_component_id encoded as tab-separated UTF-8 "
            "records sorted by example_id with a terminal LF"
        ),
        "panel": {
            "context_examples": len(examples),
            "source_observations": sum(item.source_observations for item in examples),
            "unique_sequences": len(records),
            "positive_examples": sum(item.label for item in examples),
            "negative_examples": sum(1 - item.label for item in examples),
            "homology_components": len({item.homology_component_id for item in examples}),
            "union_components": len({item.union_component_id for item in examples}),
            "sequence_ids_sha256": _set_digest(sequence_ids),
            "example_ids_sha256": _set_digest(item.example_id for item in examples),
            "example_assignment_sha256": _assignment_digest(examples),
            "examples_by_fold": list(config.expected_examples_by_fold),
            "positive_examples_by_fold": list(config.expected_positive_examples_by_fold),
            "negative_examples_by_fold": list(config.expected_negative_examples_by_fold),
        },
        "export": {
            "records": len(records),
            "fasta_sha256": _sha256(fasta_payload),
            "ordering": "ascending sequence_id",
            "header_format": ">sequence_id=<lowercase_sha256>",
            "line_format": "exactly two LF-terminated ASCII lines per record",
            "missing_sequences": 0,
            "missing_sequence_ids": [],
            "missing_sequence_ids_sha256": _set_digest(()),
            "extra_sequences": 0,
            "extra_sequence_ids": [],
            "extra_sequence_ids_sha256": _set_digest(()),
        },
        "unverified_historical_design_hint_not_evidence": historical_design_hint,
        "invariants": {
            "accepted_gate1_publication_chain_valid": True,
            "accepted_gate1_independent_receipt_valid": True,
            "all_oof_models_have_identical_context_metadata": True,
            "base_probabilities_not_consumed": True,
            "context_panel_not_collapsed_before_sequence_export": True,
            "exact_modeled_sequence_set_exported": True,
            "configured_design_hint_ids_not_used_as_exclusions": True,
            "configured_design_hint_missing_union_ids_present_in_export": True,
            "union_components_fold_disjoint": True,
        },
    }
    coverage_payload = _json_bytes(coverage)
    manifest = {
        "schema_version": 1,
        "artifact": _OUTPUT_ARTIFACT,
        "status": "candidate_pending_independent_verification",
        "production_eligible": False,
        "production_ineligibility_reason": (
            "independent_export_reconstruction_receipt_not_yet_available"
        ),
        "git_commit": git_commit,
        "config_sha256": snapshots["config"].sha256,
        "input_sha256": input_sha256,
        "code_attestation": code_attestation,
        "base_contract": {
            "artifact": _BASE_ARTIFACT,
            "git_commit": config.gate1_git_commit,
            "fold_assignment_policy": _FOLD_POLICY,
            "context_examples": config.expected_examples,
            "source_observations": config.expected_source_observations,
            "modeled_sequences": config.expected_sequences,
        },
        "extraction_contract": {
            "embedding_model": config.embedding_model,
            "representation_layer": config.representation_layer,
            "embedding_dimension": config.embedding_dimension,
            "embedding_batch_size": config.embedding_batch_size,
            "embedding_seed": config.embedding_seed,
            "model_checkpoint_sha256": config.model_checkpoint_sha256,
            "contact_regression_sha256": config.contact_regression_sha256,
            "environment_lock_sha256": config.environment_lock_sha256,
            "trust_manifest_sha256": config.trust_manifest_sha256,
            "embedding_worker_sha256": config.embedding_worker_sha256,
            "runtime": {
                "python": config.python_version,
                "torch": config.torch_version,
                "fair_esm": config.fair_esm_version,
                "numpy": config.numpy_version,
                "cuda_runtime": config.cuda_runtime,
                "cudnn_version": config.cudnn_version,
                "device_type": config.device_type,
            },
        },
        "panel": coverage["panel"],
        "unverified_historical_design_hint_not_evidence": historical_design_hint,
        "artifacts": {
            "fasta": {
                "filename": "union_esm_sequences.fasta",
                "sha256": _sha256(fasta_payload),
                "records": len(records),
            },
            "coverage_receipt": {
                "filename": "coverage_receipt.json",
                "sha256": _sha256(coverage_payload),
            },
        },
        "invariants": {
            "accepted_union_gate1_only": True,
            "code_and_config_attested": True,
            "deterministic_sequence_id_order": True,
            "exact_configured_sequence_candidate_panel": len(records) == config.expected_sequences,
            "independent_verification_required_before_production": True,
            "path_free_semantic_artifacts": True,
        },
    }
    if not all(cast(dict[str, bool], manifest["invariants"]).values()):
        raise AssertionError("union ESM manifest invariant failed")
    manifest_payload = _json_bytes(manifest)

    for name, snapshot in snapshots.items():
        _assert_unchanged(snapshot, label=name)
    for snapshot in code_snapshots:
        _assert_unchanged(snapshot, label="code inventory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    published = False
    try:
        _write_bytes(staging / "union_esm_sequences.fasta", fasta_payload)
        _write_bytes(staging / "coverage_receipt.json", coverage_payload)
        _write_bytes(staging / "manifest.json", manifest_payload)
        _write_top_manifest(
            staging / "SHA256SUMS",
            {
                "coverage_receipt.json": _sha256(coverage_payload),
                "manifest.json": _sha256(manifest_payload),
                "union_esm_sequences.fasta": _sha256(fasta_payload),
            },
        )
        for name, snapshot in snapshots.items():
            _assert_unchanged(snapshot, label=name)
        for snapshot in code_snapshots:
            _assert_unchanged(snapshot, label="code inventory")
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to replace union ESM FASTA output: {output}")
        staging.rename(output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-publication-top-manifest", type=Path, required=True)
    parser.add_argument("--gate1-top-manifest", type=Path, required=True)
    parser.add_argument("--gate1-examples", type=Path, required=True)
    parser.add_argument("--gate1-folds", type=Path, required=True)
    parser.add_argument("--gate1-oof", type=Path, required=True)
    parser.add_argument("--gate1-manifest", type=Path, required=True)
    parser.add_argument("--gate1-split-receipt", type=Path, required=True)
    parser.add_argument("--gate1-independent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_union_esm_fasta(
        gate1_publication_top_manifest_path=args.gate1_publication_top_manifest,
        gate1_top_manifest_path=args.gate1_top_manifest,
        gate1_examples_path=args.gate1_examples,
        gate1_folds_path=args.gate1_folds,
        gate1_oof_path=args.gate1_oof,
        gate1_manifest_path=args.gate1_manifest,
        gate1_split_receipt_path=args.gate1_split_receipt,
        gate1_independent_receipt_path=args.gate1_independent_receipt,
        config_path=args.config,
        code_manifest_path=args.code_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

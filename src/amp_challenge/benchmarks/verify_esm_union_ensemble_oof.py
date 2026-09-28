"""Independently reconstruct and verify nested union ESM ensemble twins.

This module deliberately imports none of the ensemble producer, baseline
models, descriptor helpers, similarity helpers, or metric helpers.  Accepted
Gate-1 and embedding task trees are read-only, content-addressed inputs; their
job-container directories are not claimed to be persistent immutable storage.
All descriptor, homology-kNN, ESM-head, nested-stack, prediction, metric, and
shared component-bootstrap calculations are reimplemented here.  Producer
files are comparison targets only; no fitted or reported producer value is
used as a reconstruction input.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import struct
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GramClass = Literal["negative", "positive"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_ABSOLUTE_BYTES_RE = re.compile(
    rb"(?:/home/|/lustre/|/tmp/|/scratch/|/mnt/|file://|(?<![A-Za-z0-9])[A-Za-z]:[\\\\/])"
)
_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_AMINO_ACID_SET = frozenset(_AMINO_ACIDS)
_MIN_LENGTH = 8
_MAX_LENGTH = 50

_ARTIFACT = "esm_union_nested_ensemble_oof_v1"
_OUTPUT_STATUS = "repeatedly_inspected_development_evidence"
_CONFIG_PATH = "configs/evaluation/esm_union_ensemble_oof_v1.toml"
_BASE_CONFIG_PATH = "configs/benchmarks/oracle_gate1_union_v1.toml"
_EMBEDDING_CONFIG_PATH = "configs/models/esm2_union_embeddings_v1.toml"
_PRODUCER_MODULE_PATH = "src/amp_challenge/benchmarks/esm_union_ensemble_oof.py"
_VERIFIER_MODULE_PATH = "src/amp_challenge/benchmarks/verify_esm_union_ensemble_oof.py"
_CONFIG_SHA256 = "53f045b05ead39147f7cea0c9d09da8ea473e2c1d3c74ef8f5f4dfcce1935b7f"

_METHODS = (
    "descriptor_logistic",
    "homology_knn",
    "equal_weight_ensemble",
    "esm2_t6_8m_target_gram_logistic",
    "descriptor_esm_probability_half_blend",
    "base_family_esm_probability_half_blend",
    "nested_union_base_esm_logistic_stack_v1",
)
_BASE_MODELS = _METHODS[:3]
_ESM_MODEL = _METHODS[3]
_DESCRIPTOR_ESM_BLEND = _METHODS[4]
_BASE_ESM_BLEND = _METHODS[5]
_STACK_MODEL = _METHODS[6]
_STACK_FEATURES = ("descriptor_logistic", "homology_knn", _ESM_MODEL)
_PROMOTION_REFERENCE = "descriptor_logistic"
_PROMOTION_CANDIDATE = _STACK_MODEL

_HANDSHAKE_FILES = frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"})
_BASE_RECEIPT_CHECKS = frozenset(
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
_EMBEDDING_RECEIPT_FIELDS = frozenset(
    {
        "acceptance_scope",
        "artifact",
        "artifact_sha256",
        "checks",
        "config_sha256",
        "git_commit",
        "limitations",
        "panel",
        "producer",
        "reference_worker_attestation",
        "runtime",
        "schema_version",
        "status",
        "tensor",
        "trust",
        "verifier_attestation",
    }
)
_EMBEDDING_RECEIPT_CHECKS = frozenset(
    {
        "accepted_panel_evidence_chain_valid",
        "accepted_panel_twins_immutable_and_identical",
        "all_eight_trusted_bundle_artifacts_rehashed",
        "canonical_index_independently_reconstructed",
        "canonical_npy_v1_little_endian_float32_verified",
        "code_manifest_and_git_attestation_valid",
        "full_matrix_independently_recomputed",
        "independent_reference_manifest_reconstructed",
        "inputs_rehashed_after_verification",
        "path_free_publication_and_receipt",
        "producer_frozen_input_manifest_reconstructed",
        "producer_reference_npy_bitwise_identical",
        "producer_reference_runtime_and_driver_identical",
        "producer_reference_tensor_bytes_bitwise_identical",
        "producer_semantic_manifest_independently_reconstructed",
        "producer_twins_bitwise_identical",
        "scope_limited_to_embedding_feature_input",
    }
)
_BASE_SEMANTIC_FILES = frozenset(
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
_BASE_TREE_FILES = frozenset(
    {
        "SHA256SUMS",
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "gate1/SHA256SUMS",
        *(f"gate1/{name}" for name in _BASE_SEMANTIC_FILES),
    }
)
_EMBEDDING_TREE_FILES = frozenset(
    {
        "SHA256SUMS",
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "embeddings/SHA256SUMS",
        "embeddings/embedding_index.csv",
        "embeddings/embedding_manifest.json",
        "embeddings/embeddings.npy",
    }
)
_PRODUCER_SEMANTIC_FILES = frozenset(
    {
        "SHA256SUMS",
        "embedding_coverage.json",
        "esm_union_oof_predictions.csv",
        "fold_models.json",
        "manifest.json",
        "metrics.json",
        "nested_fits.json",
    }
)
_SEMANTIC_OUTPUTS = _PRODUCER_SEMANTIC_FILES - {"SHA256SUMS"}
_PRODUCER_FILES = frozenset(
    {
        "SHA256SUMS",
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        *(f"ensemble/{name}" for name in _PRODUCER_SEMANTIC_FILES),
    }
)
_FIXED_CODE_PATHS = frozenset(
    {
        _CONFIG_PATH,
        _BASE_CONFIG_PATH,
        _EMBEDDING_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
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
_PREDICTION_SCHEMA = _BASE_OOF_SCHEMA
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
_BOOTSTRAP_DRAW_ENCODING = (
    "zero-based replicate, sampled-position, union_component_id encoded as tab-separated "
    "ASCII records in generator order with a terminal LF"
)
_NESTED_FEATURE_ENCODING = (
    "outer_fold, inner_fold, example_id, descriptor_logit, homology_knn_logit, esm_logit; "
    "tab-separated ASCII, floats encoded with Python float.hex(), sorted by outer, inner, "
    "example_id, terminal LF"
)
_PRODUCTION_POLICY = {
    "evaluation_status": "repeatedly_inspected_development_evidence",
    "statistical_pass_status": "eligible_for_untouched_external_validation_only",
    "production_eligible": False,
    "production_apex_weight": 0.0,
    "production_esm_weight": 0.0,
    "apex_feature_allowed": False,
    "statistical_gate_controls_production_eligibility": False,
    "blockers": [
        "union_gate1_labels_have_been_repeatedly_inspected",
        "untouched_external_confirmation_required",
        "esm2_ur50d_exact_pretraining_sequence_membership_not_established",
    ],
}
_EMBEDDING_ACCEPTANCE_SCOPE = {
    "accepted_use": "downstream_embedding_feature_input_only",
    "embedding_extraction_executed": True,
    "embedding_feature_input_eligible": True,
    "embeddings_verified": True,
    "ensemble_weight_established": False,
    "full_embedding_matrix_independently_recomputed": True,
    "independent_reference_bitwise_identical": True,
    "model_performance_evidence": False,
    "model_predictions_verified": False,
    "pretraining_membership_independence_established": False,
    "producer_twins_bitwise_identical": True,
}

_RESIDUE_MASSES_DA = {
    "A": 71.0788,
    "C": 103.1388,
    "D": 115.0886,
    "E": 129.1155,
    "F": 147.1766,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "K": 128.1741,
    "L": 113.1594,
    "M": 131.1926,
    "N": 114.1038,
    "P": 97.1167,
    "Q": 128.1307,
    "R": 156.1875,
    "S": 87.0782,
    "T": 101.1051,
    "V": 99.1326,
    "W": 186.2132,
    "Y": 163.1760,
}
_WATER_MASS_DA = 18.01528
_HYDROPHOBICITY = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}
_HYDROPHOBIC = frozenset("ACFILMVWY")
_AROMATIC = frozenset("FWY")
_BASIC = frozenset("HKR")
_ACIDIC = frozenset("DE")
_POSITIVE_PKA = {"H": 6.0, "K": 10.5, "R": 12.5}
_NEGATIVE_PKA = {"C": 8.3, "D": 3.9, "E": 4.1, "Y": 10.1}
_N_TERMINUS_PKA = 8.0
_C_TERMINUS_PKA = 3.1


class VerificationError(ValueError):
    """Raised when evidence differs from independent reconstruction."""


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
class BaseConfig:
    logistic: LogisticSettings
    knn: KnnSettings


@dataclass(frozen=True, slots=True)
class Config:
    path: Path
    sha256: str
    methods: tuple[str, ...]
    probability_clip: float
    canonical_targets: tuple[str, ...]
    gram_classes: tuple[str, ...]
    esm_l2: float
    esm_prior_strength: float
    esm_max_iterations: int
    esm_tolerance: float
    stack_l2: float
    stack_prior_strength: float
    stack_max_iterations: int
    stack_tolerance: float
    folds: int
    homology_identity_threshold: float
    similarity_bin_edges: tuple[float, ...]
    calibration_bins: int
    bootstrap_replicates: int
    bootstrap_seed: int
    base_producer_job_id: int
    base_audit_job_id: int
    base_publication_top_sha256: str
    base_semantic_top_sha256: str
    base_examples_sha256: str
    base_folds_sha256: str
    base_oof_sha256: str
    base_manifest_sha256: str
    base_split_receipt_sha256: str
    base_independent_receipt_sha256: str
    base_config_sha256: str
    embedding_protocol_git_commit: str
    embedding_producer_job_id: int
    embedding_audit_job_id: int
    embedding_publication_top_sha256: str
    embedding_semantic_top_sha256: str
    embedding_index_sha256: str
    embedding_manifest_sha256: str
    embedding_matrix_sha256: str
    embedding_tensor_data_sha256: str
    embedding_independent_receipt_sha256: str
    embedding_config_sha256: str
    embedding_model: str
    embedding_representation_layer: int
    embedding_dimension: int
    embedding_dtype: str
    embedding_records: int
    expected_examples: int
    expected_source_observations: int
    expected_sequences: int
    expected_positive_examples: int
    expected_negative_examples: int
    expected_homology_components: int
    expected_union_components: int
    expected_examples_by_fold: tuple[int, ...]
    expected_positive_examples_by_fold: tuple[int, ...]
    expected_negative_examples_by_fold: tuple[int, ...]
    expected_sequence_ids_sha256: str
    expected_outer_folds: int
    expected_ordered_outer_inner_splits: int
    expected_nested_feature_rows: int
    expected_outer_esm_fits: int
    expected_nested_esm_fits: int
    expected_outer_stack_fits: int
    expected_largest_union_component_contexts: int
    expected_union_components_over_100_contexts: int
    promotion_auc_delta_lower_minimum: float
    promotion_brier_delta_upper_maximum: float
    promotion_log_loss_delta_upper_maximum: float
    production_policy: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class TwinEvidence:
    root: Path
    run: Path
    snapshots: Mapping[str, Snapshot]
    handshake: Mapping[str, object]
    handshake_snapshots: Mapping[str, Snapshot]
    all_snapshots: tuple[Snapshot, ...]


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
class EmbeddingTable:
    matrix: NDArray[np.float32]
    row_by_sequence_id: Mapping[str, int]
    tensor_data_sha256: str
    index_sha256: str
    matrix_sha256: str


@dataclass(frozen=True, slots=True)
class Prediction:
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


def _require(condition: object, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)


def _require_no_symlink(path: Path, *, label: str, ancestors: bool = True) -> None:
    candidate = path.absolute()
    targets = (candidate, *candidate.parents) if ancestors else (candidate,)
    for target in targets:
        _require(not target.is_symlink(), f"{label} must not traverse a symbolic link")


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _require_no_symlink(requested, label=label)
    resolved = requested.resolve(strict=True)
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is not a real directory")
    return resolved


def _read_snapshot(path: str | Path, *, label: str) -> Snapshot:
    requested = Path(path)
    _require_no_symlink(requested, label=label)
    resolved = requested.resolve(strict=True)
    before = resolved.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return Snapshot(resolved, payload, _sha256(payload), _fingerprint(after))


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    try:
        current = _read_snapshot(snapshot.path, label=label)
    except FileNotFoundError as error:
        raise VerificationError(f"{label} disappeared during verification") from error
    _require(
        current.sha256 == snapshot.sha256 and current.fingerprint == snapshot.fingerprint,
        f"{label} changed during verification",
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise VerificationError(f"non-finite JSON number: {value}")


def _loads_json(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, VerificationError) as error:
        raise VerificationError(f"{label} is not valid strict UTF-8 JSON") from error


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_object(
    snapshot: Snapshot, *, label: str, canonical_pretty: bool = True
) -> dict[str, object]:
    _require(
        snapshot.payload.endswith(b"\n")
        and not snapshot.payload.endswith(b"\n\n")
        and b"\r" not in snapshot.payload,
        f"{label} must have exactly one final LF",
    )
    value = _loads_json(snapshot.payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain one object")
    result = cast(dict[str, object], value)
    if canonical_pretty:
        _require(snapshot.payload == _pretty_json_bytes(result), f"{label} is noncanonical")
    return result


def _jsonl_objects(snapshot: Snapshot, *, label: str) -> tuple[dict[str, object], ...]:
    payload = snapshot.payload
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} framing")
    output: list[dict[str, object]] = []
    for number, line in enumerate(payload[:-1].split(b"\n"), start=1):
        _require(bool(line), f"{label} line {number} is blank")
        value = _loads_json(line, label=f"{label} line {number}")
        _require(isinstance(value, dict), f"{label} line {number} is not an object")
        row = cast(dict[str, object], value)
        _require(line == _compact_json_bytes(row), f"{label} line {number} is noncanonical")
        output.append(row)
    return tuple(output)


def _exact_fields(value: Mapping[str, object], expected: Iterable[str], *, label: str) -> None:
    expected_set = set(expected)
    _require(
        set(value) == expected_set,
        f"{label} schema mismatch: missing={sorted(expected_set - set(value))}, "
        f"extra={sorted(set(value) - expected_set)}",
    )


def _sha_field(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"bad {label}")
    return cast(str, value)


def _positive_int(value: object, *, label: str) -> int:
    _require(type(value) is int and cast(int, value) > 0, f"{label} must be positive integer")
    return cast(int, value)


def _nonnegative_int(value: object, *, label: str) -> int:
    _require(type(value) is int and cast(int, value) >= 0, f"{label} must be nonnegative integer")
    return cast(int, value)


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    _require(type(value) in {int, float}, f"{label} must be numeric")
    result = float(cast(int | float, value))
    _require(math.isfinite(result) and (not positive or result > 0), f"bad {label}")
    return result


def _safe_manifest_name(value: str, *, label: str) -> str:
    pure = PurePosixPath(value)
    _require(
        bool(value)
        and not pure.is_absolute()
        and pure.as_posix() == value
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{label} contains unsafe path {value!r}",
    )
    return value


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} framing")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    output: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        _require(match is not None, f"{label} line {number} is malformed")
        assert match is not None
        digest, mode, raw_name = match.groups()
        name = _safe_manifest_name(raw_name, label=label)
        _require(mode == " " and name not in output, f"{label} line {number} is invalid")
        _require(previous is None or name > previous, f"{label} paths are not sorted")
        output[name] = digest
        previous = name
    return output


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("utf-8")


def _strict_csv(payload: bytes, *, schema: Sequence[str], label: str) -> list[dict[str, str]]:
    _require(payload.endswith(b"\n") and b"\r" not in payload, f"{label} framing")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require(reader.fieldnames == list(schema), f"{label} schema differs")
    rows = list(reader)
    _require(
        all(None not in row and None not in row.values() for row in rows), f"{label} row width"
    )
    return cast(list[dict[str, str]], rows)


def _tree_inventory(root: Path) -> frozenset[str]:
    output: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"tree contains symbolic link: {path}")
        if path.is_dir():
            continue
        _require(path.is_file(), f"tree contains a non-regular entry: {path}")
        output.add(path.relative_to(root).as_posix())
    return frozenset(output)


def _directory_inventory(root: Path) -> frozenset[str]:
    output: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"tree contains symbolic link: {path}")
        if path.is_dir():
            output.add(path.relative_to(root).as_posix())
    return frozenset(output)


def _expected_directories(files: Iterable[str]) -> frozenset[str]:
    return frozenset(
        parent.as_posix()
        for name in files
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    )


def _verify_immutable_tree(
    root: Path, *, label: str, required_checksum_marker_mode: int | None = None
) -> None:
    for path in (root, *root.rglob("*")):
        _require(not path.is_symlink(), f"{label} contains a symbolic link")
        mode = path.stat().st_mode
        _require(mode & 0o222 == 0, f"{label} remains writable: {path.name}")
        if (
            required_checksum_marker_mode is not None
            and path.is_file()
            and path.name == "SHA256SUMS"
        ):
            _require(
                stat.S_IMODE(mode) == required_checksum_marker_mode,
                f"{label} has an uncommitted checksum marker: {path}",
            )


def _snapshot_manifest_tree(
    root: Path,
    *,
    expected_inventory: frozenset[str],
    expected_top_sha256: str | None,
    label: str,
    required_top_marker_mode: int | None = None,
) -> dict[str, Snapshot]:
    _require(_tree_inventory(root) == expected_inventory, f"{label} inventory differs")
    _require(
        _directory_inventory(root) == _expected_directories(expected_inventory),
        f"{label} directory inventory differs",
    )
    snapshots = {
        name: _read_snapshot(root / name, label=f"{label} {name}")
        for name in sorted(expected_inventory)
    }
    top = snapshots["SHA256SUMS"]
    if required_top_marker_mode is not None:
        _require(
            stat.S_IMODE(top.path.stat().st_mode) == required_top_marker_mode,
            f"{label} top checksum marker is not committed",
        )
    if expected_top_sha256 is not None:
        _require(top.sha256 == expected_top_sha256, f"{label} top hash differs")
    entries = _parse_sha_manifest(top.payload, label=f"{label} top manifest")
    _require(set(entries) == set(expected_inventory) - {"SHA256SUMS"}, f"{label} coverage differs")
    for name, digest in entries.items():
        _require(snapshots[name].sha256 == digest, f"{label} checksum differs for {name}")
    return snapshots


def _verify_tree_bytes(left: Path, right: Path, *, expected: frozenset[str]) -> None:
    _require(_tree_inventory(left) == expected, "left twin inventory differs")
    _require(_tree_inventory(right) == expected, "right twin inventory differs")
    expected_directories = _expected_directories(expected)
    _require(
        _directory_inventory(left) == expected_directories,
        "left twin directory inventory differs",
    )
    _require(
        _directory_inventory(right) == expected_directories,
        "right twin directory inventory differs",
    )
    for name in sorted(expected):
        _require(
            _read_snapshot(left / name, label=f"left twin {name}").payload
            == _read_snapshot(right / name, label=f"right twin {name}").payload,
            f"twin bytes differ for {name}",
        )


def _verify_complete_twin_tree(
    root: Path,
    *,
    expected: frozenset[str],
    label: str,
    required_checksum_marker_mode: int | None = None,
) -> None:
    expected_files = frozenset(
        {
            *(f"0/{name}" for name in expected),
            *(f"1/{name}" for name in expected),
            *(f"node-receipts/{name}" for name in _HANDSHAKE_FILES),
        }
    )
    _require(_tree_inventory(root) == expected_files, f"{label} complete inventory differs")
    _require(
        _directory_inventory(root) == _expected_directories(expected_files),
        f"{label} complete directory inventory differs",
    )
    for task in ("0", "1"):
        _verify_immutable_tree(
            root / task,
            label=f"{label} twin {task}",
            required_checksum_marker_mode=required_checksum_marker_mode,
        )
    _verify_immutable_tree(root / "node-receipts", label=f"{label} handshake")


def _verify_handshake(
    twin_root: Path, *, expected_job_id: int | None = None
) -> tuple[dict[str, object], dict[str, Snapshot]]:
    receipt_root = twin_root / "node-receipts"
    _require(receipt_root.is_dir() and not receipt_root.is_symlink(), "node receipts are missing")
    _require(_tree_inventory(receipt_root) == _HANDSHAKE_FILES, "handshake inventory differs")
    _require(not _directory_inventory(receipt_root), "handshake contains an extra directory")
    _verify_immutable_tree(receipt_root, label="handshake")
    job_id = twin_root.name
    _require(job_id.isdigit(), "twin root basename is not a job ID")
    if expected_job_id is not None:
        _require(job_id == str(expected_job_id), "twin root job ID differs from config")
    receipt_payloads: dict[int, bytes] = {}
    nodes: dict[int, str] = {}
    snapshots: dict[str, Snapshot] = {}
    for task in (0, 1):
        item = _read_snapshot(receipt_root / f"{task}.receipt", label=f"receipt {task}")
        snapshots[f"{task}.receipt"] = item
        try:
            lines = item.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise VerificationError("node receipt is not UTF-8") from error
        _require(
            item.payload.endswith(b"\n")
            and b"\r" not in item.payload
            and len(lines) == 3
            and lines[0] == f"array_job_id={job_id}"
            and lines[1] == f"array_task_id={task}"
            and lines[2].startswith("node_name="),
            f"receipt {task} is invalid",
        )
        node = lines[2].removeprefix("node_name=")
        _require(_SAFE_NODE_RE.fullmatch(node) is not None, f"receipt {task} node is unsafe")
        receipt_payloads[task] = item.payload
        nodes[task] = node
    _require(nodes[0] != nodes[1], "twins did not execute on distinct nodes")
    receipt_hashes = {str(task): _sha256(payload) for task, payload in receipt_payloads.items()}
    acknowledgement_hashes: dict[str, str] = {}
    for task in (0, 1):
        item = _read_snapshot(receipt_root / f"{task}.ack", label=f"acknowledgement {task}")
        snapshots[f"{task}.ack"] = item
        expected = f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode(
            "ascii"
        )
        _require(item.payload == expected, f"acknowledgement {task} does not cross-bind sibling")
        acknowledgement_hashes[str(task)] = item.sha256
    return (
        {
            "distinct_nodes": True,
            "bidirectional_acknowledgement": True,
            "receipt_sha256": receipt_hashes,
            "acknowledgement_sha256": acknowledgement_hashes,
        },
        snapshots,
    )


def _read_twins(
    root_value: str | Path,
    *,
    expected_inventory: frozenset[str],
    expected_top_sha256: str | None,
    expected_job_id: int | None,
    label: str,
    required_checksum_marker_mode: int | None = None,
) -> TwinEvidence:
    root = _resolved_directory(root_value, label=f"{label} twin root")
    _verify_complete_twin_tree(
        root,
        expected=expected_inventory,
        label=label,
        required_checksum_marker_mode=required_checksum_marker_mode,
    )
    runs = (root / "0", root / "1")
    _verify_tree_bytes(runs[0], runs[1], expected=expected_inventory)
    left = _snapshot_manifest_tree(
        runs[0],
        expected_inventory=expected_inventory,
        expected_top_sha256=expected_top_sha256,
        label=f"{label} twin 0",
        required_top_marker_mode=required_checksum_marker_mode,
    )
    right = _snapshot_manifest_tree(
        runs[1],
        expected_inventory=expected_inventory,
        expected_top_sha256=expected_top_sha256,
        label=f"{label} twin 1",
        required_top_marker_mode=required_checksum_marker_mode,
    )
    for name in sorted(expected_inventory):
        _require(
            left[name].payload == right[name].payload,
            f"{label} captured twin bytes differ for {name}",
        )
    handshake, handshake_snapshots = _verify_handshake(root, expected_job_id=expected_job_id)
    return TwinEvidence(
        root,
        runs[0],
        left,
        handshake,
        handshake_snapshots,
        (*left.values(), *right.values(), *handshake_snapshots.values()),
    )


def _git(repository: Path, arguments: Sequence[str], *, label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments], check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError as error:
        raise VerificationError(f"Git check failed for {label}") from error


def _verify_repository(repository: Path, expected_commit: str) -> None:
    _require(_GIT_RE.fullmatch(expected_commit) is not None, "expected Git commit is invalid")
    head = _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"], label="HEAD")
    _require(head.decode().strip() == expected_commit, "HEAD differs from expected commit")
    _require(
        _git(
            repository, ["diff", "--no-ext-diff", "--quiet", "--exit-code", "--"], label="worktree"
        )
        == b"",
        "repository tracked worktree is dirty",
    )
    _require(
        _git(
            repository,
            ["diff", "--cached", "--no-ext-diff", "--quiet", "--exit-code", "--"],
            label="index",
        )
        == b"",
        "repository index is dirty",
    )
    _require(
        _git(repository, ["ls-files", "--others", "--exclude-standard", "-z"], label="untracked")
        == b"",
        "repository contains untracked files",
    )
    origin = _git(
        repository,
        ["rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
        label="origin/main",
    )
    _require(
        origin.decode().strip() == expected_commit, "commit is not synchronized to origin/main"
    )


def _committed_blob(repository: Path, commit: str, logical_path: str) -> bytes:
    kind = _git(repository, ["cat-file", "-t", f"{commit}:{logical_path}"], label=logical_path)
    _require(kind == b"blob\n", f"committed path is not a blob: {logical_path}")
    return _git(repository, ["cat-file", "blob", f"{commit}:{logical_path}"], label=logical_path)


def _verify_execution_source(repository: Path, expected_commit: str) -> str:
    executing = Path(__file__).resolve(strict=True)
    expected = (repository / _VERIFIER_MODULE_PATH).resolve(strict=True)
    _require(executing == expected, "executing verifier is a stale installation")
    snapshot = _read_snapshot(expected, label="verifier source")
    _require(
        snapshot.payload == _committed_blob(repository, expected_commit, _VERIFIER_MODULE_PATH),
        "executing verifier differs from committed blob",
    )
    return snapshot.sha256


def _int_tuple(value: object, *, label: str, length: int) -> tuple[int, ...]:
    _require(isinstance(value, list) and len(value) == length, f"{label} has wrong length")
    return tuple(_positive_int(item, label=label) for item in cast(list[object], value))


def load_config(path: str | Path) -> Config:
    """Load the exact frozen nested-union contract."""

    snapshot = _read_snapshot(path, label="ESM union ensemble config")
    _require(snapshot.sha256 == _CONFIG_SHA256, "ESM union ensemble config hash differs")
    _require(
        snapshot.payload.endswith(b"\n")
        and not snapshot.payload.endswith(b"\n\n")
        and b"\r" not in snapshot.payload,
        "ESM union ensemble config framing differs",
    )
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("ESM union ensemble config is invalid TOML") from error
    _require(
        raw.get("schema_version") == 1
        and raw.get("artifact") == _ARTIFACT
        and tuple(cast(list[str], raw.get("methods", []))) == _METHODS
        and raw.get("promotion_reference") == _PROMOTION_REFERENCE
        and raw.get("promotion_candidate") == _PROMOTION_CANDIDATE
        and tuple(cast(list[str], raw.get("stack_features", []))) == _STACK_FEATURES,
        "ESM union ensemble config identity differs",
    )
    literals = {
        "stack_input_transform": "clipped_logit",
        "stack_scaling": "column_mean_std_on_nested_outer_training_features_only",
        "descriptor_esm_probability_half_blend": (
            "0.5*descriptor_logistic+0.5*esm2_t6_8m_target_gram_logistic"
        ),
        "base_family_esm_probability_half_blend": (
            "0.5*equal_weight_ensemble+0.5*esm2_t6_8m_target_gram_logistic"
        ),
        "esm_context_features": "fixed_canonical_target_one_hot_plus_gram_one_hot",
        "esm_feature_order": (
            "intercept_then_320_standardized_esm_then_targets_in_config_order_then_gram_classes_in_config_order"
        ),
        "one_hot_convention": "full_one_hot_columns",
        "ridge_penalty": "intercept_unpenalized_all_other_coefficients_equal_l2",
        "fit_weighting": "context_equal_v1",
        "scaler_weighting": "context_equal_v1",
        "stack_scaler_weighting": "context_equal_v1",
        "unseen_target_policy": (
            "known_target_column_coefficient_remains_zero_when_absent_from_training"
        ),
        "inner_base_feature_source": (
            "refit_on_folds_excluding_outer_and_inner_never_accepted_oof"
        ),
        "outer_base_feature_source": ("accepted_outer_oof_after_exact_refit_reproduction_check"),
        "bootstrap_unit": "union_component_resample_context_multiplicity_v1",
    }
    _require(
        all(raw.get(key) == value for key, value in literals.items()), "policy literal differs"
    )
    _require(
        raw.get("source_observations_used_as_weight") is False
        and raw.get("hyperparameter_search_allowed") is False
        and raw.get("similarity_gate_allowed") is False
        and raw.get("report_component_equal_metrics") is True
        and raw.get("report_leave_largest_union_component_out") is True,
        "frozen boolean policy differs",
    )
    production = raw.get("production_policy")
    _require(production == _PRODUCTION_POLICY, "production policy differs")
    targets = tuple(cast(list[str], raw.get("canonical_targets", [])))
    grams = tuple(cast(list[str], raw.get("gram_classes", [])))
    _require(len(targets) == 7 and len(set(targets)) == 7, "canonical targets differ")
    _require(grams == ("negative", "positive"), "Gram order differs")
    hash_names = (
        "base_publication_top_sha256",
        "base_semantic_top_sha256",
        "base_examples_sha256",
        "base_folds_sha256",
        "base_oof_sha256",
        "base_manifest_sha256",
        "base_split_receipt_sha256",
        "base_independent_receipt_sha256",
        "base_config_sha256",
        "embedding_publication_top_sha256",
        "embedding_semantic_top_sha256",
        "embedding_index_sha256",
        "embedding_manifest_sha256",
        "embedding_matrix_sha256",
        "embedding_tensor_data_sha256",
        "embedding_independent_receipt_sha256",
        "embedding_config_sha256",
        "expected_sequence_ids_sha256",
    )
    hashes = {name: _sha_field(raw.get(name), label=name) for name in hash_names}
    commit = raw.get("embedding_protocol_git_commit")
    _require(
        isinstance(commit, str) and _GIT_RE.fullmatch(commit) is not None,
        "embedding protocol commit is invalid",
    )
    folds = _positive_int(raw.get("folds"), label="folds")
    _require(folds == 5, "v1 requires five folds")
    examples = _positive_int(raw.get("expected_examples"), label="expected examples")
    positives = _positive_int(raw.get("expected_positive_examples"), label="expected positives")
    negatives = _positive_int(raw.get("expected_negative_examples"), label="expected negatives")
    by_fold = _int_tuple(
        raw.get("expected_examples_by_fold"), label="examples by fold", length=folds
    )
    pos_by_fold = _int_tuple(
        raw.get("expected_positive_examples_by_fold"), label="positives by fold", length=folds
    )
    neg_by_fold = _int_tuple(
        raw.get("expected_negative_examples_by_fold"), label="negatives by fold", length=folds
    )
    _require(
        positives + negatives == examples
        and sum(by_fold) == examples
        and sum(pos_by_fold) == positives
        and sum(neg_by_fold) == negatives
        and all(
            p + n == total for p, n, total in zip(pos_by_fold, neg_by_fold, by_fold, strict=True)
        ),
        "config census arithmetic differs",
    )
    edges_raw = raw.get("similarity_bin_edges")
    _require(isinstance(edges_raw, list), "similarity bins are not an array")
    edges = tuple(_finite_float(item, label="similarity edge") for item in edges_raw)
    _require(
        edges == (0.0, 0.4, 0.6, 0.8),
        "similarity strata differ from the frozen protocol",
    )
    clip = _finite_float(raw.get("probability_clip"), label="probability clip", positive=True)
    _require(clip == 1e-6, "probability clip differs")
    return Config(
        path=snapshot.path,
        sha256=snapshot.sha256,
        methods=_METHODS,
        probability_clip=clip,
        canonical_targets=targets,
        gram_classes=grams,
        esm_l2=_finite_float(raw.get("esm_l2"), label="ESM l2", positive=True),
        esm_prior_strength=_finite_float(raw.get("esm_prior_strength"), label="ESM prior"),
        esm_max_iterations=_positive_int(raw.get("esm_max_iterations"), label="ESM iterations"),
        esm_tolerance=_finite_float(raw.get("esm_tolerance"), label="ESM tolerance", positive=True),
        stack_l2=_finite_float(raw.get("stack_l2"), label="stack l2", positive=True),
        stack_prior_strength=_finite_float(raw.get("stack_prior_strength"), label="stack prior"),
        stack_max_iterations=_positive_int(
            raw.get("stack_max_iterations"), label="stack iterations"
        ),
        stack_tolerance=_finite_float(
            raw.get("stack_tolerance"), label="stack tolerance", positive=True
        ),
        folds=folds,
        homology_identity_threshold=_finite_float(
            raw.get("homology_identity_threshold"), label="identity threshold", positive=True
        ),
        similarity_bin_edges=edges,
        calibration_bins=_positive_int(raw.get("calibration_bins"), label="calibration bins"),
        bootstrap_replicates=_positive_int(
            raw.get("bootstrap_replicates"), label="bootstrap replicates"
        ),
        bootstrap_seed=_nonnegative_int(raw.get("bootstrap_seed"), label="bootstrap seed"),
        base_producer_job_id=_positive_int(raw.get("base_producer_job_id"), label="base job"),
        base_audit_job_id=_positive_int(raw.get("base_audit_job_id"), label="base audit job"),
        embedding_protocol_git_commit=cast(str, commit),
        embedding_producer_job_id=_positive_int(
            raw.get("embedding_producer_job_id"), label="embedding job"
        ),
        embedding_audit_job_id=_positive_int(
            raw.get("embedding_audit_job_id"), label="embedding audit job"
        ),
        embedding_model=cast(str, raw.get("embedding_model")),
        embedding_representation_layer=_positive_int(
            raw.get("embedding_representation_layer"), label="representation layer"
        ),
        embedding_dimension=_positive_int(
            raw.get("embedding_dimension"), label="embedding dimension"
        ),
        embedding_dtype=cast(str, raw.get("embedding_dtype")),
        embedding_records=_positive_int(raw.get("embedding_records"), label="embedding records"),
        expected_examples=examples,
        expected_source_observations=_positive_int(
            raw.get("expected_source_observations"), label="source observations"
        ),
        expected_sequences=_positive_int(raw.get("expected_sequences"), label="sequences"),
        expected_positive_examples=positives,
        expected_negative_examples=negatives,
        expected_homology_components=_positive_int(
            raw.get("expected_homology_components"), label="homology components"
        ),
        expected_union_components=_positive_int(
            raw.get("expected_union_components"), label="union components"
        ),
        expected_examples_by_fold=by_fold,
        expected_positive_examples_by_fold=pos_by_fold,
        expected_negative_examples_by_fold=neg_by_fold,
        expected_outer_folds=_positive_int(raw.get("expected_outer_folds"), label="outer folds"),
        expected_ordered_outer_inner_splits=_positive_int(
            raw.get("expected_ordered_outer_inner_splits"), label="nested splits"
        ),
        expected_nested_feature_rows=_positive_int(
            raw.get("expected_nested_feature_rows"), label="nested rows"
        ),
        expected_outer_esm_fits=_positive_int(
            raw.get("expected_outer_esm_fits"), label="outer ESM fits"
        ),
        expected_nested_esm_fits=_positive_int(
            raw.get("expected_nested_esm_fits"), label="nested ESM fits"
        ),
        expected_outer_stack_fits=_positive_int(
            raw.get("expected_outer_stack_fits"), label="outer stack fits"
        ),
        expected_largest_union_component_contexts=_positive_int(
            raw.get("expected_largest_union_component_contexts"), label="largest component"
        ),
        expected_union_components_over_100_contexts=_positive_int(
            raw.get("expected_union_components_over_100_contexts"), label="large components"
        ),
        promotion_auc_delta_lower_minimum=_finite_float(
            raw.get("promotion_auc_delta_lower_minimum"), label="AUC promotion threshold"
        ),
        promotion_brier_delta_upper_maximum=_finite_float(
            raw.get("promotion_brier_delta_upper_maximum"), label="Brier promotion threshold"
        ),
        promotion_log_loss_delta_upper_maximum=_finite_float(
            raw.get("promotion_log_loss_delta_upper_maximum"), label="log-loss promotion threshold"
        ),
        production_policy=cast(Mapping[str, object], production),
        **hashes,
    )


def _load_base_config(snapshot: Snapshot, *, expected_sha256: str) -> BaseConfig:
    _require(snapshot.sha256 == expected_sha256, "accepted Gate-1 config hash differs")
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("accepted Gate-1 config is invalid TOML") from error
    logistic = raw.get("descriptor_logistic")
    knn = raw.get("homology_knn")
    _require(isinstance(logistic, dict) and isinstance(knn, dict), "base model tables are missing")
    _exact_fields(
        cast(dict[str, object], logistic),
        {"l2", "max_iterations", "tolerance", "prior_strength"},
        label="descriptor settings",
    )
    _exact_fields(
        cast(dict[str, object], knn),
        {"neighbors", "similarity_power", "prior_strength", "minimum_weight"},
        label="kNN settings",
    )
    return BaseConfig(
        logistic=LogisticSettings(
            l2=_finite_float(logistic["l2"], label="descriptor l2", positive=True),
            max_iterations=_positive_int(logistic["max_iterations"], label="descriptor iterations"),
            tolerance=_finite_float(
                logistic["tolerance"], label="descriptor tolerance", positive=True
            ),
            prior_strength=_finite_float(logistic["prior_strength"], label="descriptor prior"),
        ),
        knn=KnnSettings(
            neighbors=_positive_int(knn["neighbors"], label="kNN neighbors"),
            similarity_power=_finite_float(
                knn["similarity_power"], label="kNN power", positive=True
            ),
            prior_strength=_finite_float(knn["prior_strength"], label="kNN prior"),
            minimum_weight=_finite_float(knn["minimum_weight"], label="kNN floor", positive=True),
        ),
    )


def _verify_code_manifest(
    snapshot: Snapshot,
    *,
    repository: Path,
    expected_commit: str,
    config: Config,
    base_config: Snapshot,
    embedding_config: Snapshot,
) -> dict[str, object]:
    _require(snapshot.path.stat().st_mode & 0o222 == 0, "producer code manifest remains writable")
    entries = _parse_sha_manifest(snapshot.payload, label="producer code manifest")
    source_root = repository / "src" / "amp_challenge"
    source_paths: set[str] = set()
    for path in source_root.rglob("*.py"):
        _require(not path.is_symlink(), f"source entry is symbolic: {path}")
        if path.is_file():
            source_paths.add(path.relative_to(repository).as_posix())
    expected = source_paths | set(_FIXED_CODE_PATHS)
    _require(set(entries) == expected, "producer code-manifest inventory differs")
    _require(
        _PRODUCER_MODULE_PATH in entries and _VERIFIER_MODULE_PATH in entries,
        "producer code manifest omits producer or verifier",
    )
    for logical in sorted(expected):
        source = _read_snapshot(repository / logical, label=f"repository code {logical}")
        _require(source.sha256 == entries[logical], f"code-manifest checksum differs for {logical}")
        _require(
            source.payload == _committed_blob(repository, expected_commit, logical),
            f"code entry differs from committed blob: {logical}",
        )
    _require(
        entries[_CONFIG_PATH] == config.sha256
        and entries[_BASE_CONFIG_PATH] == base_config.sha256
        and entries[_EMBEDDING_CONFIG_PATH] == embedding_config.sha256,
        "producer code manifest does not bind supplied configs",
    )
    return {
        "schema_version": 1,
        "code_manifest_sha256": snapshot.sha256,
        "inventory_entries": len(entries),
        "executing_module": {
            "logical_path": _PRODUCER_MODULE_PATH,
            "sha256": entries[_PRODUCER_MODULE_PATH],
        },
        "config": {"logical_path": _CONFIG_PATH, "sha256": config.sha256},
        "base_config": {"logical_path": _BASE_CONFIG_PATH, "sha256": base_config.sha256},
        "embedding_config": {
            "logical_path": _EMBEDDING_CONFIG_PATH,
            "sha256": embedding_config.sha256,
        },
    }


def _sequence_set_sha256(values: Iterable[str]) -> str:
    items = sorted(set(values))
    payload = b"" if not items else ("\n".join(items) + "\n").encode("ascii")
    return _sha256(payload)


def _canonical_sequence(value: object, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} is not a string")
    sequence = "".join(cast(str, value).split()).upper()
    _require(sequence == value, f"{label} is not canonical")
    _require(_MIN_LENGTH <= len(sequence) <= _MAX_LENGTH, f"{label} length is invalid")
    _require(set(sequence) <= _AMINO_ACID_SET, f"{label} contains a nonstandard residue")
    return sequence


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
        for item in sorted(examples, key=lambda item: item.example_id)
    ]
    return _sha256(("\n".join(records) + "\n").encode("utf-8"))


def _fold_summary(examples: Sequence[Example], folds: int) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for fold in range(folds):
        rows = [item for item in examples if item.fold == fold]
        output[str(fold)] = {
            "examples": len(rows),
            "gram_negative_mic16_negative": sum(
                item.gram == "negative" and item.label == 0 for item in rows
            ),
            "gram_negative_mic16_positive": sum(
                item.gram == "negative" and item.label == 1 for item in rows
            ),
            "gram_positive_mic16_negative": sum(
                item.gram == "positive" and item.label == 0 for item in rows
            ),
            "gram_positive_mic16_positive": sum(
                item.gram == "positive" and item.label == 1 for item in rows
            ),
            "homology_components": len({item.homology_component_id for item in rows}),
            "negatives": sum(item.label == 0 for item in rows),
            "positives": sum(item.label == 1 for item in rows),
            "sequences": len({item.sequence_id for item in rows}),
            "source_observations": sum(item.source_observations for item in rows),
            "union_components": len({item.union_component_id for item in rows}),
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


def _verify_semantic_tree(
    snapshots: Mapping[str, Snapshot],
    *,
    prefix: str,
    semantic_names: frozenset[str],
    expected_top_sha256: str,
    label: str,
    required_marker_mode: int | None = None,
) -> None:
    semantic_top = snapshots[f"{prefix}/SHA256SUMS"]
    if required_marker_mode is not None:
        _require(
            stat.S_IMODE(semantic_top.path.stat().st_mode) == required_marker_mode,
            f"{label} semantic checksum marker is not committed",
        )
    _require(semantic_top.sha256 == expected_top_sha256, f"{label} semantic top differs")
    entries = _parse_sha_manifest(semantic_top.payload, label=f"{label} semantic manifest")
    _require(set(entries) == semantic_names, f"{label} semantic inventory differs")
    outer = _parse_sha_manifest(snapshots["SHA256SUMS"].payload, label=f"{label} top manifest")
    for name, digest in entries.items():
        _require(
            snapshots[f"{prefix}/{name}"].sha256 == digest and outer[f"{prefix}/{name}"] == digest,
            f"{label} semantic checksum differs for {name}",
        )
    _require(
        outer[f"{prefix}/SHA256SUMS"] == semantic_top.sha256,
        f"{label} top does not bind semantic manifest",
    )


def _read_examples(
    examples_snapshot: Snapshot,
    oof_snapshot: Snapshot,
    *,
    config: Config,
) -> tuple[tuple[Example, ...], dict[str, dict[str, float]]]:
    documents = _jsonl_objects(examples_snapshot, label="accepted Gate-1 examples")
    _require(len(documents) == config.expected_examples, "accepted example census differs")
    raw_by_id: dict[str, dict[str, object]] = {}
    previous: str | None = None
    for number, row in enumerate(documents, start=1):
        _exact_fields(row, _EXAMPLE_FIELDS, label=f"accepted example {number}")
        _require(row["schema_version"] == 1 and type(row["schema_version"]) is int, "bad schema")
        example_id = _sha_field(row["example_id"], label=f"example {number} ID")
        assay_id = _sha_field(row["assay_context_id"], label=f"example {number} context ID")
        _require(example_id == assay_id and example_id not in raw_by_id, "example identity differs")
        _require(previous is None or example_id > previous, "examples are not strictly sorted")
        sequence = _canonical_sequence(row["sequence"], label=f"example {number} sequence")
        sequence_id = _sha_field(row["sequence_id"], label=f"example {number} sequence ID")
        _require(_sha256(sequence.encode("ascii")) == sequence_id, "sequence ID differs")
        _require(row["canonical_target"] in config.canonical_targets, "unknown canonical target")
        _require(row["gram"] in config.gram_classes, "unknown Gram class")
        _require(type(row["label"]) is int and row["label"] in {0, 1}, "invalid label")
        _require(type(row["fold"]) is int and row["fold"] in range(config.folds), "invalid fold")
        _positive_int(row["source_observations"], label="source observations")
        _sha_field(row["homology_component_id"], label="homology component")
        _sha_field(row["union_component_id"], label="union component")
        raw_by_id[example_id] = row
        previous = example_id

    csv_rows = _strict_csv(
        oof_snapshot.payload, schema=_BASE_OOF_SCHEMA, label="accepted Gate-1 OOF"
    )
    accepted: dict[str, dict[str, float]] = {name: {} for name in _BASE_MODELS}
    identities: dict[str, float] = {}
    model_order = {name: index for index, name in enumerate(sorted(_BASE_MODELS))}
    previous_key: tuple[int, str] | None = None
    for number, row in enumerate(csv_rows, start=2):
        model = row["model"]
        example_id = row["example_id"]
        _require(model in accepted and example_id in raw_by_id, f"OOF row {number} support differs")
        key = (model_order[model], example_id)
        _require(
            previous_key is None or key > previous_key, "accepted OOF is not canonically sorted"
        )
        previous_key = key
        source = raw_by_id[example_id]
        for field in (
            "assay_context_id",
            "sequence_id",
            "sequence",
            "canonical_target",
            "gram",
            "homology_component_id",
            "union_component_id",
        ):
            _require(row[field] == source[field], f"OOF row {number} metadata differs")
        try:
            label_value = int(row["label"])
            source_count = int(row["source_observations"])
            fold_value = int(row["fold"])
            identity = float(row["max_train_identity"])
            probability = float(row["probability"])
        except ValueError as error:
            raise VerificationError(f"OOF row {number} has nonnumeric metadata") from error
        _require(
            str(label_value) == row["label"]
            and str(source_count) == row["source_observations"]
            and str(fold_value) == row["fold"]
            and label_value == source["label"]
            and source_count == source["source_observations"]
            and fold_value == source["fold"],
            f"OOF row {number} numeric metadata differs",
        )
        _require(
            math.isfinite(identity)
            and 0 <= identity < config.homology_identity_threshold
            and math.isfinite(probability)
            and 0 <= probability <= 1,
            f"OOF row {number} probability/identity is invalid",
        )
        _require(example_id not in accepted[model], f"OOF row {number} is duplicate")
        accepted[model][example_id] = probability
        prior_identity = identities.setdefault(example_id, identity)
        _require(prior_identity == identity, "base models disagree on identity metadata")
    expected_ids = set(raw_by_id)
    _require(
        all(set(probabilities) == expected_ids for probabilities in accepted.values()),
        "base OOF support differs",
    )
    for example_id in expected_ids:
        equal = float(
            np.mean(
                np.asarray(
                    [
                        accepted["descriptor_logistic"][example_id],
                        accepted["homology_knn"][example_id],
                    ],
                    dtype=np.float64,
                )
            )
        )
        _require(
            accepted["equal_weight_ensemble"][example_id] == equal,
            "accepted equal-weight ensemble is not exact",
        )
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
    positives = sum(item.label for item in examples)
    _require(
        len(examples) == config.expected_examples
        and positives == config.expected_positive_examples
        and len(examples) - positives == config.expected_negative_examples
        and sum(item.source_observations for item in examples)
        == config.expected_source_observations
        and len({item.sequence_id for item in examples}) == config.expected_sequences
        and len({item.homology_component_id for item in examples})
        == config.expected_homology_components
        and len({item.union_component_id for item in examples}) == config.expected_union_components
        and _sequence_set_sha256(item.sequence_id for item in examples)
        == config.expected_sequence_ids_sha256,
        "accepted modeled census differs",
    )
    summary = _fold_summary(examples, config.folds)
    _require(
        tuple(summary[str(fold)]["examples"] for fold in range(config.folds))
        == config.expected_examples_by_fold
        and tuple(summary[str(fold)]["positives"] for fold in range(config.folds))
        == config.expected_positive_examples_by_fold
        and tuple(summary[str(fold)]["negatives"] for fold in range(config.folds))
        == config.expected_negative_examples_by_fold,
        "accepted fold census differs",
    )
    sequence_assignments: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    homology_locations: dict[str, set[tuple[int, str]]] = defaultdict(set)
    union_folds: dict[str, set[int]] = defaultdict(set)
    for item in examples:
        sequence_assignments[item.sequence_id].add(
            (item.sequence, item.fold, item.homology_component_id, item.union_component_id)
        )
        homology_locations[item.homology_component_id].add((item.fold, item.union_component_id))
        union_folds[item.union_component_id].add(item.fold)
    _require(
        all(len(value) == 1 for value in sequence_assignments.values())
        and all(len(value) == 1 for value in homology_locations.values())
        and all(len(value) == 1 for value in union_folds.values()),
        "accepted sequence/component assignments leak across folds",
    )
    return examples, accepted


def _verify_folds(
    snapshot: Snapshot, *, examples: Sequence[Example], config: Config
) -> dict[str, object]:
    document = _json_object(snapshot, label="accepted Gate-1 folds")
    _exact_fields(
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
        label="accepted Gate-1 folds",
    )
    _require(
        document["schema_version"] == 1
        and document["artifact"] == "gate1_context_union_fold_reuse"
        and document["assignment_policy"]
        == "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
        and document["identity_threshold"] == config.homology_identity_threshold
        and document["maximum_cross_fold_identity"]
        == max(item.max_train_identity for item in examples),
        "accepted folds identity/policy differs",
    )
    expected = [
        {
            "example_id": item.example_id,
            "fold": item.fold,
            "homology_component_id": item.homology_component_id,
            "sequence_id": item.sequence_id,
            "union_component_id": item.union_component_id,
        }
        for item in examples
    ]
    _require(document["assignments"] == expected, "accepted fold assignments differ")
    _require(
        document["folds"] == _fold_summary(examples, config.folds)
        and document["canonical_targets_by_fold"] == _target_summary(examples, config.folds),
        "accepted fold summaries differ",
    )
    return document


def _verify_base_chain(
    twins: TwinEvidence,
    receipt_snapshot: Snapshot,
    *,
    config: Config,
) -> tuple[tuple[Example, ...], dict[str, dict[str, float]], dict[str, object], dict[str, object]]:
    snapshots = twins.snapshots
    _verify_semantic_tree(
        snapshots,
        prefix="gate1",
        semantic_names=_BASE_SEMANTIC_FILES,
        expected_top_sha256=config.base_semantic_top_sha256,
        label="accepted Gate-1",
    )
    semantic_hashes = _parse_sha_manifest(
        snapshots["gate1/SHA256SUMS"].payload, label="accepted Gate-1 semantic manifest"
    )
    for name, digest in (
        ("gate1/examples.jsonl", config.base_examples_sha256),
        ("gate1/folds.json", config.base_folds_sha256),
        ("gate1/oof_predictions.csv", config.base_oof_sha256),
        ("gate1/manifest.json", config.base_manifest_sha256),
        ("gate1/split_receipt.json", config.base_split_receipt_sha256),
    ):
        _require(snapshots[name].sha256 == digest, f"accepted {name} hash differs")
    _require(
        receipt_snapshot.sha256 == config.base_independent_receipt_sha256
        and receipt_snapshot.path.stat().st_mode & 0o222 == 0,
        "accepted Gate-1 receipt hash/mode differs",
    )
    receipt = _json_object(receipt_snapshot, label="accepted Gate-1 receipt")
    checks = receipt.get("checks")
    artifact_hashes = receipt.get("artifact_sha256")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("artifact")
        == "gate1_context_activity_homology_study_union_v1_independent_verification"
        and receipt.get("status") == "passed"
        and receipt.get("config_sha256") == config.base_config_sha256
        and receipt.get("publication_top_manifest_sha256") == config.base_publication_top_sha256
        and receipt.get("gate1_top_manifest_sha256") == config.base_semantic_top_sha256
        and receipt.get("production_handshake") == twins.handshake
        and receipt.get("code_manifest_sha256") == snapshots["CODE_SHA256SUMS"].sha256
        and receipt.get("frozen_input_manifest_sha256")
        == snapshots["FROZEN_INPUT_SHA256SUMS"].sha256
        and isinstance(checks, dict)
        and set(checks) == _BASE_RECEIPT_CHECKS
        and all(value is True for value in checks.values())
        and artifact_hashes == semantic_hashes,
        "accepted Gate-1 independent receipt differs",
    )
    examples, accepted = _read_examples(
        snapshots["gate1/examples.jsonl"], snapshots["gate1/oof_predictions.csv"], config=config
    )
    _verify_folds(snapshots["gate1/folds.json"], examples=examples, config=config)
    manifest = _json_object(snapshots["gate1/manifest.json"], label="accepted Gate-1 manifest")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact") == "gate1_context_activity_homology_study_union_v1"
        and manifest.get("status") == "development_evidence_not_an_untouched_evaluation_panel"
        and manifest.get("config_sha256") == config.base_config_sha256
        and manifest.get("models") == list(_BASE_MODELS)
        and isinstance(manifest.get("accepted_split"), dict),
        "accepted Gate-1 manifest differs",
    )
    split = _json_object(snapshots["gate1/split_receipt.json"], label="accepted split receipt")
    split_invariants = split.get("invariants")
    split_census = split.get("census")
    _require(
        split.get("schema_version") == 1
        and split.get("artifact") == "gate1_union_accepted_split_consumption_receipt"
        and split.get("status") == "passed"
        and split.get("assignment_policy")
        == "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
        and isinstance(split_invariants, dict)
        and bool(split_invariants)
        and all(value is True for value in split_invariants.values())
        and isinstance(split_census, dict)
        and split_census.get("context_examples") == config.expected_examples
        and split_census.get("source_observations") == config.expected_source_observations
        and split_census.get("modeled_sequences") == config.expected_sequences
        and split_census.get("homology_components") == 597
        and split_census.get("union_components") == 278
        and split_census.get("parser_sequences") == 1113,
        "accepted split receipt contract differs",
    )
    return examples, accepted, manifest, split


def _canonical_npy_bytes(tensor: bytes, *, rows: int, columns: int) -> bytes:
    dictionary = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}".encode(
            "latin1"
        )
    )
    padding = (-((10 + len(dictionary) + 1) % 64)) % 64
    header = dictionary + b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + tensor


def _read_embedding_table(
    index_snapshot: Snapshot,
    matrix_snapshot: Snapshot,
    examples: Sequence[Example],
    *,
    config: Config,
) -> tuple[EmbeddingTable, dict[str, object]]:
    rows = _strict_csv(
        index_snapshot.payload,
        schema=("row_index", "sequence_id", "sequence", "length"),
        label="embedding index",
    )
    _require(len(rows) == config.embedding_records, "embedding index census differs")
    row_by_id: dict[str, int] = {}
    sequences: dict[str, str] = {}
    prior: str | None = None
    for index, row in enumerate(rows):
        sequence_id = _sha_field(row["sequence_id"], label="embedding sequence ID")
        sequence = _canonical_sequence(row["sequence"], label="embedding sequence")
        _require(row["row_index"] == str(index), "embedding row index differs")
        _require(prior is None or sequence_id > prior, "embedding index is not sorted")
        _require(sequence_id not in row_by_id, "embedding index repeats a sequence")
        _require(_sha256(sequence.encode("ascii")) == sequence_id, "embedding sequence ID mismatch")
        _require(row["length"] == str(len(sequence)), "embedding sequence length differs")
        row_by_id[sequence_id] = index
        sequences[sequence_id] = sequence
        prior = sequence_id
    canonical_index = "row_index,sequence_id,sequence,length\n" + "".join(
        f"{index},{row['sequence_id']},{row['sequence']},{row['length']}\n"
        for index, row in enumerate(rows)
    )
    _require(
        index_snapshot.payload == canonical_index.encode("ascii"), "embedding index noncanonical"
    )
    payload = matrix_snapshot.payload
    _require(
        payload.startswith(b"\x93NUMPY\x01\x00") and len(payload) >= 10, "matrix is not NPY v1"
    )
    header_length = struct.unpack("<H", payload[8:10])[0]
    header_end = 10 + header_length
    _require(header_end <= len(payload), "NPY header is truncated")
    try:
        header = ast.literal_eval(payload[10:header_end].decode("latin1").strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise VerificationError("NPY header is invalid") from error
    _require(
        isinstance(header, dict)
        and set(header) == {"descr", "fortran_order", "shape"}
        and header["descr"] == "<f4"
        and header["fortran_order"] is False
        and header["shape"] == (config.embedding_records, config.embedding_dimension),
        "NPY shape/dtype/layout differs",
    )
    tensor = payload[header_end:]
    _require(
        len(tensor) == config.embedding_records * config.embedding_dimension * 4
        and payload
        == _canonical_npy_bytes(
            tensor, rows=config.embedding_records, columns=config.embedding_dimension
        ),
        "NPY byte count or canonical serialization differs",
    )
    tensor_sha = _sha256(tensor)
    _require(tensor_sha == config.embedding_tensor_data_sha256, "embedding tensor hash differs")
    matrix = np.frombuffer(tensor, dtype="<f4").reshape(
        config.embedding_records, config.embedding_dimension
    )
    _require(np.all(np.isfinite(matrix)), "embedding tensor contains non-finite values")
    example_sequences: dict[str, str] = {}
    for item in examples:
        previous_sequence = example_sequences.setdefault(item.sequence_id, item.sequence)
        _require(previous_sequence == item.sequence, "base sequence identity collision")
    _require(set(example_sequences) == set(row_by_id), "base/embedding support differs")
    _require(
        all(example_sequences[key] == sequences[key] for key in row_by_id),
        "base/embedding sequence bytes differ",
    )
    _require(
        _sequence_set_sha256(row_by_id) == config.expected_sequence_ids_sha256,
        "embedding sequence-set digest differs",
    )
    sequence_to_row = "".join(
        f"{sequence_id}\t{row_by_id[sequence_id]}\n" for sequence_id in sorted(row_by_id)
    ).encode("ascii")
    ordered_vectors = np.asarray(
        [matrix[row_by_id[sequence_id]] for sequence_id in sorted(row_by_id)], dtype="<f4"
    )
    coverage = {
        "schema_version": 1,
        "artifact": "esm_union_embedding_coverage_v1",
        "status": "passed",
        "contexts": len(examples),
        "unique_sequences": len(row_by_id),
        "embedding_rows": len(rows),
        "missing_sequence_ids": 0,
        "extra_sequence_ids": 0,
        "sequence_ids_sha256": _sequence_set_sha256(row_by_id),
        "index_sha256": index_snapshot.sha256,
        "matrix_sha256": matrix_snapshot.sha256,
        "tensor_data_sha256": tensor_sha,
        "sequence_to_row_sha256": _sha256(sequence_to_row),
        "sequence_to_row_digest_encoding": (
            "sequence_id and zero-based row_index pairs, tab-separated and ascending by "
            "sequence_id, with a terminal LF"
        ),
        "joined_vectors_sha256": _sha256(ordered_vectors.tobytes(order="C")),
        "joined_vector_digest_encoding": (
            "vectors ordered by ascending sequence_id and encoded as contiguous "
            "little-endian float32 bytes"
        ),
        "join_policy": "exact sequence_id and canonical sequence bytes",
    }
    _require(coverage["joined_vectors_sha256"] == tensor_sha, "joined vectors differ from tensor")
    return (
        EmbeddingTable(
            matrix, row_by_id, tensor_sha, index_snapshot.sha256, matrix_snapshot.sha256
        ),
        coverage,
    )


def _verify_embedding_chain(
    twins: TwinEvidence,
    receipt_snapshot: Snapshot,
    *,
    config: Config,
) -> tuple[dict[str, object], dict[str, object]]:
    snapshots = twins.snapshots
    _verify_semantic_tree(
        snapshots,
        prefix="embeddings",
        semantic_names=frozenset(
            {"embedding_index.csv", "embedding_manifest.json", "embeddings.npy"}
        ),
        expected_top_sha256=config.embedding_semantic_top_sha256,
        label="accepted embeddings",
    )
    for name, digest in (
        ("embeddings/embedding_index.csv", config.embedding_index_sha256),
        ("embeddings/embedding_manifest.json", config.embedding_manifest_sha256),
        ("embeddings/embeddings.npy", config.embedding_matrix_sha256),
    ):
        _require(snapshots[name].sha256 == digest, f"accepted {name} hash differs")
    _require(
        receipt_snapshot.sha256 == config.embedding_independent_receipt_sha256
        and receipt_snapshot.path.stat().st_mode & 0o222 == 0,
        "accepted embedding receipt hash/mode differs",
    )
    manifest = _json_object(
        snapshots["embeddings/embedding_manifest.json"], label="accepted embedding manifest"
    )
    tensor = manifest.get("tensor")
    extraction = manifest.get("extraction")
    candidate_scope = manifest.get("acceptance_scope")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact") == "gate1_union_esm2_embeddings_v1"
        and manifest.get("status") == "candidate_pending_independent_numerical_verification"
        and manifest.get("production_eligible") is False
        and isinstance(tensor, dict)
        and tensor.get("shape") == [config.embedding_records, config.embedding_dimension]
        and tensor.get("dtype") == config.embedding_dtype
        and tensor.get("tensor_data_sha256") == config.embedding_tensor_data_sha256
        and isinstance(extraction, dict)
        and extraction.get("model") == config.embedding_model
        and extraction.get("representation_layer") == config.embedding_representation_layer
        and extraction.get("records") == config.embedding_records
        and isinstance(candidate_scope, dict)
        and candidate_scope.get("embeddings_verified") is False
        and candidate_scope.get("ensemble_weight_established") is False,
        "accepted embedding candidate manifest differs",
    )
    receipt = _json_object(receipt_snapshot, label="accepted embedding receipt")
    producer = receipt.get("producer")
    receipt_tensor = receipt.get("tensor")
    checks = receipt.get("checks")
    artifact_hashes = receipt.get("artifact_sha256")
    _require(
        set(receipt) == _EMBEDDING_RECEIPT_FIELDS
        and receipt.get("schema_version") == 1
        and receipt.get("artifact") == "gate1_union_esm2_embeddings_v1_independent_verification"
        and receipt.get("status") == "accepted_for_downstream_embedding_feature_input_only"
        and receipt.get("git_commit") == config.embedding_protocol_git_commit
        and receipt.get("config_sha256") == config.embedding_config_sha256
        and receipt.get("acceptance_scope") == _EMBEDDING_ACCEPTANCE_SCOPE
        and isinstance(receipt_tensor, dict)
        and receipt_tensor.get("npy_sha256") == config.embedding_matrix_sha256
        and receipt_tensor.get("tensor_data_sha256") == config.embedding_tensor_data_sha256
        and receipt_tensor.get("shape") == [config.embedding_records, config.embedding_dimension]
        and receipt_tensor.get("dtype") == config.embedding_dtype
        and receipt_tensor.get("byte_order") == "little_endian"
        and receipt_tensor.get("layout") == "C_contiguous"
        and receipt_tensor.get("npy_version") == [1, 0]
        and manifest.get("runtime") == receipt.get("runtime")
        and isinstance(producer, dict)
        and producer.get("publication_top_manifest_sha256")
        == config.embedding_publication_top_sha256
        and producer.get("embedding_top_manifest_sha256") == config.embedding_semantic_top_sha256
        and producer.get("candidate_manifest_sha256") == config.embedding_manifest_sha256
        and producer.get("code_manifest_sha256") == snapshots["CODE_SHA256SUMS"].sha256
        and producer.get("frozen_input_manifest_sha256")
        == snapshots["FROZEN_INPUT_SHA256SUMS"].sha256
        and producer.get("production_handshake") == twins.handshake
        and isinstance(checks, dict)
        and set(checks) == _EMBEDDING_RECEIPT_CHECKS
        and all(value is True for value in checks.values()),
        "accepted embedding independent receipt differs",
    )
    _require(
        isinstance(artifact_hashes, dict)
        and artifact_hashes.get("SHA256SUMS") == config.embedding_publication_top_sha256
        and artifact_hashes.get("embeddings/SHA256SUMS") == config.embedding_semantic_top_sha256
        and artifact_hashes.get("embeddings/embedding_index.csv") == config.embedding_index_sha256
        and artifact_hashes.get("embeddings/embedding_manifest.json")
        == config.embedding_manifest_sha256
        and artifact_hashes.get("embeddings/embeddings.npy") == config.embedding_matrix_sha256,
        "accepted embedding receipt artifact hashes differ",
    )
    return manifest, receipt


def _expected_frozen_input_hashes(
    *,
    base: TwinEvidence,
    base_receipt: Snapshot,
    embedding: TwinEvidence,
    embedding_receipt: Snapshot,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for task in (0, 1):
        for name in sorted(_BASE_TREE_FILES):
            result[f"base/{task}/{name}"] = base.snapshots[name].sha256
        for name in sorted(_EMBEDDING_TREE_FILES):
            result[f"embedding/{task}/{name}"] = embedding.snapshots[name].sha256
    for name in sorted(_HANDSHAKE_FILES):
        result[f"base/node-receipts/{name}"] = base.handshake_snapshots[name].sha256
        result[f"embedding/node-receipts/{name}"] = embedding.handshake_snapshots[name].sha256
    result["base-independent-receipt.json"] = base_receipt.sha256
    result["embedding-independent-receipt.json"] = embedding_receipt.sha256
    return dict(sorted(result.items()))


def _verify_frozen_manifest(
    snapshot: Snapshot,
    *,
    base: TwinEvidence,
    base_receipt: Snapshot,
    embedding: TwinEvidence,
    embedding_receipt: Snapshot,
) -> dict[str, str]:
    _require(snapshot.path.stat().st_mode & 0o222 == 0, "producer frozen manifest remains writable")
    observed = _parse_sha_manifest(snapshot.payload, label="producer frozen inputs")
    expected = _expected_frozen_input_hashes(
        base=base,
        base_receipt=base_receipt,
        embedding=embedding,
        embedding_receipt=embedding_receipt,
    )
    _require(observed == expected, "producer frozen-input manifest differs from raw inputs")
    return expected


def _partition_evidence(rows: Sequence[Example]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "positives": sum(item.label for item in rows),
        "negatives": sum(1 - item.label for item in rows),
        "sequences": len({item.sequence_id for item in rows}),
        "homology_components": len({item.homology_component_id for item in rows}),
        "union_components": len({item.union_component_id for item in rows}),
        "example_ids_sha256": _sequence_set_sha256(item.example_id for item in rows),
        "sequence_ids_sha256": _sequence_set_sha256(item.sequence_id for item in rows),
        "homology_component_ids_sha256": _sequence_set_sha256(
            item.homology_component_id for item in rows
        ),
        "union_component_ids_sha256": _sequence_set_sha256(
            item.union_component_id for item in rows
        ),
        "assignment_sha256": _assignment_sha256(rows),
    }


def _require_split_separation(
    training: Sequence[Example], testing: Sequence[Example], *, label: str
) -> None:
    _require(training and testing, f"{label} has an empty partition")
    _require({item.label for item in training} == {0, 1}, f"{label} training lacks a class")
    _require({item.label for item in testing} == {0, 1}, f"{label} query lacks a class")
    for attribute in ("example_id", "sequence_id", "homology_component_id", "union_component_id"):
        left = {getattr(item, attribute) for item in training}
        right = {getattr(item, attribute) for item in testing}
        _require(left.isdisjoint(right), f"{label} leaks {attribute}")


def _charge(sequence: str, ph: float) -> float:
    positive = 1.0 / (1.0 + 10.0 ** (ph - _N_TERMINUS_PKA))
    negative = 1.0 / (1.0 + 10.0 ** (_C_TERMINUS_PKA - ph))
    for residue, pka in _POSITIVE_PKA.items():
        positive += sequence.count(residue) / (1.0 + 10.0 ** (ph - pka))
    for residue, pka in _NEGATIVE_PKA.items():
        negative += sequence.count(residue) / (1.0 + 10.0 ** (pka - ph))
    return positive - negative


def _isoelectric_point(sequence: str) -> float:
    lower = 0.0
    upper = 14.0
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        if _charge(sequence, midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


@lru_cache(maxsize=100_000)
def _descriptor_features(sequence: str) -> tuple[float, ...]:
    length = len(sequence)
    charge = _charge(sequence, 7.4)
    hydrophobicities = [_HYDROPHOBICITY[residue] for residue in sequence]
    angles = np.deg2rad(np.arange(length, dtype=np.float64) * 100.0)
    hydro = np.asarray(hydrophobicities, dtype=np.float64)
    moment = (
        math.hypot(
            float(np.sum(hydro * np.cos(angles))),
            float(np.sum(hydro * np.sin(angles))),
        )
        / length
    )
    entropy_counts = np.asarray(
        [sequence.count(residue) for residue in sorted(set(sequence))], dtype=np.float64
    )
    probabilities = entropy_counts / length
    descriptors = (
        float(length),
        float(sum(_RESIDUE_MASSES_DA[residue] for residue in sequence) + _WATER_MASS_DA),
        charge,
        charge / length,
        _isoelectric_point(sequence),
        float(np.mean(hydrophobicities)),
        moment,
        sum(residue in _HYDROPHOBIC for residue in sequence) / length,
        sum(residue in _AROMATIC for residue in sequence) / length,
        sum(residue in _BASIC for residue in sequence) / length,
        sum(residue in _ACIDIC for residue in sequence) / length,
        float(-np.sum(probabilities * np.log2(probabilities))),
        max(sequence.count(residue) for residue in set(sequence)) / length,
    )
    composition = tuple(sequence.count(residue) / length for residue in _AMINO_ACIDS)
    return descriptors + composition


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _descriptor_design(
    rows: Sequence[Example], *, targets: tuple[str, ...], mean: FloatArray, scale: FloatArray
) -> FloatArray:
    continuous = np.asarray([_descriptor_features(item.sequence) for item in rows])
    standardized = (continuous - mean) / scale
    target_index = {target: index for index, target in enumerate(targets)}
    target = np.zeros((len(rows), len(targets) + 1), dtype=np.float64)
    gram = np.zeros((len(rows), 3), dtype=np.float64)
    gram_index = {"positive": 0, "negative": 1, "unknown": 2}
    for index, item in enumerate(rows):
        target[index, target_index.get(item.canonical_target, len(targets))] = 1.0
        gram[index, gram_index[item.gram]] = 1.0
    return np.concatenate((np.ones((len(rows), 1)), standardized, target, gram), axis=1)


def _descriptor_logistic(
    training: Sequence[Example],
    testing: Sequence[Example],
    settings: LogisticSettings,
    *,
    label: str,
) -> tuple[FloatArray, dict[str, object]]:
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    _require(set(labels.tolist()) == {0.0, 1.0}, f"{label} descriptor training lacks a class")
    targets = tuple(sorted({item.canonical_target for item in training}))
    continuous = np.asarray([_descriptor_features(item.sequence) for item in training])
    mean = np.mean(continuous, axis=0)
    scale = np.std(continuous, axis=0)
    scale[scale < 1e-12] = 1.0
    design = _descriptor_design(training, targets=targets, mean=mean, scale=scale)
    query_design = _descriptor_design(testing, targets=targets, mean=mean, scale=scale)
    prior = float(
        (np.sum(labels) + 0.5 * settings.prior_strength) / (labels.size + settings.prior_strength)
    )
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(coefficient.size, settings.l2, dtype=np.float64)
    penalty[0] = 0.0
    converged = False
    iterations = 0
    for iteration in range(1, settings.max_iterations + 1):
        probability = _sigmoid(design @ coefficient)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - labels) / labels.size + penalty * coefficient
        hessian = (design.T * variance) @ design / labels.size
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        _require(np.all(np.isfinite(step)), f"{label} descriptor step is non-finite")
        coefficient -= step
        iterations = iteration
        _require(np.all(np.isfinite(coefficient)), f"{label} descriptor coefficients non-finite")
        if float(np.max(np.abs(step))) <= settings.tolerance:
            converged = True
            break
    _require(converged, f"{label} descriptor exhausted its iteration budget")
    values = np.clip(_sigmoid(query_design @ coefficient), 1e-6, 1.0 - 1e-6)
    return values, {
        "converged": True,
        "iterations": iterations,
        "coefficient_count": int(coefficient.size),
        "parameters_finite": True,
    }


@lru_cache(maxsize=1_000_000)
def _global_identity(left: str, right: str) -> float:
    if right < left:
        left, right = right, left
    previous: list[tuple[int, int, int]] = [(0, 0, 0)]
    for column in range(1, len(right) + 1):
        previous.append((-column, 0, -column))
    for row, left_residue in enumerate(left, start=1):
        current: list[tuple[int, int, int]] = [(-row, 0, -row)]
        for column, right_residue in enumerate(right, start=1):
            diagonal = previous[column - 1]
            match = int(left_residue == right_residue)
            diagonal_state = (
                diagonal[0] + (1 if match else -1),
                diagonal[1] + match,
                diagonal[2] - 1,
            )
            up = previous[column]
            up_state = (up[0] - 1, up[1], up[2] - 1)
            prior_left = current[column - 1]
            left_state = (prior_left[0] - 1, prior_left[1], prior_left[2] - 1)
            current.append(max(diagonal_state, up_state, left_state))
        previous = current
    _, matches, negative_length = previous[-1]
    return matches / -negative_length


def _knn_probability(query: Example, training: Sequence[Example], settings: KnnSettings) -> float:
    candidates = [
        index
        for index, item in enumerate(training)
        if item.canonical_target == query.canonical_target
    ]
    if not candidates:
        candidates = [index for index, item in enumerate(training) if item.gram == query.gram]
    if not candidates:
        candidates = list(range(len(training)))
    labels = np.asarray([training[index].label for index in candidates], dtype=np.float64)
    prior = float(
        (np.sum(labels) + 0.5 * settings.prior_strength) / (labels.size + settings.prior_strength)
    )
    scored = [
        (index, _global_identity(query.sequence, training[index].sequence)) for index in candidates
    ]
    ordered = sorted(
        scored,
        key=lambda pair: (
            -pair[1],
            training[pair[0]].sequence,
            training[pair[0]].canonical_target,
            pair[0],
        ),
    )[: settings.neighbors]
    weights = np.asarray(
        [
            max(identity**settings.similarity_power, settings.minimum_weight)
            for _, identity in ordered
        ],
        dtype=np.float64,
    )
    neighbor_labels = np.asarray([training[index].label for index, _ in ordered], dtype=np.float64)
    numerator = float(weights @ neighbor_labels) + settings.prior_strength * prior
    denominator = float(np.sum(weights)) + settings.prior_strength
    _require(denominator > 0, "kNN denominator is not positive")
    return float(np.clip(numerator / denominator, 1e-6, 1.0 - 1e-6))


def _fit_base(
    training: Sequence[Example],
    testing: Sequence[Example],
    *,
    base_config: BaseConfig,
    label: str,
) -> tuple[dict[str, FloatArray], dict[str, object]]:
    _require_split_separation(training, testing, label=label)
    descriptor, audit = _descriptor_logistic(training, testing, base_config.logistic, label=label)
    knn = np.asarray(
        [_knn_probability(item, training, base_config.knn) for item in testing],
        dtype=np.float64,
    )
    _require(
        descriptor.shape == (len(testing),)
        and knn.shape == (len(testing),)
        and np.all(np.isfinite(descriptor))
        and np.all(np.isfinite(knn)),
        f"{label} base predictions are invalid",
    )
    return {"descriptor_logistic": descriptor, "homology_knn": knn}, {
        "training": _partition_evidence(training),
        "query": _partition_evidence(testing),
        "descriptor_fit": audit,
        "homology_knn_fit": {"finite_predictions": True},
    }


def _ridge_logistic(
    design: FloatArray,
    labels: FloatArray,
    *,
    l2: float,
    prior_strength: float,
    max_iterations: int,
    tolerance: float,
    label: str,
) -> tuple[FloatArray, int]:
    _require(
        design.ndim == 2 and design.shape[0] == labels.size and labels.size > 0,
        f"{label} design is invalid",
    )
    _require(np.all(np.isfinite(design)) and np.all(np.isfinite(labels)), f"{label} non-finite")
    prior = float((np.sum(labels) + 0.5 * prior_strength) / (labels.size + prior_strength))
    _require(0.0 < prior < 1.0, f"{label} prior is degenerate")
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(design.shape[1], l2, dtype=np.float64)
    penalty[0] = 0.0
    for iteration in range(1, max_iterations + 1):
        probability = _sigmoid(design @ coefficient)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - labels) / labels.size + penalty * coefficient
        hessian = (design.T * variance) @ design / labels.size
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        _require(np.all(np.isfinite(step)), f"{label} Newton step is non-finite")
        coefficient -= step
        _require(np.all(np.isfinite(coefficient)), f"{label} coefficients are non-finite")
        if float(np.max(np.abs(step))) <= tolerance:
            return coefficient, iteration
    raise VerificationError(f"{label} exhausted its iteration budget")


def _embedding_rows(rows: Sequence[Example], table: EmbeddingTable) -> FloatArray:
    return np.asarray(
        [table.matrix[table.row_by_sequence_id[item.sequence_id]] for item in rows],
        dtype=np.float64,
    )


def _esm_design(
    rows: Sequence[Example],
    table: EmbeddingTable,
    mean: FloatArray,
    scale: FloatArray,
    config: Config,
) -> FloatArray:
    standardized = (_embedding_rows(rows, table) - mean) / scale
    targets = np.zeros((len(rows), len(config.canonical_targets)), dtype=np.float64)
    target_index = {name: index for index, name in enumerate(config.canonical_targets)}
    grams = np.zeros((len(rows), len(config.gram_classes)), dtype=np.float64)
    gram_index = {name: index for index, name in enumerate(config.gram_classes)}
    for index, item in enumerate(rows):
        _require(item.canonical_target in target_index, "ESM head received unknown target")
        _require(item.gram in gram_index, "ESM head received unknown Gram class")
        targets[index, target_index[item.canonical_target]] = 1.0
        grams[index, gram_index[item.gram]] = 1.0
    design = np.column_stack((np.ones(len(rows)), standardized, targets, grams))
    _require(np.all(np.isfinite(design)), "ESM design contains non-finite values")
    return design


def _fit_esm(
    training: Sequence[Example],
    *,
    table: EmbeddingTable,
    config: Config,
    outer_fold: int,
    inner_fold: int | None,
) -> tuple[dict[str, object], FloatArray, FloatArray, FloatArray]:
    raw = _embedding_rows(training, table)
    _require(
        raw.shape == (len(training), config.embedding_dimension),
        "ESM training matrix shape differs",
    )
    mean = np.mean(raw, axis=0, dtype=np.float64)
    scale = np.std(raw, axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    design = _esm_design(training, table, mean, scale, config)
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    scope = (
        f"outer {outer_fold}" if inner_fold is None else f"outer {outer_fold}/inner {inner_fold}"
    )
    coefficients, iterations = _ridge_logistic(
        design,
        labels,
        l2=config.esm_l2,
        prior_strength=config.esm_prior_strength,
        max_iterations=config.esm_max_iterations,
        tolerance=config.esm_tolerance,
        label=f"{scope} ESM head",
    )
    offset = 1 + config.embedding_dimension
    present_targets = {item.canonical_target for item in training}
    absent_indices = [
        offset + index
        for index, target in enumerate(config.canonical_targets)
        if target not in present_targets
    ]
    _require(
        all(coefficients[index] == 0.0 for index in absent_indices),
        f"{scope} absent-target coefficient moved",
    )
    evidence = {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "training": _partition_evidence(training),
        "iterations": iterations,
        "converged": True,
        "feature_order": [
            "intercept",
            *(f"esm_{index:03d}" for index in range(config.embedding_dimension)),
            *(f"target::{name}" for name in config.canonical_targets),
            *(f"gram::{name}" for name in config.gram_classes),
        ],
        "embedding_mean": [float(value) for value in mean],
        "embedding_scale": [float(value) for value in scale],
        "coefficients": [float(value) for value in coefficients],
        "absent_target_zero_coefficients": [
            config.canonical_targets[index - offset] for index in absent_indices
        ],
    }
    return evidence, mean, scale, coefficients


def _predict_esm(
    rows: Sequence[Example],
    table: EmbeddingTable,
    mean: FloatArray,
    scale: FloatArray,
    coefficients: FloatArray,
    config: Config,
) -> FloatArray:
    values = np.clip(
        _sigmoid(_esm_design(rows, table, mean, scale, config) @ coefficients),
        config.probability_clip,
        1.0 - config.probability_clip,
    )
    _require(values.shape == (len(rows),) and np.all(np.isfinite(values)), "invalid ESM output")
    return values


def _logit(value: float, clip: float) -> float:
    probability = float(np.clip(value, clip, 1.0 - clip))
    return math.log(probability / (1.0 - probability))


def _nested_digest(records: Sequence[tuple[int, int, str, tuple[float, ...]]]) -> str:
    payload = "".join(
        "\t".join((str(outer), str(inner), example_id, *(value.hex() for value in features))) + "\n"
        for outer, inner, example_id, features in sorted(records)
    ).encode("ascii")
    return _sha256(payload)


def _fit_stack(
    training: Sequence[Example],
    features: Mapping[str, tuple[float, ...]],
    *,
    outer_fold: int,
    config: Config,
) -> tuple[dict[str, object], FloatArray, FloatArray, FloatArray]:
    _require(
        set(features) == {item.example_id for item in training},
        f"outer {outer_fold} nested support differs",
    )
    matrix = np.asarray([features[item.example_id] for item in training], dtype=np.float64)
    _require(
        matrix.shape == (len(training), len(_STACK_FEATURES)) and np.all(np.isfinite(matrix)),
        "nested stack features are invalid",
    )
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    scale = np.std(matrix, axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(training)), (matrix - mean) / scale))
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    coefficients, iterations = _ridge_logistic(
        design,
        labels,
        l2=config.stack_l2,
        prior_strength=config.stack_prior_strength,
        max_iterations=config.stack_max_iterations,
        tolerance=config.stack_tolerance,
        label=f"outer {outer_fold} stack",
    )
    return (
        {
            "outer_fold": outer_fold,
            "training": _partition_evidence(training),
            "feature_names": list(_STACK_FEATURES),
            "feature_means": [float(value) for value in mean],
            "feature_scales": [float(value) for value in scale],
            "coefficients": [float(value) for value in coefficients],
            "iterations": iterations,
            "converged": True,
        },
        mean,
        scale,
        coefficients,
    )


def _predict_stack(
    features: Sequence[tuple[float, ...]],
    mean: FloatArray,
    scale: FloatArray,
    coefficients: FloatArray,
    clip: float,
) -> FloatArray:
    matrix = np.asarray(features, dtype=np.float64)
    design = np.column_stack((np.ones(len(matrix)), (matrix - mean) / scale))
    values = np.clip(_sigmoid(design @ coefficients), clip, 1.0 - clip)
    _require(np.all(np.isfinite(values)), "stack output is non-finite")
    return values


@dataclass(frozen=True, slots=True)
class OuterReconstruction:
    probabilities: Mapping[str, Mapping[str, float]]
    fold_model: Mapping[str, object]
    inner_fits: tuple[Mapping[str, object], ...]
    nested_records: tuple[tuple[int, int, str, tuple[float, ...]], ...]


def _reconstruct_outer_fold(
    ordered: Sequence[Example],
    accepted: Mapping[str, Mapping[str, float]],
    table: EmbeddingTable,
    *,
    outer: int,
    base_config: BaseConfig,
    config: Config,
) -> OuterReconstruction:
    """Rebuild one outer fold; accepted OOF enters only at outer-query prediction time."""

    outer_training = tuple(item for item in ordered if item.fold != outer)
    outer_query = tuple(item for item in ordered if item.fold == outer)
    _require_split_separation(outer_training, outer_query, label=f"outer {outer}")
    reproduced, base_evidence = _fit_base(
        outer_training,
        outer_query,
        base_config=base_config,
        label=f"outer {outer} base reproduction",
    )
    for name in ("descriptor_logistic", "homology_knn"):
        expected = np.asarray(
            [accepted[name][item.example_id] for item in outer_query], dtype=np.float64
        )
        _require(
            np.array_equal(reproduced[name], expected),
            f"outer {outer} {name} does not reproduce accepted OOF",
        )
    reproduced_equal = np.mean(
        np.stack((reproduced["descriptor_logistic"], reproduced["homology_knn"])), axis=0
    )
    accepted_equal = np.asarray(
        [accepted["equal_weight_ensemble"][item.example_id] for item in outer_query],
        dtype=np.float64,
    )
    _require(np.array_equal(reproduced_equal, accepted_equal), "outer equal mean differs")

    esm_evidence, esm_mean, esm_scale, esm_coefficients = _fit_esm(
        outer_training, table=table, config=config, outer_fold=outer, inner_fold=None
    )
    outer_esm = _predict_esm(outer_query, table, esm_mean, esm_scale, esm_coefficients, config)
    nested: dict[str, tuple[float, ...]] = {}
    inner_entries: list[Mapping[str, object]] = []
    all_records: list[tuple[int, int, str, tuple[float, ...]]] = []
    for inner in range(config.folds):
        if inner == outer:
            continue
        inner_training = tuple(item for item in ordered if item.fold not in {outer, inner})
        inner_query = tuple(item for item in ordered if item.fold == inner)
        _require_split_separation(
            inner_training, (*inner_query, *outer_query), label=f"outer {outer}/inner {inner}"
        )
        base_values, inner_base_evidence = _fit_base(
            inner_training,
            inner_query,
            base_config=base_config,
            label=f"outer {outer}/inner {inner} base",
        )
        inner_esm_evidence, mean, scale, coefficients = _fit_esm(
            inner_training,
            table=table,
            config=config,
            outer_fold=outer,
            inner_fold=inner,
        )
        esm_values = _predict_esm(inner_query, table, mean, scale, coefficients, config)
        records: list[tuple[int, int, str, tuple[float, ...]]] = []
        for index, item in enumerate(inner_query):
            feature = (
                _logit(float(base_values["descriptor_logistic"][index]), config.probability_clip),
                _logit(float(base_values["homology_knn"][index]), config.probability_clip),
                _logit(float(esm_values[index]), config.probability_clip),
            )
            _require(item.example_id not in nested, "nested feature row was generated twice")
            nested[item.example_id] = feature
            records.append((outer, inner, item.example_id, feature))
        all_records.extend(records)
        inner_entries.append(
            {
                "outer_fold": outer,
                "inner_fold": inner,
                "excluded_folds": [outer, inner],
                "base": inner_base_evidence,
                "esm": inner_esm_evidence,
                "inner_feature_rows": len(records),
                "inner_feature_rows_sha256": _nested_digest(records),
                "accepted_oof_probabilities_consumed": False,
            }
        )
    stack_evidence, stack_mean, stack_scale, stack_coefficients = _fit_stack(
        outer_training, nested, outer_fold=outer, config=config
    )
    outer_features = [
        (
            _logit(accepted["descriptor_logistic"][item.example_id], config.probability_clip),
            _logit(accepted["homology_knn"][item.example_id], config.probability_clip),
            _logit(float(outer_esm[index]), config.probability_clip),
        )
        for index, item in enumerate(outer_query)
    ]
    stack = _predict_stack(
        outer_features, stack_mean, stack_scale, stack_coefficients, config.probability_clip
    )
    probabilities: dict[str, dict[str, float]] = {method: {} for method in _METHODS}
    for index, item in enumerate(outer_query):
        descriptor = accepted["descriptor_logistic"][item.example_id]
        knn = accepted["homology_knn"][item.example_id]
        equal = accepted["equal_weight_ensemble"][item.example_id]
        esm = float(outer_esm[index])
        values = {
            "descriptor_logistic": descriptor,
            "homology_knn": knn,
            "equal_weight_ensemble": equal,
            _ESM_MODEL: esm,
            _DESCRIPTOR_ESM_BLEND: 0.5 * descriptor + 0.5 * esm,
            _BASE_ESM_BLEND: 0.5 * equal + 0.5 * esm,
            _STACK_MODEL: float(stack[index]),
        }
        for method, probability in values.items():
            _require(math.isfinite(probability) and 0 <= probability <= 1, "invalid probability")
            probabilities[method][item.example_id] = probability
    return OuterReconstruction(
        probabilities=probabilities,
        fold_model={
            "outer_fold": outer,
            "base_reproduction": base_evidence,
            "accepted_base_oof_exactly_reproduced": True,
            "esm_head": esm_evidence,
            "stack": stack_evidence,
            "outer_query": _partition_evidence(outer_query),
        },
        inner_fits=tuple(inner_entries),
        nested_records=tuple(all_records),
    )


def _reconstruct_predictions(
    examples: Sequence[Example],
    accepted: Mapping[str, Mapping[str, float]],
    table: EmbeddingTable,
    *,
    base_config: BaseConfig,
    config: Config,
) -> tuple[tuple[Prediction, ...], dict[str, object], dict[str, object]]:
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    by_method: dict[str, dict[str, float]] = {method: {} for method in _METHODS}
    fold_models: list[Mapping[str, object]] = []
    inner_fits: list[Mapping[str, object]] = []
    nested_records: list[tuple[int, int, str, tuple[float, ...]]] = []
    for outer in range(config.folds):
        reconstruction = _reconstruct_outer_fold(
            ordered,
            accepted,
            table,
            outer=outer,
            base_config=base_config,
            config=config,
        )
        for method in _METHODS:
            overlap = set(by_method[method]) & set(reconstruction.probabilities[method])
            _require(not overlap, "outer prediction support overlaps")
            by_method[method].update(reconstruction.probabilities[method])
        fold_models.append(reconstruction.fold_model)
        inner_fits.extend(reconstruction.inner_fits)
        nested_records.extend(reconstruction.nested_records)
    expected_ids = {item.example_id for item in ordered}
    _require(
        all(set(values) == expected_ids for values in by_method.values()), "OOF support differs"
    )
    _require(
        len(fold_models) == config.expected_outer_folds
        and len(inner_fits) == config.expected_ordered_outer_inner_splits
        and len(nested_records) == config.expected_nested_feature_rows,
        "nested fit census differs",
    )
    predictions = tuple(
        Prediction(
            model=method,
            **asdict(item),
            probability=by_method[method][item.example_id],
        )
        for method in _METHODS
        for item in ordered
    )
    _require(
        len(predictions) == len(_METHODS) * config.expected_examples,
        "prediction row census differs",
    )
    fold_document = {
        "schema_version": 1,
        "artifact": "esm_union_outer_fold_models_v1",
        "status": "passed",
        "feature_order": [
            "intercept",
            "320_standardized_esm",
            "seven_fixed_target_columns",
            "two_fixed_gram_columns",
        ],
        "outer_esm_fits": len(fold_models),
        "outer_stack_fits": len(fold_models),
        "folds": fold_models,
    }
    nested_document = {
        "schema_version": 1,
        "artifact": "esm_union_nested_refits_v1",
        "status": "passed",
        "ordered_outer_inner_splits": len(inner_fits),
        "descriptor_refits": len(inner_fits),
        "homology_knn_refits": len(inner_fits),
        "nested_esm_fits": len(inner_fits),
        "nested_feature_rows": len(nested_records),
        "feature_encoding": _NESTED_FEATURE_ENCODING,
        "all_nested_feature_rows_sha256": _nested_digest(nested_records),
        "accepted_ordinary_oof_used_for_stack_training": False,
        "splits": inner_fits,
    }
    return predictions, fold_document, nested_document


def _roc_auc(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: IntArray, probabilities: FloatArray) -> float | None:
    positive_count = int(np.sum(labels))
    if positive_count == 0:
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
        result += (new_positives / positive_count) * (
            true_positive / (true_positive + false_positive)
        )
        index = end
    return float(result)


def _binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, calibration_bins: int
) -> dict[str, int | float | None]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    _require(
        y.ndim == 1 and probability.ndim == 1 and y.size == probability.size and y.size > 0,
        "metric vectors are invalid",
    )
    _require(np.all((y == 0) | (y == 1)), "metric labels are invalid")
    _require(
        np.all(np.isfinite(probability)) and np.all((probability >= 0) & (probability <= 1)),
        "metric probabilities are invalid",
    )
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    prediction = probability >= 0.5
    positive = y == 1
    negative = ~positive
    sensitivity = float(np.mean(prediction[positive])) if np.any(positive) else None
    specificity = float(np.mean(~prediction[negative])) if np.any(negative) else None
    balanced = (
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
    positives = int(np.sum(y))
    return {
        "n": int(y.size),
        "positives": positives,
        "negatives": int(y.size - positives),
        "prevalence": float(np.mean(y)),
        "roc_auc": _roc_auc(y, probability),
        "average_precision": _average_precision(y, probability),
        "brier": float(np.mean(np.square(probability - y))),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "balanced_accuracy_at_0_5": balanced,
        "sensitivity_at_0_5": sensitivity,
        "specificity_at_0_5": specificity,
        "ece_equal_width": calibration_error,
    }


def _metric(rows: Sequence[Prediction], bins: int) -> dict[str, int | float | None]:
    _require(bool(rows), "metric subgroup is empty")
    return _binary_metrics(
        [item.label for item in rows],
        [item.probability for item in rows],
        calibration_bins=bins,
    )


def _component_equal_log_loss(rows: Sequence[Prediction]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        probability = float(np.clip(item.probability, 1e-15, 1.0 - 1e-15))
        grouped[item.union_component_id].append(
            -(item.label * math.log(probability) + (1 - item.label) * math.log1p(-probability))
        )
    _require(bool(grouped), "component-equal metric has no components")
    return float(np.mean([np.mean(values) for _, values in sorted(grouped.items())]))


def _metrics_document(predictions: Sequence[Prediction], *, config: Config) -> dict[str, object]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped[item.model].append(item)
    _require(tuple(grouped) == _METHODS, "metric method order differs")
    example_ids = [item.example_id for item in grouped[_METHODS[0]]]
    _require(
        all([item.example_id for item in grouped[method]] == example_ids for method in _METHODS),
        "metric support is unaligned",
    )
    component_rows: dict[str, list[str]] = defaultdict(list)
    reference_by_id = {item.example_id: item for item in grouped[_PROMOTION_REFERENCE]}
    for example_id in example_ids:
        component_rows[reference_by_id[example_id].union_component_id].append(example_id)
    components = tuple(sorted(component_rows))
    sizes = sorted((len(values) for values in component_rows.values()), reverse=True)
    _require(
        len(components) == config.expected_union_components
        and sizes[0] == config.expected_largest_union_component_contexts
        and sum(size > 100 for size in sizes) == config.expected_union_components_over_100_contexts,
        "metric component census differs",
    )
    largest = min(
        component for component, values in component_rows.items() if len(values) == sizes[0]
    )
    methods: dict[str, object] = {}
    for method in _METHODS:
        rows = grouped[method]
        by_identity: dict[str, object] = {}
        for left, right in pairwise(config.similarity_bin_edges):
            selected = [item for item in rows if left <= item.max_train_identity < right]
            if selected:
                by_identity[f"[{left:.2f},{right:.2f})"] = _metric(
                    selected, config.calibration_bins
                )
        sensitivity = [item for item in rows if item.union_component_id != largest]
        methods[method] = {
            "overall": _metric(rows, config.calibration_bins),
            "by_fold": {
                str(fold): _metric(
                    [item for item in rows if item.fold == fold], config.calibration_bins
                )
                for fold in range(config.folds)
            },
            "by_gram": {
                gram: _metric([item for item in rows if item.gram == gram], config.calibration_bins)
                for gram in config.gram_classes
            },
            "by_canonical_target": {
                target: _metric(
                    [item for item in rows if item.canonical_target == target],
                    config.calibration_bins,
                )
                for target in config.canonical_targets
            },
            "by_max_train_identity": by_identity,
            "union_component_equal_log_loss": _component_equal_log_loss(rows),
            "leave_largest_union_component_out": {
                "excluded_union_component_id": largest,
                "excluded_contexts": len(component_rows[largest]),
                "metrics": _metric(sensitivity, config.calibration_bins),
            },
        }
    aligned = {method: {item.example_id: item for item in grouped[method]} for method in _METHODS}
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss")
    bootstrap = {method: {name: [] for name in metric_names} for method in _METHODS}
    deltas = {name: [] for name in ("roc_auc", "brier", "log_loss")}
    generator = np.random.default_rng(config.bootstrap_seed)
    digest = hashlib.sha256()
    for replicate in range(config.bootstrap_replicates):
        sampled = tuple(
            str(value) for value in generator.choice(components, size=len(components), replace=True)
        )
        for position, component in enumerate(sampled):
            digest.update(f"{replicate}\t{position}\t{component}\n".encode("ascii"))
        sampled_ids = [
            example_id for component in sampled for example_id in component_rows[component]
        ]
        values: dict[str, dict[str, int | float | None]] = {}
        for method in _METHODS:
            values[method] = _metric(
                [aligned[method][example_id] for example_id in sampled_ids],
                config.calibration_bins,
            )
            for name in metric_names:
                value = values[method][name]
                _require(
                    value is not None and math.isfinite(float(value)),
                    f"bootstrap replicate {replicate + 1} has undefined {name}",
                )
                bootstrap[method][name].append(float(value))
        for name in deltas:
            candidate = values[_PROMOTION_CANDIDATE][name]
            reference = values[_PROMOTION_REFERENCE][name]
            assert candidate is not None and reference is not None
            deltas[name].append(float(candidate) - float(reference))
    for method in _METHODS:
        cast(dict[str, object], methods[method])["union_component_bootstrap_95ci"] = {
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
                "successful_replicates": len(values),
            }
            for name, values in bootstrap[method].items()
        }
    candidate_overall = cast(
        dict[str, object], cast(dict[str, object], methods[_PROMOTION_CANDIDATE])["overall"]
    )
    reference_overall = cast(
        dict[str, object], cast(dict[str, object], methods[_PROMOTION_REFERENCE])["overall"]
    )
    paired = {
        name: {
            "point": float(candidate_overall[name]) - float(reference_overall[name]),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "successful_replicates": len(values),
        }
        for name, values in deltas.items()
    }
    checks = {
        "roc_auc_delta_lower_above_minimum": paired["roc_auc"]["lower"]
        > config.promotion_auc_delta_lower_minimum,
        "brier_delta_upper_below_maximum": paired["brier"]["upper"]
        < config.promotion_brier_delta_upper_maximum,
        "log_loss_delta_upper_below_maximum": paired["log_loss"]["upper"]
        < config.promotion_log_loss_delta_upper_maximum,
    }
    matrix = np.asarray(
        [
            [aligned[method][example_id].probability for example_id in example_ids]
            for method in _METHODS
        ],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(matrix)
    _require(np.all(np.isfinite(correlation)), "method probability correlations non-finite")
    return {
        "schema_version": 1,
        "artifact": "esm_union_nested_ensemble_metrics_v1",
        "methods": methods,
        "primary_comparison": {
            "candidate": _PROMOTION_CANDIDATE,
            "reference": _PROMOTION_REFERENCE,
            "delta_definition": "candidate minus reference",
            "union_component_paired_bootstrap_95ci": paired,
        },
        "promotion_rule": {
            "scope": "eligible_for_untouched_external_validation_only",
            "requirements": {
                "roc_auc_delta_bootstrap_95ci_lower_strictly_above": config.promotion_auc_delta_lower_minimum,
                "brier_delta_bootstrap_95ci_upper_strictly_below": config.promotion_brier_delta_upper_maximum,
                "log_loss_delta_bootstrap_95ci_upper_strictly_below": config.promotion_log_loss_delta_upper_maximum,
            },
            "decision": {
                "candidate": _PROMOTION_CANDIDATE,
                "passed": all(checks.values()),
                "checks": checks,
            },
            "controls_production_eligibility": False,
        },
        "paired_component_bootstrap": {
            "unit": "union_component_id",
            "weighting": "union_component_resample_context_multiplicity_v1",
            "components": len(components),
            "replicates": config.bootstrap_replicates,
            "successful_replicates": config.bootstrap_replicates,
            "rejected_replicates": 0,
            "draw_policy": "exactly first requested draws; fail if any draw is undefined; no redraw",
            "draw_encoding": _BOOTSTRAP_DRAW_ENCODING,
            "draws_sha256": digest.hexdigest(),
            "seed": config.bootstrap_seed,
        },
        "probability_correlation": {
            method: {
                other: float(correlation[index, other_index])
                for other_index, other in enumerate(_METHODS)
            }
            for index, method in enumerate(_METHODS)
        },
        "statistical_gate_passed": all(checks.values()),
        "production_eligible": False,
    }


def _predictions_bytes(predictions: Sequence[Prediction]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(_PREDICTION_SCHEMA), lineterminator="\n")
    writer.writeheader()
    for item in predictions:
        row = asdict(item)
        row["max_train_identity"] = format(item.max_train_identity, ".17g")
        row["probability"] = format(item.probability, ".17g")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _expected_semantic_payloads(
    *,
    predictions: Sequence[Prediction],
    fold_models: Mapping[str, object],
    nested_fits: Mapping[str, object],
    coverage: Mapping[str, object],
    metrics: Mapping[str, object],
    config: Config,
    git_commit: str,
    code_manifest: Snapshot,
    frozen_manifest: Snapshot,
    code_attestation: Mapping[str, object],
    base: TwinEvidence,
    base_receipt: Snapshot,
    base_manifest: Mapping[str, object],
    split_receipt: Mapping[str, object],
    embedding: TwinEvidence,
    embedding_receipt: Snapshot,
    embedding_manifest: Mapping[str, object],
    embedding_receipt_document: Mapping[str, object],
    table: EmbeddingTable,
) -> dict[str, bytes]:
    payloads = {
        "embedding_coverage.json": _pretty_json_bytes(coverage),
        "esm_union_oof_predictions.csv": _predictions_bytes(predictions),
        "fold_models.json": _pretty_json_bytes(fold_models),
        "metrics.json": _pretty_json_bytes(metrics),
        "nested_fits.json": _pretty_json_bytes(nested_fits),
    }
    roles = {
        "embedding_coverage.json": "exact base-to-embedding sequence join evidence",
        "esm_union_oof_predictions.csv": "seven-method context-level outer OOF predictions",
        "fold_models.json": "five outer ESM heads and nested stack fits",
        "metrics.json": "metrics and shared union-component bootstrap evidence",
        "nested_fits.json": "twenty ordered outer-inner base and ESM refits",
    }
    artifacts = {
        name: {"filename": name, "role": role, "sha256": _sha256(payloads[name])}
        for name, role in sorted(roles.items())
    }
    statistical_passed = cast(bool, metrics["statistical_gate_passed"])
    manifest = {
        "schema_version": 1,
        "artifact": _ARTIFACT,
        "status": _OUTPUT_STATUS,
        "git_commit": git_commit,
        "config_sha256": config.sha256,
        "methods": list(_METHODS),
        "promotion_reference": _PROMOTION_REFERENCE,
        "promotion_candidate": _PROMOTION_CANDIDATE,
        "statistical_gate_passed": statistical_passed,
        "statistical_status": (
            "eligible_for_untouched_external_validation_only"
            if statistical_passed
            else "did_not_pass_frozen_development_gate"
        ),
        "production_policy": dict(config.production_policy),
        "production_eligible": False,
        "production_weights": {"apex": 0.0, "esm": 0.0},
        "apex": {
            "accepted_as_input": False,
            "present_in_feature_vectors": False,
            "production_weight": 0.0,
        },
        "nesting": {
            "outer_folds": config.expected_outer_folds,
            "ordered_outer_inner_splits": config.expected_ordered_outer_inner_splits,
            "nested_feature_rows": config.expected_nested_feature_rows,
            "inner_descriptor_refits": config.expected_ordered_outer_inner_splits,
            "inner_homology_knn_refits": config.expected_ordered_outer_inner_splits,
            "inner_esm_refits": config.expected_nested_esm_fits,
            "outer_esm_fits": config.expected_outer_esm_fits,
            "outer_stack_fits": config.expected_outer_stack_fits,
            "accepted_ordinary_oof_used_for_stack_training": False,
            "outer_base_source": "accepted OOF after exact refit reproduction",
        },
        "census": {
            "contexts": config.expected_examples,
            "source_observations": config.expected_source_observations,
            "sequences": config.expected_sequences,
            "positives": config.expected_positive_examples,
            "negatives": config.expected_negative_examples,
            "homology_components": config.expected_homology_components,
            "union_components": config.expected_union_components,
            "prediction_rows": len(predictions),
        },
        "accepted_gate1": {
            "producer_job_id": config.base_producer_job_id,
            "audit_job_id": config.base_audit_job_id,
            "publication_top_sha256": base.snapshots["SHA256SUMS"].sha256,
            "semantic_top_sha256": base.snapshots["gate1/SHA256SUMS"].sha256,
            "independent_receipt_sha256": base_receipt.sha256,
            "production_handshake": base.handshake,
            "accepted_split": base_manifest.get("accepted_split"),
            "split_census": split_receipt.get("census"),
        },
        "accepted_embeddings": {
            "producer_job_id": config.embedding_producer_job_id,
            "audit_job_id": config.embedding_audit_job_id,
            "protocol_git_commit": config.embedding_protocol_git_commit,
            "publication_top_sha256": embedding.snapshots["SHA256SUMS"].sha256,
            "semantic_top_sha256": embedding.snapshots["embeddings/SHA256SUMS"].sha256,
            "independent_receipt_sha256": embedding_receipt.sha256,
            "production_handshake": embedding.handshake,
            "acceptance_scope": embedding_receipt_document.get("acceptance_scope"),
            "runtime": embedding_receipt_document.get("runtime"),
            "model": config.embedding_model,
            "representation_layer": config.embedding_representation_layer,
            "shape": [config.embedding_records, config.embedding_dimension],
            "dtype": config.embedding_dtype,
            "npy_sha256": table.matrix_sha256,
            "tensor_data_sha256": table.tensor_data_sha256,
            "candidate_status": embedding_manifest.get("status"),
        },
        "input_sha256": {
            "code_manifest": code_manifest.sha256,
            "frozen_input_manifest": frozen_manifest.sha256,
            "config": config.sha256,
            "base_config": config.base_config_sha256,
            "embedding_config": config.embedding_config_sha256,
            "base_publication_top": base.snapshots["SHA256SUMS"].sha256,
            "base_independent_receipt": base_receipt.sha256,
            "embedding_publication_top": embedding.snapshots["SHA256SUMS"].sha256,
            "embedding_independent_receipt": embedding_receipt.sha256,
        },
        "code_attestation": dict(code_attestation),
        "artifacts": artifacts,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    payloads["manifest.json"] = _pretty_json_bytes(manifest)
    return payloads


def _scan_forbidden(payloads: Iterable[bytes], prefixes: Sequence[bytes]) -> None:
    for payload in payloads:
        _require(_ABSOLUTE_BYTES_RE.search(payload) is None, "publication exposes absolute path")
        for prefix in prefixes:
            _require(prefix not in payload, "publication exposes forbidden prefix")


def verify_esm_union_ensemble_oof_twins(
    *,
    producer_twin_root: str | Path,
    producer_code_manifest: str | Path,
    producer_frozen_input_manifest: str | Path,
    base_twin_root: str | Path,
    base_independent_receipt: str | Path,
    embedding_twin_root: str | Path,
    embedding_independent_receipt: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, object]:
    """Reconstruct the complete publication and return a path-free receipt."""

    repository = _resolved_directory(repo_root, label="repository root")
    config_requested = Path(config_path).absolute()
    _require_no_symlink(config_requested, label="ensemble config")
    config_file = config_requested.resolve(strict=True)
    _require(
        config_file == (repository / _CONFIG_PATH).resolve(strict=True),
        "config is not the repository logical config",
    )
    config = load_config(config_file)
    config_snapshot = _read_snapshot(config_file, label="ESM union ensemble config")
    _require(config_snapshot.sha256 == config.sha256, "ensemble config changed while loading")
    base_config_snapshot = _read_snapshot(
        repository / _BASE_CONFIG_PATH, label="accepted Gate-1 config"
    )
    embedding_config_snapshot = _read_snapshot(
        repository / _EMBEDDING_CONFIG_PATH, label="accepted embedding config"
    )
    base_config = _load_base_config(base_config_snapshot, expected_sha256=config.base_config_sha256)
    _require(
        embedding_config_snapshot.sha256 == config.embedding_config_sha256,
        "accepted embedding config hash differs",
    )
    _verify_repository(repository, expected_git_commit)
    verifier_sha256 = _verify_execution_source(repository, expected_git_commit)

    base = _read_twins(
        base_twin_root,
        expected_inventory=_BASE_TREE_FILES,
        expected_top_sha256=config.base_publication_top_sha256,
        expected_job_id=config.base_producer_job_id,
        label="accepted Gate-1",
    )
    embedding = _read_twins(
        embedding_twin_root,
        expected_inventory=_EMBEDDING_TREE_FILES,
        expected_top_sha256=config.embedding_publication_top_sha256,
        expected_job_id=config.embedding_producer_job_id,
        label="accepted embeddings",
    )
    base_receipt = _read_snapshot(base_independent_receipt, label="accepted Gate-1 receipt")
    embedding_receipt = _read_snapshot(
        embedding_independent_receipt, label="accepted embedding receipt"
    )
    examples, accepted, base_manifest, split_receipt = _verify_base_chain(
        base, base_receipt, config=config
    )
    embedding_manifest, embedding_receipt_document = _verify_embedding_chain(
        embedding, embedding_receipt, config=config
    )
    table, coverage = _read_embedding_table(
        embedding.snapshots["embeddings/embedding_index.csv"],
        embedding.snapshots["embeddings/embeddings.npy"],
        examples,
        config=config,
    )

    code_manifest = _read_snapshot(producer_code_manifest, label="producer code manifest")
    frozen_manifest = _read_snapshot(
        producer_frozen_input_manifest, label="producer frozen-input manifest"
    )
    code_attestation = _verify_code_manifest(
        code_manifest,
        repository=repository,
        expected_commit=expected_git_commit,
        config=config,
        base_config=base_config_snapshot,
        embedding_config=embedding_config_snapshot,
    )
    frozen_hashes = _verify_frozen_manifest(
        frozen_manifest,
        base=base,
        base_receipt=base_receipt,
        embedding=embedding,
        embedding_receipt=embedding_receipt,
    )

    producer = _read_twins(
        producer_twin_root,
        expected_inventory=_PRODUCER_FILES,
        expected_top_sha256=None,
        expected_job_id=None,
        label="nested ensemble producer",
        required_checksum_marker_mode=0o444,
    )
    _require(
        producer.snapshots["CODE_SHA256SUMS"].payload == code_manifest.payload
        and producer.snapshots["FROZEN_INPUT_SHA256SUMS"].payload == frozen_manifest.payload,
        "explicit producer manifests differ from the task-0 published copies",
    )
    _verify_semantic_tree(
        producer.snapshots,
        prefix="ensemble",
        semantic_names=_SEMANTIC_OUTPUTS,
        expected_top_sha256=producer.snapshots["ensemble/SHA256SUMS"].sha256,
        label="nested ensemble producer",
        required_marker_mode=0o444,
    )
    candidate_manifest = _json_object(
        producer.snapshots["ensemble/manifest.json"], label="candidate ensemble manifest"
    )
    runtime = candidate_manifest.get("runtime")
    input_hashes = candidate_manifest.get("input_sha256")
    apex = candidate_manifest.get("apex")
    production_weights = candidate_manifest.get("production_weights")
    _require(
        candidate_manifest.get("schema_version") == 1
        and candidate_manifest.get("artifact") == _ARTIFACT
        and candidate_manifest.get("status") == _OUTPUT_STATUS
        and candidate_manifest.get("git_commit") == expected_git_commit
        and candidate_manifest.get("config_sha256") == config.sha256
        and candidate_manifest.get("methods") == list(_METHODS)
        and candidate_manifest.get("promotion_reference") == _PROMOTION_REFERENCE
        and candidate_manifest.get("promotion_candidate") == _PROMOTION_CANDIDATE
        and candidate_manifest.get("production_eligible") is False
        and candidate_manifest.get("production_policy") == _PRODUCTION_POLICY
        and production_weights == {"apex": 0.0, "esm": 0.0}
        and apex
        == {
            "accepted_as_input": False,
            "present_in_feature_vectors": False,
            "production_weight": 0.0,
        }
        and isinstance(runtime, dict)
        and runtime.get("python") == platform.python_version()
        and runtime.get("numpy") == np.__version__
        and isinstance(input_hashes, dict)
        and input_hashes.get("code_manifest") == code_manifest.sha256
        and input_hashes.get("frozen_input_manifest") == frozen_manifest.sha256,
        "candidate identity, runtime, or no-APEX policy differs",
    )
    _require(
        candidate_manifest.get("code_attestation") == code_attestation,
        "candidate code attestation differs from independently verified manifest",
    )

    predictions, fold_models, nested_fits = _reconstruct_predictions(
        examples,
        accepted,
        table,
        base_config=base_config,
        config=config,
    )
    metrics = _metrics_document(predictions, config=config)
    expected_payloads = _expected_semantic_payloads(
        predictions=predictions,
        fold_models=fold_models,
        nested_fits=nested_fits,
        coverage=coverage,
        metrics=metrics,
        config=config,
        git_commit=expected_git_commit,
        code_manifest=code_manifest,
        frozen_manifest=frozen_manifest,
        code_attestation=code_attestation,
        base=base,
        base_receipt=base_receipt,
        base_manifest=base_manifest,
        split_receipt=split_receipt,
        embedding=embedding,
        embedding_receipt=embedding_receipt,
        embedding_manifest=embedding_manifest,
        embedding_receipt_document=embedding_receipt_document,
        table=table,
    )
    for name, expected in expected_payloads.items():
        _require(
            producer.snapshots[f"ensemble/{name}"].payload == expected,
            f"producer {name} differs from independent reconstruction",
        )
    semantic_hashes = {
        name: _sha256(payload) for name, payload in sorted(expected_payloads.items())
    }
    expected_top = _sha_manifest_bytes(semantic_hashes)
    _require(
        producer.snapshots["ensemble/SHA256SUMS"].payload == expected_top,
        "producer semantic top manifest differs from reconstructed bytes",
    )
    expected_outer_hashes = {
        "CODE_SHA256SUMS": code_manifest.sha256,
        "FROZEN_INPUT_SHA256SUMS": frozen_manifest.sha256,
        "ensemble/SHA256SUMS": producer.snapshots["ensemble/SHA256SUMS"].sha256,
        **{f"ensemble/{name}": digest for name, digest in semantic_hashes.items()},
    }
    _require(
        producer.snapshots["SHA256SUMS"].payload == _sha_manifest_bytes(expected_outer_hashes),
        "producer outer top manifest differs from reconstructed publication tree",
    )

    forbidden = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(isinstance(prefix, str) and bool(prefix), "forbidden prefix is empty")
        forbidden.append(prefix.encode("utf-8"))
    _scan_forbidden(
        [producer.snapshots[name].payload for name in sorted(_PRODUCER_FILES)], forbidden
    )
    for snapshot in (
        *base.all_snapshots,
        *embedding.all_snapshots,
        *producer.all_snapshots,
        base_receipt,
        embedding_receipt,
        config_snapshot,
        base_config_snapshot,
        embedding_config_snapshot,
        code_manifest,
        frozen_manifest,
    ):
        _assert_unchanged(snapshot, label=snapshot.path.name)
    for snapshot in (base_receipt, embedding_receipt, code_manifest, frozen_manifest):
        _require(
            snapshot.path.stat().st_mode & 0o222 == 0,
            f"read-only trust input became writable: {snapshot.path.name}",
        )
    _verify_tree_bytes(base.root / "0", base.root / "1", expected=_BASE_TREE_FILES)
    _verify_tree_bytes(embedding.root / "0", embedding.root / "1", expected=_EMBEDDING_TREE_FILES)
    _verify_tree_bytes(producer.root / "0", producer.root / "1", expected=_PRODUCER_FILES)
    _verify_complete_twin_tree(base.root, expected=_BASE_TREE_FILES, label="accepted Gate-1")
    _verify_complete_twin_tree(
        embedding.root, expected=_EMBEDDING_TREE_FILES, label="accepted embeddings"
    )
    _verify_complete_twin_tree(
        producer.root,
        expected=_PRODUCER_FILES,
        label="nested ensemble producer",
        required_checksum_marker_mode=0o444,
    )
    _verify_repository(repository, expected_git_commit)
    _require(
        _verify_execution_source(repository, expected_git_commit) == verifier_sha256,
        "verifier source changed during reconstruction",
    )

    primary = cast(dict[str, object], metrics["primary_comparison"])
    decision = cast(
        dict[str, object], cast(dict[str, object], metrics["promotion_rule"])["decision"]
    )
    return {
        "schema_version": 1,
        "artifact": f"{_ARTIFACT}_independent_verification",
        "status": "passed",
        "acceptance_scope": "development_candidate_for_untouched_external_validation_only",
        "production_eligible": False,
        "statistical_gate_passed": metrics["statistical_gate_passed"],
        "production_weights": {"apex": 0.0, "esm": 0.0},
        "apex": {
            "accepted_as_input": False,
            "present_in_feature_vectors": False,
            "production_weight": 0.0,
        },
        "git_commit": expected_git_commit,
        "config_sha256": config.sha256,
        "verifier_sha256": verifier_sha256,
        "producer": {
            "job_id": int(producer.root.name),
            "publication_top_manifest_sha256": producer.snapshots["SHA256SUMS"].sha256,
            "semantic_top_manifest_sha256": producer.snapshots["ensemble/SHA256SUMS"].sha256,
            "production_handshake": producer.handshake,
            "code_manifest_sha256": code_manifest.sha256,
            "frozen_input_manifest_sha256": frozen_manifest.sha256,
        },
        "accepted_inputs": {
            "gate1_publication_top_sha256": base.snapshots["SHA256SUMS"].sha256,
            "gate1_independent_receipt_sha256": base_receipt.sha256,
            "embedding_publication_top_sha256": embedding.snapshots["SHA256SUMS"].sha256,
            "embedding_independent_receipt_sha256": embedding_receipt.sha256,
            "embedding_npy_sha256": table.matrix_sha256,
            "embedding_tensor_data_sha256": table.tensor_data_sha256,
        },
        "reconstruction": {
            "contexts": len(examples),
            "prediction_rows": len(predictions),
            "outer_esm_fits": config.expected_outer_esm_fits,
            "nested_esm_fits": config.expected_nested_esm_fits,
            "outer_stack_fits": config.expected_outer_stack_fits,
            "ordered_outer_inner_splits": config.expected_ordered_outer_inner_splits,
            "nested_feature_rows": config.expected_nested_feature_rows,
            "frozen_input_entries": len(frozen_hashes),
            "primary_comparison": primary,
            "promotion_decision": decision,
        },
        "artifact_sha256": {f"ensemble/{name}": digest for name, digest in semantic_hashes.items()}
        | {
            "ensemble/SHA256SUMS": producer.snapshots["ensemble/SHA256SUMS"].sha256,
            "SHA256SUMS": producer.snapshots["SHA256SUMS"].sha256,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "checks": {
            "accepted_gate1_trust_chain_valid": True,
            "accepted_embedding_trust_chain_valid": True,
            "accepted_twin_task_trees_read_only_and_identical": True,
            "producer_twin_task_trees_read_only_and_identical": True,
            "producer_code_manifest_reconstructed": True,
            "producer_frozen_input_manifest_reconstructed": True,
            "repository_commit_clean_and_synchronized": True,
            "exact_sequence_embedding_join_reconstructed": True,
            "descriptor_outer_and_nested_refits_reconstructed": True,
            "homology_knn_outer_and_nested_refits_reconstructed": True,
            "all_25_esm_heads_reconstructed": True,
            "all_five_nested_stacks_reconstructed": True,
            "ordinary_oof_excluded_from_stack_training": True,
            "all_seven_method_predictions_reconstructed": True,
            "all_metrics_reconstructed": True,
            "shared_union_component_bootstrap_reconstructed": True,
            "all_semantic_artifacts_byte_exact": True,
            "apex_absent_and_weights_zero": True,
            "scratch_and_absolute_paths_absent": True,
            "receipt_scope_is_nonproduction": True,
        },
    }


def _write_receipt(path: Path, receipt: Mapping[str, object], *, protected: Sequence[Path]) -> None:
    requested = Path(path).absolute()
    _require(requested.name not in {"", ".", ".."}, "verification receipt has no filename")
    _require_no_symlink(requested.parent, label="receipt parent")
    _require(not os.path.lexists(requested), f"refusing to overwrite receipt: {requested}")
    for root in protected:
        protected_root = Path(root).resolve(strict=True)
        _require(
            requested != protected_root and not requested.is_relative_to(protected_root),
            "verification receipt must be outside protected inputs",
        )
    payload = _pretty_json_bytes(receipt)
    _scan_forbidden([payload], [b"/lustre/scratch/users/"])
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    _require(not os.path.lexists(target), f"refusing to overwrite receipt: {target}")
    for root in protected:
        protected_root = Path(root).resolve(strict=True)
        _require(
            target != protected_root and not target.is_relative_to(protected_root),
            "verification receipt must be outside protected inputs",
        )
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=parent)
    staging = Path(staging_name)
    linked = False
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o444)
        os.link(staging, target)
        linked = True
        source_stat = staging.stat(follow_symlinks=False)
        target_stat = target.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(target_stat.st_mode)
            and os.path.samestat(source_stat, target_stat)
            and stat.S_IMODE(target_stat.st_mode) == 0o444
            and target.read_bytes() == payload,
            "verification receipt publication changed before commit",
        )
        parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        committed = True
    except BaseException:
        if linked:
            try:
                if target.exists() and os.path.samestat(
                    staging.stat(follow_symlinks=False), target.stat(follow_symlinks=False)
                ):
                    target.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            if not committed:
                raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-twin-root", type=Path, required=True)
    parser.add_argument("--producer-code-manifest", type=Path, required=True)
    parser.add_argument("--producer-frozen-input-manifest", type=Path, required=True)
    parser.add_argument("--base-twin-root", type=Path, required=True)
    parser.add_argument("--base-independent-receipt", type=Path, required=True)
    parser.add_argument("--embedding-twin-root", type=Path, required=True)
    parser.add_argument("--embedding-independent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--forbidden-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protected = [
        _resolved_directory(args.producer_twin_root, label="producer twin root"),
        _resolved_directory(args.base_twin_root, label="accepted Gate-1 twin root"),
        _resolved_directory(args.embedding_twin_root, label="accepted embedding twin root"),
        _resolved_directory(args.repo_root, label="repository root"),
        _read_snapshot(args.producer_code_manifest, label="producer code manifest").path,
        _read_snapshot(
            args.producer_frozen_input_manifest, label="producer frozen-input manifest"
        ).path,
        _read_snapshot(args.base_independent_receipt, label="base independent receipt").path,
        _read_snapshot(
            args.embedding_independent_receipt, label="embedding independent receipt"
        ).path,
    ]
    prospective = Path(args.output).absolute().resolve(strict=False)
    for root in protected:
        _require(
            prospective != root and not prospective.is_relative_to(root),
            "verification receipt must be outside protected inputs",
        )
    receipt = verify_esm_union_ensemble_oof_twins(
        producer_twin_root=args.producer_twin_root,
        producer_code_manifest=args.producer_code_manifest,
        producer_frozen_input_manifest=args.producer_frozen_input_manifest,
        base_twin_root=args.base_twin_root,
        base_independent_receipt=args.base_independent_receipt,
        embedding_twin_root=args.embedding_twin_root,
        embedding_independent_receipt=args.embedding_independent_receipt,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    _write_receipt(Path(args.output), receipt, protected=protected)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by Slurm CLI
    raise SystemExit(main())

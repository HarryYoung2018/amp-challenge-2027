"""Fit the frozen nested ESM union ensemble on accepted Gate-1 evidence.

The command consumes two complete, immutable two-node publications: the
accepted context-level union Gate-1 benchmark and the independently verified
952 by 320 ESM2 embedding matrix.  Base probabilities used to train a stack
are always regenerated with both the outer and row-specific inner folds
excluded.  Accepted ordinary OOF probabilities are admitted only for the
outer query rows, after the corresponding base refits reproduce them exactly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.benchmarks.apex_union_oof import (
    _BASE_TREE_FILES,
    Example,
    Snapshot,
    _assignment_sha256,
    _file_sha256,
    _parse_json_object,
    _parse_sha256_manifest,
    _read_examples,
    _sequence_set_sha256,
    _strict_csv_rows,
    _verify_base_documents,
    _verify_folds,
)
from amp_challenge.benchmarks.ensemble_oof import _audit_descriptor_fit
from amp_challenge.benchmarks.oracle_gate1_union import (
    Gate1UnionConfig,
    binary_metrics,
)
from amp_challenge.benchmarks.oracle_gate1_union import (
    _config_from_payload as _base_config_from_payload,
)
from amp_challenge.models.oracle_baselines import (
    DescriptorLogisticOracle,
    HomologyKnnOracle,
    OracleInput,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

FloatArray = NDArray[np.float64]
GramClass = Literal["negative", "positive"]
DirectoryIdentity = tuple[tuple[str, tuple[int, int, int, int], str], ...]

CONFIG_SHA256 = "53f045b05ead39147f7cea0c9d09da8ea473e2c1d3c74ef8f5f4dfcce1935b7f"
ARTIFACT = "esm_union_nested_ensemble_oof_v1"
OUTPUT_STATUS = "repeatedly_inspected_development_evidence"
METHODS = (
    "descriptor_logistic",
    "homology_knn",
    "equal_weight_ensemble",
    "esm2_t6_8m_target_gram_logistic",
    "descriptor_esm_probability_half_blend",
    "base_family_esm_probability_half_blend",
    "nested_union_base_esm_logistic_stack_v1",
)
BASE_MODELS = METHODS[:3]
ESM_MODEL = METHODS[3]
DESCRIPTOR_ESM_BLEND = METHODS[4]
BASE_ESM_BLEND = METHODS[5]
STACK_MODEL = METHODS[6]
STACK_FEATURES = ("descriptor_logistic", "homology_knn", ESM_MODEL)
PROMOTION_REFERENCE = "descriptor_logistic"
PROMOTION_CANDIDATE = STACK_MODEL

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_ABSOLUTE_BYTES_RE = re.compile(
    rb"(?:/home/|/lustre/|/tmp/|/scratch/|/mnt/|file://|(?<![A-Za-z0-9])[A-Za-z]:[\\\\/])"
)
_HANDSHAKE_FILES = frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"})
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
_EMBEDDING_TREE_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "embeddings/SHA256SUMS",
        "embeddings/embedding_index.csv",
        "embeddings/embedding_manifest.json",
        "embeddings/embeddings.npy",
    }
)
_EMBEDDING_SEMANTIC_FILES = frozenset(
    {"SHA256SUMS", "embedding_index.csv", "embedding_manifest.json", "embeddings.npy"}
)
_FROZEN_INPUT_ENTRIES = frozenset(
    {
        *(f"base/0/{name}" for name in _BASE_TREE_FILES),
        *(f"base/1/{name}" for name in _BASE_TREE_FILES),
        *(f"base/node-receipts/{name}" for name in _HANDSHAKE_FILES),
        "base-independent-receipt.json",
        *(f"embedding/0/{name}" for name in _EMBEDDING_TREE_FILES),
        *(f"embedding/1/{name}" for name in _EMBEDDING_TREE_FILES),
        *(f"embedding/node-receipts/{name}" for name in _HANDSHAKE_FILES),
        "embedding-independent-receipt.json",
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
_OUTPUT_FILES = frozenset(
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
_LOGICAL_CONFIG_PATH = "configs/evaluation/esm_union_ensemble_oof_v1.toml"
_BASE_CONFIG_PATH = "configs/benchmarks/oracle_gate1_union_v1.toml"
_EMBEDDING_CONFIG_PATH = "configs/models/esm2_union_embeddings_v1.toml"
_EXECUTING_MODULE_PATH = "src/amp_challenge/benchmarks/esm_union_ensemble_oof.py"
_FIXED_CODE_PATHS = frozenset(
    {
        _LOGICAL_CONFIG_PATH,
        _BASE_CONFIG_PATH,
        _EMBEDDING_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
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


@dataclass(frozen=True, slots=True)
class EsmUnionConfig:
    path: Path
    methods: tuple[str, ...]
    promotion_reference: str
    promotion_candidate: str
    stack_features: tuple[str, ...]
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

    # These aliases intentionally make the accepted Gate-1 parser usable
    # without relaxing its union-panel census checks.
    @property
    def expected_modeled_sequences(self) -> int:
        return self.expected_sequences

    @property
    def expected_modeled_homology_components(self) -> int:
        return self.expected_homology_components

    @property
    def expected_modeled_union_components(self) -> int:
        return self.expected_union_components

    @property
    def expected_union_sequence_ids_sha256(self) -> str:
        return self.expected_sequence_ids_sha256

    @property
    def expected_base_config_sha256(self) -> str:
        return self.base_config_sha256

    @property
    def target_by_name(self) -> dict[str, str]:
        return {name: name for name in self.canonical_targets}


@dataclass(frozen=True, slots=True)
class TwinEvidence:
    root: Path
    task_root: Path
    snapshots: Mapping[str, Snapshot]
    handshake: Mapping[str, object]
    all_snapshots: tuple[Snapshot, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingTable:
    matrix: NDArray[np.float32]
    row_by_sequence_id: Mapping[str, int]
    index_snapshot: Snapshot
    matrix_snapshot: Snapshot
    tensor_data_sha256: str


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


@dataclass(frozen=True, slots=True)
class EsmUnionExecution:
    output_dir: Path
    predictions_path: Path
    fold_models_path: Path
    nested_fits_path: Path
    embedding_coverage_path: Path
    metrics_path: Path
    manifest_path: Path
    top_manifest_path: Path
    examples: int
    prediction_rows: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    for item in (candidate, *candidate.parents):
        _require(not item.is_symlink(), f"{label} must not traverse a symbolic link")


def _resolve(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _reject_symlink_chain(requested, label=label)
    return requested.resolve(strict=True)


def _snapshot(path: str | Path, *, label: str) -> Snapshot:
    source = _resolve(path, label=label)
    before = source.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    payload = source.read_bytes()
    after = source.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return Snapshot(source, payload, _sha256(payload), _fingerprint(after))


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = _snapshot(snapshot.path, label=label)
    _require(
        current.sha256 == snapshot.sha256 and current.fingerprint == snapshot.fingerprint,
        f"{label} changed during execution",
    )


def _tree_inventory(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for item in root.rglob("*"):
        logical = item.relative_to(root).as_posix()
        _require(not item.is_symlink(), f"tree contains symbolic link {logical!r}")
        mode = item.stat().st_mode
        if stat.S_ISREG(mode):
            files.add(logical)
        elif stat.S_ISDIR(mode):
            directories.add(logical)
        else:
            raise ValueError(f"tree contains non-file entry {logical!r}")
    return frozenset(files), frozenset(directories)


def _require_immutable_tree(root: Path, *, label: str) -> None:
    files, directories = _tree_inventory(root)
    for logical in ("", *sorted(files), *sorted(directories)):
        path = root if logical == "" else root / logical
        _require(path.stat().st_mode & 0o222 == 0, f"{label} remains writable: {logical or '.'}")


def _expected_directories(files: Iterable[str]) -> frozenset[str]:
    return frozenset(
        parent.as_posix()
        for name in files
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    )


def _verify_manifest_tree(
    root: Path,
    *,
    expected_files: frozenset[str],
    expected_top_sha256: str,
    label: str,
) -> dict[str, Snapshot]:
    files, directories = _tree_inventory(root)
    _require(files == expected_files, f"{label} file inventory changed")
    _require(directories == _expected_directories(expected_files), f"{label} directories changed")
    snapshots = {
        logical: _snapshot(root / logical, label=f"{label} {logical}")
        for logical in sorted(expected_files)
    }
    top = snapshots["SHA256SUMS"]
    _require(top.sha256 == expected_top_sha256, f"{label} top manifest hash changed")
    entries = _parse_sha256_manifest(top.payload, name=f"{label} top manifest")
    _require(set(entries) == set(expected_files) - {"SHA256SUMS"}, f"{label} top coverage changed")
    for logical, digest in entries.items():
        _require(snapshots[logical].sha256 == digest, f"{label} checksum mismatch for {logical}")
    return snapshots


def _verify_handshake(
    root: Path, *, expected_job_id: int, label: str
) -> tuple[dict[str, object], tuple[Snapshot, ...]]:
    receipts_root = root / "node-receipts"
    files, directories = _tree_inventory(receipts_root)
    _require(files == _HANDSHAKE_FILES and not directories, f"{label} handshake inventory changed")
    receipts: dict[int, Snapshot] = {}
    nodes: dict[int, str] = {}
    for task in (0, 1):
        item = _snapshot(receipts_root / f"{task}.receipt", label=f"{label} receipt {task}")
        try:
            lines = item.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} receipt is not UTF-8") from error
        _require(
            item.payload.endswith(b"\n")
            and b"\r" not in item.payload
            and len(lines) == 3
            and lines[0] == f"array_job_id={expected_job_id}"
            and lines[1] == f"array_task_id={task}"
            and lines[2].startswith("node_name="),
            f"{label} receipt {task} changed",
        )
        node = lines[2].removeprefix("node_name=")
        _require(_NODE_RE.fullmatch(node) is not None, f"{label} receipt node is unsafe")
        receipts[task] = item
        nodes[task] = node
    _require(nodes[0] != nodes[1], f"{label} twins did not run on distinct nodes")
    receipt_hashes = {str(task): receipts[task].sha256 for task in (0, 1)}
    acknowledgement_hashes: dict[str, str] = {}
    acknowledgements: list[Snapshot] = []
    for task in (0, 1):
        item = _snapshot(receipts_root / f"{task}.ack", label=f"{label} acknowledgement {task}")
        expected = f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode(
            "ascii"
        )
        _require(item.payload == expected, f"{label} acknowledgement {task} changed")
        acknowledgement_hashes[str(task)] = item.sha256
        acknowledgements.append(item)
    return (
        {
            "distinct_nodes": True,
            "bidirectional_acknowledgement": True,
            "receipt_sha256": receipt_hashes,
            "acknowledgement_sha256": acknowledgement_hashes,
        },
        (*receipts.values(), *acknowledgements),
    )


def _read_twins(
    root_value: str | Path,
    *,
    expected_job_id: int,
    expected_files: frozenset[str],
    expected_top_sha256: str,
    label: str,
) -> TwinEvidence:
    root = _resolve(root_value, label=f"{label} twin root")
    _require(root.is_dir() and root.name == str(expected_job_id), f"{label} job root changed")
    expected_all_files = frozenset(
        {
            *(f"0/{name}" for name in expected_files),
            *(f"1/{name}" for name in expected_files),
            *(f"node-receipts/{name}" for name in _HANDSHAKE_FILES),
        }
    )
    files, directories = _tree_inventory(root)
    _require(files == expected_all_files, f"{label} complete twin inventory changed")
    _require(
        directories == _expected_directories(expected_all_files),
        f"{label} twin directories changed",
    )
    _require_immutable_tree(root / "0", label=f"{label} task 0")
    _require_immutable_tree(root / "1", label=f"{label} task 1")
    _require_immutable_tree(root / "node-receipts", label=f"{label} handshake")
    left = _verify_manifest_tree(
        root / "0",
        expected_files=expected_files,
        expected_top_sha256=expected_top_sha256,
        label=f"{label} task 0",
    )
    right = _verify_manifest_tree(
        root / "1",
        expected_files=expected_files,
        expected_top_sha256=expected_top_sha256,
        label=f"{label} task 1",
    )
    for logical in sorted(expected_files):
        _require(
            left[logical].payload == right[logical].payload, f"{label} twins differ at {logical}"
        )
    handshake, handshake_snapshots = _verify_handshake(
        root, expected_job_id=expected_job_id, label=label
    )
    return TwinEvidence(
        root=root,
        task_root=root / "0",
        snapshots=left,
        handshake=handshake,
        all_snapshots=(*left.values(), *right.values(), *handshake_snapshots),
    )


def _assert_twin_tree_unchanged(
    twin: TwinEvidence, *, expected_files: frozenset[str], label: str
) -> None:
    expected_all_files = frozenset(
        {
            *(f"0/{name}" for name in expected_files),
            *(f"1/{name}" for name in expected_files),
            *(f"node-receipts/{name}" for name in _HANDSHAKE_FILES),
        }
    )
    files, directories = _tree_inventory(twin.root)
    _require(files == expected_all_files, f"{label} final twin inventory changed")
    _require(
        directories == _expected_directories(expected_all_files),
        f"{label} final twin directories changed",
    )
    for task in ("0", "1"):
        _require_immutable_tree(twin.root / task, label=f"{label} task {task} after fitting")
    _require_immutable_tree(twin.root / "node-receipts", label=f"{label} handshake after fitting")


def _expect_sha(snapshot: Snapshot, expected: str, *, label: str) -> None:
    _require(snapshot.sha256 == expected, f"{label} differs from the frozen hash")


def _as_tuple_int(value: object, *, label: str, length: int) -> tuple[int, ...]:
    _require(isinstance(value, list) and len(value) == length, f"{label} has wrong length")
    _require(all(type(item) is int and item > 0 for item in value), f"{label} is invalid")
    return tuple(cast(list[int], value))


def _hash_field(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"bad {label}")
    return cast(str, value)


def _config_from_snapshot(snapshot: Snapshot) -> EsmUnionConfig:
    _expect_sha(snapshot, CONFIG_SHA256, label="ESM union ensemble config")
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("ESM union ensemble config is not valid TOML") from error
    _require(
        raw.get("schema_version") == 1 and raw.get("artifact") == ARTIFACT,
        "config identity changed",
    )
    _require(tuple(raw.get("methods", ())) == METHODS, "config methods changed")
    _require(
        tuple(raw.get("stack_features", ())) == STACK_FEATURES, "config stack features changed"
    )
    _require(raw.get("promotion_reference") == PROMOTION_REFERENCE, "promotion reference changed")
    _require(raw.get("promotion_candidate") == PROMOTION_CANDIDATE, "promotion candidate changed")
    fixed_literals = {
        "stack_input_transform": "clipped_logit",
        "stack_scaling": "column_mean_std_on_nested_outer_training_features_only",
        "descriptor_esm_probability_half_blend": "0.5*descriptor_logistic+0.5*esm2_t6_8m_target_gram_logistic",
        "base_family_esm_probability_half_blend": "0.5*equal_weight_ensemble+0.5*esm2_t6_8m_target_gram_logistic",
        "esm_context_features": "fixed_canonical_target_one_hot_plus_gram_one_hot",
        "esm_feature_order": "intercept_then_320_standardized_esm_then_targets_in_config_order_then_gram_classes_in_config_order",
        "one_hot_convention": "full_one_hot_columns",
        "ridge_penalty": "intercept_unpenalized_all_other_coefficients_equal_l2",
        "fit_weighting": "context_equal_v1",
        "scaler_weighting": "context_equal_v1",
        "stack_scaler_weighting": "context_equal_v1",
        "unseen_target_policy": "known_target_column_coefficient_remains_zero_when_absent_from_training",
        "inner_base_feature_source": "refit_on_folds_excluding_outer_and_inner_never_accepted_oof",
        "outer_base_feature_source": "accepted_outer_oof_after_exact_refit_reproduction_check",
        "bootstrap_unit": "union_component_resample_context_multiplicity_v1",
    }
    _require(
        all(raw.get(key) == value for key, value in fixed_literals.items()),
        "config policy literal changed",
    )
    _require(
        raw.get("source_observations_used_as_weight") is False,
        "source observations weighting changed",
    )
    _require(raw.get("hyperparameter_search_allowed") is False, "hyperparameter policy changed")
    _require(raw.get("similarity_gate_allowed") is False, "similarity gate policy changed")
    _require(raw.get("report_component_equal_metrics") is True, "component metric policy changed")
    _require(
        raw.get("report_leave_largest_union_component_out") is True, "sensitivity policy changed"
    )
    production = raw.get("production_policy")
    _require(production == _PRODUCTION_POLICY, "production policy changed")
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
    hashes = {name: _hash_field(raw.get(name), label=name) for name in hash_names}
    commit = raw.get("embedding_protocol_git_commit")
    _require(
        isinstance(commit, str) and _GIT_RE.fullmatch(commit) is not None,
        "bad embedding protocol commit",
    )
    targets = tuple(cast(list[str], raw.get("canonical_targets")))
    grams = tuple(cast(list[str], raw.get("gram_classes")))
    _require(
        len(targets) == 7 and len(set(targets)) == 7 and all(targets), "canonical targets changed"
    )
    _require(grams == ("negative", "positive"), "Gram class order changed")
    return EsmUnionConfig(
        path=snapshot.path,
        methods=METHODS,
        promotion_reference=PROMOTION_REFERENCE,
        promotion_candidate=PROMOTION_CANDIDATE,
        stack_features=STACK_FEATURES,
        probability_clip=float(raw["probability_clip"]),
        canonical_targets=targets,
        gram_classes=grams,
        esm_l2=float(raw["esm_l2"]),
        esm_prior_strength=float(raw["esm_prior_strength"]),
        esm_max_iterations=int(raw["esm_max_iterations"]),
        esm_tolerance=float(raw["esm_tolerance"]),
        stack_l2=float(raw["stack_l2"]),
        stack_prior_strength=float(raw["stack_prior_strength"]),
        stack_max_iterations=int(raw["stack_max_iterations"]),
        stack_tolerance=float(raw["stack_tolerance"]),
        folds=int(raw["folds"]),
        homology_identity_threshold=float(raw["homology_identity_threshold"]),
        similarity_bin_edges=tuple(
            float(item) for item in cast(list[float], raw["similarity_bin_edges"])
        ),
        calibration_bins=int(raw["calibration_bins"]),
        bootstrap_replicates=int(raw["bootstrap_replicates"]),
        bootstrap_seed=int(raw["bootstrap_seed"]),
        base_producer_job_id=int(raw["base_producer_job_id"]),
        base_audit_job_id=int(raw["base_audit_job_id"]),
        embedding_protocol_git_commit=commit,
        embedding_producer_job_id=int(raw["embedding_producer_job_id"]),
        embedding_audit_job_id=int(raw["embedding_audit_job_id"]),
        embedding_model=str(raw["embedding_model"]),
        embedding_representation_layer=int(raw["embedding_representation_layer"]),
        embedding_dimension=int(raw["embedding_dimension"]),
        embedding_dtype=str(raw["embedding_dtype"]),
        embedding_records=int(raw["embedding_records"]),
        expected_examples=int(raw["expected_examples"]),
        expected_source_observations=int(raw["expected_source_observations"]),
        expected_sequences=int(raw["expected_sequences"]),
        expected_positive_examples=int(raw["expected_positive_examples"]),
        expected_negative_examples=int(raw["expected_negative_examples"]),
        expected_homology_components=int(raw["expected_homology_components"]),
        expected_union_components=int(raw["expected_union_components"]),
        expected_examples_by_fold=_as_tuple_int(
            raw["expected_examples_by_fold"], label="examples by fold", length=5
        ),
        expected_positive_examples_by_fold=_as_tuple_int(
            raw["expected_positive_examples_by_fold"], label="positives by fold", length=5
        ),
        expected_negative_examples_by_fold=_as_tuple_int(
            raw["expected_negative_examples_by_fold"], label="negatives by fold", length=5
        ),
        expected_outer_folds=int(raw["expected_outer_folds"]),
        expected_ordered_outer_inner_splits=int(raw["expected_ordered_outer_inner_splits"]),
        expected_nested_feature_rows=int(raw["expected_nested_feature_rows"]),
        expected_outer_esm_fits=int(raw["expected_outer_esm_fits"]),
        expected_nested_esm_fits=int(raw["expected_nested_esm_fits"]),
        expected_outer_stack_fits=int(raw["expected_outer_stack_fits"]),
        expected_largest_union_component_contexts=int(
            raw["expected_largest_union_component_contexts"]
        ),
        expected_union_components_over_100_contexts=int(
            raw["expected_union_components_over_100_contexts"]
        ),
        promotion_auc_delta_lower_minimum=float(raw["promotion_auc_delta_lower_minimum"]),
        promotion_brier_delta_upper_maximum=float(raw["promotion_brier_delta_upper_maximum"]),
        promotion_log_loss_delta_upper_maximum=float(raw["promotion_log_loss_delta_upper_maximum"]),
        production_policy=cast(Mapping[str, object], production),
        **hashes,
    )


def load_config(path: str | Path) -> EsmUnionConfig:
    """Load the exact precommitted ESM union ensemble configuration."""

    return _config_from_snapshot(_snapshot(path, label="ESM union ensemble config"))


def _verify_code_manifest(
    manifest: Snapshot,
    *,
    config: Snapshot,
    base_config: Snapshot,
    embedding_config: Snapshot,
) -> tuple[dict[str, object], tuple[Snapshot, ...]]:
    entries = _parse_sha256_manifest(manifest.payload, name="code manifest")
    repository = Path(__file__).resolve(strict=True).parents[3]
    package_root = repository / "src" / "amp_challenge"
    python_paths = {
        item.relative_to(repository).as_posix()
        for item in package_root.rglob("*.py")
        if item.is_file() and not item.is_symlink()
    }
    _require(
        not any(item.is_symlink() for item in package_root.rglob("*.py")),
        "repository Python source is symbolic",
    )
    expected = python_paths | set(_FIXED_CODE_PATHS)
    _require(set(entries) == expected, "code manifest inventory changed")
    supplied = {
        _LOGICAL_CONFIG_PATH: config,
        _BASE_CONFIG_PATH: base_config,
        _EMBEDDING_CONFIG_PATH: embedding_config,
    }
    snapshots: list[Snapshot] = []
    for logical in sorted(expected):
        item = supplied.get(logical)
        if item is None:
            item = _snapshot(repository / logical, label=f"repository code {logical}")
            snapshots.append(item)
        _require(item.sha256 == entries[logical], f"code manifest mismatch for {logical}")
    return (
        {
            "schema_version": 1,
            "code_manifest_sha256": manifest.sha256,
            "inventory_entries": len(entries),
            "executing_module": {
                "logical_path": _EXECUTING_MODULE_PATH,
                "sha256": entries[_EXECUTING_MODULE_PATH],
            },
            "config": {"logical_path": _LOGICAL_CONFIG_PATH, "sha256": config.sha256},
            "base_config": {"logical_path": _BASE_CONFIG_PATH, "sha256": base_config.sha256},
            "embedding_config": {
                "logical_path": _EMBEDDING_CONFIG_PATH,
                "sha256": embedding_config.sha256,
            },
        },
        tuple(snapshots),
    )


def _verify_frozen_inputs(manifest: Snapshot, snapshots: Mapping[str, Snapshot]) -> None:
    entries = _parse_sha256_manifest(manifest.payload, name="frozen input manifest")
    _require(
        set(snapshots) == set(_FROZEN_INPUT_ENTRIES), "internal frozen input inventory changed"
    )
    _require(set(entries) == set(_FROZEN_INPUT_ENTRIES), "frozen input manifest inventory changed")
    for logical, digest in entries.items():
        _require(
            snapshots[logical].sha256 == digest, f"frozen input checksum mismatch for {logical}"
        )


def _pretty_json(snapshot: Snapshot, *, label: str) -> dict[str, object]:
    value = _parse_json_object(snapshot.payload, name=label)
    expected = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _require(snapshot.payload == expected, f"{label} is not canonical pretty JSON")
    return value


def _verify_embedding_documents(
    twin: TwinEvidence,
    receipt_snapshot: Snapshot,
    *,
    config: EsmUnionConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    snapshots = twin.snapshots
    for logical, expected in (
        ("embeddings/SHA256SUMS", config.embedding_semantic_top_sha256),
        ("embeddings/embedding_index.csv", config.embedding_index_sha256),
        ("embeddings/embedding_manifest.json", config.embedding_manifest_sha256),
        ("embeddings/embeddings.npy", config.embedding_matrix_sha256),
    ):
        _expect_sha(snapshots[logical], expected, label=logical)
    semantic = _parse_sha256_manifest(
        snapshots["embeddings/SHA256SUMS"].payload, name="embedding semantic manifest"
    )
    _require(
        set(semantic) == _EMBEDDING_SEMANTIC_FILES - {"SHA256SUMS"},
        "embedding semantic inventory changed",
    )
    for logical, digest in semantic.items():
        _require(
            snapshots[f"embeddings/{logical}"].sha256 == digest,
            f"embedding semantic checksum mismatch for {logical}",
        )
    manifest = _pretty_json(
        snapshots["embeddings/embedding_manifest.json"], label="embedding manifest"
    )
    tensor = manifest.get("tensor")
    extraction = manifest.get("extraction")
    acceptance = manifest.get("acceptance_scope")
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
        and isinstance(acceptance, dict)
        and acceptance.get("embeddings_verified") is False
        and acceptance.get("ensemble_weight_established") is False,
        "embedding producer manifest contract changed",
    )
    receipt = _pretty_json(receipt_snapshot, label="embedding independent receipt")
    scope = receipt.get("acceptance_scope")
    receipt_tensor = receipt.get("tensor")
    producer = receipt.get("producer")
    checks = receipt.get("checks")
    artifact_sha = receipt.get("artifact_sha256")
    runtime = receipt.get("runtime")
    _require(
        set(receipt) == _EMBEDDING_RECEIPT_FIELDS
        and receipt.get("schema_version") == 1
        and receipt.get("artifact") == "gate1_union_esm2_embeddings_v1_independent_verification"
        and receipt.get("status") == "accepted_for_downstream_embedding_feature_input_only"
        and receipt.get("git_commit") == config.embedding_protocol_git_commit
        and receipt.get("config_sha256") == config.embedding_config_sha256
        and scope == _EMBEDDING_ACCEPTANCE_SCOPE
        and isinstance(receipt_tensor, dict)
        and receipt_tensor.get("npy_sha256") == config.embedding_matrix_sha256
        and receipt_tensor.get("tensor_data_sha256") == config.embedding_tensor_data_sha256
        and receipt_tensor.get("shape") == [config.embedding_records, config.embedding_dimension]
        and receipt_tensor.get("dtype") == config.embedding_dtype
        and receipt_tensor.get("byte_order") == "little_endian"
        and receipt_tensor.get("layout") == "C_contiguous"
        and receipt_tensor.get("npy_version") == [1, 0]
        and isinstance(runtime, dict)
        and manifest.get("runtime") == runtime
        and isinstance(producer, dict)
        and producer.get("publication_top_manifest_sha256")
        == config.embedding_publication_top_sha256
        and producer.get("embedding_top_manifest_sha256") == config.embedding_semantic_top_sha256
        and producer.get("candidate_manifest_sha256") == config.embedding_manifest_sha256
        and producer.get("code_manifest_sha256") == snapshots["CODE_SHA256SUMS"].sha256
        and producer.get("frozen_input_manifest_sha256")
        == snapshots["FROZEN_INPUT_SHA256SUMS"].sha256
        and producer.get("production_handshake") == twin.handshake
        and isinstance(checks, dict)
        and set(checks) == _EMBEDDING_RECEIPT_CHECKS
        and all(value is True for value in checks.values()),
        "embedding independent acceptance receipt changed",
    )
    _require(
        isinstance(artifact_sha, dict)
        and artifact_sha.get("SHA256SUMS") == config.embedding_publication_top_sha256
        and artifact_sha.get("embeddings/SHA256SUMS") == config.embedding_semantic_top_sha256
        and artifact_sha.get("embeddings/embedding_index.csv") == config.embedding_index_sha256
        and artifact_sha.get("embeddings/embedding_manifest.json")
        == config.embedding_manifest_sha256
        and artifact_sha.get("embeddings/embeddings.npy") == config.embedding_matrix_sha256,
        "embedding receipt artifact hashes changed",
    )
    return manifest, receipt


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
    config: EsmUnionConfig,
) -> tuple[EmbeddingTable, dict[str, object]]:
    rows = _strict_csv_rows(
        index_snapshot.payload,
        schema=("row_index", "sequence_id", "sequence", "length"),
        name="embedding index",
    )
    _require(len(rows) == config.embedding_records, "embedding index row count changed")
    row_by_id: dict[str, int] = {}
    sequences: dict[str, str] = {}
    previous: str | None = None
    for index, row in enumerate(rows):
        sequence_id = row["sequence_id"]
        sequence = row["sequence"]
        _require(row["row_index"] == str(index), "embedding row index is not canonical")
        _require(
            _SHA256_RE.fullmatch(sequence_id) is not None and sequence_id not in row_by_id,
            "bad embedding sequence ID",
        )
        _require(previous is None or sequence_id > previous, "embedding index is not sorted")
        _require(
            canonicalize_sequence(sequence) == sequence, "embedding index sequence is noncanonical"
        )
        _require(
            canonical_sequence_id(sequence) == sequence_id,
            "embedding index sequence digest changed",
        )
        _require(row["length"] == str(len(sequence)), "embedding index length changed")
        row_by_id[sequence_id] = index
        sequences[sequence_id] = sequence
        previous = sequence_id
    canonical_index = "row_index,sequence_id,sequence,length\n" + "".join(
        f"{index},{row['sequence_id']},{row['sequence']},{row['length']}\n"
        for index, row in enumerate(rows)
    )
    _require(
        index_snapshot.payload == canonical_index.encode("ascii"),
        "embedding index serialization is noncanonical",
    )
    payload = matrix_snapshot.payload
    _require(
        payload.startswith(b"\x93NUMPY\x01\x00") and len(payload) >= 10,
        "embedding matrix is not NPY v1",
    )
    header_length = struct.unpack("<H", payload[8:10])[0]
    header_end = 10 + header_length
    _require(header_end <= len(payload), "embedding NPY header is truncated")
    header = payload[10:header_end]
    try:
        document = ast.literal_eval(header.decode("latin1").strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValueError("embedding NPY header is invalid") from error
    _require(
        isinstance(document, dict)
        and set(document) == {"descr", "fortran_order", "shape"}
        and document["descr"] == "<f4"
        and document["fortran_order"] is False
        and document["shape"] == (config.embedding_records, config.embedding_dimension),
        "embedding NPY type/layout changed",
    )
    tensor = payload[header_end:]
    _require(
        len(tensor) == config.embedding_records * config.embedding_dimension * 4,
        "embedding tensor size changed",
    )
    _require(
        payload
        == _canonical_npy_bytes(
            tensor, rows=config.embedding_records, columns=config.embedding_dimension
        ),
        "embedding NPY serialization is noncanonical",
    )
    tensor_sha = _sha256(tensor)
    _require(
        tensor_sha == config.embedding_tensor_data_sha256, "embedding tensor data hash changed"
    )
    matrix = np.frombuffer(tensor, dtype="<f4").reshape(
        config.embedding_records, config.embedding_dimension
    )
    _require(np.all(np.isfinite(matrix)), "embedding matrix contains a non-finite value")
    example_sequences: dict[str, str] = {}
    for item in examples:
        previous_sequence = example_sequences.setdefault(item.sequence_id, item.sequence)
        _require(previous_sequence == item.sequence, "one sequence ID maps to multiple sequences")
    _require(set(example_sequences) == set(row_by_id), "embedding/base sequence support differs")
    _require(
        all(sequences[key] == value for key, value in example_sequences.items()),
        "embedding/base sequence bytes differ",
    )
    _require(
        _sequence_set_sha256(row_by_id) == config.expected_sequence_ids_sha256,
        "embedding sequence-set digest changed",
    )
    sequence_to_row_payload = "".join(
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
        "sequence_to_row_sha256": _sha256(sequence_to_row_payload),
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
    _require(
        coverage["joined_vectors_sha256"] == tensor_sha,
        "joined vector order differs from the canonical tensor",
    )
    return EmbeddingTable(matrix, row_by_id, index_snapshot, matrix_snapshot, tensor_sha), coverage


def _model_input(item: Example) -> OracleInput:
    return OracleInput(sequence=item.sequence, strain=item.canonical_target, gram=item.gram)


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
    for attribute in (
        "example_id",
        "sequence_id",
        "homology_component_id",
        "union_component_id",
    ):
        train = {getattr(item, attribute) for item in training}
        test = {getattr(item, attribute) for item in testing}
        _require(train.isdisjoint(test), f"{label} leaks {attribute}")


def _fit_base(
    training: Sequence[Example],
    testing: Sequence[Example],
    *,
    base_config: Gate1UnionConfig,
    label: str,
) -> tuple[dict[str, FloatArray], dict[str, object]]:
    _require_split_separation(training, testing, label=label)
    train_inputs = tuple(_model_input(item) for item in training)
    test_inputs = tuple(_model_input(item) for item in testing)
    labels = np.asarray([item.label for item in training], dtype=np.int64)
    descriptor = DescriptorLogisticOracle(
        l2=base_config.logistic.l2,
        max_iterations=base_config.logistic.max_iterations,
        tolerance=base_config.logistic.tolerance,
        prior_strength=base_config.logistic.prior_strength,
    )
    knn = HomologyKnnOracle(
        neighbors=base_config.knn.neighbors,
        similarity_power=base_config.knn.similarity_power,
        prior_strength=base_config.knn.prior_strength,
        minimum_weight=base_config.knn.minimum_weight,
    )
    descriptor.fit(train_inputs, labels)
    descriptor_audit = _audit_descriptor_fit(
        descriptor, train_inputs, labels.astype(np.float64), description=label
    )
    knn.fit(train_inputs, labels)
    probabilities = {
        descriptor.name: descriptor.predict_proba(test_inputs),
        knn.name: knn.predict_proba(test_inputs),
    }
    for name, values in probabilities.items():
        _require(
            values.shape == (len(testing),) and np.all(np.isfinite(values)),
            f"{label} {name} predictions are invalid",
        )
    return probabilities, {
        "training": _partition_evidence(training),
        "query": _partition_evidence(testing),
        "descriptor_fit": descriptor_audit,
        "homology_knn_fit": {"finite_predictions": True},
    }


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


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
    _require(
        np.all(np.isfinite(design)) and np.all(np.isfinite(labels)), f"{label} design is non-finite"
    )
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
        _require(np.all(np.isfinite(step)), f"{label} produced a non-finite Newton step")
        coefficient -= step
        _require(np.all(np.isfinite(coefficient)), f"{label} coefficients are non-finite")
        if float(np.max(np.abs(step))) <= tolerance:
            return coefficient, iteration
    raise ValueError(f"{label} exhausted its iteration budget before convergence")


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
    config: EsmUnionConfig,
) -> FloatArray:
    standardized = (_embedding_rows(rows, table) - mean) / scale
    targets = np.zeros((len(rows), len(config.canonical_targets)), dtype=np.float64)
    target_index = {name: index for index, name in enumerate(config.canonical_targets)}
    grams = np.zeros((len(rows), len(config.gram_classes)), dtype=np.float64)
    gram_index = {name: index for index, name in enumerate(config.gram_classes)}
    for index, item in enumerate(rows):
        _require(item.canonical_target in target_index, "ESM head received an unknown target")
        _require(item.gram in gram_index, "ESM head received an unknown Gram class")
        targets[index, target_index[item.canonical_target]] = 1.0
        grams[index, gram_index[item.gram]] = 1.0
    design = np.column_stack((np.ones(len(rows)), standardized, targets, grams))
    _require(np.all(np.isfinite(design)), "ESM design contains non-finite values")
    return design


def _fit_esm(
    training: Sequence[Example],
    *,
    table: EmbeddingTable,
    config: EsmUnionConfig,
    outer_fold: int,
    inner_fold: int | None,
) -> tuple[dict[str, object], FloatArray, FloatArray, FloatArray]:
    raw = _embedding_rows(training, table)
    _require(
        raw.shape == (len(training), config.embedding_dimension),
        "ESM training matrix shape changed",
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
    config: EsmUnionConfig,
) -> FloatArray:
    values = np.clip(
        _sigmoid(_esm_design(rows, table, mean, scale, config) @ coefficients),
        config.probability_clip,
        1.0 - config.probability_clip,
    )
    _require(
        values.shape == (len(rows),) and np.all(np.isfinite(values)), "ESM predictions are invalid"
    )
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
    config: EsmUnionConfig,
) -> tuple[dict[str, object], FloatArray, FloatArray, FloatArray]:
    _require(
        set(features) == {item.example_id for item in training},
        f"outer {outer_fold} nested coverage changed",
    )
    matrix = np.asarray([features[item.example_id] for item in training], dtype=np.float64)
    _require(
        matrix.shape == (len(training), len(STACK_FEATURES)) and np.all(np.isfinite(matrix)),
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
            "feature_names": list(STACK_FEATURES),
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
    _require(np.all(np.isfinite(values)), "stack predictions are non-finite")
    return values


def _read_accepted_probabilities(
    snapshot: Snapshot, examples: Sequence[Example]
) -> dict[str, dict[str, float]]:
    rows = _strict_csv_rows(snapshot.payload, schema=_BASE_OOF_SCHEMA, name="accepted Gate-1 OOF")
    expected_ids = {item.example_id for item in examples}
    output: dict[str, dict[str, float]] = {model: {} for model in BASE_MODELS}
    previous: tuple[int, str] | None = None
    order = {model: index for index, model in enumerate(sorted(BASE_MODELS))}
    for row in rows:
        model = row["model"]
        _require(model in output, "accepted OOF contains an unknown model")
        key = (order[model], row["example_id"])
        _require(previous is None or key > previous, "accepted OOF rows are not canonically sorted")
        previous = key
        try:
            probability = float(row["probability"])
        except ValueError as error:
            raise ValueError("accepted OOF probability is not numeric") from error
        _require(
            math.isfinite(probability) and 0.0 <= probability <= 1.0,
            "accepted OOF probability is invalid",
        )
        _require(row["example_id"] not in output[model], "accepted OOF repeats an example")
        output[model][row["example_id"]] = probability
    _require(
        all(set(values) == expected_ids for values in output.values()),
        "accepted OOF support changed",
    )
    for example_id in expected_ids:
        expected = float(
            np.mean(
                np.asarray(
                    [output["descriptor_logistic"][example_id], output["homology_knn"][example_id]],
                    dtype=np.float64,
                )
            )
        )
        _require(
            output["equal_weight_ensemble"][example_id] == expected,
            "accepted equal-weight mean is not exact",
        )
    return output


def _fit_all_predictions(
    examples: Sequence[Example],
    accepted: Mapping[str, Mapping[str, float]],
    table: EmbeddingTable,
    *,
    base_config: Gate1UnionConfig,
    config: EsmUnionConfig,
) -> tuple[tuple[Prediction, ...], dict[str, object], dict[str, object]]:
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    prediction_by_method: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    fold_models: list[dict[str, object]] = []
    split_evidence: list[dict[str, object]] = []
    all_nested_records: list[tuple[int, int, str, tuple[float, ...]]] = []
    nested_row_count = 0
    for outer in range(config.folds):
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
        _require(
            np.array_equal(reproduced_equal, accepted_equal),
            f"outer {outer} equal mean does not reproduce accepted OOF",
        )

        outer_esm_evidence, esm_mean, esm_scale, esm_coefficients = _fit_esm(
            outer_training, table=table, config=config, outer_fold=outer, inner_fold=None
        )
        outer_esm = _predict_esm(outer_query, table, esm_mean, esm_scale, esm_coefficients, config)
        nested: dict[str, tuple[float, ...]] = {}
        inner_entries: list[dict[str, object]] = []
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
            inner_esm_evidence, inner_mean, inner_scale, inner_coefficients = _fit_esm(
                inner_training, table=table, config=config, outer_fold=outer, inner_fold=inner
            )
            esm_values = _predict_esm(
                inner_query, table, inner_mean, inner_scale, inner_coefficients, config
            )
            records: list[tuple[int, int, str, tuple[float, ...]]] = []
            for index, item in enumerate(inner_query):
                feature = (
                    _logit(
                        float(base_values["descriptor_logistic"][index]), config.probability_clip
                    ),
                    _logit(float(base_values["homology_knn"][index]), config.probability_clip),
                    _logit(float(esm_values[index]), config.probability_clip),
                )
                _require(item.example_id not in nested, "nested feature row was generated twice")
                nested[item.example_id] = feature
                records.append((outer, inner, item.example_id, feature))
            all_nested_records.extend(records)
            nested_row_count += len(records)
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
        stack_values = _predict_stack(
            outer_features, stack_mean, stack_scale, stack_coefficients, config.probability_clip
        )
        for index, item in enumerate(outer_query):
            descriptor = accepted["descriptor_logistic"][item.example_id]
            knn = accepted["homology_knn"][item.example_id]
            equal = accepted["equal_weight_ensemble"][item.example_id]
            esm = float(outer_esm[index])
            values = {
                "descriptor_logistic": descriptor,
                "homology_knn": knn,
                "equal_weight_ensemble": equal,
                ESM_MODEL: esm,
                DESCRIPTOR_ESM_BLEND: 0.5 * descriptor + 0.5 * esm,
                BASE_ESM_BLEND: 0.5 * equal + 0.5 * esm,
                STACK_MODEL: float(stack_values[index]),
            }
            for method, probability in values.items():
                _require(
                    math.isfinite(probability) and 0.0 <= probability <= 1.0,
                    f"{method} emitted invalid probability",
                )
                prediction_by_method[method][item.example_id] = probability
        fold_models.append(
            {
                "outer_fold": outer,
                "base_reproduction": base_evidence,
                "accepted_base_oof_exactly_reproduced": True,
                "esm_head": outer_esm_evidence,
                "stack": stack_evidence,
                "outer_query": _partition_evidence(outer_query),
            }
        )
        split_evidence.extend(inner_entries)
    expected_ids = {item.example_id for item in ordered}
    _require(
        all(set(rows) == expected_ids for rows in prediction_by_method.values()),
        "prediction support changed",
    )
    _require(
        len(split_evidence) == config.expected_ordered_outer_inner_splits,
        "nested split count changed",
    )
    _require(
        nested_row_count == config.expected_nested_feature_rows, "nested feature row count changed"
    )
    _require(len(fold_models) == config.expected_outer_folds, "outer fit count changed")
    predictions = tuple(
        Prediction(
            model=method, **asdict(item), probability=prediction_by_method[method][item.example_id]
        )
        for method in METHODS
        for item in ordered
    )
    _require(
        len(predictions) == len(METHODS) * config.expected_examples,
        "prediction row count changed",
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
        "ordered_outer_inner_splits": len(split_evidence),
        "descriptor_refits": len(split_evidence),
        "homology_knn_refits": len(split_evidence),
        "nested_esm_fits": len(split_evidence),
        "nested_feature_rows": nested_row_count,
        "feature_encoding": _NESTED_FEATURE_ENCODING,
        "all_nested_feature_rows_sha256": _nested_digest(all_nested_records),
        "accepted_ordinary_oof_used_for_stack_training": False,
        "splits": split_evidence,
    }
    return predictions, fold_document, nested_document


def _metric(rows: Sequence[Prediction], bins: int) -> dict[str, int | float | None]:
    _require(bool(rows), "metric subgroup is empty")
    return binary_metrics(
        [item.label for item in rows], [item.probability for item in rows], calibration_bins=bins
    )


def _component_equal_log_loss(rows: Sequence[Prediction]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        p = float(np.clip(item.probability, 1e-15, 1.0 - 1e-15))
        grouped[item.union_component_id].append(
            -(item.label * math.log(p) + (1 - item.label) * math.log1p(-p))
        )
    _require(bool(grouped), "component-equal metric has no components")
    return float(np.mean([np.mean(values) for _, values in sorted(grouped.items())]))


def _summarize_metrics(
    predictions: Sequence[Prediction], *, config: EsmUnionConfig
) -> dict[str, object]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for item in predictions:
        grouped[item.model].append(item)
    _require(tuple(grouped) == METHODS, "metric method order changed")
    example_ids = [item.example_id for item in grouped[METHODS[0]]]
    _require(
        all([item.example_id for item in grouped[method]] == example_ids for method in METHODS),
        "metric supports are not aligned",
    )
    component_rows: dict[str, list[str]] = defaultdict(list)
    reference_by_id = {item.example_id: item for item in grouped[PROMOTION_REFERENCE]}
    for example_id in example_ids:
        component_rows[reference_by_id[example_id].union_component_id].append(example_id)
    components = tuple(sorted(component_rows))
    sizes = sorted((len(values) for values in component_rows.values()), reverse=True)
    _require(
        len(components) == config.expected_union_components, "metric union-component census changed"
    )
    _require(
        sizes[0] == config.expected_largest_union_component_contexts,
        "largest union component changed",
    )
    _require(
        sum(size > 100 for size in sizes) == config.expected_union_components_over_100_contexts,
        "large union-component census changed",
    )
    largest = min(
        (component for component, values in component_rows.items() if len(values) == sizes[0])
    )
    methods: dict[str, object] = {}
    for method in METHODS:
        rows = grouped[method]
        by_identity: dict[str, object] = {}
        edges = config.similarity_bin_edges
        for left, right in pairwise(edges):
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
    aligned = {method: {item.example_id: item for item in grouped[method]} for method in METHODS}
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss")
    bootstrap = {method: {name: [] for name in metric_names} for method in METHODS}
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
        for method in METHODS:
            values[method] = _metric(
                [aligned[method][example_id] for example_id in sampled_ids], config.calibration_bins
            )
            for name in metric_names:
                value = values[method][name]
                _require(
                    value is not None and math.isfinite(float(value)),
                    f"bootstrap replicate {replicate + 1} has undefined {name}",
                )
                bootstrap[method][name].append(float(value))
        for name in deltas:
            candidate = values[PROMOTION_CANDIDATE][name]
            reference = values[PROMOTION_REFERENCE][name]
            assert candidate is not None and reference is not None
            deltas[name].append(float(candidate) - float(reference))
    for method in METHODS:
        cast(dict[str, object], methods[method])["union_component_bootstrap_95ci"] = {
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
                "successful_replicates": len(values),
            }
            for name, values in bootstrap[method].items()
        }
    candidate_overall = cast(
        dict[str, object], cast(dict[str, object], methods[PROMOTION_CANDIDATE])["overall"]
    )
    reference_overall = cast(
        dict[str, object], cast(dict[str, object], methods[PROMOTION_REFERENCE])["overall"]
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
            for method in METHODS
        ],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(matrix)
    _require(np.all(np.isfinite(correlation)), "method probability correlations are non-finite")
    return {
        "schema_version": 1,
        "artifact": "esm_union_nested_ensemble_metrics_v1",
        "methods": methods,
        "primary_comparison": {
            "candidate": PROMOTION_CANDIDATE,
            "reference": PROMOTION_REFERENCE,
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
                "candidate": PROMOTION_CANDIDATE,
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
            method: {other: float(correlation[i, j]) for j, other in enumerate(METHODS)}
            for i, method in enumerate(METHODS)
        },
        "statistical_gate_passed": all(checks.values()),
        "production_eligible": False,
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_predictions(path: Path, rows: Sequence[Prediction]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PREDICTION_SCHEMA), lineterminator="\n")
        writer.writeheader()
        for item in rows:
            row = asdict(item)
            row["max_train_identity"] = format(item.max_train_identity, ".17g")
            row["probability"] = format(item.probability, ".17g")
            writer.writerow(row)


def _write_sha_manifest(path: Path, entries: Mapping[str, str]) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for name in sorted(entries):
            handle.write(f"{entries[name]}  {name}\n")


def _flat_directory_identity(root: Path) -> DirectoryIdentity:
    """Fingerprint every file in an exact flat publication tree."""

    root_stat = root.stat(follow_symlinks=False)
    _require(
        stat.S_ISDIR(root_stat.st_mode) and not root.is_symlink(),
        "publication staging path is not a directory",
    )
    files: list[tuple[str, tuple[int, int, int, int], str]] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        item = _snapshot(child, label="publication staging file")
        files.append((child.name, item.fingerprint, item.sha256))
    return tuple(files)


def _prepare_publication_staging(staging: Path) -> tuple[DirectoryIdentity, tuple[int, int]]:
    """Validate, sync, and freeze the complete semantic tree before commit."""

    names = {child.name for child in staging.iterdir()}
    _require(names == _OUTPUT_FILES, "publication staging inventory changed")
    marker = _snapshot(staging / "SHA256SUMS", label="publication commit marker")
    entries = _parse_sha256_manifest(marker.payload, name="publication commit marker")
    _require(
        set(entries) == _OUTPUT_FILES - {"SHA256SUMS"},
        "publication commit marker coverage changed",
    )
    for name, digest in entries.items():
        item = _snapshot(staging / name, label=f"publication payload {name}")
        _require(item.sha256 == digest, f"publication payload checksum changed for {name}")
    for child in staging.iterdir():
        with child.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(child, 0o444)
    identity = _flat_directory_identity(staging)
    root_stat = staging.stat(follow_symlinks=False)
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    os.chmod(staging, 0o555)
    return identity, root_identity


def _renameat2_directory_noreplace(staging: Path, output: Path) -> int:
    """Return zero on success or the ``renameat2(RENAME_NOREPLACE)`` errno."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    return 0 if result == 0 else (ctypes.get_errno() or errno.EIO)


def _link_commit_directory_noreplace(
    staging: Path,
    output: Path,
    *,
    expected_identity: DirectoryIdentity,
) -> None:
    """Atomically claim the name and hard-link a checksum marker last.

    Filesystems such as the cluster Lustre volume reject
    ``renameat2(RENAME_NOREPLACE)``.  An exclusive ``mkdir`` still provides a
    real no-clobber name claim.  All immutable payloads are then linked into
    that claimed directory and the already validated ``SHA256SUMS`` is linked
    last in mode 000. Its final atomic transition to mode 0444 is the commit
    point. A crash before that transition leaves an invalid, never-reused
    quarantine; every consumer requires the readable frozen marker, exact
    inventory, hashes, and read-only modes.
    """

    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        os.chmod(staging, 0o755)
        raise FileExistsError(f"refusing to replace ESM union output: {output}") from error
    except OSError:
        os.chmod(staging, 0o755)
        raise
    try:
        claim_stat = output.stat(follow_symlinks=False)
    except OSError:
        os.chmod(staging, 0o755)
        raise
    claim_identity = (claim_stat.st_dev, claim_stat.st_ino)
    _require(
        stat.S_ISDIR(claim_stat.st_mode) and not output.is_symlink(),
        "publication name claim is not a directory",
    )
    staging_stat = staging.stat(follow_symlinks=False)
    try:
        _require(
            staging_stat.st_dev == claim_stat.st_dev,
            "publication staging and destination are on different filesystems",
        )
        identity_by_name = {
            name: (fingerprint, digest) for name, fingerprint, digest in expected_identity
        }
        for name in sorted(_OUTPUT_FILES - {"SHA256SUMS"}):
            source = staging / name
            destination = output / name
            os.link(source, destination, follow_symlinks=False)
            source_stat = source.stat(follow_symlinks=False)
            destination_snapshot = _snapshot(destination, label=f"claimed publication {name}")
            fingerprint, digest = identity_by_name[name]
            _require(
                (source_stat.st_dev, source_stat.st_ino)
                == (destination_snapshot.fingerprint[0], destination_snapshot.fingerprint[1])
                and destination_snapshot.fingerprint == fingerprint
                and destination_snapshot.sha256 == digest,
                f"claimed publication link changed for {name}",
            )
        current_claim = output.stat(follow_symlinks=False)
        _require(
            (current_claim.st_dev, current_claim.st_ino) == claim_identity
            and not output.is_symlink(),
            "publication name claim changed before commit",
        )
        directory_descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        marker_source = staging / "SHA256SUMS"
        marker_destination = output / "SHA256SUMS"
        marker_snapshot = _snapshot(marker_source, label="publication marker before commit")
        marker_fingerprint, marker_digest = identity_by_name["SHA256SUMS"]
        _require(
            marker_snapshot.fingerprint == marker_fingerprint
            and marker_snapshot.sha256 == marker_digest,
            "publication commit marker changed before linking",
        )
        # Mode 000 makes the present name an unreadable pre-commit marker. The
        # final atomic chmod to 0444 below is the only commit transition.
        os.chmod(marker_source, 0o000)
        os.link(marker_source, marker_destination, follow_symlinks=False)
        source_stat = marker_source.stat(follow_symlinks=False)
        destination_stat = marker_destination.stat(follow_symlinks=False)
        _require(
            (source_stat.st_dev, source_stat.st_ino)
            == (destination_stat.st_dev, destination_stat.st_ino),
            "publication commit marker link changed",
        )
        children = {child.name: child for child in output.iterdir()}
        _require(set(children) == _OUTPUT_FILES, "claimed publication inventory changed")
        for name in sorted(_OUTPUT_FILES - {"SHA256SUMS"}):
            snapshot = _snapshot(children[name], label=f"claimed publication final {name}")
            fingerprint, digest = identity_by_name[name]
            _require(
                snapshot.fingerprint == fingerprint and snapshot.sha256 == digest,
                f"claimed publication final identity changed for {name}",
            )
        _require(
            stat.S_IMODE(destination_stat.st_mode) == 0,
            "pre-commit marker unexpectedly became readable",
        )
        directory_descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(output, 0o555)
        final_claim = output.stat(follow_symlinks=False)
        _require(
            (final_claim.st_dev, final_claim.st_ino) == claim_identity
            and stat.S_IMODE(final_claim.st_mode) == 0o555
            and not output.is_symlink(),
            "publication name claim changed while being frozen",
        )
        parent_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        # Atomic commit: no reported or semantic operation is permitted after
        # this chmod makes the canonical marker readable.
        os.chmod(marker_destination, 0o444)
    except BaseException:
        try:
            current = output.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == claim_identity and not output.is_symlink():
                os.chmod(output, 0o555)
        finally:
            os.chmod(staging, 0o755)
        raise

    # Publication is committed. Hidden staging cleanup is best-effort and can
    # never turn the accepted write-once tree into a reported failure.
    try:
        os.chmod(staging, 0o755)
        shutil.rmtree(staging)
    except OSError:
        pass


def _publish_noreplace(staging: Path, output: Path) -> None:
    expected_identity, expected_root = _prepare_publication_staging(staging)
    error_number = _renameat2_directory_noreplace(staging, output)
    if error_number == 0:
        output_stat = output.stat(follow_symlinks=False)
        _require(
            not os.path.lexists(staging)
            and os.path.lexists(output)
            and (output_stat.st_dev, output_stat.st_ino) == expected_root
            and not output.is_symlink()
            and _flat_directory_identity(output) == expected_identity,
            "atomic no-replace directory publication postcondition failed",
        )
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        os.chmod(staging, 0o755)
        raise FileExistsError(f"refusing to replace ESM union output: {output}")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        _link_commit_directory_noreplace(
            staging,
            output,
            expected_identity=expected_identity,
        )
        return
    os.chmod(staging, 0o755)
    raise OSError(error_number, os.strerror(error_number), str(output))


def _scan_publication(staging: Path) -> None:
    for path in staging.iterdir():
        payload = path.read_bytes()
        _require(
            _ABSOLUTE_BYTES_RE.search(payload) is None,
            f"public artifact {path.name} exposes an absolute path",
        )


def _protected(output: Path, roots: Sequence[Path]) -> None:
    for root in roots:
        _require(
            output != root and root not in output.parents and output not in root.parents,
            "output overlaps a protected input root",
        )


def run_esm_union_ensemble_oof(
    *,
    base_twin_root: str | Path,
    base_independent_receipt: str | Path,
    embedding_twin_root: str | Path,
    embedding_independent_receipt: str | Path,
    base_config_path: str | Path,
    embedding_config_path: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    frozen_input_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> EsmUnionExecution:
    """Run the fixed nested benchmark and publish its seven-file evidence tree."""

    _require(_GIT_RE.fullmatch(git_commit) is not None, "git_commit must be a full lowercase SHA-1")
    config_snapshot = _snapshot(config_path, label="ESM union ensemble config")
    config = _config_from_snapshot(config_snapshot)
    base_config_snapshot = _snapshot(base_config_path, label="accepted Gate-1 config")
    embedding_config_snapshot = _snapshot(embedding_config_path, label="accepted embedding config")
    _expect_sha(base_config_snapshot, config.base_config_sha256, label="accepted Gate-1 config")
    _expect_sha(
        embedding_config_snapshot, config.embedding_config_sha256, label="accepted embedding config"
    )
    base_config = _base_config_from_payload(base_config_snapshot.path, base_config_snapshot.payload)
    base = _read_twins(
        base_twin_root,
        expected_job_id=config.base_producer_job_id,
        expected_files=_BASE_TREE_FILES,
        expected_top_sha256=config.base_publication_top_sha256,
        label="accepted Gate-1",
    )
    embedding = _read_twins(
        embedding_twin_root,
        expected_job_id=config.embedding_producer_job_id,
        expected_files=_EMBEDDING_TREE_FILES,
        expected_top_sha256=config.embedding_publication_top_sha256,
        label="accepted embeddings",
    )
    base_receipt = _snapshot(base_independent_receipt, label="Gate-1 independent receipt")
    embedding_receipt = _snapshot(
        embedding_independent_receipt, label="embedding independent receipt"
    )
    _require(
        base_receipt.path.stat().st_mode & 0o222 == 0, "Gate-1 independent receipt remains writable"
    )
    _require(
        embedding_receipt.path.stat().st_mode & 0o222 == 0,
        "embedding independent receipt remains writable",
    )
    _expect_sha(
        base_receipt, config.base_independent_receipt_sha256, label="Gate-1 independent receipt"
    )
    _expect_sha(
        embedding_receipt,
        config.embedding_independent_receipt_sha256,
        label="embedding independent receipt",
    )
    code_manifest = _snapshot(code_manifest_path, label="code manifest")
    frozen_manifest = _snapshot(frozen_input_manifest_path, label="frozen input manifest")
    code_attestation, repository_snapshots = _verify_code_manifest(
        code_manifest,
        config=config_snapshot,
        base_config=base_config_snapshot,
        embedding_config=embedding_config_snapshot,
    )
    frozen_snapshots = {
        **{
            f"base/0/{name}": _snapshot(base.root / "0" / name, label=f"frozen base task 0 {name}")
            for name in sorted(_BASE_TREE_FILES)
        },
        **{
            f"base/1/{name}": _snapshot(base.root / "1" / name, label=f"frozen base task 1 {name}")
            for name in sorted(_BASE_TREE_FILES)
        },
        **{
            f"base/node-receipts/{name}": _snapshot(
                base.root / "node-receipts" / name, label=f"frozen base handshake {name}"
            )
            for name in sorted(_HANDSHAKE_FILES)
        },
        "base-independent-receipt.json": base_receipt,
        **{
            f"embedding/0/{name}": _snapshot(
                embedding.root / "0" / name, label=f"frozen embedding task 0 {name}"
            )
            for name in sorted(_EMBEDDING_TREE_FILES)
        },
        **{
            f"embedding/1/{name}": _snapshot(
                embedding.root / "1" / name, label=f"frozen embedding task 1 {name}"
            )
            for name in sorted(_EMBEDDING_TREE_FILES)
        },
        **{
            f"embedding/node-receipts/{name}": _snapshot(
                embedding.root / "node-receipts" / name, label=f"frozen embedding handshake {name}"
            )
            for name in sorted(_HANDSHAKE_FILES)
        },
        "embedding-independent-receipt.json": embedding_receipt,
    }
    _verify_frozen_inputs(frozen_manifest, frozen_snapshots)
    for logical, expected in (
        ("gate1/SHA256SUMS", config.base_semantic_top_sha256),
        ("gate1/examples.jsonl", config.base_examples_sha256),
        ("gate1/folds.json", config.base_folds_sha256),
        ("gate1/oof_predictions.csv", config.base_oof_sha256),
        ("gate1/manifest.json", config.base_manifest_sha256),
        ("gate1/split_receipt.json", config.base_split_receipt_sha256),
    ):
        _expect_sha(base.snapshots[logical], expected, label=logical)
    examples = _read_examples(
        base.snapshots["gate1/examples.jsonl"],
        base.snapshots["gate1/oof_predictions.csv"],
        config=config,
    )
    folds = _verify_folds(base.snapshots["gate1/folds.json"], examples=examples, config=config)
    base_manifest, split_receipt = _verify_base_documents(
        base.snapshots, base_receipt, examples=examples, folds_document=folds, config=config
    )
    base_receipt_document = _pretty_json(base_receipt, label="Gate-1 independent receipt")
    _require(
        base_receipt_document.get("production_handshake") == base.handshake
        and base_receipt_document.get("code_manifest_sha256")
        == base.snapshots["CODE_SHA256SUMS"].sha256
        and base_receipt_document.get("frozen_input_manifest_sha256")
        == base.snapshots["FROZEN_INPUT_SHA256SUMS"].sha256,
        "Gate-1 independent receipt does not bind the accepted twins",
    )
    embedding_manifest, embedding_receipt_document = _verify_embedding_documents(
        embedding, embedding_receipt, config=config
    )
    accepted = _read_accepted_probabilities(base.snapshots["gate1/oof_predictions.csv"], examples)
    table, coverage = _read_embedding_table(
        embedding.snapshots["embeddings/embedding_index.csv"],
        embedding.snapshots["embeddings/embeddings.npy"],
        examples,
        config=config,
    )

    requested_output = Path(output_dir).absolute()
    _reject_symlink_chain(requested_output.parent, label="output parent")
    parent = requested_output.parent.resolve(strict=True)
    _require(
        parent.is_dir() and parent.stat().st_mode & 0o200 != 0, "output parent is not writable"
    )
    output = parent / requested_output.name
    _require(not os.path.lexists(output), f"refusing to reuse ESM union output: {output}")
    _protected(output, (base.root, embedding.root, base_receipt.path, embedding_receipt.path))

    predictions, fold_models, nested_fits = _fit_all_predictions(
        examples, accepted, table, base_config=base_config, config=config
    )
    metrics = _summarize_metrics(predictions, config=config)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(parent)))
    published = False
    try:
        prediction_path = staging / "esm_union_oof_predictions.csv"
        fold_models_path = staging / "fold_models.json"
        nested_fits_path = staging / "nested_fits.json"
        coverage_path = staging / "embedding_coverage.json"
        metrics_path = staging / "metrics.json"
        manifest_path = staging / "manifest.json"
        top_path = staging / "SHA256SUMS"
        _write_predictions(prediction_path, predictions)
        _write_json(fold_models_path, fold_models)
        _write_json(nested_fits_path, nested_fits)
        _write_json(coverage_path, coverage)
        _write_json(metrics_path, metrics)
        artifact_roles = {
            "embedding_coverage.json": "exact base-to-embedding sequence join evidence",
            "esm_union_oof_predictions.csv": "seven-method context-level outer OOF predictions",
            "fold_models.json": "five outer ESM heads and nested stack fits",
            "metrics.json": "metrics and shared union-component bootstrap evidence",
            "nested_fits.json": "twenty ordered outer-inner base and ESM refits",
        }
        artifacts = {
            name: {"filename": name, "role": role, "sha256": _file_sha256(staging / name)}
            for name, role in sorted(artifact_roles.items())
        }
        statistical_passed = cast(bool, metrics["statistical_gate_passed"])
        manifest = {
            "schema_version": 1,
            "artifact": ARTIFACT,
            "status": OUTPUT_STATUS,
            "git_commit": git_commit,
            "config_sha256": config_snapshot.sha256,
            "methods": list(METHODS),
            "promotion_reference": PROMOTION_REFERENCE,
            "promotion_candidate": PROMOTION_CANDIDATE,
            "statistical_gate_passed": statistical_passed,
            "statistical_status": "eligible_for_untouched_external_validation_only"
            if statistical_passed
            else "did_not_pass_frozen_development_gate",
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
                "contexts": len(examples),
                "source_observations": sum(item.source_observations for item in examples),
                "sequences": len({item.sequence_id for item in examples}),
                "positives": sum(item.label for item in examples),
                "negatives": sum(1 - item.label for item in examples),
                "homology_components": len({item.homology_component_id for item in examples}),
                "union_components": len({item.union_component_id for item in examples}),
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
                "npy_sha256": table.matrix_snapshot.sha256,
                "tensor_data_sha256": table.tensor_data_sha256,
                "candidate_status": embedding_manifest.get("status"),
            },
            "input_sha256": {
                "code_manifest": code_manifest.sha256,
                "frozen_input_manifest": frozen_manifest.sha256,
                "config": config_snapshot.sha256,
                "base_config": base_config_snapshot.sha256,
                "embedding_config": embedding_config_snapshot.sha256,
                "base_publication_top": base.snapshots["SHA256SUMS"].sha256,
                "base_independent_receipt": base_receipt.sha256,
                "embedding_publication_top": embedding.snapshots["SHA256SUMS"].sha256,
                "embedding_independent_receipt": embedding_receipt.sha256,
            },
            "code_attestation": code_attestation,
            "artifacts": artifacts,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        }
        _write_json(manifest_path, manifest)
        top_entries = {
            name: _file_sha256(staging / name) for name in sorted(_OUTPUT_FILES - {"SHA256SUMS"})
        }
        _write_sha_manifest(top_path, top_entries)
        _require(
            {item.name for item in staging.iterdir()} == _OUTPUT_FILES, "staging inventory changed"
        )
        _scan_publication(staging)
        for index, item in enumerate(
            (
                *base.all_snapshots,
                *embedding.all_snapshots,
                base_receipt,
                embedding_receipt,
                config_snapshot,
                base_config_snapshot,
                embedding_config_snapshot,
                code_manifest,
                frozen_manifest,
                *repository_snapshots,
            )
        ):
            _assert_unchanged(item, label=f"input snapshot {index}")
        _assert_twin_tree_unchanged(base, expected_files=_BASE_TREE_FILES, label="accepted Gate-1")
        _assert_twin_tree_unchanged(
            embedding,
            expected_files=_EMBEDDING_TREE_FILES,
            label="accepted embeddings",
        )
        _require(base_receipt.path.stat().st_mode & 0o222 == 0, "Gate-1 receipt became writable")
        _require(
            embedding_receipt.path.stat().st_mode & 0o222 == 0, "embedding receipt became writable"
        )
        _publish_noreplace(staging, output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return EsmUnionExecution(
        output_dir=output,
        predictions_path=output / "esm_union_oof_predictions.csv",
        fold_models_path=output / "fold_models.json",
        nested_fits_path=output / "nested_fits.json",
        embedding_coverage_path=output / "embedding_coverage.json",
        metrics_path=output / "metrics.json",
        manifest_path=output / "manifest.json",
        top_manifest_path=output / "SHA256SUMS",
        examples=len(examples),
        prediction_rows=len(predictions),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-twin-root", required=True, type=Path)
    parser.add_argument("--base-independent-receipt", required=True, type=Path)
    parser.add_argument("--embedding-twin-root", required=True, type=Path)
    parser.add_argument("--embedding-independent-receipt", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    parser.add_argument("--frozen-input-manifest", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_esm_union_ensemble_oof(
        base_twin_root=args.base_twin_root,
        base_independent_receipt=args.base_independent_receipt,
        embedding_twin_root=args.embedding_twin_root,
        embedding_independent_receipt=args.embedding_independent_receipt,
        base_config_path=args.base_config,
        embedding_config_path=args.embedding_config,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

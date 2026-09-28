"""Authenticate the accepted candidate scorer and publish a mean-only ledger.

This adapter is intentionally narrower than an ensemble.  It copies the seven
accepted descriptor probabilities and their three arithmetic means into a
selector-facing development ledger.  The five-refit spread is isolated in a
separate diagnostics table and is never exposed as uncertainty or as a
selectable field.
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
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

SCHEMA_VERSION = 1
ARTIFACT = "candidate_activity_mean_ledger_v1"
OUTPUT_STATUS = "development_mean_only_control"
SOURCE_ARTIFACT = "candidate_activity_scoring_development_v1"
SOURCE_STATUS = "development_candidate_scoring_only"
INDEPENDENT_ARTIFACT = "candidate_activity_scoring_development_v1_independent_verification"
OPERATIONAL_ARTIFACT = "candidate_activity_scoring_v1_operational_audit"
ALLOWED_CONSUMER = "mean_only_selection_control_only"
ELIGIBILITY_SCOPE = "accepted_candidate_pool_library_eligible_flag_v1"
UNCERTAINTY_STATUS = "unavailable"
DIAGNOSTIC_SCOPE = (
    "population_sd_across_five_outer_fold_complement_refits_"
    "diagnostic_not_uncertainty_not_selectable"
)
PREDICTION_SCOPE = "declared_seven_target_activity_panel"
TRAINING_SCOPE = "all_2492_accepted_gate1_contexts_after_recipe_freeze"
CALIBRATION_SCOPE = "none_raw_logistic_probability"
TARGET_AGGREGATION = "equal_target_arithmetic_mean_within_declared_scope_v1"
FIT_SENSITIVITY_SEMANTICS = (
    "population_sd_across_five_outer_fold_complement_refits_diagnostic_not_posterior_or_"
    "aleatoric_uncertainty"
)

OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
TARGET_PANEL = (
    ("acinetobacter_baumannii", "negative"),
    ("enterococcus_faecalis", "positive"),
    ("enterococcus_faecium", "positive"),
    ("escherichia_coli", "negative"),
    ("klebsiella_pneumoniae", "negative"),
    ("pseudomonas_aeruginosa", "negative"),
    ("staphylococcus_aureus", "positive"),
)
SOURCE_FILES = (
    "SHA256SUMS",
    "candidate_activity_scores.csv",
    "fold_model_target_probabilities.npy",
    "manifest.json",
    "model_states.json",
    "oof_reproduction.json",
)
AUDIT_FILES = (
    "operational-receipt.json",
    "twin-0-independent-verification.json",
    "twin-1-independent-verification.json",
)
OUTPUT_FILES = (
    "candidate_ledger.csv",
    "model_fit_sensitivity_diagnostics.csv",
    "manifest.json",
    "SHA256SUMS",
)
SOURCE_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    "sequence",
    "length",
    "library_eligible",
    "generator_families",
    "generator_variants",
    *(f"probability_{name}" for name, _ in TARGET_PANEL),
    *(f"mean_{name}" for name in OBJECTIVES),
    *(f"model_fit_sensitivity_{name}" for name in OBJECTIVES),
    "prediction_scope",
    "training_scope",
    "calibration_scope",
    "model_release_id",
)
LEDGER_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    "sequence",
    "length",
    "eligible",
    "eligibility_scope",
    "generator_families",
    "generator_variants",
    *(f"probability_{name}" for name, _ in TARGET_PANEL),
    *(f"mean_{name}" for name in OBJECTIVES),
    "uncertainty_status",
    "prediction_scope",
    "training_scope",
    "calibration_scope",
    "model_release_id",
)
DIAGNOSTIC_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    *(f"model_fit_sensitivity_{name}" for name in OBJECTIVES),
    "diagnostic_scope",
)
FORBIDDEN_CLAIMS = (
    "aleatoric_uncertainty",
    "calibrated_uncertainty",
    "epistemic_uncertainty",
    "final_ranking",
    "hemolysis_risk",
    "mdr_eskape_activity",
    "out_of_distribution",
    "posterior_draws",
    "production_ensemble",
    "quality_probability",
    "selectivity",
)
EXCLUDED_WEIGHTS = (
    "endpoint_weight",
    "model_fit_sensitivity_weight",
    "uncertainty_weight",
)
EXCLUDED_ENDPOINTS = (
    "hemolysis_risk",
    "mdr_eskape_activity",
    "out_of_distribution",
    "quality_probability",
    "selectivity",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_LINEAGE_TOKEN_RE = re.compile(r"[^|,\r\n\x00-\x20\x7f]{1,256}")


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    """Bytes and filesystem identity captured from one authenticated file."""

    path: Path
    payload: bytes
    sha256: str
    mode: int
    fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DirectorySnapshot:
    """Identity, mode, and exact flat inventory of an evidence directory."""

    path: Path
    entries: tuple[str, ...]
    fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AcceptedScoringContract:
    producer_job_id: int
    audit_job_id: int
    git_commit: str
    config_sha256: str
    publication_top_sha256: str
    publication_tree_sha256: str
    manifest_sha256: str
    candidate_scores_sha256: str
    fold_tensor_sha256: str
    fold_tensor_raw_data_sha256: str
    model_states_sha256: str
    model_release_id: str
    oof_reproduction_sha256: str
    independent_verification_sha256: str
    operational_audit_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateActivityMeanLedgerConfig:
    path: Path
    snapshot: InputSnapshot
    expected_candidates: int
    expected_examples: int
    folds: int
    targets: int
    accepted_scoring: AcceptedScoringContract


@dataclass(frozen=True, slots=True)
class SourceRow:
    source_ordinal: str
    sequence_id: str
    sequence: str
    length: str
    library_eligible: str
    generator_families: str
    generator_variants: str
    target_probabilities: tuple[str, ...]
    means: tuple[str, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuthenticatedScoring:
    rows: tuple[SourceRow, ...]
    manifest: Mapping[str, object]
    snapshots: Mapping[str, InputSnapshot]
    directory_snapshots: Mapping[str, DirectorySnapshot]


@dataclass(frozen=True, slots=True)
class CandidateActivityMeanLedgerExecution:
    output_dir: Path
    candidate_count: int
    candidate_ledger_sha256: str
    diagnostics_sha256: str
    manifest_sha256: str
    publication_top_sha256: str


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            continue
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} cannot contain a symbolic link: {current}")


def _snapshot_file(
    path: str | Path,
    *,
    label: str,
    required_mode: int | None = None,
) -> InputSnapshot:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(absolute, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {absolute}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        mode = stat.S_IMODE(before.st_mode)
        if required_mode is not None and mode != required_mode:
            raise ValueError(f"{label} must have mode {required_mode:o}, observed {mode:o}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after):
        raise ValueError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return InputSnapshot(
        path=absolute,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        mode=mode,
        fingerprint=_fingerprint(before),
    )


def _assert_snapshot_unchanged(snapshot: InputSnapshot, *, label: str) -> None:
    current = _snapshot_file(
        snapshot.path,
        label=label,
        required_mode=snapshot.mode,
    )
    if current.fingerprint != snapshot.fingerprint or current.payload != snapshot.payload:
        raise ValueError(f"{label} changed after authentication")


def _capture_directory_snapshot(path: str | Path, *, label: str) -> DirectorySnapshot:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(absolute, label=label)
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {absolute}") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o555:
        raise ValueError(f"{label} must be a non-symbolic mode-555 directory")
    try:
        entries = tuple(sorted(os.listdir(absolute)))
    except OSError as error:
        raise ValueError(f"cannot enumerate {label}: {absolute}") from error
    try:
        after = os.lstat(absolute)
    except OSError as error:
        raise ValueError(f"cannot re-inspect {label}: {absolute}") from error
    if _fingerprint(metadata) != _fingerprint(after):
        raise ValueError(f"{label} changed while its inventory was captured")
    return DirectorySnapshot(
        path=absolute,
        entries=entries,
        fingerprint=_fingerprint(metadata),
    )


def _assert_directory_snapshot_unchanged(snapshot: DirectorySnapshot, *, label: str) -> None:
    current = _capture_directory_snapshot(snapshot.path, label=label)
    if current.fingerprint != snapshot.fingerprint or current.entries != snapshot.entries:
        raise ValueError(f"{label} changed after authentication")


def _snapshot_flat_bundle(
    root: str | Path,
    *,
    names: Sequence[str],
    label: str,
) -> tuple[DirectorySnapshot, dict[str, InputSnapshot]]:
    directory = _capture_directory_snapshot(root, label=label)
    expected_entries = tuple(sorted(names))
    if directory.entries != expected_entries:
        raise ValueError(f"{label} inventory is not exact: {list(directory.entries)}")
    snapshots = {
        name: _snapshot_file(
            directory.path / name,
            label=f"{label} {name}",
            required_mode=0o444,
        )
        for name in names
    }
    after = _capture_directory_snapshot(directory.path, label=label)
    if after.fingerprint != directory.fingerprint or after.entries != expected_entries:
        raise ValueError(f"{label} changed while its files were captured")
    return directory, snapshots


def _require_exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} keys are not exact: {observed}")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_canonical_json(snapshot: InputSnapshot, *, label: str) -> Mapping[str, Any]:
    if b"\r" in snapshot.payload or not snapshot.payload.endswith(b"\n"):
        raise ValueError(f"{label} must be canonical UTF-8 with LF termination")
    try:
        document = json.loads(
            snapshot.payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid canonical JSON") from error
    if not isinstance(document, dict) or _canonical_json_bytes(document) != snapshot.payload:
        raise ValueError(f"{label} JSON bytes are not canonical")
    return document


def _toml_table(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    return value


def load_candidate_activity_mean_ledger_config(
    path: str | Path,
) -> CandidateActivityMeanLedgerConfig:
    """Load the exact mean-only adapter contract and snapshot its bytes."""

    snapshot = _snapshot_file(path, label="mean-ledger config")
    if b"\r" in snapshot.payload or not snapshot.payload.endswith(b"\n"):
        raise ValueError("mean-ledger config must use UTF-8/LF and end with LF")
    try:
        document = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("mean-ledger config is not valid UTF-8 TOML") from error
    root_keys = {
        "schema_version",
        "artifact",
        "status",
        "automatic_production_eligible",
        "expected_candidates",
        "expected_examples",
        "folds",
        "targets",
        "objectives",
        "allowed_consumer",
        "eligibility_scope",
        "uncertainty_status",
        "diagnostic_scope",
        "prediction_scope",
        "training_scope",
        "calibration_scope",
        "accepted_scoring",
    }
    root = _require_exact_keys(document, root_keys, label="mean-ledger config")
    expected_literals = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": "predeclared_development_mean_only_handoff",
        "automatic_production_eligible": False,
        "objectives": list(OBJECTIVES),
        "allowed_consumer": ALLOWED_CONSUMER,
        "eligibility_scope": ELIGIBILITY_SCOPE,
        "uncertainty_status": UNCERTAINTY_STATUS,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "prediction_scope": PREDICTION_SCOPE,
        "training_scope": TRAINING_SCOPE,
        "calibration_scope": CALIBRATION_SCOPE,
    }
    for key, expected in expected_literals.items():
        if root[key] != expected or type(root[key]) is not type(expected):
            raise ValueError(f"mean-ledger config {key} differs from the frozen contract")
    expected_candidates = _require_int(
        root["expected_candidates"], label="expected_candidates", minimum=1
    )
    expected_examples = _require_int(
        root["expected_examples"], label="expected_examples", minimum=1
    )
    folds = _require_int(root["folds"], label="folds", minimum=2)
    targets = _require_int(root["targets"], label="targets", minimum=1)
    if folds != 5 or targets != len(TARGET_PANEL):
        raise ValueError("mean-ledger config requires five folds and the seven-target panel")

    scoring_keys = {
        "producer_job_id",
        "audit_job_id",
        "git_commit",
        "config_sha256",
        "publication_top_sha256",
        "publication_tree_sha256",
        "manifest_sha256",
        "candidate_scores_sha256",
        "fold_tensor_sha256",
        "fold_tensor_raw_data_sha256",
        "model_states_sha256",
        "model_release_id",
        "oof_reproduction_sha256",
        "independent_verification_sha256",
        "operational_audit_sha256",
    }
    source = _require_exact_keys(
        _toml_table(root["accepted_scoring"], label="accepted_scoring"),
        scoring_keys,
        label="accepted_scoring",
    )
    contract = AcceptedScoringContract(
        producer_job_id=_require_int(source["producer_job_id"], label="producer_job_id", minimum=1),
        audit_job_id=_require_int(source["audit_job_id"], label="audit_job_id", minimum=1),
        git_commit=_require_git_sha(source["git_commit"], label="source git_commit"),
        config_sha256=_require_sha(source["config_sha256"], label="source config SHA-256"),
        publication_top_sha256=_require_sha(
            source["publication_top_sha256"], label="publication top SHA-256"
        ),
        publication_tree_sha256=_require_sha(
            source["publication_tree_sha256"], label="publication tree SHA-256"
        ),
        manifest_sha256=_require_sha(source["manifest_sha256"], label="manifest SHA-256"),
        candidate_scores_sha256=_require_sha(
            source["candidate_scores_sha256"], label="candidate scores SHA-256"
        ),
        fold_tensor_sha256=_require_sha(source["fold_tensor_sha256"], label="fold tensor SHA-256"),
        fold_tensor_raw_data_sha256=_require_sha(
            source["fold_tensor_raw_data_sha256"], label="fold tensor raw SHA-256"
        ),
        model_states_sha256=_require_sha(
            source["model_states_sha256"], label="model states SHA-256"
        ),
        model_release_id=_require_sha(source["model_release_id"], label="model release ID"),
        oof_reproduction_sha256=_require_sha(
            source["oof_reproduction_sha256"], label="OOF reproduction SHA-256"
        ),
        independent_verification_sha256=_require_sha(
            source["independent_verification_sha256"],
            label="independent verification SHA-256",
        ),
        operational_audit_sha256=_require_sha(
            source["operational_audit_sha256"], label="operational audit SHA-256"
        ),
    )
    if contract.model_states_sha256 != contract.model_release_id:
        raise ValueError("model release ID must equal model_states.json SHA-256")
    return CandidateActivityMeanLedgerConfig(
        path=snapshot.path,
        snapshot=snapshot,
        expected_candidates=expected_candidates,
        expected_examples=expected_examples,
        folds=folds,
        targets=targets,
        accepted_scoring=contract,
    )


def _bundle_tree_sha256(snapshots: Mapping[str, InputSnapshot]) -> str:
    transcript = b"".join(
        f"{snapshots[name].mode:o} {snapshots[name].sha256} {name}\n".encode()
        for name in sorted(snapshots)
    )
    return hashlib.sha256(transcript).hexdigest()


def _validate_sha256sums(
    snapshot: InputSnapshot,
    *,
    snapshots: Mapping[str, InputSnapshot],
    expected_top: str,
) -> None:
    if snapshot.sha256 != expected_top:
        raise ValueError("source SHA256SUMS hash differs from the accepted top hash")
    expected = b"".join(
        f"{snapshots[name].sha256}  {name}\n".encode("ascii")
        for name in sorted(set(SOURCE_FILES) - {"SHA256SUMS"})
    )
    if snapshot.payload != expected:
        raise ValueError("source SHA256SUMS bytes or inventory are not exact")


def _validate_artifact_record(
    value: object,
    *,
    snapshot: InputSnapshot,
    label: str,
    rows: int | None = None,
) -> None:
    expected_keys = {"sha256", "size_bytes"} | ({"rows"} if rows is not None else set())
    record = _require_exact_keys(value, expected_keys, label=label)
    if record["sha256"] != snapshot.sha256 or record["size_bytes"] != len(snapshot.payload):
        raise ValueError(f"{label} does not bind the authenticated bytes")
    if rows is not None and record["rows"] != rows:
        raise ValueError(f"{label} row count differs")


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    snapshots: Mapping[str, InputSnapshot],
    config: CandidateActivityMeanLedgerConfig,
) -> None:
    expected_root = {
        "schema_version",
        "artifact",
        "status",
        "automatic_production_eligible",
        "probability_calibration",
        "prediction_scope",
        "training_scope",
        "fit_sensitivity_semantics",
        "target_aggregation",
        "model_release_id",
        "git_commit",
        "inputs",
        "targets",
        "models",
        "aggregations",
        "fold_tensor",
        "candidate_csv",
        "claims",
        "artifacts",
    }
    root = _require_exact_keys(manifest, expected_root, label="source manifest")
    contract = config.accepted_scoring
    literals = {
        "schema_version": 1,
        "artifact": SOURCE_ARTIFACT,
        "status": SOURCE_STATUS,
        "automatic_production_eligible": False,
        "probability_calibration": CALIBRATION_SCOPE,
        "prediction_scope": PREDICTION_SCOPE,
        "training_scope": TRAINING_SCOPE,
        "fit_sensitivity_semantics": FIT_SENSITIVITY_SEMANTICS,
        "target_aggregation": TARGET_AGGREGATION,
        "model_release_id": contract.model_release_id,
        "git_commit": contract.git_commit,
    }
    for key, expected in literals.items():
        if root[key] != expected or type(root[key]) is not type(expected):
            raise ValueError(f"source manifest {key} differs from the accepted contract")

    candidate_csv = _require_exact_keys(
        root["candidate_csv"], {"filename", "columns", "rows"}, label="candidate_csv"
    )
    if candidate_csv != {
        "filename": "candidate_activity_scores.csv",
        "columns": list(SOURCE_COLUMNS),
        "rows": config.expected_candidates,
    }:
        raise ValueError("source manifest candidate CSV schema differs")

    targets = root["targets"]
    expected_targets = [
        {
            "index": index,
            "name": name,
            "gram": gram,
            "probability_column": f"probability_{name}",
        }
        for index, (name, gram) in enumerate(TARGET_PANEL)
    ]
    if targets != expected_targets:
        raise ValueError("source target panel differs from the accepted seven-target panel")

    tensor = _require_exact_keys(
        root["fold_tensor"],
        {
            "filename",
            "axes",
            "shape",
            "dtype",
            "byte_order",
            "memory_order",
            "candidate_axis_order",
            "fold_axis_member_order",
            "target_axis_order",
            "raw_data_sha256",
        },
        label="fold_tensor",
    )
    expected_tensor = {
        "filename": "fold_model_target_probabilities.npy",
        "axes": ["candidate", "outer_fold_complement_model", "target"],
        "shape": [config.expected_candidates, config.folds, config.targets],
        "dtype": "float64",
        "byte_order": "little_endian",
        "memory_order": "C_contiguous",
        "candidate_axis_order": "candidate_jsonl_source_ordinal_ascending",
        "fold_axis_member_order": [f"outer_fold_{index}" for index in range(config.folds)],
        "target_axis_order": [name for name, _ in TARGET_PANEL],
        "raw_data_sha256": contract.fold_tensor_raw_data_sha256,
    }
    if tensor != expected_tensor:
        raise ValueError("source fold tensor contract differs")

    aggregations = _require_exact_keys(
        root["aggregations"],
        {
            "objectives",
            "target_reduction",
            "deployment_probability_source",
            "model_fit_sensitivity",
        },
        label="aggregations",
    )
    if aggregations != {
        "objectives": list(OBJECTIVES),
        "target_reduction": TARGET_AGGREGATION,
        "deployment_probability_source": "all_data_deployment",
        "model_fit_sensitivity": FIT_SENSITIVITY_SEMANTICS,
    }:
        raise ValueError("source aggregation contract differs")

    models = _require_exact_keys(
        root["models"],
        {
            "family",
            "accepted_oof_model",
            "fold_members",
            "deployment_member",
            "weights",
            "serialized_states_file",
        },
        label="models",
    )
    if models != {
        "family": "descriptor_logistic",
        "accepted_oof_model": "descriptor_logistic",
        "fold_members": [f"outer_fold_{index}" for index in range(config.folds)],
        "deployment_member": "all_data_deployment",
        "weights": {
            "apex": 0.0,
            "descriptor_logistic": 1.0,
            "esm_supervised": 0.0,
            "geometry": 0.0,
            "homology_knn": 0.0,
        },
        "serialized_states_file": "model_states.json",
    }:
        raise ValueError("source model declaration differs")

    claims = _require_exact_keys(
        root["claims"],
        {
            "forbidden",
            "historical_method_choice_used_fold4",
            "organizer_reference_used_for_model_fit",
        },
        label="source claims",
    )
    if claims != {
        "forbidden": list(FORBIDDEN_CLAIMS),
        "historical_method_choice_used_fold4": True,
        "organizer_reference_used_for_model_fit": False,
    }:
        raise ValueError("source claims differ")

    inputs = _require_exact_keys(
        root["inputs"], {"config", "candidate_pool", "gate1"}, label="inputs"
    )
    input_config = _require_exact_keys(
        inputs["config"], {"sha256", "size_bytes"}, label="inputs.config"
    )
    if input_config["sha256"] != contract.config_sha256:
        raise ValueError("source manifest does not bind the accepted scoring config")
    _require_int(input_config["size_bytes"], label="inputs.config.size_bytes", minimum=1)
    candidate_pool = _require_exact_keys(
        inputs["candidate_pool"],
        {
            "producer_job_id",
            "git_commit",
            "publication_top_sha256",
            "manifest_sha256",
            "candidates_sha256",
            "validation_summary_sha256",
            "final_publication_check_sha256",
            "candidates",
        },
        label="inputs.candidate_pool",
    )
    if candidate_pool["candidates"] != config.expected_candidates:
        raise ValueError("source candidate-pool census differs")
    for key in (
        "publication_top_sha256",
        "manifest_sha256",
        "candidates_sha256",
        "validation_summary_sha256",
        "final_publication_check_sha256",
    ):
        _require_sha(candidate_pool[key], label=f"inputs.candidate_pool.{key}")
    _require_git_sha(candidate_pool["git_commit"], label="candidate-pool git_commit")
    _require_int(candidate_pool["producer_job_id"], label="candidate-pool producer job", minimum=1)
    gate1 = _require_exact_keys(
        inputs["gate1"],
        {
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "publication_top_sha256",
            "semantic_top_sha256",
            "examples_sha256",
            "oof_sha256",
            "independent_receipt_sha256",
            "examples",
        },
        label="inputs.gate1",
    )
    if gate1["examples"] != config.expected_examples:
        raise ValueError("source Gate-1 example census differs")
    for key in (
        "publication_top_sha256",
        "semantic_top_sha256",
        "examples_sha256",
        "oof_sha256",
        "independent_receipt_sha256",
    ):
        _require_sha(gate1[key], label=f"inputs.gate1.{key}")
    _require_git_sha(gate1["git_commit"], label="Gate-1 git_commit")
    _require_int(gate1["producer_job_id"], label="Gate-1 producer job", minimum=1)
    _require_int(gate1["audit_job_id"], label="Gate-1 audit job", minimum=1)

    artifacts = _require_exact_keys(
        root["artifacts"],
        {
            "candidate_activity_scores.csv",
            "fold_model_target_probabilities.npy",
            "model_states.json",
            "oof_reproduction.json",
        },
        label="source artifacts",
    )
    for name in (
        "candidate_activity_scores.csv",
        "fold_model_target_probabilities.npy",
        "model_states.json",
        "oof_reproduction.json",
    ):
        _validate_artifact_record(
            artifacts[name],
            snapshot=snapshots[name],
            label=f"source artifact {name}",
            rows=config.expected_candidates if name == "candidate_activity_scores.csv" else None,
        )


def _validate_model_states(
    document: Mapping[str, Any], *, config: CandidateActivityMeanLedgerConfig
) -> None:
    root = _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact",
            "descriptor_settings",
            "feature_order",
            "states",
            "target_order",
        },
        label="model states",
    )
    if (
        root["schema_version"] != 1
        or root["artifact"] != "candidate_activity_descriptor_model_states_v1"
    ):
        raise ValueError("model states identity differs")
    if root["target_order"] != [name for name, _ in TARGET_PANEL]:
        raise ValueError("model states target order differs")
    settings = _require_exact_keys(
        root["descriptor_settings"],
        {"l2", "max_iterations", "prior_strength", "tolerance"},
        label="descriptor settings",
    )
    if settings != {"l2": 0.1, "max_iterations": 100, "prior_strength": 2.0, "tolerance": 1e-9}:
        raise ValueError("descriptor settings differ")
    if not isinstance(root["feature_order"], list) or not root["feature_order"]:
        raise ValueError("model feature order must be a non-empty list")
    states = root["states"]
    if not isinstance(states, list) or len(states) != config.folds + 1:
        raise ValueError("model states must contain five folds and deployment")
    expected_names = [f"outer_fold_{index}" for index in range(config.folds)] + [
        "all_data_deployment"
    ]
    for index, state_value in enumerate(states):
        state = _require_exact_keys(
            state_value,
            {
                "coefficient",
                "constant_probability",
                "heldout_fold",
                "mean",
                "name",
                "scale",
                "strains",
                "training_example_ids_sha256",
                "training_examples",
                "training_sequences",
                "training_union_components",
            },
            label=f"model state {index}",
        )
        if state["name"] != expected_names[index] or state["strains"] != [
            name for name, _ in TARGET_PANEL
        ]:
            raise ValueError(f"model state {index} identity differs")
        expected_fold = index if index < config.folds else None
        if state["heldout_fold"] != expected_fold:
            raise ValueError(f"model state {index} held-out fold differs")
        _require_sha(
            state["training_example_ids_sha256"],
            label=f"model state {index} training IDs",
        )
        for key in ("training_examples", "training_sequences", "training_union_components"):
            _require_int(state[key], label=f"model state {index} {key}", minimum=1)
        for key in ("coefficient", "mean", "scale"):
            values = state[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"model state {index} {key} must be a non-empty list")
            if any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"model state {index} {key} contains a non-finite value")


def _validate_oof(
    document: Mapping[str, Any], *, config: CandidateActivityMeanLedgerConfig
) -> None:
    root = _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact",
            "absolute_tolerance",
            "accepted_model",
            "by_fold",
            "examples",
            "folds",
            "maximum_absolute_error",
            "status",
        },
        label="OOF reproduction",
    )
    literals = {
        "schema_version": 1,
        "artifact": "candidate_activity_descriptor_oof_reproduction_v1",
        "absolute_tolerance": 1e-12,
        "accepted_model": "descriptor_logistic",
        "examples": config.expected_examples,
        "folds": config.folds,
        "maximum_absolute_error": 0.0,
        "status": "passed_exact_descriptor_oof_reproduction",
    }
    if any(root[key] != expected for key, expected in literals.items()):
        raise ValueError("OOF reproduction declaration differs")
    by_fold = root["by_fold"]
    if not isinstance(by_fold, list) or len(by_fold) != config.folds:
        raise ValueError("OOF by-fold census differs")
    heldout_total = 0
    for index, value in enumerate(by_fold):
        fold = _require_exact_keys(
            value,
            {"fold", "heldout_examples", "maximum_absolute_error", "training_examples"},
            label=f"OOF fold {index}",
        )
        if fold["fold"] != index or fold["maximum_absolute_error"] != 0.0:
            raise ValueError(f"OOF fold {index} identity or error differs")
        heldout = _require_int(fold["heldout_examples"], label="heldout examples", minimum=1)
        training = _require_int(fold["training_examples"], label="training examples", minimum=1)
        if heldout + training != config.expected_examples:
            raise ValueError(f"OOF fold {index} census is inconsistent")
        heldout_total += heldout
    if heldout_total != config.expected_examples:
        raise ValueError("OOF held-out folds do not partition all examples")


def _validate_fold_tensor(
    snapshot: InputSnapshot, *, config: CandidateActivityMeanLedgerConfig
) -> NDArray[np.float64]:
    try:
        stream = io.BytesIO(snapshot.payload)
        values = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("source fold tensor is not a valid non-pickle NPY array") from error
    if stream.tell() != len(snapshot.payload):
        raise ValueError("source fold tensor has trailing bytes")
    if values.shape != (config.expected_candidates, config.folds, config.targets):
        raise ValueError("source fold tensor shape differs")
    if values.dtype.str != "<f8" or not values.flags.c_contiguous:
        raise ValueError("source fold tensor must be little-endian C-contiguous float64")
    if not bool(np.all(np.isfinite(values))) or not bool(np.all((values >= 0.0) & (values <= 1.0))):
        raise ValueError("source fold tensor values must be finite probabilities")
    raw = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    if raw != config.accepted_scoring.fold_tensor_raw_data_sha256:
        raise ValueError("source fold tensor raw-data hash differs")
    return values


def _canonical_float_text(value: str, *, label: str, upper_bound: float) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{label} is not a canonical float")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= upper_bound:
        raise ValueError(f"{label} lies outside [0, {upper_bound}]")
    if value != f"{parsed:.17g}":
        raise ValueError(f"{label} does not use the canonical float spelling")
    return value


def _canonical_lineage(value: str, *, label: str) -> str:
    parts = value.split("|")
    if not parts or any(_LINEAGE_TOKEN_RE.fullmatch(part) is None for part in parts):
        raise ValueError(f"{label} is malformed")
    if parts != sorted(set(parts)):
        raise ValueError(f"{label} must be sorted and unique")
    return value


def _read_source_rows(
    snapshot: InputSnapshot,
    *,
    config: CandidateActivityMeanLedgerConfig,
    fold_tensor: NDArray[np.float64],
) -> tuple[SourceRow, ...]:
    payload = snapshot.payload
    if b"\r" in payload or b"\x00" in payload or not payload.endswith(b"\n"):
        raise ValueError("source candidate CSV must be canonical UTF-8/LF")
    try:
        text = payload.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        records = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("source candidate CSV is malformed") from error
    if not records or tuple(records[0]) != SOURCE_COLUMNS:
        raise ValueError("source candidate CSV columns or order differ")
    if any(not row for row in records[1:]):
        raise ValueError("source candidate CSV cannot contain blank records")
    if _csv_bytes(records[0], records[1:]) != payload:
        raise ValueError("source candidate CSV bytes are not canonically framed")
    if len(records) - 1 != config.expected_candidates:
        raise ValueError("source candidate CSV row count differs")
    rows: list[SourceRow] = []
    sequence_ids: set[str] = set()
    sequences: set[str] = set()
    target_start = 7
    mean_start = target_start + config.targets
    diagnostic_start = mean_start + len(OBJECTIVES)
    for expected_ordinal, values in enumerate(records[1:], start=1):
        row_number = expected_ordinal + 1
        if len(values) != len(SOURCE_COLUMNS):
            raise ValueError(f"source candidate CSV row {row_number} width differs")
        if values[0] != str(expected_ordinal):
            raise ValueError(f"source candidate CSV row {row_number} ordinal differs")
        sequence = canonicalize_sequence(values[2])
        if sequence != values[2]:
            raise ValueError(f"source candidate CSV row {row_number} sequence is not canonical")
        expected_id = canonical_sequence_id(sequence)
        if values[1] != expected_id:
            raise ValueError(f"source candidate CSV row {row_number} sequence ID differs")
        if values[1] in sequence_ids or sequence in sequences:
            raise ValueError(f"source candidate CSV row {row_number} duplicates a candidate")
        sequence_ids.add(values[1])
        sequences.add(sequence)
        if values[3] != str(len(sequence)):
            raise ValueError(f"source candidate CSV row {row_number} length differs")
        if values[4] not in {"true", "false"}:
            raise ValueError(f"source candidate CSV row {row_number} eligibility is not canonical")
        families = _canonical_lineage(values[5], label=f"row {row_number} generator families")
        variants = _canonical_lineage(values[6], label=f"row {row_number} generator variants")
        target_values = tuple(
            _canonical_float_text(
                value, label=f"row {row_number} target probability", upper_bound=1.0
            )
            for value in values[target_start:mean_start]
        )
        means = tuple(
            _canonical_float_text(value, label=f"row {row_number} objective mean", upper_bound=1.0)
            for value in values[mean_start:diagnostic_start]
        )
        target_numbers = np.asarray([float(value) for value in target_values], dtype=np.float64)
        expected_means = (
            f"{float(np.mean(target_numbers)):.17g}",
            f"{float(np.mean(target_numbers[[1, 2, 6]])):.17g}",
            f"{float(np.mean(target_numbers[[0, 3, 4, 5]])):.17g}",
        )
        if means != expected_means:
            raise ValueError(f"source candidate CSV row {row_number} objective arithmetic differs")
        diagnostics = tuple(
            _canonical_float_text(
                value,
                label=f"row {row_number} model-fit sensitivity",
                upper_bound=0.5,
            )
            for value in values[diagnostic_start : diagnostic_start + len(OBJECTIVES)]
        )
        candidate_fold_values = fold_tensor[expected_ordinal - 1]
        fold_objectives = (
            np.mean(candidate_fold_values, axis=1),
            np.mean(candidate_fold_values[:, [1, 2, 6]], axis=1),
            np.mean(candidate_fold_values[:, [0, 3, 4, 5]], axis=1),
        )
        expected_diagnostics = tuple(
            f"{float(np.std(values_for_objective, ddof=0)):.17g}"
            for values_for_objective in fold_objectives
        )
        if diagnostics != expected_diagnostics:
            raise ValueError(
                f"source candidate CSV row {row_number} model-fit diagnostic arithmetic differs"
            )
        scope_start = diagnostic_start + len(OBJECTIVES)
        if values[scope_start:] != [
            PREDICTION_SCOPE,
            TRAINING_SCOPE,
            CALIBRATION_SCOPE,
            config.accepted_scoring.model_release_id,
        ]:
            raise ValueError(f"source candidate CSV row {row_number} scope or release differs")
        rows.append(
            SourceRow(
                source_ordinal=values[0],
                sequence_id=values[1],
                sequence=sequence,
                length=values[3],
                library_eligible=values[4],
                generator_families=families,
                generator_variants=variants,
                target_probabilities=target_values,
                means=means,
                diagnostics=diagnostics,
            )
        )
    return tuple(rows)


def _validate_independent_receipt(
    receipt: Mapping[str, Any],
    *,
    config: CandidateActivityMeanLedgerConfig,
    source_manifest: Mapping[str, Any],
    source_snapshots: Mapping[str, InputSnapshot],
) -> None:
    root = _require_exact_keys(
        receipt,
        {
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "git_commit",
            "census",
            "checks",
            "claims",
            "input_sha256",
            "maximum_absolute_error",
            "model_release_id",
            "output_sha256",
        },
        label="independent verification receipt",
    )
    if (
        root["schema_version"] != 1
        or root["artifact"] != INDEPENDENT_ARTIFACT
        or root["status"] != "passed"
        or root["automatic_production_eligible"] is not False
        or root["git_commit"] != config.accepted_scoring.git_commit
        or root["model_release_id"] != config.accepted_scoring.model_release_id
    ):
        raise ValueError("independent verification receipt identity differs")
    census = _require_exact_keys(
        root["census"],
        {"candidates", "examples", "fold_tensor_shape", "folds", "targets"},
        label="receipt census",
    )
    if census != {
        "candidates": config.expected_candidates,
        "examples": config.expected_examples,
        "fold_tensor_shape": [config.expected_candidates, config.folds, config.targets],
        "folds": config.folds,
        "targets": config.targets,
    }:
        raise ValueError("independent receipt census differs")
    check_keys = {
        "candidate_identity_and_order_exact",
        "candidate_lineage_summaries_exact",
        "config_and_input_hashes_authenticated",
        "descriptor_features_reimplemented_independently",
        "descriptor_states_refit_independently",
        "forbidden_endpoint_and_uncertainty_claims_absent",
        "manifest_and_checksum_inventory_exact",
        "model_release_bound_to_states",
        "oof_descriptor_probabilities_reproduced",
        "output_bounds_and_finiteness_valid",
        "score_objective_arithmetic_reproduced",
        "tensor_axes_and_state_predictions_reproduced",
    }
    checks = _require_exact_keys(root["checks"], check_keys, label="receipt checks")
    if any(value is not True for value in checks.values()):
        raise ValueError("not every independent verification check passed")
    claims = _require_exact_keys(
        root["claims"],
        {
            "development_candidate_scoring_only",
            "model_fit_sensitivity_is_calibrated_uncertainty",
            "production_ensemble",
        },
        label="receipt claims",
    )
    if claims != {
        "development_candidate_scoring_only": True,
        "model_fit_sensitivity_is_calibrated_uncertainty": False,
        "production_ensemble": False,
    }:
        raise ValueError("independent receipt claims differ")
    inputs = _require_exact_keys(
        root["input_sha256"],
        {"candidate_jsonl", "config", "examples", "oof"},
        label="receipt inputs",
    )
    manifest_inputs = source_manifest["inputs"]
    if inputs != {
        "candidate_jsonl": manifest_inputs["candidate_pool"]["candidates_sha256"],
        "config": config.accepted_scoring.config_sha256,
        "examples": manifest_inputs["gate1"]["examples_sha256"],
        "oof": manifest_inputs["gate1"]["oof_sha256"],
    }:
        raise ValueError("independent receipt input hashes differ")
    errors = _require_exact_keys(
        root["maximum_absolute_error"],
        {
            "accepted_descriptor_oof",
            "candidate_fold_tensor_from_states",
            "candidate_score_arithmetic",
        },
        label="receipt numerical errors",
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0
        for value in errors.values()
    ):
        raise ValueError("independent receipt numerical errors are invalid")
    output = _require_exact_keys(
        root["output_sha256"], set(SOURCE_FILES), label="receipt output hashes"
    )
    if output != {name: source_snapshots[name].sha256 for name in SOURCE_FILES}:
        raise ValueError("independent receipt output hashes do not bind the scoring bundle")


def _validate_operational_receipt(
    receipt: Mapping[str, Any], *, config: CandidateActivityMeanLedgerConfig
) -> None:
    root = _require_exact_keys(
        receipt,
        {
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "git_commit",
            "audit",
            "checks",
            "independent_receipt_sha256",
            "producer",
            "twin_tree_sha256",
        },
        label="operational audit receipt",
    )
    contract = config.accepted_scoring
    if (
        root["schema_version"] != 1
        or root["artifact"] != OPERATIONAL_ARTIFACT
        or root["status"] != "passed"
        or root["automatic_production_eligible"] is not False
        or root["git_commit"] != contract.git_commit
    ):
        raise ValueError("operational audit receipt identity differs")
    checks = _require_exact_keys(
        root["checks"],
        {
            "audit_node_excludes_both_producer_nodes",
            "each_twin_independently_reconstructed_before_comparison",
            "producer_completed_in_slurm_accounting",
            "repository_clean_and_live_origin_synchronized",
            "twins_byte_identical",
        },
        label="operational checks",
    )
    if any(value is not True for value in checks.values()):
        raise ValueError("not every operational audit check passed")
    independent_hashes = _require_exact_keys(
        root["independent_receipt_sha256"], {"0", "1"}, label="independent receipt hashes"
    )
    tree_hashes = _require_exact_keys(
        root["twin_tree_sha256"], {"0", "1"}, label="twin tree hashes"
    )
    if independent_hashes != {
        "0": contract.independent_verification_sha256,
        "1": contract.independent_verification_sha256,
    }:
        raise ValueError("operational receipt independent hashes differ")
    if tree_hashes != {
        "0": contract.publication_tree_sha256,
        "1": contract.publication_tree_sha256,
    }:
        raise ValueError("operational receipt twin tree hashes differ")
    audit = _require_exact_keys(
        root["audit"], {"job_id", "node_name", "resources"}, label="audit execution"
    )
    if audit["job_id"] != contract.audit_job_id:
        raise ValueError("operational receipt audit job differs")
    audit_node = _require_string(audit["node_name"], label="audit node")
    audit_resources = _require_exact_keys(
        audit["resources"],
        {"account", "cpus_per_task", "gpus", "memory_per_node_mib", "nodes", "partition", "tasks"},
        label="audit resources",
    )
    if audit_resources != {
        "account": "bio",
        "cpus_per_task": 4,
        "gpus": 0,
        "memory_per_node_mib": 16384,
        "nodes": 1,
        "partition": "standard",
        "tasks": 1,
    }:
        raise ValueError("operational audit resources differ")
    producer = _require_exact_keys(
        root["producer"],
        {"job_id", "nodes", "observed_telemetry", "resources"},
        label="producer execution",
    )
    if producer["job_id"] != contract.producer_job_id:
        raise ValueError("operational receipt producer job differs")
    nodes = producer["nodes"]
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValueError("operational receipt must name two distinct producer nodes")
    if any(not isinstance(node, str) or not node for node in nodes):
        raise ValueError("operational receipt producer node names are invalid")
    if len(set(nodes)) != 2 or audit_node in nodes:
        raise ValueError("operational audit node must exclude both producer nodes")
    resources = _require_exact_keys(
        producer["resources"], {"cpus", "gpus", "nodes", "tasks"}, label="producer resources"
    )
    if resources != {"cpus": 8, "gpus": 0, "nodes": 2, "tasks": 2}:
        raise ValueError("operational producer resources differ")
    telemetry = _require_exact_keys(
        producer["observed_telemetry"],
        {"compute_step_elapsed_seconds", "compute_step_max_rss", "job_elapsed_seconds"},
        label="producer telemetry",
    )
    _require_int(telemetry["compute_step_elapsed_seconds"], label="compute elapsed", minimum=1)
    _require_int(telemetry["job_elapsed_seconds"], label="job elapsed", minimum=1)
    max_rss = _require_exact_keys(
        telemetry["compute_step_max_rss"], {"type", "unit", "value"}, label="producer MaxRSS"
    )
    if max_rss["type"] != "slurm_max_rss" or not all(
        isinstance(max_rss[key], str) and max_rss[key] for key in ("unit", "value")
    ):
        raise ValueError("producer MaxRSS telemetry differs")


def authenticate_candidate_activity_scoring(
    *,
    scoring_twin: str | Path,
    audit_dir: str | Path,
    config: CandidateActivityMeanLedgerConfig,
) -> AuthenticatedScoring:
    """Authenticate the exact accepted six-file scorer and three-file audit."""

    source_directory, source = _snapshot_flat_bundle(
        scoring_twin, names=SOURCE_FILES, label="scoring twin"
    )
    contract = config.accepted_scoring
    expected_hashes = {
        "SHA256SUMS": contract.publication_top_sha256,
        "candidate_activity_scores.csv": contract.candidate_scores_sha256,
        "fold_model_target_probabilities.npy": contract.fold_tensor_sha256,
        "manifest.json": contract.manifest_sha256,
        "model_states.json": contract.model_states_sha256,
        "oof_reproduction.json": contract.oof_reproduction_sha256,
    }
    for name, expected in expected_hashes.items():
        if source[name].sha256 != expected:
            raise ValueError(f"authenticated scoring file hash differs: {name}")
    _validate_sha256sums(
        source["SHA256SUMS"], snapshots=source, expected_top=contract.publication_top_sha256
    )
    if _bundle_tree_sha256(source) != contract.publication_tree_sha256:
        raise ValueError("scoring twin tree hash differs from the accepted tree")

    manifest = _load_canonical_json(source["manifest.json"], label="source manifest")
    _validate_source_manifest(manifest, snapshots=source, config=config)
    model_states = _load_canonical_json(source["model_states.json"], label="model states")
    _validate_model_states(model_states, config=config)
    oof = _load_canonical_json(source["oof_reproduction.json"], label="OOF reproduction")
    _validate_oof(oof, config=config)
    fold_tensor = _validate_fold_tensor(
        source["fold_model_target_probabilities.npy"], config=config
    )
    rows = _read_source_rows(
        source["candidate_activity_scores.csv"],
        config=config,
        fold_tensor=fold_tensor,
    )

    audit_directory, audit = _snapshot_flat_bundle(
        audit_dir, names=AUDIT_FILES, label="scoring audit"
    )
    if audit["operational-receipt.json"].sha256 != contract.operational_audit_sha256:
        raise ValueError("operational audit receipt hash differs")
    for name in ("twin-0-independent-verification.json", "twin-1-independent-verification.json"):
        if audit[name].sha256 != contract.independent_verification_sha256:
            raise ValueError(f"independent verification receipt hash differs: {name}")
    independent_zero = _load_canonical_json(
        audit["twin-0-independent-verification.json"], label="twin-0 independent receipt"
    )
    independent_one = _load_canonical_json(
        audit["twin-1-independent-verification.json"], label="twin-1 independent receipt"
    )
    if independent_zero != independent_one:
        raise ValueError("independent twin verification receipts differ")
    _validate_independent_receipt(
        independent_zero,
        config=config,
        source_manifest=manifest,
        source_snapshots=source,
    )
    operational = _load_canonical_json(
        audit["operational-receipt.json"], label="operational audit receipt"
    )
    _validate_operational_receipt(operational, config=config)
    return AuthenticatedScoring(
        rows=rows,
        manifest=manifest,
        snapshots={
            **{f"scoring/{name}": value for name, value in source.items()},
            **{f"audit/{name}": value for name, value in audit.items()},
        },
        directory_snapshots={
            "scoring directory": source_directory,
            "audit directory": audit_directory,
        },
    )


def _csv_bytes(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise AssertionError("generated CSV is not canonical UTF-8/LF")
    return payload


def _candidate_ledger_bytes_with_release(
    rows: Sequence[SourceRow], *, model_release_id: str
) -> bytes:
    output_rows = [
        (
            row.source_ordinal,
            row.sequence_id,
            row.sequence,
            row.length,
            row.library_eligible,
            ELIGIBILITY_SCOPE,
            row.generator_families,
            row.generator_variants,
            *row.target_probabilities,
            *row.means,
            UNCERTAINTY_STATUS,
            PREDICTION_SCOPE,
            TRAINING_SCOPE,
            CALIBRATION_SCOPE,
            model_release_id,
        )
        for row in rows
    ]
    if any(
        column.startswith("std_") or column.startswith("model_fit_sensitivity_")
        for column in LEDGER_COLUMNS
    ):
        raise AssertionError(
            "candidate ledger cannot expose standard deviation or sensitivity fields"
        )
    return _csv_bytes(LEDGER_COLUMNS, output_rows)


def _diagnostics_bytes(rows: Sequence[SourceRow]) -> bytes:
    return _csv_bytes(
        DIAGNOSTIC_COLUMNS,
        [(row.source_ordinal, row.sequence_id, *row.diagnostics, DIAGNOSTIC_SCOPE) for row in rows],
    )


def _artifact_record(payload: bytes, *, rows: int) -> Mapping[str, object]:
    return {
        "rows": rows,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _manifest_document(
    *,
    config: CandidateActivityMeanLedgerConfig,
    candidate_payload: bytes,
    diagnostics_payload: bytes,
) -> Mapping[str, object]:
    source = config.accepted_scoring
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": OUTPUT_STATUS,
        "automatic_production_eligible": False,
        "allowed_consumer": ALLOWED_CONSUMER,
        "adapter_config_sha256": config.snapshot.sha256,
        "model_release_id": source.model_release_id,
        "scopes": {
            "calibration": CALIBRATION_SCOPE,
            "eligibility": ELIGIBILITY_SCOPE,
            "prediction": PREDICTION_SCOPE,
            "training": TRAINING_SCOPE,
            "uncertainty_status": UNCERTAINTY_STATUS,
        },
        "objectives": list(OBJECTIVES),
        "targets": [
            {
                "gram": gram,
                "index": index,
                "name": name,
                "probability_column": f"probability_{name}",
            }
            for index, (name, gram) in enumerate(TARGET_PANEL)
        ],
        "source": {
            "artifact": SOURCE_ARTIFACT,
            "git_commit": source.git_commit,
            "config_sha256": source.config_sha256,
            "publication_top_sha256": source.publication_top_sha256,
            "publication_tree_sha256": source.publication_tree_sha256,
            "manifest_sha256": source.manifest_sha256,
            "candidate_scores_sha256": source.candidate_scores_sha256,
            "fold_tensor_sha256": source.fold_tensor_sha256,
            "fold_tensor_raw_data_sha256": source.fold_tensor_raw_data_sha256,
            "model_states_sha256": source.model_states_sha256,
            "oof_reproduction_sha256": source.oof_reproduction_sha256,
            "independent_verification_sha256": source.independent_verification_sha256,
            "operational_audit_sha256": source.operational_audit_sha256,
        },
        "candidate_ledger": {
            "columns": list(LEDGER_COLUMNS),
            "eligibility_mapping": "eligible_is_exact_library_eligible_value",
            "filename": "candidate_ledger.csv",
            **_artifact_record(candidate_payload, rows=config.expected_candidates),
        },
        "diagnostics": {
            "columns": list(DIAGNOSTIC_COLUMNS),
            "filename": "model_fit_sensitivity_diagnostics.csv",
            "scope": DIAGNOSTIC_SCOPE,
            "selectable": False,
            "uncertainty": False,
            **_artifact_record(diagnostics_payload, rows=config.expected_candidates),
        },
        "claims": {
            "diagnostics_are_uncertainty": False,
            "diagnostics_selectable": False,
            "final_ranking": False,
            "mean_only_development_handoff": True,
            "production_ensemble": False,
            "uncertainty_available": False,
        },
        "exclusions": {
            "candidate_ledger_column_prefixes": ["model_fit_sensitivity_", "std_"],
            "claims": list(FORBIDDEN_CLAIMS),
            "endpoints": list(EXCLUDED_ENDPOINTS),
            "weights": list(EXCLUDED_WEIGHTS),
        },
    }


def _sha256sums_bytes(payloads: Mapping[str, bytes]) -> bytes:
    return b"".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode("ascii")
        for name in sorted(payloads)
    )


def _write_new_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing mean-ledger artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_write_once(
    output_dir: str | Path,
    *,
    payloads: Mapping[str, bytes],
    snapshots: Mapping[str, InputSnapshot],
    directory_snapshots: Mapping[str, DirectorySnapshot] | None = None,
) -> Path:
    if set(payloads) != set(OUTPUT_FILES):
        raise ValueError("mean-ledger output inventory is not exact")
    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested):
        raise FileExistsError(f"refusing to reuse mean-ledger output: {requested}")
    parent = requested.parent
    _reject_symlink_chain(parent, label="mean-ledger output parent")
    try:
        parent_metadata = os.lstat(parent)
    except OSError as error:
        raise ValueError(f"cannot inspect mean-ledger output parent: {parent}") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("mean-ledger output parent must be a non-symbolic directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{requested.name}.staging-", dir=parent))
    claimed = False
    linked: list[str] = []
    authenticated_directories = directory_snapshots or {}

    def assert_authenticated_inputs_unchanged() -> None:
        for snapshot_label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=snapshot_label)
        # Check directories last so the final observation covers path identity,
        # mode, and inventory after every individual file observation.
        for snapshot_label, snapshot in authenticated_directories.items():
            _assert_directory_snapshot_unchanged(snapshot, label=snapshot_label)

    try:
        for name in OUTPUT_FILES:
            _write_new_bytes(staging / name, payloads[name])
            os.chmod(staging / name, 0o444)
        assert_authenticated_inputs_unchanged()
        os.mkdir(requested, mode=0o755)
        claimed = True
        for name in (*OUTPUT_FILES[:-1], "SHA256SUMS"):
            os.link(staging / name, requested / name, follow_symlinks=False)
            linked.append(name)
        assert_authenticated_inputs_unchanged()
        os.chmod(requested, 0o555)
        return requested
    except Exception:
        if claimed:
            for name in reversed(linked):
                with suppress(FileNotFoundError):
                    os.unlink(requested / name)
            with suppress(OSError):
                os.rmdir(requested)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _revalidate_authenticated_scoring_for_publication(
    *,
    config: CandidateActivityMeanLedgerConfig,
    authenticated: AuthenticatedScoring,
) -> tuple[SourceRow, ...]:
    """Reconstruct the authenticated handoff at the public write boundary."""

    reloaded_config = load_candidate_activity_mean_ledger_config(config.path)
    if reloaded_config != config:
        raise ValueError("mean-ledger config object does not match its authenticated bytes")
    expected_file_snapshots = {
        *(f"scoring/{name}" for name in SOURCE_FILES),
        *(f"audit/{name}" for name in AUDIT_FILES),
    }
    if set(authenticated.snapshots) != expected_file_snapshots:
        raise ValueError("authenticated scoring file snapshot inventory is not exact")
    expected_directory_labels = {"scoring directory", "audit directory"}
    if set(authenticated.directory_snapshots) != expected_directory_labels:
        raise ValueError("authenticated evidence directory snapshot inventory is not exact")

    source = {name: authenticated.snapshots[f"scoring/{name}"] for name in SOURCE_FILES}
    audit = {name: authenticated.snapshots[f"audit/{name}"] for name in AUDIT_FILES}
    source_directory = authenticated.directory_snapshots["scoring directory"]
    audit_directory = authenticated.directory_snapshots["audit directory"]
    if source_directory.entries != tuple(sorted(SOURCE_FILES)) or any(
        snapshot.path.parent != source_directory.path or snapshot.path.name != name
        for name, snapshot in source.items()
    ):
        raise ValueError("authenticated scoring directory does not bind the scoring files")
    if audit_directory.entries != tuple(sorted(AUDIT_FILES)) or any(
        snapshot.path.parent != audit_directory.path or snapshot.path.name != name
        for name, snapshot in audit.items()
    ):
        raise ValueError("authenticated audit directory does not bind the audit files")

    contract = config.accepted_scoring
    expected_source_hashes = {
        "SHA256SUMS": contract.publication_top_sha256,
        "candidate_activity_scores.csv": contract.candidate_scores_sha256,
        "fold_model_target_probabilities.npy": contract.fold_tensor_sha256,
        "manifest.json": contract.manifest_sha256,
        "model_states.json": contract.model_states_sha256,
        "oof_reproduction.json": contract.oof_reproduction_sha256,
    }
    for name, expected_hash in expected_source_hashes.items():
        snapshot = source[name]
        if (
            snapshot.mode != 0o444
            or snapshot.sha256 != expected_hash
            or hashlib.sha256(snapshot.payload).hexdigest() != expected_hash
        ):
            raise ValueError(f"publication source snapshot differs from its pin: {name}")
    expected_audit_hashes = {
        "operational-receipt.json": contract.operational_audit_sha256,
        "twin-0-independent-verification.json": contract.independent_verification_sha256,
        "twin-1-independent-verification.json": contract.independent_verification_sha256,
    }
    for name, expected_hash in expected_audit_hashes.items():
        snapshot = audit[name]
        if (
            snapshot.mode != 0o444
            or snapshot.sha256 != expected_hash
            or hashlib.sha256(snapshot.payload).hexdigest() != expected_hash
        ):
            raise ValueError(f"publication audit snapshot differs from its pin: {name}")
    _validate_sha256sums(
        source["SHA256SUMS"],
        snapshots=source,
        expected_top=contract.publication_top_sha256,
    )
    if _bundle_tree_sha256(source) != contract.publication_tree_sha256:
        raise ValueError("publication source snapshot tree differs from its pin")

    manifest = _load_canonical_json(source["manifest.json"], label="publication source manifest")
    _validate_source_manifest(manifest, snapshots=source, config=config)
    if authenticated.manifest != manifest:
        raise ValueError("passed source manifest does not match its authenticated bytes")
    model_states = _load_canonical_json(
        source["model_states.json"], label="publication model states"
    )
    _validate_model_states(model_states, config=config)
    oof = _load_canonical_json(
        source["oof_reproduction.json"], label="publication OOF reproduction"
    )
    _validate_oof(oof, config=config)
    fold_tensor = _validate_fold_tensor(
        source["fold_model_target_probabilities.npy"], config=config
    )
    reconstructed_rows = _read_source_rows(
        source["candidate_activity_scores.csv"],
        config=config,
        fold_tensor=fold_tensor,
    )
    if authenticated.rows != reconstructed_rows:
        raise ValueError("passed source rows do not match the authenticated scoring bytes")

    independent_zero = _load_canonical_json(
        audit["twin-0-independent-verification.json"],
        label="publication twin-0 independent receipt",
    )
    independent_one = _load_canonical_json(
        audit["twin-1-independent-verification.json"],
        label="publication twin-1 independent receipt",
    )
    if independent_zero != independent_one:
        raise ValueError("publication independent twin verification receipts differ")
    _validate_independent_receipt(
        independent_zero,
        config=config,
        source_manifest=manifest,
        source_snapshots=source,
    )
    operational = _load_canonical_json(
        audit["operational-receipt.json"],
        label="publication operational audit receipt",
    )
    _validate_operational_receipt(operational, config=config)
    return reconstructed_rows


def publish_candidate_activity_mean_ledger(
    *,
    output_dir: str | Path,
    config: CandidateActivityMeanLedgerConfig,
    authenticated: AuthenticatedScoring,
) -> CandidateActivityMeanLedgerExecution:
    """Project authenticated source rows and seal the exact four-file handoff."""

    reconstructed_rows = _revalidate_authenticated_scoring_for_publication(
        config=config,
        authenticated=authenticated,
    )
    if len(reconstructed_rows) != config.expected_candidates:
        raise ValueError("authenticated candidate count differs before publication")
    candidate_payload = _candidate_ledger_bytes_with_release(
        reconstructed_rows,
        model_release_id=config.accepted_scoring.model_release_id,
    )
    diagnostics_payload = _diagnostics_bytes(reconstructed_rows)
    manifest_payload = _canonical_json_bytes(
        _manifest_document(
            config=config,
            candidate_payload=candidate_payload,
            diagnostics_payload=diagnostics_payload,
        )
    )
    semantic_payloads = {
        "candidate_ledger.csv": candidate_payload,
        "model_fit_sensitivity_diagnostics.csv": diagnostics_payload,
        "manifest.json": manifest_payload,
    }
    payloads = {**semantic_payloads, "SHA256SUMS": _sha256sums_bytes(semantic_payloads)}
    output = _publish_write_once(
        output_dir,
        payloads=payloads,
        snapshots={"mean-ledger config": config.snapshot, **authenticated.snapshots},
        directory_snapshots=authenticated.directory_snapshots,
    )
    return CandidateActivityMeanLedgerExecution(
        output_dir=output,
        candidate_count=len(reconstructed_rows),
        candidate_ledger_sha256=hashlib.sha256(candidate_payload).hexdigest(),
        diagnostics_sha256=hashlib.sha256(diagnostics_payload).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        publication_top_sha256=hashlib.sha256(payloads["SHA256SUMS"]).hexdigest(),
    )


def run_candidate_activity_mean_ledger(
    *,
    scoring_twin: str | Path,
    audit_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> CandidateActivityMeanLedgerExecution:
    """Authenticate accepted evidence and publish the deterministic mean-only ledger."""

    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested):
        raise FileExistsError(f"refusing to reuse mean-ledger output: {requested}")
    config = load_candidate_activity_mean_ledger_config(config_path)
    authenticated = authenticate_candidate_activity_scoring(
        scoring_twin=scoring_twin,
        audit_dir=audit_dir,
        config=config,
    )
    return publish_candidate_activity_mean_ledger(
        output_dir=requested,
        config=config,
        authenticated=authenticated,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an authenticated development-only mean activity candidate ledger."
    )
    parser.add_argument("--scoring-twin", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = run_candidate_activity_mean_ledger(
            scoring_twin=args.scoring_twin,
            audit_dir=args.audit_dir,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"candidate activity mean-ledger error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_count": execution.candidate_count,
                "candidate_ledger_sha256": execution.candidate_ledger_sha256,
                "diagnostics_sha256": execution.diagnostics_sha256,
                "manifest_sha256": execution.manifest_sha256,
                "output_dir": str(execution.output_dir),
                "publication_top_sha256": execution.publication_top_sha256,
                "status": OUTPUT_STATUS,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the cluster CLI.
    raise SystemExit(main())

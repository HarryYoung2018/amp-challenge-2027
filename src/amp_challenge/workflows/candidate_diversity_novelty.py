"""Publish authenticated diversity, novelty, and reference-compliance features.

This CPU-only workflow extends the accepted mean-only activity ledger without
turning model-fit sensitivity into uncertainty.  It authenticates the accepted
mean ledger, its independent audit, the accepted training-only projection, and
the tracked organizer reference before calculating three deliberately narrow
quantities:

* an exact RapidFuzz Indel novelty against all accepted training sequences;
* the exact ``similarity > 0.80`` organizer-reference compliance predicate and
  a deterministic violating witness (but no claimed reference maximum); and
* a training-only standardized, row-normalized 33-dimensional physicochemical
  feature matrix for diversity-aware selection.

The five-file publication is write-once, read-only, byte deterministic across
RapidFuzz worker counts, and remains development evidence rather than a final
ranking, uncertainty model, endpoint ensemble, or safety assay.
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
import stat
import sys
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import process
from rapidfuzz.distance import Indel

from amp_challenge.descriptors import PeptideDescriptors, compute_descriptors
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

SCHEMA_VERSION = 1
ARTIFACT = "candidate_diversity_novelty_v1"
CONFIG_STATUS = "predeclared_development_diversity_novelty_handoff"
OUTPUT_STATUS = "development_diversity_novelty_evidence"
ALLOWED_CONSUMER = "development_diversity_novelty_selection_only"
UNCERTAINTY_STATUS = "unavailable"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
REFERENCE_THRESHOLD = 0.80

MEAN_ARTIFACT = "candidate_activity_mean_ledger_v1"
MEAN_STATUS = "development_mean_only_control"
MEAN_ALLOWED_CONSUMER = "mean_only_selection_control_only"
MEAN_INDEPENDENT_ARTIFACT = "candidate_activity_mean_ledger_v1_independent_verification"
MEAN_OPERATIONAL_ARTIFACT = "candidate_activity_mean_ledger_v1_operational_audit"
PROJECTION_RECEIPT_ARTIFACT = "native_diffusion_training_projection_stage_v1"

PREDICTION_SCOPE = "declared_seven_target_activity_panel"
TRAINING_SCOPE = "all_2492_accepted_gate1_contexts_after_recipe_freeze"
CALIBRATION_SCOPE = "none_raw_logistic_probability"
ELIGIBILITY_SCOPE = "accepted_candidate_pool_library_eligible_flag_v1"

OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
TARGET_PANEL = (
    "acinetobacter_baumannii",
    "enterococcus_faecalis",
    "enterococcus_faecium",
    "escherichia_coli",
    "klebsiella_pneumoniae",
    "pseudomonas_aeruginosa",
    "staphylococcus_aureus",
)
DESCRIPTOR_NAMES = tuple(field.name for field in fields(PeptideDescriptors))
FEATURE_NAMES = (*DESCRIPTOR_NAMES, *(f"residue_fraction_{residue}" for residue in ALPHABET))
EMBEDDING_COLUMNS = tuple(f"embedding_physchem_{index:03d}" for index in range(33))

MEAN_FILES = (
    "SHA256SUMS",
    "candidate_ledger.csv",
    "manifest.json",
    "model_fit_sensitivity_diagnostics.csv",
)
MEAN_AUDIT_FILES = (
    "operational-receipt.json",
    "twin-0-independent-verification.json",
    "twin-1-independent-verification.json",
)
TRAINING_STAGE_FILES = ("projection-receipt.json", "training_projection.jsonl")
OUTPUT_FILES = (
    "candidate_selection_ledger.csv",
    "diversity_feature_matrix.f32le",
    "feature_transform.json",
    "manifest.json",
    "SHA256SUMS",
)

MEAN_LEDGER_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    "sequence",
    "length",
    "eligible",
    "eligibility_scope",
    "generator_families",
    "generator_variants",
    *(f"probability_{name}" for name in TARGET_PANEL),
    *(f"mean_{name}" for name in OBJECTIVES),
    "uncertainty_status",
    "prediction_scope",
    "training_scope",
    "calibration_scope",
    "model_release_id",
)
SELECTION_LEDGER_COLUMNS = (
    *("source_library_eligible" if name == "eligible" else name for name in MEAN_LEDGER_COLUMNS),
    "max_train_indel_similarity",
    "nearest_train_sequence_id",
    "novelty",
    "reference_safe_0_80",
    "reference_violation_witness_sequence_id",
    "reference_violation_witness_similarity",
    "eligible",
    *EMBEDDING_COLUMNS,
)

FEATURE_SPACE = "physchem_descriptors_plus_canonical_amino_acid_fractions_v1"
TRAINING_WEIGHTING = "sampling_weight"
MEAN_ALGORITHM = "math_fsum_of_weight_times_value_divided_by_math_fsum_weights"
SCALE_ALGORITHM = (
    "population_sd_sqrt_math_fsum_weight_times_squared_centered_value_divided_by_math_fsum_weights"
)
STANDARDIZATION = (
    "candidate_raw_minus_training_weighted_mean_divided_by_training_weighted_population_sd"
)
ROW_NORMALIZATION = "euclidean_l2_after_standardization"
SIMILARITY_IMPLEMENTATION = "rapidfuzz.distance.Indel.normalized_similarity"
LENGTH_BOUND = "upper_bound_two_times_min_length_divided_by_sum_lengths"

FORBIDDEN_LEDGER_PREFIXES = (
    "cluster_",
    "model_fit_sensitivity_",
    "std_",
)
FORBIDDEN_CLAIMS = (
    "calibrated_uncertainty",
    "cluster_assignment",
    "esm_embedding",
    "final_ranking",
    "hemolysis_risk",
    "mdr_eskape_activity",
    "out_of_distribution",
    "production_ensemble",
    "quality_probability",
    "reference_witness_is_reference_maximum",
    "selectivity",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_LINEAGE_TOKEN_RE = re.compile(r"[^|,\r\n\x00-\x20\x7f]{1,256}")


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    """Immutable bytes plus the filesystem identity observed while reading."""

    path: Path
    payload: bytes
    sha256: str
    mode: int
    fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DirectorySnapshot:
    """Exact inventory and filesystem identity of a sealed flat directory."""

    path: Path
    entries: tuple[str, ...]
    fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AcceptedMeanLedgerContract:
    producer_job_id: int
    audit_job_id: int
    git_commit: str
    config_sha256: str
    publication_top_sha256: str
    publication_tree_sha256: str
    candidate_ledger_sha256: str
    diagnostics_sha256: str
    manifest_sha256: str
    independent_verification_sha256: str
    operational_audit_sha256: str
    model_release_id: str


@dataclass(frozen=True, slots=True)
class AcceptedTrainingProjectionContract:
    stage_job_id: int
    git_commit: str
    projection_sha256: str
    projection_size_bytes: int
    receipt_sha256: str
    receipt_size_bytes: int


@dataclass(frozen=True, slots=True)
class OrganizerReferenceContract:
    sha256: str
    size_bytes: int
    source_commit: str
    role: str


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    candidate_chunk_size: int
    choice_chunk_size: int
    worker_default: int
    worker_minimum: int
    worker_maximum: int


@dataclass(frozen=True, slots=True)
class CandidateDiversityNoveltyConfig:
    path: Path
    snapshot: InputSnapshot
    expected_candidates: int
    expected_training_rows: int
    expected_reference_records: int
    mean_ledger: AcceptedMeanLedgerContract
    training_projection: AcceptedTrainingProjectionContract
    organizer_reference: OrganizerReferenceContract
    policy: FeaturePolicy


@dataclass(frozen=True, slots=True)
class MeanLedgerRow:
    values: tuple[str, ...]

    @property
    def source_ordinal(self) -> str:
        return self.values[0]

    @property
    def sequence_id(self) -> str:
        return self.values[1]

    @property
    def sequence(self) -> str:
        return self.values[2]

    @property
    def source_library_eligible(self) -> str:
        return self.values[4]


@dataclass(frozen=True, slots=True)
class TrainingRow:
    sequence_id: str
    sequence: str
    sampling_weight: float


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    sequence_id: str
    sequence: str


@dataclass(frozen=True, slots=True)
class AuthenticatedDiversityInputs:
    mean_rows: tuple[MeanLedgerRow, ...]
    training_rows: tuple[TrainingRow, ...]
    reference_rows: tuple[ReferenceRow, ...]
    reference_record_count: int
    mean_manifest: Mapping[str, object]
    snapshots: Mapping[str, InputSnapshot]
    directory_snapshots: Mapping[str, DirectorySnapshot]


@dataclass(frozen=True, slots=True)
class CandidateSimilarity:
    max_train_similarity: float
    nearest_train_sequence_id: str
    reference_safe: bool
    reference_witness_sequence_id: str
    reference_witness_similarity: float | None


@dataclass(frozen=True, slots=True)
class ComputedDiversityBundle:
    selection_ledger: bytes
    feature_matrix: bytes
    feature_transform: bytes
    manifest: bytes


@dataclass(frozen=True, slots=True)
class CandidateDiversityNoveltyExecution:
    output_dir: Path
    candidate_count: int
    selection_ledger_sha256: str
    feature_matrix_sha256: str
    feature_transform_sha256: str
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
    current = _snapshot_file(snapshot.path, label=label, required_mode=snapshot.mode)
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
    after = os.lstat(absolute)
    if _fingerprint(metadata) != _fingerprint(after):
        raise ValueError(f"{label} changed while its inventory was captured")
    return DirectorySnapshot(absolute, entries, _fingerprint(metadata))


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
    expected = tuple(sorted(names))
    if directory.entries != expected:
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
    if after.fingerprint != directory.fingerprint or after.entries != expected:
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_canonical_json(snapshot: InputSnapshot, *, label: str) -> Mapping[str, Any]:
    if b"\r" in snapshot.payload or not snapshot.payload.endswith(b"\n"):
        raise ValueError(f"{label} must be canonical UTF-8 with LF termination")
    try:
        document = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
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


def load_candidate_diversity_novelty_config(
    path: str | Path,
) -> CandidateDiversityNoveltyConfig:
    """Load and snapshot the exact diversity/novelty publication contract."""

    snapshot = _snapshot_file(path, label="diversity/novelty config")
    if b"\r" in snapshot.payload or not snapshot.payload.endswith(b"\n"):
        raise ValueError("diversity/novelty config must use UTF-8/LF and end with LF")
    try:
        document = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("diversity/novelty config is not valid UTF-8 TOML") from error
    root = _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "allowed_consumer",
            "expected_candidates",
            "expected_training_rows",
            "expected_reference_records",
            "alphabet",
            "uncertainty_status",
            "reference_similarity_threshold",
            "accepted_mean_ledger",
            "accepted_training_projection",
            "organizer_reference",
            "features",
            "similarity",
            "claims",
        },
        label="diversity/novelty config",
    )
    literals = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": CONFIG_STATUS,
        "automatic_production_eligible": False,
        "allowed_consumer": ALLOWED_CONSUMER,
        "alphabet": ALPHABET,
        "uncertainty_status": UNCERTAINTY_STATUS,
        "reference_similarity_threshold": REFERENCE_THRESHOLD,
    }
    for key, expected in literals.items():
        if root[key] != expected or type(root[key]) is not type(expected):
            raise ValueError(f"diversity/novelty config {key} differs from the frozen contract")
    expected_candidates = _require_int(
        root["expected_candidates"], label="expected_candidates", minimum=1
    )
    expected_training_rows = _require_int(
        root["expected_training_rows"], label="expected_training_rows", minimum=1
    )
    expected_reference_records = _require_int(
        root["expected_reference_records"], label="expected_reference_records", minimum=1
    )

    mean_raw = _require_exact_keys(
        _toml_table(root["accepted_mean_ledger"], label="accepted_mean_ledger"),
        {
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "config_sha256",
            "publication_top_sha256",
            "publication_tree_sha256",
            "candidate_ledger_sha256",
            "diagnostics_sha256",
            "manifest_sha256",
            "independent_verification_sha256",
            "operational_audit_sha256",
            "model_release_id",
        },
        label="accepted_mean_ledger",
    )
    mean = AcceptedMeanLedgerContract(
        producer_job_id=_require_int(
            mean_raw["producer_job_id"], label="mean producer_job_id", minimum=1
        ),
        audit_job_id=_require_int(mean_raw["audit_job_id"], label="mean audit_job_id", minimum=1),
        git_commit=_require_git_sha(mean_raw["git_commit"], label="mean git_commit"),
        config_sha256=_require_sha(mean_raw["config_sha256"], label="mean config SHA-256"),
        publication_top_sha256=_require_sha(
            mean_raw["publication_top_sha256"], label="mean publication top SHA-256"
        ),
        publication_tree_sha256=_require_sha(
            mean_raw["publication_tree_sha256"], label="mean publication tree SHA-256"
        ),
        candidate_ledger_sha256=_require_sha(
            mean_raw["candidate_ledger_sha256"], label="mean candidate ledger SHA-256"
        ),
        diagnostics_sha256=_require_sha(
            mean_raw["diagnostics_sha256"], label="mean diagnostics SHA-256"
        ),
        manifest_sha256=_require_sha(mean_raw["manifest_sha256"], label="mean manifest SHA-256"),
        independent_verification_sha256=_require_sha(
            mean_raw["independent_verification_sha256"],
            label="mean independent verification SHA-256",
        ),
        operational_audit_sha256=_require_sha(
            mean_raw["operational_audit_sha256"], label="mean operational audit SHA-256"
        ),
        model_release_id=_require_sha(mean_raw["model_release_id"], label="mean model release ID"),
    )

    training_raw = _require_exact_keys(
        _toml_table(root["accepted_training_projection"], label="accepted_training_projection"),
        {
            "stage_job_id",
            "git_commit",
            "projection_sha256",
            "projection_size_bytes",
            "receipt_sha256",
            "receipt_size_bytes",
            "fields",
            "weighting",
        },
        label="accepted_training_projection",
    )
    if training_raw["fields"] != ["sequence_id", "sequence", "sampling_weight"]:
        raise ValueError("training projection fields differ from the frozen projection")
    if training_raw["weighting"] != (
        "component_equal_sampling_weight_from_accepted_train_projection"
    ):
        raise ValueError("training projection weighting differs")
    training = AcceptedTrainingProjectionContract(
        stage_job_id=_require_int(
            training_raw["stage_job_id"], label="training stage_job_id", minimum=1
        ),
        git_commit=_require_git_sha(training_raw["git_commit"], label="training git_commit"),
        projection_sha256=_require_sha(
            training_raw["projection_sha256"], label="training projection SHA-256"
        ),
        projection_size_bytes=_require_int(
            training_raw["projection_size_bytes"],
            label="training projection size",
            minimum=1,
        ),
        receipt_sha256=_require_sha(
            training_raw["receipt_sha256"], label="training receipt SHA-256"
        ),
        receipt_size_bytes=_require_int(
            training_raw["receipt_size_bytes"], label="training receipt size", minimum=1
        ),
    )

    reference_raw = _require_exact_keys(
        _toml_table(root["organizer_reference"], label="organizer_reference"),
        {"sha256", "size_bytes", "source_commit", "role"},
        label="organizer_reference",
    )
    reference = OrganizerReferenceContract(
        sha256=_require_sha(reference_raw["sha256"], label="organizer reference SHA-256"),
        size_bytes=_require_int(
            reference_raw["size_bytes"], label="organizer reference size", minimum=1
        ),
        source_commit=_require_git_sha(
            reference_raw["source_commit"], label="organizer reference source commit"
        ),
        role=_require_string(reference_raw["role"], label="organizer reference role"),
    )
    if reference.role != "compliance_reference_only":
        raise ValueError("organizer reference role must remain compliance-only")

    feature_raw = _require_exact_keys(
        _toml_table(root["features"], label="features"),
        {
            "feature_space",
            "descriptor_names",
            "residue_order",
            "feature_count",
            "descriptor_ph",
            "hydrophobic_moment_angle_degrees",
            "isoelectric_point_lower_ph",
            "isoelectric_point_upper_ph",
            "isoelectric_point_iterations",
            "free_termini",
            "fit_rows",
            "training_weighting",
            "weighted_mean_algorithm",
            "weighted_scale_algorithm",
            "standardization",
            "row_normalization",
            "output_dtype",
            "byte_order",
            "memory_order",
            "csv_float_format",
        },
        label="features",
    )
    frozen_features: Mapping[str, object] = {
        "feature_space": FEATURE_SPACE,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "residue_order": ALPHABET,
        "feature_count": len(FEATURE_NAMES),
        "descriptor_ph": 7.4,
        "hydrophobic_moment_angle_degrees": 100.0,
        "isoelectric_point_lower_ph": 0.0,
        "isoelectric_point_upper_ph": 14.0,
        "isoelectric_point_iterations": 60,
        "free_termini": True,
        "fit_rows": "accepted_training_projection_only",
        "training_weighting": TRAINING_WEIGHTING,
        "weighted_mean_algorithm": MEAN_ALGORITHM,
        "weighted_scale_algorithm": SCALE_ALGORITHM,
        "standardization": STANDARDIZATION,
        "row_normalization": ROW_NORMALIZATION,
        "output_dtype": "float32",
        "byte_order": "little_endian",
        "memory_order": "C",
        "csv_float_format": "shortest_roundtrip_little_endian_float32",
    }
    for key, expected in frozen_features.items():
        if feature_raw[key] != expected or type(feature_raw[key]) is not type(expected):
            raise ValueError(f"features.{key} differs from the frozen contract")

    similarity_raw = _require_exact_keys(
        _toml_table(root["similarity"], label="similarity"),
        {
            "implementation",
            "value_range",
            "training_reduction",
            "nearest_training_tie_break",
            "novelty",
            "reference_rule",
            "reference_threshold_equality",
            "reference_witness_tie_break",
            "reference_maximum_similarity_stored",
            "length_pruning",
            "candidate_chunk_size",
            "choice_chunk_size",
            "worker_default",
            "worker_minimum",
            "worker_maximum",
            "worker_count_affects_output_bytes",
        },
        label="similarity",
    )
    frozen_similarity: Mapping[str, object] = {
        "implementation": SIMILARITY_IMPLEMENTATION,
        "value_range": "zero_to_one",
        "training_reduction": "exact_maximum_over_all_accepted_training_sequences",
        "nearest_training_tie_break": "lexicographically_smallest_sequence_id",
        "novelty": "one_minus_max_train_indel_similarity",
        "reference_rule": (
            "unsafe_if_any_unique_organizer_reference_similarity_strictly_exceeds_threshold"
        ),
        "reference_threshold_equality": "safe",
        "reference_witness_tie_break": "lexicographically_smallest_sequence_id",
        "reference_maximum_similarity_stored": False,
        "length_pruning": LENGTH_BOUND,
        "worker_count_affects_output_bytes": False,
    }
    for key, expected in frozen_similarity.items():
        if similarity_raw[key] != expected or type(similarity_raw[key]) is not type(expected):
            raise ValueError(f"similarity.{key} differs from the frozen contract")
    policy = FeaturePolicy(
        candidate_chunk_size=_require_int(
            similarity_raw["candidate_chunk_size"], label="candidate_chunk_size", minimum=1
        ),
        choice_chunk_size=_require_int(
            similarity_raw["choice_chunk_size"], label="choice_chunk_size", minimum=1
        ),
        worker_default=_require_int(
            similarity_raw["worker_default"], label="worker_default", minimum=1
        ),
        worker_minimum=_require_int(
            similarity_raw["worker_minimum"], label="worker_minimum", minimum=1
        ),
        worker_maximum=_require_int(
            similarity_raw["worker_maximum"], label="worker_maximum", minimum=1
        ),
    )
    if not policy.worker_minimum <= policy.worker_default <= policy.worker_maximum:
        raise ValueError("similarity worker bounds do not contain worker_default")

    claims = _require_exact_keys(
        _toml_table(root["claims"], label="claims"), set(FORBIDDEN_CLAIMS), label="claims"
    )
    if any(value is not False for value in claims.values()):
        raise ValueError("every prohibited promotion claim must remain false")
    return CandidateDiversityNoveltyConfig(
        path=snapshot.path,
        snapshot=snapshot,
        expected_candidates=expected_candidates,
        expected_training_rows=expected_training_rows,
        expected_reference_records=expected_reference_records,
        mean_ledger=mean,
        training_projection=training,
        organizer_reference=reference,
        policy=policy,
    )


def _bundle_tree_sha256(snapshots: Mapping[str, InputSnapshot]) -> str:
    transcript = b"".join(
        f"{snapshots[name].mode:o} {snapshots[name].sha256} {name}\n".encode("ascii")
        for name in sorted(snapshots)
    )
    return hashlib.sha256(transcript).hexdigest()


def _validate_sha256sums(
    snapshot: InputSnapshot,
    *,
    snapshots: Mapping[str, InputSnapshot],
    names: Sequence[str],
    expected_top: str,
    label: str,
) -> None:
    if snapshot.sha256 != expected_top:
        raise ValueError(f"{label} SHA256SUMS hash differs from its accepted pin")
    expected = b"".join(
        f"{snapshots[name].sha256}  {name}\n".encode("ascii")
        for name in sorted(set(names) - {"SHA256SUMS"})
    )
    if snapshot.payload != expected:
        raise ValueError(f"{label} SHA256SUMS bytes or inventory are not exact")


def _validate_artifact_record(
    value: object,
    *,
    snapshot: InputSnapshot,
    label: str,
    rows: int | None = None,
) -> None:
    keys = {"sha256", "size_bytes"} | ({"rows"} if rows is not None else set())
    record = _require_exact_keys(value, keys, label=label)
    if record["sha256"] != snapshot.sha256 or record["size_bytes"] != len(snapshot.payload):
        raise ValueError(f"{label} does not bind the authenticated bytes")
    if rows is not None and record["rows"] != rows:
        raise ValueError(f"{label} row count differs")


def _validate_mean_manifest(
    manifest: Mapping[str, Any],
    *,
    snapshots: Mapping[str, InputSnapshot],
    config: CandidateDiversityNoveltyConfig,
) -> None:
    root = _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "allowed_consumer",
            "adapter_config_sha256",
            "model_release_id",
            "scopes",
            "objectives",
            "targets",
            "source",
            "candidate_ledger",
            "diagnostics",
            "claims",
            "exclusions",
        },
        label="mean-ledger manifest",
    )
    contract = config.mean_ledger
    literals = {
        "schema_version": 1,
        "artifact": MEAN_ARTIFACT,
        "status": MEAN_STATUS,
        "automatic_production_eligible": False,
        "allowed_consumer": MEAN_ALLOWED_CONSUMER,
        "adapter_config_sha256": contract.config_sha256,
        "model_release_id": contract.model_release_id,
        "objectives": list(OBJECTIVES),
    }
    for key, expected in literals.items():
        if root[key] != expected or type(root[key]) is not type(expected):
            raise ValueError(f"mean-ledger manifest {key} differs from the accepted contract")
    scopes = _require_exact_keys(
        root["scopes"],
        {"calibration", "eligibility", "prediction", "training", "uncertainty_status"},
        label="mean-ledger scopes",
    )
    if scopes != {
        "calibration": CALIBRATION_SCOPE,
        "eligibility": ELIGIBILITY_SCOPE,
        "prediction": PREDICTION_SCOPE,
        "training": TRAINING_SCOPE,
        "uncertainty_status": UNCERTAINTY_STATUS,
    }:
        raise ValueError("mean-ledger scopes differ from the accepted contract")
    candidate = _require_exact_keys(
        root["candidate_ledger"],
        {"columns", "eligibility_mapping", "filename", "rows", "sha256", "size_bytes"},
        label="mean-ledger candidate ledger",
    )
    if candidate["columns"] != list(MEAN_LEDGER_COLUMNS) or candidate["filename"] != (
        "candidate_ledger.csv"
    ):
        raise ValueError("mean-ledger candidate schema differs")
    if candidate["eligibility_mapping"] != "eligible_is_exact_library_eligible_value":
        raise ValueError("mean-ledger eligibility mapping differs")
    _validate_artifact_record(
        {key: candidate[key] for key in ("rows", "sha256", "size_bytes")},
        snapshot=snapshots["candidate_ledger.csv"],
        label="mean-ledger candidate artifact",
        rows=config.expected_candidates,
    )
    diagnostic = _require_exact_keys(
        root["diagnostics"],
        {
            "columns",
            "filename",
            "scope",
            "selectable",
            "sha256",
            "size_bytes",
            "rows",
            "uncertainty",
        },
        label="mean-ledger diagnostics",
    )
    expected_diagnostic_columns = [
        "source_ordinal",
        "sequence_id",
        *(f"model_fit_sensitivity_{name}" for name in OBJECTIVES),
        "diagnostic_scope",
    ]
    if (
        diagnostic["columns"] != expected_diagnostic_columns
        or diagnostic["filename"] != "model_fit_sensitivity_diagnostics.csv"
        or diagnostic["selectable"] is not False
        or diagnostic["uncertainty"] is not False
    ):
        raise ValueError("mean-ledger diagnostic separation differs")
    _validate_artifact_record(
        {key: diagnostic[key] for key in ("rows", "sha256", "size_bytes")},
        snapshot=snapshots["model_fit_sensitivity_diagnostics.csv"],
        label="mean-ledger diagnostic artifact",
        rows=config.expected_candidates,
    )
    claims = _require_exact_keys(
        root["claims"],
        {
            "diagnostics_are_uncertainty",
            "diagnostics_selectable",
            "final_ranking",
            "mean_only_development_handoff",
            "production_ensemble",
            "uncertainty_available",
        },
        label="mean-ledger claims",
    )
    if claims != {
        "diagnostics_are_uncertainty": False,
        "diagnostics_selectable": False,
        "final_ranking": False,
        "mean_only_development_handoff": True,
        "production_ensemble": False,
        "uncertainty_available": False,
    }:
        raise ValueError("mean-ledger claims differ")
    targets = root["targets"]
    if not isinstance(targets, list) or len(targets) != len(TARGET_PANEL):
        raise ValueError("mean-ledger target census differs")
    for index, (target, name) in enumerate(zip(targets, TARGET_PANEL, strict=True)):
        record = _require_exact_keys(
            target,
            {"gram", "index", "name", "probability_column"},
            label=f"mean-ledger target {index}",
        )
        if (
            record["index"] != index
            or record["name"] != name
            or record["probability_column"] != f"probability_{name}"
            or record["gram"] not in {"positive", "negative"}
        ):
            raise ValueError(f"mean-ledger target {index} differs")


def _canonical_float_text(value: str, *, label: str, upper_bound: float = 1.0) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{label} is not a canonical float")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= upper_bound:
        raise ValueError(f"{label} lies outside [0, {upper_bound}]")
    if value != f"{parsed:.17g}":
        raise ValueError(f"{label} does not use canonical float64 spelling")
    return value


def _canonical_lineage(value: str, *, label: str) -> str:
    parts = value.split("|")
    if not parts or any(_LINEAGE_TOKEN_RE.fullmatch(part) is None for part in parts):
        raise ValueError(f"{label} is malformed")
    if parts != sorted(set(parts)):
        raise ValueError(f"{label} must be sorted and unique")
    return value


def _csv_bytes(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise AssertionError("generated CSV is not canonical UTF-8/LF")
    return payload


def _read_mean_rows(
    snapshot: InputSnapshot,
    *,
    config: CandidateDiversityNoveltyConfig,
) -> tuple[MeanLedgerRow, ...]:
    if b"\r" in snapshot.payload or b"\x00" in snapshot.payload:
        raise ValueError("mean-ledger candidate CSV is not canonical UTF-8/LF")
    if not snapshot.payload.endswith(b"\n"):
        raise ValueError("mean-ledger candidate CSV must end with LF")
    try:
        reader = csv.reader(io.StringIO(snapshot.payload.decode("utf-8"), newline=""), strict=True)
        records = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("mean-ledger candidate CSV is malformed") from error
    if not records or tuple(records[0]) != MEAN_LEDGER_COLUMNS:
        raise ValueError("mean-ledger candidate CSV columns or order differ")
    if any(not record for record in records[1:]):
        raise ValueError("mean-ledger candidate CSV cannot contain blank rows")
    if _csv_bytes(records[0], records[1:]) != snapshot.payload:
        raise ValueError("mean-ledger candidate CSV framing is not canonical")
    if len(records) - 1 != config.expected_candidates:
        raise ValueError("mean-ledger candidate row count differs")

    rows: list[MeanLedgerRow] = []
    sequence_ids: set[str] = set()
    sequences: set[str] = set()
    probability_start = 8
    mean_start = probability_start + len(TARGET_PANEL)
    for expected_ordinal, values in enumerate(records[1:], start=1):
        row_number = expected_ordinal + 1
        if len(values) != len(MEAN_LEDGER_COLUMNS):
            raise ValueError(f"mean-ledger row {row_number} width differs")
        if values[0] != str(expected_ordinal):
            raise ValueError(f"mean-ledger row {row_number} ordinal differs")
        sequence = canonicalize_sequence(values[2])
        if sequence != values[2] or canonical_sequence_id(sequence) != values[1]:
            raise ValueError(f"mean-ledger row {row_number} sequence identity differs")
        if values[1] in sequence_ids or sequence in sequences:
            raise ValueError(f"mean-ledger row {row_number} duplicates a candidate")
        sequence_ids.add(values[1])
        sequences.add(sequence)
        if values[3] != str(len(sequence)):
            raise ValueError(f"mean-ledger row {row_number} length differs")
        if values[4] not in {"true", "false"}:
            raise ValueError(f"mean-ledger row {row_number} eligibility is not canonical")
        if values[5] != ELIGIBILITY_SCOPE:
            raise ValueError(f"mean-ledger row {row_number} eligibility scope differs")
        _canonical_lineage(values[6], label=f"mean-ledger row {row_number} families")
        _canonical_lineage(values[7], label=f"mean-ledger row {row_number} variants")
        for index in range(probability_start, mean_start + len(OBJECTIVES)):
            _canonical_float_text(values[index], label=f"mean-ledger row {row_number} score")
        if values[mean_start + len(OBJECTIVES) :] != [
            UNCERTAINTY_STATUS,
            PREDICTION_SCOPE,
            TRAINING_SCOPE,
            CALIBRATION_SCOPE,
            config.mean_ledger.model_release_id,
        ]:
            raise ValueError(f"mean-ledger row {row_number} scope or release differs")
        rows.append(MeanLedgerRow(tuple(values)))
    if any(name.startswith(FORBIDDEN_LEDGER_PREFIXES) for name in MEAN_LEDGER_COLUMNS):
        raise AssertionError("accepted mean-ledger schema unexpectedly exposes a prohibited field")
    return tuple(rows)


def _validate_mean_independent_receipt(
    receipt: Mapping[str, Any],
    *,
    mean_snapshots: Mapping[str, InputSnapshot],
    config: CandidateDiversityNoveltyConfig,
) -> None:
    root = _require_exact_keys(
        receipt,
        {
            "artifact",
            "automatic_production_eligible",
            "census",
            "checks",
            "claims",
            "input_sha256",
            "model_release_id",
            "output_sha256",
            "schema_version",
            "semantic_sha256",
            "status",
        },
        label="mean-ledger independent receipt",
    )
    contract = config.mean_ledger
    if (
        root["schema_version"] != 1
        or root["artifact"] != MEAN_INDEPENDENT_ARTIFACT
        or root["status"] != "passed"
        or root["automatic_production_eligible"] is not False
        or root["model_release_id"] != contract.model_release_id
    ):
        raise ValueError("mean-ledger independent receipt identity differs")
    census = _require_exact_keys(
        root["census"],
        {"candidates", "eligible_candidates", "objectives", "targets"},
        label="mean-ledger independent census",
    )
    if (
        census["candidates"] != config.expected_candidates
        or census["eligible_candidates"] != config.expected_candidates
        or census["objectives"] != len(OBJECTIVES)
        or census["targets"] != len(TARGET_PANEL)
    ):
        raise ValueError("mean-ledger independent receipt census differs")
    checks = _require_exact_keys(
        root["checks"],
        {
            "accepted_scoring_audit_authenticated",
            "accepted_scoring_hash_pins_exact",
            "accepted_scoring_twins_byte_identical",
            "candidate_identity_order_and_lineage_exact",
            "candidate_ledger_independently_reconstructed",
            "diagnostics_independently_reconstructed_and_separated",
            "ledger_checksum_inventory_exact",
            "ledger_contains_no_uncertainty_proxy_or_sensitivity",
            "manifest_declares_mean_only_development_scope",
            "model_release_and_scopes_preserved",
        },
        label="mean-ledger independent checks",
    )
    if any(value is not True for value in checks.values()):
        raise ValueError("not every mean-ledger independent check passed")
    claims = _require_exact_keys(
        root["claims"],
        {
            "allowed_consumer",
            "calibrated_uncertainty_available",
            "development_mean_only_ledger",
            "final_ranking",
            "model_fit_sensitivity_is_selectable_uncertainty",
            "production_ensemble",
        },
        label="mean-ledger independent claims",
    )
    if claims != {
        "allowed_consumer": MEAN_ALLOWED_CONSUMER,
        "calibrated_uncertainty_available": False,
        "development_mean_only_ledger": True,
        "final_ranking": False,
        "model_fit_sensitivity_is_selectable_uncertainty": False,
        "production_ensemble": False,
    }:
        raise ValueError("mean-ledger independent claims differ")
    output = _require_exact_keys(
        root["output_sha256"], set(MEAN_FILES), label="mean-ledger independent outputs"
    )
    if output != {name: mean_snapshots[name].sha256 for name in MEAN_FILES}:
        raise ValueError("mean-ledger independent receipt does not bind the accepted twin")
    inputs = _require_exact_keys(
        root["input_sha256"],
        {"config", "scoring_audit", "scoring_twins"},
        label="mean-ledger independent inputs",
    )
    if inputs["config"] != contract.config_sha256:
        raise ValueError("mean-ledger independent receipt config differs")
    for key in ("scoring_audit", "scoring_twins"):
        nested = inputs[key]
        if not isinstance(nested, dict) or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in nested.values()
        ):
            raise ValueError(f"mean-ledger independent {key} hashes are invalid")
    semantic = _require_exact_keys(
        root["semantic_sha256"],
        {"candidate_ledger_rows", "model_fit_sensitivity_diagnostic_rows"},
        label="mean-ledger semantic hashes",
    )
    for key, value in semantic.items():
        _require_sha(value, label=f"mean-ledger semantic hash {key}")


def _validate_mean_operational_receipt(
    receipt: Mapping[str, Any],
    *,
    mean_snapshots: Mapping[str, InputSnapshot],
    config: CandidateDiversityNoveltyConfig,
) -> None:
    root = _require_exact_keys(
        receipt,
        {
            "accepted_scoring",
            "artifact",
            "audit",
            "automatic_production_eligible",
            "checks",
            "config_sha256",
            "git_commit",
            "independent_receipt_sha256",
            "launchers_sha256",
            "output",
            "producer",
            "schema_version",
            "status",
            "uncertainty_status",
        },
        label="mean-ledger operational receipt",
    )
    contract = config.mean_ledger
    if (
        root["schema_version"] != 1
        or root["artifact"] != MEAN_OPERATIONAL_ARTIFACT
        or root["status"] != "passed"
        or root["automatic_production_eligible"] is not False
        or root["uncertainty_status"] != UNCERTAINTY_STATUS
        or root["git_commit"] != contract.git_commit
        or root["config_sha256"] != contract.config_sha256
    ):
        raise ValueError("mean-ledger operational receipt identity differs")
    checks = _require_exact_keys(
        root["checks"],
        {
            "accepted_source_evidence_unchanged",
            "audit_node_excludes_both_producer_nodes",
            "each_twin_independently_reconstructed_before_comparison",
            "producer_completed_in_slurm_accounting",
            "repository_clean_and_live_origin_synchronized",
            "twins_byte_identical",
            "uncertainty_remains_unavailable",
        },
        label="mean-ledger operational checks",
    )
    if any(value is not True for value in checks.values()):
        raise ValueError("not every mean-ledger operational audit check passed")
    independent = _require_exact_keys(
        root["independent_receipt_sha256"], {"0", "1"}, label="mean independent hashes"
    )
    if independent != {
        "0": contract.independent_verification_sha256,
        "1": contract.independent_verification_sha256,
    }:
        raise ValueError("mean-ledger operational independent hashes differ")
    audit = _require_exact_keys(
        root["audit"], {"job_id", "node_name", "resources"}, label="mean audit execution"
    )
    if audit["job_id"] != contract.audit_job_id:
        raise ValueError("mean-ledger audit job differs")
    _require_string(audit["node_name"], label="mean audit node")
    resources = _require_exact_keys(
        audit["resources"],
        {"account", "cpus_per_task", "gpus", "memory_per_node_mib", "nodes", "partition", "tasks"},
        label="mean audit resources",
    )
    if resources["gpus"] != 0 or resources["nodes"] != 1 or resources["tasks"] != 1:
        raise ValueError("mean-ledger audit resource semantics differ")
    output = _require_exact_keys(root["output"], {"0", "1"}, label="mean twin outputs")
    expected_files = {name: mean_snapshots[name].sha256 for name in MEAN_FILES}
    for twin_name in ("0", "1"):
        twin = _require_exact_keys(
            output[twin_name],
            {"files_sha256", "top_sha256", "tree_sha256"},
            label=f"mean twin {twin_name}",
        )
        files = _require_exact_keys(
            twin["files_sha256"], set(MEAN_FILES), label=f"mean twin {twin_name} files"
        )
        if (
            files != expected_files
            or twin["top_sha256"] != contract.publication_top_sha256
            or twin["tree_sha256"] != contract.publication_tree_sha256
        ):
            raise ValueError(f"mean-ledger operational twin {twin_name} hashes differ")
    producer = _require_exact_keys(
        root["producer"],
        {"job_id", "node_receipt_sha256", "nodes", "observed_telemetry", "resources"},
        label="mean producer execution",
    )
    if producer["job_id"] != contract.producer_job_id:
        raise ValueError("mean-ledger producer job differs")
    nodes = producer["nodes"]
    if not isinstance(nodes, list) or len(nodes) != 2 or len(set(nodes)) != 2:
        raise ValueError("mean-ledger producer did not use two distinct nodes")
    for nested_name in ("node_receipt_sha256", "observed_telemetry", "resources"):
        if not isinstance(producer[nested_name], dict):
            raise ValueError(f"mean-ledger producer {nested_name} must be a table")
    if producer["resources"].get("gpus") != 0 or producer["resources"].get("nodes") != 2:
        raise ValueError("mean-ledger producer resource semantics differ")
    accepted_scoring = root["accepted_scoring"]
    if not isinstance(accepted_scoring, dict) or set(accepted_scoring) != {
        "audit_job_id",
        "git_commit",
        "producer_job_id",
        "sha256",
    }:
        raise ValueError("mean-ledger accepted-scoring audit binding differs")
    if not isinstance(accepted_scoring["sha256"], dict):
        raise ValueError("mean-ledger accepted-scoring hashes must be a table")
    launchers = root["launchers_sha256"]
    if not isinstance(launchers, dict) or any(
        _SHA256_RE.fullmatch(value) is None
        for value in launchers.values()
        if isinstance(value, str)
    ):
        raise ValueError("mean-ledger launcher hashes are malformed")


def _parse_jsonl_document(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict) or _canonical_json_bytes(document) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return document


def _read_training_rows(
    snapshot: InputSnapshot,
    *,
    config: CandidateDiversityNoveltyConfig,
) -> tuple[TrainingRow, ...]:
    contract = config.training_projection
    if snapshot.sha256 != contract.projection_sha256 or len(snapshot.payload) != (
        contract.projection_size_bytes
    ):
        raise ValueError("training projection bytes differ from the accepted pin")
    if b"\r" in snapshot.payload or b"\x00" in snapshot.payload:
        raise ValueError("training projection must use canonical UTF-8/LF")
    lines = snapshot.payload.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("training projection must contain LF-terminated JSON rows")
    if len(lines) != config.expected_training_rows:
        raise ValueError("training projection row count differs")
    rows: list[TrainingRow] = []
    sequence_ids: set[str] = set()
    sequences: set[str] = set()
    for number, line in enumerate(lines, start=1):
        document = _parse_jsonl_document(line, label=f"training projection row {number}")
        row = _require_exact_keys(
            document,
            {"sequence_id", "sequence", "sampling_weight"},
            label=f"training projection row {number}",
        )
        sequence_id = _require_sha(row["sequence_id"], label=f"training row {number} ID")
        if not isinstance(row["sequence"], str):
            raise ValueError(f"training row {number} sequence must be a string")
        sequence = canonicalize_sequence(row["sequence"])
        if sequence != row["sequence"] or canonical_sequence_id(sequence) != sequence_id:
            raise ValueError(f"training row {number} sequence identity differs")
        if sequence_id in sequence_ids or sequence in sequences:
            raise ValueError(f"training row {number} duplicates a sequence")
        if rows and sequence_id <= rows[-1].sequence_id:
            raise ValueError("training projection must be strictly ordered by sequence_id")
        weight = row["sampling_weight"]
        if type(weight) is not float or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"training row {number} weight must be a positive JSON float")
        sequence_ids.add(sequence_id)
        sequences.add(sequence)
        rows.append(TrainingRow(sequence_id, sequence, weight))
    if not math.isclose(
        math.fsum(row.sampling_weight for row in rows),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("training projection sampling weights do not sum to one")
    return tuple(rows)


def _validate_projection_receipt(
    receipt: Mapping[str, Any],
    *,
    projection_snapshot: InputSnapshot,
    config: CandidateDiversityNoveltyConfig,
) -> None:
    root = _require_exact_keys(
        receipt,
        {
            "artifact",
            "checks",
            "config_sha256",
            "git_commit",
            "input_sha256",
            "projection",
            "schema_version",
            "stage_job_id",
            "stage_node",
            "status",
        },
        label="training projection receipt",
    )
    contract = config.training_projection
    if (
        root["schema_version"] != 1
        or root["artifact"] != PROJECTION_RECEIPT_ARTIFACT
        or root["status"] != "passed"
        or root["git_commit"] != contract.git_commit
        or root["stage_job_id"] != str(contract.stage_job_id)
    ):
        raise ValueError("training projection receipt identity differs")
    _require_string(root["stage_node"], label="training projection stage node")
    _require_sha(root["config_sha256"], label="training projection stage config")
    checks = _require_exact_keys(
        root["checks"],
        {
            "accepted_corpus_verified",
            "fold4_excluded",
            "no_overwrite_publication",
            "projection_schema_exact",
            "repository_snapshot_exact",
        },
        label="training projection receipt checks",
    )
    if any(value is not True for value in checks.values()):
        raise ValueError("not every training projection receipt check passed")
    projection = _require_exact_keys(
        root["projection"], {"fields", "filename", "rows", "sha256"}, label="projection"
    )
    if projection != {
        "fields": ["sequence_id", "sequence", "sampling_weight"],
        "filename": "training_projection.jsonl",
        "rows": config.expected_training_rows,
        "sha256": projection_snapshot.sha256,
    }:
        raise ValueError("training projection receipt does not bind the accepted projection")
    input_hashes = root["input_sha256"]
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("training projection receipt input hashes are missing")
    for key, value in input_hashes.items():
        _require_sha(value, label=f"training projection receipt input {key}")


def _read_reference_rows(
    snapshot: InputSnapshot,
    *,
    config: CandidateDiversityNoveltyConfig,
) -> tuple[tuple[ReferenceRow, ...], int]:
    contract = config.organizer_reference
    if snapshot.sha256 != contract.sha256 or len(snapshot.payload) != contract.size_bytes:
        raise ValueError("organizer reference bytes differ from the tracked pin")
    if b"\r" in snapshot.payload or b"\x00" in snapshot.payload:
        raise ValueError("organizer reference FASTA must use canonical UTF-8/LF")
    if not snapshot.payload.endswith(b"\n"):
        raise ValueError("organizer reference FASTA must end with LF")
    try:
        lines = snapshot.payload.decode("ascii").splitlines()
    except UnicodeError as error:
        raise ValueError("organizer reference FASTA must be ASCII") from error
    records: list[str] = []
    current_header: str | None = None
    current_sequence: list[str] = []
    seen_headers: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line or line.strip() != line:
            raise ValueError(f"organizer reference FASTA line {line_number} is not canonical")
        if line.startswith(">"):
            if current_header is not None:
                if not current_sequence:
                    raise ValueError("organizer reference FASTA contains an empty record")
                records.append("".join(current_sequence))
            current_header = line[1:]
            if not current_header or current_header in seen_headers:
                raise ValueError("organizer reference FASTA header is empty or duplicated")
            seen_headers.add(current_header)
            current_sequence = []
        else:
            if current_header is None:
                raise ValueError("organizer reference FASTA begins without a header")
            current_sequence.append(line)
    if current_header is None or not current_sequence:
        raise ValueError("organizer reference FASTA has no complete records")
    records.append("".join(current_sequence))
    if len(records) != config.expected_reference_records:
        raise ValueError("organizer reference FASTA record count differs")
    by_id: dict[str, str] = {}
    for number, raw_sequence in enumerate(records, start=1):
        sequence = canonicalize_sequence(raw_sequence)
        if sequence != raw_sequence:
            raise ValueError(f"organizer reference record {number} is not canonical")
        sequence_id = canonical_sequence_id(sequence)
        previous = by_id.setdefault(sequence_id, sequence)
        if previous != sequence:
            raise ValueError("organizer reference contains a sequence-ID collision")
    unique = tuple(ReferenceRow(key, by_id[key]) for key in sorted(by_id))
    return unique, len(records)


def authenticate_candidate_diversity_novelty_inputs(
    *,
    mean_ledger_twin: str | Path,
    mean_ledger_audit_dir: str | Path,
    training_projection: str | Path,
    projection_receipt: str | Path,
    organizer_reference: str | Path,
    config: CandidateDiversityNoveltyConfig,
) -> AuthenticatedDiversityInputs:
    """Authenticate every accepted input and return only reconstructed semantics."""

    mean_directory, mean = _snapshot_flat_bundle(
        mean_ledger_twin, names=MEAN_FILES, label="accepted mean-ledger twin"
    )
    mean_contract = config.mean_ledger
    expected_mean_hashes = {
        "SHA256SUMS": mean_contract.publication_top_sha256,
        "candidate_ledger.csv": mean_contract.candidate_ledger_sha256,
        "manifest.json": mean_contract.manifest_sha256,
        "model_fit_sensitivity_diagnostics.csv": mean_contract.diagnostics_sha256,
    }
    for name, expected in expected_mean_hashes.items():
        if mean[name].sha256 != expected:
            raise ValueError(f"accepted mean-ledger hash differs: {name}")
    _validate_sha256sums(
        mean["SHA256SUMS"],
        snapshots=mean,
        names=MEAN_FILES,
        expected_top=mean_contract.publication_top_sha256,
        label="accepted mean-ledger",
    )
    if _bundle_tree_sha256(mean) != mean_contract.publication_tree_sha256:
        raise ValueError("accepted mean-ledger publication tree differs")
    mean_manifest = _load_canonical_json(mean["manifest.json"], label="mean-ledger manifest")
    _validate_mean_manifest(mean_manifest, snapshots=mean, config=config)
    mean_rows = _read_mean_rows(mean["candidate_ledger.csv"], config=config)

    audit_directory, audit = _snapshot_flat_bundle(
        mean_ledger_audit_dir, names=MEAN_AUDIT_FILES, label="accepted mean-ledger audit"
    )
    if audit["operational-receipt.json"].sha256 != mean_contract.operational_audit_sha256:
        raise ValueError("accepted mean-ledger operational receipt hash differs")
    for name in ("twin-0-independent-verification.json", "twin-1-independent-verification.json"):
        if audit[name].sha256 != mean_contract.independent_verification_sha256:
            raise ValueError(f"accepted mean-ledger independent receipt hash differs: {name}")
    independent_zero = _load_canonical_json(
        audit["twin-0-independent-verification.json"], label="mean-ledger twin-0 receipt"
    )
    independent_one = _load_canonical_json(
        audit["twin-1-independent-verification.json"], label="mean-ledger twin-1 receipt"
    )
    if independent_zero != independent_one:
        raise ValueError("accepted mean-ledger independent receipts differ")
    _validate_mean_independent_receipt(independent_zero, mean_snapshots=mean, config=config)
    operational = _load_canonical_json(
        audit["operational-receipt.json"], label="mean-ledger operational receipt"
    )
    _validate_mean_operational_receipt(operational, mean_snapshots=mean, config=config)

    projection_path = Path(os.path.abspath(os.fspath(training_projection)))
    receipt_path = Path(os.path.abspath(os.fspath(projection_receipt)))
    if projection_path.parent != receipt_path.parent:
        raise ValueError("training projection and receipt must share one sealed stage directory")
    training_directory, training = _snapshot_flat_bundle(
        projection_path.parent,
        names=TRAINING_STAGE_FILES,
        label="accepted training-projection stage",
    )
    if (
        training["training_projection.jsonl"].path != projection_path
        or training["projection-receipt.json"].path != receipt_path
    ):
        raise ValueError("training stage paths do not select the exact accepted filenames")
    projection_snapshot = training["training_projection.jsonl"]
    receipt_snapshot = training["projection-receipt.json"]
    training_contract = config.training_projection
    if (
        projection_snapshot.sha256 != training_contract.projection_sha256
        or len(projection_snapshot.payload) != training_contract.projection_size_bytes
        or receipt_snapshot.sha256 != training_contract.receipt_sha256
        or len(receipt_snapshot.payload) != training_contract.receipt_size_bytes
    ):
        raise ValueError("accepted training stage hash or size differs")
    training_rows = _read_training_rows(projection_snapshot, config=config)
    projection_document = _load_canonical_json(
        receipt_snapshot, label="training projection receipt"
    )
    _validate_projection_receipt(
        projection_document, projection_snapshot=projection_snapshot, config=config
    )

    reference_snapshot = _snapshot_file(
        organizer_reference,
        label="tracked organizer reference",
        required_mode=0o644,
    )
    reference_rows, reference_record_count = _read_reference_rows(reference_snapshot, config=config)
    candidate_sequences = {row.sequence for row in mean_rows}
    training_overlap = candidate_sequences.intersection(row.sequence for row in training_rows)
    if training_overlap:
        raise ValueError("candidate library contains an exact accepted-training overlap")
    reference_overlap = candidate_sequences.intersection(row.sequence for row in reference_rows)
    if reference_overlap:
        raise ValueError("candidate library contains an exact organizer-reference overlap")
    return AuthenticatedDiversityInputs(
        mean_rows=mean_rows,
        training_rows=training_rows,
        reference_rows=reference_rows,
        reference_record_count=reference_record_count,
        mean_manifest=mean_manifest,
        snapshots={
            **{f"mean/{name}": snapshot for name, snapshot in mean.items()},
            **{f"mean_audit/{name}": snapshot for name, snapshot in audit.items()},
            **{f"training/{name}": snapshot for name, snapshot in training.items()},
            "organizer_reference": reference_snapshot,
        },
        directory_snapshots={
            "mean-ledger directory": mean_directory,
            "mean-ledger audit directory": audit_directory,
            "training stage directory": training_directory,
        },
    )


def indel_length_upper_bound(left_length: int, right_length: int) -> float:
    """Return the exact length-only upper bound for normalized Indel similarity."""

    if type(left_length) is not int or type(right_length) is not int:
        raise TypeError("sequence lengths must be integers")
    if left_length <= 0 or right_length <= 0:
        raise ValueError("sequence lengths must be positive")
    return (2.0 * min(left_length, right_length)) / (left_length + right_length)


def _can_strictly_exceed_reference_threshold(left_length: int, right_length: int) -> bool:
    # REFERENCE_THRESHOLD is frozen at the exact decimal 4/5.  This integer
    # comparison avoids a floating-point boundary decision: 2m/(a+b) > 4/5.
    return 10 * min(left_length, right_length) > 4 * (left_length + right_length)


def _validate_workers(workers: int, *, config: CandidateDiversityNoveltyConfig) -> int:
    if type(workers) is not int or not (
        config.policy.worker_minimum <= workers <= config.policy.worker_maximum
    ):
        raise ValueError(
            "workers must be an integer within the frozen inclusive worker bounds "
            f"[{config.policy.worker_minimum}, {config.policy.worker_maximum}]"
        )
    return workers


def _indel_cdist(
    queries: Sequence[str],
    choices: Sequence[str],
    *,
    workers: int,
    score_cutoff: float | None = None,
) -> NDArray[np.float64]:
    if not queries or not choices:
        raise ValueError("Indel cdist requires non-empty query and choice chunks")
    values = process.cdist(
        queries,
        choices,
        scorer=Indel.normalized_similarity,
        processor=None,
        score_cutoff=score_cutoff,
        dtype=np.float64,
        workers=workers,
    )
    result = np.asarray(values, dtype=np.float64, order="C")
    if result.shape != (len(queries), len(choices)):
        raise ValueError("RapidFuzz returned an unexpected cdist shape")
    if not bool(np.all(np.isfinite(result))) or not bool(np.all((result >= 0.0) & (result <= 1.0))):
        raise ValueError("RapidFuzz returned a non-finite or out-of-range similarity")
    return result


def _exact_training_similarities(
    candidates: Sequence[MeanLedgerRow],
    training_rows: Sequence[TrainingRow],
    *,
    config: CandidateDiversityNoveltyConfig,
    workers: int,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Compute exact train maxima with safe dynamic length-bound pruning."""

    candidate_groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        candidate_groups[len(row.sequence)].append(index)
    training_groups: dict[int, list[TrainingRow]] = defaultdict(list)
    for row in training_rows:
        training_groups[len(row.sequence)].append(row)
    for values in training_groups.values():
        values.sort(key=lambda row: row.sequence_id)

    maxima = [-1.0] * len(candidates)
    nearest = [""] * len(candidates)
    query_chunk = config.policy.candidate_chunk_size
    choice_chunk = config.policy.choice_chunk_size
    for candidate_length in sorted(candidate_groups):
        all_indices = candidate_groups[candidate_length]
        ordered_lengths = sorted(
            training_groups,
            key=lambda length: (
                -indel_length_upper_bound(candidate_length, length),
                length,
            ),
        )
        for query_start in range(0, len(all_indices), query_chunk):
            indices = all_indices[query_start : query_start + query_chunk]
            for training_length in ordered_lengths:
                upper = indel_length_upper_bound(candidate_length, training_length)
                conservative_upper = math.nextafter(upper, math.inf)
                active = [index for index in indices if conservative_upper >= maxima[index]]
                if not active:
                    continue
                choices_for_length = training_groups[training_length]
                for choice_start in range(0, len(choices_for_length), choice_chunk):
                    choice_rows = choices_for_length[choice_start : choice_start + choice_chunk]
                    matrix = _indel_cdist(
                        [candidates[index].sequence for index in active],
                        [row.sequence for row in choice_rows],
                        workers=workers,
                    )
                    for local_index, candidate_index in enumerate(active):
                        local_maximum = float(np.max(matrix[local_index]))
                        tie_positions = np.flatnonzero(matrix[local_index] == local_maximum)
                        if tie_positions.size == 0:  # pragma: no cover - guarded by np.max.
                            raise AssertionError("RapidFuzz maximum has no attaining position")
                        local_id = min(
                            choice_rows[int(position)].sequence_id for position in tie_positions
                        )
                        if local_maximum > maxima[candidate_index] or (
                            local_maximum == maxima[candidate_index]
                            and (
                                not nearest[candidate_index] or local_id < nearest[candidate_index]
                            )
                        ):
                            maxima[candidate_index] = local_maximum
                            nearest[candidate_index] = local_id
    if any(value < 0.0 or value > 1.0 for value in maxima) or any(not value for value in nearest):
        raise AssertionError("training similarity reduction did not cover every candidate")
    training_by_id = {row.sequence_id: row.sequence for row in training_rows}
    for candidate, maximum, nearest_id in zip(candidates, maxima, nearest, strict=True):
        exact_score = float(
            Indel.normalized_similarity(candidate.sequence, training_by_id[nearest_id])
        )
        if exact_score != maximum:
            raise AssertionError("nearest-training direct rescore differs from the batched maximum")
    return tuple(maxima), tuple(nearest)


def _exact_reference_witnesses(
    candidates: Sequence[MeanLedgerRow],
    reference_rows: Sequence[ReferenceRow],
    *,
    config: CandidateDiversityNoveltyConfig,
    workers: int,
) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    """Return the lexicographically first strict-threshold witness, if any.

    References are sorted globally by canonical sequence ID before chunking.
    Once a candidate has a hit, later reference chunks cannot contain a
    lexicographically smaller witness.  Safe candidates still examine every
    length-compatible reference.  No reference maximum is calculated or
    retained.
    """

    candidate_groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        candidate_groups[len(row.sequence)].append(index)
    references = tuple(sorted(reference_rows, key=lambda row: row.sequence_id))
    witness_ids = [""] * len(candidates)
    witness_scores: list[float | None] = [None] * len(candidates)
    query_chunk = config.policy.candidate_chunk_size
    choice_chunk = config.policy.choice_chunk_size
    for candidate_length in sorted(candidate_groups):
        compatible = tuple(
            row
            for row in references
            if _can_strictly_exceed_reference_threshold(candidate_length, len(row.sequence))
        )
        indices_for_length = candidate_groups[candidate_length]
        for query_start in range(0, len(indices_for_length), query_chunk):
            unresolved = indices_for_length[query_start : query_start + query_chunk]
            for choice_start in range(0, len(compatible), choice_chunk):
                if not unresolved:
                    break
                choice_rows = compatible[choice_start : choice_start + choice_chunk]
                matrix = _indel_cdist(
                    [candidates[index].sequence for index in unresolved],
                    [row.sequence for row in choice_rows],
                    workers=workers,
                    score_cutoff=REFERENCE_THRESHOLD,
                )
                next_unresolved: list[int] = []
                for local_index, candidate_index in enumerate(unresolved):
                    hits = np.flatnonzero(matrix[local_index] > REFERENCE_THRESHOLD)
                    if hits.size:
                        first = int(hits[0])
                        witness = choice_rows[first]
                        batched_score = float(matrix[local_index, first])
                        exact_score = float(
                            Indel.normalized_similarity(
                                candidates[candidate_index].sequence,
                                witness.sequence,
                            )
                        )
                        if exact_score != batched_score or not exact_score > REFERENCE_THRESHOLD:
                            raise AssertionError(
                                "reference witness direct rescore differs from the batched result"
                            )
                        witness_ids[candidate_index] = witness.sequence_id
                        witness_scores[candidate_index] = exact_score
                    else:
                        next_unresolved.append(candidate_index)
                unresolved = next_unresolved
    return tuple(witness_ids), tuple(witness_scores)


def compute_candidate_similarities(
    authenticated: AuthenticatedDiversityInputs,
    *,
    config: CandidateDiversityNoveltyConfig,
    workers: int,
) -> tuple[CandidateSimilarity, ...]:
    """Compute exact training novelty and strict reference-rule witnesses."""

    worker_count = _validate_workers(workers, config=config)
    training_maxima, nearest = _exact_training_similarities(
        authenticated.mean_rows,
        authenticated.training_rows,
        config=config,
        workers=worker_count,
    )
    witness_ids, witness_scores = _exact_reference_witnesses(
        authenticated.mean_rows,
        authenticated.reference_rows,
        config=config,
        workers=worker_count,
    )
    result: list[CandidateSimilarity] = []
    for maximum, nearest_id, witness_id, witness_score in zip(
        training_maxima,
        nearest,
        witness_ids,
        witness_scores,
        strict=True,
    ):
        if bool(witness_id) != (witness_score is not None):
            raise AssertionError("reference witness ID and similarity are not paired")
        if witness_score is not None and not witness_score > REFERENCE_THRESHOLD:
            raise AssertionError("reference witness does not strictly violate the threshold")
        result.append(
            CandidateSimilarity(
                max_train_similarity=maximum,
                nearest_train_sequence_id=nearest_id,
                reference_safe=not witness_id,
                reference_witness_sequence_id=witness_id,
                reference_witness_similarity=witness_score,
            )
        )
    return tuple(result)


def _raw_feature_vector(sequence: str) -> tuple[float, ...]:
    descriptors = compute_descriptors(sequence, ph=7.4)
    values = tuple(float(getattr(descriptors, name)) for name in DESCRIPTOR_NAMES) + tuple(
        sequence.count(residue) / len(sequence) for residue in ALPHABET
    )
    if len(values) != len(FEATURE_NAMES) or any(not math.isfinite(value) for value in values):
        raise ValueError("physicochemical feature calculation is incomplete or non-finite")
    return values


def _feature_matrices(
    candidates: Sequence[MeanLedgerRow],
    training_rows: Sequence[TrainingRow],
) -> tuple[
    NDArray[np.float32],
    tuple[float, ...],
    tuple[float, ...],
    str,
    str,
    str,
    float,
]:
    training_raw = tuple(_raw_feature_vector(row.sequence) for row in training_rows)
    weights = tuple(row.sampling_weight for row in training_rows)
    weight_sum = math.fsum(weights)
    means = tuple(
        math.fsum(
            weight * values[index] for weight, values in zip(weights, training_raw, strict=True)
        )
        / weight_sum
        for index in range(len(FEATURE_NAMES))
    )
    scales = tuple(
        math.sqrt(
            math.fsum(
                weight * (values[index] - means[index]) ** 2
                for weight, values in zip(weights, training_raw, strict=True)
            )
            / weight_sum
        )
        for index in range(len(FEATURE_NAMES))
    )
    if any(not math.isfinite(value) for value in (*means, *scales)) or any(
        value <= 0.0 for value in scales
    ):
        raise ValueError("training-only feature transform contains a zero or non-finite scale")

    candidate_raw_rows: list[tuple[float, ...]] = []
    normalized_rows: list[tuple[float, ...]] = []
    for row in candidates:
        raw = _raw_feature_vector(row.sequence)
        standardized = tuple(
            (value - mean) / scale for value, mean, scale in zip(raw, means, scales, strict=True)
        )
        norm = math.sqrt(math.fsum(value * value for value in standardized))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"candidate {row.sequence_id} has a zero or non-finite feature norm")
        normalized = tuple(value / norm for value in standardized)
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError(f"candidate {row.sequence_id} normalized features are non-finite")
        candidate_raw_rows.append(raw)
        normalized_rows.append(normalized)

    candidate_raw = np.asarray(candidate_raw_rows, dtype="<f8", order="C")
    training_raw_array = np.asarray(training_raw, dtype="<f8", order="C")
    matrix = np.asarray(normalized_rows, dtype="<f4", order="C")
    expected_shape = (len(candidates), len(FEATURE_NAMES))
    if matrix.shape != expected_shape or matrix.dtype.str != "<f4" or not matrix.flags.c_contiguous:
        raise AssertionError("diversity matrix shape, dtype, or order differs")
    if not bool(np.all(np.isfinite(matrix))):
        raise AssertionError("float32 diversity matrix contains non-finite values")
    return (
        matrix,
        means,
        scales,
        hashlib.sha256(candidate_raw.tobytes(order="C")).hexdigest(),
        hashlib.sha256(training_raw_array.tobytes(order="C")).hexdigest(),
        hashlib.sha256(np.asarray(weights, dtype="<f8").tobytes(order="C")).hexdigest(),
        weight_sum,
    )


def _float64_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("cannot serialize a non-finite float64 value")
    return f"{value:.17g}"


def _float32_text(value: np.float32) -> str:
    scalar = np.float32(value)
    if not bool(np.isfinite(scalar)):
        raise ValueError("cannot serialize a non-finite float32 value")
    text = str(scalar)
    if np.float32(float(text)).tobytes() != scalar.tobytes():
        raise AssertionError("NumPy float32 text is not round-trip exact")
    return text


def _selection_ledger_bytes(
    rows: Sequence[MeanLedgerRow],
    similarities: Sequence[CandidateSimilarity],
    matrix: NDArray[np.float32],
) -> bytes:
    if len(rows) != len(similarities) or matrix.shape != (len(rows), len(EMBEDDING_COLUMNS)):
        raise ValueError("selection ledger inputs are not row-aligned")
    output_rows: list[tuple[str, ...]] = []
    for index, (row, similarity) in enumerate(zip(rows, similarities, strict=True)):
        source_values = list(row.values)
        source_values[4] = row.source_library_eligible
        safe = similarity.reference_safe
        effective = row.source_library_eligible == "true" and safe
        witness_similarity = (
            ""
            if similarity.reference_witness_similarity is None
            else _float64_text(similarity.reference_witness_similarity)
        )
        output_rows.append(
            (
                *source_values,
                _float64_text(similarity.max_train_similarity),
                similarity.nearest_train_sequence_id,
                _float64_text(1.0 - similarity.max_train_similarity),
                "true" if safe else "false",
                similarity.reference_witness_sequence_id,
                witness_similarity,
                "true" if effective else "false",
                *(_float32_text(value) for value in matrix[index]),
            )
        )
    if any(name.startswith(FORBIDDEN_LEDGER_PREFIXES) for name in SELECTION_LEDGER_COLUMNS):
        raise AssertionError("selection ledger contains a prohibited field prefix")
    if "max_reference_indel_similarity" in SELECTION_LEDGER_COLUMNS:
        raise AssertionError("selection ledger must not claim a reference maximum")
    return _csv_bytes(SELECTION_LEDGER_COLUMNS, output_rows)


def _artifact_record(payload: bytes, *, rows: int | None = None) -> Mapping[str, object]:
    record: dict[str, object] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _fixed_quantiles(values: Sequence[float]) -> list[Mapping[str, float]]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("quantile input must contain finite values")
    ordered = sorted(values)
    probabilities = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    result: list[Mapping[str, float]] = []
    for probability in probabilities:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        value = (
            ordered[lower]
            if lower == upper
            else math.fsum(((1.0 - fraction) * ordered[lower], fraction * ordered[upper]))
        )
        result.append({"probability": probability, "value": value})
    return result


def _lineage_census(
    rows: Sequence[MeanLedgerRow], similarities: Sequence[CandidateSimilarity]
) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
    def summarize(column: int) -> Mapping[str, Mapping[str, int]]:
        counts: dict[str, list[int]] = {}
        for row, similarity in zip(rows, similarities, strict=True):
            source = row.source_library_eligible == "true"
            safe = similarity.reference_safe
            effective = source and safe
            for token in row.values[column].split("|"):
                values = counts.setdefault(token, [0, 0, 0, 0])
                values[0] += 1
                values[1] += int(source)
                values[2] += int(safe)
                values[3] += int(effective)
        return {
            token: {
                "candidates": values[0],
                "source_library_eligible": values[1],
                "reference_safe": values[2],
                "effective_eligible": values[3],
            }
            for token, values in sorted(counts.items())
        }

    return {"families": summarize(6), "variants": summarize(7)}


def _feature_transform_document(
    *,
    config: CandidateDiversityNoveltyConfig,
    matrix_payload: bytes,
    means: Sequence[float],
    scales: Sequence[float],
    candidate_raw_sha256: str,
    training_raw_sha256: str,
    weights_sha256: str,
    weight_sum: float,
) -> Mapping[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "candidate_diversity_feature_transform_v1",
        "status": OUTPUT_STATUS,
        "automatic_production_eligible": False,
        "config_sha256": config.snapshot.sha256,
        "feature_space": FEATURE_SPACE,
        "feature_names": list(FEATURE_NAMES),
        "embedding_columns": list(EMBEDDING_COLUMNS),
        "descriptor_settings": {
            "descriptor_dataclass_order": "PeptideDescriptors_field_order",
            "ph": 7.4,
            "free_termini": True,
            "hydrophobic_moment_angle_degrees": 100.0,
            "isoelectric_point": {"iterations": 60, "lower_ph": 0.0, "upper_ph": 14.0},
            "residue_fraction_order": ALPHABET,
        },
        "fit": {
            "rows": config.expected_training_rows,
            "source": "accepted_training_projection_only",
            "source_sha256": config.training_projection.projection_sha256,
            "weight_field": TRAINING_WEIGHTING,
            "weight_sum": weight_sum,
            "weight_sum_hex": weight_sum.hex(),
            "weights_float64_little_endian_sha256": weights_sha256,
            "training_raw_feature_matrix_float64_little_endian_sha256": training_raw_sha256,
            "weighted_means": list(means),
            "weighted_population_scales": list(scales),
            "weighted_mean_algorithm": MEAN_ALGORITHM,
            "weighted_scale_algorithm": SCALE_ALGORITHM,
        },
        "transform": {
            "standardization": STANDARDIZATION,
            "row_normalization": ROW_NORMALIZATION,
            "zero_row_norm_policy": "fail_closed",
            "zero_scale_policy": "fail_closed",
        },
        "candidate_raw_feature_matrix": {
            "byte_order": "little_endian",
            "dtype": "float64",
            "memory_order": "C",
            "shape": [config.expected_candidates, len(FEATURE_NAMES)],
            "sha256": candidate_raw_sha256,
        },
        "output_matrix": {
            "byte_order": "little_endian",
            "dtype": "float32",
            "filename": "diversity_feature_matrix.f32le",
            "memory_order": "C",
            "raw_bytes_sha256": hashlib.sha256(matrix_payload).hexdigest(),
            "shape": [config.expected_candidates, len(FEATURE_NAMES)],
            "size_bytes": len(matrix_payload),
        },
    }


def _manifest_document(
    *,
    config: CandidateDiversityNoveltyConfig,
    authenticated: AuthenticatedDiversityInputs,
    similarities: Sequence[CandidateSimilarity],
    selection_payload: bytes,
    matrix_payload: bytes,
    transform_payload: bytes,
) -> Mapping[str, object]:
    maxima = [item.max_train_similarity for item in similarities]
    novelty = [1.0 - value for value in maxima]
    source_eligible = sum(row.source_library_eligible == "true" for row in authenticated.mean_rows)
    reference_safe = sum(item.reference_safe for item in similarities)
    effective = sum(
        row.source_library_eligible == "true" and item.reference_safe
        for row, item in zip(authenticated.mean_rows, similarities, strict=True)
    )
    witness_count = sum(bool(item.reference_witness_sequence_id) for item in similarities)
    mean = config.mean_ledger
    training = config.training_projection
    reference = config.organizer_reference
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": OUTPUT_STATUS,
        "automatic_production_eligible": False,
        "allowed_consumer": ALLOWED_CONSUMER,
        "config_sha256": config.snapshot.sha256,
        "inputs": {
            "mean_ledger": {
                "producer_job_id": mean.producer_job_id,
                "audit_job_id": mean.audit_job_id,
                "git_commit": mean.git_commit,
                "config_sha256": mean.config_sha256,
                "publication_top_sha256": mean.publication_top_sha256,
                "publication_tree_sha256": mean.publication_tree_sha256,
                "candidate_ledger_sha256": mean.candidate_ledger_sha256,
                "diagnostics_sha256": mean.diagnostics_sha256,
                "manifest_sha256": mean.manifest_sha256,
                "independent_verification_sha256": mean.independent_verification_sha256,
                "operational_audit_sha256": mean.operational_audit_sha256,
                "model_release_id": mean.model_release_id,
            },
            "training_projection": {
                "stage_job_id": training.stage_job_id,
                "git_commit": training.git_commit,
                "projection_sha256": training.projection_sha256,
                "projection_size_bytes": training.projection_size_bytes,
                "receipt_sha256": training.receipt_sha256,
                "receipt_size_bytes": training.receipt_size_bytes,
                "rows": config.expected_training_rows,
            },
            "organizer_reference": {
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
                "source_commit": reference.source_commit,
                "record_count": authenticated.reference_record_count,
                "unique_sequence_count": len(authenticated.reference_rows),
                "role": reference.role,
            },
        },
        "candidate_selection_ledger": {
            "filename": "candidate_selection_ledger.csv",
            "columns": list(SELECTION_LEDGER_COLUMNS),
            "eligibility_mapping": (
                "eligible_equals_source_library_eligible_and_reference_safe_0_80"
            ),
            **_artifact_record(selection_payload, rows=config.expected_candidates),
        },
        "diversity_feature_matrix": {
            "filename": "diversity_feature_matrix.f32le",
            "feature_space": FEATURE_SPACE,
            "shape": [config.expected_candidates, len(FEATURE_NAMES)],
            "dtype": "float32",
            "byte_order": "little_endian",
            "memory_order": "C",
            "csv_columns": list(EMBEDDING_COLUMNS),
            **_artifact_record(matrix_payload),
        },
        "feature_transform": {
            "filename": "feature_transform.json",
            **_artifact_record(transform_payload),
        },
        "similarity": {
            "implementation": SIMILARITY_IMPLEMENTATION,
            "value_range": "zero_to_one",
            "training_reduction": "exact_maximum_over_all_accepted_training_sequences",
            "nearest_training_tie_break": "lexicographically_smallest_sequence_id",
            "novelty": "one_minus_max_train_indel_similarity",
            "reference_rule": (
                "unsafe_if_any_unique_organizer_reference_similarity_strictly_exceeds_threshold"
            ),
            "reference_similarity_threshold": REFERENCE_THRESHOLD,
            "reference_threshold_equality": "safe",
            "reference_witness_tie_break": "lexicographically_smallest_sequence_id",
            "reference_maximum_similarity_calculated_or_stored": False,
            "length_pruning": LENGTH_BOUND,
            "candidate_chunk_size": config.policy.candidate_chunk_size,
            "choice_chunk_size": config.policy.choice_chunk_size,
            "worker_bounds_inclusive": [
                config.policy.worker_minimum,
                config.policy.worker_maximum,
            ],
            "worker_count_affects_output_bytes": False,
        },
        "census": {
            "candidates": config.expected_candidates,
            "source_library_eligible": source_eligible,
            "reference_safe": reference_safe,
            "reference_unsafe_witnesses": witness_count,
            "effective_eligible": effective,
            "training_similarity": {
                "minimum": min(maxima),
                "maximum": max(maxima),
                "quantile_method": "sorted_linear_interpolation_h_equals_probability_times_n_minus_one",
                "quantiles": _fixed_quantiles(maxima),
            },
            "novelty": {
                "minimum": min(novelty),
                "maximum": max(novelty),
                "quantile_method": "sorted_linear_interpolation_h_equals_probability_times_n_minus_one",
                "quantiles": _fixed_quantiles(novelty),
            },
        },
        "lineage_census": _lineage_census(authenticated.mean_rows, similarities),
        "claims": {
            "calibrated_uncertainty": False,
            "cluster_assignment": False,
            "development_diversity_novelty_evidence": True,
            "esm_embedding": False,
            "final_ranking": False,
            "hemolysis_risk": False,
            "mdr_eskape_activity": False,
            "out_of_distribution": False,
            "production_ensemble": False,
            "quality_probability": False,
            "reference_rule_is_biological_safety": False,
            "reference_witness_is_reference_maximum": False,
            "selectivity": False,
            "uncertainty_available": False,
        },
        "exclusions": {
            "candidate_ledger_column_prefixes": list(FORBIDDEN_LEDGER_PREFIXES),
            "model_fit_sensitivity_diagnostics_consumed": False,
            "reference_maximum_similarity_field": False,
        },
    }


def compute_candidate_diversity_novelty(
    authenticated: AuthenticatedDiversityInputs,
    *,
    config: CandidateDiversityNoveltyConfig,
    workers: int,
) -> ComputedDiversityBundle:
    """Compute deterministic publication bytes from authenticated semantics."""

    if len(authenticated.mean_rows) != config.expected_candidates:
        raise ValueError("authenticated candidate census differs before computation")
    similarities = compute_candidate_similarities(authenticated, config=config, workers=workers)
    (
        matrix,
        means,
        scales,
        candidate_raw_sha256,
        training_raw_sha256,
        weights_sha256,
        weight_sum,
    ) = _feature_matrices(authenticated.mean_rows, authenticated.training_rows)
    matrix_payload = matrix.tobytes(order="C")
    selection_payload = _selection_ledger_bytes(authenticated.mean_rows, similarities, matrix)
    transform_payload = _canonical_json_bytes(
        _feature_transform_document(
            config=config,
            matrix_payload=matrix_payload,
            means=means,
            scales=scales,
            candidate_raw_sha256=candidate_raw_sha256,
            training_raw_sha256=training_raw_sha256,
            weights_sha256=weights_sha256,
            weight_sum=weight_sum,
        )
    )
    manifest_payload = _canonical_json_bytes(
        _manifest_document(
            config=config,
            authenticated=authenticated,
            similarities=similarities,
            selection_payload=selection_payload,
            matrix_payload=matrix_payload,
            transform_payload=transform_payload,
        )
    )
    return ComputedDiversityBundle(
        selection_ledger=selection_payload,
        feature_matrix=matrix_payload,
        feature_transform=transform_payload,
        manifest=manifest_payload,
    )


def _revalidate_authenticated_inputs_for_publication(
    *,
    config: CandidateDiversityNoveltyConfig,
    authenticated: AuthenticatedDiversityInputs,
) -> AuthenticatedDiversityInputs:
    reloaded_config = load_candidate_diversity_novelty_config(config.path)
    if reloaded_config != config:
        raise ValueError("config object does not match its authenticated bytes")
    expected_files = {
        *(f"mean/{name}" for name in MEAN_FILES),
        *(f"mean_audit/{name}" for name in MEAN_AUDIT_FILES),
        *(f"training/{name}" for name in TRAINING_STAGE_FILES),
        "organizer_reference",
    }
    if set(authenticated.snapshots) != expected_files:
        raise ValueError("authenticated input file snapshot inventory is not exact")
    if set(authenticated.directory_snapshots) != {
        "mean-ledger directory",
        "mean-ledger audit directory",
        "training stage directory",
    }:
        raise ValueError("authenticated input directory snapshot inventory is not exact")
    mean = {name: authenticated.snapshots[f"mean/{name}"] for name in MEAN_FILES}
    audit = {name: authenticated.snapshots[f"mean_audit/{name}"] for name in MEAN_AUDIT_FILES}
    training = {name: authenticated.snapshots[f"training/{name}"] for name in TRAINING_STAGE_FILES}
    mean_directory = authenticated.directory_snapshots["mean-ledger directory"]
    audit_directory = authenticated.directory_snapshots["mean-ledger audit directory"]
    training_directory = authenticated.directory_snapshots["training stage directory"]
    if mean_directory.entries != tuple(sorted(MEAN_FILES)) or any(
        snapshot.path.parent != mean_directory.path or snapshot.path.name != name
        for name, snapshot in mean.items()
    ):
        raise ValueError("mean-ledger directory snapshot does not bind its files")
    if audit_directory.entries != tuple(sorted(MEAN_AUDIT_FILES)) or any(
        snapshot.path.parent != audit_directory.path or snapshot.path.name != name
        for name, snapshot in audit.items()
    ):
        raise ValueError("mean-ledger audit snapshot does not bind its files")
    if training_directory.entries != tuple(sorted(TRAINING_STAGE_FILES)) or any(
        snapshot.path.parent != training_directory.path or snapshot.path.name != name
        for name, snapshot in training.items()
    ):
        raise ValueError("training-stage snapshot does not bind its files")

    reconstructed = authenticate_candidate_diversity_novelty_inputs(
        mean_ledger_twin=mean_directory.path,
        mean_ledger_audit_dir=audit_directory.path,
        training_projection=training["training_projection.jsonl"].path,
        projection_receipt=training["projection-receipt.json"].path,
        organizer_reference=authenticated.snapshots["organizer_reference"].path,
        config=reloaded_config,
    )
    if reconstructed != authenticated:
        raise ValueError("passed authenticated semantics differ from fresh reconstruction")
    return reconstructed


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
                raise OSError("short write while publishing diversity/novelty artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_write_once(
    output_dir: str | Path,
    *,
    payloads: Mapping[str, bytes],
    snapshots: Mapping[str, InputSnapshot],
    directory_snapshots: Mapping[str, DirectorySnapshot],
) -> Path:
    if set(payloads) != set(OUTPUT_FILES):
        raise ValueError("diversity/novelty output inventory is not exact")
    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested):
        raise FileExistsError(f"refusing to reuse diversity/novelty output: {requested}")
    parent = requested.parent
    _reject_symlink_chain(parent, label="diversity/novelty output parent")
    try:
        parent_metadata = os.lstat(parent)
    except OSError as error:
        raise ValueError(f"cannot inspect diversity/novelty output parent: {parent}") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("diversity/novelty output parent must be a directory")
    claimed = False
    created: list[str] = []

    def assert_inputs_unchanged() -> None:
        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)
        for label, snapshot in directory_snapshots.items():
            _assert_directory_snapshot_unchanged(snapshot, label=label)

    try:
        assert_inputs_unchanged()
        # The private directory may be visible by name while it is assembled,
        # but SHA256SUMS is the commit marker and is created only after every
        # semantic payload and the final input-invariance check succeed.
        os.mkdir(requested, mode=0o700)
        claimed = True
        for name in OUTPUT_FILES[:-1]:
            created.append(name)
            _write_new_bytes(requested / name, payloads[name])
            os.chmod(requested / name, 0o444)
        assert_inputs_unchanged()
        created.append("SHA256SUMS")
        _write_new_bytes(requested / "SHA256SUMS", payloads["SHA256SUMS"])
        os.chmod(requested / "SHA256SUMS", 0o444)
        _fsync_directory(requested)
        os.chmod(requested, 0o555)
        _fsync_directory(requested)
        _fsync_directory(parent)
        return requested
    except BaseException:
        if claimed:
            with suppress(OSError):
                os.chmod(requested, 0o700)
            for name in reversed(created):
                with suppress(FileNotFoundError):
                    os.unlink(requested / name)
            with suppress(OSError):
                os.rmdir(requested)
        raise


def publish_candidate_diversity_novelty(
    *,
    output_dir: str | Path,
    config: CandidateDiversityNoveltyConfig,
    authenticated: AuthenticatedDiversityInputs,
    workers: int | None = None,
) -> CandidateDiversityNoveltyExecution:
    """Revalidate inputs, compute all bytes, then seal the exact five-file bundle."""

    reconstructed = _revalidate_authenticated_inputs_for_publication(
        config=config, authenticated=authenticated
    )
    worker_count = config.policy.worker_default if workers is None else workers
    computed = compute_candidate_diversity_novelty(
        reconstructed,
        config=config,
        workers=_validate_workers(worker_count, config=config),
    )
    semantic_payloads = {
        "candidate_selection_ledger.csv": computed.selection_ledger,
        "diversity_feature_matrix.f32le": computed.feature_matrix,
        "feature_transform.json": computed.feature_transform,
        "manifest.json": computed.manifest,
    }
    payloads = {
        **semantic_payloads,
        "SHA256SUMS": _sha256sums_bytes(semantic_payloads),
    }
    output = _publish_write_once(
        output_dir,
        payloads=payloads,
        snapshots={"diversity/novelty config": config.snapshot, **reconstructed.snapshots},
        directory_snapshots=reconstructed.directory_snapshots,
    )
    return CandidateDiversityNoveltyExecution(
        output_dir=output,
        candidate_count=len(reconstructed.mean_rows),
        selection_ledger_sha256=hashlib.sha256(computed.selection_ledger).hexdigest(),
        feature_matrix_sha256=hashlib.sha256(computed.feature_matrix).hexdigest(),
        feature_transform_sha256=hashlib.sha256(computed.feature_transform).hexdigest(),
        manifest_sha256=hashlib.sha256(computed.manifest).hexdigest(),
        publication_top_sha256=hashlib.sha256(payloads["SHA256SUMS"]).hexdigest(),
    )


def run_candidate_diversity_novelty(
    *,
    mean_ledger_twin: str | Path,
    mean_ledger_audit_dir: str | Path,
    training_projection: str | Path,
    projection_receipt: str | Path,
    organizer_reference: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    workers: int | None = None,
) -> CandidateDiversityNoveltyExecution:
    """Authenticate accepted evidence and publish selection features."""

    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested):
        raise FileExistsError(f"refusing to reuse diversity/novelty output: {requested}")
    config = load_candidate_diversity_novelty_config(config_path)
    authenticated = authenticate_candidate_diversity_novelty_inputs(
        mean_ledger_twin=mean_ledger_twin,
        mean_ledger_audit_dir=mean_ledger_audit_dir,
        training_projection=training_projection,
        projection_receipt=projection_receipt,
        organizer_reference=organizer_reference,
        config=config,
    )
    return publish_candidate_diversity_novelty(
        output_dir=requested,
        config=config,
        authenticated=authenticated,
        workers=workers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build authenticated development diversity and novelty selection features."
    )
    parser.add_argument("--mean-ledger-twin", type=Path, required=True)
    parser.add_argument("--mean-ledger-audit-dir", type=Path, required=True)
    parser.add_argument("--training-projection", type=Path, required=True)
    parser.add_argument(
        "--projection-receipt",
        "--training-projection-receipt",
        dest="projection_receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--organizer-reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = run_candidate_diversity_novelty(
            mean_ledger_twin=args.mean_ledger_twin,
            mean_ledger_audit_dir=args.mean_ledger_audit_dir,
            training_projection=args.training_projection,
            projection_receipt=args.projection_receipt,
            organizer_reference=args.organizer_reference,
            config_path=args.config,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"candidate diversity/novelty error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_count": execution.candidate_count,
                "feature_matrix_sha256": execution.feature_matrix_sha256,
                "feature_transform_sha256": execution.feature_transform_sha256,
                "manifest_sha256": execution.manifest_sha256,
                "output_dir": str(execution.output_dir),
                "publication_top_sha256": execution.publication_top_sha256,
                "selection_ledger_sha256": execution.selection_ledger_sha256,
                "status": OUTPUT_STATUS,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the cluster CLI.
    raise SystemExit(main())

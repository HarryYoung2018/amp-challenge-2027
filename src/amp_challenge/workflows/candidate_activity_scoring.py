"""Authenticated development-only descriptor activity scoring.

This workflow deliberately implements one narrow handoff.  It replays the
accepted five-fold Gate-1 descriptor baseline, fits the five fold-complement
members and one all-data deployment member, and scores the accepted candidate
pool.  The fold-member spread is named ``model_fit_sensitivity``: it is a
refit diagnostic, not a calibrated uncertainty estimate.

The explicit development CLI must be invoked from a CPU Slurm allocation after
the preregistered inputs exist.  Its presence does not confer production status.
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
from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.models.oracle_baselines import DescriptorLogisticOracle, OracleInput
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

FloatArray = NDArray[np.float64]
GramClass = Literal["positive", "negative"]

SCHEMA_VERSION = 1
ARTIFACT = "candidate_activity_scoring_development_v1"
ACCEPTED_MODEL = "descriptor_logistic"
PROBABILITY_CALIBRATION = "none_raw_logistic_probability"
OUTPUT_STATUS = "development_candidate_scoring_only"
PREDICTION_SCOPE = "declared_seven_target_activity_panel"
TRAINING_SCOPE = "all_2492_accepted_gate1_contexts_after_recipe_freeze"
CALIBRATION_SCOPE = PROBABILITY_CALIBRATION

_EXPECTED_CANDIDATES = 115_536
_EXPECTED_EXAMPLES = 2_492
_EXPECTED_FOLDS = 5
_EXPECTED_BATCH_SIZE = 2_048
_EXPECTED_OOF_ABSOLUTE_TOLERANCE = 1e-12

_CONFIG_STATUS = "predeclared_development_candidate_scoring_only"
_TARGET_AGGREGATION = "equal_target_arithmetic_mean_within_declared_scope_v1"
_FIT_SENSITIVITY = (
    "population_sd_across_five_outer_fold_complement_refits_diagnostic_not_posterior_or_"
    "aleatoric_uncertainty"
)
_OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
_FORBIDDEN_CLAIMS = (
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
_MODEL_WEIGHTS = {
    "apex": 0.0,
    "descriptor_logistic": 1.0,
    "esm_supervised": 0.0,
    "geometry": 0.0,
    "homology_knn": 0.0,
}
_TARGET_PANEL: tuple[tuple[str, GramClass], ...] = (
    ("acinetobacter_baumannii", "negative"),
    ("enterococcus_faecalis", "positive"),
    ("enterococcus_faecium", "positive"),
    ("escherichia_coli", "negative"),
    ("klebsiella_pneumoniae", "negative"),
    ("pseudomonas_aeruginosa", "negative"),
    ("staphylococcus_aureus", "positive"),
)
_EXPECTED_TARGET_COUNTS = {"negative": 4, "positive": 3}
_OUTPUT_FILES = (
    "candidate_activity_scores.csv",
    "fold_model_target_probabilities.npy",
    "model_states.json",
    "oof_reproduction.json",
    "manifest.json",
    "SHA256SUMS",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_MEMBER_RE = re.compile(r"(?:outer_fold_[0-4]|all_data_deployment)")
_COMPONENT_RE = re.compile(r"[^\x00-\x20\x7f]{1,256}")

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
_CANDIDATE_FIELDS = frozenset(
    {"schema_version", "sequence_id", "sequence", "length", "library_eligible", "lineages"}
)
_LINEAGE_FIELDS = frozenset(
    {
        "schema_version",
        "generator_family",
        "generator_variant",
        "logical_sha256",
        "training_projection_sha256",
        "seed",
        "ordinal",
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
_SEQUENCE_FEATURE_ORDER = (
    "length",
    "molecular_weight_da",
    "net_charge",
    "charge_density",
    "isoelectric_point",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "hydrophobic_fraction",
    "aromatic_fraction",
    "basic_fraction",
    "acidic_fraction",
    "shannon_entropy",
    "max_residue_fraction",
    *(f"composition_{residue}" for residue in STANDARD_AMINO_ACIDS),
)


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    """Immutable bytes and identity metadata captured from one input file."""

    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    fingerprint: tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class DescriptorSettings:
    l2: float
    max_iterations: int
    tolerance: float
    prior_strength: float

    def as_dict(self) -> dict[str, object]:
        return {
            "l2": self.l2,
            "max_iterations": self.max_iterations,
            "prior_strength": self.prior_strength,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class CandidatePoolContract:
    producer_job_id: int
    git_commit: str
    publication_top_sha256: str
    manifest_sha256: str
    candidates_sha256: str
    validation_summary_sha256: str
    final_publication_check_sha256: str


@dataclass(frozen=True, slots=True)
class Gate1Contract:
    producer_job_id: int
    audit_job_id: int
    git_commit: str
    publication_top_sha256: str
    semantic_top_sha256: str
    examples_sha256: str
    oof_sha256: str
    independent_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    gram: GramClass


@dataclass(frozen=True, slots=True)
class CandidateActivityScoringConfig:
    path: Path
    snapshot: InputSnapshot
    schema_version: int
    artifact: str
    status: str
    automatic_production_eligible: bool
    expected_candidates: int
    expected_examples: int
    folds: int
    batch_size: int
    oof_absolute_tolerance: float
    objectives: tuple[str, ...]
    target_aggregation: str
    calibration_scope: str
    fit_sensitivity_semantics: str
    training_scope: str
    historical_method_choice_used_fold4: bool
    organizer_reference_used_for_model_fit: bool
    forbidden_claims: tuple[str, ...]
    candidate_pool: CandidatePoolContract
    gate1: Gate1Contract
    descriptor: DescriptorSettings
    model_weights: Mapping[str, float]
    targets: tuple[TargetSpec, ...]


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    source_ordinal: int
    sequence_id: str
    sequence: str
    length: int
    library_eligible: bool
    generator_families: tuple[str, ...]
    generator_variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Gate1Example:
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

    @property
    def model_input(self) -> OracleInput:
        return OracleInput(
            sequence=self.sequence,
            strain=self.canonical_target,
            gram=self.gram,
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedCandidates:
    records: tuple[CandidateRecord, ...]
    snapshots: Mapping[str, InputSnapshot]


@dataclass(frozen=True, slots=True)
class AuthenticatedGate1:
    examples: tuple[Gate1Example, ...]
    descriptor_oof: Mapping[str, float]
    snapshots: Mapping[str, InputSnapshot]


@dataclass(frozen=True, slots=True)
class FittedActivityModels:
    fold_models: tuple[DescriptorLogisticOracle, ...]
    deployment_model: DescriptorLogisticOracle
    model_states: tuple[Mapping[str, object], ...]
    oof_reproduction: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ActivityScoreBatch:
    deployment_target_probabilities: FloatArray
    fold_model_target_probabilities: FloatArray
    model_fit_sensitivity_target: FloatArray
    means: Mapping[str, FloatArray]
    model_fit_sensitivity: Mapping[str, FloatArray]


@dataclass(frozen=True, slots=True)
class CandidateActivityScoringExecution:
    output_dir: Path
    candidate_count: int
    model_release_id: str
    csv_sha256: str
    fold_tensor_sha256: str
    fold_tensor_raw_data_sha256: str
    manifest_sha256: str


_FROZEN_DESCRIPTOR_SETTINGS = DescriptorSettings(
    l2=0.10,
    max_iterations=100,
    tolerance=1e-9,
    prior_strength=2.0,
)
_FROZEN_CANDIDATE_POOL_CONTRACT = CandidatePoolContract(
    producer_job_id=225275,
    git_commit="50435d805a73ec4756cde1f7db1e6e7a06d30eb4",
    publication_top_sha256="d248fe898629f83b9b0c5fc478a0d859dec46c95a82de663812a4b107eb2174c",
    manifest_sha256="2f6ec8de3d65f9580aef12bbe9bf4b414f13a831e76b8dbc7504c158a3d91990",
    candidates_sha256="77315bfbec533f5587123714199c2dd2eb2a51a16d95957ab53927494983dbf2",
    validation_summary_sha256=("58e93fd08f670b0e8e8118bbd29a1a7c1378081fc17a646e87d3082abfbb135c"),
    final_publication_check_sha256=(
        "ae8d5d9ecfcfbd3d5007715a52cab0e7f2f598e1f709eda6468ba5385af91296"
    ),
)
_FROZEN_GATE1_CONTRACT = Gate1Contract(
    producer_job_id=223248,
    audit_job_id=223250,
    git_commit="0468cc2cbc0b7c3b2a50f7866da1b1083c8ef1a8",
    publication_top_sha256="4e259cfb43033c069598d42fd54fec49a67ba55bfbb5c1e77eeef4887815fe2f",
    semantic_top_sha256="556e06fd2b1af1e678de88008bc1b87434fb8cf1179f38c897de7ee2c9fd779d",
    examples_sha256="d3eecbf3014fd78cf7021818466893d292315b6e90e77cea85d4e1fbd5bec520",
    oof_sha256="11eea907a242606b78261a9b37107647ce393db743be95c146cbbe5caced137c",
    independent_receipt_sha256=("1f6a6ce811d3fbecdbbbe7463e6052dc7766130372926d29952c70be65743e81"),
)


def _require_exact_fields(
    value: object, *, expected: Collection[str], label: str
) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a table/object")
    document = cast(dict[str, Any], value)
    missing = set(expected) - set(document)
    extra = set(document) - set(expected)
    if missing or extra:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return document


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return cast(bool, value)


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return cast(int, value)


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return cast(int, value)


def _require_number(value: object, *, label: str, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{label} must be a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(cast(str, value)) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _require_git_sha(value: object, *, label: str) -> str:
    if type(value) is not str or _GIT_SHA_RE.fullmatch(cast(str, value)) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA")
    return cast(str, value)


def _require_name(value: object, *, label: str) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(cast(str, value)) is None:
        raise ValueError(f"{label} must be a lowercase underscore identifier")
    return cast(str, value)


def _fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in reversed((absolute, *absolute.parents)):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} cannot traverse a symbolic link: {candidate}")


def _snapshot_regular(path: str | Path, *, label: str) -> InputSnapshot:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source, label=label)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {source}") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError(f"{label} must be a regular non-symbolic file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    identities = {
        _fingerprint(item) for item in (named_before, opened_before, opened_after, named_after)
    }
    _reject_symlink_chain(source, label=label)
    if len(identities) != 1:
        raise ValueError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != named_before.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return InputSnapshot(
        path=source,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        fingerprint=_fingerprint(named_before),
    )


def _assert_snapshot_unchanged(snapshot: InputSnapshot, *, label: str) -> None:
    _reject_symlink_chain(snapshot.path, label=label)
    try:
        metadata = os.lstat(snapshot.path)
    except OSError as error:
        raise ValueError(f"{label} disappeared after authentication") from error
    if not stat.S_ISREG(metadata.st_mode) or _fingerprint(metadata) != snapshot.fingerprint:
        raise ValueError(f"{label} changed after authentication")


def _expect_hash(snapshot: InputSnapshot, expected: str, *, label: str) -> None:
    if snapshot.sha256 != expected:
        raise ValueError(
            f"{label} checksum differs from the preregistration: "
            f"expected {expected}, got {snapshot.sha256}"
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


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


def _parse_json(payload: bytes, *, label: str, canonical: bool) -> object:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must use LF framing with exactly one final LF")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict UTF-8 JSON") from error
    if canonical and _canonical_json_bytes(document) != payload:
        raise ValueError(f"{label} is not canonical compact JSON")
    return document


def _parse_canonical_jsonl(payload: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty canonical LF-framed JSONL")
    rows: list[Mapping[str, Any]] = []
    for number, line in enumerate(payload.splitlines(keepends=True), 1):
        document = _parse_json(line, label=f"{label} row {number}", canonical=True)
        if type(document) is not dict:
            raise ValueError(f"{label} row {number} must be a JSON object")
        rows.append(cast(dict[str, Any], document))
    return tuple(rows)


def _parse_sequence(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    sequence = cast(str, value)
    try:
        canonical = canonicalize_sequence(sequence)
    except ValueError as error:
        raise ValueError(f"{label} is not a valid peptide sequence") from error
    if canonical != sequence:
        raise ValueError(f"{label} must already be canonical")
    return sequence


def _parse_descriptor_settings(value: object) -> DescriptorSettings:
    raw = _require_exact_fields(
        value,
        expected={"l2", "max_iterations", "tolerance", "prior_strength"},
        label="descriptor_logistic",
    )
    settings = DescriptorSettings(
        l2=_require_number(raw["l2"], label="descriptor_logistic.l2", positive=True),
        max_iterations=_require_positive_int(
            raw["max_iterations"], label="descriptor_logistic.max_iterations"
        ),
        tolerance=_require_number(
            raw["tolerance"], label="descriptor_logistic.tolerance", positive=True
        ),
        prior_strength=_require_number(
            raw["prior_strength"], label="descriptor_logistic.prior_strength"
        ),
    )
    if settings.prior_strength < 0.0:
        raise ValueError("descriptor_logistic.prior_strength cannot be negative")
    return settings


def _parse_candidate_pool_contract(value: object) -> CandidatePoolContract:
    raw = _require_exact_fields(
        value,
        expected={
            "producer_job_id",
            "git_commit",
            "publication_top_sha256",
            "manifest_sha256",
            "candidates_sha256",
            "validation_summary_sha256",
            "final_publication_check_sha256",
        },
        label="candidate_pool",
    )
    return CandidatePoolContract(
        producer_job_id=_require_positive_int(
            raw["producer_job_id"], label="candidate_pool.producer_job_id"
        ),
        git_commit=_require_git_sha(raw["git_commit"], label="candidate_pool.git_commit"),
        publication_top_sha256=_require_sha256(
            raw["publication_top_sha256"], label="candidate_pool.publication_top_sha256"
        ),
        manifest_sha256=_require_sha256(
            raw["manifest_sha256"], label="candidate_pool.manifest_sha256"
        ),
        candidates_sha256=_require_sha256(
            raw["candidates_sha256"], label="candidate_pool.candidates_sha256"
        ),
        validation_summary_sha256=_require_sha256(
            raw["validation_summary_sha256"],
            label="candidate_pool.validation_summary_sha256",
        ),
        final_publication_check_sha256=_require_sha256(
            raw["final_publication_check_sha256"],
            label="candidate_pool.final_publication_check_sha256",
        ),
    )


def _parse_gate1_contract(value: object) -> Gate1Contract:
    raw = _require_exact_fields(
        value,
        expected={
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "publication_top_sha256",
            "semantic_top_sha256",
            "examples_sha256",
            "oof_sha256",
            "independent_receipt_sha256",
        },
        label="gate1",
    )
    return Gate1Contract(
        producer_job_id=_require_positive_int(
            raw["producer_job_id"], label="gate1.producer_job_id"
        ),
        audit_job_id=_require_positive_int(raw["audit_job_id"], label="gate1.audit_job_id"),
        git_commit=_require_git_sha(raw["git_commit"], label="gate1.git_commit"),
        publication_top_sha256=_require_sha256(
            raw["publication_top_sha256"], label="gate1.publication_top_sha256"
        ),
        semantic_top_sha256=_require_sha256(
            raw["semantic_top_sha256"], label="gate1.semantic_top_sha256"
        ),
        examples_sha256=_require_sha256(raw["examples_sha256"], label="gate1.examples_sha256"),
        oof_sha256=_require_sha256(raw["oof_sha256"], label="gate1.oof_sha256"),
        independent_receipt_sha256=_require_sha256(
            raw["independent_receipt_sha256"], label="gate1.independent_receipt_sha256"
        ),
    )


def _config_from_snapshot(snapshot: InputSnapshot) -> CandidateActivityScoringConfig:
    payload = snapshot.payload
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("scoring config must use LF framing with exactly one final LF")
    try:
        raw_value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("scoring config is not valid UTF-8 TOML") from error
    raw = _require_exact_fields(
        raw_value,
        expected={
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "expected_candidates",
            "expected_examples",
            "folds",
            "batch_size",
            "oof_absolute_tolerance",
            "objectives",
            "target_aggregation",
            "calibration_scope",
            "fit_sensitivity_semantics",
            "training_scope",
            "historical_method_choice_used_fold4",
            "organizer_reference_used_for_model_fit",
            "forbidden_claims",
            "candidate_pool",
            "gate1",
            "descriptor_logistic",
            "model_weights",
            "target",
        },
        label="scoring config",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("scoring config schema_version must be 1")
    if raw["artifact"] != ARTIFACT or type(raw["artifact"]) is not str:
        raise ValueError(f"scoring config artifact must be {ARTIFACT!r}")
    if raw["status"] != _CONFIG_STATUS or type(raw["status"]) is not str:
        raise ValueError(f"scoring config status must be {_CONFIG_STATUS!r}")
    if _require_bool(raw["automatic_production_eligible"], label="automatic_production_eligible"):
        raise ValueError("automatic_production_eligible must be false")
    folds = _require_positive_int(raw["folds"], label="folds")
    if folds != _EXPECTED_FOLDS:
        raise ValueError("this accepted Gate-1 replay requires exactly five folds")
    expected_candidates = _require_positive_int(
        raw["expected_candidates"], label="expected_candidates"
    )
    if expected_candidates != _EXPECTED_CANDIDATES:
        raise ValueError(f"expected_candidates must be the frozen value {_EXPECTED_CANDIDATES}")
    expected_examples = _require_positive_int(raw["expected_examples"], label="expected_examples")
    if expected_examples != _EXPECTED_EXAMPLES:
        raise ValueError(f"expected_examples must be the frozen value {_EXPECTED_EXAMPLES}")
    batch_size = _require_positive_int(raw["batch_size"], label="batch_size")
    if batch_size != _EXPECTED_BATCH_SIZE:
        raise ValueError(f"batch_size must be the frozen value {_EXPECTED_BATCH_SIZE}")
    absolute_tolerance = _require_number(
        raw["oof_absolute_tolerance"], label="oof_absolute_tolerance", positive=True
    )
    if absolute_tolerance != _EXPECTED_OOF_ABSOLUTE_TOLERANCE:
        raise ValueError(
            "oof_absolute_tolerance must be the frozen value "
            f"{_EXPECTED_OOF_ABSOLUTE_TOLERANCE:.17g}"
        )

    objectives_raw = raw["objectives"]
    if type(objectives_raw) is not list or tuple(objectives_raw) != _OBJECTIVES:
        raise ValueError(f"objectives must exactly equal {list(_OBJECTIVES)!r}")
    forbidden_raw = raw["forbidden_claims"]
    if type(forbidden_raw) is not list or tuple(forbidden_raw) != _FORBIDDEN_CLAIMS:
        raise ValueError(f"forbidden_claims must exactly equal {list(_FORBIDDEN_CLAIMS)!r}")
    scalar_boundaries = {
        "target_aggregation": _TARGET_AGGREGATION,
        "calibration_scope": PROBABILITY_CALIBRATION,
        "fit_sensitivity_semantics": _FIT_SENSITIVITY,
        "training_scope": TRAINING_SCOPE,
    }
    for field, expected in scalar_boundaries.items():
        if type(raw[field]) is not str or raw[field] != expected:
            raise ValueError(f"{field} must be {expected!r}")
    if not _require_bool(
        raw["historical_method_choice_used_fold4"],
        label="historical_method_choice_used_fold4",
    ):
        raise ValueError("historical_method_choice_used_fold4 must disclose true")
    if _require_bool(
        raw["organizer_reference_used_for_model_fit"],
        label="organizer_reference_used_for_model_fit",
    ):
        raise ValueError("organizer_reference_used_for_model_fit must be false")

    weights_raw = _require_exact_fields(
        raw["model_weights"], expected=set(_MODEL_WEIGHTS), label="model_weights"
    )
    weights = {
        key: _require_number(weights_raw[key], label=f"model_weights.{key}")
        for key in sorted(_MODEL_WEIGHTS)
    }
    if weights != _MODEL_WEIGHTS:
        raise ValueError(f"model_weights must exactly equal {_MODEL_WEIGHTS!r}")

    targets_raw = raw["target"]
    if type(targets_raw) is not list or len(targets_raw) != 7:
        raise ValueError("target must contain exactly seven ordered tables")
    targets: list[TargetSpec] = []
    for index, item in enumerate(targets_raw):
        target_raw = _require_exact_fields(
            item, expected={"name", "gram"}, label=f"target[{index}]"
        )
        name = _require_name(target_raw["name"], label=f"target[{index}].name")
        gram = target_raw["gram"]
        if type(gram) is not str or gram not in {"positive", "negative"}:
            raise ValueError(f"target[{index}].gram must be positive or negative")
        targets.append(TargetSpec(name=name, gram=cast(GramClass, gram)))
    target_names = tuple(item.name for item in targets)
    if target_names != tuple(sorted(target_names)) or len(set(target_names)) != len(target_names):
        raise ValueError("target names must be unique and strictly alphabetical")
    observed_counts = {
        gram: sum(item.gram == gram for item in targets) for gram in _EXPECTED_TARGET_COUNTS
    }
    if observed_counts != _EXPECTED_TARGET_COUNTS:
        raise ValueError(
            "target panel must contain four Gram-negative and three Gram-positive targets"
        )
    if tuple((item.name, item.gram) for item in targets) != _TARGET_PANEL:
        raise ValueError("target panel must exactly match the frozen seven-target Gate-1 panel")

    descriptor = _parse_descriptor_settings(raw["descriptor_logistic"])
    if descriptor != _FROZEN_DESCRIPTOR_SETTINGS:
        raise ValueError("descriptor_logistic settings differ from the frozen scoring recipe")
    candidate_pool = _parse_candidate_pool_contract(raw["candidate_pool"])
    if candidate_pool != _FROZEN_CANDIDATE_POOL_CONTRACT:
        raise ValueError("candidate_pool contract differs from the accepted frozen publication")
    gate1 = _parse_gate1_contract(raw["gate1"])
    if gate1 != _FROZEN_GATE1_CONTRACT:
        raise ValueError("gate1 contract differs from the accepted frozen publication")

    return CandidateActivityScoringConfig(
        path=snapshot.path,
        snapshot=snapshot,
        schema_version=SCHEMA_VERSION,
        artifact=ARTIFACT,
        status=_CONFIG_STATUS,
        automatic_production_eligible=False,
        expected_candidates=expected_candidates,
        expected_examples=expected_examples,
        folds=folds,
        batch_size=batch_size,
        oof_absolute_tolerance=absolute_tolerance,
        objectives=_OBJECTIVES,
        target_aggregation=_TARGET_AGGREGATION,
        calibration_scope=PROBABILITY_CALIBRATION,
        fit_sensitivity_semantics=_FIT_SENSITIVITY,
        training_scope=TRAINING_SCOPE,
        historical_method_choice_used_fold4=True,
        organizer_reference_used_for_model_fit=False,
        forbidden_claims=_FORBIDDEN_CLAIMS,
        candidate_pool=candidate_pool,
        gate1=gate1,
        descriptor=descriptor,
        model_weights=weights,
        targets=tuple(targets),
    )


def load_candidate_activity_scoring_config(
    path: str | Path,
) -> CandidateActivityScoringConfig:
    """Load the exact preregistered scoring contract from a regular file."""

    return _config_from_snapshot(_snapshot_regular(path, label="scoring config"))


def _resolve_twin_root(path: str | Path, *, job_id: int, label: str) -> Path:
    requested = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(requested, label=label)
    try:
        metadata = os.lstat(requested)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {requested}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symbolic directory")
    if requested.name not in {"0", "1"} or requested.parent.name != str(job_id):
        raise ValueError(f"{label} must be twin 0 or 1 directly below job {job_id}")
    return requested


def _parse_sha256sums(payload: bytes, *, label: str) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-framed checksum text")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"{label} line {number} is not a strict SHA-256 entry")
        digest, name = match.groups()
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name or name in {"", "."}:
            raise ValueError(f"{label} line {number} has an unsafe path")
        if previous is not None and name <= previous:
            raise ValueError(f"{label} paths must be unique and strictly sorted")
        result[name] = digest
        previous = name
    return result


def _authenticate_checksum_tree(
    root: Path,
    *,
    top_snapshot: InputSnapshot,
    label: str,
) -> tuple[Mapping[str, str], Mapping[str, InputSnapshot]]:
    entries = _parse_sha256sums(top_snapshot.payload, label=f"{label} SHA256SUMS")
    snapshots: dict[str, InputSnapshot] = {}
    for relative, expected in entries.items():
        snapshot = _snapshot_regular(root / PurePosixPath(relative), label=f"{label} {relative}")
        _expect_hash(snapshot, expected, label=f"{label} {relative}")
        snapshots[relative] = snapshot
    return entries, snapshots


def _json_object(snapshot: InputSnapshot, *, label: str, canonical: bool) -> Mapping[str, Any]:
    value = _parse_json(snapshot.payload, label=label, canonical=canonical)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _validate_candidate_receipts(
    *,
    root: Path,
    root_snapshots: Mapping[str, InputSnapshot],
    summary_snapshot: InputSnapshot,
    final_snapshot: InputSnapshot,
    config: CandidateActivityScoringConfig,
) -> None:
    summary = _json_object(summary_snapshot, label="candidate validation summary", canonical=True)
    final = _json_object(final_snapshot, label="candidate final publication check", canonical=True)
    if (
        summary.get("schema_version") != 1
        or summary.get("artifact") != "candidate_pool_v1_cpu_validation"
        or summary.get("status") != "passed"
        or summary.get("job_id") != config.candidate_pool.producer_job_id
        or summary.get("git_commit") != config.candidate_pool.git_commit
        or summary.get("candidate_count") != config.expected_candidates
        or summary.get("twins_byte_identical") is not True
        or summary.get("twins_input_bound_verified") is not True
    ):
        raise ValueError("candidate validation summary identity/status is invalid")
    if (
        final.get("schema_version") != 1
        or final.get("artifact") != "candidate_pool_v1_final_publication_check"
        or final.get("status") != "passed"
        or final.get("candidate_count") != config.expected_candidates
        or final.get("input_bound_receipts_match_final_publications") is not True
        or final.get("twins_byte_identical") is not True
        or final.get("twins_final_structure_verified") is not True
        or final.get("validation_summary_sha256") != summary_snapshot.sha256
    ):
        raise ValueError("candidate final publication check identity/status is invalid")

    summary_hashes = summary.get("output_sha256")
    final_artifacts = final.get("output_artifacts")
    if type(summary_hashes) is not dict or type(final_artifacts) is not dict:
        raise ValueError("candidate receipts omit output artifact bindings")
    required = {
        "SHA256SUMS",
        "candidates.fasta",
        "candidates.jsonl",
        "family_metrics.json",
        "manifest.json",
    }
    if set(summary_hashes) != required or set(final_artifacts) != required:
        raise ValueError("candidate receipt artifact inventory is not exact")
    current = {"SHA256SUMS": config.candidate_pool.publication_top_sha256}
    current.update({name: snapshot.sha256 for name, snapshot in root_snapshots.items()})
    for name in sorted(required):
        if summary_hashes.get(name) != current[name]:
            raise ValueError(f"candidate validation summary does not bind {name}")
        record = final_artifacts.get(name)
        if type(record) is not dict or record.get("sha256") != current[name]:
            raise ValueError(f"candidate final publication check does not bind {name}")
        path = root / name
        metadata = os.lstat(path)
        if (
            record.get("size_bytes") != metadata.st_size
            or record.get("link_count") != metadata.st_nlink
            or record.get("mode") != f"{stat.S_IMODE(metadata.st_mode):04o}"
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise ValueError(f"candidate final publication link/mode record differs for {name}")


def _validate_candidate_manifest(
    snapshot: InputSnapshot,
    *,
    config: CandidateActivityScoringConfig,
) -> None:
    manifest = _json_object(snapshot, label="candidate manifest", canonical=False)
    if manifest.get("schema_version") != 1:
        raise ValueError("candidate manifest schema_version must be 1")
    if manifest.get("status") != "candidate_input_only_no_scientific_promotion":
        raise ValueError("candidate manifest status is not an input-only release")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("candidate manifest artifacts must be an object")
    record = artifacts.get("candidates.jsonl")
    if (
        type(record) is not dict
        or record.get("sha256") != config.candidate_pool.candidates_sha256
        or record.get("records") != config.expected_candidates
    ):
        raise ValueError("candidate manifest does not bind the expected candidate JSONL")


def _lineage_sort_key(lineage: Mapping[str, Any]) -> tuple[str, str, str, str, int, int]:
    return (
        cast(str, lineage["generator_family"]),
        cast(str, lineage["generator_variant"]),
        cast(str, lineage["logical_sha256"]),
        cast(str | None, lineage["training_projection_sha256"]) or "",
        cast(int, lineage["seed"]),
        cast(int, lineage["ordinal"]),
    )


def parse_candidate_records(
    payload: bytes, *, expected_candidates: int
) -> tuple[CandidateRecord, ...]:
    """Parse the exact canonical candidate-pool JSONL schema."""

    rows = _parse_canonical_jsonl(payload, label="candidate JSONL")
    if len(rows) != expected_candidates:
        raise ValueError(
            f"candidate JSONL expected {expected_candidates} rows, observed {len(rows)}"
        )
    records: list[CandidateRecord] = []
    previous_sequence: str | None = None
    seen_ids: set[str] = set()
    for source_ordinal, row in enumerate(rows, 1):
        item = _require_exact_fields(
            row, expected=_CANDIDATE_FIELDS, label=f"candidate row {source_ordinal}"
        )
        if type(item["schema_version"]) is not int or item["schema_version"] != 1:
            raise ValueError(f"candidate row {source_ordinal} schema_version must be 1")
        sequence = _parse_sequence(
            item["sequence"], label=f"candidate row {source_ordinal} sequence"
        )
        sequence_id = _require_sha256(
            item["sequence_id"], label=f"candidate row {source_ordinal} sequence_id"
        )
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"candidate row {source_ordinal} sequence_id does not match sequence")
        if type(item["length"]) is not int or item["length"] != len(sequence):
            raise ValueError(f"candidate row {source_ordinal} length does not match sequence")
        if type(item["library_eligible"]) is not bool or item["library_eligible"] is not True:
            raise ValueError(f"candidate row {source_ordinal} must be library eligible")
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ValueError("candidate rows must be unique and ordered by ascending sequence")
        if sequence_id in seen_ids:
            raise ValueError("candidate rows repeat a sequence identity")

        lineages_raw = item["lineages"]
        if type(lineages_raw) is not list or not lineages_raw:
            raise ValueError(f"candidate row {source_ordinal} lineages must be non-empty")
        lineages: list[Mapping[str, Any]] = []
        for lineage_number, lineage_value in enumerate(lineages_raw, 1):
            lineage = _require_exact_fields(
                lineage_value,
                expected=_LINEAGE_FIELDS,
                label=f"candidate row {source_ordinal} lineage {lineage_number}",
            )
            if type(lineage["schema_version"]) is not int or lineage["schema_version"] != 1:
                raise ValueError("candidate lineage schema_version must be 1")
            _require_name(lineage["generator_family"], label="lineage generator_family")
            _require_name(lineage["generator_variant"], label="lineage generator_variant")
            _require_sha256(lineage["logical_sha256"], label="lineage logical_sha256")
            training_digest = lineage["training_projection_sha256"]
            if training_digest is not None:
                _require_sha256(training_digest, label="lineage training_projection_sha256")
            seed = _require_nonnegative_int(lineage["seed"], label="lineage seed")
            ordinal = _require_nonnegative_int(lineage["ordinal"], label="lineage ordinal")
            if seed >= 2**64 or ordinal >= 2**64:
                raise ValueError("candidate lineage seed/ordinal must fit unsigned 64-bit")
            lineages.append(lineage)
        lineage_keys = tuple(_lineage_sort_key(lineage) for lineage in lineages)
        if lineage_keys != tuple(sorted(set(lineage_keys))):
            raise ValueError("candidate lineages must be distinct and canonically ordered")
        records.append(
            CandidateRecord(
                source_ordinal=source_ordinal,
                sequence_id=sequence_id,
                sequence=sequence,
                length=len(sequence),
                library_eligible=True,
                generator_families=tuple(
                    sorted({cast(str, lineage["generator_family"]) for lineage in lineages})
                ),
                generator_variants=tuple(
                    sorted({cast(str, lineage["generator_variant"]) for lineage in lineages})
                ),
            )
        )
        previous_sequence = sequence
        seen_ids.add(sequence_id)
    return tuple(records)


def read_candidate_records(
    candidate_pool_twin_root: str | Path,
    *,
    validation_summary_path: str | Path,
    final_publication_check_path: str | Path,
    config: CandidateActivityScoringConfig,
) -> AuthenticatedCandidates:
    """Authenticate the accepted twin/receipts and parse canonical candidates."""

    root = _resolve_twin_root(
        candidate_pool_twin_root,
        job_id=config.candidate_pool.producer_job_id,
        label="candidate pool twin root",
    )
    top = _snapshot_regular(root / "SHA256SUMS", label="candidate publication SHA256SUMS")
    _expect_hash(
        top,
        config.candidate_pool.publication_top_sha256,
        label="candidate publication SHA256SUMS",
    )
    entries, root_snapshots = _authenticate_checksum_tree(
        root, top_snapshot=top, label="candidate publication"
    )
    expected_entries = {
        "candidates.fasta",
        "candidates.jsonl",
        "family_metrics.json",
        "manifest.json",
    }
    if set(entries) != expected_entries:
        raise ValueError("candidate publication artifact inventory is not exact")
    manifest = root_snapshots["manifest.json"]
    candidates = root_snapshots["candidates.jsonl"]
    _expect_hash(manifest, config.candidate_pool.manifest_sha256, label="candidate manifest")
    _expect_hash(candidates, config.candidate_pool.candidates_sha256, label="candidate JSONL")
    summary = _snapshot_regular(validation_summary_path, label="candidate validation summary")
    final = _snapshot_regular(
        final_publication_check_path, label="candidate final publication check"
    )
    _expect_hash(
        summary,
        config.candidate_pool.validation_summary_sha256,
        label="candidate validation summary",
    )
    _expect_hash(
        final,
        config.candidate_pool.final_publication_check_sha256,
        label="candidate final publication check",
    )
    _validate_candidate_manifest(manifest, config=config)
    all_root_snapshots = dict(root_snapshots)
    all_root_snapshots["SHA256SUMS"] = top
    _validate_candidate_receipts(
        root=root,
        root_snapshots=root_snapshots,
        summary_snapshot=summary,
        final_snapshot=final,
        config=config,
    )
    records = parse_candidate_records(
        candidates.payload, expected_candidates=config.expected_candidates
    )
    snapshots = {
        **{f"candidate_pool/{key}": value for key, value in all_root_snapshots.items()},
        "candidate_pool/validation_summary": summary,
        "candidate_pool/final_publication_check": final,
    }
    return AuthenticatedCandidates(records=records, snapshots=snapshots)


def parse_gate1_examples(
    payload: bytes, *, config: CandidateActivityScoringConfig
) -> tuple[Gate1Example, ...]:
    """Parse and validate the accepted Gate-1 example ledger."""

    rows = _parse_canonical_jsonl(payload, label="Gate-1 examples")
    if len(rows) != config.expected_examples:
        raise ValueError(
            f"Gate-1 examples expected {config.expected_examples} rows, observed {len(rows)}"
        )
    target_grams = {target.name: target.gram for target in config.targets}
    examples: list[Gate1Example] = []
    previous_id: str | None = None
    sequence_metadata: dict[str, tuple[str, int]] = {}
    component_folds: dict[tuple[str, str], int] = {}
    seen_ids: set[str] = set()
    folds_seen: set[int] = set()
    targets_seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        item = _require_exact_fields(
            row, expected=_EXAMPLE_FIELDS, label=f"Gate-1 example row {number}"
        )
        if type(item["schema_version"]) is not int or item["schema_version"] != 1:
            raise ValueError(f"Gate-1 example row {number} schema_version must be 1")
        example_id = _require_sha256(
            item["example_id"], label=f"Gate-1 example row {number} example_id"
        )
        assay_context_id = _require_sha256(
            item["assay_context_id"],
            label=f"Gate-1 example row {number} assay_context_id",
        )
        if example_id != assay_context_id:
            raise ValueError("Gate-1 example_id must exactly equal assay_context_id")
        if example_id in seen_ids or (previous_id is not None and example_id <= previous_id):
            raise ValueError("Gate-1 examples must be unique and strictly ordered by example_id")
        sequence = _parse_sequence(item["sequence"], label=f"Gate-1 example row {number} sequence")
        sequence_id = _require_sha256(
            item["sequence_id"], label=f"Gate-1 example row {number} sequence_id"
        )
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"Gate-1 example row {number} sequence_id does not match sequence")
        target = _require_name(
            item["canonical_target"],
            label=f"Gate-1 example row {number} canonical_target",
        )
        if target not in target_grams:
            raise ValueError(f"Gate-1 example row {number} has a target outside the declared panel")
        gram = item["gram"]
        if type(gram) is not str or gram != target_grams[target]:
            raise ValueError(f"Gate-1 example row {number} Gram class disagrees with target panel")
        if type(item["label"]) is not int or item["label"] not in {0, 1}:
            raise ValueError(f"Gate-1 example row {number} label must be 0 or 1")
        source_observations = _require_positive_int(
            item["source_observations"],
            label=f"Gate-1 example row {number} source_observations",
        )
        fold = _require_nonnegative_int(item["fold"], label=f"Gate-1 example row {number} fold")
        if fold >= config.folds:
            raise ValueError(f"Gate-1 example row {number} fold is outside [0, {config.folds})")
        components: list[str] = []
        for field in ("homology_component_id", "union_component_id"):
            value = item[field]
            if type(value) is not str or _COMPONENT_RE.fullmatch(cast(str, value)) is None:
                raise ValueError(f"Gate-1 example row {number} {field} is invalid")
            component = cast(str, value)
            previous_fold = component_folds.setdefault((field, component), fold)
            if previous_fold != fold:
                raise ValueError(f"Gate-1 {field} crosses outer folds")
            components.append(component)
        prior_sequence = sequence_metadata.setdefault(sequence_id, (sequence, fold))
        if prior_sequence != (sequence, fold):
            raise ValueError("a Gate-1 sequence identity changes sequence or outer fold")
        examples.append(
            Gate1Example(
                example_id=example_id,
                assay_context_id=assay_context_id,
                sequence_id=sequence_id,
                sequence=sequence,
                canonical_target=target,
                gram=cast(GramClass, gram),
                label=cast(int, item["label"]),
                source_observations=source_observations,
                fold=fold,
                homology_component_id=components[0],
                union_component_id=components[1],
            )
        )
        previous_id = example_id
        seen_ids.add(example_id)
        folds_seen.add(fold)
        targets_seen.add(target)
    if folds_seen != set(range(config.folds)):
        raise ValueError("Gate-1 examples do not cover every declared outer fold")
    if targets_seen != set(target_grams):
        raise ValueError("Gate-1 examples do not cover the complete declared target panel")
    labels = {item.label for item in examples}
    if labels != {0, 1}:
        raise ValueError("Gate-1 examples must contain both activity labels")
    for fold in range(config.folds):
        training_labels = {item.label for item in examples if item.fold != fold}
        if training_labels != {0, 1}:
            raise ValueError(f"outer-fold-complement {fold} must contain both activity labels")
    return tuple(examples)


def _parse_csv_int(value: str | None, *, label: str, positive: bool = False) -> int:
    if value is None or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise ValueError(f"{label} must be a canonical non-negative integer")
    result = int(value)
    if positive and result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _parse_csv_float(value: str | None, *, label: str) -> float:
    if value is None or not value or value.strip() != value:
        raise ValueError(f"{label} must be a finite decimal")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite decimal") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def parse_descriptor_oof(
    payload: bytes,
    *,
    examples: Sequence[Gate1Example],
) -> Mapping[str, float]:
    """Validate Gate-1 OOF CSV metadata and return descriptor probabilities."""

    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("Gate-1 OOF CSV must use LF framing with exactly one final LF")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Gate-1 OOF CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(_OOF_FIELDS):
        raise ValueError("Gate-1 OOF CSV header does not match the accepted schema")
    example_by_id = {item.example_id: item for item in examples}
    seen: set[tuple[str, str]] = set()
    models_by_example: dict[str, set[str]] = {item.example_id: set() for item in examples}
    descriptor: dict[str, float] = {}
    previous: tuple[str, str] | None = None
    row_count = 0
    accepted_models = {"descriptor_logistic", "equal_weight_ensemble", "homology_knn"}
    for number, row in enumerate(reader, 2):
        row_count += 1
        if set(row) != set(_OOF_FIELDS) or any(value is None for value in row.values()):
            raise ValueError(f"Gate-1 OOF row {number} has an invalid field count")
        model = row["model"]
        example_id = row["example_id"]
        if model not in accepted_models:
            raise ValueError(f"Gate-1 OOF row {number} has an unknown accepted model")
        if example_id not in example_by_id:
            raise ValueError(f"Gate-1 OOF row {number} references an unknown example")
        key = (model, example_id)
        if key in seen or (previous is not None and key <= previous):
            raise ValueError("Gate-1 OOF rows must be unique and ordered by model/example_id")
        item = example_by_id[example_id]
        expected_text = {
            "assay_context_id": item.assay_context_id,
            "sequence_id": item.sequence_id,
            "sequence": item.sequence,
            "canonical_target": item.canonical_target,
            "gram": item.gram,
        }
        if any(row[field] != value for field, value in expected_text.items()):
            raise ValueError(f"Gate-1 OOF row {number} metadata differs from examples JSONL")
        if _parse_csv_int(row["label"], label=f"Gate-1 OOF row {number} label") != item.label:
            raise ValueError(f"Gate-1 OOF row {number} label differs from examples JSONL")
        if (
            _parse_csv_int(
                row["source_observations"],
                label=f"Gate-1 OOF row {number} source_observations",
                positive=True,
            )
            != item.source_observations
            or _parse_csv_int(row["fold"], label=f"Gate-1 OOF row {number} fold") != item.fold
            or row["homology_component_id"] != item.homology_component_id
            or row["union_component_id"] != item.union_component_id
        ):
            raise ValueError(f"Gate-1 OOF row {number} split metadata differs from examples JSONL")
        maximum_identity = _parse_csv_float(
            row["max_train_identity"], label=f"Gate-1 OOF row {number} max_train_identity"
        )
        probability = _parse_csv_float(
            row["probability"], label=f"Gate-1 OOF row {number} probability"
        )
        if not 0.0 <= maximum_identity <= 1.0 or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Gate-1 OOF row {number} has a value outside [0, 1]")
        if model == ACCEPTED_MODEL:
            descriptor[example_id] = probability
        models_by_example[example_id].add(model)
        seen.add(key)
        previous = key
    expected_rows = len(examples) * len(accepted_models)
    if row_count != expected_rows:
        raise ValueError(f"Gate-1 OOF CSV expected {expected_rows} rows, observed {row_count}")
    if any(models != accepted_models for models in models_by_example.values()):
        raise ValueError("Gate-1 OOF CSV does not have exactly three models per example")
    if len(descriptor) != len(examples):
        raise ValueError("Gate-1 OOF CSV does not have one descriptor probability per example")
    return dict(sorted(descriptor.items()))


def _validate_gate1_manifest(
    snapshot: InputSnapshot,
    *,
    semantic_entries: Mapping[str, str],
    config: CandidateActivityScoringConfig,
) -> None:
    manifest = _json_object(snapshot, label="Gate-1 manifest", canonical=False)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact") != "gate1_context_activity_homology_study_union_v1"
        or manifest.get("status") != "development_evidence_not_an_untouched_evaluation_panel"
        or manifest.get("git_commit") != config.gate1.git_commit
    ):
        raise ValueError("Gate-1 manifest identity/status is invalid")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("Gate-1 manifest artifacts must be an object")
    for key, filename in (("examples", "examples.jsonl"), ("oof", "oof_predictions.csv")):
        record = artifacts.get(key)
        if (
            type(record) is not dict
            or record.get("filename") != filename
            or record.get("sha256") != semantic_entries[filename]
        ):
            raise ValueError(f"Gate-1 manifest does not bind {filename}")


def _validate_gate1_receipt(
    snapshot: InputSnapshot,
    *,
    config: CandidateActivityScoringConfig,
) -> None:
    receipt = _json_object(snapshot, label="Gate-1 independent receipt", canonical=False)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact")
        != "gate1_context_activity_homology_study_union_v1_independent_verification"
        or receipt.get("status") != "passed"
        or receipt.get("git_commit") != config.gate1.git_commit
        or receipt.get("gate1_top_manifest_sha256") != config.gate1.semantic_top_sha256
        or receipt.get("publication_top_manifest_sha256") != config.gate1.publication_top_sha256
    ):
        raise ValueError("Gate-1 independent receipt identity/status is invalid")
    artifact_hashes = receipt.get("artifact_sha256")
    if (
        type(artifact_hashes) is not dict
        or artifact_hashes.get("examples.jsonl") != config.gate1.examples_sha256
        or artifact_hashes.get("oof_predictions.csv") != config.gate1.oof_sha256
    ):
        raise ValueError("Gate-1 independent receipt artifact hashes are invalid")
    checks = receipt.get("checks")
    if (
        type(checks) is not dict
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("Gate-1 independent receipt checks did not all pass")
    handshake = receipt.get("production_handshake")
    if type(handshake) is not dict:
        raise ValueError("Gate-1 independent receipt omits the production handshake")
    if (
        handshake.get("bidirectional_acknowledgement") is not True
        or handshake.get("distinct_nodes") is not True
    ):
        raise ValueError("Gate-1 production handshake status is invalid")
    for field in ("acknowledgement_sha256", "receipt_sha256"):
        values = handshake.get(field)
        if type(values) is not dict or set(values) != {"0", "1"}:
            raise ValueError(f"Gate-1 production handshake {field} is invalid")
        for key, value in values.items():
            _require_sha256(value, label=f"Gate-1 production handshake {field}.{key}")


def read_gate1_evidence(
    gate1_twin_root: str | Path,
    *,
    independent_receipt_path: str | Path,
    config: CandidateActivityScoringConfig,
) -> AuthenticatedGate1:
    """Authenticate the accepted Gate-1 twin and independent receipt."""

    root = _resolve_twin_root(
        gate1_twin_root,
        job_id=config.gate1.producer_job_id,
        label="Gate-1 twin root",
    )
    receipt_requested = Path(os.path.abspath(os.fspath(independent_receipt_path)))
    if (
        receipt_requested.name != f"independent-verification-{config.gate1.audit_job_id}.json"
        or receipt_requested.parent.name != str(config.gate1.producer_job_id)
    ):
        raise ValueError("Gate-1 independent receipt path does not match producer/audit job IDs")
    top = _snapshot_regular(root / "SHA256SUMS", label="Gate-1 publication SHA256SUMS")
    _expect_hash(top, config.gate1.publication_top_sha256, label="Gate-1 publication SHA256SUMS")
    top_entries, top_snapshots = _authenticate_checksum_tree(
        root, top_snapshot=top, label="Gate-1 publication"
    )
    semantic = top_snapshots.get("gate1/SHA256SUMS")
    if semantic is None or semantic.sha256 != config.gate1.semantic_top_sha256:
        raise ValueError("Gate-1 publication does not bind the configured semantic manifest")
    semantic_entries, semantic_snapshots = _authenticate_checksum_tree(
        root / "gate1", top_snapshot=semantic, label="Gate-1 semantic publication"
    )
    expected_semantic = {
        "context_audit.jsonl",
        "examples.jsonl",
        "folds.json",
        "manifest.json",
        "metrics.json",
        "oof_predictions.csv",
        "split_receipt.json",
    }
    if set(semantic_entries) != expected_semantic:
        raise ValueError("Gate-1 semantic artifact inventory is not exact")
    for name, digest in semantic_entries.items():
        if top_entries.get(f"gate1/{name}") != digest:
            raise ValueError(f"Gate-1 top publication does not bind gate1/{name}")
    examples_snapshot = semantic_snapshots["examples.jsonl"]
    oof_snapshot = semantic_snapshots["oof_predictions.csv"]
    _expect_hash(examples_snapshot, config.gate1.examples_sha256, label="Gate-1 examples")
    _expect_hash(oof_snapshot, config.gate1.oof_sha256, label="Gate-1 OOF predictions")
    receipt = _snapshot_regular(receipt_requested, label="Gate-1 independent receipt")
    _expect_hash(
        receipt,
        config.gate1.independent_receipt_sha256,
        label="Gate-1 independent receipt",
    )
    _validate_gate1_manifest(
        semantic_snapshots["manifest.json"], semantic_entries=semantic_entries, config=config
    )
    _validate_gate1_receipt(receipt, config=config)
    examples = parse_gate1_examples(examples_snapshot.payload, config=config)
    descriptor_oof = parse_descriptor_oof(oof_snapshot.payload, examples=examples)
    snapshots = {
        **{f"gate1/{key}": value for key, value in top_snapshots.items()},
        "gate1/SHA256SUMS": top,
        "gate1/semantic_SHA256SUMS": semantic,
        "gate1/independent_receipt": receipt,
    }
    return AuthenticatedGate1(
        examples=examples,
        descriptor_oof=descriptor_oof,
        snapshots=snapshots,
    )


def _new_descriptor_model(settings: DescriptorSettings) -> DescriptorLogisticOracle:
    return DescriptorLogisticOracle(
        l2=settings.l2,
        max_iterations=settings.max_iterations,
        tolerance=settings.tolerance,
        prior_strength=settings.prior_strength,
    )


def _audit_descriptor_fit(
    model: DescriptorLogisticOracle,
    training_inputs: Sequence[OracleInput],
    labels: FloatArray,
    *,
    label: str,
) -> Mapping[str, object]:
    fitted = model._coefficient
    if fitted is None or not np.all(np.isfinite(fitted)):
        raise ValueError(f"{label} has missing or non-finite descriptor coefficients")
    design = model._design_matrix(tuple(training_inputs))
    if not np.all(np.isfinite(design)):
        raise ValueError(f"{label} has non-finite descriptor features")
    prior = float(
        (np.sum(labels) + 0.5 * model.prior_strength) / (labels.size + model.prior_strength)
    )
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(coefficient.size, model.l2, dtype=np.float64)
    penalty[0] = 0.0
    converged = False
    iterations = 0
    for iteration in range(1, model.max_iterations + 1):
        logits = design @ coefficient
        probability = np.empty_like(logits)
        positive = logits >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        probability[~positive] = exponent / (1.0 + exponent)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - labels) / labels.size + penalty * coefficient
        hessian = (design.T * variance) @ design / labels.size
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError(f"{label} has a non-finite convergence step")
        coefficient -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficient)):
            raise ValueError(f"{label} has non-finite coefficients")
        if float(np.max(np.abs(step))) <= model.tolerance:
            converged = True
            break
    if not converged:
        raise ValueError(f"{label} exhausted the descriptor iteration budget")
    if not np.array_equal(coefficient, fitted):
        raise ValueError(f"{label} deterministic convergence replay differs from fitted state")
    return {
        "coefficient_count": int(coefficient.size),
        "converged": True,
        "iterations": iterations,
        "parameters_finite": True,
    }


def _training_example_ids_sha256(examples: Sequence[Gate1Example]) -> str:
    identifiers = tuple(sorted(item.example_id for item in examples))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("training example identifiers must be unique")
    payload = "".join(f"{identifier}\n" for identifier in identifiers).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def export_descriptor_model_state(
    model: DescriptorLogisticOracle,
    *,
    name: str,
    heldout_fold: int | None,
    training_examples: Sequence[Gate1Example],
) -> Mapping[str, object]:
    """Export one fitted model using the frozen, path-free state contract."""

    if type(model) is not DescriptorLogisticOracle:
        raise TypeError("model must be exactly DescriptorLogisticOracle")
    if _MEMBER_RE.fullmatch(name) is None:
        raise ValueError("model state name is not a frozen member name")
    if name == "all_data_deployment":
        if heldout_fold is not None:
            raise ValueError("all_data_deployment heldout_fold must be null")
    elif type(heldout_fold) is not int or heldout_fold not in range(5):
        raise ValueError("outer model heldout_fold must be an integer in [0, 5)")
    if not training_examples:
        raise ValueError("model state training examples cannot be empty")
    strains = model._strains
    mean = model._mean
    scale = model._scale
    coefficient = model._coefficient
    constant = model._constant_probability
    if strains is None or mean is None or scale is None:
        raise ValueError("cannot export an unfitted descriptor model")
    if strains != tuple(sorted(set(strains))):
        raise ValueError("fitted descriptor strains are not canonical")
    expected_strains = tuple(sorted({item.canonical_target for item in training_examples}))
    if strains != expected_strains:
        raise ValueError("fitted descriptor strains differ from training examples")
    if mean.shape != (len(_SEQUENCE_FEATURE_ORDER),) or scale.shape != mean.shape:
        raise ValueError("descriptor standardization state has an invalid shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("descriptor standardization state is invalid")
    if (coefficient is None) == (constant is None):
        raise ValueError("descriptor state must contain exactly one fitted prediction form")
    if coefficient is not None:
        expected_coefficients = 1 + len(_SEQUENCE_FEATURE_ORDER) + len(strains) + 1 + 3
        if coefficient.shape != (expected_coefficients,) or not np.all(np.isfinite(coefficient)):
            raise ValueError("descriptor coefficient state is invalid")
        coefficient_value: list[float] | None = [float(value) for value in coefficient]
        constant_value: float | None = None
    else:
        assert constant is not None
        if not math.isfinite(constant) or not 0.0 <= constant <= 1.0:
            raise ValueError("descriptor constant probability is invalid")
        coefficient_value = None
        constant_value = float(constant)
    return {
        "name": name,
        "heldout_fold": heldout_fold,
        "training_examples": len(training_examples),
        "training_sequences": len({item.sequence_id for item in training_examples}),
        "training_union_components": len({item.union_component_id for item in training_examples}),
        "training_example_ids_sha256": _training_example_ids_sha256(training_examples),
        "strains": list(strains),
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "coefficient": coefficient_value,
        "constant_probability": constant_value,
    }


def _finite_float_list(value: object, *, label: str) -> FloatArray:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a non-empty array")
    numbers = np.asarray(
        [_require_number(item, label=f"{label}[{index}]") for index, item in enumerate(value)],
        dtype=np.float64,
    )
    if numbers.ndim != 1 or not np.all(np.isfinite(numbers)):
        raise ValueError(f"{label} must be a finite one-dimensional array")
    return numbers


def import_descriptor_model_state(
    state: object,
    *,
    settings: DescriptorSettings,
) -> DescriptorLogisticOracle:
    """Reconstitute one exported descriptor state after strict validation."""

    raw = _require_exact_fields(
        state,
        expected={
            "name",
            "heldout_fold",
            "training_examples",
            "training_sequences",
            "training_union_components",
            "training_example_ids_sha256",
            "strains",
            "mean",
            "scale",
            "coefficient",
            "constant_probability",
        },
        label="descriptor model state",
    )
    name = raw["name"]
    if type(name) is not str or _MEMBER_RE.fullmatch(cast(str, name)) is None:
        raise ValueError("descriptor model state name is invalid")
    heldout = raw["heldout_fold"]
    if name == "all_data_deployment":
        if heldout is not None:
            raise ValueError("all_data_deployment heldout_fold must be null")
    elif type(heldout) is not int or heldout not in range(5) or name != f"outer_fold_{heldout}":
        raise ValueError("outer descriptor state name/heldout_fold disagree")
    for field in ("training_examples", "training_sequences", "training_union_components"):
        _require_positive_int(raw[field], label=f"descriptor state {field}")
    _require_sha256(
        raw["training_example_ids_sha256"],
        label="descriptor state training_example_ids_sha256",
    )
    strains_raw = raw["strains"]
    if type(strains_raw) is not list or not strains_raw:
        raise ValueError("descriptor state strains must be a non-empty array")
    strains = tuple(_require_name(value, label="descriptor state strain") for value in strains_raw)
    if strains != tuple(sorted(set(strains))):
        raise ValueError("descriptor state strains must be unique and sorted")
    mean = _finite_float_list(raw["mean"], label="descriptor state mean")
    scale = _finite_float_list(raw["scale"], label="descriptor state scale")
    if mean.shape != (len(_SEQUENCE_FEATURE_ORDER),) or scale.shape != mean.shape:
        raise ValueError("descriptor state mean/scale shape is invalid")
    if np.any(scale <= 0.0):
        raise ValueError("descriptor state scales must be positive")

    coefficient_raw = raw["coefficient"]
    constant_raw = raw["constant_probability"]
    if (coefficient_raw is None) == (constant_raw is None):
        raise ValueError("descriptor state must provide coefficient xor constant_probability")
    coefficient: FloatArray | None = None
    constant: float | None = None
    if coefficient_raw is not None:
        coefficient = _finite_float_list(coefficient_raw, label="descriptor state coefficient")
        expected = 1 + len(_SEQUENCE_FEATURE_ORDER) + len(strains) + 1 + 3
        if coefficient.shape != (expected,):
            raise ValueError("descriptor state coefficient shape is invalid")
    else:
        constant = _require_number(constant_raw, label="descriptor state constant_probability")
        if not 0.0 <= constant <= 1.0:
            raise ValueError("descriptor state constant_probability must be in [0, 1]")

    model = _new_descriptor_model(settings)
    model._strains = strains
    model._mean = mean.copy()
    model._scale = scale.copy()
    model._coefficient = None if coefficient is None else coefficient.copy()
    model._constant_probability = constant
    return model


def descriptor_model_states_document(
    states: Sequence[Mapping[str, object]],
    *,
    config: CandidateActivityScoringConfig,
) -> Mapping[str, object]:
    """Build the exact six-state serialized model document."""

    expected_names = (
        *(f"outer_fold_{fold}" for fold in range(5)),
        "all_data_deployment",
    )
    if tuple(state.get("name") for state in states) != expected_names:
        raise ValueError("descriptor states are not in frozen member order")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "candidate_activity_descriptor_model_states_v1",
        "descriptor_settings": config.descriptor.as_dict(),
        "target_order": [target.name for target in config.targets],
        "feature_order": list(_SEQUENCE_FEATURE_ORDER),
        "states": list(states),
    }


def import_descriptor_model_states(
    document: object,
) -> tuple[DescriptorLogisticOracle, ...]:
    """Strictly import a complete model-states document in member order."""

    raw = _require_exact_fields(
        document,
        expected={
            "schema_version",
            "artifact",
            "descriptor_settings",
            "target_order",
            "feature_order",
            "states",
        },
        label="model states document",
    )
    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise ValueError("model states schema_version must be 1")
    if raw["artifact"] != "candidate_activity_descriptor_model_states_v1":
        raise ValueError("model states artifact identity is invalid")
    settings = _parse_descriptor_settings(raw["descriptor_settings"])
    if raw["feature_order"] != list(_SEQUENCE_FEATURE_ORDER):
        raise ValueError("model states feature order is invalid")
    targets = raw["target_order"]
    if type(targets) is not list or len(targets) != 7:
        raise ValueError("model states target order must contain seven targets")
    target_names = tuple(_require_name(value, label="model states target") for value in targets)
    if target_names != tuple(sorted(set(target_names))):
        raise ValueError("model states targets must be unique and alphabetical")
    states = raw["states"]
    if type(states) is not list or len(states) != 6:
        raise ValueError("model states must contain five outer members and one deployment member")
    expected_names = (
        *(f"outer_fold_{fold}" for fold in range(5)),
        "all_data_deployment",
    )
    if (
        tuple(state.get("name") if type(state) is dict else None for state in states)
        != expected_names
    ):
        raise ValueError("model states are not in frozen member order")
    return tuple(import_descriptor_model_state(state, settings=settings) for state in states)


def fit_activity_models(
    examples: Sequence[Gate1Example],
    accepted_descriptor_oof: Mapping[str, float],
    *,
    config: CandidateActivityScoringConfig,
) -> FittedActivityModels:
    """Reproduce accepted OOF values and fit frozen scoring members."""

    items = tuple(sorted(examples, key=lambda item: item.example_id))
    if len(items) != config.expected_examples or len({item.example_id for item in items}) != len(
        items
    ):
        raise ValueError("Gate-1 fit examples do not match the configured count/identity contract")
    if set(accepted_descriptor_oof) != {item.example_id for item in items}:
        raise ValueError("accepted descriptor OOF identities do not exactly match Gate-1 examples")
    fold_models: list[DescriptorLogisticOracle] = []
    states: list[Mapping[str, object]] = []
    by_fold: list[Mapping[str, object]] = []
    all_errors: list[float] = []
    for fold in range(config.folds):
        training = tuple(item for item in items if item.fold != fold)
        heldout = tuple(item for item in items if item.fold == fold)
        if not training or not heldout:
            raise ValueError(f"outer fold {fold} has an empty train or held-out partition")
        labels = np.asarray([item.label for item in training], dtype=np.float64)
        if set(labels.tolist()) != {0.0, 1.0}:
            raise ValueError(f"outer-fold-complement {fold} does not contain both labels")
        model = _new_descriptor_model(config.descriptor)
        training_inputs = tuple(item.model_input for item in training)
        model.fit(training_inputs, labels.astype(np.int64))
        _audit_descriptor_fit(
            model,
            training_inputs,
            labels,
            label=f"outer-fold-complement {fold}",
        )
        reproduced = model.predict_proba(tuple(item.model_input for item in heldout))
        if reproduced.shape != (len(heldout),) or not np.all(np.isfinite(reproduced)):
            raise ValueError(f"outer fold {fold} emitted invalid OOF probabilities")
        errors = np.asarray(
            [
                abs(float(probability) - accepted_descriptor_oof[item.example_id])
                for item, probability in zip(heldout, reproduced, strict=True)
            ],
            dtype=np.float64,
        )
        maximum = float(np.max(errors))
        if maximum > config.oof_absolute_tolerance:
            raise ValueError(
                f"outer fold {fold} descriptor OOF mismatch: {maximum:.17g} exceeds "
                f"{config.oof_absolute_tolerance:.17g}"
            )
        all_errors.extend(float(value) for value in errors)
        fold_models.append(model)
        states.append(
            export_descriptor_model_state(
                model,
                name=f"outer_fold_{fold}",
                heldout_fold=fold,
                training_examples=training,
            )
        )
        by_fold.append(
            {
                "fold": fold,
                "heldout_examples": len(heldout),
                "training_examples": len(training),
                "maximum_absolute_error": maximum,
            }
        )

    all_labels = np.asarray([item.label for item in items], dtype=np.float64)
    deployment_model = _new_descriptor_model(config.descriptor)
    all_inputs = tuple(item.model_input for item in items)
    deployment_model.fit(all_inputs, all_labels.astype(np.int64))
    _audit_descriptor_fit(
        deployment_model,
        all_inputs,
        all_labels,
        label="all-data deployment fit",
    )
    states.append(
        export_descriptor_model_state(
            deployment_model,
            name="all_data_deployment",
            heldout_fold=None,
            training_examples=items,
        )
    )
    maximum_error = max(all_errors)
    oof_reproduction = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "candidate_activity_descriptor_oof_reproduction_v1",
        "status": "passed_exact_descriptor_oof_reproduction",
        "accepted_model": ACCEPTED_MODEL,
        "examples": len(items),
        "folds": config.folds,
        "absolute_tolerance": config.oof_absolute_tolerance,
        "maximum_absolute_error": maximum_error,
        "by_fold": by_fold,
    }
    return FittedActivityModels(
        fold_models=tuple(fold_models),
        deployment_model=deployment_model,
        model_states=tuple(states),
        oof_reproduction=oof_reproduction,
    )


def _validate_probability_array(
    values: FloatArray,
    *,
    expected_shape: tuple[int, ...],
    label: str,
) -> None:
    if values.shape != expected_shape:
        raise ValueError(f"{label} shape {values.shape!r} differs from {expected_shape!r}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{label} must contain only finite values in [0, 1]")


def _predict_descriptor_probabilities_row_stable(
    model: DescriptorLogisticOracle,
    rows: Sequence[OracleInput],
) -> FloatArray:
    """Predict with a fixed feature-axis reduction independent of row count."""

    items = tuple(rows)
    if not items:
        return np.empty(0, dtype=np.float64)
    if model._strains is None or model._mean is None or model._scale is None:
        raise ValueError("descriptor model must be fitted before candidate inference")
    if model._constant_probability is not None:
        return np.full(len(items), model._constant_probability, dtype=np.float64)
    coefficient = model._coefficient
    if coefficient is None:
        raise ValueError("descriptor model has no fitted prediction state")
    design = model._design_matrix(items)
    # Matrix-vector BLAS kernels may change their accumulation path with the
    # number of rows.  Multiplication followed by a reduction along the fixed
    # feature axis gives every candidate the same operation order regardless
    # of the outer scoring chunk boundary.
    logits = np.sum(design * coefficient[np.newaxis, :], axis=1, dtype=np.float64)
    probability = np.empty_like(logits)
    positive = logits >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    probability[~positive] = exponent / (1.0 + exponent)
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def score_candidate_batch(
    candidates: Sequence[CandidateRecord],
    *,
    models: FittedActivityModels,
    targets: Sequence[TargetSpec],
) -> ActivityScoreBatch:
    """Score one canonical candidate batch with the six frozen model fits."""

    if not candidates:
        raise ValueError("candidate scoring batch cannot be empty")
    if len(models.fold_models) != 5:
        raise ValueError("candidate scoring requires exactly five fold-complement models")
    if len(targets) != 7:
        raise ValueError("candidate scoring requires exactly seven ordered targets")
    rows = tuple(
        OracleInput(sequence=candidate.sequence, strain=target.name, gram=target.gram)
        for candidate in candidates
        for target in targets
    )
    candidate_count = len(candidates)
    target_count = len(targets)
    deployment = _predict_descriptor_probabilities_row_stable(
        models.deployment_model,
        rows,
    ).reshape(candidate_count, target_count)
    fold_values = np.stack(
        [
            _predict_descriptor_probabilities_row_stable(model, rows).reshape(
                candidate_count,
                target_count,
            )
            for model in models.fold_models
        ],
        axis=1,
    )
    deployment = np.ascontiguousarray(deployment, dtype=np.float64)
    fold_values = np.ascontiguousarray(fold_values, dtype=np.dtype("<f8"))
    _validate_probability_array(
        deployment,
        expected_shape=(candidate_count, target_count),
        label="deployment target probabilities",
    )
    _validate_probability_array(
        fold_values,
        expected_shape=(candidate_count, 5, target_count),
        label="fold-model target probabilities",
    )

    gram_indices = {
        gram: np.asarray(
            [index for index, target in enumerate(targets) if target.gram == gram],
            dtype=np.int64,
        )
        for gram in ("positive", "negative")
    }
    if any(values.size == 0 for values in gram_indices.values()):
        raise ValueError("target panel must represent both Gram classes")
    means = {
        "broad_spectrum_activity": np.mean(deployment, axis=1),
        "gram_positive_activity": np.mean(deployment[:, gram_indices["positive"]], axis=1),
        "gram_negative_activity": np.mean(deployment[:, gram_indices["negative"]], axis=1),
    }
    fold_means = {
        "broad_spectrum_activity": np.mean(fold_values, axis=2),
        "gram_positive_activity": np.mean(fold_values[:, :, gram_indices["positive"]], axis=2),
        "gram_negative_activity": np.mean(fold_values[:, :, gram_indices["negative"]], axis=2),
    }
    sensitivity = {key: np.std(value, axis=1, ddof=0) for key, value in fold_means.items()}
    target_sensitivity = np.std(fold_values, axis=1, ddof=0)
    for key, value in means.items():
        _validate_probability_array(
            value,
            expected_shape=(candidate_count,),
            label=f"mean {key}",
        )
    for key, value in sensitivity.items():
        if value.shape != (candidate_count,) or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"model_fit_sensitivity {key} is invalid")
    if (
        target_sensitivity.shape != (candidate_count, target_count)
        or not np.all(np.isfinite(target_sensitivity))
        or np.any(target_sensitivity < 0)
    ):
        raise ValueError("per-target model_fit_sensitivity is invalid")
    return ActivityScoreBatch(
        deployment_target_probabilities=deployment,
        fold_model_target_probabilities=fold_values,
        model_fit_sensitivity_target=target_sensitivity,
        means=means,
        model_fit_sensitivity=sensitivity,
    )


def score_candidate_records(
    candidates: Sequence[CandidateRecord],
    *,
    models: FittedActivityModels,
    config: CandidateActivityScoringConfig,
) -> ActivityScoreBatch:
    """Score every candidate in bounded batches while retaining canonical axes."""

    if not candidates:
        raise ValueError("candidate collection cannot be empty")
    sequences = tuple(item.sequence for item in candidates)
    ordinals = tuple(item.source_ordinal for item in candidates)
    if sequences != tuple(sorted(set(sequences))):
        raise ValueError("candidate collection is not in unique canonical sequence order")
    if ordinals != tuple(range(1, len(candidates) + 1)):
        raise ValueError("candidate source ordinals must be exact consecutive one-based values")
    deployment_parts: list[FloatArray] = []
    fold_parts: list[FloatArray] = []
    target_sensitivity_parts: list[FloatArray] = []
    mean_parts: dict[str, list[FloatArray]] = {key: [] for key in _OBJECTIVES}
    sensitivity_parts: dict[str, list[FloatArray]] = {key: [] for key in _OBJECTIVES}
    for start in range(0, len(candidates), config.batch_size):
        batch = score_candidate_batch(
            candidates[start : start + config.batch_size],
            models=models,
            targets=config.targets,
        )
        deployment_parts.append(batch.deployment_target_probabilities)
        fold_parts.append(batch.fold_model_target_probabilities)
        target_sensitivity_parts.append(batch.model_fit_sensitivity_target)
        for key in _OBJECTIVES:
            mean_parts[key].append(batch.means[key])
            sensitivity_parts[key].append(batch.model_fit_sensitivity[key])
    deployment = np.ascontiguousarray(np.concatenate(deployment_parts, axis=0), dtype=np.float64)
    fold_values = np.ascontiguousarray(np.concatenate(fold_parts, axis=0), dtype=np.dtype("<f8"))
    target_sensitivity = np.ascontiguousarray(
        np.concatenate(target_sensitivity_parts, axis=0), dtype=np.float64
    )
    means = {key: np.concatenate(value, axis=0) for key, value in mean_parts.items()}
    sensitivity = {key: np.concatenate(value, axis=0) for key, value in sensitivity_parts.items()}
    expected_fold_shape = (len(candidates), 5, len(config.targets))
    _validate_probability_array(
        deployment,
        expected_shape=(len(candidates), len(config.targets)),
        label="complete deployment target probabilities",
    )
    _validate_probability_array(
        fold_values,
        expected_shape=expected_fold_shape,
        label="complete fold-model target probabilities",
    )
    return ActivityScoreBatch(
        deployment_target_probabilities=deployment,
        fold_model_target_probabilities=fold_values,
        model_fit_sensitivity_target=target_sensitivity,
        means=means,
        model_fit_sensitivity=sensitivity,
    )


def _validate_publication_candidates(
    candidates: AuthenticatedCandidates,
    *,
    config: CandidateActivityScoringConfig,
) -> tuple[CandidateRecord, ...]:
    records = candidates.records
    if type(records) is not tuple or len(records) != config.expected_candidates:
        raise ValueError("candidate records do not match the configured exact count")
    previous_sequence: str | None = None
    seen_sequence_ids: set[str] = set()
    for ordinal, candidate in enumerate(records, 1):
        if type(candidate) is not CandidateRecord:
            raise ValueError(f"candidate record {ordinal} has an invalid record type")
        if type(candidate.source_ordinal) is not int or candidate.source_ordinal != ordinal:
            raise ValueError("candidate source ordinals must be exact consecutive one-based values")
        sequence = _parse_sequence(
            candidate.sequence,
            label=f"candidate record {ordinal} sequence",
        )
        sequence_id = _require_sha256(
            candidate.sequence_id,
            label=f"candidate record {ordinal} sequence_id",
        )
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"candidate record {ordinal} sequence identity is invalid")
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ValueError("candidate records must be unique and in canonical sequence order")
        if sequence_id in seen_sequence_ids:
            raise ValueError("candidate records repeat a canonical sequence identity")
        if type(candidate.length) is not int or candidate.length != len(sequence):
            raise ValueError(f"candidate record {ordinal} length differs from its sequence")
        if type(candidate.library_eligible) is not bool or candidate.library_eligible is not True:
            raise ValueError(f"candidate record {ordinal} is not library eligible")
        for field, values in (
            ("generator_families", candidate.generator_families),
            ("generator_variants", candidate.generator_variants),
        ):
            if type(values) is not tuple or not values:
                raise ValueError(f"candidate record {ordinal} {field} must be a non-empty tuple")
            parsed = tuple(
                _require_name(value, label=f"candidate record {ordinal} {field}")
                for value in values
            )
            if parsed != tuple(sorted(set(parsed))):
                raise ValueError(
                    f"candidate record {ordinal} {field} must be unique and canonically ordered"
                )
        previous_sequence = sequence
        seen_sequence_ids.add(sequence_id)
    return records


def _validate_publication_gate1(
    gate1: AuthenticatedGate1,
    *,
    config: CandidateActivityScoringConfig,
) -> tuple[Gate1Example, ...]:
    examples = gate1.examples
    if type(examples) is not tuple or len(examples) != config.expected_examples:
        raise ValueError("Gate-1 examples do not match the configured exact count")
    target_grams = {target.name: target.gram for target in config.targets}
    previous_example_id: str | None = None
    seen_ids: set[str] = set()
    folds_seen: set[int] = set()
    targets_seen: set[str] = set()
    sequence_metadata: dict[str, tuple[str, int]] = {}
    component_folds: dict[tuple[str, str], int] = {}
    for number, example in enumerate(examples, 1):
        if type(example) is not Gate1Example:
            raise ValueError(f"Gate-1 example {number} has an invalid record type")
        example_id = _require_sha256(
            example.example_id,
            label=f"Gate-1 example {number} example_id",
        )
        assay_context_id = _require_sha256(
            example.assay_context_id,
            label=f"Gate-1 example {number} assay_context_id",
        )
        if example_id != assay_context_id:
            raise ValueError("Gate-1 example_id must exactly equal assay_context_id")
        if example_id in seen_ids or (
            previous_example_id is not None and example_id <= previous_example_id
        ):
            raise ValueError("Gate-1 examples must be unique and ordered by example_id")
        sequence = _parse_sequence(
            example.sequence,
            label=f"Gate-1 example {number} sequence",
        )
        sequence_id = _require_sha256(
            example.sequence_id,
            label=f"Gate-1 example {number} sequence_id",
        )
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"Gate-1 example {number} sequence identity is invalid")
        target = _require_name(
            example.canonical_target,
            label=f"Gate-1 example {number} canonical_target",
        )
        if target not in target_grams or example.gram != target_grams[target]:
            raise ValueError(f"Gate-1 example {number} target/Gram identity is invalid")
        if type(example.label) is not int or example.label not in {0, 1}:
            raise ValueError(f"Gate-1 example {number} label must be 0 or 1")
        if type(example.source_observations) is not int or example.source_observations <= 0:
            raise ValueError(f"Gate-1 example {number} source_observations must be positive")
        if type(example.fold) is not int or example.fold not in range(config.folds):
            raise ValueError(f"Gate-1 example {number} fold is outside the configured range")
        for field in ("homology_component_id", "union_component_id"):
            component = getattr(example, field)
            if type(component) is not str or _COMPONENT_RE.fullmatch(component) is None:
                raise ValueError(f"Gate-1 example {number} {field} is invalid")
            prior_fold = component_folds.setdefault((field, component), example.fold)
            if prior_fold != example.fold:
                raise ValueError(f"Gate-1 {field} crosses outer folds")
        prior_sequence = sequence_metadata.setdefault(sequence_id, (sequence, example.fold))
        if prior_sequence != (sequence, example.fold):
            raise ValueError("a Gate-1 sequence identity changes sequence or outer fold")
        previous_example_id = example_id
        seen_ids.add(example_id)
        folds_seen.add(example.fold)
        targets_seen.add(target)

    if folds_seen != set(range(config.folds)):
        raise ValueError("Gate-1 examples do not cover every configured fold")
    if targets_seen != set(target_grams):
        raise ValueError("Gate-1 examples do not cover the configured target panel")
    if {example.label for example in examples} != {0, 1}:
        raise ValueError("Gate-1 examples must contain both activity labels")
    for fold in range(config.folds):
        if {example.label for example in examples if example.fold != fold} != {0, 1}:
            raise ValueError(f"outer-fold-complement {fold} must contain both activity labels")

    accepted_oof = gate1.descriptor_oof
    if not isinstance(accepted_oof, Mapping):
        raise ValueError("accepted descriptor OOF probabilities must be a mapping")
    if set(accepted_oof) != seen_ids or len(accepted_oof) != len(examples):
        raise ValueError("accepted descriptor OOF identities do not exactly match Gate-1 examples")
    for example in examples:
        probability = _require_number(
            accepted_oof[example.example_id],
            label=f"accepted descriptor OOF {example.example_id}",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("accepted descriptor OOF probability is outside [0, 1]")
    return examples


def _validate_publication_models(
    models: FittedActivityModels,
    *,
    examples: Sequence[Gate1Example],
    config: CandidateActivityScoringConfig,
) -> tuple[DescriptorLogisticOracle, ...]:
    if type(models.fold_models) is not tuple or len(models.fold_models) != config.folds:
        raise ValueError("candidate scoring requires exactly five ordered fold models")
    if type(models.model_states) is not tuple or len(models.model_states) != config.folds + 1:
        raise ValueError("candidate scoring requires exactly six ordered model states")
    live_models = (*models.fold_models, models.deployment_model)
    if len(live_models) != 6:
        raise ValueError("candidate scoring requires five fold models and one deployment model")
    expected_states: list[Mapping[str, object]] = []
    for index, model in enumerate(live_models):
        member = f"outer_fold_{index}" if index < config.folds else "all_data_deployment"
        heldout_fold = index if index < config.folds else None
        if type(model) is not DescriptorLogisticOracle:
            raise ValueError(f"{member} must be exactly DescriptorLogisticOracle")
        if (
            type(model.l2) is not float
            or model.l2 != config.descriptor.l2
            or type(model.max_iterations) is not int
            or model.max_iterations != config.descriptor.max_iterations
            or type(model.tolerance) is not float
            or model.tolerance != config.descriptor.tolerance
            or type(model.prior_strength) is not float
            or model.prior_strength != config.descriptor.prior_strength
        ):
            raise ValueError(f"{member} settings differ from the scoring config")
        training = tuple(
            example for example in examples if heldout_fold is None or example.fold != heldout_fold
        )
        try:
            expected_states.append(
                export_descriptor_model_state(
                    model,
                    name=member,
                    heldout_fold=heldout_fold,
                    training_examples=training,
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"{member} live fitted state is invalid") from error

    try:
        document = descriptor_model_states_document(models.model_states, config=config)
        imported = import_descriptor_model_states(document)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "descriptor model states are not exactly schema-valid/importable"
        ) from error
    if len(imported) != len(expected_states):
        raise ValueError("descriptor model state import did not produce exactly six members")
    for index, (observed, expected) in enumerate(
        zip(models.model_states, expected_states, strict=True)
    ):
        try:
            observed_payload = _canonical_json_bytes(observed)
            expected_payload = _canonical_json_bytes(expected)
        except (TypeError, ValueError) as error:
            raise ValueError(f"descriptor model state {index} is not JSON serializable") from error
        if observed_payload != expected_payload:
            raise ValueError(
                f"descriptor model state {index} does not exactly bind its live fitted model"
            )
    return imported


def _validate_publication_oof_reproduction(
    models: FittedActivityModels,
    *,
    examples: Sequence[Gate1Example],
    accepted_oof: Mapping[str, float],
    imported_models: Sequence[DescriptorLogisticOracle],
    config: CandidateActivityScoringConfig,
) -> None:
    by_fold: list[dict[str, int | float]] = []
    fold_maxima: list[float] = []
    for fold in range(config.folds):
        heldout = tuple(example for example in examples if example.fold == fold)
        if not heldout:
            raise ValueError(f"outer fold {fold} has no held-out examples")
        reproduced = imported_models[fold].predict_proba(
            tuple(example.model_input for example in heldout)
        )
        _validate_probability_array(
            reproduced,
            expected_shape=(len(heldout),),
            label=f"outer fold {fold} reproduced OOF probabilities",
        )
        errors = np.asarray(
            [
                abs(float(probability) - float(accepted_oof[example.example_id]))
                for example, probability in zip(heldout, reproduced, strict=True)
            ],
            dtype=np.float64,
        )
        maximum = float(np.max(errors))
        if maximum > config.oof_absolute_tolerance:
            raise ValueError(
                f"outer fold {fold} descriptor OOF mismatch: {maximum:.17g} exceeds "
                f"{config.oof_absolute_tolerance:.17g}"
            )
        fold_maxima.append(maximum)
        by_fold.append(
            {
                "fold": fold,
                "heldout_examples": len(heldout),
                "training_examples": len(examples) - len(heldout),
                "maximum_absolute_error": maximum,
            }
        )
    reproduced_maximum = max(fold_maxima)

    document = _require_exact_fields(
        models.oof_reproduction,
        expected={
            "schema_version",
            "artifact",
            "status",
            "accepted_model",
            "examples",
            "folds",
            "absolute_tolerance",
            "maximum_absolute_error",
            "by_fold",
        },
        label="OOF reproduction document",
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("OOF reproduction schema_version must be 1")
    if document["artifact"] != "candidate_activity_descriptor_oof_reproduction_v1":
        raise ValueError("OOF reproduction artifact identity is invalid")
    if document["status"] != "passed_exact_descriptor_oof_reproduction":
        raise ValueError("OOF reproduction status did not pass")
    if document["accepted_model"] != ACCEPTED_MODEL:
        raise ValueError("OOF reproduction accepted model is invalid")
    if type(document["examples"]) is not int or document["examples"] != len(examples):
        raise ValueError("OOF reproduction example census is invalid")
    if type(document["folds"]) is not int or document["folds"] != config.folds:
        raise ValueError("OOF reproduction fold census is invalid")
    tolerance = _require_number(
        document["absolute_tolerance"],
        label="OOF reproduction absolute_tolerance",
        positive=True,
    )
    if tolerance != config.oof_absolute_tolerance:
        raise ValueError("OOF reproduction tolerance differs from the scoring config")
    maximum = _require_number(
        document["maximum_absolute_error"],
        label="OOF reproduction maximum_absolute_error",
    )
    if maximum < 0.0 or maximum > tolerance or maximum != reproduced_maximum:
        raise ValueError("OOF reproduction maximum error is not exactly reproduced")
    observed_by_fold = document["by_fold"]
    if type(observed_by_fold) is not list or len(observed_by_fold) != config.folds:
        raise ValueError("OOF reproduction by_fold must contain five ordered records")
    for fold, expected in enumerate(by_fold):
        observed = _require_exact_fields(
            observed_by_fold[fold],
            expected={
                "fold",
                "heldout_examples",
                "training_examples",
                "maximum_absolute_error",
            },
            label=f"OOF reproduction fold {fold}",
        )
        if (
            type(observed["fold"]) is not int
            or observed["fold"] != expected["fold"]
            or type(observed["heldout_examples"]) is not int
            or observed["heldout_examples"] != expected["heldout_examples"]
            or type(observed["training_examples"]) is not int
            or observed["training_examples"] != expected["training_examples"]
        ):
            raise ValueError(f"OOF reproduction fold {fold} census is invalid")
        observed_error = _require_number(
            observed["maximum_absolute_error"],
            label=f"OOF reproduction fold {fold} maximum_absolute_error",
        )
        if (
            observed_error < 0.0
            or observed_error > tolerance
            or observed_error != expected["maximum_absolute_error"]
        ):
            raise ValueError(f"OOF reproduction fold {fold} error is not exactly reproduced")


def _require_publication_score_array(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    label: str,
    upper_bound: float,
) -> FloatArray:
    if type(value) is not np.ndarray:
        raise ValueError(f"{label} must be exactly a NumPy array")
    array = cast(NDArray[Any], value)
    if array.dtype.str != "<f8":
        raise ValueError(f"{label} must have exact little-endian float64 dtype")
    if array.shape != expected_shape:
        raise ValueError(f"{label} shape {array.shape!r} differs from {expected_shape!r}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > upper_bound):
        raise ValueError(f"{label} must contain finite values in [0, {upper_bound:g}]")
    return cast(FloatArray, array)


def _validate_publication_scores(
    scores: ActivityScoreBatch,
    *,
    candidate_count: int,
    config: CandidateActivityScoringConfig,
) -> None:
    target_count = len(config.targets)
    deployment = _require_publication_score_array(
        scores.deployment_target_probabilities,
        expected_shape=(candidate_count, target_count),
        label="deployment target probabilities",
        upper_bound=1.0,
    )
    fold_values = _require_publication_score_array(
        scores.fold_model_target_probabilities,
        expected_shape=(candidate_count, config.folds, target_count),
        label="fold-model target probabilities",
        upper_bound=1.0,
    )
    target_sensitivity = _require_publication_score_array(
        scores.model_fit_sensitivity_target,
        expected_shape=(candidate_count, target_count),
        label="per-target model-fit sensitivity",
        upper_bound=0.5,
    )
    if not isinstance(scores.means, Mapping) or set(scores.means) != set(_OBJECTIVES):
        raise ValueError("score means must contain exactly the three declared objectives")
    if not isinstance(scores.model_fit_sensitivity, Mapping) or set(
        scores.model_fit_sensitivity
    ) != set(_OBJECTIVES):
        raise ValueError(
            "score model-fit sensitivities must contain exactly the three declared objectives"
        )
    means = {
        objective: _require_publication_score_array(
            scores.means[objective],
            expected_shape=(candidate_count,),
            label=f"mean {objective}",
            upper_bound=1.0,
        )
        for objective in _OBJECTIVES
    }
    sensitivities = {
        objective: _require_publication_score_array(
            scores.model_fit_sensitivity[objective],
            expected_shape=(candidate_count,),
            label=f"model-fit sensitivity {objective}",
            upper_bound=0.5,
        )
        for objective in _OBJECTIVES
    }

    gram_indices = {
        gram: np.asarray(
            [index for index, target in enumerate(config.targets) if target.gram == gram],
            dtype=np.int64,
        )
        for gram in ("positive", "negative")
    }
    if any(indices.size == 0 for indices in gram_indices.values()):
        raise ValueError("target panel must represent both Gram classes")
    expected_means = {
        "broad_spectrum_activity": np.mean(deployment, axis=1),
        "gram_positive_activity": np.mean(deployment[:, gram_indices["positive"]], axis=1),
        "gram_negative_activity": np.mean(deployment[:, gram_indices["negative"]], axis=1),
    }
    fold_means = {
        "broad_spectrum_activity": np.mean(fold_values, axis=2),
        "gram_positive_activity": np.mean(fold_values[:, :, gram_indices["positive"]], axis=2),
        "gram_negative_activity": np.mean(fold_values[:, :, gram_indices["negative"]], axis=2),
    }
    expected_target_sensitivity = np.std(fold_values, axis=1, ddof=0)
    if not np.array_equal(target_sensitivity, expected_target_sensitivity):
        raise ValueError(
            "per-target model-fit sensitivity is not the exact population SD of fold values"
        )
    for objective in _OBJECTIVES:
        if not np.array_equal(means[objective], expected_means[objective]):
            raise ValueError(f"mean {objective} is not exactly recomputed from deployment values")
        expected_sensitivity = np.std(fold_means[objective], axis=1, ddof=0)
        if not np.array_equal(sensitivities[objective], expected_sensitivity):
            raise ValueError(
                f"model-fit sensitivity {objective} is not the exact population SD of fold means"
            )


def _validate_publication_inputs(
    *,
    config: CandidateActivityScoringConfig,
    candidates: AuthenticatedCandidates,
    gate1: AuthenticatedGate1,
    models: FittedActivityModels,
    scores: ActivityScoreBatch,
) -> None:
    """Reject an inconsistent handoff before constructing any output payload.

    Candidate-wide model inference remains the independent verifier's backstop;
    this gate only replays the much smaller accepted Gate-1 OOF census.
    """

    if type(config.folds) is not int or config.folds != _EXPECTED_FOLDS:
        raise ValueError("publication requires the frozen five-fold model contract")
    if config.objectives != _OBJECTIVES:
        raise ValueError("publication objectives differ from the frozen scoring contract")
    if tuple((target.name, target.gram) for target in config.targets) != _TARGET_PANEL:
        raise ValueError("publication target panel differs from the frozen scoring contract")
    records = _validate_publication_candidates(candidates, config=config)
    examples = _validate_publication_gate1(gate1, config=config)
    imported_models = _validate_publication_models(
        models,
        examples=examples,
        config=config,
    )
    _validate_publication_oof_reproduction(
        models,
        examples=examples,
        accepted_oof=gate1.descriptor_oof,
        imported_models=imported_models,
        config=config,
    )
    _validate_publication_scores(
        scores,
        candidate_count=len(records),
        config=config,
    )


def _float_text(value: float) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("cannot serialize a non-finite float")
    return f"{result:.17g}"


def _candidate_csv_bytes(
    candidates: Sequence[CandidateRecord],
    scores: ActivityScoreBatch,
    *,
    config: CandidateActivityScoringConfig,
    model_release_id: str,
) -> tuple[bytes, tuple[str, ...]]:
    target_columns = tuple(f"probability_{target.name}" for target in config.targets)
    columns = (
        "source_ordinal",
        "sequence_id",
        "sequence",
        "length",
        "library_eligible",
        "generator_families",
        "generator_variants",
        *target_columns,
        "mean_broad_spectrum_activity",
        "mean_gram_positive_activity",
        "mean_gram_negative_activity",
        "model_fit_sensitivity_broad_spectrum_activity",
        "model_fit_sensitivity_gram_positive_activity",
        "model_fit_sensitivity_gram_negative_activity",
        "prediction_scope",
        "training_scope",
        "calibration_scope",
        "model_release_id",
    )
    if any(column.startswith("std_") for column in columns):
        raise AssertionError("candidate scoring schema cannot contain a std_ alias")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for index, candidate in enumerate(candidates):
        row: dict[str, object] = {
            "source_ordinal": candidate.source_ordinal,
            "sequence_id": candidate.sequence_id,
            "sequence": candidate.sequence,
            "length": candidate.length,
            "library_eligible": "true",
            "generator_families": "|".join(candidate.generator_families),
            "generator_variants": "|".join(candidate.generator_variants),
            "mean_broad_spectrum_activity": _float_text(
                scores.means["broad_spectrum_activity"][index]
            ),
            "mean_gram_positive_activity": _float_text(
                scores.means["gram_positive_activity"][index]
            ),
            "mean_gram_negative_activity": _float_text(
                scores.means["gram_negative_activity"][index]
            ),
            "model_fit_sensitivity_broad_spectrum_activity": _float_text(
                scores.model_fit_sensitivity["broad_spectrum_activity"][index]
            ),
            "model_fit_sensitivity_gram_positive_activity": _float_text(
                scores.model_fit_sensitivity["gram_positive_activity"][index]
            ),
            "model_fit_sensitivity_gram_negative_activity": _float_text(
                scores.model_fit_sensitivity["gram_negative_activity"][index]
            ),
            "prediction_scope": PREDICTION_SCOPE,
            "training_scope": config.training_scope,
            "calibration_scope": CALIBRATION_SCOPE,
            "model_release_id": model_release_id,
        }
        for target_index, column in enumerate(target_columns):
            row[column] = _float_text(scores.deployment_target_probabilities[index, target_index])
        writer.writerow(row)
    return stream.getvalue().encode("utf-8"), columns


def _npy_bytes(values: FloatArray) -> tuple[bytes, str]:
    tensor = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    if tensor.dtype.str != "<f8" or not tensor.flags.c_contiguous:
        raise AssertionError("fold tensor must be C-contiguous little-endian float64")
    raw_digest = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
    stream = io.BytesIO()
    np.save(stream, tensor, allow_pickle=False)
    return stream.getvalue(), raw_digest


def _artifact_record(
    payload: bytes,
    *,
    rows: int | None = None,
) -> Mapping[str, object]:
    result: dict[str, object] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _input_manifest(
    *,
    config: CandidateActivityScoringConfig,
    candidates: AuthenticatedCandidates,
    gate1: AuthenticatedGate1,
) -> Mapping[str, object]:
    return {
        "config": {
            "sha256": config.snapshot.sha256,
            "size_bytes": config.snapshot.size_bytes,
        },
        "candidate_pool": {
            "producer_job_id": config.candidate_pool.producer_job_id,
            "git_commit": config.candidate_pool.git_commit,
            "publication_top_sha256": config.candidate_pool.publication_top_sha256,
            "manifest_sha256": config.candidate_pool.manifest_sha256,
            "candidates_sha256": config.candidate_pool.candidates_sha256,
            "validation_summary_sha256": config.candidate_pool.validation_summary_sha256,
            "final_publication_check_sha256": config.candidate_pool.final_publication_check_sha256,
            "candidates": len(candidates.records),
        },
        "gate1": {
            "producer_job_id": config.gate1.producer_job_id,
            "audit_job_id": config.gate1.audit_job_id,
            "git_commit": config.gate1.git_commit,
            "publication_top_sha256": config.gate1.publication_top_sha256,
            "semantic_top_sha256": config.gate1.semantic_top_sha256,
            "examples_sha256": config.gate1.examples_sha256,
            "oof_sha256": config.gate1.oof_sha256,
            "independent_receipt_sha256": config.gate1.independent_receipt_sha256,
            "examples": len(gate1.examples),
        },
    }


def _manifest_document(
    *,
    config: CandidateActivityScoringConfig,
    candidates: AuthenticatedCandidates,
    gate1: AuthenticatedGate1,
    git_commit: str,
    model_release_id: str,
    csv_columns: Sequence[str],
    scores: ActivityScoreBatch,
    tensor_raw_sha256: str,
    artifact_records: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "status": OUTPUT_STATUS,
        "automatic_production_eligible": False,
        "probability_calibration": PROBABILITY_CALIBRATION,
        "prediction_scope": PREDICTION_SCOPE,
        "training_scope": config.training_scope,
        "fit_sensitivity_semantics": config.fit_sensitivity_semantics,
        "target_aggregation": config.target_aggregation,
        "model_release_id": model_release_id,
        "git_commit": git_commit,
        "inputs": _input_manifest(config=config, candidates=candidates, gate1=gate1),
        "targets": [
            {
                "index": index,
                "name": target.name,
                "gram": target.gram,
                "probability_column": f"probability_{target.name}",
            }
            for index, target in enumerate(config.targets)
        ],
        "models": {
            "family": ACCEPTED_MODEL,
            "accepted_oof_model": ACCEPTED_MODEL,
            "fold_members": [f"outer_fold_{fold}" for fold in range(config.folds)],
            "deployment_member": "all_data_deployment",
            "weights": dict(config.model_weights),
            "serialized_states_file": "model_states.json",
        },
        "aggregations": {
            "objectives": list(config.objectives),
            "target_reduction": config.target_aggregation,
            "deployment_probability_source": "all_data_deployment",
            "model_fit_sensitivity": config.fit_sensitivity_semantics,
        },
        "fold_tensor": {
            "filename": "fold_model_target_probabilities.npy",
            "axes": ["candidate", "outer_fold_complement_model", "target"],
            "shape": list(scores.fold_model_target_probabilities.shape),
            "dtype": "float64",
            "byte_order": "little_endian",
            "memory_order": "C_contiguous",
            "candidate_axis_order": "candidate_jsonl_source_ordinal_ascending",
            "fold_axis_member_order": [f"outer_fold_{fold}" for fold in range(config.folds)],
            "target_axis_order": [target.name for target in config.targets],
            "raw_data_sha256": tensor_raw_sha256,
        },
        "candidate_csv": {
            "filename": "candidate_activity_scores.csv",
            "columns": list(csv_columns),
            "rows": len(candidates.records),
        },
        "claims": {
            "forbidden": list(config.forbidden_claims),
            "historical_method_choice_used_fold4": config.historical_method_choice_used_fold4,
            "organizer_reference_used_for_model_fit": config.organizer_reference_used_for_model_fit,
        },
        "artifacts": dict(artifact_records),
    }


def _sha256sums_bytes(payloads: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing scoring artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_write_once(
    output_dir: str | Path,
    *,
    payloads: Mapping[str, bytes],
    snapshots: Mapping[str, InputSnapshot],
) -> Path:
    if set(payloads) != set(_OUTPUT_FILES):
        raise ValueError("output payload inventory is not exact")
    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested):
        raise FileExistsError(f"refusing to reuse scoring output: {requested}")
    parent = requested.parent
    _reject_symlink_chain(parent, label="scoring output parent")
    try:
        parent_metadata = os.lstat(parent)
    except OSError as error:
        raise ValueError(f"cannot inspect scoring output parent: {parent}") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("scoring output parent must be a non-symbolic directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{requested.name}.staging-", dir=parent))
    claimed = False
    linked: list[str] = []
    try:
        for name in _OUTPUT_FILES:
            _write_new_bytes(staging / name, payloads[name])
            os.chmod(staging / name, 0o444)
        for label, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, label=label)
        os.mkdir(requested, mode=0o755)
        claimed = True
        # SHA256SUMS is the final commit marker.
        for name in (*(_OUTPUT_FILES[:-1]), "SHA256SUMS"):
            os.link(staging / name, requested / name, follow_symlinks=False)
            linked.append(name)
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


def publish_candidate_activity_scoring(
    *,
    output_dir: str | Path,
    config: CandidateActivityScoringConfig,
    candidates: AuthenticatedCandidates,
    gate1: AuthenticatedGate1,
    models: FittedActivityModels,
    scores: ActivityScoreBatch,
    git_commit: str,
) -> CandidateActivityScoringExecution:
    """Serialize and publish the exact write-once development artifact set."""

    git_commit = _require_git_sha(git_commit, label="scorer git_commit")
    _validate_publication_inputs(
        config=config,
        candidates=candidates,
        gate1=gate1,
        models=models,
        scores=scores,
    )
    states_document = descriptor_model_states_document(models.model_states, config=config)
    states_payload = _canonical_json_bytes(states_document)
    model_release_id = hashlib.sha256(states_payload).hexdigest()
    oof_payload = _canonical_json_bytes(models.oof_reproduction)
    tensor_payload, tensor_raw_sha256 = _npy_bytes(scores.fold_model_target_probabilities)
    csv_payload, csv_columns = _candidate_csv_bytes(
        candidates.records,
        scores,
        config=config,
        model_release_id=model_release_id,
    )
    semantic_payloads = {
        "candidate_activity_scores.csv": csv_payload,
        "fold_model_target_probabilities.npy": tensor_payload,
        "model_states.json": states_payload,
        "oof_reproduction.json": oof_payload,
    }
    artifact_records = {
        "candidate_activity_scores.csv": _artifact_record(
            csv_payload, rows=len(candidates.records)
        ),
        "fold_model_target_probabilities.npy": _artifact_record(tensor_payload),
        "model_states.json": _artifact_record(states_payload),
        "oof_reproduction.json": _artifact_record(oof_payload),
    }
    manifest = _manifest_document(
        config=config,
        candidates=candidates,
        gate1=gate1,
        git_commit=git_commit,
        model_release_id=model_release_id,
        csv_columns=csv_columns,
        scores=scores,
        tensor_raw_sha256=tensor_raw_sha256,
        artifact_records=artifact_records,
    )
    manifest_payload = _canonical_json_bytes(manifest)
    checksummed = {**semantic_payloads, "manifest.json": manifest_payload}
    payloads = {**checksummed, "SHA256SUMS": _sha256sums_bytes(checksummed)}
    all_snapshots = {
        "scoring config": config.snapshot,
        **candidates.snapshots,
        **gate1.snapshots,
    }
    output = _publish_write_once(output_dir, payloads=payloads, snapshots=all_snapshots)
    return CandidateActivityScoringExecution(
        output_dir=output,
        candidate_count=len(candidates.records),
        model_release_id=model_release_id,
        csv_sha256=hashlib.sha256(csv_payload).hexdigest(),
        fold_tensor_sha256=hashlib.sha256(tensor_payload).hexdigest(),
        fold_tensor_raw_data_sha256=tensor_raw_sha256,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )


def run_candidate_activity_scoring(
    *,
    candidate_pool_twin_root: str | Path,
    candidate_validation_summary_path: str | Path,
    candidate_final_publication_check_path: str | Path,
    gate1_twin_root: str | Path,
    gate1_independent_receipt_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    git_commit: str,
) -> CandidateActivityScoringExecution:
    """Authenticate, replay, fit, score, and publish the development scorer."""

    requested_output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse scoring output: {requested_output}")
    config = load_candidate_activity_scoring_config(config_path)
    candidates = read_candidate_records(
        candidate_pool_twin_root,
        validation_summary_path=candidate_validation_summary_path,
        final_publication_check_path=candidate_final_publication_check_path,
        config=config,
    )
    gate1 = read_gate1_evidence(
        gate1_twin_root,
        independent_receipt_path=gate1_independent_receipt_path,
        config=config,
    )
    models = fit_activity_models(
        gate1.examples,
        gate1.descriptor_oof,
        config=config,
    )
    scores = score_candidate_records(candidates.records, models=models, config=config)
    return publish_candidate_activity_scoring(
        output_dir=requested_output,
        config=config,
        candidates=candidates,
        gate1=gate1,
        models=models,
        scores=scores,
        git_commit=git_commit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run authenticated CPU-only development candidate activity scoring."
    )
    parser.add_argument("--candidate-pool-twin-root", type=Path, required=True)
    parser.add_argument("--candidate-validation-summary", type=Path, required=True)
    parser.add_argument("--candidate-final-publication-check", type=Path, required=True)
    parser.add_argument("--gate1-twin-root", type=Path, required=True)
    parser.add_argument("--gate1-independent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = run_candidate_activity_scoring(
            candidate_pool_twin_root=args.candidate_pool_twin_root,
            candidate_validation_summary_path=args.candidate_validation_summary,
            candidate_final_publication_check_path=args.candidate_final_publication_check,
            gate1_twin_root=args.gate1_twin_root,
            gate1_independent_receipt_path=args.gate1_independent_receipt,
            config_path=args.config,
            output_dir=args.output,
            git_commit=args.git_commit,
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"candidate activity scoring error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_count": execution.candidate_count,
                "fold_tensor_raw_data_sha256": execution.fold_tensor_raw_data_sha256,
                "manifest_sha256": execution.manifest_sha256,
                "model_release_id": execution.model_release_id,
                "output_dir": str(execution.output_dir),
                "status": OUTPUT_STATUS,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the cluster CLI.
    raise SystemExit(main())

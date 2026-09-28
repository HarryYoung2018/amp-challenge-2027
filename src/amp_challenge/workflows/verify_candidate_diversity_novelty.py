"""Independently verify the full-pool diversity and novelty handoff.

This verifier intentionally does not import the matching producer.  It
authenticates every frozen input, independently reconstructs exact Indel
similarities and the transparent 33-dimensional feature space, and compares
both complete five-file publication twins byte for byte before emitting a
narrow, deterministic receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import stat
import sys
import tomllib
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Indel

_SCHEMA_VERSION = 1
_CONFIG_ARTIFACT = "candidate_diversity_novelty_v1"
_CONFIG_STATUS = "predeclared_development_diversity_novelty_handoff"
_OUTPUT_ARTIFACT = "candidate_diversity_novelty_v1"
_OUTPUT_STATUS = "development_diversity_novelty_evidence"
_ALLOWED_CONSUMER = "development_diversity_novelty_selection_only"
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AMINO_ACIDS = frozenset(_ALPHABET)
_FEATURE_SPACE = "physchem_descriptors_plus_canonical_amino_acid_fractions_v1"
_UNCERTAINTY_STATUS = "unavailable"
_REFERENCE_THRESHOLD = 0.80
_EXPECTED_CANDIDATES = 115_536
_EXPECTED_TRAINING_ROWS = 914
_EXPECTED_REFERENCE_RECORDS = 39_448

_TARGETS = (
    "acinetobacter_baumannii",
    "enterococcus_faecalis",
    "enterococcus_faecium",
    "escherichia_coli",
    "klebsiella_pneumoniae",
    "pseudomonas_aeruginosa",
    "staphylococcus_aureus",
)
_OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
_DESCRIPTOR_NAMES = (
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
)
_FEATURE_NAMES = _DESCRIPTOR_NAMES + tuple(f"residue_fraction_{aa}" for aa in _ALPHABET)
_EMBEDDING_COLUMNS = tuple(f"embedding_physchem_{index:03d}" for index in range(33))

_SOURCE_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    "sequence",
    "length",
    "eligible",
    "eligibility_scope",
    "generator_families",
    "generator_variants",
    *(f"probability_{target}" for target in _TARGETS),
    *(f"mean_{objective}" for objective in _OBJECTIVES),
    "uncertainty_status",
    "prediction_scope",
    "training_scope",
    "calibration_scope",
    "model_release_id",
)
_PRESERVED_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    "sequence",
    "length",
    "source_library_eligible",
    "eligibility_scope",
    "generator_families",
    "generator_variants",
    *(f"probability_{target}" for target in _TARGETS),
    *(f"mean_{objective}" for objective in _OBJECTIVES),
    "uncertainty_status",
    "prediction_scope",
    "training_scope",
    "calibration_scope",
    "model_release_id",
)
_NOVELTY_COLUMNS = (
    "max_train_indel_similarity",
    "nearest_train_sequence_id",
    "novelty",
    "reference_safe_0_80",
    "reference_violation_witness_sequence_id",
    "reference_violation_witness_similarity",
    "eligible",
)
_OUTPUT_COLUMNS = _PRESERVED_COLUMNS + _NOVELTY_COLUMNS + _EMBEDDING_COLUMNS

_MEAN_LEDGER_FILES = frozenset(
    {
        "candidate_ledger.csv",
        "model_fit_sensitivity_diagnostics.csv",
        "manifest.json",
        "SHA256SUMS",
    }
)
_MEAN_LEDGER_AUDIT_FILES = frozenset(
    {
        "operational-receipt.json",
        "twin-0-independent-verification.json",
        "twin-1-independent-verification.json",
    }
)
_OUTPUT_FILES = frozenset(
    {
        "candidate_selection_ledger.csv",
        "diversity_feature_matrix.f32le",
        "feature_transform.json",
        "manifest.json",
        "SHA256SUMS",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_LINEAGE_TOKEN_RE = re.compile(r"[^|,\r\n\x00-\x20\x7f]{1,256}")
_FORBIDDEN_COLUMN_TOKENS = (
    "aleatoric",
    "confidence_interval",
    "epistemic",
    "hemolysis",
    "mdr",
    "ood",
    "out_of_distribution",
    "posterior",
    "quality_probability",
    "selectivity",
    "sensitivity",
    "std_",
    "variance",
)

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
_EISENBERG_HYDROPHOBICITY = {
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
_HYDROPHOBIC_RESIDUES = frozenset("ACFILMVWY")
_AROMATIC_RESIDUES = frozenset("FWY")
_BASIC_RESIDUES = frozenset("HKR")
_ACIDIC_RESIDUES = frozenset("DE")
_POSITIVE_SIDECHAIN_PKA = {"H": 6.0, "K": 10.5, "R": 12.5}
_NEGATIVE_SIDECHAIN_PKA = {"C": 8.3, "D": 3.9, "E": 4.1, "Y": 10.1}
_N_TERMINUS_PKA = 8.0
_C_TERMINUS_PKA = 3.1


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Contract:
    expected_candidates: int
    expected_training_rows: int
    expected_reference_records: int
    threshold: float
    descriptor_ph: float
    hydrophobic_moment_angle_degrees: float
    isoelectric_point_lower_ph: float
    isoelectric_point_upper_ph: float
    isoelectric_point_iterations: int
    worker_minimum: int
    worker_maximum: int
    accepted_mean_ledger: Mapping[str, object]
    accepted_training_projection: Mapping[str, object]
    organizer_reference: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Candidate:
    source: Mapping[str, str]
    raw_features: tuple[float, ...]

    @property
    def sequence(self) -> str:
        return self.source["sequence"]


@dataclass(frozen=True, slots=True)
class TrainingRow:
    sequence_id: str
    sequence: str
    weight: float
    raw_features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    max_train_similarity: float
    nearest_train_sequence_id: str
    reference_safe: bool
    reference_witness_sequence_id: str
    reference_witness_similarity: float | None


@dataclass(frozen=True, slots=True)
class _SimilarityWorkerState:
    training_ids: tuple[str, ...]
    training_sequences: tuple[str, ...]
    reference_tables: Mapping[int, tuple[tuple[str, ...], tuple[str, ...]]]
    threshold: float


_SIMILARITY_WORKER_STATE: _SimilarityWorkerState | None = None
_MAX_PROCESS_CHUNKSIZE = 128


@dataclass(frozen=True, slots=True)
class Reconstruction:
    ledger_payload: bytes
    matrix_payload: bytes
    transform_document: Mapping[str, object]
    transform_payload: bytes
    rows: int
    source_eligible_rows: int
    reference_safe_rows: int
    effective_eligible_rows: int
    ledger_semantic_sha256: str
    matrix_semantic_sha256: str
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    candidate_raw_feature_sha256: str
    training_raw_feature_sha256: str
    training_weight_sha256: str
    training_weight_sum: float
    similarities: tuple[SimilarityResult, ...]
    lineage_census: Mapping[str, object]


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json(value: object) -> bytes:
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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _json_object(payload: bytes, *, label: str, canonical: bool = True) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    assert isinstance(value, dict)
    if canonical:
        _require(_canonical_json(value) == payload, f"{label} is not canonical JSON")
    return value


def _exact_mapping(value: object, *, label: str, keys: set[str]) -> Mapping[str, object]:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    observed = set(value)
    _require(
        observed == keys,
        f"{label} keys differ: missing={sorted(keys - observed)}, extra={sorted(observed - keys)}",
    )
    return value


def _fingerprint(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int, int]:
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


def _reject_symlink_chain(path: str | Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in reversed((absolute, *absolute.parents)):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {candidate}") from error
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"{label} cannot traverse a symbolic link: {candidate}",
        )


def _snapshot(
    path: str | Path,
    *,
    label: str,
    expected_mode: int | None = None,
    require_single_link: bool = False,
) -> Snapshot:
    requested = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(requested, label=label)
    try:
        named_before = os.lstat(requested)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {requested}") from error
    _require(stat.S_ISREG(named_before.st_mode), f"{label} must be a regular file")
    _require(not stat.S_ISLNK(named_before.st_mode), f"{label} must not be a symbolic link")
    if expected_mode is not None:
        _require(
            stat.S_IMODE(named_before.st_mode) == expected_mode,
            f"{label} mode must be {expected_mode:04o}",
        )
    if require_single_link:
        _require(named_before.st_nlink == 1, f"{label} must have exactly one hard link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {requested}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(requested)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    _require(
        len(
            {
                _fingerprint(item)
                for item in (named_before, opened_before, opened_after, named_after)
            }
        )
        == 1,
        f"{label} changed while it was read",
    )
    payload = b"".join(chunks)
    _require(len(payload) == named_before.st_size, f"{label} size changed while read")
    _reject_symlink_chain(requested, label=label)
    return Snapshot(
        path=requested,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        fingerprint=_fingerprint(named_before),
    )


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    _reject_symlink_chain(snapshot.path, label=label)
    try:
        metadata = os.lstat(snapshot.path)
    except OSError as error:
        raise ValueError(f"cannot recheck {label}: {snapshot.path}") from error
    _require(
        stat.S_ISREG(metadata.st_mode) and _fingerprint(metadata) == snapshot.fingerprint,
        f"{label} changed after authentication during verification",
    )


def _safe_filename(value: str, *, label: str) -> str:
    pure = PurePosixPath(value)
    _require(
        value != ""
        and not pure.is_absolute()
        and len(pure.parts) == 1
        and pure.parts[0] not in {".", ".."}
        and "\\" not in value
        and "\x00" not in value,
        f"{label} contains unsafe filename {value!r}",
    )
    return value


def _parse_sha256sums(payload: bytes, *, label: str) -> Mapping[str, str]:
    _require(
        bool(payload) and payload.endswith(b"\n") and b"\r" not in payload,
        f"{label} must be non-empty LF-framed text",
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be ASCII") from error
    entries: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00-\x1f\x7f]+)", line)
        _require(match is not None, f"{label} line {number} is malformed")
        assert match is not None
        digest, raw_name = match.groups()
        name = _safe_filename(raw_name, label=label)
        _require(name != "SHA256SUMS", f"{label} cannot list itself")
        _require(name not in entries, f"{label} repeats {name!r}")
        _require(previous is None or previous < name, f"{label} entries are not sorted")
        entries[name] = digest
        previous = name
    return entries


def _directory_fingerprint(
    path: str | Path, *, label: str, expected_mode: int = 0o555
) -> tuple[int, int, int, int, int, int, int, int, int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(absolute, label=label)
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {absolute}") from error
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be a real directory",
    )
    _require(
        stat.S_IMODE(metadata.st_mode) == expected_mode,
        f"{label} mode must be {expected_mode:04o}",
    )
    return _fingerprint(metadata)


def _authenticate_bundle(
    root: str | Path, *, expected_files: frozenset[str], label: str
) -> tuple[Mapping[str, Snapshot], tuple[int, int, int, int, int, int, int, int, int]]:
    path = Path(os.path.abspath(os.fspath(root)))
    directory_fingerprint = _directory_fingerprint(path, label=label)
    try:
        entries = list(os.scandir(path))
    except OSError as error:
        raise ValueError(f"cannot enumerate {label}") from error
    names = {entry.name for entry in entries}
    _require(
        names == set(expected_files),
        f"{label} inventory differs: missing={sorted(set(expected_files) - names)}, "
        f"extra={sorted(names - set(expected_files))}",
    )
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        _require(
            entry.is_file(follow_symlinks=False)
            and not entry.is_symlink()
            and stat.S_IMODE(metadata.st_mode) == 0o444
            and metadata.st_nlink == 1,
            f"{label} artifact {entry.name} must be a singly linked regular 0444 file",
        )
    top = _snapshot(path / "SHA256SUMS", label=f"{label} SHA256SUMS", expected_mode=0o444)
    checksums = _parse_sha256sums(top.payload, label=f"{label} SHA256SUMS")
    _require(
        set(checksums) == set(expected_files) - {"SHA256SUMS"},
        f"{label} checksum inventory is not exact",
    )
    snapshots: dict[str, Snapshot] = {"SHA256SUMS": top}
    for name in sorted(checksums):
        item = _snapshot(path / name, label=f"{label} artifact {name}", expected_mode=0o444)
        _require(item.sha256 == checksums[name], f"{label} checksum mismatch for {name}")
        snapshots[name] = item
    return snapshots, directory_fingerprint


def _authenticate_plain_directory(
    root: str | Path, *, expected_files: frozenset[str], label: str
) -> tuple[Mapping[str, Snapshot], tuple[int, int, int, int, int, int, int, int, int]]:
    path = Path(os.path.abspath(os.fspath(root)))
    directory_fingerprint = _directory_fingerprint(path, label=label)
    entries = list(os.scandir(path))
    _require({entry.name for entry in entries} == set(expected_files), f"{label} inventory differs")
    snapshots: dict[str, Snapshot] = {}
    for name in sorted(expected_files):
        snapshots[name] = _snapshot(
            path / name, label=f"{label} artifact {name}", expected_mode=0o444
        )
    return snapshots, directory_fingerprint


def _assert_directory_unchanged(
    root: str | Path,
    *,
    fingerprint: tuple[int, int, int, int, int, int, int, int, int],
    expected_files: frozenset[str],
    label: str,
) -> None:
    path = Path(os.path.abspath(os.fspath(root)))
    _require(
        _directory_fingerprint(path, label=label) == fingerprint,
        f"{label} changed during verification",
    )
    _require(
        {entry.name for entry in os.scandir(path)} == set(expected_files),
        f"{label} inventory changed during verification",
    )


def _tree_sha256(snapshots: Mapping[str, Snapshot]) -> str:
    transcript = b"".join(
        f"444 {snapshots[name].sha256} {name}\n".encode("ascii") for name in sorted(snapshots)
    )
    return hashlib.sha256(transcript).hexdigest()


def _sha256_value(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _positive_int(value: object, *, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")
    assert isinstance(value, int)
    return value


def _finite_float(value: object, *, label: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    assert isinstance(value, float)
    return value


def _parse_config(snapshot: Snapshot) -> Contract:
    try:
        document = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("diversity/novelty config must be valid UTF-8 TOML") from error
    root = _exact_mapping(
        document,
        label="diversity/novelty config",
        keys={
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
    )
    _require(root["schema_version"] == 1, "config schema_version differs")
    _require(root["artifact"] == _CONFIG_ARTIFACT, "config artifact differs")
    _require(root["status"] == _CONFIG_STATUS, "config status differs")
    _require(root["automatic_production_eligible"] is False, "config cannot be production eligible")
    _require(root["allowed_consumer"] == _ALLOWED_CONSUMER, "config allowed consumer differs")
    _require(root["alphabet"] == _ALPHABET, "config alphabet differs")
    _require(root["uncertainty_status"] == _UNCERTAINTY_STATUS, "uncertainty must be unavailable")
    expected_candidates = _positive_int(root["expected_candidates"], label="expected_candidates")
    expected_training_rows = _positive_int(
        root["expected_training_rows"], label="expected_training_rows"
    )
    expected_reference_records = _positive_int(
        root["expected_reference_records"], label="expected_reference_records"
    )
    _require(expected_candidates == _EXPECTED_CANDIDATES, "candidate census differs")
    _require(expected_training_rows == _EXPECTED_TRAINING_ROWS, "training census differs")
    _require(expected_reference_records == _EXPECTED_REFERENCE_RECORDS, "reference census differs")
    threshold = _finite_float(
        root["reference_similarity_threshold"], label="reference_similarity_threshold"
    )
    _require(threshold == _REFERENCE_THRESHOLD, "reference threshold must be exactly 0.80")

    accepted_mean = _exact_mapping(
        root["accepted_mean_ledger"],
        label="accepted_mean_ledger",
        keys={
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
    )
    for key in ("producer_job_id", "audit_job_id"):
        _positive_int(accepted_mean[key], label=f"accepted_mean_ledger.{key}")
    _require(
        isinstance(accepted_mean["git_commit"], str)
        and _GIT_SHA_RE.fullmatch(str(accepted_mean["git_commit"])) is not None,
        "accepted mean-ledger Git commit is invalid",
    )
    for key in set(accepted_mean) - {"producer_job_id", "audit_job_id", "git_commit"}:
        _sha256_value(accepted_mean[key], label=f"accepted_mean_ledger.{key}")

    accepted_projection = _exact_mapping(
        root["accepted_training_projection"],
        label="accepted_training_projection",
        keys={
            "stage_job_id",
            "git_commit",
            "projection_sha256",
            "projection_size_bytes",
            "receipt_sha256",
            "receipt_size_bytes",
            "fields",
            "weighting",
        },
    )
    _positive_int(accepted_projection["stage_job_id"], label="projection stage_job_id")
    for key in ("projection_size_bytes", "receipt_size_bytes"):
        _positive_int(accepted_projection[key], label=f"projection {key}")
    _require(
        isinstance(accepted_projection["git_commit"], str)
        and _GIT_SHA_RE.fullmatch(str(accepted_projection["git_commit"])) is not None,
        "accepted projection Git commit is invalid",
    )
    for key in ("projection_sha256", "receipt_sha256"):
        _sha256_value(accepted_projection[key], label=f"accepted_training_projection.{key}")
    _require(
        accepted_projection["fields"] == ["sequence_id", "sequence", "sampling_weight"],
        "accepted projection fields differ",
    )
    _require(
        accepted_projection["weighting"]
        == "component_equal_sampling_weight_from_accepted_train_projection",
        "accepted projection weighting differs",
    )

    organizer = _exact_mapping(
        root["organizer_reference"],
        label="organizer_reference",
        keys={"sha256", "size_bytes", "source_commit", "role"},
    )
    _sha256_value(organizer["sha256"], label="organizer_reference.sha256")
    _positive_int(organizer["size_bytes"], label="organizer_reference.size_bytes")
    _require(
        isinstance(organizer["source_commit"], str)
        and _GIT_SHA_RE.fullmatch(str(organizer["source_commit"])) is not None,
        "organizer source commit is invalid",
    )
    _require(organizer["role"] == "compliance_reference_only", "organizer role differs")

    feature_config = _exact_mapping(
        root["features"],
        label="features",
        keys={
            "descriptor_names",
            "residue_order",
            "feature_count",
            "feature_space",
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
    )
    expected_features = {
        "descriptor_names": list(_DESCRIPTOR_NAMES),
        "residue_order": _ALPHABET,
        "feature_count": len(_FEATURE_NAMES),
        "feature_space": _FEATURE_SPACE,
        "free_termini": True,
        "fit_rows": "accepted_training_projection_only",
        "training_weighting": "sampling_weight",
        "weighted_mean_algorithm": "math_fsum_of_weight_times_value_divided_by_math_fsum_weights",
        "weighted_scale_algorithm": "population_sd_sqrt_math_fsum_weight_times_squared_centered_value_divided_by_math_fsum_weights",
        "standardization": "candidate_raw_minus_training_weighted_mean_divided_by_training_weighted_population_sd",
        "row_normalization": "euclidean_l2_after_standardization",
        "output_dtype": "float32",
        "byte_order": "little_endian",
        "memory_order": "C",
        "csv_float_format": "shortest_roundtrip_little_endian_float32",
    }
    for key, expected in expected_features.items():
        _require(feature_config[key] == expected, f"features.{key} differs")
    descriptor_ph = _finite_float(feature_config["descriptor_ph"], label="descriptor_ph")
    moment_angle = _finite_float(
        feature_config["hydrophobic_moment_angle_degrees"],
        label="hydrophobic_moment_angle_degrees",
    )
    pi_lower = _finite_float(
        feature_config["isoelectric_point_lower_ph"], label="isoelectric_point_lower_ph"
    )
    pi_upper = _finite_float(
        feature_config["isoelectric_point_upper_ph"], label="isoelectric_point_upper_ph"
    )
    pi_iterations = _positive_int(
        feature_config["isoelectric_point_iterations"],
        label="isoelectric_point_iterations",
    )
    _require(
        (descriptor_ph, moment_angle, pi_lower, pi_upper, pi_iterations)
        == (7.4, 100.0, 0.0, 14.0, 60),
        "descriptor numeric settings differ from v1",
    )

    similarity = _exact_mapping(
        root["similarity"],
        label="similarity",
        keys={
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
    )
    expected_similarity = {
        "implementation": "rapidfuzz.distance.Indel.normalized_similarity",
        "value_range": "zero_to_one",
        "training_reduction": "exact_maximum_over_all_accepted_training_sequences",
        "nearest_training_tie_break": "lexicographically_smallest_sequence_id",
        "novelty": "one_minus_max_train_indel_similarity",
        "reference_rule": "unsafe_if_any_unique_organizer_reference_similarity_strictly_exceeds_threshold",
        "reference_threshold_equality": "safe",
        "reference_witness_tie_break": "lexicographically_smallest_sequence_id",
        "reference_maximum_similarity_stored": False,
        "length_pruning": "upper_bound_two_times_min_length_divided_by_sum_lengths",
        "candidate_chunk_size": 1024,
        "choice_chunk_size": 1024,
        "worker_default": 4,
        "worker_minimum": 1,
        "worker_maximum": 128,
        "worker_count_affects_output_bytes": False,
    }
    for key, expected in expected_similarity.items():
        _require(similarity[key] == expected, f"similarity.{key} differs")
    worker_minimum = _positive_int(similarity["worker_minimum"], label="worker_minimum")
    worker_maximum = _positive_int(similarity["worker_maximum"], label="worker_maximum")
    _positive_int(similarity["candidate_chunk_size"], label="candidate_chunk_size")
    _positive_int(similarity["choice_chunk_size"], label="choice_chunk_size")
    worker_default = _positive_int(similarity["worker_default"], label="worker_default")
    _require(
        worker_minimum <= worker_default <= worker_maximum,
        "worker_default lies outside inclusive worker bounds",
    )

    claims = _exact_mapping(
        root["claims"],
        label="claims",
        keys={
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
        },
    )
    _require(
        all(value is False for value in claims.values()), "all unsupported claims must be false"
    )
    return Contract(
        expected_candidates=expected_candidates,
        expected_training_rows=expected_training_rows,
        expected_reference_records=expected_reference_records,
        threshold=threshold,
        descriptor_ph=descriptor_ph,
        hydrophobic_moment_angle_degrees=moment_angle,
        isoelectric_point_lower_ph=pi_lower,
        isoelectric_point_upper_ph=pi_upper,
        isoelectric_point_iterations=pi_iterations,
        worker_minimum=worker_minimum,
        worker_maximum=worker_maximum,
        accepted_mean_ledger=dict(accepted_mean),
        accepted_training_projection=dict(accepted_projection),
        organizer_reference=dict(organizer),
    )


def _all_true_mapping(value: object, *, label: str) -> Mapping[str, object]:
    _require(isinstance(value, dict) and bool(value), f"{label} must be a non-empty object")
    assert isinstance(value, dict)
    _require(
        all(isinstance(key, str) and item is True for key, item in value.items()),
        f"{label} must contain only passed boolean checks",
    )
    return value


def _artifact_record(
    value: object,
    *,
    label: str,
    snapshot: Snapshot,
    rows: int | None = None,
) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    expected_keys = {"sha256", "size_bytes"} | ({"rows"} if rows is not None else set())
    _require(set(value) == expected_keys, f"{label} keys differ")
    _require(
        value["sha256"] == snapshot.sha256
        and value["size_bytes"] == len(snapshot.payload)
        and type(value["size_bytes"]) is int,
        f"{label} does not bind exact bytes",
    )
    if rows is not None:
        _require(value["rows"] == rows and type(value["rows"]) is int, f"{label} rows differ")


def _verify_mean_manifest(
    snapshot: Snapshot,
    *,
    snapshots: Mapping[str, Snapshot],
    contract: Contract,
) -> Mapping[str, object]:
    document = _json_object(snapshot.payload, label="accepted mean-ledger manifest")
    _exact_mapping(
        document,
        label="accepted mean-ledger manifest",
        keys={
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
    )
    accepted = contract.accepted_mean_ledger
    _require(document["schema_version"] == 1, "accepted mean-ledger schema differs")
    _require(document["artifact"] == "candidate_activity_mean_ledger_v1", "mean artifact differs")
    _require(document["status"] == "development_mean_only_control", "mean status differs")
    _require(document["automatic_production_eligible"] is False, "mean source promoted itself")
    _require(
        document["allowed_consumer"] == "mean_only_selection_control_only",
        "mean source allowed consumer differs",
    )
    _require(
        document["adapter_config_sha256"] == accepted["config_sha256"],
        "mean source config binding differs",
    )
    _require(
        document["model_release_id"] == accepted["model_release_id"],
        "mean source model release differs",
    )
    _require(document["objectives"] == list(_OBJECTIVES), "mean source objectives differ")
    _require(
        document["scopes"]
        == {
            "calibration": "none_raw_logistic_probability",
            "eligibility": "accepted_candidate_pool_library_eligible_flag_v1",
            "prediction": "declared_seven_target_activity_panel",
            "training": "all_2492_accepted_gate1_contexts_after_recipe_freeze",
            "uncertainty_status": "unavailable",
        },
        "mean source scopes differ",
    )
    ledger = _exact_mapping(
        document["candidate_ledger"],
        label="mean source candidate_ledger",
        keys={"columns", "eligibility_mapping", "filename", "rows", "sha256", "size_bytes"},
    )
    _require(
        ledger["filename"] == "candidate_ledger.csv"
        and ledger["columns"] == list(_SOURCE_COLUMNS)
        and ledger["eligibility_mapping"] == "eligible_is_exact_library_eligible_value",
        "mean source candidate ledger declaration differs",
    )
    _artifact_record(
        {key: ledger[key] for key in ("sha256", "size_bytes", "rows")},
        label="mean source candidate ledger",
        snapshot=snapshots["candidate_ledger.csv"],
        rows=contract.expected_candidates,
    )
    diagnostics = _exact_mapping(
        document["diagnostics"],
        label="mean source diagnostics",
        keys={
            "columns",
            "filename",
            "scope",
            "selectable",
            "uncertainty",
            "rows",
            "sha256",
            "size_bytes",
        },
    )
    _require(
        diagnostics["filename"] == "model_fit_sensitivity_diagnostics.csv"
        and diagnostics["rows"] == contract.expected_candidates
        and diagnostics["selectable"] is False
        and diagnostics["uncertainty"] is False
        and diagnostics["sha256"] == snapshots["model_fit_sensitivity_diagnostics.csv"].sha256
        and diagnostics["size_bytes"]
        == len(snapshots["model_fit_sensitivity_diagnostics.csv"].payload),
        "mean source diagnostic boundary differs",
    )
    diagnostic_columns = diagnostics["columns"]
    _require(
        isinstance(diagnostic_columns, list)
        and any("sensitivity" in str(column) for column in diagnostic_columns)
        and (set(diagnostic_columns) - {"source_ordinal", "sequence_id"}).isdisjoint(
            _SOURCE_COLUMNS
        ),
        "mean source diagnostic columns are not separated",
    )
    _require(
        document["claims"]
        == {
            "diagnostics_are_uncertainty": False,
            "diagnostics_selectable": False,
            "final_ranking": False,
            "mean_only_development_handoff": True,
            "production_ensemble": False,
            "uncertainty_available": False,
        },
        "mean source claims differ",
    )
    source = document["source"]
    _require(isinstance(source, dict), "mean source provenance is invalid")
    assert isinstance(source, dict)
    _require(
        source.get("git_commit") == "acecc557b255382ea41d3610aef6db2858bee768"
        and source.get("artifact") == "candidate_activity_scoring_development_v1",
        "mean source upstream provenance differs",
    )
    _require(
        snapshots["candidate_ledger.csv"].sha256 == accepted["candidate_ledger_sha256"]
        and snapshots["model_fit_sensitivity_diagnostics.csv"].sha256
        == accepted["diagnostics_sha256"]
        and snapshots["manifest.json"].sha256 == accepted["manifest_sha256"]
        and snapshots["SHA256SUMS"].sha256 == accepted["publication_top_sha256"],
        "mean source manifest and observed pins differ",
    )
    return document


def _authenticate_mean_twins(
    paths: Sequence[str | Path], *, contract: Contract
) -> tuple[
    tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
    tuple[tuple[int, int, int, int, int, int, int, int, int], ...],
]:
    _require(len(paths) == 2, "exactly two accepted mean-ledger twins are required")
    absolute = tuple(Path(os.path.abspath(os.fspath(path))) for path in paths)
    _require(absolute[0] != absolute[1], "accepted mean-ledger twin paths must be distinct")
    bundles: list[Mapping[str, Snapshot]] = []
    directory_fingerprints: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    accepted = contract.accepted_mean_ledger
    for index, path in enumerate(absolute):
        snapshots, directory_fingerprint = _authenticate_bundle(
            path,
            expected_files=_MEAN_LEDGER_FILES,
            label=f"accepted mean-ledger twin {index}",
        )
        _require(
            snapshots["SHA256SUMS"].sha256 == accepted["publication_top_sha256"],
            f"accepted mean-ledger twin {index} top hash differs",
        )
        _require(
            _tree_sha256(snapshots) == accepted["publication_tree_sha256"],
            f"accepted mean-ledger twin {index} tree hash differs",
        )
        _verify_mean_manifest(snapshots["manifest.json"], snapshots=snapshots, contract=contract)
        bundles.append(snapshots)
        directory_fingerprints.append(directory_fingerprint)
    for name in sorted(_MEAN_LEDGER_FILES):
        _require(
            bundles[0][name].payload == bundles[1][name].payload,
            f"accepted mean-ledger twins differ at {name}",
        )
    return (bundles[0], bundles[1]), tuple(directory_fingerprints)


def _authenticate_mean_audit(
    path: str | Path,
    *,
    contract: Contract,
    twins: tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
) -> tuple[Mapping[str, Snapshot], tuple[int, int, int, int, int, int, int, int, int]]:
    snapshots, directory_fingerprint = _authenticate_plain_directory(
        path, expected_files=_MEAN_LEDGER_AUDIT_FILES, label="accepted mean-ledger audit"
    )
    accepted = contract.accepted_mean_ledger
    _require(
        snapshots["operational-receipt.json"].sha256 == accepted["operational_audit_sha256"],
        "accepted mean-ledger operational receipt differs",
    )
    for index in range(2):
        name = f"twin-{index}-independent-verification.json"
        _require(
            snapshots[name].sha256 == accepted["independent_verification_sha256"],
            f"mean-ledger twin {index} independent receipt differs",
        )
        receipt = _json_object(snapshots[name].payload, label=f"mean-ledger twin {index} receipt")
        _require(
            receipt.get("schema_version") == 1
            and receipt.get("artifact")
            == "candidate_activity_mean_ledger_v1_independent_verification"
            and receipt.get("status") == "passed"
            and receipt.get("automatic_production_eligible") is False,
            f"mean-ledger twin {index} receipt identity/status differs",
        )
        _all_true_mapping(receipt.get("checks"), label=f"mean-ledger twin {index} checks")
        _require(
            receipt.get("output_sha256")
            == {name_: item.sha256 for name_, item in sorted(twins[index].items())},
            f"mean-ledger twin {index} receipt output binding differs",
        )
        census = receipt.get("census")
        _require(
            isinstance(census, dict)
            and census.get("candidates") == contract.expected_candidates
            and census.get("eligible_candidates") == contract.expected_candidates,
            f"mean-ledger twin {index} receipt census differs",
        )
        claims = receipt.get("claims")
        _require(
            isinstance(claims, dict)
            and claims.get("calibrated_uncertainty_available") is False
            and claims.get("model_fit_sensitivity_is_selectable_uncertainty") is False
            and claims.get("production_ensemble") is False,
            f"mean-ledger twin {index} receipt claims differ",
        )
    _require(
        snapshots["twin-0-independent-verification.json"].payload
        == snapshots["twin-1-independent-verification.json"].payload,
        "accepted mean-ledger twin receipts differ",
    )
    operational = _json_object(
        snapshots["operational-receipt.json"].payload,
        label="accepted mean-ledger operational audit",
    )
    _require(
        operational.get("schema_version") == 1
        and operational.get("artifact") == "candidate_activity_mean_ledger_v1_operational_audit"
        and operational.get("status") == "passed"
        and operational.get("automatic_production_eligible") is False
        and operational.get("git_commit") == accepted["git_commit"]
        and operational.get("uncertainty_status") == "unavailable",
        "mean-ledger operational audit identity/status differs",
    )
    _all_true_mapping(operational.get("checks"), label="mean-ledger operational audit checks")
    producer = operational.get("producer")
    audit = operational.get("audit")
    _require(
        isinstance(producer, dict)
        and producer.get("job_id") == accepted["producer_job_id"]
        and isinstance(audit, dict)
        and audit.get("job_id") == accepted["audit_job_id"],
        "mean-ledger operational audit jobs differ",
    )
    output = operational.get("output")
    _require(isinstance(output, dict) and set(output) == {"0", "1"}, "mean audit output differs")
    assert isinstance(output, dict)
    for index in range(2):
        item = output[str(index)]
        _require(
            isinstance(item, dict)
            and item.get("top_sha256") == twins[index]["SHA256SUMS"].sha256
            and item.get("tree_sha256") == _tree_sha256(twins[index])
            and item.get("files_sha256")
            == {name: snap.sha256 for name, snap in sorted(twins[index].items())},
            f"mean audit twin {index} output binding differs",
        )
    return snapshots, directory_fingerprint


def _canonical_float_text(
    value: str | None, *, label: str, lower: float = 0.0, upper: float = 1.0
) -> float:
    _require(value is not None and value != "", f"{label} is missing")
    assert value is not None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite decimal") from error
    _require(math.isfinite(parsed) and lower <= parsed <= upper, f"{label} is out of range")
    _require(format(parsed, ".17g") == value, f"{label} is not canonical float text")
    return parsed


def _canonical_lineage(value: str, *, label: str) -> tuple[str, ...]:
    tokens = value.split("|")
    _require(
        bool(tokens)
        and all(_LINEAGE_TOKEN_RE.fullmatch(token) is not None for token in tokens)
        and tokens == sorted(set(tokens)),
        f"{label} must contain canonical sorted unique lineage tokens",
    )
    return tuple(tokens)


def _validate_sequence(sequence: object, sequence_id: object, *, label: str) -> tuple[str, str]:
    _require(
        isinstance(sequence, str)
        and bool(sequence)
        and sequence == sequence.strip().upper()
        and set(sequence) <= _AMINO_ACIDS,
        f"{label} sequence is invalid",
    )
    assert isinstance(sequence, str)
    expected_id = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    _require(sequence_id == expected_id, f"{label} sequence identity differs")
    assert isinstance(sequence_id, str)
    return sequence, sequence_id


def _net_charge(sequence: str, *, ph: float) -> float:
    positive = 1.0 / (1.0 + 10.0 ** (ph - _N_TERMINUS_PKA))
    negative = 1.0 / (1.0 + 10.0 ** (_C_TERMINUS_PKA - ph))
    for residue, pka in _POSITIVE_SIDECHAIN_PKA.items():
        positive += sequence.count(residue) / (1.0 + 10.0 ** (ph - pka))
    for residue, pka in _NEGATIVE_SIDECHAIN_PKA.items():
        negative += sequence.count(residue) / (1.0 + 10.0 ** (pka - ph))
    return positive - negative


def _isoelectric_point(
    sequence: str, *, lower_ph: float, upper_ph: float, iterations: int
) -> float:
    lower = float(lower_ph)
    upper = float(upper_ph)
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        if _net_charge(sequence, ph=midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _hydrophobic_moment(sequence: str, *, angle_degrees: float) -> float:
    angles = np.deg2rad(np.arange(len(sequence), dtype=np.float64) * angle_degrees)
    hydrophobicities = np.asarray(
        [_EISENBERG_HYDROPHOBICITY[residue] for residue in sequence], dtype=np.float64
    )
    x_component = float(np.sum(hydrophobicities * np.cos(angles)))
    y_component = float(np.sum(hydrophobicities * np.sin(angles)))
    return math.hypot(x_component, y_component) / len(sequence)


def _shannon_entropy(sequence: str) -> float:
    counts = np.asarray(
        [sequence.count(residue) for residue in sorted(set(sequence))], dtype=np.float64
    )
    probabilities = counts / len(sequence)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _raw_features(sequence: str, *, contract: Contract) -> tuple[float, ...]:
    length = len(sequence)
    charge = _net_charge(sequence, ph=contract.descriptor_ph)
    hydrophobicities = [_EISENBERG_HYDROPHOBICITY[residue] for residue in sequence]
    counts = {residue: sequence.count(residue) for residue in set(sequence)}
    descriptors = (
        float(length),
        float(sum(_RESIDUE_MASSES_DA[residue] for residue in sequence) + _WATER_MASS_DA),
        charge,
        charge / length,
        _isoelectric_point(
            sequence,
            lower_ph=contract.isoelectric_point_lower_ph,
            upper_ph=contract.isoelectric_point_upper_ph,
            iterations=contract.isoelectric_point_iterations,
        ),
        float(np.mean(hydrophobicities)),
        _hydrophobic_moment(sequence, angle_degrees=contract.hydrophobic_moment_angle_degrees),
        sum(residue in _HYDROPHOBIC_RESIDUES for residue in sequence) / length,
        sum(residue in _AROMATIC_RESIDUES for residue in sequence) / length,
        sum(residue in _BASIC_RESIDUES for residue in sequence) / length,
        sum(residue in _ACIDIC_RESIDUES for residue in sequence) / length,
        _shannon_entropy(sequence),
        max(counts.values()) / length,
    )
    composition = tuple(sequence.count(residue) / length for residue in _ALPHABET)
    result = descriptors + composition
    _require(
        len(result) == len(_FEATURE_NAMES) and all(math.isfinite(value) for value in result),
        "feature calculation produced an invalid row",
    )
    return result


def _parse_candidates(snapshot: Snapshot, *, contract: Contract) -> tuple[Candidate, ...]:
    payload = snapshot.payload
    _require(
        bool(payload)
        and payload.endswith(b"\n")
        and not payload.endswith(b"\n\n")
        and b"\r" not in payload
        and b"\x00" not in payload,
        "accepted mean ledger must be canonical LF-framed text",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("accepted mean ledger must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    _require(reader.fieldnames == list(_SOURCE_COLUMNS), "accepted mean ledger header differs")
    candidates: list[Candidate] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for row_number, row in enumerate(reader, 1):
        _require(None not in row, f"accepted mean-ledger row {row_number} has extra columns")
        _require(row_number <= contract.expected_candidates, "accepted mean ledger has extra rows")
        _require(
            all(row.get(column) is not None for column in _SOURCE_COLUMNS),
            f"accepted mean-ledger row {row_number} has missing columns",
        )
        source_ordinal = row["source_ordinal"]
        _require(source_ordinal == str(row_number), f"mean-ledger row {row_number} ordinal differs")
        sequence, sequence_id = _validate_sequence(
            row["sequence"], row["sequence_id"], label=f"mean-ledger row {row_number}"
        )
        _require(
            row["length"] == str(len(sequence)), f"mean-ledger row {row_number} length differs"
        )
        _require(
            sequence_id not in seen_ids and sequence not in seen_sequences,
            f"mean-ledger row {row_number} duplicates a candidate",
        )
        seen_ids.add(sequence_id)
        seen_sequences.add(sequence)
        _require(row["eligible"] in {"true", "false"}, f"row {row_number} eligibility differs")
        _require(
            row["eligibility_scope"] == "accepted_candidate_pool_library_eligible_flag_v1",
            f"row {row_number} eligibility scope differs",
        )
        _canonical_lineage(row["generator_families"], label=f"row {row_number} families")
        _canonical_lineage(row["generator_variants"], label=f"row {row_number} variants")
        probabilities: dict[str, float] = {}
        for target in _TARGETS:
            probabilities[target] = _canonical_float_text(
                row[f"probability_{target}"], label=f"row {row_number} probability {target}"
            )
        target_groups = {
            "broad_spectrum_activity": _TARGETS,
            "gram_positive_activity": (
                "enterococcus_faecalis",
                "enterococcus_faecium",
                "staphylococcus_aureus",
            ),
            "gram_negative_activity": (
                "acinetobacter_baumannii",
                "escherichia_coli",
                "klebsiella_pneumoniae",
                "pseudomonas_aeruginosa",
            ),
        }
        for objective, targets in target_groups.items():
            observed = _canonical_float_text(
                row[f"mean_{objective}"], label=f"row {row_number} mean {objective}"
            )
            expected = sum(probabilities[target] for target in targets) / len(targets)
            _require(
                abs(observed - expected) <= 1e-12,
                f"row {row_number} mean {objective} differs from target arithmetic",
            )
        _require(row["uncertainty_status"] == "unavailable", f"row {row_number} adds uncertainty")
        _require(
            row["prediction_scope"] == "declared_seven_target_activity_panel"
            and row["training_scope"] == "all_2492_accepted_gate1_contexts_after_recipe_freeze"
            and row["calibration_scope"] == "none_raw_logistic_probability"
            and row["model_release_id"] == contract.accepted_mean_ledger["model_release_id"],
            f"row {row_number} source scopes or model release differ",
        )
        candidates.append(
            Candidate(source=dict(row), raw_features=_raw_features(sequence, contract=contract))
        )
    _require(
        len(candidates) == contract.expected_candidates,
        f"accepted mean ledger has {len(candidates)} rows; expected {contract.expected_candidates}",
    )
    return tuple(candidates)


def _parse_projection_receipt(
    snapshot: Snapshot, *, projection_snapshot: Snapshot, contract: Contract
) -> Mapping[str, object]:
    accepted = contract.accepted_training_projection
    _require(snapshot.sha256 == accepted["receipt_sha256"], "training receipt hash differs")
    _require(
        len(snapshot.payload) == accepted["receipt_size_bytes"], "training receipt size differs"
    )
    document = _json_object(snapshot.payload, label="accepted training projection receipt")
    _exact_mapping(
        document,
        label="accepted training projection receipt",
        keys={
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
    )
    _require(
        document["schema_version"] == 1
        and document["artifact"] == "native_diffusion_training_projection_stage_v1"
        and document["status"] == "passed"
        and document["git_commit"] == accepted["git_commit"]
        and document["stage_job_id"] == str(accepted["stage_job_id"]),
        "training projection receipt identity differs",
    )
    _all_true_mapping(document["checks"], label="training projection receipt checks")
    projection = _exact_mapping(
        document["projection"],
        label="training projection receipt projection",
        keys={"fields", "filename", "rows", "sha256"},
    )
    _require(
        projection["fields"] == accepted["fields"]
        and projection["filename"] == "training_projection.jsonl"
        and projection["rows"] == contract.expected_training_rows
        and projection["sha256"] == projection_snapshot.sha256,
        "training projection receipt binding differs",
    )
    return document


def _parse_training_projection(
    snapshot: Snapshot, *, contract: Contract
) -> tuple[TrainingRow, ...]:
    accepted = contract.accepted_training_projection
    _require(snapshot.sha256 == accepted["projection_sha256"], "training projection hash differs")
    _require(
        len(snapshot.payload) == accepted["projection_size_bytes"],
        "training projection size differs",
    )
    payload = snapshot.payload
    _require(
        bool(payload)
        and payload.endswith(b"\n")
        and not payload.endswith(b"\n\n")
        and b"\r" not in payload,
        "training projection must be canonical LF-delimited JSONL",
    )
    rows: list[TrainingRow] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for number, raw_line in enumerate(payload[:-1].split(b"\n"), 1):
        _require(bool(raw_line), f"training projection row {number} is blank")
        document = _json_object(raw_line + b"\n", label=f"training projection row {number}")
        _exact_mapping(
            document,
            label=f"training projection row {number}",
            keys={"sequence_id", "sequence", "sampling_weight"},
        )
        sequence, sequence_id = _validate_sequence(
            document["sequence"], document["sequence_id"], label=f"training projection row {number}"
        )
        weight = document["sampling_weight"]
        _require(
            type(weight) is float and math.isfinite(weight) and weight > 0.0,
            f"row {number} weight invalid",
        )
        assert isinstance(weight, float)
        _require(
            sequence_id not in seen_ids and sequence not in seen_sequences,
            "training projection duplicates sequence",
        )
        _require(
            not rows or rows[-1].sequence_id < sequence_id,
            "training projection is not sequence-ID sorted",
        )
        seen_ids.add(sequence_id)
        seen_sequences.add(sequence)
        rows.append(
            TrainingRow(
                sequence_id=sequence_id,
                sequence=sequence,
                weight=weight,
                raw_features=_raw_features(sequence, contract=contract),
            )
        )
    _require(len(rows) == contract.expected_training_rows, "training projection row census differs")
    _require(
        math.isclose(math.fsum(row.weight for row in rows), 1.0, rel_tol=0.0, abs_tol=1e-15),
        "training projection weights do not sum to one",
    )
    return tuple(rows)


def _parse_reference(snapshot: Snapshot, *, contract: Contract) -> tuple[tuple[str, str], ...]:
    accepted = contract.organizer_reference
    _require(snapshot.sha256 == accepted["sha256"], "organizer reference hash differs")
    _require(len(snapshot.payload) == accepted["size_bytes"], "organizer reference size differs")
    payload = snapshot.payload
    _require(
        bool(payload)
        and payload.endswith(b"\n")
        and b"\r" not in payload
        and b"\x00" not in payload,
        "organizer reference must be LF-framed text",
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("organizer reference must be ASCII FASTA") from error
    records: list[str] = []
    current: list[str] | None = None
    for number, line in enumerate(lines, 1):
        _require(bool(line), f"organizer reference line {number} is blank")
        if line.startswith(">"):
            _require(len(line) > 1, f"organizer reference header {number} is empty")
            if current is not None:
                _require(bool(current), f"organizer reference record before line {number} is empty")
                records.append("".join(current))
            current = []
        else:
            _require(current is not None, "organizer reference sequence precedes first header")
            current.append(line)
    _require(current is not None and bool(current), "organizer reference final record is empty")
    records.append("".join(current))
    _require(
        len(records) == contract.expected_reference_records,
        "organizer reference record census differs",
    )
    unique: dict[str, str] = {}
    for number, sequence_value in enumerate(records, 1):
        sequence, sequence_id = _validate_sequence(
            sequence_value,
            hashlib.sha256(sequence_value.encode("ascii")).hexdigest(),
            label=f"organizer reference record {number}",
        )
        previous = unique.setdefault(sequence_id, sequence)
        _require(previous == sequence, "organizer reference sequence-ID collision")
    return tuple(sorted(unique.items()))


def _length_similarity_upper_bound(left_length: int, right_length: int) -> float:
    return (2.0 * min(left_length, right_length)) / (left_length + right_length)


def _reference_choices_by_candidate_length(
    references: Sequence[tuple[str, str]],
    *,
    candidate_lengths: Sequence[int],
    threshold: float,
) -> Mapping[int, tuple[tuple[str, ...], tuple[str, ...]]]:
    _require(threshold == 0.80, "v1 reference threshold must be exactly 0.80")
    ordered_references = tuple(sorted(references))
    tables: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for candidate_length in sorted(set(candidate_lengths)):
        selected = tuple(
            (sequence_id, sequence)
            for sequence_id, sequence in ordered_references
            if 10 * min(candidate_length, len(sequence)) > 4 * (candidate_length + len(sequence))
        )
        tables[candidate_length] = (
            tuple(sequence_id for sequence_id, _ in selected),
            tuple(sequence for _, sequence in selected),
        )
    return tables


def _one_similarity_result(
    candidate_sequence: str,
    *,
    training_ids: Sequence[str],
    training_sequences: Sequence[str],
    reference_tables: Mapping[int, tuple[tuple[str, ...], tuple[str, ...]]],
    threshold: float,
) -> SimilarityResult:
    train_match = process.extractOne(
        candidate_sequence,
        training_sequences,
        scorer=Indel.normalized_similarity,
        processor=None,
    )
    _require(train_match is not None, "training similarity search returned no match")
    assert train_match is not None
    _, max_train_similarity_raw, train_index = train_match
    max_train_similarity = float(max_train_similarity_raw)
    _require(
        type(train_index) is int and 0 <= train_index < len(training_ids),
        "training similarity search returned an invalid index",
    )
    nearest_train_id = training_ids[train_index]
    recomputed_train = float(
        Indel.normalized_similarity(candidate_sequence, training_sequences[train_index])
    )
    _require(
        recomputed_train == max_train_similarity and 0.0 <= max_train_similarity <= 1.0,
        "training similarity result is inconsistent",
    )

    reference_ids, reference_sequences = reference_tables[len(candidate_sequence)]
    strict_cutoff = math.nextafter(threshold, math.inf)
    witness_id = ""
    witness_similarity: float | None = None
    for _, raw_score, reference_index in process.extract_iter(
        candidate_sequence,
        reference_sequences,
        scorer=Indel.normalized_similarity,
        processor=None,
        score_cutoff=strict_cutoff,
    ):
        _require(
            type(reference_index) is int and 0 <= reference_index < len(reference_ids),
            "reference similarity search returned an invalid index",
        )
        witness_id = reference_ids[reference_index]
        iterator_score = float(raw_score)
        exact_score = float(
            Indel.normalized_similarity(candidate_sequence, reference_sequences[reference_index])
        )
        _require(
            iterator_score == exact_score and exact_score > threshold,
            "reference witness does not strictly violate the threshold",
        )
        witness_similarity = exact_score
        break
    if not witness_id:
        _require(witness_similarity is None, "safe reference result carries a witness score")
    return SimilarityResult(
        max_train_similarity=max_train_similarity,
        nearest_train_sequence_id=nearest_train_id,
        reference_safe=witness_id == "",
        reference_witness_sequence_id=witness_id,
        reference_witness_similarity=witness_similarity,
    )


def _initialize_similarity_worker(state: _SimilarityWorkerState) -> None:
    global _SIMILARITY_WORKER_STATE
    _SIMILARITY_WORKER_STATE = state


def _evaluate_similarity_sequence(candidate_sequence: str) -> SimilarityResult:
    state = _SIMILARITY_WORKER_STATE
    if state is None:
        raise RuntimeError("similarity worker started without immutable verifier state")
    return _one_similarity_result(
        candidate_sequence,
        training_ids=state.training_ids,
        training_sequences=state.training_sequences,
        reference_tables=state.reference_tables,
        threshold=state.threshold,
    )


def _process_chunksize(item_count: int, workers: int) -> int:
    # Bound queueing/pickling overhead while retaining at least 32 chunks per
    # worker for the highly variable first-witness stopping time.
    return max(1, min(_MAX_PROCESS_CHUNKSIZE, math.ceil(item_count / (32 * workers))))


def _compute_similarities(
    candidates: Sequence[Candidate],
    training: Sequence[TrainingRow],
    references: Sequence[tuple[str, str]],
    *,
    threshold: float,
    workers: int,
) -> tuple[SimilarityResult, ...]:
    _require(type(workers) is int and workers >= 1, "workers must be a positive integer")
    ordered_training = tuple(sorted(training, key=lambda row: row.sequence_id))
    training_ids = tuple(row.sequence_id for row in ordered_training)
    training_sequences = tuple(row.sequence for row in ordered_training)
    _require(bool(training_ids), "training similarity choices cannot be empty")
    reference_tables = _reference_choices_by_candidate_length(
        references,
        candidate_lengths=tuple(len(candidate.sequence) for candidate in candidates),
        threshold=threshold,
    )
    candidate_sequences = tuple(candidate.sequence for candidate in candidates)
    state = _SimilarityWorkerState(
        training_ids=training_ids,
        training_sequences=training_sequences,
        reference_tables=reference_tables,
        threshold=threshold,
    )
    if workers == 1:
        results = tuple(
            _one_similarity_result(
                sequence,
                training_ids=training_ids,
                training_sequences=training_sequences,
                reference_tables=reference_tables,
                threshold=threshold,
            )
            for sequence in candidate_sequences
        )
    else:
        _require(
            "fork" in multiprocessing.get_all_start_methods(),
            "parallel independent verification requires the POSIX fork start method",
        )
        fork_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=fork_context,
            initializer=_initialize_similarity_worker,
            initargs=(state,),
        ) as executor:
            # ProcessPoolExecutor.map preserves input order.  Only short query
            # strings and compact results cross process boundaries; the large
            # immutable choice tables are inherited copy-on-write at fork.
            results = tuple(
                executor.map(
                    _evaluate_similarity_sequence,
                    candidate_sequences,
                    chunksize=_process_chunksize(len(candidate_sequences), workers),
                )
            )
    _require(len(results) == len(candidates), "similarity evaluation lost candidate rows")
    return results


def _fit_feature_transform(
    training: Sequence[TrainingRow],
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    total_weight = math.fsum(row.weight for row in training)
    _require(math.isfinite(total_weight) and total_weight > 0.0, "training weight sum is invalid")
    means = tuple(
        math.fsum(row.weight * row.raw_features[index] for row in training) / total_weight
        for index in range(len(_FEATURE_NAMES))
    )
    scales = tuple(
        math.sqrt(
            math.fsum(
                row.weight * (row.raw_features[index] - means[index]) ** 2 for row in training
            )
            / total_weight
        )
        for index in range(len(_FEATURE_NAMES))
    )
    _require(
        all(math.isfinite(value) for value in means)
        and all(math.isfinite(value) and value > 0.0 for value in scales),
        "training feature transform contains a non-finite or zero scale",
    )
    return means, scales, total_weight


def _normalized_candidate_features(
    candidates: Sequence[Candidate],
    *,
    means: Sequence[float],
    scales: Sequence[float],
) -> np.ndarray:
    normalized: list[tuple[float, ...]] = []
    for row_number, candidate in enumerate(candidates, 1):
        standardized = tuple(
            (candidate.raw_features[index] - means[index]) / scales[index]
            for index in range(len(_FEATURE_NAMES))
        )
        norm = math.sqrt(math.fsum(value * value for value in standardized))
        _require(
            math.isfinite(norm) and norm > 0.0,
            f"candidate {row_number} standardized feature norm is invalid",
        )
        normalized.append(tuple(value / norm for value in standardized))
    matrix = np.asarray(normalized, dtype=np.dtype("<f4"), order="C")
    _require(
        matrix.shape == (len(candidates), len(_FEATURE_NAMES))
        and matrix.dtype == np.dtype("<f4")
        and matrix.flags.c_contiguous
        and np.isfinite(matrix).all(),
        "final diversity matrix contract differs",
    )
    return matrix


def _float32_text(value: np.float32) -> str:
    _require(isinstance(value, np.float32) and np.isfinite(value), "matrix value is invalid")
    text = str(value)
    _require(np.float32(float(text)).tobytes() == value.tobytes(), "float32 text is not roundtrip")
    return text


def _csv_payload(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    _require(b"\r" not in payload and payload.endswith(b"\n"), "generated CSV is not LF-framed")
    return payload


def _semantic_digest_rows(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for values in rows:
        digest.update(_canonical_json(dict(zip(columns, values, strict=True))))
    return digest.hexdigest()


def _lineage_census(
    candidates: Sequence[Candidate], similarities: Sequence[SimilarityResult]
) -> Mapping[str, object]:
    def summarize(column: str) -> Mapping[str, object]:
        counts: dict[str, list[int]] = {}
        for candidate, similarity in zip(candidates, similarities, strict=True):
            source = candidate.source["eligible"] == "true"
            safe = similarity.reference_safe
            effective = source and safe
            for token in _canonical_lineage(candidate.source[column], label=f"source {column}"):
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

    return {
        "families": summarize("generator_families"),
        "variants": summarize("generator_variants"),
    }


def _fixed_quantiles(values: Sequence[float]) -> list[Mapping[str, float]]:
    _require(values and all(math.isfinite(value) for value in values), "quantile values invalid")
    ordered = sorted(values)
    result: list[Mapping[str, float]] = []
    for probability in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
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


def _reconstruct_payloads(
    candidates: Sequence[Candidate],
    training: Sequence[TrainingRow],
    references: Sequence[tuple[str, str]],
    *,
    contract: Contract,
    workers: int,
) -> Reconstruction:
    means, scales, total_weight = _fit_feature_transform(training)
    _require(
        math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-15),
        "accepted training weight total differs from one",
    )
    matrix = _normalized_candidate_features(candidates, means=means, scales=scales)
    candidate_raw_matrix = np.asarray(
        [candidate.raw_features for candidate in candidates], dtype="<f8", order="C"
    )
    training_raw_matrix = np.asarray([row.raw_features for row in training], dtype="<f8", order="C")
    training_weights = np.asarray([row.weight for row in training], dtype="<f8", order="C")
    matrix_payload = matrix.tobytes(order="C")
    _require(
        len(matrix_payload) == len(candidates) * len(_FEATURE_NAMES) * 4,
        "feature matrix byte count differs",
    )
    similarities = _compute_similarities(
        candidates,
        training,
        references,
        threshold=contract.threshold,
        workers=workers,
    )
    _require(len(similarities) == len(candidates), "similarity row count differs")
    output_rows: list[list[str]] = []
    source_eligible_rows = 0
    reference_safe_rows = 0
    effective_eligible_rows = 0
    for index, (candidate, similarity) in enumerate(zip(candidates, similarities, strict=True)):
        _require(
            similarity.max_train_similarity < 1.0,
            f"candidate {index + 1} exactly overlaps the accepted training projection",
        )
        source_eligible = candidate.source["eligible"] == "true"
        effective_eligible = source_eligible and similarity.reference_safe
        source_eligible_rows += int(source_eligible)
        reference_safe_rows += int(similarity.reference_safe)
        effective_eligible_rows += int(effective_eligible)
        if similarity.reference_safe:
            _require(
                similarity.reference_witness_sequence_id == ""
                and similarity.reference_witness_similarity is None,
                f"safe candidate {index + 1} carries a reference witness",
            )
            witness_similarity_text = ""
        else:
            _require(
                _SHA256_RE.fullmatch(similarity.reference_witness_sequence_id) is not None
                and similarity.reference_witness_similarity is not None
                and similarity.reference_witness_similarity > contract.threshold,
                f"unsafe candidate {index + 1} lacks an exact violating witness",
            )
            witness_similarity_text = format(similarity.reference_witness_similarity, ".17g")
        preserved = [
            candidate.source["eligible"]
            if column == "source_library_eligible"
            else candidate.source[column]
            for column in _PRESERVED_COLUMNS
        ]
        novelty = 1.0 - similarity.max_train_similarity
        novelty_values = [
            format(similarity.max_train_similarity, ".17g"),
            similarity.nearest_train_sequence_id,
            format(novelty, ".17g"),
            "true" if similarity.reference_safe else "false",
            similarity.reference_witness_sequence_id,
            witness_similarity_text,
            "true" if effective_eligible else "false",
        ]
        feature_values = [_float32_text(matrix[index, column]) for column in range(matrix.shape[1])]
        output_rows.append(preserved + novelty_values + feature_values)
    ledger_payload = _csv_payload(_OUTPUT_COLUMNS, output_rows)
    ledger_semantic_sha256 = _semantic_digest_rows(_OUTPUT_COLUMNS, output_rows)
    matrix_semantic_sha256 = hashlib.sha256(
        b"".join(
            _canonical_json(
                {
                    "source_ordinal": candidates[index].source["source_ordinal"],
                    "sequence_id": candidates[index].source["sequence_id"],
                    "values": [_float32_text(value) for value in matrix[index]],
                }
            )
            for index in range(len(candidates))
        )
    ).hexdigest()
    # The exact transform payload is attached after the frozen output schema is
    # assembled from authenticated input identities.
    return Reconstruction(
        ledger_payload=ledger_payload,
        matrix_payload=matrix_payload,
        transform_document={},
        transform_payload=b"",
        rows=len(candidates),
        source_eligible_rows=source_eligible_rows,
        reference_safe_rows=reference_safe_rows,
        effective_eligible_rows=effective_eligible_rows,
        ledger_semantic_sha256=ledger_semantic_sha256,
        matrix_semantic_sha256=matrix_semantic_sha256,
        feature_means=means,
        feature_scales=scales,
        candidate_raw_feature_sha256=hashlib.sha256(
            candidate_raw_matrix.tobytes(order="C")
        ).hexdigest(),
        training_raw_feature_sha256=hashlib.sha256(
            training_raw_matrix.tobytes(order="C")
        ).hexdigest(),
        training_weight_sha256=hashlib.sha256(training_weights.tobytes(order="C")).hexdigest(),
        training_weight_sum=total_weight,
        similarities=similarities,
        lineage_census=_lineage_census(candidates, similarities),
    )


def _feature_transform_document(
    *,
    config_snapshot: Snapshot,
    contract: Contract,
    reconstruction: Reconstruction,
) -> Mapping[str, object]:
    return {
        "schema_version": 1,
        "artifact": "candidate_diversity_feature_transform_v1",
        "status": _OUTPUT_STATUS,
        "automatic_production_eligible": False,
        "config_sha256": config_snapshot.sha256,
        "feature_space": _FEATURE_SPACE,
        "feature_names": list(_FEATURE_NAMES),
        "embedding_columns": list(_EMBEDDING_COLUMNS),
        "descriptor_settings": {
            "descriptor_dataclass_order": "PeptideDescriptors_field_order",
            "ph": contract.descriptor_ph,
            "free_termini": True,
            "hydrophobic_moment_angle_degrees": contract.hydrophobic_moment_angle_degrees,
            "isoelectric_point": {
                "iterations": contract.isoelectric_point_iterations,
                "lower_ph": contract.isoelectric_point_lower_ph,
                "upper_ph": contract.isoelectric_point_upper_ph,
            },
            "residue_fraction_order": _ALPHABET,
        },
        "fit": {
            "rows": contract.expected_training_rows,
            "source": "accepted_training_projection_only",
            "source_sha256": contract.accepted_training_projection["projection_sha256"],
            "weight_field": "sampling_weight",
            "weight_sum": reconstruction.training_weight_sum,
            "weight_sum_hex": reconstruction.training_weight_sum.hex(),
            "weights_float64_little_endian_sha256": reconstruction.training_weight_sha256,
            "training_raw_feature_matrix_float64_little_endian_sha256": (
                reconstruction.training_raw_feature_sha256
            ),
            "weighted_means": list(reconstruction.feature_means),
            "weighted_population_scales": list(reconstruction.feature_scales),
            "weighted_mean_algorithm": (
                "math_fsum_of_weight_times_value_divided_by_math_fsum_weights"
            ),
            "weighted_scale_algorithm": (
                "population_sd_sqrt_math_fsum_weight_times_squared_centered_value_"
                "divided_by_math_fsum_weights"
            ),
        },
        "transform": {
            "standardization": (
                "candidate_raw_minus_training_weighted_mean_divided_by_"
                "training_weighted_population_sd"
            ),
            "row_normalization": "euclidean_l2_after_standardization",
            "zero_row_norm_policy": "fail_closed",
            "zero_scale_policy": "fail_closed",
        },
        "candidate_raw_feature_matrix": {
            "byte_order": "little_endian",
            "dtype": "float64",
            "memory_order": "C",
            "shape": [contract.expected_candidates, len(_FEATURE_NAMES)],
            "sha256": reconstruction.candidate_raw_feature_sha256,
        },
        "output_matrix": {
            "byte_order": "little_endian",
            "dtype": "float32",
            "filename": "diversity_feature_matrix.f32le",
            "memory_order": "C",
            "raw_bytes_sha256": hashlib.sha256(reconstruction.matrix_payload).hexdigest(),
            "shape": [contract.expected_candidates, len(_FEATURE_NAMES)],
            "size_bytes": len(reconstruction.matrix_payload),
        },
    }


def _payload_record(payload: bytes, *, rows: int | None = None) -> Mapping[str, object]:
    result: dict[str, object] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _manifest_document(
    *,
    config_snapshot: Snapshot,
    contract: Contract,
    reconstruction: Reconstruction,
    reference_records: int,
    unique_references: int,
) -> Mapping[str, object]:
    maxima = [item.max_train_similarity for item in reconstruction.similarities]
    novelty = [1.0 - value for value in maxima]
    witness_count = sum(
        bool(item.reference_witness_sequence_id) for item in reconstruction.similarities
    )
    return {
        "schema_version": 1,
        "artifact": _OUTPUT_ARTIFACT,
        "status": _OUTPUT_STATUS,
        "automatic_production_eligible": False,
        "allowed_consumer": _ALLOWED_CONSUMER,
        "config_sha256": config_snapshot.sha256,
        "inputs": {
            "mean_ledger": dict(contract.accepted_mean_ledger),
            "training_projection": {
                key: contract.accepted_training_projection[key]
                for key in (
                    "stage_job_id",
                    "git_commit",
                    "projection_sha256",
                    "projection_size_bytes",
                    "receipt_sha256",
                    "receipt_size_bytes",
                )
            }
            | {"rows": contract.expected_training_rows},
            "organizer_reference": {
                "sha256": contract.organizer_reference["sha256"],
                "size_bytes": contract.organizer_reference["size_bytes"],
                "source_commit": contract.organizer_reference["source_commit"],
                "record_count": reference_records,
                "unique_sequence_count": unique_references,
                "role": contract.organizer_reference["role"],
            },
        },
        "candidate_selection_ledger": {
            "filename": "candidate_selection_ledger.csv",
            "columns": list(_OUTPUT_COLUMNS),
            "eligibility_mapping": (
                "eligible_equals_source_library_eligible_and_reference_safe_0_80"
            ),
            **_payload_record(reconstruction.ledger_payload, rows=contract.expected_candidates),
        },
        "diversity_feature_matrix": {
            "filename": "diversity_feature_matrix.f32le",
            "feature_space": _FEATURE_SPACE,
            "shape": [contract.expected_candidates, len(_FEATURE_NAMES)],
            "dtype": "float32",
            "byte_order": "little_endian",
            "memory_order": "C",
            "csv_columns": list(_EMBEDDING_COLUMNS),
            **_payload_record(reconstruction.matrix_payload),
        },
        "feature_transform": {
            "filename": "feature_transform.json",
            **_payload_record(reconstruction.transform_payload),
        },
        "similarity": {
            "implementation": "rapidfuzz.distance.Indel.normalized_similarity",
            "value_range": "zero_to_one",
            "training_reduction": "exact_maximum_over_all_accepted_training_sequences",
            "nearest_training_tie_break": "lexicographically_smallest_sequence_id",
            "novelty": "one_minus_max_train_indel_similarity",
            "reference_rule": (
                "unsafe_if_any_unique_organizer_reference_similarity_strictly_exceeds_threshold"
            ),
            "reference_similarity_threshold": contract.threshold,
            "reference_threshold_equality": "safe",
            "reference_witness_tie_break": "lexicographically_smallest_sequence_id",
            "reference_maximum_similarity_calculated_or_stored": False,
            "length_pruning": "upper_bound_two_times_min_length_divided_by_sum_lengths",
            "candidate_chunk_size": 1024,
            "choice_chunk_size": 1024,
            "worker_bounds_inclusive": [contract.worker_minimum, contract.worker_maximum],
            "worker_count_affects_output_bytes": False,
        },
        "census": {
            "candidates": contract.expected_candidates,
            "source_library_eligible": reconstruction.source_eligible_rows,
            "reference_safe": reconstruction.reference_safe_rows,
            "reference_unsafe_witnesses": witness_count,
            "effective_eligible": reconstruction.effective_eligible_rows,
            "training_similarity": {
                "minimum": min(maxima),
                "maximum": max(maxima),
                "quantile_method": (
                    "sorted_linear_interpolation_h_equals_probability_times_n_minus_one"
                ),
                "quantiles": _fixed_quantiles(maxima),
            },
            "novelty": {
                "minimum": min(novelty),
                "maximum": max(novelty),
                "quantile_method": (
                    "sorted_linear_interpolation_h_equals_probability_times_n_minus_one"
                ),
                "quantiles": _fixed_quantiles(novelty),
            },
        },
        "lineage_census": reconstruction.lineage_census,
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
            "candidate_ledger_column_prefixes": [
                "cluster_",
                "model_fit_sensitivity_",
                "std_",
            ],
            "model_fit_sensitivity_diagnostics_consumed": False,
            "reference_maximum_similarity_field": False,
        },
    }


def _validate_output_header_boundary(payload: bytes) -> None:
    try:
        header = next(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, StopIteration, csv.Error) as error:
        raise ValueError("candidate selection ledger header is invalid") from error
    _require(header == list(_OUTPUT_COLUMNS), "candidate selection ledger schema differs")
    for column in header:
        lowered = column.lower()
        if column == "uncertainty_status":
            continue
        _require(
            "uncertainty" not in lowered
            and "esm" not in lowered
            and "cluster" not in lowered
            and all(token not in lowered for token in _FORBIDDEN_COLUMN_TOKENS),
            f"candidate selection ledger exposes prohibited column {column!r}",
        )
    _require(
        "max_reference_indel_similarity" not in header,
        "candidate selection ledger must not claim a reference maximum",
    )


def verify_candidate_diversity_novelty(
    *,
    mean_ledger_twin_paths: Sequence[str | Path],
    mean_ledger_audit_dir: str | Path,
    training_projection_path: str | Path,
    training_projection_receipt_path: str | Path,
    organizer_reference_path: str | Path,
    config_path: str | Path,
    bundle_dirs: Sequence[str | Path],
    workers: int = 1,
) -> Mapping[str, object]:
    """Reconstruct once and authenticate both five-file publication twins."""

    config_snapshot = _snapshot(config_path, label="diversity/novelty config")
    contract = _parse_config(config_snapshot)
    _require(
        type(workers) is int and contract.worker_minimum <= workers <= contract.worker_maximum,
        "workers must lie inside the frozen inclusive bounds",
    )
    _require(len(bundle_dirs) == 2, "exactly two output bundle directories are required")
    output_roots = tuple(Path(os.path.abspath(os.fspath(path))) for path in bundle_dirs)
    _require(output_roots[0] != output_roots[1], "output bundle paths must be distinct")
    output_twins_list: list[Mapping[str, Snapshot]] = []
    output_directory_fingerprint_list: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    for twin_index, output_root in enumerate(output_roots):
        output, output_directory_fingerprint = _authenticate_bundle(
            output_root,
            expected_files=_OUTPUT_FILES,
            label=f"candidate diversity/novelty output twin {twin_index}",
        )
        _validate_output_header_boundary(output["candidate_selection_ledger.csv"].payload)
        output_twins_list.append(output)
        output_directory_fingerprint_list.append(output_directory_fingerprint)
    output_twins = tuple(output_twins_list)
    output_directory_fingerprints = tuple(output_directory_fingerprint_list)
    _require(
        (output_directory_fingerprints[0][0], output_directory_fingerprints[0][1])
        != (output_directory_fingerprints[1][0], output_directory_fingerprints[1][1]),
        "output bundle directories must not alias the same device/inode",
    )
    for name in sorted(_OUTPUT_FILES):
        _require(
            output_twins[0][name].payload == output_twins[1][name].payload,
            f"output publication twins differ at {name}",
        )

    mean_twins, mean_directory_fingerprints = _authenticate_mean_twins(
        mean_ledger_twin_paths, contract=contract
    )
    mean_audit, mean_audit_directory_fingerprint = _authenticate_mean_audit(
        mean_ledger_audit_dir, contract=contract, twins=mean_twins
    )
    candidates = _parse_candidates(mean_twins[0]["candidate_ledger.csv"], contract=contract)

    projection_path = Path(os.path.abspath(os.fspath(training_projection_path)))
    projection_receipt_path = Path(os.path.abspath(os.fspath(training_projection_receipt_path)))
    _require(
        projection_path.parent == projection_receipt_path.parent,
        "training projection and receipt must share one sealed directory",
    )
    training_stage, training_directory_fingerprint = _authenticate_plain_directory(
        projection_path.parent,
        expected_files=frozenset({"training_projection.jsonl", "projection-receipt.json"}),
        label="accepted training-projection stage",
    )
    _require(
        projection_path == training_stage["training_projection.jsonl"].path
        and projection_receipt_path == training_stage["projection-receipt.json"].path,
        "training paths must name the exact accepted stage files",
    )
    training = _parse_training_projection(
        training_stage["training_projection.jsonl"], contract=contract
    )
    _parse_projection_receipt(
        training_stage["projection-receipt.json"],
        projection_snapshot=training_stage["training_projection.jsonl"],
        contract=contract,
    )
    reference_snapshot = _snapshot(
        organizer_reference_path,
        label="tracked organizer reference",
        expected_mode=0o644,
        require_single_link=True,
    )
    references = _parse_reference(reference_snapshot, contract=contract)
    candidate_sequences = {candidate.sequence for candidate in candidates}
    _require(
        candidate_sequences.isdisjoint(row.sequence for row in training),
        "candidate library exactly overlaps the accepted training projection",
    )
    _require(
        candidate_sequences.isdisjoint(sequence for _, sequence in references),
        "candidate library exactly overlaps the organizer reference",
    )

    reconstruction = _reconstruct_payloads(
        candidates,
        training,
        references,
        contract=contract,
        workers=workers,
    )
    transform_document = _feature_transform_document(
        config_snapshot=config_snapshot,
        contract=contract,
        reconstruction=reconstruction,
    )
    transform_payload = _canonical_json(transform_document)
    reconstruction = replace(
        reconstruction,
        transform_document=transform_document,
        transform_payload=transform_payload,
    )
    manifest_document = _manifest_document(
        config_snapshot=config_snapshot,
        contract=contract,
        reconstruction=reconstruction,
        reference_records=contract.expected_reference_records,
        unique_references=len(references),
    )
    manifest_payload = _canonical_json(manifest_document)

    for twin_index, output in enumerate(output_twins):
        _require(
            output["candidate_selection_ledger.csv"].payload == reconstruction.ledger_payload,
            f"output twin {twin_index} candidate selection ledger does not independently "
            "reconstruct",
        )
        _require(
            output["diversity_feature_matrix.f32le"].payload == reconstruction.matrix_payload,
            f"output twin {twin_index} diversity feature matrix bits do not independently "
            "reconstruct",
        )
        observed_transform = _json_object(
            output["feature_transform.json"].payload,
            label=f"output twin {twin_index} feature transform",
        )
        _require(
            observed_transform == transform_document,
            f"output twin {twin_index} feature transform semantics differ",
        )
        _require(
            output["feature_transform.json"].payload == transform_payload,
            f"output twin {twin_index} feature transform bytes do not independently reconstruct",
        )
        observed_manifest = _json_object(
            output["manifest.json"].payload,
            label=f"output twin {twin_index} manifest",
        )
        _require(
            observed_manifest == manifest_document,
            f"output twin {twin_index} manifest semantics differ",
        )
        _require(
            output["manifest.json"].payload == manifest_payload,
            f"output twin {twin_index} manifest bytes do not independently reconstruct",
        )

    snapshots: dict[str, Snapshot] = {"config": config_snapshot, "reference": reference_snapshot}
    for twin_index, bundle in enumerate(mean_twins):
        snapshots.update({f"mean_twin_{twin_index}/{name}": item for name, item in bundle.items()})
    snapshots.update({f"mean_audit/{name}": item for name, item in mean_audit.items()})
    snapshots.update({f"training_stage/{name}": item for name, item in training_stage.items()})
    for twin_index, output in enumerate(output_twins):
        snapshots.update(
            {f"output_twin_{twin_index}/{name}": item for name, item in output.items()}
        )
    for label, snapshot in snapshots.items():
        _assert_unchanged(snapshot, label=label)
    for index, root in enumerate(mean_ledger_twin_paths):
        _assert_directory_unchanged(
            root,
            fingerprint=mean_directory_fingerprints[index],
            expected_files=_MEAN_LEDGER_FILES,
            label=f"accepted mean-ledger twin {index}",
        )
    _assert_directory_unchanged(
        mean_ledger_audit_dir,
        fingerprint=mean_audit_directory_fingerprint,
        expected_files=_MEAN_LEDGER_AUDIT_FILES,
        label="accepted mean-ledger audit",
    )
    _assert_directory_unchanged(
        projection_path.parent,
        fingerprint=training_directory_fingerprint,
        expected_files=frozenset({"training_projection.jsonl", "projection-receipt.json"}),
        label="accepted training-projection stage",
    )
    for twin_index, output_root in enumerate(output_roots):
        _assert_directory_unchanged(
            output_root,
            fingerprint=output_directory_fingerprints[twin_index],
            expected_files=_OUTPUT_FILES,
            label=f"candidate diversity/novelty output twin {twin_index}",
        )

    witness_count = reconstruction.rows - reconstruction.reference_safe_rows
    common_output_sha256 = {
        name: snapshot.sha256 for name, snapshot in sorted(output_twins[0].items())
    }
    output_twins_sha256 = {
        str(twin_index): {name: snapshot.sha256 for name, snapshot in sorted(output.items())}
        for twin_index, output in enumerate(output_twins)
    }
    checks = {
        "accepted_mean_ledger_audit_authenticated": True,
        "accepted_mean_ledger_twins_byte_identical": True,
        "accepted_training_projection_and_receipt_authenticated": True,
        "candidate_identity_order_lineage_and_activity_means_preserved": True,
        "effective_eligibility_independently_reconstructed": True,
        "each_output_twin_matches_independent_reconstruction": True,
        "exact_reference_strict_threshold_and_witnesses_reconstructed": True,
        "exact_training_nearest_similarity_ties_and_novelty_reconstructed": True,
        "feature_transform_training_only_independently_reconstructed": True,
        "float32_matrix_and_csv_cells_bit_exact": True,
        "input_and_output_inventories_modes_and_checksums_exact": True,
        "manifest_census_quantiles_and_lineage_reconstructed": True,
        "no_cluster_esm_sensitivity_or_uncertainty_proxy_exposed": True,
        "reference_threshold_equality_is_safe": True,
        "reference_witness_is_not_claimed_as_maximum": True,
        "output_twins_byte_identical": True,
    }
    return {
        "schema_version": 1,
        "artifact": "candidate_diversity_novelty_v1_independent_verification",
        "status": "passed",
        "automatic_production_eligible": False,
        "checks": checks,
        "census": {
            "candidates": reconstruction.rows,
            "source_library_eligible": reconstruction.source_eligible_rows,
            "reference_safe": reconstruction.reference_safe_rows,
            "reference_unsafe_witnesses": witness_count,
            "effective_eligible": reconstruction.effective_eligible_rows,
            "training_rows": len(training),
            "organizer_reference_records": contract.expected_reference_records,
            "organizer_reference_unique_sequences": len(references),
            "features": len(_FEATURE_NAMES),
        },
        "input_sha256": {
            "config": config_snapshot.sha256,
            "mean_ledger_twins": {
                str(index): {name: snapshot.sha256 for name, snapshot in sorted(bundle.items())}
                for index, bundle in enumerate(mean_twins)
            },
            "mean_ledger_audit": {
                name: snapshot.sha256 for name, snapshot in sorted(mean_audit.items())
            },
            "training_projection": training_stage["training_projection.jsonl"].sha256,
            "training_projection_receipt": training_stage["projection-receipt.json"].sha256,
            "organizer_reference": reference_snapshot.sha256,
        },
        "output_sha256": common_output_sha256,
        "output_twins_sha256": output_twins_sha256,
        "semantic_sha256": {
            "candidate_selection_ledger_rows": reconstruction.ledger_semantic_sha256,
            "diversity_feature_matrix_rows": reconstruction.matrix_semantic_sha256,
            "feature_transform": hashlib.sha256(transform_payload).hexdigest(),
            "manifest": hashlib.sha256(manifest_payload).hexdigest(),
        },
        "matrix": {
            "shape": [reconstruction.rows, len(_FEATURE_NAMES)],
            "dtype": "float32",
            "byte_order": "little_endian",
            "memory_order": "C",
            "raw_bytes_sha256": hashlib.sha256(reconstruction.matrix_payload).hexdigest(),
        },
        "claims": {
            "allowed_consumer": _ALLOWED_CONSUMER,
            "calibrated_uncertainty_available": False,
            "cluster_assignment": False,
            "development_diversity_novelty_evidence": True,
            "esm_embedding": False,
            "final_ranking": False,
            "model_fit_sensitivity_consumed": False,
            "production_ensemble": False,
            "reference_rule_is_biological_safety": False,
            "reference_witness_is_reference_maximum": False,
            "uncertainty_available": False,
        },
    }


def write_receipt_atomic(path: str | Path, receipt: Mapping[str, object]) -> None:
    """Create one canonical read-only receipt without replacing any path."""

    destination = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(destination.parent, label="receipt parent")
    try:
        parent_metadata = os.lstat(destination.parent)
    except OSError as error:
        raise ValueError("cannot inspect receipt parent") from error
    _require(stat.S_ISDIR(parent_metadata.st_mode), "receipt parent must be a directory")
    _require(not os.path.lexists(destination), "receipt already exists")
    payload = _canonical_json(dict(receipt))
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    _require(not os.path.lexists(temporary), "receipt staging path already exists")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            _require(count > 0, "receipt write made no progress")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(temporary, 0o444, follow_symlinks=False)
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
        directory_descriptor = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-ledger-twin", action="append", type=Path, required=True)
    parser.add_argument("--mean-ledger-audit-dir", type=Path, required=True)
    parser.add_argument("--training-projection", type=Path, required=True)
    parser.add_argument(
        "--projection-receipt",
        "--training-projection-receipt",
        dest="training_projection_receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--organizer-reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--bundle-dir",
        "--bundle",
        dest="bundle_dirs",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--receipt",
        "--output",
        dest="receipts",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt_paths = tuple(Path(os.path.abspath(os.fspath(path))) for path in args.receipts)
        _require(len(receipt_paths) == 2, "exactly two receipt paths are required")
        _require(receipt_paths[0] != receipt_paths[1], "receipt paths must be distinct")
        receipt = verify_candidate_diversity_novelty(
            mean_ledger_twin_paths=args.mean_ledger_twin,
            mean_ledger_audit_dir=args.mean_ledger_audit_dir,
            training_projection_path=args.training_projection,
            training_projection_receipt_path=args.training_projection_receipt,
            organizer_reference_path=args.organizer_reference,
            config_path=args.config,
            bundle_dirs=args.bundle_dirs,
            workers=args.workers,
        )
        receipt_snapshots: list[Snapshot] = []
        for index, receipt_path in enumerate(receipt_paths):
            write_receipt_atomic(receipt_path, receipt)
            receipt_snapshots.append(
                _snapshot(
                    receipt_path,
                    label=f"independent verification receipt {index}",
                    expected_mode=0o444,
                    require_single_link=True,
                )
            )
        _require(
            receipt_snapshots[0].payload == receipt_snapshots[1].payload,
            "independent verification receipt bytes differ",
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"candidate diversity/novelty verification error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through cluster CLI
    raise SystemExit(main())

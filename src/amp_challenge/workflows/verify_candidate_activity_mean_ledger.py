"""Independently verify the candidate-activity mean-only handoff ledger.

The accepted candidate scorer publishes useful activity means alongside a
five-refit sensitivity diagnostic.  The diagnostic is not calibrated
uncertainty and must never silently become a selectable uncertainty column.
This verifier authenticates both accepted scoring twins and their audit,
reconstructs the split mean ledger and diagnostic table without importing the
producer or scorer, and emits a deterministic receipt with that boundary made
explicit.
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
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SCHEMA_VERSION = 1
_CONFIG_ARTIFACT = "candidate_activity_mean_ledger_v1"
_CONFIG_STATUS = "predeclared_development_mean_only_handoff"
_OUTPUT_ARTIFACT = "candidate_activity_mean_ledger_v1"
_OUTPUT_STATUS = "development_mean_only_control"
_ALLOWED_CONSUMER = "mean_only_selection_control_only"
_ELIGIBILITY_SCOPE = "accepted_candidate_pool_library_eligible_flag_v1"
_UNCERTAINTY_STATUS = "unavailable"
_DIAGNOSTIC_SCOPE = (
    "population_sd_across_five_outer_fold_complement_refits_diagnostic_not_uncertainty_"
    "not_selectable"
)
_PREDICTION_SCOPE = "declared_seven_target_activity_panel"
_TRAINING_SCOPE = "all_2492_accepted_gate1_contexts_after_recipe_freeze"
_CALIBRATION_SCOPE = "none_raw_logistic_probability"
_TARGET_AGGREGATION = "equal_target_arithmetic_mean_within_declared_scope_v1"
_FIT_SENSITIVITY_SEMANTICS = (
    "population_sd_across_five_outer_fold_complement_refits_diagnostic_not_posterior_or_"
    "aleatoric_uncertainty"
)

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
_POSITIVE_TARGETS = frozenset(
    {"enterococcus_faecalis", "enterococcus_faecium", "staphylococcus_aureus"}
)
_NEGATIVE_TARGETS = frozenset(set(_TARGETS) - _POSITIVE_TARGETS)

_SCORING_FILES = frozenset(
    {
        "candidate_activity_scores.csv",
        "fold_model_target_probabilities.npy",
        "manifest.json",
        "model_states.json",
        "oof_reproduction.json",
        "SHA256SUMS",
    }
)
_AUDIT_FILES = frozenset(
    {
        "operational-receipt.json",
        "twin-0-independent-verification.json",
        "twin-1-independent-verification.json",
    }
)
_LEDGER_FILES = frozenset(
    {
        "candidate_ledger.csv",
        "model_fit_sensitivity_diagnostics.csv",
        "manifest.json",
        "SHA256SUMS",
    }
)

_SOURCE_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    "sequence",
    "length",
    "library_eligible",
    "generator_families",
    "generator_variants",
    *(f"probability_{target}" for target in _TARGETS),
    *(f"mean_{objective}" for objective in _OBJECTIVES),
    *(f"model_fit_sensitivity_{objective}" for objective in _OBJECTIVES),
    "prediction_scope",
    "training_scope",
    "calibration_scope",
    "model_release_id",
)
_LEDGER_COLUMNS = (
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
_DIAGNOSTIC_COLUMNS = (
    "source_ordinal",
    "sequence_id",
    *(f"model_fit_sensitivity_{objective}" for objective in _OBJECTIVES),
    "diagnostic_scope",
)
_FORBIDDEN_LEDGER_TOKENS = (
    "aleatoric",
    "confidence_interval",
    "epistemic",
    "hemolysis",
    "mdr",
    "out_of_distribution",
    "posterior",
    "quality_probability",
    "selectivity",
    "sensitivity",
    "std",
    "variance",
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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SEQUENCE_ID_RE = _SHA256_RE
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_LINEAGE_TOKEN_RE = re.compile(r"[^|,\r\n\x00-\x20\x7f]{1,256}")


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    expected_candidates: int
    expected_examples: int
    folds: int
    targets: int
    objectives: tuple[str, ...]
    allowed_consumer: str
    eligibility_scope: str
    uncertainty_status: str
    diagnostic_scope: str
    prediction_scope: str
    training_scope: str
    calibration_scope: str
    accepted_scoring: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Reconstruction:
    ledger_payload: bytes
    diagnostics_payload: bytes
    ledger_semantic_sha256: str
    diagnostics_semantic_sha256: str
    rows: int
    eligible_rows: int
    model_release_id: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_mapping(value: object, *, label: str, expected_keys: set[str]) -> Mapping[str, object]:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    keys = set(value)
    _require(
        keys == expected_keys,
        f"{label} keys differ: missing={sorted(expected_keys - keys)}, "
        f"extra={sorted(keys - expected_keys)}",
    )
    _require(all(isinstance(key, str) for key in value), f"{label} keys must be strings")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
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


def _canonical_json(value: object) -> bytes:
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


def _reject_symlink_chain(path: Path, *, label: str) -> None:
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


def _snapshot(path: str | Path, *, label: str) -> Snapshot:
    requested = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(requested, label=label)
    try:
        named_before = os.lstat(requested)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {requested}") from error
    _require(stat.S_ISREG(named_before.st_mode), f"{label} must be a regular file")
    _require(not stat.S_ISLNK(named_before.st_mode), f"{label} must not be a symbolic link")
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
    _require(len(payload) == named_before.st_size, f"{label} size changed while it was read")
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
        f"{label} changed during verification",
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


def _directory_metadata(path: Path, *, label: str) -> os.stat_result:
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
    _require(stat.S_IMODE(metadata.st_mode) == 0o555, f"{label} mode must be 0555")
    return metadata


def _authenticate_bundle(
    root: str | Path, *, expected_files: frozenset[str], label: str
) -> tuple[Mapping[str, Snapshot], tuple[int, int, int, int, int, int, int, int, int]]:
    path = Path(os.path.abspath(os.fspath(root)))
    metadata = _directory_metadata(path, label=label)
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
        item_metadata = entry.stat(follow_symlinks=False)
        _require(
            entry.is_file(follow_symlinks=False)
            and not entry.is_symlink()
            and stat.S_IMODE(item_metadata.st_mode) == 0o444,
            f"{label} artifact {entry.name} must be a regular 0444 file",
        )
    top = _snapshot(path / "SHA256SUMS", label=f"{label} SHA256SUMS")
    checksums = _parse_sha256sums(top.payload, label=f"{label} SHA256SUMS")
    _require(
        set(checksums) == set(expected_files) - {"SHA256SUMS"},
        f"{label} checksum inventory is not exact",
    )
    snapshots: dict[str, Snapshot] = {"SHA256SUMS": top}
    for name in sorted(checksums):
        snapshot = _snapshot(path / name, label=f"{label} artifact {name}")
        _require(
            snapshot.sha256 == checksums[name],
            f"{label} checksum mismatch for {name}",
        )
        snapshots[name] = snapshot
    return snapshots, _fingerprint(metadata)


def _authenticate_plain_directory(
    root: str | Path, *, expected_files: frozenset[str], label: str
) -> tuple[Mapping[str, Snapshot], tuple[int, int, int, int, int, int, int, int, int]]:
    path = Path(os.path.abspath(os.fspath(root)))
    metadata = _directory_metadata(path, label=label)
    entries = list(os.scandir(path))
    names = {entry.name for entry in entries}
    _require(names == set(expected_files), f"{label} inventory is not exact")
    snapshots: dict[str, Snapshot] = {}
    for entry in entries:
        item_metadata = entry.stat(follow_symlinks=False)
        _require(
            entry.is_file(follow_symlinks=False)
            and not entry.is_symlink()
            and stat.S_IMODE(item_metadata.st_mode) == 0o444,
            f"{label} artifact {entry.name} must be a regular 0444 file",
        )
    for name in sorted(expected_files):
        snapshots[name] = _snapshot(path / name, label=f"{label} artifact {name}")
    return snapshots, _fingerprint(metadata)


def _assert_directory_unchanged(
    root: str | Path,
    *,
    fingerprint: tuple[int, int, int, int, int, int, int, int, int],
    expected_files: frozenset[str],
    label: str,
) -> None:
    path = Path(os.path.abspath(os.fspath(root)))
    metadata = _directory_metadata(path, label=label)
    _require(_fingerprint(metadata) == fingerprint, f"{label} changed during verification")
    _require(
        {entry.name for entry in os.scandir(path)} == set(expected_files),
        f"{label} inventory changed during verification",
    )


def _sha256(value: object, *, label: str) -> str:
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


def _parse_config(snapshot: Snapshot) -> LedgerConfig:
    try:
        value = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("mean-ledger config must be valid UTF-8 TOML") from error
    raw = _exact_mapping(
        value,
        label="mean-ledger config",
        expected_keys={
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
        },
    )
    _require(
        type(raw["schema_version"]) is int and raw["schema_version"] == _SCHEMA_VERSION,
        "config schema_version is invalid",
    )
    _require(raw["artifact"] == _CONFIG_ARTIFACT, "config artifact is invalid")
    _require(raw["status"] == _CONFIG_STATUS, "config status is invalid")
    _require(
        raw["automatic_production_eligible"] is False,
        "config cannot declare automatic production eligibility",
    )
    expected_literals = {
        "allowed_consumer": _ALLOWED_CONSUMER,
        "eligibility_scope": _ELIGIBILITY_SCOPE,
        "uncertainty_status": _UNCERTAINTY_STATUS,
        "diagnostic_scope": _DIAGNOSTIC_SCOPE,
        "prediction_scope": _PREDICTION_SCOPE,
        "training_scope": _TRAINING_SCOPE,
        "calibration_scope": _CALIBRATION_SCOPE,
    }
    for key, expected in expected_literals.items():
        _require(raw[key] == expected, f"config {key} is invalid")
    objectives = raw["objectives"]
    _require(
        isinstance(objectives, list)
        and all(isinstance(item, str) for item in objectives)
        and tuple(objectives) == _OBJECTIVES,
        "config objectives are invalid",
    )
    accepted = _exact_mapping(
        raw["accepted_scoring"],
        label="config accepted_scoring",
        expected_keys={
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
        },
    )
    for key in ("producer_job_id", "audit_job_id"):
        _positive_int(accepted[key], label=f"accepted_scoring.{key}")
    _require(
        isinstance(accepted["git_commit"], str)
        and _GIT_SHA_RE.fullmatch(accepted["git_commit"]) is not None,
        "accepted_scoring.git_commit must be a full lowercase Git SHA",
    )
    for key in set(accepted) - {"producer_job_id", "audit_job_id", "git_commit"}:
        _sha256(accepted[key], label=f"accepted_scoring.{key}")
    _require(
        accepted["model_release_id"] == accepted["model_states_sha256"],
        "accepted scoring model release is not bound to model states",
    )
    expected_candidates = _positive_int(raw["expected_candidates"], label="expected_candidates")
    expected_examples = _positive_int(raw["expected_examples"], label="expected_examples")
    folds = _positive_int(raw["folds"], label="folds")
    targets = _positive_int(raw["targets"], label="targets")
    _require(folds == 5, "config folds must be 5")
    _require(targets == len(_TARGETS), "config targets must be 7")
    return LedgerConfig(
        expected_candidates=expected_candidates,
        expected_examples=expected_examples,
        folds=folds,
        targets=targets,
        objectives=_OBJECTIVES,
        allowed_consumer=_ALLOWED_CONSUMER,
        eligibility_scope=_ELIGIBILITY_SCOPE,
        uncertainty_status=_UNCERTAINTY_STATUS,
        diagnostic_scope=_DIAGNOSTIC_SCOPE,
        prediction_scope=_PREDICTION_SCOPE,
        training_scope=_TRAINING_SCOPE,
        calibration_scope=_CALIBRATION_SCOPE,
        accepted_scoring=dict(accepted),
    )


def _tree_sha256(snapshots: Mapping[str, Snapshot]) -> str:
    transcript = b"".join(
        f"444 {snapshots[name].sha256} {name}\n".encode() for name in sorted(snapshots)
    )
    return hashlib.sha256(transcript).hexdigest()


def _verify_scoring_manifest(
    snapshot: Snapshot, *, snapshots: Mapping[str, Snapshot], config: LedgerConfig
) -> Mapping[str, object]:
    document = _json_object(snapshot.payload, label="accepted scoring manifest")
    accepted = config.accepted_scoring
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
    _require(set(document) == expected_root, "accepted scoring manifest keys differ")
    _require(
        type(document.get("schema_version")) is int and document.get("schema_version") == 1,
        "accepted scoring manifest schema is invalid",
    )
    _require(
        document.get("artifact") == "candidate_activity_scoring_development_v1"
        and document.get("status") == "development_candidate_scoring_only",
        "accepted scoring manifest identity/status is invalid",
    )
    _require(
        document.get("automatic_production_eligible") is False,
        "accepted scoring manifest cannot be production eligible",
    )
    _require(document.get("git_commit") == accepted["git_commit"], "scoring Git binding differs")
    _require(
        document.get("model_release_id") == accepted["model_release_id"],
        "scoring model release differs from config",
    )
    for key, expected in (
        ("prediction_scope", config.prediction_scope),
        ("training_scope", config.training_scope),
        ("probability_calibration", config.calibration_scope),
        ("fit_sensitivity_semantics", _FIT_SENSITIVITY_SEMANTICS),
        ("target_aggregation", _TARGET_AGGREGATION),
    ):
        _require(document.get(key) == expected, f"accepted scoring manifest {key} is invalid")
    candidate_csv = document.get("candidate_csv")
    _require(isinstance(candidate_csv, dict), "accepted scoring candidate_csv is invalid")
    assert isinstance(candidate_csv, dict)
    _require(
        candidate_csv.get("filename") == "candidate_activity_scores.csv"
        and candidate_csv.get("rows") == config.expected_candidates
        and candidate_csv.get("columns") == list(_SOURCE_COLUMNS),
        "accepted scoring candidate CSV declaration is invalid",
    )
    expected_targets = [
        {
            "gram": "positive" if target in _POSITIVE_TARGETS else "negative",
            "index": index,
            "name": target,
            "probability_column": f"probability_{target}",
        }
        for index, target in enumerate(_TARGETS)
    ]
    _require(document.get("targets") == expected_targets, "accepted scoring targets differ")
    fold_tensor = document.get("fold_tensor")
    _require(isinstance(fold_tensor, dict), "accepted scoring fold tensor is invalid")
    assert isinstance(fold_tensor, dict)
    _require(
        fold_tensor.get("filename") == "fold_model_target_probabilities.npy"
        and fold_tensor.get("shape") == [config.expected_candidates, config.folds, config.targets]
        and fold_tensor.get("raw_data_sha256") == accepted["fold_tensor_raw_data_sha256"]
        and fold_tensor.get("target_axis_order") == list(_TARGETS),
        "accepted scoring fold tensor binding is invalid",
    )
    _require(
        fold_tensor
        == {
            "filename": "fold_model_target_probabilities.npy",
            "axes": ["candidate", "outer_fold_complement_model", "target"],
            "shape": [config.expected_candidates, config.folds, config.targets],
            "dtype": "float64",
            "byte_order": "little_endian",
            "memory_order": "C_contiguous",
            "candidate_axis_order": "candidate_jsonl_source_ordinal_ascending",
            "fold_axis_member_order": [f"outer_fold_{index}" for index in range(config.folds)],
            "target_axis_order": list(_TARGETS),
            "raw_data_sha256": accepted["fold_tensor_raw_data_sha256"],
        },
        "accepted scoring fold tensor contract differs",
    )
    _require(
        document.get("aggregations")
        == {
            "objectives": list(_OBJECTIVES),
            "target_reduction": _TARGET_AGGREGATION,
            "deployment_probability_source": "all_data_deployment",
            "model_fit_sensitivity": _FIT_SENSITIVITY_SEMANTICS,
        },
        "accepted scoring aggregation declaration differs",
    )
    _require(
        document.get("models")
        == {
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
        },
        "accepted scoring model declaration differs",
    )
    _require(
        document.get("claims")
        == {
            "forbidden": list(_FORBIDDEN_CLAIMS),
            "historical_method_choice_used_fold4": True,
            "organizer_reference_used_for_model_fit": False,
        },
        "accepted scoring claims differ",
    )
    inputs = document.get("inputs")
    _require(
        isinstance(inputs, dict) and set(inputs) == {"config", "candidate_pool", "gate1"},
        "accepted scoring manifest inputs are invalid",
    )
    assert isinstance(inputs, dict)
    scoring_config = inputs.get("config")
    _require(
        isinstance(scoring_config, dict)
        and scoring_config.get("sha256") == accepted["config_sha256"],
        "accepted scoring config binding is invalid",
    )
    candidate_pool = inputs.get("candidate_pool")
    _require(
        isinstance(candidate_pool, dict)
        and set(candidate_pool)
        == {
            "producer_job_id",
            "git_commit",
            "publication_top_sha256",
            "manifest_sha256",
            "candidates_sha256",
            "validation_summary_sha256",
            "final_publication_check_sha256",
            "candidates",
        }
        and candidate_pool.get("candidates") == config.expected_candidates,
        "accepted scoring candidate-pool provenance is invalid",
    )
    gate1 = inputs.get("gate1")
    _require(
        isinstance(gate1, dict)
        and set(gate1)
        == {
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "publication_top_sha256",
            "semantic_top_sha256",
            "examples_sha256",
            "oof_sha256",
            "independent_receipt_sha256",
            "examples",
        }
        and gate1.get("examples") == config.expected_examples,
        "accepted scoring Gate-1 provenance is invalid",
    )
    artifacts = document.get("artifacts")
    _require(isinstance(artifacts, dict), "accepted scoring artifact map is invalid")
    assert isinstance(artifacts, dict)
    expected_artifacts = set(_SCORING_FILES) - {"SHA256SUMS", "manifest.json"}
    _require(
        set(artifacts) == expected_artifacts,
        "accepted scoring artifact inventory is invalid",
    )
    for name in sorted(expected_artifacts):
        record = artifacts[name]
        _require(isinstance(record, dict), f"accepted scoring artifact {name} is invalid")
        assert isinstance(record, dict)
        _require(
            record.get("sha256") == snapshots[name].sha256
            and record.get("size_bytes") == len(snapshots[name].payload)
            and type(record.get("size_bytes")) is int,
            f"accepted scoring artifact record differs for {name}",
        )
        if name == "candidate_activity_scores.csv":
            _require(record.get("rows") == config.expected_candidates, "scoring row count differs")
    return document


def _verify_scoring_twins(
    scoring_twin_paths: Sequence[str | Path], *, config: LedgerConfig
) -> tuple[
    tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
    tuple[tuple[int, int, int, int, int, int, int, int, int], ...],
]:
    _require(len(scoring_twin_paths) == 2, "exactly two scoring twins are required")
    absolute = tuple(Path(os.path.abspath(os.fspath(path))) for path in scoring_twin_paths)
    _require(absolute[0] != absolute[1], "scoring twin paths must be distinct")
    bundles: list[Mapping[str, Snapshot]] = []
    fingerprints: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    accepted = config.accepted_scoring
    pinned = {
        "SHA256SUMS": accepted["publication_top_sha256"],
        "candidate_activity_scores.csv": accepted["candidate_scores_sha256"],
        "fold_model_target_probabilities.npy": accepted["fold_tensor_sha256"],
        "manifest.json": accepted["manifest_sha256"],
        "model_states.json": accepted["model_states_sha256"],
        "oof_reproduction.json": accepted["oof_reproduction_sha256"],
    }
    for index, path in enumerate(absolute):
        snapshots, directory_fingerprint = _authenticate_bundle(
            path,
            expected_files=_SCORING_FILES,
            label=f"accepted scoring twin {index}",
        )
        for name, expected in pinned.items():
            _require(
                snapshots[name].sha256 == expected,
                f"accepted scoring twin {index} {name} differs from config pin",
            )
        _require(
            _tree_sha256(snapshots) == accepted["publication_tree_sha256"],
            f"accepted scoring twin {index} tree hash differs from config pin",
        )
        _verify_scoring_manifest(snapshots["manifest.json"], snapshots=snapshots, config=config)
        bundles.append(snapshots)
        fingerprints.append(directory_fingerprint)
    for name in sorted(_SCORING_FILES):
        _require(
            bundles[0][name].payload == bundles[1][name].payload,
            f"accepted scoring twins differ at {name}",
        )
    return (bundles[0], bundles[1]), tuple(fingerprints)


def _all_true_mapping(value: object, *, label: str) -> Mapping[str, object]:
    _require(isinstance(value, dict) and bool(value), f"{label} must be a non-empty object")
    assert isinstance(value, dict)
    _require(
        all(isinstance(key, str) and item is True for key, item in value.items()),
        f"{label} must contain only passed boolean checks",
    )
    return value


def _verify_audit(
    audit_dir: str | Path,
    *,
    config: LedgerConfig,
    scoring_twins: tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
) -> tuple[Mapping[str, Snapshot], tuple[int, int, int, int, int, int, int, int, int]]:
    snapshots, directory_fingerprint = _authenticate_plain_directory(
        audit_dir, expected_files=_AUDIT_FILES, label="accepted scoring audit"
    )
    accepted = config.accepted_scoring
    _require(
        snapshots["operational-receipt.json"].sha256 == accepted["operational_audit_sha256"],
        "operational scoring audit differs from config pin",
    )
    for index in range(2):
        name = f"twin-{index}-independent-verification.json"
        _require(
            snapshots[name].sha256 == accepted["independent_verification_sha256"],
            f"scoring twin {index} independent receipt differs from config pin",
        )
        receipt = _json_object(snapshots[name].payload, label=f"scoring twin {index} receipt")
        _require(
            receipt.get("schema_version") == 1
            and receipt.get("artifact")
            == "candidate_activity_scoring_development_v1_independent_verification"
            and receipt.get("status") == "passed"
            and receipt.get("automatic_production_eligible") is False,
            f"scoring twin {index} independent receipt identity/status is invalid",
        )
        _all_true_mapping(receipt.get("checks"), label=f"scoring twin {index} checks")
        _require(
            receipt.get("git_commit") == accepted["git_commit"]
            and receipt.get("model_release_id") == accepted["model_release_id"],
            f"scoring twin {index} independent receipt binding is invalid",
        )
        census = receipt.get("census")
        _require(
            isinstance(census, dict)
            and census.get("candidates") == config.expected_candidates
            and census.get("examples") == config.expected_examples
            and census.get("folds") == config.folds
            and census.get("targets") == config.targets
            and census.get("fold_tensor_shape")
            == [config.expected_candidates, config.folds, config.targets],
            f"scoring twin {index} independent census is invalid",
        )
        output_sha = receipt.get("output_sha256")
        expected_output_sha = {
            name_: snapshot.sha256 for name_, snapshot in sorted(scoring_twins[index].items())
        }
        _require(
            output_sha == expected_output_sha,
            f"scoring twin {index} independent receipt output binding differs",
        )
        input_sha = receipt.get("input_sha256")
        _require(
            isinstance(input_sha, dict) and input_sha.get("config") == accepted["config_sha256"],
            f"scoring twin {index} independent receipt config binding differs",
        )
        claims = receipt.get("claims")
        _require(
            isinstance(claims, dict)
            and claims.get("development_candidate_scoring_only") is True
            and claims.get("model_fit_sensitivity_is_calibrated_uncertainty") is False
            and claims.get("production_ensemble") is False,
            f"scoring twin {index} independent receipt claim boundary is invalid",
        )
    _require(
        snapshots["twin-0-independent-verification.json"].payload
        == snapshots["twin-1-independent-verification.json"].payload,
        "accepted scoring independent twin receipts differ",
    )
    operational = _json_object(
        snapshots["operational-receipt.json"].payload, label="operational scoring audit"
    )
    _require(
        operational.get("schema_version") == 1
        and operational.get("artifact") == "candidate_activity_scoring_v1_operational_audit"
        and operational.get("status") == "passed"
        and operational.get("automatic_production_eligible") is False
        and operational.get("git_commit") == accepted["git_commit"],
        "operational scoring audit identity/status is invalid",
    )
    _all_true_mapping(operational.get("checks"), label="operational scoring audit checks")
    producer = operational.get("producer")
    audit = operational.get("audit")
    _require(
        isinstance(producer, dict)
        and producer.get("job_id") == accepted["producer_job_id"]
        and isinstance(audit, dict)
        and audit.get("job_id") == accepted["audit_job_id"],
        "operational scoring audit job binding is invalid",
    )
    expected_receipts = {
        str(index): snapshots[f"twin-{index}-independent-verification.json"].sha256
        for index in range(2)
    }
    expected_trees = {str(index): _tree_sha256(scoring_twins[index]) for index in range(2)}
    _require(
        operational.get("independent_receipt_sha256") == expected_receipts
        and operational.get("twin_tree_sha256") == expected_trees,
        "operational scoring audit twin bindings are invalid",
    )
    return snapshots, directory_fingerprint


def _canonical_csv_float(value: str | None, *, label: str, upper: float = 1.0) -> float:
    _require(value is not None and value != "", f"{label} is missing")
    assert value is not None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite decimal") from error
    _require(math.isfinite(parsed) and 0.0 <= parsed <= upper, f"{label} is out of range")
    _require(format(parsed, ".17g") == value, f"{label} is not canonical float text")
    return parsed


def _canonical_lineage(value: str, *, label: str) -> None:
    parts = value.split("|")
    _require(
        bool(parts)
        and all(_LINEAGE_TOKEN_RE.fullmatch(part) is not None for part in parts)
        and parts == sorted(set(parts)),
        f"{label} must be canonical sorted unique lineage tokens",
    )


def _semantic_digest(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json({column: row[column] for column in columns}))
    return digest.hexdigest()


def _csv_payload(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _reconstruct_from_source(snapshot: Snapshot, *, config: LedgerConfig) -> Reconstruction:
    payload = snapshot.payload
    _require(
        bool(payload)
        and payload.endswith(b"\n")
        and not payload.endswith(b"\n\n")
        and b"\r" not in payload
        and b"\x00" not in payload,
        "accepted score CSV must be non-empty canonical LF-framed text",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("accepted score CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    _require(reader.fieldnames == list(_SOURCE_COLUMNS), "accepted score CSV header is invalid")
    ledger_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []
    eligible_rows = 0
    seen_sequence_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for row_number, row in enumerate(reader, 1):
        _require(None not in row, f"accepted score row {row_number} has extra columns")
        _require(row_number <= config.expected_candidates, "accepted score CSV has extra rows")
        _require(
            all(row[column] is not None for column in _SOURCE_COLUMNS),
            f"accepted score row {row_number} has missing columns",
        )
        source_ordinal = row["source_ordinal"]
        _require(
            source_ordinal == str(row_number),
            f"accepted score row {row_number} ordinal differs",
        )
        sequence = row["sequence"]
        _require(
            bool(sequence)
            and sequence == sequence.strip().upper()
            and set(sequence) <= _AMINO_ACIDS,
            f"accepted score row {row_number} sequence is invalid",
        )
        _require(
            row["length"] == str(len(sequence)),
            f"accepted score row {row_number} length differs",
        )
        sequence_id = row["sequence_id"]
        _require(
            _SEQUENCE_ID_RE.fullmatch(sequence_id) is not None
            and sequence_id == hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            f"accepted score row {row_number} sequence identity differs",
        )
        _require(
            sequence_id not in seen_sequence_ids and sequence not in seen_sequences,
            f"accepted score row {row_number} duplicates a candidate",
        )
        seen_sequence_ids.add(sequence_id)
        seen_sequences.add(sequence)
        eligible = row["library_eligible"]
        _require(
            eligible in {"true", "false"},
            f"accepted score row {row_number} eligibility invalid",
        )
        eligible_rows += eligible == "true"
        _canonical_lineage(
            row["generator_families"],
            label=f"accepted score row {row_number} generator families",
        )
        _canonical_lineage(
            row["generator_variants"],
            label=f"accepted score row {row_number} generator variants",
        )
        probabilities: dict[str, float] = {}
        for target in _TARGETS:
            probabilities[target] = _canonical_csv_float(
                row[f"probability_{target}"],
                label=f"accepted score row {row_number} probability_{target}",
            )
        objective_targets = {
            "broad_spectrum_activity": _TARGETS,
            "gram_positive_activity": tuple(
                target for target in _TARGETS if target in _POSITIVE_TARGETS
            ),
            "gram_negative_activity": tuple(
                target for target in _TARGETS if target in _NEGATIVE_TARGETS
            ),
        }
        for objective, targets in objective_targets.items():
            observed = _canonical_csv_float(
                row[f"mean_{objective}"],
                label=f"accepted score row {row_number} mean_{objective}",
            )
            expected = sum(probabilities[target] for target in targets) / len(targets)
            _require(
                abs(observed - expected) <= 1e-12,
                f"accepted score row {row_number} mean_{objective} arithmetic differs",
            )
            _canonical_csv_float(
                row[f"model_fit_sensitivity_{objective}"],
                label=f"accepted score row {row_number} sensitivity_{objective}",
                upper=0.5,
            )
        exact_scopes = {
            "prediction_scope": config.prediction_scope,
            "training_scope": config.training_scope,
            "calibration_scope": config.calibration_scope,
            "model_release_id": str(config.accepted_scoring["model_release_id"]),
        }
        for name, expected in exact_scopes.items():
            _require(
                row[name] == expected,
                f"accepted score row {row_number} {name} differs from contract",
            )
        ledger_rows.append(
            {
                "source_ordinal": source_ordinal,
                "sequence_id": sequence_id,
                "sequence": sequence,
                "length": row["length"],
                "eligible": eligible,
                "eligibility_scope": config.eligibility_scope,
                "generator_families": row["generator_families"],
                "generator_variants": row["generator_variants"],
                **{f"probability_{target}": row[f"probability_{target}"] for target in _TARGETS},
                **{f"mean_{objective}": row[f"mean_{objective}"] for objective in _OBJECTIVES},
                "uncertainty_status": config.uncertainty_status,
                "prediction_scope": config.prediction_scope,
                "training_scope": config.training_scope,
                "calibration_scope": config.calibration_scope,
                "model_release_id": str(config.accepted_scoring["model_release_id"]),
            }
        )
        diagnostic_rows.append(
            {
                "source_ordinal": source_ordinal,
                "sequence_id": sequence_id,
                **{
                    f"model_fit_sensitivity_{objective}": row[f"model_fit_sensitivity_{objective}"]
                    for objective in _OBJECTIVES
                },
                "diagnostic_scope": config.diagnostic_scope,
            }
        )
    _require(
        len(ledger_rows) == config.expected_candidates,
        f"accepted score CSV has {len(ledger_rows)} rows; expected {config.expected_candidates}",
    )
    return Reconstruction(
        ledger_payload=_csv_payload(_LEDGER_COLUMNS, ledger_rows),
        diagnostics_payload=_csv_payload(_DIAGNOSTIC_COLUMNS, diagnostic_rows),
        ledger_semantic_sha256=_semantic_digest(ledger_rows, _LEDGER_COLUMNS),
        diagnostics_semantic_sha256=_semantic_digest(diagnostic_rows, _DIAGNOSTIC_COLUMNS),
        rows=len(ledger_rows),
        eligible_rows=eligible_rows,
        model_release_id=str(config.accepted_scoring["model_release_id"]),
    )


def _validate_ledger_header_boundary(payload: bytes) -> None:
    try:
        header = next(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, StopIteration, csv.Error) as error:
        raise ValueError("candidate ledger header is invalid") from error
    _require(header == list(_LEDGER_COLUMNS), "candidate ledger schema differs")
    for column in header:
        lowered = column.lower()
        if column == "uncertainty_status":
            continue
        _require(
            "uncertainty" not in lowered
            and all(token not in lowered for token in _FORBIDDEN_LEDGER_TOKENS),
            f"candidate ledger leaks forbidden selectable field {column!r}",
        )


def _artifact_record(
    value: object, *, label: str, snapshot: Snapshot, rows: int | None = None
) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    expected_keys = {"sha256", "size_bytes"} | ({"rows"} if rows is not None else set())
    _require(set(value) == expected_keys, f"{label} keys are invalid")
    _require(
        value.get("sha256") == snapshot.sha256
        and value.get("size_bytes") == len(snapshot.payload)
        and type(value.get("size_bytes")) is int,
        f"{label} does not bind exact artifact bytes",
    )
    if rows is not None:
        _require(
            value.get("rows") == rows and type(value.get("rows")) is int,
            f"{label} row count differs",
        )


def _verify_ledger_manifest(
    snapshot: Snapshot,
    *,
    snapshots: Mapping[str, Snapshot],
    config_snapshot: Snapshot,
    config: LedgerConfig,
    reconstruction: Reconstruction,
    scoring_twins: tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
    audit_snapshots: Mapping[str, Snapshot],
) -> Mapping[str, object]:
    """Verify the producer manifest.  Kept separate for an independent schema check."""

    document = _json_object(snapshot.payload, label="mean-ledger manifest")
    expected_root = {
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
    }
    _require(set(document) == expected_root, "mean-ledger manifest keys differ")
    _require(document["schema_version"] == 1, "mean-ledger manifest schema is invalid")
    _require(document["artifact"] == _OUTPUT_ARTIFACT, "mean-ledger manifest artifact is invalid")
    _require(document["status"] == _OUTPUT_STATUS, "mean-ledger manifest status is invalid")
    _require(
        document["automatic_production_eligible"] is False,
        "mean-ledger manifest cannot declare production eligibility",
    )
    _require(document["allowed_consumer"] == config.allowed_consumer, "allowed consumer differs")
    _require(
        document["adapter_config_sha256"] == config_snapshot.sha256,
        "mean-ledger manifest does not bind its config",
    )
    _require(
        document["model_release_id"] == reconstruction.model_release_id,
        "mean-ledger manifest model release differs",
    )
    _require(
        document["scopes"]
        == {
            "calibration": config.calibration_scope,
            "eligibility": config.eligibility_scope,
            "prediction": config.prediction_scope,
            "training": config.training_scope,
            "uncertainty_status": config.uncertainty_status,
        },
        "mean-ledger manifest scopes differ",
    )
    _require(document["objectives"] == list(_OBJECTIVES), "manifest objectives differ")
    expected_targets = [
        {
            "gram": "positive" if target in _POSITIVE_TARGETS else "negative",
            "index": index,
            "name": target,
            "probability_column": f"probability_{target}",
        }
        for index, target in enumerate(_TARGETS)
    ]
    _require(document["targets"] == expected_targets, "manifest targets differ")
    source = document["source"]
    _require(isinstance(source, dict), "mean-ledger source declaration is invalid")
    assert isinstance(source, dict)
    expected_source = {
        "artifact": "candidate_activity_scoring_development_v1",
        "git_commit": config.accepted_scoring["git_commit"],
        **{
            key: config.accepted_scoring[key]
            for key in (
                "config_sha256",
                "publication_top_sha256",
                "publication_tree_sha256",
                "manifest_sha256",
                "candidate_scores_sha256",
                "fold_tensor_sha256",
                "fold_tensor_raw_data_sha256",
                "model_states_sha256",
                "oof_reproduction_sha256",
                "independent_verification_sha256",
                "operational_audit_sha256",
            )
        },
    }
    _require(source == expected_source, "mean-ledger source hash binding differs")
    # Reconfirm that every source hash named by the manifest was observed, not
    # merely copied from the adapter config.
    _require(
        all(
            bundle["SHA256SUMS"].sha256 == source["publication_top_sha256"]
            and bundle["manifest.json"].sha256 == source["manifest_sha256"]
            and bundle["candidate_activity_scores.csv"].sha256 == source["candidate_scores_sha256"]
            and bundle["fold_model_target_probabilities.npy"].sha256 == source["fold_tensor_sha256"]
            and bundle["model_states.json"].sha256 == source["model_states_sha256"]
            and bundle["oof_reproduction.json"].sha256 == source["oof_reproduction_sha256"]
            for bundle in scoring_twins
        ),
        "mean-ledger source does not bind the observed scoring twins",
    )
    _require(
        audit_snapshots["operational-receipt.json"].sha256 == source["operational_audit_sha256"]
        and all(
            audit_snapshots[f"twin-{index}-independent-verification.json"].sha256
            == source["independent_verification_sha256"]
            for index in range(2)
        ),
        "mean-ledger source does not bind the observed scoring audit",
    )
    ledger = document["candidate_ledger"]
    _require(isinstance(ledger, dict), "mean-ledger manifest candidate_ledger is invalid")
    assert isinstance(ledger, dict)
    _require(
        set(ledger)
        == {"columns", "eligibility_mapping", "filename", "rows", "sha256", "size_bytes"},
        "mean-ledger manifest candidate ledger keys differ",
    )
    _require(
        ledger.get("filename") == "candidate_ledger.csv"
        and ledger.get("columns") == list(_LEDGER_COLUMNS)
        and ledger.get("rows") == reconstruction.rows,
        "mean-ledger manifest candidate ledger declaration differs",
    )
    _require(
        ledger.get("eligibility_mapping") == "eligible_is_exact_library_eligible_value",
        "mean-ledger eligibility mapping differs",
    )
    _artifact_record(
        {key: ledger[key] for key in ("rows", "sha256", "size_bytes") if key in ledger},
        label="candidate ledger artifact",
        snapshot=snapshots["candidate_ledger.csv"],
        rows=reconstruction.rows,
    )
    diagnostics = document["diagnostics"]
    _require(isinstance(diagnostics, dict), "mean-ledger manifest diagnostics is invalid")
    assert isinstance(diagnostics, dict)
    _require(
        set(diagnostics)
        == {
            "columns",
            "filename",
            "scope",
            "selectable",
            "uncertainty",
            "rows",
            "sha256",
            "size_bytes",
        },
        "mean-ledger manifest diagnostic keys differ",
    )
    _require(
        diagnostics.get("filename") == "model_fit_sensitivity_diagnostics.csv"
        and diagnostics.get("columns") == list(_DIAGNOSTIC_COLUMNS)
        and diagnostics.get("rows") == reconstruction.rows
        and diagnostics.get("scope") == config.diagnostic_scope
        and diagnostics.get("selectable") is False
        and diagnostics.get("uncertainty") is False,
        "mean-ledger manifest diagnostic declaration differs",
    )
    _artifact_record(
        {key: diagnostics[key] for key in ("rows", "sha256", "size_bytes") if key in diagnostics},
        label="sensitivity diagnostic artifact",
        snapshot=snapshots["model_fit_sensitivity_diagnostics.csv"],
        rows=reconstruction.rows,
    )
    claims = document["claims"]
    _require(isinstance(claims, dict), "mean-ledger manifest claims are invalid")
    assert isinstance(claims, dict)
    expected_claim_keys = {
        "diagnostics_are_uncertainty",
        "diagnostics_selectable",
        "final_ranking",
        "mean_only_development_handoff",
        "production_ensemble",
        "uncertainty_available",
    }
    _require(
        set(claims) == set(expected_claim_keys)
        and claims
        == {
            "diagnostics_are_uncertainty": False,
            "diagnostics_selectable": False,
            "final_ranking": False,
            "mean_only_development_handoff": True,
            "production_ensemble": False,
            "uncertainty_available": False,
        },
        "mean-ledger claim boundary differs",
    )
    exclusions = document["exclusions"]
    _require(
        exclusions
        == {
            "candidate_ledger_column_prefixes": ["model_fit_sensitivity_", "std_"],
            "claims": [
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
            ],
            "endpoints": [
                "hemolysis_risk",
                "mdr_eskape_activity",
                "out_of_distribution",
                "quality_probability",
                "selectivity",
            ],
            "weights": [
                "endpoint_weight",
                "model_fit_sensitivity_weight",
                "uncertainty_weight",
            ],
        },
        "mean-ledger exclusions differ",
    )
    return document


def verify_candidate_activity_mean_ledger(
    *,
    scoring_twin_paths: Sequence[str | Path],
    scoring_audit_dir: str | Path,
    config_path: str | Path,
    ledger_dir: str | Path,
) -> Mapping[str, object]:
    """Authenticate and independently reconstruct one mean-only ledger bundle."""

    config_snapshot = _snapshot(config_path, label="mean-ledger config")
    config = _parse_config(config_snapshot)
    scoring_twins, scoring_directory_fingerprints = _verify_scoring_twins(
        scoring_twin_paths, config=config
    )
    audit_snapshots, audit_directory_fingerprint = _verify_audit(
        scoring_audit_dir, config=config, scoring_twins=scoring_twins
    )
    reconstruction = _reconstruct_from_source(
        scoring_twins[0]["candidate_activity_scores.csv"], config=config
    )
    ledger_snapshots, ledger_directory_fingerprint = _authenticate_bundle(
        ledger_dir, expected_files=_LEDGER_FILES, label="candidate mean-ledger bundle"
    )
    _validate_ledger_header_boundary(ledger_snapshots["candidate_ledger.csv"].payload)
    _require(
        ledger_snapshots["candidate_ledger.csv"].payload == reconstruction.ledger_payload,
        "candidate ledger bytes do not independently reconstruct from accepted scores",
    )
    _require(
        ledger_snapshots["model_fit_sensitivity_diagnostics.csv"].payload
        == reconstruction.diagnostics_payload,
        "sensitivity diagnostic bytes do not independently reconstruct from accepted scores",
    )
    _verify_ledger_manifest(
        ledger_snapshots["manifest.json"],
        snapshots=ledger_snapshots,
        config_snapshot=config_snapshot,
        config=config,
        reconstruction=reconstruction,
        scoring_twins=scoring_twins,
        audit_snapshots=audit_snapshots,
    )

    all_snapshots: dict[str, Snapshot] = {"config": config_snapshot}
    for twin_index, bundle in enumerate(scoring_twins):
        all_snapshots.update(
            {f"scoring_twin_{twin_index}/{name}": item for name, item in bundle.items()}
        )
    all_snapshots.update({f"scoring_audit/{name}": item for name, item in audit_snapshots.items()})
    all_snapshots.update({f"ledger/{name}": item for name, item in ledger_snapshots.items()})
    for name, snapshot in all_snapshots.items():
        _assert_unchanged(snapshot, label=name)
    for index, root in enumerate(scoring_twin_paths):
        _assert_directory_unchanged(
            root,
            fingerprint=scoring_directory_fingerprints[index],
            expected_files=_SCORING_FILES,
            label=f"accepted scoring twin {index}",
        )
    _assert_directory_unchanged(
        scoring_audit_dir,
        fingerprint=audit_directory_fingerprint,
        expected_files=_AUDIT_FILES,
        label="accepted scoring audit",
    )
    _assert_directory_unchanged(
        ledger_dir,
        fingerprint=ledger_directory_fingerprint,
        expected_files=_LEDGER_FILES,
        label="candidate mean-ledger bundle",
    )

    checks = {
        "accepted_scoring_audit_authenticated": True,
        "accepted_scoring_hash_pins_exact": True,
        "accepted_scoring_twins_byte_identical": True,
        "candidate_identity_order_and_lineage_exact": True,
        "candidate_ledger_independently_reconstructed": True,
        "diagnostics_independently_reconstructed_and_separated": True,
        "ledger_checksum_inventory_exact": True,
        "ledger_contains_no_uncertainty_proxy_or_sensitivity": True,
        "manifest_declares_mean_only_development_scope": True,
        "model_release_and_scopes_preserved": True,
    }
    return {
        "schema_version": 1,
        "artifact": "candidate_activity_mean_ledger_v1_independent_verification",
        "status": "passed",
        "automatic_production_eligible": False,
        "checks": checks,
        "census": {
            "candidates": reconstruction.rows,
            "eligible_candidates": reconstruction.eligible_rows,
            "targets": config.targets,
            "objectives": len(config.objectives),
        },
        "input_sha256": {
            "config": config_snapshot.sha256,
            "scoring_twins": {
                str(index): bundle["SHA256SUMS"].sha256
                for index, bundle in enumerate(scoring_twins)
            },
            "scoring_audit": {name: item.sha256 for name, item in sorted(audit_snapshots.items())},
        },
        "output_sha256": {name: item.sha256 for name, item in sorted(ledger_snapshots.items())},
        "semantic_sha256": {
            "candidate_ledger_rows": reconstruction.ledger_semantic_sha256,
            "model_fit_sensitivity_diagnostic_rows": reconstruction.diagnostics_semantic_sha256,
        },
        "model_release_id": reconstruction.model_release_id,
        "claims": {
            "allowed_consumer": config.allowed_consumer,
            "calibrated_uncertainty_available": False,
            "development_mean_only_ledger": True,
            "final_ranking": False,
            "model_fit_sensitivity_is_selectable_uncertainty": False,
            "production_ensemble": False,
        },
    }


def write_receipt_atomic(path: str | Path, receipt: Mapping[str, object]) -> None:
    """Publish a canonical receipt without overwriting an existing file."""

    destination = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(destination.parent, label="receipt parent")
    parent_metadata = os.lstat(destination.parent)
    _require(stat.S_ISDIR(parent_metadata.st_mode), "receipt parent must be a real directory")
    _require(not destination.exists() and not destination.is_symlink(), "receipt already exists")
    payload = _canonical_json(dict(receipt))
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    _require(not temporary.exists() and not temporary.is_symlink(), "receipt staging path exists")
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
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(temporary, 0o444, follow_symlinks=False)
        # A same-filesystem hard link is an atomic create-if-absent operation;
        # unlike replace(), it cannot overwrite a receipt won by a racing run.
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-twin", action="append", type=Path, required=True)
    parser.add_argument(
        "--scoring-audit-dir", "--audit-dir", dest="scoring_audit_dir", type=Path, required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = verify_candidate_activity_mean_ledger(
            scoring_twin_paths=args.scoring_twin,
            scoring_audit_dir=args.scoring_audit_dir,
            config_path=args.config,
            ledger_dir=args.ledger_dir,
        )
        write_receipt_atomic(args.receipt, receipt)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"candidate activity mean-ledger verification error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by cluster CLI
    raise SystemExit(main())

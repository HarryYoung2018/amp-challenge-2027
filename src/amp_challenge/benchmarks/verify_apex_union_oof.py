"""Independently reconstruct and verify APEX union-fold calibration twins.

This verifier deliberately imports none of the APEX producer, Gate-1 producer,
sequence, or metric modules.  It reparses immutable accepted Gate-1 evidence and
raw APEX member inference, independently reconstructs the sequence-to-context
join, forty fold-local calibrators, every prediction and metric, and then
compares the resulting bytes with both producer publications.  Producer
outputs are comparison targets only; no fitted or reported producer value is
used as a reconstruction input.
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
import stat
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GramClass = Literal["positive", "negative"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MIN_LENGTH = 8
_MAX_LENGTH = 50

_ARTIFACT = "apex_context_activity_homology_study_union_v1"
_BASE_ARTIFACT = "gate1_context_activity_homology_study_union_v1"
_BASE_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_OUTPUT_STATUS = _BASE_STATUS
_ASSIGNMENT_POLICY = (
    "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
)
_CALIBRATION_POLICY = (
    "per_member_logistic_calibration_of_target_specific_log10_mic_signal_on_outer_train_folds"
)
_CALIBRATION_WEIGHTING = "context_equal_v1"
_BOOTSTRAP_UNIT = "union_component_id"
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
_BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_MEAN_MODEL = "apex_member_mean"
_SIMILARITY_STRATA = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8))

_LOGICAL_CONFIG_PATH = "configs/models/apex_union_oof_v1.toml"
_PRODUCER_MODULE_PATH = "src/amp_challenge/benchmarks/apex_union_oof.py"
_VERIFIER_MODULE_PATH = "src/amp_challenge/benchmarks/verify_apex_union_oof.py"
_FIXED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/benchmark_apex_union_oof_v1_twins.sbatch",
        "cluster/validate_apex_union_oof_output.sh",
        _LOGICAL_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)

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

_RAW_SCHEMA = (
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
_OUTPUT_SCHEMA = (
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
_BASE_SEMANTIC = frozenset(
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
_BASE_PUBLICATION = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "gate1/SHA256SUMS",
        *(f"gate1/{name}" for name in _BASE_SEMANTIC),
    }
)
_BASE_TREE = _BASE_PUBLICATION | {"SHA256SUMS"}
_RAW_APEX_TREE = frozenset(
    {
        "apex_member_predictions.csv",
        "apex_member_run_manifest.json",
        "apex_member_reconciliation.csv",
    }
)
_SEMANTIC_OUTPUTS = frozenset(
    {
        "apex_union_oof_predictions.csv",
        "calibrators.json",
        "manifest.json",
        "metrics.json",
        "sequence_coverage_receipt.json",
    }
)
_PRODUCER_TREE = frozenset(
    {
        "SHA256SUMS",
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "apex/SHA256SUMS",
        *(f"apex/{name}" for name in _SEMANTIC_OUTPUTS),
    }
)
_UPSTREAM_GATE1_CHECKS = frozenset(
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
_VERIFICATION_CHECKS = frozenset(
    {
        "accepted_gate1_chain_valid",
        "all_40_calibrators_recomputed",
        "all_semantic_artifacts_exact",
        "candidate_semantics_preserved",
        "canonical_serialization_verified",
        "code_and_commit_attestations_valid",
        "context_join_recomputed",
        "frozen_inputs_match_raw_sources",
        "member_mean_and_std_recomputed",
        "outer_oof_predictions_recomputed",
        "producer_publication_manifests_valid",
        "production_overlap_handshake_valid",
        "raw_apex_grid_recomputed",
        "raw_apex_reconciliation_recomputed",
        "shared_union_bootstrap_recomputed",
        "scratch_and_absolute_paths_absent",
        "twins_byte_identical",
    }
)


class VerificationError(ValueError):
    """Raised when evidence differs from independent reconstruction."""


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    endpoints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Member:
    member_id: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class Config:
    path: Path
    sha256: str
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
    targets: tuple[Target, ...]
    members: tuple[Member, ...]

    @property
    def target_by_name(self) -> dict[str, Target]:
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
class RawGrid:
    members: tuple[str, ...]
    member_hashes: Mapping[str, str]
    values: Mapping[str, Mapping[str, Mapping[str, float]]]
    sequences: Mapping[str, str]
    rows: int


@dataclass(frozen=True, slots=True)
class JoinedContext:
    example: Example
    target: Target
    signals: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class FittedCalibration:
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

    def probability(self, signal: float) -> float:
        standardized = (signal - self.signal_mean) / self.signal_scale
        value = _stable_sigmoid(self.intercept + self.slope * standardized)
        return float(
            np.clip(
                value,
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


def _require(condition: object, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_snapshot(path: Path, *, label: str) -> Snapshot:
    requested = Path(path).absolute()
    _require_no_symlink(requested, label=label, ancestors=True)
    resolved = requested.resolve(strict=True)
    before = resolved.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while read",
    )
    return Snapshot(resolved, payload, _sha256(payload), _fingerprint(after))


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = snapshot.path.stat()
    _require(
        _fingerprint(current) == snapshot.fingerprint
        and _sha256(snapshot.path.read_bytes()) == snapshot.sha256,
        f"{label} changed during verification",
    )


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    requested = Path(path).absolute()
    _require_no_symlink(requested, label=label, ancestors=True)
    resolved = requested.resolve(strict=True)
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is not a real directory")
    return resolved


def _require_no_symlink(path: Path, *, label: str, ancestors: bool) -> None:
    current = Path(path).absolute()
    while True:
        _require(not current.is_symlink(), f"{label} traverses a symbolic link")
        if not ancestors or current == current.parent:
            break
        current = current.parent


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise VerificationError(f"non-finite JSON constant {value!r}")


def _loads_json(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, VerificationError) as error:
        raise VerificationError(f"{label} is not strict UTF-8 JSON") from error


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
    payload: bytes, *, label: str, canonical_pretty: bool = False
) -> dict[str, object]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} must have exactly one final LF",
    )
    value = _loads_json(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain an object")
    result = cast(dict[str, object], value)
    if canonical_pretty:
        _require(payload == _pretty_json_bytes(result), f"{label} is not canonical pretty JSON")
    return result


def _jsonl_objects(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} LF framing")
    output: list[dict[str, object]] = []
    for number, line in enumerate(payload[:-1].split(b"\n"), start=1):
        _require(line != b"", f"{label} line {number} is blank")
        value = _loads_json(line, label=f"{label} line {number}")
        _require(isinstance(value, dict), f"{label} line {number} is not an object")
        item = cast(dict[str, object], value)
        _require(line == _compact_json_bytes(item), f"{label} line {number} is not canonical")
        output.append(item)
    return tuple(output)


def _safe_manifest_name(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    _require(
        value != ""
        and not path.is_absolute()
        and path.as_posix() == value
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} contains unsafe path {value!r}",
    )
    return value


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} LF framing")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        _require(match is not None, f"{label} line {number} is malformed")
        assert match is not None
        digest, mode, raw_name = match.groups()
        name = _safe_manifest_name(raw_name, label=label)
        _require(mode == " " and name not in result, f"{label} entry {name!r} is invalid")
        _require(previous is None or name > previous, f"{label} paths are not strictly sorted")
        result[name] = digest
        previous = name
    return result


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("utf-8")


def _tree_inventory(root: Path) -> frozenset[str]:
    result: set[str] = set()
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"tree contains symbolic entry {item}")
        if item.is_dir():
            continue
        _require(item.is_file(), f"tree contains non-regular entry {item}")
        result.add(item.relative_to(root).as_posix())
    return frozenset(result)


def _verify_immutable_tree(root: Path, *, label: str) -> None:
    _require(root.stat().st_mode & 0o222 == 0, f"{label} root remains writable")
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"{label} contains a symbolic entry")
        _require(
            item.stat().st_mode & 0o222 == 0,
            f"{label} entry remains writable: {item.relative_to(root).as_posix()}",
        )


def _snapshot_tree(root: Path, *, expected: frozenset[str], label: str) -> dict[str, Snapshot]:
    _require(_tree_inventory(root) == expected, f"{label} inventory differs from contract")
    return {name: _read_snapshot(root / name, label=f"{label} {name}") for name in sorted(expected)}


def _verify_manifest_tree(
    root: Path,
    *,
    expected: frozenset[str],
    label: str,
) -> tuple[Snapshot, dict[str, Snapshot]]:
    snapshots = _snapshot_tree(root, expected=expected, label=label)
    top = snapshots["SHA256SUMS"]
    entries = _parse_sha_manifest(top.payload, label=f"{label} top manifest")
    _require(set(entries) == set(expected) - {"SHA256SUMS"}, f"{label} top inventory mismatch")
    for name, digest in entries.items():
        _require(snapshots[name].sha256 == digest, f"{label} checksum mismatch for {name}")
    return top, snapshots


def _verify_tree_bytes(left: Path, right: Path, *, expected: frozenset[str]) -> None:
    _require(_tree_inventory(left) == expected, "left twin inventory differs from contract")
    _require(_tree_inventory(right) == expected, "right twin inventory differs from contract")
    for name in sorted(expected):
        _require(
            _read_snapshot(left / name, label=f"left twin {name}").payload
            == _read_snapshot(right / name, label=f"right twin {name}").payload,
            f"producer twin bytes differ for {name}",
        )


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
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"bad {label}")
    return cast(int, value)


def _nonnegative_int(value: object, *, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"bad {label}")
    return cast(int, value)


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    _require(isinstance(value, int | float) and not isinstance(value, bool), f"bad {label}")
    result = float(cast(int | float, value))
    _require(math.isfinite(result) and (not positive or result > 0), f"bad {label}")
    return result


def _int_tuple(value: object, *, label: str, length: int) -> tuple[int, ...]:
    _require(isinstance(value, list) and len(value) == length, f"bad {label}")
    return tuple(_nonnegative_int(item, label=label) for item in cast(list[object], value))


def _canonical_sequence(value: object, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} is not text")
    sequence = cast(str, value)
    normalized = "".join(sequence.split()).upper()
    _require(
        sequence == normalized
        and _MIN_LENGTH <= len(sequence) <= _MAX_LENGTH
        and set(sequence) <= _AMINO_ACIDS,
        f"{label} is not a canonical peptide",
    )
    return sequence


def _sequence_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _identifier_set_sha256(values: Iterable[str]) -> str:
    items = sorted(set(values))
    return _sha256(b"" if not items else ("\n".join(items) + "\n").encode("ascii"))


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
        for item in sorted(examples, key=lambda row: row.example_id)
    ]
    return _sha256(("\n".join(records) + "\n").encode("utf-8"))


def _strict_csv(payload: bytes, *, schema: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    _require(
        text.endswith("\n") and "\r" not in text and not text.startswith("\ufeff"),
        f"{label} is not canonical LF-terminated UTF-8",
    )
    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require(reader.fieldnames == list(schema), f"{label} schema differs from contract")
    rows = list(reader)
    _require(
        rows and all(None not in row and None not in row.values() for row in rows),
        f"{label} is empty or ragged",
    )
    return cast(list[dict[str, str]], rows)


def _integer_text(value: str, *, label: str, minimum: int = 0) -> int:
    _require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None, f"{label} is not canonical")
    result = int(value)
    _require(result >= minimum, f"{label} is below its minimum")
    return result


def _float_text(value: str, *, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise VerificationError(f"{label} is not numeric") from error
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _load_config(path: str | Path) -> Config:
    snapshot = _read_snapshot(Path(path), label="APEX union config")
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("APEX union config is not valid UTF-8 TOML") from error
    expected_fields = {
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
    _require(
        set(raw) == expected_fields
        and raw.get("schema_version") == 1
        and raw.get("artifact") == _ARTIFACT,
        "APEX union config schema or identity differs from v1",
    )
    _require(
        raw["calibration_weighting"] == _CALIBRATION_WEIGHTING
        and raw["bootstrap_weighting"] == _BOOTSTRAP_WEIGHTING,
        "APEX union weighting policy differs from v1",
    )
    folds = _positive_int(raw["folds"], label="folds")
    _require(folds == 5, "APEX union v1 requires five folds")
    fold_examples = _int_tuple(
        raw["expected_examples_by_fold"], label="expected examples by fold", length=folds
    )
    fold_positives = _int_tuple(
        raw["expected_positive_examples_by_fold"],
        label="expected positives by fold",
        length=folds,
    )
    fold_negatives = _int_tuple(
        raw["expected_negative_examples_by_fold"],
        label="expected negatives by fold",
        length=folds,
    )
    examples = _positive_int(raw["expected_examples"], label="expected examples")
    positives = _positive_int(raw["expected_positive_examples"], label="expected positive examples")
    negatives = _positive_int(raw["expected_negative_examples"], label="expected negative examples")
    members_expected = _positive_int(raw["expected_member_count"], label="member count")
    output_rows = _positive_int(raw["expected_output_rows"], label="output rows")
    _require(
        members_expected == 8
        and positives + negatives == examples
        and sum(fold_examples) == examples
        and sum(fold_positives) == positives
        and sum(fold_negatives) == negatives
        and all(
            fold_positives[index] + fold_negatives[index] == fold_examples[index]
            for index in range(folds)
        )
        and output_rows == examples * (members_expected + 1),
        "APEX union config census arithmetic is inconsistent",
    )

    target_tables = raw["target"]
    _require(isinstance(target_tables, list), "APEX union config target table is invalid")
    targets: list[Target] = []
    endpoints_seen: set[str] = set()
    for index, value in enumerate(cast(list[object], target_tables)):
        _require(isinstance(value, dict), f"target {index} is not a table")
        item = cast(dict[str, object], value)
        _exact_fields(item, {"name", "endpoints"}, label=f"target {index}")
        name = item["name"]
        endpoints = item["endpoints"]
        _require(
            isinstance(name, str)
            and name == name.strip()
            and bool(name)
            and isinstance(endpoints, list)
            and bool(endpoints)
            and all(isinstance(endpoint, str) for endpoint in endpoints),
            f"target {index} is invalid",
        )
        endpoint_tuple = tuple(cast(list[str], endpoints))
        _require(
            len(endpoint_tuple) == len(set(endpoint_tuple))
            and not endpoints_seen.intersection(endpoint_tuple)
            and set(endpoint_tuple) <= set(_APEX_ENDPOINT_COLUMNS),
            f"target {index} endpoint ownership is invalid",
        )
        endpoints_seen.update(endpoint_tuple)
        targets.append(Target(cast(str, name), endpoint_tuple))
    _require(
        tuple((item.name, item.endpoints) for item in targets) == _TARGET_ENDPOINTS,
        "APEX target-to-endpoint map differs from v1",
    )

    member_tables = raw["member"]
    _require(
        isinstance(member_tables, list) and len(member_tables) == members_expected,
        "APEX member inventory has the wrong size",
    )
    members: list[Member] = []
    for index, value in enumerate(cast(list[object], member_tables)):
        _require(isinstance(value, dict), f"member {index} is not a table")
        item = cast(dict[str, object], value)
        _exact_fields(item, {"member_id", "checkpoint_sha256"}, label=f"member {index}")
        member_id = item["member_id"]
        _require(isinstance(member_id, str) and bool(member_id), f"member {index} has no ID")
        members.append(
            Member(
                member_id=cast(str, member_id),
                checkpoint_sha256=_sha_field(
                    item["checkpoint_sha256"], label=f"member {index} checkpoint"
                ),
            )
        )
    _require(
        len({item.member_id for item in members}) == len(members),
        "APEX member IDs are not unique",
    )

    calibration = raw["calibration"]
    _require(isinstance(calibration, dict), "calibration settings are not a table")
    calibration_table = cast(dict[str, object], calibration)
    _exact_fields(
        calibration_table,
        {"l2", "prior_strength", "max_iterations", "tolerance"},
        label="calibration settings",
    )
    hash_names = (
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
    hashes = {name: _sha_field(raw[name], label=name) for name in hash_names}
    source_commit = raw["expected_apex_source_commit"]
    _require(
        isinstance(source_commit, str) and _GIT_RE.fullmatch(source_commit) is not None,
        "APEX source commit is invalid",
    )
    raw_sequences = _positive_int(raw["expected_apex_sequences"], label="raw sequences")
    modeled_sequences = _positive_int(raw["expected_modeled_sequences"], label="modeled sequences")
    extra_sequences = _nonnegative_int(
        raw["expected_apex_extra_sequences"], label="extra sequences"
    )
    _require(
        raw_sequences == modeled_sequences + extra_sequences,
        "APEX sequence census does not imply complete coverage",
    )
    activity_threshold = _finite_float(
        raw["activity_threshold_um"], label="activity threshold", positive=True
    )
    homology_threshold = _finite_float(
        raw["homology_identity_threshold"], label="homology threshold", positive=True
    )
    clip = _finite_float(raw["probability_clip_epsilon"], label="probability clip", positive=True)
    bins = _positive_int(raw["calibration_bins"], label="calibration bins")
    replicates = _positive_int(raw["bootstrap_replicates"], label="bootstrap replicates")
    seed = _nonnegative_int(raw["bootstrap_seed"], label="bootstrap seed")
    _require(
        activity_threshold == 16.0
        and homology_threshold == 0.8
        and 0 < clip < 0.5
        and bins >= 2
        and replicates == 1000
        and seed == 42,
        "APEX v1 thresholds or bootstrap constants differ",
    )
    prior_strength = _finite_float(calibration_table["prior_strength"], label="calibration prior")
    _require(prior_strength >= 0, "calibration prior strength is negative")
    return Config(
        path=snapshot.path,
        sha256=snapshot.sha256,
        activity_threshold_um=activity_threshold,
        folds=folds,
        homology_identity_threshold=homology_threshold,
        calibration_bins=bins,
        probability_clip_epsilon=clip,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        calibration_weighting=cast(str, raw["calibration_weighting"]),
        bootstrap_weighting=cast(str, raw["bootstrap_weighting"]),
        expected_examples=examples,
        expected_source_observations=_positive_int(
            raw["expected_source_observations"], label="source observations"
        ),
        expected_positive_examples=positives,
        expected_negative_examples=negatives,
        expected_modeled_sequences=modeled_sequences,
        expected_modeled_homology_components=_positive_int(
            raw["expected_modeled_homology_components"], label="modeled homology components"
        ),
        expected_modeled_union_components=_positive_int(
            raw["expected_modeled_union_components"], label="modeled union components"
        ),
        expected_examples_by_fold=fold_examples,
        expected_positive_examples_by_fold=fold_positives,
        expected_negative_examples_by_fold=fold_negatives,
        expected_output_rows=output_rows,
        **hashes,
        expected_apex_source_commit=cast(str, source_commit),
        expected_apex_sequences=raw_sequences,
        expected_apex_extra_sequences=extra_sequences,
        expected_member_count=members_expected,
        expected_reconciliation_comparisons=_positive_int(
            raw["expected_reconciliation_comparisons"], label="reconciliation comparisons"
        ),
        calibration_l2=_finite_float(
            calibration_table["l2"], label="calibration l2", positive=True
        ),
        calibration_prior_strength=prior_strength,
        calibration_max_iterations=_positive_int(
            calibration_table["max_iterations"], label="calibration iterations"
        ),
        calibration_tolerance=_finite_float(
            calibration_table["tolerance"], label="calibration tolerance", positive=True
        ),
        targets=tuple(targets),
        members=tuple(members),
    )


def _git(repository: Path, arguments: Sequence[str], *, label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise VerificationError(f"Git check failed for {label}") from error


def _verify_repository(repository: Path, expected_commit: str) -> None:
    _require(_GIT_RE.fullmatch(expected_commit) is not None, "expected Git commit is invalid")
    head = _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"], label="HEAD")
    _require(head.decode().strip() == expected_commit, "HEAD differs from expected commit")
    _require(
        _git(
            repository,
            ["diff", "--no-ext-diff", "--quiet", "--exit-code", "--"],
            label="worktree",
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
        _git(
            repository,
            ["ls-files", "--others", "--exclude-standard", "-z"],
            label="untracked files",
        )
        == b"",
        "repository contains untracked files",
    )
    origin = _git(
        repository,
        ["rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
        label="origin/main",
    )
    _require(
        origin.decode().strip() == expected_commit,
        "expected verifier commit is not synchronized to origin/main",
    )


def _committed_blob(repository: Path, commit: str, logical_path: str) -> bytes:
    kind = _git(repository, ["cat-file", "-t", f"{commit}:{logical_path}"], label=logical_path)
    _require(kind == b"blob\n", f"committed path is not a blob: {logical_path}")
    return _git(
        repository,
        ["cat-file", "blob", f"{commit}:{logical_path}"],
        label=logical_path,
    )


def _verify_execution_source(repository: Path, expected_commit: str) -> str:
    requested = Path(__file__)
    _require_no_symlink(requested, label="executing verifier", ancestors=True)
    executing = requested.resolve(strict=True)
    expected = (repository / _VERIFIER_MODULE_PATH).resolve(strict=True)
    _require(executing == expected, "executing verifier is a stale installation")
    snapshot = _read_snapshot(expected, label="verifier source")
    _require(
        snapshot.payload == _committed_blob(repository, expected_commit, _VERIFIER_MODULE_PATH),
        "executing verifier differs from committed blob",
    )
    return snapshot.sha256


def _verify_code_manifest(
    snapshot: Snapshot,
    *,
    repository: Path,
    expected_commit: str,
    config: Config,
) -> dict[str, object]:
    entries = _parse_sha_manifest(snapshot.payload, label="producer code manifest")
    source_root = repository / "src" / "amp_challenge"
    _require(source_root.is_dir() and not source_root.is_symlink(), "source tree is unavailable")
    source_paths: set[str] = set()
    for path in source_root.rglob("*.py"):
        _require(not path.is_symlink(), f"source entry is symbolic: {path}")
        if path.is_file():
            source_paths.add(path.relative_to(repository).as_posix())
    expected_paths = set(_FIXED_CODE_PATHS) | source_paths
    _require(set(entries) == expected_paths, "producer code manifest inventory differs")
    _require(
        _PRODUCER_MODULE_PATH in entries and _VERIFIER_MODULE_PATH in entries,
        "producer code manifest omits producer or independent verifier",
    )
    for logical in sorted(expected_paths):
        source = _read_snapshot(repository / logical, label=f"code inventory {logical}")
        _require(source.sha256 == entries[logical], f"code checksum mismatch for {logical}")
        _require(
            source.payload == _committed_blob(repository, expected_commit, logical),
            f"code entry differs from committed blob: {logical}",
        )
    _require(
        entries[_LOGICAL_CONFIG_PATH] == config.sha256,
        "producer code manifest does not bind supplied config",
    )
    return {
        "schema_version": 1,
        "code_manifest_sha256": snapshot.sha256,
        "inventory_entries": len(entries),
        "executing_module": {
            "logical_path": _PRODUCER_MODULE_PATH,
            "sha256": entries[_PRODUCER_MODULE_PATH],
        },
        "logical_config": {
            "logical_path": _LOGICAL_CONFIG_PATH,
            "sha256": config.sha256,
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


def _verify_handshake(twin_root: Path) -> dict[str, object]:
    receipt_root = twin_root / "node-receipts"
    _require(
        receipt_root.is_dir() and not receipt_root.is_symlink(),
        "producer node-receipts directory is missing",
    )
    _require(
        _tree_inventory(receipt_root) == {"0.receipt", "1.receipt", "0.ack", "1.ack"},
        "producer handshake inventory differs",
    )
    _require(receipt_root.stat().st_mode & 0o222 == 0, "producer handshake remains writable")
    job_id = twin_root.name
    _require(job_id.isdigit(), "producer twin root basename is not its array job ID")
    receipt_payloads: dict[int, bytes] = {}
    node_names: dict[int, str] = {}
    for task in (0, 1):
        receipt = _read_snapshot(receipt_root / f"{task}.receipt", label=f"task {task} receipt")
        _require(receipt.path.stat().st_mode & 0o222 == 0, f"task {task} receipt is writable")
        _require(
            receipt.payload.endswith(b"\n") and b"\r" not in receipt.payload,
            f"task {task} receipt has invalid framing",
        )
        try:
            lines = receipt.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise VerificationError("producer node receipt is not UTF-8") from error
        _require(
            len(lines) == 3
            and lines[0] == f"array_job_id={job_id}"
            and lines[1] == f"array_task_id={task}"
            and lines[2].startswith("node_name="),
            f"task {task} receipt is invalid",
        )
        node_name = lines[2].removeprefix("node_name=")
        _require(
            _SAFE_NODE_RE.fullmatch(node_name) is not None,
            f"task {task} node name is unsafe",
        )
        receipt_payloads[task] = receipt.payload
        node_names[task] = node_name
    _require(node_names[0] != node_names[1], "producer twins did not overlap on distinct nodes")
    receipt_hashes = {str(task): _sha256(payload) for task, payload in receipt_payloads.items()}
    acknowledgement_hashes: dict[str, str] = {}
    for task in (0, 1):
        acknowledgement = _read_snapshot(
            receipt_root / f"{task}.ack", label=f"task {task} acknowledgement"
        )
        _require(
            acknowledgement.path.stat().st_mode & 0o222 == 0,
            f"task {task} acknowledgement is writable",
        )
        expected = f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode(
            "ascii"
        )
        _require(
            acknowledgement.payload == expected,
            f"task {task} acknowledgement does not cross-bind its sibling",
        )
        acknowledgement_hashes[str(task)] = acknowledgement.sha256
    return {
        "distinct_nodes": True,
        "bidirectional_acknowledgement": True,
        "receipt_sha256": receipt_hashes,
        "acknowledgement_sha256": acknowledgement_hashes,
    }


def _expected_frozen_inputs(
    base_run: Path,
    independent_receipt: Snapshot,
    apex_root: Path,
) -> dict[str, str]:
    result = {
        f"base/{name}": _read_snapshot(base_run / name, label=f"base input {name}").sha256
        for name in sorted(_BASE_TREE)
    }
    result["base-independent-receipt.json"] = independent_receipt.sha256
    result.update(
        {
            f"apex/{name}": _read_snapshot(apex_root / name, label=f"raw APEX {name}").sha256
            for name in sorted(_RAW_APEX_TREE)
        }
    )
    return dict(sorted(result.items()))


def _verify_producer_publication(
    run: Path,
    *,
    base_run: Path,
    independent_receipt: Snapshot,
    apex_root: Path,
    repository: Path,
    expected_commit: str,
    config: Config,
) -> tuple[dict[str, Snapshot], dict[str, Snapshot], dict[str, object]]:
    top, snapshots = _verify_manifest_tree(
        run, expected=_PRODUCER_TREE, label="APEX union producer publication"
    )
    semantic_top = snapshots["apex/SHA256SUMS"]
    semantic_entries = _parse_sha_manifest(
        semantic_top.payload, label="APEX union semantic manifest"
    )
    _require(
        set(semantic_entries) == set(_SEMANTIC_OUTPUTS),
        "APEX union semantic manifest inventory differs",
    )
    for name, digest in semantic_entries.items():
        _require(
            snapshots[f"apex/{name}"].sha256 == digest,
            f"APEX union semantic checksum mismatch for {name}",
        )
        outer = _parse_sha_manifest(top.payload, label="APEX union top manifest")
        _require(outer[f"apex/{name}"] == digest, f"producer top manifests disagree for {name}")
    outer = _parse_sha_manifest(top.payload, label="APEX union top manifest")
    _require(
        outer["apex/SHA256SUMS"] == semantic_top.sha256,
        "producer top manifest does not bind semantic top",
    )
    code = snapshots["CODE_SHA256SUMS"]
    frozen = snapshots["FROZEN_INPUT_SHA256SUMS"]
    code_attestation = _verify_code_manifest(
        code,
        repository=repository,
        expected_commit=expected_commit,
        config=config,
    )
    observed_frozen = _parse_sha_manifest(frozen.payload, label="producer frozen inputs")
    expected_frozen = _expected_frozen_inputs(base_run, independent_receipt, apex_root)
    _require(observed_frozen == expected_frozen, "producer frozen inputs differ from raw sources")
    return (
        snapshots,
        {name: snapshots[f"apex/{name}"] for name in sorted(_SEMANTIC_OUTPUTS)},
        code_attestation,
    )


def _fold_summary(examples: Sequence[Example], folds: int) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for fold in range(folds):
        rows = [item for item in examples if item.fold == fold]
        result[str(fold)] = {
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
    return result


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
    config: Config,
) -> tuple[Example, ...]:
    documents = _jsonl_objects(examples_snapshot.payload, label="accepted Gate-1 examples")
    raw_by_id: dict[str, dict[str, object]] = {}
    previous: str | None = None
    for index, row in enumerate(documents):
        _exact_fields(row, _EXAMPLE_FIELDS, label=f"accepted example {index}")
        _require(row.get("schema_version") == 1, f"accepted example {index} schema changed")
        example_id = _sha_field(row.get("example_id"), label=f"example {index} ID")
        context_id = _sha_field(row.get("assay_context_id"), label=f"example {index} context ID")
        _require(
            example_id == context_id
            and example_id not in raw_by_id
            and (previous is None or example_id > previous),
            f"accepted example {index} identity/order is invalid",
        )
        previous = example_id
        sequence = _canonical_sequence(row.get("sequence"), label=f"example {index} sequence")
        sequence_id = _sha_field(row.get("sequence_id"), label=f"example {index} sequence ID")
        _require(_sequence_id(sequence) == sequence_id, f"example {index} sequence ID mismatch")
        target = row.get("canonical_target")
        gram = row.get("gram")
        label = row.get("label")
        fold = row.get("fold")
        _require(
            isinstance(target, str) and target in config.target_by_name,
            f"example {index} has unsupported target",
        )
        _require(gram in {"positive", "negative"}, f"example {index} has invalid Gram")
        _require(
            isinstance(label, int) and not isinstance(label, bool) and label in {0, 1},
            f"example {index} has invalid label",
        )
        _require(
            isinstance(fold, int) and not isinstance(fold, bool) and fold in range(config.folds),
            f"example {index} has invalid fold",
        )
        _positive_int(row.get("source_observations"), label=f"example {index} observations")
        _sha_field(row.get("homology_component_id"), label=f"example {index} homology")
        _sha_field(row.get("union_component_id"), label=f"example {index} union")
        raw_by_id[example_id] = row

    oof_rows = _strict_csv(
        oof_snapshot.payload,
        schema=_BASE_OOF_SCHEMA,
        label="accepted Gate-1 OOF metadata",
    )
    identities: dict[str, float] = {}
    support: dict[str, set[str]] = defaultdict(set)
    for number, row in enumerate(oof_rows, start=2):
        model = row["model"]
        example_id = row["example_id"]
        source = raw_by_id.get(example_id)
        _require(
            model in _BASE_MODELS and source is not None and example_id not in support[model],
            f"accepted OOF row {number} has invalid model/support identity",
        )
        assert source is not None
        support[model].add(example_id)
        for field in (
            "assay_context_id",
            "sequence_id",
            "sequence",
            "canonical_target",
            "gram",
            "homology_component_id",
            "union_component_id",
        ):
            _require(row[field] == source[field], f"accepted OOF row {number} metadata differs")
        for field in ("label", "source_observations", "fold"):
            _require(
                _integer_text(row[field], label=f"OOF row {number} {field}") == source[field],
                f"accepted OOF row {number} numeric metadata differs",
            )
        identity = _float_text(row["max_train_identity"], label=f"OOF row {number} identity")
        probability = _float_text(row["probability"], label=f"OOF row {number} probability")
        _require(
            0 <= identity < config.homology_identity_threshold and 0 <= probability <= 1,
            f"accepted OOF row {number} identity/probability is invalid",
        )
        if example_id in identities:
            _require(
                identities[example_id] == identity,
                "accepted base models disagree on max_train_identity metadata",
            )
        else:
            identities[example_id] = identity
    expected_ids = set(raw_by_id)
    _require(
        set(support) == set(_BASE_MODELS) and all(ids == expected_ids for ids in support.values()),
        "accepted base models do not share complete context support",
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
    _verify_example_census(examples, config=config)
    return examples


def _verify_example_census(examples: Sequence[Example], *, config: Config) -> None:
    sequences = {item.sequence_id for item in examples}
    homology = {item.homology_component_id for item in examples}
    unions = {item.union_component_id for item in examples}
    positives = sum(item.label for item in examples)
    _require(
        len(examples) == config.expected_examples
        and positives == config.expected_positive_examples
        and len(examples) - positives == config.expected_negative_examples
        and sum(item.source_observations for item in examples)
        == config.expected_source_observations
        and len(sequences) == config.expected_modeled_sequences
        and len(homology) == config.expected_modeled_homology_components
        and len(unions) == config.expected_modeled_union_components
        and _identifier_set_sha256(sequences) == config.expected_union_sequence_ids_sha256,
        "accepted modeled context census differs from config",
    )
    summary = _fold_summary(examples, config.folds)
    _require(
        tuple(summary[str(fold)]["examples"] for fold in range(config.folds))
        == config.expected_examples_by_fold
        and tuple(summary[str(fold)]["positives"] for fold in range(config.folds))
        == config.expected_positive_examples_by_fold
        and tuple(summary[str(fold)]["negatives"] for fold in range(config.folds))
        == config.expected_negative_examples_by_fold,
        "accepted modeled fold census differs from config",
    )
    sequence_assignments: dict[str, set[tuple[int, str, str, str]]] = defaultdict(set)
    homology_folds: dict[str, set[int]] = defaultdict(set)
    homology_unions: dict[str, set[str]] = defaultdict(set)
    union_folds: dict[str, set[int]] = defaultdict(set)
    for item in examples:
        sequence_assignments[item.sequence_id].add(
            (item.fold, item.homology_component_id, item.union_component_id, item.sequence)
        )
        homology_folds[item.homology_component_id].add(item.fold)
        homology_unions[item.homology_component_id].add(item.union_component_id)
        union_folds[item.union_component_id].add(item.fold)
    _require(
        all(len(values) == 1 for values in sequence_assignments.values())
        and all(len(values) == 1 for values in homology_folds.values())
        and all(len(values) == 1 for values in homology_unions.values())
        and all(len(values) == 1 for values in union_folds.values()),
        "accepted sequence/component assignments leak across folds",
    )
    for fold in range(config.folds):
        _require(
            {item.label for item in examples if item.fold == fold} == {0, 1}
            and {item.label for item in examples if item.fold != fold} == {0, 1},
            f"fold {fold} or its training complement lacks a label class",
        )


def _verify_base_chain(
    base_twin_root: Path,
    independent: Snapshot,
    *,
    config: Config,
) -> tuple[Path, tuple[Example, ...], dict[str, object], dict[str, object], dict[str, object]]:
    immediate = {entry.name: entry for entry in base_twin_root.iterdir()}
    _require(
        set(immediate) == {"0", "1", "node-receipts"}
        and all(entry.is_dir() and not entry.is_symlink() for entry in immediate.values()),
        "accepted Gate-1 twin-root inventory differs",
    )
    base_runs = (base_twin_root / "0", base_twin_root / "1")
    for index, run in enumerate(base_runs):
        _verify_immutable_tree(run, label=f"accepted Gate-1 twin {index}")
    _verify_tree_bytes(base_runs[0], base_runs[1], expected=_BASE_TREE)
    upstream_handshake = _verify_handshake(base_twin_root)
    top, snapshots = _verify_manifest_tree(
        base_runs[0], expected=_BASE_TREE, label="accepted Gate-1 publication"
    )
    semantic = _parse_sha_manifest(
        snapshots["gate1/SHA256SUMS"].payload, label="accepted Gate-1 semantic manifest"
    )
    _require(
        top.sha256 == config.expected_base_publication_top_sha256
        and snapshots["gate1/SHA256SUMS"].sha256 == config.expected_base_semantic_top_sha256
        and set(semantic) == set(_BASE_SEMANTIC),
        "accepted Gate-1 content address or inventory differs",
    )
    outer = _parse_sha_manifest(top.payload, label="accepted Gate-1 publication manifest")
    for name, digest in semantic.items():
        _require(
            snapshots[f"gate1/{name}"].sha256 == digest and outer[f"gate1/{name}"] == digest,
            f"accepted Gate-1 semantic link differs for {name}",
        )
    expected_hashes = {
        "gate1/examples.jsonl": config.expected_base_examples_sha256,
        "gate1/folds.json": config.expected_base_folds_sha256,
        "gate1/oof_predictions.csv": config.expected_base_oof_sha256,
        "gate1/manifest.json": config.expected_base_manifest_sha256,
        "gate1/split_receipt.json": config.expected_base_split_receipt_sha256,
    }
    for name, digest in expected_hashes.items():
        _require(snapshots[name].sha256 == digest, f"accepted Gate-1 {name} hash differs")
    _parse_sha_manifest(snapshots["CODE_SHA256SUMS"].payload, label="accepted Gate-1 code")
    _parse_sha_manifest(
        snapshots["FROZEN_INPUT_SHA256SUMS"].payload,
        label="accepted Gate-1 frozen inputs",
    )
    examples = _read_examples(
        snapshots["gate1/examples.jsonl"],
        snapshots["gate1/oof_predictions.csv"],
        config=config,
    )
    folds = _json_object(
        snapshots["gate1/folds.json"].payload,
        label="accepted Gate-1 folds",
        canonical_pretty=True,
    )
    _exact_fields(
        folds,
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
    expected_assignments = [
        {
            "example_id": item.example_id,
            "fold": item.fold,
            "homology_component_id": item.homology_component_id,
            "sequence_id": item.sequence_id,
            "union_component_id": item.union_component_id,
        }
        for item in examples
    ]
    _require(
        folds.get("schema_version") == 1
        and folds.get("artifact") == "gate1_context_union_fold_reuse"
        and folds.get("assignment_policy") == _ASSIGNMENT_POLICY
        and folds.get("identity_threshold") == config.homology_identity_threshold
        and folds.get("maximum_cross_fold_identity")
        == max(item.max_train_identity for item in examples)
        and folds.get("assignments") == expected_assignments
        and folds.get("folds") == _fold_summary(examples, config.folds)
        and folds.get("canonical_targets_by_fold") == _target_summary(examples, config.folds),
        "accepted Gate-1 fold document differs from reconstructed assignments",
    )
    manifest = _json_object(
        snapshots["gate1/manifest.json"].payload,
        label="accepted Gate-1 manifest",
        canonical_pretty=True,
    )
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "context_audit": "context_audit.jsonl",
        "examples": "examples.jsonl",
        "folds": "folds.json",
        "metrics": "metrics.json",
        "oof": "oof_predictions.csv",
        "split_receipt": "split_receipt.json",
    }
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact") == _BASE_ARTIFACT
        and manifest.get("status") == _BASE_STATUS
        and manifest.get("config_sha256") == config.expected_base_config_sha256
        and manifest.get("models") == list(_BASE_MODELS)
        and isinstance(artifacts, dict)
        and set(cast(dict[str, object], artifacts)) == set(expected_artifacts),
        "accepted Gate-1 manifest identity or artifact inventory differs",
    )
    assert isinstance(artifacts, dict)
    for key, filename in expected_artifacts.items():
        entry = artifacts[key]
        _require(
            isinstance(entry, dict)
            and entry.get("filename") == filename
            and entry.get("sha256") == semantic[filename],
            f"accepted Gate-1 manifest artifact {key} differs",
        )
    label_summary = manifest.get("label_summary")
    _require(
        isinstance(label_summary, dict)
        and label_summary.get("context_examples") == len(examples)
        and label_summary.get("positive_examples") == sum(item.label for item in examples)
        and label_summary.get("negative_examples") == sum(1 - item.label for item in examples)
        and label_summary.get("source_observations")
        == sum(item.source_observations for item in examples)
        and label_summary.get("modeled_sequences") == len({item.sequence_id for item in examples})
        and label_summary.get("by_fold") == folds["folds"]
        and label_summary.get("canonical_targets_by_fold") == folds["canonical_targets_by_fold"],
        "accepted Gate-1 label summary differs",
    )
    split_receipt = _json_object(
        snapshots["gate1/split_receipt.json"].payload,
        label="accepted Gate-1 split receipt",
        canonical_pretty=True,
    )
    census = split_receipt.get("census")
    invariants = split_receipt.get("invariants")
    _require(
        split_receipt.get("schema_version") == 1
        and split_receipt.get("artifact") == "gate1_union_accepted_split_consumption_receipt"
        and split_receipt.get("status") == "passed"
        and split_receipt.get("assignment_policy") == _ASSIGNMENT_POLICY
        and isinstance(invariants, dict)
        and bool(invariants)
        and all(value is True for value in invariants.values())
        and isinstance(census, dict)
        and census.get("context_examples") == config.expected_examples
        and census.get("source_observations") == config.expected_source_observations
        and census.get("modeled_sequences") == config.expected_modeled_sequences
        and census.get("homology_components") == 597
        and census.get("union_components") == 278
        and census.get("parser_sequences") == 1113,
        "accepted Gate-1 split receipt differs",
    )
    independent_document = _json_object(
        independent.payload,
        label="accepted Gate-1 independent receipt",
        canonical_pretty=True,
    )
    checks = independent_document.get("checks")
    handshake = independent_document.get("production_handshake")
    _require(
        independent.sha256 == config.expected_base_independent_receipt_sha256
        and independent_document.get("schema_version") == 1
        and independent_document.get("artifact") == f"{_BASE_ARTIFACT}_independent_verification"
        and independent_document.get("status") == "passed"
        and independent_document.get("publication_top_manifest_sha256") == top.sha256
        and independent_document.get("gate1_top_manifest_sha256")
        == snapshots["gate1/SHA256SUMS"].sha256
        and independent_document.get("config_sha256") == config.expected_base_config_sha256
        and independent_document.get("artifact_sha256") == semantic
        and isinstance(checks, dict)
        and set(checks) == set(_UPSTREAM_GATE1_CHECKS)
        and all(value is True for value in checks.values())
        and isinstance(handshake, dict)
        and handshake.get("distinct_nodes") is True
        and handshake.get("bidirectional_acknowledgement") is True
        and handshake == upstream_handshake,
        "accepted Gate-1 independent receipt differs",
    )
    for key in ("receipt_sha256", "acknowledgement_sha256"):
        values = cast(dict[str, object], handshake).get(key)
        _require(isinstance(values, dict) and set(values) == {"0", "1"}, "bad Gate-1 handshake")
        assert isinstance(values, dict)
        for value in values.values():
            _sha_field(value, label=f"Gate-1 {key}")
    return base_runs[0], examples, manifest, split_receipt, upstream_handshake


def _verify_raw_apex_manifest(
    snapshot: Snapshot,
    *,
    predictions: Snapshot,
    reconciliation: Snapshot,
    config: Config,
) -> dict[str, object]:
    document = _json_object(snapshot.payload, label="raw APEX manifest")
    outputs = document.get("outputs")
    fidelity = document.get("reconciliation")
    configured_members = [asdict(item) for item in config.members]
    configured_endpoints = [endpoint for target in config.targets for endpoint in target.endpoints]
    _require(
        document.get("format_version") == 1
        and document.get("mode") == "apex_member_introspection"
        and document.get("source_commit") == config.expected_apex_source_commit
        and document.get("member_count") == config.expected_member_count
        and document.get("n_sequences") == config.expected_apex_sequences
        and document.get("endpoints") == [label for label, _ in _APEX_ENDPOINTS]
        and configured_endpoints == list(_APEX_ENDPOINT_COLUMNS)
        and document.get("members") == configured_members
        and isinstance(outputs, dict)
        and set(outputs) == {"apex_member_predictions.csv", "apex_member_reconciliation.csv"}
        and outputs.get("apex_member_predictions.csv") == predictions.sha256
        and outputs.get("apex_member_reconciliation.csv") == reconciliation.sha256
        and isinstance(fidelity, dict)
        and fidelity.get("all_passed") is True
        and fidelity.get("comparison_count") == config.expected_reconciliation_comparisons,
        "raw APEX manifest contract differs",
    )
    return document


def _read_raw_apex_grid(
    snapshot: Snapshot,
    *,
    manifest: Mapping[str, object],
    config: Config,
) -> RawGrid:
    rows = _strict_csv(snapshot.payload, schema=_RAW_SCHEMA, label="raw APEX predictions")
    members = tuple(item.member_id for item in config.members)
    member_hashes = {item.member_id: item.checkpoint_sha256 for item in config.members}
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    sequences: dict[str, str] = {}
    for number, row in enumerate(rows, start=2):
        sequence = _canonical_sequence(row["sequence"], label=f"raw APEX row {number} sequence")
        sequence_id = row["sequence_id"]
        member = row["member_id"]
        _require(
            sequence_id == _sequence_id(sequence)
            and row["model_family"] == "apex_pathogen_member"
            and row["model_version"] == config.expected_apex_source_commit
            and member_hashes.get(member) == row["member_checkpoint_sha256"]
            and member not in values[sequence_id],
            f"raw APEX row {number} has invalid sequence/member identity",
        )
        endpoints = {
            endpoint: _float_text(row[endpoint], label=f"raw APEX row {number} {endpoint}")
            for endpoint in _APEX_ENDPOINT_COLUMNS
        }
        _require(
            all(value > 0 for value in endpoints.values()),
            f"raw APEX row {number} contains non-positive MIC",
        )
        broad = _float_text(
            row["apex_member_mean_mic_um"], label=f"raw APEX row {number} broad mean"
        )
        _require(
            broad > 0 and broad == math.fsum(endpoints.values()) / len(endpoints),
            f"raw APEX row {number} broad mean does not round trip",
        )
        if sequence_id in sequences:
            _require(sequences[sequence_id] == sequence, "raw APEX sequence-ID collision")
        else:
            sequences[sequence_id] = sequence
        values[sequence_id][member] = endpoints
    expected_members = set(members)
    _require(
        len(rows) == config.expected_apex_sequences * config.expected_member_count
        and len(values) == config.expected_apex_sequences
        and manifest.get("n_sequences") == len(values)
        and all(set(member_values) == expected_members for member_values in values.values())
        and _identifier_set_sha256(values) == config.expected_apex_sequence_ids_sha256,
        "raw APEX prediction grid is not the frozen rectangle",
    )
    return RawGrid(
        members=members,
        member_hashes=member_hashes,
        values={key: dict(value) for key, value in values.items()},
        sequences=sequences,
        rows=len(rows),
    )


def _verify_raw_apex_reconciliation(
    snapshot: Snapshot,
    *,
    manifest: Mapping[str, object],
    raw: RawGrid,
    config: Config,
) -> None:
    rows = _strict_csv(
        snapshot.payload,
        schema=_RECONCILIATION_SCHEMA,
        label="raw APEX reconciliation",
    )
    summary = manifest.get("reconciliation")
    _require(isinstance(summary, dict), "raw APEX reconciliation summary is absent")
    assert isinstance(summary, dict)
    absolute_tolerance = _finite_float(
        summary.get("absolute_tolerance"), label="raw APEX absolute tolerance"
    )
    relative_tolerance = _finite_float(
        summary.get("relative_tolerance"), label="raw APEX relative tolerance"
    )
    _require(
        absolute_tolerance >= 0 and relative_tolerance >= 0,
        "raw APEX reconciliation tolerances are negative",
    )
    endpoint_columns = dict(_APEX_ENDPOINTS)
    endpoints = set(endpoint_columns) | {"broad_mean"}
    seen: set[tuple[str, str]] = set()
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for number, row in enumerate(rows, start=2):
        sequence = _canonical_sequence(
            row["sequence"], label=f"reconciliation row {number} sequence"
        )
        sequence_id = row["sequence_id"]
        endpoint = row["endpoint"]
        key = (sequence_id, endpoint)
        _require(
            _sequence_id(sequence) == sequence_id
            and raw.sequences.get(sequence_id) == sequence
            and endpoint in endpoints
            and key not in seen,
            f"reconciliation row {number} identity/endpoint differs",
        )
        seen.add(key)
        member_count = _integer_text(
            row["member_count"], label=f"reconciliation row {number} member count", minimum=1
        )
        introspected = _float_text(
            row["introspected_mean_mic_um"], label=f"reconciliation row {number} introspected"
        )
        fidelity = _float_text(
            row["fidelity_mean_mic_um"], label=f"reconciliation row {number} fidelity"
        )
        absolute_error = _float_text(
            row["absolute_error"], label=f"reconciliation row {number} absolute error"
        )
        relative_error = _float_text(
            row["relative_error"], label=f"reconciliation row {number} relative error"
        )
        allowed_error = _float_text(
            row["allowed_error"], label=f"reconciliation row {number} allowed error"
        )
        if endpoint == "broad_mean":
            member_values = [
                math.fsum(raw.values[sequence_id][member].values()) / len(_APEX_ENDPOINTS)
                for member in raw.members
            ]
        else:
            column = endpoint_columns[endpoint]
            member_values = [raw.values[sequence_id][member][column] for member in raw.members]
        expected_introspected = math.fsum(member_values) / len(member_values)
        expected_absolute = abs(introspected - fidelity)
        expected_relative = expected_absolute / abs(fidelity)
        expected_allowed = absolute_tolerance + relative_tolerance * abs(fidelity)
        _require(
            member_count == len(raw.members)
            and introspected > 0
            and fidelity > 0
            and absolute_error >= 0
            and relative_error >= 0
            and allowed_error >= 0
            and math.isclose(
                introspected,
                expected_introspected,
                rel_tol=1e-15,
                abs_tol=1e-12,
            )
            and absolute_error == expected_absolute
            and relative_error == expected_relative
            and allowed_error == expected_allowed
            and absolute_error <= allowed_error
            and row["passed"] == "true",
            f"reconciliation row {number} does not independently reconcile",
        )
        maximum_absolute = max(maximum_absolute, absolute_error)
        maximum_relative = max(maximum_relative, relative_error)
    expected_pairs = {
        (sequence_id, endpoint) for sequence_id in raw.sequences for endpoint in endpoints
    }
    _require(
        seen == expected_pairs
        and len(rows) == config.expected_reconciliation_comparisons
        and summary.get("comparison_count") == len(rows)
        and summary.get("max_absolute_error") == maximum_absolute
        and summary.get("max_relative_error") == maximum_relative
        and summary.get("all_passed") is True,
        "raw APEX reconciliation coverage or summary differs",
    )


def _join_contexts(
    examples: Sequence[Example],
    *,
    raw: RawGrid,
    config: Config,
) -> tuple[JoinedContext, ...]:
    modeled_ids = {item.sequence_id for item in examples}
    raw_ids = set(raw.sequences)
    _require(
        not (modeled_ids - raw_ids)
        and len(raw_ids - modeled_ids) == config.expected_apex_extra_sequences
        and len(modeled_ids & raw_ids) == config.expected_modeled_sequences,
        "raw APEX sequences do not cover the complete modeled panel",
    )
    threshold_log = math.log10(config.activity_threshold_um)
    joined: list[JoinedContext] = []
    for example in examples:
        _require(
            raw.sequences[example.sequence_id] == example.sequence,
            f"raw/base sequence bytes differ for {example.sequence_id}",
        )
        target = config.target_by_name.get(example.canonical_target)
        _require(target is not None, f"no exact APEX target map for {example.canonical_target!r}")
        assert target is not None
        signals: dict[str, float] = {}
        for member in raw.members:
            endpoints = raw.values[example.sequence_id][member]
            log_mean = math.fsum(math.log10(endpoints[name]) for name in target.endpoints) / len(
                target.endpoints
            )
            signal = threshold_log - log_mean
            _require(math.isfinite(signal), "APEX activity signal is non-finite")
            signals[member] = signal
        joined.append(JoinedContext(example=example, target=target, signals=signals))
    _require(len(joined) == config.expected_examples, "context join lost modeled contexts")
    return tuple(joined)


def _coverage_document(
    examples: Sequence[Example],
    *,
    raw: RawGrid,
    joined: Sequence[JoinedContext],
    config: Config,
) -> dict[str, object]:
    modeled = {item.sequence_id for item in examples}
    available = set(raw.sequences)
    extras = sorted(available - modeled)
    targets = sorted({item.canonical_target for item in examples})
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
            "sequences": len(modeled),
            "homology_components": len({item.homology_component_id for item in examples}),
            "union_components": len({item.union_component_id for item in examples}),
            "sequence_ids_sha256": _identifier_set_sha256(modeled),
            "assignment_sha256": _assignment_sha256(examples),
            "by_fold": _fold_summary(examples, config.folds),
            "canonical_targets_by_fold": _target_summary(examples, config.folds),
            "gram_by_canonical_target": {
                gram: {
                    target: sum(
                        item.gram == gram and item.canonical_target == target for item in examples
                    )
                    for target in targets
                }
                for gram in ("negative", "positive")
            },
        },
        "raw_apex": {
            "rows": raw.rows,
            "members": len(raw.members),
            "sequences": len(available),
            "sequence_ids_sha256": _identifier_set_sha256(available),
        },
        "coverage": {
            "all_modeled_sequences_covered": modeled <= available,
            "covered_sequences": len(modeled & available),
            "covered_sequence_ids_sha256": _identifier_set_sha256(modeled & available),
            "missing_sequences": 0,
            "missing_sequence_ids": [],
            "missing_sequence_ids_sha256": _identifier_set_sha256(set()),
            "extra_sequences": len(extras),
            "extra_sequence_ids": extras,
            "extra_sequence_ids_sha256": _identifier_set_sha256(extras),
            "required_raw_member_joins": len(modeled) * len(raw.members),
        },
        "supported": {
            "contexts": len(joined),
            "sequences": len({item.example.sequence_id for item in joined}),
            "exact_canonical_target_mapping": True,
            "all_contexts_supported": len(joined) == len(examples),
            "calibration_weighting": config.calibration_weighting,
        },
    }


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _fit_calibration(
    rows: Sequence[JoinedContext],
    *,
    heldout_fold: int,
    member_id: str,
    config: Config,
) -> FittedCalibration:
    _require(heldout_fold in range(config.folds), "held-out fold is invalid")
    checkpoint_by_member = {item.member_id: item.checkpoint_sha256 for item in config.members}
    _require(member_id in checkpoint_by_member, "calibration member is not frozen")
    training = [item for item in rows if item.example.fold != heldout_fold]
    _require(bool(training), "held-out fold leaves no calibration contexts")
    labels = np.asarray([item.example.label for item in training], dtype=np.float64)
    signal = np.asarray([item.signals[member_id] for item in training], dtype=np.float64)
    _require(
        set(labels.tolist()) == {0.0, 1.0} and np.all(np.isfinite(signal)),
        "calibration complement is non-finite or lacks a class",
    )
    signal_mean = float(np.mean(signal))
    signal_scale = float(np.std(signal))
    if signal_scale < 1e-12:
        signal_scale = 1.0
    standardized = (signal - signal_mean) / signal_scale
    prior = float(
        (np.sum(labels) + 0.5 * config.calibration_prior_strength)
        / (len(labels) + config.calibration_prior_strength)
    )
    _require(0 < prior < 1, "calibration prior is degenerate")
    coefficient = np.asarray([math.log(prior / (1.0 - prior)), 0.0], dtype=np.float64)
    design = np.column_stack((np.ones(len(labels)), standardized))
    penalty = np.asarray([0.0, config.calibration_l2], dtype=np.float64)
    iterations = 0
    converged = False
    for iteration in range(1, config.calibration_max_iterations + 1):
        probabilities = np.asarray(
            [_stable_sigmoid(float(value)) for value in design @ coefficient]
        )
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (probabilities - labels) / len(labels) + penalty * coefficient
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        _require(np.all(np.isfinite(step)), "calibration Newton step is non-finite")
        coefficient -= step
        iterations = iteration
        _require(np.all(np.isfinite(coefficient)), "calibration coefficient is non-finite")
        if float(np.max(np.abs(step))) <= config.calibration_tolerance:
            converged = True
            break
    _require(converged, f"fold {heldout_fold} member {member_id!r} did not converge")
    examples = [item.example for item in training]
    positives = int(np.sum(labels))
    return FittedCalibration(
        heldout_fold=heldout_fold,
        excluded_folds=(heldout_fold,),
        training_folds=tuple(sorted({item.fold for item in examples})),
        member_id=member_id,
        member_checkpoint_sha256=checkpoint_by_member[member_id],
        weighting_policy=config.calibration_weighting,
        training_examples=len(training),
        training_positives=positives,
        training_negatives=len(training) - positives,
        training_sequences=len({item.sequence_id for item in examples}),
        training_union_components=len({item.union_component_id for item in examples}),
        training_example_ids_sha256=_identifier_set_sha256(item.example_id for item in examples),
        training_sequence_ids_sha256=_identifier_set_sha256(item.sequence_id for item in examples),
        training_union_component_ids_sha256=_identifier_set_sha256(
            item.union_component_id for item in examples
        ),
        training_assignment_sha256=_assignment_sha256(examples),
        iterations=iterations,
        converged=True,
        signal_mean=signal_mean,
        signal_scale=signal_scale,
        intercept=float(coefficient[0]),
        slope=float(coefficient[1]),
        probability_clip_epsilon=config.probability_clip_epsilon,
    )


def _prediction_from_example(
    example: Example,
    *,
    model: str,
    member_id: str,
    endpoints: str,
    signal: float,
    probability: float,
    standard_deviation: float | None,
) -> Prediction:
    return Prediction(
        model=model,
        member_id=member_id,
        example_id=example.example_id,
        assay_context_id=example.assay_context_id,
        sequence_id=example.sequence_id,
        sequence=example.sequence,
        canonical_target=example.canonical_target,
        gram=example.gram,
        label=example.label,
        source_observations=example.source_observations,
        fold=example.fold,
        homology_component_id=example.homology_component_id,
        union_component_id=example.union_component_id,
        max_train_identity=example.max_train_identity,
        apex_endpoints=endpoints,
        activity_signal=signal,
        probability=probability,
        member_probability_std=standard_deviation,
    )


def _reconstruct_predictions(
    joined: Sequence[JoinedContext], *, config: Config
) -> tuple[tuple[Prediction, ...], tuple[FittedCalibration, ...]]:
    members = tuple(item.member_id for item in config.members)
    calibrations: list[FittedCalibration] = []
    lookup: dict[tuple[int, str], FittedCalibration] = {}
    for fold in range(config.folds):
        for member in members:
            fitted = _fit_calibration(
                joined,
                heldout_fold=fold,
                member_id=member,
                config=config,
            )
            calibrations.append(fitted)
            lookup[(fold, member)] = fitted
    predictions: list[Prediction] = []
    for item in sorted(joined, key=lambda row: row.example.example_id):
        endpoints = ";".join(item.target.endpoints)
        member_probabilities: list[float] = []
        for member in members:
            probability = lookup[(item.example.fold, member)].probability(item.signals[member])
            member_probabilities.append(probability)
            predictions.append(
                _prediction_from_example(
                    item.example,
                    model=f"apex_member::{member}",
                    member_id=member,
                    endpoints=endpoints,
                    signal=item.signals[member],
                    probability=probability,
                    standard_deviation=None,
                )
            )
        predictions.append(
            _prediction_from_example(
                item.example,
                model=_MEAN_MODEL,
                member_id="",
                endpoints=endpoints,
                signal=math.fsum(item.signals.values()) / len(item.signals),
                probability=math.fsum(member_probabilities) / len(member_probabilities),
                standard_deviation=float(np.std(np.asarray(member_probabilities))),
            )
        )
    predictions.sort(key=lambda row: (row.model, row.example_id))
    _require(
        len(calibrations) == config.folds * config.expected_member_count == 40
        and len(predictions) == config.expected_output_rows,
        "calibration or prediction census differs",
    )
    return tuple(predictions), tuple(calibrations)


def _roc_auc(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    differences = positives[:, None] - negatives[None, :]
    return float((np.sum(differences > 0) + 0.5 * np.sum(differences == 0)) / differences.size)


def _average_precision(labels: IntArray, probabilities: FloatArray) -> float | None:
    positive_count = int(np.sum(labels))
    if positive_count == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    result = 0.0
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_probability[end] == sorted_probability[start]:
            end += 1
        new_positives = int(np.sum(sorted_labels[start:end]))
        true_positive += new_positives
        false_positive += end - start - new_positives
        result += (new_positives / positive_count) * (
            true_positive / (true_positive + false_positive)
        )
        start = end
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
    _require(calibration_bins >= 2, "calibration bins are invalid")
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


def _metric(rows: Sequence[Prediction], *, bins: int) -> dict[str, int | float | None]:
    _require(bool(rows), "metric subgroup is empty")
    return _binary_metrics(
        [item.label for item in rows],
        [item.probability for item in rows],
        calibration_bins=bins,
    )


def _shared_draws(
    component_ids: Sequence[str], *, replicates: int, seed: int
) -> tuple[tuple[tuple[str, ...], ...], str]:
    components = tuple(sorted(component_ids))
    _require(
        bool(components) and len(components) == len(set(components)),
        "bootstrap components are empty or duplicate",
    )
    generator = np.random.default_rng(seed)
    digest = hashlib.sha256()
    draws: list[tuple[str, ...]] = []
    for replicate in range(replicates):
        selected = tuple(
            str(value) for value in generator.choice(components, size=len(components), replace=True)
        )
        draws.append(selected)
        for position, component_id in enumerate(selected):
            digest.update(f"{replicate}\t{position}\t{component_id}\n".encode("ascii"))
    return tuple(draws), digest.hexdigest()


def _bootstrap(
    rows: Sequence[Prediction],
    *,
    draws: Sequence[Sequence[str]],
    bins: int,
) -> dict[str, dict[str, float | int]]:
    by_component: dict[str, list[Prediction]] = defaultdict(list)
    for item in rows:
        by_component[item.union_component_id].append(item)
    known = set(by_component)
    names = ("roc_auc", "average_precision", "brier", "log_loss")
    samples: dict[str, list[float]] = {name: [] for name in names}
    for draw in draws:
        _require(not (set(draw) - known), "bootstrap draw references unknown component")
        selected = [row for component_id in draw for row in by_component[component_id]]
        values = _metric(selected, bins=bins)
        for name in names:
            value = values[name]
            _require(value is not None, f"bootstrap produced undefined {name}")
            samples[name].append(float(cast(float, value)))
    point = _metric(rows, bins=bins)
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        _require(len(samples[name]) == len(draws), "bootstrap lost a replicate")
        values = np.asarray(samples[name], dtype=np.float64)
        point_value = point[name]
        _require(point_value is not None, f"overall {name} is undefined")
        result[name] = {
            "point": float(cast(float, point_value)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "successful_replicates": len(samples[name]),
        }
    return result


def _metrics_document(predictions: Sequence[Prediction], *, config: Config) -> dict[str, object]:
    by_model: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_model[prediction.model].append(prediction)
    member_models = tuple(f"apex_member::{item.member_id}" for item in config.members)
    _require(
        set(by_model) == {*member_models, _MEAN_MODEL},
        "prediction model inventory differs",
    )
    expected_ids = {item.example_id for item in by_model[_MEAN_MODEL]}
    _require(len(expected_ids) == config.expected_examples, "mean model context support differs")
    for model, rows in by_model.items():
        _require(
            len(rows) == config.expected_examples
            and {item.example_id for item in rows} == expected_ids,
            f"model {model!r} context support differs",
        )
    components = sorted({item.union_component_id for item in by_model[_MEAN_MODEL]})
    _require(
        len(components) == config.expected_modeled_union_components,
        "bootstrap union-component census differs",
    )
    draws, draw_sha256 = _shared_draws(
        components,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    model_metrics: dict[str, object] = {}
    for model in sorted(by_model):
        rows = tuple(sorted(by_model[model], key=lambda item: item.example_id))
        identity_metrics: dict[str, object] = {}
        for left, right in _SIMILARITY_STRATA:
            selected = [item for item in rows if left <= item.max_train_identity < right]
            if selected:
                identity_metrics[f"[{left:.2f},{right:.2f})"] = _metric(
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
            "by_max_train_identity": identity_metrics,
            "union_component_bootstrap_95ci": _bootstrap(
                rows,
                draws=draws,
                bins=config.calibration_bins,
            ),
        }
    aligned = {
        model: {item.example_id: item.probability for item in by_model[model]}
        for model in member_models
    }
    ordered_ids = sorted(expected_ids)
    matrix = np.asarray(
        [[aligned[model][example_id] for example_id in ordered_ids] for model in member_models],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.corrcoef(matrix)
    _require(
        correlations.shape == (len(member_models), len(member_models))
        and np.all(np.isfinite(correlations)),
        "member probability correlations are undefined",
    )
    off_diagonal = correlations[np.triu_indices(len(member_models), 1)]
    return {
        "schema_version": 1,
        "artifact": "apex_union_oof_metrics_v1",
        "models": model_metrics,
        "bootstrap": {
            "unit": _BOOTSTRAP_UNIT,
            "weighting": config.bootstrap_weighting,
            "component_ids": len(components),
            "component_ids_sha256": _identifier_set_sha256(components),
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


def _predictions_bytes(predictions: Sequence[Prediction]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(_OUTPUT_SCHEMA), lineterminator="\n")
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
    return stream.getvalue().encode("utf-8")


def _calibrators_document(
    calibrations: Sequence[FittedCalibration], *, config: Config
) -> dict[str, object]:
    return {
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
            "calibrators": len(calibrations),
        },
        "calibrators": [asdict(item) for item in calibrations],
    }


def _expected_semantic_payloads(
    *,
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    calibrations: Sequence[FittedCalibration],
    coverage: Mapping[str, object],
    metrics: Mapping[str, object],
    config: Config,
    git_commit: str,
    code_manifest: Snapshot,
    frozen_manifest: Snapshot,
    code_attestation: Mapping[str, object],
    base_snapshots: Mapping[str, Snapshot],
    base_independent: Snapshot,
    base_manifest: Mapping[str, object],
    split_receipt: Mapping[str, object],
    raw_snapshots: Mapping[str, Snapshot],
) -> dict[str, bytes]:
    prediction_payload = _predictions_bytes(predictions)
    calibrator_payload = _pretty_json_bytes(_calibrators_document(calibrations, config=config))
    coverage_payload = _pretty_json_bytes(coverage)
    metrics_payload = _pretty_json_bytes(metrics)
    partial = {
        "apex_union_oof_predictions.csv": prediction_payload,
        "calibrators.json": calibrator_payload,
        "metrics.json": metrics_payload,
        "sequence_coverage_receipt.json": coverage_payload,
    }
    artifact_roles = {
        "predictions": (
            "apex_union_oof_predictions.csv",
            "nine-model context-level outer OOF predictions",
        ),
        "calibrators": ("calibrators.json", "forty outer-fold member calibrators"),
        "coverage": (
            "sequence_coverage_receipt.json",
            "strict sequence/context reconciliation receipt",
        ),
        "metrics": (
            "metrics.json",
            "context metrics and shared union-component bootstrap",
        ),
    }
    artifacts = {
        key: {
            "filename": filename,
            "role": role,
            "sha256": _sha256(partial[filename]),
        }
        for key, (filename, role) in artifact_roles.items()
    }
    manifest = {
        "schema_version": 1,
        "artifact": _ARTIFACT,
        "status": _OUTPUT_STATUS,
        "git_commit": git_commit,
        "config_sha256": config.sha256,
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
            "outer_calibrators": len(calibrations),
            "probability_clip_epsilon": config.probability_clip_epsilon,
            "label_feature_used_only_on_outer_training_complement": True,
            "forbidden_features": list(_FORBIDDEN_FEATURES),
            "model_features": list(_MODEL_FEATURES),
        },
        "bootstrap": metrics["bootstrap"],
        "census": {
            "contexts": len(examples),
            "source_observations": sum(item.source_observations for item in examples),
            "positives": sum(item.label for item in examples),
            "negatives": sum(1 - item.label for item in examples),
            "modeled_sequences": len({item.sequence_id for item in examples}),
            "modeled_homology_components": len({item.homology_component_id for item in examples}),
            "modeled_union_components": len({item.union_component_id for item in examples}),
            "prediction_rows": len(predictions),
        },
        "models": [
            *(f"apex_member::{item.member_id}" for item in config.members),
            _MEAN_MODEL,
        ],
        "members": [asdict(item) for item in config.members],
        "accepted_gate1": {
            "artifact": _BASE_ARTIFACT,
            "publication_top_sha256": base_snapshots["SHA256SUMS"].sha256,
            "semantic_top_sha256": base_snapshots["gate1/SHA256SUMS"].sha256,
            "independent_receipt_sha256": base_independent.sha256,
            "accepted_split": base_manifest.get("accepted_split"),
            "split_consumption_receipt_sha256": base_snapshots["gate1/split_receipt.json"].sha256,
            "all_split_census": split_receipt.get("census"),
        },
        "input_sha256": {
            "code_manifest": code_manifest.sha256,
            "frozen_input_manifest": frozen_manifest.sha256,
            "config": config.sha256,
            "base_publication_top": base_snapshots["SHA256SUMS"].sha256,
            "base_semantic_top": base_snapshots["gate1/SHA256SUMS"].sha256,
            "base_examples": base_snapshots["gate1/examples.jsonl"].sha256,
            "base_folds": base_snapshots["gate1/folds.json"].sha256,
            "base_oof_metadata": base_snapshots["gate1/oof_predictions.csv"].sha256,
            "base_manifest": base_snapshots["gate1/manifest.json"].sha256,
            "base_split_receipt": base_snapshots["gate1/split_receipt.json"].sha256,
            "base_independent_receipt": base_independent.sha256,
            "apex_predictions": raw_snapshots["apex_member_predictions.csv"].sha256,
            "apex_manifest": raw_snapshots["apex_member_run_manifest.json"].sha256,
            "apex_reconciliation": raw_snapshots["apex_member_reconciliation.csv"].sha256,
        },
        "code_attestation": code_attestation,
        "artifacts": artifacts,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    return {**partial, "manifest.json": _pretty_json_bytes(manifest)}


def _scan_forbidden(payloads: Iterable[bytes], prefixes: Sequence[bytes]) -> None:
    absolute = re.compile(rb"/(?:lustre|tmp)/|/home/[^/]+/")
    for payload in payloads:
        _require(
            not any(prefix in payload for prefix in prefixes),
            "publication exposes a forbidden execution prefix",
        )
        _require(absolute.search(payload) is None, "publication exposes an absolute path")


def _require_exact_payload(actual: bytes, expected: bytes, *, label: str) -> None:
    _require(actual == expected, f"{label} differs from independent reconstruction")


def verify_apex_union_oof_twins(
    *,
    producer_twin_root: str | Path,
    base_twin_root: str | Path,
    base_independent_receipt: str | Path,
    apex_root: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, object]:
    """Return a path-free receipt after fully reconstructing both producer twins."""

    producer_twins = _resolved_directory(producer_twin_root, label="producer twin root")
    base_twins = _resolved_directory(base_twin_root, label="accepted Gate-1 twin root")
    raw_apex = _resolved_directory(apex_root, label="raw APEX root")
    repository = _resolved_directory(repo_root, label="repository root")
    supplied_config = Path(config_path).absolute()
    _require_no_symlink(supplied_config, label="APEX union config", ancestors=True)
    config_file = supplied_config.resolve(strict=True)
    expected_config = (repository / _LOGICAL_CONFIG_PATH).resolve(strict=True)
    _require(config_file == expected_config, "config is not the repository logical config")
    config = _load_config(config_file)
    _verify_repository(repository, expected_git_commit)
    verifier_sha256 = _verify_execution_source(repository, expected_git_commit)

    independent = _read_snapshot(
        Path(base_independent_receipt), label="accepted Gate-1 independent receipt"
    )
    base_run, examples, base_manifest, split_receipt, base_handshake = _verify_base_chain(
        base_twins,
        independent,
        config=config,
    )
    _, base_snapshots = _verify_manifest_tree(
        base_run, expected=_BASE_TREE, label="accepted Gate-1 publication"
    )
    _require(
        _tree_inventory(raw_apex) == _RAW_APEX_TREE,
        "raw APEX root inventory differs from the frozen three-file contract",
    )
    raw_snapshots = {
        name: _read_snapshot(raw_apex / name, label=f"raw APEX {name}")
        for name in sorted(_RAW_APEX_TREE)
    }
    _require(
        raw_snapshots["apex_member_predictions.csv"].sha256
        == config.expected_apex_predictions_sha256
        and raw_snapshots["apex_member_run_manifest.json"].sha256
        == config.expected_apex_manifest_sha256
        and raw_snapshots["apex_member_reconciliation.csv"].sha256
        == config.expected_apex_reconciliation_sha256,
        "raw APEX content address differs from config",
    )
    raw_manifest = _verify_raw_apex_manifest(
        raw_snapshots["apex_member_run_manifest.json"],
        predictions=raw_snapshots["apex_member_predictions.csv"],
        reconciliation=raw_snapshots["apex_member_reconciliation.csv"],
        config=config,
    )
    raw_grid = _read_raw_apex_grid(
        raw_snapshots["apex_member_predictions.csv"],
        manifest=raw_manifest,
        config=config,
    )
    _verify_raw_apex_reconciliation(
        raw_snapshots["apex_member_reconciliation.csv"],
        manifest=raw_manifest,
        raw=raw_grid,
        config=config,
    )

    immediate = {entry.name: entry for entry in producer_twins.iterdir()}
    _require(
        set(immediate) == {"0", "1", "node-receipts"}
        and all(entry.is_dir() and not entry.is_symlink() for entry in immediate.values()),
        "producer twin-root inventory differs",
    )
    runs = (producer_twins / "0", producer_twins / "1")
    for index, run in enumerate(runs):
        _verify_immutable_tree(run, label=f"producer twin {index}")
    _verify_tree_bytes(runs[0], runs[1], expected=_PRODUCER_TREE)
    production_handshake = _verify_handshake(producer_twins)
    publication_records = [
        _verify_producer_publication(
            run,
            base_run=base_run,
            independent_receipt=independent,
            apex_root=raw_apex,
            repository=repository,
            expected_commit=expected_git_commit,
            config=config,
        )
        for run in runs
    ]
    first_snapshots, actual_semantic, code_attestation = publication_records[0]
    second_snapshots, _, second_code_attestation = publication_records[1]
    _require(code_attestation == second_code_attestation, "producer twin code attestations differ")
    _require(
        first_snapshots["SHA256SUMS"].sha256 == second_snapshots["SHA256SUMS"].sha256,
        "producer publication top hashes differ",
    )
    candidate_manifest = _json_object(
        actual_semantic["manifest.json"].payload,
        label="candidate APEX union manifest",
        canonical_pretty=True,
    )
    runtime = candidate_manifest.get("runtime")
    _require(
        isinstance(runtime, dict)
        and runtime.get("python") == platform.python_version()
        and runtime.get("numpy") == np.__version__,
        "verifier runtime differs from the exact-numeric producer runtime",
    )
    _require(
        candidate_manifest.get("production_eligible") is False
        and candidate_manifest.get("status") == _OUTPUT_STATUS
        and candidate_manifest.get("upstream_training_independence")
        == {
            "status": "unknown",
            "external_family_weight_policy": "zero_until_training_membership_is_resolved",
        },
        "candidate APEX semantics were promoted beyond available evidence",
    )

    joined = _join_contexts(examples, raw=raw_grid, config=config)
    predictions, calibrations = _reconstruct_predictions(joined, config=config)
    coverage = _coverage_document(
        examples,
        raw=raw_grid,
        joined=joined,
        config=config,
    )
    metrics = _metrics_document(predictions, config=config)
    expected_semantic = _expected_semantic_payloads(
        examples=examples,
        predictions=predictions,
        calibrations=calibrations,
        coverage=coverage,
        metrics=metrics,
        config=config,
        git_commit=expected_git_commit,
        code_manifest=first_snapshots["CODE_SHA256SUMS"],
        frozen_manifest=first_snapshots["FROZEN_INPUT_SHA256SUMS"],
        code_attestation=code_attestation,
        base_snapshots=base_snapshots,
        base_independent=independent,
        base_manifest=base_manifest,
        split_receipt=split_receipt,
        raw_snapshots=raw_snapshots,
    )
    for filename, expected in expected_semantic.items():
        _require_exact_payload(
            actual_semantic[filename].payload,
            expected,
            label=f"producer {filename}",
        )
    semantic_hashes = {
        filename: _sha256(payload) for filename, payload in sorted(expected_semantic.items())
    }
    expected_semantic_top = _sha_manifest_bytes(semantic_hashes)
    _require_exact_payload(
        first_snapshots["apex/SHA256SUMS"].payload,
        expected_semantic_top,
        label="producer semantic top manifest",
    )

    forbidden = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(isinstance(prefix, str) and bool(prefix), "forbidden prefix is empty")
        forbidden.append(prefix.encode("utf-8"))
    _scan_forbidden(
        [first_snapshots[name].payload for name in sorted(_PRODUCER_TREE)],
        forbidden,
    )
    for snapshot in (
        independent,
        *base_snapshots.values(),
        *raw_snapshots.values(),
        *first_snapshots.values(),
        *second_snapshots.values(),
    ):
        _assert_unchanged(snapshot, label=snapshot.path.name)
    _verify_tree_bytes(runs[0], runs[1], expected=_PRODUCER_TREE)
    _verify_repository(repository, expected_git_commit)
    _require(
        _verify_execution_source(repository, expected_git_commit) == verifier_sha256,
        "verifier source changed during reconstruction",
    )

    models = cast(dict[str, object], metrics["models"])
    overall_metrics = {
        model: cast(dict[str, object], model_metrics)["overall"]
        for model, model_metrics in models.items()
    }
    return {
        "schema_version": 1,
        "artifact": f"{_ARTIFACT}_independent_verification",
        "status": "passed",
        "acceptance_scope": "development_candidate_only",
        "production_eligible": False,
        "upstream_training_independence": {
            "status": "unknown",
            "external_family_weight_policy": "zero_until_training_membership_is_resolved",
        },
        "checks": {name: True for name in sorted(_VERIFICATION_CHECKS)},
        "git_commit": expected_git_commit,
        "verifier_module_sha256": verifier_sha256,
        "config_sha256": config.sha256,
        "publication_top_manifest_sha256": first_snapshots["SHA256SUMS"].sha256,
        "semantic_top_manifest_sha256": first_snapshots["apex/SHA256SUMS"].sha256,
        "code_manifest_sha256": first_snapshots["CODE_SHA256SUMS"].sha256,
        "frozen_input_manifest_sha256": first_snapshots["FROZEN_INPUT_SHA256SUMS"].sha256,
        "input_sha256": {
            "base_publication_top": base_snapshots["SHA256SUMS"].sha256,
            "base_semantic_top": base_snapshots["gate1/SHA256SUMS"].sha256,
            "base_independent_receipt": independent.sha256,
            "base_examples": base_snapshots["gate1/examples.jsonl"].sha256,
            "base_folds": base_snapshots["gate1/folds.json"].sha256,
            "base_oof_metadata": base_snapshots["gate1/oof_predictions.csv"].sha256,
            "raw_apex_predictions": raw_snapshots["apex_member_predictions.csv"].sha256,
            "raw_apex_manifest": raw_snapshots["apex_member_run_manifest.json"].sha256,
            "raw_apex_reconciliation": raw_snapshots["apex_member_reconciliation.csv"].sha256,
        },
        "artifact_sha256": semantic_hashes,
        "census": {
            "contexts": len(examples),
            "source_observations": sum(item.source_observations for item in examples),
            "positives": sum(item.label for item in examples),
            "negatives": sum(1 - item.label for item in examples),
            "modeled_sequences": len({item.sequence_id for item in examples}),
            "modeled_homology_components": len({item.homology_component_id for item in examples}),
            "modeled_union_components": len({item.union_component_id for item in examples}),
            "raw_apex_sequences": len(raw_grid.sequences),
            "raw_apex_rows": raw_grid.rows,
            "prediction_rows": len(predictions),
        },
        "calibration": {
            "outer_folds": config.folds,
            "members": config.expected_member_count,
            "calibrators": len(calibrations),
            "weighting": config.calibration_weighting,
            "probability_clip_epsilon": config.probability_clip_epsilon,
        },
        "overall_metrics": overall_metrics,
        "bootstrap": metrics["bootstrap"],
        "production_handshake": production_handshake,
        "accepted_gate1_handshake": base_handshake,
        "limitations": {
            "raw_apex_checkpoint_inference_reexecuted": False,
            "raw_apex_claim_scope": (
                "content_addressed outputs, frozen checkpoint identifiers, rectangular grid, "
                "and fidelity reconciliation only"
            ),
            "scheduler_signed_handshake": False,
        },
    }


def _write_receipt(path: Path, receipt: Mapping[str, object], *, protected: Sequence[Path]) -> None:
    requested = Path(path).absolute()
    _require(requested.name not in {"", ".", ".."}, "verification receipt has no filename")
    _require_no_symlink(requested.parent, label="receipt parent", ancestors=True)
    _require(not os.path.lexists(requested), f"refusing to overwrite receipt: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    _require(not os.path.lexists(target), f"refusing to overwrite receipt: {target}")
    for root in protected:
        _require(
            target != root and not target.is_relative_to(root),
            "verification receipt must be outside verified trees",
        )
    payload = _pretty_json_bytes(receipt)
    _scan_forbidden([payload], [b"/lustre/scratch/users/"])
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o444)
        os.link(staging, target)
        staging.unlink()
    finally:
        if staging.exists():
            staging.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-twin-root", type=Path, required=True)
    parser.add_argument("--base-twin-root", type=Path, required=True)
    parser.add_argument("--base-independent-receipt", type=Path, required=True)
    parser.add_argument("--apex-root", type=Path, required=True)
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
        _resolved_directory(args.apex_root, label="raw APEX root"),
        _resolved_directory(args.repo_root, label="repository root"),
    ]
    prospective = Path(args.output).absolute().resolve(strict=False)
    for root in protected:
        _require(
            prospective != root and not prospective.is_relative_to(root),
            "verification receipt must be outside verified trees",
        )
    receipt = verify_apex_union_oof_twins(
        producer_twin_root=args.producer_twin_root,
        base_twin_root=args.base_twin_root,
        base_independent_receipt=args.base_independent_receipt,
        apex_root=args.apex_root,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    _write_receipt(Path(args.output), receipt, protected=protected)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the Slurm CLI
    raise SystemExit(main())

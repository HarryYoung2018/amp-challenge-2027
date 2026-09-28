"""Independently reconstruct and verify homology-study union Gate-1 twins.

The verifier intentionally imports none of the Gate-1 producer, baseline-model,
descriptor, sequence, or similarity modules.  It reparses the frozen inputs,
reconstructs the context panel, implements the two baseline algorithms and all
reported metrics independently, and compares every semantic output byte.  The
accepted parser, endpoint-context, and split attestations are verified by their
content addresses; their already-audited upstream construction is not repeated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
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
_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_AMINO_ACID_SET = frozenset(_AMINO_ACIDS)
_MIN_LENGTH = 8
_MAX_LENGTH = 50

_BENCHMARK = "gate1_context_activity_homology_study_union_v1"
_BENCHMARK_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_FOLD_POLICY = "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
_HOMOLOGY_POLICY = "global_alignment_identity_single_link_v1"
_SPLIT_ARTIFACT = "homology_study_union_split"
_SPLIT_STATUS = "development_split_not_an_untouched_evaluation_panel"
_SPLIT_COMPONENT_POLICY = "full_sequence_homology_union_every_study_key_v1"
_LABEL_POLICY = (
    "group every ledger row by assay_context_id before eligibility filtering; retain only "
    "fully bacterial_mic16-eligible contexts with unanimous label, source Gram, and "
    "canonical target"
)
_MODEL_FEATURES = ("sequence", "canonical_target", "gram")
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
_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")

_EXPECTED_NORMALIZED_TOP_SHA256 = "c83e446f89bbf48a4c3c2fea9a397328147267b231cafad6f913badb7c6c3379"
_EXPECTED_CONTEXT_TOP_SHA256 = "3a3722e685470a37ec9e5fb31aa2652cae9742966a3be3d29133692737054ae8"
_EXPECTED_SPLIT_TOP_SHA256 = "420bfa5a23475b17090b8af0ff36b67171df21b968b5f3197a28672a2fe530eb"
_EXPECTED_SPLIT_RECEIPT_SHA256 = "7786970354ad0c7f260f86931ec2b9b0b38360fc85f94327aa8bcae91d9055a6"
_EXPECTED_SPLIT_GIT_COMMIT = "121f9b2a5c2a4859c7ec8102d5699ce6cffd0f87"

_LOGICAL_CONFIG_PATH = "configs/benchmarks/oracle_gate1_union_v1.toml"
_PRODUCER_MODULE_PATH = "src/amp_challenge/benchmarks/oracle_gate1_union.py"
_VERIFIER_MODULE_PATH = "src/amp_challenge/benchmarks/oracle_gate1_union_verify.py"
_FIXED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/oracle_gate1_union_v1_twins.sbatch",
        "cluster/validate_oracle_gate1_union_output.sh",
        _LOGICAL_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)

_PUBLICATION_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
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
_PUBLICATION_MANIFEST_ENTRIES = _PUBLICATION_FILES - {"SHA256SUMS"}
_GATE1_FILES = frozenset(
    {
        "SHA256SUMS",
        "context_audit.jsonl",
        "examples.jsonl",
        "folds.json",
        "manifest.json",
        "metrics.json",
        "oof_predictions.csv",
        "split_receipt.json",
    }
)
_GATE1_MANIFEST_ENTRIES = _GATE1_FILES - {"SHA256SUMS"}
_SPLIT_TOP_ENTRIES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "split/audit.json",
        "split/components.jsonl",
        "split/grouping_edges.jsonl",
        "split/manifest.json",
        "split/sequence_assignments.jsonl",
    }
)
_UPSTREAM_SPLIT_CHECKS = frozenset(
    {
        "artifacts_and_manifest_exact",
        "code_and_input_attestations_valid",
        "executing_source_bound_to_repository",
        "fold_assignment_recomputed",
        "frozen_inputs_config_pinned",
        "homology_and_study_union_recomputed",
        "input_twins_byte_identical",
        "json_and_jsonl_canonical_lf",
        "repository_commit_and_cleanliness_verified",
        "scratch_and_absolute_paths_absent",
        "strict_context_exclusions_recomputed",
        "top_manifests_valid",
        "twins_byte_identical",
    }
)
_VERIFICATION_CHECKS = frozenset(
    {
        "publication_top_manifests_valid",
        "twins_byte_identical",
        "production_overlap_handshake_valid",
        "input_twins_and_frozen_manifests_valid",
        "accepted_split_receipt_chain_valid",
        "repository_commit_and_cleanliness_verified",
        "executing_source_bound_to_repository",
        "canonical_serialization_verified",
        "context_partition_recomputed",
        "assignment_and_component_join_recomputed",
        "cross_fold_identity_recomputed",
        "descriptor_logistic_oof_recomputed",
        "homology_knn_oof_recomputed",
        "ensemble_mean_exact",
        "metrics_and_union_bootstrap_recomputed",
        "all_semantic_artifacts_exact",
        "scratch_and_absolute_paths_absent",
    }
)

_SEQUENCE_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_ASSIGNMENT_FIELDS = frozenset(
    {"schema_version", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)
_LEDGER_FIELDS = frozenset(
    {
        "apparent_taxon_mentions",
        "assay_context_id",
        "assay_row_sha256",
        "blood_organism_mapping_version",
        "blood_organism_resolution",
        "canonical_target",
        "composite_language_marker",
        "composite_reason_codes",
        "configured_target_occurrences",
        "context_id",
        "eligible_tasks",
        "endpoint",
        "exclusion_codes",
        "expected_gram",
        "exposure_concentration",
        "gram_resolution",
        "input_line",
        "mapping_status",
        "mapping_version",
        "measurement",
        "measurement_status",
        "mic16_label",
        "mic16_reason",
        "normalized_organism",
        "normalized_target",
        "observation_id",
        "provenance_id",
        "resistance_status",
        "resolved_blood_organism",
        "schema_version",
        "sequence_id",
        "source_assay",
        "source_conditions",
        "source_field",
        "source_gram",
        "source_organism",
        "source_record_id",
        "source_record_key",
        "source_row_number",
        "source_target",
        "source_text",
        "strain_identifier_resolution",
        "strain_identifier_review_ids",
        "strain_level_eligible",
        "study_keys",
        "study_review_codes",
        "study_status",
        "target_domain",
    }
)
_CLASS_METRICS = (
    "gram_negative_mic16_negative",
    "gram_negative_mic16_positive",
    "gram_positive_mic16_negative",
    "gram_positive_mic16_positive",
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
    """Raised when an independent Gate-1 verification invariant fails."""


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
class FrozenConfig:
    path: Path
    sha256: str
    activity_threshold_um: float
    identity_threshold: float
    folds: int
    seed: int
    bootstrap_replicates: int
    calibration_bins: int
    similarity_bin_edges: tuple[float, ...]
    sequences_sha256: str
    ledger_sha256: str
    assignments_sha256: str
    components_sha256: str
    split_manifest_sha256: str
    split_audit_sha256: str
    split_top_sha256: str
    split_receipt_sha256: str
    expected_sequences: int
    expected_ledger_rows: int
    expected_homology_components: int
    expected_union_components: int
    expected_examples: int
    expected_source_observations: int
    expected_modeled_sequences: int
    expected_positives: int
    expected_negatives: int
    examples_by_fold: tuple[int, ...]
    positives_by_fold: tuple[int, ...]
    negatives_by_fold: tuple[int, ...]
    logistic: LogisticSettings
    knn: KnnSettings


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Assignment:
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    fold: int


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    canonical_target: str
    gram: GramClass
    label: int
    observation_ids: tuple[str, ...]
    fold: int
    homology_component_id: str
    union_component_id: str


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
class ReconstructedPanel:
    examples: tuple[Example, ...]
    audit_rows: tuple[dict[str, object], ...]
    census: dict[str, object]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _require_no_symlink(path: Path, *, label: str, ancestors: bool = False) -> None:
    candidate = path.absolute()
    targets = [candidate]
    if ancestors:
        targets.extend(candidate.parents)
    for target in targets:
        _require(not target.is_symlink(), f"{label} must not traverse a symbolic link: {target}")


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _require_no_symlink(requested, label=label, ancestors=True)
    resolved = requested.resolve(strict=True)
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is not a real directory")
    return resolved


def _read_snapshot(path: Path, *, label: str) -> Snapshot:
    _require_no_symlink(path, label=label, ancestors=True)
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return Snapshot(resolved, payload, _sha256(payload), _fingerprint(before))


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    try:
        current = snapshot.path.stat()
    except FileNotFoundError as error:
        raise VerificationError(f"{label} disappeared during verification") from error
    _require(
        stat.S_ISREG(current.st_mode) and _fingerprint(current) == snapshot.fingerprint,
        f"{label} changed during verification",
    )


def _reject_constant(value: str) -> object:
    raise VerificationError(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _loads_json(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, VerificationError) as error:
        raise VerificationError(f"{label} is not valid strict UTF-8 JSON") from error


def _json_object(
    payload: bytes, *, label: str, canonical_pretty: bool = False
) -> dict[str, object]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} must have exactly one final LF",
    )
    value = _loads_json(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain one object")
    result = cast(dict[str, object], value)
    if canonical_pretty:
        _require(payload == _pretty_json_bytes(result), f"{label} is not canonical pretty JSON")
    return result


def _jsonl_rows(
    payload: bytes,
    *,
    label: str,
    canonical: bool,
) -> tuple[dict[str, object], ...]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} LF framing")
    rows: list[dict[str, object]] = []
    for line_number, raw in enumerate(payload[:-1].split(b"\n"), start=1):
        _require(raw != b"", f"{label} line {line_number} is blank")
        value = _loads_json(raw, label=f"{label} line {line_number}")
        _require(isinstance(value, dict), f"{label} line {line_number} is not an object")
        row = cast(dict[str, object], value)
        if canonical:
            _require(raw + b"\n" == _compact_json_bytes(row) + b"\n", f"{label} noncanonical")
        rows.append(row)
    return tuple(rows)


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_compact_json_bytes(row) + b"\n" for row in rows)


def _safe_manifest_name(value: str, *, label: str) -> str:
    pure = PurePosixPath(value)
    _require(
        value != ""
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "." not in pure.parts
        and "\\" not in value,
        f"{label} contains unsafe path {value!r}",
    )
    return value


def _parse_sha_manifest(
    payload: bytes,
    *,
    label: str,
    require_sorted: bool = True,
) -> dict[str, str]:
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
        _require(mode == " ", f"{label} line {number} is not text-mode syntax")
        name = _safe_manifest_name(raw_name, label=label)
        _require(name not in result, f"{label} repeats {name!r}")
        if require_sorted and previous is not None:
            _require(name > previous, f"{label} paths are not strictly sorted")
        result[name] = digest
        previous = name
    return result


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("utf-8")


def _tree_inventory(root: Path) -> frozenset[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"tree contains symbolic link: {path}")
        if path.is_dir():
            continue
        _require(path.is_file(), f"tree contains a non-regular entry: {path}")
        result.add(path.relative_to(root).as_posix())
    return frozenset(result)


def _verify_manifest_tree(
    root: Path,
    *,
    expected_top_sha256: str | None,
    expected_inventory: frozenset[str] | None,
    require_sorted: bool = True,
    label: str,
) -> tuple[str, dict[str, str]]:
    top = _read_snapshot(root / "SHA256SUMS", label=f"{label} top manifest")
    if expected_top_sha256 is not None:
        _require(top.sha256 == expected_top_sha256, f"{label} top-manifest hash mismatch")
    entries = _parse_sha_manifest(
        top.payload, label=f"{label} top manifest", require_sorted=require_sorted
    )
    inventory = _tree_inventory(root)
    required = frozenset(entries) | {"SHA256SUMS"}
    if expected_inventory is not None:
        _require(inventory == expected_inventory, f"{label} inventory differs from contract")
        _require(
            set(entries) == set(expected_inventory) - {"SHA256SUMS"},
            f"{label} manifest inventory mismatch",
        )
    else:
        _require(inventory == required, f"{label} manifest does not cover its tree exactly")
    for name, expected in entries.items():
        snapshot = _read_snapshot(root / name, label=f"{label} {name}")
        _require(snapshot.sha256 == expected, f"{label} checksum mismatch for {name}")
    return top.sha256, entries


def _verify_tree_bytes(left: Path, right: Path, *, expected: frozenset[str] | None = None) -> None:
    left_inventory = _tree_inventory(left)
    right_inventory = _tree_inventory(right)
    _require(left_inventory == right_inventory, "twin tree inventories differ")
    if expected is not None:
        _require(left_inventory == expected, "twin tree inventory differs from contract")
    for name in sorted(left_inventory):
        left_payload = _read_snapshot(left / name, label=f"left twin {name}").payload
        right_payload = _read_snapshot(right / name, label=f"right twin {name}").payload
        _require(left_payload == right_payload, f"twin bytes differ for {name}")


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


def _number(value: object, *, label: str, positive: bool = False) -> float:
    _require(type(value) in {int, float}, f"{label} must be numeric")
    result = float(cast(int | float, value))
    _require(math.isfinite(result) and (not positive or result > 0), f"bad {label}")
    return result


def _string_array(
    value: object,
    *,
    label: str,
    allow_empty: bool = True,
    sorted_unique: bool = True,
) -> tuple[str, ...]:
    _require(isinstance(value, list) and (allow_empty or bool(value)), f"bad {label}")
    items = cast(list[object], value)
    _require(
        all(isinstance(item, str) and item and item == item.strip() for item in items),
        f"bad {label}",
    )
    result = tuple(cast(list[str], items))
    _require(len(set(result)) == len(result), f"duplicate {label}")
    if sorted_unique:
        _require(result == tuple(sorted(result)), f"unsorted {label}")
    return result


def _load_config(path: Path) -> FrozenConfig:
    snapshot = _read_snapshot(path, label="Gate-1 union config")
    _require(
        snapshot.payload.endswith(b"\n")
        and not snapshot.payload.endswith(b"\n\n")
        and b"\r" not in snapshot.payload,
        "Gate-1 union config must have exactly one final LF",
    )
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("Gate-1 union config is invalid") from error
    expected = {
        "schema_version",
        "activity_threshold_um",
        "homology_identity_threshold",
        "folds",
        "seed",
        "bootstrap_replicates",
        "calibration_bins",
        "similarity_bin_edges",
        "sequences_sha256",
        "endpoint_context_ledger_sha256",
        "split_assignments_sha256",
        "split_components_sha256",
        "split_manifest_sha256",
        "split_audit_sha256",
        "split_top_manifest_sha256",
        "split_independent_receipt_sha256",
        "expected_sequences",
        "expected_ledger_rows",
        "expected_split_homology_components",
        "expected_split_union_components",
        "expected_examples",
        "expected_source_observations",
        "expected_modeled_sequences",
        "expected_positive_examples",
        "expected_negative_examples",
        "expected_examples_by_fold",
        "expected_positive_examples_by_fold",
        "expected_negative_examples_by_fold",
        "descriptor_logistic",
        "homology_knn",
    }
    _exact_fields(raw, expected, label="Gate-1 union config")
    _require(raw["schema_version"] == 1 and type(raw["schema_version"]) is int, "bad config schema")
    folds = _positive_int(raw["folds"], label="folds")
    _require(folds >= 2, "at least two folds are required")
    seed = raw["seed"]
    _require(type(seed) is int, "seed must be an integer")
    threshold = _number(
        raw["homology_identity_threshold"], label="identity threshold", positive=True
    )
    _require(threshold <= 1.0, "identity threshold exceeds one")
    activity = _number(raw["activity_threshold_um"], label="activity threshold", positive=True)
    _require(activity == 16.0, "v1 activity threshold must be 16 uM")
    edges_raw = raw["similarity_bin_edges"]
    _require(isinstance(edges_raw, list), "similarity bin edges must be an array")
    edges = tuple(
        _number(item, label="similarity bin edge") for item in cast(list[object], edges_raw)
    )
    _require(
        len(edges) >= 2
        and edges[0] == 0.0
        and edges[-1] >= threshold
        and all(0.0 <= item <= 1.0 for item in edges)
        and all(left < right for left, right in itertools.pairwise(edges)),
        "invalid similarity bin edges",
    )

    def fold_array(value: object, *, label: str) -> tuple[int, ...]:
        _require(isinstance(value, list), f"{label} must be an array")
        result = tuple(_positive_int(item, label=label) for item in cast(list[object], value))
        _require(len(result) == folds, f"{label} length differs from folds")
        return result

    examples_by_fold = fold_array(raw["expected_examples_by_fold"], label="examples by fold")
    positives_by_fold = fold_array(
        raw["expected_positive_examples_by_fold"], label="positives by fold"
    )
    negatives_by_fold = fold_array(
        raw["expected_negative_examples_by_fold"], label="negatives by fold"
    )
    expected_examples = _positive_int(raw["expected_examples"], label="expected examples")
    expected_positives = _positive_int(
        raw["expected_positive_examples"], label="expected positives"
    )
    expected_negatives = _positive_int(
        raw["expected_negative_examples"], label="expected negatives"
    )
    _require(sum(examples_by_fold) == expected_examples, "example fold counts have wrong sum")
    _require(
        expected_positives + expected_negatives == expected_examples, "label counts have wrong sum"
    )
    _require(sum(positives_by_fold) == expected_positives, "positive fold counts have wrong sum")
    _require(sum(negatives_by_fold) == expected_negatives, "negative fold counts have wrong sum")
    _require(
        all(
            p + n == total
            for p, n, total in zip(
                positives_by_fold, negatives_by_fold, examples_by_fold, strict=True
            )
        ),
        "fold label counts do not sum to totals",
    )
    logistic_raw = raw["descriptor_logistic"]
    knn_raw = raw["homology_knn"]
    _require(isinstance(logistic_raw, dict) and isinstance(knn_raw, dict), "model tables missing")
    _exact_fields(
        cast(dict[str, object], logistic_raw),
        {"l2", "max_iterations", "tolerance", "prior_strength"},
        label="descriptor logistic config",
    )
    _exact_fields(
        cast(dict[str, object], knn_raw),
        {"neighbors", "similarity_power", "prior_strength", "minimum_weight"},
        label="homology kNN config",
    )
    logistic = LogisticSettings(
        l2=_number(logistic_raw["l2"], label="logistic l2", positive=True),
        max_iterations=_positive_int(logistic_raw["max_iterations"], label="logistic iterations"),
        tolerance=_number(logistic_raw["tolerance"], label="logistic tolerance", positive=True),
        prior_strength=_number(logistic_raw["prior_strength"], label="logistic prior"),
    )
    knn = KnnSettings(
        neighbors=_positive_int(knn_raw["neighbors"], label="kNN neighbors"),
        similarity_power=_number(knn_raw["similarity_power"], label="kNN power", positive=True),
        prior_strength=_number(knn_raw["prior_strength"], label="kNN prior"),
        minimum_weight=_number(
            knn_raw["minimum_weight"], label="kNN minimum weight", positive=True
        ),
    )
    _require(logistic.prior_strength >= 0 and knn.prior_strength >= 0, "negative model prior")
    calibration_bins = _positive_int(raw["calibration_bins"], label="calibration bins")
    _require(calibration_bins >= 2, "calibration bins must be at least two")
    return FrozenConfig(
        path=snapshot.path,
        sha256=snapshot.sha256,
        activity_threshold_um=activity,
        identity_threshold=threshold,
        folds=folds,
        seed=cast(int, seed),
        bootstrap_replicates=_nonnegative_int(
            raw["bootstrap_replicates"], label="bootstrap replicates"
        ),
        calibration_bins=calibration_bins,
        similarity_bin_edges=edges,
        sequences_sha256=_sha_field(raw["sequences_sha256"], label="sequences hash"),
        ledger_sha256=_sha_field(raw["endpoint_context_ledger_sha256"], label="ledger hash"),
        assignments_sha256=_sha_field(raw["split_assignments_sha256"], label="assignments hash"),
        components_sha256=_sha_field(raw["split_components_sha256"], label="components hash"),
        split_manifest_sha256=_sha_field(raw["split_manifest_sha256"], label="split manifest hash"),
        split_audit_sha256=_sha_field(raw["split_audit_sha256"], label="split audit hash"),
        split_top_sha256=_sha_field(raw["split_top_manifest_sha256"], label="split top hash"),
        split_receipt_sha256=_sha_field(
            raw["split_independent_receipt_sha256"], label="split receipt hash"
        ),
        expected_sequences=_positive_int(raw["expected_sequences"], label="expected sequences"),
        expected_ledger_rows=_positive_int(
            raw["expected_ledger_rows"], label="expected ledger rows"
        ),
        expected_homology_components=_positive_int(
            raw["expected_split_homology_components"], label="expected homology components"
        ),
        expected_union_components=_positive_int(
            raw["expected_split_union_components"], label="expected union components"
        ),
        expected_examples=expected_examples,
        expected_source_observations=_positive_int(
            raw["expected_source_observations"], label="expected source observations"
        ),
        expected_modeled_sequences=_positive_int(
            raw["expected_modeled_sequences"], label="expected modeled sequences"
        ),
        expected_positives=expected_positives,
        expected_negatives=expected_negatives,
        examples_by_fold=examples_by_fold,
        positives_by_fold=positives_by_fold,
        negatives_by_fold=negatives_by_fold,
        logistic=logistic,
        knn=knn,
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
    observed = (
        _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"], label="HEAD").decode().strip()
    )
    _require(observed == expected_commit, "repository HEAD differs from expected Gate-1 commit")
    _require(
        _git(repository, ["diff", "--no-ext-diff", "--quiet", "--exit-code", "--"], label="tree")
        == b"",
        "repository tracked tree is dirty",
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
    untracked = _git(
        repository, ["ls-files", "--others", "--exclude-standard", "-z"], label="untracked"
    )
    _require(untracked == b"", "repository contains untracked files")


def _committed_blob(repository: Path, commit: str, logical_path: str) -> bytes:
    kind = _git(repository, ["cat-file", "-t", f"{commit}:{logical_path}"], label=logical_path)
    _require(kind == b"blob\n", f"committed path is not a blob: {logical_path}")
    return _git(repository, ["cat-file", "blob", f"{commit}:{logical_path}"], label=logical_path)


def _verify_execution_source(repository: Path, expected_commit: str) -> None:
    requested = Path(__file__)
    _require_no_symlink(requested, label="executing verifier", ancestors=True)
    executing = requested.resolve(strict=True)
    expected = (repository / _VERIFIER_MODULE_PATH).resolve(strict=True)
    _require(executing == expected, "executing verifier is a stale installation")
    payload = _read_snapshot(expected, label="repository verifier source").payload
    _require(
        payload == _committed_blob(repository, expected_commit, _VERIFIER_MODULE_PATH),
        "executing verifier differs from committed blob",
    )


def _verify_code_manifest(
    payload: bytes,
    *,
    repository: Path,
    expected_commit: str,
    config: FrozenConfig,
) -> dict[str, object]:
    entries = _parse_sha_manifest(payload, label="Gate-1 code manifest")
    source_root = repository / "src/amp_challenge"
    _require(source_root.is_dir() and not source_root.is_symlink(), "source tree missing")
    source_paths: set[str] = set()
    for source in source_root.rglob("*.py"):
        _require(not source.is_symlink(), f"source is symbolic: {source}")
        if source.is_file():
            source_paths.add(source.relative_to(repository).as_posix())
    expected_paths = set(_FIXED_CODE_PATHS) | source_paths
    _require(set(entries) == expected_paths, "Gate-1 code manifest inventory mismatch")
    _require(_PRODUCER_MODULE_PATH in entries, "code manifest omits producer module")
    _require(_VERIFIER_MODULE_PATH in entries, "code manifest omits independent verifier")
    for logical_path in sorted(expected_paths):
        source = _read_snapshot(repository / logical_path, label=f"code entry {logical_path}")
        _require(source.sha256 == entries[logical_path], f"code hash mismatch: {logical_path}")
        _require(
            source.payload == _committed_blob(repository, expected_commit, logical_path),
            f"code entry differs from committed blob: {logical_path}",
        )
    digest = _sha256(payload)
    return {
        "schema_version": 1,
        "code_manifest_sha256": digest,
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


def _prefixed_tree_hashes(root: Path, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(_tree_inventory(root)):
        result[f"{prefix}/{name}"] = _read_snapshot(root / name, label=f"{prefix} {name}").sha256
    return result


def _verify_frozen_input_manifest(
    payload: bytes,
    *,
    normalized_run: Path,
    context_run: Path,
    split_run: Path,
    split_receipt: Snapshot,
) -> str:
    entries = _parse_sha_manifest(payload, label="Gate-1 frozen-input manifest")
    expected: dict[str, str] = {}
    expected.update(_prefixed_tree_hashes(normalized_run, "parser-v7"))
    expected.update(_prefixed_tree_hashes(context_run, "endpoint-context"))
    expected.update(_prefixed_tree_hashes(split_run, "homology-study-split"))
    expected["split-independent-receipt.json"] = split_receipt.sha256
    _require(
        entries == dict(sorted(expected.items())),
        "frozen-input manifest differs from paired inputs",
    )
    return _sha256(payload)


def _verify_handshake(twin_root: Path) -> dict[str, object]:
    receipt_root = twin_root / "node-receipts"
    _require(
        receipt_root.is_dir() and not receipt_root.is_symlink(), "node-receipts directory missing"
    )
    inventory = _tree_inventory(receipt_root)
    expected = frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"})
    _require(inventory == expected, "production handshake inventory mismatch")
    _require(receipt_root.stat().st_mode & 0o222 == 0, "node-receipts directory remains writable")
    root_job_id = twin_root.name
    _require(root_job_id.isdigit(), "Gate-1 twin root basename is not the array job ID")
    receipt_payloads: dict[int, bytes] = {}
    node_names: dict[int, str] = {}
    for task in (0, 1):
        snapshot = _read_snapshot(receipt_root / f"{task}.receipt", label=f"task {task} receipt")
        _require(snapshot.path.stat().st_mode & 0o222 == 0, f"task {task} receipt remains writable")
        _require(
            snapshot.payload.endswith(b"\n") and b"\r" not in snapshot.payload,
            f"task {task} receipt must use LF framing",
        )
        try:
            lines = snapshot.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise VerificationError("node receipt is not UTF-8") from error
        _require(
            len(lines) == 3
            and lines[0] == f"array_job_id={root_job_id}"
            and lines[1] == f"array_task_id={task}"
            and lines[2].startswith("node_name="),
            f"task {task} node receipt is invalid",
        )
        node_name = lines[2].removeprefix("node_name=")
        _require(_SAFE_NODE_RE.fullmatch(node_name) is not None, f"task {task} node name is unsafe")
        receipt_payloads[task] = snapshot.payload
        node_names[task] = node_name
    _require(node_names[0] != node_names[1], "production twins did not run on distinct nodes")
    ack_hashes: dict[str, str] = {}
    receipt_hashes = {str(task): _sha256(payload) for task, payload in receipt_payloads.items()}
    for task in (0, 1):
        snapshot = _read_snapshot(
            receipt_root / f"{task}.ack", label=f"task {task} acknowledgement"
        )
        _require(
            snapshot.path.stat().st_mode & 0o222 == 0,
            f"task {task} acknowledgement remains writable",
        )
        expected_payload = (
            f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode("ascii")
        )
        _require(snapshot.payload == expected_payload, f"task {task} acknowledgement is invalid")
        ack_hashes[str(task)] = snapshot.sha256
    return {
        "distinct_nodes": True,
        "bidirectional_acknowledgement": True,
        "receipt_sha256": receipt_hashes,
        "acknowledgement_sha256": ack_hashes,
    }


def _verify_split_chain(
    *,
    config: FrozenConfig,
    sequences: Snapshot,
    ledger: Snapshot,
    assignments: Snapshot,
    components: Snapshot,
    split_manifest: Snapshot,
    split_audit: Snapshot,
    split_top: Snapshot,
    split_receipt: Snapshot,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    _require(sequences.sha256 == config.sequences_sha256, "sequence hash differs from config")
    _require(ledger.sha256 == config.ledger_sha256, "ledger hash differs from config")
    _require(assignments.sha256 == config.assignments_sha256, "assignment hash differs from config")
    _require(components.sha256 == config.components_sha256, "component hash differs from config")
    _require(split_manifest.sha256 == config.split_manifest_sha256, "split manifest hash differs")
    _require(split_audit.sha256 == config.split_audit_sha256, "split audit hash differs")
    _require(split_top.sha256 == config.split_top_sha256, "split top-manifest hash differs")
    _require(split_receipt.sha256 == config.split_receipt_sha256, "split receipt hash differs")
    _require(config.split_top_sha256 == _EXPECTED_SPLIT_TOP_SHA256, "unaccepted split top hash")
    _require(
        config.split_receipt_sha256 == _EXPECTED_SPLIT_RECEIPT_SHA256,
        "unaccepted split receipt hash",
    )

    top = _parse_sha_manifest(split_top.payload, label="accepted split top manifest")
    _require(set(top) == _SPLIT_TOP_ENTRIES, "accepted split top inventory changed")
    _require(
        top["split/sequence_assignments.jsonl"] == assignments.sha256
        and top["split/components.jsonl"] == components.sha256
        and top["split/manifest.json"] == split_manifest.sha256
        and top["split/audit.json"] == split_audit.sha256,
        "accepted split top manifest does not bind consumed artifacts",
    )
    manifest = _json_object(
        split_manifest.payload, label="accepted split manifest", canonical_pretty=True
    )
    audit = _json_object(split_audit.payload, label="accepted split audit", canonical_pretty=True)
    receipt = _json_object(
        split_receipt.payload, label="accepted split receipt", canonical_pretty=True
    )
    _exact_fields(
        manifest,
        {
            "artifact",
            "artifacts",
            "config_sha256",
            "counts",
            "input",
            "policies",
            "provenance",
            "runtime",
            "schema_version",
            "status",
            "upstream",
        },
        label="accepted split manifest",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["artifact"] == _SPLIT_ARTIFACT
        and manifest["status"] == _SPLIT_STATUS,
        "accepted split manifest identity changed",
    )
    manifest_inputs = manifest["input"]
    manifest_artifacts = manifest["artifacts"]
    manifest_policies = manifest["policies"]
    counts = manifest["counts"]
    _require(
        isinstance(manifest_inputs, dict)
        and cast(dict[str, object], manifest_inputs).get("sequences")
        == {"filename": "sequences.jsonl", "sha256": sequences.sha256}
        and cast(dict[str, object], manifest_inputs).get("endpoint_context_ledger")
        == {
            "filename": "endpoint_context_ledger.jsonl",
            "sha256": ledger.sha256,
        },
        "accepted split manifest input links changed",
    )
    _require(isinstance(manifest_artifacts, dict), "accepted split artifacts are invalid")
    for key, filename, digest in (
        ("sequence_assignments", "sequence_assignments.jsonl", assignments.sha256),
        ("components", "components.jsonl", components.sha256),
        ("audit", "audit.json", split_audit.sha256),
    ):
        _require(
            cast(dict[str, object], manifest_artifacts).get(key)
            == {"filename": filename, "sha256": digest},
            f"accepted split artifact link changed: {key}",
        )
    _require(
        isinstance(manifest_policies, dict)
        and manifest_policies.get("component") == _SPLIT_COMPONENT_POLICY
        and manifest_policies.get("homology") == _HOMOLOGY_POLICY
        and "never a model feature" in str(manifest_policies.get("study", "")),
        "accepted split policies changed",
    )
    _require(isinstance(counts, dict), "accepted split counts invalid")
    expected_counts = {
        "folds": config.folds,
        "unique_sequences": config.expected_sequences,
        "sequence_assignments": config.expected_sequences,
        "homology_components": config.expected_homology_components,
        "union_components": config.expected_union_components,
        "retained_balance_contexts": config.expected_examples,
    }
    for key, value in expected_counts.items():
        _require(counts.get(key) == value, f"accepted split census changed: {key}")

    _exact_fields(
        audit,
        {
            "artifact",
            "balance",
            "graph",
            "input",
            "invariants",
            "limitations",
            "schema_version",
            "status",
        },
        label="accepted split audit",
    )
    _require(
        audit["schema_version"] == 1
        and audit["artifact"] == "homology_study_union_split_audit"
        and audit["status"] == _SPLIT_STATUS,
        "accepted split audit identity changed",
    )
    graph = audit["graph"]
    invariants = audit["invariants"]
    _require(
        isinstance(graph, dict)
        and graph.get("component_policy") == _SPLIT_COMPONENT_POLICY
        and graph.get("homology_algorithm") == _HOMOLOGY_POLICY
        and graph.get("identity_threshold") == config.identity_threshold
        and graph.get("homology_components") == config.expected_homology_components
        and graph.get("union_components") == config.expected_union_components,
        "accepted split graph contract changed",
    )
    _require(
        isinstance(invariants, dict)
        and bool(invariants)
        and all(value is True for value in invariants.values()),
        "accepted split audit invariants failed",
    )

    _exact_fields(
        receipt,
        {
            "artifact",
            "artifact_sha256",
            "balance",
            "checks",
            "code_manifest_sha256",
            "counts",
            "endpoint_context_top_manifest_sha256",
            "frozen_input_manifest_sha256",
            "git_commit",
            "graph",
            "normalized_top_manifest_sha256",
            "schema_version",
            "status",
            "top_manifest_sha256",
        },
        label="accepted split receipt",
    )
    _require(
        receipt["schema_version"] == 1
        and receipt["artifact"] == "homology_study_union_split_independent_verification"
        and receipt["status"] == "passed"
        and receipt["git_commit"] == _EXPECTED_SPLIT_GIT_COMMIT,
        "accepted split receipt identity changed",
    )
    checks = receipt["checks"]
    _require(
        isinstance(checks, dict)
        and set(checks) == _UPSTREAM_SPLIT_CHECKS
        and all(value is True for value in checks.values()),
        "accepted split receipt checks changed or failed",
    )
    artifact_hashes = receipt["artifact_sha256"]
    _require(
        isinstance(artifact_hashes, dict)
        and artifact_hashes
        == {
            "audit": split_audit.sha256,
            "components": components.sha256,
            "grouping_edges": top["split/grouping_edges.jsonl"],
            "sequence_assignments": assignments.sha256,
        },
        "accepted split receipt artifact hashes changed",
    )
    _require(receipt["top_manifest_sha256"] == split_top.sha256, "split receipt top link changed")
    _require(receipt["code_manifest_sha256"] == top["CODE_SHA256SUMS"], "split code link changed")
    _require(
        receipt["frozen_input_manifest_sha256"] == top["FROZEN_INPUT_SHA256SUMS"],
        "split input link changed",
    )
    _require(
        receipt["counts"] == counts and receipt["graph"] == graph, "split receipt semantics changed"
    )
    _require(
        receipt["normalized_top_manifest_sha256"] == _EXPECTED_NORMALIZED_TOP_SHA256
        and receipt["endpoint_context_top_manifest_sha256"] == _EXPECTED_CONTEXT_TOP_SHA256,
        "split receipt upstream links changed",
    )
    audit_balance = audit["balance"]
    receipt_balance = receipt["balance"]
    _require(
        isinstance(audit_balance, dict) and isinstance(receipt_balance, dict),
        "split balance invalid",
    )
    for key in ("by_fold", "context_census", "totals"):
        _require(
            receipt_balance.get(key) == audit_balance.get(key), f"split balance mismatch: {key}"
        )
    provenance = manifest["provenance"]
    _require(
        isinstance(provenance, dict)
        and provenance.get("git_commit") == receipt["git_commit"]
        and isinstance(provenance.get("code_manifest"), dict)
        and cast(dict[str, object], provenance["code_manifest"]).get("sha256")
        == receipt["code_manifest_sha256"],
        "split provenance chain changed",
    )
    return manifest, audit, receipt


def _canonical_sequence(value: object, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} is not a string")
    sequence = "".join(cast(str, value).split()).upper()
    _require(sequence == value, f"{label} is not canonical")
    _require(_MIN_LENGTH <= len(sequence) <= _MAX_LENGTH, f"{label} length is invalid")
    _require(set(sequence) <= _AMINO_ACID_SET, f"{label} contains a nonstandard residue")
    return sequence


def _read_sequences(snapshot: Snapshot, *, config: FrozenConfig) -> dict[str, str]:
    rows = _jsonl_rows(snapshot.payload, label="parser-v7 sequences", canonical=False)
    _require(len(rows) == config.expected_sequences, "sequence census differs from config")
    result: dict[str, str] = {}
    previous: str | None = None
    for number, row in enumerate(rows, start=1):
        label = f"sequence row {number}"
        _exact_fields(row, _SEQUENCE_FIELDS, label=label)
        sequence = _canonical_sequence(row["sequence"], label=f"{label} sequence")
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence ID")
        _require(sequence_id == _sha256(sequence.encode("ascii")), f"{label} sequence ID mismatch")
        _require(
            isinstance(row["provenance"], list) and bool(row["provenance"]), f"{label} provenance"
        )
        _require(sequence_id not in result, f"{label} repeats sequence ID")
        if previous is not None:
            _require(sequence_id > previous, "sequence table is not strictly sorted")
        result[sequence_id] = sequence
        previous = sequence_id
    return result


def _read_assignments(
    snapshot: Snapshot,
    *,
    config: FrozenConfig,
    sequences: Mapping[str, str],
) -> dict[str, Assignment]:
    rows = _jsonl_rows(snapshot.payload, label="accepted split assignments", canonical=True)
    _require(len(rows) == config.expected_sequences, "assignment census differs from config")
    result: dict[str, Assignment] = {}
    previous: str | None = None
    homology_locations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    union_folds: dict[str, set[int]] = defaultdict(set)
    for number, row in enumerate(rows, start=1):
        label = f"assignment row {number}"
        _exact_fields(row, _ASSIGNMENT_FIELDS, label=label)
        _require(
            row["schema_version"] == 1 and type(row["schema_version"]) is int, f"bad {label} schema"
        )
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence ID")
        homology_id = _sha_field(row["homology_component_id"], label=f"{label} homology ID")
        union_id = _sha_field(row["union_component_id"], label=f"{label} union ID")
        fold = row["fold"]
        _require(type(fold) is int and 0 <= cast(int, fold) < config.folds, f"bad {label} fold")
        _require(sequence_id in sequences and sequence_id not in result, f"bad {label} support")
        if previous is not None:
            _require(sequence_id > previous, "assignment table is not strictly sorted")
        assignment = Assignment(sequence_id, homology_id, union_id, cast(int, fold))
        result[sequence_id] = assignment
        homology_locations[homology_id].add((union_id, assignment.fold))
        union_folds[union_id].add(assignment.fold)
        previous = sequence_id
    _require(set(result) == set(sequences), "assignments do not exactly cover sequences")
    _require(
        all(len(value) == 1 for value in homology_locations.values()), "homology component split"
    )
    _require(all(len(value) == 1 for value in union_folds.values()), "union component split")
    _require(
        len(homology_locations) == config.expected_homology_components, "homology census changed"
    )
    _require(len(union_folds) == config.expected_union_components, "union census changed")
    _require(
        {row.fold for row in result.values()} == set(range(config.folds)), "fold support changed"
    )
    return result


def _sequence_set_sha256(sequence_ids: Iterable[str]) -> str:
    return _sha256("".join(f"{item}\n" for item in sorted(sequence_ids)).encode("ascii"))


def _read_components(
    snapshot: Snapshot,
    *,
    config: FrozenConfig,
    assignments: Mapping[str, Assignment],
) -> dict[str, dict[str, object]]:
    rows = _jsonl_rows(snapshot.payload, label="accepted split components", canonical=True)
    _require(len(rows) == config.expected_union_components, "component census differs from config")
    fields = {
        "schema_version",
        "union_component_id",
        "fold",
        "sequence_count",
        "sequence_ids_sha256",
        "grouping_edge_ids",
        "homology_component_count",
        "study_key_count",
        "balance_counts",
    }
    members: dict[str, list[Assignment]] = defaultdict(list)
    for assignment in assignments.values():
        members[assignment.union_component_id].append(assignment)
    result: dict[str, dict[str, object]] = {}
    previous: str | None = None
    for number, row in enumerate(rows, start=1):
        label = f"component row {number}"
        _exact_fields(row, fields, label=label)
        _require(
            row["schema_version"] == 1 and type(row["schema_version"]) is int, f"bad {label} schema"
        )
        component_id = _sha_field(row["union_component_id"], label=f"{label} ID")
        _require(component_id not in result and component_id in members, f"bad {label} support")
        if previous is not None:
            _require(component_id > previous, "component table is not strictly sorted")
        selected = members[component_id]
        fold = row["fold"]
        _require(
            type(fold) is int and {item.fold for item in selected} == {fold},
            f"{label} fold mismatch",
        )
        _require(row["sequence_count"] == len(selected), f"{label} sequence count mismatch")
        _require(
            row["sequence_ids_sha256"]
            == _sequence_set_sha256(item.sequence_id for item in selected),
            f"{label} sequence digest mismatch",
        )
        _require(
            row["homology_component_count"]
            == len({item.homology_component_id for item in selected}),
            f"{label} homology count mismatch",
        )
        _nonnegative_int(row["study_key_count"], label=f"{label} study count")
        grouping = row["grouping_edge_ids"]
        _require(isinstance(grouping, list) and bool(grouping), f"{label} grouping edges invalid")
        grouping_ids = [
            _sha_field(item, label=f"{label} grouping edge")
            for item in cast(list[object], grouping)
        ]
        _require(grouping_ids == sorted(set(grouping_ids)), f"{label} grouping edges unsorted")
        balance = row["balance_counts"]
        _require(
            isinstance(balance, dict)
            and set(balance) == {"component_count", "sequences", *_CLASS_METRICS},
            f"{label} balance schema invalid",
        )
        for key, value in cast(dict[str, object], balance).items():
            _nonnegative_int(value, label=f"{label} balance {key}")
        _require(
            balance["component_count"] == 1 and balance["sequences"] == len(selected),
            f"{label} balance mismatch",
        )
        result[component_id] = row
        previous = component_id
    _require(set(result) == set(members), "component table support differs from assignments")
    return result


def _context_semantics(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["sequence_id"],
        row["endpoint"],
        row["context_id"],
        json.dumps(row["source_conditions"], sort_keys=True, separators=(",", ":")),
        json.dumps(row["exposure_concentration"], sort_keys=True, separators=(",", ":")),
    )


def _group_ledger(
    snapshot: Snapshot,
    *,
    config: FrozenConfig,
    sequences: Mapping[str, str],
) -> tuple[tuple[dict[str, object], ...], dict[str, list[dict[str, object]]]]:
    rows = _jsonl_rows(snapshot.payload, label="endpoint-context ledger", canonical=True)
    _require(len(rows) == config.expected_ledger_rows, "ledger census differs from config")
    contexts: dict[str, list[dict[str, object]]] = defaultdict(list)
    observations: set[str] = set()
    previous: str | None = None
    for number, row in enumerate(rows, start=1):
        label = f"ledger row {number}"
        _exact_fields(row, _LEDGER_FIELDS, label=label)
        _require(
            row["schema_version"] == 1 and type(row["schema_version"]) is int, f"bad {label} schema"
        )
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence ID")
        _require(sequence_id in sequences, f"{label} references unknown sequence")
        for field in (
            "observation_id",
            "assay_row_sha256",
            "assay_context_id",
            "context_id",
            "provenance_id",
        ):
            _sha_field(row[field], label=f"{label} {field}")
        observation_id = cast(str, row["observation_id"])
        _require(observation_id not in observations, f"{label} repeats observation ID")
        if previous is not None:
            _require(observation_id > previous, "ledger is not strictly sorted")
        observations.add(observation_id)
        previous = observation_id
        _require(row["endpoint"] in {"mic", "hc50", "hemolysis_percent"}, f"bad {label} endpoint")
        tasks = _string_array(row["eligible_tasks"], label=f"{label} eligible tasks")
        _string_array(row["exclusion_codes"], label=f"{label} exclusions")
        _string_array(
            row["study_keys"], label=f"{label} study keys", allow_empty=False, sorted_unique=False
        )
        if "bacterial_mic16" in tasks:
            source_gram = row["source_gram"]
            gram_tasks = set(tasks) & {"gram_negative_mic16", "gram_positive_mic16"}
            _require(
                row["endpoint"] == "mic"
                and row["mapping_status"] == "mapped_single_supported_species"
                and row["gram_resolution"] == "concordant"
                and source_gram in {"negative", "positive"}
                and row["expected_gram"] == source_gram
                and gram_tasks == {f"gram_{source_gram}_mic16"}
                and type(row["mic16_label"]) is int
                and row["mic16_label"] in {0, 1}
                and isinstance(row["canonical_target"], str)
                and bool(cast(str, row["canonical_target"]).strip())
                and cast(str, row["canonical_target"])
                == cast(str, row["canonical_target"]).strip(),
                f"{label} has inconsistent MIC16 eligibility",
            )
        contexts[cast(str, row["assay_context_id"])].append(row)
    for context_id, members in contexts.items():
        semantics = _context_semantics(members[0])
        _require(
            all(_context_semantics(item) == semantics for item in members),
            f"context {context_id} has inconsistent identity semantics",
        )
    return rows, contexts


def _reconstruct_panel(
    contexts: Mapping[str, Sequence[dict[str, object]]],
    *,
    config: FrozenConfig,
    sequences: Mapping[str, str],
    assignments: Mapping[str, Assignment],
    components: Mapping[str, Mapping[str, object]],
    split_audit: Mapping[str, object],
) -> tuple[ReconstructedPanel, dict[str, dict[str, int]]]:
    examples: list[Example] = []
    audit_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    retained_sequences: set[str] = set()
    counts["ledger_assay_contexts"] = len(contexts)
    counts["ledger_raw_observations"] = sum(len(value) for value in contexts.values())
    for assay_context_id in sorted(contexts):
        members = sorted(
            contexts[assay_context_id], key=lambda row: cast(str, row["observation_id"])
        )
        first = members[0]
        endpoint = cast(str, first["endpoint"])
        sequence_id = cast(str, first["sequence_id"])
        eligible = ["bacterial_mic16" in cast(list[str], row["eligible_tasks"]) for row in members]
        eligible_count = sum(eligible)
        reason_codes: tuple[str, ...] = ()
        example: Example | None = None
        if endpoint != "mic":
            counts["non_mic_assay_contexts"] += 1
            counts["non_mic_raw_observations"] += len(members)
            status = "excluded_non_mic_endpoint"
        else:
            counts["mic_assay_contexts"] += 1
            counts["mic_raw_observations"] += len(members)
            counts["repeated_mic_assay_contexts"] += int(len(members) > 1)
            counts["eligible_raw_observations"] += eligible_count
            if eligible_count == 0:
                counts["all_ineligible_assay_contexts"] += 1
                counts["observations_in_all_ineligible_assay_contexts"] += len(members)
                status = "excluded_all_ineligible_context"
            elif eligible_count != len(members):
                counts["candidate_assay_contexts"] += 1
                counts["mixed_eligibility_assay_contexts"] += 1
                counts["observations_in_mixed_eligibility_assay_contexts"] += len(members)
                counts["eligible_observations_in_mixed_eligibility_assay_contexts"] += (
                    eligible_count
                )
                status = "excluded_mixed_eligibility_context"
                reason_codes = ("not_every_context_member_is_bacterial_mic16_eligible",)
            else:
                counts["candidate_assay_contexts"] += 1
                counts["fully_eligible_assay_contexts"] += 1
                labels = {cast(int, row["mic16_label"]) for row in members}
                grams = {cast(str, row["source_gram"]) for row in members}
                targets = {cast(str, row["canonical_target"]) for row in members}
                conflicts: list[str] = []
                if len(labels) != 1:
                    conflicts.append("mic16_label_disagreement")
                if len(grams) != 1:
                    conflicts.append("source_gram_disagreement")
                if len(targets) != 1:
                    conflicts.append("canonical_target_disagreement")
                if conflicts:
                    counts["conflicting_assay_contexts"] += 1
                    counts["eligible_label_conflict_contexts"] += int(
                        "mic16_label_disagreement" in conflicts
                    )
                    counts["eligible_gram_conflict_contexts"] += int(
                        "source_gram_disagreement" in conflicts
                    )
                    counts["eligible_target_conflict_contexts"] += int(
                        "canonical_target_disagreement" in conflicts
                    )
                    counts["observations_in_conflicting_assay_contexts"] += len(members)
                    status = "excluded_conflicting_context"
                    reason_codes = tuple(conflicts)
                else:
                    assignment = assignments.get(sequence_id)
                    _require(
                        assignment is not None,
                        f"retained context {assay_context_id} has no assignment",
                    )
                    assert assignment is not None
                    label = next(iter(labels))
                    gram = cast(GramClass, next(iter(grams)))
                    target = next(iter(targets))
                    example = Example(
                        example_id=assay_context_id,
                        assay_context_id=assay_context_id,
                        sequence_id=sequence_id,
                        sequence=sequences[sequence_id],
                        canonical_target=target,
                        gram=gram,
                        label=label,
                        observation_ids=tuple(cast(str, row["observation_id"]) for row in members),
                        fold=assignment.fold,
                        homology_component_id=assignment.homology_component_id,
                        union_component_id=assignment.union_component_id,
                    )
                    examples.append(example)
                    retained_sequences.add(sequence_id)
                    counts["retained_assay_contexts"] += 1
                    counts["retained_source_observations"] += len(members)
                    classes[f"gram_{gram}_mic16_{'positive' if label else 'negative'}"] += 1
                    status = "included_fully_eligible_unanimous_context"
        audit_rows.append(
            {
                "schema_version": 1,
                "assay_context_id": assay_context_id,
                "sequence_id": sequence_id,
                "endpoint": endpoint,
                "source_observations": len(members),
                "eligible_source_observations": eligible_count,
                "status": status,
                "reason_codes": list(reason_codes),
                "example_id": None if example is None else example.example_id,
            }
        )
    counts["retained_sequences"] = len(retained_sequences)
    census_keys = (
        "ledger_raw_observations",
        "ledger_assay_contexts",
        "mic_raw_observations",
        "mic_assay_contexts",
        "non_mic_raw_observations",
        "non_mic_assay_contexts",
        "repeated_mic_assay_contexts",
        "eligible_raw_observations",
        "candidate_assay_contexts",
        "all_ineligible_assay_contexts",
        "observations_in_all_ineligible_assay_contexts",
        "fully_eligible_assay_contexts",
        "mixed_eligibility_assay_contexts",
        "observations_in_mixed_eligibility_assay_contexts",
        "eligible_observations_in_mixed_eligibility_assay_contexts",
        "conflicting_assay_contexts",
        "eligible_label_conflict_contexts",
        "eligible_gram_conflict_contexts",
        "eligible_target_conflict_contexts",
        "observations_in_conflicting_assay_contexts",
        "retained_assay_contexts",
        "retained_source_observations",
        "retained_sequences",
    )
    census: dict[str, object] = {key: counts[key] for key in census_keys}
    census["retained_class_counts"] = {key: classes[key] for key in _CLASS_METRICS}
    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    panel = ReconstructedPanel(ordered_examples, tuple(audit_rows), census)
    _require(len(ordered_examples) == config.expected_examples, "retained example census changed")
    _require(
        sum(len(item.observation_ids) for item in ordered_examples)
        == config.expected_source_observations,
        "retained observation census changed",
    )
    _require(
        len(retained_sequences) == config.expected_modeled_sequences,
        "modeled sequence census changed",
    )
    positives = sum(item.label for item in ordered_examples)
    _require(positives == config.expected_positives, "positive census changed")
    _require(
        len(ordered_examples) - positives == config.expected_negatives, "negative census changed"
    )
    _require(census["eligible_target_conflict_contexts"] == 0, "target conflicts are not accepted")

    balance = split_audit.get("balance")
    _require(isinstance(balance, dict), "accepted split balance is invalid")
    accepted_census = balance.get("context_census")
    _require(isinstance(accepted_census, dict), "accepted split context census is invalid")
    for key, value in census.items():
        if key != "eligible_target_conflict_contexts":
            _require(
                accepted_census.get(key) == value, f"context census differs from split audit: {key}"
            )

    accepted_by_fold = balance.get("by_fold")
    _require(isinstance(accepted_by_fold, dict), "accepted split fold census is invalid")
    fold_summary: dict[str, dict[str, int]] = {}
    for fold in range(config.folds):
        selected = [item for item in ordered_examples if item.fold == fold]
        positive = sum(item.label for item in selected)
        negative = len(selected) - positive
        _require(
            len(selected) == config.examples_by_fold[fold], f"fold {fold} example count changed"
        )
        _require(positive == config.positives_by_fold[fold], f"fold {fold} positive count changed")
        _require(negative == config.negatives_by_fold[fold], f"fold {fold} negative count changed")
        class_counts = {
            metric: sum(
                item.gram == metric.split("_")[1] and item.label == int(metric.endswith("positive"))
                for item in selected
            )
            for metric in _CLASS_METRICS
        }
        accepted = accepted_by_fold.get(str(fold))
        _require(
            isinstance(accepted, dict)
            and all(accepted.get(metric) == value for metric, value in class_counts.items()),
            f"fold {fold} class counts differ from split audit",
        )
        fold_summary[str(fold)] = {
            "examples": len(selected),
            "source_observations": sum(len(item.observation_ids) for item in selected),
            "sequences": len({item.sequence_id for item in selected}),
            "homology_components": len({item.homology_component_id for item in selected}),
            "union_components": len({item.union_component_id for item in selected}),
            "positives": positive,
            "negatives": negative,
            **class_counts,
        }
    by_component: dict[str, list[Example]] = defaultdict(list)
    for example in ordered_examples:
        by_component[example.union_component_id].append(example)
    for component_id, component in components.items():
        expected_balance = component["balance_counts"]
        assert isinstance(expected_balance, dict)
        selected = by_component.get(component_id, [])
        for metric in _CLASS_METRICS:
            observed = sum(
                item.gram == metric.split("_")[1] and item.label == int(metric.endswith("positive"))
                for item in selected
            )
            _require(
                expected_balance[metric] == observed, f"component {component_id} balance changed"
            )
    return panel, fold_summary


@lru_cache(maxsize=1_000_000)
def _global_identity(left: str, right: str) -> float:
    """Independently reproduce the frozen +1/-1/-1 global alignment identity."""

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


def _cross_fold_identities(
    sequences: Mapping[str, str],
    assignments: Mapping[str, Assignment],
    modeled_sequence_ids: Iterable[str],
    threshold: float,
) -> tuple[float, dict[str, float]]:
    ordered = tuple(sorted(sequences))
    modeled = set(modeled_sequence_ids)
    modeled_maximum = {sequence_id: 0.0 for sequence_id in modeled}
    maximum = 0.0
    for left_index, left_id in enumerate(ordered):
        left = sequences[left_id]
        for right_id in ordered[left_index + 1 :]:
            if assignments[left_id].fold == assignments[right_id].fold:
                continue
            right = sequences[right_id]
            upper_bound = min(len(left), len(right)) / max(len(left), len(right))
            if upper_bound < maximum and left_id not in modeled and right_id not in modeled:
                continue
            identity = _global_identity(left, right)
            maximum = max(maximum, identity)
            _require(
                identity < threshold, f"cross-fold identity reaches threshold: {left_id}/{right_id}"
            )
            if left_id in modeled and right_id in modeled:
                modeled_maximum[left_id] = max(modeled_maximum[left_id], identity)
                modeled_maximum[right_id] = max(modeled_maximum[right_id], identity)
    _require(
        all(value < threshold for value in modeled_maximum.values()),
        "modeled identity threshold violated",
    )
    return maximum, dict(sorted(modeled_maximum.items()))


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


def _smoothed_prevalence(labels: FloatArray, strength: float) -> float:
    return float((np.sum(labels) + 0.5 * strength) / (labels.size + strength))


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _design_matrix(
    rows: Sequence[Example],
    *,
    targets: tuple[str, ...],
    mean: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    continuous = np.asarray([_descriptor_features(row.sequence) for row in rows])
    standardized = (continuous - mean) / scale
    target_index = {target: index for index, target in enumerate(targets)}
    target_matrix = np.zeros((len(rows), len(targets) + 1), dtype=np.float64)
    for row_index, row in enumerate(rows):
        target_matrix[row_index, target_index.get(row.canonical_target, len(targets))] = 1.0
    gram_matrix = np.zeros((len(rows), 3), dtype=np.float64)
    gram_index = {"positive": 0, "negative": 1, "unknown": 2}
    for row_index, row in enumerate(rows):
        gram_matrix[row_index, gram_index[row.gram]] = 1.0
    intercept = np.ones((len(rows), 1), dtype=np.float64)
    return np.concatenate((intercept, standardized, target_matrix, gram_matrix), axis=1)


def _descriptor_logistic(
    training: Sequence[Example],
    testing: Sequence[Example],
    settings: LogisticSettings,
) -> FloatArray:
    labels = np.asarray([row.label for row in training], dtype=np.float64)
    targets = tuple(sorted({row.canonical_target for row in training}))
    continuous = np.asarray([_descriptor_features(row.sequence) for row in training])
    mean = np.mean(continuous, axis=0)
    scale = np.std(continuous, axis=0)
    scale[scale < 1e-12] = 1.0
    x = _design_matrix(training, targets=targets, mean=mean, scale=scale)
    test_x = _design_matrix(testing, targets=targets, mean=mean, scale=scale)
    prior = _smoothed_prevalence(labels, settings.prior_strength)
    if np.all(labels == labels[0]):
        return np.full(len(testing), prior, dtype=np.float64)
    coefficient = np.zeros(x.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(x.shape[1], settings.l2, dtype=np.float64)
    penalty[0] = 0.0
    for _ in range(settings.max_iterations):
        probability = _sigmoid(x @ coefficient)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = (x.T @ (probability - labels)) / labels.size + penalty * coefficient
        hessian = (x.T * variance) @ x / labels.size
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficient -= step
        if float(np.max(np.abs(step))) <= settings.tolerance:
            break
    return np.clip(_sigmoid(test_x @ coefficient), 1e-6, 1.0 - 1e-6)


def _knn_probability(query: Example, training: Sequence[Example], settings: KnnSettings) -> float:
    candidates = [
        index
        for index, row in enumerate(training)
        if row.canonical_target == query.canonical_target
    ]
    if not candidates:
        candidates = [index for index, row in enumerate(training) if row.gram == query.gram]
    if not candidates:
        candidates = list(range(len(training)))
    labels = np.asarray([training[index].label for index in candidates], dtype=np.float64)
    prior = _smoothed_prevalence(labels, settings.prior_strength)
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


def _make_predictions(
    examples: Sequence[Example],
    *,
    config: FrozenConfig,
    maximum_by_sequence: Mapping[str, float],
) -> tuple[Prediction, ...]:
    items = tuple(sorted(examples, key=lambda item: item.example_id))
    output: list[Prediction] = []
    for fold in range(config.folds):
        training = tuple(item for item in items if item.fold != fold)
        testing = tuple(item for item in items if item.fold == fold)
        _require(bool(training) and bool(testing), f"fold {fold} has empty train or test set")
        logistic = _descriptor_logistic(training, testing, config.logistic)
        knn = np.asarray([_knn_probability(item, training, config.knn) for item in testing])
        probabilities = {
            "descriptor_logistic": logistic,
            "homology_knn": knn,
            "equal_weight_ensemble": np.mean(np.stack((logistic, knn), axis=0), axis=0),
        }
        for index, item in enumerate(testing):
            identity = maximum_by_sequence[item.sequence_id]
            _require(
                identity < config.identity_threshold, "held-out example reaches identity threshold"
            )
            for model, values in probabilities.items():
                probability = float(values[index])
                _require(
                    math.isfinite(probability) and 0.0 <= probability <= 1.0,
                    "invalid model probability",
                )
                output.append(
                    Prediction(
                        model=model,
                        example_id=item.example_id,
                        assay_context_id=item.assay_context_id,
                        sequence_id=item.sequence_id,
                        sequence=item.sequence,
                        canonical_target=item.canonical_target,
                        gram=item.gram,
                        label=item.label,
                        source_observations=len(item.observation_ids),
                        fold=item.fold,
                        homology_component_id=item.homology_component_id,
                        union_component_id=item.union_component_id,
                        max_train_identity=identity,
                        probability=probability,
                    )
                )
    _require(len(output) == len(items) * len(_MODELS), "OOF row count is wrong")
    return tuple(sorted(output, key=lambda item: (item.model, item.example_id)))


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


def _metric_subset(
    predictions: Sequence[Prediction], *, calibration_bins: int
) -> dict[str, int | float | None]:
    return _binary_metrics(
        [item.label for item in predictions],
        [item.probability for item in predictions],
        calibration_bins=calibration_bins,
    )


def _component_bootstrap(
    predictions: Sequence[Prediction],
    *,
    calibration_bins: int,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    names = ("roc_auc", "average_precision", "brier", "log_loss")
    point = _metric_subset(predictions, calibration_bins=calibration_bins)
    values: dict[str, list[float]] = {name: [] for name in names}
    if replicates:
        by_component: dict[str, list[Prediction]] = defaultdict(list)
        for item in predictions:
            by_component[item.union_component_id].append(item)
        component_ids = tuple(sorted(by_component))
        generator = np.random.default_rng(seed)
        for _ in range(replicates):
            sampled = generator.choice(component_ids, size=len(component_ids), replace=True)
            rows = [row for component_id in sampled for row in by_component[str(component_id)]]
            metrics = _metric_subset(rows, calibration_bins=calibration_bins)
            for name in names:
                value = metrics[name]
                if value is not None:
                    values[name].append(float(value))
    result: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        samples = np.asarray(values[name], dtype=np.float64)
        result[name] = {
            "point": cast(float | None, point[name]),
            "lower": None if samples.size == 0 else float(np.quantile(samples, 0.025)),
            "upper": None if samples.size == 0 else float(np.quantile(samples, 0.975)),
            "successful_replicates": int(samples.size),
        }
    return result


def _summarize_predictions(
    predictions: Sequence[Prediction], *, config: FrozenConfig
) -> dict[str, object]:
    by_model: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_model[prediction.model].append(prediction)
    _require(set(by_model) == set(_MODELS), "OOF model support changed")
    output: dict[str, object] = {}
    for model in sorted(by_model):
        rows = tuple(sorted(by_model[model], key=lambda item: item.example_id))
        _require(len(rows) == config.expected_examples, f"{model} OOF support is incomplete")
        by_similarity: dict[str, object] = {}
        for left, right in zip(
            config.similarity_bin_edges, config.similarity_bin_edges[1:], strict=False
        ):
            selected = [
                item
                for item in rows
                if left <= item.max_train_identity < right
                or (right == 1.0 and item.max_train_identity == 1.0)
            ]
            if selected:
                by_similarity[f"[{left:.2f},{right:.2f})"] = _metric_subset(
                    selected, calibration_bins=config.calibration_bins
                )
        output[model] = {
            "overall": _metric_subset(rows, calibration_bins=config.calibration_bins),
            "union_component_bootstrap_95ci": _component_bootstrap(
                rows,
                calibration_bins=config.calibration_bins,
                replicates=config.bootstrap_replicates,
                seed=config.seed,
            ),
            "by_fold": {
                str(fold): _metric_subset(
                    [item for item in rows if item.fold == fold],
                    calibration_bins=config.calibration_bins,
                )
                for fold in range(config.folds)
            },
            "by_gram": {
                gram: _metric_subset(
                    [item for item in rows if item.gram == gram],
                    calibration_bins=config.calibration_bins,
                )
                for gram in ("negative", "positive")
            },
            "by_canonical_target": {
                target: _metric_subset(
                    [item for item in rows if item.canonical_target == target],
                    calibration_bins=config.calibration_bins,
                )
                for target in sorted({item.canonical_target for item in rows})
            },
            "by_max_train_identity": by_similarity,
        }
    aligned = {
        name: {item.example_id: item.probability for item in by_model[name]} for name in _MODELS
    }
    support = set(aligned[_MODELS[0]])
    _require(all(set(values) == support for values in aligned.values()), "model supports differ")
    example_ids = sorted(support)
    left = np.asarray([aligned["descriptor_logistic"][item] for item in example_ids])
    right = np.asarray([aligned["homology_knn"][item] for item in example_ids])
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else None
    output["model_diversity"] = {
        "member_prediction_pearson": correlation,
        "members": ["descriptor_logistic", "homology_knn"],
        "ensemble_policy": "untrained_equal_probability_mean",
    }
    return output


def _oof_csv_bytes(predictions: Sequence[Prediction]) -> bytes:
    fieldnames = [
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
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for prediction in predictions:
        row = asdict(prediction)
        row["max_train_identity"] = f"{prediction.max_train_identity:.17g}"
        row["probability"] = f"{prediction.probability:.17g}"
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _require_exact_payload(actual: bytes, expected: bytes, *, label: str) -> None:
    _require(actual == expected, f"{label} differs from independent reconstruction")


def _verify_publication_run(
    run: Path,
    *,
    normalized_run: Path,
    context_run: Path,
    split_run: Path,
    split_receipt: Snapshot,
    repository: Path,
    expected_commit: str,
    config: FrozenConfig,
) -> tuple[str, str, str, dict[str, object]]:
    publication_top, _ = _verify_manifest_tree(
        run,
        expected_top_sha256=None,
        expected_inventory=_PUBLICATION_FILES,
        label="Gate-1 publication",
    )
    gate1_top, _ = _verify_manifest_tree(
        run / "gate1",
        expected_top_sha256=None,
        expected_inventory=_GATE1_FILES,
        label="Gate-1 semantic output",
    )
    code = _read_snapshot(run / "CODE_SHA256SUMS", label="Gate-1 code manifest")
    frozen = _read_snapshot(run / "FROZEN_INPUT_SHA256SUMS", label="Gate-1 frozen-input manifest")
    code_attestation = _verify_code_manifest(
        code.payload,
        repository=repository,
        expected_commit=expected_commit,
        config=config,
    )
    frozen_digest = _verify_frozen_input_manifest(
        frozen.payload,
        normalized_run=normalized_run,
        context_run=context_run,
        split_run=split_run,
        split_receipt=split_receipt,
    )
    _require(frozen_digest == frozen.sha256, "frozen-input manifest hash changed while reading")
    return publication_top, gate1_top, code.sha256, code_attestation


def _expected_artifacts(
    *,
    panel: ReconstructedPanel,
    fold_summary: Mapping[str, Mapping[str, int]],
    predictions: Sequence[Prediction],
    metrics: Mapping[str, object],
    maximum_cross_fold_identity: float,
    config: FrozenConfig,
    git_commit: str,
    code_attestation: Mapping[str, object],
    input_hashes: Mapping[str, str],
    split_manifest: Mapping[str, object],
    split_receipt: Mapping[str, object],
    assignments: Mapping[str, Assignment],
    components: Mapping[str, Mapping[str, object]],
) -> dict[str, bytes]:
    context_audit = _jsonl_bytes(panel.audit_rows)
    example_rows = [
        {
            "schema_version": 1,
            "example_id": item.example_id,
            "assay_context_id": item.assay_context_id,
            "sequence_id": item.sequence_id,
            "sequence": item.sequence,
            "canonical_target": item.canonical_target,
            "gram": item.gram,
            "label": item.label,
            "source_observations": len(item.observation_ids),
            "fold": item.fold,
            "homology_component_id": item.homology_component_id,
            "union_component_id": item.union_component_id,
        }
        for item in panel.examples
    ]
    examples_payload = _jsonl_bytes(example_rows)
    canonical_targets_by_fold = {
        str(fold): dict(
            sorted(
                Counter(
                    item.canonical_target for item in panel.examples if item.fold == fold
                ).items()
            )
        )
        for fold in range(config.folds)
    }
    folds_document = {
        "schema_version": 1,
        "artifact": "gate1_context_union_fold_reuse",
        "assignment_policy": _FOLD_POLICY,
        "identity_threshold": config.identity_threshold,
        "maximum_cross_fold_identity": maximum_cross_fold_identity,
        "folds": fold_summary,
        "canonical_targets_by_fold": canonical_targets_by_fold,
        "assignments": [
            {
                "example_id": item.example_id,
                "sequence_id": item.sequence_id,
                "homology_component_id": item.homology_component_id,
                "union_component_id": item.union_component_id,
                "fold": item.fold,
            }
            for item in panel.examples
        ],
    }
    folds_payload = _pretty_json_bytes(folds_document)
    oof_payload = _oof_csv_bytes(predictions)
    metrics_payload = _pretty_json_bytes(metrics)
    accepted_split = {
        "artifact": split_manifest["artifact"],
        "git_commit": split_receipt["git_commit"],
        "top_manifest_sha256": input_hashes["split_top_manifest"],
        "independent_receipt_sha256": input_hashes["split_independent_receipt"],
    }
    census = {
        "parser_sequences": len(assignments),
        "ledger_rows": cast(int, panel.census["ledger_raw_observations"]),
        "context_examples": len(panel.examples),
        "source_observations": sum(len(item.observation_ids) for item in panel.examples),
        "modeled_sequences": len({item.sequence_id for item in panel.examples}),
        "homology_components": len({item.homology_component_id for item in assignments.values()}),
        "union_components": len(components),
        "positives": sum(item.label for item in panel.examples),
        "negatives": sum(1 - item.label for item in panel.examples),
        "examples_by_fold": [fold_summary[str(fold)]["examples"] for fold in range(config.folds)],
        "positive_examples_by_fold": [
            fold_summary[str(fold)]["positives"] for fold in range(config.folds)
        ],
        "negative_examples_by_fold": [
            fold_summary[str(fold)]["negatives"] for fold in range(config.folds)
        ],
    }
    consumption_receipt = {
        "schema_version": 1,
        "artifact": "gate1_union_accepted_split_consumption_receipt",
        "status": "passed",
        "assignment_policy": _FOLD_POLICY,
        "input_sha256": input_hashes,
        "accepted_split": accepted_split,
        "code_attestation": code_attestation,
        "census": census,
        "identity": {
            "algorithm": _HOMOLOGY_POLICY,
            "threshold": config.identity_threshold,
            "maximum_cross_fold_identity": maximum_cross_fold_identity,
        },
        "invariants": {
            "accepted_independent_verification_passed": True,
            "accepted_sequence_assignments_reused_exactly": True,
            "accepted_split_hash_chain_valid": True,
            "all_contexts_grouped_before_eligibility_filtering": True,
            "component_assignment_join_exact": True,
            "cross_fold_identity_strictly_below_threshold": (
                maximum_cross_fold_identity < config.identity_threshold
            ),
            "fold_label_census_exact": True,
            "model_feature_allowlist_exact": True,
        },
    }
    split_payload = _pretty_json_bytes(consumption_receipt)
    label_summary = {
        "context_examples": len(panel.examples),
        "source_observations": sum(len(item.observation_ids) for item in panel.examples),
        "modeled_sequences": len({item.sequence_id for item in panel.examples}),
        "positive_examples": sum(item.label for item in panel.examples),
        "negative_examples": sum(1 - item.label for item in panel.examples),
        "activity_threshold_um": config.activity_threshold_um,
        "by_fold": fold_summary,
        "canonical_targets_by_fold": canonical_targets_by_fold,
        "context_census": panel.census,
    }
    roles = {
        "context_audit": "all endpoint contexts and pre-filter aggregation decisions",
        "examples": "model-facing retained context examples without study/provenance features",
        "folds": "exact example join to accepted sequence assignments",
        "oof": "three-model context-level out-of-fold predictions",
        "metrics": "context metrics with union-component bootstrap intervals",
        "split_receipt": "path-free accepted split hash/census/invariant receipt",
    }
    partial = {
        "context_audit": context_audit,
        "examples": examples_payload,
        "folds": folds_payload,
        "oof": oof_payload,
        "metrics": metrics_payload,
        "split_receipt": split_payload,
    }
    filenames = {
        "context_audit": "context_audit.jsonl",
        "examples": "examples.jsonl",
        "folds": "folds.json",
        "oof": "oof_predictions.csv",
        "metrics": "metrics.json",
        "split_receipt": "split_receipt.json",
    }
    manifest = {
        "schema_version": 1,
        "artifact": _BENCHMARK,
        "status": _BENCHMARK_STATUS,
        "config_sha256": config.sha256,
        "git_commit": git_commit,
        "input_sha256": input_hashes,
        "accepted_split": accepted_split,
        "code_attestation": code_attestation,
        "policies": {
            "label": _LABEL_POLICY,
            "fold_assignment": _FOLD_POLICY,
            "homology": _HOMOLOGY_POLICY,
            "bootstrap_unit": "union_component_id",
            "ensemble": "untrained_equal_probability_mean",
            "model_features": list(_MODEL_FEATURES),
            "forbidden_model_features": list(_FORBIDDEN_MODEL_FEATURES),
            "context_feature": "canonical_target",
        },
        "models": list(_MODELS),
        "label_summary": label_summary,
        "identity_audit": {
            "threshold": config.identity_threshold,
            "maximum_cross_fold_identity": maximum_cross_fold_identity,
        },
        "runtime": {
            "numpy": np.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "artifacts": {
            name: {
                "filename": filenames[name],
                "sha256": _sha256(partial[name]),
                "role": roles[name],
            }
            for name in roles
        },
    }
    result = {
        "context_audit.jsonl": context_audit,
        "examples.jsonl": examples_payload,
        "folds.json": folds_payload,
        "oof_predictions.csv": oof_payload,
        "metrics.json": metrics_payload,
        "split_receipt.json": split_payload,
        "manifest.json": _pretty_json_bytes(manifest),
    }
    return result


def _scan_forbidden(payloads: Iterable[bytes], prefixes: Sequence[bytes]) -> None:
    absolute = re.compile(rb"/(?:lustre|tmp)/|/home/[^/]+/")
    for payload in payloads:
        _require(
            not any(prefix in payload for prefix in prefixes), "artifact exposes forbidden path"
        )
        _require(absolute.search(payload) is None, "artifact exposes an absolute execution path")


def verify_gate1_union_twins(
    *,
    twin_root: str | Path,
    normalized_twin_root: str | Path,
    endpoint_context_twin_root: str | Path,
    homology_study_split_twin_root: str | Path,
    split_independent_receipt: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, object]:
    """Return a path-free receipt after reconstructing and verifying both twins."""

    twins = _resolved_directory(twin_root, label="Gate-1 twin root")
    normalized_twins = _resolved_directory(normalized_twin_root, label="normalized twin root")
    context_twins = _resolved_directory(
        endpoint_context_twin_root, label="endpoint-context twin root"
    )
    split_twins = _resolved_directory(homology_study_split_twin_root, label="split twin root")
    repository = _resolved_directory(repo_root, label="repository root")
    config_requested = Path(config_path)
    _require_no_symlink(config_requested, label="Gate-1 config", ancestors=True)
    config_file = config_requested.resolve(strict=True)
    _require(config_file.is_relative_to(repository), "Gate-1 config must be inside repository")
    config = _load_config(config_file)
    _require(config.path == config_file, "config resolution changed")
    split_receipt_snapshot = _read_snapshot(
        Path(split_independent_receipt), label="accepted split independent receipt"
    )
    _require(
        split_receipt_snapshot.sha256 == _EXPECTED_SPLIT_RECEIPT_SHA256,
        "accepted split receipt content address changed",
    )
    _verify_repository(repository, expected_git_commit)
    _verify_execution_source(repository, expected_git_commit)

    immediate = {path.name: path for path in twins.iterdir()}
    _require(set(immediate) == {"0", "1", "node-receipts"}, "Gate-1 job-root inventory changed")
    _require(
        all(path.is_dir() and not path.is_symlink() for path in immediate.values()),
        "Gate-1 job-root entries must be real directories",
    )
    runs = [twins / "0", twins / "1"]
    normalized_runs = [normalized_twins / "0", normalized_twins / "1"]
    context_runs = [context_twins / "0", context_twins / "1"]
    split_runs = [split_twins / "0", split_twins / "1"]
    for label, paths in (
        ("Gate-1", runs),
        ("normalized", normalized_runs),
        ("endpoint-context", context_runs),
        ("split", split_runs),
    ):
        for path in paths:
            _require(path.is_dir() and not path.is_symlink(), f"{label} twin directory missing")

    handshake = _verify_handshake(twins)
    _verify_tree_bytes(runs[0], runs[1], expected=_PUBLICATION_FILES)
    _verify_tree_bytes(normalized_runs[0], normalized_runs[1])
    _verify_tree_bytes(context_runs[0], context_runs[1])
    _verify_tree_bytes(split_runs[0], split_runs[1])
    for index in (0, 1):
        _verify_manifest_tree(
            normalized_runs[index],
            expected_top_sha256=_EXPECTED_NORMALIZED_TOP_SHA256,
            expected_inventory=None,
            require_sorted=False,
            label=f"normalized input twin {index}",
        )
        _verify_manifest_tree(
            context_runs[index],
            expected_top_sha256=_EXPECTED_CONTEXT_TOP_SHA256,
            expected_inventory=None,
            label=f"endpoint-context input twin {index}",
        )
        _verify_manifest_tree(
            split_runs[index],
            expected_top_sha256=_EXPECTED_SPLIT_TOP_SHA256,
            expected_inventory=None,
            label=f"split input twin {index}",
        )

    publication_records = [
        _verify_publication_run(
            runs[index],
            normalized_run=normalized_runs[index],
            context_run=context_runs[index],
            split_run=split_runs[index],
            split_receipt=split_receipt_snapshot,
            repository=repository,
            expected_commit=expected_git_commit,
            config=config,
        )
        for index in (0, 1)
    ]
    _require(publication_records[0] == publication_records[1], "twin attestations differ")
    publication_top, gate1_top, code_manifest_sha256, code_attestation = publication_records[0]

    normalized = normalized_runs[0]
    context = context_runs[0]
    split = split_runs[0]
    sequence_snapshot = _read_snapshot(normalized / "normalized/sequences.jsonl", label="sequences")
    ledger_snapshot = _read_snapshot(
        context / "endpoint_context/endpoint_context_ledger.jsonl", label="endpoint ledger"
    )
    assignment_snapshot = _read_snapshot(
        split / "split/sequence_assignments.jsonl", label="split assignments"
    )
    component_snapshot = _read_snapshot(split / "split/components.jsonl", label="split components")
    split_manifest_snapshot = _read_snapshot(split / "split/manifest.json", label="split manifest")
    split_audit_snapshot = _read_snapshot(split / "split/audit.json", label="split audit")
    split_top_snapshot = _read_snapshot(split / "SHA256SUMS", label="split top manifest")
    split_manifest, split_audit, upstream_receipt = _verify_split_chain(
        config=config,
        sequences=sequence_snapshot,
        ledger=ledger_snapshot,
        assignments=assignment_snapshot,
        components=component_snapshot,
        split_manifest=split_manifest_snapshot,
        split_audit=split_audit_snapshot,
        split_top=split_top_snapshot,
        split_receipt=split_receipt_snapshot,
    )
    sequences = _read_sequences(sequence_snapshot, config=config)
    assignments = _read_assignments(assignment_snapshot, config=config, sequences=sequences)
    components = _read_components(component_snapshot, config=config, assignments=assignments)
    _, contexts = _group_ledger(ledger_snapshot, config=config, sequences=sequences)
    panel, fold_summary = _reconstruct_panel(
        contexts,
        config=config,
        sequences=sequences,
        assignments=assignments,
        components=components,
        split_audit=split_audit,
    )
    maximum, maximum_by_sequence = _cross_fold_identities(
        sequences,
        assignments,
        (item.sequence_id for item in panel.examples),
        config.identity_threshold,
    )
    predictions = _make_predictions(
        panel.examples,
        config=config,
        maximum_by_sequence=maximum_by_sequence,
    )
    metrics = _summarize_predictions(predictions, config=config)
    input_hashes = {
        "config": config.sha256,
        "endpoint_context_ledger": ledger_snapshot.sha256,
        "parser_v7_sequences": sequence_snapshot.sha256,
        "split_assignments": assignment_snapshot.sha256,
        "split_audit": split_audit_snapshot.sha256,
        "split_components": component_snapshot.sha256,
        "split_independent_receipt": split_receipt_snapshot.sha256,
        "split_manifest": split_manifest_snapshot.sha256,
        "split_top_manifest": split_top_snapshot.sha256,
        "code_manifest": code_manifest_sha256,
    }
    expected_artifacts = _expected_artifacts(
        panel=panel,
        fold_summary=fold_summary,
        predictions=predictions,
        metrics=metrics,
        maximum_cross_fold_identity=maximum,
        config=config,
        git_commit=expected_git_commit,
        code_attestation=code_attestation,
        input_hashes=input_hashes,
        split_manifest=split_manifest,
        split_receipt=upstream_receipt,
        assignments=assignments,
        components=components,
    )
    output_snapshots: dict[str, Snapshot] = {}
    for filename, expected_payload in expected_artifacts.items():
        snapshot = _read_snapshot(runs[0] / "gate1" / filename, label=f"Gate-1 {filename}")
        _require_exact_payload(snapshot.payload, expected_payload, label=filename)
        output_snapshots[filename] = snapshot
    expected_gate1_top = _sha_manifest_bytes(
        {filename: _sha256(payload) for filename, payload in expected_artifacts.items()}
    )
    actual_gate1_top = _read_snapshot(runs[0] / "gate1/SHA256SUMS", label="Gate-1 top manifest")
    _require_exact_payload(
        actual_gate1_top.payload, expected_gate1_top, label="Gate-1 top manifest"
    )
    _require(actual_gate1_top.sha256 == gate1_top, "Gate-1 top hash changed")

    forbidden = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(isinstance(prefix, str) and bool(prefix), "forbidden prefix is empty")
        forbidden.append(prefix.encode("utf-8"))
    all_publication_payloads = [
        _read_snapshot(runs[0] / name, label=f"path scan {name}").payload
        for name in sorted(_PUBLICATION_FILES)
    ]
    _scan_forbidden(all_publication_payloads, forbidden)

    for snapshot in (
        sequence_snapshot,
        ledger_snapshot,
        assignment_snapshot,
        component_snapshot,
        split_manifest_snapshot,
        split_audit_snapshot,
        split_top_snapshot,
        split_receipt_snapshot,
        actual_gate1_top,
        *output_snapshots.values(),
    ):
        _assert_unchanged(snapshot, label=snapshot.path.name)
    _verify_repository(repository, expected_git_commit)
    _verify_execution_source(repository, expected_git_commit)
    _verify_tree_bytes(runs[0], runs[1], expected=_PUBLICATION_FILES)

    artifact_hashes = {
        filename: _sha256(payload) for filename, payload in sorted(expected_artifacts.items())
    }
    overall_metrics = {
        model: cast(dict[str, object], metrics[model])["overall"] for model in _MODELS
    }
    bootstrap = {
        model: cast(dict[str, object], metrics[model])["union_component_bootstrap_95ci"]
        for model in _MODELS
    }
    frozen_sha256 = _read_snapshot(
        runs[0] / "FROZEN_INPUT_SHA256SUMS", label="final frozen-input manifest"
    ).sha256
    return {
        "schema_version": 1,
        "artifact": "gate1_context_activity_homology_study_union_v1_independent_verification",
        "status": "passed",
        "checks": {name: True for name in sorted(_VERIFICATION_CHECKS)},
        "git_commit": expected_git_commit,
        "config_sha256": config.sha256,
        "publication_top_manifest_sha256": publication_top,
        "gate1_top_manifest_sha256": gate1_top,
        "code_manifest_sha256": code_manifest_sha256,
        "frozen_input_manifest_sha256": frozen_sha256,
        "input_sha256": {
            "normalized_top_manifest": _EXPECTED_NORMALIZED_TOP_SHA256,
            "endpoint_context_top_manifest": _EXPECTED_CONTEXT_TOP_SHA256,
            "split_top_manifest": _EXPECTED_SPLIT_TOP_SHA256,
            "split_independent_receipt": _EXPECTED_SPLIT_RECEIPT_SHA256,
            "sequences": sequence_snapshot.sha256,
            "endpoint_context_ledger": ledger_snapshot.sha256,
            "split_assignments": assignment_snapshot.sha256,
            "split_components": component_snapshot.sha256,
        },
        "artifact_sha256": artifact_hashes,
        "census": {
            "ledger_rows": panel.census["ledger_raw_observations"],
            "assay_contexts": panel.census["ledger_assay_contexts"],
            "context_examples": len(panel.examples),
            "source_observations": sum(len(item.observation_ids) for item in panel.examples),
            "modeled_sequences": len({item.sequence_id for item in panel.examples}),
            "positives": sum(item.label for item in panel.examples),
            "negatives": sum(1 - item.label for item in panel.examples),
            "examples_by_fold": list(config.examples_by_fold),
            "positive_examples_by_fold": list(config.positives_by_fold),
            "negative_examples_by_fold": list(config.negatives_by_fold),
        },
        "identity": {
            "algorithm": _HOMOLOGY_POLICY,
            "threshold": config.identity_threshold,
            "maximum_cross_fold_identity": maximum,
        },
        "overall_metrics": overall_metrics,
        "union_component_bootstrap_95ci": bootstrap,
        "production_handshake": handshake,
    }


def _write_receipt(
    path: Path,
    receipt: Mapping[str, object],
    *,
    protected_roots: Sequence[Path],
) -> None:
    requested = Path(path)
    _require(requested.name not in {"", ".", ".."}, "verification receipt has no filename")
    _require_no_symlink(requested.parent, label="receipt parent", ancestors=True)
    _require(not os.path.lexists(requested), f"refusing to overwrite receipt: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    _require(not os.path.lexists(target), f"refusing to overwrite receipt: {target}")
    for root in protected_roots:
        _require(
            target != root and not target.is_relative_to(root),
            "receipt must be outside verified trees",
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
    parser.add_argument("--twin-root", type=Path, required=True)
    parser.add_argument("--normalized-twin-root", type=Path, required=True)
    parser.add_argument("--endpoint-context-twin-root", type=Path, required=True)
    parser.add_argument("--homology-study-split-twin-root", type=Path, required=True)
    parser.add_argument("--split-independent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--forbidden-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protected = [
        _resolved_directory(args.twin_root, label="Gate-1 twin root"),
        _resolved_directory(args.normalized_twin_root, label="normalized twin root"),
        _resolved_directory(args.endpoint_context_twin_root, label="endpoint-context twin root"),
        _resolved_directory(args.homology_study_split_twin_root, label="split twin root"),
        _resolved_directory(args.repo_root, label="repository root"),
    ]
    output = Path(args.output)
    prospective = output.resolve(strict=False)
    for root in protected:
        _require(
            prospective != root and not prospective.is_relative_to(root),
            "receipt must be outside verified trees",
        )
    receipt = verify_gate1_union_twins(
        twin_root=args.twin_root,
        normalized_twin_root=args.normalized_twin_root,
        endpoint_context_twin_root=args.endpoint_context_twin_root,
        homology_study_split_twin_root=args.homology_study_split_twin_root,
        split_independent_receipt=args.split_independent_receipt,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    _write_receipt(output, receipt, protected_roots=protected)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the Slurm CLI
    raise SystemExit(main())

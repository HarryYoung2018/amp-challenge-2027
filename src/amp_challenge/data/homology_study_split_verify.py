"""Independently verify homology-and-study split production twins.

This module deliberately does not import the split builder, sequence helpers,
or similarity helpers.  It reparses the frozen inputs, recomputes global
alignments and both graph layers, replays the six-dimensional fold assignment,
and checks every emitted byte-addressed artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_PYTHON_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_ENDPOINTS = frozenset({"mic", "hc50", "hemolysis_percent"})

_HOMOLOGY_ALGORITHM = "global_alignment_identity_single_link_v1"
_COMPONENT_POLICY = "full_sequence_homology_union_every_study_key_v1"
_BALANCE_POLICY = "seeded_largest_normalized_load_first_global_squared_error_v1"
_OUTPUT_STATUS = "development_split_not_an_untouched_evaluation_panel"
_SIDECAR_STATUS = "development_only_pending_reviewed_taxonomy_and_study_component_split"
_BALANCE_METRICS = (
    "component_count",
    "sequences",
    "gram_negative_mic16_negative",
    "gram_negative_mic16_positive",
    "gram_positive_mic16_negative",
    "gram_positive_mic16_positive",
)
_CLASS_METRICS = _BALANCE_METRICS[2:]

_EXPECTED_SEQUENCE_SHA256 = "76305e284ce3689970c89e3f2b45c4adc0ceef2375e74dfcddc50882d8b14884"
_EXPECTED_LEDGER_SHA256 = "acd2ca8bfdf23588532463b399ffcf4a8068986aa28d75170cbf71b93c18f9f6"
_EXPECTED_STUDY_SHA256 = "9bcf564f95bad6f4ddd49ddfcdf6c72f552fbd662c31693502b33e25b5c4ffaf"
_EXPECTED_CONTEXT_MANIFEST_SHA256 = (
    "7d671640b24b39e5ed5fbe804f5d120c55653f869450c7bbd71fb6cd991637a9"
)
_EXPECTED_CONTEXT_TOP_SHA256 = "3a3722e685470a37ec9e5fb31aa2652cae9742966a3be3d29133692737054ae8"
_EXPECTED_NORMALIZED_TOP_SHA256 = "c83e446f89bbf48a4c3c2fea9a397328147267b231cafad6f913badb7c6c3379"

_MIXED_CONTEXT_IDS = frozenset(
    {
        "36601c8095df6398b58357e77363b2cd75d41ec14b78fbba429e18047b87b2a2",
        "e60a99b390dcd9b06eaf6453a92538b808e6f46a92ce8e548a10cb347683d176",
    }
)
_CONFLICT_CONTEXT_IDS = frozenset(
    {
        "28066cb89f97a09d76d6d48110de575c16ba75c38322a8af23fc48a24a9b9f7f",
        "3446209b01f8170912a25d400901660c74815408f32066fb5b3ee941058015d0",
        "34cf3c0f54c19ff62c388a1812ab9ffc51c2c2eb4c74fa0233ad77ee1d299575",
        "4a13c23ded33f89539b87d527382dd38001ca72221eac6d824b0d590fb17f0a5",
        "e9563b7249351d0bd8d68ea86de52c4058baf7a92a185c3f60f6ad95ca35da95",
    }
)

_TOP_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "split/audit.json",
        "split/components.jsonl",
        "split/grouping_edges.jsonl",
        "split/manifest.json",
        "split/sequence_assignments.jsonl",
    }
)
_TOP_MANIFEST_ENTRIES = _TOP_FILES - {"SHA256SUMS"}
_SPLIT_FILES = frozenset(
    {
        "audit.json",
        "components.jsonl",
        "grouping_edges.jsonl",
        "manifest.json",
        "sequence_assignments.jsonl",
    }
)
_REQUIRED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/build_homology_study_split_v1_twins.sbatch",
        "cluster/validate_homology_study_split_output.sh",
        "configs/data/homology_study_split_v1.toml",
        "pyproject.toml",
        "uv.lock",
    }
)
_SIDECAR_TOP_ENTRIES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "endpoint_context/audit.json",
        "endpoint_context/contexts.jsonl",
        "endpoint_context/endpoint_context_ledger.jsonl",
        "endpoint_context/manifest.json",
        "endpoint_context/study_membership.jsonl",
    }
)
_SIDECAR_MANIFEST_FIELDS = frozenset(
    {
        "artifact",
        "artifacts",
        "config_sha256",
        "counts",
        "input",
        "policies",
        "provenance",
        "schema_version",
        "status",
    }
)
_SIDECAR_POLICY_FIELDS = frozenset(
    {
        "blood_organism_mapping",
        "blood_organism_mapping_version",
        "mapping_version",
        "mic16",
        "strain_identifier_reviews",
        "study_anomalies",
        "study_keys",
        "target_mapping",
    }
)

_SEQUENCE_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_STUDY_FIELDS = frozenset(
    {
        "citation_reference",
        "citation_title",
        "ignored_pubmed_tokens",
        "pmids",
        "provenance_id",
        "schema_version",
        "sequence_id",
        "source",
        "source_record_id",
        "source_record_key",
        "source_row_number",
        "source_sha256",
        "source_version",
        "study_keys",
        "study_review_codes",
        "study_status",
    }
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
_MEASUREMENT_FIELDS = frozenset(
    {
        "lower",
        "lower_inclusive",
        "raw_value",
        "relation",
        "source_unit",
        "unit",
        "upper",
        "upper_inclusive",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {"schema_version", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)
_COMPONENT_FIELDS = frozenset(
    {
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
)
_EDGE_FIELDS = frozenset(
    {
        "schema_version",
        "edge_id",
        "edge_type",
        "group_key",
        "sequence_ids",
        "sequence_count",
        "source_membership_count",
    }
)


class VerificationError(ValueError):
    """Raised when a frozen contract does not verify."""


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    path: Path
    sha256: str
    threshold: float
    folds: int
    seed: int
    weights: dict[str, float]
    minima: dict[str, int]
    expected_sequences: int
    expected_assays: int
    expected_memberships: int


@dataclass(frozen=True, slots=True)
class JsonlDocument:
    payload: bytes
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class GraphResult:
    edges: tuple[dict[str, Any], ...]
    components: tuple[dict[str, Any], ...]
    homology_by_sequence: dict[str, str]
    component_by_sequence: dict[str, str]
    component_members: dict[str, tuple[str, ...]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(result: os.stat_result) -> tuple[int, int, int, int]:
    return (result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns)


def _read_regular_file(path: Path, *, label: str) -> bytes:
    _require(not path.is_symlink(), f"{label} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise VerificationError(f"{label} is missing") from error
    before = resolved.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return payload


def _require_no_lexical_symlink(path: Path, *, label: str, ancestors: bool = False) -> None:
    candidates = (path, *path.parents) if ancestors else (path,)
    for candidate in candidates:
        if os.path.lexists(candidate):
            _require(not candidate.is_symlink(), f"{label} must not traverse a symbolic link")


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _require_no_lexical_symlink(requested, label=label, ancestors=True)
    resolved = requested.resolve(strict=True)
    _require(resolved.is_dir(), f"{label} is not a directory")
    return resolved


def _git(repo_root: Path, arguments: Sequence[str], *, label: str) -> bytes:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *arguments],
            capture_output=True,
            check=False,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VerificationError(f"{label} could not execute Git") from error
    _require(result.returncode == 0, f"{label} failed")
    return result.stdout


def _verify_repository_state(repo_root: Path, expected_commit: str) -> None:
    _require(_GIT_RE.fullmatch(expected_commit) is not None, "expected Git commit is invalid")
    top_level = _git(repo_root, ["rev-parse", "--show-toplevel"], label="repository root")
    try:
        declared_root = Path(top_level.decode("utf-8").strip()).resolve(strict=True)
    except (UnicodeDecodeError, FileNotFoundError) as error:
        raise VerificationError("repository root reported by Git is invalid") from error
    _require(declared_root == repo_root, "supplied repository is not the Git top level")
    _git(
        repo_root,
        ["cat-file", "-e", f"{expected_commit}^{{commit}}"],
        label="expected Git commit",
    )
    head = _git(
        repo_root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        label="repository HEAD",
    )
    _require(
        head == f"{expected_commit}\n".encode("ascii"), "repository HEAD differs from expected"
    )
    _git(
        repo_root,
        ["diff", "--no-ext-diff", "--quiet", "--exit-code", "--"],
        label="tracked working tree cleanliness",
    )
    _git(
        repo_root,
        ["diff", "--cached", "--no-ext-diff", "--quiet", "--exit-code", "--"],
        label="Git index cleanliness",
    )
    untracked = _git(
        repo_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        label="untracked-file inventory",
    )
    _require(not untracked, "repository contains untracked files")


def _committed_blob(repo_root: Path, expected_commit: str, relative: str) -> bytes:
    specification = f"{expected_commit}:{relative}"
    object_type = _git(
        repo_root,
        ["cat-file", "-t", specification],
        label=f"committed code object {relative}",
    )
    _require(object_type == b"blob\n", f"committed code path is not a blob: {relative}")
    return _git(
        repo_root,
        ["cat-file", "blob", specification],
        label=f"committed code blob {relative}",
    )


def _verify_execution_path(repo_root: Path) -> Path:
    expected = repo_root / "src/amp_challenge/data/homology_study_split_verify.py"
    executing = Path(__file__)
    _require(executing.is_absolute(), "executing verifier path must be absolute")
    _require_no_lexical_symlink(expected, label="repository verifier", ancestors=True)
    _require_no_lexical_symlink(executing, label="executing verifier", ancestors=True)
    expected_resolved = expected.resolve(strict=True)
    executing_resolved = executing.resolve(strict=True)
    _require(
        expected_resolved == executing_resolved,
        "executing verifier is not the repository verifier (stale installation rejected)",
    )
    _read_regular_file(expected_resolved, label="executing repository verifier")
    return expected_resolved


def _reject_constant(value: str) -> None:
    raise VerificationError(f"JSON contains forbidden constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _loads_json(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise VerificationError(f"{label} is not valid JSON") from error


def _canonical_compact(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _read_regular_file(path, label=label)
    _require(payload.endswith(b"\n") and not payload.endswith(b"\n\n"), f"{label} LF framing")
    _require(b"\r" not in payload, f"{label} contains CR bytes")
    value = _loads_json(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain one object")
    expected = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _require(payload == expected, f"{label} is not canonical pretty JSON")
    return value


def _read_jsonl(
    path: Path,
    *,
    label: str,
    canonical: bool = True,
    accepted_noncanonical_sha256: str | None = None,
) -> JsonlDocument:
    payload = _read_regular_file(path, label=label)
    if canonical:
        _require(
            accepted_noncanonical_sha256 is None,
            f"{label} supplied a noncanonical-input pin in strict mode",
        )
    else:
        _require(
            accepted_noncanonical_sha256 == _EXPECTED_SEQUENCE_SHA256,
            "noncanonical JSONL is restricted to the hash-pinned parser-v7 sequence table",
        )
        _require(
            _sha256_bytes(payload) == accepted_noncanonical_sha256,
            "noncanonical parser-v7 sequence table hash mismatch",
        )
    _require(payload and payload.endswith(b"\n"), f"{label} needs one final LF")
    _require(b"\r" not in payload, f"{label} contains CR bytes")
    raw_lines = payload[:-1].split(b"\n")
    _require(all(raw_lines), f"{label} contains a blank row")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(raw_lines, start=1):
        value = _loads_json(raw, label=f"{label} row {number}")
        _require(isinstance(value, dict), f"{label} row {number} must be an object")
        if canonical:
            _require(
                raw == _canonical_compact(value),
                f"{label} row {number} is not canonical compact JSON",
            )
        rows.append(value)
    return JsonlDocument(payload=payload, rows=tuple(rows))


def _safe_manifest_name(raw: str, *, label: str) -> str:
    pure = PurePosixPath(raw)
    _require(
        raw == pure.as_posix()
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "\\" not in raw,
        f"{label} has unsafe path {raw!r}",
    )
    return raw


def _parse_checksum_manifest(
    path: Path, *, label: str, require_sorted: bool = True
) -> dict[str, str]:
    payload = _read_regular_file(path, label=label)
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} LF framing")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    entries: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"{label} row {number} has invalid syntax")
        assert match is not None
        digest, raw_name = match.groups()
        name = _safe_manifest_name(raw_name, label=label)
        _require(name not in entries, f"{label} repeats {name}")
        if require_sorted:
            _require(previous is None or name > previous, f"{label} paths are not strictly sorted")
        entries[name] = digest
        previous = name
    return entries


def _tree_inventory(root: Path) -> frozenset[str]:
    _require(root.is_dir() and not root.is_symlink(), f"tree is missing or symbolic: {root}")
    files: set[str] = set()
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            _require(
                not (current_path / directory).is_symlink(), "tree contains a directory symlink"
            )
        for name in names:
            path = current_path / name
            _require(path.is_file() and not path.is_symlink(), "tree contains a non-regular file")
            files.add(path.relative_to(root).as_posix())
    return frozenset(files)


def _verify_manifest_files(root: Path, entries: Mapping[str, str], *, label: str) -> None:
    for relative, expected in entries.items():
        payload = _read_regular_file(root / relative, label=f"{label} {relative}")
        _require(_sha256_bytes(payload) == expected, f"{label} checksum mismatch for {relative}")


def _verify_tree_top_manifest(
    root: Path,
    *,
    expected_sha256: str,
    label: str,
    require_sorted: bool = True,
) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    payload = _read_regular_file(manifest, label=f"{label} top manifest")
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} top manifest hash mismatch")
    if not require_sorted:
        _require(
            expected_sha256 == _EXPECTED_NORMALIZED_TOP_SHA256,
            "relaxed manifest ordering is restricted to the hash-pinned parser-v7 input",
        )
    entries = _parse_checksum_manifest(
        manifest,
        label=f"{label} top manifest",
        require_sorted=require_sorted,
    )
    _require(
        set(entries) == set(_tree_inventory(root)) - {"SHA256SUMS"}, f"{label} inventory mismatch"
    )
    _verify_manifest_files(root, entries, label=label)
    return entries


def _exact_fields(row: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    _require(
        set(row) == expected,
        f"{label} schema mismatch: missing={sorted(expected - set(row))}, extra={sorted(set(row) - expected)}",
    )


def _sha(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"{label} SHA-256")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} positive integer",
    )
    return value


def _string_list(
    value: Any, *, label: str, empty: bool, sorted_values: bool = False
) -> tuple[str, ...]:
    _require(isinstance(value, list) and (empty or value), f"{label} must be a string list")
    _require(
        all(isinstance(item, str) and item and item == item.strip() for item in value),
        f"{label} invalid string",
    )
    _require(len(value) == len(set(value)), f"{label} repeats values")
    if sorted_values:
        _require(value == sorted(value), f"{label} must be sorted")
    return tuple(value)


def _typed_frame(value: Any) -> Any:
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        _require(math.isfinite(value), "typed digest float is not finite")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, tuple):
        return ["tuple", [_typed_frame(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_typed_frame(item) for item in value]]
    if isinstance(value, Mapping):
        _require(all(isinstance(key, str) for key in value), "typed mapping key is not a string")
        return ["mapping", [[key, _typed_frame(value[key])] for key in sorted(value)]]
    raise VerificationError(f"unsupported typed digest value {type(value).__name__}")


def _digest(namespace: str, *parts: Any) -> str:
    _require(isinstance(namespace, str) and namespace, "digest namespace is empty")
    return _sha256_bytes(
        _canonical_compact(
            {"namespace": _typed_frame(namespace), "parts": [_typed_frame(part) for part in parts]}
        )
    )


def _load_config(path: Path) -> FrozenConfig:
    payload = _read_regular_file(path, label="split config")
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        "split config must use canonical LF framing",
    )
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("split config is invalid UTF-8 TOML") from error
    fields = frozenset(
        {
            "schema_version",
            "identity_threshold",
            "folds",
            "seed",
            "normalized_sequences_sha256",
            "endpoint_context_ledger_sha256",
            "study_membership_sha256",
            "endpoint_context_manifest_sha256",
            "endpoint_context_top_manifest_sha256",
            "expected_unique_sequences",
            "expected_assay_observations",
            "expected_study_memberships",
            "balance_weight",
            "balance_minimum",
        }
    )
    _exact_fields(raw, fields, label="split config")
    _require(
        raw["schema_version"] == 1 and not isinstance(raw["schema_version"], bool),
        "config schema version",
    )
    expected_hashes = {
        "normalized_sequences_sha256": _EXPECTED_SEQUENCE_SHA256,
        "endpoint_context_ledger_sha256": _EXPECTED_LEDGER_SHA256,
        "study_membership_sha256": _EXPECTED_STUDY_SHA256,
        "endpoint_context_manifest_sha256": _EXPECTED_CONTEXT_MANIFEST_SHA256,
        "endpoint_context_top_manifest_sha256": _EXPECTED_CONTEXT_TOP_SHA256,
    }
    _require(
        all(raw[key] == value for key, value in expected_hashes.items()),
        "config frozen input hashes changed",
    )
    _require(
        raw["identity_threshold"] == 0.8 and not isinstance(raw["identity_threshold"], bool),
        "config threshold changed",
    )
    _require(
        isinstance(raw["folds"], int)
        and not isinstance(raw["folds"], bool)
        and raw["folds"] == 5
        and isinstance(raw["seed"], int)
        and not isinstance(raw["seed"], bool)
        and raw["seed"] == 42,
        "config fold/seed changed",
    )
    _require(
        all(
            isinstance(raw[key], int) and not isinstance(raw[key], bool)
            for key in (
                "expected_unique_sequences",
                "expected_assay_observations",
                "expected_study_memberships",
            )
        )
        and raw["expected_unique_sequences"] == 1113
        and raw["expected_assay_observations"] == 6079
        and raw["expected_study_memberships"] == 1136,
        "config frozen counts changed",
    )
    weights = raw["balance_weight"]
    minima = raw["balance_minimum"]
    _require(
        isinstance(weights, dict) and set(weights) == set(_BALANCE_METRICS), "config weight schema"
    )
    _require(
        isinstance(minima, dict) and set(minima) == set(_CLASS_METRICS), "config minima schema"
    )
    parsed_weights: dict[str, float] = {}
    for metric in _BALANCE_METRICS:
        value = weights[metric]
        _require(
            isinstance(value, int | float) and not isinstance(value, bool), "config weight type"
        )
        parsed_weights[metric] = float(value)
    _require(
        all(math.isfinite(value) and value == 1.0 for value in parsed_weights.values()),
        "config weights changed",
    )
    _require(
        all(
            isinstance(minima[metric], int)
            and not isinstance(minima[metric], bool)
            and minima[metric] == 1
            for metric in _CLASS_METRICS
        ),
        "config minima changed",
    )
    return FrozenConfig(
        path=path.resolve(strict=True),
        sha256=_sha256_bytes(payload),
        threshold=0.8,
        folds=5,
        seed=42,
        weights=parsed_weights,
        minima={metric: 1 for metric in _CLASS_METRICS},
        expected_sequences=1113,
        expected_assays=6079,
        expected_memberships=1136,
    )


def _canonical_sequence(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a sequence string")
    _require(value == value.strip().upper(), f"{label} is not canonical uppercase")
    _require(set(value) <= _AMINO_ACIDS, f"{label} contains a non-standard residue")
    return value


def _sequence_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


@lru_cache(maxsize=1_000_000)
def _global_identity(left: str, right: str) -> float:
    """Independent +1/-1/-1 global alignment with deterministic tie semantics."""

    if right < left:
        left, right = right, left
    previous = [(0, 0, 0)]
    previous.extend((-column, 0, -column) for column in range(1, len(right) + 1))
    for row_index, residue in enumerate(left, start=1):
        current = [(-row_index, 0, -row_index)]
        for column_index, other in enumerate(right, start=1):
            prior = previous[column_index - 1]
            match = int(residue == other)
            candidates = (
                (prior[0] + (1 if match else -1), prior[1] + match, prior[2] - 1),
                (
                    previous[column_index][0] - 1,
                    previous[column_index][1],
                    previous[column_index][2] - 1,
                ),
                (current[-1][0] - 1, current[-1][1], current[-1][2] - 1),
            )
            best = candidates[0]
            for candidate in candidates[1:]:
                if candidate > best:
                    best = candidate
            current.append(best)
        previous = current
    _, matches, negative_length = previous[-1]
    return matches / -negative_length


class _Dsu:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def join(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if b < a:
            a, b = b, a
        self.parent[b] = a


def _homology_partition(
    sequences: Mapping[str, str], threshold: float
) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(sequences)
    groups = _Dsu(ordered)
    for left_index, left_id in enumerate(ordered):
        left = sequences[left_id]
        for right_id in ordered[left_index + 1 :]:
            right = sequences[right_id]
            if min(len(left), len(right)) / max(len(left), len(right)) < threshold:
                continue
            if _global_identity(left, right) >= threshold:
                groups.join(left_id, right_id)
    members: dict[str, list[str]] = defaultdict(list)
    for sequence_id in ordered:
        members[groups.find(sequence_id)].append(sequence_id)
    return tuple(sorted((tuple(value) for value in members.values()), key=lambda value: value[0]))


def _load_sequences(path: Path, config: FrozenConfig) -> dict[str, str]:
    document = _read_jsonl(
        path,
        label="normalized sequences",
        canonical=False,
        accepted_noncanonical_sha256=_EXPECTED_SEQUENCE_SHA256,
    )
    _require(
        _sha256_bytes(document.payload) == _EXPECTED_SEQUENCE_SHA256, "sequence file hash mismatch"
    )
    _require(len(document.rows) == config.expected_sequences, "sequence count mismatch")
    sequences: dict[str, str] = {}
    for number, row in enumerate(document.rows, start=1):
        _exact_fields(row, _SEQUENCE_FIELDS, label=f"sequence row {number}")
        sequence = _canonical_sequence(row["sequence"], label=f"sequence row {number}")
        sequence_id = _sha(row["sequence_id"], label=f"sequence row {number}")
        _require(sequence_id == _sequence_id(sequence), f"sequence row {number} ID mismatch")
        _require(sequence_id not in sequences, f"sequence row {number} repeats ID")
        _require(
            isinstance(row["provenance"], list) and row["provenance"], "sequence provenance empty"
        )
        sequences[sequence_id] = sequence
    return sequences


def _load_studies(
    path: Path, sequences: Mapping[str, str], config: FrozenConfig
) -> tuple[dict[str, set[str]], Counter[str], dict[str, dict[str, Any]]]:
    document = _read_jsonl(path, label="study memberships")
    _require(_sha256_bytes(document.payload) == _EXPECTED_STUDY_SHA256, "study file hash mismatch")
    _require(len(document.rows) == config.expected_memberships, "study membership count mismatch")
    key_sequences: dict[str, set[str]] = defaultdict(set)
    key_memberships: Counter[str] = Counter()
    by_provenance: dict[str, dict[str, Any]] = {}
    covered: set[str] = set()
    statuses = {
        "explicit_pmid",
        "explicit_pmid_with_ignored_tokens",
        "reference_title_fallback",
        "source_record_singleton",
    }
    for number, row in enumerate(document.rows, start=1):
        label = f"study membership row {number}"
        _exact_fields(row, _STUDY_FIELDS, label=label)
        _require(
            row["schema_version"] == 1 and not isinstance(row["schema_version"], bool),
            f"{label} version",
        )
        sid = _sha(row["sequence_id"], label=f"{label} sequence_id")
        _require(sid in sequences, f"{label} unknown sequence")
        provenance_id = _sha(row["provenance_id"], label=f"{label} provenance_id")
        _require(provenance_id not in by_provenance, f"{label} repeats provenance")
        source_record_key = row["source_record_key"]
        _require(
            source_record_key == f"source-record:{provenance_id}", f"{label} source key mismatch"
        )
        for field in ("source", "source_version", "source_record_id"):
            _require(
                isinstance(row[field], str) and row[field] == row[field].strip() and row[field],
                f"{label} {field}",
            )
        _sha(row["source_sha256"], label=f"{label} source SHA")
        _positive_int(row["source_row_number"], label=f"{label} source row")
        status = row["study_status"]
        _require(status in statuses, f"{label} status")
        keys = _string_list(row["study_keys"], label=f"{label} study keys", empty=False)
        pmids = _string_list(row["pmids"], label=f"{label} PMIDs", empty=True)
        ignored = _string_list(
            row["ignored_pubmed_tokens"],
            label=f"{label} ignored PMIDs",
            empty=True,
            sorted_values=True,
        )
        reviews = _string_list(
            row["study_review_codes"], label=f"{label} reviews", empty=True, sorted_values=True
        )
        _require(all(re.fullmatch(r"[1-9][0-9]*", pmid) for pmid in pmids), f"{label} bad PMID")
        _require(tuple(sorted(pmids, key=int)) == pmids, f"{label} PMIDs are not numeric sorted")
        reference, title = row["citation_reference"], row["citation_title"]
        for field_name, value in (("reference", reference), ("title", title)):
            _require(
                value is None or (isinstance(value, str) and value and value == value.strip()),
                f"{label} {field_name}",
            )
        if pmids:
            _require(keys == tuple(f"pmid:{pmid}" for pmid in pmids), f"{label} PMID keys")
            expected_status = "explicit_pmid_with_ignored_tokens" if ignored else "explicit_pmid"
            _require(status == expected_status, f"{label} explicit status")
        elif reference is not None or title is not None:
            expected_key = "reference-title:" + _digest(
                "amp-challenge:reference-title-study:v1", reference or "", title or ""
            )
            _require(
                keys == (expected_key,) and status == "reference_title_fallback",
                f"{label} fallback key",
            )
        else:
            _require(
                keys == (source_record_key,) and status == "source_record_singleton",
                f"{label} singleton key",
            )
        by_provenance[provenance_id] = {
            "sequence_id": sid,
            "source_record_id": row["source_record_id"],
            "source_record_key": source_record_key,
            "source_row_number": row["source_row_number"],
            "study_status": status,
            "study_keys": keys,
            "study_review_codes": reviews,
        }
        covered.add(sid)
        for key in keys:
            key_sequences[key].add(sid)
            key_memberships[key] += 1
    _require(covered == set(sequences), "study memberships do not cover sequences")
    _require(len(key_sequences) == 435, "study-key census mismatch")
    _require(
        sum(len(value) > 1 for value in key_sequences.values()) == 199,
        "shared-study census mismatch",
    )
    _require(
        all(
            len(value) == 1
            for key, value in key_sequences.items()
            if key.startswith("source-record:")
        ),
        "source singleton merged",
    )
    return dict(key_sequences), key_memberships, by_provenance


def _measurement(value: Any, *, label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    _exact_fields(value, _MEASUREMENT_FIELDS, label=label)


def _exposure_identity(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    _require(isinstance(value, dict), "exposure identity must be an object or null")
    return (
        value["relation"],
        value["lower"],
        value["lower_inclusive"],
        value["upper"],
        value["upper_inclusive"],
        value["unit"],
    )


def _summarize_mic_groups(
    mic_groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Counter[str]], dict[str, Any]]:
    by_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    mixed: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    repeated = 0
    all_ineligible_contexts = 0
    all_ineligible_rows = 0
    fully_eligible = 0
    eligible_rows = 0
    retained_contexts = 0
    retained_rows = 0
    retained_sequences: set[str] = set()
    for context_id in sorted(mic_groups):
        members = sorted(mic_groups[context_id], key=lambda row: row["observation_id"])
        _require(bool(members), f"assay context {context_id} is empty")
        repeated += int(len(members) > 1)
        first = members[0]
        semantic = (
            first["sequence_id"],
            first["endpoint"],
            first["context_id"],
            tuple(first["source_conditions"]),
            _exposure_identity(first["exposure_concentration"]),
        )
        _require(
            all(
                (
                    row["sequence_id"],
                    row["endpoint"],
                    row["context_id"],
                    tuple(row["source_conditions"]),
                    _exposure_identity(row["exposure_concentration"]),
                )
                == semantic
                for row in members
            ),
            f"assay context {context_id} has inconsistent identity semantics",
        )
        flags = ["bacterial_mic16" in row["eligible_tasks"] for row in members]
        eligible_count = sum(flags)
        eligible_rows += eligible_count
        if eligible_count == 0:
            all_ineligible_contexts += 1
            all_ineligible_rows += len(members)
            continue
        if eligible_count != len(members):
            mixed.append(
                {
                    "assay_context_id": context_id,
                    "observations": len(members),
                    "eligible_observations": eligible_count,
                    "ineligible_observations": len(members) - eligible_count,
                    "sequence_id": first["sequence_id"],
                }
            )
            continue
        fully_eligible += 1
        labels = {row["mic16_label"] for row in members}
        grams = {row["source_gram"] for row in members}
        reasons: list[str] = []
        if len(labels) != 1:
            reasons.append("mic16_label_disagreement")
        if len(grams) != 1:
            reasons.append("source_gram_disagreement")
        if reasons:
            conflicts.append(
                {
                    "assay_context_id": context_id,
                    "observations": len(members),
                    "sequence_id": first["sequence_id"],
                    "reason_codes": reasons,
                    "mic16_labels": sorted(labels),
                    "source_grams": sorted(grams),
                }
            )
            continue
        retained_contexts += 1
        retained_rows += len(members)
        sequence_id = first["sequence_id"]
        retained_sequences.add(sequence_id)
        gram = next(iter(grams))
        label = next(iter(labels))
        metric = f"gram_{gram}_mic16_{'positive' if label == 1 else 'negative'}"
        by_sequence[sequence_id][metric] += 1
    return dict(by_sequence), {
        "repeated_mic_assay_contexts": repeated,
        "eligible_raw_observations": eligible_rows,
        "candidate_assay_contexts": fully_eligible + len(mixed),
        "all_ineligible_assay_contexts": all_ineligible_contexts,
        "observations_in_all_ineligible_assay_contexts": all_ineligible_rows,
        "fully_eligible_assay_contexts": fully_eligible,
        "mixed_eligibility_assay_contexts": len(mixed),
        "observations_in_mixed_eligibility_assay_contexts": sum(
            row["observations"] for row in mixed
        ),
        "eligible_observations_in_mixed_eligibility_assay_contexts": sum(
            row["eligible_observations"] for row in mixed
        ),
        "excluded_mixed_eligibility": mixed,
        "conflicting_assay_contexts": len(conflicts),
        "eligible_label_conflict_contexts": sum(
            "mic16_label_disagreement" in row["reason_codes"] for row in conflicts
        ),
        "eligible_gram_conflict_contexts": sum(
            "source_gram_disagreement" in row["reason_codes"] for row in conflicts
        ),
        "observations_in_conflicting_assay_contexts": sum(row["observations"] for row in conflicts),
        "excluded_conflicts": conflicts,
        "retained_assay_contexts": retained_contexts,
        "retained_source_observations": retained_rows,
        "retained_sequences": len(retained_sequences),
        "retained_class_counts": {
            metric: sum(counts[metric] for counts in by_sequence.values())
            for metric in _CLASS_METRICS
        },
    }


def _load_balance(
    path: Path,
    sequences: Mapping[str, str],
    membership: Mapping[str, Mapping[str, Any]],
    config: FrozenConfig,
) -> tuple[dict[str, Counter[str]], dict[str, Any]]:
    document = _read_jsonl(path, label="endpoint-context ledger")
    _require(
        _sha256_bytes(document.payload) == _EXPECTED_LEDGER_SHA256, "ledger file hash mismatch"
    )
    _require(len(document.rows) == config.expected_assays, "ledger row count mismatch")
    observations: set[str] = set()
    all_contexts: set[str] = set()
    non_mic_contexts: set[str] = set()
    mic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    non_mic_rows = 0
    for number, row in enumerate(document.rows, start=1):
        label = f"ledger row {number}"
        _exact_fields(row, _LEDGER_FIELDS, label=label)
        _require(
            row["schema_version"] == 1 and not isinstance(row["schema_version"], bool),
            f"{label} version",
        )
        sid = _sha(row["sequence_id"], label=f"{label} sequence")
        _require(sid in sequences, f"{label} unknown sequence")
        for field in (
            "observation_id",
            "assay_row_sha256",
            "assay_context_id",
            "context_id",
            "provenance_id",
        ):
            _sha(row[field], label=f"{label} {field}")
        observation_id = row["observation_id"]
        _require(observation_id not in observations, f"{label} duplicate observation")
        observations.add(observation_id)
        provenance = membership.get(row["provenance_id"])
        _require(provenance is not None, f"{label} unknown provenance")
        linked = {
            "sequence_id": sid,
            "source_record_id": row["source_record_id"],
            "source_record_key": row["source_record_key"],
            "source_row_number": row["source_row_number"],
            "study_status": row["study_status"],
            "study_keys": tuple(row["study_keys"]),
            "study_review_codes": tuple(row["study_review_codes"]),
        }
        _require(linked == provenance, f"{label} differs from study provenance")
        _measurement(row["measurement"], label=f"{label} measurement")
        if row["exposure_concentration"] is not None:
            _measurement(row["exposure_concentration"], label=f"{label} exposure")
        tasks = _string_list(
            row["eligible_tasks"], label=f"{label} tasks", empty=True, sorted_values=True
        )
        for field in (
            "exclusion_codes",
            "study_review_codes",
            "composite_reason_codes",
            "strain_identifier_review_ids",
        ):
            _string_list(row[field], label=f"{label} {field}", empty=True, sorted_values=True)
        _string_list(row["study_keys"], label=f"{label} study keys", empty=False)
        _string_list(row["source_conditions"], label=f"{label} conditions", empty=True)
        _require(row["endpoint"] in _ENDPOINTS, f"{label} endpoint")
        binary = "bacterial_mic16" in tasks
        if binary:
            _require(
                row["endpoint"] == "mic"
                and row["mapping_status"] == "mapped_single_supported_species"
                and row["gram_resolution"] == "concordant"
                and row["source_gram"] in {"negative", "positive"}
                and isinstance(row["mic16_label"], int)
                and not isinstance(row["mic16_label"], bool)
                and row["mic16_label"] in {0, 1}
                and isinstance(row["canonical_target"], str)
                and bool(row["canonical_target"]),
                f"{label} inconsistent bacterial_mic16 flag",
            )
        context_id = row["assay_context_id"]
        all_contexts.add(context_id)
        if row["endpoint"] == "mic":
            mic_groups[context_id].append(row)
        else:
            non_mic_rows += 1
            non_mic_contexts.add(context_id)
    _require(not (set(mic_groups) & non_mic_contexts), "context ID crosses MIC/non-MIC")

    by_sequence, context_counts = _summarize_mic_groups(mic_groups)
    _require(
        {row["assay_context_id"] for row in context_counts["excluded_mixed_eligibility"]}
        == _MIXED_CONTEXT_IDS,
        "mixed-context sentinels changed",
    )
    _require(
        {row["assay_context_id"] for row in context_counts["excluded_conflicts"]}
        == _CONFLICT_CONTEXT_IDS,
        "conflict-context sentinels changed",
    )
    census = {
        "ledger_raw_observations": len(document.rows),
        "ledger_assay_contexts": len(all_contexts),
        "mic_raw_observations": sum(len(value) for value in mic_groups.values()),
        "mic_assay_contexts": len(mic_groups),
        "non_mic_raw_observations": non_mic_rows,
        "non_mic_assay_contexts": len(non_mic_contexts),
        **context_counts,
    }
    _require(
        census["ledger_raw_observations"] == 6079
        and census["ledger_assay_contexts"] == 5859
        and census["mic_raw_observations"] == 5903
        and census["mic_assay_contexts"] == 5691
        and census["non_mic_raw_observations"] == 176
        and census["non_mic_assay_contexts"] == 168,
        "endpoint/context census changed",
    )
    _require(
        census["retained_assay_contexts"] == 2492
        and census["retained_source_observations"] == 2592
        and census["retained_class_counts"]
        == {
            "gram_negative_mic16_negative": 458,
            "gram_negative_mic16_positive": 1088,
            "gram_positive_mic16_negative": 303,
            "gram_positive_mic16_positive": 643,
        },
        "strict binary-context census changed",
    )
    return dict(by_sequence), census


def _build_graph(
    sequences: Mapping[str, str],
    studies: Mapping[str, set[str]],
    study_memberships: Mapping[str, int],
    balance: Mapping[str, Counter[str]],
    threshold: float,
) -> GraphResult:
    homology_groups = _homology_partition(sequences, threshold)
    _require(len(homology_groups) == 597, "homology component census changed")
    homology_by_sequence: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    union = _Dsu(sorted(sequences))
    for members in homology_groups:
        component_id = _digest(
            "amp-challenge:homology-component:v1", _HOMOLOGY_ALGORITHM, threshold, members
        )
        for sid in members:
            homology_by_sequence[sid] = component_id
        for sid in members[1:]:
            union.join(members[0], sid)
        edges.append(
            {
                "schema_version": 1,
                "edge_id": _digest(
                    "amp-challenge:grouping-edge:v1",
                    "homology_component",
                    _HOMOLOGY_ALGORITHM,
                    threshold,
                    component_id,
                    members,
                ),
                "edge_type": "homology_component",
                "group_key": component_id,
                "sequence_ids": list(members),
                "sequence_count": len(members),
                "source_membership_count": None,
            }
        )
    for study_key in sorted(studies):
        members = tuple(sorted(studies[study_key]))
        for sid in members[1:]:
            union.join(members[0], sid)
        edges.append(
            {
                "schema_version": 1,
                "edge_id": _digest(
                    "amp-challenge:grouping-edge:v1",
                    "study_key",
                    study_key,
                    members,
                    study_memberships[study_key],
                ),
                "edge_type": "study_key",
                "group_key": study_key,
                "sequence_ids": list(members),
                "sequence_count": len(members),
                "source_membership_count": study_memberships[study_key],
            }
        )
    edges.sort(key=lambda row: (row["edge_type"], row["group_key"]))
    _require(len(edges) == 1032, "grouping-edge census changed")
    edge_ids_by_sequence: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        for sid in edge["sequence_ids"]:
            edge_ids_by_sequence[sid].add(edge["edge_id"])
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(sequences):
        groups[union.find(sid)].append(sid)
    components: list[dict[str, Any]] = []
    component_by_sequence: dict[str, str] = {}
    component_members: dict[str, tuple[str, ...]] = {}
    for raw_members in groups.values():
        members = tuple(sorted(raw_members))
        component_id = _digest(
            "amp-challenge:homology-study-component:v1",
            _COMPONENT_POLICY,
            _HOMOLOGY_ALGORITHM,
            threshold,
            members,
        )
        component_members[component_id] = members
        for sid in members:
            component_by_sequence[sid] = component_id
        homology_ids = {homology_by_sequence[sid] for sid in members}
        study_keys = {
            key for key, values in studies.items() if any(sid in values for sid in members)
        }
        edge_ids = sorted({edge_id for sid in members for edge_id in edge_ids_by_sequence[sid]})
        counts = Counter({"component_count": 1, "sequences": len(members)})
        for sid in members:
            counts.update(balance.get(sid, {}))
        components.append(
            {
                "id": component_id,
                "members": members,
                "homology_component_count": len(homology_ids),
                "study_key_count": len(study_keys),
                "grouping_edge_ids": edge_ids,
                "balance_counts": {metric: int(counts[metric]) for metric in _BALANCE_METRICS},
            }
        )
    components.sort(key=lambda row: row["id"])
    _require(len(components) == 278, "union-component census changed")
    expected_histogram = Counter(
        {
            1: 143,
            2: 49,
            3: 25,
            4: 12,
            5: 10,
            6: 4,
            7: 10,
            8: 7,
            9: 2,
            10: 1,
            11: 1,
            13: 2,
            14: 1,
            16: 1,
            17: 3,
            18: 1,
            23: 1,
            24: 1,
            30: 1,
            31: 1,
            38: 1,
            239: 1,
        }
    )
    _require(
        Counter(len(row["members"]) for row in components) == expected_histogram,
        "union-size histogram changed",
    )
    return GraphResult(
        edges=tuple(edges),
        components=tuple(components),
        homology_by_sequence=homology_by_sequence,
        component_by_sequence=component_by_sequence,
        component_members=component_members,
    )


def _score(
    loads: Sequence[Mapping[str, int]], totals: Mapping[str, int], config: FrozenConfig
) -> float:
    result = 0.0
    for metric in _BALANCE_METRICS:
        target = totals[metric] / config.folds
        result += config.weights[metric] * sum(
            ((load[metric] - target) / target) ** 2 for load in loads
        )
    return result


def _assign(
    components: Sequence[Mapping[str, Any]], config: FrozenConfig
) -> tuple[dict[str, int], list[dict[str, int]], dict[str, int], float]:
    totals = {
        metric: sum(row["balance_counts"][metric] for row in components)
        for metric in _BALANCE_METRICS
    }
    _require(all(value > 0 for value in totals.values()), "a balance total is zero")
    ordered = sorted(
        components,
        key=lambda row: (
            -max(
                row["balance_counts"][metric] / (totals[metric] / config.folds)
                for metric in _BALANCE_METRICS
            ),
            -len(row["members"]),
            _digest("homology-study-split-v1:component-order", config.seed, row["id"]),
        ),
    )
    loads = [{metric: 0 for metric in _BALANCE_METRICS} for _ in range(config.folds)]
    assignments: dict[str, int] = {}
    for index, component in enumerate(ordered):
        candidates = (
            [fold for fold in range(config.folds) if loads[fold]["component_count"] == 0]
            if index < config.folds
            else list(range(config.folds))
        )
        choices: list[tuple[float, str, int]] = []
        for fold in candidates:
            projected = [dict(load) for load in loads]
            for metric in _BALANCE_METRICS:
                projected[fold][metric] += component["balance_counts"][metric]
            choices.append(
                (
                    _score(projected, totals, config),
                    _digest(
                        "homology-study-split-v1:fold-placement", config.seed, component["id"], fold
                    ),
                    fold,
                )
            )
        _, _, chosen = min(choices)
        assignments[component["id"]] = chosen
        for metric in _BALANCE_METRICS:
            loads[chosen][metric] += component["balance_counts"][metric]
    _require(all(load["component_count"] for load in loads), "fold assignment is empty")
    _require(
        all(load[metric] >= config.minima[metric] for load in loads for metric in _CLASS_METRICS),
        "fold assignment misses a class minimum",
    )
    return assignments, loads, totals, _score(loads, totals, config)


def _sequence_set_sha256(sequence_ids: Sequence[str]) -> str:
    payload = "".join(f"{sequence_id}\n" for sequence_id in sorted(sequence_ids)).encode("ascii")
    return _sha256_bytes(payload)


def _manifest_entry(value: Any, *, label: str) -> dict[str, str]:
    _require(isinstance(value, dict), f"{label} must be an object")
    _exact_fields(value, frozenset({"filename", "sha256"}), label=label)
    filename = value["filename"]
    _require(
        isinstance(filename, str) and filename and filename == filename.strip(),
        f"{label} filename",
    )
    _safe_manifest_name(filename, label=label)
    digest = _sha(value["sha256"], label=f"{label} sha256")
    return {"filename": filename, "sha256": digest}


def _verify_sidecar(context_run: Path, config: FrozenConfig) -> tuple[dict[str, Any], str]:
    top_entries = _verify_tree_top_manifest(
        context_run,
        expected_sha256=_EXPECTED_CONTEXT_TOP_SHA256,
        label="endpoint-context input",
    )
    _require(set(top_entries) == _SIDECAR_TOP_ENTRIES, "endpoint-context inventory changed")
    output = context_run / "endpoint_context"
    manifest_path = output / "manifest.json"
    manifest_payload = _read_regular_file(manifest_path, label="endpoint-context manifest")
    _require(
        _sha256_bytes(manifest_payload) == _EXPECTED_CONTEXT_MANIFEST_SHA256,
        "endpoint-context manifest hash mismatch",
    )
    manifest = _read_canonical_json(manifest_path, label="endpoint-context manifest")
    _exact_fields(manifest, _SIDECAR_MANIFEST_FIELDS, label="endpoint-context manifest")
    _require(
        manifest["schema_version"] == 1
        and not isinstance(manifest["schema_version"], bool)
        and manifest["artifact"] == "dramp_endpoint_context_sidecar"
        and manifest["status"] == _SIDECAR_STATUS,
        "endpoint-context manifest identity changed",
    )
    _sha(manifest["config_sha256"], label="endpoint-context config")

    counts = manifest["counts"]
    _require(isinstance(counts, dict), "endpoint-context counts must be an object")
    _exact_fields(
        counts,
        frozenset({"unique_sequences", "assay_observations", "contexts", "study_memberships"}),
        label="endpoint-context counts",
    )
    _require(
        counts["unique_sequences"] == config.expected_sequences
        and counts["assay_observations"] == config.expected_assays
        and counts["study_memberships"] == config.expected_memberships,
        "endpoint-context counts differ from split config",
    )
    _positive_int(counts["contexts"], label="endpoint-context contexts")

    inputs = manifest["input"]
    _require(isinstance(inputs, dict), "endpoint-context input must be an object")
    _exact_fields(
        inputs,
        frozenset({"assays", "normalized_data_manifest", "normalized_summary", "sequences"}),
        label="endpoint-context input",
    )
    parsed_inputs = {
        name: _manifest_entry(entry, label=f"endpoint-context input {name}")
        for name, entry in inputs.items()
    }
    _require(
        parsed_inputs["sequences"]
        == {"filename": "sequences.jsonl", "sha256": _EXPECTED_SEQUENCE_SHA256},
        "endpoint-context manifest sequence binding changed",
    )

    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict), "endpoint-context artifacts must be an object")
    expected_filenames = {
        "audit": "audit.json",
        "contexts": "contexts.jsonl",
        "endpoint_context_ledger": "endpoint_context_ledger.jsonl",
        "study_membership": "study_membership.jsonl",
    }
    _exact_fields(artifacts, frozenset(expected_filenames), label="endpoint-context artifacts")
    for name, filename in expected_filenames.items():
        entry = _manifest_entry(artifacts[name], label=f"endpoint-context artifact {name}")
        payload = _read_regular_file(output / filename, label=f"endpoint-context {filename}")
        _require(
            entry == {"filename": filename, "sha256": _sha256_bytes(payload)},
            f"endpoint-context artifact binding changed for {name}",
        )
    _require(
        artifacts["endpoint_context_ledger"]
        == {"filename": "endpoint_context_ledger.jsonl", "sha256": _EXPECTED_LEDGER_SHA256},
        "endpoint-context ledger binding changed",
    )
    _require(
        artifacts["study_membership"]
        == {"filename": "study_membership.jsonl", "sha256": _EXPECTED_STUDY_SHA256},
        "study-membership binding changed",
    )

    policies = manifest["policies"]
    _require(isinstance(policies, dict), "endpoint-context policies must be an object")
    _exact_fields(policies, _SIDECAR_POLICY_FIELDS, label="endpoint-context policies")
    _require(
        isinstance(policies["study_keys"], str)
        and "never a model feature" in policies["study_keys"],
        "endpoint-context study keys are not declared grouping-only",
    )
    provenance = manifest["provenance"]
    _require(isinstance(provenance, dict), "endpoint-context provenance must be an object")
    _exact_fields(
        provenance,
        frozenset({"code_manifest", "git_commit"}),
        label="endpoint-context provenance",
    )
    code = _manifest_entry(provenance["code_manifest"], label="endpoint-context code manifest")
    _require(
        code
        == {
            "filename": "CODE_SHA256SUMS",
            "sha256": top_entries["CODE_SHA256SUMS"],
        },
        "endpoint-context code-manifest binding changed",
    )
    _require(
        isinstance(provenance["git_commit"], str)
        and _GIT_RE.fullmatch(provenance["git_commit"]) is not None,
        "endpoint-context Git commit is invalid",
    )
    return manifest, _EXPECTED_CONTEXT_TOP_SHA256


def _verify_code_manifest(
    run: Path,
    repo_root: Path,
    config: FrozenConfig,
    expected_git_commit: str,
) -> str:
    path = run / "CODE_SHA256SUMS"
    payload = _read_regular_file(path, label="split code manifest")
    entries = _parse_checksum_manifest(path, label="split code manifest")
    expected_config = (repo_root / "configs/data/homology_study_split_v1.toml").resolve(strict=True)
    _require(config.path == expected_config, "split config is not the repository frozen config")
    python_root = repo_root / "src/amp_challenge"
    expected_python = {
        path.relative_to(repo_root).as_posix()
        for path in python_root.rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    expected = expected_python | set(_REQUIRED_CODE_PATHS)
    _require(set(entries) == expected, "split code inventory differs from repository inventory")
    for relative in sorted(expected):
        requested = repo_root / relative
        _require(not requested.is_symlink(), f"code inventory path is a symlink: {relative}")
        resolved = requested.resolve(strict=True)
        _require(resolved.is_relative_to(repo_root), f"code inventory path escapes: {relative}")
        current = _read_regular_file(resolved, label=f"code inventory {relative}")
        committed = _committed_blob(repo_root, expected_git_commit, relative)
        _require(
            _sha256_bytes(current) == entries[relative],
            f"code inventory checksum mismatch for {relative}",
        )
        _require(current == committed, f"code inventory differs from committed blob: {relative}")
        _require(
            _sha256_bytes(committed) == entries[relative],
            f"code manifest does not attest committed blob: {relative}",
        )
    return _sha256_bytes(payload)


def _expected_frozen_entries(
    normalized_run: Path,
    context_run: Path,
    normalized_entries: Mapping[str, str],
    context_entries: Mapping[str, str],
) -> dict[str, str]:
    result = {f"parser-v7/{name}": digest for name, digest in normalized_entries.items()}
    result["parser-v7/SHA256SUMS"] = _sha256_bytes(
        _read_regular_file(normalized_run / "SHA256SUMS", label="normalized top manifest")
    )
    result.update({f"endpoint-context/{name}": digest for name, digest in context_entries.items()})
    result["endpoint-context/SHA256SUMS"] = _sha256_bytes(
        _read_regular_file(context_run / "SHA256SUMS", label="endpoint-context top manifest")
    )
    return dict(sorted(result.items()))


def _verify_frozen_input_manifest(
    run: Path,
    normalized_run: Path,
    context_run: Path,
    normalized_entries: Mapping[str, str],
    context_entries: Mapping[str, str],
) -> str:
    path = run / "FROZEN_INPUT_SHA256SUMS"
    payload = _read_regular_file(path, label="split frozen-input manifest")
    entries = _parse_checksum_manifest(path, label="split frozen-input manifest")
    expected = _expected_frozen_entries(
        normalized_run, context_run, normalized_entries, context_entries
    )
    _require(entries == expected, "split frozen-input attestation differs from accepted inputs")
    return _sha256_bytes(payload)


def _verify_output_top(run: Path) -> tuple[str, dict[str, str]]:
    inventory = _tree_inventory(run)
    _require(inventory == _TOP_FILES, "split run inventory mismatch")
    split_inventory = _tree_inventory(run / "split")
    _require(split_inventory == _SPLIT_FILES, "split artifact inventory mismatch")
    path = run / "SHA256SUMS"
    payload = _read_regular_file(path, label="split top manifest")
    entries = _parse_checksum_manifest(path, label="split top manifest")
    _require(set(entries) == _TOP_MANIFEST_ENTRIES, "split top manifest inventory mismatch")
    _verify_manifest_files(run, entries, label="split top manifest")
    return _sha256_bytes(payload), entries


def _expected_semantic_artifacts(
    sequences: Mapping[str, str],
    studies: Mapping[str, set[str]],
    graph: GraphResult,
    census: Mapping[str, Any],
    folds: Mapping[str, int],
    loads: Sequence[Mapping[str, int]],
    totals: Mapping[str, int],
    objective: float,
    config: FrozenConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assignments = [
        {
            "schema_version": 1,
            "sequence_id": sequence_id,
            "homology_component_id": graph.homology_by_sequence[sequence_id],
            "union_component_id": graph.component_by_sequence[sequence_id],
            "fold": folds[graph.component_by_sequence[sequence_id]],
        }
        for sequence_id in sorted(sequences)
    ]
    component_rows = [
        {
            "schema_version": 1,
            "union_component_id": component["id"],
            "fold": folds[component["id"]],
            "sequence_count": len(component["members"]),
            "sequence_ids_sha256": _sequence_set_sha256(component["members"]),
            "grouping_edge_ids": component["grouping_edge_ids"],
            "homology_component_count": component["homology_component_count"],
            "study_key_count": component["study_key_count"],
            "balance_counts": component["balance_counts"],
        }
        for component in graph.components
    ]
    by_sequence = {row["sequence_id"]: row for row in assignments}
    all_edges_single_component = all(
        len({by_sequence[sid]["union_component_id"] for sid in edge["sequence_ids"]}) == 1
        for edge in graph.edges
    )
    all_edges_single_fold = all(
        len({by_sequence[sid]["fold"] for sid in edge["sequence_ids"]}) == 1 for edge in graph.edges
    )
    invariants = {
        "all_sequences_assigned_once": len(assignments) == len(sequences)
        and set(by_sequence) == set(sequences),
        "all_homology_and_study_edges_within_one_component": all_edges_single_component,
        "all_homology_and_study_edges_within_one_fold": all_edges_single_fold,
        "all_components_have_one_fold": len(folds) == len(graph.components),
        "all_folds_nonempty": set(folds.values()) == set(range(config.folds)),
        "study_memberships_cover_all_sequences": set().union(*studies.values()) == set(sequences),
        "source_record_study_keys_are_singletons": all(
            len(members) == 1
            for key, members in studies.items()
            if key.startswith("source-record:")
        ),
        "all_mic_contexts_partitioned": census["mic_assay_contexts"]
        == census["all_ineligible_assay_contexts"]
        + census["mixed_eligibility_assay_contexts"]
        + census["fully_eligible_assay_contexts"],
        "all_ledger_endpoints_partitioned": census["ledger_raw_observations"]
        == census["mic_raw_observations"] + census["non_mic_raw_observations"],
        "all_ledger_contexts_partitioned": census["ledger_assay_contexts"]
        == census["mic_assay_contexts"] + census["non_mic_assay_contexts"],
        "all_mic_observations_partitioned": census["mic_raw_observations"]
        == census["observations_in_all_ineligible_assay_contexts"]
        + census["observations_in_mixed_eligibility_assay_contexts"]
        + census["observations_in_conflicting_assay_contexts"]
        + census["retained_source_observations"],
        "candidate_binary_contexts_partitioned": census["candidate_assay_contexts"]
        == census["mixed_eligibility_assay_contexts"]
        + census["conflicting_assay_contexts"]
        + census["retained_assay_contexts"],
        "fully_eligible_binary_contexts_partitioned": census["fully_eligible_assay_contexts"]
        == census["conflicting_assay_contexts"] + census["retained_assay_contexts"],
        "eligible_binary_observations_partitioned": census["eligible_raw_observations"]
        == census["eligible_observations_in_mixed_eligibility_assay_contexts"]
        + census["observations_in_conflicting_assay_contexts"]
        + census["retained_source_observations"],
        "balance_totals_equal_component_sum": all(
            totals[metric]
            == sum(component["balance_counts"][metric] for component in graph.components)
            for metric in _BALANCE_METRICS
        ),
        "balance_class_totals_equal_retained_context_census": all(
            totals[metric] == census["retained_class_counts"][metric] for metric in _CLASS_METRICS
        ),
        "all_balance_minima_met": all(
            loads[fold][metric] >= config.minima[metric]
            for fold in range(config.folds)
            for metric in _CLASS_METRICS
        ),
        "input_snapshots_unchanged": True,
        "code_inventory_unchanged": True,
    }
    _require(all(invariants.values()), "an independently recomputed invariant failed")
    component_sizes = Counter(len(component["members"]) for component in graph.components)
    largest = max(graph.components, key=lambda row: (len(row["members"]), row["id"]))
    study_edges = [edge for edge in graph.edges if edge["edge_type"] == "study_key"]
    audit = {
        "schema_version": 1,
        "artifact": "homology_study_union_split_audit",
        "status": _OUTPUT_STATUS,
        "input": {
            "unique_sequences": len(sequences),
            "assay_observations": config.expected_assays,
            "study_memberships": config.expected_memberships,
        },
        "graph": {
            "component_policy": _COMPONENT_POLICY,
            "homology_algorithm": _HOMOLOGY_ALGORITHM,
            "identity_threshold": config.threshold,
            "homology_components": sum(
                edge["edge_type"] == "homology_component" for edge in graph.edges
            ),
            "study_keys": len(study_edges),
            "shared_study_keys": sum(edge["sequence_count"] > 1 for edge in study_edges),
            "study_keys_merging_homology_components": sum(
                len({graph.homology_by_sequence[sid] for sid in edge["sequence_ids"]}) > 1
                for edge in study_edges
            ),
            "union_components": len(graph.components),
            "component_size_histogram": {
                str(size): count for size, count in sorted(component_sizes.items())
            },
            "largest_component": {
                "union_component_id": largest["id"],
                "sequences": len(largest["members"]),
                "homology_components": largest["homology_component_count"],
                "study_keys": largest["study_key_count"],
                "balance_counts": largest["balance_counts"],
            },
        },
        "balance": {
            "policy": _BALANCE_POLICY,
            "folds": config.folds,
            "seed": config.seed,
            "weights": dict(config.weights),
            "minimum_per_fold": dict(config.minima),
            "target": "corpus total for each dimension divided by fold count",
            "component_order": (
                "descending max(component_i/target_i), descending sequence count, "
                "ascending typed seeded digest"
            ),
            "component_order_digest_namespace": "homology-study-split-v1:component-order",
            "placement": (
                "forced distinct first K then minimum weighted global normalized squared error"
            ),
            "fold_placement_digest_namespace": "homology-study-split-v1:fold-placement",
            "unit": (
                "unique MIC assay_context_id retained only when every member is bacterial_mic16 "
                "eligible and all labels and source Gram values agree"
            ),
            "context_census": dict(census),
            "totals": dict(totals),
            "targets_per_fold": {
                metric: totals[metric] / config.folds for metric in _BALANCE_METRICS
            },
            "objective_score": objective,
            "objective_score_hex": objective.hex(),
            "by_fold": {str(fold): dict(loads[fold]) for fold in range(config.folds)},
        },
        "invariants": invariants,
        "limitations": [
            "study keys are conservative grouping proxies and are never model features",
            "reference-title keys can over-merge while missing metadata can under-merge studies",
            "single-link transitivity does not imply every within-component pair reaches threshold",
            "large indivisible components can make exact fold balance impossible",
            "DRAMP outcomes were previously inspected and are not an untouched evaluation panel",
        ],
    }
    return assignments, component_rows, audit


def _verify_split_manifest(
    split_dir: Path,
    config: FrozenConfig,
    sidecar_manifest: Mapping[str, Any],
    code_manifest_sha256: str,
    artifact_digests: Mapping[str, str],
    expected_git_commit: str,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    manifest = _read_canonical_json(split_dir / "manifest.json", label="split manifest")
    _exact_fields(
        manifest,
        frozenset(
            {
                "schema_version",
                "artifact",
                "status",
                "config_sha256",
                "input",
                "upstream",
                "policies",
                "counts",
                "artifacts",
                "provenance",
                "runtime",
            }
        ),
        label="split manifest",
    )
    runtime = manifest["runtime"]
    _require(isinstance(runtime, dict), "split runtime must be an object")
    _exact_fields(runtime, frozenset({"python"}), label="split runtime")
    _require(
        isinstance(runtime["python"], str)
        and _PYTHON_VERSION_RE.fullmatch(runtime["python"]) is not None,
        "split Python version is invalid",
    )
    provenance = manifest["provenance"]
    _require(isinstance(provenance, dict), "split provenance must be an object")
    _exact_fields(provenance, frozenset({"git_commit", "code_manifest"}), label="split provenance")
    git_commit = provenance["git_commit"]
    _require(
        isinstance(git_commit, str) and _GIT_RE.fullmatch(git_commit) is not None,
        "split Git commit is invalid",
    )
    _require(git_commit == expected_git_commit, "split Git commit differs from expectation")
    expected = {
        "schema_version": 1,
        "artifact": "homology_study_union_split",
        "status": _OUTPUT_STATUS,
        "config_sha256": config.sha256,
        "input": {
            "sequences": {
                "filename": "sequences.jsonl",
                "sha256": _EXPECTED_SEQUENCE_SHA256,
            },
            "endpoint_context_ledger": {
                "filename": "endpoint_context_ledger.jsonl",
                "sha256": _EXPECTED_LEDGER_SHA256,
            },
            "study_membership": {
                "filename": "study_membership.jsonl",
                "sha256": _EXPECTED_STUDY_SHA256,
            },
            "endpoint_context_manifest": {
                "filename": "manifest.json",
                "sha256": _EXPECTED_CONTEXT_MANIFEST_SHA256,
            },
            "endpoint_context_top_manifest": {
                "filename": "SHA256SUMS",
                "sha256": _EXPECTED_CONTEXT_TOP_SHA256,
            },
        },
        "upstream": {
            "artifact": sidecar_manifest["artifact"],
            "status": sidecar_manifest["status"],
            "git_commit": sidecar_manifest["provenance"]["git_commit"],
        },
        "policies": {
            "component": _COMPONENT_POLICY,
            "homology": _HOMOLOGY_ALGORITHM,
            "study": "union every exact study key; grouping-only and never a model feature",
            "grouping_edges": "audit-only and never a model feature",
            "balance": _BALANCE_POLICY,
            "binary_balance_unit": (
                "unique MIC assay_context_id; require every member bacterial_mic16 eligible "
                "and unanimous determinate label and source Gram; otherwise exclude in full"
            ),
            "identifier": "typed canonical JSON with domain-separated SHA-256",
        },
        "counts": dict(expected_counts),
        "artifacts": {
            name: {"filename": f"{name}.jsonl", "sha256": digest}
            for name, digest in artifact_digests.items()
            if name != "audit"
        }
        | {"audit": {"filename": "audit.json", "sha256": artifact_digests["audit"]}},
        "provenance": {
            "git_commit": git_commit,
            "code_manifest": {
                "filename": "CODE_SHA256SUMS",
                "sha256": code_manifest_sha256,
            },
        },
        "runtime": {"python": runtime["python"]},
    }
    _require(manifest == expected, "split manifest differs from independently derived manifest")
    return manifest


def _verify_no_path_leakage(run: Path, forbidden_prefixes: Sequence[bytes]) -> None:
    absolute_path = re.compile(rb"/(?:lustre|tmp)/|/home/[^/\s\"']+/")
    for relative in sorted(_TOP_FILES):
        payload = _read_regular_file(run / relative, label=f"path-leak scan {relative}")
        for prefix in forbidden_prefixes:
            _require(prefix not in payload, f"split artifact {relative} leaks a forbidden path")
        _require(
            absolute_path.search(payload) is None,
            f"split artifact {relative} leaks an absolute execution path",
        )


def _verify_tree_bytes(left: Path, right: Path, *, expected: frozenset[str] | None = None) -> None:
    left_inventory = _tree_inventory(left)
    right_inventory = _tree_inventory(right)
    _require(left_inventory == right_inventory, "twin tree inventories differ")
    if expected is not None:
        _require(left_inventory == expected, "twin tree inventory differs from contract")
    for relative in sorted(left_inventory):
        left_payload = _read_regular_file(left / relative, label=f"left twin {relative}")
        right_payload = _read_regular_file(right / relative, label=f"right twin {relative}")
        _require(left_payload == right_payload, f"twin bytes differ for {relative}")


def _verify_one_twin(
    *,
    run: Path,
    normalized_run: Path,
    context_run: Path,
    config: FrozenConfig,
    repo_root: Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[bytes],
) -> dict[str, Any]:
    top_manifest_sha256, top_entries = _verify_output_top(run)
    normalized_entries = _verify_tree_top_manifest(
        normalized_run,
        expected_sha256=_EXPECTED_NORMALIZED_TOP_SHA256,
        label="normalized input",
        require_sorted=False,
    )
    sidecar_manifest, context_top_sha256 = _verify_sidecar(context_run, config)
    context_entries = _parse_checksum_manifest(
        context_run / "SHA256SUMS", label="endpoint-context top manifest"
    )
    frozen_manifest_sha256 = _verify_frozen_input_manifest(
        run,
        normalized_run,
        context_run,
        normalized_entries,
        context_entries,
    )
    code_manifest_sha256 = _verify_code_manifest(run, repo_root, config, expected_git_commit)

    sequences_path = normalized_run / "normalized/sequences.jsonl"
    ledger_path = context_run / "endpoint_context/endpoint_context_ledger.jsonl"
    studies_path = context_run / "endpoint_context/study_membership.jsonl"
    sequences = _load_sequences(sequences_path, config)
    studies, study_memberships, membership = _load_studies(studies_path, sequences, config)
    balance, census = _load_balance(ledger_path, sequences, membership, config)
    graph = _build_graph(sequences, studies, study_memberships, balance, config.threshold)
    folds, loads, totals, objective = _assign(graph.components, config)
    expected_assignments, expected_components, expected_audit = _expected_semantic_artifacts(
        sequences,
        studies,
        graph,
        census,
        folds,
        loads,
        totals,
        objective,
        config,
    )

    split_dir = run / "split"
    assignments = _read_jsonl(
        split_dir / "sequence_assignments.jsonl", label="sequence assignments"
    )
    components = _read_jsonl(split_dir / "components.jsonl", label="components")
    edges = _read_jsonl(split_dir / "grouping_edges.jsonl", label="grouping edges")
    _require(
        list(assignments.rows) == expected_assignments,
        "sequence assignments differ from independent reconstruction",
    )
    _require(
        list(components.rows) == expected_components,
        "component rows differ from independent reconstruction",
    )
    _require(
        list(edges.rows) == list(graph.edges),
        "grouping edges differ from independent reconstruction",
    )
    audit = _read_canonical_json(split_dir / "audit.json", label="split audit")
    _require(audit == expected_audit, "split audit differs from independent reconstruction")

    artifact_digests = {
        "sequence_assignments": _sha256_bytes(assignments.payload),
        "components": _sha256_bytes(components.payload),
        "grouping_edges": _sha256_bytes(edges.payload),
        "audit": _sha256_bytes(_read_regular_file(split_dir / "audit.json", label="split audit")),
    }
    expected_counts = {
        "unique_sequences": len(sequences),
        "homology_components": sum(
            edge["edge_type"] == "homology_component" for edge in graph.edges
        ),
        "study_keys": len(studies),
        "union_components": len(graph.components),
        "folds": config.folds,
        "sequence_assignments": len(expected_assignments),
        "grouping_edges": len(graph.edges),
        "retained_balance_contexts": census["retained_assay_contexts"],
        "excluded_mixed_eligibility_balance_contexts": census["mixed_eligibility_assay_contexts"],
        "excluded_conflicting_balance_contexts": census["conflicting_assay_contexts"],
    }
    manifest = _verify_split_manifest(
        split_dir,
        config,
        sidecar_manifest,
        code_manifest_sha256,
        artifact_digests,
        expected_git_commit,
        expected_counts,
    )
    _require(
        top_entries["CODE_SHA256SUMS"] == code_manifest_sha256
        and top_entries["FROZEN_INPUT_SHA256SUMS"] == frozen_manifest_sha256,
        "top manifest does not bind provenance manifests",
    )
    _verify_no_path_leakage(run, forbidden_prefixes)

    # Re-read all long-lived attestations after the expensive reconstruction.
    _verify_tree_top_manifest(
        normalized_run,
        expected_sha256=_EXPECTED_NORMALIZED_TOP_SHA256,
        label="normalized input final",
        require_sorted=False,
    )
    _verify_sidecar(context_run, config)
    _verify_frozen_input_manifest(
        run,
        normalized_run,
        context_run,
        normalized_entries,
        context_entries,
    )
    _verify_code_manifest(run, repo_root, config, expected_git_commit)
    _verify_repository_state(repo_root, expected_git_commit)
    final_top_sha256, final_top_entries = _verify_output_top(run)
    _require(
        final_top_sha256 == top_manifest_sha256 and final_top_entries == top_entries,
        "split output changed during independent verification",
    )

    return {
        "top_manifest_sha256": top_manifest_sha256,
        "code_manifest_sha256": code_manifest_sha256,
        "frozen_input_manifest_sha256": frozen_manifest_sha256,
        "normalized_top_manifest_sha256": _EXPECTED_NORMALIZED_TOP_SHA256,
        "endpoint_context_top_manifest_sha256": context_top_sha256,
        "git_commit": manifest["provenance"]["git_commit"],
        "counts": expected_counts,
        "graph": expected_audit["graph"],
        "balance": {
            "context_census": census,
            "totals": dict(totals),
            "objective_score_hex": objective.hex(),
            "by_fold": expected_audit["balance"]["by_fold"],
        },
        "artifact_sha256": dict(sorted(artifact_digests.items())),
    }


def verify_homology_study_split_twins(
    *,
    twin_root: str | Path,
    normalized_twin_root: str | Path,
    endpoint_context_twin_root: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify both production twins and return a path-free deterministic receipt."""

    twins = _resolved_directory(twin_root, label="split twin root")
    normalized_twins = _resolved_directory(normalized_twin_root, label="normalized twin root")
    context_twins = _resolved_directory(
        endpoint_context_twin_root, label="endpoint-context twin root"
    )
    repository = _resolved_directory(repo_root, label="repository root")
    requested_config = Path(config_path)
    _require_no_lexical_symlink(requested_config, label="split config", ancestors=True)
    config_file = requested_config.resolve(strict=True)
    _require(repository.is_dir(), "repository root is not a directory")
    _require(config_file.is_relative_to(repository), "split config must be in the repository")
    _require(
        _GIT_RE.fullmatch(expected_git_commit) is not None,
        "expected Git commit must be a full lowercase SHA-1",
    )
    _verify_execution_path(repository)
    _verify_repository_state(repository, expected_git_commit)
    config = _load_config(config_file)
    encoded_prefixes = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(isinstance(prefix, str) and prefix, "forbidden path prefix is empty")
        encoded_prefixes.append(prefix.encode("utf-8"))

    runs = [twins / "0", twins / "1"]
    normalized_runs = [normalized_twins / "0", normalized_twins / "1"]
    context_runs = [context_twins / "0", context_twins / "1"]
    for label, paths in (
        ("split", runs),
        ("normalized", normalized_runs),
        ("endpoint-context", context_runs),
    ):
        for path in paths:
            _require(
                path.is_dir() and not path.is_symlink(),
                f"required {label} twin directory is absent or symbolic",
            )
    _verify_tree_bytes(runs[0], runs[1], expected=_TOP_FILES)
    _verify_tree_bytes(normalized_runs[0], normalized_runs[1])
    _verify_tree_bytes(context_runs[0], context_runs[1])

    receipts = [
        _verify_one_twin(
            run=run,
            normalized_run=normalized_run,
            context_run=context_run,
            config=config,
            repo_root=repository,
            expected_git_commit=expected_git_commit,
            forbidden_prefixes=encoded_prefixes,
        )
        for run, normalized_run, context_run in zip(
            runs, normalized_runs, context_runs, strict=True
        )
    ]
    _require(receipts[0] == receipts[1], "semantic twin receipts differ")
    _verify_repository_state(repository, expected_git_commit)
    _verify_execution_path(repository)
    return {
        "schema_version": 1,
        "artifact": "homology_study_union_split_independent_verification",
        "status": "passed",
        "checks": {
            "top_manifests_valid": True,
            "twins_byte_identical": True,
            "input_twins_byte_identical": True,
            "frozen_inputs_config_pinned": True,
            "code_and_input_attestations_valid": True,
            "repository_commit_and_cleanliness_verified": True,
            "executing_source_bound_to_repository": True,
            "json_and_jsonl_canonical_lf": True,
            "homology_and_study_union_recomputed": True,
            "strict_context_exclusions_recomputed": True,
            "fold_assignment_recomputed": True,
            "artifacts_and_manifest_exact": True,
            "scratch_and_absolute_paths_absent": True,
        },
        **receipts[0],
    }


def _write_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    protected_roots: Sequence[Path] = (),
) -> None:
    requested = Path(path)
    _require(requested.name not in {"", ".", ".."}, "verification receipt has no filename")
    _require_no_lexical_symlink(
        requested.parent, label="verification receipt parent", ancestors=True
    )
    _require(
        not os.path.lexists(requested),
        f"refusing to overwrite verification receipt: {requested}",
    )
    requested.parent.mkdir(parents=True, exist_ok=True)
    _require_no_lexical_symlink(
        requested.parent, label="verification receipt parent", ancestors=True
    )
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    for root in protected_roots:
        protected = root.resolve(strict=True)
        _require(
            target != protected and not target.is_relative_to(protected),
            "verification receipt must be outside every verified input and output tree",
        )
    _require(not os.path.lexists(target), f"verification receipt appeared: {target}")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                receipt,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        staging.unlink()
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        staging.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-root", type=Path, required=True)
    parser.add_argument("--normalized-twin-root", type=Path, required=True)
    parser.add_argument("--endpoint-context-twin-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--forbidden-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protected_roots = [
        _resolved_directory(args.twin_root, label="split twin root"),
        _resolved_directory(args.normalized_twin_root, label="normalized twin root"),
        _resolved_directory(args.endpoint_context_twin_root, label="endpoint-context twin root"),
        _resolved_directory(args.repo_root, label="repository root"),
    ]
    requested_output = Path(args.output)
    _require_no_lexical_symlink(
        requested_output.parent, label="verification receipt parent", ancestors=True
    )
    prospective_output = requested_output.resolve(strict=False)
    for root in protected_roots:
        _require(
            prospective_output != root and not prospective_output.is_relative_to(root),
            "verification receipt must be outside every verified input and output tree",
        )
    receipt = verify_homology_study_split_twins(
        twin_root=args.twin_root,
        normalized_twin_root=args.normalized_twin_root,
        endpoint_context_twin_root=args.endpoint_context_twin_root,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    _write_receipt(args.output, receipt, protected_roots=protected_roots)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the Slurm CLI
    raise SystemExit(main())

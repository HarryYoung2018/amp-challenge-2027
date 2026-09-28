"""Build a deterministic full-sequence homology-and-study split.

The endpoint-context sidecar is immutable input.  This builder treats every
canonical sequence as a graph vertex, connects vertices through full-union
single-link homology components and every study-key hyperedge, and assigns the
resulting connected components to folds.  Study metadata is written only to an
audit-only edge table and must never be used as a model feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import cluster_sequences

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_HOMOLOGY_ALGORITHM = "global_alignment_identity_single_link_v1"
_COMPONENT_POLICY = "full_sequence_homology_union_every_study_key_v1"
_BALANCE_POLICY = "seeded_largest_normalized_load_first_global_squared_error_v1"

_BALANCE_METRICS = (
    "component_count",
    "sequences",
    "gram_negative_mic16_negative",
    "gram_negative_mic16_positive",
    "gram_positive_mic16_negative",
    "gram_positive_mic16_positive",
)
_CLASS_METRICS = _BALANCE_METRICS[2:]
_ENDPOINTS = frozenset({"mic", "hc50", "hemolysis_percent"})

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
_STUDY_STATUSES = frozenset(
    {
        "explicit_pmid",
        "explicit_pmid_with_ignored_tokens",
        "reference_title_fallback",
        "source_record_singleton",
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
_SIDECAR_TOP_ENTRIES = (
    "CODE_SHA256SUMS",
    "FROZEN_INPUT_SHA256SUMS",
    "endpoint_context/audit.json",
    "endpoint_context/contexts.jsonl",
    "endpoint_context/endpoint_context_ledger.jsonl",
    "endpoint_context/manifest.json",
    "endpoint_context/study_membership.jsonl",
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


@dataclass(frozen=True, slots=True)
class HomologyStudySplitConfig:
    path: Path
    identity_threshold: float
    folds: int
    seed: int
    normalized_sequences_sha256: str
    endpoint_context_ledger_sha256: str
    study_membership_sha256: str
    endpoint_context_manifest_sha256: str
    endpoint_context_top_manifest_sha256: str
    expected_unique_sequences: int
    expected_assay_observations: int
    expected_study_memberships: int
    balance_weights: Mapping[str, float]
    balance_minima: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class HomologyStudySplitExecution:
    output_dir: Path
    sequence_assignments_path: Path
    components_path: Path
    grouping_edges_path: Path
    audit_path: Path
    manifest_path: Path
    unique_sequences: int
    homology_components: int
    union_components: int


@dataclass(frozen=True, slots=True)
class _GraphComponent:
    union_component_id: str
    sequence_ids: tuple[str, ...]
    homology_component_ids: tuple[str, ...]
    study_keys: tuple[str, ...]
    grouping_edge_ids: tuple[str, ...]
    balance_counts: Mapping[str, int]


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        while parent != self._parent[parent]:
            self._parent[parent] = self._parent[self._parent[parent]]
            parent = self._parent[parent]
        while value != parent:
            next_value = self._parent[value]
            self._parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _typed_id_frame(value: object) -> object:
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stable ID parts must not contain non-finite floats")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, tuple):
        return ["tuple", [_typed_id_frame(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_typed_id_frame(item) for item in value]]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("stable ID mapping keys must be strings")
        return [
            "mapping",
            [[key, _typed_id_frame(value[key])] for key in sorted(value)],
        ]
    raise TypeError(f"unsupported stable ID part type: {type(value).__name__}")


def _stable_digest(namespace: str, *parts: object) -> str:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("stable ID namespace must be a non-empty string")
    payload = _canonical_json(
        {
            "namespace": _typed_id_frame(namespace),
            "parts": [_typed_id_frame(part) for part in parts],
        }
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _fingerprint(result: os.stat_result) -> tuple[int, int, int, int]:
    return (result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns)


def _read_snapshot(path: str | Path, *, name: str) -> InputSnapshot:
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {requested}")
    source = requested.resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} is not a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    before_fingerprint = _fingerprint(before)
    if before_fingerprint != _fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{name} changed while it was being read: {source}")
    return InputSnapshot(
        path=source,
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=before_fingerprint,
    )


def _assert_snapshot_unchanged(snapshot: InputSnapshot, *, name: str) -> None:
    try:
        current = snapshot.path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} disappeared during the build") from error
    if not stat.S_ISREG(current.st_mode) or _fingerprint(current) != snapshot.fingerprint:
        raise ValueError(f"{name} changed during the build")


def _require_exact_fields(
    row: Mapping[str, object],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    if set(row) != expected:
        missing = sorted(expected - set(row))
        extra = sorted(set(row) - expected)
        raise ValueError(f"{name} schema mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty edge-trimmed string")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _parse_json_object(payload: bytes, *, name: str) -> dict[str, object]:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{name} must use canonical LF framing with one final LF")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return cast(dict[str, object], value)


def _parse_jsonl(payload: bytes, *, name: str) -> list[dict[str, object]]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{name} must be non-empty canonical LF-delimited JSONL")
    raw_lines = payload[:-1].split(b"\n")
    rows: list[dict[str, object]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw:
            raise ValueError(f"{name} line {line_number} is blank")
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{name} line {line_number} is not valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{name} line {line_number} is not a JSON object")
        rows.append(cast(dict[str, object], value))
    return rows


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_sha256_manifest(payload: bytes, *, name: str) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{name} must be non-empty LF-delimited text with a final LF")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error
    entries: dict[str, str] = {}
    previous: str | None = None
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise ValueError(f"{name} line {line_number} is not a SHA-256 manifest entry")
        digest, mode, filename = match.groups()
        if mode != " ":
            raise ValueError(f"{name} line {line_number} must use text-mode checksum syntax")
        pure = PurePosixPath(filename)
        if pure.is_absolute() or ".." in pure.parts or "\\" in filename:
            raise ValueError(f"{name} line {line_number} has an unsafe path")
        if filename in entries:
            raise ValueError(f"{name} repeats path {filename!r}")
        if previous is not None and filename <= previous:
            raise ValueError(f"{name} paths must be strictly sorted")
        entries[filename] = digest
        previous = filename
    return entries


def _config_from_payload(path: Path, payload: bytes) -> HomologyStudySplitConfig:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("homology-study split config must use canonical LF framing")
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("homology-study split config is not valid UTF-8 TOML") from error
    expected = {
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
    if set(raw) != expected:
        raise ValueError(
            "homology-study split config schema mismatch: "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
        )
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("homology-study split config schema_version must be 1")
    threshold_raw = raw["identity_threshold"]
    if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, int | float):
        raise ValueError("identity_threshold must be numeric")
    threshold = float(threshold_raw)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("identity_threshold must be finite and in (0, 1]")
    folds = _require_positive_int(raw["folds"], field="folds")
    if folds < 2:
        raise ValueError("folds must be at least two")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    weight_raw = raw["balance_weight"]
    minimum_raw = raw["balance_minimum"]
    if not isinstance(weight_raw, dict) or set(weight_raw) != set(_BALANCE_METRICS):
        raise ValueError(f"balance_weight must contain exactly {list(_BALANCE_METRICS)}")
    if not isinstance(minimum_raw, dict) or set(minimum_raw) != set(_CLASS_METRICS):
        raise ValueError(f"balance_minimum must contain exactly {list(_CLASS_METRICS)}")
    weights: dict[str, float] = {}
    for metric in _BALANCE_METRICS:
        value = weight_raw[metric]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"balance_weight.{metric} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"balance_weight.{metric} must be finite and positive")
        weights[metric] = number
    minima: dict[str, int] = {}
    for metric in _CLASS_METRICS:
        value = minimum_raw[metric]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"balance_minimum.{metric} must be a non-negative integer")
        minima[metric] = value

    expected_sequences = _require_positive_int(
        raw["expected_unique_sequences"], field="expected_unique_sequences"
    )
    if folds > expected_sequences:
        raise ValueError("fold count cannot exceed the expected sequence count")
    return HomologyStudySplitConfig(
        path=path,
        identity_threshold=threshold,
        folds=folds,
        seed=seed,
        normalized_sequences_sha256=_require_sha256(
            raw["normalized_sequences_sha256"], field="normalized_sequences_sha256"
        ),
        endpoint_context_ledger_sha256=_require_sha256(
            raw["endpoint_context_ledger_sha256"], field="endpoint_context_ledger_sha256"
        ),
        study_membership_sha256=_require_sha256(
            raw["study_membership_sha256"], field="study_membership_sha256"
        ),
        endpoint_context_manifest_sha256=_require_sha256(
            raw["endpoint_context_manifest_sha256"], field="endpoint_context_manifest_sha256"
        ),
        endpoint_context_top_manifest_sha256=_require_sha256(
            raw["endpoint_context_top_manifest_sha256"],
            field="endpoint_context_top_manifest_sha256",
        ),
        expected_unique_sequences=expected_sequences,
        expected_assay_observations=_require_positive_int(
            raw["expected_assay_observations"], field="expected_assay_observations"
        ),
        expected_study_memberships=_require_positive_int(
            raw["expected_study_memberships"], field="expected_study_memberships"
        ),
        balance_weights=weights,
        balance_minima=minima,
    )


def load_config(path: str | Path) -> HomologyStudySplitConfig:
    snapshot = _read_snapshot(path, name="homology-study split config")
    return _config_from_payload(snapshot.path, snapshot.payload)


def _validate_code_manifest(
    snapshot: InputSnapshot,
    *,
    config: HomologyStudySplitConfig,
    config_snapshot: InputSnapshot,
) -> tuple[InputSnapshot, ...]:
    entries = _parse_sha256_manifest(snapshot.payload, name="code manifest")
    repository_root = Path(__file__).resolve(strict=True).parents[3]
    expected_config = repository_root / "configs/data/homology_study_split_v1.toml"
    if config.path != expected_config:
        raise ValueError("split config must be configs/data/homology_study_split_v1.toml")
    python_paths = {
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "src/amp_challenge").rglob("*.py")
        if path.is_file()
    }
    expected = python_paths | set(_REQUIRED_CODE_PATHS)
    if set(entries) != expected:
        raise ValueError(
            "code manifest inventory mismatch: "
            f"missing={sorted(expected - set(entries))}, extra={sorted(set(entries) - expected)}"
        )
    snapshots: list[InputSnapshot] = []
    for relative in sorted(expected):
        requested = repository_root / relative
        if requested.is_symlink():
            raise ValueError(f"code manifest path must not be a symbolic link: {relative}")
        source = requested.resolve(strict=True)
        try:
            source.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(f"code manifest path escapes repository: {relative}") from error
        current = (
            config_snapshot
            if source == config_snapshot.path
            else _read_snapshot(source, name=f"code inventory {relative}")
        )
        if current.sha256 != entries[relative]:
            raise ValueError(f"code manifest checksum mismatch for {relative}")
        snapshots.append(current)
    return tuple(snapshots)


def _validate_sidecar_attestation(
    *,
    config: HomologyStudySplitConfig,
    sequences: InputSnapshot,
    ledger: InputSnapshot,
    memberships: InputSnapshot,
    manifest_snapshot: InputSnapshot,
    top_snapshot: InputSnapshot,
) -> dict[str, object]:
    configured = {
        "normalized sequences": (sequences.sha256, config.normalized_sequences_sha256),
        "endpoint-context ledger": (ledger.sha256, config.endpoint_context_ledger_sha256),
        "study membership": (memberships.sha256, config.study_membership_sha256),
        "endpoint-context manifest": (
            manifest_snapshot.sha256,
            config.endpoint_context_manifest_sha256,
        ),
        "endpoint-context top manifest": (
            top_snapshot.sha256,
            config.endpoint_context_top_manifest_sha256,
        ),
    }
    for name, (observed, expected) in configured.items():
        if observed != expected:
            raise ValueError(f"{name} hash differs from the frozen config")

    top = _parse_sha256_manifest(top_snapshot.payload, name="endpoint-context top manifest")
    if tuple(top) != _SIDECAR_TOP_ENTRIES:
        raise ValueError("endpoint-context top manifest has an unexpected inventory")
    expected_top = {
        "endpoint_context/endpoint_context_ledger.jsonl": ledger.sha256,
        "endpoint_context/manifest.json": manifest_snapshot.sha256,
        "endpoint_context/study_membership.jsonl": memberships.sha256,
    }
    for entry, digest in expected_top.items():
        if top.get(entry) != digest:
            raise ValueError(f"endpoint-context top manifest does not attest {entry}")

    manifest = _parse_json_object(manifest_snapshot.payload, name="endpoint-context manifest")
    _require_exact_fields(
        manifest,
        expected=_SIDECAR_MANIFEST_FIELDS,
        name="endpoint-context manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["artifact"] != "dramp_endpoint_context_sidecar"
        or manifest["status"]
        != "development_only_pending_reviewed_taxonomy_and_study_component_split"
    ):
        raise ValueError("endpoint-context manifest identity/status mismatch")
    _require_sha256(manifest["config_sha256"], field="endpoint-context config_sha256")

    counts = manifest["counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "unique_sequences",
        "assay_observations",
        "contexts",
        "study_memberships",
    }:
        raise ValueError("endpoint-context manifest counts schema mismatch")
    expected_counts = {
        "unique_sequences": config.expected_unique_sequences,
        "assay_observations": config.expected_assay_observations,
        "study_memberships": config.expected_study_memberships,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("endpoint-context manifest counts differ from the frozen config")
    _require_positive_int(counts.get("contexts"), field="endpoint-context contexts")

    inputs = manifest["input"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "assays",
        "normalized_data_manifest",
        "normalized_summary",
        "sequences",
    }:
        raise ValueError("endpoint-context manifest input schema mismatch")
    for name, entry in inputs.items():
        if not isinstance(entry, dict) or set(entry) != {"filename", "sha256"}:
            raise ValueError(f"endpoint-context manifest input {name!r} schema mismatch")
        _require_nonempty_string(entry["filename"], field=f"endpoint-context input {name} filename")
        _require_sha256(entry["sha256"], field=f"endpoint-context input {name} sha256")
    if inputs["sequences"] != {"filename": sequences.path.name, "sha256": sequences.sha256}:
        raise ValueError("endpoint-context manifest does not attest the supplied sequences")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "audit",
        "contexts",
        "endpoint_context_ledger",
        "study_membership",
    }:
        raise ValueError("endpoint-context manifest artifact schema mismatch")
    for name, entry in artifacts.items():
        if not isinstance(entry, dict) or set(entry) != {"filename", "sha256"}:
            raise ValueError(f"endpoint-context manifest artifact {name!r} schema mismatch")
        _require_nonempty_string(
            entry["filename"], field=f"endpoint-context artifact {name} filename"
        )
        _require_sha256(entry["sha256"], field=f"endpoint-context artifact {name} sha256")
    if artifacts["endpoint_context_ledger"] != {
        "filename": ledger.path.name,
        "sha256": ledger.sha256,
    }:
        raise ValueError("endpoint-context manifest does not attest the supplied ledger")
    if artifacts["study_membership"] != {
        "filename": memberships.path.name,
        "sha256": memberships.sha256,
    }:
        raise ValueError("endpoint-context manifest does not attest supplied study membership")

    policies = manifest["policies"]
    if not isinstance(policies, dict) or set(policies) != _SIDECAR_POLICY_FIELDS:
        raise ValueError("endpoint-context manifest policy schema mismatch")
    if not isinstance(policies["study_keys"], str) or "never a model feature" not in cast(
        str, policies["study_keys"]
    ):
        raise ValueError("endpoint-context manifest does not preserve grouping-only study policy")
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"code_manifest", "git_commit"}:
        raise ValueError("endpoint-context manifest provenance schema mismatch")
    code_entry = provenance["code_manifest"]
    if not isinstance(code_entry, dict) or set(code_entry) != {"filename", "sha256"}:
        raise ValueError("endpoint-context manifest code provenance schema mismatch")
    _require_sha256(code_entry["sha256"], field="endpoint-context code manifest sha256")
    if (
        not isinstance(provenance["git_commit"], str)
        or _GIT_SHA1.fullmatch(provenance["git_commit"]) is None
    ):
        raise ValueError("endpoint-context manifest Git commit is invalid")
    return manifest


def _validate_sequence_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected: int,
) -> dict[str, str]:
    if len(rows) != expected:
        raise ValueError("sequence row count differs from the frozen config")
    sequences: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        name = f"sequence row {index}"
        _require_exact_fields(row, expected=_SEQUENCE_FIELDS, name=name)
        raw_sequence = row["sequence"]
        raw_id = row["sequence_id"]
        if not isinstance(raw_sequence, str) or not isinstance(raw_id, str):
            raise ValueError(f"{name} sequence and sequence_id must be strings")
        sequence = canonicalize_sequence(raw_sequence)
        if raw_id != canonical_sequence_id(sequence):
            raise ValueError(f"{name} sequence_id mismatch")
        if raw_id in sequences:
            raise ValueError(f"{name} repeats sequence_id")
        provenance = row["provenance"]
        if not isinstance(provenance, list) or not provenance:
            raise ValueError(f"{name} provenance must be a non-empty array")
        sequences[raw_id] = sequence
    return sequences


def _string_array(
    value: object,
    *,
    field: str,
    allow_empty: bool,
    require_sorted: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(
            f"{field} must be a {'possibly empty ' if allow_empty else ''}string array"
        )
    if not all(isinstance(item, str) and item and item == item.strip() for item in value):
        raise ValueError(f"{field} contains an invalid string")
    strings = cast(list[str], value)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{field} must contain unique values")
    if require_sorted and strings != sorted(strings):
        raise ValueError(f"{field} must be sorted")
    return tuple(strings)


def _validate_study_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    sequences: Mapping[str, str],
    expected: int,
) -> tuple[
    dict[str, set[str]],
    Counter[str],
    dict[str, dict[str, object]],
]:
    if len(rows) != expected:
        raise ValueError("study membership row count differs from the frozen config")
    key_sequences: dict[str, set[str]] = defaultdict(set)
    key_memberships: Counter[str] = Counter()
    provenance_owner: dict[str, str] = {}
    covered: set[str] = set()
    for index, row in enumerate(rows, start=1):
        name = f"study membership row {index}"
        _require_exact_fields(row, expected=_STUDY_FIELDS, name=name)
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"{name} schema_version must be 1")
        sequence_id = _require_sha256(row["sequence_id"], field=f"{name} sequence_id")
        if sequence_id not in sequences:
            raise ValueError(f"{name} references an unknown sequence")
        provenance_id = _require_sha256(row["provenance_id"], field=f"{name} provenance_id")
        if provenance_id in provenance_owner:
            raise ValueError(f"{name} repeats provenance_id")
        provenance_owner[provenance_id] = sequence_id
        for field in ("source", "source_version", "source_record_id", "source_record_key"):
            _require_nonempty_string(row[field], field=f"{name} {field}")
        _require_sha256(row["source_sha256"], field=f"{name} source_sha256")
        _require_positive_int(row["source_row_number"], field=f"{name} source_row_number")
        if row["study_status"] not in _STUDY_STATUSES:
            raise ValueError(f"{name} has an invalid study_status")
        source_record_key = cast(str, row["source_record_key"])
        if source_record_key != f"source-record:{provenance_id}":
            raise ValueError(f"{name} source_record_key does not bind provenance_id")
        study_keys = _string_array(row["study_keys"], field=f"{name} study_keys", allow_empty=False)
        pmids = _string_array(row["pmids"], field=f"{name} pmids", allow_empty=True)
        if any(re.fullmatch(r"[1-9][0-9]*", pmid) is None for pmid in pmids):
            raise ValueError(f"{name} pmids must contain positive decimal identifiers")
        if tuple(sorted(pmids, key=int)) != pmids:
            raise ValueError(f"{name} pmids must be numerically sorted")
        ignored_tokens = _string_array(
            row["ignored_pubmed_tokens"],
            field=f"{name} ignored_pubmed_tokens",
            allow_empty=True,
            require_sorted=True,
        )
        _string_array(
            row["study_review_codes"],
            field=f"{name} study_review_codes",
            allow_empty=True,
            require_sorted=True,
        )
        for field in ("citation_reference", "citation_title"):
            value = row[field]
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"{name} {field} must be null or a non-empty string")
        reference = cast(str | None, row["citation_reference"])
        title = cast(str | None, row["citation_title"])
        status = cast(str, row["study_status"])
        if pmids:
            if study_keys != tuple(f"pmid:{pmid}" for pmid in pmids):
                raise ValueError(f"{name} PMID study keys mismatch")
            expected_status = (
                "explicit_pmid_with_ignored_tokens" if ignored_tokens else "explicit_pmid"
            )
            if status != expected_status:
                raise ValueError(f"{name} explicit PMID status mismatch")
        elif reference is not None or title is not None:
            expected_key = "reference-title:" + _stable_digest(
                "amp-challenge:reference-title-study:v1",
                reference or "",
                title or "",
            )
            if study_keys != (expected_key,) or status != "reference_title_fallback":
                raise ValueError(f"{name} reference-title study identity mismatch")
        elif study_keys != (source_record_key,) or status != "source_record_singleton":
            raise ValueError(f"{name} source-record singleton study identity mismatch")
        covered.add(sequence_id)
        for study_key in study_keys:
            key_sequences[study_key].add(sequence_id)
            key_memberships[study_key] += 1
    if covered != set(sequences):
        raise ValueError("study memberships must cover every sequence exactly as a set")
    singleton_keys = [key for key in key_sequences if key.startswith("source-record:")]
    if any(len(key_sequences[key]) != 1 for key in singleton_keys):
        raise ValueError("source-record study keys must be sequence singletons")
    membership_by_provenance = {
        cast(str, row["provenance_id"]): {
            "sequence_id": row["sequence_id"],
            "source_record_id": row["source_record_id"],
            "source_record_key": row["source_record_key"],
            "source_row_number": row["source_row_number"],
            "study_status": row["study_status"],
            "study_keys": tuple(cast(list[str], row["study_keys"])),
            "study_review_codes": tuple(cast(list[str], row["study_review_codes"])),
        }
        for row in rows
    }
    return dict(key_sequences), key_memberships, membership_by_provenance


def _validate_measurement(value: object, *, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    _require_exact_fields(value, expected=_MEASUREMENT_FIELDS, name=name)


def _exposure_identity(value: object) -> tuple[object, ...] | None:
    if value is None:
        return None
    exposure = cast(Mapping[str, object], value)
    return (
        exposure["relation"],
        exposure["lower"],
        exposure["lower_inclusive"],
        exposure["upper"],
        exposure["upper_inclusive"],
        exposure["unit"],
    )


def _validate_ledger_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    sequences: Mapping[str, str],
    membership_by_provenance: Mapping[str, Mapping[str, object]],
    expected: int,
) -> tuple[dict[str, Counter[str]], dict[str, object]]:
    if len(rows) != expected:
        raise ValueError("endpoint-context ledger row count differs from the frozen config")
    observation_ids: set[str] = set()
    all_context_ids: set[str] = set()
    non_mic_context_ids: set[str] = set()
    non_mic_observations = 0
    context_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        name = f"endpoint-context ledger row {index}"
        _require_exact_fields(row, expected=_LEDGER_FIELDS, name=name)
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"{name} schema_version must be 1")
        sequence_id = _require_sha256(row["sequence_id"], field=f"{name} sequence_id")
        if sequence_id not in sequences:
            raise ValueError(f"{name} references an unknown sequence")
        for field in (
            "observation_id",
            "assay_row_sha256",
            "assay_context_id",
            "context_id",
            "provenance_id",
        ):
            _require_sha256(row[field], field=f"{name} {field}")
        observation_id = cast(str, row["observation_id"])
        if observation_id in observation_ids:
            raise ValueError(f"{name} repeats observation_id")
        observation_ids.add(observation_id)
        if row["endpoint"] not in _ENDPOINTS:
            raise ValueError(f"{name} has an invalid endpoint")
        assay_context_id = cast(str, row["assay_context_id"])
        all_context_ids.add(assay_context_id)
        provenance_id = cast(str, row["provenance_id"])
        membership = membership_by_provenance.get(provenance_id)
        if membership is None:
            raise ValueError(f"{name} references unknown study provenance")
        membership_fields = {
            "sequence_id": row["sequence_id"],
            "source_record_id": row["source_record_id"],
            "source_record_key": row["source_record_key"],
            "source_row_number": row["source_row_number"],
            "study_status": row["study_status"],
            "study_keys": tuple(cast(list[str], row["study_keys"])),
            "study_review_codes": tuple(cast(list[str], row["study_review_codes"])),
        }
        if membership_fields != membership:
            raise ValueError(f"{name} differs from its study membership")
        _validate_measurement(row["measurement"], name=f"{name} measurement")
        exposure = row["exposure_concentration"]
        if exposure is not None:
            _validate_measurement(exposure, name=f"{name} exposure_concentration")
        eligible_tasks = _string_array(
            row["eligible_tasks"],
            field=f"{name} eligible_tasks",
            allow_empty=True,
            require_sorted=True,
        )
        _string_array(
            row["exclusion_codes"],
            field=f"{name} exclusion_codes",
            allow_empty=True,
            require_sorted=True,
        )
        _string_array(row["study_keys"], field=f"{name} study_keys", allow_empty=False)
        _string_array(
            row["study_review_codes"],
            field=f"{name} study_review_codes",
            allow_empty=True,
            require_sorted=True,
        )
        _string_array(row["source_conditions"], field=f"{name} source_conditions", allow_empty=True)
        _string_array(
            row["composite_reason_codes"],
            field=f"{name} composite_reason_codes",
            allow_empty=True,
            require_sorted=True,
        )
        _string_array(
            row["strain_identifier_review_ids"],
            field=f"{name} strain_identifier_review_ids",
            allow_empty=True,
            require_sorted=True,
        )
        is_binary_eligible = "bacterial_mic16" in eligible_tasks
        if is_binary_eligible and (
            row["endpoint"] != "mic"
            or row["mapping_status"] != "mapped_single_supported_species"
            or row["gram_resolution"] != "concordant"
            or row["source_gram"] not in {"negative", "positive"}
            or isinstance(row["mic16_label"], bool)
            or row["mic16_label"] not in {0, 1}
            or not isinstance(row["canonical_target"], str)
            or not cast(str, row["canonical_target"])
        ):
            raise ValueError(f"{name} has inconsistent bacterial_mic16 eligibility")
        if row["endpoint"] == "mic":
            context_rows[assay_context_id].append(row)
        else:
            non_mic_observations += 1
            non_mic_context_ids.add(assay_context_id)

    if set(context_rows) & non_mic_context_ids:
        raise ValueError("one assay_context_id spans MIC and non-MIC endpoints")

    counts_by_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    mixed_eligibility: list[dict[str, object]] = []
    eligible_conflicts: list[dict[str, object]] = []
    repeated_contexts = 0
    all_ineligible_contexts = 0
    all_ineligible_observations = 0
    fully_eligible_contexts = 0
    eligible_raw_observations = 0
    retained_contexts = 0
    retained_observations = 0
    retained_sequences: set[str] = set()
    for assay_context_id in sorted(context_rows):
        members = sorted(
            context_rows[assay_context_id], key=lambda row: cast(str, row["observation_id"])
        )
        repeated_contexts += int(len(members) > 1)
        first = members[0]
        semantic = (
            first["sequence_id"],
            first["endpoint"],
            first["context_id"],
            tuple(cast(list[str], first["source_conditions"])),
            _exposure_identity(first["exposure_concentration"]),
        )
        for member in members[1:]:
            current = (
                member["sequence_id"],
                member["endpoint"],
                member["context_id"],
                tuple(cast(list[str], member["source_conditions"])),
                _exposure_identity(member["exposure_concentration"]),
            )
            if current != semantic:
                raise ValueError(f"assay_context_id {assay_context_id} has inconsistent semantics")
        eligibility = ["bacterial_mic16" in member["eligible_tasks"] for member in members]
        eligible_count = sum(eligibility)
        eligible_raw_observations += eligible_count
        if eligible_count == 0:
            all_ineligible_contexts += 1
            all_ineligible_observations += len(members)
            continue
        if eligible_count != len(members):
            mixed_eligibility.append(
                {
                    "assay_context_id": assay_context_id,
                    "observations": len(members),
                    "eligible_observations": eligible_count,
                    "ineligible_observations": len(members) - eligible_count,
                    "sequence_id": first["sequence_id"],
                }
            )
            continue
        fully_eligible_contexts += 1
        labels = {cast(int, member["mic16_label"]) for member in members}
        grams = {cast(str, member["source_gram"]) for member in members}
        reason_codes: list[str] = []
        if len(labels) != 1:
            reason_codes.append("mic16_label_disagreement")
        if len(grams) != 1:
            reason_codes.append("source_gram_disagreement")
        if reason_codes:
            eligible_conflicts.append(
                {
                    "assay_context_id": assay_context_id,
                    "observations": len(members),
                    "sequence_id": first["sequence_id"],
                    "reason_codes": reason_codes,
                    "mic16_labels": sorted(labels),
                    "source_grams": sorted(grams),
                }
            )
            continue
        retained_contexts += 1
        retained_observations += len(members)
        sequence_id = cast(str, first["sequence_id"])
        retained_sequences.add(sequence_id)
        gram = cast(str, first["source_gram"])
        label = next(iter(labels))
        metric = f"gram_{gram}_mic16_{'positive' if label == 1 else 'negative'}"
        counts_by_sequence[sequence_id][metric] += 1
    audit = {
        "ledger_raw_observations": len(rows),
        "ledger_assay_contexts": len(all_context_ids),
        "mic_raw_observations": sum(len(value) for value in context_rows.values()),
        "mic_assay_contexts": len(context_rows),
        "non_mic_raw_observations": non_mic_observations,
        "non_mic_assay_contexts": len(non_mic_context_ids),
        "repeated_mic_assay_contexts": repeated_contexts,
        "eligible_raw_observations": eligible_raw_observations,
        "candidate_assay_contexts": fully_eligible_contexts + len(mixed_eligibility),
        "all_ineligible_assay_contexts": all_ineligible_contexts,
        "observations_in_all_ineligible_assay_contexts": all_ineligible_observations,
        "fully_eligible_assay_contexts": fully_eligible_contexts,
        "mixed_eligibility_assay_contexts": len(mixed_eligibility),
        "observations_in_mixed_eligibility_assay_contexts": sum(
            cast(int, value["observations"]) for value in mixed_eligibility
        ),
        "eligible_observations_in_mixed_eligibility_assay_contexts": sum(
            cast(int, value["eligible_observations"]) for value in mixed_eligibility
        ),
        "excluded_mixed_eligibility": mixed_eligibility,
        "conflicting_assay_contexts": len(eligible_conflicts),
        "eligible_label_conflict_contexts": sum(
            "mic16_label_disagreement" in cast(list[str], value["reason_codes"])
            for value in eligible_conflicts
        ),
        "eligible_gram_conflict_contexts": sum(
            "source_gram_disagreement" in cast(list[str], value["reason_codes"])
            for value in eligible_conflicts
        ),
        "observations_in_conflicting_assay_contexts": sum(
            cast(int, value["observations"]) for value in eligible_conflicts
        ),
        "excluded_conflicts": eligible_conflicts,
        "retained_assay_contexts": retained_contexts,
        "retained_source_observations": retained_observations,
        "retained_sequences": len(retained_sequences),
        "retained_class_counts": {
            metric: sum(counts[metric] for counts in counts_by_sequence.values())
            for metric in _CLASS_METRICS
        },
    }
    return dict(counts_by_sequence), audit


def _sequence_set_sha256(sequence_ids: Sequence[str]) -> str:
    payload = ("".join(f"{sequence_id}\n" for sequence_id in sorted(sequence_ids))).encode("ascii")
    return _sha256_bytes(payload)


def _build_components(
    *,
    sequences: Mapping[str, str],
    study_key_sequences: Mapping[str, set[str]],
    study_key_memberships: Mapping[str, int],
    balance_by_sequence: Mapping[str, Counter[str]],
    identity_threshold: float,
) -> tuple[list[_GraphComponent], list[dict[str, object]], dict[str, str]]:
    homology = cluster_sequences(sequences.values(), identity_threshold=identity_threshold)
    homology_members: dict[str, tuple[str, ...]] = {}
    homology_by_sequence: dict[str, str] = {}
    edges: list[dict[str, object]] = []
    dsu = _DisjointSet(sequences)
    for component in homology:
        members = tuple(sorted(canonical_sequence_id(sequence) for sequence in component))
        component_id = _stable_digest(
            "amp-challenge:homology-component:v1",
            _HOMOLOGY_ALGORITHM,
            identity_threshold,
            members,
        )
        homology_members[component_id] = members
        for sequence_id in members:
            homology_by_sequence[sequence_id] = component_id
        for sequence_id in members[1:]:
            dsu.union(members[0], sequence_id)
        edge_id = _stable_digest(
            "amp-challenge:grouping-edge:v1",
            "homology_component",
            _HOMOLOGY_ALGORITHM,
            identity_threshold,
            component_id,
            members,
        )
        edges.append(
            {
                "schema_version": 1,
                "edge_id": edge_id,
                "edge_type": "homology_component",
                "group_key": component_id,
                "sequence_ids": list(members),
                "sequence_count": len(members),
                "source_membership_count": None,
            }
        )

    for study_key in sorted(study_key_sequences):
        members = tuple(sorted(study_key_sequences[study_key]))
        for sequence_id in members[1:]:
            dsu.union(members[0], sequence_id)
        edge_id = _stable_digest(
            "amp-challenge:grouping-edge:v1",
            "study_key",
            study_key,
            members,
            study_key_memberships[study_key],
        )
        edges.append(
            {
                "schema_version": 1,
                "edge_id": edge_id,
                "edge_type": "study_key",
                "group_key": study_key,
                "sequence_ids": list(members),
                "sequence_count": len(members),
                "source_membership_count": study_key_memberships[study_key],
            }
        )
    edges.sort(key=lambda row: (cast(str, row["edge_type"]), cast(str, row["group_key"])))

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for sequence_id in sorted(sequences):
        members_by_root[dsu.find(sequence_id)].append(sequence_id)
    edge_by_id = {cast(str, edge["edge_id"]): edge for edge in edges}
    edge_ids_by_sequence: dict[str, set[str]] = defaultdict(set)
    for edge_id, edge in edge_by_id.items():
        for sequence_id in cast(list[str], edge["sequence_ids"]):
            edge_ids_by_sequence[sequence_id].add(edge_id)

    components: list[_GraphComponent] = []
    for members_list in members_by_root.values():
        members = tuple(sorted(members_list))
        union_component_id = _stable_digest(
            "amp-challenge:homology-study-component:v1",
            _COMPONENT_POLICY,
            _HOMOLOGY_ALGORITHM,
            identity_threshold,
            members,
        )
        homology_ids = tuple(sorted({homology_by_sequence[item] for item in members}))
        study_keys = tuple(
            sorted(
                key
                for key, key_members in study_key_sequences.items()
                if any(item in key_members for item in members)
            )
        )
        grouping_edge_ids = tuple(
            sorted({edge_id for item in members for edge_id in edge_ids_by_sequence[item]})
        )
        balance = Counter({"component_count": 1, "sequences": len(members)})
        for sequence_id in members:
            balance.update(balance_by_sequence.get(sequence_id, {}))
        balance_counts = {metric: int(balance[metric]) for metric in _BALANCE_METRICS}
        components.append(
            _GraphComponent(
                union_component_id=union_component_id,
                sequence_ids=members,
                homology_component_ids=homology_ids,
                study_keys=study_keys,
                grouping_edge_ids=grouping_edge_ids,
                balance_counts=balance_counts,
            )
        )
    components.sort(key=lambda value: value.union_component_id)
    return components, edges, homology_by_sequence


def _objective_score(
    loads: Mapping[int, Mapping[str, int]],
    *,
    totals: Mapping[str, int],
    weights: Mapping[str, float],
    folds: int,
) -> float:
    score = 0.0
    for metric in _BALANCE_METRICS:
        target = totals[metric] / folds
        score += weights[metric] * sum(
            ((loads[fold][metric] - target) / target) ** 2 for fold in range(folds)
        )
    return score


def _assign_folds(
    components: Sequence[_GraphComponent],
    *,
    config: HomologyStudySplitConfig,
) -> tuple[dict[str, int], dict[int, dict[str, int]], dict[str, int], float]:
    if len(components) < config.folds:
        raise ValueError("cannot create nonempty folds from fewer union components than folds")
    totals = {
        metric: sum(component.balance_counts[metric] for component in components)
        for metric in _BALANCE_METRICS
    }
    zero_metrics = [metric for metric in _BALANCE_METRICS if totals[metric] <= 0]
    if zero_metrics:
        raise ValueError(f"balance dimensions have zero corpus total: {zero_metrics}")
    for metric in _CLASS_METRICS:
        minimum = config.balance_minima[metric]
        supporting_components = sum(
            component.balance_counts[metric] > 0 for component in components
        )
        if totals[metric] < minimum * config.folds or (
            minimum > 0 and supporting_components < config.folds
        ):
            raise ValueError(f"balance minimum for {metric} is infeasible")

    def difficulty(component: _GraphComponent) -> float:
        return max(
            component.balance_counts[metric] / (totals[metric] / config.folds)
            for metric in _BALANCE_METRICS
        )

    ordered = sorted(
        components,
        key=lambda component: (
            -difficulty(component),
            -len(component.sequence_ids),
            _stable_digest(
                "homology-study-split-v1:component-order",
                config.seed,
                component.union_component_id,
            ),
        ),
    )
    loads: dict[int, dict[str, int]] = {
        fold: {metric: 0 for metric in _BALANCE_METRICS} for fold in range(config.folds)
    }
    assignments: dict[str, int] = {}
    for index, component in enumerate(ordered):
        candidates = (
            [fold for fold in range(config.folds) if loads[fold]["component_count"] == 0]
            if index < config.folds
            else list(range(config.folds))
        )
        scored: list[tuple[float, str, int]] = []
        for fold in candidates:
            projected = {key: dict(value) for key, value in loads.items()}
            for metric in _BALANCE_METRICS:
                projected[fold][metric] += component.balance_counts[metric]
            score = _objective_score(
                projected,
                totals=totals,
                weights=config.balance_weights,
                folds=config.folds,
            )
            tie = _stable_digest(
                "homology-study-split-v1:fold-placement",
                config.seed,
                component.union_component_id,
                fold,
            )
            scored.append((score, tie, fold))
        _, _, chosen = min(scored)
        assignments[component.union_component_id] = chosen
        for metric in _BALANCE_METRICS:
            loads[chosen][metric] += component.balance_counts[metric]

    if any(loads[fold]["component_count"] == 0 for fold in range(config.folds)):
        raise AssertionError("fold assignment produced an empty fold")
    for fold in range(config.folds):
        for metric in _CLASS_METRICS:
            if loads[fold][metric] < config.balance_minima[metric]:
                raise ValueError(
                    f"deterministic assignment misses balance minimum for fold {fold}/{metric}"
                )
    score = _objective_score(
        loads,
        totals=totals,
        weights=config.balance_weights,
        folds=config.folds,
    )
    return assignments, loads, totals, score


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def build_homology_study_split(
    *,
    sequences_path: str | Path,
    endpoint_context_ledger_path: str | Path,
    study_membership_path: str | Path,
    endpoint_context_manifest_path: str | Path,
    endpoint_context_top_manifest_path: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> HomologyStudySplitExecution:
    """Build and atomically publish a full-union component split."""

    if _GIT_SHA1.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase 40-character Git SHA")
    requested_output = Path(output_dir)
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse homology-study split output: {requested_output}")
    output = requested_output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse homology-study split output: {output}")

    snapshots = {
        "config": _read_snapshot(config_path, name="homology-study split config"),
        "sequences": _read_snapshot(sequences_path, name="normalized sequences"),
        "ledger": _read_snapshot(endpoint_context_ledger_path, name="endpoint-context ledger"),
        "memberships": _read_snapshot(study_membership_path, name="study membership"),
        "sidecar_manifest": _read_snapshot(
            endpoint_context_manifest_path, name="endpoint-context manifest"
        ),
        "sidecar_top": _read_snapshot(
            endpoint_context_top_manifest_path, name="endpoint-context top manifest"
        ),
        "code_manifest": _read_snapshot(code_manifest_path, name="code manifest"),
    }
    config = _config_from_payload(snapshots["config"].path, snapshots["config"].payload)
    code_snapshots = _validate_code_manifest(
        snapshots["code_manifest"],
        config=config,
        config_snapshot=snapshots["config"],
    )
    sidecar_manifest = _validate_sidecar_attestation(
        config=config,
        sequences=snapshots["sequences"],
        ledger=snapshots["ledger"],
        memberships=snapshots["memberships"],
        manifest_snapshot=snapshots["sidecar_manifest"],
        top_snapshot=snapshots["sidecar_top"],
    )

    sequence_rows = _parse_jsonl(snapshots["sequences"].payload, name="normalized sequences")
    sequences = _validate_sequence_rows(
        sequence_rows,
        expected=config.expected_unique_sequences,
    )
    study_rows = _parse_jsonl(snapshots["memberships"].payload, name="study membership")
    study_key_sequences, study_key_memberships, membership_by_provenance = _validate_study_rows(
        study_rows,
        sequences=sequences,
        expected=config.expected_study_memberships,
    )
    ledger_rows = _parse_jsonl(snapshots["ledger"].payload, name="endpoint-context ledger")
    balance_by_sequence, balance_audit = _validate_ledger_rows(
        ledger_rows,
        sequences=sequences,
        membership_by_provenance=membership_by_provenance,
        expected=config.expected_assay_observations,
    )

    components, grouping_edges, homology_by_sequence = _build_components(
        sequences=sequences,
        study_key_sequences=study_key_sequences,
        study_key_memberships=study_key_memberships,
        balance_by_sequence=balance_by_sequence,
        identity_threshold=config.identity_threshold,
    )
    fold_by_component, fold_loads, balance_totals, objective_score = _assign_folds(
        components,
        config=config,
    )
    component_by_sequence = {
        sequence_id: component.union_component_id
        for component in components
        for sequence_id in component.sequence_ids
    }
    sequence_assignments = [
        {
            "schema_version": 1,
            "sequence_id": sequence_id,
            "homology_component_id": homology_by_sequence[sequence_id],
            "union_component_id": component_by_sequence[sequence_id],
            "fold": fold_by_component[component_by_sequence[sequence_id]],
        }
        for sequence_id in sorted(sequences)
    ]
    component_rows = [
        {
            "schema_version": 1,
            "union_component_id": component.union_component_id,
            "fold": fold_by_component[component.union_component_id],
            "sequence_count": len(component.sequence_ids),
            "sequence_ids_sha256": _sequence_set_sha256(component.sequence_ids),
            "grouping_edge_ids": list(component.grouping_edge_ids),
            "homology_component_count": len(component.homology_component_ids),
            "study_key_count": len(component.study_keys),
            "balance_counts": dict(component.balance_counts),
        }
        for component in components
    ]

    assignment_by_sequence = {row["sequence_id"]: row for row in sequence_assignments}
    homology_edges = [row for row in grouping_edges if row["edge_type"] == "homology_component"]
    study_edges = [row for row in grouping_edges if row["edge_type"] == "study_key"]
    all_edges_single_fold = all(
        len({assignment_by_sequence[item]["fold"] for item in row["sequence_ids"]}) == 1
        for row in grouping_edges
    )
    all_edges_single_component = all(
        len({assignment_by_sequence[item]["union_component_id"] for item in row["sequence_ids"]})
        == 1
        for row in grouping_edges
    )
    component_sizes = Counter(len(component.sequence_ids) for component in components)
    largest = max(components, key=lambda value: (len(value.sequence_ids), value.union_component_id))
    invariants = {
        "all_sequences_assigned_once": len(sequence_assignments) == len(sequences)
        and set(assignment_by_sequence) == set(sequences),
        "all_homology_and_study_edges_within_one_component": all_edges_single_component,
        "all_homology_and_study_edges_within_one_fold": all_edges_single_fold,
        "all_components_have_one_fold": len(fold_by_component) == len(components),
        "all_folds_nonempty": set(fold_by_component.values()) == set(range(config.folds)),
        "study_memberships_cover_all_sequences": {row["sequence_id"] for row in study_rows}
        == set(sequences),
        "source_record_study_keys_are_singletons": all(
            len(study_key_sequences[key]) == 1
            for key in study_key_sequences
            if key.startswith("source-record:")
        ),
        "all_mic_contexts_partitioned": cast(int, balance_audit["mic_assay_contexts"])
        == cast(int, balance_audit["all_ineligible_assay_contexts"])
        + cast(int, balance_audit["mixed_eligibility_assay_contexts"])
        + cast(int, balance_audit["fully_eligible_assay_contexts"]),
        "all_ledger_endpoints_partitioned": cast(int, balance_audit["ledger_raw_observations"])
        == cast(int, balance_audit["mic_raw_observations"])
        + cast(int, balance_audit["non_mic_raw_observations"]),
        "all_ledger_contexts_partitioned": cast(int, balance_audit["ledger_assay_contexts"])
        == cast(int, balance_audit["mic_assay_contexts"])
        + cast(int, balance_audit["non_mic_assay_contexts"]),
        "all_mic_observations_partitioned": cast(int, balance_audit["mic_raw_observations"])
        == cast(int, balance_audit["observations_in_all_ineligible_assay_contexts"])
        + cast(int, balance_audit["observations_in_mixed_eligibility_assay_contexts"])
        + cast(int, balance_audit["observations_in_conflicting_assay_contexts"])
        + cast(int, balance_audit["retained_source_observations"]),
        "candidate_binary_contexts_partitioned": cast(
            int, balance_audit["candidate_assay_contexts"]
        )
        == cast(int, balance_audit["mixed_eligibility_assay_contexts"])
        + cast(int, balance_audit["conflicting_assay_contexts"])
        + cast(int, balance_audit["retained_assay_contexts"]),
        "fully_eligible_binary_contexts_partitioned": cast(
            int, balance_audit["fully_eligible_assay_contexts"]
        )
        == cast(int, balance_audit["conflicting_assay_contexts"])
        + cast(int, balance_audit["retained_assay_contexts"]),
        "eligible_binary_observations_partitioned": cast(
            int, balance_audit["eligible_raw_observations"]
        )
        == cast(int, balance_audit["eligible_observations_in_mixed_eligibility_assay_contexts"])
        + cast(int, balance_audit["observations_in_conflicting_assay_contexts"])
        + cast(int, balance_audit["retained_source_observations"]),
        "balance_totals_equal_component_sum": all(
            balance_totals[metric]
            == sum(component.balance_counts[metric] for component in components)
            for metric in _BALANCE_METRICS
        ),
        "balance_class_totals_equal_retained_context_census": all(
            balance_totals[metric]
            == cast(dict[str, int], balance_audit["retained_class_counts"])[metric]
            for metric in _CLASS_METRICS
        ),
        "all_balance_minima_met": all(
            fold_loads[fold][metric] >= config.balance_minima[metric]
            for fold in range(config.folds)
            for metric in _CLASS_METRICS
        ),
        "input_snapshots_unchanged": True,
        "code_inventory_unchanged": True,
    }
    if not all(invariants.values()):
        raise AssertionError("homology-study split invariant failed")

    for name, snapshot in snapshots.items():
        _assert_snapshot_unchanged(snapshot, name=name)
    for snapshot in code_snapshots:
        _assert_snapshot_unchanged(snapshot, name="code inventory")

    audit = {
        "schema_version": 1,
        "artifact": "homology_study_union_split_audit",
        "status": "development_split_not_an_untouched_evaluation_panel",
        "input": {
            "unique_sequences": len(sequences),
            "assay_observations": len(ledger_rows),
            "study_memberships": len(study_rows),
        },
        "graph": {
            "component_policy": _COMPONENT_POLICY,
            "homology_algorithm": _HOMOLOGY_ALGORITHM,
            "identity_threshold": config.identity_threshold,
            "homology_components": len(homology_edges),
            "study_keys": len(study_edges),
            "shared_study_keys": sum(row["sequence_count"] > 1 for row in study_edges),
            "study_keys_merging_homology_components": sum(
                len({homology_by_sequence[item] for item in row["sequence_ids"]}) > 1
                for row in study_edges
            ),
            "union_components": len(components),
            "component_size_histogram": {
                str(size): count for size, count in sorted(component_sizes.items())
            },
            "largest_component": {
                "union_component_id": largest.union_component_id,
                "sequences": len(largest.sequence_ids),
                "homology_components": len(largest.homology_component_ids),
                "study_keys": len(largest.study_keys),
                "balance_counts": dict(largest.balance_counts),
            },
        },
        "balance": {
            "policy": _BALANCE_POLICY,
            "folds": config.folds,
            "seed": config.seed,
            "weights": dict(config.balance_weights),
            "minimum_per_fold": dict(config.balance_minima),
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
            "context_census": balance_audit,
            "totals": balance_totals,
            "targets_per_fold": {
                metric: balance_totals[metric] / config.folds for metric in _BALANCE_METRICS
            },
            "objective_score": objective_score,
            "objective_score_hex": objective_score.hex(),
            "by_fold": {str(fold): fold_loads[fold] for fold in range(config.folds)},
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

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        artifact_paths = {
            "sequence_assignments": staging / "sequence_assignments.jsonl",
            "components": staging / "components.jsonl",
            "grouping_edges": staging / "grouping_edges.jsonl",
            "audit": staging / "audit.json",
        }
        _write_jsonl(artifact_paths["sequence_assignments"], sequence_assignments)
        _write_jsonl(artifact_paths["components"], component_rows)
        _write_jsonl(artifact_paths["grouping_edges"], grouping_edges)
        _write_json(artifact_paths["audit"], audit)
        manifest = {
            "schema_version": 1,
            "artifact": "homology_study_union_split",
            "status": "development_split_not_an_untouched_evaluation_panel",
            "config_sha256": snapshots["config"].sha256,
            "input": {
                "sequences": {
                    "filename": snapshots["sequences"].path.name,
                    "sha256": snapshots["sequences"].sha256,
                },
                "endpoint_context_ledger": {
                    "filename": snapshots["ledger"].path.name,
                    "sha256": snapshots["ledger"].sha256,
                },
                "study_membership": {
                    "filename": snapshots["memberships"].path.name,
                    "sha256": snapshots["memberships"].sha256,
                },
                "endpoint_context_manifest": {
                    "filename": snapshots["sidecar_manifest"].path.name,
                    "sha256": snapshots["sidecar_manifest"].sha256,
                },
                "endpoint_context_top_manifest": {
                    "filename": snapshots["sidecar_top"].path.name,
                    "sha256": snapshots["sidecar_top"].sha256,
                },
            },
            "upstream": {
                "artifact": sidecar_manifest["artifact"],
                "status": sidecar_manifest["status"],
                "git_commit": cast(dict[str, object], sidecar_manifest["provenance"])["git_commit"],
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
            "counts": {
                "unique_sequences": len(sequences),
                "homology_components": len(homology_edges),
                "study_keys": len(study_edges),
                "union_components": len(components),
                "folds": config.folds,
                "sequence_assignments": len(sequence_assignments),
                "grouping_edges": len(grouping_edges),
                "retained_balance_contexts": balance_audit["retained_assay_contexts"],
                "excluded_mixed_eligibility_balance_contexts": balance_audit[
                    "mixed_eligibility_assay_contexts"
                ],
                "excluded_conflicting_balance_contexts": balance_audit[
                    "conflicting_assay_contexts"
                ],
            },
            "artifacts": {
                name: {"filename": path.name, "sha256": _sha256_bytes(path.read_bytes())}
                for name, path in artifact_paths.items()
            },
            "provenance": {
                "git_commit": git_commit,
                "code_manifest": {
                    "filename": snapshots["code_manifest"].path.name,
                    "sha256": snapshots["code_manifest"].sha256,
                },
            },
            "runtime": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            },
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        for name, snapshot in snapshots.items():
            _assert_snapshot_unchanged(snapshot, name=name)
        for snapshot in code_snapshots:
            _assert_snapshot_unchanged(snapshot, name="code inventory")
        if os.path.lexists(output):
            raise FileExistsError(f"homology-study split output appeared during build: {output}")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return HomologyStudySplitExecution(
        output_dir=output,
        sequence_assignments_path=output / "sequence_assignments.jsonl",
        components_path=output / "components.jsonl",
        grouping_edges_path=output / "grouping_edges.jsonl",
        audit_path=output / "audit.json",
        manifest_path=output / "manifest.json",
        unique_sequences=len(sequences),
        homology_components=len(homology_edges),
        union_components=len(components),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--endpoint-context-ledger", type=Path, required=True)
    parser.add_argument("--study-membership", type=Path, required=True)
    parser.add_argument("--endpoint-context-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-context-top-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_homology_study_split(
        sequences_path=args.sequences,
        endpoint_context_ledger_path=args.endpoint_context_ledger,
        study_membership_path=args.study_membership,
        endpoint_context_manifest_path=args.endpoint_context_manifest,
        endpoint_context_top_manifest_path=args.endpoint_context_top_manifest,
        config_path=args.config,
        code_manifest_path=args.code_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "homology_components": result.homology_components,
                "union_components": result.union_components,
                "unique_sequences": result.unique_sequences,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

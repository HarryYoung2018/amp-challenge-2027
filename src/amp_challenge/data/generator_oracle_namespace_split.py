"""Build the leakage-safe generator/oracle namespace split v1.

The partition decision sees only sequence-to-union membership and endpoint names.
Source fold, role, weight, assay value, relation, and label fields are never used.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import stat
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_JOB = re.compile(r"[1-9][0-9]{0,19}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_ENDPOINT = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{8,50}")
_SAFETY = frozenset({"hc50", "hemolysis_percent"})
_MAX_CONFIG_BYTES = 64 * 1024
_LIMIT_CEILINGS = {
    "maximum_input_file_bytes": 33_554_432,
    "maximum_input_records": 10_000,
    "maximum_output_file_bytes": 16_777_216,
    "maximum_output_records": 4_096,
    "maximum_study_keys_per_sequence": 64,
    "maximum_json_depth": 16,
    "maximum_json_containers": 4_096,
    "maximum_json_string_bytes": 65_536,
}
_AUTHORITIES = {
    "parser_v7_job": 222995,
    "endpoint_context_job": 223141,
    "endpoint_context_audit_job": 223143,
    "homology_study_split_job": 223222,
    "homology_study_split_audit_job": 223225,
    "categorical_corpus_job": 223443,
    "categorical_corpus_audit_job": 223445,
    "gate1_union_job": 223248,
    "gate1_union_audit_job": 223250,
}
_TRIPLES = ["012", "013", "014", "023", "024", "034", "123", "124", "134", "234"]
_EXPECTED_FOLD_FIELDS = {
    "generator_fold_sequences",
    "generator_fold_union_components",
    "generator_fold_homology_components",
    "generator_fold_study_keys",
}
_EXPECTED_INTEGER_FIELDS = {
    "total_sequences",
    "total_assay_observations",
    "total_endpoint_ledger_rows",
    "total_union_components",
    "total_homology_components",
    "total_study_keys",
    "total_study_membership_rows",
    "total_gate1_contexts",
    "total_gate1_sequences",
    "total_gate1_union_components",
    "generator_sequences",
    "generator_union_components",
    "generator_homology_components",
    "generator_study_keys",
    "generator_study_membership_rows",
    "oracle_sequences",
    "oracle_union_components",
    "oracle_homology_components",
    "oracle_study_keys",
    "oracle_study_membership_rows",
    "activity_contexts",
    "activity_sequences",
    "activity_union_components",
    "activity_homology_components",
    "activity_positive",
    "activity_negative",
    "activity_gram_positive",
    "activity_gram_negative",
    "hc50_observations",
    "hc50_sequences",
    "hc50_union_components",
    "hc50_homology_components",
    "hemolysis_percent_observations",
    "hemolysis_percent_sequences",
    "hemolysis_percent_union_components",
    "hemolysis_percent_homology_components",
    "any_safety_sequences",
    "any_safety_union_components",
    "hc50_hemolysis_sequence_overlap",
}
_ACTIVITY_TARGETS = {
    "enterococcus_faecalis",
    "enterococcus_faecium",
    "escherichia_coli",
    "klebsiella_pneumoniae",
    "pseudomonas_aeruginosa",
    "staphylococcus_aureus",
}
_CLAIMS = {
    "execution_authorized": False,
    "oracle_calls_authorized": False,
    "scientific_evidence_accepted": False,
    "production_input_eligible": False,
    "biological_superiority_claim_allowed": False,
}
_INPUT_PATHS = {
    "parser_manifest": "parser/SHA256SUMS",
    "sequences": "parser/sequences.jsonl",
    "assays": "parser/assays.jsonl",
    "endpoint_manifest": "endpoint/SHA256SUMS",
    "endpoint_ledger": "endpoint/endpoint_context_ledger.jsonl",
    "study_membership": "endpoint/study_membership.jsonl",
    "endpoint_receipt": "endpoint/receipt.json",
    "split_manifest": "split/SHA256SUMS",
    "assignments": "split/sequence_assignments.jsonl",
    "split_receipt": "split/receipt.json",
    "corpus_manifest": "corpus/SHA256SUMS",
    "corpus": "corpus/corpus.jsonl",
    "corpus_receipt": "corpus/receipt.json",
    "gate1_manifest": "gate1/SHA256SUMS",
    "gate1_examples": "gate1/examples.jsonl",
    "gate1_receipt": "gate1/receipt.json",
}
_STAGING_MANIFEST = "STAGING_MANIFEST.json"
_STAGING_SUMS = "STAGING_SHA256SUMS"
_AUTHORITY_BINDINGS = {
    "parser_manifest": (222995, "parser_manifest"),
    "sequences": (222995, "parser_manifest"),
    "assays": (222995, "parser_manifest"),
    "endpoint_manifest": (223141, "endpoint_receipt"),
    "endpoint_ledger": (223141, "endpoint_receipt"),
    "study_membership": (223141, "endpoint_receipt"),
    "endpoint_receipt": (223143, "endpoint_receipt"),
    "split_manifest": (223222, "split_receipt"),
    "assignments": (223222, "split_receipt"),
    "split_receipt": (223225, "split_receipt"),
    "corpus_manifest": (223443, "corpus_receipt"),
    "corpus": (223443, "corpus_receipt"),
    "corpus_receipt": (223445, "corpus_receipt"),
    "gate1_manifest": (223248, "gate1_receipt"),
    "gate1_examples": (223248, "gate1_receipt"),
    "gate1_receipt": (223250, "gate1_receipt"),
}
_OUTPUT_FILES = (
    "component_assignments.jsonl",
    "generator_sequence_ids.jsonl",
    "oracle_sequence_ids.jsonl",
    "generator_corpus.jsonl",
    "oracle_corpus.jsonl",
    "generator_endpoint_availability.jsonl",
    "oracle_endpoint_availability.jsonl",
    "generator_study_membership.jsonl",
    "oracle_study_membership.jsonl",
    "downstream_fold_triples.jsonl",
    "summary.json",
    "manifest.json",
)
_GENERATOR_VIEW_FILES = (
    "component_assignments.jsonl",
    "sequence_ids.jsonl",
    "corpus.jsonl",
    "endpoint_availability.jsonl",
    "study_membership.jsonl",
    "downstream_fold_triples.jsonl",
    "summary.json",
    "manifest.json",
)
_LABEL_DERIVED_FIELDS = frozenset(
    {
        "total_gate1_contexts",
        "total_gate1_sequences",
        "total_gate1_union_components",
        "activity_contexts",
        "activity_sequences",
        "activity_union_components",
        "activity_homology_components",
        "activity_positive",
        "activity_negative",
        "activity_gram_positive",
        "activity_gram_negative",
        "activity_targets",
    }
)


class NamespaceSplitError(RuntimeError):
    """Fail-closed input, policy, or publication error."""


@dataclass(frozen=True)
class Snapshot:
    payload: bytes
    sha256: str
    size: int
    fingerprint: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class PublicationBinding:
    """Identity of a marker-committed, immutable directory tree."""

    root_dev: int
    root_ino: int
    marker_sha256: str
    marker_name: str
    files: dict[str, dict[str, int | str]]


@dataclass(frozen=True)
class BuiltArtifacts:
    """Audit namespace and the narrower generator-visible bytes."""

    namespace: dict[str, bytes]
    generator_view: dict[str, bytes]


@dataclass(frozen=True)
class NamespaceExecution:
    namespace: PublicationBinding
    generator_view: PublicationBinding


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    sha256: str
    max_input_bytes: int
    max_input_records: int
    max_output_bytes: int
    max_output_records: int
    max_study_keys: int
    max_json_depth: int
    max_json_containers: int
    max_json_string_bytes: int
    folds: int
    inputs: dict[str, tuple[str, int]]
    expected: dict[str, Any]


def accepted_source_paths(twin_id: int) -> dict[str, str]:
    if isinstance(twin_id, bool) or twin_id not in {0, 1}:
        raise NamespaceSplitError("twin_id must be 0 or 1")
    task = str(twin_id)
    return {
        "parser_manifest": f"data/runs/dramp-parser-v7/222995/{task}/SHA256SUMS",
        "sequences": f"data/runs/dramp-parser-v7/222995/{task}/normalized/sequences.jsonl",
        "assays": f"data/runs/dramp-parser-v7/222995/{task}/normalized/assays.jsonl",
        "endpoint_manifest": f"data/runs/endpoint-context-v7/223141/{task}/SHA256SUMS",
        "endpoint_ledger": f"data/runs/endpoint-context-v7/223141/{task}/endpoint_context/endpoint_context_ledger.jsonl",
        "study_membership": f"data/runs/endpoint-context-v7/223141/{task}/endpoint_context/study_membership.jsonl",
        "endpoint_receipt": "data/runs/endpoint-context-v7-audits/223141/independent-verification-223143.json",
        "split_manifest": f"data/runs/homology-study-split-v1/223222/{task}/SHA256SUMS",
        "assignments": f"data/runs/homology-study-split-v1/223222/{task}/split/sequence_assignments.jsonl",
        "split_receipt": "data/runs/homology-study-split-v1-audits/223222/independent-verification-223225.json",
        "corpus_manifest": f"data/runs/categorical-diffusion-corpus-v1/223443/{task}/SHA256SUMS",
        "corpus": f"data/runs/categorical-diffusion-corpus-v1/223443/{task}/corpus/corpus.jsonl",
        "corpus_receipt": "data/runs/categorical-diffusion-corpus-v1-audits/223443/independent-verification-223445.json",
        "gate1_manifest": f"benchmarks/oracle-gate1-union-v1/223248/{task}/SHA256SUMS",
        "gate1_examples": f"benchmarks/oracle-gate1-union-v1/223248/{task}/gate1/examples.jsonl",
        "gate1_receipt": "benchmarks/oracle-gate1-union-v1-audits/223248/independent-verification-223250.json",
    }


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise NamespaceSplitError("short output write")
        view = view[written:]


def _parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise NamespaceSplitError("relative path must be text")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in {"", ".", ".."} for p in path.parts):
        raise NamespaceSplitError(f"unsafe relative path: {relative}")
    return path.parts


def _open_root(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise NamespaceSplitError("input root must be absolute")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _validate_json_shape(
    value: object,
    *,
    maximum_depth: int,
    maximum_containers: int,
    maximum_string_bytes: int,
    label: str,
) -> None:
    """Bound decoded JSON structure before any semantic traversal."""

    stack: list[tuple[object, int]] = [(value, 0)]
    containers = 0
    while stack:
        item, depth = stack.pop()
        if isinstance(item, str):
            if len(item.encode("utf-8")) > maximum_string_bytes:
                raise NamespaceSplitError(f"JSON string exceeds byte cap: {label}")
            continue
        if isinstance(item, dict):
            containers += 1
            if depth > maximum_depth or containers > maximum_containers:
                raise NamespaceSplitError(f"JSON structure exceeds cap: {label}")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise NamespaceSplitError(f"JSON object key is not text: {label}")
                if len(key.encode("utf-8")) > maximum_string_bytes:
                    raise NamespaceSplitError(f"JSON key exceeds byte cap: {label}")
                stack.append((child, depth + 1))
            continue
        if isinstance(item, list):
            containers += 1
            if depth > maximum_depth or containers > maximum_containers:
                raise NamespaceSplitError(f"JSON structure exceeds cap: {label}")
            stack.extend((child, depth + 1) for child in item)
            continue
        if item is not None and not isinstance(item, bool | int | float):
            raise NamespaceSplitError(f"unsupported decoded JSON value: {label}")


def _snapshot(root_fd: int, relative: str, limit: int, *, immutable: bool = True) -> Snapshot:
    parts = _parts(relative)
    parent = os.dup(root_fd)
    fd = -1
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = nxt
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(parts[-1], flags, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise NamespaceSplitError(f"input is not a single-link regular file: {relative}")
        if immutable and before.st_mode & 0o222:
            raise NamespaceSplitError(f"input must be immutable: {relative}")
        if before.st_size > limit:
            raise NamespaceSplitError(f"input exceeds byte cap: {relative}")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)

        def fields(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
                value.st_mode,
                value.st_nlink,
            )

        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or fields(before) != fields(after)
            or len(payload) != before.st_size
        ):
            raise NamespaceSplitError(f"input changed while read: {relative}")
        return Snapshot(payload, _sha(payload), len(payload), fields(after))
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent)


def _bounded_integer(value: object, name: str, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
        raise NamespaceSplitError(f"{name} must be an integer in [1, {ceiling}]")
    return value


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise NamespaceSplitError(f"invalid {name}")
    return value


def _study_key(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 0 < len(value) <= 2048
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise NamespaceSplitError("invalid study key")
    return value


def _endpoint(value: object) -> str:
    if not isinstance(value, str) or not _ENDPOINT.fullmatch(value):
        raise NamespaceSplitError("invalid endpoint name")
    return value


def _read_config(path: Path, expected_config_sha256: str) -> bytes:
    if not isinstance(expected_config_sha256, str) or not _SHA.fullmatch(expected_config_sha256):
        raise NamespaceSplitError(
            "expected_config_sha256 must be exactly 64 lowercase hex characters"
        )
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise NamespaceSplitError("config path must be an absolute file path")
    parent_fd = _open_root(path.parent)
    try:
        snapshot = _snapshot(parent_fd, path.name, _MAX_CONFIG_BYTES, immutable=False)
    finally:
        os.close(parent_fd)
    if snapshot.sha256 != expected_config_sha256:
        raise NamespaceSplitError("config content pin mismatch")
    return snapshot.payload


def load_config(path: Path, expected_config_sha256: str) -> Config:
    payload = _read_config(path, expected_config_sha256)
    raw = tomllib.loads(payload.decode("utf-8"))
    expected_top = {
        "schema_version",
        "artifact",
        "status",
        *tuple(_CLAIMS),
        "policy",
        "limits",
        "authorities",
        "expected",
        "inputs",
    }
    if set(raw) != expected_top:
        raise NamespaceSplitError("unexpected config fields")
    if (
        isinstance(raw.get("schema_version"), bool)
        or raw.get("schema_version") != 1
        or raw.get("artifact") != "generator_oracle_namespace_split_v1"
        or raw.get("status") != "predeclared_non_authorizing_data_preparation_only"
    ):
        raise NamespaceSplitError("unexpected config identity or status")
    if any(raw.get(k) is not False for k in _CLAIMS):
        raise NamespaceSplitError("all authorization and claim flags must be false")
    policy = raw["policy"]
    expected_policy_fields = {
        "partition",
        "partition_information",
        "safety_endpoints",
        "generator_fold_count",
        "generator_fold_order",
        "generator_fold_assignment",
        "downstream_checkpoint_training_fold_triples",
        "source_fold_roles_used",
        "source_sampling_weights_used",
        "endpoint_values_used",
    }
    if not isinstance(policy, dict) or set(policy) != expected_policy_fields:
        raise NamespaceSplitError("unexpected policy fields")
    exact_policy = {
        "partition": "whole_union_component_to_oracle_if_any_member_has_hc50_or_hemolysis_percent_observation_else_generator",
        "partition_information": "endpoint_availability_only_never_measurement_relation_bound_or_value",
        "generator_fold_order": "union_components_sorted_by_descending_sequence_count_then_union_component_id",
        "generator_fold_assignment": "assign_to_smallest_current_sequence_count_then_smallest_fold_index",
    }
    if any(policy.get(key) != value for key, value in exact_policy.items()):
        raise NamespaceSplitError("unexpected namespace or fold policy")
    if policy.get("safety_endpoints") != ["hc50", "hemolysis_percent"]:
        raise NamespaceSplitError("unexpected safety endpoint declaration")
    if (
        policy.get("endpoint_values_used") is not False
        or policy.get("source_fold_roles_used") is not False
        or policy.get("source_sampling_weights_used") is not False
    ):
        raise NamespaceSplitError("leakage-sensitive source fields must be disabled")
    if policy.get("generator_fold_count") != 5 or isinstance(
        policy.get("generator_fold_count"), bool
    ):
        raise NamespaceSplitError("generator_fold_count must be exactly 5")
    triples = policy.get("downstream_checkpoint_training_fold_triples")
    if triples != _TRIPLES:
        raise NamespaceSplitError("unexpected downstream fold-triple declaration")
    if raw.get("authorities") != _AUTHORITIES:
        raise NamespaceSplitError("authority job pins differ from the accepted set")
    limits = raw["limits"]
    if not isinstance(limits, dict) or set(limits) != set(_LIMIT_CEILINGS):
        raise NamespaceSplitError("unexpected limit fields")
    bounded_limits = {
        name: _bounded_integer(limits.get(name), name, ceiling)
        for name, ceiling in _LIMIT_CEILINGS.items()
    }
    specs: dict[str, tuple[str, int]] = {}
    if not isinstance(raw.get("inputs"), dict) or set(raw["inputs"]) != set(_INPUT_PATHS):
        raise NamespaceSplitError("unexpected input pin fields")
    for name, relative in _INPUT_PATHS.items():
        item = raw["inputs"][name]
        if not isinstance(item, dict) or set(item) != {"sha256", "bytes"}:
            raise NamespaceSplitError(f"bad input pin fields: {name}")
        digest, size = item["sha256"], item["bytes"]
        if (
            not isinstance(digest, str)
            or not _SHA.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= bounded_limits["maximum_input_file_bytes"]
        ):
            raise NamespaceSplitError(f"bad input pin: {name}")
        specs[name] = (relative, size)
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        raise NamespaceSplitError("expected census must be a table")
    expected_fields = _EXPECTED_INTEGER_FIELDS | _EXPECTED_FOLD_FIELDS | {"activity_targets"}
    if set(expected) != expected_fields:
        raise NamespaceSplitError("expected census schema is not exact")
    for name in _EXPECTED_INTEGER_FIELDS:
        value = expected[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000:
            raise NamespaceSplitError(f"invalid expected census: {name}")
    for name in _EXPECTED_FOLD_FIELDS:
        value = expected[name]
        if (
            not isinstance(value, list)
            or len(value) != 5
            or any(
                isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 10_000
                for item in value
            )
        ):
            raise NamespaceSplitError(f"invalid expected fold census: {name}")
    targets = expected["activity_targets"]
    if (
        not isinstance(targets, dict)
        or set(targets) != _ACTIVITY_TARGETS
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 10_000
            for item in targets.values()
        )
    ):
        raise NamespaceSplitError("invalid expected activity-target census")
    return Config(
        path,
        raw,
        _sha(payload),
        bounded_limits["maximum_input_file_bytes"],
        bounded_limits["maximum_input_records"],
        bounded_limits["maximum_output_file_bytes"],
        bounded_limits["maximum_output_records"],
        bounded_limits["maximum_study_keys_per_sequence"],
        bounded_limits["maximum_json_depth"],
        bounded_limits["maximum_json_containers"],
        bounded_limits["maximum_json_string_bytes"],
        5,
        specs,
        dict(expected),
    )


def _authenticate_staging_manifest(
    snapshot: Snapshot,
    *,
    expected_sha256: str,
    cfg: Config,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or not _SHA.fullmatch(expected_sha256):
        raise NamespaceSplitError("expected_staging_manifest_sha256 must be 64 lowercase hex")
    if snapshot.sha256 != expected_sha256:
        raise NamespaceSplitError("staging manifest content pin mismatch")
    try:
        document = json.loads(snapshot.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise NamespaceSplitError("staging manifest is invalid JSON") from error
    _validate_json_shape(
        document,
        maximum_depth=cfg.max_json_depth,
        maximum_containers=cfg.max_json_containers,
        maximum_string_bytes=cfg.max_json_string_bytes,
        label="staging manifest",
    )
    if not isinstance(document, dict) or _canonical(document) != snapshot.payload:
        raise NamespaceSplitError("staging manifest is not canonical JSON")
    if set(document) != {
        "schema_version",
        "artifact",
        "status",
        "claims",
        "producer_identity",
        "producer_inventory",
        "inputs",
    }:
        raise NamespaceSplitError("staging manifest schema mismatch")
    if (
        document["schema_version"] != 1
        or isinstance(document["schema_version"], bool)
        or document["artifact"] != "generator_oracle_namespace_staging_v1"
        or document["status"] != "authenticated_non_authorizing_staging_only"
        or document["claims"] != _CLAIMS
    ):
        raise NamespaceSplitError("staging manifest identity/status mismatch")
    identity = document["producer_identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "config_sha256",
        "git_commit",
        "job_id",
        "source_path",
        "source_sha256",
        "twin_id",
        "runtime_environment_sha256",
    }:
        raise NamespaceSplitError("staging producer identity schema mismatch")
    twin_id = identity.get("twin_id")
    if (
        not isinstance(identity.get("git_commit"), str)
        or not _GIT.fullmatch(identity["git_commit"])
        or not isinstance(identity.get("job_id"), str)
        or not _JOB.fullmatch(identity["job_id"])
        or identity.get("source_path")
        != "src/amp_challenge/data/generator_oracle_namespace_split.py"
        or identity.get("config_sha256") != cfg.sha256
        or not isinstance(identity.get("source_sha256"), str)
        or not _SHA.fullmatch(identity["source_sha256"])
        or not isinstance(identity.get("runtime_environment_sha256"), str)
        or not _SHA.fullmatch(identity["runtime_environment_sha256"])
        or isinstance(twin_id, bool)
        or twin_id not in {0, 1}
    ):
        raise NamespaceSplitError("invalid staging producer identity")
    inventory = document["producer_inventory"]
    expected_inventory_paths = {
        "src/amp_challenge/data/generator_oracle_namespace_split.py",
        "src/amp_challenge/data/generator_oracle_namespace_stage.py",
    }
    if not isinstance(inventory, dict) or set(inventory) != expected_inventory_paths:
        raise NamespaceSplitError("staging producer inventory schema mismatch")
    for path, digest in inventory.items():
        if not isinstance(path, str) or not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise NamespaceSplitError("invalid staging producer inventory")
    if inventory[identity["source_path"]] != identity["source_sha256"]:
        raise NamespaceSplitError("producer source inventory differs from identity")
    evidence = document["inputs"]
    source_paths = accepted_source_paths(twin_id)
    if not isinstance(evidence, dict) or set(evidence) != set(_INPUT_PATHS):
        raise NamespaceSplitError("staging input evidence schema mismatch")
    for name, item in evidence.items():
        if not isinstance(item, dict) or set(item) != {
            "bytes",
            "logical_role",
            "sha256",
            "source_dev",
            "source_ino",
            "source_mode",
            "source_nlink",
            "source_relative_path",
            "source_size",
            "staged_relative_path",
            "upstream_job_id",
            "upstream_receipt_logical",
            "upstream_receipt_sha256",
        }:
            raise NamespaceSplitError(f"staging evidence fields mismatch: {name}")
        expected_relative, expected_bytes = cfg.inputs[name]
        upstream_job, receipt_logical = _AUTHORITY_BINDINGS[name]
        if (
            item["bytes"] != expected_bytes
            or isinstance(item["bytes"], bool)
            or item["sha256"] != cfg.raw["inputs"][name]["sha256"]
            or item["source_relative_path"] != source_paths[name]
            or item["staged_relative_path"] != expected_relative
            or item["logical_role"] != name
            or item["source_size"] != expected_bytes
            or item["upstream_job_id"] != upstream_job
            or item["upstream_receipt_logical"] != receipt_logical
            or item["upstream_receipt_sha256"] != cfg.raw["inputs"][receipt_logical]["sha256"]
            or isinstance(item["source_dev"], bool)
            or not isinstance(item["source_dev"], int)
            or item["source_dev"] < 0
            or isinstance(item["source_ino"], bool)
            or not isinstance(item["source_ino"], int)
            or item["source_ino"] <= 0
            or isinstance(item["source_mode"], bool)
            or not isinstance(item["source_mode"], int)
            or not 0 <= item["source_mode"] <= 0o7777
            or isinstance(item["source_nlink"], bool)
            or not isinstance(item["source_nlink"], int)
            or item["source_nlink"] < 1
        ):
            raise NamespaceSplitError(f"staging evidence mismatch: {name}")
    return document


def _jsonl(
    snapshot: Snapshot,
    name: str,
    cap: int,
    *,
    maximum_depth: int = 16,
    maximum_containers: int = 4_096,
    maximum_string_bytes: int = 65_536,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    position = 0
    number = 0
    payload = snapshot.payload
    while position < len(payload):
        end = payload.find(b"\n", position)
        if end < 0:
            raise NamespaceSplitError(f"JSONL input lacks final newline: {name}")
        number += 1
        if number > cap:
            raise NamespaceSplitError(f"input exceeds record cap: {name}")
        line = payload[position:end]
        position = end + 1
        if not line:
            raise NamespaceSplitError(f"empty JSON line in {name}:{number}")
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise NamespaceSplitError(f"invalid JSON in {name}:{number}") from exc
        _validate_json_shape(
            row,
            maximum_depth=maximum_depth,
            maximum_containers=maximum_containers,
            maximum_string_bytes=maximum_string_bytes,
            label=f"{name}:{number}",
        )
        if not isinstance(row, dict):
            raise NamespaceSplitError(f"non-object JSON in {name}:{number}")
        result.append(row)
    return result


def _expect(actual: int | list[int], expected: dict[str, Any], key: str) -> None:
    if key in expected and actual != expected[key]:
        raise NamespaceSplitError(f"census mismatch for {key}: {actual!r} != {expected[key]!r}")


def _preflight_output_records(counts: dict[str, int], maximum: int) -> None:
    expected_names = set(_OUTPUT_FILES) | {"SHA256SUMS"}
    if set(counts) != expected_names:
        raise NamespaceSplitError("output record prediction is incomplete")
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= maximum:
            raise NamespaceSplitError(f"predicted output record cap exceeded: {name}")


def _encode_rows(
    rows: Iterable[dict[str, Any]], *, expected_records: int, maximum_bytes: int, name: str
) -> bytes:
    payload = bytearray()
    observed = 0
    for row in rows:
        observed += 1
        if observed > expected_records:
            raise NamespaceSplitError(f"output record prediction undercounted: {name}")
        encoded = _canonical(row)
        if len(payload) + len(encoded) > maximum_bytes:
            raise NamespaceSplitError(f"output byte cap exceeded: {name}")
        payload.extend(encoded)
    if observed != expected_records:
        raise NamespaceSplitError(f"output record prediction mismatch: {name}")
    return bytes(payload)


def _project_endpoint(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    projected = []
    for row in rows:
        sequence_id = _token(row.get("sequence_id"), "endpoint sequence_id")
        endpoint = _endpoint(row.get("endpoint"))
        projected.append({"sequence_id": sequence_id, "endpoint": endpoint})
    return projected


def _build(
    cfg: Config,
    snapshots: dict[str, Snapshot],
    staging_manifest: dict[str, Any],
    staging_manifest_sha256: str,
) -> BuiltArtifacts:
    def parse_rows(logical: str) -> list[dict[str, Any]]:
        return _jsonl(
            snapshots[logical],
            logical,
            cfg.max_input_records,
            maximum_depth=cfg.max_json_depth,
            maximum_containers=cfg.max_json_containers,
            maximum_string_bytes=cfg.max_json_string_bytes,
        )

    sequences_rows = parse_rows("sequences")
    assignment_rows = parse_rows("assignments")
    assay_projection = _project_endpoint(parse_rows("assays"))
    ledger_projection = _project_endpoint(parse_rows("endpoint_ledger"))

    sequence_by_id: dict[str, str] = {}
    for row in sequences_rows:
        sid = _token(row.get("sequence_id"), "sequence_id")
        sequence = row.get("sequence")
        if (
            not isinstance(sequence, str)
            or not _SEQUENCE.fullmatch(sequence)
            or sid in sequence_by_id
        ):
            raise NamespaceSplitError("invalid or duplicate sequence row")
        sequence_by_id[sid] = sequence
    assignments: dict[str, tuple[str, str]] = {}
    union_members: dict[str, set[str]] = defaultdict(set)
    for row in assignment_rows:
        sid = _token(row.get("sequence_id"), "assignment sequence_id")
        homology = _token(row.get("homology_component_id"), "homology_component_id")
        union = _token(row.get("union_component_id"), "union_component_id")
        if sid in assignments:
            raise NamespaceSplitError("invalid or duplicate assignment row")
        assignments[sid] = (homology, union)
        union_members[union].add(sid)
    if set(assignments) != set(sequence_by_id):
        raise NamespaceSplitError("sequence and assignment IDs differ")

    if Counter(
        (r["sequence_id"], r["endpoint"]) for r in assay_projection if r["endpoint"] in _SAFETY
    ) != Counter(
        (r["sequence_id"], r["endpoint"]) for r in ledger_projection if r["endpoint"] in _SAFETY
    ):
        raise NamespaceSplitError("parser and endpoint safety availability differ")
    safety_by_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assay_projection:
        if row["sequence_id"] not in assignments:
            raise NamespaceSplitError("assay references unknown sequence")
        if row["endpoint"] in _SAFETY:
            safety_by_sequence[row["sequence_id"]][row["endpoint"]] += 1
    safety_sequence_ids = frozenset(safety_by_sequence)
    oracle_unions = {assignments[sid][1] for sid in safety_sequence_ids}
    namespace = {
        sid: ("oracle" if assignments[sid][1] in oracle_unions else "generator")
        for sid in assignments
    }

    components = sorted(union_members, key=lambda u: (-len(union_members[u]), u))
    fold_loads = [0] * cfg.folds
    fold_by_union: dict[str, int] = {}
    for union in components:
        if union in oracle_unions:
            continue
        fold = min(range(cfg.folds), key=lambda i: (fold_loads[i], i))
        fold_by_union[union] = fold
        fold_loads[fold] += len(union_members[union])

    # Descriptive corpus, study, and Gate1 fields are not parsed until the
    # namespace and fold decisions are complete.
    corpus_rows = parse_rows("corpus")
    corpus_by_id: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        sid = _token(row.get("sequence_id"), "corpus sequence_id")
        _token(row.get("homology_component_id"), "corpus homology_component_id")
        _token(row.get("union_component_id"), "corpus union_component_id")
        if sid in corpus_by_id:
            raise NamespaceSplitError("invalid or duplicate corpus row")
        corpus_by_id[sid] = row
        if (
            sid not in assignments
            or row.get("sequence") != sequence_by_id[sid]
            or (row.get("homology_component_id"), row.get("union_component_id")) != assignments[sid]
        ):
            raise NamespaceSplitError("corpus conflicts with accepted sequence assignment")
    if set(corpus_by_id) != set(assignments):
        raise NamespaceSplitError("corpus ID coverage differs")

    study_rows = parse_rows("study_membership")
    gate_rows = parse_rows("gate1_examples")

    studies: dict[str, set[str]] = defaultdict(set)
    study_key_entries: Counter[str] = Counter()
    study_membership_rows = {"generator": 0, "oracle": 0}
    for row in study_rows:
        sid, keys = _token(row.get("sequence_id"), "study sequence_id"), row.get("study_keys")
        if sid not in assignments or not isinstance(keys, list) or len(keys) > cfg.max_study_keys:
            raise NamespaceSplitError("invalid study membership row")
        study_key_entries[sid] += len(keys)
        if study_key_entries[sid] > cfg.max_study_keys:
            raise NamespaceSplitError("cumulative study-key cap exceeded")
        studies[sid].update(_study_key(key) for key in keys)
        study_membership_rows[namespace[sid]] += 1

    # Assert all grouping layers are namespace-disjoint.
    for layer in (1, 0):
        owners: dict[str, str] = {}
        for sid, pair in assignments.items():
            key = pair[layer]
            if key in owners and owners[key] != namespace[sid]:
                raise NamespaceSplitError("component crosses namespaces")
            owners[key] = namespace[sid]
    study_owners: dict[str, str] = {}
    for sid, keys in studies.items():
        for key in keys:
            if key in study_owners and study_owners[key] != namespace[sid]:
                raise NamespaceSplitError("study key crosses namespaces")
            study_owners[key] = namespace[sid]

    for row in ledger_projection:
        sid = row["sequence_id"]
        if sid not in namespace:
            raise NamespaceSplitError("endpoint ledger references unknown sequence")

    ns_ids = {
        n: {s for s, value in namespace.items() if value == n} for n in ("generator", "oracle")
    }
    ns_union = {n: {assignments[s][1] for s in ns_ids[n]} for n in ns_ids}
    ns_hom = {n: {assignments[s][0] for s in ns_ids[n]} for n in ns_ids}
    ns_study = {n: {k for s in ns_ids[n] for k in studies[s]} for n in ns_ids}
    ns_sequences = {n: {sequence_by_id[s] for s in ns_ids[n]} for n in ns_ids}
    if (
        ns_sequences["generator"] & ns_sequences["oracle"]
        or ns_ids["generator"] & ns_ids["oracle"]
        or ns_union["generator"] & ns_union["oracle"]
        or ns_hom["generator"] & ns_hom["oracle"]
        or ns_study["generator"] & ns_study["oracle"]
    ):
        raise NamespaceSplitError("namespace overlap")
    metrics = {
        "total_sequences": len(assignments),
        "total_assay_observations": len(assay_projection),
        "total_endpoint_ledger_rows": len(ledger_projection),
        "total_union_components": len(union_members),
        "total_homology_components": len({v[0] for v in assignments.values()}),
        "total_study_keys": len(set().union(*studies.values()) if studies else set()),
        "total_study_membership_rows": len(study_rows),
        "total_gate1_contexts": len(gate_rows),
        "total_gate1_sequences": len({r.get("sequence_id") for r in gate_rows}),
        "total_gate1_union_components": len({r.get("union_component_id") for r in gate_rows}),
    }
    for ns in ("generator", "oracle"):
        metrics.update(
            {
                f"{ns}_sequences": len(ns_ids[ns]),
                f"{ns}_union_components": len(ns_union[ns]),
                f"{ns}_homology_components": len(ns_hom[ns]),
                f"{ns}_study_keys": len(ns_study[ns]),
                f"{ns}_study_membership_rows": study_membership_rows[ns],
            }
        )
    metrics["generator_fold_sequences"] = fold_loads
    metrics["generator_fold_union_components"] = [
        sum(f == i for f in fold_by_union.values()) for i in range(cfg.folds)
    ]
    metrics["generator_fold_homology_components"] = [
        len(
            {
                assignments[s][0]
                for s in ns_ids["generator"]
                if fold_by_union[assignments[s][1]] == i
            }
        )
        for i in range(cfg.folds)
    ]
    metrics["generator_fold_study_keys"] = [
        len(
            {
                k
                for s in ns_ids["generator"]
                if fold_by_union[assignments[s][1]] == i
                for k in studies[s]
            }
        )
        for i in range(cfg.folds)
    ]
    hc_ids = {s for s, counts in safety_by_sequence.items() if counts["hc50"]}
    hem_ids = {s for s, counts in safety_by_sequence.items() if counts["hemolysis_percent"]}
    metrics.update(
        {
            "hc50_observations": sum(safety_by_sequence[s]["hc50"] for s in safety_by_sequence),
            "hc50_sequences": len(hc_ids),
            "hc50_union_components": len({assignments[s][1] for s in hc_ids}),
            "hc50_homology_components": len({assignments[s][0] for s in hc_ids}),
            "hemolysis_percent_observations": sum(
                safety_by_sequence[s]["hemolysis_percent"] for s in safety_by_sequence
            ),
            "hemolysis_percent_sequences": len(hem_ids),
            "hemolysis_percent_union_components": len({assignments[s][1] for s in hem_ids}),
            "hemolysis_percent_homology_components": len({assignments[s][0] for s in hem_ids}),
            "any_safety_sequences": len(safety_sequence_ids),
            "any_safety_union_components": len(oracle_unions),
            "hc50_hemolysis_sequence_overlap": len(hc_ids & hem_ids),
        }
    )
    activity = []
    for row in gate_rows:
        sid = _token(row.get("sequence_id"), "Gate1 sequence_id")
        union = _token(row.get("union_component_id"), "Gate1 union_component_id")
        _token(row.get("homology_component_id"), "Gate1 homology_component_id")
        _token(row.get("canonical_target"), "Gate1 canonical_target")
        if (
            row.get("gram") not in {"positive", "negative"}
            or isinstance(row.get("label"), bool)
            or row.get("label") not in {0, 1}
        ):
            raise NamespaceSplitError("invalid Gate1 descriptive fields")
        if (
            sid not in assignments
            or union != assignments[sid][1]
            or row.get("homology_component_id") != assignments[sid][0]
        ):
            raise NamespaceSplitError("Gate1 row conflicts with accepted union assignment")
        if union in oracle_unions:
            activity.append(row)
    activity_target_counts = Counter(r.get("canonical_target") for r in activity)
    metrics.update(
        {
            "activity_contexts": len(activity),
            "activity_sequences": len({r["sequence_id"] for r in activity}),
            "activity_union_components": len({r["union_component_id"] for r in activity}),
            "activity_homology_components": len(
                {assignments[r["sequence_id"]][0] for r in activity}
            ),
            "activity_positive": sum(r.get("label") == 1 for r in activity),
            "activity_negative": sum(r.get("label") == 0 for r in activity),
            "activity_gram_positive": sum(r.get("gram") == "positive" for r in activity),
            "activity_gram_negative": sum(r.get("gram") == "negative" for r in activity),
            "activity_targets": {
                target: activity_target_counts[target] for target in sorted(_ACTIVITY_TARGETS)
            },
        }
    )
    for key, actual in metrics.items():
        _expect(actual, cfg.expected, key)

    endpoint_counts = Counter(namespace[row["sequence_id"]] for row in ledger_projection)
    record_counts = {
        "component_assignments.jsonl": len(union_members),
        "generator_sequence_ids.jsonl": len(ns_ids["generator"]),
        "oracle_sequence_ids.jsonl": len(ns_ids["oracle"]),
        "generator_corpus.jsonl": len(ns_ids["generator"]),
        "oracle_corpus.jsonl": len(ns_ids["oracle"]),
        "generator_endpoint_availability.jsonl": endpoint_counts["generator"],
        "oracle_endpoint_availability.jsonl": endpoint_counts["oracle"],
        "generator_study_membership.jsonl": len(ns_ids["generator"]),
        "oracle_study_membership.jsonl": len(ns_ids["oracle"]),
        "downstream_fold_triples.jsonl": len(_TRIPLES),
        "summary.json": 1,
        "manifest.json": 1,
        "SHA256SUMS": len(_OUTPUT_FILES),
    }
    _preflight_output_records(record_counts, cfg.max_output_records)

    def component_output_rows() -> Iterator[dict[str, Any]]:
        for union in sorted(union_members):
            members = union_members[union]
            yield {
                "schema_version": 1,
                "union_component_id": union,
                "namespace": "oracle" if union in oracle_unions else "generator",
                "generator_fold": fold_by_union.get(union),
                "sequences": len(members),
                "homology_components": len({assignments[s][0] for s in members}),
                "study_keys": len({key for sid in members for key in studies[sid]}),
                "safety_endpoints_available": sorted(
                    {endpoint for sid in members for endpoint in safety_by_sequence.get(sid, {})}
                ),
            }

    def sequence_id_rows(selected: str) -> Iterator[dict[str, Any]]:
        for sid in sorted(ns_ids[selected]):
            yield {"schema_version": 1, "sequence_id": sid}

    def corpus_output_rows(selected: str) -> Iterator[dict[str, Any]]:
        for sid in sorted(ns_ids[selected]):
            homology, union = assignments[sid]
            yield {
                "schema_version": 1,
                "namespace": selected,
                "sequence_id": sid,
                "sequence": sequence_by_id[sid],
                "homology_component_id": homology,
                "union_component_id": union,
                "generator_fold": fold_by_union.get(union),
            }

    def study_output_rows(selected: str) -> Iterator[dict[str, Any]]:
        for sid in sorted(ns_ids[selected]):
            yield {
                "schema_version": 1,
                "namespace": selected,
                "sequence_id": sid,
                "study_keys": sorted(studies[sid]),
            }

    def endpoint_output_rows(selected: str) -> Iterator[dict[str, Any]]:
        for row in ledger_projection:
            sid = row["sequence_id"]
            if namespace[sid] == selected:
                yield {
                    "schema_version": 1,
                    "namespace": selected,
                    "sequence_id": sid,
                    "endpoint": row["endpoint"],
                }

    semantic_rows: dict[str, Iterable[dict[str, Any]]] = {
        "component_assignments.jsonl": component_output_rows(),
        "generator_sequence_ids.jsonl": sequence_id_rows("generator"),
        "oracle_sequence_ids.jsonl": sequence_id_rows("oracle"),
        "generator_corpus.jsonl": corpus_output_rows("generator"),
        "oracle_corpus.jsonl": corpus_output_rows("oracle"),
        "generator_endpoint_availability.jsonl": endpoint_output_rows("generator"),
        "oracle_endpoint_availability.jsonl": endpoint_output_rows("oracle"),
        "generator_study_membership.jsonl": study_output_rows("generator"),
        "oracle_study_membership.jsonl": study_output_rows("oracle"),
        "downstream_fold_triples.jsonl": (
            {
                "schema_version": 1,
                "ordinal": ordinal,
                "triple": triple,
                "training_folds": [int(fold) for fold in triple],
                "status": "downstream_declaration_only_not_a_trained_checkpoint",
                "producer_git_commit": staging_manifest["producer_identity"]["git_commit"],
                "producer_source_sha256": staging_manifest["producer_identity"]["source_sha256"],
                "staging_manifest_sha256": staging_manifest_sha256,
            }
            for ordinal, triple in enumerate(_TRIPLES)
        ),
    }
    payloads = {
        name: _encode_rows(
            rows,
            expected_records=record_counts[name],
            maximum_bytes=cfg.max_output_bytes,
            name=name,
        )
        for name, rows in semantic_rows.items()
    }
    namespace_metrics = {
        key: value for key, value in metrics.items() if key not in _LABEL_DERIVED_FIELDS
    }
    summary = {
        "schema_version": 1,
        "artifact": "generator_oracle_namespace_split_v1",
        "status": "non_authorizing_data_preparation_only",
        "claims": _CLAIMS,
        "policy": {
            "assignment_information": "union_membership_and_endpoint_name_availability_only",
            "safety_endpoints": sorted(_SAFETY),
            "generator_folds": cfg.folds,
            "source_fold_role_weight_used": False,
            "endpoint_values_relations_bounds_labels_used": False,
        },
        "counts": namespace_metrics,
        "overlap": {
            "exact_sequences": 0,
            "union_components": 0,
            "homology_components": 0,
            "study_keys": 0,
        },
    }
    payloads["summary.json"] = _encode_rows(
        (summary,),
        expected_records=record_counts["summary.json"],
        maximum_bytes=cfg.max_output_bytes,
        name="summary.json",
    )
    manifest_artifacts = {
        name: {"sha256": _sha(data), "bytes": len(data), "records": record_counts[name]}
        for name, data in sorted(payloads.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact": "generator_oracle_namespace_split_v1",
        "status": "pending_independent_verification",
        "claims": _CLAIMS,
        "config_sha256": cfg.sha256,
        "staging": {
            "manifest_sha256": staging_manifest_sha256,
            "producer_identity": staging_manifest["producer_identity"],
            "source_inputs": staging_manifest["inputs"],
        },
        "inputs": {
            name: {"sha256": snap.sha256, "bytes": snap.size}
            for name, snap in sorted(snapshots.items())
        },
        "artifacts": manifest_artifacts,
    }
    payloads["manifest.json"] = _encode_rows(
        (manifest,),
        expected_records=record_counts["manifest.json"],
        maximum_bytes=cfg.max_output_bytes,
        name="manifest.json",
    )
    checksum = bytearray()
    for name in _OUTPUT_FILES:
        line = f"{_sha(payloads[name])}  {name}\n".encode()
        if len(checksum) + len(line) > cfg.max_output_bytes:
            raise NamespaceSplitError("output byte cap exceeded: SHA256SUMS")
        checksum.extend(line)
    payloads["SHA256SUMS"] = bytes(checksum)

    generator_component_records = len(ns_union["generator"])
    generator_record_counts = {
        "component_assignments.jsonl": generator_component_records,
        "sequence_ids.jsonl": len(ns_ids["generator"]),
        "corpus.jsonl": len(ns_ids["generator"]),
        "endpoint_availability.jsonl": endpoint_counts["generator"],
        "study_membership.jsonl": len(ns_ids["generator"]),
        "downstream_fold_triples.jsonl": len(_TRIPLES),
        "summary.json": 1,
        "manifest.json": 1,
        "SHA256SUMS": len(_GENERATOR_VIEW_FILES),
    }
    for name, count in generator_record_counts.items():
        if count > cfg.max_output_records:
            raise NamespaceSplitError(f"generator-view output record cap exceeded: {name}")
    generator_view = {
        "component_assignments.jsonl": _encode_rows(
            (row for row in component_output_rows() if row["namespace"] == "generator"),
            expected_records=generator_component_records,
            maximum_bytes=cfg.max_output_bytes,
            name="generator view component assignments",
        ),
        "sequence_ids.jsonl": payloads["generator_sequence_ids.jsonl"],
        "corpus.jsonl": payloads["generator_corpus.jsonl"],
        "endpoint_availability.jsonl": payloads["generator_endpoint_availability.jsonl"],
        "study_membership.jsonl": payloads["generator_study_membership.jsonl"],
        "downstream_fold_triples.jsonl": payloads["downstream_fold_triples.jsonl"],
    }
    generator_summary = {
        "schema_version": 1,
        "artifact": "generator_namespace_view_v1",
        "status": "non_authorizing_generator_input_only",
        "claims": _CLAIMS,
        "counts": {
            key: namespace_metrics[key]
            for key in sorted(namespace_metrics)
            if key.startswith("generator_")
        }
        | {"generator_endpoint_rows": endpoint_counts["generator"]},
        "visible_information": [
            "canonical_sequence",
            "endpoint_name_availability",
            "generator_fold",
            "homology_component_id",
            "study_key",
            "union_component_id",
        ],
    }
    generator_view["summary.json"] = _encode_rows(
        (generator_summary,),
        expected_records=1,
        maximum_bytes=cfg.max_output_bytes,
        name="generator view summary",
    )
    generator_manifest_artifacts = {
        name: {
            "bytes": len(data),
            "records": generator_record_counts[name],
            "sha256": _sha(data),
        }
        for name, data in sorted(generator_view.items())
    }
    generator_manifest = {
        "schema_version": 1,
        "artifact": "generator_namespace_view_v1",
        "status": "pending_independent_verification",
        "claims": _CLAIMS,
        "config_sha256": cfg.sha256,
        "producer_identity": staging_manifest["producer_identity"],
        "artifacts": generator_manifest_artifacts,
    }
    generator_view["manifest.json"] = _encode_rows(
        (generator_manifest,),
        expected_records=1,
        maximum_bytes=cfg.max_output_bytes,
        name="generator view manifest",
    )
    generator_checksum = bytearray()
    for name in _GENERATOR_VIEW_FILES:
        line = f"{_sha(generator_view[name])}  {name}\n".encode()
        if len(generator_checksum) + len(line) > cfg.max_output_bytes:
            raise NamespaceSplitError("generator-view checksum byte cap exceeded")
        generator_checksum.extend(line)
    generator_view["SHA256SUMS"] = bytes(generator_checksum)

    return BuiltArtifacts(payloads, generator_view)


def _snapshot_held_file(
    descriptor: int,
    maximum_bytes: int,
    label: str,
    *,
    required_nlink: int = 1,
) -> Snapshot:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != required_nlink
        or before.st_size > maximum_bytes
    ):
        raise NamespaceSplitError(f"published file identity mismatch: {label}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(1 << 20, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
        after.st_nlink,
    ) or len(payload) != before.st_size:
        raise NamespaceSplitError(f"published file changed while read: {label}")
    return Snapshot(
        payload,
        _sha(payload),
        len(payload),
        (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_nlink,
        ),
    )


def _expected_tree(payloads: dict[str, bytes]) -> tuple[set[str], dict[str, set[str]]]:
    directories: set[str] = set()
    children: dict[str, set[str]] = defaultdict(set)
    for relative in payloads:
        parts = _parts(relative)
        parent = ""
        for component in parts[:-1]:
            child = f"{parent}/{component}" if parent else component
            directories.add(child)
            children[parent].add(component)
            parent = child
        children[parent].add(parts[-1])
    for directory in directories:
        children.setdefault(directory, set())
    return directories, children


def _entry_matches(directory_fd: int, name: str, descriptor: int, label: str) -> os.stat_result:
    held = os.fstat(descriptor)
    entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (held.st_dev, held.st_ino, held.st_mode) != (entry.st_dev, entry.st_ino, entry.st_mode):
        raise NamespaceSplitError(f"filesystem entry was substituted: {label}")
    return held


def _publish_claimed_tree(
    output: Path,
    payloads: dict[str, bytes],
    *,
    marker_artifact: str,
    identity: dict[str, Any],
    maximum_file_bytes: int,
    phase_hook: Callable[[str], None] | None = None,
) -> PublicationBinding:
    """Claim a final directory in place and commit it with a sibling marker.

    A successful ``mkdirat`` owns the final pathname. Any later failure leaves an
    unmarked tombstone and never removes or replaces a pathname. The only commit
    transition is unlinking the pre-fsynced marker's temporary hard link after its
    final link and the complete claimed tree have been verified. Consumers accept
    only a single-link marker, so every pre-commit state is rejected.
    """

    if (
        not output.is_absolute()
        or any(part in {"", ".", ".."} for part in output.parts[1:])
        or output.name in {"", ".", ".."}
        or len(os.fsencode(output.name)) > 180
    ):
        raise NamespaceSplitError("output must be a safe new absolute path")
    if not payloads or len(payloads) > 128:
        raise NamespaceSplitError("publication payload inventory is empty or too large")
    if any(
        not isinstance(data, bytes) or len(data) > maximum_file_bytes for data in payloads.values()
    ):
        raise NamespaceSplitError("publication payload exceeds byte cap")
    _validate_json_shape(
        identity,
        maximum_depth=16,
        maximum_containers=4_096,
        maximum_string_bytes=65_536,
        label="publication identity",
    )
    directories, expected_children = _expected_tree(payloads)
    marker_name = f"{output.name}.complete"
    temporary_marker = f".{marker_name}.prepared-{os.getpid()}-{secrets.token_hex(12)}"
    parent_fd = _open_root(output.parent)
    directory_fds: dict[str, int] = {}
    file_fds: dict[str, int] = {}
    marker_fd = -1
    marker_link_fd = -1
    try:
        try:
            os.mkdir(output.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise NamespaceSplitError("publication claim already exists or is stale") from error
        root_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        directory_fds[""] = root_fd
        root_info = _entry_matches(parent_fd, output.name, root_fd, "claimed root")
        if stat.S_IMODE(root_info.st_mode) != 0o700:
            raise NamespaceSplitError("claimed root has unexpected mode")
        if phase_hook is not None:
            phase_hook("claimed")

        for relative in sorted(directories, key=lambda value: (value.count("/"), value)):
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            os.mkdir(parts[-1], 0o700, dir_fd=directory_fds[parent])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fds[parent],
            )
            directory_fds[relative] = descriptor
            _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
        if phase_hook is not None:
            phase_hook("directories_created")

        for relative in sorted(payloads):
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
                dir_fd=directory_fds[parent],
            )
            file_fds[relative] = descriptor
            _write_all(descriptor, payloads[relative])
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
        if phase_hook is not None:
            phase_hook("files_written")

        for relative in sorted(
            directory_fds, key=lambda value: (value.count("/"), value), reverse=True
        ):
            descriptor = directory_fds[relative]
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        os.fsync(parent_fd)
        if phase_hook is not None:
            phase_hook("tree_sealed")

        file_inventory: dict[str, dict[str, int | str]] = {}
        for relative, descriptor in sorted(file_fds.items()):
            image = _snapshot_held_file(descriptor, maximum_file_bytes, relative)
            if image.payload != payloads[relative]:
                raise NamespaceSplitError(f"publication readback mismatch: {relative}")
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
            file_inventory[relative] = {
                "bytes": image.size,
                "dev": image.fingerprint[0],
                "ino": image.fingerprint[1],
                "mode": stat.S_IMODE(image.fingerprint[5]),
                "nlink": image.fingerprint[6],
                "sha256": image.sha256,
            }
        directory_inventory: dict[str, dict[str, int]] = {}
        for relative, descriptor in sorted(directory_fds.items()):
            observed = set(os.listdir(descriptor))
            if observed != expected_children[relative]:
                raise NamespaceSplitError(f"publication tree inventory mismatch: {relative or '.'}")
            info = os.fstat(descriptor)
            if stat.S_IMODE(info.st_mode) != 0o555:
                raise NamespaceSplitError(f"publication directory is not sealed: {relative or '.'}")
            if relative:
                parts = _parts(relative)
                parent = "/".join(parts[:-1])
                _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
                directory_inventory[relative] = {
                    "dev": info.st_dev,
                    "ino": info.st_ino,
                    "mode": stat.S_IMODE(info.st_mode),
                    "nlink": info.st_nlink,
                }
        root_info = _entry_matches(parent_fd, output.name, root_fd, "sealed root")
        marker = {
            "schema_version": 2,
            "artifact": marker_artifact,
            "status": "committed_complete",
            "commit_protocol": "mkdirat_claim_populate_held_dirfd_then_single_link_marker_v2",
            "identity": identity,
            "root": {
                "dev": root_info.st_dev,
                "ino": root_info.st_ino,
                "mode": stat.S_IMODE(root_info.st_mode),
            },
            "directories": directory_inventory,
            "files": file_inventory,
        }
        marker_payload = _canonical(marker)
        if len(marker_payload) > 1_048_576:
            raise NamespaceSplitError("completion marker exceeds byte cap")
        marker_fd = os.open(
            temporary_marker,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_fd,
        )
        _write_all(marker_fd, marker_payload)
        os.fchmod(marker_fd, 0o444)
        os.fsync(marker_fd)
        prepared = _snapshot_held_file(
            marker_fd,
            len(marker_payload),
            "prepared completion marker",
        )
        if prepared.payload != marker_payload:
            raise NamespaceSplitError("prepared completion marker readback mismatch")
        if phase_hook is not None:
            phase_hook("marker_prepared")

        try:
            os.link(
                temporary_marker,
                marker_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise NamespaceSplitError("completion marker already exists") from error
        if phase_hook is not None:
            phase_hook("marker_linked")
        marker_link_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        linked = _snapshot_held_file(
            marker_link_fd,
            len(marker_payload),
            "linked completion marker",
            required_nlink=2,
        )
        prepared_linked = _snapshot_held_file(
            marker_fd,
            len(marker_payload),
            "prepared completion marker",
            required_nlink=2,
        )
        marker_entry = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            linked.payload != marker_payload
            or prepared_linked.payload != marker_payload
            or (linked.fingerprint[0], linked.fingerprint[1])
            != (prepared_linked.fingerprint[0], prepared_linked.fingerprint[1])
            or (marker_entry.st_dev, marker_entry.st_ino)
            != (linked.fingerprint[0], linked.fingerprint[1])
        ):
            raise NamespaceSplitError("completion marker link/readback mismatch")

        _entry_matches(parent_fd, output.name, root_fd, "root before commit")
        for relative, descriptor in sorted(file_fds.items()):
            rebound = _snapshot_held_file(descriptor, maximum_file_bytes, relative)
            if (
                rebound.payload != payloads[relative]
                or rebound.sha256 != file_inventory[relative]["sha256"]
            ):
                raise NamespaceSplitError(f"publication changed before commit: {relative}")
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
        for relative, descriptor in directory_fds.items():
            if set(os.listdir(descriptor)) != expected_children[relative]:
                raise NamespaceSplitError(
                    f"publication inventory changed before commit: {relative}"
                )
            if relative:
                parts = _parts(relative)
                parent = "/".join(parts[:-1])
                _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
        rebound_marker = _snapshot_held_file(
            marker_link_fd,
            len(marker_payload),
            "linked completion marker before commit",
            required_nlink=2,
        )
        rebound_prepared = _snapshot_held_file(
            marker_fd,
            len(marker_payload),
            "prepared completion marker before commit",
            required_nlink=2,
        )
        if rebound_marker.payload != marker_payload or rebound_prepared.payload != marker_payload:
            raise NamespaceSplitError("completion marker changed before commit")
        _entry_matches(parent_fd, marker_name, marker_link_fd, "completion marker before commit")
        _entry_matches(parent_fd, temporary_marker, marker_fd, "prepared marker before commit")
        _entry_matches(parent_fd, output.name, root_fd, "root immediately before commit")
        os.fsync(parent_fd)
        if phase_hook is not None:
            phase_hook("verified_before_commit")

        result = PublicationBinding(
            root_info.st_dev,
            root_info.st_ino,
            _sha(marker_payload),
            marker_name,
            file_inventory,
        )
        # Recheck after the last injectable/fallible phase so a deterministic
        # substitution cannot turn the prepared marker into an attacker-owned
        # single-link destination when the temporary link is removed.
        final_linked = _snapshot_held_file(
            marker_link_fd,
            len(marker_payload),
            "linked completion marker at commit",
            required_nlink=2,
        )
        final_prepared = _snapshot_held_file(
            marker_fd,
            len(marker_payload),
            "prepared completion marker at commit",
            required_nlink=2,
        )
        if final_linked.payload != marker_payload or final_prepared.payload != marker_payload:
            raise NamespaceSplitError("completion marker changed at commit")
        _entry_matches(parent_fd, marker_name, marker_link_fd, "completion marker at commit")
        _entry_matches(parent_fd, temporary_marker, marker_fd, "prepared marker at commit")
        final_root = _entry_matches(parent_fd, output.name, root_fd, "root at commit")
        if stat.S_IMODE(final_root.st_mode) != 0o555:
            raise NamespaceSplitError("publication root is not sealed at commit")
        for relative, descriptor in file_fds.items():
            final_image = _snapshot_held_file(descriptor, maximum_file_bytes, relative)
            if (
                final_image.payload != payloads[relative]
                or final_image.sha256 != file_inventory[relative]["sha256"]
            ):
                raise NamespaceSplitError(f"publication file changed at commit: {relative}")
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
        for relative, descriptor in directory_fds.items():
            if relative:
                parts = _parts(relative)
                parent = "/".join(parts[:-1])
                info = _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
                if stat.S_IMODE(info.st_mode) != 0o555:
                    raise NamespaceSplitError(
                        f"publication directory is not sealed at commit: {relative}"
                    )
            if set(os.listdir(descriptor)) != expected_children[relative]:
                raise NamespaceSplitError(
                    f"publication inventory changed at commit: {relative or '.'}"
                )
        # Formal commit point. There are deliberately no fallible publication
        # operations after this unlink: consumers now observe a single-link marker.
        os.unlink(temporary_marker, dir_fd=parent_fd)
        return result
    finally:
        for descriptor in file_fds.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        for descriptor in directory_fds.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if marker_link_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(marker_link_fd)
        if marker_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(marker_fd)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def _read_committed_tree(
    output: Path,
    *,
    expected_marker_artifact: str,
    expected_files: set[str],
    maximum_file_bytes: int,
    maximum_json_depth: int,
    maximum_json_containers: int,
    maximum_json_string_bytes: int,
) -> tuple[dict[str, Snapshot], dict[str, Any], PublicationBinding]:
    """Consume only a fully committed tree while holding and rechecking all fds."""

    if (
        not output.is_absolute()
        or any(part in {"", ".", ".."} for part in output.parts[1:])
        or output.name in {"", ".", ".."}
    ):
        raise NamespaceSplitError("committed input path is unsafe")
    for relative in expected_files:
        _parts(relative)
    expected_directories, expected_children = _expected_tree(
        {relative: b"" for relative in expected_files}
    )
    parent_fd = _open_root(output.parent)
    marker_fd = -1
    directory_fds: dict[str, int] = {}
    file_fds: dict[str, int] = {}
    try:
        marker_name = f"{output.name}.complete"
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        marker_image = _snapshot_held_file(marker_fd, 1_048_576, "completion marker")
        try:
            marker = json.loads(marker_image.payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise NamespaceSplitError("completion marker is invalid JSON") from error
        _validate_json_shape(
            marker,
            maximum_depth=maximum_json_depth,
            maximum_containers=maximum_json_containers,
            maximum_string_bytes=maximum_json_string_bytes,
            label="completion marker",
        )
        if (
            not isinstance(marker, dict)
            or _canonical(marker) != marker_image.payload
            or set(marker)
            != {
                "schema_version",
                "artifact",
                "status",
                "commit_protocol",
                "identity",
                "root",
                "directories",
                "files",
            }
            or marker.get("schema_version") != 2
            or isinstance(marker.get("schema_version"), bool)
            or marker.get("artifact") != expected_marker_artifact
            or marker.get("status") != "committed_complete"
            or marker.get("commit_protocol")
            != "mkdirat_claim_populate_held_dirfd_then_single_link_marker_v2"
        ):
            raise NamespaceSplitError("completion marker schema/status mismatch")
        if not isinstance(marker.get("identity"), dict):
            raise NamespaceSplitError("completion marker identity is malformed")
        if not isinstance(marker.get("root"), dict) or set(marker["root"]) != {
            "dev",
            "ino",
            "mode",
        }:
            raise NamespaceSplitError("completion marker root binding is malformed")
        if marker["root"].get("mode") != 0o555:
            raise NamespaceSplitError("completion marker does not bind a sealed root")
        if not isinstance(marker.get("files"), dict) or set(marker["files"]) != expected_files:
            raise NamespaceSplitError("completion marker file inventory mismatch")
        if (
            not isinstance(marker.get("directories"), dict)
            or set(marker["directories"]) != expected_directories
        ):
            raise NamespaceSplitError("completion marker directory inventory mismatch")

        root_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        directory_fds[""] = root_fd
        root_info = _entry_matches(parent_fd, output.name, root_fd, "committed root")
        if stat.S_IMODE(root_info.st_mode) != 0o555 or (root_info.st_dev, root_info.st_ino) != (
            marker["root"].get("dev"),
            marker["root"].get("ino"),
        ):
            raise NamespaceSplitError("committed root differs from marker")
        for relative in sorted(expected_directories, key=lambda value: (value.count("/"), value)):
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fds[parent],
            )
            directory_fds[relative] = descriptor
            info = _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
            binding = marker["directories"].get(relative)
            if (
                not isinstance(binding, dict)
                or set(binding) != {"dev", "ino", "mode", "nlink"}
                or (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_nlink)
                != (
                    binding.get("dev"),
                    binding.get("ino"),
                    binding.get("mode"),
                    binding.get("nlink"),
                )
                or stat.S_IMODE(info.st_mode) != 0o555
            ):
                raise NamespaceSplitError(f"committed directory binding mismatch: {relative}")

        snapshots: dict[str, Snapshot] = {}
        for relative in sorted(expected_files):
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fds[parent],
            )
            file_fds[relative] = descriptor
            image = _snapshot_held_file(descriptor, maximum_file_bytes, relative)
            info = _entry_matches(directory_fds[parent], parts[-1], descriptor, relative)
            binding = marker["files"].get(relative)
            if (
                not isinstance(binding, dict)
                or set(binding) != {"bytes", "dev", "ino", "mode", "nlink", "sha256"}
                or (
                    image.size,
                    info.st_dev,
                    info.st_ino,
                    stat.S_IMODE(info.st_mode),
                    info.st_nlink,
                    image.sha256,
                )
                != (
                    binding.get("bytes"),
                    binding.get("dev"),
                    binding.get("ino"),
                    binding.get("mode"),
                    binding.get("nlink"),
                    binding.get("sha256"),
                )
            ):
                raise NamespaceSplitError(f"committed file binding mismatch: {relative}")
            snapshots[relative] = image

        for relative, descriptor in directory_fds.items():
            if set(os.listdir(descriptor)) != expected_children[relative]:
                raise NamespaceSplitError(f"committed tree inventory mismatch: {relative or '.'}")
        rebound_marker = _snapshot_held_file(marker_fd, 1_048_576, "completion marker")
        if rebound_marker.payload != marker_image.payload:
            raise NamespaceSplitError("completion marker changed during consumption")
        marker_entry = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        if (marker_entry.st_dev, marker_entry.st_ino) != (
            marker_image.fingerprint[0],
            marker_image.fingerprint[1],
        ):
            raise NamespaceSplitError("completion marker was substituted")
        _entry_matches(parent_fd, output.name, root_fd, "committed root after read")
        for relative, descriptor in directory_fds.items():
            if relative:
                parts = _parts(relative)
                parent = "/".join(parts[:-1])
                _entry_matches(
                    directory_fds[parent],
                    parts[-1],
                    descriptor,
                    f"committed directory after read: {relative}",
                )
            if set(os.listdir(descriptor)) != expected_children[relative]:
                raise NamespaceSplitError(
                    f"committed tree changed during consumption: {relative or '.'}"
                )
        for relative, descriptor in file_fds.items():
            image = _snapshot_held_file(descriptor, maximum_file_bytes, relative)
            parts = _parts(relative)
            parent = "/".join(parts[:-1])
            _entry_matches(
                directory_fds[parent],
                parts[-1],
                descriptor,
                f"committed file after read: {relative}",
            )
            if (
                image.fingerprint != snapshots[relative].fingerprint
                or image.payload != snapshots[relative].payload
            ):
                raise NamespaceSplitError(f"committed file changed during consumption: {relative}")
        return (
            snapshots,
            marker,
            PublicationBinding(
                root_info.st_dev,
                root_info.st_ino,
                marker_image.sha256,
                marker_name,
                marker["files"],
            ),
        )
    finally:
        for descriptor in file_fds.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        for descriptor in directory_fds.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if marker_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(marker_fd)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def _publish(
    output: Path,
    payloads: dict[str, bytes],
    *,
    identity: dict[str, Any] | None = None,
    phase_hook: Callable[[str], None] | None = None,
) -> PublicationBinding:
    expected_payloads = set(_OUTPUT_FILES) | {"SHA256SUMS"}
    if set(payloads) != expected_payloads:
        raise NamespaceSplitError("publication payload inventory mismatch")
    return _publish_claimed_tree(
        output,
        payloads,
        marker_artifact="generator_oracle_namespace_split_complete_v2",
        identity=identity or {"test_fixture": True},
        maximum_file_bytes=max(len(value) for value in payloads.values()),
        phase_hook=phase_hook,
    )


def _execute_namespace_split(
    config_path: Path,
    input_root: Path,
    output_dir: Path,
    generator_view_dir: Path,
    *,
    expected_config_sha256: str,
    expected_staging_manifest_sha256: str,
) -> NamespaceExecution:
    cfg = load_config(config_path, expected_config_sha256)
    committed, staging_marker, staging_binding = _read_committed_tree(
        input_root,
        expected_marker_artifact="generator_oracle_namespace_staging_complete_v2",
        expected_files=set(_INPUT_PATHS.values()) | {_STAGING_MANIFEST, _STAGING_SUMS},
        maximum_file_bytes=cfg.max_input_bytes,
        maximum_json_depth=cfg.max_json_depth,
        maximum_json_containers=cfg.max_json_containers,
        maximum_json_string_bytes=cfg.max_json_string_bytes,
    )
    staging_snapshot = committed[_STAGING_MANIFEST]
    staging_sums_snapshot = committed[_STAGING_SUMS]
    snapshots = {name: committed[relative] for name, relative in _INPUT_PATHS.items()}
    staging_manifest = _authenticate_staging_manifest(
        staging_snapshot,
        expected_sha256=expected_staging_manifest_sha256,
        cfg=cfg,
    )
    marker_identity = staging_marker.get("identity")
    producer_identity = staging_manifest["producer_identity"]
    expected_marker_identity = {
        "config_sha256": cfg.sha256,
        "producer_git_commit": producer_identity["git_commit"],
        "producer_job_id": producer_identity["job_id"],
        "producer_source_sha256": producer_identity["source_sha256"],
        "runtime_environment_sha256": producer_identity["runtime_environment_sha256"],
        "stager_source_sha256": staging_manifest["producer_inventory"][
            "src/amp_challenge/data/generator_oracle_namespace_stage.py"
        ],
        "twin_id": producer_identity["twin_id"],
    }
    if marker_identity != expected_marker_identity:
        raise NamespaceSplitError("staging completion marker identity mismatch")
    expected_staging_sums = (
        b"".join(
            f"{snapshots[name].sha256}  {_INPUT_PATHS[name]}\n".encode()
            for name in sorted(_INPUT_PATHS, key=_INPUT_PATHS.get)
        )
        + f"{staging_snapshot.sha256}  {_STAGING_MANIFEST}\n".encode()
    )
    if staging_sums_snapshot.payload != expected_staging_sums:
        raise NamespaceSplitError("staging checksum marker mismatch")
    for name, snapshot in snapshots.items():
        _, expected_size = cfg.inputs[name]
        expected_sha = cfg.raw["inputs"][name]["sha256"]
        if snapshot.size != expected_size or snapshot.sha256 != expected_sha:
            raise NamespaceSplitError(f"content pin mismatch: {name}")
    for receipt in ("endpoint_receipt", "split_receipt", "corpus_receipt", "gate1_receipt"):
        try:
            receipt_document = json.loads(snapshots[receipt].payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise NamespaceSplitError(f"authority receipt is invalid JSON: {receipt}") from error
        _validate_json_shape(
            receipt_document,
            maximum_depth=cfg.max_json_depth,
            maximum_containers=cfg.max_json_containers,
            maximum_string_bytes=cfg.max_json_string_bytes,
            label=receipt,
        )
        if not isinstance(receipt_document, dict) or receipt_document.get("status") != "passed":
            raise NamespaceSplitError(f"authority receipt not passed: {receipt}")
    artifacts = _build(
        cfg,
        snapshots,
        staging_manifest,
        expected_staging_manifest_sha256,
    )
    common_identity = {
        **expected_marker_identity,
        "staging_completion_marker_sha256": staging_binding.marker_sha256,
        "staging_manifest_sha256": expected_staging_manifest_sha256,
    }
    namespace_binding = _publish(
        output_dir,
        artifacts.namespace,
        identity={
            **common_identity,
            "surface": "audit_namespace",
        },
    )
    generator_binding = _publish_claimed_tree(
        generator_view_dir,
        artifacts.generator_view,
        marker_artifact="generator_namespace_view_complete_v2",
        identity={**common_identity, "surface": "generator_view"},
        maximum_file_bytes=cfg.max_output_bytes,
    )
    return NamespaceExecution(namespace_binding, generator_binding)


def build_namespace_split(
    config_path: Path,
    input_root: Path,
    output_dir: Path,
    *,
    generator_view_dir: Path | None = None,
    expected_config_sha256: str,
    expected_staging_manifest_sha256: str,
) -> Path:
    if generator_view_dir is None:
        generator_view_dir = output_dir.with_name(f"{output_dir.name}-generator-view")
    _execute_namespace_split(
        config_path,
        input_root,
        output_dir,
        generator_view_dir,
        expected_config_sha256=expected_config_sha256,
        expected_staging_manifest_sha256=expected_staging_manifest_sha256,
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-view-dir", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-staging-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    execution = _execute_namespace_split(
        args.config,
        args.input_root,
        args.output_dir,
        args.generator_view_dir,
        expected_config_sha256=args.expected_config_sha256,
        expected_staging_manifest_sha256=args.expected_staging_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "generator_view": {
                    "completion_marker_sha256": execution.generator_view.marker_sha256,
                    "root_dev": execution.generator_view.root_dev,
                    "root_ino": execution.generator_view.root_ino,
                },
                "namespace": {
                    "completion_marker_sha256": execution.namespace.marker_sha256,
                    "root_dev": execution.namespace.root_dev,
                    "root_ino": execution.namespace.root_ino,
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

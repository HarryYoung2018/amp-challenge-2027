"""Independently verify generator/oracle namespace-split producer twins.

This verifier intentionally does not import either namespace producer module.  It
re-parses the frozen inputs, derives the partition and folds with a separate heap
implementation, reconstructs every semantic output, authenticates both staging
chains, and emits a non-authorizing receipt through an atomic no-replace link.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import os
import re
import secrets
import stat
import subprocess
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_PATTERN = re.compile(r"[0-9a-f]{40}")
JOB_PATTERN = re.compile(r"[1-9][0-9]{0,19}")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
ENDPOINT_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
SEQUENCE_PATTERN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{8,50}")
NODE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

CLAIMS = {
    "execution_authorized": False,
    "oracle_calls_authorized": False,
    "scientific_evidence_accepted": False,
    "production_input_eligible": False,
    "biological_superiority_claim_allowed": False,
}
SAFETY_ENDPOINTS = frozenset({"hc50", "hemolysis_percent"})
FOLD_TRIPLES = ("012", "013", "014", "023", "024", "034", "123", "124", "134", "234")
INPUT_PATHS = {
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
AUTHORITY_BINDINGS = {
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
AUTHORITIES = {
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
OUTPUT_FILES = (
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
SEMANTIC_TWIN_FILES = (*OUTPUT_FILES[:9], "summary.json")
STAGING_FILES = frozenset(INPUT_PATHS.values()) | {
    "STAGING_MANIFEST.json",
    "STAGING_SHA256SUMS",
}
OUTPUT_INVENTORY = frozenset(OUTPUT_FILES) | {"SHA256SUMS"}
GENERATOR_VIEW_FILES = (
    "component_assignments.jsonl",
    "sequence_ids.jsonl",
    "corpus.jsonl",
    "endpoint_availability.jsonl",
    "study_membership.jsonl",
    "downstream_fold_triples.jsonl",
    "summary.json",
    "manifest.json",
)
GENERATOR_VIEW_INVENTORY = frozenset(GENERATOR_VIEW_FILES) | {"SHA256SUMS"}
AUDIT_CENSUS_FILES = ("label_census.json", "manifest.json")
AUDIT_CENSUS_INVENTORY = frozenset(AUDIT_CENSUS_FILES) | {"SHA256SUMS"}
LABEL_DERIVED_FIELDS = frozenset(
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
SOURCE_PATHS = {
    "producer": "src/amp_challenge/data/generator_oracle_namespace_split.py",
    "stager": "src/amp_challenge/data/generator_oracle_namespace_stage.py",
    "verifier": "src/amp_challenge/data/generator_oracle_namespace_verify.py",
    "record": "src/amp_challenge/data/generator_oracle_namespace_record.py",
    "runtime": "src/amp_challenge/data/generator_oracle_namespace_runtime.py",
    "finalizer": "src/amp_challenge/data/generator_oracle_namespace_finalize.py",
    "producer_launcher": "cluster/slurm/build_generator_oracle_namespace_split_v1_twins.sbatch",
    "audit_launcher": "cluster/slurm/audit_generator_oracle_namespace_split_v1_twins.sbatch",
    "finalizer_launcher": "cluster/slurm/finalize_generator_oracle_namespace_split_v1.sbatch",
    "audit_submitter": "cluster/slurm/submit_generator_oracle_namespace_split_v1_audit.sh",
    "config": "configs/data/generator_oracle_namespace_split_v1.toml",
    "pyproject": "pyproject.toml",
    "uv_lock": "uv.lock",
}
EXPECTED_LIMITS = {
    "maximum_input_file_bytes": 33_554_432,
    "maximum_input_records": 10_000,
    "maximum_output_file_bytes": 16_777_216,
    "maximum_output_records": 4_096,
    "maximum_study_keys_per_sequence": 64,
    "maximum_json_depth": 16,
    "maximum_json_containers": 4_096,
    "maximum_json_string_bytes": 65_536,
}
MAX_JSON_DEPTH = 16
MAX_JSON_CONTAINERS = 4_096
MAX_JSON_STRING_BYTES = 65_536
EXPECTED_POLICY = {
    "partition": "whole_union_component_to_oracle_if_any_member_has_hc50_or_hemolysis_percent_observation_else_generator",
    "partition_information": "endpoint_availability_only_never_measurement_relation_bound_or_value",
    "safety_endpoints": ["hc50", "hemolysis_percent"],
    "generator_fold_count": 5,
    "generator_fold_order": "union_components_sorted_by_descending_sequence_count_then_union_component_id",
    "generator_fold_assignment": "assign_to_smallest_current_sequence_count_then_smallest_fold_index",
    "downstream_checkpoint_training_fold_triples": list(FOLD_TRIPLES),
    "source_fold_roles_used": False,
    "source_sampling_weights_used": False,
    "endpoint_values_used": False,
}
ACTIVITY_TARGETS = frozenset(
    {
        "enterococcus_faecalis",
        "enterococcus_faecium",
        "escherichia_coli",
        "klebsiella_pneumoniae",
        "pseudomonas_aeruginosa",
        "staphylococcus_aureus",
    }
)


class NamespaceVerificationError(RuntimeError):
    """A fail-closed independent-verification error."""


@dataclass(frozen=True)
class FileImage:
    payload: bytes
    sha256: str
    size: int
    fingerprint: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class VerifiedConfig:
    payload: bytes
    sha256: str
    document: dict[str, Any]
    expected: dict[str, Any]
    input_pins: dict[str, tuple[str, int]]


@dataclass(frozen=True)
class DerivedNamespace:
    payloads: dict[str, bytes]
    generator_view_payloads: dict[str, bytes]
    audit_census_payloads: dict[str, bytes]
    metrics: dict[str, Any]
    overlaps: dict[str, int]
    namespace_ids: dict[str, frozenset[str]]
    artifact_records: dict[str, int]


@dataclass(frozen=True)
class TwinEvidence:
    twin_id: int
    node_name: str
    runtime_environment_sha256: str
    staging_manifest_sha256: str
    output_manifest_sha256: str
    output_sums_sha256: str
    semantic_sha256: dict[str, str]
    source_bindings: dict[str, dict[str, int | str]]
    metrics: dict[str, Any]
    artifact_records: dict[str, int]
    surfaces: dict[str, Any]


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise NamespaceVerificationError(message)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bound_decoded_json(
    value: object,
    label: str,
    *,
    maximum_depth: int = MAX_JSON_DEPTH,
    maximum_containers: int = MAX_JSON_CONTAINERS,
    maximum_string_bytes: int = MAX_JSON_STRING_BYTES,
) -> None:
    queue: list[tuple[object, int]] = [(value, 0)]
    containers = 0
    while queue:
        item, depth = queue.pop()
        if isinstance(item, str):
            _fail(
                len(item.encode("utf-8")) <= maximum_string_bytes,
                f"JSON string exceeds cap: {label}",
            )
        elif isinstance(item, dict):
            containers += 1
            _fail(
                depth <= maximum_depth and containers <= maximum_containers,
                f"JSON structure exceeds cap: {label}",
            )
            for key, child in item.items():
                _fail(isinstance(key, str), f"JSON object key is not text: {label}")
                _fail(
                    len(key.encode("utf-8")) <= maximum_string_bytes,
                    f"JSON key exceeds cap: {label}",
                )
                queue.append((child, depth + 1))
        elif isinstance(item, list):
            containers += 1
            _fail(
                depth <= maximum_depth and containers <= maximum_containers,
                f"JSON structure exceeds cap: {label}",
            )
            queue.extend((child, depth + 1) for child in item)
        else:
            _fail(
                item is None or isinstance(item, bool | int | float),
                f"unsupported decoded JSON value: {label}",
            )


def _safe_parts(relative: str) -> tuple[str, ...]:
    _fail(isinstance(relative, str), "relative path must be text")
    value = PurePosixPath(relative)
    _fail(
        not value.is_absolute()
        and bool(value.parts)
        and all(part not in {"", ".", ".."} for part in value.parts),
        f"unsafe relative path: {relative}",
    )
    return value.parts


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _open_absolute_directory(path: Path) -> int:
    _fail(
        path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts[1:]),
        "verified directory must be a traversal-free absolute path",
    )
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(root_fd: int, relative: str) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in _safe_parts(relative):
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _read_file(
    root_fd: int,
    relative: str,
    maximum_bytes: int,
    *,
    required_mode: int | None = None,
) -> FileImage:
    parts = _safe_parts(relative)
    parent = os.dup(root_fd)
    descriptor = -1
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            os.close(parent)
            parent = child
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        _fail(stat.S_ISREG(before.st_mode), f"verified input is not regular: {relative}")
        _fail(before.st_nlink == 1, f"verified input is not single-link: {relative}")
        if required_mode is not None:
            _fail(
                stat.S_IMODE(before.st_mode) == required_mode,
                f"verified input has unexpected mode: {relative}",
            )
        _fail(before.st_size <= maximum_bytes, f"verified input exceeds byte cap: {relative}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            observed += len(chunk)
            _fail(observed <= maximum_bytes, f"verified input exceeds byte cap: {relative}")
        after = os.fstat(descriptor)
        _fail(
            _stat_fingerprint(before) == _stat_fingerprint(after) and observed == before.st_size,
            f"verified input changed while read: {relative}",
        )
        return FileImage(b"".join(chunks), digest.hexdigest(), observed, _stat_fingerprint(after))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _read_descriptor(
    descriptor: int,
    maximum_bytes: int,
    label: str,
    *,
    required_nlink: int = 1,
) -> FileImage:
    before = os.fstat(descriptor)
    _fail(
        stat.S_ISREG(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o444
        and before.st_nlink == required_nlink
        and before.st_size <= maximum_bytes,
        f"held file identity mismatch: {label}",
    )
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed = 0
    while True:
        chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - observed))
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        observed += len(chunk)
        _fail(observed <= maximum_bytes, f"held file exceeds byte cap: {label}")
    after = os.fstat(descriptor)
    _fail(
        _stat_fingerprint(before) == _stat_fingerprint(after) and observed == before.st_size,
        f"held file changed while read: {label}",
    )
    return FileImage(b"".join(chunks), digest.hexdigest(), observed, _stat_fingerprint(after))


def _layout(paths: set[str]) -> tuple[set[str], dict[str, set[str]]]:
    directories: set[str] = set()
    children: dict[str, set[str]] = defaultdict(set)
    for relative in paths:
        parts = _safe_parts(relative)
        parent = ""
        for part in parts[:-1]:
            child = f"{parent}/{part}" if parent else part
            directories.add(child)
            children[parent].add(part)
            parent = child
        children[parent].add(parts[-1])
    for directory in directories:
        children.setdefault(directory, set())
    return directories, children


def _consume_marker_committed_tree(
    path: Path,
    *,
    marker_artifact: str,
    expected_files: set[str],
    maximum_file_bytes: int,
) -> tuple[dict[str, FileImage], dict[str, Any], tuple[int, int], dict[str, Any]]:
    """Independently consume a marker-committed tree with all entries held."""

    _fail(
        path.is_absolute()
        and path.name not in {"", ".", ".."}
        and all(part not in {"", ".", ".."} for part in path.parts[1:]),
        "committed tree path is unsafe",
    )
    expected_directories, expected_children = _layout(expected_files)
    parent_fd = _open_absolute_directory(path.parent)
    marker_fd = -1
    directories: dict[str, int] = {}
    files: dict[str, int] = {}
    try:
        marker_name = f"{path.name}.complete"
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        marker_image = _read_descriptor(marker_fd, 1_048_576, "completion marker")
        marker = _json_document(marker_image, "completion marker")
        _fail(
            set(marker)
            == {
                "schema_version",
                "artifact",
                "status",
                "commit_protocol",
                "identity",
                "root",
                "directories",
                "files",
            }
            and marker.get("schema_version") == 2
            and not isinstance(marker.get("schema_version"), bool)
            and marker.get("artifact") == marker_artifact
            and marker.get("status") == "committed_complete"
            and marker.get("commit_protocol")
            == "mkdirat_claim_populate_held_dirfd_then_single_link_marker_v2"
            and isinstance(marker.get("identity"), dict),
            "completion marker schema/status mismatch",
        )
        root_binding = marker.get("root")
        directory_bindings = marker.get("directories")
        file_bindings = marker.get("files")
        _fail(
            isinstance(root_binding, dict)
            and set(root_binding) == {"dev", "ino", "mode"}
            and root_binding.get("mode") == 0o555,
            "completion marker root binding mismatch",
        )
        _fail(
            isinstance(directory_bindings, dict)
            and set(directory_bindings) == expected_directories,
            "completion marker directory inventory mismatch",
        )
        _fail(
            isinstance(file_bindings, dict) and set(file_bindings) == expected_files,
            "completion marker file inventory mismatch",
        )

        root_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        directories[""] = root_fd
        root = os.fstat(root_fd)
        root_entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            (root.st_dev, root.st_ino, root.st_mode)
            == (root_entry.st_dev, root_entry.st_ino, root_entry.st_mode)
            and (root.st_dev, root.st_ino, stat.S_IMODE(root.st_mode))
            == (root_binding.get("dev"), root_binding.get("ino"), root_binding.get("mode")),
            "committed root entry/descriptor/marker mismatch",
        )
        for relative in sorted(expected_directories, key=lambda value: (value.count("/"), value)):
            parts = _safe_parts(relative)
            parent = "/".join(parts[:-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directories[parent],
            )
            directories[relative] = descriptor
            held = os.fstat(descriptor)
            entry = os.stat(parts[-1], dir_fd=directories[parent], follow_symlinks=False)
            binding = directory_bindings.get(relative)
            _fail(
                isinstance(binding, dict)
                and set(binding) == {"dev", "ino", "mode", "nlink"}
                and (held.st_dev, held.st_ino, held.st_mode)
                == (entry.st_dev, entry.st_ino, entry.st_mode)
                and (held.st_dev, held.st_ino, stat.S_IMODE(held.st_mode), held.st_nlink)
                == (
                    binding.get("dev"),
                    binding.get("ino"),
                    binding.get("mode"),
                    binding.get("nlink"),
                )
                and stat.S_IMODE(held.st_mode) == 0o555,
                f"committed directory binding mismatch: {relative}",
            )

        images: dict[str, FileImage] = {}
        for relative in sorted(expected_files):
            parts = _safe_parts(relative)
            parent = "/".join(parts[:-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directories[parent],
            )
            files[relative] = descriptor
            image = _read_descriptor(descriptor, maximum_file_bytes, relative)
            entry = os.stat(parts[-1], dir_fd=directories[parent], follow_symlinks=False)
            binding = file_bindings.get(relative)
            _fail(
                isinstance(binding, dict)
                and set(binding) == {"bytes", "dev", "ino", "mode", "nlink", "sha256"}
                and (
                    image.size,
                    image.fingerprint[0],
                    image.fingerprint[1],
                    stat.S_IMODE(image.fingerprint[5]),
                    image.fingerprint[6],
                    image.sha256,
                )
                == (
                    binding.get("bytes"),
                    binding.get("dev"),
                    binding.get("ino"),
                    binding.get("mode"),
                    binding.get("nlink"),
                    binding.get("sha256"),
                )
                and (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink)
                == (
                    image.fingerprint[0],
                    image.fingerprint[1],
                    image.fingerprint[5],
                    image.fingerprint[6],
                ),
                f"committed file binding mismatch: {relative}",
            )
            images[relative] = image

        for relative, descriptor in directories.items():
            _fail(
                set(os.listdir(descriptor)) == expected_children[relative],
                f"committed directory inventory mismatch: {relative or '.'}",
            )
        _fail(
            _read_descriptor(marker_fd, 1_048_576, "completion marker").fingerprint
            == marker_image.fingerprint,
            "completion marker changed during consumption",
        )
        marker_entry = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            (marker_entry.st_dev, marker_entry.st_ino, marker_entry.st_nlink)
            == (marker_image.fingerprint[0], marker_image.fingerprint[1], 1),
            "completion marker was substituted or is not committed",
        )
        root_rebound = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            (root_rebound.st_dev, root_rebound.st_ino, root_rebound.st_mode)
            == (root.st_dev, root.st_ino, root.st_mode),
            "committed root was substituted during consumption",
        )
        for relative, descriptor in directories.items():
            if relative:
                parts = _safe_parts(relative)
                parent = "/".join(parts[:-1])
                entry = os.stat(parts[-1], dir_fd=directories[parent], follow_symlinks=False)
                held = os.fstat(descriptor)
                _fail(
                    (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink)
                    == (held.st_dev, held.st_ino, held.st_mode, held.st_nlink),
                    f"committed directory was substituted: {relative}",
                )
            _fail(
                set(os.listdir(descriptor)) == expected_children[relative],
                f"committed directory changed during consumption: {relative or '.'}",
            )
        for relative, descriptor in files.items():
            rebound = _read_descriptor(descriptor, maximum_file_bytes, relative)
            parts = _safe_parts(relative)
            parent = "/".join(parts[:-1])
            entry = os.stat(parts[-1], dir_fd=directories[parent], follow_symlinks=False)
            _fail(
                rebound.fingerprint == images[relative].fingerprint
                and rebound.payload == images[relative].payload,
                f"committed file changed during consumption: {relative}",
            )
            _fail(
                (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink)
                == (
                    rebound.fingerprint[0],
                    rebound.fingerprint[1],
                    rebound.fingerprint[5],
                    rebound.fingerprint[6],
                ),
                f"committed file was substituted: {relative}",
            )
        return images, marker, (root.st_dev, root.st_ino), file_bindings
    finally:
        for descriptor in files.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        for descriptor in directories.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if marker_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(marker_fd)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def _tree_inventory(
    root_fd: int,
    *,
    maximum_entries: int = 128,
    maximum_depth: int = 4,
) -> dict[str, tuple[str, int, int, int, int]]:
    _fail(
        isinstance(maximum_entries, int)
        and not isinstance(maximum_entries, bool)
        and 0 < maximum_entries <= 4096
        and isinstance(maximum_depth, int)
        and not isinstance(maximum_depth, bool)
        and 0 <= maximum_depth <= 16,
        "invalid tree inventory bound",
    )
    inventory: dict[str, tuple[str, int, int, int, int]] = {}

    def visit(directory_fd: int, prefix: str, depth: int) -> None:
        _fail(depth <= maximum_depth, "verified tree exceeds depth cap")
        names: list[str] = []
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                names.append(entry.name)
                _fail(
                    len(inventory) + len(names) <= maximum_entries,
                    "verified tree exceeds entry cap",
                )
        for name in sorted(names):
            _fail(name not in {"", ".", ".."} and "/" not in name, "unsafe directory entry")
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            _fail(len(inventory) < maximum_entries, "verified tree exceeds entry cap")
            if stat.S_ISDIR(info.st_mode):
                inventory[relative] = (
                    "directory",
                    stat.S_IMODE(info.st_mode),
                    info.st_nlink,
                    info.st_dev,
                    info.st_ino,
                )
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    visit(child, relative, depth + 1)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                inventory[relative] = (
                    "file",
                    stat.S_IMODE(info.st_mode),
                    info.st_nlink,
                    info.st_dev,
                    info.st_ino,
                )
            else:
                raise NamespaceVerificationError(f"unsafe filesystem entry: {relative}")

    before = os.fstat(root_fd)
    visit(root_fd, "", 0)
    after = os.fstat(root_fd)
    _fail(
        (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns),
        "verified tree changed during inventory",
    )
    return inventory


def _json_document(
    image: FileImage, label: str, *, require_canonical: bool = True
) -> dict[str, Any]:
    try:
        value = json.loads(image.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise NamespaceVerificationError(f"invalid JSON document: {label}") from error
    _bound_decoded_json(value, label)
    _fail(isinstance(value, dict), f"JSON document is not an object: {label}")
    if require_canonical:
        _fail(_canonical(value) == image.payload, f"JSON document is not canonical LF: {label}")
    return value


def _json_rows(
    image: FileImage,
    label: str,
    maximum_records: int,
    *,
    require_canonical: bool,
) -> list[dict[str, Any]]:
    _fail(image.payload.endswith(b"\n"), f"JSONL lacks final LF: {label}")
    rows: list[dict[str, Any]] = []
    start = 0
    while start < len(image.payload):
        end = image.payload.find(b"\n", start)
        _fail(end >= 0, f"JSONL lacks final LF: {label}")
        line = image.payload[start:end]
        start = end + 1
        _fail(bool(line), f"JSONL contains an empty record: {label}")
        _fail(len(rows) < maximum_records, f"JSONL exceeds record cap: {label}")
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise NamespaceVerificationError(f"invalid JSONL record: {label}") from error
        _bound_decoded_json(value, label)
        _fail(isinstance(value, dict), f"JSONL record is not an object: {label}")
        if require_canonical:
            _fail(_canonical(value) == line + b"\n", f"noncanonical JSONL record: {label}")
        rows.append(value)
    return rows


def _checksum_lines(payload: bytes, expected_paths: tuple[str, ...], label: str) -> dict[str, str]:
    _fail(payload.endswith(b"\n"), f"checksum list lacks final LF: {label}")
    parsed: dict[str, str] = {}
    observed_order: list[str] = []
    for raw in payload[:-1].split(b"\n"):
        _fail(len(raw) >= 67 and raw[64:66] == b"  ", f"malformed checksum line: {label}")
        try:
            digest = raw[:64].decode("ascii")
            relative = raw[66:].decode("ascii")
        except UnicodeDecodeError as error:
            raise NamespaceVerificationError(f"non-ASCII checksum line: {label}") from error
        _fail(bool(SHA_PATTERN.fullmatch(digest)), f"malformed checksum digest: {label}")
        _safe_parts(relative)
        _fail(relative not in parsed, f"duplicate checksum path: {label}")
        parsed[relative] = digest
        observed_order.append(relative)
    _fail(tuple(observed_order) == expected_paths, f"checksum inventory/order mismatch: {label}")
    return parsed


def _source_relative_paths(twin_id: int) -> dict[str, str]:
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


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    _fail(completed.returncode == 0, "Git authentication command failed")
    return completed.stdout


def _bounded_command(arguments: list[str], label: str, maximum_bytes: int = 65_536) -> bytes:
    completed = subprocess.run(
        arguments,
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    _fail(completed.returncode == 0, f"{label} command failed")
    _fail(
        len(completed.stdout) <= maximum_bytes and len(completed.stderr) <= maximum_bytes,
        f"{label} command output exceeds cap",
    )
    return completed.stdout


def _slurm_memory_mib(value: str, *, alloc_cpus: int, nodes: int) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT]?)([cn]?)", value)
    _fail(match is not None, "Slurm memory quantity is malformed")
    assert match is not None
    quantity = int(match.group(1))
    unit = match.group(2)
    scope = match.group(3)
    if unit == "K":
        _fail(quantity % 1024 == 0, "Slurm memory quantity is not an exact MiB value")
        quantity //= 1024
    elif unit == "G":
        quantity *= 1024
    elif unit == "T":
        quantity *= 1024 * 1024
    # Slurm reports bare and M quantities in MiB. A c/n suffix denotes a
    # per-CPU/per-node request; AllocTRES memory has no suffix and is total.
    if scope == "c":
        quantity *= alloc_cpus
    elif scope == "n":
        quantity *= nodes
    return quantity


def _sacct_completed_job(
    job_id: str,
    *,
    expected_job_name: str,
    expected_alloc_cpus: int,
    expected_nodes: int,
    expected_total_memory_mib: int,
) -> dict[str, Any]:
    """Return one exact authoritative root-job accounting row or fail closed."""

    _fail(bool(JOB_PATTERN.fullmatch(job_id)), "accounting job ID is malformed")
    fields = "JobIDRaw,JobName,State,ExitCode,NodeList,AllocCPUS,NNodes,ReqMem,AllocTRES"
    raw = _bounded_command(
        ["/usr/bin/sacct", "-X", "-j", job_id, "-n", "-P", f"--format={fields}"],
        "Slurm accounting",
    )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise NamespaceVerificationError("Slurm accounting output is not ASCII") from error
    rows = [line.split("|") for line in text.splitlines() if line]
    _fail(len(rows) == 1 and len(rows[0]) == 9, "Slurm accounting root row is not unique")
    (
        observed_id,
        job_name,
        state,
        exit_code,
        node_list,
        alloc_cpus,
        node_count,
        requested_memory,
        allocated_tres,
    ) = rows[0]
    _fail(
        observed_id == job_id
        and job_name == expected_job_name
        and state == "COMPLETED"
        and exit_code == "0:0"
        and alloc_cpus == str(expected_alloc_cpus)
        and node_count == str(expected_nodes)
        and bool(node_list)
        and len(node_list) <= 4_096
        and bool(allocated_tres)
        and len(allocated_tres) <= 4_096,
        "Slurm accounting identity/resource/completion mismatch",
    )
    _fail(
        _slurm_memory_mib(
            requested_memory,
            alloc_cpus=expected_alloc_cpus,
            nodes=expected_nodes,
        )
        == expected_total_memory_mib,
        "Slurm accounting requested-memory mismatch",
    )
    expanded = _bounded_command(
        ["/usr/bin/scontrol", "show", "hostnames", node_list],
        "Slurm node expansion",
    )
    try:
        nodes = expanded.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise NamespaceVerificationError("Slurm node expansion is not ASCII") from error
    _fail(
        len(nodes) == expected_nodes
        and len(set(nodes)) == expected_nodes
        and all(bool(NODE_PATTERN.fullmatch(node)) for node in nodes),
        "Slurm node expansion mismatch",
    )
    tres: dict[str, str] = {}
    for field in allocated_tres.split(","):
        _fail(field.count("=") == 1, "Slurm AllocTRES field is malformed")
        key, value = field.split("=", 1)
        _fail(key and value and key not in tres, "Slurm AllocTRES key is malformed")
        tres[key] = value
    _fail(
        tres.get("cpu") == str(expected_alloc_cpus)
        and tres.get("node") == str(expected_nodes)
        and "mem" in tres
        and _slurm_memory_mib(
            tres["mem"],
            alloc_cpus=expected_alloc_cpus,
            nodes=expected_nodes,
        )
        == expected_total_memory_mib,
        "Slurm AllocTRES does not bind requested CPU/node/memory resources",
    )
    return {
        "job_id": observed_id,
        "job_name": job_name,
        "state": state,
        "exit_code": exit_code,
        "node_list": node_list,
        "nodes": sorted(nodes),
        "alloc_cpus": int(alloc_cpus),
        "node_count": int(node_count),
        "requested_memory": requested_memory,
        "allocated_tres": allocated_tres,
    }


def _authenticate_repository(repository: Path, expected_commit: str) -> dict[str, str]:
    _fail(repository.is_absolute(), "repository root must be absolute")
    _fail(bool(GIT_PATTERN.fullmatch(expected_commit)), "expected Git commit is malformed")
    _fail(
        _git(repository, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        == expected_commit,
        "repository HEAD differs from expected commit",
    )
    _fail(
        not _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
        "repository is not exactly clean",
    )
    root_fd = _open_absolute_directory(repository)
    inventory: dict[str, str] = {}
    try:
        for logical, relative in SOURCE_PATHS.items():
            image = _read_file(root_fd, relative, 2 * 1024 * 1024)
            committed = _git(repository, "cat-file", "blob", f"{expected_commit}:{relative}")
            _fail(
                image.payload == committed, f"working source differs from committed blob: {logical}"
            )
            inventory[relative] = image.sha256
    finally:
        os.close(root_fd)
    return inventory


def _authenticate_producer_commit(
    repository: Path,
    producer_commit: str,
    audit_commit: str,
) -> dict[str, str]:
    _fail(bool(GIT_PATTERN.fullmatch(producer_commit)), "producer Git commit is malformed")
    completed = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repository),
            "merge-base",
            "--is-ancestor",
            producer_commit,
            audit_commit,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    _fail(completed.returncode == 0, "producer commit is not an ancestor of audit commit")
    inventory: dict[str, str] = {}
    for logical in (
        "producer",
        "stager",
        "record",
        "runtime",
        "producer_launcher",
        "config",
        "pyproject",
        "uv_lock",
    ):
        relative = SOURCE_PATHS[logical]
        inventory[relative] = _sha(
            _git(repository, "cat-file", "blob", f"{producer_commit}:{relative}")
        )
    return inventory


def _load_config(path: Path, expected_sha256: str) -> VerifiedConfig:
    _fail(path.is_absolute(), "config path must be absolute")
    _fail(bool(SHA_PATTERN.fullmatch(expected_sha256)), "config SHA-256 is malformed")
    parent = _open_absolute_directory(path.parent)
    try:
        image = _read_file(parent, path.name, 64 * 1024)
    finally:
        os.close(parent)
    _fail(image.sha256 == expected_sha256, "config content pin mismatch")
    try:
        document = tomllib.loads(image.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise NamespaceVerificationError("invalid namespace config") from error
    _fail(isinstance(document, dict), "namespace config is not a table")
    _fail(
        set(document)
        == {
            "schema_version",
            "artifact",
            "status",
            *CLAIMS,
            "policy",
            "limits",
            "authorities",
            "expected",
            "inputs",
        },
        "namespace config top-level schema mismatch",
    )
    _fail(
        document["schema_version"] == 1
        and not isinstance(document["schema_version"], bool)
        and document["artifact"] == "generator_oracle_namespace_split_v1"
        and document["status"] == "predeclared_non_authorizing_data_preparation_only",
        "namespace config identity mismatch",
    )
    _fail(all(document[key] is False for key in CLAIMS), "namespace config authorizes a claim")
    _fail(document.get("policy") == EXPECTED_POLICY, "namespace config policy mismatch")
    _fail(document.get("limits") == EXPECTED_LIMITS, "namespace config limits mismatch")
    _fail(document.get("authorities") == AUTHORITIES, "namespace config authority mismatch")
    expected = document.get("expected")
    _fail(isinstance(expected, dict), "namespace expected census is not a table")
    _fail(
        isinstance(expected.get("activity_targets"), dict)
        and set(expected["activity_targets"]) == ACTIVITY_TARGETS,
        "namespace target census schema mismatch",
    )
    for key, value in expected.items():
        if key == "activity_targets":
            _fail(
                all(
                    isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 10_000
                    for item in value.values()
                ),
                "namespace target census values are invalid",
            )
        elif key.startswith("generator_fold_"):
            _fail(
                isinstance(value, list)
                and len(value) == 5
                and all(
                    isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 10_000
                    for item in value
                ),
                f"invalid fold census: {key}",
            )
        else:
            _fail(
                isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10_000,
                f"invalid census: {key}",
            )
    raw_pins = document.get("inputs")
    _fail(
        isinstance(raw_pins, dict) and set(raw_pins) == set(INPUT_PATHS),
        "input pin schema mismatch",
    )
    pins: dict[str, tuple[str, int]] = {}
    for logical in INPUT_PATHS:
        item = raw_pins[logical]
        _fail(
            isinstance(item, dict) and set(item) == {"sha256", "bytes"}, "input pin fields mismatch"
        )
        digest, size = item["sha256"], item["bytes"]
        _fail(
            isinstance(digest, str) and bool(SHA_PATTERN.fullmatch(digest)),
            "input digest malformed",
        )
        _fail(
            isinstance(size, int)
            and not isinstance(size, bool)
            and 0 < size <= EXPECTED_LIMITS["maximum_input_file_bytes"],
            "input size pin malformed",
        )
        pins[logical] = (digest, size)
    return VerifiedConfig(image.payload, image.sha256, document, dict(expected), pins)


def _token(value: object, label: str) -> str:
    _fail(
        isinstance(value, str) and bool(TOKEN_PATTERN.fullmatch(value)), f"invalid token: {label}"
    )
    return value


def _endpoint(value: object, label: str) -> str:
    _fail(
        isinstance(value, str) and bool(ENDPOINT_PATTERN.fullmatch(value)),
        f"invalid endpoint: {label}",
    )
    return value


def _study_key(value: object) -> str:
    _fail(
        isinstance(value, str)
        and 0 < len(value) <= 2048
        and all(ord(character) >= 32 and ord(character) != 127 for character in value),
        "invalid study key",
    )
    return value


def _endpoint_projection(rows: list[dict[str, Any]], label: str) -> list[tuple[str, str]]:
    return [
        (
            _token(row.get("sequence_id"), f"{label} sequence_id"),
            _endpoint(row.get("endpoint"), label),
        )
        for row in rows
    ]


def _expect_metrics(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    _fail(set(observed) == set(expected), "reconstructed census schema differs from config")
    for key in sorted(expected):
        _fail(observed[key] == expected[key], f"reconstructed census mismatch: {key}")


def _derive_namespace(
    config: VerifiedConfig,
    inputs: dict[str, FileImage],
    *,
    producer_identity: dict[str, Any],
    staging_manifest_sha256: str,
) -> DerivedNamespace:
    maximum_records = EXPECTED_LIMITS["maximum_input_records"]
    sequences_rows = _json_rows(
        inputs["sequences"], "sequences", maximum_records, require_canonical=False
    )
    assignment_rows = _json_rows(
        inputs["assignments"], "assignments", maximum_records, require_canonical=False
    )
    assay_pairs = _endpoint_projection(
        _json_rows(inputs["assays"], "assays", maximum_records, require_canonical=False),
        "assays",
    )
    ledger_pairs = _endpoint_projection(
        _json_rows(
            inputs["endpoint_ledger"],
            "endpoint ledger",
            maximum_records,
            require_canonical=False,
        ),
        "endpoint ledger",
    )

    sequence_by_id: dict[str, str] = {}
    for row in sequences_rows:
        sequence_id = _token(row.get("sequence_id"), "sequence_id")
        sequence = row.get("sequence")
        _fail(
            isinstance(sequence, str)
            and bool(SEQUENCE_PATTERN.fullmatch(sequence))
            and sequence_id not in sequence_by_id,
            "invalid or duplicate sequence row",
        )
        sequence_by_id[sequence_id] = sequence

    assignment_by_id: dict[str, tuple[str, str]] = {}
    union_members: dict[str, set[str]] = defaultdict(set)
    for row in assignment_rows:
        sequence_id = _token(row.get("sequence_id"), "assignment sequence_id")
        homology = _token(row.get("homology_component_id"), "homology component")
        union = _token(row.get("union_component_id"), "union component")
        _fail(sequence_id not in assignment_by_id, "duplicate assignment sequence_id")
        assignment_by_id[sequence_id] = (homology, union)
        union_members[union].add(sequence_id)
    _fail(set(sequence_by_id) == set(assignment_by_id), "sequence/assignment coverage mismatch")

    for sequence_id, _ in (*assay_pairs, *ledger_pairs):
        _fail(sequence_id in assignment_by_id, "endpoint row references unknown sequence")
    assay_safety = Counter(pair for pair in assay_pairs if pair[1] in SAFETY_ENDPOINTS)
    ledger_safety = Counter(pair for pair in ledger_pairs if pair[1] in SAFETY_ENDPOINTS)
    _fail(assay_safety == ledger_safety, "assay/ledger safety availability mismatch")
    safety_by_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence_id, endpoint in assay_pairs:
        if endpoint in SAFETY_ENDPOINTS:
            safety_by_sequence[sequence_id][endpoint] += 1
    safety_ids = frozenset(safety_by_sequence)
    oracle_unions = {assignment_by_id[sequence_id][1] for sequence_id in safety_ids}
    namespace = {
        sequence_id: (
            "oracle" if assignment_by_id[sequence_id][1] in oracle_unions else "generator"
        )
        for sequence_id in assignment_by_id
    }

    # Independent implementation: a heap carries the current fold loads instead
    # of repeatedly scanning all folds as the producer does.
    fold_heap = [(0, fold) for fold in range(5)]
    heapq.heapify(fold_heap)
    fold_by_union: dict[str, int] = {}
    for union, members in sorted(union_members.items(), key=lambda item: (-len(item[1]), item[0])):
        if union in oracle_unions:
            continue
        load, fold = heapq.heappop(fold_heap)
        fold_by_union[union] = fold
        heapq.heappush(fold_heap, (load + len(members), fold))
    fold_loads = [0] * 5
    for union, fold in fold_by_union.items():
        fold_loads[fold] += len(union_members[union])

    corpus_rows = _json_rows(
        inputs["corpus"], "categorical corpus", maximum_records, require_canonical=False
    )
    corpus_by_id: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        sequence_id = _token(row.get("sequence_id"), "corpus sequence_id")
        homology = _token(row.get("homology_component_id"), "corpus homology component")
        union = _token(row.get("union_component_id"), "corpus union component")
        _fail(sequence_id not in corpus_by_id, "duplicate corpus sequence_id")
        _fail(
            sequence_id in assignment_by_id
            and row.get("sequence") == sequence_by_id[sequence_id]
            and (homology, union) == assignment_by_id[sequence_id],
            "corpus conflicts with accepted sequence/assignment",
        )
        corpus_by_id[sequence_id] = row
    _fail(set(corpus_by_id) == set(assignment_by_id), "corpus coverage mismatch")

    study_rows = _json_rows(
        inputs["study_membership"],
        "study membership",
        maximum_records,
        require_canonical=False,
    )
    studies: dict[str, set[str]] = defaultdict(set)
    study_entries: Counter[str] = Counter()
    study_row_counts: Counter[str] = Counter()
    for row in study_rows:
        sequence_id = _token(row.get("sequence_id"), "study sequence_id")
        keys = row.get("study_keys")
        _fail(
            sequence_id in assignment_by_id
            and isinstance(keys, list)
            and len(keys) <= EXPECTED_LIMITS["maximum_study_keys_per_sequence"],
            "invalid study-membership row",
        )
        study_entries[sequence_id] += len(keys)
        _fail(
            study_entries[sequence_id] <= EXPECTED_LIMITS["maximum_study_keys_per_sequence"],
            "cumulative study-key cap exceeded",
        )
        studies[sequence_id].update(_study_key(key) for key in keys)
        study_row_counts[namespace[sequence_id]] += 1

    gate_rows = _json_rows(
        inputs["gate1_examples"], "Gate1 examples", maximum_records, require_canonical=False
    )
    validated_gate_rows: list[dict[str, Any]] = []
    for row in gate_rows:
        sequence_id = _token(row.get("sequence_id"), "Gate1 sequence_id")
        homology = _token(row.get("homology_component_id"), "Gate1 homology component")
        union = _token(row.get("union_component_id"), "Gate1 union component")
        _token(row.get("canonical_target"), "Gate1 target")
        _fail(
            row.get("gram") in {"positive", "negative"}
            and isinstance(row.get("label"), int)
            and not isinstance(row.get("label"), bool)
            and row["label"] in {0, 1},
            "invalid Gate1 label/gram",
        )
        _fail(
            sequence_id in assignment_by_id and (homology, union) == assignment_by_id[sequence_id],
            "Gate1 assignment mismatch",
        )
        validated_gate_rows.append(row)

    namespace_ids = {
        name: frozenset(sequence_id for sequence_id, value in namespace.items() if value == name)
        for name in ("generator", "oracle")
    }
    union_sets = {
        name: {assignment_by_id[sequence_id][1] for sequence_id in ids}
        for name, ids in namespace_ids.items()
    }
    homology_sets = {
        name: {assignment_by_id[sequence_id][0] for sequence_id in ids}
        for name, ids in namespace_ids.items()
    }
    study_sets = {
        name: {key for sequence_id in ids for key in studies[sequence_id]}
        for name, ids in namespace_ids.items()
    }
    sequence_sets = {
        name: {sequence_by_id[sequence_id] for sequence_id in ids}
        for name, ids in namespace_ids.items()
    }
    overlaps = {
        "exact_sequences": len(sequence_sets["generator"] & sequence_sets["oracle"]),
        "union_components": len(union_sets["generator"] & union_sets["oracle"]),
        "homology_components": len(homology_sets["generator"] & homology_sets["oracle"]),
        "study_keys": len(study_sets["generator"] & study_sets["oracle"]),
    }
    _fail(all(value == 0 for value in overlaps.values()), "generator/oracle namespace overlap")

    activity = [row for row in validated_gate_rows if row["union_component_id"] in oracle_unions]
    _fail(
        all(row["canonical_target"] in ACTIVITY_TARGETS for row in activity),
        "unexpected oracle-namespace Gate1 target",
    )
    target_counts = Counter(row["canonical_target"] for row in activity)
    hc50_ids = {sequence_id for sequence_id, counts in safety_by_sequence.items() if counts["hc50"]}
    hemolysis_ids = {
        sequence_id
        for sequence_id, counts in safety_by_sequence.items()
        if counts["hemolysis_percent"]
    }
    metrics: dict[str, Any] = {
        "total_sequences": len(assignment_by_id),
        "total_assay_observations": len(assay_pairs),
        "total_endpoint_ledger_rows": len(ledger_pairs),
        "total_union_components": len(union_members),
        "total_homology_components": len({value[0] for value in assignment_by_id.values()}),
        "total_study_keys": len({key for values in studies.values() for key in values}),
        "total_study_membership_rows": len(study_rows),
        "total_gate1_contexts": len(validated_gate_rows),
        "total_gate1_sequences": len({row["sequence_id"] for row in validated_gate_rows}),
        "total_gate1_union_components": len(
            {row["union_component_id"] for row in validated_gate_rows}
        ),
        "generator_sequences": len(namespace_ids["generator"]),
        "generator_union_components": len(union_sets["generator"]),
        "generator_homology_components": len(homology_sets["generator"]),
        "generator_study_keys": len(study_sets["generator"]),
        "generator_study_membership_rows": study_row_counts["generator"],
        "oracle_sequences": len(namespace_ids["oracle"]),
        "oracle_union_components": len(union_sets["oracle"]),
        "oracle_homology_components": len(homology_sets["oracle"]),
        "oracle_study_keys": len(study_sets["oracle"]),
        "oracle_study_membership_rows": study_row_counts["oracle"],
        "generator_fold_sequences": fold_loads,
        "generator_fold_union_components": [
            sum(fold == index for fold in fold_by_union.values()) for index in range(5)
        ],
        "generator_fold_homology_components": [
            len(
                {
                    assignment_by_id[sequence_id][0]
                    for sequence_id in namespace_ids["generator"]
                    if fold_by_union[assignment_by_id[sequence_id][1]] == index
                }
            )
            for index in range(5)
        ],
        "generator_fold_study_keys": [
            len(
                {
                    key
                    for sequence_id in namespace_ids["generator"]
                    if fold_by_union[assignment_by_id[sequence_id][1]] == index
                    for key in studies[sequence_id]
                }
            )
            for index in range(5)
        ],
        "activity_contexts": len(activity),
        "activity_sequences": len({row["sequence_id"] for row in activity}),
        "activity_union_components": len({row["union_component_id"] for row in activity}),
        "activity_homology_components": len(
            {assignment_by_id[row["sequence_id"]][0] for row in activity}
        ),
        "activity_positive": sum(row["label"] == 1 for row in activity),
        "activity_negative": sum(row["label"] == 0 for row in activity),
        "activity_gram_positive": sum(row["gram"] == "positive" for row in activity),
        "activity_gram_negative": sum(row["gram"] == "negative" for row in activity),
        "hc50_observations": sum(counts["hc50"] for counts in safety_by_sequence.values()),
        "hc50_sequences": len(hc50_ids),
        "hc50_union_components": len({assignment_by_id[item][1] for item in hc50_ids}),
        "hc50_homology_components": len({assignment_by_id[item][0] for item in hc50_ids}),
        "hemolysis_percent_observations": sum(
            counts["hemolysis_percent"] for counts in safety_by_sequence.values()
        ),
        "hemolysis_percent_sequences": len(hemolysis_ids),
        "hemolysis_percent_union_components": len(
            {assignment_by_id[item][1] for item in hemolysis_ids}
        ),
        "hemolysis_percent_homology_components": len(
            {assignment_by_id[item][0] for item in hemolysis_ids}
        ),
        "any_safety_sequences": len(safety_ids),
        "any_safety_union_components": len(oracle_unions),
        "hc50_hemolysis_sequence_overlap": len(hc50_ids & hemolysis_ids),
        "activity_targets": {target: target_counts[target] for target in sorted(ACTIVITY_TARGETS)},
    }
    _expect_metrics(metrics, config.expected)

    def encode(rows: list[dict[str, Any]], label: str) -> bytes:
        _fail(
            len(rows) <= EXPECTED_LIMITS["maximum_output_records"],
            f"reconstructed output exceeds record cap: {label}",
        )
        payload = b"".join(_canonical(row) for row in rows)
        _fail(
            len(payload) <= EXPECTED_LIMITS["maximum_output_file_bytes"],
            f"reconstructed output exceeds byte cap: {label}",
        )
        return payload

    components = [
        {
            "schema_version": 1,
            "union_component_id": union,
            "namespace": "oracle" if union in oracle_unions else "generator",
            "generator_fold": fold_by_union.get(union),
            "sequences": len(union_members[union]),
            "homology_components": len(
                {assignment_by_id[sequence_id][0] for sequence_id in union_members[union]}
            ),
            "study_keys": len(
                {key for sequence_id in union_members[union] for key in studies[sequence_id]}
            ),
            "safety_endpoints_available": sorted(
                {
                    endpoint
                    for sequence_id in union_members[union]
                    for endpoint in safety_by_sequence.get(sequence_id, {})
                }
            ),
        }
        for union in sorted(union_members)
    ]
    payloads: dict[str, bytes] = {"component_assignments.jsonl": encode(components, "components")}
    for selected in ("generator", "oracle"):
        ordered_ids = sorted(namespace_ids[selected])
        payloads[f"{selected}_sequence_ids.jsonl"] = encode(
            [{"schema_version": 1, "sequence_id": sequence_id} for sequence_id in ordered_ids],
            f"{selected} IDs",
        )
        payloads[f"{selected}_corpus.jsonl"] = encode(
            [
                {
                    "schema_version": 1,
                    "namespace": selected,
                    "sequence_id": sequence_id,
                    "sequence": sequence_by_id[sequence_id],
                    "homology_component_id": assignment_by_id[sequence_id][0],
                    "union_component_id": assignment_by_id[sequence_id][1],
                    "generator_fold": fold_by_union.get(assignment_by_id[sequence_id][1]),
                }
                for sequence_id in ordered_ids
            ],
            f"{selected} corpus",
        )
        payloads[f"{selected}_endpoint_availability.jsonl"] = encode(
            [
                {
                    "schema_version": 1,
                    "namespace": selected,
                    "sequence_id": sequence_id,
                    "endpoint": endpoint,
                }
                for sequence_id, endpoint in ledger_pairs
                if namespace[sequence_id] == selected
            ],
            f"{selected} endpoint availability",
        )
        payloads[f"{selected}_study_membership.jsonl"] = encode(
            [
                {
                    "schema_version": 1,
                    "namespace": selected,
                    "sequence_id": sequence_id,
                    "study_keys": sorted(studies[sequence_id]),
                }
                for sequence_id in ordered_ids
            ],
            f"{selected} study membership",
        )
    payloads["downstream_fold_triples.jsonl"] = encode(
        [
            {
                "schema_version": 1,
                "ordinal": ordinal,
                "triple": triple,
                "training_folds": [int(fold) for fold in triple],
                "status": "downstream_declaration_only_not_a_trained_checkpoint",
                "producer_git_commit": producer_identity["git_commit"],
                "producer_source_sha256": producer_identity["source_sha256"],
                "staging_manifest_sha256": staging_manifest_sha256,
            }
            for ordinal, triple in enumerate(FOLD_TRIPLES)
        ],
        "fold triples",
    )
    namespace_metrics = {
        key: value for key, value in metrics.items() if key not in LABEL_DERIVED_FIELDS
    }
    summary = {
        "schema_version": 1,
        "artifact": "generator_oracle_namespace_split_v1",
        "status": "non_authorizing_data_preparation_only",
        "claims": CLAIMS,
        "policy": {
            "assignment_information": "union_membership_and_endpoint_name_availability_only",
            "safety_endpoints": sorted(SAFETY_ENDPOINTS),
            "generator_folds": 5,
            "source_fold_role_weight_used": False,
            "endpoint_values_relations_bounds_labels_used": False,
        },
        "counts": namespace_metrics,
        "overlap": overlaps,
    }
    payloads["summary.json"] = encode([summary], "summary")

    generator_components = [row for row in components if row["namespace"] == "generator"]
    generator_view_payloads = {
        "component_assignments.jsonl": encode(generator_components, "generator components"),
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
        "claims": CLAIMS,
        "counts": {
            key: namespace_metrics[key]
            for key in sorted(namespace_metrics)
            if key.startswith("generator_")
        }
        | {
            "generator_endpoint_rows": sum(
                namespace[sequence_id] == "generator" for sequence_id, _ in ledger_pairs
            )
        },
        "visible_information": [
            "canonical_sequence",
            "endpoint_name_availability",
            "generator_fold",
            "homology_component_id",
            "study_key",
            "union_component_id",
        ],
    }
    generator_view_payloads["summary.json"] = encode([generator_summary], "generator summary")
    generator_artifacts = {
        name: {
            "bytes": len(value),
            "records": value.count(b"\n"),
            "sha256": _sha(value),
        }
        for name, value in sorted(generator_view_payloads.items())
    }
    generator_manifest = {
        "schema_version": 1,
        "artifact": "generator_namespace_view_v1",
        "status": "pending_independent_verification",
        "claims": CLAIMS,
        "config_sha256": config.sha256,
        "producer_identity": producer_identity,
        "artifacts": generator_artifacts,
    }
    generator_view_payloads["manifest.json"] = encode([generator_manifest], "generator manifest")
    generator_view_payloads["SHA256SUMS"] = b"".join(
        f"{_sha(generator_view_payloads[name])}  {name}\n".encode() for name in GENERATOR_VIEW_FILES
    )

    audit_counts = {key: metrics[key] for key in sorted(LABEL_DERIVED_FIELDS)}
    label_census = {
        "schema_version": 1,
        "artifact": "generator_oracle_namespace_label_census_v1",
        "status": "audit_only_not_generator_visible",
        "claims": CLAIMS,
        "counts": audit_counts,
    }
    audit_census_payloads = {"label_census.json": encode([label_census], "audit label census")}
    audit_manifest = {
        "schema_version": 1,
        "artifact": "generator_oracle_namespace_label_census_v1",
        "status": "pending_independent_verification_audit_only",
        "claims": CLAIMS,
        "config_sha256": config.sha256,
        "producer_identity": producer_identity,
        "artifacts": {
            "label_census.json": {
                "bytes": len(audit_census_payloads["label_census.json"]),
                "records": 1,
                "sha256": _sha(audit_census_payloads["label_census.json"]),
            }
        },
    }
    audit_census_payloads["manifest.json"] = encode([audit_manifest], "audit manifest")
    audit_census_payloads["SHA256SUMS"] = b"".join(
        f"{_sha(audit_census_payloads[name])}  {name}\n".encode() for name in AUDIT_CENSUS_FILES
    )
    return DerivedNamespace(
        payloads,
        generator_view_payloads,
        audit_census_payloads,
        metrics,
        overlaps,
        namespace_ids,
        {name: payload.count(b"\n") for name, payload in payloads.items()},
    )


def _assert_staging_inventory(root_fd: int) -> tuple[int, int]:
    root_info = os.fstat(root_fd)
    _fail(stat.S_IMODE(root_info.st_mode) == 0o555, "staging root mode is not 0555")
    inventory = _tree_inventory(root_fd)
    observed_files = {path for path, item in inventory.items() if item[0] == "file"}
    expected_directories = {path.split("/", 1)[0] for path in INPUT_PATHS.values()}
    observed_directories = {path for path, item in inventory.items() if item[0] == "directory"}
    _fail(observed_files == STAGING_FILES, "staging complete file inventory mismatch")
    _fail(observed_directories == expected_directories, "staging directory inventory mismatch")
    for path, item in inventory.items():
        if item[0] == "file":
            _fail(item[1] == 0o444 and item[2] == 1, f"staging file mode/link mismatch: {path}")
        else:
            _fail(item[1] == 0o555, f"staging directory mode mismatch: {path}")
    return root_info.st_dev, root_info.st_ino


def _assert_output_inventory(root_fd: int) -> tuple[int, int]:
    root_info = os.fstat(root_fd)
    _fail(stat.S_IMODE(root_info.st_mode) == 0o555, "output root mode is not 0555")
    inventory = _tree_inventory(root_fd)
    _fail(set(inventory) == OUTPUT_INVENTORY, "output complete inventory mismatch")
    for path, item in inventory.items():
        _fail(item[0] == "file", f"output contains an unexpected directory: {path}")
        _fail(item[1] == 0o444 and item[2] == 1, f"output mode/link mismatch: {path}")
    return root_info.st_dev, root_info.st_ino


def _strict_record(path_root_fd: int, relative: str, expected_lines: int) -> dict[str, str]:
    image = _read_file(path_root_fd, relative, 16 * 1024, required_mode=0o444)
    _fail(image.payload.endswith(b"\n"), f"producer record lacks final LF: {relative}")
    values: dict[str, str] = {}
    lines = image.payload[:-1].split(b"\n")
    _fail(len(lines) == expected_lines, f"producer record line-count mismatch: {relative}")
    for line in lines:
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError as error:
            raise NamespaceVerificationError(f"producer record is not ASCII: {relative}") from error
        _fail(text.count("=") == 1, f"producer record field is malformed: {relative}")
        key, value = text.split("=", 1)
        _fail(
            bool(TOKEN_PATTERN.fullmatch(key)) and key not in values, "producer record key mismatch"
        )
        _fail(
            value and all(32 <= ord(char) < 127 for char in value), "producer record value mismatch"
        )
        values[key] = value
    return values


def _verify_handshake_and_completions(
    records_root: Path,
    *,
    producer_job_id: str,
    expected_commit: str,
    config_sha256: str,
    producer_source_sha256: str,
    stager_source_sha256: str,
    record_source_sha256: str,
    runtime_source_sha256: str,
    launcher_sha256: str,
) -> tuple[dict[int, str], dict[int, dict[str, str]], str, str]:
    root_fd = _open_absolute_directory(records_root)
    try:
        root_info = os.fstat(root_fd)
        _fail(stat.S_IMODE(root_info.st_mode) == 0o555, "producer records root is not 0555")
        inventory = _tree_inventory(root_fd)
        expected_names = {
            f"{twin}.{suffix}"
            for twin in (0, 1)
            for suffix in ("receipt", "ack", "completion", "completion-ack")
        }
        _fail(set(inventory) == expected_names, "producer record inventory mismatch")
        _fail(
            all(
                item[0] == "file" and item[1] == 0o444 and item[2] == 1
                for item in inventory.values()
            ),
            "producer record mode/link mismatch",
        )
        receipts = {twin: _strict_record(root_fd, f"{twin}.receipt", 14) for twin in (0, 1)}
        receipt_images = {
            twin: _read_file(root_fd, f"{twin}.receipt", 16 * 1024, required_mode=0o444)
            for twin in (0, 1)
        }
        nodes: dict[int, str] = {}
        runtime_sha256: set[str] = set()
        runtime_marker_sha256: set[str] = set()
        for twin, record in receipts.items():
            expected = {
                "array_job_id": producer_job_id,
                "array_task_id": str(twin),
                "twin_id": str(twin),
                "job_name": "amp-ns-split-v2",
                "git_commit": expected_commit,
                "config_sha256": config_sha256,
                "producer_source_sha256": producer_source_sha256,
                "stager_source_sha256": stager_source_sha256,
                "record_source_sha256": record_source_sha256,
                "runtime_source_sha256": runtime_source_sha256,
                "launcher_sha256": launcher_sha256,
            }
            _fail(
                set(record)
                == set(expected)
                | {
                    "node_name",
                    "runtime_environment_sha256",
                    "runtime_attestation_marker_sha256",
                },
                "producer receipt schema mismatch",
            )
            _fail(
                all(record[key] == value for key, value in expected.items()),
                "producer receipt mismatch",
            )
            _fail(bool(NODE_PATTERN.fullmatch(record["node_name"])), "producer node name malformed")
            for key in ("runtime_environment_sha256", "runtime_attestation_marker_sha256"):
                _fail(bool(SHA_PATTERN.fullmatch(record[key])), "runtime binding digest malformed")
            nodes[twin] = record["node_name"]
            runtime_sha256.add(record["runtime_environment_sha256"])
            runtime_marker_sha256.add(record["runtime_attestation_marker_sha256"])
        _fail(nodes[0] != nodes[1], "producer twins did not execute on distinct nodes")
        _fail(
            len(runtime_sha256) == 1 and len(runtime_marker_sha256) == 1,
            "producer twins did not bind the same job-scoped runtime",
        )

        for twin in (0, 1):
            sibling = 1 - twin
            ack = _strict_record(root_fd, f"{twin}.ack", 4)
            _fail(
                ack
                == {
                    "array_job_id": producer_job_id,
                    "array_task_id": str(twin),
                    "git_commit": expected_commit,
                    "observed_sibling_receipt_sha256": receipt_images[sibling].sha256,
                },
                "producer receipt cross-ack mismatch",
            )
        completions = {twin: _strict_record(root_fd, f"{twin}.completion", 17) for twin in (0, 1)}
        completion_images = {
            twin: _read_file(root_fd, f"{twin}.completion", 16 * 1024, required_mode=0o444)
            for twin in (0, 1)
        }
        digest_fields = {
            "runtime_environment_sha256",
            "staging_manifest_sha256",
            "staging_completion_marker_sha256",
            "namespace_manifest_sha256",
            "namespace_sums_sha256",
            "namespace_completion_marker_sha256",
            "generator_completion_marker_sha256",
        }
        inode_fields = {
            "staging_root_dev",
            "staging_root_ino",
            "namespace_root_dev",
            "namespace_root_ino",
            "generator_root_dev",
            "generator_root_ino",
        }
        identity_fields = {"array_job_id", "array_task_id", "git_commit", "status"}
        for twin, completion in completions.items():
            _fail(
                set(completion) == digest_fields | inode_fields | identity_fields,
                "producer completion schema mismatch",
            )
            _fail(
                completion["array_job_id"] == producer_job_id
                and completion["array_task_id"] == str(twin)
                and completion["git_commit"] == expected_commit
                and completion["status"] == "completed_pending_independent_verification"
                and completion["runtime_environment_sha256"] in runtime_sha256,
                "producer completion identity/status mismatch",
            )
            for key in digest_fields:
                _fail(
                    bool(SHA_PATTERN.fullmatch(completion[key])),
                    f"producer completion digest malformed: {key}",
                )
            for key in inode_fields:
                _fail(
                    completion[key].isdigit() and int(completion[key]) > 0,
                    f"producer inode malformed: {key}",
                )
        for twin in (0, 1):
            sibling = 1 - twin
            ack = _strict_record(root_fd, f"{twin}.completion-ack", 4)
            _fail(
                ack
                == {
                    "array_job_id": producer_job_id,
                    "array_task_id": str(twin),
                    "git_commit": expected_commit,
                    "observed_sibling_completion_sha256": completion_images[sibling].sha256,
                },
                "producer completion cross-ack mismatch",
            )
        return (
            nodes,
            completions,
            next(iter(runtime_sha256)),
            next(iter(runtime_marker_sha256)),
        )
    finally:
        os.close(root_fd)


def _inventory_runtime_environment(environment: Path) -> dict[str, Any]:
    """Independently inventory a bounded job-scoped environment."""

    root_fd = _open_absolute_directory(environment)
    entries: dict[str, dict[str, int | str]] = {}
    total_bytes = 0

    def visit(directory_fd: int, prefix: str, depth: int) -> None:
        nonlocal total_bytes
        _fail(depth <= 16, "runtime environment exceeds depth cap")
        names: list[str] = []
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                _fail(
                    entry.name not in {"", ".", ".."}
                    and "/" not in entry.name
                    and len(entries) + len(names) < 20_000,
                    "runtime environment exceeds entry cap or has an unsafe name",
                )
                names.append(entry.name)
        for name in sorted(names):
            relative = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            common: dict[str, int | str] = {
                "dev": info.st_dev,
                "ino": info.st_ino,
                "mode": stat.S_IMODE(info.st_mode),
                "nlink": info.st_nlink,
            }
            if stat.S_ISDIR(info.st_mode):
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    held = os.fstat(descriptor)
                    _fail(
                        (held.st_dev, held.st_ino, held.st_mode, held.st_nlink)
                        == (info.st_dev, info.st_ino, info.st_mode, info.st_nlink),
                        "runtime directory was substituted",
                    )
                    entries[relative] = {"type": "directory", **common}
                    visit(descriptor, relative, depth + 1)
                    rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    _fail(
                        (rebound.st_dev, rebound.st_ino, rebound.st_mode, rebound.st_nlink)
                        == (held.st_dev, held.st_ino, held.st_mode, held.st_nlink),
                        "runtime directory changed while inventoried",
                    )
                finally:
                    os.close(descriptor)
            elif stat.S_ISREG(info.st_mode):
                _fail(info.st_size <= 256 * 1024 * 1024, "runtime file exceeds byte cap")
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=directory_fd,
                )
                try:
                    before = os.fstat(descriptor)
                    digest = hashlib.sha256()
                    observed = 0
                    while True:
                        chunk = os.read(descriptor, min(1 << 20, info.st_size + 1 - observed))
                        if not chunk:
                            break
                        digest.update(chunk)
                        observed += len(chunk)
                        _fail(observed <= 256 * 1024 * 1024, "runtime file exceeds byte cap")
                    after = os.fstat(descriptor)
                    rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    _fail(
                        _stat_fingerprint(before)
                        == _stat_fingerprint(after)
                        == _stat_fingerprint(rebound)
                        and observed == before.st_size,
                        "runtime file changed or was substituted",
                    )
                    total_bytes += observed
                    _fail(total_bytes <= 4 * 1024 * 1024 * 1024, "runtime exceeds total byte cap")
                    entries[relative] = {
                        "type": "file",
                        **common,
                        "bytes": observed,
                        "sha256": digest.hexdigest(),
                    }
                finally:
                    os.close(descriptor)
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(name, dir_fd=directory_fd)
                _fail(
                    bool(target) and len(os.fsencode(target)) <= 4_096,
                    "runtime symlink target is invalid",
                )
                rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                _fail(
                    (rebound.st_dev, rebound.st_ino, rebound.st_mode, rebound.st_nlink)
                    == (info.st_dev, info.st_ino, info.st_mode, info.st_nlink),
                    "runtime symlink changed while inventoried",
                )
                entries[relative] = {"type": "symlink", **common, "target": target}
            else:
                raise NamespaceVerificationError("runtime environment has an unsafe entry type")

    try:
        before = os.fstat(root_fd)
        visit(root_fd, "", 0)
        after = os.fstat(root_fd)
        _fail(
            (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_nlink),
            "runtime environment root changed",
        )
        return {
            "schema_version": 1,
            "artifact": "generator_oracle_namespace_runtime_inventory_v1",
            "root": {"dev": after.st_dev, "ino": after.st_ino},
            "entries": entries,
            "entry_count": len(entries),
            "total_regular_file_bytes": total_bytes,
        }
    finally:
        os.close(root_fd)


def _verify_runtime_attestation(
    *,
    environment: Path,
    attestation: Path,
    expected_inventory_sha256: str,
    expected_marker_sha256: str,
    expected_commit: str,
    expected_job_id: str,
    source_inventory: dict[str, str],
    role: str = "producer",
) -> dict[str, Any]:
    images, marker, root_inode, file_bindings = _consume_marker_committed_tree(
        attestation,
        marker_artifact="generator_oracle_namespace_runtime_inventory_complete_v1",
        expected_files={"environment_inventory.json"},
        maximum_file_bytes=16 * 1024 * 1024,
    )
    marker_sha256 = _sha(_canonical(marker))
    _fail(marker_sha256 == expected_marker_sha256, "runtime completion marker pin mismatch")
    image = images["environment_inventory.json"]
    _fail(image.sha256 == expected_inventory_sha256, "runtime inventory digest mismatch")
    try:
        document = json.loads(image.payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise NamespaceVerificationError("runtime inventory is invalid JSON") from error
    _bound_decoded_json(
        document,
        "runtime inventory",
        maximum_depth=20,
        maximum_containers=25_000,
        maximum_string_bytes=65_536,
    )
    _fail(
        isinstance(document, dict) and _canonical(document) == image.payload,
        "runtime inventory is not canonical JSON",
    )
    _fail(
        marker.get("identity")
        == {
            "environment_inventory_sha256": expected_inventory_sha256,
            "git_commit": expected_commit,
            "job_id": expected_job_id,
            "role": role,
        },
        "runtime marker identity mismatch",
    )
    expected = _inventory_runtime_environment(environment)
    expected.update(
        {
            "git_commit": expected_commit,
            "job_id": expected_job_id,
            "role": role,
            "pyproject_sha256": source_inventory[SOURCE_PATHS["pyproject"]],
            "uv_lock_sha256": source_inventory[SOURCE_PATHS["uv_lock"]],
        }
    )
    _fail(document == expected, "runtime environment reconstruction mismatch")
    return {
        "attestation_root_dev": root_inode[0],
        "attestation_root_ino": root_inode[1],
        "completion_marker_sha256": marker_sha256,
        "environment_inventory_sha256": expected_inventory_sha256,
        "inventory_file_binding": file_bindings["environment_inventory.json"],
        "entry_count": document["entry_count"],
        "total_regular_file_bytes": document["total_regular_file_bytes"],
    }


def _verify_staging(
    staging_root: Path,
    source_root: Path,
    config: VerifiedConfig,
    *,
    twin_id: int,
    producer_job_id: str,
    expected_commit: str,
    source_inventory: dict[str, str],
    runtime_environment_sha256: str,
) -> tuple[
    dict[str, FileImage],
    dict[str, Any],
    str,
    tuple[int, int],
    dict[str, dict[str, int | str]],
    str,
    dict[str, Any],
]:
    images, completion_marker, root_inode, staged_file_bindings = _consume_marker_committed_tree(
        staging_root,
        marker_artifact="generator_oracle_namespace_staging_complete_v2",
        expected_files=set(STAGING_FILES),
        maximum_file_bytes=EXPECTED_LIMITS["maximum_input_file_bytes"],
    )
    source_fd = _open_absolute_directory(source_root)
    try:
        manifest_image = images["STAGING_MANIFEST.json"]
        sums_image = images["STAGING_SHA256SUMS"]
        staged_inputs = {logical: images[relative] for logical, relative in INPUT_PATHS.items()}
        manifest = _json_document(manifest_image, "staging manifest")
        _fail(
            set(manifest)
            == {
                "schema_version",
                "artifact",
                "status",
                "claims",
                "producer_identity",
                "producer_inventory",
                "inputs",
            },
            "staging manifest schema mismatch",
        )
        _fail(
            manifest["schema_version"] == 1
            and not isinstance(manifest["schema_version"], bool)
            and manifest["artifact"] == "generator_oracle_namespace_staging_v1"
            and manifest["status"] == "authenticated_non_authorizing_staging_only"
            and manifest["claims"] == CLAIMS,
            "staging manifest identity/status mismatch",
        )
        identity = manifest["producer_identity"]
        _fail(
            isinstance(identity, dict)
            and identity
            == {
                "config_sha256": config.sha256,
                "git_commit": expected_commit,
                "job_id": producer_job_id,
                "source_path": SOURCE_PATHS["producer"],
                "source_sha256": source_inventory[SOURCE_PATHS["producer"]],
                "twin_id": twin_id,
                "runtime_environment_sha256": runtime_environment_sha256,
            },
            "staging producer identity mismatch",
        )
        _fail(
            manifest["producer_inventory"]
            == {
                SOURCE_PATHS["producer"]: source_inventory[SOURCE_PATHS["producer"]],
                SOURCE_PATHS["stager"]: source_inventory[SOURCE_PATHS["stager"]],
            },
            "staging producer source inventory mismatch",
        )
        _fail(
            completion_marker.get("identity")
            == {
                "config_sha256": config.sha256,
                "producer_git_commit": expected_commit,
                "producer_job_id": producer_job_id,
                "producer_source_sha256": source_inventory[SOURCE_PATHS["producer"]],
                "runtime_environment_sha256": runtime_environment_sha256,
                "stager_source_sha256": source_inventory[SOURCE_PATHS["stager"]],
                "twin_id": twin_id,
            },
            "staging completion marker identity mismatch",
        )
        evidence = manifest["inputs"]
        _fail(
            isinstance(evidence, dict) and set(evidence) == set(INPUT_PATHS),
            "staging evidence schema mismatch",
        )
        source_relatives = _source_relative_paths(twin_id)
        source_bindings: dict[str, dict[str, int | str]] = {}
        for logical, relative in INPUT_PATHS.items():
            item = evidence[logical]
            _fail(
                isinstance(item, dict)
                and set(item)
                == {
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
                },
                f"staging evidence fields mismatch: {logical}",
            )
            pin_sha, pin_bytes = config.input_pins[logical]
            authority_job, authority_receipt = AUTHORITY_BINDINGS[logical]
            source_image = _read_file(
                source_fd,
                source_relatives[logical],
                EXPECTED_LIMITS["maximum_input_file_bytes"],
            )
            source_mode = stat.S_IMODE(source_image.fingerprint[5])
            _fail(
                item
                == {
                    "bytes": pin_bytes,
                    "logical_role": logical,
                    "sha256": pin_sha,
                    "source_dev": source_image.fingerprint[0],
                    "source_ino": source_image.fingerprint[1],
                    "source_mode": source_mode,
                    "source_nlink": 1,
                    "source_relative_path": source_relatives[logical],
                    "source_size": pin_bytes,
                    "staged_relative_path": relative,
                    "upstream_job_id": authority_job,
                    "upstream_receipt_logical": authority_receipt,
                    "upstream_receipt_sha256": config.input_pins[authority_receipt][0],
                },
                f"staging source/job/inode binding mismatch: {logical}",
            )
            _fail(
                source_image.sha256 == pin_sha
                and source_image.size == pin_bytes
                and staged_inputs[logical].sha256 == pin_sha
                and staged_inputs[logical].size == pin_bytes
                and staged_inputs[logical].payload == source_image.payload,
                f"staged/source content mismatch: {logical}",
            )
            source_bindings[logical] = {
                "dev": source_image.fingerprint[0],
                "ino": source_image.fingerprint[1],
                "bytes": source_image.size,
                "sha256": source_image.sha256,
            }
        staging_checksum_paths = (
            *(INPUT_PATHS[logical] for logical in sorted(INPUT_PATHS, key=INPUT_PATHS.get)),
            "STAGING_MANIFEST.json",
        )
        sums = _checksum_lines(
            sums_image.payload,
            staging_checksum_paths,
            "staging SHA256SUMS",
        )
        _fail(
            tuple(sums) == staging_checksum_paths,
            "staging checksum order/inventory mismatch",
        )
        for logical, relative in INPUT_PATHS.items():
            _fail(sums[relative] == staged_inputs[logical].sha256, "staging checksum mismatch")
        _fail(
            sums["STAGING_MANIFEST.json"] == manifest_image.sha256,
            "staging manifest checksum mismatch",
        )
        for receipt in ("endpoint_receipt", "split_receipt", "corpus_receipt", "gate1_receipt"):
            value = _json_document(staged_inputs[receipt], receipt, require_canonical=False)
            _fail(value.get("status") == "passed", f"upstream authority did not pass: {receipt}")
        return (
            staged_inputs,
            manifest,
            manifest_image.sha256,
            root_inode,
            source_bindings,
            _sha(_canonical(completion_marker)),
            staged_file_bindings,
        )
    finally:
        os.close(source_fd)


def _verify_output(
    output_root: Path,
    config: VerifiedConfig,
    inputs: dict[str, FileImage],
    staging_manifest: dict[str, Any],
    staging_manifest_sha256: str,
    staging_completion_marker_sha256: str,
) -> tuple[
    DerivedNamespace,
    str,
    str,
    tuple[int, int],
    dict[str, str],
    dict[str, Any],
    str,
]:
    images, completion_marker, root_inode, output_file_bindings = _consume_marker_committed_tree(
        output_root,
        marker_artifact="generator_oracle_namespace_split_complete_v2",
        expected_files=set(OUTPUT_INVENTORY),
        maximum_file_bytes=EXPECTED_LIMITS["maximum_output_file_bytes"],
    )
    try:
        sums_image = images["SHA256SUMS"]
        sums = _checksum_lines(sums_image.payload, OUTPUT_FILES, "output SHA256SUMS")
        _fail(tuple(sums) == OUTPUT_FILES, "output checksum order/inventory mismatch")
        _fail(
            all(sums[name] == images[name].sha256 for name in OUTPUT_FILES),
            "output checksum mismatch",
        )
        identity = staging_manifest["producer_identity"]
        _fail(
            completion_marker.get("identity")
            == {
                "config_sha256": config.sha256,
                "producer_git_commit": identity["git_commit"],
                "producer_job_id": identity["job_id"],
                "producer_source_sha256": identity["source_sha256"],
                "runtime_environment_sha256": identity["runtime_environment_sha256"],
                "stager_source_sha256": staging_manifest["producer_inventory"][
                    SOURCE_PATHS["stager"]
                ],
                "staging_completion_marker_sha256": staging_completion_marker_sha256,
                "staging_manifest_sha256": staging_manifest_sha256,
                "surface": "audit_namespace",
                "twin_id": identity["twin_id"],
            },
            "namespace completion marker identity mismatch",
        )
        derived = _derive_namespace(
            config,
            inputs,
            producer_identity=identity,
            staging_manifest_sha256=staging_manifest_sha256,
        )
        for name, expected_payload in derived.payloads.items():
            _fail(
                images[name].payload == expected_payload,
                f"semantic reconstruction mismatch: {name}",
            )
            _json_rows(
                images[name],
                name,
                EXPECTED_LIMITS["maximum_output_records"],
                require_canonical=True,
            )
        manifest = _json_document(images["manifest.json"], "output manifest")
        expected_artifacts = {
            name: {
                "sha256": images[name].sha256,
                "bytes": images[name].size,
                "records": len(
                    _json_rows(
                        images[name],
                        name,
                        EXPECTED_LIMITS["maximum_output_records"],
                        require_canonical=True,
                    )
                ),
            }
            for name in OUTPUT_FILES
            if name != "manifest.json"
        }
        expected_manifest = {
            "schema_version": 1,
            "artifact": "generator_oracle_namespace_split_v1",
            "status": "pending_independent_verification",
            "claims": CLAIMS,
            "config_sha256": config.sha256,
            "staging": {
                "manifest_sha256": staging_manifest_sha256,
                "producer_identity": identity,
                "source_inputs": staging_manifest["inputs"],
            },
            "inputs": {
                logical: {"sha256": image.sha256, "bytes": image.size}
                for logical, image in sorted(inputs.items())
            },
            "artifacts": expected_artifacts,
        }
        _fail(manifest == expected_manifest, "output manifest reconstruction mismatch")
        return (
            derived,
            images["manifest.json"].sha256,
            sums_image.sha256,
            root_inode,
            {name: images[name].sha256 for name in SEMANTIC_TWIN_FILES},
            output_file_bindings,
            _sha(_canonical(completion_marker)),
        )
    finally:
        pass


def _verify_separated_surface(
    path: Path,
    *,
    marker_artifact: str,
    expected_files: tuple[str, ...],
    expected_payloads: dict[str, bytes],
    config: VerifiedConfig,
    staging_manifest: dict[str, Any],
    staging_manifest_sha256: str,
    staging_completion_marker_sha256: str,
    surface: str,
    require_generator_safe: bool,
) -> tuple[tuple[int, int], str, dict[str, Any]]:
    inventory = set(expected_files) | {"SHA256SUMS"}
    images, completion_marker, root_inode, file_bindings = _consume_marker_committed_tree(
        path,
        marker_artifact=marker_artifact,
        expected_files=inventory,
        maximum_file_bytes=EXPECTED_LIMITS["maximum_output_file_bytes"],
    )
    _fail(set(expected_payloads) == inventory, f"reconstructed {surface} inventory mismatch")
    for name, payload in expected_payloads.items():
        _fail(images[name].payload == payload, f"{surface} reconstruction mismatch: {name}")
    checksum = _checksum_lines(images["SHA256SUMS"].payload, expected_files, surface)
    _fail(
        all(checksum[name] == images[name].sha256 for name in expected_files),
        f"{surface} checksum mismatch",
    )
    producer_identity = staging_manifest["producer_identity"]
    expected_identity = {
        "config_sha256": config.sha256,
        "producer_git_commit": producer_identity["git_commit"],
        "producer_job_id": producer_identity["job_id"],
        "producer_source_sha256": producer_identity["source_sha256"],
        "runtime_environment_sha256": producer_identity["runtime_environment_sha256"],
        "stager_source_sha256": staging_manifest["producer_inventory"][SOURCE_PATHS["stager"]],
        "staging_completion_marker_sha256": staging_completion_marker_sha256,
        "staging_manifest_sha256": staging_manifest_sha256,
        "surface": surface,
        "twin_id": producer_identity["twin_id"],
    }
    _fail(
        completion_marker.get("identity") == expected_identity,
        f"{surface} completion marker identity mismatch",
    )
    if require_generator_safe:
        visible = b"".join(images[name].payload for name in sorted(images)) + _canonical(
            completion_marker
        )
        _fail(
            all(
                token not in visible
                for token in (b'"label"', b'"gram"', b'"canonical_target"', b'"activity_')
            ),
            "generator-visible surface contains an audit-only field",
        )
        for name in expected_files:
            _json_rows(
                images[name],
                f"generator view {name}",
                EXPECTED_LIMITS["maximum_output_records"],
                require_canonical=True,
            )
    return root_inode, _sha(_canonical(completion_marker)), file_bindings


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        _fail(written > 0, "short publication write")
        remaining = remaining[written:]


def _write_probe_file(parent_fd: int, name: str, payload: bytes) -> FileImage:
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
        dir_fd=parent_fd,
    )
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        image = _read_descriptor(descriptor, len(payload), name)
        _fail(image.payload == payload, "probe file readback mismatch")
        return image
    finally:
        os.close(descriptor)


def _run_publication_probe(probe_root: Path) -> dict[str, Any]:
    """Exercise mkdir claims and single-link marker commit on the target filesystem."""

    _fail(
        probe_root.is_absolute()
        and probe_root.name not in {"", ".", ".."}
        and all(part not in {"", ".", ".."} for part in probe_root.parts[1:]),
        "unsafe probe root",
    )
    parent_fd = _open_absolute_directory(probe_root.parent)
    probe_fd = -1
    prepared_fd = -1
    linked_fd = -1
    marker_name = f"{probe_root.name}.complete"
    temporary = f".{marker_name}.prepared-{os.getpid()}-{secrets.token_hex(12)}"
    try:
        try:
            os.mkdir(probe_root.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise NamespaceVerificationError("publication probe claim already exists") from error
        probe_fd = os.open(
            probe_root.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        root = os.fstat(probe_fd)
        root_entry = os.stat(probe_root.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            (root.st_dev, root.st_ino, stat.S_IMODE(root.st_mode))
            == (root_entry.st_dev, root_entry.st_ino, 0o700),
            "probe claim entry/descriptor mismatch",
        )
        try:
            os.mkdir(probe_root.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            mkdir_race_preserved_owner = True
        else:
            raise NamespaceVerificationError("second probe publisher acquired the owner claim")
        rebound = os.stat(probe_root.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            (rebound.st_dev, rebound.st_ino) == (root.st_dev, root.st_ino),
            "failed probe claim changed owner root",
        )

        payload = _canonical(
            {
                "artifact": "generator_oracle_namespace_target_filesystem_probe_payload_v2",
                "claims": CLAIMS,
                "status": "non_authorizing_primitive_probe",
            }
        )
        image = _write_probe_file(probe_fd, "probe.json", payload)
        os.fchmod(probe_fd, 0o555)
        os.fsync(probe_fd)
        os.fsync(parent_fd)
        root = os.fstat(probe_fd)
        marker = {
            "schema_version": 2,
            "artifact": "generator_oracle_namespace_target_filesystem_probe_complete_v2",
            "status": "committed_complete",
            "commit_protocol": "mkdirat_claim_populate_held_dirfd_then_single_link_marker_v2",
            "identity": {
                "filesystem_probe": True,
                "mkdirat_race_preserved_owner": mkdir_race_preserved_owner,
            },
            "root": {"dev": root.st_dev, "ino": root.st_ino, "mode": 0o555},
            "directories": {},
            "files": {
                "probe.json": {
                    "bytes": image.size,
                    "dev": image.fingerprint[0],
                    "ino": image.fingerprint[1],
                    "mode": 0o444,
                    "nlink": 1,
                    "sha256": image.sha256,
                }
            },
        }
        marker_payload = _canonical(marker)
        prepared_fd = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_fd,
        )
        _write_all(prepared_fd, marker_payload)
        os.fchmod(prepared_fd, 0o444)
        os.fsync(prepared_fd)
        prepared = _read_descriptor(prepared_fd, len(marker_payload), "prepared probe marker")
        _fail(prepared.payload == marker_payload, "prepared probe marker mismatch")
        try:
            os.link(
                temporary,
                marker_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise NamespaceVerificationError("probe marker destination already exists") from error
        linked_fd = os.open(
            marker_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        linked = _read_descriptor(
            linked_fd,
            len(marker_payload),
            "linked probe marker",
            required_nlink=2,
        )
        rebound_prepared = _read_descriptor(
            prepared_fd,
            len(marker_payload),
            "prepared probe marker",
            required_nlink=2,
        )
        marker_entry = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        root_entry = os.stat(probe_root.name, dir_fd=parent_fd, follow_symlinks=False)
        probe_entry = os.stat("probe.json", dir_fd=probe_fd, follow_symlinks=False)
        _fail(
            linked.payload == marker_payload
            and rebound_prepared.payload == marker_payload
            and (linked.fingerprint[0], linked.fingerprint[1])
            == (prepared.fingerprint[0], prepared.fingerprint[1])
            and (marker_entry.st_dev, marker_entry.st_ino, marker_entry.st_nlink)
            == (prepared.fingerprint[0], prepared.fingerprint[1], 2)
            and (root_entry.st_dev, root_entry.st_ino, stat.S_IMODE(root_entry.st_mode))
            == (root.st_dev, root.st_ino, 0o555)
            and (probe_entry.st_dev, probe_entry.st_ino, probe_entry.st_nlink)
            == (image.fingerprint[0], image.fingerprint[1], 1),
            "probe marker/root/file binding mismatch",
        )
        os.fsync(parent_fd)
        result = {
            "status": "passed",
            "filesystem_dev": root.st_dev,
            "probe_root_ino": root.st_ino,
            "mkdirat_claim_winner_preserved": True,
            "single_link_marker_commit_used": True,
            "probe_marker_sha256": _sha(marker_payload),
        }
        final_linked = _read_descriptor(
            linked_fd,
            len(marker_payload),
            "linked probe marker at commit",
            required_nlink=2,
        )
        final_prepared = _read_descriptor(
            prepared_fd,
            len(marker_payload),
            "prepared probe marker at commit",
            required_nlink=2,
        )
        final_marker_entry = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        final_root_entry = os.stat(probe_root.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            final_linked.payload == marker_payload
            and final_prepared.payload == marker_payload
            and (final_marker_entry.st_dev, final_marker_entry.st_ino, final_marker_entry.st_nlink)
            == (prepared.fingerprint[0], prepared.fingerprint[1], 2)
            and (final_root_entry.st_dev, final_root_entry.st_ino, final_root_entry.st_mode)
            == (root.st_dev, root.st_ino, root.st_mode),
            "probe publication changed at commit",
        )
        # Sole commit point. Until this unlink, the marker has two links and
        # consumers reject it. No publication operation follows.
        os.unlink(temporary, dir_fd=parent_fd)
        return result
    finally:
        if linked_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(linked_fd)
        if prepared_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(prepared_fd)
        if probe_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(probe_fd)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def _atomic_publish_receipt(output: Path, payload: bytes) -> None:
    """Hard-link a pre-fsynced receipt; temporary unlink is the commit point."""

    _fail(
        output.is_absolute()
        and output.name not in {"", ".", ".."}
        and all(part not in {"", ".", ".."} for part in output.parts[1:]),
        "unsafe receipt path",
    )
    _fail(payload.endswith(b"\n") and 0 < len(payload) <= 1_048_576, "invalid receipt payload")
    parent_fd = _open_absolute_directory(output.parent)
    temporary = f".{output.name}.prepared-{os.getpid()}-{secrets.token_hex(12)}"
    descriptor = -1
    linked_descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        prepared = _read_descriptor(descriptor, len(payload), "prepared receipt")
        _fail(prepared.payload == payload, "prepared receipt content mismatch")
        try:
            os.link(
                temporary,
                output.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise NamespaceVerificationError("audit receipt already exists") from error
        linked_descriptor = os.open(
            output.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        linked = _read_descriptor(
            linked_descriptor,
            len(payload),
            "linked receipt",
            required_nlink=2,
        )
        rebound = _read_descriptor(
            descriptor,
            len(payload),
            "prepared receipt",
            required_nlink=2,
        )
        entry = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            (entry.st_dev, entry.st_ino, entry.st_nlink)
            == (prepared.fingerprint[0], prepared.fingerprint[1], 2)
            and linked.payload == payload
            and rebound.payload == payload
            and (linked.fingerprint[0], linked.fingerprint[1])
            == (prepared.fingerprint[0], prepared.fingerprint[1])
            and (rebound.fingerprint[0], rebound.fingerprint[1])
            == (prepared.fingerprint[0], prepared.fingerprint[1]),
            "receipt publication inode/content mismatch",
        )
        os.fsync(parent_fd)
        final_linked = _read_descriptor(
            linked_descriptor,
            len(payload),
            "linked receipt at commit",
            required_nlink=2,
        )
        final_prepared = _read_descriptor(
            descriptor,
            len(payload),
            "prepared receipt at commit",
            required_nlink=2,
        )
        final_entry = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        _fail(
            final_linked.payload == payload
            and final_prepared.payload == payload
            and (final_entry.st_dev, final_entry.st_ino, final_entry.st_nlink)
            == (prepared.fingerprint[0], prepared.fingerprint[1], 2),
            "receipt publication changed at commit",
        )
        # Sole commit point. A destination with link count two is prepared and
        # rejected; a single-link destination is committed. No operation follows.
        os.unlink(temporary, dir_fd=parent_fd)
    finally:
        if linked_descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(linked_descriptor)
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def _read_committed_receipt(path: Path) -> tuple[dict[str, Any], FileImage]:
    _fail(
        path.is_absolute()
        and path.name not in {"", ".", ".."}
        and all(part not in {"", ".", ".."} for part in path.parts[1:]),
        "unsafe receipt path",
    )
    parent_fd = _open_absolute_directory(path.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        image = _read_descriptor(descriptor, 1_048_576, "committed receipt")
        document = _json_document(image, "committed receipt")
        entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        rebound = _read_descriptor(descriptor, 1_048_576, "committed receipt")
        _fail(
            (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink)
            == (
                image.fingerprint[0],
                image.fingerprint[1],
                image.fingerprint[5],
                1,
            )
            and rebound.fingerprint == image.fingerprint
            and rebound.payload == image.payload,
            "committed receipt was substituted or changed",
        )
        return document, image
    finally:
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def verify_twins(
    *,
    config_path: Path,
    expected_config_sha256: str,
    repository_root: Path,
    expected_audit_git_commit: str,
    expected_producer_git_commit: str,
    scratch_prefix: Path,
    source_root: Path,
    producer_job_root: Path,
    producer_job_id: str,
    audit_job_id: str,
    audit_node: str,
    scheduler_excluded_nodes: tuple[str, ...],
    audit_runtime_environment_sha256: str,
    audit_runtime_attestation_marker_sha256: str,
    probe_root: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    _fail(bool(JOB_PATTERN.fullmatch(producer_job_id)), "producer job ID is malformed")
    _fail(bool(JOB_PATTERN.fullmatch(audit_job_id)), "audit job ID is malformed")
    _fail(bool(NODE_PATTERN.fullmatch(audit_node)), "audit node name is malformed")
    for digest in (
        audit_runtime_environment_sha256,
        audit_runtime_attestation_marker_sha256,
    ):
        _fail(bool(SHA_PATTERN.fullmatch(digest)), "audit runtime digest is malformed")
    _fail(scratch_prefix.is_absolute(), "scratch prefix must be absolute")
    for path, label in (
        (source_root, "source root"),
        (producer_job_root, "producer job root"),
        (probe_root, "publication probe root"),
        (output_receipt, "audit candidate receipt"),
    ):
        _fail(path.is_absolute() and _is_within(path, scratch_prefix), f"{label} escaped scratch")
    _fail(
        not _is_within(probe_root, producer_job_root)
        and not _is_within(output_receipt, producer_job_root)
        and not _is_within(output_receipt, probe_root)
        and not _is_within(probe_root, output_receipt.parent),
        "audit publication overlaps a verified or sibling tree",
    )
    audit_source_inventory = _authenticate_repository(repository_root, expected_audit_git_commit)
    producer_source_inventory = _authenticate_producer_commit(
        repository_root,
        expected_producer_git_commit,
        expected_audit_git_commit,
    )
    config = _load_config(config_path, expected_config_sha256)
    _fail(
        audit_source_inventory[SOURCE_PATHS["config"]] == config.sha256
        and producer_source_inventory[SOURCE_PATHS["config"]] == config.sha256,
        "config differs from authenticated repository inventory",
    )
    nodes, completions, runtime_sha256, runtime_marker_sha256 = _verify_handshake_and_completions(
        producer_job_root / "records",
        producer_job_id=producer_job_id,
        expected_commit=expected_producer_git_commit,
        config_sha256=config.sha256,
        producer_source_sha256=producer_source_inventory[SOURCE_PATHS["producer"]],
        stager_source_sha256=producer_source_inventory[SOURCE_PATHS["stager"]],
        record_source_sha256=producer_source_inventory[SOURCE_PATHS["record"]],
        runtime_source_sha256=producer_source_inventory[SOURCE_PATHS["runtime"]],
        launcher_sha256=producer_source_inventory[SOURCE_PATHS["producer_launcher"]],
    )
    producer_scheduler = _sacct_completed_job(
        producer_job_id,
        expected_job_name="amp-ns-split-v2",
        expected_alloc_cpus=4,
        expected_nodes=2,
        expected_total_memory_mib=16_384,
    )
    _fail(
        set(producer_scheduler["nodes"]) == set(nodes.values()),
        "authoritative producer scheduler nodes differ from filesystem records",
    )
    _fail(
        len(scheduler_excluded_nodes) == 2
        and all(bool(NODE_PATTERN.fullmatch(node)) for node in scheduler_excluded_nodes)
        and set(scheduler_excluded_nodes) == set(nodes.values())
        and audit_node not in scheduler_excluded_nodes,
        "scheduler exclusion does not bind both producer nodes",
    )
    producer_runtime = _verify_runtime_attestation(
        environment=producer_job_root / "runtime" / "environment",
        attestation=producer_job_root / "runtime" / "attestation",
        expected_inventory_sha256=runtime_sha256,
        expected_marker_sha256=runtime_marker_sha256,
        expected_commit=expected_producer_git_commit,
        expected_job_id=producer_job_id,
        source_inventory=producer_source_inventory,
    )

    evidence: list[TwinEvidence] = []
    for twin_id in (0, 1):
        (
            inputs,
            staging,
            staging_sha,
            staging_inode,
            source_bindings,
            staging_marker_sha,
            staging_file_bindings,
        ) = _verify_staging(
            producer_job_root / "staging" / str(twin_id),
            source_root,
            config,
            twin_id=twin_id,
            producer_job_id=producer_job_id,
            expected_commit=expected_producer_git_commit,
            source_inventory=producer_source_inventory,
            runtime_environment_sha256=runtime_sha256,
        )
        (
            derived,
            manifest_sha,
            sums_sha,
            output_inode,
            semantic_sha,
            output_file_bindings,
            output_marker_sha,
        ) = _verify_output(
            producer_job_root / "split" / str(twin_id),
            config,
            inputs,
            staging,
            staging_sha,
            staging_marker_sha,
        )
        generator_inode, generator_marker_sha, generator_file_bindings = _verify_separated_surface(
            producer_job_root / "generator-view" / str(twin_id),
            marker_artifact="generator_namespace_view_complete_v2",
            expected_files=GENERATOR_VIEW_FILES,
            expected_payloads=derived.generator_view_payloads,
            config=config,
            staging_manifest=staging,
            staging_manifest_sha256=staging_sha,
            staging_completion_marker_sha256=staging_marker_sha,
            surface="generator_view",
            require_generator_safe=True,
        )
        completion = completions[twin_id]
        expected_completion_bindings = {
            "staging_manifest_sha256": staging_sha,
            "staging_completion_marker_sha256": staging_marker_sha,
            "staging_root_dev": str(staging_inode[0]),
            "staging_root_ino": str(staging_inode[1]),
            "namespace_manifest_sha256": manifest_sha,
            "namespace_sums_sha256": sums_sha,
            "namespace_completion_marker_sha256": output_marker_sha,
            "namespace_root_dev": str(output_inode[0]),
            "namespace_root_ino": str(output_inode[1]),
            "generator_completion_marker_sha256": generator_marker_sha,
            "generator_root_dev": str(generator_inode[0]),
            "generator_root_ino": str(generator_inode[1]),
        }
        _fail(
            all(completion[key] == value for key, value in expected_completion_bindings.items()),
            f"producer completion inode/digest binding mismatch: twin {twin_id}",
        )
        surfaces = {
            "staging": {
                "root_dev": staging_inode[0],
                "root_ino": staging_inode[1],
                "completion_marker_sha256": staging_marker_sha,
                "file_bindings": staging_file_bindings,
            },
            "audit_namespace": {
                "root_dev": output_inode[0],
                "root_ino": output_inode[1],
                "completion_marker_sha256": output_marker_sha,
                "file_bindings": output_file_bindings,
            },
            "generator_view": {
                "root_dev": generator_inode[0],
                "root_ino": generator_inode[1],
                "completion_marker_sha256": generator_marker_sha,
                "file_bindings": generator_file_bindings,
            },
        }
        evidence.append(
            TwinEvidence(
                twin_id=twin_id,
                node_name=nodes[twin_id],
                runtime_environment_sha256=runtime_sha256,
                staging_manifest_sha256=staging_sha,
                output_manifest_sha256=manifest_sha,
                output_sums_sha256=sums_sha,
                semantic_sha256=semantic_sha,
                source_bindings=source_bindings,
                metrics=derived.metrics,
                artifact_records=derived.artifact_records,
                surfaces=surfaces,
            )
        )
    _fail(
        evidence[0].semantic_sha256 == evidence[1].semantic_sha256
        and len(evidence[0].semantic_sha256) == 10,
        "producer twins differ in the exact ten provenance-invariant outputs",
    )
    _fail(evidence[0].metrics == evidence[1].metrics, "producer twin census differs")
    _fail(
        evidence[0].artifact_records == evidence[1].artifact_records,
        "producer twin artifact-record census differs",
    )
    for logical in INPUT_PATHS:
        _fail(
            evidence[0].source_bindings[logical]["sha256"]
            == evidence[1].source_bindings[logical]["sha256"],
            f"accepted source twins differ: {logical}",
        )
    probe = _run_publication_probe(probe_root)
    checks = {
        "repository_commit_and_cleanliness_authenticated": True,
        "producer_stager_record_runtime_verifier_and_launchers_bound_to_git": True,
        "fresh_locked_job_runtime_inventory_reconstructed": True,
        "staging_checksums_complete_inventory_and_file_inodes_verified": True,
        "source_job_config_twin_and_inode_bindings_reconstructed": True,
        "upstream_authority_receipt_content_pins_verified": True,
        "canonical_sequence_union_homology_and_study_disjointness_recomputed": True,
        "endpoint_availability_and_label_census_recomputed_in_separate_surface": True,
        "generator_visible_surface_is_label_free": True,
        "generator_fold_distribution_recomputed_with_independent_heap": True,
        "fold_triple_declaration_reconstructed": True,
        "all_outputs_manifests_markers_and_file_inodes_reconstructed": True,
        "exact_ten_provenance_invariant_outputs_match_between_twins": True,
        "distinct_nodes_authoritative_producer_accounting_and_audit_exclusion_verified": True,
        "target_filesystem_mkdir_claim_and_single_link_marker_probed": True,
        "atomic_no_replace_candidate_receipt_publication_used": True,
    }
    receipt = {
        "schema_version": 2,
        "artifact": "generator_oracle_namespace_split_independent_verification_candidate_v2",
        "status": "verification_passed_candidate_pending_scheduler_finalization",
        "claims": CLAIMS,
        "identity": {
            "producer_job_id": producer_job_id,
            "audit_job_id": audit_job_id,
            "audit_job_name": "amp-ns-split-audit-v2",
            "audit_node": audit_node,
            "audit_declared_resources": {
                "alloc_cpus": 4,
                "node_count": 1,
                "memory_per_node_mib": 16_384,
            },
            "scheduler_excluded_producer_nodes": sorted(scheduler_excluded_nodes),
            "audit_git_commit": expected_audit_git_commit,
            "producer_git_commit": expected_producer_git_commit,
            "config_sha256": config.sha256,
            "audit_runtime_environment_sha256": audit_runtime_environment_sha256,
            "audit_runtime_attestation_marker_sha256": audit_runtime_attestation_marker_sha256,
            "audit_source_inventory": audit_source_inventory,
            "producer_source_inventory": producer_source_inventory,
        },
        "producer_scheduler": producer_scheduler,
        "producer_runtime": producer_runtime,
        "checks": checks,
        "census": evidence[0].metrics,
        "artifact_records": evidence[0].artifact_records,
        "overlap": {
            "exact_sequences": 0,
            "union_components": 0,
            "homology_components": 0,
            "study_keys": 0,
        },
        "fold_triples": list(FOLD_TRIPLES),
        "provenance_invariant_output_names": sorted(evidence[0].semantic_sha256),
        "twins": [
            {
                "twin_id": item.twin_id,
                "node_name": item.node_name,
                "runtime_environment_sha256": item.runtime_environment_sha256,
                "staging_manifest_sha256": item.staging_manifest_sha256,
                "output_manifest_sha256": item.output_manifest_sha256,
                "output_sums_sha256": item.output_sums_sha256,
                "semantic_sha256": item.semantic_sha256,
                "source_bindings": item.source_bindings,
                "surfaces": item.surfaces,
            }
            for item in evidence
        ],
        "publication_probe": probe,
        "candidate_state": {
            "content_verification_passed": True,
            "authoritative_audit_sacct_pending": True,
            "formal_commit_point": "single_link_candidate_receipt",
            "final_receipt_required": True,
        },
        "validity_markers": {
            "staging_authenticated": True,
            "namespace_reconstruction_passed": True,
            "producer_twins_passed": True,
            "content_verification_passed": True,
            "authoritative_scheduler_finalization_passed": False,
            "independent_verification_passed": False,
            "downstream_execution_authorized": False,
            "scientific_evidence_accepted": False,
            "production_input_eligible": False,
        },
    }
    _fail(all(checks.values()), "independent content verification did not pass every check")
    payload = _canonical(receipt)
    _fail(
        b"/lustre/" not in payload and b"/home/" not in payload, "candidate leaks an absolute path"
    )
    _atomic_publish_receipt(output_receipt, payload)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-audit-git-commit", required=True)
    parser.add_argument("--expected-producer-git-commit", required=True)
    parser.add_argument("--scratch-prefix", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--producer-job-root", type=Path, required=True)
    parser.add_argument("--producer-job-id", required=True)
    parser.add_argument("--audit-job-id", required=True)
    parser.add_argument("--audit-node", required=True)
    parser.add_argument("--scheduler-excluded-node", action="append", required=True)
    parser.add_argument("--audit-runtime-environment-sha256", required=True)
    parser.add_argument("--audit-runtime-attestation-marker-sha256", required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = verify_twins(
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        repository_root=args.repository_root,
        expected_audit_git_commit=args.expected_audit_git_commit,
        expected_producer_git_commit=args.expected_producer_git_commit,
        scratch_prefix=args.scratch_prefix,
        source_root=args.source_root,
        producer_job_root=args.producer_job_root,
        producer_job_id=args.producer_job_id,
        audit_job_id=args.audit_job_id,
        audit_node=args.audit_node,
        scheduler_excluded_nodes=tuple(args.scheduler_excluded_node),
        audit_runtime_environment_sha256=args.audit_runtime_environment_sha256,
        audit_runtime_attestation_marker_sha256=(args.audit_runtime_attestation_marker_sha256),
        probe_root=args.probe_root,
        output_receipt=args.output_receipt,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "producer_job_id": args.producer_job_id,
                "receipt_sha256": _sha(_canonical(receipt)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare and package the accepted union-panel ESM2 embedding candidate.

The legacy extraction worker is deliberately left byte-for-byte unchanged.  The
``prepare-worker-input`` command adapts the independently accepted union FASTA
receipt to the worker's private manifest schema.  The ``package`` command then
validates every raw worker output and all of its provenance before publishing a
small, path-free semantic directory.  Publication is candidate evidence only:
an independent full GPU re-extraction is still required.

This module imports no benchmark producer.  Apart from NumPy, its implementation
uses only the Python standard library so that validation does not depend on the
legacy ESM environment.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_SEQUENCE_HEADER_RE = re.compile(r">sequence_id=([0-9a-f]{64})")
_DRIVER_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")
_SAFE_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ABSOLUTE_BYTES_RE = re.compile(
    rb"(?:/home/|/lustre/|/tmp/|/scratch/|/mnt/|file://|"
    rb"(?<![A-Za-z0-9])[A-Za-z]:[\\\\/])"
)

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MIN_SEQUENCE_LENGTH = 8
_MAX_SEQUENCE_LENGTH = 50
_EXPECTED_SEQUENCES = 952
_EXPECTED_DIMENSION = 320
_EXPECTED_TENSOR_BYTES = _EXPECTED_SEQUENCES * _EXPECTED_DIMENSION * 4
_EXPECTED_NPY_BYTES = _EXPECTED_TENSOR_BYTES + 128
_EXPECTED_PANEL_CONFIG_SHA256 = "616ea482e3310e71d3712dd849c9307f4fd490842d4fb38e268ee1abc31d5a4c"
_EXPECTED_PANEL_CODE_MANIFEST_SHA256 = (
    "76dc4cc13f96a7df6178bef06a58987e4d48789f9a60748ca516a8f81cfbffe1"
)
_EXPECTED_PANEL_FROZEN_MANIFEST_SHA256 = (
    "74e0383dabf5a0935afd65c4aec859893a3768516ebf70155ac368345928ec19"
)
_EXPECTED_PANEL_GIT_COMMIT = "b63c47150e19fde0a8af8ce51830f3bccf652af4"
_EXPECTED_CONFIG_HASHES = {
    "panel_publication_top_sha256": "8815e45b46bde98474e3a85711da082457da47102e1d73cf33ed1806932a1894",
    "panel_top_sha256": "0f56badbdebec55c24c799d62ce0358f5aec846a7675c9afccbf550c884a2ba8",
    "panel_fasta_sha256": "fd8b791687578c7a8b9ea7212b5319a3aaf8b46e03d8988706772ecc4d11e419",
    "panel_manifest_sha256": "bc04dbcc8e09cfd27dc83dbc9610fe4250b9b5b279495b324b8d6d4138d16900",
    "panel_coverage_receipt_sha256": "f3602885542fc319848e588a8eaffb5655f1afb167678f95772b9ff027132a2b",
    "panel_independent_receipt_sha256": "f3b121f30deea0d883f53a4a5a2d270b87d1e8726f0941bf5efaef7eb2215de8",
    "expected_sequence_ids_sha256": "45a704812a51d51876b8e32e86c8890db5b7d6aa52de1b1f35634e797acd3f03",
    "embedding_worker_sha256": "34fdb77e87aaa44044960655259cc60fb20b165bfb87c38ba19bee21565f2cba",
    "trust_manifest_sha256": "5e62db4fd3241e1fa6bbd83cc0f26e299c6d16e9fa1e449304fab2e4c98fd9ff",
    "bundle_verification_receipt_sha256": "ad9655f57cbb4d033c88fe38dfdfeffb56fd24274749c2caded890667e63e850",
    "model_checkpoint_sha256": "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928",
    "contact_regression_sha256": "8f7a4557d57713b97ba0e484303007efb7230d25299c0ac47a0a1b12a87bbb9d",
    "environment_lock_sha256": "aaf37baa3adf5070dd3090c43527daa2696741a7e3057c72c78d51a80af9e234",
}
_EXPECTED_AMPDIFFUSION_COMMIT = "1a862af9078e6b55c87d1fa576f3da81851ba94b"

_CONFIG_LOGICAL_PATH = "configs/models/esm2_union_embeddings_v1.toml"
_PACKAGE_MODULE_PATH = "src/amp_challenge/benchmarks/package_union_esm_embeddings.py"
_VERIFIER_MODULE_PATH = "src/amp_challenge/benchmarks/verify_union_esm_embeddings.py"
_WORKER_LOGICAL_PATH = "integrations/ampdiffusion/esm2_embedding_worker.py"
_ADAPTER_LOGICAL_PATH = "integrations/ampdiffusion/adapter.py"
_TRUST_MANIFEST_LOGICAL_PATH = "integrations/ampdiffusion/artifacts.toml"
_REFERENCE_WORKER_LOGICAL_PATH = "integrations/ampdiffusion/esm2_union_reference_worker.py"
_REQUIRED_CODE_PATHS = frozenset(
    {
        "cluster/slurm/audit_union_esm_embeddings_v1_twins.sbatch",
        "cluster/slurm/extract_union_esm_embeddings_v1_twins.sbatch",
        "cluster/slurm/validate_union_esm_embeddings.sbatch",
        "cluster/validate_union_esm_embeddings_output.sh",
        _CONFIG_LOGICAL_PATH,
        _ADAPTER_LOGICAL_PATH,
        _PACKAGE_MODULE_PATH,
        _REFERENCE_WORKER_LOGICAL_PATH,
        _VERIFIER_MODULE_PATH,
        _WORKER_LOGICAL_PATH,
        _TRUST_MANIFEST_LOGICAL_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)

_MODEL_RELATIVE_PATH = "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt"
_CONTACT_RELATIVE_PATH = "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt"
_LOCK_RELATIVE_PATH = "source/uv.lock"

_PANEL_PUBLICATION_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "panel/SHA256SUMS",
        "panel/coverage_receipt.json",
        "panel/manifest.json",
        "panel/union_esm_sequences.fasta",
    }
)
_PANEL_SEMANTIC_FILES = frozenset(
    {"SHA256SUMS", "coverage_receipt.json", "manifest.json", "union_esm_sequences.fasta"}
)
_RAW_EMBEDDING_FILES = frozenset(
    {"embeddings.npy", "embedding_index.csv", "embedding_manifest.json"}
)
_PUBLISHED_FILES = frozenset(
    {"SHA256SUMS", "embeddings.npy", "embedding_index.csv", "embedding_manifest.json"}
)

_PANEL_RECEIPT_CHECKS = frozenset(
    {
        "accepted_gate1_evidence_chain_valid",
        "accepted_gate1_twins_immutable_and_identical",
        "all_contexts_preserved_before_unique_sequence_projection",
        "all_semantic_artifacts_independently_reconstructed",
        "base_probabilities_not_consumed",
        "canonical_serialization_verified",
        "code_and_config_bound_to_synchronized_commit",
        "exact_sorted_fasta_reconstructed",
        "export_frozen_input_manifests_reconstructed",
        "export_twins_immutable_and_identical",
        "no_embedding_or_model_evidence_claimed",
        "path_free_artifacts_and_receipt",
        "production_overlap_handshake_valid",
        "union_components_fold_disjoint",
    }
)

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "panel_git_commit",
        "panel_producer_job_id",
        "panel_audit_job_id",
        "panel_publication_top_sha256",
        "panel_top_sha256",
        "panel_fasta_sha256",
        "panel_manifest_sha256",
        "panel_coverage_receipt_sha256",
        "panel_independent_receipt_sha256",
        "expected_sequences",
        "expected_sequence_ids_sha256",
        "embedding_model",
        "representation_layer",
        "embedding_dimension",
        "pooling",
        "embedding_dtype",
        "embedding_batch_size",
        "embedding_seed",
        "embedding_worker_sha256",
        "trust_manifest_sha256",
        "bundle_verification_receipt_sha256",
        "ampdiffusion_source_commit",
        "model_checkpoint_sha256",
        "contact_regression_sha256",
        "environment_lock_sha256",
        "runtime",
        "determinism",
        "acceptance",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "python",
        "torch",
        "fair_esm",
        "numpy",
        "cuda_runtime",
        "cudnn",
        "device_type",
        "device_name",
        "device_capability",
    }
)
_DETERMINISM_FIELDS = frozenset(
    {
        "cublas_workspace_config",
        "torch_deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "tf32",
        "exact_twin_bytes_required",
        "independent_full_reextraction_required",
        "cuda_driver_policy",
    }
)
_ACCEPTANCE_FIELDS = frozenset(
    {
        "producer_status",
        "verified_status",
        "embeddings_verified_by_producer",
        "model_predictions_verified",
        "model_performance_evidence",
        "ensemble_weight_established",
        "pretraining_membership_independence_established",
    }
)


class PackagingError(ValueError):
    """Raised when an embedding packaging invariant fails."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class FastaRecord:
    sequence_id: str
    sequence: str


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    path: Path
    sha256: str
    artifact: str
    panel_git_commit: str
    panel_producer_job_id: int
    panel_audit_job_id: int
    panel_publication_top_sha256: str
    panel_top_sha256: str
    panel_fasta_sha256: str
    panel_manifest_sha256: str
    panel_coverage_receipt_sha256: str
    panel_independent_receipt_sha256: str
    expected_sequences: int
    expected_sequence_ids_sha256: str
    embedding_model: str
    representation_layer: int
    embedding_dimension: int
    pooling: str
    embedding_dtype: str
    embedding_batch_size: int
    embedding_seed: int
    embedding_worker_sha256: str
    trust_manifest_sha256: str
    bundle_verification_receipt_sha256: str
    ampdiffusion_source_commit: str
    model_checkpoint_sha256: str
    contact_regression_sha256: str
    environment_lock_sha256: str
    runtime: Mapping[str, object]
    determinism: Mapping[str, object]
    acceptance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PanelEvidence:
    root: Path
    records: tuple[FastaRecord, ...]
    receipt: Mapping[str, object]
    receipt_snapshot: Snapshot
    snapshots: tuple[Snapshot, ...]
    handshake: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TrustEvidence:
    receipt_snapshot: Snapshot
    manifest_snapshot: Snapshot
    snapshots: tuple[Snapshot, ...]
    raw_worker_trust: Mapping[str, object]
    semantic_trust: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MatrixEvidence:
    snapshot: Snapshot
    tensor_payload: bytes
    tensor_data_sha256: str
    rows: int
    columns: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackagingError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(path).absolute()
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate):
            _require(not candidate.is_symlink(), f"{label} traverses a symbolic link")


def _read_snapshot(path: Path, *, label: str) -> Snapshot:
    requested = Path(path)
    _reject_symlink_chain(requested, label=label)
    try:
        before_path = requested.stat(follow_symlinks=False)
    except OSError as error:
        raise PackagingError(f"{label} is unavailable") from error
    _require(stat.S_ISREG(before_path.st_mode), f"{label} is not a regular file")
    _require(not requested.is_symlink(), f"{label} is a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise PackagingError(f"cannot securely open {label}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = requested.stat(follow_symlinks=False)
    except OSError as error:
        raise PackagingError(f"{label} changed while read") from error
    fingerprints = {_fingerprint(item) for item in (before_path, before, after, after_path)}
    _require(len(fingerprints) == 1, f"{label} changed while read")
    payload = b"".join(chunks)
    _require(len(payload) == before.st_size, f"{label} changed size while read")
    return Snapshot(
        path=requested.resolve(strict=True),
        payload=payload,
        sha256=_sha256(payload),
        fingerprint=_fingerprint(before),
    )


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    observed = _read_snapshot(snapshot.path, label=label)
    _require(
        observed.fingerprint == snapshot.fingerprint
        and observed.sha256 == snapshot.sha256
        and observed.payload == snapshot.payload,
        f"{label} changed during validation",
    )


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _reject_symlink_chain(requested, label=label)
    _require(not requested.is_symlink(), f"{label} is a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PackagingError(f"{label} is unavailable") from error
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is not a real directory")
    return resolved


def _resolved_regular_file(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _reject_symlink_chain(requested, label=label)
    try:
        metadata = requested.stat(follow_symlinks=False)
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PackagingError(f"{label} is unavailable") from error
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not requested.is_symlink()
        and resolved.is_file()
        and not resolved.is_symlink(),
        f"{label} is not a real regular file",
    )
    return resolved


def _exact_keys(value: Mapping[str, object], expected: Iterable[str], *, label: str) -> None:
    wanted = set(expected)
    observed = set(value)
    _require(
        observed == wanted,
        f"{label} schema mismatch: missing={sorted(wanted - observed)}, "
        f"extra={sorted(observed - wanted)}",
    )


def _mapping(value: object, *, label: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return cast(dict[str, object], value)


def _sha_field(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"bad {label}")
    return cast(str, value)


def _string(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and value == value.strip() and value != "", f"bad {label}")
    return cast(str, value)


def _positive_int(value: object, *, label: str) -> int:
    _require(type(value) is int and cast(int, value) > 0, f"{label} must be positive")
    return cast(int, value)


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise PackagingError(f"JSON contains non-finite constant {value}")


def _read_pretty_json(snapshot: Snapshot, *, label: str) -> dict[str, object]:
    _require(snapshot.payload.endswith(b"\n") and b"\r" not in snapshot.payload, f"{label} framing")
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackagingError(f"{label} is not valid UTF-8 JSON") from error
    document = _mapping(value, label=label)
    _require(snapshot.payload == _pretty_json_bytes(document), f"{label} is not canonical JSON")
    return document


def _assert_path_free(value: object, *, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), f"{label} has a non-string key")
            _assert_path_free(item, label=label)
        return
    if isinstance(value, list):
        for item in value:
            _assert_path_free(item, label=label)
        return
    if not isinstance(value, str):
        return
    _require(not value.startswith(("/", "~", "file://")), f"{label} exposes an absolute path")
    _require(_WINDOWS_ABSOLUTE_RE.match(value) is None, f"{label} exposes an absolute path")
    _require(
        _ABSOLUTE_BYTES_RE.search(value.encode("utf-8")) is None,
        f"{label} exposes an absolute path",
    )


def _scan_path_free_payloads(payloads: Iterable[bytes], forbidden_prefixes: Sequence[str]) -> None:
    prefixes = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(isinstance(prefix, str) and prefix != "", "forbidden prefix is empty")
        prefixes.append(prefix.encode("utf-8"))
    for payload in payloads:
        _require(_ABSOLUTE_BYTES_RE.search(payload) is None, "publication exposes an absolute path")
        for prefix in prefixes:
            _require(prefix not in payload, "publication exposes a forbidden path prefix")


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} framing")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise PackagingError(f"{label} is not UTF-8") from error
    entries: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"{label} row {number} is malformed")
        assert match is not None
        digest, name = match.groups()
        _safe_relative_path(name, label=f"{label} row {number}")
        _require(name not in entries, f"{label} repeats {name}")
        if previous is not None:
            _require(name > previous, f"{label} paths are not sorted")
        entries[name] = digest
        previous = name
    return entries


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("utf-8")


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    _require(value != "" and "\\" not in value, f"{label} path is unsafe")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} path is unsafe",
    )
    return path


def _file_inventory(root: Path) -> frozenset[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "validated tree contains a symbolic link")
        _require(path.is_dir() or path.is_file(), "validated tree contains a special entry")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return frozenset(files)


def _directory_inventory(root: Path) -> frozenset[str]:
    directories: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "validated tree contains a symbolic link")
        _require(path.is_dir() or path.is_file(), "validated tree contains a special entry")
        if path.is_dir():
            directories.add(path.relative_to(root).as_posix())
    return frozenset(directories)


def _verify_read_only_tree(root: Path, *, label: str, include_root: bool = True) -> None:
    paths = (root, *root.rglob("*")) if include_root else tuple(root.rglob("*"))
    for path in paths:
        _require(not path.is_symlink(), f"{label} contains a symbolic link")
        _require(path.stat().st_mode & 0o222 == 0, f"{label} contains a writable entry")


def _parse_config(snapshot: Snapshot) -> EmbeddingContract:
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PackagingError("embedding config is not valid UTF-8 TOML") from error
    _exact_keys(raw, _CONFIG_FIELDS, label="embedding config")
    _require(raw["schema_version"] == 1 and type(raw["schema_version"]) is int, "bad config schema")
    _require(raw["artifact"] == "gate1_union_esm2_embeddings_v1", "bad config artifact")
    commit = _string(raw["panel_git_commit"], label="panel Git commit")
    _require(_GIT_RE.fullmatch(commit) is not None, "bad panel Git commit")
    _require(commit == _EXPECTED_PANEL_GIT_COMMIT, "accepted panel Git commit changed")
    runtime = _mapping(raw["runtime"], label="runtime config")
    determinism = _mapping(raw["determinism"], label="determinism config")
    acceptance = _mapping(raw["acceptance"], label="acceptance config")
    _exact_keys(runtime, _RUNTIME_FIELDS, label="runtime config")
    _exact_keys(determinism, _DETERMINISM_FIELDS, label="determinism config")
    _exact_keys(acceptance, _ACCEPTANCE_FIELDS, label="acceptance config")
    expected_runtime = {
        "python": "3.10.19",
        "torch": "2.5.1+cu121",
        "fair_esm": "2.0.0",
        "numpy": "2.2.6",
        "cuda_runtime": "12.1",
        "cudnn": 90100,
        "device_type": "cuda",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "device_capability": [8, 0],
    }
    _require(runtime == expected_runtime, "runtime config differs from the accepted contract")
    expected_determinism = {
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "exact_twin_bytes_required": True,
        "independent_full_reextraction_required": True,
        "cuda_driver_policy": "producer_twins_and_independent_audit_exact_match",
    }
    _require(
        determinism == expected_determinism,
        "determinism config differs from the accepted contract",
    )
    expected_acceptance = {
        "producer_status": "candidate_pending_independent_numerical_verification",
        "verified_status": "accepted_for_downstream_embedding_feature_input_only",
        "embeddings_verified_by_producer": False,
        "model_predictions_verified": False,
        "model_performance_evidence": False,
        "ensemble_weight_established": False,
        "pretraining_membership_independence_established": False,
    }
    _require(acceptance == expected_acceptance, "acceptance config overstates the evidence")
    expected_sequences = _positive_int(raw["expected_sequences"], label="expected sequences")
    embedding_dimension = _positive_int(raw["embedding_dimension"], label="embedding dimension")
    _require(expected_sequences == _EXPECTED_SEQUENCES, "expected sequence count must be 952")
    _require(embedding_dimension == _EXPECTED_DIMENSION, "embedding dimension must be 320")
    _require(raw["panel_producer_job_id"] == 223287, "accepted panel producer job changed")
    _require(raw["panel_audit_job_id"] == 223294, "accepted panel audit job changed")
    _require(raw["embedding_model"] == "esm2_t6_8M_UR50D", "bad embedding model")
    _require(raw["representation_layer"] == 6, "bad representation layer")
    _require(
        raw["pooling"] == "arithmetic mean over residue representations; BOS/EOS/padding excluded",
        "bad pooling contract",
    )
    _require(raw["embedding_dtype"] == "float32", "bad embedding dtype")
    _require(raw["embedding_batch_size"] == 128, "bad embedding batch size")
    _require(raw["embedding_seed"] == 20260902, "bad embedding seed")
    hashes = {
        name: _sha_field(raw[name], label=name)
        for name in (
            "panel_publication_top_sha256",
            "panel_top_sha256",
            "panel_fasta_sha256",
            "panel_manifest_sha256",
            "panel_coverage_receipt_sha256",
            "panel_independent_receipt_sha256",
            "expected_sequence_ids_sha256",
            "embedding_worker_sha256",
            "trust_manifest_sha256",
            "bundle_verification_receipt_sha256",
            "model_checkpoint_sha256",
            "contact_regression_sha256",
            "environment_lock_sha256",
        )
    }
    _require(hashes == _EXPECTED_CONFIG_HASHES, "configured content addresses changed")
    source_commit = _string(raw["ampdiffusion_source_commit"], label="source commit")
    _require(_GIT_RE.fullmatch(source_commit) is not None, "bad AMP-Diffusion source commit")
    _require(
        source_commit == _EXPECTED_AMPDIFFUSION_COMMIT,
        "AMP-Diffusion source commit changed",
    )
    return EmbeddingContract(
        path=snapshot.path,
        sha256=snapshot.sha256,
        artifact=cast(str, raw["artifact"]),
        panel_git_commit=commit,
        panel_producer_job_id=_positive_int(
            raw["panel_producer_job_id"], label="panel producer job ID"
        ),
        panel_audit_job_id=_positive_int(raw["panel_audit_job_id"], label="panel audit job ID"),
        panel_publication_top_sha256=hashes["panel_publication_top_sha256"],
        panel_top_sha256=hashes["panel_top_sha256"],
        panel_fasta_sha256=hashes["panel_fasta_sha256"],
        panel_manifest_sha256=hashes["panel_manifest_sha256"],
        panel_coverage_receipt_sha256=hashes["panel_coverage_receipt_sha256"],
        panel_independent_receipt_sha256=hashes["panel_independent_receipt_sha256"],
        expected_sequences=expected_sequences,
        expected_sequence_ids_sha256=hashes["expected_sequence_ids_sha256"],
        embedding_model=cast(str, raw["embedding_model"]),
        representation_layer=cast(int, raw["representation_layer"]),
        embedding_dimension=embedding_dimension,
        pooling=cast(str, raw["pooling"]),
        embedding_dtype=cast(str, raw["embedding_dtype"]),
        embedding_batch_size=cast(int, raw["embedding_batch_size"]),
        embedding_seed=cast(int, raw["embedding_seed"]),
        embedding_worker_sha256=hashes["embedding_worker_sha256"],
        trust_manifest_sha256=hashes["trust_manifest_sha256"],
        bundle_verification_receipt_sha256=hashes["bundle_verification_receipt_sha256"],
        ampdiffusion_source_commit=source_commit,
        model_checkpoint_sha256=hashes["model_checkpoint_sha256"],
        contact_regression_sha256=hashes["contact_regression_sha256"],
        environment_lock_sha256=hashes["environment_lock_sha256"],
        runtime=dict(runtime),
        determinism=dict(determinism),
        acceptance=dict(acceptance),
    )


def load_config(path: str | Path) -> EmbeddingContract:
    """Load and strictly validate the embedding publication contract."""

    return _parse_config(_read_snapshot(Path(path), label="embedding config"))


def _sequence_set_digest(records: Sequence[FastaRecord]) -> str:
    identifiers = sorted({record.sequence_id for record in records})
    payload = b"" if not identifiers else ("\n".join(identifiers) + "\n").encode("ascii")
    return _sha256(payload)


def _read_strict_fasta(snapshot: Snapshot) -> tuple[FastaRecord, ...]:
    payload = snapshot.payload
    _require(payload and payload.endswith(b"\n"), "panel FASTA lacks a terminal LF")
    _require(b"\r" not in payload, "panel FASTA is not LF-only")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise PackagingError("panel FASTA is not ASCII") from error
    lines = text[:-1].split("\n")
    _require(len(lines) % 2 == 0, "panel FASTA is not composed of two-line records")
    records: list[FastaRecord] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for offset in range(0, len(lines), 2):
        header = lines[offset]
        sequence = lines[offset + 1]
        match = _SEQUENCE_HEADER_RE.fullmatch(header)
        _require(match is not None, f"invalid FASTA header at line {offset + 1}")
        assert match is not None
        sequence_id = match.group(1)
        _require(
            _MIN_SEQUENCE_LENGTH <= len(sequence) <= _MAX_SEQUENCE_LENGTH,
            f"invalid sequence length at line {offset + 2}",
        )
        _require(
            sequence == sequence.upper() and set(sequence) <= _AMINO_ACIDS,
            f"non-canonical sequence at line {offset + 2}",
        )
        _require(
            sequence_id == _sha256(sequence.encode("ascii")),
            f"sequence identifier mismatch at line {offset + 1}",
        )
        _require(
            sequence_id not in seen_ids and sequence not in seen_sequences,
            f"duplicate FASTA record at line {offset + 1}",
        )
        seen_ids.add(sequence_id)
        seen_sequences.add(sequence)
        records.append(FastaRecord(sequence_id=sequence_id, sequence=sequence))
    _require(
        [record.sequence_id for record in records]
        == sorted(record.sequence_id for record in records),
        "panel FASTA is not ordered by ascending sequence_id",
    )
    return tuple(records)


def _verify_handshake(twin_root: Path) -> dict[str, object]:
    receipt_root = twin_root / "node-receipts"
    _require(
        receipt_root.is_dir() and not receipt_root.is_symlink(),
        "panel node-receipts directory is missing",
    )
    _require(
        _file_inventory(receipt_root) == frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"}),
        "panel handshake inventory changed",
    )
    _require(not _directory_inventory(receipt_root), "panel handshake contains a directory")
    _verify_read_only_tree(receipt_root, label="panel handshake")
    job_id = twin_root.name
    _require(job_id.isdigit(), "panel twin root basename is not a Slurm job ID")
    receipts: dict[int, Snapshot] = {}
    node_names: dict[int, str] = {}
    for task in (0, 1):
        snapshot = _read_snapshot(
            receipt_root / f"{task}.receipt", label=f"panel task {task} receipt"
        )
        _require(
            snapshot.payload.endswith(b"\n") and b"\r" not in snapshot.payload,
            "bad receipt framing",
        )
        try:
            lines = snapshot.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise PackagingError("panel handshake receipt is not UTF-8") from error
        _require(
            len(lines) == 3
            and lines[0] == f"array_job_id={job_id}"
            and lines[1] == f"array_task_id={task}"
            and lines[2].startswith("node_name="),
            f"panel task {task} receipt is invalid",
        )
        node = lines[2].removeprefix("node_name=")
        _require(_SAFE_NODE_RE.fullmatch(node) is not None, "unsafe panel node name")
        receipts[task] = snapshot
        node_names[task] = node
    _require(node_names[0] != node_names[1], "panel twins did not run on distinct nodes")
    receipt_hashes = {str(task): receipts[task].sha256 for task in (0, 1)}
    acknowledgement_hashes: dict[str, str] = {}
    for task in (0, 1):
        acknowledgement = _read_snapshot(
            receipt_root / f"{task}.ack", label=f"panel task {task} acknowledgement"
        )
        expected = f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode(
            "ascii"
        )
        _require(acknowledgement.payload == expected, "panel sibling acknowledgement changed")
        acknowledgement_hashes[str(task)] = acknowledgement.sha256
    return {
        "distinct_nodes": True,
        "bidirectional_acknowledgement": True,
        "receipt_sha256": receipt_hashes,
        "acknowledgement_sha256": acknowledgement_hashes,
    }


def _verify_manifest_tree(
    root: Path,
    *,
    expected_inventory: frozenset[str],
    expected_top_sha256: str,
    label: str,
) -> tuple[Snapshot, dict[str, str], tuple[Snapshot, ...]]:
    _require(_file_inventory(root) == expected_inventory, f"{label} inventory changed")
    expected_directories = {
        parent.as_posix()
        for name in expected_inventory
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    _require(
        _directory_inventory(root) == expected_directories,
        f"{label} directory inventory changed",
    )
    top = _read_snapshot(root / "SHA256SUMS", label=f"{label} SHA256SUMS")
    _require(top.sha256 == expected_top_sha256, f"{label} top hash changed")
    entries = _parse_sha_manifest(top.payload, label=f"{label} SHA256SUMS")
    _require(set(entries) == set(expected_inventory) - {"SHA256SUMS"}, f"{label} coverage gap")
    snapshots = [top]
    for name, digest in entries.items():
        snapshot = _read_snapshot(root / name, label=f"{label} {name}")
        _require(snapshot.sha256 == digest, f"{label} hash mismatch for {name}")
        snapshots.append(snapshot)
    return top, entries, tuple(snapshots)


def _verify_panel_receipt(
    receipt: Mapping[str, object],
    *,
    contract: EmbeddingContract,
) -> None:
    _exact_keys(
        receipt,
        {
            "schema_version",
            "artifact",
            "status",
            "acceptance_scope",
            "checks",
            "git_commit",
            "config_sha256",
            "publication_top_manifest_sha256",
            "panel_top_manifest_sha256",
            "code_manifest_sha256",
            "frozen_input_manifest_sha256",
            "verifier_attestation",
            "input_sha256",
            "artifact_sha256",
            "panel",
            "declared_extraction_contract_not_execution_evidence",
            "accepted_gate1",
            "production_handshake",
            "accepted_gate1_production_handshake",
        },
        label="panel independent receipt",
    )
    _require(receipt["schema_version"] == 1, "bad panel receipt schema")
    _require(
        receipt["artifact"] == "gate1_union_esm2_exact_panel_fasta_v1_independent_verification",
        "wrong panel receipt artifact",
    )
    _require(
        receipt["status"] == "accepted_for_embedding_input_only",
        "panel is not independently accepted for embedding input",
    )
    _require(receipt["git_commit"] == contract.panel_git_commit, "panel Git commit changed")
    _require(
        receipt["publication_top_manifest_sha256"] == contract.panel_publication_top_sha256,
        "panel publication link changed",
    )
    _require(
        receipt["panel_top_manifest_sha256"] == contract.panel_top_sha256,
        "panel semantic link changed",
    )
    _require(
        receipt["config_sha256"] == _EXPECTED_PANEL_CONFIG_SHA256
        and receipt["code_manifest_sha256"] == _EXPECTED_PANEL_CODE_MANIFEST_SHA256
        and receipt["frozen_input_manifest_sha256"] == _EXPECTED_PANEL_FROZEN_MANIFEST_SHA256,
        "accepted panel config/code/frozen attestation changed",
    )
    checks = _mapping(receipt["checks"], label="panel receipt checks")
    _exact_keys(checks, _PANEL_RECEIPT_CHECKS, label="panel receipt checks")
    _require(all(value is True for value in checks.values()), "panel receipt has a failed check")
    scope = _mapping(receipt["acceptance_scope"], label="panel acceptance scope")
    _exact_keys(
        scope,
        {
            "accepted_use",
            "downstream_evidence_requirement",
            "embedding_extraction_executed",
            "embedding_input_eligible",
            "embeddings_verified",
            "model_performance_evidence",
            "model_predictions_verified",
        },
        label="panel acceptance scope",
    )
    _require(
        scope["embedding_input_eligible"] is True
        and scope["embedding_extraction_executed"] is False
        and scope["embeddings_verified"] is False
        and scope["model_performance_evidence"] is False
        and scope["model_predictions_verified"] is False,
        "panel receipt overstates its acceptance scope",
    )
    artifacts = _mapping(receipt["artifact_sha256"], label="panel artifact hashes")
    expected_artifact_links = {
        "SHA256SUMS": contract.panel_publication_top_sha256,
        "panel/SHA256SUMS": contract.panel_top_sha256,
        "panel/coverage_receipt.json": contract.panel_coverage_receipt_sha256,
        "panel/manifest.json": contract.panel_manifest_sha256,
        "panel/union_esm_sequences.fasta": contract.panel_fasta_sha256,
    }
    _require(artifacts == expected_artifact_links, "panel receipt artifact links changed")
    panel = _mapping(receipt["panel"], label="panel receipt census")
    _require(
        panel.get("unique_sequences") == contract.expected_sequences
        and panel.get("sequence_ids_sha256") == contract.expected_sequence_ids_sha256,
        "panel receipt census changed",
    )
    extraction = _mapping(
        receipt["declared_extraction_contract_not_execution_evidence"],
        label="declared extraction contract",
    )
    _require(
        extraction.get("status")
        == "declared_contract_hash_bound_but_extraction_not_executed_or_verified"
        and extraction.get("embedding_model") == contract.embedding_model
        and extraction.get("representation_layer") == contract.representation_layer
        and extraction.get("embedding_dimension") == contract.embedding_dimension
        and extraction.get("embedding_batch_size") == contract.embedding_batch_size
        and extraction.get("embedding_seed") == contract.embedding_seed
        and extraction.get("model_checkpoint_sha256") == contract.model_checkpoint_sha256
        and extraction.get("contact_regression_sha256") == contract.contact_regression_sha256
        and extraction.get("environment_lock_sha256") == contract.environment_lock_sha256
        and extraction.get("trust_manifest_sha256") == contract.trust_manifest_sha256
        and extraction.get("embedding_worker_sha256") == contract.embedding_worker_sha256,
        "panel receipt extraction contract changed",
    )


def _verify_panel_documents(
    *,
    manifest: Mapping[str, object],
    coverage: Mapping[str, object],
    receipt: Mapping[str, object],
    contract: EmbeddingContract,
) -> None:
    _require(manifest.get("schema_version") == 1, "bad panel manifest schema")
    _require(
        manifest.get("artifact") == "gate1_union_esm2_exact_panel_fasta_v1"
        and manifest.get("status") == "candidate_pending_independent_verification"
        and manifest.get("production_eligible") is False,
        "panel producer manifest semantics changed",
    )
    _require(
        manifest.get("git_commit") == contract.panel_git_commit, "panel manifest commit changed"
    )
    _require(
        manifest.get("config_sha256") == _EXPECTED_PANEL_CONFIG_SHA256,
        "panel manifest config link changed",
    )
    manifest_artifacts = _mapping(manifest.get("artifacts"), label="panel manifest artifacts")
    fasta = _mapping(manifest_artifacts.get("fasta"), label="panel manifest FASTA")
    coverage_link = _mapping(
        manifest_artifacts.get("coverage_receipt"), label="panel manifest coverage link"
    )
    _require(
        fasta
        == {
            "filename": "union_esm_sequences.fasta",
            "records": contract.expected_sequences,
            "sha256": contract.panel_fasta_sha256,
        }
        and coverage_link
        == {
            "filename": "coverage_receipt.json",
            "sha256": contract.panel_coverage_receipt_sha256,
        },
        "panel manifest artifact links changed",
    )
    manifest_panel = _mapping(manifest.get("panel"), label="panel manifest census")
    receipt_panel = _mapping(receipt["panel"], label="panel receipt census")
    _require(manifest_panel == receipt_panel, "panel census differs from independent receipt")
    _require(coverage.get("schema_version") == 1, "bad coverage receipt schema")
    _require(
        coverage.get("artifact") == "gate1_union_esm2_exact_panel_coverage_v1"
        and coverage.get("status") == "passed",
        "panel coverage receipt did not pass",
    )
    coverage_inputs = _mapping(coverage.get("input_sha256"), label="coverage input hashes")
    _require(
        coverage_inputs.get("config") == _EXPECTED_PANEL_CONFIG_SHA256,
        "coverage receipt config link changed",
    )
    coverage_panel = _mapping(coverage.get("panel"), label="coverage panel census")
    export = _mapping(coverage.get("export"), label="coverage export")
    _require(coverage_panel == receipt_panel, "coverage census differs from accepted receipt")
    _require(
        export.get("records") == contract.expected_sequences
        and export.get("fasta_sha256") == contract.panel_fasta_sha256
        and export.get("ordering") == "ascending sequence_id"
        and export.get("missing_sequences") == 0
        and export.get("extra_sequences") == 0
        and export.get("missing_sequence_ids") == []
        and export.get("extra_sequence_ids") == [],
        "panel coverage is not exact",
    )


def _validate_panel_evidence(
    *,
    twin_root: str | Path,
    independent_receipt: str | Path,
    contract: EmbeddingContract,
) -> PanelEvidence:
    root = _resolved_directory(twin_root, label="accepted panel twin root")
    _require(
        root.name == str(contract.panel_producer_job_id),
        "panel root does not match the configured producer job",
    )
    immediate = {path.name: path for path in root.iterdir()}
    _require(set(immediate) == {"0", "1", "node-receipts"}, "panel job-root inventory changed")
    _require(
        all(path.is_dir() and not path.is_symlink() for path in immediate.values()),
        "panel job-root entries are not real directories",
    )
    receipt_snapshot = _read_snapshot(Path(independent_receipt), label="panel independent receipt")
    _require(
        receipt_snapshot.path.stat().st_mode & 0o222 == 0,
        "panel independent receipt remains writable",
    )
    _require(
        receipt_snapshot.sha256 == contract.panel_independent_receipt_sha256,
        "panel independent receipt is not the configured content address",
    )
    receipt = _read_pretty_json(receipt_snapshot, label="panel independent receipt")
    _verify_panel_receipt(receipt, contract=contract)
    handshake = _verify_handshake(root)
    handshake_snapshots = tuple(
        _read_snapshot(root / "node-receipts" / name, label=f"panel handshake {name}")
        for name in ("0.ack", "0.receipt", "1.ack", "1.receipt")
    )
    _require(
        handshake == receipt["production_handshake"],
        "panel handshake differs from the independent receipt",
    )
    runs = (root / "0", root / "1")
    run_snapshots: list[tuple[Snapshot, ...]] = []
    run_top_entries: list[dict[str, str]] = []
    panel_top_entries: list[dict[str, str]] = []
    for index, run in enumerate(runs):
        _require(
            set(path.name for path in run.iterdir())
            == {"SHA256SUMS", "CODE_SHA256SUMS", "FROZEN_INPUT_SHA256SUMS", "panel"},
            f"panel twin {index} immediate inventory changed",
        )
        _require(
            (run / "panel").is_dir() and not (run / "panel").is_symlink(), "panel directory missing"
        )
        publication_top, publication_entries, publication_snapshots = _verify_manifest_tree(
            run,
            expected_inventory=_PANEL_PUBLICATION_FILES,
            expected_top_sha256=contract.panel_publication_top_sha256,
            label=f"panel publication twin {index}",
        )
        panel_top, semantic_entries, semantic_snapshots = _verify_manifest_tree(
            run / "panel",
            expected_inventory=_PANEL_SEMANTIC_FILES,
            expected_top_sha256=contract.panel_top_sha256,
            label=f"panel semantic twin {index}",
        )
        _require(
            publication_entries["panel/SHA256SUMS"] == panel_top.sha256,
            "publication does not bind the panel top manifest",
        )
        _verify_read_only_tree(run, label=f"accepted panel twin {index}")
        run_snapshots.append(
            tuple(
                {item.path: item for item in (*publication_snapshots, *semantic_snapshots)}.values()
            )
        )
        run_top_entries.append(publication_entries)
        panel_top_entries.append(semantic_entries)
        _require(
            publication_top.sha256 == contract.panel_publication_top_sha256, "panel top changed"
        )
    _require(run_top_entries[0] == run_top_entries[1], "panel publication manifests differ")
    _require(panel_top_entries[0] == panel_top_entries[1], "panel semantic manifests differ")
    for name in sorted(_PANEL_PUBLICATION_FILES):
        left = _read_snapshot(runs[0] / name, label=f"left panel twin {name}")
        right = _read_snapshot(runs[1] / name, label=f"right panel twin {name}")
        _require(left.payload == right.payload, f"accepted panel twins differ at {name}")
    artifacts = _mapping(receipt["artifact_sha256"], label="panel artifact hashes")
    _require(
        run_top_entries[0]["CODE_SHA256SUMS"] == receipt["code_manifest_sha256"]
        and run_top_entries[0]["FROZEN_INPUT_SHA256SUMS"]
        == receipt["frozen_input_manifest_sha256"],
        "panel code/frozen manifest link changed",
    )
    for logical, digest in artifacts.items():
        _require(
            _read_snapshot(runs[0] / logical, label=f"panel artifact {logical}").sha256 == digest,
            f"panel receipt hash mismatch for {logical}",
        )
    fasta_snapshot = _read_snapshot(
        runs[0] / "panel/union_esm_sequences.fasta", label="accepted panel FASTA"
    )
    manifest_snapshot = _read_snapshot(
        runs[0] / "panel/manifest.json", label="accepted panel manifest"
    )
    coverage_snapshot = _read_snapshot(
        runs[0] / "panel/coverage_receipt.json", label="accepted panel coverage receipt"
    )
    _require(fasta_snapshot.sha256 == contract.panel_fasta_sha256, "panel FASTA hash changed")
    _require(
        manifest_snapshot.sha256 == contract.panel_manifest_sha256, "panel manifest hash changed"
    )
    _require(
        coverage_snapshot.sha256 == contract.panel_coverage_receipt_sha256,
        "panel coverage receipt hash changed",
    )
    records = _read_strict_fasta(fasta_snapshot)
    _require(len(records) == contract.expected_sequences, "panel FASTA count changed")
    _require(
        _sequence_set_digest(records) == contract.expected_sequence_ids_sha256,
        "panel FASTA sequence-ID digest changed",
    )
    manifest = _read_pretty_json(manifest_snapshot, label="accepted panel manifest")
    coverage = _read_pretty_json(coverage_snapshot, label="accepted panel coverage receipt")
    _verify_panel_documents(
        manifest=manifest,
        coverage=coverage,
        receipt=receipt,
        contract=contract,
    )
    snapshots = tuple(
        {
            item.path: item
            for item in (
                receipt_snapshot,
                *handshake_snapshots,
                *run_snapshots[0],
                *run_snapshots[1],
            )
        }.values()
    )
    for snapshot in snapshots:
        _assert_unchanged(snapshot, label=f"panel evidence {snapshot.path.name}")
    return PanelEvidence(
        root=root,
        records=records,
        receipt=receipt,
        receipt_snapshot=receipt_snapshot,
        snapshots=snapshots,
        handshake=handshake,
    )


def _assert_panel_tree_stable(panel: PanelEvidence) -> None:
    _require(
        {path.name for path in panel.root.iterdir()} == {"0", "1", "node-receipts"},
        "accepted panel job-root inventory changed during packaging",
    )
    for index in (0, 1):
        run = panel.root / str(index)
        _require(
            _file_inventory(run) == _PANEL_PUBLICATION_FILES
            and _directory_inventory(run) == frozenset({"panel"}),
            f"accepted panel twin {index} inventory changed during packaging",
        )
        _verify_read_only_tree(run, label=f"accepted panel twin {index}")
    _require(
        _file_inventory(panel.root / "node-receipts")
        == frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"})
        and not _directory_inventory(panel.root / "node-receipts"),
        "accepted panel handshake inventory changed during packaging",
    )
    _require(_verify_handshake(panel.root) == panel.handshake, "accepted panel handshake changed")


def _worker_input_manifest(
    *,
    panel: PanelEvidence,
    contract: EmbeddingContract,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "purpose": "gate1_esm2_embedding_input",
        "status": "private_worker_compatibility_input_not_public_evidence",
        "artifact": {
            "filename": "union_esm_sequences.fasta",
            "sha256": contract.panel_fasta_sha256,
        },
        "records": contract.expected_sequences,
        "ordering": "ascending sequence_id",
        "sequence_ids_sha256": contract.expected_sequence_ids_sha256,
        "config": {"artifact": contract.artifact, "sha256": contract.sha256},
        "source_panel": {
            "acceptance_status": "accepted_for_embedding_input_only",
            "git_commit": contract.panel_git_commit,
            "publication_top_manifest_sha256": contract.panel_publication_top_sha256,
            "panel_top_manifest_sha256": contract.panel_top_sha256,
            "manifest_sha256": contract.panel_manifest_sha256,
            "coverage_receipt_sha256": contract.panel_coverage_receipt_sha256,
            "independent_receipt_sha256": panel.receipt_snapshot.sha256,
        },
        "evidence_scope": {
            "embedding_input_accepted": True,
            "embedding_extraction_executed": False,
            "embeddings_verified": False,
            "model_evidence": False,
            "prediction_evidence": False,
            "performance_evidence": False,
            "model_predictions_verified": False,
            "model_performance_evidence": False,
        },
    }
    _assert_path_free(value, label="private worker-input manifest")
    return value


def _ensure_output_parent(path: Path, *, label: str) -> Path:
    requested = Path(path)
    _require(requested.name not in {"", ".", ".."}, f"{label} has no basename")
    _reject_symlink_chain(requested.parent, label=f"{label} parent")
    requested.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(requested.parent, label=f"{label} parent")
    parent = requested.parent.resolve(strict=True)
    _require(not requested.parent.is_symlink(), f"{label} parent is a symbolic link")
    return parent / requested.name


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    target = _ensure_output_parent(path, label="output file")
    _require(not os.path.lexists(target), f"refusing to overwrite output: {target}")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, mode)
        os.link(staging, target)
        staging.unlink()
    finally:
        if staging.exists():
            staging.unlink()


def prepare_worker_input(
    *,
    panel_twin_root: str | Path,
    panel_independent_receipt: str | Path,
    config_path: str | Path,
    output_manifest: str | Path,
) -> dict[str, object]:
    """Validate the accepted panel and emit a private legacy worker manifest."""

    config_snapshot = _read_snapshot(Path(config_path), label="embedding config")
    contract = _parse_config(config_snapshot)
    panel = _validate_panel_evidence(
        twin_root=panel_twin_root,
        independent_receipt=panel_independent_receipt,
        contract=contract,
    )
    target = Path(output_manifest).resolve(strict=False)
    _require(
        target != panel.root and not target.is_relative_to(panel.root),
        "private manifest must be outside the accepted panel tree",
    )
    value = _worker_input_manifest(panel=panel, contract=contract)
    payload = _pretty_json_bytes(value)
    _scan_path_free_payloads([payload], ())
    for snapshot in (*panel.snapshots, config_snapshot):
        _assert_unchanged(snapshot, label=f"worker-input source {snapshot.path.name}")
    _assert_panel_tree_stable(panel)
    _write_bytes_exclusive(Path(output_manifest), payload, mode=0o444)
    _assert_unchanged(config_snapshot, label="embedding config")
    _assert_panel_tree_stable(panel)
    return value


def _bundle_file(bundle_root: Path, relative_text: str, *, label: str) -> Path:
    relative = _safe_relative_path(relative_text, label=label)
    current = bundle_root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), f"{label} traverses a symbolic link")
    try:
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise PackagingError(f"{label} is unavailable") from error
    _require(resolved.is_relative_to(bundle_root), f"{label} escapes the bundle root")
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is not a regular file")
    return resolved


def _validate_trust_evidence(
    *,
    trust_receipt: str | Path,
    trust_manifest: str | Path,
    bundle_root: str | Path,
    contract: EmbeddingContract,
) -> TrustEvidence:
    bundle = _resolved_directory(bundle_root, label="AMP-Diffusion bundle root")
    receipt_snapshot = _read_snapshot(Path(trust_receipt), label="bundle verification receipt")
    manifest_snapshot = _read_snapshot(Path(trust_manifest), label="AMP-Diffusion trust manifest")
    _require(
        receipt_snapshot.path.stat().st_mode & 0o222 == 0
        and manifest_snapshot.path.stat().st_mode & 0o222 == 0,
        "trust receipt/manifest must be immutable snapshots",
    )
    _require(
        receipt_snapshot.sha256 == contract.bundle_verification_receipt_sha256,
        "bundle verification receipt hash changed",
    )
    _require(
        manifest_snapshot.sha256 == contract.trust_manifest_sha256,
        "AMP-Diffusion trust manifest hash changed",
    )
    try:
        receipt_value = json.loads(
            receipt_snapshot.payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackagingError("bundle verification receipt is not valid JSON") from error
    receipt = _mapping(receipt_value, label="bundle verification receipt")
    canonical_receipt = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    _require(
        receipt_snapshot.payload == canonical_receipt,
        "bundle verification receipt is not canonical compact JSON",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_type",
            "component",
            "bundle_root",
            "integration",
            "source",
            "trust_manifest",
            "artifacts",
        },
        label="bundle verification receipt",
    )
    _require(
        receipt["schema_version"] == 1
        and receipt["receipt_type"] == "ampdiffusion_bundle_verification"
        and receipt["component"] == "generation"
        and receipt["bundle_root"] == str(bundle),
        "bundle verification receipt identity changed",
    )
    integration = _string(receipt["integration"], label="trust integration")
    source = _mapping(receipt["source"], label="trust source")
    _exact_keys(source, {"repository", "commit", "license"}, label="trust source")
    _require(
        isinstance(source["repository"], str)
        and cast(str, source["repository"]).startswith("https://")
        and source["commit"] == contract.ampdiffusion_source_commit
        and isinstance(source["license"], str)
        and source["license"] != "",
        "trust source identity changed",
    )
    manifest_link = _mapping(receipt["trust_manifest"], label="trust manifest link")
    _exact_keys(manifest_link, {"filename", "sha256", "size"}, label="trust manifest link")
    _require(
        manifest_link["filename"] == manifest_snapshot.path.name
        and manifest_link["sha256"] == manifest_snapshot.sha256
        and manifest_link["size"] == len(manifest_snapshot.payload),
        "trust receipt does not bind the supplied trust manifest",
    )
    try:
        trust_document = tomllib.loads(manifest_snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PackagingError("AMP-Diffusion trust manifest is not valid TOML") from error
    _exact_keys(
        trust_document,
        {
            "format_version",
            "integration",
            "source",
            "runtime",
            "competition_reference",
            "trusted_file",
        },
        label="AMP-Diffusion trust manifest",
    )
    _require(
        trust_document["format_version"] == 1 and trust_document["integration"] == integration,
        "trust manifest identity changed",
    )
    trust_source = _mapping(trust_document["source"], label="trust-manifest source")
    _require(trust_source == source, "trust receipt and manifest source identities differ")
    trusted_files = trust_document["trusted_file"]
    _require(isinstance(trusted_files, list) and trusted_files, "trust manifest has no files")
    generation_specs: dict[str, dict[str, object]] = {}
    for index, raw_spec in enumerate(cast(list[object], trusted_files)):
        spec = _mapping(raw_spec, label=f"trusted file {index}")
        _require(
            set(spec)
            in (
                {"path", "sha256", "size", "components", "role"},
                {"path", "sha256", "size", "components", "role", "url"},
            ),
            f"trusted file {index} schema changed",
        )
        relative_text = _string(spec["path"], label=f"trusted file {index} path")
        _safe_relative_path(relative_text, label=f"trusted file {index}")
        _sha_field(spec["sha256"], label=f"trusted file {index} digest")
        _positive_int(spec["size"], label=f"trusted file {index} size")
        components = spec["components"]
        _require(
            isinstance(components, list)
            and components
            and all(isinstance(item, str) and item for item in components),
            f"trusted file {index} components are invalid",
        )
        _string(spec["role"], label=f"trusted file {index} role")
        _require(relative_text not in generation_specs, "trust manifest repeats a path")
        if "generation" in components:
            generation_specs[relative_text] = spec
    artifacts = receipt["artifacts"]
    _require(isinstance(artifacts, list) and artifacts, "trust receipt has no artifacts")
    receipt_specs: dict[str, dict[str, object]] = {}
    artifact_snapshots: list[Snapshot] = []
    previous: str | None = None
    for index, raw_spec in enumerate(cast(list[object], artifacts)):
        spec = _mapping(raw_spec, label=f"receipt artifact {index}")
        _exact_keys(spec, {"path", "sha256", "size", "role"}, label=f"receipt artifact {index}")
        relative_text = _string(spec["path"], label=f"receipt artifact {index} path")
        _safe_relative_path(relative_text, label=f"receipt artifact {index}")
        if previous is not None:
            _require(relative_text > previous, "trust receipt artifact paths are not sorted")
        previous = relative_text
        _require(relative_text not in receipt_specs, "trust receipt repeats an artifact")
        digest = _sha_field(spec["sha256"], label=f"receipt artifact {index} digest")
        size = _positive_int(spec["size"], label=f"receipt artifact {index} size")
        role = _string(spec["role"], label=f"receipt artifact {index} role")
        trusted = generation_specs.get(relative_text)
        _require(trusted is not None, f"trust manifest omits receipt artifact {relative_text}")
        assert trusted is not None
        _require(
            trusted["sha256"] == digest and trusted["size"] == size and trusted["role"] == role,
            f"trust manifest metadata differs for {relative_text}",
        )
        artifact_path = _bundle_file(
            bundle, relative_text, label=f"trusted artifact {relative_text}"
        )
        snapshot = _read_snapshot(artifact_path, label=f"trusted artifact {relative_text}")
        _require(
            snapshot.sha256 == digest and len(snapshot.payload) == size,
            f"trusted artifact bytes changed: {relative_text}",
        )
        artifact_snapshots.append(snapshot)
        receipt_specs[relative_text] = spec
    _require(
        set(receipt_specs) == set(generation_specs),
        "bundle verification receipt does not cover every generation trust entry",
    )
    core_expectations = {
        _MODEL_RELATIVE_PATH: (contract.model_checkpoint_sha256, "pickle_checkpoint"),
        _CONTACT_RELATIVE_PATH: (contract.contact_regression_sha256, "pickle_checkpoint"),
        _LOCK_RELATIVE_PATH: (contract.environment_lock_sha256, "environment_lock"),
    }
    core: dict[str, dict[str, object]] = {}
    role_names = {
        _MODEL_RELATIVE_PATH: "model_checkpoint",
        _CONTACT_RELATIVE_PATH: "contact_regression_checkpoint",
        _LOCK_RELATIVE_PATH: "environment_lock",
    }
    for relative_text, (digest, role) in core_expectations.items():
        spec = receipt_specs.get(relative_text)
        _require(spec is not None, f"trust evidence omits {relative_text}")
        assert spec is not None
        _require(
            spec["sha256"] == digest and spec["role"] == role,
            f"trust evidence pin changed for {relative_text}",
        )
        core[role_names[relative_text]] = {
            "path": relative_text,
            "sha256": spec["sha256"],
            "size": spec["size"],
        }
    raw_worker_trust: dict[str, object] = {
        "integration": integration,
        "source_commit": contract.ampdiffusion_source_commit,
        "bundle_root": str(bundle),
        "trust_manifest_sha256": manifest_snapshot.sha256,
        "verification_receipt_sha256": receipt_snapshot.sha256,
        "artifacts": core,
    }
    semantic_trust: dict[str, object] = {
        "integration": integration,
        "source": {
            "repository": source["repository"],
            "commit": source["commit"],
            "license": source["license"],
        },
        "trust_manifest_sha256": manifest_snapshot.sha256,
        "bundle_verification_receipt_sha256": receipt_snapshot.sha256,
        "artifacts": core,
    }
    _assert_path_free(semantic_trust, label="semantic trust evidence")
    all_snapshots = (receipt_snapshot, manifest_snapshot, *artifact_snapshots)
    for snapshot in all_snapshots:
        _assert_unchanged(snapshot, label=f"trust evidence {snapshot.path.name}")
    return TrustEvidence(
        receipt_snapshot=receipt_snapshot,
        manifest_snapshot=manifest_snapshot,
        snapshots=tuple(all_snapshots),
        raw_worker_trust=raw_worker_trust,
        semantic_trust=semantic_trust,
    )


def _canonical_index_bytes(records: Sequence[FastaRecord]) -> bytes:
    lines = ["row_index,sequence_id,sequence,length\n"]
    for index, record in enumerate(records):
        lines.append(f"{index},{record.sequence_id},{record.sequence},{len(record.sequence)}\n")
    return "".join(lines).encode("ascii")


def _canonical_npy_bytes(tensor_payload: bytes, *, rows: int, columns: int) -> bytes:
    _require(len(tensor_payload) == rows * columns * 4, "tensor payload has the wrong size")
    dictionary = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}"
    ).encode("latin1")
    prefix_length = len(b"\x93NUMPY") + 2 + 2
    padding = (-((prefix_length + len(dictionary) + 1) % 64)) % 64
    header = dictionary + (b" " * padding) + b"\n"
    _require(len(header) < 2**16, "NPY v1 header is too large")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + tensor_payload


def _validate_matrix_snapshot(
    snapshot: Snapshot,
    *,
    expected_rows: int,
    expected_columns: int,
) -> MatrixEvidence:
    payload = snapshot.payload
    _require(payload.startswith(b"\x93NUMPY\x01\x00"), "embedding matrix is not NPY v1.0")
    _require(len(payload) >= 10, "embedding NPY header is truncated")
    header_length = struct.unpack("<H", payload[8:10])[0]
    header_end = 10 + header_length
    _require(header_end <= len(payload), "embedding NPY header length is invalid")
    header = payload[10:header_end]
    _require(header.endswith(b"\n") and b"\r" not in header, "embedding NPY header framing changed")
    try:
        parsed = ast.literal_eval(header.decode("latin1").strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise PackagingError("embedding NPY header is invalid") from error
    header_document = _mapping(parsed, label="embedding NPY header")
    _exact_keys(
        header_document,
        {"descr", "fortran_order", "shape"},
        label="embedding NPY header",
    )
    _require(header_document["descr"] == "<f4", "embedding matrix is not little-endian float32")
    _require(header_document["fortran_order"] is False, "embedding matrix is not C-contiguous")
    _require(
        header_document["shape"] == (expected_rows, expected_columns),
        "embedding matrix shape changed",
    )
    tensor = payload[header_end:]
    _require(
        len(tensor) == expected_rows * expected_columns * 4,
        "embedding matrix has a non-canonical tensor byte count",
    )
    _require(
        payload == _canonical_npy_bytes(tensor, rows=expected_rows, columns=expected_columns),
        "embedding matrix is not the canonical NPY v1.0 encoding",
    )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PackagingError("embedding matrix cannot be loaded safely") from error
    _require(isinstance(loaded, np.ndarray), "embedding matrix did not load as an array")
    _require(
        loaded.shape == (expected_rows, expected_columns)
        and loaded.dtype.str == "<f4"
        and loaded.flags.c_contiguous,
        "loaded embedding matrix contract changed",
    )
    values = np.frombuffer(tensor, dtype=np.dtype("<f4"))
    _require(values.size == expected_rows * expected_columns, "embedding tensor size changed")
    _require(bool(np.isfinite(values).all()), "embedding matrix contains a non-finite value")
    return MatrixEvidence(
        snapshot=snapshot,
        tensor_payload=tensor,
        tensor_data_sha256=_sha256(tensor),
        rows=expected_rows,
        columns=expected_columns,
    )


def _git(repository: Path, arguments: Sequence[str], *, label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise PackagingError(f"Git check failed for {label}") from error


def _committed_blob(repository: Path, commit: str, logical_path: str) -> bytes:
    kind = _git(repository, ["cat-file", "-t", f"{commit}:{logical_path}"], label=logical_path)
    _require(kind == b"blob\n", f"committed path is not a blob: {logical_path}")
    return _git(
        repository,
        ["cat-file", "blob", f"{commit}:{logical_path}"],
        label=logical_path,
    )


def _verify_repository(repository: Path, expected_commit: str) -> None:
    _require(_GIT_RE.fullmatch(expected_commit) is not None, "expected Git commit is invalid")
    head = _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"], label="HEAD")
    _require(head.decode().strip() == expected_commit, "repository HEAD differs from expected")
    _git(repository, ["diff", "--no-ext-diff", "--quiet", "--exit-code", "--"], label="tree")
    _git(
        repository,
        ["diff", "--cached", "--no-ext-diff", "--quiet", "--exit-code", "--"],
        label="index",
    )
    _require(
        _git(repository, ["ls-files", "--others", "--exclude-standard", "-z"], label="untracked")
        == b"",
        "repository contains untracked files",
    )
    origin = _git(
        repository,
        ["rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
        label="origin/main",
    )
    _require(origin.decode().strip() == expected_commit, "expected commit is not on origin/main")


def _validate_code_attestation(
    *,
    repository: Path,
    expected_commit: str,
    code_manifest: str | Path,
    contract: EmbeddingContract,
) -> tuple[dict[str, str], tuple[Snapshot, ...]]:
    _verify_repository(repository, expected_commit)
    executing = Path(__file__).resolve(strict=True)
    expected_executing = (repository / _PACKAGE_MODULE_PATH).resolve(strict=True)
    _require(executing == expected_executing, "executing packager is a stale installation")
    manifest_snapshot = _read_snapshot(Path(code_manifest), label="embedding code manifest")
    _require(
        manifest_snapshot.path.stat().st_mode & 0o222 == 0,
        "embedding code manifest remains writable",
    )
    entries = _parse_sha_manifest(manifest_snapshot.payload, label="embedding code manifest")
    source_root = repository / "src/amp_challenge"
    _require(
        source_root.is_dir() and not source_root.is_symlink(),
        "repository Python source tree is unavailable",
    )
    source_paths: set[str] = set()
    for source in source_root.rglob("*.py"):
        _require(not source.is_symlink(), "repository Python source tree contains a symlink")
        if source.is_file():
            source_paths.add(source.relative_to(repository).as_posix())
    expected_paths = set(_REQUIRED_CODE_PATHS) | source_paths
    _require(
        set(entries) == expected_paths,
        "embedding code-manifest inventory differs from the complete protocol source set",
    )
    snapshots = [manifest_snapshot]
    for logical_path, expected_sha256 in entries.items():
        relative = _safe_relative_path(logical_path, label="code-manifest entry")
        requested = repository.joinpath(*relative.parts)
        snapshot = _read_snapshot(requested, label=f"code entry {logical_path}")
        _require(snapshot.sha256 == expected_sha256, f"code hash mismatch: {logical_path}")
        _require(
            snapshot.payload == _committed_blob(repository, expected_commit, logical_path),
            f"code entry differs from committed blob: {logical_path}",
        )
        snapshots.append(snapshot)
    _require(
        entries[_PACKAGE_MODULE_PATH]
        == _sha256(_committed_blob(repository, expected_commit, _PACKAGE_MODULE_PATH)),
        "code manifest does not bind this package module",
    )
    _require(
        entries[_VERIFIER_MODULE_PATH]
        == _sha256(_committed_blob(repository, expected_commit, _VERIFIER_MODULE_PATH)),
        "code manifest does not bind the independent embedding verifier",
    )
    _require(
        entries[_CONFIG_LOGICAL_PATH] == contract.sha256,
        "code manifest does not bind the supplied embedding config",
    )
    _require(
        entries[_WORKER_LOGICAL_PATH] == contract.embedding_worker_sha256,
        "code manifest does not bind the accepted legacy worker",
    )
    _require(
        entries[_TRUST_MANIFEST_LOGICAL_PATH] == contract.trust_manifest_sha256,
        "code manifest does not bind the AMP-Diffusion trust manifest",
    )
    return entries, tuple(snapshots)


def _validate_worker_input_manifest(
    snapshot: Snapshot,
    *,
    panel: PanelEvidence,
    contract: EmbeddingContract,
) -> dict[str, object]:
    _require(
        snapshot.path.stat().st_mode & 0o222 == 0,
        "private worker-input manifest remains writable",
    )
    observed = _read_pretty_json(snapshot, label="private worker-input manifest")
    expected = _worker_input_manifest(panel=panel, contract=contract)
    _require(
        observed == expected, "private worker-input manifest was not independently reconstructed"
    )
    _require(snapshot.payload == _pretty_json_bytes(expected), "private worker-input bytes changed")
    return observed


def _validate_raw_worker_manifest(
    manifest: Mapping[str, object],
    *,
    manifest_snapshot: Snapshot,
    matrix: MatrixEvidence,
    index_snapshot: Snapshot,
    worker_input_snapshot: Snapshot,
    trust: TrustEvidence,
    contract: EmbeddingContract,
    worker_snapshot: Snapshot,
) -> None:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "benchmark",
            "model",
            "representation_layer",
            "embedding_dimension",
            "pooling",
            "dtype",
            "records",
            "ordering",
            "input_fasta_sha256",
            "input_fasta_manifest_sha256",
            "trust",
            "determinism",
            "runtime",
            "outputs",
            "worker_sha256",
        },
        label="raw worker manifest",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["benchmark"] == "esm2_embedding_extraction"
        and manifest["model"] == contract.embedding_model
        and manifest["representation_layer"] == contract.representation_layer
        and manifest["embedding_dimension"] == contract.embedding_dimension
        and manifest["pooling"] == contract.pooling
        and manifest["dtype"] == contract.embedding_dtype
        and manifest["records"] == contract.expected_sequences
        and manifest["ordering"] == "input FASTA order (ascending sequence_id)",
        "raw worker model/tensor contract changed",
    )
    _require(
        manifest["input_fasta_sha256"] == contract.panel_fasta_sha256
        and manifest["input_fasta_manifest_sha256"] == worker_input_snapshot.sha256,
        "raw worker input provenance changed",
    )
    raw_runtime = _mapping(manifest["runtime"], label="raw worker runtime")
    _exact_keys(raw_runtime, _RUNTIME_FIELDS, label="raw worker runtime")
    _require(raw_runtime == dict(contract.runtime), "raw worker runtime differs from config")
    raw_determinism = _mapping(manifest["determinism"], label="raw worker determinism")
    expected_determinism = {
        "seed": contract.embedding_seed,
        "batch_size": contract.embedding_batch_size,
        "torch_deterministic_algorithms": contract.determinism["torch_deterministic_algorithms"],
        "cudnn_benchmark": contract.determinism["cudnn_benchmark"],
        "cudnn_deterministic": contract.determinism["cudnn_deterministic"],
        "tf32": contract.determinism["tf32"],
        "cublas_workspace_config": contract.determinism["cublas_workspace_config"],
    }
    _require(raw_determinism == expected_determinism, "raw worker determinism changed")
    _require(manifest["trust"] == trust.raw_worker_trust, "raw worker trust provenance changed")
    outputs = _mapping(manifest["outputs"], label="raw worker outputs")
    _require(
        outputs
        == {
            "embeddings.npy": matrix.snapshot.sha256,
            "embedding_index.csv": index_snapshot.sha256,
        },
        "raw worker output hashes changed",
    )
    _require(
        manifest["worker_sha256"] == contract.embedding_worker_sha256
        and manifest["worker_sha256"] == worker_snapshot.sha256,
        "raw worker source hash changed",
    )
    _require(
        manifest_snapshot.sha256 == _sha256(manifest_snapshot.payload), "raw manifest hash failed"
    )


def _validate_raw_embeddings(
    *,
    raw_embedding_root: str | Path,
    records: Sequence[FastaRecord],
    worker_input_snapshot: Snapshot,
    trust: TrustEvidence,
    contract: EmbeddingContract,
    worker_snapshot: Snapshot,
) -> tuple[MatrixEvidence, Snapshot, Snapshot, tuple[Snapshot, ...]]:
    root = _resolved_directory(raw_embedding_root, label="raw embedding root")
    _require(_file_inventory(root) == _RAW_EMBEDDING_FILES, "raw embedding inventory changed")
    _require(not _directory_inventory(root), "raw embedding root contains an unexpected directory")
    _require(
        set(path.name for path in root.iterdir()) == set(_RAW_EMBEDDING_FILES),
        "raw embedding root contains an unexpected directory",
    )
    _verify_read_only_tree(root, label="raw embedding output")
    matrix_snapshot = _read_snapshot(root / "embeddings.npy", label="raw embedding matrix")
    index_snapshot = _read_snapshot(root / "embedding_index.csv", label="raw embedding index")
    manifest_snapshot = _read_snapshot(
        root / "embedding_manifest.json", label="raw worker manifest"
    )
    _require(
        len(matrix_snapshot.payload) == _EXPECTED_NPY_BYTES,
        "raw embedding NPY file size changed",
    )
    matrix = _validate_matrix_snapshot(
        matrix_snapshot,
        expected_rows=contract.expected_sequences,
        expected_columns=contract.embedding_dimension,
    )
    _require(
        len(matrix.tensor_payload) == _EXPECTED_TENSOR_BYTES,
        "raw embedding tensor payload size changed",
    )
    expected_index = _canonical_index_bytes(records)
    _require(index_snapshot.payload == expected_index, "raw embedding index differs from the panel")
    manifest = _read_pretty_json(manifest_snapshot, label="raw worker manifest")
    _validate_raw_worker_manifest(
        manifest,
        manifest_snapshot=manifest_snapshot,
        matrix=matrix,
        index_snapshot=index_snapshot,
        worker_input_snapshot=worker_input_snapshot,
        trust=trust,
        contract=contract,
        worker_snapshot=worker_snapshot,
    )
    snapshots = (matrix_snapshot, index_snapshot, manifest_snapshot)
    for snapshot in snapshots:
        _assert_unchanged(snapshot, label=f"raw embedding {snapshot.path.name}")
    return matrix, index_snapshot, manifest_snapshot, snapshots


def _assert_raw_tree_stable(root: Path) -> None:
    _require(
        _file_inventory(root) == _RAW_EMBEDDING_FILES and not _directory_inventory(root),
        "raw embedding inventory changed during packaging",
    )
    _verify_read_only_tree(root, label="raw embedding output")


def _validate_driver_version(value: str) -> str:
    _require(
        isinstance(value, str)
        and value == value.strip()
        and _DRIVER_VERSION_RE.fullmatch(value) is not None,
        "CUDA driver version must be a dotted numeric version",
    )
    return value


def _semantic_manifest(
    *,
    contract: EmbeddingContract,
    panel: PanelEvidence,
    trust: TrustEvidence,
    matrix: MatrixEvidence,
    index_snapshot: Snapshot,
    raw_manifest_snapshot: Snapshot,
    worker_input_snapshot: Snapshot,
    code_entries: Mapping[str, str],
    code_manifest_snapshot: Snapshot,
    expected_git_commit: str,
    cuda_driver_version: str,
) -> dict[str, object]:
    runtime = dict(contract.runtime)
    runtime["cuda_driver_version"] = cuda_driver_version
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact": contract.artifact,
        "status": contract.acceptance["producer_status"],
        "production_eligible": False,
        "production_ineligibility_reason": (
            "independent_full_gpu_reextraction_and_exact_numerical_comparison_pending"
        ),
        "acceptance_scope": {
            "candidate_embedding_tensor_present": True,
            "embeddings_verified": False,
            "model_evidence": False,
            "prediction_evidence": False,
            "performance_evidence": False,
            "model_predictions_verified": False,
            "model_performance_evidence": False,
            "ensemble_weight_established": False,
            "pretraining_membership_independence_established": False,
            "accepted_use": "independent_numerical_verification_input_only",
        },
        "tensor": {
            "shape": [contract.expected_sequences, contract.embedding_dimension],
            "dtype": "float32",
            "byte_order": "little_endian",
            "layout": "C_contiguous",
            "container": "NPY",
            "npy_version": [1, 0],
            "tensor_data_bytes": len(matrix.tensor_payload),
            "tensor_data_sha256": matrix.tensor_data_sha256,
        },
        "extraction": {
            "model": contract.embedding_model,
            "representation_layer": contract.representation_layer,
            "pooling": contract.pooling,
            "records": contract.expected_sequences,
            "ordering": "accepted panel ascending sequence_id",
        },
        "runtime": runtime,
        "determinism": {
            "seed": contract.embedding_seed,
            "batch_size": contract.embedding_batch_size,
            **dict(contract.determinism),
        },
        "panel_provenance": {
            "git_commit": contract.panel_git_commit,
            "publication_top_manifest_sha256": contract.panel_publication_top_sha256,
            "panel_top_manifest_sha256": contract.panel_top_sha256,
            "fasta_sha256": contract.panel_fasta_sha256,
            "manifest_sha256": contract.panel_manifest_sha256,
            "coverage_receipt_sha256": contract.panel_coverage_receipt_sha256,
            "independent_receipt_sha256": panel.receipt_snapshot.sha256,
            "sequence_ids_sha256": contract.expected_sequence_ids_sha256,
            "records": contract.expected_sequences,
            "acceptance_status": "accepted_for_embedding_input_only",
        },
        "trust": dict(trust.semantic_trust),
        "input_attestation": {
            "config_sha256": contract.sha256,
            "code_manifest_sha256": code_manifest_snapshot.sha256,
            "private_worker_input_manifest_sha256": worker_input_snapshot.sha256,
            "raw_worker_output_manifest_sha256": raw_manifest_snapshot.sha256,
        },
        "code_attestation": {
            "git_commit": expected_git_commit,
            "package_module": {
                "logical_path": _PACKAGE_MODULE_PATH,
                "sha256": code_entries[_PACKAGE_MODULE_PATH],
            },
            "independent_verifier": {
                "logical_path": _VERIFIER_MODULE_PATH,
                "sha256": code_entries[_VERIFIER_MODULE_PATH],
                "included_in_producer_code_manifest": True,
            },
            "legacy_worker": {
                "logical_path": _WORKER_LOGICAL_PATH,
                "sha256": code_entries[_WORKER_LOGICAL_PATH],
                "preserved_byte_for_byte": True,
            },
        },
        "artifacts": {
            "embeddings.npy": {
                "sha256": matrix.snapshot.sha256,
                "size": len(matrix.snapshot.payload),
                "tensor_data_sha256": matrix.tensor_data_sha256,
            },
            "embedding_index.csv": {
                "sha256": index_snapshot.sha256,
                "size": len(index_snapshot.payload),
                "records": contract.expected_sequences,
            },
        },
        "independent_verification": {
            "status": "required_not_yet_performed",
            "full_reextraction_required": True,
            "exact_tensor_data_sha256_match_required": True,
            "exact_twin_bytes_required": True,
            "cuda_driver_exact_match_required": True,
            "verified_status_after_acceptance": contract.acceptance["verified_status"],
        },
    }
    _assert_path_free(value, label="semantic embedding manifest")
    return value


def _assert_snapshots_unchanged(snapshots: Iterable[Snapshot], *, label: str) -> None:
    seen: set[Path] = set()
    for snapshot in snapshots:
        if snapshot.path in seen:
            continue
        seen.add(snapshot.path)
        _assert_unchanged(snapshot, label=f"{label} {snapshot.path.name}")


def _flat_directory_identity(
    root: Path,
) -> tuple[
    tuple[int, int, int],
    tuple[tuple[str, tuple[int, int, int, int, int, int], str], ...],
]:
    """Fingerprint the exact flat publication tree, including its inodes."""

    try:
        root_stat = root.stat(follow_symlinks=False)
    except OSError as error:
        raise PackagingError("publication staging directory is unavailable") from error
    _require(
        stat.S_ISDIR(root_stat.st_mode) and not root.is_symlink(),
        "publication staging path is not a directory",
    )
    files: list[tuple[str, tuple[int, int, int, int, int, int], str]] = []
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise PackagingError("publication staging directory cannot be inspected") from error
    for child in children:
        snapshot = _read_snapshot(child, label="publication staging file")
        files.append((child.name, snapshot.fingerprint, snapshot.sha256))
    return (
        (root_stat.st_dev, root_stat.st_ino, stat.S_IMODE(root_stat.st_mode)),
        tuple(files),
    )


def _renameat2_directory_noreplace(source: Path, target: Path) -> int:
    """Return zero on success or the Linux ``renameat2`` errno."""

    if sys.platform != "linux":
        return errno.ENOSYS
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        return errno.ENOSYS
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    return 0 if result == 0 else (ctypes.get_errno() or errno.EIO)


def _gnu_mv_directory_noreplace(
    source: Path,
    target: Path,
    *,
    expected_identity: tuple[
        tuple[int, int, int],
        tuple[tuple[str, tuple[int, int, int, int, int, int], str], ...],
    ],
) -> None:
    """Use GNU mv's no-clobber mode and prove whether the move occurred."""

    try:
        completed = subprocess.run(
            [
                "/usr/bin/mv",
                "--no-clobber",
                "-T",
                "--",
                os.fspath(source),
                os.fspath(target),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env={"LC_ALL": "C"},
        )
    except OSError as error:
        raise PackagingError("GNU mv no-replace directory publication is unavailable") from error

    if os.path.lexists(source):
        if os.path.lexists(target):
            raise PackagingError(f"refusing to overwrite output directory: {target}")
        if completed.returncode == 0:
            raise PackagingError("GNU mv no-replace directory publication made no progress")
        raise PackagingError("GNU mv no-replace directory publication failed")
    _require(
        os.path.lexists(target),
        "GNU mv no-replace directory publication lost the staging directory",
    )
    _require(
        _flat_directory_identity(target) == expected_identity,
        "GNU mv no-replace directory publication changed the staged tree identity",
    )


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Publish *source* without accepting a replacement or ambiguous move."""

    expected_identity = _flat_directory_identity(source)
    error_number = _renameat2_directory_noreplace(source, target)
    if error_number == 0:
        _require(
            not os.path.lexists(source)
            and os.path.lexists(target)
            and _flat_directory_identity(target) == expected_identity,
            "atomic no-replace directory publication postcondition failed",
        )
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PackagingError(f"refusing to overwrite output directory: {target}")
    unavailable = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}
    if error_number in unavailable:
        _gnu_mv_directory_noreplace(
            source,
            target,
            expected_identity=expected_identity,
        )
        return
    raise PackagingError(
        f"atomic no-replace directory publication failed: {os.strerror(error_number)}"
    )


def _publish_directory(
    *,
    output_dir: str | Path,
    matrix_payload: bytes,
    index_payload: bytes,
    manifest_payload: bytes,
    before_publish: Callable[[], None],
) -> Path:
    requested = Path(output_dir)
    target = _ensure_output_parent(requested, label="embedding output directory")
    _require(not os.path.lexists(target), f"refusing to overwrite output directory: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    published = False
    try:
        payloads = {
            "embeddings.npy": matrix_payload,
            "embedding_index.csv": index_payload,
            "embedding_manifest.json": manifest_payload,
        }
        for filename, payload in payloads.items():
            path = staging / filename
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        checksums = _sha_manifest_bytes(
            {filename: _sha256(payload) for filename, payload in payloads.items()}
        )
        checksum_path = staging / "SHA256SUMS"
        with checksum_path.open("xb") as handle:
            handle.write(checksums)
            handle.flush()
            os.fsync(handle.fileno())
        _require(_file_inventory(staging) == _PUBLISHED_FILES, "staging inventory changed")
        _require(not _directory_inventory(staging), "staging contains an unexpected directory")
        before_publish()
        for path in staging.iterdir():
            os.chmod(path, 0o444)
        os.chmod(staging, 0o555)
        _require(not os.path.lexists(target), f"refusing to overwrite output directory: {target}")
        _rename_directory_noreplace(staging, target)
        published = True
        return target
    finally:
        if not published and staging.exists():
            with suppress(OSError):
                os.chmod(staging, 0o700)
                for path in staging.iterdir():
                    if not path.is_symlink():
                        os.chmod(path, 0o600)
                shutil.rmtree(staging)


def package_union_esm_embeddings(
    *,
    raw_embedding_root: str | Path,
    worker_input_manifest: str | Path,
    panel_twin_root: str | Path,
    panel_independent_receipt: str | Path,
    trust_receipt: str | Path,
    trust_manifest: str | Path,
    bundle_root: str | Path,
    config_path: str | Path,
    code_manifest: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    cuda_driver_version: str,
    output_dir: str | Path,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, object]:
    """Validate raw extraction evidence and publish a path-free candidate tree."""

    _require(sys.byteorder == "little", "packaging requires a little-endian host")
    _require(
        Path(output_dir).name == "embeddings", "semantic output directory must be named embeddings"
    )
    driver = _validate_driver_version(cuda_driver_version)
    repository = _resolved_directory(repo_root, label="repository root")
    requested_config = _resolved_regular_file(config_path, label="embedding config")
    _require(
        requested_config == (repository / _CONFIG_LOGICAL_PATH).resolve(strict=True),
        "packager received the wrong logical config",
    )
    requested_trust_manifest = _resolved_regular_file(
        trust_manifest,
        label="AMP-Diffusion trust manifest",
    )
    config_snapshot = _read_snapshot(requested_config, label="embedding config")
    contract = _parse_config(config_snapshot)
    code_entries, code_snapshots = _validate_code_attestation(
        repository=repository,
        expected_commit=expected_git_commit,
        code_manifest=code_manifest,
        contract=contract,
    )
    code_manifest_snapshot = code_snapshots[0]
    worker_snapshot = next(
        snapshot
        for snapshot in code_snapshots
        if snapshot.path == (repository / _WORKER_LOGICAL_PATH).resolve(strict=True)
    )
    panel = _validate_panel_evidence(
        twin_root=panel_twin_root,
        independent_receipt=panel_independent_receipt,
        contract=contract,
    )
    worker_input_snapshot = _read_snapshot(
        Path(worker_input_manifest), label="private worker-input manifest"
    )
    _validate_worker_input_manifest(worker_input_snapshot, panel=panel, contract=contract)
    trust = _validate_trust_evidence(
        trust_receipt=trust_receipt,
        trust_manifest=requested_trust_manifest,
        bundle_root=bundle_root,
        contract=contract,
    )
    matrix, index_snapshot, raw_manifest_snapshot, raw_snapshots = _validate_raw_embeddings(
        raw_embedding_root=raw_embedding_root,
        records=panel.records,
        worker_input_snapshot=worker_input_snapshot,
        trust=trust,
        contract=contract,
        worker_snapshot=worker_snapshot,
    )
    manifest = _semantic_manifest(
        contract=contract,
        panel=panel,
        trust=trust,
        matrix=matrix,
        index_snapshot=index_snapshot,
        raw_manifest_snapshot=raw_manifest_snapshot,
        worker_input_snapshot=worker_input_snapshot,
        code_entries=code_entries,
        code_manifest_snapshot=code_manifest_snapshot,
        expected_git_commit=expected_git_commit,
        cuda_driver_version=driver,
    )
    manifest_payload = _pretty_json_bytes(manifest)
    _scan_path_free_payloads(
        [index_snapshot.payload, manifest_payload],
        forbidden_prefixes,
    )
    raw_root = _resolved_directory(raw_embedding_root, label="raw embedding root")
    protected_roots = (
        panel.root,
        raw_root,
        repository,
        _resolved_directory(bundle_root, label="AMP-Diffusion bundle root"),
    )
    prospective = Path(output_dir).resolve(strict=False)
    for protected in protected_roots:
        _require(
            prospective != protected and not prospective.is_relative_to(protected),
            "embedding output must be outside every validated input tree",
        )
    source_snapshots = (
        config_snapshot,
        worker_input_snapshot,
        *panel.snapshots,
        *trust.snapshots,
        *code_snapshots,
        *raw_snapshots,
    )

    def before_publish() -> None:
        _assert_snapshots_unchanged(source_snapshots, label="pre-publication input")
        _assert_panel_tree_stable(panel)
        _assert_raw_tree_stable(raw_root)
        _verify_repository(repository, expected_git_commit)

    published = _publish_directory(
        output_dir=output_dir,
        matrix_payload=matrix.snapshot.payload,
        index_payload=index_snapshot.payload,
        manifest_payload=manifest_payload,
        before_publish=before_publish,
    )
    _require(_file_inventory(published) == _PUBLISHED_FILES, "published inventory changed")
    _require(not _directory_inventory(published), "published output contains a directory")
    _verify_read_only_tree(published, label="published embedding candidate")
    published_top = _read_snapshot(published / "SHA256SUMS", label="published SHA256SUMS")
    published_entries = _parse_sha_manifest(published_top.payload, label="published SHA256SUMS")
    _require(
        published_entries
        == {
            "embedding_index.csv": index_snapshot.sha256,
            "embedding_manifest.json": _sha256(manifest_payload),
            "embeddings.npy": matrix.snapshot.sha256,
        },
        "published SHA256SUMS changed",
    )
    _assert_snapshots_unchanged(source_snapshots, label="post-publication input")
    _assert_panel_tree_stable(panel)
    _assert_raw_tree_stable(raw_root)
    _verify_repository(repository, expected_git_commit)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-worker-input",
        help="validate the accepted union panel and write a private legacy manifest",
    )
    prepare.add_argument("--panel-twin-root", type=Path, required=True)
    prepare.add_argument("--panel-independent-receipt", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-manifest", type=Path, required=True)

    package = commands.add_parser(
        "package",
        help="validate raw worker evidence and publish canonical candidate artifacts",
    )
    package.add_argument("--raw-embedding-root", type=Path, required=True)
    package.add_argument("--worker-input-manifest", type=Path, required=True)
    package.add_argument("--panel-twin-root", type=Path, required=True)
    package.add_argument("--panel-independent-receipt", type=Path, required=True)
    package.add_argument("--trust-receipt", type=Path, required=True)
    package.add_argument("--trust-manifest", type=Path, required=True)
    package.add_argument("--bundle-root", type=Path, required=True)
    package.add_argument("--config", type=Path, required=True)
    package.add_argument("--code-manifest", type=Path, required=True)
    package.add_argument("--repo-root", type=Path, required=True)
    package.add_argument("--expected-git-commit", required=True)
    package.add_argument("--cuda-driver-version", required=True)
    package.add_argument("--forbidden-prefix", action="append", default=[])
    package.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-worker-input":
        result = prepare_worker_input(
            panel_twin_root=args.panel_twin_root,
            panel_independent_receipt=args.panel_independent_receipt,
            config_path=args.config,
            output_manifest=args.output_manifest,
        )
    else:
        result = package_union_esm_embeddings(
            raw_embedding_root=args.raw_embedding_root,
            worker_input_manifest=args.worker_input_manifest,
            panel_twin_root=args.panel_twin_root,
            panel_independent_receipt=args.panel_independent_receipt,
            trust_receipt=args.trust_receipt,
            trust_manifest=args.trust_manifest,
            bundle_root=args.bundle_root,
            config_path=args.config,
            code_manifest=args.code_manifest,
            repo_root=args.repo_root,
            expected_git_commit=args.expected_git_commit,
            cuda_driver_version=args.cuda_driver_version,
            output_dir=args.output_dir,
            forbidden_prefixes=args.forbidden_prefix,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the Slurm launcher
    raise SystemExit(main())

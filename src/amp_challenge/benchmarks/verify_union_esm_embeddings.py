"""Independently verify the exact union-panel ESM2 embedding twins.

This verifier intentionally imports none of the embedding producers or workers.
It validates the accepted 952-sequence panel, the complete AMP-Diffusion trust
bundle, both immutable producer publications, and a separately computed GPU
reference.  Acceptance is deliberately narrow: the verified matrix may be used
as a downstream embedding feature input, but this receipt says nothing about
predictions, performance, ensemble weights, or pretraining independence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_DRIVER_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ABSOLUTE_BYTES_RE = re.compile(
    rb"(?:/home/|/lustre/|/tmp/|/scratch/|/mnt/|file://|(?<![A-Za-z0-9])[A-Za-z]:[\\\\/])"
)
_HEADER_RE = re.compile(r">sequence_id=([0-9a-f]{64})")

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_EXPECTED_ROWS = 952
_EXPECTED_COLUMNS = 320
_EXPECTED_TENSOR_BYTES = _EXPECTED_ROWS * _EXPECTED_COLUMNS * 4
_EXPECTED_NPY_BYTES = _EXPECTED_TENSOR_BYTES + 128

_CONFIG_PATH = "configs/models/esm2_union_embeddings_v1.toml"
_PACKAGE_PATH = "src/amp_challenge/benchmarks/package_union_esm_embeddings.py"
_VERIFIER_PATH = "src/amp_challenge/benchmarks/verify_union_esm_embeddings.py"
_LEGACY_WORKER_PATH = "integrations/ampdiffusion/esm2_embedding_worker.py"
_REFERENCE_WORKER_PATH = "integrations/ampdiffusion/esm2_union_reference_worker.py"
_TRUST_MANIFEST_PATH = "integrations/ampdiffusion/artifacts.toml"
_ADAPTER_PATH = "integrations/ampdiffusion/adapter.py"

_CODE_FIXED_PATHS = frozenset(
    {
        "cluster/slurm/audit_union_esm_embeddings_v1_twins.sbatch",
        "cluster/slurm/extract_union_esm_embeddings_v1_twins.sbatch",
        "cluster/slurm/validate_union_esm_embeddings.sbatch",
        "cluster/validate_union_esm_embeddings_output.sh",
        _CONFIG_PATH,
        _ADAPTER_PATH,
        _TRUST_MANIFEST_PATH,
        _LEGACY_WORKER_PATH,
        _REFERENCE_WORKER_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)

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
_PRODUCER_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "embeddings/SHA256SUMS",
        "embeddings/embedding_index.csv",
        "embeddings/embedding_manifest.json",
        "embeddings/embeddings.npy",
    }
)
_EMBEDDING_FILES = frozenset(
    {"SHA256SUMS", "embedding_index.csv", "embedding_manifest.json", "embeddings.npy"}
)
_REFERENCE_FILES = frozenset({"reference_embeddings.npy", "reference_manifest.json"})
_HANDSHAKE_FILES = frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"})

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

_AUDIT_CHECKS = frozenset(
    {
        "accepted_panel_evidence_chain_valid",
        "accepted_panel_twins_immutable_and_identical",
        "all_eight_trusted_bundle_artifacts_rehashed",
        "canonical_index_independently_reconstructed",
        "canonical_npy_v1_little_endian_float32_verified",
        "code_manifest_and_git_attestation_valid",
        "full_matrix_independently_recomputed",
        "independent_reference_manifest_reconstructed",
        "inputs_rehashed_after_verification",
        "path_free_publication_and_receipt",
        "producer_frozen_input_manifest_reconstructed",
        "producer_semantic_manifest_independently_reconstructed",
        "producer_twins_bitwise_identical",
        "producer_reference_npy_bitwise_identical",
        "producer_reference_tensor_bytes_bitwise_identical",
        "producer_reference_runtime_and_driver_identical",
        "scope_limited_to_embedding_feature_input",
    }
)

_PINNED_PANEL = {
    "git_commit": "b63c47150e19fde0a8af8ce51830f3bccf652af4",
    "producer_job_id": 223287,
    "audit_job_id": 223294,
    "publication_top_sha256": "8815e45b46bde98474e3a85711da082457da47102e1d73cf33ed1806932a1894",
    "semantic_top_sha256": "0f56badbdebec55c24c799d62ce0358f5aec846a7675c9afccbf550c884a2ba8",
    "fasta_sha256": "fd8b791687578c7a8b9ea7212b5319a3aaf8b46e03d8988706772ecc4d11e419",
    "manifest_sha256": "bc04dbcc8e09cfd27dc83dbc9610fe4250b9b5b279495b324b8d6d4138d16900",
    "coverage_sha256": "f3602885542fc319848e588a8eaffb5655f1afb167678f95772b9ff027132a2b",
    "independent_receipt_sha256": "f3b121f30deea0d883f53a4a5a2d270b87d1e8726f0941bf5efaef7eb2215de8",
    "sequence_ids_sha256": "45a704812a51d51876b8e32e86c8890db5b7d6aa52de1b1f35634e797acd3f03",
    "config_sha256": "616ea482e3310e71d3712dd849c9307f4fd490842d4fb38e268ee1abc31d5a4c",
    "code_manifest_sha256": "76dc4cc13f96a7df6178bef06a58987e4d48789f9a60748ca516a8f81cfbffe1",
    "frozen_manifest_sha256": "74e0383dabf5a0935afd65c4aec859893a3768516ebf70155ac368345928ec19",
}

_EXPECTED_RUNTIME = {
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
_EXPECTED_DETERMINISM = {
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "tf32": False,
    "exact_twin_bytes_required": True,
    "independent_full_reextraction_required": True,
    "cuda_driver_policy": "producer_twins_and_independent_audit_exact_match",
}
_EXPECTED_ACCEPTANCE = {
    "producer_status": "candidate_pending_independent_numerical_verification",
    "verified_status": "accepted_for_downstream_embedding_feature_input_only",
    "embeddings_verified_by_producer": False,
    "model_predictions_verified": False,
    "model_performance_evidence": False,
    "ensemble_weight_established": False,
    "pretraining_membership_independence_established": False,
}
_ACCEPTED_SCOPE = {
    "embedding_feature_input_eligible": True,
    "accepted_use": "downstream_embedding_feature_input_only",
    "embedding_extraction_executed": True,
    "embeddings_verified": True,
    "full_embedding_matrix_independently_recomputed": True,
    "producer_twins_bitwise_identical": True,
    "independent_reference_bitwise_identical": True,
    "model_predictions_verified": False,
    "model_performance_evidence": False,
    "ensemble_weight_established": False,
    "pretraining_membership_independence_established": False,
}


class VerificationError(ValueError):
    """Raised when an embedding audit invariant fails."""


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
class ArtifactSpec:
    sha256: str
    size: int
    role: str


@dataclass(frozen=True, slots=True)
class Contract:
    snapshot: Snapshot
    values: Mapping[str, object]

    def __getattr__(self, name: str) -> object:
        try:
            return self.values[name]
        except KeyError as error:  # pragma: no cover - programmer error
            raise AttributeError(name) from error


@dataclass(frozen=True, slots=True)
class MatrixEvidence:
    snapshot: Snapshot
    tensor_payload: bytes
    tensor_data_sha256: str
    rows: int
    columns: int


@dataclass(frozen=True, slots=True)
class PanelEvidence:
    root: Path
    records: tuple[FastaRecord, ...]
    receipt: Mapping[str, object]
    receipt_snapshot: Snapshot
    handshake: Mapping[str, object]
    snapshots: tuple[Snapshot, ...]
    relative_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TrustEvidence:
    bundle_root: Path
    receipt_snapshot: Snapshot
    manifest_snapshot: Snapshot
    artifacts: Mapping[str, ArtifactSpec]
    artifact_snapshots: tuple[Snapshot, ...]
    integration: str
    source: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProducerEvidence:
    root: Path
    handshake: Mapping[str, object]
    top_snapshot: Snapshot
    top_entries: Mapping[str, str]
    semantic_top_snapshot: Snapshot
    semantic_entries: Mapping[str, str]
    code_manifest: Snapshot
    frozen_manifest: Snapshot
    matrix: MatrixEvidence
    index: Snapshot
    semantic_manifest: Snapshot
    semantic_document: Mapping[str, object]
    snapshots: tuple[Snapshot, ...]


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    root: Path
    matrix: MatrixEvidence
    manifest: Snapshot
    document: Mapping[str, object]
    runtime: Mapping[str, object]
    snapshots: tuple[Snapshot, ...]


_REQUIRED_ARTIFACTS: Mapping[str, ArtifactSpec] = {
    "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt": ArtifactSpec(
        "8f7a4557d57713b97ba0e484303007efb7230d25299c0ac47a0a1b12a87bbb9d",
        1511,
        "pickle_checkpoint",
    ),
    "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt": ArtifactSpec(
        "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928",
        30099493,
        "pickle_checkpoint",
    ),
    "source/checkpoint/model.pt": ArtifactSpec(
        "6a3f347df7c02ff6008ac3d2d4826daeadf7418cb6862a6599b4f710e1d7f8aa",
        132526179,
        "pickle_checkpoint",
    ),
    "source/pyproject.toml": ArtifactSpec(
        "eeb7551dee5cfa917dbfaed1e0c29eb82c541d8d08312529d86bddeb6ec14601",
        1551,
        "environment_manifest",
    ),
    "source/src/ampdiffusion_starter_kit/__init__.py": ArtifactSpec(
        "6d4e5d597db1cd13a6df937e0bc682802b1f274aca03f39da2015295e3b52c69",
        72,
        "executable_code",
    ),
    "source/src/ampdiffusion_starter_kit/generate.py": ArtifactSpec(
        "20a7ce7a9086f647e04b191876cb12176fc3af9f6ba68d913b91722f6f9b5f24",
        12071,
        "executable_code",
    ),
    "source/src/ampdiffusion_starter_kit/model.py": ArtifactSpec(
        "3abc19641c5f3824d3358caf5b598005e59efd9d2f00c5218c7e329adbc0a638",
        21326,
        "executable_code",
    ),
    "source/uv.lock": ArtifactSpec(
        "aaf37baa3adf5070dd3090c43527daa2696741a7e3057c72c78d51a80af9e234",
        42112,
        "environment_lock",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    for item in (candidate, *candidate.parents):
        _require(not item.is_symlink(), f"{label} must not traverse a symbolic link")


def _resolve_without_symlinks(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _reject_symlink_chain(requested, label=label)
    return requested.resolve(strict=True)


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    resolved = _resolve_without_symlinks(path, label=label)
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is not a real directory")
    return resolved


def _read_snapshot(path: str | Path, *, label: str) -> Snapshot:
    resolved = _resolve_without_symlinks(path, label=label)
    before = resolved.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return Snapshot(
        path=resolved,
        payload=payload,
        sha256=_sha256(payload),
        fingerprint=_fingerprint(after),
    )


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = _read_snapshot(snapshot.path, label=label)
    _require(
        current.sha256 == snapshot.sha256 and current.fingerprint == snapshot.fingerprint,
        f"{label} changed during verification",
    )


def _safe_relative_path(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and value != "" and "\\" not in value, f"bad {label}")
    text = cast(str, value)
    pure = PurePosixPath(text)
    _require(
        not pure.is_absolute()
        and pure.as_posix() == text
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"unsafe {label}",
    )
    return text


def _tree_inventory(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        _require(not item.is_symlink(), f"tree contains symbolic link: {relative}")
        mode = item.stat().st_mode
        if stat.S_ISREG(mode):
            files.add(relative)
        elif stat.S_ISDIR(mode):
            directories.add(relative)
        else:
            raise VerificationError(f"tree contains non-file entry: {relative}")
    return frozenset(files), frozenset(directories)


def _verify_immutable_tree(root: Path, *, label: str) -> None:
    files, directories = _tree_inventory(root)
    for relative in ("", *sorted(files), *sorted(directories)):
        path = root if relative == "" else root / relative
        _require(path.stat().st_mode & 0o222 == 0, f"{label} remains writable: {relative or '.'}")


def _reject_constant(value: str) -> object:
    raise VerificationError(f"non-finite JSON number {value}")


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
        raise VerificationError(f"{label} is not strict UTF-8 JSON") from error


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _compact_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _read_pretty_json(snapshot: Snapshot, *, label: str) -> dict[str, object]:
    value = _loads_json(snapshot.payload, label=label)
    _require(isinstance(value, dict), f"{label} is not an object")
    document = cast(dict[str, object], value)
    _require(
        snapshot.payload == _pretty_json_bytes(document), f"{label} is not canonical pretty JSON"
    )
    return document


def _read_compact_json(snapshot: Snapshot, *, label: str) -> dict[str, object]:
    value = _loads_json(snapshot.payload, label=label)
    _require(isinstance(value, dict), f"{label} is not an object")
    document = cast(dict[str, object], value)
    _require(
        snapshot.payload == _compact_json_bytes(document), f"{label} is not canonical compact JSON"
    )
    return document


def _mapping(value: object, *, label: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return cast(dict[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: Iterable[str], *, label: str) -> None:
    _require(set(value) == set(expected), f"{label} fields changed")


def _sha_field(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"bad {label}")
    return cast(str, value)


def _positive_int(value: object, *, label: str) -> int:
    _require(type(value) is int and cast(int, value) > 0, f"bad {label}")
    return cast(int, value)


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(payload.endswith(b"\n") and b"\r" not in payload, f"bad {label} framing")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not ASCII") from error
    _require(lines, f"{label} is empty")
    entries: dict[str, str] = {}
    previous: str | None = None
    for line in lines:
        _require(len(line) >= 67 and line[64:66] == "  ", f"bad {label} line")
        digest = _sha_field(line[:64], label=f"{label} digest")
        name = _safe_relative_path(line[66:], label=f"{label} path")
        _require(name not in entries, f"{label} repeats {name}")
        _require(previous is None or name > previous, f"{label} is not sorted")
        entries[name] = digest
        previous = name
    return entries


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("ascii")


def _assert_path_free(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_path_free(key, label=label)
            _assert_path_free(item, label=label)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_path_free(item, label=label)
    elif isinstance(value, str):
        _require(
            not value.startswith("/") and not _WINDOWS_ABSOLUTE_RE.match(value),
            f"{label} exposes an absolute path",
        )
        _require("file://" not in value, f"{label} exposes an absolute path")


def _scan_path_free(payloads: Iterable[bytes], forbidden_prefixes: Sequence[str]) -> None:
    forbidden = [prefix.encode("utf-8") for prefix in forbidden_prefixes]
    for prefix in forbidden:
        _require(prefix != b"", "forbidden prefix is empty")
    for payload in payloads:
        _require(
            _ABSOLUTE_BYTES_RE.search(payload) is None, "public evidence exposes an absolute path"
        )
        for prefix in forbidden:
            _require(prefix not in payload, "public evidence exposes a forbidden prefix")


def _verify_manifest_tree(
    root: Path,
    *,
    expected_inventory: frozenset[str],
    expected_top_sha256: str | None,
    label: str,
) -> tuple[Snapshot, dict[str, str], tuple[Snapshot, ...]]:
    files, directories = _tree_inventory(root)
    _require(files == expected_inventory, f"{label} file inventory changed")
    expected_directories = {
        parent.as_posix()
        for name in expected_inventory
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    _require(directories == expected_directories, f"{label} directory inventory changed")
    top = _read_snapshot(root / "SHA256SUMS", label=f"{label} SHA256SUMS")
    if expected_top_sha256 is not None:
        _require(top.sha256 == expected_top_sha256, f"{label} top checksum changed")
    entries = _parse_sha_manifest(top.payload, label=f"{label} SHA256SUMS")
    _require(
        set(entries) == set(expected_inventory) - {"SHA256SUMS"},
        f"{label} checksum coverage changed",
    )
    snapshots = [top]
    for name, digest in entries.items():
        snapshot = _read_snapshot(root / name, label=f"{label} {name}")
        _require(snapshot.sha256 == digest, f"{label} hash mismatch for {name}")
        snapshots.append(snapshot)
    return top, entries, tuple(snapshots)


def _verify_tree_bytes(left: Path, right: Path, *, inventory: Iterable[str], label: str) -> None:
    for name in sorted(inventory):
        left_snapshot = _read_snapshot(left / name, label=f"left {label} {name}")
        right_snapshot = _read_snapshot(right / name, label=f"right {label} {name}")
        _require(left_snapshot.payload == right_snapshot.payload, f"{label} twins differ at {name}")


def _verify_handshake(root: Path, *, label: str) -> tuple[dict[str, object], tuple[Snapshot, ...]]:
    receipts_root = root / "node-receipts"
    _require(
        receipts_root.is_dir() and not receipts_root.is_symlink(),
        f"{label} handshake directory missing",
    )
    files, directories = _tree_inventory(receipts_root)
    _require(files == _HANDSHAKE_FILES and not directories, f"{label} handshake inventory changed")
    _verify_immutable_tree(receipts_root, label=f"{label} handshake")
    job_id = root.name
    _require(job_id.isdigit(), f"{label} root basename is not a Slurm job ID")
    receipts: dict[int, Snapshot] = {}
    nodes: dict[int, str] = {}
    for task in (0, 1):
        snapshot = _read_snapshot(
            receipts_root / f"{task}.receipt", label=f"{label} receipt {task}"
        )
        _require(
            snapshot.payload.endswith(b"\n") and b"\r" not in snapshot.payload,
            f"bad {label} receipt framing",
        )
        try:
            lines = snapshot.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise VerificationError(f"{label} receipt is not UTF-8") from error
        _require(
            len(lines) == 3
            and lines[0] == f"array_job_id={job_id}"
            and lines[1] == f"array_task_id={task}"
            and lines[2].startswith("node_name="),
            f"invalid {label} receipt {task}",
        )
        node = lines[2].removeprefix("node_name=")
        _require(_NODE_RE.fullmatch(node) is not None, f"unsafe {label} node name")
        receipts[task] = snapshot
        nodes[task] = node
    _require(nodes[0] != nodes[1], f"{label} twins did not run on distinct nodes")
    receipt_hashes = {str(task): receipts[task].sha256 for task in (0, 1)}
    acknowledgement_hashes: dict[str, str] = {}
    acknowledgements: list[Snapshot] = []
    for task in (0, 1):
        snapshot = _read_snapshot(
            receipts_root / f"{task}.ack", label=f"{label} acknowledgement {task}"
        )
        expected = f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode(
            "ascii"
        )
        _require(snapshot.payload == expected, f"{label} sibling acknowledgement changed")
        acknowledgement_hashes[str(task)] = snapshot.sha256
        acknowledgements.append(snapshot)
    return (
        {
            "distinct_nodes": True,
            "bidirectional_acknowledgement": True,
            "receipt_sha256": receipt_hashes,
            "acknowledgement_sha256": acknowledgement_hashes,
        },
        (*receipts.values(), *acknowledgements),
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


def _parse_config(snapshot: Snapshot) -> Contract:
    try:
        document = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("embedding config is not valid UTF-8 TOML") from error
    _exact_keys(document, _CONFIG_FIELDS, label="embedding config")
    _require(
        type(document["schema_version"]) is int and document["schema_version"] == 1,
        "bad embedding config schema",
    )
    _require(
        document["artifact"] == "gate1_union_esm2_embeddings_v1", "bad embedding config artifact"
    )

    exact_scalars = {
        "panel_git_commit": _PINNED_PANEL["git_commit"],
        "panel_producer_job_id": _PINNED_PANEL["producer_job_id"],
        "panel_audit_job_id": _PINNED_PANEL["audit_job_id"],
        "panel_publication_top_sha256": _PINNED_PANEL["publication_top_sha256"],
        "panel_top_sha256": _PINNED_PANEL["semantic_top_sha256"],
        "panel_fasta_sha256": _PINNED_PANEL["fasta_sha256"],
        "panel_manifest_sha256": _PINNED_PANEL["manifest_sha256"],
        "panel_coverage_receipt_sha256": _PINNED_PANEL["coverage_sha256"],
        "panel_independent_receipt_sha256": _PINNED_PANEL["independent_receipt_sha256"],
        "expected_sequences": _EXPECTED_ROWS,
        "expected_sequence_ids_sha256": _PINNED_PANEL["sequence_ids_sha256"],
        "embedding_model": "esm2_t6_8M_UR50D",
        "representation_layer": 6,
        "embedding_dimension": _EXPECTED_COLUMNS,
        "pooling": "arithmetic mean over residue representations; BOS/EOS/padding excluded",
        "embedding_dtype": "float32",
        "embedding_batch_size": 128,
        "embedding_seed": 20260902,
        "embedding_worker_sha256": "34fdb77e87aaa44044960655259cc60fb20b165bfb87c38ba19bee21565f2cba",
        "trust_manifest_sha256": "5e62db4fd3241e1fa6bbd83cc0f26e299c6d16e9fa1e449304fab2e4c98fd9ff",
        "bundle_verification_receipt_sha256": "ad9655f57cbb4d033c88fe38dfdfeffb56fd24274749c2caded890667e63e850",
        "ampdiffusion_source_commit": "1a862af9078e6b55c87d1fa576f3da81851ba94b",
        "model_checkpoint_sha256": _REQUIRED_ARTIFACTS[
            "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt"
        ].sha256,
        "contact_regression_sha256": _REQUIRED_ARTIFACTS[
            "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt"
        ].sha256,
        "environment_lock_sha256": _REQUIRED_ARTIFACTS["source/uv.lock"].sha256,
    }
    for key, expected in exact_scalars.items():
        _require(document[key] == expected, f"embedding config changed {key}")
    for key in (
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
    ):
        _sha_field(document[key], label=key)
    _require(
        _GIT_RE.fullmatch(cast(str, document["panel_git_commit"])) is not None,
        "bad panel Git commit",
    )
    _require(
        _GIT_RE.fullmatch(cast(str, document["ampdiffusion_source_commit"])) is not None,
        "bad AMP-Diffusion commit",
    )
    runtime = _mapping(document["runtime"], label="config runtime")
    determinism = _mapping(document["determinism"], label="config determinism")
    acceptance = _mapping(document["acceptance"], label="config acceptance")
    _require(runtime == _EXPECTED_RUNTIME, "config runtime changed")
    _require(determinism == _EXPECTED_DETERMINISM, "config determinism changed")
    _require(acceptance == _EXPECTED_ACCEPTANCE, "config acceptance scope changed")
    return Contract(snapshot=snapshot, values=document)


def load_config(path: str | Path) -> Contract:
    """Read and strictly validate the pinned embedding-audit contract."""

    return _parse_config(_read_snapshot(path, label="embedding config"))


def _set_digest(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    payload = b"" if not ordered else ("\n".join(ordered) + "\n").encode("ascii")
    return _sha256(payload)


def _read_strict_fasta(snapshot: Snapshot) -> tuple[FastaRecord, ...]:
    payload = snapshot.payload
    _require(
        payload
        and payload.endswith(b"\n")
        and not payload.endswith(b"\n\n")
        and b"\r" not in payload,
        "accepted panel FASTA has invalid LF framing",
    )
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError("accepted panel FASTA is not ASCII") from error
    _require(len(lines) == 2 * _EXPECTED_ROWS, "accepted panel FASTA count changed")
    records: list[FastaRecord] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    previous: str | None = None
    for offset in range(0, len(lines), 2):
        match = _HEADER_RE.fullmatch(lines[offset])
        _require(match is not None, f"invalid FASTA header at line {offset + 1}")
        assert match is not None
        sequence_id = match.group(1)
        sequence = lines[offset + 1]
        _require(8 <= len(sequence) <= 50, f"invalid FASTA length at line {offset + 2}")
        _require(
            sequence == sequence.upper() and set(sequence) <= _AMINO_ACIDS,
            f"invalid FASTA sequence at line {offset + 2}",
        )
        _require(
            _sha256(sequence.encode("ascii")) == sequence_id,
            f"FASTA ID mismatch at line {offset + 1}",
        )
        _require(
            sequence_id not in seen_ids and sequence not in seen_sequences, "duplicate FASTA record"
        )
        _require(
            previous is None or sequence_id > previous,
            "accepted panel FASTA is not strictly sorted",
        )
        records.append(FastaRecord(sequence_id=sequence_id, sequence=sequence))
        seen_ids.add(sequence_id)
        seen_sequences.add(sequence)
        previous = sequence_id
    _require(
        _set_digest(seen_ids) == _PINNED_PANEL["sequence_ids_sha256"],
        "panel sequence-ID digest changed",
    )
    return tuple(records)


def _verify_panel_receipt(document: Mapping[str, object], *, contract: Contract) -> None:
    _exact_keys(
        document,
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
    _require(
        document["schema_version"] == 1
        and document["artifact"] == "gate1_union_esm2_exact_panel_fasta_v1_independent_verification"
        and document["status"] == "accepted_for_embedding_input_only",
        "panel independent receipt identity changed",
    )
    _require(
        document["git_commit"] == contract.panel_git_commit, "panel receipt Git commit changed"
    )
    _require(
        document["publication_top_manifest_sha256"] == contract.panel_publication_top_sha256
        and document["panel_top_manifest_sha256"] == contract.panel_top_sha256,
        "panel receipt top-manifest links changed",
    )
    _require(
        document["config_sha256"] == _PINNED_PANEL["config_sha256"]
        and document["code_manifest_sha256"] == _PINNED_PANEL["code_manifest_sha256"]
        and document["frozen_input_manifest_sha256"] == _PINNED_PANEL["frozen_manifest_sha256"],
        "panel receipt code/config/frozen links changed",
    )
    checks = _mapping(document["checks"], label="panel receipt checks")
    _exact_keys(checks, _PANEL_RECEIPT_CHECKS, label="panel receipt checks")
    _require(
        all(value is True for value in checks.values()), "panel receipt contains a failed check"
    )
    scope = _mapping(document["acceptance_scope"], label="panel receipt scope")
    _require(
        scope.get("embedding_input_eligible") is True
        and scope.get("embedding_extraction_executed") is False
        and scope.get("embeddings_verified") is False
        and scope.get("model_predictions_verified") is False
        and scope.get("model_performance_evidence") is False,
        "panel receipt overstates its scope",
    )
    artifacts = _mapping(document["artifact_sha256"], label="panel receipt artifacts")
    _require(
        artifacts
        == {
            "SHA256SUMS": contract.panel_publication_top_sha256,
            "panel/SHA256SUMS": contract.panel_top_sha256,
            "panel/coverage_receipt.json": contract.panel_coverage_receipt_sha256,
            "panel/manifest.json": contract.panel_manifest_sha256,
            "panel/union_esm_sequences.fasta": contract.panel_fasta_sha256,
        },
        "panel receipt artifact links changed",
    )
    panel = _mapping(document["panel"], label="panel receipt census")
    _require(
        panel.get("unique_sequences") == _EXPECTED_ROWS
        and panel.get("sequence_ids_sha256") == contract.expected_sequence_ids_sha256,
        "panel receipt census changed",
    )
    extraction = _mapping(
        document["declared_extraction_contract_not_execution_evidence"],
        label="panel extraction declaration",
    )
    expected_extraction = {
        "status": "declared_contract_hash_bound_but_extraction_not_executed_or_verified",
        "embedding_model": contract.embedding_model,
        "representation_layer": contract.representation_layer,
        "embedding_dimension": contract.embedding_dimension,
        "embedding_batch_size": contract.embedding_batch_size,
        "embedding_seed": contract.embedding_seed,
        "model_checkpoint_sha256": contract.model_checkpoint_sha256,
        "contact_regression_sha256": contract.contact_regression_sha256,
        "environment_lock_sha256": contract.environment_lock_sha256,
        "trust_manifest_sha256": contract.trust_manifest_sha256,
        "embedding_worker_sha256": contract.embedding_worker_sha256,
    }
    _require(extraction == expected_extraction, "panel extraction declaration changed")


def _verify_panel_documents(
    *,
    manifest: Mapping[str, object],
    coverage: Mapping[str, object],
    receipt: Mapping[str, object],
    contract: Contract,
) -> None:
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact") == "gate1_union_esm2_exact_panel_fasta_v1"
        and manifest.get("status") == "candidate_pending_independent_verification"
        and manifest.get("production_eligible") is False,
        "panel manifest semantics changed",
    )
    _require(
        manifest.get("git_commit") == contract.panel_git_commit, "panel manifest commit changed"
    )
    artifacts = _mapping(manifest.get("artifacts"), label="panel manifest artifacts")
    _require(
        artifacts.get("fasta")
        == {
            "filename": "union_esm_sequences.fasta",
            "records": _EXPECTED_ROWS,
            "sha256": contract.panel_fasta_sha256,
        }
        and artifacts.get("coverage_receipt")
        == {
            "filename": "coverage_receipt.json",
            "sha256": contract.panel_coverage_receipt_sha256,
        },
        "panel manifest artifact links changed",
    )
    receipt_panel = _mapping(receipt["panel"], label="panel receipt census")
    _require(manifest.get("panel") == receipt_panel, "panel manifest census changed")
    _require(
        coverage.get("schema_version") == 1
        and coverage.get("artifact") == "gate1_union_esm2_exact_panel_coverage_v1"
        and coverage.get("status") == "passed"
        and coverage.get("panel") == receipt_panel,
        "panel coverage receipt changed",
    )
    export = _mapping(coverage.get("export"), label="panel coverage export")
    _require(
        export.get("records") == _EXPECTED_ROWS
        and export.get("fasta_sha256") == contract.panel_fasta_sha256
        and export.get("ordering") == "ascending sequence_id"
        and export.get("missing_sequences") == 0
        and export.get("extra_sequences") == 0
        and export.get("missing_sequence_ids") == []
        and export.get("extra_sequence_ids") == [],
        "panel coverage is not exact",
    )


def _validate_panel_evidence(
    *, twin_root: str | Path, independent_receipt: str | Path, contract: Contract
) -> PanelEvidence:
    root = _resolved_directory(twin_root, label="accepted panel twin root")
    _require(root.name == str(contract.panel_producer_job_id), "wrong accepted panel job root")
    files, directories = _tree_inventory(root)
    expected_files = {f"{task}/{name}" for task in (0, 1) for name in _PANEL_PUBLICATION_FILES} | {
        f"node-receipts/{name}" for name in _HANDSHAKE_FILES
    }
    expected_directories = {
        "0",
        "0/panel",
        "1",
        "1/panel",
        "node-receipts",
    }
    _require(
        files == expected_files and directories == expected_directories,
        "panel job-root inventory changed",
    )
    receipt_snapshot = _read_snapshot(independent_receipt, label="panel independent receipt")
    _require(
        receipt_snapshot.path.stat().st_mode & 0o222 == 0,
        "panel independent receipt remains writable",
    )
    _require(
        receipt_snapshot.sha256 == contract.panel_independent_receipt_sha256,
        "wrong panel receipt hash",
    )
    receipt = _read_pretty_json(receipt_snapshot, label="panel independent receipt")
    _verify_panel_receipt(receipt, contract=contract)
    handshake, handshake_snapshots = _verify_handshake(root, label="accepted panel")
    _require(handshake == receipt["production_handshake"], "panel receipt handshake changed")
    run_snapshots: list[Snapshot] = []
    relative_hashes: dict[str, str] = {}
    top_records: list[tuple[Snapshot, Mapping[str, str], Snapshot, Mapping[str, str]]] = []
    for task in (0, 1):
        run = root / str(task)
        top, top_entries, snapshots = _verify_manifest_tree(
            run,
            expected_inventory=_PANEL_PUBLICATION_FILES,
            expected_top_sha256=cast(str, contract.panel_publication_top_sha256),
            label=f"accepted panel twin {task}",
        )
        semantic_top, semantic_entries, semantic_snapshots = _verify_manifest_tree(
            run / "panel",
            expected_inventory=_PANEL_SEMANTIC_FILES,
            expected_top_sha256=cast(str, contract.panel_top_sha256),
            label=f"accepted panel semantic twin {task}",
        )
        _require(
            top_entries["panel/SHA256SUMS"] == semantic_top.sha256, "panel top manifest is unbound"
        )
        _verify_immutable_tree(run, label=f"accepted panel twin {task}")
        run_snapshots.extend((*snapshots, *semantic_snapshots))
        top_records.append((top, top_entries, semantic_top, semantic_entries))
        for name in _PANEL_PUBLICATION_FILES:
            relative_hashes[f"{contract.panel_producer_job_id}/{task}/{name}"] = _read_snapshot(
                run / name, label=f"panel frozen input {task}/{name}"
            ).sha256
    _verify_tree_bytes(
        root / "0", root / "1", inventory=_PANEL_PUBLICATION_FILES, label="accepted panel"
    )
    _require(
        top_records[0][1] == top_records[1][1] and top_records[0][3] == top_records[1][3],
        "panel manifest twins differ",
    )
    for name in _HANDSHAKE_FILES:
        relative_hashes[f"{contract.panel_producer_job_id}/node-receipts/{name}"] = _read_snapshot(
            root / "node-receipts" / name, label=f"panel frozen handshake {name}"
        ).sha256
    artifacts = _mapping(receipt["artifact_sha256"], label="panel receipt artifacts")
    _require(
        top_records[0][1]["CODE_SHA256SUMS"] == receipt["code_manifest_sha256"]
        and top_records[0][1]["FROZEN_INPUT_SHA256SUMS"] == receipt["frozen_input_manifest_sha256"],
        "panel code/frozen manifest binding changed",
    )
    _require(
        top_records[0][1]["CODE_SHA256SUMS"] == _PINNED_PANEL["code_manifest_sha256"]
        and top_records[0][1]["FROZEN_INPUT_SHA256SUMS"] == _PINNED_PANEL["frozen_manifest_sha256"],
        "accepted panel producer attestations changed",
    )
    for name, digest in artifacts.items():
        _require(
            _read_snapshot(root / "0" / name, label=f"panel artifact {name}").sha256 == digest,
            f"panel receipt mismatch at {name}",
        )
    fasta = _read_snapshot(root / "0/panel/union_esm_sequences.fasta", label="accepted panel FASTA")
    manifest_snapshot = _read_snapshot(
        root / "0/panel/manifest.json", label="accepted panel manifest"
    )
    coverage_snapshot = _read_snapshot(
        root / "0/panel/coverage_receipt.json", label="accepted panel coverage"
    )
    _require(
        fasta.sha256 == contract.panel_fasta_sha256
        and manifest_snapshot.sha256 == contract.panel_manifest_sha256
        and coverage_snapshot.sha256 == contract.panel_coverage_receipt_sha256,
        "accepted panel semantic hashes changed",
    )
    records = _read_strict_fasta(fasta)
    _verify_panel_documents(
        manifest=_read_pretty_json(manifest_snapshot, label="accepted panel manifest"),
        coverage=_read_pretty_json(coverage_snapshot, label="accepted panel coverage"),
        receipt=receipt,
        contract=contract,
    )
    snapshots = tuple(
        {
            snapshot.path: snapshot
            for snapshot in (receipt_snapshot, *handshake_snapshots, *run_snapshots)
        }.values()
    )
    return PanelEvidence(
        root=root,
        records=records,
        receipt=receipt,
        receipt_snapshot=receipt_snapshot,
        handshake=handshake,
        snapshots=snapshots,
        relative_hashes=relative_hashes,
    )


def _bundle_file(bundle: Path, logical_path: str, *, label: str) -> Path:
    relative = _safe_relative_path(logical_path, label=label)
    candidate = bundle.joinpath(*PurePosixPath(relative).parts)
    resolved = _resolve_without_symlinks(candidate, label=label)
    _require(resolved.is_relative_to(bundle), f"{label} escapes the bundle")
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is not a real file")
    return resolved


def _validate_trust_evidence(
    *,
    trust_receipt: str | Path,
    trust_manifest: str | Path,
    bundle_root: str | Path,
    contract: Contract,
) -> TrustEvidence:
    bundle = _resolved_directory(bundle_root, label="AMP-Diffusion bundle root")
    receipt_snapshot = _read_snapshot(trust_receipt, label="AMP-Diffusion trust receipt")
    manifest_snapshot = _read_snapshot(trust_manifest, label="AMP-Diffusion trust manifest")
    _require(
        receipt_snapshot.sha256 == contract.bundle_verification_receipt_sha256,
        "AMP-Diffusion trust receipt hash changed",
    )
    _require(
        manifest_snapshot.sha256 == contract.trust_manifest_sha256,
        "AMP-Diffusion trust manifest hash changed",
    )
    receipt = _read_compact_json(receipt_snapshot, label="AMP-Diffusion trust receipt")
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
        label="AMP-Diffusion trust receipt",
    )
    _require(
        receipt["schema_version"] == 1
        and receipt["receipt_type"] == "ampdiffusion_bundle_verification"
        and receipt["component"] == "generation"
        and receipt["bundle_root"] == str(bundle),
        "AMP-Diffusion trust receipt identity changed",
    )
    integration = receipt["integration"]
    _require(integration == "ampdiffusion_starter_fidelity_v0", "AMP-Diffusion integration changed")
    source = _mapping(receipt["source"], label="AMP-Diffusion source")
    _require(
        source
        == {
            "repository": "https://github.com/szczurek-lab/ampdiffusion-starter-kit.git",
            "commit": contract.ampdiffusion_source_commit,
            "license": "MIT",
        },
        "AMP-Diffusion source identity changed",
    )
    manifest_link = _mapping(receipt["trust_manifest"], label="trust manifest link")
    _require(
        manifest_link
        == {
            "filename": manifest_snapshot.path.name,
            "sha256": manifest_snapshot.sha256,
            "size": len(manifest_snapshot.payload),
        },
        "trust receipt does not bind the supplied manifest",
    )
    try:
        trust_document = tomllib.loads(manifest_snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("AMP-Diffusion trust manifest is invalid") from error
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
        trust_document["format_version"] == 1
        and trust_document["integration"] == integration
        and trust_document["source"] == source,
        "AMP-Diffusion trust manifest identity changed",
    )
    raw_trusted = trust_document["trusted_file"]
    _require(isinstance(raw_trusted, list) and raw_trusted, "trust manifest has no artifacts")
    generation: dict[str, ArtifactSpec] = {}
    for index, item in enumerate(cast(list[object], raw_trusted)):
        spec = _mapping(item, label=f"trust manifest artifact {index}")
        _require(
            set(spec)
            in (
                {"path", "sha256", "size", "components", "role"},
                {"path", "sha256", "size", "components", "role", "url"},
            ),
            f"trust manifest artifact {index} fields changed",
        )
        logical = _safe_relative_path(spec["path"], label=f"trust manifest artifact {index} path")
        components = spec["components"]
        _require(
            isinstance(components, list)
            and components
            and all(isinstance(value, str) and value for value in components),
            f"trust manifest artifact {index} components changed",
        )
        parsed = ArtifactSpec(
            _sha_field(spec["sha256"], label=f"trust manifest artifact {index} SHA"),
            _positive_int(spec["size"], label=f"trust manifest artifact {index} size"),
            cast(str, spec["role"]),
        )
        _require(isinstance(parsed.role, str) and parsed.role != "", "empty trust artifact role")
        if "generation" in components:
            _require(logical not in generation, "duplicate generation trust artifact")
            generation[logical] = parsed
    _require(generation == _REQUIRED_ARTIFACTS, "generation trust inventory or pins changed")

    raw_receipt_artifacts = receipt["artifacts"]
    _require(isinstance(raw_receipt_artifacts, list), "trust receipt artifacts are not a list")
    receipt_artifacts: dict[str, ArtifactSpec] = {}
    artifact_snapshots: list[Snapshot] = []
    previous: str | None = None
    for index, item in enumerate(cast(list[object], raw_receipt_artifacts)):
        spec = _mapping(item, label=f"trust receipt artifact {index}")
        _exact_keys(
            spec, {"path", "sha256", "size", "role"}, label=f"trust receipt artifact {index}"
        )
        logical = _safe_relative_path(spec["path"], label=f"trust receipt artifact {index} path")
        _require(previous is None or logical > previous, "trust receipt artifacts are not sorted")
        observed = ArtifactSpec(
            _sha_field(spec["sha256"], label=f"trust receipt artifact {index} SHA"),
            _positive_int(spec["size"], label=f"trust receipt artifact {index} size"),
            cast(str, spec["role"]),
        )
        _require(logical not in receipt_artifacts, "duplicate trust receipt artifact")
        _require(
            generation.get(logical) == observed, f"trust receipt metadata changed for {logical}"
        )
        artifact = _read_snapshot(
            _bundle_file(bundle, logical, label=f"trusted artifact {logical}"),
            label=f"trusted artifact {logical}",
        )
        _require(
            artifact.sha256 == observed.sha256 and len(artifact.payload) == observed.size,
            f"trusted artifact bytes changed: {logical}",
        )
        receipt_artifacts[logical] = observed
        artifact_snapshots.append(artifact)
        previous = logical
    _require(
        receipt_artifacts == _REQUIRED_ARTIFACTS, "trust receipt does not cover all 8 artifacts"
    )
    return TrustEvidence(
        bundle_root=bundle,
        receipt_snapshot=receipt_snapshot,
        manifest_snapshot=manifest_snapshot,
        artifacts=receipt_artifacts,
        artifact_snapshots=tuple(artifact_snapshots),
        integration=cast(str, integration),
        source=source,
    )


def _canonical_index_bytes(records: Sequence[FastaRecord]) -> bytes:
    rows = ["row_index,sequence_id,sequence,length\n"]
    for index, record in enumerate(records):
        rows.append(f"{index},{record.sequence_id},{record.sequence},{len(record.sequence)}\n")
    return "".join(rows).encode("ascii")


def _canonical_npy_bytes(tensor_payload: bytes, *, rows: int, columns: int) -> bytes:
    _require(len(tensor_payload) == rows * columns * 4, "tensor payload has the wrong size")
    dictionary = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}"
    ).encode("latin1")
    prefix_length = len(b"\x93NUMPY") + 2 + 2
    padding = (-((prefix_length + len(dictionary) + 1) % 64)) % 64
    header = dictionary + b" " * padding + b"\n"
    _require(len(header) < 2**16, "NPY header is too large")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + tensor_payload


def _validate_matrix_snapshot(
    snapshot: Snapshot, *, expected_rows: int, expected_columns: int
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
        raw_document = ast.literal_eval(header.decode("latin1").strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise VerificationError("embedding NPY header is invalid") from error
    document = _mapping(raw_document, label="embedding NPY header")
    _exact_keys(document, {"descr", "fortran_order", "shape"}, label="embedding NPY header")
    _require(document["descr"] == "<f4", "embedding matrix dtype is not <f4")
    _require(document["fortran_order"] is False, "embedding matrix is not C-contiguous")
    _require(
        document["shape"] == (expected_rows, expected_columns), "embedding matrix shape changed"
    )
    tensor = payload[header_end:]
    _require(
        len(tensor) == expected_rows * expected_columns * 4, "embedding tensor byte count changed"
    )
    _require(
        payload == _canonical_npy_bytes(tensor, rows=expected_rows, columns=expected_columns),
        "embedding matrix is not canonical NPY v1.0",
    )
    _require(
        all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", tensor)),
        "embedding matrix contains a non-finite value",
    )
    return MatrixEvidence(
        snapshot=snapshot,
        tensor_payload=tensor,
        tensor_data_sha256=_sha256(tensor),
        rows=expected_rows,
        columns=expected_columns,
    )


def _core_trust_artifacts(trust: TrustEvidence) -> dict[str, dict[str, object]]:
    roles = {
        "model_checkpoint": "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt",
        "contact_regression_checkpoint": "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt",
        "environment_lock": "source/uv.lock",
    }
    return {
        role: {
            "path": path,
            "sha256": trust.artifacts[path].sha256,
            "size": trust.artifacts[path].size,
        }
        for role, path in roles.items()
    }


def _semantic_trust(trust: TrustEvidence) -> dict[str, object]:
    return {
        "integration": trust.integration,
        "source": dict(trust.source),
        "trust_manifest_sha256": trust.manifest_snapshot.sha256,
        "bundle_verification_receipt_sha256": trust.receipt_snapshot.sha256,
        "artifacts": _core_trust_artifacts(trust),
    }


def _reference_trust(trust: TrustEvidence) -> dict[str, object]:
    return {
        "integration": trust.integration,
        "source": dict(trust.source),
        "trust_receipt_sha256": trust.receipt_snapshot.sha256,
        "trust_manifest_sha256": trust.manifest_snapshot.sha256,
        "trusted_artifact_count": len(trust.artifacts),
        "required_artifacts": {
            path: {"sha256": spec.sha256, "size": spec.size, "role": spec.role}
            for path, spec in sorted(trust.artifacts.items())
        },
    }


def _worker_input_manifest(*, panel: PanelEvidence, contract: Contract) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "gate1_esm2_embedding_input",
        "status": "private_worker_compatibility_input_not_public_evidence",
        "artifact": {
            "filename": "union_esm_sequences.fasta",
            "sha256": contract.panel_fasta_sha256,
        },
        "records": _EXPECTED_ROWS,
        "ordering": "ascending sequence_id",
        "sequence_ids_sha256": contract.expected_sequence_ids_sha256,
        "config": {"artifact": contract.artifact, "sha256": contract.snapshot.sha256},
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


def _raw_worker_manifest(
    *,
    contract: Contract,
    matrix: MatrixEvidence,
    index: Snapshot,
    worker_input_sha256: str,
    trust: TrustEvidence,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "esm2_embedding_extraction",
        "model": contract.embedding_model,
        "representation_layer": contract.representation_layer,
        "embedding_dimension": contract.embedding_dimension,
        "pooling": contract.pooling,
        "dtype": contract.embedding_dtype,
        "records": contract.expected_sequences,
        "ordering": "input FASTA order (ascending sequence_id)",
        "input_fasta_sha256": contract.panel_fasta_sha256,
        "input_fasta_manifest_sha256": worker_input_sha256,
        "trust": {
            "integration": trust.integration,
            "source_commit": contract.ampdiffusion_source_commit,
            "bundle_root": str(trust.bundle_root),
            "trust_manifest_sha256": trust.manifest_snapshot.sha256,
            "verification_receipt_sha256": trust.receipt_snapshot.sha256,
            "artifacts": _core_trust_artifacts(trust),
        },
        "determinism": {
            "seed": contract.embedding_seed,
            "batch_size": contract.embedding_batch_size,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
        },
        "runtime": dict(cast(Mapping[str, object], contract.runtime)),
        "outputs": {
            "embeddings.npy": matrix.snapshot.sha256,
            "embedding_index.csv": index.sha256,
        },
        "worker_sha256": contract.embedding_worker_sha256,
    }


def _git(repository: Path, arguments: Sequence[str], *, label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise VerificationError(f"Git check failed for {label}") from error


def _committed_blob(repository: Path, commit: str, logical_path: str) -> bytes:
    kind = _git(repository, ["cat-file", "-t", f"{commit}:{logical_path}"], label=logical_path)
    _require(kind == b"blob\n", f"committed path is not a blob: {logical_path}")
    return _git(repository, ["cat-file", "blob", f"{commit}:{logical_path}"], label=logical_path)


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


def _expected_code_paths(repository: Path) -> frozenset[str]:
    source_root = repository / "src/amp_challenge"
    _require(
        source_root.is_dir() and not source_root.is_symlink(), "package source tree is missing"
    )
    paths = set(_CODE_FIXED_PATHS)
    for source in source_root.rglob("*.py"):
        _require(not source.is_symlink(), "package source tree contains a symbolic link")
        if source.is_file():
            paths.add(source.relative_to(repository).as_posix())
    return frozenset(paths)


def _validate_code_attestation(
    *,
    repository: Path,
    expected_commit: str,
    manifest: Snapshot,
    contract: Contract,
) -> tuple[dict[str, str], tuple[Snapshot, ...]]:
    _verify_repository(repository, expected_commit)
    executing = _resolve_without_symlinks(Path(__file__), label="executing verifier")
    expected_executing = _resolve_without_symlinks(
        repository / _VERIFIER_PATH,
        label="repository verifier",
    )
    _require(executing == expected_executing, "executing verifier is a stale installation")
    _require(manifest.path.stat().st_mode & 0o222 == 0, "producer code manifest remains writable")
    entries = _parse_sha_manifest(manifest.payload, label="producer code manifest")
    _require(
        set(entries) == set(_expected_code_paths(repository)), "producer code inventory changed"
    )
    snapshots = [manifest]
    for logical, digest in entries.items():
        relative = _safe_relative_path(logical, label="code manifest path")
        snapshot = _read_snapshot(
            repository.joinpath(*PurePosixPath(relative).parts),
            label=f"code input {logical}",
        )
        _require(snapshot.sha256 == digest, f"producer code hash changed: {logical}")
        _require(
            snapshot.payload == _committed_blob(repository, expected_commit, logical),
            f"code input differs from committed blob: {logical}",
        )
        snapshots.append(snapshot)
    expected_links = {
        _CONFIG_PATH: contract.snapshot.sha256,
        _LEGACY_WORKER_PATH: cast(str, contract.embedding_worker_sha256),
        _TRUST_MANIFEST_PATH: cast(str, contract.trust_manifest_sha256),
    }
    for logical, digest in expected_links.items():
        _require(entries.get(logical) == digest, f"code manifest link changed: {logical}")
    for logical in (_PACKAGE_PATH, _VERIFIER_PATH, _REFERENCE_WORKER_PATH):
        _require(logical in entries, f"code manifest omits {logical}")
        _require(
            entries[logical] == _sha256(_committed_blob(repository, expected_commit, logical)),
            f"code manifest does not bind committed {logical}",
        )
    return entries, tuple(snapshots)


def _expected_frozen_input_hashes(
    *, panel: PanelEvidence, trust: TrustEvidence, worker_input_sha256: str
) -> dict[str, str]:
    expected = dict(panel.relative_hashes)
    expected.update(
        {
            "artifacts.toml": trust.manifest_snapshot.sha256,
            "panel-independent-receipt.json": panel.receipt_snapshot.sha256,
            "trust-receipt.json": trust.receipt_snapshot.sha256,
            "worker-input-manifest.json": worker_input_sha256,
        }
    )
    expected.update({f"bundle/{logical}": spec.sha256 for logical, spec in trust.artifacts.items()})
    return expected


def _validate_frozen_manifest(
    snapshot: Snapshot,
    *,
    panel: PanelEvidence,
    trust: TrustEvidence,
    worker_input_sha256: str,
) -> dict[str, str]:
    _require(snapshot.path.stat().st_mode & 0o222 == 0, "producer frozen manifest remains writable")
    entries = _parse_sha_manifest(snapshot.payload, label="producer frozen input manifest")
    expected = _expected_frozen_input_hashes(
        panel=panel,
        trust=trust,
        worker_input_sha256=worker_input_sha256,
    )
    _require(
        entries == expected, "producer frozen input manifest was not independently reconstructed"
    )
    _require(
        snapshot.payload == _sha_manifest_bytes(expected), "producer frozen manifest bytes changed"
    )
    return entries


def _validate_driver(value: object) -> str:
    _require(
        isinstance(value, str)
        and value == value.strip()
        and _DRIVER_RE.fullmatch(value) is not None,
        "CUDA driver version is invalid",
    )
    return cast(str, value)


def _validate_producer_twins(root_path: str | Path) -> ProducerEvidence:
    root = _resolved_directory(root_path, label="producer embedding twin root")
    files, directories = _tree_inventory(root)
    expected_files = {f"{task}/{name}" for task in (0, 1) for name in _PRODUCER_FILES} | {
        f"node-receipts/{name}" for name in _HANDSHAKE_FILES
    }
    expected_directories = {
        "0",
        "0/embeddings",
        "1",
        "1/embeddings",
        "node-receipts",
    }
    _require(
        files == expected_files and directories == expected_directories,
        "producer job-root inventory changed",
    )
    handshake, handshake_snapshots = _verify_handshake(root, label="producer embedding")
    top_records: list[tuple[Snapshot, dict[str, str], Snapshot, dict[str, str]]] = []
    run_snapshots: list[Snapshot] = []
    for task in (0, 1):
        run = root / str(task)
        top, entries, snapshots = _verify_manifest_tree(
            run,
            expected_inventory=_PRODUCER_FILES,
            expected_top_sha256=None,
            label=f"producer embedding twin {task}",
        )
        semantic_top, semantic_entries, semantic_snapshots = _verify_manifest_tree(
            run / "embeddings",
            expected_inventory=_EMBEDDING_FILES,
            expected_top_sha256=None,
            label=f"producer semantic twin {task}",
        )
        _require(
            entries["embeddings/SHA256SUMS"] == semantic_top.sha256,
            "producer semantic top is unbound",
        )
        _verify_immutable_tree(run, label=f"producer embedding twin {task}")
        run_snapshots.extend((*snapshots, *semantic_snapshots))
        top_records.append((top, entries, semantic_top, semantic_entries))
    _verify_tree_bytes(
        root / "0", root / "1", inventory=_PRODUCER_FILES, label="producer embedding"
    )
    _require(
        top_records[0][1] == top_records[1][1] and top_records[0][3] == top_records[1][3],
        "producer twin manifests differ",
    )
    run = root / "0"
    matrix_snapshot = _read_snapshot(run / "embeddings/embeddings.npy", label="producer matrix")
    _require(len(matrix_snapshot.payload) == _EXPECTED_NPY_BYTES, "producer NPY size changed")
    matrix = _validate_matrix_snapshot(
        matrix_snapshot,
        expected_rows=_EXPECTED_ROWS,
        expected_columns=_EXPECTED_COLUMNS,
    )
    index = _read_snapshot(run / "embeddings/embedding_index.csv", label="producer index")
    manifest = _read_snapshot(
        run / "embeddings/embedding_manifest.json", label="producer semantic manifest"
    )
    document = _read_pretty_json(manifest, label="producer semantic manifest")
    _assert_path_free(document, label="producer semantic manifest")
    code_manifest = _read_snapshot(run / "CODE_SHA256SUMS", label="producer code manifest")
    frozen_manifest = _read_snapshot(
        run / "FROZEN_INPUT_SHA256SUMS", label="producer frozen manifest"
    )
    snapshots = tuple(
        {
            snapshot.path: snapshot
            for snapshot in (
                *handshake_snapshots,
                *run_snapshots,
                matrix_snapshot,
                index,
                manifest,
                code_manifest,
                frozen_manifest,
            )
        }.values()
    )
    return ProducerEvidence(
        root=root,
        handshake=handshake,
        top_snapshot=top_records[0][0],
        top_entries=top_records[0][1],
        semantic_top_snapshot=top_records[0][2],
        semantic_entries=top_records[0][3],
        code_manifest=code_manifest,
        frozen_manifest=frozen_manifest,
        matrix=matrix,
        index=index,
        semantic_manifest=manifest,
        semantic_document=document,
        snapshots=snapshots,
    )


def _semantic_manifest(
    *,
    contract: Contract,
    panel: PanelEvidence,
    trust: TrustEvidence,
    matrix: MatrixEvidence,
    index: Snapshot,
    raw_manifest_sha256: str,
    worker_input_sha256: str,
    code_entries: Mapping[str, str],
    code_manifest_sha256: str,
    expected_git_commit: str,
    cuda_driver_version: str,
) -> dict[str, object]:
    runtime = dict(cast(Mapping[str, object], contract.runtime))
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
            "shape": [_EXPECTED_ROWS, _EXPECTED_COLUMNS],
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
            "records": _EXPECTED_ROWS,
            "ordering": "accepted panel ascending sequence_id",
        },
        "runtime": runtime,
        "determinism": {
            "seed": contract.embedding_seed,
            "batch_size": contract.embedding_batch_size,
            **dict(cast(Mapping[str, object], contract.determinism)),
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
            "records": _EXPECTED_ROWS,
            "acceptance_status": "accepted_for_embedding_input_only",
        },
        "trust": _semantic_trust(trust),
        "input_attestation": {
            "config_sha256": contract.snapshot.sha256,
            "code_manifest_sha256": code_manifest_sha256,
            "private_worker_input_manifest_sha256": worker_input_sha256,
            "raw_worker_output_manifest_sha256": raw_manifest_sha256,
        },
        "code_attestation": {
            "git_commit": expected_git_commit,
            "package_module": {
                "logical_path": _PACKAGE_PATH,
                "sha256": code_entries[_PACKAGE_PATH],
            },
            "independent_verifier": {
                "logical_path": _VERIFIER_PATH,
                "sha256": code_entries[_VERIFIER_PATH],
                "included_in_producer_code_manifest": True,
            },
            "legacy_worker": {
                "logical_path": _LEGACY_WORKER_PATH,
                "sha256": code_entries[_LEGACY_WORKER_PATH],
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
                "sha256": index.sha256,
                "size": len(index.payload),
                "records": _EXPECTED_ROWS,
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
    _assert_path_free(value, label="reconstructed producer manifest")
    return value


def _reference_manifest(
    *,
    contract: Contract,
    panel: PanelEvidence,
    trust: TrustEvidence,
    matrix: MatrixEvidence,
    reference_worker_sha256: str,
    cuda_driver_version: str,
) -> dict[str, object]:
    runtime = dict(_EXPECTED_RUNTIME)
    runtime["cuda_driver_version"] = cuda_driver_version
    return {
        "schema_version": 1,
        "artifact": "gate1_union_esm2_t6_layer6_mean_private_reference_v1",
        "status": "private_reference_recomputation_not_an_acceptance_receipt",
        "input": {
            "fasta_sha256": contract.panel_fasta_sha256,
            "records": _EXPECTED_ROWS,
            "sequence_ids_sha256": contract.expected_sequence_ids_sha256,
            "ordering": "ascending sequence_id",
        },
        "model": {
            "name": contract.embedding_model,
            "checkpoint_sha256": contract.model_checkpoint_sha256,
            "contact_regression_sha256": contract.contact_regression_sha256,
            "representation_layer": contract.representation_layer,
            "embedding_dimension": contract.embedding_dimension,
            "pooling": "arithmetic mean of residues 1:L+1; BOS/EOS/padding excluded",
        },
        "trust": _reference_trust(trust),
        "determinism": {
            "seed": contract.embedding_seed,
            "batch_size": contract.embedding_batch_size,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        },
        "runtime": runtime,
        "reference_worker_sha256": reference_worker_sha256,
        "acceptance_scope": {
            "private_audit_material": True,
            "production_embeddings_accepted": False,
            "downstream_predictions_verified": False,
            "model_performance_evidence": False,
            "ensemble_weight_authorized": False,
        },
        "output": {
            "filename": "reference_embeddings.npy",
            "shape": [_EXPECTED_ROWS, _EXPECTED_COLUMNS],
            "dtype": "<f4",
            "order": "C",
            "npy_format_version": [1, 0],
            "npy_sha256": matrix.snapshot.sha256,
            "tensor_data_sha256": matrix.tensor_data_sha256,
            "tensor_data_encoding": ("row-major contiguous little-endian IEEE-754 float32 bytes"),
        },
    }


def _validate_reference(
    root_path: str | Path,
    *,
    contract: Contract,
    panel: PanelEvidence,
    trust: TrustEvidence,
    code_entries: Mapping[str, str],
) -> ReferenceEvidence:
    root = _resolved_directory(root_path, label="private reference root")
    files, directories = _tree_inventory(root)
    _require(files == _REFERENCE_FILES and not directories, "private reference inventory changed")
    _verify_immutable_tree(root, label="private reference")
    matrix_snapshot = _read_snapshot(root / "reference_embeddings.npy", label="reference matrix")
    _require(len(matrix_snapshot.payload) == _EXPECTED_NPY_BYTES, "reference NPY size changed")
    matrix = _validate_matrix_snapshot(
        matrix_snapshot,
        expected_rows=_EXPECTED_ROWS,
        expected_columns=_EXPECTED_COLUMNS,
    )
    manifest = _read_snapshot(root / "reference_manifest.json", label="reference manifest")
    document = _read_pretty_json(manifest, label="reference manifest")
    _assert_path_free(document, label="reference manifest")
    runtime = _mapping(document.get("runtime"), label="reference runtime")
    driver = _validate_driver(runtime.get("cuda_driver_version"))
    reference_worker_sha256 = code_entries[_REFERENCE_WORKER_PATH]
    expected = _reference_manifest(
        contract=contract,
        panel=panel,
        trust=trust,
        matrix=matrix,
        reference_worker_sha256=reference_worker_sha256,
        cuda_driver_version=driver,
    )
    _require(document == expected, "reference manifest was not independently reconstructed")
    _require(manifest.payload == _pretty_json_bytes(expected), "reference manifest bytes changed")
    return ReferenceEvidence(
        root=root,
        matrix=matrix,
        manifest=manifest,
        document=document,
        runtime=runtime,
        snapshots=(matrix_snapshot, manifest),
    )


def _require_exact_numerical_agreement(
    producer: MatrixEvidence,
    reference: MatrixEvidence,
    *,
    producer_runtime: Mapping[str, object],
    reference_runtime: Mapping[str, object],
) -> None:
    _require(producer_runtime == reference_runtime, "producer/reference runtime or driver mismatch")
    _require(
        producer.snapshot.payload == reference.snapshot.payload,
        "producer/reference NPY bytes differ",
    )
    _require(
        producer.tensor_payload == reference.tensor_payload,
        "producer/reference tensor bytes differ",
    )
    _require(
        producer.snapshot.sha256 == reference.snapshot.sha256
        and producer.tensor_data_sha256 == reference.tensor_data_sha256,
        "producer/reference numerical hashes differ",
    )


def _reconstruct_and_validate_producer(
    *,
    producer: ProducerEvidence,
    contract: Contract,
    panel: PanelEvidence,
    trust: TrustEvidence,
    code_entries: Mapping[str, str],
    expected_git_commit: str,
    cuda_driver_version: str,
) -> tuple[str, str]:
    expected_index = _canonical_index_bytes(panel.records)
    _require(
        producer.index.payload == expected_index, "producer embedding index differs from panel"
    )
    worker_input_payload = _pretty_json_bytes(
        _worker_input_manifest(panel=panel, contract=contract)
    )
    worker_input_sha256 = _sha256(worker_input_payload)
    raw_manifest_payload = _pretty_json_bytes(
        _raw_worker_manifest(
            contract=contract,
            matrix=producer.matrix,
            index=producer.index,
            worker_input_sha256=worker_input_sha256,
            trust=trust,
        )
    )
    raw_manifest_sha256 = _sha256(raw_manifest_payload)
    expected_manifest = _semantic_manifest(
        contract=contract,
        panel=panel,
        trust=trust,
        matrix=producer.matrix,
        index=producer.index,
        raw_manifest_sha256=raw_manifest_sha256,
        worker_input_sha256=worker_input_sha256,
        code_entries=code_entries,
        code_manifest_sha256=producer.code_manifest.sha256,
        expected_git_commit=expected_git_commit,
        cuda_driver_version=cuda_driver_version,
    )
    expected_manifest_payload = _pretty_json_bytes(expected_manifest)
    _require(
        producer.semantic_document == expected_manifest
        and producer.semantic_manifest.payload == expected_manifest_payload,
        "producer semantic manifest was not independently reconstructed",
    )
    expected_semantic_entries = {
        "embedding_index.csv": producer.index.sha256,
        "embedding_manifest.json": _sha256(expected_manifest_payload),
        "embeddings.npy": producer.matrix.snapshot.sha256,
    }
    expected_semantic_top = _sha_manifest_bytes(expected_semantic_entries)
    _require(
        producer.semantic_entries == expected_semantic_entries
        and producer.semantic_top_snapshot.payload == expected_semantic_top,
        "producer semantic checksum manifest was not independently reconstructed",
    )
    expected_top_entries = {
        "CODE_SHA256SUMS": producer.code_manifest.sha256,
        "FROZEN_INPUT_SHA256SUMS": producer.frozen_manifest.sha256,
        "embeddings/SHA256SUMS": producer.semantic_top_snapshot.sha256,
        "embeddings/embedding_index.csv": producer.index.sha256,
        "embeddings/embedding_manifest.json": producer.semantic_manifest.sha256,
        "embeddings/embeddings.npy": producer.matrix.snapshot.sha256,
    }
    _require(
        producer.top_entries == expected_top_entries
        and producer.top_snapshot.payload == _sha_manifest_bytes(expected_top_entries),
        "producer publication checksum manifest was not independently reconstructed",
    )
    return worker_input_sha256, raw_manifest_sha256


def _assert_snapshots_unchanged(snapshots: Iterable[Snapshot], *, label: str) -> None:
    seen: set[Path] = set()
    for snapshot in snapshots:
        if snapshot.path in seen:
            continue
        seen.add(snapshot.path)
        _assert_unchanged(snapshot, label=f"{label} {snapshot.path.name}")


def _assert_evidence_trees_stable(
    *, producer: ProducerEvidence, panel: PanelEvidence, reference: ReferenceEvidence
) -> None:
    producer_files, producer_directories = _tree_inventory(producer.root)
    expected_producer_files = {f"{task}/{name}" for task in (0, 1) for name in _PRODUCER_FILES} | {
        f"node-receipts/{name}" for name in _HANDSHAKE_FILES
    }
    _require(
        producer_files == expected_producer_files
        and producer_directories == {"0", "0/embeddings", "1", "1/embeddings", "node-receipts"},
        "producer evidence inventory changed during audit",
    )
    panel_files, panel_directories = _tree_inventory(panel.root)
    expected_panel_files = {
        f"{task}/{name}" for task in (0, 1) for name in _PANEL_PUBLICATION_FILES
    } | {f"node-receipts/{name}" for name in _HANDSHAKE_FILES}
    _require(
        panel_files == expected_panel_files
        and panel_directories == {"0", "0/panel", "1", "1/panel", "node-receipts"},
        "panel evidence inventory changed during audit",
    )
    reference_files, reference_directories = _tree_inventory(reference.root)
    _require(
        reference_files == _REFERENCE_FILES and not reference_directories,
        "reference evidence inventory changed during audit",
    )
    for task in (0, 1):
        _verify_immutable_tree(producer.root / str(task), label=f"producer twin {task}")
        _verify_immutable_tree(panel.root / str(task), label=f"panel twin {task}")
    _verify_immutable_tree(producer.root / "node-receipts", label="producer handshake")
    _verify_immutable_tree(panel.root / "node-receipts", label="panel handshake")
    _verify_immutable_tree(reference.root, label="private reference")


def verify_union_esm_embedding_twins(
    *,
    producer_twin_root: str | Path,
    panel_twin_root: str | Path,
    panel_independent_receipt: str | Path,
    reference_root: str | Path,
    trust_receipt: str | Path,
    trust_manifest: str | Path,
    bundle_root: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, object]:
    """Return a path-free, narrow acceptance receipt after a full exact audit."""

    _require(sys.byteorder == "little", "embedding verification requires a little-endian host")
    repository = _resolved_directory(repo_root, label="repository root")
    _require(_GIT_RE.fullmatch(expected_git_commit) is not None, "expected Git commit is invalid")
    config_file = _resolve_without_symlinks(config_path, label="embedding config")
    expected_config = _resolve_without_symlinks(
        repository / _CONFIG_PATH,
        label="repository embedding config",
    )
    _require(
        config_file == expected_config,
        "verifier received the wrong logical config",
    )
    trust_file = _resolve_without_symlinks(
        trust_manifest,
        label="AMP-Diffusion trust manifest",
    )
    expected_trust = _resolve_without_symlinks(
        repository / _TRUST_MANIFEST_PATH,
        label="repository AMP-Diffusion trust manifest",
    )
    _require(
        trust_file == expected_trust,
        "verifier received the wrong logical trust manifest",
    )
    config_snapshot = _read_snapshot(config_file, label="embedding config")
    contract = _parse_config(config_snapshot)
    _verify_repository(repository, expected_git_commit)

    panel = _validate_panel_evidence(
        twin_root=panel_twin_root,
        independent_receipt=panel_independent_receipt,
        contract=contract,
    )
    trust = _validate_trust_evidence(
        trust_receipt=trust_receipt,
        trust_manifest=trust_file,
        bundle_root=bundle_root,
        contract=contract,
    )
    producer = _validate_producer_twins(producer_twin_root)
    code_entries, code_snapshots = _validate_code_attestation(
        repository=repository,
        expected_commit=expected_git_commit,
        manifest=producer.code_manifest,
        contract=contract,
    )
    worker_input_sha256 = _sha256(
        _pretty_json_bytes(_worker_input_manifest(panel=panel, contract=contract))
    )
    _validate_frozen_manifest(
        producer.frozen_manifest,
        panel=panel,
        trust=trust,
        worker_input_sha256=worker_input_sha256,
    )
    reference = _validate_reference(
        reference_root,
        contract=contract,
        panel=panel,
        trust=trust,
        code_entries=code_entries,
    )
    producer_runtime = _mapping(producer.semantic_document.get("runtime"), label="producer runtime")
    reference_driver = _validate_driver(reference.runtime.get("cuda_driver_version"))
    _reconstructed_worker_input, raw_manifest_sha256 = _reconstruct_and_validate_producer(
        producer=producer,
        contract=contract,
        panel=panel,
        trust=trust,
        code_entries=code_entries,
        expected_git_commit=expected_git_commit,
        cuda_driver_version=reference_driver,
    )
    _require(
        _reconstructed_worker_input == worker_input_sha256, "worker input reconstruction changed"
    )
    _require_exact_numerical_agreement(
        producer.matrix,
        reference.matrix,
        producer_runtime=producer_runtime,
        reference_runtime=reference.runtime,
    )

    forbidden = list(forbidden_prefixes)
    panel_publication_text = [
        _read_snapshot(panel.root / "0" / name, label=f"panel path scan {name}").payload
        for name in sorted(_PANEL_PUBLICATION_FILES)
    ]
    _scan_path_free(
        [
            *panel_publication_text,
            producer.code_manifest.payload,
            producer.frozen_manifest.payload,
            producer.index.payload,
            producer.semantic_manifest.payload,
            producer.semantic_top_snapshot.payload,
            producer.top_snapshot.payload,
            reference.manifest.payload,
        ],
        forbidden,
    )
    source_snapshots = (
        config_snapshot,
        *panel.snapshots,
        trust.receipt_snapshot,
        trust.manifest_snapshot,
        *trust.artifact_snapshots,
        *producer.snapshots,
        *reference.snapshots,
        *code_snapshots,
    )
    _assert_snapshots_unchanged(source_snapshots, label="post-audit input")
    _assert_evidence_trees_stable(producer=producer, panel=panel, reference=reference)
    _verify_repository(repository, expected_git_commit)
    _verify_tree_bytes(
        producer.root / "0",
        producer.root / "1",
        inventory=_PRODUCER_FILES,
        label="producer embedding",
    )
    _verify_tree_bytes(
        panel.root / "0",
        panel.root / "1",
        inventory=_PANEL_PUBLICATION_FILES,
        label="accepted panel",
    )

    verifier_sha256 = code_entries[_VERIFIER_PATH]
    reference_worker_sha256 = code_entries[_REFERENCE_WORKER_PATH]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact": "gate1_union_esm2_embeddings_v1_independent_verification",
        "status": "accepted_for_downstream_embedding_feature_input_only",
        "acceptance_scope": dict(_ACCEPTED_SCOPE),
        "checks": {name: True for name in sorted(_AUDIT_CHECKS)},
        "git_commit": expected_git_commit,
        "config_sha256": contract.snapshot.sha256,
        "producer": {
            "publication_top_manifest_sha256": producer.top_snapshot.sha256,
            "embedding_top_manifest_sha256": producer.semantic_top_snapshot.sha256,
            "code_manifest_sha256": producer.code_manifest.sha256,
            "frozen_input_manifest_sha256": producer.frozen_manifest.sha256,
            "production_handshake": dict(producer.handshake),
            "candidate_manifest_sha256": producer.semantic_manifest.sha256,
            "reconstructed_raw_worker_manifest_sha256": raw_manifest_sha256,
            "reconstructed_private_worker_input_manifest_sha256": worker_input_sha256,
        },
        "panel": {
            "git_commit": contract.panel_git_commit,
            "publication_top_manifest_sha256": contract.panel_publication_top_sha256,
            "panel_top_manifest_sha256": contract.panel_top_sha256,
            "fasta_sha256": contract.panel_fasta_sha256,
            "manifest_sha256": contract.panel_manifest_sha256,
            "coverage_receipt_sha256": contract.panel_coverage_receipt_sha256,
            "independent_receipt_sha256": panel.receipt_snapshot.sha256,
            "sequence_ids_sha256": contract.expected_sequence_ids_sha256,
            "records": _EXPECTED_ROWS,
            "production_handshake": dict(panel.handshake),
        },
        "tensor": {
            "shape": [_EXPECTED_ROWS, _EXPECTED_COLUMNS],
            "dtype": "float32",
            "byte_order": "little_endian",
            "layout": "C_contiguous",
            "container": "NPY",
            "npy_version": [1, 0],
            "npy_sha256": producer.matrix.snapshot.sha256,
            "tensor_data_bytes": len(producer.matrix.tensor_payload),
            "tensor_data_sha256": producer.matrix.tensor_data_sha256,
            "comparison_policy": "exact_npy_and_tensor_bytes_on_matching_runtime_gpu_and_driver",
        },
        "runtime": dict(reference.runtime),
        "trust": {
            **_reference_trust(trust),
            "all_generation_artifacts_rehashed_before_and_after": True,
        },
        "artifact_sha256": {
            "SHA256SUMS": producer.top_snapshot.sha256,
            "embeddings/SHA256SUMS": producer.semantic_top_snapshot.sha256,
            "embeddings/embedding_index.csv": producer.index.sha256,
            "embeddings/embedding_manifest.json": producer.semantic_manifest.sha256,
            "embeddings/embeddings.npy": producer.matrix.snapshot.sha256,
            "private_reference/reference_embeddings.npy": reference.matrix.snapshot.sha256,
            "private_reference/reference_manifest.json": reference.manifest.sha256,
        },
        "verifier_attestation": {
            "logical_path": _VERIFIER_PATH,
            "sha256": verifier_sha256,
            "git_commit": expected_git_commit,
            "included_in_producer_code_manifest": True,
            "executing_source_matches_committed_blob": True,
        },
        "reference_worker_attestation": {
            "logical_path": _REFERENCE_WORKER_PATH,
            "sha256": reference_worker_sha256,
            "included_in_producer_code_manifest": True,
            "independent_of_legacy_worker_implementation": True,
            "full_inference_executed": True,
        },
        "limitations": [
            "accepts_only_the_hash_bound_952_by_320_embedding_matrix_as_feature_input",
            "does_not_verify_any_model_prediction_or_performance_claim",
            "does_not_establish_an_ensemble_weight",
            "does_not_establish_ampdiffusion_pretraining_membership_independence",
            "exact_bytes_are_required_only_under_the_pinned_matching_a100_runtime_and_driver_contract",
            "surrounding_audit_publication_immutability_is_enforced_by_the_slurm_launcher",
        ],
    }
    _assert_path_free(receipt, label="independent embedding receipt")
    payload = _pretty_json_bytes(receipt)
    _scan_path_free([payload], forbidden)
    return receipt


def _write_receipt(
    path: str | Path,
    receipt: Mapping[str, object],
    *,
    protected_roots: Sequence[Path],
) -> Path:
    requested = Path(path)
    _require(requested.name not in {"", ".", ".."}, "verification receipt has no filename")
    _reject_symlink_chain(requested.parent, label="receipt parent")
    _require(not os.path.lexists(requested), "refusing to overwrite verification receipt")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = _resolve_without_symlinks(requested.parent, label="receipt parent")
    _require(parent.is_dir() and not parent.is_symlink(), "receipt parent is unsafe")
    target = parent / requested.name
    _require(not os.path.lexists(target), "refusing to overwrite verification receipt")
    for root in protected_roots:
        _require(
            target != root and not target.is_relative_to(root),
            "receipt must be outside every verified input tree",
        )
    payload = _pretty_json_bytes(receipt)
    _assert_path_free(receipt, label="independent embedding receipt")
    _scan_path_free([payload], ())
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
        return target
    finally:
        if staging.exists():
            staging.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-twin-root", type=Path, required=True)
    parser.add_argument("--panel-twin-root", type=Path, required=True)
    parser.add_argument("--panel-independent-receipt", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--trust-receipt", type=Path, required=True)
    parser.add_argument("--trust-manifest", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--forbidden-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protected = [
        _resolved_directory(args.producer_twin_root, label="producer embedding twin root"),
        _resolved_directory(args.panel_twin_root, label="accepted panel twin root"),
        _resolved_directory(args.reference_root, label="private reference root"),
        _resolved_directory(args.bundle_root, label="AMP-Diffusion bundle root"),
        _resolved_directory(args.repo_root, label="repository root"),
    ]
    prospective = Path(args.output).resolve(strict=False)
    for root in protected:
        _require(
            prospective != root and not prospective.is_relative_to(root),
            "receipt must be outside every verified input tree",
        )
    receipt = verify_union_esm_embedding_twins(
        producer_twin_root=args.producer_twin_root,
        panel_twin_root=args.panel_twin_root,
        panel_independent_receipt=args.panel_independent_receipt,
        reference_root=args.reference_root,
        trust_receipt=args.trust_receipt,
        trust_manifest=args.trust_manifest,
        bundle_root=args.bundle_root,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    _write_receipt(args.output, receipt, protected_roots=protected)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the Slurm CLI
    raise SystemExit(main())

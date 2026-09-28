"""Immutable semantic bundles for the native-diffusion v1 pilot.

This module is deliberately neutral: it contains no trainer, evaluator, gate,
or Slurm logic.  It provides the shared byte boundary those separate
processes use to publish and reopen exact contract-shaped bundles.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
)

_BUNDLE_KINDS = frozenset({"trainer", "evaluator", "pilot"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_MAX_SEMANTIC_TEXT_BYTES = 64 * 1024 * 1024
_MAX_BUNDLE_ARTIFACT_BYTES = 1 << 30
_GIT_REPOSITORY_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Path-free identity of one completely hashed, sealed bundle file."""

    relative_path: str
    size: int
    sha256: str
    mode: str
    link_count: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, label="file snapshot path")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("file snapshot size must be a nonnegative integer")
        _sha256(self.sha256, label="file snapshot digest")
        if self.mode != "0444" or self.link_count != 1:
            raise ValueError("file snapshot is not an immutable single-link file")
        for field in (self.device, self.inode, self.mtime_ns, self.ctime_ns):
            if type(field) is not int or field < 0:
                raise ValueError("file snapshot identity fields must be nonnegative integers")

    @property
    def fingerprint(self) -> tuple[int, int, int, int, int, int, int]:
        """Return the lstat fields used for bounded reauthentication."""

        return (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
            0o444,
            self.link_count,
        )


@dataclass(frozen=True, slots=True)
class BundleSnapshot:
    """Complete immutable tree identity returned after bundle verification."""

    bundle_kind: str
    root: Path
    tree_bytes: bytes
    tree_sha256: str
    file_snapshots: tuple[FileSnapshot, ...]

    def __post_init__(self) -> None:
        _bundle_kind(self.bundle_kind)
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("bundle snapshot root must be an absolute Path")
        if type(self.tree_bytes) is not bytes or not self.tree_bytes:
            raise ValueError("bundle tree bytes must be non-empty exact bytes")
        _sha256(self.tree_sha256, label="bundle tree digest")
        if hashlib.sha256(self.tree_bytes).hexdigest() != self.tree_sha256:
            raise ValueError("bundle tree bytes and digest differ")
        if type(self.file_snapshots) is not tuple or any(
            type(item) is not FileSnapshot for item in self.file_snapshots
        ):
            raise TypeError("bundle file snapshots must be an exact tuple of FileSnapshot values")
        paths = tuple(item.relative_path for item in self.file_snapshots)
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("bundle file snapshots are empty or duplicated")

    def file(self, relative_path: str) -> FileSnapshot:
        """Return the unique sealed identity for a contract inventory path."""

        path = _relative_path(relative_path, label="bundle file path")
        for item in self.file_snapshots:
            if item.relative_path == path:
                return item
        raise KeyError(path)

    def read_bytes(self, relative_path: str, *, maximum_bytes: int | None = None) -> bytes:
        """Read bytes only if they still match this snapshot's inode and digest."""

        item = self.file(relative_path)
        limit = item.size if maximum_bytes is None else maximum_bytes
        if type(limit) is not int or limit < 0:
            raise ValueError("maximum_bytes must be a nonnegative exact integer or None")
        if item.size > limit:
            raise ValueError(f"sealed artifact exceeds read limit: {item.relative_path}")
        path = self.root.joinpath(*PurePosixPath(item.relative_path).parts)
        payload, observed, digest = _read_sealed_file(path, maximum_bytes=limit)
        if observed != item.fingerprint or digest != item.sha256:
            raise ValueError(
                f"sealed artifact changed after bundle verification: {item.relative_path}"
            )
        return payload


@dataclass(frozen=True, slots=True)
class RepositoryEntry:
    """One tracked regular blob included in ``CODE_SHA256SUMS``."""

    relative_path: str
    git_mode: str
    git_object: str
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, label="repository entry path")
        if self.git_mode not in {"100644", "100755"}:
            raise ValueError("repository entry has a non-regular Git mode")
        if type(self.git_object) is not str or _GIT_COMMIT_RE.fullmatch(self.git_object) is None:
            raise ValueError("repository entry has an invalid blob object ID")
        _sha256(self.sha256, label="repository entry digest")


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    """Clean synchronized Git identity and its exact tracked-code manifest."""

    git_commit: str
    code_sha256sums: bytes
    code_sha256: str
    entries: tuple[RepositoryEntry, ...]

    def __post_init__(self) -> None:
        if type(self.git_commit) is not str or _GIT_COMMIT_RE.fullmatch(self.git_commit) is None:
            raise ValueError("repository snapshot has an invalid Git commit")
        if hashlib.sha256(self.code_sha256sums).hexdigest() != self.code_sha256:
            raise ValueError("repository code manifest digest differs from its bytes")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("repository snapshot entries must be a non-empty tuple")
        if any(type(item) is not RepositoryEntry for item in self.entries):
            raise TypeError("repository snapshot entries must be exact RepositoryEntry values")
        paths = tuple(item.relative_path for item in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("repository snapshot entries must be strictly path-ordered")
        observed = parse_sha256sums(self.code_sha256sums, label="CODE_SHA256SUMS")
        if observed != {item.relative_path: item.sha256 for item in self.entries}:
            raise ValueError("repository entries differ from CODE_SHA256SUMS")


@dataclass(frozen=True, slots=True)
class _DirectorySnapshot:
    relative_path: str
    device: int
    inode: int
    os_size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    link_count: int

    @property
    def fingerprint(self) -> tuple[int, int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.os_size,
            self.mtime_ns,
            self.ctime_ns,
            self.mode,
            self.link_count,
        )


def _plain_json(value: object, *, label: str) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key or key in result:
                raise ValueError(f"{label} contains an invalid or duplicate object key")
            result[key] = _plain_json(item, label=f"{label}.{key}")
        return result
    if type(value) in {list, tuple}:
        return [_plain_json(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{label} contains a non-JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite JSON as sorted compact UTF-8 with exactly one final LF."""

    plain = _plain_json(value, label="canonical JSON")
    try:
        return (
            json.dumps(
                plain,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise ValueError("value cannot be encoded as canonical finite UTF-8 JSON") from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_canonical_json(payload: bytes, *, label: str = "JSON") -> object:
    """Parse only the exact canonical encoding emitted by :func:`canonical_json_bytes`."""

    if (
        type(payload) is not bytes
        or not payload
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or b"\r" in payload
    ):
        raise ValueError(f"{label} must be non-empty single-LF canonical JSON bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid finite UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not in the canonical byte encoding")
    return value


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    """Encode one or more canonical JSON objects, one per LF-terminated line."""

    if type(rows) not in {list, tuple} or not rows:
        raise ValueError("canonical JSONL rows must be a non-empty list or tuple")
    payloads: list[bytes] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"canonical JSONL row {index} must be an object")
        payloads.append(canonical_json_bytes(row))
    return b"".join(payloads)


def parse_canonical_jsonl(
    payload: bytes,
    *,
    label: str = "JSONL",
) -> tuple[dict[str, object], ...]:
    """Parse non-empty JSONL whose every line is independently canonical."""

    if (
        type(payload) is not bytes
        or not payload
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or b"\r" in payload
    ):
        raise ValueError(f"{label} must be non-empty LF-terminated canonical JSONL")
    rows: list[dict[str, object]] = []
    for number, raw in enumerate(payload[:-1].split(b"\n"), start=1):
        value = parse_canonical_json(raw + b"\n", label=f"{label} line {number}")
        if type(value) is not dict:
            raise ValueError(f"{label} line {number} must be an exact JSON object")
        rows.append(value)
    return tuple(rows)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: object, *, label: str, allow_dot: bool = False) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    if allow_dot and value == ".":
        return value
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def parse_sha256sums(payload: bytes, *, label: str = "SHA256SUMS") -> dict[str, str]:
    """Parse a non-empty, strictly C/path-sorted two-space checksum manifest."""

    if (
        type(payload) is not bytes
        or not payload
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or b"\r" in payload
    ):
        raise ValueError(f"{label} must be non-empty LF-terminated bytes")
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be ASCII") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"{label} line {number} is malformed")
        digest = _sha256(line[:64], label=f"{label} line {number} digest")
        path = _relative_path(line[66:], label=f"{label} line {number} path")
        if previous is not None and path <= previous:
            raise ValueError(f"{label} paths must be strictly ordered and unique")
        result[path] = digest
        previous = path
    if sha256sums_bytes(result) != payload:
        raise ValueError(f"{label} is not in the canonical checksum encoding")
    return result


def sha256sums_bytes(entries: Mapping[str, str]) -> bytes:
    """Encode a canonical two-space SHA-256 manifest."""

    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("checksum entries must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for path, digest in entries.items():
        normalized[_relative_path(path, label="checksum path")] = _sha256(
            digest,
            label=f"checksum for {path!r}",
        )
    if len(normalized) != len(entries):
        raise ValueError("checksum manifest contains duplicate normalized paths")
    return "".join(f"{normalized[path]}  {path}\n" for path in sorted(normalized)).encode("ascii")


def canonical_sha256sums_bytes(entries: Mapping[str, str]) -> bytes:
    """Compatibility spelling for the canonical SHA-256 manifest encoder."""

    return sha256sums_bytes(entries)


def parse_sha256_sidecar(payload: bytes, *, label: str = "SHA-256 sidecar") -> str:
    """Parse exactly 64 lowercase digest bytes plus one LF."""

    if type(payload) is not bytes or len(payload) != 65 or payload[-1:] != b"\n":
        raise ValueError(f"{label} must contain exactly 64 lowercase hex bytes plus LF")
    try:
        value = payload[:64].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not ASCII") from error
    return _sha256(value, label=label)


def _bundle_kind(value: object) -> str:
    if type(value) is not str or value not in _BUNDLE_KINDS:
        raise ValueError("bundle_kind must be exactly trainer, evaluator, or pilot")
    return value


def _inventory(
    contract: NativeDiffusionV1PilotContract,
    bundle_kind: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    kind = _bundle_kind(bundle_kind)
    files = tuple(getattr(contract, f"{kind}_bundle_files"))
    raw_directories = contract.table("outputs")[f"{kind}_bundle_directories"]
    if type(raw_directories) is not tuple:
        raise ValueError("bundle directory inventory is not an immutable tuple")
    directories = tuple(raw_directories)
    for path in files:
        _relative_path(path, label=f"{kind} file inventory path")
    for path in directories:
        _relative_path(path, label=f"{kind} directory inventory path", allow_dot=True)
    if not files or files[-1] != "manifest.json" or not directories or directories[0] != ".":
        raise ValueError("bundle inventory has no root or final manifest")
    return files, directories


def _runtime_path_bearing(value: str) -> bool:
    if "\x00" in value or "\r" in value:
        return True
    if value in {".", "..", "~"} or value.startswith(("~/", "~\\", "//", "\\\\")):
        return True
    if _URI_RE.match(value) is not None:
        return True
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def validate_path_free_document(value: object, *, label: str = "semantic manifest") -> None:
    """Reject machine-local paths while allowing normalized bundle-local artifact names."""

    plain = _plain_json(value, label=label)

    def visit(item: object, *, parent_key: str | None, in_artifacts: bool) -> None:
        if type(item) is str:
            if _runtime_path_bearing(item):
                raise ValueError(f"{label} contains a runtime path or URI")
            if parent_key is not None:
                lowered = parent_key.lower()
                if any(
                    token in lowered for token in ("runtime_path", "absolute_path", "scratch_root")
                ):
                    raise ValueError(f"{label} contains a path-bearing runtime field")
            return
        if type(item) is list:
            for child in item:
                visit(child, parent_key=parent_key, in_artifacts=in_artifacts)
            return
        if type(item) is dict:
            for key, child in item.items():
                if _runtime_path_bearing(key):
                    raise ValueError(f"{label} contains a runtime path key")
                if in_artifacts:
                    _relative_path(key, label=f"{label} artifact key")
                visit(child, parent_key=key, in_artifacts=in_artifacts or key == "artifacts")

    visit(plain, parent_key=None, in_artifacts=False)


def _normalize_payloads(
    contract: NativeDiffusionV1PilotContract,
    *,
    bundle_kind: str,
    payloads: Mapping[str, bytes],
) -> dict[str, bytes]:
    files, _directories = _inventory(contract, bundle_kind)
    if not isinstance(payloads, Mapping):
        raise TypeError("bundle payloads must be a mapping")
    expected = set(files) - {"manifest.json"}
    if any(type(path) is not str for path in payloads):
        raise ValueError("bundle payload inventory contains a non-string path")
    observed_paths = set(payloads)
    if observed_paths != expected:
        raise ValueError(
            f"{bundle_kind} payload inventory mismatch: "
            f"missing={sorted(expected - observed_paths)}, "
            f"extra={sorted(observed_paths - expected)}"
        )
    result: dict[str, bytes] = {}
    for path in files:
        if path == "manifest.json":
            continue
        payload = payloads[path]
        if type(payload) is not bytes or not payload:
            raise ValueError(f"bundle payload {path!r} must be non-empty exact bytes")
        result[path] = payload
    if hashlib.sha256(result["pilot_execution_v1.toml"]).hexdigest() != contract.config_sha256:
        raise ValueError("bundled child contract bytes do not match the authenticated contract")
    if hashlib.sha256(result["unconditional_v1.toml"]).hexdigest() != contract.parent_config_sha256:
        raise ValueError("bundled parent contract bytes do not match the authenticated contract")
    parse_sha256sums(result["CODE_SHA256SUMS"], label="CODE_SHA256SUMS")
    parse_sha256sums(result["FROZEN_INPUT_SHA256SUMS"], label="FROZEN_INPUT_SHA256SUMS")
    for path, payload in result.items():
        if path.endswith(".json"):
            parsed = parse_canonical_json(payload, label=path)
            if type(parsed) is not dict:
                raise ValueError(f"{path} must contain an exact JSON object")
        elif path.endswith(".jsonl"):
            parse_canonical_jsonl(payload, label=path)
        elif path.endswith(".sha256"):
            parse_sha256_sidecar(payload, label=path)
    return result


def build_bundle_manifest(
    contract: NativeDiffusionV1PilotContract,
    *,
    bundle_kind: str,
    payloads: Mapping[str, bytes],
    fields: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate the exact manifest, binding every non-manifest payload."""

    kind = _bundle_kind(bundle_kind)
    normalized = _normalize_payloads(contract, bundle_kind=kind, payloads=payloads)
    artifacts = {path: hashlib.sha256(payload).hexdigest() for path, payload in normalized.items()}
    return _manifest_from_artifact_hashes(
        contract,
        bundle_kind=kind,
        fields=fields,
        artifacts=artifacts,
    )


def _manifest_from_artifact_hashes(
    contract: NativeDiffusionV1PilotContract,
    *,
    bundle_kind: str,
    fields: Mapping[str, object],
    artifacts: Mapping[str, str],
) -> dict[str, object]:
    """Validate fixed manifest fields against an already sealed artifact map."""

    kind = _bundle_kind(bundle_kind)
    files, _directories = _inventory(contract, kind)
    if not isinstance(fields, Mapping) or any(type(key) is not str for key in fields):
        raise TypeError("manifest fields must be a string-keyed mapping")
    if not isinstance(artifacts, Mapping) or any(type(key) is not str for key in artifacts):
        raise TypeError("manifest artifact hashes must be a string-keyed mapping")
    expected_artifact_paths = set(files) - {"manifest.json"}
    if set(artifacts) != expected_artifact_paths:
        raise ValueError("manifest artifact hash inventory differs from the bundle contract")
    normalized_artifacts = {
        path: _sha256(artifacts[path], label=f"manifest artifact {path!r}")
        for path in files
        if path != "manifest.json"
    }
    raw_fields = contract.table("outputs")[f"{kind}_manifest_fields"]
    if type(raw_fields) is not tuple:
        raise ValueError("manifest field contract is not an immutable tuple")
    expected_fields = tuple(raw_fields)
    allowed_input = set(expected_fields)
    observed_fields = set(fields)
    if observed_fields != allowed_input and observed_fields != allowed_input - {"artifacts"}:
        raise ValueError(
            f"{kind} manifest top-level schema mismatch: "
            f"missing={sorted((allowed_input - {'artifacts'}) - observed_fields)}, "
            f"extra={sorted(observed_fields - allowed_input)}"
        )
    if "artifacts" in fields and fields["artifacts"] != normalized_artifacts:
        raise ValueError("manifest artifacts do not bind every non-manifest payload digest")
    plain_fields = _plain_json(fields, label=f"{kind} manifest")
    if type(plain_fields) is not dict:  # pragma: no cover - mapping conversion invariant
        raise RuntimeError("manifest conversion did not produce an object")
    plain_fields["artifacts"] = normalized_artifacts
    document = {field: plain_fields[field] for field in expected_fields}
    if document["schema_version"] != contract.table("outputs")["schema_version"]:
        raise ValueError("manifest schema version differs from the authenticated contract")
    if document["artifact"] != contract.document["artifact"]:
        raise ValueError("manifest artifact differs from the authenticated child artifact")
    if document["child_contract_sha256"] != contract.config_sha256:
        raise ValueError("manifest child contract digest differs")
    if document["parent_contract_sha256"] != contract.parent_config_sha256:
        raise ValueError("manifest parent contract digest differs")
    commit = document["git_commit"]
    if type(commit) is not str or _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("manifest git_commit must be a lowercase forty-character object ID")
    if "outer_fold" in document:
        outer_fold = document["outer_fold"]
        if type(outer_fold) is not int:
            raise ValueError("manifest outer_fold must be an exact integer")
        fold = contract.fold(outer_fold)
        if kind == "trainer":
            if canonical_json_bytes(document["fit_identity"]) != contract.fit_identity_bytes(
                outer_fold
            ):
                raise ValueError("trainer manifest fit identity differs from the contract")
        elif document["fit_identity_sha256"] != fold.fit_identity_sha256:
            raise ValueError("evaluator manifest fit identity digest differs from the contract")
    validate_path_free_document(document, label=f"{kind} manifest")
    if (
        type(parse_canonical_json(canonical_json_bytes(document), label=f"{kind} manifest"))
        is not dict
    ):
        raise RuntimeError("manifest canonical round trip failed")
    return document


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        try:
            observed = os.lstat(current)
        except FileNotFoundError as error:
            raise ValueError(f"path ancestor does not exist: {current}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"path traverses a symbolic link: {current}")
        if current.parent == current:
            return
        current = current.parent


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


def _hash_sealed_file(path: Path) -> tuple[FileSnapshot, tuple[int, int, int, int, int, int, int]]:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or not 0 < before.st_size <= _MAX_BUNDLE_ARTIFACT_BYTES
    ):
        raise ValueError(
            f"bundle artifact is not a bounded non-empty 0444 single-link regular file: {path.name}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or stat.S_IMODE(opened_before.st_mode) != 0o444
            or opened_before.st_nlink != 1
            or not 0 < opened_before.st_size <= _MAX_BUNDLE_ARTIFACT_BYTES
        ):
            raise ValueError(f"bundle artifact changed before hashing: {path.name}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = os.lstat(path)
    fingerprints = {
        _stat_fingerprint(value) for value in (before, opened_before, opened_after, named_after)
    }
    if len(fingerprints) != 1 or byte_count != before.st_size:
        raise ValueError(f"bundle artifact changed while being hashed: {path.name}")
    fingerprint = fingerprints.pop()
    return (
        FileSnapshot(
            relative_path="placeholder",
            size=before.st_size,
            sha256=digest.hexdigest(),
            mode="0444",
            link_count=before.st_nlink,
            device=before.st_dev,
            inode=before.st_ino,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
        ),
        fingerprint,
    )


def _read_sealed_file(
    path: Path,
    *,
    maximum_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int, int, int], str]:
    before = os.lstat(path)
    if before.st_size > maximum_bytes:
        raise ValueError(f"sealed artifact exceeds read limit: {path.name}")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise ValueError(f"sealed artifact is not a 0444 single-link regular file: {path.name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or stat.S_IMODE(opened_before.st_mode) != 0o444
            or opened_before.st_nlink != 1
        ):
            raise ValueError(f"sealed artifact changed before reading: {path.name}")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = os.lstat(path)
    fingerprints = {
        _stat_fingerprint(value) for value in (before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != before.st_size:
        raise ValueError(f"sealed artifact changed while being read: {path.name}")
    return payload, fingerprints.pop(), hashlib.sha256(payload).hexdigest()


def _directory_snapshot(path: Path, relative_path: str) -> _DirectorySnapshot:
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o555
        or stat.S_ISLNK(observed.st_mode)
    ):
        raise ValueError(f"bundle directory is not a real 0555 directory: {relative_path}")
    return _DirectorySnapshot(
        relative_path=relative_path,
        device=observed.st_dev,
        inode=observed.st_ino,
        os_size=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
        mode=stat.S_IMODE(observed.st_mode),
        link_count=observed.st_nlink,
    )


def _direct_children(
    directory: str,
    *,
    files: Sequence[str],
    directories: Sequence[str],
) -> dict[str, str]:
    parent = PurePosixPath(directory)
    entry_types = {
        **{path: "file" for path in files},
        **{path: "directory" for path in directories},
    }
    result: dict[str, str] = {}
    for path, kind in entry_types.items():
        if path != directory and PurePosixPath(path).parent == parent:
            result[PurePosixPath(path).name] = kind
    return result


def _scan_directory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with os.scandir(path) as entries:
        for entry in entries:
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(observed.st_mode):
                kind = "file"
            elif stat.S_ISDIR(observed.st_mode):
                kind = "directory"
            elif stat.S_ISLNK(observed.st_mode):
                kind = "symlink"
            else:
                kind = "special"
            result[entry.name] = kind
    return result


def _bundle_path(root: Path, relative_path: str) -> Path:
    if relative_path == ".":
        return root
    return root.joinpath(*PurePosixPath(relative_path).parts)


def _capture_bundle(
    contract: NativeDiffusionV1PilotContract,
    *,
    bundle_kind: str,
    root: Path,
) -> tuple[BundleSnapshot, tuple[_DirectorySnapshot, ...]]:
    files, directories = _inventory(contract, bundle_kind)
    _reject_symlink_chain(root)
    directory_before: list[_DirectorySnapshot] = []
    for relative in directories:
        path = _bundle_path(root, relative)
        before = _directory_snapshot(path, relative)
        expected_children = _direct_children(relative, files=files, directories=directories)
        actual_children = _scan_directory(path)
        if actual_children != expected_children:
            raise ValueError(
                f"bundle directory inventory differs at {relative}: "
                f"expected={expected_children}, observed={actual_children}"
            )
        expected_links = 2 + sum(kind == "directory" for kind in expected_children.values())
        if before.link_count != expected_links:
            raise ValueError(f"bundle directory link count differs at {relative}")
        directory_before.append(before)

    snapshots: list[FileSnapshot] = []
    entries: dict[str, object] = {}
    for relative in directories:
        children = _direct_children(relative, files=files, directories=directories)
        entries[relative] = {
            "type": "directory",
            "mode": "0555",
            "size": len(children),
            "sha256": hashlib.sha256(canonical_json_bytes(children)).hexdigest(),
            "link_count": 2 + sum(kind == "directory" for kind in children.values()),
        }
    for relative in files:
        path = _bundle_path(root, relative)
        placeholder, _fingerprint = _hash_sealed_file(path)
        item = FileSnapshot(
            relative_path=relative,
            size=placeholder.size,
            sha256=placeholder.sha256,
            mode=placeholder.mode,
            link_count=placeholder.link_count,
            device=placeholder.device,
            inode=placeholder.inode,
            mtime_ns=placeholder.mtime_ns,
            ctime_ns=placeholder.ctime_ns,
        )
        snapshots.append(item)
        entries[relative] = {
            "type": "file",
            "mode": item.mode,
            "size": item.size,
            "sha256": item.sha256,
            "link_count": item.link_count,
        }

    for before in directory_before:
        after = _directory_snapshot(_bundle_path(root, before.relative_path), before.relative_path)
        if after.fingerprint != before.fingerprint:
            raise ValueError(f"bundle directory changed while being hashed: {before.relative_path}")
    tree_bytes = contract.bundle_tree_map_bytes(bundle_kind, entries)
    snapshot = BundleSnapshot(
        bundle_kind=bundle_kind,
        root=root,
        tree_bytes=tree_bytes,
        tree_sha256=hashlib.sha256(tree_bytes).hexdigest(),
        file_snapshots=tuple(snapshots),
    )
    return snapshot, tuple(directory_before)


def _reauthenticate(snapshot: BundleSnapshot, directories: Sequence[_DirectorySnapshot]) -> None:
    for item in snapshot.file_snapshots:
        path = _bundle_path(snapshot.root, item.relative_path)
        try:
            observed = os.lstat(path)
        except OSError as error:
            raise ValueError(f"sealed bundle file disappeared: {item.relative_path}") from error
        if _stat_fingerprint(observed) != item.fingerprint or not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"sealed bundle file changed: {item.relative_path}")
    for item in directories:
        observed = _directory_snapshot(
            _bundle_path(snapshot.root, item.relative_path),
            item.relative_path,
        )
        if observed.fingerprint != item.fingerprint:
            raise ValueError(f"sealed bundle directory changed: {item.relative_path}")


def _validate_reopened_payloads(
    contract: NativeDiffusionV1PilotContract,
    snapshot: BundleSnapshot,
) -> None:
    semantic_payloads: dict[str, bytes] = {}
    for item in snapshot.file_snapshots:
        if item.relative_path == "manifest.json":
            continue
        path = item.relative_path
        if (
            path in {"CODE_SHA256SUMS", "FROZEN_INPUT_SHA256SUMS"}
            or path.endswith(".json")
            or path.endswith(".jsonl")
            or path.endswith(".sha256")
            or path.endswith(".toml")
        ):
            semantic_payloads[path] = snapshot.read_bytes(
                path,
                maximum_bytes=_MAX_SEMANTIC_TEXT_BYTES,
            )
    # Validate every semantic payload without attempting to parse binary archives.
    if (
        hashlib.sha256(semantic_payloads["pilot_execution_v1.toml"]).hexdigest()
        != contract.config_sha256
    ):
        raise ValueError("sealed bundle child contract bytes differ")
    if (
        hashlib.sha256(semantic_payloads["unconditional_v1.toml"]).hexdigest()
        != contract.parent_config_sha256
    ):
        raise ValueError("sealed bundle parent contract bytes differ")
    parse_sha256sums(semantic_payloads["CODE_SHA256SUMS"], label="CODE_SHA256SUMS")
    parse_sha256sums(
        semantic_payloads["FROZEN_INPUT_SHA256SUMS"],
        label="FROZEN_INPUT_SHA256SUMS",
    )
    for path, payload in semantic_payloads.items():
        if path.endswith(".json"):
            parsed = parse_canonical_json(payload, label=path)
            if type(parsed) is not dict:
                raise ValueError(f"{path} must contain an exact JSON object")
        elif path.endswith(".jsonl"):
            parse_canonical_jsonl(payload, label=path)
        elif path.endswith(".sha256"):
            parse_sha256_sidecar(payload, label=path)

    manifest_payload = snapshot.read_bytes(
        "manifest.json",
        maximum_bytes=_MAX_SEMANTIC_TEXT_BYTES,
    )
    manifest = parse_canonical_json(manifest_payload, label="manifest.json")
    if type(manifest) is not dict:
        raise ValueError("manifest.json must contain an exact JSON object")
    artifacts = {
        item.relative_path: item.sha256
        for item in snapshot.file_snapshots
        if item.relative_path != "manifest.json"
    }
    rebuilt = _manifest_from_artifact_hashes(
        contract,
        bundle_kind=snapshot.bundle_kind,
        fields=manifest,
        artifacts=artifacts,
    )
    if manifest != rebuilt:
        raise ValueError("manifest.json schema or fixed identities differ")
    validate_path_free_document(manifest, label="manifest.json")


def verify_bundle(
    contract: NativeDiffusionV1PilotContract,
    *,
    bundle_kind: str,
    root: str | os.PathLike[str],
    expected_tree_sha256: str | None = None,
    expected_tree_bytes: bytes | None = None,
) -> BundleSnapshot:
    """Reopen and authenticate one exact immutable semantic bundle."""

    kind = _bundle_kind(bundle_kind)
    absolute = _absolute(root)
    snapshot, directories = _capture_bundle(
        contract,
        bundle_kind=kind,
        root=absolute,
    )
    if (
        expected_tree_sha256 is not None
        and _sha256(expected_tree_sha256, label="expected bundle tree digest")
        != snapshot.tree_sha256
    ):
        raise ValueError("bundle tree digest differs from the expected identity")
    if expected_tree_bytes is not None and (
        type(expected_tree_bytes) is not bytes or expected_tree_bytes != snapshot.tree_bytes
    ):
        raise ValueError("bundle tree bytes differ from the expected identity")
    _validate_reopened_payloads(contract, snapshot)
    _reauthenticate(snapshot, directories)
    return snapshot


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_staging(root: Path, directories: Sequence[str]) -> None:
    for relative in sorted(
        directories, key=lambda value: len(PurePosixPath(value).parts), reverse=True
    ):
        path = _bundle_path(root, relative)
        _fsync_directory(path)
        os.chmod(path, 0o555)


def _renameat2_noreplace(staging: Path, output: Path) -> int:
    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
    except (AttributeError, OSError):
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    return 0 if result == 0 else (ctypes.get_errno() or errno.EIO)


def _make_directories(root: Path, directories: Sequence[str]) -> None:
    for relative in sorted(
        (value for value in directories if value != "."),
        key=lambda value: len(PurePosixPath(value).parts),
    ):
        os.mkdir(_bundle_path(root, relative), 0o700)


def _remove_private_tree(path: Path) -> None:
    if not os.path.lexists(path):
        return
    for current, directory_names, _file_names in os.walk(path, topdown=False, followlinks=False):
        for name in directory_names:
            candidate = Path(current) / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
    if not path.is_symlink():
        os.chmod(path, 0o700)
    shutil.rmtree(path)


def _publish_by_links(
    staging: Path,
    output: Path,
    *,
    files: Sequence[str],
    directories: Sequence[str],
) -> None:
    os.mkdir(output, 0o700)
    claim = os.lstat(output)
    claim_identity = (claim.st_dev, claim.st_ino)
    manifest_committed = False
    try:
        _make_directories(output, directories)
        for relative in files:
            if relative == "manifest.json":
                continue
            os.link(
                _bundle_path(staging, relative),
                _bundle_path(output, relative),
                follow_symlinks=False,
            )
        # The manifest is the final file introduced into the publication claim.
        # Keep it unreadable until staging has been removed and every target
        # file consequently has the contractual single link.
        os.chmod(_bundle_path(staging, "manifest.json"), 0o000)
        os.link(
            _bundle_path(staging, "manifest.json"),
            _bundle_path(output, "manifest.json"),
            follow_symlinks=False,
        )
        _seal_staging(output, directories)
        observed = os.lstat(output)
        if (observed.st_dev, observed.st_ino) != claim_identity:
            raise RuntimeError("bundle publication claim changed identity")
        _remove_private_tree(staging)
        # The final chmod is the fallback's atomic semantic commit point.
        manifest_path = _bundle_path(output, "manifest.json")
        os.chmod(manifest_path, 0o444)
        manifest_committed = True
        _fsync_file(manifest_path)
        _fsync_directory(output)
        _fsync_directory(output.parent)
    except BaseException:
        if not manifest_committed:
            observed = os.lstat(output)
            if (observed.st_dev, observed.st_ino) == claim_identity:
                _remove_private_tree(output)
        raise
    finally:
        if staging.exists():
            _remove_private_tree(staging)


def publish_bundle(
    contract: NativeDiffusionV1PilotContract,
    *,
    bundle_kind: str,
    output_dir: str | os.PathLike[str],
    payloads: Mapping[str, bytes],
    manifest: Mapping[str, object],
) -> BundleSnapshot:
    """Stage, seal, and atomically publish a new exact bundle without overwrite."""

    kind = _bundle_kind(bundle_kind)
    files, directories = _inventory(contract, kind)
    normalized = _normalize_payloads(contract, bundle_kind=kind, payloads=payloads)
    complete_manifest = build_bundle_manifest(
        contract,
        bundle_kind=kind,
        payloads=normalized,
        fields=manifest,
    )
    complete_payloads = {**normalized, "manifest.json": canonical_json_bytes(complete_manifest)}
    output = _absolute(output_dir)
    parent = output.parent
    _reject_symlink_chain(parent)
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("bundle output parent must be a real directory")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite semantic bundle: {output}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    published = False
    prepublication: BundleSnapshot | None = None
    try:
        _make_directories(staging, directories)
        for relative in files:
            if relative == "manifest.json":
                continue
            _write_new(_bundle_path(staging, relative), complete_payloads[relative])
        # Contractual commit artifact: it is always created after every payload.
        _write_new(_bundle_path(staging, "manifest.json"), complete_payloads["manifest.json"])
        _seal_staging(staging, directories)
        prepublication = verify_bundle(contract, bundle_kind=kind, root=staging)

        error_number = _renameat2_noreplace(staging, output)
        if error_number == 0:
            published = True
            _fsync_directory(parent)
        elif error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"refusing to overwrite semantic bundle: {output}")
        elif error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            _publish_by_links(
                staging,
                output,
                files=files,
                directories=directories,
            )
            published = True
        else:
            raise OSError(error_number, os.strerror(error_number), str(output))
    finally:
        if not published and staging.exists():
            _remove_private_tree(staging)
    if prepublication is None:  # pragma: no cover - successful publication invariant
        raise RuntimeError("bundle publication has no staged tree identity")
    return verify_bundle(
        contract,
        bundle_kind=kind,
        root=output,
        expected_tree_bytes=prepublication.tree_bytes,
        expected_tree_sha256=prepublication.tree_sha256,
    )


def _run_git(repository: Path, *arguments: str) -> bytes:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Git attestation command failed: git {' '.join(arguments)}") from error
    return completed.stdout


def _git_tree(repository: Path, commit: str) -> tuple[tuple[str, str, str], ...]:
    raw = _run_git(repository, "ls-tree", "-rz", "--full-tree", "-r", commit)
    result: list[tuple[str, str, str]] = []
    prior: bytes | None = None
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, name_bytes = record.split(b"\t", maxsplit=1)
            mode, kind, object_id = metadata.decode("ascii").split(" ")
            name = name_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git tree contains a malformed record") from error
        if prior is not None and name_bytes <= prior:
            raise ValueError("Git tree paths are unordered or duplicated")
        prior = name_bytes
        _relative_path(name, label="Git tree path")
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError("Git tree must contain only regular file blobs")
        if _GIT_COMMIT_RE.fullmatch(object_id) is None:
            raise ValueError("Git tree contains an invalid blob object ID")
        result.append((name, mode, object_id))
    if not result:
        raise ValueError("Git commit tree cannot be empty")
    return tuple(result)


def _read_worktree_file(path: Path) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
    _reject_symlink_chain(path)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"tracked path is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise ValueError(f"tracked path changed before reading: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = os.lstat(path)
    fingerprints = {
        _stat_fingerprint(value) for value in (before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != before.st_size:
        raise ValueError(f"tracked file changed while being read: {path}")
    return payload, fingerprints.pop()


def _forbidden_ignored_source_paths(payload: bytes) -> tuple[str, ...]:
    forbidden: list[str] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        try:
            name = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("ignored source inventory is not UTF-8") from error
        pure = PurePosixPath(name)
        cache = (
            len(pure.parts) >= 4
            and pure.parts[:2] == ("src", "amp_challenge")
            and pure.parent.name == "__pycache__"
            and pure.suffix in {".pyc", ".pyo"}
        )
        if not cache:
            forbidden.append(name)
    return tuple(forbidden)


def build_repository_snapshot(
    repository_root: str | os.PathLike[str],
    *,
    expected_commit: str,
) -> RepositorySnapshot:
    """Bind a clean HEAD/upstream/origin-main worktree to its exact commit blobs."""

    if type(expected_commit) is not str or _GIT_COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a lowercase forty-character Git object ID")
    overrides = tuple(
        sorted(
            name
            for name in os.environ
            if name in _GIT_REPOSITORY_ENVIRONMENT or name.startswith("GIT_CONFIG_")
        )
    )
    if overrides:
        raise ValueError(f"Git repository-selection environment is forbidden: {overrides}")
    repository = _absolute(repository_root)
    _reject_symlink_chain(repository)
    if not repository.is_dir():
        raise ValueError("repository_root must be a real directory")
    top = _absolute(_run_git(repository, "rev-parse", "--show-toplevel").decode("utf-8").strip())
    if top != repository:
        raise ValueError("repository_root must be the exact worktree top level")
    if _run_git(repository, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise ValueError("Git replacement refs are forbidden")
    observed = {
        "HEAD": _run_git(repository, "rev-parse", "--verify", "HEAD^{commit}")
        .decode("ascii")
        .strip(),
        "upstream": _run_git(repository, "rev-parse", "--verify", "@{upstream}^{commit}")
        .decode("ascii")
        .strip(),
        "origin/main": _run_git(
            repository,
            "rev-parse",
            "--verify",
            "refs/remotes/origin/main^{commit}",
        )
        .decode("ascii")
        .strip(),
    }
    if set(observed.values()) != {expected_commit}:
        raise ValueError("HEAD, upstream, and cached origin/main must equal expected_commit")
    status_arguments = (
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if _run_git(repository, *status_arguments):
        raise ValueError("repository worktree must be clean before artifact production")
    ignored = _run_git(
        repository,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "src",
    )
    if _forbidden_ignored_source_paths(ignored):
        raise ValueError("ignored source shadow artifacts are forbidden")

    tree = _git_tree(repository, expected_commit)
    raw_names = _run_git(repository, "ls-files", "-z")
    try:
        tracked_names = tuple(item.decode("utf-8") for item in raw_names.split(b"\0") if item)
    except UnicodeDecodeError as error:
        raise ValueError("tracked path inventory is not UTF-8") from error
    if tracked_names != tuple(item[0] for item in tree):
        raise ValueError("tracked worktree inventory differs from the expected commit tree")

    entries: list[RepositoryEntry] = []
    fingerprints: dict[str, tuple[bytes, tuple[int, int, int, int, int, int, int]]] = {}
    for name, mode, object_id in tree:
        path = repository.joinpath(*PurePosixPath(name).parts)
        payload, fingerprint = _read_worktree_file(path)
        observed_mode = "100755" if fingerprint[-2] & stat.S_IXUSR else "100644"
        if observed_mode != mode:
            raise ValueError(f"tracked file mode differs from commit: {name}")
        if payload != _run_git(repository, "cat-file", "blob", object_id):
            raise ValueError(f"tracked file bytes differ from commit: {name}")
        entries.append(
            RepositoryEntry(
                relative_path=name,
                git_mode=mode,
                git_object=object_id,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        fingerprints[name] = (payload, fingerprint)

    if _run_git(repository, "ls-files", "-z") != raw_names:
        raise ValueError("tracked inventory changed during repository attestation")
    for name, _mode, _object_id in tree:
        current = _read_worktree_file(repository.joinpath(*PurePosixPath(name).parts))
        if current != fingerprints[name]:
            raise ValueError(f"tracked file changed during repository attestation: {name}")
    if _run_git(repository, *status_arguments):
        raise ValueError("repository changed during repository attestation")
    ignored_after = _run_git(
        repository,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "src",
    )
    if ignored_after != ignored:
        raise ValueError("ignored source inventory changed during repository attestation")
    manifest = sha256sums_bytes({item.relative_path: item.sha256 for item in entries})
    return RepositorySnapshot(
        git_commit=expected_commit,
        code_sha256sums=manifest,
        code_sha256=hashlib.sha256(manifest).hexdigest(),
        entries=tuple(entries),
    )


__all__ = [
    "BundleSnapshot",
    "FileSnapshot",
    "RepositoryEntry",
    "RepositorySnapshot",
    "build_bundle_manifest",
    "build_repository_snapshot",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "canonical_sha256sums_bytes",
    "parse_canonical_json",
    "parse_canonical_jsonl",
    "parse_sha256_sidecar",
    "parse_sha256sums",
    "publish_bundle",
    "sha256sums_bytes",
    "validate_path_free_document",
    "verify_bundle",
]

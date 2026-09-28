"""Deterministic immutable phase publications for sequential evaluation.

The sequential-v2 workflow crosses several outcome-isolation boundaries.  A
consumer must therefore be able to distinguish a complete predecessor from a
partially written directory.  This module provides that small publication
primitive without knowing anything about models, candidates, or outcomes.

Every publication is constructed in a private sibling directory, inventories
exactly the payload paths written through :class:`PhaseBuilder`, adds a
canonical receipt and checksum manifest, and verifies the frozen staging tree.
It commits with Linux ``renameat2(RENAME_NOREPLACE)`` where supported.  On
filesystems such as the cluster's Lustre/NFS mounts that reject that operation,
an exclusive directory claim plus verified hardlinks and a private marker copy
uses the transition of ``SHA256SUMS`` from mode ``0000`` to ``0444`` as the
atomic commit marker.  The readable marker is the phase seal; its own SHA-256
is the value bound by a downstream receipt.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "SHA256SUMS"
RECEIPT_NAME = "receipt.json"
_RESERVED_PATHS = frozenset({MANIFEST_NAME, RECEIPT_NAME})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


@dataclass(frozen=True, slots=True)
class PhaseSeal:
    """Path-independent authenticated identity and payload-byte capability."""

    artifact: str
    seal_sha256: str
    receipt_sha256: str
    predecessor_seals: tuple[tuple[str, str], ...]
    payload_sha256: tuple[tuple[str, str], ...]
    payload_bytes: tuple[tuple[str, bytes], ...]
    files: tuple[str, ...]
    metadata_json: bytes

    def read_payload_bytes(self, relative_path: str | PurePosixPath) -> bytes:
        """Return bytes captured from the exact inode authenticated by this seal."""

        logical = validate_relative_path(relative_path)
        matches = tuple(payload for path, payload in self.payload_bytes if path == logical)
        if len(matches) != 1:
            raise ValueError(f"sealed phase lacks exact payload {logical}")
        return matches[0]


def verify_phase_capability(
    seal: PhaseSeal,
    *,
    expected_artifact: str | None = None,
    expected_payload_paths: Iterable[str | PurePosixPath] | None = None,
    expected_predecessor_seals: Mapping[str, str] | None = None,
    expected_seal_sha256: str | None = None,
) -> PhaseSeal:
    """Authenticate a pathless, descriptor-captured phase capability.

    ``verify_phase`` authenticates the filesystem tree and captures its exact
    bytes.  Fresh workers receive only the resulting rootless capability, so
    this function reconstructs the canonical receipt and checksum manifest
    from those captured bytes before any phase-specific decoder runs.
    """

    if type(seal) is not PhaseSeal:
        raise TypeError("phase capability must be a PhaseSeal")
    artifact = _normalize_artifact(seal.artifact)
    seal_sha256 = _normalize_sha256(seal.seal_sha256, label="phase capability seal")
    receipt_sha256 = _normalize_sha256(
        seal.receipt_sha256,
        label="phase capability receipt",
    )

    if type(seal.predecessor_seals) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in seal.predecessor_seals
    ):
        raise ValueError("phase capability predecessors must be immutable key/digest pairs")
    try:
        predecessor_mapping = dict(seal.predecessor_seals)
    except (TypeError, ValueError) as error:
        raise ValueError("phase capability predecessor pairs are invalid") from error
    predecessors = _normalize_predecessors(predecessor_mapping)
    if len(predecessor_mapping) != len(seal.predecessor_seals) or predecessors != (
        seal.predecessor_seals
    ):
        raise ValueError("phase capability predecessor ordering or uniqueness changed")

    if type(seal.payload_sha256) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in seal.payload_sha256
    ):
        raise ValueError("phase capability payload hashes must be immutable path/digest pairs")
    payload_digests: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for path, digest in seal.payload_sha256:
        logical = validate_relative_path(path)
        if logical in _RESERVED_PATHS or logical in seen_paths:
            raise ValueError("phase capability payload paths are reserved or duplicated")
        seen_paths.add(logical)
        payload_digests.append(
            (logical, _normalize_sha256(digest, label=f"phase capability payload {logical}"))
        )
    payloads = tuple(sorted(payload_digests))
    if payloads != seal.payload_sha256:
        raise ValueError("phase capability payload hashes are not canonically ordered")

    if type(seal.payload_bytes) is not tuple or any(
        type(item) is not tuple or len(item) != 2 for item in seal.payload_bytes
    ):
        raise ValueError("phase capability bytes must be immutable path/bytes pairs")
    captured_paths = tuple(path for path, _payload in seal.payload_bytes)
    if captured_paths != tuple(path for path, _digest in payloads):
        raise ValueError("phase capability bytes and payload hashes do not align")
    for (path, payload), (_logical, digest) in zip(
        seal.payload_bytes,
        payloads,
        strict=True,
    ):
        if type(path) is not str or type(payload) is not bytes:
            raise ValueError("phase capability payload entries require exact string/bytes types")
        if sha256_bytes(payload) != digest:
            raise ValueError(f"phase capability payload bytes differ from their digest: {path}")

    if type(seal.metadata_json) is not bytes:
        raise ValueError("phase capability metadata must be canonical captured bytes")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"phase capability metadata duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"phase capability metadata contains invalid constant {value}")

    try:
        metadata = json.loads(
            seal.metadata_json.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("phase capability metadata is not strict UTF-8 JSON") from error
    if not isinstance(metadata, dict) or canonical_json_bytes(metadata) != seal.metadata_json:
        raise ValueError("phase capability metadata is not a canonical JSON object")

    receipt_payload = canonical_json_bytes(
        {
            "artifact": artifact,
            "metadata": metadata,
            "payloads": dict(payloads),
            "predecessor_seals": dict(predecessors),
            "schema_version": 1,
            "status": "sealed",
        }
    )
    if sha256_bytes(receipt_payload) != receipt_sha256:
        raise ValueError("phase capability receipt identity cannot be reconstructed")
    manifest_payload = checksum_manifest_bytes({**dict(payloads), RECEIPT_NAME: receipt_sha256})
    if sha256_bytes(manifest_payload) != seal_sha256:
        raise ValueError("phase capability seal identity cannot be reconstructed")
    expected_files = tuple(sorted((*dict(payloads), RECEIPT_NAME, MANIFEST_NAME)))
    if (
        type(seal.files) is not tuple
        or any(type(path) is not str for path in seal.files)
        or seal.files != expected_files
    ):
        raise ValueError("phase capability file inventory differs from its manifest")

    if expected_artifact is not None and artifact != _normalize_artifact(expected_artifact):
        raise ValueError("phase capability artifact differs from the expected value")
    if expected_payload_paths is not None:
        expected_paths = _normalize_payload_paths(expected_payload_paths)
        if tuple(path for path, _digest in payloads) != expected_paths:
            raise ValueError("phase capability payload inventory differs from expected paths")
    if expected_predecessor_seals is not None:
        expected_predecessors = _normalize_predecessors(expected_predecessor_seals)
        if predecessors != expected_predecessors:
            raise ValueError("phase capability predecessor bindings differ")
    if expected_seal_sha256 is not None and seal_sha256 != _normalize_sha256(
        expected_seal_sha256,
        label="expected phase capability seal",
    ):
        raise ValueError("phase capability seal differs from the expected value")
    return seal


@dataclass(frozen=True, slots=True)
class _Snapshot:
    sha256: str
    size: int
    fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TreeInventory:
    root_fingerprint: tuple[int, ...]
    file_fingerprints: tuple[tuple[str, tuple[int, ...]], ...]
    directories: tuple[str, ...]

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(path for path, _ in self.file_fingerprints)


@dataclass(frozen=True, slots=True)
class _PhaseCommitExpectation:
    """Byte identities needed by the publication fallback."""

    seal_sha256: str
    receipt_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    files: tuple[str, ...]


def sha256_bytes(payload: bytes) -> str:
    """Return lowercase SHA-256 for an immutable byte payload."""

    if not isinstance(payload, bytes):
        raise TypeError("SHA-256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _normalize_json(value: object, *, label: str = "JSON value") -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} object keys must be strings")
            normalized[key] = _normalize_json(item, label=f"{label}.{key}")
        return normalized
    if isinstance(value, tuple | list):
        return [
            _normalize_json(item, label=f"{label}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value as compact sorted UTF-8 plus exactly one LF."""

    normalized = _normalize_json(value)
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize mapping records as compact canonical JSON Lines."""

    if isinstance(records, str | bytes | bytearray):
        raise TypeError("JSONL records must be an iterable of mappings")
    output = bytearray()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"JSONL record {index} must be a mapping")
        output.extend(canonical_json_bytes(record))
    return bytes(output)


def validate_relative_path(path: str | PurePosixPath) -> str:
    """Return a strict portable relative payload path or raise ``ValueError``."""

    if isinstance(path, bytes) or not isinstance(path, str | PurePosixPath):
        raise TypeError("phase payload path must be text")
    raw = str(path)
    if (
        not raw
        or raw == "."
        or raw.startswith("/")
        or raw.endswith("/")
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError(f"unsafe phase payload path: {raw!r}")
    logical = PurePosixPath(raw)
    if logical.as_posix() != raw or any(
        component in {"", ".", ".."} or _PATH_COMPONENT.fullmatch(component) is None
        for component in logical.parts
    ):
        raise ValueError(f"unsafe phase payload path: {raw!r}")
    return raw


def _normalize_artifact(value: str) -> str:
    if type(value) is not str or _ARTIFACT.fullmatch(value) is None:
        raise ValueError("phase artifact must match [a-z0-9][a-z0-9_.-]{0,127}")
    return value


def _normalize_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _normalize_predecessors(
    values: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise TypeError("predecessor seals must be a mapping")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in values.items():
        if type(raw_path) is not str:
            raise TypeError("predecessor seal paths must be exact strings")
        path = validate_relative_path(raw_path)
        if path in normalized:
            raise ValueError(f"duplicate predecessor seal binding: {path}")
        normalized[path] = _normalize_sha256(raw_digest, label=f"predecessor seal {path}")
    return tuple(sorted(normalized.items()))


def _normalize_payload_paths(paths: Iterable[str | PurePosixPath]) -> tuple[str, ...]:
    if isinstance(paths, str | bytes | PurePosixPath):
        raise TypeError("payload paths must be an iterable of paths")
    normalized = tuple(validate_relative_path(path) for path in paths)
    if len(set(normalized)) != len(normalized):
        raise ValueError("payload paths must be unique")
    for path in normalized:
        if path in _RESERVED_PATHS:
            raise ValueError(f"payload path is reserved by the phase protocol: {path}")
    for left in normalized:
        prefix = f"{left}/"
        if any(right.startswith(prefix) for right in normalized if right != left):
            raise ValueError(f"payload path is an ancestor of another payload: {left}")
    return tuple(sorted(normalized))


def checksum_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    """Build the strict sorted two-space checksum-manifest representation."""

    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("checksum manifest entries must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in entries.items():
        path = validate_relative_path(raw_path)
        if path == MANIFEST_NAME:
            raise ValueError("a checksum manifest cannot bind itself")
        if path in normalized:
            raise ValueError(f"duplicate checksum-manifest path: {path}")
        normalized[path] = _normalize_sha256(raw_digest, label=f"checksum for {path}")
    return "".join(f"{normalized[path]}  {path}\n" for path in sorted(normalized)).encode("ascii")


def _parse_checksum_manifest(payload: bytes) -> dict[str, str]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("phase checksum manifest must be non-empty LF-terminated text")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("phase checksum manifest must be ASCII") from error
    entries: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed checksum-manifest line {number}")
        digest, raw_path = match.groups()
        path = validate_relative_path(raw_path)
        if path == MANIFEST_NAME or path in entries:
            raise ValueError(f"invalid checksum-manifest path on line {number}")
        entries[path] = digest
    if checksum_manifest_bytes(entries) != payload:
        raise ValueError("checksum-manifest entries are not in canonical sorted order")
    return entries


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _stable_file_identity(fingerprint: tuple[int, ...]) -> tuple[int, ...]:
    """Return inode attributes that linking/unlinking must not change."""

    if len(fingerprint) != 9:
        raise ValueError("file fingerprint has an unexpected shape")
    return (
        fingerprint[0],
        fingerprint[1],
        fingerprint[2],
        fingerprint[4],
        fingerprint[5],
        fingerprint[6],
        fingerprint[7],
    )


def _directory_authority(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the stable identity and write-authority attributes of a directory."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    while True:
        with suppress(FileNotFoundError):
            if stat.S_ISLNK(os.lstat(candidate).st_mode):
                raise ValueError(f"{label} must not traverse a symbolic link: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _snapshot_regular(
    path: Path,
    *,
    label: str,
    required_mode: int | None = None,
    required_links: int = 1,
) -> _Snapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open {label}: {path}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != required_links:
            raise ValueError(f"{label} must be a {required_links}-link regular file")
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise ValueError(f"{label} must have mode {required_mode:04o}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    if _fingerprint(before) != _fingerprint(after) or _fingerprint(after) != _fingerprint(observed):
        raise RuntimeError(f"{label} changed while it was read")
    return _Snapshot(
        sha256=digest.hexdigest(),
        size=after.st_size,
        fingerprint=_fingerprint(after),
    )


def _snapshot_regular_bytes(
    path: Path,
    *,
    label: str,
    required_mode: int,
    required_links: int = 1,
    max_bytes: int | None = None,
) -> tuple[_Snapshot, bytes]:
    """Snapshot one regular file and retain bytes from that same descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open {label}: {path}") from error
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != required_links:
            raise ValueError(f"{label} must be a {required_links}-link regular file")
        if stat.S_IMODE(before.st_mode) != required_mode:
            raise ValueError(f"{label} must have mode {required_mode:04o}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds its byte bound")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            payload.extend(chunk)
            if max_bytes is not None and len(payload) > max_bytes:
                raise RuntimeError(f"{label} grew beyond its byte bound while being read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    if _fingerprint(before) != _fingerprint(after) or _fingerprint(after) != _fingerprint(observed):
        raise RuntimeError(f"{label} changed while it was read")
    snapshot = _Snapshot(
        sha256=digest.hexdigest(),
        size=after.st_size,
        fingerprint=_fingerprint(after),
    )
    return snapshot, bytes(payload)


def _scan_tree(
    root: Path,
    *,
    require_read_only: bool,
    max_entries: int | None = None,
) -> _TreeInventory:
    root_metadata = os.lstat(root)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("phase root must be a real directory")
    if require_read_only and stat.S_IMODE(root_metadata.st_mode) != 0o555:
        raise ValueError("phase root must have mode 0555")
    files: list[tuple[str, tuple[int, ...]]] = []
    directories: list[str] = []

    def visit(directory: Path, relative: PurePosixPath | None = None) -> None:
        try:
            iterator = os.scandir(directory)
        except OSError as error:
            raise ValueError(f"cannot inventory phase directory: {directory}") from error
        with iterator:
            for entry in iterator:
                if max_entries is not None and len(files) + len(directories) >= max_entries:
                    raise ValueError("phase tree exceeds its entry bound")
                child_relative = (
                    PurePosixPath(entry.name)
                    if relative is None
                    else relative / PurePosixPath(entry.name)
                )
                logical = validate_relative_path(child_relative)
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(f"phase tree contains a symbolic link: {logical}")
                if stat.S_ISDIR(metadata.st_mode):
                    if require_read_only and stat.S_IMODE(metadata.st_mode) != 0o555:
                        raise ValueError(f"phase directory must have mode 0555: {logical}")
                    directories.append(logical)
                    visit(Path(entry.path), child_relative)
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink != 1:
                        raise ValueError(f"phase file must have one hard link: {logical}")
                    if require_read_only and stat.S_IMODE(metadata.st_mode) != 0o444:
                        raise ValueError(f"phase file must have mode 0444: {logical}")
                    files.append((logical, _fingerprint(metadata)))
                else:
                    raise ValueError(f"phase tree contains a non-regular entry: {logical}")

    visit(root)
    return _TreeInventory(
        root_fingerprint=_fingerprint(root_metadata),
        file_fingerprints=tuple(sorted(files)),
        directories=tuple(sorted(directories)),
    )


def _scan_open_tree(
    root_descriptor: int,
    *,
    require_read_only: bool,
    max_entries: int | None = None,
) -> _TreeInventory:
    """Inventory a tree beneath an already-pinned root descriptor."""

    root_metadata = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("phase root descriptor must name a directory")
    if require_read_only and stat.S_IMODE(root_metadata.st_mode) != 0o555:
        raise ValueError("phase root must have mode 0555")
    files: list[tuple[str, tuple[int, ...]]] = []
    directories: list[str] = []

    def visit(directory_descriptor: int, relative: PurePosixPath | None = None) -> None:
        try:
            iterator = os.scandir(directory_descriptor)
        except OSError as error:
            raise ValueError("cannot inventory pinned phase directory") from error
        with iterator:
            for entry in iterator:
                if max_entries is not None and len(files) + len(directories) >= max_entries:
                    raise ValueError("phase tree exceeds its entry bound")
                child_relative = (
                    PurePosixPath(entry.name)
                    if relative is None
                    else relative / PurePosixPath(entry.name)
                )
                logical = validate_relative_path(child_relative)
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(f"phase tree contains a symbolic link: {logical}")
                if stat.S_ISDIR(metadata.st_mode):
                    if require_read_only and stat.S_IMODE(metadata.st_mode) != 0o555:
                        raise ValueError(f"phase directory must have mode 0555: {logical}")
                    child_descriptor = os.open(
                        entry.name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        opened = os.fstat(child_descriptor)
                        if (opened.st_dev, opened.st_ino) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        ):
                            raise RuntimeError(
                                f"phase directory changed while it was pinned: {logical}"
                            )
                        directories.append(logical)
                        visit(child_descriptor, child_relative)
                    finally:
                        os.close(child_descriptor)
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink != 1:
                        raise ValueError(f"phase file must have one hard link: {logical}")
                    if require_read_only and stat.S_IMODE(metadata.st_mode) != 0o444:
                        raise ValueError(f"phase file must have mode 0444: {logical}")
                    files.append((logical, _fingerprint(metadata)))
                else:
                    raise ValueError(f"phase tree contains a non-regular entry: {logical}")

    visit(root_descriptor)
    return _TreeInventory(
        root_fingerprint=_fingerprint(root_metadata),
        file_fingerprints=tuple(sorted(files)),
        directories=tuple(sorted(directories)),
    )


def _expected_directories(files: Iterable[str]) -> tuple[str, ...]:
    directories: set[str] = set()
    for filename in files:
        parent = PurePosixPath(filename).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(directories))


def _load_canonical_receipt(payload: bytes) -> dict[str, object]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("phase receipt is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or canonical_json_bytes(document) != payload:
        raise ValueError("phase receipt is not canonical compact JSON")
    expected = {
        "artifact",
        "metadata",
        "payloads",
        "predecessor_seals",
        "schema_version",
        "status",
    }
    if set(document) != expected:
        raise ValueError("phase receipt has an unexpected schema")
    return document


def verify_phase(
    root: str | Path,
    *,
    expected_artifact: str | None = None,
    expected_payload_paths: Iterable[str | PurePosixPath] | None = None,
    expected_predecessor_seals: Mapping[str, str] | None = None,
    expected_seal_sha256: str | None = None,
) -> PhaseSeal:
    """Strictly authenticate a complete phase tree for downstream use."""

    requested = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_chain(requested, label="phase root")
    before = _scan_tree(requested, require_read_only=True)
    if MANIFEST_NAME not in before.files or RECEIPT_NAME not in before.files:
        raise ValueError("phase tree lacks its receipt or checksum seal")

    manifest_path = requested / MANIFEST_NAME
    manifest_snapshot, manifest_payload = _snapshot_regular_bytes(
        manifest_path,
        label="phase checksum manifest",
        required_mode=0o444,
    )
    entries = _parse_checksum_manifest(manifest_payload)
    expected_files = tuple(sorted((*entries, MANIFEST_NAME)))
    if before.files != expected_files:
        raise ValueError("phase checksum manifest does not bind the exact file inventory")
    if before.directories != _expected_directories(before.files):
        raise ValueError("phase tree contains an empty or unexpected directory")

    snapshots: dict[str, _Snapshot] = {}
    authenticated_bytes: dict[str, bytes] = {}
    for relative, expected_digest in sorted(entries.items()):
        snapshot, payload = _snapshot_regular_bytes(
            requested / relative,
            label=f"phase artifact {relative}",
            required_mode=0o444,
        )
        if snapshot.sha256 != expected_digest:
            raise ValueError(f"phase artifact checksum differs: {relative}")
        snapshots[relative] = snapshot
        authenticated_bytes[relative] = payload

    receipt_payload = authenticated_bytes[RECEIPT_NAME]
    receipt = _load_canonical_receipt(receipt_payload)
    artifact = _normalize_artifact(receipt["artifact"])
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("phase receipt version or status changed")
    if type(receipt["status"]) is not str or receipt["status"] != "sealed":
        raise ValueError("phase receipt version or status changed")

    predecessor_raw = receipt["predecessor_seals"]
    if not isinstance(predecessor_raw, dict):
        raise ValueError("phase predecessor seals must be an object")
    predecessors = _normalize_predecessors(predecessor_raw)
    payload_raw = receipt["payloads"]
    if not isinstance(payload_raw, dict):
        raise ValueError("phase receipt payloads must be an object")
    payloads = tuple(
        sorted(
            (
                validate_relative_path(path),
                _normalize_sha256(digest, label=f"receipt payload {path}"),
            )
            for path, digest in payload_raw.items()
        )
    )
    if any(path in _RESERVED_PATHS for path, _ in payloads):
        raise ValueError("phase receipt lists a reserved path as a payload")
    manifest_payloads = tuple(
        sorted((path, digest) for path, digest in entries.items() if path != RECEIPT_NAME)
    )
    if payloads != manifest_payloads:
        raise ValueError("phase receipt and checksum manifest payload inventories differ")
    metadata = receipt["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("phase receipt metadata must be an object")

    if expected_artifact is not None and artifact != _normalize_artifact(expected_artifact):
        raise ValueError("phase artifact identity differs from the expected value")
    if expected_payload_paths is not None:
        expected_paths = _normalize_payload_paths(expected_payload_paths)
        if tuple(path for path, _ in payloads) != expected_paths:
            raise ValueError("phase payload inventory differs from the expected paths")
    if expected_predecessor_seals is not None:
        expected_predecessors = _normalize_predecessors(expected_predecessor_seals)
        if predecessors != expected_predecessors:
            raise ValueError("phase predecessor seal bindings differ")
    seal_sha256 = manifest_snapshot.sha256
    if expected_seal_sha256 is not None and seal_sha256 != _normalize_sha256(
        expected_seal_sha256, label="expected phase seal"
    ):
        raise ValueError("phase seal SHA-256 differs from the expected value")

    after = _scan_tree(requested, require_read_only=True)
    if before != after:
        raise RuntimeError("phase tree changed while it was authenticated")
    return PhaseSeal(
        artifact=artifact,
        seal_sha256=seal_sha256,
        receipt_sha256=snapshots[RECEIPT_NAME].sha256,
        predecessor_seals=predecessors,
        payload_sha256=payloads,
        payload_bytes=tuple((path, authenticated_bytes[path]) for path, _digest in payloads),
        files=before.files,
        metadata_json=canonical_json_bytes(metadata),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _try_rename_directory_noreplace(source: Path, destination: Path) -> bool:
    """Atomically rename without replacement, or return false if unsupported."""

    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
    except (AttributeError, OSError):
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"refusing to replace phase destination: {destination}",
            destination,
        )
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        return False
    raise OSError(
        error_number,
        f"atomic no-replace phase publication failed: {os.strerror(error_number)}",
        destination,
    )


def relocate_sealed_phase_noreplace(
    source: str | Path,
    destination: str | Path,
    *,
    expected_seal_sha256: str,
    expected_payload_sha256: Mapping[str, str],
) -> str:
    """Relocate a worker outbox phase without opening its payload bytes.

    Fresh sequential workers publish into unrelated private outboxes so their
    argv, environment, and working directory reveal no final-DAG or original
    stage path.  After the worker exits, its supervised result supplies the
    expected seal and payload digests.  This controller-side transition checks
    the immutable tree, receipt/checksum inventory, owner-controlled parents,
    and marker bytes before publishing it under its final name.

    An exclusive ``mkdir`` claim, verified hardlinks, and a private
    hidden-marker copy preserve every payload and receipt inode without a
    check-then-rename race.  The checksum marker's transition from mode
    ``0000`` to ``0444`` is the sole commit point on every filesystem.

    Payload files are never opened here.  Full semantic verification remains
    the publishing worker's responsibility and the procedural attestation is
    authoritative only because it arrived over that worker's exclusive result
    descriptor.
    """

    source_path = Path(os.path.abspath(os.fspath(source)))
    destination_path = Path(os.path.abspath(os.fspath(destination)))
    if source_path == destination_path:
        raise ValueError("phase relocation source and destination must differ")
    if source_path in destination_path.parents or destination_path in source_path.parents:
        raise ValueError("phase relocation source and destination must not contain one another")
    if (
        _PATH_COMPONENT.fullmatch(source_path.name) is None
        or _PATH_COMPONENT.fullmatch(destination_path.name) is None
    ):
        raise ValueError("phase relocation basename is unsafe")
    expected_seal = _normalize_sha256(
        expected_seal_sha256,
        label="expected relocated phase seal",
    )
    if not isinstance(expected_payload_sha256, Mapping):
        raise TypeError("expected relocated payload hashes must be a mapping")
    expected_payloads = tuple(
        sorted(
            (
                validate_relative_path(path),
                _normalize_sha256(digest, label=f"expected relocated payload {path}"),
            )
            for path, digest in expected_payload_sha256.items()
        )
    )
    if not expected_payloads or len(dict(expected_payloads)) != len(expected_payloads):
        raise ValueError("expected relocated payload inventory must be nonempty and unique")
    if any(path in _RESERVED_PATHS for path, _digest in expected_payloads):
        raise ValueError("expected relocated payload inventory contains a reserved path")
    expected_files = tuple(sorted((*dict(expected_payloads), RECEIPT_NAME, MANIFEST_NAME)))
    expected_directories = _expected_directories(expected_files)
    maximum_entries = len(expected_files) + len(expected_directories)
    maximum_marker_bytes = len(
        checksum_manifest_bytes({**dict(expected_payloads), RECEIPT_NAME: "0" * 64})
    )

    def trusted_parent(path: Path, *, label: str) -> tuple[int, ...]:
        _reject_symlink_chain(path, label=label)
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError(
                f"{label} must be a current-user-owned real directory without group/world write"
            )
        return _directory_authority(metadata)

    source_parent_before = trusted_parent(
        source_path.parent,
        label="phase relocation source parent",
    )
    destination_parent_before = trusted_parent(
        destination_path.parent,
        label="phase relocation destination parent",
    )
    if source_parent_before[0] != destination_parent_before[0]:
        raise ValueError("phase relocation requires source and destination on one filesystem")
    if os.path.lexists(destination_path):
        raise FileExistsError(f"refusing to replace phase destination: {destination_path}")
    _reject_symlink_chain(source_path, label="phase relocation source")
    before = _scan_tree(
        source_path,
        require_read_only=True,
        max_entries=maximum_entries,
    )
    if before.files != expected_files or before.directories != expected_directories:
        raise ValueError("relocated phase tree has an unexpected inventory")
    before_files = dict(before.file_fingerprints)
    marker_snapshot, marker_payload = _snapshot_regular_bytes(
        source_path / MANIFEST_NAME,
        label="relocated phase checksum marker",
        required_mode=0o444,
        max_bytes=maximum_marker_bytes,
    )
    if marker_snapshot.fingerprint != before_files[MANIFEST_NAME]:
        raise RuntimeError("relocated phase checksum marker changed after inventory")
    if marker_snapshot.sha256 != expected_seal:
        raise ValueError("relocated phase checksum marker differs from worker attestation")
    manifest = _parse_checksum_manifest(marker_payload)
    if set(manifest) != {*dict(expected_payloads), RECEIPT_NAME} or any(
        manifest[path] != digest for path, digest in expected_payloads
    ):
        raise ValueError("relocated phase checksum manifest differs from worker attestation")
    receipt_sha256 = _normalize_sha256(
        manifest[RECEIPT_NAME],
        label="relocated phase receipt",
    )
    confirmed = _scan_tree(
        source_path,
        require_read_only=True,
        max_entries=maximum_entries,
    )
    if confirmed != before:
        raise RuntimeError("relocated phase tree changed after marker authentication")
    if _fingerprint(os.lstat(source_path)) != before.root_fingerprint:
        raise RuntimeError("relocated phase root changed after inventory")

    expectation = _PhaseCommitExpectation(
        seal_sha256=expected_seal,
        receipt_sha256=receipt_sha256,
        payload_sha256=expected_payloads,
        files=expected_files,
    )
    if (
        trusted_parent(
            source_path.parent,
            label="phase relocation source parent",
        )
        != source_parent_before
        or trusted_parent(
            destination_path.parent,
            label="phase relocation destination parent",
        )
        != destination_parent_before
    ):
        raise RuntimeError("phase relocation parent changed before publication")
    _link_commit_directory_noreplace(
        source_path,
        destination_path,
        expected=expectation,
        source_parent_authority=source_parent_before,
        destination_parent_authority=destination_parent_before,
        source_root_fingerprint=before.root_fingerprint,
        source_file_fingerprints=before_files,
    )
    return expected_seal


def relocate_phase_capability_noreplace_at(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    expected: PhaseSeal,
) -> str:
    """Publish an authenticated sealed phase between already-pinned parents.

    The caller retains ownership of both descriptors. The implementation owns
    readable duplicates for the entire transfer, refuses an existing target,
    authenticates every linked byte against ``expected``, removes all private
    aliases, and makes the destination checksum marker readable only at the
    final commit point. No pathname is reopened to choose either parent.
    """

    verified = verify_phase_capability(expected)
    if type(source_name) is not str or _PATH_COMPONENT.fullmatch(source_name) is None:
        raise ValueError("phase source basename is unsafe")
    if type(destination_name) is not str or _PATH_COMPONENT.fullmatch(destination_name) is None:
        raise ValueError("phase destination basename is unsafe")
    if type(source_parent_descriptor) is not int or source_parent_descriptor < 0:
        raise ValueError("phase source parent descriptor must be a nonnegative exact integer")
    if type(destination_parent_descriptor) is not int or destination_parent_descriptor < 0:
        raise ValueError("phase destination parent descriptor must be a nonnegative exact integer")
    source_parent = os.fstat(source_parent_descriptor)
    destination_parent = os.fstat(destination_parent_descriptor)
    if source_parent.st_dev != destination_parent.st_dev:
        raise ValueError("phase publication requires one filesystem")
    if (source_parent.st_dev, source_parent.st_ino) == (
        destination_parent.st_dev,
        destination_parent.st_ino,
    ) and source_name == destination_name:
        raise ValueError("phase relocation source and destination must differ")
    _link_commit_directory_noreplace(
        Path(source_name),
        Path(destination_name),
        expected=verified,
        pinned_source_parent_descriptor=source_parent_descriptor,
        pinned_destination_parent_descriptor=destination_parent_descriptor,
    )
    return verified.seal_sha256


def _snapshot_open_regular(
    descriptor: int,
    *,
    label: str,
    required_mode: int,
    required_links: int,
) -> _Snapshot:
    """Authenticate a regular file through an already-owned descriptor."""

    digest = hashlib.sha256()
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != required_links:
        raise ValueError(f"{label} must be a {required_links}-link regular file")
    if stat.S_IMODE(before.st_mode) != required_mode:
        raise ValueError(f"{label} must have mode {required_mode:04o}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError(f"{label} changed while it was read")
    return _Snapshot(
        sha256=digest.hexdigest(),
        size=after.st_size,
        fingerprint=_fingerprint(after),
    )


def _snapshot_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
    required_mode: int,
    required_links: int,
) -> _Snapshot:
    """Authenticate a direct child without resolving the claimed root again."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise ValueError(f"cannot safely open {label}: {name}") from error
    try:
        snapshot = _snapshot_open_regular(
            descriptor,
            label=label,
            required_mode=required_mode,
            required_links=required_links,
        )
        observed = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if snapshot.fingerprint != _fingerprint(observed):
            raise RuntimeError(f"{label} changed while it was read")
        return snapshot
    finally:
        os.close(descriptor)


def _open_pinned_directory(path: Path, *, label: str) -> tuple[int, tuple[int, ...]]:
    """Open a real directory and tie its descriptor to its current path name."""

    _reject_symlink_chain(path, label=label)
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} must be a real directory")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
        authority = _directory_authority(before)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or _directory_authority(opened) != authority
            or _directory_authority(named) != authority
        ):
            raise RuntimeError(f"{label} changed while it was pinned")
        return descriptor, authority
    except BaseException:
        os.close(descriptor)
        raise


def _duplicate_trusted_directory(
    descriptor: int,
    *,
    label: str,
) -> tuple[int, tuple[int, ...]]:
    """Own a readable duplicate of a caller-pinned private directory."""

    if type(descriptor) is not int or descriptor < 0:
        raise ValueError(f"{label} descriptor must be a nonnegative exact integer")
    try:
        duplicate = os.open(
            ".",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
    except OSError as error:
        raise ValueError(f"cannot duplicate {label} descriptor") from error
    try:
        original = os.fstat(descriptor)
        opened = os.fstat(duplicate)
        authority = _directory_authority(original)
        if (
            not stat.S_ISDIR(original.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _directory_authority(opened) != authority
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) & 0o022
        ):
            raise ValueError(
                f"{label} descriptor must name a current-user-owned directory "
                "without group/world write"
            )
        return duplicate, authority
    except BaseException:
        os.close(duplicate)
        raise


def _directory_entry_still_names_open(
    parent_descriptor: int,
    name: str,
    opened_descriptor: int,
    identity: tuple[int, int],
) -> bool:
    """Return whether a pinned parent/name still resolves to an open directory."""

    try:
        opened = os.fstat(opened_descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(opened.st_mode)
        and not stat.S_ISLNK(opened.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and (opened.st_dev, opened.st_ino) == identity
        and (named.st_dev, named.st_ino) == identity
    )


def _directory_path_still_names_open(
    path: Path,
    opened_descriptor: int,
    authority: tuple[int, ...],
) -> bool:
    """Return whether an absolute parent path still resolves to its pinned inode."""

    try:
        opened = os.fstat(opened_descriptor)
        named = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISDIR(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and _directory_authority(opened) == authority
        and _directory_authority(named) == authority
    )


def _parent_binding_still_valid(
    path: Path,
    descriptor: int,
    authority: tuple[int, ...],
    *,
    path_bound: bool,
) -> bool:
    """Check either a pathname binding or an intentionally descriptor-only one."""

    if path_bound:
        return _directory_path_still_names_open(path, descriptor, authority)
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and _directory_authority(metadata) == authority


def _quarantine_failed_link_commit(
    *,
    claim_descriptor: int,
    claim_identity: tuple[int, int],
    marker_descriptor: int | None,
    marker_identity: tuple[int, int] | None,
) -> None:
    """Quarantine only the exact claim inodes retained by descriptors."""

    if marker_descriptor is not None and marker_identity is not None:
        with suppress(OSError):
            opened_marker = os.fstat(marker_descriptor)
            if (
                stat.S_ISREG(opened_marker.st_mode)
                and (opened_marker.st_dev, opened_marker.st_ino) == marker_identity
            ):
                os.fchmod(marker_descriptor, 0o000)
    with suppress(OSError):
        opened_claim = os.fstat(claim_descriptor)
        if (
            stat.S_ISDIR(opened_claim.st_mode)
            and (opened_claim.st_dev, opened_claim.st_ino) == claim_identity
        ):
            os.fchmod(claim_descriptor, 0o000)


def _quarantine_unpinned_claim(parent_descriptor: int, name: str) -> None:
    """Best-effort quarantine after an ambiguous claim-acquisition failure."""

    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        identity = (opened.st_dev, opened.st_ino)
        if (
            stat.S_ISDIR(opened.st_mode)
            and not stat.S_ISLNK(named.st_mode)
            and stat.S_ISDIR(named.st_mode)
            and (named.st_dev, named.st_ino) == identity
            and opened.st_uid == os.geteuid()
            and stat.S_IMODE(opened.st_mode) & 0o077 == 0
        ):
            os.fchmod(descriptor, 0o000)
    except OSError:
        return
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _mkdir_claim_noreplace(
    parent_descriptor: int,
    name: str,
    *,
    destination: Path,
) -> None:
    """Atomically claim one destination basename or report non-replacement."""

    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace phase destination: {destination}") from error


def _link_commit_directory_noreplace(
    staging: Path,
    destination: Path,
    *,
    expected: PhaseSeal | _PhaseCommitExpectation,
    source_parent_authority: tuple[int, ...] | None = None,
    destination_parent_authority: tuple[int, ...] | None = None,
    source_root_fingerprint: tuple[int, ...] | None = None,
    source_file_fingerprints: Mapping[str, tuple[int, ...]] | None = None,
    pinned_source_parent_descriptor: int | None = None,
    pinned_destination_parent_descriptor: int | None = None,
) -> None:
    """Fail-closed no-replace publication for filesystems lacking renameat2.

    The destination name is claimed atomically with ``mkdir``.  Payload and
    receipt files are hardlinked and authenticated; the checksum marker is
    copied directly with mode ``0000``.  Every private source link is removed,
    all public files are proved to have link count one, and making the marker
    ``0444`` is the one-way final commit transition.
    """

    relocation_fields = (
        source_parent_authority,
        destination_parent_authority,
        source_root_fingerprint,
        source_file_fingerprints,
    )
    relocation = any(value is not None for value in relocation_fields)
    if relocation and any(value is None for value in relocation_fields):
        raise ValueError("relocation publication requires complete source fingerprints")
    expected_files = expected.files
    expected_digests = {
        **dict(expected.payload_sha256),
        RECEIPT_NAME: expected.receipt_sha256,
        MANIFEST_NAME: expected.seal_sha256,
    }
    if source_file_fingerprints is not None and set(source_file_fingerprints) != set(
        expected_files
    ):
        raise ValueError("relocation source fingerprints differ from expected files")
    expected_directories = _expected_directories(expected_files)
    expected_marker_size = len(
        checksum_manifest_bytes(
            {
                **dict(expected.payload_sha256),
                RECEIPT_NAME: expected.receipt_sha256,
            }
        )
    )
    source_parent_path_bound = pinned_source_parent_descriptor is None
    if source_parent_path_bound:
        staging_parent_descriptor, opened_staging_parent_authority = _open_pinned_directory(
            staging.parent,
            label="phase source parent",
        )
    else:
        assert pinned_source_parent_descriptor is not None
        staging_parent_descriptor, opened_staging_parent_authority = _duplicate_trusted_directory(
            pinned_source_parent_descriptor,
            label="phase source parent",
        )
    if (
        source_parent_authority is not None
        and opened_staging_parent_authority != source_parent_authority
    ):
        os.close(staging_parent_descriptor)
        raise RuntimeError("phase source parent changed before it was pinned")
    destination_parent_path_bound = pinned_destination_parent_descriptor is None
    try:
        if destination_parent_path_bound:
            destination_parent_descriptor, opened_destination_parent_authority = (
                _open_pinned_directory(
                    destination.parent,
                    label="phase destination parent",
                )
            )
        else:
            assert pinned_destination_parent_descriptor is not None
            destination_parent_descriptor, opened_destination_parent_authority = (
                _duplicate_trusted_directory(
                    pinned_destination_parent_descriptor,
                    label="phase destination parent",
                )
            )
    except BaseException:
        os.close(staging_parent_descriptor)
        raise
    if (
        destination_parent_authority is not None
        and opened_destination_parent_authority != destination_parent_authority
    ):
        os.close(destination_parent_descriptor)
        os.close(staging_parent_descriptor)
        raise RuntimeError("phase destination parent changed before it was pinned")
    if os.fstat(staging_parent_descriptor).st_dev != os.fstat(destination_parent_descriptor).st_dev:
        os.close(destination_parent_descriptor)
        os.close(staging_parent_descriptor)
        raise ValueError("phase publication requires one filesystem")
    staged_root_descriptor: int | None = None
    staged_marker_descriptor: int | None = None
    try:
        staged_root_descriptor = os.open(
            staging.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=staging_parent_descriptor,
        )
        staged_root_before = os.fstat(staged_root_descriptor)
        staged_root_identity = (staged_root_before.st_dev, staged_root_before.st_ino)
        named_root = os.stat(
            staging.name,
            dir_fd=staging_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_root.st_mode)
            or (named_root.st_dev, named_root.st_ino) != staged_root_identity
        ):
            raise RuntimeError("phase source root changed while it was pinned")
        if source_root_fingerprint is not None and (
            _fingerprint(staged_root_before) != source_root_fingerprint
            or _fingerprint(named_root) != source_root_fingerprint
        ):
            raise RuntimeError("relocation source root changed before fallback publication")
        staged_marker_descriptor = os.open(
            MANIFEST_NAME,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=staged_root_descriptor,
        )
        marker_before = os.fstat(staged_marker_descriptor)
    except BaseException:
        if (
            source_root_fingerprint is not None
            and len(source_root_fingerprint) >= 2
            and staged_root_descriptor is not None
        ):
            with suppress(OSError):
                opened_root = os.fstat(staged_root_descriptor)
                if (opened_root.st_dev, opened_root.st_ino) == (
                    source_root_fingerprint[0],
                    source_root_fingerprint[1],
                ):
                    if staged_marker_descriptor is not None:
                        with suppress(OSError):
                            os.fchmod(staged_marker_descriptor, 0o000)
                    with suppress(OSError):
                        os.fchmod(staged_root_descriptor, 0o000)
        if staged_marker_descriptor is not None:
            os.close(staged_marker_descriptor)
        if staged_root_descriptor is not None:
            os.close(staged_root_descriptor)
        os.close(destination_parent_descriptor)
        os.close(staging_parent_descriptor)
        raise
    marker_descriptor: int | None = None
    marker_identity: tuple[int, int] | None = None
    claim_descriptor: int | None = None
    claim_identity: tuple[int, int] | None = None
    directory_descriptors: dict[str, int] = {}
    source_directory_descriptors: dict[str, int] = {}
    source_directory_identities: dict[str, tuple[int, int]] = {}
    source_file_identities: dict[str, tuple[int, ...]] = {}
    source_requires_quarantine = False
    claim_acquisition_started = False
    claim_collision = False
    committed = False
    try:
        if (
            not stat.S_ISREG(marker_before.st_mode)
            or stat.S_IMODE(marker_before.st_mode) != 0o444
            or marker_before.st_nlink != 1
            or marker_before.st_size != expected_marker_size
        ):
            source_requires_quarantine = True
            raise RuntimeError("staged phase checksum marker changed before fallback")
        if (
            source_file_fingerprints is not None
            and _fingerprint(marker_before) != (source_file_fingerprints[MANIFEST_NAME])
        ):
            source_requires_quarantine = True
            raise RuntimeError("relocation checksum marker changed before fallback")

        # Atomically reserve the public name before changing the private
        # source.  A late EEXIST therefore leaves the still-sealed source
        # untouched, while every state created below remains uncommitted until
        # the destination marker's final mode transition.
        claim_acquisition_started = True
        try:
            _mkdir_claim_noreplace(
                destination_parent_descriptor,
                destination.name,
                destination=destination,
            )
        except FileExistsError:
            claim_collision = True
            raise
        claim = os.stat(
            destination.name,
            dir_fd=destination_parent_descriptor,
            follow_symlinks=False,
        )
        claim_identity = (claim.st_dev, claim.st_ino)
        if stat.S_ISLNK(claim.st_mode) or not stat.S_ISDIR(claim.st_mode):
            raise RuntimeError("phase publication claim is not a real directory")
        claim_descriptor = os.open(
            destination.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=destination_parent_descriptor,
        )
        if not _directory_entry_still_names_open(
            destination_parent_descriptor,
            destination.name,
            claim_descriptor,
            claim_identity,
        ):
            raise RuntimeError("phase publication claim changed immediately after creation")

        # Once this transition is attempted, every later failure conservatively
        # quarantines the exact source descriptors.  Only the pre-transition
        # no-replace path is permitted to leave the source readable.
        source_requires_quarantine = True
        os.fchmod(staged_marker_descriptor, 0o000)
        hidden_marker = _snapshot_open_regular(
            staged_marker_descriptor,
            label="hidden staged phase checksum marker",
            required_mode=0o000,
            required_links=1,
        )
        if hidden_marker.sha256 != expected.seal_sha256:
            raise RuntimeError("staged phase checksum marker bytes changed before fallback")
        os.lseek(staged_marker_descriptor, 0, os.SEEK_SET)
        marker_payload = bytearray()
        while chunk := os.read(staged_marker_descriptor, 1024 * 1024):
            marker_payload.extend(chunk)

        for relative in sorted(
            expected_directories,
            key=lambda value: (value.count("/"), value),
        ):
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            source_parent_descriptor = (
                staged_root_descriptor
                if not parent_key
                else source_directory_descriptors[parent_key]
            )
            source_descriptor = os.open(
                logical.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=source_parent_descriptor,
            )
            source_metadata = os.fstat(source_descriptor)
            source_identity = (source_metadata.st_dev, source_metadata.st_ino)
            if stat.S_IMODE(
                source_metadata.st_mode
            ) != 0o555 or not _directory_entry_still_names_open(
                source_parent_descriptor,
                logical.name,
                source_descriptor,
                source_identity,
            ):
                os.close(source_descriptor)
                raise RuntimeError(f"phase source child changed while pinned: {relative}")
            source_directory_descriptors[relative] = source_descriptor
            source_directory_identities[relative] = source_identity
            parent_descriptor = (
                claim_descriptor if not parent_key else directory_descriptors[parent_key]
            )
            os.mkdir(logical.name, 0o700, dir_fd=parent_descriptor)
            directory_descriptors[relative] = os.open(
                logical.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        for relative in expected_files:
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            source_directory = (
                staged_root_descriptor
                if not parent_key
                else source_directory_descriptors[parent_key]
            )
            target_directory = (
                claim_descriptor if not parent_key else directory_descriptors[parent_key]
            )
            source_before = os.stat(
                logical.name,
                dir_fd=source_directory,
                follow_symlinks=False,
            )
            source_file_identities[relative] = _stable_file_identity(_fingerprint(source_before))
            required_mode = 0o000 if relative == MANIFEST_NAME else 0o444
            if (
                not stat.S_ISREG(source_before.st_mode)
                or stat.S_IMODE(source_before.st_mode) != required_mode
                or source_before.st_nlink != 1
            ):
                raise RuntimeError(f"staged phase file changed before linking: {relative}")
            if (
                source_file_fingerprints is not None
                and relative != MANIFEST_NAME
                and _fingerprint(source_before) != source_file_fingerprints[relative]
            ):
                raise RuntimeError(f"relocation source inode changed before linking: {relative}")
            if relative == MANIFEST_NAME:
                marker_descriptor = os.open(
                    logical.name,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o000,
                    dir_fd=target_directory,
                )
                remaining = memoryview(marker_payload)
                while remaining:
                    written = os.write(marker_descriptor, remaining)
                    if written <= 0:
                        raise OSError(errno.EIO, "short checksum-marker write")
                    remaining = remaining[written:]
                os.fsync(marker_descriptor)
                target_after = os.stat(
                    logical.name,
                    dir_fd=target_directory,
                    follow_symlinks=False,
                )
                marker_identity = (target_after.st_dev, target_after.st_ino)
                linked = _snapshot_open_regular(
                    marker_descriptor,
                    label=f"linked phase artifact {relative}",
                    required_mode=0o000,
                    required_links=1,
                )
            else:
                os.link(
                    logical.name,
                    logical.name,
                    src_dir_fd=source_directory,
                    dst_dir_fd=target_directory,
                    follow_symlinks=False,
                )
                source_after = os.stat(
                    logical.name,
                    dir_fd=source_directory,
                    follow_symlinks=False,
                )
                target_after = os.stat(
                    logical.name,
                    dir_fd=target_directory,
                    follow_symlinks=False,
                )
                if (
                    (source_after.st_dev, source_after.st_ino)
                    != (source_before.st_dev, source_before.st_ino)
                    or (target_after.st_dev, target_after.st_ino)
                    != (source_before.st_dev, source_before.st_ino)
                    or source_after.st_nlink != 2
                    or target_after.st_nlink != 2
                ):
                    raise RuntimeError(f"phase hardlink identity changed: {relative}")
                if source_file_fingerprints is not None:
                    expected_identity = _stable_file_identity(source_file_fingerprints[relative])
                    if (
                        _stable_file_identity(_fingerprint(source_after)) != expected_identity
                        or _stable_file_identity(_fingerprint(target_after)) != expected_identity
                    ):
                        raise RuntimeError(f"relocation hardlink metadata changed: {relative}")
                    linked = None
                else:
                    linked = _snapshot_regular_at(
                        target_directory,
                        logical.name,
                        label=f"linked phase artifact {relative}",
                        required_mode=0o444,
                        required_links=2,
                    )
            if linked is not None and linked.sha256 != expected_digests[relative]:
                raise RuntimeError(f"linked phase artifact bytes changed: {relative}")

        # Remove every private source link before the public checksum marker can
        # become a valid seal.  A committed tree therefore has no mutable alias.
        remaining_source_directories = set(expected_directories)

        def require_source_directory_bindings() -> None:
            if not _parent_binding_still_valid(
                staging.parent,
                staging_parent_descriptor,
                opened_staging_parent_authority,
                path_bound=source_parent_path_bound,
            ) or not _directory_entry_still_names_open(
                staging_parent_descriptor,
                staging.name,
                staged_root_descriptor,
                staged_root_identity,
            ):
                raise RuntimeError("phase source root changed before private-link removal")
            for relative in sorted(remaining_source_directories):
                logical = PurePosixPath(relative)
                parent = logical.parent.as_posix()
                parent_key = "" if parent == "." else parent
                source_parent = (
                    staged_root_descriptor
                    if not parent_key
                    else source_directory_descriptors[parent_key]
                )
                if not _directory_entry_still_names_open(
                    source_parent,
                    logical.name,
                    source_directory_descriptors[relative],
                    source_directory_identities[relative],
                ):
                    raise RuntimeError(
                        f"phase source child changed before private-link removal: {relative}"
                    )

        require_source_directory_bindings()
        os.fchmod(staged_root_descriptor, 0o700)
        for relative in expected_directories:
            os.fchmod(source_directory_descriptors[relative], 0o700)
        os.close(staged_marker_descriptor)
        staged_marker_descriptor = None
        for relative in expected_files:
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            source_directory = (
                staged_root_descriptor
                if not parent_key
                else source_directory_descriptors[parent_key]
            )
            require_source_directory_bindings()
            source_named = os.stat(
                logical.name,
                dir_fd=source_directory,
                follow_symlinks=False,
            )
            expected_links = 1 if relative == MANIFEST_NAME else 2
            if (
                not stat.S_ISREG(source_named.st_mode)
                or source_named.st_nlink != expected_links
                or _stable_file_identity(_fingerprint(source_named))
                != source_file_identities[relative]
            ):
                raise RuntimeError(
                    f"phase source file changed before private-link removal: {relative}"
                )
            os.unlink(logical.name, dir_fd=source_directory)
        for relative in sorted(
            expected_directories,
            key=lambda value: (-value.count("/"), value),
        ):
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            source_parent_descriptor = (
                staged_root_descriptor
                if not parent_key
                else source_directory_descriptors[parent_key]
            )
            require_source_directory_bindings()
            os.rmdir(logical.name, dir_fd=source_parent_descriptor)
            remaining_source_directories.remove(relative)
        require_source_directory_bindings()
        os.rmdir(staging.name, dir_fd=staging_parent_descriptor)

        for relative in expected_files:
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            target_directory = (
                claim_descriptor if not parent_key else directory_descriptors[parent_key]
            )
            if relative == MANIFEST_NAME:
                published = _snapshot_open_regular(
                    marker_descriptor,
                    label=f"single-link phase artifact {relative}",
                    required_mode=0o000,
                    required_links=1,
                )
            elif source_file_fingerprints is None:
                published = _snapshot_regular_at(
                    target_directory,
                    logical.name,
                    label=f"single-link phase artifact {relative}",
                    required_mode=0o444,
                    required_links=1,
                )
            else:
                published_metadata = os.stat(
                    logical.name,
                    dir_fd=target_directory,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(published_metadata.st_mode)
                    or stat.S_IMODE(published_metadata.st_mode) != 0o444
                    or published_metadata.st_nlink != 1
                    or _stable_file_identity(_fingerprint(published_metadata))
                    != _stable_file_identity(source_file_fingerprints[relative])
                ):
                    raise RuntimeError(f"published relocation inode changed: {relative}")
                published = None
            if published is not None and published.sha256 != expected_digests[relative]:
                raise RuntimeError(f"published phase artifact bytes changed: {relative}")
        inventory = _scan_open_tree(
            claim_descriptor,
            require_read_only=False,
            max_entries=len(expected_files) + len(expected_directories),
        )
        if inventory.files != expected_files or inventory.directories != expected_directories:
            raise RuntimeError("claimed phase inventory changed before commit")
        inventory_files = dict(inventory.file_fingerprints)

        for relative in sorted(
            expected_directories,
            key=lambda value: (-value.count("/"), value),
        ):
            directory_descriptor = directory_descriptors[relative]
            os.fchmod(directory_descriptor, 0o555)
            os.fsync(directory_descriptor)
        os.fchmod(claim_descriptor, 0o555)
        os.fsync(claim_descriptor)
        os.fsync(staging_parent_descriptor)
        os.fsync(destination_parent_descriptor)
        if source_parent_path_bound:
            _reject_symlink_chain(staging.parent, label="phase source parent")
        if destination_parent_path_bound:
            _reject_symlink_chain(destination.parent, label="phase destination parent")
        if (
            not _parent_binding_still_valid(
                staging.parent,
                staging_parent_descriptor,
                opened_staging_parent_authority,
                path_bound=source_parent_path_bound,
            )
            or not _parent_binding_still_valid(
                destination.parent,
                destination_parent_descriptor,
                opened_destination_parent_authority,
                path_bound=destination_parent_path_bound,
            )
            or not _directory_entry_still_names_open(
                destination_parent_descriptor,
                destination.name,
                claim_descriptor,
                claim_identity,
            )
            or stat.S_IMODE(os.fstat(claim_descriptor).st_mode) != 0o555
        ):
            raise RuntimeError("phase publication claim changed before commit")
        for relative in expected_directories:
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            target_parent_descriptor = (
                claim_descriptor if not parent_key else directory_descriptors[parent_key]
            )
            directory_descriptor = directory_descriptors[relative]
            directory_metadata = os.fstat(directory_descriptor)
            if (
                not _directory_entry_still_names_open(
                    target_parent_descriptor,
                    logical.name,
                    directory_descriptor,
                    (directory_metadata.st_dev, directory_metadata.st_ino),
                )
                or stat.S_IMODE(directory_metadata.st_mode) != 0o555
            ):
                raise RuntimeError(f"phase child directory changed before commit: {relative}")
        for relative in expected_files:
            logical = PurePosixPath(relative)
            parent = logical.parent.as_posix()
            parent_key = "" if parent == "." else parent
            target_directory = (
                claim_descriptor if not parent_key else directory_descriptors[parent_key]
            )
            current = os.stat(
                logical.name,
                dir_fd=target_directory,
                follow_symlinks=False,
            )
            if _fingerprint(current) != inventory_files[relative]:
                raise RuntimeError(f"phase file changed before commit: {relative}")
        named_marker = os.stat(
            MANIFEST_NAME,
            dir_fd=claim_descriptor,
            follow_symlinks=False,
        )
        opened_marker = os.fstat(marker_descriptor)
        if (
            not stat.S_ISREG(named_marker.st_mode)
            or (named_marker.st_dev, named_marker.st_ino) != marker_identity
            or (opened_marker.st_dev, opened_marker.st_ino) != marker_identity
            or named_marker.st_nlink != 1
            or stat.S_IMODE(named_marker.st_mode) != 0o000
        ):
            raise RuntimeError("phase checksum marker changed before commit")

        # Revalidate the externally meaningful name immediately before the
        # one-way marker transition.  All earlier inventory checks are against
        # the pinned claim descriptor and cannot substitute for this binding.
        if destination_parent_path_bound:
            _reject_symlink_chain(destination.parent, label="phase destination parent")
        if not _parent_binding_still_valid(
            destination.parent,
            destination_parent_descriptor,
            opened_destination_parent_authority,
            path_bound=destination_parent_path_bound,
        ) or not _directory_entry_still_names_open(
            destination_parent_descriptor,
            destination.name,
            claim_descriptor,
            claim_identity,
        ):
            raise RuntimeError("phase publication claim changed at commit")

        # No fallible operation follows this one-way commit transition.
        os.fchmod(marker_descriptor, 0o444)
        committed = True
    finally:
        if not committed and claim_descriptor is not None and claim_identity is not None:
            _quarantine_failed_link_commit(
                claim_descriptor=claim_descriptor,
                claim_identity=claim_identity,
                marker_descriptor=marker_descriptor,
                marker_identity=marker_identity,
            )
        elif not committed and claim_acquisition_started and not claim_collision:
            _quarantine_unpinned_claim(
                destination_parent_descriptor,
                destination.name,
            )
        if not committed and source_requires_quarantine and staged_root_descriptor is not None:
            if staged_marker_descriptor is not None:
                with suppress(OSError):
                    os.fchmod(staged_marker_descriptor, 0o000)
            with suppress(OSError):
                os.fchmod(staged_root_descriptor, 0o000)
        for descriptor in directory_descriptors.values():
            with suppress(OSError):
                os.close(descriptor)
        for descriptor in source_directory_descriptors.values():
            with suppress(OSError):
                os.close(descriptor)
        if claim_descriptor is not None:
            with suppress(OSError):
                os.close(claim_descriptor)
        if marker_descriptor is not None:
            with suppress(OSError):
                os.close(marker_descriptor)
        if staged_marker_descriptor is not None:
            with suppress(OSError):
                os.close(staged_marker_descriptor)
        if staged_root_descriptor is not None:
            with suppress(OSError):
                os.close(staged_root_descriptor)
        with suppress(OSError):
            os.close(destination_parent_descriptor)
        with suppress(OSError):
            os.close(staging_parent_descriptor)


def _remove_owned_tree(root: Path, identity: tuple[int, int]) -> None:
    with suppress(OSError):
        metadata = os.lstat(root)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            return

        def remove(directory: Path) -> None:
            os.chmod(directory, 0o700, follow_symlinks=False)
            for entry in tuple(os.scandir(directory)):
                child = Path(entry.path)
                child_metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(child_metadata.st_mode) and not stat.S_ISLNK(
                    child_metadata.st_mode
                ):
                    remove(child)
                else:
                    child.unlink()
            directory.rmdir()

        remove(root)


class PhaseBuilder:
    """Construct and atomically publish one deterministic sealed phase."""

    def __init__(
        self,
        destination: str | Path,
        *,
        artifact: str,
        predecessor_seals: Mapping[str, str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._destination = Path(os.path.abspath(os.fspath(destination)))
        self._artifact = _normalize_artifact(artifact)
        self._predecessors = _normalize_predecessors(predecessor_seals)
        raw_metadata: Mapping[str, object] = {} if metadata is None else metadata
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("phase metadata must be a mapping")
        normalized_metadata = _normalize_json(raw_metadata, label="phase metadata")
        if not isinstance(normalized_metadata, dict):
            raise AssertionError("normalized phase metadata is not an object")
        self._metadata: dict[str, object] = normalized_metadata
        self._staging: Path | None = None
        self._staging_identity: tuple[int, int] | None = None
        self._written: dict[str, _Snapshot] = {}
        self._published = False
        self._closed = False

    @property
    def staging_dir(self) -> Path:
        if self._staging is None or self._closed:
            raise RuntimeError("phase builder has no active staging directory")
        return self._staging

    def __enter__(self) -> PhaseBuilder:
        if self._staging is not None or self._closed:
            raise RuntimeError("phase builder cannot be entered more than once")
        if _PATH_COMPONENT.fullmatch(self._destination.name) is None:
            raise ValueError("phase destination basename is unsafe")
        parent = self._destination.parent
        _reject_symlink_chain(parent, label="phase destination parent")
        parent_metadata = os.lstat(parent)
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise ValueError("phase destination parent must be a real existing directory")
        if os.path.lexists(self._destination):
            raise FileExistsError(f"refusing to reuse phase destination: {self._destination}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self._destination.name}.stage.", dir=str(parent))
        )
        os.chmod(staging, 0o700)
        metadata = os.lstat(staging)
        self._staging = staging
        self._staging_identity = (metadata.st_dev, metadata.st_ino)
        return self

    def _require_active(self) -> Path:
        if self._staging is None or self._closed or self._published:
            raise RuntimeError("phase builder is not active")
        return self._staging

    def _prepare_payload_path(self, relative_path: str | PurePosixPath) -> tuple[str, Path]:
        staging = self._require_active()
        logical = validate_relative_path(relative_path)
        if logical in _RESERVED_PATHS:
            raise ValueError(f"payload path is reserved by the phase protocol: {logical}")
        if logical in self._written:
            raise FileExistsError(f"phase payload was already written: {logical}")
        for existing in self._written:
            if logical.startswith(f"{existing}/") or existing.startswith(f"{logical}/"):
                raise ValueError("phase payload paths cannot be ancestors of one another")
        destination = staging / logical
        current = staging
        for component in PurePosixPath(logical).parts[:-1]:
            current /= component
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                metadata = os.lstat(current)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"unsafe phase payload parent: {current}") from None
        return logical, destination

    def write_chunks(
        self,
        relative_path: str | PurePosixPath,
        chunks: Iterable[bytes],
    ) -> str:
        """Write one payload exclusively and return its SHA-256."""

        if isinstance(chunks, str | bytes | bytearray):
            raise TypeError("phase payload chunks must be an iterable of bytes")
        logical, destination = self._prepare_payload_path(relative_path)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination, flags, 0o600)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                for index, chunk in enumerate(chunks):
                    if not isinstance(chunk, bytes):
                        raise TypeError(f"phase payload chunk {index} must be bytes")
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(FileNotFoundError):
                destination.unlink()
            raise
        snapshot = _snapshot_regular(destination, label=f"staged phase payload {logical}")
        if snapshot.sha256 != digest.hexdigest():
            raise RuntimeError(f"phase payload changed immediately after writing: {logical}")
        self._written[logical] = snapshot
        return snapshot.sha256

    def write_bytes(self, relative_path: str | PurePosixPath, payload: bytes) -> str:
        if not isinstance(payload, bytes):
            raise TypeError("phase payload must be bytes")
        return self.write_chunks(relative_path, (payload,))

    def write_json(self, relative_path: str | PurePosixPath, value: object) -> str:
        return self.write_bytes(relative_path, canonical_json_bytes(value))

    def write_jsonl(
        self,
        relative_path: str | PurePosixPath,
        records: Iterable[Mapping[str, object]],
    ) -> str:
        if isinstance(records, str | bytes | bytearray):
            raise TypeError("JSONL records must be an iterable of mappings")

        def chunks() -> Iterable[bytes]:
            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise TypeError(f"JSONL record {index} must be a mapping")
                yield canonical_json_bytes(record)

        return self.write_chunks(relative_path, chunks())

    def _write_internal(self, name: str, payload: bytes) -> _Snapshot:
        staging = self._require_active()
        destination = staging / name
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with suppress(FileNotFoundError):
                destination.unlink()
            raise
        snapshot = _snapshot_regular(destination, label=f"staged phase {name}")
        if snapshot.sha256 != sha256_bytes(payload):
            raise RuntimeError(f"internal phase artifact changed after writing: {name}")
        return snapshot

    def publish(
        self,
        *,
        expected_payload_paths: Iterable[str | PurePosixPath] | None = None,
    ) -> PhaseSeal:
        """Verify, freeze, and atomically commit the phase directory."""

        staging = self._require_active()
        expected = (
            tuple(sorted(self._written))
            if expected_payload_paths is None
            else _normalize_payload_paths(expected_payload_paths)
        )
        if tuple(sorted(self._written)) != expected:
            raise ValueError("written phase payloads differ from the expected inventory")
        inventory = _scan_tree(staging, require_read_only=False)
        if inventory.files != expected or inventory.directories != _expected_directories(expected):
            raise ValueError("staged phase tree differs from payloads written by the builder")
        for logical, original in sorted(self._written.items()):
            current = _snapshot_regular(
                staging / logical,
                label=f"staged phase payload {logical}",
            )
            if current != original:
                raise RuntimeError(f"staged phase payload changed before publication: {logical}")

        payload_digests = {path: self._written[path].sha256 for path in expected}
        receipt = {
            "artifact": self._artifact,
            "metadata": self._metadata,
            "payloads": payload_digests,
            "predecessor_seals": dict(self._predecessors),
            "schema_version": 1,
            "status": "sealed",
        }
        receipt_snapshot = self._write_internal(RECEIPT_NAME, canonical_json_bytes(receipt))
        manifest_entries = {**payload_digests, RECEIPT_NAME: receipt_snapshot.sha256}
        manifest_payload = checksum_manifest_bytes(manifest_entries)
        manifest_snapshot = self._write_internal(MANIFEST_NAME, manifest_payload)

        all_files = tuple(sorted((*expected, RECEIPT_NAME, MANIFEST_NAME)))
        for relative in all_files:
            os.chmod(staging / relative, 0o444, follow_symlinks=False)
        directories = _expected_directories(all_files)
        for relative in sorted(directories, key=lambda value: (-value.count("/"), value)):
            os.chmod(staging / relative, 0o555, follow_symlinks=False)
            _fsync_directory(staging / relative)
        os.chmod(staging, 0o555, follow_symlinks=False)
        _fsync_directory(staging)

        staged = verify_phase(
            staging,
            expected_artifact=self._artifact,
            expected_payload_paths=expected,
            expected_predecessor_seals=dict(self._predecessors),
            expected_seal_sha256=manifest_snapshot.sha256,
        )
        if staged.receipt_sha256 != receipt_snapshot.sha256:
            raise RuntimeError("verified staging receipt differs from the written receipt")

        renamed = _try_rename_directory_noreplace(staging, self._destination)
        if not renamed:
            _link_commit_directory_noreplace(
                staging,
                self._destination,
                expected=staged,
            )
        self._published = True
        _fsync_directory(self._destination.parent)
        published = verify_phase(
            self._destination,
            expected_artifact=self._artifact,
            expected_payload_paths=expected,
            expected_predecessor_seals=dict(self._predecessors),
            expected_seal_sha256=staged.seal_sha256,
        )
        if published != PhaseSeal(
            artifact=staged.artifact,
            seal_sha256=staged.seal_sha256,
            receipt_sha256=staged.receipt_sha256,
            predecessor_seals=staged.predecessor_seals,
            payload_sha256=staged.payload_sha256,
            payload_bytes=staged.payload_bytes,
            files=staged.files,
            metadata_json=staged.metadata_json,
        ):
            raise RuntimeError("phase identity changed across atomic publication")
        return published

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if (
                not self._published
                and self._staging is not None
                and self._staging_identity is not None
            ):
                _remove_owned_tree(self._staging, self._staging_identity)
        finally:
            self._closed = True


def publish_phase(
    destination: str | Path,
    *,
    artifact: str,
    payloads: Mapping[str, bytes],
    predecessor_seals: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PhaseSeal:
    """Convenience wrapper for an in-memory payload mapping."""

    if not isinstance(payloads, Mapping):
        raise TypeError("phase payloads must be a mapping")
    normalized_paths = _normalize_payload_paths(payloads)
    normalized_payloads = {
        validate_relative_path(path): payload for path, payload in payloads.items()
    }
    with PhaseBuilder(
        destination,
        artifact=artifact,
        predecessor_seals=predecessor_seals,
        metadata=metadata,
    ) as builder:
        for relative in normalized_paths:
            builder.write_bytes(relative, normalized_payloads[relative])
        return builder.publish(expected_payload_paths=normalized_paths)


__all__ = [
    "MANIFEST_NAME",
    "RECEIPT_NAME",
    "PhaseBuilder",
    "PhaseSeal",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "checksum_manifest_bytes",
    "publish_phase",
    "relocate_phase_capability_noreplace_at",
    "relocate_sealed_phase_noreplace",
    "sha256_bytes",
    "validate_relative_path",
    "verify_phase",
    "verify_phase_capability",
]

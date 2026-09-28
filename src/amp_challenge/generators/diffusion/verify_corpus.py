"""Independently verify a categorical-diffusion training corpus.

The verifier deliberately does not import the corpus producer.  It reparses the
frozen inputs, reconstructs the sequence/assignment join and component-equal
sampling weights, and compares all three semantic artifacts byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]")
_URI_RE = re.compile(r"(?:file|smb|ssh|s3|gs|https?)://", re.IGNORECASE)

_ARTIFACT = "categorical_diffusion_corpus_v1"
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_MIN_LENGTH = 8
_MAX_LENGTH = 50
_FOLDS = 5
_HOLDOUT_FOLD = 4
_COMPONENT_WEIGHTING = "homology_component_equal_within_role_v1"
_WEIGHT_FORMULA = "1/(homology_component_count_in_role*homology_component_size_in_role)"
_ROLES = ("train", "validation")
_ARTIFACT_FILES = frozenset({"corpus.jsonl", "summary.json", "manifest.json"})
_FROZEN_HASHES = {
    "sequences_sha256": "76305e284ce3689970c89e3f2b45c4adc0ceef2375e74dfcddc50882d8b14884",
    "assignments_sha256": "a8149e03bf2cdf15c70de723613f4b5367db5b7fc887a05b621d3b77834a56a0",
    "parser_top_manifest_sha256": "c83e446f89bbf48a4c3c2fea9a397328147267b231cafad6f913badb7c6c3379",
    "split_top_manifest_sha256": "420bfa5a23475b17090b8af0ff36b67171df21b968b5f3197a28672a2fe530eb",
    "split_independent_receipt_sha256": "7786970354ad0c7f260f86931ec2b9b0b38360fc85f94327aa8bcae91d9055a6",
}
_FROZEN_COUNTS = {
    "expected_sequences": 1113,
    "expected_train_sequences": 914,
    "expected_validation_sequences": 199,
    "expected_homology_components": 597,
    "expected_train_homology_components": 471,
    "expected_validation_homology_components": 126,
    "expected_union_components": 278,
    "expected_train_union_components": 221,
    "expected_validation_union_components": 57,
}
_FROZEN_SEQUENCES_BY_FOLD = (305, 209, 199, 201, 199)

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "alphabet",
        "min_length",
        "max_length",
        "folds",
        "holdout_fold",
        "component_weighting",
        "sequences_sha256",
        "assignments_sha256",
        "parser_top_manifest_sha256",
        "split_top_manifest_sha256",
        "split_independent_receipt_sha256",
        "expected_sequences",
        "expected_train_sequences",
        "expected_validation_sequences",
        "expected_homology_components",
        "expected_train_homology_components",
        "expected_validation_homology_components",
        "expected_union_components",
        "expected_train_union_components",
        "expected_validation_union_components",
        "expected_sequences_by_fold",
    }
)
_SEQUENCE_FIELDS = frozenset({"sequence_id", "sequence", "provenance"})
_ASSIGNMENT_FIELDS = frozenset(
    {"schema_version", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)
_CORPUS_FIELDS = frozenset(
    {
        "schema_version",
        "sequence_id",
        "sequence",
        "fold",
        "role",
        "homology_component_id",
        "homology_component_size",
        "union_component_id",
        "sampling_weight",
    }
)


class VerificationError(ValueError):
    """Raised when any byte-level or semantic corpus invariant fails."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    path: Path
    sha256: str
    artifact: str
    alphabet: str
    min_length: int
    max_length: int
    folds: int
    holdout_fold: int
    component_weighting: str
    sequences_sha256: str
    assignments_sha256: str
    parser_top_manifest_sha256: str
    split_top_manifest_sha256: str
    split_independent_receipt_sha256: str
    expected_sequences: int
    expected_train_sequences: int
    expected_validation_sequences: int
    expected_homology_components: int
    expected_train_homology_components: int
    expected_validation_homology_components: int
    expected_union_components: int
    expected_train_union_components: int
    expected_validation_union_components: int
    expected_sequences_by_fold: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Assignment:
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    fold: int


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int


def _fail(message: str) -> NoReturn:
    raise VerificationError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pread_exact(descriptor: int, size: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        _require(chunk != b"", f"{label} ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink(path: Path, *, label: str, include_leaf: bool = True) -> None:
    """Reject symbolic links in every existing portion of ``path``."""

    target = _absolute(path)
    candidates = list(reversed(target.parents))
    if include_leaf:
        candidates.append(target)
    for candidate in candidates:
        try:
            mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise VerificationError(f"cannot inspect {label}: {candidate}") from error
        _require(not stat.S_ISLNK(mode), f"{label} traverses a symbolic link: {candidate}")


def _read_snapshot(path: str | Path, *, label: str) -> Snapshot:
    requested = _absolute(path)
    _require_no_symlink(requested, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise VerificationError(f"cannot open {label}: {requested}") from error
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
        try:
            named = os.lstat(requested)
        except FileNotFoundError as error:
            raise VerificationError(f"{label} changed while it was read") from error
        _require_no_symlink(requested, label=label)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        stat.S_IMODE(before.st_mode),
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        stat.S_IMODE(after.st_mode),
    )
    identity_named = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
        stat.S_IMODE(named.st_mode),
    )
    _require(
        identity_before == identity_after == identity_named and stat.S_ISREG(named.st_mode),
        f"{label} changed while it was read",
    )
    payload = b"".join(chunks)
    _require(len(payload) == before.st_size, f"{label} changed while it was read")
    return Snapshot(
        path=requested,
        payload=payload,
        sha256=_sha256(payload),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        mode=stat.S_IMODE(before.st_mode),
    )


def _assert_snapshot_unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = _read_snapshot(snapshot.path, label=label)
    _require(
        (
            current.device,
            current.inode,
            current.size,
            current.mtime_ns,
            current.ctime_ns,
            current.mode,
            current.sha256,
        )
        == (
            snapshot.device,
            snapshot.inode,
            snapshot.size,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
            snapshot.mode,
            snapshot.sha256,
        ),
        f"{label} changed during verification",
    )


def _snapshot_identity(snapshot: Snapshot) -> tuple[int, int, int, int, int, int, str]:
    return (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
        snapshot.mode,
        snapshot.sha256,
    )


def _assert_aggregate_unchanged(
    snapshots: Mapping[str, Snapshot],
    *,
    root: Path,
    root_identity: DirectoryIdentity,
) -> None:
    """Repeat a whole-state check so later reads cannot hide earlier drift."""

    expected = {label: _snapshot_identity(snapshot) for label, snapshot in snapshots.items()}
    for pass_number in (1, 2):
        root_before = _directory_identity(root, label="artifact directory")
        inventory_before = _artifact_inventory(root)
        observed = {
            label: _snapshot_identity(
                _read_snapshot(snapshot.path, label=f"{label} aggregate pass {pass_number}")
            )
            for label, snapshot in snapshots.items()
        }
        inventory_after = _artifact_inventory(root)
        root_after = _directory_identity(root, label="artifact directory")
        _require(
            root_before == root_identity == root_after,
            "artifact directory changed during final aggregate check",
        )
        _require(
            inventory_before == _ARTIFACT_FILES == inventory_after,
            "artifact inventory changed during final aggregate check",
        )
        _require(observed == expected, "a verified file changed during final aggregate check")


def _artifact_root(path: str | Path) -> Path:
    requested = _absolute(path)
    _require_no_symlink(requested, label="artifact directory")
    try:
        mode = os.lstat(requested).st_mode
    except OSError as error:
        raise VerificationError(f"cannot inspect artifact directory: {requested}") from error
    _require(stat.S_ISDIR(mode), "artifact directory is not a real directory")
    _require(
        stat.S_IMODE(mode) == 0o555,
        "artifact directory mode must be exactly 0555",
    )
    return requested.resolve(strict=True)


def _directory_identity(path: Path, *, label: str) -> DirectoryIdentity:
    _require_no_symlink(path, label=label)
    try:
        observed = os.lstat(path)
    except OSError as error:
        raise VerificationError(f"cannot inspect {label}: {path}") from error
    _require(stat.S_ISDIR(observed.st_mode), f"{label} is not a real directory")
    _require(stat.S_IMODE(observed.st_mode) == 0o555, f"{label} mode must be exactly 0555")
    return DirectoryIdentity(device=observed.st_dev, inode=observed.st_ino)


def _artifact_inventory(root: Path) -> frozenset[str]:
    names: set[str] = set()
    try:
        entries = list(os.scandir(root))
    except OSError as error:
        raise VerificationError("cannot enumerate artifact directory") from error
    for entry in entries:
        _require(
            not entry.is_symlink(), f"artifact directory contains symbolic entry {entry.name!r}"
        )
        _require(
            entry.is_file(follow_symlinks=False),
            f"artifact directory contains non-file {entry.name!r}",
        )
        _require(
            stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode) == 0o444,
            f"artifact file {entry.name!r} mode must be exactly 0444",
        )
        names.add(entry.name)
    return frozenset(names)


def _duplicate_pairs(label: str):
    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    return hook


def _reject_constant(value: str) -> NoReturn:
    _fail(f"JSON contains non-finite constant {value!r}")


def _require_finite_tree(value: object, *, label: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")
    elif isinstance(value, Mapping):
        for item in value.values():
            _require_finite_tree(item, label=label)
    elif isinstance(value, list | tuple):
        for item in value:
            _require_finite_tree(item, label=label)


def _loads_json(payload: bytes, *, label: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_pairs(label),
            parse_constant=_reject_constant,
        )
    except VerificationError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise VerificationError(f"{label} is invalid JSON") from error
    _require_finite_tree(value, label=label)
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise VerificationError("value cannot be represented as canonical JSON") from error


def _parse_json_document(payload: bytes, *, label: str) -> dict[str, object]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} must have exactly one final LF",
    )
    value = _loads_json(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    document = cast(dict[str, object], value)
    _require(payload == _canonical_json_bytes(document), f"{label} is not canonical compact JSON")
    return document


def _parse_jsonl(
    payload: bytes,
    *,
    label: str,
    canonical: bool,
) -> tuple[dict[str, object], ...]:
    _require(
        payload and payload.endswith(b"\n") and b"\r" not in payload,
        f"{label} must be LF-terminated JSONL",
    )
    rows: list[dict[str, object]] = []
    for number, raw_line in enumerate(payload[:-1].split(b"\n"), start=1):
        _require(raw_line != b"", f"{label} line {number} is blank")
        value = _loads_json(raw_line, label=f"{label} line {number}")
        _require(isinstance(value, dict), f"{label} line {number} must be an object")
        row = cast(dict[str, object], value)
        if canonical:
            _require(
                raw_line + b"\n" == _canonical_json_bytes(row),
                f"{label} line {number} is not canonical compact JSON",
            )
        rows.append(row)
    return tuple(rows)


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _exact_fields(value: Mapping[str, object], expected: Iterable[str], *, label: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    _require(
        observed == expected_set,
        f"{label} schema mismatch: missing={sorted(expected_set - observed)}, "
        f"extra={sorted(observed - expected_set)}",
    )


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    _require(
        type(value) is int and cast(int, value) >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return cast(int, value)


def _sha_field(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return cast(str, value)


def _load_config(snapshot: Snapshot) -> FrozenConfig:
    payload = snapshot.payload
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        "config must have exactly one final LF",
    )
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("config is invalid TOML") from error
    _require_finite_tree(raw, label="config")
    _exact_fields(raw, _CONFIG_FIELDS, label="config")
    _require(
        type(raw["schema_version"]) is int and raw["schema_version"] == 1,
        "config schema_version must be 1",
    )
    _require(raw["artifact"] == _ARTIFACT, "config artifact differs from v1")
    _require(raw["alphabet"] == _ALPHABET, "config alphabet differs from v1")
    _require(
        type(raw["min_length"]) is int and raw["min_length"] == _MIN_LENGTH,
        "config min_length differs from v1",
    )
    _require(
        type(raw["max_length"]) is int and raw["max_length"] == _MAX_LENGTH,
        "config max_length differs from v1",
    )
    _require(type(raw["folds"]) is int and raw["folds"] == _FOLDS, "config folds differs from v1")
    _require(
        type(raw["holdout_fold"]) is int and raw["holdout_fold"] == _HOLDOUT_FOLD,
        "config holdout_fold differs from v1",
    )
    _require(
        raw["component_weighting"] == _COMPONENT_WEIGHTING,
        "config component weighting differs from v1",
    )

    hash_fields = (
        "sequences_sha256",
        "assignments_sha256",
        "parser_top_manifest_sha256",
        "split_top_manifest_sha256",
        "split_independent_receipt_sha256",
    )
    hashes = {name: _sha_field(raw[name], label=f"config {name}") for name in hash_fields}
    count_names = (
        "expected_sequences",
        "expected_train_sequences",
        "expected_validation_sequences",
        "expected_homology_components",
        "expected_train_homology_components",
        "expected_validation_homology_components",
        "expected_union_components",
        "expected_train_union_components",
        "expected_validation_union_components",
    )
    counts = {name: _integer(raw[name], label=f"config {name}", minimum=1) for name in count_names}
    folds_raw = raw["expected_sequences_by_fold"]
    _require(
        isinstance(folds_raw, list) and len(folds_raw) == _FOLDS,
        "config expected_sequences_by_fold must have five entries",
    )
    fold_counts = tuple(
        _integer(value, label=f"config fold {index} count", minimum=1)
        for index, value in enumerate(cast(list[object], folds_raw))
    )
    _require(hashes == _FROZEN_HASHES, "config frozen input hashes changed")
    _require(counts == _FROZEN_COUNTS, "config frozen census changed")
    _require(
        fold_counts == _FROZEN_SEQUENCES_BY_FOLD,
        "config frozen per-fold census changed",
    )

    _require(
        counts["expected_sequences"]
        == counts["expected_train_sequences"] + counts["expected_validation_sequences"],
        "config sequence role counts do not conserve the total",
    )
    _require(
        sum(fold_counts) == counts["expected_sequences"],
        "config fold counts do not conserve sequences",
    )
    _require(
        fold_counts[_HOLDOUT_FOLD] == counts["expected_validation_sequences"]
        and sum(fold_counts[index] for index in range(_FOLDS) if index != _HOLDOUT_FOLD)
        == counts["expected_train_sequences"],
        "config fold counts disagree with roles",
    )
    for component in ("homology", "union"):
        _require(
            counts[f"expected_{component}_components"]
            == counts[f"expected_train_{component}_components"]
            + counts[f"expected_validation_{component}_components"],
            f"config {component} role counts do not conserve the total",
        )
        for role in _ROLES:
            _require(
                counts[f"expected_{role}_{component}_components"]
                <= counts[f"expected_{role}_sequences"],
                f"config has more {role} {component} components than sequences",
            )
    _require(
        counts["expected_union_components"] <= counts["expected_homology_components"],
        "config has more union than homology components",
    )
    for role in _ROLES:
        _require(
            counts[f"expected_{role}_union_components"]
            <= counts[f"expected_{role}_homology_components"],
            f"config has more {role} union than homology components",
        )

    return FrozenConfig(
        path=snapshot.path,
        sha256=snapshot.sha256,
        artifact=cast(str, raw["artifact"]),
        alphabet=cast(str, raw["alphabet"]),
        min_length=cast(int, raw["min_length"]),
        max_length=cast(int, raw["max_length"]),
        folds=cast(int, raw["folds"]),
        holdout_fold=cast(int, raw["holdout_fold"]),
        component_weighting=cast(str, raw["component_weighting"]),
        sequences_sha256=hashes["sequences_sha256"],
        assignments_sha256=hashes["assignments_sha256"],
        parser_top_manifest_sha256=hashes["parser_top_manifest_sha256"],
        split_top_manifest_sha256=hashes["split_top_manifest_sha256"],
        split_independent_receipt_sha256=hashes["split_independent_receipt_sha256"],
        expected_sequences=counts["expected_sequences"],
        expected_train_sequences=counts["expected_train_sequences"],
        expected_validation_sequences=counts["expected_validation_sequences"],
        expected_homology_components=counts["expected_homology_components"],
        expected_train_homology_components=counts["expected_train_homology_components"],
        expected_validation_homology_components=counts["expected_validation_homology_components"],
        expected_union_components=counts["expected_union_components"],
        expected_train_union_components=counts["expected_train_union_components"],
        expected_validation_union_components=counts["expected_validation_union_components"],
        expected_sequences_by_fold=fold_counts,
    )


def _canonical_sequence(value: object, *, config: FrozenConfig, label: str) -> str:
    _require(isinstance(value, str) and value != "", f"{label} must be a non-empty string")
    sequence = cast(str, value)
    _require(sequence == sequence.strip().upper(), f"{label} is not canonical uppercase")
    _require(
        config.min_length <= len(sequence) <= config.max_length,
        f"{label} length is outside the configured range",
    )
    _require(set(sequence) <= set(config.alphabet), f"{label} contains a non-alphabet residue")
    return sequence


def _load_sequences(
    snapshot: Snapshot,
    *,
    config: FrozenConfig,
) -> dict[str, str]:
    _require(snapshot.sha256 == config.sequences_sha256, "sequence input SHA-256 mismatch")
    # The accepted parser-v7 bytes are hash pinned.  Parse them strictly, but do
    # not reinterpret their historical JSON escaping as this producer's format.
    rows = _parse_jsonl(snapshot.payload, label="sequence input", canonical=False)
    _require(len(rows) == config.expected_sequences, "sequence input count mismatch")
    sequences: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        label = f"sequence input row {number}"
        _exact_fields(row, _SEQUENCE_FIELDS, label=label)
        sequence = _canonical_sequence(row["sequence"], config=config, label=f"{label} sequence")
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence_id")
        expected_id = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        _require(sequence_id == expected_id, f"{label} sequence_id is not the canonical SHA-256")
        _require(sequence_id not in sequences, f"{label} repeats sequence_id")
        provenance = row["provenance"]
        _require(
            isinstance(provenance, list) and bool(provenance),
            f"{label} provenance must be non-empty",
        )
        sequences[sequence_id] = sequence
    return sequences


def _load_assignments(
    snapshot: Snapshot,
    *,
    sequences: Mapping[str, str],
    config: FrozenConfig,
) -> dict[str, Assignment]:
    _require(snapshot.sha256 == config.assignments_sha256, "assignment input SHA-256 mismatch")
    rows = _parse_jsonl(snapshot.payload, label="assignment input", canonical=True)
    _require(len(rows) == config.expected_sequences, "assignment input count mismatch")
    assignments: dict[str, Assignment] = {}
    previous: str | None = None
    for number, row in enumerate(rows, start=1):
        label = f"assignment input row {number}"
        _exact_fields(row, _ASSIGNMENT_FIELDS, label=label)
        _require(
            type(row["schema_version"]) is int and row["schema_version"] == 1,
            f"{label} schema_version must be 1",
        )
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence_id")
        _require(
            previous is None or sequence_id > previous,
            "assignment input is not strictly ordered by sequence_id",
        )
        previous = sequence_id
        _require(sequence_id not in assignments, f"{label} repeats sequence_id")
        homology_id = _sha_field(
            row["homology_component_id"], label=f"{label} homology_component_id"
        )
        union_id = _sha_field(row["union_component_id"], label=f"{label} union_component_id")
        fold = _integer(row["fold"], label=f"{label} fold")
        _require(fold < config.folds, f"{label} fold is outside the configured range")
        assignments[sequence_id] = Assignment(sequence_id, homology_id, union_id, fold)
    sequence_ids = set(sequences)
    assignment_ids = set(assignments)
    _require(
        assignment_ids == sequence_ids,
        "sequence/assignment join is not one-to-one: "
        f"missing={sorted(sequence_ids - assignment_ids)}, extra={sorted(assignment_ids - sequence_ids)}",
    )
    return assignments


def _role(fold: int, *, config: FrozenConfig) -> str:
    return "validation" if fold == config.holdout_fold else "train"


def _component_census(
    assignments: Mapping[str, Assignment],
    *,
    config: FrozenConfig,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    Counter[tuple[str, str]],
]:
    fold_counts = Counter(item.fold for item in assignments.values())
    role_counts = Counter(_role(item.fold, config=config) for item in assignments.values())
    homology_by_role = {role: set() for role in _ROLES}
    union_by_role = {role: set() for role in _ROLES}
    homology_semantics: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    union_semantics: dict[str, set[tuple[str, int]]] = defaultdict(set)
    component_sizes: Counter[tuple[str, str]] = Counter()
    for item in assignments.values():
        role = _role(item.fold, config=config)
        homology_by_role[role].add(item.homology_component_id)
        union_by_role[role].add(item.union_component_id)
        homology_semantics[item.homology_component_id].add(
            (role, item.union_component_id, item.fold)
        )
        union_semantics[item.union_component_id].add((role, item.fold))
        component_sizes[(role, item.homology_component_id)] += 1
    for component_id, semantics in homology_semantics.items():
        _require(
            len(semantics) == 1,
            f"homology component {component_id} crosses a fold, role, or union component",
        )
    for component_id, semantics in union_semantics.items():
        _require(len(semantics) == 1, f"union component {component_id} crosses a fold or role")

    observed_folds = {str(fold): fold_counts[fold] for fold in range(config.folds)}
    _require(
        tuple(observed_folds[str(fold)] for fold in range(config.folds))
        == config.expected_sequences_by_fold,
        "assignment fold counts differ from config",
    )
    counts = {
        "sequences": len(assignments),
        "train_sequences": role_counts["train"],
        "validation_sequences": role_counts["validation"],
    }
    _require(
        counts
        == {
            "sequences": config.expected_sequences,
            "train_sequences": config.expected_train_sequences,
            "validation_sequences": config.expected_validation_sequences,
        },
        "assignment role counts differ from config",
    )
    homology_counts = {
        "total": len(set().union(*homology_by_role.values())),
        "train": len(homology_by_role["train"]),
        "validation": len(homology_by_role["validation"]),
    }
    _require(
        homology_counts
        == {
            "total": config.expected_homology_components,
            "train": config.expected_train_homology_components,
            "validation": config.expected_validation_homology_components,
        },
        "homology component counts differ from config",
    )
    union_counts = {
        "total": len(set().union(*union_by_role.values())),
        "train": len(union_by_role["train"]),
        "validation": len(union_by_role["validation"]),
    }
    _require(
        union_counts
        == {
            "total": config.expected_union_components,
            "train": config.expected_train_union_components,
            "validation": config.expected_validation_union_components,
        },
        "union component counts differ from config",
    )
    return counts, observed_folds, homology_counts, union_counts, component_sizes


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def _expected_documents(
    *,
    sequences: Mapping[str, str],
    assignments: Mapping[str, Assignment],
    config: FrozenConfig,
    config_sha256: str,
    sequences_sha256: str,
    assignments_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    counts, fold_counts, homology_counts, union_counts, component_sizes = _component_census(
        assignments, config=config
    )
    corpus_rows: list[dict[str, object]] = []
    for sequence_id in sorted(sequences):
        assignment = assignments[sequence_id]
        role = _role(assignment.fold, config=config)
        weight = 1.0 / (
            homology_counts[role] * component_sizes[(role, assignment.homology_component_id)]
        )
        corpus_rows.append(
            {
                "schema_version": 1,
                "sequence_id": sequence_id,
                "sequence": sequences[sequence_id],
                "fold": assignment.fold,
                "role": role,
                "homology_component_id": assignment.homology_component_id,
                "homology_component_size": component_sizes[
                    (role, assignment.homology_component_id)
                ],
                "union_component_id": assignment.union_component_id,
                "sampling_weight": weight,
            }
        )
    role_weight_sums: dict[str, float] = {}
    component_mass: dict[str, dict[str, float]] = {}
    for role in _ROLES:
        role_rows = [row for row in corpus_rows if row["role"] == role]
        role_sum = math.fsum(cast(float, row["sampling_weight"]) for row in role_rows)
        masses: dict[str, list[float]] = defaultdict(list)
        for row in role_rows:
            masses[cast(str, row["homology_component_id"])].append(
                cast(float, row["sampling_weight"])
            )
        observed_masses = [math.fsum(values) for values in masses.values()]
        expected_mass = 1.0 / homology_counts[role]
        _require(
            math.isclose(role_sum, 1.0, rel_tol=0.0, abs_tol=1e-15)
            and bool(observed_masses)
            and all(
                math.isfinite(value)
                and value > 0.0
                and math.isclose(value, expected_mass, rel_tol=0.0, abs_tol=1e-15)
                for value in observed_masses
            ),
            f"{role} component-equal sampling weights failed reconstruction",
        )
        role_weight_sums[role] = role_sum
        component_mass[role] = {
            "expected": expected_mass,
            "min": min(observed_masses),
            "max": max(observed_masses),
        }
    lengths_by_role: dict[str, list[int]] = {role: [] for role in _ROLES}
    for sequence_id, sequence in sequences.items():
        role = _role(assignments[sequence_id].fold, config=config)
        lengths_by_role[role].append(len(sequence))
    length_counts = {
        "total": _histogram(len(sequence) for sequence in sequences.values()),
        "train": _histogram(lengths_by_role["train"]),
        "validation": _histogram(lengths_by_role["validation"]),
    }
    policy = {
        "alphabet": config.alphabet,
        "min_length": config.min_length,
        "max_length": config.max_length,
        "folds": config.folds,
        "holdout_fold": config.holdout_fold,
        "role_by_fold": {str(fold): _role(fold, config=config) for fold in range(config.folds)},
        "component_weighting": config.component_weighting,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact": config.artifact,
        "counts": counts,
        "fold_counts": fold_counts,
        "homology_component_counts": homology_counts,
        "union_component_counts": union_counts,
        "length_counts": length_counts,
        "sampling_weight_checks": {
            "formula": _WEIGHT_FORMULA,
            "role_weight_sums": role_weight_sums,
            "homology_component_mass": component_mass,
        },
        "policy": policy,
    }
    corpus_payload = _jsonl_bytes(corpus_rows)
    summary_payload = _canonical_json_bytes(summary)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": config.artifact,
        "config_sha256": config_sha256,
        "input": {
            "sequences": {"sha256": sequences_sha256},
            "assignments": {"sha256": assignments_sha256},
        },
        "provenance": {
            "parser_top_manifest_sha256": config.parser_top_manifest_sha256,
            "split_top_manifest_sha256": config.split_top_manifest_sha256,
            "split_independent_receipt_sha256": config.split_independent_receipt_sha256,
        },
        "artifacts": {
            "corpus": {"sha256": _sha256(corpus_payload)},
            "summary": {"sha256": _sha256(summary_payload)},
        },
        "counts": counts,
        "fold_counts": fold_counts,
        "homology_component_counts": homology_counts,
        "union_component_counts": union_counts,
        "length_counts": length_counts,
        "sampling_weight_checks": summary["sampling_weight_checks"],
        "policy": policy,
    }
    return corpus_rows, summary, manifest


def _reject_path_semantics(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            _require(
                not any(
                    token in lowered for token in ("path", "filename", "directory", "uri", "url")
                ),
                f"{label} contains path-bearing field {key!r}",
            )
            _reject_path_semantics(item, label=label)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_path_semantics(item, label=label)
    elif isinstance(value, str):
        if value == _WEIGHT_FORMULA:
            return
        _require("\x00" not in value, f"{label} contains a NUL string")
        _require("/" not in value and "\\" not in value, f"{label} contains path-bearing text")
        _require(_WINDOWS_PATH_RE.search(value) is None, f"{label} contains a Windows path")
        _require(_URI_RE.search(value) is None, f"{label} contains a URI")
        _require(value not in {".", "..", "~"}, f"{label} contains path-bearing text")


def _verify_corpus_rows(
    actual: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
) -> None:
    _require(len(actual) == len(expected), "corpus row count mismatch")
    role_sums: dict[str, list[float]] = defaultdict(list)
    component_sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    component_ids: dict[str, set[str]] = defaultdict(set)
    previous: str | None = None
    for number, (row, expected_row) in enumerate(zip(actual, expected, strict=True), start=1):
        label = f"corpus row {number}"
        _exact_fields(row, _CORPUS_FIELDS, label=label)
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence_id")
        _require(
            previous is None or sequence_id > previous,
            "corpus is not strictly ordered by sequence_id",
        )
        previous = sequence_id
        weight_value = row["sampling_weight"]
        _require(type(weight_value) in {int, float}, f"{label} sampling_weight is not numeric")
        weight = float(cast(int | float, weight_value))
        _require(
            math.isfinite(weight) and weight > 0.0,
            f"{label} sampling_weight is not positive finite",
        )
        _require(row == expected_row, f"{label} differs from independent reconstruction")
        role = cast(str, row["role"])
        homology_id = cast(str, row["homology_component_id"])
        _require(
            type(row["homology_component_size"]) is int and row["homology_component_size"] > 0,
            f"{label} homology_component_size is not positive",
        )
        role_sums[role].append(weight)
        component_sums[(role, homology_id)].append(weight)
        component_ids[role].add(homology_id)
    for role in _ROLES:
        _require(
            math.isclose(math.fsum(role_sums[role]), 1.0, rel_tol=0.0, abs_tol=1e-15),
            f"{role} sampling weights do not sum to one",
        )
        component_mass = 1.0 / len(component_ids[role])
        for (component_role, component_id), values in component_sums.items():
            if component_role == role:
                _require(
                    math.isclose(math.fsum(values), component_mass, rel_tol=0.0, abs_tol=1e-15),
                    f"{role} homology component {component_id} does not have equal mass",
                )


def verify_diffusion_corpus(
    *,
    config_path: str | Path,
    sequences_path: str | Path,
    assignments_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Verify and return a path-free receipt without writing it."""

    root = _artifact_root(artifact_dir)
    root_identity = _directory_identity(root, label="artifact directory")
    inventory = _artifact_inventory(root)
    _require(
        inventory == _ARTIFACT_FILES,
        "artifact inventory mismatch: "
        f"missing={sorted(_ARTIFACT_FILES - inventory)}, extra={sorted(inventory - _ARTIFACT_FILES)}",
    )
    snapshots = {
        "config": _read_snapshot(config_path, label="config"),
        "sequences": _read_snapshot(sequences_path, label="sequence input"),
        "assignments": _read_snapshot(assignments_path, label="assignment input"),
        "corpus": _read_snapshot(root / "corpus.jsonl", label="corpus artifact"),
        "summary": _read_snapshot(root / "summary.json", label="summary artifact"),
        "manifest": _read_snapshot(root / "manifest.json", label="manifest artifact"),
    }
    config = _load_config(snapshots["config"])
    sequences = _load_sequences(snapshots["sequences"], config=config)
    assignments = _load_assignments(snapshots["assignments"], sequences=sequences, config=config)
    expected_corpus, expected_summary, expected_manifest = _expected_documents(
        sequences=sequences,
        assignments=assignments,
        config=config,
        config_sha256=snapshots["config"].sha256,
        sequences_sha256=snapshots["sequences"].sha256,
        assignments_sha256=snapshots["assignments"].sha256,
    )

    actual_corpus = _parse_jsonl(
        snapshots["corpus"].payload, label="corpus artifact", canonical=True
    )
    actual_summary = _parse_json_document(snapshots["summary"].payload, label="summary artifact")
    actual_manifest = _parse_json_document(snapshots["manifest"].payload, label="manifest artifact")
    _reject_path_semantics(actual_corpus, label="corpus artifact")
    _reject_path_semantics(actual_summary, label="summary artifact")
    _reject_path_semantics(actual_manifest, label="manifest artifact")
    _verify_corpus_rows(actual_corpus, expected_corpus)

    expected_corpus_payload = _jsonl_bytes(expected_corpus)
    expected_summary_payload = _canonical_json_bytes(expected_summary)
    expected_manifest_payload = _canonical_json_bytes(expected_manifest)
    _require(
        snapshots["corpus"].payload == expected_corpus_payload,
        "corpus bytes differ from independent reconstruction",
    )
    _require(
        snapshots["summary"].payload == expected_summary_payload,
        "summary bytes differ from independent reconstruction",
    )
    _require(
        actual_summary == expected_summary,
        "summary semantics differ from independent reconstruction",
    )
    _require(
        snapshots["manifest"].payload == expected_manifest_payload,
        "manifest bytes differ from independent reconstruction",
    )
    _require(
        actual_manifest == expected_manifest,
        "manifest semantics differ from independent reconstruction",
    )

    for label, snapshot in snapshots.items():
        _assert_snapshot_unchanged(snapshot, label=label)
    _require(
        _artifact_inventory(root) == _ARTIFACT_FILES,
        "artifact inventory changed during verification",
    )
    _require(
        _directory_identity(root, label="artifact directory") == root_identity,
        "artifact directory inode changed during verification",
    )
    _assert_aggregate_unchanged(snapshots, root=root, root_identity=root_identity)

    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact": "categorical_diffusion_corpus_independent_verification_v1",
        "status": "passed",
        "verified_artifact": config.artifact,
        "input_sha256": {
            "config": snapshots["config"].sha256,
            "sequences": snapshots["sequences"].sha256,
            "assignments": snapshots["assignments"].sha256,
        },
        "artifact_sha256": {
            "corpus": snapshots["corpus"].sha256,
            "summary": snapshots["summary"].sha256,
            "manifest": snapshots["manifest"].sha256,
        },
        "counts": expected_summary["counts"],
        "invariants": {
            "artifact_root_identity_stable": True,
            "artifact_inventory_exact": True,
            "artifact_permissions_immutable": True,
            "canonical_bytes_exact": True,
            "component_roles_and_folds_isolated": True,
            "config_and_input_hashes_valid": True,
            "fold_role_and_component_counts_exact": True,
            "one_to_one_join_exact": True,
            "sampling_weights_and_component_mass_exact": True,
            "semantic_outputs_location_free": True,
        },
    }
    _reject_path_semantics(receipt, label="verification receipt")
    return receipt


def verify_corpus(
    *,
    config_path: str | Path,
    sequences_path: str | Path,
    assignments_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, object]:
    """Compatibility spelling for :func:`verify_diffusion_corpus`."""

    return verify_diffusion_corpus(
        config_path=config_path,
        sequences_path=sequences_path,
        assignments_path=assignments_path,
        artifact_dir=artifact_dir,
    )


def write_receipt(
    path: str | Path,
    receipt: Mapping[str, object],
    *,
    artifact_dir: str | Path,
    config_path: str | Path,
    sequences_path: str | Path,
    assignments_path: str | Path,
) -> Path:
    """Reverify around atomic publication of one canonical read-only receipt."""

    root = _artifact_root(artifact_dir)
    requested = _absolute(path)
    _require(requested.name not in {"", ".", ".."}, "receipt path has no filename")
    prospective = requested.resolve(strict=False)
    _require(
        prospective != root and not prospective.is_relative_to(root),
        "verification receipt must be outside the verified artifact directory",
    )
    _require_no_symlink(requested.parent, label="receipt parent")
    _require(not os.path.lexists(requested), f"refusing to overwrite receipt: {requested}")
    try:
        requested.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VerificationError(f"cannot create receipt parent: {requested.parent}") from error
    _require_no_symlink(requested.parent, label="receipt parent")
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    _require(
        target != root and not target.is_relative_to(root),
        "verification receipt must be outside the verified artifact directory",
    )
    _require(not os.path.lexists(target), f"refusing to overwrite receipt: {target}")
    receipt_document = dict(receipt)
    _reject_path_semantics(receipt_document, label="verification receipt")
    payload = _canonical_json_bytes(receipt_document)

    directory_descriptor: int | None = None
    staging_descriptor: int | None = None
    staging_name: str | None = None
    linked_identity: tuple[int, int] | None = None
    publication_complete = False
    cleanup_error: OSError | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_descriptor = os.open(parent, directory_flags)
        opened_parent = os.fstat(directory_descriptor)
        _require(stat.S_ISDIR(opened_parent.st_mode), "receipt parent is not a directory")

        staging_flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for _ in range(128):
            candidate = f".{target.name}-{secrets.token_hex(16)}"
            try:
                staging_descriptor = os.open(
                    candidate,
                    staging_flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            staging_name = candidate
            break
        _require(
            staging_descriptor is not None and staging_name is not None,
            "cannot allocate a unique receipt staging file",
        )
        offset = 0
        while offset < len(payload):
            written = os.write(staging_descriptor, payload[offset:])
            _require(written > 0, "receipt staging write made no progress")
            offset += written
        os.fchmod(staging_descriptor, 0o444)
        os.fsync(staging_descriptor)
        staged = os.fstat(staging_descriptor)
        _require(
            stat.S_ISREG(staged.st_mode)
            and stat.S_IMODE(staged.st_mode) == 0o444
            and staged.st_size == len(payload),
            "receipt staging file identity or mode differs",
        )
        _require(
            _pread_exact(staging_descriptor, staged.st_size, label="receipt staging file")
            == payload,
            "receipt staging bytes differ before publication",
        )
        before_publication = verify_diffusion_corpus(
            config_path=config_path,
            sequences_path=sequences_path,
            assignments_path=assignments_path,
            artifact_dir=root,
        )
        _require(
            _canonical_json_bytes(before_publication) == payload,
            "supplied receipt differs from immediate pre-publication verification",
        )
        try:
            os.link(
                staging_name,
                target.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise VerificationError(f"refusing to overwrite receipt: {target}") from error
        linked_identity = (staged.st_dev, staged.st_ino)
        published = os.stat(
            target.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        staged_after_link = os.fstat(staging_descriptor)
        _require(
            (published.st_dev, published.st_ino, published.st_size)
            == (staged.st_dev, staged.st_ino, staged.st_size)
            == (
                staged_after_link.st_dev,
                staged_after_link.st_ino,
                staged_after_link.st_size,
            )
            and stat.S_ISREG(published.st_mode)
            and stat.S_IMODE(published.st_mode) == 0o444,
            "published receipt does not match its descriptor-bound staging file",
        )
        os.unlink(staging_name, dir_fd=directory_descriptor)
        staging_name = None
        os.fsync(directory_descriptor)

        named_parent = os.stat(parent, follow_symlinks=False)
        final_opened_parent = os.fstat(directory_descriptor)
        _require(
            (named_parent.st_dev, named_parent.st_ino)
            == (opened_parent.st_dev, opened_parent.st_ino)
            == (final_opened_parent.st_dev, final_opened_parent.st_ino),
            "receipt parent directory changed during publication",
        )
        final_published = os.stat(
            target.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        _require(
            (final_published.st_dev, final_published.st_ino, final_published.st_size)
            == (staged.st_dev, staged.st_ino, staged.st_size)
            and stat.S_ISREG(final_published.st_mode)
            and stat.S_IMODE(final_published.st_mode) == 0o444,
            "published receipt changed before publication completed",
        )
        _require(
            _pread_exact(staging_descriptor, staged.st_size, label="published receipt") == payload,
            "published receipt bytes changed before publication completed",
        )
        after_publication = verify_diffusion_corpus(
            config_path=config_path,
            sequences_path=sequences_path,
            assignments_path=assignments_path,
            artifact_dir=root,
        )
        _require(
            _canonical_json_bytes(after_publication) == payload,
            "supplied receipt differs from immediate post-publication verification",
        )
        _require_no_symlink(parent, label="receipt parent")
        latest_named_parent = os.stat(parent, follow_symlinks=False)
        latest_opened_parent = os.fstat(directory_descriptor)
        _require(
            (latest_named_parent.st_dev, latest_named_parent.st_ino)
            == (opened_parent.st_dev, opened_parent.st_ino)
            == (latest_opened_parent.st_dev, latest_opened_parent.st_ino),
            "receipt parent directory changed during post-publication verification",
        )
        final_target = os.stat(
            target.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        _require(
            (final_target.st_dev, final_target.st_ino) == linked_identity
            and final_target.st_size == len(payload)
            and stat.S_ISREG(final_target.st_mode)
            and stat.S_IMODE(final_target.st_mode) == 0o444,
            "published receipt changed during post-publication verification",
        )
        publication_complete = True
    except VerificationError:
        raise
    except OSError as error:
        raise VerificationError(f"cannot publish verification receipt: {target}") from error
    finally:
        if (
            linked_identity is not None
            and not publication_complete
            and directory_descriptor is not None
        ):
            try:
                current_target = os.stat(
                    target.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                current_target = None
            except OSError as error:
                cleanup_error = error
                current_target = None
            if (
                current_target is not None
                and (
                    current_target.st_dev,
                    current_target.st_ino,
                )
                == linked_identity
            ):
                try:
                    os.unlink(target.name, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
                except OSError as error:
                    cleanup_error = error
        if staging_name is not None and directory_descriptor is not None:
            try:
                os.unlink(staging_name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except FileNotFoundError:
                pass
            except OSError as error:
                cleanup_error = error
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if cleanup_error is not None:
            raise VerificationError("failed to clean an unpublished receipt") from cleanup_error
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _artifact_root(args.artifact_dir)
    requested_receipt = _absolute(args.receipt).resolve(strict=False)
    _require(
        requested_receipt != root and not requested_receipt.is_relative_to(root),
        "verification receipt must be outside the verified artifact directory",
    )
    receipt = verify_diffusion_corpus(
        config_path=args.config,
        sequences_path=args.sequences,
        assignments_path=args.assignments,
        artifact_dir=root,
    )
    write_receipt(
        args.receipt,
        receipt,
        artifact_dir=root,
        config_path=args.config,
        sequences_path=args.sequences,
        assignments_path=args.assignments,
    )
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())

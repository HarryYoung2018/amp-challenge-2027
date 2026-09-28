"""Independently verify the native-diffusion-v1 development projections.

This module is deliberately self contained.  In particular, it does not import
the projection producer, the diffusion data loader, the sequence helpers, or
the v1 contract parser.  It reconstructs every trainer and score row directly
from the authenticated development projection, the metadata-only assignment
ledger, and the parent TOML bytes.

Fold 4 is a locked holdout.  The verifier accepts its assignment metadata so it
can prove the census and exclusion boundary, but its API has no fold-4 sequence
input and no code path which can open one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

PARENT_CONTRACT_SHA256 = "4a13952a606e154b652f8bbc1e8f698b5993ee1a41181867c979729709573d40"
TRAINING_PROJECTION_SHA256 = "127a0eb88c5dc10c94904dcc5a3e98ff75a55890dc29af807e99f3f93c61ae46"
UNION_ASSIGNMENTS_SHA256 = "a8149e03bf2cdf15c70de723613f4b5367db5b7fc887a05b621d3b77834a56a0"

PROJECTION_ARTIFACT = "native_categorical_diffusion_v1_development_projection"
VERIFICATION_ARTIFACT = (
    "native_categorical_diffusion_v1_development_projection_independent_verification"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_WEIGHT_SUM_HEX = "0x1.0000000000000p+0"
_DEVELOPMENT_FOLDS = (0, 1, 2, 3)
_LOCKED_HOLDOUT_FOLD = 4
_TRAIN_FIELDS = ("sequence_id", "sequence", "sampling_weight")
_SCORE_FIELDS = (
    "schema_version",
    "sequence_id",
    "sequence",
    "fold",
    "homology_component_id",
    "union_component_id",
    "sampling_weight",
)
_ASSIGNMENT_FIELDS = frozenset(
    {"schema_version", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)
_TRAIN_FIELD_SET = frozenset(_TRAIN_FIELDS)
_SCORE_FIELD_SET = frozenset(_SCORE_FIELDS)

_CODE_PATHS = (
    "cluster/slurm/build_native_diffusion_v1_development_projections_twins.sbatch",
    "configs/diffusion/unconditional_v1.toml",
    "pyproject.toml",
    "src/amp_challenge/__init__.py",
    "src/amp_challenge/constants.py",
    "src/amp_challenge/generators/__init__.py",
    "src/amp_challenge/generators/diffusion/__init__.py",
    "src/amp_challenge/generators/diffusion/categorical.py",
    "src/amp_challenge/generators/diffusion/data.py",
    "src/amp_challenge/generators/diffusion/v1/__init__.py",
    "src/amp_challenge/generators/diffusion/v1/contract.py",
    "src/amp_challenge/generators/diffusion/v1/projection.py",
    "src/amp_challenge/generators/diffusion/v1/verify_projection.py",
    "src/amp_challenge/sequences.py",
    "uv.lock",
)

_OUTPUT_FILES = (
    "CODE_SHA256SUMS",
    "FROZEN_INPUT_SHA256SUMS",
    "SHA256SUMS",
    "folds/0/score.jsonl",
    "folds/0/train.jsonl",
    "folds/1/score.jsonl",
    "folds/1/train.jsonl",
    "folds/2/score.jsonl",
    "folds/2/train.jsonl",
    "folds/3/score.jsonl",
    "folds/3/train.jsonl",
    "manifest.json",
    "summary.json",
)
_TOP_MANIFEST_FILES = tuple(name for name in _OUTPUT_FILES if name != "SHA256SUMS")
_OUTPUT_DIRECTORIES = (".", "folds", "folds/0", "folds/1", "folds/2", "folds/3")


class VerificationError(ValueError):
    """Raised when a byte-level, provenance, or leakage invariant fails."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Stable bytes and filesystem identity for one regular file."""

    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    links: int

    @property
    def fingerprint(self) -> tuple[int, int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
            self.mode,
            self.links,
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    fold: int


@dataclass(frozen=True, slots=True)
class ProjectionRow:
    sequence_id: str
    sequence: str
    sampling_weight: float


@dataclass(frozen=True, slots=True)
class ContractView:
    snapshot: Snapshot
    alphabet: str
    min_length: int
    max_length: int
    levels: int
    cosine_offset: float
    development_folds: tuple[int, ...]
    locked_holdout_fold: int
    expected_sequences: int
    expected_development_sequences: int
    expected_locked_holdout_sequences: int
    expected_sequences_by_fold: tuple[int, ...]
    expected_development_homology_components: int
    expected_locked_holdout_homology_components: int
    expected_development_union_components: int
    expected_locked_holdout_union_components: int
    expected_outer_training_sequences: tuple[int, ...]
    expected_outer_score_sequences: tuple[int, ...]
    expected_outer_training_homology_components: tuple[int, ...]
    expected_outer_score_homology_components: tuple[int, ...]
    expected_outer_training_union_components: tuple[int, ...]
    expected_outer_score_union_components: tuple[int, ...]
    expected_score_selected_tokens: tuple[int, ...]
    expected_score_selected_token_bins: tuple[tuple[int, ...], ...]
    expected_louco_union_components: tuple[int, ...]
    expected_louco_homology_components: tuple[int, ...]
    expected_louco_selected_tokens: tuple[int, ...]
    expected_louco_selected_token_bins: tuple[tuple[int, ...], ...]


def _fail(message: str) -> NoReturn:
    raise VerificationError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink(path: Path, *, label: str, include_leaf: bool = True) -> None:
    target = _absolute(path)
    candidates = list(reversed(target.parents))
    if include_leaf:
        candidates.append(target)
    for candidate in candidates:
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise VerificationError(f"cannot inspect {label}: {candidate}") from error
        _require(
            not stat.S_ISLNK(observed.st_mode),
            f"{label} traverses a symbolic link: {candidate}",
        )


def _pread_exact(descriptor: int, size: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        _require(chunk != b"", f"{label} ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _read_snapshot(
    path: str | os.PathLike[str],
    *,
    label: str,
    expected_mode: int | None = None,
    require_single_link: bool = False,
) -> Snapshot:
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
        payload = _pread_exact(descriptor, before.st_size, label=label)
        after = os.fstat(descriptor)
        try:
            named = os.stat(requested, follow_symlinks=False)
        except OSError as error:
            raise VerificationError(f"{label} changed while being read") from error
    finally:
        os.close(descriptor)
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        stat.S_IMODE(before.st_mode),
        before.st_nlink,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        stat.S_IMODE(after.st_mode),
        after.st_nlink,
    )
    named_fingerprint = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
        stat.S_IMODE(named.st_mode),
        named.st_nlink,
    )
    _require(
        before_fingerprint == after_fingerprint == named_fingerprint
        and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    mode = stat.S_IMODE(before.st_mode)
    if expected_mode is not None:
        _require(mode == expected_mode, f"{label} mode must be {expected_mode:04o}")
    if require_single_link:
        _require(before.st_nlink == 1, f"{label} must not be hard-linked")
    return Snapshot(
        path=requested,
        payload=payload,
        sha256=_sha256(payload),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        mode=mode,
        links=before.st_nlink,
    )


def _unchanged(snapshot: Snapshot, *, label: str) -> None:
    current = _read_snapshot(
        snapshot.path,
        label=label,
        expected_mode=snapshot.mode,
        require_single_link=snapshot.links == 1,
    )
    _require(
        current.fingerprint == snapshot.fingerprint
        and current.sha256 == snapshot.sha256
        and current.payload == snapshot.payload,
        f"{label} changed during verification",
    )


def _directory_fingerprint(path: Path, *, label: str, expected_mode: int) -> tuple[int, int, int]:
    _require_no_symlink(path, label=label)
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise VerificationError(f"cannot inspect {label}: {path}") from error
    _require(stat.S_ISDIR(observed.st_mode), f"{label} is not a directory")
    _require(
        stat.S_IMODE(observed.st_mode) == expected_mode,
        f"{label} mode must be {expected_mode:04o}",
    )
    return (observed.st_dev, observed.st_ino, stat.S_IMODE(observed.st_mode))


def _walk_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = {"."}

    def visit(directory: Path, relative: PurePosixPath | None) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise VerificationError(f"cannot enumerate projection artifact: {directory}") from error
        for entry in entries:
            child_relative = (
                PurePosixPath(entry.name) if relative is None else relative / entry.name
            )
            name = child_relative.as_posix()
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise VerificationError(
                    f"cannot inspect projection artifact entry: {name}"
                ) from error
            _require(not stat.S_ISLNK(observed.st_mode), f"artifact contains symlink: {name}")
            if stat.S_ISDIR(observed.st_mode):
                directories.add(name)
                visit(Path(entry.path), child_relative)
            elif stat.S_ISREG(observed.st_mode):
                files.add(name)
            else:
                _fail(f"artifact contains non-regular entry: {name}")

    visit(root, None)
    return files, directories


def _snapshot_output_tree(
    artifact_dir: str | os.PathLike[str],
) -> tuple[Path, dict[str, Snapshot], dict[str, tuple[int, int, int]]]:
    root = _absolute(artifact_dir)
    initial_root = _directory_fingerprint(
        root,
        label="artifact root",
        expected_mode=0o555,
    )
    files, directories = _walk_tree(root)
    _require(files == set(_OUTPUT_FILES), "projection artifact file inventory is not exact")
    _require(
        directories == set(_OUTPUT_DIRECTORIES),
        "projection artifact directory inventory is not exact",
    )
    directory_snapshots = {
        name: _directory_fingerprint(
            root if name == "." else root / name,
            label=f"artifact directory {name}",
            expected_mode=0o555,
        )
        for name in _OUTPUT_DIRECTORIES
    }
    _require(
        directory_snapshots["."] == initial_root,
        "artifact root changed while its inventory was enumerated",
    )
    snapshots = {
        name: _read_snapshot(
            root / name,
            label=f"artifact file {name}",
            expected_mode=0o444,
            require_single_link=True,
        )
        for name in _OUTPUT_FILES
    }
    return root, snapshots, directory_snapshots


def _resnapshot_output_tree(
    root: Path,
    snapshots: Mapping[str, Snapshot],
    directories: Mapping[str, tuple[int, int, int]],
) -> None:
    files_now, directories_now = _walk_tree(root)
    _require(files_now == set(_OUTPUT_FILES), "artifact inventory changed during verification")
    _require(
        directories_now == set(_OUTPUT_DIRECTORIES),
        "artifact directory inventory changed during verification",
    )
    for name, original in snapshots.items():
        _unchanged(original, label=f"artifact file {name}")
    for name, original in directories.items():
        current = _directory_fingerprint(
            root if name == "." else root / name,
            label=f"artifact directory {name}",
            expected_mode=0o555,
        )
        _require(current == original, f"artifact directory {name} changed during verification")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"JSON contains forbidden non-finite constant {value}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise VerificationError("cannot serialize canonical JSON") from error
    return (text + "\n").encode("utf-8")


def _parse_json(payload: bytes, *, label: str) -> dict[str, Any]:
    _require(payload.endswith(b"\n"), f"{label} must end in exactly one LF")
    _require(b"\r" not in payload and b"\x00" not in payload, f"{label} has forbidden bytes")
    _require(not payload.endswith(b"\n\n"), f"{label} has a blank trailing line")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except VerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{label} is not strict UTF-8 JSON") from error
    _require(type(value) is dict, f"{label} must be a JSON object")
    _require(_canonical_json_bytes(value) == payload, f"{label} is not canonical JSON")
    return value


def _parse_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    _require(payload != b"" and payload.endswith(b"\n"), f"{label} must end in LF")
    _require(b"\r" not in payload and b"\x00" not in payload, f"{label} has forbidden bytes")
    raw_lines = payload.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for number, raw_line in enumerate(raw_lines, 1):
        _require(raw_line not in {b"", b"\n"}, f"{label} row {number} is blank")
        rows.append(_parse_json(raw_line, label=f"{label} row {number}"))
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("ascii")


def _safe_manifest_name(name: str) -> bool:
    if not name or "\x00" in name or "\n" in name or "\r" in name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _parse_sha_manifest(
    payload: bytes,
    *,
    label: str,
    expected_names: Sequence[str],
) -> dict[str, str]:
    _require(payload != b"" and payload.endswith(b"\n"), f"{label} must end in LF")
    _require(b"\r" not in payload and b"\x00" not in payload, f"{label} has forbidden bytes")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} must be ASCII") from error
    entries: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        _require(
            len(line) >= 67 and line[64:66] == "  ",
            f"{label} line {number} has invalid checksum syntax",
        )
        digest, name = line[:64], line[66:]
        _require(_SHA256_RE.fullmatch(digest) is not None, f"{label} has invalid SHA-256")
        _require(_safe_manifest_name(name), f"{label} has unsafe filename {name!r}")
        _require(name not in entries, f"{label} repeats filename {name!r}")
        entries[name] = digest
    _require(tuple(sorted(entries)) == tuple(expected_names), f"{label} inventory is not exact")
    _require(_sha_manifest_bytes(entries) == payload, f"{label} is not canonical and sorted")
    return entries


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(value)
    _require(
        observed == expected,
        f"{label} schema mismatch: missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )


def _table(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    _require(type(value) is dict, f"contract {name} must be a TOML table")
    return value


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    result = value
    if minimum is not None:
        _require(result >= minimum, f"{label} must be at least {minimum}")
    return result


def _float(value: object, *, label: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    return value


def _string(value: object, *, label: str) -> str:
    _require(type(value) is str, f"{label} must be a string")
    return value


def _integer_array(value: object, *, label: str, length: int | None = None) -> tuple[int, ...]:
    _require(type(value) is list and value, f"{label} must be a non-empty integer array")
    _require(all(type(item) is int for item in value), f"{label} must contain only integers")
    result = tuple(value)
    if length is not None:
        _require(len(result) == length, f"{label} must contain exactly {length} entries")
    return result


def _integer_matrix(
    value: object,
    *,
    label: str,
    rows: int,
    columns: int,
) -> tuple[tuple[int, ...], ...]:
    _require(type(value) is list and len(value) == rows, f"{label} must have {rows} rows")
    result = tuple(
        _integer_array(row, label=f"{label}[{index}]", length=columns)
        for index, row in enumerate(value)
    )
    return result


def _load_contract(path: str | os.PathLike[str]) -> ContractView:
    snapshot = _read_snapshot(path, label="parent v1 contract")
    _require(snapshot.sha256 == PARENT_CONTRACT_SHA256, "parent v1 contract SHA-256 mismatch")
    try:
        document = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("parent v1 contract is not strict UTF-8 TOML") from error
    _require(type(document) is dict, "parent v1 contract must be a mapping")
    _require(_integer(document.get("schema_version"), label="schema_version") == 1, "bad schema")
    _require(
        document.get("artifact") == "native_categorical_diffusion_unconditional_v1",
        "parent v1 contract artifact mismatch",
    )

    input_table = _table(document, "input")
    development = _table(document, "development")
    model = _table(document, "model")
    diffusion = _table(document, "diffusion")
    calibration = _table(document, "calibration")
    artifacts = _table(document, "artifacts")
    leakage = _table(document, "leakage")

    _require(
        input_table.get("training_projection_sha256") == TRAINING_PROJECTION_SHA256,
        "contract training projection hash mismatch",
    )
    _require(
        input_table.get("union_assignments_sha256") == UNION_ASSIGNMENTS_SHA256,
        "contract union-assignment hash mismatch",
    )
    development_folds = _integer_array(
        input_table.get("development_folds"), label="input.development_folds", length=4
    )
    locked_holdout_fold = _integer(
        input_table.get("locked_holdout_fold"), label="input.locked_holdout_fold"
    )
    _require(development_folds == _DEVELOPMENT_FOLDS, "development folds must be 0..3")
    _require(locked_holdout_fold == _LOCKED_HOLDOUT_FOLD, "locked holdout must be fold 4")
    _require(
        input_table.get("component_weighting")
        == "homology_component_equal_recomputed_within_each_fit_and_score_role",
        "contract role weighting is not frozen",
    )
    _require(
        input_table.get("model_input_fields") == list(_TRAIN_FIELDS),
        "contract trainer fields are not least privilege",
    )
    _require(
        development.get("outer_folds") == list(development_folds)
        and development.get("outer_training_rule") == "other_three_development_folds_only"
        and development.get("outer_score_rule") == "held_out_development_fold_only",
        "contract outer-fold rules are not frozen",
    )
    _require(
        development.get("calibration_crossfit")
        == "leave_one_union_component_out_within_each_outer_score_fold"
        and development.get("calibration_exclusion_unit") == "union_component_id",
        "contract LOUCO policy is not frozen",
    )
    _require(
        development.get("selected_token_count_definition")
        == "sum_levels_1_to_64_of_fixed_cosine_mask_count_per_sequence_per_replicate"
        and development.get("selected_token_mask_probability")
        == "clip_one_minus_cosine_alpha_bar_offset_0p008_at_level_divided_by_64"
        and development.get("selected_token_mask_count")
        == "minimum_length_maximum_one_ceil_length_times_mask_probability",
        "contract selected-token rules are not frozen",
    )
    alphabet = _string(model.get("alphabet"), label="model.alphabet")
    min_length = _integer(model.get("min_length"), label="model.min_length", minimum=1)
    max_length = _integer(model.get("max_length"), label="model.max_length", minimum=min_length)
    levels = _integer(diffusion.get("levels"), label="diffusion.levels", minimum=1)
    cosine_offset = _float(diffusion.get("cosine_offset"), label="diffusion.cosine_offset")
    _require(
        alphabet == _ALPHABET
        and levels == 64
        and cosine_offset == 0.008
        and diffusion.get("schedule") == "cosine_alpha_bar"
        and diffusion.get("mask_count") == "ceil_length_times_mask_probability",
        "contract sequence or diffusion schedule is not frozen",
    )
    _require(
        calibration.get("timestep_bins") == ["1_to_16", "17_to_32", "33_to_48", "49_to_64"],
        "contract timestep bins are not frozen",
    )
    _require(
        artifacts.get("canonical_json") is True
        and artifacts.get("reject_nonfinite_json") is True
        and artifacts.get("file_mode") == "0444"
        and artifacts.get("directory_mode") == "0555"
        and artifacts.get("manifest_published_last") is True
        and artifacts.get("semantic_manifests_path_free") is True
        and artifacts.get("independent_verifier_imports_producer") is False,
        "contract artifact policy is not frozen",
    )
    _require(
        leakage.get("trainer_allowed_fields") == list(_TRAIN_FIELDS)
        and leakage.get("fold4_visible_before_unlock") is False
        and leakage.get("fold4_used_for_calibration") is False,
        "contract leakage boundary is not frozen",
    )

    folds = len(development_folds)
    return ContractView(
        snapshot=snapshot,
        alphabet=alphabet,
        min_length=min_length,
        max_length=max_length,
        levels=levels,
        cosine_offset=cosine_offset,
        development_folds=development_folds,
        locked_holdout_fold=locked_holdout_fold,
        expected_sequences=_integer(
            input_table.get("expected_sequences"), label="expected_sequences"
        ),
        expected_development_sequences=_integer(
            input_table.get("expected_development_sequences"),
            label="expected_development_sequences",
        ),
        expected_locked_holdout_sequences=_integer(
            input_table.get("expected_locked_holdout_sequences"),
            label="expected_locked_holdout_sequences",
        ),
        expected_sequences_by_fold=_integer_array(
            input_table.get("expected_sequences_by_fold"),
            label="expected_sequences_by_fold",
            length=folds + 1,
        ),
        expected_development_homology_components=_integer(
            input_table.get("expected_development_homology_components"),
            label="expected_development_homology_components",
        ),
        expected_locked_holdout_homology_components=_integer(
            input_table.get("expected_locked_holdout_homology_components"),
            label="expected_locked_holdout_homology_components",
        ),
        expected_development_union_components=_integer(
            input_table.get("expected_development_union_components"),
            label="expected_development_union_components",
        ),
        expected_locked_holdout_union_components=_integer(
            input_table.get("expected_locked_holdout_union_components"),
            label="expected_locked_holdout_union_components",
        ),
        expected_outer_training_sequences=_integer_array(
            development.get("expected_outer_training_sequences"),
            label="expected_outer_training_sequences",
            length=folds,
        ),
        expected_outer_score_sequences=_integer_array(
            development.get("expected_outer_score_sequences"),
            label="expected_outer_score_sequences",
            length=folds,
        ),
        expected_outer_training_homology_components=_integer_array(
            development.get("expected_outer_training_homology_components"),
            label="expected_outer_training_homology_components",
            length=folds,
        ),
        expected_outer_score_homology_components=_integer_array(
            development.get("expected_outer_score_homology_components"),
            label="expected_outer_score_homology_components",
            length=folds,
        ),
        expected_outer_training_union_components=_integer_array(
            development.get("expected_outer_training_union_components"),
            label="expected_outer_training_union_components",
            length=folds,
        ),
        expected_outer_score_union_components=_integer_array(
            development.get("expected_outer_score_union_components"),
            label="expected_outer_score_union_components",
            length=folds,
        ),
        expected_score_selected_tokens=_integer_array(
            development.get("expected_score_selected_tokens_per_replicate_by_fold"),
            label="expected_score_selected_tokens_per_replicate_by_fold",
            length=folds,
        ),
        expected_score_selected_token_bins=_integer_matrix(
            development.get("expected_score_selected_tokens_per_replicate_by_fold_bin"),
            label="expected_score_selected_tokens_per_replicate_by_fold_bin",
            rows=folds,
            columns=4,
        ),
        expected_louco_union_components=_integer_array(
            development.get("minimum_louco_union_components_by_fold"),
            label="minimum_louco_union_components_by_fold",
            length=folds,
        ),
        expected_louco_homology_components=_integer_array(
            development.get("minimum_louco_homology_components_by_fold"),
            label="minimum_louco_homology_components_by_fold",
            length=folds,
        ),
        expected_louco_selected_tokens=_integer_array(
            development.get("minimum_louco_selected_tokens_per_replicate_by_fold"),
            label="minimum_louco_selected_tokens_per_replicate_by_fold",
            length=folds,
        ),
        expected_louco_selected_token_bins=_integer_matrix(
            development.get("minimum_louco_selected_tokens_per_replicate_by_fold_bin"),
            label="minimum_louco_selected_tokens_per_replicate_by_fold_bin",
            rows=folds,
            columns=4,
        ),
    )


def _validate_digest(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} invalid")
    return value


def _load_assignments(snapshot: Snapshot, contract: ContractView) -> tuple[Assignment, ...]:
    _require(snapshot.sha256 == UNION_ASSIGNMENTS_SHA256, "union assignments SHA-256 mismatch")
    raw_rows = _parse_jsonl(snapshot.payload, label="union assignments")
    _require(len(raw_rows) == contract.expected_sequences, "union assignment row census mismatch")
    rows: list[Assignment] = []
    previous: str | None = None
    homology_to_union: dict[str, str] = {}
    homology_to_fold: dict[str, int] = {}
    union_to_fold: dict[str, int] = {}
    for number, raw in enumerate(raw_rows, 1):
        _exact_keys(raw, set(_ASSIGNMENT_FIELDS), label=f"assignment row {number}")
        _require(raw["schema_version"] == 1 and type(raw["schema_version"]) is int, "bad schema")
        sequence_id = _validate_digest(raw["sequence_id"], label="assignment sequence_id")
        homology = _validate_digest(
            raw["homology_component_id"], label="assignment homology_component_id"
        )
        union = _validate_digest(raw["union_component_id"], label="assignment union_component_id")
        fold = _integer(raw["fold"], label="assignment fold")
        _require(fold in (*contract.development_folds, contract.locked_holdout_fold), "bad fold")
        _require(
            previous is None or sequence_id > previous, "assignments are not strictly ID-sorted"
        )
        previous = sequence_id
        prior_union = homology_to_union.setdefault(homology, union)
        prior_homology_fold = homology_to_fold.setdefault(homology, fold)
        prior_union_fold = union_to_fold.setdefault(union, fold)
        _require(prior_union == union, "one homology component spans union components")
        _require(prior_homology_fold == fold, "one homology component spans folds")
        _require(prior_union_fold == fold, "one union component spans folds")
        rows.append(Assignment(sequence_id, homology, union, fold))
    return tuple(rows)


def _load_projection(snapshot: Snapshot, contract: ContractView) -> tuple[ProjectionRow, ...]:
    _require(snapshot.sha256 == TRAINING_PROJECTION_SHA256, "training projection SHA-256 mismatch")
    raw_rows = _parse_jsonl(snapshot.payload, label="accepted training projection")
    _require(
        len(raw_rows) == contract.expected_development_sequences,
        "training projection row census mismatch",
    )
    rows: list[ProjectionRow] = []
    previous: str | None = None
    for number, raw in enumerate(raw_rows, 1):
        _exact_keys(raw, set(_TRAIN_FIELD_SET), label=f"training projection row {number}")
        sequence_id = _validate_digest(raw["sequence_id"], label="projection sequence_id")
        sequence = _string(raw["sequence"], label="projection sequence")
        _require(
            contract.min_length <= len(sequence) <= contract.max_length
            and sequence == sequence.upper()
            and all(residue in contract.alphabet for residue in sequence),
            f"projection sequence {sequence_id} is not canonical",
        )
        try:
            encoded = sequence.encode("ascii")
        except (
            UnicodeEncodeError
        ) as error:  # pragma: no cover - alphabet check normally catches this
            raise VerificationError("projection sequence is not ASCII") from error
        _require(_sha256(encoded) == sequence_id, "projection sequence ID does not match sequence")
        weight = raw["sampling_weight"]
        _require(
            type(weight) is float and math.isfinite(weight) and weight > 0.0,
            "projection sampling weight must be a positive finite JSON float",
        )
        _require(previous is None or sequence_id > previous, "projection is not strictly ID-sorted")
        previous = sequence_id
        rows.append(ProjectionRow(sequence_id, sequence, weight))
    return tuple(rows)


def _role_weights(assignments: Sequence[Assignment]) -> dict[str, float]:
    sizes = Counter(row.homology_component_id for row in assignments)
    component_count = len(sizes)
    _require(component_count > 0, "a projection role has no homology components")
    weights = {
        row.sequence_id: 1.0 / (component_count * sizes[row.homology_component_id])
        for row in assignments
    }
    total = math.fsum(weights[row.sequence_id] for row in assignments)
    _require(total.hex() == _WEIGHT_SUM_HEX, "role sampling weights do not sum to exact one")
    for component in sorted(sizes):
        mass = math.fsum(
            weights[row.sequence_id]
            for row in assignments
            if row.homology_component_id == component
        )
        _require(
            math.isclose(mass, 1.0 / component_count, rel_tol=0.0, abs_tol=1e-15),
            "homology-component sampling mass is not equal",
        )
    return weights


def _selected_token_bins(length: int, *, levels: int, offset: float) -> tuple[int, ...]:
    alpha_zero = math.cos((offset / (1.0 + offset)) * math.pi / 2.0) ** 2
    totals = [0, 0, 0, 0]
    bin_width = levels // 4
    _require(levels == bin_width * 4, "diffusion levels cannot form four exact bins")
    for level in range(1, levels + 1):
        alpha = (
            math.cos(((level / levels + offset) / (1.0 + offset)) * math.pi / 2.0) ** 2
        ) / alpha_zero
        probability = min(1.0, max(0.0, 1.0 - alpha))
        selected = min(length, max(1, math.ceil(length * probability)))
        totals[(level - 1) // bin_width] += selected
    return tuple(totals)


def _sequence_ids_sha256(sequence_ids: Sequence[str]) -> str:
    _require(tuple(sequence_ids) == tuple(sorted(sequence_ids)), "sequence IDs are not sorted")
    return _sha256(b"".join((sequence_id + "\n").encode("utf-8") for sequence_id in sequence_ids))


def _validate_censuses(
    contract: ContractView,
    assignments: Sequence[Assignment],
    projection: Sequence[ProjectionRow],
) -> tuple[dict[str, Assignment], dict[str, ProjectionRow]]:
    assignment_by_id = {row.sequence_id: row for row in assignments}
    projection_by_id = {row.sequence_id: row for row in projection}
    _require(len(assignment_by_id) == len(assignments), "assignment IDs are not unique")
    _require(len(projection_by_id) == len(projection), "projection IDs are not unique")
    folds = Counter(row.fold for row in assignments)
    _require(
        tuple(folds[index] for index in (*contract.development_folds, contract.locked_holdout_fold))
        == contract.expected_sequences_by_fold,
        "assignment fold census differs from contract",
    )
    _require(
        sum(folds.values())
        == contract.expected_sequences
        == contract.expected_development_sequences + contract.expected_locked_holdout_sequences,
        "assignment global census differs from contract",
    )
    development_ids = {
        row.sequence_id for row in assignments if row.fold in contract.development_folds
    }
    holdout_ids = {
        row.sequence_id for row in assignments if row.fold == contract.locked_holdout_fold
    }
    _require(
        set(projection_by_id) == development_ids,
        "training projection is not exactly folds 0..3",
    )
    _require(not set(projection_by_id) & holdout_ids, "fold-4 sequence leaked into projection")

    development_rows = [assignment_by_id[name] for name in sorted(development_ids)]
    holdout_rows = [assignment_by_id[name] for name in sorted(holdout_ids)]
    _require(
        len({row.homology_component_id for row in development_rows})
        == contract.expected_development_homology_components,
        "development homology-component census mismatch",
    )
    _require(
        len({row.homology_component_id for row in holdout_rows})
        == contract.expected_locked_holdout_homology_components,
        "fold-4 homology-component census mismatch",
    )
    _require(
        len({row.union_component_id for row in development_rows})
        == contract.expected_development_union_components,
        "development union-component census mismatch",
    )
    _require(
        len({row.union_component_id for row in holdout_rows})
        == contract.expected_locked_holdout_union_components,
        "fold-4 union-component census mismatch",
    )

    accepted_weights = _role_weights(development_rows)
    for row in projection:
        _require(
            row.sampling_weight == accepted_weights[row.sequence_id],
            "accepted projection sampling weight is not the independent component-equal value",
        )
    return assignment_by_id, projection_by_id


def _build_expected_fold(
    *,
    contract: ContractView,
    outer_fold: int,
    assignment_by_id: Mapping[str, Assignment],
    projection_by_id: Mapping[str, ProjectionRow],
) -> tuple[bytes, bytes, dict[str, object], dict[str, int | list[int]]]:
    train_folds = tuple(fold for fold in contract.development_folds if fold != outer_fold)
    train_assignments = tuple(
        assignment_by_id[name]
        for name in sorted(projection_by_id)
        if assignment_by_id[name].fold in train_folds
    )
    score_assignments = tuple(
        assignment_by_id[name]
        for name in sorted(projection_by_id)
        if assignment_by_id[name].fold == outer_fold
    )
    _require(train_assignments and score_assignments, f"outer fold {outer_fold} has an empty role")
    train_weights = _role_weights(train_assignments)
    score_weights = _role_weights(score_assignments)
    train_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for assignment in train_assignments:
        source = projection_by_id[assignment.sequence_id]
        train_rows.append(
            {
                "sequence_id": assignment.sequence_id,
                "sequence": source.sequence,
                "sampling_weight": train_weights[assignment.sequence_id],
            }
        )
    for assignment in score_assignments:
        source = projection_by_id[assignment.sequence_id]
        score_rows.append(
            {
                "schema_version": 1,
                "sequence_id": assignment.sequence_id,
                "sequence": source.sequence,
                "fold": assignment.fold,
                "homology_component_id": assignment.homology_component_id,
                "union_component_id": assignment.union_component_id,
                "sampling_weight": score_weights[assignment.sequence_id],
            }
        )

    token_bins_by_id = {
        row.sequence_id: _selected_token_bins(
            len(projection_by_id[row.sequence_id].sequence),
            levels=contract.levels,
            offset=contract.cosine_offset,
        )
        for row in score_assignments
    }
    score_bins = tuple(
        sum(token_bins_by_id[row.sequence_id][index] for row in score_assignments)
        for index in range(4)
    )
    score_total = sum(score_bins)
    union_rows: dict[str, list[Assignment]] = defaultdict(list)
    for row in score_assignments:
        union_rows[row.union_component_id].append(row)
    homology_all = {row.homology_component_id for row in score_assignments}
    louco_union = len(union_rows) - 1
    louco_homology_values: list[int] = []
    louco_total_values: list[int] = []
    louco_bin_values: list[tuple[int, ...]] = []
    for union_id in sorted(union_rows):
        heldout = union_rows[union_id]
        heldout_homology = {row.homology_component_id for row in heldout}
        heldout_bins = tuple(
            sum(token_bins_by_id[row.sequence_id][index] for row in heldout) for index in range(4)
        )
        remaining_bins = tuple(score_bins[index] - heldout_bins[index] for index in range(4))
        louco_homology_values.append(len(homology_all - heldout_homology))
        louco_total_values.append(sum(remaining_bins))
        louco_bin_values.append(remaining_bins)
    _require(louco_union > 0 and louco_homology_values, "LOUCO has no calibration support")
    louco = {
        "union_components": louco_union,
        "homology_components": min(louco_homology_values),
        "selected_tokens_per_replicate": min(louco_total_values),
        "selected_tokens_per_replicate_by_timestep_bin": [
            min(values[index] for values in louco_bin_values) for index in range(4)
        ],
    }
    train_ids = [row.sequence_id for row in train_assignments]
    score_ids = [row.sequence_id for row in score_assignments]
    summary = {
        "fold": outer_fold,
        "train": {
            "folds": list(train_folds),
            "rows": len(train_rows),
            "homology_components": len({row.homology_component_id for row in train_assignments}),
            "union_components": len({row.union_component_id for row in train_assignments}),
            "sequence_ids_sha256": _sequence_ids_sha256(train_ids),
            "sampling_weight_sum_hex": _WEIGHT_SUM_HEX,
        },
        "score": {
            "rows": len(score_rows),
            "homology_components": len({row.homology_component_id for row in score_assignments}),
            "union_components": len({row.union_component_id for row in score_assignments}),
            "sequence_ids_sha256": _sequence_ids_sha256(score_ids),
            "sampling_weight_sum_hex": _WEIGHT_SUM_HEX,
            "selected_tokens_per_replicate": score_total,
            "selected_tokens_per_replicate_by_timestep_bin": list(score_bins),
        },
        "louco_minimum": louco,
    }
    derived = {
        "train_sequences": len(train_rows),
        "score_sequences": len(score_rows),
        "train_homology": summary["train"]["homology_components"],  # type: ignore[index]
        "score_homology": summary["score"]["homology_components"],  # type: ignore[index]
        "train_union": summary["train"]["union_components"],  # type: ignore[index]
        "score_union": summary["score"]["union_components"],  # type: ignore[index]
        "score_tokens": score_total,
        "score_bins": list(score_bins),
        "louco_union": louco["union_components"],
        "louco_homology": louco["homology_components"],
        "louco_tokens": louco["selected_tokens_per_replicate"],
        "louco_bins": louco["selected_tokens_per_replicate_by_timestep_bin"],
    }
    return _jsonl_bytes(train_rows), _jsonl_bytes(score_rows), summary, derived


def _validate_fold_contract(
    contract: ContractView,
    derived: Sequence[Mapping[str, int | list[int]]],
) -> None:
    comparisons: tuple[tuple[str, tuple[object, ...], tuple[object, ...]], ...] = (
        (
            "outer training sequence",
            tuple(item["train_sequences"] for item in derived),
            contract.expected_outer_training_sequences,
        ),
        (
            "outer score sequence",
            tuple(item["score_sequences"] for item in derived),
            contract.expected_outer_score_sequences,
        ),
        (
            "outer training homology",
            tuple(item["train_homology"] for item in derived),
            contract.expected_outer_training_homology_components,
        ),
        (
            "outer score homology",
            tuple(item["score_homology"] for item in derived),
            contract.expected_outer_score_homology_components,
        ),
        (
            "outer training union",
            tuple(item["train_union"] for item in derived),
            contract.expected_outer_training_union_components,
        ),
        (
            "outer score union",
            tuple(item["score_union"] for item in derived),
            contract.expected_outer_score_union_components,
        ),
        (
            "score selected-token",
            tuple(item["score_tokens"] for item in derived),
            contract.expected_score_selected_tokens,
        ),
        (
            "score selected-token bin",
            tuple(tuple(item["score_bins"]) for item in derived),  # type: ignore[arg-type]
            contract.expected_score_selected_token_bins,
        ),
        (
            "LOUCO union",
            tuple(item["louco_union"] for item in derived),
            contract.expected_louco_union_components,
        ),
        (
            "LOUCO homology",
            tuple(item["louco_homology"] for item in derived),
            contract.expected_louco_homology_components,
        ),
        (
            "LOUCO selected-token",
            tuple(item["louco_tokens"] for item in derived),
            contract.expected_louco_selected_tokens,
        ),
        (
            "LOUCO selected-token bin",
            tuple(tuple(item["louco_bins"]) for item in derived),  # type: ignore[arg-type]
            contract.expected_louco_selected_token_bins,
        ),
    )
    for label, observed, expected in comparisons:
        _require(observed == expected, f"{label} census differs from parent contract")


def _expected_summary(
    contract: ContractView, fold_summaries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": PROJECTION_ARTIFACT,
        "status": "passed",
        "parent_contract_sha256": contract.snapshot.sha256,
        "coverage": {
            "development_sequences": contract.expected_development_sequences,
            "fold4_sequence_rows": 0,
            "score_occurrences_per_sequence": 1,
            "train_occurrences_per_sequence": 3,
        },
        "folds": [dict(item) for item in fold_summaries],
    }


def _expected_manifest(
    *,
    contract: ContractView,
    expected_git_commit: str,
    code_manifest: Snapshot,
    frozen_manifest: Snapshot,
    summary: Snapshot,
    fold_snapshots: Mapping[int, tuple[Snapshot, Snapshot]],
) -> dict[str, object]:
    fold_artifacts: list[dict[str, object]] = []
    for fold in contract.development_folds:
        train_snapshot, score_snapshot = fold_snapshots[fold]
        fold_artifacts.append(
            {
                "fold": fold,
                "train": {
                    "filename": f"folds/{fold}/train.jsonl",
                    "role": "trainer_input",
                    "rows": contract.expected_outer_training_sequences[fold],
                    "fields": list(_TRAIN_FIELDS),
                    "sha256": train_snapshot.sha256,
                },
                "score": {
                    "filename": f"folds/{fold}/score.jsonl",
                    "role": "evaluator_score_ledger",
                    "rows": contract.expected_outer_score_sequences[fold],
                    "fields": list(_SCORE_FIELDS),
                    "sha256": score_snapshot.sha256,
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact": PROJECTION_ARTIFACT,
        "status": "passed",
        "parent_contract_sha256": contract.snapshot.sha256,
        "input": {
            "training_projection": {
                "rows": contract.expected_development_sequences,
                "sha256": TRAINING_PROJECTION_SHA256,
            },
            "union_assignments": {
                "rows": contract.expected_sequences,
                "sha256": UNION_ASSIGNMENTS_SHA256,
            },
        },
        "provenance": {
            "git_commit": expected_git_commit,
            "code_sha256sums_sha256": code_manifest.sha256,
            "frozen_input_sha256sums_sha256": frozen_manifest.sha256,
        },
        "counts": {
            "development_sequences": contract.expected_development_sequences,
            "development_homology_components": contract.expected_development_homology_components,
            "development_union_components": contract.expected_development_union_components,
            "outer_training_sequences": list(contract.expected_outer_training_sequences),
            "outer_score_sequences": list(contract.expected_outer_score_sequences),
            "outer_training_homology_components": list(
                contract.expected_outer_training_homology_components
            ),
            "outer_score_homology_components": list(
                contract.expected_outer_score_homology_components
            ),
            "outer_training_union_components": list(
                contract.expected_outer_training_union_components
            ),
            "outer_score_union_components": list(contract.expected_outer_score_union_components),
        },
        "policies": {
            "development_folds": list(contract.development_folds),
            "locked_holdout_fold": contract.locked_holdout_fold,
            "outer_training_rule": "other_three_development_folds_only",
            "outer_score_rule": "held_out_development_fold_only",
            "row_order": "ascending_sequence_id",
            "trainer_visible_fields": list(_TRAIN_FIELDS),
            "score_visible_fields": list(_SCORE_FIELDS),
            "component_weighting": (
                "homology_component_equal_recomputed_within_each_fit_and_score_role"
            ),
            "role_weight_formula": (
                "1/(role_homology_component_count*role_homology_component_size)"
            ),
            "calibration_crossfit": ("leave_one_union_component_out_within_each_outer_score_fold"),
            "calibration_exclusion_unit": "union_component_id",
            "selected_token_count_definition": (
                "sum_levels_1_to_64_of_fixed_cosine_mask_count_per_sequence_per_replicate"
            ),
            "fold4_assignment_metadata_input": True,
            "fold4_sequence_input": False,
            "serialization": (
                "utf8_canonical_json_sorted_keys_compact_separators_ensure_ascii_false_"
                "allow_nan_false_single_lf"
            ),
        },
        "artifacts": {
            "summary": {
                "filename": "summary.json",
                "role": "semantic_summary",
                "sha256": summary.sha256,
            },
            "folds": fold_artifacts,
        },
    }


def _verify_code_manifest(
    snapshot: Snapshot,
    *,
    repository_root: str | os.PathLike[str],
    contract: ContractView,
) -> tuple[Snapshot, ...]:
    entries = _parse_sha_manifest(
        snapshot.payload,
        label="CODE_SHA256SUMS",
        expected_names=_CODE_PATHS,
    )
    root = _absolute(repository_root)
    _require_no_symlink(root, label="repository root")
    try:
        observed_root = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise VerificationError(f"cannot inspect repository root: {root}") from error
    _require(stat.S_ISDIR(observed_root.st_mode), "repository root is not a directory")
    code_snapshots: list[Snapshot] = []
    for name in _CODE_PATHS:
        source = root.joinpath(*PurePosixPath(name).parts)
        code = _read_snapshot(source, label=f"attested repository file {name}")
        _require(code.sha256 == entries[name], f"CODE_SHA256SUMS mismatch for {name}")
        code_snapshots.append(code)
    config_code = code_snapshots[_CODE_PATHS.index("configs/diffusion/unconditional_v1.toml")]
    _require(
        config_code.sha256 == contract.snapshot.sha256
        and config_code.payload == contract.snapshot.payload,
        "attested repository contract differs from supplied parent contract",
    )
    return tuple(code_snapshots)


def _verify_frozen_manifest(snapshot: Snapshot) -> None:
    expected = {
        "sequence_assignments.jsonl": UNION_ASSIGNMENTS_SHA256,
        "training_projection.jsonl": TRAINING_PROJECTION_SHA256,
        "unconditional_v1.toml": PARENT_CONTRACT_SHA256,
    }
    observed = _parse_sha_manifest(
        snapshot.payload,
        label="FROZEN_INPUT_SHA256SUMS",
        expected_names=tuple(sorted(expected)),
    )
    _require(observed == expected, "FROZEN_INPUT_SHA256SUMS does not bind exact inputs")


def _assert_disjoint_inputs(
    artifact_root: Path,
    inputs: Sequence[Snapshot],
) -> None:
    identities: set[tuple[int, int]] = set()
    for snapshot in inputs:
        identity = (snapshot.device, snapshot.inode)
        _require(identity not in identities, "verification inputs alias the same inode")
        identities.add(identity)
        try:
            snapshot.path.relative_to(artifact_root)
        except ValueError:
            pass
        else:
            _fail("verification input is inside the artifact being verified")


def verify_development_projection(
    *,
    parent_contract_path: str | os.PathLike[str],
    training_projection_path: str | os.PathLike[str],
    assignments_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    expected_git_commit: str,
) -> dict[str, object]:
    """Reconstruct and verify the immutable v1 development-projection tree.

    The returned dictionary is a path-free deterministic receipt.  This
    function performs no writes.
    """

    _require(
        type(expected_git_commit) is str
        and _GIT_COMMIT_RE.fullmatch(expected_git_commit) is not None,
        "expected Git commit must be 40 lowercase hexadecimal characters",
    )
    root, output, directories = _snapshot_output_tree(artifact_dir)
    contract = _load_contract(parent_contract_path)
    training_snapshot = _read_snapshot(training_projection_path, label="training projection input")
    assignment_snapshot = _read_snapshot(assignments_path, label="union assignment input")
    _assert_disjoint_inputs(root, (contract.snapshot, training_snapshot, assignment_snapshot))

    top_entries = _parse_sha_manifest(
        output["SHA256SUMS"].payload,
        label="SHA256SUMS",
        expected_names=_TOP_MANIFEST_FILES,
    )
    for name in _TOP_MANIFEST_FILES:
        _require(top_entries[name] == output[name].sha256, f"SHA256SUMS mismatch for {name}")

    _verify_frozen_manifest(output["FROZEN_INPUT_SHA256SUMS"])
    code_snapshots = _verify_code_manifest(
        output["CODE_SHA256SUMS"],
        repository_root=repository_root,
        contract=contract,
    )

    assignments = _load_assignments(assignment_snapshot, contract)
    projection = _load_projection(training_snapshot, contract)
    assignment_by_id, projection_by_id = _validate_censuses(contract, assignments, projection)

    fold_summaries: list[dict[str, object]] = []
    derived: list[dict[str, int | list[int]]] = []
    fold_snapshots: dict[int, tuple[Snapshot, Snapshot]] = {}
    score_occurrences: Counter[str] = Counter()
    train_occurrences: Counter[str] = Counter()
    for fold in contract.development_folds:
        expected_train, expected_score, fold_summary, fold_derived = _build_expected_fold(
            contract=contract,
            outer_fold=fold,
            assignment_by_id=assignment_by_id,
            projection_by_id=projection_by_id,
        )
        train_snapshot = output[f"folds/{fold}/train.jsonl"]
        score_snapshot = output[f"folds/{fold}/score.jsonl"]
        _require(
            train_snapshot.payload == expected_train,
            f"fold {fold} trainer projection differs from independent reconstruction",
        )
        _require(
            score_snapshot.payload == expected_score,
            f"fold {fold} score ledger differs from independent reconstruction",
        )
        # Parse even byte-identical reconstructions to make the least-privilege
        # and occurrence checks explicit verifier invariants.
        train_rows = _parse_jsonl(train_snapshot.payload, label=f"fold {fold} train")
        score_rows = _parse_jsonl(score_snapshot.payload, label=f"fold {fold} score")
        for row in train_rows:
            _exact_keys(row, set(_TRAIN_FIELD_SET), label=f"fold {fold} train row")
            train_occurrences[row["sequence_id"]] += 1
        for row in score_rows:
            _exact_keys(row, set(_SCORE_FIELD_SET), label=f"fold {fold} score row")
            _require(row["fold"] == fold and type(row["fold"]) is int, "score fold mismatch")
            score_occurrences[row["sequence_id"]] += 1
        fold_summaries.append(fold_summary)
        derived.append(fold_derived)
        fold_snapshots[fold] = (train_snapshot, score_snapshot)

    _validate_fold_contract(contract, derived)
    _require(
        set(score_occurrences) == set(projection_by_id) and set(score_occurrences.values()) == {1},
        "each development sequence must occur in exactly one score ledger",
    )
    _require(
        set(train_occurrences) == set(projection_by_id) and set(train_occurrences.values()) == {3},
        "each development sequence must occur in exactly three trainer projections",
    )

    expected_summary = _expected_summary(contract, fold_summaries)
    _require(
        output["summary.json"].payload == _canonical_json_bytes(expected_summary),
        "summary.json differs from independent reconstruction",
    )
    _parse_json(output["summary.json"].payload, label="summary.json")
    expected_manifest = _expected_manifest(
        contract=contract,
        expected_git_commit=expected_git_commit,
        code_manifest=output["CODE_SHA256SUMS"],
        frozen_manifest=output["FROZEN_INPUT_SHA256SUMS"],
        summary=output["summary.json"],
        fold_snapshots=fold_snapshots,
    )
    _require(
        output["manifest.json"].payload == _canonical_json_bytes(expected_manifest),
        "manifest.json differs from independent reconstruction",
    )
    _parse_json(output["manifest.json"].payload, label="manifest.json")

    for snapshot in (contract.snapshot, training_snapshot, assignment_snapshot, *code_snapshots):
        _unchanged(snapshot, label="verification input")
    _resnapshot_output_tree(root, output, directories)

    return {
        "schema_version": 1,
        "artifact": VERIFICATION_ARTIFACT,
        "status": "passed",
        "verified_artifact": PROJECTION_ARTIFACT,
        "parent_contract_sha256": contract.snapshot.sha256,
        "git_commit": expected_git_commit,
        "input": {
            "training_projection": {
                "rows": len(projection),
                "sha256": training_snapshot.sha256,
            },
            "union_assignments": {
                "rows": len(assignments),
                "sha256": assignment_snapshot.sha256,
            },
        },
        "output": {
            "files": len(_OUTPUT_FILES),
            "sha256sums_sha256": output["SHA256SUMS"].sha256,
            "manifest_sha256": output["manifest.json"].sha256,
            "summary_sha256": output["summary.json"].sha256,
        },
        "coverage": {
            "development_sequences": len(projection),
            "fold4_assignment_rows": contract.expected_locked_holdout_sequences,
            "fold4_sequence_rows": 0,
            "score_occurrences_per_sequence": 1,
            "train_occurrences_per_sequence": 3,
        },
        "checks": {
            "byte_exact_reconstruction": True,
            "code_and_input_hashes_verified": True,
            "exact_inventory_and_modes": True,
            "fold4_sequences_opened": False,
            "louco_support_recomputed": True,
            "producer_imported": False,
            "role_local_weights_recomputed": True,
            "selected_tokens_recomputed": True,
        },
    }


def _rollback_created_receipt(
    parent_descriptor: int,
    filename: str,
    created_identity: tuple[int, int],
) -> None:
    """Remove only the directory entry that still names our newly created inode."""

    try:
        observed = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        return
    if (observed.st_dev, observed.st_ino) != created_identity:
        return
    with suppress(OSError):
        os.unlink(filename, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def write_receipt(path: str | os.PathLike[str], receipt: Mapping[str, object]) -> Path:
    """Publish a new immutable canonical receipt without overwriting a path."""

    target = _absolute(path)
    parent = target.parent
    _require_no_symlink(parent, label="receipt parent")
    _require_no_symlink(target, label="receipt target")
    _require(not target.exists(), f"refusing to overwrite verification receipt: {target}")
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError as error:
        raise VerificationError(f"cannot inspect receipt parent: {parent}") from error
    _require(stat.S_ISDIR(parent_stat.st_mode), "receipt parent is not a directory")
    payload = _canonical_json_bytes(dict(receipt))
    file_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor: int | None = None
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        try:
            parent_descriptor = os.open(parent, directory_flags)
            opened_parent = os.fstat(parent_descriptor)
            _require(
                stat.S_ISDIR(opened_parent.st_mode)
                and (opened_parent.st_dev, opened_parent.st_ino)
                == (parent_stat.st_dev, parent_stat.st_ino),
                "receipt parent changed before publication",
            )
            descriptor = os.open(
                target.name,
                file_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            created = os.fstat(descriptor)
            _require(stat.S_ISREG(created.st_mode), "created receipt is not a regular file")
            created_identity = (created.st_dev, created.st_ino)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "receipt write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        except VerificationError:
            raise
        except OSError as error:
            raise VerificationError(f"cannot publish verification receipt: {target}") from error
        published = _read_snapshot(
            target,
            label="published verification receipt",
            expected_mode=0o444,
            require_single_link=True,
        )
        _require(published.payload == payload, "published verification receipt bytes changed")
        _require(
            (published.device, published.inode) == created_identity,
            "published verification receipt changed inode",
        )
        try:
            os.fsync(parent_descriptor)
        except OSError as error:
            raise VerificationError(f"cannot fsync receipt parent: {parent}") from error
        return target
    except BaseException:
        if parent_descriptor is not None and created_identity is not None:
            _rollback_created_receipt(parent_descriptor, target.name, created_identity)
        raise
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if parent_descriptor is not None:
            with suppress(OSError):
                os.close(parent_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--training-projection", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact_root = _absolute(arguments.artifact_dir)
    receipt_path = _absolute(arguments.receipt)
    try:
        _require(
            receipt_path != artifact_root and not receipt_path.is_relative_to(artifact_root),
            "verification receipt must be outside the verified artifact tree",
        )
        receipt = verify_development_projection(
            parent_contract_path=arguments.parent_contract,
            training_projection_path=arguments.training_projection,
            assignments_path=arguments.assignments,
            artifact_dir=artifact_root,
            repository_root=arguments.repository_root,
            expected_git_commit=arguments.expected_git_commit,
        )
        write_receipt(receipt_path, receipt)
    except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"AMP native diffusion v1 projection verification error: {error}", file=sys.stderr)
        return 2
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

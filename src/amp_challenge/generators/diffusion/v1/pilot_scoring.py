"""Deterministic scoring, calibration, and selection for the v1 R128 pilot.

This module is deliberately NumPy-only. It consumes evaluator-visible score
rows, a sealed C0 archive, and archived all-class residual logits; it never
imports a trainer or performs neural inference. Its objects are numerical
intermediates, never authorization evidence; only the pending independent
artifact verifier may issue a scientific pilot decision.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    CONFIG_SHA256,
    PARENT_CONFIG_SHA256,
    NativeDiffusionV1PilotContract,
)
from amp_challenge.generators.diffusion.v1.pilot_data import AuthenticatedCountPrior
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = 20
MASK_TOKEN = 21
MAX_LENGTH = 50
LEVELS = 64
REPLICATES = 1
EVALUATION_SEED = 20260905
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPLICATES = 10_000
CHECKPOINT_STEPS = (250, 500, 1000, 2000, 4000)
TIMESTEP_BINS = ((1, 16), (17, 32), (33, 48), (49, 64))
RESIDUAL_LAMBDAS = (0.125, 0.25, 0.5, 0.75, 1.0)
TEMPERATURES = (1.0, 1.25, 1.5, 2.0, 3.0)
ECE_BINS = 15

_SCORE_FIELDS = frozenset(
    {
        "schema_version",
        "sequence_id",
        "sequence",
        "fold",
        "homology_component_id",
        "union_component_id",
        "sampling_weight",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NPZ_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_SEED_DOMAIN = b"amp-challenge/native-categorical-diffusion/stateless-seed/v1\0"
_CASE_DOMAIN = b"amp-challenge/native-categorical-diffusion/validation-case/v1\0"
_BOOTSTRAP_TRANSCRIPT_DOMAIN = (
    b"amp-challenge/native-categorical-diffusion/v1/pilot-bootstrap-draws/v1\0"
)
_DOS_EPOCH = (1980, 1, 1, 0, 0, 0)
_MAX_SCORE_BYTES = 64 * 1024 * 1024
FloatArray = NDArray[np.float64]
_ARRAY_INTEGRITY_DOMAIN = b"amp-challenge/native-diffusion-v1/array-integrity/v1\0"
_LEDGER_ARRAY_NAMES = (
    "sequence_id",
    "homology_component_id",
    "union_component_id",
    "length",
    "sampling_weight",
    "clean_tokens",
    "attention_mask",
    "case_id",
    "row_index",
    "level",
    "replicate",
    "row_seed",
    "mask_count",
    "corrupted_tokens",
    "selected_mask",
)
_ARCHIVE_ARRAY_NAMES = (
    "case_id",
    "case_offsets",
    "position",
    "target_token",
    "count_log_probability",
    "checkpoint_step",
    "residual_logit",
)
_VERIFIED_EVIDENCE_BINDING_FIELDS = (
    "release_receipt_sha256",
    "trainer_bundle_digest_map_sha256",
    "evaluator_bundle_digest_map_sha256",
    "checkpoint_digest_map_sha256",
    "count_prior_digest_map_sha256",
    "producer_reinference_sha256",
    "cpu_reconstruction_sha256",
)
_VERIFIED_EVIDENCE_CHECK_FIELDS = (
    "all_four_outer_fits",
    "release_chain_verified",
    "trainer_bundles_verified",
    "evaluator_bundles_verified",
    "count_priors_reconstructed",
    "checkpoints_authenticated",
    "producer_reinference_exact",
    "cpu_reconstruction_exact",
)


class _MetricFactoryToken:
    __slots__ = ("used",)

    def __init__(self) -> None:
        self.used = False


def _consume_metric_factory_token(token: object) -> None:
    if type(token) is not _MetricFactoryToken or token.used:
        raise TypeError("metric objects may only be created by scoring factories")
    token.used = True


def _raw_arrays(
    owner: object,
    names: Sequence[str],
) -> dict[str, NDArray[np.generic]]:
    return {name: object.__getattribute__(owner, name) for name in names}


def _owning_readonly_copy(raw: NDArray[np.generic]) -> NDArray[np.generic]:
    copied = raw.copy(order="C")
    copied.flags.writeable = False
    return copied


def _defensive_array(raw: NDArray[np.generic]) -> NDArray[np.generic]:
    return raw.copy(order="C")


def _arrays_integrity_sha256(
    arrays: Mapping[str, NDArray[np.generic]],
) -> str:
    digest = hashlib.sha256(_ARRAY_INTEGRITY_DOMAIN)
    for name, raw in arrays.items():
        if type(name) is not str or type(raw) is not np.ndarray or not raw.flags.c_contiguous:
            raise TypeError("array integrity input must be named C-contiguous ndarrays")
        for value in (name.encode("ascii"), raw.dtype.str.encode("ascii")):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        digest.update(len(raw.shape).to_bytes(8, "big"))
        for dimension in raw.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(raw.nbytes.to_bytes(8, "big"))
        digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


def _require_owned_readonly_arrays(
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    label: str,
) -> None:
    if any(
        type(raw) is not np.ndarray
        or not raw.flags.owndata
        or not raw.flags.c_contiguous
        or raw.flags.writeable
        for raw in arrays.values()
    ):
        raise ValueError(f"{label} arrays must remain owning, C-contiguous, and read-only")


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_uint64(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value


def _require_finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite canonical float")
    return value


def _canonical_json_bytes(value: object) -> bytes:
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            observed = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect score projection path: {candidate}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"score projection path traverses a symlink: {candidate}")


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


def _read_regular_bytes(path: str | os.PathLike[str]) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect score projection: {source}") from error
    if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
        raise ValueError("score projection must be a single-link regular file")
    if not 0 < named_before.st_size <= _MAX_SCORE_BYTES:
        raise ValueError("score projection has an invalid byte size")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open score projection: {source}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(65536, _MAX_SCORE_BYTES + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_SCORE_BYTES:
                raise ValueError("score projection exceeds its maximum byte size")
        after = os.fstat(descriptor)
        try:
            named_after = os.lstat(source)
        except OSError as error:
            raise ValueError("score projection changed while it was read") from error
        _reject_symlink_chain(source)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        _fingerprint(named_before) != _fingerprint(before)
        or _fingerprint(before) != _fingerprint(after)
        or _fingerprint(before) != _fingerprint(named_after)
        or not stat.S_ISREG(named_after.st_mode)
        or before.st_nlink != 1
        or len(payload) != before.st_size
    ):
        raise ValueError("score projection changed while it was read")
    return payload


@dataclass(frozen=True, slots=True)
class ScoreRow:
    """One exact evaluator-visible development score row."""

    sequence_id: str
    sequence: str
    fold: int
    homology_component_id: str
    union_component_id: str
    sampling_weight: float

    def __post_init__(self) -> None:
        _require_sha256(self.sequence_id, label="score sequence_id")
        _require_sha256(self.homology_component_id, label="score homology_component_id")
        _require_sha256(self.union_component_id, label="score union_component_id")
        if type(self.fold) is not int or self.fold not in (0, 1, 2, 3):
            raise ValueError("score fold must be an integer in 0..3")
        sequence = canonicalize_sequence(self.sequence, min_length=8, max_length=50)
        if sequence != self.sequence or canonical_sequence_id(sequence) != self.sequence_id:
            raise ValueError("score sequence is not canonical or does not match sequence_id")
        if _require_finite_float(self.sampling_weight, label="score sampling_weight") <= 0.0:
            raise ValueError("score sampling_weight must be positive")

    def canonical_record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "homology_component_id": self.homology_component_id,
            "sampling_weight": self.sampling_weight,
            "schema_version": 1,
            "sequence": self.sequence,
            "sequence_id": self.sequence_id,
            "union_component_id": self.union_component_id,
        }


def parse_score_rows(
    payload: bytes,
    *,
    expected_sha256: str,
    outer_fold: int,
    expected_rows: int | None = None,
    expected_homology_components: int | None = None,
    expected_union_components: int | None = None,
) -> tuple[ScoreRow, ...]:
    """Authenticate and parse one canonical score JSONL projection."""

    if type(payload) is not bytes or not payload or len(payload) > _MAX_SCORE_BYTES:
        raise ValueError("score projection payload must be non-empty bounded bytes")
    expected = _require_sha256(expected_sha256, label="expected score projection SHA-256")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"score projection SHA-256 expected {expected}, got {observed}")
    if type(outer_fold) is not int or outer_fold not in (0, 1, 2, 3):
        raise ValueError("outer_fold must be an exact integer in 0..3")
    if not payload.endswith(b"\n") or b"\r" in payload or b"\x00" in payload:
        raise ValueError("score projection must be LF-terminated canonical JSONL")
    rows: list[ScoreRow] = []
    for number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            document = json.loads(
                line[:-1].decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"score projection row {number} is invalid JSON") from error
        if type(document) is not dict or _canonical_json_bytes(document) != line:
            raise ValueError(f"score projection row {number} is not canonical JSON")
        if set(document) != _SCORE_FIELDS:
            raise ValueError(f"score projection row {number} has an invalid schema")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError(f"score projection row {number} schema_version must equal 1")
        if type(document["sequence"]) is not str:
            raise TypeError(f"score projection row {number} sequence must be an exact string")
        if type(document["sampling_weight"]) is not float:
            raise TypeError(f"score projection row {number} sampling_weight must be an exact float")
        row = ScoreRow(
            sequence_id=document["sequence_id"],
            sequence=document["sequence"],
            fold=document["fold"],
            homology_component_id=document["homology_component_id"],
            union_component_id=document["union_component_id"],
            sampling_weight=document["sampling_weight"],
        )
        if row.fold != outer_fold:
            raise ValueError("score projection contains a row from a different outer fold")
        rows.append(row)
    values = tuple(rows)
    identifiers = tuple(row.sequence_id for row in values)
    if not values or identifiers != tuple(sorted(identifiers)):
        raise ValueError("score rows must be non-empty and ordered by sequence_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("score sequence IDs must be unique")
    if expected_rows is not None and (
        type(expected_rows) is not int or len(values) != expected_rows
    ):
        raise ValueError("score row census differs from the expected contract")
    homology_groups: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in values:
        homology_groups[row.homology_component_id].append(row)
    if expected_homology_components is not None and (
        type(expected_homology_components) is not int
        or len(homology_groups) != expected_homology_components
    ):
        raise ValueError("score homology-component census differs from the contract")
    union_count = len({row.union_component_id for row in values})
    if expected_union_components is not None and (
        type(expected_union_components) is not int or union_count != expected_union_components
    ):
        raise ValueError("score union-component census differs from the contract")
    component_count = len(homology_groups)
    for group in homology_groups.values():
        expected_weight = 1.0 / (component_count * len(group))
        if any(row.sampling_weight.hex() != expected_weight.hex() for row in group):
            raise ValueError("score sampling_weight differs from the exact role-local formula")
    if math.fsum(row.sampling_weight for row in values).hex() != (1.0).hex():
        raise ValueError("score sampling weights must sum to exactly binary64 one")
    return values


def load_score_rows(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    outer_fold: int,
    expected_rows: int | None = None,
    expected_homology_components: int | None = None,
    expected_union_components: int | None = None,
) -> tuple[ScoreRow, ...]:
    """Read and strictly validate one authenticated score projection."""

    return parse_score_rows(
        _read_regular_bytes(path),
        expected_sha256=expected_sha256,
        outer_fold=outer_fold,
        expected_rows=expected_rows,
        expected_homology_components=expected_homology_components,
        expected_union_components=expected_union_components,
    )


def load_contract_fold_score_rows(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    path: str | os.PathLike[str],
) -> tuple[ScoreRow, ...]:
    """Load one score projection using only its authenticated fold pins."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    fold = contract.fold(outer_fold)
    return load_score_rows(
        path,
        expected_sha256=fold.score_sha256,
        outer_fold=outer_fold,
        expected_rows=fold.score_rows,
        expected_homology_components=fold.score_homology_components,
        expected_union_components=fold.score_union_components,
    )


def load_contract_score_corruption_ledger(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    path: str | os.PathLike[str],
) -> ScoreCorruptionLedger:
    """Load and deterministically corrupt one exact contract score fold."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be an exact NativeDiffusionV1PilotContract")
    contract.revalidate()
    fold = contract.fold(outer_fold)
    rows = load_contract_fold_score_rows(contract, outer_fold, path)
    ledger = build_score_corruption_ledger(
        rows,
        parent_contract_sha256=contract.parent_config_sha256,
        evaluation_seed=contract.table("evaluation")["evaluation_seed"],
    )
    arrays = ledger._validated_arrays()
    selected_by_bin = tuple(
        int(
            np.sum(
                arrays["mask_count"][(arrays["level"] >= first) & (arrays["level"] <= last)],
                dtype=np.uint64,
            )
        )
        for first, last in TIMESTEP_BINS
    )
    if (
        len(arrays["case_id"]) != fold.score_cases
        or int(np.sum(arrays["mask_count"], dtype=np.uint64)) != fold.score_selected_tokens
        or selected_by_bin != fold.score_selected_tokens_by_timestep_bin
    ):
        raise ValueError("contract score corruption census differs from the authenticated fold")
    return ledger


def namespaced_seed(root_seed: int, namespace: str, *parts: str | int) -> int:
    """Derive the contract's first-eight-byte big-endian SHA-256 uint64."""

    root = _require_uint64(root_seed, label="root_seed")
    if type(namespace) is not str or not namespace:
        raise ValueError("namespace must be a non-empty exact string")
    payload = bytearray(_SEED_DOMAIN)

    def append(tag: bytes, value: bytes) -> None:
        payload.extend(tag)
        payload.extend(len(value).to_bytes(8, "big"))
        payload.extend(value)

    append(b"r", root.to_bytes(8, "big"))
    append(b"n", namespace.encode("utf-8"))
    for part in parts:
        if type(part) is str:
            append(b"s", part.encode("utf-8"))
        elif type(part) is int:
            append(b"i", str(part).encode("ascii"))
        else:
            raise TypeError("seed parts must be exact strings or integers")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validation_case_id(
    parent_contract_sha256: str,
    sequence_id: str,
    level: int,
    replicate: int,
    row_seed: int,
) -> str:
    """Return the exact path-free validation-case identity."""

    parent = _require_sha256(parent_contract_sha256, label="parent contract SHA-256")
    if parent != PARENT_CONFIG_SHA256:
        raise ValueError("corruption ledger parent contract digest differs from the frozen parent")
    sequence = _require_sha256(sequence_id, label="sequence_id")
    if type(level) is not int or not 1 <= level <= 2**16 - 1:
        raise ValueError("level must be an exact uint16 in 1..65535")
    if type(replicate) is not int or not 0 <= replicate <= 2**16 - 1:
        raise ValueError("replicate must be an exact uint16")
    seed = _require_uint64(row_seed, label="row_seed")
    digest = hashlib.sha256()
    digest.update(_CASE_DOMAIN)
    for value in (
        parent.encode("ascii"),
        sequence.encode("ascii"),
        level.to_bytes(2, "big"),
        replicate.to_bytes(2, "big"),
        seed.to_bytes(8, "big"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _cosine_mask_count(length: int, level: int) -> int:
    levels = np.asarray([level], dtype="<i8")
    lengths = np.asarray([length], dtype="<i8")
    offset = 0.008
    timestep = levels.astype(np.float64) / float(LEVELS)
    angle_zero = offset / (1.0 + offset) * np.pi / 2.0
    angle = (timestep + offset) / (1.0 + offset) * np.pi / 2.0
    probability = np.clip(
        1.0 - np.square(np.cos(angle)) / np.square(np.cos(angle_zero)),
        0.0,
        1.0,
    )
    return int(np.minimum(lengths, np.maximum(np.ceil(probability * lengths), 1))[0])


@dataclass(frozen=True, slots=True)
class ScoreCorruptionLedger:
    """Exact evaluator corruption arrays in contract member order."""

    rows: tuple[ScoreRow, ...]
    parent_contract_sha256: str
    sequence_id: NDArray[np.bytes_]
    homology_component_id: NDArray[np.bytes_]
    union_component_id: NDArray[np.bytes_]
    length: NDArray[np.uint8]
    sampling_weight: NDArray[np.float64]
    clean_tokens: NDArray[np.uint8]
    attention_mask: NDArray[np.bool_]
    case_id: NDArray[np.bytes_]
    row_index: NDArray[np.uint16]
    level: NDArray[np.uint8]
    replicate: NDArray[np.uint8]
    row_seed: NDArray[np.uint64]
    mask_count: NDArray[np.uint8]
    corrupted_tokens: NDArray[np.uint8]
    selected_mask: NDArray[np.bool_]
    arrays_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_corruption_ledger(self)
        for name, raw in _raw_arrays(self, _LEDGER_ARRAY_NAMES).items():
            object.__setattr__(self, name, _owning_readonly_copy(raw))
        _validate_corruption_ledger(self)
        object.__setattr__(
            self,
            "arrays_sha256",
            _arrays_integrity_sha256(_raw_arrays(self, _LEDGER_ARRAY_NAMES)),
        )

    def __getattribute__(self, name: str) -> object:
        if name in _LEDGER_ARRAY_NAMES:
            raw = object.__getattribute__(self, name)
            try:
                object.__getattribute__(self, "arrays_sha256")
            except AttributeError:
                return raw
            self.revalidate()
            return _defensive_array(raw)
        return object.__getattribute__(self, name)

    def revalidate(self) -> None:
        arrays = _raw_arrays(self, _LEDGER_ARRAY_NAMES)
        _require_owned_readonly_arrays(arrays, label="corruption ledger")
        expected = _require_sha256(
            object.__getattribute__(self, "arrays_sha256"),
            label="corruption ledger array integrity SHA-256",
        )
        if _arrays_integrity_sha256(arrays) != expected:
            raise ValueError("corruption ledger arrays changed after construction")
        _validate_corruption_ledger(self)

    def _validated_arrays(self) -> dict[str, NDArray[np.generic]]:
        self.revalidate()
        return _raw_arrays(self, _LEDGER_ARRAY_NAMES)

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {name: _defensive_array(raw) for name, raw in self._validated_arrays().items()}

    def npz_bytes(self) -> bytes:
        """Return the deterministic score-corruption artifact bytes."""

        return deterministic_npz_bytes(corruption_npz_schema(len(self.rows)), self.arrays())


def build_score_corruption_ledger(
    rows: Sequence[ScoreRow],
    *,
    parent_contract_sha256: str,
    evaluation_seed: int = EVALUATION_SEED,
) -> ScoreCorruptionLedger:
    """Build the single-replicate, all-level fixed score corruption ledger."""

    values = tuple(rows)
    if not values or any(type(row) is not ScoreRow for row in values):
        raise TypeError("rows must be a non-empty sequence of exact ScoreRow values")
    identifiers = tuple(row.sequence_id for row in values)
    if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(identifiers):
        raise ValueError("score rows must have unique ascending sequence IDs")
    folds = {row.fold for row in values}
    if len(folds) != 1:
        raise ValueError("one corruption ledger may contain exactly one outer fold")
    parent = _require_sha256(parent_contract_sha256, label="parent contract SHA-256")
    if parent != PARENT_CONFIG_SHA256:
        raise ValueError("corruption ledger parent contract digest differs from the frozen parent")
    seed_root = _require_uint64(evaluation_seed, label="evaluation_seed")
    if seed_root != EVALUATION_SEED:
        raise ValueError("pilot score corruptions require evaluation seed 20260905")
    row_count = len(values)
    case_count = row_count * LEVELS * REPLICATES
    sequence_id = np.asarray([row.sequence_id.encode("ascii") for row in values], dtype="|S64")
    homology_id = np.asarray(
        [row.homology_component_id.encode("ascii") for row in values], dtype="|S64"
    )
    union_id = np.asarray([row.union_component_id.encode("ascii") for row in values], dtype="|S64")
    lengths = np.asarray([len(row.sequence) for row in values], dtype="|u1")
    weights = np.asarray([row.sampling_weight for row in values], dtype="<f8")
    clean = np.full((row_count, MAX_LENGTH), PAD_TOKEN, dtype="|u1")
    attention = np.zeros((row_count, MAX_LENGTH), dtype="|b1")
    residue_index = {symbol: index for index, symbol in enumerate(ALPHABET)}
    for row_index_value, row in enumerate(values):
        encoded = [residue_index[symbol] for symbol in row.sequence]
        clean[row_index_value, : len(encoded)] = encoded
        attention[row_index_value, : len(encoded)] = True
    case_ids = np.empty(case_count, dtype="|S64")
    row_indices = np.empty(case_count, dtype="<u2")
    case_levels = np.empty(case_count, dtype="|u1")
    replicates = np.zeros(case_count, dtype="|u1")
    row_seeds = np.empty(case_count, dtype="<u8")
    mask_counts = np.empty(case_count, dtype="|u1")
    corrupted = np.empty((case_count, MAX_LENGTH), dtype="|u1")
    selected = np.zeros((case_count, MAX_LENGTH), dtype="|b1")
    case_index = 0
    for score_row_index, row in enumerate(values):
        length = len(row.sequence)
        for level in range(1, LEVELS + 1):
            replicate = 0
            row_seed = namespaced_seed(
                seed_root,
                "validation",
                parent,
                row.sequence_id,
                level,
                replicate,
            )
            count = _cosine_mask_count(length, level)
            rng = np.random.Generator(np.random.PCG64(row_seed))
            positions = rng.choice(
                np.arange(length, dtype=np.int64),
                size=count,
                replace=False,
            )
            case_ids[case_index] = validation_case_id(
                parent, row.sequence_id, level, replicate, row_seed
            ).encode("ascii")
            row_indices[case_index] = score_row_index
            case_levels[case_index] = level
            row_seeds[case_index] = row_seed
            mask_counts[case_index] = count
            corrupted[case_index] = clean[score_row_index]
            selected[case_index, positions] = True
            corrupted[case_index, positions] = MASK_TOKEN
            case_index += 1
    result = ScoreCorruptionLedger(
        rows=values,
        parent_contract_sha256=parent,
        sequence_id=sequence_id,
        homology_component_id=homology_id,
        union_component_id=union_id,
        length=lengths,
        sampling_weight=weights,
        clean_tokens=clean,
        attention_mask=attention,
        case_id=case_ids,
        row_index=row_indices,
        level=case_levels,
        replicate=replicates,
        row_seed=row_seeds,
        mask_count=mask_counts,
        corrupted_tokens=corrupted,
        selected_mask=selected,
    )
    return result


def _validate_corruption_ledger(ledger: ScoreCorruptionLedger) -> None:
    parent = _require_sha256(
        ledger.parent_contract_sha256,
        label="ledger parent contract SHA-256",
    )
    if parent != PARENT_CONFIG_SHA256:
        raise ValueError("ledger parent contract digest differs from the frozen parent")
    if (
        type(ledger.rows) is not tuple
        or not ledger.rows
        or any(type(row) is not ScoreRow for row in ledger.rows)
    ):
        raise TypeError("ledger rows must be a non-empty tuple of ScoreRow values")
    identifiers = tuple(row.sequence_id for row in ledger.rows)
    if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(identifiers):
        raise ValueError("ledger rows must have unique ascending sequence IDs")
    if len({row.fold for row in ledger.rows}) != 1:
        raise ValueError("ledger rows must belong to one outer fold")
    homology_groups: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in ledger.rows:
        homology_groups[row.homology_component_id].append(row)
    component_count = len(homology_groups)
    for group in homology_groups.values():
        expected_weight = 1.0 / (component_count * len(group))
        if any(row.sampling_weight.hex() != expected_weight.hex() for row in group):
            raise ValueError("ledger row weights differ from the score-role formula")
    if math.fsum(row.sampling_weight for row in ledger.rows).hex() != (1.0).hex():
        raise ValueError("ledger row weights must sum to binary64 one")
    row_count = len(ledger.rows)
    case_count = row_count * LEVELS
    arrays = _raw_arrays(ledger, _LEDGER_ARRAY_NAMES)
    schema = (
        ("sequence_id", "|S64", (row_count,)),
        ("homology_component_id", "|S64", (row_count,)),
        ("union_component_id", "|S64", (row_count,)),
        ("length", "|u1", (row_count,)),
        ("sampling_weight", "<f8", (row_count,)),
        ("clean_tokens", "|u1", (row_count, MAX_LENGTH)),
        ("attention_mask", "|b1", (row_count, MAX_LENGTH)),
        ("case_id", "|S64", (case_count,)),
        ("row_index", "<u2", (case_count,)),
        ("level", "|u1", (case_count,)),
        ("replicate", "|u1", (case_count,)),
        ("row_seed", "<u8", (case_count,)),
        ("mask_count", "|u1", (case_count,)),
        ("corrupted_tokens", "|u1", (case_count, MAX_LENGTH)),
        ("selected_mask", "|b1", (case_count, MAX_LENGTH)),
    )
    for name, dtype, shape in schema:
        raw = arrays[name]
        if type(raw) is not np.ndarray or raw.dtype != np.dtype(dtype) or raw.shape != shape:
            raise TypeError(f"corruption member {name} violates its exact dtype or shape")
        if not raw.flags.c_contiguous:
            raise ValueError(f"corruption member {name} must be C-contiguous")
    expected_sequence_id = np.asarray(
        [row.sequence_id.encode("ascii") for row in ledger.rows],
        dtype="|S64",
    )
    expected_homology_id = np.asarray(
        [row.homology_component_id.encode("ascii") for row in ledger.rows],
        dtype="|S64",
    )
    expected_union_id = np.asarray(
        [row.union_component_id.encode("ascii") for row in ledger.rows],
        dtype="|S64",
    )
    expected_lengths = np.asarray(
        [len(row.sequence) for row in ledger.rows],
        dtype="|u1",
    )
    expected_weights = np.asarray(
        [row.sampling_weight for row in ledger.rows],
        dtype="<f8",
    )
    if not all(
        np.array_equal(observed, expected)
        for observed, expected in (
            (arrays["sequence_id"], expected_sequence_id),
            (arrays["homology_component_id"], expected_homology_id),
            (arrays["union_component_id"], expected_union_id),
            (arrays["length"], expected_lengths),
            (arrays["sampling_weight"], expected_weights),
        )
    ):
        raise ValueError("corruption row metadata differs from the exact score rows")
    expected_clean = np.full((row_count, MAX_LENGTH), PAD_TOKEN, dtype="|u1")
    expected_attention = np.zeros((row_count, MAX_LENGTH), dtype="|b1")
    residue_index = {symbol: index for index, symbol in enumerate(ALPHABET)}
    for index, row in enumerate(ledger.rows):
        tokens = [residue_index[symbol] for symbol in row.sequence]
        expected_clean[index, : len(tokens)] = tokens
        expected_attention[index, : len(tokens)] = True
    if not np.array_equal(arrays["clean_tokens"], expected_clean) or not np.array_equal(
        arrays["attention_mask"], expected_attention
    ):
        raise ValueError("corruption clean tokens or attention mask differs from score rows")
    if not np.array_equal(
        np.sum(arrays["selected_mask"], axis=1, dtype=np.uint64),
        arrays["mask_count"].astype(np.uint64),
    ):
        raise ValueError("selected-mask census differs from mask_count")
    if np.any(arrays["selected_mask"] & ~arrays["attention_mask"][arrays["row_index"]]):
        raise ValueError("a corruption selected a padding position")
    expected_corrupted = arrays["clean_tokens"][arrays["row_index"]].copy()
    expected_corrupted[arrays["selected_mask"]] = MASK_TOKEN
    if not np.array_equal(expected_corrupted, arrays["corrupted_tokens"]):
        raise ValueError("corrupted tokens differ from clean tokens plus selected masks")
    expected_case_row = np.repeat(np.arange(row_count, dtype="<u2"), LEVELS)
    expected_levels = np.tile(np.arange(1, LEVELS + 1, dtype="|u1"), row_count)
    if (
        not np.array_equal(arrays["row_index"], expected_case_row)
        or not np.array_equal(arrays["level"], expected_levels)
        or np.any(arrays["replicate"] != 0)
    ):
        raise ValueError("corruption cases violate canonical row/level/replicate order")
    for case_index in range(case_count):
        row_index = int(arrays["row_index"][case_index])
        row = ledger.rows[row_index]
        level = int(arrays["level"][case_index])
        row_seed = namespaced_seed(
            EVALUATION_SEED,
            "validation",
            parent,
            row.sequence_id,
            level,
            0,
        )
        if int(arrays["row_seed"][case_index]) != row_seed:
            raise ValueError("corruption row seed differs from its stateless key")
        expected_case_id = validation_case_id(
            parent,
            row.sequence_id,
            level,
            0,
            row_seed,
        ).encode("ascii")
        if arrays["case_id"][case_index] != expected_case_id:
            raise ValueError("corruption case ID differs from its framed identity")
        count = _cosine_mask_count(len(row.sequence), level)
        if int(arrays["mask_count"][case_index]) != count:
            raise ValueError("corruption mask count differs from the cosine schedule")
        rng = np.random.Generator(np.random.PCG64(row_seed))
        positions = rng.choice(
            np.arange(len(row.sequence), dtype=np.int64),
            size=count,
            replace=False,
        )
        expected_selected = np.zeros(MAX_LENGTH, dtype="|b1")
        expected_selected[positions] = True
        if not np.array_equal(arrays["selected_mask"][case_index], expected_selected):
            raise ValueError("corruption selected positions differ from the PCG64 ledger")


@dataclass(frozen=True, slots=True)
class NpzArraySpec:
    """One exact endian-qualified deterministic NPZ member."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NPZ_NAME_RE.fullmatch(self.name) is None:
            raise ValueError("NPZ member name must be a lowercase identifier")
        if type(self.dtype) is not str or self.dtype not in {
            "|S64",
            "|u1",
            "|b1",
            "<u2",
            "<u8",
            "<f4",
            "<f8",
        }:
            raise ValueError("NPZ member dtype is outside the frozen portable set")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise ValueError("NPZ member shape must contain positive exact integers")


def _validated_npz_schema(
    schema: Sequence[NpzArraySpec],
) -> tuple[NpzArraySpec, ...]:
    entries = tuple(schema)
    if not entries or any(type(item) is not NpzArraySpec for item in entries):
        raise TypeError("schema must contain exact NpzArraySpec values")
    if len({item.name for item in entries}) != len(entries):
        raise ValueError("NPZ schema member names must be unique")
    if len(entries) >= 2**16:
        raise ValueError("NPZ schema has too many members for the frozen non-ZIP64 format")
    return entries


def _npy_v1_prefix(item: NpzArraySpec) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        output,
        {
            "descr": np.dtype(item.dtype).str,
            "fortran_order": False,
            "shape": item.shape,
        },
    )
    prefix = output.getvalue()
    if not prefix.startswith(b"\x93NUMPY\x01\x00"):
        raise RuntimeError("NumPy did not produce the frozen NPY-v1 header")
    return prefix


def _npz_member_layout(
    entries: tuple[NpzArraySpec, ...],
) -> tuple[tuple[NpzArraySpec, bytes, bytes, int], ...]:
    layout: list[tuple[NpzArraySpec, bytes, bytes, int]] = []
    local_bytes = 0
    central_bytes = 0
    for item in entries:
        filename = f"{item.name}.npy".encode("ascii")
        prefix = _npy_v1_prefix(item)
        data_bytes = math.prod(item.shape) * np.dtype(item.dtype).itemsize
        member_bytes = len(prefix) + data_bytes
        if member_bytes > zipfile.ZIP64_LIMIT:
            raise ValueError(f"NPZ member {item.name} exceeds the frozen non-ZIP64 size bound")
        layout.append((item, filename, prefix, member_bytes))
        local_bytes += 30 + len(filename) + member_bytes
        central_bytes += 46 + len(filename)
    if local_bytes > zipfile.ZIP64_LIMIT or central_bytes > zipfile.ZIP64_LIMIT:
        raise ValueError("NPZ schema exceeds the frozen non-ZIP64 archive bounds")
    return tuple(layout)


def _preinspect_deterministic_npz(
    payload: bytes,
    entries: tuple[NpzArraySpec, ...],
) -> None:
    """Reject unsafe or non-canonical ZIP/NPY metadata before ``np.load``."""

    layout = _npz_member_layout(entries)
    expected_size = 22 + sum(
        76 + 2 * len(filename) + member_bytes for _, filename, _, member_bytes in layout
    )
    if len(payload) != expected_size:
        raise ValueError("NPZ payload byte size differs from its exact schema bound")

    local_header = struct.Struct("<4s5H3I2H")
    central_header = struct.Struct("<4s6H3I5H2I")
    end_record = struct.Struct("<4s4H2IH")
    expected_external_attr = (stat.S_IFREG | 0o444) << 16
    expected_dos_date = (1 << 5) | 1
    local_offsets: list[int] = []
    member_crcs: list[int] = []
    cursor = 0

    for item, filename, npy_prefix, member_bytes in layout:
        local_offsets.append(cursor)
        try:
            (
                signature,
                extract_version,
                flags,
                compression,
                modified_time,
                modified_date,
                crc32,
                compressed_size,
                uncompressed_size,
                filename_size,
                extra_size,
            ) = local_header.unpack_from(payload, cursor)
        except struct.error as error:
            raise ValueError("NPZ local header is truncated") from error
        if (
            signature != b"PK\x03\x04"
            or extract_version != 20
            or flags != 0
            or compression != zipfile.ZIP_STORED
            or modified_time != 0
            or modified_date != expected_dos_date
            or compressed_size != member_bytes
            or uncompressed_size != member_bytes
            or filename_size != len(filename)
            or extra_size != 0
        ):
            raise ValueError(f"NPZ member {item.name} has unsafe local ZIP metadata")
        name_start = cursor + local_header.size
        data_start = name_start + filename_size
        data_stop = data_start + member_bytes
        if payload[name_start:data_start] != filename:
            raise ValueError("NPZ local member inventory or order differs from its schema")
        member_payload = memoryview(payload)[data_start:data_stop]
        if (
            len(member_payload) != member_bytes
            or member_payload[: len(npy_prefix)].tobytes() != npy_prefix
        ):
            raise ValueError(f"NPZ member {item.name} is not the exact canonical NPY-v1 payload")
        observed_crc = zlib.crc32(member_payload) & 0xFFFFFFFF
        if crc32 != observed_crc:
            raise ValueError(f"NPZ member {item.name} has an invalid stored-data CRC")
        member_crcs.append(observed_crc)
        cursor = data_stop

    central_start = cursor
    for index, ((item, filename, _, member_bytes), local_offset, crc32) in enumerate(
        zip(layout, local_offsets, member_crcs, strict=True)
    ):
        try:
            (
                signature,
                create_version,
                extract_version,
                flags,
                compression,
                modified_time,
                modified_date,
                central_crc32,
                compressed_size,
                uncompressed_size,
                filename_size,
                extra_size,
                comment_size,
                disk_start,
                internal_attr,
                external_attr,
                observed_local_offset,
            ) = central_header.unpack_from(payload, cursor)
        except struct.error as error:
            raise ValueError("NPZ central-directory header is truncated") from error
        if (
            signature != b"PK\x01\x02"
            or create_version != (3 << 8) | 20
            or extract_version != 20
            or flags != 0
            or compression != zipfile.ZIP_STORED
            or modified_time != 0
            or modified_date != expected_dos_date
            or central_crc32 != crc32
            or compressed_size != member_bytes
            or uncompressed_size != member_bytes
            or filename_size != len(filename)
            or extra_size != 0
            or comment_size != 0
            or disk_start != 0
            or internal_attr != 0
            or external_attr != expected_external_attr
            or observed_local_offset != local_offset
        ):
            raise ValueError(f"NPZ member {item.name} has unsafe central ZIP metadata")
        name_start = cursor + central_header.size
        name_stop = name_start + filename_size
        if payload[name_start:name_stop] != filename:
            raise ValueError("NPZ central member inventory or order differs from its schema")
        cursor = name_stop
        if index + 1 < len(layout) and payload[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("NPZ central directory contains an unexpected record")

    central_size = cursor - central_start
    try:
        (
            signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            observed_central_size,
            observed_central_start,
            archive_comment_size,
        ) = end_record.unpack_from(payload, cursor)
    except struct.error as error:
        raise ValueError("NPZ end-of-central-directory record is truncated") from error
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_disk != 0
        or disk_entries != len(layout)
        or total_entries != len(layout)
        or observed_central_size != central_size
        or observed_central_start != central_start
        or archive_comment_size != 0
        or cursor + end_record.size != len(payload)
    ):
        raise ValueError("NPZ archive has an unsafe or non-canonical end record")

    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r", allowZip64=False) as archive:
            members = archive.infolist()
            if archive.comment != b"" or archive.start_dir != central_start:
                raise ValueError("NPZ archive comment or central offset is not canonical")
            if tuple(member.filename for member in members) != tuple(
                filename.decode("ascii") for _, filename, _, _ in layout
            ):
                raise ValueError("NPZ member inventory or order differs from its schema")
            for member, (_, filename, _, member_bytes), local_offset, crc32 in zip(
                members,
                layout,
                local_offsets,
                member_crcs,
                strict=True,
            ):
                if (
                    member.orig_filename != filename.decode("ascii")
                    or member.date_time != _DOS_EPOCH
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.flag_bits != 0
                    or member.create_system != 3
                    or member.create_version != 20
                    or member.extract_version != 20
                    or member.internal_attr != 0
                    or member.external_attr != expected_external_attr
                    or member.extra != b""
                    or member.comment != b""
                    or member.file_size != member_bytes
                    or member.compress_size != member_bytes
                    or member.header_offset != local_offset
                    or crc32 != member.CRC
                    or member.is_dir()
                ):
                    raise ValueError("NPZ archive has unsafe ZIP member metadata")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError("NPZ payload is not a valid bounded non-ZIP64 archive") from error


def deterministic_npz_bytes(
    schema: Sequence[NpzArraySpec],
    arrays: Mapping[str, NDArray[np.generic]],
) -> bytes:
    """Serialize fixed-order ZIP_STORED NPY-v1.0 members without pickle."""

    entries = _validated_npz_schema(schema)
    _npz_member_layout(entries)
    if type(arrays) is not dict:
        raise TypeError("NPZ arrays must be a plain insertion-ordered dict")
    names = tuple(item.name for item in entries)
    if tuple(arrays) != names:
        raise ValueError("NPZ arrays must follow the exact schema member order")
    prepared: list[tuple[str, NDArray[np.generic]]] = []
    for item in entries:
        raw = arrays[item.name]
        if type(raw) is not np.ndarray:
            raise TypeError(f"NPZ member {item.name} must be an exact ndarray")
        if raw.dtype != np.dtype(item.dtype):
            raise TypeError(
                f"NPZ member {item.name} dtype must be {item.dtype}, got {raw.dtype.str}"
            )
        if raw.shape != item.shape:
            raise ValueError(f"NPZ member {item.name} must have shape {item.shape}")
        if raw.dtype.hasobject:
            raise TypeError("object arrays are forbidden in pilot NPZ artifacts")
        if raw.dtype.kind == "f" and not bool(np.isfinite(raw).all()):
            raise ValueError(f"NPZ member {item.name} contains a non-finite value")
        prepared.append((item.name, np.ascontiguousarray(raw)))
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for name, values in prepared:
            member = io.BytesIO()
            np.lib.format.write_array(member, values, version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_DOS_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.internal_attr = 0
            info.flag_bits = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def load_deterministic_npz_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
    schema: Sequence[NpzArraySpec],
) -> dict[str, NDArray[np.generic]]:
    """Authenticate and load one exact deterministic, pickle-free NPZ archive."""

    if type(payload) is not bytes or not payload:
        raise ValueError("NPZ payload must be non-empty exact bytes")
    expected = _require_sha256(expected_sha256, label="expected NPZ SHA-256")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("NPZ payload SHA-256 differs from its sealed digest")
    entries = _validated_npz_schema(schema)
    _preinspect_deterministic_npz(payload, entries)
    arrays: dict[str, NDArray[np.generic]] = {}
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            if tuple(loaded.files) != tuple(item.name for item in entries):
                raise ValueError("NPZ member inventory or order differs from its schema")
            for item in entries:
                arrays[item.name] = loaded[item.name].copy()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("NPZ payload is not valid pickle-free archive data") from error
    if deterministic_npz_bytes(entries, arrays) != payload:
        raise ValueError("NPZ payload bytes are not in the deterministic frozen encoding")
    for values in arrays.values():
        values.flags.writeable = False
    return arrays


def corruption_npz_schema(row_count: int) -> tuple[NpzArraySpec, ...]:
    """Return the concrete contract schema for one score-corruption archive."""

    if type(row_count) is not int or not 1 <= row_count <= 2**16 - 1:
        raise ValueError("row_count must be an exact positive uint16")
    case_count = row_count * LEVELS
    return (
        NpzArraySpec("sequence_id", "|S64", (row_count,)),
        NpzArraySpec("homology_component_id", "|S64", (row_count,)),
        NpzArraySpec("union_component_id", "|S64", (row_count,)),
        NpzArraySpec("length", "|u1", (row_count,)),
        NpzArraySpec("sampling_weight", "<f8", (row_count,)),
        NpzArraySpec("clean_tokens", "|u1", (row_count, MAX_LENGTH)),
        NpzArraySpec("attention_mask", "|b1", (row_count, MAX_LENGTH)),
        NpzArraySpec("case_id", "|S64", (case_count,)),
        NpzArraySpec("row_index", "<u2", (case_count,)),
        NpzArraySpec("level", "|u1", (case_count,)),
        NpzArraySpec("replicate", "|u1", (case_count,)),
        NpzArraySpec("row_seed", "<u8", (case_count,)),
        NpzArraySpec("mask_count", "|u1", (case_count,)),
        NpzArraySpec("corrupted_tokens", "|u1", (case_count, MAX_LENGTH)),
        NpzArraySpec("selected_mask", "|b1", (case_count, MAX_LENGTH)),
    )


def count_prior_npz_schema() -> tuple[NpzArraySpec, ...]:
    """Return the frozen contract schema for ``count_prior.npz``."""

    return (
        NpzArraySpec("effective_count_scale", "<f8", (1,)),
        NpzArraySpec("unigram_probability", "<f8", (20,)),
        NpzArraySpec("length_edges", "|u1", (6,)),
        NpzArraySpec("relative_position_probability", "<f8", (5, 10, 20)),
        NpzArraySpec("log_relative_position_probability", "<f8", (5, 10, 20)),
    )


def residual_logits_npz_schema(
    case_count: int,
    selected_token_count: int,
    *,
    checkpoint_count: int = len(CHECKPOINT_STEPS),
) -> tuple[NpzArraySpec, ...]:
    """Return the concrete contract schema for archived scoring logits."""

    for label, value in (
        ("case_count", case_count),
        ("selected_token_count", selected_token_count),
        ("checkpoint_count", checkpoint_count),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive exact integer")
    return (
        NpzArraySpec("case_id", "|S64", (case_count,)),
        NpzArraySpec("case_offsets", "<u8", (case_count + 1,)),
        NpzArraySpec("position", "|u1", (selected_token_count,)),
        NpzArraySpec("target_token", "|u1", (selected_token_count,)),
        NpzArraySpec("count_log_probability", "<f8", (selected_token_count, 20)),
        NpzArraySpec("checkpoint_step", "<u2", (checkpoint_count,)),
        NpzArraySpec(
            "residual_logit",
            "<f4",
            (checkpoint_count, selected_token_count, 20),
        ),
    )


def pilot_bootstrap_npz_schema() -> tuple[NpzArraySpec, ...]:
    """Return the frozen contract schema for ``pilot_bootstrap.npz``."""

    return (
        NpzArraySpec("checkpoint_step", "<u2", (len(CHECKPOINT_STEPS),)),
        NpzArraySpec("r128_mean_nll", "<f8", (len(CHECKPOINT_STEPS),)),
        NpzArraySpec("c0_mean_nll", "<f8", (1,)),
        NpzArraySpec("c0t_mean_nll", "<f8", (1,)),
        NpzArraySpec("selected_comparator_index", "|u1", (1,)),
        NpzArraySpec(
            "r128_bootstrap_mean_nll",
            "<f8",
            (len(CHECKPOINT_STEPS), BOOTSTRAP_REPLICATES),
        ),
        NpzArraySpec(
            "comparator_bootstrap_mean_nll",
            "<f8",
            (BOOTSTRAP_REPLICATES,),
        ),
        NpzArraySpec(
            "relative_improvement",
            "<f8",
            (len(CHECKPOINT_STEPS), BOOTSTRAP_REPLICATES),
        ),
    )


def gather_count_log_probability(
    ledger: ScoreCorruptionLedger,
    log_relative_position_probability: NDArray[np.float64],
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.float64],
]:
    """Gather sealed C0 cells in case/ascending-position order.

    The log table must come from the byte-verified trainer count-prior archive.
    This helper only indexes it; it never refits or renormalizes C0 from score
    rows.
    """

    if type(ledger) is not ScoreCorruptionLedger:
        raise TypeError("ledger must be an exact ScoreCorruptionLedger")
    ledger_arrays = ledger._validated_arrays()
    table = log_relative_position_probability
    if (
        type(table) is not np.ndarray
        or table.dtype != np.dtype("<f8")
        or table.shape != (5, 10, 20)
        or not table.flags.c_contiguous
        or table.flags.writeable
        or not bool(np.isfinite(table).all())
    ):
        raise TypeError("sealed C0 log table must be read-only finite C-contiguous <f8 [5,10,20]")
    selected_count = int(math.fsum(int(value) for value in ledger_arrays["mask_count"]))
    offsets = np.empty(len(ledger_arrays["case_id"]) + 1, dtype="<u8")
    offsets[0] = 0
    np.cumsum(ledger_arrays["mask_count"], dtype=np.uint64, out=offsets[1:])
    positions_out = np.empty(selected_count, dtype="|u1")
    targets = np.empty(selected_count, dtype="|u1")
    count_log = np.empty((selected_count, 20), dtype="<f8")
    length_edges = np.asarray([8, 15, 20, 25, 33, 51], dtype="|u1")
    cursor = 0
    for case_index, row_index_raw in enumerate(ledger_arrays["row_index"]):
        row_index = int(row_index_raw)
        length = int(ledger_arrays["length"][row_index])
        length_bin = int(np.searchsorted(length_edges, length, side="right") - 1)
        positions = np.flatnonzero(ledger_arrays["selected_mask"][case_index])
        stop = cursor + len(positions)
        positions_out[cursor:stop] = positions.astype("|u1")
        targets[cursor:stop] = ledger_arrays["clean_tokens"][row_index, positions]
        for local_index, position_raw in enumerate(positions):
            position = int(position_raw)
            position_bin = min(9, (10 * position) // length)
            count_log[cursor + local_index] = table[length_bin, position_bin]
        cursor = stop
    if cursor != selected_count:
        raise RuntimeError("C0 gather produced an inconsistent selected-token census")
    return offsets, positions_out, targets, count_log


@dataclass(frozen=True, slots=True)
class ScoringArchive:
    """Validated linkage between a fixed corruption ledger, C0, and raw logits."""

    ledger: ScoreCorruptionLedger
    count_prior: AuthenticatedCountPrior = field(repr=False)
    count_prior_sha256: str = field(init=False)
    sealed_log_relative_position_probability: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    case_id: NDArray[np.bytes_]
    case_offsets: NDArray[np.uint64]
    position: NDArray[np.uint8]
    target_token: NDArray[np.uint8]
    count_log_probability: NDArray[np.float64]
    checkpoint_step: NDArray[np.uint16]
    residual_logit: NDArray[np.float32]
    arrays_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.count_prior) is not AuthenticatedCountPrior:
            raise TypeError("count_prior must be an exact AuthenticatedCountPrior")
        decoded = self.count_prior.revalidate()
        rebound = AuthenticatedCountPrior(
            payload=self.count_prior.payload,
            sha256=self.count_prior.sha256,
            prior=decoded,
        )
        table = np.ascontiguousarray(
            decoded.log_relative_position_probability,
            dtype="<f8",
        ).copy()
        table.flags.writeable = False
        object.__setattr__(self, "count_prior", rebound)
        object.__setattr__(self, "count_prior_sha256", rebound.sha256)
        object.__setattr__(self, "sealed_log_relative_position_probability", table)
        _validate_scoring_archive(self)
        for name, raw in _raw_arrays(self, _ARCHIVE_ARRAY_NAMES).items():
            object.__setattr__(self, name, _owning_readonly_copy(raw))
        _validate_scoring_archive(self)
        object.__setattr__(
            self,
            "arrays_sha256",
            _arrays_integrity_sha256(self._raw_integrity_arrays()),
        )

    def __getattribute__(self, name: str) -> object:
        if name in (*_ARCHIVE_ARRAY_NAMES, "sealed_log_relative_position_probability"):
            raw = object.__getattribute__(self, name)
            try:
                object.__getattribute__(self, "arrays_sha256")
            except AttributeError:
                return raw
            self.revalidate()
            return _defensive_array(raw)
        return object.__getattribute__(self, name)

    def _raw_integrity_arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "sealed_log_relative_position_probability": object.__getattribute__(
                self,
                "sealed_log_relative_position_probability",
            ),
            **_raw_arrays(self, _ARCHIVE_ARRAY_NAMES),
        }

    def revalidate(self) -> None:
        self.ledger.revalidate()
        self.count_prior.revalidate()
        arrays = self._raw_integrity_arrays()
        _require_owned_readonly_arrays(arrays, label="scoring archive")
        expected = _require_sha256(
            object.__getattribute__(self, "arrays_sha256"),
            label="scoring archive array integrity SHA-256",
        )
        if _arrays_integrity_sha256(arrays) != expected:
            raise ValueError("scoring archive arrays changed after construction")
        _validate_scoring_archive(self)

    def _validated_arrays(self) -> dict[str, NDArray[np.generic]]:
        self.revalidate()
        return _raw_arrays(self, _ARCHIVE_ARRAY_NAMES)

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {name: _defensive_array(raw) for name, raw in self._validated_arrays().items()}

    def npz_bytes(self) -> bytes:
        """Return the deterministic score-residual-logit artifact bytes."""

        self.revalidate()
        raw_arrays = _raw_arrays(self, _ARCHIVE_ARRAY_NAMES)
        snapshot = {name: _defensive_array(raw) for name, raw in raw_arrays.items()}
        payload = deterministic_npz_bytes(
            residual_logits_npz_schema(
                len(raw_arrays["case_id"]),
                len(raw_arrays["target_token"]),
                checkpoint_count=len(raw_arrays["checkpoint_step"]),
            ),
            snapshot,
        )
        self.revalidate()
        return payload


def build_scoring_archive(
    ledger: ScoreCorruptionLedger,
    *,
    count_prior: AuthenticatedCountPrior,
    case_id: NDArray[np.bytes_],
    case_offsets: NDArray[np.uint64],
    position: NDArray[np.uint8],
    target_token: NDArray[np.uint8],
    count_log_probability: NDArray[np.float64],
    checkpoint_step: NDArray[np.uint16],
    residual_logit: NDArray[np.float32],
) -> ScoringArchive:
    """Validate sealed evaluator arrays without coercing any dtype or value."""

    if type(ledger) is not ScoreCorruptionLedger:
        raise TypeError("ledger must be an exact ScoreCorruptionLedger")
    return ScoringArchive(
        ledger=ledger,
        count_prior=count_prior,
        case_id=case_id,
        case_offsets=case_offsets,
        position=position,
        target_token=target_token,
        count_log_probability=count_log_probability,
        checkpoint_step=checkpoint_step,
        residual_logit=residual_logit,
    )


def _validate_scoring_archive(result: ScoringArchive) -> None:
    if type(result.ledger) is not ScoreCorruptionLedger:
        raise TypeError("scoring archive ledger must be an exact ScoreCorruptionLedger")
    _validate_corruption_ledger(result.ledger)
    ledger = result.ledger
    ledger_arrays = _raw_arrays(ledger, _LEDGER_ARRAY_NAMES)
    if type(result.count_prior) is not AuthenticatedCountPrior:
        raise TypeError("scoring archive count prior must remain authenticated")
    decoded_prior = result.count_prior.revalidate()
    if result.count_prior_sha256 != result.count_prior.sha256:
        raise ValueError("scoring archive count-prior digest is not artifact-derived")
    _require_sha256(result.count_prior_sha256, label="sealed count-prior SHA-256")
    table = object.__getattribute__(
        result,
        "sealed_log_relative_position_probability",
    )
    if (
        type(table) is not np.ndarray
        or table.dtype != np.dtype("<f8")
        or table.shape != (5, 10, 20)
        or not table.flags.c_contiguous
        or not bool(np.isfinite(table).all())
    ):
        raise TypeError("sealed C0 log table must be finite C-contiguous <f8 [5,10,20]")
    decoded_table = decoded_prior.log_relative_position_probability
    if table.tobytes(order="C") != decoded_table.tobytes(order="C"):
        raise ValueError("scoring archive C0 table is not artifact-derived")
    arrays = _raw_arrays(result, _ARCHIVE_ARRAY_NAMES)
    selected_count = int(math.fsum(int(value) for value in ledger_arrays["mask_count"]))
    schema = residual_logits_npz_schema(
        len(ledger_arrays["case_id"]),
        selected_count,
        checkpoint_count=len(arrays["checkpoint_step"]),
    )
    if tuple(int(value) for value in arrays["checkpoint_step"]) != CHECKPOINT_STEPS:
        raise ValueError("checkpoint_step differs from the frozen five-checkpoint order")
    for item in schema:
        raw = arrays[item.name]
        if type(raw) is not np.ndarray:
            raise TypeError(f"score member {item.name} must be an exact ndarray")
        if raw.dtype != np.dtype(item.dtype) or raw.shape != item.shape:
            raise TypeError(f"score member {item.name} violates its exact dtype or shape")
        if not raw.flags.c_contiguous:
            raise ValueError(f"score member {item.name} must be C-contiguous")
        if raw.dtype.kind == "f" and not bool(np.isfinite(raw).all()):
            raise ValueError(f"score member {item.name} contains a non-finite value")
    if not np.array_equal(arrays["case_id"], ledger_arrays["case_id"]):
        raise ValueError("score case IDs do not match the corruption ledger")
    expected_offsets, expected_positions, expected_targets, expected_count_log = (
        gather_count_log_probability(ledger, table)
    )
    if not np.array_equal(arrays["case_offsets"], expected_offsets):
        raise ValueError("score case offsets do not exactly match mask counts")
    if not np.array_equal(arrays["position"], expected_positions):
        raise ValueError("score positions are not the exact sealed-C0 gather")
    if not np.array_equal(arrays["target_token"], expected_targets):
        raise ValueError("score targets are not the exact sealed-C0 gather")
    if not np.array_equal(arrays["count_log_probability"], expected_count_log):
        raise ValueError("score count logits are not the exact gather from sealed C0")
    cursor = 0
    for case_index, row_index_raw in enumerate(ledger_arrays["row_index"]):
        row_index = int(row_index_raw)
        positions = np.flatnonzero(ledger_arrays["selected_mask"][case_index])
        stop = cursor + len(positions)
        if not np.array_equal(arrays["position"][cursor:stop], positions.astype("|u1")):
            raise ValueError("score positions do not follow case then ascending-position order")
        expected_targets = ledger_arrays["clean_tokens"][row_index, positions]
        if not np.array_equal(arrays["target_token"][cursor:stop], expected_targets):
            raise ValueError("score target tokens do not match the clean corruption source")
        cursor = stop
    if cursor != selected_count:
        raise RuntimeError("selected-token validation ended at the wrong offset")
    probabilities = np.exp(arrays["count_log_probability"])
    if np.any(probabilities <= 0.0) or not np.allclose(
        np.sum(probabilities, axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("count_log_probability is not a finite normalized C0 distribution")


@dataclass(frozen=True, slots=True)
class _ValidatedScoringView:
    """Private raw-array view bracketed by full archive validation."""

    archive: ScoringArchive = field(repr=False)
    archive_arrays: Mapping[str, NDArray[np.generic]] = field(repr=False)
    ledger_arrays: Mapping[str, NDArray[np.generic]] = field(repr=False)
    rows: tuple[ScoreRow, ...]

    def revalidate(self) -> None:
        """Fail closed if the archive changed while the view was in use."""

        self.archive.revalidate()


def _validated_scoring_view(archive: ScoringArchive) -> _ValidatedScoringView:
    if type(archive) is not ScoringArchive:
        raise TypeError("archive must be an exact ScoringArchive")
    archive.revalidate()
    ledger = archive.ledger
    return _ValidatedScoringView(
        archive=archive,
        archive_arrays=MappingProxyType(_raw_arrays(archive, _ARCHIVE_ARRAY_NAMES)),
        ledger_arrays=MappingProxyType(_raw_arrays(ledger, _LEDGER_ARRAY_NAMES)),
        rows=ledger.rows,
    )


def stable_probabilities(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute max-subtracted float64 softmax with no implicit coercion."""

    if type(logits) is not np.ndarray or logits.dtype != np.dtype("<f8"):
        raise TypeError("logits must be an exact little-endian float64 ndarray")
    if logits.ndim != 2 or logits.shape[1] != len(ALPHABET) or not logits.flags.c_contiguous:
        raise ValueError("logits must be C-contiguous with shape (N, 20)")
    if not bool(np.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=1, keepdims=True, dtype=np.float64)
    probabilities = exponentials / denominator
    if not bool(np.isfinite(probabilities).all()) or np.any(probabilities < 0.0):
        raise ValueError("stable softmax produced an invalid probability")
    return np.asarray(probabilities, dtype="<f8", order="C")


def _target_nll(
    count_log_probability: NDArray[np.float64],
    target_token: NDArray[np.uint8],
    residual_logit: NDArray[np.float32] | None,
    *,
    residual_lambda: float,
    temperature: float,
) -> NDArray[np.float64]:
    if (
        type(residual_lambda) is not float
        or not math.isfinite(residual_lambda)
        or residual_lambda < 0.0
    ):
        raise ValueError("residual_lambda must be a non-negative finite exact float")
    if type(temperature) is not float or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite exact float")
    logits = count_log_probability.copy()
    if residual_logit is not None:
        if (
            type(residual_logit) is not np.ndarray
            or residual_logit.dtype != np.dtype("<f4")
            or residual_logit.shape != count_log_probability.shape
            or not bool(np.isfinite(residual_logit).all())
        ):
            raise TypeError("residual logits must be aligned finite little-endian float32")
        logits += residual_lambda * residual_logit.astype("<f8")
    elif residual_lambda != 0.0:
        raise ValueError("a nonzero residual lambda requires residual logits")
    logits /= temperature
    maximum = np.max(logits, axis=1)
    shifted = logits - maximum[:, None]
    log_normalizer = maximum + np.log(np.sum(np.exp(shifted), axis=1, dtype=np.float64))
    indices = np.arange(len(target_token), dtype=np.int64)
    result = log_normalizer - logits[indices, target_token.astype(np.int64)]
    if not bool(np.isfinite(result).all()) or np.any(result < 0.0):
        raise ValueError("target NLL calculation produced invalid values")
    return np.asarray(result, dtype="<f8")


def _case_means_from_view(
    view: _ValidatedScoringView,
    token_values: NDArray[np.float64],
) -> FloatArray:
    archive_arrays = view.archive_arrays
    if (
        type(token_values) is not np.ndarray
        or token_values.dtype != np.dtype("<f8")
        or token_values.shape != (int(archive_arrays["case_offsets"][-1]),)
        or not bool(np.isfinite(token_values).all())
    ):
        raise TypeError("token values must be aligned finite little-endian float64")
    ledger_arrays = view.ledger_arrays
    result = np.empty(len(ledger_arrays["case_id"]), dtype="<f8")
    for case_index in range(len(result)):
        start = int(archive_arrays["case_offsets"][case_index])
        stop = int(archive_arrays["case_offsets"][case_index + 1])
        if stop <= start:
            raise ValueError("every score case must contain a selected token")
        result[case_index] = math.fsum(token_values[start:stop].tolist()) / (stop - start)
    return result


def _case_means(archive: ScoringArchive, token_values: NDArray[np.float64]) -> FloatArray:
    view = _validated_scoring_view(archive)
    result = _case_means_from_view(view, token_values)
    view.revalidate()
    return result


def _row_bin_means_from_view(
    view: _ValidatedScoringView,
    case_values: FloatArray,
) -> FloatArray:
    case_count = len(view.ledger_arrays["case_id"])
    row_count = len(view.rows)
    if (
        type(case_values) is not np.ndarray
        or case_values.dtype != np.dtype("<f8")
        or case_values.shape != (case_count,)
        or not bool(np.isfinite(case_values).all())
    ):
        raise TypeError("case values must be aligned finite little-endian float64")
    result = np.empty((row_count, len(TIMESTEP_BINS)), dtype="<f8")
    for row_index in range(row_count):
        row_start = row_index * LEVELS
        for bin_index, (first_level, last_level) in enumerate(TIMESTEP_BINS):
            start = row_start + first_level - 1
            stop = row_start + last_level
            values = case_values[start:stop]
            expected = last_level - first_level + 1
            if len(values) != expected:
                raise ValueError("row does not have complete timestep-bin case coverage")
            result[row_index, bin_index] = math.fsum(values.tolist()) / expected
    return result


def _row_bin_means(archive: ScoringArchive, case_values: FloatArray) -> FloatArray:
    view = _validated_scoring_view(archive)
    result = _row_bin_means_from_view(view, case_values)
    view.revalidate()
    return result


@dataclass(frozen=True, slots=True, order=True)
class CalibrationChoice:
    residual_lambda: float
    temperature: float

    def __post_init__(self) -> None:
        if (
            type(self.residual_lambda) is not float
            or self.residual_lambda < 0.0
            or not math.isfinite(self.residual_lambda)
        ):
            raise ValueError("calibration lambda must be a non-negative exact float")
        if (
            type(self.temperature) is not float
            or self.temperature <= 0.0
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("calibration temperature must be a positive exact float")


@dataclass(frozen=True, slots=True)
class LoucoPlan:
    """One component-specific, four-bin score-only calibration plan."""

    method: str
    checkpoint_step: int | None
    component_ids: tuple[str, ...]
    choices: tuple[tuple[CalibrationChoice, ...], ...]
    leaveout_objectives: tuple[tuple[float, ...], ...]
    grid: tuple[CalibrationChoice, ...]

    def __post_init__(self) -> None:
        if self.method not in {"C0T", "R128"}:
            raise ValueError("LOUCO method must be C0T or R128")
        if self.method == "C0T" and self.checkpoint_step is not None:
            raise ValueError("C0T cannot bind a neural checkpoint")
        if self.method == "R128" and self.checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError("R128 LOUCO plan must bind a frozen checkpoint")
        if (
            not self.component_ids
            or self.component_ids != tuple(sorted(self.component_ids))
            or len(set(self.component_ids)) != len(self.component_ids)
        ):
            raise ValueError("LOUCO components must be unique and ascending")
        if any(_SHA256_RE.fullmatch(value) is None for value in self.component_ids):
            raise ValueError("LOUCO components must be SHA-256 identities")
        if (
            len(self.choices) != len(self.component_ids)
            or len(self.leaveout_objectives) != len(self.component_ids)
            or any(len(value) != len(TIMESTEP_BINS) for value in self.choices)
            or any(len(value) != len(TIMESTEP_BINS) for value in self.leaveout_objectives)
        ):
            raise ValueError("LOUCO plan must contain four choices per component")
        if any(choice not in self.grid for choices in self.choices for choice in choices):
            raise ValueError("LOUCO plan selected a choice outside its grid")
        if self.grid != _calibration_grid(self.method):
            raise ValueError("LOUCO plan grid differs from the frozen method grid")
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for objectives in self.leaveout_objectives
            for value in objectives
        ):
            raise ValueError("LOUCO objectives must be finite non-negative floats")

    def choice(self, component_id: str, bin_index: int) -> CalibrationChoice:
        _require_sha256(component_id, label="union component ID")
        if type(bin_index) is not int or not 0 <= bin_index < len(TIMESTEP_BINS):
            raise ValueError("timestep bin index must be an exact integer in 0..3")
        try:
            component_index = self.component_ids.index(component_id)
        except ValueError as error:
            raise KeyError(component_id) from error
        return self.choices[component_index][bin_index]

    def revalidate(self) -> None:
        self.__post_init__()


def _calibration_grid(method: str) -> tuple[CalibrationChoice, ...]:
    if method == "C0T":
        return tuple(CalibrationChoice(0.0, temperature) for temperature in TEMPERATURES)
    if method == "R128":
        return tuple(
            CalibrationChoice(residual_lambda, temperature)
            for residual_lambda in RESIDUAL_LAMBDAS
            for temperature in TEMPERATURES
        )
    raise ValueError("calibration method must be exactly C0T or R128")


def _choice_tie_key(choice: CalibrationChoice) -> tuple[float, float, float]:
    return (
        choice.residual_lambda,
        abs(choice.temperature - 1.0),
        choice.temperature,
    )


def select_calibration_choice(
    grid: Sequence[CalibrationChoice],
    objectives: Sequence[float],
) -> tuple[CalibrationChoice, float]:
    """Select minimum NLL with exact lambda/temperature tie ordering."""

    choices = tuple(grid)
    values = tuple(objectives)
    if (
        not choices
        or any(type(choice) is not CalibrationChoice for choice in choices)
        or len(set(choices)) != len(choices)
        or len(values) != len(choices)
    ):
        raise ValueError("calibration grid and objectives must be unique and aligned")
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("calibration objectives must be finite non-negative exact floats")
    best_index = min(
        range(len(choices)),
        key=lambda index: (values[index], *_choice_tie_key(choices[index])),
    )
    return choices[best_index], values[best_index]


def _fit_louco_plan_from_view(
    view: _ValidatedScoringView,
    *,
    method: str,
    checkpoint_step: int | None = None,
) -> LoucoPlan:
    archive_arrays = view.archive_arrays
    if method not in {"C0T", "R128"}:
        raise ValueError("method must be exactly C0T or R128")
    if method == "C0T":
        if checkpoint_step is not None:
            raise ValueError("C0T does not accept a checkpoint")
        residual = None
    else:
        if type(checkpoint_step) is not int or checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError("R128 checkpoint_step must be one of the five frozen steps")
        checkpoint_index = CHECKPOINT_STEPS.index(checkpoint_step)
        residual = archive_arrays["residual_logit"][checkpoint_index]
    grid = _calibration_grid(method)
    row_grid = np.empty(
        (len(view.rows), len(TIMESTEP_BINS), len(grid)),
        dtype="<f8",
    )
    for grid_index, choice in enumerate(grid):
        token_nll = _target_nll(
            archive_arrays["count_log_probability"],
            archive_arrays["target_token"],
            residual,
            residual_lambda=choice.residual_lambda,
            temperature=choice.temperature,
        )
        row_grid[:, :, grid_index] = _row_bin_means_from_view(
            view,
            _case_means_from_view(view, token_nll),
        )
    component_ids = tuple(sorted({row.union_component_id for row in view.rows}))
    if len(component_ids) < 2:
        raise ValueError("LOUCO calibration requires at least two union components")
    row_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(view.rows):
        row_indices[row.union_component_id].append(index)
    component_mass: dict[str, float] = {}
    component_loss: dict[str, FloatArray] = {}
    for component_id in component_ids:
        indices = row_indices[component_id]
        component_mass[component_id] = math.fsum(
            view.rows[index].sampling_weight for index in indices
        )
        loss = np.empty((len(TIMESTEP_BINS), len(grid)), dtype="<f8")
        for bin_index in range(len(TIMESTEP_BINS)):
            for grid_index in range(len(grid)):
                loss[bin_index, grid_index] = math.fsum(
                    view.rows[index].sampling_weight * float(row_grid[index, bin_index, grid_index])
                    for index in indices
                )
        component_loss[component_id] = loss
    total_mass = math.fsum(component_mass[value] for value in component_ids)
    if total_mass.hex() != (1.0).hex():
        raise ValueError("full score-fold calibration mass must equal binary64 one")
    total_loss = np.empty((len(TIMESTEP_BINS), len(grid)), dtype="<f8")
    for bin_index in range(len(TIMESTEP_BINS)):
        for grid_index in range(len(grid)):
            total_loss[bin_index, grid_index] = math.fsum(
                float(component_loss[value][bin_index, grid_index]) for value in component_ids
            )
    all_choices: list[tuple[CalibrationChoice, ...]] = []
    all_objectives: list[tuple[float, ...]] = []
    for component_id in component_ids:
        mass = math.fsum((total_mass, -component_mass[component_id]))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("LOUCO exclusion left non-positive calibration mass")
        component_choices: list[CalibrationChoice] = []
        component_objectives: list[float] = []
        for bin_index in range(len(TIMESTEP_BINS)):
            objectives: list[float] = []
            for grid_index in range(len(grid)):
                numerator = math.fsum(
                    (
                        float(total_loss[bin_index, grid_index]),
                        -float(component_loss[component_id][bin_index, grid_index]),
                    )
                )
                objective = numerator / mass
                if not math.isfinite(objective) or objective < 0.0:
                    raise ValueError("LOUCO objective is invalid")
                objectives.append(objective)
            selected_choice, selected_objective = select_calibration_choice(
                grid,
                tuple(objectives),
            )
            component_choices.append(selected_choice)
            component_objectives.append(selected_objective)
        all_choices.append(tuple(component_choices))
        all_objectives.append(tuple(component_objectives))
    return LoucoPlan(
        method=method,
        checkpoint_step=checkpoint_step,
        component_ids=component_ids,
        choices=tuple(all_choices),
        leaveout_objectives=tuple(all_objectives),
        grid=grid,
    )


def fit_louco_plan(
    archive: ScoringArchive,
    *,
    method: str,
    checkpoint_step: int | None = None,
) -> LoucoPlan:
    """Fit exact per-union-component/per-bin LOUCO choices from score rows."""

    view = _validated_scoring_view(archive)
    result = _fit_louco_plan_from_view(
        view,
        method=method,
        checkpoint_step=checkpoint_step,
    )
    view.revalidate()
    return result


def verify_zero_lambda_equivalence(
    archive: ScoringArchive,
    *,
    checkpoint_step: int,
) -> tuple[bool, ...]:
    """Report the five zero-lambda diagnostics; none is selection-eligible."""

    if type(checkpoint_step) is not int or checkpoint_step not in CHECKPOINT_STEPS:
        raise ValueError("checkpoint_step must be one of the five frozen steps")
    if type(archive) is not ScoringArchive:
        raise TypeError("archive must be an exact ScoringArchive")
    archive_arrays = archive._validated_arrays()
    residual = archive_arrays["residual_logit"][CHECKPOINT_STEPS.index(checkpoint_step)]
    results: list[bool] = []
    for temperature in TEMPERATURES:
        control = _target_nll(
            archive_arrays["count_log_probability"],
            archive_arrays["target_token"],
            None,
            residual_lambda=0.0,
            temperature=temperature,
        )
        diagnostic = _target_nll(
            archive_arrays["count_log_probability"],
            archive_arrays["target_token"],
            residual,
            residual_lambda=0.0,
            temperature=temperature,
        )
        results.append(bool(np.array_equal(control, diagnostic)))
    if not all(results):
        raise ValueError("zero-lambda R128 diagnostic differs from C0T")
    return tuple(results)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Fold or equal-fold metric plus ECE sufficient statistics."""

    nll: float
    perplexity: float
    top1_accuracy: float
    top3_accuracy: float
    multiclass_brier: float
    ece: float
    ece_mass: tuple[float, ...]
    ece_confidence: tuple[float, ...]
    ece_correct: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("nll", "perplexity", "multiclass_brier", "ece"):
            value = _require_finite_float(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("top1_accuracy", "top3_accuracy"):
            value = _require_finite_float(getattr(self, name), label=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.top3_accuracy < self.top1_accuracy:
            raise ValueError("top3_accuracy cannot be below top1_accuracy")
        if self.multiclass_brier > 2.0 or self.ece > 1.0:
            raise ValueError("Brier or ECE exceeds its valid range")
        try:
            expected_perplexity = math.exp(self.nll)
        except OverflowError as error:
            raise ValueError("NLL is too large for finite perplexity") from error
        if self.perplexity.hex() != expected_perplexity.hex():
            raise ValueError("perplexity is not the exact exponential of NLL")
        vectors = (self.ece_mass, self.ece_confidence, self.ece_correct)
        if any(type(value) is not tuple or len(value) != ECE_BINS for value in vectors):
            raise ValueError("ECE sufficient statistics must be fifteen-element tuples")
        for mass, confidence, correct in zip(*vectors, strict=True):
            if any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    mass,
                    confidence,
                    correct,
                )
            ):
                raise ValueError("ECE sufficient statistics must be finite exact floats")
            if mass < 0.0 or not 0.0 <= confidence <= mass or not 0.0 <= correct <= mass:
                raise ValueError("ECE sufficient statistics are outside their valid ranges")
        expected_ece = _ece(self.ece_mass, self.ece_confidence, self.ece_correct)
        if not math.isclose(self.ece, expected_ece, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("ECE differs from its sufficient statistics")

    def revalidate(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class RowNll:
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    sampling_weight: float
    nll: float
    nll_by_timestep_bin: tuple[float, ...]
    overall: MetricSummary
    timestep_bins: tuple[MetricSummary, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.sequence_id, label="row metric sequence_id")
        _require_sha256(self.homology_component_id, label="row metric homology component")
        _require_sha256(self.union_component_id, label="row metric union component")
        if _require_finite_float(self.sampling_weight, label="row metric weight") <= 0.0:
            raise ValueError("row metric weight must be positive")
        if _require_finite_float(self.nll, label="row metric NLL") < 0.0:
            raise ValueError("row metric NLL must be non-negative")
        if (
            type(self.nll_by_timestep_bin) is not tuple
            or len(self.nll_by_timestep_bin) != len(TIMESTEP_BINS)
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0.0
                for value in self.nll_by_timestep_bin
            )
        ):
            raise ValueError("row metric bin NLLs must be four finite non-negative floats")
        if type(self.overall) is not MetricSummary or (
            type(self.timestep_bins) is not tuple
            or len(self.timestep_bins) != len(TIMESTEP_BINS)
            or any(type(summary) is not MetricSummary for summary in self.timestep_bins)
        ):
            raise TypeError("row metrics require exact overall and timestep summaries")
        self.overall.revalidate()
        for summary in self.timestep_bins:
            summary.revalidate()
        if self.nll.hex() != self.overall.nll.hex() or tuple(
            value.hex() for value in self.nll_by_timestep_bin
        ) != tuple(summary.nll.hex() for summary in self.timestep_bins):
            raise ValueError("row NLL fields differ from their complete metric summaries")
        expected_overall = _aggregate_weighted_metric_summaries(
            tuple((0.25, summary) for summary in self.timestep_bins)
        )
        if not _metric_summaries_close(self.overall, expected_overall):
            raise ValueError("row overall summary differs from its four timestep-bin summaries")

    def revalidate(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class FoldMethodMetrics:
    """Cross-calibrated metrics for one method/checkpoint and outer fold."""

    method: str
    checkpoint_step: int | None
    outer_fold: int
    row_nll: tuple[RowNll, ...]
    overall: MetricSummary
    timestep_bins: tuple[MetricSummary, ...]
    louco_plan: LoucoPlan | None
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _consume_metric_factory_token(self._factory_token)
        object.__setattr__(self, "_factory_token", None)
        _validate_fold_method_metrics(self)

    def revalidate(self) -> None:
        _validate_fold_method_metrics(self)


def _validate_fold_method_metrics(value: FoldMethodMetrics) -> None:
    if value.method not in {"C0", "C0T", "R128"}:
        raise ValueError("fold metric method must be C0, C0T, or R128")
    if value.method == "R128" and value.checkpoint_step not in CHECKPOINT_STEPS:
        raise ValueError("R128 fold metrics require a frozen checkpoint")
    if value.method != "R128" and value.checkpoint_step is not None:
        raise ValueError("count-control fold metrics cannot bind a checkpoint")
    if type(value.outer_fold) is not int or value.outer_fold not in (0, 1, 2, 3):
        raise ValueError("outer_fold must be an exact integer in 0..3")
    if type(value.row_nll) is not tuple or any(type(row) is not RowNll for row in value.row_nll):
        raise TypeError("row metrics must be a tuple of exact RowNll values")
    for row in value.row_nll:
        row.revalidate()
        expected_row_nll = math.fsum(row.nll_by_timestep_bin) / len(TIMESTEP_BINS)
        if not math.isclose(row.nll, expected_row_nll, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("row overall NLL differs from its equal timestep-bin mean")
    identifiers = tuple(row.sequence_id for row in value.row_nll)
    if not identifiers or identifiers != tuple(sorted(identifiers)):
        raise ValueError("row metrics must be non-empty and ordered by sequence_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("row metric sequence IDs must be unique")
    if math.fsum(row.sampling_weight for row in value.row_nll).hex() != (1.0).hex():
        raise ValueError("fold row metric weights must sum to binary64 one")
    if type(value.overall) is not MetricSummary or (
        type(value.timestep_bins) is not tuple
        or len(value.timestep_bins) != len(TIMESTEP_BINS)
        or any(type(summary) is not MetricSummary for summary in value.timestep_bins)
    ):
        raise TypeError("fold metrics require exact overall and timestep summaries")
    value.overall.revalidate()
    for summary in value.timestep_bins:
        summary.revalidate()
    expected_overall = _aggregate_weighted_metric_summaries(
        tuple((row.sampling_weight, row.overall) for row in value.row_nll)
    )
    if value.overall.nll.hex() != expected_overall.nll.hex():
        raise ValueError("fold overall NLL summary differs from row evidence")
    if value.overall != expected_overall:
        raise ValueError("fold overall summary differs from complete row evidence")
    for index, summary in enumerate(value.timestep_bins):
        expected = _aggregate_weighted_metric_summaries(
            tuple((row.sampling_weight, row.timestep_bins[index]) for row in value.row_nll)
        )
        if summary.nll.hex() != expected.nll.hex():
            raise ValueError("fold timestep NLL summary differs from row evidence")
        if summary != expected:
            raise ValueError("fold timestep summary differs from complete row evidence")
    component_ids = tuple(sorted({row.union_component_id for row in value.row_nll}))
    if value.method == "C0" and value.louco_plan is not None:
        raise ValueError("raw C0 cannot carry a LOUCO plan")
    if value.method != "C0" and (
        type(value.louco_plan) is not LoucoPlan
        or value.louco_plan.method != value.method
        or value.louco_plan.checkpoint_step != value.checkpoint_step
        or value.louco_plan.component_ids != component_ids
    ):
        raise ValueError("cross-calibrated metrics require their exact LOUCO components")
    if value.louco_plan is not None:
        value.louco_plan.revalidate()


def _new_fold_method_metrics(**values: object) -> FoldMethodMetrics:
    return FoldMethodMetrics(**values, _factory_token=_MetricFactoryToken())


def _ece(
    mass: Sequence[float],
    confidence: Sequence[float],
    correct: Sequence[float],
) -> float:
    if not (len(mass) == len(confidence) == len(correct) == ECE_BINS):
        raise ValueError("ECE requires exactly fifteen aligned bins")
    total = math.fsum(mass)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("ECE requires positive finite mass")
    terms: list[float] = []
    for bin_mass, bin_confidence, bin_correct in zip(mass, confidence, correct, strict=True):
        if not all(
            math.isfinite(value)
            for value in (
                bin_mass,
                bin_confidence,
                bin_correct,
            )
        ):
            raise ValueError("ECE sufficient statistics must be finite")
        if bin_mass < 0.0 or bin_confidence < 0.0 or bin_correct < 0.0:
            raise ValueError("ECE sufficient statistics cannot be negative")
        if bin_mass > 0.0:
            terms.append(bin_mass / total * abs(bin_correct / bin_mass - bin_confidence / bin_mass))
    result = math.fsum(terms)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("ECE is outside [0, 1]")
    return result


def _metric_summary(
    *,
    nll: float,
    top1: float,
    top3: float,
    brier: float,
    ece_mass: Sequence[float],
    ece_confidence: Sequence[float],
    ece_correct: Sequence[float],
) -> MetricSummary:
    try:
        perplexity = math.exp(nll)
    except OverflowError as error:
        raise ValueError("NLL is too large for finite perplexity") from error
    mass = tuple(float(value) for value in ece_mass)
    confidence = tuple(float(value) for value in ece_confidence)
    correct = tuple(float(value) for value in ece_correct)
    return MetricSummary(
        nll=float(nll),
        perplexity=float(perplexity),
        top1_accuracy=float(top1),
        top3_accuracy=float(top3),
        multiclass_brier=float(brier),
        ece=float(_ece(mass, confidence, correct)),
        ece_mass=mass,
        ece_confidence=confidence,
        ece_correct=correct,
    )


def _aggregate_weighted_metric_summaries(
    values: tuple[tuple[float, MetricSummary], ...],
) -> MetricSummary:
    """Rebuild every aggregate field from complete per-row sufficient statistics."""

    if not values or any(
        type(weight) is not float
        or not math.isfinite(weight)
        or weight <= 0.0
        or type(summary) is not MetricSummary
        for weight, summary in values
    ):
        raise ValueError("weighted metric summaries must be non-empty and valid")
    for _, summary in values:
        summary.revalidate()
    return _metric_summary(
        nll=math.fsum(weight * summary.nll for weight, summary in values),
        top1=math.fsum(weight * summary.top1_accuracy for weight, summary in values),
        top3=math.fsum(weight * summary.top3_accuracy for weight, summary in values),
        brier=math.fsum(weight * summary.multiclass_brier for weight, summary in values),
        ece_mass=tuple(
            math.fsum(weight * summary.ece_mass[index] for weight, summary in values)
            for index in range(ECE_BINS)
        ),
        ece_confidence=tuple(
            math.fsum(weight * summary.ece_confidence[index] for weight, summary in values)
            for index in range(ECE_BINS)
        ),
        ece_correct=tuple(
            math.fsum(weight * summary.ece_correct[index] for weight, summary in values)
            for index in range(ECE_BINS)
        ),
    )


def _metric_summaries_close(left: MetricSummary, right: MetricSummary) -> bool:
    # Compare only independent quantities across alternate reduction trees.
    # MetricSummary.revalidate() separately binds the derived perplexity and
    # ECE values to NLL and the ECE sufficient statistics, respectively.
    scalar_fields = (
        "nll",
        "top1_accuracy",
        "top3_accuracy",
        "multiclass_brier",
    )
    vector_fields = ("ece_mass", "ece_confidence", "ece_correct")
    return all(
        math.isclose(getattr(left, name), getattr(right, name), rel_tol=0.0, abs_tol=1e-15)
        for name in scalar_fields
    ) and all(
        all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
            for a, b in zip(getattr(left, name), getattr(right, name), strict=True)
        )
        for name in vector_fields
    )


def _timestep_bin_index(level: int) -> int:
    if type(level) is not int or not 1 <= level <= LEVELS:
        raise ValueError("level must be an exact integer in 1..64")
    return (level - 1) // 16


def _score_fold_from_view(
    view: _ValidatedScoringView,
    *,
    method: str,
    checkpoint_step: int | None = None,
) -> FoldMethodMetrics:
    archive_arrays = view.archive_arrays
    ledger_arrays = view.ledger_arrays
    if method not in {"C0", "C0T", "R128"}:
        raise ValueError("method must be exactly C0, C0T, or R128")
    if method == "C0":
        if checkpoint_step is not None:
            raise ValueError("raw C0 does not accept a checkpoint")
        louco_plan = None
        residual = None
    elif method == "C0T":
        if checkpoint_step is not None:
            raise ValueError("C0T does not bind a checkpoint")
        louco_plan = _fit_louco_plan_from_view(view, method="C0T")
        residual = None
    else:
        if type(checkpoint_step) is not int or checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError("R128 requires one frozen checkpoint_step")
        louco_plan = _fit_louco_plan_from_view(
            view,
            method="R128",
            checkpoint_step=checkpoint_step,
        )
        residual = archive_arrays["residual_logit"][CHECKPOINT_STEPS.index(checkpoint_step)]

    case_count = len(ledger_arrays["case_id"])
    case_values = np.empty((case_count, 4), dtype="<f8")
    case_ece_mass = np.zeros((case_count, ECE_BINS), dtype="<f8")
    case_ece_confidence = np.zeros((case_count, ECE_BINS), dtype="<f8")
    case_ece_correct = np.zeros((case_count, ECE_BINS), dtype="<f8")
    for case_index in range(case_count):
        start = int(archive_arrays["case_offsets"][case_index])
        stop = int(archive_arrays["case_offsets"][case_index + 1])
        if stop <= start:
            raise ValueError("score case has no selected token")
        score_row_index = int(ledger_arrays["row_index"][case_index])
        row = view.rows[score_row_index]
        level = int(ledger_arrays["level"][case_index])
        if method == "C0":
            choice = CalibrationChoice(0.0, 1.0)
        else:
            if louco_plan is None:  # pragma: no cover - branch invariant
                raise RuntimeError("cross-calibrated method lost its LOUCO plan")
            choice = louco_plan.choice(row.union_component_id, _timestep_bin_index(level))
        count_values = archive_arrays["count_log_probability"][start:stop]
        residual_values = None if residual is None else residual[start:stop]
        targets = archive_arrays["target_token"][start:stop]
        target_indices = targets.astype(np.int64)
        indices = np.arange(stop - start, dtype=np.int64)
        if method == "C0":
            probabilities = np.asarray(np.exp(count_values), dtype="<f8", order="C")
            token_nll = np.asarray(
                -count_values[indices, target_indices],
                dtype="<f8",
                order="C",
            )
        else:
            logits = count_values.copy()
            if residual_values is not None:
                logits += choice.residual_lambda * residual_values.astype("<f8")
            logits /= choice.temperature
            probabilities = stable_probabilities(np.asarray(logits, dtype="<f8", order="C"))
            token_nll = _target_nll(
                count_values,
                targets,
                residual_values,
                residual_lambda=choice.residual_lambda,
                temperature=choice.temperature,
            )
        predictions = np.argmax(probabilities, axis=1)
        correct = predictions == target_indices
        ranking = np.argsort(-probabilities, axis=1, kind="stable")
        top3 = np.any(ranking[:, :3] == target_indices[:, None], axis=1)
        target_probability = probabilities[indices, target_indices]
        brier = (
            np.sum(np.square(probabilities), axis=1, dtype=np.float64)
            - 2.0 * target_probability
            + 1.0
        )
        confidence = probabilities[indices, predictions]
        bins = np.minimum(
            ECE_BINS - 1,
            np.floor(ECE_BINS * confidence).astype(np.int64),
        )
        token_count = stop - start
        case_values[case_index] = (
            math.fsum(token_nll.tolist()) / token_count,
            int(np.count_nonzero(correct)) / token_count,
            int(np.count_nonzero(top3)) / token_count,
            math.fsum(brier.tolist()) / token_count,
        )
        inverse_count = 1.0 / token_count
        for bin_index in range(ECE_BINS):
            selected = bins == bin_index
            count = int(np.count_nonzero(selected))
            if count:
                case_ece_mass[case_index, bin_index] = count * inverse_count
                case_ece_confidence[case_index, bin_index] = (
                    math.fsum(confidence[selected].tolist()) * inverse_count
                )
                case_ece_correct[case_index, bin_index] = (
                    int(np.count_nonzero(correct[selected])) * inverse_count
                )

    rows = view.rows
    row_nll: list[RowNll] = []
    row_values = np.empty((len(rows), 4), dtype="<f8")
    row_bin_values = np.empty((len(rows), len(TIMESTEP_BINS), 4), dtype="<f8")
    row_ece_mass = np.empty((len(rows), ECE_BINS), dtype="<f8")
    row_ece_confidence = np.empty((len(rows), ECE_BINS), dtype="<f8")
    row_ece_correct = np.empty((len(rows), ECE_BINS), dtype="<f8")
    row_bin_ece_mass = np.empty((len(rows), len(TIMESTEP_BINS), ECE_BINS), dtype="<f8")
    row_bin_ece_confidence = np.empty_like(row_bin_ece_mass)
    row_bin_ece_correct = np.empty_like(row_bin_ece_mass)
    for row_index, row in enumerate(rows):
        start = row_index * LEVELS
        stop = start + LEVELS
        for metric_index in range(4):
            row_values[row_index, metric_index] = (
                math.fsum(case_values[start:stop, metric_index].tolist()) / LEVELS
            )
        for ece_bin in range(ECE_BINS):
            row_ece_mass[row_index, ece_bin] = (
                math.fsum(case_ece_mass[start:stop, ece_bin].tolist()) / LEVELS
            )
            row_ece_confidence[row_index, ece_bin] = (
                math.fsum(case_ece_confidence[start:stop, ece_bin].tolist()) / LEVELS
            )
            row_ece_correct[row_index, ece_bin] = (
                math.fsum(case_ece_correct[start:stop, ece_bin].tolist()) / LEVELS
            )
        for timestep_bin, (first_level, last_level) in enumerate(TIMESTEP_BINS):
            bin_start = start + first_level - 1
            bin_stop = start + last_level
            bin_cases = last_level - first_level + 1
            for metric_index in range(4):
                row_bin_values[row_index, timestep_bin, metric_index] = (
                    math.fsum(case_values[bin_start:bin_stop, metric_index].tolist()) / bin_cases
                )
            for ece_bin in range(ECE_BINS):
                row_bin_ece_mass[row_index, timestep_bin, ece_bin] = (
                    math.fsum(case_ece_mass[bin_start:bin_stop, ece_bin].tolist()) / bin_cases
                )
                row_bin_ece_confidence[row_index, timestep_bin, ece_bin] = (
                    math.fsum(case_ece_confidence[bin_start:bin_stop, ece_bin].tolist()) / bin_cases
                )
                row_bin_ece_correct[row_index, timestep_bin, ece_bin] = (
                    math.fsum(case_ece_correct[bin_start:bin_stop, ece_bin].tolist()) / bin_cases
                )
        row_overall = _metric_summary(
            nll=float(row_values[row_index, 0]),
            top1=float(row_values[row_index, 1]),
            top3=float(row_values[row_index, 2]),
            brier=float(row_values[row_index, 3]),
            ece_mass=tuple(float(value) for value in row_ece_mass[row_index]),
            ece_confidence=tuple(float(value) for value in row_ece_confidence[row_index]),
            ece_correct=tuple(float(value) for value in row_ece_correct[row_index]),
        )
        row_timestep_bins = tuple(
            _metric_summary(
                nll=float(row_bin_values[row_index, bin_index, 0]),
                top1=float(row_bin_values[row_index, bin_index, 1]),
                top3=float(row_bin_values[row_index, bin_index, 2]),
                brier=float(row_bin_values[row_index, bin_index, 3]),
                ece_mass=tuple(float(value) for value in row_bin_ece_mass[row_index, bin_index]),
                ece_confidence=tuple(
                    float(value) for value in row_bin_ece_confidence[row_index, bin_index]
                ),
                ece_correct=tuple(
                    float(value) for value in row_bin_ece_correct[row_index, bin_index]
                ),
            )
            for bin_index in range(len(TIMESTEP_BINS))
        )
        row_nll.append(
            RowNll(
                sequence_id=row.sequence_id,
                homology_component_id=row.homology_component_id,
                union_component_id=row.union_component_id,
                sampling_weight=row.sampling_weight,
                nll=float(row_values[row_index, 0]),
                nll_by_timestep_bin=tuple(
                    float(value) for value in row_bin_values[row_index, :, 0]
                ),
                overall=row_overall,
                timestep_bins=row_timestep_bins,
            )
        )

    weights = tuple(row.sampling_weight for row in rows)

    def aggregate(
        values: FloatArray,
        mass: FloatArray,
        confidence: FloatArray,
        correct: FloatArray,
    ) -> MetricSummary:
        return _metric_summary(
            nll=math.fsum(weights[index] * float(values[index, 0]) for index in range(len(rows))),
            top1=math.fsum(weights[index] * float(values[index, 1]) for index in range(len(rows))),
            top3=math.fsum(weights[index] * float(values[index, 2]) for index in range(len(rows))),
            brier=math.fsum(weights[index] * float(values[index, 3]) for index in range(len(rows))),
            ece_mass=tuple(
                math.fsum(
                    weights[index] * float(mass[index, ece_bin]) for index in range(len(rows))
                )
                for ece_bin in range(ECE_BINS)
            ),
            ece_confidence=tuple(
                math.fsum(
                    weights[index] * float(confidence[index, ece_bin]) for index in range(len(rows))
                )
                for ece_bin in range(ECE_BINS)
            ),
            ece_correct=tuple(
                math.fsum(
                    weights[index] * float(correct[index, ece_bin]) for index in range(len(rows))
                )
                for ece_bin in range(ECE_BINS)
            ),
        )

    overall = aggregate(
        row_values,
        row_ece_mass,
        row_ece_confidence,
        row_ece_correct,
    )
    by_bin = tuple(
        aggregate(
            row_bin_values[:, bin_index, :],
            row_bin_ece_mass[:, bin_index, :],
            row_bin_ece_confidence[:, bin_index, :],
            row_bin_ece_correct[:, bin_index, :],
        )
        for bin_index in range(len(TIMESTEP_BINS))
    )
    outer_folds = {row.fold for row in rows}
    if len(outer_folds) != 1:  # pragma: no cover - ledger invariant
        raise RuntimeError("score rows changed folds while scoring")
    return _new_fold_method_metrics(
        method=method,
        checkpoint_step=checkpoint_step,
        outer_fold=outer_folds.pop(),
        row_nll=tuple(row_nll),
        overall=overall,
        timestep_bins=by_bin,
        louco_plan=louco_plan,
    )


def score_fold(
    archive: ScoringArchive,
    *,
    method: str,
    checkpoint_step: int | None = None,
) -> FoldMethodMetrics:
    """Score C0, cross-calibrated C0T, or cross-calibrated R128."""

    view = _validated_scoring_view(archive)
    result = _score_fold_from_view(
        view,
        method=method,
        checkpoint_step=checkpoint_step,
    )
    view.revalidate()
    return result


def score_all_fold_methods(archive: ScoringArchive) -> tuple[FoldMethodMetrics, ...]:
    """Score the evaluator's exact seven-method order under one validated view."""

    view = _validated_scoring_view(archive)
    result = (
        _score_fold_from_view(view, method="C0"),
        _score_fold_from_view(view, method="C0T"),
        *(
            _score_fold_from_view(view, method="R128", checkpoint_step=step)
            for step in CHECKPOINT_STEPS
        ),
    )
    view.revalidate()
    return result


@dataclass(frozen=True, slots=True)
class EqualFoldMetrics:
    """Four-fold equal-weight aggregate, including aggregate ECE."""

    method: str
    checkpoint_step: int | None
    folds: tuple[FoldMethodMetrics, ...]
    overall: MetricSummary
    timestep_bins: tuple[MetricSummary, ...]
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _consume_metric_factory_token(self._factory_token)
        object.__setattr__(self, "_factory_token", None)
        _validate_equal_fold_metrics(self)

    def revalidate(self) -> None:
        _validate_equal_fold_metrics(self)


def _validate_equal_fold_metrics(value: EqualFoldMetrics) -> None:
    if type(value.folds) is not tuple or any(
        type(fold) is not FoldMethodMetrics for fold in value.folds
    ):
        raise TypeError("equal-fold metrics require exact FoldMethodMetrics values")
    for fold in value.folds:
        fold.revalidate()
    if tuple(fold.outer_fold for fold in value.folds) != (0, 1, 2, 3):
        raise ValueError("equal-fold metrics require folds 0, 1, 2, 3 in order")
    if any(
        fold.method != value.method or fold.checkpoint_step != value.checkpoint_step
        for fold in value.folds
    ):
        raise ValueError("equal-fold metrics contain a method/checkpoint mismatch")
    if type(value.overall) is not MetricSummary or (
        type(value.timestep_bins) is not tuple
        or len(value.timestep_bins) != len(TIMESTEP_BINS)
        or any(type(summary) is not MetricSummary for summary in value.timestep_bins)
    ):
        raise TypeError("equal-fold metrics require exact overall and timestep summaries")
    value.overall.revalidate()
    for summary in value.timestep_bins:
        summary.revalidate()

    def expected_summary(summaries: Sequence[MetricSummary]) -> MetricSummary:
        items = tuple(summaries)
        return _metric_summary(
            nll=math.fsum(item.nll for item in items) / 4.0,
            top1=math.fsum(item.top1_accuracy for item in items) / 4.0,
            top3=math.fsum(item.top3_accuracy for item in items) / 4.0,
            brier=math.fsum(item.multiclass_brier for item in items) / 4.0,
            ece_mass=tuple(
                math.fsum(item.ece_mass[index] for item in items) / 4.0 for index in range(ECE_BINS)
            ),
            ece_confidence=tuple(
                math.fsum(item.ece_confidence[index] for item in items) / 4.0
                for index in range(ECE_BINS)
            ),
            ece_correct=tuple(
                math.fsum(item.ece_correct[index] for item in items) / 4.0
                for index in range(ECE_BINS)
            ),
        )

    if value.overall != expected_summary(tuple(fold.overall for fold in value.folds)):
        raise ValueError("equal-fold overall summary differs from its four folds")
    for index, summary in enumerate(value.timestep_bins):
        expected = expected_summary(tuple(fold.timestep_bins[index] for fold in value.folds))
        if summary != expected:
            raise ValueError("equal-fold timestep summary differs from its four folds")
    for attribute in (
        "sequence_id",
        "homology_component_id",
        "union_component_id",
    ):
        seen: set[str] = set()
        for fold in value.folds:
            identifiers = {getattr(row, attribute) for row in fold.row_nll}
            if seen & identifiers:
                raise ValueError(f"equal-fold {attribute} identities must be pairwise disjoint")
            seen.update(identifiers)


def _new_equal_fold_metrics(**values: object) -> EqualFoldMetrics:
    return EqualFoldMetrics(**values, _factory_token=_MetricFactoryToken())


def aggregate_equal_folds(
    folds: Sequence[FoldMethodMetrics],
) -> EqualFoldMetrics:
    """Aggregate four development folds with exact weight one quarter."""

    values = tuple(folds)
    if len(values) != 4 or any(type(value) is not FoldMethodMetrics for value in values):
        raise TypeError("folds must contain exact FoldMethodMetrics values")
    if tuple(value.outer_fold for value in values) != (0, 1, 2, 3):
        raise ValueError("folds must be exact outer folds 0, 1, 2, 3 in order")
    for value in values:
        value.revalidate()
    methods = {(value.method, value.checkpoint_step) for value in values}
    if len(methods) != 1:
        raise ValueError("all folds must describe one method/checkpoint")
    method, checkpoint_step = methods.pop()

    def aggregate(summaries: Sequence[MetricSummary]) -> MetricSummary:
        items = tuple(summaries)
        if len(items) != 4:
            raise ValueError("equal-fold aggregation requires four summaries")
        return _metric_summary(
            nll=math.fsum(value.nll for value in items) / 4.0,
            top1=math.fsum(value.top1_accuracy for value in items) / 4.0,
            top3=math.fsum(value.top3_accuracy for value in items) / 4.0,
            brier=math.fsum(value.multiclass_brier for value in items) / 4.0,
            ece_mass=tuple(
                math.fsum(value.ece_mass[index] for value in items) / 4.0
                for index in range(ECE_BINS)
            ),
            ece_confidence=tuple(
                math.fsum(value.ece_confidence[index] for value in items) / 4.0
                for index in range(ECE_BINS)
            ),
            ece_correct=tuple(
                math.fsum(value.ece_correct[index] for value in items) / 4.0
                for index in range(ECE_BINS)
            ),
        )

    return _new_equal_fold_metrics(
        method=method,
        checkpoint_step=checkpoint_step,
        folds=values,
        overall=aggregate(tuple(value.overall for value in values)),
        timestep_bins=tuple(
            aggregate(tuple(value.timestep_bins[index] for value in values))
            for index in range(len(TIMESTEP_BINS))
        ),
    )


def select_strongest_count_control(
    c0: EqualFoldMetrics,
    c0t: EqualFoldMetrics,
) -> EqualFoldMetrics:
    """Choose one global count comparator, breaking an exact tie toward C0."""

    if type(c0) is not EqualFoldMetrics or type(c0t) is not EqualFoldMetrics:
        raise TypeError("count controls must be exact EqualFoldMetrics values")
    c0.revalidate()
    c0t.revalidate()
    if c0.method != "C0" or c0t.method != "C0T":
        raise ValueError("count-control inputs must be ordered C0 then C0T")
    for left, right in zip(c0.folds, c0t.folds, strict=True):
        if tuple(row.sequence_id for row in left.row_nll) != tuple(
            row.sequence_id for row in right.row_nll
        ):
            raise ValueError("count-control score rows do not align")
    return c0 if c0.overall.nll <= c0t.overall.nll else c0t


@dataclass(frozen=True, slots=True)
class BootstrapPanel:
    """Shared fold-stratified union-component bootstrap NLL samples."""

    method_names: tuple[str, ...]
    point_mean_nll: tuple[float, ...]
    samples: NDArray[np.float64]
    component_counts_by_fold: tuple[int, ...]
    seed: int
    parent_contract_sha256: str
    draws_sha256: str
    arrays_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_bootstrap_panel(self)
        raw = object.__getattribute__(self, "samples")
        object.__setattr__(self, "samples", _owning_readonly_copy(raw))
        _validate_bootstrap_panel(self)
        object.__setattr__(
            self,
            "arrays_sha256",
            _arrays_integrity_sha256({"samples": object.__getattribute__(self, "samples")}),
        )

    def __getattribute__(self, name: str) -> object:
        if name == "samples":
            raw = object.__getattribute__(self, name)
            try:
                object.__getattribute__(self, "arrays_sha256")
            except AttributeError:
                return raw
            self.revalidate()
            return _defensive_array(raw)
        return object.__getattribute__(self, name)

    def revalidate(self) -> None:
        raw = object.__getattribute__(self, "samples")
        arrays = {"samples": raw}
        _require_owned_readonly_arrays(arrays, label="bootstrap panel")
        expected = _require_sha256(
            object.__getattribute__(self, "arrays_sha256"),
            label="bootstrap panel array integrity SHA-256",
        )
        if _arrays_integrity_sha256(arrays) != expected:
            raise ValueError("bootstrap samples changed after construction")
        _validate_bootstrap_panel(self)

    def _validated_samples(self) -> NDArray[np.float64]:
        self.revalidate()
        return object.__getattribute__(self, "samples")

    def values(self, method_name: str) -> NDArray[np.float64]:
        if type(method_name) is not str:
            raise TypeError("method_name must be an exact string")
        try:
            index = self.method_names.index(method_name)
        except ValueError as error:
            raise KeyError(method_name) from error
        return _defensive_array(self._validated_samples()[index])


def _validate_bootstrap_panel(panel: BootstrapPanel) -> None:
    if (
        not panel.method_names
        or len(set(panel.method_names)) != len(panel.method_names)
        or any(type(value) is not str or not value for value in panel.method_names)
    ):
        raise ValueError("bootstrap method names must be non-empty and unique")
    if (
        type(panel.point_mean_nll) is not tuple
        or len(panel.point_mean_nll) != len(panel.method_names)
        or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in panel.point_mean_nll
        )
    ):
        raise ValueError("bootstrap point NLLs must align with methods")
    samples = object.__getattribute__(panel, "samples")
    if (
        type(samples) is not np.ndarray
        or samples.dtype != np.dtype("<f8")
        or samples.ndim != 2
        or samples.shape[0] != len(panel.method_names)
        or samples.shape[1] < 2
        or not samples.flags.c_contiguous
        or not bool(np.isfinite(samples).all())
        or bool(np.any(samples < 0.0))
    ):
        raise ValueError("bootstrap samples must be finite aligned float64 NLLs")
    if (
        type(panel.component_counts_by_fold) is not tuple
        or len(panel.component_counts_by_fold) != 4
        or any(type(value) is not int or value < 2 for value in panel.component_counts_by_fold)
    ):
        raise ValueError("bootstrap requires at least two components in every fold")
    _require_uint64(panel.seed, label="bootstrap seed")
    if (
        _require_sha256(panel.parent_contract_sha256, label="parent contract SHA-256")
        != PARENT_CONFIG_SHA256
    ):
        raise ValueError("bootstrap panel parent digest differs from the frozen parent")
    _require_sha256(panel.draws_sha256, label="bootstrap draw transcript SHA-256")


def shared_stratified_bootstrap(
    methods: Sequence[tuple[str, EqualFoldMetrics]],
    *,
    parent_contract_sha256: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    require_contract_replicates: bool = True,
) -> BootstrapPanel:
    """Generate shared PCG64DXSM component draws and fixed-choice NLLs."""

    entries = tuple(methods)
    if not entries or any(
        type(entry) is not tuple
        or len(entry) != 2
        or type(entry[0]) is not str
        or not entry[0]
        or type(entry[1]) is not EqualFoldMetrics
        for entry in entries
    ):
        raise TypeError("methods must contain (name, EqualFoldMetrics) tuples")
    for _, method in entries:
        method.revalidate()
    names = tuple(entry[0] for entry in entries)
    if len(set(names)) != len(names):
        raise ValueError("bootstrap method names must be unique")
    if type(require_contract_replicates) is not bool:
        raise TypeError("require_contract_replicates must be an exact boolean")
    if type(replicates) is not int or replicates < 2:
        raise ValueError("bootstrap replicates must be an exact integer at least two")
    if require_contract_replicates and replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("pilot bootstrap requires exactly 10000 replicates")
    root_seed = _require_uint64(seed, label="bootstrap seed")
    parent = _require_sha256(parent_contract_sha256, label="parent contract SHA-256")
    if parent != PARENT_CONFIG_SHA256:
        raise ValueError("bootstrap parent contract digest differs from the frozen parent")
    if require_contract_replicates and root_seed != BOOTSTRAP_SEED:
        raise ValueError("pilot bootstrap requires the frozen root seed")
    references = entries[0][1].folds
    for _, method in entries[1:]:
        for reference, candidate in zip(references, method.folds, strict=True):
            if len(reference.row_nll) != len(candidate.row_nll):
                raise ValueError("bootstrap methods have different score-row censuses")
            for left, right in zip(reference.row_nll, candidate.row_nll, strict=True):
                if (
                    left.sequence_id != right.sequence_id
                    or left.homology_component_id != right.homology_component_id
                    or left.union_component_id != right.union_component_id
                    or left.sampling_weight.hex() != right.sampling_weight.hex()
                ):
                    raise ValueError("bootstrap method rows do not align exactly")

    fold_components: list[tuple[str, ...]] = []
    fold_masses: list[dict[str, float]] = []
    fold_numerators: list[list[dict[str, float]]] = []
    for fold_index in range(4):
        reference_rows = references[fold_index].row_nll
        component_ids = tuple(sorted({row.union_component_id for row in reference_rows}))
        if len(component_ids) < 2:
            raise ValueError("bootstrap needs at least two union components per fold")
        grouped: dict[str, list[int]] = defaultdict(list)
        for row_index, row in enumerate(reference_rows):
            grouped[row.union_component_id].append(row_index)
        masses = {
            component_id: math.fsum(
                reference_rows[index].sampling_weight for index in grouped[component_id]
            )
            for component_id in component_ids
        }
        if math.fsum(masses[value] for value in component_ids).hex() != (1.0).hex():
            raise ValueError("bootstrap reference fold mass must equal binary64 one")
        method_numerators: list[dict[str, float]] = []
        for _, method in entries:
            rows = method.folds[fold_index].row_nll
            method_numerators.append(
                {
                    component_id: math.fsum(
                        rows[index].sampling_weight * rows[index].nll
                        for index in grouped[component_id]
                    )
                    for component_id in component_ids
                }
            )
        fold_components.append(component_ids)
        fold_masses.append(masses)
        fold_numerators.append(method_numerators)

    samples = np.empty((len(entries), replicates), dtype="<f8")
    transcript = hashlib.sha256()
    transcript.update(_BOOTSTRAP_TRANSCRIPT_DOMAIN)
    for replicate in range(replicates):
        fold_estimates = np.empty((len(entries), 4), dtype="<f8")
        for fold_index in range(4):
            component_ids = fold_components[fold_index]
            rng_seed = namespaced_seed(
                root_seed,
                "bootstrap",
                parent,
                replicate,
                fold_index,
            )
            rng = np.random.Generator(np.random.PCG64DXSM(rng_seed))
            draws = rng.integers(
                0,
                len(component_ids),
                size=len(component_ids),
                dtype=np.int64,
            )
            multiplicities: Counter[str] = Counter()
            for slot, draw in enumerate(draws):
                component_id = component_ids[int(draw)]
                multiplicities[component_id] += 1
                transcript.update(
                    f"{replicate}\t{fold_index}\t{slot}\t{component_id}\n".encode("ascii")
                )
            mass = math.fsum(
                multiplicities[component_id] * fold_masses[fold_index][component_id]
                for component_id in component_ids
            )
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError("bootstrap draw has invalid fold mass")
            for method_index in range(len(entries)):
                numerator = math.fsum(
                    multiplicities[component_id]
                    * fold_numerators[fold_index][method_index][component_id]
                    for component_id in component_ids
                )
                estimate = numerator / mass
                if not math.isfinite(estimate) or estimate < 0.0:
                    raise ValueError("bootstrap draw has an invalid NLL")
                fold_estimates[method_index, fold_index] = estimate
        for method_index in range(len(entries)):
            samples[method_index, replicate] = (
                math.fsum(fold_estimates[method_index].tolist()) / 4.0
            )
    return BootstrapPanel(
        method_names=names,
        point_mean_nll=tuple(entry[1].overall.nll for entry in entries),
        samples=samples,
        component_counts_by_fold=tuple(len(value) for value in fold_components),
        seed=root_seed,
        parent_contract_sha256=parent,
        draws_sha256=transcript.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    """Earliest R128 checkpoint within one bootstrap SD of the best."""

    checkpoint_steps: tuple[int, ...]
    mean_nll: tuple[float, ...]
    best_checkpoint_step: int
    best_bootstrap_standard_error: float
    eligibility_threshold: float
    eligible_checkpoint_steps: tuple[int, ...]
    selected_checkpoint_step: int

    def __post_init__(self) -> None:
        if self.checkpoint_steps != CHECKPOINT_STEPS:
            raise ValueError("checkpoint selection must use the frozen step order")
        if len(self.mean_nll) != len(CHECKPOINT_STEPS) or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in self.mean_nll
        ):
            raise ValueError("checkpoint mean NLL values are invalid")
        if (
            self.best_checkpoint_step not in CHECKPOINT_STEPS
            or self.selected_checkpoint_step not in CHECKPOINT_STEPS
            or not self.eligible_checkpoint_steps
            or any(value not in CHECKPOINT_STEPS for value in self.eligible_checkpoint_steps)
        ):
            raise ValueError("checkpoint selection contains a non-frozen step")
        if self.selected_checkpoint_step != min(self.eligible_checkpoint_steps):
            raise ValueError("checkpoint selection did not choose the earliest eligible step")
        _require_finite_float(
            self.best_bootstrap_standard_error,
            label="best bootstrap standard error",
        )
        _require_finite_float(self.eligibility_threshold, label="eligibility threshold")
        best_index = min(
            range(len(self.mean_nll)),
            key=lambda index: (self.mean_nll[index], self.checkpoint_steps[index]),
        )
        if self.best_checkpoint_step != self.checkpoint_steps[best_index]:
            raise ValueError("checkpoint selection best step differs from its mean NLLs")
        expected_threshold = self.mean_nll[best_index] + self.best_bootstrap_standard_error
        if self.eligibility_threshold.hex() != expected_threshold.hex():
            raise ValueError("checkpoint selection threshold is internally inconsistent")
        expected_eligible = tuple(
            step
            for step, mean in zip(self.checkpoint_steps, self.mean_nll, strict=True)
            if mean <= expected_threshold
        )
        if self.eligible_checkpoint_steps != expected_eligible:
            raise ValueError("checkpoint selection eligibility differs from its threshold")

    def revalidate(self) -> None:
        self.__post_init__()


def select_r128_checkpoint(
    checkpoints: Sequence[EqualFoldMetrics],
    bootstrap_mean_nll: NDArray[np.float64],
) -> CheckpointSelection:
    """Apply the pilot's preregistered one-standard-error checkpoint rule."""

    values = tuple(checkpoints)
    if (
        len(values) != len(CHECKPOINT_STEPS)
        or any(type(value) is not EqualFoldMetrics for value in values)
        or tuple(value.checkpoint_step for value in values) != CHECKPOINT_STEPS
        or any(value.method != "R128" for value in values)
    ):
        raise ValueError("checkpoints must be the five ordered R128 equal-fold metrics")
    for value in values:
        value.revalidate()
    if (
        type(bootstrap_mean_nll) is not np.ndarray
        or bootstrap_mean_nll.dtype != np.dtype("<f8")
        or bootstrap_mean_nll.ndim != 2
        or bootstrap_mean_nll.shape[0] != len(CHECKPOINT_STEPS)
        or bootstrap_mean_nll.shape[1] != BOOTSTRAP_REPLICATES
        or not bool(np.isfinite(bootstrap_mean_nll).all())
        or bool(np.any(bootstrap_mean_nll < 0.0))
    ):
        raise ValueError("checkpoint bootstrap NLL must be finite float64 shape (5, 10000)")
    means = tuple(value.overall.nll for value in values)
    best_index = min(
        range(len(values)),
        key=lambda index: (means[index], CHECKPOINT_STEPS[index]),
    )
    standard_error = float(np.std(bootstrap_mean_nll[best_index], ddof=1))
    if not math.isfinite(standard_error) or standard_error < 0.0:
        raise ValueError("best checkpoint bootstrap standard error is invalid")
    threshold = means[best_index] + standard_error
    eligible = tuple(
        step for step, mean in zip(CHECKPOINT_STEPS, means, strict=True) if mean <= threshold
    )
    return CheckpointSelection(
        checkpoint_steps=CHECKPOINT_STEPS,
        mean_nll=means,
        best_checkpoint_step=CHECKPOINT_STEPS[best_index],
        best_bootstrap_standard_error=standard_error,
        eligibility_threshold=threshold,
        eligible_checkpoint_steps=eligible,
        selected_checkpoint_step=min(eligible),
    )


@dataclass(frozen=True, slots=True)
class VerifiedPilotEvidence:
    """Path-free digest bindings minted only after independent reconstruction.

    This value does not replace the signed-off receipt or repository result
    pin.  It prevents the numerical gate from accepting loose caller booleans
    and records exactly which independently checked evidence set it consumed.
    """

    schema_version: int
    child_contract_sha256: str
    parent_contract_sha256: str
    git_commit: str
    outer_folds: tuple[int, ...]
    bindings: Mapping[str, str]
    checks: Mapping[str, bool]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("verified pilot evidence schema_version must be exact integer 1")
        if self.child_contract_sha256 != CONFIG_SHA256:
            raise ValueError("verified evidence child contract digest changed")
        if self.parent_contract_sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("verified evidence parent contract digest changed")
        if (
            type(self.git_commit) is not str
            or re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None
        ):
            raise ValueError("verified evidence Git commit must be a lowercase object ID")
        if self.outer_folds != (0, 1, 2, 3):
            raise ValueError("verified evidence must cover outer folds 0, 1, 2, 3")
        if not isinstance(self.bindings, Mapping) or set(self.bindings) != set(
            _VERIFIED_EVIDENCE_BINDING_FIELDS
        ):
            raise ValueError("verified evidence binding schema changed")
        frozen_bindings = {
            key: _require_sha256(self.bindings[key], label=f"verified evidence {key}")
            for key in _VERIFIED_EVIDENCE_BINDING_FIELDS
        }
        if not isinstance(self.checks, Mapping) or set(self.checks) != set(
            _VERIFIED_EVIDENCE_CHECK_FIELDS
        ):
            raise ValueError("verified evidence check schema changed")
        frozen_checks: dict[str, bool] = {}
        for key in _VERIFIED_EVIDENCE_CHECK_FIELDS:
            value = self.checks[key]
            if type(value) is not bool or value is not True:
                raise ValueError("verified evidence requires every reconstruction check to pass")
            frozen_checks[key] = value
        object.__setattr__(self, "bindings", MappingProxyType(frozen_bindings))
        object.__setattr__(self, "checks", MappingProxyType(frozen_checks))
        expected = hashlib.sha256(_canonical_json_bytes(self._unsigned_record())).hexdigest()
        if _require_sha256(self.evidence_sha256, label="verified evidence SHA-256") != expected:
            raise ValueError("verified evidence SHA-256 differs from its complete bindings")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "child_contract_sha256": self.child_contract_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "git_commit": self.git_commit,
            "outer_folds": list(self.outer_folds),
            "bindings": dict(self.bindings),
            "checks": dict(self.checks),
        }

    def canonical_record(self) -> dict[str, object]:
        """Return complete evidence identity as a fresh canonical record."""

        self.revalidate()
        return {**self._unsigned_record(), "evidence_sha256": self.evidence_sha256}

    def revalidate(self) -> None:
        self.__post_init__()


def build_verified_pilot_evidence(
    *,
    child_contract_sha256: str,
    parent_contract_sha256: str,
    git_commit: str,
    bindings: Mapping[str, str],
    checks: Mapping[str, bool],
) -> VerifiedPilotEvidence:
    """Bind the independent verifier's exact digest maps and all-true checks."""

    unsigned = {
        "schema_version": 1,
        "child_contract_sha256": child_contract_sha256,
        "parent_contract_sha256": parent_contract_sha256,
        "git_commit": git_commit,
        "outer_folds": [0, 1, 2, 3],
        "bindings": dict(bindings),
        "checks": dict(checks),
    }
    digest = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    return VerifiedPilotEvidence(
        schema_version=1,
        child_contract_sha256=child_contract_sha256,
        parent_contract_sha256=parent_contract_sha256,
        git_commit=git_commit,
        outer_folds=(0, 1, 2, 3),
        bindings=bindings,
        checks=checks,
        evidence_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class PilotGateDecision:
    """Fail-closed scientific pilot decision."""

    status: str
    candidate_checkpoint_step: int
    comparator_method: str
    point_relative_nll_improvement: float | None
    bootstrap_lower_95: float | None
    bootstrap_upper_95: float | None
    fold_relative_improvement: tuple[float, ...]
    timestep_bin_relative_improvement: tuple[float, ...]
    checks: Mapping[str, bool]
    verified_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "invalid_run",
            "development_no_go_v1_pilot",
            "development_continue_v1_full_matrix_authorized",
        }:
            raise ValueError("pilot decision status is invalid")
        if self.candidate_checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError("pilot decision checkpoint is invalid")
        if self.comparator_method not in {"C0", "C0T"}:
            raise ValueError("pilot comparator must be C0 or C0T")
        for value in (
            self.point_relative_nll_improvement,
            self.bootstrap_lower_95,
            self.bootstrap_upper_95,
        ):
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise ValueError("pilot decision contains a non-finite comparison")
        if not isinstance(self.checks, Mapping) or any(
            type(key) is not str or type(value) is not bool for key, value in self.checks.items()
        ):
            raise ValueError("pilot gate checks must be exact booleans")
        frozen_checks = MappingProxyType(dict(self.checks))
        object.__setattr__(self, "checks", frozen_checks)
        validity_keys = {"all_four_outer_fits", "evidence_valid"}
        scientific_keys = {
            "minimum_mean_relative_nll_improvement",
            "bootstrap_lower_bound_strictly_positive",
            "maximum_outer_fold_relative_nll_regression",
            "maximum_timestep_bin_relative_nll_regression",
            "maximum_ece",
            "maximum_ece_regression",
        }
        if self.status == "invalid_run":
            if (
                set(self.checks) != validity_keys
                or all(self.checks.values())
                or self.point_relative_nll_improvement is not None
                or self.bootstrap_lower_95 is not None
                or self.bootstrap_upper_95 is not None
                or self.fold_relative_improvement != ()
                or self.timestep_bin_relative_improvement != ()
                or self.verified_evidence_sha256 is not None
            ):
                raise ValueError("invalid_run must be coupled only to failed validity gates")
            return
        if (
            self.verified_evidence_sha256 is None
            or _SHA256_RE.fullmatch(self.verified_evidence_sha256) is None
        ):
            raise ValueError(
                "scientific pilot decisions require the pending independent evidence receipt"
            )
        if (
            set(self.checks) != validity_keys | scientific_keys
            or not all(self.checks[key] for key in validity_keys)
            or self.point_relative_nll_improvement is None
            or self.bootstrap_lower_95 is None
            or self.bootstrap_upper_95 is None
            or type(self.fold_relative_improvement) is not tuple
            or len(self.fold_relative_improvement) != 4
            or type(self.timestep_bin_relative_improvement) is not tuple
            or len(self.timestep_bin_relative_improvement) != len(TIMESTEP_BINS)
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    *self.fold_relative_improvement,
                    *self.timestep_bin_relative_improvement,
                )
            )
        ):
            raise ValueError("scientific pilot decision is not coupled to complete evidence")
        passed = all(self.checks.values())
        if passed != (self.status == "development_continue_v1_full_matrix_authorized"):
            raise ValueError("pilot decision status differs from its coupled gate checks")

    def revalidate(self) -> None:
        self.__post_init__()


def _evaluate_fixed_pilot_gate(
    candidate: EqualFoldMetrics,
    comparator: EqualFoldMetrics,
    *,
    candidate_bootstrap_mean_nll: NDArray[np.float64],
    comparator_bootstrap_mean_nll: NDArray[np.float64],
    all_four_outer_fits: bool,
    evidence_valid: bool,
    verified_evidence: VerifiedPilotEvidence | None = None,
) -> PilotGateDecision:
    """Apply all R128 continuation gates using one fixed global comparator."""

    if type(candidate) is not EqualFoldMetrics or candidate.method != "R128":
        raise TypeError("candidate must be R128 EqualFoldMetrics")
    if type(comparator) is not EqualFoldMetrics or comparator.method not in {"C0", "C0T"}:
        raise TypeError("comparator must be C0 or C0T EqualFoldMetrics")
    candidate.revalidate()
    comparator.revalidate()
    if type(all_four_outer_fits) is not bool or type(evidence_valid) is not bool:
        raise TypeError("pilot validity flags must be exact booleans")
    if all_four_outer_fits and evidence_valid and verified_evidence is None:
        raise ValueError("positive pilot authorization requires an independent evidence receipt")
    if verified_evidence is not None:
        if type(verified_evidence) is not VerifiedPilotEvidence:
            raise TypeError("verified_evidence must be an exact VerifiedPilotEvidence")
        verified_evidence.revalidate()
        if not all_four_outer_fits or not evidence_valid:
            raise ValueError("verified evidence cannot accompany a failed validity gate")
    for name, raw in (
        ("candidate", candidate_bootstrap_mean_nll),
        ("comparator", comparator_bootstrap_mean_nll),
    ):
        if (
            type(raw) is not np.ndarray
            or raw.dtype != np.dtype("<f8")
            or raw.ndim != 1
            or len(raw) != BOOTSTRAP_REPLICATES
            or not bool(np.isfinite(raw).all())
            or bool(np.any(raw < 0.0))
        ):
            raise ValueError(f"{name} bootstrap NLL must be a finite float64 vector")
    if candidate_bootstrap_mean_nll.shape != comparator_bootstrap_mean_nll.shape:
        raise ValueError("candidate and comparator bootstrap samples must align")
    if candidate.checkpoint_step is None:  # pragma: no cover - class invariant
        raise RuntimeError("R128 candidate lost its checkpoint")
    if not evidence_valid or not all_four_outer_fits:
        return PilotGateDecision(
            status="invalid_run",
            candidate_checkpoint_step=candidate.checkpoint_step,
            comparator_method=comparator.method,
            point_relative_nll_improvement=None,
            bootstrap_lower_95=None,
            bootstrap_upper_95=None,
            fold_relative_improvement=(),
            timestep_bin_relative_improvement=(),
            checks=MappingProxyType(
                {
                    "all_four_outer_fits": all_four_outer_fits,
                    "evidence_valid": evidence_valid,
                }
            ),
            verified_evidence_sha256=None,
        )
    if comparator.overall.nll <= 0.0 or np.any(comparator_bootstrap_mean_nll <= 0.0):
        raise ValueError("relative improvement requires positive comparator NLL")
    for candidate_fold, comparator_fold in zip(candidate.folds, comparator.folds, strict=True):
        if tuple(row.sequence_id for row in candidate_fold.row_nll) != tuple(
            row.sequence_id for row in comparator_fold.row_nll
        ):
            raise ValueError("candidate and comparator score rows do not align")
    point = (comparator.overall.nll - candidate.overall.nll) / comparator.overall.nll
    relative_samples = (
        comparator_bootstrap_mean_nll - candidate_bootstrap_mean_nll
    ) / comparator_bootstrap_mean_nll
    if not bool(np.isfinite(relative_samples).all()):
        raise ValueError("bootstrap relative improvement is non-finite")
    lower, upper = np.quantile(relative_samples, [0.025, 0.975], method="linear")
    fold_relative = tuple(
        (right.overall.nll - left.overall.nll) / right.overall.nll
        for left, right in zip(candidate.folds, comparator.folds, strict=True)
    )
    if any(not math.isfinite(value) for value in fold_relative):
        raise ValueError("fold relative improvement is non-finite")
    timestep_relative = tuple(
        (right.nll - left.nll) / right.nll
        for left, right in zip(
            candidate.timestep_bins,
            comparator.timestep_bins,
            strict=True,
        )
    )
    if any(not math.isfinite(value) for value in timestep_relative):
        raise ValueError("timestep-bin relative improvement is non-finite")
    candidate_ece = (
        *(value.overall.ece for value in candidate.folds),
        candidate.overall.ece,
    )
    comparator_ece = (
        *(value.overall.ece for value in comparator.folds),
        comparator.overall.ece,
    )
    checks_dict = {
        "all_four_outer_fits": True,
        "evidence_valid": True,
        "minimum_mean_relative_nll_improvement": point >= 0.02,
        "bootstrap_lower_bound_strictly_positive": float(lower) > 0.0,
        "maximum_outer_fold_relative_nll_regression": all(
            value >= -0.01 for value in fold_relative
        ),
        "maximum_timestep_bin_relative_nll_regression": all(
            value >= -0.01 for value in timestep_relative
        ),
        "maximum_ece": all(value <= 0.10 for value in candidate_ece),
        "maximum_ece_regression": all(
            left - right <= 0.02 for left, right in zip(candidate_ece, comparator_ece, strict=True)
        ),
    }
    passed = all(checks_dict.values())
    return PilotGateDecision(
        status=(
            "development_continue_v1_full_matrix_authorized"
            if passed
            else "development_no_go_v1_pilot"
        ),
        candidate_checkpoint_step=candidate.checkpoint_step,
        comparator_method=comparator.method,
        point_relative_nll_improvement=float(point),
        bootstrap_lower_95=float(lower),
        bootstrap_upper_95=float(upper),
        fold_relative_improvement=fold_relative,
        timestep_bin_relative_improvement=timestep_relative,
        checks=MappingProxyType(checks_dict),
        verified_evidence_sha256=verified_evidence.evidence_sha256,
    )


@dataclass(frozen=True, slots=True)
class PilotEvaluation:
    """Coupled checkpoint, comparator, shared-bootstrap, and gate result."""

    checkpoint_selection: CheckpointSelection
    comparator_method: str
    bootstrap: BootstrapPanel
    decision: PilotGateDecision

    def __post_init__(self) -> None:
        if type(self.checkpoint_selection) is not CheckpointSelection:
            raise TypeError("pilot evaluation checkpoint selection has an invalid type")
        if type(self.bootstrap) is not BootstrapPanel:
            raise TypeError("pilot evaluation bootstrap has an invalid type")
        if type(self.decision) is not PilotGateDecision:
            raise TypeError("pilot evaluation decision has an invalid type")
        self.checkpoint_selection.revalidate()
        self.bootstrap.revalidate()
        self.decision.revalidate()
        if self.comparator_method not in {"C0", "C0T"}:
            raise ValueError("pilot evaluation comparator must be C0 or C0T")
        if self.decision.comparator_method != self.comparator_method:
            raise ValueError("pilot decision and comparator selection disagree")
        if (
            self.decision.candidate_checkpoint_step
            != self.checkpoint_selection.selected_checkpoint_step
        ):
            raise ValueError("pilot decision did not gate the selected checkpoint")
        if (
            self.bootstrap.samples.shape[1] != BOOTSTRAP_REPLICATES
            or self.bootstrap.seed != BOOTSTRAP_SEED
        ):
            raise ValueError("pilot evaluation did not use the frozen bootstrap")
        expected_methods = (
            *(f"R128-{step:06d}" for step in CHECKPOINT_STEPS),
            "C0",
            "C0T",
        )
        if self.bootstrap.method_names != expected_methods:
            raise ValueError("pilot evaluation bootstrap method order is not frozen")
        if self.bootstrap.point_mean_nll[: len(CHECKPOINT_STEPS)] != (
            self.checkpoint_selection.mean_nll
        ):
            raise ValueError("pilot evaluation point NLLs disagree with checkpoint selection")
        expected_comparator = (
            "C0"
            if self.bootstrap.point_mean_nll[len(CHECKPOINT_STEPS)]
            <= self.bootstrap.point_mean_nll[len(CHECKPOINT_STEPS) + 1]
            else "C0T"
        )
        if self.comparator_method != expected_comparator:
            raise ValueError("pilot evaluation did not select the strongest count comparator")
        best_index = CHECKPOINT_STEPS.index(self.checkpoint_selection.best_checkpoint_step)
        expected_standard_error = float(np.std(self.bootstrap.samples[best_index], ddof=1))
        if (
            not math.isfinite(expected_standard_error)
            or self.checkpoint_selection.best_bootstrap_standard_error.hex()
            != expected_standard_error.hex()
        ):
            raise ValueError("checkpoint selection standard error differs from bootstrap evidence")

    def revalidate(self) -> None:
        self.__post_init__()

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return ``pilot_bootstrap.npz`` arrays in frozen contract order."""

        self.revalidate()
        comparator_offset = 0 if self.comparator_method == "C0" else 1
        comparator_index = len(CHECKPOINT_STEPS) + comparator_offset
        comparator_samples = np.ascontiguousarray(
            self.bootstrap.samples[comparator_index],
            dtype="<f8",
        )
        if np.any(comparator_samples <= 0.0):
            raise ValueError("pilot bootstrap relative improvement requires positive control NLL")
        r128_samples = np.ascontiguousarray(
            self.bootstrap.samples[: len(CHECKPOINT_STEPS)],
            dtype="<f8",
        )
        relative = np.ascontiguousarray(
            (comparator_samples[None, :] - r128_samples) / comparator_samples[None, :],
            dtype="<f8",
        )
        return {
            "checkpoint_step": np.asarray(CHECKPOINT_STEPS, dtype="<u2"),
            "r128_mean_nll": np.asarray(
                self.bootstrap.point_mean_nll[: len(CHECKPOINT_STEPS)],
                dtype="<f8",
            ),
            "c0_mean_nll": np.asarray(
                [self.bootstrap.point_mean_nll[len(CHECKPOINT_STEPS)]],
                dtype="<f8",
            ),
            "c0t_mean_nll": np.asarray(
                [self.bootstrap.point_mean_nll[len(CHECKPOINT_STEPS) + 1]],
                dtype="<f8",
            ),
            "selected_comparator_index": np.asarray(
                [comparator_offset],
                dtype="|u1",
            ),
            "r128_bootstrap_mean_nll": r128_samples.copy(),
            "comparator_bootstrap_mean_nll": comparator_samples.copy(),
            "relative_improvement": relative,
        }

    def npz_bytes(self) -> bytes:
        """Return deterministic bytes for the coupled pilot bootstrap artifact."""

        self.revalidate()
        return deterministic_npz_bytes(pilot_bootstrap_npz_schema(), self.arrays())


def evaluate_pilot_gate(
    checkpoints: Sequence[EqualFoldMetrics],
    c0: EqualFoldMetrics,
    c0t: EqualFoldMetrics,
    *,
    parent_contract_sha256: str,
) -> PilotEvaluation:
    """Select the pilot result and return a fail-closed unauthenticated gate.

    The shared 10,000-draw panel is built internally so an unpaired or
    undersized array cannot authorize continuation.  Public authorization is
    intentionally unavailable on this unauthenticated path; callers must use
    :func:`evaluate_verified_pilot_gate` with a digest-bound evidence value.
    Raw caller booleans are not accepted as proof of completed outer fits or
    valid evidence.
    """

    values = tuple(checkpoints)
    if (
        len(values) != len(CHECKPOINT_STEPS)
        or any(type(value) is not EqualFoldMetrics for value in values)
        or tuple(value.checkpoint_step for value in values) != CHECKPOINT_STEPS
        or any(value.method != "R128" for value in values)
    ):
        raise ValueError("pilot evaluation requires all five ordered R128 checkpoints")
    if type(c0) is not EqualFoldMetrics or c0.method != "C0":
        raise TypeError("pilot evaluation requires raw C0 equal-fold metrics")
    if type(c0t) is not EqualFoldMetrics or c0t.method != "C0T":
        raise TypeError("pilot evaluation requires C0T equal-fold metrics")
    for value in (*values, c0, c0t):
        value.revalidate()
    method_entries = (
        *(
            (f"R128-{step:06d}", metric)
            for step, metric in zip(CHECKPOINT_STEPS, values, strict=True)
        ),
        ("C0", c0),
        ("C0T", c0t),
    )
    bootstrap = shared_stratified_bootstrap(
        method_entries,
        parent_contract_sha256=parent_contract_sha256,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        require_contract_replicates=True,
    )
    checkpoint_samples = np.asarray(bootstrap.samples[: len(CHECKPOINT_STEPS)], dtype="<f8")
    selection = select_r128_checkpoint(values, checkpoint_samples)
    comparator = select_strongest_count_control(c0, c0t)
    selected_index = CHECKPOINT_STEPS.index(selection.selected_checkpoint_step)
    comparator_index = len(CHECKPOINT_STEPS) + (0 if comparator.method == "C0" else 1)
    decision = _evaluate_fixed_pilot_gate(
        values[selected_index],
        comparator,
        candidate_bootstrap_mean_nll=bootstrap.samples[selected_index],
        comparator_bootstrap_mean_nll=bootstrap.samples[comparator_index],
        all_four_outer_fits=False,
        evidence_valid=False,
    )
    return PilotEvaluation(
        checkpoint_selection=selection,
        comparator_method=comparator.method,
        bootstrap=bootstrap,
        decision=decision,
    )


def evaluate_verified_pilot_gate(
    checkpoints: Sequence[EqualFoldMetrics],
    c0: EqualFoldMetrics,
    c0t: EqualFoldMetrics,
    *,
    parent_contract_sha256: str,
    verified_evidence: VerifiedPilotEvidence,
) -> PilotEvaluation:
    """Evaluate the fixed scientific gates against independently bound evidence.

    The numerical panel is constructed through the same fail-closed public
    path, then the decision alone is reissued with a digest-bound evidence
    capability.  Loose booleans remain insufficient.
    """

    if type(verified_evidence) is not VerifiedPilotEvidence:
        raise TypeError("verified_evidence must be an exact VerifiedPilotEvidence")
    verified_evidence.revalidate()
    if verified_evidence.parent_contract_sha256 != parent_contract_sha256:
        raise ValueError("verified evidence and requested parent contract differ")
    values = tuple(checkpoints)
    unauthenticated = evaluate_pilot_gate(
        values,
        c0,
        c0t,
        parent_contract_sha256=parent_contract_sha256,
    )
    selected_step = unauthenticated.checkpoint_selection.selected_checkpoint_step
    selected_index = CHECKPOINT_STEPS.index(selected_step)
    comparator = select_strongest_count_control(c0, c0t)
    comparator_index = len(CHECKPOINT_STEPS) + (0 if comparator.method == "C0" else 1)
    decision = _evaluate_fixed_pilot_gate(
        values[selected_index],
        comparator,
        candidate_bootstrap_mean_nll=unauthenticated.bootstrap.samples[selected_index],
        comparator_bootstrap_mean_nll=unauthenticated.bootstrap.samples[comparator_index],
        all_four_outer_fits=True,
        evidence_valid=True,
        verified_evidence=verified_evidence,
    )
    return PilotEvaluation(
        checkpoint_selection=unauthenticated.checkpoint_selection,
        comparator_method=unauthenticated.comparator_method,
        bootstrap=unauthenticated.bootstrap,
        decision=decision,
    )

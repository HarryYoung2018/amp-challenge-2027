"""Trainer-only data and count-prior primitives for the v1 R128 pilot.

The public execution layer is intentionally elsewhere.  This module accepts
only the three fields exposed by a frozen outer-fold training projection,
authenticates the complete JSONL byte stream before parsing it, and builds the
fold-local ``C0`` table without consulting score rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1.pilot_contract import (
    NativeDiffusionV1PilotContract,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 8
MAX_LENGTH = 50
LENGTH_EDGES = (8, 15, 20, 25, 33, 51)
RELATIVE_POSITION_BINS = 10
UNIGRAM_PSEUDOCOUNT = 0.5
RELATIVE_POSITION_PRIOR_MASS = 20.0
LOG_FLOOR = 1e-12

_MAX_PROJECTION_BYTES = 8 * 1024 * 1024
_MAX_COUNT_PRIOR_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRAIN_FIELDS = frozenset({"sequence_id", "sequence", "sampling_weight"})
_DOS_EPOCH = (1980, 1, 1, 0, 0, 0)
_COUNT_PRIOR_SEAL_CAPABILITY = object()
_COUNT_PRIOR_SCHEMA = (
    ("effective_count_scale", np.dtype("<f8"), (1,)),
    ("unigram_probability", np.dtype("<f8"), (20,)),
    ("length_edges", np.dtype("|u1"), (6,)),
    ("relative_position_probability", np.dtype("<f8"), (5, 10, 20)),
    ("log_relative_position_probability", np.dtype("<f8"), (5, 10, 20)),
)


@dataclass(frozen=True, slots=True)
class PilotTrainingRow:
    """The complete information available to one v1 pilot trainer."""

    sequence_id: str
    sequence: str
    sampling_weight: float

    def __post_init__(self) -> None:
        _require_sha256(self.sequence_id, label="sequence_id")
        if type(self.sequence) is not str:
            raise TypeError("sequence must be an exact string")
        try:
            canonical = canonicalize_sequence(
                self.sequence,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("sequence must be a canonical peptide of length 8..50") from error
        if canonical != self.sequence or canonical_sequence_id(canonical) != self.sequence_id:
            raise ValueError("sequence_id does not match the canonical sequence")
        if (
            type(self.sampling_weight) is not float
            or not math.isfinite(self.sampling_weight)
            or self.sampling_weight <= 0.0
        ):
            raise ValueError("sampling_weight must be a positive finite JSON float")


@dataclass(frozen=True, slots=True)
class PilotTrainingProjection:
    """An authenticated, strictly ordered, trainer-only outer-fold projection."""

    rows: tuple[PilotTrainingRow, ...]
    sha256: str
    byte_count: int
    token_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, label="projection sha256")
        if type(self.rows) is not tuple or not self.rows:
            raise ValueError("training projection rows must be a non-empty tuple")
        if any(type(row) is not PilotTrainingRow for row in self.rows):
            raise TypeError("training projection may contain only PilotTrainingRow values")
        identifiers = tuple(row.sequence_id for row in self.rows)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("training projection rows must be unique and ordered by sequence_id")
        if math.fsum(row.sampling_weight for row in self.rows).hex() != (1.0).hex():
            raise ValueError("training projection weights must sum to exactly binary64 one")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        expected_tokens = sum(len(row.sequence) for row in self.rows)
        if type(self.token_count) is not int or self.token_count != expected_tokens:
            raise ValueError("token_count does not match the training rows")
        payload = self.canonical_bytes()
        if len(payload) != self.byte_count:
            raise ValueError("byte_count does not match canonical training projection bytes")
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError("projection sha256 does not match canonical training rows")

    @property
    def probabilities(self) -> tuple[float, ...]:
        return tuple(row.sampling_weight for row in self.rows)

    def canonical_bytes(self) -> bytes:
        """Reconstruct the exact canonical JSONL authenticated by ``sha256``."""

        return b"".join(
            _canonical_json_bytes(
                {
                    "sequence_id": row.sequence_id,
                    "sequence": row.sequence,
                    "sampling_weight": row.sampling_weight,
                }
            )
            for row in self.rows
        )

    def revalidate(self) -> bytes:
        """Recheck rows, ordering, census, physical bytes, and identity."""

        if type(self) is not PilotTrainingProjection:
            raise TypeError("projection must be an exact PilotTrainingProjection")
        for row in self.rows:
            if type(row) is not PilotTrainingRow:
                raise TypeError("training projection contains an invalid row type")
            PilotTrainingRow(row.sequence_id, row.sequence, row.sampling_weight)
        self.__post_init__()
        return self.canonical_bytes()


@dataclass(frozen=True, slots=True)
class CountPrior:
    """Validated float64 fold-local C0 arrays.

    ``CountPrior`` is the decoded value, not an artifact identity.  Code that
    crosses a trainer, scorer, bridge, or metadata boundary must use
    :class:`AuthenticatedCountPrior`, which binds these arrays to the exact
    deterministic NPZ bytes and their physical SHA-256.
    """

    effective_count_scale: NDArray[np.float64]
    unigram_probability: NDArray[np.float64]
    length_edges: NDArray[np.uint8]
    relative_position_probability: NDArray[np.float64]
    log_relative_position_probability: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = {
            "effective_count_scale": self.effective_count_scale,
            "unigram_probability": self.unigram_probability,
            "length_edges": self.length_edges,
            "relative_position_probability": self.relative_position_probability,
            "log_relative_position_probability": self.log_relative_position_probability,
        }
        validated: dict[str, NDArray[np.generic]] = {}
        for name, dtype, shape in _COUNT_PRIOR_SCHEMA:
            raw = values[name]
            if type(raw) is not np.ndarray:
                raise TypeError(f"count-prior member {name} must be an ndarray")
            if raw.dtype != dtype:
                raise TypeError(
                    f"count-prior member {name} dtype must be {dtype.str}, got {raw.dtype.str}"
                )
            if raw.shape != shape:
                raise ValueError(f"count-prior member {name} must have shape {shape}")
            if raw.dtype.kind == "f" and not bool(np.isfinite(raw).all()):
                raise ValueError(f"count-prior member {name} must be finite")
            copied = np.ascontiguousarray(raw).copy()
            copied.flags.writeable = False
            validated[name] = copied
        scale = cast(NDArray[np.float64], validated["effective_count_scale"])
        if scale[0] <= 0.0 or not float(scale[0]).is_integer():
            raise ValueError("effective_count_scale must contain one positive integer-valued float")
        edges = cast(NDArray[np.uint8], validated["length_edges"])
        if not np.array_equal(edges, np.asarray(LENGTH_EDGES, dtype="|u1")):
            raise ValueError("length_edges differ from the frozen C0 bins")
        unigram = cast(NDArray[np.float64], validated["unigram_probability"])
        relative = cast(NDArray[np.float64], validated["relative_position_probability"])
        logs = cast(NDArray[np.float64], validated["log_relative_position_probability"])
        if np.any(unigram <= 0.0) or not math.isclose(
            math.fsum(unigram.tolist()), 1.0, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("unigram_probability must be positive and normalized")
        if np.any(relative < LOG_FLOOR):
            raise ValueError("relative_position_probability violates the log floor")
        if not np.allclose(np.sum(relative, axis=2), 1.0, rtol=0.0, atol=1e-15):
            raise ValueError("relative_position_probability cells must be normalized")
        if not np.array_equal(logs, np.log(relative)):
            raise ValueError("log_relative_position_probability is not the exact NumPy log")
        for name, value in validated.items():
            object.__setattr__(self, name, value)

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return defensive read-only copies in the frozen NPZ member order."""

        result: dict[str, NDArray[np.generic]] = {}
        for name, _, _ in _COUNT_PRIOR_SCHEMA:
            copied = np.ascontiguousarray(getattr(self, name)).copy()
            copied.flags.writeable = False
            result[name] = copied
        return result

    def npz_bytes(self) -> bytes:
        return deterministic_count_prior_npz_bytes(self.arrays())

    @property
    def npz_sha256(self) -> str:
        return hashlib.sha256(self.npz_bytes()).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class AuthenticatedCountPrior:
    """One inseparable C0 payload, physical digest, and decoded array value.

    NumPy's writeable flag is not a security boundary: callers can deliberately
    re-enable it on an owning array.  Consequently every consumer boundary must
    call :meth:`revalidate`.  That method hashes and reparses the immutable NPZ
    payload, enforces the deterministic archive form and full ``CountPrior``
    semantics, and then compares every stored array bit-for-bit.
    """

    payload: bytes
    sha256: str
    prior: CountPrior

    def __post_init__(self) -> None:
        self.revalidate()

    @classmethod
    def from_projection(
        cls,
        projection: PilotTrainingProjection,
    ) -> AuthenticatedCountPrior:
        """Build and seal the exact fold-local C0 for one projection."""

        if type(projection) is not PilotTrainingProjection:
            raise TypeError("projection must be a PilotTrainingProjection")
        projection.revalidate()
        prior = build_count_prior(projection)
        payload = prior.npz_bytes()
        return cls(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            prior=prior,
        )

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> AuthenticatedCountPrior:
        """Authenticate exact NPZ bytes before exposing their decoded C0."""

        expected = _require_sha256(
            expected_sha256,
            label="expected count-prior sha256",
        )
        prior = _decode_count_prior_npz_bytes(payload, expected_sha256=expected)
        return cls(payload=payload, sha256=expected, prior=prior)

    def revalidate(self) -> CountPrior:
        """Return a fresh decoded prior after rechecking every binding."""

        if type(self.payload) is not bytes:
            raise TypeError("authenticated count-prior payload must be exact bytes")
        digest = _require_sha256(
            self.sha256,
            label="authenticated count-prior sha256",
        )
        if type(self.prior) is not CountPrior:
            raise TypeError("authenticated count-prior value must be a CountPrior")
        decoded = _decode_count_prior_npz_bytes(
            self.payload,
            expected_sha256=digest,
        )
        stored = _revalidate_stored_count_prior(self.prior)
        if not _count_priors_equal(stored, decoded):
            raise ValueError("authenticated count-prior arrays do not match the sealed NPZ payload")
        return decoded

    def __eq__(self, other: object) -> bool:
        if type(other) is not AuthenticatedCountPrior:
            return NotImplemented
        return self.sha256 == other.sha256 and self.payload == other.payload


@dataclass(frozen=True, slots=True, eq=False)
class SealedCountPrior:
    """A C0 artifact continuously bound to its immutable trainer-root file."""

    artifact: AuthenticatedCountPrior
    trainer_root: Path
    path: Path
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _COUNT_PRIOR_SEAL_CAPABILITY:
            raise RuntimeError("sealed count prior requires the internal file capability")
        self.revalidate()

    def revalidate(self) -> AuthenticatedCountPrior:
        if type(self.artifact) is not AuthenticatedCountPrior:
            raise TypeError("sealed count-prior artifact has an invalid type")
        if not isinstance(self.trainer_root, Path) or not isinstance(self.path, Path):
            raise TypeError("sealed count-prior paths must be pathlib.Path values")
        root = Path(os.path.abspath(os.fspath(self.trainer_root)))
        path = Path(os.path.abspath(os.fspath(self.path)))
        if root != self.trainer_root or path != self.path or path != root / "count_prior.npz":
            raise ValueError("sealed count-prior path escaped its exact trainer root")
        _reject_symlink_chain(root)
        try:
            root_stat = os.lstat(root)
        except OSError as error:
            raise ValueError("sealed count-prior trainer root is unavailable") from error
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("sealed count-prior trainer root is not a directory")
        payload = _read_regular_bytes(
            path,
            maximum_bytes=_MAX_COUNT_PRIOR_BYTES,
            label="sealed count-prior archive",
            required_mode=0o444,
        )
        observed = AuthenticatedCountPrior.from_bytes(
            payload,
            expected_sha256=self.artifact.sha256,
        )
        self.artifact.revalidate()
        if (
            observed != self.artifact
            or observed.prior.npz_bytes() != self.artifact.prior.npz_bytes()
        ):
            raise ValueError("sealed count-prior file differs from its authenticated value")
        return self.artifact


def seal_count_prior_npz(
    projection: PilotTrainingProjection,
    trainer_root: str | os.PathLike[str],
) -> SealedCountPrior:
    """Build, no-overwrite publish, reopen, and bind one fold-local C0 file."""

    if type(projection) is not PilotTrainingProjection:
        raise TypeError("projection must be an exact PilotTrainingProjection")
    projection.revalidate()
    root = Path(os.path.abspath(os.fspath(trainer_root)))
    _reject_symlink_chain(root)
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise ValueError("trainer root must already exist") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("trainer root must be a non-symlink directory")
    path = root / "count_prior.npz"
    if os.path.lexists(path):
        raise FileExistsError("count-prior publication is strictly no-overwrite")
    artifact = AuthenticatedCountPrior.from_projection(projection)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=".pilot-c0-", dir=root)
    temporary = Path(raw_temporary)
    published = False
    try:
        view = memoryview(artifact.payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short count-prior write")
            written += count
        os.fdatasync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        published = True
        os.unlink(temporary)
        _fsync_directory(root)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    if not published:  # pragma: no cover - successful link sets this first
        raise RuntimeError("count-prior publication did not complete")
    return SealedCountPrior(
        artifact=artifact,
        trainer_root=root,
        path=path,
        _capability=_COUNT_PRIOR_SEAL_CAPABILITY,
    )


def load_sealed_count_prior_npz(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    trainer_root: str | os.PathLike[str] | None = None,
) -> SealedCountPrior:
    """Load an existing exact trainer-root C0 file into a continuous receipt."""

    source = Path(os.path.abspath(os.fspath(path)))
    root = source.parent if trainer_root is None else Path(os.path.abspath(os.fspath(trainer_root)))
    if source != root / "count_prior.npz":
        raise ValueError("sealed count-prior path must be trainer_root/count_prior.npz")
    artifact = load_count_prior_npz(source, expected_sha256=expected_sha256)
    return SealedCountPrior(
        artifact=artifact,
        trainer_root=root,
        path=source,
        _capability=_COUNT_PRIOR_SEAL_CAPABILITY,
    )


def load_pilot_training_projection(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_rows: int,
) -> PilotTrainingProjection:
    """Authenticate and parse one exact three-field training projection."""

    expected = _require_sha256(expected_sha256, label="expected projection sha256")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive exact integer")
    payload = _read_regular_bytes(
        path,
        maximum_bytes=_MAX_PROJECTION_BYTES,
        label="training projection",
    )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"training projection SHA-256 expected {expected}, got {observed}")
    documents = _parse_canonical_jsonl(payload)
    if len(documents) != expected_rows:
        raise ValueError(f"training projection expected {expected_rows} rows, got {len(documents)}")
    rows: list[PilotTrainingRow] = []
    for number, document in enumerate(documents, start=1):
        label = f"training projection row {number}"
        if set(document) != _TRAIN_FIELDS:
            raise ValueError(
                f"{label} schema mismatch: missing={sorted(_TRAIN_FIELDS - set(document))}, "
                f"extra={sorted(set(document) - _TRAIN_FIELDS)}"
            )
        sequence_id = _require_sha256(document["sequence_id"], label=f"{label} sequence_id")
        sequence = document["sequence"]
        weight = document["sampling_weight"]
        if type(sequence) is not str:
            raise TypeError(f"{label} sequence must be an exact string")
        if type(weight) is not float:
            raise TypeError(f"{label} sampling_weight must be an exact JSON float")
        rows.append(PilotTrainingRow(sequence_id, sequence, weight))
    result = PilotTrainingProjection(
        rows=tuple(rows),
        sha256=observed,
        byte_count=len(payload),
        token_count=sum(len(row.sequence) for row in rows),
    )
    if result.sha256 != expected:  # pragma: no cover - construction invariant
        raise RuntimeError("projection identity changed during construction")
    return result


def load_contract_fold_training_projection(
    contract: NativeDiffusionV1PilotContract,
    outer_fold: int,
    path: str | os.PathLike[str],
) -> PilotTrainingProjection:
    """Load node-local train-only bytes using one authenticated fold's exact pins."""

    if type(contract) is not NativeDiffusionV1PilotContract:
        raise TypeError("contract must be a NativeDiffusionV1PilotContract")
    contract.revalidate()
    fold = contract.fold(outer_fold)
    projection = load_pilot_training_projection(
        path,
        expected_sha256=fold.train_sha256,
        expected_rows=fold.train_rows,
    )
    if projection.sha256 != fold.train_sha256 or len(projection.rows) != fold.train_rows:
        raise RuntimeError("authenticated fold projection changed after loading")
    return projection


def build_count_prior(projection: PilotTrainingProjection) -> CountPrior:
    """Build the parent-pinned C0 table in exact ascending row order."""

    if type(projection) is not PilotTrainingProjection:
        raise TypeError("projection must be a PilotTrainingProjection")
    projection.revalidate()
    rows = projection.rows
    scale = len(rows)
    residue_index = {residue: index for index, residue in enumerate(ALPHABET)}
    unigram_counts = np.zeros(20, dtype="<f8")
    position_counts = np.zeros((5, RELATIVE_POSITION_BINS, 20), dtype="<f8")
    edges = np.asarray(LENGTH_EDGES, dtype="|u1")
    for row in rows:
        length = len(row.sequence)
        residue_mass = scale * row.sampling_weight / length
        length_bin = int(np.searchsorted(edges, length, side="right") - 1)
        if not 0 <= length_bin < 5:  # pragma: no cover - row validation invariant
            raise RuntimeError("validated training length has no C0 length bin")
        for position, residue in enumerate(row.sequence):
            index = residue_index[residue]
            unigram_counts[index] += residue_mass
            position_bin = min(9, (10 * position) // length)
            position_counts[length_bin, position_bin, index] += residue_mass
    if not math.isclose(
        float(np.sum(unigram_counts)),
        float(scale),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("effective unigram evidence does not equal the training-row count")
    unigram = unigram_counts + UNIGRAM_PSEUDOCOUNT
    unigram /= np.sum(unigram)
    relative = position_counts + RELATIVE_POSITION_PRIOR_MASS * unigram[None, None, :]
    relative /= np.sum(relative, axis=2, keepdims=True)
    relative = np.maximum(relative, LOG_FLOOR)
    relative /= np.sum(relative, axis=2, keepdims=True)
    log_relative = np.log(relative)
    return CountPrior(
        effective_count_scale=np.asarray([float(scale)], dtype="<f8"),
        unigram_probability=np.asarray(unigram, dtype="<f8"),
        length_edges=edges,
        relative_position_probability=np.asarray(relative, dtype="<f8"),
        log_relative_position_probability=np.asarray(log_relative, dtype="<f8"),
    )


def deterministic_count_prior_npz_bytes(
    arrays: Mapping[str, NDArray[np.generic]],
) -> bytes:
    """Serialize the exact count-prior schema as deterministic ZIP_STORED NPY v1.0."""

    if type(arrays) is not dict:
        raise TypeError("count-prior arrays must be a plain dict")
    expected_names = tuple(name for name, _, _ in _COUNT_PRIOR_SCHEMA)
    if tuple(arrays) != expected_names:
        raise ValueError("count-prior arrays must follow the exact schema member order")
    prepared: list[tuple[str, NDArray[np.generic]]] = []
    for name, dtype, shape in _COUNT_PRIOR_SCHEMA:
        raw = arrays[name]
        if type(raw) is not np.ndarray:
            raise TypeError(f"count-prior member {name} must be an ndarray")
        if raw.dtype != dtype:
            raise TypeError(
                f"count-prior member {name} dtype must be {dtype.str}, got {raw.dtype.str}"
            )
        if raw.shape != shape:
            raise ValueError(f"count-prior member {name} must have shape {shape}")
        if raw.dtype.hasobject:
            raise TypeError("object arrays are forbidden in count-prior archives")
        if raw.dtype.kind == "f" and not bool(np.isfinite(raw).all()):
            raise ValueError(f"count-prior member {name} contains a non-finite value")
        prepared.append((name, np.ascontiguousarray(raw)))

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
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


def _decode_count_prior_npz_bytes(payload: bytes, *, expected_sha256: str) -> CountPrior:
    """Decode one byte-pinned deterministic C0 archive without permitting pickle."""

    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_COUNT_PRIOR_BYTES:
        raise ValueError("count-prior archive has an invalid byte size")
    expected = _require_sha256(expected_sha256, label="expected count-prior sha256")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"count-prior SHA-256 expected {expected}, got {observed}")
    arrays: dict[str, NDArray[np.generic]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            expected_members = tuple(f"{name}.npy" for name, _, _ in _COUNT_PRIOR_SCHEMA)
            members = archive.infolist()
            if tuple(member.filename for member in members) != expected_members:
                raise ValueError("count-prior archive has an invalid member inventory or order")
            if any(
                member.compress_type != zipfile.ZIP_STORED
                or member.file_size <= 0
                or member.file_size > _MAX_COUNT_PRIOR_BYTES
                or member.flag_bits & 0x1
                for member in members
            ):
                raise ValueError("count-prior archive has unsafe ZIP member metadata")
            if sum(member.file_size for member in members) > _MAX_COUNT_PRIOR_BYTES:
                raise ValueError("count-prior archive members exceed the byte bound")
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            expected_names = tuple(name for name, _, _ in _COUNT_PRIOR_SCHEMA)
            if tuple(loaded.files) != expected_names:
                raise ValueError("count-prior archive has an invalid member inventory or order")
            for name, _, _ in _COUNT_PRIOR_SCHEMA:
                arrays[name] = loaded[name].copy()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("count-prior archive is not valid pickle-free NPZ data") from error
    if deterministic_count_prior_npz_bytes(arrays) != payload:
        raise ValueError("count-prior archive bytes are not in the deterministic frozen format")
    return CountPrior(
        effective_count_scale=cast(NDArray[np.float64], arrays["effective_count_scale"]),
        unigram_probability=cast(NDArray[np.float64], arrays["unigram_probability"]),
        length_edges=cast(NDArray[np.uint8], arrays["length_edges"]),
        relative_position_probability=cast(
            NDArray[np.float64], arrays["relative_position_probability"]
        ),
        log_relative_position_probability=cast(
            NDArray[np.float64], arrays["log_relative_position_probability"]
        ),
    )


def load_count_prior_npz_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
) -> AuthenticatedCountPrior:
    """Return the single authenticated representation of in-memory C0 bytes."""

    return AuthenticatedCountPrior.from_bytes(
        payload,
        expected_sha256=expected_sha256,
    )


def load_count_prior_npz(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> AuthenticatedCountPrior:
    """Snapshot and load a sealed single-link count_prior.npz artifact."""

    payload = _read_regular_bytes(
        path,
        maximum_bytes=_MAX_COUNT_PRIOR_BYTES,
        label="count-prior archive",
        required_mode=0o444,
    )
    return AuthenticatedCountPrior.from_bytes(
        payload,
        expected_sha256=expected_sha256,
    )


def _revalidate_stored_count_prior(prior: CountPrior) -> CountPrior:
    values: dict[str, NDArray[np.generic]] = {}
    for name, _, _ in _COUNT_PRIOR_SCHEMA:
        raw = getattr(prior, name)
        if type(raw) is not np.ndarray:
            raise TypeError(f"count-prior member {name} must remain an ndarray")
        if raw.flags.writeable:
            raise ValueError(f"count-prior member {name} must remain read-only")
        values[name] = raw
    return CountPrior(
        effective_count_scale=cast(NDArray[np.float64], values["effective_count_scale"]),
        unigram_probability=cast(NDArray[np.float64], values["unigram_probability"]),
        length_edges=cast(NDArray[np.uint8], values["length_edges"]),
        relative_position_probability=cast(
            NDArray[np.float64], values["relative_position_probability"]
        ),
        log_relative_position_probability=cast(
            NDArray[np.float64], values["log_relative_position_probability"]
        ),
    )


def _count_priors_equal(left: CountPrior, right: CountPrior) -> bool:
    for name, dtype, shape in _COUNT_PRIOR_SCHEMA:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if (
            type(left_value) is not np.ndarray
            or type(right_value) is not np.ndarray
            or left_value.dtype != dtype
            or right_value.dtype != dtype
            or left_value.shape != shape
            or right_value.shape != shape
            or not np.array_equal(left_value, right_value)
        ):
            return False
    return True


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
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


def _reject_nonfinite(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label=label)


def _parse_canonical_jsonl(payload: bytes) -> tuple[dict[str, Any], ...]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("training projection must be non-empty canonical LF JSONL")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(payload[:-1].split(b"\n"), start=1):
        if not raw:
            raise ValueError(f"training projection line {number} is blank")
        try:
            document = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"training projection line {number} is invalid JSON") from error
        if type(document) is not dict:
            raise ValueError(f"training projection line {number} must be an object")
        _reject_nonfinite(document, label=f"training projection line {number}")
        if _canonical_json_bytes(document) != raw + b"\n":
            raise ValueError(f"training projection line {number} is not canonical compact JSON")
        rows.append(cast(dict[str, Any], document))
    return tuple(rows)


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect pilot data path: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"pilot data path traverses a symlink: {candidate}")


def _fsync_directory(path: Path) -> None:
    _reject_symlink_chain(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes(
    path: str | os.PathLike[str],
    *,
    maximum_bytes: int,
    label: str,
    required_mode: int | None = None,
) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {source}") from error
    if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link non-symlink regular file")
    if required_mode is not None and stat.S_IMODE(named_before.st_mode) != required_mode:
        raise ValueError(f"{label} must have exact mode {required_mode:04o}")
    if not 0 < named_before.st_size <= maximum_bytes:
        raise ValueError(f"{label} size must be 1..{maximum_bytes} bytes")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or opened_before.st_nlink != 1:
            raise ValueError(f"{label} must remain a single-link regular file")
        if required_mode is not None and stat.S_IMODE(opened_before.st_mode) != required_mode:
            raise ValueError(f"{label} mode changed from exact {required_mode:04o}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(65536, maximum_bytes + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    _reject_symlink_chain(source)
    payload = b"".join(chunks)
    if (
        _fingerprint(named_before) != _fingerprint(opened_before)
        or _fingerprint(opened_before) != _fingerprint(opened_after)
        or _fingerprint(opened_before) != _fingerprint(named_after)
        or len(payload) != opened_before.st_size
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload

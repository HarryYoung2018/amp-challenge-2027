"""Strict, framework-neutral data contract for native categorical diffusion.

The accepted corpus contains both development and validation rows.  This module
makes the split explicit and constructs the only admissible v0 training
distribution: component-equal sampling over folds 0--3.  Fold 4 never
contributes to row draws or to the separately modeled length prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

ACCEPTED_CORPUS_SHA256 = "c03595a8650b732307ada7e030f996d65345238efaea55d25925ac433bec8bc7"
ACCEPTED_TRAINING_PROJECTION_SHA256 = (
    "127a0eb88c5dc10c94904dcc5a3e98ff75a55890dc29af807e99f3f93c61ae46"
)
TRAIN_FOLDS = frozenset({0, 1, 2, 3})
VALIDATION_FOLD = 4
MIN_LENGTH = 8
MAX_LENGTH = 50

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROW_FIELDS = frozenset(
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
_TRAINING_PROJECTION_FIELDS = frozenset({"sequence_id", "sequence", "sampling_weight"})
_SEED_DOMAIN = b"amp-challenge/native-categorical-diffusion/stateless-seed/v1\0"


@dataclass(frozen=True, slots=True)
class DiffusionCorpusRow:
    """One fully validated row from categorical-diffusion corpus v1."""

    sequence_id: str
    sequence: str
    fold: int
    role: str
    homology_component_id: str
    homology_component_size: int
    union_component_id: str
    sampling_weight: float


@dataclass(frozen=True, slots=True)
class NativeDiffusionCorpus:
    """Canonical corpus rows with an explicit immutable role boundary."""

    rows: tuple[DiffusionCorpusRow, ...]
    sha256: str

    @property
    def train_rows(self) -> tuple[DiffusionCorpusRow, ...]:
        return tuple(row for row in self.rows if row.role == "train")

    @property
    def validation_rows(self) -> tuple[DiffusionCorpusRow, ...]:
        return tuple(row for row in self.rows if row.role == "validation")


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """The complete trainer-visible projection allowed by the v0 contract."""

    sequence_id: str
    sequence: str
    sampling_weight: float


@dataclass(frozen=True, slots=True)
class LengthPrior:
    """Component-weighted empirical length distribution from training only."""

    lengths: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.lengths
            or len(self.lengths) != len(self.probabilities)
            or tuple(sorted(set(self.lengths))) != self.lengths
            or any(
                type(length) is not int or not MIN_LENGTH <= length <= MAX_LENGTH
                for length in self.lengths
            )
        ):
            raise ValueError("length prior requires unique sorted integer lengths in 8..50")
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in self.probabilities
        ) or not math.isclose(
            math.fsum(self.probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "length prior probabilities must be positive finite floats summing to one"
            )

    def draw(
        self,
        *,
        root_seed: int,
        draw_start: int,
        draw_count: int,
        namespace: str = "proposal",
    ) -> tuple[int, ...]:
        indices = _draw_indices(
            self.probabilities,
            root_seed=root_seed,
            draw_start=draw_start,
            draw_count=draw_count,
            namespace=namespace,
        )
        return tuple(self.lengths[index] for index in indices)


@dataclass(frozen=True, slots=True)
class TrainingDistribution:
    """Rows drawn according to corpus weights, with no second loss weighting.

    ``probabilities`` are the normalized corpus ``sampling_weight`` values.
    A training loss should average the per-sequence losses from :meth:`draw`
    directly; multiplying those losses by the row weights again would square
    the intended component correction.
    """

    rows: tuple[TrainingRow, ...]
    probabilities: tuple[float, ...]
    length_prior: LengthPrior

    def __post_init__(self) -> None:
        if not self.rows or len(self.rows) != len(self.probabilities):
            raise ValueError("training rows and probabilities must be non-empty and aligned")
        if tuple(row.sequence_id for row in self.rows) != tuple(
            sorted(row.sequence_id for row in self.rows)
        ):
            raise ValueError("training rows must be ordered by sequence_id")
        if self.probabilities != tuple(row.sampling_weight for row in self.rows):
            raise ValueError("training probabilities must equal the single row-draw weights")
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in self.probabilities
        ) or not math.isclose(
            math.fsum(self.probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("training probabilities must be positive finite floats summing to one")

    def draw(
        self,
        *,
        root_seed: int,
        draw_start: int,
        draw_count: int,
        namespace: str = "minibatch",
    ) -> tuple[TrainingRow, ...]:
        indices = _draw_indices(
            self.probabilities,
            root_seed=root_seed,
            draw_start=draw_start,
            draw_count=draw_count,
            namespace=namespace,
        )
        return tuple(self.rows[index] for index in indices)


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


def _read_regular_bytes(path: str | Path) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect diffusion corpus: {source}") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError("diffusion corpus must be a regular, non-symbolic file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open diffusion corpus: {source}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(source)
    except OSError as error:
        raise ValueError("diffusion corpus changed while it was read") from error
    _reject_symlink_chain(source)
    fingerprints = {
        (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            stat.S_IMODE(value.st_mode),
        )
        for value in (named_before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != opened_before.st_size:
        raise ValueError("diffusion corpus changed while it was read")
    return payload


def _reject_symlink_chain(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect diffusion data path: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"diffusion data path traverses a symlink: {candidate}")


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parse_rows(payload: bytes) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError("diffusion corpus must be non-empty canonical LF-delimited JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload[:-1].split(b"\n"), start=1):
        if not raw_line:
            raise ValueError(f"diffusion corpus line {line_number} is blank")
        try:
            document = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"diffusion corpus line {line_number} is invalid JSON") from error
        if not isinstance(document, dict):
            raise ValueError(f"diffusion corpus line {line_number} must be an object")
        _reject_nonfinite(document, label=f"diffusion corpus line {line_number}")
        if raw_line + b"\n" != _canonical_json_bytes(document):
            raise ValueError(f"diffusion corpus line {line_number} is not canonical compact JSON")
        rows.append(cast(dict[str, Any], document))
    return rows


def _row(document: Mapping[str, object], *, number: int) -> DiffusionCorpusRow:
    label = f"diffusion corpus row {number}"
    if set(document) != _ROW_FIELDS:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(_ROW_FIELDS - set(document))}, "
            f"extra={sorted(set(document) - _ROW_FIELDS)}"
        )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError(f"{label} schema_version must be integer 1")
    sequence_id = _sha256(document["sequence_id"], label=f"{label} sequence_id")
    sequence_raw = document["sequence"]
    if not isinstance(sequence_raw, str):
        raise ValueError(f"{label} sequence must be a string")
    sequence = canonicalize_sequence(sequence_raw)
    if sequence != sequence_raw:
        raise ValueError(f"{label} sequence is not canonical")
    if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(f"{label} sequence length must lie in {MIN_LENGTH}..{MAX_LENGTH}")
    if canonical_sequence_id(sequence) != sequence_id:
        raise ValueError(f"{label} sequence_id does not match sequence")

    fold = document["fold"]
    if type(fold) is not int or fold not in {*TRAIN_FOLDS, VALIDATION_FOLD}:
        raise ValueError(f"{label} fold must be an integer in 0..4")
    expected_role = "validation" if fold == VALIDATION_FOLD else "train"
    role = document["role"]
    if role != expected_role:
        raise ValueError(f"{label} role does not match its fold")
    homology_id = _sha256(document["homology_component_id"], label=f"{label} homology_component_id")
    union_id = _sha256(document["union_component_id"], label=f"{label} union_component_id")
    component_size = document["homology_component_size"]
    if type(component_size) is not int or component_size <= 0:
        raise ValueError(f"{label} homology_component_size must be a positive integer")
    weight = document["sampling_weight"]
    if type(weight) is not float or not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"{label} sampling_weight must be a positive finite JSON float")
    return DiffusionCorpusRow(
        sequence_id=sequence_id,
        sequence=sequence,
        fold=fold,
        role=cast(str, role),
        homology_component_id=homology_id,
        homology_component_size=component_size,
        union_component_id=union_id,
        sampling_weight=weight,
    )


def _validate_population(rows: Sequence[DiffusionCorpusRow]) -> None:
    if not rows:
        raise ValueError("diffusion corpus has no rows")
    sequence_ids = tuple(row.sequence_id for row in rows)
    if sequence_ids != tuple(sorted(sequence_ids)) or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("diffusion corpus rows must be strictly ordered by sequence_id")
    if {row.fold for row in rows} != {*TRAIN_FOLDS, VALIDATION_FOLD}:
        raise ValueError("diffusion corpus must contain every fold 0..4")

    homology_owner: dict[str, tuple[str, int, str]] = {}
    union_owner: dict[str, tuple[int, str]] = {}
    component_sizes: Counter[tuple[str, str]] = Counter()
    components_by_role: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        owner = (row.union_component_id, row.fold, row.role)
        if homology_owner.setdefault(row.homology_component_id, owner) != owner:
            raise ValueError("a homology component crosses a union, fold, or role boundary")
        union_role = (row.fold, row.role)
        if union_owner.setdefault(row.union_component_id, union_role) != union_role:
            raise ValueError("a union component crosses a fold or role boundary")
        component_sizes[(row.role, row.homology_component_id)] += 1
        components_by_role[row.role].add(row.homology_component_id)

    for role in ("train", "validation"):
        role_rows = tuple(row for row in rows if row.role == role)
        if not role_rows or not components_by_role[role]:
            raise ValueError(f"diffusion corpus has no {role} rows")
        component_count = len(components_by_role[role])
        for row in role_rows:
            actual_size = component_sizes[(role, row.homology_component_id)]
            if row.homology_component_size != actual_size:
                raise ValueError("homology_component_size differs from the observed corpus size")
            expected_weight = 1.0 / (component_count * actual_size)
            if not math.isclose(
                row.sampling_weight,
                expected_weight,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("sampling_weight differs from the component-equal formula")
        if not math.isclose(
            math.fsum(row.sampling_weight for row in role_rows),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{role} sampling weights do not sum to one")


def load_native_diffusion_corpus(
    path: str | Path,
    *,
    expected_sha256: str = ACCEPTED_CORPUS_SHA256,
) -> NativeDiffusionCorpus:
    """Load the exact accepted corpus, or an explicitly hash-pinned fixture."""

    expected = _sha256(expected_sha256, label="expected corpus sha256")
    payload = _read_regular_bytes(path)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"diffusion corpus SHA-256 expected {expected}, got {observed}")
    rows = tuple(
        _row(document, number=number) for number, document in enumerate(_parse_rows(payload), 1)
    )
    _validate_population(rows)
    return NativeDiffusionCorpus(rows=rows, sha256=observed)


def namespaced_seed(root_seed: int, namespace: str, *parts: str | int) -> int:
    """Derive one stable uint64 seed without mutable global RNG state."""

    if (
        isinstance(root_seed, bool)
        or not isinstance(root_seed, int | np.integer)
        or not 0 <= int(root_seed) < 2**64
    ):
        raise ValueError("root_seed must be an unsigned 64-bit integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    payload = bytearray(_SEED_DOMAIN)

    def append(tag: bytes, value: bytes) -> None:
        payload.extend(tag)
        payload.extend(len(value).to_bytes(8, "big"))
        payload.extend(value)

    append(b"r", int(root_seed).to_bytes(8, "big"))
    append(b"n", namespace.encode("utf-8"))
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, str | int | np.integer):
            raise TypeError("seed namespace parts must be strings or integers")
        if isinstance(part, str):
            append(b"s", part.encode("utf-8"))
        else:
            append(b"i", str(int(part)).encode("ascii"))
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _uniform(root_seed: int, namespace: str, ordinal: int) -> float:
    bits = namespaced_seed(root_seed, namespace, ordinal)
    return (bits >> 11) / float(1 << 53)


def _draw_indices(
    probabilities: Sequence[float],
    *,
    root_seed: int,
    draw_start: int,
    draw_count: int,
    namespace: str,
) -> tuple[int, ...]:
    if isinstance(draw_start, bool) or not isinstance(draw_start, int) or draw_start < 0:
        raise ValueError("draw_start must be a non-negative integer")
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 0:
        raise ValueError("draw_count must be a non-negative integer")
    namespaced_seed(root_seed, namespace, draw_start)
    values = tuple(float(value) for value in probabilities)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("sampling probabilities must be non-empty, positive, and finite")
    total = math.fsum(values)
    normalized = np.asarray([value / total for value in values], dtype=np.float64)
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    result: list[int] = []
    for ordinal in range(draw_start, draw_start + draw_count):
        index = int(
            np.searchsorted(
                cumulative,
                _uniform(root_seed, namespace, ordinal),
                side="right",
            )
        )
        result.append(min(index, len(values) - 1))
    return tuple(result)


def build_training_distribution(corpus: NativeDiffusionCorpus) -> TrainingDistribution:
    """Build a row-order-invariant folds-0--3 distribution and length prior."""

    invalid = [
        row.sequence_id
        for row in corpus.rows
        if (row.role == "train") != (row.fold in TRAIN_FOLDS)
        or (row.role == "validation") != (row.fold == VALIDATION_FOLD)
    ]
    if invalid:
        raise ValueError("corpus role/fold boundary is inconsistent")
    source_rows = tuple(
        sorted(
            (row for row in corpus.rows if row.role == "train"),
            key=lambda row: row.sequence_id,
        )
    )
    if not source_rows or any(row.fold == VALIDATION_FOLD for row in source_rows):
        raise ValueError("training distribution must contain folds 0--3 and no fold-4 rows")
    total = math.fsum(row.sampling_weight for row in source_rows)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("training-row sampling weights must sum to one")
    # Preserve the corpus values byte-for-value in the trainer-visible rows.
    # ``_draw_indices`` normalizes its CDF defensively, so rewriting accepted
    # weights here would only introduce an unnecessary second representation.
    probabilities = tuple(row.sampling_weight for row in source_rows)
    rows = tuple(
        TrainingRow(
            sequence_id=row.sequence_id,
            sequence=row.sequence,
            sampling_weight=probability,
        )
        for row, probability in zip(source_rows, probabilities, strict=True)
    )

    length_mass: dict[int, list[float]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        length_mass[len(row.sequence)].append(probability)
    lengths = tuple(sorted(length_mass))
    raw_length_probabilities = tuple(math.fsum(length_mass[length]) for length in lengths)
    length_total = math.fsum(raw_length_probabilities)
    length_prior = LengthPrior(
        lengths=lengths,
        probabilities=tuple(value / length_total for value in raw_length_probabilities),
    )
    return TrainingDistribution(
        rows=rows,
        probabilities=probabilities,
        length_prior=length_prior,
    )


def load_training_distribution(
    path: str | Path,
    *,
    expected_sha256: str = ACCEPTED_CORPUS_SHA256,
) -> TrainingDistribution:
    """Load a corpus but return only the three trainer-authorized fields."""

    corpus = load_native_diffusion_corpus(path, expected_sha256=expected_sha256)
    return build_training_distribution(corpus)


def training_projection_bytes(corpus: NativeDiffusionCorpus) -> bytes:
    """Return the canonical three-field training-only view of a corpus."""

    distribution = build_training_distribution(corpus)
    return b"".join(
        _canonical_json_bytes(
            {
                "sequence_id": row.sequence_id,
                "sequence": row.sequence,
                "sampling_weight": row.sampling_weight,
            }
        )
        for row in distribution.rows
    )


def _distribution_from_training_rows(rows: Sequence[TrainingRow]) -> TrainingDistribution:
    ordered = tuple(rows)
    if not ordered:
        raise ValueError("training projection has no rows")
    if tuple(row.sequence_id for row in ordered) != tuple(
        sorted(row.sequence_id for row in ordered)
    ) or len({row.sequence_id for row in ordered}) != len(ordered):
        raise ValueError("training projection rows must be strictly ordered by sequence_id")
    probabilities = tuple(row.sampling_weight for row in ordered)
    length_mass: dict[int, list[float]] = defaultdict(list)
    for row in ordered:
        length_mass[len(row.sequence)].append(row.sampling_weight)
    lengths = tuple(sorted(length_mass))
    raw = tuple(math.fsum(length_mass[length]) for length in lengths)
    total = math.fsum(raw)
    return TrainingDistribution(
        rows=ordered,
        probabilities=probabilities,
        length_prior=LengthPrior(
            lengths=lengths,
            probabilities=tuple(value / total for value in raw),
        ),
    )


def load_training_projection(
    path: str | Path,
    *,
    expected_sha256: str = ACCEPTED_TRAINING_PROJECTION_SHA256,
    expected_rows: int = 914,
) -> TrainingDistribution:
    """Load the byte-pinned view that is the trainer's complete data mount."""

    expected = _sha256(expected_sha256, label="expected training projection sha256")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive integer")
    payload = _read_regular_bytes(path)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"training projection SHA-256 expected {expected}, got {observed}")
    documents = _parse_rows(payload)
    if len(documents) != expected_rows:
        raise ValueError(f"training projection expected {expected_rows} rows, got {len(documents)}")
    rows: list[TrainingRow] = []
    for number, document in enumerate(documents, 1):
        label = f"training projection row {number}"
        if set(document) != _TRAINING_PROJECTION_FIELDS:
            raise ValueError(
                f"{label} schema mismatch: "
                f"missing={sorted(_TRAINING_PROJECTION_FIELDS - set(document))}, "
                f"extra={sorted(set(document) - _TRAINING_PROJECTION_FIELDS)}"
            )
        sequence_id = _sha256(document["sequence_id"], label=f"{label} sequence_id")
        sequence_raw = document["sequence"]
        if type(sequence_raw) is not str:
            raise ValueError(f"{label} sequence must be a string")
        sequence = canonicalize_sequence(sequence_raw)
        if (
            sequence != sequence_raw
            or not MIN_LENGTH <= len(sequence) <= MAX_LENGTH
            or canonical_sequence_id(sequence) != sequence_id
        ):
            raise ValueError(f"{label} has a noncanonical sequence identity or length")
        weight = document["sampling_weight"]
        if type(weight) is not float or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"{label} sampling_weight must be a positive finite JSON float")
        rows.append(
            TrainingRow(
                sequence_id=sequence_id,
                sequence=sequence,
                sampling_weight=weight,
            )
        )
    return _distribution_from_training_rows(rows)


def write_training_projection(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    expected_corpus_sha256: str = ACCEPTED_CORPUS_SHA256,
    expected_projection_sha256: str | None = None,
) -> str:
    """Publish a read-only train view without exposing fold 4 to the GPU process."""

    corpus = load_native_diffusion_corpus(
        corpus_path,
        expected_sha256=expected_corpus_sha256,
    )
    payload = training_projection_bytes(corpus)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_projection_sha256 is not None and digest != _sha256(
        expected_projection_sha256,
        label="expected training projection sha256",
    ):
        raise ValueError("derived training projection SHA-256 differs from the contract")
    destination = Path(os.path.abspath(os.fspath(output_path)))
    parent = destination.parent
    _reject_symlink_chain(parent)
    if not parent.is_dir():
        raise ValueError("training projection parent must be an existing directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite training projection: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=parent,
    )
    temporary = Path(temporary_name)
    linked = False
    linked_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        staged = temporary.stat(follow_symlinks=False)
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        linked_identity = (staged.st_dev, staged.st_ino)
        published = destination.stat(follow_symlinks=False)
        if (published.st_dev, published.st_ino) != linked_identity:
            raise RuntimeError("training projection publication changed inode identity")
        temporary.unlink()
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if linked and linked_identity is not None:
            try:
                current = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == linked_identity:
                    destination.unlink()
        raise
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-projection-sha256")
    args = parser.parse_args(argv)
    digest = write_training_projection(
        args.corpus,
        args.output,
        expected_projection_sha256=args.expected_projection_sha256,
    )
    print(
        json.dumps(
            {"rows": 914, "sha256": digest},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

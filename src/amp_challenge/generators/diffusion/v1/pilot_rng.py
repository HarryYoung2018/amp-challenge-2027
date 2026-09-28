"""Fit-identity-bound stateless RNG primitives for the native v1 pilot."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.v1.pilot_data import (
    ALPHABET,
    MAX_LENGTH,
    PilotTrainingProjection,
    PilotTrainingRow,
)

TRAINING_ROOT_SEED = 42
LEVELS = 64
COSINE_OFFSET = 0.008
CONTEXT_DROPOUT = 0.15
PAD_TOKEN_INDEX = 20
MASK_TOKEN_INDEX = 21

_SEED_DOMAIN = b"amp-challenge/native-categorical-diffusion/stateless-seed/v1\0"
_UINT64_RANGE = 1 << 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CorruptedBatch:
    """Fixed-count scheduled corruption plus loss-neutral context masks."""

    tokens: NDArray[np.int64]
    scheduled_mask: NDArray[np.bool_]
    context_dropout_mask: NDArray[np.bool_]
    mask_counts: NDArray[np.int64]
    corruption_seeds: tuple[int, ...]
    context_dropout_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        arrays = (
            (self.tokens, np.dtype("<i8"), "tokens"),
            (self.scheduled_mask, np.dtype("|b1"), "scheduled_mask"),
            (self.context_dropout_mask, np.dtype("|b1"), "context_dropout_mask"),
        )
        for value, dtype, name in arrays:
            if type(value) is not np.ndarray or value.dtype != dtype or value.ndim != 2:
                raise TypeError(f"{name} must be an exact two-dimensional {dtype.str} ndarray")
            if not value.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
        if not (self.tokens.shape == self.scheduled_mask.shape == self.context_dropout_mask.shape):
            raise ValueError("corruption arrays must have identical shapes")
        if (
            type(self.mask_counts) is not np.ndarray
            or self.mask_counts.dtype != np.dtype("<i8")
            or self.mask_counts.shape != (self.tokens.shape[0],)
        ):
            raise TypeError("mask_counts must be exact int64 with one value per row")
        if bool(np.any(self.scheduled_mask & self.context_dropout_mask)):
            raise ValueError("scheduled and context-dropout masks cannot overlap")
        if not np.array_equal(
            np.sum(self.scheduled_mask, axis=1, dtype=np.int64), self.mask_counts
        ):
            raise ValueError("scheduled-mask census differs from mask_counts")
        if not self.mask_counts.flags.c_contiguous:
            raise ValueError("mask_counts must be C-contiguous")
        for label, seeds in (
            ("corruption_seeds", self.corruption_seeds),
            ("context_dropout_seeds", self.context_dropout_seeds),
        ):
            if (
                type(seeds) is not tuple
                or len(seeds) != self.tokens.shape[0]
                or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)
            ):
                raise ValueError(f"{label} must contain one uint64 per row")


def namespaced_seed(root_seed: int, namespace: str, *parts: str | int) -> int:
    """Derive the contract's first-eight-byte big-endian SHA-256 uint64."""

    if type(root_seed) is not int or not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an exact unsigned 64-bit integer")
    if type(namespace) is not str or not namespace:
        raise ValueError("namespace must be a non-empty exact string")
    payload = bytearray(_SEED_DOMAIN)

    def append(tag: bytes, value: bytes) -> None:
        payload.extend(tag)
        payload.extend(len(value).to_bytes(8, "big"))
        payload.extend(value)

    append(b"r", root_seed.to_bytes(8, "big"))
    append(b"n", namespace.encode("utf-8"))
    for part in parts:
        if type(part) is str:
            append(b"s", part.encode("utf-8"))
        elif type(part) is int:
            append(b"i", str(part).encode("ascii"))
        else:
            raise TypeError("seed parts must be exact strings or exact integers")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def uint64_uniform(value: int) -> float:
    """Map uint64 bits to the frozen half-open binary64 uniform."""

    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("value must be an exact unsigned 64-bit integer")
    return (value >> 11) / float(1 << 53)


def weighted_minibatch_indices(
    projection: PilotTrainingProjection,
    *,
    fit_identity_sha256: str,
    global_draw_start: int,
    draw_count: int,
    root_seed: int = TRAINING_ROOT_SEED,
) -> tuple[int, ...]:
    """Draw rows with replacement from the ascending-ID binary64 CDF."""

    if type(projection) is not PilotTrainingProjection:
        raise TypeError("projection must be a PilotTrainingProjection")
    fit_identity = _fit_identity(fit_identity_sha256)
    _nonnegative_integer(global_draw_start, label="global_draw_start")
    _nonnegative_integer(draw_count, label="draw_count")
    probabilities = tuple(row.sampling_weight for row in projection.rows)
    total = math.fsum(probabilities)
    if total.hex() != (1.0).hex():  # pragma: no cover - projection invariant
        raise RuntimeError("projection weights changed before minibatch sampling")
    normalized = np.asarray([value / total for value in probabilities], dtype="<f8")
    cumulative = np.cumsum(normalized, dtype=np.float64)
    cumulative[-1] = 1.0
    result: list[int] = []
    for ordinal in range(global_draw_start, global_draw_start + draw_count):
        seed_bits = namespaced_seed(root_seed, "minibatch", fit_identity, ordinal)
        uniform = uint64_uniform(seed_bits)
        index = int(np.searchsorted(cumulative, uniform, side="right"))
        result.append(min(index, len(probabilities) - 1))
    return tuple(result)


def weighted_minibatch_rows(
    projection: PilotTrainingProjection,
    *,
    fit_identity_sha256: str,
    global_draw_start: int,
    draw_count: int,
    root_seed: int = TRAINING_ROOT_SEED,
) -> tuple[PilotTrainingRow, ...]:
    indices = weighted_minibatch_indices(
        projection,
        fit_identity_sha256=fit_identity_sha256,
        global_draw_start=global_draw_start,
        draw_count=draw_count,
        root_seed=root_seed,
    )
    return tuple(projection.rows[index] for index in indices)


def timestep_levels(
    *,
    fit_identity_sha256: str,
    global_draw_start: int,
    draw_count: int,
    levels: int = LEVELS,
    root_seed: int = TRAINING_ROOT_SEED,
) -> NDArray[np.int64]:
    """Return uniform integer timesteps via unbiased uint64 rejection."""

    fit_identity = _fit_identity(fit_identity_sha256)
    _nonnegative_integer(global_draw_start, label="global_draw_start")
    _nonnegative_integer(draw_count, label="draw_count")
    if type(levels) is not int or levels <= 0 or levels > np.iinfo(np.int64).max:
        raise ValueError("levels must be an exact positive int64-compatible integer")
    return np.asarray(
        [
            1
            + bounded_integer(
                root_seed,
                "timestep",
                levels,
                fit_identity,
                ordinal,
            )
            for ordinal in range(global_draw_start, global_draw_start + draw_count)
        ],
        dtype="<i8",
    )


def bounded_integer(
    root_seed: int,
    namespace: str,
    upper_bound: int,
    *parts: str | int,
) -> int:
    """Draw in ``range(upper_bound)`` with the contract's retry suffix."""

    if type(upper_bound) is not int or not 1 <= upper_bound <= _UINT64_RANGE:
        raise ValueError("upper_bound must be an exact integer in 1..2**64")
    acceptance_limit = _UINT64_RANGE - (_UINT64_RANGE % upper_bound)
    retry = 0
    while True:
        value = namespaced_seed(root_seed, namespace, *parts, retry)
        if value < acceptance_limit:
            return value % upper_bound
        retry += 1


def cosine_mask_counts(
    lengths: NDArray[np.int64],
    levels: NDArray[np.int64],
    *,
    total_levels: int = LEVELS,
    offset: float = COSINE_OFFSET,
) -> NDArray[np.int64]:
    """Compute fixed positive-level counts from the parent cosine schedule."""

    _int64_vector(lengths, label="lengths")
    _int64_vector(levels, label="levels")
    if lengths.shape != levels.shape or lengths.size == 0:
        raise ValueError("lengths and levels must be aligned non-empty vectors")
    if bool(np.any((lengths < 1) | (lengths > MAX_LENGTH))):
        raise ValueError("lengths must lie in 1..50")
    if type(total_levels) is not int or total_levels <= 0:
        raise ValueError("total_levels must be a positive exact integer")
    if bool(np.any((levels < 1) | (levels > total_levels))):
        raise ValueError("levels must lie in 1..total_levels")
    if type(offset) is not float or not math.isfinite(offset) or not 0.0 <= offset < 1.0:
        raise ValueError("offset must be a finite exact float in [0,1)")
    timestep = levels.astype(np.float64) / float(total_levels)
    angle_zero = offset / (1.0 + offset) * np.pi / 2.0
    angle = (timestep + offset) / (1.0 + offset) * np.pi / 2.0
    alpha_bar = np.square(np.cos(angle)) / np.square(np.cos(angle_zero))
    probability = np.clip(1.0 - alpha_bar, 0.0, 1.0)
    counts = np.ceil(probability * lengths).astype("<i8")
    return np.minimum(lengths, np.maximum(counts, 1)).astype("<i8", copy=False)


def corrupt_training_batch(
    clean_tokens: NDArray[np.int64],
    attention_mask: NDArray[np.bool_],
    levels: NDArray[np.int64],
    *,
    global_draw_ordinals: Sequence[int],
    sequence_ids: Sequence[str],
    fit_identity_sha256: str,
    root_seed: int = TRAINING_ROOT_SEED,
    context_dropout_probability: float = CONTEXT_DROPOUT,
) -> CorruptedBatch:
    """Apply scheduled masks then loss-neutral visible-context dropout per row."""

    _int64_matrix(clean_tokens, label="clean_tokens")
    _bool_matrix(attention_mask, label="attention_mask")
    _int64_vector(levels, label="levels")
    if clean_tokens.shape != attention_mask.shape:
        raise ValueError("clean_tokens and attention_mask must have identical shapes")
    batch, width = clean_tokens.shape
    if batch <= 0 or width <= 0 or width > MAX_LENGTH or levels.shape != (batch,):
        raise ValueError("corruption inputs have invalid batch dimensions")
    ordinals = tuple(global_draw_ordinals)
    identifiers = tuple(sequence_ids)
    if len(ordinals) != batch or any(type(value) is not int or value < 0 for value in ordinals):
        raise ValueError("global_draw_ordinals must contain one non-negative exact integer per row")
    if len(identifiers) != batch:
        raise ValueError("sequence_ids must contain one value per row")
    for identifier in identifiers:
        _fit_identity(identifier, label="sequence_id")
    fit_identity = _fit_identity(fit_identity_sha256)
    if (
        type(context_dropout_probability) is not float
        or not math.isfinite(context_dropout_probability)
        or not 0.0 <= context_dropout_probability < 1.0
    ):
        raise ValueError("context_dropout_probability must be a finite exact float in [0,1)")
    lengths = np.sum(attention_mask, axis=1, dtype=np.int64).astype("<i8", copy=False)
    expected_prefix = np.arange(width)[None, :] < lengths[:, None]
    if not np.array_equal(attention_mask, expected_prefix):
        raise ValueError("attention_mask must be a contiguous non-empty prefix")
    if bool(np.any(lengths < 8)):
        raise ValueError("each corruption row must contain 8..50 valid tokens")
    if bool(np.any(attention_mask & ((clean_tokens < 0) | (clean_tokens >= 20)))):
        raise ValueError("valid clean positions must contain residue tokens")
    if bool(np.any(~attention_mask & (clean_tokens != PAD_TOKEN_INDEX))):
        raise ValueError("clean padding positions must contain PAD")
    for row, identifier in enumerate(identifiers):
        decoded = "".join(ALPHABET[int(token)] for token in clean_tokens[row, : lengths[row]])
        if hashlib.sha256(decoded.encode("ascii")).hexdigest() != identifier:
            raise ValueError("sequence_id does not match the clean token row")
    counts = cosine_mask_counts(lengths, levels)
    corrupted = clean_tokens.copy()
    scheduled = np.zeros_like(attention_mask)
    context = np.zeros_like(attention_mask)
    corruption_seeds: list[int] = []
    context_seeds: list[int] = []
    for row in range(batch):
        length = int(lengths[row])
        level = int(levels[row])
        ordinal = ordinals[row]
        sequence_id = identifiers[row]
        corruption_seed = namespaced_seed(
            root_seed,
            "corruption",
            fit_identity,
            ordinal,
            sequence_id,
            level,
        )
        corruption_rng = np.random.Generator(np.random.PCG64(corruption_seed))
        valid_positions = np.arange(length, dtype=np.int64)
        chosen = corruption_rng.choice(
            valid_positions,
            size=int(counts[row]),
            replace=False,
        )
        scheduled[row, chosen] = True
        corrupted[row, chosen] = MASK_TOKEN_INDEX
        context_seed = namespaced_seed(
            root_seed,
            "context_dropout",
            fit_identity,
            ordinal,
            sequence_id,
            level,
        )
        context_rng = np.random.Generator(np.random.PCG64(context_seed))
        uniforms = context_rng.random(length, dtype=np.float64)
        dropped = (~scheduled[row, :length]) & (uniforms < context_dropout_probability)
        context[row, :length] = dropped
        corrupted[row, np.flatnonzero(dropped)] = MASK_TOKEN_INDEX
        corruption_seeds.append(corruption_seed)
        context_seeds.append(context_seed)
    return CorruptedBatch(
        tokens=np.asarray(corrupted, dtype="<i8"),
        scheduled_mask=np.asarray(scheduled, dtype="|b1"),
        context_dropout_mask=np.asarray(context, dtype="|b1"),
        mask_counts=counts,
        corruption_seeds=tuple(corruption_seeds),
        context_dropout_seeds=tuple(context_seeds),
    )


def initialization_seed(
    fit_identity_sha256: str,
    *,
    root_seed: int = TRAINING_ROOT_SEED,
) -> int:
    return namespaced_seed(root_seed, "initialization", _fit_identity(fit_identity_sha256), "model")


def model_dropout_seed(
    fit_identity_sha256: str,
    optimizer_step: int,
    *,
    root_seed: int = TRAINING_ROOT_SEED,
) -> int:
    if type(optimizer_step) is not int or optimizer_step <= 0:
        raise ValueError("optimizer_step must be a positive exact integer")
    return namespaced_seed(
        root_seed,
        "model_dropout",
        _fit_identity(fit_identity_sha256),
        optimizer_step,
    )


def reseed_torch_for_initialization(
    fit_identity_sha256: str,
    *,
    root_seed: int = TRAINING_ROOT_SEED,
) -> int:
    seed = initialization_seed(fit_identity_sha256, root_seed=root_seed)
    _reseed_torch(seed)
    return seed


def reseed_torch_for_model_dropout(
    fit_identity_sha256: str,
    optimizer_step: int,
    *,
    root_seed: int = TRAINING_ROOT_SEED,
) -> int:
    seed = model_dropout_seed(
        fit_identity_sha256,
        optimizer_step,
        root_seed=root_seed,
    )
    _reseed_torch(seed)
    return seed


def _reseed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_identity(value: object, *, label: str = "fit_identity_sha256") -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative exact integer")
    return value


def _int64_vector(value: object, *, label: str) -> None:
    if type(value) is not np.ndarray or value.dtype != np.dtype("<i8") or value.ndim != 1:
        raise TypeError(f"{label} must be an exact one-dimensional int64 ndarray")


def _int64_matrix(value: object, *, label: str) -> None:
    if type(value) is not np.ndarray or value.dtype != np.dtype("<i8") or value.ndim != 2:
        raise TypeError(f"{label} must be an exact two-dimensional int64 ndarray")


def _bool_matrix(value: object, *, label: str) -> None:
    if type(value) is not np.ndarray or value.dtype != np.dtype("|b1") or value.ndim != 2:
        raise TypeError(f"{label} must be an exact two-dimensional bool ndarray")


_RNG_CALLABLE_SNAPSHOT = tuple(
    (
        name,
        value,
        value.__code__,
        None
        if value.__defaults__ is None
        else tuple((type(default), default) for default in value.__defaults__),
        None
        if value.__kwdefaults__ is None
        else tuple((key, type(default), default) for key, default in value.__kwdefaults__.items()),
    )
    for name, value in (
        ("namespaced_seed", namespaced_seed),
        ("uint64_uniform", uint64_uniform),
        ("weighted_minibatch_indices", weighted_minibatch_indices),
        ("weighted_minibatch_rows", weighted_minibatch_rows),
        ("timestep_levels", timestep_levels),
        ("bounded_integer", bounded_integer),
        ("cosine_mask_counts", cosine_mask_counts),
        ("corrupt_training_batch", corrupt_training_batch),
        ("initialization_seed", initialization_seed),
        ("model_dropout_seed", model_dropout_seed),
        ("reseed_torch_for_initialization", reseed_torch_for_initialization),
        ("reseed_torch_for_model_dropout", reseed_torch_for_model_dropout),
        ("_reseed_torch", _reseed_torch),
        ("_fit_identity", _fit_identity),
        ("_nonnegative_integer", _nonnegative_integer),
        ("_int64_vector", _int64_vector),
        ("_int64_matrix", _int64_matrix),
        ("_bool_matrix", _bool_matrix),
    )
)
_RNG_CONSTANT_SNAPSHOT = tuple(
    (name, type(value), value)
    for name, value in (
        ("TRAINING_ROOT_SEED", TRAINING_ROOT_SEED),
        ("LEVELS", LEVELS),
        ("COSINE_OFFSET", COSINE_OFFSET),
        ("CONTEXT_DROPOUT", CONTEXT_DROPOUT),
        ("PAD_TOKEN_INDEX", PAD_TOKEN_INDEX),
        ("MASK_TOKEN_INDEX", MASK_TOKEN_INDEX),
        ("ALPHABET", ALPHABET),
        ("MAX_LENGTH", MAX_LENGTH),
        ("_SEED_DOMAIN", _SEED_DOMAIN),
        ("_UINT64_RANGE", _UINT64_RANGE),
        ("_SHA256_RE", _SHA256_RE),
    )
)
_CORRUPTED_BATCH_CLASS = CorruptedBatch
_CORRUPTED_BATCH_METHOD_SNAPSHOT = tuple(
    (name, value, value.__code__)
    for name, value in (
        ("__init__", CorruptedBatch.__init__),
        ("__post_init__", CorruptedBatch.__post_init__),
    )
)


def assert_pilot_rng_execution_surface() -> None:
    """Reject replacement or in-place edits anywhere in the frozen RNG graph."""

    namespace = globals()
    for (
        name,
        expected,
        expected_code,
        expected_defaults,
        expected_kwdefaults,
    ) in _RNG_CALLABLE_SNAPSHOT:
        observed = namespace.get(name)
        observed_defaults = getattr(observed, "__defaults__", None)
        observed_kwdefaults = getattr(observed, "__kwdefaults__", None)
        default_snapshot = (
            None
            if observed_defaults is None
            else tuple((type(default), default) for default in observed_defaults)
        )
        kwdefault_snapshot = (
            None
            if observed_kwdefaults is None
            else tuple(
                (key, type(default), default) for key, default in observed_kwdefaults.items()
            )
        )
        if (
            observed is not expected
            or getattr(observed, "__code__", None) is not expected_code
            or default_snapshot != expected_defaults
            or kwdefault_snapshot != expected_kwdefaults
        ):
            raise RuntimeError(f"frozen pilot RNG callable was overridden: {name}")
    for name, expected_type, expected in _RNG_CONSTANT_SNAPSHOT:
        observed = namespace.get(name)
        if type(observed) is not expected_type or observed != expected:
            raise RuntimeError(f"frozen pilot RNG constant was overridden: {name}")
    if namespace.get("CorruptedBatch") is not _CORRUPTED_BATCH_CLASS:
        raise RuntimeError("frozen pilot RNG class was overridden: CorruptedBatch")
    for name, expected, expected_code in _CORRUPTED_BATCH_METHOD_SNAPSHOT:
        observed = getattr(_CORRUPTED_BATCH_CLASS, name, None)
        if observed is not expected or getattr(observed, "__code__", None) is not expected_code:
            raise RuntimeError(f"frozen pilot RNG callable was overridden: CorruptedBatch.{name}")

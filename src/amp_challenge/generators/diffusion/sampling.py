"""Deterministic CPU-side reverse sampler for native diffusion v0.

The neural-network boundary is a NumPy logit callback.  Random categorical
draws are keyed to stable candidate identities rather than a mutable batch RNG,
so changing batch size or input order cannot change a candidate's randomness.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from amp_challenge.sequences import canonical_sequence_id

from .categorical import CosineMaskSchedule, PeptideVocabulary
from .contract import CONFIG_SHA256

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DIFFUSION_LEVELS = 64
MIN_LENGTH = 8
MAX_LENGTH = 50
RAW_PROPOSALS_PER_SEED = 2048
SAMPLING_BATCH_SEQUENCES = 256

_CHECKPOINT_RE = re.compile(r"[0-9a-f]{64}")
_DRAW_DOMAIN = b"amp-challenge/native-categorical-diffusion/categorical-draw/v1\0"

LogitProvider = Callable[
    [NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]],
    NDArray[np.floating],
]


@dataclass(frozen=True, slots=True)
class SampledCandidate:
    """One raw proposal; duplicates intentionally remain separate records."""

    ordinal: int
    seed: int
    sequence_id: str
    sequence: str
    length: int
    checkpoint_logical_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        _uint64(self.ordinal, label="ordinal")
        _uint64(self.seed, label="seed")
        if (
            not isinstance(self.checkpoint_logical_sha256, str)
            or _CHECKPOINT_RE.fullmatch(self.checkpoint_logical_sha256) is None
        ):
            raise ValueError("checkpoint_logical_sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.contract_sha256, str)
            or _CHECKPOINT_RE.fullmatch(self.contract_sha256) is None
        ):
            raise ValueError("contract_sha256 must be a lowercase SHA-256 digest")
        if type(self.length) is not int or not MIN_LENGTH <= self.length <= MAX_LENGTH:
            raise ValueError("candidate length must be an integer in 8..50")
        if type(self.sequence) is not str or len(self.sequence) != self.length:
            raise ValueError("candidate sequence must be a string matching its declared length")
        if set(self.sequence) - set(ALPHABET):
            raise ValueError("candidate sequence contains a noncanonical residue")
        if canonical_sequence_id(self.sequence) != self.sequence_id:
            raise ValueError("candidate sequence_id does not match its sequence")

    def canonical_record(self) -> dict[str, object]:
        return {
            "checkpoint_logical_sha256": self.checkpoint_logical_sha256,
            "contract_sha256": self.contract_sha256,
            "length": self.length,
            "ordinal": self.ordinal,
            "schema_version": 1,
            "seed": self.seed,
            "sequence": self.sequence,
            "sequence_id": self.sequence_id,
        }


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Canonical raw output and the exact input length-plan identity."""

    candidates: tuple[SampledCandidate, ...]
    length_plan_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.candidates) is not tuple
            or not self.candidates
            or any(type(item) is not SampledCandidate for item in self.candidates)
        ):
            raise ValueError("sampling result must contain a tuple of sampled candidates")
        ordinals = tuple(item.ordinal for item in self.candidates)
        if ordinals != tuple(sorted(ordinals)) or len(ordinals) != len(set(ordinals)):
            raise ValueError("sampling result candidates must have unique sorted ordinals")
        if (
            len({item.seed for item in self.candidates}) != 1
            or len({item.checkpoint_logical_sha256 for item in self.candidates}) != 1
        ):
            raise ValueError("sampling result must use one seed and one logical checkpoint")
        if len({item.contract_sha256 for item in self.candidates}) != 1:
            raise ValueError("sampling result must use one diffusion contract")
        if (
            not isinstance(self.length_plan_sha256, str)
            or _CHECKPOINT_RE.fullmatch(self.length_plan_sha256) is None
        ):
            raise ValueError("length_plan_sha256 must be a lowercase SHA-256 digest")
        expected = _length_plan_hash(tuple((item.ordinal, item.length) for item in self.candidates))
        if self.length_plan_sha256 != expected:
            raise ValueError("length_plan_sha256 does not match the candidate records")

    def canonical_jsonl_bytes(self) -> bytes:
        return b"".join(_canonical_json_bytes(item.canonical_record()) for item in self.candidates)


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


def _uint64(value: object, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | np.integer)
        or not 0 <= int(value) < 2**64
    ):
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return int(value)


def _draw_seed(
    checkpoint_logical_sha256: str,
    seed: int,
    ordinal: int,
    level: int,
) -> int:
    """Derive the one PCG64DXSM stream assigned to a candidate and level."""

    digest = hashlib.sha256()
    digest.update(_DRAW_DOMAIN)
    for payload in (
        checkpoint_logical_sha256.encode("ascii"),
        seed.to_bytes(8, "big"),
        ordinal.to_bytes(8, "big"),
        level.to_bytes(2, "big"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big")


def _length_plan_hash(items: Sequence[tuple[int, int]]) -> str:
    payload = b"".join(
        _canonical_json_bytes({"length": length, "ordinal": ordinal}) for ordinal, length in items
    )
    return hashlib.sha256(payload).hexdigest()


def canonical_length_plan(
    lengths: Sequence[int],
    *,
    ordinals: Sequence[int] | None = None,
    require_locked_count: bool = True,
) -> tuple[tuple[tuple[int, int], ...], str]:
    """Validate and canonically order the shared ordinal-to-length plan."""

    if not isinstance(require_locked_count, bool):
        raise TypeError("require_locked_count must be boolean")
    raw_lengths = tuple(lengths)
    if not raw_lengths:
        raise ValueError("lengths must contain at least one proposal")
    if require_locked_count and len(raw_lengths) != RAW_PROPOSALS_PER_SEED:
        raise ValueError(f"locked v0 sampling requires exactly {RAW_PROPOSALS_PER_SEED} proposals")
    if any(
        isinstance(length, bool)
        or not isinstance(length, int | np.integer)
        or not MIN_LENGTH <= int(length) <= MAX_LENGTH
        for length in raw_lengths
    ):
        raise ValueError("every proposal length must be an integer in 8..50")
    normalized_lengths = tuple(int(length) for length in raw_lengths)

    if ordinals is None:
        normalized_ordinals = tuple(range(len(normalized_lengths)))
    else:
        raw_ordinals = tuple(ordinals)
        if len(raw_ordinals) != len(normalized_lengths) or any(
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int | np.integer)
            or not 0 <= int(ordinal) < 2**64
            for ordinal in raw_ordinals
        ):
            raise ValueError("ordinals must align and contain unsigned 64-bit integers")
        normalized_ordinals = tuple(int(ordinal) for ordinal in raw_ordinals)
    if len(set(normalized_ordinals)) != len(normalized_ordinals):
        raise ValueError("candidate ordinals must be unique")

    plan = tuple(sorted(zip(normalized_ordinals, normalized_lengths, strict=True)))
    return plan, _length_plan_hash(plan)


def _probabilities(logits: object, *, expected_shape: tuple[int, int, int]) -> NDArray[np.float64]:
    raw = np.asarray(logits)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("logit provider must return a real numeric array")
    if raw.shape != expected_shape:
        raise ValueError(f"logit provider returned shape {raw.shape}; expected {expected_shape}")
    values = raw.astype(np.float64, copy=True)
    if np.any(~np.isfinite(values)):
        raise ValueError("logit provider returned a non-finite value")
    values -= np.max(values, axis=-1, keepdims=True)
    np.exp(values, out=values)
    totals = np.sum(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("logit provider produced an invalid categorical distribution")
    values /= totals
    return values


def _categorical_index(probabilities: NDArray[np.float64], uniform: float) -> int:
    if probabilities.shape != (len(ALPHABET),):
        raise ValueError("residue probability vector must have exactly 20 entries")
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    return min(int(np.searchsorted(cumulative, uniform, side="right")), len(ALPHABET) - 1)


def sample_unconditional_v0(
    logit_provider: LogitProvider,
    lengths: Sequence[int],
    *,
    checkpoint_logical_sha256: str,
    seed: int,
    contract_sha256: str = CONFIG_SHA256,
    ordinals: Sequence[int] | None = None,
    batch_size: int = SAMPLING_BATCH_SEQUENCES,
    require_locked_count: bool = True,
) -> SamplingResult:
    """Sample raw v0 proposals with fixed-count monotone confidence commits.

    ``logit_provider`` receives corrupted tokens, prefix masks, current levels,
    and lengths, and must return finite ``[batch, width, 20]`` logits.  The
    provider is never given randomness and the sampler performs no rejection,
    retry, filtering, deduplication, conditioning, or oracle call.
    """

    if not callable(logit_provider):
        raise TypeError("logit_provider must be callable")
    if (
        not isinstance(checkpoint_logical_sha256, str)
        or _CHECKPOINT_RE.fullmatch(checkpoint_logical_sha256) is None
    ):
        raise ValueError("checkpoint_logical_sha256 must be a lowercase SHA-256 digest")
    root_seed = _uint64(seed, label="seed")
    if not isinstance(contract_sha256, str) or _CHECKPOINT_RE.fullmatch(contract_sha256) is None:
        raise ValueError("contract_sha256 must be a lowercase SHA-256 digest")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    plan, length_plan_sha256 = canonical_length_plan(
        lengths,
        ordinals=ordinals,
        require_locked_count=require_locked_count,
    )
    if require_locked_count and batch_size != SAMPLING_BATCH_SEQUENCES:
        raise ValueError("locked v0 sampling requires batches of 256 proposals")
    if require_locked_count and contract_sha256 != CONFIG_SHA256:
        raise ValueError("locked v0 sampling requires the accepted diffusion contract")
    vocabulary = PeptideVocabulary(ALPHABET)
    schedule = CosineMaskSchedule(offset=0.008)
    candidates: list[SampledCandidate] = []

    for start in range(0, len(plan), batch_size):
        batch_plan = plan[start : start + batch_size]
        batch_ordinals = np.asarray([item[0] for item in batch_plan], dtype=np.uint64)
        batch_lengths = np.asarray([item[1] for item in batch_plan], dtype=np.int64)
        # Width is part of the production execution contract.  It must not vary
        # with the length composition of a batch, including the final short one.
        width = MAX_LENGTH
        attention_mask = np.arange(width)[None, :] < batch_lengths[:, None]
        tokens = np.full((len(batch_plan), width), vocabulary.pad_index, dtype=np.int64)
        tokens[attention_mask] = vocabulary.mask_index

        for level in range(DIFFUSION_LEVELS, 0, -1):
            current_counts = schedule.mask_counts(
                batch_lengths,
                level,
                total_levels=DIFFUSION_LEVELS,
            )
            target_counts = schedule.mask_counts(
                batch_lengths,
                level - 1,
                total_levels=DIFFUSION_LEVELS,
            )
            observed_counts = np.sum(tokens == vocabulary.mask_index, axis=1, dtype=np.int64)
            if not np.array_equal(observed_counts, current_counts):
                raise RuntimeError("reverse sampler mask count drifted from the cosine schedule")
            commit_counts = current_counts - target_counts
            if np.any(commit_counts < 0):
                raise RuntimeError("reverse sampler attempted to increase the mask count")
            if not np.any(commit_counts):
                continue

            provider_tokens = tokens.copy()
            provider_mask = attention_mask.copy()
            provider_levels = np.full(len(batch_plan), level, dtype=np.int64)
            provider_lengths = batch_lengths.copy()
            for value in (provider_tokens, provider_mask, provider_levels, provider_lengths):
                value.flags.writeable = False
            logits = logit_provider(
                provider_tokens,
                provider_mask,
                provider_levels,
                provider_lengths,
            )
            probabilities = _probabilities(
                logits,
                expected_shape=(len(batch_plan), width, len(ALPHABET)),
            )

            for row_index, commit_count_raw in enumerate(commit_counts):
                commit_count = int(commit_count_raw)
                if commit_count == 0:
                    continue
                masked_positions = np.flatnonzero(
                    attention_mask[row_index] & (tokens[row_index] == vocabulary.mask_index)
                )
                confidence = np.max(probabilities[row_index, masked_positions], axis=1)
                # lexsort uses the final key as primary: confidence descending,
                # then position ascending for an exact deterministic tie break.
                ranked = np.lexsort((masked_positions, -confidence))
                selected = masked_positions[ranked[:commit_count]]
                ordinal = int(batch_ordinals[row_index])
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        _draw_seed(
                            checkpoint_logical_sha256,
                            root_seed,
                            ordinal,
                            level,
                        )
                    )
                )
                uniforms = rng.random(commit_count)
                for position, uniform in zip(selected, uniforms, strict=True):
                    tokens[row_index, position] = _categorical_index(
                        probabilities[row_index, position],
                        float(uniform),
                    )

            remaining = np.sum(tokens == vocabulary.mask_index, axis=1, dtype=np.int64)
            if not np.array_equal(remaining, target_counts):
                raise RuntimeError("reverse sampler did not commit the scheduled residue count")

        if np.any(tokens[attention_mask] >= len(ALPHABET)):
            raise RuntimeError("reverse sampler left or emitted a special token")
        if np.any(tokens[~attention_mask] != vocabulary.pad_index):
            raise RuntimeError("reverse sampler modified padded positions")
        sequences = vocabulary.decode(tokens)
        for (ordinal, expected_length), sequence in zip(batch_plan, sequences, strict=True):
            if len(sequence) != expected_length:
                raise RuntimeError("decoded proposal length differs from its frozen length plan")
            candidates.append(
                SampledCandidate(
                    ordinal=ordinal,
                    seed=root_seed,
                    sequence_id=canonical_sequence_id(sequence),
                    sequence=sequence,
                    length=expected_length,
                    checkpoint_logical_sha256=checkpoint_logical_sha256,
                    contract_sha256=contract_sha256,
                )
            )

    ordered = tuple(sorted(candidates, key=lambda item: item.ordinal))
    if len(ordered) != len(plan):
        raise RuntimeError("reverse sampler did not preserve the raw proposal census")
    return SamplingResult(candidates=ordered, length_plan_sha256=length_plan_sha256)

"""Small, framework-neutral primitives for categorical peptide diffusion.

These utilities define the corruption contract that a PyTorch denoiser can
later implement.  Keeping tokenization/noising in NumPy makes the behavior easy
to test independently of GPU libraries and prevents the official generation
entry point from inheriting a heavyweight training environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class EncodedPeptides:
    tokens: NDArray[np.int64]
    attention_mask: NDArray[np.bool_]


def _integer_matrix(value: object, *, label: str) -> NDArray[np.int64]:
    values = np.asarray(value)
    if values.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional batch")
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{label} must have an integer dtype")
    return values.astype(np.int64, copy=False)


def _boolean_matrix(value: object, *, label: str) -> NDArray[np.bool_]:
    values = np.asarray(value)
    if values.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional batch")
    if values.dtype.kind != "b":
        raise TypeError(f"{label} must have a boolean dtype")
    return values.astype(np.bool_, copy=False)


def _validated_clean_batch(
    tokens: object,
    attention_mask: object,
    *,
    vocabulary: PeptideVocabulary,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    clean = _integer_matrix(tokens, label="tokens")
    valid = _boolean_matrix(attention_mask, label="attention_mask")
    if valid.shape != clean.shape:
        raise ValueError("tokens and attention_mask must have matching 2D shapes")
    if clean.shape[0] == 0:
        raise ValueError("tokens must contain at least one peptide")
    lengths = np.sum(valid, axis=1, dtype=np.int64)
    if np.any(lengths == 0):
        raise ValueError("every peptide needs at least one valid token")
    prefix_mask = np.arange(clean.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(valid, prefix_mask):
        raise ValueError("attention_mask must be a contiguous True prefix in every row")
    if np.any(valid & ((clean < 0) | (clean >= len(vocabulary.alphabet)))):
        raise ValueError("valid positions must contain clean amino-acid tokens")
    if np.any(~valid & (clean != vocabulary.pad_index)):
        raise ValueError("positions outside attention_mask must be PAD")
    return clean, valid


def _integer_vector(
    value: object,
    *,
    label: str,
    size: int,
) -> NDArray[np.int64]:
    values = np.asarray(value)
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{label} must have an integer dtype")
    if values.ndim == 0:
        values = np.repeat(values, size)
    if values.shape != (size,):
        raise ValueError(f"{label} must be scalar or have one value per peptide")
    return values.astype(np.int64, copy=False)


class PeptideVocabulary:
    """Competition-safe amino-acid vocabulary with PAD and absorbing MASK."""

    def __init__(self, alphabet: str = DEFAULT_ALPHABET) -> None:
        if not isinstance(alphabet, str) or not alphabet:
            raise TypeError("alphabet must be a non-empty string")
        if len(alphabet) != len(set(alphabet)):
            raise ValueError("alphabet must contain unique symbols")
        self.alphabet = alphabet
        self.pad_index = len(alphabet)
        self.mask_index = len(alphabet) + 1
        self.size = len(alphabet) + 2
        self._encode = {symbol: index for index, symbol in enumerate(alphabet)}

    def encode(self, sequences: Sequence[str], *, max_length: int | None = None) -> EncodedPeptides:
        if isinstance(sequences, str):
            raise TypeError("sequences must be a sequence of peptide strings")
        source = tuple(sequences)
        if not source:
            raise ValueError("at least one sequence is required")
        if any(not isinstance(sequence, str) for sequence in source):
            raise TypeError("every sequence must be a string")
        normalized = tuple(sequence.strip().upper() for sequence in source)
        inferred_max = max(map(len, normalized))
        if max_length is not None and (
            isinstance(max_length, bool) or not isinstance(max_length, int | np.integer)
        ):
            raise TypeError("max_length must be an integer")
        width = inferred_max if max_length is None else int(max_length)
        if width <= 0 or inferred_max > width:
            raise ValueError("max_length must fit every non-empty sequence")
        tokens = np.full((len(normalized), width), self.pad_index, dtype=np.int64)
        attention_mask = np.zeros((len(normalized), width), dtype=bool)
        for row, sequence in enumerate(normalized):
            if not sequence:
                raise ValueError("sequences cannot be empty")
            invalid = set(sequence) - set(self.alphabet)
            if invalid:
                raise ValueError(f"invalid amino-acid symbols: {sorted(invalid)}")
            encoded = [self._encode[symbol] for symbol in sequence]
            tokens[row, : len(encoded)] = encoded
            attention_mask[row, : len(encoded)] = True
        return EncodedPeptides(tokens=tokens, attention_mask=attention_mask)

    def decode(
        self,
        tokens: NDArray[np.integer],
        *,
        allow_mask: bool = False,
    ) -> list[str]:
        if not isinstance(allow_mask, bool):
            raise TypeError("allow_mask must be boolean")
        values = _integer_matrix(tokens, label="tokens")
        decoded: list[str] = []
        for row in values:
            symbols: list[str] = []
            seen_padding = False
            for raw_token in row:
                token = int(raw_token)
                if token == self.pad_index:
                    seen_padding = True
                    continue
                if seen_padding:
                    raise ValueError("non-padding token appears after PAD")
                if token == self.mask_index:
                    if not allow_mask:
                        raise ValueError("cannot decode a masked peptide")
                    symbols.append("?")
                elif 0 <= token < len(self.alphabet):
                    symbols.append(self.alphabet[token])
                else:
                    raise ValueError(f"token index out of vocabulary: {token}")
            decoded.append("".join(symbols))
        return decoded


@dataclass(frozen=True)
class CosineMaskSchedule:
    """Smooth absorbing-mask probability from clean ``t=0`` to masked ``t=1``."""

    offset: float = 0.008

    def __post_init__(self) -> None:
        if (
            isinstance(self.offset, bool)
            or not isinstance(self.offset, Real)
            or not np.isfinite(self.offset)
            or not 0 <= self.offset < 1
        ):
            raise ValueError("offset must be in [0, 1)")

    def probability(self, timestep: float | NDArray[np.floating]) -> NDArray[np.float64]:
        raw_values = np.asarray(timestep)
        if raw_values.dtype.kind not in {"i", "u", "f"}:
            raise TypeError("diffusion timesteps must have a real numeric dtype")
        values = raw_values.astype(np.float64, copy=False)
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("diffusion timesteps must lie in [0, 1]")
        angle_0 = self.offset / (1.0 + self.offset) * np.pi / 2.0
        angle_t = (values + self.offset) / (1.0 + self.offset) * np.pi / 2.0
        alpha_bar = np.square(np.cos(angle_t)) / np.square(np.cos(angle_0))
        return np.clip(1.0 - alpha_bar, 0.0, 1.0)

    def mask_counts(
        self,
        lengths: int | NDArray[np.integer],
        levels: int | NDArray[np.integer],
        *,
        total_levels: int,
    ) -> NDArray[np.int64]:
        """Return the fixed number of masked residues at discrete cosine levels.

        Level zero is clean.  Every positive level masks at least one residue,
        and the terminal level masks the complete peptide.  This is the v0
        MaskGIT-style corruption contract; it is deliberately separate from
        :meth:`AbsorbingDiffusion.corrupt`, whose Bernoulli behavior is retained
        for compatibility with the original primitives.
        """

        if isinstance(total_levels, bool) or not isinstance(total_levels, int | np.integer):
            raise TypeError("total_levels must be an integer")
        total = int(total_levels)
        if total <= 0:
            raise ValueError("total_levels must be positive")

        raw_lengths = np.asarray(lengths)
        if raw_lengths.dtype.kind not in {"i", "u"}:
            raise TypeError("lengths must have an integer dtype")
        if raw_lengths.ndim == 0:
            raw_lengths = raw_lengths.reshape(1)
        if raw_lengths.ndim != 1:
            raise ValueError("lengths must be scalar or one-dimensional")
        length_values = raw_lengths.astype(np.int64, copy=False)
        if np.any(length_values <= 0):
            raise ValueError("lengths must be positive")

        level_values = _integer_vector(levels, label="levels", size=len(length_values))
        if np.any((level_values < 0) | (level_values > total)):
            raise ValueError(f"levels must lie in 0..{total}")
        probabilities = self.probability(level_values / float(total))
        counts = np.ceil(probabilities * length_values).astype(np.int64)
        positive_counts = np.minimum(length_values, np.maximum(counts, 1))
        return np.where(level_values == 0, 0, positive_counts).astype(np.int64, copy=False)


class AbsorbingDiffusion:
    """Forward corruption that replaces amino-acid tokens with one MASK state."""

    def __init__(
        self,
        vocabulary: PeptideVocabulary | None = None,
        schedule: CosineMaskSchedule | None = None,
    ) -> None:
        self.vocabulary = vocabulary or PeptideVocabulary()
        self.schedule = schedule or CosineMaskSchedule()

    def corrupt(
        self,
        tokens: NDArray[np.integer],
        attention_mask: NDArray[np.bool_],
        timestep: float | NDArray[np.floating],
        *,
        rng: np.random.Generator,
        ensure_masked: bool = True,
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        """Return legacy Bernoulli corruption and selected loss positions.

        ``ensure_masked=True`` preserves the original API exactly: an otherwise
        empty positive-time row receives one uniformly chosen mask.  New v0
        training must use :meth:`corrupt_fixed_count`, which has an explicit
        fixed-count distribution and stateless per-row seeds.
        """

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if not isinstance(ensure_masked, bool):
            raise TypeError("ensure_masked must be boolean")
        clean, valid = _validated_clean_batch(
            tokens,
            attention_mask,
            vocabulary=self.vocabulary,
        )

        time = np.asarray(timestep)
        if time.dtype.kind not in {"i", "u", "f"}:
            raise TypeError("diffusion timesteps must have a real numeric dtype")
        if time.ndim == 0:
            time = np.repeat(time, clean.shape[0])
        if time.shape != (clean.shape[0],):
            raise ValueError("timestep must be scalar or have one value per peptide")
        probability = self.schedule.probability(time)
        selected = (rng.random(clean.shape) < probability[:, None]) & valid

        if ensure_masked:
            for row in range(clean.shape[0]):
                if probability[row] > 0 and not np.any(selected[row]):
                    valid_positions = np.flatnonzero(valid[row])
                    chosen = int(rng.choice(valid_positions))
                    selected[row, chosen] = True

        corrupted = clean.copy()
        corrupted[selected] = self.vocabulary.mask_index
        return corrupted, selected

    def corrupt_fixed_count(
        self,
        tokens: NDArray[np.integer],
        attention_mask: NDArray[np.bool_],
        levels: int | NDArray[np.integer],
        *,
        total_levels: int,
        row_seeds: Sequence[int],
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        """Mask the exact scheduled count using independent per-row seeds.

        A seed belongs to a stable row occurrence rather than to a batch.  If a
        caller derives it from a sequence identity plus global draw ordinal,
        corruption is unchanged by row order or batch boundaries.
        """

        clean, valid = _validated_clean_batch(
            tokens,
            attention_mask,
            vocabulary=self.vocabulary,
        )
        seeds = tuple(row_seeds)
        if len(seeds) != clean.shape[0]:
            raise ValueError("row_seeds must have one value per peptide")
        for seed in seeds:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int | np.integer)
                or not 0 <= int(seed) < 2**64
            ):
                raise ValueError("row_seeds must contain unsigned 64-bit integers")

        lengths = np.sum(valid, axis=1, dtype=np.int64)
        counts = self.schedule.mask_counts(lengths, levels, total_levels=total_levels)
        selected = np.zeros_like(valid)
        for row, (count, seed) in enumerate(zip(counts, seeds, strict=True)):
            if count == 0:
                continue
            positions = np.flatnonzero(valid[row])
            rng = np.random.Generator(np.random.PCG64(int(seed)))
            chosen = rng.choice(
                positions,
                size=int(count),
                replace=False,
            )
            selected[row, chosen] = True

        corrupted = clean.copy()
        corrupted[selected] = self.vocabulary.mask_index
        return corrupted, selected


def classifier_free_guidance(
    unconditional_logits: NDArray[np.floating],
    conditional_logits: NDArray[np.floating],
    guidance_scale: float,
) -> NDArray[np.float64]:
    """Blend unconditional and property-conditioned categorical logits."""

    raw_unconditional = np.asarray(unconditional_logits)
    raw_conditional = np.asarray(conditional_logits)
    if raw_unconditional.dtype.kind not in {"i", "u", "f"} or raw_conditional.dtype.kind not in {
        "i",
        "u",
        "f",
    }:
        raise TypeError("guidance logits must have real numeric dtypes")
    unconditional = raw_unconditional.astype(np.float64, copy=False)
    conditional = raw_conditional.astype(np.float64, copy=False)
    if unconditional.shape != conditional.shape:
        raise ValueError("conditional and unconditional logits must have identical shapes")
    if np.any(~np.isfinite(unconditional)) or np.any(~np.isfinite(conditional)):
        raise ValueError("guidance logits must be finite")
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, Real)
        or not np.isfinite(guidance_scale)
        or guidance_scale < 0
    ):
        raise ValueError("guidance_scale must be finite and non-negative")
    return unconditional + guidance_scale * (conditional - unconditional)

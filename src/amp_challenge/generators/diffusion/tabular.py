"""Tiny property-conditioned denoiser adapter for local search smoke tests.

This is not the cluster model.  It is a factorized categorical policy whose
closed-form gradients and KLs make the proposal's learning claims falsifiable
without PyTorch or training.  The same replay interface can later feed a neural
denoiser in an isolated runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp

from amp_challenge.generators.diffusion.categorical import PeptideVocabulary
from amp_challenge.generators.diffusion.replay import (
    CompleteTransitionKernel,
    EndpointReplayBatch,
    KLSummary,
    summarize_kl,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _log_softmax(logits: FloatArray) -> FloatArray:
    """Compute row-wise log probabilities after removing huge common offsets."""

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = shifted - logsumexp(shifted, axis=-1, keepdims=True)
    if np.any(~np.isfinite(values)):
        raise ValueError("logit gaps are not representable in float64")
    return np.asarray(values, dtype=np.float64)


def _categorical_logit_kl(
    numerator_logits: FloatArray,
    denominator_logits: FloatArray,
) -> FloatArray:
    """Return categorical KLs without mistaking underflow for missing support."""

    numerator_log_probability = _log_softmax(numerator_logits)
    denominator_log_probability = _log_softmax(denominator_logits)
    numerator_probability = np.exp(numerator_log_probability)
    terms = numerator_probability * (numerator_log_probability - denominator_log_probability)
    divergence = np.asarray(np.sum(terms, axis=-1), dtype=np.float64)
    roundoff = (
        64.0
        * np.finfo(np.float64).eps
        * np.maximum(
            1.0,
            np.sum(np.abs(terms), axis=-1),
        )
    )
    if np.any(divergence < -roundoff):
        raise FloatingPointError("categorical logit KL became negative beyond float64 roundoff")
    return np.maximum(divergence, 0.0)


def _immutable_float_array(value: object, *, name: str, ndim: int) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _centered_logits(value: object, *, name: str) -> FloatArray:
    """Canonicalize softmax-equivalent rows so small updates remain representable."""

    array = _immutable_float_array(value, name=name, ndim=3)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = array - np.max(array, axis=-1, keepdims=True)
    if np.any(~np.isfinite(centered)):
        raise ValueError(f"{name} contains logit gaps that are not representable")
    centered = np.asarray(centered, dtype=np.float64)
    centered.setflags(write=False)
    return centered


@dataclass(frozen=True, slots=True)
class DenoisingPreference:
    """A same-context winner/loser pair for bounded DPO-style supervision."""

    winner: str
    loser: str
    context_id: str
    corruption_mask: tuple[bool, ...]
    timestep: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.winner, str) or not isinstance(self.loser, str):
            raise ValueError("preference endpoints must be strings")
        winner = self.winner.strip().upper()
        loser = self.loser.strip().upper()
        if not winner or not loser or winner == loser:
            raise ValueError("preference endpoints must be distinct non-empty sequences")
        if len(winner) != len(loser):
            raise ValueError("the first preference slice requires equal-length endpoints")
        mask_values = tuple(self.corruption_mask)
        if any(not isinstance(value, bool | np.bool_) for value in mask_values):
            raise ValueError("corruption_mask must contain booleans")
        mask = tuple(bool(value) for value in mask_values)
        if len(mask) != len(winner) or not any(mask):
            raise ValueError("corruption_mask must select at least one aligned position")
        if not self.context_id:
            raise ValueError("context_id cannot be empty")
        if not np.isfinite(self.timestep) or not 0 < self.timestep <= 1:
            raise ValueError("preference timestep must lie in (0, 1]")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("preference weight must be finite and positive")
        object.__setattr__(self, "winner", winner)
        object.__setattr__(self, "loser", loser)
        object.__setattr__(self, "corruption_mask", mask)


@dataclass(frozen=True, slots=True)
class TabularPolicyUpdate:
    """Accepted trust-region projection and its auditable diagnostics."""

    policy: TabularConditionalDenoiser
    step_scale: float
    local_kl_old_new: float
    local_transition_kl_summary: KLSummary
    reference_path_kl_new_reference: float
    reference_transition_kl_summary: KLSummary
    preference_margin_before: tuple[float, ...]
    preference_margin_after: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TabularConditionalDenoiser:
    """Factorized categorical logits indexed by context and residue position."""

    vocabulary: PeptideVocabulary
    contexts: tuple[str, ...]
    logits: FloatArray
    reference_logits: FloatArray
    version: int = 0

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        if not contexts or any(not isinstance(context, str) or not context for context in contexts):
            raise ValueError("contexts must be non-empty strings")
        if len(set(contexts)) != len(contexts):
            raise ValueError("contexts must be unique")
        logits = _centered_logits(self.logits, name="logits")
        reference = _centered_logits(self.reference_logits, name="reference_logits")
        expected_prefix = (len(contexts), logits.shape[1])
        if logits.shape != reference.shape or logits.shape[:2] != expected_prefix:
            raise ValueError("logits and reference_logits must align with contexts and length")
        if logits.shape[2] != len(self.vocabulary.alphabet):
            raise ValueError("the logits alphabet axis must match the vocabulary")
        if isinstance(self.version, bool) or not isinstance(self.version, int | np.integer):
            raise ValueError("version must be an integer")
        if self.version < 0:
            raise ValueError("version must be non-negative")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "reference_logits", reference)

    @classmethod
    def uniform(
        cls,
        *,
        vocabulary: PeptideVocabulary,
        contexts: Sequence[str],
        length: int,
    ) -> TabularConditionalDenoiser:
        if length <= 0:
            raise ValueError("length must be positive")
        context_tuple = tuple(contexts)
        logits = np.zeros((len(context_tuple), length, len(vocabulary.alphabet)))
        return cls(vocabulary, context_tuple, logits, logits.copy())

    @property
    def length(self) -> int:
        return int(self.logits.shape[1])

    def probabilities(self, context_id: str) -> FloatArray:
        """Return per-position residue probabilities for one property context."""

        context = self._context_index(context_id)
        logits = self.logits[context]
        probabilities = np.exp(_log_softmax(logits))
        return np.asarray(probabilities, dtype=np.float64)

    def log_probability(
        self,
        sequence: str,
        *,
        context_id: str,
        corruption_mask: Sequence[bool] | BoolArray | None = None,
    ) -> float:
        """Return the exact factorized log probability on selected positions."""

        tokens = self._tokens(sequence)
        if corruption_mask is None:
            mask = np.ones(self.length, dtype=bool)
        else:
            mask = np.asarray(corruption_mask)
            if mask.dtype.kind != "b" or mask.shape != (self.length,) or not np.any(mask):
                raise ValueError("corruption_mask must be a non-empty boolean position vector")
        context = self._context_index(context_id)
        context_logits = self.logits[context]
        log_probabilities = _log_softmax(context_logits)
        positions = np.flatnonzero(mask)
        return float(log_probabilities[positions, tokens[positions]].sum())

    def complete_transition_kernel(self, context_id: str) -> CompleteTransitionKernel:
        """Return the full fixed-schedule position/remask/residue kernel."""

        residue_probabilities = self.probabilities(context_id)
        position = np.full((1, self.length), 1.0 / self.length)
        remask = np.zeros((1, self.length, 2), dtype=np.float64)
        remask[:, :, 1] = 1.0  # The local adapter always reveals the selected position.
        residue = np.full(
            (1, self.length, 2, len(self.vocabulary.alphabet)),
            1.0 / len(self.vocabulary.alphabet),
        )
        residue[0, :, 1, :] = residue_probabilities
        return CompleteTransitionKernel(position, remask, residue)

    def local_transition_kl(
        self,
        denominator: TabularConditionalDenoiser,
        *,
        context_id: str,
    ) -> float:
        """Return mean-position ``KL(self || denominator)`` in logit space."""

        numerator_context, denominator_context = self._aligned_contexts(
            denominator,
            context_id=context_id,
        )
        per_position = _categorical_logit_kl(
            self.logits[numerator_context],
            denominator.logits[denominator_context],
        )
        return float(np.mean(per_position))

    def reference_path_kl(
        self,
        reference: TabularConditionalDenoiser,
        *,
        context_id: str,
    ) -> float:
        """Return length-step path ``KL(self || reference)`` in logit space."""

        numerator_context, reference_context = self._aligned_contexts(
            reference,
            context_id=context_id,
        )
        per_position = _categorical_logit_kl(
            self.logits[numerator_context],
            reference.logits[reference_context],
        )
        return float(np.sum(per_position))

    def update(
        self,
        *,
        context_id: str,
        replay_batch: EndpointReplayBatch | None,
        replay_microbatch_size: int | None = None,
        preferences: Sequence[DenoisingPreference] = (),
        learning_rate: float,
        preference_beta: float,
        local_kl_limit: float,
        reference_path_kl_limit: float,
        local_kl_p99_limit: float | None = None,
        reference_transition_kl_p99_limit: float | None = None,
        preference_clip: float = 4.0,
        maximum_backtracks: int = 24,
    ) -> TabularPolicyUpdate:
        """Take the largest backtracked step satisfying both KL controls."""

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(preference_beta) or preference_beta <= 0:
            raise ValueError("preference_beta must be finite and positive")
        if not np.isfinite(local_kl_limit) or local_kl_limit < 0:
            raise ValueError("local_kl_limit must be finite and non-negative")
        if not np.isfinite(reference_path_kl_limit) or reference_path_kl_limit < 0:
            raise ValueError("reference_path_kl_limit must be finite and non-negative")
        if local_kl_p99_limit is not None and (
            not np.isfinite(local_kl_p99_limit) or local_kl_p99_limit < 0
        ):
            raise ValueError("local_kl_p99_limit must be finite and non-negative")
        if reference_transition_kl_p99_limit is not None and (
            not np.isfinite(reference_transition_kl_p99_limit)
            or reference_transition_kl_p99_limit < 0
        ):
            raise ValueError("reference_transition_kl_p99_limit must be finite and non-negative")
        if not np.isfinite(preference_clip) or preference_clip <= 0:
            raise ValueError("preference_clip must be finite and positive")
        if isinstance(maximum_backtracks, bool) or not isinstance(
            maximum_backtracks,
            int | np.integer,
        ):
            raise ValueError("maximum_backtracks must be an integer")
        if maximum_backtracks < 0:
            raise ValueError("maximum_backtracks must be non-negative")
        pairs = tuple(preferences)
        if replay_batch is None and not pairs:
            raise ValueError("an update needs a replay batch or preference pairs")
        if replay_microbatch_size is not None and (
            isinstance(replay_microbatch_size, bool)
            or not isinstance(replay_microbatch_size, int | np.integer)
            or replay_microbatch_size <= 0
        ):
            raise ValueError("replay_microbatch_size must be a positive integer or None")
        context = self._context_index(context_id)
        self._validate_preference_stratum(
            pairs,
            context_id=context_id,
            replay_batch=replay_batch,
        )
        gradient = np.zeros_like(self.logits)
        if replay_batch is not None:
            self._add_replay_gradient(
                gradient[context],
                replay_batch,
                context_id=context_id,
                microbatch_size=(
                    len(replay_batch.sequences)
                    if replay_microbatch_size is None
                    else int(replay_microbatch_size)
                ),
            )
        margins_before = self._add_preference_gradient(
            gradient[context],
            pairs,
            context_id=context_id,
            preference_beta=preference_beta,
            preference_clip=preference_clip,
        )
        if not np.any(gradient):
            raise ValueError("the supplied replay data produced a zero update direction")

        reference_policy = TabularConditionalDenoiser(
            self.vocabulary,
            self.contexts,
            self.reference_logits,
            self.reference_logits,
            self.version,
        )
        for backtrack in range(maximum_backtracks + 1):
            scale = learning_rate * 0.5**backtrack
            candidate_logits = np.array(self.logits, copy=True)
            candidate_logits += scale * gradient
            candidate = TabularConditionalDenoiser(
                self.vocabulary,
                self.contexts,
                candidate_logits,
                self.reference_logits,
                self.version + 1,
            )
            local_kl = self.local_transition_kl(
                candidate,
                context_id=context_id,
            )
            reference_kl = candidate.reference_path_kl(
                reference_policy,
                context_id=context_id,
            )
            local_summary = self._transition_kl_summary(
                candidate,
                context_id=context_id,
            )
            reference_summary = candidate._transition_kl_summary(
                reference_policy,
                context_id=context_id,
            )
            local_tail_ok = local_kl_p99_limit is None or local_summary.p99 <= local_kl_p99_limit
            reference_tail_ok = (
                reference_transition_kl_p99_limit is None
                or reference_summary.p99 <= reference_transition_kl_p99_limit
            )
            if (
                local_kl <= local_kl_limit
                and reference_kl <= reference_path_kl_limit
                and local_tail_ok
                and reference_tail_ok
            ):
                margins_after = tuple(
                    candidate._preference_margin_against(pair, baseline_logits=self.logits)
                    for pair in pairs
                )
                return TabularPolicyUpdate(
                    policy=candidate,
                    step_scale=scale,
                    local_kl_old_new=local_kl,
                    local_transition_kl_summary=local_summary,
                    reference_path_kl_new_reference=reference_kl,
                    reference_transition_kl_summary=reference_summary,
                    preference_margin_before=margins_before,
                    preference_margin_after=margins_after,
                )
        raise RuntimeError("no positive backtracked step satisfies both KL limits")

    def _transition_kl_summary(
        self,
        denominator: TabularConditionalDenoiser,
        *,
        context_id: str,
    ) -> KLSummary:
        """Return position-level KL tails in this policy's declared direction."""

        numerator_context, denominator_context = self._aligned_contexts(
            denominator,
            context_id=context_id,
        )
        return summarize_kl(
            _categorical_logit_kl(
                self.logits[numerator_context],
                denominator.logits[denominator_context],
            )
        )

    def _add_replay_gradient(
        self,
        gradient: FloatArray,
        batch: EndpointReplayBatch,
        *,
        context_id: str,
        microbatch_size: int,
    ) -> None:
        if batch.context_id != context_id:
            raise ValueError("replay batch and policy update contexts must match")
        if batch.alphabet != self.vocabulary.alphabet:
            raise ValueError("replay batch and policy alphabets must match")
        if batch.clean_tokens.shape[1] != self.length:
            raise ValueError("replay batch length must match the tabular policy")
        if not np.all(batch.corruption_mask == batch.corruption_mask[:1]):
            raise ValueError("the first replay slice requires one common corruption mask")
        probabilities = self.probabilities(context_id)
        positions = np.flatnonzero(batch.corruption_mask[0])
        targets = batch.clean_tokens[:, positions]
        if np.any(targets < 0) or np.any(targets >= len(self.vocabulary.alphabet)):
            raise ValueError("replay targets must be amino-acid tokens")
        normalization = float(len(positions))
        gradient[positions] -= (float(np.sum(batch.weights)) / normalization) * probabilities[
            positions
        ]
        for start in range(0, len(batch.sequences), microbatch_size):
            stop = min(start + microbatch_size, len(batch.sequences))
            chunk_targets = targets[start:stop]
            row_positions = np.broadcast_to(positions[None, :], chunk_targets.shape)
            contributions = np.broadcast_to(
                batch.weights[start:stop, None] / normalization,
                chunk_targets.shape,
            )
            np.add.at(
                gradient,
                (row_positions.reshape(-1), chunk_targets.reshape(-1)),
                contributions.reshape(-1),
            )

    def _add_preference_gradient(
        self,
        gradient: FloatArray,
        pairs: tuple[DenoisingPreference, ...],
        *,
        context_id: str,
        preference_beta: float,
        preference_clip: float,
    ) -> tuple[float, ...]:
        margins: list[float] = []
        for pair in pairs:
            if pair.context_id != context_id:
                raise ValueError("preference and policy update contexts must match")
            winner = self._tokens(pair.winner)
            loser = self._tokens(pair.loser)
            mask = np.asarray(pair.corruption_mask, dtype=bool)
            margin = self._preference_margin_against(pair, baseline_logits=self.logits)
            scaled_margin = preference_beta * margin
            clipped_margin = float(np.clip(scaled_margin, -preference_clip, preference_clip))
            derivative = 0.0 if abs(scaled_margin) >= preference_clip else 1.0
            coefficient = (
                pair.weight
                * preference_beta
                * derivative
                * float(expit(-clipped_margin))
                / float(np.count_nonzero(mask))
            )
            for position in np.flatnonzero(mask):
                gradient[position, winner[position]] += coefficient
                gradient[position, loser[position]] -= coefficient
            margins.append(margin)
        return tuple(margins)

    def _preference_margin_against(
        self,
        pair: DenoisingPreference,
        *,
        baseline_logits: FloatArray,
    ) -> float:
        current = self.log_probability(
            pair.winner,
            context_id=pair.context_id,
            corruption_mask=pair.corruption_mask,
        ) - self.log_probability(
            pair.loser,
            context_id=pair.context_id,
            corruption_mask=pair.corruption_mask,
        )
        baseline = self._log_probability_from_logits(
            pair.winner,
            pair,
            logits=baseline_logits,
        ) - self._log_probability_from_logits(
            pair.loser,
            pair,
            logits=baseline_logits,
        )
        # ``ell`` in the replay contract is the mean masked log-score.  Keeping
        # the margin on that scale makes the value and the /|M| gradient above
        # derivatives of the same bounded preference objective.
        mask_size = float(np.count_nonzero(pair.corruption_mask))
        return (current - baseline) / mask_size

    def _log_probability_from_logits(
        self,
        sequence: str,
        pair: DenoisingPreference,
        *,
        logits: FloatArray,
    ) -> float:
        context = self._context_index(pair.context_id)
        tokens = self._tokens(sequence)
        mask = np.asarray(pair.corruption_mask, dtype=bool)
        context_logits = logits[context]
        log_probabilities = _log_softmax(context_logits)
        positions = np.flatnonzero(mask)
        return float(log_probabilities[positions, tokens[positions]].sum())

    def _validate_preference_stratum(
        self,
        pairs: tuple[DenoisingPreference, ...],
        *,
        context_id: str,
        replay_batch: EndpointReplayBatch | None,
    ) -> None:
        if not pairs:
            return
        first_mask = np.asarray(pairs[0].corruption_mask, dtype=bool)
        first_timestep = pairs[0].timestep
        for pair in pairs:
            if pair.context_id != context_id:
                raise ValueError("preference and policy update contexts must match")
            if pair.timestep != first_timestep or not np.array_equal(
                np.asarray(pair.corruption_mask, dtype=bool),
                first_mask,
            ):
                raise ValueError("preference pairs must share one timestep and corruption mask")
        if replay_batch is not None and (
            replay_batch.timestep != first_timestep
            or not np.array_equal(replay_batch.corruption_mask[0], first_mask)
        ):
            raise ValueError(
                "replay and preference examples must share one timestep and corruption mask"
            )

    def _aligned_contexts(
        self,
        other: TabularConditionalDenoiser,
        *,
        context_id: str,
    ) -> tuple[int, int]:
        if not isinstance(other, TabularConditionalDenoiser):
            raise TypeError("denominator must be a TabularConditionalDenoiser")
        if self.vocabulary.alphabet != other.vocabulary.alphabet:
            raise ValueError("KL policies must use the same alphabet")
        if self.length != other.length:
            raise ValueError("KL policies must use the same sequence length")
        return self._context_index(context_id), other._context_index(context_id)

    def _tokens(self, sequence: str) -> NDArray[np.int64]:
        encoded = self.vocabulary.encode([sequence], max_length=self.length)
        if not bool(np.all(encoded.attention_mask)):
            raise ValueError("sequence length must match the tabular policy")
        return encoded.tokens[0]

    def _context_index(self, context_id: str) -> int:
        try:
            return self.contexts.index(context_id)
        except ValueError as error:
            raise ValueError(f"unknown property context: {context_id!r}") from error

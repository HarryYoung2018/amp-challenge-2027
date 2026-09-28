"""Framework-neutral replay planning and complete-transition KL mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.categorical import (
    AbsorbingDiffusion,
    PeptideVocabulary,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _normalized_sequence(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip().upper()


def effective_sample_size(weights: Sequence[float] | FloatArray) -> float:
    """Return ``(sum w)^2 / sum(w^2)`` for finite non-negative weights."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    scale = float(np.max(values))
    if scale <= 0:
        return 0.0
    scaled = values / scale
    total = float(scaled.sum())
    return total * total / float(scaled @ scaled)


@dataclass(frozen=True, slots=True)
class PositiveEndpointWeights:
    """Parent-anchor and child weights for one context-specific lineage."""

    update_enabled: bool
    parent_weight: float
    child_weights: FloatArray
    effective_sample_size: float


def positive_endpoint_weights(
    advantages: Sequence[float] | FloatArray,
    accepted: Sequence[bool] | BoolArray,
    *,
    temperature: float,
    clip: float,
) -> PositiveEndpointWeights:
    """Build bounded exponential weights without promoting an all-worse batch.

    The no-op parent has advantage zero.  Rejected children receive weight zero.
    When no child passes the relative, absolute, and feasibility gates, every
    returned weight is zero and the caller must skip the positive update.
    """

    advantage = np.asarray(advantages, dtype=np.float64)
    acceptance = np.asarray(accepted)
    if advantage.ndim != 1 or advantage.size == 0:
        raise ValueError("advantages must be a non-empty vector")
    if np.any(~np.isfinite(advantage)):
        raise ValueError("advantages must be finite")
    if acceptance.dtype.kind != "b" or acceptance.shape != advantage.shape:
        raise ValueError("accepted must be a matching boolean vector")
    if np.any(acceptance & (advantage <= 0)):
        raise ValueError("accepted children must have strictly positive advantages")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")

    child_weights = np.zeros_like(advantage)
    if not np.any(acceptance):
        child_weights.setflags(write=False)
        return PositiveEndpointWeights(False, 0.0, child_weights, 0.0)

    accepted_advantages = advantage[acceptance]
    logits = np.concatenate(
        [np.array([0.0]), np.clip(accepted_advantages / temperature, -clip, clip)]
    )
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    weights /= weights.sum()
    child_weights[acceptance] = weights[1:]
    child_weights.setflags(write=False)
    return PositiveEndpointWeights(
        update_enabled=True,
        parent_weight=float(weights[0]),
        child_weights=child_weights,
        effective_sample_size=effective_sample_size(weights),
    )


@dataclass(frozen=True, slots=True)
class EndpointReplayBatch:
    """Common-corruption denoising examples for one parent and its winners."""

    sequences: tuple[str, ...]
    context_id: str
    alphabet: str
    clean_tokens: IntArray
    corrupted_tokens: IntArray
    attention_mask: BoolArray
    corruption_mask: BoolArray
    weights: FloatArray
    timestep: float

    def __post_init__(self) -> None:
        sequences = tuple(
            _normalized_sequence(sequence, name="replay sequence") for sequence in self.sequences
        )
        if not sequences or len(sequences) != len(set(sequences)):
            raise ValueError("replay sequences must be non-empty and unique")
        if any(not sequence for sequence in sequences):
            raise ValueError("replay sequences cannot contain empty strings")
        if len({len(sequence) for sequence in sequences}) != 1:
            raise ValueError("the first replay slice requires equal-length sequences")
        if not self.context_id:
            raise ValueError("context_id cannot be empty")
        if not isinstance(self.alphabet, str) or not self.alphabet:
            raise ValueError("alphabet must be a non-empty string")
        expected_rows = len(sequences)
        clean = np.asarray(self.clean_tokens)
        corrupted = np.asarray(self.corrupted_tokens)
        attention = np.asarray(self.attention_mask)
        corruption = np.asarray(self.corruption_mask)
        arrays = (clean, corrupted, attention, corruption)
        if any(array.ndim != 2 or array.shape[0] != expected_rows for array in arrays):
            raise ValueError("token and mask arrays must align with replay sequences")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("token and mask arrays must have identical shapes")
        if clean.dtype.kind not in "iu" or corrupted.dtype.kind not in "iu":
            raise ValueError("clean and corrupted tokens must be integer arrays")
        if np.any(clean < 0) or np.any(corrupted < 0):
            raise ValueError("clean and corrupted tokens must be non-negative")
        if attention.dtype.kind != "b" or corruption.dtype.kind != "b":
            raise ValueError("attention and corruption masks must be boolean arrays")
        if np.any(corruption & ~attention):
            raise ValueError("corruption_mask must be contained in attention_mask")
        if not np.all(corruption == corruption[:1]) or not np.any(corruption[0]):
            raise ValueError("replay rows must share one non-empty corruption mask")
        sequence_lengths = np.asarray([len(sequence) for sequence in sequences])
        if not np.array_equal(attention.sum(axis=1), sequence_lengths):
            raise ValueError("attention_mask must match replay sequence lengths")
        vocabulary = PeptideVocabulary(self.alphabet)
        encoded = vocabulary.encode(sequences, max_length=clean.shape[1])
        if not np.array_equal(clean, encoded.tokens) or not np.array_equal(
            attention,
            encoded.attention_mask,
        ):
            raise ValueError("clean tokens and attention mask must encode replay sequences")
        expected_corrupted = np.array(clean, copy=True)
        expected_corrupted[corruption] = vocabulary.mask_index
        if not np.array_equal(corrupted, expected_corrupted):
            raise ValueError("corrupted tokens must match the recorded absorbing-mask state")
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if weights.shape != (expected_rows,) or np.any(weights <= 0):
            raise ValueError("weights must be positive and align with replay sequences")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("replay weights must sum to one")
        if not np.isfinite(self.timestep) or not 0 < self.timestep <= 1:
            raise ValueError("timestep must lie in (0, 1]")
        clean = np.array(clean, dtype=np.int64, copy=True)
        corrupted = np.array(corrupted, dtype=np.int64, copy=True)
        attention = np.array(attention, dtype=bool, copy=True)
        corruption = np.array(corruption, dtype=bool, copy=True)
        for array in (clean, corrupted, attention, corruption, weights):
            array.setflags(write=False)
        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "clean_tokens", clean)
        object.__setattr__(self, "corrupted_tokens", corrupted)
        object.__setattr__(self, "attention_mask", attention)
        object.__setattr__(self, "corruption_mask", corruption)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, slots=True)
class EndpointReplayPlan:
    """A positive batch or an explicit all-worse skip decision."""

    update_enabled: bool
    batch: EndpointReplayBatch | None
    rejected_child_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.update_enabled != (self.batch is not None):
            raise ValueError("update_enabled must agree with batch presence")


class EndpointReplayBuilder:
    """Create common-mask endpoint supervision through one small interface."""

    def __init__(
        self,
        *,
        vocabulary: PeptideVocabulary | None = None,
        diffusion: AbsorbingDiffusion | None = None,
    ) -> None:
        self.vocabulary = vocabulary or PeptideVocabulary()
        self.diffusion = diffusion or AbsorbingDiffusion(self.vocabulary)
        if self.diffusion.vocabulary.alphabet != self.vocabulary.alphabet:
            raise ValueError("diffusion and replay vocabulary alphabets must match")

    def build(
        self,
        parent: str,
        children: Sequence[str],
        *,
        context_id: str,
        advantages: Sequence[float] | FloatArray,
        accepted: Sequence[bool] | BoolArray,
        timestep: float,
        temperature: float,
        clip: float,
        rng: np.random.Generator,
    ) -> EndpointReplayPlan:
        """Build one equal-length, same-context batch with a common mask."""

        normalized_parent = _normalized_sequence(parent, name="parent")
        child_sequences = tuple(_normalized_sequence(child, name="child") for child in children)
        if not child_sequences:
            raise ValueError("children cannot be empty")
        if not context_id:
            raise ValueError("context_id cannot be empty")
        if any(len(child) != len(normalized_parent) for child in child_sequences):
            raise ValueError("the first replay slice requires equal-length parent/child pairs")
        advantage_values = np.asarray(advantages, dtype=np.float64)
        acceptance_values = np.asarray(accepted)
        if advantage_values.shape != (len(child_sequences),):
            raise ValueError("advantages must have one value per child")
        if acceptance_values.shape != (len(child_sequences),):
            raise ValueError("accepted must have one value per child")
        weights = positive_endpoint_weights(
            advantage_values,
            acceptance_values,
            temperature=temperature,
            clip=clip,
        )
        rejected = tuple(
            index for index, is_accepted in enumerate(np.asarray(accepted)) if not is_accepted
        )
        if not weights.update_enabled:
            return EndpointReplayPlan(False, None, rejected)

        acceptance = np.asarray(acceptance_values, dtype=bool)
        winners = tuple(
            child
            for child, is_accepted in zip(child_sequences, acceptance, strict=True)
            if is_accepted
        )
        sequences = (normalized_parent, *winners)
        if len(set(sequences)) != len(sequences):
            raise ValueError("accepted children must differ from the parent and each other")
        encoded = self.vocabulary.encode(sequences, max_length=len(normalized_parent))
        parent_corrupted, parent_mask = self.diffusion.corrupt(
            encoded.tokens[:1],
            encoded.attention_mask[:1],
            timestep,
            rng=rng,
            ensure_masked=True,
        )
        del parent_corrupted
        corruption_mask = np.repeat(parent_mask, len(sequences), axis=0)
        corrupted = encoded.tokens.copy()
        corrupted[corruption_mask] = self.vocabulary.mask_index
        row_weights = np.concatenate(
            [np.array([weights.parent_weight]), weights.child_weights[acceptance]]
        )
        return EndpointReplayPlan(
            update_enabled=True,
            batch=EndpointReplayBatch(
                sequences=sequences,
                context_id=context_id,
                alphabet=self.vocabulary.alphabet,
                clean_tokens=encoded.tokens,
                corrupted_tokens=corrupted,
                attention_mask=encoded.attention_mask,
                corruption_mask=corruption_mask,
                weights=row_weights,
                timestep=float(timestep),
            ),
            rejected_child_indices=rejected,
        )


@dataclass(frozen=True, slots=True)
class CompleteTransitionKernel:
    """Chain-factorized reverse transition at a collection of states.

    The factors are ``P(position)``, ``P(remask | position)``, and
    ``P(residue | position, remask)``.  This is deliberately broader than a
    residue-logit tensor, so the resulting KL has path-level meaning when
    aggregated under the correct state-visitation measure.
    """

    position: FloatArray
    remask_given_position: FloatArray
    residue_given_position_remask: FloatArray

    def __post_init__(self) -> None:
        position = _probabilities(self.position, name="position", ndim=2)
        remask = _probabilities(
            self.remask_given_position,
            name="remask_given_position",
            ndim=3,
        )
        residue = _probabilities(
            self.residue_given_position_remask,
            name="residue_given_position_remask",
            ndim=4,
        )
        n_states, n_positions = position.shape
        if remask.shape[:2] != (n_states, n_positions):
            raise ValueError("remask factors must align with states and positions")
        if residue.shape[:3] != (*remask.shape[:2], remask.shape[2]):
            raise ValueError("residue factors must align with states, positions, and remask")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "remask_given_position", remask)
        object.__setattr__(self, "residue_given_position_remask", residue)

    @property
    def n_states(self) -> int:
        return int(self.position.shape[0])


@dataclass(frozen=True, slots=True)
class KLSummary:
    """Distribution-aware KL diagnostics for a batched update.

    A mean alone can hide a small number of extreme policy changes.  The fixed
    quantiles make local and cluster runs comparable without retaining every
    per-item divergence in the audit artifact.
    """

    mean: float
    p50: float
    p95: float
    p99: float
    maximum: float
    sample_count: int

    def __post_init__(self) -> None:
        values = (self.mean, self.p50, self.p95, self.p99, self.maximum)
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("KL diagnostics must be finite and non-negative")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int | np.integer
        ):
            raise ValueError("sample_count must be an integer")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not self.p50 <= self.p95 <= self.p99 <= self.maximum:
            raise ValueError("KL quantiles must be ordered")


def summarize_kl(
    per_item: Sequence[float] | FloatArray,
    *,
    weights: Sequence[float] | FloatArray | None = None,
) -> KLSummary:
    """Summarize non-negative per-item KLs with optional frequency weights."""

    values = np.asarray(per_item, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("per_item KL must be a non-empty vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("per_item KL must be finite and non-negative")
    if weights is None:
        normalized_weights = np.full(values.size, 1.0 / values.size)
    else:
        normalized_weights = _state_weights(weights, n_states=values.size, normalize=True)

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = np.cumsum(normalized_weights[order])
    # Avoid a final cumulative mass such as 0.9999999999999999 causing the
    # 0.99 quantile to miss the last supported item after later refactors.
    cumulative[-1] = 1.0

    def weighted_quantile(probability: float) -> float:
        position = int(np.searchsorted(cumulative, probability, side="left"))
        return float(ordered[min(position, ordered.size - 1)])

    return KLSummary(
        mean=float(normalized_weights @ values),
        p50=weighted_quantile(0.50),
        p95=weighted_quantile(0.95),
        p99=weighted_quantile(0.99),
        maximum=float(np.max(values[normalized_weights > 0.0])),
        sample_count=int(np.count_nonzero(normalized_weights)),
    )


def _probabilities(values: object, *, name: str, ndim: int) -> FloatArray:
    probabilities = np.array(values, dtype=np.float64, copy=True)
    if probabilities.ndim != ndim or 0 in probabilities.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}D probability array")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    row_mass = probabilities.sum(axis=-1, keepdims=True)
    if not np.allclose(row_mass, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} probabilities must sum to one on the final axis")
    # The tolerance above is an input convenience, not permission to feed
    # non-probability measures into a KL.  Canonicalize accepted rows so the
    # chain-rule calculation retains KL non-negativity.
    probabilities /= row_mass
    probabilities.setflags(write=False)
    return probabilities


def complete_transition_kl(
    numerator: CompleteTransitionKernel,
    denominator: CompleteTransitionKernel,
) -> FloatArray:
    """Return per-state ``KL(numerator || denominator)`` by the chain rule."""

    return _complete_transition_kl(
        numerator,
        denominator,
        active_states=np.ones(numerator.n_states, dtype=bool),
    )


def _complete_transition_kl(
    numerator: CompleteTransitionKernel,
    denominator: CompleteTransitionKernel,
    *,
    active_states: BoolArray,
) -> FloatArray:
    if numerator.position.shape != denominator.position.shape:
        raise ValueError("position kernels must have identical shapes")
    if numerator.remask_given_position.shape != denominator.remask_given_position.shape:
        raise ValueError("remask kernels must have identical shapes")
    if (
        numerator.residue_given_position_remask.shape
        != denominator.residue_given_position_remask.shape
    ):
        raise ValueError("residue kernels must have identical shapes")

    if active_states.shape != (numerator.n_states,):
        raise ValueError("active_states must have one value per transition state")
    position_kl = _categorical_kl(
        numerator.position,
        denominator.position,
        active=active_states,
    )
    active_positions = active_states[:, None] & (numerator.position > 0)
    remask_kl = _categorical_kl(
        numerator.remask_given_position,
        denominator.remask_given_position,
        active=active_positions,
    )
    active_remask = active_positions[:, :, None] & (numerator.remask_given_position > 0)
    residue_kl = _categorical_kl(
        numerator.residue_given_position_remask,
        denominator.residue_given_position_remask,
        active=active_remask,
    )
    conditional_residue = np.sum(numerator.remask_given_position * residue_kl, axis=-1)
    conditional = remask_kl + conditional_residue
    total = position_kl + np.sum(numerator.position * conditional, axis=-1)
    return np.asarray(total, dtype=np.float64)


def _categorical_kl(
    numerator: FloatArray,
    denominator: FloatArray,
    *,
    active: BoolArray,
) -> FloatArray:
    if active.shape != numerator.shape[:-1]:
        raise ValueError("active conditioning mask must match categorical batches")
    supported_event = active[..., None]
    unsupported = supported_event & (numerator > 0) & (denominator <= 0)
    if np.any(unsupported):
        raise ValueError("denominator transition kernel lacks numerator support")
    terms = np.zeros_like(numerator)
    positive = supported_event & (numerator > 0)
    terms[positive] = numerator[positive] * (
        np.log(numerator[positive]) - np.log(denominator[positive])
    )
    divergence = np.sum(terms, axis=-1)
    roundoff = (
        64.0
        * np.finfo(np.float64).eps
        * np.maximum(
            1.0,
            np.sum(np.abs(terms), axis=-1),
        )
    )
    if np.any(divergence < -roundoff):
        raise FloatingPointError("categorical KL became negative beyond float64 roundoff")
    return np.maximum(divergence, 0.0)


def local_trust_region_kl(
    old: CompleteTransitionKernel,
    new: CompleteTransitionKernel,
    *,
    state_weights: Sequence[float] | FloatArray | None = None,
) -> float:
    """Return empirical mean ``KL(old || new)`` at forward-noised anchors."""

    return local_trust_region_kl_diagnostics(
        old,
        new,
        state_weights=state_weights,
    ).mean


def local_trust_region_kl_diagnostics(
    old: CompleteTransitionKernel,
    new: CompleteTransitionKernel,
    *,
    state_weights: Sequence[float] | FloatArray | None = None,
) -> KLSummary:
    """Return mean and tail ``KL(old || new)`` across replay anchors."""

    weights = _state_weights(state_weights, n_states=old.n_states, normalize=True)
    per_state = _complete_transition_kl(
        old,
        new,
        active_states=weights > 0,
    )
    return summarize_kl(per_state, weights=weights)


def frozen_reference_path_kl(
    current: CompleteTransitionKernel,
    reference: CompleteTransitionKernel,
    *,
    visitation_weights: Sequence[float] | FloatArray,
) -> float:
    """Return ``sum_s d_current(s) KL(current || reference)``.

    ``visitation_weights`` must be the current policy's expected state-occupancy
    measure, or an unbiased empirical estimate of it.  They are not normalized:
    their sum may equal the number of reverse steps.  Only with that occupancy
    interpretation is the result a path-KL estimate rather than an arbitrary
    weighted sum of local transition KLs.
    """

    weights = _state_weights(
        visitation_weights,
        n_states=current.n_states,
        normalize=False,
    )
    per_state = _complete_transition_kl(
        current,
        reference,
        active_states=weights > 0,
    )
    return float(weights @ per_state)


def frozen_reference_path_kl_diagnostics(
    current: CompleteTransitionKernel,
    reference: CompleteTransitionKernel,
    *,
    trajectory_ids: Sequence[str | int],
    state_weights: Sequence[float] | FloatArray | None = None,
) -> KLSummary:
    """Summarize complete reference-path KL after grouping transition states.

    Tail diagnostics must be computed over whole trajectories, not individual
    transition states.  ``state_weights`` may carry unbiased multiplicities or
    importance weights; when omitted, each logged transition contributes once.
    """

    identifiers = tuple(trajectory_ids)
    if len(identifiers) != current.n_states:
        raise ValueError("trajectory_ids must have one value per transition state")
    if any(not isinstance(value, str | int) or isinstance(value, bool) for value in identifiers):
        raise ValueError("trajectory_ids must contain strings or integers")
    if not identifiers:
        raise ValueError("trajectory_ids cannot be empty")
    if state_weights is None:
        weights = np.ones(current.n_states, dtype=np.float64)
    else:
        weights = _state_weights(
            state_weights,
            n_states=current.n_states,
            normalize=False,
        )
    per_state = _complete_transition_kl(
        current,
        reference,
        active_states=weights > 0,
    )
    totals: dict[str | int, float] = {}
    for identifier, weight, divergence in zip(
        identifiers,
        weights,
        per_state,
        strict=True,
    ):
        totals.setdefault(identifier, 0.0)
        totals[identifier] += float(weight * divergence)
    return summarize_kl(tuple(totals.values()))


def _state_weights(
    values: Sequence[float] | FloatArray | None,
    *,
    n_states: int,
    normalize: bool,
) -> FloatArray:
    if values is None:
        weights = np.full(n_states, 1.0 / n_states)
    else:
        weights = np.asarray(values, dtype=np.float64)
        if weights.shape != (n_states,):
            raise ValueError("state weights must have one value per transition state")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("state weights must be finite, non-negative, and non-zero")
        scale = float(np.max(weights))
        if scale <= 0:
            raise ValueError("state weights must be finite, non-negative, and non-zero")
        if normalize:
            scaled = weights / scale
            weights = scaled / scaled.sum()
    return np.asarray(weights, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ReplayDiagnostics:
    effective_sample_size: float
    effective_sample_fraction: float
    maximum_normalized_weight: float
    version_lag: int
    stale: bool


def replay_diagnostics(
    weights: Sequence[float] | FloatArray,
    *,
    behavior_version: int,
    current_version: int,
    maximum_version_lag: int,
    minimum_ess_fraction: float,
) -> ReplayDiagnostics:
    """Audit importance/replay weight concentration and policy staleness."""

    values = np.asarray(weights, dtype=np.float64)
    ess = effective_sample_size(values)
    scale = float(np.max(values))
    scaled = values / scale if scale > 0 else np.zeros_like(values)
    scaled_total = float(scaled.sum())
    normalized = scaled / scaled_total if scaled_total > 0 else np.zeros_like(values)
    if behavior_version < 0 or current_version < behavior_version:
        raise ValueError("policy versions must be ordered non-negative integers")
    if maximum_version_lag < 0:
        raise ValueError("maximum_version_lag must be non-negative")
    if not 0 < minimum_ess_fraction <= 1:
        raise ValueError("minimum_ess_fraction must lie in (0, 1]")
    lag = current_version - behavior_version
    fraction = ess / len(values)
    return ReplayDiagnostics(
        effective_sample_size=ess,
        effective_sample_fraction=fraction,
        maximum_normalized_weight=float(np.max(normalized)),
        version_lag=lag,
        stale=lag > maximum_version_lag or fraction < minimum_ess_fraction,
    )

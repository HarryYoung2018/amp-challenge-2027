"""Research-only native peptide proposal traces and conditional path diagnostics.

These probabilities condition on parent, length, start level and operator.
They are augmented remask/commit PATH probabilities, not marginalized endpoint
probabilities, acquisition-selection likelihoods, or whole-search guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule, PeptideVocabulary
from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    NativeDenoiser,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    ALPHABET,
    NATIVE_ENDPOINT_DEFAULTS,
    NativeEndpointConfig,
    NativeTransitionState,
    _json_hash,
    _seed,
    _state_contract,
    _validate_model,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.subset_kernel import (
    SubsetCommitDraw,
    complete_subset_commit_kl,
)

# Declared before CPU comparisons; these qualify neural batch arithmetic only.
PROBABILITY_ATOL = 1e-6
PROBABILITY_RTOL = 1e-5


@dataclass(frozen=True, slots=True)
class NativeProposalStep:
    level: int
    before_tokens_sha256: str
    after_tokens_sha256: str
    draw: SubsetCommitDraw


@dataclass(frozen=True, slots=True)
class NativeProposalTrace:
    parent: str
    parent_sha256: str
    endpoint: str
    endpoint_sha256: str
    start_level: int
    seed: int
    ordinal: int
    model_sha256: str
    model_config_sha256: str
    kernel_epsilon: float
    remasked_positions: tuple[int, ...]
    initial_remask_log_probability: float
    steps: tuple[NativeProposalStep, ...]
    augmented_path_log_probability: float
    scope: str = "conditional_parent_length_start_level_operator_path_not_endpoint_or_whole_search"


def _sequence_hash(sequence: str) -> str:
    import hashlib

    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _model_config_hash(model: NativeDenoiser) -> str:
    from dataclasses import asdict

    return _json_hash(asdict(model.config))


def sample_native_proposal(
    model: NativeDenoiser,
    parent: str,
    *,
    start_level: int,
    seed: int,
    ordinal: int = 0,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> NativeProposalTrace:
    """Single-candidate wrapper preserving explicit per-candidate randomness."""
    return sample_native_proposals(
        model, (parent,), start_levels=(start_level,), seed=seed, ordinals=(ordinal,), config=config
    )[0]


def sample_native_proposals(
    model: NativeDenoiser,
    parents: tuple[str, ...],
    *,
    start_levels: tuple[int, ...],
    seed: int,
    ordinals: tuple[int, ...],
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> tuple[NativeProposalTrace, ...]:
    """Batched uniform remask/commits; bounded rows, microbatches and model scans.

    No rejection, uniqueness filtering, oracle evaluation or production insertion.
    Position RNG is independent of model weights and batching/order. Residue
    probabilities are tolerance-qualified, not promised GPU-bitwise invariant.
    """
    parents, start_levels, ordinals = tuple(parents), tuple(start_levels), tuple(ordinals)
    if (
        not 1 <= len(parents) <= config.maximum_replay_rows
        or len(start_levels) != len(parents)
        or len(ordinals) != len(parents)
        or len(set(ordinals)) != len(ordinals)
    ):
        raise ValueError("proposal batch needs bounded aligned rows and unique explicit ordinals")
    _validate_model(model, config)
    if any(
        not isinstance(parent, str)
        or set(parent) - set(ALPHABET)
        or not model.config.min_length <= len(parent) <= model.config.max_length
        for parent in parents
    ):
        raise ValueError("parent must be a bounded canonical peptide")
    if any(
        type(level) is not int or not 1 <= level <= model.config.levels for level in start_levels
    ):
        raise ValueError("start level must be a positive native discrete level")
    model_hash = canonical_model_logical_hash(model)
    architecture_hash = _model_config_hash(model)
    tokens = PeptideVocabulary().encode(parents, max_length=model.config.max_length).tokens
    initial_positions, initial_logps = [], []
    for row, (parent, level, ordinal) in enumerate(
        zip(parents, start_levels, ordinals, strict=True)
    ):
        count = int(
            CosineMaskSchedule().mask_counts(len(parent), level, total_levels=model.config.levels)[
                0
            ]
        )
        positions = tuple(
            sorted(
                int(value)
                for value in _seed(seed, ordinal, "initial-remask", level).choice(
                    len(parent), size=count, replace=False
                )
            )
        )
        tokens[row, list(positions)] = MASK_TOKEN_INDEX
        initial_positions.append(positions)
        initial_logps.append(-math.log(math.comb(len(parent), count)))
    path_logps = initial_logps.copy()
    steps: list[list[NativeProposalStep]] = [[] for _ in parents]
    for level in range(max(start_levels), 0, -1):
        active = [row for row, start in enumerate(start_levels) if start >= level]
        states = tuple(
            NativeTransitionState(tokens[row], len(parents[row]), level) for row in active
        )
        kernels = _validated_transition_kernels(model, states, config)
        for row, kernel in zip(active, kernels, strict=True):
            draw = kernel.sample(_seed(seed, ordinals[row], "reverse-commit", level))
            before = _json_hash(tokens[row].tolist())
            tokens[row, list(draw.positions)] = draw.residues
            steps[row].append(
                NativeProposalStep(level, before, _json_hash(tokens[row].tolist()), draw)
            )
            path_logps[row] += draw.log_probability
    endpoints = PeptideVocabulary().decode(tokens)
    if canonical_model_logical_hash(model) != model_hash:
        raise RuntimeError("policy changed during proposal generation")
    return tuple(
        NativeProposalTrace(
            parent,
            _sequence_hash(parent),
            endpoints[row],
            _sequence_hash(endpoints[row]),
            start_levels[row],
            seed,
            ordinals[row],
            model_hash,
            architecture_hash,
            config.epsilon,
            initial_positions[row],
            initial_logps[row],
            tuple(steps[row]),
            path_logps[row],
        )
        for row, parent in enumerate(parents)
    )


def replay_native_trace(
    model: NativeDenoiser,
    trace: NativeProposalTrace,
    *,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
    authenticate_sampling: bool = False,
) -> tuple[float, tuple[NativeTransitionState, ...]]:
    """Reconstruct every state/action and score the path under this policy.

    With authenticate_sampling=True, also recheck current model identity, explicit
    per-step random draws and logged probabilities. Reference scoring reconstructs
    the same supported events without mistaking them for reference-policy samples.
    """
    _validate_model(model, config)
    if (
        trace.parent_sha256 != _sequence_hash(trace.parent)
        or trace.endpoint_sha256 != _sequence_hash(trace.endpoint)
        or trace.model_config_sha256 != _model_config_hash(model)
        or trace.kernel_epsilon != config.epsilon
    ):
        raise ValueError("proposal trace sequence/config identity mismatch")
    if authenticate_sampling and trace.model_sha256 != canonical_model_logical_hash(model):
        raise ValueError("trace was not sampled from the supplied current policy")
    if (
        type(trace.start_level) is not int
        or not 1 <= trace.start_level <= model.config.levels
        or len(trace.steps) != trace.start_level
    ):
        raise ValueError("trace reverse levels are incomplete")
    tokens = (
        PeptideVocabulary().encode([trace.parent], max_length=model.config.max_length).tokens[0]
    )
    count = int(
        CosineMaskSchedule().mask_counts(
            len(trace.parent), trace.start_level, total_levels=model.config.levels
        )[0]
    )
    positions = trace.remasked_positions
    if (
        len(positions) != count
        or tuple(sorted(set(positions))) != positions
        or any(type(value) is not int or not 0 <= value < len(trace.parent) for value in positions)
    ):
        raise ValueError("trace initial remask event is not supported")
    if authenticate_sampling:
        expected = tuple(
            sorted(
                int(value)
                for value in _seed(
                    trace.seed, trace.ordinal, "initial-remask", trace.start_level
                ).choice(len(trace.parent), size=count, replace=False)
            )
        )
        if positions != expected:
            raise ValueError("trace initial remask randomness mismatch")
    tokens[list(positions)] = MASK_TOKEN_INDEX
    path_logp = -math.log(math.comb(len(trace.parent), count))
    if not math.isclose(path_logp, trace.initial_remask_log_probability, rel_tol=0, abs_tol=1e-12):
        raise ValueError("trace initial remask log probability mismatch")
    states = []
    for level, step in zip(range(trace.start_level, 0, -1), trace.steps, strict=True):
        if step.level != level or step.before_tokens_sha256 != _json_hash(tokens.tolist()):
            raise ValueError("trace predecessor state/level mismatch")
        state = NativeTransitionState(tokens, len(trace.parent), level)
        states.append(state)
        masked, count = _state_contract(state, model.config)
        if (
            len(step.draw.positions) != count
            or len(step.draw.residues) != count
            or tuple(sorted(set(step.draw.positions))) != step.draw.positions
            or any(type(value) is not int or value not in masked for value in step.draw.positions)
            or any(type(value) is not int or not 0 <= value < 20 for value in step.draw.residues)
        ):
            raise ValueError("trace commit event is not supported")
        tokens[list(step.draw.positions)] = step.draw.residues
        if step.after_tokens_sha256 != _json_hash(tokens.tolist()):
            raise ValueError("trace successor state mismatch")
    if PeptideVocabulary().decode(tokens[None, :])[0] != trace.endpoint:
        raise ValueError("trace endpoint mismatch")
    kernels = _validated_transition_kernels(model, tuple(states), config)
    for step, kernel in zip(trace.steps, kernels, strict=True):
        value = kernel.log_probability(step.draw.positions, step.draw.residues)
        if authenticate_sampling:
            expected_draw = kernel.sample(
                _seed(trace.seed, trace.ordinal, "reverse-commit", step.level)
            )
            if (
                expected_draw.positions != step.draw.positions
                or expected_draw.residues != step.draw.residues
                or not math.isclose(
                    value,
                    step.draw.log_probability,
                    rel_tol=PROBABILITY_RTOL,
                    abs_tol=PROBABILITY_ATOL,
                )
            ):
                raise ValueError("trace current-policy random draw/log probability mismatch")
        path_logp += value
    if authenticate_sampling and not math.isclose(
        path_logp,
        trace.augmented_path_log_probability,
        rel_tol=PROBABILITY_RTOL,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("trace complete path log probability mismatch")
    return path_logp, tuple(states)


@dataclass(frozen=True, slots=True)
class ConditionalPathKLMonteCarlo:
    mean: float
    monte_carlo_standard_error: float
    trajectory_values: tuple[float, ...]
    trajectory_count: int
    sampled_log_ratio_mean: float
    current_model_sha256: str
    reference_model_sha256: str
    parent_sha256: str
    seed: int
    start_level: int
    state_weighting: str = "one_per_current_policy_visited_state_then_equal_trajectory_mean"
    scope: str = "conditional_current_rollout_path_kl_mc_not_global_bound"


def current_rollout_reference_kl(
    model: NativeDenoiser,
    reference: NativeDenoiser,
    parent: str,
    *,
    start_level: int,
    seed: int,
    trajectories: int = 8,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> ConditionalPathKLMonteCarlo:
    """Generate current-policy rollouts internally; arbitrary anchors cannot enter.

    The fixed parent/start-level protocol uses independently keyed trajectories,
    unit state occupancies, and within-parent sample variance for MCSE. Finite
    estimates, including small MCSE, are not certified path/global KL bounds.
    """
    if type(trajectories) is not int or not 2 <= trajectories <= 32:
        raise ValueError("path MC requires 2..32 bounded current-policy trajectories")
    if model.config != reference.config:
        raise ValueError("current/reference architecture and schedule differ")
    _validate_model(model, config)
    _validate_model(reference, config)
    current_hash = canonical_model_logical_hash(model)
    reference_hash = canonical_model_logical_hash(reference)
    values, log_ratios = [], []
    traces = sample_native_proposals(
        model,
        (parent,) * trajectories,
        start_levels=(start_level,) * trajectories,
        seed=seed,
        ordinals=tuple(range(trajectories)),
        config=config,
    )
    for trace in traces:
        current_logp, states = replay_native_trace(
            model, trace, config=config, authenticate_sampling=True
        )
        reference_logp, _ = replay_native_trace(reference, trace, config=config)
        current_kernels = _validated_transition_kernels(model, states, config)
        reference_kernels = _validated_transition_kernels(reference, states, config)
        values.append(
            sum(
                complete_subset_commit_kl(current, frozen)
                for current, frozen in zip(current_kernels, reference_kernels, strict=True)
            )
        )
        log_ratios.append(current_logp - reference_logp)
    if (
        canonical_model_logical_hash(model) != current_hash
        or canonical_model_logical_hash(reference) != reference_hash
    ):
        raise RuntimeError("current/reference weights changed during MC rollouts")
    array = np.asarray(values, dtype=np.float64)
    return ConditionalPathKLMonteCarlo(
        float(array.mean()),
        float(array.std(ddof=1) / np.sqrt(trajectories)),
        tuple(float(value) for value in array),
        trajectories,
        float(np.mean(log_ratios)),
        current_hash,
        reference_hash,
        _sequence_hash(parent),
        seed,
        start_level,
    )

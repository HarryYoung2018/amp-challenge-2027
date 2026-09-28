"""Additive research adapter for real NativeDenoiser updates and proposal paths.

No production sampler, checkpoint, acceptance flag or generator mixture is
modified. The policy is direct-logit and unconditional: context IDs are replay
provenance, not learned conditioning. Residual checkpoints requiring a C0 offset
are outside this adapter's contract. Finite anchor diagnostics are never path KL.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule, PeptideVocabulary
from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    PAD_TOKEN_INDEX,
    NativeDenoiser,
    NativeDenoiserConfig,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.replay import EndpointReplayBatch, KLSummary, summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import (
    UniformSubsetCommitKernel,
    complete_subset_commit_kl,
    positive_residue_probabilities,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True, slots=True)
class NativeEndpointConfig:
    """Predeclared bounded engineering defaults, not scientific tuned controls."""

    epsilon: float = 1e-3
    learning_rate: float = 0.01
    gradient_norm_limit: float = 1.0
    maximum_backtracks: int = 8
    microbatch_rows: int = 16
    maximum_replay_rows: int = 128
    maximum_parameters: int = 8_000_000

    def __post_init__(self) -> None:
        for key in ("epsilon", "learning_rate", "gradient_norm_limit"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.epsilon >= 1:
            raise ValueError("epsilon must be below one")
        for key, maximum in (
            ("maximum_backtracks", 8),
            ("microbatch_rows", 16),
            ("maximum_replay_rows", 128),
            ("maximum_parameters", 8_000_000),
        ):
            value = getattr(self, key)
            if (
                type(value) is not int
                or not (0 if key == "maximum_backtracks" else 1) <= value <= maximum
            ):
                raise ValueError(f"{key} exceeds the bounded adapter contract")


@dataclass(frozen=True, slots=True)
class AnchorKLLimits:
    local_mean: float = 0.01
    local_p99: float = 0.05
    reference_mean: float = 0.05
    reference_p99: float = 0.25

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value < 0
            for value in asdict(self).values()
        ):
            raise ValueError("finite-anchor limits must be finite and nonnegative")


NATIVE_ENDPOINT_DEFAULTS = NativeEndpointConfig()
ANCHOR_KL_DEFAULTS = AnchorKLLimits()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _seed(seed: int, ordinal: int, phase: str, level: int) -> np.random.Generator:
    if (
        type(seed) is not int
        or not 0 <= seed < 2**64
        or type(ordinal) is not int
        or not 0 <= ordinal < 2**64
    ):
        raise ValueError("seed/ordinal must be unsigned 64-bit integers")
    # Policy-independent draws: model hashes MUST NOT change subset randomness.
    value = int(_json_hash(["native-uniform-subset-v1", seed, ordinal, phase, level])[:32], 16)
    return np.random.Generator(np.random.PCG64DXSM(value))


def _validate_model(model: NativeDenoiser, config: NativeEndpointConfig) -> None:
    if type(model) is not NativeDenoiser:
        raise TypeError("research adapter requires the exact direct-logit NativeDenoiser")
    if sum(parameter.numel() for parameter in model.parameters()) > config.maximum_parameters:
        raise ValueError("model exceeds bounded parameter count")
    if not 8 <= model.config.min_length <= model.config.max_length <= 50:
        raise ValueError("peptide adapter supports lengths 8 through 50")
    if not 1 <= model.config.levels <= 64:
        raise ValueError("peptide adapter supports at most 64 reverse levels")
    if any(
        parameter.dtype != torch.float32 or not torch.isfinite(parameter).all()
        for parameter in model.parameters()
    ):
        raise ValueError("native parameters must be finite float32")


@contextmanager
def _evaluation_mode(model: NativeDenoiser):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@dataclass(frozen=True, slots=True)
class NativeTransitionState:
    tokens: NDArray[np.int64]
    length: int
    level: int

    def __post_init__(self) -> None:
        raw = np.asarray(self.tokens)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError("transition state tokens must be an integer vector")
        tokens = np.array(raw, dtype=np.int64, copy=True)
        if (
            type(self.length) is not int
            or not 1 <= self.length <= len(tokens)
            or type(self.level) is not int
            or self.level <= 0
        ):
            raise ValueError("invalid transition length/level")
        if np.any(
            (tokens[: self.length] < 0)
            | ((tokens[: self.length] >= 20) & (tokens[: self.length] != MASK_TOKEN_INDEX))
        ) or np.any(tokens[self.length :] != PAD_TOKEN_INDEX):
            raise ValueError("transition has invalid residue/mask/padding tokens")
        tokens.setflags(write=False)
        object.__setattr__(self, "tokens", tokens)


def _state_contract(
    state: NativeTransitionState, model_config: NativeDenoiserConfig
) -> tuple[tuple[int, ...], int]:
    if (
        len(state.tokens) != model_config.max_length
        or not model_config.min_length <= state.length <= model_config.max_length
        or not 1 <= state.level <= model_config.levels
    ):
        raise ValueError("transition state does not match native architecture")
    positions = tuple(int(value) for value in np.flatnonzero(state.tokens == MASK_TOKEN_INDEX))
    schedule = CosineMaskSchedule()
    current = int(
        schedule.mask_counts(state.length, state.level, total_levels=model_config.levels)[0]
    )
    following = int(
        schedule.mask_counts(state.length, state.level - 1, total_levels=model_config.levels)[0]
    )
    if len(positions) != current:
        raise ValueError("state mask count differs from fixed-count native schedule")
    return positions, current - following


def native_transition_kernel(
    model: NativeDenoiser,
    state: NativeTransitionState,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> UniformSubsetCommitKernel:
    """Single-state convenience wrapper around the bounded batch implementation."""
    return native_transition_kernels(model, (state,), config)[0]


def _validated_transition_kernels(
    model: NativeDenoiser,
    states: tuple[NativeTransitionState, ...],
    config: NativeEndpointConfig,
) -> tuple[UniformSubsetCommitKernel, ...]:
    """Internal forward path: enclosing operation authenticates model once."""
    contracts = [_state_contract(state, model.config) for state in states]
    result: list[UniformSubsetCommitKernel | None] = [None] * len(states)
    active = []
    for index, (positions, count) in enumerate(contracts):
        if count == 0:
            result[index] = UniformSubsetCommitKernel(positions, 0, np.empty((len(positions), 0)))
        else:
            active.append(index)
    device = next(model.parameters()).device
    with _evaluation_mode(model), torch.no_grad():
        for begin in range(0, len(active), config.microbatch_rows):
            indices = active[begin : begin + config.microbatch_rows]
            chunk = [states[index] for index in indices]
            lengths = torch.tensor([state.length for state in chunk], device=device)
            logits = (
                model(
                    torch.tensor(
                        np.stack([state.tokens for state in chunk]), dtype=torch.long, device=device
                    ),
                    torch.arange(model.config.max_length, device=device)[None, :]
                    < lengths[:, None],
                    torch.tensor([state.level for state in chunk], device=device),
                    lengths,
                )
                .double()
                .cpu()
                .numpy()
            )
            for row, index in enumerate(indices):
                positions, count = contracts[index]
                probabilities = positive_residue_probabilities(
                    logits[row, list(positions)], config.epsilon
                )
                result[index] = UniformSubsetCommitKernel(positions, count, probabilities)
    if any(kernel is None for kernel in result):
        raise RuntimeError("incomplete batched native kernel")
    return tuple(result)


def native_transition_kernels(
    model: NativeDenoiser,
    states: tuple[NativeTransitionState, ...],
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> tuple[UniformSubsetCommitKernel, ...]:
    """Bounded ≤128 states, ≤16 per forward; no neural call for k=0 rows."""
    states = tuple(states)
    if not 1 <= len(states) <= config.maximum_replay_rows:
        raise ValueError("native inference batch exceeds the row budget")
    _validate_model(model, config)
    return _validated_transition_kernels(model, states, config)


def _replay_hash(batch: EndpointReplayBatch) -> str:
    return _json_hash(
        {
            "sequences": batch.sequences,
            "context_id": batch.context_id,
            "alphabet": batch.alphabet,
            "clean": batch.clean_tokens.tolist(),
            "corrupted": batch.corrupted_tokens.tolist(),
            "attention": batch.attention_mask.tolist(),
            "corruption": batch.corruption_mask.tolist(),
            "weights": batch.weights.tolist(),
            "timestep": batch.timestep,
        }
    )


@dataclass(frozen=True, slots=True)
class NativeReplayConstruction:
    batch: EndpointReplayBatch
    source_replay_sha256: str
    native_replay_sha256: str
    level: int
    seed: int


def fixed_count_native_replay(
    batch: EndpointReplayBatch, model_config: NativeDenoiserConfig, *, level: int, seed: int
) -> NativeReplayConstruction:
    """EXPLICIT new corruption, preserving endpoints/weights/context, not the old mask."""
    if (
        batch.alphabet != ALPHABET
        or type(level) is not int
        or not 1 <= level <= model_config.levels
    ):
        raise ValueError("native replay requires canonical alphabet and exact discrete level")
    length = len(batch.sequences[0])
    if not model_config.min_length <= length <= model_config.max_length:
        raise ValueError("replay endpoints exceed model length bounds")
    encoded = PeptideVocabulary().encode(batch.sequences, max_length=model_config.max_length)
    count = int(
        CosineMaskSchedule().mask_counts(length, level, total_levels=model_config.levels)[0]
    )
    selected = _seed(seed, 0, "common-native-replay-mask", level).choice(
        length, size=count, replace=False
    )
    mask = np.zeros_like(encoded.attention_mask)
    mask[:, selected] = True
    corrupted = encoded.tokens.copy()
    corrupted[mask] = MASK_TOKEN_INDEX
    result = EndpointReplayBatch(
        batch.sequences,
        batch.context_id,
        ALPHABET,
        encoded.tokens,
        corrupted,
        encoded.attention_mask,
        mask,
        batch.weights,
        level / model_config.levels,
    )
    return NativeReplayConstruction(result, _replay_hash(batch), _replay_hash(result), level, seed)


def _native_replay_states(
    batch: EndpointReplayBatch, model_config: NativeDenoiserConfig, config: NativeEndpointConfig
) -> tuple[NativeTransitionState, ...]:
    if batch.alphabet != ALPHABET or len(batch.sequences) > config.maximum_replay_rows:
        raise ValueError("native replay vocabulary/row budget mismatch")
    real_level = batch.timestep * model_config.levels
    level = round(real_level)
    if not math.isclose(real_level, level, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            "native replay timestep must be an exact discrete level; rebuild explicitly"
        )
    states = tuple(
        NativeTransitionState(batch.corrupted_tokens[index], len(sequence), level)
        for index, sequence in enumerate(batch.sequences)
    )
    for state in states:
        _state_contract(state, model_config)
    return states


def mixture_endpoint_row_losses(
    logits: torch.Tensor,
    batch: EndpointReplayBatch,
    *,
    epsilon: float,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    """Masked NLL of the SAME epsilon-mixture used by the sampling kernel."""
    if not 0 < epsilon < 1 or isinstance(epsilon, bool):
        raise ValueError("invalid mixture epsilon")
    stop = len(batch.sequences) if stop is None else stop
    device = logits.device
    selected = torch.tensor(batch.corruption_mask[start:stop].copy(), device=device)
    targets = torch.tensor(batch.clean_tokens[start:stop].copy(), dtype=torch.long, device=device)
    if logits.shape != (*selected.shape, 20) or not torch.isfinite(logits).all():
        raise ValueError("endpoint logit shape/finite mismatch")
    log_softmax = torch.log_softmax(logits.double(), dim=-1)
    log_probabilities = torch.logaddexp(
        log_softmax + math.log1p(-epsilon), torch.full_like(log_softmax, math.log(epsilon / 20))
    )
    safe_targets = torch.where(selected, targets, 0)
    token_nll = -log_probabilities.gather(-1, safe_targets[..., None]).squeeze(-1)
    return (token_nll * selected).sum(dim=1) / selected.sum(dim=1)


@dataclass(frozen=True, slots=True)
class NativeEndpointDirection:
    base_model_sha256: str
    replay_sha256: str
    context_id: str
    config: NativeEndpointConfig
    objective_before: float
    gradient_norm_before_clip: float
    gradients: tuple[tuple[str, torch.Tensor], ...]


def propose_endpoint_direction(
    model: NativeDenoiser,
    batch: EndpointReplayBatch,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> NativeEndpointDirection:
    """One weighted, globally clipped real neural gradient; no KL decision here."""
    _validate_model(model, config)
    states = _native_replay_states(batch, model.config, config)
    base_hash = canonical_model_logical_hash(model)
    work = copy.deepcopy(model).eval()
    work.zero_grad(set_to_none=True)
    device = next(work.parameters()).device
    objective = 0.0
    for start in range(0, len(states), config.microbatch_rows):
        stop = min(start + config.microbatch_rows, len(states))
        logits = work(
            torch.tensor(
                batch.corrupted_tokens[start:stop].copy(), dtype=torch.long, device=device
            ),
            torch.tensor(batch.attention_mask[start:stop].copy(), device=device),
            torch.full((stop - start,), states[0].level, dtype=torch.long, device=device),
            torch.tensor([state.length for state in states[start:stop]], device=device),
        )
        row_losses = mixture_endpoint_row_losses(
            logits, batch, epsilon=config.epsilon, start=start, stop=stop
        )
        weights = torch.tensor(batch.weights[start:stop].copy(), dtype=torch.float64, device=device)
        loss = (weights * row_losses).sum()  # global weights, NO microbatch renormalization
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite endpoint objective")
        loss.backward()
        objective += float(loss.detach().cpu())
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        work.parameters(), config.gradient_norm_limit, error_if_nonfinite=True
    )
    gradients = tuple(
        (name, parameter.grad.detach().cpu().clone())
        for name, parameter in work.named_parameters()
        if parameter.grad is not None
    )
    if not gradients or not any(torch.count_nonzero(value) for _, value in gradients):
        raise ValueError("endpoint replay produced no neural update direction")
    if canonical_model_logical_hash(model) != base_hash:
        raise RuntimeError("caller model changed during gradient proposal")
    return NativeEndpointDirection(
        base_hash,
        _replay_hash(batch),
        batch.context_id,
        config,
        objective,
        float(gradient_norm.cpu()),
        gradients,
    )


def endpoint_candidate(
    model: NativeDenoiser, direction: NativeEndpointDirection, *, backtracks: int = 0
) -> NativeDenoiser:
    """Construct a bounded SGD candidate without modifying caller state."""
    if canonical_model_logical_hash(model) != direction.base_model_sha256:
        raise ValueError("endpoint direction belongs to a different base model")
    if type(backtracks) is not int or not 0 <= backtracks <= direction.config.maximum_backtracks:
        raise ValueError("backtracking budget exceeded")
    candidate = copy.deepcopy(model).eval()
    gradients = dict(direction.gradients)
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            if name in gradients:
                gradient = gradients[name]
                if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
                    raise ValueError("invalid neural gradient payload")
                parameter.add_(
                    gradient.to(parameter.device),
                    alpha=-direction.config.learning_rate * 0.5**backtracks,
                )
    _validate_model(candidate, direction.config)
    return candidate


@dataclass(frozen=True, slots=True)
class NativeAnchorDiagnostics:
    local_old_candidate: KLSummary
    frozen_reference_candidate_reference: KLSummary
    probe_replay_sha256: str
    diagnostic_scope: str = "finite_fixed_anchor_transition_kl_only_not_path_or_global_kl"


def anchor_diagnostics(
    old: NativeDenoiser,
    candidate: NativeDenoiser,
    reference: NativeDenoiser,
    probes: EndpointReplayBatch,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> NativeAnchorDiagnostics:
    if old.config != candidate.config or old.config != reference.config:
        raise ValueError("paired native policies must share architecture and transition schedule")
    states = _native_replay_states(probes, old.config, config)
    if not any(_state_contract(state, old.config)[1] > 0 for state in states):
        raise ValueError("anchor set must include an active scheduled transition")
    local, frozen = [], []
    for old_kernel, candidate_kernel, reference_kernel in zip(
        native_transition_kernels(old, states, config),
        native_transition_kernels(candidate, states, config),
        native_transition_kernels(reference, states, config),
        strict=True,
    ):
        local.append(complete_subset_commit_kl(old_kernel, candidate_kernel))
        frozen.append(complete_subset_commit_kl(candidate_kernel, reference_kernel))
    return NativeAnchorDiagnostics(
        summarize_kl(local, weights=probes.weights),
        summarize_kl(frozen, weights=probes.weights),
        _replay_hash(probes),
    )


@dataclass(frozen=True, slots=True)
class NativeEndpointUpdate:
    accepted: bool
    model: NativeDenoiser | None
    direction: NativeEndpointDirection
    diagnostics: NativeAnchorDiagnostics
    backtracks: int
    enforcement_enabled: bool
    reference_model_sha256: str
    scientific_evidence_accepted: bool = False
    production_input_eligible: bool = False


def select_endpoint_update(
    model: NativeDenoiser,
    reference: NativeDenoiser,
    direction: NativeEndpointDirection,
    probes: EndpointReplayBatch,
    *,
    limits: AnchorKLLimits = ANCHOR_KL_DEFAULTS,
    enforce_kl: bool = True,
) -> NativeEndpointUpdate:
    """Separate finite-probe enforcement; no-KL uses the identical first proposal."""
    if type(enforce_kl) is not bool:
        raise TypeError("enforce_kl must be boolean")
    if probes.context_id != direction.context_id:
        raise ValueError("probe and endpoint context metadata differ")
    frozen_hash = canonical_model_logical_hash(reference)
    last = None
    for backtracks in range(direction.config.maximum_backtracks + 1):
        candidate = endpoint_candidate(model, direction, backtracks=backtracks)
        diagnostics = anchor_diagnostics(model, candidate, reference, probes, direction.config)
        local, frozen = (
            diagnostics.local_old_candidate,
            diagnostics.frozen_reference_candidate_reference,
        )
        accepted = not enforce_kl or (
            local.mean <= limits.local_mean
            and local.p99 <= limits.local_p99
            and frozen.mean <= limits.reference_mean
            and frozen.p99 <= limits.reference_p99
        )
        if (
            canonical_model_logical_hash(reference) != frozen_hash
            or canonical_model_logical_hash(model) != direction.base_model_sha256
        ):
            raise RuntimeError("caller/frozen-reference state changed during selection")
        last = NativeEndpointUpdate(
            accepted,
            candidate if accepted else None,
            direction,
            diagnostics,
            backtracks,
            enforce_kl,
            frozen_hash,
        )
        if accepted:
            return last
    if last is None:
        raise RuntimeError("no candidate evaluated")
    return last

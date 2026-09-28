"""Real mixed-length native denoising; separate from lineage replay records."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule, PeptideVocabulary
from amp_challenge.generators.diffusion.model import (
    MASK_TOKEN_INDEX,
    NativeDenoiser,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeEndpointConfig,
    NativeEndpointDirection,
    NativeTransitionState,
    _json_hash,
    _seed,
    _state_contract,
    _validate_model,
    native_transition_kernels,
)
from amp_challenge.generators.diffusion.replay import KLSummary, summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl


@dataclass(frozen=True, slots=True)
class NativeWeightedReplay:
    sequences: tuple[str, ...]
    context_id: str
    states: tuple[NativeTransitionState, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.sequences) is not tuple
            or not 1 <= len(self.sequences) <= 128
            or len(set(self.sequences)) != len(self.sequences)
            or type(self.states) is not tuple
            or len(self.states) != len(self.sequences)
            or not isinstance(self.context_id, str)
            or not self.context_id
        ):
            raise ValueError("mixed native replay inventory/context differs")
        if any(type(state) is not NativeTransitionState for state in self.states):
            raise ValueError("mixed native replay state type differs")
        if any(
            type(seq) is not str
            or not 8 <= len(seq) <= 50
            or set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            for seq in self.sequences
        ):
            raise ValueError("mixed native replay requires canonical bounded peptide sequences")
        widths = {len(state.tokens) for state in self.states}
        if len(widths) != 1:
            raise ValueError("mixed native replay widths differ")
        encoded = PeptideVocabulary().encode(self.sequences, max_length=next(iter(widths)))
        for index, state in enumerate(self.states):
            mask = state.tokens == MASK_TOKEN_INDEX
            expected = encoded.tokens[index].copy()
            expected[mask] = MASK_TOKEN_INDEX
            if (
                state.length != len(self.sequences[index])
                or not mask.any()
                or not np.array_equal(expected, state.tokens)
            ):
                raise ValueError("mixed native replay does not encode recorded sequences/masks")
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if (
            weights.shape != (len(self.sequences),)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=1e-12)
        ):
            raise ValueError("mixed native weights must be positive and globally normalized")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    @property
    def sha256(self) -> str:
        return _json_hash(
            {
                "kind": "native-mixed-weighted-replay-v1",
                "sequences": self.sequences,
                "context": self.context_id,
                "weights": self.weights.tolist(),
                "states": [
                    (state.tokens.tolist(), state.length, state.level) for state in self.states
                ],
            }
        )


def build_weighted_replay(
    model: NativeDenoiser,
    sequences: tuple[str, ...],
    weights: np.ndarray,
    *,
    context_id: str,
    seed: int,
    ordinal: int,
    active_probes: bool = False,
) -> NativeWeightedReplay:
    """Fixed-count random corruption, semantic sequence keys, no transport IDs."""
    if not 1 <= len(sequences) <= 128:
        raise ValueError("mixed native replay exceeds row budget")
    if type(active_probes) is not bool or not 1 <= model.config.levels <= 64:
        raise ValueError("mixed native corruption mode/level budget differs")
    encoded = PeptideVocabulary().encode(sequences, max_length=model.config.max_length)
    states = []
    for row, sequence in enumerate(sequences):
        if not model.config.min_length <= len(sequence) <= model.config.max_length:
            raise ValueError("mixed native sequence length outside model support")
        rng = _seed(seed, ordinal, "weighted-replay-" + sequence, 0)
        counts = CosineMaskSchedule().mask_counts(
            np.full(model.config.levels + 1, len(sequence), dtype=np.int64),
            np.arange(model.config.levels + 1),
            total_levels=model.config.levels,
        )
        levels = [
            level
            for level in range(1, model.config.levels + 1)
            if not active_probes or counts[level] > counts[level - 1]
        ]
        level = int(rng.choice(levels))
        positions = rng.choice(len(sequence), size=int(counts[level]), replace=False)
        tokens = encoded.tokens[row].copy()
        tokens[positions] = MASK_TOKEN_INDEX
        states.append(NativeTransitionState(tokens, len(sequence), level))
    return NativeWeightedReplay(sequences, context_id, tuple(states), weights)


def propose_weighted_direction(
    model: NativeDenoiser,
    replay: NativeWeightedReplay,
    config: NativeEndpointConfig = NATIVE_ENDPOINT_DEFAULTS,
) -> NativeEndpointDirection:
    """One global epsilon-mixture NLL gradient, not per-length clipped gradients."""
    _validate_model(model, config)
    if len(replay.sequences) > config.maximum_replay_rows:
        raise ValueError("weighted native replay exceeds configured row budget")
    for state in replay.states:
        _state_contract(state, model.config)
    before = canonical_model_logical_hash(model)
    work = copy.deepcopy(model).eval()
    work.zero_grad(set_to_none=True)
    device = next(work.parameters()).device
    clean = PeptideVocabulary().encode(replay.sequences, max_length=model.config.max_length).tokens
    objective = 0.0
    for start in range(0, len(replay.states), config.microbatch_rows):
        stop = min(start + config.microbatch_rows, len(replay.states))
        chunk = replay.states[start:stop]
        tokens = torch.tensor(np.stack([state.tokens for state in chunk]), device=device)
        lengths = torch.tensor([state.length for state in chunk], device=device)
        logits = work(
            tokens,
            torch.arange(model.config.max_length, device=device)[None, :] < lengths[:, None],
            torch.tensor([state.level for state in chunk], device=device),
            lengths,
        )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite mixed native logits")
        logp = torch.logaddexp(
            torch.log_softmax(logits.double(), dim=-1) + math.log1p(-config.epsilon),
            torch.full_like(logits.double(), math.log(config.epsilon / 20)),
        )
        mask = tokens == MASK_TOKEN_INDEX
        targets = torch.tensor(clean[start:stop].copy(), device=device)
        safe_targets = torch.where(mask, targets, 0)
        losses = -logp.gather(-1, safe_targets[..., None]).squeeze(-1)
        row_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1)
        weighted = (
            row_losses * torch.tensor(replay.weights[start:stop].copy(), device=device)
        ).sum()
        if not torch.isfinite(weighted):
            raise FloatingPointError("nonfinite mixed native objective")
        weighted.backward()
        objective += float(weighted.detach().cpu())
    norm = torch.nn.utils.clip_grad_norm_(
        work.parameters(), config.gradient_norm_limit, error_if_nonfinite=True
    )
    gradients = tuple(
        (name, parameter.grad.detach().cpu().clone())
        for name, parameter in work.named_parameters()
        if parameter.grad is not None
    )
    if not gradients or not any(torch.count_nonzero(value) for _, value in gradients):
        raise ValueError("mixed native replay produced no neural direction")
    if canonical_model_logical_hash(model) != before:
        raise RuntimeError("caller model changed during mixed native update")
    return NativeEndpointDirection(
        before, replay.sha256, replay.context_id, config, objective, float(norm.cpu()), gradients
    )


@dataclass(frozen=True, slots=True)
class WeightedAnchorDiagnostics:
    local_old_candidate: KLSummary
    candidate_frozen_reference: KLSummary
    replay_sha256: str
    scope: str = "finite_active_anchor_transition_only_not_path_or_global_kl"


def weighted_anchor_diagnostics(
    old: NativeDenoiser,
    candidate: NativeDenoiser,
    reference: NativeDenoiser,
    probes: NativeWeightedReplay,
    config: NativeEndpointConfig,
) -> WeightedAnchorDiagnostics:
    if old.config != candidate.config or old.config != reference.config:
        raise ValueError("weighted native policies do not share architecture")
    if not all(_state_contract(state, old.config)[1] > 0 for state in probes.states):
        raise ValueError("weighted anchor diagnostics require active transition probes")
    local, frozen = [], []
    for old_kernel, candidate_kernel, reference_kernel in zip(
        native_transition_kernels(old, probes.states, config),
        native_transition_kernels(candidate, probes.states, config),
        native_transition_kernels(reference, probes.states, config),
        strict=True,
    ):
        local.append(complete_subset_commit_kl(old_kernel, candidate_kernel))
        frozen.append(complete_subset_commit_kl(candidate_kernel, reference_kernel))
    return WeightedAnchorDiagnostics(
        summarize_kl(local, weights=probes.weights),
        summarize_kl(frozen, weights=probes.weights),
        probes.sha256,
    )

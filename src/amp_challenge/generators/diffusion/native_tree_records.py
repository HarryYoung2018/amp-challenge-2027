"""Bounded tree-search records; posterior scores are not observed oracle truth."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass

import numpy as np

from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    NativePosteriorScore,
)
from amp_challenge.generators.diffusion.subset_kernel import SubsetCommitDraw

TR2D2_CONFIG_SHA256 = "a27fa602333f3370fc631c7bdf45a3ce109aeb8ddd76483dbe5980cbcebdabd4"
MAXIMUM_GENERATION_BYTES = 64 * 1024 * 1024


def _fresh_record_fields(value):
    """Expose current fields to JSON without recursively copying the record tree."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True, slots=True)
class TreeRoot:
    triple: str
    length: int
    length_log_probability: float
    generator_anchor_sha256: str
    behavior_sha256: str
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class TreeTransition:
    level: int
    rng_ordinal: int
    before_tokens_sha256: str
    after_tokens_sha256: str
    draw: SubsetCommitDraw
    reference_log_probability: float


@dataclass(frozen=True, slots=True)
class TreePath:
    endpoint: str
    transitions: tuple[TreeTransition, ...]
    behavior_log_probability: float
    reference_log_probability: float
    scope: str = "conditional_checkpoint_length_model_path_not_whole_search_or_endpoint_marginal"


@dataclass(frozen=True, slots=True)
class TreeSelection:
    node_id: int
    eligible_children: tuple[int, ...]
    puct_scores: tuple[tuple[float, float], ...]
    pareto_children: tuple[int, ...]
    selected_child: int
    conditional_log_probability: float


@dataclass(frozen=True, slots=True)
class TreeExpansion:
    expansion_index: int
    triple: str
    iteration: int
    selected_node: int
    selections: tuple[TreeSelection, ...]
    attempt_indices: tuple[int, ...]
    posterior_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class TreeAttempt:
    attempt_index: int
    expansion_index: int
    child_node: int
    child_prefix_steps: int
    path: TreePath
    posterior: NativePosteriorScore
    rejection_reason: str | None
    first_attempt_index: int
    buffer_after_attempt_indices: tuple[int, ...]
    conditional_retention_probability: float


@dataclass(frozen=True, slots=True)
class TreeReplayBuffer:
    triple: str
    attempt_indices: tuple[int, ...]
    log_weights: tuple[float, ...]
    normalized_weights: tuple[float, ...]
    effective_sample_size: float
    maximum_weight: float
    weight_diagnostics_pass_protocol_thresholds: bool
    scope: str = "tree_guided_offpolicy_search_distillation_not_exact_importance_sampling"


@dataclass(frozen=True, slots=True)
class NativeTreeGeneration:
    round_index: int
    seed: int
    history_sha256: str
    evaluator_binding: FrozenNativePosteriorBinding
    roots: tuple[TreeRoot, ...]
    expansions: tuple[TreeExpansion, ...]
    attempts: tuple[TreeAttempt, ...]
    replay_buffers: tuple[TreeReplayBuffer, ...]
    shortlisted_sequences: tuple[str, ...]
    status: str
    configuration_sha256: str = TR2D2_CONFIG_SHA256
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self,
                default=_fresh_record_fields,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def check_output_budget(self) -> None:
        import json

        if len(json.dumps(asdict(self), allow_nan=False).encode()) > MAXIMUM_GENERATION_BYTES:
            raise ValueError("tree generation exceeds serialized output budget")


def pareto_indices(values: tuple[tuple[float, float], ...]) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(values)
        if not any(
            all(a >= b for a, b in zip(other, value, strict=True))
            and any(a > b for a, b in zip(other, value, strict=True))
            for other in values
        )
    )


def normalized_tree_weights(log_weights: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
    if not 1 <= len(log_weights) <= 20 or not all(map(math.isfinite, log_weights)):
        raise ValueError("tree replay requires bounded finite logweights")
    values = np.asarray(log_weights, dtype=np.float64)
    weights = np.exp(np.clip(values - values.max(), -10.0, 0.0))
    weights /= weights.sum()
    return tuple(map(float, weights)), float(1 / np.dot(weights, weights))

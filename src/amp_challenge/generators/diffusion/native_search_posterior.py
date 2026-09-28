"""Shared externally frozen posterior seam, not an oracle authentication layer.

The outer controller authenticates/fits the actual posterior and its code/features.
Feasibility means only its fixed context-supplied cheap checks, never an imputed
HC50 endpoint or hidden oracle truth. All bindings/receipts are provenance, not
random seeds. Numeric search may depend on returned means, but transport IDs must
not affect random choices. A matching digest cannot prove a callable's behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from amp_challenge.generators.diffusion.native_baseline_operators import (
    NormalizedObjectiveContext,
    sequence_id,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string

REPRESENTATION = "esm320_plus_normalized_length"


@dataclass(frozen=True, slots=True)
class FrozenNativePosteriorBinding:
    history_sha256: str
    objective_context_sha256: str
    posterior_sha256: str
    feature_source_sha256: str
    evaluator_source_sha256: str
    representation: str = REPRESENTATION
    prediction_kind: str = "frozen_generation_vector_posterior_mean_not_thompson"

    def __post_init__(self) -> None:
        if any(
            not hash_string(value)
            for value in (
                self.history_sha256,
                self.objective_context_sha256,
                self.posterior_sha256,
                self.feature_source_sha256,
                self.evaluator_source_sha256,
            )
        ) or (
            self.representation != REPRESENTATION
            or self.prediction_kind != "frozen_generation_vector_posterior_mean_not_thompson"
        ):
            raise ValueError("native search requires a fixed non-spectral posterior identity")


@dataclass(frozen=True, slots=True)
class NativePosteriorScore:
    objectives: tuple[float, float]
    feasible: bool

    def validate(self, context: NormalizedObjectiveContext) -> None:
        context.scalarize(self.objectives)
        if type(self.feasible) is not bool:
            raise ValueError("native posterior feasibility must be explicit")


@dataclass(frozen=True, slots=True)
class NativePosteriorBatch:
    sequence_ids: tuple[str, ...]
    scores: tuple[NativePosteriorScore, ...]
    receipt_sha256: str


class FrozenNativePosterior(Protocol):
    binding: FrozenNativePosteriorBinding

    def evaluate(self, sequences: tuple[str, ...]) -> NativePosteriorBatch: ...


def read_native_posterior(
    evaluator: FrozenNativePosterior,
    sequences: tuple[str, ...],
    *,
    expected_binding: FrozenNativePosteriorBinding,
    context: NormalizedObjectiveContext,
) -> NativePosteriorBatch:
    """Check exact frozen identity and returned order around a bounded callback.

    The caller must construct expected_binding from its fixed code/feature pins
    and current verified history, not opportunistically trust callback metadata.
    Duplicate sequences are allowed as independently logged proposal attempts.
    """
    if type(expected_binding) is not FrozenNativePosteriorBinding:
        raise TypeError("native posterior expected binding differs")
    expected_binding.__post_init__()
    if type(context) is not NormalizedObjectiveContext:
        raise TypeError("native posterior objective context differs")
    context.__post_init__()
    if expected_binding.objective_context_sha256 != context.context_sha256:
        raise ValueError("native posterior objective context identity differs")
    if (
        type(sequences) is not tuple
        or not 1 <= len(sequences) <= 128
        or any(
            type(seq) is not str
            or not 8 <= len(seq) <= 50
            or set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            for seq in sequences
        )
    ):
        raise ValueError("native posterior needs bounded ordered canonical peptide inputs")
    if type(evaluator.binding) is not FrozenNativePosteriorBinding or (
        evaluator.binding != expected_binding
    ):
        raise ValueError("native posterior provider binding differs before evaluation")
    batch = evaluator.evaluate(sequences)
    if evaluator.binding != expected_binding:
        raise ValueError("native posterior provider changed during evaluation")
    if (
        type(batch) is not NativePosteriorBatch
        or type(batch.sequence_ids) is not tuple
        or batch.sequence_ids != tuple(map(sequence_id, sequences))
        or type(batch.scores) is not tuple
        or len(batch.scores) != len(sequences)
        or not hash_string(batch.receipt_sha256)
    ):
        raise ValueError("native posterior returned inventory/order/receipt differs")
    for score in batch.scores:
        if type(score) is not NativePosteriorScore:
            raise TypeError("native posterior score record differs")
        score.validate(context)
    return batch

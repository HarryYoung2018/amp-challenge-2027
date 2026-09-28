"""Outcome-free global pool commitment stage for sequential-v2.

The public functions accept only typed candidate views and rotation identities.
There is deliberately no argument for source examples, labels, reveal stores,
models, or paths containing outcomes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from amp_challenge.acquisition.sequential_v2_selector import (
    RANDOM,
    GuardEvaluation,
    PoolCandidate,
    RandomPoolCandidate,
    RepairStep,
    SelectedSeat,
    SelectionResult,
    select_pool_policy,
    select_random,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    CEILING,
    EXPECTED_POLICY_RUNS,
    EXPECTED_POOL_COMMITTED_ASSOCIATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
    NO_QUERY,
    PolicyRunSpec,
    RotationSpec,
    ordered_rotations,
    policy_runs_for_rotation,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREDICTION_POLICIES = (
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
)


def ordered_id_stream_sha256(values: Sequence[str]) -> str:
    """Hash an order-sensitive unique ID stream, permitting the empty stream."""

    identifiers = tuple(values)
    if len(set(identifiers)) != len(identifiers) or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in identifiers
    ):
        raise ValueError("ordered ID stream must contain unique lowercase SHA-256 values")
    return hashlib.sha256(
        "".join(f"{value}\n" for value in identifiers).encode("ascii")
    ).hexdigest()


def _hex(value: float | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("selection result float must be finite binary64")
    return value.hex()


def _seat_document(seat: SelectedSeat) -> dict[str, object]:
    score = seat.acquisition_score
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise ValueError("selection acquisition score must be an integer or finite float")
    if isinstance(score, float):
        if not math.isfinite(score):
            raise ValueError("selection acquisition score must be finite")
        score_kind = "float_hex"
        score_value: object = score.hex()
    else:
        score_kind = "unsigned_integer"
        if score < 0:
            raise ValueError("integer acquisition priority must be nonnegative")
        score_value = score
    return {
        "sequence_id": seat.sequence_id,
        "requested_role": seat.requested_role,
        "applied_role": seat.applied_role,
        "acquisition_score_kind": score_kind,
        "acquisition_score": score_value,
        "scalar_mean_hex": _hex(seat.scalar_mean),
    }


def _repair_document(step: RepairStep) -> dict[str, object]:
    return {
        "step_number": step.step_number,
        "requested_role": step.requested_role,
        "removed_sequence_id": step.removed_sequence_id,
        "added_sequence_id": step.added_sequence_id,
    }


def _guard_document(evaluation: GuardEvaluation) -> dict[str, object]:
    return {
        "stage": evaluation.stage,
        "sequence_ids": list(evaluation.sequence_ids),
        "objective_means_hex": [value.hex() for value in evaluation.objective_means],
        "scalar_mean_hex": evaluation.scalar_mean.hex(),
        "scalar_loss_hex": evaluation.scalar_loss.hex(),
        "objective_losses_hex": [value.hex() for value in evaluation.objective_losses],
        "passed": evaluation.passed,
    }


def selection_result_document(result: SelectionResult) -> dict[str, object]:
    """Serialize every selector field with exact binary64 encodings."""

    if not isinstance(result, SelectionResult):
        raise TypeError("selection result must be a SelectionResult")
    return {
        "schema_version": 1,
        "rotation_id": result.rotation_id,
        "policy": result.policy,
        "seats": [_seat_document(seat) for seat in result.seats],
        "mean_control_sequence_ids": list(result.mean_control_sequence_ids),
        "requested_sequence_ids": list(result.requested_sequence_ids),
        "requested_complete": result.requested_complete,
        "requested_role_counts": dict(sorted(result.requested_role_counts.items())),
        "applied_role_counts": dict(sorted(result.applied_role_counts.items())),
        "component_counts": dict(sorted(result.component_counts.items())),
        "scalar_loss_hex": _hex(result.scalar_loss),
        "objective_losses_hex": (
            None
            if result.objective_losses is None
            else [value.hex() for value in result.objective_losses]
        ),
        "repair_trace": [_repair_document(step) for step in result.repair_trace],
        "guard_evaluations": [
            _guard_document(evaluation) for evaluation in result.guard_evaluations
        ],
        "fallback_reason": result.fallback_reason,
    }


@dataclass(frozen=True, slots=True)
class InputViewBinding:
    """Content identity of the only capsule visible to one selector worker."""

    kind: str
    phase_seal_sha256: str
    payload_sha256: str | None
    candidate_count: int
    candidate_ids_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"none", "prediction", "random_minimal", "ceiling_minimal"}:
            raise ValueError("input-view kind is outside the frozen selector capabilities")
        if (
            not isinstance(self.phase_seal_sha256, str)
            or _SHA256.fullmatch(self.phase_seal_sha256) is None
        ):
            raise ValueError("input-view phase seal must be a lowercase SHA-256")
        if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int):
            raise ValueError("input-view candidate census must be an exact integer")
        if self.kind == "none":
            if (
                self.payload_sha256 is not None
                or self.candidate_count != 0
                or self.candidate_ids_sha256 != ordered_id_stream_sha256(())
            ):
                raise ValueError("no-query input binding cannot contain a candidate payload")
        elif (
            not isinstance(self.payload_sha256, str)
            or _SHA256.fullmatch(self.payload_sha256) is None
            or self.candidate_count < 1
        ):
            raise ValueError("selector input binding lacks a valid payload or candidate census")
        if (
            not isinstance(self.candidate_ids_sha256, str)
            or _SHA256.fullmatch(self.candidate_ids_sha256) is None
        ):
            raise ValueError("input-view candidate-ID digest must be a lowercase SHA-256")

    @classmethod
    def for_candidates(
        cls,
        *,
        kind: str,
        phase_seal_sha256: str,
        payload_sha256: str,
        candidate_ids: Sequence[str],
    ) -> InputViewBinding:
        identifiers = tuple(candidate_ids)
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError("input-view candidate IDs must use ascending canonical order")
        return cls(
            kind=kind,
            phase_seal_sha256=phase_seal_sha256,
            payload_sha256=payload_sha256,
            candidate_count=len(identifiers),
            candidate_ids_sha256=ordered_id_stream_sha256(identifiers),
        )

    def document(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "phase_seal_sha256": self.phase_seal_sha256,
            "payload_sha256": self.payload_sha256,
            "candidate_count": self.candidate_count,
            "candidate_ids_sha256": self.candidate_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class RotationViewProvenance:
    """Authenticated prediction/minimal-view identities for one rotation."""

    spec: RotationSpec
    candidate_count: int
    candidate_ids_sha256: str
    prediction_view_seal_sha256: str
    prediction_view_payload_sha256: str
    random_view_seal_sha256: str
    random_view_payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RotationSpec):
            raise TypeError("view provenance spec must be a RotationSpec")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count != EXPECTED_SUPPORT_BY_FOLD[self.spec.pool_fold]
        ):
            raise ValueError("view provenance candidate census differs from accepted support")
        for label, value in (
            ("candidate-ID digest", self.candidate_ids_sha256),
            ("prediction-view seal", self.prediction_view_seal_sha256),
            ("prediction-view payload", self.prediction_view_payload_sha256),
            ("random-view seal", self.random_view_seal_sha256),
            ("random-view payload", self.random_view_payload_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"view provenance {label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class PoolCommitment:
    """One sealed policy-run commitment produced before any pool reveal."""

    run: PolicyRunSpec
    input_view: InputViewBinding
    selected_sequence_ids: tuple[str, ...]
    selection_result: SelectionResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.run, PolicyRunSpec):
            raise TypeError("pool commitment run must be a PolicyRunSpec")
        if not isinstance(self.input_view, InputViewBinding):
            raise TypeError("pool commitment must bind one typed input view")
        expected_view_kind = (
            "none"
            if self.run.policy == NO_QUERY
            else "random_minimal"
            if self.run.policy == RANDOM
            else "ceiling_minimal"
            if self.run.policy == CEILING
            else "prediction"
        )
        if self.input_view.kind != expected_view_kind:
            raise ValueError("policy run and input-view capability kind disagree")
        if (
            self.run.policy != NO_QUERY
            and self.input_view.candidate_count
            != (EXPECTED_SUPPORT_BY_FOLD[self.run.rotation.pool_fold])
        ):
            raise ValueError("input-view candidate census differs from accepted fold support")
        if not isinstance(self.selected_sequence_ids, tuple):
            raise TypeError("pool commitment selected_sequence_ids must be an immutable tuple")
        selected = self.selected_sequence_ids
        if (
            len(selected) != self.run.expected_pool_selection_count
            or len(set(selected)) != len(selected)
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in selected
            )
        ):
            raise ValueError("pool commitment has an invalid selected-sequence census or identity")
        if self.run.policy in {NO_QUERY, CEILING}:
            if self.selection_result is not None:
                raise ValueError(
                    "no-query and ceiling commitments must not invent selector results"
                )
            if self.run.policy == CEILING and selected != tuple(sorted(selected)):
                raise ValueError(
                    "ceiling commitment must use ascending canonical sequence-ID order"
                )
            if self.run.policy == CEILING and ordered_id_stream_sha256(selected) != (
                self.input_view.candidate_ids_sha256
            ):
                raise ValueError(
                    "ceiling commitment must contain the complete bound candidate-ID stream"
                )
        else:
            result = self.selection_result
            if not isinstance(result, SelectionResult):
                raise ValueError("budgeted commitment must preserve its complete selector result")
            if result.rotation_id != self.run.rotation.rotation_id:
                raise ValueError("selector result rotation differs from the policy run")
            if result.policy != self.run.policy:
                raise ValueError("selector result policy differs from the policy run")
            if result.sequence_ids != selected:
                raise ValueError("selector result and committed sequence order differ")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "track_id": self.run.track_id,
            "rotation_id": self.run.rotation.rotation_id,
            "policy": self.run.policy,
            "seed": self.run.seed,
            "selection_kind": self.run.selection_kind,
            "input_view": self.input_view.document(),
            "selected_sequence_ids": list(self.selected_sequence_ids),
            "selected_sequence_count": len(self.selected_sequence_ids),
            "selected_sequence_ids_sha256": ordered_id_stream_sha256(self.selected_sequence_ids),
            "selection_result": (
                None
                if self.selection_result is None
                else selection_result_document(self.selection_result)
            ),
        }


def _validate_prediction_view(
    spec: RotationSpec,
    pool_candidates: Sequence[PoolCandidate],
) -> tuple[PoolCandidate, ...]:
    if not isinstance(spec, RotationSpec):
        raise TypeError("selection spec must be a RotationSpec")
    pool = tuple(pool_candidates)
    if not pool or any(not isinstance(item, PoolCandidate) for item in pool):
        raise TypeError("pool view must contain only PoolCandidate instances")
    pool_ids = tuple(item.sequence_id for item in pool)
    if pool_ids != tuple(sorted(set(pool_ids))):
        raise ValueError("pool candidates must use canonical sequence-ID order")
    if len(pool) != EXPECTED_SUPPORT_BY_FOLD[spec.pool_fold]:
        raise ValueError("prediction-view census differs from accepted fold support")
    if any(item.rotation_id != spec.rotation_id for item in pool):
        raise ValueError("candidate view rotation differs from the requested selection")
    if any(not item.eligible for item in pool):
        raise ValueError("published selector views may contain only support-eligible candidates")
    return pool


def _validate_random_view(
    spec: RotationSpec,
    random_candidates: Sequence[RandomPoolCandidate],
) -> tuple[RandomPoolCandidate, ...]:
    if not isinstance(spec, RotationSpec):
        raise TypeError("selection spec must be a RotationSpec")
    random = tuple(random_candidates)
    if not random or any(not isinstance(item, RandomPoolCandidate) for item in random):
        raise TypeError("random view must contain only RandomPoolCandidate instances")
    ids = tuple(item.sequence_id for item in random)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("random candidates must use canonical sequence-ID order")
    if len(random) != EXPECTED_SUPPORT_BY_FOLD[spec.pool_fold]:
        raise ValueError("random-view census differs from accepted fold support")
    if any(item.rotation_id != spec.rotation_id for item in random):
        raise ValueError("random candidate rotation differs from the requested selection")
    if any(not item.eligible for item in random):
        raise ValueError("published random view may contain only support-eligible candidates")
    return random


def select_prediction_view(
    spec: RotationSpec,
    pool_candidates: Sequence[PoolCandidate],
    *,
    input_view_seal_sha256: str,
    input_view_payload_sha256: str,
) -> tuple[PoolCommitment, ...]:
    """Select four budgeted tracks from only the sealed prediction-bearing view."""

    pool = _validate_prediction_view(spec, pool_candidates)
    binding = InputViewBinding.for_candidates(
        kind="prediction",
        phase_seal_sha256=input_view_seal_sha256,
        payload_sha256=input_view_payload_sha256,
        candidate_ids=tuple(item.sequence_id for item in pool),
    )
    commitments: list[PoolCommitment] = []
    for run in policy_runs_for_rotation(spec):
        if run.policy not in _PREDICTION_POLICIES:
            continue
        result = select_pool_policy(
            pool,
            run.policy,
            outer_fold=spec.outer_fold,
            pool_fold=spec.pool_fold,
        )
        selected = result.sequence_ids
        commitments.append(
            PoolCommitment(
                run=run,
                input_view=binding,
                selected_sequence_ids=selected,
                selection_result=result,
            )
        )
    expected = tuple(
        run for run in policy_runs_for_rotation(spec) if run.policy in _PREDICTION_POLICIES
    )
    if tuple(commitment.run for commitment in commitments) != expected:
        raise AssertionError("rotation commitments escaped the frozen policy-run order")
    return tuple(commitments)


def select_random_view(
    spec: RotationSpec,
    random_candidates: Sequence[RandomPoolCandidate],
    *,
    input_view_seal_sha256: str,
    input_view_payload_sha256: str,
) -> tuple[PoolCommitment, ...]:
    """Select five seeds while prediction, feature, and novelty values are unavailable."""

    random = _validate_random_view(spec, random_candidates)
    binding = InputViewBinding.for_candidates(
        kind="random_minimal",
        phase_seal_sha256=input_view_seal_sha256,
        payload_sha256=input_view_payload_sha256,
        candidate_ids=tuple(item.sequence_id for item in random),
    )
    commitments: list[PoolCommitment] = []
    for run in policy_runs_for_rotation(spec):
        if run.policy != RANDOM:
            continue
        assert run.seed is not None
        result = select_random(
            random,
            outer_fold=spec.outer_fold,
            pool_fold=spec.pool_fold,
            seed=run.seed,
        )
        commitments.append(
            PoolCommitment(
                run=run,
                input_view=binding,
                selected_sequence_ids=result.sequence_ids,
                selection_result=result,
            )
        )
    expected = tuple(run for run in policy_runs_for_rotation(spec) if run.policy == RANDOM)
    if tuple(commitment.run for commitment in commitments) != expected:
        raise AssertionError("random commitments escaped the frozen seed order")
    return tuple(commitments)


def select_ceiling_view(
    spec: RotationSpec,
    candidates: Sequence[RandomPoolCandidate],
    *,
    input_view_seal_sha256: str,
    input_view_payload_sha256: str,
) -> PoolCommitment:
    """Commit the complete support set from a prediction-free minimal capsule."""

    minimal = _validate_random_view(spec, candidates)
    binding = InputViewBinding.for_candidates(
        kind="ceiling_minimal",
        phase_seal_sha256=input_view_seal_sha256,
        payload_sha256=input_view_payload_sha256,
        candidate_ids=tuple(item.sequence_id for item in minimal),
    )
    run = next(run for run in policy_runs_for_rotation(spec) if run.policy == CEILING)
    return PoolCommitment(
        run=run,
        input_view=binding,
        selected_sequence_ids=tuple(item.sequence_id for item in minimal),
        selection_result=None,
    )


def make_no_query_commitment(
    spec: RotationSpec,
    *,
    protocol_seal_sha256: str,
) -> PoolCommitment:
    """Construct the zero-selection control without opening a candidate capsule."""

    if not isinstance(spec, RotationSpec):
        raise TypeError("no-query spec must be a RotationSpec")
    run = next(run for run in policy_runs_for_rotation(spec) if run.policy == NO_QUERY)
    binding = InputViewBinding(
        kind="none",
        phase_seal_sha256=protocol_seal_sha256,
        payload_sha256=None,
        candidate_count=0,
        candidate_ids_sha256=ordered_id_stream_sha256(()),
    )
    return PoolCommitment(
        run=run,
        input_view=binding,
        selected_sequence_ids=(),
        selection_result=None,
    )


def assemble_rotation_commitments(
    spec: RotationSpec,
    no_query_commitment: PoolCommitment,
    prediction_commitments: Sequence[PoolCommitment],
    random_commitments: Sequence[PoolCommitment],
    ceiling_commitment: PoolCommitment,
    *,
    protocol_seal_sha256: str,
    view_provenance: RotationViewProvenance,
) -> tuple[PoolCommitment, ...]:
    """Merge exact capsule partitions bound to authenticated input-view seals."""

    if not isinstance(spec, RotationSpec):
        raise TypeError("commitment assembly spec must be a RotationSpec")
    if not isinstance(view_provenance, RotationViewProvenance):
        raise TypeError("commitment assembly requires typed authenticated view provenance")
    if view_provenance.spec != spec:
        raise ValueError("authenticated view provenance differs from the requested rotation")
    prediction = tuple(prediction_commitments)
    random = tuple(random_commitments)
    if any(
        not isinstance(item, PoolCommitment)
        for item in (
            no_query_commitment,
            *prediction,
            *random,
            ceiling_commitment,
        )
    ):
        raise TypeError("commitment assembly accepts only PoolCommitment instances")
    expected = policy_runs_for_rotation(spec)
    expected_no_query = next(run for run in expected if run.policy == NO_QUERY)
    expected_prediction = tuple(run for run in expected if run.policy in _PREDICTION_POLICIES)
    expected_random = tuple(run for run in expected if run.policy == RANDOM)
    expected_ceiling = next(run for run in expected if run.policy == CEILING)
    if no_query_commitment.run != expected_no_query:
        raise ValueError("no-query capsule differs from the exact rotation control")
    if tuple(item.run for item in prediction) != expected_prediction:
        raise ValueError("prediction capsules differ from the exact four-track partition")
    if tuple(item.run for item in random) != expected_random:
        raise ValueError("random capsules differ from the exact five-seed partition")
    if ceiling_commitment.run != expected_ceiling:
        raise ValueError("ceiling capsule differs from the exact rotation control")

    expected_none_binding = InputViewBinding(
        kind="none",
        phase_seal_sha256=protocol_seal_sha256,
        payload_sha256=None,
        candidate_count=0,
        candidate_ids_sha256=ordered_id_stream_sha256(()),
    )
    if no_query_commitment.input_view != expected_none_binding:
        raise ValueError("no-query capsule is not bound to the authenticated protocol seal")

    prediction_bindings = {item.input_view for item in prediction}
    random_bindings = {item.input_view for item in random}
    if len(prediction_bindings) != 1:
        raise ValueError("prediction capsules do not share one exact input-view binding")
    if len(random_bindings) != 1:
        raise ValueError("random capsules do not share one exact input-view binding")
    prediction_binding = next(iter(prediction_bindings))
    random_binding = next(iter(random_bindings))
    if (
        prediction_binding.kind != "prediction"
        or prediction_binding.phase_seal_sha256 != view_provenance.prediction_view_seal_sha256
        or prediction_binding.payload_sha256 != view_provenance.prediction_view_payload_sha256
    ):
        raise ValueError("prediction capsules are not bound to the authenticated prediction view")
    if (
        random_binding.kind != "random_minimal"
        or random_binding.phase_seal_sha256 != view_provenance.random_view_seal_sha256
        or random_binding.payload_sha256 != view_provenance.random_view_payload_sha256
    ):
        raise ValueError("random capsules are not bound to the authenticated minimal view")
    ceiling_binding = ceiling_commitment.input_view
    if (
        ceiling_binding.kind != "ceiling_minimal"
        or ceiling_binding.phase_seal_sha256 != view_provenance.random_view_seal_sha256
        or ceiling_binding.payload_sha256 != view_provenance.random_view_payload_sha256
    ):
        raise ValueError("ceiling capsule is not bound to the authenticated minimal view")
    candidate_identity = (
        prediction_binding.candidate_count,
        prediction_binding.candidate_ids_sha256,
    )
    if (
        candidate_identity
        != (
            view_provenance.candidate_count,
            view_provenance.candidate_ids_sha256,
        )
        or candidate_identity
        != (
            random_binding.candidate_count,
            random_binding.candidate_ids_sha256,
        )
        or candidate_identity
        != (
            ceiling_binding.candidate_count,
            ceiling_binding.candidate_ids_sha256,
        )
    ):
        raise ValueError("selector input views do not bind one candidate identity set")

    inputs = (no_query_commitment, *prediction, *random, ceiling_commitment)
    by_track = {item.run.track_id: item for item in inputs}
    if len(by_track) != len(inputs) or set(by_track) != {run.track_id for run in expected}:
        raise ValueError("commitment capsules do not cover the exact rotation policy runs")
    assembled = tuple(by_track[run.track_id] for run in expected)
    if any(item.run.rotation != spec for item in assembled):
        raise ValueError("commitment capsule escaped the requested rotation")
    return assembled


def assemble_campaign_commitments(
    commitments_by_rotation: Mapping[str, Sequence[PoolCommitment]],
    *,
    protocol_seal_sha256: str,
    view_provenance_by_rotation: Mapping[str, RotationViewProvenance],
) -> tuple[PoolCommitment, ...]:
    """Revalidate every rotation's authenticated provenance at the global barrier."""

    rotations = ordered_rotations()
    expected_ids = tuple(rotation.rotation_id for rotation in rotations)
    if not isinstance(commitments_by_rotation, Mapping) or tuple(commitments_by_rotation) != (
        expected_ids
    ):
        raise ValueError("campaign commitments must bind the exact 20 rotations in frozen order")
    if (
        not isinstance(view_provenance_by_rotation, Mapping)
        or tuple(view_provenance_by_rotation) != expected_ids
    ):
        raise ValueError("campaign provenance must bind the exact 20 rotations in frozen order")
    commitments: list[PoolCommitment] = []
    for rotation in rotations:
        current = tuple(commitments_by_rotation[rotation.rotation_id])
        if len(current) != 11:
            raise ValueError("rotation commitments must contain exactly eleven tracks")
        validated = assemble_rotation_commitments(
            rotation,
            current[0],
            (current[1], current[7], current[8], current[9]),
            current[2:7],
            current[10],
            protocol_seal_sha256=protocol_seal_sha256,
            view_provenance=view_provenance_by_rotation[rotation.rotation_id],
        )
        commitments.extend(validated)
    if len(commitments) != EXPECTED_POLICY_RUNS:
        raise AssertionError("campaign commitment census differs from 220")
    if sum(len(item.selected_sequence_ids) for item in commitments) != (
        EXPECTED_POOL_COMMITTED_ASSOCIATIONS
    ):
        raise AssertionError("campaign committed-sequence association census changed")
    return tuple(commitments)


__all__ = [
    "InputViewBinding",
    "PoolCommitment",
    "RotationViewProvenance",
    "assemble_campaign_commitments",
    "assemble_rotation_commitments",
    "make_no_query_commitment",
    "ordered_id_stream_sha256",
    "select_ceiling_view",
    "select_prediction_view",
    "select_random_view",
    "selection_result_document",
]

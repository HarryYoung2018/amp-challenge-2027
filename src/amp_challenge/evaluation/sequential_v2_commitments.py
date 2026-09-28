"""Strict persisted pool commitments for sequential mixed-acquisition v2.

This module is deliberately limited to outcome-free commitment artifacts.  It
does not open Gate-1, stage leaves, or outcome vaults, and it does not launch
workers.  Each selector entry point decodes only its explicitly supplied
rootless candidate-view capability.  Every filesystem publication is delegated
to the generic sequential-v2 phase seal.  Consumers operate on descriptor-
captured ``PhaseSeal`` capabilities and supply the complete expected binding
surface.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from amp_challenge.acquisition.sequential_v2_selector import (
    GuardEvaluation,
    RepairStep,
    SelectedSeat,
    SelectionResult,
    random_priority,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    prediction_view_from_campaign,
    random_minimal_view_from_campaign,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    CEILING,
    EXPECTED_POLICY_RUNS,
    EXPECTED_POOL_COMMITTED_ASSOCIATIONS,
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
    NO_QUERY,
    RANDOM,
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_run_by_track_id,
    policy_runs_for_rotation,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_phase,
    sha256_bytes,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_select import (
    InputViewBinding,
    PoolCommitment,
    RotationViewProvenance,
    assemble_rotation_commitments,
    make_no_query_commitment,
    ordered_id_stream_sha256,
    select_ceiling_view,
    select_prediction_view,
    select_random_view,
    selection_result_document,
)

SCHEMA_VERSION = 1

PREDICTION_SELECTOR_ARTIFACT = "sequential_v2_prediction_selector_commitments_v1"
RANDOM_SELECTOR_ARTIFACT = "sequential_v2_random_selector_commitments_v1"
CEILING_SELECTOR_ARTIFACT = "sequential_v2_ceiling_selector_commitment_v1"
POOL_COMMITMENT_ARTIFACT = "sequential_v2_pool_commitment_v1"
ROTATION_INDEX_ARTIFACT = "sequential_v2_rotation_pool_commitment_index_v1"
CAMPAIGN_BARRIER_ARTIFACT = "sequential_v2_pool_commitment_campaign_barrier_v1"

SELECTOR_PAYLOAD_PATHS = ("commitments.jsonl", "selector-summary.json")
POOL_COMMITMENT_PAYLOAD_PATHS = ("commitment.json",)
ROTATION_INDEX_PAYLOAD_PATHS = (
    "commitment-index.jsonl",
    "rotation-summary.json",
    "view-provenance.json",
)
CAMPAIGN_BARRIER_PAYLOAD_PATHS = (
    "campaign-summary.json",
    "commitment-index.jsonl",
    "rotation-index.jsonl",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"seqv2-div70:[0-9a-f]{64}\Z")
_PREDICTION_POLICIES = (
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
)
_SELECTOR_KINDS = ("prediction", "random", "ceiling")
_FALLBACK_REASONS = frozenset(
    {
        "hard_median_floor_infeasible",
        "repair_candidate_infeasible",
        "reward_guard_failed",
    }
)
_EMPTY_ID_STREAM_SHA256 = ordered_id_stream_sha256(())


def _require_frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} fold identities must be exact integers")
    canonical = rotation_by_id(value.rotation_id)
    if value != canonical:
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    _require_frozen_rotation(value.rotation, label=f"{label} rotation")
    if type(value.policy) is not str or (value.seed is not None and type(value.seed) is not int):
        raise TypeError(f"{label} policy and seed must use exact scalar types")
    canonical = policy_run_by_track_id(value.track_id)
    if value != canonical:
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _require_exact_commitment_graph(commitment: object) -> PoolCommitment:
    if type(commitment) is not PoolCommitment:
        raise TypeError("commitment must be an exact PoolCommitment")
    _require_frozen_run(commitment.run, label="commitment run")
    if type(commitment.input_view) is not InputViewBinding:
        raise TypeError("commitment input view must be an exact InputViewBinding")
    result = commitment.selection_result
    if result is not None:
        if type(result) is not SelectionResult:
            raise TypeError("commitment selection result must be an exact SelectionResult")
        if any(type(item) is not SelectedSeat for item in result.seats):
            raise TypeError("commitment seats must be exact SelectedSeat values")
        if any(type(item) is not RepairStep for item in result.repair_trace):
            raise TypeError("commitment repairs must be exact RepairStep values")
        if any(type(item) is not GuardEvaluation for item in result.guard_evaluations):
            raise TypeError("commitment guards must be exact GuardEvaluation values")
    return commitment


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _exact_object(value: object, fields: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} must contain exactly {sorted(fields)!r}")
    return value


def _strict_json(payload: bytes, *, label: str) -> object:
    if type(payload) is not bytes or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be LF-terminated canonical JSON")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains invalid JSON constant {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical compact JSON")
    return value


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"{label} must be nonempty canonical JSON Lines")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        row = _strict_json(line, label=f"{label} row {index}")
        if type(row) is not dict:
            raise ValueError(f"{label} row {index} must be an object")
        rows.append(row)
    return tuple(rows)


def _float_hex(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical binary64 hexadecimal string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not a hexadecimal binary64 value") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{label} is not canonical finite binary64 hexadecimal text")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} is below its frozen minimum")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} is above its frozen maximum")
    return parsed


def _optional_float_hex(value: object, *, label: str) -> float | None:
    return None if value is None else _float_hex(value, label=label)


def _same_float(left: float, right: float) -> bool:
    return left.hex() == right.hex()


def _ids(
    value: object,
    *,
    label: str,
    expected_count: int | None = None,
    maximum_count: int | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    result = tuple(value)
    if any(type(item) is not str or _SHA256.fullmatch(item) is None for item in result):
        raise ValueError(f"{label} must contain lowercase SHA-256 IDs")
    if (not result and not allow_empty) or len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique IDs")
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} IDs")
    if maximum_count is not None and len(result) > maximum_count:
        raise ValueError(f"{label} must contain at most {maximum_count} IDs")
    return result


def _count_mapping(value: object, *, label: str, component_keys: bool) -> Mapping[str, int]:
    if type(value) is not dict or not value:
        raise ValueError(f"{label} must be a nonempty object")
    parsed: dict[str, int] = {}
    for key, count in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{label} keys must be nonempty strings")
        if component_keys:
            if _COMPONENT.fullmatch(key) is None:
                raise ValueError(f"{label} contains an invalid diversity component ID")
        elif key not in {"exploit", "diversity", "novelty", "random"}:
            raise ValueError(f"{label} contains an invalid role")
        parsed[key] = _integer(count, label=f"{label}.{key}", minimum=1)
    return MappingProxyType(parsed)


def _seat_from_document(value: object, *, label: str) -> SelectedSeat:
    raw = _exact_object(
        value,
        {
            "sequence_id",
            "requested_role",
            "applied_role",
            "acquisition_score_kind",
            "acquisition_score",
            "scalar_mean_hex",
        },
        label=label,
    )
    sequence_id = _sha256(raw["sequence_id"], label=f"{label}.sequence_id")
    requested = _text(raw["requested_role"], label=f"{label}.requested_role")
    applied = _text(raw["applied_role"], label=f"{label}.applied_role")
    if requested not in {"exploit", "diversity", "novelty", "random"} or applied not in {
        "exploit",
        "diversity",
        "novelty",
        "random",
    }:
        raise ValueError(f"{label} has an invalid seat role")
    kind = _text(raw["acquisition_score_kind"], label=f"{label}.acquisition_score_kind")
    if kind == "float_hex":
        score: float | int = _float_hex(
            raw["acquisition_score"],
            label=f"{label}.acquisition_score",
            minimum=0.0,
            maximum=1.0,
        )
    elif kind == "unsigned_integer":
        score = _integer(raw["acquisition_score"], label=f"{label}.acquisition_score")
        if score >= 2**64:
            raise ValueError(f"{label}.acquisition_score exceeds unsigned 64-bit priority")
    else:
        raise ValueError(f"{label} has an unknown acquisition score kind")
    scalar = (
        None
        if raw["scalar_mean_hex"] is None
        else _float_hex(
            raw["scalar_mean_hex"],
            label=f"{label}.scalar_mean_hex",
            minimum=1e-6,
            maximum=0.999999,
        )
    )
    return SelectedSeat(sequence_id, requested, applied, score, scalar)


def _repair_from_document(value: object, *, label: str) -> RepairStep:
    raw = _exact_object(
        value,
        {"step_number", "requested_role", "removed_sequence_id", "added_sequence_id"},
        label=label,
    )
    role = _text(raw["requested_role"], label=f"{label}.requested_role")
    if role not in {"diversity", "novelty"}:
        raise ValueError(f"{label} repair role must be diversity or novelty")
    removed = _sha256(raw["removed_sequence_id"], label=f"{label}.removed_sequence_id")
    added = _sha256(raw["added_sequence_id"], label=f"{label}.added_sequence_id")
    if removed == added:
        raise ValueError(f"{label} cannot replace a sequence by itself")
    return RepairStep(
        _integer(raw["step_number"], label=f"{label}.step_number", minimum=1),
        role,
        removed,
        added,
    )


def _guard_from_document(value: object, *, label: str) -> GuardEvaluation:
    raw = _exact_object(
        value,
        {
            "stage",
            "sequence_ids",
            "objective_means_hex",
            "scalar_mean_hex",
            "scalar_loss_hex",
            "objective_losses_hex",
            "passed",
        },
        label=label,
    )
    stage = _text(raw["stage"], label=f"{label}.stage")
    if stage not in {
        "requested",
        "after_diversity_repair",
        "after_novelty_repair",
        "fallback",
    }:
        raise ValueError(f"{label} has an invalid guard stage")
    sequence_ids = _ids(raw["sequence_ids"], label=f"{label}.sequence_ids", expected_count=10)
    means_raw = raw["objective_means_hex"]
    losses_raw = raw["objective_losses_hex"]
    if type(means_raw) is not list or len(means_raw) != 3:
        raise ValueError(f"{label}.objective_means_hex must contain three values")
    if type(losses_raw) is not list or len(losses_raw) != 3:
        raise ValueError(f"{label}.objective_losses_hex must contain three values")
    means = tuple(
        _float_hex(
            item,
            label=f"{label}.objective_means_hex[{index}]",
            minimum=1e-6,
            maximum=0.999999,
        )
        for index, item in enumerate(means_raw)
    )
    losses = tuple(
        _float_hex(item, label=f"{label}.objective_losses_hex[{index}]")
        for index, item in enumerate(losses_raw)
    )
    scalar_mean = _float_hex(
        raw["scalar_mean_hex"],
        label=f"{label}.scalar_mean_hex",
        minimum=1e-6,
        maximum=0.999999,
    )
    scalar_loss = _float_hex(raw["scalar_loss_hex"], label=f"{label}.scalar_loss_hex")
    expected_mean = math.fsum(means) / 3.0
    expected_loss = math.fsum(losses) / 3.0
    if not _same_float(scalar_mean, expected_mean) or not _same_float(scalar_loss, expected_loss):
        raise ValueError(f"{label} scalar values do not equal their objective means")
    passed = _boolean(raw["passed"], label=f"{label}.passed")
    expected_passed = (
        scalar_loss <= 0.02 or math.isclose(scalar_loss, 0.02, rel_tol=0.0, abs_tol=1e-12)
    ) and all(
        value <= 0.03 or math.isclose(value, 0.03, rel_tol=0.0, abs_tol=1e-12) for value in losses
    )
    if passed is not expected_passed:
        raise ValueError(f"{label} pass flag differs from the frozen reward guard")
    return GuardEvaluation(
        stage=stage,
        sequence_ids=sequence_ids,
        objective_means=means,  # type: ignore[arg-type]
        scalar_mean=scalar_mean,
        scalar_loss=scalar_loss,
        objective_losses=losses,  # type: ignore[arg-type]
        passed=passed,
    )


def _requested_roles(policy: str) -> tuple[str, ...]:
    if policy == MEAN:
        return ("exploit",) * 10
    if policy == MEAN_NINE_DIVERSITY_ONE:
        return ("exploit",) * 9 + ("diversity",)
    if policy == MEAN_NINE_NOVELTY_ONE:
        return ("exploit",) * 9 + ("novelty",)
    if policy == MIXED:
        return ("exploit",) * 8 + ("diversity", "novelty")
    if policy == RANDOM:
        return ("random",) * 10
    raise ValueError(f"selection result has unsupported policy {policy!r}")


def _zero_objective_losses(values: tuple[float, float, float] | None) -> bool:
    return values is not None and all(value == 0.0 for value in values)


def _validate_selection_result(result: SelectionResult, *, run: PolicyRunSpec | None) -> None:
    if type(result) is not SelectionResult:
        raise TypeError("decoded selection result must be a SelectionResult")
    if run is not None:
        _require_frozen_run(run, label="selection result run")
    rotation = rotation_by_id(result.rotation_id)
    if result.policy not in {*_PREDICTION_POLICIES, RANDOM}:
        raise ValueError("selection result policy is not a pool selector policy")
    if run is not None and (
        result.rotation_id != run.rotation.rotation_id or result.policy != run.policy
    ):
        raise ValueError("selection result differs from its policy-run identity")
    seats = tuple(result.seats)
    if len(seats) != 10 or any(type(item) is not SelectedSeat for item in seats):
        raise ValueError("selection result must contain exactly ten typed seats")
    final_ids = tuple(item.sequence_id for item in seats)
    if len(set(final_ids)) != 10 or any(_SHA256.fullmatch(item) is None for item in final_ids):
        raise ValueError("selection result seats must contain ten unique sequence IDs")
    roles = _requested_roles(result.policy)
    if tuple(item.requested_role for item in seats) != roles:
        raise ValueError("selection requested-seat roles differ from the frozen policy")
    requested_counts = dict(Counter(item.requested_role for item in seats))
    applied_counts = dict(Counter(item.applied_role for item in seats))
    if dict(result.requested_role_counts) != requested_counts:
        raise ValueError("selection requested role counts differ from its seats")
    if dict(result.applied_role_counts) != applied_counts:
        raise ValueError("selection applied role counts differ from its seats")
    component_counts = dict(result.component_counts)
    if sum(component_counts.values()) != 10 or any(
        count < 1 or count > 2 or _COMPONENT.fullmatch(component) is None
        for component, count in component_counts.items()
    ):
        raise ValueError("selection component counts violate the strict cap or census")
    requested = tuple(result.requested_sequence_ids)
    if len(set(requested)) != len(requested) or any(
        _SHA256.fullmatch(item) is None for item in requested
    ):
        raise ValueError("selection requested IDs are invalid or duplicated")
    if result.requested_complete is not (len(requested) == 10):
        raise ValueError("selection requested_complete differs from its ID census")

    if result.policy == RANDOM:
        if run is None or run.seed is None:
            raise ValueError("random selection validation requires its seeded policy run")
        if (
            tuple(result.mean_control_sequence_ids)
            or requested != final_ids
            or result.requested_complete is not True
            or result.scalar_loss is not None
            or result.objective_losses is not None
            or tuple(result.repair_trace)
            or tuple(result.guard_evaluations)
            or result.fallback_reason is not None
            or requested_counts != {"random": 10}
            or applied_counts != {"random": 10}
        ):
            raise ValueError("random selection contains prediction-policy state")
        for seat in seats:
            if type(seat.acquisition_score) is not int or seat.scalar_mean is not None:
                raise ValueError("random seats require unsigned priorities and null scalar means")
            expected = random_priority(
                sequence_id=seat.sequence_id,
                outer_fold=rotation.outer_fold,
                pool_fold=rotation.pool_fold,
                seed=run.seed,
            )
            if seat.acquisition_score != expected:
                raise ValueError("random seat priority differs from the frozen keyed hash")
        return

    for seat in seats:
        if type(seat.acquisition_score) is not float or type(seat.scalar_mean) is not float:
            raise ValueError("prediction seats require binary64 score and scalar mean")
        if seat.applied_role not in {seat.requested_role, "exploit"}:
            raise ValueError("prediction seat applied role cannot invent an acquisition arm")
        if seat.applied_role == "exploit" and not _same_float(
            seat.acquisition_score, seat.scalar_mean
        ):
            raise ValueError("exploit seat score must equal its raw scalar mean")

    mean_control = tuple(result.mean_control_sequence_ids)
    if (
        len(mean_control) != 10
        or len(set(mean_control)) != 10
        or any(_SHA256.fullmatch(item) is None for item in mean_control)
    ):
        raise ValueError("prediction selection requires ten unique mean-control IDs")
    if result.scalar_loss is None or result.objective_losses is None:
        raise ValueError("prediction selection must preserve its exact reward losses")
    if len(result.objective_losses) != 3 or not all(
        type(value) is float and math.isfinite(value) for value in result.objective_losses
    ):
        raise ValueError("prediction selection objective losses are invalid")
    expected_loss = math.fsum(result.objective_losses) / 3.0
    if type(result.scalar_loss) is not float or not _same_float(result.scalar_loss, expected_loss):
        raise ValueError("prediction selection scalar loss differs from objective losses")

    repairs = tuple(result.repair_trace)
    if any(type(step) is not RepairStep for step in repairs) or tuple(
        step.step_number for step in repairs
    ) != tuple(range(1, len(repairs) + 1)):
        raise ValueError("selection repair trace step numbers are not canonical")
    repair_roles = tuple(step.requested_role for step in repairs)
    expected_repair_order = tuple(role for role in ("novelty", "diversity") if role in roles)
    if (
        len(set(repair_roles)) != len(repair_roles)
        or any(role not in expected_repair_order for role in repair_roles)
        or tuple(expected_repair_order.index(role) for role in repair_roles)
        != tuple(sorted(expected_repair_order.index(role) for role in repair_roles))
    ):
        raise ValueError("selection repair roles differ from novelty-then-diversity order")

    current = list(requested)
    guard_expectations: list[tuple[str, tuple[str, ...]]] = []
    if result.requested_complete:
        guard_expectations.append(("requested", tuple(current)))
        for step in repairs:
            position = roles.index(step.requested_role)
            if current[position] != step.removed_sequence_id or step.added_sequence_id in current:
                raise ValueError("selection repair trace does not transform the requested IDs")
            current[position] = step.added_sequence_id
            guard_expectations.append((f"after_{step.requested_role}_repair", tuple(current)))
    elif repairs:
        raise ValueError("an incomplete requested selection cannot contain repair steps")

    fallback = result.fallback_reason
    guards = tuple(result.guard_evaluations)
    if any(type(guard) is not GuardEvaluation for guard in guards):
        raise ValueError("selection guard trace must contain exact GuardEvaluation values")
    if fallback is None:
        if result.policy == MEAN:
            if (
                requested != final_ids
                or mean_control != final_ids
                or repairs
                or guards
                or requested_counts != {"exploit": 10}
                or applied_counts != {"exploit": 10}
                or result.scalar_loss != 0.0
                or not _zero_objective_losses(result.objective_losses)
            ):
                raise ValueError("mean selection differs from the exact same-cap control")
            return
        if not result.requested_complete or tuple(current) != final_ids:
            raise ValueError("nonfallback selection does not equal its repaired requested IDs")
        if tuple((guard.stage, guard.sequence_ids) for guard in guards) != tuple(
            guard_expectations
        ):
            raise ValueError("selection guard trace does not follow the requested/repair states")
        if not guards or any(guard.passed for guard in guards[:-1]) or not guards[-1].passed:
            raise ValueError("nonfallback reward-guard trace has invalid pass progression")
        if not _same_float(result.scalar_loss, guards[-1].scalar_loss) or any(
            not _same_float(left, right)
            for left, right in zip(
                result.objective_losses, guards[-1].objective_losses, strict=True
            )
        ):
            raise ValueError("selection result losses differ from its terminal guard")
        repaired_roles = set(repair_roles)
        expected_applied = tuple("exploit" if role in repaired_roles else role for role in roles)
        if tuple(seat.applied_role for seat in seats) != expected_applied:
            raise ValueError("selection applied roles differ from its repair trace")
        return

    if type(fallback) is not str or fallback not in _FALLBACK_REASONS:
        raise ValueError("selection fallback reason is outside the frozen reasons")
    if final_ids != mean_control or any(seat.applied_role != "exploit" for seat in seats):
        raise ValueError("fallback selection must emit the exact mean-control IDs")
    if result.scalar_loss != 0.0 or not _zero_objective_losses(result.objective_losses):
        raise ValueError("fallback selection must report zero control-relative loss")
    if fallback == "hard_median_floor_infeasible" and (
        result.requested_complete or repairs or guard_expectations
    ):
        raise ValueError("hard-floor fallback must preserve an incomplete un-repaired request")
    expected_guards = (*guard_expectations, ("fallback", mean_control))
    if tuple((guard.stage, guard.sequence_ids) for guard in guards) != expected_guards:
        raise ValueError("fallback guard trace does not terminate at the mean control")
    if not guards or any(guard.passed for guard in guards[:-1]) or not guards[-1].passed:
        raise ValueError("fallback guard trace has invalid pass progression")
    terminal = guards[-1]
    if terminal.scalar_loss != 0.0 or any(value != 0.0 for value in terminal.objective_losses):
        raise ValueError("fallback terminal guard must have zero loss")


def _selection_result_from_document(
    value: object,
    *,
    run: PolicyRunSpec | None,
) -> SelectionResult:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "rotation_id",
            "policy",
            "seats",
            "mean_control_sequence_ids",
            "requested_sequence_ids",
            "requested_complete",
            "requested_role_counts",
            "applied_role_counts",
            "component_counts",
            "scalar_loss_hex",
            "objective_losses_hex",
            "repair_trace",
            "guard_evaluations",
            "fallback_reason",
        },
        label="selection result",
    )
    if _integer(raw["schema_version"], label="selection result schema_version") != 1:
        raise ValueError("selection result schema_version must be one")
    rotation_id = _text(raw["rotation_id"], label="selection result rotation_id")
    rotation_by_id(rotation_id)
    policy = _text(raw["policy"], label="selection result policy")
    seats_raw = raw["seats"]
    if type(seats_raw) is not list or len(seats_raw) != 10:
        raise ValueError("selection result seats must contain exactly ten rows")
    seats = tuple(
        _seat_from_document(item, label=f"selection result seat {index}")
        for index, item in enumerate(seats_raw)
    )
    objectives_raw = raw["objective_losses_hex"]
    if objectives_raw is None:
        objective_losses: tuple[float, float, float] | None = None
    else:
        if type(objectives_raw) is not list or len(objectives_raw) != 3:
            raise ValueError("selection result objective losses must contain three values")
        parsed_losses = tuple(
            _float_hex(item, label=f"selection result objective loss {index}")
            for index, item in enumerate(objectives_raw)
        )
        objective_losses = parsed_losses  # type: ignore[assignment]
    repairs_raw = raw["repair_trace"]
    guards_raw = raw["guard_evaluations"]
    if type(repairs_raw) is not list or type(guards_raw) is not list:
        raise ValueError("selection result traces must be JSON arrays")
    fallback = raw["fallback_reason"]
    if fallback is not None and type(fallback) is not str:
        raise ValueError("selection result fallback_reason must be text or null")
    result = SelectionResult(
        rotation_id=rotation_id,
        policy=policy,
        seats=seats,
        mean_control_sequence_ids=_ids(
            raw["mean_control_sequence_ids"],
            label="selection result mean-control IDs",
            allow_empty=True,
        ),
        requested_sequence_ids=_ids(
            raw["requested_sequence_ids"],
            label="selection result requested IDs",
            maximum_count=10,
            allow_empty=True,
        ),
        requested_complete=_boolean(
            raw["requested_complete"], label="selection result requested_complete"
        ),
        requested_role_counts=_count_mapping(
            raw["requested_role_counts"],
            label="selection result requested_role_counts",
            component_keys=False,
        ),
        applied_role_counts=_count_mapping(
            raw["applied_role_counts"],
            label="selection result applied_role_counts",
            component_keys=False,
        ),
        component_counts=_count_mapping(
            raw["component_counts"],
            label="selection result component_counts",
            component_keys=True,
        ),
        scalar_loss=_optional_float_hex(
            raw["scalar_loss_hex"], label="selection result scalar_loss_hex"
        ),
        objective_losses=objective_losses,
        repair_trace=tuple(
            _repair_from_document(item, label=f"selection result repair {index}")
            for index, item in enumerate(repairs_raw)
        ),
        guard_evaluations=tuple(
            _guard_from_document(item, label=f"selection result guard {index}")
            for index, item in enumerate(guards_raw)
        ),
        fallback_reason=fallback,
    )
    _validate_selection_result(result, run=run)
    if canonical_json_bytes(selection_result_document(result)) != canonical_json_bytes(raw):
        raise ValueError("selection result does not round-trip through its exact serializer")
    return result


def decode_selection_result(
    payload: bytes,
    *,
    run: PolicyRunSpec | None = None,
) -> SelectionResult:
    """Decode one canonical selection-result document with full cross-checks."""

    return _selection_result_from_document(
        _strict_json(payload, label="selection result"),
        run=run,
    )


def _input_view_binding_from_document(value: object) -> InputViewBinding:
    raw = _exact_object(
        value,
        {
            "kind",
            "phase_seal_sha256",
            "payload_sha256",
            "candidate_count",
            "candidate_ids_sha256",
        },
        label="input-view binding",
    )
    payload_sha = raw["payload_sha256"]
    if payload_sha is not None:
        payload_sha = _sha256(payload_sha, label="input-view payload")
    binding = InputViewBinding(
        kind=_text(raw["kind"], label="input-view kind"),
        phase_seal_sha256=_sha256(raw["phase_seal_sha256"], label="input-view phase seal"),
        payload_sha256=payload_sha,
        candidate_count=_integer(raw["candidate_count"], label="input-view candidate_count"),
        candidate_ids_sha256=_sha256(
            raw["candidate_ids_sha256"], label="input-view candidate-ID digest"
        ),
    )
    if canonical_json_bytes(binding.document()) != canonical_json_bytes(raw):
        raise ValueError("input-view binding does not round-trip exactly")
    return binding


def decode_input_view_binding(payload: bytes) -> InputViewBinding:
    """Decode one canonical selector input-view binding."""

    return _input_view_binding_from_document(_strict_json(payload, label="input-view binding"))


def rotation_view_provenance_document(
    provenance: RotationViewProvenance,
) -> dict[str, object]:
    if type(provenance) is not RotationViewProvenance:
        raise TypeError("view provenance must be a RotationViewProvenance")
    return {
        "schema_version": SCHEMA_VERSION,
        "rotation_id": provenance.spec.rotation_id,
        "candidate_count": provenance.candidate_count,
        "candidate_ids_sha256": provenance.candidate_ids_sha256,
        "prediction_view_seal_sha256": provenance.prediction_view_seal_sha256,
        "prediction_view_payload_sha256": provenance.prediction_view_payload_sha256,
        "random_view_seal_sha256": provenance.random_view_seal_sha256,
        "random_view_payload_sha256": provenance.random_view_payload_sha256,
    }


def _rotation_view_provenance_from_document(value: object) -> RotationViewProvenance:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "rotation_id",
            "candidate_count",
            "candidate_ids_sha256",
            "prediction_view_seal_sha256",
            "prediction_view_payload_sha256",
            "random_view_seal_sha256",
            "random_view_payload_sha256",
        },
        label="rotation view provenance",
    )
    if _integer(raw["schema_version"], label="view provenance schema_version") != 1:
        raise ValueError("view provenance schema_version must be one")
    rotation_id = _text(raw["rotation_id"], label="view provenance rotation_id")
    spec = rotation_by_id(rotation_id)
    provenance = RotationViewProvenance(
        spec=spec,
        candidate_count=_integer(
            raw["candidate_count"], label="view provenance candidate_count", minimum=1
        ),
        candidate_ids_sha256=_sha256(
            raw["candidate_ids_sha256"], label="view provenance candidate IDs"
        ),
        prediction_view_seal_sha256=_sha256(
            raw["prediction_view_seal_sha256"], label="prediction view seal"
        ),
        prediction_view_payload_sha256=_sha256(
            raw["prediction_view_payload_sha256"], label="prediction view payload"
        ),
        random_view_seal_sha256=_sha256(raw["random_view_seal_sha256"], label="random view seal"),
        random_view_payload_sha256=_sha256(
            raw["random_view_payload_sha256"], label="random view payload"
        ),
    )
    if provenance.prediction_view_seal_sha256 == provenance.random_view_seal_sha256:
        raise ValueError("prediction and random-minimal view seals must be distinct")
    if canonical_json_bytes(rotation_view_provenance_document(provenance)) != canonical_json_bytes(
        raw
    ):
        raise ValueError("view provenance does not round-trip exactly")
    return provenance


def decode_rotation_view_provenance(payload: bytes) -> RotationViewProvenance:
    """Decode one canonical rotation-view provenance document."""

    return _rotation_view_provenance_from_document(
        _strict_json(payload, label="rotation view provenance")
    )


def _pool_commitment_from_document(value: object) -> PoolCommitment:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "track_id",
            "rotation_id",
            "policy",
            "seed",
            "selection_kind",
            "input_view",
            "selected_sequence_ids",
            "selected_sequence_count",
            "selected_sequence_ids_sha256",
            "selection_result",
        },
        label="pool commitment",
    )
    if _integer(raw["schema_version"], label="pool commitment schema_version") != 1:
        raise ValueError("pool commitment schema_version must be one")
    track_id = _text(raw["track_id"], label="pool commitment track_id")
    run = policy_run_by_track_id(track_id)
    if (
        type(raw["rotation_id"]) is not str
        or raw["rotation_id"] != run.rotation.rotation_id
        or type(raw["policy"]) is not str
        or raw["policy"] != run.policy
        or type(raw["selection_kind"]) is not str
        or raw["selection_kind"] != run.selection_kind
        or raw["seed"] != run.seed
        or type(raw["seed"]) is not type(run.seed)
    ):
        raise ValueError("pool commitment identity fields differ from its canonical track")
    selected = _ids(
        raw["selected_sequence_ids"],
        label="pool commitment selected IDs",
        expected_count=run.expected_pool_selection_count,
        allow_empty=run.policy == NO_QUERY,
    )
    if _integer(raw["selected_sequence_count"], label="pool commitment selected count") != len(
        selected
    ):
        raise ValueError("pool commitment selected count differs from its IDs")
    if _sha256(
        raw["selected_sequence_ids_sha256"], label="pool commitment selected-ID digest"
    ) != ordered_id_stream_sha256(selected):
        raise ValueError("pool commitment selected-ID digest differs from its ordered IDs")
    result_raw = raw["selection_result"]
    result = None if result_raw is None else _selection_result_from_document(result_raw, run=run)
    commitment = PoolCommitment(
        run=run,
        input_view=_input_view_binding_from_document(raw["input_view"]),
        selected_sequence_ids=selected,
        selection_result=result,
    )
    if result is not None:
        _validate_selection_result(result, run=run)
    if canonical_json_bytes(commitment.document()) != canonical_json_bytes(raw):
        raise ValueError("pool commitment does not round-trip through its exact serializer")
    return commitment


def decode_pool_commitment(payload: bytes) -> PoolCommitment:
    """Decode one canonical pool commitment and its complete selector trace."""

    return _pool_commitment_from_document(_strict_json(payload, label="pool commitment"))


def _validate_view_binding(
    commitment: PoolCommitment,
    *,
    protocol_seal_sha256: str,
    provenance: RotationViewProvenance,
) -> None:
    protocol = _sha256(protocol_seal_sha256, label="protocol seal")
    if commitment.run.rotation != provenance.spec:
        raise ValueError("commitment and view provenance rotations differ")
    binding = commitment.input_view
    expected_kind: str
    expected_seal: str
    expected_payload: str | None
    expected_count: int
    expected_ids: str
    if commitment.run.policy == NO_QUERY:
        expected_kind = "none"
        expected_seal = protocol
        expected_payload = None
        expected_count = 0
        expected_ids = _EMPTY_ID_STREAM_SHA256
    elif commitment.run.policy in _PREDICTION_POLICIES:
        expected_kind = "prediction"
        expected_seal = provenance.prediction_view_seal_sha256
        expected_payload = provenance.prediction_view_payload_sha256
        expected_count = provenance.candidate_count
        expected_ids = provenance.candidate_ids_sha256
    else:
        expected_kind = "random_minimal" if commitment.run.policy == RANDOM else "ceiling_minimal"
        expected_seal = provenance.random_view_seal_sha256
        expected_payload = provenance.random_view_payload_sha256
        expected_count = provenance.candidate_count
        expected_ids = provenance.candidate_ids_sha256
    if (
        binding.kind != expected_kind
        or binding.phase_seal_sha256 != expected_seal
        or binding.payload_sha256 != expected_payload
        or binding.candidate_count != expected_count
        or binding.candidate_ids_sha256 != expected_ids
    ):
        raise ValueError("commitment input binding differs from authenticated view provenance")


def _canonical_commitment(commitment: PoolCommitment) -> tuple[PoolCommitment, bytes]:
    _require_exact_commitment_graph(commitment)
    payload = canonical_json_bytes(commitment.document())
    decoded = decode_pool_commitment(payload)
    if canonical_json_bytes(decoded.document()) != payload:
        raise AssertionError("commitment codec changed canonical bytes")
    return decoded, payload


def _commitment_bytes(commitment: PoolCommitment) -> bytes:
    """Return the exact canonical identity, preserving signed binary64 zero."""

    _require_exact_commitment_graph(commitment)
    return canonical_json_bytes(commitment.document())


def _commitment_sequence_bytes(commitments: Sequence[PoolCommitment]) -> bytes:
    values = tuple(commitments)
    for item in values:
        _require_exact_commitment_graph(item)
    return canonical_jsonl_bytes(item.document() for item in values)


def _protocol_predecessor() -> str:
    return "protocol/SHA256SUMS"


def _prepare_barrier_predecessor() -> str:
    return "prepare/global/SHA256SUMS"


def _evidence_predecessor(spec: RotationSpec) -> str:
    return f"prepare/rotations/{spec.rotation_id}/evidence/SHA256SUMS"


def _view_predecessor(spec: RotationSpec, kind: str) -> str:
    if kind == "prediction":
        role = "prediction-view"
    elif kind == "random-minimal":
        role = "random-minimal-view"
    else:
        raise ValueError("view predecessor kind is invalid")
    return f"prepare/rotations/{spec.rotation_id}/{role}/SHA256SUMS"


def _selector_predecessor(spec: RotationSpec, kind: str) -> str:
    if kind not in _SELECTOR_KINDS:
        raise ValueError("selector predecessor kind is invalid")
    return f"select/selector-outputs/{spec.rotation_id}/{kind}/SHA256SUMS"


def pool_commitment_relative_path(run: PolicyRunSpec) -> str:
    """Return the frozen repository-relative path for one commitment leaf."""

    _require_frozen_run(run, label="commitment path run")
    return f"select/rotations/{run.rotation.rotation_id}/commitments/{run.track_id}"


def _commitment_predecessor(run: PolicyRunSpec) -> str:
    return f"{pool_commitment_relative_path(run)}/SHA256SUMS"


def rotation_commitment_index_relative_path(spec: RotationSpec) -> str:
    """Return the frozen repository-relative path for one rotation index."""

    _require_frozen_rotation(spec, label="rotation index spec")
    return f"select/rotations/{spec.rotation_id}/global"


def _rotation_predecessor(spec: RotationSpec) -> str:
    return f"{rotation_commitment_index_relative_path(spec)}/SHA256SUMS"


def _identity_metadata(
    publication_identity: SequentialV2PublicationIdentity,
    *,
    scope_id: str,
) -> dict[str, object]:
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("publication_identity must be a SequentialV2PublicationIdentity")
    return publication_identity.metadata(phase="select", scope_id=scope_id)


def _verify_identity_metadata(
    seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    *,
    scope_id: str,
) -> None:
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("publication_identity must be a SequentialV2PublicationIdentity")
    publication_identity.verify_metadata(
        seal.metadata_json,
        phase="select",
        scope_id=scope_id,
    )


def _verify_prepare_campaign_context(
    prepare_campaign: PrepareCampaignCapability,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_prepare_campaign_seal_sha256: str,
) -> PrepareCampaignCapability:
    if type(prepare_campaign) is not PrepareCampaignCapability:
        raise TypeError("selection phases require a PrepareCampaignCapability")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("selection phases require a ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    return verify_prepare_campaign_capability(
        prepare_campaign,
        publication_identity=publication_identity,
        expected_campaign_seal_sha256=_sha256(
            expected_prepare_campaign_seal_sha256,
            label="prepare campaign seal",
        ),
        expected_protocol_seal_sha256=protocol.seal.seal_sha256,
    )


def _selector_artifact(kind: str) -> str:
    try:
        return {
            "prediction": PREDICTION_SELECTOR_ARTIFACT,
            "random": RANDOM_SELECTOR_ARTIFACT,
            "ceiling": CEILING_SELECTOR_ARTIFACT,
        }[kind]
    except KeyError as error:
        raise ValueError("selector kind is invalid") from error


def _selector_kind_for_run(run: PolicyRunSpec) -> str | None:
    if run.policy == NO_QUERY:
        return None
    if run.policy in _PREDICTION_POLICIES:
        return "prediction"
    if run.policy == RANDOM:
        return "random"
    if run.policy == CEILING:
        return "ceiling"
    raise AssertionError("frozen policy escaped its selector partition")


def _selector_expected_runs(spec: RotationSpec, kind: str) -> tuple[PolicyRunSpec, ...]:
    runs = policy_runs_for_rotation(spec)
    if kind == "prediction":
        return tuple(run for run in runs if run.policy in _PREDICTION_POLICIES)
    if kind == "random":
        return tuple(run for run in runs if run.policy == RANDOM)
    if kind == "ceiling":
        return tuple(run for run in runs if run.policy == CEILING)
    raise ValueError("selector kind is invalid")


def _validate_provenance(
    provenance: RotationViewProvenance,
    *,
    spec: RotationSpec,
) -> RotationViewProvenance:
    _require_frozen_rotation(spec, label="phase rotation")
    if type(provenance) is not RotationViewProvenance:
        raise TypeError("phase requires typed rotation and view provenance")
    _require_frozen_rotation(provenance.spec, label="view provenance rotation")
    if provenance.spec != spec:
        raise ValueError("view provenance differs from the requested rotation")
    if provenance.candidate_count != EXPECTED_SUPPORT_BY_FOLD[spec.pool_fold]:
        raise ValueError("view provenance candidate census differs from frozen support")
    if provenance.prediction_view_seal_sha256 == provenance.random_view_seal_sha256:
        raise ValueError("prediction and random-minimal view seals must be distinct")
    return provenance


def _selector_predecessors(
    spec: RotationSpec,
    kind: str,
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    provenance: RotationViewProvenance,
) -> dict[str, str]:
    provenance = _validate_provenance(provenance, spec=spec)
    result = {
        _protocol_predecessor(): _sha256(protocol_seal_sha256, label="protocol seal"),
        _prepare_barrier_predecessor(): _sha256(
            prepare_barrier_seal_sha256, label="prepare barrier seal"
        ),
    }
    if kind == "prediction":
        result[_view_predecessor(spec, "prediction")] = provenance.prediction_view_seal_sha256
    elif kind in {"random", "ceiling"}:
        result[_view_predecessor(spec, "random-minimal")] = provenance.random_view_seal_sha256
    else:
        raise ValueError("selector kind is invalid")
    return result


def _validate_selector_commitments(
    commitments: Sequence[PoolCommitment],
    *,
    spec: RotationSpec,
    kind: str,
    protocol_seal_sha256: str,
    provenance: RotationViewProvenance,
) -> tuple[PoolCommitment, ...]:
    provenance = _validate_provenance(provenance, spec=spec)
    values = tuple(_canonical_commitment(item)[0] for item in commitments)
    expected_runs = _selector_expected_runs(spec, kind)
    if tuple(item.run for item in values) != expected_runs:
        raise ValueError(f"{kind} selector commitments differ from frozen run order")
    for item in values:
        _validate_view_binding(
            item,
            protocol_seal_sha256=protocol_seal_sha256,
            provenance=provenance,
        )
    return values


def _selector_summary_document(
    spec: RotationSpec,
    kind: str,
    commitments: Sequence[PoolCommitment],
    *,
    commitments_payload_sha256: str,
) -> dict[str, object]:
    values = tuple(commitments)
    if not values:
        raise ValueError("selector summary cannot describe an empty partition")
    bindings = {item.input_view for item in values}
    if len(bindings) != 1:
        raise ValueError("selector commitments must share one exact input-view binding")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": _selector_artifact(kind),
        "rotation_id": spec.rotation_id,
        "selector_kind": kind,
        "input_view": next(iter(bindings)).document(),
        "track_count": len(values),
        "track_ids": [item.run.track_id for item in values],
        "selected_sequence_association_count": sum(
            len(item.selected_sequence_ids) for item in values
        ),
        "commitments_payload_sha256": _sha256(
            commitments_payload_sha256,
            label="selector commitments payload",
        ),
    }


def _publish_selector_phase(
    destination: str | Path,
    commitments: Sequence[PoolCommitment],
    *,
    spec: RotationSpec,
    kind: str,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    values = _validate_selector_commitments(
        commitments,
        spec=spec,
        kind=kind,
        protocol_seal_sha256=protocol_seal_sha256,
        provenance=view_provenance,
    )
    commitments_payload = canonical_jsonl_bytes(item.document() for item in values)
    summary = _selector_summary_document(
        spec,
        kind,
        values,
        commitments_payload_sha256=sha256_bytes(commitments_payload),
    )
    seal = publish_phase(
        destination,
        artifact=_selector_artifact(kind),
        payloads={
            "commitments.jsonl": commitments_payload,
            "selector-summary.json": canonical_json_bytes(summary),
        },
        predecessor_seals=_selector_predecessors(
            spec,
            kind,
            protocol_seal_sha256=protocol_seal_sha256,
            prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
            provenance=view_provenance,
        ),
        metadata=_identity_metadata(publication_identity, scope_id=spec.rotation_id),
    )
    _verify_selector_commitment_phase_capability_raw(
        seal,
        spec=spec,
        selector_kind=kind,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        view_provenance=view_provenance,
        publication_identity=publication_identity,
        expected_seal_sha256=seal.seal_sha256,
    )
    return seal


def publish_prediction_selector_commitments(
    destination: str | Path,
    *,
    spec: RotationSpec,
    prediction_view_seal: PhaseSeal,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Seal the exact four prediction-selector commitments for one rotation."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    provenance = campaign.rotation_view_provenance(spec=spec)
    candidates = prediction_view_from_campaign(
        campaign,
        spec=spec,
        prediction_view_seal=prediction_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    commitments = select_prediction_view(
        spec,
        candidates,
        input_view_seal_sha256=provenance.prediction_view_seal_sha256,
        input_view_payload_sha256=provenance.prediction_view_payload_sha256,
    )
    return _publish_selector_phase(
        destination,
        commitments,
        spec=spec,
        kind="prediction",
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
    )


def publish_random_selector_commitments(
    destination: str | Path,
    *,
    spec: RotationSpec,
    random_minimal_view_seal: PhaseSeal,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Seal the exact five random-seed commitments for one rotation."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    provenance = campaign.rotation_view_provenance(spec=spec)
    candidates = random_minimal_view_from_campaign(
        campaign,
        spec=spec,
        random_minimal_view_seal=random_minimal_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    commitments = select_random_view(
        spec,
        candidates,
        input_view_seal_sha256=provenance.random_view_seal_sha256,
        input_view_payload_sha256=provenance.random_view_payload_sha256,
    )
    return _publish_selector_phase(
        destination,
        commitments,
        spec=spec,
        kind="random",
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
    )


def publish_ceiling_selector_commitment(
    destination: str | Path,
    *,
    spec: RotationSpec,
    random_minimal_view_seal: PhaseSeal,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Seal the sole full-acquisition-fold ceiling commitment for one rotation."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    provenance = campaign.rotation_view_provenance(spec=spec)
    candidates = random_minimal_view_from_campaign(
        campaign,
        spec=spec,
        random_minimal_view_seal=random_minimal_view_seal,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    commitment = select_ceiling_view(
        spec,
        candidates,
        input_view_seal_sha256=provenance.random_view_seal_sha256,
        input_view_payload_sha256=provenance.random_view_payload_sha256,
    )
    return _publish_selector_phase(
        destination,
        (commitment,),
        spec=spec,
        kind="ceiling",
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
    )


def _verify_selector_commitment_phase_capability_raw(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    selector_kind: str,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[PoolCommitment, ...]:
    """Authenticate and decode one rootless prediction/random/ceiling output."""

    predecessors = _selector_predecessors(
        spec,
        selector_kind,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        provenance=view_provenance,
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=_selector_artifact(selector_kind),
        expected_payload_paths=SELECTOR_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=_sha256(expected_seal_sha256, label="selector phase seal"),
    )
    _verify_identity_metadata(verified, publication_identity, scope_id=spec.rotation_id)
    commitments_payload = verified.read_payload_bytes("commitments.jsonl")
    rows = _strict_jsonl(
        commitments_payload,
        label=f"{selector_kind} selector commitments",
    )
    commitments = _validate_selector_commitments(
        tuple(_pool_commitment_from_document(row) for row in rows),
        spec=spec,
        kind=selector_kind,
        protocol_seal_sha256=protocol_seal_sha256,
        provenance=view_provenance,
    )
    expected_summary = _selector_summary_document(
        spec,
        selector_kind,
        commitments,
        commitments_payload_sha256=sha256_bytes(commitments_payload),
    )
    summary_payload = verified.read_payload_bytes("selector-summary.json")
    _strict_json(summary_payload, label=f"{selector_kind} selector summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("selector summary differs from its decoded commitments")
    return commitments


def verify_selector_commitment_phase_capability(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    selector_kind: str,
    input_view_seal: PhaseSeal,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[PoolCommitment, ...]:
    """Authenticate one selector output through the label-free prepare barrier."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    provenance = campaign.rotation_view_provenance(spec=spec)
    if selector_kind == "prediction":
        candidates = prediction_view_from_campaign(
            campaign,
            spec=spec,
            prediction_view_seal=input_view_seal,
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        )
        expected = select_prediction_view(
            spec,
            candidates,
            input_view_seal_sha256=provenance.prediction_view_seal_sha256,
            input_view_payload_sha256=provenance.prediction_view_payload_sha256,
        )
    elif selector_kind in {"random", "ceiling"}:
        minimal = random_minimal_view_from_campaign(
            campaign,
            spec=spec,
            random_minimal_view_seal=input_view_seal,
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        )
        expected = (
            select_random_view(
                spec,
                minimal,
                input_view_seal_sha256=provenance.random_view_seal_sha256,
                input_view_payload_sha256=provenance.random_view_payload_sha256,
            )
            if selector_kind == "random"
            else (
                select_ceiling_view(
                    spec,
                    minimal,
                    input_view_seal_sha256=provenance.random_view_seal_sha256,
                    input_view_payload_sha256=provenance.random_view_payload_sha256,
                ),
            )
        )
    else:
        raise ValueError("selector kind is invalid")
    decoded = _verify_selector_commitment_phase_capability_raw(
        seal,
        spec=spec,
        selector_kind=selector_kind,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_seal_sha256,
    )
    if _commitment_sequence_bytes(decoded) != _commitment_sequence_bytes(expected):
        raise ValueError("selector output differs from deterministic selection replay")
    return decoded


def _pool_commitment_predecessors(
    run: PolicyRunSpec,
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    input_view: InputViewBinding,
    selector_output_seal_sha256: str | None,
) -> dict[str, str]:
    _require_frozen_run(run, label="commitment predecessor run")
    if type(input_view) is not InputViewBinding:
        raise TypeError("commitment predecessors require typed run and input view")
    protocol = _sha256(protocol_seal_sha256, label="protocol seal")
    result = {
        _protocol_predecessor(): protocol,
        _prepare_barrier_predecessor(): _sha256(
            prepare_barrier_seal_sha256,
            label="prepare barrier seal",
        ),
    }
    kind = _selector_kind_for_run(run)
    if kind is None:
        if selector_output_seal_sha256 is not None:
            raise ValueError("no-query commitment cannot bind a selector output")
        if input_view != InputViewBinding(
            kind="none",
            phase_seal_sha256=protocol,
            payload_sha256=None,
            candidate_count=0,
            candidate_ids_sha256=_EMPTY_ID_STREAM_SHA256,
        ):
            raise ValueError("no-query commitment is not bound to the exact protocol")
        return result
    selector_seal = _sha256(
        selector_output_seal_sha256,
        label=f"{kind} selector output seal",
    )
    view_kind = "prediction" if kind == "prediction" else "random-minimal"
    result[_view_predecessor(run.rotation, view_kind)] = input_view.phase_seal_sha256
    result[_selector_predecessor(run.rotation, kind)] = selector_seal
    return result


def _publish_pool_commitment_raw(
    destination: str | Path,
    commitment: PoolCommitment,
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    selector_output_seal_sha256: str | None,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Publish one physically separate, single-track commitment leaf."""

    value, payload = _canonical_commitment(commitment)
    _validate_provenance(view_provenance, spec=value.run.rotation)
    _validate_view_binding(
        value,
        protocol_seal_sha256=protocol_seal_sha256,
        provenance=view_provenance,
    )
    predecessors = _pool_commitment_predecessors(
        value.run,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        input_view=value.input_view,
        selector_output_seal_sha256=selector_output_seal_sha256,
    )
    seal = publish_phase(
        destination,
        artifact=POOL_COMMITMENT_ARTIFACT,
        payloads={"commitment.json": payload},
        predecessor_seals=predecessors,
        metadata=_identity_metadata(
            publication_identity,
            scope_id=value.run.track_id,
        ),
    )
    decoded = _verify_pool_commitment_phase_capability_raw(
        seal,
        run=value.run,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        selector_output_seal_sha256=selector_output_seal_sha256,
        view_provenance=view_provenance,
        publication_identity=publication_identity,
        expected_seal_sha256=seal.seal_sha256,
    )
    if _commitment_bytes(decoded) != _commitment_bytes(value):
        raise RuntimeError("published commitment leaf changed its decoded value")
    return seal


def _verify_pool_commitment_phase_capability_raw(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    selector_output_seal_sha256: str | None,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> PoolCommitment:
    """Authenticate and decode one rootless single-track commitment leaf."""

    _require_frozen_run(run, label="commitment verification run")
    provenance = _validate_provenance(view_provenance, spec=run.rotation)
    expected_binding = (
        InputViewBinding(
            kind="none",
            phase_seal_sha256=_sha256(protocol_seal_sha256, label="protocol seal"),
            payload_sha256=None,
            candidate_count=0,
            candidate_ids_sha256=_EMPTY_ID_STREAM_SHA256,
        )
        if run.policy == NO_QUERY
        else InputViewBinding(
            kind=(
                "prediction"
                if run.policy in _PREDICTION_POLICIES
                else "random_minimal"
                if run.policy == RANDOM
                else "ceiling_minimal"
            ),
            phase_seal_sha256=(
                provenance.prediction_view_seal_sha256
                if run.policy in _PREDICTION_POLICIES
                else provenance.random_view_seal_sha256
            ),
            payload_sha256=(
                provenance.prediction_view_payload_sha256
                if run.policy in _PREDICTION_POLICIES
                else provenance.random_view_payload_sha256
            ),
            candidate_count=provenance.candidate_count,
            candidate_ids_sha256=provenance.candidate_ids_sha256,
        )
    )
    predecessors = _pool_commitment_predecessors(
        run,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        input_view=expected_binding,
        selector_output_seal_sha256=selector_output_seal_sha256,
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=POOL_COMMITMENT_ARTIFACT,
        expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=_sha256(expected_seal_sha256, label="commitment leaf seal"),
    )
    _verify_identity_metadata(verified, publication_identity, scope_id=run.track_id)
    commitment = decode_pool_commitment(verified.read_payload_bytes("commitment.json"))
    if commitment.run != run or commitment.input_view != expected_binding:
        raise ValueError("commitment leaf differs from its expected run or input view")
    _validate_view_binding(
        commitment,
        protocol_seal_sha256=protocol_seal_sha256,
        provenance=provenance,
    )
    return commitment


def _commitment_from_selector_capability(
    run: PolicyRunSpec,
    selector_output_seal: PhaseSeal | None,
    *,
    expected_selector_output_seal_sha256: str | None,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
) -> tuple[PoolCommitment, str | None]:
    kind = _selector_kind_for_run(run)
    if kind is None:
        if selector_output_seal is not None or expected_selector_output_seal_sha256 is not None:
            raise ValueError("no-query commitment cannot receive selector authority")
        return (
            make_no_query_commitment(
                run.rotation,
                protocol_seal_sha256=protocol_seal_sha256,
            ),
            None,
        )
    if type(selector_output_seal) is not PhaseSeal:
        raise TypeError("budgeted commitment requires a rootless selector PhaseSeal")
    expected_selector_seal = _sha256(
        expected_selector_output_seal_sha256,
        label=f"expected {kind} selector output seal",
    )
    commitments = _verify_selector_commitment_phase_capability_raw(
        selector_output_seal,
        spec=run.rotation,
        selector_kind=kind,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        view_provenance=view_provenance,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_selector_seal,
    )
    matches = tuple(value for value in commitments if value.run == run)
    if len(matches) != 1:
        raise ValueError("selector capability does not contain one exact requested track")
    return matches[0], expected_selector_seal


def publish_pool_commitment(
    destination: str | Path,
    commitment: PoolCommitment,
    *,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    selector_output_seal: PhaseSeal | None,
    expected_selector_output_seal_sha256: str | None,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Publish one track leaf through an authenticated prepare-global index."""

    _require_exact_commitment_graph(commitment)
    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    provenance = campaign.rotation_view_provenance(spec=commitment.run.rotation)
    expected, selector_seal_sha256 = _commitment_from_selector_capability(
        commitment.run,
        selector_output_seal,
        expected_selector_output_seal_sha256=expected_selector_output_seal_sha256,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
    )
    if _commitment_bytes(commitment) != _commitment_bytes(expected):
        raise ValueError("commitment differs from its authenticated selector output")
    return _publish_pool_commitment_raw(
        destination,
        commitment,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        selector_output_seal_sha256=selector_seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
    )


def verify_pool_commitment_phase_capability(
    seal: PhaseSeal,
    *,
    run: PolicyRunSpec,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    selector_output_seal: PhaseSeal | None,
    expected_selector_output_seal_sha256: str | None,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> PoolCommitment:
    """Authenticate one track leaf through an authenticated prepare index."""

    _require_frozen_run(run, label="commitment verification run")
    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    provenance = campaign.rotation_view_provenance(spec=run.rotation)
    expected, selector_seal_sha256 = _commitment_from_selector_capability(
        run,
        selector_output_seal,
        expected_selector_output_seal_sha256=expected_selector_output_seal_sha256,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
    )
    decoded = _verify_pool_commitment_phase_capability_raw(
        seal,
        run=run,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        selector_output_seal_sha256=selector_seal_sha256,
        view_provenance=provenance,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_seal_sha256,
    )
    if _commitment_bytes(decoded) != _commitment_bytes(expected):
        raise ValueError("commitment leaf differs from its authenticated selector output")
    return decoded


@dataclass(frozen=True, slots=True)
class CommitmentIndexRow:
    """Exact authenticated index record for one single-track leaf."""

    run: PolicyRunSpec
    relative_path: str
    leaf_seal_sha256: str
    commitment_payload_sha256: str
    selected_sequence_count: int
    selected_sequence_ids_sha256: str
    selector_kind: str | None
    selector_output_seal_sha256: str | None
    input_view: InputViewBinding

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="commitment index run")
        if type(self.input_view) is not InputViewBinding:
            raise TypeError("commitment index row requires typed run and input view")
        if type(self.relative_path) is not str or self.relative_path != (
            pool_commitment_relative_path(self.run)
        ):
            raise ValueError("commitment index row has a noncanonical leaf path")
        _sha256(self.leaf_seal_sha256, label="commitment index leaf seal")
        _sha256(self.commitment_payload_sha256, label="commitment index payload")
        _sha256(self.selected_sequence_ids_sha256, label="commitment index selected IDs")
        if type(self.selected_sequence_count) is not int or (
            self.selected_sequence_count != self.run.expected_pool_selection_count
        ):
            raise ValueError("commitment index selected count differs from frozen policy")
        expected_kind = _selector_kind_for_run(self.run)
        if self.selector_kind != expected_kind or type(self.selector_kind) is not type(
            expected_kind
        ):
            raise ValueError("commitment index selector kind differs from frozen policy")
        if expected_kind is None:
            if self.selector_output_seal_sha256 is not None:
                raise ValueError("no-query index row cannot name a selector output seal")
        else:
            _sha256(
                self.selector_output_seal_sha256,
                label="commitment index selector output seal",
            )
        expected_view_kind = (
            "none"
            if expected_kind is None
            else "prediction"
            if expected_kind == "prediction"
            else "random_minimal"
            if expected_kind == "random"
            else "ceiling_minimal"
        )
        if self.input_view.kind != expected_view_kind:
            raise ValueError("commitment index input-view kind differs from frozen policy")
        if expected_kind is not None and (
            self.input_view.candidate_count != EXPECTED_SUPPORT_BY_FOLD[self.run.rotation.pool_fold]
        ):
            raise ValueError("commitment index candidate census differs from frozen support")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": self.run.track_id,
            "rotation_id": self.run.rotation.rotation_id,
            "policy": self.run.policy,
            "seed": self.run.seed,
            "selection_kind": self.run.selection_kind,
            "relative_path": self.relative_path,
            "leaf_artifact": POOL_COMMITMENT_ARTIFACT,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "commitment_payload_sha256": self.commitment_payload_sha256,
            "selected_sequence_count": self.selected_sequence_count,
            "selected_sequence_ids_sha256": self.selected_sequence_ids_sha256,
            "selector_kind": self.selector_kind,
            "selector_output_seal_sha256": self.selector_output_seal_sha256,
            "input_view": self.input_view.document(),
        }


def _commitment_index_row_from_document(value: object) -> CommitmentIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "track_id",
            "rotation_id",
            "policy",
            "seed",
            "selection_kind",
            "relative_path",
            "leaf_artifact",
            "leaf_seal_sha256",
            "commitment_payload_sha256",
            "selected_sequence_count",
            "selected_sequence_ids_sha256",
            "selector_kind",
            "selector_output_seal_sha256",
            "input_view",
        },
        label="commitment index row",
    )
    if _integer(raw["schema_version"], label="commitment index schema_version") != 1:
        raise ValueError("commitment index schema_version must be one")
    run = policy_run_by_track_id(_text(raw["track_id"], label="commitment index track_id"))
    if (
        type(raw["rotation_id"]) is not str
        or raw["rotation_id"] != run.rotation.rotation_id
        or type(raw["policy"]) is not str
        or raw["policy"] != run.policy
        or raw["seed"] != run.seed
        or type(raw["seed"]) is not type(run.seed)
        or type(raw["selection_kind"]) is not str
        or raw["selection_kind"] != run.selection_kind
        or raw["leaf_artifact"] != POOL_COMMITMENT_ARTIFACT
        or type(raw["leaf_artifact"]) is not str
    ):
        raise ValueError("commitment index identity fields differ from canonical track")
    selector_kind = raw["selector_kind"]
    if selector_kind is not None and type(selector_kind) is not str:
        raise ValueError("commitment index selector kind must be text or null")
    selector_seal = raw["selector_output_seal_sha256"]
    if selector_seal is not None:
        selector_seal = _sha256(selector_seal, label="commitment index selector seal")
    row = CommitmentIndexRow(
        run=run,
        relative_path=_text(raw["relative_path"], label="commitment index relative_path"),
        leaf_seal_sha256=_sha256(raw["leaf_seal_sha256"], label="commitment index leaf seal"),
        commitment_payload_sha256=_sha256(
            raw["commitment_payload_sha256"],
            label="commitment index payload",
        ),
        selected_sequence_count=_integer(
            raw["selected_sequence_count"],
            label="commitment index selected count",
        ),
        selected_sequence_ids_sha256=_sha256(
            raw["selected_sequence_ids_sha256"],
            label="commitment index selected IDs",
        ),
        selector_kind=selector_kind,
        selector_output_seal_sha256=selector_seal,
        input_view=_input_view_binding_from_document(raw["input_view"]),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("commitment index row does not round-trip exactly")
    return row


def _payload_digest(seal: PhaseSeal, path: str) -> str:
    matches = tuple(digest for current, digest in seal.payload_sha256 if current == path)
    if len(matches) != 1:
        raise ValueError(f"phase lacks one exact payload digest for {path}")
    return matches[0]


def _commitment_index_row(
    commitment: PoolCommitment,
    seal: PhaseSeal,
    *,
    selector_output_seal_sha256: str | None,
) -> CommitmentIndexRow:
    return CommitmentIndexRow(
        run=commitment.run,
        relative_path=pool_commitment_relative_path(commitment.run),
        leaf_seal_sha256=seal.seal_sha256,
        commitment_payload_sha256=_payload_digest(seal, "commitment.json"),
        selected_sequence_count=len(commitment.selected_sequence_ids),
        selected_sequence_ids_sha256=ordered_id_stream_sha256(commitment.selected_sequence_ids),
        selector_kind=_selector_kind_for_run(commitment.run),
        selector_output_seal_sha256=selector_output_seal_sha256,
        input_view=commitment.input_view,
    )


def _selector_seal_mapping(
    selector_phase_seals: Mapping[str, PhaseSeal],
) -> dict[str, PhaseSeal]:
    if not isinstance(selector_phase_seals, Mapping):
        raise ValueError("selector phase map must contain prediction, random, and ceiling")
    result = dict(selector_phase_seals)
    if set(result) != set(_SELECTOR_KINDS):
        raise ValueError("selector phase map must contain prediction, random, and ceiling")
    if any(type(item) is not PhaseSeal for item in result.values()):
        raise TypeError("selector phase map values must be PhaseSeal capabilities")
    return result


def _selector_digest_mapping(
    values: Mapping[str, str],
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must contain prediction, random, and ceiling")
    captured = dict(values)
    if set(captured) != set(_SELECTOR_KINDS):
        raise ValueError(f"{label} must contain prediction, random, and ceiling")
    return {kind: _sha256(captured[kind], label=f"{label} {kind}") for kind in _SELECTOR_KINDS}


def _seal_digest_sequence(
    values: Sequence[str],
    *,
    expected_count: int,
    label: str,
) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be an ordered sequence")
    result = tuple(values)
    if len(result) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} digests")
    return tuple(_sha256(value, label=f"{label}[{index}]") for index, value in enumerate(result))


def _leaf_seal_sequence(
    seals: Sequence[PhaseSeal],
    *,
    expected_count: int,
    label: str,
) -> tuple[PhaseSeal, ...]:
    if isinstance(seals, str | bytes) or not isinstance(seals, Sequence):
        raise TypeError(f"{label} must be an ordered sequence")
    result = tuple(seals)
    if len(result) != expected_count or any(type(item) is not PhaseSeal for item in result):
        raise ValueError(f"{label} must contain exactly {expected_count} PhaseSeal values")
    return result


def _rotation_commitment_rows(
    *,
    spec: RotationSpec,
    selector_phase_seals: Mapping[str, PhaseSeal],
    commitment_leaf_seals: Sequence[PhaseSeal],
    expected_selector_seal_sha256_by_kind: Mapping[str, str],
    expected_commitment_leaf_seal_sha256s: Sequence[str],
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
) -> tuple[CommitmentIndexRow, ...]:
    selectors = _selector_seal_mapping(selector_phase_seals)
    expected_selectors = _selector_digest_mapping(
        expected_selector_seal_sha256_by_kind,
        label="expected rotation selector seals",
    )
    provenance = _validate_provenance(view_provenance, spec=spec)
    partitions = {
        kind: _verify_selector_commitment_phase_capability_raw(
            selectors[kind],
            spec=spec,
            selector_kind=kind,
            protocol_seal_sha256=protocol_seal_sha256,
            prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
            view_provenance=provenance,
            publication_identity=publication_identity,
            expected_seal_sha256=expected_selectors[kind],
        )
        for kind in _SELECTOR_KINDS
    }
    no_query = make_no_query_commitment(
        spec,
        protocol_seal_sha256=protocol_seal_sha256,
    )
    assembled = assemble_rotation_commitments(
        spec,
        no_query,
        partitions["prediction"],
        partitions["random"],
        partitions["ceiling"][0],
        protocol_seal_sha256=protocol_seal_sha256,
        view_provenance=provenance,
    )
    leaves = _leaf_seal_sequence(
        commitment_leaf_seals,
        expected_count=11,
        label="rotation commitment leaves",
    )
    expected_leaves = _seal_digest_sequence(
        expected_commitment_leaf_seal_sha256s,
        expected_count=11,
        label="expected rotation commitment leaf seals",
    )
    rows: list[CommitmentIndexRow] = []
    for expected, leaf, expected_leaf_seal in zip(
        assembled,
        leaves,
        expected_leaves,
        strict=True,
    ):
        selector_kind = _selector_kind_for_run(expected.run)
        selector_seal = None if selector_kind is None else expected_selectors[selector_kind]
        decoded = _verify_pool_commitment_phase_capability_raw(
            leaf,
            run=expected.run,
            protocol_seal_sha256=protocol_seal_sha256,
            prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
            selector_output_seal_sha256=selector_seal,
            view_provenance=provenance,
            publication_identity=publication_identity,
            expected_seal_sha256=expected_leaf_seal,
        )
        if _commitment_bytes(decoded) != _commitment_bytes(expected):
            raise ValueError("commitment leaf differs from its authenticated selector output")
        rows.append(
            _commitment_index_row(
                decoded,
                leaf,
                selector_output_seal_sha256=selector_seal,
            )
        )
    return tuple(rows)


def _rotation_predecessors(
    *,
    spec: RotationSpec,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    prepare_evidence_seal_sha256: str,
    selector_seals: Mapping[str, str],
    commitment_rows: Sequence[CommitmentIndexRow],
) -> dict[str, str]:
    if set(selector_seals) != set(_SELECTOR_KINDS):
        raise ValueError("rotation predecessor selector seals are incomplete")
    rows = tuple(commitment_rows)
    if tuple(row.run for row in rows) != policy_runs_for_rotation(spec):
        raise ValueError("rotation commitment index differs from frozen track order")
    result = {
        _protocol_predecessor(): _sha256(protocol_seal_sha256, label="protocol seal"),
        _prepare_barrier_predecessor(): _sha256(
            prepare_barrier_seal_sha256,
            label="prepare barrier seal",
        ),
        _evidence_predecessor(spec): _sha256(
            prepare_evidence_seal_sha256,
            label="prepare evidence seal",
        ),
    }
    result.update(
        {
            _selector_predecessor(spec, kind): _sha256(
                selector_seals[kind],
                label=f"{kind} selector seal",
            )
            for kind in _SELECTOR_KINDS
        }
    )
    result.update({_commitment_predecessor(row.run): row.leaf_seal_sha256 for row in rows})
    if len(result) != 17:
        raise AssertionError("rotation predecessor census changed")
    return result


def _rotation_summary_document(
    spec: RotationSpec,
    rows: Sequence[CommitmentIndexRow],
    *,
    commitment_index_sha256: str,
    view_provenance_sha256: str,
) -> dict[str, object]:
    values = tuple(rows)
    if tuple(row.run for row in values) != policy_runs_for_rotation(spec):
        raise ValueError("rotation summary rows differ from frozen track order")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ROTATION_INDEX_ARTIFACT,
        "rotation_id": spec.rotation_id,
        "commitment_count": len(values),
        "selected_sequence_association_count": sum(row.selected_sequence_count for row in values),
        "prediction_commitment_count": sum(row.selector_kind == "prediction" for row in values),
        "random_commitment_count": sum(row.selector_kind == "random" for row in values),
        "ceiling_commitment_count": sum(row.selector_kind == "ceiling" for row in values),
        "no_query_commitment_count": sum(row.selector_kind is None for row in values),
        "commitment_index_sha256": _sha256(
            commitment_index_sha256,
            label="rotation commitment index",
        ),
        "view_provenance_sha256": _sha256(
            view_provenance_sha256,
            label="rotation view provenance",
        ),
    }


def _publish_rotation_commitment_index_raw(
    destination: str | Path,
    *,
    spec: RotationSpec,
    selector_phase_seals: Mapping[str, PhaseSeal],
    commitment_leaf_seals: Sequence[PhaseSeal],
    expected_selector_seal_sha256_by_kind: Mapping[str, str],
    expected_commitment_leaf_seal_sha256s: Sequence[str],
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    prepare_evidence_seal_sha256: str,
    view_provenance: RotationViewProvenance,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Publish one 11-track rotation index after all physical leaves verify."""

    rows = _rotation_commitment_rows(
        spec=spec,
        selector_phase_seals=selector_phase_seals,
        commitment_leaf_seals=commitment_leaf_seals,
        expected_selector_seal_sha256_by_kind=(expected_selector_seal_sha256_by_kind),
        expected_commitment_leaf_seal_sha256s=(expected_commitment_leaf_seal_sha256s),
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        view_provenance=view_provenance,
        publication_identity=publication_identity,
    )
    selectors = _selector_seal_mapping(selector_phase_seals)
    selector_digests = {kind: selectors[kind].seal_sha256 for kind in _SELECTOR_KINDS}
    commitment_payload = canonical_jsonl_bytes(row.document() for row in rows)
    provenance_payload = canonical_json_bytes(rotation_view_provenance_document(view_provenance))
    summary_payload = canonical_json_bytes(
        _rotation_summary_document(
            spec,
            rows,
            commitment_index_sha256=sha256_bytes(commitment_payload),
            view_provenance_sha256=sha256_bytes(provenance_payload),
        )
    )
    predecessors = _rotation_predecessors(
        spec=spec,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        prepare_evidence_seal_sha256=prepare_evidence_seal_sha256,
        selector_seals=selector_digests,
        commitment_rows=rows,
    )
    seal = publish_phase(
        destination,
        artifact=ROTATION_INDEX_ARTIFACT,
        payloads={
            "commitment-index.jsonl": commitment_payload,
            "rotation-summary.json": summary_payload,
            "view-provenance.json": provenance_payload,
        },
        predecessor_seals=predecessors,
        metadata=_identity_metadata(publication_identity, scope_id=spec.rotation_id),
    )
    verified_rows, _provenance = _verify_rotation_index_capability(
        seal,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        publication_identity=publication_identity,
        expected_seal_sha256=seal.seal_sha256,
        expected_evidence_seal_sha256=prepare_evidence_seal_sha256,
    )
    if verified_rows != rows:
        raise RuntimeError("published rotation index changed its authenticated rows")
    return seal


def publish_rotation_commitment_index(
    destination: str | Path,
    *,
    spec: RotationSpec,
    selector_phase_seals: Mapping[str, PhaseSeal],
    commitment_leaf_seals: Sequence[PhaseSeal],
    expected_selector_seal_sha256_by_kind: Mapping[str, str],
    expected_commitment_leaf_seal_sha256s: Sequence[str],
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Publish a rotation index using only the safe prepare-global capability."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    return _publish_rotation_commitment_index_raw(
        destination,
        spec=spec,
        selector_phase_seals=selector_phase_seals,
        commitment_leaf_seals=commitment_leaf_seals,
        expected_selector_seal_sha256_by_kind=(expected_selector_seal_sha256_by_kind),
        expected_commitment_leaf_seal_sha256s=(expected_commitment_leaf_seal_sha256s),
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        prepare_evidence_seal_sha256=campaign.evidence_seal_sha256(spec=spec),
        view_provenance=campaign.rotation_view_provenance(spec=spec),
        publication_identity=publication_identity,
    )


def _verify_rotation_index_capability(
    seal: PhaseSeal,
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
    expected_evidence_seal_sha256: str | None = None,
) -> tuple[tuple[CommitmentIndexRow, ...], RotationViewProvenance]:
    expected_seal = _sha256(expected_seal_sha256, label="rotation index seal")
    authenticated = verify_phase_capability(
        seal,
        expected_artifact=ROTATION_INDEX_ARTIFACT,
        expected_payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal,
    )
    preliminary_rows = tuple(
        _commitment_index_row_from_document(row)
        for row in _strict_jsonl(
            authenticated.read_payload_bytes("commitment-index.jsonl"),
            label="rotation commitment index",
        )
    )
    if not preliminary_rows:
        raise ValueError("rotation commitment index cannot be empty")
    spec = preliminary_rows[0].run.rotation
    if tuple(row.run for row in preliminary_rows) != policy_runs_for_rotation(spec):
        raise ValueError("rotation commitment rows differ from frozen track order")
    provenance = decode_rotation_view_provenance(
        authenticated.read_payload_bytes("view-provenance.json")
    )
    _validate_provenance(provenance, spec=spec)
    protocol = _sha256(protocol_seal_sha256, label="protocol seal")
    prepare = _sha256(prepare_barrier_seal_sha256, label="prepare barrier seal")
    selector_seals: dict[str, str] = {}
    for row in preliminary_rows:
        _validate_index_row_view_binding(
            row,
            protocol_seal_sha256=protocol,
            provenance=provenance,
        )
        if row.selector_kind is not None:
            assert row.selector_output_seal_sha256 is not None
            previous = selector_seals.setdefault(
                row.selector_kind,
                row.selector_output_seal_sha256,
            )
            if previous != row.selector_output_seal_sha256:
                raise ValueError("rotation index binds multiple seals for one selector")
    if set(selector_seals) != set(_SELECTOR_KINDS):
        raise ValueError("rotation index does not bind all three selector outputs")
    evidence_key = _evidence_predecessor(spec)
    predecessor_mapping = dict(authenticated.predecessor_seals)
    if expected_evidence_seal_sha256 is None:
        evidence_seal = _sha256(
            predecessor_mapping.get(evidence_key),
            label="rotation prepare evidence seal",
        )
    else:
        evidence_seal = _sha256(
            expected_evidence_seal_sha256,
            label="rotation prepare evidence seal",
        )
    predecessors = _rotation_predecessors(
        spec=spec,
        protocol_seal_sha256=protocol,
        prepare_barrier_seal_sha256=prepare,
        prepare_evidence_seal_sha256=evidence_seal,
        selector_seals=selector_seals,
        commitment_rows=preliminary_rows,
    )
    verified = verify_phase_capability(
        authenticated,
        expected_artifact=ROTATION_INDEX_ARTIFACT,
        expected_payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=expected_seal,
    )
    _verify_identity_metadata(verified, publication_identity, scope_id=spec.rotation_id)
    commitment_payload = verified.read_payload_bytes("commitment-index.jsonl")
    provenance_payload = verified.read_payload_bytes("view-provenance.json")
    expected_summary = _rotation_summary_document(
        spec,
        preliminary_rows,
        commitment_index_sha256=sha256_bytes(commitment_payload),
        view_provenance_sha256=sha256_bytes(provenance_payload),
    )
    summary_payload = verified.read_payload_bytes("rotation-summary.json")
    _strict_json(summary_payload, label="rotation commitment summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("rotation summary differs from its index and provenance")
    return preliminary_rows, provenance


def verify_rotation_commitment_index_capability(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[CommitmentIndexRow, ...]:
    """Authenticate one rootless rotation index and its exact 11 rows."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    rows, provenance = _verify_rotation_index_capability(
        seal,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_seal_sha256,
        expected_evidence_seal_sha256=campaign.evidence_seal_sha256(spec=spec),
    )
    if _view_provenance_bytes(provenance) != _view_provenance_bytes(
        campaign.rotation_view_provenance(spec=spec)
    ):
        raise ValueError("rotation index capability differs from expected rotation")
    return rows


@dataclass(frozen=True, slots=True)
class RotationCommitmentBundle:
    """Rootless result of one complete rotation commitment publication.

    This value is an immutable procedural result, not authority derived from a
    filesystem path.  It deliberately retains only the frozen rotation identity
    and the twelve descriptor-captured phase capabilities that the controller
    must independently anchor before global assembly.
    """

    spec: RotationSpec
    commitment_leaf_seals: tuple[PhaseSeal, ...]
    rotation_index_seal: PhaseSeal

    def __post_init__(self) -> None:
        spec = _require_frozen_rotation(self.spec, label="rotation commitment bundle spec")
        leaves = self.commitment_leaf_seals
        expected_runs = policy_runs_for_rotation(spec)
        if (
            type(leaves) is not tuple
            or len(leaves) != len(expected_runs)
            or any(type(item) is not PhaseSeal for item in leaves)
        ):
            raise ValueError("rotation commitment bundle requires eleven exact leaf seals")
        if type(self.rotation_index_seal) is not PhaseSeal:
            raise TypeError("rotation commitment bundle requires an exact index PhaseSeal")

        decoded_runs: list[PolicyRunSpec] = []
        for leaf in leaves:
            verified = verify_phase_capability(
                leaf,
                expected_artifact=POOL_COMMITMENT_ARTIFACT,
                expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
                expected_seal_sha256=leaf.seal_sha256,
            )
            decoded_runs.append(
                decode_pool_commitment(verified.read_payload_bytes("commitment.json")).run
            )
        if tuple(decoded_runs) != expected_runs:
            raise ValueError("rotation commitment bundle leaves differ from frozen track order")

        index = verify_phase_capability(
            self.rotation_index_seal,
            expected_artifact=ROTATION_INDEX_ARTIFACT,
            expected_payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
            expected_seal_sha256=self.rotation_index_seal.seal_sha256,
        )
        rows = tuple(
            _commitment_index_row_from_document(row)
            for row in _strict_jsonl(
                index.read_payload_bytes("commitment-index.jsonl"),
                label="rotation commitment bundle index",
            )
        )
        if tuple(row.run for row in rows) != expected_runs or tuple(
            row.leaf_seal_sha256 for row in rows
        ) != tuple(leaf.seal_sha256 for leaf in leaves):
            raise ValueError("rotation commitment bundle index differs from its ordered leaves")
        predecessors = dict(index.predecessor_seals)
        if any(
            predecessors.get(_commitment_predecessor(run)) != leaf.seal_sha256
            for run, leaf in zip(expected_runs, leaves, strict=True)
        ):
            raise ValueError("rotation commitment bundle index does not bind every leaf")


def _require_empty_rotation_bundle_destination(destination: str | Path) -> Path:
    root = Path(os.path.abspath(os.fspath(destination)))
    candidate = root
    try:
        while True:
            metadata = os.lstat(candidate)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("rotation commitment destination must not traverse a symlink")
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
    except FileNotFoundError as error:
        raise ValueError("rotation commitment destination must be an existing directory") from error
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("rotation commitment destination must be an existing real directory")
    with os.scandir(root) as entries:
        if next(entries, None) is not None:
            raise FileExistsError("refusing to reuse nonempty rotation commitment destination")
    return root


def publish_rotation_commitment_bundle(
    destination: str | Path,
    *,
    spec: RotationSpec,
    selector_phase_seals: Mapping[str, PhaseSeal],
    expected_selector_seal_sha256_by_kind: Mapping[str, str],
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> RotationCommitmentBundle:
    """Publish all eleven commitment leaves and then their rotation index.

    ``destination`` is an existing empty private worker output directory.  All
    three selector capabilities and their controller-authoritative digests are
    authenticated and decoded before the directory is touched.  Any later
    failure can therefore leave only a non-authoritative prefix: ``global`` is
    published last, and every constituent phase uses no-replace publication.
    """

    frozen_spec = _require_frozen_rotation(spec, label="rotation commitment bundle spec")
    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    selectors = _selector_seal_mapping(selector_phase_seals)
    expected_selectors = _selector_digest_mapping(
        expected_selector_seal_sha256_by_kind,
        label="expected rotation selector seals",
    )
    provenance = campaign.rotation_view_provenance(spec=frozen_spec)
    partitions = {
        kind: _verify_selector_commitment_phase_capability_raw(
            selectors[kind],
            spec=frozen_spec,
            selector_kind=kind,
            protocol_seal_sha256=campaign.protocol_seal_sha256,
            prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
            view_provenance=provenance,
            publication_identity=publication_identity,
            expected_seal_sha256=expected_selectors[kind],
        )
        for kind in _SELECTOR_KINDS
    }
    commitments = assemble_rotation_commitments(
        frozen_spec,
        make_no_query_commitment(
            frozen_spec,
            protocol_seal_sha256=campaign.protocol_seal_sha256,
        ),
        partitions["prediction"],
        partitions["random"],
        partitions["ceiling"][0],
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        view_provenance=provenance,
    )

    root = _require_empty_rotation_bundle_destination(destination)
    commitment_root = root / "commitments"
    commitment_root.mkdir(mode=0o700)
    leaves: list[PhaseSeal] = []
    for commitment in commitments:
        selector_kind = _selector_kind_for_run(commitment.run)
        leaves.append(
            _publish_pool_commitment_raw(
                commitment_root / commitment.run.track_id,
                commitment,
                protocol_seal_sha256=campaign.protocol_seal_sha256,
                prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
                selector_output_seal_sha256=(
                    None if selector_kind is None else expected_selectors[selector_kind]
                ),
                view_provenance=provenance,
                publication_identity=publication_identity,
            )
        )
    ordered_leaves = tuple(leaves)
    expected_leaf_digests = tuple(leaf.seal_sha256 for leaf in ordered_leaves)
    with os.scandir(commitment_root) as entries:
        actual_names = tuple(sorted(entry.name for entry in entries))
    expected_names = tuple(sorted(run.track_id for run in policy_runs_for_rotation(frozen_spec)))
    if actual_names != expected_names:
        raise RuntimeError("rotation commitment destination has an unexpected leaf inventory")
    index = _publish_rotation_commitment_index_raw(
        root / "global",
        spec=frozen_spec,
        selector_phase_seals=selectors,
        commitment_leaf_seals=ordered_leaves,
        expected_selector_seal_sha256_by_kind=expected_selectors,
        expected_commitment_leaf_seal_sha256s=expected_leaf_digests,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        prepare_evidence_seal_sha256=campaign.evidence_seal_sha256(spec=frozen_spec),
        view_provenance=provenance,
        publication_identity=publication_identity,
    )
    return RotationCommitmentBundle(
        spec=frozen_spec,
        commitment_leaf_seals=ordered_leaves,
        rotation_index_seal=index,
    )


def _validate_index_row_view_binding(
    row: CommitmentIndexRow,
    *,
    protocol_seal_sha256: str,
    provenance: RotationViewProvenance,
) -> None:
    if row.run.rotation != provenance.spec:
        raise ValueError("commitment index row and provenance rotations differ")
    kind = row.selector_kind
    if kind is None:
        expected = InputViewBinding(
            kind="none",
            phase_seal_sha256=protocol_seal_sha256,
            payload_sha256=None,
            candidate_count=0,
            candidate_ids_sha256=_EMPTY_ID_STREAM_SHA256,
        )
    else:
        prediction = kind == "prediction"
        expected = InputViewBinding(
            kind=(
                "prediction"
                if prediction
                else "random_minimal"
                if kind == "random"
                else "ceiling_minimal"
            ),
            phase_seal_sha256=(
                provenance.prediction_view_seal_sha256
                if prediction
                else provenance.random_view_seal_sha256
            ),
            payload_sha256=(
                provenance.prediction_view_payload_sha256
                if prediction
                else provenance.random_view_payload_sha256
            ),
            candidate_count=provenance.candidate_count,
            candidate_ids_sha256=provenance.candidate_ids_sha256,
        )
    if row.input_view != expected:
        raise ValueError("commitment index row differs from rotation view provenance")


def _campaign_rotation_view_provenance(
    commitment_rows: Sequence[CommitmentIndexRow],
    *,
    spec: RotationSpec,
    protocol_seal_sha256: str,
) -> RotationViewProvenance:
    """Recover and validate one rotation's common views from its global rows."""

    _require_frozen_rotation(spec, label="campaign view rotation")
    rows = tuple(commitment_rows)
    if any(type(row) is not CommitmentIndexRow for row in rows):
        raise TypeError("campaign view validation requires exact commitment index rows")
    if tuple(row.run for row in rows) != policy_runs_for_rotation(spec):
        raise ValueError("campaign view rows differ from frozen rotation track order")
    by_kind = {
        kind: tuple(row for row in rows if row.selector_kind == kind)
        for kind in (None, *_SELECTOR_KINDS)
    }
    if (
        len(by_kind[None]) != 1
        or len(by_kind["prediction"]) != len(_selector_expected_runs(spec, "prediction"))
        or len(by_kind["random"]) != len(_selector_expected_runs(spec, "random"))
        or len(by_kind["ceiling"]) != len(_selector_expected_runs(spec, "ceiling"))
    ):
        raise ValueError("campaign view rows differ from the frozen selector partition")
    for kind in _SELECTOR_KINDS:
        selector_seals = {row.selector_output_seal_sha256 for row in by_kind[kind]}
        if len(selector_seals) != 1:
            raise ValueError("campaign rows bind multiple seals for one selector output")

    prediction_binding = by_kind["prediction"][0].input_view
    random_binding = by_kind["random"][0].input_view
    provenance = RotationViewProvenance(
        spec=spec,
        candidate_count=prediction_binding.candidate_count,
        candidate_ids_sha256=prediction_binding.candidate_ids_sha256,
        prediction_view_seal_sha256=prediction_binding.phase_seal_sha256,
        prediction_view_payload_sha256=_sha256(
            prediction_binding.payload_sha256,
            label="campaign prediction-view payload",
        ),
        random_view_seal_sha256=random_binding.phase_seal_sha256,
        random_view_payload_sha256=_sha256(
            random_binding.payload_sha256,
            label="campaign random-view payload",
        ),
    )
    _validate_provenance(provenance, spec=spec)
    protocol = _sha256(protocol_seal_sha256, label="campaign protocol seal")
    for row in rows:
        _validate_index_row_view_binding(
            row,
            protocol_seal_sha256=protocol,
            provenance=provenance,
        )
    return provenance


def _campaign_view_provenances(
    commitment_rows: Sequence[CommitmentIndexRow],
    *,
    protocol_seal_sha256: str,
) -> tuple[RotationViewProvenance, ...]:
    rows = tuple(commitment_rows)
    if len(rows) != EXPECTED_POLICY_RUNS or any(
        type(row) is not CommitmentIndexRow for row in rows
    ):
        raise ValueError("campaign view validation requires the exact 220 commitment rows")
    result: list[RotationViewProvenance] = []
    for spec in ordered_rotations():
        rotation_rows = tuple(row for row in rows if row.run.rotation == spec)
        result.append(
            _campaign_rotation_view_provenance(
                rotation_rows,
                spec=spec,
                protocol_seal_sha256=protocol_seal_sha256,
            )
        )
    return tuple(result)


def _view_provenance_bytes(provenance: RotationViewProvenance) -> bytes:
    return canonical_json_bytes(rotation_view_provenance_document(provenance))


@dataclass(frozen=True, slots=True)
class RotationIndexRow:
    """Exact campaign record for one authenticated rotation index phase."""

    spec: RotationSpec
    phase_seal_sha256: str
    commitment_count: int
    selected_sequence_association_count: int

    def __post_init__(self) -> None:
        _require_frozen_rotation(self.spec, label="rotation index row spec")
        _sha256(self.phase_seal_sha256, label="rotation index phase seal")
        if type(self.commitment_count) is not int or self.commitment_count != 11:
            raise ValueError("rotation index row must bind exactly 11 commitments")
        expected = 90 + EXPECTED_SUPPORT_BY_FOLD[self.spec.pool_fold]
        if (
            type(self.selected_sequence_association_count) is not int
            or self.selected_sequence_association_count != expected
        ):
            raise ValueError("rotation index row association census changed")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation_id": self.spec.rotation_id,
            "relative_path": rotation_commitment_index_relative_path(self.spec),
            "artifact": ROTATION_INDEX_ARTIFACT,
            "phase_seal_sha256": self.phase_seal_sha256,
            "commitment_count": self.commitment_count,
            "selected_sequence_association_count": (self.selected_sequence_association_count),
        }


def _rotation_index_row_from_document(value: object) -> RotationIndexRow:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "rotation_id",
            "relative_path",
            "artifact",
            "phase_seal_sha256",
            "commitment_count",
            "selected_sequence_association_count",
        },
        label="campaign rotation index row",
    )
    if _integer(raw["schema_version"], label="rotation index schema_version") != 1:
        raise ValueError("rotation index schema_version must be one")
    spec = rotation_by_id(_text(raw["rotation_id"], label="rotation index rotation_id"))
    if (
        raw["relative_path"] != rotation_commitment_index_relative_path(spec)
        or type(raw["relative_path"]) is not str
        or raw["artifact"] != ROTATION_INDEX_ARTIFACT
        or type(raw["artifact"]) is not str
    ):
        raise ValueError("campaign rotation index identity or path is noncanonical")
    row = RotationIndexRow(
        spec=spec,
        phase_seal_sha256=_sha256(
            raw["phase_seal_sha256"],
            label="rotation index phase seal",
        ),
        commitment_count=_integer(
            raw["commitment_count"],
            label="rotation index commitment count",
        ),
        selected_sequence_association_count=_integer(
            raw["selected_sequence_association_count"],
            label="rotation index association count",
        ),
    )
    if canonical_json_bytes(row.document()) != canonical_json_bytes(raw):
        raise ValueError("campaign rotation index row does not round-trip exactly")
    return row


def _campaign_predecessors(
    commitment_rows: Sequence[CommitmentIndexRow],
    rotation_rows: Sequence[RotationIndexRow],
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
) -> dict[str, str]:
    commitments = tuple(commitment_rows)
    rotations = tuple(rotation_rows)
    if tuple(row.run for row in commitments) != ordered_policy_runs():
        raise ValueError("campaign commitment rows differ from frozen track order")
    if tuple(row.spec for row in rotations) != ordered_rotations():
        raise ValueError("campaign rotation rows differ from frozen rotation order")
    result = {
        _protocol_predecessor(): _sha256(protocol_seal_sha256, label="protocol seal"),
        _prepare_barrier_predecessor(): _sha256(
            prepare_barrier_seal_sha256,
            label="prepare barrier seal",
        ),
    }
    result.update({_rotation_predecessor(row.spec): row.phase_seal_sha256 for row in rotations})
    result.update({_commitment_predecessor(row.run): row.leaf_seal_sha256 for row in commitments})
    if len(result) != 242:
        raise AssertionError("campaign predecessor census changed")
    return result


def _validate_campaign_rows(
    commitment_rows: Sequence[CommitmentIndexRow],
    rotation_rows: Sequence[RotationIndexRow],
) -> None:
    commitments = tuple(commitment_rows)
    rotations = tuple(rotation_rows)
    if (
        len(commitments) != EXPECTED_POLICY_RUNS
        or tuple(row.run for row in commitments) != ordered_policy_runs()
    ):
        raise ValueError("campaign commitment index is not the exact 220-track order")
    if (
        len(rotations) != EXPECTED_ROTATIONS
        or tuple(row.spec for row in rotations) != ordered_rotations()
    ):
        raise ValueError("campaign rotation index is not the exact 20-rotation order")
    for rotation_row in rotations:
        subset = tuple(row for row in commitments if row.run.rotation == rotation_row.spec)
        if (
            len(subset) != rotation_row.commitment_count
            or sum(row.selected_sequence_count for row in subset)
            != rotation_row.selected_sequence_association_count
        ):
            raise ValueError("campaign indices disagree on a rotation census")
    if sum(row.selected_sequence_count for row in commitments) != (
        EXPECTED_POOL_COMMITTED_ASSOCIATIONS
    ):
        raise ValueError("campaign commitment association census differs from 4400")


def _campaign_summary_document(
    commitment_rows: Sequence[CommitmentIndexRow],
    rotation_rows: Sequence[RotationIndexRow],
    *,
    commitment_index_sha256: str,
    rotation_index_sha256: str,
) -> dict[str, object]:
    commitments = tuple(commitment_rows)
    rotations = tuple(rotation_rows)
    _validate_campaign_rows(commitments, rotations)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": CAMPAIGN_BARRIER_ARTIFACT,
        "rotation_count": len(rotations),
        "commitment_count": len(commitments),
        "no_query_commitment_count": sum(row.selector_kind is None for row in commitments),
        "prediction_commitment_count": sum(
            row.selector_kind == "prediction" for row in commitments
        ),
        "random_commitment_count": sum(row.selector_kind == "random" for row in commitments),
        "ceiling_commitment_count": sum(row.selector_kind == "ceiling" for row in commitments),
        "budgeted_selected_sequence_association_count": sum(
            row.selected_sequence_count
            for row in commitments
            if row.run.selection_kind == "budgeted"
        ),
        "ceiling_selected_sequence_association_count": sum(
            row.selected_sequence_count for row in commitments if row.run.policy == CEILING
        ),
        "selected_sequence_association_count": sum(
            row.selected_sequence_count for row in commitments
        ),
        "commitment_index_sha256": _sha256(
            commitment_index_sha256,
            label="campaign commitment index",
        ),
        "rotation_index_sha256": _sha256(
            rotation_index_sha256,
            label="campaign rotation index",
        ),
    }


def _verify_pool_leaf_against_index_row(
    seal: PhaseSeal,
    row: CommitmentIndexRow,
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PoolCommitment:
    predecessors = _pool_commitment_predecessors(
        row.run,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        input_view=row.input_view,
        selector_output_seal_sha256=row.selector_output_seal_sha256,
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=POOL_COMMITMENT_ARTIFACT,
        expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=row.leaf_seal_sha256,
    )
    _verify_identity_metadata(
        verified,
        publication_identity,
        scope_id=row.run.track_id,
    )
    if _payload_digest(verified, "commitment.json") != row.commitment_payload_sha256:
        raise ValueError("commitment index payload digest differs from its leaf")
    commitment = decode_pool_commitment(verified.read_payload_bytes("commitment.json"))
    if commitment.run != row.run or commitment.input_view != row.input_view:
        raise ValueError("commitment index identity differs from its leaf")
    if (
        len(commitment.selected_sequence_ids) != row.selected_sequence_count
        or ordered_id_stream_sha256(commitment.selected_sequence_ids)
        != row.selected_sequence_ids_sha256
    ):
        raise ValueError("commitment index selection identity differs from its leaf")
    return commitment


def _publish_pool_commitment_campaign_barrier_raw(
    destination: str | Path,
    *,
    rotation_index_seals: Sequence[PhaseSeal],
    commitment_leaf_seals: Sequence[PhaseSeal],
    expected_rotation_index_seal_sha256s: Sequence[str],
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Publish the 242-predecessor barrier authorizing pool reveal."""

    rotations = _leaf_seal_sequence(
        rotation_index_seals,
        expected_count=EXPECTED_ROTATIONS,
        label="campaign rotation indices",
    )
    expected_rotation_seals = _seal_digest_sequence(
        expected_rotation_index_seal_sha256s,
        expected_count=EXPECTED_ROTATIONS,
        label="expected campaign rotation index seals",
    )
    leaves = _leaf_seal_sequence(
        commitment_leaf_seals,
        expected_count=EXPECTED_POLICY_RUNS,
        label="campaign commitment leaves",
    )
    commitment_rows: list[CommitmentIndexRow] = []
    rotation_rows: list[RotationIndexRow] = []
    for expected_spec, rotation_seal, expected_rotation_seal in zip(
        ordered_rotations(),
        rotations,
        expected_rotation_seals,
        strict=True,
    ):
        rows, provenance = _verify_rotation_index_capability(
            rotation_seal,
            protocol_seal_sha256=protocol_seal_sha256,
            prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
            publication_identity=publication_identity,
            expected_seal_sha256=expected_rotation_seal,
        )
        if provenance.spec != expected_spec:
            raise ValueError("campaign rotation seal order differs from frozen rotations")
        commitment_rows.extend(rows)
        rotation_rows.append(
            RotationIndexRow(
                spec=expected_spec,
                phase_seal_sha256=rotation_seal.seal_sha256,
                commitment_count=len(rows),
                selected_sequence_association_count=sum(
                    row.selected_sequence_count for row in rows
                ),
            )
        )
    rows_tuple = tuple(commitment_rows)
    rotation_tuple = tuple(rotation_rows)
    _validate_campaign_rows(rows_tuple, rotation_tuple)
    for row, leaf in zip(rows_tuple, leaves, strict=True):
        _verify_pool_leaf_against_index_row(
            leaf,
            row,
            protocol_seal_sha256=protocol_seal_sha256,
            prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
            publication_identity=publication_identity,
        )
    commitment_payload = canonical_jsonl_bytes(row.document() for row in rows_tuple)
    rotation_payload = canonical_jsonl_bytes(row.document() for row in rotation_tuple)
    summary_payload = canonical_json_bytes(
        _campaign_summary_document(
            rows_tuple,
            rotation_tuple,
            commitment_index_sha256=sha256_bytes(commitment_payload),
            rotation_index_sha256=sha256_bytes(rotation_payload),
        )
    )
    predecessors = _campaign_predecessors(
        rows_tuple,
        rotation_tuple,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
    )
    seal = publish_phase(
        destination,
        artifact=CAMPAIGN_BARRIER_ARTIFACT,
        payloads={
            "campaign-summary.json": summary_payload,
            "commitment-index.jsonl": commitment_payload,
            "rotation-index.jsonl": rotation_payload,
        },
        predecessor_seals=predecessors,
        metadata=_identity_metadata(publication_identity, scope_id="global"),
    )
    _verify_pool_commitment_campaign_barrier_capability_raw(
        seal,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        publication_identity=publication_identity,
        expected_seal_sha256=seal.seal_sha256,
    )
    return seal


def publish_pool_commitment_campaign_barrier(
    destination: str | Path,
    *,
    rotation_index_seals: Sequence[PhaseSeal],
    commitment_leaf_seals: Sequence[PhaseSeal],
    expected_rotation_index_seal_sha256s: Sequence[str],
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> PhaseSeal:
    """Publish the global select barrier through authenticated prepare authority."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    rotation_seals = _leaf_seal_sequence(
        rotation_index_seals,
        expected_count=EXPECTED_ROTATIONS,
        label="campaign rotation indices",
    )
    expected_rotation_seals = _seal_digest_sequence(
        expected_rotation_index_seal_sha256s,
        expected_count=EXPECTED_ROTATIONS,
        label="expected campaign rotation index seals",
    )
    for spec, rotation_seal, expected_rotation_seal in zip(
        ordered_rotations(),
        rotation_seals,
        expected_rotation_seals,
        strict=True,
    ):
        _rows, provenance = _verify_rotation_index_capability(
            rotation_seal,
            protocol_seal_sha256=campaign.protocol_seal_sha256,
            prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
            publication_identity=publication_identity,
            expected_seal_sha256=expected_rotation_seal,
            expected_evidence_seal_sha256=campaign.evidence_seal_sha256(spec=spec),
        )
        if _view_provenance_bytes(provenance) != _view_provenance_bytes(
            campaign.rotation_view_provenance(spec=spec)
        ):
            raise ValueError("rotation index differs from prepare-global provenance")
    return _publish_pool_commitment_campaign_barrier_raw(
        destination,
        rotation_index_seals=rotation_seals,
        commitment_leaf_seals=commitment_leaf_seals,
        expected_rotation_index_seal_sha256s=expected_rotation_seals,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        publication_identity=publication_identity,
    )


def _decode_campaign_indices(
    seal: PhaseSeal,
) -> tuple[tuple[CommitmentIndexRow, ...], tuple[RotationIndexRow, ...]]:
    commitment_rows = tuple(
        _commitment_index_row_from_document(row)
        for row in _strict_jsonl(
            seal.read_payload_bytes("commitment-index.jsonl"),
            label="campaign commitment index",
        )
    )
    rotation_rows = tuple(
        _rotation_index_row_from_document(row)
        for row in _strict_jsonl(
            seal.read_payload_bytes("rotation-index.jsonl"),
            label="campaign rotation index",
        )
    )
    _validate_campaign_rows(commitment_rows, rotation_rows)
    return commitment_rows, rotation_rows


def _verify_pool_commitment_campaign_barrier_capability_raw(
    seal: PhaseSeal,
    *,
    protocol_seal_sha256: str,
    prepare_barrier_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[CommitmentIndexRow, ...]:
    """Authenticate the complete label-free global selection barrier."""

    expected_seal = _sha256(expected_seal_sha256, label="campaign barrier seal")
    authenticated = verify_phase_capability(
        seal,
        expected_artifact=CAMPAIGN_BARRIER_ARTIFACT,
        expected_payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal,
    )
    preliminary_commitments, preliminary_rotations = _decode_campaign_indices(authenticated)
    _campaign_view_provenances(
        preliminary_commitments,
        protocol_seal_sha256=protocol_seal_sha256,
    )
    predecessors = _campaign_predecessors(
        preliminary_commitments,
        preliminary_rotations,
        protocol_seal_sha256=protocol_seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
    )
    verified = verify_phase_capability(
        authenticated,
        expected_artifact=CAMPAIGN_BARRIER_ARTIFACT,
        expected_payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=expected_seal,
    )
    _verify_identity_metadata(verified, publication_identity, scope_id="global")
    commitment_rows, rotation_rows = _decode_campaign_indices(verified)
    commitment_payload = verified.read_payload_bytes("commitment-index.jsonl")
    rotation_payload = verified.read_payload_bytes("rotation-index.jsonl")
    expected_summary = _campaign_summary_document(
        commitment_rows,
        rotation_rows,
        commitment_index_sha256=sha256_bytes(commitment_payload),
        rotation_index_sha256=sha256_bytes(rotation_payload),
    )
    summary_payload = verified.read_payload_bytes("campaign-summary.json")
    _strict_json(summary_payload, label="campaign commitment summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("campaign summary differs from its authenticated indices")
    return commitment_rows


def verify_pool_commitment_campaign_barrier_capability(
    seal: PhaseSeal,
    *,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[CommitmentIndexRow, ...]:
    """Authenticate the global select barrier through prepare-global authority."""

    campaign = _verify_prepare_campaign_context(
        prepare_campaign,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    rows = _verify_pool_commitment_campaign_barrier_capability_raw(
        seal,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
        prepare_barrier_seal_sha256=campaign.seal.seal_sha256,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_seal_sha256,
    )
    actual_provenances = _campaign_view_provenances(
        rows,
        protocol_seal_sha256=campaign.protocol_seal_sha256,
    )
    expected_provenances = tuple(
        campaign.rotation_view_provenance(spec=spec) for spec in ordered_rotations()
    )
    if tuple(map(_view_provenance_bytes, actual_provenances)) != tuple(
        map(_view_provenance_bytes, expected_provenances)
    ):
        raise ValueError("campaign commitment views differ from prepare-global provenance")
    return rows


def verify_pool_commitment_campaign_barrier_for_reveal(
    seal: PhaseSeal,
    *,
    protocol_capability: ProtocolCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_seal_sha256: str,
) -> tuple[CommitmentIndexRow, ...]:
    """Authenticate select-global for reveal without a prepare payload capability.

    The controller-authoritative select-global digest is released only after
    :func:`verify_pool_commitment_campaign_barrier_capability` has checked the
    complete prepare-global provenance.  A reveal process therefore needs only
    that digest, the independently authenticated protocol, and the externally
    anchored prepare-global digest to reconstruct the exact label-free barrier.
    """

    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("reveal barrier verification requires a ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    return _verify_pool_commitment_campaign_barrier_capability_raw(
        seal,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        prepare_barrier_seal_sha256=_sha256(
            expected_prepare_campaign_seal_sha256,
            label="prepare campaign seal",
        ),
        publication_identity=publication_identity,
        expected_seal_sha256=expected_seal_sha256,
    )


def _campaign_anchor_digests(seal: PhaseSeal) -> tuple[str, str]:
    authenticated = verify_phase_capability(
        seal,
        expected_artifact=CAMPAIGN_BARRIER_ARTIFACT,
        expected_payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
        expected_seal_sha256=seal.seal_sha256,
    )
    predecessors = dict(authenticated.predecessor_seals)
    return (
        _sha256(predecessors.get(_protocol_predecessor()), label="protocol seal"),
        _sha256(
            predecessors.get(_prepare_barrier_predecessor()),
            label="prepare barrier seal",
        ),
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedPoolCommitmentCapability:
    """Structurally verified pathless one-track commitment capsule.

    Construction proves internal seal and inclusion consistency.  A fresh
    downstream process must call :func:`verify_pool_commitment_capability` with
    an authenticated protocol capability plus controller-authoritative prepare-
    and selection-barrier digests before treating the capsule as reveal
    authority.  It never needs a prepare-global payload capability.
    """

    run: PolicyRunSpec
    commitment: PoolCommitment
    commitment_leaf: PhaseSeal
    selection_barrier: PhaseSeal
    selection_barrier_seal_sha256: str
    commitment_index_payload_sha256: str
    commitment_row_sha256: str
    commitment_row_index: int
    publication_identity: SequentialV2PublicationIdentity

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="commitment capability run")
        _require_exact_commitment_graph(self.commitment)
        if self.commitment.run != self.run:
            raise ValueError("commitment capability run and commitment disagree")
        if (
            type(self.commitment_leaf) is not PhaseSeal
            or type(self.selection_barrier) is not PhaseSeal
        ):
            raise TypeError("commitment capability requires two rootless PhaseSeal values")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("commitment capability requires a publication identity")
        barrier_seal = _sha256(
            self.selection_barrier_seal_sha256,
            label="commitment capability campaign barrier",
        )
        if barrier_seal != self.selection_barrier.seal_sha256:
            raise ValueError("commitment capability campaign barrier identity changed")
        protocol, prepare = _campaign_anchor_digests(self.selection_barrier)
        rows = _verify_pool_commitment_campaign_barrier_capability_raw(
            self.selection_barrier,
            protocol_seal_sha256=protocol,
            prepare_barrier_seal_sha256=prepare,
            publication_identity=self.publication_identity,
            expected_seal_sha256=barrier_seal,
        )
        if type(self.commitment_row_index) is not int or not (
            0 <= self.commitment_row_index < len(rows)
        ):
            raise ValueError("commitment capability row index is out of range")
        expected_global_index = ordered_policy_runs().index(self.run)
        if self.commitment_row_index != expected_global_index:
            raise ValueError("commitment capability row index differs from frozen run order")
        row = rows[self.commitment_row_index]
        if row.run != self.run:
            raise ValueError("commitment capability row does not resolve to its run")
        index_payload_sha256 = _payload_digest(
            self.selection_barrier,
            "commitment-index.jsonl",
        )
        if (
            _sha256(
                self.commitment_index_payload_sha256,
                label="commitment capability index payload",
            )
            != index_payload_sha256
        ):
            raise ValueError("commitment capability index payload identity changed")
        row_sha256 = sha256_bytes(canonical_json_bytes(row.document()))
        if (
            _sha256(
                self.commitment_row_sha256,
                label="commitment capability row",
            )
            != row_sha256
        ):
            raise ValueError("commitment capability row identity changed")
        decoded = _verify_pool_leaf_against_index_row(
            self.commitment_leaf,
            row,
            protocol_seal_sha256=protocol,
            prepare_barrier_seal_sha256=prepare,
            publication_identity=self.publication_identity,
        )
        if _commitment_bytes(decoded) != _commitment_bytes(self.commitment):
            raise ValueError("commitment capability value differs from its sealed leaf")


def pool_commitment_capability_from_seals(
    commitment_leaf: PhaseSeal,
    selection_barrier: PhaseSeal,
    *,
    run: PolicyRunSpec,
    protocol_capability: ProtocolCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_selection_barrier_seal_sha256: str,
) -> AuthenticatedPoolCommitmentCapability:
    """Derive a least-capability reveal authorization from two rootless seals."""

    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("commitment capability derivation requires a ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    prepare_barrier_seal_sha256 = _sha256(
        expected_prepare_campaign_seal_sha256,
        label="prepare campaign seal",
    )
    rows = _verify_pool_commitment_campaign_barrier_capability_raw(
        selection_barrier,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    _require_frozen_run(run, label="commitment capability derivation run")
    index = ordered_policy_runs().index(run)
    row = rows[index]
    commitment = _verify_pool_leaf_against_index_row(
        commitment_leaf,
        row,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        prepare_barrier_seal_sha256=prepare_barrier_seal_sha256,
        publication_identity=publication_identity,
    )
    return AuthenticatedPoolCommitmentCapability(
        run=run,
        commitment=commitment,
        commitment_leaf=commitment_leaf,
        selection_barrier=selection_barrier,
        selection_barrier_seal_sha256=selection_barrier.seal_sha256,
        commitment_index_payload_sha256=_payload_digest(
            selection_barrier,
            "commitment-index.jsonl",
        ),
        commitment_row_sha256=sha256_bytes(canonical_json_bytes(row.document())),
        commitment_row_index=index,
        publication_identity=publication_identity,
    )


def verify_pool_commitment_capability(
    capability: AuthenticatedPoolCommitmentCapability,
    *,
    protocol_capability: ProtocolCapability,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    expected_selection_barrier_seal_sha256: str,
) -> AuthenticatedPoolCommitmentCapability:
    """Re-derive one capsule under fresh controller-authoritative digests."""

    if type(capability) is not AuthenticatedPoolCommitmentCapability:
        raise TypeError(
            "commitment capability verification requires an AuthenticatedPoolCommitmentCapability"
        )
    verified = pool_commitment_capability_from_seals(
        capability.commitment_leaf,
        capability.selection_barrier,
        run=capability.run,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        publication_identity=publication_identity,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
    )
    if (
        capability.run != verified.run
        or _commitment_bytes(capability.commitment) != _commitment_bytes(verified.commitment)
        or capability.commitment_leaf.seal_sha256 != verified.commitment_leaf.seal_sha256
        or capability.selection_barrier.seal_sha256 != verified.selection_barrier.seal_sha256
        or capability.selection_barrier_seal_sha256 != verified.selection_barrier_seal_sha256
        or capability.commitment_index_payload_sha256 != verified.commitment_index_payload_sha256
        or capability.commitment_row_sha256 != verified.commitment_row_sha256
        or capability.commitment_row_index != verified.commitment_row_index
        or capability.publication_identity != verified.publication_identity
    ):
        raise ValueError("commitment capability differs from its authoritative derivation")
    return verified


__all__ = [
    "CAMPAIGN_BARRIER_ARTIFACT",
    "CAMPAIGN_BARRIER_PAYLOAD_PATHS",
    "CEILING_SELECTOR_ARTIFACT",
    "POOL_COMMITMENT_ARTIFACT",
    "POOL_COMMITMENT_PAYLOAD_PATHS",
    "PREDICTION_SELECTOR_ARTIFACT",
    "RANDOM_SELECTOR_ARTIFACT",
    "ROTATION_INDEX_ARTIFACT",
    "ROTATION_INDEX_PAYLOAD_PATHS",
    "SELECTOR_PAYLOAD_PATHS",
    "AuthenticatedPoolCommitmentCapability",
    "CommitmentIndexRow",
    "RotationCommitmentBundle",
    "RotationIndexRow",
    "decode_input_view_binding",
    "decode_pool_commitment",
    "decode_rotation_view_provenance",
    "decode_selection_result",
    "pool_commitment_capability_from_seals",
    "pool_commitment_relative_path",
    "publish_ceiling_selector_commitment",
    "publish_pool_commitment",
    "publish_pool_commitment_campaign_barrier",
    "publish_prediction_selector_commitments",
    "publish_random_selector_commitments",
    "publish_rotation_commitment_bundle",
    "publish_rotation_commitment_index",
    "rotation_commitment_index_relative_path",
    "rotation_view_provenance_document",
    "verify_pool_commitment_campaign_barrier_capability",
    "verify_pool_commitment_campaign_barrier_for_reveal",
    "verify_pool_commitment_capability",
    "verify_pool_commitment_phase_capability",
    "verify_rotation_commitment_index_capability",
    "verify_selector_commitment_phase_capability",
]

"""Actual ten-student endpoint distillation with named, bounded KL guards.

No hidden oracle, path-importance teacher, production insertion or clock reset.
All proposals are private copies; publication belongs to the consuming driver.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import torch

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    _state_contract,
    _validated_transition_kernels,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_proposals import (
    replay_native_trace,
    sample_native_proposals,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    ENDPOINT_CONFIG_SHA256,
    REFERENCE_MEASURE,
    TRIPLES,
    EndpointTeacher,
    teacher_admission,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.diffusion.replay import summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string


@dataclass(frozen=True, slots=True)
class OperatorPathReport:
    """Callback-supplied conditional-operator measure, authenticated outside.

    This checks shape/identity only, not truth of another implementation. The
    controller pins the actual operator plan/code and independently audits paths.
    """

    source_sha256: str
    plan_sha256: str
    receipt_sha256: str
    old_model_sha256: str
    candidate_model_sha256: str
    reference_model_sha256: str
    path_count: int
    mean: float
    monte_carlo_standard_error: float
    active_transition_p99: float
    scope: str = "DECLARED_CONDITIONAL_OPERATOR_PATH_NOT_GLOBAL_SEARCH"

    def __post_init__(self):
        if (
            any(
                not hash_string(value)
                for value in (
                    self.source_sha256,
                    self.plan_sha256,
                    self.receipt_sha256,
                    self.old_model_sha256,
                    self.candidate_model_sha256,
                    self.reference_model_sha256,
                )
            )
            or type(self.path_count) is not int
            or not 2 <= self.path_count <= 128
        ):
            raise ValueError("operator path report identity/budget differs")
        if (
            any(
                type(value) is not float or not math.isfinite(value) or value < 0
                for value in (
                    self.mean,
                    self.monte_carlo_standard_error,
                    self.active_transition_p99,
                )
            )
            or self.scope != "DECLARED_CONDITIONAL_OPERATOR_PATH_NOT_GLOBAL_SEARCH"
        ):
            raise ValueError("operator path report numerical measure differs")


class OperatorPathGuard(Protocol):
    source_sha256: str
    plan_sha256: str

    def evaluate(
        self, old, candidate, reference, *, triple: str, candidate_index: int
    ) -> OperatorPathReport: ...


@dataclass(frozen=True, slots=True)
class SharedEndpointUpdate:
    record_json: str
    sha256: str
    status: str
    accepted: bool
    backtracks: int | None
    kl_enforced: bool
    operator_guard_present: bool
    campaign_eligible: bool = False
    scientific_evidence_accepted: bool = False
    production_eligible: bool = False


def endpoint_source_identities():
    root = Path(__file__).resolve().parents[4]
    files = ["configs/diffusion/native_shared_endpoint_v1.toml"] + [
        "src/amp_challenge/generators/diffusion/" + name + ".py"
        for name in (
            "native_shared_endpoint_records",
            "native_matched_feasibility",
            "native_matched_feasibility_verify",
            "native_evolution_math",
            "native_evolution_records",
            "native_shared_endpoint",
            "native_weighted_training",
            "native_endpoint",
            "native_proposals",
            "native_initialization",
            "native_baseline_operators",
            "model",
            "categorical",
            "subset_kernel",
            "replay",
        )
    ]
    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    if result[files[0]] != ENDPOINT_CONFIG_SHA256:
        raise ValueError("shared endpoint predeclared config bytes differ")
    return result


def replay_document(replay):
    return {
        "sha256": replay.sha256,
        "sequences": replay.sequences,
        "context_id": replay.context_id,
        "weights": replay.weights.tolist(),
        "states": [
            {"tokens": state.tokens.tolist(), "length": state.length, "level": state.level}
            for state in replay.states
        ],
    }


def gradient_identity(direction):
    result = hashlib.sha256()
    for name, values in direction.gradients:
        result.update(name.encode() + b"\0" + values.numpy().tobytes())
    return result.hexdigest()


def endpoint_rng(seed, *keys):
    return np.random.Generator(
        np.random.PCG64DXSM(int(_json_hash(["shared-endpoint-v1", seed, keys])[:32], 16))
    )


def selected_anchors(unit, seed):
    if len(unit.sequences) < 64:
        raise ValueError("each endpoint student needs64 distinct generator-only anchors")
    return tuple(
        sorted(
            unit.sequences, key=lambda seq: _json_hash([seed, unit.triple, "endpoint-anchor", seq])
        )[:64]
    )


def interpolate_policy(old, proposed, backtracks):
    """Exact proposed copy at b0; no synthetic aggregated-gradient round trip."""
    if type(backtracks) is not int or not 0 <= backtracks <= 8 or old.config != proposed.config:
        raise ValueError("endpoint aggregate displacement backtrack differs")
    if backtracks == 0:
        return copy.deepcopy(proposed).eval()
    result = copy.deepcopy(old).eval()
    proposed_parameters = dict(proposed.named_parameters())
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            parameter.add_(proposed_parameters[name] - parameter, alpha=2.0**-backtracks)
    return result


def fullmask_reference_paths(unit, candidate, *, seed, semantic_sha256, check):
    """Eight iid length/path draws under the named fullmask generator measure."""
    rng = endpoint_rng(seed, semantic_sha256, unit.triple, "fullmask-audit-lengths")
    parents = tuple(unit.sequences[int(rng.integers(len(unit.sequences)))] for _ in range(8))
    path_seed = int(
        _json_hash([seed, semantic_sha256, unit.triple, "fullmask-audit-paths"])[:16], 16
    )
    check()
    traces = sample_native_proposals(
        candidate,
        parents,
        start_levels=(candidate.config.levels,) * 8,
        seed=path_seed,
        ordinals=tuple(range(8)),
    )
    values, active = [], []
    for trace in traces:
        check()
        _, states = replay_native_trace(candidate, trace, authenticate_sampling=True)
        current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
        frozen = _validated_transition_kernels(unit.reference, states, NATIVE_ENDPOINT_DEFAULTS)
        terms = [
            complete_subset_commit_kl(left, right)
            for left, right in zip(current, frozen, strict=True)
        ]
        values.append(math.fsum(terms))
        active.extend(
            value
            for state, value in zip(states, terms, strict=True)
            if _state_contract(state, candidate.config)[1] > 0
        )
    array = np.asarray(values)
    if not active or not np.isfinite(array).all():
        raise FloatingPointError("invalid current-path reference diagnostics")
    return {
        "measure": REFERENCE_MEASURE,
        "path_count": 8,
        "path_values": values,
        "mean": float(array.mean()),
        "monte_carlo_standard_error": float(array.std(ddof=1) / math.sqrt(8)),
        "active_transitions": active,
        "active_summary": asdict(summarize_kl(active)),
        "traces": [asdict(trace) for trace in traces],
    }


def _operator_report(guard, expected_source, expected_plan, unit, candidate, backtracks, check):
    if guard is None:
        return None
    if (guard.source_sha256, guard.plan_sha256) != (expected_source, expected_plan):
        raise ValueError("operator guard source/plan changed before call")
    copies = tuple(copy.deepcopy(model) for model in (unit.model, candidate, unit.reference))
    identities = tuple(canonical_model_logical_hash(model) for model in copies)
    check()
    report = guard.evaluate(*copies, triple=unit.triple, candidate_index=backtracks)
    check()
    if type(report) is not OperatorPathReport:
        raise TypeError("operator guard returned wrong report type")
    report.__post_init__()
    if (guard.source_sha256, guard.plan_sha256) != (expected_source, expected_plan) or tuple(
        canonical_model_logical_hash(model) for model in copies
    ) != identities:
        raise ValueError("operator guard mutated source/plan/model state")
    if (report.source_sha256, report.plan_sha256) != (expected_source, expected_plan) or (
        report.old_model_sha256,
        report.candidate_model_sha256,
        report.reference_model_sha256,
    ) != identities:
        raise ValueError("operator guard report model/source identity differs")
    return asdict(report)


def _candidate_summary(rows):
    path_mean = math.fsum(row["fullmask_reference"]["mean"] for row in rows) / 10
    path_mcse = (
        math.sqrt(
            math.fsum(row["fullmask_reference"]["monte_carlo_standard_error"] ** 2 for row in rows)
        )
        / 10
    )
    active, frequencies = [], []
    for row in rows:
        values = row["fullmask_reference"]["active_transitions"]
        active.extend(values)
        frequencies.extend([1 / (10 * len(values))] * len(values))
    active_summary = asdict(summarize_kl(active, weights=frequencies))
    per_student_passed = all(
        row["local"]["local_old_candidate"]["mean"] <= 0.01
        and row["local"]["local_old_candidate"]["p99"] <= 0.02
        and row["fullmask_reference"]["mean"] <= 0.08
        and row["fullmask_reference"]["active_summary"]["p99"] <= 0.02
        and (
            row["operator"] is None
            or (
                row["operator"]["mean"] <= 0.08 and row["operator"]["active_transition_p99"] <= 0.02
            )
        )
        for row in rows
    )
    return {
        "reference_measure": REFERENCE_MEASURE,
        "equal_mixture_path_mean": path_mean,
        "stratified_mixture_mcse": path_mcse,
        "equal_component_active_transition_summary": active_summary,
        "per_student_and_mixture_kl_passed": per_student_passed
        and path_mean <= 0.08
        and active_summary["p99"] <= 0.02,
    }


def update_shared_endpoints(
    units,
    teacher: EndpointTeacher,
    *,
    seed: int,
    deadline: float,
    enforce_kl: bool = True,
    operator_guard: OperatorPathGuard | None = None,
    expected_operator_source: str | None = None,
    expected_operator_plan: str | None = None,
    clock=time.monotonic,
    matched_feasibility=None,
    _replay_stop=None,
):
    """Return private replacement units and receipt; originals never mutate.

    A short protected teacher is explicit no-update. Numerical/timeout/guard
    failures retain a bounded partial receipt and original ten units. A real
    caller owns hard preemption of blocking callbacks and durable publication.
    """
    started = clock()
    if (
        type(seed) is not int
        or not 0 <= seed < 2**63
        or type(enforce_kl) is not bool
        or not math.isfinite(deadline)
    ):
        raise ValueError("endpoint seed/mode/deadline differs")
    units = tuple(units)
    if tuple(unit.triple for unit in units) != TRIPLES:
        raise ValueError("shared endpoint update requires ordered ten audited students")
    for unit in units:
        unit.check()
    if type(teacher) is not EndpointTeacher:
        raise TypeError("endpoint teacher record differs")
    teacher.__post_init__()
    all_training = set().union(*(set(unit.initialization.training_sequence_ids) for unit in units))
    if all_training.intersection(target.sequence_id for target in teacher.targets):
        raise ValueError("endpoint target overlaps generator training namespace")
    if operator_guard is None:
        if expected_operator_source is not None or expected_operator_plan is not None:
            raise ValueError("operator source/plan provided without actual guard")
    elif not hash_string(expected_operator_source) or not hash_string(expected_operator_plan):
        raise ValueError("operator guard needs caller-pinned source and plan")
    from amp_challenge.generators.diffusion.native_matched_feasibility import (
        MatchedFeasibilityRequirement,
        NativeMatchedFeasibility,
    )

    if (
        matched_feasibility is not None
        and type(matched_feasibility) is not MatchedFeasibilityRequirement
    ):
        raise TypeError("matched feasibility requires the native public requirement")
    feasibility_requirement = (
        None if matched_feasibility is None else matched_feasibility.document()
    )
    if (
        feasibility_requirement is not None
        and matched_feasibility.context_sha256 != teacher.objective_context_sha256
    ):
        raise ValueError("matched feasibility teacher context differs")
    feasibility_guard = None
    sources = endpoint_source_identities()
    old_identities = tuple(unit.policy_sha256 for unit in units)
    admission, weights = teacher_admission(teacher)
    checks, stop_check = 0, None

    def check():
        nonlocal checks, stop_check
        checks += 1
        if _replay_stop == checks or (
            _replay_stop is None and clock() >= min(deadline, started + 180)
        ):
            stop_check = checks
            raise TimeoutError("shared endpoint original/wave deadline")

    training, candidates, replacement = [], [], units
    status, accepted, chosen = "insufficient_targets_no_update", False, None
    error = None
    try:
        check()
        if admission["admitted"]:
            proposed, probe_batches = [], []
            for unit in units:
                check()
                anchors = selected_anchors(unit, seed)
                sequences = anchors + tuple(target.sequence for target in teacher.targets)
                mixed_weights = np.concatenate([np.full(64, 0.5 / 64), 0.5 * weights])
                training_seed = int(
                    _json_hash([seed, teacher.semantic_sha256, unit.triple, "endpoint-training"])[
                        :16
                    ],
                    16,
                )
                work = unit.model
                unit_record = {"triple": unit.triple, "anchors": anchors, "steps": []}
                training.append(unit_record)
                for step in range(4):
                    check()
                    replay = build_weighted_replay(
                        work,
                        sequences,
                        mixed_weights,
                        context_id=teacher.objective_context_sha256,
                        seed=training_seed,
                        ordinal=step,
                    )
                    direction = propose_weighted_direction(work, replay)
                    work = endpoint_candidate(work, direction)
                    unit_record["steps"].append(
                        {
                            "replay": replay_document(replay),
                            "objective_before": direction.objective_before,
                            "gradient_norm_before_clip": direction.gradient_norm_before_clip,
                            "gradient_sha256": gradient_identity(direction),
                            "model_sha256": canonical_model_logical_hash(work),
                        }
                    )
                proposed.append(work)
                probes = build_weighted_replay(
                    unit.model,
                    sequences,
                    mixed_weights,
                    context_id=teacher.objective_context_sha256,
                    seed=training_seed,
                    ordinal=4,
                    active_probes=True,
                )
                probe_batches.append(probes)
                unit_record["probes"] = replay_document(probes)
                unit_record["proposed_model_sha256"] = canonical_model_logical_hash(work)
            if matched_feasibility is not None:
                check()
                if matched_feasibility.document() != feasibility_requirement:
                    raise ValueError("matched feasibility requirement changed during training")
                feasibility_guard = NativeMatchedFeasibility(
                    units,
                    proposed,
                    matched_feasibility.wave,
                    predicate=matched_feasibility.predicate,
                    predicate_sha256=matched_feasibility.predicate_sha256,
                    context_sha256=matched_feasibility.context_sha256,
                    seed=seed,
                    deadline=SimpleNamespace(check=lambda stage: check()),
                )
            for backtracks in range(9):
                candidate_record = {
                    "backtracks": backtracks,
                    "students": [],
                    "summary": None,
                    "matched_feasibility": None,
                }
                candidates.append(candidate_record)
                models = []
                for unit, proposal, probes in zip(units, proposed, probe_batches, strict=True):
                    check()
                    candidate = interpolate_policy(unit.model, proposal, backtracks)
                    models.append(candidate)
                    row = {
                        "triple": unit.triple,
                        "candidate_sha256": canonical_model_logical_hash(candidate),
                        "local": asdict(
                            weighted_anchor_diagnostics(
                                unit.model,
                                candidate,
                                unit.reference,
                                probes,
                                NATIVE_ENDPOINT_DEFAULTS,
                            )
                        ),
                        "fullmask_reference": None,
                        "operator": None,
                    }
                    candidate_record["students"].append(row)
                    row["fullmask_reference"] = fullmask_reference_paths(
                        unit,
                        candidate,
                        seed=seed,
                        semantic_sha256=teacher.semantic_sha256,
                        check=check,
                    )
                    row["operator"] = _operator_report(
                        operator_guard,
                        expected_operator_source,
                        expected_operator_plan,
                        unit,
                        candidate,
                        backtracks,
                        check,
                    )
                candidate_record["summary"] = _candidate_summary(candidate_record["students"])
                passed = candidate_record["summary"]["per_student_and_mixture_kl_passed"]
                check()
                feasible_passed = matched_feasibility is None
                if feasibility_guard is not None:
                    try:
                        report = feasibility_guard.evaluate(candidate_index=backtracks)
                    finally:
                        if feasibility_guard.records:
                            candidate_record["matched_feasibility"] = feasibility_guard.records[-1]
                    if tuple(row["candidate_model_sha256"] for row in report["students"]) != tuple(
                        canonical_model_logical_hash(model) for model in models
                    ):
                        raise ValueError("matched feasibility did not probe the candidate models")
                    if matched_feasibility.document() != feasibility_requirement:
                        raise ValueError("matched feasibility external requirement changed")
                    feasible_passed = report["summary"]["passed"] is True
                    check()
                if (not enforce_kl or passed) and feasible_passed:
                    replacements = []
                    for unit, model in zip(units, models, strict=True):
                        new = copy.copy(unit)
                        new.model, new.policy_sha256 = model, canonical_model_logical_hash(model)
                        replacements.append(new)
                    replacement = tuple(replacements)
                    accepted, chosen = True, backtracks
                    status = (
                        "accepted_guarded_update"
                        if enforce_kl
                        else "accepted_no_kl_enforcement_caps_recorded"
                    )
                    break
            if not accepted:
                status = "all_backtracks_rejected_ten_students_unchanged"
        check()
    except (TimeoutError, ValueError, FloatingPointError, RuntimeError, TypeError) as failure:
        if isinstance(failure, TimeoutError) and stop_check is None:
            # A caller-owned blocking guard may time out between our checkpoints.
            # Record the last checked boundary, without certifying external time.
            stop_check = checks
        replacement, accepted, chosen = units, False, None
        status = (
            "partial_deadline_no_commit"
            if isinstance(failure, TimeoutError)
            else "numerical_or_guard_failure_no_commit"
        )
        error = {"type": type(failure).__name__, "message": str(failure)[:512]}
    for unit in units:
        unit.check()
    if (
        tuple(unit.policy_sha256 for unit in units) != old_identities
        or endpoint_source_identities() != sources
    ):
        raise ValueError("endpoint original student/source identity changed")
    if stop_check is None:
        try:
            check()
        except TimeoutError as failure:
            replacement, accepted, chosen = units, False, None
            status, error = (
                "partial_deadline_no_commit",
                {"type": "TimeoutError", "message": str(failure)},
            )
    payload = {
        "artifact": "native_shared_endpoint_update_v2_matched_feasibility"
        if matched_feasibility is not None
        else "native_shared_endpoint_update_v1",
        "matched_feasibility_requirement": feasibility_requirement,
        "matched_feasibility_plan_sha256": None
        if feasibility_guard is None
        else feasibility_guard.plan_sha256,
        "config_sha256": ENDPOINT_CONFIG_SHA256,
        "source_identities": sources,
        "teacher": asdict(teacher),
        "teacher_sha256": teacher.sha256,
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "seed": seed,
        "old_models": [(unit.triple, unit.policy_sha256, unit.reference_sha256) for unit in units],
        "new_models": [(unit.triple, unit.policy_sha256) for unit in replacement],
        "admission": admission,
        "training": training,
        "candidates": candidates,
        "enforce_kl": enforce_kl,
        "operator_source_sha256": expected_operator_source,
        "operator_plan_sha256": expected_operator_plan,
        "operator_scope": "actual_operator_callback_supplied_external_authentication_required"
        if operator_guard is not None
        else "FULLMASK_GENERATOR_PATH_only_actual_operator_campaign_compatibility_unresolved",
        "status": status,
        "accepted": accepted,
        "backtracks": chosen,
        "error": error,
        "checks": checks,
        "stop_check": stop_check,
        "campaign_eligible": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 128 * 1024**2:
        raise ValueError("shared endpoint receipt exceeds128MiB")
    receipt_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    # Encoding and hashing belong to the original deadline too. A late failure
    # receipt may still be serialized, but it cannot authorize replacement units.
    if stop_check is None and clock() >= min(deadline, started + 180):
        checks += 1
        stop_check = checks
        replacement, accepted, chosen = units, False, None
        status = "partial_deadline_no_commit"
        payload.update(
            new_models=[(unit.triple, unit.policy_sha256) for unit in units],
            status=status,
            accepted=False,
            backtracks=None,
            checks=checks,
            stop_check=stop_check,
            error={
                "type": "TimeoutError",
                "message": "shared endpoint receipt serialization deadline",
            },
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 128 * 1024**2:
            raise ValueError("shared endpoint failure receipt exceeds128MiB")
        receipt_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    receipt = SharedEndpointUpdate(
        encoded,
        receipt_sha256,
        status,
        accepted,
        chosen,
        enforce_kl,
        operator_guard is not None,
    )
    return replacement, receipt

"""Private shared proposal plus actual partial guards; old trainer stays intact."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import NATIVE_ENDPOINT_DEFAULTS, _json_hash
from amp_challenge.generators.diffusion.native_ga_partial_guard import (
    GAPartialOperatorGuard,
    combined_summary,
)
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    CONFIG_SHA256,
    PartialGuardedUpdate,
    PartialGuardPlan,
    encode_record,
    read_clock,
    source_identities,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
from amp_challenge.generators.diffusion.native_shared_endpoint import (
    fullmask_reference_paths,
    interpolate_policy,
    update_shared_endpoints,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    EndpointTeacher,
    teacher_admission,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import _replay
from amp_challenge.generators.diffusion.native_weighted_training import weighted_anchor_diagnostics


def update_ga_partial_endpoints(
    units,
    teacher,
    plan,
    *,
    deadline,
    counter,
    enforce_kl=True,
    initial=False,
    clock=time.monotonic,
    matched_feasibility=None,
):
    """No tuple escapes as accepted until this wrapper's final source/time seal.

    counter must be the exact active counter covering every original/reference
    model, with a fresh zero ledger; existing work is never reset or subtracted.
    The old inner accepted flag is only an authenticated private proposal.
    """
    units = tuple(units)
    if (
        type(plan) is not PartialGuardPlan
        or type(teacher) is not EndpointTeacher
        or type(enforce_kl) is not bool
        or type(initial) is not bool
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
    ):
        raise TypeError("partial endpoint teacher/mode types differ")
    plan.__post_init__()
    teacher.__post_init__()
    if type(counter) is not NativeWorkCounter:
        raise TypeError("partial endpoint requires the exact native work counter")
    counter.require_active(units, fresh=True)
    if initial != (plan.round_index == 1):
        raise ValueError("partial endpoint initial generation differs")
    original = tuple(
        unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True)
    )
    if original != plan.units:
        raise ValueError("partial endpoint old state differs from external plan")
    from amp_challenge.generators.diffusion.native_ga_matched_feasibility import (
        GAMatchedFeasibility,
        GAMatchedRequirement,
    )

    if matched_feasibility is not None and type(matched_feasibility) is not GAMatchedRequirement:
        raise TypeError("partial update needs exact GA matched requirement")
    requirement = None if matched_feasibility is None else matched_feasibility.document()
    if (
        requirement is not None
        and matched_feasibility.context_sha256 != teacher.objective_context_sha256
    ):
        raise ValueError("GA matched requirement and teacher context differ")
    matched_guard = None
    sources = source_identities()
    if _json_hash(sources) != plan.source_sha256:
        raise ValueError("partial endpoint source plan differs")
    effective = min(float(deadline), plan.prefix_deadline, read_clock(clock) + 180)
    checks = 0
    work_before = counter.document()["total"]

    def check():
        nonlocal checks
        checks += 1
        if read_clock(clock) >= effective:
            raise TimeoutError("partial endpoint original wave deadline")
        observed = counter.document()["total"]
        if any(
            observed[key] - work_before[key] > cap
            for key, cap in (
                ("row_forwards", 390400 + (92160 if requirement is not None else 0)),
                ("forward_calls", 28880 + (10944 if requirement is not None else 0)),
                ("backward_calls", 320),
            )
        ):
            raise RuntimeError("partial endpoint complete neural work ceiling exceeded")

    payload = {
        "artifact": "native_ga_partial_update_v2",
        "matched_feasibility_requirement": requirement,
        "matched_feasibility_plan_sha256": None,
        "configuration_sha256": CONFIG_SHA256,
        "plan": asdict(plan),
        "plan_sha256": plan.sha256,
        "sources": sources,
        "teacher": asdict(teacher),
        "teacher_sha256": teacher.sha256,
        "initial": initial,
        "enforce_kl": enforce_kl,
        "admission": None,
        "private_shared_proposal": None,
        "old_occupancy": [],
        "candidates": [],
        "inner_accepted_is_arm_commit": False,
        "status": "not_started",
        "accepted": False,
        "updated": False,
        "backtracks": None,
        "error": None,
        "old_models": [unit.policy_sha256 for unit in units],
        "new_models": [unit.policy_sha256 for unit in units],
        "campaign_eligible": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }
    replacement, proposed = units, units
    guard = GAPartialOperatorGuard(plan, units, check=check, counter=counter)
    try:
        check()
        admission, _ = teacher_admission(teacher)
        payload["admission"] = admission
        training = not initial and admission["admitted"]
        raw = None
        if training:
            with counter.at("shared_private_proposal"):
                proposed, inner = update_shared_endpoints(
                    units,
                    teacher,
                    seed=plan.seed,
                    deadline=effective,
                    enforce_kl=False,
                    clock=lambda: read_clock(clock),
                )
            payload["private_shared_proposal"] = asdict(inner)
            if not inner.accepted:
                raise RuntimeError("private shared proposal failed: " + inner.status)
            raw = json.loads(inner.record_json)
            if inner.backtracks != 0 or len(raw["candidates"]) != 1:
                raise ValueError("private proposal must be the one unbacktracked four-step result")
        if training and matched_feasibility is not None:
            if matched_feasibility.document() != requirement:
                raise ValueError("GA matched requirement changed during proposal")
            matched_guard = GAMatchedFeasibility(
                units,
                tuple(unit.model for unit in proposed),
                plan,
                predicate=matched_feasibility.predicate,
                predicate_sha256=matched_feasibility.predicate_sha256,
                context_sha256=matched_feasibility.context_sha256,
                expected_source_sha256=matched_feasibility.source_sha256,
                counter=counter,
                check=check,
            )
            payload["matched_feasibility_plan_sha256"] = matched_guard.plan_sha256
        guard.prepare_old()
        payload["old_occupancy"] = guard.old_rows
        for b in range(9 if training else 1):
            entry = {
                "backtracks": b,
                "students": [],
                "partial": [],
                "summary": None,
                "matched_feasibility": None,
            }
            payload["candidates"].append(entry)
            models = []
            for index, unit in enumerate(units):
                check()
                candidate = (
                    interpolate_policy(unit.model, proposed[index].model, b)
                    if training
                    else unit.model
                )
                models.append(candidate)
                if training and b == 0:
                    shared = raw["candidates"][0]["students"][index]
                    if shared["candidate_sha256"] != canonical_model_logical_hash(candidate):
                        raise ValueError("private b0 model/diagnostic identity differs")
                else:
                    if training:
                        probes = _replay(raw["training"][index]["probes"])
                        with counter.at("outer_anchor_diagnostics"):
                            local = asdict(
                                weighted_anchor_diagnostics(
                                    unit.model,
                                    candidate,
                                    unit.reference,
                                    probes,
                                    NATIVE_ENDPOINT_DEFAULTS,
                                )
                            )
                    else:
                        # This is an exact unchanged-policy fact, not fabricated
                        # sampled anchor observations or a teacher-admission pass.
                        local = {
                            "local_old_candidate": {"mean": 0.0, "p99": 0.0, "sample_count": 0},
                            "scope": "analytic_identical_old_candidate_no_training",
                        }
                    with counter.at("outer_fullmask_diagnostics"):
                        fullmask = fullmask_reference_paths(
                            unit,
                            candidate,
                            seed=plan.seed,
                            semantic_sha256=teacher.semantic_sha256,
                            check=check,
                        )
                    shared = {
                        "triple": unit.triple,
                        "candidate_sha256": canonical_model_logical_hash(candidate),
                        "local": local,
                        "fullmask_reference": fullmask,
                        "operator": None,
                    }
                entry["students"].append(shared)
                partial = guard.evaluate(unit, candidate, candidate_index=b)
                entry["partial"].append(partial)
            entry["summary"] = combined_summary(entry["students"], entry["partial"], plan)
            check()
            feasible_passed = matched_feasibility is None
            if matched_guard is not None:
                try:
                    measured = matched_guard.evaluate(candidate_index=b)
                finally:
                    if matched_guard.records:
                        entry["matched_feasibility"] = matched_guard.records[-1]
                if tuple(row["candidate_model_sha256"] for row in measured["students"]) != tuple(
                    canonical_model_logical_hash(model) for model in models
                ):
                    raise ValueError("GA feasibility did not probe the candidate models")
                feasible_passed = measured["summary"]["passed"] is True
            elif matched_feasibility is not None:
                if training or any(
                    canonical_model_logical_hash(model) != unit.policy_sha256
                    for unit, model in zip(units, models, strict=True)
                ):
                    raise ValueError(
                        "GA feasibility identity proof requires untrained identical policies"
                    )
                entry["matched_feasibility"] = {
                    "status": "analytic_identical_policy_no_training",
                    "drop": 0.0,
                    "sampled_paths": 0,
                }
                feasible_passed = True
            if matched_feasibility is not None and matched_feasibility.document() != requirement:
                raise ValueError("GA matched requirement changed during diagnostics")
            check()
            if (not enforce_kl or entry["summary"]["all_kl_passed"]) and feasible_passed:
                copies = []
                for unit, model in zip(units, models, strict=True):
                    new = copy.copy(unit)
                    new.model, new.policy_sha256 = model, canonical_model_logical_hash(model)
                    copies.append(new)
                replacement = tuple(copies)
                payload.update(
                    status="accepted_guarded_update"
                    if training and enforce_kl
                    else "accepted_no_kl_update"
                    if training
                    else "qualified_unchanged_policy"
                    if enforce_kl
                    else "accepted_unchanged_no_kl_caps_recorded",
                    accepted=True,
                    updated=training,
                    backtracks=b,
                    new_models=[unit.policy_sha256 for unit in replacement],
                )
                break
        if not payload["accepted"]:
            payload["status"] = "all_backtracks_rejected_no_commit"
        check()
    except (
        ValueError,
        TypeError,
        RuntimeError,
        FloatingPointError,
        TimeoutError,
        OSError,
    ) as error:
        replacement = units
        payload.update(
            status="partial_deadline_no_commit"
            if isinstance(error, TimeoutError)
            else "numerical_or_guard_failure_no_commit",
            accepted=False,
            updated=False,
            backtracks=None,
            new_models=payload["old_models"],
            error={"type": type(error).__name__, "message": str(error)[:512]},
        )
    payload["old_occupancy"] = guard.old_rows
    attached = {id(row) for entry in payload["candidates"] for row in entry["partial"]}
    payload["unfinished_partial_records"] = [
        row for row in guard.records if id(row) not in attached
    ]
    payload["work"] = counter.document()
    payload["checks"] = checks
    # Source, original state and serialization are all inside the original budget.
    try:
        if (
            source_identities() != sources
            or plan.sha256 != payload["plan_sha256"]
            or teacher.sha256 != payload["teacher_sha256"]
            or tuple(
                unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True)
            )
            != original
        ):
            raise ValueError("partial endpoint source/original state changed")
        encoded, sha = encode_record(payload)
        check()
        if (
            source_identities() != sources
            or plan.sha256 != payload["plan_sha256"]
            or teacher.sha256 != payload["teacher_sha256"]
            or tuple(
                unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True)
            )
            != original
            or [canonical_model_logical_hash(unit.model) for unit in replacement]
            != payload["new_models"]
        ):
            raise ValueError("partial endpoint source/model changed during encoding")
        # No caller callback after this final binding pass. The caller's hard
        # deadline must also bound this finite seal and the return itself.
    except (
        ValueError,
        TypeError,
        RuntimeError,
        FloatingPointError,
        TimeoutError,
        OSError,
    ) as error:
        replacement = units
        payload.update(
            status="partial_deadline_no_commit"
            if isinstance(error, TimeoutError)
            else "final_integrity_no_commit",
            accepted=False,
            updated=False,
            backtracks=None,
            new_models=payload["old_models"],
            error={"type": type(error).__name__, "message": str(error)[:512]},
        )
        try:
            encoded, sha = encode_record(payload)
        except ValueError:
            # Preserve hashes/counts of the oversized prefix, never a usable
            # model. Full resident diagnostic objects remain caller-local only.
            payload = {
                "artifact": "native_ga_partial_update_v2",
                "matched_feasibility_requirement": requirement,
                "status": "diagnostic_byte_cap_no_commit",
                "accepted": False,
                "updated": False,
                "backtracks": None,
                "plan_sha256": plan.sha256,
                "completed_candidate_count": len(payload["candidates"]),
                "work": counter.document(),
                "error": payload["error"],
                "campaign_eligible": False,
                "scientific_evidence_accepted": False,
                "production_eligible": False,
            }
            encoded, sha = encode_record(payload)
    return replacement, PartialGuardedUpdate(
        encoded, sha, payload["status"], payload["accepted"], payload["backtracks"]
    )

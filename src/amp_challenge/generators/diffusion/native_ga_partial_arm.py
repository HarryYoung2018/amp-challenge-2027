"""Real eligible GA/native successor; private guarded updates are not publication."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict

from safetensors.torch import save

from amp_challenge.generators.diffusion.model import _canonical_model_state
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_ga_endpoint_arm import (
    _validate_history,
    checkpoint_order,
)
from amp_challenge.generators.diffusion.native_ga_endpoint_records import (
    ChargedEndpointEligibility,
    build_charged_teacher,
)
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    CONFIG_SHA256,
    ELIGIBLE_CONTRACT_SHA256,
    PartialGAWave,
    PartialGuardPlan,
    encode_record,
    plain,
    read_clock,
    source_identities,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_ga_partial_update import update_ga_partial_endpoints
from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.diffusion.native_search_posterior import read_native_posterior
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.search.peptide_ga_driver_v2 import PrivateGACollisions
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    EligibleGAKernelInput,
    EligibleGAPrefix,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3_verify import verify_eligible_prefix
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import sequence_key

READY = "ready_private_composition_required"


class _KLGuardStopped(RuntimeError):
    pass


class _PolicyGuardStopped(RuntimeError):
    pass


def propose_partial_ga_wave(
    units,
    kernel_input,
    prefix_result,
    teacher_eligibility,
    *,
    context,
    evaluator,
    posterior_binding,
    expected_history_sha256,
    expected_eligible_query_ids,
    expected_kernel_source_sha256,
    expected_eligibility_source_sha256,
    expected_prefix_deadline,
    expected_behavior_versions,
    native_ordinal,
    deadline,
    enforce_kl=True,
    clock=time.monotonic,
    matched_feasibility=None,
):
    """Consume the verified complete prefix; never regenerate/replace its work.

    deadline already includes the original controller wave/run epochs. Full
    source/clock/provider authentication and durable publication remain external.
    """
    started = read_clock(clock)
    if (
        type(kernel_input) is not EligibleGAKernelInput
        or type(prefix_result) is not EligibleGAPrefix
    ):
        raise TypeError("partial GA requires exact eligible kernel/result types")
    if (
        type(deadline) not in (int, float)
        or type(expected_prefix_deadline) not in (int, float)
        or not math.isfinite(deadline)
        or not math.isfinite(expected_prefix_deadline)
        or expected_prefix_deadline > deadline
        or type(enforce_kl) is not bool
    ):
        raise ValueError("partial GA original deadline/mode differs")
    history = kernel_input.history
    _validate_history(context, history)
    if (
        type(native_ordinal) is not int
        or not 0 <= native_ordinal <= 28 * 65536
        or type(expected_behavior_versions) is not dict
        or set(expected_behavior_versions) != set(TRIPLES)
    ):
        raise ValueError("partial GA actual ordinal/version inventory differs")
    units = tuple(units)
    if tuple(unit.triple for unit in units) != TRIPLES:
        raise ValueError("partial GA requires ordered ten students")
    original_versions = expected_behavior_versions.copy()
    bindings = tuple(unit_binding(unit, original_versions[unit.triple]) for unit in units)
    if any(row[3] >= history.round_index for row in bindings):
        raise ValueError("partial GA actual behavior versions cannot be round counters")
    requirement = None if matched_feasibility is None else matched_feasibility.document()
    if (
        requirement is not None
        and matched_feasibility.context_sha256 != context.objective.context_sha256
    ):
        raise ValueError("partial wave matched feasibility context differs")
    sources = source_identities()
    source_sha = _json_hash(sources)
    if context.driver.implementation_sha256 != source_sha or (
        kernel_input.configuration_id,
        kernel_input.training_sequence_keys,
        expected_eligibility_source_sha256,
    ) != (
        context.driver.configuration_id,
        context.driver.training_sequence_keys,
        context.eligibility_source_sha256,
    ):
        raise ValueError("partial GA context implementation/config/exclusion binding differs")
    training = set(kernel_input.training_sequence_keys)
    if not set().union(*(unit.initialization.training_sequence_ids for unit in units)) <= training:
        raise ValueError("partial GA training exclusions omit initializer inventory")
    effective = min(float(deadline), float(expected_prefix_deadline), started + 180)
    original_input = kernel_input.sha256
    original_output = prefix_result.output_sha256
    original_prefix_bytes = prefix_result.canonical_bytes()
    pins = {
        "expected_kernel_source_sha256": expected_kernel_source_sha256,
        "expected_contract_sha256": ELIGIBLE_CONTRACT_SHA256,
        "expected_eligibility_source_sha256": expected_eligibility_source_sha256,
        "expected_objective_context_sha256": context.objective.context_sha256,
        "expected_history_sha256": expected_history_sha256,
        "expected_eligible_query_ids": expected_eligible_query_ids,
        "expected_deadline_monotonic": expected_prefix_deadline,
    }
    payload = {
        "artifact": "native_ga_partial_wave_v2",
        "matched_feasibility_requirement": requirement,
        "configuration_sha256": CONFIG_SHA256,
        "sources": sources,
        "context": asdict(context),
        "kernel_input": asdict(kernel_input),
        "eligible_prefix": asdict(prefix_result),
        "external_prefix_pins": pins,
        "teacher_eligibility": None,
        "teacher": None,
        "plan": None,
        "update": None,
        "native_ordinal": native_ordinal,
        "next_native_ordinal": native_ordinal,
        "old_units": bindings,
        "working_versions": original_versions.copy(),
        "next_behavior_versions": original_versions.copy(),
        "native_batches": [],
        "pool": [],
        "predictions": [],
        "ranked_positions": [],
        "enforce_kl": enforce_kl,
        "status": "not_started",
        "error": None,
        "posterior_binding": None,
        "checkpoints": [],
        "working_models": [],
        "campaign_eligible": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
        "selected_tuning_result_available": False,
    }
    working, committed = units, units
    checkpoint_payloads = ()
    counter = NativeWorkCounter(units)

    def check():
        if read_clock(clock) >= effective:
            raise TimeoutError("partial GA original controller wave deadline")
        work = counter.document()["total"]
        if (
            work["row_forwards"] > 390400 + 2088960 + (92160 if requirement is not None else 0)
            or work["forward_calls"] > 28880 + 2088960 + (10944 if requirement is not None else 0)
            or work["backward_calls"] > 320
        ):
            raise RuntimeError("partial GA complete native work ceiling exceeded")

    try:
        with counter:
            check()
            verify_eligible_prefix(prefix_result, kernel_input, **pins)
            check()
            if prefix_result.status != "complete":
                payload["status"] = prefix_result.status
            else:
                if type(teacher_eligibility) is not ChargedEndpointEligibility:
                    raise TypeError("partial GA needs an explicit successful-only teacher bridge")
                teacher_eligibility.validate(history, context)
                if (
                    frozenset(teacher_eligibility.query_ids) != expected_eligible_query_ids
                    or kernel_input.eligibility.query_ids != expected_eligible_query_ids
                ):
                    raise ValueError("partial GA teacher/kernel eligibility subsets differ")
                teacher = build_charged_teacher(history, teacher_eligibility, context)
                payload["teacher_eligibility"], payload["teacher"] = (
                    asdict(teacher_eligibility),
                    asdict(teacher),
                )
                prefix = prefix_result.prefix
                plan = PartialGuardPlan(
                    history.seed,
                    history.round_index,
                    native_ordinal,
                    65536 - len(prefix.attempts),
                    checkpoint_order(history.seed),
                    tuple(prefix.accepted_sequences[128:]),
                    kernel_input.semantic_sha256,
                    kernel_input.sha256,
                    prefix_result.output_sha256,
                    history.sha256,
                    _json_hash(plain(asdict(kernel_input.eligibility))),
                    float(expected_prefix_deadline),
                    bindings,
                    source_sha,
                )
                payload["plan"] = asdict(plan)
                working, update = update_ga_partial_endpoints(
                    units,
                    teacher,
                    plan,
                    deadline=effective,
                    counter=counter,
                    enforce_kl=enforce_kl,
                    initial=history.round_index == 1,
                    matched_feasibility=matched_feasibility,
                    clock=clock,
                )
                payload["update"] = asdict(update)
                if not update.accepted:
                    if update.status == "all_backtracks_rejected_no_commit":
                        if matched_feasibility is not None:
                            raise _PolicyGuardStopped("partial policy guards rejected the policy")
                        if enforce_kl:
                            raise _KLGuardStopped("partial KL caps rejected the policy")
                    raise RuntimeError("partial guarded policy stopped: " + update.status)
                updated = json.loads(update.record_json)["updated"]
                payload["working_versions"] = {
                    key: value + int(updated) for key, value in original_versions.items()
                }
                check()
                edits = {
                    row.edit.sequence: row.edit
                    for row in prefix.attempts
                    if row.rejection_reason is None
                }
                fitness = {
                    row.sequence_key: row.fitness for row in kernel_input.eligible_population()
                }

                def parent_fitness(sequence):
                    keys = edits[sequence].parent_sequence_keys
                    return math.fsum(fitness[key] for key in keys) / len(keys)

                pool = payload["pool"]
                for seq in prefix.accepted_sequences[:128]:
                    pool.append(
                        {
                            "sequence": seq,
                            "branch": "unaltered_ga",
                            "native_ordinal": None,
                            "parent_fitness": parent_fitness(seq),
                        }
                    )
                seen = set(map(sequence_key, prefix.accepted_sequences))
                charged = {sequence_key(row.sequence) for row in history.observations}
                native_seed = int(
                    _json_hash(
                        [
                            "native-ga-endpoint-v1",
                            kernel_input.stream_sha256,
                            history.seed,
                            history.round_index,
                            "native-paths",
                        ]
                    )[:16],
                    16,
                )
                while (
                    len(pool) < 256
                    and payload["next_native_ordinal"] < native_ordinal + plan.remaining_attempts
                ):
                    check()
                    first = payload["next_native_ordinal"]
                    count = min(256 - len(pool), native_ordinal + plan.remaining_attempts - first)
                    slots = [
                        {
                            "ordinal": ordinal,
                            "triple": units[plan.checkpoint_order[ordinal % 10]].triple,
                            "parent_index": (ordinal - native_ordinal) % 128,
                            "trace": None,
                            "consumed": False,
                            "rejection": None,
                        }
                        for ordinal in range(first, first + count)
                    ]
                    payload["native_batches"].append(slots)
                    payload["next_native_ordinal"] += count
                    for index in plan.checkpoint_order:
                        selected = [row for row in slots if row["triple"] == units[index].triple]
                        if not selected:
                            continue
                        check()
                        with counter.at("actual_native_candidate_generation"):
                            traces = sample_native_proposals(
                                working[index].model,
                                tuple(plan.parents[row["parent_index"]] for row in selected),
                                start_levels=(32,) * len(selected),
                                seed=native_seed,
                                ordinals=tuple(row["ordinal"] for row in selected),
                            )
                        for row, trace in zip(selected, traces, strict=True):
                            row["trace"] = asdict(trace)
                        check()
                    for row in slots:
                        seq = row["trace"]["endpoint"]
                        key = sequence_key(seq)
                        rejection = (
                            "training_overlap"
                            if key in training
                            else "charged_collision"
                            if key in charged
                            else "generated_duplicate"
                            if key in seen
                            else None
                        )
                        row.update(rejection=rejection, consumed=True)
                        seen.add(key)
                        if rejection is None:
                            pool.append(
                                {
                                    "sequence": seq,
                                    "branch": "native_refinement",
                                    "native_ordinal": row["ordinal"],
                                    "parent_fitness": parent_fitness(
                                        plan.parents[row["parent_index"]]
                                    ),
                                }
                            )
                if len(pool) != 256:
                    payload["status"] = "abstained_incomplete_native_branch"
                else:
                    if (
                        posterior_binding.history_sha256,
                        posterior_binding.objective_context_sha256,
                        posterior_binding.feature_source_sha256,
                        posterior_binding.evaluator_source_sha256,
                    ) != (
                        history.sha256,
                        context.objective.context_sha256,
                        context.feature_source_sha256,
                        context.evaluator_source_sha256,
                    ):
                        raise ValueError("partial GA posterior source/history/context differs")
                    payload["posterior_binding"] = asdict(posterior_binding)
                    for offset in (0, 128):
                        check()
                        batch = read_native_posterior(
                            evaluator,
                            tuple(row["sequence"] for row in pool[offset : offset + 128]),
                            expected_binding=posterior_binding,
                            context=context.objective,
                        )
                        payload["predictions"].append(asdict(batch))
                        check()
                    scores = [
                        score for batch in payload["predictions"] for score in batch["scores"]
                    ]
                    ranked = sorted(
                        (i for i, score in enumerate(scores) if score["feasible"]),
                        key=lambda i: (
                            -context.objective.scalarize(tuple(scores[i]["objectives"])),
                            -pool[i]["parent_fitness"],
                            sequence_key(pool[i]["sequence"]),
                        ),
                    )
                    payload["ranked_positions"] = ranked
                    payload["status"] = (
                        READY if len(ranked) >= 14 else "abstained_insufficient_feasible_candidates"
                    )
            check()
            # Actual private weights remain evidence even if the later arm fails.
            checkpoint_payloads = tuple(
                (unit.triple, save(_canonical_model_state(unit.model), metadata=None))
                for unit in working
            )
            if sum(len(value) for _, value in checkpoint_payloads) > 256 * 1024**2:
                raise ValueError("partial GA private checkpoint byte cap")
            check()
    except (
        ValueError,
        TypeError,
        RuntimeError,
        FloatingPointError,
        TimeoutError,
        OSError,
    ) as error:
        payload.update(
            status="stopped_policy_guard"
            if isinstance(error, _PolicyGuardStopped)
            else "stopped_kl_guard"
            if isinstance(error, _KLGuardStopped)
            else "stopped_deadline"
            if isinstance(error, TimeoutError)
            else "stopped_integrity_or_numerical_failure",
            ranked_positions=[],
            error={"type": type(error).__name__, "message": str(error)[:512]},
        )
    payload["work"] = counter.document()
    payload["working_models"] = [(unit.triple, unit.policy_sha256) for unit in working]
    payload["checkpoints"] = [
        (triple, hashlib.sha256(value).hexdigest()) for triple, value in checkpoint_payloads
    ]
    if payload["status"] == READY:
        payload["next_behavior_versions"] = payload["working_versions"]
    try:
        if (
            source_identities() != sources
            or (matched_feasibility is not None and matched_feasibility.document() != requirement)
            or expected_behavior_versions != original_versions
            or asdict(context) != payload["context"]
            or (
                payload["teacher_eligibility"] is not None
                and asdict(teacher_eligibility) != payload["teacher_eligibility"]
            )
            or (
                payload["posterior_binding"] is not None
                and asdict(posterior_binding) != payload["posterior_binding"]
            )
            or kernel_input.sha256 != original_input
            or prefix_result.output_sha256 != original_output
            or prefix_result.canonical_bytes() != original_prefix_bytes
            or tuple(unit_binding(unit, row[3]) for unit, row in zip(units, bindings, strict=True))
            != bindings
        ):
            raise ValueError("partial GA final source/history/prefix/original state differs")
        for unit in working:
            unit.check()
        encoded, sha = encode_record(payload)
        check()
        if (
            source_identities() != sources
            or (matched_feasibility is not None and matched_feasibility.document() != requirement)
            or expected_behavior_versions != original_versions
            or asdict(context) != payload["context"]
            or (
                payload["teacher_eligibility"] is not None
                and asdict(teacher_eligibility) != payload["teacher_eligibility"]
            )
            or (
                payload["posterior_binding"] is not None
                and asdict(posterior_binding) != payload["posterior_binding"]
            )
            or kernel_input.sha256 != original_input
            or prefix_result.output_sha256 != original_output
            or prefix_result.canonical_bytes() != original_prefix_bytes
            or tuple(unit_binding(unit, row[3]) for unit, row in zip(units, bindings, strict=True))
            != bindings
        ):
            raise ValueError("partial GA post-encoding source/input drift")
        for unit in working:
            unit.check()
        # The last external clock callback was before the binding guard above.
        # Outer hard preemption still bounds this callback-free seal and return.
        if payload["status"] == READY:
            committed = working
    except (
        ValueError,
        TypeError,
        RuntimeError,
        FloatingPointError,
        TimeoutError,
        OSError,
    ) as error:
        payload.update(
            status="stopped_deadline"
            if isinstance(error, TimeoutError)
            else "stopped_final_integrity",
            ranked_positions=[],
            next_behavior_versions=original_versions.copy(),
            error={"type": type(error).__name__, "message": str(error)[:512]},
        )
        try:
            encoded, sha = encode_record(payload)
        except ValueError:
            payload = {
                "artifact": "native_ga_partial_wave_v2",
                "matched_feasibility_requirement": requirement,
                "status": "stopped_diagnostic_byte_cap",
                "ranked_positions": [],
                "next_native_ordinal": payload["next_native_ordinal"],
                "next_behavior_versions": original_versions.copy(),
                "work": counter.document(),
                "prefix_sha256": original_output,
                "campaign_eligible": False,
                "scientific_evidence_accepted": False,
                "production_eligible": False,
            }
            encoded, sha = encode_record(payload)
    ranking = (
        tuple(payload["pool"][index]["sequence"] for index in payload["ranked_positions"])
        if payload["status"] == READY
        else ()
    )
    return committed, PartialGAWave(
        encoded,
        sha,
        payload["status"],
        ranking,
        payload["next_native_ordinal"],
        tuple((triple, payload["next_behavior_versions"][triple]) for triple in TRIPLES),
        checkpoint_payloads,
    )


def compose_partial_ga_seats(result, history, collisions):
    """Separate private composition only; future reserves never reach the arm."""
    if (
        type(result) is not PartialGAWave
        or type(collisions) is not PrivateGACollisions
        or hashlib.sha256(result.record_json.encode()).hexdigest() != result.sha256
    ):
        raise TypeError("partial GA private composition types/seal differ")
    collisions.__post_init__()
    raw = json.loads(result.record_json)
    if raw["status"] != result.status:
        raise ValueError("partial GA private composition status differs")
    if not result.method_pool:
        return ()
    if raw["kernel_input"]["history"] != plain(asdict(history)) or set(
        collisions.charged_sequence_keys
    ) != {sequence_key(row.sequence) for row in history.observations}:
        raise ValueError("partial GA private composition history/charge inventory differs")
    if result.method_pool != tuple(
        raw["pool"][index]["sequence"] for index in raw["ranked_positions"]
    ):
        raise ValueError("partial GA sealed ranked pool differs")
    excluded = set(collisions.charged_sequence_keys) | set(
        collisions.upcoming_reserve_sequence_keys
    )
    seats = tuple(seq for seq in result.method_pool if sequence_key(seq) not in excluded)[:14]
    return seats if len(seats) == 14 else ()

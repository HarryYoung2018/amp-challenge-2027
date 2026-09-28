"""Executed GA/native endpoint baseline, without KG or private oracle access.

Returned models are private proposals. Only a ready result can replace all ten;
durable publication, reserve composition and authenticated clocks remain distinct.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors.torch import save

from amp_challenge.evaluation.sequential_v2_seals import PhaseBuilder
from amp_challenge.generators.diffusion.model import _canonical_model_state
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_ga_endpoint_records import (
    ARM_CONFIG_SHA256,
    ARTIFACT,
    ChargedEndpointEligibility,
    GAEndpointContext,
    GAEndpointWave,
    build_charged_teacher,
)
from amp_challenge.generators.diffusion.native_proposals import sample_native_proposals
from amp_challenge.generators.diffusion.native_search_posterior import read_native_posterior
from amp_challenge.generators.diffusion.native_shared_endpoint import (
    endpoint_source_identities,
    update_shared_endpoints,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_records import TRIPLES
from amp_challenge.generators.search.peptide_ga_driver_v2 import PrivateGACollisions
from amp_challenge.generators.search.peptide_ga_tunable_v2 import generate_prefix
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    GAKernelInput,
    canonical_json_bytes,
    digest,
    population_from_charged,
    require,
    sequence_key,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_verify import verify_prefix
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

READY = "ready_private_composition_required"


def arm_source_identities():
    root = Path(__file__).resolve().parents[4]
    files = ["configs/search/native_ga_endpoint_no_kg_v1.toml"] + [
        "src/amp_challenge/generators/diffusion/" + name + ".py"
        for name in (
            "native_ga_endpoint_records",
            "native_ga_endpoint_arm",
            "native_ga_endpoint_verify",
            "native_ga_endpoint_driver",
            "native_search_posterior",
        )
    ]
    files += [
        "src/amp_challenge/generators/search/" + name + ".py"
        for name in (
            "peptide_ga",
            "peptide_ga_records",
            "peptide_ga_tunable_v2",
            "peptide_ga_tunable_v2_records",
            "peptide_ga_tunable_v2_verify",
            "peptide_ga_driver_v2",
            "verified_charged_history",
            "records",
        )
    ]
    files += ["src/amp_challenge/evaluation/sequential_v2_seals.py"]
    result = {name: digest((root / name).read_bytes()) for name in files}
    require(result[files[0]] == ARM_CONFIG_SHA256, "GA endpoint config bytes differ")
    return {**endpoint_source_identities(), **result}


def arm_implementation_sha256():
    return digest(canonical_json_bytes(arm_source_identities()))


def checkpoint_order(seed):
    key = _json_hash(["native-ga-endpoint-v1", seed, "checkpoint-order"])
    return tuple(
        int(value)
        for value in np.random.Generator(np.random.PCG64DXSM(int(key[:32], 16))).permutation(10)
    )


def _validate_history(context, history):
    require(
        type(context) is GAEndpointContext and type(history) is VerifiedHistorySnapshot,
        "GA endpoint context/history type differs",
    )
    context.__post_init__()
    history.__post_init__()
    driver = context.driver
    require(
        (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
        )
        == (
            driver.run_id,
            driver.seed,
            driver.objective_context_sha256,
            driver.oracle_bundle_sha256,
        ),
        "GA endpoint history source/run differs",
    )
    require(
        not set(driver.training_sequence_keys).intersection(
            sequence_key(row.sequence) for row in history.observations
        ),
        "GA endpoint charged history overlaps excluded training namespace",
    )


def propose_ga_endpoint_wave(
    units,
    history,
    eligibility,
    *,
    context,
    evaluator,
    posterior_binding,
    native_ordinal=0,
    behavior_version=0,
    deadline,
    enforce_kl=True,
    operator_guard=None,
    expected_operator_source=None,
    expected_operator_plan=None,
    clock=time.monotonic,
):
    """Real 128-GA/128-native proposal, supervised update, and charged-only score.

    deadline is the caller's ORIGINAL run deadline, never a resumed budget. The
    local180-second cap includes preflight, all numerical work and serialization.
    Callers must externally preempt blocking providers and include model loading.
    """
    started = clock()
    _validate_history(context, history)
    require(
        type(native_ordinal) is int
        and 0 <= native_ordinal <= 28 * ATTEMPT_CAP
        and type(behavior_version) is int
        and 0 <= behavior_version <= 27
        and type(enforce_kl) is bool
        and math.isfinite(deadline),
        "GA endpoint state/budget differs",
    )
    sources = arm_source_identities()
    require(
        context.driver.implementation_sha256 == digest(canonical_json_bytes(sources)),
        "GA endpoint executable bytes differ from fixed context",
    )
    units = tuple(units)
    require(
        tuple(unit.triple for unit in units) == TRIPLES,
        "GA endpoint needs exact ten-model inventory",
    )
    for unit in units:
        unit.check()
        require(
            unit.model.config.levels == 64
            and unit.model.config.min_length == 8
            and unit.model.config.max_length == 50,
            "GA endpoint requires native64-level8..50 support",
        )
    training = set(context.driver.training_sequence_keys)
    require(
        set().union(*(unit.initialization.training_sequence_ids for unit in units)) <= training,
        "GA endpoint exclusions omit generator training IDs",
    )
    effective_deadline = min(deadline, started + 180)
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if clock() >= effective_deadline:
            raise TimeoutError("GA endpoint original/wave deadline")

    working, update, teacher, prefix, kernel = units, None, None, None, None
    native_batches, pool, predictions, ranked = [], [], [], []
    status, error = "paused_incomplete_wave", None
    next_ordinal = native_ordinal
    order = checkpoint_order(history.seed)
    work_version = behavior_version
    try:
        check()
        if history.complete and history.round_index == 29:
            status = "budget_complete_pending_controller_terminal"
        elif history.complete and not any(
            row.status == "successful" for row in history.observations
        ):
            status = "abstained_no_successful_parent"
        elif history.complete:
            require(
                type(eligibility) is ChargedEndpointEligibility,
                "GA endpoint eligibility record missing",
            )
            teacher = build_charged_teacher(history, eligibility, context)
            require(
                posterior_binding.history_sha256 == history.sha256
                and posterior_binding.objective_context_sha256 == context.objective.context_sha256
                and posterior_binding.feature_source_sha256 == context.feature_source_sha256
                and posterior_binding.evaluator_source_sha256 == context.evaluator_source_sha256,
                "GA endpoint posterior history/source pins differ",
            )
            if history.round_index >= 2:
                working, update = update_shared_endpoints(
                    units,
                    teacher,
                    seed=history.seed,
                    deadline=effective_deadline,
                    enforce_kl=enforce_kl,
                    operator_guard=operator_guard,
                    expected_operator_source=expected_operator_source,
                    expected_operator_plan=expected_operator_plan,
                    clock=clock,
                )
                if update.accepted:
                    work_version += 1
                elif update.status != "insufficient_targets_no_update":
                    raise RuntimeError("shared endpoint stop: " + update.status)
            check()
            kernel = GAKernelInput(
                context.driver.configuration_id,
                history.seed,
                history.round_index,
                history.observations,
                context.driver.training_sequence_keys,
            )
            while prefix is None or prefix.status == "in_progress":
                check()
                prefix = generate_prefix(kernel, resume=prefix, max_new_attempts=32)
            verify_prefix(prefix, kernel)
            if prefix.status != "complete":
                status = "abstained_incomplete_ga_prefix"
            else:
                edits = {
                    row.edit.sequence: row.edit
                    for row in prefix.attempts
                    if row.rejection_reason is None
                }
                fitness = {row.sequence_key: row.fitness for row in population_from_charged(kernel)}

                def parent_fitness(sequence):
                    keys = edits[sequence].parent_sequence_keys
                    return math.fsum(fitness[key] for key in keys) / len(keys)

                for sequence in prefix.accepted_sequences[:128]:
                    pool.append(
                        {
                            "sequence": sequence,
                            "branch": "unaltered_ga",
                            "native_ordinal": None,
                            "parent_fitness": parent_fitness(sequence),
                        }
                    )
                parents = prefix.accepted_sequences[128:]
                seen = set(map(sequence_key, prefix.accepted_sequences))
                charged = {sequence_key(row.sequence) for row in history.observations}
                native_seed = int(
                    _json_hash(
                        [
                            "native-ga-endpoint-v1",
                            kernel.stream_sha256,
                            history.seed,
                            history.round_index,
                            "native-paths",
                        ]
                    )[:16],
                    16,
                )
                while (
                    len(pool) < 256
                    and len(prefix.attempts) + next_ordinal - native_ordinal < ATTEMPT_CAP
                ):
                    check()
                    count = min(
                        256 - len(pool),
                        ATTEMPT_CAP - len(prefix.attempts) - next_ordinal + native_ordinal,
                    )
                    slots = [
                        {
                            "ordinal": ordinal,
                            "triple": units[order[ordinal % 10]].triple,
                            "parent_index": (ordinal - native_ordinal) % 128,
                            "trace": None,
                            "rejection": None,
                            "consumed": False,
                        }
                        for ordinal in range(next_ordinal, next_ordinal + count)
                    ]
                    native_batches.append(slots)
                    next_ordinal += count  # planned work is never silently retried/free
                    for index in order:
                        selected = [slot for slot in slots if slot["triple"] == units[index].triple]
                        if not selected:
                            continue
                        check()
                        traces = sample_native_proposals(
                            working[index].model,
                            tuple(parents[slot["parent_index"]] for slot in selected),
                            start_levels=(32,) * len(selected),
                            seed=native_seed,
                            ordinals=tuple(slot["ordinal"] for slot in selected),
                        )
                        for slot, trace in zip(selected, traces, strict=True):
                            slot["trace"] = asdict(trace)
                        check()
                    for slot in slots:
                        sequence = slot["trace"]["endpoint"]
                        key = sequence_key(sequence)
                        rejection = (
                            "training_overlap"
                            if key in training
                            else "charged_collision"
                            if key in charged
                            else "generated_duplicate"
                            if key in seen
                            else None
                        )
                        slot.update(rejection=rejection, consumed=True)
                        seen.add(key)
                        if rejection is None:
                            pool.append(
                                {
                                    "sequence": sequence,
                                    "branch": "native_refinement",
                                    "native_ordinal": slot["ordinal"],
                                    "parent_fitness": parent_fitness(parents[slot["parent_index"]]),
                                }
                            )
                if len(pool) != 256:
                    status = "abstained_incomplete_native_branch"
                else:
                    for offset in (0, 128):
                        check()
                        batch = read_native_posterior(
                            evaluator,
                            tuple(row["sequence"] for row in pool[offset : offset + 128]),
                            expected_binding=posterior_binding,
                            context=context.objective,
                        )
                        predictions.append(asdict(batch))
                        check()
                    scores = [score for batch in predictions for score in batch["scores"]]
                    ranked = sorted(
                        (i for i, score in enumerate(scores) if score["feasible"]),
                        key=lambda i: (
                            -context.objective.scalarize(tuple(scores[i]["objectives"])),
                            -pool[i]["parent_fitness"],
                            sequence_key(pool[i]["sequence"]),
                        ),
                    )
                    status = (
                        READY if len(ranked) >= 14 else "abstained_insufficient_feasible_candidates"
                    )
        check()
    except (TimeoutError, ValueError, TypeError, RuntimeError, FloatingPointError) as failure:
        status = (
            "stopped_deadline"
            if isinstance(failure, TimeoutError)
            else "stopped_numerical_or_provider_failure"
        )
        error = {"type": type(failure).__name__, "message": str(failure)[:512]}
        ranked = []
    for unit in (*units, *working):
        unit.check()
    require(arm_source_identities() == sources, "GA endpoint source bytes changed during execution")
    # Preserve actual working weights even when later work fails, for path replay;
    # only the separate committed model inventory controls returned replacements.
    checkpoints = tuple(
        (unit.triple, save(_canonical_model_state(unit.model), metadata=None)) for unit in working
    )
    require(
        sum(len(data) for _, data in checkpoints) <= 256 * 1024**2,
        "GA endpoint checkpoint byte cap",
    )
    committed = working if status == READY else units
    payload = {
        "artifact": ARTIFACT,
        "config_sha256": ARM_CONFIG_SHA256,
        "source_identities": sources,
        "context": asdict(context),
        "history": asdict(history),
        "history_sha256": history.sha256,
        "eligibility": None if eligibility is None else asdict(eligibility),
        "teacher": None if teacher is None else asdict(teacher),
        "update": None if update is None else asdict(update),
        "prefix": None if prefix is None else asdict(prefix),
        "posterior_binding": None if posterior_binding is None else asdict(posterior_binding),
        "native_batches": native_batches,
        "pool": pool,
        "predictions": predictions,
        "ranked_positions": ranked,
        "native_ordinal": native_ordinal,
        "next_native_ordinal": next_ordinal,
        "behavior_version": behavior_version,
        "working_behavior_version": work_version,
        "next_behavior_version": work_version if status == READY else behavior_version,
        "checkpoint_order": order,
        "enforce_kl": enforce_kl,
        "status": status,
        "error": error,
        "checks": checks,
        "old_models": [(unit.triple, unit.policy_sha256, unit.reference_sha256) for unit in units],
        "working_models": [(unit.triple, unit.policy_sha256) for unit in working],
        "committed_models": [(unit.triple, unit.policy_sha256) for unit in committed],
        "checkpoints": [(triple, digest(data)) for triple, data in checkpoints],
        "selected_tuning_result_available": False,
        "actual_operator_campaign_compatibility": "unresolved",
        "campaign_eligible": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }
    encoded = canonical_json_bytes(payload)
    require(len(encoded) <= 256 * 1024**2, "GA endpoint wave record byte cap")
    content_sha = digest(encoded)
    if clock() >= effective_deadline:
        status, committed, ranked = "stopped_deadline", units, []
        payload.update(
            status=status,
            ranked_positions=[],
            next_behavior_version=behavior_version,
            committed_models=[(unit.triple, unit.policy_sha256) for unit in units],
            error={"type": "TimeoutError", "message": "GA endpoint serialization deadline"},
        )
        encoded = canonical_json_bytes(payload)
        content_sha = digest(encoded)
    result = GAEndpointWave(
        encoded.decode(),
        content_sha,
        status,
        tuple(pool[i]["sequence"] for i in ranked),
        next_ordinal,
        work_version if status == READY else behavior_version,
        checkpoints,
    )
    return committed, result


def publish_ga_endpoint_wave(
    destination,
    result,
    *,
    previous_phase_sha256=None,
    deadline,
    clock=time.monotonic,
    metadata=None,
):
    """Exclusive evidence publication; a late seal is retained but no handoff returns."""
    require(
        type(result) is GAEndpointWave and digest(result.record_json.encode()) == result.sha256,
        "GA endpoint publication record seal differs",
    )
    require(math.isfinite(deadline), "GA endpoint publication deadline differs")
    raw = json.loads(result.record_json)
    require(
        tuple(triple for triple, _ in result.checkpoint_payloads) == TRIPLES
        and raw["checkpoints"]
        == [[triple, digest(data)] for triple, data in result.checkpoint_payloads]
        and sum(len(data) for _, data in result.checkpoint_payloads) <= 256 * 1024**2,
        "GA endpoint publication checkpoint inventory differs",
    )
    if clock() >= deadline:
        raise TimeoutError("GA endpoint publication original deadline")
    predecessors = {} if previous_phase_sha256 is None else {"previous": previous_phase_sha256}
    with PhaseBuilder(
        destination, artifact=ARTIFACT, predecessor_seals=predecessors, metadata=metadata
    ) as builder:
        builder.write_bytes("wave.json", result.record_json.encode())
        paths = ["wave.json"]
        for triple, data in result.checkpoint_payloads:
            name = f"working_models/{triple}.safetensors"
            builder.write_bytes(name, data)
            paths.append(name)
        if clock() >= deadline:
            raise TimeoutError("GA endpoint staged publication deadline")
        seal = builder.publish(expected_payload_paths=paths)
    if clock() >= deadline:
        raise TimeoutError("GA endpoint late publication retained without a usable handoff")
    return seal


def compose_ga_endpoint_method_seats(result, history, collisions):
    """Outer private composition only; reserve identities are never serialized."""
    require(
        type(result) is GAEndpointWave and type(collisions) is PrivateGACollisions,
        "GA endpoint private composition shape differs",
    )
    collisions.__post_init__()
    history.__post_init__()
    require(
        digest(result.record_json.encode()) == result.sha256
        and json.loads(result.record_json)["history_sha256"] == history.sha256,
        "GA endpoint private composition history/seal differs",
    )
    payload = json.loads(result.record_json)
    require(
        result.status == payload["status"]
        and result.ranked_sequences
        == tuple(payload["pool"][index]["sequence"] for index in payload["ranked_positions"]),
        "GA endpoint private composition pool differs from sealed ranking",
    )
    require(
        set(collisions.charged_sequence_keys)
        == {sequence_key(row.sequence) for row in history.observations},
        "GA endpoint charged collision inventory differs",
    )
    forbidden = set(collisions.charged_sequence_keys) | set(
        collisions.upcoming_reserve_sequence_keys
    )
    seats = tuple(seq for seq in result.method_pool if sequence_key(seq) not in forbidden)[:14]
    return seats if len(seats) == 14 else ()

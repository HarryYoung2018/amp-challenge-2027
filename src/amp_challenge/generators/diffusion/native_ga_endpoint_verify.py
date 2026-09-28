"""Separate GA-arm numerical/phase reconstruction, with no posterior or oracle calls.

Accepted native trace/gradient primitives are shared. This same-account verifier
does not authenticate external eligibility, posterior behavior or clock causality.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors.torch import load

from amp_challenge.evaluation.sequential_v2_seals import verify_phase
from amp_challenge.generators.diffusion.model import (
    _validated_loaded_state,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.diffusion.native_ga_endpoint_records import (
    ARM_CONFIG_SHA256,
    ARTIFACT,
    GAEndpointContext,
)
from amp_challenge.generators.diffusion.native_proposals import replay_native_trace
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    NativePosteriorScore,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import SharedEndpointUpdate
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    TRIPLES,
    EndpointTarget,
    EndpointTeacher,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import (
    _trace,
    verify_shared_endpoint_update,
)
from amp_challenge.generators.search.peptide_ga_driver_v2 import _batch
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    GAKernelInput,
    canonical_json_bytes,
    digest,
    hash_string,
    require,
    sequence_key,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_verify import verify_prefix
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def _same(left, right, message):
    require(canonical_json_bytes(left) == canonical_json_bytes(right), message)


def _sources(raw):
    root = Path(__file__).resolve().parents[4]
    shared = (
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
    names = [
        "configs/diffusion/native_shared_endpoint_v1.toml",
        "configs/search/native_ga_endpoint_no_kg_v1.toml",
    ]
    names += [
        "src/amp_challenge/generators/diffusion/" + name + ".py"
        for name in (
            *shared,
            "native_ga_endpoint_records",
            "native_ga_endpoint_arm",
            "native_ga_endpoint_verify",
            "native_ga_endpoint_driver",
            "native_search_posterior",
        )
    ]
    names += [
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
    names += ["src/amp_challenge/evaluation/sequential_v2_seals.py"]
    require(set(raw) == set(names), "GA endpoint source inventory differs")
    require(
        raw == {name: digest((root / name).read_bytes()) for name in names},
        "GA endpoint source bytes differ",
    )
    require(raw[names[1]] == ARM_CONFIG_SHA256, "GA endpoint declared configuration differs")


def _teacher(history, eligibility, context):
    eligibility.validate(history, context)
    origins = dict(zip(eligibility.query_ids, eligibility.origins, strict=True))
    rows = [
        row
        for row in history.observations
        if row.status == "successful" and row.query_id in origins
    ]

    def scalar(row):
        return math.fsum(value * 0.5 for value in row.objectives)

    rows.sort(key=lambda row: (-scalar(row), sequence_key(row.sequence)))
    rows = rows[:64]
    maximum = scalar(rows[0]) if rows else 0.0
    return EndpointTeacher(
        tuple(
            EndpointTarget(
                row.sequence,
                "charged_endpoint",
                float(max(-math.log(2), min(0.0, (scalar(row) - maximum) / 0.1))),
                origins[row.query_id],
            )
            for row in rows
        ),
        "charged_endpoint",
        context.objective.context_sha256,
        digest(canonical_json_bytes(asdict(eligibility))),
        min(28, history.round_index),
    )


def verify_ga_endpoint_wave(
    units,
    history,
    eligibility,
    *,
    context,
    result,
    expected_committed_units=None,
    phase=None,
    operator_report_verifier=None,
):
    """Replay completed math and authenticate optional exact immutable output phase."""
    require(
        digest(result.record_json.encode()) == result.sha256
        and len(result.record_json.encode()) <= 256 * 1024**2,
        "GA endpoint record seal/size differs",
    )
    raw = json.loads(result.record_json)
    require(
        canonical_json_bytes(raw).decode() == result.record_json, "GA endpoint noncanonical record"
    )
    require(
        raw["artifact"] == ARTIFACT
        and raw["config_sha256"] == ARM_CONFIG_SHA256
        and all(
            raw[name] is False and getattr(result, name) is False
            for name in ("campaign_eligible", "scientific_evidence_accepted", "production_eligible")
        )
        and raw["selected_tuning_result_available"] is False
        and raw["actual_operator_campaign_compatibility"] == "unresolved",
        "GA endpoint qualification differs",
    )
    _sources(raw["source_identities"])
    require(
        type(context) is GAEndpointContext and type(history) is VerifiedHistorySnapshot,
        "GA endpoint verifier context/history type differs",
    )
    context.__post_init__()
    history.__post_init__()
    fixed = context.driver
    require(
        (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
        )
        == (fixed.run_id, fixed.seed, fixed.objective_context_sha256, fixed.oracle_bundle_sha256),
        "GA endpoint verifier history run/source/context differs",
    )
    training = set(fixed.training_sequence_keys)
    require(
        not training.intersection(sequence_key(row.sequence) for row in history.observations),
        "GA endpoint verifier charged history overlaps excluded training namespace",
    )
    require(
        digest(canonical_json_bytes(raw["source_identities"]))
        == context.driver.implementation_sha256,
        "GA endpoint frozen implementation identity differs",
    )
    _same(raw["context"], asdict(context), "GA endpoint context changed")
    _same(raw["history"], asdict(history), "GA endpoint charged records changed")
    require(raw["history_sha256"] == history.sha256, "GA endpoint history digest differs")
    _same(
        raw["eligibility"],
        None if eligibility is None else asdict(eligibility),
        "GA endpoint eligibility changed",
    )
    require(
        tuple(unit.triple for unit in units) == TRIPLES,
        "GA endpoint verifier needs exact ten units",
    )
    for unit in units:
        unit.check()
        require(
            unit.model.config.levels == 64
            and unit.model.config.min_length == 8
            and unit.model.config.max_length == 50,
            "GA endpoint verifier requires native64-level8..50 support",
        )
    require(
        set().union(*(unit.initialization.training_sequence_ids for unit in units)) <= training,
        "GA endpoint verifier exclusions omit generator training IDs",
    )
    _same(
        raw["old_models"],
        [(unit.triple, unit.policy_sha256, unit.reference_sha256) for unit in units],
        "GA endpoint original model identity differs",
    )
    require(
        tuple(triple for triple, _ in result.checkpoint_payloads) == TRIPLES,
        "GA endpoint safe tensor inventory differs",
    )
    _same(
        raw["checkpoints"],
        [(triple, digest(data)) for triple, data in result.checkpoint_payloads],
        "GA endpoint checkpoint file identities differ",
    )
    require(
        sum(len(data) for _, data in result.checkpoint_payloads) <= 256 * 1024**2,
        "GA endpoint aggregate checkpoint size differs",
    )
    working = []
    for unit, (_, data) in zip(units, result.checkpoint_payloads, strict=True):
        require(0 < len(data) <= 256 * 1024**2, "GA endpoint checkpoint size differs")
        new = copy.copy(unit)
        new.model = copy.deepcopy(unit.model).eval()
        new.model.load_state_dict(_validated_loaded_state(new.model, load(data)), strict=True)
        new.policy_sha256 = canonical_model_logical_hash(new.model)
        working.append(new)
    working = tuple(working)
    _same(
        raw["working_models"],
        [(unit.triple, unit.policy_sha256) for unit in working],
        "GA endpoint working neural state differs",
    )
    teacher = None if raw["teacher"] is None else _teacher(history, eligibility, context)
    _same(
        raw["teacher"],
        None if teacher is None else asdict(teacher),
        "GA endpoint observed teacher selection differs",
    )
    version = raw["behavior_version"]
    require(
        type(version) is int and 0 <= version <= 27 and type(raw["enforce_kl"]) is bool,
        "GA endpoint version/KL mode differs",
    )
    if raw["update"] is not None:
        require(
            2 <= history.round_index <= 28 and teacher is not None,
            "GA endpoint unused/initial update",
        )
        update = SharedEndpointUpdate(**raw["update"])
        require(
            update.kl_enforced == raw["enforce_kl"], "GA endpoint shared update KL mode differs"
        )
        verify_shared_endpoint_update(
            units,
            teacher,
            update,
            expected_new_units=working,
            operator_report_verifier=operator_report_verifier,
        )
        version += int(update.accepted)
        require(
            update.accepted
            or update.status == "insufficient_targets_no_update"
            or raw["prefix"] is None,
            "GA endpoint continued after shared update stop",
        )
    else:
        _same(
            raw["working_models"],
            [(unit.triple, unit.policy_sha256) for unit in units],
            "GA endpoint model changed without supervised update",
        )
    require(
        raw["working_behavior_version"] == version, "GA endpoint rebuilt behavior version differs"
    )
    seed_key = _json_hash(["native-ga-endpoint-v1", history.seed, "checkpoint-order"])
    order = tuple(
        int(i)
        for i in np.random.Generator(np.random.PCG64DXSM(int(seed_key[:32], 16))).permutation(10)
    )
    require(
        tuple(raw["checkpoint_order"]) == order, "GA endpoint seeded checkpoint assignment differs"
    )
    ordinal = raw["native_ordinal"]
    require(
        type(ordinal) is int and 0 <= ordinal <= 28 * ATTEMPT_CAP,
        "GA endpoint attempt state differs",
    )
    prefix = _batch(raw["prefix"])
    if prefix is not None:
        require(
            teacher is not None and (history.round_index == 1 or raw["update"] is not None),
            "GA endpoint prefix bypassed its declared teacher/update stage",
        )
    pool = []
    if prefix is not None:
        kernel = GAKernelInput(
            context.driver.configuration_id,
            history.seed,
            history.round_index,
            history.observations,
            context.driver.training_sequence_keys,
        )
        verify_prefix(prefix, kernel)
        if prefix.status == "complete":
            observed = {
                sequence_key(row.sequence): math.fsum(0.5 * value for value in row.objectives)
                for row in history.observations
                if row.status == "successful"
            }
            edits = {
                row.edit.sequence: row.edit
                for row in prefix.attempts
                if row.rejection_reason is None
            }

            def fitness(sequence):
                keys = edits[sequence].parent_sequence_keys
                return math.fsum(observed[key] for key in keys) / len(keys)

            pool = [
                {
                    "sequence": sequence,
                    "branch": "unaltered_ga",
                    "native_ordinal": None,
                    "parent_fitness": fitness(sequence),
                }
                for sequence in prefix.accepted_sequences[:128]
            ]
            parents = prefix.accepted_sequences[128:]
            seen = set(map(sequence_key, prefix.accepted_sequences))
            charged = set(map(sequence_key, (row.sequence for row in history.observations)))
            training = set(context.driver.training_sequence_keys)
            path_seed = int(
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
            partial = False
            for slots in raw["native_batches"]:
                require(
                    not partial
                    and 1
                    <= len(slots)
                    <= min(
                        256 - len(pool),
                        ATTEMPT_CAP - len(prefix.attempts) - ordinal + raw["native_ordinal"],
                    ),
                    "GA endpoint native batch planning budget/order differs",
                )
                require(
                    len({slot["consumed"] for slot in slots}) == 1,
                    "GA endpoint batch consumed only in part",
                )
                for slot in slots:
                    index = order[ordinal % 10]
                    parent_index = (ordinal - raw["native_ordinal"]) % 128
                    require(
                        (slot["ordinal"], slot["triple"], slot["parent_index"])
                        == (ordinal, units[index].triple, parent_index),
                        "GA endpoint attempted checkpoint/parent assignment differs",
                    )
                    if slot["trace"] is not None:
                        trace = _trace(slot["trace"])
                        require(
                            (trace.parent, trace.ordinal, trace.seed, trace.start_level)
                            == (parents[parent_index], ordinal, path_seed, 32),
                            "GA endpoint conditional path input differs",
                        )
                        replay_native_trace(working[index].model, trace, authenticate_sampling=True)
                    if slot["consumed"]:
                        require(slot["trace"] is not None, "GA endpoint consumed missing trace")
                        seq = slot["trace"]["endpoint"]
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
                        require(
                            slot["rejection"] == rejection,
                            "GA endpoint collision/duplicate credit differs",
                        )
                        seen.add(key)
                        if rejection is None:
                            pool.append(
                                {
                                    "sequence": seq,
                                    "branch": "native_refinement",
                                    "native_ordinal": ordinal,
                                    "parent_fitness": fitness(parents[parent_index]),
                                }
                            )
                    else:
                        require(
                            slot["rejection"] is None,
                            "GA endpoint unconsumed attempt has a decision",
                        )
                        partial = True
                    ordinal += 1
        else:
            require(
                not raw["native_batches"], "GA endpoint native work precedes complete GA prefix"
            )
    else:
        require(not raw["native_batches"], "GA endpoint native work lacks GA source")
    _same(raw["pool"], pool, "GA endpoint128+128 pool/parent credit differs")
    require(raw["next_native_ordinal"] == ordinal, "GA endpoint global attempt advancement differs")
    predictions = raw["predictions"]
    require(
        len(predictions) <= 2 and (not predictions or len(pool) == 256),
        "GA endpoint scoring cap differs",
    )
    if predictions:
        binding = FrozenNativePosteriorBinding(**raw["posterior_binding"])
        require(
            (
                binding.history_sha256,
                binding.objective_context_sha256,
                binding.feature_source_sha256,
                binding.evaluator_source_sha256,
            )
            == (
                history.sha256,
                context.objective.context_sha256,
                context.feature_source_sha256,
                context.evaluator_source_sha256,
            ),
            "GA endpoint posterior source differs",
        )
    scores = []
    for offset, batch in enumerate(predictions):
        require(
            batch["sequence_ids"]
            == [sequence_key(row["sequence"]) for row in pool[128 * offset : 128 * offset + 128]]
            and len(batch["scores"]) == 128
            and hash_string(batch["receipt_sha256"]),
            "GA endpoint posterior order/receipt differs",
        )
        for row in batch["scores"]:
            score = NativePosteriorScore(tuple(row["objectives"]), row["feasible"])
            score.validate(context.objective)
            scores.append(score)
    ready = raw["status"] == "ready_private_composition_required"
    ranked = (
        sorted(
            (i for i, score in enumerate(scores) if score.feasible),
            key=lambda i: (
                -context.objective.scalarize(scores[i].objectives),
                -pool[i]["parent_fitness"],
                sequence_key(pool[i]["sequence"]),
            ),
        )
        if len(scores) == 256
        else []
    )
    if ready:
        require(
            len(pool) == 256 and len(ranked) >= 14 and raw["error"] is None,
            "GA endpoint ready result incomplete",
        )
        require(
            sum(row["branch"] == "unaltered_ga" for row in pool) == 128,
            "GA endpoint branch mixture differs",
        )
    elif raw["status"] in (
        "paused_incomplete_wave",
        "budget_complete_pending_controller_terminal",
        "abstained_no_successful_parent",
    ):
        require(
            prefix is None and teacher is None and raw["update"] is None and not predictions,
            "GA endpoint non-adaptive status contains adaptive work",
        )
        if raw["status"] == "paused_incomplete_wave":
            require(not history.complete, "GA endpoint paused status lacks incomplete history")
        elif raw["status"] == "budget_complete_pending_controller_terminal":
            require(
                history.complete and history.round_index == 29,
                "GA endpoint terminal denominator differs",
            )
        else:
            require(
                history.complete
                and history.round_index <= 28
                and not any(row.status == "successful" for row in history.observations),
                "GA endpoint parent abstention lacks failed history",
            )
    elif raw["status"] == "abstained_incomplete_ga_prefix":
        require(
            prefix is not None and prefix.status == "attempt_cap_exhausted",
            "GA endpoint GA-prefix abstention lacks exhausted budget",
        )
    elif raw["status"] == "abstained_incomplete_native_branch":
        require(
            prefix is not None
            and prefix.status == "complete"
            and len(pool) < 256
            and len(prefix.attempts) + ordinal - raw["native_ordinal"] == ATTEMPT_CAP,
            "GA endpoint native abstention lacks exhausted joint budget",
        )
    elif raw["status"] == "abstained_insufficient_feasible_candidates":
        require(
            len(scores) == 256 and len(ranked) < 14,
            "GA endpoint abstention lacks infeasible evidence",
        )
    elif raw["status"] not in (
        "paused_incomplete_wave",
        "budget_complete_pending_controller_terminal",
        "abstained_no_successful_parent",
        "abstained_incomplete_ga_prefix",
        "abstained_incomplete_native_branch",
        "stopped_deadline",
        "stopped_numerical_or_provider_failure",
    ):
        raise ValueError("GA endpoint unknown terminal status")
    if raw["status"].startswith("stopped_"):
        require(isinstance(raw["error"], dict), "GA endpoint stopped result lacks failure record")
        require(
            (raw["error"].get("type") == "TimeoutError") == (raw["status"] == "stopped_deadline"),
            "GA endpoint stop status/error type differs",
        )
        ranked = []
    else:
        require(raw["error"] is None, "GA endpoint successful/abstained status carries an error")
    require(raw["ranked_positions"] == ranked, "GA endpoint posterior ranking differs")
    committed = working if ready else units
    _same(
        raw["committed_models"],
        [(unit.triple, unit.policy_sha256) for unit in committed],
        "GA endpoint atomic commit differs",
    )
    require(
        raw["next_behavior_version"] == (version if ready else raw["behavior_version"]),
        "GA endpoint committed version differs",
    )
    require(
        (
            result.status,
            result.ranked_sequences,
            result.next_native_ordinal,
            result.next_behavior_version,
        )
        == (
            raw["status"],
            tuple(pool[i]["sequence"] for i in ranked),
            ordinal,
            raw["next_behavior_version"],
        ),
        "GA endpoint summary differs",
    )
    if expected_committed_units is not None:
        _same(
            raw["committed_models"],
            [
                (unit.triple, canonical_model_logical_hash(unit.model))
                for unit in expected_committed_units
            ],
            "GA endpoint returned weights differ",
        )
    if phase is not None:
        names = ["wave.json"] + [f"working_models/{triple}.safetensors" for triple in TRIPLES]
        seal = verify_phase(phase, expected_artifact=ARTIFACT, expected_payload_paths=names)
        require(
            seal.read_payload_bytes("wave.json") == result.record_json.encode(),
            "GA endpoint persisted receipt differs",
        )
        for triple, data in result.checkpoint_payloads:
            require(
                seal.read_payload_bytes(f"working_models/{triple}.safetensors") == data,
                "GA endpoint persisted neural bytes differ",
            )
    return committed, {
        "reconstructed": True,
        "status": result.status,
        "scientific_evidence_accepted": False,
    }

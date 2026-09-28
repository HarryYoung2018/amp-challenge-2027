"""Independent partial-GA receipt reconstruction, not a producer replay.

Reuses accepted native gradient/kernel/trace primitives and separately authored
eligible-prefix/shared-update checks. It never calls the new arm, guard or update
producer, samples diagnostic paths, queries a posterior, or authenticates external
failure causes, clocks, oracle truth, or real-runtime capacity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict

import numpy as np
import torch
from safetensors.torch import save

from amp_challenge.generators.diffusion.model import (
    _canonical_model_state,
    canonical_model_logical_hash,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    _state_contract,
    _validated_transition_kernels,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_ga_endpoint_records import (
    ChargedEndpointEligibility,
    GAEndpointContext,
)
from amp_challenge.generators.diffusion.native_ga_partial_records import (
    CONFIG_SHA256,
    ELIGIBLE_CONTRACT_SHA256,
    GA_MATCHED_CONFIG_SHA256,
    MAX_RECEIPT_BYTES,
    MEASURE,
    PartialGAWave,
    PartialGuardedUpdate,
    PartialGuardPlan,
    plain,
    source_identities,
    unit_binding,
)
from amp_challenge.generators.diffusion.native_proposals import replay_native_trace
from amp_challenge.generators.diffusion.native_search_posterior import (
    FrozenNativePosteriorBinding,
    NativePosteriorScore,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import SharedEndpointUpdate
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    ENDPOINT_CONFIG_SHA256,
    TRIPLES,
    EndpointTarget,
    EndpointTeacher,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import (
    _admission,
    _record_contract,
    _reference_paths,
    _replay,
    _same,
    _source_bytes,
    _trace,
    verify_shared_endpoint_update,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.diffusion.replay import summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import complete_subset_commit_kl
from amp_challenge.generators.search.peptide_ga_driver_v2 import PrivateGACollisions
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    EligibleGAKernelInput,
    EligibleGAPrefix,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3_verify import verify_eligible_prefix
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_json_bytes,
    digest,
    hash_string,
    sequence_key,
)
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

_COUNTS = ("row_forwards", "forward_calls", "grad_enabled_forwards", "backward_calls")
_READY = "ready_private_composition_required"
_ACCEPTED = {
    "accepted_guarded_update",
    "accepted_no_kl_update",
    "qualified_unchanged_policy",
    "accepted_unchanged_no_kl_caps_recorded",
}
_FAILURES = {
    "partial_deadline_no_commit",
    "numerical_or_guard_failure_no_commit",
    "final_integrity_no_commit",
    "diagnostic_byte_cap_no_commit",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(receipt, expected_type, artifact):
    _require(type(receipt) is expected_type, "partial verifier receipt type differs")
    _require(
        all(
            getattr(receipt, field) is False
            for field in (
                "campaign_eligible",
                "scientific_evidence_accepted",
                "production_eligible",
            )
        ),
        "partial verifier receipt acquired authority",
    )
    _require(type(receipt.record_json) is str, "partial verifier JSON type differs")
    data = receipt.record_json.encode()
    _require(
        len(data) <= MAX_RECEIPT_BYTES and hashlib.sha256(data).hexdigest() == receipt.sha256,
        "partial verifier receipt byte limit/seal differs",
    )
    raw = json.loads(data)
    _require(
        type(raw) is dict and raw.get("artifact") == artifact, "partial verifier artifact differs"
    )
    _require(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() == data,
        "partial verifier JSON is not canonical/finite",
    )
    _require(
        all(
            raw.get(key) is False
            for key in ("campaign_eligible", "scientific_evidence_accepted", "production_eligible")
        ),
        "partial verifier payload acquired authority",
    )
    _require(raw.get("status") == receipt.status, "partial verifier status summary differs")
    return raw


def _add(work, phase, rows, calls, backwards=0):
    if not rows and not calls and not backwards:
        return
    target = work.setdefault(phase, dict.fromkeys(_COUNTS, 0))
    target["row_forwards"] += rows
    target["forward_calls"] += calls
    target["grad_enabled_forwards"] += backwards
    target["backward_calls"] += backwards


def _path_cost(raw):
    active = sum(bool(step["draw"]["positions"]) for step in raw["steps"])
    return active, math.ceil(active / 16)


def _sampling_cost(traces):
    levels = {}
    for trace in traces:
        for step in trace["steps"]:
            levels[step["level"]] = levels.get(step["level"], 0) + bool(step["draw"]["positions"])
    return sum(levels.values()), sum(math.ceil(value / 16) for value in levels.values())


def _fullmask_cost(raw):
    rows, calls = _sampling_cost(raw["traces"])
    for trace in raw["traces"]:
        count, batches = _path_cost(trace)
        rows += 3 * count
        calls += 3 * batches
    return rows, calls


def _work_contract(raw, known, *, exact, wave=False, matched=False):
    _require(
        type(raw) is dict
        and set(raw)
        == {
            "phases",
            "total",
            "shared_inner_subphase_split_measured",
            "external_feature_model_work_counted_here",
        },
        "partial work record shape differs",
    )
    _require(
        raw["shared_inner_subphase_split_measured"] is False
        and raw["external_feature_model_work_counted_here"] is False,
        "partial work record overclaims its measurement scope",
    )
    _require(type(raw["phases"]) is dict, "partial work phases differ")
    for phase, counts in raw["phases"].items():
        _require(
            type(phase) is str and type(counts) is dict and set(counts) == set(_COUNTS),
            "partial work phase/count inventory differs",
        )
        _require(
            all(type(value) is int and value >= 0 for value in counts.values()),
            "partial work counts must be nonnegative integers",
        )
        _require(
            counts["backward_calls"] <= counts["grad_enabled_forwards"] <= counts["forward_calls"],
            "partial gradient/forward count ordering differs",
        )
    totals = {key: sum(row[key] for row in raw["phases"].values()) for key in _COUNTS}
    _same(raw["total"], totals, "partial work total does not sum phases")
    row_cap, call_cap = 390400 + (92160 if matched else 0), 28880 + (10944 if matched else 0)
    if wave:
        row_cap += 2088960
        call_cap += 2088960
    _require(
        totals["row_forwards"] <= row_cap
        and totals["forward_calls"] <= call_cap
        and totals["backward_calls"] <= 320,
        "partial work exceeds declared producer cap",
    )
    if exact:
        _same(
            raw["phases"],
            known,
            "partial complete work counts differ from reconstructed native calls",
        )
    else:
        for phase, row in known.items():
            observed = raw["phases"].get(phase, dict.fromkeys(_COUNTS, 0))
            _require(
                all(observed[key] >= row[key] for key in _COUNTS),
                "partial work omits independently completed operations",
            )


def _plan_key(plan):
    value = _json_hash(
        [
            "ga-partial-guard-v2",
            plan.seed,
            plan.round_index,
            plan.kernel_semantic_sha256,
            plan.native_ordinal,
            plan.remaining_attempts,
            plan.checkpoint_order,
            plan.parents,
        ]
    )
    _require(value == plan.numerical_sha256, "partial plan semantic identity differs")
    return value


def _draws(plan, triple):
    student = TRIPLES.index(triple)
    assigned_residue = plan.checkpoint_order.index(student)
    first = plan.native_ordinal + ((assigned_residue - plan.native_ordinal) % 10)
    count = len(range(first, plan.native_ordinal + plan.remaining_attempts, 10))
    _require(count > 0, "partial student has no legal ordinal support")
    result = []
    for replicate in range(8):
        identity = _json_hash([_plan_key(plan), triple, replicate, "iid-legal-ordinal"])
        rng = np.random.Generator(np.random.PCG64DXSM(int(identity[:32], 16)))
        ordinal = first + 10 * int(rng.integers(count))
        index = (ordinal - plan.native_ordinal) % 128
        result.append(
            {
                "replicate": replicate,
                "legal_native_ordinal": ordinal,
                "parent_index": index,
                "parent": plan.parents[index],
                "legal_student_count": count,
                "log_probability": -math.log(count),
            }
        )
    return result


def _path_seed(plan, triple, role):
    return int(_json_hash([_plan_key(plan), triple, role, "probe-paths"])[:16], 16)


def _trace_for(raw, model, *, parent, seed, ordinal, start=32):
    trace = _trace(raw)
    _require(
        (trace.parent, trace.seed, trace.ordinal, trace.start_level)
        == (parent, seed, ordinal, start),
        "partial native parent/seed/replicate/start differs",
    )
    return trace, replay_native_trace(model, trace, authenticate_sampling=True)


def _active(terms, states, model):
    values = [
        float(value)
        for value, state in zip(terms, states, strict=True)
        if _state_contract(state, model.config)[1] > 0
    ]
    _require(
        values and all(math.isfinite(value) and value >= 0 for value in values),
        "partial active conditional KL is empty/nonfinite/negative",
    )
    return values


def _path_summary(paths):
    values, weights = [], []
    for row in paths:
        values.extend(row["active"])
        weights.extend([1 / (len(paths) * len(row["active"]))] * len(row["active"]))
    return asdict(summarize_kl(values, weights=weights))


def _old_occupancy(units, plan, rows, work):
    _require(type(rows) is list and len(rows) <= 10, "partial old occupancy count differs")
    result = {}
    incomplete = False
    for unit, row in zip(units, rows, strict=False):
        _require(
            not incomplete and row["triple"] == unit.triple and type(row["authenticated"]) is bool,
            "partial old occupancy order/completion differs",
        )
        draws = _draws(plan, unit.triple)
        _same(row["draws"], draws, "partial old parent IID law differs")
        _require(
            len(row["traces"]) in (0, 8),
            "partial old sampling batch is neither absent nor complete",
        )
        states = []
        for index, trace in enumerate(row["traces"]):
            _, (_, visited) = _trace_for(
                trace,
                unit.model,
                parent=draws[index]["parent"],
                seed=_path_seed(plan, unit.triple, "old"),
                ordinal=index,
            )
            states.append(visited)
        if row["traces"]:
            _add(work, "partial_old_sampling", *_sampling_cost(row["traces"]))
        if row["authenticated"]:
            _require(len(states) == 8, "partial authenticated old row lacks eight paths")
            for trace in row["traces"]:
                _add(work, "partial_old_authentication", *_path_cost(trace))
            result[unit.triple] = tuple(states)
        else:
            incomplete = True
    return result


def _partial_record(unit, candidate, plan, b, raw, old_states, work):
    _require(
        raw["triple"] == unit.triple
        and raw["candidate_index"] == b
        and raw["plan_sha256"] == plan.sha256
        and raw["measure"] == MEASURE
        and type(raw["complete"]) is bool,
        "partial candidate record plan/ordinal/measure differs",
    )
    _same(
        raw["models"],
        [canonical_model_logical_hash(model) for model in (unit.model, candidate, unit.reference)],
        "partial candidate old/current/reference identities differ",
    )
    _require(
        len(old_states) == 8 and len(raw["local_paths"]) <= 8 and len(raw["candidate_paths"]) <= 8,
        "partial conditional path inventory differs",
    )
    for states, row in zip(old_states, raw["local_paths"], strict=False):
        old = _validated_transition_kernels(unit.model, states, NATIVE_ENDPOINT_DEFAULTS)
        current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
        terms = [complete_subset_commit_kl(a, b) for a, b in zip(old, current, strict=True)]
        _same(
            row,
            {"transition_kl": terms, "active": _active(terms, states, unit.model)},
            "partial old-occupancy local KL differs",
        )
        count = sum(_state_contract(state, unit.model.config)[1] > 0 for state in states)
        _add(work, "partial_old_candidate_kernels", 2 * count, 2 * math.ceil(count / 16))
    _require(
        not raw["candidate_paths"] or len(raw["local_paths"]) == 8,
        "partial candidate occupancy precedes complete local diagnostics",
    )
    draws = _draws(plan, unit.triple)
    complete_paths = []
    incomplete = False
    for index, row in enumerate(raw["candidate_paths"]):
        _require(not incomplete, "partial path continues after incomplete diagnostic")
        _same(row["draw"], draws[index], "partial candidate parent IID law differs")
        trace, (current_logp, states) = _trace_for(
            row["trace"],
            candidate,
            parent=draws[index]["parent"],
            seed=_path_seed(plan, unit.triple, "candidate"),
            ordinal=index,
        )
        if row["transition_kl"] is None:
            _require(
                row["active"] is None and row["path_sum"] is None,
                "partial unfinished path has completed scalar fields",
            )
            incomplete = True
            continue
        reference_logp, _ = replay_native_trace(unit.reference, trace)
        current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
        reference = _validated_transition_kernels(unit.reference, states, NATIVE_ENDPOINT_DEFAULTS)
        terms = [complete_subset_commit_kl(a, b) for a, b in zip(current, reference, strict=True)]
        expected = {
            "draw": draws[index],
            "trace": row["trace"],
            "transition_kl": terms,
            "active": _active(terms, states, candidate),
            "path_sum": math.fsum(terms),
            "current_joint_log_probability": draws[index]["log_probability"] + current_logp,
            "reference_joint_log_probability": draws[index]["log_probability"] + reference_logp,
            "sampled_conditional_log_ratio": current_logp - reference_logp,
        }
        _same(row, expected, "partial candidate-reference path KL/log probabilities differ")
        complete_paths.append(row)
        count, calls = _path_cost(row["trace"])
        _add(work, "partial_candidate_reference_replay", 2 * count, 2 * calls)
        _add(work, "partial_candidate_reference_kernels", 2 * count, 2 * calls)
    if raw["candidate_paths"]:
        # A partial recorded suffix proves at least these returned sampling rows,
        # not the unrecorded remainder of the original eight-row native batch.
        _add(
            work,
            "partial_candidate_sampling",
            *_sampling_cost([row["trace"] for row in raw["candidate_paths"]]),
        )
    if raw["complete"]:
        _require(
            len(raw["local_paths"]) == len(complete_paths) == 8,
            "partial complete diagnostic lacks all local/current paths",
        )
        values = np.asarray([row["path_sum"] for row in complete_paths])
        _same(
            raw["local_summary"],
            _path_summary(raw["local_paths"]),
            "partial local equal-path quantiles differ",
        )
        _same(
            raw["reference_active_summary"],
            _path_summary(complete_paths),
            "partial reference equal-path quantiles differ",
        )
        _require(
            raw["path_mean"] == float(values.mean())
            and raw["path_mcse"] == float(values.std(ddof=1) / math.sqrt(8)),
            "partial path mean/MCSE differs",
        )
    else:
        _require(
            all(
                raw[key] is None
                for key in ("local_summary", "path_mean", "path_mcse", "reference_active_summary")
            ),
            "partial incomplete diagnostic carries completed summary",
        )


def _combined(shared, partial, plan):
    _require(
        len(shared) == len(partial) == 10 and all(row["complete"] for row in partial),
        "partial mixture lacks ten complete students",
    )
    counts = [_draws(plan, triple)[0]["legal_student_count"] for triple in TRIPLES]
    _require(
        sum(counts) == plan.remaining_attempts, "partial legal masses do not sum total support"
    )
    mixtures = []
    for label, masses in (
        ("equal_ten", [0.1] * 10),
        ("legal_ordinal_mass", [value / plan.remaining_attempts for value in counts]),
    ):
        full_values, full_weights, local_values, local_weights, ref_values, ref_weights = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for mass, old, current in zip(masses, shared, partial, strict=True):
            values = old["fullmask_reference"]["active_transitions"]
            full_values.extend(values)
            full_weights.extend([mass / len(values)] * len(values))
            for paths, values, weights in (
                (current["local_paths"], local_values, local_weights),
                (current["candidate_paths"], ref_values, ref_weights),
            ):
                for path in paths:
                    values.extend(path["active"])
                    weights.extend([mass / (8 * len(path["active"]))] * len(path["active"]))
        mixtures.append(
            {
                "mixture": label,
                "student_masses": masses,
                "old_local_mean": math.fsum(
                    mass * row["local"]["local_old_candidate"]["mean"]
                    for mass, row in zip(masses, shared, strict=True)
                ),
                "old_local_p99_upper_bound": max(
                    row["local"]["local_old_candidate"]["p99"] for row in shared
                ),
                "fullmask_mean": math.fsum(
                    mass * row["fullmask_reference"]["mean"]
                    for mass, row in zip(masses, shared, strict=True)
                ),
                "fullmask_mcse": math.sqrt(
                    math.fsum(
                        (mass * row["fullmask_reference"]["monte_carlo_standard_error"]) ** 2
                        for mass, row in zip(masses, shared, strict=True)
                    )
                ),
                "fullmask_active": asdict(summarize_kl(full_values, weights=full_weights)),
                "partial_local": asdict(summarize_kl(local_values, weights=local_weights)),
                "partial_path_mean": math.fsum(
                    mass * row["path_mean"] for mass, row in zip(masses, partial, strict=True)
                ),
                "partial_path_mcse": math.sqrt(
                    math.fsum(
                        (mass * row["path_mcse"]) ** 2
                        for mass, row in zip(masses, partial, strict=True)
                    )
                ),
                "partial_reference_active": asdict(summarize_kl(ref_values, weights=ref_weights)),
            }
        )
    per_student = all(
        old["local"]["local_old_candidate"]["mean"] <= 0.01
        and old["local"]["local_old_candidate"]["p99"] <= 0.02
        and old["fullmask_reference"]["mean"] <= 0.08
        and old["fullmask_reference"]["active_summary"]["p99"] <= 0.02
        and current["local_summary"]["mean"] <= 0.01
        and current["local_summary"]["p99"] <= 0.02
        and current["path_mean"] <= 0.08
        and current["reference_active_summary"]["p99"] <= 0.02
        for old, current in zip(shared, partial, strict=True)
    )
    passed = per_student and all(
        row["old_local_mean"] <= 0.01
        and row["old_local_p99_upper_bound"] <= 0.02
        and row["fullmask_mean"] <= 0.08
        and row["fullmask_active"]["p99"] <= 0.02
        and row["partial_local"]["mean"] <= 0.01
        and row["partial_local"]["p99"] <= 0.02
        and row["partial_path_mean"] <= 0.08
        and row["partial_reference_active"]["p99"] <= 0.02
        for row in mixtures
    )
    return {"mixtures": mixtures, "all_per_student_passed": per_student, "all_kl_passed": passed}


def _shared_summary(rows):
    mean = math.fsum(row["fullmask_reference"]["mean"] for row in rows) / 10
    mcse = (
        math.sqrt(
            math.fsum(row["fullmask_reference"]["monte_carlo_standard_error"] ** 2 for row in rows)
        )
        / 10
    )
    values, masses = [], []
    for row in rows:
        active = row["fullmask_reference"]["active_transitions"]
        values.extend(active)
        masses.extend([1 / (10 * len(active))] * len(active))
    summary = asdict(summarize_kl(values, weights=masses))
    passed = (
        all(
            row["local"]["local_old_candidate"]["mean"] <= 0.01
            and row["local"]["local_old_candidate"]["p99"] <= 0.02
            and row["fullmask_reference"]["mean"] <= 0.08
            and row["fullmask_reference"]["active_summary"]["p99"] <= 0.02
            for row in rows
        )
        and mean <= 0.08
        and summary["p99"] <= 0.02
    )
    return {
        "reference_measure": "FULLMASK_GENERATOR_PATH",
        "equal_mixture_path_mean": mean,
        "stratified_mixture_mcse": mcse,
        "equal_component_active_transition_summary": summary,
        "per_student_and_mixture_kl_passed": passed,
    }


def _shared_proposal(units, teacher, wrapped, work):
    """Rebuild the one private four-step proposal and b0 diagnostics only once."""
    receipt = SharedEndpointUpdate(**wrapped)
    _require(
        receipt.kl_enforced is False
        and receipt.operator_guard_present is False
        and receipt.campaign_eligible is False
        and receipt.production_eligible is False
        and receipt.scientific_evidence_accepted is False,
        "partial private proposal changes its private no-KL qualification",
    )
    if not receipt.accepted:
        verify_shared_endpoint_update(units, teacher, receipt)
        raw = json.loads(receipt.record_json)
        _require(
            raw["enforce_kl"] is False
            and raw["operator_source_sha256"] is None
            and raw["operator_plan_sha256"] is None
            and len(raw["candidates"]) <= 1,
            "partial failed private proposal changes the b0/no-operator contract",
        )
        # The accepted independent verifier above has already reconstructed these
        # operations. Count only fully recorded work; never redo its neural steps
        # or guess the cost of the interrupted call that produced no record.
        for row in raw["training"]:
            for step in row["steps"]:
                n = len(step["replay"]["states"])
                _add(work, "shared_private_proposal", n, math.ceil(n / 16), math.ceil(n / 16))
        for candidate in raw["candidates"]:
            _require(candidate["backtracks"] == 0, "partial failed private proposal is not b0")
            for index, row in enumerate(candidate["students"]):
                n = len(raw["training"][index]["probes"]["states"])
                _add(work, "shared_private_proposal", 3 * n, 3 * math.ceil(n / 16))
                if row["fullmask_reference"] is not None:
                    _add(
                        work, "shared_private_proposal", *_fullmask_cost(row["fullmask_reference"])
                    )
        return None, None, raw
    _require(
        not receipt.kl_enforced and not receipt.operator_guard_present and receipt.backtracks == 0,
        "partial private proposal changed no-KL/b0/operator contract",
    )
    _require(
        len(receipt.record_json.encode()) <= MAX_RECEIPT_BYTES
        and hashlib.sha256(receipt.record_json.encode()).hexdigest() == receipt.sha256,
        "partial private proposal seal differs",
    )
    raw = json.loads(receipt.record_json)
    _record_contract(raw)
    _source_bytes(raw["source_identities"])
    _require(
        raw["enforce_kl"] is False
        and raw["accepted"] is True
        and raw["status"] == receipt.status == "accepted_no_kl_enforcement_caps_recorded"
        and raw["backtracks"] == 0
        and raw["operator_source_sha256"] is None
        and raw["operator_plan_sha256"] is None
        and len(raw["training"]) == 10
        and len(raw["candidates"]) == 1,
        "partial private proposal was not exactly one complete four-step b0",
    )
    _same(raw["teacher"], asdict(teacher), "partial private proposal teacher differs")
    _require(
        raw["teacher_sha256"] == teacher.sha256
        and raw["teacher_semantic_sha256"] == teacher.semantic_sha256
        and raw["config_sha256"] == ENDPOINT_CONFIG_SHA256,
        "partial private proposal teacher/config pin differs",
    )
    _same(
        raw["old_models"],
        [(unit.triple, unit.policy_sha256, unit.reference_sha256) for unit in units],
        "partial private proposal initial units differ",
    )
    training_ids = set().union(*(unit.initialization.training_sequence_ids for unit in units))
    _require(
        not training_ids.intersection(row.sequence_id for row in teacher.targets),
        "partial private target overlaps generator training inventory",
    )
    admission, weights = _admission(teacher)
    _same(raw["admission"], admission, "partial private teacher concentration differs")
    _require(admission["admitted"], "partial private update lacks admitted targets")
    proposals, probes = [], []
    for unit, row in zip(units, raw["training"], strict=True):
        anchors = tuple(
            sorted(
                unit.sequences,
                key=lambda seq: _json_hash([raw["seed"], unit.triple, "endpoint-anchor", seq]),
            )[:64]
        )
        _require(
            row["triple"] == unit.triple
            and tuple(row["anchors"]) == anchors
            and len(row["steps"]) == 4,
            "partial private anchor/step inventory differs",
        )
        sequences = anchors + tuple(row.sequence for row in teacher.targets)
        mixture = np.concatenate([np.full(64, 0.5 / 64), 0.5 * weights])
        seed = int(
            _json_hash([raw["seed"], teacher.semantic_sha256, unit.triple, "endpoint-training"])[
                :16
            ],
            16,
        )
        model = unit.model
        for ordinal, step in enumerate(row["steps"]):
            replay = _replay(step["replay"])
            expected = build_weighted_replay(
                model,
                sequences,
                mixture,
                context_id=teacher.objective_context_sha256,
                seed=seed,
                ordinal=ordinal,
            )
            _require(
                replay.sha256 == expected.sha256, "partial fresh-corruption/target weights differ"
            )
            direction = propose_weighted_direction(model, replay)
            gradient = hashlib.sha256()
            for name, value in direction.gradients:
                gradient.update(name.encode() + b"\0" + value.numpy().tobytes())
            _require(
                (
                    step["objective_before"],
                    step["gradient_norm_before_clip"],
                    step["gradient_sha256"],
                )
                == (
                    direction.objective_before,
                    direction.gradient_norm_before_clip,
                    gradient.hexdigest(),
                ),
                "partial real neural direction/gradient norm differs",
            )
            model = endpoint_candidate(model, direction)
            _require(
                step["model_sha256"] == canonical_model_logical_hash(model),
                "partial four-step native weights differ",
            )
            n = len(replay.states)
            _add(work, "shared_private_proposal", n, math.ceil(n / 16), math.ceil(n / 16))
        _require(
            row["proposed_model_sha256"] == canonical_model_logical_hash(model),
            "partial private proposed-model identity differs",
        )
        replay = _replay(row["probes"])
        expected = build_weighted_replay(
            unit.model,
            sequences,
            mixture,
            context_id=teacher.objective_context_sha256,
            seed=seed,
            ordinal=4,
            active_probes=True,
        )
        _require(
            replay.sha256 == expected.sha256, "partial private local probe reconstruction differs"
        )
        proposals.append(model)
        probes.append(replay)
    b0 = raw["candidates"][0]
    _require(
        b0["backtracks"] == 0 and len(b0["students"]) == 10, "partial private b0 inventory differs"
    )
    for unit, model, probes_row, row in zip(units, proposals, probes, b0["students"], strict=True):
        _require(
            row["triple"] == unit.triple
            and row["candidate_sha256"] == canonical_model_logical_hash(model)
            and row["operator"] is None,
            "partial private b0 checkpoint/operator differs",
        )
        _same(
            row["local"],
            asdict(
                weighted_anchor_diagnostics(
                    unit.model, model, unit.reference, probes_row, NATIVE_ENDPOINT_DEFAULTS
                )
            ),
            "partial private b0 local KL differs",
        )
        count = len(probes_row.states)
        _add(work, "shared_private_proposal", 3 * count, 3 * math.ceil(count / 16))
        _reference_paths(
            unit, model, row["fullmask_reference"], raw["seed"], teacher.semantic_sha256
        )
        _add(work, "shared_private_proposal", *_fullmask_cost(row["fullmask_reference"]))
    _same(
        b0["summary"], _shared_summary(b0["students"]), "partial private b0 mixture summary differs"
    )
    _same(
        raw["new_models"],
        [
            (unit.triple, canonical_model_logical_hash(model))
            for unit, model in zip(units, proposals, strict=True)
        ],
        "partial private b0 replacement identities differ",
    )
    return tuple(proposals), tuple(probes), raw


def _interpolate(old, proposal, b):
    result = copy.deepcopy(proposal if b == 0 else old).eval()
    if b:
        proposed = dict(proposal.named_parameters())
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                parameter.add_(proposed[name] - parameter, alpha=2.0**-b)
    return result


def _compare_units(expected, actual, versions):
    _require(type(actual) is tuple and len(actual) == 10, "partial returned unit inventory differs")
    _same(
        [unit_binding(unit, versions[unit.triple]) for unit in actual],
        [unit_binding(unit, versions[unit.triple]) for unit in expected],
        "partial returned model/reference/corpus/initializer/version identities differ",
    )


def _update_core(units, teacher, plan, receipt, expected_enforce_kl, matched_feasibility=None):
    _require(
        type(expected_enforce_kl) is bool and type(plan) is PartialGuardPlan,
        "partial update verifier expects explicit mode and exact plan",
    )
    plan.__post_init__()
    _require(type(teacher) is EndpointTeacher, "partial update verifier teacher type differs")
    teacher.__post_init__()
    original_plan = canonical_json_bytes(asdict(plan))
    original_teacher = canonical_json_bytes(asdict(teacher))
    units = tuple(units)
    initial_binding = tuple(
        unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True)
    )
    _require(
        initial_binding == plan.units and tuple(unit.triple for unit in units) == TRIPLES,
        "partial update original caller inventory differs",
    )
    sources = source_identities()
    _require(_json_hash(sources) == plan.source_sha256, "partial update source plan differs")
    raw = _read(receipt, PartialGuardedUpdate, "native_ga_partial_update_v2")
    requirement = raw.get("matched_feasibility_requirement")
    _require(
        (requirement is not None) == (matched_feasibility is not None),
        "partial update lacks externally pinned GA feasibility requirement",
    )
    if matched_feasibility is not None:
        from amp_challenge.generators.diffusion.native_ga_matched_feasibility_verify import (
            _source,
            verify_ga_matched_feasibility,
        )

        _same(
            requirement,
            {
                "amendment_sha256": GA_MATCHED_CONFIG_SHA256,
                "predicate_sha256": matched_feasibility.predicate_sha256,
                "context_sha256": matched_feasibility.context_sha256,
                "source_sha256": matched_feasibility.source_sha256,
                "scope": "conditional_legal_ga_partial_native_attempt_not_selected_pool",
            },
            "partial update matched amendment/requirement differs",
        )
        _require(
            _source() == matched_feasibility.source_sha256
            and matched_feasibility.context_sha256 == teacher.objective_context_sha256
            and getattr(matched_feasibility.predicate, "source_sha256", None)
            == matched_feasibility.predicate_sha256,
            "partial update matched source/context differs",
        )
    _require(
        type(receipt.accepted) is bool
        and receipt.accepted == raw["accepted"]
        and receipt.backtracks == raw["backtracks"],
        "partial update acceptance summary differs",
    )
    _require(
        raw["status"] in _ACCEPTED | _FAILURES | {"all_backtracks_rejected_no_commit"},
        "partial update status unsupported",
    )
    work = {}
    if raw["status"] == "diagnostic_byte_cap_no_commit":
        _require(
            raw["accepted"] is False
            and raw["updated"] is False
            and raw["backtracks"] is None
            and raw["plan_sha256"] == plan.sha256
            and type(raw["completed_candidate_count"]) is int
            and 0 <= raw["completed_candidate_count"] <= 9,
            "partial byte-cap fallback cannot authorize a model",
        )
        _work_contract(raw["work"], {}, exact=False, matched=matched_feasibility is not None)
        _require(
            source_identities() == sources
            and canonical_json_bytes(asdict(plan)) == original_plan
            and canonical_json_bytes(asdict(teacher)) == original_teacher,
            "partial byte fallback inputs changed during inspection",
        )
        return units, raw, {}, False
    _same(raw["sources"], sources, "partial update actual source inventory differs")
    _same(raw["plan"], asdict(plan), "partial update plan differs from caller plan")
    _same(raw["teacher"], asdict(teacher), "partial update exact teacher differs")
    _require(
        raw["plan_sha256"] == plan.sha256
        and raw["teacher_sha256"] == teacher.sha256
        and raw["configuration_sha256"] == CONFIG_SHA256
        and raw["enforce_kl"] is expected_enforce_kl
        and raw["initial"] is (plan.round_index == 1)
        and raw["inner_accepted_is_arm_commit"] is False,
        "partial update source/context/mode/lifecycle differs",
    )
    _same(
        raw["old_models"],
        [unit.policy_sha256 for unit in units],
        "partial update old models differ",
    )
    admission, _ = _admission(teacher)
    _require(
        raw["admission"] is not None
        or not (raw["old_occupancy"] or raw["candidates"] or raw["accepted"]),
        "partial guard/acceptance omits its completed teacher admission",
    )
    if raw["admission"] is not None:
        _same(raw["admission"], admission, "partial update independent teacher admission differs")
    training = plan.round_index != 1 and admission["admitted"]
    proposals, probes, private = None, None, None
    if raw["private_shared_proposal"] is not None:
        _require(
            training and raw["admission"] is not None,
            "partial private SGD exists outside admitted noninitial lifecycle",
        )
        proposals, probes, private = _shared_proposal(
            units, teacher, raw["private_shared_proposal"], work
        )
        _require(private["seed"] == plan.seed, "partial private proposal seed differs from plan")
    if training and proposals is None:
        _require(
            not raw["old_occupancy"] and not raw["candidates"],
            "partial guard ran without the required complete private proposal",
        )
    old = _old_occupancy(units, plan, raw["old_occupancy"], work)
    entries = raw["candidates"]
    _require(
        type(entries) is list and len(entries) <= (9 if training else 1),
        "partial common-backtrack budget differs",
    )
    selected = None
    candidate_models = {}
    last_complete = True
    for b, entry in enumerate(entries):
        _require(
            selected is None
            and last_complete
            and entry["backtracks"] == b
            and len(old) == 10
            and len(entry["students"]) <= 10
            and len(entry["partial"]) <= len(entry["students"]),
            "partial candidate sequence or prerequisite occupancy differs",
        )
        for index, shared in enumerate(entry["students"]):
            unit = units[index]
            model = _interpolate(unit.model, proposals[index], b) if training else unit.model
            candidate_models[(b, index)] = model
            _require(
                shared["triple"] == unit.triple
                and shared["candidate_sha256"] == canonical_model_logical_hash(model)
                and shared["operator"] is None,
                "partial candidate checkpoint/operator differs",
            )
            if training and b == 0:
                _same(
                    shared,
                    private["candidates"][0]["students"][index],
                    "partial b0 did not reuse the checked private diagnostics",
                )
            else:
                if training:
                    expected = asdict(
                        weighted_anchor_diagnostics(
                            unit.model,
                            model,
                            unit.reference,
                            probes[index],
                            NATIVE_ENDPOINT_DEFAULTS,
                        )
                    )
                    count = len(probes[index].states)
                    _add(work, "outer_anchor_diagnostics", 3 * count, 3 * math.ceil(count / 16))
                else:
                    expected = {
                        "local_old_candidate": {"mean": 0.0, "p99": 0.0, "sample_count": 0},
                        "scope": "analytic_identical_old_candidate_no_training",
                    }
                _same(shared["local"], expected, "partial old/candidate anchor diagnostic differs")
                _reference_paths(
                    unit, model, shared["fullmask_reference"], plan.seed, teacher.semantic_sha256
                )
                _add(
                    work,
                    "outer_fullmask_diagnostics",
                    *_fullmask_cost(shared["fullmask_reference"]),
                )
            if index < len(entry["partial"]):
                row = entry["partial"][index]
                _partial_record(unit, model, plan, b, row, old[unit.triple], work)
                _require(row["complete"], "partial attached guard row is incomplete")
        last_complete = entry["summary"] is not None
        if last_complete:
            summary = _combined(entry["students"], entry["partial"], plan)
            _same(entry["summary"], summary, "partial common mixture/acceptance summary differs")
            feasible_passed = matched_feasibility is None
            matched = entry.get("matched_feasibility")
            if matched_feasibility is None:
                _require(matched is None, "partial update has unexpected matched report")
            elif not training:
                _same(
                    matched,
                    {
                        "status": "analytic_identical_policy_no_training",
                        "drop": 0.0,
                        "sampled_paths": 0,
                    },
                    "partial unchanged feasibility proof differs",
                )
                _require(
                    raw["matched_feasibility_plan_sha256"] is None,
                    "partial unchanged policy claims probe plan",
                )
                feasible_passed = True
            elif matched is not None and matched["status"] == "complete":
                _require(matched["candidate_index"] == b, "partial matched candidate order differs")
                _work_contract(matched["work_before"], work, exact=True, matched=True)
                report = verify_ga_matched_feasibility(
                    matched,
                    units=units,
                    proposals=proposals,
                    plan=plan,
                    predicate=matched_feasibility.predicate,
                    predicate_sha256=matched_feasibility.predicate_sha256,
                    context_sha256=matched_feasibility.context_sha256,
                    expected_source_sha256=matched_feasibility.source_sha256,
                    expected_plan_sha256=raw["matched_feasibility_plan_sha256"],
                )
                delta = report["sampling_work_delta"]
                _add(
                    work,
                    "ga_matched_feasibility_sampling",
                    delta["row_forwards"],
                    delta["forward_calls"],
                )
                _work_contract(matched["work_after"], work, exact=True, matched=True)
                feasible_passed = report["passed"] is True
            else:
                _require(
                    b == len(entries) - 1 and raw["status"] in _FAILURES,
                    "partial missing/incomplete matched report cannot advance",
                )
                _require(
                    matched is None or matched.get("status") == "failed",
                    "partial unmatched report status differs",
                )
                last_complete = False
            if (not expected_enforce_kl or summary["all_kl_passed"]) and feasible_passed:
                selected = b
    unfinished = raw["unfinished_partial_records"]
    _require(
        type(unfinished) is list and len(unfinished) <= 1,
        "partial unfinished guard ledger is not a single terminal suffix",
    )
    for row in unfinished:
        _require(
            entries and entries[-1]["summary"] is None,
            "unfinished guard follows a complete summary",
        )
        b = len(entries) - 1
        index = len(entries[-1]["partial"])
        _require(
            index < len(entries[-1]["students"]) and (b, index) in candidate_models,
            "unfinished guard is not the current next student",
        )
        _partial_record(
            units[index], candidate_models[(b, index)], plan, b, row, old[units[index].triple], work
        )
    accepted = raw["status"] in _ACCEPTED
    _require(
        raw["accepted"] is accepted and type(raw["updated"]) is bool,
        "partial update accepted/status relationship differs",
    )
    replacements = units
    if accepted:
        _require(
            selected is not None
            and raw["backtracks"] == selected
            and not unfinished
            and raw["error"] is None
            and raw["updated"] is training,
            "partial acceptance lacks its first complete passing candidate",
        )
        expected_status = (
            "accepted_guarded_update"
            if training and expected_enforce_kl
            else "accepted_no_kl_update"
            if training
            else "qualified_unchanged_policy"
            if expected_enforce_kl
            else "accepted_unchanged_no_kl_caps_recorded"
        )
        _require(
            raw["status"] == expected_status,
            "partial update status silently changes KL/training mode",
        )
        copies = []
        for index, unit in enumerate(units):
            result = copy.copy(unit)
            result.model = candidate_models[(selected, index)]
            result.policy_sha256 = canonical_model_logical_hash(result.model)
            copies.append(result)
        replacements = tuple(copies)
    else:
        _require(
            raw["updated"] is False and raw["backtracks"] is None,
            "partial no-commit receipt advances model state",
        )
        if raw["status"] == "all_backtracks_rejected_no_commit":
            _require(
                (expected_enforce_kl or matched_feasibility is not None)
                and selected is None
                and len(entries) == (9 if training else 1)
                and last_complete
                and not unfinished
                and raw["error"] is None,
                "partial all-KL-rejected lacks every failing common candidate",
            )
        else:
            _require(
                type(raw["error"]) is dict and type(raw["error"].get("message")) is str,
                "partial failure lacks a bounded diagnostic",
            )
            _require(
                raw["status"] != "partial_deadline_no_commit"
                or raw["error"].get("type") == "TimeoutError",
                "partial deadline status has incompatible diagnostic type",
            )
    _same(
        raw["new_models"],
        [unit.policy_sha256 for unit in replacements],
        "partial update atomic replacement model identities differ",
    )
    exact = accepted or raw["status"] == "all_backtracks_rejected_no_commit"
    _work_contract(raw["work"], work, exact=exact, matched=matched_feasibility is not None)
    _require(
        source_identities() == sources
        and canonical_json_bytes(asdict(plan)) == original_plan
        and canonical_json_bytes(asdict(teacher)) == original_teacher
        and tuple(unit_binding(unit, row[3]) for unit, row in zip(units, plan.units, strict=True))
        == initial_binding,
        "partial update source/original units changed during audit",
    )
    return replacements, raw, work, exact


def verify_partial_ga_update(
    units,
    teacher,
    plan,
    receipt,
    *,
    expected_enforce_kl,
    expected_new_units=None,
    matched_feasibility=None,
):
    """Offline current-math audit; partial failure cause/timing remain external."""
    replacements, raw, _, exact = _update_core(
        units, teacher, plan, receipt, expected_enforce_kl, matched_feasibility
    )
    if expected_new_units is not None:
        # unit_binding describes original behavior versions (0..27); the next
        # version is separately checked in the wave and may legitimately be 28.
        versions = {row[0]: row[3] for row in plan.units}
        _compare_units(replacements, expected_new_units, versions)
    return {
        "reconstructed": raw["status"] != "diagnostic_byte_cap_no_commit",
        "status": receipt.status,
        "accepted": receipt.accepted,
        "complete_native_work_reconstructed": exact,
        "matched_feasibility_enforced": matched_feasibility is not None,
        "failure_cause_authenticated": False,
        "external_timing_authenticated": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }


def _wave_teacher(history, eligibility, context, expected_ids):
    _require(
        type(eligibility) is ChargedEndpointEligibility,
        "partial wave requires the external charged endpoint eligibility record",
    )
    eligibility.validate(history, context)
    _require(
        frozenset(eligibility.query_ids) == expected_ids,
        "partial wave teacher and numerical-parent subsets differ",
    )
    origins = dict(zip(eligibility.query_ids, eligibility.origins, strict=True))
    eligible = [row for row in history.observations if row.query_id in expected_ids]
    _require(
        all(row.status == "successful" for row in eligible),
        "partial teacher eligibility includes an unsuccessful charged row",
    )
    ordered = sorted(
        eligible,
        key=lambda row: (-context.objective.scalarize(row.objectives), sequence_key(row.sequence)),
    )[:64]
    maximum = context.objective.scalarize(ordered[0].objectives) if ordered else 0.0
    return EndpointTeacher(
        tuple(
            EndpointTarget(
                row.sequence,
                "charged_endpoint",
                float(
                    max(
                        -math.log(2),
                        min(0.0, (context.objective.scalarize(row.objectives) - maximum) / 0.1),
                    )
                ),
                origins[row.query_id],
            )
            for row in ordered
        ),
        "charged_endpoint",
        context.objective.context_sha256,
        digest(canonical_json_bytes(asdict(eligibility))),
        min(28, history.round_index),
    )


def _wave_plan(kernel_input, prefix_result, bindings, source_sha, ordinal, deadline):
    history = kernel_input.history
    key = _json_hash(["native-ga-endpoint-v1", history.seed, "checkpoint-order"])
    order = tuple(
        int(value)
        for value in np.random.Generator(np.random.PCG64DXSM(int(key[:32], 16))).permutation(10)
    )
    return PartialGuardPlan(
        history.seed,
        history.round_index,
        ordinal,
        65536 - len(prefix_result.prefix.attempts),
        order,
        tuple(prefix_result.prefix.accepted_sequences[128:]),
        kernel_input.semantic_sha256,
        kernel_input.sha256,
        prefix_result.output_sha256,
        history.sha256,
        _json_hash(plain(asdict(kernel_input.eligibility))),
        float(deadline),
        bindings,
        source_sha,
    )


def _wave_pool(working, kernel_input, prefix, plan, raw, work):
    """Authenticate recorded batched events without generating a second stream."""
    history = kernel_input.history
    eligible_ids = kernel_input.eligibility.query_ids
    fitness = {
        sequence_key(row.sequence): math.fsum(0.5 * value for value in row.objectives)
        for row in history.observations
        if row.query_id in eligible_ids
    }
    edits = {row.edit.sequence: row.edit for row in prefix.attempts if row.rejection_reason is None}

    def parent_fitness(sequence):
        parents = edits[sequence].parent_sequence_keys
        return math.fsum(fitness[key] for key in parents) / len(parents)

    pool = [
        {
            "sequence": sequence,
            "branch": "unaltered_ga",
            "native_ordinal": None,
            "parent_fitness": parent_fitness(sequence),
        }
        for sequence in prefix.accepted_sequences[:128]
    ]
    _require(
        type(raw["pool"]) is list and 128 <= len(raw["pool"]) <= 256,
        "partial native pool omits or exceeds its fixed GA half",
    )
    seen = set(map(sequence_key, prefix.accepted_sequences))
    charged = {sequence_key(row.sequence) for row in history.observations}
    training = set(kernel_input.training_sequence_keys)
    seed = int(
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
    cursor = plan.native_ordinal
    limit = cursor + plan.remaining_attempts
    terminal_partial = False
    batches = raw["native_batches"]
    _require(
        type(batches) is list and len(batches) <= plan.remaining_attempts,
        "partial native batch ledger exceeds remaining paid attempt quota",
    )
    for slots in batches:
        _require(
            not terminal_partial and len(pool) < 256 and cursor < limit,
            "partial native batches continue after completion or interrupted suffix",
        )
        count = min(256 - len(pool), limit - cursor)
        _require(
            type(slots) is list and len(slots) == count,
            "partial native planned batch is not the bounded remaining-slot count",
        )
        for offset, row in enumerate(slots):
            ordinal = cursor + offset
            _require(
                type(row) is dict
                and set(row)
                == {"ordinal", "triple", "parent_index", "trace", "consumed", "rejection"}
                and type(row["ordinal"]) is int
                and type(row["parent_index"]) is int
                and (row["ordinal"], row["triple"], row["parent_index"])
                == (
                    ordinal,
                    working[plan.checkpoint_order[ordinal % 10]].triple,
                    (ordinal - plan.native_ordinal) % 128,
                )
                and type(row["consumed"]) is bool,
                "partial native assigned ordinal/checkpoint/parent differs",
            )
        cursor += count  # Planned ordinals are spent even if a later group fails.
        missing_group = False
        for index in plan.checkpoint_order:
            selected = [row for row in slots if row["triple"] == working[index].triple]
            if not selected:
                continue
            present = [row["trace"] is not None for row in selected]
            _require(
                all(present) or not any(present),
                "partial native checkpoint group has a non-atomic returned trace inventory",
            )
            if not any(present):
                missing_group = True
                continue
            _require(not missing_group, "partial native later group follows an unreturned group")
            for row in selected:
                _trace_for(
                    row["trace"],
                    working[index].model,
                    parent=plan.parents[row["parent_index"]],
                    seed=seed,
                    ordinal=row["ordinal"],
                )
            _add(
                work,
                "actual_native_candidate_generation",
                *_sampling_cost([row["trace"] for row in selected]),
            )
        unconsumed = False
        for row in slots:
            if not row["consumed"]:
                _require(row["rejection"] is None, "unconsumed native slot declares a rejection")
                unconsumed = True
                continue
            _require(
                not unconsumed and not missing_group and row["trace"] is not None,
                "partial native consumption is not a prefix after all groups returned",
            )
            sequence = row["trace"]["endpoint"]
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
            _require(
                row["rejection"] == rejection,
                "partial native collision/rejection reconstruction differs",
            )
            seen.add(key)
            if rejection is None:
                pool.append(
                    {
                        "sequence": sequence,
                        "branch": "native_refinement",
                        "native_ordinal": row["ordinal"],
                        "parent_fitness": parent_fitness(plan.parents[row["parent_index"]]),
                    }
                )
        terminal_partial = missing_group or unconsumed
    _same(raw["pool"], pool, "partial native pool order/parent-fitness/branch differs")
    _require(raw["next_native_ordinal"] == cursor, "partial native planned attempt counter differs")
    return pool, not terminal_partial


def _wave_scores(raw, pool, history, context, expected_binding):
    batches = raw["predictions"]
    _require(
        type(batches) is list and len(batches) <= 2, "partial native posterior batch cap differs"
    )
    if raw["posterior_binding"] is not None:
        _require(
            type(expected_binding) is FrozenNativePosteriorBinding,
            "partial recorded scores require externally expected posterior binding",
        )
        expected_binding.__post_init__()
        _same(
            raw["posterior_binding"],
            asdict(expected_binding),
            "partial recorded posterior differs from caller binding",
        )
        _require(
            (
                expected_binding.history_sha256,
                expected_binding.objective_context_sha256,
                expected_binding.feature_source_sha256,
                expected_binding.evaluator_source_sha256,
            )
            == (
                history.sha256,
                context.objective.context_sha256,
                context.feature_source_sha256,
                context.evaluator_source_sha256,
            ),
            "partial posterior history/context/feature/evaluator binding differs",
        )
        _require(len(pool) == 256, "partial posterior was exposed an underfilled pool")
    else:
        _require(not batches, "partial scores lack a frozen posterior binding")
    scores = []
    for batch_index, batch in enumerate(batches):
        _require(
            type(batch) is dict
            and set(batch) == {"sequence_ids", "scores", "receipt_sha256"}
            and hash_string(batch["receipt_sha256"])
            and len(batch["scores"]) == 128,
            "partial posterior receipt/shape differs",
        )
        _same(
            batch["sequence_ids"],
            [
                sequence_key(row["sequence"])
                for row in pool[128 * batch_index : 128 * (batch_index + 1)]
            ],
            "partial posterior sequence identities/order differ",
        )
        for row in batch["scores"]:
            _require(
                type(row) is dict and set(row) == {"objectives", "feasible"},
                "partial posterior score shape differs",
            )
            score = NativePosteriorScore(tuple(row["objectives"]), row["feasible"])
            score.validate(context.objective)
            scores.append(score)
    return sorted(
        (index for index, score in enumerate(scores) if score.feasible),
        key=lambda index: (
            -context.objective.scalarize(scores[index].objectives),
            -pool[index]["parent_fitness"],
            sequence_key(pool[index]["sequence"]),
        ),
    )


def verify_partial_ga_wave(
    units,
    kernel_input,
    prefix_result,
    teacher_eligibility,
    receipt,
    *,
    context,
    expected_history_sha256,
    expected_eligible_query_ids,
    expected_kernel_source_sha256,
    expected_eligibility_source_sha256,
    expected_prefix_deadline,
    expected_behavior_versions,
    native_ordinal,
    expected_enforce_kl,
    expected_new_units=None,
    expected_posterior_binding=None,
    private_collisions=None,
    expected_seats=None,
    matched_feasibility=None,
):
    """Offline reconstruction with explicit external pins, never a live evaluator.

    The producer's clock, unrecorded interrupted operations, external posterior
    fit/receipt authority and prior-prefix receipt chain remain outside this
    arithmetic audit. This audit consumes its own bounded CPU allocation.
    """
    _require(
        type(kernel_input) is EligibleGAKernelInput
        and type(prefix_result) is EligibleGAPrefix
        and type(context) is GAEndpointContext
        and type(expected_enforce_kl) is bool,
        "partial wave expected kernel/prefix/context/mode types differ",
    )
    context.__post_init__()
    history = kernel_input.history
    _require(type(history) is VerifiedHistorySnapshot, "partial wave history type differs")
    history.__post_init__()
    _require(
        (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
        )
        == (
            context.driver.run_id,
            context.driver.seed,
            context.driver.objective_context_sha256,
            context.driver.oracle_bundle_sha256,
        ),
        "partial wave context/run/history differs",
    )
    _require(
        type(native_ordinal) is int
        and 0 <= native_ordinal <= 28 * 65536
        and type(expected_behavior_versions) is dict
        and set(expected_behavior_versions) == set(TRIPLES)
        and type(expected_prefix_deadline) in (int, float)
        and math.isfinite(expected_prefix_deadline),
        "partial wave caller ordinal/version/deadline inventory differs",
    )
    versions = expected_behavior_versions.copy()
    units = tuple(units)
    _require(
        tuple(unit.triple for unit in units) == TRIPLES,
        "partial wave requires ten ordered students",
    )
    bindings = tuple(unit_binding(unit, versions[unit.triple]) for unit in units)
    _require(
        all(row[3] < history.round_index for row in bindings),
        "partial wave original versions claim a future generation",
    )
    sources = source_identities()
    source_sha = _json_hash(sources)
    _require(
        context.driver.implementation_sha256 == source_sha
        and (
            kernel_input.configuration_id,
            kernel_input.training_sequence_keys,
            expected_eligibility_source_sha256,
        )
        == (
            context.driver.configuration_id,
            context.driver.training_sequence_keys,
            context.eligibility_source_sha256,
        ),
        "partial wave implementation/config/exclusion pin differs",
    )
    training = set(kernel_input.training_sequence_keys)
    _require(
        not training.intersection(sequence_key(row.sequence) for row in history.observations)
        and set().union(*(unit.initialization.training_sequence_ids for unit in units)) <= training,
        "partial wave charged/training or generator-exclusion inventory differs",
    )
    original_input, original_prefix = kernel_input.sha256, prefix_result.canonical_bytes()
    original_context = canonical_json_bytes(asdict(context))
    pins = {
        "expected_kernel_source_sha256": expected_kernel_source_sha256,
        "expected_contract_sha256": ELIGIBLE_CONTRACT_SHA256,
        "expected_eligibility_source_sha256": expected_eligibility_source_sha256,
        "expected_objective_context_sha256": context.objective.context_sha256,
        "expected_history_sha256": expected_history_sha256,
        "expected_eligible_query_ids": expected_eligible_query_ids,
        "expected_deadline_monotonic": expected_prefix_deadline,
    }
    verify_eligible_prefix(prefix_result, kernel_input, **pins)
    raw = _read(receipt, PartialGAWave, "native_ga_partial_wave_v2")
    requirement = raw.get("matched_feasibility_requirement")
    _require(
        (requirement is not None) == (matched_feasibility is not None),
        "partial wave matched requirement missing",
    )
    if matched_feasibility is not None:
        _same(
            requirement,
            {
                "amendment_sha256": GA_MATCHED_CONFIG_SHA256,
                "predicate_sha256": matched_feasibility.predicate_sha256,
                "context_sha256": matched_feasibility.context_sha256,
                "source_sha256": matched_feasibility.source_sha256,
                "scope": "conditional_legal_ga_partial_native_attempt_not_selected_pool",
            },
            "partial wave external matched requirement differs",
        )
        _require(
            matched_feasibility.context_sha256 == context.objective.context_sha256,
            "partial wave external matched context differs",
        )
    stopped = {
        "stopped_kl_guard",
        "stopped_policy_guard",
        "stopped_deadline",
        "stopped_integrity_or_numerical_failure",
        "stopped_final_integrity",
        "stopped_diagnostic_byte_cap",
    }
    _require(
        raw["status"]
        in stopped
        | {
            _READY,
            "abstained_incomplete_native_branch",
            "abstained_insufficient_feasible_candidates",
            prefix_result.status,
        },
        "partial wave status unsupported",
    )
    _require(
        type(receipt.next_native_ordinal) is int
        and receipt.next_native_ordinal == raw["next_native_ordinal"]
        and type(receipt.next_behavior_versions) is tuple,
        "partial wave next-state wrapper differs",
    )
    _same(
        receipt.next_behavior_versions,
        [(triple, raw["next_behavior_versions"][triple]) for triple in TRIPLES],
        "partial wave next-version wrapper differs",
    )
    _require(
        set(raw["next_behavior_versions"]) == set(TRIPLES)
        and all(type(value) is int for value in raw["next_behavior_versions"].values()),
        "partial wave next-version inventory is not exact",
    )
    if raw["status"] == "stopped_diagnostic_byte_cap":
        _require(
            raw["prefix_sha256"] == prefix_result.output_sha256
            and raw["ranked_positions"] == []
            and receipt.ranked_sequences == (),
            "partial wave byte fallback releases unverified output",
        )
        _same(
            raw["next_behavior_versions"],
            versions,
            "partial byte fallback advances behavior versions",
        )
        remaining = (
            65536 - len(prefix_result.prefix.attempts) if prefix_result.status == "complete" else 0
        )
        _require(
            native_ordinal <= receipt.next_native_ordinal <= native_ordinal + remaining,
            "partial byte fallback resets/exceeds claimed native ordinal scope",
        )
        _work_contract(
            raw["work"], {}, exact=False, wave=True, matched=matched_feasibility is not None
        )
        if expected_new_units is not None:
            _compare_units(units, expected_new_units, versions)
        _require(
            expected_seats is None or expected_seats == (),
            "partial byte fallback cannot supply seats",
        )
        _require(
            source_identities() == sources
            and kernel_input.sha256 == original_input
            and prefix_result.canonical_bytes() == original_prefix
            and canonical_json_bytes(asdict(context)) == original_context
            and expected_behavior_versions == versions
            and tuple(unit_binding(unit, versions[unit.triple]) for unit in units) == bindings,
            "partial byte fallback source/input/old-model identity changed during inspection",
        )
        return {
            "reconstructed": False,
            "status": receipt.status,
            "accepted": False,
            "complete_native_work_reconstructed": False,
            "failure_cause_authenticated": False,
            "external_timing_authenticated": False,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
        }
    _require(
        raw["configuration_sha256"] == CONFIG_SHA256
        and raw["native_ordinal"] == native_ordinal
        and raw["enforce_kl"] is expected_enforce_kl
        and raw["selected_tuning_result_available"] is False,
        "partial wave configuration/initial ordinal/mode/qualification differs",
    )
    for field, expected in (
        ("sources", sources),
        ("context", asdict(context)),
        ("kernel_input", plain(asdict(kernel_input))),
        ("eligible_prefix", asdict(prefix_result)),
        ("external_prefix_pins", plain(pins)),
        ("old_units", bindings),
    ):
        _same(raw[field], expected, "partial wave exact " + field + " binding differs")
    work, working, update_raw, plan = {}, units, None, None
    exact_update, complete_native = True, True
    working_versions = versions.copy()
    if raw["teacher_eligibility"] is not None:
        _require(
            prefix_result.status == "complete",
            "partial wave teacher precedes complete eligible prefix",
        )
        teacher = _wave_teacher(history, teacher_eligibility, context, expected_eligible_query_ids)
        _same(
            raw["teacher_eligibility"],
            asdict(teacher_eligibility),
            "partial wave true target origin/eligibility record differs",
        )
        _same(
            raw["teacher"],
            asdict(teacher),
            "partial wave independently selected targets/weights differ",
        )
        if raw["plan"] is not None:
            plan = _wave_plan(
                kernel_input,
                prefix_result,
                bindings,
                source_sha,
                native_ordinal,
                expected_prefix_deadline,
            )
            _same(
                raw["plan"],
                asdict(plan),
                "partial wave independent coupled parent/checkpoint law differs",
            )
        if raw["update"] is not None:
            _require(
                plan is not None, "partial wave update lacks its independently reconstructed plan"
            )
            wrapped = PartialGuardedUpdate(**raw["update"])
            working, update_raw, work, exact_update = _update_core(
                units, teacher, plan, wrapped, expected_enforce_kl, matched_feasibility
            )
            if update_raw["accepted"]:
                working_versions = {
                    key: value + int(update_raw["updated"]) for key, value in versions.items()
                }
    else:
        _require(
            raw["teacher"] is None and raw["plan"] is None and raw["update"] is None,
            "partial wave downstream state exists without its verified teacher",
        )
    pool = []
    if raw["pool"] or raw["native_batches"]:
        _require(
            update_raw is not None and update_raw["accepted"] and plan is not None,
            "partial wave consumes a policy without accepted guarded qualification",
        )
        pool, complete_native = _wave_pool(
            working, kernel_input, prefix_result.prefix, plan, raw, work
        )
    else:
        _require(
            raw["next_native_ordinal"] == native_ordinal,
            "partial wave resets/advances attempts without a planned native ledger",
        )
    ranked = _wave_scores(raw, pool, history, context, expected_posterior_binding)
    status = raw["status"]
    if status == _READY or status == "abstained_insufficient_feasible_candidates":
        _require(
            update_raw is not None
            and update_raw["accepted"]
            and len(pool) == 256
            and len(raw["predictions"]) == 2
            and complete_native
            and (len(ranked) >= 14) is (status == _READY),
            "partial scored wave status contradicts its completed feasible pool",
        )
        _same(raw["ranked_positions"], ranked, "partial ranked posterior/parent tie order differs")
    elif status == "abstained_incomplete_native_branch":
        _require(
            plan is not None
            and update_raw is not None
            and update_raw["accepted"]
            and 128 <= len(pool) < 256
            and complete_native
            and raw["next_native_ordinal"] == native_ordinal + plan.remaining_attempts
            and not raw["predictions"]
            and raw["posterior_binding"] is None,
            "partial native underfill lacks exhausted exact attempts",
        )
        _require(not raw["ranked_positions"], "partial native underfill releases a ranking")
    elif status in stopped:
        propagated_prefix_stop = (
            status == prefix_result.status
            and prefix_result.status != "complete"
            and raw["teacher_eligibility"] is None
            and not pool
            and not raw["native_batches"]
            and not raw["predictions"]
            and raw["error"] is None
        )
        _require(
            not raw["ranked_positions"]
            and (
                propagated_prefix_stop
                or (
                    type(raw["error"]) is dict
                    and type(raw["error"].get("type")) is str
                    and type(raw["error"].get("message")) is str
                    and len(raw["error"]["message"]) <= 512
                )
            ),
            "partial stopped wave releases a ranking or lacks bounded diagnostic",
        )
        if status in ("stopped_kl_guard", "stopped_policy_guard"):
            _require(
                (
                    expected_enforce_kl
                    if status == "stopped_kl_guard"
                    else matched_feasibility is not None
                )
                and (status == "stopped_policy_guard") == (matched_feasibility is not None)
                and update_raw is not None
                and update_raw["status"] == "all_backtracks_rejected_no_commit",
                "partial KL stop lacks a fully rejected enforced update",
            )
        if status == "stopped_deadline" and not propagated_prefix_stop:
            _require(
                raw["error"]["type"] == "TimeoutError", "partial deadline diagnostic type differs"
            )
    else:
        _require(
            prefix_result.status != "complete"
            and status == prefix_result.status
            and raw["teacher_eligibility"] is None
            and not pool
            and not raw["native_batches"]
            and not raw["predictions"]
            and not raw["ranked_positions"],
            "partial wave special-prefix lifecycle differs",
        )
    if status not in stopped:
        _require(raw["error"] is None, "partial completed wave contains an unresolved failure")
    expected_ranked = tuple(pool[index]["sequence"] for index in ranked) if status == _READY else ()
    _require(
        type(receipt.ranked_sequences) is tuple and receipt.ranked_sequences == expected_ranked,
        "partial wave released ranking differs from independently reconstructed order",
    )
    _same(raw["working_versions"], working_versions, "partial private working versions differ")
    _same(
        raw["next_behavior_versions"],
        working_versions if status == _READY else versions,
        "partial arm failure/non-ready wave commits private versions",
    )
    _same(
        raw["working_models"],
        [(unit.triple, unit.policy_sha256) for unit in working],
        "partial working models differ from the reconstructed private update",
    )
    _require(
        type(receipt.checkpoint_payloads) is tuple
        and sum(len(value) for _, value in receipt.checkpoint_payloads) <= 256 * 1024**2,
        "partial private checkpoint payload shape/byte cap differs",
    )
    if receipt.checkpoint_payloads:
        _require(
            tuple(triple for triple, _ in receipt.checkpoint_payloads) == TRIPLES,
            "partial private checkpoint ordered inventory differs",
        )
        for unit, (_, value) in zip(working, receipt.checkpoint_payloads, strict=True):
            _require(
                type(value) is bytes
                and value == save(_canonical_model_state(unit.model), metadata=None),
                "partial private checkpoint bytes differ from reconstructed working weights",
            )
    else:
        _require(status in stopped, "partial completed wave omits its private checkpoint evidence")
    _same(
        raw["checkpoints"],
        [
            (triple, hashlib.sha256(value).hexdigest())
            for triple, value in receipt.checkpoint_payloads
        ],
        "partial checkpoint receipt identities differ",
    )
    exact = exact_update and complete_native and status not in stopped
    _work_contract(
        raw["work"], work, exact=exact, wave=True, matched=matched_feasibility is not None
    )
    if expected_new_units is not None:
        _compare_units(working if status == _READY else units, expected_new_units, versions)
    seats = None
    if private_collisions is not None:
        _require(
            type(private_collisions) is PrivateGACollisions,
            "partial private collision record type differs",
        )
        private_collisions.__post_init__()
        _require(
            set(private_collisions.charged_sequence_keys)
            == {sequence_key(row.sequence) for row in history.observations},
            "partial private composition drops a charged collision",
        )
        excluded = set(private_collisions.charged_sequence_keys) | set(
            private_collisions.upcoming_reserve_sequence_keys
        )
        selected = tuple(seq for seq in expected_ranked if sequence_key(seq) not in excluded)[:14]
        seats = selected if len(selected) == 14 else ()
    if expected_seats is not None:
        _require(
            seats is not None and type(expected_seats) is tuple and expected_seats == seats,
            "partial private returned seats differ or lack caller collision inventory",
        )
    _require(
        source_identities() == sources
        and kernel_input.sha256 == original_input
        and prefix_result.canonical_bytes() == original_prefix
        and canonical_json_bytes(asdict(context)) == original_context
        and expected_behavior_versions == versions
        and tuple(unit_binding(unit, versions[unit.triple]) for unit in units) == bindings,
        "partial wave source/input/context/old-model identity changed during reconstruction",
    )
    return {
        "reconstructed": True,
        "status": status,
        "accepted": status == _READY,
        "ranked_sequences": expected_ranked,
        "private_seats": seats,
        "complete_native_work_reconstructed": exact,
        "failure_cause_authenticated": False,
        "external_timing_authenticated": False,
        "posterior_provider_authenticated": False,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }

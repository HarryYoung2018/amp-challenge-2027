"""Independent sequencing/reconstruction of recorded shared endpoint updates.

Reuses accepted native gradient, kernel and trace scorers, never the new update
producer or a live posterior/oracle. Operator reports need their own verifier.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _state_contract,
    _validated_transition_kernels,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_proposals import (
    NativeProposalStep,
    NativeProposalTrace,
    replay_native_trace,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    ENDPOINT_CONFIG_SHA256,
    TRIPLES,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    NativeWeightedReplay,
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.diffusion.replay import summarize_kl
from amp_challenge.generators.diffusion.subset_kernel import (
    SubsetCommitDraw,
    complete_subset_commit_kl,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import hash_string


def _trace(raw):
    result = dict(raw)
    result["remasked_positions"] = tuple(result["remasked_positions"])
    result["steps"] = tuple(
        NativeProposalStep(
            row["level"],
            row["before_tokens_sha256"],
            row["after_tokens_sha256"],
            SubsetCommitDraw(
                tuple(row["draw"]["positions"]),
                tuple(row["draw"]["residues"]),
                row["draw"]["log_probability"],
            ),
        )
        for row in result["steps"]
    )
    return NativeProposalTrace(**result)


def _replay(raw):
    result = NativeWeightedReplay(
        tuple(raw["sequences"]),
        raw["context_id"],
        tuple(
            NativeTransitionState(
                np.asarray(row["tokens"], dtype=np.int64), row["length"], row["level"]
            )
            for row in raw["states"]
        ),
        np.asarray(raw["weights"]),
    )
    if result.sha256 != raw["sha256"]:
        raise ValueError("shared endpoint replay content identity differs")
    return result


def _same(left, right, message):
    if json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) != json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    ):
        raise ValueError(message)


def _source_bytes(payload):
    root = Path(__file__).resolve().parents[4]
    expected = ["configs/diffusion/native_shared_endpoint_v1.toml"] + [
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
    if set(payload) != set(expected):
        raise ValueError("shared endpoint source inventory differs")
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
    if actual != payload or actual[expected[0]] != ENDPOINT_CONFIG_SHA256:
        raise ValueError("shared endpoint source/config bytes differ")


def _admission(teacher):
    """Recompute source weights without calling the new producer's admission."""
    indices = [i for i, row in enumerate(teacher.targets) if row.role == teacher.protected_role]
    if len(indices) < 40:
        return {
            "admitted": False,
            "reason": "insufficient_distinct_protected_targets",
            "protected_rows": len(indices),
        }, None
    values = np.asarray([row.log_weight for row in teacher.targets], dtype=np.float64)
    weights = np.exp(values - np.max(values))
    weights = weights / np.sum(weights)

    def summarize(raw):
        normalized = np.asarray(raw, dtype=np.float64)
        normalized = normalized / np.sum(normalized)
        ess = float(1 / np.sum(normalized * normalized))
        maximum = float(np.max(normalized))
        return {
            "rows": len(normalized),
            "ess": ess,
            "ess_fraction": ess / len(normalized),
            "maximum_weight": maximum,
            "passed": ess / len(normalized) >= 0.2 and maximum <= 0.05 + 1e-15,
        }

    reports = {
        "protected_targets": summarize(weights[indices]),
        "all_targets": summarize(weights),
        "anchors": summarize(np.ones(64)),
        "combined": summarize(np.concatenate([np.full(64, 0.5 / 64), 0.5 * weights])),
    }
    if not all(report["passed"] for report in reports.values()):
        raise ValueError("shared endpoint reconstructed source weight guard failed")
    return {"admitted": True, "reason": "source_specific_weight_guards_passed", **reports}, weights


def _record_contract(payload):
    statuses = {
        "insufficient_targets_no_update",
        "accepted_guarded_update",
        "accepted_no_kl_enforcement_caps_recorded",
        "all_backtracks_rejected_ten_students_unchanged",
        "partial_deadline_no_commit",
        "numerical_or_guard_failure_no_commit",
    }
    if (
        payload["artifact"]
        != (
            "native_shared_endpoint_update_v2_matched_feasibility"
            if payload.get("matched_feasibility_requirement") is not None
            else "native_shared_endpoint_update_v1"
        )
        or payload["status"] not in statuses
        or type(payload["accepted"]) is not bool
        or type(payload["enforce_kl"]) is not bool
        or any(
            payload[key] is not False
            for key in ("campaign_eligible", "scientific_evidence_accepted", "production_eligible")
        )
        or type(payload["seed"]) is not int
        or not 0 <= payload["seed"] < 2**63
        or type(payload["checks"]) is not int
        or payload["checks"] < 1
    ):
        raise ValueError("shared endpoint record status/mode/qualification differs")
    status, error = payload["status"], payload["error"]
    is_success = status.startswith("accepted_")
    if payload["accepted"] != is_success or (
        is_success
        and (
            payload["enforce_kl"] != (status == "accepted_guarded_update")
            or error is not None
            or payload["stop_check"] is not None
        )
    ):
        raise ValueError("shared endpoint acceptance/status consistency differs")
    if status == "partial_deadline_no_commit":
        if (
            not isinstance(error, dict)
            or error.get("type") != "TimeoutError"
            or type(payload["stop_check"]) is not int
            or not 1 <= payload["stop_check"] <= payload["checks"]
        ):
            raise ValueError("shared endpoint deadline stop record differs")
    elif status == "numerical_or_guard_failure_no_commit":
        if (
            not isinstance(error, dict)
            or error.get("type")
            not in ("ValueError", "FloatingPointError", "RuntimeError", "TypeError")
            or payload["stop_check"] is not None
        ):
            raise ValueError("shared endpoint numerical failure record differs")
    elif error is not None or payload["stop_check"] is not None:
        raise ValueError("shared endpoint terminal status has a contradictory failure")
    present = payload["operator_source_sha256"] is not None
    if (
        present
        and not all(
            hash_string(payload[key]) for key in ("operator_source_sha256", "operator_plan_sha256")
        )
    ) or (not present and payload["operator_plan_sha256"] is not None):
        raise ValueError("shared endpoint operator source/plan differs")
    expected_scope = (
        "actual_operator_callback_supplied_external_authentication_required"
        if present
        else "FULLMASK_GENERATOR_PATH_only_actual_operator_campaign_compatibility_unresolved"
    )
    if payload["operator_scope"] != expected_scope:
        raise ValueError("shared endpoint operator measure scope differs")


def _operator_contract(raw, payload, unit, candidate):
    if (
        raw["source_sha256"] != payload["operator_source_sha256"]
        or raw["plan_sha256"] != payload["operator_plan_sha256"]
        or not hash_string(raw["receipt_sha256"])
        or raw["old_model_sha256"] != unit.policy_sha256
        or raw["candidate_model_sha256"] != canonical_model_logical_hash(candidate)
        or raw["reference_model_sha256"] != unit.reference_sha256
        or type(raw["path_count"]) is not int
        or not 2 <= raw["path_count"] <= 128
        or raw["scope"] != "DECLARED_CONDITIONAL_OPERATOR_PATH_NOT_GLOBAL_SEARCH"
        or any(
            type(raw[key]) is not float or not math.isfinite(raw[key]) or raw[key] < 0
            for key in ("mean", "monte_carlo_standard_error", "active_transition_p99")
        )
    ):
        raise ValueError("shared endpoint operator report identity/measure differs")


def _reference_paths(unit, candidate, raw, seed, semantic):
    if (
        raw["measure"] != "FULLMASK_GENERATOR_PATH"
        or raw["path_count"] != 8
        or len(raw["traces"]) != 8
    ):
        raise ValueError("shared endpoint reference path measure/count differs")
    key = _json_hash(
        ["shared-endpoint-v1", seed, (semantic, unit.triple, "fullmask-audit-lengths")]
    )
    rng = np.random.Generator(np.random.PCG64DXSM(int(key[:32], 16)))
    parents = tuple(unit.sequences[int(rng.integers(len(unit.sequences)))] for _ in range(8))
    path_seed = int(_json_hash([seed, semantic, unit.triple, "fullmask-audit-paths"])[:16], 16)
    totals, active = [], []
    for index, recorded in enumerate(raw["traces"]):
        trace = _trace(recorded)
        if (trace.parent, trace.seed, trace.ordinal, trace.start_level) != (
            parents[index],
            path_seed,
            index,
            candidate.config.levels,
        ):
            raise ValueError("shared endpoint fullmask checkpoint/length/path assignment differs")
        _, states = replay_native_trace(candidate, trace, authenticate_sampling=True)
        current = _validated_transition_kernels(candidate, states, NATIVE_ENDPOINT_DEFAULTS)
        frozen = _validated_transition_kernels(unit.reference, states, NATIVE_ENDPOINT_DEFAULTS)
        values = [
            complete_subset_commit_kl(left, right)
            for left, right in zip(current, frozen, strict=True)
        ]
        totals.append(math.fsum(values))
        active.extend(
            value
            for state, value in zip(states, values, strict=True)
            if _state_contract(state, candidate.config)[1] > 0
        )
    array = np.asarray(totals)
    _same(raw["path_values"], totals, "shared endpoint path KL sums differ")
    _same(raw["active_transitions"], active, "shared endpoint active transition inventory differs")
    _same(
        raw["active_summary"],
        asdict(summarize_kl(active)),
        "shared endpoint active quantiles differ",
    )
    if raw["mean"] != float(array.mean()) or raw["monte_carlo_standard_error"] != float(
        array.std(ddof=1) / math.sqrt(8)
    ):
        raise ValueError("shared endpoint path mean/MCSE differs")


def verify_shared_endpoint_update(
    units,
    teacher,
    receipt,
    *,
    expected_new_units=None,
    operator_report_verifier=None,
    matched_feasibility=None,
    check=lambda: None,
):
    """Reconstruct observed math; partial receipt does not certify failure cause.

    Same-CPU exact arithmetic is the qualified replay boundary. This reconstructs
    completed gradients and KL paths even for a no-commit partial failure; external
    timing/failure receipts remain necessary for scientific resource claims.
    Complete matched candidates are replayed; incomplete matched path prefixes
    remain retained but are not authenticated by this reader.
    """
    if (
        receipt.campaign_eligible is not False
        or receipt.production_eligible is not False
        or receipt.scientific_evidence_accepted is not False
    ):
        raise ValueError("shared endpoint receipt qualification differs")
    if (
        len(receipt.record_json.encode()) > 128 * 1024**2
        or hashlib.sha256(receipt.record_json.encode()).hexdigest() != receipt.sha256
    ):
        raise ValueError("shared endpoint receipt seal differs")
    payload = json.loads(receipt.record_json)
    _record_contract(payload)
    _source_bytes(payload["source_identities"])
    check()
    requirement = payload.get("matched_feasibility_requirement")
    if (requirement is not None) != (matched_feasibility is not None):
        raise ValueError(
            "shared endpoint needs the externally pinned matched feasibility requirement"
        )
    if matched_feasibility is not None:
        matched_feasibility.wave.check()
        _same(
            requirement,
            {
                "contract": "native_matched_feasibility_v1_20260914",
                "wave_sha256": matched_feasibility.wave.sha256,
                "predicate_sha256": matched_feasibility.predicate_sha256,
                "context_sha256": matched_feasibility.context_sha256,
                "source_sha256": matched_feasibility.source_sha256,
            },
            "shared endpoint external matched requirement differs",
        )
        from amp_challenge.generators.diffusion.native_matched_feasibility_verify import (
            _current_source,
            verify_native_matched_feasibility,
        )

        if (
            matched_feasibility.wave.status != "complete"
            or matched_feasibility.context_sha256 != teacher.objective_context_sha256
            or _current_source() != matched_feasibility.source_sha256
            or getattr(matched_feasibility.predicate, "source_sha256", None)
            != matched_feasibility.predicate_sha256
        ):
            raise ValueError("shared endpoint external matched source/context differs")
    elif payload.get("matched_feasibility_plan_sha256") is not None:
        raise ValueError("shared endpoint matched plan lacks public requirement")
    if tuple(unit.triple for unit in units) != TRIPLES:
        raise ValueError("shared endpoint verifier needs exact ten-student inventory")
    for unit in units:
        unit.check()
    teacher.__post_init__()
    _same(payload["teacher"], asdict(teacher), "shared endpoint teacher provenance differs")
    if (
        payload["teacher_sha256"] != teacher.sha256
        or payload["teacher_semantic_sha256"] != teacher.semantic_sha256
        or payload["config_sha256"] != ENDPOINT_CONFIG_SHA256
    ):
        raise ValueError("shared endpoint teacher semantic/config identity differs")
    _same(
        payload["old_models"],
        [(unit.triple, unit.policy_sha256, unit.reference_sha256) for unit in units],
        "shared endpoint original models differ",
    )
    training_ids = set().union(*(unit.initialization.training_sequence_ids for unit in units))
    if training_ids.intersection(row.sequence_id for row in teacher.targets):
        raise ValueError("shared endpoint reconstructed target overlaps generator training")
    admission, weights = _admission(teacher)
    _same(
        payload["admission"], admission, "shared endpoint source concentration diagnostics differ"
    )
    if (
        len(payload["training"]) > 10
        or len(payload["candidates"]) > 9
        or (not admission["admitted"] and (payload["training"] or payload["candidates"]))
    ):
        raise ValueError("shared endpoint training/candidate budget differs")
    if payload["status"] == "insufficient_targets_no_update" and admission["admitted"]:
        raise ValueError("shared endpoint no-update status contradicts admitted source")
    proposals, probes = [], []
    for ordinal, row in enumerate(payload["training"]):
        unit = units[ordinal]
        anchors = tuple(
            sorted(
                unit.sequences,
                key=lambda seq: _json_hash([payload["seed"], unit.triple, "endpoint-anchor", seq]),
            )[:64]
        )
        if (
            row["triple"] != unit.triple
            or tuple(row["anchors"]) != anchors
            or len(row["steps"]) > 4
        ):
            raise ValueError("shared endpoint anchor/step inventory differs")
        sequences = anchors + tuple(target.sequence for target in teacher.targets)
        mixed_weights = np.concatenate([np.full(64, 0.5 / 64), 0.5 * weights])
        training_seed = int(
            _json_hash(
                [payload["seed"], teacher.semantic_sha256, unit.triple, "endpoint-training"]
            )[:16],
            16,
        )
        work = unit.model
        for step, observed in enumerate(row["steps"]):
            replay = _replay(observed["replay"])
            expected = build_weighted_replay(
                work,
                sequences,
                mixed_weights,
                context_id=teacher.objective_context_sha256,
                seed=training_seed,
                ordinal=step,
            )
            if replay.sha256 != expected.sha256:
                raise ValueError("shared endpoint fresh corruption/weights differ")
            direction = propose_weighted_direction(work, replay)
            gradient = hashlib.sha256()
            for name, value in direction.gradients:
                gradient.update(name.encode() + b"\0" + value.numpy().tobytes())
            if (
                observed["objective_before"],
                observed["gradient_norm_before_clip"],
                observed["gradient_sha256"],
            ) != (
                direction.objective_before,
                direction.gradient_norm_before_clip,
                gradient.hexdigest(),
            ):
                raise ValueError("shared endpoint actual gradient/objective differs")
            work = endpoint_candidate(work, direction)
            if observed["model_sha256"] != canonical_model_logical_hash(work):
                raise ValueError("shared endpoint neural step reconstruction differs")
        if "probes" in row:
            if len(row["steps"]) != 4 or row[
                "proposed_model_sha256"
            ] != canonical_model_logical_hash(work):
                raise ValueError("shared endpoint four-step proposal differs")
            probe = _replay(row["probes"])
            expected = build_weighted_replay(
                unit.model,
                sequences,
                mixed_weights,
                context_id=teacher.objective_context_sha256,
                seed=training_seed,
                ordinal=4,
                active_probes=True,
            )
            if probe.sha256 != expected.sha256:
                raise ValueError("shared endpoint matched local probe differs")
            proposals.append(work)
            probes.append(probe)
    new_models = [(unit.triple, unit.policy_sha256) for unit in units]
    accepted_index = None
    matched_verified = 0
    matched_partial = False
    for backtracks, attempt in enumerate(payload["candidates"]):
        if (
            accepted_index is not None
            or attempt["backtracks"] != backtracks
            or len(proposals) != 10
            or len(attempt["students"]) > 10
        ):
            raise ValueError("shared endpoint candidate/backtracking sequence differs")
        full_rows = []
        candidate_models = []
        for index, row in enumerate(attempt["students"]):
            unit = units[index]
            candidate = copy.deepcopy(proposals[index] if backtracks == 0 else unit.model).eval()
            if backtracks:
                proposed_parameters = dict(proposals[index].named_parameters())
                with torch.no_grad():
                    for name, parameter in candidate.named_parameters():
                        parameter.add_(
                            proposed_parameters[name] - parameter, alpha=2.0**-backtracks
                        )
            candidate_models.append(candidate)
            if row["triple"] != unit.triple or row[
                "candidate_sha256"
            ] != canonical_model_logical_hash(candidate):
                raise ValueError("shared endpoint aggregate displacement differs")
            _same(
                row["local"],
                asdict(
                    weighted_anchor_diagnostics(
                        unit.model,
                        candidate,
                        unit.reference,
                        probes[index],
                        NATIVE_ENDPOINT_DEFAULTS,
                    )
                ),
                "shared endpoint local conditional-KL reconstruction differs",
            )
            if row["fullmask_reference"] is not None:
                _reference_paths(
                    unit,
                    candidate,
                    row["fullmask_reference"],
                    payload["seed"],
                    teacher.semantic_sha256,
                )
                full_rows.append(row)
            if row["operator"] is not None:
                _operator_contract(row["operator"], payload, unit, candidate)
                if operator_report_verifier is None:
                    raise ValueError(
                        "shared endpoint actual operator report needs independent verifier"
                    )
                operator_report_verifier(
                    unit.model, candidate, unit.reference, row["operator"], unit.triple, backtracks
                )
        if attempt["summary"] is not None:
            if len(full_rows) != 10:
                raise ValueError("shared endpoint summary lacks complete ten-student paths")
            if any(
                (row["operator"] is not None) != (payload["operator_source_sha256"] is not None)
                for row in full_rows
            ):
                raise ValueError("shared endpoint candidate omitted required operator guard")
            path_mean = math.fsum(row["fullmask_reference"]["mean"] for row in full_rows) / 10
            mcse = (
                math.sqrt(
                    math.fsum(
                        row["fullmask_reference"]["monte_carlo_standard_error"] ** 2
                        for row in full_rows
                    )
                )
                / 10
            )
            active, frequencies = [], []
            for row in full_rows:
                values = row["fullmask_reference"]["active_transitions"]
                active.extend(values)
                frequencies.extend([1 / (10 * len(values))] * len(values))
            summary = asdict(summarize_kl(active, weights=frequencies))
            passed = (
                all(
                    row["local"]["local_old_candidate"]["mean"] <= 0.01
                    and row["local"]["local_old_candidate"]["p99"] <= 0.02
                    and row["fullmask_reference"]["mean"] <= 0.08
                    and row["fullmask_reference"]["active_summary"]["p99"] <= 0.02
                    and (
                        row["operator"] is None
                        or (
                            row["operator"]["mean"] <= 0.08
                            and row["operator"]["active_transition_p99"] <= 0.02
                        )
                    )
                    for row in full_rows
                )
                and path_mean <= 0.08
                and summary["p99"] <= 0.02
            )
            _same(
                attempt["summary"],
                {
                    "reference_measure": "FULLMASK_GENERATOR_PATH",
                    "equal_mixture_path_mean": path_mean,
                    "stratified_mixture_mcse": mcse,
                    "equal_component_active_transition_summary": summary,
                    "per_student_and_mixture_kl_passed": passed,
                },
                "shared endpoint mixture guard differs",
            )
            feasible_passed = matched_feasibility is None
            matched_record = attempt.get("matched_feasibility")
            if matched_feasibility is None and matched_record is not None:
                raise ValueError("shared endpoint unexpected matched report")
            if matched_feasibility is not None and matched_record is not None:
                if matched_record["candidate_index"] != backtracks:
                    raise ValueError("shared endpoint matched candidate order differs")
                if matched_record["status"] == "complete":
                    verified = verify_native_matched_feasibility(
                        matched_record,
                        units=units,
                        proposals=proposals,
                        wave=matched_feasibility.wave,
                        predicate=matched_feasibility.predicate,
                        predicate_sha256=matched_feasibility.predicate_sha256,
                        context_sha256=matched_feasibility.context_sha256,
                        seed=payload["seed"],
                        expected_plan_sha256=payload["matched_feasibility_plan_sha256"],
                        expected_source_sha256=matched_feasibility.source_sha256,
                        check=check,
                    )
                    matched_verified += 1
                    feasible_passed = verified["passed"] is True
                elif (
                    matched_record["status"] != "failed"
                    or payload["status"]
                    not in ("partial_deadline_no_commit", "numerical_or_guard_failure_no_commit")
                    or backtracks != len(payload["candidates"]) - 1
                ):
                    raise ValueError("shared endpoint incomplete matched report cannot advance")
            if (
                matched_feasibility is not None
                and (matched_record is None or matched_record["status"] != "complete")
                and (
                    payload["status"]
                    not in ("partial_deadline_no_commit", "numerical_or_guard_failure_no_commit")
                    or backtracks != len(payload["candidates"]) - 1
                )
            ):
                raise ValueError("shared endpoint missing completed matched report")
            if matched_record is not None and matched_record["status"] != "complete":
                matched_partial = True
            if (not payload["enforce_kl"] or passed) and feasible_passed:
                accepted_index = backtracks
                new_models = [
                    (unit.triple, canonical_model_logical_hash(model))
                    for unit, model in zip(units, candidate_models, strict=True)
                ]
    accepted = payload["accepted"]
    if not accepted:
        new_models = [(unit.triple, unit.policy_sha256) for unit in units]
    if accepted and (accepted_index is None or payload["backtracks"] != accepted_index):
        raise ValueError("shared endpoint acceptance lacks passing complete candidate")
    if not accepted and payload["backtracks"] is not None:
        raise ValueError("shared endpoint no-commit result declares chosen displacement")
    if payload["status"] == "all_backtracks_rejected_ten_students_unchanged" and (
        accepted_index is not None or len(payload["candidates"]) != 9
    ):
        raise ValueError("shared endpoint rejection lacks all nine failing candidates")
    _same(payload["new_models"], new_models, "shared endpoint atomic model identities differ")
    if expected_new_units is not None:
        _same(
            new_models,
            [
                (unit.triple, canonical_model_logical_hash(unit.model))
                for unit in expected_new_units
            ],
            "shared endpoint returned models differ",
        )
    if (
        receipt.status,
        receipt.accepted,
        receipt.backtracks,
        receipt.kl_enforced,
        receipt.operator_guard_present,
    ) != (
        payload["status"],
        payload["accepted"],
        payload["backtracks"],
        payload["enforce_kl"],
        payload["operator_source_sha256"] is not None,
    ):
        raise ValueError("shared endpoint receipt summary differs")
    check()
    return {
        "reconstructed": True,
        "matched_feasibility_enforced": matched_feasibility is not None,
        "matched_complete_candidates_verified": matched_verified,
        "matched_partial_paths_unverified": matched_partial,
        "accepted": accepted,
        "status": receipt.status,
        "reference_measure": "FULLMASK_GENERATOR_PATH",
        "scientific_evidence_accepted": False,
    }

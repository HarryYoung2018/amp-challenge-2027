"""Independent complete TR2 atomic proposal/gate/model/version/work readback.

Never calls the v4 producer. Failed partial records and external timing are not
authenticated by this complete-receipt reader.
"""

import hashlib
import json
import math
from dataclasses import asdict

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import NATIVE_ENDPOINT_DEFAULTS, _json_hash
from amp_challenge.generators.diffusion.native_ga_partial_verify import (
    _fullmask_cost,
    _sampling_cost,
)
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_shared_endpoint import (
    OperatorPathReport,
    interpolate_policy,
)
from amp_challenge.generators.diffusion.native_shared_endpoint_verify import (
    _reference_paths,
    _replay,
    _same,
)
from amp_challenge.generators.diffusion.native_tr2_matched_feasibility_verify import (
    verify_tr2_matched_feasibility,
)
from amp_challenge.generators.diffusion.native_tr2_operator_kl import realized_node_plan
from amp_challenge.generators.diffusion.native_tr2_operator_kl_verify import (
    verify_tr2_operator_record,
)
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v4_records import (
    CONTRACT,
    MAXIMUM_BYTES,
    GuardedReplayAdvanceV4,
    TR2MatchedRequirement,
    source_v4,
)
from amp_challenge.generators.diffusion.native_tr2d2_replay_v2_records import (
    ReplayAdvanceV2,
    source_bytes,
)
from amp_challenge.generators.diffusion.native_tr2d2_replay_v2_verify import (
    _verify_replacement_lineage,
    verify_replay_advance_v2,
)
from amp_challenge.generators.diffusion.native_weighted_training import weighted_anchor_diagnostics

KEYS = ("row_forwards", "forward_calls", "grad_enabled_forwards", "backward_calls")


def _require(condition, message):
    if not condition:
        raise ValueError("independent TR2 v4 " + message)


def _add(work, phase, rows, calls, backwards=0):
    if not rows and not calls and not backwards:
        return
    target = work.setdefault(phase, dict.fromkeys(KEYS, 0))
    for key, value in zip(KEYS, (rows, calls, backwards, backwards), strict=True):
        target[key] += value


def _document(work):
    return dict(
        phases=work,
        total={k: sum(p[k] for p in work.values()) for k in KEYS},
        shared_inner_subphase_split_measured=False,
        external_feature_model_work_counted_here=False,
    )


def _kl_passed(row):
    return (
        all(
            r["local_old_candidate"]["mean"] <= 0.01 and r["local_old_candidate"]["p99"] <= 0.02
            for r in row["local"]
        )
        and row["fullmask_reference"]["mean"] <= 0.08
        and row["fullmask_reference"]["active_summary"]["p99"] <= 0.02
        and row["operator"]["mean"] <= 0.08
        and row["operator"]["active_transition_p99"] <= 0.02
    )


def verify_guarded_replay_v4(
    tree,
    generation,
    history,
    receipt,
    *,
    expected_versions,
    expected_new_units,
    expected_new_versions,
    expected_source_sha256,
    expected_previous_head_sha256,
    expected_posterior_sha256,
    expected_deadline,
    matched_feasibility,
    check=lambda: None,
):
    check()
    _require(
        type(receipt) is GuardedReplayAdvanceV4
        and type(matched_feasibility) is TR2MatchedRequirement,
        "needs exact receipt and external requirement",
    )
    encoded = receipt.record_json.encode()
    _require(
        len(encoded) <= MAXIMUM_BYTES and hashlib.sha256(encoded).hexdigest() == receipt.sha256,
        "receipt byte seal differs",
    )
    raw = json.loads(receipt.record_json)
    source = source_v4()
    requirement = matched_feasibility.document()
    _require(source == expected_source_sha256 == raw["source_sha256"], "source differs")
    _same(raw["contract"], CONTRACT, "independent TR2 v4 contract differs")
    _require(raw["contract_sha256"] == _json_hash(CONTRACT), "contract identity differs")
    _same(
        raw["matched_feasibility_requirement"],
        requirement,
        "independent TR2 v4 external requirement differs",
    )
    _require(
        requirement["context_sha256"]
        == history.objective_context_sha256
        == tree.context.context_sha256
        and raw["history_sha256"] == history.sha256
        and raw["round_index"] == history.round_index
        and raw["kl_enforced"] is True
        and raw["feasibility_change_enforced"] is True
        and raw["campaign_eligible"] is False
        and raw["scientific_evidence_accepted"] is False
        and type(raw["original_deadline"]) in (float, int)
        and math.isfinite(raw["original_deadline"])
        and raw["original_deadline"] == expected_deadline,
        "context, mode, deadline or authority differs",
    )
    _require(
        tuple(u.triple for u in tree._units) == TRIPLES
        and tuple(u.triple for u in expected_new_units) == TRIPLES,
        "checkpoint inventory differs",
    )
    old_ids = tuple(canonical_model_logical_hash(u.model) for u in tree._units)
    proposal_raw = json.dumps(
        raw["proposal"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    private_versions = {r["triple"]: r["new_version"] for r in raw["proposal"]["students"]}
    proposal_readback = verify_replay_advance_v2(
        tree,
        generation,
        history,
        ReplayAdvanceV2(proposal_raw, hashlib.sha256(proposal_raw.encode()).hexdigest()),
        expected_versions=expected_versions,
        expected_new_units=None,
        expected_new_versions=private_versions,
        expected_source_sha256=_json_hash(source_bytes()),
        expected_previous_head_sha256=expected_previous_head_sha256,
        expected_posterior_sha256=expected_posterior_sha256,
        reconstruct_private_units=True,
    )
    proposals = proposal_readback.pop("reconstructed_private_units")
    check()
    changed = [
        i
        for i, r in enumerate(raw["proposal"]["students"])
        if r["status"] == "four_actual_weighted_steps"
    ]
    known = {}
    for row in raw["proposal"]["students"]:
        for step in row["steps"]:
            n, m = len(step["replay"]["sequences"]), len(step["probes"]["sequences"])
            _add(
                known,
                "shared_private_proposal",
                n + 3 * m,
                math.ceil(n / 16) + 3 * math.ceil(m / 16),
                math.ceil(n / 16),
            )
    selected, selected_models = None, tuple(u.model for u in tree._units)
    reconstructed = []
    if changed:
        _require(
            generation is not None
            and 1 <= len(raw["candidates"]) <= 9
            and raw["unchanged_feasibility"] is None,
            "trained candidate inventory differs",
        )
        _same(
            raw["operator_plan"],
            realized_node_plan(tree._units, generation.collection),
            "independent TR2 v4 operator plan differs",
        )
        for b, candidate_row in enumerate(raw["candidates"]):
            check()
            _require(
                selected is None
                and type(candidate_row["backtracks"]) is int
                and candidate_row["backtracks"] == b
                and len(candidate_row["students"]) == len(changed),
                "candidate order or inventory differs",
            )
            candidates = tuple(
                interpolate_policy(u.model, p.model, b)
                for u, p in zip(tree._units, proposals, strict=True)
            )
            passed = []
            for i, row in zip(changed, candidate_row["students"], strict=True):
                unit, candidate = tree._units[i], candidates[i]
                _require(
                    row["triple"] == unit.triple
                    and row["candidate_sha256"] == canonical_model_logical_hash(candidate)
                    and len(row["local"]) == 4,
                    "local candidate binding differs",
                )
                for step, local in zip(
                    raw["proposal"]["students"][i]["steps"], row["local"], strict=True
                ):
                    probes = _replay(step["probes"])
                    actual = asdict(
                        weighted_anchor_diagnostics(
                            unit.model, candidate, unit.reference, probes, NATIVE_ENDPOINT_DEFAULTS
                        )
                    )
                    _same(local, actual, "independent TR2 v4 local KL differs")
                    n = len(probes.states)
                    _add(known, "aggregate_local", 3 * n, 3 * math.ceil(n / 16))
                    check()
                _reference_paths(
                    unit, candidate, row["fullmask_reference"], history.seed, history.sha256
                )
                _add(known, "aggregate_reference", *_fullmask_cost(row["fullmask_reference"]))
                check()
                verify_tr2_operator_record(
                    tree._units,
                    generation.collection,
                    candidate,
                    row["operator_record"],
                    OperatorPathReport(**row["operator"]),
                )
                n, m = _sampling_cost(row["operator_record"]["paths"])
                _add(known, "aggregate_operator", 2 * n, 2 * m)
                passed.append(_kl_passed(row))
                _require(
                    type(row["passed"]) is bool and row["passed"] == passed[-1],
                    "student KL acceptance differs",
                )
            _require(
                type(candidate_row["kl_passed"]) is bool
                and candidate_row["kl_passed"] == all(passed),
                "tuple KL acceptance differs",
            )
            matched = candidate_row["matched_feasibility"]
            _require(
                type(matched) is dict and matched["candidate_index"] == b,
                "matched report missing or candidate differs",
            )
            _same(
                matched["work_before"],
                _document(known),
                "independent TR2 v4 matched opening work differs",
            )
            result = verify_tr2_matched_feasibility(
                matched,
                units=tree._units,
                proposals=tuple(p.model for p in proposals),
                generation=generation.collection,
                predicate=matched_feasibility.predicate,
                predicate_sha256=matched_feasibility.predicate_sha256,
                context_sha256=matched_feasibility.context_sha256,
                expected_source_sha256=matched_feasibility.source_sha256,
                expected_plan_sha256=raw["matched_feasibility_plan_sha256"],
                check=check,
            )
            delta = result["sampling_work_delta"]
            _add(
                known,
                "tr2_matched_feasibility_sampling",
                delta["row_forwards"],
                delta["forward_calls"],
            )
            _same(
                matched["work_after"],
                _document(known),
                "independent TR2 v4 matched closing work differs",
            )
            combined = all(passed) and result["passed"]
            _require(
                type(candidate_row["passed"]) is bool and candidate_row["passed"] == combined,
                "combined feasibility acceptance differs",
            )
            reconstructed.append(result)
            if combined:
                selected, selected_models = b, candidates
        _require(selected is not None or len(raw["candidates"]) == 9, "incomplete rejected family")
        status = (
            "accepted_guarded_update"
            if selected is not None
            else "all_backtracks_rejected_unchanged_tuple"
        )
    else:
        _require(
            not raw["candidates"]
            and raw["matched_feasibility_plan_sha256"] is None
            and "operator_plan" not in raw,
            "unchanged tuple acquired sampled evidence",
        )
        _same(
            raw["unchanged_feasibility"],
            dict(status="analytic_identical_policy_no_training", drop=0.0, sampled_paths=0),
            "independent TR2 v4 unchanged identity proof differs",
        )
        _require(
            tuple(canonical_model_logical_hash(u.model) for u in proposals) == old_ids,
            "untrained proposal differs",
        )
        status = (
            "unchanged_common_initial"
            if history.round_index == 1
            else "unchanged_terminal_round29"
            if history.round_index == 29
            else "unchanged_replay_support_rejected"
        )
    _require(
        raw["chosen_backtracks"] == selected and raw["status"] == status,
        "terminal acceptance status differs",
    )
    versions = dict(expected_versions)
    if selected is not None:
        for i in changed:
            versions[tree._units[i].triple] += 1
    _same(raw["new_versions"], versions, "independent TR2 v4 committed versions differ")
    _same(expected_new_versions, versions, "independent TR2 v4 external versions differ")
    identities = {
        u.triple: canonical_model_logical_hash(m)
        for u, m in zip(tree._units, selected_models, strict=True)
    }
    _same(
        raw["new_policy_identities"],
        tuple(identities.items()),
        "independent TR2 v4 committed model inventory differs",
    )
    for old, actual in zip(tree._units, expected_new_units, strict=True):
        _verify_replacement_lineage(old, actual)
        actual.check()
        _require(
            actual.policy_sha256
            == identities[old.triple]
            == canonical_model_logical_hash(actual.model),
            "external committed model differs",
        )
    _same(raw["native_work"], _document(known), "independent TR2 v4 complete native work differs")
    _require(
        all(
            raw["native_work"]["total"][k] <= cap
            for k, cap in (
                ("row_forwards", 540160),
                ("forward_calls", 42400),
                ("backward_calls", 160),
                ("grad_enabled_forwards", 160),
            )
        ),
        "work ceiling differs",
    )
    check()
    _require(
        source_v4() == source == expected_source_sha256
        and matched_feasibility.document() == requirement
        and old_ids == tuple(canonical_model_logical_hash(u.model) for u in tree._units),
        "frozen source, requirement or old models changed",
    )
    return dict(
        reconstructed=True,
        proposal=proposal_readback,
        matched_candidates=reconstructed,
        accepted=selected is not None,
        chosen_backtracks=selected,
        status=status,
        native_work=_document(known),
        complete_native_work_reconstructed=True,
        external_timing_authenticated=False,
        scientific_evidence_accepted=False,
    )

"""Atomic TR2 replay: existing KL gates plus measured matched feasibility."""

import copy
import hashlib
import json
from dataclasses import asdict

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import NATIVE_ENDPOINT_DEFAULTS, _json_hash
from amp_challenge.generators.diffusion.native_ga_partial_work import NativeWorkCounter
from amp_challenge.generators.diffusion.native_initialization import TRIPLES
from amp_challenge.generators.diffusion.native_shared_endpoint import (
    fullmask_reference_paths,
    interpolate_policy,
)
from amp_challenge.generators.diffusion.native_tr2_matched_feasibility import TR2MatchedFeasibility
from amp_challenge.generators.diffusion.native_tr2_operator_kl import TR2OperatorGuard
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v3 import (
    NativeTR2D2GuardedV3,
    _passed,
    _probes,
    _stage,
)
from amp_challenge.generators.diffusion.native_tr2d2_guarded_v4_records import (
    CONTRACT,
    MAXIMUM_BYTES,
    GuardedReplayAdvanceV4,
    TR2MatchedRequirement,
    source_v4,
)
from amp_challenge.generators.diffusion.native_weighted_training import weighted_anchor_diagnostics


class NativeTR2D2GuardedV4(NativeTR2D2GuardedV3):
    def __init__(self, *args, matched_feasibility, feature_batching_sha256=None, **kwargs):
        from amp_challenge.generators.diffusion.native_tr2_feature_batching_records import (
            batching_source,
            check_batching_configuration,
        )

        if feature_batching_sha256 is not None:
            check_batching_configuration(feature_batching_sha256)
        if type(matched_feasibility) is not TR2MatchedRequirement:
            raise TypeError("TR2 v4 requires the external matched-feasibility requirement")
        super().__init__(*args, **kwargs)
        if tuple(u.triple for u in self.tree._units) != TRIPLES:
            raise ValueError("TR2 v4 requires all ten ordered checkpoints")
        self.matched_feasibility = matched_feasibility
        self.requirement = matched_feasibility.document()
        if self.requirement["context_sha256"] != self.tree.context.context_sha256:
            raise ValueError("TR2 v4 public context differs")
        self.source_sha256 = source_v4()
        self.contract_sha256 = _json_hash(CONTRACT)
        self.feature_batching_sha256 = feature_batching_sha256
        self.feature_batching_source_sha256 = (
            None if feature_batching_sha256 is None else batching_source()
        )
        self._feature_batching_binding = (
            self.feature_batching_sha256,
            self.feature_batching_source_sha256,
        )
        self._check()

    def _check(self):
        if self.failure is not None:
            raise ValueError("TR2 v4 terminal failure cannot restart")
        if (
            source_v4() != self.source_sha256
            or _json_hash(CONTRACT) != self.contract_sha256
            or self.matched_feasibility.document() != self.requirement
        ):
            raise ValueError("TR2 v4 source/contract/public requirement changed")
        if (
            self.feature_batching_sha256,
            self.feature_batching_source_sha256,
        ) != self._feature_batching_binding:
            raise ValueError("TR2 v4 feature-batching selection changed")
        if self.feature_batching_sha256 is not None:
            from amp_challenge.generators.diffusion.native_tr2_feature_batching_records import (
                batching_source,
                check_batching_configuration,
            )

            check_batching_configuration(self.feature_batching_sha256)
            if batching_source() != self.feature_batching_source_sha256:
                raise ValueError("TR2 v4 feature-batching source changed")
        self._replay._check()

    def collect(self, evaluator, *, expected_posterior_sha256):
        self._check()
        return self._replay.collect(
            evaluator,
            expected_posterior_sha256=expected_posterior_sha256,
            feature_batching_sha256=self.feature_batching_sha256,
        )

    def advance(self, history, *, expected_previous_head_sha256, outer_deadline):
        self._check()
        admission = self._replay._history_admission(history, expected_previous_head_sha256)
        if admission == "already_advanced":
            return self.receipt
        staged = _stage(self._replay)
        payload = dict(
            contract=CONTRACT,
            contract_sha256=self.contract_sha256,
            source_sha256=self.source_sha256,
            history_sha256=history.sha256,
            round_index=history.round_index,
            proposal=None,
            candidates=[],
            chosen_backtracks=None,
            status="started",
            kl_enforced=True,
            feasibility_change_enforced=True,
            matched_feasibility_requirement=self.requirement,
            matched_feasibility_plan_sha256=None,
            unchanged_feasibility=None,
            native_work=None,
            campaign_eligible=False,
            scientific_evidence_accepted=False,
        )

        def check():
            self._check()
            staged._time(history.round_index, "guarded_v4_acceptance")

        def deadline_poll():
            # Sampling keeps full integrity checks at each model boundary.
            staged._time(history.round_index, "guarded_v4_acceptance")

        counter = NativeWorkCounter(self.tree._units)
        try:
            with counter:
                try:
                    with counter.at("shared_private_proposal"):
                        proposal = staged.advance(
                            history,
                            expected_previous_head_sha256=expected_previous_head_sha256,
                            outer_deadline=outer_deadline,
                        )
                    if proposal is None:
                        self._replay._wave_deadlines = dict(staged._wave_deadlines)
                        return None
                    payload["proposal"] = json.loads(proposal.record_json)
                    check()
                    changed = [
                        i
                        for i, row in enumerate(payload["proposal"]["students"])
                        if row["status"] == "four_actual_weighted_steps"
                    ]
                    replacements, accepted = self.tree._units, False
                    if changed:
                        operator = TR2OperatorGuard(
                            self.tree._units,
                            self.generation.collection,
                            deadline=staged._wave_deadlines[history.round_index],
                            clock=staged.clock,
                        )
                        payload["operator_plan"] = operator.plan
                        requirement = self.matched_feasibility
                        feasibility = TR2MatchedFeasibility(
                            self.tree._units,
                            tuple(u.model for u in staged.tree._units),
                            self.generation.collection,
                            predicate=requirement.predicate,
                            predicate_sha256=requirement.predicate_sha256,
                            context_sha256=requirement.context_sha256,
                            expected_source_sha256=requirement.source_sha256,
                            counter=counter,
                            check=check,
                            deadline_poll=deadline_poll,
                        )
                        payload["matched_feasibility_plan_sha256"] = feasibility.plan_sha256
                        probes = {
                            i: tuple(
                                _probes(step["probes"])
                                for step in payload["proposal"]["students"][i]["steps"]
                            )
                            for i in changed
                        }
                        for b in range(9):
                            candidate_row = dict(
                                backtracks=b,
                                students=[],
                                kl_passed=False,
                                matched_feasibility=None,
                                passed=False,
                            )
                            payload["candidates"].append(candidate_row)
                            candidates = list(self.tree._units)
                            for i in changed:
                                check()
                                unit = self.tree._units[i]
                                candidate = interpolate_policy(
                                    unit.model, staged.tree._units[i].model, b
                                )
                                row = dict(
                                    triple=unit.triple,
                                    candidate_sha256=canonical_model_logical_hash(candidate),
                                    local=[],
                                )
                                candidate_row["students"].append(row)
                                for probe in probes[i]:
                                    with counter.at("aggregate_local"):
                                        row["local"].append(
                                            asdict(
                                                weighted_anchor_diagnostics(
                                                    unit.model,
                                                    candidate,
                                                    unit.reference,
                                                    probe,
                                                    NATIVE_ENDPOINT_DEFAULTS,
                                                )
                                            )
                                        )
                                    check()
                                with counter.at("aggregate_reference"):
                                    row["fullmask_reference"] = fullmask_reference_paths(
                                        unit,
                                        candidate,
                                        seed=history.seed,
                                        semantic_sha256=history.sha256,
                                        check=check,
                                    )
                                with counter.at("aggregate_operator"):
                                    report = operator.evaluate(
                                        unit.model,
                                        candidate,
                                        unit.reference,
                                        triple=unit.triple,
                                        candidate_index=b,
                                    )
                                row["operator"], row["operator_record"] = (
                                    asdict(report),
                                    operator.records[-1],
                                )
                                check()
                                row["passed"] = _passed(row)
                                replacement = copy.copy(unit)
                                replacement.model, replacement.policy_sha256 = (
                                    candidate,
                                    row["candidate_sha256"],
                                )
                                candidates[i] = replacement
                            candidate_row["kl_passed"] = all(
                                row["passed"] for row in candidate_row["students"]
                            )
                            try:
                                matched = feasibility.evaluate(candidate_index=b)
                            finally:
                                if len(feasibility.records) > b:
                                    candidate_row["matched_feasibility"] = feasibility.records[b]
                            if tuple(
                                r["candidate_model_sha256"] for r in matched["students"]
                            ) != tuple(canonical_model_logical_hash(u.model) for u in candidates):
                                raise ValueError(
                                    "TR2 v4 feasibility measured a different candidate tuple"
                                )
                            candidate_row["passed"] = (
                                candidate_row["kl_passed"] and matched["summary"]["passed"] is True
                            )
                            check()
                            if candidate_row["passed"]:
                                accepted, replacements = True, tuple(candidates)
                                payload["chosen_backtracks"] = b
                                break
                        payload["status"] = (
                            "accepted_guarded_update"
                            if accepted
                            else "all_backtracks_rejected_unchanged_tuple"
                        )
                        if not accepted:
                            staged.versions = dict(self._replay.versions)
                    else:
                        if tuple(
                            canonical_model_logical_hash(u.model) for u in staged.tree._units
                        ) != tuple(u.policy_sha256 for u in self.tree._units):
                            raise ValueError("TR2 v4 untrained tuple differs")
                        payload["unchanged_feasibility"] = dict(
                            status="analytic_identical_policy_no_training",
                            drop=0.0,
                            sampled_paths=0,
                        )
                        payload["status"] = (
                            "unchanged_common_initial"
                            if history.round_index == 1
                            else "unchanged_terminal_round29"
                            if history.round_index == 29
                            else "unchanged_replay_support_rejected"
                        )
                    staged.tree._units = replacements
                    payload["new_versions"] = dict(staged.versions)
                    payload["new_policy_identities"] = staged.policy_identities
                    payload["original_deadline"] = staged._wave_deadlines[history.round_index]
                    payload["native_work"] = counter.document()
                    totals = payload["native_work"]["total"]
                    if any(
                        totals[k] > cap
                        for k, cap in (
                            ("row_forwards", 540160),
                            ("forward_calls", 42400),
                            ("backward_calls", 160),
                            ("grad_enabled_forwards", 160),
                        )
                    ):
                        raise ValueError("TR2 v4 complete update work ceiling exceeded")
                    if any(
                        v["backward_calls"] or v["grad_enabled_forwards"]
                        for k, v in payload["native_work"]["phases"].items()
                        if k != "shared_private_proposal"
                    ):
                        raise ValueError("TR2 v4 gradients outside the private proposal")
                    check()
                    raw = json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    if len(raw.encode()) > MAXIMUM_BYTES:
                        raise ValueError("TR2 v4 receipt byte ceiling exceeded")
                    receipt = GuardedReplayAdvanceV4(raw, hashlib.sha256(raw.encode()).hexdigest())
                    check()
                except BaseException as error:
                    payload["native_work"] = counter.document()
                    self.failure = dict(
                        phase="guarded_v4_advance",
                        error=f"{type(error).__name__}: {error}"[:512],
                        prefix=payload,
                        staged_failure=None if staged.failure is None else asdict(staged.failure),
                    )
                    raise
            # Cleanup remains paid work before publication. Retain failures from
            # the counter's exit as well as expiry after its successful exit.
            check()
        except BaseException as error:
            if self.failure is None:
                payload["native_work"] = counter.document()
                self.failure = dict(
                    phase="guarded_v4_finalization",
                    error=f"{type(error).__name__}: {error}"[:512],
                    prefix=payload,
                    replacement_commit_permitted=False,
                    staged_failure=None if staged.failure is None else asdict(staged.failure),
                )
            raise
        self._replay, self.receipt = staged, receipt
        return receipt

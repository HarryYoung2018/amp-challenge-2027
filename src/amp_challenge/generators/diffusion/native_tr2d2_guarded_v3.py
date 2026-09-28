"""Prospective four-step TR2 replay with atomic aggregate KL acceptance.

Composes the retained replay-v2 numerical proposal on a private staged adapter.
No candidate becomes the live tree until the guarded receipt is sealed on time.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
)
from amp_challenge.generators.diffusion.native_shared_endpoint import (
    fullmask_reference_paths,
    interpolate_policy,
)
from amp_challenge.generators.diffusion.native_tr2_operator_kl import (
    TR2OperatorGuard,
    source_sha256,
)
from amp_challenge.generators.diffusion.native_tr2d2_replay_v2 import NativeTR2D2ReplayV2
from amp_challenge.generators.diffusion.native_tr2d2_replay_v2_records import source_bytes
from amp_challenge.generators.diffusion.native_weighted_training import (
    NativeWeightedReplay,
    weighted_anchor_diagnostics,
)

CONTRACT = {
    "version": "native-tr2-guarded-replay-v3",
    "proposal": "four_actual_replay_v2_weighted_steps_staged",
    "acceptance": "whole_tuple_aggregate_displacement_half_backtracks_0_through_8",
    "local_measure": "each_of_four_retained_weighted_active_probe_sets",
    "local_mean_limit": 0.01,
    "local_p99_limit": 0.02,
    "reference_path_mean_limit": 0.08,
    "reference_transition_p99_limit": 0.02,
    "operator_path_mean_limit": 0.08,
    "operator_transition_p99_limit": 0.02,
    "reference_paths_per_student": 8,
    "operator_paths_per_student": 8,
    "terminal": "no_unused_round29_update",
    "clock": "original_wave_180_and_scientific_7200_seconds_including_all_guards",
    "feasibility_change_enforced": False,
    "global_protocol_adopted": False,
    "campaign_eligible": False,
    "scientific_evidence_accepted": False,
}
CONTRACT_SHA256 = _json_hash(CONTRACT)


def guarded_source_sha256():
    return _json_hash(
        {
            "proposal": source_bytes(),
            "probe": source_sha256(),
            "guarded": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "contract": CONTRACT_SHA256,
        }
    )


@dataclass(frozen=True, slots=True)
class GuardedReplayAdvanceV3:
    record_json: str
    sha256: str


def _stage(adapter):
    result = copy.copy(adapter)
    result.tree = copy.copy(adapter.tree)
    result.versions = dict(adapter.versions)
    result.source = dict(adapter.source)
    result._wave_deadlines = dict(adapter._wave_deadlines)
    result._attempted_advances = set(adapter._attempted_advances)
    result._attempted_collections = set(adapter._attempted_collections)
    return result


def _probes(document):
    return NativeWeightedReplay(
        tuple(document["sequences"]),
        document["context_id"],
        tuple(
            NativeTransitionState(
                np.array(row["tokens"], dtype=np.int64), row["length"], row["level"]
            )
            for row in document["states"]
        ),
        np.array(document["weights"]),
    )


def _passed(row):
    return (
        all(
            local["local_old_candidate"]["mean"] <= CONTRACT["local_mean_limit"]
            and local["local_old_candidate"]["p99"] <= CONTRACT["local_p99_limit"]
            for local in row["local"]
        )
        and row["fullmask_reference"]["mean"] <= CONTRACT["reference_path_mean_limit"]
        and row["fullmask_reference"]["active_summary"]["p99"]
        <= CONTRACT["reference_transition_p99_limit"]
        and row["operator"]["mean"] <= CONTRACT["operator_path_mean_limit"]
        and row["operator"]["active_transition_p99"] <= CONTRACT["operator_transition_p99_limit"]
    )


class NativeTR2D2GuardedV3:
    """In-process numerical adapter; outer caller owns hard preemption/durability."""

    def __init__(self, *args, **kwargs):
        self._replay = NativeTR2D2ReplayV2(*args, **kwargs)
        self.source_sha256 = guarded_source_sha256()
        self.receipt = self.failure = None

    @property
    def tree(self):
        return self._replay.tree

    @property
    def generation(self):
        return self._replay.generation

    @property
    def versions(self):
        return dict(self._replay.versions)

    @property
    def policy_identities(self):
        return self._replay.policy_identities

    def _check(self):
        if self.failure is not None:
            raise ValueError("TR2 guarded-v3 terminal failure cannot restart")
        if guarded_source_sha256() != self.source_sha256 or _json_hash(CONTRACT) != CONTRACT_SHA256:
            raise ValueError("TR2 guarded-v3 source/contract changed")
        self._replay._check()

    def collect(self, evaluator, *, expected_posterior_sha256):
        self._check()
        return self._replay.collect(evaluator, expected_posterior_sha256=expected_posterior_sha256)

    def advance(self, history, *, expected_previous_head_sha256, outer_deadline):
        self._check()
        admission = self._replay._history_admission(history, expected_previous_head_sha256)
        if admission == "already_advanced":
            return self.receipt
        staged = _stage(self._replay)
        payload = {
            "contract": CONTRACT,
            "contract_sha256": CONTRACT_SHA256,
            "source_sha256": self.source_sha256,
            "history_sha256": history.sha256,
            "round_index": history.round_index,
            "proposal": None,
            "candidates": [],
            "chosen_backtracks": None,
            "status": "started",
            "kl_enforced": True,
            "feasibility_change_enforced": False,
            "campaign_eligible": False,
            "scientific_evidence_accepted": False,
        }

        def check():
            self._check()
            staged._time(history.round_index, "guarded_aggregate_acceptance")

        try:
            proposal = staged.advance(
                history,
                expected_previous_head_sha256=expected_previous_head_sha256,
                outer_deadline=outer_deadline,
            )
            if proposal is None:
                # A pause retains the earliest cap but publishes no candidate.
                self._replay._wave_deadlines = dict(staged._wave_deadlines)
                return None
            payload["proposal"] = json.loads(proposal.record_json)
            check()
            changed = [
                index
                for index, row in enumerate(payload["proposal"]["students"])
                if row["status"] == "four_actual_weighted_steps"
            ]
            replacements = self.tree._units
            accepted = False
            if changed:
                guard = TR2OperatorGuard(
                    self.tree._units,
                    self.generation.collection,
                    deadline=staged._wave_deadlines[history.round_index],
                    clock=staged.clock,
                )
                payload["operator_plan"] = guard.plan
                active_probes = {
                    index: tuple(
                        _probes(step["probes"])
                        for step in payload["proposal"]["students"][index]["steps"]
                    )
                    for index in changed
                }
                for backtracks in range(9):
                    candidate_row = {"backtracks": backtracks, "students": [], "passed": False}
                    payload["candidates"].append(candidate_row)
                    candidates = list(self.tree._units)
                    for index in changed:
                        check()
                        unit = self.tree._units[index]
                        candidate = interpolate_policy(
                            unit.model, staged.tree._units[index].model, backtracks
                        )
                        row = {
                            "triple": unit.triple,
                            "candidate_sha256": canonical_model_logical_hash(candidate),
                            "local": [],
                        }
                        candidate_row["students"].append(row)
                        for probes in active_probes[index]:
                            row["local"].append(
                                asdict(
                                    weighted_anchor_diagnostics(
                                        unit.model,
                                        candidate,
                                        unit.reference,
                                        probes,
                                        NATIVE_ENDPOINT_DEFAULTS,
                                    )
                                )
                            )
                            check()
                        row["fullmask_reference"] = fullmask_reference_paths(
                            unit,
                            candidate,
                            seed=history.seed,
                            semantic_sha256=history.sha256,
                            check=check,
                        )
                        report = guard.evaluate(
                            unit.model,
                            candidate,
                            unit.reference,
                            triple=unit.triple,
                            candidate_index=backtracks,
                        )
                        row["operator"] = asdict(report)
                        row["operator_record"] = guard.records[-1]
                        check()
                        row["passed"] = _passed(row)
                        replacement = copy.copy(unit)
                        replacement.model = candidate
                        replacement.policy_sha256 = row["candidate_sha256"]
                        candidates[index] = replacement
                    candidate_row["passed"] = all(
                        row["passed"] for row in candidate_row["students"]
                    )
                    if candidate_row["passed"]:
                        accepted, replacements = True, tuple(candidates)
                        payload["chosen_backtracks"] = backtracks
                        break
                payload["status"] = (
                    "accepted_guarded_update"
                    if accepted
                    else "all_backtracks_rejected_unchanged_tuple"
                )
                # Staging's proposed versions are committed only on acceptance.
                if not accepted:
                    staged.versions = dict(self._replay.versions)
            else:
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
            check()
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            receipt = GuardedReplayAdvanceV3(raw, hashlib.sha256(raw.encode()).hexdigest())
            check()
        except BaseException as error:
            self.failure = {
                "phase": "guarded_advance",
                "error": f"{type(error).__name__}: {error}"[:512],
                "prefix": payload,
                "staged_failure": None if staged.failure is None else asdict(staged.failure),
            }
            raise
        self._replay, self.receipt = staged, receipt
        return receipt

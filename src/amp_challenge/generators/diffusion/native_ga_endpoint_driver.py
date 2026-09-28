"""Persistent GA endpoint history/seat handoff, not an oracle transport.

Future reserves never enter this worker. Its caller supplies only14 composed
method seats; the controller owns two reserve seats and every charged response.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

from amp_challenge.evaluation.sequential_v2_seals import PhaseBuilder, verify_phase
from amp_challenge.generators.diffusion.native_ga_endpoint_arm import (
    READY,
    propose_ga_endpoint_wave,
    publish_ga_endpoint_wave,
)
from amp_challenge.generators.diffusion.native_ga_endpoint_records import (
    ARTIFACT,
    ChargedEndpointEligibility,
    GAEndpointWave,
)
from amp_challenge.generators.diffusion.native_ga_endpoint_verify import verify_ga_endpoint_wave
from amp_challenge.generators.diffusion.native_shared_endpoint_records import (
    TRIPLES,
    EndpointOrigin,
)
from amp_challenge.generators.search.peptide_ga_driver_v2 import _history
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_json_bytes,
    hash_string,
    require,
)
from amp_challenge.generators.search.verified_charged_history import read_verified_history

SEAT_ARTIFACT = "native_ga_endpoint_method_seats_v1"
MAX_RUN_BYTES = 5 * 1024**3


@dataclass(frozen=True, slots=True)
class FailedGAEndpointHandoff:
    """Retain already-computed evidence, never a delivered seat/model handoff."""

    stage: str
    error_type: str
    message: str
    result: GAEndpointWave
    model_commit_permitted: bool = False
    released_method_seats: tuple[str, ...] = ()


class GAEndpointDriver:
    """Caller-owned monotonic epoch includes loading and every resumed wave.

    Models advance atomically only after a14-seat handoff is persisted within
    the original deadline. Candidate evidence alone is not scientific authority.
    Blocking callbacks need outer hard preemption and authenticated resource logs.
    """

    def __init__(self, units, context, *, original_started_at, clock=time.monotonic):
        require(
            math.isfinite(original_started_at) and clock() >= original_started_at,
            "GA endpoint original monotonic start differs",
        )
        self.units, self.context = tuple(units), context
        self.context.__post_init__()
        self.original_started_at, self.original_deadline = (
            original_started_at,
            original_started_at + 7200,
        )
        self.clock, self.round_index = clock, 1
        self.native_ordinal, self.behavior_version = 0, 0
        self.phase_head = self.last_history = self.wave_started_at = self.pending = None
        self.seats, self.seat_previous_heads, self.terminal, self.published_bytes = {}, {}, False, 0
        self.origins, self.phase_count = {}, 0
        self.last_failure = None

    def _failed_handoff(self, result, stage, failure):
        self.terminal = True
        self.last_failure = FailedGAEndpointHandoff(
            stage, type(failure).__name__, str(failure)[:512], result
        )

    @property
    def deadline(self):
        return (
            min(self.original_deadline, self.wave_started_at + 180)
            if self.wave_started_at is not None
            else self.original_deadline
        )

    def _growth(self, history):
        previous = self.last_history
        if previous is not None:
            require(
                previous.round_index <= history.round_index <= previous.round_index + 1
                and history.observations[: len(previous.observations)] == previous.observations,
                "GA endpoint history was shortened, rewritten or skipped",
            )
            if previous.round_index == history.round_index and previous.complete:
                require(
                    previous.previous_wave_head_sha256 == history.previous_wave_head_sha256,
                    "GA endpoint paused wave changed predecessor head",
                )
        if history.round_index > 1:
            seats = self.seats.get(history.round_index - 1)
            require(seats is not None, "GA endpoint history advanced without previous method seats")
            begin = 64 + 16 * (history.round_index - 2)
            observed = tuple(row.sequence for row in history.observations[begin : begin + 14])
            require(
                observed == seats[: len(observed)], "GA endpoint revealed method seat order changed"
            )
            if history.complete:
                require(
                    history.previous_wave_head_sha256
                    != self.seat_previous_heads[history.round_index - 1],
                    "GA endpoint next complete wave lacks new controller head",
                )

    def _origin_records(self, history, eligibility):
        if eligibility is None:
            return {}
        eligibility.validate(history, self.context)
        incoming = dict(zip(eligibility.query_ids, eligibility.origins, strict=True))
        require(
            all(
                query not in self.origins or self.origins[query] == origin
                for query, origin in incoming.items()
            ),
            "GA endpoint immutable original generation/behavior was rewritten",
        )
        return incoming

    def propose(
        self,
        destination,
        callback,
        *,
        previous_wave_head_sha256,
        eligibility,
        evaluator,
        posterior_binding,
        enforce_kl=True,
        operator_guard=None,
        expected_operator_source=None,
        expected_operator_plan=None,
    ):
        require(
            not self.terminal and self.pending is None,
            "GA endpoint driver is terminal or awaiting seats",
        )
        require(self.phase_count < 128, "GA endpoint phase cap")
        if self.wave_started_at is None:
            self.wave_started_at = self.clock()
        if self.clock() >= self.deadline:
            self.terminal = True
            raise TimeoutError("GA endpoint driver original/wave deadline")
        fixed = self.context.driver
        history = read_verified_history(
            callback,
            provider_sha256=fixed.history_provider_sha256,
            run_id=fixed.run_id,
            seed=fixed.seed,
            round_index=self.round_index,
            objective_context_sha256=fixed.objective_context_sha256,
            oracle_bundle_sha256=fixed.oracle_bundle_sha256,
            previous_wave_head_sha256=previous_wave_head_sha256,
        )
        self._growth(history)
        incoming = self._origin_records(history, eligibility)
        candidates, result = propose_ga_endpoint_wave(
            self.units,
            history,
            eligibility,
            context=self.context,
            evaluator=evaluator,
            posterior_binding=posterior_binding,
            native_ordinal=self.native_ordinal,
            behavior_version=self.behavior_version,
            deadline=self.deadline,
            enforce_kl=enforce_kl,
            operator_guard=operator_guard,
            expected_operator_source=expected_operator_source,
            expected_operator_plan=expected_operator_plan,
            clock=self.clock,
        )
        size = len(result.record_json.encode()) + sum(
            len(data) for _, data in result.checkpoint_payloads
        )
        try:
            require(
                self.published_bytes + size + 1024**2 <= MAX_RUN_BYTES,
                "GA endpoint five-GiB publication cap",
            )
            seal = publish_ga_endpoint_wave(
                destination,
                result,
                previous_phase_sha256=self.phase_head,
                deadline=self.deadline,
                clock=self.clock,
                metadata={
                    "original_started_at": self.original_started_at,
                    "original_deadline": self.original_deadline,
                    "wave_started_at": self.wave_started_at,
                },
            )
        except (TimeoutError, ValueError, OSError) as failure:
            self._failed_handoff(result, "wave_publication", failure)
            raise
        self.published_bytes += sum(len(data) for _, data in seal.payload_bytes) + 1024**2
        require(self.published_bytes <= MAX_RUN_BYTES, "GA endpoint final publication cap")
        self.phase_head, self.last_history = seal.seal_sha256, history
        self.origins.update(incoming)
        self.phase_count += 1
        if result.status == READY:
            self.pending = (candidates, result, history, seal.seal_sha256)
        elif result.status != "paused_incomplete_wave":
            self.terminal = True
        return result, seal

    def commit_method_seats(self, destination, sequences, *, composition_receipt_sha256):
        require(
            not self.terminal and self.pending is not None,
            "GA endpoint has no live proposal to compose",
        )
        candidates, result, history, predecessor = self.pending
        if self.phase_count >= 128:
            failure = ValueError("GA endpoint phase cap")
            self._failed_handoff(result, "method_seat_publication", failure)
            raise failure
        require(
            type(sequences) is tuple
            and len(sequences) == len(set(sequences)) == 14
            and hash_string(composition_receipt_sha256),
            "GA endpoint requires14 unique composed method seats",
        )
        positions = tuple(result.method_pool.index(seq) for seq in sequences)
        require(
            positions == tuple(sorted(positions)),
            "GA endpoint composed seats violate ranked ordering",
        )
        if self.clock() >= self.deadline:
            failure = TimeoutError("GA endpoint method-seat original deadline")
            self._failed_handoff(result, "method_seat_publication", failure)
            raise failure
        payload = {
            "artifact": SEAT_ARTIFACT,
            "wave_sha256": predecessor,
            "history_sha256": history.sha256,
            "round_index": history.round_index,
            "ranked_positions": positions,
            "sequences": sequences,
            "composition_receipt_sha256": composition_receipt_sha256,
            "method_seats": 14,
            "private_reserve_seats": 2,
            "campaign_eligible": False,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
        }
        try:
            encoded = canonical_json_bytes(payload)
            require(
                self.published_bytes + len(encoded) + 1024**2 <= MAX_RUN_BYTES,
                "GA endpoint seat publication cap",
            )
            with PhaseBuilder(
                destination, artifact=SEAT_ARTIFACT, predecessor_seals={"wave": predecessor}
            ) as builder:
                builder.write_bytes("seats.json", encoded)
                if self.clock() >= self.deadline:
                    raise TimeoutError("GA endpoint staged method-seat deadline")
                seal = builder.publish(expected_payload_paths=("seats.json",))
            if self.clock() >= self.deadline:
                raise TimeoutError(
                    "GA endpoint late seat evidence retained without released handoff"
                )
            self.published_bytes += sum(len(data) for _, data in seal.payload_bytes) + 1024**2
            require(self.published_bytes <= MAX_RUN_BYTES, "GA endpoint seat publication cap")
        except (TimeoutError, ValueError, OSError) as failure:
            self._failed_handoff(result, "method_seat_publication", failure)
            raise
        self.units = candidates
        self.phase_count += 1
        self.native_ordinal, self.behavior_version = (
            result.next_native_ordinal,
            result.next_behavior_version,
        )
        self.seats[self.round_index] = sequences
        self.seat_previous_heads[self.round_index] = history.previous_wave_head_sha256
        self.round_index += 1
        self.phase_head, self.pending, self.wave_started_at = seal.seal_sha256, None, None
        return sequences, seal


def reconstruct_ga_endpoint_driver(
    initial_units,
    context,
    phase_paths,
    *,
    original_started_at,
    clock=time.monotonic,
    operator_report_verifier=None,
):
    """Reconstruct completed math/history/seats without future reserve disclosure.

    The caller supplies the original epoch; no clock reset. Same-account phase
    reconstruction does not authenticate scientific provider behavior or timing.
    """
    require(
        type(phase_paths) is tuple and len(phase_paths) <= 128, "GA endpoint phase inventory cap"
    )
    state = GAEndpointDriver(
        initial_units, context, original_started_at=original_started_at, clock=clock
    )
    for path in phase_paths:
        require(not state.terminal, "GA endpoint phases continue after terminal state")
        seal = verify_phase(path)
        state.phase_count += 1
        state.published_bytes += sum(len(data) for _, data in seal.payload_bytes) + 1024**2
        require(state.published_bytes <= MAX_RUN_BYTES, "GA endpoint reconstructed byte cap")
        if seal.artifact == ARTIFACT:
            require(state.pending is None, "GA endpoint next proposal precedes method-seat handoff")
            require(
                dict(seal.predecessor_seals)
                == ({} if state.phase_head is None else {"previous": state.phase_head}),
                "GA endpoint predecessor phase differs",
            )
            metadata = json.loads(seal.metadata_json)
            require(
                metadata["original_started_at"] == original_started_at
                and metadata["original_deadline"] == state.original_deadline,
                "GA endpoint original clock was reset",
            )
            wave_start = metadata["wave_started_at"]
            require(
                type(wave_start) in (float, int)
                and math.isfinite(wave_start)
                and original_started_at <= wave_start <= state.original_deadline,
                "GA endpoint wave start differs",
            )
            if state.wave_started_at is not None:
                require(
                    state.wave_started_at == wave_start, "GA endpoint paused-wave clock was reset"
                )
            state.wave_started_at = wave_start
            raw = json.loads(seal.read_payload_bytes("wave.json"))
            history = _history(raw["history"])
            require(
                history.round_index == state.round_index, "GA endpoint reconstructed round differs"
            )
            state._growth(history)
            source = raw["eligibility"]
            if source is not None:
                source = ChargedEndpointEligibility(
                    **{
                        **source,
                        "query_ids": tuple(source["query_ids"]),
                        "origins": tuple(EndpointOrigin(**origin) for origin in source["origins"]),
                    }
                )
            incoming = state._origin_records(history, source)
            result = GAEndpointWave(
                seal.read_payload_bytes("wave.json").decode(),
                dict(seal.payload_sha256)["wave.json"],
                raw["status"],
                tuple(raw["pool"][index]["sequence"] for index in raw["ranked_positions"]),
                raw["next_native_ordinal"],
                raw["next_behavior_version"],
                tuple(
                    (triple, seal.read_payload_bytes(f"working_models/{triple}.safetensors"))
                    for triple in TRIPLES
                ),
            )
            require(
                raw["native_ordinal"] == state.native_ordinal
                and raw["behavior_version"] == state.behavior_version,
                "GA endpoint attempt/model version reset",
            )
            candidates, _ = verify_ga_endpoint_wave(
                state.units,
                history,
                source,
                context=context,
                result=result,
                phase=path,
                operator_report_verifier=operator_report_verifier,
            )
            state.phase_head, state.last_history = seal.seal_sha256, history
            state.origins.update(incoming)
            if result.status == READY:
                state.pending = (candidates, result, history, seal.seal_sha256)
            elif result.status != "paused_incomplete_wave":
                state.terminal = True
        else:
            require(
                seal.artifact == SEAT_ARTIFACT and state.pending is not None,
                "GA endpoint unexpected phase or orphan method seats",
            )
            candidates, result, history, predecessor = state.pending
            require(
                tuple(name for name, _ in seal.payload_sha256) == ("seats.json",),
                "GA endpoint method-seat payload inventory differs",
            )
            require(
                dict(seal.predecessor_seals) == {"wave": predecessor},
                "GA endpoint seat predecessor differs",
            )
            raw = json.loads(seal.read_payload_bytes("seats.json"))
            require(
                raw["artifact"] == SEAT_ARTIFACT
                and raw["wave_sha256"] == predecessor
                and raw["history_sha256"] == history.sha256
                and raw["round_index"] == state.round_index
                and raw["method_seats"] == 14
                and raw["private_reserve_seats"] == 2
                and hash_string(raw["composition_receipt_sha256"])
                and all(
                    raw[name] is False
                    for name in (
                        "campaign_eligible",
                        "scientific_evidence_accepted",
                        "production_eligible",
                    )
                ),
                "GA endpoint method-seat metadata differs",
            )
            positions = raw["ranked_positions"]
            require(
                type(positions) is list
                and len(positions) == 14
                and positions == sorted(set(positions))
                and all(
                    type(index) is int and 0 <= index < len(result.method_pool)
                    for index in positions
                )
                and raw["sequences"] == [result.method_pool[index] for index in positions],
                "GA endpoint method seats do not reconstruct from ranking",
            )
            state.units = candidates
            state.native_ordinal, state.behavior_version = (
                result.next_native_ordinal,
                result.next_behavior_version,
            )
            state.seats[state.round_index] = tuple(raw["sequences"])
            state.seat_previous_heads[state.round_index] = history.previous_wave_head_sha256
            state.round_index += 1
            state.phase_head, state.pending, state.wave_started_at = seal.seal_sha256, None, None
    return state

"""Additive context-eligible native updates; no oracle, ranking or campaign authority."""

from __future__ import annotations

import copy
import hashlib
import time
from contextlib import suppress

import numpy as np

from amp_challenge.generators.diffusion import native_baseline_context_records as records
from amp_challenge.generators.diffusion.model import canonical_model_logical_hash
from amp_challenge.generators.diffusion.native_baseline_context_verify import verify_context_advance
from amp_challenge.generators.diffusion.native_baseline_operators import (
    REWARD_LIMITS,
    NativeBaselineAdvance,
    NativeBaselineEnsemble,
    NativeBaselineStep,
    sequence_id,
    verify_native_pool,
)
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    _json_hash,
    endpoint_candidate,
)
from amp_challenge.generators.diffusion.native_weighted_training import (
    build_weighted_replay,
    propose_weighted_direction,
    weighted_anchor_diagnostics,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import require
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot


def _weighted_rows(unit, history, eligibility, context, mode, step):
    # Membership is checked BEFORE scalarization, thresholding, top64 and max.
    supported = []
    for row in history.observations:
        if row.status != "successful" or row.query_id not in eligibility.query_ids:
            continue
        value = context.scalarize(row.objectives)
        if mode == "arcadiamp_style_iterative_d3pm" and value < 0.5:
            continue
        supported.append((row, value))
    if not supported:
        return None

    def order(sequence, role):
        return _json_hash(
            [
                "native-baseline-replay-v1",
                history.seed,
                unit.triple,
                history.round_index,
                step,
                role,
                sequence,
            ]
        )

    anchors = tuple(sorted(unit.sequences, key=lambda seq: order(seq, "anchor"))[:64])
    selected = tuple(sorted(supported, key=lambda pair: order(pair[0].sequence, "revealed"))[:64])
    values = np.asarray([pair[1] for pair in selected], dtype=np.float64)
    if mode == "diffusion_reward_kl_no_search":
        weights = np.exp(np.clip((values - values.max()) / 0.1, -5.0, 0.0))
    else:
        weights = np.clip(values, 0.05, 1.0)
    weights = np.concatenate(
        (np.full(len(anchors), 0.5 / len(anchors)), 0.5 * weights / weights.sum())
    )
    return (
        anchors + tuple(pair[0].sequence for pair in selected),
        weights,
        tuple(pair[0].query_id for pair in selected),
    )


def _advance_unit(unit, history, eligibility, context, checkpoint, progress):
    working, steps, selections = unit.model, [], []
    status = "unchanged_by_declared_schedule"

    def receipt():
        return NativeBaselineAdvance(
            unit.triple,
            history.round_index,
            history.sha256,
            history.receipt_sha256,
            status,
            tuple(steps),
            unit.policy_sha256,
            canonical_model_logical_hash(working),
            unit.reference_sha256,
        )

    scheduled = (context.mode == "diffusion_reward_kl_no_search" and history.round_index == 1) or (
        context.mode == "arcadiamp_style_iterative_d3pm" and 2 <= history.round_index <= 28
    )
    if scheduled:
        for index in range(1 if context.mode == "diffusion_reward_kl_no_search" else 4):
            checkpoint("student_" + unit.triple + "_step_" + str(index))
            rows = _weighted_rows(
                unit, history, eligibility, context.objective, context.mode, index
            )
            if rows is None:
                status = "no_supported_revealed_rows_no_update"
                break
            sequences, weights, selected = rows
            seed = int(_json_hash([history.seed, unit.triple, "native-training"])[:16], 16)
            arguments = {
                "context_id": context.objective.context_sha256,
                "seed": seed,
                "ordinal": history.round_index * 4 + index,
            }
            replay = build_weighted_replay(working, sequences, weights, **arguments)
            probes = build_weighted_replay(
                unit.reference, sequences, weights, active_probes=True, **arguments
            )
            direction = propose_weighted_direction(working, replay, NATIVE_ENDPOINT_DEFAULTS)
            gradient = hashlib.sha256()
            for name, tensor in direction.gradients:
                gradient.update(name.encode() + b"\0" + tensor.numpy().tobytes())
            enforce = context.mode == "diffusion_reward_kl_no_search"
            for backtracks in range(
                NATIVE_ENDPOINT_DEFAULTS.maximum_backtracks + 1 if enforce else 1
            ):
                checkpoint("student_" + unit.triple + "_backtrack_" + str(backtracks))
                candidate = endpoint_candidate(working, direction, backtracks=backtracks)
                diagnostics = weighted_anchor_diagnostics(
                    working, candidate, unit.reference, probes, NATIVE_ENDPOINT_DEFAULTS
                )
                local, frozen = (
                    diagnostics.local_old_candidate,
                    diagnostics.candidate_frozen_reference,
                )
                accepted = not enforce or (
                    local.mean <= REWARD_LIMITS.local_mean
                    and local.p99 <= REWARD_LIMITS.local_p99
                    and frozen.mean <= REWARD_LIMITS.reference_mean
                    and frozen.p99 <= REWARD_LIMITS.reference_p99
                )
                if accepted:
                    break
            after = (
                canonical_model_logical_hash(candidate) if accepted else direction.base_model_sha256
            )
            steps.append(
                NativeBaselineStep(
                    replay,
                    probes,
                    direction.base_model_sha256,
                    after,
                    gradient.hexdigest(),
                    direction.objective_before,
                    direction.gradient_norm_before_clip,
                    accepted,
                    backtracks,
                    enforce,
                    diagnostics,
                )
            )
            selections.append(selected)
            if accepted:
                working, status = candidate, "updated"
            else:
                status = "constrained_update_rejected_policy_frozen"
            progress(receipt(), tuple(selections))
            if not accepted:
                break
    replacement = copy.copy(unit)
    replacement.model, replacement.policy_sha256 = working, canonical_model_logical_hash(working)
    return replacement, receipt(), tuple(selections)


def _admit(history, eligibility, context, expected):
    require(type(history) is VerifiedHistorySnapshot, "exact raw history required")
    require(type(eligibility) is records.NativeBaselineEligibility, "exact eligibility required")
    require(
        type(expected) is records.NativeBaselineExpectations, "exact external expectations required"
    )
    history.__post_init__()
    eligibility.__post_init__()
    expected.__post_init__()
    require(
        (
            history.run_id,
            history.seed,
            history.objective_context_sha256,
            history.oracle_bundle_sha256,
            history.previous_wave_head_sha256,
        )
        == (
            context.run_id,
            context.seed,
            context.objective.context_sha256,
            context.oracle_bundle_sha256,
            expected.previous_wave_head_sha256,
        ),
        "external raw history identity differs",
    )
    require(
        history.sha256 == eligibility.history_sha256 == expected.history_sha256
        and history.objective_context_sha256
        == eligibility.objective_context_sha256
        == expected.objective_context_sha256,
        "external history/context binding differs",
    )
    require(
        (eligibility.source_sha256, eligibility.receipt_sha256, eligibility.query_ids)
        == (
            expected.eligibility_source_sha256,
            expected.eligibility_receipt_sha256,
            expected.eligible_query_ids,
        ),
        "external applicability source/receipt/subset differs",
    )
    require(
        eligibility.query_ids
        <= frozenset(row.query_id for row in history.observations if row.status == "successful"),
        "applicability must contain successful query IDs only",
    )


class NativeBaselineContextEnsemble:
    """In-memory composition. Original-clock enforcement is not hard preemption.

    New admissions are independently reconstructed before commit. Idempotent
    reuse requires the original exact expectations, not a rotated receipt.
    Begun work failures are terminal and retain unaccepted evidence. A pool
    still needs the separately expected update-envelope head when verified.
    """

    def __init__(self, context, initializations, generator_sequences, *, monotonic=time.monotonic):
        require(type(context) is records.NativeBaselineContext, "exact context required")
        context.__post_init__()
        context_sha = records.document_sha(context)
        sources = records.source_identities()
        require(
            records.document_sha(sources) == context.implementation_sha256,
            "actual context baseline source pin differs",
        )
        now = records.read_clock(monotonic)
        deadline = records.finite_clock(context.deadline_monotonic)
        require(now < deadline and deadline - now <= 7200, "original 7200-second deadline differs")
        require(records.document_sha(context) == context_sha, "constructor context drift")
        self.context, self.monotonic = context, monotonic
        self._context_sha, self._sources = context_sha, sources
        self._deadline, self._last_clock = deadline, now
        self.inner = NativeBaselineEnsemble(
            context.mode,
            initializations,
            generator_sequences,
            context.objective,
            run_id=context.run_id,
            seed=context.seed,
            oracle_bundle_sha256=context.oracle_bundle_sha256,
        )
        self.last_envelope = None
        self._accepted_identity = None
        self._pool_identity = None
        self.last_failure = None
        self._terminal = False

    def _clock(self):
        now = records.read_clock(self.monotonic)
        require(
            self._last_clock <= now < self._deadline, "original deadline/clock ordering violated"
        )
        self._last_clock = now
        return now

    def _fixed(self):
        require(
            (self.last_envelope is None) == (self._accepted_identity is None),
            "committed envelope head was cleared",
        )
        require(type(self.context) is records.NativeBaselineContext, "context type drift")
        self.context.__post_init__()
        require(
            records.document_sha(self.context) == self._context_sha
            and self.context.deadline_monotonic == self._deadline,
            "fixed context/deadline drift",
        )
        require(records.source_identities() == self._sources, "executed repository source drift")
        require(
            (self.inner.mode, self.inner.run_id, self.inner.seed, self.inner.oracle_bundle_sha256)
            == (
                self.context.mode,
                self.context.run_id,
                self.context.seed,
                self.context.oracle_bundle_sha256,
            ),
            "inner fixed identity drift",
        )
        require(
            records.document_sha(self.inner.context) == records.document_sha(self.context.objective)
            and self.inner.config == NATIVE_ENDPOINT_DEFAULTS,
            "inner recipe/context drift",
        )
        require(
            self.inner._history is None or type(self.inner._history) is VerifiedHistorySnapshot,
            "retained history type drift",
        )
        records.inner_binding(self.inner)
        require(
            records.pool_identity(self.inner._pool) == self._pool_identity,
            "actual committed native pool identity differs",
        )

    def _accepted(self):
        require(
            type(self.last_envelope) is records.ContextEnvelope,
            "accepted eligibility envelope required",
        )
        require(
            (self.last_envelope.payload, self.last_envelope.sha256) == self._accepted_identity,
            "actual committed envelope identity differs",
        )
        document = self.last_envelope.document()
        require(
            document["artifact"] == records.ARTIFACT
            and document["configuration_sha256"] == records.CONFIG_SHA256
            and all(
                document[key] is False
                for key in (
                    "scientific_evidence_accepted",
                    "campaign_eligible",
                    "production_eligible",
                )
            ),
            "accepted envelope contract/authority differs",
        )
        require(
            document["context"] == records.plain(self.context)
            and document["sources"] == self._sources,
            "accepted context/source drift",
        )
        bound = records.inner_binding(self.inner)
        require(document["new_units"] == bound["units"], "accepted resulting unit lineage differs")
        if document["status"] == "completed":
            require(
                self.inner._history is not None
                and records.plain(self.inner._history) == document["history"]
                and records.plain(self.inner._receipts) == document["advances"],
                "accepted raw history/numerical receipts differ",
            )
        else:
            require(
                document["status"] == "paused_incomplete_history"
                and bound == document["old_inner"],
                "paused inner state changed",
            )
        return document

    def _growth(self, history, eligibility, expected, prior):
        require(
            expected.previous_update_envelope_sha256
            == (None if self.last_envelope is None else self.last_envelope.sha256),
            "expected update predecessor differs",
        )
        if prior is None:
            require(
                self.inner._history is None
                and self.inner._pool is None
                and not self.inner._receipts
                and all(unit.policy_sha256 == unit.reference_sha256 for unit in self.inner._units),
                "virgin inner policy/history required",
            )
        else:
            old = records.history_from_document(prior["history"])
            previous = records.eligibility_from_document(prior["eligibility"])
            require(
                history.observations[: len(old.observations)] == old.observations,
                "prior raw charged prefix changed",
            )
            require(
                eligibility.source_sha256 == previous.source_sha256
                and all(
                    (row.query_id in eligibility.query_ids) == (row.query_id in previous.query_ids)
                    for row in old.observations
                ),
                "immutable prior applicability changed",
            )
            if prior["status"] == "paused_incomplete_history":
                require(history.round_index == old.round_index, "paused round changed")
        require(
            not any(
                sequence_id(row.sequence) in self.inner._training_ids
                for row in history.observations
            ),
            "raw charged generator-training overlap",
        )
        old = self.inner._history
        require(
            history.round_index == (1 if old is None else old.round_index + 1),
            "contiguous raw-history round required",
        )
        if old is not None:
            require(
                self.context.mode == "arcadiamp_style_iterative_d3pm",
                "frozen reward arm cannot consume adaptive responses",
            )
            require(
                history.observations[: len(old.observations)] == old.observations,
                "committed charged prefix changed",
            )
            require(
                self.inner._pool is not None and self.inner._pool.status == "complete",
                "completed prior candidate pool required",
            )

    def advance(self, history, eligibility, *, expected):
        require(not self._terminal, "context baseline is terminal after failed work")
        self._fixed()
        _admit(history, eligibility, self.context, expected)
        prior = None if self.last_envelope is None else self._accepted()
        old_binding = records.inner_binding(self.inner)
        runtime = records.caller_state(self.inner._units)
        inputs_sha = records.document_sha((history, eligibility, expected))
        previous_identity = (
            None
            if self.last_envelope is None
            else (self.last_envelope.payload, self.last_envelope.sha256)
        )
        previous = self.last_envelope
        accepted_identity = self._accepted_identity
        timing, phase = [], "entry"
        document = {
            "artifact": records.ARTIFACT,
            "configuration_sha256": records.CONFIG_SHA256,
            "context": records.plain(self.context),
            "expected": records.plain(expected),
            "history": records.plain(history),
            "eligibility": records.plain(eligibility),
            "numerical_input_sha256": records.numerical_input_sha256(
                self.context, history, eligibility
            ),
            "previous_envelope_sha256": expected.previous_update_envelope_sha256,
            "status": "completed" if history.complete else "paused_incomplete_history",
            "old_inner": old_binding,
            "advances": [],
            "selected_query_ids": [],
            "new_units": old_binding["units"],
            "sources": self._sources,
            "timing": timing,
            "failure": None,
            "scientific_evidence_accepted": False,
            "campaign_eligible": False,
            "production_eligible": False,
        }

        def bindings():
            self._fixed()
            _admit(history, eligibility, self.context, expected)
            require(
                records.document_sha((history, eligibility, expected)) == inputs_sha,
                "raw input/applicability drift",
            )
            require(
                records.inner_binding(self.inner) == old_binding
                and records.caller_state(self.inner._units) == runtime,
                "caller inner/mode/gradient drift",
            )
            require(
                self.last_envelope is previous
                and (previous is None or (previous.payload, previous.sha256) == previous_identity),
                "accepted predecessor drift",
            )
            if previous is not None:
                require(
                    type(previous) is records.ContextEnvelope
                    and type(previous.payload) is bytes
                    and type(previous.sha256) is str,
                    "accepted predecessor exact types drift",
                )
                previous.document()
            require(
                self._accepted_identity == accepted_identity, "committed acceptance identity drift"
            )

        def checkpoint(name):
            nonlocal phase
            phase = name
            now = self._clock()
            bindings()  # External clock precedes callback-free drift checks.
            timing.append({"phase": name, "at_monotonic": now})

        try:
            checkpoint("entry")
            if prior is not None and records.plain(history) == prior["history"]:
                require(
                    records.plain(eligibility) == prior["eligibility"]
                    and records.plain(expected) == prior["expected"],
                    "idempotent exact applicability/expectations differ",
                )
                checkpoint("idempotent_return")
                return previous
            self._growth(history, eligibility, expected, prior)
            replacements, advances = [], []
            if history.complete:
                for unit in self.inner._units:

                    def progress(advance, selected):
                        document["advances"] = records.plain((*advances, advance))
                        document["selected_query_ids"] = records.plain(
                            (*document["selected_query_ids"][: len(advances)], selected)
                        )

                    replacement, advance, selected = _advance_unit(
                        unit, history, eligibility, self.context, checkpoint, progress
                    )
                    replacements.append(replacement)
                    advances.append(advance)
                    document["advances"] = records.plain(advances)
                    document["selected_query_ids"] = records.plain(
                        (*document["selected_query_ids"][: len(advances) - 1], selected)
                    )
            else:
                replacements = list(self.inner._units)
            staged = tuple(replacements)
            document["new_units"] = [records.unit_binding(unit) for unit in staged]
            staged_sha = records.document_sha(document["new_units"])
            staged_runtime = records.caller_state(staged)
            checkpoint("before_seal_and_independent_reconstruction")
            envelope = records.seal_envelope(document)
            sealed_identity = (envelope.payload, envelope.sha256)
            verify_context_advance(
                self.inner,
                previous,
                envelope,
                context=self.context,
                expected=expected,
                expected_new_units=staged,
                monotonic=self._clock,
            )
            self._clock()
            bindings()
            require(
                type(envelope) is records.ContextEnvelope
                and type(envelope.payload) is bytes
                and type(envelope.sha256) is str,
                "encoded envelope exact types drift",
            )
            envelope.document()
            require(
                (envelope.payload, envelope.sha256) == sealed_identity, "encoded envelope drift"
            )
            require(
                records.document_sha([records.unit_binding(unit) for unit in staged]) == staged_sha
                and records.caller_state(staged) == staged_runtime,
                "staged replacement drift",
            )
            if history.complete:
                self.inner._units, self.inner._receipts = staged, tuple(advances)
                self.inner._history, self.inner._pool = history, None
                self._pool_identity = None
            self.last_envelope = envelope
            self._accepted_identity = sealed_identity
            return envelope
        except Exception as exc:
            self._terminal = True
            document.update(
                status="failed",
                new_units=old_binding["units"],
                failure={"type": type(exc).__name__, "message": str(exc), "phase": phase},
            )
            # A sealing failure cannot erase available unaccepted in-memory evidence.
            self.last_failure = document
            with suppress(Exception):
                self.last_failure = records.seal_envelope(document)
            raise

    def _pool_operation(self, *, max_new_attempts=None, pool=None, expected_update_envelope_sha256):
        require(not self._terminal, "context baseline is terminal after failed work")
        self._fixed()
        admitted = self._accepted()
        require(
            admitted["status"] == "completed"
            and self.last_envelope.sha256 == expected_update_envelope_sha256,
            "externally expected completed update envelope required for pool",
        )
        old = records.inner_binding(self.inner)
        runtime = records.caller_state(self.inner._units)
        envelope = self.last_envelope
        identity = (envelope.payload, envelope.sha256)
        staged = copy.copy(self.inner)

        def bindings():
            self._fixed()
            require(
                records.inner_binding(self.inner) == old
                and records.caller_state(self.inner._units) == runtime,
                "pool caller state drift",
            )
            require(
                self.last_envelope is envelope and (envelope.payload, envelope.sha256) == identity,
                "pool eligibility envelope drift",
            )
            self._accepted()

        try:
            self._clock()
            bindings()
            if pool is None:
                pool = staged.propose(max_new_attempts=max_new_attempts)
            verify_native_pool(staged, pool)
            pool_identity = records.pool_identity(pool)
            self._clock()
            bindings()
            require(records.pool_identity(pool) == pool_identity, "verified pool value/type drift")
            if max_new_attempts is not None:
                self.inner._pool = pool
                self._pool_identity = pool_identity
            return pool
        except Exception as exc:
            self._terminal = True
            self.last_failure = {
                "status": "failed_unaccepted_pool_work",
                "type": type(exc).__name__,
                "message": str(exc),
                "update_envelope_sha256": envelope.sha256,
                "available_pool": records.plain(pool if pool is not None else staged._pool),
                "old_inner": old,
                "scientific_evidence_accepted": False,
                "production_eligible": False,
            }
            raise

    def propose(self, *, expected_update_envelope_sha256, max_new_attempts=128):
        return self._pool_operation(
            max_new_attempts=max_new_attempts,
            expected_update_envelope_sha256=expected_update_envelope_sha256,
        )

    def verify_pool(self, pool, *, expected_update_envelope_sha256):
        return self._pool_operation(
            pool=pool, expected_update_envelope_sha256=expected_update_envelope_sha256
        )

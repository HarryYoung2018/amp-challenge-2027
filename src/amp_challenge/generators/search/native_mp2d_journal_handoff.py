"""Same-live-object MP2D selection to durable journal, without transport.

Private driver/journal/bridge fields below are deliberately pinned compatibility
dependencies. This is neither a public recovery API nor a privacy boundary.
The native driver's selection may already have advanced when publication fails.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from inspect import getattr_static
from pathlib import Path

from amp_challenge.evaluation import sequential_v2_seals
from amp_challenge.evaluation.sequential_v2_seals import PhaseSeal, verify_phase
from amp_challenge.generators.diffusion import native_mp2d_driver as native
from amp_challenge.generators.diffusion.native_mp2d_driver import (
    NativeMP2DDriver,
    NativeMP2DPreparation,
    NativeMP2DSeats,
)
from amp_challenge.generators.search import durable_dispatch_journal as journal_module
from amp_challenge.generators.search import durable_dispatch_journal_records as jr
from amp_challenge.generators.search import durable_dispatch_journal_verify as journal_checker
from amp_challenge.generators.search import native_mp2d_journal_handoff_records as records
from amp_challenge.generators.search.durable_dispatch_journal import DurableDispatchJournal
from amp_challenge.generators.search.durable_dispatch_journal_verify import (
    JournalHistoryCallback,
    verify_dispatch_journal,
)
from amp_challenge.models.charged_probability_learner import GaussianLearnerSnapshot
from amp_challenge.representations import run_feature_cache_records as feature_records
from amp_challenge.representations import run_feature_cache_verify as feature_checker
from amp_challenge.representations import run_feature_cache_views as feature_views
from amp_challenge.representations.run_feature_cache_records import canonical, finite_clock
from amp_challenge.representations.run_feature_cache_views import NativeFeaturePosterior

require = jr.require


def _checkpoint_state(checkpoint):
    require(type(checkpoint) is jr.JournalCheckpoint, "handoff checkpoint exact type differs")
    return (
        checkpoint.genesis_sha256,
        checkpoint.event_count,
        checkpoint.head_sha256,
        checkpoint.charged_count,
    )


def _phase_state(seal):
    if seal is None:
        return None
    require(type(seal) is PhaseSeal, "handoff phase exact type differs")
    return (
        seal.artifact,
        seal.seal_sha256,
        seal.receipt_sha256,
        seal.predecessor_seals,
        seal.payload_sha256,
        seal.payload_bytes,
        seal.files,
        seal.metadata_json,
    )


def _native_state(result):
    require(
        type(result) in (NativeMP2DPreparation, NativeMP2DSeats),
        "handoff native result type differs",
    )
    common = (id(result), result.round_index, result.status, result.record_payload)
    if type(result) is NativeMP2DPreparation:
        return (*common, result.poll_ordinal, result.history_sha256, result.wave_sha256)
    require(type(result) is NativeMP2DSeats, "handoff native result type differs")
    return (*common, result.preparation_sha256, result.method_sequences)


def _result_state(result):
    require(
        type(result) is records.MP2DJournalHandoffResult, "handoff retained result type differs"
    )
    return (
        id(result),
        result.round_index,
        result.status,
        result.preparation_sha256,
        result.seats_sha256,
        _checkpoint_state(result.previous_checkpoint),
        None if result.checkpoint is None else _checkpoint_state(result.checkpoint),
        _phase_state(result.intent_seal),
        _phase_state(result.completion_seal),
        result.record_payload,
    )


class NativeMP2DJournalHandoff:
    def __init__(
        self,
        driver,
        journal,
        *,
        journal_binding,
        initial_checkpoint,
        journal_source_inventory,
        journal_authenticator,
        handoff_root,
        monotonic=time.monotonic,
    ):
        require(
            type(driver) is NativeMP2DDriver
            and type(journal) is DurableDispatchJournal
            and type(journal_binding) is jr.JournalBinding
            and type(initial_checkpoint) is jr.JournalCheckpoint
            and type(journal_authenticator) is jr.CallbackPin
            and callable(monotonic),
            "handoff requires exact live native/journal authorities",
        )
        journal_binding.__post_init__()
        initial_checkpoint.__post_init__()
        journal_authenticator.__post_init__()
        require(
            initial_checkpoint.event_count == 1
            and initial_checkpoint.charged_count == 64
            and initial_checkpoint.genesis_sha256 == journal_binding.sha256
            and journal.checkpoint == initial_checkpoint,
            "handoff requires the exactly adopted imported initial block",
        )
        require(
            driver._next_round == 1
            and driver._current is None
            and driver._history is None
            and not driver._requests
            and not driver._selections
            and not driver._busy
            and not driver.stopped
            and not journal._busy,
            "handoff must be constructed before the first native preparation",
        )
        self._driver, self._journal, self._bridge = driver, journal, driver._bridge
        self._binding, self._inventory, self._auth = (
            journal_binding,
            journal_source_inventory,
            journal_authenticator,
        )
        self._clock, self._root = monotonic, records.directory(handoff_root)
        self._run_root = records.directory(Path(self._bridge.binding.run_root))
        require(
            self._root.parent == self._run_root
            and journal.root.parent == self._run_root
            and self._root != journal.root
            and journal._parent == self._run_root
            and not any(self._root.iterdir()),
            "handoff needs an empty sibling root under the existing feature run root",
        )
        binding = self._bridge.binding
        require(
            (binding.run_id, binding.arm_id, binding.seed, binding.objective_context_sha256)
            == (
                journal_binding.run_id,
                journal_binding.arm_id,
                journal_binding.seed,
                journal_binding.objective_context_sha256,
            )
            and journal_binding.arm_id == native.MP2D_ARM
            and driver._pins[0] == journal_binding.provider_sha256
            and driver._pins[1] == journal_binding.oracle_bundle_sha256
            and jr.source_inventory_sha256(journal_source_inventory)
            == journal_binding.implementation_sha256
            and journal._binding is journal_binding
            and journal._inventory == journal_source_inventory
            and journal._callbacks[0] is journal_authenticator
            and driver._clock is monotonic
            and getattr(self._bridge, "_RunFeatureBridge__clock", None) is monotonic,
            "handoff run/provider/oracle/source/clock binding differs",
        )
        self._identity = (
            driver,
            journal,
            self._bridge,
            journal_binding,
            journal_authenticator,
            monotonic,
        )
        self._binding_bytes = jr.canonical(journal_binding.document())
        self._driver_bindings = driver._bindings()
        self._driver_objects = driver._objects()
        self._driver_models = driver._models()
        self._journal_callbacks = tuple(
            (row, row.target, row.source_sha256) for row in journal._callbacks
        )
        self._root_identity = (self._root.stat().st_dev, self._root.stat().st_ino)
        self._journal_identity = (
            journal.root,
            journal._parent,
            journal._repository,
            journal._root_fd,
            journal._parent_fd,
            journal._pending_fd,
            journal._lock_fd,
            journal._root_identity,
            journal._pending_identity,
        )
        self._sources = {
            Path(path): jr.digest(Path(path).read_bytes())
            for path in (
                __file__,
                records.__file__,
                native.__file__,
                journal_module.__file__,
                jr.__file__,
                journal_checker.__file__,
                sequential_v2_seals.__file__,
                feature_views.__file__,
                feature_records.__file__,
                feature_checker.__file__,
            )
        }
        self._functions_saved = self._functions()
        self._epoch, self._original_deadline = binding.original_epoch, binding.original_deadline
        self._deadline = min(self._original_deadline, self._epoch + 7200)
        self._last_time, self._busy, self._stopped = self._epoch, False, False
        self._next_round, self._last_result, self._last_failure = 1, None, None
        self._requests, self._phases, self._phase_snapshots = {}, {}, {}
        self._last_checkpoint, self._last_events = initial_checkpoint, ()
        self._prior_seats, self._previous_completion = (), None
        self._prepared = self._seats = self._callback = None
        self._native_saved = self._callback_saved = self._selected_saved = None
        self._numerical_saved = self._returned_report = self._feature_saved = None
        self._arguments = self._fingerprint_saved = None
        self._previous = initial_checkpoint
        self._observed_checkpoint = initial_checkpoint
        self._intent = self._completion = self._planned = None
        self._stage, self._wave_clock = "constructor", None
        self._driver_saved, self._journal_saved = self._driver_state(), self._journal_state()
        self._busy = driver._busy = journal._busy = True
        self._control_saved = self._control()
        try:
            self._checkpoint("constructor_admission")
            report = self._report(initial_checkpoint)
            require(
                report.history is not None
                and report.history.complete
                and report.history.round_index == 1
                and len(report.history.observations) == 64
                and not report.outstanding,
                "handoff initial reconstruction differs",
            )
            self._last_events = report.event_sha256s
            self._control_saved = self._control()
            self._checkpoint("constructor_complete")
        finally:
            self._busy = driver._busy = journal._busy = False
            self._control_saved = self._control()

    @property
    def next_round_index(self):
        return self._next_round

    @property
    def stopped(self):
        return self._stopped

    @property
    def last_result(self):
        return self._last_result

    @property
    def last_failure(self):
        return self._last_failure

    @staticmethod
    def _functions():
        return (
            NativeMP2DDriver.select_wave,
            NativeMP2DDriver._guard_state,
            NativeMP2DDriver._guard,
            NativeMP2DDriver._validate_wave,
            NativeMP2DDriver._models,
            NativeMP2DDriver._objects,
            DurableDispatchJournal.seal_wave,
            DurableDispatchJournal._guard,
            DurableDispatchJournal._capture,
            JournalHistoryCallback.__call__,
            verify_dispatch_journal,
            journal_checker.verify_dispatch_journal,
            verify_phase,
            NativeFeaturePosterior._guard,
            records.publish_intent,
            records.publish_completion,
            records.readback_phase,
            records.readback_phases,
            records._read_phase,
            records._admit_tree,
            records.quota_bytes,
            records.native_canonical,
            records.planned_wave,
            records.inventory,
            records.encode_document,
            NativeMP2DJournalHandoff._numerical_state,
            NativeMP2DDriver.prepare_wave,
            native.fit_charged_learner,
            native.NativeFeaturePosterior,
            native.run_native_mp2d_equal_ten,
            records.PhaseBuilder.__enter__,
            records.PhaseBuilder.write_bytes,
            records.PhaseBuilder.publish,
            records.verify_phase,
            records.inspect_feature_tree,
        )

    def _ownership(self):
        require(
            all(
                left is right
                for left, right in zip(self._functions_saved, self._functions(), strict=True)
            ),
            "handoff invoked delegate changed",
        )
        current = (
            self._driver,
            self._journal,
            self._bridge,
            self._binding,
            self._auth,
            self._clock,
        )
        require(
            all(left is right for left, right in zip(self._identity, current, strict=True))
            and self._driver._bridge is self._bridge
            and self._driver._clock is self._clock
            and self._journal._binding is self._binding
            and self._journal._callbacks[0] is self._auth
            and all(
                left is right
                for left, right in zip(
                    self._driver_objects, NativeMP2DDriver._objects(self._driver), strict=True
                )
            ),
            "handoff live ownership changed",
        )

    def _control(self):
        return (
            self._busy,
            self._stopped,
            self._next_round,
            self._deadline,
            self._wave_clock,
            self._last_time,
            self._arguments,
            self._fingerprint_saved,
            _checkpoint_state(self._previous),
            _checkpoint_state(self._last_checkpoint),
            self._last_events,
            self._prior_seats,
            self._previous_completion,
            id(self._last_result),
            id(self._last_failure),
            self._native_saved,
            self._callback_saved,
            self._selected_saved,
            self._numerical_saved,
            self._feature_saved,
            id(self._prepared),
            id(self._seats),
            id(self._callback),
            None
            if self._returned_report is None
            else (id(self._returned_report[0]), self._returned_report[1]),
            tuple(sorted(self._sources.items())),
            self._root,
            self._run_root,
            tuple((path, snapshot) for path, snapshot in self._phase_snapshots.items()),
            self._driver_saved,
            self._journal_saved,
            self._root_identity,
            self._journal_identity,
            self._driver_bindings,
            self._driver_models,
            self._binding_bytes,
            self._epoch,
            self._original_deadline,
            _checkpoint_state(self._observed_checkpoint),
            None
            if self._planned is None
            else (id(self._planned), self._planned.payload, self._planned.sha256),
            _phase_state(self._intent),
            _phase_state(self._completion),
        )

    def _driver_state(self):
        driver = self._driver
        return (
            driver._next_round,
            driver._prior_seats,
            driver.stopped,
            id(driver._current),
            id(driver._history),
            driver._history_bytes,
            id(driver._wave),
            None if driver._wave is None else driver._seal(driver._wave),
            tuple(sorted(driver._waves.items())),
            driver._deadline,
            tuple(
                (key, _native_state(value)) for key, value in sorted(driver._preparations.items())
            ),
            tuple(
                (key, value[0], _native_state(value[1]))
                for key, value in sorted(driver._selections.items())
            ),
            tuple(sorted(driver._info.items())),
            tuple(
                (key, value[0], id(value[1]), _native_state(value[2]))
                for key, value in sorted(driver._requests.items())
            ),
            driver._prior_head,
            canonical(driver._last_rows),
            driver._old_query_ids,
            driver._old_eligible,
            id(driver._wave_identity),
            driver._wave_seal,
            driver._pins,
            None
            if driver._active is None
            else (driver._fingerprint(driver._active[0]), driver._active[1]),
            tuple((key, id(value)) for key, value in sorted(driver._numerical_work.items())),
        )

    def _selection_transition(self, before, seats, retained_bytes, last_failure):
        after = self._driver_state()
        require(
            all(
                after[index] == value
                for index, value in enumerate(before)
                if index not in (0, 1, 2, 11)
            )
            and len(after[11]) == len(before[11]) + 1
            and all(row in after[11] for row in before[11])
            and self._driver.retained_bytes == retained_bytes + len(seats.record_payload)
            and self._driver.last_failure
            is (last_failure if seats.status == "selected" else seats),
            "handoff native selection changed unrelated retained state",
        )
        require(
            self._driver.stopped is (seats.status != "selected"),
            "handoff native selection stop state differs",
        )
        return after

    def _journal_transition(self, before, requests, following, round_index):
        after = self._journal_state()
        expected_requests = (*requests, *self._binding.reserves[round_index - 1])
        require(
            after[0] == _checkpoint_state(following)
            and after[1] == round_index
            and after[3] == tuple(row.sha256 for row in expected_requests)
            and all(after[index] == before[index] for index in (2, 4, 5, 6, 7, 8, 9, 10, 11))
            and all(row in after[12] for row in before[12])
            and all(row in after[13] for row in before[13])
            and not after[14]
            and after[15] >= before[15],
            "handoff journal seal changed unrelated live state",
        )
        return after

    def _journal_state(self):
        journal = self._journal
        return (
            _checkpoint_state(journal.checkpoint),
            journal._wave_index,
            journal._terminal_count,
            tuple(row.sha256 for row in journal._requests),
            journal._closed,
            journal._poisoned,
            journal._stopped,
            jr.canonical({key: row.document() for key, row in journal._intents.items()}),
            tuple(sorted(journal._acks.items())),
            frozenset(journal._faults),
            frozenset(journal._terminal_intents),
            frozenset(journal._external_ids),
            tuple(sorted(journal._watch.items())),
            tuple(sorted(journal._content_watch.items())),
            journal._pending_content,
            journal._bytes,
        )

    def _callback_state(self):
        callback = self._callback
        if callback is None:
            return None
        require(
            type(callback) is JournalHistoryCallback,
            "handoff requires the actual journal history callback",
        )
        args = callback._arguments
        return (
            id(callback),
            callback._root,
            args["trusted_parent"],
            jr.canonical(args["expected_binding"].document()),
            _checkpoint_state(args["expected_checkpoint"]),
            args["expected_source_inventory"],
            id(args["authenticator"]),
            id(args["authenticator"].target),
            args["authenticator"].source_sha256,
            callback._round,
            callback._head,
            callback._provider,
            callback._binding,
            callback._checkpoint,
            id(callback._auth_target),
            callback._auth_source,
        )

    def _guard_state(self):
        self._ownership()
        require(self._control() == self._control_saved, "handoff clock/control state changed")
        require(
            not self._busy or (self._driver._busy is True and self._journal._busy is True),
            "handoff busy ownership was released during a callback",
        )
        records.directory(self._root)
        require(
            (self._root.stat().st_dev, self._root.stat().st_ino) == self._root_identity,
            "handoff root identity changed",
        )
        require(
            all(
                left is right
                for left, right in zip(self._functions_saved, self._functions(), strict=True)
            ),
            "handoff invoked delegate changed",
        )
        require(
            jr.canonical(self._binding.document()) == self._binding_bytes
            and self._journal._inventory == self._inventory
            and self._driver._bindings() == self._driver_bindings
            and all(
                left is right
                for left, right in zip(self._driver_objects, self._driver._objects(), strict=True)
            )
            and self._driver._models() == self._driver_models,
            "handoff source/model/run bindings changed",
        )
        require(
            (
                self._journal.root,
                self._journal._parent,
                self._journal._repository,
                self._journal._root_fd,
                self._journal._parent_fd,
                self._journal._pending_fd,
                self._journal._lock_fd,
                self._journal._root_identity,
                self._journal._pending_identity,
            )
            == self._journal_identity,
            "handoff journal descriptor ownership changed",
        )
        for index, (original, target, pin) in enumerate(self._journal_callbacks):
            callback = self._journal._callbacks[index]
            require(
                callback is original
                and callback.target is target
                and callback.source_sha256 == pin,
                "handoff journal callback binding changed",
            )
            require(
                type(getattr_static(target, "source_sha256", None)) is str
                and getattr_static(target, "source_sha256") == pin,
                "handoff requires ordinary unchanged source strings",
            )
        advertised = getattr_static(self._driver._feasibility, "source_sha256", None)
        require(
            type(advertised) is str and advertised == self._driver._pins[5],
            "handoff feasibility source changed or uses a descriptor",
        )
        require(
            getattr(self._bridge, "_RunFeatureBridge__clock", None) is self._clock
            and self._bridge.binding.original_epoch == self._epoch
            and self._bridge.binding.original_deadline == self._original_deadline,
            "handoff original feature clock changed",
        )
        require(
            all(jr.digest(path.read_bytes()) == digest for path, digest in self._sources.items()),
            "handoff loaded source bytes changed",
        )
        NativeMP2DDriver._guard_state(self._driver)
        require(
            self._driver_state() == self._driver_saved,
            "handoff native state changed outside selection",
        )
        require(
            self._journal_state() == self._journal_saved,
            "handoff journal state changed outside its seal",
        )
        require(
            self._callback_state() == self._callback_saved,
            "handoff original history callback changed",
        )
        if self._prepared is not None:
            require(
                _native_state(self._prepared) == self._native_saved, "handoff preparation changed"
            )
            require(
                self._numerical_state() == self._numerical_saved,
                "handoff actual fitted/posterior binding changed",
            )
            require(
                self._feature_state() == self._feature_saved, "handoff paid feature state changed"
            )
        if self._seats is not None:
            require(
                _native_state(self._seats) == self._selected_saved, "handoff returned seats changed"
            )
        require(
            tuple(self._phases) == tuple(self._phase_snapshots),
            "handoff retained phase inventory changed",
        )
        for name, seal in self._phases.items():
            require(
                _phase_state(seal) == self._phase_snapshots[name], "handoff retained phase changed"
            )
        if self._returned_report is not None:
            report, saved = self._returned_report
            require(
                self._report_state(report) == saved,
                "handoff returned journal reconstruction changed",
            )

    def _guard(self):
        self._guard_state()
        NativeMP2DDriver._guard(self._driver)
        DurableDispatchJournal._guard(self._journal)
        self._guard_state()

    def _checkpoint(self, stage):
        self._stage = stage
        previous = self._last_time
        self._guard()
        before = time.monotonic()
        value = self._clock()
        self._guard()
        after = time.monotonic()
        require(
            self._stage == stage and self._last_time == previous,
            "handoff clock callback changed timing state",
        )
        finite_clock(value)
        require(
            before <= value <= after and value >= previous,
            "handoff clock is not the same absolute monotonic domain",
        )
        self._last_time = value
        self._control_saved = self._control()
        if after >= self._deadline:
            raise TimeoutError("handoff original scientific/wave deadline exhausted")
        return value

    @staticmethod
    def _report_state(report):
        require(type(report) is jr.JournalReport, "handoff checker result type differs")
        return jr.canonical(asdict(report))

    def _report(self, checkpoint):
        self._checkpoint("before_journal_reconstruction")
        report = verify_dispatch_journal(
            self._journal.root,
            trusted_parent=self._run_root,
            expected_binding=self._binding,
            expected_checkpoint=checkpoint,
            expected_source_inventory=self._inventory,
            authenticator=self._auth,
        )
        self._guard_state()
        self._returned_report = (report, self._report_state(report))
        self._control_saved = self._control()
        require(
            not report.stopped
            and not report.ambiguous_tail
            and not report.extension_requires_adoption
            and report.checkpoint == checkpoint
            and jr.canonical(report.binding.document()) == self._binding_bytes,
            "handoff reconstruction lacks an exact unstopped adopted checkpoint",
        )
        self._checkpoint("after_journal_reconstruction")
        return report

    def _numerical_state(self):
        driver = self._driver
        round_index = self._prepared.round_index
        learner = driver._numerical_work[(round_index, "learner_fit")]
        posterior = driver._numerical_work[(round_index, "posterior")]
        wave = driver._numerical_work[(round_index, "native_wave")]
        require(
            type(learner) is GaussianLearnerSnapshot
            and type(posterior) is NativeFeaturePosterior
            and wave is driver._wave,
            "handoff requires actual retained fit/posterior/native wave",
        )
        NativeFeaturePosterior._guard(posterior, external_source=False)
        return (
            id(learner),
            learner.numerical_sha256,
            learner.history_sha256,
            learner.transform.sha256,
            learner.fit_query_ids,
            learner.fit_sequence_ids,
            learner.feature_receipt_sha256,
            id(posterior),
            canonical(asdict(posterior.binding)),
            id(wave),
            driver._seal(wave),
        )

    def _feature_state(self):
        return canonical(
            (
                self._bridge.counters.document(),
                self._bridge._RunFeatureBridge__accepted_head,
                self._bridge.evidence_head,
            )
        )

    def _admit_preparation(self, preparation, checkpoint, head, started):
        driver = self._driver
        document = records.native_document(preparation, NativeMP2DPreparation)
        require(
            preparation.status == "prepared"
            and preparation.round_index == self._next_round
            and driver._current is preparation
            and driver._next_round == self._next_round
            and driver._prior_seats == self._prior_seats
            and not driver.stopped
            and not driver._busy,
            "handoff lacks its current complete native preparation",
        )
        key = (preparation.round_index, preparation.poll_ordinal)
        saved, callback, actual = driver._requests[key]
        require(
            type(saved) is str
            and jr.pin(saved)
            and actual is preparation
            and driver._preparations[key] is preparation
            and preparation.sha256 not in driver._selections,
            "handoff preparation was replaced or previously selected",
        )
        require(
            type(callback) is JournalHistoryCallback,
            "handoff requires an actual retained journal callback",
        )
        args = callback._arguments
        require(
            callback._root == self._journal.root
            and args["trusted_parent"] == self._run_root
            and args["expected_binding"] is self._binding
            and args["expected_checkpoint"] == checkpoint
            and args["expected_source_inventory"] == self._inventory
            and args["authenticator"] is self._auth
            and callback._round == preparation.round_index
            and callback._head == head
            and callback._provider == self._binding.provider_sha256
            and callback._binding == self._binding_bytes
            and callback._checkpoint == jr.canonical(asdict(checkpoint))
            and callback._auth_target is self._auth.target
            and callback._auth_source == self._auth.source_sha256,
            "handoff original journal callback expectations differ",
        )
        history = driver._history
        require(
            history is not None
            and history.complete
            and history.round_index == preparation.round_index
            and history.run_id == self._binding.run_id
            and history.seed == self._binding.seed
            and history.objective_context_sha256 == self._binding.objective_context_sha256
            and history.oracle_bundle_sha256 == self._binding.oracle_bundle_sha256
            and history.previous_wave_head_sha256 == head
            and history.receipt_sha256 == checkpoint.head_sha256
            and len(history.observations)
            == checkpoint.charged_count
            == 64 + 16 * (preparation.round_index - 1)
            and canonical(asdict(history)) == driver._history_bytes
            and preparation.history_sha256 == history.sha256,
            "handoff preparation history/head differs from adopted journal",
        )
        require(
            document["kind"] == "preparation"
            and document["status"] == "prepared"
            and document["round_index"] == preparation.round_index
            and document["poll_ordinal"] == preparation.poll_ordinal
            and document["history_sha256"] == history.sha256
            and document["history_receipt_sha256"] == checkpoint.head_sha256
            and document["previous_wave_head_sha256"] == head
            and document["charged_count"] == checkpoint.charged_count
            and document["original_epoch"] == self._epoch
            and document["original_deadline"] == self._original_deadline
            and document["original_wave_started_at"] == started
            and document["effective_deadline"] == self._deadline
            and driver._deadline == self._deadline
            and driver._waves[preparation.round_index]
            == (started, head, preparation.poll_ordinal, self._deadline),
            "handoff original native round/clock document differs",
        )
        wave = driver._wave
        require(
            type(wave.record_json) is str
            and len(wave.record_json) <= records.MAX_NATIVE_BYTES
            and wave.record_json.isascii(),
            "handoff raw-wave admission differs",
        )
        raw_wave = wave.record_json.encode("ascii")
        require(
            len(raw_wave) <= records.MAX_NATIVE_BYTES
            and jr.digest(raw_wave) == preparation.wave_sha256 == wave.sha256,
            "handoff raw-wave byte/hash differs",
        )
        learner = driver._numerical_work[(preparation.round_index, "learner_fit")]
        posterior = driver._numerical_work[(preparation.round_index, "posterior")]
        require(
            document["wave"] == json.loads(canonical(asdict(wave)))
            and document["wave_sha256"] == wave.sha256
            and wave.stop_reason == native.COMPLETE
            and document["learner"]
            == {
                "numerical_sha256": learner.numerical_sha256,
                "feature_receipt_sha256": learner.feature_receipt_sha256,
                "fit_query_ids": list(learner.fit_query_ids),
                "fit_sequence_ids": list(learner.fit_sequence_ids),
            },
            "handoff raw native result/fit fields differ",
        )
        NativeMP2DDriver._validate_wave(
            driver, wave, posterior.binding, frozenset(learner.fit_query_ids)
        )
        self._prepared, self._callback = preparation, callback
        self._native_saved, self._callback_saved = (
            _native_state(preparation),
            self._callback_state(),
        )
        self._numerical_saved, self._feature_saved = self._numerical_state(), self._feature_state()
        self._driver_saved, self._journal_saved = self._driver_state(), self._journal_state()

    def _seats_returned(self, seats, allowed, filter_sha256):
        document = records.native_document(seats, NativeMP2DSeats)
        eligible = tuple(
            sequence
            for sequence in self._driver._wave.candidates
            if native.sequence_id(sequence) in allowed
        )
        expected = eligible[:14] if len(eligible) >= 14 else ()
        require(
            seats.round_index == self._prepared.round_index
            and seats.preparation_sha256 == self._prepared.sha256
            and document["kind"] == "seats"
            and document["status"] == seats.status
            and document["preparation_sha256"] == self._prepared.sha256,
            "handoff returned native seat record differs",
        )
        if seats.status in ("selected", "underfilled"):
            require(
                seats.method_sequences == expected
                and seats.status == ("selected" if expected else "underfilled")
                and document["method_sequences"] == list(expected)
                and document["allowed_sequence_ids"] == sorted(allowed)
                and document["external_filter_sha256"] == filter_sha256
                and document["eligible_pool_count"] == len(eligible)
                and document["history_sha256"] == self._prepared.history_sha256
                and document["wave_sha256"] == self._prepared.wave_sha256
                and document["original_wave_started_at"] == self._wave_clock[1]
                and document["effective_deadline"] == self._deadline,
                "handoff selection changed frozen order/filter/history/clock",
            )
            require(
                self._driver._next_round == self._prepared.round_index + bool(expected)
                and self._driver._prior_seats == (expected if expected else self._prior_seats),
                "handoff native selection state advanced inconsistently",
            )
        else:
            require(
                seats.status == "stopped_failure"
                and not seats.method_sequences
                and self._driver.stopped,
                "handoff native stop schema differs",
            )
        saved, actual = self._driver._selections[self._prepared.sha256]
        expected_fingerprint = jr.digest(
            canonical((self._prepared.sha256, sorted(allowed), filter_sha256))
        )
        require(
            saved == expected_fingerprint and actual is seats,
            "handoff actual selection cache differs",
        )
        self._seats, self._selected_saved = seats, _native_state(seats)

    def _clock_document(self):
        return {
            "original_epoch": self._epoch,
            "original_deadline": self._original_deadline,
            "original_wave_started_at": self._wave_clock[1],
            "effective_deadline": self._deadline,
            "last_checked_monotonic": self._last_time,
            "final_readback_and_closing_clock_not_in_payload": True,
            "timing_authenticated": False,
        }

    def _remember_phase(self, path, seal):
        self._phases[path] = seal
        self._phase_snapshots[path] = _phase_state(seal)
        self._control_saved = self._control()

    def _read_event(self):
        before = self._previous
        path = self._journal.root / jr.event_name(before.event_count)
        result = verify_phase(
            path,
            expected_artifact=jr.EVENT_ARTIFACT,
            expected_payload_paths=(jr.EVENT_PAYLOAD,),
            expected_predecessor_seals={"previous_event": before.head_sha256},
        )
        require(
            result.read_payload_bytes(jr.EVENT_PAYLOAD) == self._planned.payload
            and json.loads(result.metadata_json)
            == {
                "genesis_sha256": self._binding.sha256,
                "event_sha256": self._planned.sha256,
                "ordinal": before.event_count,
            },
            "handoff actual wave-event bytes/publication metadata differ",
        )
        return result

    def _final_readback(self, expected_event_seal):
        self._checkpoint("before_final_readback")
        records.readback_phases(
            tuple(self._phases.items()),
            handoff_root=self._root,
            run_root=self._run_root,
            deadline=self._deadline,
        )
        event_seal = self._read_event()
        require(
            event_seal == expected_event_seal, "handoff journal event changed after final callback"
        )
        DurableDispatchJournal._capture(self._journal)
        self._guard_state()
        if time.monotonic() >= self._deadline:
            raise TimeoutError("handoff original deadline expired during final readback")

    def _failure(self, error, safe, *, status="stopped_failure"):
        self._stopped = True

        def bounded(value, limit):
            return str(value)[:limit].encode("ascii", "backslashreplace")[:limit].decode("ascii")

        # Only locally captured, admitted values: a failing callback may have
        # corrupted the live journal checkpoint or returned native records.
        previous = jr.JournalCheckpoint(*safe["previous"])
        observed = jr.JournalCheckpoint(*safe["observed"])
        intent = None if safe["intent"] is None else PhaseSeal(*safe["intent"])
        completion = None if safe["completion"] is None else PhaseSeal(*safe["completion"])
        payload = records.encode_document(
            {
                "artifact": "native_mp2d_journal_handoff_failure_v1",
                "schema_version": 1,
                "status": status,
                "round_index": safe["round"],
                "preparation_sha256": safe["preparation"],
                "seats_sha256": safe["seats"],
                "stage": bounded(self._stage, 80),
                "error_type": bounded(type(error).__name__, 80),
                "error": bounded(error, 256),
                "last_checked_monotonic": safe["last_time"],
                "effective_deadline": safe["deadline"],
                "previous_checkpoint": asdict(previous),
                "observed_checkpoint": asdict(observed),
                "intent_seal_sha256": None if intent is None else intent.seal_sha256,
                "completion_seal_sha256": None if completion is None else completion.seal_sha256,
                "attempted_publication_path_ascii_prefixes": safe["paths"],
                "known_publication_notes": [
                    bounded(note, 128) for note in getattr(error, "__notes__", ())[:2]
                ],
                "journal_rollback_claimed": False,
                "dispatch_authorized": False,
                **records.FLAGS,
            }
        )
        require(
            len(payload) <= records.FAILURE_RESERVE_BYTES,
            "handoff compact failure reserve exceeded",
        )
        result = records.MP2DJournalHandoffResult(
            safe["round"],
            status,
            safe["preparation"],
            safe["seats"],
            previous,
            observed,
            intent,
            completion,
            payload,
        )
        self._last_result = self._last_failure = result
        return result

    def seal_wave(
        self,
        preparation,
        *,
        expected_checkpoint,
        expected_wave_head_sha256,
        original_wave_started_at,
        allowed_sequence_ids,
        external_filter_sha256,
        method_query_ids,
        replicate_ids,
    ):
        require(
            type(preparation) is NativeMP2DPreparation, "handoff preparation exact type differs"
        )
        require(
            type(preparation.record_payload) is bytes
            and len(preparation.record_payload) <= records.MAX_NATIVE_BYTES,
            "handoff preparation admission exceeded",
        )
        require(
            type(expected_checkpoint) is jr.JournalCheckpoint, "handoff checkpoint type differs"
        )
        expected_checkpoint.__post_init__()
        finite_clock(original_wave_started_at)
        require(
            type(preparation.round_index) is int
            and 1 <= preparation.round_index <= 28
            and jr.pin(expected_wave_head_sha256)
            and jr.pin(external_filter_sha256)
            and type(allowed_sequence_ids) is frozenset
            and all(jr.pin(value) for value in allowed_sequence_ids)
            and type(method_query_ids) is tuple
            and len(method_query_ids) == 14
            and len(set(method_query_ids)) == 14
            and all(jr.identifier(value) for value in method_query_ids)
            and type(replicate_ids) is tuple
            and len(replicate_ids) == 14
            and all(type(value) is int and value >= 0 for value in replicate_ids),
            "handoff ordinary request shape differs",
        )
        require(
            preparation.status == "prepared",
            "handoff cannot seal an incomplete/stopped native preparation",
        )
        arguments = (
            (
                id(preparation),
                preparation.round_index,
                preparation.poll_ordinal,
                preparation.status,
                preparation.history_sha256,
                preparation.wave_sha256,
                preparation.sha256,
            ),
            _checkpoint_state(expected_checkpoint),
            expected_wave_head_sha256,
            original_wave_started_at,
            tuple(sorted(allowed_sequence_ids)),
            external_filter_sha256,
            method_query_ids,
            replicate_ids,
        )
        fingerprint = jr.digest(canonical(arguments))
        self._ownership()
        if preparation.round_index in self._requests:
            saved, result, state = self._requests[preparation.round_index]
            require(
                saved == fingerprint and _result_state(result) == state,
                "handoff changed historical duplicate",
            )
            return result
        require(
            not self._stopped
            and not self._busy
            and not self._journal._busy
            and not self._driver._busy,
            "handoff is stopped or busy",
        )
        require(
            preparation.round_index == self._next_round, "handoff round skipped or already consumed"
        )
        require(
            self._journal.checkpoint == expected_checkpoint,
            "handoff expected checkpoint is not the live adopted tip",
        )
        safe = {
            "round": preparation.round_index,
            "preparation": preparation.sha256,
            "seats": None,
            "previous": _checkpoint_state(expected_checkpoint),
            "observed": _checkpoint_state(expected_checkpoint),
            "intent": None,
            "completion": None,
            "last_time": self._last_time,
            "paths": [],
            "deadline": min(
                self._original_deadline, self._epoch + 7200, original_wave_started_at + 120
            ),
        }
        self._previous, self._observed_checkpoint = expected_checkpoint, expected_checkpoint
        self._arguments, self._fingerprint_saved = arguments, fingerprint
        self._wave_clock = (preparation.round_index, original_wave_started_at)
        self._deadline = safe["deadline"]
        self._seats = self._selected_saved = self._intent = self._completion = self._planned = None
        self._returned_report = None
        self._prepared = preparation
        self._busy = True
        try:
            self._admit_preparation(
                preparation,
                expected_checkpoint,
                expected_wave_head_sha256,
                original_wave_started_at,
            )
            self._driver._busy = self._journal._busy = True
            self._control_saved = self._control()
            now = self._checkpoint("seal_admission")
            safe["last_time"] = now
            require(
                self._epoch <= original_wave_started_at <= now,
                "handoff original wave epoch differs",
            )
            report = self._report(expected_checkpoint)
            require(
                report.history is not None
                and report.history.complete
                and canonical(asdict(report.history)) == self._driver._history_bytes
                and not report.outstanding
                and report.event_sha256s[: len(self._last_events)] == self._last_events,
                "handoff reconstructed predecessor history/prefix differs",
            )
            require(
                allowed_sequence_ids
                <= frozenset(map(native.sequence_id, self._driver._wave.candidates)),
                "handoff filter includes another wave's candidates",
            )
            self._checkpoint("before_native_selection")
            before_driver = self._driver_saved
            retained_bytes, last_failure = self._driver.retained_bytes, self._driver.last_failure
            self._driver._busy = False
            try:
                seats = self._driver.select_wave(
                    preparation_sha256=preparation.sha256,
                    allowed_sequence_ids=allowed_sequence_ids,
                    external_filter_sha256=external_filter_sha256,
                )
            finally:
                self._driver._busy = True
            self._ownership()
            require(
                self._control() == self._control_saved,
                "handoff control changed during native selection",
            )
            records.native_document(seats, NativeMP2DSeats)
            safe["seats"] = seats.sha256
            next_driver = self._selection_transition(
                before_driver, seats, retained_bytes, last_failure
            )
            self._seats_returned(seats, allowed_sequence_ids, external_filter_sha256)
            self._driver_saved = next_driver
            self._control_saved = self._control()
            if seats.status != "selected":
                result = self._failure(
                    ValueError("native selection returned no submittable wave"),
                    safe,
                    status=seats.status,
                )
            else:
                self._checkpoint("after_native_selection")
                safe["last_time"] = self._last_time
                requests, event, following = records.planned_wave(
                    self._binding,
                    expected_checkpoint,
                    preparation.round_index,
                    seats.method_sequences,
                    method_query_ids,
                    replicate_ids,
                )
                self._planned = event
                self._control_saved = self._control()
                forbidden = (
                    *self._binding.initial_requests,
                    *(row for pair in self._binding.reserves for row in pair),
                )
                require(
                    not set(method_query_ids)
                    & {row.query_id for row in (*forbidden, *report.observations)}
                    and not set(seats.method_sequences)
                    & {row.sequence for row in (*forbidden, *report.observations)},
                    "handoff selected requests collide with charged/private inventory",
                )
                intent_path = self._root / f"wave-{preparation.round_index:02d}-intent"
                completion_path = self._root / f"wave-{preparation.round_index:02d}-complete"
                predecessors = {"journal_previous_event": expected_checkpoint.head_sha256}
                if self._previous_completion is not None:
                    predecessors["previous_handoff_completion"] = self._previous_completion
                document = {
                    "artifact": records.INTENT_ARTIFACT,
                    "schema_version": 1,
                    "status": "intent_not_dispatch_authority",
                    "round_index": preparation.round_index,
                    "run_id": self._binding.run_id,
                    "arm_id": self._binding.arm_id,
                    "seed": self._binding.seed,
                    "journal_genesis_sha256": self._binding.sha256,
                    "preparation_sha256": preparation.sha256,
                    "seats_sha256": seats.sha256,
                    "preparation_payload_sha256": jr.digest(preparation.record_payload),
                    "seats_payload_sha256": jr.digest(seats.record_payload),
                    "wave_sha256": preparation.wave_sha256,
                    "history_sha256": preparation.history_sha256,
                    "previous_checkpoint": asdict(expected_checkpoint),
                    "previous_wave_head_sha256": expected_wave_head_sha256,
                    "planned_event_sha256": event.sha256,
                    "method_requests": [row.document() for row in requests],
                    "external_filter_sha256": external_filter_sha256,
                    "allowed_sequence_ids": sorted(allowed_sequence_ids),
                    "handoff_sources": {
                        str(path): digest for path, digest in self._sources.items()
                    },
                    "journal_source_inventory": self._inventory,
                    "driver_binding_sha256": jr.digest(self._driver_bindings),
                    "dispatch_authorized": False,
                    **self._clock_document(),
                    **records.FLAGS,
                }
                safe["paths"].append(
                    str(intent_path)[:192].encode("ascii", "backslashreplace")[:192].decode("ascii")
                )
                intent = records.publish_intent(
                    intent_path,
                    preparation=preparation,
                    seats=seats,
                    planned_event=event,
                    document=document,
                    predecessors=predecessors,
                    handoff_root=self._root,
                    run_root=self._run_root,
                    checkpoint=self._checkpoint,
                    deadline=self._deadline,
                )
                self._guard_state()
                safe["intent"] = _phase_state(intent)
                self._intent = intent
                self._remember_phase(intent_path, self._intent)
                self._checkpoint("before_journal_seal")
                safe["last_time"] = self._last_time
                before_journal = self._journal_saved
                safe["paths"].append(
                    str(self._journal.root / jr.event_name(expected_checkpoint.event_count))[:192]
                    .encode("ascii", "backslashreplace")[:192]
                    .decode("ascii")
                )
                self._journal._busy = False
                try:
                    observed = self._journal.seal_wave(requests)
                finally:
                    self._journal._busy = True
                self._ownership()
                require(
                    self._control() == self._control_saved,
                    "handoff control changed during journal seal",
                )
                require(
                    type(observed) is jr.JournalCheckpoint and observed == following,
                    "handoff journal returned another event/state",
                )
                next_journal = self._journal_transition(
                    before_journal, requests, following, preparation.round_index
                )
                safe["observed"] = _checkpoint_state(observed)
                self._observed_checkpoint, self._journal_saved = observed, next_journal
                self._control_saved = self._control()
                event_seal = self._read_event()
                self._checkpoint("after_journal_seal")
                after = self._report(following)
                require(
                    after.history is not None
                    and not after.history.complete
                    and after.history.round_index == preparation.round_index + 1
                    and after.history.observations == report.history.observations
                    and after.history.previous_wave_head_sha256 == event.sha256
                    and after.history.receipt_sha256 == event.sha256
                    and not after.outstanding
                    and after.event_sha256s == (*report.event_sha256s, event.sha256),
                    "handoff post-seal reconstruction is not the unchanged-charge next incomplete history",
                )
                completion = {
                    "artifact": records.COMPLETION_ARTIFACT,
                    "schema_version": 1,
                    "status": "sealed_wave",
                    "round_index": preparation.round_index,
                    "preparation_sha256": preparation.sha256,
                    "seats_sha256": seats.sha256,
                    "intent_seal_sha256": self._intent.seal_sha256,
                    "previous_checkpoint": asdict(expected_checkpoint),
                    "checkpoint": asdict(following),
                    "journal_event_sha256": event.sha256,
                    "journal_phase_sha256": event_seal.seal_sha256,
                    "post_seal_history_sha256": after.history.sha256,
                    "post_seal_round_index": after.history.round_index,
                    "post_seal_history_complete": False,
                    "method_sequences": seats.method_sequences,
                    "method_count": 14,
                    "reserve_count": 2,
                    "dispatch_authorized": False,
                    "checkpoint_requires_external_adoption": True,
                    **self._clock_document(),
                    **records.FLAGS,
                }
                safe["paths"].append(
                    str(completion_path)[:192]
                    .encode("ascii", "backslashreplace")[:192]
                    .decode("ascii")
                )
                completion_seal = records.publish_completion(
                    completion_path,
                    document=completion,
                    intent_seal_sha256=self._intent.seal_sha256,
                    journal_phase_sha256=event_seal.seal_sha256,
                    handoff_root=self._root,
                    run_root=self._run_root,
                    checkpoint=self._checkpoint,
                    deadline=self._deadline,
                )
                self._guard_state()
                safe["completion"] = _phase_state(completion_seal)
                self._completion = completion_seal
                self._remember_phase(completion_path, self._completion)
                result = records.MP2DJournalHandoffResult(
                    preparation.round_index,
                    "sealed_wave",
                    preparation.sha256,
                    seats.sha256,
                    expected_checkpoint,
                    following,
                    self._intent,
                    self._completion,
                    self._completion.read_payload_bytes("completion.json"),
                )
                result_state = _result_state(result)
                self._final_readback(event_seal)
                require(
                    _result_state(result) == result_state,
                    "handoff result changed during final readback",
                )
                self._last_result, self._last_checkpoint, self._last_events = (
                    result,
                    following,
                    after.event_sha256s,
                )
                self._previous_completion, self._prior_seats = (
                    self._completion.seal_sha256,
                    seats.method_sequences,
                )
                self._next_round = preparation.round_index + 1
        except BaseException as error:
            result = self._failure(error, safe)
            if not isinstance(error, Exception):
                self._requests[safe["round"]] = (fingerprint, result, _result_state(result))
                raise
        finally:
            self._busy = self._driver._busy = self._journal._busy = False
            if not self._stopped:
                self._control_saved = self._control()
        self._requests[safe["round"]] = (fingerprint, result, _result_state(result))
        return result

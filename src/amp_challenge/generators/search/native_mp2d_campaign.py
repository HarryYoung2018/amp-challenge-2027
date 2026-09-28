"""Actual persistent MP2D campaign with external authorities and paid final fitting.

Same-live-object driver/journal compatibility dependencies are deliberate. This
in-process composition is not a privacy boundary or scientific data admission.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from amp_challenge.generators.diffusion.native_baseline_operators import (
    NormalizedObjectiveContext,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_mp2d_driver import MP2D_ARM, NativeMP2DDriver
from amp_challenge.generators.search import durable_dispatch_journal_records as j
from amp_challenge.generators.search.durable_dispatch_journal import DurableDispatchJournal
from amp_challenge.generators.search.durable_dispatch_journal_verify import (
    JournalHistoryCallback,
    verify_dispatch_journal,
)
from amp_challenge.generators.search.native_mp2d_journal_handoff import NativeMP2DJournalHandoff
from amp_challenge.generators.search.native_static_campaign import (
    ClockBoundDispatch,
    _load_audited_units,
)
from amp_challenge.generators.search.paid_terminal import (
    CONTRACT as PAID_TERMINAL_CONTRACT,
)
from amp_challenge.generators.search.paid_terminal import (
    LEARNER_SOURCE,
    TIMING_AMENDMENT_SHA256,
    complete_paid_terminal,
    terminal_timing_summary,
    timing_amendment_sha256,
)
from amp_challenge.models.charged_probability_learner import GeneratorFeatureTransform
from amp_challenge.representations.candidate_features import _exclusive_write
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    FeatureAssemblyBinding,
    FeatureIntent,
    PrivateFeatureRelease,
    finite_clock,
)
from amp_challenge.representations.run_feature_cache_records import canonical as feature_canonical

CONTRACT_DOCUMENT = {
    "contract": "native_mp2d_campaign_v2_20260914",
    "arm": MP2D_ARM,
    "logical_queries": 512,
    "common_initial": 64,
    "waves": 28,
    "method_seats": 14,
    "reserve_seats": 2,
    "selection": "native_mp2d_equal_ten_v2_and_existing_journal_handoff",
    "wave_seconds": 120,
    "maximum_seconds": 7200,
    "terminal": PAID_TERMINAL_CONTRACT,
    "terminal_timing_amendment_sha256": TIMING_AMENDMENT_SHA256,
    "scientific_clock": "before_provisioning_through_terminal_worker_close_and_publication",
    "restart": "exclusive_execution_root_fail_stop_no_automatic_redispatch",
    "historical_v2_terminal_timing_compliant": False,
}


def contract_document(arm_id):
    timing_amendment_sha256()
    j.require(arm_id == MP2D_ARM, "exact MP2D arm required")
    return dict(CONTRACT_DOCUMENT)


def timing_context(epoch, deadline, clock_epoch_id, *, arm_id=MP2D_ARM):
    epoch, deadline = finite_clock(epoch), finite_clock(deadline)
    j.require(
        0 < deadline - epoch <= 7200 and j.identifier(clock_epoch_id), "original MP2D clock differs"
    )
    return {
        "contract_sha256": j.digest(j.canonical(contract_document(arm_id))),
        "original_epoch": epoch,
        "original_deadline": deadline,
        "clock_epoch_id": clock_epoch_id,
    }


@dataclass(frozen=True, slots=True)
class MP2DCampaignInputs:
    initializations: tuple
    corpora: tuple
    context: NormalizedObjectiveContext
    bridge: RunFeatureBridge
    transform: GeneratorFeatureTransform
    binding: j.JournalBinding
    source_inventory: tuple
    initial_source_receipt: bytes
    initial_copy_receipt: bytes
    authenticator: j.CallbackPin
    transport: j.CallbackPin
    permission: j.CallbackPin
    collector: j.CallbackPin
    eligibility: j.CallbackPin
    feasibility: j.CallbackPin
    attestor: j.CallbackPin
    release: j.CallbackPin


def load_mp2d_initializations(*, checkpoints, audit, device, check):
    return _load_audited_units(checkpoints=checkpoints, audit=audit, device=device, check=check)


def _attested_statement(inputs, kind, expected, *, root, invoke):
    """Route a terminal statement through the original guarded callback boundary."""
    payload = j.canonical(expected)
    receipt = invoke(
        inputs.attestor,
        kind,
        kind,
        payload,
        retain=lambda value: _exclusive_write(root / (kind + ".receipt"), j.receipt_bytes(value)),
    )
    semantic = invoke(inputs.authenticator, "authenticate_" + kind, kind, receipt, payload)
    j.require(semantic == payload, "external timing attestation differs from expected statement")
    return j.digest(receipt)


def execute_mp2d_campaign(
    inputs,
    *,
    output_root,
    original_epoch,
    original_deadline,
    clock_epoch_id,
    monotonic=time.monotonic,
):
    """Run 64 imported + 448 charged queries and the actual final-history fit.

    output_root must exist, privately owned by the provisioning entry point.
    The exclusive execution child prevents repeats, including after failures.
    Callbacks are trusted code, not a sandbox; source pins detect accidental
    drift and do not attest evaluator quality or externally observed timing.
    """
    j.require(type(inputs) is MP2DCampaignInputs, "exact MP2D campaign inputs required")
    root = Path(output_root) / "execution"
    j.require(root.is_absolute(), "absolute controller output root required")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    journal = None
    phase, last = "admission", finite_clock(original_epoch)
    binding, bridge, transform = inputs.binding, inputs.bridge, inputs.transform
    contract = contract_document(binding.arm_id)
    timing = timing_context(
        original_epoch, original_deadline, clock_epoch_id, arm_id=binding.arm_id
    )
    contract_sha = timing["contract_sha256"]
    journal_parent = Path(bridge.binding.run_root)
    j.require(
        journal_parent.parent == Path(output_root) and journal_parent != root,
        "MP2D features must use the fixed run-owned directory",
    )
    callbacks = (
        inputs.authenticator,
        inputs.transport,
        inputs.permission,
        inputs.collector,
        inputs.eligibility,
        inputs.feasibility,
        inputs.attestor,
        inputs.release,
    )
    source_path = Path(__file__).resolve()
    source_sha = file_digest(source_path)
    frozen = None

    def save(name, value):
        payload = j.canonical(value)
        _exclusive_write(root / name, payload)
        return j.digest(payload)

    def check(name):
        nonlocal phase, last
        phase = name
        now = finite_clock(monotonic())
        j.require(now >= last, "original monotonic clock moved backwards")
        if now >= original_deadline:
            raise TimeoutError(f"MP2D campaign original deadline exceeded at {name}")
        last = now
        j.require(file_digest(source_path) == source_sha, "campaign implementation changed")
        for pin in callbacks:
            j.require(
                type(pin) is j.CallbackPin
                and getattr(pin.target, "source_sha256", None) == pin.source_sha256,
                "campaign callback source changed",
            )
        if frozen is not None:
            j.require(
                (binding.sha256, bridge.binding.sha256, transform.sha256) == frozen,
                "campaign binding/feature transform changed",
            )
        return now

    def invoke(pin, name, *args, retain=None):
        check("before_" + name)
        checkpoint = None if journal is None else journal.checkpoint
        if journal is not None:
            journal._guard()
        result = pin.target(*args)
        if retain is not None:
            retain(result)
        check("after_" + name)
        if journal is not None:
            journal._guard()
            j.require(journal.checkpoint == checkpoint, "external callback changed journal")
        return result

    def reconstruct():
        check("before_journal_reconstruction")
        report = verify_dispatch_journal(
            journal_parent / "journal",
            trusted_parent=journal_parent,
            expected_binding=binding,
            expected_checkpoint=journal.checkpoint,
            expected_source_inventory=inputs.source_inventory,
            authenticator=inputs.authenticator,
        )
        check("after_journal_reconstruction")
        j.require(
            not report.extension_requires_adoption
            and not report.ambiguous_tail
            and not report.stopped
            and not report.outstanding
            and report.history is not None
            and report.history.complete,
            "journal lacks a complete unambiguous adopted charged history",
        )
        return report

    def assembly_for(history):
        history_sha = history.sha256
        assembly = invoke(inputs.eligibility, "eligibility", history, transform)
        j.require(type(assembly) is FeatureAssemblyBinding, "exact eligibility binding required")
        assembly.__post_init__()
        j.require(
            history.sha256 == history_sha
            and assembly.history_sha256 == history_sha
            and assembly.eligibility_source_sha256 == inputs.eligibility.source_sha256
            and assembly.raw_history_payload == feature_canonical(asdict(history))
            and assembly.transform_sha256 == transform.sha256,
            "eligibility authority differs from verified history/transform",
        )
        successful = {row.query_id for row in history.observations if row.status == "successful"}
        j.require(
            set(assembly.eligible_query_ids) <= successful, "eligibility includes failed rows"
        )
        save(f"assembly-{history.round_index:02d}.json", assembly.document())
        return assembly

    def release_for(history, previous):
        release = invoke(inputs.release, "release_authority", history, previous)
        j.require(
            type(release) is PrivateFeatureRelease, "exact private-release authority required"
        )
        release.__post_init__()
        revealed = history.observations if history.round_index == 1 else history.observations[-2:]
        j.require(
            (
                release.run_id,
                release.history_sha256,
                release.objective_context_sha256,
                release.source_sha256,
                release.expected_previous_release_head,
                release.revealed_sequence_ids,
            )
            == (
                binding.run_id,
                history.sha256,
                binding.objective_context_sha256,
                inputs.release.source_sha256,
                previous,
                tuple(sequence_id(row.sequence) for row in revealed),
            ),
            "release authority does not match the exact charged prefix",
        )
        save(f"release-{history.round_index:02d}.json", release.document())
        return release

    def intent(history, purpose):
        return FeatureIntent(
            purpose,
            history.sha256,
            binding.objective_context_sha256,
            history.round_index,
            0,
            bridge.accepted_head,
            original_deadline,
        )

    try:
        check("admission")
        j.require(
            type(binding) is j.JournalBinding
            and type(bridge) is RunFeatureBridge
            and type(transform) is GeneratorFeatureTransform
            and type(inputs.context) is NormalizedObjectiveContext,
            "exact MP2D journal, feature and objective bindings required",
        )
        binding.__post_init__()
        j.require(
            binding.arm_id == MP2D_ARM
            and inputs.context.context_sha256 == binding.objective_context_sha256
            and binding.timing_context_sha256 == j.digest(j.canonical(timing)),
            "MP2D arm/prospective timing binding differs",
        )
        for pin, kind in ((inputs.permission, "permission"), (inputs.transport, "transport")):
            j.require(
                type(pin.target) is ClockBoundDispatch
                and pin.target.kind == kind
                and pin.target.clock is monotonic
                and pin.target.timing == timing,
                "permission and transport require the original clock boundary wrapper",
            )
        feature = bridge.binding
        j.require(
            (feature.run_id, feature.arm_id, feature.seed, feature.objective_context_sha256)
            == (binding.run_id, binding.arm_id, binding.seed, binding.objective_context_sha256)
            and (feature.original_epoch, feature.original_deadline)
            == (original_epoch, original_deadline)
            and bridge._RunFeatureBridge__clock is monotonic,
            "feature bridge must retain the original run and clock",
        )
        frozen = (binding.sha256, feature.sha256, transform.sha256)
        save(
            "started.json",
            {
                **timing,
                "contract": contract,
                "journal_binding": binding.document(),
                "feature_binding_sha256": feature.sha256,
                "transform_sha256": transform.sha256,
                "implementation_sha256": source_sha,
                "callback_sources": [pin.source_sha256 for pin in callbacks],
                "ten_checkpoint_mixture": len(inputs.initializations) == 10,
                "scientific_evidence_accepted": False,
            },
        )
        journal = DurableDispatchJournal.create(
            journal_parent / "journal",
            trusted_parent=journal_parent,
            binding=binding,
            expected_source_inventory=inputs.source_inventory,
            authenticator=inputs.authenticator,
            transport=inputs.transport,
            permission=inputs.permission,
            repository=Path(feature.repository),
        )
        check("before_import_initial")
        journal.import_initial(inputs.initial_source_receipt, inputs.initial_copy_receipt)
        report = reconstruct()
        history = report.history
        j.require(
            history.round_index == 1 and len(history.observations) == 64, "initial count differs"
        )
        shared = (*binding.initial_requests, *(r for pair in binding.reserves for r in pair))
        bridge.preload_private(
            tuple(r.sequence for r in shared), intent(history, "private_preload")
        )
        check("after_private_preload")
        driver = NativeMP2DDriver(
            inputs.initializations,
            inputs.corpora,
            inputs.context,
            bridge,
            transform,
            history_provider_sha256=binding.provider_sha256,
            oracle_bundle_sha256=binding.oracle_bundle_sha256,
            learner_source_sha256=file_digest(Path(feature.repository) / LEARNER_SOURCE),
            eligibility_source_sha256=inputs.eligibility.source_sha256,
            release_source_sha256=inputs.release.source_sha256,
            feasibility=inputs.feasibility.target,
            feasibility_source_sha256=inputs.feasibility.source_sha256,
            monotonic=monotonic,
        )
        handoff_root = journal_parent / "handoffs"
        handoff_root.mkdir(mode=0o700, exist_ok=False)
        handoff = NativeMP2DJournalHandoff(
            driver,
            journal,
            journal_binding=binding,
            initial_checkpoint=journal.checkpoint,
            journal_source_inventory=inputs.source_inventory,
            journal_authenticator=inputs.authenticator,
            handoff_root=handoff_root,
            monotonic=monotonic,
        )
        received_at = None
        previous_release = feature.sha256
        shared_ids = frozenset(sequence_id(row.sequence) for row in shared)
        for wave_index in range(1, 29):
            # The same original wave start pays for reconstruction, authority,
            # fitting, collection, selection and durable publication/return.
            wave_started = check("wave_start")
            report = reconstruct()
            history = report.history
            j.require(
                history.round_index == wave_index
                and len(history.observations) == 64 + 16 * (wave_index - 1),
                "adaptive prefix differs",
            )
            assembly = assembly_for(history)
            release = release_for(history, previous_release)
            callback = JournalHistoryCallback(
                journal_parent / "journal",
                trusted_parent=journal_parent,
                expected_binding=binding,
                expected_checkpoint=journal.checkpoint,
                expected_source_inventory=inputs.source_inventory,
                authenticator=inputs.authenticator,
                expected_round_index=wave_index,
                expected_wave_head_sha256=history.previous_wave_head_sha256,
            )
            prepared = driver.prepare_wave(
                callback,
                round_index=wave_index,
                poll_ordinal=0,
                previous_wave_head_sha256=history.previous_wave_head_sha256,
                original_wave_started_at=wave_started,
                assembly=assembly,
                expected_assembly=assembly,
                release=release,
                expected_release=release,
            )
            _exclusive_write(root / f"preparation-{wave_index:02d}.json", prepared.record_payload)
            check("after_native_preparation")
            j.require(
                prepared.status == "prepared", "MP2D preparation stopped; retain native result"
            )
            excluded = shared_ids | {sequence_id(row.sequence) for row in history.observations}
            allowed = frozenset(sequence_id(seq) for seq in driver._wave.candidates) - excluded
            filter_sha = j.digest(
                j.canonical(
                    {
                        "source_sha256": source_sha,
                        "history_sha256": history.sha256,
                        "excluded": sorted(excluded),
                    }
                )
            )
            handed = handoff.seal_wave(
                prepared,
                expected_checkpoint=journal.checkpoint,
                expected_wave_head_sha256=history.previous_wave_head_sha256,
                original_wave_started_at=wave_started,
                allowed_sequence_ids=allowed,
                external_filter_sha256=filter_sha,
                method_query_ids=tuple(f"method-{wave_index:02d}-{seat:02d}" for seat in range(14)),
                replicate_ids=(0,) * 14,
            )
            _exclusive_write(root / f"handoff-{wave_index:02d}.json", handed.record_payload)
            check("after_native_handoff")
            j.require(handed.status == "sealed_wave", "MP2D handoff stopped; retain native result")
            j.require(
                last < min(original_deadline, wave_started + 120),
                "original wave deadline expired at outer handoff return",
            )
            previous_release = release.sha256
            for _ in range(16):
                check("before_dispatch")
                journal.dispatch_next()
                check("after_dispatch")
            # Dispatch remains outside the handoff. The next wave reconstructs
            # this complete prefix before any adaptive fit or proposal.
            journal._guard()
            pending = tuple(
                j.OutstandingDispatch(request, journal._acks[key])
                for key, request in sorted(
                    journal._intents.items(), key=lambda item: item[1].charge_index
                )
                if key not in journal._terminal_intents
            )
            j.require(len(pending) == 16, "wave must have exactly 16 acknowledged pending charges")

            def retain_arrival(value, wave_index=wave_index):
                nonlocal received_at
                received_at = finite_clock(monotonic())
                j.require(
                    type(value) is tuple and len(value) <= 16, "bounded receipt tuple required"
                )
                for row in value:
                    j.require(
                        type(row) is tuple and len(row) == 2 and j.pin(row[0]),
                        "receipt pair differs",
                    )
                    j.receipt_bytes(row[1])
                save(
                    f"arrival-{wave_index:02d}.json",
                    {
                        "locally_received_at": received_at,
                        "receipts": [[key, receipt.hex()] for key, receipt in value],
                    },
                )

            arrival = invoke(
                inputs.collector,
                "collect",
                pending,
                original_deadline,
                retain=retain_arrival,
            )
            j.require(
                type(arrival) is tuple and len(arrival) == 16,
                "collector must return all 16 receipts",
            )
            returned = {}
            for row in arrival:
                j.require(
                    type(row) is tuple and len(row) == 2 and j.pin(row[0]), "receipt pair differs"
                )
                j.receipt_bytes(row[1])
                j.require(row[0] not in returned, "duplicate returned intent")
                returned[row[0]] = row[1]
            j.require(
                set(returned) == {r.request.intent_sha256 for r in pending},
                "returned intent inventory differs",
            )
            for row in pending:
                check("before_terminal_record")
                key = row.request.intent_sha256
                journal.record_terminal(key, returned[key])
            check("after_wave_terminal_records")
            save(
                f"wave-{wave_index:02d}.json",
                {
                    "checkpoint": asdict(journal.checkpoint),
                    "locally_recorded_at": last,
                },
            )
        report = reconstruct()
        history = report.history
        j.require(
            report.checkpoint.charged_count == 512
            and report.adaptive_attempts == 448
            and history.round_index == 29
            and len(history.observations) == 512,
            "final charged accounting differs",
        )
        closure = {
            "contract_sha256": contract_sha,
            "timing_context": timing,
            "journal_checkpoint": asdict(report.checkpoint),
            "history_sha256": history.sha256,
            "query_closed_at": received_at,
        }
        closure_sha = _attested_statement(
            inputs, "query_closure", closure, root=root, invoke=invoke
        )
        release = release_for(history, previous_release)
        bridge.release_private(release, intent(history, "release"), expected_release=release)
        check("after_terminal_private_release")
        assembly = assembly_for(history)
        decision = complete_paid_terminal(
            history=history,
            expected_history_sha256=history.sha256,
            bridge=bridge,
            transform=transform,
            assembly=assembly,
            expected_assembly=assembly,
            terminal_query_ids=frozenset(assembly.eligible_query_ids),
            terminal_eligibility_source_sha256=assembly.eligibility_source_sha256,
            terminal_eligibility_receipt_sha256=assembly.eligibility_receipt_sha256,
            query_closed_at=received_at,
            query_closure_receipt_sha256=closure_sha,
            learner_source_sha256=file_digest(Path(feature.repository) / LEARNER_SOURCE),
            output_root=root / "terminal",
            monotonic=monotonic,
        )
        check("after_paid_terminal_return")
        completion = {
            "contract_sha256": contract_sha,
            "timing_context": timing,
            "journal_checkpoint": asdict(report.checkpoint),
            "history_sha256": history.sha256,
            "terminal_decision_sha256": decision["decision_sha256"],
            "terminal_returned_at": last,
            "terminal_timing": terminal_timing_summary(
                original_epoch=original_epoch,
                original_deadline=original_deadline,
                query_closed_at=received_at,
                decision_completed_at=decision["scientific_completed_at"],
                terminal_returned_at=last,
            ),
        }
        completion_sha = _attested_statement(
            inputs, "terminal_completion", completion, root=root, invoke=invoke
        )
        result = {
            "status": "completed_mp2d_lifecycle",
            "scientific_evidence_accepted": False,
            "logical_charges": 512,
            "adaptive_attempts": 448,
            "common_initial_logical_charges": 64,
            "common_initial_physical_calls_in_this_run": 0,
            "query_closed_at": received_at,
            "terminal_completion_receipt_sha256": completion_sha,
            "terminal_timing": completion["terminal_timing"],
            "journal_checkpoint": asdict(report.checkpoint),
            "history_sha256": history.sha256,
            "terminal_status": decision["status"],
            "selected_sequence_id": decision["selected_sequence_id"],
            "feature_counters": bridge.counters.document(),
        }
        result_sha = save("completed.json", result)
        check("after_completion_publication")
        return {
            **result,
            "result_sha256": result_sha,
            "returned_at": last,
            "completion_attestation_and_publication_seconds": last
            - completion["terminal_returned_at"],
        }
    except BaseException as error:
        cleanup_errors = []
        if journal is not None:
            try:
                journal.stop(detail=f"MP2D runner failed at {phase}")
            except BaseException as cleanup:
                cleanup_errors.append(type(cleanup).__name__)
        try:
            bridge.abort(error)
        except BaseException as cleanup:
            cleanup_errors.append(type(cleanup).__name__)
        save(
            "failure.json",
            {
                "status": "failed_mp2d_lifecycle",
                "phase": phase,
                "type": type(error).__name__,
                "message": str(error),
                "last_monotonic": last,
                "original_deadline": original_deadline,
                "known_checkpoint": None if journal is None else asdict(journal.checkpoint),
                "cleanup_errors": cleanup_errors,
                "scientific_evidence_accepted": False,
            },
        )
        raise
    finally:
        if journal is not None:
            journal.close()

"""Execute the existing eligible GA through charged dispatch and paid terminal fit.

Tuning, objective validity and external receipt authorities remain caller-owned.
The GA uses observed eligible parent fitness and its original first-prefix seats.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from amp_challenge.generators.diffusion import native_ga_partial_driver_records as selection_records
from amp_challenge.generators.diffusion.native_baseline_operators import (
    NormalizedObjectiveContext,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_ga_partial_driver_records import (
    GASelectionAuthority,
    selection_guard,
    validate_selection,
)
from amp_challenge.generators.search import durable_dispatch_journal_records as j
from amp_challenge.generators.search.durable_dispatch_journal import DurableDispatchJournal
from amp_challenge.generators.search.durable_dispatch_journal_verify import (
    JournalHistoryCallback,
    verify_dispatch_journal,
)
from amp_challenge.generators.search.native_static_campaign import ClockBoundDispatch
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
from amp_challenge.generators.search.peptide_ga_driver_v2 import PrivateGACollisions
from amp_challenge.generators.search.peptide_ga_eligible_driver_v3 import execute_eligible_ga_step
from amp_challenge.generators.search.peptide_ga_eligible_driver_v3_records import (
    EligibleGADriverContext,
    check_sources,
    implementation_sha256,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    GAContextEligibility,
    eligible_implementation_sha256,
)
from amp_challenge.models.charged_probability_learner import GeneratorFeatureTransform
from amp_challenge.representations.candidate_features import _exclusive_write
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    FeatureAssemblyBinding,
    finite_clock,
)
from amp_challenge.representations.run_feature_cache_records import canonical as feature_canonical

ARM_ID = "tuned_peptide_ga"
CONTRACT_DOCUMENT = {
    "contract": "eligible_ga_campaign_v2_20260914",
    "arm": ARM_ID,
    "logical_queries": 512,
    "common_initial": 64,
    "waves": 28,
    "method_seats": 14,
    "reserve_seats": 2,
    "selection": "eligible_v3_256_attempt_order_prefix_then_first_14_privately_available",
    "tuning": "externally_authenticated_common_GA_selection_required",
    "wave_seconds": 180,
    "maximum_seconds": 7200,
    "candidate_feature_opportunities": 0,
    "terminal": PAID_TERMINAL_CONTRACT,
    "terminal_timing_amendment_sha256": TIMING_AMENDMENT_SHA256,
    "scientific_clock": "before_provisioning_through_terminal_close_and_publication",
    "restart": "exclusive_output_fail_stop_no_automatic_redispatch",
    "historical_v2_terminal_timing_compliant": False,
    "global_successor_protocol_adopted": False,
}


def contract_document(arm_id=ARM_ID):
    timing_amendment_sha256()
    j.require(arm_id == ARM_ID, "exact plain GA arm required")
    return dict(CONTRACT_DOCUMENT)


def timing_context(epoch, deadline, clock_epoch_id, *, arm_id=ARM_ID):
    epoch, deadline = finite_clock(epoch), finite_clock(deadline)
    j.require(
        0 < deadline - epoch <= 7200 and j.identifier(clock_epoch_id), "original GA clock differs"
    )
    return {
        "contract_sha256": j.digest(j.canonical(contract_document(arm_id))),
        "original_epoch": epoch,
        "original_deadline": deadline,
        "clock_epoch_id": clock_epoch_id,
    }


@dataclass(frozen=True, slots=True)
class EligibleGACampaignInputs:
    configuration_id: str
    training_sequence_keys: tuple[str, ...]
    selection: GASelectionAuthority
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
    attestor: j.CallbackPin


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


def execute_eligible_ga_campaign(
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
    j.require(
        type(inputs) is EligibleGACampaignInputs, "exact eligible GA campaign inputs required"
    )
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
        "eligible GA features must use the fixed run-owned directory",
    )
    callbacks = (
        inputs.authenticator,
        inputs.transport,
        inputs.permission,
        inputs.collector,
        inputs.eligibility,
        inputs.attestor,
    )
    source_path = Path(__file__).resolve()
    source_sha = file_digest(source_path)
    selection_source = Path(selection_records.__file__).resolve()
    selection_source_sha = file_digest(selection_source)
    frozen = None
    ga_context = None
    ga_context_bytes = None
    selection = None
    wave_deadline = None
    fixed_inputs = tuple(getattr(inputs, field.name) for field in fields(inputs))

    def save(name, value):
        payload = j.canonical(value)
        _exclusive_write(root / name, payload)
        return j.digest(payload)

    def check(name):
        nonlocal phase, last
        phase = name
        now = finite_clock(monotonic())
        j.require(now >= last, "original monotonic clock moved backwards")
        if now >= original_deadline or (wave_deadline is not None and now >= wave_deadline):
            raise TimeoutError(f"eligible GA campaign original deadline exceeded at {name}")
        last = now
        j.require(file_digest(source_path) == source_sha, "campaign implementation changed")
        j.require(
            file_digest(selection_source) == selection_source_sha,
            "common GA selection implementation changed",
        )
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
        j.require(
            all(
                getattr(inputs, field.name) is value
                for field, value in zip(fields(inputs), fixed_inputs, strict=True)
            ),
            "campaign input identity changed",
        )
        if ga_context is not None:
            j.require(j.canonical(asdict(ga_context)) == ga_context_bytes, "GA context changed")
            check_sources(ga_context)
        if selection is not None:
            selection_guard(selection)
        finished = time.monotonic()
        if finished >= original_deadline or (
            wave_deadline is not None and finished >= wave_deadline
        ):
            raise TimeoutError("eligible GA campaign deadline exceeded after callback-free guards")
        last = max(last, finished)
        return last

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

    def assembly_for(history, *, persist=False):
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
        if persist:
            save(f"assembly-{history.round_index:02d}.json", assembly.document())
        return assembly

    def resolver(history):
        assembly = assembly_for(history)
        return GAContextEligibility(
            history.sha256,
            binding.objective_context_sha256,
            assembly.eligibility_source_sha256,
            assembly.eligibility_receipt_sha256,
            frozenset(assembly.eligible_query_ids),
        )

    resolver.provider_sha256 = j.digest(
        j.canonical(
            {
                "controller_source_sha256": source_sha,
                "eligibility_source_sha256": inputs.eligibility.source_sha256,
                "transform_sha256": transform.sha256,
            }
        )
    )

    def wave_clock():
        # The GA driver owns its source/history/eligibility checks. Bound its
        # complete work to this wave without recursively re-running the outer
        # journal and common-selection checks at every inner numerical poll.
        before = time.monotonic()
        value = finite_clock(monotonic())
        after = time.monotonic()
        j.require(before <= value <= after, "GA clock differs from the actual absolute domain")
        if after >= original_deadline or (wave_deadline is not None and after >= wave_deadline):
            raise TimeoutError("GA original wave/global clock exhausted")
        return value

    try:
        check("admission")
        j.require(
            type(binding) is j.JournalBinding
            and type(bridge) is RunFeatureBridge
            and type(transform) is GeneratorFeatureTransform
            and type(inputs.context) is NormalizedObjectiveContext
            and type(inputs.selection) is GASelectionAuthority,
            "exact eligible GA journal, feature and objective bindings required",
        )
        binding.__post_init__()
        j.require(
            binding.arm_id == ARM_ID
            and inputs.context.context_sha256 == binding.objective_context_sha256
            and binding.timing_context_sha256 == j.digest(j.canonical(timing)),
            "eligible GA arm/prospective timing binding differs",
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
        j.require(
            transform.representation == "esm320_plus_normalized_length"
            and bridge.counters.candidate_opportunities == 0
            and bridge.counters.private_rows == 0,
            "plain GA must start without candidate or private feature acquisition",
        )
        ga_context = EligibleGADriverContext(
            run_id=binding.run_id,
            seed=binding.seed,
            configuration_id=inputs.configuration_id,
            objective_context_sha256=binding.objective_context_sha256,
            oracle_bundle_sha256=binding.oracle_bundle_sha256,
            history_provider_sha256=binding.provider_sha256,
            implementation_sha256=implementation_sha256(),
            training_sequence_keys=inputs.training_sequence_keys,
            kernel_source_sha256=eligible_implementation_sha256(),
            eligibility_provider_sha256=resolver.provider_sha256,
            eligibility_source_sha256=inputs.eligibility.source_sha256,
            deadline_monotonic=float(original_deadline),
            clock_epoch_id=clock_epoch_id,
        )
        ga_context_bytes = j.canonical(asdict(ga_context))
        selection = validate_selection(inputs.selection, ga_context, checkpoint=check)
        check("after_common_selection_admission")
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
                "configuration_id": ga_context.configuration_id,
                "common_selection_sha256": selection.selection_sha256,
                "ga_context": asdict(ga_context),
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
        ga_root = root / "ga"
        ga_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        received_at = None
        for wave_index in range(1, 29):
            # The same original wave start pays for reconstruction, authority,
            # fitting, collection, selection and durable publication/return.
            wave_started = check("wave_start")
            wave_deadline = min(original_deadline, wave_started + 180)
            report = reconstruct()
            history = report.history
            j.require(
                history.round_index == wave_index
                and len(history.observations) == 64 + 16 * (wave_index - 1),
                "adaptive prefix differs",
            )
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
            private = PrivateGACollisions(
                tuple(sorted(sequence_id(row.sequence) for row in history.observations)),
                tuple(
                    sorted(
                        row.identity.canonical_sequence_id
                        for pair in binding.reserves[wave_index - 1 :]
                        for row in pair
                    )
                ),
            )
            check("before_ga_generation")
            seats = execute_eligible_ga_step(
                ga_root,
                ga_context,
                callback,
                resolver,
                round_index=wave_index,
                previous_wave_head_sha256=history.previous_wave_head_sha256,
                private=private,
                monotonic=wave_clock,
            )
            save(f"ga-result-{wave_index:02d}.json", asdict(seats))
            check("after_ga_generation")
            j.require(
                seats.status == "ready" and len(seats.method_seats) == 14,
                "eligible GA stopped; retain result without overflow or retry",
            )
            template = binding.initial_requests[0].identity
            requests = tuple(
                j.JournalRequest(
                    f"method-{wave_index:02d}-{index:02d}",
                    sequence,
                    replace(template, canonical_sequence_id=sequence_id(sequence), replicate_id=0),
                )
                for index, sequence in enumerate(seats.method_seats)
            )
            for request in requests:
                binding.validate_request(request)
            before_seal = journal.checkpoint
            save(
                f"request-plan-{wave_index:02d}.json",
                {
                    "ga_phase_sha256": seats.phase_sha256,
                    "previous_checkpoint": asdict(before_seal),
                    "method_requests": [request.document() for request in requests],
                    "reserve_requests": [
                        request.document() for request in binding.reserves[wave_index - 1]
                    ],
                    "original_wave_started_at": wave_started,
                    "effective_deadline": min(original_deadline, wave_started + 180),
                    "common_selection_sha256": selection.selection_sha256,
                },
            )
            check("before_wave_seal")
            j.require(
                last < min(original_deadline, wave_started + 180),
                "original wave deadline expired before sealing",
            )
            sealed = journal.seal_wave(requests)
            save(
                f"handoff-{wave_index:02d}.json",
                {
                    "status": "sealed_wave",
                    "round_index": wave_index,
                    "checkpoint": asdict(sealed),
                    "ga_phase_sha256": seats.phase_sha256,
                    "method_sequences": seats.method_seats,
                    "reserve_count": 2,
                    "method_count": 14,
                    "original_wave_started_at": wave_started,
                    "effective_deadline": min(original_deadline, wave_started + 180),
                    "scientific_evidence_accepted": False,
                },
            )
            check("after_ga_handoff")
            j.require(
                last < min(original_deadline, wave_started + 180),
                "original wave deadline expired at outer handoff return",
            )
            wave_deadline = None
            for _ in range(16):
                check("before_dispatch")
                journal.dispatch_next()
                check("after_dispatch")
            # Dispatch remains outside the handoff. The next wave reconstructs
            # this complete prefix before any GA proposal.
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
        final_eligibility = resolver(history)
        assembly = assembly_for(history, persist=True)
        j.require(
            frozenset(assembly.eligible_query_ids) == final_eligibility.query_ids
            and assembly.eligibility_receipt_sha256 == final_eligibility.receipt_sha256,
            "terminal and GA eligibility authorities disagree",
        )
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
            "status": "completed_eligible_ga_lifecycle",
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
            "common_selection_sha256": selection.selection_sha256,
            "configuration_id": ga_context.configuration_id,
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
                journal.stop(detail=f"eligible GA runner failed at {phase}")
            except BaseException as cleanup:
                cleanup_errors.append(type(cleanup).__name__)
        try:
            bridge.abort(error)
        except BaseException as cleanup:
            cleanup_errors.append(type(cleanup).__name__)
        save(
            "failure.json",
            {
                "status": "failed_eligible_ga_lifecycle",
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

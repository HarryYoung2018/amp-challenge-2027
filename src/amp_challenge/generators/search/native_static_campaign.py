"""Execute native static or iterative baselines through paid terminal fitting on one clock.

Provisioned receipt/eligibility authorities remain external. Their validity is
not established by this runner. Successful execution is not scientific admission.
The low-level journal retains its historical wire protocol; this outer timing
contract is explicitly different and bound in its timing_context_sha256.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from amp_challenge.generators.diffusion import native_baseline_context_records as context_records
from amp_challenge.generators.diffusion.native_baseline_context import NativeBaselineContextEnsemble
from amp_challenge.generators.diffusion.native_baseline_operators import (
    MODES,
    NativeBaselineEnsemble,
    NormalizedObjectiveContext,
    sequence_id,
)
from amp_challenge.generators.diffusion.native_baseline_seat_driver import NativeBaselineSeatDriver
from amp_challenge.generators.diffusion.native_initialization import (
    TRIPLES,
    load_audited_native_initialization,
)
from amp_challenge.generators.diffusion.native_search_posterior import FrozenNativePosteriorBinding
from amp_challenge.generators.search import durable_dispatch_journal_records as j
from amp_challenge.generators.search.durable_dispatch_journal import DurableDispatchJournal
from amp_challenge.generators.search.durable_dispatch_journal_verify import verify_dispatch_journal
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
from amp_challenge.models.charged_probability_learner import (
    GeneratorFeatureTransform,
    fit_charged_learner,
)
from amp_challenge.representations.candidate_features import _exclusive_write
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.run_feature_cache_bridge import RunFeatureBridge
from amp_challenge.representations.run_feature_cache_records import (
    FeatureAssemblyBinding,
    FeatureIntent,
    finite_clock,
)
from amp_challenge.representations.run_feature_cache_records import (
    canonical as feature_canonical,
)
from amp_challenge.representations.run_feature_cache_views import (
    NativeFeaturePosterior,
    assemble_charged,
    native_evaluator_source_sha256,
)

CONTRACT = "native_static_campaign_v2_20260914"
CONTRACT_DOCUMENT = {
    "contract": CONTRACT,
    "arm": MODES[0],
    "logical_queries": 512,
    "common_initial": 64,
    "waves": 28,
    "method_seats": 14,
    "reserve_seats": 2,
    "selection": "native_baseline_selection_v1_20260914",
    "terminal": PAID_TERMINAL_CONTRACT,
    "terminal_timing_amendment_sha256": TIMING_AMENDMENT_SHA256,
    "query_closure": "all 512 terminal receipts locally returned and authenticated",
    "scientific_clock": "starts before provisioning; includes final fit, worker close and publication",
    "maximum_seconds": 7200,
    "restart": "refuse existing output; retain charges; no automatic redispatch",
    "historical_v2_terminal_timing_compliant": False,
}
CONTRACT_SHA256 = j.digest(j.canonical(CONTRACT_DOCUMENT))


REWARD_CONTRACT_DOCUMENT = {
    **CONTRACT_DOCUMENT,
    "contract": "native_reward_static_campaign_v2_20260914",
    "arm": MODES[1],
    "policy_update": "one context-eligible reward/KL update from initial 64, then frozen",
    "update_eligibility": "same externally pinned successful/feasible rows as initial charged fit",
}


ITERATIVE_CONTRACT_DOCUMENT = {
    **CONTRACT_DOCUMENT,
    "contract": "native_iterative_campaign_v2_20260914",
    "arm": MODES[2],
    "policy_update": "context-eligible native ARCADIAMP updates before waves 2 through 28",
    "selection_schedule": "256 native candidates and 14 seats from each complete charged history",
    "update_eligibility": "same externally pinned successful/feasible rows as each charged fit",
    "journal_replay": "independent reconstruction before every history-consuming step",
    "terminal_policy_update": "none; no unused round-29 update",
}


def contract_document(arm_id):
    timing_amendment_sha256()
    j.require(arm_id in MODES, "only the three native baseline arms are supported")
    return dict(
        {
            MODES[0]: CONTRACT_DOCUMENT,
            MODES[1]: REWARD_CONTRACT_DOCUMENT,
            MODES[2]: ITERATIVE_CONTRACT_DOCUMENT,
        }[arm_id]
    )


def timing_context(epoch, deadline, clock_epoch_id, *, arm_id=MODES[0]):
    epoch, deadline = finite_clock(epoch), finite_clock(deadline)
    j.require(0 < deadline - epoch <= 7200, "original duration must be in (0, 7200]")
    j.require(j.identifier(clock_epoch_id), "original clock epoch identifier required")
    return {
        "contract_sha256": j.digest(j.canonical(contract_document(arm_id))),
        "original_epoch": epoch,
        "original_deadline": deadline,
        "clock_epoch_id": clock_epoch_id,
    }


class ClockBoundDispatch:
    """Bind the last permission/transport boundary to the original live clock.

    A late returned acknowledgement is retained before the timeout is raised;
    it never authorizes a replacement attempt. Provision before issuing shared
    initial copy receipts because this wrapper changes the callback source pin.
    """

    def __init__(self, delegate, *, kind, timing, monotonic, receipt_root):
        j.require(type(delegate) is j.CallbackPin, "pinned dispatch delegate required")
        j.require(kind in ("permission", "transport"), "dispatch boundary kind differs")
        self.delegate, self.kind, self.timing, self.clock = delegate, kind, dict(timing), monotonic
        self.receipt_root = Path(receipt_root)
        self.last = timing["original_epoch"]
        self.source_sha256 = j.digest(
            j.canonical(
                {
                    "wrapper_source": file_digest(Path(__file__)),
                    "kind": kind,
                    "delegate_source": delegate.source_sha256,
                    "timing": timing,
                }
            )
        )
        self._fixed = (delegate, kind, j.canonical(timing), monotonic, self.receipt_root)

    def __call__(self, request):
        def check():
            j.require(
                (self.delegate, self.kind, j.canonical(self.timing), self.clock, self.receipt_root)
                == self._fixed,
                "dispatch wrapper binding changed",
            )
            j.require(
                getattr(self.delegate.target, "source_sha256", None) == self.delegate.source_sha256,
                "dispatch delegate source changed",
            )
            now = finite_clock(self.clock())
            j.require(now >= self.last, "dispatch original clock moved backwards")
            self.last = now
            if now >= self.timing["original_deadline"]:
                raise TimeoutError("original deadline at the actual dispatch boundary")

        check()
        value = self.delegate.target(request)
        if self.kind == "transport":
            j.receipt_bytes(value)
            _exclusive_write(
                self.receipt_root / f"dispatch-return-{request.charge_index:03d}.receipt",
                value,
            )
        check()
        return value


@dataclass(frozen=True, slots=True)
class StaticCampaignInputs:
    """Authorities must be provisioned independently of candidate/outcome ranking.

    collect(outstanding, absolute_deadline) returns an exact tuple of 16
    (intent_sha256, raw_terminal_receipt) pairs, possibly out of charge order.
    It must retain any partial arrivals if it raises; the runner never invents
    missing/timeout receipts. authorize(history, transform) returns the frozen
    FeatureAssemblyBinding. attest(kind, expected_bytes) obtains a receipt that
    the journal's independently pinned authenticator verifies.
    """

    ensemble: NativeBaselineEnsemble | NativeBaselineContextEnsemble
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


def _load_audited_units(*, checkpoints, audit, device, check):
    """Load all ten real audited checkpoints and their pinned training corpora.

    Call only inside the already-started allocation/clock. No feature transform
    or oracle objective is inferred from generator training data here.
    """
    units, corpora = [], []
    checkpoints = Path(checkpoints)
    for ordinal, triple in enumerate(TRIPLES):
        check()
        unit = load_audited_native_initialization(checkpoints, audit, triple=triple)
        directory = checkpoints / f"{ordinal:02d}"
        projection = (directory / "training_projection.jsonl").read_bytes()
        manifest = json.loads((directory / "manifest.json").read_bytes())
        j.require(
            j.digest(projection) == manifest["artifacts"]["training_projection.jsonl"]["sha256"],
            "audited training projection changed",
        )
        corpus = tuple(json.loads(line)["sequence"] for line in projection.splitlines())
        j.require(
            tuple(sorted(map(sequence_id, corpus))) == unit.training_sequence_ids,
            "training corpus differs from audited initializer",
        )
        unit.model.to(device)
        units.append(unit)
        corpora.append(corpus)
        check()
    return tuple(units), tuple(corpora)


def load_categorical_ensemble(*, checkpoints, audit, binding, device, check):
    units, corpora = _load_audited_units(
        checkpoints=checkpoints, audit=audit, device=device, check=check
    )
    return NativeBaselineEnsemble(
        MODES[0],
        units,
        corpora,
        NormalizedObjectiveContext(binding.objective_context_sha256),
        run_id=binding.run_id,
        seed=binding.seed,
        oracle_bundle_sha256=binding.oracle_bundle_sha256,
    )


def _load_context_ensemble(
    *,
    checkpoints,
    audit,
    binding,
    device,
    check,
    original_deadline,
    clock_epoch_id,
    monotonic,
    expected_arm,
):
    """Provision the actual eligible reward operator inside the original paid clock."""
    j.require(binding.arm_id == expected_arm, "context loader arm identity differs")
    units, corpora = _load_audited_units(
        checkpoints=checkpoints, audit=audit, device=device, check=check
    )
    context = context_records.NativeBaselineContext(
        expected_arm,
        binding.run_id,
        binding.seed,
        NormalizedObjectiveContext(binding.objective_context_sha256),
        binding.oracle_bundle_sha256,
        context_records.implementation_sha256(),
        original_deadline,
        clock_epoch_id,
    )
    ensemble = NativeBaselineContextEnsemble(context, units, corpora, monotonic=monotonic)
    check()
    return ensemble


def load_reward_ensemble(**kwargs):
    return _load_context_ensemble(expected_arm=MODES[1], **kwargs)


def load_iterative_ensemble(**kwargs):
    return _load_context_ensemble(expected_arm=MODES[2], **kwargs)


def execute_static_campaign(inputs, **kwargs):
    j.require(type(inputs) is StaticCampaignInputs, "exact static campaign inputs required")
    j.require(inputs.binding.arm_id in MODES[:2], "static entry requires a frozen-stream arm")
    return execute_baseline_campaign(inputs, **kwargs)


def execute_baseline_campaign(
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
    j.require(type(inputs) is StaticCampaignInputs, "exact static campaign inputs required")
    root = Path(output_root) / "execution"
    j.require(root.is_absolute(), "absolute controller output root required")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    journal = None
    phase, last = "admission", finite_clock(original_epoch)
    binding, bridge, transform = inputs.binding, inputs.bridge, inputs.transform
    contract = contract_document(binding.arm_id)
    iterative = binding.arm_id == MODES[2]
    timing = timing_context(
        original_epoch, original_deadline, clock_epoch_id, arm_id=binding.arm_id
    )
    contract_sha = timing["contract_sha256"]
    inner = (
        inputs.ensemble.inner
        if type(inputs.ensemble) is NativeBaselineContextEnsemble
        else inputs.ensemble
    )
    callbacks = (
        inputs.authenticator,
        inputs.transport,
        inputs.permission,
        inputs.collector,
        inputs.eligibility,
        inputs.feasibility,
        inputs.attestor,
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
            raise TimeoutError(f"static campaign original deadline exceeded at {name}")
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
            root / "journal",
            trusted_parent=root,
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

    def attested(kind, expected):
        payload = j.canonical(expected)
        receipt = invoke(
            inputs.attestor,
            kind,
            kind,
            payload,
            retain=lambda value: _exclusive_write(
                root / (kind + ".receipt"), j.receipt_bytes(value)
            ),
        )
        semantic = invoke(inputs.authenticator, "authenticate_" + kind, kind, receipt, payload)
        j.require(
            semantic == payload, "external timing attestation differs from expected statement"
        )
        return j.digest(receipt)

    try:
        check("admission")
        j.require(
            type(binding) is j.JournalBinding
            and type(bridge) is RunFeatureBridge
            and type(transform) is GeneratorFeatureTransform
            and (
                (binding.arm_id == MODES[0] and type(inputs.ensemble) is NativeBaselineEnsemble)
                or (
                    binding.arm_id in MODES[1:]
                    and type(inputs.ensemble) is NativeBaselineContextEnsemble
                )
            ),
            "exact native baseline, journal and feature inputs required",
        )
        binding.__post_init__()
        j.require(
            binding.arm_id == inner.mode
            and binding.arm_id in MODES
            and binding.timing_context_sha256 == j.digest(j.canonical(timing)),
            "baseline arm/prospective timing binding differs",
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
        j.require(
            (inner.run_id, inner.seed, inner.oracle_bundle_sha256)
            == (binding.run_id, binding.seed, binding.oracle_bundle_sha256),
            "native ensemble run/oracle binding differs",
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
                "ten_checkpoint_mixture": inner.protocol_ten_checkpoint_mixture,
                "scientific_evidence_accepted": False,
            },
        )
        journal = DurableDispatchJournal.create(
            root / "journal",
            trusted_parent=root,
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
        driver = NativeBaselineSeatDriver(
            inputs.ensemble,
            root / "seats",
            deadline_monotonic=float(original_deadline),
            clock_epoch_id=clock_epoch_id,
            monotonic=monotonic,
        )
        method_inventory = []

        def select_requests(history):
            assembly = None
            generation = {"expected_previous_head_sha256": history.previous_wave_head_sha256}
            if binding.arm_id in MODES[1:]:
                assembly = assembly_for(history)
                eligible_ids = frozenset(assembly.eligible_query_ids)
                generation["eligibility"] = context_records.NativeBaselineEligibility(
                    history.sha256,
                    binding.objective_context_sha256,
                    assembly.eligibility_source_sha256,
                    assembly.eligibility_receipt_sha256,
                    eligible_ids,
                )
                generation["expectations"] = context_records.NativeBaselineExpectations(
                    history.sha256,
                    binding.objective_context_sha256,
                    assembly.eligibility_source_sha256,
                    assembly.eligibility_receipt_sha256,
                    eligible_ids,
                    history.previous_wave_head_sha256,
                    None
                    if inputs.ensemble.last_envelope is None
                    else inputs.ensemble.last_envelope.sha256,
                )
            pool = driver.generate(history, **generation)
            check("after_native_generation")
            if assembly is None:
                assembly = assembly_for(history)
            data = assemble_charged(
                bridge,
                history,
                transform,
                assembly,
                intent(history, "charged"),
                expected_authority=assembly,
            )
            check("before_history_fit")
            learner = fit_charged_learner(history, transform, **data)
            check("after_history_fit")
            port = bridge.scoped_consumer(intent(history, "candidate"))
            pins = json.loads(port.binding_payload)
            posterior_binding = FrozenNativePosteriorBinding(
                history.sha256,
                binding.objective_context_sha256,
                learner.numerical_sha256,
                pins["feature_source_sha256"],
                native_evaluator_source_sha256(
                    pins["provider_sha256"], inputs.feasibility.source_sha256
                ),
            )
            posterior = NativeFeaturePosterior(
                port,
                learner,
                posterior_binding,
                inner.context,
                feasibility=inputs.feasibility.target,
                feasibility_source_sha256=inputs.feasibility.source_sha256,
            )
            private_ids = {
                row.identity.canonical_sequence_id for wave in binding.reserves for row in wave
            }
            submitted_ids = {sequence_id(row.sequence) for row in history.observations}
            permitted = (
                frozenset(map(sequence_id, pool.accepted_sequences)) - private_ids - submitted_ids
            )
            schedule = driver.select(
                posterior, expected_binding=posterior_binding, permitted_sequence_ids=permitted
            )
            check("after_native_selection")
            template = binding.initial_requests[0].identity
            waves = tuple(
                tuple(
                    j.JournalRequest(
                        f"method-{wave_index:02d}-{seat:02d}",
                        sequence,
                        replace(
                            template, canonical_sequence_id=sequence_id(sequence), replicate_id=0
                        ),
                    )
                    for seat, sequence in enumerate(wave)
                )
                for wave_index, wave in enumerate(schedule.waves, history.round_index)
            )
            j.require(
                len(waves) == (1 if iterative else 28) and all(len(w) == 14 for w in waves),
                "native schedule size differs",
            )
            method_inventory.extend(r for w in waves for r in w)
            all_rows = (
                *binding.initial_requests,
                *(r for w in binding.reserves for r in w),
                *method_inventory,
            )
            j.require(
                len({row.query_id for row in all_rows}) == len(all_rows)
                and len({row.identity.key for row in all_rows}) == len(all_rows)
                and len(all_rows) == (120 + 14 * history.round_index if iterative else 512),
                "committed shared/method request inventory collides",
            )
            for row in all_rows:
                binding.validate_request(row)
            save(
                f"wave-{history.round_index:02d}-request-plan.json"
                if iterative
                else "frozen-request-plan.json",
                {
                    "schedule_sha256": schedule.sha256,
                    "method_waves": [[r.document() for r in wave] for wave in waves],
                    "reserve_waves": [[r.document() for r in wave] for wave in binding.reserves],
                    "before_adaptive_attempts": journal.checkpoint.charged_count == 64,
                },
            )
            return waves

        waves = select_requests(history)
        received_at = None
        for wave_index in range(1, 29):
            if iterative and wave_index > 1:
                report = reconstruct()
                history = report.history
                j.require(
                    history.round_index == wave_index
                    and len(history.observations) == 64 + 16 * (wave_index - 1),
                    "iterative update lacks the exact preceding complete wave",
                )
                waves = select_requests(history)
            wave = waves[0] if iterative else waves[wave_index - 1]
            check("before_wave_seal")
            journal.seal_wave(wave)
            for _ in range(16):
                check("before_dispatch")
                journal.dispatch_next()
                check("after_dispatch")
            # Producer guards remain active for every event. The iterative arm
            # independently reconstructs the next history before using it; static
            # arms keep their frozen stream until the final independent replay.
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
        closure_sha = attested("query_closure", closure)
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
        completion_sha = attested("terminal_completion", completion)
        result = {
            "status": (
                "completed_categorical_lifecycle"
                if binding.arm_id == MODES[0]
                else (
                    "completed_iterative_d3pm_lifecycle"
                    if iterative
                    else "completed_reward_kl_lifecycle"
                )
            ),
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
                journal.stop(detail=f"static runner failed at {phase}")
            except BaseException as cleanup:
                cleanup_errors.append(type(cleanup).__name__)
        try:
            bridge.abort(error)
        except BaseException as cleanup:
            cleanup_errors.append(type(cleanup).__name__)
        save(
            "failure.json",
            {
                "status": "failed_static_lifecycle",
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

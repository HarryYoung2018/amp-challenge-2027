"""Additive context-eligible plain GA driver; no fitter, issuer, tuner or oracle."""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import PhaseBuilder
from amp_challenge.generators.search import peptide_ga_driver_v2 as legacy
from amp_challenge.generators.search.peptide_ga_eligible_driver_v3_records import (
    ARTIFACT,
    MAX_PHASE_BYTES,
    MAX_PHASES,
    MAX_RUN_BYTES,
    PAYLOADS,
    TERMINAL_STATUSES,
    EligibleGADriverFailure,
    check_resolver,
    check_sources,
    expectation_pins,
    finite_clock,
    kernel_input,
    resolve_eligibility,
    validate_growth,
)
from amp_challenge.generators.search.peptide_ga_eligible_driver_v3_verify import (
    encoded_phase_size,
    private_positions,
    reconstruct_eligible_ga_driver,
    validate_selection,
    verify_actual_predecessor,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3 import generate_eligible_prefix
from amp_challenge.generators.search.peptide_ga_eligible_v3_records import eligible_batch_digest
from amp_challenge.generators.search.peptide_ga_eligible_v3_verify import _verify_with_checkpoint
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    canonical_json_bytes,
    require,
    sequence_key,
)
from amp_challenge.generators.search.verified_charged_history import read_verified_history


def _stopped_prefix(prefix):
    if prefix is None or prefix.status == "stopped_deadline":
        return prefix
    result = replace(prefix, status="stopped_deadline")
    return replace(result, output_sha256=eligible_batch_digest(result))


def execute_eligible_ga_step(
    root,
    context,
    history_callback,
    eligibility_reconstructor,
    *,
    round_index,
    previous_wave_head_sha256,
    private,
    max_new_attempts=ATTEMPT_CAP,
    monotonic=time.monotonic,
):
    """Return method seats only after exact independent replay and timely publication.

    The original deadline/epoch and authenticated providers are caller-owned.
    Exceptions after entry are terminal handoffs, not an implicit retry grant.
    """
    check_sources(context)
    check_resolver(context, eligibility_reconstructor)
    require(type(private) is legacy.PrivateGACollisions, "eligible private inventory type differs")
    private.__post_init__()
    require(
        type(max_new_attempts) is int and 0 <= max_new_attempts <= ATTEMPT_CAP,
        "eligible driver attempt chunk differs",
    )
    require(callable(monotonic), "eligible driver clock must be callable")
    context_bytes = canonical_json_bytes(asdict(context))
    private_bytes = canonical_json_bytes(asdict(private))
    history = expected = prefix = None
    history_bytes = eligibility_bytes = None
    prefix_evidence = published = None

    def now():
        return finite_clock(monotonic())

    def time_checkpoint():
        if now() >= context.deadline_monotonic:
            raise TimeoutError("eligible driver original deadline exhausted")

    def guard():
        check_sources(context)
        check_resolver(context, eligibility_reconstructor)
        require(
            callable(history_callback)
            and getattr(history_callback, "provider_sha256", None)
            == context.history_provider_sha256,
            "eligible driver history provider changed",
        )
        require(
            canonical_json_bytes(asdict(context)) == context_bytes
            and canonical_json_bytes(asdict(private)) == private_bytes,
            "eligible driver caller context/private input drifted",
        )
        if history is not None:
            require(
                canonical_json_bytes(asdict(history)) == history_bytes,
                "eligible raw history drifted",
            )
        if expected is not None:
            require(
                canonical_json_bytes(expected.document()) == eligibility_bytes,
                "eligible driver authority input drifted",
            )

    try:
        time_checkpoint()
        guard()
        prior = reconstruct_eligible_ga_driver(
            root, context, eligibility_reconstructor, checkpoint=time_checkpoint
        )
        guard()
        time_checkpoint()
        if prior and prior[-1].result.status in TERMINAL_STATUSES:
            guard()
            return prior[-1].result
        history = read_verified_history(
            history_callback,
            provider_sha256=context.history_provider_sha256,
            run_id=context.run_id,
            seed=context.seed,
            round_index=round_index,
            objective_context_sha256=context.objective_context_sha256,
            oracle_bundle_sha256=context.oracle_bundle_sha256,
            previous_wave_head_sha256=previous_wave_head_sha256,
        )
        history_bytes = canonical_json_bytes(asdict(history))
        time_checkpoint()
        expected = resolve_eligibility(context, history, eligibility_reconstructor)
        eligibility_bytes = canonical_json_bytes(expected.document())
        guard()
        charged_count = len(private.charged_sequence_keys)
        validate_growth(prior, history, charged_count)
        require(
            {sequence_key(row.sequence) for row in history.observations}
            <= set(private.charged_sequence_keys)
            and not set(context.training_sequence_keys).intersection(private.charged_sequence_keys),
            "eligible driver private charged/training inventory differs",
        )
        same_history = [row for row in prior if row.history.sha256 == history.sha256]
        require(
            all(row.eligibility == expected for row in same_history),
            "eligible driver changed same-history applicability",
        )
        old_prefixes = [row.prefix for row in same_history if row.prefix is not None]
        old = old_prefixes[-1] if old_prefixes else None
        positions = ()
        if now() >= context.deadline_monotonic:
            # Preserve actual pending resume claims rather than minting an empty cursor.
            if old is not None and old.status == "in_progress":
                prefix = generate_eligible_prefix(
                    kernel_input(context, history, expected),
                    expected_kernel_source_sha256=context.kernel_source_sha256,
                    deadline_monotonic=context.deadline_monotonic,
                    clock=now,
                    resume=old,
                    max_new_attempts=0,
                    **expectation_pins(context, history, expected),
                )
            else:
                prefix = _stopped_prefix(old)
            status = "stopped_deadline"
        elif not history.complete or charged_count > len(history.observations):
            status = "paused_incomplete_wave"
        elif history.round_index == 29:
            status = "budget_complete_pending_controller_terminal"
        else:
            if old is not None and old.status == "complete":
                prefix = old
            else:
                require(old is None or old.status == "in_progress", "eligible prior is terminal")
                prefix = generate_eligible_prefix(
                    kernel_input(context, history, expected),
                    expected_kernel_source_sha256=context.kernel_source_sha256,
                    deadline_monotonic=context.deadline_monotonic,
                    clock=now,
                    resume=old,
                    max_new_attempts=max_new_attempts,
                    **expectation_pins(context, history, expected),
                )
            prefix_evidence = prefix.canonical_bytes()
            guard()
            if prefix.status != "stopped_deadline":
                try:
                    _verify_with_checkpoint(
                        prefix,
                        kernel_input(context, history, expected),
                        expected_kernel_source_sha256=context.kernel_source_sha256,
                        expected_contract_sha256=context.kernel_contract_sha256,
                        expected_deadline_monotonic=context.deadline_monotonic,
                        checkpoint=time_checkpoint,
                        **expectation_pins(context, history, expected),
                    )
                except TimeoutError:
                    prefix = _stopped_prefix(prefix)
            status = {
                "in_progress": "paused_prefix",
                "attempt_cap_exhausted": "abstained_incomplete_prefix",
                "abstained_no_eligible_parent": "abstained_no_eligible_parent",
                "stopped_deadline": "stopped_deadline",
                "complete": "ready",
            }[prefix.status]
            if status == "ready":
                try:
                    positions = private_positions(prefix, private)
                except ValueError:
                    status = "abstained_insufficient_seats"
        guard()
        verify_actual_predecessor(prefix, old)
        require(
            canonical_json_bytes(
                resolve_eligibility(context, history, eligibility_reconstructor).document()
            )
            == eligibility_bytes,
            "eligible external authority changed during work",
        )
        guard()
        if now() >= context.deadline_monotonic:
            prefix, positions, status = _stopped_prefix(prefix), (), "stopped_deadline"

        def encode():
            prefix_bytes = None if prefix is None else prefix.canonical_bytes()
            sequences = (
                [] if not positions else [prefix.accepted_sequences[index] for index in positions]
            )
            document = {
                "status": status,
                "round_index": history.round_index,
                "charged_count": charged_count,
                "history_sha256": history.sha256,
                "eligibility_receipt_sha256": expected.receipt_sha256,
                "prefix_sha256": None if prefix is None else prefix.output_sha256,
                "selected_prefix_positions": list(positions),
                "selected_sequences": sequences,
            }
            validate_selection(history, expected, prefix, document)
            if status == "ready":
                earlier_ready = [
                    row
                    for row in prior
                    if row.history.round_index == history.round_index
                    and row.result.status == "ready"
                ]
                require(
                    not earlier_ready
                    or tuple(sequences) == earlier_ready[-1].result.selected_sequences,
                    "eligible private recomposition changed already recorded seats",
                )
            payloads = {
                "context.json": asdict(context),
                "history.json": asdict(history),
                "eligibility.json": expected.document(),
                "prefix.json": None if prefix is None else asdict(prefix),
                "selection.json": document,
            }
            encoded_payloads = {
                name: canonical_json_bytes(value) for name, value in payloads.items()
            }
            require(
                (None if prefix is None else prefix.canonical_bytes()) == prefix_bytes,
                "eligible driver prefix drifted during output encoding",
            )
            return encoded_payloads

        encoded = encode()
        prefix_evidence = encoded["prefix.json"]
        guard()
        if now() >= context.deadline_monotonic and status != "stopped_deadline":
            prefix, positions, status = _stopped_prefix(prefix), (), "stopped_deadline"
            encoded = encode()
            prefix_evidence = encoded["prefix.json"]
            guard()
            now()  # A second encoding cannot conceal an invalid clock value.
        if prior and all(
            prior[-1].seal.read_payload_bytes(name) == value for name, value in encoded.items()
        ):
            time_checkpoint()
            guard()
            return prior[-1].result
        require(len(prior) < MAX_PHASES, "eligible driver phase cap exceeded")
        predecessors = {} if not prior else {"previous_driver_phase": prior[-1].seal.seal_sha256}
        size = encoded_phase_size(encoded, predecessors)
        require(
            size <= MAX_PHASE_BYTES
            and sum(row.physical_bytes for row in prior) + size <= MAX_RUN_BYTES,
            "eligible driver physical publication byte cap exceeded",
        )
        with PhaseBuilder(
            Path(root) / f"phase-{len(prior):06d}",
            artifact=ARTIFACT,
            predecessor_seals=predecessors,
        ) as builder:
            for name, value in encoded.items():
                builder.write_bytes(name, value)
            published = builder.publish(expected_payload_paths=PAYLOADS).seal_sha256
        guard()
        # Terminal evidence can be reconstructed offline; no new time or seats.
        result = reconstruct_eligible_ga_driver(root, context, eligibility_reconstructor)[-1].result
        guard()
        current = now()
        if current >= context.deadline_monotonic and result.status != "stopped_deadline":
            raise TimeoutError(
                "eligible driver publication/reconstruction crossed original deadline"
            )
        guard()
        return result
    except Exception as error:
        raise EligibleGADriverFailure(
            f"eligible GA run must stop: {type(error).__name__}: {error}",
            prefix_evidence=prefix_evidence,
            published_phase_sha256=published,
        ) from error

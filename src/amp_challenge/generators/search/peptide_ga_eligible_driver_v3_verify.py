"""Independent durable reconstruction, not producer invocation or issuer proof.

The accepted independent kernel edit replay is reused, together with structural
legacy history/JSON helpers and immutable I/O. No GA proposer is called here.
Offline reconstruction describes historical records, not a fresh runtime seat grant.
"""

from __future__ import annotations

import stat
from dataclasses import asdict
from pathlib import Path

from amp_challenge.evaluation import sequential_v2_seals as seals
from amp_challenge.generators.search import peptide_ga_driver_v2 as legacy
from amp_challenge.generators.search.peptide_ga_eligible_driver_v3_records import (
    ARTIFACT,
    MAX_PHASE_BYTES,
    MAX_PHASES,
    MAX_RUN_BYTES,
    PAYLOADS,
    STATUSES,
    EligibleGADriverResult,
    ReconstructedEligibleGAPhase,
    check_resolver,
    check_sources,
    decode_eligibility,
    decode_prefix,
    expectation_pins,
    kernel_input,
    resolve_eligibility,
    validate_growth,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3_verify import _verify_with_checkpoint
from amp_challenge.generators.search.peptide_ga_selection_policy_impl_v1 import (
    controller_first_available_prefix_positions,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_json_bytes,
    digest,
    require,
    sequence_key,
)


def encoded_phase_size(payloads, predecessors):
    """Exact metadata-free PhaseBuilder physical bytes, including its two markers."""
    hashes = {name: digest(value) for name, value in payloads.items()}
    receipt = seals.canonical_json_bytes(
        {
            "artifact": ARTIFACT,
            "metadata": {},
            "payloads": hashes,
            "predecessor_seals": predecessors,
            "schema_version": 1,
            "status": "sealed",
        }
    )
    marker = seals.checksum_manifest_bytes({**hashes, "receipt.json": digest(receipt)})
    return sum(map(len, payloads.values())) + len(receipt) + len(marker)


def verify_actual_predecessor(prefix, previous):
    """Bind retained claims to the actual already replayed immediate prior wrapper."""
    if prefix is None:
        return  # No kernel invocation, hence no new attempts or claimed resume.
    if previous is None:
        require(prefix.prior_resume_sha256 is None, "eligible prefix invents a prior resume")
        return
    require(
        (
            prefix.input_sha256,
            prefix.semantic_sha256,
            prefix.source_sha256,
            prefix.contract_sha256,
            prefix.deadline_monotonic,
        )
        == (
            previous.input_sha256,
            previous.semantic_sha256,
            previous.source_sha256,
            previous.contract_sha256,
            previous.deadline_monotonic,
        ),
        "eligible prefix changed its same-history input or original deadline",
    )
    if prefix == previous:
        return  # Reuse, not another generation.
    if previous.status == "complete" and prefix.status == "stopped_deadline":
        require(
            prefix.prefix == previous.prefix
            and prefix.prior_resume_sha256 == previous.prior_resume_sha256
            and prefix.prior_resume_attempt_count == previous.prior_resume_attempt_count
            and prefix.resume_reconstruction_completed == previous.resume_reconstruction_completed,
            "late completed-prefix reuse changed prior work",
        )
        return
    require(
        previous.status == "in_progress"
        and prefix.prior_resume_sha256 == previous.output_sha256
        and prefix.prior_resume_attempt_count == len(previous.prefix.attempts),
        "eligible prefix prior resume seal/count does not match actual predecessor",
    )
    if prefix.resume_reconstruction_completed:
        require(
            prefix.prefix.attempts[: len(previous.prefix.attempts)] == previous.prefix.attempts,
            "eligible prefix rewrote its actual predecessor attempts",
        )
    else:
        require(
            prefix.status == "stopped_deadline"
            and prefix.prefix is None
            and prefix.next_attempt_index is None,
            "unverified prior replay invents authenticated work or fresh quota",
        )


def validate_selection(history, eligibility, prefix, document):
    require(
        type(document) is dict
        and set(document)
        == {
            "status",
            "round_index",
            "charged_count",
            "history_sha256",
            "eligibility_receipt_sha256",
            "prefix_sha256",
            "selected_prefix_positions",
            "selected_sequences",
        },
        "eligible driver selection schema differs",
    )
    status = document["status"]
    require(
        type(status) is str
        and status in STATUSES
        and type(document["round_index"]) is int
        and document["round_index"] == history.round_index
        and document["history_sha256"] == history.sha256
        and document["eligibility_receipt_sha256"] == eligibility.receipt_sha256
        and document["prefix_sha256"] == (None if prefix is None else prefix.output_sha256),
        "eligible driver selection identity differs",
    )
    positions, sequences = document["selected_prefix_positions"], document["selected_sequences"]
    require(type(positions) is list and type(sequences) is list, "eligible driver seats differ")
    complete_adaptive = (
        history.complete
        and history.round_index <= 28
        and document["charged_count"] == len(history.observations)
    )
    if status == "ready":
        require(
            complete_adaptive
            and prefix is not None
            and prefix.status == "complete"
            and len(positions) == len(sequences) == 14
            and all(type(value) is int and 0 <= value < 256 for value in positions)
            and positions == sorted(set(positions))
            and sequences == [prefix.accepted_sequences[value] for value in positions],
            "eligible driver ready seats do not reconstruct",
        )
    else:
        require(not positions and not sequences, "eligible driver non-ready state exposes seats")
    if status == "paused_incomplete_wave":
        require(
            (not history.complete or document["charged_count"] > len(history.observations))
            and prefix is None,
            "eligible driver incomplete-wave pause differs",
        )
    elif status == "budget_complete_pending_controller_terminal":
        require(
            history.round_index == 29
            and history.complete
            and document["charged_count"] == 512
            and prefix is None,
            "eligible driver terminal denominator differs",
        )
    elif status == "stopped_deadline":
        require(prefix is None or prefix.status == "stopped_deadline", "stopped wrapper differs")
    elif status != "ready":
        expected = {
            "paused_prefix": "in_progress",
            "abstained_no_eligible_parent": "abstained_no_eligible_parent",
            "abstained_incomplete_prefix": "attempt_cap_exhausted",
            "abstained_insufficient_seats": "complete",
        }[status]
        require(
            complete_adaptive and prefix is not None and prefix.status == expected,
            "eligible driver prefix status differs",
        )
        if status == "abstained_no_eligible_parent":
            require(not eligibility.query_ids, "eligible driver empty-parent abstention differs")


def reconstruct_eligible_ga_driver(root, context, eligibility_reconstructor, *, checkpoint=None):
    """Independently reconstruct each phase; optional checkpoint charges runtime replay."""
    checkpoint = (lambda: None) if checkpoint is None else checkpoint
    checkpoint()
    check_sources(context)
    check_resolver(context, eligibility_reconstructor)
    context_bytes = canonical_json_bytes(asdict(context))
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "eligible driver root is not a real directory")
    entries = sorted(root.iterdir())
    require(len(entries) <= MAX_PHASES, "eligible driver phase count exceeded")
    records, cumulative = [], 0
    for index, path in enumerate(entries):
        checkpoint()
        require(
            path.name == f"phase-{index:06d}" and path.is_dir() and not path.is_symlink(),
            "eligible driver inventory contains a partial, gap or unexpected entry",
        )
        physical = 0
        for child in path.iterdir():
            metadata = child.lstat()
            require(
                child.name in (*PAYLOADS, "receipt.json", "SHA256SUMS")
                and stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1,
                "eligible driver payload is unexpected or nonregular",
            )
            physical += metadata.st_size
        require(physical <= MAX_PHASE_BYTES, "eligible driver phase byte cap exceeded")
        predecessors = (
            {} if not records else {"previous_driver_phase": records[-1].seal.seal_sha256}
        )
        seal = seals.verify_phase(
            path,
            expected_artifact=ARTIFACT,
            expected_payload_paths=PAYLOADS,
            expected_predecessor_seals=predecessors,
        )
        require(seal.metadata_json == seals.canonical_json_bytes({}), "unexpected phase metadata")
        captured = dict(seal.payload_bytes)
        exact_bytes = encoded_phase_size(captured, predecessors)
        cumulative += exact_bytes
        require(
            physical == exact_bytes <= MAX_PHASE_BYTES and cumulative <= MAX_RUN_BYTES,
            "eligible driver physical byte inventory/cap differs",
        )
        require(
            captured["context.json"] == context_bytes, "eligible driver persisted context differs"
        )
        history = legacy._history(legacy._json(captured["history.json"]))
        recorded = decode_eligibility(legacy._json(captured["eligibility.json"]))
        prefix = decode_prefix(legacy._json(captured["prefix.json"]))
        document = legacy._json(captured["selection.json"])
        expected = resolve_eligibility(context, history, eligibility_reconstructor)
        expected_bytes = canonical_json_bytes(expected.document())
        require(
            captured["eligibility.json"] == expected_bytes, "recorded external authority differs"
        )
        checkpoint()
        validate_growth(tuple(records), history, document["charged_count"])
        prior = [row for row in records if row.history.sha256 == history.sha256]
        require(
            all(row.eligibility == recorded for row in prior),
            "eligible driver changed same-history applicability",
        )
        if prefix is not None:
            _verify_with_checkpoint(
                prefix,
                kernel_input(context, history, expected),
                expected_kernel_source_sha256=context.kernel_source_sha256,
                expected_contract_sha256=context.kernel_contract_sha256,
                expected_deadline_monotonic=context.deadline_monotonic,
                checkpoint=checkpoint,
                **expectation_pins(context, history, expected),
            )
        previous_prefixes = [row.prefix for row in prior if row.prefix is not None]
        verify_actual_predecessor(prefix, previous_prefixes[-1] if previous_prefixes else None)
        validate_selection(history, recorded, prefix, document)
        earlier_ready = [
            row
            for row in records
            if row.history.round_index == history.round_index and row.result.status == "ready"
        ]
        if document["status"] == "ready" and earlier_ready:
            require(
                tuple(document["selected_sequences"])
                == earlier_ready[-1].result.selected_sequences,
                "eligible driver changed already recorded same-round seats",
            )
        require(
            canonical_json_bytes(
                resolve_eligibility(context, history, eligibility_reconstructor).document()
            )
            == expected_bytes,
            "external authority changed during independent replay",
        )
        require(
            canonical_json_bytes(asdict(history)) == captured["history.json"]
            and canonical_json_bytes(recorded.document()) == captured["eligibility.json"]
            and canonical_json_bytes(None if prefix is None else asdict(prefix))
            == captured["prefix.json"],
            "eligible driver captured history/authority/prefix drifted during replay",
        )
        checkpoint()
        result = EligibleGADriverResult(
            seal.seal_sha256,
            document["status"],
            history.round_index,
            document["charged_count"],
            history.sha256,
            recorded.receipt_sha256,
            document["prefix_sha256"],
            tuple(document["selected_prefix_positions"]),
            tuple(document["selected_sequences"]),
        )
        records.append(
            ReconstructedEligibleGAPhase(seal, history, recorded, prefix, result, exact_bytes)
        )
    check_sources(context)
    check_resolver(context, eligibility_reconstructor)
    require(
        canonical_json_bytes(asdict(context)) == context_bytes, "eligible driver context drifted"
    )
    require(
        all(
            canonical_json_bytes(asdict(row.history)) == row.seal.read_payload_bytes("history.json")
            and canonical_json_bytes(row.eligibility.document())
            == row.seal.read_payload_bytes("eligibility.json")
            and canonical_json_bytes(None if row.prefix is None else asdict(row.prefix))
            == row.seal.read_payload_bytes("prefix.json")
            for row in records
        ),
        "eligible driver prior captured state drifted",
    )
    require(sorted(root.iterdir()) == entries, "eligible driver inventory changed during replay")
    checkpoint()
    return tuple(records)


def private_positions(prefix, private):
    require(
        type(private) is legacy.PrivateGACollisions,
        "eligible driver private inventory type differs",
    )
    private.__post_init__()
    require(prefix.status == "complete", "private composition requires a complete wrapper")
    blocked = set(private.charged_sequence_keys) | set(private.upcoming_reserve_sequence_keys)
    return controller_first_available_prefix_positions(
        tuple(sequence_key(value) not in blocked for value in prefix.accepted_sequences),
        seat_count=14,
    )


def verify_eligible_private_composition(phase, private):
    require(
        type(phase) is ReconstructedEligibleGAPhase, "eligible private audit phase type differs"
    )
    require(
        type(private) is legacy.PrivateGACollisions, "eligible private audit inventory type differs"
    )
    private.__post_init__()
    require(
        len(private.charged_sequence_keys) == phase.result.charged_count
        and {sequence_key(row.sequence) for row in phase.history.observations}
        <= set(private.charged_sequence_keys),
        "eligible private audit charged inventory differs",
    )
    require(
        phase.prefix is not None and phase.prefix.status == "complete", "private audit lacks prefix"
    )
    if phase.result.status == "ready":
        require(
            private_positions(phase.prefix, private) == phase.result.selected_prefix_positions,
            "eligible private first-available composition differs",
        )
    elif phase.result.status == "abstained_insufficient_seats":
        try:
            private_positions(phase.prefix, private)
        except ValueError:
            return
        raise ValueError("private inventory does not support insufficient seats")
    else:
        raise ValueError("phase has no private composition decision")

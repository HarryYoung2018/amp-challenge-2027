"""Independent eligible-population/stream/edit replay, without proposer imports."""

from __future__ import annotations

import math

from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    NEW_CONTRACT_SHA256,
    EligibleGAPrefix,
    check_expected_input,
    eligible_batch_digest,
    eligible_implementation_sha256,
)
from amp_challenge.generators.search.peptide_ga_records import ArchiveIndividual
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    CONTRACT_SHA256,
    OBJECTIVES,
    PREFIX_SIZE,
    GAAttempt,
    GAEdit,
    batch_digest,
    canonical_json_bytes,
    digest,
    hash_string,
    parameters,
    require,
    sequence_key,
)
from amp_challenge.generators.search.peptide_ga_verifier import _replay_edit, _ReplayRNG


def _reconstruct(result, input_record, checkpoint):
    """Separate numerical reconstruction; checkpoint charges inline resume work."""
    history = input_record.history
    ids = input_record.eligibility.query_ids
    rows, contextual_rows, population = [], [], []
    for row in history.observations:
        checkpoint()
        eligible = row.query_id in ids
        numerical = {
            "charge_index": row.charge_index,
            "sequence": row.sequence,
            "status": row.status,
            "objectives": row.objectives if eligible else None,
        }
        rows.append(numerical)
        contextual_rows.append({**numerical, "eligible": eligible})
        if eligible:
            means = tuple((OBJECTIVES[index], row.objectives[index]) for index in range(2))
            population.append(
                ArchiveIndividual(
                    row.sequence,
                    sequence_key(row.sequence),
                    row.query_id,
                    (row.query_id,),
                    means,
                    math.fsum(0.5 * value for _, value in means),
                )
            )
    stream = digest(
        b"amp/tunable-peptide-ga/semantic-stream/v2\0"
        + canonical_json_bytes(
            {
                "configuration_id": input_record.configuration_id,
                "root_seed": history.seed,
                "round_index": history.round_index,
                "observations": rows,
                "training_sequence_keys": input_record.training_sequence_keys,
                "contract_sha256": CONTRACT_SHA256,
            }
        )
    )
    semantic = digest(
        b"amp/context-eligible-peptide-ga/semantic/v3\0"
        + canonical_json_bytes(
            {
                "objective_context_sha256": history.objective_context_sha256,
                "configuration_id": input_record.configuration_id,
                "training_sequence_keys": input_record.training_sequence_keys,
                "seed": history.seed,
                "round_index": history.round_index,
                "observations": contextual_rows,
            }
        )
    )
    seed_hash = digest(
        b"amp/tunable-peptide-ga/round-seed/v2\0"
        + canonical_json_bytes({"root_seed": history.seed, "round_index": history.round_index})
    )
    seed = int.from_bytes(bytes.fromhex(seed_hash)[:8], "big") & (2**63 - 1)
    require(
        result.input_sha256 == input_record.sha256
        and result.semantic_sha256 == semantic == input_record.semantic_sha256
        and stream == input_record.stream_sha256
        and seed == input_record.round_seed,
        "eligible GA independently reconstructed input/semantic identity differs",
    )
    special = (
        "paused_incomplete_history"
        if not history.complete
        else "budget_complete_pending_controller_terminal"
        if history.round_index == 29
        else "abstained_no_eligible_parent"
        if not population
        else None
    )
    if result.prefix is None:
        require(
            result.status in (special, "stopped_deadline"),
            "eligible GA special status is unsupported by raw history",
        )
        return
    require(special is None, "eligible GA produced attempts without eligible complete input")
    batch = result.prefix
    require(
        batch.input_sha256 == input_record.sha256
        and batch.stream_sha256 == stream
        and batch.round_seed == seed
        and batch.configuration_id == input_record.configuration_id
        and batch.output_sha256 == batch_digest(batch),
        "eligible GA inner batch binding differs",
    )
    population = tuple(sorted(population, key=lambda item: (-item.fitness, item.sequence_key)))
    config = parameters(input_record.configuration_id)
    submitted = frozenset(sequence_key(row.sequence) for row in history.observations)
    training = frozenset(input_record.training_sequence_keys)
    seen, accepted = {}, []
    for index, actual in enumerate(batch.attempts):
        checkpoint()
        actual.__post_init__()
        require(len(accepted) < PREFIX_SIZE, "eligible GA attempt follows complete prefix")
        sequence, operator, parents, description, factors = _replay_edit(
            population, config, _ReplayRNG(seed, stream, index)
        )
        key = sequence_key(sequence)
        reason = (
            "length_out_of_support"
            if not 8 <= len(sequence) <= 50
            else "exact_training_overlap"
            if key in training
            else "previously_submitted_collision"
            if key in submitted
            else "generated_duplicate"
            if key in seen
            else None
        )
        expected = GAAttempt(
            index,
            GAEdit(sequence, operator, parents, description, factors),
            reason,
            None if reason else len(accepted),
            seen.get(key, index),
        )
        require(actual == expected, "eligible GA independent edit/decision replay differs")
        seen.setdefault(key, index)
        if reason is None:
            accepted.append(sequence)
    expected_status = (
        "complete"
        if len(accepted) == PREFIX_SIZE
        else "attempt_cap_exhausted"
        if len(batch.attempts) == ATTEMPT_CAP
        else "in_progress"
    )
    require(
        batch.accepted_sequences == tuple(accepted) and batch.status == expected_status,
        "eligible GA accepted order/count or inner status differs",
    )


def _verify_with_checkpoint(
    result,
    input_record,
    *,
    expected_kernel_source_sha256,
    expected_contract_sha256,
    expected_eligibility_source_sha256,
    expected_objective_context_sha256,
    expected_history_sha256,
    expected_eligible_query_ids,
    expected_deadline_monotonic,
    checkpoint,
):
    require(type(result) is EligibleGAPrefix, "eligible GA wrapper type differs")
    result.__post_init__()
    require(
        hash_string(expected_kernel_source_sha256)
        and expected_contract_sha256 == NEW_CONTRACT_SHA256
        and type(expected_deadline_monotonic) in (float, int)
        and math.isfinite(expected_deadline_monotonic)
        and result.deadline_monotonic == expected_deadline_monotonic,
        "eligible GA external source/contract/original deadline differs",
    )
    pins = {
        "expected_history_sha256": expected_history_sha256,
        "expected_objective_context_sha256": expected_objective_context_sha256,
        "expected_eligibility_source_sha256": expected_eligibility_source_sha256,
        "expected_eligible_query_ids": expected_eligible_query_ids,
    }
    check_expected_input(input_record, **pins)
    checkpoint()
    original_input = input_record.sha256
    original_output = result.canonical_bytes()
    require(
        result.source_sha256 == expected_kernel_source_sha256 == eligible_implementation_sha256()
        and result.output_sha256 == eligible_batch_digest(result),
        "eligible GA actual source/output seal differs",
    )
    checkpoint()
    _reconstruct(result, input_record, checkpoint)
    checkpoint()
    require(result.output_sha256 == eligible_batch_digest(result), "eligible GA output drifted")
    require(result.canonical_bytes() == original_output, "eligible GA encoded result drifted")
    check_expected_input(input_record, **pins)
    require(input_record.sha256 == original_input, "eligible GA input changed during replay")
    require(
        eligible_implementation_sha256() == expected_kernel_source_sha256,
        "eligible GA source changed during replay/encoding",
    )
    checkpoint()


def verify_eligible_prefix(
    result,
    input_record,
    *,
    expected_kernel_source_sha256,
    expected_contract_sha256,
    expected_eligibility_source_sha256,
    expected_objective_context_sha256,
    expected_history_sha256,
    expected_eligible_query_ids,
    expected_deadline_monotonic,
):
    """Offline bounded-CPU audit; does not authenticate elapsed time or oracle truth."""
    _verify_with_checkpoint(
        result,
        input_record,
        expected_kernel_source_sha256=expected_kernel_source_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_eligibility_source_sha256=expected_eligibility_source_sha256,
        expected_objective_context_sha256=expected_objective_context_sha256,
        expected_history_sha256=expected_history_sha256,
        expected_eligible_query_ids=expected_eligible_query_ids,
        expected_deadline_monotonic=expected_deadline_monotonic,
        checkpoint=lambda: None,
    )

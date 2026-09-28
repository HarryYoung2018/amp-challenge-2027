"""Independent replay of the tunable kernel; does not import its proposer.

Reuses the separately implemented, reviewed v1 numerical replay/edit path,
not the v1 or v2 proposer. Same-account structural consistency is the claim;
external oracle-response authenticity is the future controller's responsibility.
"""

from __future__ import annotations

import math

from amp_challenge.generators.search.peptide_ga_records import ArchiveIndividual, sequence_key
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    OBJECTIVES,
    PREFIX_SIZE,
    GAAttempt,
    GAEdit,
    GAKernelBatch,
    GAKernelInput,
    batch_digest,
    parameters,
    require,
)
from amp_challenge.generators.search.peptide_ga_verifier import _replay_edit, _ReplayRNG


def verify_prefix(batch: GAKernelBatch, input_record: GAKernelInput) -> None:
    require(
        type(input_record) is GAKernelInput and type(batch) is GAKernelBatch,
        "kernel replay types differ",
    )
    input_record.__post_init__()
    batch.__post_init__()
    require(
        batch.input_sha256 == input_record.sha256
        and batch.stream_sha256 == input_record.stream_sha256
        and batch.configuration_id == input_record.configuration_id
        and batch.round_seed == input_record.round_seed,
        "kernel replay input binding differs",
    )
    require(batch.output_sha256 == batch_digest(batch), "kernel output digest differs")
    # Build the fitness archive here instead of sharing the proposer's archive
    # projection function. Failed charges remain in the submitted exclusion set.
    population = []
    for record in input_record.observations:
        if record.status == "successful":
            assert record.objectives is not None
            means = tuple((OBJECTIVES[index], record.objectives[index]) for index in range(2))
            population.append(
                ArchiveIndividual(
                    record.sequence,
                    sequence_key(record.sequence),
                    record.query_id,
                    (record.query_id,),
                    means,
                    math.fsum(value * 0.5 for _, value in means),
                )
            )
    population = tuple(sorted(population, key=lambda item: (-item.fitness, item.sequence_key)))
    config = parameters(input_record.configuration_id)
    submitted = frozenset(sequence_key(item.sequence) for item in input_record.observations)
    training = frozenset(input_record.training_sequence_keys)
    seen, accepted = {}, []
    for index, actual in enumerate(batch.attempts):
        actual.__post_init__()
        require(len(accepted) < PREFIX_SIZE, "attempt appended after prefix completed")
        replay = _ReplayRNG(batch.round_seed, batch.stream_sha256, index)
        sequence, operator, parents, description, factors = _replay_edit(population, config, replay)
        key = sequence_key(sequence)
        if len(sequence) < 8 or len(sequence) > 50:
            reason = "length_out_of_support"
        elif key in training:
            reason = "exact_training_overlap"
        elif key in submitted:
            reason = "previously_submitted_collision"
        elif key in seen:
            reason = "generated_duplicate"
        else:
            reason = None
        expected = GAAttempt(
            index,
            GAEdit(sequence, operator, parents, description, factors),
            reason,
            None if reason else len(accepted),
            seen.get(key, index),
        )
        require(
            actual == expected, "attempt differs from independently reconstructed edit/decision"
        )
        seen.setdefault(key, index)
        if reason is None:
            accepted.append(sequence)
    require(batch.accepted_sequences == tuple(accepted), "retained offspring order/count differs")
    expected_status = (
        "complete"
        if len(accepted) == PREFIX_SIZE
        else "attempt_cap_exhausted"
        if len(batch.attempts) == ATTEMPT_CAP
        else "in_progress"
    )
    require(batch.status == expected_status, "reconstructed status differs")

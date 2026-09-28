"""Bounded eligible GA edits; raw charges are retained, never status-projected."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from amp_challenge.generators.search.peptide_ga_eligible_v3_records import (
    NEW_CONTRACT_SHA256,
    EligibleGAPrefix,
    check_expected_input,
    eligible_batch_digest,
    eligible_implementation_sha256,
)
from amp_challenge.generators.search.peptide_ga_eligible_v3_verify import _verify_with_checkpoint
from amp_challenge.generators.search.peptide_ga_tunable_v2 import propose_edit
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    PREFIX_SIZE,
    GAAttempt,
    GAKernelBatch,
    batch_digest,
    hash_string,
    parameters,
    require,
    sequence_key,
)


class _Deadline(Exception):
    pass


def generate_eligible_prefix(
    input_record,
    *,
    expected_kernel_source_sha256,
    expected_eligibility_source_sha256,
    expected_objective_context_sha256,
    expected_history_sha256,
    expected_eligible_query_ids,
    deadline_monotonic,
    clock=time.monotonic,
    resume=None,
    max_new_attempts=None,
):
    """Use the caller's original absolute deadline, including replay and encoding.

    Source/history/subset drift raises without releasing a result. Deadline stops
    retain a nonauthorizing prefix only as evidence. Blocking calls are not
    hard-preempted, and this seam performs no oracle or durable controller action.
    """
    require(
        type(deadline_monotonic) in (float, int)
        and math.isfinite(deadline_monotonic)
        and callable(clock)
        and hash_string(expected_kernel_source_sha256),
        "eligible GA original deadline/clock/source differs",
    )
    require(
        max_new_attempts is None
        or (type(max_new_attempts) is int and 0 <= max_new_attempts <= ATTEMPT_CAP),
        "eligible GA attempt chunk differs",
    )
    pins = {
        "expected_history_sha256": expected_history_sha256,
        "expected_objective_context_sha256": expected_objective_context_sha256,
        "expected_eligibility_source_sha256": expected_eligibility_source_sha256,
        "expected_eligible_query_ids": expected_eligible_query_ids,
    }
    check_expected_input(input_record, **pins)
    input_sha, semantic, stream, seed = (
        input_record.sha256,
        input_record.semantic_sha256,
        input_record.stream_sha256,
        input_record.round_seed,
    )

    def bindings():
        require(
            eligible_implementation_sha256() == expected_kernel_source_sha256,
            "eligible GA executable source differs or changed",
        )
        check_expected_input(input_record, **pins)
        require(input_record.sha256 == input_sha, "eligible GA raw input changed during execution")

    def checkpoint():
        current = clock()
        require(
            type(current) in (float, int) and math.isfinite(current),
            "eligible GA clock returned a nonfinite/non-numeric value",
        )
        if current >= deadline_monotonic:
            raise _Deadline

    bindings()
    attempts, accepted, seen = [], [], {}
    active = False
    status = "stopped_deadline"
    prior_sha, prior_count, prior_completed = None, 0, False

    def inner():
        if not active:
            return None
        state = (
            "complete"
            if len(accepted) == PREFIX_SIZE
            else "attempt_cap_exhausted"
            if len(attempts) == ATTEMPT_CAP
            else "in_progress"
        )
        batch = GAKernelBatch(
            input_sha,
            stream,
            input_record.configuration_id,
            seed,
            tuple(attempts),
            tuple(accepted),
            state,
            "0" * 64,
        )
        return replace(batch, output_sha256=batch_digest(batch))

    def encode(state):
        result = EligibleGAPrefix(
            input_sha,
            semantic,
            expected_kernel_source_sha256,
            NEW_CONTRACT_SHA256,
            deadline_monotonic,
            state,
            inner(),
            prior_sha,
            prior_count,
            prior_completed,
            "0" * 64,
        )
        result = replace(result, output_sha256=eligible_batch_digest(result))
        result.canonical_bytes()
        return result

    if resume is not None:
        require(
            type(resume) is EligibleGAPrefix
            and resume.status == "in_progress"
            and resume.deadline_monotonic == deadline_monotonic,
            "eligible GA resume requires in-progress prefix and unchanged original deadline",
        )
        resume.__post_init__()
        require(
            resume.input_sha256 == input_sha
            and resume.semantic_sha256 == semantic
            and resume.source_sha256 == expected_kernel_source_sha256
            and resume.contract_sha256 == NEW_CONTRACT_SHA256
            and resume.output_sha256 == eligible_batch_digest(resume),
            "eligible GA prior resume structural/header/seal binding differs",
        )
        # These are retained claims until the numerical replay below completes.
        prior_sha, prior_count = resume.output_sha256, len(resume.prefix.attempts)
    try:
        checkpoint()
        if resume is not None:
            _verify_with_checkpoint(
                resume,
                input_record,
                expected_kernel_source_sha256=expected_kernel_source_sha256,
                expected_contract_sha256=NEW_CONTRACT_SHA256,
                expected_deadline_monotonic=deadline_monotonic,
                checkpoint=checkpoint,
                **pins,
            )
            attempts = list(resume.prefix.attempts)
            accepted = list(resume.prefix.accepted_sequences)
            seen = {sequence_key(row.edit.sequence): row.first_attempt_index for row in attempts}
            active = True
            prior_completed = True
        checkpoint()
        history = input_record.history
        if not history.complete:
            status = "paused_incomplete_history"
        elif history.round_index == 29:
            status = "budget_complete_pending_controller_terminal"
        elif not input_record.eligibility.query_ids:
            status = "abstained_no_eligible_parent"
        else:
            active = True
            population = input_record.eligible_population()
            config = parameters(input_record.configuration_id)
            submitted = {sequence_key(row.sequence) for row in history.observations}
            training = frozenset(input_record.training_sequence_keys)
            stop = min(
                ATTEMPT_CAP,
                len(attempts) + (ATTEMPT_CAP if max_new_attempts is None else max_new_attempts),
            )
            while len(accepted) < PREFIX_SIZE and len(attempts) < stop:
                checkpoint()
                index = len(attempts)
                edit = propose_edit(
                    population, config, seed=seed, stream_sha256=stream, attempt_index=index
                )
                key = sequence_key(edit.sequence)
                reason = (
                    "length_out_of_support"
                    if not 8 <= len(edit.sequence) <= 50
                    else "exact_training_overlap"
                    if key in training
                    else "previously_submitted_collision"
                    if key in submitted
                    else "generated_duplicate"
                    if key in seen
                    else None
                )
                attempts.append(
                    GAAttempt(
                        index, edit, reason, None if reason else len(accepted), seen.get(key, index)
                    )
                )
                seen.setdefault(key, index)
                if reason is None:
                    accepted.append(edit.sequence)
            status = inner().status
        checkpoint()
        bindings()
        checkpoint()
        result = encode(status)
        bindings()
        checkpoint()
        return result
    except _Deadline:
        bindings()
        result = encode("stopped_deadline")
        bindings()
        # Late evidence is never usable. Still reject malformed clock/drift after
        # stop encoding; do not recursively serialize another deadline receipt.
        current = clock()
        require(
            type(current) in (float, int) and math.isfinite(current),
            "eligible GA stopped clock differs",
        )
        return result

"""Tunable real peptide-GA operators; no oracle client or query authority.

Parent selection and counter RNG reuse the reviewed v1 numerical primitives.
The four edits below preserve the existing v1 proposer semantics. Its frozen
configuration, archive, publication and scientific interfaces are not changed.
"""

from __future__ import annotations

from dataclasses import replace

from amp_challenge.generators.search.peptide_ga import _categorical, _CounterRNG, _select_parent
from amp_challenge.generators.search.peptide_ga_records import ArchiveIndividual, sequence_key
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    ATTEMPT_CAP,
    PREFIX_SIZE,
    GAAttempt,
    GAEdit,
    GAKernelBatch,
    GAKernelInput,
    TunableGAParameters,
    batch_digest,
    hash_string,
    parameters,
    population_from_charged,
    require,
)
from amp_challenge.generators.search.records import ProbabilityFactor


def propose_edit(
    population: tuple[ArchiveIndividual, ...],
    config: TunableGAParameters,
    *,
    seed: int,
    stream_sha256: str,
    attempt_index: int,
) -> GAEdit:
    """Pure numeric edit, also exposed for explicit-stream legacy parity tests."""
    require(type(config) is TunableGAParameters, "parameter type differs")
    config.__post_init__()
    require(
        type(population) is tuple
        and 1 <= len(population) <= 512
        and all(type(row) is ArchiveIndividual for row in population),
        "numeric parent population differs",
    )
    for row in population:
        row.__post_init__()
    require(
        tuple(sorted(population, key=lambda row: (-row.fitness, row.sequence_key))) == population
        and len({row.sequence_key for row in population}) == len(population),
        "numeric parent ordering/uniqueness differs",
    )
    require(
        type(seed) is int
        and 0 <= seed < 2**63
        and hash_string(stream_sha256)
        and type(attempt_index) is int
        and 0 <= attempt_index < ATTEMPT_CAP,
        "numeric random stream differs",
    )
    return _propose_edit(
        population, config, seed=seed, stream_sha256=stream_sha256, attempt_index=attempt_index
    )


def _propose_edit(population, config, *, seed, stream_sha256, attempt_index):
    rng = _CounterRNG(seed=seed, input_sha256=stream_sha256, attempt_index=attempt_index)
    parent, factors = _select_parent(population, config, rng, label="parent.0")
    available = tuple(
        (name, rate)
        for name, rate in config.operator_rates
        if not (
            (name == "insertion" and len(parent.sequence) >= config.max_length)
            or (name == "deletion" and len(parent.sequence) <= config.min_length)
            or (name == "two_parent_crossover" and len(population) < 2)
        )
    )
    operator, probability = _categorical(rng, available, label="operator")
    factors = [*factors, ProbabilityFactor("operator", probability)]
    parents = (parent.sequence_key,)
    if operator == "substitution":
        position = rng.below(len(parent.sequence), "edit.position")
        residues = config.alphabet.replace(parent.sequence[position], "")
        residue = residues[rng.below(len(residues), "edit.residue")]
        child = parent.sequence[:position] + residue + parent.sequence[position + 1 :]
        description = (operator, position, parent.sequence[position], residue)
        factors += [
            ProbabilityFactor("edit.position", 1 / len(parent.sequence)),
            ProbabilityFactor("edit.residue", 1 / len(residues)),
        ]
    elif operator == "insertion":
        position = rng.below(len(parent.sequence) + 1, "edit.position")
        residue = config.alphabet[rng.below(len(config.alphabet), "edit.residue")]
        child = parent.sequence[:position] + residue + parent.sequence[position:]
        description = (operator, position, residue)
        factors += [
            ProbabilityFactor("edit.position", 1 / (len(parent.sequence) + 1)),
            ProbabilityFactor("edit.residue", 1 / len(config.alphabet)),
        ]
    elif operator == "deletion":
        position = rng.below(len(parent.sequence), "edit.position")
        child = parent.sequence[:position] + parent.sequence[position + 1 :]
        description = (operator, position, parent.sequence[position])
        factors.append(ProbabilityFactor("edit.position", 1 / len(parent.sequence)))
    else:
        second, second_factors = _select_parent(
            population, config, rng, label="parent.1", forbidden_key=parent.sequence_key
        )
        left = 1 + rng.below(len(parent.sequence) - 1, "edit.cut.0")
        right = 1 + rng.below(len(second.sequence) - 1, "edit.cut.1")
        child = parent.sequence[:left] + second.sequence[right:]
        parents = (parent.sequence_key, second.sequence_key)
        description = (operator, left, right, *parents)
        factors += [
            *second_factors,
            ProbabilityFactor("edit.cut.0", 1 / (len(parent.sequence) - 1)),
            ProbabilityFactor("edit.cut.1", 1 / (len(second.sequence) - 1)),
        ]
    return GAEdit(child, operator, parents, description, tuple(factors))


def generate_prefix(
    input_record: GAKernelInput,
    *,
    resume: GAKernelBatch | None = None,
    max_new_attempts: int | None = None,
) -> GAKernelBatch:
    require(type(input_record) is GAKernelInput, "kernel input type differs")
    input_record.__post_init__()
    if max_new_attempts is not None:
        require(
            type(max_new_attempts) is int and 0 <= max_new_attempts <= ATTEMPT_CAP,
            "attempt limit differs",
        )
    input_sha256 = input_record.sha256
    stream_sha256 = input_record.stream_sha256
    seed = input_record.round_seed
    attempts, accepted = [], []
    if resume is not None:
        require(type(resume) is GAKernelBatch, "resume type differs")
        resume.__post_init__()
        require(resume.status == "in_progress", "only an incomplete prefix may resume")
        replayed = generate_prefix(input_record, max_new_attempts=resume.next_attempt_index)
        require(replayed == resume, "resume prefix differs from exact replay")
        attempts, accepted = list(resume.attempts), list(resume.accepted_sequences)
    start = len(attempts)
    stop = ATTEMPT_CAP if max_new_attempts is None else min(ATTEMPT_CAP, start + max_new_attempts)
    training = frozenset(input_record.training_sequence_keys)
    submitted = {sequence_key(row.sequence) for row in input_record.observations}
    seen = {}
    for row in attempts:
        seen.setdefault(sequence_key(row.edit.sequence), row.attempt_index)
    population = population_from_charged(input_record)
    config = parameters(input_record.configuration_id)
    for index in range(start, stop):
        if len(accepted) == PREFIX_SIZE:
            break
        edit = _propose_edit(
            population, config, seed=seed, stream_sha256=stream_sha256, attempt_index=index
        )
        key = sequence_key(edit.sequence)
        if not 8 <= len(edit.sequence) <= 50:
            rejection = "length_out_of_support"
        elif key in training:
            rejection = "exact_training_overlap"
        elif key in submitted:
            rejection = "previously_submitted_collision"
        elif key in seen:
            rejection = "generated_duplicate"
        else:
            rejection = None
        position = len(accepted) if rejection is None else None
        attempts.append(GAAttempt(index, edit, rejection, position, seen.get(key, index)))
        seen.setdefault(key, index)
        if rejection is None:
            accepted.append(edit.sequence)
    status = (
        "complete"
        if len(accepted) == PREFIX_SIZE
        else "attempt_cap_exhausted"
        if len(attempts) == ATTEMPT_CAP
        else "in_progress"
    )
    batch = GAKernelBatch(
        input_sha256,
        stream_sha256,
        config.configuration_id,
        seed,
        tuple(attempts),
        tuple(accepted),
        status,
        "0" * 64,
    )
    return replace(batch, output_sha256=batch_digest(batch))

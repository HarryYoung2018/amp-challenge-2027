"""Clean-room verifier for peptide-GA batches (intentionally no proposer import)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from fractions import Fraction
from pathlib import Path

from amp_challenge.generators.search.campaign_ledger import (
    EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    verify_campaign,
)
from amp_challenge.generators.search.peptide_ga_exclusion_authority import (
    load_collision_exclusion_authority_from_paths,
)
from amp_challenge.generators.search.peptide_ga_policy_registry import (
    load_selection_policy_registry_from_path,
    validate_campaign_proposal_policy_binding,
)
from amp_challenge.generators.search.peptide_ga_records import (
    PEPTIDE_GA_ARTIFACT,
    PEPTIDE_GA_CAMPAIGN_CONFIGURATION_ID,
    PEPTIDE_GA_CONFIG_MAX_BYTES,
    PEPTIDE_GA_MAX_QUERIES,
    PEPTIDE_GA_MAX_RESPONSES,
    PEPTIDE_GA_MAX_ROUNDS,
    PEPTIDE_GA_PENDING_SELECTION_POLICY_VERSION,
    PEPTIDE_GA_POLICY_VERSION,
    PEPTIDE_GA_RNG_VERSION,
    PEPTIDE_GA_SIGNED_63_MAX,
    ArchiveIndividual,
    AuthenticatedWaveArchive,
    CollisionExclusions,
    FitnessTruthContract,
    PeptideGAAttempt,
    PeptideGABatch,
    PeptideGAConfig,
    PeptideGAError,
    canonical_json_bytes,
    make_public_exclusion_receipt,
    preflight_peptide_ga_batch_structure,
    public_exclusion_receipt_document,
    round_timing_receipt_inventory_sha256,
    sequence_key,
    sha256_bytes,
)
from amp_challenge.generators.search.records import ProbabilityFactor

_INPUT = b"amp/fixed-default-peptide-ga/input/v1\0"
_ARCHIVE = b"amp/fixed-default-peptide-ga/archive/v1\0"
_OUTPUT = b"amp/fixed-default-peptide-ga/output/v1\0"
_RNG = b"amp/fixed-default-peptide-ga/rng/v1\0"
_PATH_TEXT_MAX_LENGTH = 4096
_BUILTIN_PATH_TYPE = type(Path("."))
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_RNG_REJECTION_CAP = 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PeptideGAError(message)


def _exact_path(value: object, *, label: str) -> Path:
    _require(
        type(value) is str or type(value) is _BUILTIN_PATH_TYPE,
        f"{label} must use an exact built-in path type",
    )
    text = value if type(value) is str else str(value)
    assert isinstance(text, str)
    _require(
        0 < len(text) <= _PATH_TEXT_MAX_LENGTH and "\0" not in text,
        f"{label} length differs",
    )
    result = Path(text)
    _require(".." not in result.parts, f"{label} must not contain parent traversal")
    return result


def _resolved_path(value: Path, *, label: str) -> Path:
    try:
        return value.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PeptideGAError(f"{label} resolution failed") from error


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    assert isinstance(value, str)
    return value


def _identifier(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _IDENTIFIER_RE.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    assert isinstance(value, str)
    return value


def _require_registry_outside_campaign_root(
    root: str | Path,
    *,
    policy_registry_path: str | Path,
    trusted_policy_registry_parent: str | Path,
) -> None:
    """Keep controller-trusted policy authority outside the campaign output tree."""

    campaign_root = _exact_path(root, label="campaign root")
    registry_path = _exact_path(policy_registry_path, label="policy registry path")
    registry_parent = _exact_path(
        trusted_policy_registry_parent,
        label="trusted policy registry parent",
    )
    _require(
        campaign_root.is_absolute()
        and registry_path.is_absolute()
        and registry_parent.is_absolute(),
        "campaign and policy registry paths must be absolute",
    )
    resolved_campaign_root = _resolved_path(campaign_root, label="campaign root")
    resolved_registry_path = _resolved_path(registry_path, label="policy registry path")
    resolved_registry_parent = _resolved_path(
        registry_parent,
        label="trusted policy registry parent",
    )
    _require(
        not registry_path.is_relative_to(campaign_root)
        and not registry_parent.is_relative_to(campaign_root)
        and not resolved_registry_path.is_relative_to(resolved_campaign_root)
        and not resolved_registry_parent.is_relative_to(resolved_campaign_root),
        "policy registry must remain outside the campaign output root",
    )


def _bounded_integer(value: object, *, maximum: int, label: str) -> int:
    _require(type(value) is int and 0 <= value <= maximum, f"{label} differs")
    assert isinstance(value, int)
    return value


def _verify_trusted_config_source(
    config: PeptideGAConfig,
    *,
    config_path: str | Path,
    trusted_config_parent: str | Path,
    expected_config_sha256: str,
) -> None:
    _require(type(config) is PeptideGAConfig, "config type differs")
    config.__post_init__()
    _sha256(expected_config_sha256, label="expected config SHA-256")
    parent = _exact_path(trusted_config_parent, label="trusted config parent")
    target = _exact_path(config_path, label="trusted config path")
    _require(parent.is_absolute() and target.is_absolute(), "trusted config paths must be absolute")
    _require(target.parent == parent, "config is not a direct child of trusted config parent")
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError as error:
        raise PeptideGAError("trusted config parent stat failed") from error
    _require(
        stat.S_ISDIR(parent_stat.st_mode)
        and parent_stat.st_uid == os.geteuid()
        and parent_stat.st_mode & 0o022 == 0,
        "trusted config parent metadata differs",
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as error:
        raise PeptideGAError("trusted config parent open failed") from error
    try:
        directory_before = os.fstat(directory_fd)
        _require(
            _stat_identity(parent_stat) == _stat_identity(directory_before),
            "trusted config parent changed before opening",
        )
        try:
            entry_before = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
            file_fd = os.open(target.name, file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise PeptideGAError("trusted config open failed") from error
        try:
            before = os.fstat(file_fd)
            _require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and before.st_uid == os.geteuid()
                and before.st_mode & 0o022 == 0
                and 0 < before.st_size <= PEPTIDE_GA_CONFIG_MAX_BYTES,
                "trusted config metadata differs",
            )
            _require(
                _stat_identity(entry_before) == _stat_identity(before),
                "trusted config entry changed before reading",
            )
            payload_buffer = bytearray()
            while len(payload_buffer) <= PEPTIDE_GA_CONFIG_MAX_BYTES:
                chunk = os.read(
                    file_fd,
                    min(65_536, PEPTIDE_GA_CONFIG_MAX_BYTES + 1 - len(payload_buffer)),
                )
                if not chunk:
                    break
                payload_buffer.extend(chunk)
            _require(
                0 < len(payload_buffer) <= PEPTIDE_GA_CONFIG_MAX_BYTES,
                "trusted config size differs",
            )
            after = os.fstat(file_fd)
            _require(
                _stat_identity(before) == _stat_identity(after),
                "trusted config changed while reading",
            )
            try:
                entry_after = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise PeptideGAError("trusted config entry recheck failed") from error
            _require(
                _stat_identity(after) == _stat_identity(entry_after),
                "trusted config entry changed while reading",
            )
        finally:
            os.close(file_fd)
        directory_after = os.fstat(directory_fd)
        _require(
            _stat_identity(directory_before) == _stat_identity(directory_after),
            "trusted config parent changed while reading",
        )
    finally:
        os.close(directory_fd)
    payload = bytes(payload_buffer)
    _require(sha256_bytes(payload) == expected_config_sha256, "expected config SHA-256 differs")
    _require(config.config_sha256 == expected_config_sha256, "parsed config SHA-256 differs")
    _require(
        config.config_source_bytes == payload, "parsed config source differs from trusted file"
    )


def _finite_mean(values: list[float], *, objective: str) -> float:
    try:
        mean = math.fsum(values) / len(values)
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise PeptideGAError(f"non-finite aggregate for objective {objective}") from error
    _require(math.isfinite(mean), f"non-finite aggregate for objective {objective}")
    return mean


def _finite_fitness(means: tuple[tuple[str, float], ...], config: PeptideGAConfig) -> float:
    try:
        fitness = math.fsum(dict(means)[name] * weight for name, weight in config.objective_weights)
    except (OverflowError, ValueError) as error:
        raise PeptideGAError("non-finite scalarized fitness") from error
    _require(math.isfinite(fitness), "non-finite scalarized fitness")
    return fitness


def _individual(item: ArchiveIndividual) -> dict[str, object]:
    return {
        "first_proposal_id": item.first_proposal_id,
        "fitness": item.fitness,
        "objective_means": [list(row) for row in item.objective_means],
        "sequence": item.sequence,
        "sequence_key": item.sequence_key,
        "successful_evaluation_ids": list(item.successful_evaluation_ids),
    }


def _archive_document(value: AuthenticatedWaveArchive) -> dict[str, object]:
    return {
        "authentication_status": value.authentication_status,
        "campaign_id": value.campaign_id,
        "phase": value.phase,
        "elapsed_inventory_sha256": value.elapsed_inventory_sha256,
        "event_count": value.event_count,
        "event_inventory_sha256": value.event_inventory_sha256,
        "header_sha256": value.header_sha256,
        "head_seal_sha256": value.head_seal_sha256,
        "individuals": [_individual(item) for item in value.individuals],
        "last_event_sha256": value.last_event_sha256,
        "proposal_count": value.proposal_count,
        "query_count": value.query_count,
        "response_count": value.response_count,
        "round_seal_inventory_sha256": value.round_seal_inventory_sha256,
        "round_timing_receipt_inventory_sha256": (value.round_timing_receipt_inventory_sha256),
        "round_timing_receipt_sha256s": list(value.round_timing_receipt_sha256s),
        "round_count": value.round_count,
        "scientific_elapsed_ns": value.scientific_elapsed_ns,
        "truth_contract_sha256": value.truth_contract_sha256,
        "wave_id": value.wave_id,
    }


def _input_document(
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    batch: PeptideGABatch,
) -> dict[str, object]:
    return {
        "archive_sha256": archive.archive_sha256,
        "batch_id": batch.batch_id,
        "campaign_id": archive.campaign_id,
        "config_sha256": batch.config_sha256,
        "controller_private_reserve_identity_visible_to_adapter": False,
        "organizer_reference_input_present": False,
        "seed": batch.seed,
        "submitted_set_sha256": exclusions.submitted_set_sha256,
        "training_homology_exclusion_enforced": False,
        "training_set_sha256": exclusions.training_set_sha256,
        "wave_id": archive.wave_id,
    }


def _trace(value: object) -> list[dict[str, object]]:
    return [{"name": factor.name, "probability": factor.probability} for factor in value.factors]


def _attempt_document(item: PeptideGAAttempt) -> dict[str, object]:
    proposal = item.proposal
    selection = proposal.selection
    edge = item.edge
    event = item.provenance
    return {
        "accepted_position": item.accepted_position,
        "attempt_index": item.attempt_index,
        "edge": {
            "behavior_log_probabilities": list(edge.behavior_log_probabilities),
            "edge_id": edge.edge_id,
            "edit_description": list(edge.edit_description),
            "operator": edge.operator,
            "parent_sequence_keys": list(edge.parent_sequence_keys),
            "proposal_id": edge.proposal_id,
            "proposal_trace": _trace(edge.proposal_trace),
            "random_stream": edge.random_stream,
            "rollout_id": edge.rollout_id,
            "sample_index": edge.sample_index,
            "sampling_parameters": [list(row) for row in edge.sampling_parameters],
        },
        "proposal": {
            "cheap_predictions": [list(row) for row in proposal.cheap_predictions],
            "hard_valid": proposal.hard_valid,
            "niche_id": proposal.niche_id,
            "policy_version": proposal.policy_version,
            "proposal_id": proposal.proposal_id,
            "proposal_round": proposal.proposal_round,
            "rejection_reason": proposal.rejection_reason,
            "rollout_id": proposal.rollout_id,
            "selection": {
                "eligible_proposal_ids": list(selection.eligible_proposal_ids),
                "policy_version": selection.policy_version,
                "propensity": _trace(selection.propensity),
                "seed": selection.seed,
                "selected": selection.selected,
                "selection_set_id": selection.selection_set_id,
            },
            "sequence": proposal.sequence,
        },
        "provenance": {
            name: getattr(event, name)
            for name in (
                "event_index",
                "proposal_id",
                "edge_id",
                "sequence_key",
                "first_attempt_proposal_id",
                "duplicate_within_prefix",
                "hard_valid",
                "rejection_reason",
            )
        },
    }


def _output_document(batch: PeptideGABatch) -> dict[str, object]:
    return {
        "accepted_proposal_ids": list(batch.accepted_proposal_ids),
        "accepted_sequences": list(batch.accepted_sequences),
        "archive_sha256": batch.archive_sha256,
        "artifact": batch.artifact,
        "attempts": [_attempt_document(item) for item in batch.attempts],
        "authorization": {
            "automatic_production_eligible": batch.automatic_production_eligible,
            "biological_superiority_claim_allowed": batch.biological_superiority_claim_allowed,
            "execution_authorized": batch.execution_authorized,
            "scientific_evidence_accepted": batch.scientific_evidence_accepted,
        },
        "batch_id": batch.batch_id,
        "campaign_id": batch.campaign_id,
        "config_sha256": batch.config_sha256,
        "controller_private_reserve_seats_emitted": batch.controller_private_reserve_seats_emitted,
        "input_sha256": batch.input_sha256,
        "next_attempt_index": batch.next_attempt_index,
        "oracle_query_identities_constructed": batch.oracle_query_identities_constructed,
        "public_exclusion_receipt": public_exclusion_receipt_document(
            batch.public_exclusion_receipt,
            include_sha256=True,
        ),
        "schema_version": batch.schema_version,
        "seed": batch.seed,
        "status": batch.status,
        "wave_id": batch.wave_id,
    }


class _ReplayRNG:
    def __init__(self, seed: int, input_sha256: str, attempt: int) -> None:
        self.seed = seed
        self.input_sha256 = input_sha256
        self.attempt = attempt

    def below(self, limit: int, label: str) -> int:
        _require(limit > 0, "RNG support is empty")
        ceiling = (1 << 256) // limit * limit
        for counter in range(_RNG_REJECTION_CAP):
            document = {
                "attempt_index": self.attempt,
                "counter": counter,
                "input_sha256": self.input_sha256,
                "label": label,
                "seed": self.seed,
            }
            value = int.from_bytes(
                hashlib.sha256(_RNG + canonical_json_bytes(document)).digest(), "big"
            )
            if value < ceiling:
                return value % limit
        raise PeptideGAError("RNG replay rejection cap exhausted")


def _categorical(
    rng: _ReplayRNG, rows: tuple[tuple[str, float], ...], label: str
) -> tuple[str, float]:
    fractions = tuple((name, Fraction(str(weight))) for name, weight in rows)
    denominator = math.lcm(*(value.denominator for _, value in fractions))
    units = tuple(
        (name, value.numerator * (denominator // value.denominator)) for name, value in fractions
    )
    total = sum(value for _, value in units)
    draw = rng.below(total, label)
    cursor = 0
    for name, value in units:
        cursor += value
        if draw < cursor:
            return name, value / total
    raise AssertionError("categorical replay escaped support")


def _parent(
    population: tuple[ArchiveIndividual, ...],
    config: PeptideGAConfig,
    rng: _ReplayRNG,
    label: str,
    forbidden: str | None = None,
) -> tuple[ArchiveIndividual, tuple[ProbabilityFactor, ...]]:
    full = tuple(item for item in population if item.sequence_key != forbidden)
    elite_count = max(1, math.ceil(len(population) * config.elite_fraction))
    elite_keys = {item.sequence_key for item in population[:elite_count]}
    elite = tuple(item for item in full if item.sequence_key in elite_keys)
    if elite and len(elite) < len(full):
        branch, probability = _categorical(
            rng,
            (
                ("elite", config.elite_parent_probability),
                ("population", 1.0 - config.elite_parent_probability),
            ),
            f"{label}.branch",
        )
        pool = elite if branch == "elite" else full
    else:
        probability, pool = 1.0, elite or full
    factors = [
        ProbabilityFactor("branch" if label == "parent.0" else f"{label}.branch", probability)
    ]
    participants = []
    for draw_index in range(config.tournament_size):
        participants.append(pool[rng.below(len(pool), f"{label}.tournament.{draw_index}")])
        factors.append(ProbabilityFactor(f"{label}.tournament_draw.{draw_index}", 1.0 / len(pool)))
    return min(participants, key=lambda item: (-item.fitness, item.sequence_key)), tuple(factors)


def _replay_edit(
    population: tuple[ArchiveIndividual, ...],
    config: PeptideGAConfig,
    rng: _ReplayRNG,
) -> tuple[str, str, tuple[str, ...], tuple[object, ...], tuple[ProbabilityFactor, ...]]:
    parent1, parent_factors = _parent(population, config, rng, "parent.0")
    available = tuple(
        (name, rate)
        for name, rate in config.operator_rates
        if not (
            (name == "insertion" and len(parent1.sequence) >= config.max_length)
            or (name == "deletion" and len(parent1.sequence) <= config.min_length)
            or (name == "two_parent_crossover" and len(population) < 2)
        )
    )
    operator, operator_probability = _categorical(rng, available, "operator")
    factors = [*parent_factors, ProbabilityFactor("operator", operator_probability)]
    parents = (parent1.sequence_key,)
    if operator == "substitution":
        position = rng.below(len(parent1.sequence), "edit.position")
        residues = config.alphabet.replace(parent1.sequence[position], "")
        residue = residues[rng.below(len(residues), "edit.residue")]
        child = parent1.sequence[:position] + residue + parent1.sequence[position + 1 :]
        edit = ("substitution", position, parent1.sequence[position], residue)
        factors += [
            ProbabilityFactor("edit.position", 1.0 / len(parent1.sequence)),
            ProbabilityFactor("edit.residue", 1.0 / len(residues)),
        ]
    elif operator == "insertion":
        position = rng.below(len(parent1.sequence) + 1, "edit.position")
        residue = config.alphabet[rng.below(len(config.alphabet), "edit.residue")]
        child = parent1.sequence[:position] + residue + parent1.sequence[position:]
        edit = ("insertion", position, residue)
        factors += [
            ProbabilityFactor("edit.position", 1.0 / (len(parent1.sequence) + 1)),
            ProbabilityFactor("edit.residue", 1.0 / len(config.alphabet)),
        ]
    elif operator == "deletion":
        position = rng.below(len(parent1.sequence), "edit.position")
        child = parent1.sequence[:position] + parent1.sequence[position + 1 :]
        edit = ("deletion", position, parent1.sequence[position])
        factors.append(ProbabilityFactor("edit.position", 1.0 / len(parent1.sequence)))
    else:
        parent2, parent2_factors = _parent(
            population, config, rng, "parent.1", parent1.sequence_key
        )
        cut1 = 1 + rng.below(len(parent1.sequence) - 1, "edit.cut.0")
        cut2 = 1 + rng.below(len(parent2.sequence) - 1, "edit.cut.1")
        child = parent1.sequence[:cut1] + parent2.sequence[cut2:]
        parents = (parent1.sequence_key, parent2.sequence_key)
        edit = ("two_parent_crossover", cut1, cut2, *parents)
        factors += [
            *parent2_factors,
            ProbabilityFactor("edit.cut.0", 1.0 / (len(parent1.sequence) - 1)),
            ProbabilityFactor("edit.cut.1", 1.0 / (len(parent2.sequence) - 1)),
        ]
    return child, operator, parents, edit, tuple(factors)


def _verify_batch_against_archive(
    batch: PeptideGABatch,
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    *,
    config: PeptideGAConfig,
) -> None:
    """Independently replay all proposal semantics and authenticate every seal."""

    preflight_peptide_ga_batch_structure(batch)
    _require(type(archive) is AuthenticatedWaveArchive, "archive type differs")
    _require(type(exclusions) is CollisionExclusions, "exclusion type differs")
    _require(type(config) is PeptideGAConfig, "config type differs")
    archive.__post_init__()
    exclusions.__post_init__()
    config.__post_init__()
    _require(batch.artifact == PEPTIDE_GA_ARTIFACT, "artifact differs")
    expected_archive = sha256_bytes(_ARCHIVE + canonical_json_bytes(_archive_document(archive)))
    _require(archive.archive_sha256 == expected_archive, "archive seal differs")
    _require(batch.config_sha256 == config.config_sha256, "config seal differs")
    _require(
        sha256_bytes(config.config_source_bytes) == config.config_sha256, "config source differs"
    )
    _require(
        exclusions
        == CollisionExclusions.from_keys(
            training_sequence_keys=exclusions.training_sequence_keys,
            submitted_sequence_keys=exclusions.submitted_sequence_keys,
        ),
        "exclusion seals differ",
    )
    _require(
        (batch.campaign_id, batch.wave_id, batch.archive_sha256)
        == (archive.campaign_id, archive.wave_id, archive.archive_sha256),
        "archive identity differs",
    )
    expected_input = sha256_bytes(
        _INPUT + canonical_json_bytes(_input_document(archive, exclusions, batch))
    )
    _require(batch.input_sha256 == expected_input, "input seal differs")
    expected_public_exclusions = make_public_exclusion_receipt(
        campaign_id=archive.campaign_id,
        wave_id=archive.wave_id,
        authenticated_pre_wave_head_sha256=archive.head_seal_sha256,
        authenticated_pre_wave_round_count=archive.round_count,
        batch_id=batch.batch_id,
        seed=batch.seed,
        config_sha256=config.config_sha256,
        input_sha256=expected_input,
        training_count=exclusions.training_count,
        submitted_count=exclusions.submitted_count,
        training_set_sha256=exclusions.training_set_sha256,
        submitted_set_sha256=exclusions.submitted_set_sha256,
    )
    _require(
        batch.public_exclusion_receipt == expected_public_exclusions,
        "public-exclusion receipt differs",
    )
    _require(batch.next_attempt_index == len(batch.attempts), "attempt cursor differs")
    _require(len(batch.attempts) <= config.proposal_attempt_cap, "attempt cap exceeded")
    population = archive.individuals
    _require(bool(population), "archive population is empty")
    _require(
        population
        == tuple(sorted(population, key=lambda item: (-item.fitness, item.sequence_key))),
        "archive population order differs",
    )
    archive_keys = {item.sequence_key for item in population}
    first_generated: dict[str, str] = {}
    accepted_ids: list[str] = []
    accepted_sequences: list[str] = []
    for index, attempt in enumerate(batch.attempts):
        _require(attempt.attempt_index == index, "attempt indices differ")
        child, operator, parents, edit, factors = _replay_edit(
            population, config, _ReplayRNG(batch.seed, expected_input, index)
        )
        key = sequence_key(child)
        duplicate = key in first_generated
        if not config.min_length <= len(child) <= config.max_length:
            reason = "length_out_of_support"
        elif set(child) - set(config.alphabet):
            reason = "alphabet_out_of_support"
        elif key in exclusions.training_sequence_key_set:
            reason = "exact_training_overlap"
        elif key in exclusions.submitted_sequence_key_set or key in archive_keys:
            reason = "previously_submitted_collision"
        elif duplicate:
            reason = "generated_duplicate"
        else:
            reason = None
        valid = reason is None
        proposal_id = (
            "ga-proposal-"
            + sha256_bytes(
                b"amp/fixed-default-peptide-ga/proposal-id/v1\0"
                + bytes.fromhex(expected_input)
                + index.to_bytes(8, "big")
            )[:32]
        )
        edge_id = "ga-edge-" + proposal_id.removeprefix("ga-proposal-")
        rollout_id = f"ga-rollout-{batch.batch_id}-{batch.seed}"
        proposal, edge, event = attempt.proposal, attempt.edge, attempt.provenance
        _require(
            (proposal.proposal_id, proposal.rollout_id, proposal.sequence)
            == (proposal_id, rollout_id, child),
            "proposal identity or sequence differs",
        )
        _require(
            (proposal.hard_valid, proposal.rejection_reason) == (valid, reason), "validity differs"
        )
        _require(
            proposal.policy_version == PEPTIDE_GA_POLICY_VERSION
            and proposal.proposal_round == archive.round_count
            and proposal.niche_id == "fixed-default-peptide-ga"
            and proposal.cheap_predictions == (),
            "proposal provenance differs",
        )
        selection = proposal.selection
        _require(
            selection.selected is False
            and selection.eligible_proposal_ids == ()
            and selection.propensity.factors == (ProbabilityFactor("controller_selection", 0.0),)
            and selection.selection_set_id == f"ga-controller-pending-{batch.batch_id}-{index}"
            and selection.policy_version == PEPTIDE_GA_PENDING_SELECTION_POLICY_VERSION
            and selection.seed == batch.seed,
            "controller-owned selection boundary differs",
        )
        _require(
            (edge.edge_id, edge.proposal_id, edge.rollout_id) == (edge_id, proposal_id, rollout_id),
            "edge identity differs",
        )
        _require(
            edge.parent_sequence_keys == parents
            and edge.operator == operator
            and edge.edit_description == edit
            and edge.proposal_trace.factors == factors,
            "edge semantics or exact propensity differs",
        )
        _require(
            edge.random_stream == batch.seed
            and edge.sample_index == index
            and edge.sampling_parameters
            == (
                ("algorithm", PEPTIDE_GA_POLICY_VERSION),
                ("archive_sha256", archive.archive_sha256),
                ("config_sha256", config.config_sha256),
                ("input_sha256", expected_input),
                ("rng", PEPTIDE_GA_RNG_VERSION),
            ),
            "edge replay identity differs",
        )
        accepted_position = len(accepted_ids) if valid else None
        _require(attempt.accepted_position == accepted_position, "accepted position differs")
        _require(
            (event.event_index, event.proposal_id, event.edge_id, event.sequence_key)
            == (index, proposal_id, edge_id, key),
            "attempt provenance identity differs",
        )
        _require(
            event.first_attempt_proposal_id == first_generated.get(key, proposal_id)
            and event.duplicate_within_prefix is duplicate
            and event.hard_valid is valid
            and event.rejection_reason == reason,
            "attempt provenance semantics differ",
        )
        first_generated.setdefault(key, proposal_id)
        if valid:
            accepted_ids.append(proposal_id)
            accepted_sequences.append(child)
    _require(tuple(accepted_ids) == batch.accepted_proposal_ids, "accepted IDs differ")
    _require(tuple(accepted_sequences) == batch.accepted_sequences, "accepted sequences differ")
    expected_status = (
        "complete"
        if len(accepted_ids) == config.candidate_prefix_size
        else "attempt_cap_exhausted"
        if len(batch.attempts) == config.proposal_attempt_cap
        else "in_progress"
    )
    _require(batch.status == expected_status, "batch status differs")
    _require(batch.oracle_query_identities_constructed == 0, "adapter created query identity")
    _require(batch.controller_private_reserve_seats_emitted == 0, "adapter emitted reserve seat")
    expected_output = sha256_bytes(_OUTPUT + canonical_json_bytes(_output_document(batch)))
    _require(batch.output_sha256 == expected_output, "output seal differs")


def verify_unverified_peptide_ga_fixture_batch(
    batch: PeptideGABatch,
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    *,
    config: PeptideGAConfig,
) -> None:
    """Verify only an explicitly unverified deterministic fixture."""

    _require(type(archive) is AuthenticatedWaveArchive, "archive type differs")
    _require(
        archive.authentication_status == "unverified_object_fixture_only",
        "object verifier accepts only an explicitly unverified fixture",
    )
    _verify_batch_against_archive(batch, archive, exclusions, config=config)


def verify_peptide_ga_batch_from_path(
    batch: PeptideGABatch,
    root: str | Path,
    exclusions: CollisionExclusions,
    *,
    trusted_parent: str | Path,
    wave_id: str,
    expected_header_sha256: str,
    expected_head_seal_sha256: str,
    expected_round_count: int,
    expected_query_count: int,
    expected_response_count: int,
    config: PeptideGAConfig,
    config_path: str | Path,
    trusted_config_parent: str | Path,
    expected_config_sha256: str,
    policy_registry_path: str | Path,
    trusted_policy_registry_parent: str | Path,
    expected_policy_registry_sha256: str,
    exclusion_asset_path: str | Path,
    exclusion_receipt_path: str | Path,
    trusted_exclusion_parent: str | Path,
    expected_exclusion_asset_sha256: str,
    expected_exclusion_receipt_sha256: str,
    expected_exclusion_issuer_identity_sha256: str,
    truth_contract: FitnessTruthContract,
) -> AuthenticatedWaveArchive:
    """Reopen the campaign and independently reconstruct the scored archive."""

    _require(type(batch) is PeptideGABatch, "batch type differs")
    _require(type(exclusions) is CollisionExclusions, "exclusion type differs")
    _require(type(config) is PeptideGAConfig, "config type differs")
    _require(type(truth_contract) is FitnessTruthContract, "truth contract type differs")
    exclusions.__post_init__()
    config.__post_init__()
    truth_contract.__post_init__()
    _identifier(wave_id, label="wave ID")
    _sha256(expected_header_sha256, label="expected campaign header SHA-256")
    _sha256(expected_head_seal_sha256, label="expected campaign head seal")
    _bounded_integer(
        expected_round_count, maximum=PEPTIDE_GA_MAX_ROUNDS, label="expected round count"
    )
    _bounded_integer(
        expected_query_count, maximum=PEPTIDE_GA_MAX_QUERIES, label="expected query count"
    )
    _bounded_integer(
        expected_response_count,
        maximum=PEPTIDE_GA_MAX_RESPONSES,
        label="expected response count",
    )
    _require_registry_outside_campaign_root(
        root,
        policy_registry_path=policy_registry_path,
        trusted_policy_registry_parent=trusted_policy_registry_parent,
    )
    _verify_trusted_config_source(
        config,
        config_path=config_path,
        trusted_config_parent=trusted_config_parent,
        expected_config_sha256=expected_config_sha256,
    )
    policy_registry = load_selection_policy_registry_from_path(
        policy_registry_path,
        trusted_registry_parent=trusted_policy_registry_parent,
        expected_registry_sha256=expected_policy_registry_sha256,
    )
    _require(
        policy_registry.registry_sha256 == truth_contract.selection_policy_registry_sha256,
        "truth contract policy registry digest differs",
    )
    verified = verify_campaign(
        root,
        trusted_parent=trusted_parent,
        expected_header_sha256=expected_header_sha256,
        expected_head_seal_sha256=expected_head_seal_sha256,
        expected_round_count=expected_round_count,
        replay_limits=EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    )
    exclusion_authority = load_collision_exclusion_authority_from_paths(
        exclusion_asset_path,
        exclusion_receipt_path,
        trusted_parent=trusted_exclusion_parent,
        campaign_root=root,
        expected_asset_sha256=expected_exclusion_asset_sha256,
        expected_receipt_sha256=expected_exclusion_receipt_sha256,
        expected_issuer_identity_sha256=expected_exclusion_issuer_identity_sha256,
        campaign_id=verified.header.campaign_id,
        phase=verified.header.phase,
        wave_id=wave_id,
        authenticated_pre_wave_head_sha256=expected_head_seal_sha256,
        authenticated_pre_wave_round_count=expected_round_count,
        protocol_sha256=verified.header.protocol_sha256,
        config_sha256=config.config_sha256,
        policy_registry_sha256=policy_registry.registry_sha256,
        selection_policy_implementation_manifest_sha256=(
            policy_registry.implementation_manifest_sha256
        ),
        truth_contract_sha256=truth_contract.sha256,
    )
    _require(exclusions == exclusion_authority.exclusions, "external exclusion inventory differs")
    _require(
        truth_contract.objective_names == tuple(name for name, _ in config.objective_weights),
        "truth-contract objectives differ from GA config",
    )
    _require(
        verified.header.configuration_id
        == truth_contract.campaign_configuration_id
        == PEPTIDE_GA_CAMPAIGN_CONFIGURATION_ID,
        "campaign configuration differs from truth contract",
    )
    _require(verified.terminal is False, "terminal campaign cannot generate a next wave")
    _require(
        expected_round_count < PEPTIDE_GA_MAX_ROUNDS,
        "campaign reached the maximum round count",
    )
    _require(
        verified.query_count == expected_query_count == expected_response_count
        and verified.response_count == expected_response_count
        and not verified.outstanding_query_ids,
        "campaign wave is incomplete",
    )
    proposals: dict[str, tuple[str, str]] = {}
    queries: dict[str, tuple[str, str, str]] = {}
    outcomes: dict[str, list[tuple[str, dict[str, float]]]] = {}
    elapsed: list[object] = []
    last_event_type = ""
    wave_bindings: dict[str, tuple[int, str]] = {}
    selection_bindings: dict[str, tuple[str, str, int]] = {}
    for raw in verified.event_documents:
        event = json.loads(raw)
        last_event_type = event["event_type"]
        elapsed.append(event["scientific_elapsed_ns"])
        payload = event["payload"]
        if event["event_type"] == "proposal":
            proposal = payload["proposal_record"]
            namespace, history_wave_id, selection_set_id = (
                validate_campaign_proposal_policy_binding(
                    registry=policy_registry,
                    config_sha256=config.config_sha256,
                    campaign_id=verified.header.campaign_id,
                    phase=verified.header.phase,
                    header_sha256=verified.header_sha256,
                    round_seals=verified.round_seals,
                    round_index=event["round_index"],
                    proposal_record=proposal,
                    edge_record=payload["edge_record"],
                )
            )
            wave_binding = (event["round_index"], selection_set_id)
            previous_wave_binding = wave_bindings.setdefault(history_wave_id, wave_binding)
            _require(previous_wave_binding == wave_binding, "history wave ID was reused")
            binding = (namespace, history_wave_id, event["round_index"])
            previous_binding = selection_bindings.setdefault(selection_set_id, binding)
            _require(previous_binding == binding, "history selection-set ID was reused")
            sequence = proposal["sequence"]
            _require(
                type(sequence) is str
                and config.min_length <= len(sequence) <= config.max_length
                and sequence == sequence.upper()
                and set(sequence) <= set(config.alphabet),
                "archive proposal sequence support differs",
            )
            proposals[proposal["proposal_id"]] = (
                sequence,
                sequence_key(sequence),
            )
        elif event["event_type"] == "query":
            _require(
                payload["evaluator_version"] == truth_contract.evaluator_version
                and payload["fidelity"] == truth_contract.fidelity,
                "query truth contract differs",
            )
            proposal_id = payload["proposal_id"]
            identity = payload["query_identity"]
            _require(
                set(identity)
                == {
                    "canonical_sequence_id",
                    "oracle_contract_sha256",
                    "evaluator_sha256",
                    "checkpoint_sha256",
                    "endpoint_context_sha256",
                    "transform_sha256",
                    "replicate_id",
                },
                "query identity schema differs",
            )
            _require(
                identity["canonical_sequence_id"] == proposals[proposal_id][1],
                "query sequence truth identity differs",
            )
            for name in (
                "oracle_contract_sha256",
                "evaluator_sha256",
                "checkpoint_sha256",
                "endpoint_context_sha256",
                "transform_sha256",
            ):
                _require(
                    identity[name] == getattr(truth_contract, name),
                    f"query identity {name} differs from truth contract",
                )
            _require(
                type(identity["replicate_id"]) is int
                and 0 <= identity["replicate_id"] <= PEPTIDE_GA_SIGNED_63_MAX,
                "query identity replicate ID differs",
            )
            queries[payload["query_id"]] = (
                proposal_id,
                payload["evaluator_version"],
                payload["fidelity"],
            )
        elif event["event_type"] == "response":
            _require(payload["query_id"] in queries, "response query differs")
            if payload["status"] == "succeeded":
                evaluation = payload["evaluation_record"]
                proposal_id, evaluator, fidelity = queries[payload["query_id"]]
                _require(
                    evaluation["proposal_id"] == proposal_id
                    and evaluation["evaluator_version"] == evaluator
                    and evaluation["fidelity"] == fidelity,
                    "evaluation truth identity differs",
                )
                outcome_map: dict[str, float] = {}
                _require(
                    type(evaluation["outcomes"]) is list
                    and len(evaluation["outcomes"]) == len(truth_contract.objective_names),
                    "evaluation outcome truth schema differs",
                )
                for row in evaluation["outcomes"]:
                    _require(
                        type(row) is list
                        and len(row) == 2
                        and type(row[0]) is str
                        and row[0] not in outcome_map
                        and type(row[1]) is float
                        and math.isfinite(row[1]),
                        "evaluation outcome differs",
                    )
                    outcome_map[row[0]] = row[1]
                _require(
                    tuple(outcome_map) == truth_contract.objective_names,
                    "evaluation outcome truth schema differs",
                )
                key = proposals[proposal_id][1]
                outcomes.setdefault(key, []).append((evaluation["evaluation_id"], outcome_map))
        else:
            raise PeptideGAError("campaign event type differs")
    _require(last_event_type == "response", "archive head is not a completed wave boundary")
    _require(bool(outcomes), "campaign archive has no successful fitness outcomes")
    first_by_key: dict[str, str] = {}
    sequence_by_key: dict[str, str] = {}
    for proposal_id, (sequence, key) in proposals.items():
        first_by_key.setdefault(key, proposal_id)
        sequence_by_key.setdefault(key, sequence)
    individuals = []
    for key, evaluations in outcomes.items():
        means = tuple(
            (
                name,
                _finite_mean([row[name] for _, row in evaluations], objective=name),
            )
            for name, _ in config.objective_weights
        )
        individuals.append(
            ArchiveIndividual(
                sequence=sequence_by_key[key],
                sequence_key=key,
                first_proposal_id=first_by_key[key],
                successful_evaluation_ids=tuple(
                    sorted(evaluation_id for evaluation_id, _ in evaluations)
                ),
                objective_means=means,
                fitness=_finite_fitness(means, config),
            )
        )
    individuals.sort(key=lambda item: (-item.fitness, item.sequence_key))
    round_inventory = sha256_bytes(
        b"amp/fixed-default-peptide-ga/round-seal-inventory/v1\0"
        + canonical_json_bytes(list(verified.round_seals))
    )
    round_timing_inventory = round_timing_receipt_inventory_sha256(
        verified.round_timing_receipt_sha256s
    )
    event_inventory = sha256_bytes(
        b"amp/fixed-default-peptide-ga/event-inventory/v1\0"
        + canonical_json_bytes([sha256_bytes(raw) for raw in verified.event_documents])
    )
    elapsed_inventory = sha256_bytes(
        b"amp/fixed-default-peptide-ga/elapsed-inventory/v1\0" + canonical_json_bytes(elapsed)
    )
    archive_without_digest = {
        "authentication_status": "verified_campaign_path_truth_contract_unaccepted",
        "campaign_id": verified.header.campaign_id,
        "phase": verified.header.phase,
        "elapsed_inventory_sha256": elapsed_inventory,
        "event_count": len(verified.event_documents),
        "event_inventory_sha256": event_inventory,
        "header_sha256": verified.header_sha256,
        "head_seal_sha256": expected_head_seal_sha256,
        "individuals": [_individual(item) for item in individuals],
        "last_event_sha256": verified.last_event_sha256,
        "proposal_count": verified.proposal_count,
        "query_count": verified.query_count,
        "response_count": verified.response_count,
        "round_count": expected_round_count,
        "round_seal_inventory_sha256": round_inventory,
        "round_timing_receipt_inventory_sha256": round_timing_inventory,
        "round_timing_receipt_sha256s": list(verified.round_timing_receipt_sha256s),
        "scientific_elapsed_ns": verified.scientific_elapsed_ns,
        "truth_contract_sha256": truth_contract.sha256,
        "wave_id": wave_id,
    }
    archive = AuthenticatedWaveArchive(
        campaign_id=verified.header.campaign_id,
        phase=verified.header.phase,
        wave_id=wave_id,
        header_sha256=verified.header_sha256,
        head_seal_sha256=expected_head_seal_sha256,
        last_event_sha256=verified.last_event_sha256,
        round_count=expected_round_count,
        event_count=len(verified.event_documents),
        proposal_count=verified.proposal_count,
        query_count=verified.query_count,
        response_count=verified.response_count,
        scientific_elapsed_ns=verified.scientific_elapsed_ns,
        authentication_status="verified_campaign_path_truth_contract_unaccepted",
        round_seal_inventory_sha256=round_inventory,
        round_timing_receipt_sha256s=verified.round_timing_receipt_sha256s,
        round_timing_receipt_inventory_sha256=round_timing_inventory,
        event_inventory_sha256=event_inventory,
        elapsed_inventory_sha256=elapsed_inventory,
        truth_contract_sha256=truth_contract.sha256,
        individuals=tuple(individuals),
        archive_sha256=sha256_bytes(_ARCHIVE + canonical_json_bytes(archive_without_digest)),
    )
    _verify_batch_against_archive(batch, archive, exclusions, config=config)
    return archive


__all__ = [
    "verify_peptide_ga_batch_from_path",
    "verify_unverified_peptide_ga_fixture_batch",
]

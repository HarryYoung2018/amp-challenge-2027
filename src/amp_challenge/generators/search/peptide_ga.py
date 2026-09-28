"""Deterministic fixed-default peptide-GA proposal adapter with no oracle authority.

The adapter consumes only a completed, authenticated campaign-head view.  It
does not accept oracle clients, organizer-reference sequences, or query
identity fields.  The enclosing controller remains responsible for submission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from amp_challenge.evaluation.evolutionary_kl_protocol import FROZEN_PROTOCOL_SHA256
from amp_challenge.generators.search.campaign_ledger import (
    EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    CampaignHeader,
    VerifiedCampaign,
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
    PEPTIDE_GA_ATTEMPT_CAP,
    PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
    PEPTIDE_GA_CAMPAIGN_CONFIGURATION_ID,
    PEPTIDE_GA_CONFIG_MAX_BYTES,
    PEPTIDE_GA_MAX_EVENTS,
    PEPTIDE_GA_MAX_PROPOSALS,
    PEPTIDE_GA_MAX_QUERIES,
    PEPTIDE_GA_MAX_RESPONSES,
    PEPTIDE_GA_MAX_ROUNDS,
    PEPTIDE_GA_MAX_SCIENTIFIC_ELAPSED_NS,
    PEPTIDE_GA_PENDING_SELECTION_POLICY_VERSION,
    PEPTIDE_GA_POLICY_VERSION,
    PEPTIDE_GA_RNG_VERSION,
    PEPTIDE_GA_SIGNED_63_MAX,
    AdapterAttemptProvenance,
    ArchiveIndividual,
    AuthenticatedSelectionPolicyRegistry,
    AuthenticatedWaveArchive,
    CollisionExclusions,
    FitnessTruthContract,
    PeptideGAAttempt,
    PeptideGABatch,
    PeptideGAConfig,
    PeptideGAError,
    PublicExclusionReceipt,
    canonical_json_bytes,
    make_public_exclusion_receipt,
    preflight_peptide_ga_batch_structure,
    public_exclusion_receipt_document,
    round_timing_receipt_inventory_sha256,
    sequence_key,
    sha256_bytes,
)
from amp_challenge.generators.search.records import (
    EdgeRecord,
    ProbabilityFactor,
    ProbabilityTrace,
    ProposalRecord,
    SelectionDecision,
)

_CAMPAIGN_EVENT_HASH_DOMAIN = b"amp/evolutionary-kl/campaign-event/v1\0"
_CAMPAIGN_GENESIS_HASH_DOMAIN = b"amp/evolutionary-kl/campaign-genesis/v1\0"
_INPUT_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/input/v1\0"
_ARCHIVE_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/archive/v1\0"
_OUTPUT_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/output/v1\0"
_RNG_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/rng/v1\0"
_PATH_TEXT_MAX_LENGTH = 4096
_BUILTIN_PATH_TYPE = type(Path("."))
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RNG_REJECTION_CAP = 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PeptideGAError(message)


def _strict_keys(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    _require(type(value) is dict and set(value) == keys, f"{label} schema differs")
    assert isinstance(value, dict)
    return value


def _identifier(value: object, *, label: str, maximum_length: int = 128) -> str:
    _require(
        type(value) is str
        and len(value) <= maximum_length
        and _IDENTIFIER_RE.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    assert isinstance(value, str)
    return value


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
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


def _read_bounded_config_source(path: object) -> bytes:
    """Read one stable regular config inode before parsing or hashing it."""

    target = _exact_path(path, label="peptide-GA config path")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        file_fd = os.open(target, flags)
    except OSError as error:
        raise PeptideGAError("peptide-GA config open failed") from error
    try:
        before = os.fstat(file_fd)
        _require(stat.S_ISREG(before.st_mode), "peptide-GA config is not a regular file")
        _require(before.st_nlink == 1, "peptide-GA config link count differs")
        _require(
            0 < before.st_size <= PEPTIDE_GA_CONFIG_MAX_BYTES,
            "peptide-GA config size differs",
        )
        payload = bytearray()
        while len(payload) <= PEPTIDE_GA_CONFIG_MAX_BYTES:
            chunk = os.read(file_fd, min(65_536, PEPTIDE_GA_CONFIG_MAX_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        _require(
            0 < len(payload) <= PEPTIDE_GA_CONFIG_MAX_BYTES,
            "peptide-GA config size differs",
        )
        after = os.fstat(file_fd)
        _require(
            _stat_identity(before) == _stat_identity(after),
            "peptide-GA config changed while reading",
        )
        try:
            entry = os.stat(target, follow_symlinks=False)
        except OSError as error:
            raise PeptideGAError("peptide-GA config entry recheck failed") from error
        _require(
            _stat_identity(after) == _stat_identity(entry),
            "peptide-GA config entry changed while reading",
        )
    finally:
        os.close(file_fd)
    return bytes(payload)


def _exact_float(value: object, *, label: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    assert isinstance(value, float)
    return value


def load_peptide_ga_config(path: str | Path) -> PeptideGAConfig:
    """Load the exact development-only GA configuration."""

    payload = _read_bounded_config_source(path)
    try:
        value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PeptideGAError("peptide-GA config is not canonical UTF-8 TOML") from error
    root = _strict_keys(
        value,
        {
            "artifact",
            "automatic_production_eligible",
            "biological_superiority_claim_allowed",
            "composition",
            "evidence",
            "execution_authorized",
            "fitness",
            "scientific_evidence_accepted",
            "schema_version",
            "search",
            "status",
            "support",
            "tuning_study_accepted",
        },
        label="peptide-GA config",
    )
    _require(
        type(root["schema_version"]) is int and root["schema_version"] == 1,
        "peptide-GA config version differs",
    )
    _require(
        root["artifact"] == "fixed_default_peptide_ga_development_config_v1",
        "config artifact differs",
    )
    _require(
        root["status"] == "development_only_blocked_on_tuning_and_campaign_assets",
        "config status differs",
    )
    support = _strict_keys(
        root["support"],
        {"alphabet", "exact_training_overlap_forbidden", "max_length", "min_length"},
        label="support",
    )
    _require(support["exact_training_overlap_forbidden"] is True, "exact overlap exclusion differs")
    composition = _strict_keys(
        root["composition"],
        {
            "adapter_observes_private_reserve_identities",
            "adapter_emits_reserve_seats",
            "candidate_prefix_size",
            "controller_candidate_consumption",
            "controller_owns_oracle_query_identity",
            "controller_private_reserve_seats",
            "method_controlled_seats",
        },
        label="composition",
    )
    _require(
        composition["controller_owns_oracle_query_identity"] is True, "query ownership differs"
    )
    _require(composition["adapter_emits_reserve_seats"] is False, "reserve ownership differs")
    _require(
        composition["adapter_observes_private_reserve_identities"] is False,
        "adapter cannot observe controller-private reserves",
    )
    _require(
        composition["controller_candidate_consumption"]
        == "privately_veto_collisions_then_take_first_14_in_frozen_prefix_order",
        "controller candidate consumption differs",
    )
    search = _strict_keys(
        root["search"],
        {
            "elite_fraction",
            "elite_parent_probability",
            "operator_rates",
            "proposal_attempt_cap",
            "resume",
            "rng",
            "tournament_size",
        },
        label="search",
    )
    _require(search["rng"] == PEPTIDE_GA_RNG_VERSION, "RNG version differs")
    _require(search["resume"] == "attempt_index_and_retained_prefix_exact_replay", "resume differs")
    rates = _strict_keys(
        search["operator_rates"],
        {"deletion", "insertion", "substitution", "two_parent_crossover"},
        label="operator rates",
    )
    fitness = _strict_keys(root["fitness"], {"objective_weights"}, label="fitness")
    weights = _strict_keys(
        fitness["objective_weights"],
        {"gram_negative_activity", "gram_positive_activity"},
        label="objective weights",
    )
    evidence = _strict_keys(
        root["evidence"],
        {
            "accepted_campaign_archive",
            "accepted_controller_private_reserve_receipt",
            "accepted_objective_truth_mapping",
            "accepted_training_sequences",
            "accepted_tuning_study",
            "current_v1_resource_allocation",
            "oracle_or_model_execution",
            "successor_training_homology_enforced_by_adapter",
            "successor_training_homology_exclusion",
            "successor_v2_namespace_composition_receipt",
        },
        label="evidence",
    )
    _require(
        evidence
        == {
            "accepted_campaign_archive": "missing_execution_blocking",
            "accepted_controller_private_reserve_receipt": "missing_execution_blocking",
            "accepted_objective_truth_mapping": "missing_execution_blocking",
            "accepted_training_sequences": "missing_execution_blocking",
            "accepted_tuning_study": "missing_execution_blocking",
            "current_v1_resource_allocation": (
                "zero_adapter_tuning_calls_and_zero_adapter_tuning_gpu_hours"
            ),
            "oracle_or_model_execution": "not_permitted_by_this_adapter",
            "successor_training_homology_enforced_by_adapter": False,
            "successor_training_homology_exclusion": "missing_execution_blocking",
            "successor_v2_namespace_composition_receipt": "missing_execution_blocking",
        },
        "evidence block differs",
    )
    return PeptideGAConfig(
        alphabet=support["alphabet"],
        min_length=support["min_length"],
        max_length=support["max_length"],
        proposal_attempt_cap=search["proposal_attempt_cap"],
        method_controlled_seats=composition["method_controlled_seats"],
        controller_private_reserve_seats=composition["controller_private_reserve_seats"],
        candidate_prefix_size=composition["candidate_prefix_size"],
        tournament_size=search["tournament_size"],
        elite_fraction=_exact_float(search["elite_fraction"], label="elite fraction"),
        elite_parent_probability=_exact_float(
            search["elite_parent_probability"], label="elite parent probability"
        ),
        operator_rates=tuple(
            (name, _exact_float(rates[name], label=f"operator rate {name}"))
            for name in ("substitution", "insertion", "deletion", "two_parent_crossover")
        ),
        objective_weights=tuple(
            (name, _exact_float(weights[name], label=f"objective weight {name}"))
            for name in ("gram_positive_activity", "gram_negative_activity")
        ),
        execution_authorized=root["execution_authorized"],
        scientific_evidence_accepted=root["scientific_evidence_accepted"],
        automatic_production_eligible=root["automatic_production_eligible"],
        biological_superiority_claim_allowed=root["biological_superiority_claim_allowed"],
        tuning_study_accepted=root["tuning_study_accepted"],
        config_sha256=sha256_bytes(payload),
        config_source_bytes=payload,
    )


def _verify_trusted_config_source(
    config: PeptideGAConfig,
    *,
    config_path: str | Path,
    trusted_config_parent: str | Path,
    expected_config_sha256: str,
) -> None:
    """Bind the parsed config to one non-symlink regular file under a trusted parent."""

    _require(type(config) is PeptideGAConfig, "peptide-GA config type differs")
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
    _require(stat.S_ISDIR(parent_stat.st_mode), "trusted config parent is not a directory")
    _require(parent_stat.st_uid == os.geteuid(), "trusted config parent owner differs")
    _require(parent_stat.st_mode & 0o022 == 0, "trusted config parent is group/world writable")
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
            _require(stat.S_ISREG(before.st_mode), "trusted config is not a regular file")
            _require(before.st_nlink == 1, "trusted config link count differs")
            _require(before.st_uid == os.geteuid(), "trusted config owner differs")
            _require(before.st_mode & 0o022 == 0, "trusted config is group/world writable")
            _require(
                0 < before.st_size <= PEPTIDE_GA_CONFIG_MAX_BYTES,
                "trusted config size differs",
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


def _campaign_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _parse_event(raw: bytes, *, position: int) -> dict[str, object]:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            _require(key not in result, f"campaign event {position} duplicates JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PeptideGAError(f"campaign event {position} is not strict JSON") from error
    _require(type(value) is dict, f"campaign event {position} is not an object")
    _require(_campaign_json_bytes(value) == raw, f"campaign event {position} is not canonical")
    assert isinstance(value, dict)
    return value


def _preflight_verified_campaign(verified: VerifiedCampaign) -> None:
    """Bound fixture objects before any replay, collection, or serialization."""

    _require(type(verified) is VerifiedCampaign, "archive input must be a VerifiedCampaign")
    _require(type(verified.header) is CampaignHeader, "campaign header type differs")
    _require(
        type(verified.round_seals) is tuple
        and 0 < len(verified.round_seals) <= PEPTIDE_GA_MAX_ROUNDS,
        "campaign round-seal inventory differs",
    )
    for index, digest in enumerate(verified.round_seals):
        _sha256(digest, label=f"campaign round seal {index}")
    _require(
        type(verified.round_timing_receipt_sha256s) is tuple
        and len(verified.round_timing_receipt_sha256s) == len(verified.round_seals),
        "campaign round timing-receipt inventory differs",
    )
    for index, digest in enumerate(verified.round_timing_receipt_sha256s):
        _sha256(digest, label=f"campaign timing receipt {index}")
    _require(
        type(verified.event_documents) is tuple
        and 0 < len(verified.event_documents) <= PEPTIDE_GA_MAX_EVENTS,
        "campaign event inventory differs",
    )
    cumulative_event_bytes = 0
    for index, document in enumerate(verified.event_documents):
        _require(type(document) is bytes and bool(document), f"campaign event {index} differs")
        cumulative_event_bytes += len(document)
        _require(
            cumulative_event_bytes
            <= EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS.max_cumulative_event_bytes,
            "campaign cumulative event bytes exceed replay cap",
        )
    for value, maximum, label in (
        (verified.proposal_count, PEPTIDE_GA_MAX_PROPOSALS, "campaign proposal count"),
        (verified.query_count, PEPTIDE_GA_MAX_QUERIES, "campaign query count"),
        (verified.response_count, PEPTIDE_GA_MAX_RESPONSES, "campaign response count"),
        (
            verified.scientific_elapsed_ns,
            PEPTIDE_GA_MAX_SCIENTIFIC_ELAPSED_NS,
            "campaign scientific elapsed ns",
        ),
    ):
        _bounded_integer(value, maximum=maximum, label=label)
    _require(
        len(verified.event_documents)
        == verified.proposal_count + verified.query_count + verified.response_count,
        "campaign response count differs from event inventory",
    )
    _require(
        type(verified.outstanding_query_ids) is tuple
        and len(verified.outstanding_query_ids) <= PEPTIDE_GA_MAX_QUERIES,
        "campaign outstanding-query inventory differs",
    )
    for query_id in verified.outstanding_query_ids:
        _identifier(query_id, label="campaign outstanding query ID")
    _sha256(verified.header_sha256, label="campaign header SHA-256")
    _require(
        verified.header_sha256 == verified.header.sha256,
        "campaign header digest differs",
    )
    _sha256(verified.last_event_sha256, label="campaign last event SHA-256")
    _require(type(verified.terminal) is bool, "campaign terminal flag differs")


def _build_wave_archive_from_verified(
    verified: VerifiedCampaign,
    *,
    wave_id: str,
    expected_head_seal_sha256: str,
    expected_round_count: int,
    expected_query_count: int,
    expected_response_count: int,
    config: PeptideGAConfig,
    truth_contract: FitnessTruthContract,
    policy_registry: AuthenticatedSelectionPolicyRegistry,
    authentication_status: str,
) -> AuthenticatedWaveArchive:
    """Reconstruct a scored immutable view only after every current query completed."""

    _preflight_verified_campaign(verified)
    _require(type(config) is PeptideGAConfig, "peptide-GA config type differs")
    _require(type(truth_contract) is FitnessTruthContract, "truth contract type differs")
    _require(
        type(policy_registry) is AuthenticatedSelectionPolicyRegistry,
        "policy registry type differs",
    )
    config.__post_init__()
    truth_contract.__post_init__()
    policy_registry.__post_init__()
    _require(
        policy_registry.registry_sha256 == truth_contract.selection_policy_registry_sha256,
        "truth contract policy registry digest differs",
    )
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
    _identifier(wave_id, label="wave ID")
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
    _require(len(verified.round_seals) == expected_round_count > 0, "round count differs")
    _require(
        expected_round_count < PEPTIDE_GA_MAX_ROUNDS,
        "campaign reached the maximum round count",
    )
    _require(verified.round_seals[-1] == expected_head_seal_sha256, "campaign head differs")
    _require(verified.query_count == expected_query_count, "query count differs")
    _require(verified.response_count == expected_response_count, "response count differs")
    _require(verified.query_count == verified.response_count, "wave has incomplete responses")
    _require(not verified.outstanding_query_ids, "wave has outstanding queries")
    _require(verified.terminal is False, "terminal campaign cannot generate a next wave")
    _require(len(verified.event_documents) > 0, "archive is empty")

    previous = sha256_bytes(_CAMPAIGN_GENESIS_HASH_DOMAIN + bytes.fromhex(verified.header_sha256))
    proposals: dict[str, tuple[str, str]] = {}
    queries: dict[str, tuple[str, str, str]] = {}
    outcomes_by_sequence: dict[str, list[tuple[str, dict[str, float]]]] = {}
    proposal_count = query_count = response_count = 0
    last_type = ""
    wave_bindings: dict[str, tuple[int, str]] = {}
    selection_bindings: dict[str, tuple[str, str, int]] = {}
    for position, raw in enumerate(verified.event_documents):
        event = _parse_event(raw, position=position)
        expected_keys = {
            "event_position",
            "event_sha256",
            "event_type",
            "identity_key",
            "payload",
            "previous_event_sha256",
            "round_index",
            "round_position",
            "schema_version",
            "scientific_elapsed_ns",
        }
        _require(set(event) == expected_keys, "campaign event schema differs")
        _require(event["event_position"] == position, "campaign event positions differ")
        _require(event["previous_event_sha256"] == previous, "campaign event chain differs")
        base = {key: value for key, value in event.items() if key != "event_sha256"}
        digest = sha256_bytes(_CAMPAIGN_EVENT_HASH_DOMAIN + _campaign_json_bytes(base))
        _require(event["event_sha256"] == digest, "campaign event digest differs")
        payload = event["payload"]
        _require(type(payload) is dict, "campaign event payload is not an object")
        assert isinstance(payload, dict)
        event_type = event["event_type"]
        last_type = str(event_type)
        if event_type == "proposal":
            record = payload.get("proposal_record")
            _require(type(record) is dict, "proposal record is absent")
            assert isinstance(record, dict)
            proposal_id = record.get("proposal_id")
            sequence = record.get("sequence")
            _identifier(proposal_id, label="archive proposal ID")
            assert isinstance(proposal_id, str)
            _require(proposal_id not in proposals, "proposal ID differs")
            _require(type(sequence) is str, "proposal sequence differs")
            _require(
                config.min_length <= len(sequence) <= config.max_length
                and sequence == sequence.upper()
                and set(sequence) <= set(config.alphabet),
                "archive proposal sequence support differs",
            )
            selection = record.get("selection")
            _require(type(selection) is dict, "proposal selection record is absent")
            assert isinstance(selection, dict)
            edge_record = payload.get("edge_record")
            _require(type(edge_record) is dict, "proposal edge record is absent")
            assert isinstance(edge_record, dict)
            namespace, history_wave_id, selection_set_id = (
                validate_campaign_proposal_policy_binding(
                    registry=policy_registry,
                    config_sha256=config.config_sha256,
                    campaign_id=verified.header.campaign_id,
                    phase=verified.header.phase,
                    header_sha256=verified.header_sha256,
                    round_seals=verified.round_seals,
                    round_index=event["round_index"],
                    proposal_record=record,
                    edge_record=edge_record,
                )
            )
            wave_binding = (event["round_index"], selection_set_id)
            prior_wave_binding = wave_bindings.setdefault(history_wave_id, wave_binding)
            _require(prior_wave_binding == wave_binding, "history wave ID was reused")
            binding = (namespace, history_wave_id, event["round_index"])
            prior_binding = selection_bindings.setdefault(selection_set_id, binding)
            _require(prior_binding == binding, "history selection-set ID was reused")
            key = sequence_key(sequence)
            _require(event["identity_key"] == key, "archive proposal sequence key differs")
            proposals[proposal_id] = (sequence, key)
            proposal_count += 1
        elif event_type == "query":
            _require(
                set(payload)
                == {
                    "batch_id",
                    "batch_position",
                    "call_position",
                    "evaluator_version",
                    "fidelity",
                    "planned_cost",
                    "proposal_id",
                    "query_id",
                    "query_identity",
                },
                "query payload schema differs",
            )
            query_id = payload.get("query_id")
            proposal_id = payload.get("proposal_id")
            _identifier(query_id, label="archive query ID")
            _identifier(proposal_id, label="archive query proposal ID")
            assert isinstance(query_id, str)
            assert isinstance(proposal_id, str)
            _require(query_id not in queries, "query ID differs")
            _require(proposal_id in proposals, "query proposal differs")
            _require(
                payload.get("evaluator_version") == truth_contract.evaluator_version,
                "query evaluator differs",
            )
            _require(payload.get("fidelity") == truth_contract.fidelity, "query fidelity differs")
            identity = payload.get("query_identity")
            _require(type(identity) is dict, "query identity differs")
            assert isinstance(identity, dict)
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
                identity.get("canonical_sequence_id") == proposals[proposal_id][1],
                "query truth identity differs from proposal sequence",
            )
            for name in (
                "oracle_contract_sha256",
                "evaluator_sha256",
                "checkpoint_sha256",
                "endpoint_context_sha256",
                "transform_sha256",
            ):
                _require(
                    identity.get(name) == getattr(truth_contract, name),
                    f"query identity {name} differs from truth contract",
                )
            _require(
                type(identity.get("replicate_id")) is int
                and 0 <= identity["replicate_id"] <= PEPTIDE_GA_SIGNED_63_MAX,
                "query identity replicate ID differs",
            )
            queries[query_id] = (
                proposal_id,
                truth_contract.evaluator_version,
                truth_contract.fidelity,
            )
            query_count += 1
        elif event_type == "response":
            _require(
                set(payload)
                == {
                    "call_position",
                    "evaluation_record",
                    "query_id",
                    "response_id",
                    "status",
                    "status_detail",
                },
                "response payload schema differs",
            )
            query_id = payload.get("query_id")
            _require(type(query_id) is str and query_id in queries, "response query differs")
            if payload.get("status") == "succeeded":
                evaluation = payload.get("evaluation_record")
                _require(type(evaluation) is dict, "successful response lacks evaluation")
                assert isinstance(evaluation, dict)
                proposal_id, evaluator_version, fidelity = queries[query_id]
                _require(
                    evaluation.get("proposal_id") == proposal_id, "evaluation proposal differs"
                )
                _require(
                    evaluation.get("evaluator_version") == evaluator_version,
                    "evaluation evaluator differs",
                )
                _require(evaluation.get("fidelity") == fidelity, "evaluation fidelity differs")
                evaluation_id = evaluation.get("evaluation_id")
                rows = evaluation.get("outcomes")
                _identifier(evaluation_id, label="archive evaluation ID")
                assert isinstance(evaluation_id, str)
                _require(type(rows) is list, "evaluation payload differs")
                outcome_map: dict[str, float] = {}
                assert isinstance(rows, list)
                required = truth_contract.objective_names
                _require(len(rows) == len(required), "evaluation truth outcome schema differs")
                for row in rows:
                    _require(type(row) is list and len(row) == 2, "evaluation outcome row differs")
                    name, value = row
                    _require(
                        type(name) is str and name not in outcome_map,
                        "evaluation outcome name differs",
                    )
                    _require(
                        type(value) is float and math.isfinite(value), "evaluation outcome differs"
                    )
                    outcome_map[name] = value
                _require(tuple(outcome_map) == required, "evaluation truth outcome schema differs")
                sequence, key = proposals[proposal_id]
                _require(
                    config.min_length <= len(sequence) <= config.max_length
                    and sequence == sequence.upper()
                    and set(sequence) <= set(config.alphabet),
                    "successful archive sequence support differs",
                )
                outcomes_by_sequence.setdefault(key, []).append((evaluation_id, outcome_map))
            response_count += 1
        elif event_type == "recommendation":
            raise PeptideGAError("terminal campaign cannot generate a next wave")
        else:
            raise PeptideGAError("campaign event type differs")
        previous = digest
    _require(previous == verified.last_event_sha256, "campaign last event digest differs")
    _require(
        len(verified.event_documents)
        == verified.proposal_count + verified.query_count + verified.response_count,
        "campaign event counts differ",
    )
    _require(
        (proposal_count, query_count, response_count)
        == (verified.proposal_count, verified.query_count, verified.response_count),
        "reconstructed campaign counts differ",
    )
    _require(last_type == "response", "archive head is not a completed wave boundary")
    _require(bool(outcomes_by_sequence), "completed archive has no successful fitness outcomes")

    first_by_key: dict[str, str] = {}
    sequence_by_key: dict[str, str] = {}
    for proposal_id, (sequence, key) in proposals.items():
        first_by_key.setdefault(key, proposal_id)
        sequence_by_key.setdefault(key, sequence)
    individuals: list[ArchiveIndividual] = []
    for key, evaluations in outcomes_by_sequence.items():
        means = tuple(
            (
                name,
                _finite_mean([row[name] for _, row in evaluations], objective=name),
            )
            for name, _ in config.objective_weights
        )
        fitness = _finite_fitness(means, config)
        individuals.append(
            ArchiveIndividual(
                sequence=sequence_by_key[key],
                sequence_key=key,
                first_proposal_id=first_by_key[key],
                successful_evaluation_ids=tuple(
                    sorted(evaluation_id for evaluation_id, _ in evaluations)
                ),
                objective_means=means,
                fitness=fitness,
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
        b"amp/fixed-default-peptide-ga/elapsed-inventory/v1\0"
        + canonical_json_bytes(
            [
                _parse_event(raw, position=index)["scientific_elapsed_ns"]
                for index, raw in enumerate(verified.event_documents)
            ]
        )
    )
    base = {
        "authentication_status": authentication_status,
        "campaign_id": verified.header.campaign_id,
        "phase": verified.header.phase,
        "elapsed_inventory_sha256": elapsed_inventory,
        "event_count": len(verified.event_documents),
        "event_inventory_sha256": event_inventory,
        "header_sha256": verified.header_sha256,
        "head_seal_sha256": expected_head_seal_sha256,
        "individuals": [_individual_document(item) for item in individuals],
        "last_event_sha256": verified.last_event_sha256,
        "proposal_count": verified.proposal_count,
        "query_count": verified.query_count,
        "response_count": verified.response_count,
        "round_seal_inventory_sha256": round_inventory,
        "round_timing_receipt_inventory_sha256": round_timing_inventory,
        "round_timing_receipt_sha256s": list(verified.round_timing_receipt_sha256s),
        "round_count": expected_round_count,
        "scientific_elapsed_ns": verified.scientific_elapsed_ns,
        "truth_contract_sha256": truth_contract.sha256,
        "wave_id": wave_id,
    }
    digest = sha256_bytes(_ARCHIVE_HASH_DOMAIN + canonical_json_bytes(base))
    return AuthenticatedWaveArchive(
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
        authentication_status=authentication_status,
        round_seal_inventory_sha256=round_inventory,
        round_timing_receipt_sha256s=verified.round_timing_receipt_sha256s,
        round_timing_receipt_inventory_sha256=round_timing_inventory,
        event_inventory_sha256=event_inventory,
        elapsed_inventory_sha256=elapsed_inventory,
        truth_contract_sha256=truth_contract.sha256,
        individuals=tuple(individuals),
        archive_sha256=digest,
    )


def build_unverified_wave_archive_fixture(
    verified: VerifiedCampaign,
    *,
    wave_id: str,
    expected_head_seal_sha256: str,
    expected_round_count: int,
    expected_query_count: int,
    expected_response_count: int,
    config: PeptideGAConfig,
    truth_contract: FitnessTruthContract,
    policy_registry: AuthenticatedSelectionPolicyRegistry,
) -> AuthenticatedWaveArchive:
    """Build an explicitly unverified engineering fixture from a Python object."""

    return _build_wave_archive_from_verified(
        verified,
        wave_id=wave_id,
        expected_head_seal_sha256=expected_head_seal_sha256,
        expected_round_count=expected_round_count,
        expected_query_count=expected_query_count,
        expected_response_count=expected_response_count,
        config=config,
        truth_contract=truth_contract,
        policy_registry=policy_registry,
        authentication_status="unverified_object_fixture_only",
    )


def _build_authenticated_wave_archive(
    root: str | Path,
    *,
    trusted_parent: str | Path,
    wave_id: str,
    expected_header_sha256: str,
    expected_head_seal_sha256: str,
    expected_round_count: int,
    expected_query_count: int,
    expected_response_count: int,
    config: PeptideGAConfig,
    truth_contract: FitnessTruthContract,
    policy_registry: AuthenticatedSelectionPolicyRegistry,
) -> AuthenticatedWaveArchive:
    """Authenticate the on-disk campaign before constructing a next-wave view."""

    verified = verify_campaign(
        root,
        trusted_parent=trusted_parent,
        expected_header_sha256=expected_header_sha256,
        expected_head_seal_sha256=expected_head_seal_sha256,
        expected_round_count=expected_round_count,
        replay_limits=EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    )
    return _build_wave_archive_from_verified(
        verified,
        wave_id=wave_id,
        expected_head_seal_sha256=expected_head_seal_sha256,
        expected_round_count=expected_round_count,
        expected_query_count=expected_query_count,
        expected_response_count=expected_response_count,
        config=config,
        truth_contract=truth_contract,
        policy_registry=policy_registry,
        authentication_status="verified_campaign_path_truth_contract_unaccepted",
    )


def _individual_document(item: ArchiveIndividual) -> dict[str, object]:
    return {
        "first_proposal_id": item.first_proposal_id,
        "fitness": item.fitness,
        "objective_means": [list(row) for row in item.objective_means],
        "sequence": item.sequence,
        "sequence_key": item.sequence_key,
        "successful_evaluation_ids": list(item.successful_evaluation_ids),
    }


class _CounterRNG:
    def __init__(self, *, seed: int, input_sha256: str, attempt_index: int) -> None:
        self.seed = seed
        self.input_sha256 = input_sha256
        self.attempt_index = attempt_index

    def below(self, limit: int, label: str) -> int:
        _require(type(limit) is int and limit > 0, "RNG limit must be positive")
        ceiling = (1 << 256) // limit * limit
        for counter in range(_RNG_REJECTION_CAP):
            payload = canonical_json_bytes(
                {
                    "attempt_index": self.attempt_index,
                    "counter": counter,
                    "input_sha256": self.input_sha256,
                    "label": label,
                    "seed": self.seed,
                }
            )
            value = int.from_bytes(hashlib.sha256(_RNG_HASH_DOMAIN + payload).digest(), "big")
            if value < ceiling:
                return value % limit
        raise PeptideGAError("RNG rejection cap exhausted")


def _fraction(value: float) -> Fraction:
    return Fraction(str(value))


def _categorical(
    rng: _CounterRNG,
    rows: tuple[tuple[str, float], ...],
    *,
    label: str,
) -> tuple[str, float]:
    fractions = tuple((name, _fraction(weight)) for name, weight in rows)
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
    raise AssertionError("categorical draw escaped support")


def _select_parent(
    population: tuple[ArchiveIndividual, ...],
    config: PeptideGAConfig,
    rng: _CounterRNG,
    *,
    label: str,
    forbidden_key: str | None = None,
) -> tuple[ArchiveIndividual, tuple[ProbabilityFactor, ...]]:
    full = tuple(item for item in population if item.sequence_key != forbidden_key)
    _require(bool(full), "parent selection support is empty")
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
            label=f"{label}.branch",
        )
        pool = elite if branch == "elite" else full
    else:
        probability = 1.0
        pool = elite or full
    factors = [
        ProbabilityFactor("branch" if label == "parent.0" else f"{label}.branch", probability)
    ]
    participants: list[ArchiveIndividual] = []
    for draw_index in range(config.tournament_size):
        participants.append(pool[rng.below(len(pool), f"{label}.tournament.{draw_index}")])
        factors.append(ProbabilityFactor(f"{label}.tournament_draw.{draw_index}", 1.0 / len(pool)))
    winner = min(participants, key=lambda item: (-item.fitness, item.sequence_key))
    return winner, tuple(factors)


def _input_document(
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    *,
    batch_id: str,
    seed: int,
    config_sha256: str,
) -> dict[str, object]:
    return {
        "archive_sha256": archive.archive_sha256,
        "batch_id": batch_id,
        "campaign_id": archive.campaign_id,
        "config_sha256": config_sha256,
        "controller_private_reserve_identity_visible_to_adapter": False,
        "organizer_reference_input_present": False,
        "seed": seed,
        "submitted_set_sha256": exclusions.submitted_set_sha256,
        "training_homology_exclusion_enforced": False,
        "training_set_sha256": exclusions.training_set_sha256,
        "wave_id": archive.wave_id,
    }


def _proposal_document(value: ProposalRecord) -> dict[str, object]:
    return {
        "cheap_predictions": [list(row) for row in value.cheap_predictions],
        "hard_valid": value.hard_valid,
        "niche_id": value.niche_id,
        "policy_version": value.policy_version,
        "proposal_id": value.proposal_id,
        "proposal_round": value.proposal_round,
        "rejection_reason": value.rejection_reason,
        "rollout_id": value.rollout_id,
        "selection": {
            "eligible_proposal_ids": list(value.selection.eligible_proposal_ids),
            "policy_version": value.selection.policy_version,
            "propensity": [
                {"name": factor.name, "probability": factor.probability}
                for factor in value.selection.propensity.factors
            ],
            "seed": value.selection.seed,
            "selected": value.selection.selected,
            "selection_set_id": value.selection.selection_set_id,
        },
        "sequence": value.sequence,
    }


def _edge_document(value: EdgeRecord) -> dict[str, object]:
    return {
        "behavior_log_probabilities": list(value.behavior_log_probabilities),
        "edge_id": value.edge_id,
        "edit_description": list(value.edit_description),
        "operator": value.operator,
        "parent_sequence_keys": list(value.parent_sequence_keys),
        "proposal_id": value.proposal_id,
        "proposal_trace": [
            {"name": factor.name, "probability": factor.probability}
            for factor in value.proposal_trace.factors
        ],
        "random_stream": value.random_stream,
        "rollout_id": value.rollout_id,
        "sample_index": value.sample_index,
        "sampling_parameters": [list(row) for row in value.sampling_parameters],
    }


def _event_document(value: AdapterAttemptProvenance) -> dict[str, object]:
    return {
        name: getattr(value, name)
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
    }


def _attempt_document(value: PeptideGAAttempt) -> dict[str, object]:
    return {
        "accepted_position": value.accepted_position,
        "attempt_index": value.attempt_index,
        "edge": _edge_document(value.edge),
        "proposal": _proposal_document(value.proposal),
        "provenance": _event_document(value.provenance),
    }


def _output_document(batch: PeptideGABatch, *, include_sha256: bool) -> dict[str, object]:
    value = {
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
    if include_sha256:
        value["output_sha256"] = batch.output_sha256
    return value


def _empty_batch(
    archive: AuthenticatedWaveArchive,
    *,
    config: PeptideGAConfig,
    batch_id: str,
    seed: int,
    input_sha256: str,
    public_exclusion_receipt: PublicExclusionReceipt,
) -> PeptideGABatch:
    return PeptideGABatch(
        artifact=PEPTIDE_GA_ARTIFACT,
        schema_version=1,
        status="in_progress",
        campaign_id=archive.campaign_id,
        wave_id=archive.wave_id,
        batch_id=batch_id,
        seed=seed,
        config_sha256=config.config_sha256,
        archive_sha256=archive.archive_sha256,
        input_sha256=input_sha256,
        public_exclusion_receipt=public_exclusion_receipt,
        attempts=(),
        accepted_proposal_ids=(),
        accepted_sequences=(),
        next_attempt_index=0,
        output_sha256="0" * 64,
    )


def _generate_peptide_ga_batch(
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    *,
    config: PeptideGAConfig,
    batch_id: str,
    seed: int,
    resume: PeptideGABatch | None = None,
    max_new_attempts: int | None = None,
) -> PeptideGABatch:
    """Generate or resume the 14 method seats; reserve seats stay controller-private."""

    _require(type(archive) is AuthenticatedWaveArchive, "archive view type differs")
    _require(type(exclusions) is CollisionExclusions, "collision exclusions type differs")
    _require(type(config) is PeptideGAConfig, "peptide-GA config type differs")
    archive.__post_init__()
    exclusions.__post_init__()
    config.__post_init__()
    _require(
        sha256_bytes(config.config_source_bytes) == config.config_sha256, "config seal differs"
    )
    _require(
        exclusions
        == CollisionExclusions.from_keys(
            training_sequence_keys=exclusions.training_sequence_keys,
            submitted_sequence_keys=exclusions.submitted_sequence_keys,
        ),
        "exclusion seals differ",
    )
    archive_base = {
        "authentication_status": archive.authentication_status,
        "campaign_id": archive.campaign_id,
        "phase": archive.phase,
        "elapsed_inventory_sha256": archive.elapsed_inventory_sha256,
        "event_count": archive.event_count,
        "event_inventory_sha256": archive.event_inventory_sha256,
        "header_sha256": archive.header_sha256,
        "head_seal_sha256": archive.head_seal_sha256,
        "individuals": [_individual_document(item) for item in archive.individuals],
        "last_event_sha256": archive.last_event_sha256,
        "proposal_count": archive.proposal_count,
        "query_count": archive.query_count,
        "response_count": archive.response_count,
        "round_count": archive.round_count,
        "round_seal_inventory_sha256": archive.round_seal_inventory_sha256,
        "round_timing_receipt_inventory_sha256": (archive.round_timing_receipt_inventory_sha256),
        "round_timing_receipt_sha256s": list(archive.round_timing_receipt_sha256s),
        "scientific_elapsed_ns": archive.scientific_elapsed_ns,
        "truth_contract_sha256": archive.truth_contract_sha256,
        "wave_id": archive.wave_id,
    }
    _require(
        sha256_bytes(_ARCHIVE_HASH_DOMAIN + canonical_json_bytes(archive_base))
        == archive.archive_sha256,
        "archive seal differs",
    )
    _identifier(batch_id, label="batch ID", maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH)
    _bounded_integer(seed, maximum=PEPTIDE_GA_SIGNED_63_MAX, label="seed")
    _require(bool(archive.individuals), "archive population is empty")
    if max_new_attempts is not None:
        _require(
            type(max_new_attempts) is int and 0 <= max_new_attempts <= config.proposal_attempt_cap,
            "new-attempt limit differs",
        )
    input_sha256 = sha256_bytes(
        _INPUT_HASH_DOMAIN
        + canonical_json_bytes(
            _input_document(
                archive,
                exclusions,
                batch_id=batch_id,
                seed=seed,
                config_sha256=config.config_sha256,
            )
        )
    )
    public_exclusion_receipt = make_public_exclusion_receipt(
        campaign_id=archive.campaign_id,
        wave_id=archive.wave_id,
        authenticated_pre_wave_head_sha256=archive.head_seal_sha256,
        authenticated_pre_wave_round_count=archive.round_count,
        batch_id=batch_id,
        seed=seed,
        config_sha256=config.config_sha256,
        input_sha256=input_sha256,
        training_count=exclusions.training_count,
        submitted_count=exclusions.submitted_count,
        training_set_sha256=exclusions.training_set_sha256,
        submitted_set_sha256=exclusions.submitted_set_sha256,
    )
    current = _empty_batch(
        archive,
        config=config,
        batch_id=batch_id,
        seed=seed,
        input_sha256=input_sha256,
        public_exclusion_receipt=public_exclusion_receipt,
    )
    if resume is not None:
        _require(type(resume) is PeptideGABatch, "resume batch type differs")
        preflight_peptide_ga_batch_structure(resume)
        _require(resume.status == "in_progress", "only an in-progress batch can resume")
        _require(
            (
                resume.campaign_id,
                resume.wave_id,
                resume.batch_id,
                resume.seed,
                resume.config_sha256,
                resume.archive_sha256,
                resume.input_sha256,
            )
            == (
                archive.campaign_id,
                archive.wave_id,
                batch_id,
                seed,
                config.config_sha256,
                archive.archive_sha256,
                input_sha256,
            ),
            "resume identity differs",
        )
        # A resume is accepted only if its entire retained prefix is byte-for-byte
        # the deterministic prefix produced from the same inputs.
        replay = _generate_peptide_ga_batch(
            archive,
            exclusions,
            config=config,
            batch_id=batch_id,
            seed=seed,
            max_new_attempts=resume.next_attempt_index,
        )
        _require(replay.attempts == resume.attempts, "resume attempt prefix differs")
        _require(
            replay.accepted_proposal_ids == resume.accepted_proposal_ids,
            "resume accepted prefix differs",
        )
        _require(
            replay.accepted_sequences == resume.accepted_sequences,
            "resume accepted sequence prefix differs",
        )
        _require(
            replay.public_exclusion_receipt == resume.public_exclusion_receipt,
            "resume public-exclusion receipt differs",
        )
        _require(replay.output_sha256 == resume.output_sha256, "resume output seal differs")
        current = resume

    attempts = list(current.attempts)
    accepted_ids = list(current.accepted_proposal_ids)
    accepted_sequences = list(current.accepted_sequences)
    accepted_sequence_keys = {sequence_key(value) for value in accepted_sequences}
    first_generated: dict[str, str] = {}
    for item in attempts:
        first_generated.setdefault(item.proposal.sequence_key, item.proposal.proposal_id)
    attempt_index = current.next_attempt_index
    stop_index = config.proposal_attempt_cap
    if max_new_attempts is not None:
        stop_index = min(stop_index, attempt_index + max_new_attempts)
    population = archive.individuals
    archive_keys = {item.sequence_key for item in population}
    while len(accepted_ids) < config.candidate_prefix_size and attempt_index < stop_index:
        rng = _CounterRNG(seed=seed, input_sha256=input_sha256, attempt_index=attempt_index)
        parent1, parent_factors = _select_parent(population, config, rng, label="parent.0")
        available = [
            (name, rate)
            for name, rate in config.operator_rates
            if not (
                (name == "insertion" and len(parent1.sequence) >= config.max_length)
                or (name == "deletion" and len(parent1.sequence) <= config.min_length)
                or (name == "two_parent_crossover" and len(population) < 2)
            )
        ]
        operator, operator_probability = _categorical(rng, tuple(available), label="operator")
        factors = list(parent_factors)
        factors.append(ProbabilityFactor("operator", operator_probability))
        parents = (parent1.sequence_key,)
        if operator == "substitution":
            position = rng.below(len(parent1.sequence), "edit.position")
            residues = config.alphabet.replace(parent1.sequence[position], "")
            residue = residues[rng.below(len(residues), "edit.residue")]
            child = parent1.sequence[:position] + residue + parent1.sequence[position + 1 :]
            edit = ("substitution", position, parent1.sequence[position], residue)
            factors.extend(
                (
                    ProbabilityFactor("edit.position", 1.0 / len(parent1.sequence)),
                    ProbabilityFactor("edit.residue", 1.0 / len(residues)),
                )
            )
        elif operator == "insertion":
            position = rng.below(len(parent1.sequence) + 1, "edit.position")
            residue = config.alphabet[rng.below(len(config.alphabet), "edit.residue")]
            child = parent1.sequence[:position] + residue + parent1.sequence[position:]
            edit = ("insertion", position, residue)
            factors.extend(
                (
                    ProbabilityFactor("edit.position", 1.0 / (len(parent1.sequence) + 1)),
                    ProbabilityFactor("edit.residue", 1.0 / len(config.alphabet)),
                )
            )
        elif operator == "deletion":
            position = rng.below(len(parent1.sequence), "edit.position")
            child = parent1.sequence[:position] + parent1.sequence[position + 1 :]
            edit = ("deletion", position, parent1.sequence[position])
            factors.append(ProbabilityFactor("edit.position", 1.0 / len(parent1.sequence)))
        else:
            parent2, parent2_factors = _select_parent(
                population, config, rng, label="parent.1", forbidden_key=parent1.sequence_key
            )
            cut1 = 1 + rng.below(len(parent1.sequence) - 1, "edit.cut.0")
            cut2 = 1 + rng.below(len(parent2.sequence) - 1, "edit.cut.1")
            child = parent1.sequence[:cut1] + parent2.sequence[cut2:]
            parents = (parent1.sequence_key, parent2.sequence_key)
            edit = ("two_parent_crossover", cut1, cut2, parent1.sequence_key, parent2.sequence_key)
            factors.extend(parent2_factors)
            factors.extend(
                (
                    ProbabilityFactor("edit.cut.0", 1.0 / (len(parent1.sequence) - 1)),
                    ProbabilityFactor("edit.cut.1", 1.0 / (len(parent2.sequence) - 1)),
                )
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
        elif duplicate or key in accepted_sequence_keys:
            reason = "generated_duplicate"
        else:
            reason = None
        hard_valid = reason is None
        proposal_id = (
            "ga-proposal-"
            + sha256_bytes(
                b"amp/fixed-default-peptide-ga/proposal-id/v1\0"
                + bytes.fromhex(input_sha256)
                + attempt_index.to_bytes(8, "big")
            )[:32]
        )
        edge_id = "ga-edge-" + proposal_id.removeprefix("ga-proposal-")
        rollout_id = f"ga-rollout-{batch_id}-{seed}"
        proposal = ProposalRecord(
            proposal_id=proposal_id,
            rollout_id=rollout_id,
            sequence=child,
            hard_valid=hard_valid,
            rejection_reason=reason,
            selection=SelectionDecision(
                selected=False,
                propensity=ProbabilityTrace((ProbabilityFactor("controller_selection", 0.0),)),
                selection_set_id=f"ga-controller-pending-{batch_id}-{attempt_index}",
                eligible_proposal_ids=(),
                policy_version=PEPTIDE_GA_PENDING_SELECTION_POLICY_VERSION,
                seed=seed,
            ),
            policy_version=PEPTIDE_GA_POLICY_VERSION,
            proposal_round=archive.round_count,
            niche_id="fixed-default-peptide-ga",
            cheap_predictions=(),
        )
        edge = EdgeRecord(
            edge_id=edge_id,
            proposal_id=proposal_id,
            rollout_id=rollout_id,
            parent_sequence_keys=parents,
            operator=operator,
            edit_description=edit,
            proposal_trace=ProbabilityTrace(tuple(factors)),
            random_stream=seed,
            sample_index=attempt_index,
            sampling_parameters=(
                ("algorithm", PEPTIDE_GA_POLICY_VERSION),
                ("archive_sha256", archive.archive_sha256),
                ("config_sha256", config.config_sha256),
                ("input_sha256", input_sha256),
                ("rng", PEPTIDE_GA_RNG_VERSION),
            ),
        )
        first_id = first_generated.get(key, proposal_id)
        accepted_position = len(accepted_ids) if hard_valid else None
        provenance = AdapterAttemptProvenance(
            event_index=attempt_index,
            proposal_id=proposal_id,
            edge_id=edge_id,
            sequence_key=key,
            first_attempt_proposal_id=first_id,
            duplicate_within_prefix=duplicate,
            hard_valid=hard_valid,
            rejection_reason=reason,
        )
        attempts.append(
            PeptideGAAttempt(attempt_index, accepted_position, proposal, edge, provenance)
        )
        first_generated.setdefault(key, proposal_id)
        if hard_valid:
            accepted_ids.append(proposal_id)
            accepted_sequences.append(child)
            accepted_sequence_keys.add(key)
        attempt_index += 1

    if len(accepted_ids) == config.candidate_prefix_size:
        status = "complete"
    elif attempt_index == config.proposal_attempt_cap:
        status = "attempt_cap_exhausted"
    else:
        status = "in_progress"
    batch = replace(
        current,
        status=status,
        attempts=tuple(attempts),
        accepted_proposal_ids=tuple(accepted_ids),
        accepted_sequences=tuple(accepted_sequences),
        next_attempt_index=attempt_index,
        output_sha256="0" * 64,
    )
    output_sha256 = sha256_bytes(
        _OUTPUT_HASH_DOMAIN + canonical_json_bytes(_output_document(batch, include_sha256=False))
    )
    return replace(batch, output_sha256=output_sha256)


def generate_unverified_peptide_ga_fixture_batch(
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    *,
    config: PeptideGAConfig,
    batch_id: str,
    seed: int,
    resume: PeptideGABatch | None = None,
    max_new_attempts: int | None = None,
) -> PeptideGABatch:
    """Generate only from an explicitly unverified engineering fixture."""

    _require(type(archive) is AuthenticatedWaveArchive, "archive view type differs")
    _require(
        archive.authentication_status == "unverified_object_fixture_only",
        "object generation accepts only an explicitly unverified fixture",
    )
    return _generate_peptide_ga_batch(
        archive,
        exclusions,
        config=config,
        batch_id=batch_id,
        seed=seed,
        resume=resume,
        max_new_attempts=max_new_attempts,
    )


def generate_peptide_ga_batch_from_path(
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
    batch_id: str,
    seed: int,
    resume: PeptideGABatch | None = None,
    max_new_attempts: int | None = None,
) -> PeptideGABatch:
    """Authenticate and replay the campaign path in the same call that generates."""

    _require(type(exclusions) is CollisionExclusions, "collision exclusions type differs")
    _require(type(config) is PeptideGAConfig, "peptide-GA config type differs")
    _require(type(truth_contract) is FitnessTruthContract, "truth contract type differs")
    exclusions.__post_init__()
    config.__post_init__()
    truth_contract.__post_init__()
    _identifier(wave_id, label="wave ID")
    _identifier(batch_id, label="batch ID", maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH)
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
    _bounded_integer(seed, maximum=PEPTIDE_GA_SIGNED_63_MAX, label="seed")
    if max_new_attempts is not None:
        _bounded_integer(
            max_new_attempts,
            maximum=PEPTIDE_GA_ATTEMPT_CAP,
            label="new-attempt limit",
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
    archive = _build_authenticated_wave_archive(
        root,
        trusted_parent=trusted_parent,
        wave_id=wave_id,
        expected_header_sha256=expected_header_sha256,
        expected_head_seal_sha256=expected_head_seal_sha256,
        expected_round_count=expected_round_count,
        expected_query_count=expected_query_count,
        expected_response_count=expected_response_count,
        config=config,
        truth_contract=truth_contract,
        policy_registry=policy_registry,
    )
    exclusion_authority = load_collision_exclusion_authority_from_paths(
        exclusion_asset_path,
        exclusion_receipt_path,
        trusted_parent=trusted_exclusion_parent,
        campaign_root=root,
        expected_asset_sha256=expected_exclusion_asset_sha256,
        expected_receipt_sha256=expected_exclusion_receipt_sha256,
        expected_issuer_identity_sha256=expected_exclusion_issuer_identity_sha256,
        campaign_id=archive.campaign_id,
        phase=archive.phase,
        wave_id=wave_id,
        authenticated_pre_wave_head_sha256=archive.head_seal_sha256,
        authenticated_pre_wave_round_count=archive.round_count,
        protocol_sha256=FROZEN_PROTOCOL_SHA256,
        config_sha256=config.config_sha256,
        policy_registry_sha256=policy_registry.registry_sha256,
        selection_policy_implementation_manifest_sha256=(
            policy_registry.implementation_manifest_sha256
        ),
        truth_contract_sha256=truth_contract.sha256,
    )
    _require(exclusions == exclusion_authority.exclusions, "external exclusion inventory differs")
    return _generate_peptide_ga_batch(
        archive,
        exclusion_authority.exclusions,
        config=config,
        batch_id=batch_id,
        seed=seed,
        resume=resume,
        max_new_attempts=max_new_attempts,
    )


__all__ = [
    "build_unverified_wave_archive_fixture",
    "generate_peptide_ga_batch_from_path",
    "generate_unverified_peptide_ga_fixture_batch",
    "load_peptide_ga_config",
]

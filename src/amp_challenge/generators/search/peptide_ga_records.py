"""Immutable, non-authorizing records for the fixed-default peptide-GA adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from amp_challenge.generators.search.records import (
    EdgeRecord,
    ProbabilityFactor,
    ProbabilityTrace,
    ProposalRecord,
    SelectionDecision,
)

PEPTIDE_GA_ARTIFACT = "fixed_default_peptide_ga_proposal_batch_v1"
PEPTIDE_GA_POLICY_VERSION = "fixed-default-peptide-ga-development-v1"
PEPTIDE_GA_RNG_VERSION = "sha256-rejection-counter-v1"
PEPTIDE_GA_ATTEMPT_CAP = 65_536
PEPTIDE_GA_METHOD_SEATS = 14
PEPTIDE_GA_PRIVATE_RESERVE_SEATS = 2
PEPTIDE_GA_CANDIDATE_PREFIX_SIZE = 256
PEPTIDE_GA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PEPTIDE_GA_MIN_LENGTH = 8
PEPTIDE_GA_MAX_LENGTH = 50
# The frozen central v1 protocol still uses this historical method identifier.
# It is retained only as an explicit compatibility namespace; no adapter artifact
# or policy below claims that fixed defaults were empirically tuned.
PEPTIDE_GA_CAMPAIGN_CONFIGURATION_ID = "tuned_peptide_ga"
PEPTIDE_GA_NAMESPACE = "fixed-default-peptide-ga"
PEPTIDE_GA_BOOTSTRAP_NAMESPACE = "fixed-default-peptide-ga-bootstrap"
PEPTIDE_GA_BOOTSTRAP_POLICY_VERSION = "fixed-default-peptide-ga-bootstrap-v1"
PEPTIDE_GA_BOOTSTRAP_SELECTION_POLICY_VERSION = "fixed-default-peptide-ga-bootstrap-selection-v1"
PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION = "fixed-default-peptide-ga-controller-selection-v1"
PEPTIDE_GA_PENDING_SELECTION_POLICY_VERSION = (
    "fixed-default-peptide-ga-pending-controller-selection-v1"
)
PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT = "bootstrap_selection_membership"
PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT = "controller_first_available_prefix_positions"
PEPTIDE_GA_SELECTION_POLICY_SOURCE_PATH = (
    "src/amp_challenge/generators/search/peptide_ga_selection_policy_impl_v1.py"
)
PEPTIDE_GA_SELECTION_POLICY_SOURCE_SHA256 = (
    "7b231ad6454b8c710408ef63d556369bf239cd5a4734777f23f9250e126e1801"
)
PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_RECIPE = (
    "sha256-domain-canonical-json-complete-source-v1"
)
PEPTIDE_GA_SELECTION_ID_DERIVATION = "fixed-default-peptide-ga-wave-selection-id-v1"
PEPTIDE_GA_SELECTION_SET_ID_PREFIX = "fixed-ga-selection-"
PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT = 65_536
PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT = 512
PEPTIDE_GA_TRAINING_EXCLUSION_MAX_BYTES = 4_390_913
PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_BYTES = 34_305
PEPTIDE_GA_CONFIG_MAX_BYTES = 65_536
PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES = 65_536
PEPTIDE_GA_POLICY_REGISTRY_MAX_ROWS = 64
PEPTIDE_GA_SELECTION_ELIGIBLE_MAX_COUNT = PEPTIDE_GA_CANDIDATE_PREFIX_SIZE
PEPTIDE_GA_BATCH_ID_MAX_LENGTH = 64
PEPTIDE_GA_IDENTIFIER_MAX_LENGTH = 128
PEPTIDE_GA_SIGNED_63_MAX = (1 << 63) - 1
PEPTIDE_GA_MAX_ROUNDS = 30
PEPTIDE_GA_MAX_EVENTS = 66_561
PEPTIDE_GA_MAX_PROPOSALS = 65_536
PEPTIDE_GA_MAX_QUERIES = 512
PEPTIDE_GA_MAX_RESPONSES = 512
PEPTIDE_GA_MAX_SCIENTIFIC_ELAPSED_NS = 7_200_000_000_000
PEPTIDE_GA_ROUND_TIMING_INVENTORY_HASH_DOMAIN = b"amp/evolutionary-kl/bridge-round-timing/v1\0"
PEPTIDE_GA_SELECTION_ELIGIBLE_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/ordered-eligible/v1\0"
PEPTIDE_GA_SELECTION_SET_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/selection-set/v1\0"
PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_HASH_DOMAIN = (
    b"amp/fixed-default-peptide-ga/selection-policy-implementation/v1\0"
)
PEPTIDE_GA_SELECTION_POLICY_MANIFEST_HASH_DOMAIN = (
    b"amp/fixed-default-peptide-ga/selection-policy-manifest/v1\0"
)
PEPTIDE_GA_PUBLIC_EXCLUSION_RECEIPT_HASH_DOMAIN = (
    b"amp/fixed-default-peptide-ga/public-exclusion-receipt/v1\0"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class PeptideGAError(ValueError):
    """Raised when the adapter or its independent verifier fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PeptideGAError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    assert isinstance(value, str)
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


def _nonnegative_bounded_integer(value: object, *, maximum: int, label: str) -> int:
    _require(
        type(value) is int and 0 <= value <= maximum,
        f"{label} must be an integer in [0, {maximum}]",
    )
    assert isinstance(value, int)
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode one canonical JSON value without accepting non-finite numbers."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def selection_policy_implementation_document(
    *,
    policy_version: str,
    entry_point: str,
    source_sha256: str = PEPTIDE_GA_SELECTION_POLICY_SOURCE_SHA256,
) -> dict[str, str]:
    """Canonical recipe binding one named policy to the complete source bytes."""

    return {
        "entry_point": _identifier(entry_point, label="selection implementation entry point"),
        "policy_version": _identifier(
            policy_version,
            label="selection implementation policy version",
        ),
        "recipe": PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_RECIPE,
        "source_path": PEPTIDE_GA_SELECTION_POLICY_SOURCE_PATH,
        "source_sha256": _sha256(
            source_sha256,
            label="selection implementation source digest",
        ),
    }


def derive_selection_policy_implementation_sha256(
    *,
    policy_version: str,
    entry_point: str,
    source_sha256: str = PEPTIDE_GA_SELECTION_POLICY_SOURCE_SHA256,
) -> str:
    return sha256_bytes(
        PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_HASH_DOMAIN
        + canonical_json_bytes(
            selection_policy_implementation_document(
                policy_version=policy_version,
                entry_point=entry_point,
                source_sha256=source_sha256,
            )
        )
    )


PEPTIDE_GA_BOOTSTRAP_SELECTION_IMPLEMENTATION_SHA256 = (
    derive_selection_policy_implementation_sha256(
        policy_version=PEPTIDE_GA_BOOTSTRAP_SELECTION_POLICY_VERSION,
        entry_point=PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT,
    )
)
PEPTIDE_GA_CONTROLLER_SELECTION_IMPLEMENTATION_SHA256 = (
    derive_selection_policy_implementation_sha256(
        policy_version=PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
        entry_point=PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT,
    )
)
PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256 = sha256_bytes(
    PEPTIDE_GA_SELECTION_POLICY_MANIFEST_HASH_DOMAIN
    + canonical_json_bytes(
        [
            selection_policy_implementation_document(
                policy_version=PEPTIDE_GA_BOOTSTRAP_SELECTION_POLICY_VERSION,
                entry_point=PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT,
            ),
            selection_policy_implementation_document(
                policy_version=PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
                entry_point=PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT,
            ),
        ]
    )
)


def round_timing_receipt_inventory_sha256(
    round_timing_receipt_sha256s: tuple[str, ...],
) -> str:
    """Match the campaign-secondary bridge's ordered timing-inventory seal."""

    _require(
        type(round_timing_receipt_sha256s) is tuple
        and 0 < len(round_timing_receipt_sha256s) <= PEPTIDE_GA_MAX_ROUNDS,
        "round timing-receipt inventory differs",
    )
    rows = bytearray()
    for index, digest in enumerate(round_timing_receipt_sha256s):
        rows.extend(
            canonical_json_bytes(
                {
                    "round_index": index,
                    "timing_receipt_sha256": _sha256(digest, label=f"round timing receipt {index}"),
                }
            )
        )
        rows.extend(b"\n")
    return sha256_bytes(PEPTIDE_GA_ROUND_TIMING_INVENTORY_HASH_DOMAIN + bytes(rows))


def sequence_key(sequence: str) -> str:
    return sha256_bytes(sequence.encode("ascii"))


def exclusion_set_sha256(label: str, keys: tuple[str, ...]) -> str:
    return sha256_bytes(
        b"amp/fixed-default-peptide-ga/exclusion-set/v1\0"
        + label.encode("ascii")
        + b"\0"
        + canonical_json_bytes(list(keys))
    )


def ordered_eligible_proposal_ids_sha256(proposal_ids: tuple[str, ...]) -> str:
    """Seal one bounded ordered public eligibility list without sorting it."""

    _require(
        type(proposal_ids) is tuple
        and 0 < len(proposal_ids) <= PEPTIDE_GA_SELECTION_ELIGIBLE_MAX_COUNT,
        "eligible proposal inventory differs",
    )
    observed: set[str] = set()
    for proposal_id in proposal_ids:
        normalized = _identifier(proposal_id, label="eligible proposal ID")
        _require(normalized not in observed, "eligible proposal IDs are not unique")
        observed.add(normalized)
    return sha256_bytes(
        PEPTIDE_GA_SELECTION_ELIGIBLE_HASH_DOMAIN + canonical_json_bytes(list(proposal_ids))
    )


def derive_wave_selection_set_id(
    *,
    namespace: str,
    campaign_id: str,
    phase: str,
    wave_id: str,
    authenticated_pre_wave_head_sha256: str,
    authenticated_pre_wave_round_count: int,
    batch_output_sha256: str,
    ordered_eligible_sha256: str,
    policy_version: str,
    policy_implementation_sha256: str,
    seed: int,
    config_sha256: str,
    derivation: str = PEPTIDE_GA_SELECTION_ID_DERIVATION,
) -> str:
    """Derive one collision-resistant, bounded, reserve-blind selection-set ID."""

    document = {
        "authenticated_pre_wave_head_sha256": _sha256(
            authenticated_pre_wave_head_sha256,
            label="selection pre-wave head",
        ),
        "authenticated_pre_wave_round_count": _nonnegative_bounded_integer(
            authenticated_pre_wave_round_count,
            maximum=PEPTIDE_GA_MAX_ROUNDS,
            label="selection pre-wave round count",
        ),
        "batch_output_sha256": _sha256(
            batch_output_sha256,
            label="selection batch output digest",
        ),
        "campaign_id": _identifier(campaign_id, label="selection campaign ID"),
        "config_sha256": _sha256(config_sha256, label="selection config digest"),
        "derivation": _identifier(derivation, label="selection ID derivation"),
        "namespace": _identifier(namespace, label="selection namespace"),
        "ordered_eligible_sha256": _sha256(
            ordered_eligible_sha256,
            label="selection eligible digest",
        ),
        "phase": _identifier(phase, label="selection phase"),
        "policy_implementation_sha256": _sha256(
            policy_implementation_sha256,
            label="selection policy implementation digest",
        ),
        "policy_version": _identifier(policy_version, label="selection policy version"),
        "seed": _nonnegative_bounded_integer(
            seed,
            maximum=PEPTIDE_GA_SIGNED_63_MAX,
            label="selection seed",
        ),
        "wave_id": _identifier(wave_id, label="selection wave ID"),
    }
    return PEPTIDE_GA_SELECTION_SET_ID_PREFIX + sha256_bytes(
        PEPTIDE_GA_SELECTION_SET_HASH_DOMAIN + canonical_json_bytes(document)
    )


@dataclass(frozen=True, slots=True)
class PeptideGAConfig:
    """Strictly frozen development configuration; never an execution grant."""

    alphabet: str
    min_length: int
    max_length: int
    proposal_attempt_cap: int
    method_controlled_seats: int
    controller_private_reserve_seats: int
    candidate_prefix_size: int
    tournament_size: int
    elite_fraction: float
    elite_parent_probability: float
    operator_rates: tuple[tuple[str, float], ...]
    objective_weights: tuple[tuple[str, float], ...]
    execution_authorized: bool
    scientific_evidence_accepted: bool
    automatic_production_eligible: bool
    biological_superiority_claim_allowed: bool
    tuning_study_accepted: bool
    config_sha256: str
    config_source_bytes: bytes

    def __post_init__(self) -> None:
        if (
            type(self.alphabet) is not str
            or self.alphabet != PEPTIDE_GA_ALPHABET
            or len(set(self.alphabet)) != len(self.alphabet)
        ):
            raise PeptideGAError("peptide-GA alphabet differs from the frozen canonical support")
        if (
            type(self.min_length) is not int
            or type(self.max_length) is not int
            or (
                self.min_length,
                self.max_length,
            )
            != (PEPTIDE_GA_MIN_LENGTH, PEPTIDE_GA_MAX_LENGTH)
        ):
            raise PeptideGAError("peptide-GA length support differs")
        if type(self.proposal_attempt_cap) is not int or (
            self.proposal_attempt_cap != PEPTIDE_GA_ATTEMPT_CAP
        ):
            raise PeptideGAError("peptide-GA proposal attempt cap differs")
        if type(self.method_controlled_seats) is not int or (
            self.method_controlled_seats != PEPTIDE_GA_METHOD_SEATS
        ):
            raise PeptideGAError("peptide-GA method seat count differs")
        if type(self.controller_private_reserve_seats) is not int or (
            self.controller_private_reserve_seats != PEPTIDE_GA_PRIVATE_RESERVE_SEATS
        ):
            raise PeptideGAError("controller-private reserve seat count differs")
        if type(self.candidate_prefix_size) is not int or (
            self.candidate_prefix_size != PEPTIDE_GA_CANDIDATE_PREFIX_SIZE
        ):
            raise PeptideGAError("peptide-GA candidate prefix size differs")
        if type(self.tournament_size) is not int or self.tournament_size < 1:
            raise PeptideGAError("tournament size must be positive")
        if self.tournament_size != 3:
            raise PeptideGAError("development tournament size differs")
        for name, value in (
            ("elite_fraction", self.elite_fraction),
            ("elite_parent_probability", self.elite_parent_probability),
        ):
            if type(value) is not float or not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise PeptideGAError(f"{name} must be a finite float in (0, 1]")
        _require(
            type(self.operator_rates) is tuple and len(self.operator_rates) == 4,
            "operator-rate inventory differs",
        )
        expected_operators = ("substitution", "insertion", "deletion", "two_parent_crossover")
        _require(
            all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is float
                for row in self.operator_rates
            ),
            "operator-rate rows differ",
        )
        if tuple(name for name, _ in self.operator_rates) != expected_operators:
            raise PeptideGAError("operator names/order differ")
        if any(
            type(rate) is not float or not math.isfinite(rate) or rate <= 0.0
            for _, rate in self.operator_rates
        ):
            raise PeptideGAError("operator rates must be positive finite floats")
        if not math.isclose(math.fsum(rate for _, rate in self.operator_rates), 1.0, abs_tol=1e-15):
            raise PeptideGAError("operator rates must sum to one")
        _require(
            type(self.objective_weights) is tuple and len(self.objective_weights) == 2,
            "objective-weight inventory differs",
        )
        _require(
            all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is float
                for row in self.objective_weights
            ),
            "objective-weight rows differ",
        )
        if not self.objective_weights or any(
            type(weight) is not float or not math.isfinite(weight) or weight <= 0.0
            for _, weight in self.objective_weights
        ):
            raise PeptideGAError("objective weights must be positive finite floats")
        if len({name for name, _ in self.objective_weights}) != len(self.objective_weights):
            raise PeptideGAError("objective names must be unique")
        if not math.isclose(
            math.fsum(weight for _, weight in self.objective_weights), 1.0, abs_tol=1e-15
        ):
            raise PeptideGAError("objective weights must sum to one")
        if (self.elite_fraction, self.elite_parent_probability) != (0.25, 0.5):
            raise PeptideGAError("development elitism settings differ")
        if self.operator_rates != (
            ("substitution", 0.45),
            ("insertion", 0.15),
            ("deletion", 0.15),
            ("two_parent_crossover", 0.25),
        ):
            raise PeptideGAError("development operator rates differ")
        if self.objective_weights != (
            ("gram_positive_activity", 0.5),
            ("gram_negative_activity", 0.5),
        ):
            raise PeptideGAError("development objective weights differ")
        if any(
            value is not False
            for value in (
                self.execution_authorized,
                self.scientific_evidence_accepted,
                self.automatic_production_eligible,
                self.biological_superiority_claim_allowed,
                self.tuning_study_accepted,
            )
        ):
            raise PeptideGAError("all peptide-GA authorization/evidence/tuning flags must be false")
        _sha256(self.config_sha256, label="config SHA-256")
        if type(self.config_source_bytes) is not bytes:
            raise PeptideGAError("config source bytes are unavailable")
        if not 0 < len(self.config_source_bytes) <= PEPTIDE_GA_CONFIG_MAX_BYTES:
            raise PeptideGAError("config source byte ceiling differs")
        if sha256_bytes(self.config_source_bytes) != self.config_sha256:
            raise PeptideGAError("config source digest differs")

    def operator_rate(self, name: str) -> float:
        return dict(self.operator_rates)[name]


@dataclass(frozen=True, slots=True)
class ArchiveIndividual:
    """One unique sequence and scalarized fitness from a sealed completed wave."""

    sequence: str
    sequence_key: str
    first_proposal_id: str
    successful_evaluation_ids: tuple[str, ...]
    objective_means: tuple[tuple[str, float], ...]
    fitness: float

    def __post_init__(self) -> None:
        _require(type(self.sequence) is str, "archive sequence type differs")
        _require(
            PEPTIDE_GA_MIN_LENGTH <= len(self.sequence) <= PEPTIDE_GA_MAX_LENGTH
            and self.sequence == self.sequence.upper()
            and set(self.sequence) <= set(PEPTIDE_GA_ALPHABET),
            "archive sequence support differs",
        )
        _sha256(self.sequence_key, label="archive sequence key")
        _require(sequence_key(self.sequence) == self.sequence_key, "archive sequence key differs")
        _identifier(self.first_proposal_id, label="archive first proposal ID")
        _require(
            type(self.successful_evaluation_ids) is tuple
            and 0 < len(self.successful_evaluation_ids) <= PEPTIDE_GA_MAX_RESPONSES,
            "archive evaluation inventory differs",
        )
        previous_id: str | None = None
        for evaluation_id in self.successful_evaluation_ids:
            _identifier(evaluation_id, label="archive evaluation ID")
            _require(
                previous_id is None or previous_id < evaluation_id,
                "archive evaluation IDs must be sorted and unique",
            )
            previous_id = evaluation_id
        _require(
            type(self.objective_means) is tuple
            and len(self.objective_means) == 2
            and all(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is float
                for row in self.objective_means
            ),
            "archive objective means differ",
        )
        _require(
            tuple(name for name, _ in self.objective_means)
            == ("gram_positive_activity", "gram_negative_activity"),
            "archive objective names differ",
        )
        _require(
            all(type(value) is float and math.isfinite(value) for _, value in self.objective_means),
            "archive objective mean is non-finite",
        )
        _require(
            type(self.fitness) is float and math.isfinite(self.fitness), "archive fitness differs"
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedWaveArchive:
    """Immutable adapter view reconstructed from one authenticated campaign head."""

    campaign_id: str
    phase: str
    wave_id: str
    header_sha256: str
    head_seal_sha256: str
    last_event_sha256: str
    round_count: int
    event_count: int
    proposal_count: int
    query_count: int
    response_count: int
    scientific_elapsed_ns: int
    authentication_status: Literal[
        "verified_campaign_path_truth_contract_unaccepted",
        "unverified_object_fixture_only",
    ]
    round_seal_inventory_sha256: str
    round_timing_receipt_sha256s: tuple[str, ...]
    round_timing_receipt_inventory_sha256: str
    event_inventory_sha256: str
    elapsed_inventory_sha256: str
    truth_contract_sha256: str
    individuals: tuple[ArchiveIndividual, ...]
    archive_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="archive campaign ID")
        _require(
            type(self.phase) is str and self.phase in {"screen", "confirmation"},
            "archive campaign phase differs",
        )
        _identifier(self.wave_id, label="archive wave ID")
        for field_name in (
            "header_sha256",
            "head_seal_sha256",
            "last_event_sha256",
            "round_seal_inventory_sha256",
            "round_timing_receipt_inventory_sha256",
            "event_inventory_sha256",
            "elapsed_inventory_sha256",
            "truth_contract_sha256",
            "archive_sha256",
        ):
            _sha256(getattr(self, field_name), label=f"archive {field_name}")
        _nonnegative_bounded_integer(
            self.round_count, maximum=PEPTIDE_GA_MAX_ROUNDS, label="archive round count"
        )
        _require(self.round_count > 0, "archive round count must be positive")
        _nonnegative_bounded_integer(
            self.event_count, maximum=PEPTIDE_GA_MAX_EVENTS, label="archive event count"
        )
        _nonnegative_bounded_integer(
            self.proposal_count,
            maximum=PEPTIDE_GA_MAX_PROPOSALS,
            label="archive proposal count",
        )
        _nonnegative_bounded_integer(
            self.query_count, maximum=PEPTIDE_GA_MAX_QUERIES, label="archive query count"
        )
        _nonnegative_bounded_integer(
            self.response_count,
            maximum=PEPTIDE_GA_MAX_RESPONSES,
            label="archive response count",
        )
        _nonnegative_bounded_integer(
            self.scientific_elapsed_ns,
            maximum=PEPTIDE_GA_MAX_SCIENTIFIC_ELAPSED_NS,
            label="archive scientific elapsed ns",
        )
        _require(
            self.event_count == self.proposal_count + self.query_count + self.response_count,
            "archive event counts differ",
        )
        _require(self.query_count == self.response_count, "archive query/response counts differ")
        _require(
            type(self.authentication_status) is str
            and self.authentication_status
            in {
                "verified_campaign_path_truth_contract_unaccepted",
                "unverified_object_fixture_only",
            },
            "archive authentication status differs",
        )
        _require(
            type(self.round_timing_receipt_sha256s) is tuple
            and len(self.round_timing_receipt_sha256s) == self.round_count,
            "archive round timing-receipt inventory differs",
        )
        _require(
            round_timing_receipt_inventory_sha256(self.round_timing_receipt_sha256s)
            == self.round_timing_receipt_inventory_sha256,
            "archive round timing-receipt seal differs",
        )
        _require(
            type(self.individuals) is tuple and 0 < len(self.individuals) <= self.response_count,
            "archive population size differs",
        )
        _require(
            all(type(item) is ArchiveIndividual for item in self.individuals),
            "archive population type differs",
        )
        previous_rank: tuple[float, str] | None = None
        observed_sequence_keys: set[str] = set()
        for item in self.individuals:
            item.__post_init__()
            rank = (-item.fitness, item.sequence_key)
            _require(
                previous_rank is None or previous_rank <= rank,
                "archive population order differs",
            )
            _require(
                item.sequence_key not in observed_sequence_keys,
                "archive population sequence keys are not unique",
            )
            previous_rank = rank
            observed_sequence_keys.add(item.sequence_key)


@dataclass(frozen=True, slots=True)
class SelectionPolicyRegistryRow:
    """One bounded policy rule authorized only by an authenticated registry file."""

    namespace: str
    phase: str
    round_start: int
    round_end: int
    proposal_policy_version: str
    selection_policy_version: str
    selection_policy_entry_point: str
    selection_policy_implementation_sha256: str
    config_sha256: str
    selection_id_derivation: str

    def __post_init__(self) -> None:
        for name in (
            "namespace",
            "proposal_policy_version",
            "selection_policy_version",
            "selection_policy_entry_point",
            "selection_id_derivation",
        ):
            _identifier(getattr(self, name), label=f"policy registry {name}")
        _require(
            type(self.phase) is str and self.phase in {"screen", "confirmation"},
            "policy registry phase differs",
        )
        _nonnegative_bounded_integer(
            self.round_start,
            maximum=PEPTIDE_GA_MAX_ROUNDS,
            label="policy registry round start",
        )
        _nonnegative_bounded_integer(
            self.round_end,
            maximum=PEPTIDE_GA_MAX_ROUNDS,
            label="policy registry round end",
        )
        _require(self.round_start <= self.round_end, "policy registry round interval differs")
        _sha256(
            self.selection_policy_implementation_sha256,
            label="policy registry implementation digest",
        )
        expected_implementation = {
            PEPTIDE_GA_BOOTSTRAP_SELECTION_POLICY_VERSION: (
                PEPTIDE_GA_BOOTSTRAP_SELECTION_ENTRY_POINT,
                PEPTIDE_GA_BOOTSTRAP_SELECTION_IMPLEMENTATION_SHA256,
            ),
            PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION: (
                PEPTIDE_GA_CONTROLLER_SELECTION_ENTRY_POINT,
                PEPTIDE_GA_CONTROLLER_SELECTION_IMPLEMENTATION_SHA256,
            ),
        }.get(self.selection_policy_version)
        _require(
            expected_implementation
            == (
                self.selection_policy_entry_point,
                self.selection_policy_implementation_sha256,
            ),
            "policy registry executable implementation binding differs",
        )
        _sha256(self.config_sha256, label="policy registry config digest")
        _require(
            self.selection_id_derivation == PEPTIDE_GA_SELECTION_ID_DERIVATION,
            "policy registry selection-ID derivation differs",
        )

    def matches(
        self,
        *,
        namespace: str,
        phase: str,
        proposal_round: int,
        proposal_policy_version: str,
        selection_policy_version: str,
        config_sha256: str,
    ) -> bool:
        return (
            self.namespace == namespace
            and self.phase == phase
            and self.round_start <= proposal_round <= self.round_end
            and self.proposal_policy_version == proposal_policy_version
            and self.selection_policy_version == selection_policy_version
            and self.config_sha256 == config_sha256
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedSelectionPolicyRegistry:
    """Canonical registry bytes authenticated from a trusted path and external digest."""

    rows: tuple[SelectionPolicyRegistryRow, ...]
    registry_sha256: str
    registry_source_bytes: bytes
    implementation_source_sha256: str
    implementation_manifest_sha256: str
    authentication_status: Literal["verified_trusted_path"]
    implementation_authentication_status: Literal["verified_complete_source_bytes"]
    execution_authorized: bool = False
    scientific_evidence_accepted: bool = False

    def __post_init__(self) -> None:
        _require(
            type(self.rows) is tuple and 0 < len(self.rows) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_ROWS,
            "policy registry row inventory differs",
        )
        _require(
            all(type(row) is SelectionPolicyRegistryRow for row in self.rows),
            "policy registry row type differs",
        )
        _sha256(self.registry_sha256, label="policy registry digest")
        _require(
            type(self.registry_source_bytes) is bytes
            and 0 < len(self.registry_source_bytes) <= PEPTIDE_GA_POLICY_REGISTRY_MAX_BYTES,
            "policy registry source size differs",
        )
        _require(
            sha256_bytes(self.registry_source_bytes) == self.registry_sha256,
            "policy registry source digest differs",
        )
        _require(
            self.implementation_source_sha256 == PEPTIDE_GA_SELECTION_POLICY_SOURCE_SHA256,
            "selection implementation source digest differs",
        )
        _require(
            self.implementation_manifest_sha256
            == PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256,
            "selection implementation manifest digest differs",
        )
        _require(
            type(self.authentication_status) is str
            and self.authentication_status == "verified_trusted_path",
            "policy registry authentication status differs",
        )
        _require(
            type(self.implementation_authentication_status) is str
            and self.implementation_authentication_status == "verified_complete_source_bytes",
            "selection implementation authentication status differs",
        )
        _require(
            self.execution_authorized is False and self.scientific_evidence_accepted is False,
            "policy registry cannot grant authority or evidence",
        )
        previous_key: tuple[object, ...] | None = None
        for row in self.rows:
            row.__post_init__()
            key = (
                row.phase,
                row.namespace,
                row.round_start,
                row.round_end,
                row.proposal_policy_version,
                row.selection_policy_version,
                row.selection_policy_entry_point,
                row.config_sha256,
            )
            _require(previous_key is None or previous_key < key, "policy registry rows differ")
            previous_key = key
        for left_index, left in enumerate(self.rows):
            for right in self.rows[left_index + 1 :]:
                same_match_dimensions = (
                    left.namespace == right.namespace
                    and left.phase == right.phase
                    and left.proposal_policy_version == right.proposal_policy_version
                    and left.selection_policy_version == right.selection_policy_version
                    and left.config_sha256 == right.config_sha256
                )
                overlap = (
                    left.round_start <= right.round_end and right.round_start <= left.round_end
                )
                _require(
                    not (same_match_dimensions and overlap),
                    "policy registry rows overlap",
                )

    def matching_row(
        self,
        *,
        namespace: str,
        phase: str,
        proposal_round: int,
        proposal_policy_version: str,
        selection_policy_version: str,
        config_sha256: str,
    ) -> SelectionPolicyRegistryRow:
        matches = tuple(
            row
            for row in self.rows
            if row.matches(
                namespace=namespace,
                phase=phase,
                proposal_round=proposal_round,
                proposal_policy_version=proposal_policy_version,
                selection_policy_version=selection_policy_version,
                config_sha256=config_sha256,
            )
        )
        _require(len(matches) == 1, "campaign proposal must match exactly one policy registry row")
        return matches[0]


def public_exclusion_receipt_document(
    receipt: PublicExclusionReceipt,
    *,
    include_sha256: bool,
) -> dict[str, object]:
    value = {
        "authenticated_pre_wave_head_sha256": receipt.authenticated_pre_wave_head_sha256,
        "authenticated_pre_wave_round_count": receipt.authenticated_pre_wave_round_count,
        "batch_id": receipt.batch_id,
        "campaign_id": receipt.campaign_id,
        "config_sha256": receipt.config_sha256,
        "input_sha256": receipt.input_sha256,
        "seed": receipt.seed,
        "submitted_count": receipt.submitted_count,
        "submitted_set_sha256": receipt.submitted_set_sha256,
        "training_count": receipt.training_count,
        "training_set_sha256": receipt.training_set_sha256,
        "wave_id": receipt.wave_id,
    }
    if include_sha256:
        value["receipt_sha256"] = receipt.receipt_sha256
    return value


@dataclass(frozen=True, slots=True)
class PublicExclusionReceipt:
    """Public exclusions sealed by generation before private reserve composition."""

    campaign_id: str
    wave_id: str
    authenticated_pre_wave_head_sha256: str
    authenticated_pre_wave_round_count: int
    batch_id: str
    seed: int
    config_sha256: str
    input_sha256: str
    training_count: int
    submitted_count: int
    training_set_sha256: str
    submitted_set_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="public-exclusion campaign ID")
        _identifier(self.wave_id, label="public-exclusion wave ID")
        _identifier(
            self.batch_id,
            label="public-exclusion batch ID",
            maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
        )
        _sha256(
            self.authenticated_pre_wave_head_sha256,
            label="public-exclusion pre-wave head",
        )
        _nonnegative_bounded_integer(
            self.authenticated_pre_wave_round_count,
            maximum=PEPTIDE_GA_MAX_ROUNDS,
            label="public-exclusion pre-wave round count",
        )
        _nonnegative_bounded_integer(
            self.seed,
            maximum=PEPTIDE_GA_SIGNED_63_MAX,
            label="public-exclusion seed",
        )
        _nonnegative_bounded_integer(
            self.training_count,
            maximum=PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT,
            label="public-exclusion training count",
        )
        _nonnegative_bounded_integer(
            self.submitted_count,
            maximum=PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT,
            label="public-exclusion submitted count",
        )
        for value, label in (
            (self.config_sha256, "public-exclusion config digest"),
            (self.input_sha256, "public-exclusion input digest"),
            (self.training_set_sha256, "public-exclusion training-set digest"),
            (self.submitted_set_sha256, "public-exclusion submitted-set digest"),
            (self.receipt_sha256, "public-exclusion receipt digest"),
        ):
            _sha256(value, label=label)
        expected = sha256_bytes(
            PEPTIDE_GA_PUBLIC_EXCLUSION_RECEIPT_HASH_DOMAIN
            + canonical_json_bytes(public_exclusion_receipt_document(self, include_sha256=False))
        )
        _require(self.receipt_sha256 == expected, "public-exclusion receipt seal differs")


def make_public_exclusion_receipt(
    *,
    campaign_id: str,
    wave_id: str,
    authenticated_pre_wave_head_sha256: str,
    authenticated_pre_wave_round_count: int,
    batch_id: str,
    seed: int,
    config_sha256: str,
    input_sha256: str,
    training_count: int,
    submitted_count: int,
    training_set_sha256: str,
    submitted_set_sha256: str,
) -> PublicExclusionReceipt:
    values = {
        "authenticated_pre_wave_head_sha256": authenticated_pre_wave_head_sha256,
        "authenticated_pre_wave_round_count": authenticated_pre_wave_round_count,
        "batch_id": batch_id,
        "campaign_id": campaign_id,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "seed": seed,
        "submitted_count": submitted_count,
        "submitted_set_sha256": submitted_set_sha256,
        "training_count": training_count,
        "training_set_sha256": training_set_sha256,
        "wave_id": wave_id,
    }
    return PublicExclusionReceipt(
        campaign_id=campaign_id,
        wave_id=wave_id,
        authenticated_pre_wave_head_sha256=authenticated_pre_wave_head_sha256,
        authenticated_pre_wave_round_count=authenticated_pre_wave_round_count,
        batch_id=batch_id,
        seed=seed,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        training_count=training_count,
        submitted_count=submitted_count,
        training_set_sha256=training_set_sha256,
        submitted_set_sha256=submitted_set_sha256,
        receipt_sha256=sha256_bytes(
            PEPTIDE_GA_PUBLIC_EXCLUSION_RECEIPT_HASH_DOMAIN + canonical_json_bytes(values)
        ),
    )


@dataclass(frozen=True, slots=True)
class FitnessTruthContract:
    """Pinned mapping used to interpret archived outcomes, without granting acceptance."""

    evaluator_version: str
    fidelity: str
    objective_names: tuple[str, ...]
    oracle_contract_sha256: str
    evaluator_sha256: str
    checkpoint_sha256: str
    endpoint_context_sha256: str
    transform_sha256: str
    campaign_configuration_id: str
    selection_policy_registry_sha256: str
    selection_policy_implementation_manifest_sha256: str
    truth_mapping_sha256: str
    accepted_for_execution: bool = False
    accepted_for_scientific_evidence: bool = False

    def __post_init__(self) -> None:
        _identifier(self.evaluator_version, label="truth contract evaluator version")
        _identifier(self.fidelity, label="truth contract fidelity")
        if type(self.objective_names) is not tuple or self.objective_names != (
            "gram_positive_activity",
            "gram_negative_activity",
        ):
            raise PeptideGAError("truth contract objectives differ")
        for name in (
            "oracle_contract_sha256",
            "evaluator_sha256",
            "checkpoint_sha256",
            "endpoint_context_sha256",
            "transform_sha256",
            "selection_policy_registry_sha256",
            "selection_policy_implementation_manifest_sha256",
            "truth_mapping_sha256",
        ):
            digest = getattr(self, name)
            _sha256(digest, label=f"truth contract {name}")
        _identifier(
            self.campaign_configuration_id,
            label="truth contract campaign configuration ID",
        )
        if self.campaign_configuration_id != PEPTIDE_GA_CAMPAIGN_CONFIGURATION_ID:
            raise PeptideGAError("truth contract campaign configuration differs")
        if (
            self.selection_policy_implementation_manifest_sha256
            != PEPTIDE_GA_SELECTION_POLICY_IMPLEMENTATION_MANIFEST_SHA256
        ):
            raise PeptideGAError("truth contract selection implementation manifest differs")
        if (
            self.accepted_for_execution is not False
            or self.accepted_for_scientific_evidence is not False
        ):
            raise PeptideGAError("development-v1 truth contract must remain unaccepted")

    @property
    def sha256(self) -> str:
        return sha256_bytes(
            b"amp/fixed-default-peptide-ga/truth-contract/v1\0"
            + canonical_json_bytes(
                {
                    "accepted_for_execution": self.accepted_for_execution,
                    "accepted_for_scientific_evidence": self.accepted_for_scientific_evidence,
                    "evaluator_version": self.evaluator_version,
                    "fidelity": self.fidelity,
                    "oracle_contract_sha256": self.oracle_contract_sha256,
                    "evaluator_sha256": self.evaluator_sha256,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "endpoint_context_sha256": self.endpoint_context_sha256,
                    "transform_sha256": self.transform_sha256,
                    "campaign_configuration_id": self.campaign_configuration_id,
                    "selection_policy_registry_sha256": (self.selection_policy_registry_sha256),
                    "selection_policy_implementation_manifest_sha256": (
                        self.selection_policy_implementation_manifest_sha256
                    ),
                    "objective_names": list(self.objective_names),
                    "truth_mapping_sha256": self.truth_mapping_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class CollisionExclusions:
    """Sequence-key-only exclusions; private reserve sequences never enter the adapter."""

    training_sequence_keys: tuple[str, ...]
    submitted_sequence_keys: tuple[str, ...]
    training_count: int
    submitted_count: int
    training_canonical_bytes: int
    submitted_canonical_bytes: int
    training_set_sha256: str
    submitted_set_sha256: str
    training_sequence_key_set: frozenset[str] = field(init=False, repr=False, compare=False)
    submitted_sequence_key_set: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fields = (
            (
                "training",
                self.training_sequence_keys,
                self.training_count,
                self.training_canonical_bytes,
                self.training_set_sha256,
                PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT,
                PEPTIDE_GA_TRAINING_EXCLUSION_MAX_BYTES,
            ),
            (
                "submitted",
                self.submitted_sequence_keys,
                self.submitted_count,
                self.submitted_canonical_bytes,
                self.submitted_set_sha256,
                PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT,
                PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_BYTES,
            ),
        )
        for label, keys, count, byte_count, digest, max_count, max_bytes in fields:
            if type(count) is not int or not 0 <= count <= max_count:
                raise PeptideGAError(f"{label} exclusion count differs")
            if type(byte_count) is not int or not 0 <= byte_count <= max_bytes:
                raise PeptideGAError(f"{label} exclusion byte count differs")
            _sha256(digest, label=f"{label} exclusion-set digest")
            if type(keys) is not tuple or len(keys) != count:
                raise PeptideGAError(f"{label} exclusion inventory differs")
            previous_key: str | None = None
            for key in keys:
                _sha256(key, label=f"{label} sequence key")
                if previous_key is not None and previous_key >= key:
                    raise PeptideGAError(f"{label} sequence keys must be sorted and unique")
                previous_key = key
            encoded_bytes = len(canonical_json_bytes(list(keys)))
            if byte_count != encoded_bytes:
                raise PeptideGAError(f"{label} exclusion byte count differs")
            if exclusion_set_sha256(label, keys) != digest:
                raise PeptideGAError(f"{label} exclusion-set digest differs")
        object.__setattr__(
            self, "training_sequence_key_set", frozenset(self.training_sequence_keys)
        )
        object.__setattr__(
            self, "submitted_sequence_key_set", frozenset(self.submitted_sequence_keys)
        )

    @classmethod
    def from_keys(
        cls,
        *,
        training_sequence_keys: tuple[str, ...],
        submitted_sequence_keys: tuple[str, ...],
    ) -> CollisionExclusions:
        for label, keys, maximum in (
            ("training", training_sequence_keys, PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT),
            ("submitted", submitted_sequence_keys, PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT),
        ):
            _require(
                type(keys) is tuple and len(keys) <= maximum,
                f"{label} exclusion count ceiling exceeded before collection",
            )
            for key in keys:
                _sha256(key, label=f"{label} sequence key")
        training = tuple(sorted(set(training_sequence_keys)))
        submitted = tuple(sorted(set(submitted_sequence_keys)))
        return cls(
            training_sequence_keys=training,
            submitted_sequence_keys=submitted,
            training_count=len(training),
            submitted_count=len(submitted),
            training_canonical_bytes=len(canonical_json_bytes(list(training))),
            submitted_canonical_bytes=len(canonical_json_bytes(list(submitted))),
            training_set_sha256=exclusion_set_sha256("training", training),
            submitted_set_sha256=exclusion_set_sha256("submitted", submitted),
        )


@dataclass(frozen=True, slots=True)
class AdapterAttemptProvenance:
    """Attempt facts; the campaign controller owns transposition admission."""

    event_index: int
    proposal_id: str
    edge_id: str
    sequence_key: str
    first_attempt_proposal_id: str
    duplicate_within_prefix: bool
    hard_valid: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        _nonnegative_bounded_integer(
            self.event_index,
            maximum=PEPTIDE_GA_ATTEMPT_CAP - 1,
            label="attempt provenance event index",
        )
        _identifier(self.proposal_id, label="attempt provenance proposal ID")
        _identifier(self.edge_id, label="attempt provenance edge ID")
        _sha256(self.sequence_key, label="attempt provenance sequence key")
        _identifier(
            self.first_attempt_proposal_id,
            label="attempt provenance first proposal ID",
        )
        _require(
            type(self.duplicate_within_prefix) is bool and type(self.hard_valid) is bool,
            "attempt provenance boolean fields differ",
        )
        if self.rejection_reason is not None:
            _identifier(self.rejection_reason, label="attempt provenance rejection reason")
        _require(
            (self.rejection_reason is None) is self.hard_valid,
            "attempt provenance validity differs",
        )


@dataclass(frozen=True, slots=True)
class PeptideGAAttempt:
    """One retained attempted child, whether accepted, collided, or invalid."""

    attempt_index: int
    accepted_position: int | None
    proposal: ProposalRecord
    edge: EdgeRecord
    provenance: AdapterAttemptProvenance

    def __post_init__(self) -> None:
        _nonnegative_bounded_integer(
            self.attempt_index,
            maximum=PEPTIDE_GA_ATTEMPT_CAP - 1,
            label="GA attempt index",
        )
        if self.accepted_position is not None:
            _nonnegative_bounded_integer(
                self.accepted_position,
                maximum=PEPTIDE_GA_CANDIDATE_PREFIX_SIZE - 1,
                label="GA accepted position",
            )
        _require(type(self.proposal) is ProposalRecord, "GA attempt proposal type differs")
        _require(type(self.edge) is EdgeRecord, "GA attempt edge type differs")
        _require(
            type(self.provenance) is AdapterAttemptProvenance,
            "GA attempt provenance type differs",
        )
        _require(
            self.attempt_index == self.provenance.event_index,
            "GA attempt/provenance index differs",
        )


def _preflight_bounded_nested_tuple(
    value: object,
    *,
    label: str,
    maximum_items: int,
    maximum_nodes: int = 128,
    maximum_depth: int = 4,
) -> None:
    _require(type(value) is tuple, f"{label} type differs")
    assert isinstance(value, tuple)
    _require(len(value) <= maximum_items, f"{label} item ceiling exceeded")
    pending: list[tuple[object, int]] = [(item, 1) for item in reversed(value)]
    node_count = 0
    while pending:
        item, depth = pending.pop()
        node_count += 1
        _require(node_count <= maximum_nodes, f"{label} node ceiling exceeded")
        if type(item) is tuple:
            assert isinstance(item, tuple)
            _require(depth < maximum_depth, f"{label} depth ceiling exceeded")
            _require(len(item) <= maximum_items, f"{label} nested item ceiling exceeded")
            pending.extend((child, depth + 1) for child in reversed(item))
        elif type(item) is str:
            _require(0 < len(item) <= PEPTIDE_GA_IDENTIFIER_MAX_LENGTH, f"{label} string differs")
        elif type(item) is int:
            assert isinstance(item, int)
            _require(abs(item) <= PEPTIDE_GA_SIGNED_63_MAX, f"{label} integer differs")
        elif type(item) is float:
            _require(math.isfinite(item), f"{label} float differs")
        else:
            _require(type(item) is bool or item is None, f"{label} scalar differs")


def preflight_peptide_ga_batch_structure(batch: object) -> PeptideGABatch:
    """Cap every nested batch inventory before hashing, set creation, or replay."""

    _require(type(batch) is PeptideGABatch, "peptide-GA batch type differs")
    assert isinstance(batch, PeptideGABatch)
    _require(
        type(batch.artifact) is str
        and batch.artifact == PEPTIDE_GA_ARTIFACT
        and type(batch.schema_version) is int
        and batch.schema_version == 1,
        "peptide-GA batch artifact/version differs",
    )
    _require(
        type(batch.status) is str
        and batch.status in {"complete", "in_progress", "attempt_cap_exhausted"},
        "peptide-GA batch status differs",
    )
    _identifier(batch.campaign_id, label="peptide-GA campaign ID")
    _identifier(batch.wave_id, label="peptide-GA wave ID")
    _identifier(
        batch.batch_id,
        label="peptide-GA batch ID",
        maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
    )
    _nonnegative_bounded_integer(
        batch.seed,
        maximum=PEPTIDE_GA_SIGNED_63_MAX,
        label="peptide-GA seed",
    )
    _nonnegative_bounded_integer(
        batch.next_attempt_index,
        maximum=PEPTIDE_GA_ATTEMPT_CAP,
        label="peptide-GA next attempt index",
    )
    for field_name in (
        "execution_authorized",
        "scientific_evidence_accepted",
        "automatic_production_eligible",
        "biological_superiority_claim_allowed",
    ):
        _require(
            getattr(batch, field_name) is False,
            f"peptide-GA {field_name} cannot grant authority",
        )
    for field_name in (
        "oracle_query_identities_constructed",
        "controller_private_reserve_seats_emitted",
    ):
        _require(
            type(getattr(batch, field_name)) is int and getattr(batch, field_name) == 0,
            f"peptide-GA {field_name} must remain zero",
        )
    for field_name in ("config_sha256", "archive_sha256", "input_sha256", "output_sha256"):
        _sha256(getattr(batch, field_name), label=f"peptide-GA {field_name}")
    _require(
        type(batch.attempts) is tuple and len(batch.attempts) <= PEPTIDE_GA_ATTEMPT_CAP,
        "peptide-GA attempt inventory differs",
    )
    _require(
        type(batch.accepted_proposal_ids) is tuple
        and len(batch.accepted_proposal_ids) <= PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
        "peptide-GA accepted proposal inventory differs",
    )
    _require(
        type(batch.accepted_sequences) is tuple
        and len(batch.accepted_sequences) <= PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
        "peptide-GA accepted sequence inventory differs",
    )
    _require(
        type(batch.public_exclusion_receipt) is PublicExclusionReceipt,
        "peptide-GA public-exclusion receipt type differs",
    )
    receipt = batch.public_exclusion_receipt
    _identifier(receipt.campaign_id, label="public-exclusion campaign ID")
    _identifier(receipt.wave_id, label="public-exclusion wave ID")
    _identifier(
        receipt.batch_id,
        label="public-exclusion batch ID",
        maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
    )
    for value, maximum, label in (
        (receipt.authenticated_pre_wave_round_count, PEPTIDE_GA_MAX_ROUNDS, "pre-wave round"),
        (receipt.seed, PEPTIDE_GA_SIGNED_63_MAX, "seed"),
        (receipt.training_count, PEPTIDE_GA_TRAINING_EXCLUSION_MAX_COUNT, "training count"),
        (receipt.submitted_count, PEPTIDE_GA_SUBMITTED_EXCLUSION_MAX_COUNT, "submitted count"),
    ):
        _nonnegative_bounded_integer(value, maximum=maximum, label=f"public-exclusion {label}")
    for value, label in (
        (receipt.authenticated_pre_wave_head_sha256, "pre-wave head"),
        (receipt.config_sha256, "config digest"),
        (receipt.input_sha256, "input digest"),
        (receipt.training_set_sha256, "training-set digest"),
        (receipt.submitted_set_sha256, "submitted-set digest"),
        (receipt.receipt_sha256, "receipt digest"),
    ):
        _sha256(value, label=f"public-exclusion {label}")
    for proposal_id in batch.accepted_proposal_ids:
        _identifier(proposal_id, label="peptide-GA accepted proposal ID")
    for sequence in batch.accepted_sequences:
        _require(
            type(sequence) is str
            and PEPTIDE_GA_MIN_LENGTH <= len(sequence) <= PEPTIDE_GA_MAX_LENGTH
            and sequence == sequence.upper()
            and set(sequence) <= set(PEPTIDE_GA_ALPHABET),
            "peptide-GA accepted sequence differs",
        )
    for attempt in batch.attempts:
        _require(type(attempt) is PeptideGAAttempt, "peptide-GA attempt type differs")
        _require(type(attempt.proposal) is ProposalRecord, "GA attempt proposal type differs")
        _require(type(attempt.edge) is EdgeRecord, "GA attempt edge type differs")
        _require(
            type(attempt.provenance) is AdapterAttemptProvenance,
            "GA attempt provenance type differs",
        )
        proposal = attempt.proposal
        edge = attempt.edge
        provenance = attempt.provenance
        _nonnegative_bounded_integer(
            attempt.attempt_index,
            maximum=PEPTIDE_GA_ATTEMPT_CAP - 1,
            label="GA attempt index",
        )
        if attempt.accepted_position is not None:
            _nonnegative_bounded_integer(
                attempt.accepted_position,
                maximum=PEPTIDE_GA_CANDIDATE_PREFIX_SIZE - 1,
                label="GA accepted position",
            )
        for value, label in (
            (proposal.proposal_id, "GA proposal ID"),
            (proposal.rollout_id, "GA proposal rollout ID"),
            (proposal.policy_version, "GA proposal policy"),
            (edge.edge_id, "GA edge ID"),
            (edge.proposal_id, "GA edge proposal ID"),
            (edge.rollout_id, "GA edge rollout ID"),
            (edge.operator, "GA edge operator"),
            (provenance.proposal_id, "GA provenance proposal ID"),
            (provenance.edge_id, "GA provenance edge ID"),
            (provenance.first_attempt_proposal_id, "GA provenance first proposal ID"),
        ):
            _identifier(value, label=label)
        _require(
            type(proposal.sequence) is str
            and 0 < len(proposal.sequence) <= 2 * PEPTIDE_GA_MAX_LENGTH
            and proposal.sequence == proposal.sequence.upper()
            and set(proposal.sequence) <= set(PEPTIDE_GA_ALPHABET),
            "GA proposal sequence differs",
        )
        _require(type(proposal.hard_valid) is bool, "GA proposal validity type differs")
        if proposal.rejection_reason is not None:
            _identifier(proposal.rejection_reason, label="GA proposal rejection reason")
        _nonnegative_bounded_integer(
            proposal.proposal_round,
            maximum=PEPTIDE_GA_MAX_ROUNDS,
            label="GA proposal round",
        )
        if proposal.niche_id is not None:
            _identifier(proposal.niche_id, label="GA proposal namespace")
        _preflight_bounded_nested_tuple(
            proposal.cheap_predictions,
            label="GA cheap predictions",
            maximum_items=16,
            maximum_nodes=64,
        )
        selection = proposal.selection
        _require(type(selection) is SelectionDecision, "GA attempt selection type differs")
        _require(type(selection.selected) is bool, "GA selection flag differs")
        _identifier(selection.selection_set_id, label="GA selection-set ID")
        _identifier(selection.policy_version, label="GA selection policy")
        _nonnegative_bounded_integer(
            selection.seed,
            maximum=PEPTIDE_GA_SIGNED_63_MAX,
            label="GA selection seed",
        )
        _require(
            type(selection.eligible_proposal_ids) is tuple
            and len(selection.eligible_proposal_ids) <= PEPTIDE_GA_SELECTION_ELIGIBLE_MAX_COUNT,
            "GA attempt eligible inventory ceiling exceeded",
        )
        for proposal_id in selection.eligible_proposal_ids:
            _identifier(proposal_id, label="GA eligible proposal ID")
        _require(type(selection.propensity) is ProbabilityTrace, "GA selection trace type differs")
        _require(
            type(selection.propensity.factors) is tuple
            and 0 < len(selection.propensity.factors) <= 16,
            "GA selection trace factor ceiling exceeded",
        )
        _require(
            all(type(factor) is ProbabilityFactor for factor in selection.propensity.factors),
            "GA selection factor type differs",
        )
        for factor in selection.propensity.factors:
            _identifier(factor.name, label="GA selection factor name")
            _require(
                type(factor.probability) is float
                and math.isfinite(factor.probability)
                and 0.0 <= factor.probability <= 1.0,
                "GA selection factor probability differs",
            )
        _require(
            type(proposal.cheap_predictions) is tuple and len(proposal.cheap_predictions) <= 16,
            "GA cheap-prediction ceiling exceeded",
        )
        _require(
            type(edge.parent_sequence_keys) is tuple and len(edge.parent_sequence_keys) <= 2,
            "GA edge parent ceiling exceeded",
        )
        for key in edge.parent_sequence_keys:
            _sha256(key, label="GA edge parent sequence key")
        _preflight_bounded_nested_tuple(
            edge.edit_description,
            label="GA edge edit description",
            maximum_items=8,
        )
        _require(type(edge.proposal_trace) is ProbabilityTrace, "GA proposal trace type differs")
        _require(
            type(edge.proposal_trace.factors) is tuple
            and 0 < len(edge.proposal_trace.factors) <= 16,
            "GA proposal trace factor ceiling exceeded",
        )
        _require(
            all(type(factor) is ProbabilityFactor for factor in edge.proposal_trace.factors),
            "GA proposal factor type differs",
        )
        for factor in edge.proposal_trace.factors:
            _identifier(factor.name, label="GA proposal factor name")
            _require(
                type(factor.probability) is float
                and math.isfinite(factor.probability)
                and 0.0 < factor.probability <= 1.0,
                "GA proposal factor probability differs",
            )
        _require(
            type(edge.behavior_log_probabilities) is tuple
            and len(edge.behavior_log_probabilities) == 0,
            "GA behavior-log inventory differs",
        )
        for value, label in (
            (edge.random_stream, "GA edge random stream"),
            (edge.sample_index, "GA edge sample index"),
        ):
            if value is not None:
                _nonnegative_bounded_integer(
                    value,
                    maximum=PEPTIDE_GA_SIGNED_63_MAX,
                    label=label,
                )
        _preflight_bounded_nested_tuple(
            edge.sampling_parameters,
            label="GA edge sampling parameters",
            maximum_items=32,
            maximum_nodes=256,
        )
        _nonnegative_bounded_integer(
            provenance.event_index,
            maximum=PEPTIDE_GA_ATTEMPT_CAP - 1,
            label="GA provenance event index",
        )
        _sha256(provenance.sequence_key, label="GA provenance sequence key")
        _require(
            type(provenance.duplicate_within_prefix) is bool
            and type(provenance.hard_valid) is bool,
            "GA provenance booleans differ",
        )
        if provenance.rejection_reason is not None:
            _identifier(provenance.rejection_reason, label="GA provenance rejection reason")
    receipt.__post_init__()
    _require(
        len(set(batch.accepted_proposal_ids)) == len(batch.accepted_proposal_ids),
        "peptide-GA accepted proposal IDs are not unique",
    )
    _require(
        len({sequence_key(sequence) for sequence in batch.accepted_sequences})
        == len(batch.accepted_sequences),
        "peptide-GA accepted sequences are not unique",
    )
    observed_proposal_ids: set[str] = set()
    observed_edge_ids: set[str] = set()
    accepted_by_position: list[tuple[str, str]] = []
    for expected_index, attempt in enumerate(batch.attempts):
        attempt.proposal.selection.propensity.__post_init__()
        attempt.proposal.selection.__post_init__()
        attempt.proposal.__post_init__()
        attempt.edge.proposal_trace.__post_init__()
        attempt.edge.__post_init__()
        attempt.provenance.__post_init__()
        attempt.__post_init__()
        _require(attempt.attempt_index == expected_index, "peptide-GA attempt order differs")
        _require(
            attempt.proposal.proposal_id not in observed_proposal_ids,
            "peptide-GA attempt proposal IDs are not unique",
        )
        _require(
            attempt.edge.edge_id not in observed_edge_ids,
            "peptide-GA attempt edge IDs are not unique",
        )
        observed_proposal_ids.add(attempt.proposal.proposal_id)
        observed_edge_ids.add(attempt.edge.edge_id)
        _require(
            attempt.edge.proposal_id
            == attempt.proposal.proposal_id
            == attempt.provenance.proposal_id
            and attempt.edge.edge_id == attempt.provenance.edge_id
            and attempt.proposal.sequence_key == attempt.provenance.sequence_key,
            "peptide-GA attempt identity links differ",
        )
        if attempt.accepted_position is not None:
            _require(
                attempt.accepted_position == len(accepted_by_position),
                "peptide-GA accepted positions differ",
            )
            accepted_by_position.append((attempt.proposal.proposal_id, attempt.proposal.sequence))
    _require(
        tuple(proposal_id for proposal_id, _ in accepted_by_position)
        == batch.accepted_proposal_ids,
        "peptide-GA accepted proposal linkage differs",
    )
    _require(
        tuple(sequence for _, sequence in accepted_by_position) == batch.accepted_sequences,
        "peptide-GA accepted sequence linkage differs",
    )
    return batch


@dataclass(frozen=True, slots=True)
class PeptideGABatch:
    """Sealed proposal result; the controller still owns query identity and submission."""

    artifact: str
    schema_version: int
    status: Literal["complete", "in_progress", "attempt_cap_exhausted"]
    campaign_id: str
    wave_id: str
    batch_id: str
    seed: int
    config_sha256: str
    archive_sha256: str
    input_sha256: str
    public_exclusion_receipt: PublicExclusionReceipt
    attempts: tuple[PeptideGAAttempt, ...]
    accepted_proposal_ids: tuple[str, ...]
    accepted_sequences: tuple[str, ...]
    next_attempt_index: int
    output_sha256: str
    execution_authorized: bool = False
    scientific_evidence_accepted: bool = False
    automatic_production_eligible: bool = False
    biological_superiority_claim_allowed: bool = False
    oracle_query_identities_constructed: int = 0
    controller_private_reserve_seats_emitted: int = 0

    def __post_init__(self) -> None:
        preflight_peptide_ga_batch_structure(self)
        if (
            type(self.artifact) is not str
            or self.artifact != PEPTIDE_GA_ARTIFACT
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise PeptideGAError("peptide-GA batch artifact/version differs")
        if type(self.status) is not str or self.status not in {
            "complete",
            "in_progress",
            "attempt_cap_exhausted",
        }:
            raise PeptideGAError("peptide-GA batch status differs")
        _identifier(self.campaign_id, label="peptide-GA campaign ID")
        _identifier(self.wave_id, label="peptide-GA wave ID")
        _identifier(
            self.batch_id,
            label="peptide-GA batch ID",
            maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
        )
        _nonnegative_bounded_integer(
            self.seed, maximum=PEPTIDE_GA_SIGNED_63_MAX, label="peptide-GA seed"
        )
        for field_name in ("config_sha256", "archive_sha256", "input_sha256", "output_sha256"):
            _sha256(getattr(self, field_name), label=f"peptide-GA {field_name}")
        self.public_exclusion_receipt.__post_init__()
        _require(
            (
                self.public_exclusion_receipt.campaign_id,
                self.public_exclusion_receipt.wave_id,
                self.public_exclusion_receipt.batch_id,
                self.public_exclusion_receipt.seed,
                self.public_exclusion_receipt.config_sha256,
                self.public_exclusion_receipt.input_sha256,
            )
            == (
                self.campaign_id,
                self.wave_id,
                self.batch_id,
                self.seed,
                self.config_sha256,
                self.input_sha256,
            ),
            "peptide-GA public-exclusion receipt identity differs",
        )
        _require(
            type(self.attempts) is tuple and len(self.attempts) <= PEPTIDE_GA_ATTEMPT_CAP,
            "peptide-GA attempt inventory differs",
        )
        _require(
            all(type(attempt) is PeptideGAAttempt for attempt in self.attempts),
            "peptide-GA attempt type differs",
        )
        _require(
            type(self.accepted_proposal_ids) is tuple
            and type(self.accepted_sequences) is tuple
            and len(self.accepted_proposal_ids)
            == len(self.accepted_sequences)
            <= PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
            "peptide-GA accepted inventory differs",
        )
        observed_accepted_ids: set[str] = set()
        for proposal_id in self.accepted_proposal_ids:
            _identifier(proposal_id, label="peptide-GA accepted proposal ID")
            _require(
                proposal_id not in observed_accepted_ids,
                "peptide-GA accepted proposal IDs are not unique",
            )
            observed_accepted_ids.add(proposal_id)
        observed_accepted_keys: set[str] = set()
        for sequence in self.accepted_sequences:
            _require(
                type(sequence) is str
                and sequence == sequence.upper()
                and set(sequence) <= set(PEPTIDE_GA_ALPHABET)
                and PEPTIDE_GA_MIN_LENGTH <= len(sequence) <= PEPTIDE_GA_MAX_LENGTH,
                "peptide-GA accepted sequence differs",
            )
            key = sequence_key(sequence)
            _require(
                key not in observed_accepted_keys,
                "peptide-GA accepted sequences are not unique",
            )
            observed_accepted_keys.add(key)
        _nonnegative_bounded_integer(
            self.next_attempt_index,
            maximum=PEPTIDE_GA_ATTEMPT_CAP,
            label="peptide-GA next attempt index",
        )
        _require(
            self.next_attempt_index == len(self.attempts),
            "peptide-GA attempt cursor differs",
        )
        observed_proposal_ids: set[str] = set()
        observed_edge_ids: set[str] = set()
        accepted_by_position: list[tuple[str, str]] = []
        for expected_index, attempt in enumerate(self.attempts):
            attempt.proposal.__post_init__()
            attempt.edge.__post_init__()
            attempt.provenance.__post_init__()
            attempt.__post_init__()
            _require(attempt.attempt_index == expected_index, "peptide-GA attempt order differs")
            _require(
                attempt.proposal.proposal_id not in observed_proposal_ids,
                "peptide-GA attempt proposal IDs are not unique",
            )
            _require(
                attempt.edge.edge_id not in observed_edge_ids,
                "peptide-GA attempt edge IDs are not unique",
            )
            observed_proposal_ids.add(attempt.proposal.proposal_id)
            observed_edge_ids.add(attempt.edge.edge_id)
            _require(
                attempt.edge.proposal_id
                == attempt.proposal.proposal_id
                == attempt.provenance.proposal_id
                and attempt.edge.edge_id == attempt.provenance.edge_id
                and attempt.proposal.sequence_key == attempt.provenance.sequence_key,
                "peptide-GA attempt identity links differ",
            )
            if attempt.accepted_position is None:
                _require(not attempt.proposal.hard_valid, "hard-valid GA attempt was not retained")
            else:
                _require(attempt.proposal.hard_valid, "invalid GA attempt was retained")
                _require(
                    attempt.accepted_position == len(accepted_by_position),
                    "peptide-GA accepted positions differ",
                )
                accepted_by_position.append(
                    (attempt.proposal.proposal_id, attempt.proposal.sequence)
                )
        _require(
            tuple(proposal_id for proposal_id, _ in accepted_by_position)
            == self.accepted_proposal_ids
            and tuple(sequence for _, sequence in accepted_by_position) == self.accepted_sequences,
            "peptide-GA accepted inventory is not attempt-linked",
        )
        if any(
            value is not False
            for value in (
                self.execution_authorized,
                self.scientific_evidence_accepted,
                self.automatic_production_eligible,
                self.biological_superiority_claim_allowed,
            )
        ):
            raise PeptideGAError("peptide-GA result cannot grant authority or scientific evidence")
        if type(self.oracle_query_identities_constructed) is not int or (
            self.oracle_query_identities_constructed != 0
        ):
            raise PeptideGAError("peptide-GA adapter cannot construct oracle query identities")
        if type(self.controller_private_reserve_seats_emitted) is not int or (
            self.controller_private_reserve_seats_emitted != 0
        ):
            raise PeptideGAError("controller-private reserve seats cannot be emitted by adapter")

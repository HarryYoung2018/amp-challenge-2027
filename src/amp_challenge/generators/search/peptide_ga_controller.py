"""Authenticated controller-private composition for a reserve-blind GA prefix."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from amp_challenge.generators.search import peptide_ga_selection_policy_impl_v1
from amp_challenge.generators.search.campaign_ledger import OracleQueryIdentity
from amp_challenge.generators.search.peptide_ga_policy_registry import (
    load_selection_policy_registry_from_path,
)
from amp_challenge.generators.search.peptide_ga_records import (
    PEPTIDE_GA_ALPHABET,
    PEPTIDE_GA_ATTEMPT_CAP,
    PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
    PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
    PEPTIDE_GA_CONTROLLER_SELECTION_IMPLEMENTATION_SHA256,
    PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
    PEPTIDE_GA_MAX_LENGTH,
    PEPTIDE_GA_MAX_ROUNDS,
    PEPTIDE_GA_METHOD_SEATS,
    PEPTIDE_GA_MIN_LENGTH,
    PEPTIDE_GA_NAMESPACE,
    PEPTIDE_GA_POLICY_VERSION,
    PEPTIDE_GA_PRIVATE_RESERVE_SEATS,
    PEPTIDE_GA_RNG_VERSION,
    PEPTIDE_GA_SELECTION_ID_DERIVATION,
    PEPTIDE_GA_SIGNED_63_MAX,
    AuthenticatedSelectionPolicyRegistry,
    AuthenticatedWaveArchive,
    CollisionExclusions,
    FitnessTruthContract,
    PeptideGAAttempt,
    PeptideGABatch,
    PeptideGAConfig,
    PeptideGAError,
    canonical_json_bytes,
    derive_wave_selection_set_id,
    ordered_eligible_proposal_ids_sha256,
    preflight_peptide_ga_batch_structure,
    sha256_bytes,
)
from amp_challenge.generators.search.peptide_ga_verifier import (
    verify_peptide_ga_batch_from_path,
    verify_unverified_peptide_ga_fixture_batch,
)
from amp_challenge.generators.search.records import (
    EdgeRecord,
    ProbabilityFactor,
    ProbabilityTrace,
    ProposalRecord,
    SelectionDecision,
)

_PRIVATE_RESERVE_MAX_COUNT = 56
_CHARGED_MAX_COUNT = 512
_PRIVATE_RESERVE_BINDING_MAX_BYTES = 65_536
_CHARGED_MAX_BYTES = 34_305
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_COMPOSITION_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/controller-composition/v1\0"
_PROPOSAL_ID_HASH_DOMAIN = b"amp/fixed-default-peptide-ga/proposal-id/v1\0"
_COMPOSED_EDGE_PARAMETER_NAMES = {
    "algorithm",
    "archive_sha256",
    "config_sha256",
    "input_sha256",
    "rng",
    "selection_batch_output_sha256",
    "selection_campaign_id",
    "selection_config_sha256",
    "selection_id_derivation",
    "selection_namespace",
    "selection_ordered_eligible_sha256",
    "selection_policy_implementation_sha256",
    "selection_pre_wave_head_sha256",
    "selection_pre_wave_round_count",
    "selection_wave_id",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PeptideGAError(message)


def _key_inventory_bytes(keys: tuple[str, ...]) -> int:
    return len(canonical_json_bytes(list(keys)))


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{label} is invalid",
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


def _reserve_binding_document(value: ControllerPrivateReserveBinding) -> dict[str, object]:
    return {
        "batch_id": value.batch_id,
        "batch_position": value.batch_position,
        "campaign_id": value.campaign_id,
        "canonical_sequence": value.canonical_sequence,
        "canonical_sequence_id": value.canonical_sequence_id,
        "phase": value.phase,
        "query_identity": value.query_identity.document(),
        "wave_id": value.wave_id,
    }


@dataclass(frozen=True, slots=True)
class ControllerPrivateReserveBinding:
    """One private reserve query bound to its canonical sequence and exact batch seat."""

    campaign_id: str
    phase: str
    wave_id: str
    batch_id: str
    batch_position: int
    canonical_sequence: str
    canonical_sequence_id: str
    query_identity: OracleQueryIdentity

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="private reserve campaign ID")
        _require(
            type(self.phase) is str and self.phase in {"screen", "confirmation"},
            "private reserve phase differs",
        )
        _identifier(self.wave_id, label="private reserve wave ID")
        _identifier(
            self.batch_id,
            label="private reserve batch ID",
            maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
        )
        _require(
            type(self.batch_position) is int
            and PEPTIDE_GA_METHOD_SEATS
            <= self.batch_position
            < PEPTIDE_GA_METHOD_SEATS + PEPTIDE_GA_PRIVATE_RESERVE_SEATS,
            "private reserve batch position differs",
        )
        _require(
            type(self.canonical_sequence) is str
            and PEPTIDE_GA_MIN_LENGTH <= len(self.canonical_sequence) <= PEPTIDE_GA_MAX_LENGTH
            and self.canonical_sequence == self.canonical_sequence.upper()
            and set(self.canonical_sequence) <= set(PEPTIDE_GA_ALPHABET),
            "private reserve canonical sequence differs",
        )
        _sha256(self.canonical_sequence_id, label="private reserve canonical sequence ID")
        _require(
            sha256_bytes(self.canonical_sequence.encode("ascii")) == self.canonical_sequence_id,
            "private reserve canonical sequence digest differs",
        )
        _require(
            type(self.query_identity) is OracleQueryIdentity,
            "private reserve query identity type differs",
        )
        self.query_identity.__post_init__()
        _require(
            type(self.query_identity.replicate_id) is int
            and 0 <= self.query_identity.replicate_id <= PEPTIDE_GA_SIGNED_63_MAX,
            "private reserve query replicate differs",
        )
        _require(
            self.query_identity.canonical_sequence_id == self.canonical_sequence_id,
            "private reserve query identity sequence differs",
        )

    @property
    def query_identity_key(self) -> str:
        return self.query_identity.key

    @property
    def schedule_key(self) -> tuple[str, str, str, str, int]:
        return (
            self.campaign_id,
            self.phase,
            self.wave_id,
            self.batch_id,
            self.batch_position,
        )


@dataclass(frozen=True, slots=True)
class ControllerPrivateCollisionInventory:
    """Controller-only collision state; never a generator input or public output."""

    charged_query_identity_keys: tuple[str, ...]
    scheduled_reserve_bindings: tuple[ControllerPrivateReserveBinding, ...]
    charged_count: int
    scheduled_reserve_count: int
    charged_canonical_bytes: int
    scheduled_reserve_binding_canonical_bytes: int

    def __post_init__(self) -> None:
        _require(
            type(self.charged_count) is int and 0 <= self.charged_count <= _CHARGED_MAX_COUNT,
            "charged count differs",
        )
        _require(
            type(self.charged_canonical_bytes) is int
            and 0 <= self.charged_canonical_bytes <= _CHARGED_MAX_BYTES,
            "charged byte count differs",
        )
        _require(
            type(self.charged_query_identity_keys) is tuple
            and len(self.charged_query_identity_keys) == self.charged_count,
            "charged inventory differs",
        )
        previous_key: str | None = None
        for key in self.charged_query_identity_keys:
            _sha256(key, label="charged key")
            _require(
                previous_key is None or previous_key < key,
                "charged keys must be sorted and unique",
            )
            previous_key = key
        _require(
            self.charged_canonical_bytes == _key_inventory_bytes(self.charged_query_identity_keys),
            "charged byte count differs",
        )
        _require(
            type(self.scheduled_reserve_count) is int
            and 0 <= self.scheduled_reserve_count <= _PRIVATE_RESERVE_MAX_COUNT,
            "scheduled reserve count differs",
        )
        _require(
            type(self.scheduled_reserve_bindings) is tuple
            and len(self.scheduled_reserve_bindings) == self.scheduled_reserve_count,
            "scheduled reserve binding inventory differs",
        )
        _require(
            type(self.scheduled_reserve_binding_canonical_bytes) is int
            and 0
            <= self.scheduled_reserve_binding_canonical_bytes
            <= _PRIVATE_RESERVE_BINDING_MAX_BYTES,
            "scheduled reserve binding byte count differs",
        )
        previous_schedule_key: tuple[str, str, str, str, int] | None = None
        reserve_query_keys: set[str] = set()
        reserve_sequence_ids: set[str] = set()
        for binding in self.scheduled_reserve_bindings:
            _require(
                type(binding) is ControllerPrivateReserveBinding,
                "scheduled reserve binding type differs",
            )
            binding.__post_init__()
            _require(
                previous_schedule_key is None or previous_schedule_key < binding.schedule_key,
                "scheduled reserve bindings must be ordered and seat-unique",
            )
            _require(
                binding.query_identity_key not in reserve_query_keys,
                "scheduled reserve query identities are not unique",
            )
            _require(
                binding.canonical_sequence_id not in reserve_sequence_ids,
                "scheduled reserve canonical sequences are not unique",
            )
            previous_schedule_key = binding.schedule_key
            reserve_query_keys.add(binding.query_identity_key)
            reserve_sequence_ids.add(binding.canonical_sequence_id)
        expected_binding_bytes = len(
            canonical_json_bytes(
                [_reserve_binding_document(binding) for binding in self.scheduled_reserve_bindings]
            )
        )
        _require(
            self.scheduled_reserve_binding_canonical_bytes == expected_binding_bytes,
            "scheduled reserve binding byte count differs",
        )
        _require(
            not (frozenset(self.charged_query_identity_keys) & reserve_query_keys),
            "charged and reserve query identities overlap",
        )

    @classmethod
    def from_bindings(
        cls,
        *,
        charged_query_identity_keys: tuple[str, ...],
        scheduled_reserve_bindings: tuple[ControllerPrivateReserveBinding, ...],
    ) -> ControllerPrivateCollisionInventory:
        _require(
            type(charged_query_identity_keys) is tuple
            and len(charged_query_identity_keys) <= _CHARGED_MAX_COUNT,
            "charged count ceiling exceeded before collection",
        )
        _require(
            type(scheduled_reserve_bindings) is tuple
            and len(scheduled_reserve_bindings) <= _PRIVATE_RESERVE_MAX_COUNT,
            "scheduled reserve count ceiling exceeded before collection",
        )
        for key in charged_query_identity_keys:
            _sha256(key, label="charged key")
        for binding in scheduled_reserve_bindings:
            _require(
                type(binding) is ControllerPrivateReserveBinding,
                "scheduled reserve binding type differs",
            )
            binding.__post_init__()
        charged = tuple(sorted(set(charged_query_identity_keys)))
        reserve_bindings = tuple(
            sorted(scheduled_reserve_bindings, key=lambda row: row.schedule_key)
        )
        binding_bytes = len(
            canonical_json_bytes(
                [_reserve_binding_document(binding) for binding in reserve_bindings]
            )
        )
        _require(
            binding_bytes <= _PRIVATE_RESERVE_BINDING_MAX_BYTES,
            "scheduled reserve binding byte ceiling exceeded",
        )
        return cls(
            charged_query_identity_keys=charged,
            scheduled_reserve_bindings=reserve_bindings,
            charged_count=len(charged),
            scheduled_reserve_count=len(reserve_bindings),
            charged_canonical_bytes=_key_inventory_bytes(charged),
            scheduled_reserve_binding_canonical_bytes=binding_bytes,
        )


@dataclass(frozen=True, slots=True)
class ControllerOwnedGACandidate:
    """One selected proposal and controller-created logical query identity."""

    prefix_position: int
    attempt_index: int
    proposal: ProposalRecord
    edge: EdgeRecord
    query_identity: OracleQueryIdentity

    def __post_init__(self) -> None:
        _require(
            type(self.prefix_position) is int
            and 0 <= self.prefix_position < PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
            "GA controller prefix position differs",
        )
        _require(
            type(self.attempt_index) is int and 0 <= self.attempt_index < PEPTIDE_GA_ATTEMPT_CAP,
            "GA controller attempt index differs",
        )
        _require(type(self.proposal) is ProposalRecord, "GA controller proposal type differs")
        _require(type(self.edge) is EdgeRecord, "GA controller edge type differs")
        _require(
            type(self.query_identity) is OracleQueryIdentity,
            "GA controller query identity type differs",
        )


def _composition_document(value: ControllerGAComposition) -> dict[str, object]:
    return {
        "archive_sha256": value.archive_sha256,
        "authenticated_pre_wave_head_sha256": value.authenticated_pre_wave_head_sha256,
        "authenticated_pre_wave_round_count": value.authenticated_pre_wave_round_count,
        "batch_id": value.batch_id,
        "batch_output_sha256": value.batch_output_sha256,
        "campaign_id": value.campaign_id,
        "config_sha256": value.config_sha256,
        "input_sha256": value.input_sha256,
        "ordered_eligible_sha256": value.ordered_eligible_sha256,
        "phase": value.phase,
        "policy_registry_sha256": value.policy_registry_sha256,
        "public_exclusion_receipt_sha256": value.public_exclusion_receipt_sha256,
        "query_identity_keys": [candidate.query_identity.key for candidate in value.selected],
        "replicate_id": value.replicate_id,
        "seed": value.seed,
        "selected_prefix_positions": list(value.selected_prefix_positions),
        "selected_attempt_indices": [candidate.attempt_index for candidate in value.selected],
        "selected_proposal_ids": [candidate.proposal.proposal_id for candidate in value.selected],
        "selected_sequence_ids": [candidate.proposal.sequence_key for candidate in value.selected],
        "selection_policy_implementation_sha256": (value.selection_policy_implementation_sha256),
        "selection_set_id": value.selection_set_id,
        "truth_contract_sha256": value.truth_contract_sha256,
        "wave_id": value.wave_id,
    }


@dataclass(frozen=True, slots=True)
class ControllerGAComposition:
    """Exactly 14 selected seats with no retained private reserve identities."""

    campaign_id: str
    phase: str
    wave_id: str
    batch_id: str
    authenticated_pre_wave_head_sha256: str
    authenticated_pre_wave_round_count: int
    archive_sha256: str
    batch_output_sha256: str
    config_sha256: str
    input_sha256: str
    public_exclusion_receipt_sha256: str
    policy_registry_sha256: str
    selection_policy_implementation_sha256: str
    truth_contract_sha256: str
    seed: int
    replicate_id: int
    selection_set_id: str
    ordered_eligible_sha256: str
    selected: tuple[ControllerOwnedGACandidate, ...]
    selected_prefix_positions: tuple[int, ...]
    composition_sha256: str
    execution_authorized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="GA composition campaign ID")
        _require(
            type(self.phase) is str and self.phase in {"screen", "confirmation"},
            "GA composition phase differs",
        )
        _identifier(self.wave_id, label="GA composition wave ID")
        _identifier(
            self.batch_id,
            label="GA composition batch ID",
            maximum_length=PEPTIDE_GA_BATCH_ID_MAX_LENGTH,
        )
        _require(
            type(self.authenticated_pre_wave_round_count) is int
            and 0 <= self.authenticated_pre_wave_round_count <= PEPTIDE_GA_MAX_ROUNDS,
            "GA composition pre-wave round differs",
        )
        for value, label in ((self.seed, "seed"), (self.replicate_id, "replicate ID")):
            _require(
                type(value) is int and 0 <= value <= PEPTIDE_GA_SIGNED_63_MAX,
                f"GA composition {label} differs",
            )
        for value, label in (
            (self.authenticated_pre_wave_head_sha256, "GA composition pre-wave head"),
            (self.archive_sha256, "GA composition archive digest"),
            (self.batch_output_sha256, "GA composition batch output digest"),
            (self.config_sha256, "GA composition config digest"),
            (self.input_sha256, "GA composition input digest"),
            (self.public_exclusion_receipt_sha256, "GA composition public-exclusion digest"),
            (self.policy_registry_sha256, "GA composition policy registry digest"),
            (
                self.selection_policy_implementation_sha256,
                "GA composition policy implementation digest",
            ),
            (self.truth_contract_sha256, "GA composition truth-contract digest"),
            (self.ordered_eligible_sha256, "GA composition eligible digest"),
            (self.composition_sha256, "GA composition digest"),
        ):
            _sha256(value, label=label)
        _require(
            type(self.selection_set_id) is str and 0 < len(self.selection_set_id) <= 128,
            "GA composition selection-set ID differs",
        )
        _require(
            type(self.selected) is tuple and len(self.selected) == PEPTIDE_GA_METHOD_SEATS,
            "GA composition seat count differs",
        )
        _require(
            all(type(candidate) is ControllerOwnedGACandidate for candidate in self.selected),
            "GA composition selected type differs",
        )
        _require(
            type(self.selected_prefix_positions) is tuple
            and len(self.selected_prefix_positions) == PEPTIDE_GA_METHOD_SEATS,
            "GA composition position inventory differs",
        )
        for position in self.selected_prefix_positions:
            _require(
                type(position) is int and 0 <= position < PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
                "GA composition position inventory differs",
            )
        positions: list[int] = []
        attempt_indices: list[int] = []
        proposal_ids: set[str] = set()
        sequence_keys: set[str] = set()
        query_keys: set[str] = set()
        for candidate in self.selected:
            candidate.__post_init__()
            _require(
                type(candidate.proposal.sequence) is str
                and PEPTIDE_GA_MIN_LENGTH
                <= len(candidate.proposal.sequence)
                <= PEPTIDE_GA_MAX_LENGTH
                and candidate.proposal.sequence == candidate.proposal.sequence.upper()
                and set(candidate.proposal.sequence) <= set(PEPTIDE_GA_ALPHABET),
                "GA composition proposal sequence support differs",
            )
            selection = candidate.proposal.selection
            _require(
                type(selection) is SelectionDecision
                and type(selection.eligible_proposal_ids) is tuple
                and len(selection.eligible_proposal_ids) == PEPTIDE_GA_METHOD_SEATS
                and type(selection.propensity) is ProbabilityTrace
                and type(selection.propensity.factors) is tuple
                and len(selection.propensity.factors) == 1,
                "GA composition selection structure differs",
            )
            _require(type(selection.selected) is bool, "GA composition selection flag differs")
            _identifier(selection.selection_set_id, label="GA composition selection-set ID")
            _identifier(selection.policy_version, label="GA composition selection policy")
            _require(
                type(selection.seed) is int and 0 <= selection.seed <= PEPTIDE_GA_SIGNED_63_MAX,
                "GA composition selection seed differs",
            )
            for proposal_id in selection.eligible_proposal_ids:
                _identifier(proposal_id, label="GA composition eligible proposal ID")
            _require(
                type(candidate.proposal.cheap_predictions) is tuple
                and len(candidate.proposal.cheap_predictions) == 0,
                "GA composition cheap-prediction inventory differs",
            )
            _require(
                type(candidate.edge.parent_sequence_keys) is tuple
                and len(candidate.edge.parent_sequence_keys) <= 2
                and type(candidate.edge.edit_description) is tuple
                and len(candidate.edge.edit_description) <= 5
                and type(candidate.edge.proposal_trace) is ProbabilityTrace
                and type(candidate.edge.proposal_trace.factors) is tuple
                and 0 < len(candidate.edge.proposal_trace.factors) <= 16
                and type(candidate.edge.behavior_log_probabilities) is tuple
                and len(candidate.edge.behavior_log_probabilities) == 0
                and type(candidate.edge.sampling_parameters) is tuple
                and len(candidate.edge.sampling_parameters) == len(_COMPOSED_EDGE_PARAMETER_NAMES),
                "GA composition edge structure differs",
            )
            for factor in (*selection.propensity.factors, *candidate.edge.proposal_trace.factors):
                _require(
                    type(factor) is ProbabilityFactor
                    and type(factor.name) is str
                    and 0 < len(factor.name) <= 128
                    and type(factor.probability) is float
                    and math.isfinite(factor.probability)
                    and 0.0 <= factor.probability <= 1.0,
                    "GA composition factor differs",
                )
                factor.__post_init__()
            for value, label in (
                (candidate.proposal.proposal_id, "GA composition proposal ID"),
                (candidate.proposal.rollout_id, "GA composition proposal rollout ID"),
                (candidate.proposal.policy_version, "GA composition proposal policy"),
                (candidate.edge.edge_id, "GA composition edge ID"),
                (candidate.edge.proposal_id, "GA composition edge proposal ID"),
                (candidate.edge.rollout_id, "GA composition edge rollout ID"),
                (candidate.edge.operator, "GA composition edge operator"),
            ):
                _identifier(value, label=label)
            _require(
                type(candidate.proposal.hard_valid) is bool
                and candidate.proposal.rejection_reason is None
                and type(candidate.proposal.proposal_round) is int
                and 0 <= candidate.proposal.proposal_round <= PEPTIDE_GA_MAX_ROUNDS,
                "GA composition proposal structure differs",
            )
            if candidate.proposal.niche_id is not None:
                _identifier(candidate.proposal.niche_id, label="GA composition proposal namespace")
            for key in candidate.edge.parent_sequence_keys:
                _sha256(key, label="GA composition parent sequence key")
            for value in candidate.edge.edit_description:
                _require(
                    (type(value) is str and 0 < len(value) <= 128)
                    or (type(value) is int and 0 <= value <= PEPTIDE_GA_SIGNED_63_MAX),
                    "GA composition edit description differs",
                )
            for row in candidate.edge.sampling_parameters:
                _require(
                    type(row) is tuple
                    and len(row) == 2
                    and type(row[0]) is str
                    and 0 < len(row[0]) <= 128
                    and (
                        (type(row[1]) is str and 0 < len(row[1]) <= 128)
                        or (type(row[1]) is int and 0 <= row[1] <= PEPTIDE_GA_SIGNED_63_MAX)
                    ),
                    "GA composition sampling parameter differs",
                )
            for name in (
                "canonical_sequence_id",
                "oracle_contract_sha256",
                "evaluator_sha256",
                "checkpoint_sha256",
                "endpoint_context_sha256",
                "transform_sha256",
            ):
                _sha256(
                    getattr(candidate.query_identity, name),
                    label=f"GA composition query identity {name}",
                )
            _require(
                type(candidate.query_identity.replicate_id) is int
                and 0 <= candidate.query_identity.replicate_id <= PEPTIDE_GA_SIGNED_63_MAX,
                "GA composition query identity replicate differs",
            )
            selection.propensity.__post_init__()
            selection.__post_init__()
            candidate.proposal.__post_init__()
            candidate.edge.proposal_trace.__post_init__()
            candidate.edge.__post_init__()
            candidate.query_identity.__post_init__()
            positions.append(candidate.prefix_position)
            attempt_indices.append(candidate.attempt_index)
            expected_proposal_id = (
                "ga-proposal-"
                + sha256_bytes(
                    _PROPOSAL_ID_HASH_DOMAIN
                    + bytes.fromhex(self.input_sha256)
                    + candidate.attempt_index.to_bytes(8, "big")
                )[:32]
            )
            expected_rollout_id = f"ga-rollout-{self.batch_id}-{self.seed}"
            _require(
                candidate.proposal.proposal_id == expected_proposal_id
                and candidate.edge.proposal_id == expected_proposal_id
                and candidate.edge.edge_id
                == "ga-edge-" + expected_proposal_id.removeprefix("ga-proposal-")
                and candidate.proposal.rollout_id == expected_rollout_id
                and candidate.edge.rollout_id == expected_rollout_id
                and candidate.edge.random_stream == self.seed
                and candidate.edge.sample_index == candidate.attempt_index
                and candidate.query_identity.canonical_sequence_id
                == candidate.proposal.sequence_key,
                "GA composition candidate identity differs",
            )
            _require(
                candidate.proposal.hard_valid
                and candidate.proposal.rejection_reason is None
                and candidate.proposal.policy_version == PEPTIDE_GA_POLICY_VERSION
                and candidate.proposal.niche_id == PEPTIDE_GA_NAMESPACE
                and candidate.proposal.proposal_round == self.authenticated_pre_wave_round_count,
                "GA composition proposal provenance differs",
            )
            _require(
                selection.selected
                and selection.selection_set_id == self.selection_set_id
                and selection.policy_version == PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION
                and selection.seed == self.seed
                and selection.propensity.factors
                == (ProbabilityFactor("controller_private_first_valid_prefix", 1.0),),
                "GA composition candidate selection-set differs",
            )
            _require(
                candidate.query_identity.replicate_id == self.replicate_id,
                "GA composition query replicate differs",
            )
            parameters = dict(candidate.edge.sampling_parameters)
            _require(
                len(parameters) == len(candidate.edge.sampling_parameters)
                and set(parameters) == _COMPOSED_EDGE_PARAMETER_NAMES,
                "GA composition edge parameter schema differs",
            )
            _require(
                parameters["algorithm"] == PEPTIDE_GA_POLICY_VERSION
                and parameters["archive_sha256"] == self.archive_sha256
                and parameters["config_sha256"] == self.config_sha256
                and parameters["input_sha256"] == self.input_sha256
                and parameters["rng"] == PEPTIDE_GA_RNG_VERSION
                and parameters["selection_batch_output_sha256"] == self.batch_output_sha256
                and parameters["selection_campaign_id"] == self.campaign_id
                and parameters["selection_config_sha256"] == self.config_sha256
                and parameters["selection_id_derivation"] == PEPTIDE_GA_SELECTION_ID_DERIVATION
                and parameters["selection_namespace"] == PEPTIDE_GA_NAMESPACE
                and parameters["selection_ordered_eligible_sha256"] == self.ordered_eligible_sha256
                and parameters["selection_policy_implementation_sha256"]
                == self.selection_policy_implementation_sha256
                and parameters["selection_pre_wave_head_sha256"]
                == self.authenticated_pre_wave_head_sha256
                and type(parameters["selection_pre_wave_round_count"]) is int
                and parameters["selection_pre_wave_round_count"]
                == self.authenticated_pre_wave_round_count
                and parameters["selection_wave_id"] == self.wave_id,
                "GA composition edge selection binding differs",
            )
            _require(
                candidate.query_identity.key not in query_keys,
                "GA composition query identities are not unique",
            )
            _require(
                candidate.proposal.proposal_id not in proposal_ids,
                "GA composition proposal IDs are not unique",
            )
            _require(
                candidate.proposal.sequence_key not in sequence_keys,
                "GA composition retained sequences are not unique",
            )
            proposal_ids.add(candidate.proposal.proposal_id)
            sequence_keys.add(candidate.proposal.sequence_key)
            query_keys.add(candidate.query_identity.key)
        _require(
            tuple(positions) == self.selected_prefix_positions
            and self.selected_prefix_positions
            == tuple(sorted(set(self.selected_prefix_positions))),
            "GA composition positions are not ordered and unique",
        )
        _require(
            attempt_indices == sorted(set(attempt_indices)),
            "GA composition attempt indices are not ordered and unique",
        )
        eligible = tuple(candidate.proposal.proposal_id for candidate in self.selected)
        _require(
            ordered_eligible_proposal_ids_sha256(eligible) == self.ordered_eligible_sha256,
            "GA composition eligible digest differs",
        )
        _require(
            all(
                candidate.proposal.selection.eligible_proposal_ids == eligible
                for candidate in self.selected
            ),
            "GA composition eligible inventories differ",
        )
        _require(
            self.selection_set_id
            == derive_wave_selection_set_id(
                namespace=PEPTIDE_GA_NAMESPACE,
                campaign_id=self.campaign_id,
                phase=self.phase,
                wave_id=self.wave_id,
                authenticated_pre_wave_head_sha256=(self.authenticated_pre_wave_head_sha256),
                authenticated_pre_wave_round_count=(self.authenticated_pre_wave_round_count),
                batch_output_sha256=self.batch_output_sha256,
                ordered_eligible_sha256=self.ordered_eligible_sha256,
                policy_version=PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
                policy_implementation_sha256=(self.selection_policy_implementation_sha256),
                seed=self.seed,
                config_sha256=self.config_sha256,
            ),
            "GA composition selection-set derivation differs",
        )
        expected_digest = sha256_bytes(
            _COMPOSITION_HASH_DOMAIN + canonical_json_bytes(_composition_document(self))
        )
        _require(self.composition_sha256 == expected_digest, "GA composition seal differs")
        _require(self.execution_authorized is False, "GA composition cannot authorize execution")


def _selection_binding_parameters(
    attempt: PeptideGAAttempt,
    *,
    batch: PeptideGABatch,
    archive: AuthenticatedWaveArchive,
    ordered_eligible_sha256: str,
    policy_implementation_sha256: str,
) -> tuple[tuple[str, object], ...]:
    return (
        *attempt.edge.sampling_parameters,
        ("selection_batch_output_sha256", batch.output_sha256),
        ("selection_campaign_id", batch.campaign_id),
        ("selection_config_sha256", batch.config_sha256),
        ("selection_id_derivation", PEPTIDE_GA_SELECTION_ID_DERIVATION),
        ("selection_namespace", PEPTIDE_GA_NAMESPACE),
        ("selection_ordered_eligible_sha256", ordered_eligible_sha256),
        ("selection_policy_implementation_sha256", policy_implementation_sha256),
        ("selection_pre_wave_head_sha256", archive.head_seal_sha256),
        ("selection_pre_wave_round_count", archive.round_count),
        ("selection_wave_id", batch.wave_id),
    )


def _compose_verified_batch(
    batch: PeptideGABatch,
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    private_collisions: ControllerPrivateCollisionInventory,
    *,
    truth_contract: FitnessTruthContract,
    policy_registry: AuthenticatedSelectionPolicyRegistry,
    replicate_id: int,
) -> ControllerGAComposition:
    """Compose only after an entry point independently replayed the batch."""

    preflight_peptide_ga_batch_structure(batch)
    _require(
        batch.status == "complete"
        and len(batch.accepted_proposal_ids) == PEPTIDE_GA_CANDIDATE_PREFIX_SIZE
        and len(batch.accepted_sequences) == PEPTIDE_GA_CANDIDATE_PREFIX_SIZE,
        "GA controller composition requires a complete fixed public prefix",
    )
    batch.__post_init__()
    _require(type(archive) is AuthenticatedWaveArchive, "GA archive type differs")
    _require(type(exclusions) is CollisionExclusions, "GA exclusions type differs")
    _require(type(truth_contract) is FitnessTruthContract, "truth contract type differs")
    _require(
        type(policy_registry) is AuthenticatedSelectionPolicyRegistry,
        "policy registry type differs",
    )
    archive.__post_init__()
    exclusions.__post_init__()
    truth_contract.__post_init__()
    policy_registry.__post_init__()
    _require(
        archive.truth_contract_sha256 == truth_contract.sha256,
        "GA archive truth contract differs",
    )
    _require(
        policy_registry.registry_sha256 == truth_contract.selection_policy_registry_sha256,
        "truth contract policy registry digest differs",
    )
    row = policy_registry.matching_row(
        namespace=PEPTIDE_GA_NAMESPACE,
        phase=archive.phase,
        proposal_round=archive.round_count,
        proposal_policy_version=PEPTIDE_GA_POLICY_VERSION,
        selection_policy_version=PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
        config_sha256=batch.config_sha256,
    )
    _require(
        row.selection_policy_implementation_sha256
        == PEPTIDE_GA_CONTROLLER_SELECTION_IMPLEMENTATION_SHA256,
        "controller policy implementation digest differs",
    )
    _require(
        batch.public_exclusion_receipt.training_set_sha256 == exclusions.training_set_sha256
        and batch.public_exclusion_receipt.training_count == exclusions.training_count
        and batch.public_exclusion_receipt.submitted_set_sha256 == exclusions.submitted_set_sha256
        and batch.public_exclusion_receipt.submitted_count == exclusions.submitted_count,
        "public-exclusion receipt differs before private composition",
    )
    attempt_by_id: dict[str, PeptideGAAttempt] = {}
    for attempt in batch.attempts:
        if attempt.accepted_position is not None:
            _require(
                attempt.proposal.proposal_id not in attempt_by_id,
                "GA accepted attempt proposal IDs are not unique",
            )
            attempt_by_id[attempt.proposal.proposal_id] = attempt
    _require(
        tuple(attempt_by_id) == batch.accepted_proposal_ids,
        "GA accepted public batch differs before private composition",
    )

    # Everything above is derived exclusively from the independently replayed
    # public batch, campaign archive, exclusions, and externally pinned registry.
    # Do not inspect controller-private state until those bindings are complete.
    _require(
        type(private_collisions) is ControllerPrivateCollisionInventory,
        "private collision inventory type differs",
    )
    private_collisions.__post_init__()
    _require(
        type(replicate_id) is int and 0 <= replicate_id <= PEPTIDE_GA_SIGNED_63_MAX,
        "replicate ID differs",
    )
    _require(
        not (
            (exclusions.training_sequence_key_set | exclusions.submitted_sequence_key_set)
            & frozenset(
                binding.canonical_sequence_id
                for binding in private_collisions.scheduled_reserve_bindings
            )
        ),
        "pre-reserve public exclusions contain a private reserve sequence",
    )
    charged = frozenset(private_collisions.charged_query_identity_keys)
    _require(
        all(
            binding.campaign_id == batch.campaign_id and binding.phase == archive.phase
            for binding in private_collisions.scheduled_reserve_bindings
        ),
        "scheduled reserve campaign or phase differs",
    )
    reserve_seats_by_wave: dict[str, tuple[str, list[int]]] = {}
    reserve_wave_by_batch: dict[str, str] = {}
    for binding in private_collisions.scheduled_reserve_bindings:
        prior = reserve_seats_by_wave.setdefault(binding.wave_id, (binding.batch_id, []))
        _require(prior[0] == binding.batch_id, "scheduled reserve wave reuses a batch identity")
        prior_wave = reserve_wave_by_batch.setdefault(binding.batch_id, binding.wave_id)
        _require(prior_wave == binding.wave_id, "scheduled reserve batch reuses a wave identity")
        prior[1].append(binding.batch_position)
    expected_reserve_positions = list(
        range(
            PEPTIDE_GA_METHOD_SEATS,
            PEPTIDE_GA_METHOD_SEATS + PEPTIDE_GA_PRIVATE_RESERVE_SEATS,
        )
    )
    _require(
        all(
            positions == expected_reserve_positions
            for _, positions in reserve_seats_by_wave.values()
        ),
        "scheduled reserve wave does not bind the exact reserve seats",
    )
    reserve = frozenset(
        binding.query_identity_key for binding in private_collisions.scheduled_reserve_bindings
    )
    expected_reserve = frozenset(
        OracleQueryIdentity(
            canonical_sequence_id=binding.canonical_sequence_id,
            oracle_contract_sha256=truth_contract.oracle_contract_sha256,
            evaluator_sha256=truth_contract.evaluator_sha256,
            checkpoint_sha256=truth_contract.checkpoint_sha256,
            endpoint_context_sha256=truth_contract.endpoint_context_sha256,
            transform_sha256=truth_contract.transform_sha256,
            replicate_id=replicate_id,
        ).key
        for binding in private_collisions.scheduled_reserve_bindings
    )
    _require(
        reserve == expected_reserve,
        "scheduled reserve query identities do not match their canonical sequences",
    )
    current_reserve = tuple(
        binding
        for binding in private_collisions.scheduled_reserve_bindings
        if binding.wave_id == batch.wave_id
    )
    _require(
        len(current_reserve) == PEPTIDE_GA_PRIVATE_RESERVE_SEATS
        and all(
            binding.campaign_id == batch.campaign_id
            and binding.phase == archive.phase
            and binding.batch_id == batch.batch_id
            for binding in current_reserve
        )
        and tuple(binding.batch_position for binding in current_reserve)
        == tuple(expected_reserve_positions),
        "scheduled reserve bindings do not fill the exact current batch seats",
    )
    prefix: list[tuple[int, PeptideGAAttempt, OracleQueryIdentity]] = []
    availability: list[bool] = []
    retained_sequence_keys: set[str] = set()
    retained_query_keys: set[str] = set()
    for position, (proposal_id, sequence) in enumerate(
        zip(batch.accepted_proposal_ids, batch.accepted_sequences, strict=True)
    ):
        attempt = attempt_by_id.get(proposal_id)
        _require(
            attempt is not None
            and attempt.proposal.hard_valid
            and attempt.proposal.sequence == sequence,
            "GA prefix proposal differs",
        )
        identity = OracleQueryIdentity(
            canonical_sequence_id=attempt.proposal.sequence_key,
            oracle_contract_sha256=truth_contract.oracle_contract_sha256,
            evaluator_sha256=truth_contract.evaluator_sha256,
            checkpoint_sha256=truth_contract.checkpoint_sha256,
            endpoint_context_sha256=truth_contract.endpoint_context_sha256,
            transform_sha256=truth_contract.transform_sha256,
            replicate_id=replicate_id,
        )
        prefix.append((position, attempt, identity))
        availability.append(identity.key not in charged and identity.key not in reserve)
    try:
        selected_positions = (
            peptide_ga_selection_policy_impl_v1.controller_first_available_prefix_positions(
                tuple(availability), seat_count=PEPTIDE_GA_METHOD_SEATS
            )
        )
    except ValueError as error:
        raise PeptideGAError("insufficient collision-free GA prefix") from error
    retained = [prefix[position] for position in selected_positions]
    for _, attempt, identity in retained:
        _require(
            attempt.proposal.sequence_key not in retained_sequence_keys,
            "GA retained sequences are not unique",
        )
        _require(identity.key not in retained_query_keys, "GA retained query keys are not unique")
        retained_sequence_keys.add(attempt.proposal.sequence_key)
        retained_query_keys.add(identity.key)
    _require(len(retained) == PEPTIDE_GA_METHOD_SEATS, "insufficient collision-free GA prefix")
    eligible = tuple(attempt.proposal.proposal_id for _, attempt, _ in retained)
    eligible_sha256 = ordered_eligible_proposal_ids_sha256(eligible)
    selection_set_id = derive_wave_selection_set_id(
        namespace=PEPTIDE_GA_NAMESPACE,
        campaign_id=batch.campaign_id,
        phase=archive.phase,
        wave_id=batch.wave_id,
        authenticated_pre_wave_head_sha256=archive.head_seal_sha256,
        authenticated_pre_wave_round_count=archive.round_count,
        batch_output_sha256=batch.output_sha256,
        ordered_eligible_sha256=eligible_sha256,
        policy_version=PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
        policy_implementation_sha256=row.selection_policy_implementation_sha256,
        seed=batch.seed,
        config_sha256=batch.config_sha256,
        derivation=row.selection_id_derivation,
    )
    selected = tuple(
        ControllerOwnedGACandidate(
            prefix_position=position,
            attempt_index=attempt.attempt_index,
            proposal=replace(
                attempt.proposal,
                selection=SelectionDecision(
                    selected=True,
                    propensity=ProbabilityTrace(
                        (ProbabilityFactor("controller_private_first_valid_prefix", 1.0),)
                    ),
                    selection_set_id=selection_set_id,
                    eligible_proposal_ids=eligible,
                    policy_version=PEPTIDE_GA_CONTROLLER_SELECTION_POLICY_VERSION,
                    seed=batch.seed,
                ),
            ),
            edge=replace(
                attempt.edge,
                sampling_parameters=_selection_binding_parameters(
                    attempt,
                    batch=batch,
                    archive=archive,
                    ordered_eligible_sha256=eligible_sha256,
                    policy_implementation_sha256=row.selection_policy_implementation_sha256,
                ),
            ),
            query_identity=identity,
        )
        for position, attempt, identity in retained
    )
    positions = tuple(candidate.prefix_position for candidate in selected)
    document = {
        "archive_sha256": archive.archive_sha256,
        "authenticated_pre_wave_head_sha256": archive.head_seal_sha256,
        "authenticated_pre_wave_round_count": archive.round_count,
        "batch_id": batch.batch_id,
        "batch_output_sha256": batch.output_sha256,
        "campaign_id": batch.campaign_id,
        "config_sha256": batch.config_sha256,
        "input_sha256": batch.input_sha256,
        "ordered_eligible_sha256": eligible_sha256,
        "phase": archive.phase,
        "policy_registry_sha256": policy_registry.registry_sha256,
        "public_exclusion_receipt_sha256": batch.public_exclusion_receipt.receipt_sha256,
        "query_identity_keys": [candidate.query_identity.key for candidate in selected],
        "replicate_id": replicate_id,
        "seed": batch.seed,
        "selected_attempt_indices": [candidate.attempt_index for candidate in selected],
        "selected_prefix_positions": list(positions),
        "selected_proposal_ids": [candidate.proposal.proposal_id for candidate in selected],
        "selected_sequence_ids": [candidate.proposal.sequence_key for candidate in selected],
        "selection_policy_implementation_sha256": (row.selection_policy_implementation_sha256),
        "selection_set_id": selection_set_id,
        "truth_contract_sha256": truth_contract.sha256,
        "wave_id": batch.wave_id,
    }
    digest = sha256_bytes(_COMPOSITION_HASH_DOMAIN + canonical_json_bytes(document))
    return ControllerGAComposition(
        campaign_id=batch.campaign_id,
        phase=archive.phase,
        wave_id=batch.wave_id,
        batch_id=batch.batch_id,
        authenticated_pre_wave_head_sha256=archive.head_seal_sha256,
        authenticated_pre_wave_round_count=archive.round_count,
        archive_sha256=archive.archive_sha256,
        batch_output_sha256=batch.output_sha256,
        config_sha256=batch.config_sha256,
        input_sha256=batch.input_sha256,
        public_exclusion_receipt_sha256=batch.public_exclusion_receipt.receipt_sha256,
        policy_registry_sha256=policy_registry.registry_sha256,
        selection_policy_implementation_sha256=(row.selection_policy_implementation_sha256),
        truth_contract_sha256=truth_contract.sha256,
        seed=batch.seed,
        replicate_id=replicate_id,
        selection_set_id=selection_set_id,
        ordered_eligible_sha256=eligible_sha256,
        selected=selected,
        selected_prefix_positions=positions,
        composition_sha256=digest,
    )


def compose_unverified_controller_private_ga_fixture_seats(
    batch: PeptideGABatch,
    archive: AuthenticatedWaveArchive,
    exclusions: CollisionExclusions,
    private_collisions: ControllerPrivateCollisionInventory,
    *,
    config: PeptideGAConfig,
    truth_contract: FitnessTruthContract,
    policy_registry: AuthenticatedSelectionPolicyRegistry,
    replicate_id: int = 0,
) -> ControllerGAComposition:
    """Fixture-only composition; independently replay the unverified batch first."""

    verify_unverified_peptide_ga_fixture_batch(batch, archive, exclusions, config=config)
    return _compose_verified_batch(
        batch,
        archive,
        exclusions,
        private_collisions,
        truth_contract=truth_contract,
        policy_registry=policy_registry,
        replicate_id=replicate_id,
    )


def compose_controller_private_ga_seats_from_path(
    batch: PeptideGABatch,
    root: str | Path,
    exclusions: CollisionExclusions,
    private_collisions: ControllerPrivateCollisionInventory,
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
    replicate_id: int = 0,
) -> ControllerGAComposition:
    """Authenticate and independently replay the public batch before private access."""

    archive = verify_peptide_ga_batch_from_path(
        batch,
        root,
        exclusions,
        trusted_parent=trusted_parent,
        wave_id=wave_id,
        expected_header_sha256=expected_header_sha256,
        expected_head_seal_sha256=expected_head_seal_sha256,
        expected_round_count=expected_round_count,
        expected_query_count=expected_query_count,
        expected_response_count=expected_response_count,
        config=config,
        config_path=config_path,
        trusted_config_parent=trusted_config_parent,
        expected_config_sha256=expected_config_sha256,
        policy_registry_path=policy_registry_path,
        trusted_policy_registry_parent=trusted_policy_registry_parent,
        expected_policy_registry_sha256=expected_policy_registry_sha256,
        exclusion_asset_path=exclusion_asset_path,
        exclusion_receipt_path=exclusion_receipt_path,
        trusted_exclusion_parent=trusted_exclusion_parent,
        expected_exclusion_asset_sha256=expected_exclusion_asset_sha256,
        expected_exclusion_receipt_sha256=expected_exclusion_receipt_sha256,
        expected_exclusion_issuer_identity_sha256=expected_exclusion_issuer_identity_sha256,
        truth_contract=truth_contract,
    )
    policy_registry = load_selection_policy_registry_from_path(
        policy_registry_path,
        trusted_registry_parent=trusted_policy_registry_parent,
        expected_registry_sha256=expected_policy_registry_sha256,
    )
    return _compose_verified_batch(
        batch,
        archive,
        exclusions,
        private_collisions,
        truth_contract=truth_contract,
        policy_registry=policy_registry,
        replicate_id=replicate_id,
    )


def compose_controller_private_ga_seats(*args: object, **kwargs: object) -> ControllerGAComposition:
    """Fail-closed compatibility name for the removed unauthenticated composer."""

    del args, kwargs
    raise PeptideGAError(
        "unauthenticated composition was removed; use the path or explicit fixture entry point"
    )


__all__ = [
    "ControllerGAComposition",
    "ControllerOwnedGACandidate",
    "ControllerPrivateCollisionInventory",
    "ControllerPrivateReserveBinding",
    "compose_controller_private_ga_seats",
    "compose_controller_private_ga_seats_from_path",
    "compose_unverified_controller_private_ga_fixture_seats",
]

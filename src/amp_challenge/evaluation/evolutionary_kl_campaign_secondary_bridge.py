"""Fail-closed campaign-ledger to secondary-evidence bridge.

The bridge preserves exact sealed campaign identities while translating the
ledger's query/response vocabulary into the secondary reducer's raw JSONL.
Every output remains non-authorizing.  In particular, opaque controller and
timing receipt digests are bound but never promoted to independently verified
facts by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from amp_challenge.evaluation.evolutionary_kl_protocol import (
    CONFIRMATION_METHOD_IDS,
    CONFIRMATION_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    EvolutionaryKLProtocol,
)
from amp_challenge.evaluation.evolutionary_kl_secondary_evidence import (
    MAX_AUXILIARY_SEQUENCE_LENGTH,
    MAX_CONTRACT_BYTES,
    MAX_RAW_EVIDENCE_BYTES,
    MAX_SEQUENCE_SET_BYTES,
    MAX_TRAINING_SEQUENCE_RECORDS,
    RAW_HEADER_ARTIFACT,
    RAW_QUERY_ARTIFACT,
    SEQUENCE_SET_ROW_ARTIFACT,
    SecondaryEvidenceError,
    SecondaryEvidenceTrustAnchors,
    SecondaryYieldEvidence,
    compute_authenticated_secondary_yield,
    query_identity_inventory_sha256,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from amp_challenge.generators.search.campaign_ledger import (
    EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    QUERY_IDENTITY_FIELDS,
    ReplayedCampaignQuery,
    VerifiedCampaign,
    replay_verified_campaign_event_documents,
    verify_campaign,
)
from amp_challenge.sequences import canonicalize_sequence

BRIDGE_ARTIFACT = "evolutionary_kl_campaign_secondary_bridge_receipt_v1"
STOP_METADATA_ARTIFACT = "evolutionary_kl_campaign_secondary_stop_metadata_v1"
OUTCOME_MAPPING_ARTIFACT = "evolutionary_kl_campaign_secondary_outcome_mapping_v1"
BRIDGE_RECEIPT_HASH_DOMAIN = b"amp/evolutionary-kl/campaign-secondary-bridge/v1\0"
ROUND_SEAL_INVENTORY_HASH_DOMAIN = b"amp/evolutionary-kl/bridge-round-seals/v1\0"
ROUND_TIMING_INVENTORY_HASH_DOMAIN = b"amp/evolutionary-kl/bridge-round-timing/v1\0"
SOURCE_QUERY_INVENTORY_HASH_DOMAIN = b"amp/evolutionary-kl/bridge-source-queries/v1\0"
MAX_BRIDGE_CUMULATIVE_EVENT_BYTES = 8 * MAX_RAW_EVIDENCE_BYTES
MAX_BRIDGE_CONSTRAINT_COUNT = 64
MAX_BRIDGE_STOP_CALL_COUNT = 512
MAX_BRIDGE_SCIENTIFIC_ELAPSED_NANOSECONDS = 7_200_000_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONSTRAINT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_CAMPAIGN_RESPONSE_STATUSES = ("censored", "failed", "missing", "partial", "timeout")
_STOP_REASONS = frozenset(
    {"unique_call_budget_reached", "scientific_wall_limit_reached", "algorithmic_failure"}
)
_TIMING_AUTHORITY_STATUS = "external_digest_bindings_not_independently_verified"
_KNOWN_UNPROVEN_AUTHORITIES = (
    "campaign_head_external_persistence",
    "chemical_form_and_pre_submission_support_enforcement",
    "common_initial_and_random_reserve_authentication",
    "oracle_transport_and_endpoint_semantics",
    "scheduler_timing_sealed_prefix_and_stop_receipts",
    "streaming_replay_capacity_and_real_scale",
    "terminal_posterior_mean_argmax_evidence",
)
_TRUTH_SOURCE_FIELDS = (
    "oracle_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
)
_CONSTRAINT_CONTRACT_KEYS = frozenset(
    {"schema_version", "artifact", "status", "protocol_sha256", "constraints"}
)
_CONSTRAINT_RULE_KEYS = frozenset({"constraint_id", "operator", "threshold"})
_TRUTH_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "protocol_sha256",
        "constraint_contract_sha256",
        "oracle_contract_sha256",
        "objective_ids",
        "objective_bounds_hex",
        "complete_response_status",
        "complete_requires_atomic_all_finite_uncensored",
        "noncomplete_is_ineligible",
        "constraint_pass_recomputed_from_raw_value",
    }
)
_SEQUENCE_SET_ROW_KEYS = frozenset({"schema_version", "artifact", "sequence_id", "sequence"})
_MAPPING_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "protocol_sha256",
        "constraint_contract_sha256",
        "oracle_contract_sha256",
        "truth_contract_sha256",
        "truth_source_identity",
        "campaign_success_status",
        "secondary_complete_status",
        "campaign_non_success_statuses",
        "non_success_status_mapping",
        "objective_outcome_names",
        "constraint_outcome_names",
        "success_requires_exact_outcome_inventory",
        "success_values_are_atomic",
        "success_values_are_uncensored",
        "non_success_values_absent",
        "non_success_censoring_unknown_is_null",
        "execution_authorized",
        "scientific_claim_authorized",
        "production_authorized",
    }
)
_AUTHORITY_FIELDS = (
    "run_id",
    "phase",
    "method_id",
    "seed",
    "campaign_header_sha256",
    "campaign_head_seal_sha256",
    "campaign_round_count",
    "campaign_round_seal_inventory_sha256",
    "campaign_round_timing_receipt_inventory_sha256",
    "campaign_last_event_sha256",
    "campaign_event_count",
    "campaign_proposal_count",
    "campaign_query_count",
    "campaign_response_count",
    "stop_metadata_sha256",
    "outcome_mapping_contract_sha256",
    "constraint_contract_sha256",
    "oracle_contract_sha256",
    "support_contract_sha256",
    "training_sequence_set_sha256",
    "homology_contract_sha256",
    "reference_contract_sha256",
    "reference_sequence_set_sha256",
    "truth_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
)
_AUTHORITY_DIGEST_FIELDS = (
    "campaign_header_sha256",
    "campaign_head_seal_sha256",
    "campaign_round_seal_inventory_sha256",
    "campaign_round_timing_receipt_inventory_sha256",
    "campaign_last_event_sha256",
    "stop_metadata_sha256",
    "outcome_mapping_contract_sha256",
    "constraint_contract_sha256",
    "oracle_contract_sha256",
    "support_contract_sha256",
    "training_sequence_set_sha256",
    "homology_contract_sha256",
    "reference_contract_sha256",
    "reference_sequence_set_sha256",
    "truth_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
)
_RECEIPT_KEYS = frozenset(
    {
        "artifact",
        "authorization",
        "bridge_receipt_sha256",
        "campaign",
        "known_unproven_authorities",
        "mapping",
        "protocol_sha256",
        "schema_version",
        "secondary",
        "source_authority",
        "status",
        "stop_metadata",
        "stop_metadata_sha256",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {"execution_authorized", "scientific_claim_authorized", "production_authorized"}
)
_RECEIPT_SECONDARY_KEYS = frozenset(
    {
        "query_identity_inventory_sha256",
        "raw_evidence_sha256",
        "secondary_evidence_sha256",
    }
)
_RECEIPT_MAPPING_KEYS = frozenset(
    {
        "campaign_call_position_origin",
        "campaign_success_status",
        "outcome_mapping_contract_sha256",
        "raw_charged_call_position_origin",
        "secondary_complete_status",
    }
)
_RECEIPT_CAMPAIGN_KEYS = frozenset(
    {
        "event_count",
        "event_document_inventory_sha256",
        "header_sha256",
        "head_seal_sha256",
        "last_event_sha256",
        "proposal_count",
        "proposal_sequence_inventory_sha256",
        "query_count",
        "response_count",
        "round_count",
        "round_seal_inventory_sha256",
        "round_seals",
        "round_timing_receipt_inventory_sha256",
        "round_timing_receipt_sha256s",
        "source_query_inventory_sha256",
        "terminal",
    }
)
_STOP_DOCUMENT_KEYS = frozenset(
    {
        "artifact",
        "authority_status",
        "execution_authorized",
        "production_authorized",
        "round_timing_receipt_sha256s",
        "schema_version",
        "scientific_claim_authorized",
        "scientific_elapsed_nanoseconds",
        "sealed_charged_call_count",
        "sealed_prefix_receipt_sha256",
        "status",
        "stop_reason",
        "stop_receipt_sha256",
        "timing_authority_receipt_sha256",
        "unsealed_discarded_call_count",
    }
)


class CampaignSecondaryBridgeError(ValueError):
    """Raised when a campaign cannot be translated without inventing evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignSecondaryBridgeError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and len(value) == 64 and _SHA256_RE.fullmatch(value) is not None,
        f"{label} invalid",
    )
    assert isinstance(value, str)
    return value


def _identifier(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _IDENTIFIER_RE.fullmatch(value) is not None,
        f"{label} invalid",
    )
    assert isinstance(value, str)
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a non-negative integer")
    assert isinstance(value, int)
    return value


@dataclass(frozen=True, slots=True)
class CampaignSecondaryBridgeAuthority:
    """Caller-pinned source and semantic identities, held outside producer bytes."""

    run_id: str
    phase: str
    method_id: str
    seed: int
    campaign_header_sha256: str
    campaign_head_seal_sha256: str
    campaign_round_count: int
    campaign_round_seal_inventory_sha256: str
    campaign_round_timing_receipt_inventory_sha256: str
    campaign_last_event_sha256: str
    campaign_event_count: int
    campaign_proposal_count: int
    campaign_query_count: int
    campaign_response_count: int
    stop_metadata_sha256: str
    outcome_mapping_contract_sha256: str
    constraint_contract_sha256: str
    oracle_contract_sha256: str
    support_contract_sha256: str
    training_sequence_set_sha256: str
    homology_contract_sha256: str
    reference_contract_sha256: str
    reference_sequence_set_sha256: str
    truth_contract_sha256: str
    evaluator_sha256: str
    checkpoint_sha256: str
    endpoint_context_sha256: str
    transform_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.run_id, label="authority run ID")
        _require(self.phase == "confirmation", "authority phase differs")
        _identifier(self.method_id, label="authority method ID")
        _nonnegative_integer(self.seed, label="authority seed")
        for field in (
            "campaign_round_count",
            "campaign_event_count",
            "campaign_proposal_count",
            "campaign_query_count",
            "campaign_response_count",
        ):
            _nonnegative_integer(getattr(self, field), label=f"authority {field}")
        _require(self.campaign_round_count > 0, "authority campaign must have a sealed round")
        digests: list[str] = []
        for field in _AUTHORITY_DIGEST_FIELDS:
            digests.append(_sha256(getattr(self, field), label=f"authority {field}"))
        _require(len(set(digests)) == len(digests), "bridge authority digest roles alias")

    def document(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _AUTHORITY_FIELDS}


@dataclass(frozen=True, slots=True)
class CampaignSecondaryStopMetadata:
    """Explicit prefix/stop/timing bindings with no timing-authority claim."""

    sealed_charged_call_count: int
    unsealed_discarded_call_count: int
    stop_reason: Literal[
        "unique_call_budget_reached",
        "scientific_wall_limit_reached",
        "algorithmic_failure",
    ]
    scientific_elapsed_nanoseconds: int
    round_timing_receipt_sha256s: tuple[str, ...]
    sealed_prefix_receipt_sha256: str
    stop_receipt_sha256: str
    timing_authority_receipt_sha256: str
    execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    production_authorized: bool = False

    def __post_init__(self) -> None:
        sealed_call_count = _nonnegative_integer(
            self.sealed_charged_call_count,
            label="sealed call count",
        )
        discarded_call_count = _nonnegative_integer(
            self.unsealed_discarded_call_count,
            label="discarded call count",
        )
        _require(
            sealed_call_count <= MAX_BRIDGE_STOP_CALL_COUNT,
            "sealed call count exceeds the frozen ceiling",
        )
        _require(
            discarded_call_count <= MAX_BRIDGE_STOP_CALL_COUNT,
            "discarded call count exceeds the frozen ceiling",
        )
        _require(self.stop_reason in _STOP_REASONS, "stop reason differs")
        elapsed_nanoseconds = _nonnegative_integer(
            self.scientific_elapsed_nanoseconds,
            label="scientific elapsed time",
        )
        _require(
            elapsed_nanoseconds <= MAX_BRIDGE_SCIENTIFIC_ELAPSED_NANOSECONDS,
            "scientific elapsed time exceeds the frozen ceiling",
        )
        _require(
            type(self.round_timing_receipt_sha256s) is tuple
            and bool(self.round_timing_receipt_sha256s)
            and len(self.round_timing_receipt_sha256s)
            <= EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS.max_rounds,
            "round timing-receipt inventory differs",
        )
        for index, digest in enumerate(self.round_timing_receipt_sha256s):
            _sha256(digest, label=f"round timing receipt {index}")
        for field in (
            "sealed_prefix_receipt_sha256",
            "stop_receipt_sha256",
            "timing_authority_receipt_sha256",
        ):
            _sha256(getattr(self, field), label=field)
        _require(
            len(
                {
                    self.sealed_prefix_receipt_sha256,
                    self.stop_receipt_sha256,
                    self.timing_authority_receipt_sha256,
                }
            )
            == 3,
            "stop authority digest roles alias",
        )
        _require(
            self.execution_authorized is False
            and self.scientific_claim_authorized is False
            and self.production_authorized is False,
            "stop metadata cannot authorize execution, claims, or production",
        )

    def document(self) -> dict[str, object]:
        return {
            "artifact": STOP_METADATA_ARTIFACT,
            "authority_status": _TIMING_AUTHORITY_STATUS,
            "execution_authorized": False,
            "production_authorized": False,
            "round_timing_receipt_sha256s": list(self.round_timing_receipt_sha256s),
            "schema_version": 1,
            "scientific_claim_authorized": False,
            "scientific_elapsed_nanoseconds": self.scientific_elapsed_nanoseconds,
            "sealed_charged_call_count": self.sealed_charged_call_count,
            "sealed_prefix_receipt_sha256": self.sealed_prefix_receipt_sha256,
            "status": "external_bindings_required_non_authorizing",
            "stop_reason": self.stop_reason,
            "stop_receipt_sha256": self.stop_receipt_sha256,
            "timing_authority_receipt_sha256": self.timing_authority_receipt_sha256,
            "unsealed_discarded_call_count": self.unsealed_discarded_call_count,
        }

    def document_bytes(self) -> bytes:
        self.__post_init__()
        return canonical_json_bytes(self.document())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.document_bytes())


@dataclass(frozen=True, slots=True)
class CampaignSecondaryStructuralEnvelope:
    """Self-consistent bytes that are not an authenticated or accepted replay.

    Direct construction proves only canonical structure and internal digest
    consistency.  Only the path-reopening replay API can produce a
    :class:`VerifiedCampaignSecondaryReplay`.
    """

    raw_evidence_bytes: bytes
    receipt_bytes: bytes
    trust_anchors: SecondaryEvidenceTrustAnchors
    bridge_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.raw_evidence_bytes) is bytes, "bridge raw evidence must be bytes")
        _require(type(self.receipt_bytes) is bytes, "bridge receipt must be bytes")
        _require(
            0 < len(self.raw_evidence_bytes) <= MAX_RAW_EVIDENCE_BYTES,
            "bridge raw evidence byte bound exceeded",
        )
        _require(
            type(self.trust_anchors) is SecondaryEvidenceTrustAnchors,
            "bridge trust-anchor type differs",
        )
        self.trust_anchors.__post_init__()
        _require(
            sha256_bytes(self.raw_evidence_bytes) == self.trust_anchors.raw_evidence_sha256,
            "bridge raw evidence digest differs from its trust anchor",
        )
        supplied_digest = _sha256(
            self.bridge_receipt_sha256,
            label="bridge receipt SHA-256",
        )
        receipt = _exact_object(
            _strict_json_object(self.receipt_bytes, label="bridge receipt"),
            _RECEIPT_KEYS,
            label="bridge receipt",
        )
        _require(
            receipt["artifact"] == BRIDGE_ARTIFACT
            and receipt["schema_version"] == 1
            and type(receipt["schema_version"]) is int
            and receipt["status"] == "structural_non_authorizing_non_acceptance_envelope"
            and receipt["protocol_sha256"] == FROZEN_PROTOCOL_SHA256,
            "bridge receipt identity differs",
        )
        authorization = _exact_object(
            receipt["authorization"],
            _AUTHORIZATION_KEYS,
            label="bridge receipt authorization",
        )
        _require(
            authorization
            == {
                "execution_authorized": False,
                "production_authorized": False,
                "scientific_claim_authorized": False,
            },
            "bridge receipt cannot authorize execution, claims, or production",
        )
        _require(
            receipt["known_unproven_authorities"] == list(_KNOWN_UNPROVEN_AUTHORITIES),
            "bridge receipt unproven-authority inventory differs",
        )
        embedded_digest = _sha256(
            receipt.get("bridge_receipt_sha256"),
            label="embedded bridge receipt SHA-256",
        )
        unsigned = {key: value for key, value in receipt.items() if key != "bridge_receipt_sha256"}
        recomputed_digest = sha256_bytes(
            BRIDGE_RECEIPT_HASH_DOMAIN + canonical_json_bytes(unsigned)
        )
        _require(
            supplied_digest == embedded_digest == recomputed_digest,
            "bridge receipt self-hash differs",
        )
        secondary = _exact_object(
            receipt["secondary"],
            _RECEIPT_SECONDARY_KEYS,
            label="bridge receipt secondary binding",
        )
        _sha256(secondary["secondary_evidence_sha256"], label="secondary evidence SHA-256")
        _require(
            secondary.get("raw_evidence_sha256") == self.trust_anchors.raw_evidence_sha256
            and secondary.get("query_identity_inventory_sha256")
            == self.trust_anchors.query_identity_inventory_sha256,
            "bridge receipt secondary trust binding differs",
        )
        source_authority = _exact_object(
            receipt["source_authority"],
            frozenset(_AUTHORITY_FIELDS),
            label="bridge receipt source authority",
        )
        source_authority_record = CampaignSecondaryBridgeAuthority(**source_authority)  # type: ignore[arg-type]
        for field in (
            "run_id",
            "phase",
            "method_id",
            "seed",
            "constraint_contract_sha256",
            "oracle_contract_sha256",
            "support_contract_sha256",
            "training_sequence_set_sha256",
            "homology_contract_sha256",
            "reference_contract_sha256",
            "reference_sequence_set_sha256",
            "truth_contract_sha256",
            "evaluator_sha256",
            "checkpoint_sha256",
            "endpoint_context_sha256",
            "transform_sha256",
        ):
            _require(
                getattr(source_authority_record, field) == getattr(self.trust_anchors, field),
                f"bridge receipt trust-anchor field differs: {field}",
            )
        mapping = _exact_object(
            receipt["mapping"],
            _RECEIPT_MAPPING_KEYS,
            label="bridge receipt mapping",
        )
        _require(
            mapping
            == {
                "campaign_call_position_origin": 0,
                "campaign_success_status": "succeeded",
                "outcome_mapping_contract_sha256": (
                    source_authority_record.outcome_mapping_contract_sha256
                ),
                "raw_charged_call_position_origin": 1,
                "secondary_complete_status": "complete",
            },
            "bridge receipt mapping differs",
        )
        stop_document = _exact_object(
            receipt["stop_metadata"],
            _STOP_DOCUMENT_KEYS,
            label="bridge receipt stop metadata",
        )
        stop_round_timings = stop_document["round_timing_receipt_sha256s"]
        _require(
            type(stop_round_timings) is list
            and 0 < len(stop_round_timings) <= EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS.max_rounds,
            "bridge receipt stop timing inventory differs",
        )
        stop_record = CampaignSecondaryStopMetadata(
            sealed_charged_call_count=stop_document["sealed_charged_call_count"],  # type: ignore[arg-type]
            unsealed_discarded_call_count=stop_document["unsealed_discarded_call_count"],  # type: ignore[arg-type]
            stop_reason=stop_document["stop_reason"],  # type: ignore[arg-type]
            scientific_elapsed_nanoseconds=stop_document["scientific_elapsed_nanoseconds"],  # type: ignore[arg-type]
            round_timing_receipt_sha256s=tuple(stop_round_timings),
            sealed_prefix_receipt_sha256=stop_document["sealed_prefix_receipt_sha256"],  # type: ignore[arg-type]
            stop_receipt_sha256=stop_document["stop_receipt_sha256"],  # type: ignore[arg-type]
            timing_authority_receipt_sha256=stop_document["timing_authority_receipt_sha256"],  # type: ignore[arg-type]
            execution_authorized=stop_document["execution_authorized"],  # type: ignore[arg-type]
            scientific_claim_authorized=stop_document["scientific_claim_authorized"],  # type: ignore[arg-type]
            production_authorized=stop_document["production_authorized"],  # type: ignore[arg-type]
        )
        _require(stop_record.document() == stop_document, "bridge receipt stop metadata differs")
        _require(
            receipt["stop_metadata_sha256"]
            == source_authority_record.stop_metadata_sha256
            == stop_record.sha256,
            "bridge receipt stop metadata digest differs",
        )
        _require(
            stop_record.sealed_charged_call_count + stop_record.unsealed_discarded_call_count
            == source_authority_record.campaign_query_count,
            "bridge receipt stop call arithmetic differs",
        )
        campaign = _exact_object(
            receipt["campaign"],
            _RECEIPT_CAMPAIGN_KEYS,
            label="bridge receipt campaign",
        )
        _require(campaign["terminal"] is True, "bridge receipt campaign is not terminal")
        campaign_bindings = {
            "event_count": source_authority_record.campaign_event_count,
            "header_sha256": source_authority_record.campaign_header_sha256,
            "head_seal_sha256": source_authority_record.campaign_head_seal_sha256,
            "last_event_sha256": source_authority_record.campaign_last_event_sha256,
            "proposal_count": source_authority_record.campaign_proposal_count,
            "query_count": source_authority_record.campaign_query_count,
            "response_count": source_authority_record.campaign_response_count,
            "round_count": source_authority_record.campaign_round_count,
            "round_seal_inventory_sha256": (
                source_authority_record.campaign_round_seal_inventory_sha256
            ),
            "round_timing_receipt_inventory_sha256": (
                source_authority_record.campaign_round_timing_receipt_inventory_sha256
            ),
        }
        for field, expected in campaign_bindings.items():
            _require(campaign[field] == expected, f"bridge receipt campaign {field} differs")
        for field in (
            "event_document_inventory_sha256",
            "proposal_sequence_inventory_sha256",
            "round_seal_inventory_sha256",
            "round_timing_receipt_inventory_sha256",
            "source_query_inventory_sha256",
        ):
            _sha256(campaign[field], label=f"bridge receipt campaign {field}")
        round_seals = campaign["round_seals"]
        round_timings = campaign["round_timing_receipt_sha256s"]
        _require(
            type(round_seals) is list
            and type(round_timings) is list
            and 0 < len(round_seals) <= EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS.max_rounds
            and len(round_seals)
            == len(round_timings)
            == source_authority_record.campaign_round_count,
            "bridge receipt round inventories differ",
        )
        for index, digest in enumerate(round_seals):
            _sha256(digest, label=f"bridge receipt round seal {index}")
        for index, digest in enumerate(round_timings):
            _sha256(digest, label=f"bridge receipt round timing {index}")
        _require(
            bool(round_seals)
            and round_seals[-1] == source_authority_record.campaign_head_seal_sha256,
            "bridge receipt round head differs",
        )
        _require(
            round_timings == list(stop_record.round_timing_receipt_sha256s),
            "bridge receipt timing inventories differ",
        )
        round_seal_tuple = tuple(round_seals)
        round_timing_tuple = tuple(round_timings)
        _require(
            campaign["round_seal_inventory_sha256"]
            == campaign_round_seal_inventory_sha256(round_seal_tuple)  # type: ignore[arg-type]
            and campaign["round_timing_receipt_inventory_sha256"]
            == campaign_round_timing_receipt_inventory_sha256(  # type: ignore[arg-type]
                round_timing_tuple
            ),
            "bridge receipt round inventory digest differs",
        )
        _validate_receipt_digest_roles(
            source_authority_record,
            stop_record,
            campaign=campaign,
            secondary=secondary,
            bridge_receipt_sha256=supplied_digest,
        )


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    _require(type(payload) is bytes, f"{label} must be bytes")
    _require(0 < len(payload) <= MAX_CONTRACT_BYTES, f"{label} byte bound exceeded")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            _require(key not in result, f"{label} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise CampaignSecondaryBridgeError(f"{label} contains invalid constant {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except CampaignSecondaryBridgeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise CampaignSecondaryBridgeError(f"{label} is not strict UTF-8 JSON") from error
    _require(type(parsed) is dict, f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, ValueError) as error:
        raise CampaignSecondaryBridgeError(f"{label} is not finite canonical JSON") from error
    _require(canonical == payload, f"{label} is not canonical JSON")
    assert isinstance(parsed, dict)
    return parsed


def _exact_object(value: object, keys: frozenset[str], *, label: str) -> dict[str, object]:
    _require(type(value) is dict and set(value) == set(keys), f"{label} keys differ")
    assert isinstance(value, dict)
    _require(all(type(key) is str for key in value), f"{label} keys differ")
    return value


def _pinned_object(payload: bytes, expected_sha256: str, *, label: str) -> dict[str, object]:
    _require(
        type(payload) is bytes and 0 < len(payload) <= MAX_CONTRACT_BYTES,
        f"{label} byte bound exceeded",
    )
    _require(
        sha256_bytes(payload) == _sha256(expected_sha256, label=f"expected {label}"),
        f"{label} digest differs",
    )
    return _strict_json_object(payload, label=label)


def _constraint_outcome_names(payload: bytes, expected_sha256: str) -> tuple[str, ...]:
    document = _exact_object(
        _pinned_object(payload, expected_sha256, label="constraint contract"),
        _CONSTRAINT_CONTRACT_KEYS,
        label="constraint contract",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "constraint contract schema differs",
    )
    _require(
        document["artifact"] == "evolutionary_kl_constraint_semantics_v1"
        and document["status"] == "accepted_content_pinned"
        and document["protocol_sha256"] == FROZEN_PROTOCOL_SHA256,
        "constraint contract identity differs",
    )
    raw_rules = document["constraints"]
    _require(type(raw_rules) is list and bool(raw_rules), "constraint rules must be nonempty")
    assert isinstance(raw_rules, list)
    _require(
        len(raw_rules) <= MAX_BRIDGE_CONSTRAINT_COUNT,
        "constraint rule count exceeds bridge cap",
    )
    names: list[str] = []
    for index, raw in enumerate(raw_rules):
        rule = _exact_object(raw, _CONSTRAINT_RULE_KEYS, label=f"constraint rule {index}")
        name = rule["constraint_id"]
        _require(
            type(name) is str and _CONSTRAINT_ID_RE.fullmatch(name) is not None,
            f"constraint rule {index} ID invalid",
        )
        _require(rule["operator"] in {"lt", "le", "gt", "ge"}, f"constraint rule {index} differs")
        threshold = rule["threshold"]
        try:
            finite_threshold = type(threshold) in {int, float} and math.isfinite(float(threshold))
        except OverflowError:
            finite_threshold = False
        _require(finite_threshold, f"constraint rule {index} threshold differs")
        assert isinstance(name, str)
        names.append(name)
    _require(
        tuple(names) == tuple(sorted(set(names))), "constraint names must be sorted and unique"
    )
    return tuple(names)


def _objective_outcome_names(
    protocol: EvolutionaryKLProtocol,
    payload: bytes,
    expected_sha256: str,
    *,
    constraint_contract_sha256: str,
    oracle_contract_sha256: str,
) -> tuple[str, ...]:
    document = _exact_object(
        _pinned_object(payload, expected_sha256, label="truth contract"),
        _TRUTH_CONTRACT_KEYS,
        label="truth contract",
    )
    expected = {
        "artifact": "evolutionary_kl_truth_semantics_contract_v1",
        "complete_requires_atomic_all_finite_uncensored": True,
        "complete_response_status": "complete",
        "constraint_contract_sha256": constraint_contract_sha256,
        "constraint_pass_recomputed_from_raw_value": True,
        "noncomplete_is_ineligible": True,
        "objective_bounds_hex": [value.hex() for value in protocol.objective_bounds],
        "objective_ids": list(protocol.primary_objectives),
        "oracle_contract_sha256": oracle_contract_sha256,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "schema_version": 1,
        "status": "accepted_content_pinned",
    }
    _require(document == expected, "truth contract differs from the frozen truth semantics")
    return tuple(protocol.primary_objectives)


def _training_sequences(payload: bytes, expected_sha256: str) -> frozenset[str]:
    _require(type(payload) is bytes, "training sequence set must be bytes")
    _require(
        0 < len(payload) <= MAX_SEQUENCE_SET_BYTES, "training sequence set byte bound exceeded"
    )
    _require(sha256_bytes(payload) == expected_sha256, "training sequence set digest differs")
    _require(payload.endswith(b"\n"), "training sequence set must end in LF")
    rows: list[tuple[str, str]] = []
    for index, raw in enumerate(payload.splitlines(keepends=True)):
        _require(index < MAX_TRAINING_SEQUENCE_RECORDS, "training sequence row bound exceeded")
        document = _exact_object(
            _strict_json_object(raw, label=f"training sequence row {index}"),
            _SEQUENCE_SET_ROW_KEYS,
            label=f"training sequence row {index}",
        )
        _require(
            document["schema_version"] == 1
            and type(document["schema_version"]) is int
            and document["artifact"] == SEQUENCE_SET_ROW_ARTIFACT,
            f"training sequence row {index} identity differs",
        )
        sequence = document["sequence"]
        _require(type(sequence) is str, f"training sequence row {index} differs")
        assert isinstance(sequence, str)
        try:
            canonical = canonicalize_sequence(
                sequence,
                min_length=1,
                max_length=MAX_AUXILIARY_SEQUENCE_LENGTH,
            )
        except (TypeError, ValueError) as error:
            raise CampaignSecondaryBridgeError(
                f"training sequence row {index} is outside canonical support"
            ) from error
        _require(canonical == sequence, f"training sequence row {index} is not canonical")
        sequence_id = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        _require(
            document["sequence_id"] == sequence_id, f"training sequence row {index} ID differs"
        )
        rows.append((sequence_id, sequence))
    _require(bool(rows), "training sequence set cannot be empty")
    _require(rows == sorted(set(rows)), "training sequence set must be sorted and unique")
    return frozenset(sequence for _sequence_id, sequence in rows)


def _truth_source(authority: CampaignSecondaryBridgeAuthority) -> dict[str, str]:
    return {field: getattr(authority, field) for field in _TRUTH_SOURCE_FIELDS}


def _expected_mapping_document(
    *,
    constraint_contract_sha256: str,
    oracle_contract_sha256: str,
    truth_contract_sha256: str,
    truth_source_identity: dict[str, str],
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
) -> dict[str, object]:
    return {
        "artifact": OUTCOME_MAPPING_ARTIFACT,
        "campaign_non_success_statuses": list(_CAMPAIGN_RESPONSE_STATUSES),
        "campaign_success_status": "succeeded",
        "constraint_contract_sha256": constraint_contract_sha256,
        "constraint_outcome_names": list(constraint_names),
        "execution_authorized": False,
        "non_success_censoring_unknown_is_null": True,
        "non_success_status_mapping": {status: status for status in _CAMPAIGN_RESPONSE_STATUSES},
        "non_success_values_absent": True,
        "objective_outcome_names": list(objective_names),
        "oracle_contract_sha256": oracle_contract_sha256,
        "production_authorized": False,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "schema_version": 1,
        "scientific_claim_authorized": False,
        "secondary_complete_status": "complete",
        "status": "externally_digest_pinned_non_authorizing",
        "success_requires_exact_outcome_inventory": True,
        "success_values_are_atomic": True,
        "success_values_are_uncensored": True,
        "truth_contract_sha256": truth_contract_sha256,
        "truth_source_identity": truth_source_identity,
    }


def campaign_secondary_outcome_mapping_contract_template_bytes(
    protocol: EvolutionaryKLProtocol,
    *,
    constraint_contract_bytes: bytes,
    constraint_contract_sha256: str,
    oracle_contract_sha256: str,
    truth_contract_bytes: bytes,
    truth_contract_sha256: str,
    evaluator_sha256: str,
    checkpoint_sha256: str,
    endpoint_context_sha256: str,
    transform_sha256: str,
) -> bytes:
    """Build a non-authorizing contract template for external review and pinning.

    Producing these bytes does not establish that a real oracle transport obeys
    the asserted atomic/uncensored success semantics.
    """

    for value, label in (
        (constraint_contract_sha256, "constraint contract SHA-256"),
        (oracle_contract_sha256, "oracle contract SHA-256"),
        (truth_contract_sha256, "truth contract SHA-256"),
        (evaluator_sha256, "evaluator SHA-256"),
        (checkpoint_sha256, "checkpoint SHA-256"),
        (endpoint_context_sha256, "endpoint context SHA-256"),
        (transform_sha256, "transform SHA-256"),
    ):
        _sha256(value, label=label)

    constraint_names = _constraint_outcome_names(
        constraint_contract_bytes,
        constraint_contract_sha256,
    )
    objective_names = _objective_outcome_names(
        protocol,
        truth_contract_bytes,
        truth_contract_sha256,
        constraint_contract_sha256=constraint_contract_sha256,
        oracle_contract_sha256=oracle_contract_sha256,
    )
    _require(not set(objective_names) & set(constraint_names), "outcome name roles overlap")
    return canonical_json_bytes(
        _expected_mapping_document(
            constraint_contract_sha256=constraint_contract_sha256,
            oracle_contract_sha256=oracle_contract_sha256,
            truth_contract_sha256=truth_contract_sha256,
            truth_source_identity={
                "oracle_contract_sha256": oracle_contract_sha256,
                "evaluator_sha256": evaluator_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "endpoint_context_sha256": endpoint_context_sha256,
                "transform_sha256": transform_sha256,
            },
            objective_names=objective_names,
            constraint_names=constraint_names,
        )
    )


def _validate_mapping_contract(
    payload: bytes,
    *,
    authority: CampaignSecondaryBridgeAuthority,
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
) -> None:
    document = _exact_object(
        _pinned_object(
            payload,
            authority.outcome_mapping_contract_sha256,
            label="outcome mapping contract",
        ),
        _MAPPING_KEYS,
        label="outcome mapping contract",
    )
    _require(
        document
        == _expected_mapping_document(
            constraint_contract_sha256=authority.constraint_contract_sha256,
            oracle_contract_sha256=authority.oracle_contract_sha256,
            truth_contract_sha256=authority.truth_contract_sha256,
            truth_source_identity=_truth_source(authority),
            objective_names=objective_names,
            constraint_names=constraint_names,
        ),
        "outcome mapping contract differs",
    )


def _validate_protocol_identity(
    protocol: EvolutionaryKLProtocol,
    authority: CampaignSecondaryBridgeAuthority,
) -> None:
    _require(type(protocol) is EvolutionaryKLProtocol, "protocol type differs")
    _require(
        authority.phase == "confirmation"
        and authority.method_id in CONFIRMATION_METHOD_IDS
        and authority.seed in CONFIRMATION_SEEDS,
        "bridge authority is outside the frozen confirmation inventory",
    )
    _require(
        protocol.confirmation_method_ids == CONFIRMATION_METHOD_IDS
        and protocol.confirmation_seeds == CONFIRMATION_SEEDS
        and protocol.total_unique_calls == 512
        and protocol.initial_design_unique_calls == 64
        and protocol.unique_calls_per_batch == 16
        and len(protocol.call_checkpoints) == 29
        and protocol.resource_limits.proposal_attempt_cap == 65_536
        and protocol.scientific_wall_seconds == 7200,
        "protocol bridge domain differs",
    )
    limits = EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS
    _require(
        limits.max_rounds == len(protocol.call_checkpoints) + 1
        and limits.max_cumulative_event_bytes == MAX_BRIDGE_CUMULATIVE_EVENT_BYTES
        and limits.max_events
        == protocol.resource_limits.proposal_attempt_cap + 2 * protocol.total_unique_calls + 1
        and limits.max_proposals == protocol.resource_limits.proposal_attempt_cap
        and limits.max_queries == protocol.total_unique_calls
        and limits.max_responses == protocol.total_unique_calls
        and limits.max_dag_reachability_node_visits == 1_048_576
        and limits.max_dag_reachability_edge_scans == 1_048_576
        and limits.max_append_preflight_inventory_items == 1_048_576
        and limits.max_append_preflight_nested_node_visits == 1_048_576,
        "frozen research replay limits differ",
    )


def _validate_campaign_replay_caps(
    protocol: EvolutionaryKLProtocol,
    campaign: VerifiedCampaign,
    authority: CampaignSecondaryBridgeAuthority,
) -> None:
    """Reject impossible or unsafe campaign shapes before semantic replay.

    The campaign ledger stores every event document in memory.  These bounds
    are therefore intentionally stricter than the per-round on-disk file cap.
    """

    _require(type(campaign) is VerifiedCampaign, "source campaign must use VerifiedCampaign")
    _require(
        type(campaign.round_seals) is tuple
        and type(campaign.round_timing_receipt_sha256s) is tuple
        and len(campaign.round_seals) == len(campaign.round_timing_receipt_sha256s),
        "campaign round inventories differ",
    )
    _require(
        type(campaign.event_documents) is tuple,
        "campaign event-document inventory differs",
    )
    for field in ("proposal_count", "query_count", "response_count"):
        _nonnegative_integer(getattr(campaign, field), label=f"campaign {field}")
    limits = EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS
    for owner, round_count, proposal_count, query_count, response_count, event_count in (
        (
            "authority",
            authority.campaign_round_count,
            authority.campaign_proposal_count,
            authority.campaign_query_count,
            authority.campaign_response_count,
            authority.campaign_event_count,
        ),
        (
            "campaign",
            len(campaign.round_seals),
            campaign.proposal_count,
            campaign.query_count,
            campaign.response_count,
            len(campaign.event_documents),
        ),
    ):
        _require(
            1 <= round_count <= limits.max_rounds,
            f"{owner} round count exceeds bridge cap",
        )
        _require(
            0 <= proposal_count <= limits.max_proposals,
            f"{owner} proposal count exceeds bridge cap",
        )
        _require(
            protocol.initial_design_unique_calls <= query_count <= limits.max_queries,
            f"{owner} query count exceeds bridge domain",
        )
        _require(query_count == response_count, f"{owner} query/response arithmetic differs")
        _require(
            0 < event_count <= limits.max_events,
            f"{owner} event count exceeds bridge cap",
        )
        _require(
            event_count == proposal_count + query_count + response_count + 1,
            f"{owner} terminal event arithmetic differs",
        )
    _require(
        len(campaign.event_documents) == authority.campaign_event_count,
        "campaign event-document count differs from authority",
    )
    cumulative_bytes = 0
    for position, document in enumerate(campaign.event_documents):
        _require(type(document) is bytes, f"campaign event document {position} is not bytes")
        cumulative_bytes += len(document)
        _require(
            cumulative_bytes <= MAX_BRIDGE_CUMULATIVE_EVENT_BYTES,
            "campaign cumulative event bytes exceed bridge cap",
        )


def _validate_campaign_authority(
    campaign: VerifiedCampaign,
    authority: CampaignSecondaryBridgeAuthority,
) -> None:
    _require(type(campaign) is VerifiedCampaign, "source campaign must use VerifiedCampaign")
    _require(campaign.header.campaign_id == authority.run_id, "campaign run ID differs")
    _require(campaign.header.phase == authority.phase, "campaign phase differs")
    _require(campaign.header.configuration_id == authority.method_id, "campaign method differs")
    _require(campaign.header.seed == authority.seed, "campaign seed differs")
    _require(
        campaign.header.protocol_sha256 == FROZEN_PROTOCOL_SHA256,
        "campaign protocol differs",
    )
    expected = {
        "campaign_header_sha256": campaign.header_sha256,
        "campaign_head_seal_sha256": campaign.round_seals[-1],
        "campaign_round_count": len(campaign.round_seals),
        "campaign_round_seal_inventory_sha256": campaign_round_seal_inventory_sha256(
            campaign.round_seals
        ),
        "campaign_round_timing_receipt_inventory_sha256": (
            campaign_round_timing_receipt_inventory_sha256(campaign.round_timing_receipt_sha256s)
        ),
        "campaign_last_event_sha256": campaign.last_event_sha256,
        "campaign_event_count": len(campaign.event_documents),
        "campaign_proposal_count": campaign.proposal_count,
        "campaign_query_count": campaign.query_count,
        "campaign_response_count": campaign.response_count,
    }
    for field, value in expected.items():
        _require(getattr(authority, field) == value, f"campaign authority {field} differs")
    _require(campaign.terminal is True, "bridge requires a terminal sealed campaign")
    _require(not campaign.outstanding_query_ids, "bridge campaign has outstanding responses")
    _require(
        campaign.query_count == campaign.response_count,
        "bridge campaign query/response counts differ",
    )


def _validate_semantic_payload_digests(
    authority: CampaignSecondaryBridgeAuthority,
    *,
    constraint_contract_bytes: bytes,
    oracle_contract_bytes: bytes,
    support_contract_bytes: bytes,
    training_sequence_set_bytes: bytes,
    homology_contract_bytes: bytes,
    reference_contract_bytes: bytes,
    reference_sequence_set_bytes: bytes,
    truth_contract_bytes: bytes,
) -> None:
    payloads = (
        (
            "constraint contract",
            constraint_contract_bytes,
            authority.constraint_contract_sha256,
            MAX_CONTRACT_BYTES,
        ),
        (
            "oracle contract",
            oracle_contract_bytes,
            authority.oracle_contract_sha256,
            MAX_CONTRACT_BYTES,
        ),
        (
            "support contract",
            support_contract_bytes,
            authority.support_contract_sha256,
            MAX_CONTRACT_BYTES,
        ),
        (
            "training sequence set",
            training_sequence_set_bytes,
            authority.training_sequence_set_sha256,
            MAX_SEQUENCE_SET_BYTES,
        ),
        (
            "homology contract",
            homology_contract_bytes,
            authority.homology_contract_sha256,
            MAX_CONTRACT_BYTES,
        ),
        (
            "reference contract",
            reference_contract_bytes,
            authority.reference_contract_sha256,
            MAX_CONTRACT_BYTES,
        ),
        (
            "reference sequence set",
            reference_sequence_set_bytes,
            authority.reference_sequence_set_sha256,
            MAX_SEQUENCE_SET_BYTES,
        ),
        (
            "truth contract",
            truth_contract_bytes,
            authority.truth_contract_sha256,
            MAX_CONTRACT_BYTES,
        ),
    )
    for label, payload, expected, maximum_bytes in payloads:
        _require(
            type(payload) is bytes and 0 < len(payload) <= maximum_bytes,
            f"{label} byte bound exceeded",
        )
        _require(sha256_bytes(payload) == expected, f"{label} digest differs")


def _validate_stop_metadata(
    protocol: EvolutionaryKLProtocol,
    campaign: VerifiedCampaign,
    authority: CampaignSecondaryBridgeAuthority,
    stop: CampaignSecondaryStopMetadata,
) -> None:
    _require(type(stop) is CampaignSecondaryStopMetadata, "stop metadata type differs")
    stop.__post_init__()
    _require(stop.sha256 == authority.stop_metadata_sha256, "stop metadata digest differs")
    _require(
        stop.round_timing_receipt_sha256s == campaign.round_timing_receipt_sha256s,
        "stop timing-receipt inventory differs from the sealed campaign",
    )
    _require(
        len(stop.round_timing_receipt_sha256s) == authority.campaign_round_count,
        "stop timing-receipt count differs",
    )
    _require(
        stop.scientific_elapsed_nanoseconds == campaign.scientific_elapsed_ns,
        "stop elapsed time differs from the sealed campaign",
    )
    row_count = campaign.query_count
    _require(
        stop.sealed_charged_call_count in protocol.call_checkpoints,
        "sealed prefix is not a frozen call checkpoint",
    )
    _require(
        stop.unsealed_discarded_call_count == row_count - stop.sealed_charged_call_count
        and 0 <= stop.unsealed_discarded_call_count <= protocol.unique_calls_per_batch,
        "unsealed discarded-call count differs",
    )
    _require(
        protocol.initial_design_unique_calls <= row_count <= protocol.total_unique_calls,
        "campaign query count lies outside the secondary domain",
    )
    wall_ns = protocol.scientific_wall_seconds * 1_000_000_000
    if stop.stop_reason == "unique_call_budget_reached":
        _require(
            row_count == stop.sealed_charged_call_count == protocol.total_unique_calls
            and stop.scientific_elapsed_nanoseconds <= wall_ns,
            "unique-call stop metadata differs",
        )
    elif stop.stop_reason == "scientific_wall_limit_reached":
        _require(
            stop.sealed_charged_call_count < protocol.total_unique_calls
            and stop.scientific_elapsed_nanoseconds == wall_ns,
            "scientific-wall stop metadata differs",
        )
    else:
        _require(
            stop.sealed_charged_call_count < protocol.total_unique_calls
            and stop.scientific_elapsed_nanoseconds < wall_ns,
            "algorithmic stop metadata differs",
        )


def _sequence_from_query(
    query: ReplayedCampaignQuery,
    protocol: EvolutionaryKLProtocol,
) -> str:
    _require(type(query.sequence_bytes) is bytes, "proposal sequence is not immutable bytes")
    try:
        sequence = query.sequence_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise CampaignSecondaryBridgeError("proposal sequence is not ASCII") from error
    try:
        canonical = canonicalize_sequence(
            sequence,
            min_length=protocol.support_min_length,
            max_length=protocol.support_max_length,
        )
    except (TypeError, ValueError) as error:
        raise CampaignSecondaryBridgeError("submitted proposal lies outside support") from error
    _require(canonical == sequence, "submitted proposal is not canonical")
    return sequence


def _validate_query_identity(
    query: ReplayedCampaignQuery,
    sequence: str,
    authority: CampaignSecondaryBridgeAuthority,
) -> None:
    identity = query.identity
    expected = {
        "canonical_sequence_id": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        **_truth_source(authority),
        "replicate_id": identity.replicate_id,
    }
    _require(identity.document() == expected, f"query {query.call_position} identity differs")


def _raw_query_row(
    query: ReplayedCampaignQuery,
    *,
    sequence: str,
    evidence_identity: dict[str, object],
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
) -> dict[str, object]:
    _require(query.response_status is not None, f"query {query.call_position} has no response")
    status = query.response_status
    if status == "succeeded":
        _require(
            query.evaluation_outcomes is not None,
            f"query {query.call_position} success lacks outcomes",
        )
        outcome_names = tuple(name for name, _value in query.evaluation_outcomes)
        required_names = tuple((*objective_names, *constraint_names))
        _require(
            len(outcome_names) == len(set(outcome_names))
            and set(outcome_names) == set(required_names),
            f"query {query.call_position} outcome inventory differs from pinned contracts",
        )
        outcomes = dict(query.evaluation_outcomes)
        response_status = "complete"
        atomic = True
        objectives: dict[str, float | None] = {name: outcomes[name] for name in objective_names}
        objective_censored: dict[str, bool | None] = {name: False for name in objective_names}
        constraints: dict[str, dict[str, float | bool | None]] = {
            name: {"censored": False, "value": outcomes[name]} for name in constraint_names
        }
    else:
        _require(status in _CAMPAIGN_RESPONSE_STATUSES, "campaign response status differs")
        _require(
            query.evaluation_outcomes is None,
            f"query {query.call_position} non-success carries outcomes",
        )
        response_status = status
        atomic = False
        objectives = {name: None for name in objective_names}
        objective_censored = {name: None for name in objective_names}
        constraints = {name: {"censored": None, "value": None} for name in constraint_names}
    return {
        "artifact": RAW_QUERY_ARTIFACT,
        "atomic_response_complete": atomic,
        "charged_call_position": query.call_position + 1,
        "constraints": constraints,
        "evidence_identity": evidence_identity,
        "objective_censored": objective_censored,
        "objectives": objectives,
        "query_identity": query.identity.document(),
        "response_status": response_status,
        "schema_version": 1,
        "sequence": sequence,
    }


def _append_bounded_raw_jsonl_line(
    chunks: list[bytes],
    document: dict[str, object],
    cumulative_bytes: int,
) -> int:
    try:
        encoded = canonical_json_bytes(document)
    except (TypeError, ValueError) as error:
        raise CampaignSecondaryBridgeError("raw evidence row is not canonical JSON") from error
    _require(
        encoded.endswith(b"\n") and not encoded.endswith(b"\n\n"),
        "canonical raw evidence line framing differs",
    )
    updated = cumulative_bytes + len(encoded)
    _require(updated <= MAX_RAW_EVIDENCE_BYTES, "raw evidence byte bound exceeded incrementally")
    chunks.append(encoded)
    return updated


def _evidence_identity(authority: CampaignSecondaryBridgeAuthority) -> dict[str, object]:
    return {
        "constraint_contract_sha256": authority.constraint_contract_sha256,
        "homology_contract_sha256": authority.homology_contract_sha256,
        "method_id": authority.method_id,
        "oracle_contract_sha256": authority.oracle_contract_sha256,
        "phase": authority.phase,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "reference_contract_sha256": authority.reference_contract_sha256,
        "reference_sequence_set_sha256": authority.reference_sequence_set_sha256,
        "run_id": authority.run_id,
        "seed": authority.seed,
        "support_contract_sha256": authority.support_contract_sha256,
        "training_sequence_set_sha256": authority.training_sequence_set_sha256,
        "truth_contract_sha256": authority.truth_contract_sha256,
    }


def _inventory_sha256(domain: bytes, rows: list[dict[str, object]]) -> str:
    return sha256_bytes(domain + canonical_jsonl_bytes(rows))


def campaign_round_seal_inventory_sha256(round_seals: tuple[str, ...]) -> str:
    """Hash one ordered seal inventory without granting it external authority."""

    _require(
        type(round_seals) is tuple
        and bool(round_seals)
        and len(round_seals) <= EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS.max_rounds,
        "round seal inventory differs",
    )
    rows: list[dict[str, object]] = []
    for index, digest in enumerate(round_seals):
        rows.append(
            {
                "round_index": index,
                "round_seal_sha256": _sha256(digest, label=f"round seal {index}"),
            }
        )
    return _inventory_sha256(ROUND_SEAL_INVENTORY_HASH_DOMAIN, rows)


def campaign_round_timing_receipt_inventory_sha256(
    round_timing_receipt_sha256s: tuple[str, ...],
) -> str:
    """Hash one ordered timing inventory without authenticating its issuers."""

    _require(
        type(round_timing_receipt_sha256s) is tuple
        and bool(round_timing_receipt_sha256s)
        and len(round_timing_receipt_sha256s) <= EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS.max_rounds,
        "round timing-receipt inventory differs",
    )
    rows: list[dict[str, object]] = []
    for index, digest in enumerate(round_timing_receipt_sha256s):
        rows.append(
            {
                "round_index": index,
                "timing_receipt_sha256": _sha256(
                    digest,
                    label=f"round timing receipt {index}",
                ),
            }
        )
    return _inventory_sha256(ROUND_TIMING_INVENTORY_HASH_DOMAIN, rows)


def _require_distinct_digest_roles(bindings: list[tuple[str, str]]) -> None:
    """Permit repeated bytes only when they name the exact same logical role."""

    digest_roles: dict[str, str] = {}
    for role, raw_digest in bindings:
        digest = _sha256(raw_digest, label=f"{role} digest")
        prior_role = digest_roles.setdefault(digest, role)
        _require(prior_role == role, f"digest roles alias: {prior_role} and {role}")


def _authority_digest_role(field: str, *, round_count: int) -> str:
    if field == "campaign_header_sha256":
        return "campaign.header"
    if field == "campaign_head_seal_sha256":
        return f"campaign.round_seal.{round_count - 1}"
    if field == "campaign_round_seal_inventory_sha256":
        return "campaign.round_seal_inventory"
    if field == "campaign_round_timing_receipt_inventory_sha256":
        return "campaign.round_timing_inventory"
    if field == "campaign_last_event_sha256":
        return "campaign.last_event"
    if field == "stop_metadata_sha256":
        return "stop.metadata"
    return f"authority.{field}"


def _source_digest_bindings(
    authority: CampaignSecondaryBridgeAuthority,
    stop: CampaignSecondaryStopMetadata,
    *,
    round_seals: tuple[str, ...],
    round_timings: tuple[str, ...],
) -> list[tuple[str, str]]:
    bindings = [
        (
            _authority_digest_role(field, round_count=authority.campaign_round_count),
            getattr(authority, field),
        )
        for field in _AUTHORITY_DIGEST_FIELDS
    ]
    bindings.extend(
        [
            ("campaign.header", authority.campaign_header_sha256),
            ("campaign.last_event", authority.campaign_last_event_sha256),
            (
                "campaign.round_seal_inventory",
                campaign_round_seal_inventory_sha256(round_seals),
            ),
            (
                "campaign.round_timing_inventory",
                campaign_round_timing_receipt_inventory_sha256(round_timings),
            ),
            ("stop.metadata", stop.sha256),
            ("stop.sealed_prefix_receipt", stop.sealed_prefix_receipt_sha256),
            ("stop.stop_receipt", stop.stop_receipt_sha256),
            ("stop.timing_authority_receipt", stop.timing_authority_receipt_sha256),
        ]
    )
    bindings.extend(
        (f"campaign.round_seal.{index}", digest) for index, digest in enumerate(round_seals)
    )
    bindings.extend(
        (f"campaign.round_timing.{index}", digest) for index, digest in enumerate(round_timings)
    )
    bindings.extend(
        (f"campaign.round_timing.{index}", digest)
        for index, digest in enumerate(stop.round_timing_receipt_sha256s)
    )
    return bindings


def _validate_source_digest_roles(
    authority: CampaignSecondaryBridgeAuthority,
    stop: CampaignSecondaryStopMetadata,
    campaign: VerifiedCampaign,
) -> None:
    _require(type(stop) is CampaignSecondaryStopMetadata, "stop metadata type differs")
    stop.__post_init__()
    _require_distinct_digest_roles(
        _source_digest_bindings(
            authority,
            stop,
            round_seals=campaign.round_seals,
            round_timings=campaign.round_timing_receipt_sha256s,
        )
    )


def _validate_receipt_digest_roles(
    authority: CampaignSecondaryBridgeAuthority,
    stop: CampaignSecondaryStopMetadata,
    *,
    campaign: dict[str, object],
    secondary: dict[str, object],
    bridge_receipt_sha256: str,
) -> None:
    round_seals = campaign["round_seals"]
    round_timings = campaign["round_timing_receipt_sha256s"]
    _require(
        type(round_seals) is list and all(type(value) is str for value in round_seals),
        "bridge receipt round-seal inventory differs",
    )
    _require(
        type(round_timings) is list and all(type(value) is str for value in round_timings),
        "bridge receipt round-timing inventory differs",
    )
    seal_tuple = tuple(round_seals)
    timing_tuple = tuple(round_timings)
    bindings = _source_digest_bindings(
        authority,
        stop,
        round_seals=seal_tuple,  # type: ignore[arg-type]
        round_timings=timing_tuple,  # type: ignore[arg-type]
    )
    bindings.extend(
        [
            ("campaign.header", campaign["header_sha256"]),
            ("campaign.last_event", campaign["last_event_sha256"]),
            (
                f"campaign.round_seal.{authority.campaign_round_count - 1}",
                campaign["head_seal_sha256"],
            ),
            ("campaign.round_seal_inventory", campaign["round_seal_inventory_sha256"]),
            (
                "campaign.round_timing_inventory",
                campaign["round_timing_receipt_inventory_sha256"],
            ),
            ("campaign.event_document_inventory", campaign["event_document_inventory_sha256"]),
            (
                "campaign.proposal_sequence_inventory",
                campaign["proposal_sequence_inventory_sha256"],
            ),
            ("campaign.source_query_inventory", campaign["source_query_inventory_sha256"]),
            ("secondary.raw_evidence", secondary["raw_evidence_sha256"]),
            (
                "secondary.query_identity_inventory",
                secondary["query_identity_inventory_sha256"],
            ),
            ("secondary.evidence", secondary["secondary_evidence_sha256"]),
            ("bridge.receipt", bridge_receipt_sha256),
        ]
    )
    _require_distinct_digest_roles(bindings)  # type: ignore[arg-type]


def _source_query_inventory_rows(
    queries: tuple[ReplayedCampaignQuery, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for expected_position, query in enumerate(queries):
        _require(query.call_position == expected_position, "source query order differs")
        _require(
            query.response_event_position is not None
            and query.response_id is not None
            and query.response_status is not None,
            f"source query {expected_position} is unmatched",
        )
        rows.append(
            {
                "batch_id": query.batch_id,
                "batch_position": query.batch_position,
                "call_position": query.call_position,
                "evaluator_version": query.evaluator_version,
                "fidelity": query.fidelity,
                "identity_key": query.identity.key,
                "planned_cost_hex": query.planned_cost.hex(),
                "proposal_id": query.proposal_id,
                "query_event_position": query.event_position,
                "query_id": query.query_id,
                "response_event_position": query.response_event_position,
                "response_id": query.response_id,
                "response_status": query.response_status,
                "sequence_ascii_sha256": sha256_bytes(query.sequence_bytes),
            }
        )
    return rows


def _receipt_document(
    *,
    campaign: VerifiedCampaign,
    authority: CampaignSecondaryBridgeAuthority,
    stop: CampaignSecondaryStopMetadata,
    event_inventory_sha256: str,
    proposal_sequence_inventory_sha256: str,
    source_query_inventory_sha256: str,
    raw_evidence_sha256: str,
    query_inventory_sha256: str,
    secondary_evidence_sha256: str,
) -> dict[str, object]:
    return {
        "artifact": BRIDGE_ARTIFACT,
        "authorization": {
            "execution_authorized": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
        "campaign": {
            "event_count": authority.campaign_event_count,
            "event_document_inventory_sha256": event_inventory_sha256,
            "header_sha256": authority.campaign_header_sha256,
            "head_seal_sha256": authority.campaign_head_seal_sha256,
            "last_event_sha256": authority.campaign_last_event_sha256,
            "proposal_count": authority.campaign_proposal_count,
            "proposal_sequence_inventory_sha256": proposal_sequence_inventory_sha256,
            "query_count": authority.campaign_query_count,
            "response_count": authority.campaign_response_count,
            "round_count": authority.campaign_round_count,
            "round_seal_inventory_sha256": campaign_round_seal_inventory_sha256(
                campaign.round_seals
            ),
            "round_seals": list(campaign.round_seals),
            "round_timing_receipt_inventory_sha256": (
                campaign_round_timing_receipt_inventory_sha256(
                    campaign.round_timing_receipt_sha256s
                )
            ),
            "round_timing_receipt_sha256s": list(campaign.round_timing_receipt_sha256s),
            "source_query_inventory_sha256": source_query_inventory_sha256,
            "terminal": True,
        },
        "known_unproven_authorities": list(_KNOWN_UNPROVEN_AUTHORITIES),
        "mapping": {
            "campaign_call_position_origin": 0,
            "campaign_success_status": "succeeded",
            "outcome_mapping_contract_sha256": authority.outcome_mapping_contract_sha256,
            "raw_charged_call_position_origin": 1,
            "secondary_complete_status": "complete",
        },
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "schema_version": 1,
        "secondary": {
            "query_identity_inventory_sha256": query_inventory_sha256,
            "raw_evidence_sha256": raw_evidence_sha256,
            "secondary_evidence_sha256": secondary_evidence_sha256,
        },
        "source_authority": authority.document(),
        "status": "structural_non_authorizing_non_acceptance_envelope",
        "stop_metadata": stop.document(),
        "stop_metadata_sha256": stop.sha256,
    }


def _construct_campaign_secondary_bridge(
    protocol: EvolutionaryKLProtocol,
    campaign: VerifiedCampaign,
    *,
    authority: CampaignSecondaryBridgeAuthority,
    stop_metadata: CampaignSecondaryStopMetadata,
    outcome_mapping_contract_bytes: bytes,
    constraint_contract_bytes: bytes,
    oracle_contract_bytes: bytes,
    support_contract_bytes: bytes,
    training_sequence_set_bytes: bytes,
    homology_contract_bytes: bytes,
    reference_contract_bytes: bytes,
    reference_sequence_set_bytes: bytes,
    truth_contract_bytes: bytes,
) -> tuple[CampaignSecondaryStructuralEnvelope, SecondaryYieldEvidence]:
    _require(
        type(authority) is CampaignSecondaryBridgeAuthority,
        "bridge authority type differs",
    )
    authority.__post_init__()
    _validate_protocol_identity(protocol, authority)
    _validate_campaign_replay_caps(protocol, campaign, authority)
    _validate_campaign_authority(campaign, authority)
    _validate_source_digest_roles(authority, stop_metadata, campaign)
    _validate_semantic_payload_digests(
        authority,
        constraint_contract_bytes=constraint_contract_bytes,
        oracle_contract_bytes=oracle_contract_bytes,
        support_contract_bytes=support_contract_bytes,
        training_sequence_set_bytes=training_sequence_set_bytes,
        homology_contract_bytes=homology_contract_bytes,
        reference_contract_bytes=reference_contract_bytes,
        reference_sequence_set_bytes=reference_sequence_set_bytes,
        truth_contract_bytes=truth_contract_bytes,
    )
    _validate_stop_metadata(protocol, campaign, authority, stop_metadata)
    constraint_names = _constraint_outcome_names(
        constraint_contract_bytes,
        authority.constraint_contract_sha256,
    )
    objective_names = _objective_outcome_names(
        protocol,
        truth_contract_bytes,
        authority.truth_contract_sha256,
        constraint_contract_sha256=authority.constraint_contract_sha256,
        oracle_contract_sha256=authority.oracle_contract_sha256,
    )
    _require(not set(objective_names) & set(constraint_names), "outcome name roles overlap")
    _validate_mapping_contract(
        outcome_mapping_contract_bytes,
        authority=authority,
        objective_names=objective_names,
        constraint_names=constraint_names,
    )
    training_sequences = _training_sequences(
        training_sequence_set_bytes,
        authority.training_sequence_set_sha256,
    )
    replay = replay_verified_campaign_event_documents(
        campaign,
        replay_limits=EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    )
    _require(len(replay.queries) == campaign.query_count, "replayed query count differs")
    source_query_rows = _source_query_inventory_rows(replay.queries)

    evidence_identity = _evidence_identity(authority)
    identities = tuple(query.identity for query in replay.queries)
    query_inventory = query_identity_inventory_sha256(identities)
    raw_header = {
        "artifact": RAW_HEADER_ARTIFACT,
        **evidence_identity,
        "execution_authorized": False,
        "production_authorized": False,
        "query_identity_fields": list(QUERY_IDENTITY_FIELDS),
        "query_identity_inventory_sha256": query_inventory,
        "row_count": len(replay.queries),
        "schema_version": 1,
        "scientific_claim_authorized": False,
        "scientific_elapsed_nanoseconds": stop_metadata.scientific_elapsed_nanoseconds,
        "sealed_charged_call_count": stop_metadata.sealed_charged_call_count,
        "status": "controller_digest_pinned_raw_query_evidence",
        "stop_reason": stop_metadata.stop_reason,
        "truth_source_identity": _truth_source(authority),
        "unsealed_discarded_call_count": stop_metadata.unsealed_discarded_call_count,
    }
    raw_chunks: list[bytes] = []
    raw_size = _append_bounded_raw_jsonl_line(raw_chunks, raw_header, 0)
    for query in replay.queries:
        sequence = _sequence_from_query(query, protocol)
        _require(
            sequence not in training_sequences,
            f"query {query.call_position} submitted an exact training-overlap sequence",
        )
        _validate_query_identity(query, sequence, authority)
        raw_size = _append_bounded_raw_jsonl_line(
            raw_chunks,
            _raw_query_row(
                query,
                sequence=sequence,
                evidence_identity=evidence_identity,
                objective_names=objective_names,
                constraint_names=constraint_names,
            ),
            raw_size,
        )
    raw_evidence = b"".join(raw_chunks)
    trust_anchors = SecondaryEvidenceTrustAnchors(
        run_id=authority.run_id,
        phase=authority.phase,
        method_id=authority.method_id,
        seed=authority.seed,
        raw_evidence_sha256=sha256_bytes(raw_evidence),
        query_identity_inventory_sha256=query_inventory,
        constraint_contract_sha256=authority.constraint_contract_sha256,
        oracle_contract_sha256=authority.oracle_contract_sha256,
        support_contract_sha256=authority.support_contract_sha256,
        training_sequence_set_sha256=authority.training_sequence_set_sha256,
        homology_contract_sha256=authority.homology_contract_sha256,
        reference_contract_sha256=authority.reference_contract_sha256,
        reference_sequence_set_sha256=authority.reference_sequence_set_sha256,
        truth_contract_sha256=authority.truth_contract_sha256,
        evaluator_sha256=authority.evaluator_sha256,
        checkpoint_sha256=authority.checkpoint_sha256,
        endpoint_context_sha256=authority.endpoint_context_sha256,
        transform_sha256=authority.transform_sha256,
    )
    try:
        evidence = compute_authenticated_secondary_yield(
            protocol,
            raw_evidence,
            trust_anchors=trust_anchors,
            constraint_contract_bytes=constraint_contract_bytes,
            oracle_contract_bytes=oracle_contract_bytes,
            support_contract_bytes=support_contract_bytes,
            training_sequence_set_bytes=training_sequence_set_bytes,
            homology_contract_bytes=homology_contract_bytes,
            reference_contract_bytes=reference_contract_bytes,
            reference_sequence_set_bytes=reference_sequence_set_bytes,
            truth_contract_bytes=truth_contract_bytes,
        )
    except SecondaryEvidenceError as error:
        raise CampaignSecondaryBridgeError(
            "secondary replay rejected bridged raw evidence"
        ) from error
    source_query_inventory = _inventory_sha256(
        SOURCE_QUERY_INVENTORY_HASH_DOMAIN,
        source_query_rows,
    )
    unsigned_receipt = _receipt_document(
        campaign=campaign,
        authority=authority,
        stop=stop_metadata,
        event_inventory_sha256=replay.event_document_inventory_sha256,
        proposal_sequence_inventory_sha256=replay.proposal_sequence_inventory_sha256,
        source_query_inventory_sha256=source_query_inventory,
        raw_evidence_sha256=trust_anchors.raw_evidence_sha256,
        query_inventory_sha256=query_inventory,
        secondary_evidence_sha256=evidence.evidence_sha256,
    )
    receipt_sha256 = sha256_bytes(
        BRIDGE_RECEIPT_HASH_DOMAIN + canonical_json_bytes(unsigned_receipt)
    )
    receipt = canonical_json_bytes({**unsigned_receipt, "bridge_receipt_sha256": receipt_sha256})
    return (
        CampaignSecondaryStructuralEnvelope(
            raw_evidence_bytes=raw_evidence,
            receipt_bytes=receipt,
            trust_anchors=trust_anchors,
            bridge_receipt_sha256=receipt_sha256,
        ),
        evidence,
    )


def build_campaign_secondary_structural_envelope(
    protocol: EvolutionaryKLProtocol,
    campaign: VerifiedCampaign,
    *,
    authority: CampaignSecondaryBridgeAuthority,
    stop_metadata: CampaignSecondaryStopMetadata,
    outcome_mapping_contract_bytes: bytes,
    constraint_contract_bytes: bytes,
    oracle_contract_bytes: bytes,
    support_contract_bytes: bytes,
    training_sequence_set_bytes: bytes,
    homology_contract_bytes: bytes,
    reference_contract_bytes: bytes,
    reference_sequence_set_bytes: bytes,
    truth_contract_bytes: bytes,
) -> CampaignSecondaryStructuralEnvelope:
    """Build an unverified, non-acceptance envelope from a campaign object.

    A ``VerifiedCampaign`` instance can be constructed or replaced directly in
    Python.  This helper therefore provides structural diagnostics only.  It
    must never be used as evidence that the source path was authenticated.
    """

    bridge, _evidence = _construct_campaign_secondary_bridge(
        protocol,
        campaign,
        authority=authority,
        stop_metadata=stop_metadata,
        outcome_mapping_contract_bytes=outcome_mapping_contract_bytes,
        constraint_contract_bytes=constraint_contract_bytes,
        oracle_contract_bytes=oracle_contract_bytes,
        support_contract_bytes=support_contract_bytes,
        training_sequence_set_bytes=training_sequence_set_bytes,
        homology_contract_bytes=homology_contract_bytes,
        reference_contract_bytes=reference_contract_bytes,
        reference_sequence_set_bytes=reference_sequence_set_bytes,
        truth_contract_bytes=truth_contract_bytes,
    )
    return bridge


def _reopen_campaign_for_bridge(
    protocol: EvolutionaryKLProtocol,
    root: str | Path,
    *,
    trusted_parent: str | Path,
    authority: CampaignSecondaryBridgeAuthority,
) -> VerifiedCampaign:
    _require(
        type(authority) is CampaignSecondaryBridgeAuthority,
        "bridge authority type differs",
    )
    authority.__post_init__()
    _validate_protocol_identity(protocol, authority)
    return verify_campaign(
        root,
        trusted_parent=trusted_parent,
        expected_header_sha256=authority.campaign_header_sha256,
        expected_head_seal_sha256=authority.campaign_head_seal_sha256,
        expected_round_count=authority.campaign_round_count,
        replay_limits=EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
    )


def build_campaign_secondary_structural_envelope_from_path(
    protocol: EvolutionaryKLProtocol,
    root: str | Path,
    *,
    trusted_parent: str | Path,
    authority: CampaignSecondaryBridgeAuthority,
    stop_metadata: CampaignSecondaryStopMetadata,
    outcome_mapping_contract_bytes: bytes,
    constraint_contract_bytes: bytes,
    oracle_contract_bytes: bytes,
    support_contract_bytes: bytes,
    training_sequence_set_bytes: bytes,
    homology_contract_bytes: bytes,
    reference_contract_bytes: bytes,
    reference_sequence_set_bytes: bytes,
    truth_contract_bytes: bytes,
) -> CampaignSecondaryStructuralEnvelope:
    """Reopen a sealed path, then emit a still-non-acceptance envelope."""

    campaign = _reopen_campaign_for_bridge(
        protocol,
        root,
        trusted_parent=trusted_parent,
        authority=authority,
    )
    return build_campaign_secondary_structural_envelope(
        protocol,
        campaign,
        authority=authority,
        stop_metadata=stop_metadata,
        outcome_mapping_contract_bytes=outcome_mapping_contract_bytes,
        constraint_contract_bytes=constraint_contract_bytes,
        oracle_contract_bytes=oracle_contract_bytes,
        support_contract_bytes=support_contract_bytes,
        training_sequence_set_bytes=training_sequence_set_bytes,
        homology_contract_bytes=homology_contract_bytes,
        reference_contract_bytes=reference_contract_bytes,
        reference_sequence_set_bytes=reference_sequence_set_bytes,
        truth_contract_bytes=truth_contract_bytes,
    )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedCampaignSecondaryReplay:
    """Result obtainable only after reopening and fully replaying a source path.

    ``Verified`` here is deliberately narrow: the repository verifier reopened
    the sealed path and rebuilt equal bytes.  It does not authenticate external
    timing, oracle semantics, chemical form, or scientific eligibility.
    """

    envelope: CampaignSecondaryStructuralEnvelope
    evidence: SecondaryYieldEvidence
    campaign_header_sha256: str
    campaign_head_seal_sha256: str
    campaign_round_count: int
    campaign_round_seal_inventory_sha256: str
    campaign_round_timing_receipt_inventory_sha256: str
    source_path_reopened: Literal[True]
    execution_authorized: Literal[False]
    scientific_claim_authorized: Literal[False]
    production_authorized: Literal[False]

    def __new__(cls, *_args: object, **_kwargs: object) -> VerifiedCampaignSecondaryReplay:
        raise TypeError("use replay_campaign_secondary_bridge_from_path")


def _verified_replay_result(
    *,
    envelope: CampaignSecondaryStructuralEnvelope,
    evidence: SecondaryYieldEvidence,
    authority: CampaignSecondaryBridgeAuthority,
) -> VerifiedCampaignSecondaryReplay:
    result = object.__new__(VerifiedCampaignSecondaryReplay)
    object.__setattr__(result, "envelope", envelope)
    object.__setattr__(result, "evidence", evidence)
    object.__setattr__(result, "campaign_header_sha256", authority.campaign_header_sha256)
    object.__setattr__(result, "campaign_head_seal_sha256", authority.campaign_head_seal_sha256)
    object.__setattr__(result, "campaign_round_count", authority.campaign_round_count)
    object.__setattr__(
        result,
        "campaign_round_seal_inventory_sha256",
        authority.campaign_round_seal_inventory_sha256,
    )
    object.__setattr__(
        result,
        "campaign_round_timing_receipt_inventory_sha256",
        authority.campaign_round_timing_receipt_inventory_sha256,
    )
    object.__setattr__(result, "source_path_reopened", True)
    object.__setattr__(result, "execution_authorized", False)
    object.__setattr__(result, "scientific_claim_authorized", False)
    object.__setattr__(result, "production_authorized", False)
    return result


def replay_campaign_secondary_bridge_from_path(
    protocol: EvolutionaryKLProtocol,
    root: str | Path,
    envelope: CampaignSecondaryStructuralEnvelope,
    *,
    trusted_parent: str | Path,
    expected_bridge_receipt_sha256: str,
    authority: CampaignSecondaryBridgeAuthority,
    stop_metadata: CampaignSecondaryStopMetadata,
    outcome_mapping_contract_bytes: bytes,
    constraint_contract_bytes: bytes,
    oracle_contract_bytes: bytes,
    support_contract_bytes: bytes,
    training_sequence_set_bytes: bytes,
    homology_contract_bytes: bytes,
    reference_contract_bytes: bytes,
    reference_sequence_set_bytes: bytes,
    truth_contract_bytes: bytes,
) -> VerifiedCampaignSecondaryReplay:
    """Reopen, authenticate, rebuild, and compare one structural envelope.

    ``expected_bridge_receipt_sha256`` must be supplied from a separately
    persisted controller/auditor channel.  Copying it from ``envelope`` does
    not establish independent authentication or resolve the named blockers.
    """

    _require(
        type(envelope) is CampaignSecondaryStructuralEnvelope,
        "bridge envelope type differs",
    )
    expected_digest = _sha256(
        expected_bridge_receipt_sha256,
        label="expected bridge receipt SHA-256",
    )
    _require(
        envelope.bridge_receipt_sha256 == expected_digest,
        "bridge receipt anchor differs",
    )
    campaign = _reopen_campaign_for_bridge(
        protocol,
        root,
        trusted_parent=trusted_parent,
        authority=authority,
    )
    rebuilt, evidence = _construct_campaign_secondary_bridge(
        protocol,
        campaign,
        authority=authority,
        stop_metadata=stop_metadata,
        outcome_mapping_contract_bytes=outcome_mapping_contract_bytes,
        constraint_contract_bytes=constraint_contract_bytes,
        oracle_contract_bytes=oracle_contract_bytes,
        support_contract_bytes=support_contract_bytes,
        training_sequence_set_bytes=training_sequence_set_bytes,
        homology_contract_bytes=homology_contract_bytes,
        reference_contract_bytes=reference_contract_bytes,
        reference_sequence_set_bytes=reference_sequence_set_bytes,
        truth_contract_bytes=truth_contract_bytes,
    )
    _require(rebuilt.bridge_receipt_sha256 == expected_digest, "rebuilt bridge receipt differs")
    _require(rebuilt == envelope, "bridge bytes or trust anchors differ from replay")
    return _verified_replay_result(
        envelope=rebuilt,
        evidence=evidence,
        authority=authority,
    )


__all__ = [
    "BRIDGE_ARTIFACT",
    "MAX_BRIDGE_CONSTRAINT_COUNT",
    "MAX_BRIDGE_CUMULATIVE_EVENT_BYTES",
    "MAX_BRIDGE_SCIENTIFIC_ELAPSED_NANOSECONDS",
    "MAX_BRIDGE_STOP_CALL_COUNT",
    "OUTCOME_MAPPING_ARTIFACT",
    "STOP_METADATA_ARTIFACT",
    "CampaignSecondaryBridgeAuthority",
    "CampaignSecondaryBridgeError",
    "CampaignSecondaryStopMetadata",
    "CampaignSecondaryStructuralEnvelope",
    "VerifiedCampaignSecondaryReplay",
    "build_campaign_secondary_structural_envelope",
    "build_campaign_secondary_structural_envelope_from_path",
    "campaign_round_seal_inventory_sha256",
    "campaign_round_timing_receipt_inventory_sha256",
    "campaign_secondary_outcome_mapping_contract_template_bytes",
    "replay_campaign_secondary_bridge_from_path",
]

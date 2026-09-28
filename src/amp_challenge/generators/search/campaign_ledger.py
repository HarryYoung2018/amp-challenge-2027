"""Crash-safe, append-only campaign rounds for peptide search.

This module is deliberately an engineering boundary.  It serializes existing
search records, but it never calls an oracle and its sealed headers explicitly
deny scientific or production authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from amp_challenge.evaluation.evolutionary_kl_protocol import (
    CONFIRMATION_METHOD_IDS,
    CONFIRMATION_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    SCREEN_CONFIGURATION_IDS,
    SCREEN_SEEDS,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseBuilder,
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    checksum_manifest_bytes,
    relocate_phase_capability_noreplace_at,
    sha256_bytes,
    validate_relative_path,
)
from amp_challenge.generators.search.batching import (
    BatchExecutionPlan,
    BatchLimits,
    batch_execution_plan_from_mapping,
)
from amp_challenge.generators.search.ledger import TranspositionEvent
from amp_challenge.generators.search.records import (
    EdgeRecord,
    EvaluationRecord,
    ProbabilityFactor,
    ProbabilityTrace,
    ProposalRecord,
    SelectionDecision,
)

MAX_UNIQUE_ORACLE_CALLS = 512
MAX_SCIENTIFIC_ELAPSED_NS = 7_200_000_000_000
MAX_CAMPAIGN_EVENT_IDENTIFIER_LENGTH = 128
MAX_CAMPAIGN_SEQUENCE_LENGTH = 256
MAX_CAMPAIGN_EVALUATION_OUTCOMES = 66
MAX_CAMPAIGN_CHEAP_PREDICTIONS = 66
MAX_CAMPAIGN_PROBABILITY_FACTORS = 64
MAX_CAMPAIGN_SAMPLING_PARAMETERS = 64
MAX_CAMPAIGN_EDIT_ITEMS = 256
MAX_CAMPAIGN_ELIGIBLE_PROPOSAL_IDS = 65_536
MAX_CAMPAIGN_NESTED_VALUE_NODES = 131_072
MAX_CAMPAIGN_NESTED_VALUE_DEPTH = 16
MAX_CAMPAIGN_APPEND_PREFLIGHT_INVENTORY_ITEMS = 1_048_576
MAX_CAMPAIGN_APPEND_PREFLIGHT_NESTED_NODE_VISITS = 1_048_576
CAMPAIGN_OPERATOR_PARENT_CARDINALITIES = (
    ("crossover", 2),
    ("deletion", 1),
    ("initial_design", 0),
    ("insertion", 1),
    ("partial_remask", 1),
    ("seed", 0),
    ("substitution", 1),
    ("two_parent_crossover", 2),
)
CAMPAIGN_ARTIFACT = "evolutionary_kl_peptide_search_campaign_v1"
ROUND_ARTIFACT = "evolutionary_kl_peptide_search_campaign_round_v1"
EVENT_HASH_DOMAIN = b"amp/evolutionary-kl/campaign-event/v1\0"
GENESIS_HASH_DOMAIN = b"amp/evolutionary-kl/campaign-genesis/v1\0"
EVENT_DOCUMENT_INVENTORY_DOMAIN = b"amp/evolutionary-kl/campaign-event-documents/v1\0"
PROPOSAL_SEQUENCE_INVENTORY_DOMAIN = b"amp/evolutionary-kl/campaign-proposal-sequences/v1\0"
ABSTENTION_IDENTITY_KEY = hashlib.sha256(b"amp/evolutionary-kl/terminal-abstention/v1").hexdigest()
QUERY_IDENTITY_FIELDS = (
    "canonical_sequence_id",
    "oracle_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
    "replicate_id",
)
_ROUND_NAME = re.compile(r"round-([0-9]{6})\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROUND_FILES = ("SHA256SUMS", "campaign.json", "events.jsonl", "receipt.json", "round.json")
_ROUND_FILE_BOUNDS = {
    "SHA256SUMS": 4096,
    "campaign.json": 65536,
    "events.jsonl": 256 * 1024 * 1024,
    "receipt.json": 262144,
    "round.json": 65536,
}
_EXPECTED_CLUSTER_BATCHING = {
    "profile": "cluster_batch_first_v1",
    "rollout_batch_size": 128,
    "proposal_batch_size": 65536,
    "surrogate_batch_size": 8192,
    "kg_candidate_chunk_size": 256,
    "kg_fantasy_chunk_size": 512,
    "kg_max_joint_size": 16,
    "kg_max_combinations": 65536,
    "oracle_batch_size": 32,
    "replay_max_sequences": 1024,
    "replay_max_tokens": 32768,
    "replay_length_bucket_boundaries": [8, 16, 24, 32, 40, 50],
    "gradient_accumulation_steps": 8,
}
_REPLAY_LIMIT_FIELDS = (
    "max_rounds",
    "max_cumulative_event_bytes",
    "max_events",
    "max_proposals",
    "max_queries",
    "max_responses",
    "max_dag_reachability_node_visits",
    "max_dag_reachability_edge_scans",
    "max_append_preflight_inventory_items",
    "max_append_preflight_nested_node_visits",
)


class CampaignLedgerError(ValueError):
    """Raised when a campaign tree or requested append fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignLedgerError(message)


def _require_exact_mapping_keys(
    value: object,
    expected: Collection[str],
    *,
    label: str,
) -> dict[str, object]:
    _require(type(value) is dict, label)
    assert isinstance(value, dict)
    _require(
        len(value) == len(expected) and all(key in value for key in expected),
        label,
    )
    return value


def _identifier(value: object, *, label: str) -> str:
    _require(type(value) is str and _ID.fullmatch(value) is not None, f"{label} is invalid")
    assert isinstance(value, str)
    return value


def _sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a non-negative integer")
    assert isinstance(value, int)
    return value


def _finite(value: object, *, label: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    assert isinstance(value, float)
    return value


@dataclass(frozen=True, slots=True)
class CampaignReplayLimits:
    """Fail-closed ceilings applied while replaying or appending sealed rounds."""

    max_rounds: int
    max_cumulative_event_bytes: int
    max_events: int
    max_proposals: int
    max_queries: int
    max_responses: int
    max_dag_reachability_node_visits: int
    max_dag_reachability_edge_scans: int
    max_append_preflight_inventory_items: int
    max_append_preflight_nested_node_visits: int

    def __post_init__(self) -> None:
        for field in _REPLAY_LIMIT_FIELDS:
            value = _nonnegative_integer(getattr(self, field), label=f"replay limit {field}")
            _require(value > 0, f"replay limit {field} must be positive")


EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS = CampaignReplayLimits(
    max_rounds=30,
    max_cumulative_event_bytes=128 * 1024 * 1024,
    max_events=66_561,
    max_proposals=65_536,
    max_queries=512,
    max_responses=512,
    max_dag_reachability_node_visits=1_048_576,
    max_dag_reachability_edge_scans=1_048_576,
    max_append_preflight_inventory_items=MAX_CAMPAIGN_APPEND_PREFLIGHT_INVENTORY_ITEMS,
    max_append_preflight_nested_node_visits=(MAX_CAMPAIGN_APPEND_PREFLIGHT_NESTED_NODE_VISITS),
)


def _validated_replay_limits(replay_limits: object) -> CampaignReplayLimits:
    _require(
        type(replay_limits) is CampaignReplayLimits,
        "campaign replay limits must use the exact frozen contract or a stricter copy",
    )
    assert isinstance(replay_limits, CampaignReplayLimits)
    replay_limits.__post_init__()
    for field in _REPLAY_LIMIT_FIELDS:
        _require(
            getattr(replay_limits, field) <= getattr(EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS, field),
            f"campaign replay limit {field} weakens the frozen research ceiling",
        )
    return replay_limits


def _batching_document(plan: BatchExecutionPlan) -> dict[str, object]:
    _require(type(plan) is BatchExecutionPlan, "batch plan must be a BatchExecutionPlan")
    return {
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "kg_candidate_chunk_size": plan.kg_candidate_chunk_size,
        "kg_fantasy_chunk_size": plan.kg_fantasy_chunk_size,
        "kg_max_combinations": plan.kg_max_combinations,
        "kg_max_joint_size": plan.kg_max_joint_size,
        "oracle_batch_size": plan.oracle_batch_size,
        "profile": plan.profile,
        "proposal_batch_size": plan.proposal_batch_size,
        "replay_length_bucket_boundaries": list(plan.replay_limits.length_bucket_boundaries),
        "replay_max_sequences": plan.replay_limits.max_sequences,
        "replay_max_tokens": plan.replay_limits.max_tokens,
        "rollout_batch_size": plan.rollout_batch_size,
        "surrogate_batch_size": plan.surrogate_batch_size,
    }


@dataclass(frozen=True, slots=True)
class CampaignHeader:
    """Immutable campaign identity and exact engineering resource ceilings."""

    campaign_id: str
    phase: Literal["screen", "confirmation"]
    configuration_id: str
    seed: int
    protocol_sha256: str
    batch_plan: BatchExecutionPlan

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="campaign ID")
        _require(
            type(self.phase) is str and self.phase in {"screen", "confirmation"},
            "campaign phase is invalid",
        )
        allowed = SCREEN_CONFIGURATION_IDS if self.phase == "screen" else CONFIRMATION_METHOD_IDS
        seeds = SCREEN_SEEDS if self.phase == "screen" else CONFIRMATION_SEEDS
        _require(self.configuration_id in allowed, "configuration is not allowed in this phase")
        _require(type(self.seed) is int and self.seed in seeds, "seed is not frozen for this phase")
        _require(
            self.protocol_sha256 == FROZEN_PROTOCOL_SHA256,
            "campaign protocol differs from the frozen successor-v1 protocol",
        )
        batching = _batching_document(self.batch_plan)
        _require(
            batching == _EXPECTED_CLUSTER_BATCHING,
            "campaign batching must equal the frozen cluster-first profile",
        )

    def document(self) -> dict[str, object]:
        """Return the canonical, explicitly non-authorizing header."""

        return {
            "artifact": CAMPAIGN_ARTIFACT,
            "authorization": {
                "automatic_production_eligible": False,
                "biological_superiority_claim_allowed": False,
                "oracle_execution_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "batching": _batching_document(self.batch_plan),
            "campaign_id": self.campaign_id,
            "configuration_id": self.configuration_id,
            "evidence_class": "engineering_fixture_only_not_scientific_evidence",
            "limits": {
                "scientific_elapsed_nanoseconds": MAX_SCIENTIFIC_ELAPSED_NS,
                "scientific_wall_seconds": 7200,
                "unique_oracle_calls": MAX_UNIQUE_ORACLE_CALLS,
            },
            "phase": self.phase,
            "protocol_sha256": self.protocol_sha256,
            "query_identity_fields": list(QUERY_IDENTITY_FIELDS),
            "schema_version": 1,
            "seed": self.seed,
            "status": "sealed_engineering_ledger_only",
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.document()))


@dataclass(frozen=True, slots=True)
class OracleQueryIdentity:
    """The frozen seven-field logical-call identity from the protocol."""

    canonical_sequence_id: str
    oracle_contract_sha256: str
    evaluator_sha256: str
    checkpoint_sha256: str
    endpoint_context_sha256: str
    transform_sha256: str
    replicate_id: int

    def __post_init__(self) -> None:
        for name in QUERY_IDENTITY_FIELDS[:-1]:
            _sha256(getattr(self, name), label=f"query identity {name}")
        _nonnegative_integer(self.replicate_id, label="query identity replicate ID")

    def document(self) -> dict[str, object]:
        return {
            "canonical_sequence_id": self.canonical_sequence_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "endpoint_context_sha256": self.endpoint_context_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "oracle_contract_sha256": self.oracle_contract_sha256,
            "replicate_id": self.replicate_id,
            "transform_sha256": self.transform_sha256,
        }

    @property
    def key(self) -> str:
        return sha256_bytes(
            b"amp/evolutionary-kl/oracle-query-identity/v1\0"
            + canonical_json_bytes(self.document())
        )


@dataclass(frozen=True, slots=True)
class ProposalLedgerEvent:
    proposal: ProposalRecord
    edge: EdgeRecord
    transposition: TranspositionEvent
    query_disposition: Literal["query_requested", "unevaluated", "rejected"]
    unevaluated_reason: str | None
    scientific_elapsed_ns: int

    def __post_init__(self) -> None:
        _require(type(self.proposal) is ProposalRecord, "proposal event requires ProposalRecord")
        _require(type(self.edge) is EdgeRecord, "proposal event requires EdgeRecord")
        _require(
            type(self.transposition) is TranspositionEvent,
            "proposal event requires TranspositionEvent",
        )
        _require(
            type(self.query_disposition) is str
            and self.query_disposition in {"query_requested", "unevaluated", "rejected"},
            "proposal query disposition is invalid",
        )
        if self.unevaluated_reason is not None:
            _identifier(self.unevaluated_reason, label="unevaluated reason")
        _nonnegative_integer(self.scientific_elapsed_ns, label="proposal elapsed ns")


@dataclass(frozen=True, slots=True)
class QueryLedgerEvent:
    query_id: str
    proposal_id: str
    identity: OracleQueryIdentity
    fidelity: str
    evaluator_version: str
    batch_id: str
    batch_position: int
    planned_cost: float
    scientific_elapsed_ns: int

    def __post_init__(self) -> None:
        for name in ("query_id", "proposal_id", "fidelity", "evaluator_version", "batch_id"):
            _identifier(getattr(self, name), label=f"query {name}")
        _require(type(self.identity) is OracleQueryIdentity, "query identity type differs")
        _nonnegative_integer(self.batch_position, label="query batch position")
        _require(_finite(self.planned_cost, label="query planned cost") >= 0.0, "cost is negative")
        _nonnegative_integer(self.scientific_elapsed_ns, label="query elapsed ns")


ResponseStatus = Literal["succeeded", "failed", "missing", "censored", "partial", "timeout"]


@dataclass(frozen=True, slots=True)
class ResponseLedgerEvent:
    response_id: str
    query_id: str
    status: ResponseStatus
    evaluation: EvaluationRecord | None
    status_detail: str | None
    scientific_elapsed_ns: int

    def __post_init__(self) -> None:
        _identifier(self.response_id, label="response ID")
        _identifier(self.query_id, label="response query ID")
        _require(
            type(self.status) is str
            and self.status in {"succeeded", "failed", "missing", "censored", "partial", "timeout"},
            "response status is invalid",
        )
        if self.status == "succeeded":
            _require(type(self.evaluation) is EvaluationRecord, "success requires EvaluationRecord")
            _require(self.status_detail is None, "success cannot carry failure detail")
        else:
            _require(
                self.evaluation is None, "non-success response cannot masquerade as evaluation"
            )
            _identifier(self.status_detail, label="non-success response detail")
        _nonnegative_integer(self.scientific_elapsed_ns, label="response elapsed ns")


@dataclass(frozen=True, slots=True)
class RecommendationLedgerEvent:
    recommendation_id: str
    proposal_id: str | None
    posterior_snapshot_sha256: str
    terminal_posterior_mean_utility: float
    abstention_reason: str | None
    scientific_elapsed_ns: int

    def __post_init__(self) -> None:
        _identifier(self.recommendation_id, label="recommendation ID")
        if self.proposal_id is not None:
            _identifier(self.proposal_id, label="recommended proposal ID")
            _require(self.abstention_reason is None, "selected recommendation cannot abstain")
        else:
            _identifier(self.abstention_reason, label="abstention reason")
        _sha256(self.posterior_snapshot_sha256, label="posterior snapshot")
        _finite(
            self.terminal_posterior_mean_utility,
            label="terminal posterior-mean utility",
        )
        _nonnegative_integer(self.scientific_elapsed_ns, label="recommendation elapsed ns")


CampaignEvent = (
    ProposalLedgerEvent | QueryLedgerEvent | ResponseLedgerEvent | RecommendationLedgerEvent
)


@dataclass(frozen=True, slots=True)
class _AppendPreflightSummary:
    event_count: int
    proposal_count: int
    query_count: int
    response_count: int
    inventory_items: int


@dataclass(frozen=True, slots=True)
class ResumeAuthority:
    """Caller-held checkpoint required to append after any prior sealed round."""

    round_count: int
    event_count: int
    proposal_count: int
    query_count: int
    response_count: int
    scientific_elapsed_ns: int
    head_seal_sha256: str | None
    last_event_sha256: str | None

    def __post_init__(self) -> None:
        for name in (
            "round_count",
            "event_count",
            "proposal_count",
            "query_count",
            "response_count",
            "scientific_elapsed_ns",
        ):
            _nonnegative_integer(getattr(self, name), label=f"resume {name}")
        empty = self.round_count == 0
        _require((self.head_seal_sha256 is None) == empty, "resume head presence is inconsistent")
        _require((self.last_event_sha256 is None) == empty, "resume event head is inconsistent")
        if empty:
            _require(
                self.event_count
                == self.proposal_count
                == self.query_count
                == self.response_count
                == self.scientific_elapsed_ns
                == 0,
                "empty resume authority must contain zero counts",
            )
        else:
            _sha256(self.head_seal_sha256, label="resume head seal")
            _sha256(self.last_event_sha256, label="resume last event")


EMPTY_RESUME_AUTHORITY = ResumeAuthority(0, 0, 0, 0, 0, 0, None, None)


@dataclass(frozen=True, slots=True)
class VerifiedCampaign:
    """Immutable result reconstructed solely from authenticated on-disk bytes."""

    header: CampaignHeader
    header_sha256: str
    round_seals: tuple[str, ...]
    round_timing_receipt_sha256s: tuple[str, ...]
    event_documents: tuple[bytes, ...]
    proposal_count: int
    query_count: int
    response_count: int
    outstanding_query_ids: tuple[str, ...]
    scientific_elapsed_ns: int
    last_event_sha256: str
    terminal: bool

    def resume_authority(self) -> ResumeAuthority:
        return ResumeAuthority(
            round_count=len(self.round_seals),
            event_count=len(self.event_documents),
            proposal_count=self.proposal_count,
            query_count=self.query_count,
            response_count=self.response_count,
            scientific_elapsed_ns=self.scientific_elapsed_ns,
            head_seal_sha256=self.round_seals[-1],
            last_event_sha256=self.last_event_sha256,
        )


@dataclass(frozen=True, slots=True)
class ReplayedCampaignProposal:
    """One proposal identity reconstructed from canonical campaign event bytes."""

    event_position: int
    proposal_id: str
    sequence_key: str
    sequence_bytes: bytes


@dataclass(frozen=True, slots=True)
class ReplayedCampaignQuery:
    """One charged logical call and its optional response, in call order."""

    event_position: int
    call_position: int
    query_id: str
    proposal_id: str
    identity: OracleQueryIdentity
    sequence_bytes: bytes
    fidelity: str
    evaluator_version: str
    batch_id: str
    batch_position: int
    planned_cost: float
    response_event_position: int | None
    response_id: str | None
    response_status: ResponseStatus | None
    response_status_detail: str | None
    evaluation_outcomes: tuple[tuple[str, float], ...] | None


@dataclass(frozen=True, slots=True)
class ReplayedCampaignEvents:
    """Immutable semantic view derived again from ``VerifiedCampaign`` bytes."""

    proposals: tuple[ReplayedCampaignProposal, ...]
    queries: tuple[ReplayedCampaignQuery, ...]
    event_document_inventory_sha256: str
    proposal_sequence_inventory_sha256: str


def _preflight_identifier(value: object, *, label: str) -> str:
    _require(
        type(value) is str and 0 < len(value) <= MAX_CAMPAIGN_EVENT_IDENTIFIER_LENGTH,
        f"{label} exceeds the campaign event identifier bound",
    )
    return _identifier(value, label=label)


def _preflight_sha256(value: object, *, label: str) -> str:
    _require(
        type(value) is str and len(value) == 64,
        f"{label} exceeds the campaign digest bound",
    )
    return _sha256(value, label=label)


def _preflight_optional_identifier(value: object, *, label: str) -> None:
    if value is not None:
        _preflight_identifier(value, label=label)


def _preflight_nonnegative_integer(
    value: object,
    *,
    label: str,
    maximum: int = (1 << 63) - 1,
) -> int:
    integer = _nonnegative_integer(value, label=label)
    _require(integer <= maximum, f"{label} exceeds its integer bound")
    return integer


def _preflight_campaign_header(header: object) -> CampaignHeader:
    _require(type(header) is CampaignHeader, "campaign header type differs")
    assert isinstance(header, CampaignHeader)
    for value, label in (
        (header.campaign_id, "campaign ID"),
        (header.phase, "campaign phase"),
        (header.configuration_id, "campaign configuration ID"),
    ):
        _preflight_identifier(value, label=label)
    _preflight_nonnegative_integer(header.seed, label="campaign seed")
    _preflight_sha256(header.protocol_sha256, label="campaign protocol")
    plan = header.batch_plan
    _require(type(plan) is BatchExecutionPlan, "campaign batch plan type differs")
    _preflight_identifier(plan.profile, label="campaign batch profile")
    for name in (
        "rollout_batch_size",
        "proposal_batch_size",
        "surrogate_batch_size",
        "kg_candidate_chunk_size",
        "kg_fantasy_chunk_size",
        "kg_max_joint_size",
        "kg_max_combinations",
        "oracle_batch_size",
        "gradient_accumulation_steps",
    ):
        _preflight_nonnegative_integer(getattr(plan, name), label=f"campaign batch {name}")
    batch_replay = plan.replay_limits
    _require(type(batch_replay) is BatchLimits, "campaign batch replay-limit type differs")
    _preflight_nonnegative_integer(
        batch_replay.max_sequences,
        label="campaign batch replay max sequences",
    )
    _preflight_nonnegative_integer(
        batch_replay.max_tokens,
        label="campaign batch replay max tokens",
    )
    _require(
        type(batch_replay.length_bucket_boundaries) is tuple
        and len(batch_replay.length_bucket_boundaries)
        == len(_EXPECTED_CLUSTER_BATCHING["replay_length_bucket_boundaries"]),
        "campaign batch replay bucket inventory differs",
    )
    for boundary in batch_replay.length_bucket_boundaries:
        _preflight_nonnegative_integer(boundary, label="campaign batch replay bucket")
    header.__post_init__()
    return header


def _preflight_resume_authority(
    authority: object,
    *,
    replay_limits: CampaignReplayLimits,
) -> ResumeAuthority:
    _require(type(authority) is ResumeAuthority, "resume authority type differs")
    assert isinstance(authority, ResumeAuthority)
    for name, maximum in (
        ("round_count", replay_limits.max_rounds),
        ("event_count", replay_limits.max_events),
        ("proposal_count", replay_limits.max_proposals),
        ("query_count", replay_limits.max_queries),
        ("response_count", replay_limits.max_responses),
        ("scientific_elapsed_ns", MAX_SCIENTIFIC_ELAPSED_NS),
    ):
        value = _preflight_nonnegative_integer(getattr(authority, name), label=f"resume {name}")
        _require(value <= maximum, f"resume {name} exceeds the replay bound")
    if authority.head_seal_sha256 is not None:
        _preflight_sha256(authority.head_seal_sha256, label="resume head seal")
    if authority.last_event_sha256 is not None:
        _preflight_sha256(authority.last_event_sha256, label="resume last event")
    authority.__post_init__()
    return authority


def _preflight_nested_inventory(
    value: object,
    *,
    label: str,
    maximum_items: int,
    maximum_node_visits: int = MAX_CAMPAIGN_NESTED_VALUE_NODES,
) -> int:
    _require(type(value) in (list, tuple), f"{label} must be an array")
    assert isinstance(value, list | tuple)
    _require(len(value) <= maximum_items, f"{label} exceeds its item bound")
    _require(
        type(maximum_node_visits) is int and maximum_node_visits >= 0,
        f"{label} nested-node work ceiling differs",
    )
    _require(
        len(value) <= maximum_node_visits,
        f"{label} exceeds the cumulative nested-node work ceiling",
    )
    stack: list[tuple[Iterator[object], int]] = [(iter(value), 1)]
    node_count = 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        node_count += 1
        _require(
            node_count <= MAX_CAMPAIGN_NESTED_VALUE_NODES,
            f"{label} exceeds the nested-node bound",
        )
        _require(
            node_count <= maximum_node_visits,
            f"{label} exceeds the cumulative nested-node work ceiling",
        )
        if type(item) in (list, tuple):
            assert isinstance(item, list | tuple)
            _require(
                depth < MAX_CAMPAIGN_NESTED_VALUE_DEPTH,
                f"{label} exceeds the nesting-depth bound",
            )
            _require(
                len(item) <= MAX_CAMPAIGN_NESTED_VALUE_NODES,
                f"{label} exceeds the nested-node bound",
            )
            stack.append((iter(item), depth + 1))
        elif type(item) is str:
            _require(
                len(item) <= MAX_CAMPAIGN_EVENT_IDENTIFIER_LENGTH,
                f"{label} contains an oversized string",
            )
        elif type(item) is int:
            assert isinstance(item, int)
            _require(abs(item) <= (1 << 63) - 1, f"{label} contains an oversized integer")
        elif type(item) is float:
            _finite(item, label=f"{label} value")
        else:
            _require(item is None or type(item) is bool, f"{label} contains an invalid value")
    return node_count


def _preflight_trace(trace: ProbabilityTrace, *, label: str) -> None:
    _require(type(trace) is ProbabilityTrace, f"{label} type differs")
    _require(type(trace.factors) is tuple and bool(trace.factors), f"{label} factors differ")
    _require(
        len(trace.factors) <= MAX_CAMPAIGN_PROBABILITY_FACTORS,
        f"{label} exceeds the factor bound",
    )
    for factor in trace.factors:
        _require(type(factor) is ProbabilityFactor, f"{label} factor type differs")
        _preflight_identifier(factor.name, label=f"{label} factor")
        _require(
            type(factor.probability) is float
            and 0.0 <= _finite(factor.probability, label=f"{label} probability") <= 1.0,
            f"{label} probability differs",
        )


def _preflight_selection(selection: SelectionDecision) -> None:
    _require(type(selection) is SelectionDecision, "selection type differs")
    _require(type(selection.selected) is bool, "selection selected flag differs")
    _preflight_identifier(selection.selection_set_id, label="selection set ID")
    _preflight_identifier(selection.policy_version, label="selection policy version")
    _preflight_nonnegative_integer(selection.seed, label="selection seed")
    _preflight_trace(selection.propensity, label="selection propensity")
    _require(
        type(selection.eligible_proposal_ids) is tuple,
        "selection eligible proposal inventory type differs",
    )
    _require(
        len(selection.eligible_proposal_ids) <= MAX_CAMPAIGN_ELIGIBLE_PROPOSAL_IDS,
        "selection eligible proposal inventory exceeds its bound",
    )
    for proposal_id in selection.eligible_proposal_ids:
        _preflight_identifier(proposal_id, label="eligible proposal ID")


def _preflight_proposal(proposal: ProposalRecord) -> None:
    _require(type(proposal) is ProposalRecord, "proposal record type differs")
    _require(type(proposal.hard_valid) is bool, "proposal hard-valid flag differs")
    for name in ("proposal_id", "rollout_id", "policy_version"):
        _preflight_identifier(getattr(proposal, name), label=f"proposal {name}")
    _preflight_optional_identifier(proposal.niche_id, label="proposal niche ID")
    _preflight_optional_identifier(proposal.rejection_reason, label="proposal rejection reason")
    _preflight_nonnegative_integer(proposal.proposal_round, label="proposal round")
    _require(
        type(proposal.sequence) is str
        and 0 < len(proposal.sequence) <= MAX_CAMPAIGN_SEQUENCE_LENGTH,
        "proposal sequence exceeds the campaign sequence bound",
    )
    try:
        proposal.sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise CampaignLedgerError("proposal sequence is not bounded ASCII") from error
    _require(
        proposal.sequence == "".join(proposal.sequence.split()).upper(),
        "proposal sequence is not canonical",
    )
    _require(
        type(proposal.cheap_predictions) is tuple,
        "proposal cheap-prediction inventory type differs",
    )
    _require(
        len(proposal.cheap_predictions) <= MAX_CAMPAIGN_CHEAP_PREDICTIONS,
        "proposal cheap-prediction inventory exceeds its bound",
    )
    for row in proposal.cheap_predictions:
        _require(type(row) is tuple and len(row) == 2, "cheap prediction row differs")
        name, value = row
        _preflight_identifier(name, label="cheap-prediction name")
        _finite(value, label="cheap-prediction value")
    _preflight_selection(proposal.selection)


def _preflight_edge(
    edge: EdgeRecord,
    *,
    maximum_nested_node_visits: int = 2 * MAX_CAMPAIGN_NESTED_VALUE_NODES,
) -> int:
    _require(type(edge) is EdgeRecord, "edge record type differs")
    for name in ("edge_id", "proposal_id", "rollout_id", "operator"):
        _preflight_identifier(getattr(edge, name), label=f"edge {name}")
    _require(type(edge.parent_sequence_keys) is tuple, "edge parent inventory type differs")
    expected_parent_count = _expected_operator_parent_cardinality(edge.operator)
    _require(
        len(edge.parent_sequence_keys) == expected_parent_count,
        f"edge operator {edge.operator!r} requires exactly "
        f"{expected_parent_count} parent sequence keys",
    )
    for parent in edge.parent_sequence_keys:
        _preflight_sha256(parent, label="edge parent sequence key")
    _require(type(edge.edit_description) is tuple, "edge edit description type differs")
    edit_node_visits = _preflight_nested_inventory(
        edge.edit_description,
        label="edge edit description",
        maximum_items=MAX_CAMPAIGN_EDIT_ITEMS,
        maximum_node_visits=maximum_nested_node_visits,
    )
    _preflight_trace(edge.proposal_trace, label="edge proposal trace")
    _require(
        type(edge.behavior_log_probabilities) is tuple and not edge.behavior_log_probabilities,
        "edge behavior log probabilities exceed the supported empty contract",
    )
    for name in ("random_stream", "sample_index"):
        value = getattr(edge, name)
        if value is not None:
            _preflight_nonnegative_integer(value, label=f"edge {name}")
    _require(type(edge.sampling_parameters) is tuple, "edge sampling parameters type differs")
    sampling_node_visits = _preflight_nested_inventory(
        edge.sampling_parameters,
        label="edge sampling parameters",
        maximum_items=MAX_CAMPAIGN_SAMPLING_PARAMETERS,
        maximum_node_visits=maximum_nested_node_visits - edit_node_visits,
    )
    for row in edge.sampling_parameters:
        _require(type(row) is tuple and len(row) == 2, "sampling parameter row differs")
        _preflight_identifier(row[0], label="sampling parameter name")
    return edit_node_visits + sampling_node_visits


def _preflight_transposition(transposition: TranspositionEvent) -> None:
    _require(type(transposition) is TranspositionEvent, "transposition record type differs")
    for name in ("proposal_id", "edge_id", "first_proposal_id"):
        _preflight_identifier(getattr(transposition, name), label=f"transposition {name}")
    _preflight_nonnegative_integer(transposition.event_index, label="transposition event index")
    for name in (
        "duplicate",
        "hard_valid",
        "selected_for_evaluation",
        "has_any_cached_evaluation",
        "dag_edge_admitted",
    ):
        _require(type(getattr(transposition, name)) is bool, f"transposition {name} flag differs")
    _preflight_sha256(transposition.sequence_key, label="transposition sequence key")
    _preflight_optional_identifier(
        transposition.graph_rejection_reason,
        label="transposition graph rejection reason",
    )


def _preflight_evaluation(evaluation: EvaluationRecord) -> None:
    _require(type(evaluation) is EvaluationRecord, "evaluation record type differs")
    for name in ("evaluation_id", "proposal_id", "fidelity", "evaluator_version", "batch_id"):
        _preflight_identifier(getattr(evaluation, name), label=f"evaluation {name}")
    _require(type(evaluation.outcomes) is tuple, "evaluation outcome inventory type differs")
    _require(
        len(evaluation.outcomes) <= MAX_CAMPAIGN_EVALUATION_OUTCOMES,
        "evaluation outcome inventory exceeds its bound",
    )
    for row in evaluation.outcomes:
        _require(type(row) is tuple and len(row) == 2, "evaluation outcome row differs")
        name, value = row
        _preflight_identifier(name, label="evaluation outcome name")
        _finite(value, label="evaluation outcome value")
    _finite(evaluation.cost, label="evaluation cost")
    _preflight_nonnegative_integer(evaluation.replicate, label="evaluation replicate")


def _bounded_append_inventory_length(
    value: object,
    *,
    maximum_items: int,
    label: str,
) -> int:
    _require(type(value) is tuple, f"{label} inventory type differs")
    assert isinstance(value, tuple)
    _require(len(value) <= maximum_items, f"{label} inventory exceeds its bound")
    return len(value)


def _append_event_inventory_items(event: CampaignEvent) -> int:
    """Return a constant-time upper count for variable top-level item visits."""

    if type(event) is ProposalLedgerEvent:
        assert isinstance(event, ProposalLedgerEvent)
        _require(type(event.proposal) is ProposalRecord, "proposal record type differs")
        _require(type(event.edge) is EdgeRecord, "edge record type differs")
        _require(
            type(event.transposition) is TranspositionEvent,
            "transposition record type differs",
        )
        proposal = event.proposal
        edge = event.edge
        _require(type(proposal.selection) is SelectionDecision, "selection type differs")
        selection = proposal.selection
        _require(
            type(selection.propensity) is ProbabilityTrace, "selection propensity type differs"
        )
        _require(type(edge.proposal_trace) is ProbabilityTrace, "edge proposal trace type differs")
        inventory_items = sum(
            (
                _bounded_append_inventory_length(
                    proposal.cheap_predictions,
                    maximum_items=MAX_CAMPAIGN_CHEAP_PREDICTIONS,
                    label="proposal cheap prediction",
                ),
                _bounded_append_inventory_length(
                    selection.propensity.factors,
                    maximum_items=MAX_CAMPAIGN_PROBABILITY_FACTORS,
                    label="selection propensity factor",
                ),
                _bounded_append_inventory_length(
                    selection.eligible_proposal_ids,
                    maximum_items=MAX_CAMPAIGN_ELIGIBLE_PROPOSAL_IDS,
                    label="selection eligible proposal",
                ),
                _bounded_append_inventory_length(
                    edge.parent_sequence_keys,
                    maximum_items=2,
                    label="edge parent",
                ),
                _bounded_append_inventory_length(
                    edge.edit_description,
                    maximum_items=MAX_CAMPAIGN_EDIT_ITEMS,
                    label="edge edit",
                ),
                _bounded_append_inventory_length(
                    edge.proposal_trace.factors,
                    maximum_items=MAX_CAMPAIGN_PROBABILITY_FACTORS,
                    label="edge proposal-trace factor",
                ),
                _bounded_append_inventory_length(
                    edge.behavior_log_probabilities,
                    maximum_items=0,
                    label="edge behavior log-probability",
                ),
                2
                * _bounded_append_inventory_length(
                    edge.sampling_parameters,
                    maximum_items=MAX_CAMPAIGN_SAMPLING_PARAMETERS,
                    label="edge sampling parameter",
                ),
            )
        )
        return inventory_items
    if type(event) is ResponseLedgerEvent:
        assert isinstance(event, ResponseLedgerEvent)
        if event.evaluation is None:
            return 0
        _require(type(event.evaluation) is EvaluationRecord, "evaluation record type differs")
        return _bounded_append_inventory_length(
            event.evaluation.outcomes,
            maximum_items=MAX_CAMPAIGN_EVALUATION_OUTCOMES,
            label="evaluation outcome",
        )
    _require(
        type(event) in (QueryLedgerEvent, RecommendationLedgerEvent),
        "campaign event has an unsupported exact type",
    )
    return 0


def _preflight_append_event_batch(
    events: object,
    *,
    replay_limits: CampaignReplayLimits,
) -> _AppendPreflightSummary:
    """Bound aggregate live-object work before detailed traversal or path access."""

    _require(type(events) is tuple and bool(events), "campaign round events must be nonempty tuple")
    assert isinstance(events, tuple)
    _require(len(events) <= replay_limits.max_events, "campaign event replay cap exceeded")
    proposal_count = 0
    query_count = 0
    response_count = 0
    inventory_items = 0
    for event in events:
        if type(event) is ProposalLedgerEvent:
            proposal_count += 1
        elif type(event) is QueryLedgerEvent:
            query_count += 1
        elif type(event) is ResponseLedgerEvent:
            response_count += 1
        inventory_items += _append_event_inventory_items(event)
        _require(
            inventory_items <= replay_limits.max_append_preflight_inventory_items,
            "campaign append aggregate inventory work exceeds replay cap",
        )
    _require(
        proposal_count <= replay_limits.max_proposals,
        "campaign proposal replay cap exceeded before new round",
    )
    _require(
        query_count <= replay_limits.max_queries,
        "campaign query replay cap exceeded before new round",
    )
    _require(
        response_count <= replay_limits.max_responses,
        "campaign response replay cap exceeded before new round",
    )
    return _AppendPreflightSummary(
        event_count=len(events),
        proposal_count=proposal_count,
        query_count=query_count,
        response_count=response_count,
        inventory_items=inventory_items,
    )


def _preflight_campaign_event_for_append(
    event: CampaignEvent,
    *,
    maximum_nested_node_visits: int,
) -> int:
    nested_node_visits = 0
    if type(event) is ProposalLedgerEvent:
        assert isinstance(event, ProposalLedgerEvent)
        _preflight_proposal(event.proposal)
        nested_node_visits = _preflight_edge(
            event.edge,
            maximum_nested_node_visits=maximum_nested_node_visits,
        )
        _preflight_transposition(event.transposition)
        _preflight_identifier(event.query_disposition, label="proposal query disposition")
        _preflight_optional_identifier(event.unevaluated_reason, label="unevaluated reason")
        elapsed = event.scientific_elapsed_ns
    elif type(event) is QueryLedgerEvent:
        assert isinstance(event, QueryLedgerEvent)
        for name in ("query_id", "proposal_id", "fidelity", "evaluator_version", "batch_id"):
            _preflight_identifier(getattr(event, name), label=f"query {name}")
        _require(type(event.identity) is OracleQueryIdentity, "query identity type differs")
        for name in QUERY_IDENTITY_FIELDS[:-1]:
            _preflight_sha256(
                getattr(event.identity, name),
                label=f"query identity {name}",
            )
        _preflight_nonnegative_integer(
            event.identity.replicate_id,
            label="query identity replicate ID",
        )
        _preflight_nonnegative_integer(event.batch_position, label="query batch position")
        _finite(event.planned_cost, label="query planned cost")
        elapsed = event.scientific_elapsed_ns
    elif type(event) is ResponseLedgerEvent:
        assert isinstance(event, ResponseLedgerEvent)
        _preflight_identifier(event.response_id, label="response ID")
        _preflight_identifier(event.query_id, label="response query ID")
        _preflight_identifier(event.status, label="response status")
        _preflight_optional_identifier(event.status_detail, label="response status detail")
        if event.evaluation is not None:
            _preflight_evaluation(event.evaluation)
        elapsed = event.scientific_elapsed_ns
    elif type(event) is RecommendationLedgerEvent:
        assert isinstance(event, RecommendationLedgerEvent)
        _preflight_identifier(event.recommendation_id, label="recommendation ID")
        _preflight_optional_identifier(event.proposal_id, label="recommended proposal ID")
        _preflight_optional_identifier(event.abstention_reason, label="abstention reason")
        _preflight_sha256(event.posterior_snapshot_sha256, label="posterior snapshot")
        _finite(
            event.terminal_posterior_mean_utility,
            label="terminal posterior-mean utility",
        )
        elapsed = event.scientific_elapsed_ns
    else:
        raise CampaignLedgerError("campaign event has an unsupported exact type")
    _preflight_nonnegative_integer(
        elapsed,
        label="campaign event scientific elapsed ns",
        maximum=MAX_SCIENTIFIC_ELAPSED_NS,
    )
    return nested_node_visits


def _trace_document(trace: ProbabilityTrace) -> list[dict[str, object]]:
    return [{"name": factor.name, "probability": factor.probability} for factor in trace.factors]


def _trace_from_document(value: object) -> ProbabilityTrace:
    _require(type(value) is list and bool(value), "probability trace is invalid")
    assert isinstance(value, list)
    _require(
        len(value) <= MAX_CAMPAIGN_PROBABILITY_FACTORS,
        "probability trace exceeds the factor bound",
    )
    factors: list[ProbabilityFactor] = []
    for row in value:
        row = _require_exact_mapping_keys(
            row,
            ("name", "probability"),
            label="trace row differs",
        )
        _preflight_identifier(row["name"], label="probability factor name")
        _require(
            0.0 <= _finite(row["probability"], label="probability factor value") <= 1.0,
            "probability factor value differs",
        )
        factors.append(ProbabilityFactor(name=row["name"], probability=row["probability"]))
    return ProbabilityTrace(tuple(factors))


def _selection_document(value: SelectionDecision) -> dict[str, object]:
    return {
        "eligible_proposal_ids": list(value.eligible_proposal_ids),
        "policy_version": value.policy_version,
        "propensity": _trace_document(value.propensity),
        "seed": value.seed,
        "selected": value.selected,
        "selection_set_id": value.selection_set_id,
    }


def _selection_from_document(value: object) -> SelectionDecision:
    keys = {
        "eligible_proposal_ids",
        "policy_version",
        "propensity",
        "seed",
        "selected",
        "selection_set_id",
    }
    value = _require_exact_mapping_keys(value, keys, label="selection record schema differs")
    eligible = value["eligible_proposal_ids"]
    _require(type(eligible) is list, "eligible proposal IDs must be a list")
    assert isinstance(eligible, list)
    _require(
        len(eligible) <= MAX_CAMPAIGN_ELIGIBLE_PROPOSAL_IDS,
        "selection eligible proposal inventory exceeds its bound",
    )
    _preflight_identifier(value["selection_set_id"], label="selection set ID")
    _preflight_identifier(value["policy_version"], label="selection policy version")
    _require(type(value["selected"]) is bool, "selection selected flag differs")
    _preflight_nonnegative_integer(value["seed"], label="selection seed")
    for proposal_id in eligible:
        _preflight_identifier(proposal_id, label="eligible proposal ID")
    selection = SelectionDecision(
        selected=value["selected"],
        propensity=_trace_from_document(value["propensity"]),
        selection_set_id=value["selection_set_id"],
        eligible_proposal_ids=tuple(eligible),
        policy_version=value["policy_version"],
        seed=value["seed"],
    )
    _preflight_selection(selection)
    return selection


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
        "selection": _selection_document(value.selection),
        "sequence": value.sequence,
    }


def _proposal_from_document(value: object) -> ProposalRecord:
    keys = {
        "cheap_predictions",
        "hard_valid",
        "niche_id",
        "policy_version",
        "proposal_id",
        "proposal_round",
        "rejection_reason",
        "rollout_id",
        "selection",
        "sequence",
    }
    value = _require_exact_mapping_keys(value, keys, label="proposal record schema differs")
    predictions = value["cheap_predictions"]
    _require(type(predictions) is list, "cheap predictions must be a list")
    assert isinstance(predictions, list)
    _require(
        len(predictions) <= MAX_CAMPAIGN_CHEAP_PREDICTIONS,
        "proposal cheap-prediction inventory exceeds its bound",
    )
    for name in ("proposal_id", "rollout_id", "policy_version"):
        _preflight_identifier(value[name], label=f"proposal {name}")
    _preflight_optional_identifier(value["niche_id"], label="proposal niche ID")
    _preflight_optional_identifier(value["rejection_reason"], label="proposal rejection reason")
    sequence = value["sequence"]
    _require(
        type(sequence) is str and 0 < len(sequence) <= MAX_CAMPAIGN_SEQUENCE_LENGTH,
        "proposal sequence exceeds the campaign sequence bound",
    )
    _require(type(value["hard_valid"]) is bool, "proposal hard-valid flag differs")
    _preflight_nonnegative_integer(value["proposal_round"], label="proposal round")
    for row in predictions:
        _require(type(row) is list and len(row) == 2, "cheap prediction row differs")
        assert isinstance(row, list)
        _preflight_identifier(row[0], label="cheap-prediction name")
        _finite(row[1], label="cheap-prediction value")
    proposal = ProposalRecord(
        proposal_id=value["proposal_id"],
        rollout_id=value["rollout_id"],
        sequence=sequence,
        hard_valid=value["hard_valid"],
        rejection_reason=value["rejection_reason"],
        selection=_selection_from_document(value["selection"]),
        policy_version=value["policy_version"],
        proposal_round=value["proposal_round"],
        niche_id=value["niche_id"],
        cheap_predictions=tuple(tuple(row) for row in predictions),
    )
    _preflight_proposal(proposal)
    return proposal


def _freeze_json_tuple(value: object) -> object:
    if type(value) is list:
        assert isinstance(value, list)
        return tuple(_freeze_json_tuple(item) for item in value)
    return value


def _edge_document(value: EdgeRecord) -> dict[str, object]:
    return {
        "behavior_log_probabilities": list(value.behavior_log_probabilities),
        "edge_id": value.edge_id,
        "edit_description": list(value.edit_description),
        "operator": value.operator,
        "parent_sequence_keys": list(value.parent_sequence_keys),
        "proposal_id": value.proposal_id,
        "proposal_trace": _trace_document(value.proposal_trace),
        "random_stream": value.random_stream,
        "rollout_id": value.rollout_id,
        "sample_index": value.sample_index,
        "sampling_parameters": [[name, value] for name, value in value.sampling_parameters],
    }


def _expected_operator_parent_cardinality(operator: object) -> int:
    operator_name = _identifier(operator, label="edge operator")
    expected = next(
        (
            cardinality
            for registered_operator, cardinality in CAMPAIGN_OPERATOR_PARENT_CARDINALITIES
            if registered_operator == operator_name
        ),
        None,
    )
    _require(expected is not None, "edge operator has no registered parent-cardinality contract")
    assert expected is not None
    return expected


def _validate_operator_parent_cardinality(operator: object, parents: object) -> None:
    expected = _expected_operator_parent_cardinality(operator)
    operator_name = str(operator)
    _require(type(parents) is list, "edge parents must be a list")
    assert isinstance(parents, list)
    _require(
        len(parents) == expected,
        f"edge operator {operator_name!r} requires exactly {expected} parent sequence keys",
    )


def _edge_from_document(value: object) -> EdgeRecord:
    keys = {
        "behavior_log_probabilities",
        "edge_id",
        "edit_description",
        "operator",
        "parent_sequence_keys",
        "proposal_id",
        "proposal_trace",
        "random_stream",
        "rollout_id",
        "sample_index",
        "sampling_parameters",
    }
    value = _require_exact_mapping_keys(value, keys, label="edge record schema differs")
    for name in ("edge_id", "proposal_id", "rollout_id", "operator"):
        _preflight_identifier(value[name], label=f"edge {name}")
    _preflight_nested_inventory(
        value["edit_description"],
        label="edge edit description",
        maximum_items=MAX_CAMPAIGN_EDIT_ITEMS,
    )
    _preflight_nested_inventory(
        value["sampling_parameters"],
        label="edge sampling parameters",
        maximum_items=MAX_CAMPAIGN_SAMPLING_PARAMETERS,
    )
    edits = _freeze_json_tuple(value["edit_description"])
    parents = value["parent_sequence_keys"]
    behaviors = value["behavior_log_probabilities"]
    parameters = value["sampling_parameters"]
    _require(type(edits) is tuple, "edge edits must be a list")
    _validate_operator_parent_cardinality(value["operator"], parents)
    assert isinstance(parents, list)
    _require(type(behaviors) is list, "behavior probabilities must be a list")
    _require(type(parameters) is list, "sampling parameters must be a list")
    assert isinstance(behaviors, list)
    _require(not behaviors, "edge behavior log probabilities exceed the supported empty contract")
    for name in ("random_stream", "sample_index"):
        scalar = value[name]
        if scalar is not None:
            _preflight_nonnegative_integer(scalar, label=f"edge {name}")
    _require(
        all(type(row) is list and len(row) == 2 for row in parameters),
        "sampling parameter rows differ",
    )
    for row in parameters:
        assert isinstance(row, list)
        _preflight_identifier(row[0], label="sampling parameter name")
    edge = EdgeRecord(
        edge_id=value["edge_id"],
        proposal_id=value["proposal_id"],
        rollout_id=value["rollout_id"],
        parent_sequence_keys=tuple(parents),
        operator=value["operator"],
        edit_description=edits,
        proposal_trace=_trace_from_document(value["proposal_trace"]),
        behavior_log_probabilities=tuple(behaviors),
        random_stream=value["random_stream"],
        sample_index=value["sample_index"],
        sampling_parameters=tuple((row[0], _freeze_json_tuple(row[1])) for row in parameters),
    )
    _preflight_edge(edge)
    return edge


def _transposition_document(value: TranspositionEvent) -> dict[str, object]:
    return {
        "dag_edge_admitted": value.dag_edge_admitted,
        "duplicate": value.duplicate,
        "edge_id": value.edge_id,
        "event_index": value.event_index,
        "first_proposal_id": value.first_proposal_id,
        "graph_rejection_reason": value.graph_rejection_reason,
        "hard_valid": value.hard_valid,
        "has_any_cached_evaluation": value.has_any_cached_evaluation,
        "proposal_id": value.proposal_id,
        "selected_for_evaluation": value.selected_for_evaluation,
        "sequence_key": value.sequence_key,
    }


def _transposition_from_document(value: object) -> TranspositionEvent:
    keys = {
        "dag_edge_admitted",
        "duplicate",
        "edge_id",
        "event_index",
        "first_proposal_id",
        "graph_rejection_reason",
        "hard_valid",
        "has_any_cached_evaluation",
        "proposal_id",
        "selected_for_evaluation",
        "sequence_key",
    }
    value = _require_exact_mapping_keys(value, keys, label="transposition schema differs")
    for name in ("proposal_id", "edge_id", "first_proposal_id"):
        _preflight_identifier(value[name], label=f"transposition {name}")
    _preflight_nonnegative_integer(value["event_index"], label="transposition event index")
    _preflight_sha256(value["sequence_key"], label="transposition sequence key")
    _preflight_optional_identifier(
        value["graph_rejection_reason"],
        label="transposition graph rejection reason",
    )
    transposition = TranspositionEvent(**value)
    _preflight_transposition(transposition)
    return transposition


def _evaluation_document(value: EvaluationRecord) -> dict[str, object]:
    return {
        "batch_id": value.batch_id,
        "cost": value.cost,
        "evaluation_id": value.evaluation_id,
        "evaluator_version": value.evaluator_version,
        "fidelity": value.fidelity,
        "outcomes": [list(row) for row in value.outcomes],
        "proposal_id": value.proposal_id,
        "replicate": value.replicate,
    }


def _evaluation_from_document(value: object) -> EvaluationRecord:
    keys = {
        "batch_id",
        "cost",
        "evaluation_id",
        "evaluator_version",
        "fidelity",
        "outcomes",
        "proposal_id",
        "replicate",
    }
    value = _require_exact_mapping_keys(value, keys, label="evaluation schema differs")
    outcomes = value["outcomes"]
    _require(type(outcomes) is list, "evaluation outcomes must be a list")
    assert isinstance(outcomes, list)
    _require(
        len(outcomes) <= MAX_CAMPAIGN_EVALUATION_OUTCOMES,
        "evaluation outcome inventory exceeds its bound",
    )
    for name in ("evaluation_id", "proposal_id", "fidelity", "evaluator_version", "batch_id"):
        _preflight_identifier(value[name], label=f"evaluation {name}")
    _finite(value["cost"], label="evaluation cost")
    _preflight_nonnegative_integer(value["replicate"], label="evaluation replicate")
    for row in outcomes:
        _require(type(row) is list and len(row) == 2, "evaluation outcome row differs")
        assert isinstance(row, list)
        _preflight_identifier(row[0], label="evaluation outcome name")
        _finite(row[1], label="evaluation outcome value")
    evaluation = EvaluationRecord(
        evaluation_id=value["evaluation_id"],
        proposal_id=value["proposal_id"],
        fidelity=value["fidelity"],
        evaluator_version=value["evaluator_version"],
        cost=value["cost"],
        outcomes=tuple(tuple(row) for row in outcomes),
        batch_id=value["batch_id"],
        replicate=value["replicate"],
    )
    _preflight_evaluation(evaluation)
    return evaluation


def _header_from_document(value: object) -> CampaignHeader:
    keys = {
        "artifact",
        "authorization",
        "batching",
        "campaign_id",
        "configuration_id",
        "evidence_class",
        "limits",
        "phase",
        "protocol_sha256",
        "query_identity_fields",
        "schema_version",
        "seed",
        "status",
    }
    value = _require_exact_mapping_keys(value, keys, label="campaign header schema differs")
    _require(value["artifact"] == CAMPAIGN_ARTIFACT, "campaign artifact differs")
    _require(value["schema_version"] == 1, "campaign schema version differs")
    _require(value["status"] == "sealed_engineering_ledger_only", "campaign status differs")
    _require(
        value["evidence_class"] == "engineering_fixture_only_not_scientific_evidence",
        "campaign evidence class differs",
    )
    _require(
        value["authorization"]
        == {
            "automatic_production_eligible": False,
            "biological_superiority_claim_allowed": False,
            "oracle_execution_authorized": False,
            "scientific_evidence_accepted": False,
        },
        "campaign authorization must remain entirely false",
    )
    _require(
        value["limits"]
        == {
            "scientific_elapsed_nanoseconds": MAX_SCIENTIFIC_ELAPSED_NS,
            "scientific_wall_seconds": 7200,
            "unique_oracle_calls": MAX_UNIQUE_ORACLE_CALLS,
        },
        "campaign limits differ from the exact ceilings",
    )
    _require(
        value["query_identity_fields"] == list(QUERY_IDENTITY_FIELDS),
        "query identity fields differ from the frozen contract",
    )
    batching = value["batching"]
    _require(type(batching) is dict, "campaign batching is not an object")
    assert isinstance(batching, dict)
    header = CampaignHeader(
        campaign_id=value["campaign_id"],
        phase=value["phase"],
        configuration_id=value["configuration_id"],
        seed=value["seed"],
        protocol_sha256=value["protocol_sha256"],
        batch_plan=batch_execution_plan_from_mapping(batching),
    )
    _require(header.document() == value, "campaign header is not canonical for its semantics")
    return header


def _strict_json(payload: bytes, *, label: str) -> object:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            _require(key not in result, f"{label} duplicates JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise CampaignLedgerError(f"{label} contains invalid constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignLedgerError(f"{label} is not strict UTF-8 JSON") from error
    _require(canonical_json_bytes(value) == payload, f"{label} is not canonical JSON")
    return value


def _jsonl(payload: bytes, *, label: str) -> tuple[tuple[dict[str, object], bytes], ...]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} is invalid")
    result: list[tuple[dict[str, object], bytes]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        value = _strict_json(line, label=f"{label} row {index}")
        _require(type(value) is dict, f"{label} row {index} must be an object")
        assert isinstance(value, dict)
        result.append((value, line))
    return tuple(result)


def _query_identity_from_document(value: object) -> OracleQueryIdentity:
    value = _require_exact_mapping_keys(
        value,
        QUERY_IDENTITY_FIELDS,
        label="query identity schema differs",
    )
    for name in QUERY_IDENTITY_FIELDS[:-1]:
        _preflight_sha256(value[name], label=f"query identity {name}")
    _preflight_nonnegative_integer(
        value["replicate_id"],
        label="query identity replicate ID",
    )
    return OracleQueryIdentity(**value)


@dataclass
class _State:
    header: CampaignHeader
    previous_event_sha256: str
    replay_limits: CampaignReplayLimits
    event_count: int = 0
    proposal_count: int = 0
    query_count: int = 0
    response_count: int = 0
    recommendation_count: int = 0
    cumulative_event_bytes: int = 0
    dag_reachability_node_visits: int = 0
    dag_reachability_edge_scans: int = 0
    scientific_elapsed_ns: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        self.replay_limits = _validated_replay_limits(self.replay_limits)
        self.proposals: dict[str, ProposalRecord] = {}
        self.proposal_dispositions: dict[str, str] = {}
        self.proposal_query_counts: dict[str, int] = {}
        self.edges: set[str] = set()
        self.first_proposal_by_sequence: dict[str, str] = {}
        self.successful_evaluation_sequences: set[str] = set()
        self.query_by_id: dict[str, dict[str, object]] = {}
        self.query_keys: set[str] = set()
        self.responses: set[str] = set()
        self.response_ids: set[str] = set()
        self.evaluation_ids: set[str] = set()
        self.batch_rounds: dict[str, int] = {}
        self.batch_next_positions: dict[str, int] = {}
        self.dag_nodes: set[str] = set()
        self.children: dict[str, set[str]] = {}


def _new_state(
    header: CampaignHeader,
    *,
    replay_limits: CampaignReplayLimits,
) -> _State:
    header = _preflight_campaign_header(header)
    genesis = sha256_bytes(GENESIS_HASH_DOMAIN + bytes.fromhex(header.sha256))
    return _State(
        header=header,
        previous_event_sha256=genesis,
        replay_limits=replay_limits,
    )


def _charge_dag_reachability_work(
    state: _State,
    *,
    node_visits: int = 0,
    edge_scans: int = 0,
) -> None:
    state.dag_reachability_node_visits += node_visits
    state.dag_reachability_edge_scans += edge_scans
    limits = state.replay_limits
    _require(
        state.dag_reachability_node_visits <= limits.max_dag_reachability_node_visits,
        "campaign DAG reachability node-visit replay cap exceeded",
    )
    _require(
        state.dag_reachability_edge_scans <= limits.max_dag_reachability_edge_scans,
        "campaign DAG reachability edge-scan replay cap exceeded",
    )


def _reachable(state: _State, start: str, target: str) -> bool:
    if start not in state.dag_nodes:
        return False
    pending = [start]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        _charge_dag_reachability_work(state, node_visits=1)
        if current == target:
            return True
        child_inventory = state.children.get(current, ())
        _charge_dag_reachability_work(state, edge_scans=len(child_inventory))
        children = tuple(sorted(child_inventory))
        pending.extend(reversed(children))
    return False


def _dag_admission(
    state: _State, proposal: ProposalRecord, edge: EdgeRecord
) -> tuple[bool, str | None]:
    key = proposal.sequence_key
    if not proposal.hard_valid:
        return False, "hard_invalid"
    if any(parent not in state.dag_nodes for parent in edge.parent_sequence_keys):
        return False, "unknown_parent"
    if key in edge.parent_sequence_keys or any(
        _reachable(state, key, parent) for parent in edge.parent_sequence_keys
    ):
        return False, "cycle"
    state.dag_nodes.add(key)
    state.children.setdefault(key, set())
    for parent in edge.parent_sequence_keys:
        state.children.setdefault(parent, set()).add(key)
    return True, None


def _event_payload(event: CampaignEvent, state: _State) -> tuple[str, str, dict[str, object], int]:
    if type(event) is ProposalLedgerEvent:
        return (
            "proposal",
            event.proposal.sequence_key,
            {
                "edge_record": _edge_document(event.edge),
                "proposal_record": _proposal_document(event.proposal),
                "query_disposition": event.query_disposition,
                "transposition_event": _transposition_document(event.transposition),
                "unevaluated_reason": event.unevaluated_reason,
            },
            event.scientific_elapsed_ns,
        )
    if type(event) is QueryLedgerEvent:
        return (
            "query",
            event.identity.key,
            {
                "batch_id": event.batch_id,
                "batch_position": event.batch_position,
                "call_position": state.query_count,
                "evaluator_version": event.evaluator_version,
                "fidelity": event.fidelity,
                "planned_cost": event.planned_cost,
                "proposal_id": event.proposal_id,
                "query_id": event.query_id,
                "query_identity": event.identity.document(),
            },
            event.scientific_elapsed_ns,
        )
    if type(event) is ResponseLedgerEvent:
        query = state.query_by_id.get(event.query_id)
        _require(query is not None, "response references an unknown query")
        assert query is not None
        return (
            "response",
            query["identity_key"],
            {
                "call_position": query["call_position"],
                "evaluation_record": (
                    None if event.evaluation is None else _evaluation_document(event.evaluation)
                ),
                "query_id": event.query_id,
                "response_id": event.response_id,
                "status": event.status,
                "status_detail": event.status_detail,
            },
            event.scientific_elapsed_ns,
        )
    if type(event) is RecommendationLedgerEvent:
        identity_key = ABSTENTION_IDENTITY_KEY
        if event.proposal_id is not None:
            proposal = state.proposals.get(event.proposal_id)
            _require(proposal is not None, "recommendation references an unknown proposal")
            assert proposal is not None
            identity_key = proposal.sequence_key
        return (
            "recommendation",
            identity_key,
            {
                "abstention_reason": event.abstention_reason,
                "posterior_snapshot_sha256": event.posterior_snapshot_sha256,
                "proposal_id": event.proposal_id,
                "recommendation_id": event.recommendation_id,
                "terminal_posterior_mean_utility": event.terminal_posterior_mean_utility,
            },
            event.scientific_elapsed_ns,
        )
    raise TypeError("campaign event has an unsupported exact type")


def _build_event_document(
    event: CampaignEvent,
    state: _State,
    *,
    round_index: int,
    round_position: int,
) -> dict[str, object]:
    event_type, identity_key, payload, elapsed = _event_payload(event, state)
    base = {
        "event_position": state.event_count,
        "event_type": event_type,
        "identity_key": identity_key,
        "payload": payload,
        "previous_event_sha256": state.previous_event_sha256,
        "round_index": round_index,
        "round_position": round_position,
        "schema_version": 1,
        "scientific_elapsed_ns": elapsed,
    }
    return {
        **base,
        "event_sha256": sha256_bytes(EVENT_HASH_DOMAIN + canonical_json_bytes(base)),
    }


def _process_proposal(state: _State, document: dict[str, object]) -> None:
    payload = document["payload"]
    keys = {
        "edge_record",
        "proposal_record",
        "query_disposition",
        "transposition_event",
        "unevaluated_reason",
    }
    payload = _require_exact_mapping_keys(payload, keys, label="proposal event payload differs")
    proposal = _proposal_from_document(payload["proposal_record"])
    edge = _edge_from_document(payload["edge_record"])
    transposition = _transposition_from_document(payload["transposition_event"])
    _require(
        _proposal_document(proposal) == payload["proposal_record"],
        "proposal record does not round-trip exactly",
    )
    _require(
        _edge_document(edge) == payload["edge_record"],
        "edge record does not round-trip exactly",
    )
    _require(
        _transposition_document(transposition) == payload["transposition_event"],
        "transposition record does not round-trip exactly",
    )
    _require(proposal.proposal_round == document["round_index"], "proposal round differs")
    _require(proposal.proposal_id not in state.proposals, "duplicate proposal ID")
    _require(edge.edge_id not in state.edges, "duplicate edge ID")
    _require(edge.proposal_id == proposal.proposal_id, "edge proposal ID differs")
    _require(edge.rollout_id == proposal.rollout_id, "edge rollout ID differs")
    _require(document["identity_key"] == proposal.sequence_key, "proposal identity key differs")
    first = state.first_proposal_by_sequence.get(proposal.sequence_key, proposal.proposal_id)
    duplicate = proposal.sequence_key in state.first_proposal_by_sequence
    admitted, graph_reason = _dag_admission(state, proposal, edge)
    expected_transposition = TranspositionEvent(
        event_index=state.proposal_count,
        proposal_id=proposal.proposal_id,
        edge_id=edge.edge_id,
        sequence_key=proposal.sequence_key,
        first_proposal_id=first,
        duplicate=duplicate,
        hard_valid=proposal.hard_valid,
        selected_for_evaluation=proposal.selection.selected,
        has_any_cached_evaluation=(proposal.sequence_key in state.successful_evaluation_sequences),
        dag_edge_admitted=admitted,
        graph_rejection_reason=graph_reason,
    )
    _require(transposition == expected_transposition, "transposition semantics differ")
    disposition = payload["query_disposition"]
    reason = payload["unevaluated_reason"]
    _require(
        type(disposition) is str and disposition in {"query_requested", "unevaluated", "rejected"},
        "proposal disposition differs",
    )
    if not proposal.hard_valid:
        _require(disposition == "rejected", "hard-invalid proposal must be logged rejected")
        _require(reason == proposal.rejection_reason, "rejected proposal reason differs")
    elif disposition == "query_requested":
        _require(proposal.selection.selected, "query-requested proposal was not selected")
        _require(reason is None, "query-requested proposal cannot have unevaluated reason")
    else:
        _require(disposition == "unevaluated", "valid unqueried proposal must be unevaluated")
        _identifier(reason, label="unevaluated proposal reason")
    state.proposals[proposal.proposal_id] = proposal
    state.proposal_dispositions[proposal.proposal_id] = disposition
    state.proposal_query_counts[proposal.proposal_id] = 0
    state.edges.add(edge.edge_id)
    state.first_proposal_by_sequence.setdefault(proposal.sequence_key, proposal.proposal_id)
    state.proposal_count += 1


def _process_query(state: _State, document: dict[str, object]) -> None:
    payload = document["payload"]
    keys = {
        "batch_id",
        "batch_position",
        "call_position",
        "evaluator_version",
        "fidelity",
        "planned_cost",
        "proposal_id",
        "query_id",
        "query_identity",
    }
    payload = _require_exact_mapping_keys(payload, keys, label="query event payload differs")
    query_id = _identifier(payload["query_id"], label="query ID")
    proposal_id = _identifier(payload["proposal_id"], label="query proposal ID")
    batch_id = _identifier(payload["batch_id"], label="query batch ID")
    identity = _query_identity_from_document(payload["query_identity"])
    _require(
        identity.document() == payload["query_identity"],
        "query identity does not round-trip exactly",
    )
    proposal = state.proposals.get(proposal_id)
    _require(proposal is not None, "query references an unknown proposal")
    assert proposal is not None
    _require(proposal.hard_valid and proposal.selection.selected, "query proposal is not eligible")
    _require(
        state.proposal_dispositions[proposal_id] == "query_requested",
        "query proposal was logged as unevaluated",
    )
    _require(
        identity.canonical_sequence_id == proposal.sequence_key,
        "query sequence identity differs from proposal",
    )
    _require(document["identity_key"] == identity.key, "query identity key differs")
    _require(query_id not in state.query_by_id, "duplicate query ID")
    _require(identity.key not in state.query_keys, "duplicate logical oracle query identity")
    _require(payload["call_position"] == state.query_count, "query call position is not contiguous")
    _require(state.query_count < MAX_UNIQUE_ORACLE_CALLS, "unique oracle-call ceiling exceeded")
    batch_position = _nonnegative_integer(payload["batch_position"], label="batch position")
    round_index = document["round_index"]
    previous_round = state.batch_rounds.setdefault(batch_id, round_index)
    _require(previous_round == round_index, "query batch spans sealed rounds")
    expected_batch_position = state.batch_next_positions.get(batch_id, 0)
    _require(batch_position == expected_batch_position, "query batch position is not contiguous")
    _require(
        batch_position < state.header.batch_plan.oracle_batch_size,
        "oracle batch-size ceiling exceeded",
    )
    _identifier(payload["fidelity"], label="query fidelity")
    _identifier(payload["evaluator_version"], label="query evaluator version")
    _require(_finite(payload["planned_cost"], label="query planned cost") >= 0.0, "negative cost")
    state.query_by_id[query_id] = {
        "batch_id": batch_id,
        "call_position": state.query_count,
        "evaluator_version": payload["evaluator_version"],
        "fidelity": payload["fidelity"],
        "identity": identity,
        "identity_key": identity.key,
        "planned_cost": payload["planned_cost"],
        "proposal_id": proposal_id,
    }
    state.query_keys.add(identity.key)
    state.batch_next_positions[batch_id] = batch_position + 1
    state.proposal_query_counts[proposal_id] += 1
    state.query_count += 1


def _process_response(state: _State, document: dict[str, object]) -> None:
    payload = document["payload"]
    keys = {
        "call_position",
        "evaluation_record",
        "query_id",
        "response_id",
        "status",
        "status_detail",
    }
    payload = _require_exact_mapping_keys(payload, keys, label="response event payload differs")
    query_id = _identifier(payload["query_id"], label="response query ID")
    response_id = _identifier(payload["response_id"], label="response ID")
    query = state.query_by_id.get(query_id)
    _require(query is not None, "response references an unknown query")
    assert query is not None
    _require(query_id not in state.responses, "query has duplicate responses")
    _require(response_id not in state.response_ids, "duplicate response ID")
    _require(payload["call_position"] == query["call_position"], "response call position differs")
    _require(document["identity_key"] == query["identity_key"], "response identity key differs")
    status = payload["status"]
    _require(
        type(status) is str
        and status in {"succeeded", "failed", "missing", "censored", "partial", "timeout"},
        "response status differs",
    )
    evaluation_payload = payload["evaluation_record"]
    if status == "succeeded":
        evaluation = _evaluation_from_document(evaluation_payload)
        _require(
            _evaluation_document(evaluation) == evaluation_payload,
            "evaluation record does not round-trip exactly",
        )
        _require(payload["status_detail"] is None, "successful response has failure detail")
        _require(evaluation.evaluation_id not in state.evaluation_ids, "duplicate evaluation ID")
        _require(evaluation.proposal_id == query["proposal_id"], "evaluation proposal differs")
        _require(evaluation.fidelity == query["fidelity"], "evaluation fidelity differs")
        _require(
            evaluation.evaluator_version == query["evaluator_version"],
            "evaluation version differs",
        )
        _require(evaluation.batch_id == query["batch_id"], "evaluation batch differs")
        _require(evaluation.cost == query["planned_cost"], "evaluation cost differs")
        identity = query["identity"]
        _require(evaluation.replicate == identity.replicate_id, "evaluation replicate differs")
        state.evaluation_ids.add(evaluation.evaluation_id)
        proposal = state.proposals[query["proposal_id"]]
        state.successful_evaluation_sequences.add(proposal.sequence_key)
    else:
        _require(evaluation_payload is None, "non-success response carries an evaluation")
        _identifier(payload["status_detail"], label="non-success response detail")
    state.responses.add(query_id)
    state.response_ids.add(response_id)
    state.response_count += 1


def _process_recommendation(state: _State, document: dict[str, object]) -> None:
    payload = document["payload"]
    keys = {
        "abstention_reason",
        "posterior_snapshot_sha256",
        "proposal_id",
        "recommendation_id",
        "terminal_posterior_mean_utility",
    }
    payload = _require_exact_mapping_keys(payload, keys, label="recommendation payload differs")
    _identifier(payload["recommendation_id"], label="recommendation ID")
    _sha256(payload["posterior_snapshot_sha256"], label="posterior snapshot")
    _finite(payload["terminal_posterior_mean_utility"], label="terminal utility")
    _require(
        not (set(state.query_by_id) - state.responses), "terminal recommendation has pending calls"
    )
    missing_requests = tuple(
        proposal_id
        for proposal_id, disposition in state.proposal_dispositions.items()
        if disposition == "query_requested" and state.proposal_query_counts[proposal_id] == 0
    )
    _require(not missing_requests, "terminal recommendation has unsubmitted query requests")
    proposal_id = payload["proposal_id"]
    if proposal_id is None:
        _identifier(payload["abstention_reason"], label="abstention reason")
        _require(document["identity_key"] == ABSTENTION_IDENTITY_KEY, "abstention key differs")
    else:
        proposal_id = _identifier(proposal_id, label="recommended proposal ID")
        proposal = state.proposals.get(proposal_id)
        _require(proposal is not None and proposal.hard_valid, "recommended proposal is invalid")
        assert proposal is not None
        _require(payload["abstention_reason"] is None, "selected recommendation also abstains")
        _require(document["identity_key"] == proposal.sequence_key, "recommendation key differs")
    state.recommendation_count += 1
    _require(state.recommendation_count == 1, "campaign has multiple terminal recommendations")
    state.terminal = True


def _process_event(
    state: _State,
    document: dict[str, object],
    *,
    round_index: int,
    round_position: int,
) -> None:
    keys = {
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
    _require_exact_mapping_keys(document, keys, label="campaign event schema differs")
    _require(not state.terminal, "campaign contains an event after terminal recommendation")
    _require(document["schema_version"] == 1, "campaign event version differs")
    _require(document["event_position"] == state.event_count, "event position is not contiguous")
    _require(document["round_index"] == round_index, "event round index differs")
    _require(document["round_position"] == round_position, "round position is not contiguous")
    _sha256(document["identity_key"], label="event identity key")
    _require(
        document["previous_event_sha256"] == state.previous_event_sha256,
        "event content-hash chain is broken",
    )
    base = {name: value for name, value in document.items() if name != "event_sha256"}
    expected_hash = sha256_bytes(EVENT_HASH_DOMAIN + canonical_json_bytes(base))
    _require(document["event_sha256"] == expected_hash, "event content hash differs")
    elapsed = _nonnegative_integer(document["scientific_elapsed_ns"], label="event elapsed ns")
    _require(elapsed >= state.scientific_elapsed_ns, "scientific elapsed time regressed")
    _require(elapsed <= MAX_SCIENTIFIC_ELAPSED_NS, "scientific wall-time ceiling exceeded")
    event_type = document["event_type"]
    if event_type == "proposal":
        _process_proposal(state, document)
    elif event_type == "query":
        _process_query(state, document)
    elif event_type == "response":
        _process_response(state, document)
    elif event_type == "recommendation":
        _process_recommendation(state, document)
    else:
        raise CampaignLedgerError("campaign event type differs")
    state.scientific_elapsed_ns = elapsed
    state.previous_event_sha256 = expected_hash
    state.event_count += 1


def replay_verified_campaign_event_documents(
    campaign: VerifiedCampaign,
    *,
    replay_limits: CampaignReplayLimits = EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
) -> ReplayedCampaignEvents:
    """Strictly replay an authenticated campaign's immutable event bytes.

    This detects accidental or adversarial construction of an inconsistent
    ``VerifiedCampaign`` value.  It does not authenticate the round-seal or
    timing-receipt digests independently; callers must pin those identities
    outside the campaign producer.
    """

    replay_limits = _validated_replay_limits(replay_limits)
    _require(type(campaign) is VerifiedCampaign, "campaign must use the verified type")
    _require(type(campaign.header) is CampaignHeader, "verified campaign header type differs")
    _preflight_campaign_header(campaign.header)
    _require(campaign.header_sha256 == campaign.header.sha256, "campaign header digest differs")
    _require(
        type(campaign.round_seals) is tuple and bool(campaign.round_seals),
        "campaign round-seal inventory differs",
    )
    _require(
        len(campaign.round_seals) <= replay_limits.max_rounds,
        "campaign round count exceeds replay cap",
    )
    for index, digest in enumerate(campaign.round_seals):
        _preflight_sha256(digest, label=f"campaign round seal {index}")
    _require(
        type(campaign.round_timing_receipt_sha256s) is tuple
        and len(campaign.round_timing_receipt_sha256s) == len(campaign.round_seals),
        "campaign timing-receipt inventory differs",
    )
    for index, digest in enumerate(campaign.round_timing_receipt_sha256s):
        _preflight_sha256(digest, label=f"campaign timing receipt {index}")
    _require(
        type(campaign.event_documents) is tuple and bool(campaign.event_documents),
        "campaign event-document inventory differs",
    )
    _require(
        len(campaign.event_documents) <= replay_limits.max_events,
        "campaign event count exceeds replay cap",
    )
    for field, maximum in (
        ("proposal_count", replay_limits.max_proposals),
        ("query_count", replay_limits.max_queries),
        ("response_count", replay_limits.max_responses),
    ):
        count = _nonnegative_integer(getattr(campaign, field), label=f"campaign {field}")
        _require(count <= maximum, f"campaign {field.removesuffix('_count')} replay cap exceeded")

    state = _new_state(campaign.header, replay_limits=replay_limits)
    proposals: list[ReplayedCampaignProposal] = []
    query_order: list[str] = []
    query_rows: dict[str, dict[str, object]] = {}
    expected_round = 0
    expected_round_position = 0
    for event_position, raw in enumerate(campaign.event_documents):
        _require(type(raw) is bytes, "campaign event document must be immutable bytes")
        state.cumulative_event_bytes += len(raw)
        _require(
            state.cumulative_event_bytes <= replay_limits.max_cumulative_event_bytes,
            "campaign cumulative event bytes exceed replay cap",
        )
        value = _strict_json(raw, label=f"campaign event document {event_position}")
        _require(type(value) is dict, "campaign event document must be an object")
        assert isinstance(value, dict)
        round_index = value.get("round_index")
        round_position = value.get("round_position")
        if round_index == expected_round + 1:
            _require(
                expected_round_position > 0,
                "campaign replay encountered an empty round",
            )
            expected_round += 1
            expected_round_position = 0
        _require(round_index == expected_round, "campaign replay round index is not contiguous")
        _require(
            round_position == expected_round_position,
            "campaign replay round position is not contiguous",
        )
        _process_event(
            state,
            value,
            round_index=expected_round,
            round_position=expected_round_position,
        )
        _validate_replay_state_limits(state, replay_limits)
        event_type = value["event_type"]
        payload = value["payload"]
        assert isinstance(payload, dict)
        if event_type == "proposal":
            proposal = state.proposals[payload["proposal_record"]["proposal_id"]]  # type: ignore[index]
            sequence_bytes = proposal.sequence.encode("ascii")
            proposals.append(
                ReplayedCampaignProposal(
                    event_position=event_position,
                    proposal_id=proposal.proposal_id,
                    sequence_key=proposal.sequence_key,
                    sequence_bytes=sequence_bytes,
                )
            )
        elif event_type == "query":
            query_id = payload["query_id"]
            assert isinstance(query_id, str)
            query = state.query_by_id[query_id]
            proposal = state.proposals[query["proposal_id"]]
            query_order.append(query_id)
            query_rows[query_id] = {
                "call_position": query["call_position"],
                "event_position": event_position,
                "identity": query["identity"],
                "proposal_id": query["proposal_id"],
                "sequence_bytes": proposal.sequence.encode("ascii"),
                "fidelity": query["fidelity"],
                "evaluator_version": query["evaluator_version"],
                "batch_id": query["batch_id"],
                "batch_position": payload["batch_position"],
                "planned_cost": query["planned_cost"],
            }
        elif event_type == "response":
            query_id = payload["query_id"]
            assert isinstance(query_id, str)
            query_row = query_rows[query_id]
            evaluation_payload = payload["evaluation_record"]
            outcomes: tuple[tuple[str, float], ...] | None = None
            if evaluation_payload is not None:
                evaluation = _evaluation_from_document(evaluation_payload)
                outcomes = evaluation.outcomes
            query_row.update(
                {
                    "evaluation_outcomes": outcomes,
                    "response_event_position": event_position,
                    "response_id": payload["response_id"],
                    "response_status": payload["status"],
                    "response_status_detail": payload["status_detail"],
                }
            )
        expected_round_position += 1

    _require(
        expected_round + 1 == len(campaign.round_seals),
        "campaign event rounds differ from the round-seal inventory",
    )
    _require(state.event_count == len(campaign.event_documents), "campaign event count differs")
    _require(state.proposal_count == campaign.proposal_count, "campaign proposal count differs")
    _require(state.query_count == campaign.query_count, "campaign query count differs")
    _require(state.response_count == campaign.response_count, "campaign response count differs")
    _require(
        state.previous_event_sha256 == campaign.last_event_sha256, "campaign event head differs"
    )
    _require(state.terminal is campaign.terminal, "campaign terminal status differs")
    outstanding = tuple(sorted(set(state.query_by_id) - state.responses))
    _require(outstanding == campaign.outstanding_query_ids, "campaign outstanding calls differ")
    _require(
        type(campaign.scientific_elapsed_ns) is int
        and state.scientific_elapsed_ns
        <= campaign.scientific_elapsed_ns
        <= MAX_SCIENTIFIC_ELAPSED_NS,
        "campaign scientific elapsed time differs",
    )

    queries: list[ReplayedCampaignQuery] = []
    for call_position, query_id in enumerate(query_order):
        row = query_rows[query_id]
        _require(row["call_position"] == call_position, "campaign query order differs")
        identity = row["identity"]
        _require(type(identity) is OracleQueryIdentity, "campaign replay query identity differs")
        queries.append(
            ReplayedCampaignQuery(
                event_position=row["event_position"],  # type: ignore[arg-type]
                call_position=call_position,
                query_id=query_id,
                proposal_id=row["proposal_id"],  # type: ignore[arg-type]
                identity=identity,
                sequence_bytes=row["sequence_bytes"],  # type: ignore[arg-type]
                fidelity=row["fidelity"],  # type: ignore[arg-type]
                evaluator_version=row["evaluator_version"],  # type: ignore[arg-type]
                batch_id=row["batch_id"],  # type: ignore[arg-type]
                batch_position=row["batch_position"],  # type: ignore[arg-type]
                planned_cost=row["planned_cost"],  # type: ignore[arg-type]
                response_event_position=row.get("response_event_position"),  # type: ignore[arg-type]
                response_id=row.get("response_id"),  # type: ignore[arg-type]
                response_status=row.get("response_status"),  # type: ignore[arg-type]
                response_status_detail=row.get("response_status_detail"),  # type: ignore[arg-type]
                evaluation_outcomes=row.get("evaluation_outcomes"),  # type: ignore[arg-type]
            )
        )

    event_hasher = hashlib.sha256(EVENT_DOCUMENT_INVENTORY_DOMAIN)
    for raw in campaign.event_documents:
        event_hasher.update(raw)
    event_inventory = event_hasher.hexdigest()
    proposal_hasher = hashlib.sha256(PROPOSAL_SEQUENCE_INVENTORY_DOMAIN)
    for proposal in proposals:
        proposal_hasher.update(
            canonical_json_bytes(
                {
                    "event_position": proposal.event_position,
                    "proposal_id": proposal.proposal_id,
                    "sequence_ascii_hex": proposal.sequence_bytes.hex(),
                    "sequence_key": proposal.sequence_key,
                }
            )
        )
    proposal_inventory = proposal_hasher.hexdigest()
    return ReplayedCampaignEvents(
        proposals=tuple(proposals),
        queries=tuple(queries),
        event_document_inventory_sha256=event_inventory,
        proposal_sequence_inventory_sha256=proposal_inventory,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    current = path.absolute()
    while True:
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise CampaignLedgerError(f"{label} traverses a symbolic link")
        if current.parent == current:
            return
        current = current.parent


def _trusted_directory(path: Path, *, label: str) -> tuple[Path, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(absolute, label=label)
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise CampaignLedgerError(f"{label} is unavailable") from error
    _require(
        not stat.S_ISLNK(metadata.st_mode) and stat.S_ISDIR(metadata.st_mode), f"{label} differs"
    )
    _require(metadata.st_uid == os.geteuid(), f"{label} is not owned by the current user")
    _require(stat.S_IMODE(metadata.st_mode) & 0o022 == 0, f"{label} is group/world writable")
    return absolute, metadata


def _trusted_campaign_root(root: str | Path, *, trusted_parent: str | Path) -> Path:
    parent, parent_metadata = _trusted_directory(Path(trusted_parent), label="campaign parent")
    requested, root_metadata = _trusted_directory(Path(root), label="campaign root")
    _require(requested.parent == parent, "campaign root escapes its trusted direct parent")
    observed_parent = os.lstat(requested.parent)
    _require(
        (observed_parent.st_dev, observed_parent.st_ino)
        == (parent_metadata.st_dev, parent_metadata.st_ino),
        "campaign parent identity changed",
    )
    _require(root_metadata.st_nlink >= 2, "campaign root link metadata is invalid")
    return requested


def _open_campaign_location(
    root: str | Path,
    *,
    trusted_parent: str | Path,
) -> tuple[Path, int, int, tuple[int, ...]]:
    parent, parent_metadata = _trusted_directory(Path(trusted_parent), label="campaign parent")
    requested = Path(os.path.abspath(os.fspath(root)))
    _require(requested.parent == parent, "campaign root escapes its trusted direct parent")
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_parent = os.fstat(parent_descriptor)
        _require(
            _directory_fingerprint(opened_parent) == _directory_fingerprint(parent_metadata),
            "campaign parent changed while it was pinned",
        )
        try:
            named_root = os.stat(
                requested.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require(
                not stat.S_ISLNK(named_root.st_mode) and stat.S_ISDIR(named_root.st_mode),
                "campaign root is a symbolic link or non-directory",
            )
            root_descriptor = os.open(
                requested.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise CampaignLedgerError(
                "campaign root is unavailable, changed, or a symbolic link"
            ) from error
        try:
            opened_root = os.fstat(root_descriptor)
            observed_root = os.stat(
                requested.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require(
                _directory_fingerprint(opened_root) == _directory_fingerprint(observed_root),
                "campaign root changed while it was pinned",
            )
            _require(
                opened_root.st_uid == os.geteuid()
                and stat.S_IMODE(opened_root.st_mode) & 0o022 == 0,
                "campaign root authority differs",
            )
            return (
                requested,
                parent_descriptor,
                root_descriptor,
                _directory_fingerprint(opened_root),
            )
        except BaseException:
            os.close(root_descriptor)
            raise
    except BaseException:
        os.close(parent_descriptor)
        raise


def create_campaign_root(trusted_parent: str | Path, directory_name: str) -> Path:
    """Create one direct-child campaign directory exclusively and durably."""

    parent, _metadata = _trusted_directory(Path(trusted_parent), label="campaign parent")
    _require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", directory_name) is not None,
        "campaign directory name is unsafe",
    )
    destination = parent / directory_name
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    created_fingerprint: tuple[int, ...] | None = None
    try:
        opened_parent = os.fstat(parent_descriptor)
        _require(
            (opened_parent.st_dev, opened_parent.st_ino) == (_metadata.st_dev, _metadata.st_ino),
            "campaign parent changed before root creation",
        )
        try:
            os.mkdir(directory_name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise CampaignLedgerError("refusing to reuse a campaign root") from error
        os.fsync(parent_descriptor)
        child_descriptor = os.open(
            directory_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            child = os.fstat(child_descriptor)
            observed = os.stat(directory_name, dir_fd=parent_descriptor, follow_symlinks=False)
            _require(
                _directory_fingerprint(child) == _directory_fingerprint(observed)
                and stat.S_IMODE(child.st_mode) == 0o700,
                "created campaign root identity differs",
            )
            created_fingerprint = _directory_fingerprint(child)
        finally:
            os.close(child_descriptor)
    finally:
        os.close(parent_descriptor)
    result = _trusted_campaign_root(destination, trusted_parent=parent)
    _require(
        created_fingerprint is not None
        and _directory_fingerprint(os.lstat(result)) == created_fingerprint,
        "created campaign root changed before return",
    )
    return result


def _directory_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _round_entries(
    root_descriptor: int,
    *,
    maximum_rounds: int | None = None,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if maximum_rounds is not None:
        _nonnegative_integer(maximum_rounds, label="maximum campaign rounds")
    found: list[tuple[str, tuple[int, ...]]] = []
    with os.scandir(root_descriptor) as entries:
        for entry in entries:
            if maximum_rounds is not None:
                _require(
                    len(found) < maximum_rounds,
                    "campaign round count exceeds replay cap",
                )
            metadata = entry.stat(follow_symlinks=False)
            match = _ROUND_NAME.fullmatch(entry.name)
            _require(
                match is not None and not entry.is_symlink() and stat.S_ISDIR(metadata.st_mode),
                "campaign root contains an unsealed or unexpected entry",
            )
            found.append((entry.name, _directory_fingerprint(metadata)))
    found.sort()
    expected = tuple(f"round-{index:06d}" for index in range(len(found)))
    _require(tuple(name for name, _fingerprint in found) == expected, "round positions differ")
    return tuple(found)


def _verify_pinned_round(
    root_descriptor: int,
    entry: tuple[str, tuple[int, ...]],
    *,
    maximum_events_bytes: int | None = None,
) -> PhaseSeal:
    if maximum_events_bytes is not None:
        _nonnegative_integer(maximum_events_bytes, label="remaining campaign event bytes")
    name, expected_fingerprint = entry
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        before = _directory_fingerprint(opened)
        observed = _directory_fingerprint(
            os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        )
        _require(
            before == observed == expected_fingerprint,
            "campaign round changed while it was pinned",
        )
        _require(stat.S_IMODE(opened.st_mode) == 0o555, "campaign round must have mode 0555")
        first_inventory = _round_file_inventory(descriptor)
        payloads: dict[str, bytes] = {}
        for filename in _ROUND_FILES:
            maximum_bytes = _ROUND_FILE_BOUNDS[filename]
            if filename == "events.jsonl" and maximum_events_bytes is not None:
                maximum_bytes = min(maximum_bytes, maximum_events_bytes)
            payloads[filename] = _read_round_file(
                descriptor,
                filename,
                root_descriptor=root_descriptor,
                round_name=name,
                expected_round_fingerprint=expected_fingerprint,
                expected_fingerprint=first_inventory[filename],
                maximum_bytes=maximum_bytes,
            )
        manifest = _parse_round_manifest(payloads["SHA256SUMS"])
        _require(
            tuple(sorted(manifest))
            == ("campaign.json", "events.jsonl", "receipt.json", "round.json"),
            "round checksum manifest inventory differs",
        )
        for filename, expected_sha256 in manifest.items():
            _require(
                sha256_bytes(payloads[filename]) == expected_sha256,
                f"round artifact checksum differs: {filename}",
            )
        receipt = _strict_json(payloads["receipt.json"], label=f"{name} receipt")
        receipt_keys = {
            "artifact",
            "metadata",
            "payloads",
            "predecessor_seals",
            "schema_version",
            "status",
        }
        receipt = _require_exact_mapping_keys(receipt, receipt_keys, label="round receipt differs")
        _require(receipt["artifact"] == ROUND_ARTIFACT, "round receipt artifact differs")
        _require(
            receipt["schema_version"] == 1 and receipt["status"] == "sealed",
            "receipt status differs",
        )
        receipt_payloads = receipt["payloads"]
        _require(type(receipt_payloads) is dict, "round receipt payloads differ")
        _require(
            receipt_payloads
            == {
                filename: manifest[filename]
                for filename in ("campaign.json", "events.jsonl", "round.json")
            },
            "round receipt payload hashes differ",
        )
        predecessors_raw = receipt["predecessor_seals"]
        _require(type(predecessors_raw) is dict, "round receipt predecessors differ")
        assert isinstance(predecessors_raw, dict)
        predecessors: list[tuple[str, str]] = []
        for path, digest in predecessors_raw.items():
            _require(type(path) is str, "round predecessor path differs")
            logical = validate_relative_path(path)
            predecessors.append((logical, _sha256(digest, label=f"round predecessor {logical}")))
        _require(
            tuple(predecessors) == tuple(sorted(predecessors)),
            "round receipt predecessors are not canonical",
        )
        metadata = receipt["metadata"]
        _require(type(metadata) is dict, "round receipt metadata differs")
        second_inventory = _round_file_inventory(descriptor)
        _require(first_inventory == second_inventory, "campaign round files changed during read")
        after = _directory_fingerprint(os.fstat(descriptor))
        observed_after = _directory_fingerprint(
            os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        )
        _require(
            before == after == observed_after,
            "campaign round changed while it was authenticated",
        )
        return PhaseSeal(
            artifact=ROUND_ARTIFACT,
            seal_sha256=sha256_bytes(payloads["SHA256SUMS"]),
            receipt_sha256=sha256_bytes(payloads["receipt.json"]),
            predecessor_seals=tuple(predecessors),
            payload_sha256=tuple(sorted(receipt_payloads.items())),
            payload_bytes=tuple(
                (filename, payloads[filename])
                for filename in ("campaign.json", "events.jsonl", "round.json")
            ),
            files=tuple(sorted(_ROUND_FILES)),
            metadata_json=canonical_json_bytes(metadata),
        )
    finally:
        os.close(descriptor)


def _file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _round_file_inventory(descriptor: int) -> dict[str, tuple[int, ...]]:
    inventory: dict[str, tuple[int, ...]] = {}
    with os.scandir(descriptor) as entries:
        for entry in entries:
            _require(
                len(inventory) < len(_ROUND_FILES),
                "sealed round file inventory exceeds bound",
            )
            metadata = entry.stat(follow_symlinks=False)
            _require(
                entry.name in _ROUND_FILES
                and not entry.is_symlink()
                and stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1
                and stat.S_IMODE(metadata.st_mode) == 0o444,
                "sealed round contains an unexpected or unsafe file",
            )
            _require(entry.name not in inventory, "sealed round duplicates a filename")
            inventory[entry.name] = _file_fingerprint(metadata)
    _require(
        tuple(sorted(inventory)) == tuple(sorted(_ROUND_FILES)), "round file inventory differs"
    )
    return inventory


def _require_round_read_binding(
    round_descriptor: int,
    *,
    root_descriptor: int,
    round_name: str,
    expected_round_fingerprint: tuple[int, ...],
) -> None:
    try:
        opened = os.fstat(round_descriptor)
        named = os.stat(round_name, dir_fd=root_descriptor, follow_symlinks=False)
    except OSError as error:
        raise CampaignLedgerError("campaign round binding changed during file read") from error
    _require(
        stat.S_ISDIR(opened.st_mode)
        and stat.S_IMODE(opened.st_mode) == 0o555
        and _directory_fingerprint(opened)[:6]
        == _directory_fingerprint(named)[:6]
        == expected_round_fingerprint[:6],
        "campaign round binding changed during file read",
    )


def _read_round_file(
    round_descriptor: int,
    filename: str,
    *,
    root_descriptor: int,
    round_name: str,
    expected_round_fingerprint: tuple[int, ...],
    expected_fingerprint: tuple[int, ...],
    maximum_bytes: int,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    _require(
        type(nofollow) is int and nofollow != 0,
        "O_NOFOLLOW is required for sealed round reads",
    )
    _require(
        type(nonblock) is int and nonblock != 0,
        "O_NONBLOCK is required for sealed round reads",
    )
    assert isinstance(nofollow, int) and isinstance(nonblock, int)
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0),
            dir_fd=round_descriptor,
        )
    except OSError as error:
        raise CampaignLedgerError(
            f"sealed round file could not be safely opened: {filename}"
        ) from error
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o444
            and _file_fingerprint(before) == expected_fingerprint,
            f"sealed round file changed before read: {filename}",
        )
        _require(before.st_size <= maximum_bytes, f"sealed round file exceeds bound: {filename}")
        _require_round_read_binding(
            round_descriptor,
            root_descriptor=root_descriptor,
            round_name=round_name,
            expected_round_fingerprint=expected_round_fingerprint,
        )
        chunks = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(chunks))):
            chunks.extend(chunk)
            _require(len(chunks) <= maximum_bytes, f"sealed round file exceeds bound: {filename}")
        after = os.fstat(descriptor)
        observed = os.stat(filename, dir_fd=round_descriptor, follow_symlinks=False)
        _require(
            _file_fingerprint(before) == _file_fingerprint(after) == _file_fingerprint(observed),
            f"sealed round file changed during read: {filename}",
        )
        _require_round_read_binding(
            round_descriptor,
            root_descriptor=root_descriptor,
            round_name=round_name,
            expected_round_fingerprint=expected_round_fingerprint,
        )
        return bytes(chunks)
    finally:
        os.close(descriptor)


def _parse_round_manifest(payload: bytes) -> dict[str, str]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, "round manifest differs")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise CampaignLedgerError("round manifest is not ASCII") from error
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        _require(match is not None, "round manifest row differs")
        assert match is not None
        digest, filename = match.groups()
        _require(filename not in result and filename != "SHA256SUMS", "manifest path differs")
        result[filename] = digest
    _require(checksum_manifest_bytes(result) == payload, "round manifest is not canonical")
    return result


def _require_absent(descriptor: int, name: str, *, label: str) -> None:
    try:
        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise CampaignLedgerError(f"refusing to replace existing {label}")


def _relocate_sealed_outbox_noreplace(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    expected: PhaseSeal,
) -> None:
    try:
        relocate_phase_capability_noreplace_at(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
            expected=expected,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, FileExistsError):
            raise CampaignLedgerError("refusing to replace existing campaign round") from error
        raise CampaignLedgerError("descriptor-relative campaign publication failed") from error


def _publish_round_descriptor_relative(
    *,
    campaign_root: Path,
    parent_descriptor: int,
    root_descriptor: int,
    header: CampaignHeader,
    round_index: int,
    predecessors: dict[str, str],
    rows: list[dict[str, object]],
    summary: dict[str, object],
    maximum_events_bytes: int,
) -> PhaseSeal:
    _nonnegative_integer(maximum_events_bytes, label="new round event bytes")
    destination_name = f"round-{round_index:06d}"
    root_identity = os.fstat(root_descriptor)
    outbox_key = sha256_bytes(
        f"{root_identity.st_dev}:{root_identity.st_ino}:{header.sha256}:{round_index}".encode()
    )[:20]
    outbox_name = f"campaign-outbox-{outbox_key}-{round_index:06d}"
    _require_absent(root_descriptor, destination_name, label="campaign round")
    observed_root = os.stat(
        campaign_root.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    _require(
        (observed_root.st_dev, observed_root.st_ino)
        == (root_identity.st_dev, root_identity.st_ino),
        "campaign root path no longer names the pinned root",
    )
    outbox_path = campaign_root.parent / outbox_name
    try:
        source_metadata = os.stat(outbox_name, dir_fd=parent_descriptor, follow_symlinks=False)
        path_seal: PhaseSeal | None = None
    except FileNotFoundError:
        with PhaseBuilder(
            outbox_path,
            artifact=ROUND_ARTIFACT,
            predecessor_seals=predecessors,
            metadata=_metadata_document(header, summary),
        ) as builder:
            builder.write_json("campaign.json", header.document())
            builder.write_jsonl("events.jsonl", rows)
            builder.write_json("round.json", summary)
            path_seal = builder.publish(
                expected_payload_paths=("campaign.json", "events.jsonl", "round.json")
            )
        source_metadata = os.stat(
            outbox_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    source_seal = _verify_pinned_round(
        parent_descriptor,
        (outbox_name, _directory_fingerprint(source_metadata)),
        maximum_events_bytes=maximum_events_bytes,
    )
    expected_payload_bytes = (
        ("campaign.json", canonical_json_bytes(header.document())),
        ("events.jsonl", canonical_jsonl_bytes(rows)),
        ("round.json", canonical_json_bytes(summary)),
    )
    _require(
        source_seal.predecessor_seals == tuple(sorted(predecessors.items()))
        and source_seal.payload_bytes == expected_payload_bytes
        and source_seal.metadata_json == canonical_json_bytes(_metadata_document(header, summary)),
        "pre-existing campaign outbox differs from the exact requested round",
    )
    if path_seal is not None:
        _require(source_seal == path_seal, "outbox identity differs from constructed round")
    observed_root = os.stat(
        campaign_root.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    _require(
        (observed_root.st_dev, observed_root.st_ino)
        == (root_identity.st_dev, root_identity.st_ino),
        "campaign root path changed before publication",
    )
    _relocate_sealed_outbox_noreplace(
        parent_descriptor,
        outbox_name,
        root_descriptor,
        destination_name,
        expected=source_seal,
    )
    os.fsync(root_descriptor)
    os.fsync(parent_descriptor)
    return source_seal


def _phase_payload(seal: PhaseSeal, path: str) -> bytes:
    return seal.read_payload_bytes(path)


def _round_summary(
    *,
    header: CampaignHeader,
    state_before: tuple[int, int, int, int, int, int, str],
    state_after: _State,
    round_index: int,
    previous_round_seal: str | None,
    scientific_end_elapsed_ns: int,
    timing_receipt_sha256: str,
) -> dict[str, object]:
    (
        event_start,
        proposal_start,
        query_start,
        response_start,
        recommendation_start,
        scientific_start,
        previous_event,
    ) = state_before
    return {
        "artifact": ROUND_ARTIFACT,
        "campaign_header_sha256": header.sha256,
        "counts": {
            "cumulative_events": state_after.event_count,
            "cumulative_proposals": state_after.proposal_count,
            "cumulative_queries": state_after.query_count,
            "cumulative_recommendations": state_after.recommendation_count,
            "cumulative_responses": state_after.response_count,
            "round_events": state_after.event_count - event_start,
            "round_proposals": state_after.proposal_count - proposal_start,
            "round_queries": state_after.query_count - query_start,
            "round_recommendations": state_after.recommendation_count - recommendation_start,
            "round_responses": state_after.response_count - response_start,
        },
        "event_end_exclusive": state_after.event_count,
        "event_start": event_start,
        "last_event_sha256": state_after.previous_event_sha256,
        "limits": {
            "scientific_elapsed_nanoseconds": MAX_SCIENTIFIC_ELAPSED_NS,
            "unique_oracle_calls": MAX_UNIQUE_ORACLE_CALLS,
        },
        "previous_event_sha256": previous_event,
        "previous_round_seal_sha256": previous_round_seal,
        "round_index": round_index,
        "schema_version": 1,
        "scientific_end_elapsed_ns": scientific_end_elapsed_ns,
        "scientific_segment_elapsed_ns": scientific_end_elapsed_ns - scientific_start,
        "scientific_start_elapsed_ns": scientific_start,
        "status": "sealed_complete",
        "terminal": state_after.terminal,
        "timing_receipt_sha256": timing_receipt_sha256,
    }


def _state_checkpoint(state: _State) -> tuple[int, int, int, int, int, int, str]:
    return (
        state.event_count,
        state.proposal_count,
        state.query_count,
        state.response_count,
        state.recommendation_count,
        state.scientific_elapsed_ns,
        state.previous_event_sha256,
    )


def _metadata_document(header: CampaignHeader, summary: dict[str, object]) -> dict[str, object]:
    return {
        "automatic_production_eligible": False,
        "campaign_header_sha256": header.sha256,
        "oracle_execution_authorized": False,
        "round_index": summary["round_index"],
        "scientific_evidence_accepted": False,
        "terminal": summary["terminal"],
    }


def _validate_replay_state_limits(state: _State, limits: CampaignReplayLimits) -> None:
    _require(state.event_count <= limits.max_events, "campaign event replay cap exceeded")
    _require(state.proposal_count <= limits.max_proposals, "campaign proposal replay cap exceeded")
    _require(state.query_count <= limits.max_queries, "campaign query replay cap exceeded")
    _require(state.response_count <= limits.max_responses, "campaign response replay cap exceeded")
    _require(
        state.dag_reachability_node_visits <= limits.max_dag_reachability_node_visits,
        "campaign DAG reachability node-visit replay cap exceeded",
    )
    _require(
        state.dag_reachability_edge_scans <= limits.max_dag_reachability_edge_scans,
        "campaign DAG reachability edge-scan replay cap exceeded",
    )


def _verify_campaign_open(
    root_descriptor: int,
    root_fingerprint: tuple[int, ...],
    *,
    expected_header_sha256: str | None = None,
    expected_round_count: int | None = None,
    replay_limits: CampaignReplayLimits = EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
) -> tuple[VerifiedCampaign, _State]:
    replay_limits = _validated_replay_limits(replay_limits)
    entries = _round_entries(
        root_descriptor,
        maximum_rounds=replay_limits.max_rounds,
    )
    _require(bool(entries), "campaign has no sealed rounds")
    if expected_round_count is not None:
        _nonnegative_integer(expected_round_count, label="expected round count")
        _require(
            len(entries) == expected_round_count,
            "campaign round count differs from caller-held authority",
        )
    _require(
        len(entries) <= replay_limits.max_rounds,
        "campaign round count exceeds replay cap",
    )
    header: CampaignHeader | None = None
    header_bytes: bytes | None = None
    state: _State | None = None
    round_seals: list[str] = []
    round_timing_receipt_sha256s: list[str] = []
    event_documents: list[bytes] = []
    previous_round_seal: str | None = None
    cumulative_event_bytes = 0
    for round_index, entry in enumerate(entries):
        name = entry[0]
        remaining_event_bytes = replay_limits.max_cumulative_event_bytes - cumulative_event_bytes
        seal = _verify_pinned_round(
            root_descriptor,
            entry,
            maximum_events_bytes=remaining_event_bytes,
        )
        current_header_bytes = _phase_payload(seal, "campaign.json")
        current_header_value = _strict_json(current_header_bytes, label=f"{name} campaign header")
        current_header = _header_from_document(current_header_value)
        if header is None:
            header = current_header
            header_bytes = current_header_bytes
            state = _new_state(header, replay_limits=replay_limits)
            if expected_header_sha256 is not None:
                _require(header.sha256 == expected_header_sha256, "campaign header digest differs")
        else:
            _require(current_header_bytes == header_bytes, "campaign header changed across rounds")
        assert header is not None and state is not None
        expected_predecessors = {"campaign-header.json": header.sha256}
        if previous_round_seal is not None:
            expected_predecessors[f"round-{round_index - 1:06d}/SHA256SUMS"] = previous_round_seal
        _require(
            dict(seal.predecessor_seals) == expected_predecessors,
            "round receipt predecessor chain differs",
        )
        before = _state_checkpoint(state)
        event_payload = _phase_payload(seal, "events.jsonl")
        cumulative_event_bytes += len(event_payload)
        _require(
            cumulative_event_bytes <= replay_limits.max_cumulative_event_bytes,
            "campaign cumulative event bytes exceed replay cap",
        )
        state.cumulative_event_bytes = cumulative_event_bytes
        _require(
            state.event_count + event_payload.count(b"\n") <= replay_limits.max_events,
            "campaign event count exceeds replay cap",
        )
        rows = _jsonl(event_payload, label=f"{name} events")
        _require(bool(rows), "sealed campaign round cannot be empty")
        for round_position, (document, raw) in enumerate(rows):
            _process_event(
                state,
                document,
                round_index=round_index,
                round_position=round_position,
            )
            _validate_replay_state_limits(state, replay_limits)
            event_documents.append(raw)
        summary_payload = _phase_payload(seal, "round.json")
        summary = _strict_json(summary_payload, label=f"{name} summary")
        _require(type(summary) is dict, "round summary must be an object")
        assert isinstance(summary, dict)
        end_elapsed = _nonnegative_integer(
            summary.get("scientific_end_elapsed_ns"),
            label="round scientific end elapsed ns",
        )
        _require(end_elapsed >= state.scientific_elapsed_ns, "round end precedes its last event")
        _require(end_elapsed <= MAX_SCIENTIFIC_ELAPSED_NS, "round wall-time ceiling exceeded")
        timing_digest = _sha256(summary.get("timing_receipt_sha256"), label="timing receipt")
        state.scientific_elapsed_ns = end_elapsed
        expected_summary = _round_summary(
            header=header,
            state_before=before,
            state_after=state,
            round_index=round_index,
            previous_round_seal=previous_round_seal,
            scientific_end_elapsed_ns=end_elapsed,
            timing_receipt_sha256=timing_digest,
        )
        _require(summary == expected_summary, "round summary differs from reconstructed events")
        metadata = _strict_json(seal.metadata_json, label=f"{name} receipt metadata")
        _require(metadata == _metadata_document(header, summary), "round receipt metadata differs")
        previous_round_seal = seal.seal_sha256
        round_seals.append(seal.seal_sha256)
        round_timing_receipt_sha256s.append(timing_digest)
    assert header is not None and state is not None
    verified = VerifiedCampaign(
        header=header,
        header_sha256=header.sha256,
        round_seals=tuple(round_seals),
        round_timing_receipt_sha256s=tuple(round_timing_receipt_sha256s),
        event_documents=tuple(event_documents),
        proposal_count=state.proposal_count,
        query_count=state.query_count,
        response_count=state.response_count,
        outstanding_query_ids=tuple(sorted(set(state.query_by_id) - state.responses)),
        scientific_elapsed_ns=state.scientific_elapsed_ns,
        last_event_sha256=state.previous_event_sha256,
        terminal=state.terminal,
    )
    _require(
        _directory_fingerprint(os.fstat(root_descriptor)) == root_fingerprint,
        "campaign root changed during verification",
    )
    return verified, state


def verify_campaign(
    root: str | Path,
    *,
    trusted_parent: str | Path,
    expected_header_sha256: str | None = None,
    expected_head_seal_sha256: str | None = None,
    expected_round_count: int | None = None,
    replay_limits: CampaignReplayLimits = EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
) -> VerifiedCampaign:
    """Independently authenticate and semantically replay every sealed round.

    The frozen research ceilings apply when ``replay_limits`` is omitted.
    Callers may only supply a stricter copy; ``None`` and weaker ceilings are
    rejected before the campaign path is opened.
    """

    replay_limits = _validated_replay_limits(replay_limits)
    if expected_header_sha256 is not None:
        _preflight_sha256(expected_header_sha256, label="expected campaign header")
    if expected_head_seal_sha256 is not None:
        _preflight_sha256(expected_head_seal_sha256, label="expected campaign head")
    _campaign_root, parent_descriptor, root_descriptor, root_fingerprint = _open_campaign_location(
        root, trusted_parent=trusted_parent
    )
    try:
        verified, _state = _verify_campaign_open(
            root_descriptor,
            root_fingerprint,
            expected_header_sha256=expected_header_sha256,
            expected_round_count=expected_round_count,
            replay_limits=replay_limits,
        )
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)
    if expected_head_seal_sha256 is not None:
        _require(
            verified.round_seals[-1] == expected_head_seal_sha256,
            "campaign head seal differs from caller-held authority",
        )
    return verified


def _require_resume(verified: VerifiedCampaign, authority: ResumeAuthority) -> None:
    _require(authority == verified.resume_authority(), "caller-held resume authority differs")


def _append_campaign_round_open(
    *,
    campaign_root: Path,
    parent_descriptor: int,
    root_descriptor: int,
    root_fingerprint: tuple[int, ...],
    header: CampaignHeader,
    resume_authority: ResumeAuthority,
    events: tuple[CampaignEvent, ...],
    scientific_end_elapsed_ns: int,
    timing_receipt_sha256: str,
    replay_limits: CampaignReplayLimits,
    preflight_summary: _AppendPreflightSummary,
) -> VerifiedCampaign:
    """Append one sealed round without replacing any existing byte or directory."""

    replay_limits = _validated_replay_limits(replay_limits)
    header = _preflight_campaign_header(header)
    resume_authority = _preflight_resume_authority(
        resume_authority,
        replay_limits=replay_limits,
    )
    _require(
        type(preflight_summary) is _AppendPreflightSummary
        and preflight_summary.event_count == len(events),
        "campaign append preflight summary differs",
    )
    _preflight_sha256(timing_receipt_sha256, label="timing receipt")
    end_elapsed = _nonnegative_integer(
        scientific_end_elapsed_ns,
        label="scientific end elapsed ns",
    )
    _require(end_elapsed <= MAX_SCIENTIFIC_ELAPSED_NS, "scientific wall-time ceiling exceeded")
    entries = _round_entries(root_descriptor, maximum_rounds=replay_limits.max_rounds)
    names = tuple(name for name, _fingerprint in entries)
    _require(len(names) < replay_limits.max_rounds, "campaign round count exceeds replay cap")
    if not names:
        _require(
            _directory_fingerprint(os.fstat(root_descriptor)) == root_fingerprint,
            "empty campaign root changed before append",
        )
        _require(resume_authority == EMPTY_RESUME_AUTHORITY, "genesis requires empty authority")
        state = _new_state(header, replay_limits=replay_limits)
        previous_round_seal = None
        existing_documents: tuple[bytes, ...] = ()
        existing_seals: tuple[str, ...] = ()
        existing_timing_receipts: tuple[str, ...] = ()
    else:
        verified, state = _verify_campaign_open(
            root_descriptor,
            root_fingerprint,
            expected_header_sha256=header.sha256,
            replay_limits=replay_limits,
        )
        _require_resume(verified, resume_authority)
        _require(not verified.terminal, "cannot append after terminal recommendation")
        previous_round_seal = verified.round_seals[-1]
        existing_documents = verified.event_documents
        existing_seals = verified.round_seals
        existing_timing_receipts = verified.round_timing_receipt_sha256s
    round_index = len(names)
    before = _state_checkpoint(state)
    _require(
        state.event_count + preflight_summary.event_count <= replay_limits.max_events,
        "campaign event replay cap exceeded before new round",
    )
    _require(
        state.proposal_count + preflight_summary.proposal_count <= replay_limits.max_proposals,
        "campaign proposal replay cap exceeded before new round",
    )
    _require(
        state.query_count + preflight_summary.query_count <= replay_limits.max_queries,
        "campaign query replay cap exceeded before new round",
    )
    _require(
        state.response_count + preflight_summary.response_count <= replay_limits.max_responses,
        "campaign response replay cap exceeded before new round",
    )
    _require(
        preflight_summary.proposal_count <= header.batch_plan.proposal_batch_size,
        "proposal batch-size ceiling exceeded",
    )
    nested_node_visits = 0
    for event in events:
        nested_node_visits += _preflight_campaign_event_for_append(
            event,
            maximum_nested_node_visits=(
                replay_limits.max_append_preflight_nested_node_visits - nested_node_visits
            ),
        )
    rows: list[dict[str, object]] = []
    new_round_event_bytes = 0
    for round_position, event in enumerate(events):
        document = _build_event_document(
            event,
            state,
            round_index=round_index,
            round_position=round_position,
        )
        raw_document = canonical_json_bytes(document)
        _require(
            raw_document.endswith(b"\n") and raw_document.count(b"\n") == 1,
            "campaign event canonical line framing differs",
        )
        next_cumulative_event_bytes = state.cumulative_event_bytes + len(raw_document)
        _require(
            next_cumulative_event_bytes <= replay_limits.max_cumulative_event_bytes,
            "campaign cumulative event bytes exceed replay cap before new round",
        )
        _process_event(
            state,
            document,
            round_index=round_index,
            round_position=round_position,
        )
        state.cumulative_event_bytes = next_cumulative_event_bytes
        _validate_replay_state_limits(state, replay_limits)
        new_round_event_bytes += len(raw_document)
        rows.append(document)
    _require(end_elapsed >= state.scientific_elapsed_ns, "round end precedes its last event")
    _require(end_elapsed >= before[5], "cumulative scientific clock regressed")
    state.scientific_elapsed_ns = end_elapsed
    summary = _round_summary(
        header=header,
        state_before=before,
        state_after=state,
        round_index=round_index,
        previous_round_seal=previous_round_seal,
        scientific_end_elapsed_ns=end_elapsed,
        timing_receipt_sha256=timing_receipt_sha256,
    )
    predecessors = {"campaign-header.json": header.sha256}
    if previous_round_seal is not None:
        predecessors[f"round-{round_index - 1:06d}/SHA256SUMS"] = previous_round_seal
    new_seal = _publish_round_descriptor_relative(
        campaign_root=campaign_root,
        parent_descriptor=parent_descriptor,
        root_descriptor=root_descriptor,
        header=header,
        round_index=round_index,
        predecessors=predecessors,
        rows=rows,
        summary=summary,
        maximum_events_bytes=new_round_event_bytes,
    )
    published, _published_state = _verify_campaign_open(
        root_descriptor,
        _directory_fingerprint(os.fstat(root_descriptor)),
        expected_header_sha256=header.sha256,
        replay_limits=replay_limits,
    )
    _require(published.round_seals[-1] == new_seal.seal_sha256, "published head seal differs")
    _require(len(published.round_seals) == round_index + 1, "published round count differs")
    _require(
        published.event_documents[: len(existing_documents)] == existing_documents
        and published.round_seals[: len(existing_seals)] == existing_seals
        and published.round_timing_receipt_sha256s[: len(existing_timing_receipts)]
        == existing_timing_receipts,
        "append changed a previously authenticated prefix",
    )
    return published


def append_campaign_round(
    root: str | Path,
    *,
    trusted_parent: str | Path,
    header: CampaignHeader,
    resume_authority: ResumeAuthority,
    events: tuple[CampaignEvent, ...],
    scientific_end_elapsed_ns: int,
    timing_receipt_sha256: str,
    replay_limits: CampaignReplayLimits = EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS,
) -> VerifiedCampaign:
    """Append one round under the frozen or a stricter replay ceiling."""

    replay_limits = _validated_replay_limits(replay_limits)
    header = _preflight_campaign_header(header)
    resume_authority = _preflight_resume_authority(
        resume_authority,
        replay_limits=replay_limits,
    )
    preflight_summary = _preflight_append_event_batch(events, replay_limits=replay_limits)
    _require(
        preflight_summary.proposal_count <= header.batch_plan.proposal_batch_size,
        "proposal batch-size ceiling exceeded",
    )
    _require(
        _nonnegative_integer(
            scientific_end_elapsed_ns,
            label="scientific end elapsed ns",
        )
        <= MAX_SCIENTIFIC_ELAPSED_NS,
        "scientific wall-time ceiling exceeded",
    )
    _preflight_sha256(timing_receipt_sha256, label="timing receipt")
    campaign_root, parent_descriptor, root_descriptor, root_fingerprint = _open_campaign_location(
        root, trusted_parent=trusted_parent
    )
    try:
        return _append_campaign_round_open(
            campaign_root=campaign_root,
            parent_descriptor=parent_descriptor,
            root_descriptor=root_descriptor,
            root_fingerprint=root_fingerprint,
            header=header,
            resume_authority=resume_authority,
            events=events,
            scientific_end_elapsed_ns=scientific_end_elapsed_ns,
            timing_receipt_sha256=timing_receipt_sha256,
            replay_limits=replay_limits,
            preflight_summary=preflight_summary,
        )
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)


__all__ = [
    "ABSTENTION_IDENTITY_KEY",
    "CAMPAIGN_ARTIFACT",
    "CAMPAIGN_OPERATOR_PARENT_CARDINALITIES",
    "EMPTY_RESUME_AUTHORITY",
    "EVOLUTIONARY_KL_RESEARCH_REPLAY_LIMITS",
    "MAX_SCIENTIFIC_ELAPSED_NS",
    "MAX_UNIQUE_ORACLE_CALLS",
    "CampaignHeader",
    "CampaignLedgerError",
    "CampaignReplayLimits",
    "OracleQueryIdentity",
    "ProposalLedgerEvent",
    "QueryLedgerEvent",
    "RecommendationLedgerEvent",
    "ReplayedCampaignEvents",
    "ReplayedCampaignProposal",
    "ReplayedCampaignQuery",
    "ResponseLedgerEvent",
    "ResumeAuthority",
    "VerifiedCampaign",
    "append_campaign_round",
    "create_campaign_root",
    "replay_verified_campaign_event_documents",
    "verify_campaign",
]

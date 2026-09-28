"""Authenticated yield/diversity reduction for evolutionary-KL runs.

The reducer consumes canonical, controller-pinned raw query evidence and
content-pinned contracts.  It derives string support, exact-overlap,
training-homology, organizer-reference, and complete-truth eligibility before
clustering the distinct eligible sequences.  Producer-supplied eligibility or
cluster labels are not part of the accepted schema.

This module is an evidence reducer only.  Its outputs explicitly authorize
neither execution, scientific claims, nor production use.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from rapidfuzz.distance import Indel

from amp_challenge.evaluation.evolutionary_kl_protocol import (
    CONFIRMATION_METHOD_IDS,
    CONFIRMATION_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    PRIMARY_COMPARATOR_IDS,
    CalibrationGateMetrics,
    EvolutionaryKLProtocol,
    FullMethodIntegrityEvidenceReceipt,
    IndependentResultEvidenceReceipt,
    PairedSecondaryGateMetrics,
    ResearchGateDecision,
    _validate_protocol,
    research_gate_decision,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from amp_challenge.generators.search.campaign_ledger import (
    QUERY_IDENTITY_FIELDS,
    OracleQueryIdentity,
)
from amp_challenge.sequences import canonicalize_sequence
from amp_challenge.similarity import global_sequence_identity

RAW_HEADER_ARTIFACT = "evolutionary_kl_secondary_raw_header_v1"
RAW_QUERY_ARTIFACT = "evolutionary_kl_secondary_raw_query_v1"
SUPPORT_CONTRACT_ARTIFACT = "evolutionary_kl_sequence_support_contract_v1"
ORACLE_CONTRACT_ARTIFACT = "evolutionary_kl_oracle_query_execution_contract_v1"
HOMOLOGY_CONTRACT_ARTIFACT = "evolutionary_kl_training_homology_contract_v1"
REFERENCE_CONTRACT_ARTIFACT = "evolutionary_kl_reference_compliance_contract_v1"
TRUTH_CONTRACT_ARTIFACT = "evolutionary_kl_truth_semantics_contract_v1"
SEQUENCE_SET_ROW_ARTIFACT = "evolutionary_kl_canonical_sequence_set_row_v1"
SECONDARY_EVIDENCE_ARTIFACT = "evolutionary_kl_secondary_yield_evidence_v1"
SECONDARY_COHORT_ROW_ARTIFACT = "evolutionary_kl_confirmation_secondary_cohort_row_v1"
SECONDARY_COHORT_ARTIFACT = "evolutionary_kl_confirmation_secondary_cohort_v1"
SECONDARY_UNCERTAINTY_ARTIFACT = "evolutionary_kl_confirmation_secondary_uncertainty_v1"

_SECONDARY_COHORT_HASH_DOMAIN = b"amp/evolutionary-kl/confirmation-secondary-cohort/v1\0"
_SECONDARY_UNCERTAINTY_HASH_DOMAIN = b"amp/evolutionary-kl/confirmation-secondary-uncertainty/v1\0"
_BOOTSTRAP_INDEX_HASH_DOMAIN = b"amp/evolutionary-kl/secondary-seed-bootstrap-indices/v1\0"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONSTRAINT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_PHASE = "confirmation"
_FULL_METHOD_ID = "counterfactual_softkg_evolutionary_diffusion"
_COMPLETE_STATUS = "complete"
_RESPONSE_STATUSES = frozenset({"complete", "failed", "missing", "censored", "partial", "timeout"})
_STOP_REASONS = frozenset(
    {"unique_call_budget_reached", "scientific_wall_limit_reached", "algorithmic_failure"}
)
_TRUTH_SOURCE_FIELDS = (
    "oracle_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
)
_TRUST_ANCHOR_FIELDS = (
    "run_id",
    "phase",
    "method_id",
    "seed",
    "raw_evidence_sha256",
    "query_identity_inventory_sha256",
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
_SECONDARY_METRIC_IDS = (
    "valid_unique_reference_safe_yield",
    "hill2_effective_identity70_clusters",
    "largest_identity70_cluster_share",
)
_BINDING_FIELDS = (
    "protocol_sha256",
    "phase",
    "method_id",
    "seed",
    "run_id",
    "constraint_contract_sha256",
    "oracle_contract_sha256",
    "support_contract_sha256",
    "training_sequence_set_sha256",
    "homology_contract_sha256",
    "reference_contract_sha256",
    "reference_sequence_set_sha256",
    "truth_contract_sha256",
)
_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        *_BINDING_FIELDS,
        "row_count",
        "sealed_charged_call_count",
        "unsealed_discarded_call_count",
        "stop_reason",
        "scientific_elapsed_nanoseconds",
        "query_identity_fields",
        "query_identity_inventory_sha256",
        "truth_source_identity",
        "execution_authorized",
        "scientific_claim_authorized",
        "production_authorized",
    }
)
_ROW_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "evidence_identity",
        "charged_call_position",
        "query_identity",
        "sequence",
        "response_status",
        "atomic_response_complete",
        "objectives",
        "objective_censored",
        "constraints",
    }
)
_CONSTRAINT_VALUE_KEYS = frozenset({"value", "censored"})
_CONSTRAINT_CONTRACT_KEYS = frozenset(
    {"schema_version", "artifact", "status", "protocol_sha256", "constraints"}
)
_CONSTRAINT_RULE_KEYS = frozenset({"constraint_id", "operator", "threshold"})
_SUPPORT_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "protocol_sha256",
        "alphabet",
        "min_length",
        "max_length",
        "canonical_sequence_function",
        "linear_unmodified_required",
        "free_termini_required",
    }
)
_HOMOLOGY_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "protocol_sha256",
        "training_sequence_set_sha256",
        "identity_metric",
        "exclusion_operator",
        "identity_threshold_hex",
    }
)
_REFERENCE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "protocol_sha256",
        "reference_sequence_set_sha256",
        "similarity_metric",
        "validator_commit",
        "implementation",
        "implementation_equivalence",
        "unsafe_operator",
        "max_similarity_hex",
        "post_generation_compliance_only",
    }
)
_ORACLE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "protocol_sha256",
        "constraint_contract_sha256",
        "support_contract_sha256",
        "training_sequence_set_sha256",
        "homology_contract_sha256",
        "query_identity_fields",
        "logical_call_unit",
        "all_required_endpoints_return_atomically",
        "sequence_support_contract_enforced_before_submission",
        "linear_unmodified_free_termini_attestation_authenticated",
        "exact_training_overlap_forbidden_before_submission",
        "same_identity_resubmission_forbidden",
        "failed_missing_censored_partial_timeout_submissions_charged",
    }
)
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
_OPERATORS = frozenset({"lt", "le", "gt", "ge"})
MAX_RAW_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_CONTRACT_BYTES = 1024 * 1024
MAX_SEQUENCE_SET_BYTES = 16 * 1024 * 1024
MAX_TRAINING_SEQUENCE_RECORDS = 914
MAX_REFERENCE_SEQUENCE_RECORDS = 40_000
MAX_AUXILIARY_SEQUENCE_LENGTH = 50
MAX_AUXILIARY_TOTAL_RESIDUES = 2_000_000
MAX_HOMOLOGY_ALIGNMENT_CELLS = 80_000_000
MAX_REFERENCE_COMPARISONS = 10_000_000
MAX_REFERENCE_COMPARISON_CHARACTER_WORK = 400_000_000
MAX_IDENTITY70_ALIGNMENT_CELLS = 40_000_000


class SecondaryEvidenceError(ValueError):
    """Raised when secondary raw evidence is unauthenticated or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SecondaryEvidenceError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} invalid")
    assert isinstance(value, str)
    return value


@dataclass(frozen=True, slots=True)
class SecondaryEvidenceTrustAnchors:
    """Controller-owned identities supplied out of band from producer bytes."""

    run_id: str
    phase: str
    method_id: str
    seed: int
    raw_evidence_sha256: str
    query_identity_inventory_sha256: str
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
        _require(
            type(self.run_id) is str and _IDENTIFIER_RE.fullmatch(self.run_id) is not None,
            "trusted run ID invalid",
        )
        _require(self.phase == _PHASE and type(self.phase) is str, "trusted phase differs")
        _require(
            type(self.method_id) is str and _IDENTIFIER_RE.fullmatch(self.method_id) is not None,
            "trusted method ID invalid",
        )
        _require(type(self.seed) is int and self.seed >= 0, "trusted seed invalid")
        digest_values = tuple(
            getattr(self, field)
            for field in (
                "raw_evidence_sha256",
                "query_identity_inventory_sha256",
                "constraint_contract_sha256",
                "support_contract_sha256",
                "training_sequence_set_sha256",
                "homology_contract_sha256",
                "reference_contract_sha256",
                "reference_sequence_set_sha256",
                "truth_contract_sha256",
                *_TRUTH_SOURCE_FIELDS,
            )
        )
        for index, digest in enumerate(digest_values):
            _sha256(digest, label=f"trusted digest {index}")
        _require(len(set(digest_values)) == len(digest_values), "trusted digest roles alias")


@dataclass(frozen=True, slots=True)
class SecondaryYieldEvidence:
    """Non-authorizing, self-authenticating derived yield/diversity evidence."""

    run_id: str
    phase: str
    method_id: str
    seed: int
    raw_evidence_sha256: str
    query_identity_inventory_sha256: str
    constraint_contract_sha256: str
    oracle_contract_sha256: str
    support_contract_sha256: str
    training_sequence_set_sha256: str
    homology_contract_sha256: str
    reference_contract_sha256: str
    reference_sequence_set_sha256: str
    truth_contract_sha256: str
    truth_source_identity: tuple[tuple[str, str], ...]
    charged_submitted_identity_count: int
    sealed_charged_call_count: int
    unsealed_discarded_call_count: int
    stop_reason: str
    scientific_elapsed_nanoseconds: int
    valid_unique_sequences: tuple[str, ...]
    identity70_components: tuple[tuple[str, ...], ...]
    valid_unique_reference_safe_yield: float
    hill2_effective_identity70_clusters: float
    largest_identity70_cluster_share: float
    evidence_sha256: str
    execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    production_authorized: bool = False

    def __post_init__(self) -> None:
        _validate_evidence(self)

    @property
    def valid_unique_count(self) -> int:
        return len(self.valid_unique_sequences)

    @property
    def identity70_cluster_sizes(self) -> tuple[int, ...]:
        return tuple(len(component) for component in self.identity70_components)

    def document_bytes(self) -> bytes:
        """Serialize only after recomputing all derived fields and the self-hash."""

        _validate_evidence(self)
        return canonical_json_bytes(_evidence_document(self, include_digest=True))


@dataclass(frozen=True, slots=True)
class ConfirmationSecondaryEvidenceCohortRow:
    """One identity-retaining row in the exact 15-run confirmation inventory."""

    phase: str
    method_id: str
    seed: int
    run_id: str
    evidence_sha256: str
    evidence: SecondaryYieldEvidence
    trust_anchors: SecondaryEvidenceTrustAnchors

    def __post_init__(self) -> None:
        _validate_confirmation_secondary_row(self)


@dataclass(frozen=True, slots=True)
class ConfirmationSecondaryEvidenceCohort:
    """Sealed, non-authorizing inventory of all 15 confirmation run records."""

    rows: tuple[ConfirmationSecondaryEvidenceCohortRow, ...]
    cohort_sha256: str
    execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    production_authorized: bool = False

    def __post_init__(self) -> None:
        _validate_confirmation_secondary_cohort(self)

    def document_bytes(self) -> bytes:
        """Serialize after revalidating every retained row and the cohort seal."""

        _validate_confirmation_secondary_cohort(self)
        return canonical_json_bytes(
            _confirmation_secondary_cohort_document(self, include_digest=True)
        )


@dataclass(frozen=True, slots=True)
class SecondaryPairedBootstrapSummary:
    """Descriptive full-minus-comparator effects over the exact five seed blocks."""

    metric_id: str
    seed_effects: tuple[float, ...]
    arithmetic_mean_effect: float
    percentile_interval: tuple[float, float]

    def __post_init__(self) -> None:
        _require(self.metric_id in _SECONDARY_METRIC_IDS, "secondary uncertainty metric differs")
        _require(
            type(self.seed_effects) is tuple
            and len(self.seed_effects) == len(CONFIRMATION_SEEDS)
            and all(type(value) is float and math.isfinite(value) for value in self.seed_effects),
            "secondary uncertainty seed effects differ",
        )
        expected_mean = math.fsum(self.seed_effects) / len(self.seed_effects)
        _require(
            type(self.arithmetic_mean_effect) is float
            and self.arithmetic_mean_effect == expected_mean,
            "secondary uncertainty mean differs",
        )
        _require(
            type(self.percentile_interval) is tuple
            and len(self.percentile_interval) == 2
            and all(
                type(value) is float and math.isfinite(value) for value in self.percentile_interval
            )
            and self.percentile_interval[0] <= self.percentile_interval[1],
            "secondary uncertainty interval differs",
        )


@dataclass(frozen=True, slots=True)
class SecondaryComparatorPairedUncertainty:
    """Three descriptive paired summaries for one primary comparator."""

    comparator_id: str
    summaries: tuple[SecondaryPairedBootstrapSummary, ...]

    def __post_init__(self) -> None:
        _require(
            self.comparator_id in PRIMARY_COMPARATOR_IDS,
            "secondary uncertainty comparator differs",
        )
        _require(
            type(self.summaries) is tuple
            and all(type(summary) is SecondaryPairedBootstrapSummary for summary in self.summaries),
            "secondary uncertainty summaries differ",
        )
        _require(
            tuple(summary.metric_id for summary in self.summaries) == _SECONDARY_METRIC_IDS,
            "secondary uncertainty summary order differs",
        )


@dataclass(frozen=True, slots=True)
class ConfirmationSecondaryPairedUncertainty:
    """Non-authorizing paired seed-block bootstrap over the three observed methods."""

    protocol_sha256: str
    cohort_sha256: str
    observed_method_ids: tuple[str, ...]
    seed_ids: tuple[int, ...]
    bootstrap_unit: str
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_statistic: str
    shared_bootstrap_index_sha256: str
    interval_type: str
    interval_quantiles: tuple[float, float]
    quantile_convention: str
    comparators: tuple[SecondaryComparatorPairedUncertainty, ...]
    uncertainty_sha256: str
    execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    production_authorized: bool = False

    def __post_init__(self) -> None:
        _validate_confirmation_secondary_uncertainty(self)

    def document_bytes(self) -> bytes:
        """Serialize this descriptive record after revalidating its self-seal."""

        _validate_confirmation_secondary_uncertainty(self)
        return canonical_json_bytes(
            _confirmation_secondary_uncertainty_document(self, include_digest=True)
        )


@dataclass(frozen=True, slots=True)
class _ConstraintRule:
    constraint_id: str
    operator: str
    threshold: float


@dataclass(frozen=True, slots=True)
class _ParsedRow:
    position: int
    identity: OracleQueryIdentity
    sequence: str | None
    forbidden_support_or_overlap: bool
    eligible: bool


def _exact_object(value: object, keys: frozenset[str], *, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(
        all(type(key) is str for key in value) and set(value) == set(keys), f"{label} keys differ"
    )
    return value


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    _require(type(payload) is bytes, f"{label} must be bytes")
    _require(0 < len(payload) <= MAX_CONTRACT_BYTES, f"{label} byte bound exceeded")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SecondaryEvidenceError(f"{label} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise SecondaryEvidenceError(f"{label} contains invalid constant {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SecondaryEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise SecondaryEvidenceError(f"{label} is not strict UTF-8 JSON") from error
    _require(type(parsed) is dict, f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, ValueError) as error:
        raise SecondaryEvidenceError(f"{label} is not finite canonical JSON") from error
    _require(canonical == payload, f"{label} is not canonical JSON")
    assert isinstance(parsed, dict)
    return parsed


def _strict_jsonl(
    payload: bytes,
    *,
    label: str,
    max_bytes: int,
    max_rows: int,
) -> tuple[dict[str, object], ...]:
    _require(type(payload) is bytes and bool(payload), f"{label} must be nonempty bytes")
    _require(len(payload) <= max_bytes, f"{label} byte bound exceeded")
    _require(payload.endswith(b"\n"), f"{label} must end in LF")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(payload.splitlines(keepends=True), start=1):
        _require(index <= max_rows, f"{label} row bound exceeded")
        _require(line not in {b"", b"\n"}, f"{label} contains a blank row")
        rows.append(_strict_json_object(line, label=f"{label} row {index}"))
    return tuple(rows)


def _pinned_document(payload: bytes, expected_sha256: str, *, label: str) -> dict[str, object]:
    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
    _require(sha256_bytes(payload) == expected, f"{label} digest differs")
    return _strict_json_object(payload, label=label)


def _finite_number(value: object, *, label: str) -> float:
    _require(type(value) in {int, float}, f"{label} must be a non-Boolean real number")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise SecondaryEvidenceError(f"{label} must be finite") from error
    _require(math.isfinite(parsed), f"{label} must be finite")
    return parsed


def _optional_finite_number(value: object, *, label: str) -> float | None:
    return None if value is None else _finite_number(value, label=label)


def _hex_float(value: object, *, label: str) -> float:
    _require(type(value) is str, f"{label} must be a hexadecimal float")
    assert isinstance(value, str)
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise SecondaryEvidenceError(f"{label} must be a hexadecimal float") from error
    _require(math.isfinite(parsed) and parsed.hex() == value, f"{label} is not canonical")
    return parsed


def _sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _parse_sequence_set(payload: bytes, expected_sha256: str, *, label: str) -> tuple[str, ...]:
    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
    _require(sha256_bytes(payload) == expected, f"{label} digest differs")
    max_rows = (
        MAX_TRAINING_SEQUENCE_RECORDS
        if label == "training sequence set"
        else MAX_REFERENCE_SEQUENCE_RECORDS
    )
    rows = _strict_jsonl(
        payload,
        label=label,
        max_bytes=MAX_SEQUENCE_SET_BYTES,
        max_rows=max_rows,
    )
    parsed: list[tuple[str, str]] = []
    for index, raw in enumerate(rows):
        row = _exact_object(raw, _SEQUENCE_SET_ROW_KEYS, label=f"{label} row {index}")
        _require(
            row["schema_version"] == 1 and type(row["schema_version"]) is int,
            f"{label} schema differs",
        )
        _require(row["artifact"] == SEQUENCE_SET_ROW_ARTIFACT, f"{label} artifact differs")
        sequence = row["sequence"]
        _require(type(sequence) is str, f"{label} sequence must be text")
        assert isinstance(sequence, str)
        try:
            canonical = canonicalize_sequence(
                sequence,
                min_length=1,
                max_length=MAX_AUXILIARY_SEQUENCE_LENGTH,
            )
        except (TypeError, ValueError) as error:
            raise SecondaryEvidenceError(f"{label} contains an invalid sequence") from error
        _require(canonical == sequence, f"{label} sequence is not canonical")
        sequence_id = _sha256(row["sequence_id"], label=f"{label} sequence ID")
        _require(sequence_id == _sequence_digest(sequence), f"{label} sequence ID differs")
        parsed.append((sequence_id, sequence))
    _require(bool(parsed), f"{label} cannot be empty")
    _require(parsed == sorted(set(parsed)), f"{label} must be sorted and unique")
    _require(
        sum(len(sequence) for _sequence_id, sequence in parsed) <= MAX_AUXILIARY_TOTAL_RESIDUES,
        f"{label} total-residue bound exceeded",
    )
    return tuple(sequence for _sequence_id, sequence in parsed)


def _parse_constraint_contract(payload: bytes, expected_sha256: str) -> tuple[_ConstraintRule, ...]:
    document = _exact_object(
        _pinned_document(payload, expected_sha256, label="constraint contract"),
        _CONSTRAINT_CONTRACT_KEYS,
        label="constraint contract",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "constraint contract schema differs",
    )
    _require(
        document["artifact"] == "evolutionary_kl_constraint_semantics_v1",
        "constraint contract artifact differs",
    )
    _require(document["status"] == "accepted_content_pinned", "constraint contract is not accepted")
    _require(
        document["protocol_sha256"] == FROZEN_PROTOCOL_SHA256,
        "constraint contract protocol differs",
    )
    raw_rules = document["constraints"]
    _require(type(raw_rules) is list and bool(raw_rules), "constraint rules must be nonempty")
    rules: list[_ConstraintRule] = []
    assert isinstance(raw_rules, list)
    for index, raw in enumerate(raw_rules):
        row = _exact_object(raw, _CONSTRAINT_RULE_KEYS, label=f"constraint rule {index}")
        constraint_id = row["constraint_id"]
        operator = row["operator"]
        _require(
            type(constraint_id) is str and _CONSTRAINT_ID_RE.fullmatch(constraint_id) is not None,
            f"constraint rule {index} ID invalid",
        )
        _require(
            type(operator) is str and operator in _OPERATORS,
            f"constraint rule {index} operator invalid",
        )
        assert isinstance(constraint_id, str) and isinstance(operator, str)
        rules.append(
            _ConstraintRule(
                constraint_id,
                operator,
                _finite_number(row["threshold"], label=f"constraint rule {index} threshold"),
            )
        )
    _require(
        tuple(rule.constraint_id for rule in rules)
        == tuple(sorted({rule.constraint_id for rule in rules})),
        "constraint rules must be sorted and unique",
    )
    return tuple(rules)


def _rule_passes(rule: _ConstraintRule, value: float) -> bool:
    if rule.operator == "lt":
        return value < rule.threshold
    if rule.operator == "le":
        return value <= rule.threshold
    if rule.operator == "gt":
        return value > rule.threshold
    if rule.operator == "ge":
        return value >= rule.threshold
    raise AssertionError("validated operator became unreachable")


def _parse_support_contract(
    protocol: EvolutionaryKLProtocol,
    payload: bytes,
    expected_sha256: str,
) -> None:
    document = _exact_object(
        _pinned_document(payload, expected_sha256, label="support contract"),
        _SUPPORT_CONTRACT_KEYS,
        label="support contract",
    )
    expected = {
        "schema_version": 1,
        "artifact": SUPPORT_CONTRACT_ARTIFACT,
        "status": "accepted_content_pinned",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "alphabet": protocol.support_alphabet,
        "min_length": protocol.support_min_length,
        "max_length": protocol.support_max_length,
        "canonical_sequence_function": "amp_challenge.sequences.canonicalize_sequence",
        "linear_unmodified_required": True,
        "free_termini_required": True,
    }
    _require(document == expected, "support contract differs from the frozen support domain")


def _parse_homology_contract(
    payload: bytes,
    expected_sha256: str,
    *,
    training_sequence_set_sha256: str,
) -> float:
    document = _exact_object(
        _pinned_document(payload, expected_sha256, label="homology contract"),
        _HOMOLOGY_CONTRACT_KEYS,
        label="homology contract",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "homology contract schema differs",
    )
    _require(
        document["artifact"] == HOMOLOGY_CONTRACT_ARTIFACT, "homology contract artifact differs"
    )
    _require(document["status"] == "accepted_content_pinned", "homology contract is not accepted")
    _require(
        document["protocol_sha256"] == FROZEN_PROTOCOL_SHA256, "homology contract protocol differs"
    )
    _require(
        document["training_sequence_set_sha256"] == training_sequence_set_sha256,
        "homology contract training set differs",
    )
    _require(
        document["identity_metric"] == "amp_challenge.similarity.global_sequence_identity",
        "homology identity metric differs",
    )
    _require(document["exclusion_operator"] == "ge", "homology exclusion operator differs")
    threshold = _hex_float(document["identity_threshold_hex"], label="homology threshold")
    _require(0.0 < threshold <= 1.0, "homology threshold lies outside (0, 1]")
    return threshold


def _parse_reference_contract(
    protocol: EvolutionaryKLProtocol,
    payload: bytes,
    expected_sha256: str,
    *,
    reference_sequence_set_sha256: str,
) -> float:
    document = _exact_object(
        _pinned_document(payload, expected_sha256, label="reference contract"),
        _REFERENCE_CONTRACT_KEYS,
        label="reference contract",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "reference contract schema differs",
    )
    _require(
        document["artifact"] == REFERENCE_CONTRACT_ARTIFACT, "reference contract artifact differs"
    )
    _require(document["status"] == "accepted_content_pinned", "reference contract is not accepted")
    _require(
        document["protocol_sha256"] == FROZEN_PROTOCOL_SHA256, "reference contract protocol differs"
    )
    _require(
        document["reference_sequence_set_sha256"] == reference_sequence_set_sha256,
        "reference contract set differs",
    )
    _require(document["similarity_metric"] == "Levenshtein.ratio", "reference metric differs")
    _require(
        document["similarity_metric"] == protocol.organizer_reference_similarity,
        "reference metric differs from protocol",
    )
    _require(
        document["validator_commit"] == protocol.organizer_reference_validator_commit,
        "reference validator commit differs",
    )
    _require(
        document["implementation"] == "rapidfuzz.distance.Indel.normalized_similarity",
        "reference implementation differs",
    )
    _require(
        document["implementation_equivalence"]
        == "RapidFuzz_Indel_normalized_similarity_equals_Levenshtein_ratio",
        "reference implementation equivalence declaration differs",
    )
    _require(document["unsafe_operator"] == "gt", "reference comparison operator differs")
    _require(document["post_generation_compliance_only"] is True, "reference role differs")
    threshold = _hex_float(document["max_similarity_hex"], label="reference threshold")
    _require(
        threshold == protocol.organizer_reference_max_similarity,
        "reference threshold differs from protocol",
    )
    return threshold


def _parse_oracle_contract(
    protocol: EvolutionaryKLProtocol,
    payload: bytes,
    anchors: SecondaryEvidenceTrustAnchors,
) -> None:
    document = _exact_object(
        _pinned_document(payload, anchors.oracle_contract_sha256, label="oracle contract"),
        _ORACLE_CONTRACT_KEYS,
        label="oracle contract",
    )
    expected = {
        "schema_version": 1,
        "artifact": ORACLE_CONTRACT_ARTIFACT,
        "status": "accepted_content_pinned",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "constraint_contract_sha256": anchors.constraint_contract_sha256,
        "support_contract_sha256": anchors.support_contract_sha256,
        "training_sequence_set_sha256": anchors.training_sequence_set_sha256,
        "homology_contract_sha256": anchors.homology_contract_sha256,
        "query_identity_fields": list(QUERY_IDENTITY_FIELDS),
        "logical_call_unit": "first_submission_of_one_exact_query_identity_within_one_run",
        "all_required_endpoints_return_atomically": True,
        "sequence_support_contract_enforced_before_submission": True,
        "linear_unmodified_free_termini_attestation_authenticated": True,
        "exact_training_overlap_forbidden_before_submission": True,
        "same_identity_resubmission_forbidden": True,
        "failed_missing_censored_partial_timeout_submissions_charged": True,
    }
    _require(document == expected, "oracle contract differs from required query semantics")
    _require(
        protocol.oracle_query_contract.logical_call_unit == expected["logical_call_unit"]
        and protocol.oracle_query_contract.all_required_endpoints_return_atomically
        and protocol.oracle_query_contract.same_identity_resubmission_after_any_submission_is_forbidden
        and protocol.oracle_query_contract.failed_submitted_call_is_charged
        and protocol.oracle_query_contract.missing_submitted_response_is_charged
        and protocol.oracle_query_contract.censored_submitted_response_is_charged
        and protocol.oracle_query_contract.partial_submitted_response_is_charged_and_ineligible
        and protocol.oracle_query_contract.timeout_after_submission_is_charged
        and not protocol.oracle_query_contract.failed_or_ineligible_call_may_be_replaced_without_charge
        and protocol.exact_training_overlap_forbidden,
        "oracle contract differs from frozen protocol semantics",
    )


def _parse_truth_contract(
    protocol: EvolutionaryKLProtocol,
    payload: bytes,
    expected_sha256: str,
    *,
    constraint_contract_sha256: str,
    oracle_contract_sha256: str,
) -> None:
    document = _exact_object(
        _pinned_document(payload, expected_sha256, label="truth contract"),
        _TRUTH_CONTRACT_KEYS,
        label="truth contract",
    )
    expected = {
        "schema_version": 1,
        "artifact": TRUTH_CONTRACT_ARTIFACT,
        "status": "accepted_content_pinned",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "constraint_contract_sha256": constraint_contract_sha256,
        "oracle_contract_sha256": oracle_contract_sha256,
        "objective_ids": list(protocol.primary_objectives),
        "objective_bounds_hex": [value.hex() for value in protocol.objective_bounds],
        "complete_response_status": _COMPLETE_STATUS,
        "complete_requires_atomic_all_finite_uncensored": True,
        "noncomplete_is_ineligible": True,
        "constraint_pass_recomputed_from_raw_value": True,
    }
    _require(document == expected, "truth contract differs from the frozen truth semantics")


def _expected_binding(anchors: SecondaryEvidenceTrustAnchors) -> dict[str, object]:
    return {
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "phase": anchors.phase,
        "method_id": anchors.method_id,
        "seed": anchors.seed,
        "run_id": anchors.run_id,
        "constraint_contract_sha256": anchors.constraint_contract_sha256,
        "oracle_contract_sha256": anchors.oracle_contract_sha256,
        "support_contract_sha256": anchors.support_contract_sha256,
        "training_sequence_set_sha256": anchors.training_sequence_set_sha256,
        "homology_contract_sha256": anchors.homology_contract_sha256,
        "reference_contract_sha256": anchors.reference_contract_sha256,
        "reference_sequence_set_sha256": anchors.reference_sequence_set_sha256,
        "truth_contract_sha256": anchors.truth_contract_sha256,
    }


def _truth_source_identity(anchors: SecondaryEvidenceTrustAnchors) -> dict[str, str]:
    return {field: getattr(anchors, field) for field in _TRUTH_SOURCE_FIELDS}


def _parse_query_identity(
    value: object,
    *,
    row_number: int,
    sequence: str,
    anchors: SecondaryEvidenceTrustAnchors,
) -> OracleQueryIdentity:
    identity = _exact_object(
        value, frozenset(QUERY_IDENTITY_FIELDS), label=f"query row {row_number} identity"
    )
    truth_source = _truth_source_identity(anchors)
    for field in QUERY_IDENTITY_FIELDS[:-1]:
        item = identity[field]
        _require(
            type(item) is str and bool(item), f"query row {row_number} identity {field} invalid"
        )
        assert isinstance(item, str)
        if field.endswith("sha256") or field == "canonical_sequence_id":
            _sha256(item, label=f"query row {row_number} identity {field}")
        if field == "canonical_sequence_id":
            _require(
                item == _sequence_digest(sequence),
                f"query row {row_number} sequence identity differs",
            )
        elif field in truth_source:
            _require(item == truth_source[field], f"query row {row_number} truth source differs")
    try:
        parsed = OracleQueryIdentity(**identity)
    except (TypeError, ValueError) as error:
        raise SecondaryEvidenceError(f"query row {row_number} ledger identity differs") from error
    _require(parsed.document() == identity, f"query row {row_number} ledger identity differs")
    return parsed


def query_identity_inventory_sha256(
    identities: tuple[OracleQueryIdentity, ...],
) -> str:
    """Hash the canonical sorted inventory of exact logical-query identities."""

    _require(type(identities) is tuple and bool(identities), "query inventory must be nonempty")
    encoded: list[tuple[str, dict[str, object]]] = []
    for identity in identities:
        _require(type(identity) is OracleQueryIdentity, "query identity must use ledger type")
        identity.__post_init__()
        document = identity.document()
        encoded.append((identity.key, document))
    encoded.sort(key=lambda item: item[0])
    _require(
        len({key for key, _document in encoded}) == len(encoded), "query identity is duplicated"
    )
    return sha256_bytes(
        b"amp/evolutionary-kl/secondary-query-inventory/v1\0"
        + canonical_jsonl_bytes(document for _key, document in encoded)
    )


def _homology_eligible(
    sequence: str,
    training_sequences: tuple[str, ...],
    *,
    threshold: float,
    work: list[int],
) -> bool:
    for training in training_sequences:
        upper_bound = min(len(sequence), len(training)) / max(len(sequence), len(training))
        if upper_bound < threshold:
            continue
        work[0] += len(sequence) * len(training)
        _require(work[0] <= MAX_HOMOLOGY_ALIGNMENT_CELLS, "homology alignment work bound exceeded")
        if global_sequence_identity(sequence, training) >= threshold:
            return False
    return True


def _reference_safe(
    sequence: str,
    reference_sequences: tuple[str, ...],
    *,
    threshold: float,
    work: list[int],
) -> bool:
    for reference in reference_sequences:
        upper_bound = 2.0 * min(len(sequence), len(reference)) / (len(sequence) + len(reference))
        if upper_bound <= threshold:
            continue
        work[0] += 1
        _require(
            work[0] <= MAX_REFERENCE_COMPARISONS,
            "reference comparison-count bound exceeded",
        )
        work[1] += len(sequence) + len(reference)
        _require(
            work[1] <= MAX_REFERENCE_COMPARISON_CHARACTER_WORK,
            "reference character-work bound exceeded",
        )
        similarity = float(Indel.normalized_similarity(sequence, reference, score_cutoff=threshold))
        if similarity > threshold:
            return False
    return True


def _parse_raw_query_row(
    raw: dict[str, object],
    *,
    row_number: int,
    anchors: SecondaryEvidenceTrustAnchors,
    protocol: EvolutionaryKLProtocol,
    constraint_rules: tuple[_ConstraintRule, ...],
    training_sequence_set: frozenset[str],
    training_sequences: tuple[str, ...],
    homology_threshold: float,
    reference_sequences: tuple[str, ...],
    reference_threshold: float,
    homology_cache: dict[str, bool],
    reference_cache: dict[str, bool],
    homology_work: list[int],
    reference_work: list[int],
) -> _ParsedRow:
    row = _exact_object(raw, _ROW_KEYS, label=f"query row {row_number}")
    _require(
        row["schema_version"] == 1 and type(row["schema_version"]) is int,
        f"query row {row_number} schema differs",
    )
    _require(row["artifact"] == RAW_QUERY_ARTIFACT, f"query row {row_number} artifact differs")
    binding = _exact_object(
        row["evidence_identity"],
        frozenset(_BINDING_FIELDS),
        label=f"query row {row_number} evidence identity",
    )
    _require(
        binding == _expected_binding(anchors), f"query row {row_number} evidence identity differs"
    )
    position = row["charged_call_position"]
    _require(
        type(position) is int and position == row_number,
        f"query row {row_number} charged position differs",
    )
    sequence_value = row["sequence"]
    _require(type(sequence_value) is str, f"query row {row_number} sequence must be text")
    assert isinstance(sequence_value, str)
    try:
        sequence = canonicalize_sequence(
            sequence_value,
            min_length=protocol.support_min_length,
            max_length=protocol.support_max_length,
        )
    except (TypeError, ValueError) as error:
        raise SecondaryEvidenceError(
            f"query row {row_number} is outside canonical support"
        ) from error
    _require(sequence == sequence_value, f"query row {row_number} sequence is not canonical")
    identity = _parse_query_identity(
        row["query_identity"],
        row_number=row_number,
        sequence=sequence,
        anchors=anchors,
    )
    exact_overlap = sequence in training_sequence_set

    objective_ids = tuple(protocol.primary_objectives)
    objectives = _exact_object(
        row["objectives"], frozenset(objective_ids), label=f"query row {row_number} objectives"
    )
    censoring = _exact_object(
        row["objective_censored"],
        frozenset(objective_ids),
        label=f"query row {row_number} objective censoring",
    )
    objective_values: list[float | None] = []
    for objective in objective_ids:
        value = _optional_finite_number(
            objectives[objective], label=f"query row {row_number} objective {objective}"
        )
        if value is not None:
            _require(
                protocol.objective_bounds[0] <= value <= protocol.objective_bounds[1],
                f"query row {row_number} objective is outside bounds",
            )
        _require(
            type(censoring[objective]) is bool or censoring[objective] is None,
            f"query row {row_number} objective censoring differs",
        )
        objective_values.append(value)

    constraint_ids = tuple(rule.constraint_id for rule in constraint_rules)
    constraints = _exact_object(
        row["constraints"], frozenset(constraint_ids), label=f"query row {row_number} constraints"
    )
    constraints_complete = True
    constraints_uncensored = True
    constraints_pass = True
    for rule in constraint_rules:
        item = _exact_object(
            constraints[rule.constraint_id],
            _CONSTRAINT_VALUE_KEYS,
            label=f"query row {row_number} constraint {rule.constraint_id}",
        )
        value = _optional_finite_number(
            item["value"], label=f"query row {row_number} constraint {rule.constraint_id}"
        )
        censored = item["censored"]
        _require(
            type(censored) is bool or censored is None,
            f"query row {row_number} constraint censoring differs",
        )
        constraints_complete &= value is not None
        constraints_uncensored &= censored is False
        constraints_pass &= value is not None and _rule_passes(rule, value)

    status = row["response_status"]
    atomic = row["atomic_response_complete"]
    _require(
        type(status) is str and status in _RESPONSE_STATUSES,
        f"query row {row_number} status differs",
    )
    _require(type(atomic) is bool, f"query row {row_number} atomic flag differs")
    objectives_complete = all(value is not None for value in objective_values)
    objectives_uncensored = all(censoring[objective] is False for objective in objective_ids)
    if status == _COMPLETE_STATUS:
        _require(
            atomic is True
            and objectives_complete
            and objectives_uncensored
            and constraints_complete
            and constraints_uncensored,
            f"query row {row_number} complete response is incomplete",
        )
    else:
        _require(
            atomic is False, f"query row {row_number} noncomplete response claims atomic truth"
        )
    truth_complete = (
        status == _COMPLETE_STATUS
        and atomic is True
        and objectives_complete
        and objectives_uncensored
        and constraints_complete
        and constraints_uncensored
        and constraints_pass
    )
    forbidden = exact_overlap
    homology_eligible = False
    reference_safe = False
    if truth_complete and not forbidden:
        if sequence not in homology_cache:
            homology_cache[sequence] = _homology_eligible(
                sequence,
                training_sequences,
                threshold=homology_threshold,
                work=homology_work,
            )
        homology_eligible = homology_cache[sequence]
        if homology_eligible:
            if sequence not in reference_cache:
                reference_cache[sequence] = _reference_safe(
                    sequence,
                    reference_sequences,
                    threshold=reference_threshold,
                    work=reference_work,
                )
            reference_safe = reference_cache[sequence]
    return _ParsedRow(
        position=position,
        identity=identity,
        sequence=sequence,
        forbidden_support_or_overlap=forbidden,
        eligible=(not forbidden and truth_complete and homology_eligible and reference_safe),
    )


def _identity70_components(sequences: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    _require(len(sequences) <= 512, "identity-70 clustering sequence bound exceeded")
    parent = list(range(len(sequences)))
    alignment_cells = 0

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for left_index, right_index in combinations(range(len(sequences)), 2):
        left = sequences[left_index]
        right = sequences[right_index]
        if min(len(left), len(right)) / max(len(left), len(right)) < 0.70:
            continue
        alignment_cells += len(left) * len(right)
        _require(
            alignment_cells <= MAX_IDENTITY70_ALIGNMENT_CELLS,
            "identity-70 alignment work bound exceeded",
        )
        if global_sequence_identity(left, right) >= 0.70:
            union(left_index, right_index)
    groups: dict[int, list[str]] = {}
    for index, sequence in enumerate(sequences):
        groups.setdefault(find(index), []).append(sequence)
    components = tuple(tuple(sorted(group)) for group in groups.values())
    return tuple(sorted(components, key=lambda component: component[0]))


def _yield_diversity(
    valid_sequences: tuple[str, ...],
    charged_count: int,
    components: tuple[tuple[str, ...], ...],
) -> tuple[float, float, float]:
    valid_count = len(valid_sequences)
    _require(charged_count > 0, "zero charged identity count is a missing gate")
    yield_value = valid_count / charged_count
    if valid_count == 0:
        return yield_value, 0.0, 0.0
    proportions = tuple(len(component) / valid_count for component in components)
    hill2 = 1.0 / math.fsum(proportion * proportion for proportion in proportions)
    largest = max(len(component) for component in components) / valid_count
    return yield_value, hill2, largest


def _protocol_is_supported(protocol: EvolutionaryKLProtocol) -> bool:
    if type(protocol) is not EvolutionaryKLProtocol:
        return False
    try:
        # A dataclass instance can be constructed or copied without going
        # through the digest-verifying loader.  Re-run the protocol's complete
        # frozen semantic validator before reading any reducer-facing fields.
        _validate_protocol(protocol)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    promotion = protocol.promotion_gate
    return (
        protocol.confirmation_method_ids == CONFIRMATION_METHOD_IDS
        and protocol.confirmation_seeds == CONFIRMATION_SEEDS
        and protocol.oracle_query_contract.query_identity_fields == QUERY_IDENTITY_FIELDS
        and protocol.initial_design_unique_calls == 64
        and protocol.unique_calls_per_batch == 16
        and protocol.adaptive_batches == 28
        and protocol.total_unique_calls == 512
        and protocol.scientific_wall_seconds == 7200
        and protocol.stopping_rules.stop_at_unique_calls_or_wall_seconds_whichever_first
        and protocol.stopping_rules.discard_unsealed_partial_batch
        and promotion.identity70_metric == "amp_challenge.similarity.global_sequence_identity"
        and promotion.identity70_threshold == 0.70
        and promotion.identity70_linkage
        == "connected_components_single_linkage_edges_at_or_above_threshold"
        and promotion.yield_zero_charged_identity_value == "missing_required_gate_failure"
    )


def compute_authenticated_secondary_yield(
    protocol: EvolutionaryKLProtocol,
    raw_evidence_bytes: bytes,
    *,
    trust_anchors: SecondaryEvidenceTrustAnchors,
    constraint_contract_bytes: bytes,
    oracle_contract_bytes: bytes,
    support_contract_bytes: bytes,
    training_sequence_set_bytes: bytes,
    homology_contract_bytes: bytes,
    reference_contract_bytes: bytes,
    reference_sequence_set_bytes: bytes,
    truth_contract_bytes: bytes,
) -> SecondaryYieldEvidence:
    """Recompute one confirmation run's frozen yield/diversity evidence.

    Every digest in ``trust_anchors`` must come from a controller channel that
    is independent of the producer bytes.  Passing a digest copied from the
    supplied evidence is not authentication.
    """

    _require(_protocol_is_supported(protocol), "protocol secondary evidence domain differs")
    _require(
        type(trust_anchors) is SecondaryEvidenceTrustAnchors,
        "trust anchors must use the exact frozen type",
    )
    trust_anchors.__post_init__()
    _require(
        trust_anchors.method_id in protocol.confirmation_method_ids,
        "trusted method is outside the confirmation cohort",
    )
    _require(
        trust_anchors.seed in protocol.confirmation_seeds,
        "trusted seed is outside the confirmation cohort",
    )
    _require(type(raw_evidence_bytes) is bytes, "raw evidence must be immutable bytes")
    _require(
        0 < len(raw_evidence_bytes) <= MAX_RAW_EVIDENCE_BYTES,
        "raw evidence byte bound exceeded",
    )
    _require(
        sha256_bytes(raw_evidence_bytes) == trust_anchors.raw_evidence_sha256,
        "raw evidence differs from the out-of-band digest",
    )
    for label, payload in (
        ("constraint contract", constraint_contract_bytes),
        ("oracle contract", oracle_contract_bytes),
        ("support contract", support_contract_bytes),
        ("homology contract", homology_contract_bytes),
        ("reference contract", reference_contract_bytes),
        ("truth contract", truth_contract_bytes),
    ):
        _require(
            type(payload) is bytes and len(payload) <= MAX_CONTRACT_BYTES,
            f"{label} byte bound exceeded",
        )

    constraint_rules = _parse_constraint_contract(
        constraint_contract_bytes,
        trust_anchors.constraint_contract_sha256,
    )
    _parse_support_contract(
        protocol,
        support_contract_bytes,
        trust_anchors.support_contract_sha256,
    )
    _parse_oracle_contract(protocol, oracle_contract_bytes, trust_anchors)
    training_sequences = _parse_sequence_set(
        training_sequence_set_bytes,
        trust_anchors.training_sequence_set_sha256,
        label="training sequence set",
    )
    homology_threshold = _parse_homology_contract(
        homology_contract_bytes,
        trust_anchors.homology_contract_sha256,
        training_sequence_set_sha256=trust_anchors.training_sequence_set_sha256,
    )
    reference_sequences = _parse_sequence_set(
        reference_sequence_set_bytes,
        trust_anchors.reference_sequence_set_sha256,
        label="reference sequence set",
    )
    reference_threshold = _parse_reference_contract(
        protocol,
        reference_contract_bytes,
        trust_anchors.reference_contract_sha256,
        reference_sequence_set_sha256=trust_anchors.reference_sequence_set_sha256,
    )
    _parse_truth_contract(
        protocol,
        truth_contract_bytes,
        trust_anchors.truth_contract_sha256,
        constraint_contract_sha256=trust_anchors.constraint_contract_sha256,
        oracle_contract_sha256=trust_anchors.oracle_contract_sha256,
    )

    documents = _strict_jsonl(
        raw_evidence_bytes,
        label="raw secondary evidence",
        max_bytes=MAX_RAW_EVIDENCE_BYTES,
        max_rows=protocol.total_unique_calls + 1,
    )
    _require(
        len(documents) >= protocol.initial_design_unique_calls + 1,
        "raw evidence misses the initial design",
    )
    header = _exact_object(documents[0], _HEADER_KEYS, label="raw evidence header")
    _require(
        header["schema_version"] == 1 and type(header["schema_version"]) is int,
        "raw evidence header schema differs",
    )
    _require(header["artifact"] == RAW_HEADER_ARTIFACT, "raw evidence header artifact differs")
    _require(
        header["status"] == "controller_digest_pinned_raw_query_evidence",
        "raw evidence header status differs",
    )
    for field, expected in _expected_binding(trust_anchors).items():
        _require(
            header[field] == expected and type(header[field]) is type(expected),
            f"raw evidence header {field} differs",
        )
    _require(
        header["query_identity_fields"] == list(QUERY_IDENTITY_FIELDS),
        "raw evidence identity fields differ",
    )
    _require(
        header["query_identity_inventory_sha256"] == trust_anchors.query_identity_inventory_sha256,
        "raw evidence inventory identity differs",
    )
    truth_source = _exact_object(
        header["truth_source_identity"],
        frozenset(_TRUTH_SOURCE_FIELDS),
        label="raw evidence truth source",
    )
    _require(
        truth_source == _truth_source_identity(trust_anchors), "raw evidence truth source differs"
    )
    _require(
        header["execution_authorized"] is False
        and header["scientific_claim_authorized"] is False
        and header["production_authorized"] is False,
        "raw evidence cannot authorize execution, claims, or production",
    )

    row_count = header["row_count"]
    sealed_count = header["sealed_charged_call_count"]
    discarded_count = header["unsealed_discarded_call_count"]
    stop_reason = header["stop_reason"]
    elapsed = header["scientific_elapsed_nanoseconds"]
    _require(
        type(row_count) is int and row_count == len(documents) - 1, "raw evidence row count differs"
    )
    _require(
        type(sealed_count) is int and sealed_count in protocol.call_checkpoints,
        "sealed charged-call count differs",
    )
    _require(
        type(discarded_count) is int
        and discarded_count == row_count - sealed_count
        and 0 <= discarded_count <= protocol.unique_calls_per_batch,
        "unsealed discarded-call count differs",
    )
    _require(sealed_count <= row_count <= protocol.total_unique_calls, "charged-call count differs")
    _require(type(stop_reason) is str and stop_reason in _STOP_REASONS, "stop reason differs")
    wall_nanoseconds = protocol.scientific_wall_seconds * 1_000_000_000
    _require(type(elapsed) is int and elapsed >= 0, "scientific elapsed time differs")
    if stop_reason == "unique_call_budget_reached":
        _require(
            row_count == sealed_count == protocol.total_unique_calls,
            "budget stop did not reach 512 sealed calls",
        )
        _require(elapsed <= wall_nanoseconds, "budget stop occurred after the wall limit")
    elif stop_reason == "scientific_wall_limit_reached":
        _require(
            sealed_count < protocol.total_unique_calls and elapsed == wall_nanoseconds,
            "wall stop semantics differ",
        )
    else:
        _require(
            sealed_count < protocol.total_unique_calls and elapsed < wall_nanoseconds,
            "algorithmic stop semantics differ",
        )

    training_set = frozenset(training_sequences)
    homology_cache: dict[str, bool] = {}
    reference_cache: dict[str, bool] = {}
    homology_work = [0]
    reference_work = [0, 0]
    rows = tuple(
        _parse_raw_query_row(
            raw,
            row_number=index,
            anchors=trust_anchors,
            protocol=protocol,
            constraint_rules=constraint_rules,
            training_sequence_set=training_set,
            training_sequences=training_sequences,
            homology_threshold=homology_threshold,
            reference_sequences=reference_sequences,
            reference_threshold=reference_threshold,
            homology_cache=homology_cache,
            reference_cache=reference_cache,
            homology_work=homology_work,
            reference_work=reference_work,
        )
        for index, raw in enumerate(documents[1:], start=1)
    )
    identities = tuple(row.identity for row in rows)
    _require(len(set(identities)) == len(identities), "logical query identity is duplicated")
    observed_inventory = query_identity_inventory_sha256(identities)
    _require(
        observed_inventory == trust_anchors.query_identity_inventory_sha256,
        "query identity inventory digest differs",
    )
    _require(
        not any(row.forbidden_support_or_overlap for row in rows),
        "a forbidden support/overlap query was submitted",
    )

    valid_sequences = tuple(
        sorted({row.sequence for row in rows[:sealed_count] if row.eligible and row.sequence})
    )
    components = _identity70_components(valid_sequences)
    yield_value, hill2, largest = _yield_diversity(valid_sequences, row_count, components)
    truth_identity = tuple((field, getattr(trust_anchors, field)) for field in _TRUTH_SOURCE_FIELDS)
    unsigned = _unsigned_evidence_document(
        trust_anchors=trust_anchors,
        truth_source_identity=truth_identity,
        charged_count=row_count,
        sealed_count=sealed_count,
        discarded_count=discarded_count,
        stop_reason=stop_reason,
        elapsed=elapsed,
        valid_sequences=valid_sequences,
        components=components,
        yield_value=yield_value,
        hill2=hill2,
        largest=largest,
    )
    evidence_sha256 = sha256_bytes(canonical_json_bytes(unsigned))
    return SecondaryYieldEvidence(
        run_id=trust_anchors.run_id,
        phase=trust_anchors.phase,
        method_id=trust_anchors.method_id,
        seed=trust_anchors.seed,
        raw_evidence_sha256=trust_anchors.raw_evidence_sha256,
        query_identity_inventory_sha256=observed_inventory,
        constraint_contract_sha256=trust_anchors.constraint_contract_sha256,
        oracle_contract_sha256=trust_anchors.oracle_contract_sha256,
        support_contract_sha256=trust_anchors.support_contract_sha256,
        training_sequence_set_sha256=trust_anchors.training_sequence_set_sha256,
        homology_contract_sha256=trust_anchors.homology_contract_sha256,
        reference_contract_sha256=trust_anchors.reference_contract_sha256,
        reference_sequence_set_sha256=trust_anchors.reference_sequence_set_sha256,
        truth_contract_sha256=trust_anchors.truth_contract_sha256,
        truth_source_identity=truth_identity,
        charged_submitted_identity_count=row_count,
        sealed_charged_call_count=sealed_count,
        unsealed_discarded_call_count=discarded_count,
        stop_reason=stop_reason,
        scientific_elapsed_nanoseconds=elapsed,
        valid_unique_sequences=valid_sequences,
        identity70_components=components,
        valid_unique_reference_safe_yield=yield_value,
        hill2_effective_identity70_clusters=hill2,
        largest_identity70_cluster_share=largest,
        evidence_sha256=evidence_sha256,
    )


def _unsigned_evidence_document(
    *,
    trust_anchors: SecondaryEvidenceTrustAnchors,
    truth_source_identity: tuple[tuple[str, str], ...],
    charged_count: int,
    sealed_count: int,
    discarded_count: int,
    stop_reason: str,
    elapsed: int,
    valid_sequences: tuple[str, ...],
    components: tuple[tuple[str, ...], ...],
    yield_value: float,
    hill2: float,
    largest: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": SECONDARY_EVIDENCE_ARTIFACT,
        "status": "derived_non_authorizing_evidence",
        **_expected_binding(trust_anchors),
        "raw_evidence_sha256": trust_anchors.raw_evidence_sha256,
        "query_identity_inventory_sha256": trust_anchors.query_identity_inventory_sha256,
        "truth_source_identity": dict(truth_source_identity),
        "charged_submitted_identity_count": charged_count,
        "sealed_charged_call_count": sealed_count,
        "unsealed_discarded_call_count": discarded_count,
        "stop_reason": stop_reason,
        "scientific_elapsed_nanoseconds": elapsed,
        "valid_unique_count": len(valid_sequences),
        "valid_unique_sequences": list(valid_sequences),
        "identity70_components": [list(component) for component in components],
        "identity70_cluster_sizes": [len(component) for component in components],
        "valid_unique_reference_safe_yield_hex": yield_value.hex(),
        "hill2_effective_identity70_clusters_hex": hill2.hex(),
        "largest_identity70_cluster_share_hex": largest.hex(),
        "execution_authorized": False,
        "scientific_claim_authorized": False,
        "production_authorized": False,
    }


def _anchors_from_evidence(evidence: SecondaryYieldEvidence) -> SecondaryEvidenceTrustAnchors:
    truth_source = dict(evidence.truth_source_identity)
    return SecondaryEvidenceTrustAnchors(
        run_id=evidence.run_id,
        phase=evidence.phase,
        method_id=evidence.method_id,
        seed=evidence.seed,
        raw_evidence_sha256=evidence.raw_evidence_sha256,
        query_identity_inventory_sha256=evidence.query_identity_inventory_sha256,
        constraint_contract_sha256=evidence.constraint_contract_sha256,
        oracle_contract_sha256=evidence.oracle_contract_sha256,
        support_contract_sha256=evidence.support_contract_sha256,
        training_sequence_set_sha256=evidence.training_sequence_set_sha256,
        homology_contract_sha256=evidence.homology_contract_sha256,
        reference_contract_sha256=evidence.reference_contract_sha256,
        reference_sequence_set_sha256=evidence.reference_sequence_set_sha256,
        truth_contract_sha256=evidence.truth_contract_sha256,
        evaluator_sha256=truth_source.get("evaluator_sha256", ""),
        checkpoint_sha256=truth_source.get("checkpoint_sha256", ""),
        endpoint_context_sha256=truth_source.get("endpoint_context_sha256", ""),
        transform_sha256=truth_source.get("transform_sha256", ""),
    )


def _evidence_document(
    evidence: SecondaryYieldEvidence,
    *,
    include_digest: bool,
) -> dict[str, object]:
    anchors = _anchors_from_evidence(evidence)
    document = _unsigned_evidence_document(
        trust_anchors=anchors,
        truth_source_identity=evidence.truth_source_identity,
        charged_count=evidence.charged_submitted_identity_count,
        sealed_count=evidence.sealed_charged_call_count,
        discarded_count=evidence.unsealed_discarded_call_count,
        stop_reason=evidence.stop_reason,
        elapsed=evidence.scientific_elapsed_nanoseconds,
        valid_sequences=evidence.valid_unique_sequences,
        components=evidence.identity70_components,
        yield_value=evidence.valid_unique_reference_safe_yield,
        hill2=evidence.hill2_effective_identity70_clusters,
        largest=evidence.largest_identity70_cluster_share,
    )
    if include_digest:
        document["evidence_sha256"] = evidence.evidence_sha256
    return document


def _validate_evidence(evidence: SecondaryYieldEvidence) -> None:
    _require(type(evidence) is SecondaryYieldEvidence, "secondary evidence type differs")
    anchors = _anchors_from_evidence(evidence)
    _require(evidence.phase == _PHASE, "secondary evidence phase differs")
    _require(evidence.method_id in CONFIRMATION_METHOD_IDS, "secondary evidence method differs")
    _require(evidence.seed in CONFIRMATION_SEEDS, "secondary evidence seed differs")
    _sha256(evidence.evidence_sha256, label="secondary evidence self-hash")
    _require(
        evidence.evidence_sha256
        not in {
            evidence.raw_evidence_sha256,
            evidence.query_identity_inventory_sha256,
            evidence.constraint_contract_sha256,
            evidence.oracle_contract_sha256,
            evidence.support_contract_sha256,
            evidence.training_sequence_set_sha256,
            evidence.homology_contract_sha256,
            evidence.reference_contract_sha256,
            evidence.reference_sequence_set_sha256,
            evidence.truth_contract_sha256,
        },
        "secondary evidence self-hash aliases an input role",
    )
    _require(
        type(evidence.truth_source_identity) is tuple
        and tuple(field for field, _value in evidence.truth_source_identity) == _TRUTH_SOURCE_FIELDS
        and dict(evidence.truth_source_identity) == _truth_source_identity(anchors),
        "secondary evidence truth source differs",
    )
    _require(
        type(evidence.charged_submitted_identity_count) is int
        and 64 <= evidence.charged_submitted_identity_count <= 512,
        "secondary evidence charged count differs",
    )
    _require(
        type(evidence.sealed_charged_call_count) is int
        and evidence.sealed_charged_call_count in tuple(range(64, 513, 16)),
        "secondary evidence sealed count differs",
    )
    _require(
        type(evidence.unsealed_discarded_call_count) is int
        and evidence.unsealed_discarded_call_count
        == evidence.charged_submitted_identity_count - evidence.sealed_charged_call_count
        and 0 <= evidence.unsealed_discarded_call_count <= 16,
        "secondary evidence discarded count differs",
    )
    _require(evidence.stop_reason in _STOP_REASONS, "secondary evidence stop reason differs")
    _require(
        type(evidence.scientific_elapsed_nanoseconds) is int
        and evidence.scientific_elapsed_nanoseconds >= 0,
        "secondary evidence elapsed time differs",
    )
    wall = 7_200_000_000_000
    if evidence.stop_reason == "unique_call_budget_reached":
        _require(
            evidence.charged_submitted_identity_count == evidence.sealed_charged_call_count == 512
            and evidence.scientific_elapsed_nanoseconds <= wall,
            "secondary evidence budget-stop semantics differ",
        )
    elif evidence.stop_reason == "scientific_wall_limit_reached":
        _require(
            evidence.sealed_charged_call_count < 512
            and evidence.scientific_elapsed_nanoseconds == wall,
            "secondary evidence wall-stop semantics differ",
        )
    else:
        _require(
            evidence.sealed_charged_call_count < 512
            and evidence.scientific_elapsed_nanoseconds < wall,
            "secondary evidence algorithmic-stop semantics differ",
        )
    _require(
        type(evidence.valid_unique_sequences) is tuple
        and evidence.valid_unique_sequences == tuple(sorted(set(evidence.valid_unique_sequences)))
        and len(evidence.valid_unique_sequences) <= evidence.sealed_charged_call_count,
        "secondary evidence valid sequence census differs",
    )
    for sequence in evidence.valid_unique_sequences:
        _require(type(sequence) is str, "secondary evidence sequence must be text")
        try:
            canonical = canonicalize_sequence(sequence, min_length=8, max_length=50)
        except (TypeError, ValueError) as error:
            raise SecondaryEvidenceError(
                "secondary evidence sequence is outside support"
            ) from error
        _require(canonical == sequence, "secondary evidence sequence is not canonical")
    expected_components = _identity70_components(evidence.valid_unique_sequences)
    _require(
        type(evidence.identity70_components) is tuple
        and evidence.identity70_components == expected_components,
        "secondary evidence identity-70 components differ",
    )
    expected_yield, expected_hill2, expected_largest = _yield_diversity(
        evidence.valid_unique_sequences,
        evidence.charged_submitted_identity_count,
        expected_components,
    )
    _require(
        type(evidence.valid_unique_reference_safe_yield) is float
        and evidence.valid_unique_reference_safe_yield == expected_yield,
        "secondary evidence yield differs",
    )
    _require(
        type(evidence.hill2_effective_identity70_clusters) is float
        and evidence.hill2_effective_identity70_clusters == expected_hill2,
        "secondary evidence Hill-2 differs",
    )
    _require(
        type(evidence.largest_identity70_cluster_share) is float
        and evidence.largest_identity70_cluster_share == expected_largest,
        "secondary evidence largest-cluster share differs",
    )
    _require(
        evidence.execution_authorized is False
        and evidence.scientific_claim_authorized is False
        and evidence.production_authorized is False,
        "secondary evidence cannot authorize execution, claims, or production",
    )
    observed = sha256_bytes(
        canonical_json_bytes(_evidence_document(evidence, include_digest=False))
    )
    _require(observed == evidence.evidence_sha256, "secondary evidence self-hash differs")


def revalidate_secondary_yield_evidence(
    evidence: SecondaryYieldEvidence,
    *,
    trust_anchors: SecondaryEvidenceTrustAnchors,
    expected_evidence_sha256: str,
) -> SecondaryYieldEvidence:
    """Revalidate one derived record against controller-owned identities."""

    _require(type(evidence) is SecondaryYieldEvidence, "secondary evidence type differs")
    _require(type(trust_anchors) is SecondaryEvidenceTrustAnchors, "trust-anchor type differs")
    trust_anchors.__post_init__()
    _validate_evidence(evidence)
    expected = _sha256(expected_evidence_sha256, label="expected secondary evidence SHA-256")
    _require(evidence.evidence_sha256 == expected, "secondary evidence digest differs")
    _require(
        _anchors_from_evidence(evidence) == trust_anchors, "secondary evidence trust anchors differ"
    )
    return evidence


def _paired_secondary_gate_metrics(
    protocol: EvolutionaryKLProtocol,
    *,
    full_evidence: SecondaryYieldEvidence,
    full_trust_anchors: SecondaryEvidenceTrustAnchors,
    expected_full_evidence_sha256: str,
    comparator_evidence: SecondaryYieldEvidence,
    comparator_trust_anchors: SecondaryEvidenceTrustAnchors,
    expected_comparator_evidence_sha256: str,
) -> PairedSecondaryGateMetrics:
    """Create one non-authoritative count row after two records revalidate.

    ``PairedSecondaryGateMetrics`` deliberately carries no method, seed, run,
    or evidence identity.  It is therefore suitable only as internal input to
    the pure gate math.  Callers must use the exact-cohort research wrapper,
    never this return value, as an evidence boundary.
    """

    _require(_protocol_is_supported(protocol), "protocol secondary evidence domain differs")
    full = revalidate_secondary_yield_evidence(
        full_evidence,
        trust_anchors=full_trust_anchors,
        expected_evidence_sha256=expected_full_evidence_sha256,
    )
    comparator = revalidate_secondary_yield_evidence(
        comparator_evidence,
        trust_anchors=comparator_trust_anchors,
        expected_evidence_sha256=expected_comparator_evidence_sha256,
    )
    _require(
        full.method_id == "counterfactual_softkg_evolutionary_diffusion",
        "full secondary evidence method differs",
    )
    _require(
        comparator.method_id in protocol.primary_comparator_ids,
        "secondary evidence comparator differs",
    )
    _require(full.phase == comparator.phase == _PHASE, "paired secondary phases differ")
    _require(full.seed == comparator.seed, "paired secondary seeds differ")
    common_fields = (
        "constraint_contract_sha256",
        "oracle_contract_sha256",
        "support_contract_sha256",
        "training_sequence_set_sha256",
        "homology_contract_sha256",
        "reference_contract_sha256",
        "reference_sequence_set_sha256",
        "truth_contract_sha256",
        "truth_source_identity",
    )
    _require(
        all(getattr(full, field) == getattr(comparator, field) for field in common_fields),
        "paired secondary semantic identities differ",
    )
    _require(full.run_id != comparator.run_id, "paired secondary runs must be distinct")
    return PairedSecondaryGateMetrics(
        full_valid_unique_count=full.valid_unique_count,
        full_charged_submitted_identity_count=full.charged_submitted_identity_count,
        full_identity70_cluster_sizes=full.identity70_cluster_sizes,
        comparator_valid_unique_count=comparator.valid_unique_count,
        comparator_charged_submitted_identity_count=(comparator.charged_submitted_identity_count),
        comparator_identity70_cluster_sizes=comparator.identity70_cluster_sizes,
    )


def confirmation_secondary_cohort_row(
    evidence: SecondaryYieldEvidence,
    *,
    trust_anchors: SecondaryEvidenceTrustAnchors,
) -> ConfirmationSecondaryEvidenceCohortRow:
    """Retain one evidence object and all of its externally assigned identity."""

    _require(type(evidence) is SecondaryYieldEvidence, "secondary cohort evidence type differs")
    _require(
        type(trust_anchors) is SecondaryEvidenceTrustAnchors,
        "secondary cohort trust-anchor type differs",
    )
    return ConfirmationSecondaryEvidenceCohortRow(
        phase=evidence.phase,
        method_id=evidence.method_id,
        seed=evidence.seed,
        run_id=evidence.run_id,
        evidence_sha256=evidence.evidence_sha256,
        evidence=evidence,
        trust_anchors=trust_anchors,
    )


def _trust_anchor_document(anchors: SecondaryEvidenceTrustAnchors) -> dict[str, object]:
    _require(type(anchors) is SecondaryEvidenceTrustAnchors, "cohort trust-anchor type differs")
    anchors.__post_init__()
    return {field: getattr(anchors, field) for field in _TRUST_ANCHOR_FIELDS}


def _confirmation_secondary_row_document(
    row: ConfirmationSecondaryEvidenceCohortRow,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": SECONDARY_COHORT_ROW_ARTIFACT,
        "phase": row.phase,
        "method_id": row.method_id,
        "seed": row.seed,
        "run_id": row.run_id,
        "evidence_sha256": row.evidence_sha256,
        "trust_anchors": _trust_anchor_document(row.trust_anchors),
    }


def _validate_confirmation_secondary_row(row: ConfirmationSecondaryEvidenceCohortRow) -> None:
    _require(
        type(row) is ConfirmationSecondaryEvidenceCohortRow,
        "secondary cohort row type differs",
    )
    _require(type(row.evidence) is SecondaryYieldEvidence, "secondary cohort evidence type differs")
    _require(
        type(row.trust_anchors) is SecondaryEvidenceTrustAnchors,
        "secondary cohort trust-anchor type differs",
    )
    _require(row.phase == _PHASE and type(row.phase) is str, "secondary cohort phase differs")
    _require(row.method_id in CONFIRMATION_METHOD_IDS, "secondary cohort method differs")
    _require(
        type(row.seed) is int and row.seed in CONFIRMATION_SEEDS, "secondary cohort seed differs"
    )
    _require(
        type(row.run_id) is str and _IDENTIFIER_RE.fullmatch(row.run_id) is not None,
        "secondary cohort run ID differs",
    )
    _sha256(row.evidence_sha256, label="secondary cohort evidence digest")
    _require(
        (row.phase, row.method_id, row.seed, row.run_id, row.evidence_sha256)
        == (
            row.evidence.phase,
            row.evidence.method_id,
            row.evidence.seed,
            row.evidence.run_id,
            row.evidence.evidence_sha256,
        ),
        "secondary cohort row differs from retained evidence identity",
    )
    _require(
        (row.phase, row.method_id, row.seed, row.run_id)
        == (
            row.trust_anchors.phase,
            row.trust_anchors.method_id,
            row.trust_anchors.seed,
            row.trust_anchors.run_id,
        ),
        "secondary cohort row differs from retained trust identity",
    )
    revalidate_secondary_yield_evidence(
        row.evidence,
        trust_anchors=row.trust_anchors,
        expected_evidence_sha256=row.evidence_sha256,
    )


def _confirmation_secondary_cohort_unsigned_document(
    rows: tuple[ConfirmationSecondaryEvidenceCohortRow, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": SECONDARY_COHORT_ARTIFACT,
        "status": "derived_non_authorizing_exact_confirmation_inventory",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "phase": _PHASE,
        "observed_method_ids": list(CONFIRMATION_METHOD_IDS),
        "seed_ids": list(CONFIRMATION_SEEDS),
        "row_count": len(rows),
        "rows": [_confirmation_secondary_row_document(row) for row in rows],
        "execution_authorized": False,
        "scientific_claim_authorized": False,
        "production_authorized": False,
    }


def _confirmation_secondary_cohort_document(
    cohort: ConfirmationSecondaryEvidenceCohort,
    *,
    include_digest: bool,
) -> dict[str, object]:
    document = _confirmation_secondary_cohort_unsigned_document(cohort.rows)
    if include_digest:
        document["cohort_sha256"] = cohort.cohort_sha256
    return document


def _confirmation_secondary_cohort_digest(
    rows: tuple[ConfirmationSecondaryEvidenceCohortRow, ...],
) -> str:
    return sha256_bytes(
        _SECONDARY_COHORT_HASH_DOMAIN
        + canonical_json_bytes(_confirmation_secondary_cohort_unsigned_document(rows))
    )


def _validate_confirmation_secondary_cohort(
    cohort: ConfirmationSecondaryEvidenceCohort,
) -> None:
    _require(
        type(cohort) is ConfirmationSecondaryEvidenceCohort,
        "secondary cohort type differs",
    )
    _require(
        type(cohort.rows) is tuple
        and len(cohort.rows) == len(CONFIRMATION_METHOD_IDS) * len(CONFIRMATION_SEEDS)
        and all(type(row) is ConfirmationSecondaryEvidenceCohortRow for row in cohort.rows),
        "secondary cohort must contain exactly 15 typed rows",
    )
    expected_keys = tuple(
        (method_id, seed) for method_id in CONFIRMATION_METHOD_IDS for seed in CONFIRMATION_SEEDS
    )
    observed_keys = tuple((row.method_id, row.seed) for row in cohort.rows)
    _require(observed_keys == expected_keys, "secondary cohort method/seed inventory differs")
    for row in cohort.rows:
        _validate_confirmation_secondary_row(row)
    _require(
        len({row.run_id for row in cohort.rows}) == len(cohort.rows),
        "secondary cohort reuses a run ID",
    )
    _require(
        len({row.evidence_sha256 for row in cohort.rows}) == len(cohort.rows),
        "secondary cohort reuses an evidence digest",
    )
    _require(
        len({row.trust_anchors.raw_evidence_sha256 for row in cohort.rows}) == len(cohort.rows),
        "secondary cohort reuses a raw-evidence digest",
    )
    common_fields = (
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
    first = cohort.rows[0].trust_anchors
    _require(
        all(
            getattr(row.trust_anchors, field) == getattr(first, field)
            for row in cohort.rows[1:]
            for field in common_fields
        ),
        "secondary cohort semantic identities differ across runs",
    )
    _sha256(cohort.cohort_sha256, label="secondary cohort digest")
    _require(
        cohort.cohort_sha256
        not in {row.evidence_sha256 for row in cohort.rows}
        | {row.trust_anchors.raw_evidence_sha256 for row in cohort.rows},
        "secondary cohort digest aliases a run digest",
    )
    _require(
        cohort.execution_authorized is False
        and cohort.scientific_claim_authorized is False
        and cohort.production_authorized is False,
        "secondary cohort cannot authorize execution, claims, or production",
    )
    _require(
        _confirmation_secondary_cohort_digest(cohort.rows) == cohort.cohort_sha256,
        "secondary cohort self-seal differs",
    )


def build_confirmation_secondary_evidence_cohort(
    protocol: EvolutionaryKLProtocol,
    rows: tuple[ConfirmationSecondaryEvidenceCohortRow, ...],
) -> ConfirmationSecondaryEvidenceCohort:
    """Build the exact non-authorizing 15-run inventory in canonical order."""

    _require(_protocol_is_supported(protocol), "protocol secondary evidence domain differs")
    _require(type(rows) is tuple, "secondary cohort rows must be a tuple")
    return ConfirmationSecondaryEvidenceCohort(
        rows=rows,
        cohort_sha256=_confirmation_secondary_cohort_digest(rows),
    )


def revalidate_confirmation_secondary_evidence_cohort(
    protocol: EvolutionaryKLProtocol,
    cohort: ConfirmationSecondaryEvidenceCohort,
    *,
    expected_cohort_sha256: str,
) -> ConfirmationSecondaryEvidenceCohort:
    """Require an out-of-band cohort digest and revalidate all 15 run records."""

    _require(_protocol_is_supported(protocol), "protocol secondary evidence domain differs")
    _require(type(cohort) is ConfirmationSecondaryEvidenceCohort, "secondary cohort type differs")
    expected = _sha256(expected_cohort_sha256, label="expected secondary cohort SHA-256")
    _validate_confirmation_secondary_cohort(cohort)
    _require(cohort.cohort_sha256 == expected, "secondary cohort digest differs")
    return cohort


def _legacy_secondary_metrics_from_cohort(
    protocol: EvolutionaryKLProtocol,
    cohort: ConfirmationSecondaryEvidenceCohort,
) -> dict[str, dict[int, PairedSecondaryGateMetrics]]:
    by_key = {(row.method_id, row.seed): row for row in cohort.rows}
    secondary: dict[str, dict[int, PairedSecondaryGateMetrics]] = {}
    for comparator_id in protocol.primary_comparator_ids:
        by_seed: dict[int, PairedSecondaryGateMetrics] = {}
        for seed in protocol.confirmation_seeds:
            full = by_key[(_FULL_METHOD_ID, seed)]
            comparator = by_key[(comparator_id, seed)]
            by_seed[seed] = _paired_secondary_gate_metrics(
                protocol,
                full_evidence=full.evidence,
                full_trust_anchors=full.trust_anchors,
                expected_full_evidence_sha256=full.evidence_sha256,
                comparator_evidence=comparator.evidence,
                comparator_trust_anchors=comparator.trust_anchors,
                expected_comparator_evidence_sha256=comparator.evidence_sha256,
            )
        secondary[comparator_id] = by_seed
    _validate_legacy_secondary_inventory(secondary)
    return secondary


def _validate_legacy_secondary_inventory(secondary: object) -> None:
    """Fail closed before identity-free arithmetic crosses into the pure gate."""

    _require(type(secondary) is dict, "downstream secondary gate inventory differs")
    assert isinstance(secondary, dict)
    _require(
        tuple(secondary) == PRIMARY_COMPARATOR_IDS,
        "downstream secondary comparator keys differ",
    )
    for comparator_id in PRIMARY_COMPARATOR_IDS:
        by_seed = secondary[comparator_id]
        _require(type(by_seed) is dict, "downstream secondary seed inventory differs")
        _require(
            tuple(by_seed) == CONFIRMATION_SEEDS,
            "downstream secondary seed keys differ",
        )
        _require(
            all(type(metrics) is PairedSecondaryGateMetrics for metrics in by_seed.values()),
            "downstream secondary metric type differs",
        )


def research_gate_decision_from_authenticated_secondary_cohort(
    protocol: EvolutionaryKLProtocol,
    *,
    screen_metrics: Mapping[str, Mapping[int, object]],
    confirmation_metrics: Mapping[str, Mapping[int, object]],
    secondary_cohort: ConfirmationSecondaryEvidenceCohort,
    expected_secondary_cohort_sha256: str,
    calibration: CalibrationGateMetrics,
    full_method_integrity_evidence: FullMethodIntegrityEvidenceReceipt | None,
    expected_integrity_trusted_receipt_sha256: str | None,
    independent_result_evidence: IndependentResultEvidenceReceipt | None,
    expected_independent_trusted_receipt_sha256: str | None,
) -> ResearchGateDecision:
    """Run pure gate math only after authenticating the exact 15-run cohort.

    There is intentionally no caller-supplied secondary count mapping in this
    API.  The expected cohort digest is a controller trust input; this wrapper
    does not replay raw bytes.  The returned v1 decision remains
    non-production-authorizing under the frozen blocked protocol.
    """

    cohort = revalidate_confirmation_secondary_evidence_cohort(
        protocol,
        secondary_cohort,
        expected_cohort_sha256=expected_secondary_cohort_sha256,
    )
    secondary = _legacy_secondary_metrics_from_cohort(protocol, cohort)
    _validate_legacy_secondary_inventory(secondary)
    return research_gate_decision(
        protocol,
        screen_metrics=screen_metrics,
        confirmation_metrics=confirmation_metrics,
        secondary=secondary,
        calibration=calibration,
        full_method_integrity_evidence=full_method_integrity_evidence,
        expected_integrity_trusted_receipt_sha256=expected_integrity_trusted_receipt_sha256,
        independent_result_evidence=independent_result_evidence,
        expected_independent_trusted_receipt_sha256=(expected_independent_trusted_receipt_sha256),
    )


def _bootstrap_seed_indices(
    *,
    bootstrap_seed: int,
    samples: int,
    seed_count: int,
) -> tuple[tuple[int, ...], ...]:
    _require(type(bootstrap_seed) is int and bootstrap_seed >= 0, "bootstrap seed differs")
    _require(type(samples) is int and samples > 0, "bootstrap sample count differs")
    _require(type(seed_count) is int and seed_count > 0, "bootstrap seed count differs")
    modulus = 1 << 256
    unbiased_limit = modulus - modulus % seed_count
    seed_bytes = bootstrap_seed.to_bytes(16, "big", signed=False)
    result: list[tuple[int, ...]] = []
    for replicate in range(samples):
        row: list[int] = []
        for draw in range(seed_count):
            nonce = 0
            while True:
                value = int.from_bytes(
                    hashlib.sha256(
                        _BOOTSTRAP_INDEX_HASH_DOMAIN
                        + seed_bytes
                        + replicate.to_bytes(8, "big")
                        + draw.to_bytes(2, "big")
                        + nonce.to_bytes(2, "big")
                    ).digest(),
                    "big",
                )
                if value < unbiased_limit:
                    row.append(value % seed_count)
                    break
                nonce += 1
                _require(nonce <= 65_535, "bootstrap rejection sampler exhausted")
        result.append(tuple(row))
    return tuple(result)


def _bootstrap_index_sha256(
    indices: tuple[tuple[int, ...], ...],
    *,
    bootstrap_seed: int,
    seed_count: int,
) -> str:
    flat = bytes(index for row in indices for index in row)
    return sha256_bytes(
        _BOOTSTRAP_INDEX_HASH_DOMAIN
        + bootstrap_seed.to_bytes(16, "big", signed=False)
        + len(indices).to_bytes(8, "big")
        + seed_count.to_bytes(2, "big")
        + flat
    )


def _linear_type7_quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    _require(bool(sorted_values), "bootstrap replicate vector is empty")
    _require(0.0 <= probability <= 1.0, "bootstrap probability differs")
    h = (len(sorted_values) - 1) * probability
    lower = math.floor(h)
    upper = math.ceil(h)
    fraction = h - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _secondary_metric_triplet(evidence: SecondaryYieldEvidence) -> tuple[float, float, float]:
    _validate_evidence(evidence)
    return (
        evidence.valid_unique_reference_safe_yield,
        evidence.hill2_effective_identity70_clusters,
        evidence.largest_identity70_cluster_share,
    )


def _bootstrap_summary(
    metric_id: str,
    effects: tuple[float, ...],
    indices: tuple[tuple[int, ...], ...],
    quantiles: tuple[float, float],
) -> SecondaryPairedBootstrapSummary:
    replicates = tuple(
        math.fsum(effects[index] for index in index_row) / len(index_row) for index_row in indices
    )
    ordered = tuple(sorted(replicates))
    return SecondaryPairedBootstrapSummary(
        metric_id=metric_id,
        seed_effects=effects,
        arithmetic_mean_effect=math.fsum(effects) / len(effects),
        percentile_interval=(
            _linear_type7_quantile(ordered, quantiles[0]),
            _linear_type7_quantile(ordered, quantiles[1]),
        ),
    )


def _secondary_uncertainty_unsigned_document(
    *,
    cohort_sha256: str,
    shared_bootstrap_index_sha256: str,
    comparators: tuple[SecondaryComparatorPairedUncertainty, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": SECONDARY_UNCERTAINTY_ARTIFACT,
        "status": "descriptive_three_method_confirmation_secondary_only",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "cohort_sha256": cohort_sha256,
        "observed_method_ids": list(CONFIRMATION_METHOD_IDS),
        "seed_ids": list(CONFIRMATION_SEEDS),
        "bootstrap_unit": (
            "seed_block_resample_with_same_seed_index_shared_across_all_compared_methods"
        ),
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_908,
        "bootstrap_statistic": (
            "arithmetic_mean_of_five_paired_full_minus_comparator_seed_effects"
        ),
        "shared_bootstrap_index_sha256": shared_bootstrap_index_sha256,
        "interval_type": "percentile_two_sided",
        "interval_quantiles_hex": [(0.025).hex(), (0.975).hex()],
        "quantile_convention": (
            "sorted_B_replicates_linear_type7_h_equals_B_minus_1_times_p_"
            "interpolate_between_floor_and_ceil"
        ),
        "comparators": [
            {
                "comparator_id": comparator.comparator_id,
                "summaries": [
                    {
                        "metric_id": summary.metric_id,
                        "seed_effects_hex": [value.hex() for value in summary.seed_effects],
                        "arithmetic_mean_effect_hex": summary.arithmetic_mean_effect.hex(),
                        "percentile_interval_hex": [
                            value.hex() for value in summary.percentile_interval
                        ],
                    }
                    for summary in comparator.summaries
                ],
            }
            for comparator in comparators
        ],
        "execution_authorized": False,
        "scientific_claim_authorized": False,
        "production_authorized": False,
    }


def _confirmation_secondary_uncertainty_document(
    uncertainty: ConfirmationSecondaryPairedUncertainty,
    *,
    include_digest: bool,
) -> dict[str, object]:
    document = _secondary_uncertainty_unsigned_document(
        cohort_sha256=uncertainty.cohort_sha256,
        shared_bootstrap_index_sha256=uncertainty.shared_bootstrap_index_sha256,
        comparators=uncertainty.comparators,
    )
    if include_digest:
        document["uncertainty_sha256"] = uncertainty.uncertainty_sha256
    return document


def _validate_confirmation_secondary_uncertainty(
    uncertainty: ConfirmationSecondaryPairedUncertainty,
) -> None:
    _require(
        type(uncertainty) is ConfirmationSecondaryPairedUncertainty,
        "secondary uncertainty type differs",
    )
    _require(
        uncertainty.protocol_sha256 == FROZEN_PROTOCOL_SHA256,
        "secondary uncertainty protocol differs",
    )
    _sha256(uncertainty.cohort_sha256, label="secondary uncertainty cohort digest")
    _sha256(
        uncertainty.shared_bootstrap_index_sha256,
        label="secondary uncertainty bootstrap-index digest",
    )
    _sha256(uncertainty.uncertainty_sha256, label="secondary uncertainty self-seal")
    _require(
        uncertainty.observed_method_ids == CONFIRMATION_METHOD_IDS,
        "secondary uncertainty observed-method cohort differs",
    )
    _require(uncertainty.seed_ids == CONFIRMATION_SEEDS, "secondary uncertainty seeds differ")
    _require(
        uncertainty.bootstrap_unit
        == "seed_block_resample_with_same_seed_index_shared_across_all_compared_methods"
        and uncertainty.bootstrap_samples == 10_000
        and uncertainty.bootstrap_seed == 20_260_908
        and uncertainty.bootstrap_statistic
        == "arithmetic_mean_of_five_paired_full_minus_comparator_seed_effects",
        "secondary uncertainty bootstrap contract differs",
    )
    _require(
        uncertainty.interval_type == "percentile_two_sided"
        and uncertainty.interval_quantiles == (0.025, 0.975)
        and uncertainty.quantile_convention
        == (
            "sorted_B_replicates_linear_type7_h_equals_B_minus_1_times_p_"
            "interpolate_between_floor_and_ceil"
        ),
        "secondary uncertainty interval contract differs",
    )
    _require(
        type(uncertainty.comparators) is tuple
        and all(
            type(item) is SecondaryComparatorPairedUncertainty for item in uncertainty.comparators
        )
        and tuple(item.comparator_id for item in uncertainty.comparators) == PRIMARY_COMPARATOR_IDS,
        "secondary uncertainty comparator inventory differs",
    )
    for comparator in uncertainty.comparators:
        comparator.__post_init__()
        for summary in comparator.summaries:
            summary.__post_init__()
    _require(
        uncertainty.execution_authorized is False
        and uncertainty.scientific_claim_authorized is False
        and uncertainty.production_authorized is False,
        "secondary uncertainty cannot authorize execution, claims, or production",
    )
    expected = sha256_bytes(
        _SECONDARY_UNCERTAINTY_HASH_DOMAIN
        + canonical_json_bytes(
            _confirmation_secondary_uncertainty_document(uncertainty, include_digest=False)
        )
    )
    _require(expected == uncertainty.uncertainty_sha256, "secondary uncertainty self-seal differs")


def confirmation_secondary_paired_uncertainty(
    protocol: EvolutionaryKLProtocol,
    cohort: ConfirmationSecondaryEvidenceCohort,
    *,
    expected_cohort_sha256: str,
) -> ConfirmationSecondaryPairedUncertainty:
    """Describe paired secondary effects with one shared seed-block bootstrap.

    This reports only the three methods and five seeds present in the exact
    confirmation cohort.  It is descriptive, does not resample queries, and
    grants no execution, scientific-claim, registry, or production authority.
    """

    validated = revalidate_confirmation_secondary_evidence_cohort(
        protocol,
        cohort,
        expected_cohort_sha256=expected_cohort_sha256,
    )
    indices = _bootstrap_seed_indices(
        bootstrap_seed=protocol.paired_bootstrap_seed,
        samples=protocol.paired_bootstrap_samples,
        seed_count=len(protocol.confirmation_seeds),
    )
    index_sha256 = _bootstrap_index_sha256(
        indices,
        bootstrap_seed=protocol.paired_bootstrap_seed,
        seed_count=len(protocol.confirmation_seeds),
    )
    by_key = {(row.method_id, row.seed): row for row in validated.rows}
    comparator_results: list[SecondaryComparatorPairedUncertainty] = []
    quantiles = protocol.statistical_reporting.paired_bootstrap_interval_quantiles
    _require(len(quantiles) == 2, "secondary uncertainty quantile count differs")
    exact_quantiles = (quantiles[0], quantiles[1])
    for comparator_id in protocol.primary_comparator_ids:
        effects_by_metric: list[list[float]] = [[], [], []]
        for seed in protocol.confirmation_seeds:
            full_values = _secondary_metric_triplet(by_key[(_FULL_METHOD_ID, seed)].evidence)
            comparator_values = _secondary_metric_triplet(by_key[(comparator_id, seed)].evidence)
            for index, (full_value, comparator_value) in enumerate(
                zip(full_values, comparator_values, strict=True)
            ):
                effects_by_metric[index].append(full_value - comparator_value)
        summaries = tuple(
            _bootstrap_summary(metric_id, tuple(effects_by_metric[index]), indices, exact_quantiles)
            for index, metric_id in enumerate(_SECONDARY_METRIC_IDS)
        )
        comparator_results.append(
            SecondaryComparatorPairedUncertainty(
                comparator_id=comparator_id,
                summaries=summaries,
            )
        )
    comparators = tuple(comparator_results)
    unsigned = _secondary_uncertainty_unsigned_document(
        cohort_sha256=validated.cohort_sha256,
        shared_bootstrap_index_sha256=index_sha256,
        comparators=comparators,
    )
    uncertainty_sha256 = sha256_bytes(
        _SECONDARY_UNCERTAINTY_HASH_DOMAIN + canonical_json_bytes(unsigned)
    )
    return ConfirmationSecondaryPairedUncertainty(
        protocol_sha256=FROZEN_PROTOCOL_SHA256,
        cohort_sha256=validated.cohort_sha256,
        observed_method_ids=protocol.confirmation_method_ids,
        seed_ids=protocol.confirmation_seeds,
        bootstrap_unit=protocol.de_novo_paired_bootstrap_unit,
        bootstrap_samples=protocol.paired_bootstrap_samples,
        bootstrap_seed=protocol.paired_bootstrap_seed,
        bootstrap_statistic=protocol.statistical_reporting.paired_bootstrap_statistic,
        shared_bootstrap_index_sha256=index_sha256,
        interval_type=protocol.statistical_reporting.paired_bootstrap_interval_type,
        interval_quantiles=exact_quantiles,
        quantile_convention=(protocol.statistical_reporting.paired_bootstrap_quantile_convention),
        comparators=comparators,
        uncertainty_sha256=uncertainty_sha256,
    )


__all__ = [
    "HOMOLOGY_CONTRACT_ARTIFACT",
    "ORACLE_CONTRACT_ARTIFACT",
    "RAW_HEADER_ARTIFACT",
    "RAW_QUERY_ARTIFACT",
    "REFERENCE_CONTRACT_ARTIFACT",
    "SECONDARY_COHORT_ARTIFACT",
    "SECONDARY_COHORT_ROW_ARTIFACT",
    "SECONDARY_EVIDENCE_ARTIFACT",
    "SECONDARY_UNCERTAINTY_ARTIFACT",
    "SEQUENCE_SET_ROW_ARTIFACT",
    "SUPPORT_CONTRACT_ARTIFACT",
    "TRUTH_CONTRACT_ARTIFACT",
    "ConfirmationSecondaryEvidenceCohort",
    "ConfirmationSecondaryEvidenceCohortRow",
    "ConfirmationSecondaryPairedUncertainty",
    "SecondaryComparatorPairedUncertainty",
    "SecondaryEvidenceError",
    "SecondaryEvidenceTrustAnchors",
    "SecondaryPairedBootstrapSummary",
    "SecondaryYieldEvidence",
    "build_confirmation_secondary_evidence_cohort",
    "compute_authenticated_secondary_yield",
    "confirmation_secondary_cohort_row",
    "confirmation_secondary_paired_uncertainty",
    "query_identity_inventory_sha256",
    "research_gate_decision_from_authenticated_secondary_cohort",
    "revalidate_confirmation_secondary_evidence_cohort",
    "revalidate_secondary_yield_evidence",
]

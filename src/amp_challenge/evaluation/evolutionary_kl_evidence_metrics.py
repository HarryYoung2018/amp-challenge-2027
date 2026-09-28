"""Authenticated primary-metric reduction for evolutionary/KL research runs.

This module accepts only a pathless :class:`~.sequential_v2_seals.PhaseSeal`
whose exact seal and independent authority predecessor are supplied by the
caller.  It recomputes the submitted-query inventory and feasible two-objective
hypervolume from raw oracle rows under a caller-pinned canonical constraint
contract. Producer-supplied metric scalars are never an input, and a retained
constraint-pass flag must equal the contract-derived result.

The immutable v1 protocol is still blocked and this reducer does not authorize
execution, scientific claims, or production use.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import pairwise

from amp_challenge.evaluation.evolutionary_kl_protocol import (
    FROZEN_PROTOCOL_SHA256,
    EvolutionaryKLProtocol,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
    verify_phase_capability,
)

QUERY_PHASE_ARTIFACT = "evolutionary_kl_authenticated_oracle_query_cohort_v1"
QUERY_ROW_ARTIFACT = "evolutionary_kl_authenticated_oracle_query_row_v1"
CONSTRAINT_CONTRACT_ARTIFACT = "evolutionary_kl_constraint_semantics_v1"
QUERY_PAYLOAD_PATH = "oracle-queries.jsonl"
TRUSTED_RECEIPT_PREDECESSOR = "authority/trusted-receipt"

_FROZEN_CALL_CHECKPOINTS = tuple(range(64, 513, 16))
_FROZEN_HYPERVOLUME_AUC_DENOMINATOR_CALLS = 448
_FROZEN_QUERY_IDENTITY_FIELDS = (
    "canonical_sequence_id",
    "oracle_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
    "replicate_id",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONSTRAINT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_QUERY_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "run_id",
        "row_count",
        "unique_query_identity_count",
        "query_identity_fields",
        "query_identity_inventory_sha256",
        "constraint_ids",
        "constraint_contract_sha256",
        "trusted_receipt_sha256",
    }
)
_QUERY_ROW_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "run_id",
        "charged_call_position",
        "query_identity",
        "response_status",
        "atomic_response_complete",
        "objectives",
        "objective_censored",
        "constraints",
        "canonical_support_eligible",
        "exact_training_overlap",
        "successor_homology_eligible",
    }
)
_CONSTRAINT_KEYS = frozenset({"value", "passed", "censored"})
_CONSTRAINT_CONTRACT_KEYS = frozenset(
    {"schema_version", "artifact", "status", "protocol_sha256", "constraints"}
)
_CONSTRAINT_RULE_KEYS = frozenset({"constraint_id", "operator", "threshold"})
_OPERATORS = frozenset({"lt", "le", "gt", "ge"})
_RESPONSE_STATUSES = frozenset({"complete", "failed", "missing", "censored", "partial", "timeout"})


class EvidenceMetricError(ValueError):
    """Raised when authenticated evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class PrimaryHypervolumeEvidence:
    """Deterministic evidence derived from one authenticated query stream."""

    run_id: str
    phase_seal_sha256: str
    phase_receipt_sha256: str
    trusted_receipt_sha256: str
    constraint_contract_sha256: str
    query_identity_inventory_sha256: str
    charged_submitted_identity_count: int
    feasible_identity_count: int
    sealed_checkpoints: tuple[tuple[int, float], ...]
    carried_checkpoints: tuple[tuple[int, float], ...]
    normalized_hypervolume_auc: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        _validate_evidence_instance(self)

    def document_bytes(self) -> bytes:
        """Return the canonical, hash-authenticated evidence document."""

        _validate_evidence_instance(self)
        document = _evidence_document(self, include_digest=True)
        return canonical_json_bytes(document)


@dataclass(frozen=True, slots=True)
class _QueryRow:
    position: int
    identity: tuple[tuple[str, str], ...]
    objectives: tuple[float, float] | None
    feasible: bool


@dataclass(frozen=True, slots=True)
class _ConstraintRule:
    constraint_id: str
    operator: str
    threshold: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceMetricError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} invalid")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    _require(type(payload) is bytes, f"{label} must be bytes")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceMetricError(f"{label} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise EvidenceMetricError(f"{label} contains invalid constant {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceMetricError(f"{label} is not strict UTF-8 JSON") from error
    _require(type(parsed) is dict, f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, ValueError) as error:
        raise EvidenceMetricError(f"{label} is not finite canonical JSON") from error
    _require(canonical == payload, f"{label} is not canonical JSON")
    return parsed


def _strict_jsonl(payload: bytes) -> tuple[dict[str, object], ...]:
    _require(type(payload) is bytes and payload, "oracle query payload must be nonempty bytes")
    _require(payload.endswith(b"\n"), "oracle query payload must end in LF")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(payload.splitlines(keepends=True), start=1):
        _require(line not in {b"", b"\n"}, "oracle query payload contains a blank row")
        rows.append(_strict_json_object(line, label=f"oracle query row {index}"))
    return tuple(rows)


def _exact_object(
    value: object,
    expected_keys: frozenset[str] | set[str],
    *,
    label: str,
) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(
        all(type(key) is str for key in value) and set(value) == set(expected_keys),
        f"{label} keys differ",
    )
    return value


def _finite_number(value: object, *, label: str) -> float:
    _require(type(value) in {int, float}, f"{label} must be a non-Boolean real number")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise EvidenceMetricError(f"{label} must be finite") from error
    _require(math.isfinite(parsed), f"{label} must be finite")
    return parsed


def _optional_finite_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label=label)


def _parse_constraint_contract(
    payload: bytes,
    *,
    expected_sha256: str,
) -> tuple[_ConstraintRule, ...]:
    expected = _sha256(expected_sha256, label="expected constraint contract SHA-256")
    _require(type(payload) is bytes, "constraint contract must be canonical bytes")
    _require(sha256_bytes(payload) == expected, "constraint contract digest differs")
    document = _exact_object(
        _strict_json_object(payload, label="constraint contract"),
        _CONSTRAINT_CONTRACT_KEYS,
        label="constraint contract",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "constraint contract schema differs",
    )
    _require(
        document["artifact"] == CONSTRAINT_CONTRACT_ARTIFACT and type(document["artifact"]) is str,
        "constraint contract artifact differs",
    )
    _require(
        document["status"] == "accepted_content_pinned" and type(document["status"]) is str,
        "constraint contract is not accepted",
    )
    _require(
        document["protocol_sha256"] == FROZEN_PROTOCOL_SHA256,
        "constraint contract protocol differs",
    )
    raw_rules = document["constraints"]
    _require(
        type(raw_rules) is list and bool(raw_rules),
        "constraint rules must be a nonempty array",
    )
    rules: list[_ConstraintRule] = []
    for index, raw_rule in enumerate(raw_rules):
        rule = _exact_object(raw_rule, _CONSTRAINT_RULE_KEYS, label=f"constraint rule {index}")
        constraint_id = rule["constraint_id"]
        operator = rule["operator"]
        _require(
            type(constraint_id) is str and _CONSTRAINT_ID_RE.fullmatch(constraint_id) is not None,
            f"constraint rule {index} ID invalid",
        )
        _require(
            type(operator) is str and operator in _OPERATORS,
            f"constraint rule {index} operator invalid",
        )
        rules.append(
            _ConstraintRule(
                constraint_id=constraint_id,
                operator=operator,
                threshold=_finite_number(
                    rule["threshold"], label=f"constraint rule {index} threshold"
                ),
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
    raise AssertionError("validated constraint operator became unreachable")


def _parse_identity(
    value: object,
    *,
    identity_fields: tuple[str, ...],
    expected_oracle_contract_sha256: str,
    row_number: int,
) -> tuple[tuple[str, str], ...]:
    identity = _exact_object(value, set(identity_fields), label=f"row {row_number} identity")
    parsed: list[tuple[str, str]] = []
    for field in identity_fields:
        item = identity[field]
        _require(
            type(item) is str and bool(item),
            f"row {row_number} identity field {field} must be nonempty text",
        )
        if field.endswith("sha256") or field == "canonical_sequence_id":
            _sha256(item, label=f"row {row_number} identity field {field}")
        if field == "oracle_contract_sha256":
            _require(
                item == expected_oracle_contract_sha256,
                f"row {row_number} oracle contract digest differs",
            )
        parsed.append((field, item))
    return tuple(parsed)


def _parse_query_row(
    raw: dict[str, object],
    *,
    row_number: int,
    expected_run_id: str,
    identity_fields: tuple[str, ...],
    objective_ids: tuple[str, str],
    objective_bounds: tuple[float, float],
    constraint_rules: tuple[_ConstraintRule, ...],
    expected_constraint_contract_sha256: str,
) -> _QueryRow:
    row = _exact_object(raw, _QUERY_ROW_KEYS, label=f"oracle query row {row_number}")
    _require(
        type(row["schema_version"]) is int and row["schema_version"] == 1,
        f"row {row_number} schema version differs",
    )
    _require(
        type(row["artifact"]) is str and row["artifact"] == QUERY_ROW_ARTIFACT,
        f"row {row_number} artifact differs",
    )
    _require(
        type(row["run_id"]) is str and row["run_id"] == expected_run_id,
        f"row {row_number} belongs to a different run",
    )
    position = row["charged_call_position"]
    _require(
        type(position) is int and position >= 1,
        f"row {row_number} charged position must be a positive integer",
    )
    assert isinstance(position, int)
    identity = _parse_identity(
        row["query_identity"],
        identity_fields=identity_fields,
        expected_oracle_contract_sha256=expected_constraint_contract_sha256,
        row_number=row_number,
    )
    status = row["response_status"]
    _require(
        type(status) is str and status in _RESPONSE_STATUSES,
        f"row {row_number} response status differs",
    )
    atomic = row["atomic_response_complete"]
    _require(type(atomic) is bool, f"row {row_number} atomic-response flag must be Boolean")

    objectives = _exact_object(
        row["objectives"], set(objective_ids), label=f"row {row_number} objectives"
    )
    censoring = _exact_object(
        row["objective_censored"],
        set(objective_ids),
        label=f"row {row_number} objective censoring",
    )
    parsed_objectives: list[float | None] = []
    for objective in objective_ids:
        value = _optional_finite_number(
            objectives[objective], label=f"row {row_number} objective {objective}"
        )
        if value is not None:
            _require(
                objective_bounds[0] <= value <= objective_bounds[1],
                f"row {row_number} objective {objective} is outside frozen bounds",
            )
        parsed_objectives.append(value)
        _require(
            type(censoring[objective]) is bool,
            f"row {row_number} objective censoring must be Boolean",
        )

    constraint_ids = tuple(rule.constraint_id for rule in constraint_rules)
    constraints = _exact_object(
        row["constraints"], set(constraint_ids), label=f"row {row_number} constraints"
    )
    constraints_complete = True
    constraints_pass = True
    constraints_uncensored = True
    for rule in constraint_rules:
        constraint_id = rule.constraint_id
        item = _exact_object(
            constraints[constraint_id],
            _CONSTRAINT_KEYS,
            label=f"row {row_number} constraint {constraint_id}",
        )
        value = _optional_finite_number(
            item["value"], label=f"row {row_number} constraint {constraint_id} value"
        )
        passed = item["passed"]
        censored = item["censored"]
        _require(
            passed is None or type(passed) is bool,
            f"row {row_number} constraint pass flag must be Boolean or null",
        )
        _require(
            type(censored) is bool,
            f"row {row_number} constraint censoring must be Boolean",
        )
        if value is None or censored is True:
            _require(
                passed is None,
                f"row {row_number} constraint pass flag must be null without exact raw truth",
            )
            semantic_pass: bool | None = None
        else:
            semantic_pass = _rule_passes(rule, value)
            _require(
                type(passed) is bool and passed is semantic_pass,
                f"row {row_number} constraint pass flag differs from pinned semantics",
            )
        constraints_complete &= value is not None and type(passed) is bool
        constraints_pass &= semantic_pass is True
        constraints_uncensored &= censored is False

    support_eligible = row["canonical_support_eligible"]
    exact_overlap = row["exact_training_overlap"]
    homology_eligible = row["successor_homology_eligible"]
    _require(type(support_eligible) is bool, f"row {row_number} support flag must be Boolean")
    _require(type(exact_overlap) is bool, f"row {row_number} overlap flag must be Boolean")
    _require(type(homology_eligible) is bool, f"row {row_number} homology flag must be Boolean")

    objectives_complete = all(value is not None for value in parsed_objectives)
    objectives_uncensored = all(censoring[objective] is False for objective in objective_ids)
    response_complete = status == "complete" and atomic is True
    if status == "complete":
        _require(
            atomic is True
            and objectives_complete
            and objectives_uncensored
            and constraints_complete
            and constraints_uncensored,
            f"row {row_number} claims a non-atomic or incomplete complete response",
        )
    else:
        _require(
            atomic is False,
            f"row {row_number} non-complete response cannot claim atomic completeness",
        )
    feasible = (
        response_complete
        and objectives_complete
        and objectives_uncensored
        and constraints_complete
        and constraints_uncensored
        and constraints_pass
        and support_eligible is True
        and exact_overlap is False
        and homology_eligible is True
    )
    objective_pair: tuple[float, float] | None = None
    if objectives_complete:
        left, right = parsed_objectives
        assert left is not None and right is not None
        objective_pair = (left, right)
    return _QueryRow(
        position=position,
        identity=identity,
        objectives=objective_pair,
        feasible=feasible,
    )


def _identity_document(identity: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(identity)


def query_identity_inventory_sha256(
    identities: tuple[tuple[tuple[str, str], ...], ...],
) -> str:
    """Return the canonical digest of a sorted, duplicate-free identity inventory."""

    _require(type(identities) is tuple and bool(identities), "identity inventory must be nonempty")
    encoded: list[tuple[bytes, dict[str, str]]] = []
    for identity in identities:
        _require(type(identity) is tuple and bool(identity), "query identity must be a tuple")
        document = _identity_document(identity)
        _require(len(document) == len(identity), "query identity fields are duplicated")
        encoded.append((canonical_json_bytes(document), document))
    encoded.sort(key=lambda item: item[0])
    _require(
        all(left[0] != right[0] for left, right in pairwise(encoded)),
        "query identity inventory contains duplicates",
    )
    return sha256_bytes(canonical_jsonl_bytes(document for _payload, document in encoded))


def _hypervolume_2d(
    points: tuple[tuple[float, float], ...],
    *,
    reference: tuple[float, float],
) -> float:
    terms: list[float] = []
    best_second = reference[1]
    for first, second in sorted(set(points), reverse=True):
        if second > best_second:
            terms.append((first - reference[0]) * (second - best_second))
            best_second = second
    value = math.fsum(terms)
    _require(math.isfinite(value) and 0.0 <= value <= 1.0, "computed hypervolume is invalid")
    return value


def _unsigned_evidence_document(
    *,
    run_id: str,
    phase_seal_sha256: str,
    phase_receipt_sha256: str,
    trusted_receipt_sha256: str,
    constraint_contract_sha256: str,
    query_identity_inventory_sha256_value: str,
    charged_submitted_identity_count: int,
    feasible_identity_count: int,
    sealed_checkpoints: tuple[tuple[int, float], ...],
    carried_checkpoints: tuple[tuple[int, float], ...],
    normalized_hypervolume_auc: float,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": "evolutionary_kl_primary_hypervolume_evidence_v1",
        "status": "derived_from_authenticated_sealed_raw_oracle_queries",
        "run_id": run_id,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "phase_seal_sha256": phase_seal_sha256,
        "phase_receipt_sha256": phase_receipt_sha256,
        "trusted_receipt_sha256": trusted_receipt_sha256,
        "constraint_contract_sha256": constraint_contract_sha256,
        "query_identity_inventory_sha256": query_identity_inventory_sha256_value,
        "charged_submitted_identity_count": charged_submitted_identity_count,
        "feasible_identity_count": feasible_identity_count,
        "sealed_checkpoints": [
            {"charged_call_position": checkpoint, "hypervolume_hex": value.hex()}
            for checkpoint, value in sealed_checkpoints
        ],
        "carried_checkpoints": [
            {"charged_call_position": checkpoint, "hypervolume_hex": value.hex()}
            for checkpoint, value in carried_checkpoints
        ],
        "normalized_hypervolume_auc_hex": normalized_hypervolume_auc.hex(),
    }
    return document


def _evidence_document(
    evidence: PrimaryHypervolumeEvidence,
    *,
    include_digest: bool,
) -> dict[str, object]:
    document = _unsigned_evidence_document(
        run_id=evidence.run_id,
        phase_seal_sha256=evidence.phase_seal_sha256,
        phase_receipt_sha256=evidence.phase_receipt_sha256,
        trusted_receipt_sha256=evidence.trusted_receipt_sha256,
        constraint_contract_sha256=evidence.constraint_contract_sha256,
        query_identity_inventory_sha256_value=evidence.query_identity_inventory_sha256,
        charged_submitted_identity_count=evidence.charged_submitted_identity_count,
        feasible_identity_count=evidence.feasible_identity_count,
        sealed_checkpoints=evidence.sealed_checkpoints,
        carried_checkpoints=evidence.carried_checkpoints,
        normalized_hypervolume_auc=evidence.normalized_hypervolume_auc,
    )
    if include_digest:
        document["evidence_sha256"] = evidence.evidence_sha256
    return document


def _validate_evidence_instance(evidence: PrimaryHypervolumeEvidence) -> None:
    _require(
        type(evidence.run_id) is str and _RUN_ID_RE.fullmatch(evidence.run_id) is not None,
        "evidence run ID is invalid",
    )
    for label, value in (
        ("evidence phase seal", evidence.phase_seal_sha256),
        ("evidence phase receipt", evidence.phase_receipt_sha256),
        ("evidence trusted receipt", evidence.trusted_receipt_sha256),
        ("evidence constraint contract", evidence.constraint_contract_sha256),
        ("evidence query inventory", evidence.query_identity_inventory_sha256),
        ("evidence self-hash", evidence.evidence_sha256),
    ):
        _sha256(value, label=label)
    _require(
        evidence.phase_seal_sha256 != evidence.trusted_receipt_sha256
        and evidence.phase_receipt_sha256 != evidence.trusted_receipt_sha256,
        "evidence producer phase cannot stand in for independent authority",
    )
    _require(
        type(evidence.charged_submitted_identity_count) is int
        and type(evidence.feasible_identity_count) is int
        and evidence.charged_submitted_identity_count >= _FROZEN_CALL_CHECKPOINTS[0]
        and evidence.charged_submitted_identity_count <= _FROZEN_CALL_CHECKPOINTS[-1]
        and 0 <= evidence.feasible_identity_count <= evidence.charged_submitted_identity_count,
        "evidence identity counts are invalid",
    )
    for label, checkpoints in (
        ("sealed", evidence.sealed_checkpoints),
        ("carried", evidence.carried_checkpoints),
    ):
        _require(
            type(checkpoints) is tuple and bool(checkpoints),
            f"evidence {label} checkpoints invalid",
        )
        previous_position = 0
        previous_value = 0.0
        for checkpoint in checkpoints:
            _require(
                type(checkpoint) is tuple
                and len(checkpoint) == 2
                and type(checkpoint[0]) is int
                and type(checkpoint[1]) is float
                and checkpoint[0] > previous_position
                and math.isfinite(checkpoint[1])
                and 0.0 <= checkpoint[1] <= 1.0
                and checkpoint[1] >= previous_value,
                f"evidence {label} checkpoint is invalid",
            )
            previous_position, previous_value = checkpoint
    expected_sealed_positions = tuple(
        checkpoint
        for checkpoint in _FROZEN_CALL_CHECKPOINTS
        if checkpoint <= evidence.charged_submitted_identity_count
    )
    _require(
        tuple(position for position, _value in evidence.sealed_checkpoints)
        == expected_sealed_positions,
        "evidence sealed checkpoint census differs",
    )
    _require(
        tuple(position for position, _value in evidence.carried_checkpoints)
        == _FROZEN_CALL_CHECKPOINTS
        and evidence.carried_checkpoints[: len(evidence.sealed_checkpoints)]
        == evidence.sealed_checkpoints
        and all(
            value == evidence.sealed_checkpoints[-1][1]
            for _position, value in evidence.carried_checkpoints[len(evidence.sealed_checkpoints) :]
        ),
        "evidence carried checkpoint reduction differs",
    )
    expected_auc = (
        math.fsum(
            (right_position - left_position) * (left_value + right_value) / 2.0
            for (left_position, left_value), (right_position, right_value) in pairwise(
                evidence.carried_checkpoints
            )
        )
        / _FROZEN_HYPERVOLUME_AUC_DENOMINATOR_CALLS
    )
    _require(
        type(evidence.normalized_hypervolume_auc) is float
        and math.isfinite(evidence.normalized_hypervolume_auc)
        and 0.0 <= evidence.normalized_hypervolume_auc <= 1.0,
        "evidence normalized AUC is invalid",
    )
    _require(
        evidence.normalized_hypervolume_auc == expected_auc,
        "evidence normalized AUC reduction differs",
    )
    observed = sha256_bytes(
        canonical_json_bytes(_evidence_document(evidence, include_digest=False))
    )
    _require(observed == evidence.evidence_sha256, "evidence self-hash differs")


def compute_authenticated_primary_hypervolume(
    protocol: EvolutionaryKLProtocol,
    query_capability: PhaseSeal,
    *,
    expected_phase_seal_sha256: str,
    expected_trusted_receipt_sha256: str,
    expected_run_id: str,
    expected_query_identity_inventory_sha256: str,
    constraint_contract_bytes: bytes,
    expected_constraint_contract_sha256: str,
) -> PrimaryHypervolumeEvidence:
    """Recompute feasible 2-D HV checkpoints and normalized AUC from raw rows.

    The phase seal, trusted receipt, query inventory, and constraint-contract
    digest are caller-supplied authorities. The contract content determines
    feasibility from raw constraint values; a producer pass flag is retained
    only as a redundant value that must agree exactly.
    """

    _require(type(protocol) is EvolutionaryKLProtocol, "protocol type differs")
    phase_seal = _sha256(expected_phase_seal_sha256, label="expected phase seal SHA-256")
    trusted_receipt = _sha256(
        expected_trusted_receipt_sha256,
        label="expected trusted receipt SHA-256",
    )
    inventory_expected = _sha256(
        expected_query_identity_inventory_sha256,
        label="expected query identity inventory SHA-256",
    )
    constraint_rules = _parse_constraint_contract(
        constraint_contract_bytes,
        expected_sha256=expected_constraint_contract_sha256,
    )
    constraint_contract_sha256 = sha256_bytes(constraint_contract_bytes)
    _require(phase_seal != trusted_receipt, "producer phase seal cannot self-authorize")
    _require(
        type(expected_run_id) is str and _RUN_ID_RE.fullmatch(expected_run_id) is not None,
        "expected run ID is invalid",
    )
    _require(
        protocol.primary_metric
        == "normalized_feasible_two_objective_hypervolume_auc_by_submitted_query_identity"
        and len(protocol.primary_objectives) == 2
        and protocol.objective_bounds == (0.0, 1.0)
        and protocol.hypervolume_reference == (0.0, 0.0)
        and protocol.oracle_query_contract.query_identity_fields == _FROZEN_QUERY_IDENTITY_FIELDS
        and protocol.call_checkpoints == _FROZEN_CALL_CHECKPOINTS
        and protocol.hypervolume_auc_denominator_calls == _FROZEN_HYPERVOLUME_AUC_DENOMINATOR_CALLS,
        "protocol primary-metric domain differs from the supported frozen contract",
    )

    try:
        verified = verify_phase_capability(
            query_capability,
            expected_artifact=QUERY_PHASE_ARTIFACT,
            expected_payload_paths=(QUERY_PAYLOAD_PATH,),
            expected_predecessor_seals={TRUSTED_RECEIPT_PREDECESSOR: trusted_receipt},
            expected_seal_sha256=phase_seal,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise EvidenceMetricError("oracle query phase authentication failed") from error
    _require(
        trusted_receipt not in {verified.seal_sha256, verified.receipt_sha256},
        "producer phase or receipt cannot stand in for independent authority",
    )

    metadata = _strict_json_object(verified.metadata_json, label="oracle query phase metadata")
    _exact_object(metadata, _QUERY_METADATA_KEYS, label="oracle query phase metadata")
    identity_fields = protocol.oracle_query_contract.query_identity_fields
    _require(
        metadata["schema_version"] == 1 and type(metadata["schema_version"]) is int,
        "oracle query metadata schema version differs",
    )
    _require(
        metadata["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
        and type(metadata["protocol_sha256"]) is str,
        "oracle query metadata protocol differs",
    )
    _require(
        metadata["run_id"] == expected_run_id and type(metadata["run_id"]) is str,
        "oracle query metadata run differs",
    )
    _require(
        metadata["query_identity_fields"] == list(identity_fields),
        "oracle query metadata identity fields differ",
    )
    _require(
        metadata["constraint_ids"] == [rule.constraint_id for rule in constraint_rules],
        "oracle query metadata constraint inventory differs",
    )
    _require(
        metadata["constraint_contract_sha256"] == constraint_contract_sha256,
        "oracle query metadata constraint contract differs",
    )
    _require(
        metadata["trusted_receipt_sha256"] == trusted_receipt,
        "oracle query metadata trusted receipt differs",
    )
    _require(
        metadata["query_identity_inventory_sha256"] == inventory_expected,
        "oracle query metadata identity inventory differs",
    )

    raw_rows = _strict_jsonl(verified.read_payload_bytes(QUERY_PAYLOAD_PATH))
    _require(
        type(metadata["row_count"]) is int and metadata["row_count"] == len(raw_rows),
        "oracle query metadata row count differs",
    )
    rows = tuple(
        _parse_query_row(
            raw,
            row_number=index,
            expected_run_id=expected_run_id,
            identity_fields=identity_fields,
            objective_ids=(protocol.primary_objectives[0], protocol.primary_objectives[1]),
            objective_bounds=(protocol.objective_bounds[0], protocol.objective_bounds[1]),
            constraint_rules=constraint_rules,
            expected_constraint_contract_sha256=constraint_contract_sha256,
        )
        for index, raw in enumerate(raw_rows, start=1)
    )
    identities = tuple(row.identity for row in rows)
    positions = tuple(row.position for row in rows)
    _require(len(set(identities)) == len(identities), "submitted query identity is duplicated")
    _require(len(set(positions)) == len(positions), "charged call position is duplicated")
    _require(
        positions == tuple(range(1, len(rows) + 1)),
        "charged call positions must be ordered and contiguous from one",
    )
    _require(
        protocol.call_checkpoints[0] <= len(rows) <= protocol.total_unique_calls,
        "authenticated query stream does not reach a valid sealed checkpoint",
    )
    _require(
        type(metadata["unique_query_identity_count"]) is int
        and metadata["unique_query_identity_count"] == len(rows),
        "oracle query metadata unique identity count differs",
    )
    inventory_observed = query_identity_inventory_sha256(identities)
    _require(inventory_observed == inventory_expected, "query identity inventory digest differs")

    sealed_checkpoints = tuple(
        checkpoint for checkpoint in protocol.call_checkpoints if checkpoint <= len(rows)
    )
    _require(bool(sealed_checkpoints), "authenticated query stream has no sealed checkpoint")
    sealed_values: list[tuple[int, float]] = []
    for checkpoint in sealed_checkpoints:
        points = tuple(
            row.objectives
            for row in rows[:checkpoint]
            if row.feasible and row.objectives is not None
        )
        sealed_values.append(
            (
                checkpoint,
                _hypervolume_2d(points, reference=(0.0, 0.0)),
            )
        )
    last_value = sealed_values[-1][1]
    carried = tuple(sealed_values) + tuple(
        (checkpoint, last_value) for checkpoint in protocol.call_checkpoints[len(sealed_values) :]
    )
    area = math.fsum(
        (right_checkpoint - left_checkpoint) * (left_value + right_value) / 2.0
        for (left_checkpoint, left_value), (right_checkpoint, right_value) in pairwise(carried)
    )
    normalized_auc = area / protocol.hypervolume_auc_denominator_calls
    _require(
        math.isfinite(normalized_auc) and 0.0 <= normalized_auc <= 1.0,
        "normalized hypervolume AUC is invalid",
    )
    feasible_identity_count = sum(row.feasible for row in rows)
    sealed_checkpoint_values = tuple(sealed_values)
    evidence_sha256 = sha256_bytes(
        canonical_json_bytes(
            _unsigned_evidence_document(
                run_id=expected_run_id,
                phase_seal_sha256=verified.seal_sha256,
                phase_receipt_sha256=verified.receipt_sha256,
                trusted_receipt_sha256=trusted_receipt,
                constraint_contract_sha256=constraint_contract_sha256,
                query_identity_inventory_sha256_value=inventory_observed,
                charged_submitted_identity_count=len(rows),
                feasible_identity_count=feasible_identity_count,
                sealed_checkpoints=sealed_checkpoint_values,
                carried_checkpoints=carried,
                normalized_hypervolume_auc=normalized_auc,
            )
        )
    )
    return PrimaryHypervolumeEvidence(
        run_id=expected_run_id,
        phase_seal_sha256=verified.seal_sha256,
        phase_receipt_sha256=verified.receipt_sha256,
        trusted_receipt_sha256=trusted_receipt,
        constraint_contract_sha256=constraint_contract_sha256,
        query_identity_inventory_sha256=inventory_observed,
        charged_submitted_identity_count=len(rows),
        feasible_identity_count=feasible_identity_count,
        sealed_checkpoints=sealed_checkpoint_values,
        carried_checkpoints=carried,
        normalized_hypervolume_auc=normalized_auc,
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "CONSTRAINT_CONTRACT_ARTIFACT",
    "QUERY_PAYLOAD_PATH",
    "QUERY_PHASE_ARTIFACT",
    "QUERY_ROW_ARTIFACT",
    "TRUSTED_RECEIPT_PREDECESSOR",
    "EvidenceMetricError",
    "PrimaryHypervolumeEvidence",
    "compute_authenticated_primary_hypervolume",
    "query_identity_inventory_sha256",
]

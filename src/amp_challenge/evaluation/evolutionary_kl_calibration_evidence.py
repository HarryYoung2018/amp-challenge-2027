"""Fail-closed calibration reduction from separately sealed raw cohorts.

The reducer never accepts producer-computed ECE, coverage, or feasibility
labels.  A caller must pin two distinct phase seals, their external authority
receipts, and a canonical constraint-semantics contract.  The truth phase must
also name the prediction phase as its presubmission predecessor.

The immutable evolutionary/KL v1 protocol remains execution-blocking.  This
module is evidence plumbing only and cannot authorize scientific or production
claims.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from amp_challenge.evaluation.evolutionary_kl_evidence_metrics import (
    TRUSTED_RECEIPT_PREDECESSOR,
    query_identity_inventory_sha256,
)
from amp_challenge.evaluation.evolutionary_kl_protocol import (
    FROZEN_PROTOCOL_SHA256,
    CalibrationSeedMetrics,
    EvolutionaryKLProtocol,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    sha256_bytes,
    verify_phase_capability,
)

PREDICTION_PHASE_ARTIFACT = "evolutionary_kl_presubmission_prediction_cohort_v1"
PREDICTION_ROW_ARTIFACT = "evolutionary_kl_presubmission_prediction_row_v1"
TRUTH_PHASE_ARTIFACT = "evolutionary_kl_authenticated_calibration_truth_cohort_v1"
TRUTH_ROW_ARTIFACT = "evolutionary_kl_authenticated_calibration_truth_row_v1"
CONSTRAINT_CONTRACT_ARTIFACT = "evolutionary_kl_constraint_semantics_v1"
PREDICTION_PAYLOAD_PATH = "presubmission-predictions.jsonl"
TRUTH_PAYLOAD_PATH = "calibration-truth.jsonl"
PREDICTION_PHASE_PREDECESSOR = "presubmission/prediction-phase"
_FROZEN_OBJECTIVE_IDS = ("gram_positive_activity", "gram_negative_activity")
_FROZEN_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
_FROZEN_ECE_BIN_COUNT = 10
_FROZEN_QUERY_IDENTITY_FIELDS = (
    "canonical_sequence_id",
    "oracle_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
    "replicate_id",
)
_MIN_QUERY_COUNT = 64
_MAX_QUERY_COUNT = 512

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONSTRAINT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_OPERATORS = frozenset({"lt", "le", "gt", "ge"})
_COMMON_METADATA_KEYS = {
    "schema_version",
    "protocol_sha256",
    "run_id",
    "method_id",
    "seed",
    "row_count",
    "unique_query_identity_count",
    "query_identity_fields",
    "query_identity_inventory_sha256",
    "objective_ids",
    "constraint_contract_sha256",
    "trusted_receipt_sha256",
}
_PREDICTION_METADATA_KEYS = frozenset(
    _COMMON_METADATA_KEYS | {"coverage_levels", "interval_convention", "prediction_stage"}
)
_TRUTH_METADATA_KEYS = frozenset(
    _COMMON_METADATA_KEYS | {"constraint_ids", "prediction_phase_seal_sha256"}
)
_PREDICTION_ROW_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "run_id",
        "method_id",
        "seed",
        "charged_call_position",
        "query_identity",
        "prediction_stage",
        "interval_convention",
        "constraint_contract_sha256",
        "joint_feasibility_probability",
        "objective_intervals",
    }
)
_TRUTH_ROW_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "run_id",
        "method_id",
        "seed",
        "charged_call_position",
        "query_identity",
        "response_status",
        "atomic_response_complete",
        "objectives",
        "objective_censored",
        "constraints",
    }
)
_INTERVAL_KEYS = frozenset({"lower", "upper"})
_TRUTH_CONSTRAINT_KEYS = frozenset({"value", "censored"})
_CONSTRAINT_CONTRACT_KEYS = frozenset(
    {"schema_version", "artifact", "status", "protocol_sha256", "constraints"}
)
_CONSTRAINT_RULE_KEYS = frozenset({"constraint_id", "operator", "threshold"})


class CalibrationEvidenceError(ValueError):
    """Raised when calibration evidence cannot be authenticated or reduced."""


@dataclass(frozen=True, slots=True)
class EqualMassBinEvidence:
    """Sufficient statistics for one deterministic equal-mass ECE bin."""

    bin_index: int
    count: int
    probability_sum: float
    feasible_truth_count: int
    contribution: float


@dataclass(frozen=True, slots=True)
class CoverageEvidence:
    """Inclusive interval-coverage evidence for one objective and level."""

    objective: str
    level: float
    covered_count: int
    total_count: int
    coverage: float


@dataclass(frozen=True, slots=True)
class CalibrationAtomEvidence:
    """One joined prediction/truth atom retained for exact pooled reduction."""

    charged_call_position: int
    query_identity: tuple[tuple[str, str], ...]
    joint_feasibility_probability: float
    feasible_truth: bool
    coverage_hits: tuple[tuple[str, tuple[tuple[float, bool], ...]], ...]


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    """Self-hashed calibration metrics for one exact method/seed cohort."""

    run_id: str
    method_id: str
    seed: int
    prediction_phase_seal_sha256: str
    prediction_phase_receipt_sha256: str
    prediction_trusted_receipt_sha256: str
    truth_phase_seal_sha256: str
    truth_phase_receipt_sha256: str
    truth_trusted_receipt_sha256: str
    constraint_contract_sha256: str
    query_identity_inventory_sha256: str
    query_count: int
    atoms: tuple[CalibrationAtomEvidence, ...]
    ece_bins: tuple[EqualMassBinEvidence, ...]
    ece: float
    objective_coverage: tuple[CoverageEvidence, ...]
    pooled_coverage: tuple[CoverageEvidence, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _validate_evidence(self)

    @property
    def coverage90(self) -> float:
        matches = tuple(item.coverage for item in self.pooled_coverage if item.level == 0.90)
        _require(len(matches) == 1, "calibration evidence lacks exact pooled 90% coverage")
        return matches[0]

    def seed_metrics(self) -> CalibrationSeedMetrics:
        """Return the frozen gate's per-seed scalar view."""

        _validate_evidence(self)
        return CalibrationSeedMetrics(
            ece=self.ece,
            coverage_by_level=tuple(item.coverage for item in self.pooled_coverage),
        )

    def document_bytes(self) -> bytes:
        """Serialize only after independently rechecking the canonical self-hash."""

        _validate_evidence(self)
        document = _evidence_document(self, include_digest=True)
        return canonical_json_bytes(document)


@dataclass(frozen=True, slots=True)
class _ConstraintRule:
    constraint_id: str
    operator: str
    threshold: float


@dataclass(frozen=True, slots=True)
class _Prediction:
    position: int
    identity: tuple[tuple[str, str], ...]
    probability: float
    intervals: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...]


@dataclass(frozen=True, slots=True)
class _Truth:
    position: int
    identity: tuple[tuple[str, str], ...]
    objectives: tuple[tuple[str, float], ...]
    feasible: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationEvidenceError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} invalid")
    return value


def _finite(value: object, *, label: str) -> float:
    _require(type(value) in {int, float}, f"{label} must be a non-Boolean real number")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise CalibrationEvidenceError(f"{label} must be finite") from error
    _require(math.isfinite(parsed), f"{label} must be finite")
    return parsed


def _exact_object(
    value: object, keys: set[str] | frozenset[str], *, label: str
) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(
        all(type(key) is str for key in value) and set(value) == set(keys), f"{label} keys differ"
    )
    return value


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    _require(type(payload) is bytes, f"{label} must be bytes")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CalibrationEvidenceError(f"{label} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise CalibrationEvidenceError(f"{label} contains invalid constant {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationEvidenceError(f"{label} is not strict UTF-8 JSON") from error
    _require(type(parsed) is dict, f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, ValueError) as error:
        raise CalibrationEvidenceError(f"{label} is not finite canonical JSON") from error
    _require(canonical == payload, f"{label} is not canonical JSON")
    return parsed


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    _require(type(payload) is bytes and bool(payload), f"{label} must be nonempty bytes")
    _require(payload.endswith(b"\n"), f"{label} must end in LF")
    rows = tuple(
        _strict_json_object(line, label=f"{label} row {index}")
        for index, line in enumerate(payload.splitlines(keepends=True), start=1)
    )
    _require(
        all(line not in {b"", b"\n"} for line in payload.splitlines(keepends=True)),
        f"{label} has blank rows",
    )
    return rows


def _identity(
    value: object,
    *,
    fields: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    document = _exact_object(value, set(fields), label=label)
    result: list[tuple[str, str]] = []
    for field in fields:
        item = document[field]
        _require(type(item) is str and bool(item), f"{label}.{field} must be nonempty text")
        if field == "canonical_sequence_id" or field.endswith("sha256"):
            _sha256(item, label=f"{label}.{field}")
        result.append((field, item))
    return tuple(result)


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
        type(raw_rules) is list and bool(raw_rules), "constraint rules must be a nonempty array"
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
                threshold=_finite(rule["threshold"], label=f"constraint rule {index} threshold"),
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


def _validate_common_metadata(
    metadata: dict[str, object],
    *,
    expected_run_id: str,
    expected_method_id: str,
    expected_seed: int,
    expected_inventory: str,
    expected_row_count: int,
    expected_trusted_receipt: str,
    expected_constraint_contract: str,
    protocol: EvolutionaryKLProtocol,
) -> None:
    _require(
        metadata["schema_version"] == 1 and type(metadata["schema_version"]) is int,
        "cohort metadata schema differs",
    )
    _require(
        metadata["protocol_sha256"] == FROZEN_PROTOCOL_SHA256, "cohort metadata protocol differs"
    )
    _require(
        metadata["run_id"] == expected_run_id and type(metadata["run_id"]) is str,
        "cohort metadata run differs",
    )
    _require(
        metadata["method_id"] == expected_method_id and type(metadata["method_id"]) is str,
        "cohort metadata method differs",
    )
    _require(
        metadata["seed"] == expected_seed and type(metadata["seed"]) is int,
        "cohort metadata seed differs",
    )
    _require(
        metadata["row_count"] == expected_row_count and type(metadata["row_count"]) is int,
        "cohort metadata row count differs",
    )
    _require(
        metadata["unique_query_identity_count"] == expected_row_count
        and type(metadata["unique_query_identity_count"]) is int,
        "cohort metadata unique count differs",
    )
    _require(
        metadata["query_identity_fields"]
        == list(protocol.oracle_query_contract.query_identity_fields),
        "cohort identity fields differ",
    )
    _require(
        metadata["query_identity_inventory_sha256"] == expected_inventory,
        "cohort identity inventory differs",
    )
    _require(
        metadata["objective_ids"] == list(protocol.primary_objectives),
        "cohort objective inventory differs",
    )
    _require(
        metadata["constraint_contract_sha256"] == expected_constraint_contract,
        "cohort constraint contract differs",
    )
    _require(
        metadata["trusted_receipt_sha256"] == expected_trusted_receipt,
        "cohort trusted receipt differs",
    )


def _parse_prediction(
    raw: dict[str, object],
    *,
    index: int,
    protocol: EvolutionaryKLProtocol,
    run_id: str,
    method_id: str,
    seed: int,
    contract_sha256: str,
) -> _Prediction:
    row = _exact_object(raw, _PREDICTION_ROW_KEYS, label=f"prediction row {index}")
    _require(
        row["schema_version"] == 1 and type(row["schema_version"]) is int,
        f"prediction row {index} schema differs",
    )
    _require(
        row["artifact"] == PREDICTION_ROW_ARTIFACT and type(row["artifact"]) is str,
        f"prediction row {index} artifact differs",
    )
    _require(
        row["run_id"] == run_id and type(row["run_id"]) is str,
        f"prediction row {index} run differs",
    )
    _require(
        row["method_id"] == method_id and type(row["method_id"]) is str,
        f"prediction row {index} method differs",
    )
    _require(
        row["seed"] == seed and type(row["seed"]) is int, f"prediction row {index} seed differs"
    )
    position = row["charged_call_position"]
    _require(type(position) is int and position >= 1, f"prediction row {index} position invalid")
    _require(
        row["prediction_stage"] == "sealed_before_matching_oracle_submission",
        f"prediction row {index} is not presubmission",
    )
    _require(
        row["interval_convention"] == "equal_tailed_marginal",
        f"prediction row {index} interval convention differs",
    )
    _require(
        row["constraint_contract_sha256"] == contract_sha256,
        f"prediction row {index} constraint contract differs",
    )
    identity = _identity(
        row["query_identity"],
        fields=protocol.oracle_query_contract.query_identity_fields,
        label=f"prediction row {index} identity",
    )
    probability = _finite(
        row["joint_feasibility_probability"], label=f"prediction row {index} probability"
    )
    _require(0.0 <= probability <= 1.0, f"prediction row {index} probability outside [0, 1]")
    interval_document = _exact_object(
        row["objective_intervals"],
        set(protocol.primary_objectives),
        label=f"prediction row {index} intervals",
    )
    parsed_intervals: list[tuple[str, tuple[tuple[float, float, float], ...]]] = []
    level_labels = tuple(f"{level:.2f}" for level in protocol.coverage_levels)
    for objective in protocol.primary_objectives:
        by_level = _exact_object(
            interval_document[objective],
            set(level_labels),
            label=f"prediction row {index} {objective} intervals",
        )
        intervals: list[tuple[float, float, float]] = []
        for level, level_label in zip(protocol.coverage_levels, level_labels, strict=True):
            bounds = _exact_object(
                by_level[level_label],
                _INTERVAL_KEYS,
                label=f"prediction row {index} {objective} {level_label} interval",
            )
            lower = _finite(
                bounds["lower"], label=f"prediction row {index} {objective} {level_label} lower"
            )
            upper = _finite(
                bounds["upper"], label=f"prediction row {index} {objective} {level_label} upper"
            )
            _require(
                protocol.objective_bounds[0] <= lower <= upper <= protocol.objective_bounds[1],
                f"prediction row {index} interval outside frozen bounds",
            )
            if intervals:
                _previous_level, previous_lower, previous_upper = intervals[-1]
                _require(
                    lower <= previous_lower and upper >= previous_upper,
                    f"prediction row {index} equal-tailed intervals are not nested",
                )
            intervals.append((level, lower, upper))
        parsed_intervals.append((objective, tuple(intervals)))
    return _Prediction(
        position=position,
        identity=identity,
        probability=probability,
        intervals=tuple(parsed_intervals),
    )


def _parse_truth(
    raw: dict[str, object],
    *,
    index: int,
    protocol: EvolutionaryKLProtocol,
    run_id: str,
    method_id: str,
    seed: int,
    rules: tuple[_ConstraintRule, ...],
) -> _Truth:
    row = _exact_object(raw, _TRUTH_ROW_KEYS, label=f"truth row {index}")
    _require(
        row["schema_version"] == 1 and type(row["schema_version"]) is int,
        f"truth row {index} schema differs",
    )
    _require(
        row["artifact"] == TRUTH_ROW_ARTIFACT and type(row["artifact"]) is str,
        f"truth row {index} artifact differs",
    )
    _require(
        row["run_id"] == run_id and type(row["run_id"]) is str, f"truth row {index} run differs"
    )
    _require(
        row["method_id"] == method_id and type(row["method_id"]) is str,
        f"truth row {index} method differs",
    )
    _require(row["seed"] == seed and type(row["seed"]) is int, f"truth row {index} seed differs")
    position = row["charged_call_position"]
    _require(type(position) is int and position >= 1, f"truth row {index} position invalid")
    _require(row["response_status"] == "complete", f"truth row {index} response is incomplete")
    _require(row["atomic_response_complete"] is True, f"truth row {index} response is not atomic")
    identity = _identity(
        row["query_identity"],
        fields=protocol.oracle_query_contract.query_identity_fields,
        label=f"truth row {index} identity",
    )
    objective_document = _exact_object(
        row["objectives"], set(protocol.primary_objectives), label=f"truth row {index} objectives"
    )
    censor_document = _exact_object(
        row["objective_censored"],
        set(protocol.primary_objectives),
        label=f"truth row {index} censoring",
    )
    objectives: list[tuple[str, float]] = []
    for objective in protocol.primary_objectives:
        value = _finite(
            objective_document[objective], label=f"truth row {index} objective {objective}"
        )
        _require(
            protocol.objective_bounds[0] <= value <= protocol.objective_bounds[1],
            f"truth row {index} objective outside frozen bounds",
        )
        _require(censor_document[objective] is False, f"truth row {index} objective is censored")
        objectives.append((objective, value))
    constraint_document = _exact_object(
        row["constraints"],
        {rule.constraint_id for rule in rules},
        label=f"truth row {index} constraints",
    )
    passes: list[bool] = []
    for rule in rules:
        item = _exact_object(
            constraint_document[rule.constraint_id],
            _TRUTH_CONSTRAINT_KEYS,
            label=f"truth row {index} constraint {rule.constraint_id}",
        )
        value = _finite(
            item["value"], label=f"truth row {index} constraint {rule.constraint_id} value"
        )
        _require(item["censored"] is False, f"truth row {index} constraint is censored")
        passes.append(_rule_passes(rule, value))
    return _Truth(
        position=position,
        identity=identity,
        objectives=tuple(objectives),
        feasible=all(passes),
    )


def _coverage_document(item: CoverageEvidence) -> dict[str, object]:
    return {
        "objective": item.objective,
        "level_hex": item.level.hex(),
        "covered_count": item.covered_count,
        "total_count": item.total_count,
        "coverage_hex": item.coverage.hex(),
    }


def _atom_document(item: CalibrationAtomEvidence) -> dict[str, object]:
    return {
        "charged_call_position": item.charged_call_position,
        "query_identity": dict(item.query_identity),
        "joint_feasibility_probability_hex": item.joint_feasibility_probability.hex(),
        "feasible_truth": item.feasible_truth,
        "coverage_hits": {
            objective: {f"{level:.2f}": covered for level, covered in coverage_by_level}
            for objective, coverage_by_level in item.coverage_hits
        },
    }


def _unsigned_evidence_document(
    *,
    run_id: str,
    method_id: str,
    seed: int,
    prediction_phase_seal_sha256: str,
    prediction_phase_receipt_sha256: str,
    prediction_trusted_receipt_sha256: str,
    truth_phase_seal_sha256: str,
    truth_phase_receipt_sha256: str,
    truth_trusted_receipt_sha256: str,
    constraint_contract_sha256: str,
    query_identity_inventory_sha256_value: str,
    query_count: int,
    atoms: tuple[CalibrationAtomEvidence, ...],
    ece_bins: tuple[EqualMassBinEvidence, ...],
    ece: float,
    objective_coverage: tuple[CoverageEvidence, ...],
    pooled_coverage: tuple[CoverageEvidence, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": "evolutionary_kl_authenticated_calibration_evidence_v1",
        "status": "derived_from_sealed_presubmission_predictions_and_authenticated_truth",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "run_id": run_id,
        "method_id": method_id,
        "seed": seed,
        "prediction_phase_seal_sha256": prediction_phase_seal_sha256,
        "prediction_phase_receipt_sha256": prediction_phase_receipt_sha256,
        "prediction_trusted_receipt_sha256": prediction_trusted_receipt_sha256,
        "truth_phase_seal_sha256": truth_phase_seal_sha256,
        "truth_phase_receipt_sha256": truth_phase_receipt_sha256,
        "truth_trusted_receipt_sha256": truth_trusted_receipt_sha256,
        "constraint_contract_sha256": constraint_contract_sha256,
        "query_identity_inventory_sha256": query_identity_inventory_sha256_value,
        "query_count": query_count,
        "atoms": [_atom_document(item) for item in atoms],
        "ece_bins": [
            {
                "bin_index": item.bin_index,
                "count": item.count,
                "probability_sum_hex": item.probability_sum.hex(),
                "feasible_truth_count": item.feasible_truth_count,
                "contribution_hex": item.contribution.hex(),
            }
            for item in ece_bins
        ],
        "ece_hex": ece.hex(),
        "objective_coverage": [_coverage_document(item) for item in objective_coverage],
        "pooled_coverage": [_coverage_document(item) for item in pooled_coverage],
    }


def _evidence_document(evidence: CalibrationEvidence, *, include_digest: bool) -> dict[str, object]:
    document = _unsigned_evidence_document(
        run_id=evidence.run_id,
        method_id=evidence.method_id,
        seed=evidence.seed,
        prediction_phase_seal_sha256=evidence.prediction_phase_seal_sha256,
        prediction_phase_receipt_sha256=evidence.prediction_phase_receipt_sha256,
        prediction_trusted_receipt_sha256=evidence.prediction_trusted_receipt_sha256,
        truth_phase_seal_sha256=evidence.truth_phase_seal_sha256,
        truth_phase_receipt_sha256=evidence.truth_phase_receipt_sha256,
        truth_trusted_receipt_sha256=evidence.truth_trusted_receipt_sha256,
        constraint_contract_sha256=evidence.constraint_contract_sha256,
        query_identity_inventory_sha256_value=evidence.query_identity_inventory_sha256,
        query_count=evidence.query_count,
        atoms=evidence.atoms,
        ece_bins=evidence.ece_bins,
        ece=evidence.ece,
        objective_coverage=evidence.objective_coverage,
        pooled_coverage=evidence.pooled_coverage,
    )
    if include_digest:
        document["evidence_sha256"] = evidence.evidence_sha256
    return document


def _validate_evidence(evidence: CalibrationEvidence) -> None:
    _require(
        type(evidence.run_id) is str and _RUN_ID_RE.fullmatch(evidence.run_id) is not None,
        "evidence run ID invalid",
    )
    _require(
        type(evidence.method_id) is str
        and _CONSTRAINT_ID_RE.fullmatch(evidence.method_id) is not None,
        "evidence method ID invalid",
    )
    _require(type(evidence.seed) is int and evidence.seed >= 0, "evidence seed invalid")
    for label, digest in (
        ("prediction phase seal", evidence.prediction_phase_seal_sha256),
        ("prediction phase receipt", evidence.prediction_phase_receipt_sha256),
        ("prediction trusted receipt", evidence.prediction_trusted_receipt_sha256),
        ("truth phase seal", evidence.truth_phase_seal_sha256),
        ("truth phase receipt", evidence.truth_phase_receipt_sha256),
        ("truth trusted receipt", evidence.truth_trusted_receipt_sha256),
        ("constraint contract", evidence.constraint_contract_sha256),
        ("query inventory", evidence.query_identity_inventory_sha256),
        ("evidence", evidence.evidence_sha256),
    ):
        _sha256(digest, label=label)
    _require(
        len(
            {
                evidence.prediction_phase_seal_sha256,
                evidence.prediction_phase_receipt_sha256,
                evidence.prediction_trusted_receipt_sha256,
                evidence.truth_phase_seal_sha256,
                evidence.truth_phase_receipt_sha256,
                evidence.truth_trusted_receipt_sha256,
            }
        )
        == 6,
        "evidence phase and authority digests must be distinct",
    )
    _require(
        type(evidence.query_count) is int
        and _MIN_QUERY_COUNT <= evidence.query_count <= _MAX_QUERY_COUNT,
        "evidence query count invalid",
    )
    _require(
        type(evidence.atoms) is tuple
        and len(evidence.atoms) == evidence.query_count
        and all(type(item) is CalibrationAtomEvidence for item in evidence.atoms),
        "evidence calibration atoms invalid",
    )
    for expected_position, atom in enumerate(evidence.atoms, start=1):
        _require(
            type(atom.charged_call_position) is int
            and atom.charged_call_position == expected_position,
            "evidence atom charged positions differ",
        )
        _require(
            type(atom.query_identity) is tuple
            and tuple(field for field, _value in atom.query_identity)
            == _FROZEN_QUERY_IDENTITY_FIELDS,
            "evidence atom query identity fields differ",
        )
        for field, value in atom.query_identity:
            _require(type(value) is str and bool(value), "evidence atom identity value invalid")
            if field == "canonical_sequence_id" or field.endswith("sha256"):
                _sha256(value, label=f"evidence atom identity {field}")
        _require(
            type(atom.joint_feasibility_probability) is float
            and math.isfinite(atom.joint_feasibility_probability)
            and 0.0 <= atom.joint_feasibility_probability <= 1.0,
            "evidence atom probability invalid",
        )
        _require(type(atom.feasible_truth) is bool, "evidence atom truth invalid")
        _require(
            type(atom.coverage_hits) is tuple
            and tuple(objective for objective, _hits in atom.coverage_hits)
            == _FROZEN_OBJECTIVE_IDS,
            "evidence atom objective coverage inventory differs",
        )
        for _objective, hits in atom.coverage_hits:
            _require(
                type(hits) is tuple
                and tuple(level for level, _covered in hits) == _FROZEN_COVERAGE_LEVELS
                and all(type(covered) is bool for _level, covered in hits),
                "evidence atom coverage levels or values differ",
            )
    identities = tuple(atom.query_identity for atom in evidence.atoms)
    try:
        inventory = query_identity_inventory_sha256(identities)
    except ValueError as error:
        raise CalibrationEvidenceError("evidence atom identity inventory invalid") from error
    _require(
        inventory == evidence.query_identity_inventory_sha256,
        "evidence atom identity inventory differs",
    )
    expected_bins, expected_ece, expected_objective, expected_pooled = _reduce_atoms_contract(
        evidence.atoms,
        bin_count=_FROZEN_ECE_BIN_COUNT,
        objective_ids=_FROZEN_OBJECTIVE_IDS,
        coverage_levels=_FROZEN_COVERAGE_LEVELS,
    )
    _require(
        evidence.ece_bins == expected_bins
        and type(evidence.ece) is float
        and evidence.ece == expected_ece
        and evidence.objective_coverage == expected_objective
        and evidence.pooled_coverage == expected_pooled,
        "evidence calibration reduction differs from raw atoms",
    )
    observed = sha256_bytes(
        canonical_json_bytes(_evidence_document(evidence, include_digest=False))
    )
    _require(observed == evidence.evidence_sha256, "calibration evidence self-hash differs")


def _reduce_atoms_contract(
    atoms: tuple[CalibrationAtomEvidence, ...],
    *,
    bin_count: int,
    objective_ids: tuple[str, ...],
    coverage_levels: tuple[float, ...],
) -> tuple[
    tuple[EqualMassBinEvidence, ...],
    float,
    tuple[CoverageEvidence, ...],
    tuple[CoverageEvidence, ...],
]:
    _require(len(atoms) >= bin_count, "calibration atom cohort is too small")
    ordered = sorted(
        atoms,
        key=lambda item: (
            item.joint_feasibility_probability,
            canonical_json_bytes(dict(item.query_identity)),
        ),
    )
    ece_bins: list[EqualMassBinEvidence] = []
    for bin_index in range(bin_count):
        members = tuple(
            atom
            for rank, atom in enumerate(ordered)
            if rank * bin_count // len(ordered) == bin_index
        )
        _require(bool(members), "equal-mass ECE bin is empty")
        probability_sum = math.fsum(item.joint_feasibility_probability for item in members)
        truth_count = sum(item.feasible_truth for item in members)
        contribution = (
            len(members)
            / len(ordered)
            * abs(probability_sum / len(members) - truth_count / len(members))
        )
        ece_bins.append(
            EqualMassBinEvidence(
                bin_index=bin_index,
                count=len(members),
                probability_sum=probability_sum,
                feasible_truth_count=truth_count,
                contribution=contribution,
            )
        )
    ece = math.fsum(item.contribution for item in ece_bins)

    objective_coverage: list[CoverageEvidence] = []
    pooled_hits = {level: 0 for level in coverage_levels}
    for objective in objective_ids:
        for level in coverage_levels:
            hits = sum(dict(dict(atom.coverage_hits)[objective])[level] for atom in atoms)
            pooled_hits[level] += hits
            objective_coverage.append(
                CoverageEvidence(
                    objective=objective,
                    level=level,
                    covered_count=hits,
                    total_count=len(atoms),
                    coverage=hits / len(atoms),
                )
            )
    pooled_coverage = tuple(
        CoverageEvidence(
            objective="pooled_objective_pairs",
            level=level,
            covered_count=pooled_hits[level],
            total_count=len(objective_ids) * len(atoms),
            coverage=pooled_hits[level] / (len(objective_ids) * len(atoms)),
        )
        for level in coverage_levels
    )
    return tuple(ece_bins), ece, tuple(objective_coverage), pooled_coverage


def _reduce_atoms(
    protocol: EvolutionaryKLProtocol,
    atoms: tuple[CalibrationAtomEvidence, ...],
) -> tuple[
    tuple[EqualMassBinEvidence, ...],
    float,
    tuple[CoverageEvidence, ...],
    tuple[CoverageEvidence, ...],
]:
    return _reduce_atoms_contract(
        atoms,
        bin_count=protocol.ece_equal_mass_bins,
        objective_ids=protocol.primary_objectives,
        coverage_levels=protocol.coverage_levels,
    )


def pooled_calibration_metrics(
    protocol: EvolutionaryKLProtocol,
    evidence_by_seed: Mapping[int, CalibrationEvidence],
    *,
    expected_method_id: str,
    expected_seeds: tuple[int, ...],
    expected_evidence_sha256_by_seed: Mapping[int, str],
) -> CalibrationSeedMetrics:
    """Pool raw atoms only after binding every seed to external evidence authority.

    The expected digest mapping is an out-of-band controller trust anchor. A
    digest copied from an evidence object is not independent authority.
    """

    _require(type(protocol) is EvolutionaryKLProtocol, "protocol type differs")
    _require(
        type(expected_method_id) is str and expected_method_id in protocol.confirmation_method_ids,
        "expected pooled method differs",
    )
    _require(
        type(expected_seeds) is tuple
        and bool(expected_seeds)
        and all(type(seed) is int and seed >= 0 for seed in expected_seeds)
        and len(set(expected_seeds)) == len(expected_seeds),
        "expected pooled seeds are invalid",
    )
    _require(
        isinstance(evidence_by_seed, Mapping)
        and all(type(seed) is int for seed in evidence_by_seed)
        and set(evidence_by_seed) == set(expected_seeds),
        "pooled calibration seed census differs",
    )
    _require(
        isinstance(expected_evidence_sha256_by_seed, Mapping)
        and all(type(seed) is int for seed in expected_evidence_sha256_by_seed)
        and set(expected_evidence_sha256_by_seed) == set(expected_seeds),
        "external calibration evidence digest census differs",
    )
    expected_digests = {
        seed: _sha256(digest, label=f"pooled seed {seed} expected evidence")
        for seed, digest in expected_evidence_sha256_by_seed.items()
    }
    _require(
        len(set(expected_digests.values())) == len(expected_digests),
        "external calibration evidence digests must be unique",
    )
    atoms: list[CalibrationAtomEvidence] = []
    truth_by_identity: dict[tuple[tuple[str, str], ...], bool] = {}
    for seed in expected_seeds:
        evidence = evidence_by_seed[seed]
        _require(type(evidence) is CalibrationEvidence, f"pooled seed {seed} evidence type differs")
        _validate_evidence(evidence)
        _require(
            evidence.seed == seed and evidence.method_id == expected_method_id,
            f"pooled seed {seed} evidence identity differs",
        )
        _require(
            evidence.evidence_sha256 == expected_digests[seed],
            f"pooled seed {seed} evidence differs from external authority",
        )
        for atom in evidence.atoms:
            prior_truth = truth_by_identity.get(atom.query_identity)
            _require(
                prior_truth is None or prior_truth is atom.feasible_truth,
                "pooled calibration truth conflicts for one exact query identity",
            )
            truth_by_identity[atom.query_identity] = atom.feasible_truth
            atoms.append(atom)
    _bins, ece, _objective_coverage, pooled_coverage = _reduce_atoms(protocol, tuple(atoms))
    return CalibrationSeedMetrics(
        ece=ece,
        coverage_by_level=tuple(item.coverage for item in pooled_coverage),
    )


def compute_authenticated_calibration_evidence(
    protocol: EvolutionaryKLProtocol,
    prediction_capability: PhaseSeal,
    truth_capability: PhaseSeal,
    *,
    expected_prediction_phase_seal_sha256: str,
    expected_prediction_trusted_receipt_sha256: str,
    expected_truth_phase_seal_sha256: str,
    expected_truth_trusted_receipt_sha256: str,
    expected_run_id: str,
    expected_method_id: str,
    expected_seed: int,
    expected_query_identity_inventory_sha256: str,
    constraint_contract_bytes: bytes,
    expected_constraint_contract_sha256: str,
) -> CalibrationEvidence:
    """Recompute ECE and marginal coverage from exact joined raw cohorts."""

    _require(type(protocol) is EvolutionaryKLProtocol, "protocol type differs")
    prediction_seal = _sha256(
        expected_prediction_phase_seal_sha256, label="expected prediction phase seal"
    )
    prediction_trust = _sha256(
        expected_prediction_trusted_receipt_sha256, label="expected prediction trusted receipt"
    )
    truth_seal = _sha256(expected_truth_phase_seal_sha256, label="expected truth phase seal")
    truth_trust = _sha256(
        expected_truth_trusted_receipt_sha256, label="expected truth trusted receipt"
    )
    inventory_expected = _sha256(
        expected_query_identity_inventory_sha256, label="expected query inventory"
    )
    _require(
        len({prediction_seal, prediction_trust, truth_seal, truth_trust}) == 4,
        "phase and authority digests must be distinct",
    )
    _require(
        type(expected_run_id) is str and _RUN_ID_RE.fullmatch(expected_run_id) is not None,
        "expected run ID invalid",
    )
    _require(
        type(expected_method_id) is str and expected_method_id in protocol.confirmation_method_ids,
        "expected method is outside calibration cohort",
    )
    _require(type(expected_seed) is int and expected_seed >= 0, "expected seed invalid")
    _require(
        protocol.ece_equal_mass_bins == _FROZEN_ECE_BIN_COUNT
        and protocol.coverage_levels == _FROZEN_COVERAGE_LEVELS
        and protocol.primary_objectives == _FROZEN_OBJECTIVE_IDS
        and protocol.objective_bounds == (0.0, 1.0)
        and protocol.oracle_query_contract.query_identity_fields == _FROZEN_QUERY_IDENTITY_FIELDS
        and protocol.initial_design_unique_calls == _MIN_QUERY_COUNT
        and protocol.total_unique_calls == _MAX_QUERY_COUNT,
        "protocol calibration contract differs",
    )
    rules = _parse_constraint_contract(
        constraint_contract_bytes,
        expected_sha256=expected_constraint_contract_sha256,
    )
    contract_sha256 = sha256_bytes(constraint_contract_bytes)

    try:
        prediction_phase = verify_phase_capability(
            prediction_capability,
            expected_artifact=PREDICTION_PHASE_ARTIFACT,
            expected_payload_paths=(PREDICTION_PAYLOAD_PATH,),
            expected_predecessor_seals={TRUSTED_RECEIPT_PREDECESSOR: prediction_trust},
            expected_seal_sha256=prediction_seal,
        )
        truth_phase = verify_phase_capability(
            truth_capability,
            expected_artifact=TRUTH_PHASE_ARTIFACT,
            expected_payload_paths=(TRUTH_PAYLOAD_PATH,),
            expected_predecessor_seals={
                PREDICTION_PHASE_PREDECESSOR: prediction_seal,
                TRUSTED_RECEIPT_PREDECESSOR: truth_trust,
            },
            expected_seal_sha256=truth_seal,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise CalibrationEvidenceError("calibration phase authentication failed") from error
    _require(
        prediction_trust not in {prediction_phase.seal_sha256, prediction_phase.receipt_sha256}
        and truth_trust not in {truth_phase.seal_sha256, truth_phase.receipt_sha256},
        "producer phase cannot stand in for external calibration authority",
    )

    prediction_rows_raw = _strict_jsonl(
        prediction_phase.read_payload_bytes(PREDICTION_PAYLOAD_PATH),
        label="prediction cohort",
    )
    truth_rows_raw = _strict_jsonl(
        truth_phase.read_payload_bytes(TRUTH_PAYLOAD_PATH),
        label="truth cohort",
    )
    _require(
        len(prediction_rows_raw) == len(truth_rows_raw)
        and protocol.initial_design_unique_calls
        <= len(prediction_rows_raw)
        <= protocol.total_unique_calls,
        "prediction/truth cohort census is incomplete",
    )
    for phase, keys, rows, trust in (
        (prediction_phase, _PREDICTION_METADATA_KEYS, prediction_rows_raw, prediction_trust),
        (truth_phase, _TRUTH_METADATA_KEYS, truth_rows_raw, truth_trust),
    ):
        metadata = _exact_object(
            _strict_json_object(phase.metadata_json, label="calibration phase metadata"),
            keys,
            label="calibration phase metadata",
        )
        _validate_common_metadata(
            metadata,
            expected_run_id=expected_run_id,
            expected_method_id=expected_method_id,
            expected_seed=expected_seed,
            expected_inventory=inventory_expected,
            expected_row_count=len(rows),
            expected_trusted_receipt=trust,
            expected_constraint_contract=contract_sha256,
            protocol=protocol,
        )
        if phase.artifact == PREDICTION_PHASE_ARTIFACT:
            _require(
                metadata["coverage_levels"] == list(protocol.coverage_levels),
                "prediction coverage levels differ",
            )
            _require(
                metadata["interval_convention"] == "equal_tailed_marginal",
                "prediction interval convention differs",
            )
            _require(
                metadata["prediction_stage"] == "sealed_before_matching_oracle_submission",
                "prediction cohort is post-submission",
            )
        else:
            _require(
                metadata["constraint_ids"] == [rule.constraint_id for rule in rules],
                "truth constraint inventory differs",
            )
            _require(
                metadata["prediction_phase_seal_sha256"] == prediction_seal,
                "truth does not bind prediction phase",
            )

    predictions = tuple(
        _parse_prediction(
            row,
            index=index,
            protocol=protocol,
            run_id=expected_run_id,
            method_id=expected_method_id,
            seed=expected_seed,
            contract_sha256=contract_sha256,
        )
        for index, row in enumerate(prediction_rows_raw, start=1)
    )
    truths = tuple(
        _parse_truth(
            row,
            index=index,
            protocol=protocol,
            run_id=expected_run_id,
            method_id=expected_method_id,
            seed=expected_seed,
            rules=rules,
        )
        for index, row in enumerate(truth_rows_raw, start=1)
    )
    for label, rows in (("prediction", predictions), ("truth", truths)):
        identities = tuple(row.identity for row in rows)
        positions = tuple(row.position for row in rows)
        _require(len(set(identities)) == len(rows), f"{label} query identity duplicated")
        _require(len(set(positions)) == len(rows), f"{label} charged position duplicated")
        _require(
            positions == tuple(range(1, len(rows) + 1)), f"{label} charged positions not contiguous"
        )
        try:
            inventory = query_identity_inventory_sha256(identities)
        except ValueError as error:
            raise CalibrationEvidenceError(f"{label} identity inventory invalid") from error
        _require(inventory == inventory_expected, f"{label} query inventory differs")
    _require(
        tuple((row.position, row.identity) for row in predictions)
        == tuple((row.position, row.identity) for row in truths),
        "prediction and truth identity streams do not join exactly",
    )

    atoms: list[CalibrationAtomEvidence] = []
    for prediction, truth in zip(predictions, truths, strict=True):
        objective_values = dict(truth.objectives)
        interval_values = dict(prediction.intervals)
        coverage_hits = tuple(
            (
                objective,
                tuple(
                    (level, lower <= objective_values[objective] <= upper)
                    for level, lower, upper in interval_values[objective]
                ),
            )
            for objective in protocol.primary_objectives
        )
        atoms.append(
            CalibrationAtomEvidence(
                charged_call_position=prediction.position,
                query_identity=prediction.identity,
                joint_feasibility_probability=prediction.probability,
                feasible_truth=truth.feasible,
                coverage_hits=coverage_hits,
            )
        )
    atom_tuple = tuple(atoms)
    ece_bins_tuple, ece, objective_coverage_tuple, pooled_coverage = _reduce_atoms(
        protocol,
        atom_tuple,
    )
    unsigned = _unsigned_evidence_document(
        run_id=expected_run_id,
        method_id=expected_method_id,
        seed=expected_seed,
        prediction_phase_seal_sha256=prediction_phase.seal_sha256,
        prediction_phase_receipt_sha256=prediction_phase.receipt_sha256,
        prediction_trusted_receipt_sha256=prediction_trust,
        truth_phase_seal_sha256=truth_phase.seal_sha256,
        truth_phase_receipt_sha256=truth_phase.receipt_sha256,
        truth_trusted_receipt_sha256=truth_trust,
        constraint_contract_sha256=contract_sha256,
        query_identity_inventory_sha256_value=inventory_expected,
        query_count=len(truths),
        atoms=atom_tuple,
        ece_bins=ece_bins_tuple,
        ece=ece,
        objective_coverage=objective_coverage_tuple,
        pooled_coverage=pooled_coverage,
    )
    evidence_sha256 = sha256_bytes(canonical_json_bytes(unsigned))
    return CalibrationEvidence(
        run_id=expected_run_id,
        method_id=expected_method_id,
        seed=expected_seed,
        prediction_phase_seal_sha256=prediction_phase.seal_sha256,
        prediction_phase_receipt_sha256=prediction_phase.receipt_sha256,
        prediction_trusted_receipt_sha256=prediction_trust,
        truth_phase_seal_sha256=truth_phase.seal_sha256,
        truth_phase_receipt_sha256=truth_phase.receipt_sha256,
        truth_trusted_receipt_sha256=truth_trust,
        constraint_contract_sha256=contract_sha256,
        query_identity_inventory_sha256=inventory_expected,
        query_count=len(truths),
        atoms=atom_tuple,
        ece_bins=ece_bins_tuple,
        ece=ece,
        objective_coverage=objective_coverage_tuple,
        pooled_coverage=pooled_coverage,
        evidence_sha256=evidence_sha256,
    )


__all__ = [
    "CONSTRAINT_CONTRACT_ARTIFACT",
    "PREDICTION_PAYLOAD_PATH",
    "PREDICTION_PHASE_ARTIFACT",
    "PREDICTION_PHASE_PREDECESSOR",
    "PREDICTION_ROW_ARTIFACT",
    "TRUTH_PAYLOAD_PATH",
    "TRUTH_PHASE_ARTIFACT",
    "TRUTH_ROW_ARTIFACT",
    "CalibrationAtomEvidence",
    "CalibrationEvidence",
    "CalibrationEvidenceError",
    "CoverageEvidence",
    "EqualMassBinEvidence",
    "compute_authenticated_calibration_evidence",
    "pooled_calibration_metrics",
]

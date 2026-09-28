"""Authenticated paired reporting for the blocked evolutionary/KL v1 study.

The reducer accepts exactly the full method and two primary comparators over one
five-seed cohort.  Evidence is usable only when an independently supplied
digest authenticates the complete phase/method/seed inventory.  Reports are
development-only summaries: neither phase can authorize a confirmatory claim
or promotion under the published v1 protocol.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Literal

from amp_challenge.evaluation.evolutionary_kl_evidence_metrics import (
    PrimaryHypervolumeEvidence,
)
from amp_challenge.evaluation.evolutionary_kl_protocol import (
    CONFIRMATION_METHOD_IDS,
    CONFIRMATION_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    PRIMARY_COMPARATOR_IDS,
    SCREEN_SEEDS,
    EvolutionaryKLProtocol,
)
from amp_challenge.evaluation.sequential_v2_seals import canonical_json_bytes, sha256_bytes

Phase = Literal["screen", "confirmation"]
FULL_METHOD_ID = "counterfactual_softkg_evolutionary_diffusion"
INVENTORY_ARTIFACT = "evolutionary_kl_authoritative_primary_evidence_inventory_v1"
REPORT_ARTIFACT = "evolutionary_kl_authenticated_paired_statistical_report_v1"
BOOTSTRAP_ALGORITHM = "sha256_counter_rejection_uint64_be_seed_block_v1"
BOOTSTRAP_DRAW_DIGEST_DOMAIN = b"amp/evolutionary-kl/bootstrap-draw-inventory/v1\0"
BOOTSTRAP_INDEX_DOMAIN = b"amp/evolutionary-kl/bootstrap-index/v1\0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class StatisticalReportingError(ValueError):
    """Raised when paired evidence or reporting state fails closed."""


@dataclass(frozen=True, slots=True)
class AuthoritativePrimaryEvidenceDigest:
    """One externally authenticated phase/method/seed inventory row."""

    phase: Phase
    method_id: str
    seed: int
    run_id: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _validate_inventory_row(self)


@dataclass(frozen=True, slots=True)
class PrimaryEvidenceAssignment:
    """One evidence object assigned to its claimed scientific cohort slot."""

    phase: Phase
    method_id: str
    seed: int
    evidence: PrimaryHypervolumeEvidence

    def __post_init__(self) -> None:
        _validate_assignment(self)


@dataclass(frozen=True, slots=True)
class AuthenticatedPrimaryMetric:
    """The authenticated scalar retained in the canonical report."""

    phase: Phase
    method_id: str
    seed: int
    run_id: str
    evidence_sha256: str
    normalized_hypervolume_auc: float


@dataclass(frozen=True, slots=True)
class PairedComparatorStatistics:
    """Exact five-pair development statistics for one comparator."""

    comparator_id: str
    seeds: tuple[int, ...]
    full_values: tuple[float, ...]
    comparator_values: tuple[float, ...]
    paired_effects: tuple[float, ...]
    full_mean: float
    full_median: float
    comparator_mean: float
    comparator_median: float
    paired_effect_mean: float
    paired_effect_median: float
    hodges_lehmann_15_walsh: float
    strict_success_threshold: float
    strict_successes: int
    exact_one_sided_sign_p: float
    exact_two_sided_sign_p: float
    bootstrap_95_type7_lower: float
    bootstrap_95_type7_upper: float


@dataclass(frozen=True, slots=True)
class PairedStatisticalReport:
    """Canonical, self-hashed, non-authorizing paired report."""

    phase: Phase
    cohort_label: str
    authoritative_inventory_sha256: str
    authenticated_metrics: tuple[AuthenticatedPrimaryMetric, ...]
    bootstrap_algorithm: str
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_draw_inventory_sha256: str
    comparisons: tuple[PairedComparatorStatistics, ...]
    confirmatory_claim_allowed: bool
    promotion_authorized: bool
    report_sha256: str

    def __post_init__(self) -> None:
        _validate_report(self)

    def document_bytes(self) -> bytes:
        """Return canonical report bytes after detecting in-memory tampering."""

        _validate_report(self)
        return canonical_json_bytes(_report_document(self, include_digest=True))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatisticalReportingError(message)


def _sha256(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} invalid")
    assert isinstance(value, str)
    return value


def _validate_phase(value: object) -> Phase:
    _require(type(value) is str and value in {"screen", "confirmation"}, "phase invalid")
    return value  # type: ignore[return-value]


def _validate_inventory_row(row: AuthoritativePrimaryEvidenceDigest) -> None:
    _validate_phase(row.phase)
    _require(
        type(row.method_id) is str and _METHOD_RE.fullmatch(row.method_id) is not None,
        "inventory method ID invalid",
    )
    _require(type(row.seed) is int and row.seed >= 0, "inventory seed invalid")
    _require(
        type(row.run_id) is str and _RUN_ID_RE.fullmatch(row.run_id) is not None,
        "inventory run ID invalid",
    )
    _sha256(row.evidence_sha256, label="inventory evidence SHA-256")


def _validate_assignment(assignment: PrimaryEvidenceAssignment) -> None:
    _validate_phase(assignment.phase)
    _require(
        type(assignment.method_id) is str
        and _METHOD_RE.fullmatch(assignment.method_id) is not None,
        "assignment method ID invalid",
    )
    _require(type(assignment.seed) is int and assignment.seed >= 0, "assignment seed invalid")
    _require(
        type(assignment.evidence) is PrimaryHypervolumeEvidence,
        "assignment evidence type differs",
    )
    assignment.evidence.document_bytes()


def _inventory_document(
    rows: tuple[AuthoritativePrimaryEvidenceDigest, ...],
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: (row.phase, row.method_id, row.seed))
    return {
        "schema_version": 1,
        "artifact": INVENTORY_ARTIFACT,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "rows": [
            {
                "phase": row.phase,
                "method_id": row.method_id,
                "seed": row.seed,
                "run_id": row.run_id,
                "evidence_sha256": row.evidence_sha256,
            }
            for row in ordered
        ],
    }


def authoritative_primary_evidence_inventory_sha256(
    rows: tuple[AuthoritativePrimaryEvidenceDigest, ...],
) -> str:
    """Hash an inventory; the returned digest is not authority by itself."""

    _require(type(rows) is tuple and bool(rows), "authoritative inventory must be nonempty tuple")
    for row in rows:
        _require(type(row) is AuthoritativePrimaryEvidenceDigest, "inventory row type differs")
        _validate_inventory_row(row)
    slots = tuple((row.phase, row.method_id, row.seed) for row in rows)
    run_ids = tuple(row.run_id for row in rows)
    digests = tuple(row.evidence_sha256 for row in rows)
    _require(len(set(slots)) == len(slots), "authoritative inventory duplicates a slot")
    _require(len(set(run_ids)) == len(run_ids), "authoritative inventory duplicates a run ID")
    _require(len(set(digests)) == len(digests), "authoritative inventory duplicates evidence")
    return sha256_bytes(canonical_json_bytes(_inventory_document(rows)))


def type7_quantile(values: tuple[float, ...], probability: float) -> float:
    """Return the frozen linear type-7 sample quantile."""

    _require(type(values) is tuple and bool(values), "type-7 values must be nonempty tuple")
    _require(
        type(probability) is float and math.isfinite(probability) and 0.0 <= probability <= 1.0,
        "type-7 probability invalid",
    )
    _require(
        all(type(value) is float and math.isfinite(value) for value in values),
        "type-7 values invalid",
    )
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


def _bootstrap_draws(*, seed: int, samples: int, blocks: int) -> tuple[tuple[int, ...], ...]:
    limit = 1 << 64
    rejection_limit = limit - limit % blocks
    result: list[tuple[int, ...]] = []
    for replicate in range(samples):
        draw: list[int] = []
        for position in range(blocks):
            nonce = 0
            while True:
                payload = (
                    BOOTSTRAP_INDEX_DOMAIN
                    + seed.to_bytes(8, "big")
                    + replicate.to_bytes(8, "big")
                    + position.to_bytes(8, "big")
                    + nonce.to_bytes(8, "big")
                )
                candidate = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
                if candidate < rejection_limit:
                    draw.append(candidate % blocks)
                    break
                nonce += 1
        result.append(tuple(draw))
    return tuple(result)


def _bootstrap_draw_digest(draws: tuple[tuple[int, ...], ...], *, seed: int, blocks: int) -> str:
    header = (
        BOOTSTRAP_DRAW_DIGEST_DOMAIN
        + seed.to_bytes(8, "big")
        + len(draws).to_bytes(8, "big")
        + blocks.to_bytes(8, "big")
    )
    return sha256_bytes(header + bytes(index for draw in draws for index in draw))


def _sign_p_values(*, successes: int, pairs: int) -> tuple[float, float]:
    upper = sum(math.comb(pairs, count) for count in range(successes, pairs + 1)) / 2**pairs
    lower = sum(math.comb(pairs, count) for count in range(successes + 1)) / 2**pairs
    return upper, min(1.0, 2.0 * min(lower, upper))


def _comparison(
    *,
    phase: Phase,
    comparator_id: str,
    seeds: tuple[int, ...],
    values: dict[tuple[str, int], float],
    draws: tuple[tuple[int, ...], ...],
) -> PairedComparatorStatistics:
    full = tuple(values[(FULL_METHOD_ID, seed)] for seed in seeds)
    comparator = tuple(values[(comparator_id, seed)] for seed in seeds)
    effects = tuple(left - right for left, right in zip(full, comparator, strict=True))
    threshold = 1e-12 if phase == "screen" else 0.05 + 1e-12
    successes = sum(effect > threshold for effect in effects)
    one_sided, two_sided = _sign_p_values(successes=successes, pairs=len(seeds))
    bootstrap = tuple(math.fsum(effects[index] for index in draw) / len(draw) for draw in draws)
    walsh = tuple(
        (effects[left] + effects[right]) / 2.0
        for left in range(len(effects))
        for right in range(left, len(effects))
    )
    return PairedComparatorStatistics(
        comparator_id=comparator_id,
        seeds=seeds,
        full_values=full,
        comparator_values=comparator,
        paired_effects=effects,
        full_mean=math.fsum(full) / len(full),
        full_median=_median(full),
        comparator_mean=math.fsum(comparator) / len(comparator),
        comparator_median=_median(comparator),
        paired_effect_mean=math.fsum(effects) / len(effects),
        paired_effect_median=_median(effects),
        hodges_lehmann_15_walsh=_median(walsh),
        strict_success_threshold=threshold,
        strict_successes=successes,
        exact_one_sided_sign_p=one_sided,
        exact_two_sided_sign_p=two_sided,
        bootstrap_95_type7_lower=type7_quantile(bootstrap, 0.025),
        bootstrap_95_type7_upper=type7_quantile(bootstrap, 0.975),
    )


def reduce_authenticated_paired_statistics(
    protocol: EvolutionaryKLProtocol,
    *,
    phase: Phase,
    evidence: tuple[PrimaryEvidenceAssignment, ...],
    authoritative_inventory: tuple[AuthoritativePrimaryEvidenceDigest, ...],
    expected_authoritative_inventory_sha256: str,
) -> PairedStatisticalReport:
    """Authenticate and reduce one exact 15-run paired development cohort."""

    _validate_reducer_protocol(protocol)
    selected_phase = _validate_phase(phase)
    expected_digest = _sha256(
        expected_authoritative_inventory_sha256,
        label="externally expected authoritative inventory SHA-256",
    )
    observed_digest = authoritative_primary_evidence_inventory_sha256(authoritative_inventory)
    _require(observed_digest == expected_digest, "authoritative inventory digest differs")
    _require(type(evidence) is tuple, "evidence assignments must be tuple")

    seeds = SCREEN_SEEDS if selected_phase == "screen" else CONFIRMATION_SEEDS
    methods = (FULL_METHOD_ID, *PRIMARY_COMPARATOR_IDS)
    expected_slots = {(selected_phase, method, seed) for method in methods for seed in seeds}
    inventory_by_slot = {
        (row.phase, row.method_id, row.seed): row for row in authoritative_inventory
    }
    _require(set(inventory_by_slot) == expected_slots, "authoritative inventory cohort differs")

    assignments: dict[tuple[Phase, str, int], PrimaryEvidenceAssignment] = {}
    object_ids: set[int] = set()
    for assignment in evidence:
        _require(type(assignment) is PrimaryEvidenceAssignment, "evidence assignment type differs")
        _validate_assignment(assignment)
        slot = (assignment.phase, assignment.method_id, assignment.seed)
        _require(slot not in assignments, "evidence assignments duplicate a slot")
        _require(id(assignment.evidence) not in object_ids, "duplicate evidence object used")
        assignments[slot] = assignment
        object_ids.add(id(assignment.evidence))
    _require(set(assignments) == expected_slots, "evidence assignment cohort differs")

    metrics: list[AuthenticatedPrimaryMetric] = []
    used_digests: set[str] = set()
    used_run_ids: set[str] = set()
    for slot in sorted(expected_slots):
        assignment = assignments[slot]
        item = assignment.evidence
        item.document_bytes()
        authority = inventory_by_slot[slot]
        _require(item.run_id == authority.run_id, "evidence run ID differs from authority")
        _require(
            item.evidence_sha256 == authority.evidence_sha256,
            "evidence self-hash differs from authority",
        )
        _require(item.evidence_sha256 not in used_digests, "duplicate evidence digest used")
        _require(item.run_id not in used_run_ids, "duplicate evidence run ID used")
        used_digests.add(item.evidence_sha256)
        used_run_ids.add(item.run_id)
        metrics.append(
            AuthenticatedPrimaryMetric(
                phase=selected_phase,
                method_id=assignment.method_id,
                seed=assignment.seed,
                run_id=item.run_id,
                evidence_sha256=item.evidence_sha256,
                normalized_hypervolume_auc=item.normalized_hypervolume_auc,
            )
        )

    draws = _bootstrap_draws(seed=protocol.paired_bootstrap_seed, samples=10000, blocks=5)
    value_map = {
        (metric.method_id, metric.seed): metric.normalized_hypervolume_auc for metric in metrics
    }
    comparisons = tuple(
        _comparison(
            phase=selected_phase,
            comparator_id=comparator_id,
            seeds=seeds,
            values=value_map,
            draws=draws,
        )
        for comparator_id in PRIMARY_COMPARATOR_IDS
    )
    cohort_label = (
        "descriptive_development_screen"
        if selected_phase == "screen"
        else "published_unsequestered_development_confirmation_not_confirmatory"
    )
    unsigned = PairedStatisticalReport.__new__(PairedStatisticalReport)
    for name, value in (
        ("phase", selected_phase),
        ("cohort_label", cohort_label),
        ("authoritative_inventory_sha256", observed_digest),
        ("authenticated_metrics", tuple(metrics)),
        ("bootstrap_algorithm", BOOTSTRAP_ALGORITHM),
        ("bootstrap_samples", 10000),
        ("bootstrap_seed", protocol.paired_bootstrap_seed),
        (
            "bootstrap_draw_inventory_sha256",
            _bootstrap_draw_digest(draws, seed=protocol.paired_bootstrap_seed, blocks=5),
        ),
        ("comparisons", comparisons),
        ("confirmatory_claim_allowed", False),
        ("promotion_authorized", False),
    ):
        object.__setattr__(unsigned, name, value)
    digest = sha256_bytes(canonical_json_bytes(_report_document(unsigned, include_digest=False)))
    object.__setattr__(unsigned, "report_sha256", digest)
    _validate_report(unsigned)
    return unsigned


def _validate_reducer_protocol(protocol: EvolutionaryKLProtocol) -> None:
    _require(type(protocol) is EvolutionaryKLProtocol, "protocol type differs")
    reporting = protocol.statistical_reporting
    _require(
        protocol.artifact == "evolutionary_kl_research_protocol_v1"
        and protocol.status == "predeclared_blocked_on_prerequisites"
        and not protocol.automatic_production_eligible
        and not protocol.execution_authorized
        and not protocol.biological_superiority_claim_allowed
        and protocol.screen_seeds == SCREEN_SEEDS
        and protocol.confirmation_seeds == CONFIRMATION_SEEDS
        and protocol.confirmation_method_ids == CONFIRMATION_METHOD_IDS
        and protocol.primary_comparator_ids == PRIMARY_COMPARATOR_IDS
        and protocol.confirmation_seed_status
        == "published_unsequestered_development_only_not_confirmatory"
        and protocol.paired_bootstrap_samples == 10000
        and protocol.paired_bootstrap_seed == 20260908
        and protocol.de_novo_paired_bootstrap_unit
        == "seed_block_resample_with_same_seed_index_shared_across_all_compared_methods"
        and protocol.screen_effect == "full_normalized_hv_auc_minus_comparator_normalized_hv_auc"
        and protocol.screen_comparison_absolute_tolerance == 1e-12
        and protocol.screen_pair_success
        == "finite_effect_strictly_greater_than_zero_plus_tolerance"
        and protocol.confirmation_effect
        == "full_normalized_hv_auc_minus_comparator_normalized_hv_auc"
        and protocol.confirmation_additive_materiality_margin == 0.05
        and protocol.confirmation_comparison_absolute_tolerance == 1e-12
        and protocol.confirmation_pair_success
        == "finite_effect_strictly_greater_than_additive_margin_plus_tolerance"
        and protocol.confirmation_missing_nonfinite_or_tie
        == "missing_nonfinite_or_effect_at_or_below_margin_plus_tolerance_is_pair_failure"
        and protocol.confirmation_one_sided_alpha == 0.05
        and protocol.confirmation_exact_sign_test_minimum_p == 0.03125
        and protocol.confirmation_pairs_required_above_margin == 5
        and protocol.confirmation_comparator_gate
        == "all_five_fixed_seed_pairs_are_successes_and_one_sided_exact_sign_p_at_most_alpha"
        and protocol.confirmation_global_iut_gate == "both_primary_comparator_gates_must_pass"
        and reporting.paired_bootstrap_statistic
        == "arithmetic_mean_of_five_paired_full_minus_comparator_seed_effects"
        and reporting.paired_bootstrap_interval_level == 0.95
        and reporting.paired_bootstrap_interval_type == "percentile_two_sided"
        and reporting.paired_bootstrap_interval_quantiles == (0.025, 0.975)
        and reporting.paired_bootstrap_quantile_convention
        == (
            "sorted_B_replicates_linear_type7_h_equals_B_minus_1_times_p_"
            "interpolate_between_floor_and_ceil"
        )
        and reporting.hodges_lehmann_estimand
        == (
            "median_of_all_15_walsh_averages_d_i_plus_d_j_over_2_for_1_less_"
            "than_or_equal_to_i_less_than_or_equal_to_j_less_than_or_equal_to_5"
        )
        and reporting.median_convention
        == "arithmetic_mean_of_two_central_sorted_values_when_even_otherwise_central_sorted_value",
        "protocol paired-reporting contract differs",
    )


def _float_hex(value: float) -> str:
    return value.hex()


def _comparison_document(item: PairedComparatorStatistics) -> dict[str, object]:
    return {
        "comparator_id": item.comparator_id,
        "seeds": list(item.seeds),
        "full_values_hex": [_float_hex(value) for value in item.full_values],
        "comparator_values_hex": [_float_hex(value) for value in item.comparator_values],
        "paired_effects_hex": [_float_hex(value) for value in item.paired_effects],
        "full_mean_hex": _float_hex(item.full_mean),
        "full_median_hex": _float_hex(item.full_median),
        "comparator_mean_hex": _float_hex(item.comparator_mean),
        "comparator_median_hex": _float_hex(item.comparator_median),
        "paired_effect_mean_hex": _float_hex(item.paired_effect_mean),
        "paired_effect_median_hex": _float_hex(item.paired_effect_median),
        "hodges_lehmann_15_walsh_hex": _float_hex(item.hodges_lehmann_15_walsh),
        "strict_success_threshold_hex": _float_hex(item.strict_success_threshold),
        "strict_successes": item.strict_successes,
        "exact_one_sided_sign_p_hex": _float_hex(item.exact_one_sided_sign_p),
        "exact_two_sided_sign_p_hex": _float_hex(item.exact_two_sided_sign_p),
        "bootstrap_95_type7_lower_hex": _float_hex(item.bootstrap_95_type7_lower),
        "bootstrap_95_type7_upper_hex": _float_hex(item.bootstrap_95_type7_upper),
    }


def _report_document(report: PairedStatisticalReport, *, include_digest: bool) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": REPORT_ARTIFACT,
        "status": "development_only_no_confirmatory_or_promotion_authority",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "phase": report.phase,
        "cohort_label": report.cohort_label,
        "authoritative_inventory_sha256": report.authoritative_inventory_sha256,
        "authenticated_metrics": [
            {
                "phase": item.phase,
                "method_id": item.method_id,
                "seed": item.seed,
                "run_id": item.run_id,
                "evidence_sha256": item.evidence_sha256,
                "normalized_hypervolume_auc_hex": _float_hex(item.normalized_hypervolume_auc),
            }
            for item in report.authenticated_metrics
        ],
        "bootstrap": {
            "algorithm": report.bootstrap_algorithm,
            "samples": report.bootstrap_samples,
            "seed": report.bootstrap_seed,
            "draw_inventory_sha256": report.bootstrap_draw_inventory_sha256,
            "interval": "two_sided_95_percentile_type7",
            "paired_unit": "five_seed_blocks_shared_across_methods",
        },
        "comparisons": [_comparison_document(item) for item in report.comparisons],
        "confirmatory_claim_allowed": report.confirmatory_claim_allowed,
        "promotion_authorized": report.promotion_authorized,
    }
    if include_digest:
        document["report_sha256"] = report.report_sha256
    return document


def _validate_report(report: PairedStatisticalReport) -> None:
    phase = _validate_phase(report.phase)
    seeds = SCREEN_SEEDS if phase == "screen" else CONFIRMATION_SEEDS
    methods = (FULL_METHOD_ID, *PRIMARY_COMPARATOR_IDS)
    expected_slots = tuple(
        sorted((phase, method_id, seed) for method_id in methods for seed in seeds)
    )
    expected_label = (
        "descriptive_development_screen"
        if phase == "screen"
        else "published_unsequestered_development_confirmation_not_confirmatory"
    )
    _require(report.cohort_label == expected_label, "report cohort label differs")
    _sha256(report.authoritative_inventory_sha256, label="report inventory SHA-256")
    _sha256(report.bootstrap_draw_inventory_sha256, label="bootstrap draw inventory SHA-256")
    _sha256(report.report_sha256, label="report self-hash")
    _require(
        report.bootstrap_algorithm == BOOTSTRAP_ALGORITHM
        and report.bootstrap_samples == 10000
        and report.bootstrap_seed == 20260908,
        "report bootstrap contract differs",
    )
    _require(
        report.confirmatory_claim_allowed is False and report.promotion_authorized is False,
        "blocked v1 report cannot authorize claims or promotion",
    )
    _require(
        type(report.authenticated_metrics) is tuple and len(report.authenticated_metrics) == 15,
        "report must contain exactly 15 authenticated metrics",
    )
    slots: list[tuple[Phase, str, int]] = []
    run_ids: list[str] = []
    evidence_digests: list[str] = []
    inventory_rows: list[AuthoritativePrimaryEvidenceDigest] = []
    value_map: dict[tuple[str, int], float] = {}
    for item in report.authenticated_metrics:
        _require(type(item) is AuthenticatedPrimaryMetric, "report metric type differs")
        _require(item.phase == phase, "report mixes phases")
        _require(item.method_id in methods, "report metric method differs")
        _require(type(item.seed) is int and item.seed in seeds, "report metric seed differs")
        _require(
            type(item.run_id) is str and _RUN_ID_RE.fullmatch(item.run_id) is not None,
            "report metric run ID invalid",
        )
        _sha256(item.evidence_sha256, label="report evidence SHA-256")
        _require(
            type(item.normalized_hypervolume_auc) is float
            and math.isfinite(item.normalized_hypervolume_auc)
            and 0.0 <= item.normalized_hypervolume_auc <= 1.0,
            "report metric invalid",
        )
        slots.append((item.phase, item.method_id, item.seed))
        run_ids.append(item.run_id)
        evidence_digests.append(item.evidence_sha256)
        value_map[(item.method_id, item.seed)] = item.normalized_hypervolume_auc
        inventory_rows.append(
            AuthoritativePrimaryEvidenceDigest(
                phase=item.phase,
                method_id=item.method_id,
                seed=item.seed,
                run_id=item.run_id,
                evidence_sha256=item.evidence_sha256,
            )
        )

    _require(tuple(slots) == expected_slots, "report metric cohort or canonical order differs")
    _require(len(set(run_ids)) == len(run_ids), "report duplicates a run ID")
    _require(len(set(evidence_digests)) == len(evidence_digests), "report duplicates evidence")
    observed_inventory = authoritative_primary_evidence_inventory_sha256(tuple(inventory_rows))
    _require(
        observed_inventory == report.authoritative_inventory_sha256,
        "report inventory binding differs",
    )

    draws = _bootstrap_draws(seed=report.bootstrap_seed, samples=report.bootstrap_samples, blocks=5)
    expected_draw_digest = _bootstrap_draw_digest(draws, seed=report.bootstrap_seed, blocks=5)
    _require(
        report.bootstrap_draw_inventory_sha256 == expected_draw_digest,
        "report bootstrap draw inventory differs",
    )
    _require(
        type(report.comparisons) is tuple
        and all(type(item) is PairedComparatorStatistics for item in report.comparisons),
        "report comparison type differs",
    )
    expected_comparisons = tuple(
        _comparison(
            phase=phase,
            comparator_id=comparator_id,
            seeds=seeds,
            values=value_map,
            draws=draws,
        )
        for comparator_id in PRIMARY_COMPARATOR_IDS
    )
    _require(
        report.comparisons == expected_comparisons,
        "report comparison statistics differ from authenticated metrics",
    )
    observed = sha256_bytes(canonical_json_bytes(_report_document(report, include_digest=False)))
    _require(observed == report.report_sha256, "report self-hash differs")

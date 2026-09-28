"""Pure deterministic metric reduction for sequential mixed-acquisition v2.

This module starts *after* the finalization worker has authenticated every
global and leaf capability.  It deliberately has no filesystem, process,
wire, or :class:`PhaseSeal` API.  The boundary layer supplies the immutable
values below; this module verifies their semantic alignment, computes the
frozen reductions, and returns the seven canonical payloads that may later be
published by the sealed finalization worker.

The independent verifier must not import this producer implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from amp_challenge.acquisition.sequential_v2_selector import (
    GuardEvaluation,
    PoolCandidate,
)
from amp_challenge.evaluation.sequential_v2_primitives import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    TARGETS,
    ContextRow,
    aggregate_deterministic_outer_units,
    aggregate_observed_sequence_outcomes,
    aggregate_random_outer_units,
    bootstrap_indices_sha256,
    mean_pairwise_feature_cosine_distance,
    observed_batch_reward,
    paired_outer_bootstrap,
    summarize_target_metrics,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    CEILING,
    EXPECTED_CONTEXTS_BY_FOLD,
    EXPECTED_POLICY_RUNS,
    EXPECTED_SUPPORT_BY_FOLD,
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
    NO_QUERY,
    POLICY_ORDER,
    RANDOM,
    PolicyRunSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_run_by_track_id,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from amp_challenge.evaluation.sequential_v2_select import PoolCommitment
from amp_challenge.evaluation.sequential_v2_update import OuterContextPrediction

SCHEMA_VERSION = 1
FINALIZE_ARTIFACT = "sequential_v2_finalize_campaign_v1"
PROMOTION_DECISION_ARTIFACT = "sequential_v2_promotion_decision_v1"
FINALIZE_SUMMARY_ARTIFACT = "sequential_v2_finalize_summary_v1"

FINALIZE_PAYLOAD_PATHS = (
    "outer-fold-units.jsonl",
    "paired-comparisons.jsonl",
    "policy-point-estimates.jsonl",
    "policy-rotation-metrics.jsonl",
    "promotion-decision.json",
    "rotation-metrics.jsonl",
    "summary.json",
)

GUARDED_POLICIES = (
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
)
GUARD_STATUSES = ("not_applicable", "passed", "exact_mean_fallback")
EXPECTED_GUARDED_TRACKS = 60

AGGREGATE_METRIC_NAMES = (
    "macro_brier",
    "macro_negative_log_likelihood",
    "macro_roc_auc",
    "macro_average_precision",
    "macro_ece_10",
    "next_round_outer_top10_mean_reward",
    "queried_outcome_mean_reward",
    "queried_unique_diversity_components",
    "queried_mean_pairwise_feature_cosine_distance",
    "revealed_context_count",
)

COMPARISON_ORDER = (
    "mixed_minus_mean_macro_brier",
    "mixed_minus_mean_macro_negative_log_likelihood",
    "mixed_minus_mean_next_round_outer_top10_mean_reward",
    "mixed_minus_mean_queried_mean_pairwise_feature_cosine_distance",
    "mixed_minus_mean_queried_unique_diversity_components",
    "mixed_minus_no_query_macro_brier",
    "mixed_minus_random_macro_brier",
)

PROMOTION_ITEM_ORDER = (
    "brier_vs_mean",
    "nll_vs_mean",
    "outer_reward_vs_mean",
    "queried_diversity_vs_mean",
    "brier_vs_no_query_and_random",
    "guarded_selection_validity",
)

_COMPARISON_SPECS = (
    (
        COMPARISON_ORDER[0],
        MIXED,
        MEAN,
        "macro_brier",
        "lower",
    ),
    (
        COMPARISON_ORDER[1],
        MIXED,
        MEAN,
        "macro_negative_log_likelihood",
        "lower",
    ),
    (
        COMPARISON_ORDER[2],
        MIXED,
        MEAN,
        "next_round_outer_top10_mean_reward",
        "higher",
    ),
    (
        COMPARISON_ORDER[3],
        MIXED,
        MEAN,
        "queried_mean_pairwise_feature_cosine_distance",
        "higher",
    ),
    (
        COMPARISON_ORDER[4],
        MIXED,
        MEAN,
        "queried_unique_diversity_components",
        "higher",
    ),
    (
        COMPARISON_ORDER[5],
        MIXED,
        NO_QUERY,
        "macro_brier",
        "lower",
    ),
    (
        COMPARISON_ORDER[6],
        MIXED,
        RANDOM,
        "macro_brier",
        "lower",
    ),
)


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _hex(value: object, *, label: str) -> str:
    return _finite_float(value, label=label).hex()


def _canonical_hex_float(value: object, *, label: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be canonical binary64 hexadecimal text")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be canonical binary64 hexadecimal text") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{label} must be canonical finite binary64 hexadecimal text")
    return parsed


def _optional_finite(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    canonical = policy_run_by_track_id(value.track_id)
    if value != canonical:
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _canonical_sha256_ids(
    values: object,
    *,
    label: str,
    expected_count: int | None = None,
    require_sorted: bool = False,
) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(item) is not str for item in values):
        raise TypeError(f"{label} must be an exact immutable text tuple")
    result = tuple(values)
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    if require_sorted and result != tuple(sorted(result)):
        raise ValueError(f"{label} must use ascending order")
    if any(
        len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        for item in result
    ):
        raise ValueError(f"{label} must contain lowercase SHA-256 identifiers")
    return result


def _ordered_text_stream_sha256(values: Sequence[str], *, label: str) -> str:
    identifiers = tuple(values)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} must be nonempty and unique")
    if any(
        type(item) is not str or not item or "\n" in item or "\r" in item for item in identifiers
    ):
        raise ValueError(f"{label} contains an invalid identifier")
    return hashlib.sha256(
        "".join(f"{identifier}\n" for identifier in identifiers).encode("ascii")
    ).hexdigest()


def _mean(values: Sequence[float], *, label: str) -> float:
    parsed = np.asarray(
        [_finite_float(value, label=f"{label} value") for value in values],
        dtype=np.float64,
    )
    if parsed.ndim != 1 or not len(parsed):
        raise ValueError(f"{label} requires at least one value")
    return float(np.mean(parsed, dtype=np.float64))


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be canonical LF-terminated JSON")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} must be a canonical JSON object")
    return value


def _target_metric_document(summary: Mapping[str, object]) -> dict[str, object]:
    """Encode the trusted primitive result without persisting binary64 decimals."""

    expected = {
        "by_target",
        "macro",
        "defined_targets",
        "defined_target_counts",
        "target_order",
        "nll_probability_clip_hex",
        "ece_bins",
    }
    if type(summary) is not dict or set(summary) != expected:
        raise ValueError("target metric primitive returned an unexpected schema")
    if summary["target_order"] != list(TARGETS):
        raise ValueError("target metric primitive returned a different target order")
    by_target = summary["by_target"]
    macro = summary["macro"]
    defined = summary["defined_targets"]
    defined_counts = summary["defined_target_counts"]
    if not all(type(value) is dict for value in (by_target, macro, defined, defined_counts)):
        raise ValueError("target metric primitive returned malformed metric objects")
    metric_names = ("brier", "nll", "roc_auc", "average_precision", "ece_10")
    output_by_target: dict[str, object] = {}
    for target in TARGETS:
        raw = by_target[target]
        if type(raw) is not dict or set(raw) != {
            "n",
            "positives",
            "negatives",
            *metric_names,
            "undefined_reason",
        }:
            raise ValueError(f"target metric row {target!r} has an unexpected schema")
        output_by_target[target] = {
            "n": raw["n"],
            "positives": raw["positives"],
            "negatives": raw["negatives"],
            **{
                name: (
                    None
                    if raw[name] is None
                    else _hex(raw[name], label=f"target metric {target} {name}")
                )
                for name in metric_names
            },
            "undefined_reason": raw["undefined_reason"],
        }
    return {
        "target_order": list(TARGETS),
        "by_target": output_by_target,
        "macro": {
            name: _hex(macro[name], label=f"macro target metric {name}") for name in metric_names
        },
        "defined_targets": {name: list(defined[name]) for name in metric_names},
        "defined_target_counts": {name: defined_counts[name] for name in metric_names},
        "nll_probability_clip_hex": summary["nll_probability_clip_hex"],
        "ece_bins": summary["ece_bins"],
    }


def _validate_target_metric_result_document(
    document: dict[str, Any],
) -> dict[str, float]:
    """Reject any field that could republish row identity or raw observations."""

    top_level = {
        "target_order",
        "by_target",
        "macro",
        "defined_targets",
        "defined_target_counts",
        "nll_probability_clip_hex",
        "ece_bins",
    }
    if set(document) != top_level or document["target_order"] != list(TARGETS):
        raise ValueError("target metric result differs from the exact top-level schema")
    if document["nll_probability_clip_hex"] != (1e-15).hex() or document["ece_bins"] != (
        "[0,.1),[.1,.2),...,[.8,.9),[.9,1]"
    ):
        raise ValueError("target metric result numerical convention changed")
    by_target = document["by_target"]
    macro = document["macro"]
    defined = document["defined_targets"]
    counts = document["defined_target_counts"]
    metric_names = ("brier", "nll", "roc_auc", "average_precision", "ece_10")
    if (
        type(by_target) is not dict
        or set(by_target) != set(TARGETS)
        or type(macro) is not dict
        or set(macro) != set(metric_names)
        or type(defined) is not dict
        or set(defined) != set(metric_names)
        or type(counts) is not dict
        or set(counts) != set(metric_names)
    ):
        raise ValueError("target metric result has an invalid nested metric inventory")

    numeric_by_target: dict[str, dict[str, float | None]] = {}
    for target in TARGETS:
        row = by_target[target]
        if type(row) is not dict or set(row) != {
            "n",
            "positives",
            "negatives",
            *metric_names,
            "undefined_reason",
        }:
            raise ValueError(f"target metric result row {target!r} has an unexpected schema")
        n = row["n"]
        positives = row["positives"]
        negatives = row["negatives"]
        if (
            type(n) is not int
            or type(positives) is not int
            or type(negatives) is not int
            or min(n, positives, negatives) < 0
            or positives + negatives != n
        ):
            raise ValueError(f"target metric result row {target!r} has an invalid census")
        numeric: dict[str, float | None] = {}
        for name in metric_names:
            value = row[name]
            numeric[name] = (
                None
                if value is None
                else _canonical_hex_float(
                    value,
                    label=f"target metric result row {target} {name}",
                )
            )
        if n == 0:
            if (
                any(value is not None for value in numeric.values())
                or row["undefined_reason"] != "no_outer_contexts"
            ):
                raise ValueError(f"empty target metric row {target!r} has invented values")
        else:
            if any(numeric[name] is None for name in ("brier", "nll", "ece_10")):
                raise ValueError(f"populated target metric row {target!r} lacks core metrics")
            if (numeric["roc_auc"] is not None) != (positives > 0 and negatives > 0):
                raise ValueError(f"target metric row {target!r} has invalid ROC-AUC support")
            if (numeric["average_precision"] is not None) != (positives > 0):
                raise ValueError(
                    f"target metric row {target!r} has invalid average-precision support"
                )
            expected_reason: object
            if numeric["roc_auc"] is not None and numeric["average_precision"] is not None:
                expected_reason = None
            else:
                expected_reason = {
                    "roc_auc": (
                        None if numeric["roc_auc"] is not None else "requires_both_binary_classes"
                    ),
                    "average_precision": (
                        None
                        if numeric["average_precision"] is not None
                        else "requires_at_least_one_positive"
                    ),
                }
            if row["undefined_reason"] != expected_reason:
                raise ValueError(f"target metric row {target!r} has a false undefined reason")
            if (
                not 0.0 <= float(numeric["brier"]) <= 1.0
                or float(numeric["nll"]) < 0.0
                or not 0.0 <= float(numeric["ece_10"]) <= 1.0
                or (numeric["roc_auc"] is not None and not 0.0 <= float(numeric["roc_auc"]) <= 1.0)
                or (
                    numeric["average_precision"] is not None
                    and not 0.0 <= float(numeric["average_precision"]) <= 1.0
                )
            ):
                raise ValueError(f"target metric row {target!r} lies outside metric bounds")
        numeric_by_target[target] = numeric

    macro_values: dict[str, float] = {}
    for name in metric_names:
        expected_targets = tuple(
            target for target in TARGETS if numeric_by_target[target][name] is not None
        )
        target_list = defined[name]
        count = counts[name]
        if (
            type(target_list) is not list
            or any(type(item) is not str for item in target_list)
            or tuple(target_list) != expected_targets
            or type(count) is not int
            or count != len(expected_targets)
            or count < 1
        ):
            raise ValueError(f"target metric result defined-target census changed for {name}")
        observed_macro = _canonical_hex_float(
            macro[name],
            label=f"target metric result macro {name}",
        )
        expected_macro = _mean(
            tuple(float(numeric_by_target[target][name]) for target in expected_targets),
            label=f"target metric result macro {name}",
        )
        if observed_macro != expected_macro:
            raise ValueError(f"target metric result macro {name} differs from target rows")
        macro_values[name] = observed_macro
    return macro_values


@dataclass(frozen=True, slots=True)
class TargetMetricResult:
    """Immutable target summary plus the five numeric macros used downstream."""

    canonical_document: bytes
    macro_brier: float
    macro_negative_log_likelihood: float
    macro_roc_auc: float
    macro_average_precision: float
    macro_ece_10: float

    def __post_init__(self) -> None:
        document = _strict_json_object(
            self.canonical_document,
            label="target metric result",
        )
        macro = _validate_target_metric_result_document(document)
        pairs = (
            ("brier", self.macro_brier),
            ("nll", self.macro_negative_log_likelihood),
            ("roc_auc", self.macro_roc_auc),
            ("average_precision", self.macro_average_precision),
            ("ece_10", self.macro_ece_10),
        )
        for name, value in pairs:
            parsed = _finite_float(value, label=f"target metric result {name}")
            if macro[name] != parsed:
                raise ValueError(f"target metric result {name} differs from its document")

    @classmethod
    def from_columns(
        cls,
        targets: Sequence[str],
        labels: Sequence[int],
        probabilities: Sequence[float],
    ) -> TargetMetricResult:
        raw = summarize_target_metrics(targets, labels, probabilities)
        document = _target_metric_document(raw)
        macro = raw["macro"]
        assert isinstance(macro, dict)
        return cls(
            canonical_document=canonical_json_bytes(document),
            macro_brier=float(macro["brier"]),
            macro_negative_log_likelihood=float(macro["nll"]),
            macro_roc_auc=float(macro["roc_auc"]),
            macro_average_precision=float(macro["average_precision"]),
            macro_ece_10=float(macro["ece_10"]),
        )

    def document(self) -> dict[str, object]:
        return _strict_json_object(self.canonical_document, label="target metric result")


@dataclass(frozen=True, slots=True)
class AggregateMetricValues:
    """The ten metrics propagated through seed, rotation, and fold means."""

    macro_brier: float
    macro_negative_log_likelihood: float
    macro_roc_auc: float
    macro_average_precision: float
    macro_ece_10: float
    next_round_outer_top10_mean_reward: float
    queried_outcome_mean_reward: float | None
    queried_unique_diversity_components: float
    queried_mean_pairwise_feature_cosine_distance: float | None
    revealed_context_count: float

    def __post_init__(self) -> None:
        for name in AGGREGATE_METRIC_NAMES:
            value = getattr(self, name)
            if name in {
                "queried_outcome_mean_reward",
                "queried_mean_pairwise_feature_cosine_distance",
            }:
                _optional_finite(value, label=f"aggregate metric {name}")
            else:
                _finite_float(value, label=f"aggregate metric {name}")
        if self.queried_unique_diversity_components < 0.0:
            raise ValueError("queried component mean cannot be negative")
        if self.revealed_context_count < 0.0:
            raise ValueError("revealed context mean cannot be negative")

    def value(self, name: str) -> float | None:
        if name not in AGGREGATE_METRIC_NAMES:
            raise ValueError(f"unknown aggregate metric {name!r}")
        return getattr(self, name)

    def document(self) -> dict[str, object]:
        return {
            f"{name}_hex": (
                None
                if self.value(name) is None
                else _hex(self.value(name), label=f"aggregate metric {name}")
            )
            for name in AGGREGATE_METRIC_NAMES
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | None]) -> AggregateMetricValues:
        if type(values) is not dict or set(values) != set(AGGREGATE_METRIC_NAMES):
            raise ValueError("aggregate metric map differs from the frozen metric set")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PoolSelectionEvidence:
    """Outcome-free commitment fields required by metric reduction.

    ``from_commitment`` is the intended bridge from an authenticated reveal
    leaf.  Direct construction remains a value convenience, never an
    authentication authority.
    """

    run: PolicyRunSpec
    selected_sequence_ids: tuple[str, ...]
    mean_control_sequence_ids: tuple[str, ...]
    guard_evaluations: tuple[GuardEvaluation, ...]
    fallback_reason: str | None

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="pool selection evidence run")
        selected = _canonical_sha256_ids(
            self.selected_sequence_ids,
            label="pool selected sequence IDs",
            expected_count=run.expected_pool_selection_count,
        )
        mean_control = _canonical_sha256_ids(
            self.mean_control_sequence_ids,
            label="pool mean-control sequence IDs",
        )
        if type(self.guard_evaluations) is not tuple or any(
            type(item) is not GuardEvaluation for item in self.guard_evaluations
        ):
            raise TypeError("pool guard evaluations must be an exact immutable tuple")
        if self.fallback_reason is not None and (
            type(self.fallback_reason) is not str or not self.fallback_reason
        ):
            raise ValueError("pool fallback reason must be nonempty text or null")
        for index, evaluation in enumerate(self.guard_evaluations):
            if type(evaluation.stage) is not str or not evaluation.stage:
                raise ValueError(f"pool guard evaluation {index} has an invalid stage")
            _canonical_sha256_ids(
                evaluation.sequence_ids,
                label=f"pool guard evaluation {index} sequence IDs",
                expected_count=10,
            )
            if (
                type(evaluation.objective_means) is not tuple
                or len(evaluation.objective_means) != 3
                or type(evaluation.objective_losses) is not tuple
                or len(evaluation.objective_losses) != 3
                or any(
                    type(item) is not float or not math.isfinite(item)
                    for item in (*evaluation.objective_means, *evaluation.objective_losses)
                )
                or type(evaluation.scalar_mean) is not float
                or not math.isfinite(evaluation.scalar_mean)
                or type(evaluation.scalar_loss) is not float
                or not math.isfinite(evaluation.scalar_loss)
                or type(evaluation.passed) is not bool
            ):
                raise ValueError(f"pool guard evaluation {index} has invalid numeric evidence")

        if run.policy in GUARDED_POLICIES:
            if len(mean_control) != 10:
                raise ValueError("guarded policy lacks its exact ten-sequence mean control")
            if not self.guard_evaluations:
                raise ValueError("guarded policy lacks a final guard evaluation")
            final = self.guard_evaluations[-1]
            if tuple(final.sequence_ids) != selected:
                raise ValueError("final guard evaluation differs from committed selected IDs")
            if final.passed is not True:
                raise ValueError("guarded selection has no passing final evaluation")
            if self.fallback_reason is None:
                if final.stage == "fallback":
                    raise ValueError("passing nonfallback selection cannot end at fallback")
            elif final.stage != "fallback" or selected != mean_control or not mean_control:
                raise ValueError("exact mean fallback does not match its control selection")
        elif self.guard_evaluations or self.fallback_reason is not None:
            raise ValueError("unguarded policy cannot carry guard or fallback evidence")

        if run.policy == MEAN and mean_control != selected:
            raise ValueError("mean policy selection differs from its mean control")
        if run.policy in {RANDOM, NO_QUERY, CEILING} and mean_control:
            raise ValueError("random/no-query/ceiling policy cannot invent a mean control")

    @classmethod
    def from_commitment(cls, commitment: PoolCommitment) -> PoolSelectionEvidence:
        if type(commitment) is not PoolCommitment:
            raise TypeError("pool selection evidence requires an exact PoolCommitment")
        result = commitment.selection_result
        return cls(
            run=commitment.run,
            selected_sequence_ids=commitment.selected_sequence_ids,
            mean_control_sequence_ids=(
                () if result is None else tuple(result.mean_control_sequence_ids)
            ),
            guard_evaluations=(() if result is None else tuple(result.guard_evaluations)),
            fallback_reason=(None if result is None else result.fallback_reason),
        )

    @property
    def guard_status(self) -> str:
        if self.run.policy not in GUARDED_POLICIES:
            return "not_applicable"
        return "passed" if self.fallback_reason is None else "exact_mean_fallback"


@dataclass(frozen=True, slots=True)
class FinalizationTrackInput:
    """Fully authenticated values for one frozen policy-run reduction."""

    run: PolicyRunSpec
    outer_context_predictions: tuple[OuterContextPrediction, ...]
    outer_outcomes: tuple[ContextRow, ...]
    pool_selection: PoolSelectionEvidence
    pool_revealed_contexts: tuple[ContextRow, ...]
    prediction_view: tuple[PoolCandidate, ...]
    outer_selected_sequence_ids: tuple[str, ...]
    outer_selected_unique_component_count: int
    outer_selected_max_component_occupancy: int

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="finalization track input run")
        if type(self.pool_selection) is not PoolSelectionEvidence:
            raise TypeError("finalization track requires exact pool-selection evidence")
        if self.pool_selection.run != run:
            raise ValueError("pool selection evidence belongs to another track")

        predictions = self.outer_context_predictions
        outcomes = self.outer_outcomes
        expected_contexts = EXPECTED_CONTEXTS_BY_FOLD[run.rotation.outer_fold]
        if type(predictions) is not tuple or any(
            type(item) is not OuterContextPrediction for item in predictions
        ):
            raise TypeError("outer predictions must be an exact immutable tuple")
        if type(outcomes) is not tuple or any(type(item) is not ContextRow for item in outcomes):
            raise TypeError("outer outcomes must be an exact immutable ContextRow tuple")
        if len(predictions) != expected_contexts or len(outcomes) != expected_contexts:
            raise ValueError("outer prediction/outcome census differs from the frozen fold")
        if any(item.run != run for item in predictions):
            raise ValueError("outer context prediction belongs to another track")
        if any(item.fold != run.rotation.outer_fold for item in outcomes):
            raise ValueError("outer outcome belongs to another fold")

        revealed = self.pool_revealed_contexts
        if type(revealed) is not tuple or any(type(item) is not ContextRow for item in revealed):
            raise TypeError("pool reveal must be an exact immutable ContextRow tuple")
        reveal_ids = tuple(item.example_id for item in revealed)
        if reveal_ids != tuple(sorted(set(reveal_ids))) or any(
            item.fold != run.rotation.pool_fold for item in revealed
        ):
            raise ValueError("pool reveal has the wrong example order, identity, or fold")
        represented = {item.sequence_id for item in revealed}
        selected = set(self.pool_selection.selected_sequence_ids)
        if run.policy == NO_QUERY:
            if revealed or selected:
                raise ValueError("no-query pool acquisition must be exactly empty")
        elif not revealed or represented != selected:
            raise ValueError("pool reveal does not represent exactly the committed sequences")

        view = self.prediction_view
        if type(view) is not tuple or any(type(item) is not PoolCandidate for item in view):
            raise TypeError("prediction view must be an exact immutable PoolCandidate tuple")
        view_ids = tuple(item.sequence_id for item in view)
        if (
            len(view) != EXPECTED_SUPPORT_BY_FOLD[run.rotation.pool_fold]
            or view_ids != tuple(sorted(set(view_ids)))
            or any(
                item.rotation_id != run.rotation.rotation_id or not item.eligible for item in view
            )
        ):
            raise ValueError("prediction view differs from the frozen supported pool")
        if any(sequence_id not in set(view_ids) for sequence_id in selected):
            raise ValueError("pool selected ID is absent from the common prediction view")

        _canonical_sha256_ids(
            self.outer_selected_sequence_ids,
            label="outer selected sequence IDs",
            expected_count=run.expected_outer_selection_count,
        )
        if (
            type(self.outer_selected_unique_component_count) is not int
            or not 5 <= self.outer_selected_unique_component_count <= 10
            or type(self.outer_selected_max_component_occupancy) is not int
            or not 1 <= self.outer_selected_max_component_occupancy <= 2
        ):
            raise ValueError("outer selected component census violates the strict cap")


@dataclass(frozen=True, slots=True)
class RotationMetric:
    """One reduced track row; no probability, label, sequence, or selected ID remains."""

    run: PolicyRunSpec
    target_metrics: TargetMetricResult
    next_round_outer_top10_mean_reward: float
    queried_outcome_mean_reward: float | None
    queried_unique_diversity_components: int
    queried_mean_pairwise_feature_cosine_distance: float | None
    queried_metric_null_reason: str | None
    revealed_context_count: int
    per_target_revealed_context_counts: tuple[tuple[str, int], ...]
    guard_status: str
    outer_selected_unique_component_count: int
    outer_selected_max_component_occupancy: int

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="rotation metric run")
        if type(self.target_metrics) is not TargetMetricResult:
            raise TypeError("rotation metric requires an exact TargetMetricResult")
        _finite_float(
            self.next_round_outer_top10_mean_reward,
            label="next-round outer reward",
        )
        queried_reward = _optional_finite(
            self.queried_outcome_mean_reward,
            label="queried outcome reward",
        )
        queried_distance = _optional_finite(
            self.queried_mean_pairwise_feature_cosine_distance,
            label="queried pairwise feature distance",
        )
        if (
            type(self.queried_unique_diversity_components) is not int
            or self.queried_unique_diversity_components < 0
            or type(self.revealed_context_count) is not int
            or self.revealed_context_count < 0
        ):
            raise ValueError("rotation metric queried censuses must be nonnegative integers")
        if (
            type(self.per_target_revealed_context_counts) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
                for item in self.per_target_revealed_context_counts
            )
            or tuple(target for target, _count in self.per_target_revealed_context_counts)
            != TARGETS
            or any(
                type(count) is not int or count < 0
                for _target, count in self.per_target_revealed_context_counts
            )
            or sum(count for _target, count in self.per_target_revealed_context_counts)
            != self.revealed_context_count
        ):
            raise ValueError("rotation metric revealed target census is invalid")
        if self.guard_status not in GUARD_STATUSES:
            raise ValueError("rotation metric guard status is outside the frozen enum")
        if (run.policy in GUARDED_POLICIES) != (self.guard_status != "not_applicable"):
            raise ValueError("rotation metric guard status disagrees with its policy")
        if (
            type(self.outer_selected_unique_component_count) is not int
            or not 5 <= self.outer_selected_unique_component_count <= 10
            or type(self.outer_selected_max_component_occupancy) is not int
            or not 1 <= self.outer_selected_max_component_occupancy <= 2
        ):
            raise ValueError("rotation metric outer component census is invalid")
        target_document = self.target_metrics.document()
        by_target = target_document["by_target"]
        assert isinstance(by_target, dict)
        target_context_count = sum(by_target[target]["n"] for target in TARGETS)
        if target_context_count != EXPECTED_CONTEXTS_BY_FOLD[run.rotation.outer_fold]:
            raise ValueError("rotation metric target census differs from the frozen outer fold")
        if run.policy == NO_QUERY:
            if (
                queried_reward is not None
                or queried_distance is not None
                or self.queried_metric_null_reason != "no_query_has_empty_acquisition"
                or self.queried_unique_diversity_components != 0
                or self.revealed_context_count != 0
                or any(count != 0 for _target, count in self.per_target_revealed_context_counts)
            ):
                raise ValueError("no-query metric violates the exact empty-acquisition convention")
        elif (
            queried_reward is None
            or queried_distance is None
            or self.queried_metric_null_reason is not None
            or self.queried_unique_diversity_components < 1
            or self.revealed_context_count < 1
        ):
            raise ValueError("queried policy metric lacks its descriptive acquisition values")

    @property
    def aggregate_metrics(self) -> AggregateMetricValues:
        return AggregateMetricValues(
            macro_brier=self.target_metrics.macro_brier,
            macro_negative_log_likelihood=(self.target_metrics.macro_negative_log_likelihood),
            macro_roc_auc=self.target_metrics.macro_roc_auc,
            macro_average_precision=self.target_metrics.macro_average_precision,
            macro_ece_10=self.target_metrics.macro_ece_10,
            next_round_outer_top10_mean_reward=(self.next_round_outer_top10_mean_reward),
            queried_outcome_mean_reward=self.queried_outcome_mean_reward,
            queried_unique_diversity_components=float(self.queried_unique_diversity_components),
            queried_mean_pairwise_feature_cosine_distance=(
                self.queried_mean_pairwise_feature_cosine_distance
            ),
            revealed_context_count=float(self.revealed_context_count),
        )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run": self.run.document(),
            "target_metrics": self.target_metrics.document(),
            "next_round_outer_top10_mean_reward_hex": _hex(
                self.next_round_outer_top10_mean_reward,
                label="next-round outer reward",
            ),
            "queried_outcome_mean_reward_hex": (
                None
                if self.queried_outcome_mean_reward is None
                else _hex(self.queried_outcome_mean_reward, label="queried outcome reward")
            ),
            "queried_unique_diversity_components": self.queried_unique_diversity_components,
            "queried_mean_pairwise_feature_cosine_distance_hex": (
                None
                if self.queried_mean_pairwise_feature_cosine_distance is None
                else _hex(
                    self.queried_mean_pairwise_feature_cosine_distance,
                    label="queried pairwise feature distance",
                )
            ),
            "queried_metric_null_reason": self.queried_metric_null_reason,
            "revealed_context_count": self.revealed_context_count,
            "per_target_revealed_context_counts": dict(self.per_target_revealed_context_counts),
            "guard_status": self.guard_status,
            "outer_selected_unique_component_count": (self.outer_selected_unique_component_count),
            "outer_selected_max_component_occupancy": (self.outer_selected_max_component_occupancy),
        }

    @classmethod
    def from_reduced_columns(
        cls,
        *,
        run: PolicyRunSpec,
        targets: Sequence[str],
        labels: Sequence[int],
        probabilities: Sequence[float],
        next_round_outer_top10_mean_reward: float,
        queried_outcome_mean_reward: float | None,
        queried_unique_diversity_components: int,
        queried_mean_pairwise_feature_cosine_distance: float | None,
        revealed_context_count: int,
        per_target_revealed_context_counts: Mapping[str, int],
        guard_status: str,
        outer_selected_unique_component_count: int,
        outer_selected_max_component_occupancy: int,
    ) -> RotationMetric:
        """Construct a redacted row from already joined metric columns.

        This is useful for deterministic aggregation tests and remains a pure
        value constructor.  The production finalizer uses :func:`evaluate_track`
        so these columns are never caller authority.
        """

        if type(per_target_revealed_context_counts) is not dict or set(
            per_target_revealed_context_counts
        ) != set(TARGETS):
            raise ValueError("reduced revealed target counts must bind all seven targets")
        return cls(
            run=run,
            target_metrics=TargetMetricResult.from_columns(targets, labels, probabilities),
            next_round_outer_top10_mean_reward=next_round_outer_top10_mean_reward,
            queried_outcome_mean_reward=queried_outcome_mean_reward,
            queried_unique_diversity_components=queried_unique_diversity_components,
            queried_mean_pairwise_feature_cosine_distance=(
                queried_mean_pairwise_feature_cosine_distance
            ),
            queried_metric_null_reason=(
                "no_query_has_empty_acquisition" if run.policy == NO_QUERY else None
            ),
            revealed_context_count=revealed_context_count,
            per_target_revealed_context_counts=tuple(
                (target, per_target_revealed_context_counts[target]) for target in TARGETS
            ),
            guard_status=guard_status,
            outer_selected_unique_component_count=outer_selected_unique_component_count,
            outer_selected_max_component_occupancy=outer_selected_max_component_occupancy,
        )


def _context_identity_from_prediction(
    prediction: OuterContextPrediction,
) -> tuple[str, str, str, str, int]:
    context = prediction.context
    return (
        context.example_id,
        context.sequence_id,
        context.target,
        context.gram,
        context.fold,
    )


def _context_identity_from_outcome(row: ContextRow) -> tuple[str, str, str, str, int]:
    return (row.example_id, row.sequence_id, row.target, row.gram, row.fold)


def evaluate_track(value: FinalizationTrackInput) -> RotationMetric:
    """Join one authenticated prediction/outcome track and reduce its metrics."""

    if type(value) is not FinalizationTrackInput:
        raise TypeError("track evaluation requires an exact FinalizationTrackInput")
    predictions = value.outer_context_predictions
    outcomes = value.outer_outcomes
    prediction_keys = tuple(_context_identity_from_prediction(item) for item in predictions)
    outcome_keys = tuple(_context_identity_from_outcome(item) for item in outcomes)
    prediction_example_ids = tuple(key[0] for key in prediction_keys)
    outcome_example_ids = tuple(key[0] for key in outcome_keys)
    if prediction_example_ids != tuple(sorted(set(prediction_example_ids))):
        raise ValueError("outer context predictions must use ascending unique example IDs")
    if outcome_example_ids != tuple(sorted(set(outcome_example_ids))):
        raise ValueError("outer outcomes must use ascending unique example IDs")
    if prediction_keys != outcome_keys:
        raise ValueError("outer prediction and outcome identity streams differ exactly")

    pool_ids = value.pool_selection.selected_sequence_ids
    view_by_id = {item.sequence_id: item for item in value.prediction_view}
    if len(view_by_id) != len(value.prediction_view):
        raise ValueError("prediction view contains duplicate sequence IDs")
    if value.run.policy == NO_QUERY:
        queried_reward = None
        unique_components = 0
        queried_distance = None
    else:
        revealed_outcomes = {
            item.sequence_id: item
            for item in aggregate_observed_sequence_outcomes(value.pool_revealed_contexts)
        }
        queried_reward = observed_batch_reward(revealed_outcomes, pool_ids)
        selected_candidates = tuple(view_by_id[sequence_id] for sequence_id in pool_ids)
        unique_components = len({item.diversity_component_id for item in selected_candidates})
        queried_distance = mean_pairwise_feature_cosine_distance(
            tuple(item.features for item in selected_candidates)
        )

    outer_outcomes = {
        item.sequence_id: item for item in aggregate_observed_sequence_outcomes(outcomes)
    }
    outer_reward = observed_batch_reward(
        outer_outcomes,
        value.outer_selected_sequence_ids,
    )
    per_target = {
        target: sum(row.target == target for row in value.pool_revealed_contexts)
        for target in TARGETS
    }
    return RotationMetric.from_reduced_columns(
        run=value.run,
        targets=tuple(row.target for row in outcomes),
        labels=tuple(row.label for row in outcomes),
        probabilities=tuple(item.probability for item in predictions),
        next_round_outer_top10_mean_reward=outer_reward,
        queried_outcome_mean_reward=queried_reward,
        queried_unique_diversity_components=unique_components,
        queried_mean_pairwise_feature_cosine_distance=queried_distance,
        revealed_context_count=len(value.pool_revealed_contexts),
        per_target_revealed_context_counts=per_target,
        guard_status=value.pool_selection.guard_status,
        outer_selected_unique_component_count=(value.outer_selected_unique_component_count),
        outer_selected_max_component_occupancy=(value.outer_selected_max_component_occupancy),
    )


def _aggregate_metric_values(values: Sequence[AggregateMetricValues]) -> AggregateMetricValues:
    if not values:
        raise ValueError("aggregate metric reduction requires at least one row")
    result: dict[str, float | None] = {}
    for name in AGGREGATE_METRIC_NAMES:
        column = tuple(item.value(name) for item in values)
        if all(item is None for item in column):
            result[name] = None
        elif any(item is None for item in column):
            raise ValueError(f"aggregate metric {name} mixes null and finite values")
        else:
            result[name] = _mean(
                tuple(float(item) for item in column if item is not None),
                label=f"aggregate metric {name}",
            )
    return AggregateMetricValues.from_mapping(result)


@dataclass(frozen=True, slots=True)
class _PolicyRotationMetric:
    rotation_id: str
    rotation_document: Mapping[str, object]
    policy: str
    contributing_track_ids: tuple[str, ...]
    metrics: AggregateMetricValues

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation": dict(self.rotation_document),
            "policy": self.policy,
            "contributing_track_count": len(self.contributing_track_ids),
            "contributing_track_ids_sha256": _ordered_text_stream_sha256(
                self.contributing_track_ids,
                label="contributing track IDs",
            ),
            "metrics": self.metrics.document(),
        }


@dataclass(frozen=True, slots=True)
class _OuterFoldUnit:
    policy: str
    outer_fold: int
    contributing_rotation_ids: tuple[str, ...]
    metrics: AggregateMetricValues

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": self.policy,
            "outer_fold": self.outer_fold,
            "contributing_rotation_count": len(self.contributing_rotation_ids),
            "contributing_rotation_ids_sha256": _ordered_text_stream_sha256(
                self.contributing_rotation_ids,
                label="contributing rotation IDs",
            ),
            "metrics": self.metrics.document(),
        }


@dataclass(frozen=True, slots=True)
class _PolicyPointEstimate:
    policy: str
    metrics: AggregateMetricValues

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": self.policy,
            "outer_fold_unit_count": 5,
            "metrics": self.metrics.document(),
        }


@dataclass(frozen=True, slots=True)
class PairedComparison:
    comparison_id: str
    candidate_policy: str
    control_policy: str
    metric: str
    outer_fold_differences: tuple[float, ...]
    point: float
    lower: float
    upper: float
    improved_outer_fold_count: int
    replicates: int
    seed: int
    resample_indices_sha256: str

    def __post_init__(self) -> None:
        if self.comparison_id not in COMPARISON_ORDER:
            raise ValueError("paired comparison ID is outside the frozen order")
        if self.candidate_policy not in POLICY_ORDER or self.control_policy not in POLICY_ORDER:
            raise ValueError("paired comparison policy is outside the frozen registry")
        if self.metric not in AGGREGATE_METRIC_NAMES:
            raise ValueError("paired comparison metric is outside the frozen registry")
        if type(self.outer_fold_differences) is not tuple or len(self.outer_fold_differences) != 5:
            raise ValueError("paired comparison requires five outer-fold differences")
        for label, value in (
            ("point", self.point),
            ("lower", self.lower),
            ("upper", self.upper),
            *(
                (f"outer fold {index}", value)
                for index, value in enumerate(self.outer_fold_differences)
            ),
        ):
            _finite_float(value, label=f"paired comparison {label}")
        if (
            type(self.improved_outer_fold_count) is not int
            or not 0 <= self.improved_outer_fold_count <= 5
            or self.replicates != BOOTSTRAP_REPLICATES
            or self.seed != BOOTSTRAP_SEED
            or self.resample_indices_sha256 != bootstrap_indices_sha256()
        ):
            raise ValueError("paired comparison bootstrap metadata changed")
        reconstructed = paired_outer_bootstrap(
            {fold: self.outer_fold_differences[fold] for fold in range(5)},
            {fold: 0.0 for fold in range(5)},
        )
        if (
            self.point != reconstructed.point
            or self.lower != reconstructed.lower
            or self.upper != reconstructed.upper
        ):
            raise ValueError("paired comparison interval differs from its five differences")
        spec = next(item for item in _COMPARISON_SPECS if item[0] == self.comparison_id)
        expected_improved = sum(
            difference < 0.0 if spec[4] == "lower" else difference > 0.0
            for difference in self.outer_fold_differences
        )
        if (
            self.candidate_policy != spec[1]
            or self.control_policy != spec[2]
            or self.metric != spec[3]
            or self.improved_outer_fold_count != expected_improved
        ):
            raise ValueError("paired comparison identity or improvement count changed")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "comparison_id": self.comparison_id,
            "candidate_policy": self.candidate_policy,
            "control_policy": self.control_policy,
            "metric": self.metric,
            "difference": "candidate_minus_control",
            "outer_fold_differences_hex": [
                _hex(value, label="paired outer-fold difference")
                for value in self.outer_fold_differences
            ],
            "point_hex": _hex(self.point, label="paired comparison point"),
            "lower_hex": _hex(self.lower, label="paired comparison lower bound"),
            "upper_hex": _hex(self.upper, label="paired comparison upper bound"),
            "improved_outer_fold_count": self.improved_outer_fold_count,
            "replicates": self.replicates,
            "seed": self.seed,
            "resample_indices_sha256": self.resample_indices_sha256,
        }


def _aggregate_campaign(
    rows: tuple[RotationMetric, ...],
) -> tuple[
    tuple[_PolicyRotationMetric, ...],
    tuple[_OuterFoldUnit, ...],
    tuple[_PolicyPointEstimate, ...],
]:
    by_track = {row.run.track_id: row for row in rows}
    policy_rotation_rows: list[_PolicyRotationMetric] = []
    for rotation in ordered_rotations():
        rotation_runs = tuple(run for run in ordered_policy_runs() if run.rotation == rotation)
        for policy in POLICY_ORDER:
            contributing = tuple(run for run in rotation_runs if run.policy == policy)
            expected_count = 5 if policy == RANDOM else 1
            if len(contributing) != expected_count:
                raise AssertionError("frozen policy-rotation contribution count changed")
            metrics = _aggregate_metric_values(
                tuple(by_track[run.track_id].aggregate_metrics for run in contributing)
            )
            policy_rotation_rows.append(
                _PolicyRotationMetric(
                    rotation_id=rotation.rotation_id,
                    rotation_document=rotation.document(),
                    policy=policy,
                    contributing_track_ids=tuple(run.track_id for run in contributing),
                    metrics=metrics,
                )
            )
    if len(policy_rotation_rows) != 140:
        raise AssertionError("policy-rotation census changed")

    by_policy_rotation = {(row.policy, row.rotation_id): row for row in policy_rotation_rows}
    outer_rows: list[_OuterFoldUnit] = []
    for policy in POLICY_ORDER:
        for outer_fold in range(5):
            rotations = tuple(
                rotation for rotation in ordered_rotations() if rotation.outer_fold == outer_fold
            )
            metric_map: dict[str, float | None] = {}
            for name in AGGREGATE_METRIC_NAMES:
                if policy == RANDOM:
                    raw_random = {
                        (run.rotation.rotation_id, int(run.seed)): float(
                            by_track[run.track_id].aggregate_metrics.value(name)
                        )
                        for run in rows_by_policy(rows, RANDOM)
                        if by_track[run.track_id].aggregate_metrics.value(name) is not None
                    }
                    metric_map[name] = (
                        None
                        if not raw_random
                        else aggregate_random_outer_units(raw_random)[outer_fold]
                    )
                else:
                    rotation_values = {
                        rotation.rotation_id: by_policy_rotation[
                            (policy, rotation.rotation_id)
                        ].metrics.value(name)
                        for rotation in ordered_rotations()
                    }
                    if all(value is None for value in rotation_values.values()):
                        metric_map[name] = None
                    elif any(value is None for value in rotation_values.values()):
                        raise ValueError(f"policy {policy} metric {name} mixes null and finite")
                    else:
                        metric_map[name] = aggregate_deterministic_outer_units(
                            {
                                key: float(value)
                                for key, value in rotation_values.items()
                                if value is not None
                            }
                        )[outer_fold]
            outer_rows.append(
                _OuterFoldUnit(
                    policy=policy,
                    outer_fold=outer_fold,
                    contributing_rotation_ids=tuple(rotation.rotation_id for rotation in rotations),
                    metrics=AggregateMetricValues.from_mapping(metric_map),
                )
            )
    if len(outer_rows) != 35:
        raise AssertionError("outer-fold unit census changed")

    point_rows = tuple(
        _PolicyPointEstimate(
            policy=policy,
            metrics=_aggregate_metric_values(
                tuple(row.metrics for row in outer_rows if row.policy == policy)
            ),
        )
        for policy in POLICY_ORDER
    )
    return tuple(policy_rotation_rows), tuple(outer_rows), point_rows


def rows_by_policy(rows: Sequence[RotationMetric], policy: str) -> tuple[PolicyRunSpec, ...]:
    """Return exact run identities for one policy, retaining frozen row order."""

    if policy not in POLICY_ORDER:
        raise ValueError("unknown policy")
    return tuple(row.run for row in rows if row.run.policy == policy)


def _paired_comparisons(
    outer_rows: tuple[_OuterFoldUnit, ...],
) -> tuple[PairedComparison, ...]:
    by_policy_fold = {(row.policy, row.outer_fold): row for row in outer_rows}
    result: list[PairedComparison] = []
    for comparison_id, candidate, control, metric, direction in _COMPARISON_SPECS:
        candidate_values = {
            fold: by_policy_fold[(candidate, fold)].metrics.value(metric) for fold in range(5)
        }
        control_values = {
            fold: by_policy_fold[(control, fold)].metrics.value(metric) for fold in range(5)
        }
        if any(value is None for value in (*candidate_values.values(), *control_values.values())):
            raise ValueError(f"paired comparison {comparison_id} cannot consume null values")
        candidate_finite = {fold: float(candidate_values[fold]) for fold in range(5)}
        control_finite = {fold: float(control_values[fold]) for fold in range(5)}
        interval = paired_outer_bootstrap(candidate_finite, control_finite)
        differences = tuple(candidate_finite[fold] - control_finite[fold] for fold in range(5))
        improved = sum(
            difference < 0.0 if direction == "lower" else difference > 0.0
            for difference in differences
        )
        result.append(
            PairedComparison(
                comparison_id=comparison_id,
                candidate_policy=candidate,
                control_policy=control,
                metric=metric,
                outer_fold_differences=differences,
                point=interval.point,
                lower=interval.lower,
                upper=interval.upper,
                improved_outer_fold_count=improved,
                replicates=interval.replicates,
                seed=interval.seed,
                resample_indices_sha256=interval.resample_indices_sha256,
            )
        )
    return tuple(result)


def _float_requirement(
    requirement_id: str,
    operator: str,
    observed: float,
    threshold: float,
) -> dict[str, object]:
    operations = {
        "lt": observed < threshold,
        "le": observed <= threshold,
        "ge": observed >= threshold,
        "gt": observed > threshold,
    }
    if operator not in operations:
        raise ValueError("unknown float promotion comparator")
    return {
        "requirement_id": requirement_id,
        "operator": operator,
        "threshold_hex": threshold.hex(),
        "observed_hex": observed.hex(),
        "threshold_integer": None,
        "observed_integer": None,
        "numeric_representation": "binary64_hex",
        "passed": operations[operator],
    }


def _integer_requirement(
    requirement_id: str,
    operator: str,
    observed: int,
    threshold: int,
) -> dict[str, object]:
    operations = {
        "ge": observed >= threshold,
        "eq": observed == threshold,
    }
    if operator not in operations:
        raise ValueError("unknown integer promotion comparator")
    return {
        "requirement_id": requirement_id,
        "operator": operator,
        "threshold_hex": None,
        "observed_hex": None,
        "threshold_integer": threshold,
        "observed_integer": observed,
        "numeric_representation": "integer",
        "passed": operations[operator],
    }


def promotion_decision_document(
    comparisons: Sequence[PairedComparison],
    *,
    guarded_track_count: int,
    valid_guarded_track_count: int,
) -> dict[str, object]:
    """Apply the six all-required frozen promotion items exactly."""

    values = tuple(comparisons)
    if (
        len(values) != len(COMPARISON_ORDER)
        or tuple(item.comparison_id for item in values) != COMPARISON_ORDER
        or any(type(item) is not PairedComparison for item in values)
    ):
        raise ValueError("promotion decision requires seven comparisons in frozen order")
    if (
        type(guarded_track_count) is not int
        or type(valid_guarded_track_count) is not int
        or not 0 <= valid_guarded_track_count <= guarded_track_count
    ):
        raise ValueError("promotion guard census is invalid")
    by_id = {item.comparison_id: item for item in values}
    brier = by_id[COMPARISON_ORDER[0]]
    nll = by_id[COMPARISON_ORDER[1]]
    reward = by_id[COMPARISON_ORDER[2]]
    distance = by_id[COMPARISON_ORDER[3]]
    components = by_id[COMPARISON_ORDER[4]]
    no_query = by_id[COMPARISON_ORDER[5]]
    random = by_id[COMPARISON_ORDER[6]]

    item_requirements = (
        (
            "brier_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_brier_point_strictly_below_zero",
                    "lt",
                    brier.point,
                    0.0,
                ),
                _integer_requirement(
                    "mixed_brier_improved_outer_folds_at_least_three",
                    "ge",
                    brier.improved_outer_fold_count,
                    3,
                ),
                _float_requirement(
                    "mixed_minus_mean_brier_ci_upper_at_most_0_005",
                    "le",
                    brier.upper,
                    0.005,
                ),
            ),
        ),
        (
            "nll_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_nll_ci_upper_at_most_0_01",
                    "le",
                    nll.upper,
                    0.01,
                ),
            ),
        ),
        (
            "outer_reward_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_outer_reward_ci_lower_at_least_minus_0_01",
                    "ge",
                    reward.lower,
                    -0.01,
                ),
            ),
        ),
        (
            "queried_diversity_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_feature_distance_point_strictly_above_zero",
                    "gt",
                    distance.point,
                    0.0,
                ),
                _float_requirement(
                    "mixed_minus_mean_unique_components_point_at_least_zero",
                    "ge",
                    components.point,
                    0.0,
                ),
            ),
        ),
        (
            "brier_vs_no_query_and_random",
            (
                _float_requirement(
                    "mixed_minus_no_query_brier_point_strictly_below_zero",
                    "lt",
                    no_query.point,
                    0.0,
                ),
                _float_requirement(
                    "mixed_minus_random_brier_point_strictly_below_zero",
                    "lt",
                    random.point,
                    0.0,
                ),
            ),
        ),
        (
            "guarded_selection_validity",
            (
                _integer_requirement(
                    "guarded_track_count_equals_sixty",
                    "eq",
                    guarded_track_count,
                    EXPECTED_GUARDED_TRACKS,
                ),
                _integer_requirement(
                    "valid_guarded_track_count_equals_sixty",
                    "eq",
                    valid_guarded_track_count,
                    EXPECTED_GUARDED_TRACKS,
                ),
            ),
        ),
    )
    if tuple(item_id for item_id, _requirements in item_requirements) != PROMOTION_ITEM_ORDER:
        raise AssertionError("promotion item order changed")
    items = [
        {
            "item_id": item_id,
            "requirements": list(requirements),
            "passed": all(bool(requirement["passed"]) for requirement in requirements),
        }
        for item_id, requirements in item_requirements
    ]
    passed = all(bool(item["passed"]) for item in items)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": PROMOTION_DECISION_ARTIFACT,
        "items": items,
        "all_items_required": True,
        "promising_for_prospective_followup": passed,
        "disposition": (
            "promising_for_new_prospective_or_chronological_evaluation" if passed else "v2_no_go"
        ),
        "authorizes_final_50000_library": False,
        "authorizes_final_top_100": False,
        "authorizes_uncertainty_quota": False,
    }


@dataclass(frozen=True, slots=True)
class FinalizationPayloads:
    """The seven canonical, unpublished finalization payload byte strings."""

    payloads: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if (
            type(self.payloads) is not tuple
            or tuple(path for path, _payload in self.payloads) != FINALIZE_PAYLOAD_PATHS
            or any(
                type(path) is not str or type(payload) is not bytes
                for path, payload in self.payloads
            )
        ):
            raise ValueError("finalization payload inventory or order changed")

    def read(self, path: str) -> bytes:
        try:
            return dict(self.payloads)[path]
        except KeyError as error:
            raise ValueError(f"unknown finalization payload path {path!r}") from error


def finalize_rotation_metrics(rows: Sequence[RotationMetric]) -> FinalizationPayloads:
    """Aggregate exactly 220 already reduced rows into the seven frozen payloads."""

    values = tuple(rows)
    expected_runs = ordered_policy_runs()
    if (
        len(values) != EXPECTED_POLICY_RUNS
        or any(type(item) is not RotationMetric for item in values)
        or tuple(item.run for item in values) != expected_runs
    ):
        raise ValueError("rotation metrics must contain 220 exact rows in frozen track order")
    policy_rotation, outer_units, point_estimates = _aggregate_campaign(values)
    comparisons = _paired_comparisons(outer_units)
    guarded = tuple(item for item in values if item.run.policy in GUARDED_POLICIES)
    valid_guarded = tuple(
        item for item in guarded if item.guard_status in {"passed", "exact_mean_fallback"}
    )
    decision = promotion_decision_document(
        comparisons,
        guarded_track_count=len(guarded),
        valid_guarded_track_count=len(valid_guarded),
    )

    rotation_payload = canonical_jsonl_bytes(item.document() for item in values)
    policy_rotation_payload = canonical_jsonl_bytes(item.document() for item in policy_rotation)
    outer_payload = canonical_jsonl_bytes(item.document() for item in outer_units)
    point_payload = canonical_jsonl_bytes(item.document() for item in point_estimates)
    comparison_payload = canonical_jsonl_bytes(item.document() for item in comparisons)
    decision_payload = canonical_json_bytes(decision)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact": FINALIZE_SUMMARY_ARTIFACT,
        "rotation_metric_count": len(values),
        "policy_rotation_metric_count": len(policy_rotation),
        "outer_fold_unit_count": len(outer_units),
        "policy_point_estimate_count": len(point_estimates),
        "paired_comparison_count": len(comparisons),
        "guarded_track_count": len(guarded),
        "valid_guarded_track_count": len(valid_guarded),
        "bootstrap_indices_sha256": bootstrap_indices_sha256(),
        "rotation_metrics_sha256": sha256_bytes(rotation_payload),
        "policy_rotation_metrics_sha256": sha256_bytes(policy_rotation_payload),
        "outer_fold_units_sha256": sha256_bytes(outer_payload),
        "policy_point_estimates_sha256": sha256_bytes(point_payload),
        "paired_comparisons_sha256": sha256_bytes(comparison_payload),
        "promotion_decision_sha256": sha256_bytes(decision_payload),
        "promising_for_prospective_followup": decision["promising_for_prospective_followup"],
    }
    return FinalizationPayloads(
        (
            ("outer-fold-units.jsonl", outer_payload),
            ("paired-comparisons.jsonl", comparison_payload),
            ("policy-point-estimates.jsonl", point_payload),
            ("policy-rotation-metrics.jsonl", policy_rotation_payload),
            ("promotion-decision.json", decision_payload),
            ("rotation-metrics.jsonl", rotation_payload),
            ("summary.json", canonical_json_bytes(summary)),
        )
    )


def finalize_campaign(values: Sequence[FinalizationTrackInput]) -> FinalizationPayloads:
    """Validate and reduce the complete authenticated 220-track campaign."""

    inputs = tuple(values)
    if (
        len(inputs) != EXPECTED_POLICY_RUNS
        or any(type(item) is not FinalizationTrackInput for item in inputs)
        or tuple(item.run for item in inputs) != ordered_policy_runs()
    ):
        raise ValueError("finalization inputs must contain 220 exact tracks in frozen order")

    canonical_outcomes: dict[int, tuple[ContextRow, ...]] = {}
    canonical_views: dict[str, tuple[PoolCandidate, ...]] = {}
    for item in inputs:
        prior_outcomes = canonical_outcomes.setdefault(
            item.run.rotation.outer_fold,
            item.outer_outcomes,
        )
        if prior_outcomes != item.outer_outcomes:
            raise ValueError("same outer-fold outcome payload differs across rotations or tracks")
        prior_view = canonical_views.setdefault(
            item.run.rotation.rotation_id,
            item.prediction_view,
        )
        if prior_view != item.prediction_view:
            raise ValueError("common prediction view differs across one rotation's tracks")
    return finalize_rotation_metrics(tuple(evaluate_track(item) for item in inputs))


__all__ = [
    "AGGREGATE_METRIC_NAMES",
    "COMPARISON_ORDER",
    "FINALIZE_ARTIFACT",
    "FINALIZE_PAYLOAD_PATHS",
    "FINALIZE_SUMMARY_ARTIFACT",
    "GUARDED_POLICIES",
    "GUARD_STATUSES",
    "PROMOTION_DECISION_ARTIFACT",
    "PROMOTION_ITEM_ORDER",
    "AggregateMetricValues",
    "FinalizationPayloads",
    "FinalizationTrackInput",
    "PairedComparison",
    "PoolSelectionEvidence",
    "RotationMetric",
    "TargetMetricResult",
    "evaluate_track",
    "finalize_campaign",
    "finalize_rotation_metrics",
    "promotion_decision_document",
]

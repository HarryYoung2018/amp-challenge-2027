"""Canonical semantic records for native-diffusion v1 pilot evidence.

The numerical core deliberately returns strongly validated Python objects.
This module is the narrow conversion boundary from those objects to the
path-free JSON documents stored in evaluator and pilot bundles.  It does not
read files, run a model, publish artifacts, or decide whether evidence is
authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

from amp_challenge.generators.diffusion.v1.pilot_scoring import (
    BootstrapPanel,
    CheckpointSelection,
    EqualFoldMetrics,
    FoldMethodMetrics,
    LoucoPlan,
    MetricSummary,
    PilotGateDecision,
    RowNll,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FOLD_METHOD_ORDER = (
    "C0",
    "C0T",
    "R128-000250",
    "R128-000500",
    "R128-001000",
    "R128-002000",
    "R128-004000",
)


def canonical_json_bytes(value: object) -> bytes:
    """Return compact sorted finite UTF-8 JSON with one terminal LF."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def metric_summary_record(value: MetricSummary) -> dict[str, object]:
    """Serialize a complete metric summary and its ECE sufficient statistics."""

    if type(value) is not MetricSummary:
        raise TypeError("value must be an exact MetricSummary")
    value.revalidate()
    return {
        "nll": value.nll,
        "perplexity": value.perplexity,
        "top1_accuracy": value.top1_accuracy,
        "top3_accuracy": value.top3_accuracy,
        "multiclass_brier": value.multiclass_brier,
        "ece": value.ece,
        "ece_mass": list(value.ece_mass),
        "ece_confidence": list(value.ece_confidence),
        "ece_correct": list(value.ece_correct),
    }


def row_nll_record(value: RowNll) -> dict[str, object]:
    """Serialize the row evidence needed for exact stratified bootstrapping."""

    if type(value) is not RowNll:
        raise TypeError("value must be an exact RowNll")
    value.revalidate()
    return {
        "sequence_id": value.sequence_id,
        "homology_component_id": value.homology_component_id,
        "union_component_id": value.union_component_id,
        "sampling_weight": value.sampling_weight,
        "nll": value.nll,
        "nll_by_timestep_bin": list(value.nll_by_timestep_bin),
    }


def louco_plan_record(value: LoucoPlan | None) -> dict[str, object] | None:
    """Serialize every frozen LOUCO choice and leave-out objective."""

    if value is None:
        return None
    if type(value) is not LoucoPlan:
        raise TypeError("value must be an exact LoucoPlan or None")
    value.revalidate()
    return {
        "method": value.method,
        "checkpoint_step": value.checkpoint_step,
        "component_ids": list(value.component_ids),
        "choices": [
            [
                {
                    "residual_lambda": choice.residual_lambda,
                    "temperature": choice.temperature,
                }
                for choice in choices
            ]
            for choices in value.choices
        ],
        "leaveout_objectives": [list(values) for values in value.leaveout_objectives],
        "grid": [
            {
                "residual_lambda": choice.residual_lambda,
                "temperature": choice.temperature,
            }
            for choice in value.grid
        ],
    }


def fold_method_record(value: FoldMethodMetrics) -> dict[str, object]:
    """Serialize one fold/method result without dropping bootstrap evidence."""

    if type(value) is not FoldMethodMetrics:
        raise TypeError("value must be an exact FoldMethodMetrics")
    value.revalidate()
    return {
        "method": value.method,
        "checkpoint_step": value.checkpoint_step,
        "outer_fold": value.outer_fold,
        "row_nll": [row_nll_record(row) for row in value.row_nll],
        "overall": metric_summary_record(value.overall),
        "timestep_bins": [metric_summary_record(item) for item in value.timestep_bins],
        "louco_plan": louco_plan_record(value.louco_plan),
    }


def fold_metrics_document(
    methods: Sequence[FoldMethodMetrics],
    *,
    child_contract_sha256: str,
    parent_contract_sha256: str,
) -> dict[str, object]:
    """Build the exact evaluator ``fold_metrics.json`` document."""

    values = tuple(methods)
    if len(values) != len(_FOLD_METHOD_ORDER) or any(
        type(value) is not FoldMethodMetrics for value in values
    ):
        raise TypeError("methods must contain the seven exact fold metrics")
    for value in values:
        value.revalidate()
    observed_order = tuple(
        value.method
        if value.checkpoint_step is None
        else f"{value.method}-{value.checkpoint_step:06d}"
        for value in values
    )
    if observed_order != _FOLD_METHOD_ORDER:
        raise ValueError("fold metric methods are not in the frozen order")
    folds = {value.outer_fold for value in values}
    if len(folds) != 1:
        raise ValueError("fold metric methods must all describe one outer fold")
    child = _sha256(child_contract_sha256, label="child contract SHA-256")
    parent = _sha256(parent_contract_sha256, label="parent contract SHA-256")
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_fold_metrics",
        "child_contract_sha256": child,
        "parent_contract_sha256": parent,
        "outer_fold": folds.pop(),
        "method_order": list(_FOLD_METHOD_ORDER),
        "methods": [fold_method_record(value) for value in values],
    }
    canonical_json_bytes(document)
    return document


def equal_fold_metrics_record(value: EqualFoldMetrics) -> dict[str, object]:
    """Serialize a four-fold aggregate while retaining fold-level summaries."""

    if type(value) is not EqualFoldMetrics:
        raise TypeError("value must be an exact EqualFoldMetrics")
    value.revalidate()
    return {
        "method": value.method,
        "checkpoint_step": value.checkpoint_step,
        "overall": metric_summary_record(value.overall),
        "timestep_bins": [metric_summary_record(item) for item in value.timestep_bins],
        "folds": [
            {
                "outer_fold": fold.outer_fold,
                "overall": metric_summary_record(fold.overall),
                "timestep_bins": [metric_summary_record(item) for item in fold.timestep_bins],
            }
            for fold in value.folds
        ],
    }


def checkpoint_selection_record(value: CheckpointSelection) -> dict[str, object]:
    """Serialize the complete one-standard-error checkpoint selection."""

    if type(value) is not CheckpointSelection:
        raise TypeError("value must be an exact CheckpointSelection")
    value.revalidate()
    return {
        "checkpoint_steps": list(value.checkpoint_steps),
        "mean_nll": list(value.mean_nll),
        "best_checkpoint_step": value.best_checkpoint_step,
        "best_bootstrap_standard_error": value.best_bootstrap_standard_error,
        "eligibility_threshold": value.eligibility_threshold,
        "eligible_checkpoint_steps": list(value.eligible_checkpoint_steps),
        "selected_checkpoint_step": value.selected_checkpoint_step,
    }


def bootstrap_record(value: BootstrapPanel) -> dict[str, object]:
    """Serialize bootstrap identity and census; numeric samples remain in NPZ."""

    if type(value) is not BootstrapPanel:
        raise TypeError("value must be an exact BootstrapPanel")
    value.revalidate()
    return {
        "method_names": list(value.method_names),
        "point_mean_nll": list(value.point_mean_nll),
        "component_counts_by_fold": list(value.component_counts_by_fold),
        "replicates": value.samples.shape[1],
        "seed": value.seed,
        "parent_contract_sha256": value.parent_contract_sha256,
        "draws_sha256": value.draws_sha256,
    }


def gate_decision_record(value: PilotGateDecision) -> dict[str, object]:
    """Serialize a validated fail-closed or evidence-authorized gate result."""

    if type(value) is not PilotGateDecision:
        raise TypeError("value must be an exact PilotGateDecision")
    value.revalidate()
    checks = dict(value.checks)
    if tuple(sorted(checks)) != tuple(checks):
        checks = {key: checks[key] for key in sorted(checks)}
    return {
        "status": value.status,
        "candidate_checkpoint_step": value.candidate_checkpoint_step,
        "comparator_method": value.comparator_method,
        "point_relative_nll_improvement": value.point_relative_nll_improvement,
        "bootstrap_lower_95": value.bootstrap_lower_95,
        "bootstrap_upper_95": value.bootstrap_upper_95,
        "fold_relative_improvement": list(value.fold_relative_improvement),
        "timestep_bin_relative_improvement": list(value.timestep_bin_relative_improvement),
        "checks": checks,
        "verified_evidence_sha256": value.verified_evidence_sha256,
    }


def pilot_metrics_document(
    methods: Sequence[EqualFoldMetrics],
    *,
    checkpoint_selection: CheckpointSelection,
    comparator_method: str,
    bootstrap: BootstrapPanel,
    gate: PilotGateDecision,
    child_contract_sha256: str,
    parent_contract_sha256: str,
) -> dict[str, object]:
    """Build the exact path-free ``pilot_metrics.json`` document."""

    values = tuple(methods)
    if len(values) != len(_FOLD_METHOD_ORDER) or any(
        type(value) is not EqualFoldMetrics for value in values
    ):
        raise TypeError("methods must contain seven exact equal-fold metrics")
    observed_order = tuple(
        value.method
        if value.checkpoint_step is None
        else f"{value.method}-{value.checkpoint_step:06d}"
        for value in values
    )
    if observed_order != _FOLD_METHOD_ORDER:
        raise ValueError("pilot metric methods are not in the frozen order")
    if comparator_method not in {"C0", "C0T"}:
        raise ValueError("comparator_method must be C0 or C0T")
    for value in values:
        value.revalidate()
    checkpoint_selection.revalidate()
    bootstrap.revalidate()
    gate.revalidate()
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_pilot_metrics",
        "child_contract_sha256": _sha256(
            child_contract_sha256,
            label="child contract SHA-256",
        ),
        "parent_contract_sha256": _sha256(
            parent_contract_sha256,
            label="parent contract SHA-256",
        ),
        "method_order": list(_FOLD_METHOD_ORDER),
        "methods": [equal_fold_metrics_record(value) for value in values],
        "checkpoint_selection": checkpoint_selection_record(checkpoint_selection),
        "comparator_selection": {"method": comparator_method},
        "bootstrap": bootstrap_record(bootstrap),
        "gates": gate_decision_record(gate),
    }
    canonical_json_bytes(document)
    return document


def decision_document(
    gate: PilotGateDecision,
    *,
    child_contract_sha256: str,
    parent_contract_sha256: str,
    pilot_metrics_sha256: str,
    pilot_bootstrap_sha256: str,
) -> dict[str, object]:
    """Build the small decision artifact bound to its complete metric evidence."""

    gate.revalidate()
    record = gate_decision_record(gate)
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact": "native_categorical_diffusion_v1_r128_pilot_decision",
        "child_contract_sha256": _sha256(
            child_contract_sha256,
            label="child contract SHA-256",
        ),
        "parent_contract_sha256": _sha256(
            parent_contract_sha256,
            label="parent contract SHA-256",
        ),
        "pilot_metrics_sha256": _sha256(
            pilot_metrics_sha256,
            label="pilot metrics SHA-256",
        ),
        "pilot_bootstrap_sha256": _sha256(
            pilot_bootstrap_sha256,
            label="pilot bootstrap SHA-256",
        ),
        "decision_status": gate.status,
        "decision": record,
    }
    canonical_json_bytes(document)
    return document


def document_sha256(value: Mapping[str, object]) -> str:
    """Return the physical digest of one canonical semantic document."""

    if type(value) is not dict:
        raise TypeError("value must be a plain dict")
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    """Internal exact-float helper retained for schema extensions."""

    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    return value

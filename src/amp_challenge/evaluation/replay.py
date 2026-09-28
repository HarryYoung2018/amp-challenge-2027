"""Replay candidate-acquisition policies on strictly out-of-fold predictions.

The candidate ledger uses the same model-facing columns as ``amp-select`` and
adds revealed ``outcome_<objective>`` values plus explicit prediction-scope and
fold columns. Outcomes are kept outside ``CandidateBatch`` until after selection,
which makes accidental label-based acquisition harder and directly testable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tomllib
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from amp_challenge.acquisition import (
    CandidateBatch,
    MixedAcquisitionSelector,
    SelectionConfig,
)
from amp_challenge.workflows.select import (
    CandidateLedger,
    load_selection_config,
    read_candidate_ledger,
)

_POLICIES = ("mixed", "lcb", "mean", "random")
_METRICS = (
    "reward_mean",
    "reward_sum",
    "best_reward",
    "cumulative_regret",
    "simple_regret",
    "hit_rate_any",
    "hit_rate_all",
    "category_coverage",
    "unique_clusters",
    "mean_pairwise_cosine_distance",
)

_PAIRED_METRIC_DIRECTION = {
    "reward_mean": 1.0,
    "best_reward": 1.0,
    "hit_rate_any": 1.0,
    "hit_rate_all": 1.0,
    "category_coverage": 1.0,
    "mean_pairwise_cosine_distance": 1.0,
    "cumulative_regret": -1.0,
    "simple_regret": -1.0,
}


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Replay schema and policy choices loaded from a versioned TOML file."""

    path: Path
    selection_config_path: Path
    objectives: tuple[str, ...]
    outcome_thresholds: Mapping[str, float]
    policies: tuple[str, ...]
    seeds: tuple[int, ...]
    group_column: str
    prediction_scope_column: str
    required_prediction_scope: str
    prediction_fold_column: str
    outcome_fold_column: str


@dataclass(frozen=True, slots=True)
class ReplayExecution:
    """Completed replay and its deterministic output artifacts."""

    runs_path: Path
    selections_path: Path
    summary_path: Path
    run_count: int
    selection_count: int
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ReplayData:
    ledger: CandidateLedger
    group_by_index: tuple[str, ...]
    outcomes: np.ndarray


def _string(document: Mapping[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"replay config {name!r} must be a non-empty string")
    return value.strip()


def load_replay_config(path: str | Path) -> ReplayConfig:
    """Load and validate a replay configuration."""

    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    allowed = {
        "schema_version",
        "selection_config",
        "objectives",
        "outcome_thresholds",
        "policies",
        "seeds",
        "group_column",
        "prediction_scope_column",
        "required_prediction_scope",
        "prediction_fold_column",
        "outcome_fold_column",
    }
    unexpected = set(document) - allowed
    if unexpected:
        raise ValueError(f"unexpected replay config key(s): {sorted(unexpected)}")
    if document.get("schema_version") != 1:
        raise ValueError("replay config schema_version must be 1")

    objectives_raw = document.get("objectives")
    if (
        not isinstance(objectives_raw, list)
        or not objectives_raw
        or not all(isinstance(item, str) and item.strip() for item in objectives_raw)
    ):
        raise ValueError("replay config objectives must be a non-empty string array")
    objectives = tuple(item.strip() for item in objectives_raw)
    if len(objectives) != len(set(objectives)):
        raise ValueError("replay objectives must be unique")

    thresholds_raw = document.get("outcome_thresholds")
    if not isinstance(thresholds_raw, dict) or set(thresholds_raw) != set(objectives):
        raise ValueError("outcome_thresholds must contain exactly every replay objective")
    thresholds: dict[str, float] = {}
    for name, value in thresholds_raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"outcome threshold for {name!r} must be finite")
        thresholds[str(name)] = float(value)

    policies_raw = document.get("policies")
    if (
        not isinstance(policies_raw, list)
        or not policies_raw
        or not all(isinstance(item, str) for item in policies_raw)
    ):
        raise ValueError("replay policies must be a non-empty string array")
    policies = tuple(item.strip() for item in policies_raw)
    if len(policies) != len(set(policies)):
        raise ValueError("replay policies must be unique")
    unknown_policies = set(policies) - set(_POLICIES)
    if unknown_policies:
        raise ValueError(f"unsupported replay policies: {sorted(unknown_policies)}")

    seeds_raw = document.get("seeds")
    if (
        not isinstance(seeds_raw, list)
        or not seeds_raw
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds_raw)
    ):
        raise ValueError("replay seeds must be a non-empty integer array")
    seeds = tuple(seeds_raw)
    if len(seeds) != len(set(seeds)):
        raise ValueError("replay seeds must be unique")

    selection_relative = _string(document, "selection_config")
    selection_path = (config_path.parent / selection_relative).resolve()
    selection = load_selection_config(selection_path)
    if selection.uncertainty_mode == "unavailable" and "lcb" in policies:
        raise ValueError("replay policy 'lcb' requires uncertainty")
    columns = {
        name: _string(document, name)
        for name in (
            "group_column",
            "prediction_scope_column",
            "required_prediction_scope",
            "prediction_fold_column",
            "outcome_fold_column",
        )
    }
    declared_columns = {
        columns["group_column"],
        columns["prediction_scope_column"],
        columns["prediction_fold_column"],
        columns["outcome_fold_column"],
    }
    if len(declared_columns) != 4:
        raise ValueError("replay provenance column names must be distinct")
    return ReplayConfig(
        path=config_path,
        selection_config_path=selection_path,
        objectives=objectives,
        outcome_thresholds=thresholds,
        policies=policies,
        seeds=seeds,
        group_column=columns["group_column"],
        prediction_scope_column=columns["prediction_scope_column"],
        required_prediction_scope=columns["required_prediction_scope"],
        prediction_fold_column=columns["prediction_fold_column"],
        outcome_fold_column=columns["outcome_fold_column"],
    )


def _finite(value: str, *, column: str, row: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"ledger row {row} has invalid {column!r}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"ledger row {row} has non-finite {column!r}")
    return parsed


def _read_replay_data(
    ledger_path: Path,
    config: ReplayConfig,
    *,
    uncertainty_mode: str,
) -> _ReplayData:
    ledger = read_candidate_ledger(
        ledger_path,
        objectives=config.objectives,
        uncertainty_mode=uncertainty_mode,
    )
    required_columns = {
        config.group_column,
        config.prediction_scope_column,
        config.prediction_fold_column,
        config.outcome_fold_column,
        *(f"outcome_{name}" for name in config.objectives),
    }
    missing = required_columns - set(ledger.fieldnames)
    if missing:
        raise ValueError(f"replay ledger is missing required column(s): {sorted(missing)}")

    groups: list[str] = []
    outcomes: list[tuple[float, ...]] = []
    for metadata in ledger.metadata:
        values = metadata.as_dict()
        row = metadata.ledger_row
        group = values[config.group_column].strip()
        scope = values[config.prediction_scope_column].strip()
        prediction_fold = values[config.prediction_fold_column].strip()
        outcome_fold = values[config.outcome_fold_column].strip()
        if not group:
            raise ValueError(f"ledger row {row} has an empty replay group")
        if scope != config.required_prediction_scope:
            raise ValueError(
                f"ledger row {row} prediction scope {scope!r} is not "
                f"{config.required_prediction_scope!r}"
            )
        if not prediction_fold or not outcome_fold or prediction_fold != outcome_fold:
            raise ValueError(f"ledger row {row} lacks matching prediction/outcome fold provenance")
        groups.append(group)
        outcomes.append(
            tuple(
                _finite(
                    values[f"outcome_{name}"],
                    column=f"outcome_{name}",
                    row=row,
                )
                for name in config.objectives
            )
        )
    return _ReplayData(
        ledger=ledger,
        group_by_index=tuple(groups),
        outcomes=np.asarray(outcomes, dtype=np.float64),
    )


def _subset_candidates(candidates: CandidateBatch, indices: np.ndarray) -> CandidateBatch:
    return CandidateBatch(
        sequences=tuple(candidates.sequences[index] for index in indices),
        objective_mean=candidates.objective_mean[indices],
        objective_std=candidates.objective_std[indices],
        novelty=None if candidates.novelty is None else candidates.novelty[indices],
        embeddings=None if candidates.embeddings is None else candidates.embeddings[indices],
        cluster_ids=(
            None
            if candidates.cluster_ids is None
            else tuple(candidates.cluster_ids[index] for index in indices)
        ),
        start_ids=(
            None
            if candidates.start_ids is None
            else tuple(candidates.start_ids[index] for index in indices)
        ),
        rollout_ids=(
            None
            if candidates.rollout_ids is None
            else tuple(candidates.rollout_ids[index] for index in indices)
        ),
        specialist_scores={
            name: values[indices] for name, values in candidates.specialist_scores.items()
        },
        eligible=candidates.eligible[indices],
        uncertainty_available=candidates.uncertainty_available,
    )


def _validate_start_replay_provenance(data: _ReplayData, config: ReplayConfig) -> None:
    """Require each start lineage to live in exactly one replay group/fold."""

    starts = data.ledger.candidates.start_ids
    rollouts = data.ledger.candidates.rollout_ids
    if starts is None or rollouts is None:
        raise ValueError("start-aware replay requires start_id and rollout_id ledger columns")
    provenance_by_start: dict[str, tuple[str, str, str]] = {}
    for index, (start_id, metadata) in enumerate(zip(starts, data.ledger.metadata, strict=True)):
        values = metadata.as_dict()
        provenance = (
            data.group_by_index[index],
            values[config.prediction_fold_column].strip(),
            values[config.outcome_fold_column].strip(),
        )
        previous = provenance_by_start.setdefault(start_id, provenance)
        if previous != provenance:
            raise ValueError(
                f"start_id {start_id!r} crosses replay group/fold provenance: "
                f"{previous!r} versus {provenance!r}"
            )


def _policy_seed(seed: int, group: str, policy: str) -> int:
    payload = f"{seed}\0{group}\0{policy}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _select(
    policy: str,
    candidates: CandidateBatch,
    base: SelectionConfig,
    *,
    seed: int,
):
    if policy == "mixed":
        policy_candidates = candidates
        selection = replace(base, seed=seed)
    elif policy == "lcb":
        if base.uncertainty_mode == "unavailable" or not candidates.uncertainty_available:
            raise ValueError("replay policy 'lcb' requires uncertainty")
        policy_candidates = candidates
        selection = replace(
            base,
            strategy_mix={"exploit": 1.0},
            specialist_quotas={},
            seed=seed,
        )
    elif policy == "mean":
        policy_candidates = candidates
        selection = replace(
            base,
            strategy_mix={"exploit": 1.0},
            specialist_quotas={},
            risk_beta=0.0,
            seed=seed,
        )
    elif policy == "random":
        # Zero model values make the selector's random arm a genuinely random
        # priority while preserving identical eligibility and cluster caps.
        policy_candidates = CandidateBatch(
            sequences=candidates.sequences,
            objective_mean=np.zeros_like(candidates.objective_mean),
            objective_std=np.zeros_like(candidates.objective_std),
            cluster_ids=candidates.cluster_ids,
            start_ids=candidates.start_ids,
            rollout_ids=candidates.rollout_ids,
            eligible=candidates.eligible,
            uncertainty_available=candidates.uncertainty_available,
        )
        selection = replace(
            base,
            strategy_mix={"random": 1.0},
            specialist_quotas={},
            risk_beta=0.0,
            quality_floor_quantile=0.0,
            seed=seed,
        )
    else:  # pragma: no cover - config validation guards this
        raise ValueError(f"unsupported policy: {policy}")
    return MixedAcquisitionSelector(selection).select(policy_candidates)


def _mean_pairwise_cosine_distance(embeddings: np.ndarray | None) -> float | None:
    if embeddings is None or len(embeddings) < 2:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings, dtype=np.float64),
        where=norms > 0,
    )
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    upper = np.triu_indices(len(embeddings), k=1)
    return float(np.mean(1.0 - similarities[upper]))


def _metric_row(
    *,
    policy: str,
    group: str,
    seed: int,
    candidates: CandidateBatch,
    outcomes: np.ndarray,
    selected: Sequence[int],
    reasons: Sequence[str],
    thresholds: np.ndarray,
    objective_weights: np.ndarray,
    max_per_cluster: int | None,
    strict_cluster_cap: bool,
    max_per_start: int | None,
) -> dict[str, object]:
    eligible = np.asarray(candidates.eligible, dtype=bool)
    scalar = outcomes @ objective_weights
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_scalar = scalar[selected_array]
    optimum = _oracle_optimum_rewards(
        scalar,
        eligible=eligible,
        cluster_ids=candidates.cluster_ids,
        max_per_cluster=max_per_cluster if strict_cluster_cap else None,
        start_ids=candidates.start_ids,
        max_per_start=max_per_start,
        batch_size=len(selected_array),
    )
    selected_outcomes = outcomes[selected_array]
    hits = selected_outcomes >= thresholds[None, :]
    embeddings = None if candidates.embeddings is None else candidates.embeddings[selected_array]
    clusters = (
        None
        if candidates.cluster_ids is None
        else {candidates.cluster_ids[index] for index in selected_array}
    )
    row: dict[str, object] = {
        "policy": policy,
        "group": group,
        "seed": seed,
        "candidate_count": len(candidates.sequences),
        "eligible_count": int(np.count_nonzero(eligible)),
        "batch_size": len(selected_array),
        "reward_mean": float(np.mean(selected_scalar)),
        "reward_sum": float(np.sum(selected_scalar)),
        "best_reward": float(np.max(selected_scalar)),
        "cumulative_regret": float(max(0.0, np.sum(optimum) - np.sum(selected_scalar))),
        "simple_regret": float(max(0.0, np.max(scalar[eligible]) - np.max(selected_scalar))),
        "hit_rate_any": float(np.mean(np.any(hits, axis=1))),
        "hit_rate_all": float(np.mean(np.all(hits, axis=1))),
        "category_coverage": float(np.mean(np.any(hits, axis=0))),
        "unique_clusters": None if clusters is None else len(clusters),
        "mean_pairwise_cosine_distance": _mean_pairwise_cosine_distance(embeddings),
        "strategy_counts": json.dumps(dict(sorted(Counter(reasons).items())), sort_keys=True),
    }
    for column in range(outcomes.shape[1]):
        row[f"mean_outcome_{column}"] = float(np.mean(selected_outcomes[:, column]))
        row[f"hit_count_{column}"] = int(np.count_nonzero(hits[:, column]))
    return row


def _oracle_optimum_rewards(
    rewards: np.ndarray,
    *,
    eligible: np.ndarray,
    cluster_ids: Sequence[str] | None,
    max_per_cluster: int | None,
    start_ids: Sequence[str] | None,
    max_per_start: int | None,
    batch_size: int,
) -> np.ndarray:
    """Return the exact maximum-reward batch under hard cluster/start caps."""

    if max_per_cluster is not None and cluster_ids is None:
        raise ValueError("max_per_cluster requires cluster IDs for replay regret")
    if max_per_start is not None and start_ids is None:
        raise ValueError("max_per_start requires start IDs for replay regret")
    if max_per_cluster is not None and max_per_start is not None:
        assert cluster_ids is not None
        assert start_ids is not None
        return _intersection_cap_optimum_rewards(
            rewards,
            eligible=eligible,
            cluster_ids=cluster_ids,
            max_per_cluster=max_per_cluster,
            start_ids=start_ids,
            max_per_start=max_per_start,
            batch_size=batch_size,
        )

    order = sorted(
        (index for index in range(len(rewards)) if eligible[index]),
        key=lambda index: (-rewards[index], index),
    )
    selected: list[float] = []
    cluster_counts: Counter[str] = Counter()
    start_counts: Counter[str] = Counter()
    for index in order:
        if cluster_ids is not None and max_per_cluster is not None:
            cluster = cluster_ids[index]
            if cluster_counts[cluster] >= max_per_cluster:
                continue
        if start_ids is not None and max_per_start is not None:
            start = start_ids[index]
            if start_counts[start] >= max_per_start:
                continue
        if cluster_ids is not None and max_per_cluster is not None:
            cluster_counts[cluster_ids[index]] += 1
        if start_ids is not None and max_per_start is not None:
            start_counts[start_ids[index]] += 1
        selected.append(float(rewards[index]))
        if len(selected) == batch_size:
            return np.asarray(selected, dtype=np.float64)
    raise ValueError("hard group constraints make the replay batch infeasible")


def _intersection_cap_optimum_rewards(
    rewards: np.ndarray,
    *,
    eligible: np.ndarray,
    cluster_ids: Sequence[str],
    max_per_cluster: int,
    start_ids: Sequence[str],
    max_per_start: int,
    batch_size: int,
) -> np.ndarray:
    """Solve the two-partition-cap reward optimum as a zero-gap binary MILP."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except ImportError as error:  # pragma: no cover - required project dependency
        raise RuntimeError("SciPy/HiGHS is required for intersecting replay group caps") from error

    eligible_indices = np.asarray(np.flatnonzero(eligible), dtype=np.int64)
    if len(eligible_indices) < batch_size:
        raise ValueError("hard group constraints make the replay batch infeasible")
    cluster_names = tuple(sorted({cluster_ids[index] for index in eligible_indices}))
    start_names = tuple(sorted({start_ids[index] for index in eligible_indices}))
    cluster_row = {name: offset + 1 for offset, name in enumerate(cluster_names)}
    start_row = {name: offset + 1 + len(cluster_names) for offset, name in enumerate(start_names)}
    row_count = 1 + len(cluster_names) + len(start_names)
    rows: list[int] = []
    columns: list[int] = []
    for local_index, global_index in enumerate(eligible_indices):
        rows.extend((0, cluster_row[cluster_ids[global_index]], start_row[start_ids[global_index]]))
        columns.extend((local_index, local_index, local_index))
    constraint_matrix = coo_array(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(row_count, len(eligible_indices)),
    ).tocsr()
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.empty(row_count, dtype=np.float64)
    lower[0] = float(batch_size)
    upper[0] = float(batch_size)
    upper[1 : 1 + len(cluster_names)] = float(max_per_cluster)
    upper[1 + len(cluster_names) :] = float(max_per_start)
    options: dict[str, object] = {
        "presolve": True,
        "mip_rel_gap": 0.0,
        "threads": 1,
        "parallel": False,
        "random_seed": 0,
        "mip_feasibility_tolerance": 1e-9,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*",
            category=RuntimeWarning,
        )
        solution = milp(
            c=-np.asarray(rewards[eligible_indices], dtype=np.float64),
            integrality=np.ones(len(eligible_indices), dtype=np.uint8),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(constraint_matrix, lb=lower, ub=upper),
            options=options,
        )
    mip_gap = getattr(solution, "mip_gap", None)
    if (
        not solution.success
        or solution.status != 0
        or solution.x is None
        or mip_gap is None
        or not math.isfinite(float(mip_gap))
        or float(mip_gap) > 1e-12
    ):
        raise RuntimeError(
            "replay regret oracle did not certify an optimal zero-gap solution: "
            f"status={solution.status} message={solution.message!s} mip_gap={mip_gap}"
        )
    rounded = np.rint(solution.x)
    if np.any(np.abs(solution.x - rounded) > 1e-7):
        raise RuntimeError("replay regret oracle returned a fractional solution")
    chosen_local = np.flatnonzero(rounded.astype(bool))
    if len(chosen_local) != batch_size:
        raise RuntimeError("replay regret oracle returned the wrong batch size")
    chosen = eligible_indices[chosen_local]
    if any(
        count > max_per_cluster
        for count in Counter(cluster_ids[index] for index in chosen).values()
    ):
        raise RuntimeError("replay regret oracle violated the cluster cap")
    if any(
        count > max_per_start for count in Counter(start_ids[index] for index in chosen).values()
    ):
        raise RuntimeError("replay regret oracle violated the start cap")
    selected_total = math.fsum(float(rewards[index]) for index in chosen)
    if solution.fun is None or not math.isclose(
        selected_total,
        -float(solution.fun),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise RuntimeError("replay regret oracle objective is inconsistent")
    ordered = sorted(chosen, key=lambda index: (-rewards[index], int(index)))
    return np.asarray([rewards[index] for index in ordered], dtype=np.float64)


def _objective_weights(config: SelectionConfig, n_objectives: int) -> np.ndarray:
    if config.objective_weights is None:
        return np.full(n_objectives, 1.0 / n_objectives, dtype=np.float64)
    weights = np.asarray(config.objective_weights, dtype=np.float64)
    if (
        weights.shape != (n_objectives,)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0)
        or np.sum(weights) <= 0
    ):
        raise ValueError("selection objective_weights must be non-negative and match objectives")
    return weights / np.sum(weights)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_runs(
    rows: Sequence[Mapping[str, object]], objectives: Sequence[str]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    numeric_metrics = (*_METRICS, *(f"mean_outcome_{name}" for name in objectives))
    policies = sorted({str(row["policy"]) for row in rows})
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy]
        metrics: dict[str, object] = {"evaluations": len(subset)}
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            metrics[metric] = (
                None
                if not values
                else {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
        metrics["mean_outcomes"] = {
            name: metrics.pop(f"mean_outcome_{name}") for name in objectives
        }
        summary[policy] = metrics
    return summary


def _paired_mixed_improvements(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare policies on group means so seed repeats are not pseudo-replicates."""

    policies = sorted({str(row["policy"]) for row in rows})
    if "mixed" not in policies:
        return {}
    groups = sorted({str(row["group"]) for row in rows})
    group_means: dict[tuple[str, str, str], float] = {}
    for policy in policies:
        for group in groups:
            subset = [row for row in rows if row["policy"] == policy and row["group"] == group]
            for metric in _PAIRED_METRIC_DIRECTION:
                values = [float(row[metric]) for row in subset if row.get(metric) is not None]
                if values:
                    group_means[(policy, group, metric)] = float(np.mean(values))

    comparisons: dict[str, object] = {}
    for baseline in (policy for policy in policies if policy != "mixed"):
        metrics: dict[str, object] = {}
        for metric, direction in _PAIRED_METRIC_DIRECTION.items():
            improvements = {
                group: direction
                * (group_means[("mixed", group, metric)] - group_means[(baseline, group, metric)])
                for group in groups
                if ("mixed", group, metric) in group_means
                and (baseline, group, metric) in group_means
            }
            values = list(improvements.values())
            if not values:
                continue
            tolerance = 1e-12
            metrics[metric] = {
                "positive_means_mixed_is_better": True,
                "mean_improvement": float(np.mean(values)),
                "median_improvement": float(np.median(values)),
                "improved_groups": sum(value > tolerance for value in values),
                "tied_groups": sum(abs(value) <= tolerance for value in values),
                "worsened_groups": sum(value < -tolerance for value in values),
                "groups": len(values),
                "by_group": dict(sorted(improvements.items())),
            }
        comparisons[f"mixed_vs_{baseline}"] = metrics
    return comparisons


def run_replay(
    ledger_path: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> ReplayExecution:
    """Run all configured policies without exposing outcomes during selection."""

    input_path = Path(ledger_path).resolve()
    config = load_replay_config(config_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    base = load_selection_config(config.selection_config_path)
    data = _read_replay_data(
        input_path,
        config,
        uncertainty_mode=base.uncertainty_mode,
    )
    start_aware = base.rollouts_per_start is not None
    if start_aware:
        _validate_start_replay_provenance(data, config)
    thresholds = np.asarray(
        [config.outcome_thresholds[name] for name in config.objectives], dtype=np.float64
    )
    objective_weights = _objective_weights(base, len(config.objectives))
    groups = tuple(sorted(set(data.group_by_index)))
    run_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for group in groups:
        global_indices = np.asarray(
            [index for index, value in enumerate(data.group_by_index) if value == group],
            dtype=np.int64,
        )
        candidates = _subset_candidates(data.ledger.candidates, global_indices)
        outcomes = data.outcomes[global_indices]
        if np.count_nonzero(candidates.eligible) < base.batch_size:
            raise ValueError(
                f"replay group {group!r} has fewer eligible candidates than batch_size"
            )
        for configured_seed in config.seeds:
            for policy in config.policies:
                actual_seed = _policy_seed(configured_seed, group, policy)
                result = _select(policy, candidates, base, seed=actual_seed)
                run_row = _metric_row(
                    policy=policy,
                    group=group,
                    seed=configured_seed,
                    candidates=candidates,
                    outcomes=outcomes,
                    selected=result.indices,
                    reasons=result.reasons,
                    thresholds=thresholds,
                    objective_weights=objective_weights,
                    max_per_cluster=base.max_per_cluster,
                    strict_cluster_cap=base.strict_cluster_cap,
                    max_per_start=base.max_per_start,
                )
                for column, name in enumerate(config.objectives):
                    run_row[f"mean_outcome_{name}"] = run_row.pop(f"mean_outcome_{column}")
                    run_row[f"hit_count_{name}"] = run_row.pop(f"hit_count_{column}")
                run_rows.append(run_row)

                for rank, local_index in enumerate(result.indices, start=1):
                    global_index = int(global_indices[local_index])
                    selected_outcomes = data.outcomes[global_index]
                    hit_vector = selected_outcomes >= thresholds
                    row: dict[str, object] = {
                        "policy": policy,
                        "group": group,
                        "seed": configured_seed,
                        "rank": rank,
                        "sequence_id": data.ledger.metadata[global_index].sequence_id,
                        "sequence": data.ledger.candidates.sequences[global_index],
                        "reason": result.reasons[rank - 1],
                        "conservative_score": result.conservative_scores[rank - 1],
                        "acquisition_score": result.acquisition_scores[rank - 1],
                        "scalar_outcome": float(selected_outcomes @ objective_weights),
                        "hit_any": bool(np.any(hit_vector)),
                        "hit_all": bool(np.all(hit_vector)),
                        "cluster_id": (
                            None
                            if data.ledger.candidates.cluster_ids is None
                            else data.ledger.candidates.cluster_ids[global_index]
                        ),
                    }
                    if start_aware:
                        assert data.ledger.candidates.start_ids is not None
                        assert data.ledger.candidates.rollout_ids is not None
                        row.update(
                            {
                                "start_id": data.ledger.candidates.start_ids[global_index],
                                "rollout_id": data.ledger.candidates.rollout_ids[global_index],
                            }
                        )
                        if result.start_evidence:
                            evidence = result.start_evidence[rank - 1]
                            row.update(
                                {
                                    "start_rank": evidence.start_rank,
                                    "start_eligible_rollout_count": (
                                        evidence.eligible_rollout_count
                                    ),
                                    "start_rollout_value_mean": evidence.rollout_value_mean,
                                    "start_rollout_value_dispersion": (
                                        evidence.rollout_value_dispersion
                                    ),
                                    "rollout_ucb_score": evidence.rollout_ucb_score,
                                }
                            )
                    for column, name in enumerate(config.objectives):
                        row[f"outcome_{name}"] = float(selected_outcomes[column])
                        row[f"hit_{name}"] = bool(hit_vector[column])
                    selection_rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.csv"
    selections_path = output / "selections.csv"
    summary_path = output / "summary.json"
    run_fields = [
        "policy",
        "group",
        "seed",
        "candidate_count",
        "eligible_count",
        "batch_size",
        *_METRICS,
        *(f"mean_outcome_{name}" for name in config.objectives),
        *(f"hit_count_{name}" for name in config.objectives),
        "strategy_counts",
    ]
    selection_fields = [
        "policy",
        "group",
        "seed",
        "rank",
        "sequence_id",
        "sequence",
        "reason",
        "conservative_score",
        "acquisition_score",
        "scalar_outcome",
        "hit_any",
        "hit_all",
        "cluster_id",
        *(
            (
                "start_id",
                "rollout_id",
                "start_rank",
                "start_eligible_rollout_count",
                "start_rollout_value_mean",
                "start_rollout_value_dispersion",
                "rollout_ucb_score",
            )
            if start_aware
            else ()
        ),
        *(f"outcome_{name}" for name in config.objectives),
        *(f"hit_{name}" for name in config.objectives),
    ]
    _write_csv(runs_path, run_fields, run_rows)
    _write_csv(selections_path, selection_fields, selection_rows)
    summary: dict[str, object] = {
        "schema_version": 2 if start_aware else 1,
        "warning": (
            "Replay validity depends on the ledger's predictions genuinely being produced "
            "without fitting the corresponding outcome fold."
        ),
        "ledger_sha256": _sha256(input_path),
        "replay_config_sha256": _sha256(config.path),
        "selection_config_sha256": _sha256(config.selection_config_path),
        "objectives": list(config.objectives),
        "policies": list(config.policies),
        "seeds": list(config.seeds),
        "groups": list(groups),
        "run_count": len(run_rows),
        "selection_count": len(selection_rows),
        "runs_sha256": _sha256(runs_path),
        "selections_sha256": _sha256(selections_path),
        "policy_metrics": _aggregate_runs(run_rows, config.objectives),
        "paired_group_improvements": _paired_mixed_improvements(run_rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReplayExecution(
        runs_path=runs_path,
        selections_path=selections_path,
        summary_path=summary_path,
        run_count=len(run_rows),
        selection_count=len(selection_rows),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = run_replay(
        args.ledger,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "runs": execution.run_count,
                "selections": execution.selection_count,
                "summary": str(execution.summary_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

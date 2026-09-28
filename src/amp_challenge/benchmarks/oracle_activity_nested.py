"""Deterministic, component-balanced nested activity validation primitives.

Binary MIC16 prediction only. This module carries no search, safety or production
authority. Fixed sequence features are supplied after oracle namespace filtering.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ARMS = ("descriptors_context", "esm_context", "spectral_context", "esm_spectral_context")
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
SEEDS = (17, 42, 91, 137, 271)
SOLVER = {"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12, "maxls": 50}
EPS = 1e-12


def component_weights(rows: list[dict]) -> np.ndarray:
    """w_i = 1 / (G n_g): each observed training union has total weight 1/G."""
    if not rows:
        raise ValueError("cannot weight an empty context set")
    counts = Counter(row["union_component_id"] for row in rows)
    return np.asarray([1.0 / (len(counts) * counts[row["union_component_id"]]) for row in rows])


def label_array(rows: list[dict]) -> np.ndarray:
    if not rows or any(type(row["label"]) is not int or row["label"] not in (0, 1) for row in rows):
        raise ValueError("activity labels must be nonempty binary integers")
    return np.asarray([row["label"] for row in rows], dtype=np.float64)


def group_folds(sequences: list[dict], count: int = 3) -> dict[str, int]:
    """Inner folds use all outer-training oracle sequences, never their labels."""
    groups = Counter(row["union_component_id"] for row in sequences)
    if type(count) is not int or count < 2 or len(groups) < count:
        raise ValueError("insufficient whole unions for inner folds")
    loads = [0] * count
    result = {}
    for uid in sorted(groups, key=lambda key: (-groups[key], key)):
        fold = min(range(count), key=lambda index: (loads[index], index))
        result[uid] = fold
        loads[fold] += groups[uid]
    return result


def _context(row: dict) -> tuple[str, str]:
    return row["canonical_target"], row["gram"]


def _prior(rows: list[dict]) -> float:
    """Jeffreys smoothing of group-average labels: (sum_g mean(y_g)+.5)/(G+1)."""
    weight = component_weights(rows)
    groups = len({row["union_component_id"] for row in rows})
    return float((groups * np.dot(weight, label_array(rows)) + 0.5) / (groups + 1))


def proper_scores(rows: list[dict], probabilities: np.ndarray) -> dict[str, float]:
    y = label_array(rows)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probability vector")
    clipped = np.clip(p, EPS, 1 - EPS)
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    brier = (p - y) ** 2
    weights = component_weights(rows)
    return {
        "component_log_loss": float(np.dot(weights, losses)),
        "component_brier": float(np.dot(weights, brier)),
        "row_log_loss": float(losses.mean()),
        "row_brier": float(brier.mean()),
    }


@dataclass
class ActivityModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    targets: tuple[str, ...]
    grams: tuple[str, ...]
    train_rows: list[dict]
    penalty: float
    iterations: int
    objective: float | None
    constant_training_labels: bool

    def document(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "targets": self.targets,
            "grams": self.grams,
            "penalty": self.penalty,
            "iterations": self.iterations,
            "objective": self.objective,
            "constant_training_labels": self.constant_training_labels,
            "training_example_ids": sorted(row["example_id"] for row in self.train_rows),
            "training_union_ids": sorted({row["union_component_id"] for row in self.train_rows}),
        }


def _design(x: np.ndarray, rows: list[dict], mean, scale, targets, grams) -> np.ndarray:
    if (
        x.ndim != 2
        or x.shape[0] != len(rows)
        or x.shape[1] != len(mean)
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid or nonfinite feature matrix")
    categorical = np.asarray(
        [
            [float(row["canonical_target"] == value) for value in targets]
            + [float(row["gram"] == value) for value in grams]
            for row in rows
        ],
        dtype=np.float64,
    ).reshape(len(rows), len(targets) + len(grams))
    return np.column_stack((np.ones(len(rows)), (x - mean) / scale, categorical))


def fit_activity(x: np.ndarray, rows: list[dict], penalty: float) -> ActivityModel:
    """Minimize sum_i w_i logistic_loss_i + lambda/2 ||beta_without_intercept||²."""
    x = np.asarray(x, dtype=np.float64)
    y = label_array(rows)
    if (
        x.ndim != 2
        or x.shape[0] != len(rows)
        or not 1 <= x.shape[1] <= 353
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid bounded feature matrix")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    weight = component_weights(rows)
    mean = weight @ x
    scale = np.sqrt(weight @ ((x - mean) ** 2))
    # A mathematically constant column remains constant even when weighted
    # summation rounds its mean by one ulp.
    scale = np.where((scale > 0) & np.any(x != x[0], axis=0), scale, 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("nonfinite training transform")
    targets = tuple(sorted({row["canonical_target"] for row in rows}))
    grams = tuple(sorted({row["gram"] for row in rows}))
    design = _design(x, rows, mean, scale, targets, grams)
    beta = np.zeros(design.shape[1])
    constant = len(set(y)) == 1
    iterations, objective = 0, None
    if not constant:

        def loss_and_gradient(value):
            logits = design @ value
            loss = np.dot(weight, np.logaddexp(0, logits) - y * logits)
            gradient = design.T @ (weight * (expit(logits) - y))
            loss += penalty * np.dot(value[1:], value[1:]) / 2
            gradient[1:] += penalty * value[1:]
            return float(loss), gradient

        result = minimize(loss_and_gradient, beta, jac=True, method="L-BFGS-B", options=SOLVER)
        if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise ValueError(f"logistic solver failed without retry: {result.message}")
        beta, iterations, objective = result.x, int(result.nit), float(result.fun)
    return ActivityModel(
        mean, scale, beta, targets, grams, rows, penalty, iterations, objective, constant
    )


def predict_activity(
    model: ActivityModel, x: np.ndarray, rows: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    design = _design(
        np.asarray(x, dtype=np.float64), rows, model.mean, model.scale, model.targets, model.grams
    )
    probabilities = expit(design @ model.coefficients)
    details = []
    for index, row in enumerate(rows):
        context_rows = [r for r in model.train_rows if _context(r) == _context(row)]
        groups = len({r["union_component_id"] for r in context_rows})
        classes = {r["label"] for r in context_rows}
        reason = None
        if model.constant_training_labels:
            reason = "constant_training_labels"
        elif not context_rows:
            reason = "unseen_target_gram_context"
        elif groups < 2:
            reason = "context_fewer_than_two_training_unions"
        elif len(classes) < 2:
            reason = "constant_context_labels"
        # The same train-only diagnostic prior is retained for every row,
        # including supported contexts where the fitted prediction is used.
        gram_rows = [r for r in model.train_rows if r["gram"] == row["gram"]]
        selected, level = (
            (context_rows, "target_gram")
            if context_rows
            else (gram_rows, "gram")
            if gram_rows
            else (model.train_rows, "global")
        )
        prior_probability = _prior(selected)
        if reason:
            probabilities[index] = prior_probability
        details.append(
            {
                "abstained": reason is not None,
                "reason": reason,
                "fallback_prior_level": level if reason else None,
                "training_context_unions": groups,
                "training_context_classes": sorted(classes),
                "diagnostic_prior_probability": prior_probability,
                "diagnostic_prior_level": level,
                "diagnostic_prior_training_unions": len(
                    {r["union_component_id"] for r in selected}
                ),
            }
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite predictions")
    return probabilities, details


def nested_validation(
    sequences: list[dict], rows: list[dict], matrices: dict[str, np.ndarray], penalties=LAMBDAS
) -> dict[str, Any]:
    """Outer held-out labels/features never select transforms, penalties or calibration."""
    if (
        not 1 <= len(penalties) <= 4
        or len(set(penalties)) != len(penalties)
        or any(not np.isfinite(v) or v <= 0 for v in penalties)
    ):
        raise ValueError("penalty grid exceeds fixed fit budget")
    if not 1 <= len(sequences) <= 415 or not 1 <= len(rows) <= 858:
        raise ValueError("oracle data exceeds declared row budget")
    if set(matrices) != set(ARMS) or any(len(value) != len(rows) for value in matrices.values()):
        raise ValueError("arm matrices must align with context rows")
    label_array(rows)
    sequence_by_id = {row["sequence_id"]: row for row in sequences}
    if len(sequence_by_id) != len(sequences):
        raise ValueError("duplicate oracle sequence identity")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate activity example identity")
    union_fold: dict[str, set[int]] = defaultdict(set)
    for row in sequences:
        union_fold[row["union_component_id"]].add(row["oof_fold"])
    if any(len(folds) != 1 for folds in union_fold.values()) or {
        r["oof_fold"] for r in sequences
    } != set(range(5)):
        raise ValueError("oracle unions must remain whole across five outer folds")
    for row in rows:
        sequence = sequence_by_id[row["sequence_id"]]
        if (
            row["oof_fold"] != sequence["oof_fold"]
            or row["union_component_id"] != sequence["union_component_id"]
        ):
            raise ValueError("activity identity differs from frozen oracle fold")
    predictions, models, inner_predictions, assignments = [], [], [], []
    fit_attempts = 0
    for outer in range(5):
        train_indices = [i for i, row in enumerate(rows) if row["oof_fold"] != outer]
        test_indices = [i for i, row in enumerate(rows) if row["oof_fold"] == outer]
        if not train_indices or not test_indices:
            raise ValueError("outer activity fold is empty")
        inner_map = group_folds([row for row in sequences if row["oof_fold"] != outer])
        assignments.append({"outer_fold": outer, "inner_union_folds": inner_map})
        train_rows, test_rows = [rows[i] for i in train_indices], [rows[i] for i in test_indices]
        for arm in ARMS:
            scores = []
            for penalty in penalties:
                inner_rows, inner_p = [], []
                for inner in range(3):
                    fit_indices = [
                        i
                        for i in train_indices
                        if inner_map[rows[i]["union_component_id"]] != inner
                    ]
                    validation_indices = [
                        i
                        for i in train_indices
                        if inner_map[rows[i]["union_component_id"]] == inner
                    ]
                    if not fit_indices or not validation_indices:
                        raise ValueError("inner activity fold is empty")
                    fit_rows = [rows[i] for i in fit_indices]
                    validation_rows = [rows[i] for i in validation_indices]
                    fit_attempts += 1
                    model = fit_activity(matrices[arm][fit_indices], fit_rows, float(penalty))
                    p, details = predict_activity(
                        model, matrices[arm][validation_indices], validation_rows
                    )
                    inner_rows.extend(validation_rows)
                    inner_p.extend(p.tolist())
                    inner_predictions.extend(
                        {
                            "arm": arm,
                            "outer_fold": outer,
                            "inner_fold": inner,
                            "penalty": penalty,
                            "example_id": row["example_id"],
                            "probability": float(value),
                            **detail,
                        }
                        for row, value, detail in zip(validation_rows, p, details, strict=True)
                    )
                scores.append(
                    {"penalty": penalty, **proper_scores(inner_rows, np.asarray(inner_p))}
                )
            chosen = min(scores, key=lambda item: (item["component_log_loss"], -item["penalty"]))[
                "penalty"
            ]
            fit_attempts += 1
            model = fit_activity(matrices[arm][train_indices], train_rows, float(chosen))
            p, details = predict_activity(model, matrices[arm][test_indices], test_rows)
            models.append(
                {"arm": arm, "outer_fold": outer, "inner_selection": scores, **model.document()}
            )
            predictions.extend(
                {"arm": arm, **row, "probability": float(value), **detail}
                for row, value, detail in zip(test_rows, p, details, strict=True)
            )
    expected = len(ARMS) * 5 * (len(penalties) * 3 + 1)
    if fit_attempts != expected or expected > 260:
        raise ValueError("fit budget invariant differs")
    return {
        "predictions": predictions,
        "models": models,
        "inner_predictions": inner_predictions,
        "inner_assignments": assignments,
        "fit_attempts": fit_attempts,
    }


def _group_losses(rows: list[dict]) -> dict[str, tuple[float, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["union_component_id"]].append(row)
    return {
        uid: (score["component_log_loss"], score["component_brier"])
        for uid, subset in sorted(groups.items())
        for score in [proper_scores(subset, np.asarray([r["probability"] for r in subset]))]
    }


def _descriptive_subsets(selected: list[dict], dominant: str, *, prior=False) -> dict:
    subsets = {
        "all": selected,
        "dominant_union_holdout": [r for r in selected if r["union_component_id"] == dominant],
        "excluding_dominant_union": [r for r in selected if r["union_component_id"] != dominant],
    }
    for key in ("canonical_target", "gram", "oof_fold"):
        for value in sorted({row[key] for row in selected}):
            subsets[f"{key}:{value}"] = [r for r in selected if r[key] == value]
    result = {}
    for name, subset in subsets.items():
        if not subset:
            result[name] = None
            continue
        support = (
            {
                "prior_level_counts": dict(
                    sorted(Counter(r["diagnostic_prior_level"] for r in subset).items())
                ),
                "training_prior_union_support_counts": dict(
                    sorted(
                        Counter(str(r["diagnostic_prior_training_unions"]) for r in subset).items()
                    )
                ),
                "training_context_union_support_counts": dict(
                    sorted(Counter(str(r["training_context_unions"]) for r in subset).items())
                ),
            }
            if prior
            else {
                "abstentions": sum(r["abstained"] for r in subset),
                "fallback_reasons": dict(
                    sorted(Counter(r["reason"] for r in subset if r["abstained"]).items())
                ),
            }
        )
        result[name] = {
            "contexts": len(subset),
            "sequences": len({r["sequence_id"] for r in subset}),
            "unions": len({r["union_component_id"] for r in subset}),
            "positive": sum(r["label"] for r in subset),
            "negative": sum(r["label"] == 0 for r in subset),
            **support,
            **proper_scores(subset, np.asarray([r["probability"] for r in subset])),
        }
    return result


def summarize_predictions(predictions: list[dict], sequences: list[dict], replicates=2000) -> dict:
    """Paired OOF group bootstrap, conditional on fitted predictors; not retraining uncertainty."""
    if type(replicates) is not int or not 100 <= replicates <= 2000:
        raise ValueError("bootstrap replicate budget differs")
    dominant = min(
        Counter(row["union_component_id"] for row in sequences),
        key=lambda uid: (-sum(r["union_component_id"] == uid for r in sequences), uid),
    )
    arms = {}
    group_losses = {}
    prior_identities = None
    prior_rows = []
    for arm in ARMS:
        selected = [row for row in predictions if row["arm"] == arm]
        group_losses[arm] = _group_losses(selected)
        identities = {
            row["example_id"]: (
                row["diagnostic_prior_probability"],
                row["diagnostic_prior_level"],
                row["diagnostic_prior_training_unions"],
                row["training_context_unions"],
            )
            for row in selected
        }
        if len(identities) != len(selected):
            raise ValueError("diagnostic prior requires unique example identities per arm")
        if prior_identities is None:
            prior_identities = identities
            prior_rows = [
                {**row, "probability": row["diagnostic_prior_probability"]} for row in selected
            ]
        elif identities != prior_identities:
            raise ValueError("train-only diagnostic prior must be identical across matched arms")
        arms[arm] = _descriptive_subsets(selected, dominant)
    baseline = group_losses[ARMS[0]]
    ids = sorted(baseline)
    comparisons = {}
    for arm in ARMS[1:]:
        if set(group_losses[arm]) != set(ids):
            raise ValueError("paired comparisons require identical observed unions")
        delta = np.asarray([np.subtract(group_losses[arm][uid], baseline[uid]) for uid in ids])
        checks = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
            values = delta[draws].mean(axis=1)
            checks.append(
                {
                    "seed": seed,
                    "log_loss_bonferroni_95_family_interval": np.quantile(
                        values[:, 0], [0.05 / 6, 1 - 0.05 / 6]
                    ).tolist(),
                    "brier_descriptive_95_interval": np.quantile(
                        values[:, 1], [0.025, 0.975]
                    ).tolist(),
                }
            )
        comparisons[arm] = {
            "difference_direction": "arm_minus_descriptors_lower_is_better",
            "observed_unions": len(ids),
            "log_loss_difference": float(delta[:, 0].mean()),
            "brier_difference": float(delta[:, 1].mean()),
            "primary_seed": 17,
            "bootstrap_results": checks,
        }
    return {
        "arms": arms,
        "diagnostic_prior_baseline": {
            "scope": "train_only_smoothed_context_hierarchy_descriptive_not_primary_comparison",
            "optimizer_fits": 0,
            "primary_comparison_family_member": False,
            "scores": _descriptive_subsets(prior_rows, dominant, prior=True),
        },
        "paired_comparisons": comparisons,
        "dominant_union_id": dominant,
        "dominant_union_sequences": sum(r["union_component_id"] == dominant for r in sequences),
        "uncertainty_scope": "conditional_OOF_predictor_whole_union_resampling_not_retraining_variance",
        "primary_family": "three_log_loss_differences_against_descriptors_context",
        "bootstrap_replicates_per_seed": replicates,
        "calibration_fitted": False,
        "binary_label_not_continuous_MIC": True,
        "safety_model_fitted": False,
        "production_input_eligible": False,
        "scientific_superiority_accepted": False,
    }

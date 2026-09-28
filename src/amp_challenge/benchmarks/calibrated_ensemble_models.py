"""Bounded weighted logistic models, train-only OOD geometry and calibration.

These are controller-private study primitives, not an adaptive learner or a
free per-candidate oracle interface. No weights depend on a held-out outcome.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

from amp_challenge.benchmarks.calibrated_ensemble_contract import SOLVER, require


def restricted_weights(rows, group_draw):
    counts = Counter(row["union_component_id"] for row in rows)
    require(bool(counts) and set(counts) <= set(group_draw), "missing training group draw")
    masses = np.asarray([group_draw[group] for group in counts], dtype=np.float64)
    require(
        np.isfinite(masses).all() and np.all(masses > 0),
        "group draws must be strictly positive and finite",
    )
    total = float(masses.sum())
    require(np.isfinite(total) and total > 0, "invalid total training group mass")
    return np.asarray(
        [
            group_draw[row["union_component_id"]] / total / counts[row["union_component_id"]]
            for row in rows
        ]
    )


def _transform(x, weight):
    mean = weight @ x
    scale = np.sqrt(weight @ ((x - mean) ** 2))
    scale = np.where((scale > 0) & np.any(x != x[0], axis=0), scale, 1.0)
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), "nonfinite training transform")
    return mean, scale


@dataclass(frozen=True)
class WeightedModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    targets: tuple[str, ...]
    grams: tuple[str, ...]
    constant_training_labels: bool
    iterations: int
    objective: float | None

    def design(self, x, rows):
        x = np.asarray(x, dtype=np.float64)
        require(
            x.shape == (len(rows), len(self.mean)) and np.isfinite(x).all(),
            "unaligned bounded prediction features",
        )
        categories = np.asarray(
            [
                [float(row["canonical_target"] == target) for target in self.targets]
                + [float(row["gram"] == gram) for gram in self.grams]
                for row in rows
            ]
        ).reshape(len(rows), len(self.targets) + len(self.grams))
        return np.column_stack((np.ones(len(rows)), (x - self.mean) / self.scale, categories))

    def predict(self, x, rows):
        probability = expit(self.design(x, rows) @ self.coefficients)
        require(np.isfinite(probability).all(), "nonfinite raw model probability")
        return probability


def fit_weighted_model(x, rows, group_draw):
    x = np.asarray(x, dtype=np.float64)
    require(
        x.ndim == 2
        and x.shape[0] == len(rows)
        and 1 <= len(rows) <= 858
        and 1 <= x.shape[1] <= 353
        and np.isfinite(x).all(),
        "invalid bounded training features",
    )
    require(
        all(type(row["label"]) is int and row["label"] in (0, 1) for row in rows),
        "nonbinary training labels",
    )
    y = np.asarray([row["label"] for row in rows], dtype=np.float64)
    weight = restricted_weights(rows, group_draw)
    mean, scale = _transform(x, weight)
    targets = tuple(sorted({row["canonical_target"] for row in rows}))
    grams = tuple(sorted({row["gram"] for row in rows}))
    beta = np.zeros(1 + x.shape[1] + len(targets) + len(grams))
    constant = len(set(y)) == 1
    shell = WeightedModel(mean, scale, beta, targets, grams, constant, 0, None)
    if constant:
        return shell
    design = shell.design(x, rows)

    def objective(value):
        logits = design @ value
        loss = (
            np.dot(weight, np.logaddexp(0, logits) - y * logits) + np.dot(value[1:], value[1:]) / 2
        )
        gradient = design.T @ (weight * (expit(logits) - y))
        gradient[1:] += value[1:]
        return float(loss), gradient

    result = minimize(objective, beta, jac=True, method="L-BFGS-B", options=SOLVER)
    require(
        result.success and np.isfinite(result.x).all() and np.isfinite(result.fun),
        f"weighted base solver failed without retry: {result.message}",
    )
    return WeightedModel(
        mean, scale, result.x, targets, grams, False, int(result.nit), float(result.fun)
    )


@dataclass(frozen=True)
class Calibrator:
    a: float
    b: float
    fitted: bool
    eligible_unions: int
    negative_unions: int
    positive_unions: int
    iterations: int
    objective: float | None

    def predict(self, probability):
        probability = np.asarray(probability, dtype=np.float64)
        require(
            np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
            "invalid calibration input probability",
        )
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        if not self.fitted:
            # An identity fallback leaves the original probability unchanged.
            return np.asarray(probability, dtype=np.float64).copy()
        return expit(self.a * (np.log(clipped) - np.log1p(-clipped)) + self.b)


def fit_calibrator(probabilities, rows, eligible, group_draw):
    eligible = np.asarray(eligible)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(
        eligible.dtype == bool and eligible.shape == probabilities.shape == (len(rows),),
        "calibration mask/row alignment differs",
    )
    require(
        np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)),
        "invalid calibration probability",
    )
    chosen = [row for row, selected in zip(rows, eligible, strict=True) if selected]
    require(
        all(type(row["label"]) is int and row["label"] in (0, 1) for row in chosen),
        "nonbinary eligible calibration label",
    )
    groups = {row["union_component_id"] for row in chosen}
    labels = [
        {row["union_component_id"] for row in chosen if row["label"] == value} for value in (0, 1)
    ]
    counts = (len(groups), len(labels[0]), len(labels[1]))
    if counts[0] < 10 or min(counts[1:]) < 3:
        return Calibrator(1.0, 0.0, False, *counts, 0, None)
    weight = restricted_weights(chosen, group_draw)
    clipped = np.clip(probabilities[eligible], 1e-6, 1 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    design = np.column_stack((logits, np.ones(len(chosen))))
    y = np.asarray([row["label"] for row in chosen], dtype=np.float64)
    center = np.asarray([1.0, 0.0])

    def objective(value):
        score = design @ value
        difference = value - center
        loss = np.dot(weight, np.logaddexp(0, score) - y * score) + 0.05 * np.dot(
            difference, difference
        )
        gradient = design.T @ (weight * (expit(score) - y)) + 0.1 * difference
        return float(loss), gradient

    result = minimize(
        objective,
        center,
        jac=True,
        method="L-BFGS-B",
        bounds=((0.05, 5.0), (-5.0, 5.0)),
        options=SOLVER,
    )
    require(
        result.success and np.isfinite(result.x).all() and np.isfinite(result.fun),
        f"calibration solver failed without retry: {result.message}",
    )
    return Calibrator(
        float(result.x[0]), float(result.x[1]), True, *counts, int(result.nit), float(result.fun)
    )


def _distances(first, second):
    # Direct coordinate differences avoid cancellation at exactly equal points.
    return np.sqrt(cdist(first, second, metric="sqeuclidean") / first.shape[1])


@dataclass(frozen=True)
class OODGeometry:
    mean: np.ndarray
    scale: np.ndarray
    standardized_training: np.ndarray
    threshold: float
    training_reference_distances: np.ndarray

    def distances(self, x):
        values = np.asarray(x, dtype=np.float64)
        require(
            values.ndim == 2 and values.shape[1] == len(self.mean) and np.isfinite(values).all(),
            "OOD query feature alignment differs",
        )
        return _distances((values - self.mean) / self.scale, self.standardized_training).min(axis=1)


def fit_ood_geometry(x, sequences):
    x = np.asarray(x, dtype=np.float64)
    require(
        x.ndim == 2
        and x.shape[0] == len(sequences)
        and 1 <= len(sequences) <= 415
        and x.shape[1] == 321
        and np.isfinite(x).all(),
        "invalid OOD training features",
    )
    groups = [row["union_component_id"] for row in sequences]
    require(len(set(groups)) >= 2, "OOD geometry requires another training union")
    weight = restricted_weights(sequences, dict.fromkeys(groups, 1.0))
    mean, scale = _transform(x, weight)
    standardized = (x - mean) / scale
    distance = _distances(standardized, standardized)
    same_union = np.equal(np.asarray(groups)[:, None], np.asarray(groups)[None, :])
    distance[same_union] = np.inf
    nearest = distance.min(axis=1)
    require(np.isfinite(nearest).all(), "nonfinite other-union OOD reference distance")
    order = np.argsort(nearest, kind="stable")
    cumulative = np.cumsum(weight[order])
    quantile_index = min(int(np.searchsorted(cumulative, 0.95, side="left")), len(order) - 1)
    return OODGeometry(mean, scale, standardized, float(nearest[order[quantile_index]]), nearest)

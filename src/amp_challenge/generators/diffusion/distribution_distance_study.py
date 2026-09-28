"""Exact finite categorical transition diagnostics and explicit update gates.

Scope is the supplied categorical distributions, NOT full generated peptides or
unobserved neural-model contexts. KL uses natural logarithms. Wasserstein uses
an explicitly supplied normalized ground metric; equal numerical thresholds
across different metrics do not mean equal constraints. Objective ratio clipping
is separate from the exact total-variation acceptance gate.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeWarning, linprog
from scipy.special import rel_entr

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Neutral free amino-acid elemental composition, ordered C, H, N, O, S.
# A common peptide-bond subtraction does not change pairwise feature differences.
_COMPOSITION = np.asarray(
    [
        [3, 7, 1, 2, 0],
        [3, 7, 1, 2, 1],
        [4, 7, 1, 4, 0],
        [5, 9, 1, 4, 0],
        [9, 11, 1, 2, 0],
        [2, 5, 1, 2, 0],
        [6, 9, 3, 2, 0],
        [6, 13, 1, 2, 0],
        [6, 14, 2, 2, 0],
        [6, 13, 1, 2, 0],
        [5, 11, 1, 2, 1],
        [4, 8, 2, 3, 0],
        [5, 9, 1, 2, 0],
        [5, 10, 2, 3, 0],
        [6, 14, 4, 2, 0],
        [3, 7, 1, 3, 0],
        [4, 9, 1, 3, 0],
        [5, 11, 1, 2, 0],
        [11, 12, 2, 2, 0],
        [9, 11, 1, 3, 0],
    ],
    dtype=float,
)


def amino_acid_ground_cost(identity_weight: float = 0.05) -> NDArray[np.float64]:
    """Declared exploratory chemistry metric, not a validated activity metric.

    Each elemental count is range-scaled across the fixed canonical alphabet;
    Euclidean distances are divided by their diameter. A convex mixture with
    identity/Hamming distance separates composition isomers (isoleucine/leucine).
    The mixing weight must be frozen before comparison, not tuned on test reward.
    At weight=1, Wasserstein collapses exactly to total variation.
    """
    if not np.isfinite(identity_weight) or not 0 < identity_weight <= 1:
        raise ValueError("identity_weight must lie in (0, 1]")
    features = _COMPOSITION / np.ptp(_COMPOSITION, axis=0)
    distances = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
    distances /= distances.max()
    return (1 - identity_weight) * distances + identity_weight * (1 - np.eye(20))


def _probabilities(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim < 1 or array.shape[-1] < 2 or array.size == 0:
        raise ValueError("probabilities require nonempty rows with at least two categories")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(array.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to one; no implicit normalization")
    return array


def _pair(old: ArrayLike, new: ArrayLike):
    old, new = _probabilities(old), _probabilities(new)
    if old.shape != new.shape:
        raise ValueError("old and new must have identical shapes; no implicit broadcasting")
    return old, new


def _ground_cost(value: ArrayLike, size: int) -> NDArray[np.float64]:
    cost = np.asarray(value, dtype=float)
    if cost.shape != (size, size) or not np.isfinite(cost).all():
        raise ValueError("ground metric must be finite with shape (categories, categories)")
    if np.any(cost < 0) or np.any(cost > 1 + 1e-12):
        raise ValueError("ground metric must have declared normalized scale [0, 1]")
    if not np.allclose(cost, cost.T, rtol=0, atol=1e-12):
        raise ValueError("ground metric must be symmetric")
    if not np.allclose(np.diag(cost), 0, rtol=0, atol=1e-12):
        raise ValueError("ground metric diagonal must be zero")
    if np.any(cost[~np.eye(size, dtype=bool)] <= 0):
        raise ValueError("distinct categories must have positive distance")
    if np.any(cost[:, :, None] > cost[:, None, :] + cost.T[None, :, :] + 1e-12):
        raise ValueError("ground metric violates the triangle inequality")
    return cost


def categorical_distance_diagnostics(
    old: ArrayLike, new: ArrayLike, ground_cost: ArrayLike
) -> dict[str, NDArray[np.float64]]:
    """Compute all diagnostics once; final category axis, arbitrary batch axes.

    Transport is solved without entropic regularization using linear programming.
    Marginal feasibility is independently checked at absolute tolerance 1e-8.
    Zero support is retained: KL can legitimately be positive infinity.
    """
    old, new = _pair(old, new)
    size = old.shape[-1]
    cost = _ground_cost(ground_cost, size)
    midpoint = (old + new) / 2
    result = {
        "kl": np.sum(rel_entr(new, old), axis=-1),
        "reverse_kl": np.sum(rel_entr(old, new), axis=-1),
        "total_variation": np.sum(np.abs(new - old), axis=-1) / 2,
        "jensen_shannon": (
            np.sum(rel_entr(new, midpoint), axis=-1) + np.sum(rel_entr(old, midpoint), axis=-1)
        )
        / 2,
    }
    constraints = np.concatenate(
        [
            np.kron(np.eye(size), np.ones((1, size))),
            np.tile(np.eye(size), (1, size)),
        ]
    )
    distances = []
    for before, after in zip(old.reshape(-1, size), new.reshape(-1, size), strict=True):
        if np.array_equal(before, after):
            distances.append(0.0)
            continue
        target = np.concatenate([before, after])
        # HiGHS shares a process-wide scheduler with the portfolio/replay MILPs.
        # Match their single-thread contract regardless of which solver runs first.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unrecognized options detected.*", category=OptimizeWarning
            )
            solution = linprog(
                cost.ravel(),
                A_eq=constraints,
                b_eq=target,
                bounds=(0, None),
                method="highs",
                options={"primal_feasibility_tolerance": 1e-9, "threads": 1, "parallel": False},
            )
        if not solution.success:
            raise RuntimeError(f"optimal transport failed: {solution.message}")
        if np.max(np.abs(constraints @ solution.x - target)) > 1e-8:
            raise RuntimeError("optimal transport failed independent marginal check")
        distances.append(float(solution.fun))
    result["wasserstein_1"] = np.asarray(distances).reshape(old.shape[:-1])
    return result


@dataclass(frozen=True)
class UpdateDecision:
    accepted: bool
    metric: str
    metric_limit: float
    total_variation_limit: float
    diagnostics: dict[str, NDArray[np.float64]]
    scope: str = "each supplied categorical transition; not full peptide distribution"


def accept_categorical_update(
    old: ArrayLike,
    new: ArrayLike,
    ground_cost: ArrayLike,
    *,
    metric: str,
    metric_limit: float,
    tv_limit: float = 0.05,
) -> UpdateDecision:
    """Accept only if EVERY supplied row passes both independently named gates."""
    if metric not in {"kl", "reverse_kl", "jensen_shannon", "wasserstein_1", "total_variation"}:
        raise ValueError("unknown comparison metric")
    if not np.isfinite(metric_limit) or metric_limit < 0:
        raise ValueError("metric_limit must be finite and nonnegative")
    if not np.isfinite(tv_limit) or not 0 <= tv_limit <= 0.05:
        raise ValueError("tv_limit must be in [0, .05] for this study")
    diagnostics = categorical_distance_diagnostics(old, new, ground_cost)
    accepted = bool(
        np.all(diagnostics["total_variation"] <= tv_limit)
        and np.all(diagnostics[metric] <= metric_limit)
    )
    return UpdateDecision(accepted, metric, metric_limit, tv_limit, diagnostics)


def clip_probability_ratios(old: ArrayLike, new: ArrayLike, width: float = 0.05):
    """Return clipped ratios for objective use, NOT a normalized updated policy.

    Undefined 0/0 entries are set to one; new positive mass at old zero support
    is rejected because it cannot be estimated by old-policy importance sampling.
    """
    old, new = _pair(old, new)
    if not np.isfinite(width) or not 0 <= width < 1:
        raise ValueError("clipping width must be in [0, 1)")
    if np.any((old == 0) & (new > 0)):
        raise ValueError("new probability outside old support has no finite ratio")
    ratios = np.divide(new, old, out=np.ones_like(old), where=old > 0)
    return np.clip(ratios, 1 - width, 1 + width)


def backtrack_categorical_update(
    old: ArrayLike,
    proposed: ArrayLike,
    ground_cost: ArrayLike,
    *,
    metric: str,
    metric_limit: float,
    tv_limit: float = 0.05,
    max_backtracks: int = 20,
) -> tuple[NDArray[np.float64], float, UpdateDecision]:
    """Return an accepted categorical mixture, or the unchanged old policy.

    This is an executable distribution mixture, NOT interpolation or acceptance
    of neural parameters. A caller must actually sample this mixture or separately
    recompute/check probabilities after its own parameter update.
    """
    old, proposed = _pair(old, proposed)
    if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, int):
        raise ValueError("max_backtracks must be a nonnegative integer")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    for step in range(max_backtracks + 1):
        fraction = 2.0**-step
        candidate = (1 - fraction) * old + fraction * proposed
        decision = accept_categorical_update(
            old,
            candidate,
            ground_cost,
            metric=metric,
            metric_limit=metric_limit,
            tv_limit=tv_limit,
        )
        if decision.accepted:
            return candidate, fraction, decision
    decision = accept_categorical_update(
        old, old, ground_cost, metric=metric, metric_limit=metric_limit, tv_limit=tv_limit
    )
    return old.copy(), 0.0, decision

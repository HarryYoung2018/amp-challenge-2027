"""Terminal posterior-mean decisions, kept separate from acquisition.

Knowledge Gradient chooses which point to *measure*.  This module implements
the different fixed-action decision made after those measurements: recommend
the feasible real decision with the largest posterior-mean utility when it
beats the declared outside option, and otherwise abstain.
Posterior variance enters only through declared chance constraints; it never
acts as an exploration bonus or information-value score.  The current seam
accepts exactly one preference row.  Supporting an unresolved multi-row
preference measure requires a shared terminal-timing contract with KG because
``max_z E_w[u_w(z)]`` and ``E_w[max_z u_w(z)]`` are different decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from amp_challenge.acquisition.soft_kg import SoftKGProblem
from amp_challenge.models.posterior import JointGaussianPosterior

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _binary_fraction(value: float | np.floating) -> Fraction:
    """Return the exact rational represented by a binary floating scalar."""

    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _checked_weighted_reductions(
    values: FloatArray,
    weights: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Evaluate terminal weighted sums exactly before one float64 rounding.

    Terminal recommendation is a small, once-per-run decision boundary.  Exact
    rational accumulation avoids BLAS-dependent cancellation and intermediate
    reduction order without changing the represented float64 inputs.  An exact
    zero is canonically rounded to positive zero, so signed-zero inputs cannot
    create a platform-dependent ranking distinction.
    """

    rows = np.asarray(values, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if rows.ndim != 2 or weight_values.shape != (rows.shape[1],):
        raise ValueError(f"{name} dimensions do not match")
    if np.any(~np.isfinite(rows)) or np.any(~np.isfinite(weight_values)):
        raise ValueError(f"{name} requires finite operands")

    result = np.empty(rows.shape[0], dtype=np.float64)
    for row_index, row in enumerate(rows):
        exact = Fraction(0)
        for value, weight in zip(row, weight_values, strict=True):
            exact += _binary_fraction(value) * _binary_fraction(weight)
        try:
            rounded = float(exact)
        except OverflowError as error:
            raise FloatingPointError(f"{name} is not representable in float64") from error
        if not np.isfinite(rounded) or (exact != 0 and rounded == 0.0):
            raise FloatingPointError(f"{name} is not representable in float64")
        result[row_index] = rounded
    return result


def _decision_indices(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    raw_values = tuple(values)
    if not raw_values:
        raise ValueError(f"{name} cannot be empty")
    parsed: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{name} must contain integers")
        index = int(value)
        if index < 0:
            raise ValueError(f"{name} must be non-negative")
        parsed.append(index)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must be unique")
    return tuple(parsed)


def _readonly_float_array(value: object, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_bool_array(value: object, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must be boolean")
    array = np.array(raw, dtype=bool, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PosteriorMeanRecommendation:
    """Auditable hard ranking of explicitly recommendable fixed actions.

    Array entries follow ``decision_indices``.  Ties in
    ``expected_posterior_mean_utility`` retain that caller-declared order.
    Synthetic always-safe outside actions cannot appear in the real ranking or
    be exported. Their best posterior-mean value is retained solely to decide
    whether the terminal action should be an explicit abstention.
    """

    decision_indices: tuple[int, ...]
    expected_posterior_mean_utility: FloatArray
    chance_feasible: BoolArray
    constraint_satisfaction_probability: FloatArray
    outside_option_decision_indices: tuple[int, ...] = ()
    outside_option_posterior_mean_utility: float | None = None

    def __post_init__(self) -> None:
        decisions = _decision_indices(self.decision_indices, name="decision_indices")
        utility = _readonly_float_array(
            self.expected_posterior_mean_utility,
            name="expected_posterior_mean_utility",
        )
        feasible = _readonly_bool_array(self.chance_feasible, name="chance_feasible")
        probability = _readonly_float_array(
            self.constraint_satisfaction_probability,
            name="constraint_satisfaction_probability",
        )
        raw_outside_options = tuple(self.outside_option_decision_indices)
        outside_options = (
            ()
            if not raw_outside_options
            else _decision_indices(
                raw_outside_options,
                name="outside_option_decision_indices",
            )
        )
        if set(decisions) & set(outside_options):
            raise ValueError("real and outside-option decision indices must be disjoint")
        outside_value = self.outside_option_posterior_mean_utility
        if (not outside_options) != (outside_value is None):
            raise ValueError(
                "outside-option indices and utility must either both be present or absent"
            )
        if outside_value is not None and not np.isfinite(outside_value):
            raise ValueError("outside_option_posterior_mean_utility must be finite")
        expected_vector_shape = (len(decisions),)
        if utility.shape != expected_vector_shape:
            raise ValueError("expected_posterior_mean_utility must match decision_indices")
        if feasible.shape != expected_vector_shape:
            raise ValueError("chance_feasible must match decision_indices")
        if probability.ndim != 2 or probability.shape[0] != len(decisions):
            raise ValueError("constraint_satisfaction_probability must have one row per decision")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("constraint satisfaction probabilities must lie in [0, 1]")
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "expected_posterior_mean_utility", utility)
        object.__setattr__(self, "chance_feasible", feasible)
        object.__setattr__(self, "constraint_satisfaction_probability", probability)
        object.__setattr__(self, "outside_option_decision_indices", outside_options)
        object.__setattr__(
            self,
            "outside_option_posterior_mean_utility",
            None if outside_value is None else float(outside_value),
        )

    @property
    def ranked_decision_indices(self) -> tuple[int, ...]:
        """Return feasible decisions in descending mean-utility order."""

        feasible_positions = np.flatnonzero(self.chance_feasible)
        order = feasible_positions[
            np.argsort(
                -self.expected_posterior_mean_utility[feasible_positions],
                kind="stable",
            )
        ]
        return tuple(self.decision_indices[int(position)] for position in order)

    @property
    def selected_decision_index(self) -> int | None:
        """Return the best real decision, or ``None`` when the system abstains."""

        ranked = self.ranked_decision_indices
        if not ranked:
            return None
        best_real = ranked[0]
        best_real_position = self.decision_indices.index(best_real)
        best_real_utility = float(self.expected_posterior_mean_utility[best_real_position])
        if (
            self.outside_option_posterior_mean_utility is not None
            and best_real_utility <= self.outside_option_posterior_mean_utility
        ):
            return None
        return best_real

    @property
    def selected_posterior_mean_utility(self) -> float | None:
        """Return the emitted real decision's mean utility, if not abstaining."""

        selected = self.selected_decision_index
        if selected is None:
            return None
        position = self.decision_indices.index(selected)
        return float(self.expected_posterior_mean_utility[position])

    @property
    def terminal_posterior_mean_utility(self) -> float | None:
        """Return the value of the emitted real action or explicit abstention."""

        selected_utility = self.selected_posterior_mean_utility
        if selected_utility is not None:
            return selected_utility
        return self.outside_option_posterior_mean_utility

    @property
    def abstention_reason(self) -> str | None:
        """Explain why no real decision was emitted."""

        if self.selected_decision_index is not None:
            return None
        if not self.ranked_decision_indices:
            return "no_chance_feasible_real_decision"
        if self.outside_option_posterior_mean_utility is not None:
            return "outside_option_weakly_dominates"
        return "no_chance_feasible_real_decision"


def recommend_posterior_mean(
    belief: JointGaussianPosterior,
    problem: SoftKGProblem,
    *,
    recommendable_indices: Sequence[int],
) -> PosteriorMeanRecommendation:
    """Hard-rank real decisions by expected posterior-mean utility.

    ``recommendable_indices`` is intentionally explicit and must be a subset
    of the problem's frozen decision set.  The caller therefore defines the
    real exportable set and cannot accidentally receive a synthetic no-action
    decision used only to make constrained KG well-defined. The outside option
    still participates by value: if it weakly dominates the best safe real
    decision, the result emits ``None`` as an abstention rather than its index.

    The current interface accepts exactly one preference row.  This fail-closed
    rule prevents silently mixing KG's preference-contingent ``E_w[max_z]``
    value with a fixed pre-resolution ``max_z E_w`` recommendation.  A future
    multi-preference implementation must declare one timing for both seams.
    The hard ranking does not use KG, UCB, posterior variance,
    softmax/logsumexp over decisions, or ``problem.base_measure``.  Variance is
    consulted only to enforce ``problem.constraints``.
    """

    decisions = _decision_indices(
        recommendable_indices,
        name="recommendable_indices",
    )
    problem_decisions = set(problem.decision_indices)
    if not set(decisions) <= problem_decisions:
        raise ValueError("recommendable_indices must be a subset of problem.decision_indices")
    outside_options = tuple(problem.always_safe_decisions)
    if set(decisions) & set(outside_options):
        raise ValueError("recommendable_indices cannot include always-safe outside decisions")
    if problem.preferences.weights.shape[0] != 1:
        raise ValueError(
            "posterior-mean recommendation requires exactly one preference row until "
            "terminal preference timing is shared with KG"
        )
    if max(problem.decision_indices) >= belief.n_points:
        raise ValueError("problem decision_indices must refer to posterior points")
    if max(problem.objective_outputs) >= belief.n_outputs:
        raise ValueError("problem objective_outputs must refer to posterior outputs")
    if any(constraint.output_index >= belief.n_outputs for constraint in problem.constraints):
        raise ValueError("problem constraints must refer to posterior outputs")

    point_positions = np.asarray(decisions, dtype=np.intp)
    objective_positions = np.asarray(problem.objective_outputs, dtype=np.intp)
    objective_mean = belief.mean[np.ix_(point_positions, objective_positions)]
    expected_utility = _checked_weighted_reductions(
        objective_mean,
        problem.preferences.weights[0],
        name="expected posterior-mean utility",
    )

    n_decisions = len(decisions)
    n_constraints = len(problem.constraints)
    satisfaction = np.empty((n_decisions, n_constraints), dtype=np.float64)
    feasible = np.ones(n_decisions, dtype=bool)
    for column, constraint in enumerate(problem.constraints):
        output = constraint.output_index
        output_mean = belief.mean[point_positions, output]
        output_variance = np.maximum(
            belief.covariance[point_positions, output, point_positions, output],
            0.0,
        )
        satisfaction[:, column] = constraint.satisfaction_probability(
            output_mean,
            output_variance,
        )
        feasible &= constraint.is_satisfied(output_mean, output_variance)

    outside_value: float | None = None
    if outside_options:
        outside_positions = np.asarray(outside_options, dtype=np.intp)
        outside_objective_mean = belief.mean[np.ix_(outside_positions, objective_positions)]
        outside_utility = _checked_weighted_reductions(
            outside_objective_mean,
            problem.preferences.weights[0],
            name="outside-option posterior-mean utility",
        )
        outside_value = float(np.max(outside_utility))

    return PosteriorMeanRecommendation(
        decision_indices=decisions,
        expected_posterior_mean_utility=expected_utility,
        chance_feasible=feasible,
        constraint_satisfaction_probability=satisfaction,
        outside_option_decision_indices=outside_options,
        outside_option_posterior_mean_utility=outside_value,
    )

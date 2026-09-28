"""Pure counterfactual-contrast mathematics for evolutionary search.

The functions in this module operate on model-relative parent/child contrasts.
They do not turn an in-silico comparison into a biological causal effect.  The
module intentionally depends only on NumPy so its covariance and gating
contracts can be smoke-tested without fitting a surrogate model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Kernel = Callable[[FloatArray, FloatArray], ArrayLike]
Scalarizer = Callable[[FloatArray, object], ArrayLike]


@dataclass(frozen=True, slots=True)
class ContrastMoments:
    """Posterior moments of a scalarized child-minus-parent contrast."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("contrast mean must be finite")
        if not np.isfinite(self.variance) or self.variance < 0:
            raise ValueError("contrast variance must be finite and non-negative")

    @property
    def standard_deviation(self) -> float:
        """Return the posterior standard deviation of the contrast."""

        return float(np.sqrt(self.variance))


@dataclass(frozen=True, slots=True)
class AdvantageGateResult:
    """Auditable decision about which children may enter a positive update."""

    advantages: tuple[float, ...]
    accepted: tuple[bool, ...]

    @property
    def update_enabled(self) -> bool:
        """Whether at least one child passed every relative and absolute gate."""

        return any(self.accepted)

    @property
    def accepted_indices(self) -> tuple[int, ...]:
        """Indices of children admitted to the positive endpoint set."""

        return tuple(index for index, accepted in enumerate(self.accepted) if accepted)


@dataclass(frozen=True, slots=True)
class HistoricalInfluenceCredit:
    """Immutable audit record for model-relative historical influence.

    These values summarize how compatible each observed marginal is with
    posterior samples of the optimum proxy, then propagate that attribution
    through a caller-supplied similarity measure.  They are posterior influence
    scores, not causal effects of collecting or removing an observation.
    """

    log_predictive_evidence: FloatArray
    rank_fraction: FloatArray
    observed_credit: FloatArray
    normalized_similarity: FloatArray
    similarity_mass: FloatArray
    propagated_credit: FloatArray
    decayed_weight: FloatArray
    decay_exponent: float

    def __post_init__(self) -> None:
        arrays = {
            "log_predictive_evidence": np.array(
                self.log_predictive_evidence, dtype=np.float64, copy=True
            ),
            "rank_fraction": np.array(self.rank_fraction, dtype=np.float64, copy=True),
            "observed_credit": np.array(self.observed_credit, dtype=np.float64, copy=True),
            "normalized_similarity": np.array(
                self.normalized_similarity, dtype=np.float64, copy=True
            ),
            "similarity_mass": np.array(self.similarity_mass, dtype=np.float64, copy=True),
            "propagated_credit": np.array(self.propagated_credit, dtype=np.float64, copy=True),
            "decayed_weight": np.array(self.decayed_weight, dtype=np.float64, copy=True),
        }
        observed = arrays["log_predictive_evidence"]
        if observed.ndim != 1 or observed.size == 0:
            raise ValueError("log_predictive_evidence must be a non-empty vector")
        for name in ("rank_fraction", "observed_credit"):
            if arrays[name].shape != observed.shape:
                raise ValueError(f"{name} must match log_predictive_evidence")

        similarity = arrays["normalized_similarity"]
        if similarity.ndim != 2 or similarity.shape[1] != observed.size or similarity.shape[0] == 0:
            raise ValueError("normalized_similarity must have shape (n_targets, n_observations)")
        targets = similarity.shape[0]
        for name in ("similarity_mass", "propagated_credit", "decayed_weight"):
            if arrays[name].shape != (targets,):
                raise ValueError(f"{name} must have one value per similarity target")
        if any(np.any(~np.isfinite(array)) for array in arrays.values()):
            raise ValueError("historical influence audit arrays must be finite")
        if np.any((arrays["rank_fraction"] < 0) | (arrays["rank_fraction"] > 1)):
            raise ValueError("rank_fraction must lie in [0, 1]")
        if np.any(arrays["observed_credit"] <= 0):
            raise ValueError("observed_credit must be strictly positive")
        if np.any(similarity < 0) or np.any(arrays["similarity_mass"] < 0):
            raise ValueError("similarity values and masses must be non-negative")
        nonempty = arrays["similarity_mass"] > 0
        row_sums = np.sum(similarity, axis=1)
        if not np.allclose(row_sums[nonempty], 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("non-empty normalized similarity rows must sum to one")
        if not np.allclose(row_sums[~nonempty], 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("zero-mass normalized similarity rows must remain zero")
        if np.any(arrays["propagated_credit"] <= 0) or np.any(arrays["decayed_weight"] <= 0):
            raise ValueError("propagated credits and weights must be strictly positive")
        if not np.isfinite(self.decay_exponent) or self.decay_exponent < 0:
            raise ValueError("decay_exponent must be finite and non-negative")

        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "decay_exponent", float(self.decay_exponent))


def _finite_points(values: ArrayLike, *, name: str) -> FloatArray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional point matrix")
    if np.any(~np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _kernel_matrix(
    kernel: Kernel,
    left: FloatArray,
    right: FloatArray,
    *,
    name: str,
) -> FloatArray:
    values = np.asarray(kernel(left, right), dtype=np.float64)
    expected = (left.shape[0], right.shape[0])
    if values.shape != expected:
        raise ValueError(f"{name} must return a kernel matrix with shape {expected}")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} returned non-finite kernel values")
    return values


def gp_difference_kernel(
    children: ArrayLike,
    parents: ArrayLike,
    *,
    kernel: Kernel,
    other_children: ArrayLike | None = None,
    other_parents: ArrayLike | None = None,
) -> FloatArray:
    """Evaluate the GP kernel between child-minus-parent edge observations.

    For edges ``(x_plus, p)`` and ``(x_plus_prime, p_prime)``, the returned
    covariance is ``k(x_plus, x_plus_prime) - k(x_plus, p_prime) -
    k(p, x_plus_prime) + k(p, p_prime)``.  Omitting the ``other_*`` arguments
    returns the square covariance matrix for one collection of edges.
    """

    left_children = _finite_points(children, name="children")
    left_parents = _finite_points(parents, name="parents")
    if left_children.shape != left_parents.shape:
        raise ValueError("children and parents must have identical shapes")
    if (other_children is None) != (other_parents is None):
        raise ValueError("other_children and other_parents must be supplied together")

    if other_children is None:
        right_children = left_children
        right_parents = left_parents
    else:
        right_children = _finite_points(other_children, name="other_children")
        right_parents = _finite_points(other_parents, name="other_parents")
        if right_children.shape != right_parents.shape:
            raise ValueError("other_children and other_parents must have identical shapes")
        if right_children.shape[1] != left_children.shape[1]:
            raise ValueError("left and right edge endpoints must share a feature dimension")

    child_child = _kernel_matrix(
        kernel,
        left_children,
        right_children,
        name="kernel",
    )
    child_parent = _kernel_matrix(
        kernel,
        left_children,
        right_parents,
        name="kernel",
    )
    parent_child = _kernel_matrix(
        kernel,
        left_parents,
        right_children,
        name="kernel",
    )
    parent_parent = _kernel_matrix(
        kernel,
        left_parents,
        right_parents,
        name="kernel",
    )
    difference = child_child - child_parent - parent_child + parent_parent
    if np.any(~np.isfinite(difference)):
        raise ValueError("difference kernel produced non-finite values")
    return np.asarray(difference, dtype=np.float64)


def observed_contrast_noise_variance(
    child_variance: ArrayLike,
    parent_variance: ArrayLike,
    covariance: ArrayLike = 0.0,
) -> FloatArray:
    """Return assay-noise variance for an observed child-parent contrast.

    Inputs follow NumPy broadcasting.  ``covariance`` captures shared plate,
    batch, or replicate noise and is therefore subtracted twice.  Its magnitude
    is checked against the two marginal variances so the supplied 2x2 noise
    covariance is positive semidefinite at every broadcast position.
    """

    try:
        child, parent, shared = np.broadcast_arrays(
            np.asarray(child_variance, dtype=np.float64),
            np.asarray(parent_variance, dtype=np.float64),
            np.asarray(covariance, dtype=np.float64),
        )
    except ValueError as error:
        raise ValueError("contrast-noise inputs must have broadcast-compatible shapes") from error

    if np.any(~np.isfinite(child)) or np.any(~np.isfinite(parent)):
        raise ValueError("child and parent noise variances must be finite")
    if np.any(child < 0) or np.any(parent < 0):
        raise ValueError("child and parent noise variances must be non-negative")
    if np.any(~np.isfinite(shared)):
        raise ValueError("child-parent noise covariance must be finite")

    geometric_limit = np.sqrt(child) * np.sqrt(parent)
    nonzero_limit = geometric_limit > 0.0
    covariance_ratio = np.zeros_like(geometric_limit)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(
            np.abs(shared),
            geometric_limit,
            out=covariance_ratio,
            where=nonzero_limit,
        )
    relative_tolerance = 64.0 * np.finfo(np.float64).eps
    incompatible = (~nonzero_limit & (shared != 0.0)) | (
        nonzero_limit & (covariance_ratio > 1.0 + relative_tolerance)
    )
    if np.any(incompatible):
        raise ValueError("noise covariance is incompatible with the marginal variances")

    arithmetic_scale = np.maximum(np.maximum(child, parent), np.abs(shared))
    safe_scale = np.where(arithmetic_scale > 0.0, arithmetic_scale, 1.0)
    child_scaled = child / safe_scale
    parent_scaled = parent / safe_scale
    shared_scaled = shared / safe_scale
    scaled_tolerance = 64.0 * np.finfo(np.float64).eps
    child_root = np.sqrt(child_scaled)
    parent_root = np.sqrt(parent_scaled)
    root_product = child_root * parent_root
    scaled_variance = (child_root - parent_root) ** 2 + 2.0 * (root_product - shared_scaled)
    if np.any(scaled_variance < -scaled_tolerance):
        raise ValueError("observed contrast noise variance cannot be negative")
    variance = np.maximum(scaled_variance, 0.0) * arithmetic_scale
    if np.any(~np.isfinite(variance)):
        raise ValueError("observed contrast noise variance is not representable")
    return np.asarray(variance, dtype=np.float64)


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validated_joint_covariance(values: ArrayLike, *, objectives: int) -> FloatArray:
    covariance = np.asarray(values, dtype=np.float64)
    expected = (2 * objectives, 2 * objectives)
    if covariance.shape != expected:
        raise ValueError(f"joint_covariance must have shape {expected}")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("joint_covariance must contain only finite values")

    scale = float(np.max(np.abs(covariance)))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * covariance.shape[0]
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=tolerance):
        raise ValueError("joint_covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2.0
    if float(np.min(np.linalg.eigvalsh(covariance))) < -tolerance:
        raise ValueError("joint_covariance must be positive semidefinite")
    return covariance


def linear_paired_contrast(
    child_mean: ArrayLike,
    parent_mean: ArrayLike,
    joint_covariance: ArrayLike,
    weights: ArrayLike,
) -> ContrastMoments:
    """Return analytic moments for a linear, context-specific contrast.

    ``joint_covariance`` orders variables as all child objectives followed by
    all parent objectives.  This preserves both cross-objective covariance and
    child-parent covariance instead of subtracting independent marginals.
    """

    child = _finite_vector(child_mean, name="child_mean")
    parent = _finite_vector(parent_mean, name="parent_mean")
    preference = _finite_vector(weights, name="weights")
    if child.shape != parent.shape or preference.shape != child.shape:
        raise ValueError("child_mean, parent_mean, and weights must have identical shapes")
    covariance = _validated_joint_covariance(joint_covariance, objectives=child.size)

    direction = np.concatenate([preference, -preference])
    endpoint_scale = float(max(np.max(np.abs(child)), np.max(np.abs(parent))))
    weight_scale = float(np.max(np.abs(preference)))
    if endpoint_scale == 0.0 or weight_scale == 0.0:
        mean = 0.0
    else:
        scaled_difference = child / endpoint_scale - parent / endpoint_scale
        scaled_mean = float((preference / weight_scale) @ scaled_difference)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = float((scaled_mean * weight_scale) * endpoint_scale)
        if not np.isfinite(mean):
            raise ValueError("linear contrast mean is not representable in float64")
    variance = float(direction @ covariance @ direction)
    scale = float(np.max(np.abs(covariance)))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * covariance.shape[0]
    if variance < -tolerance:
        raise ValueError("linear contrast variance cannot be negative")
    return ContrastMoments(mean=mean, variance=max(variance, 0.0))


def sampled_paired_contrast(
    joint_outcome_draws: ArrayLike,
    scalarizer: Scalarizer,
    *,
    context: object,
) -> ContrastMoments:
    """Estimate nonlinear scalarized contrast moments from paired joint draws.

    Draws have shape ``(n_draws, 2, n_objectives)`` with child outcomes at
    index zero and their *jointly drawn* parent outcomes at index one.  The
    scalarizer receives a two-dimensional outcome matrix and the recorded
    context, and must return one utility per draw.  Applying it before taking
    the difference preserves nonlinearities and posterior dependence.
    """

    draws = np.asarray(joint_outcome_draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[0] < 2 or draws.shape[1] != 2 or draws.shape[2] == 0:
        raise ValueError("joint_outcome_draws must have shape (n_draws >= 2, 2, n_objectives)")
    if np.any(~np.isfinite(draws)):
        raise ValueError("joint_outcome_draws must contain only finite values")

    child_utility = np.asarray(scalarizer(draws[:, 0, :], context), dtype=np.float64)
    parent_utility = np.asarray(scalarizer(draws[:, 1, :], context), dtype=np.float64)
    expected = (draws.shape[0],)
    if child_utility.shape != expected or parent_utility.shape != expected:
        raise ValueError(f"scalarizer must return one utility per draw with shape {expected}")
    if np.any(~np.isfinite(child_utility)) or np.any(~np.isfinite(parent_utility)):
        raise ValueError("scalarizer must return only finite utilities")

    contrasts = child_utility - parent_utility
    return ContrastMoments(
        mean=float(np.mean(contrasts)),
        variance=float(np.var(contrasts, ddof=1)),
    )


def conservative_paired_advantage(
    contrast_mean: ArrayLike,
    contrast_variance: ArrayLike,
    *,
    risk_kappa: float,
    violations: ArrayLike | None = None,
    multipliers: ArrayLike | None = None,
) -> FloatArray:
    """Compute conservative scalarized advantages for a batch of children."""

    mean = _finite_vector(contrast_mean, name="contrast_mean")
    variance = _finite_vector(contrast_variance, name="contrast_variance")
    if variance.shape != mean.shape:
        raise ValueError("contrast_mean and contrast_variance must have identical shapes")
    if np.any(variance < 0):
        raise ValueError("contrast_variance must be non-negative")
    if not np.isfinite(risk_kappa) or risk_kappa < 0:
        raise ValueError("risk_kappa must be finite and non-negative")
    if (violations is None) != (multipliers is None):
        raise ValueError("violations and multipliers must be supplied together")

    penalty = np.zeros_like(mean)
    if violations is not None:
        violation_matrix = np.asarray(violations, dtype=np.float64)
        multiplier_values = np.asarray(multipliers, dtype=np.float64)
        if (
            violation_matrix.ndim != 2
            or violation_matrix.shape[0] != mean.size
            or violation_matrix.shape[1] == 0
        ):
            raise ValueError("violations must have shape (n_children, n_constraints)")
        if np.any(~np.isfinite(violation_matrix)) or np.any(violation_matrix < 0):
            raise ValueError("violations must be finite and non-negative")
        if multiplier_values.ndim == 1:
            if multiplier_values.shape != (violation_matrix.shape[1],):
                raise ValueError("multipliers must have one value per constraint")
            multiplier_values = np.broadcast_to(multiplier_values, violation_matrix.shape)
        elif multiplier_values.shape != violation_matrix.shape:
            raise ValueError("context-specific multipliers must match the violations matrix shape")
        if np.any(~np.isfinite(multiplier_values)) or np.any(multiplier_values < 0):
            raise ValueError("multipliers must be finite and non-negative")
        with np.errstate(over="ignore", invalid="ignore"):
            penalty = np.sum(violation_matrix * multiplier_values, axis=1)
        if np.any(~np.isfinite(penalty)):
            raise ValueError("constraint penalty is not representable in float64")

    with np.errstate(over="ignore", invalid="ignore"):
        risk_penalty = risk_kappa * np.sqrt(variance)
        advantage = mean - risk_penalty - penalty
    if np.any(~np.isfinite(advantage)):
        raise ValueError("conservative advantage is not representable in float64")
    return np.asarray(advantage, dtype=np.float64)


def gate_paired_advantages(
    contrast_mean: ArrayLike,
    contrast_variance: ArrayLike,
    absolute_utility_risk_score: ArrayLike,
    chance_feasible: ArrayLike,
    *,
    risk_kappa: float,
    minimum_utility: ArrayLike,
    violations: ArrayLike | None = None,
    multipliers: ArrayLike | None = None,
) -> AdvantageGateResult:
    """Gate positive endpoint updates using relative, absolute, and safety tests.

    If every child has non-positive conservative advantage, falls below its
    context-specific absolute utility floor, or is infeasible, ``update_enabled``
    is false.  Callers must then skip the positive endpoint update rather than
    normalize the least-bad child into a winner.
    """

    advantages = conservative_paired_advantage(
        contrast_mean,
        contrast_variance,
        risk_kappa=risk_kappa,
        violations=violations,
        multipliers=multipliers,
    )
    utility_risk_score = _finite_vector(
        absolute_utility_risk_score,
        name="absolute_utility_risk_score",
    )
    if utility_risk_score.shape != advantages.shape:
        raise ValueError("absolute_utility_risk_score must have one value per child")

    feasible = np.asarray(chance_feasible)
    if feasible.dtype.kind != "b" or feasible.shape != advantages.shape:
        raise ValueError("chance_feasible must be a boolean vector with one value per child")

    floor = np.asarray(minimum_utility, dtype=np.float64)
    if floor.ndim == 0:
        floor = np.full(advantages.shape, float(floor), dtype=np.float64)
    if floor.shape != advantages.shape or np.any(~np.isfinite(floor)):
        raise ValueError("minimum_utility must be finite and scalar or one value per child")

    accepted = (advantages > 0.0) & (utility_risk_score >= floor) & feasible
    return AdvantageGateResult(
        advantages=tuple(float(value) for value in advantages),
        accepted=tuple(bool(value) for value in accepted),
    )


def _tie_safe_rank_fraction(values: FloatArray) -> FloatArray:
    """Return zero-to-one midranks, assigning exactly equal values equal rank."""

    if values.size == 1:
        return np.array([0.5], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        midrank = 0.5 * (start + end - 1) / (values.size - 1)
        ranks[order[start:end]] = midrank
        start = end
    return ranks


def historical_posterior_influence_credit(
    optimum_proxy_samples: ArrayLike,
    observed_mean: ArrayLike,
    observed_variance: ArrayLike,
    similarity: ArrayLike,
    *,
    credit_floor: float = 0.1,
    credit_ceiling: float = 1.0,
    variance_floor: float = 1e-12,
    sensitivity: float = 1.0,
    iteration: float = 0.0,
    half_life: float = 20.0,
    weight_floor: float = 0.1,
    weight_ceiling: float = 1.0,
) -> HistoricalInfluenceCredit:
    """Compute bounded historical posterior-influence weights.

    Each observation receives the log mean predictive density of posterior
    optimum-proxy samples under its Gaussian marginal.  Deterministic midranks
    map that evidence into bounded observed credit.  An explicit, nonnegative
    homology/embedding similarity matrix then propagates credit from observed
    points to candidate or branch rows.  A zero-similarity row falls back to the
    neutral credit and weight ``1``.

    The decay exponent is ``sensitivity / (1 + iteration / half_life)``.  It
    makes every positive propagated credit approach neutral weight ``1`` as
    the search progresses, while explicit weight clamps limit early influence.
    This is model-relative data attribution and does not identify a causal
    effect of an observation on the biological outcome.
    """

    proxies = _finite_vector(optimum_proxy_samples, name="optimum_proxy_samples")
    means = _finite_vector(observed_mean, name="observed_mean")
    variances = _finite_vector(observed_variance, name="observed_variance")
    if variances.shape != means.shape:
        raise ValueError("observed_mean and observed_variance must have identical shapes")
    if np.any(variances < 0):
        raise ValueError("observed_variance must be non-negative")

    similarities = np.asarray(similarity, dtype=np.float64)
    expected_columns = means.size
    if similarities.ndim != 2 or similarities.shape[0] == 0:
        raise ValueError("similarity must be a non-empty two-dimensional matrix")
    if similarities.shape[1] != expected_columns:
        raise ValueError("similarity must have one column per observed point")
    if np.any(~np.isfinite(similarities)) or np.any(similarities < 0):
        raise ValueError("similarity must be finite and non-negative")

    scalar_parameters = {
        "credit_floor": credit_floor,
        "credit_ceiling": credit_ceiling,
        "variance_floor": variance_floor,
        "sensitivity": sensitivity,
        "iteration": iteration,
        "half_life": half_life,
        "weight_floor": weight_floor,
        "weight_ceiling": weight_ceiling,
    }
    if any(not np.isfinite(value) for value in scalar_parameters.values()):
        raise ValueError("historical influence parameters must be finite")
    if not 0 < credit_floor <= 1.0 <= credit_ceiling:
        raise ValueError("credit bounds must satisfy 0 < floor <= 1 <= ceiling")
    if not 0 < weight_floor <= 1.0 <= weight_ceiling:
        raise ValueError("weight bounds must satisfy 0 < floor <= 1 <= ceiling")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    if sensitivity < 0 or iteration < 0:
        raise ValueError("sensitivity and iteration must be non-negative")
    if half_life <= 0:
        raise ValueError("half_life must be positive")

    effective_variance = variances + variance_floor
    if np.any(~np.isfinite(effective_variance)):
        raise ValueError("observed variance plus variance_floor must remain finite")
    standard_deviation = np.sqrt(effective_variance)
    maximum_z = np.sqrt(np.finfo(np.float64).max / 4.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = (proxies[:, None] - means[None, :]) / standard_deviation[None, :]
    standardized = np.nan_to_num(
        standardized,
        nan=0.0,
        posinf=maximum_z,
        neginf=-maximum_z,
    )
    standardized = np.clip(standardized, -maximum_z, maximum_z)
    log_density = -0.5 * (
        np.log(2.0 * np.pi) + np.log(effective_variance)[None, :] + np.square(standardized)
    )
    maximum_log_density = np.max(log_density, axis=0)
    log_evidence = maximum_log_density + np.log(
        np.mean(np.exp(log_density - maximum_log_density[None, :]), axis=0)
    )
    if np.any(~np.isfinite(log_evidence)):
        raise FloatingPointError("log predictive evidence became non-finite")

    rank_fraction = _tie_safe_rank_fraction(log_evidence)
    observed_credit = credit_floor + (credit_ceiling - credit_floor) * rank_fraction
    similarity_mass = np.sum(similarities, axis=1)
    if np.any(~np.isfinite(similarity_mass)):
        raise ValueError("similarity row sums must remain finite")
    normalized_similarity = np.zeros_like(similarities)
    nonempty = similarity_mass > 0
    normalized_similarity[nonempty] = similarities[nonempty] / similarity_mass[nonempty, None]
    propagated_credit = np.ones(similarities.shape[0], dtype=np.float64)
    propagated_credit[nonempty] = normalized_similarity[nonempty] @ observed_credit
    propagated_credit = np.clip(propagated_credit, credit_floor, credit_ceiling)

    decay_exponent = sensitivity / (1.0 + iteration / half_life)
    log_weight = decay_exponent * np.log(propagated_credit)
    log_weight = np.clip(log_weight, np.log(weight_floor), np.log(weight_ceiling))
    decayed_weight = np.exp(log_weight)
    decayed_weight[~nonempty] = 1.0

    return HistoricalInfluenceCredit(
        log_predictive_evidence=log_evidence,
        rank_fraction=rank_fraction,
        observed_credit=observed_credit,
        normalized_similarity=normalized_similarity,
        similarity_mass=similarity_mass,
        propagated_credit=propagated_credit,
        decayed_weight=decayed_weight,
        decay_exponent=decay_exponent,
    )

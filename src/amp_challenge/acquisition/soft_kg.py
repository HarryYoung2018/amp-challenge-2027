"""Dense Gaussian soft Knowledge Gradient for small audited frontiers.

The module deliberately exposes one acquisition seam and keeps posterior
conditioning, common-random-number fantasies, chance constraints, stable
log-mean-exp, and Monte Carlo error accounting behind it.  It is intended for
the first 16--64 candidate research slice.  A cluster implementation may
replace the dense Gaussian representation with a low-rank adapter while
preserving :class:`GaussianSoftKG`'s observable contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import comb, fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, logsumexp, ndtr

from amp_challenge.models.posterior import (
    MAX_RELATIVE_EIGENVALUE_CUTOFF,
    JointGaussianPosterior,
    psd_joseph_conditioned_covariance,
    solve_psd_with_projected_square_root,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _binary_fraction(value: float | np.floating) -> Fraction:
    """Return the exact rational represented by a binary floating scalar."""

    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _fraction_to_float64(value: Fraction, *, name: str) -> float:
    """Round an exact rational to float64 or reject range loss."""

    try:
        result = float(value)
    except OverflowError as error:
        raise FloatingPointError(f"{name} is not representable in float64") from error
    if not np.isfinite(result) or (value != 0 and result == 0.0):
        raise FloatingPointError(f"{name} is not representable in float64")
    return result


def _exact_binary_dot(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
) -> float:
    """Evaluate a short ill-conditioned dot product exactly before rounding."""

    total = Fraction(0)
    for left_value, right_value in zip(left.flat, right.flat, strict=True):
        total += _binary_fraction(left_value) * _binary_fraction(right_value)
    return _fraction_to_float64(total, name=name)


def _cancellation_mask(
    result: NDArray[np.longdouble],
    absolute_accumulation: NDArray[np.longdouble],
    *,
    term_count: int,
) -> BoolArray:
    """Identify reductions whose extended-precision result is not trustworthy."""

    error_factor = np.longdouble(8 * (term_count + 1)) * np.longdouble(np.finfo(np.longdouble).eps)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        bound = error_factor * absolute_accumulation
    return np.asarray(
        (absolute_accumulation > 0.0) & (np.abs(result) <= bound),
        dtype=bool,
    )


def _checked_positive_sum(values: FloatArray, *, name: str) -> float:
    """Sum validated positive float64 values without silent overflow."""

    scale = float(np.max(values))
    scaled_total = fsum(float(value / scale) for value in values)
    if scale > np.finfo(np.float64).max / scaled_total:
        raise ValueError(f"{name} is not representable in float64")
    total = scale * scaled_total
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return total


def _checked_nonnegative_ratios(
    numerators: FloatArray,
    denominators: FloatArray,
    *,
    name: str,
    eligible: BoolArray | None = None,
) -> FloatArray:
    """Divide non-negative finite values or reject unrepresentable scores."""

    scores = np.zeros_like(numerators, dtype=np.float64)
    positions = np.ones(numerators.shape, dtype=bool) if eligible is None else eligible
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores[positions] = numerators[positions] / denominators[positions]
    if np.any(~np.isfinite(scores)):
        raise ValueError(f"{name} is not representable in float64")
    if np.any(positions & (numerators > 0.0) & (scores == 0.0)):
        raise ValueError(f"{name} is not representable in float64")
    return scores


def _stable_sample_mean_and_standard_error(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return row moments without overflowing a representable mean or error."""

    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("sample moments require a matrix with at least two columns")
    if np.any(~np.isfinite(values)):
        raise ValueError("sample moments require finite values")
    sample_count = values.shape[1]
    extended_values = np.asarray(values, dtype=np.longdouble)
    extended_means = np.empty(values.shape[0], dtype=np.longdouble)
    for row_index, row in enumerate(values):
        try:
            exact_float_sum = fsum(float(value) for value in row)
        except OverflowError:
            scale = float(np.max(np.abs(row)))
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                scaled_row = np.asarray(row / scale, dtype=np.float64)
            if np.any((row != 0.0) & (scaled_row == 0.0)):
                raise FloatingPointError(
                    "soft-KG estimate or standard error is not representable"
                ) from None
            scaled_total = fsum(float(value) for value in scaled_row)
            extended_total = np.longdouble(scale) * np.longdouble(scaled_total)
        else:
            extended_total = np.longdouble(exact_float_sum)
        extended_means[row_index] = extended_total / np.longdouble(sample_count)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        deviations = extended_values - extended_means[:, None]
        extended_variances = np.sum(
            deviations * deviations,
            axis=1,
            dtype=np.longdouble,
        ) / np.longdouble(sample_count - 1)
        extended_errors = np.sqrt(extended_variances / np.longdouble(sample_count))
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if (
        np.any(~np.isfinite(extended_means))
        or np.any(~np.isfinite(extended_errors))
        or np.any(np.abs(extended_means) > float64_limit)
        or np.any(extended_errors > float64_limit)
    ):
        raise FloatingPointError("soft-KG estimate or standard error is not representable")
    means = np.asarray(extended_means, dtype=np.float64)
    errors = np.asarray(extended_errors, dtype=np.float64)
    if np.any((extended_means != 0.0) & (means == 0.0)) or np.any(
        (extended_errors > 0.0) & (errors == 0.0)
    ):
        raise FloatingPointError("soft-KG estimate or standard error is not representable")
    return means, errors


def _normalize_positive(values: FloatArray, *, axis: int | None = None) -> FloatArray:
    scale = np.max(values, axis=axis, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("normalization requires positive mass")
    scaled = values / scale
    if np.any((values > 0.0) & (scaled == 0.0)):
        raise ValueError("normalization would erase positive mass")
    normalized = np.asarray(
        scaled / np.sum(scaled, axis=axis, keepdims=True),
        dtype=np.float64,
    )
    if np.any((values > 0.0) & (normalized == 0.0)):
        raise ValueError("normalization would erase positive mass")
    return normalized


def _canonical_log_positive_masses(values: FloatArray, *, name: str) -> FloatArray:
    """Normalize positive float64 masses before taking logs in extended range."""

    masses = np.asarray(values, dtype=np.float64)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    extended = np.asarray(masses, dtype=np.longdouble)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        scaled = extended / np.max(extended)
        normalized = scaled / np.sum(scaled, dtype=np.longdouble)
        extended_logs = np.log(normalized)
    if np.any(normalized == 0.0) or np.any(~np.isfinite(extended_logs)):
        raise ValueError(f"{name} log normalization is not representable")
    logs = np.asarray(extended_logs, dtype=np.float64)
    if np.any(~np.isfinite(logs)):
        raise ValueError(f"{name} log normalization is not representable")
    logs = np.asarray(logs - logsumexp(logs), dtype=np.float64)
    strictly_less = masses[:, None] < masses[None, :]
    if np.any(strictly_less & ~(logs[:, None] < logs[None, :])):
        raise ValueError(f"{name} distinctions are not representable in log space")
    logs.setflags(write=False)
    return logs


def _stable_centered_logits(
    values: FloatArray,
    centers: FloatArray | float,
    *,
    divisor: float,
    name: str,
) -> FloatArray:
    """Return ``(values - centers) / divisor`` without opposite-sign overflow."""

    value_array, center_array = np.broadcast_arrays(
        np.asarray(values, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
    )
    if np.any(np.isnan(value_array)) or np.any(np.isposinf(value_array)):
        raise ValueError(f"{name} values may contain finite values or -inf only")
    if np.any(~np.isfinite(center_array)):
        raise ValueError(f"{name} centers must be finite")
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(f"{name} divisor must be finite and positive")

    result = np.full(value_array.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(value_array)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        differences = value_array[finite] - center_array[finite]
        regular = np.isfinite(differences)
        finite_result = np.empty_like(differences)
        finite_result[regular] = differences[regular] / divisor

        overflowing = ~regular
        if np.any(overflowing):
            left = value_array[finite][overflowing]
            right = center_array[finite][overflowing]
            scales = np.maximum(np.abs(left), np.abs(right))
            scaled_differences = left / scales - right / scales
            scaled_divisors = divisor / scales
            finite_result[overflowing] = scaled_differences / scaled_divisors
    result[finite] = finite_result
    unequal = finite & (value_array != center_array)
    if np.any(np.isnan(result)) or np.any(result > 0.0):
        raise FloatingPointError(f"{name} became invalid")
    if np.any(unequal & (result == 0.0)):
        raise FloatingPointError(f"{name} is not representable in float64")
    return result


def _standardized_upper_difference(
    upper_bound: float,
    mean: FloatArray,
    variance: FloatArray,
) -> FloatArray:
    """Return ``(upper_bound - mean) / sqrt(variance)`` without overflow warnings.

    The caller supplies strictly positive finite variances.  Ordinary
    subtraction is preferable for same-scale, same-sign operands because it
    retains nearby-float differences.  Only an overflowing subtraction is
    recomputed after a common scaling.  A mathematically out-of-range ratio is
    represented by signed infinity, which is an exact saturation input for
    ``ndtr`` and ``log_ndtr`` rather than a non-finite input moment.
    """

    means = np.asarray(mean, dtype=np.float64)
    variances = np.asarray(variance, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        standard_deviation = np.sqrt(variances)
        difference = np.float64(upper_bound) - means
        standardized = difference / standard_deviation

    overflowing_difference = ~np.isfinite(difference)
    if np.any(overflowing_difference):
        selected_means = means[overflowing_difference]
        selected_deviation = standard_deviation[overflowing_difference]
        scale = np.maximum(np.abs(selected_means), abs(upper_bound))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            scaled_difference = np.float64(upper_bound) / scale - selected_means / scale
            scaled_deviation = selected_deviation / scale
            standardized = np.array(standardized, dtype=np.float64, copy=True)
            standardized[overflowing_difference] = scaled_difference / scaled_deviation

    if np.any(np.isnan(standardized)):
        raise FloatingPointError("constraint standardization became invalid")
    return np.asarray(standardized, dtype=np.float64)


def _checked_affine_float64(
    base: FloatArray | float,
    multiplier: float,
    offset: FloatArray | float,
    *,
    name: str,
) -> FloatArray:
    """Evaluate an affine expression with exact fallback near cancellation."""

    base_values, offset_values = np.broadcast_arrays(
        np.asarray(base, dtype=np.float64),
        np.asarray(offset, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(base_values))
        or not np.isfinite(multiplier)
        or np.any(~np.isfinite(offset_values))
    ):
        raise FloatingPointError(f"{name} requires finite operands")
    extended_base = np.asarray(base_values, dtype=np.longdouble)
    extended_offset = np.asarray(offset_values, dtype=np.longdouble)
    extended_multiplier = np.longdouble(multiplier)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        product = extended_multiplier * extended_offset
        extended = extended_base + product
        accumulation = np.abs(extended_base) + np.abs(product)
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if np.any(~np.isfinite(extended)) or np.any(np.abs(extended) > float64_limit):
        raise FloatingPointError(f"{name} is not representable in float64")
    result = np.asarray(extended, dtype=np.float64)
    suspicious = _cancellation_mask(extended, accumulation, term_count=2)
    if np.any(suspicious):
        result = np.array(result, dtype=np.float64, copy=True)
        for flat_index in np.flatnonzero(suspicious):
            exact = _binary_fraction(base_values.flat[flat_index]) + _binary_fraction(
                np.float64(multiplier)
            ) * _binary_fraction(offset_values.flat[flat_index])
            result.flat[flat_index] = _fraction_to_float64(exact, name=name)
    if np.any(~suspicious & (extended != 0.0) & (result == 0.0)):
        raise FloatingPointError(f"{name} is not representable in float64")
    return result


def _checked_weighted_utilities(
    objective_mean: FloatArray,
    preference_weights: FloatArray,
) -> FloatArray:
    """Compute finite weighted utilities with exact cancellation fallback."""

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        extended_objectives = np.asarray(objective_mean, dtype=np.longdouble)
        extended_preferences = np.asarray(preference_weights, dtype=np.longdouble)
        extended = np.einsum(
            "fdo,ko->fkd",
            extended_objectives,
            extended_preferences,
            optimize=False,
        )
        accumulation = np.einsum(
            "fdo,ko->fkd",
            np.abs(extended_objectives),
            np.abs(extended_preferences),
            optimize=False,
        )
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if np.any(~np.isfinite(extended)) or np.any(np.abs(extended) > float64_limit):
        raise FloatingPointError("preference utility is not representable in float64")
    utility = np.asarray(extended, dtype=np.float64)
    suspicious = _cancellation_mask(
        extended,
        accumulation,
        term_count=objective_mean.shape[-1],
    )
    if np.any(suspicious):
        utility = np.array(utility, dtype=np.float64, copy=True)
        for fantasy, preference, decision in np.argwhere(suspicious):
            utility[fantasy, preference, decision] = _exact_binary_dot(
                np.asarray(objective_mean[fantasy, decision], dtype=np.float64),
                np.asarray(preference_weights[preference], dtype=np.float64),
                name="preference utility",
            )
    if np.any(~suspicious & (extended != 0.0) & (utility == 0.0)):
        raise FloatingPointError("preference utility is not representable in float64")
    return utility


def _checked_log_weighted_average(
    values: FloatArray,
    log_weights: FloatArray,
) -> FloatArray:
    """Average signed rows with exact fallback for ill-conditioned sums."""

    value_array = np.asarray(values, dtype=np.float64)
    weight_logs = np.asarray(log_weights, dtype=np.float64)
    if value_array.ndim != 2 or weight_logs.shape != (value_array.shape[1],):
        raise ValueError("signed weighted average dimensions do not match")
    if np.any(~np.isfinite(value_array)) or np.any(~np.isfinite(weight_logs)):
        raise ValueError("signed weighted average inputs must be finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        shifted_logs = np.asarray(weight_logs, dtype=np.longdouble) - np.max(
            np.asarray(weight_logs, dtype=np.longdouble)
        )
        relative_weights = np.exp(shifted_logs)
        normalized_weights = relative_weights / np.sum(
            relative_weights,
            dtype=np.longdouble,
        )
        extended_values = np.asarray(value_array, dtype=np.longdouble)
        extended = np.einsum(
            "ij,j->i",
            extended_values,
            normalized_weights,
            optimize=False,
        )
        accumulation = np.einsum(
            "ij,j->i",
            np.abs(extended_values),
            normalized_weights,
            optimize=False,
        )
    if np.any(relative_weights == 0.0) or np.any(~np.isfinite(extended)):
        raise FloatingPointError(
            "preference-averaged soft value is not representable in the locked accumulator"
        )
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if np.any(np.abs(extended) > float64_limit):
        raise FloatingPointError("preference-averaged soft value is not representable in float64")
    averaged = np.asarray(extended, dtype=np.float64)
    suspicious = _cancellation_mask(
        extended,
        accumulation,
        term_count=value_array.shape[1],
    )
    if np.any(suspicious):
        averaged = np.array(averaged, dtype=np.float64, copy=True)
        for row_index in np.flatnonzero(suspicious):
            averaged[row_index] = _exact_binary_dot(
                np.asarray(value_array[row_index], dtype=np.float64),
                normalized_weights,
                name="preference-averaged soft value",
            )
    if np.any(~suspicious & (extended != 0.0) & (averaged == 0.0)):
        raise FloatingPointError("preference-averaged soft value is not representable in float64")
    return averaged


def _float_array(value: object, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _integer_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    parsed_values: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{name} must contain integers")
        parsed_values.append(int(value))
    parsed = tuple(parsed_values)
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    if any(value < 0 for value in parsed):
        raise ValueError(f"{name} must be non-negative")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must be unique")
    return parsed


def _optional_positive_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer or None")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def maximum_exhaustive_pool_size(
    *,
    batch_size: int,
    max_combinations: int,
    candidate_count: int,
) -> int:
    """Return the largest screened pool whose exact q-search fits the guard."""

    for value, name in (
        (batch_size, "batch_size"),
        (max_combinations, "max_combinations"),
        (candidate_count, "candidate_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{name} must be an integer")
    q = int(batch_size)
    guard = int(max_combinations)
    candidates = int(candidate_count)
    if q <= 0 or guard <= 0 or candidates < 0:
        raise ValueError(
            "batch_size and max_combinations must be positive; candidate_count cannot be negative"
        )
    if candidates < q:
        return candidates
    if comb(q, q) > guard:
        return q - 1

    low = q
    high = candidates
    while low < high:
        midpoint = (low + high + 1) // 2
        if comb(midpoint, q) <= guard:
            low = midpoint
        else:
            high = midpoint - 1
    return low


def stable_soft_value(
    values: Sequence[float] | FloatArray,
    *,
    temperature: float,
    base_measure: Sequence[float] | FloatArray | None = None,
) -> float:
    """Return a max-shifted log-mean-exp value.

    ``base_measure`` defines the decision measure and is normalized internally.
    Entries in ``values`` may be ``-inf`` to mask chance-infeasible decisions,
    but at least one value with positive mass must remain finite.
    """

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty vector")
    if np.any(np.isnan(vector)) or np.any(np.isposinf(vector)):
        raise ValueError("values may contain finite values or -inf only")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if base_measure is None:
        log_measure = _canonical_log_positive_masses(
            np.ones(vector.size, dtype=np.float64),
            name="base_measure",
        )
    else:
        measure = np.asarray(base_measure, dtype=np.float64)
        if measure.shape != vector.shape:
            raise ValueError("base_measure must match values")
        log_measure = _canonical_log_positive_masses(
            measure,
            name="base_measure",
        )
    finite = np.isfinite(vector)
    if not np.any(finite):
        raise ValueError("at least one decision value must remain finite")
    maximum = float(np.max(vector[finite]))
    centered_logits = _stable_centered_logits(
        vector[finite],
        maximum,
        divisor=temperature,
        name="soft-value centered logits",
    )
    log_weighted_sum = float(logsumexp(log_measure[finite] + centered_logits))
    if not np.isfinite(log_weighted_sum):
        raise FloatingPointError("soft-value normalization became non-finite")
    result = _checked_affine_float64(
        maximum,
        temperature,
        log_weighted_sum,
        name="soft value",
    )
    return float(result)


@dataclass(frozen=True, slots=True)
class PreferenceMeasure:
    """Frozen finite measure over normalized linear objective scalarizations.

    ``log_probabilities`` is the canonical measure.  The ordinary probability
    view is retained for inspection and may underflow to zero for a mass ratio
    that float64 cannot represent directly.
    """

    weights: FloatArray
    probabilities: FloatArray | None = None
    log_probabilities: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        weights = _float_array(self.weights, name="preference weights")
        if weights.ndim != 2 or 0 in weights.shape:
            raise ValueError("preference weights must have shape (n_preferences, n_objectives)")
        if np.any(weights < 0) or np.any(np.max(weights, axis=1) <= 0):
            raise ValueError("each preference must have non-negative, non-zero weights")
        weights = _normalize_positive(weights, axis=1)
        if self.probabilities is None:
            log_probabilities = _canonical_log_positive_masses(
                np.ones(weights.shape[0], dtype=np.float64),
                name="preference probabilities",
            )
        else:
            probabilities = _float_array(self.probabilities, name="preference probabilities")
            if probabilities.shape != (weights.shape[0],):
                raise ValueError("preference probabilities must match preference rows")
            log_probabilities = _canonical_log_positive_masses(
                probabilities,
                name="preference probabilities",
            )
        weights = np.array(weights, dtype=np.float64, copy=True)
        log_probabilities = np.array(log_probabilities, dtype=np.float64, copy=True)
        probabilities = np.array(np.exp(log_probabilities), dtype=np.float64, copy=True)
        weights.setflags(write=False)
        log_probabilities.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "log_probabilities", log_probabilities)


@dataclass(frozen=True, slots=True)
class UpperChanceConstraint:
    """Require ``P(output <= upper_bound) >= 1 - max_violation_probability``."""

    output_index: int
    upper_bound: float
    max_violation_probability: float

    def __post_init__(self) -> None:
        if isinstance(self.output_index, bool) or not isinstance(
            self.output_index,
            int | np.integer,
        ):
            raise ValueError("constraint output_index must be an integer")
        if self.output_index < 0:
            raise ValueError("constraint output_index must be non-negative")
        if not np.isfinite(self.upper_bound):
            raise ValueError("constraint upper_bound must be finite")
        if not 0 < self.max_violation_probability < 1:
            raise ValueError("max_violation_probability must lie in (0, 1)")

    def satisfaction_probability(
        self,
        mean: ArrayLike,
        variance: ArrayLike,
    ) -> FloatArray:
        """Return ``P(Y <= upper_bound)`` without conflating jitter and certainty."""

        try:
            output_mean, output_variance = np.broadcast_arrays(
                np.asarray(mean, dtype=np.float64),
                np.asarray(variance, dtype=np.float64),
            )
        except ValueError as error:
            raise ValueError("constraint mean and variance must be broadcast-compatible") from error
        if np.any(~np.isfinite(output_mean)) or np.any(~np.isfinite(output_variance)):
            raise ValueError("constraint moments must be finite")
        if np.any(output_variance < 0.0):
            raise ValueError("constraint variance must be non-negative")
        probability = np.empty_like(output_mean)
        deterministic = output_variance == 0.0
        probability[deterministic] = (output_mean[deterministic] <= self.upper_bound).astype(
            np.float64
        )
        uncertain = ~deterministic
        standardized = _standardized_upper_difference(
            self.upper_bound,
            output_mean[uncertain],
            output_variance[uncertain],
        )
        probability[uncertain] = ndtr(standardized)
        return np.asarray(probability, dtype=np.float64)

    def is_satisfied(
        self,
        mean: ArrayLike,
        variance: ArrayLike,
    ) -> BoolArray:
        """Compare Gaussian violation mass in log space without rounding to one."""

        try:
            output_mean, output_variance = np.broadcast_arrays(
                np.asarray(mean, dtype=np.float64),
                np.asarray(variance, dtype=np.float64),
            )
        except ValueError as error:
            raise ValueError("constraint mean and variance must be broadcast-compatible") from error
        if np.any(~np.isfinite(output_mean)) or np.any(~np.isfinite(output_variance)):
            raise ValueError("constraint moments must be finite")
        if np.any(output_variance < 0.0):
            raise ValueError("constraint variance must be non-negative")
        satisfied = np.empty_like(output_mean, dtype=bool)
        deterministic = output_variance == 0.0
        satisfied[deterministic] = output_mean[deterministic] <= self.upper_bound
        uncertain = ~deterministic
        standardized = _standardized_upper_difference(
            self.upper_bound,
            output_mean[uncertain],
            output_variance[uncertain],
        )
        satisfied[uncertain] = log_ndtr(-standardized) <= np.log(self.max_violation_probability)
        return np.asarray(satisfied, dtype=bool)


@dataclass(frozen=True, slots=True)
class SoftKGProblem:
    """One frozen decision problem shared by every acquisition candidate.

    ``log_base_measure`` is authoritative.  ``base_measure`` is its normalized
    ordinary-arithmetic view and may underflow for an otherwise retained,
    strictly positive log mass.
    """

    decision_indices: tuple[int, ...]
    objective_outputs: tuple[int, ...]
    preferences: PreferenceMeasure
    base_measure: FloatArray
    constraints: tuple[UpperChanceConstraint, ...] = ()
    always_safe_decisions: tuple[int, ...] = ()
    log_base_measure: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        decisions = _integer_tuple(self.decision_indices, name="decision_indices")
        objectives = _integer_tuple(self.objective_outputs, name="objective_outputs")
        if self.preferences.weights.shape[1] != len(objectives):
            raise ValueError("preference width must match objective_outputs")
        measure = _float_array(self.base_measure, name="base_measure")
        if measure.shape != (len(decisions),) or np.any(measure <= 0):
            raise ValueError("base_measure must be strictly positive for every decision")
        log_measure = _canonical_log_positive_masses(
            measure,
            name="base_measure",
        )
        measure = np.array(np.exp(log_measure), dtype=np.float64, copy=True)
        measure.setflags(write=False)
        log_measure = np.array(log_measure, dtype=np.float64, copy=True)
        log_measure.setflags(write=False)
        constraints = tuple(self.constraints)
        constraint_outputs = [constraint.output_index for constraint in constraints]
        if len(constraint_outputs) != len(set(constraint_outputs)):
            raise ValueError("constraint output indices must be unique")
        if sum(constraint.max_violation_probability for constraint in constraints) >= 1:
            raise ValueError("constraint violation probabilities must sum to less than one")
        safe = (
            ()
            if not self.always_safe_decisions
            else _integer_tuple(
                self.always_safe_decisions,
                name="always_safe_decisions",
            )
        )
        if len(set(safe)) != len(safe) or not set(safe) <= set(decisions):
            raise ValueError("always_safe_decisions must be unique decision indices")
        if constraints and not safe:
            raise ValueError("constrained problems require an always-safe outside decision")
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "objective_outputs", objectives)
        object.__setattr__(self, "base_measure", measure)
        object.__setattr__(self, "log_base_measure", log_measure)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "always_safe_decisions", safe)


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    """Potential observations, their costs, and exact structural eligibility."""

    indices: tuple[int, ...]
    costs: FloatArray
    eligible: BoolArray | None = None

    def __post_init__(self) -> None:
        indices = _integer_tuple(self.indices, name="evaluation indices")
        costs = _float_array(self.costs, name="evaluation costs")
        if costs.shape != (len(indices),) or np.any(costs <= 0):
            raise ValueError("costs must be a positive vector matching evaluation indices")
        if self.eligible is None:
            eligible = np.ones(len(indices), dtype=bool)
        else:
            raw_eligible = np.asarray(self.eligible)
            if raw_eligible.dtype.kind != "b":
                raise ValueError("eligible must be a boolean vector")
            eligible = np.array(raw_eligible, dtype=bool, copy=True)
            if eligible.shape != (len(indices),):
                raise ValueError("eligible must match evaluation indices")
        eligible.setflags(write=False)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "eligible", eligible)


@dataclass(frozen=True, slots=True)
class SoftKGResult:
    """Signed estimates and an explicitly heuristic, cost-aware decision score."""

    evaluation_indices: tuple[int, ...]
    estimate: FloatArray
    standard_error: FloatArray
    standard_error_penalized_estimate: FloatArray
    score: FloatArray
    eligible: BoolArray

    @property
    def ranked_evaluation_indices(self) -> tuple[int, ...]:
        selectable = np.flatnonzero(self.eligible & (self.score > 0.0))
        order = selectable[np.argsort(-self.score[selectable], kind="stable")]
        return tuple(self.evaluation_indices[int(position)] for position in order)


@dataclass(frozen=True, slots=True)
class JointSoftKGResult:
    """Auditable exhaustive small-q soft-KG batch comparison.

    Every row is one jointly conditioned evaluation batch.  ``score`` is the
    non-negative, standard-error-penalized joint value divided by the total
    declared evaluation cost; it is not a sum of singleton acquisition scores.
    """

    evaluation_batches: tuple[tuple[int, ...], ...]
    estimate: FloatArray
    standard_error: FloatArray
    standard_error_penalized_estimate: FloatArray
    score: FloatArray
    total_cost: FloatArray

    @property
    def ranked_evaluation_batches(self) -> tuple[tuple[int, ...], ...]:
        selectable = np.flatnonzero(self.score > 0.0)
        order = selectable[np.argsort(-self.score[selectable], kind="stable")]
        return tuple(self.evaluation_batches[int(position)] for position in order)

    @property
    def selected_evaluation_indices(self) -> tuple[int, ...]:
        """Return the best positive-value batch, or the no-evaluation option."""

        ranked = self.ranked_evaluation_batches
        return () if not ranked else ranked[0]


@dataclass(frozen=True, slots=True)
class BeamSearchDepthTrace:
    """One complete, deterministic depth of approximate joint search.

    ``scored`` retains every group and acquisition statistic evaluated at this
    depth.  ``retained_evaluation_batches`` is the lexicographically tie-broken
    frontier used to construct the next depth. Expansion counts are recorded
    before and after cost/completion filtering, with remaining cap headroom;
    an insufficient cap fails before a depth is scored.
    """

    depth: int
    generated_group_count: int
    completion_feasible_group_count: int
    scored: JointSoftKGResult
    retained_evaluation_batches: tuple[tuple[int, ...], ...]
    beam_pruned_group_count: int
    remaining_group_budget: int


@dataclass(frozen=True, slots=True)
class BeamJointSoftKGResult:
    """Auditable bounded-width approximation to exhaustive joint soft-KG."""

    batch_size: int
    beam_width: int
    max_groups_scored: int
    total_groups_scored: int
    approximation_status: str
    depth_trace: tuple[BeamSearchDepthTrace, ...]

    @property
    def final_result(self) -> JointSoftKGResult:
        """Return all final-depth groups reached by the bounded search."""

        return self.depth_trace[-1].scored

    @property
    def selected_evaluation_indices(self) -> tuple[int, ...]:
        """Return the best positive final batch, or the no-evaluation option."""

        return self.final_result.selected_evaluation_indices


def _signed_joint_ranking(result: JointSoftKGResult) -> tuple[int, ...]:
    """Fixed-budget ranking without discarding finite nonpositive estimates."""
    signed = result.standard_error_penalized_estimate / result.total_cost
    if not np.all(np.isfinite(signed)):
        raise FloatingPointError("fixed-budget joint scores must be finite")
    return tuple(
        sorted(
            range(len(result.evaluation_batches)),
            key=lambda position: (-float(signed[position]), result.evaluation_batches[position]),
        )
    )


@dataclass(frozen=True, slots=True)
class FixedBudgetBeamJointSoftKGResult(BeamJointSoftKGResult):
    """Prospective mandatory allocation; original clipped diagnostics retained."""

    allocation_mode: str = "fixed_budget_signed_penalized_gain_every_depth"

    @property
    def selected_evaluation_indices(self) -> tuple[int, ...]:
        ranked = _signed_joint_ranking(self.final_result)
        return () if not ranked else self.final_result.evaluation_batches[ranked[0]]


class GaussianSoftKG:
    """Score a batch through one fixed Gaussian decision problem.

    This implementation validates dense decision and augmented
    decision/observation covariance blocks with full-state eigendecompositions.
    Its worst-case per-group cost is therefore cubic in the combined state, in
    addition to fantasy contractions.  This implementation is intentionally
    bounded to a small frontier; callers should not pass the 50k production
    library.
    """

    def __init__(
        self,
        problem: SoftKGProblem,
        *,
        temperature: float,
        observed_outputs: Sequence[int] | None = None,
        n_fantasies: int = 256,
        standard_error_multiplier: float = 1.0,
        seed: int = 42,
        relative_eigenvalue_cutoff: float = 1e-10,
        candidate_chunk_size: int | None = 64,
        fantasy_chunk_size: int | None = 1024,
    ) -> None:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if isinstance(n_fantasies, bool) or not isinstance(
            n_fantasies,
            int | np.integer,
        ):
            raise ValueError("n_fantasies must be an integer")
        if n_fantasies < 2:
            raise ValueError("n_fantasies must be at least two")
        if not np.isfinite(standard_error_multiplier) or standard_error_multiplier < 0:
            raise ValueError("standard_error_multiplier must be finite and non-negative")
        if (
            not np.isfinite(relative_eigenvalue_cutoff)
            or relative_eigenvalue_cutoff <= 0
            or relative_eigenvalue_cutoff > MAX_RELATIVE_EIGENVALUE_CUTOFF
        ):
            raise ValueError(
                "relative_eigenvalue_cutoff must be finite and lie in "
                f"(0, {MAX_RELATIVE_EIGENVALUE_CUTOFF}]"
            )
        self.problem = problem
        self.temperature = float(temperature)
        self.observed_outputs = (
            None
            if observed_outputs is None
            else tuple(sorted(_integer_tuple(observed_outputs, name="observed_outputs")))
        )
        self.n_fantasies = int(n_fantasies)
        self.standard_error_multiplier = float(standard_error_multiplier)
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.relative_eigenvalue_cutoff = float(relative_eigenvalue_cutoff)
        self.candidate_chunk_size = _optional_positive_integer(
            candidate_chunk_size,
            name="candidate_chunk_size",
        )
        self.fantasy_chunk_size = _optional_positive_integer(
            fantasy_chunk_size,
            name="fantasy_chunk_size",
        )

    def score(self, belief: JointGaussianPosterior, batch: EvaluationBatch) -> SoftKGResult:
        """Return signed soft-KG estimates and conservative selection scores."""

        self._validate_indices(belief, batch)
        groups = tuple((index,) for index in batch.indices)
        estimates, errors = self._joint_estimates(belief, groups)
        penalized = _checked_affine_float64(
            estimates,
            -self.standard_error_multiplier,
            errors,
            name="singleton soft-KG penalized estimate",
        )
        eligible = np.array(batch.eligible, dtype=bool, copy=True)
        scores = _checked_nonnegative_ratios(
            np.maximum(penalized, 0.0),
            batch.costs,
            name="singleton soft-KG score",
            eligible=eligible,
        )
        for array in (estimates, errors, penalized, scores, eligible):
            array.setflags(write=False)
        return SoftKGResult(
            evaluation_indices=batch.indices,
            estimate=estimates,
            standard_error=errors,
            standard_error_penalized_estimate=penalized,
            score=scores,
            eligible=eligible,
        )

    def score_joint_groups(
        self,
        belief: JointGaussianPosterior,
        batch: EvaluationBatch,
        evaluation_groups: Sequence[Sequence[int]],
        *,
        max_total_cost: float | None = None,
        max_groups: int = 4096,
    ) -> JointSoftKGResult:
        """Score an explicit, deduplicated collection of joint batches.

        Groups are canonicalized independently of caller order and evaluated
        together, so every same-depth comparison uses one frozen matrix of
        common-random-number fantasies.  Each group must have one common
        positive size, contain only distinct eligible entries from ``batch``,
        and satisfy the optional total-cost bound.  This is the shared scoring
        seam for approximate optimizers; :meth:`select_joint` remains the
        exhaustive small-frontier reference.
        """

        if isinstance(max_groups, bool) or not isinstance(max_groups, int | np.integer):
            raise ValueError("max_groups must be an integer")
        parsed_max_groups = int(max_groups)
        if parsed_max_groups <= 0:
            raise ValueError("max_groups must be positive")
        parsed_max_total_cost: float | None = None
        if max_total_cost is not None:
            if isinstance(max_total_cost, bool) or not isinstance(
                max_total_cost,
                int | float | np.integer | np.floating,
            ):
                raise ValueError("max_total_cost must be finite and positive")
            parsed_max_total_cost = float(max_total_cost)
            if not np.isfinite(parsed_max_total_cost) or parsed_max_total_cost <= 0:
                raise ValueError("max_total_cost must be finite and positive")

        self._validate_indices(belief, batch)
        try:
            raw_groups = tuple(evaluation_groups)
        except TypeError as error:
            raise ValueError("evaluation_groups must be a finite sequence") from error
        if not raw_groups:
            raise ValueError("evaluation_groups cannot be empty")

        candidate_positions = {index: position for position, index in enumerate(batch.indices)}
        canonical_groups: set[tuple[int, ...]] = set()
        group_size: int | None = None
        for raw_group in raw_groups:
            try:
                group = tuple(sorted(_integer_tuple(raw_group, name="joint evaluation group")))
            except TypeError as error:
                raise ValueError("each evaluation group must be a finite sequence") from error
            if group_size is None:
                group_size = len(group)
            elif len(group) != group_size:
                raise ValueError("evaluation groups must have one common positive size")
            unknown = set(group) - set(candidate_positions)
            if unknown:
                raise ValueError("evaluation groups must refer to candidates in batch")
            if any(not batch.eligible[candidate_positions[index]] for index in group):
                raise ValueError("evaluation groups may contain only eligible candidates")
            canonical_groups.add(group)

        groups = tuple(sorted(canonical_groups))
        if len(groups) > parsed_max_groups:
            raise ValueError("explicit joint group scoring exceeds max_groups")
        total_costs: list[float] = []
        retained_groups: list[tuple[int, ...]] = []
        for group in groups:
            positions = [candidate_positions[index] for index in group]
            total_cost = _checked_positive_sum(
                batch.costs[positions],
                name="joint evaluation total cost",
            )
            if parsed_max_total_cost is not None and total_cost > parsed_max_total_cost:
                continue
            retained_groups.append(group)
            total_costs.append(total_cost)
        if not retained_groups:
            raise ValueError("no eligible evaluation batch satisfies max_total_cost")

        evaluated_groups = tuple(retained_groups)
        estimates, errors = self._joint_estimates(belief, evaluated_groups)
        penalized = _checked_affine_float64(
            estimates,
            -self.standard_error_multiplier,
            errors,
            name="joint soft-KG penalized estimate",
        )
        costs = np.asarray(total_costs, dtype=np.float64)
        scores = _checked_nonnegative_ratios(
            np.maximum(penalized, 0.0),
            costs,
            name="joint soft-KG score",
        )
        for array in (estimates, errors, penalized, costs, scores):
            array.setflags(write=False)
        return JointSoftKGResult(
            evaluation_batches=evaluated_groups,
            estimate=estimates,
            standard_error=errors,
            standard_error_penalized_estimate=penalized,
            score=scores,
            total_cost=costs,
        )

    def select_joint(
        self,
        belief: JointGaussianPosterior,
        batch: EvaluationBatch,
        *,
        batch_size: int,
        max_total_cost: float | None = None,
        max_combinations: int = 4096,
    ) -> JointSoftKGResult:
        """Exhaustively select a covariance-aware small-q evaluation batch.

        Only structurally eligible candidates participate.  Every feasible
        combination is conditioned jointly, including cross-candidate
        covariance and each point's observation noise.  The explicit
        ``max_combinations`` guard keeps this exact API on its intended small
        frontier; larger frontiers should first be screened or use a dedicated
        approximate batch optimizer.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int | np.integer):
            raise ValueError("batch_size must be an integer")
        parsed_batch_size = int(batch_size)
        if parsed_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(max_combinations, bool) or not isinstance(
            max_combinations,
            int | np.integer,
        ):
            raise ValueError("max_combinations must be an integer")
        parsed_max_combinations = int(max_combinations)
        if parsed_max_combinations <= 0:
            raise ValueError("max_combinations must be positive")
        parsed_max_total_cost: float | None = None
        if max_total_cost is not None:
            if isinstance(max_total_cost, bool) or not isinstance(
                max_total_cost,
                int | float | np.integer | np.floating,
            ):
                raise ValueError("max_total_cost must be finite and positive")
            parsed_max_total_cost = float(max_total_cost)
            if not np.isfinite(parsed_max_total_cost) or parsed_max_total_cost <= 0:
                raise ValueError("max_total_cost must be finite and positive")

        self._validate_indices(belief, batch)
        eligible_positions = tuple(int(position) for position in np.flatnonzero(batch.eligible))
        if parsed_batch_size > len(eligible_positions):
            raise ValueError("batch_size cannot exceed the number of eligible candidates")
        combination_count = comb(len(eligible_positions), parsed_batch_size)
        if combination_count > parsed_max_combinations:
            raise ValueError(
                "joint batch search exceeds max_combinations; screen the frontier "
                "or raise the explicit guard"
            )

        evaluation_batches: list[tuple[int, ...]] = []
        total_costs: list[float] = []
        for positions in combinations(eligible_positions, parsed_batch_size):
            total_cost = _checked_positive_sum(
                batch.costs[list(positions)],
                name="joint evaluation total cost",
            )
            if parsed_max_total_cost is not None and total_cost > parsed_max_total_cost:
                continue
            evaluation_batches.append(
                tuple(sorted(batch.indices[position] for position in positions))
            )
            total_costs.append(total_cost)
        if not evaluation_batches:
            raise ValueError("no eligible evaluation batch satisfies max_total_cost")

        ordered = sorted(zip(evaluation_batches, total_costs, strict=True))
        evaluation_batches = [group for group, _cost in ordered]
        total_costs = [cost for _group, cost in ordered]

        groups = tuple(evaluation_batches)
        estimates, errors = self._joint_estimates(belief, groups)
        penalized = _checked_affine_float64(
            estimates,
            -self.standard_error_multiplier,
            errors,
            name="joint soft-KG penalized estimate",
        )
        costs = np.asarray(total_costs, dtype=np.float64)
        scores = _checked_nonnegative_ratios(
            np.maximum(penalized, 0.0),
            costs,
            name="joint soft-KG score",
        )
        for array in (estimates, errors, penalized, costs, scores):
            array.setflags(write=False)
        return JointSoftKGResult(
            evaluation_batches=groups,
            estimate=estimates,
            standard_error=errors,
            standard_error_penalized_estimate=penalized,
            score=scores,
            total_cost=costs,
        )

    def select_joint_beam(
        self,
        belief: JointGaussianPosterior,
        batch: EvaluationBatch,
        *,
        batch_size: int,
        beam_width: int = 4,
        max_total_cost: float | None = None,
        max_groups_scored: int = 1024,
        fixed_budget: bool = False,
    ) -> BeamJointSoftKGResult:
        """Select a joint batch with deterministic bounded-width beam search.

        At each depth, all unique one-candidate extensions of the retained
        frontier are scored jointly with common frozen fantasies.  Rankings use
        descending score followed by the canonical group tuple, making exact
        ties independent of candidate input order.  ``beam_width=1`` is the
        corresponding deterministic greedy search.  Partial groups are not
        required to have positive value, because information synergy may only
        emerge after later candidates are added; only the final selection uses
        the positive-score/no-action rule.  If the total scoring cap cannot
        cover an entire generated depth, the search fails closed instead of
        screening a candidate-index-dependent prefix.

        ``fixed_budget=True`` prospectively removes the no-action option and
        ranks signed uncertainty-penalized gain per cost at EVERY depth. It
        retains the original clipped scores as diagnostics; a nonpositive
        estimate is never relabeled evidence of positive information gain.
        """

        if type(fixed_budget) is not bool:
            raise ValueError("fixed_budget must be an explicit boolean")
        for value, name in (
            (batch_size, "batch_size"),
            (beam_width, "beam_width"),
            (max_groups_scored, "max_groups_scored"),
        ):
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise ValueError(f"{name} must be an integer")
        parsed_batch_size = int(batch_size)
        parsed_beam_width = int(beam_width)
        parsed_max_groups_scored = int(max_groups_scored)
        if parsed_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if parsed_beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if parsed_max_groups_scored <= 0:
            raise ValueError("max_groups_scored must be positive")
        if parsed_max_groups_scored < parsed_batch_size:
            raise ValueError("max_groups_scored must allow at least one group per depth")

        parsed_max_total_cost: float | None = None
        if max_total_cost is not None:
            if isinstance(max_total_cost, bool) or not isinstance(
                max_total_cost,
                int | float | np.integer | np.floating,
            ):
                raise ValueError("max_total_cost must be finite and positive")
            parsed_max_total_cost = float(max_total_cost)
            if not np.isfinite(parsed_max_total_cost) or parsed_max_total_cost <= 0:
                raise ValueError("max_total_cost must be finite and positive")

        self._validate_indices(belief, batch)
        candidate_positions = {index: position for position, index in enumerate(batch.indices)}
        eligible_candidates = tuple(
            sorted(index for index in batch.indices if batch.eligible[candidate_positions[index]])
        )
        if parsed_batch_size > len(eligible_candidates):
            raise ValueError("batch_size cannot exceed the number of eligible candidates")

        def group_total_cost(group: tuple[int, ...]) -> float:
            positions = [candidate_positions[index] for index in group]
            return _checked_positive_sum(
                batch.costs[positions],
                name="joint evaluation total cost",
            )

        def can_complete(group: tuple[int, ...], total_cost: float, depth: int) -> bool:
            remaining_slots = parsed_batch_size - depth
            if remaining_slots == 0:
                return parsed_max_total_cost is None or total_cost <= parsed_max_total_cost
            remaining_candidates = [index for index in eligible_candidates if index not in group]
            if len(remaining_candidates) < remaining_slots:
                return False
            if parsed_max_total_cost is None:
                return True
            cheapest_positions = sorted(
                (candidate_positions[index] for index in remaining_candidates),
                key=lambda position: (batch.costs[position], batch.indices[position]),
            )[:remaining_slots]
            completion_cost = _checked_positive_sum(
                np.concatenate(
                    (
                        np.asarray([total_cost], dtype=np.float64),
                        batch.costs[cheapest_positions],
                    )
                ),
                name="joint evaluation completion cost",
            )
            return completion_cost <= parsed_max_total_cost

        frontier: tuple[tuple[int, ...], ...] = ((),)
        traces: list[BeamSearchDepthTrace] = []
        total_groups_scored = 0
        any_beam_pruning = False

        for depth in range(1, parsed_batch_size + 1):
            expansions = {
                tuple(sorted((*group, candidate)))
                for group in frontier
                for candidate in eligible_candidates
                if candidate not in group
            }
            generated_groups = tuple(sorted(expansions))
            feasible_groups = tuple(
                group
                for group in generated_groups
                if can_complete(group, group_total_cost(group), depth)
            )
            if not feasible_groups:
                raise ValueError("no eligible evaluation batch satisfies max_total_cost")

            remaining_depths = parsed_batch_size - depth
            available_now = parsed_max_groups_scored - total_groups_scored - remaining_depths
            if available_now <= 0:
                raise RuntimeError("joint beam search exhausted its reserved depth budget")
            if len(feasible_groups) > available_now:
                raise ValueError(
                    "max_groups_scored is insufficient for the generated frontier; "
                    "joint beam search refuses index-biased truncation"
                )
            scored_groups = feasible_groups
            scored = self.score_joint_groups(
                belief,
                batch,
                scored_groups,
                max_total_cost=parsed_max_total_cost,
                max_groups=available_now,
            )
            total_groups_scored += len(scored.evaluation_batches)

            ranked_positions = (
                _signed_joint_ranking(scored)
                if fixed_budget
                else sorted(
                    range(len(scored.evaluation_batches)),
                    key=lambda position: (
                        -float(scored.score[position]),
                        scored.evaluation_batches[position],
                    ),
                )
            )
            retained_count = min(parsed_beam_width, len(ranked_positions))
            frontier = tuple(
                scored.evaluation_batches[position]
                for position in ranked_positions[:retained_count]
            )
            beam_pruned_count = (
                max(0, len(scored.evaluation_batches) - retained_count)
                if depth < parsed_batch_size
                else 0
            )
            any_beam_pruning |= beam_pruned_count > 0
            traces.append(
                BeamSearchDepthTrace(
                    depth=depth,
                    generated_group_count=len(generated_groups),
                    completion_feasible_group_count=len(feasible_groups),
                    scored=scored,
                    retained_evaluation_batches=frontier,
                    beam_pruned_group_count=beam_pruned_count,
                    remaining_group_budget=parsed_max_groups_scored - total_groups_scored,
                )
            )

        approximation_status = "beam_pruned" if any_beam_pruning else "exact_frontier_covered"
        result_type = FixedBudgetBeamJointSoftKGResult if fixed_budget else BeamJointSoftKGResult
        return result_type(
            batch_size=parsed_batch_size,
            beam_width=parsed_beam_width,
            max_groups_scored=parsed_max_groups_scored,
            total_groups_scored=total_groups_scored,
            approximation_status=approximation_status,
            depth_trace=tuple(traces),
        )

    def _joint_estimates(
        self,
        belief: JointGaussianPosterior,
        evaluation_groups: tuple[tuple[int, ...], ...],
    ) -> tuple[FloatArray, FloatArray]:
        """Return joint KG moments with chunk-boundary-invariant fantasies."""

        if not evaluation_groups:
            raise ValueError("evaluation_groups cannot be empty")
        evaluation_groups = tuple(tuple(sorted(group)) for group in evaluation_groups)
        group_size = len(evaluation_groups[0])
        if group_size == 0 or any(len(group) != group_size for group in evaluation_groups):
            raise ValueError("evaluation groups must have one common positive size")

        n_outputs = belief.n_outputs
        decisions = np.asarray(self.problem.decision_indices, dtype=np.int64)
        observed_outputs = np.asarray(
            tuple(range(n_outputs)) if self.observed_outputs is None else self.observed_outputs,
            dtype=np.int64,
        )
        flat_covariance = belief.covariance.reshape(
            belief.n_points * n_outputs, belief.n_points * n_outputs
        )
        decision_flat = np.asarray(
            [point * n_outputs + output for point in decisions for output in range(n_outputs)],
            dtype=np.int64,
        )
        decision_mean = belief.mean[decisions].reshape(-1)
        decision_covariance = flat_covariance[np.ix_(decision_flat, decision_flat)]
        current_value = float(
            self._preference_averaged_values(
                decision_mean.reshape(1, len(decisions), n_outputs),
                np.diag(decision_covariance).reshape(len(decisions), n_outputs),
            )[0]
        )

        common_normals = np.random.Generator(np.random.PCG64(self.seed)).standard_normal(
            (self.n_fantasies, group_size * len(observed_outputs))
        )
        estimates = np.empty(len(evaluation_groups), dtype=np.float64)
        errors = np.empty_like(estimates)
        candidate_chunk = min(
            self.candidate_chunk_size or len(evaluation_groups),
            len(evaluation_groups),
        )
        fantasy_chunk = min(self.fantasy_chunk_size or self.n_fantasies, self.n_fantasies)
        n_decision_entries = len(decision_flat)
        n_observation_entries = group_size * len(observed_outputs)

        for candidate_start in range(0, len(evaluation_groups), candidate_chunk):
            candidate_stop = min(len(evaluation_groups), candidate_start + candidate_chunk)
            group_chunk = evaluation_groups[candidate_start:candidate_stop]
            n_candidates = len(group_chunk)
            gains = np.empty(
                (n_candidates, n_decision_entries, n_observation_entries),
                dtype=np.float64,
            )
            square_roots = np.empty(
                (n_candidates, n_observation_entries, n_observation_entries),
                dtype=np.float64,
            )
            posterior_variances = np.empty(
                (n_candidates, len(decisions), n_outputs),
                dtype=np.float64,
            )

            for local_position, group in enumerate(group_chunk):
                observation_flat = np.asarray(
                    [point * n_outputs + output for point in group for output in observed_outputs],
                    dtype=np.int64,
                )
                latent_observation_covariance = flat_covariance[
                    np.ix_(observation_flat, observation_flat)
                ].copy()
                n_observed_outputs = len(observed_outputs)
                observation_noise = np.zeros_like(latent_observation_covariance)
                for group_position, point in enumerate(group):
                    start = group_position * n_observed_outputs
                    stop = start + n_observed_outputs
                    noise_block = belief.observation_noise[point][
                        np.ix_(observed_outputs, observed_outputs)
                    ]
                    observation_noise[start:stop, start:stop] = noise_block
                observation_covariance = latent_observation_covariance + observation_noise
                cross_covariance = flat_covariance[np.ix_(decision_flat, observation_flat)]
                solved_cross, square_root = self._solve_observation_covariance(
                    observation_covariance,
                    cross_covariance.T,
                )
                gain = solved_cross.T
                exact_decision_positions = tuple(
                    position
                    for position, flat_index in enumerate(decision_flat)
                    if any(
                        flat_index == observation_flat[observation_position]
                        and observation_noise[observation_position, observation_position] == 0.0
                        for observation_position in range(len(observation_flat))
                    )
                )
                posterior_covariance = psd_joseph_conditioned_covariance(
                    decision_covariance,
                    gain,
                    cross_covariance,
                    latent_observation_covariance,
                    observation_noise,
                    name="soft-KG conditioned decision covariance",
                    exact_zero_indices=exact_decision_positions,
                    collapse_factor_roundoff=True,
                )
                gains[local_position] = gain
                square_roots[local_position] = square_root
                posterior_variances[local_position] = np.diag(posterior_covariance).reshape(
                    len(decisions), n_outputs
                )

            fantasy_gains = np.empty((n_candidates, self.n_fantasies), dtype=np.float64)
            for fantasy_start in range(0, self.n_fantasies, fantasy_chunk):
                fantasy_stop = min(self.n_fantasies, fantasy_start + fantasy_chunk)
                normals = common_normals[fantasy_start:fantasy_stop]
                observation_deviation = np.einsum(
                    "fk,crk->cfr",
                    normals,
                    square_roots,
                    optimize=False,
                )
                mean_deviation = np.einsum(
                    "cfr,cdr->cfd",
                    observation_deviation,
                    gains,
                    optimize=False,
                )
                n_chunk_fantasies = fantasy_stop - fantasy_start
                fantasy_means = (decision_mean[None, None, :] + mean_deviation).reshape(
                    n_candidates * n_chunk_fantasies, len(decisions), n_outputs
                )
                fantasy_variances = np.broadcast_to(
                    posterior_variances[:, None, :, :],
                    (n_candidates, n_chunk_fantasies, len(decisions), n_outputs),
                ).reshape(n_candidates * n_chunk_fantasies, len(decisions), n_outputs)
                fantasy_values = self._preference_averaged_values(
                    fantasy_means,
                    fantasy_variances,
                ).reshape(n_candidates, n_chunk_fantasies)
                with np.errstate(over="ignore", invalid="ignore"):
                    chunk_gains = fantasy_values - current_value
                if np.any(~np.isfinite(chunk_gains)):
                    raise FloatingPointError("soft-KG fantasy gain is not representable")
                fantasy_gains[:, fantasy_start:fantasy_stop] = chunk_gains

            chunk_estimates, chunk_errors = _stable_sample_mean_and_standard_error(fantasy_gains)
            estimates[candidate_start:candidate_stop] = chunk_estimates
            errors[candidate_start:candidate_stop] = chunk_errors

        return estimates, errors

    def _preference_averaged_values(
        self,
        mean: FloatArray,
        variance: FloatArray,
    ) -> FloatArray:
        if mean.ndim != 3:
            raise ValueError("mean must have shape (n_fantasies, n_decisions, n_outputs)")
        feasible = self._chance_feasible(mean, variance)
        objective_mean = mean[:, :, self.problem.objective_outputs]
        utility = _checked_weighted_utilities(
            objective_mean,
            self.problem.preferences.weights,
        )
        masked = np.where(feasible[:, None, :], utility, -np.inf)
        maximum = np.max(masked, axis=2)
        centered_logits = _stable_centered_logits(
            masked,
            maximum[:, :, None],
            divisor=self.temperature,
            name="preference soft-value centered logits",
        )
        log_weighted_sum = logsumexp(
            self.problem.log_base_measure[None, None, :] + centered_logits,
            axis=2,
        )
        if np.any(~np.isfinite(log_weighted_sum)):
            raise FloatingPointError("soft-value normalization became non-finite")
        values = _checked_affine_float64(
            maximum,
            self.temperature,
            log_weighted_sum,
            name="preference soft value",
        )
        return _checked_log_weighted_average(
            values,
            self.problem.preferences.log_probabilities,
        )

    def _chance_feasible(self, mean: FloatArray, variance: FloatArray) -> BoolArray:
        if variance.ndim == 2:
            if variance.shape != mean.shape[1:]:
                raise ValueError("variance must match the decision and output dimensions")
        elif variance.ndim == 3:
            if variance.shape != mean.shape:
                raise ValueError("fantasy-specific variance must match mean")
        else:
            raise ValueError("variance must have two or three dimensions")
        feasible = np.ones(mean.shape[:2], dtype=bool)
        for constraint in self.problem.constraints:
            output_mean = mean[:, :, constraint.output_index]
            if variance.ndim == 2:
                output_variance = np.maximum(variance[:, constraint.output_index], 0.0)[None, :]
                output_variance = np.broadcast_to(output_variance, output_mean.shape)
            else:
                output_variance = np.maximum(variance[:, :, constraint.output_index], 0.0)
            constraint_satisfied = constraint.is_satisfied(
                output_mean,
                output_variance,
            )
            feasible &= constraint_satisfied
        safe_positions = {
            self.problem.decision_indices.index(index)
            for index in self.problem.always_safe_decisions
        }
        if safe_positions:
            feasible[:, list(safe_positions)] = True
        return feasible

    def _solve_observation_covariance(
        self,
        covariance: FloatArray,
        right_hand_side: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return a scale-safe PSD solve and its matching projected square root."""

        return solve_psd_with_projected_square_root(
            covariance,
            right_hand_side,
            relative_eigenvalue_cutoff=self.relative_eigenvalue_cutoff,
        )

    def _validate_indices(self, belief: JointGaussianPosterior, batch: EvaluationBatch) -> None:
        point_indices = (*self.problem.decision_indices, *batch.indices)
        if max(point_indices) >= belief.n_points:
            raise ValueError("decision and evaluation indices must refer to belief points")
        output_indices = [*self.problem.objective_outputs]
        output_indices.extend(constraint.output_index for constraint in self.problem.constraints)
        if self.observed_outputs is not None:
            output_indices.extend(self.observed_outputs)
        if max(output_indices) >= belief.n_outputs:
            raise ValueError("objective, constraint, and observed outputs must exist in belief")

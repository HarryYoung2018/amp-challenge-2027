"""Finite joint Gaussian posteriors for small acquisition frontiers.

The representation is deliberately dense.  If ``m = n_points * n_outputs``,
storage is ``O(m**2)`` and validation or coherent sampling uses an ``O(m**3)``
eigendecomposition.  The conditioning arithmetic includes
``O(q**3 + m * q**2 + m**2 * q)`` work, but validated returned posteriors and
Joseph-form checks also perform dense full-state eigendecompositions, so the
end-to-end worst case remains cubic in ``m + q``.  Exact-rational cancellation
fallbacks have input-dependent arbitrary-precision cost.  This is appropriate
for the audited small frontiers used by soft-KG, not for the 50k production
library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag, eigh

FloatArray = NDArray[np.float64]
MAX_RELATIVE_EIGENVALUE_CUTOFF = 1e-4


def _binary_fraction(value: float | np.floating) -> Fraction:
    """Return the exact rational represented by a binary floating scalar."""

    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _fraction_to_float64(value: Fraction, *, name: str) -> float:
    """Round an exact rational to float64 or reject range loss."""

    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is not representable in float64") from error
    if not np.isfinite(result) or (value != 0 and result == 0.0):
        raise ValueError(f"{name} is not representable in float64")
    return result


def _exact_schur_entry(
    parent: FloatArray,
    gain: FloatArray,
    cross: FloatArray,
    *,
    row: int,
    column: int,
    name: str,
) -> float:
    """Evaluate one ill-conditioned Schur entry exactly before rounding."""

    result = _binary_fraction(parent[row, column])
    for observation in range(gain.shape[1]):
        result -= _binary_fraction(gain[row, observation]) * _binary_fraction(
            cross[column, observation]
        )
    return _fraction_to_float64(result, name=name)


def _exact_joseph_entry(
    parent: FloatArray,
    gain: FloatArray,
    cross: FloatArray,
    latent: FloatArray,
    noise: FloatArray,
    *,
    row: int,
    column: int,
    name: str,
) -> float:
    """Evaluate one ill-conditioned Joseph-form entry exactly before rounding."""

    result = _binary_fraction(parent[row, column])
    observation_size = gain.shape[1]
    for observation in range(observation_size):
        result -= _binary_fraction(gain[row, observation]) * _binary_fraction(
            cross[column, observation]
        )
        result -= _binary_fraction(cross[row, observation]) * _binary_fraction(
            gain[column, observation]
        )
    for left_observation in range(observation_size):
        left_gain = _binary_fraction(gain[row, left_observation])
        for right_observation in range(observation_size):
            right_gain = _binary_fraction(gain[column, right_observation])
            result += (
                left_gain
                * _binary_fraction(latent[left_observation, right_observation])
                * right_gain
            )
            result += (
                left_gain
                * _binary_fraction(noise[left_observation, right_observation])
                * right_gain
            )
    return _fraction_to_float64(result, name=name)


def _exact_noise_entry(
    gain: FloatArray,
    noise: FloatArray,
    *,
    row: int,
    column: int,
) -> Fraction:
    """Return one exact propagated-noise covariance entry."""

    result = Fraction(0)
    for left_observation in range(gain.shape[1]):
        left_gain = _binary_fraction(gain[row, left_observation])
        for right_observation in range(gain.shape[1]):
            result += (
                left_gain
                * _binary_fraction(noise[left_observation, right_observation])
                * _binary_fraction(gain[column, right_observation])
            )
    return result


def _roundoff_tolerance(*, scale: float, size: int) -> float:
    return 128.0 * np.finfo(np.float64).eps * scale * size


def _relative_eigen_tolerance(*, scale: float, size: int) -> float:
    return 128.0 * np.finfo(np.float64).eps * scale * size


def _validate_relative_eigenvalue_cutoff(value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > MAX_RELATIVE_EIGENVALUE_CUTOFF:
        raise ValueError(
            "relative_eigenvalue_cutoff must be finite and lie in "
            f"(0, {MAX_RELATIVE_EIGENVALUE_CUTOFF}]"
        )
    return parsed


def _readonly_float_array(value: object, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _checked_symmetric_average(matrix: FloatArray, *, name: str) -> FloatArray:
    """Symmetrize without erasing subnormal entries by halving first."""

    values = np.asarray(matrix, dtype=np.float64)
    if np.array_equal(values, values.T):
        return np.array(values, dtype=np.float64, copy=True)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        extended = (
            np.asarray(values, dtype=np.longdouble) + np.asarray(values.T, dtype=np.longdouble)
        ) / np.longdouble(2.0)
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if np.any(~np.isfinite(extended)) or np.any(np.abs(extended) > float64_limit):
        raise ValueError(f"{name} symmetric average is not representable in float64")
    result = np.asarray(extended, dtype=np.float64)
    if np.any((extended != 0.0) & (result == 0.0)):
        raise ValueError(f"{name} symmetric average is not representable in float64")
    return result


def _symmetric_psd(
    matrix: ArrayLike,
    *,
    name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    diagonal = np.diag(values)
    if np.any(diagonal < 0.0):
        raise ValueError(f"{name} must be positive semidefinite")
    zero_diagonal = diagonal == 0.0
    if np.any(values[zero_diagonal, :] != 0.0) or np.any(values[:, zero_diagonal] != 0.0):
        raise ValueError(f"{name} must be positive semidefinite")
    positive = np.flatnonzero(~zero_diagonal)
    symmetric = _checked_symmetric_average(values, name=name)
    if len(positive):
        diagonal_scale = np.sqrt(diagonal[positive])
        raw_block = values[np.ix_(positive, positive)]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            correlation = (raw_block / diagonal_scale[:, None]) / diagonal_scale[None, :]
        if np.any(~np.isfinite(correlation)):
            raise ValueError(f"{name} is not representable after diagonal equilibration")
        correlation_scale = float(np.max(np.abs(correlation)))
        correlation_tolerance = _roundoff_tolerance(
            scale=correlation_scale,
            size=len(positive),
        )
        if not np.allclose(
            correlation,
            correlation.T,
            rtol=1e-10,
            atol=correlation_tolerance,
        ):
            raise ValueError(f"{name} must be symmetric")
        symmetric_correlation = _checked_symmetric_average(
            correlation,
            name=f"{name} equilibrated correlation",
        )
        eigenvalues = eigh(
            symmetric_correlation,
            eigvals_only=True,
            check_finite=False,
        )
        spectral_scale = float(np.max(np.abs(eigenvalues)))
        negative_tolerance = _roundoff_tolerance(
            scale=spectral_scale,
            size=len(positive),
        )
        if float(eigenvalues.min()) < -negative_tolerance:
            raise ValueError(f"{name} must be positive semidefinite")
    return np.asarray(symmetric, dtype=np.float64)


def _equilibrated_eigensystem(
    matrix: FloatArray,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray, FloatArray, float]:
    """Factor a validated covariance in unit-diagonal correlation space."""

    diagonal = np.diag(matrix)
    positive = np.flatnonzero(diagonal > 0.0)
    diagonal_scale = np.sqrt(diagonal[positive])
    if not len(positive):
        return (
            positive,
            diagonal_scale,
            np.empty(0, dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            0.0,
        )
    block = matrix[np.ix_(positive, positive)]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        correlation = (block / diagonal_scale[:, None]) / diagonal_scale[None, :]
    correlation = _checked_symmetric_average(
        correlation,
        name="equilibrated covariance correlation",
    )
    eigenvalues, eigenvectors = eigh(correlation, check_finite=False)
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    return positive, diagonal_scale, eigenvalues, eigenvectors, spectral_scale


def _pivoted_cholesky_factor(
    matrix: FloatArray,
    *,
    rank: int,
) -> tuple[FloatArray, tuple[int, ...]] | None:
    """Factor a resolved PSD support while preserving structural sparsity."""

    size = matrix.shape[0]
    factor = np.zeros((size, size), dtype=np.float64)
    if rank == 0:
        return factor, ()
    residual_diagonal = np.array(np.diag(matrix), dtype=np.float64, copy=True)
    available = np.ones(size, dtype=bool)
    pivots: list[int] = []
    scale = float(np.max(np.abs(matrix)))
    tolerance = _roundoff_tolerance(scale=scale, size=size) * 4.0
    for column in range(rank):
        maximum = float(np.max(residual_diagonal[available]))
        if not np.isfinite(maximum) or maximum <= 0.0:
            return None
        near_maximum = np.flatnonzero(available & (residual_diagonal >= maximum - tolerance))
        if not len(near_maximum):
            return None
        pivot = int(near_maximum[0])
        pivots.append(pivot)
        pivot_root = float(np.sqrt(maximum))
        factor[pivot, column] = pivot_root
        for row in np.flatnonzero(available):
            row_index = int(row)
            if row_index == pivot:
                continue
            correction = fsum(
                float(factor[row_index, prior] * factor[pivot, prior]) for prior in range(column)
            )
            factor[row_index, column] = (matrix[row_index, pivot] - correction) / pivot_root
        available[pivot] = False
        for row in np.flatnonzero(available):
            row_index = int(row)
            explained = fsum(float(factor[row_index, prior] ** 2) for prior in range(column + 1))
            residual = float(matrix[row_index, row_index] - explained)
            if residual < -tolerance:
                return None
            residual_diagonal[row_index] = max(residual, 0.0)
        residual_diagonal[pivot] = 0.0
    if np.any(~np.isfinite(factor)):
        return None
    reconstructed = factor @ factor.T
    if not np.allclose(
        reconstructed,
        matrix,
        rtol=2048.0 * np.finfo(np.float64).eps * size,
        atol=tolerance,
    ):
        return None
    return factor, tuple(pivots)


def _equilibrated_covariance_factor(matrix: FloatArray) -> FloatArray:
    """Return ``L`` with ``L @ L.T`` equal to the roundoff-projected covariance."""

    factor = np.zeros_like(matrix, dtype=np.float64)
    positive, diagonal_scale, eigenvalues, eigenvectors, spectral_scale = _equilibrated_eigensystem(
        matrix
    )
    if not len(positive):
        return factor
    tolerance = _relative_eigen_tolerance(
        scale=spectral_scale,
        size=len(eigenvalues),
    )
    retained = np.where(eigenvalues > tolerance, eigenvalues, 0.0)
    block_factor = diagonal_scale[:, None] * (eigenvectors * np.sqrt(retained))
    factor[np.ix_(positive, positive)] = block_factor
    return factor


def psd_schur_complement(
    parent_covariance: ArrayLike,
    gain: ArrayLike,
    cross_covariance: ArrayLike,
    *,
    name: str = "conditioned covariance",
    exact_zero_indices: Sequence[int] = (),
    collapse_positive_roundoff: bool = False,
) -> FloatArray:
    """Return a validated Schur complement without inventing covariance support."""

    parent = np.asarray(parent_covariance, dtype=np.float64)
    gain_values = np.asarray(gain, dtype=np.float64)
    cross = np.asarray(cross_covariance, dtype=np.float64)
    if parent.ndim != 2 or parent.shape[0] == 0 or parent.shape[0] != parent.shape[1]:
        raise ValueError("parent_covariance must be a non-empty square matrix")
    if gain_values.ndim != 2 or gain_values.shape[0] != parent.shape[0]:
        raise ValueError("gain must have one row per parent covariance entry")
    if cross.shape != gain_values.shape:
        raise ValueError("cross_covariance must match gain")
    if any(np.any(~np.isfinite(value)) for value in (parent, gain_values, cross)):
        raise ValueError("Schur-complement inputs must contain only finite values")
    exact_zeros = (
        _indices(
            exact_zero_indices,
            size=parent.shape[0],
            name="exact_zero_indices",
            unique=True,
        )
        if exact_zero_indices
        else ()
    )
    if not isinstance(collapse_positive_roundoff, bool):
        raise TypeError("collapse_positive_roundoff must be a bool")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        extended_gain = np.asarray(gain_values, dtype=np.longdouble)
        extended_cross = np.asarray(cross, dtype=np.longdouble)
        extended_update = extended_gain @ extended_cross.T
        extended_parent = np.asarray(parent, dtype=np.longdouble)
        extended_conditioned = extended_parent - extended_update
        absolute_accumulation = np.abs(extended_parent) + (
            np.abs(extended_gain) @ np.abs(extended_cross).T
        )
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if (
        np.any(~np.isfinite(extended_update))
        or np.any(np.abs(extended_update) > float64_limit)
        or np.any(~np.isfinite(extended_conditioned))
        or np.any(np.abs(extended_conditioned) > float64_limit)
    ):
        raise ValueError(f"{name} is not representable in float64")
    update = np.asarray(extended_update, dtype=np.float64)
    raw_conditioned = np.asarray(extended_conditioned, dtype=np.float64)
    error_factor = np.longdouble(8 * (gain_values.shape[1] + 2)) * np.longdouble(
        np.finfo(np.longdouble).eps
    )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cancellation_bound = error_factor * absolute_accumulation
    suspicious = (absolute_accumulation > 0.0) & (
        np.abs(extended_conditioned) <= cancellation_bound
    )
    if np.any(suspicious):
        raw_conditioned = np.array(raw_conditioned, dtype=np.float64, copy=True)
        for row, column in np.argwhere(suspicious):
            raw_conditioned[row, column] = _exact_schur_entry(
                parent,
                gain_values,
                cross,
                row=int(row),
                column=int(column),
                name=name,
            )
    if np.any((~suspicious) & (extended_conditioned != 0.0) & (raw_conditioned == 0.0)):
        raise ValueError(f"{name} is not representable in float64")
    operation_scale = np.maximum.reduce(
        (np.abs(parent), np.abs(update), np.abs(parent.T), np.abs(update.T))
    )
    symmetry_tolerance = 512.0 * np.finfo(np.float64).eps * parent.shape[0] * operation_scale
    with np.errstate(over="ignore", invalid="ignore"):
        asymmetry = np.abs(raw_conditioned - raw_conditioned.T)
    if np.any(~np.isfinite(asymmetry)) or np.any(asymmetry > symmetry_tolerance):
        raise ValueError(f"{name} must be symmetric")
    conditioned = _checked_symmetric_average(
        raw_conditioned,
        name=name,
    )

    # Clean only a mathematically declared collapse, a non-positive roundoff
    # residual, or (when the caller has proved an entirely noiseless update) a
    # positive roundoff residual. A small positive posterior variance after a
    # noisy observation is real uncertainty and must never be zeroed merely
    # because the prior and update were large.
    parent_diagonal = np.abs(np.diag(parent))
    update_diagonal = np.abs(np.diag(update))
    local_scale = np.maximum(parent_diagonal, update_diagonal)
    collapse_tolerance = 512.0 * np.finfo(np.float64).eps * parent.shape[0] * local_scale
    diagonal = np.diag(conditioned)
    declared = np.zeros(parent.shape[0], dtype=bool)
    declared[list(exact_zeros)] = True
    collapsed = declared | (
        (np.abs(diagonal) <= collapse_tolerance) & ((diagonal <= 0.0) | collapse_positive_roundoff)
    )
    if np.any(collapsed):
        element_scale = np.maximum(np.abs(parent), np.abs(update))
        element_tolerance = 512.0 * np.finfo(np.float64).eps * parent.shape[0] * element_scale
        collapsible = collapsed & np.all(
            np.abs(conditioned) <= element_tolerance,
            axis=1,
        )
        if np.any(declared & ~collapsible):
            raise ValueError(f"{name} violates a declared exact covariance collapse")
        conditioned[collapsible, :] = 0.0
        conditioned[:, collapsible] = 0.0
    return _symmetric_psd(
        conditioned,
        name=name,
        expected_shape=parent.shape,
    )


def psd_joseph_conditioned_covariance(
    parent_covariance: ArrayLike,
    gain: ArrayLike,
    cross_covariance: ArrayLike,
    latent_observation_covariance: ArrayLike,
    observation_noise: ArrayLike,
    *,
    name: str = "conditioned covariance",
    exact_zero_indices: Sequence[int] = (),
    collapse_factor_roundoff: bool = False,
) -> FloatArray:
    """Return a factorized Joseph-form conditional covariance.

    The usual Schur subtraction can erase a positive observation-noise scale
    when adding that noise to a much larger latent variance rounds back to the
    latent variance.  Factoring the joint latent covariance and the noise
    separately computes

    ``Cov(x - K y) + K R K.T``

    without subtracting nearly equal covariance matrices.  This retains any
    representable positive noise while remaining valid for singular latent
    covariances and exact observations.
    """

    parent = np.asarray(parent_covariance, dtype=np.float64)
    gain_values = np.asarray(gain, dtype=np.float64)
    cross = np.asarray(cross_covariance, dtype=np.float64)
    latent = np.asarray(latent_observation_covariance, dtype=np.float64)
    noise = np.asarray(observation_noise, dtype=np.float64)
    if parent.ndim != 2 or parent.shape[0] == 0 or parent.shape[0] != parent.shape[1]:
        raise ValueError("parent_covariance must be a non-empty square matrix")
    if gain_values.ndim != 2 or gain_values.shape[0] != parent.shape[0]:
        raise ValueError("gain must have one row per parent covariance entry")
    observation_size = gain_values.shape[1]
    if observation_size == 0:
        raise ValueError("gain must have at least one observation column")
    if cross.shape != gain_values.shape:
        raise ValueError("cross_covariance must match gain")
    if latent.shape != (observation_size, observation_size):
        raise ValueError("latent_observation_covariance must match the gain columns")
    if noise.shape != latent.shape:
        raise ValueError("observation_noise must match latent_observation_covariance")
    if any(np.any(~np.isfinite(value)) for value in (parent, gain_values, cross, latent, noise)):
        raise ValueError("Joseph-form inputs must contain only finite values")
    if not isinstance(collapse_factor_roundoff, bool):
        raise TypeError("collapse_factor_roundoff must be a bool")
    exact_zeros = (
        _indices(
            exact_zero_indices,
            size=parent.shape[0],
            name="exact_zero_indices",
            unique=True,
        )
        if exact_zero_indices
        else ()
    )

    joint = np.block([[parent, cross], [cross.T, latent]])
    joint = _symmetric_psd(
        joint,
        name="joint latent covariance",
        expected_shape=joint.shape,
    )
    validated_noise = _symmetric_psd(
        noise,
        name="observation noise covariance",
        expected_shape=noise.shape,
    )
    state_size = parent.shape[0]
    validated_parent = np.asarray(joint[:state_size, :state_size], dtype=np.float64)
    validated_cross = np.asarray(joint[:state_size, state_size:], dtype=np.float64)
    validated_latent = np.asarray(joint[state_size:, state_size:], dtype=np.float64)
    latent_factor = _equilibrated_covariance_factor(joint)
    noise_factor = _equilibrated_covariance_factor(validated_noise)
    transform = np.concatenate(
        (np.eye(parent.shape[0], dtype=np.float64), -gain_values),
        axis=1,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        residual_factor = transform @ latent_factor
        propagated_noise_factor = gain_values @ noise_factor
        latent_factor_conditioned = residual_factor @ residual_factor.T
        noise_factor_conditioned = propagated_noise_factor @ propagated_noise_factor.T
        conditioned = latent_factor_conditioned + noise_factor_conditioned
    if np.any(~np.isfinite(conditioned)):
        raise ValueError(f"{name} is not representable in float64")
    factor_conditioned = np.asarray(conditioned, dtype=np.float64)
    # A factor row whose every component is bounded by its dot-product
    # roundoff has no resolved conditional support. Internal callers whose
    # gain came from this module's projected solve may declare that algebra;
    # arbitrary public gains preserve exact rational residuals by default.
    with np.errstate(over="ignore", invalid="ignore"):
        latent_accumulation = np.abs(transform) @ np.abs(latent_factor)
        noise_accumulation = np.abs(gain_values) @ np.abs(noise_factor)
        factor_tolerance_scale = (
            512.0 * np.finfo(np.float64).eps * (parent.shape[0] + observation_size)
        )
        latent_roundoff = factor_tolerance_scale * latent_accumulation
        noise_roundoff = factor_tolerance_scale * noise_accumulation
    latent_roundoff_only = np.all(
        np.isfinite(latent_roundoff) & (np.abs(residual_factor) <= latent_roundoff),
        axis=1,
    )
    noise_roundoff_only = np.all(
        np.isfinite(noise_roundoff) & (np.abs(propagated_noise_factor) <= noise_roundoff),
        axis=1,
    )
    roundoff_only = latent_roundoff_only & noise_roundoff_only
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        extended_gain = np.asarray(gain_values, dtype=np.longdouble)
        extended_parent = np.asarray(validated_parent, dtype=np.longdouble)
        extended_cross = np.asarray(validated_cross, dtype=np.longdouble)
        extended_latent = np.asarray(validated_latent, dtype=np.longdouble)
        extended_noise = np.asarray(validated_noise, dtype=np.longdouble)
        left_cross_update = extended_gain @ extended_cross.T
        right_cross_update = extended_cross @ extended_gain.T
        latent_update = (extended_gain @ extended_latent) @ extended_gain.T
        noise_update = (extended_gain @ extended_noise) @ extended_gain.T
        direct_conditioned = (
            extended_parent - left_cross_update - right_cross_update + latent_update + noise_update
        )
        absolute_accumulation = (
            np.abs(extended_parent)
            + np.abs(extended_gain) @ np.abs(extended_cross).T
            + np.abs(extended_cross) @ np.abs(extended_gain).T
            + (np.abs(extended_gain) @ np.abs(extended_latent)) @ np.abs(extended_gain).T
            + (np.abs(extended_gain) @ np.abs(extended_noise)) @ np.abs(extended_gain).T
        )
    operation_count = 2 + 2 * observation_size + 2 * observation_size**2
    error_factor = np.longdouble(16 * operation_count) * np.longdouble(np.finfo(np.longdouble).eps)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cancellation_bound = error_factor * absolute_accumulation
    suspicious = (absolute_accumulation > 0.0) & (np.abs(direct_conditioned) <= cancellation_bound)
    float64_limit = np.longdouble(np.finfo(np.float64).max)
    if np.any(~np.isfinite(direct_conditioned)) or np.any(
        np.abs(direct_conditioned) > float64_limit
    ):
        raise ValueError(f"{name} is not representable in float64")
    conditioned = np.asarray(direct_conditioned, dtype=np.float64)
    if np.any((~suspicious) & (direct_conditioned != 0.0) & (conditioned == 0.0)):
        raise ValueError(f"{name} is not representable in float64")
    if exact_zeros:
        declared_entries = np.zeros_like(suspicious, dtype=bool)
        declared_entries[list(exact_zeros), :] = True
        declared_entries[:, list(exact_zeros)] = True
        suspicious &= ~declared_entries
    exactly_resolved = np.zeros_like(suspicious, dtype=bool)
    if collapse_factor_roundoff:
        # The caller has supplied a gain from this module's projected PSD
        # solve.  In a singular noiseless update, mixing ordinary Joseph
        # entries with factor-reconstructed cancellation entries can be
        # elementwise accurate yet globally indefinite after diagonal
        # equilibration.  Keep one coherent Gram representation for the full
        # matrix; the exact-noise refinement below still preserves a resolved
        # positive noise contribution at float64 boundaries.
        conditioned = np.array(factor_conditioned, dtype=np.float64, copy=True)
        structurally_unaffected = np.all(gain_values == 0.0, axis=1) & np.all(
            validated_cross == 0.0,
            axis=1,
        )
        if np.any(structurally_unaffected):
            # An exactly zero observation cross-covariance makes this row and
            # column algebraically identical to the parent, without a
            # cancellation. Restore those entries bit-for-bit so conditioning
            # is idempotent and does not perturb an independent tiny scale.
            conditioned[structurally_unaffected, :] = validated_parent[
                structurally_unaffected,
                :,
            ]
            conditioned[:, structurally_unaffected] = validated_parent[
                :,
                structurally_unaffected,
            ]
    if np.any(suspicious):
        conditioned = np.array(conditioned, dtype=np.float64, copy=True)
        if collapse_factor_roundoff:
            if np.any(validated_noise != 0.0):
                for row, column in np.argwhere(suspicious):
                    exact_noise = _exact_noise_entry(
                        gain_values,
                        validated_noise,
                        row=int(row),
                        column=int(column),
                    )
                    if exact_noise != 0:
                        latent_component = (
                            Fraction(0)
                            if latent_roundoff_only[row] or latent_roundoff_only[column]
                            else _binary_fraction(latent_factor_conditioned[row, column])
                        )
                        combined = latent_component + exact_noise
                        conditioned[row, column] = _fraction_to_float64(
                            combined,
                            name=name,
                        )
        else:
            for row, column in np.argwhere(suspicious):
                exact_value = _exact_joseph_entry(
                    validated_parent,
                    gain_values,
                    validated_cross,
                    validated_latent,
                    validated_noise,
                    row=int(row),
                    column=int(column),
                    name=name,
                )
                conditioned[row, column] = exact_value
                exactly_resolved[row, column] = exact_value != 0.0
    conditioned = _checked_symmetric_average(conditioned, name=name)
    resolved_rows = np.any(exactly_resolved, axis=1) | np.any(
        exactly_resolved,
        axis=0,
    )
    if collapse_factor_roundoff:
        collapsed = np.array(roundoff_only, copy=True)
    else:
        collapsed = np.array(roundoff_only & ~resolved_rows, copy=True)
    collapsed[list(exact_zeros)] = True
    if np.any(collapsed):
        conditioned[collapsed, :] = 0.0
        conditioned[:, collapsed] = 0.0
    return _symmetric_psd(
        conditioned,
        name=name,
        expected_shape=parent.shape,
    )


def _indices(
    values: Sequence[int],
    *,
    size: int,
    name: str,
    unique: bool,
) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{name} must contain integers")
        index = int(value)
        if index < 0 or index >= size:
            raise ValueError(f"{name} must refer to existing entries")
        parsed.append(index)
    if unique and len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must be unique")
    return tuple(parsed)


def _supported_eigen_projection(
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    right_hand_side: FloatArray,
    *,
    relative_eigenvalue_cutoff: float,
) -> tuple[NDArray[np.bool_], FloatArray, FloatArray, float]:
    """Project scaled right sides onto a retained correlation eigenspace."""

    relative_eigenvalue_cutoff = _validate_relative_eigenvalue_cutoff(relative_eigenvalue_cutoff)
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    negative_tolerance = _roundoff_tolerance(
        scale=spectral_scale,
        size=len(eigenvalues),
    )
    if float(eigenvalues.min()) < -negative_tolerance:
        raise ValueError("predictive observation covariance must be positive semidefinite")
    cutoff = max(relative_eigenvalue_cutoff * spectral_scale, negative_tolerance)
    retained = eigenvalues > cutoff
    column_scale = np.max(np.abs(right_hand_side), axis=0)
    safe_column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    scaled_right = right_hand_side / safe_column_scale[None, :]
    if np.any((right_hand_side != 0.0) & (scaled_right == 0.0)):
        raise ValueError("scaled PSD right-hand side is not representable")
    projected = eigenvectors[:, retained].T @ scaled_right
    reconstruction = eigenvectors[:, retained] @ projected
    residual = scaled_right - reconstruction
    residual_scale = np.max(np.abs(scaled_right), axis=0)
    residual_norm = np.max(np.abs(residual), axis=0)
    # Affine-support validation is a floating-point question, not permission
    # to absorb directions discarded by the caller's modeling rank cutoff.
    # Coupling the two lets a large cutoff accept arbitrary unsupported data.
    relative_support_tolerance = 256.0 * np.finfo(np.float64).eps * len(eigenvalues)
    if np.any(residual_norm > relative_support_tolerance * residual_scale):
        raise ValueError("values are incompatible with degenerate predictive covariance")

    return retained, projected, column_scale, spectral_scale


def solve_psd_with_projected_square_root(
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    *,
    relative_eigenvalue_cutoff: float,
) -> tuple[FloatArray, FloatArray]:
    """Return a support-consistent solve and its matching PSD square root.

    The rank cutoff is relative to the covariance spectral scale. Negative
    eigenvalues are tolerated only at float64 roundoff scale, independently of
    that modeling cutoff. The right-hand side must lie in the retained affine
    support, so a singular Gaussian cannot silently condition on an impossible
    observation or cross-covariance direction. Diagonal equilibration returns
    ``D^-1 R Lambda^-1 R.T D^-1 b`` on the retained correlation eigenspace. For
    a singular matrix that scale-selected representative is generally not the
    Euclidean Moore--Penrose solution of the original covariance. Public arrays
    and results are float64 even where an implementation uses ``longdouble``
    intermediates; NumPy's ``longdouble`` width is platform-ABI-dependent.
    """

    relative_eigenvalue_cutoff = _validate_relative_eigenvalue_cutoff(relative_eigenvalue_cutoff)
    covariance = np.asarray(matrix, dtype=np.float64)
    right = np.asarray(right_hand_side, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] == 0
        or covariance.shape[0] != covariance.shape[1]
    ):
        raise ValueError("matrix must be a non-empty square matrix")
    if right.ndim != 2 or right.shape[0] != covariance.shape[0]:
        raise ValueError("right_hand_side must be a matrix with one row per covariance dimension")
    if np.any(~np.isfinite(right)):
        raise ValueError("PSD solve inputs must contain only finite values")
    symmetric = _symmetric_psd(
        covariance,
        name="predictive observation covariance",
        expected_shape=covariance.shape,
    )
    solution = np.zeros_like(right, dtype=np.float64)
    square_root = np.zeros_like(symmetric, dtype=np.float64)
    positive, diagonal_scale, eigenvalues, eigenvectors, _ = _equilibrated_eigensystem(symmetric)
    zero = np.setdiff1d(np.arange(len(symmetric)), positive, assume_unique=True)
    if len(zero) and np.any(right[zero] != 0.0):
        raise ValueError("values are incompatible with degenerate predictive covariance")
    if len(positive):
        right_scale = np.max(np.abs(right), axis=0)
        safe_right_scale = np.maximum(right_scale, 1.0)
        scaled_right = right / safe_right_scale[None, :]
        if np.any((right != 0.0) & (scaled_right == 0.0)):
            raise ValueError("scaled PSD right-hand side is not representable")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            equilibrated_right = scaled_right[positive] / diagonal_scale[:, None]
        if np.any(~np.isfinite(equilibrated_right)):
            raise ValueError("equilibrated PSD right-hand side is not representable")
        if np.any((scaled_right[positive] != 0.0) & (equilibrated_right == 0.0)):
            raise ValueError("equilibrated PSD right-hand side is not representable")
        retained, projected, column_scale, spectral_scale = _supported_eigen_projection(
            eigenvalues,
            eigenvectors,
            equilibrated_right,
            relative_eigenvalue_cutoff=relative_eigenvalue_cutoff,
        )
        projected_values = np.where(retained, eigenvalues, 0.0)
        eigen_factor = eigenvectors * np.sqrt(projected_values)
        resolved_cutoff = _relative_eigen_tolerance(
            scale=spectral_scale,
            size=len(eigenvalues),
        )
        resolved = eigenvalues > resolved_cutoff
        factor = diagonal_scale[:, None] * eigen_factor
        sparse_factorization: tuple[FloatArray, tuple[int, ...]] | None = None
        if np.array_equal(retained, resolved):
            block = symmetric[np.ix_(positive, positive)]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                correlation = (block / diagonal_scale[:, None]) / diagonal_scale[None, :]
            sparse_factorization = _pivoted_cholesky_factor(
                correlation,
                rank=int(np.count_nonzero(retained)),
            )
            if sparse_factorization is not None:
                sparse_factor, _pivots = sparse_factorization
                factor = diagonal_scale[:, None] * sparse_factor
        square_root[np.ix_(positive, positive)] = factor
        if np.any(retained):
            retained_vectors = np.asarray(
                eigenvectors[:, retained],
                dtype=np.longdouble,
            )
            retained_values = np.asarray(
                eigenvalues[retained],
                dtype=np.longdouble,
            )
            normalized_right = np.asarray(equilibrated_right, dtype=np.longdouble) / np.asarray(
                np.where(column_scale > 0.0, column_scale, 1.0)[None, :],
                dtype=np.longdouble,
            )
            normalized_solution = retained_vectors @ (
                np.asarray(projected, dtype=np.longdouble) / retained_values[:, None]
            )
            correlation_extended = np.asarray(
                symmetric[np.ix_(positive, positive)],
                dtype=np.longdouble,
            ) / np.asarray(diagonal_scale[:, None], dtype=np.longdouble)
            correlation_extended = correlation_extended / np.asarray(
                diagonal_scale[None, :],
                dtype=np.longdouble,
            )
            for _iteration in range(3):
                residual = normalized_right - correlation_extended @ normalized_solution
                correction = retained_vectors @ (
                    (retained_vectors.T @ residual) / retained_values[:, None]
                )
                normalized_solution += correction
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                extended_solution = (
                    normalized_solution
                    * np.asarray(column_scale[None, :], dtype=np.longdouble)
                    * np.asarray(safe_right_scale[None, :], dtype=np.longdouble)
                    / np.asarray(diagonal_scale[:, None], dtype=np.longdouble)
                )
            float64_limit = np.longdouble(np.finfo(np.float64).max)
            if np.any(~np.isfinite(extended_solution)) or np.any(
                np.abs(extended_solution) > float64_limit
            ):
                raise ValueError("positive-semidefinite solve is not representable in float64")
            float_solution = np.asarray(extended_solution, dtype=np.float64)
            if np.any((extended_solution != 0.0) & (float_solution == 0.0)):
                raise ValueError("positive-semidefinite solve is not representable in float64")
            solution[positive] = float_solution
    if np.any(~np.isfinite(solution)):
        raise ValueError("positive-semidefinite solve is not representable in float64")
    return np.asarray(solution, dtype=np.float64), np.asarray(square_root, dtype=np.float64)


def _validate_psd_support(
    matrix: FloatArray,
    right_hand_side: FloatArray,
    *,
    rcond: float,
) -> None:
    """Reject components outside a degenerate Gaussian's affine support."""

    rcond = _validate_relative_eigenvalue_cutoff(rcond)
    raw_covariance = np.asarray(matrix, dtype=np.float64)
    if (
        raw_covariance.ndim != 2
        or raw_covariance.shape[0] == 0
        or raw_covariance.shape[0] != raw_covariance.shape[1]
    ):
        raise ValueError("matrix must be a non-empty square matrix")
    covariance = _symmetric_psd(
        raw_covariance,
        name="predictive observation covariance",
        expected_shape=raw_covariance.shape,
    )
    right = np.asarray(right_hand_side, dtype=np.float64)
    if right.ndim != 2 or right.shape[0] != covariance.shape[0]:
        raise ValueError("right_hand_side must be a matrix with one row per covariance dimension")
    if np.any(~np.isfinite(right)):
        raise ValueError("PSD solve inputs must contain only finite values")
    positive, diagonal_scale, eigenvalues, eigenvectors, _ = _equilibrated_eigensystem(covariance)
    zero = np.setdiff1d(np.arange(len(covariance)), positive, assume_unique=True)
    if len(zero) and np.any(right[zero] != 0.0):
        raise ValueError("values are incompatible with degenerate predictive covariance")
    if len(positive):
        right_scale = np.max(np.abs(right), axis=0)
        safe_right_scale = np.maximum(right_scale, 1.0)
        scaled_right = right / safe_right_scale[None, :]
        if np.any((right != 0.0) & (scaled_right == 0.0)):
            raise ValueError("scaled PSD right-hand side is not representable")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            equilibrated_right = scaled_right[positive] / diagonal_scale[:, None]
        if np.any(~np.isfinite(equilibrated_right)):
            raise ValueError("equilibrated PSD right-hand side is not representable")
        if np.any((scaled_right[positive] != 0.0) & (equilibrated_right == 0.0)):
            raise ValueError("equilibrated PSD right-hand side is not representable")
        _supported_eigen_projection(
            eigenvalues,
            eigenvectors,
            equilibrated_right,
            relative_eigenvalue_cutoff=rcond,
        )


def _psd_solve(matrix: FloatArray, right_hand_side: FloatArray, *, rcond: float) -> FloatArray:
    """Return a scale-safe, support-consistent positive-semidefinite solve."""

    solution, _ = solve_psd_with_projected_square_root(
        matrix,
        right_hand_side,
        relative_eigenvalue_cutoff=rcond,
    )
    return solution


@dataclass(frozen=True, slots=True)
class JointGaussianPosterior:
    """Immutable joint posterior over finite points and multiple outputs.

    ``mean[i, a]`` is the latent-function mean for output ``a`` at point ``i``.
    ``covariance[i, a, j, b]`` is its covariance with output ``b`` at point
    ``j``.  ``observation_noise[i]`` is the within-point output covariance for
    a future noisy observation; observation noise is independent across points.

    Acquisition code consumes this object directly, keeping validation and
    posterior updates in one module rather than duplicating Gaussian state.
    """

    mean: FloatArray
    covariance: FloatArray
    observation_noise: FloatArray | None = None

    def __post_init__(self) -> None:
        mean = _readonly_float_array(self.mean, name="mean")
        if mean.ndim != 2 or 0 in mean.shape:
            raise ValueError("mean must have shape (n_points, n_outputs)")
        n_points, n_outputs = mean.shape
        covariance = _readonly_float_array(self.covariance, name="covariance")
        expected = (n_points, n_outputs, n_points, n_outputs)
        if covariance.shape != expected:
            raise ValueError(f"covariance must have shape {expected}")
        flat_covariance = _symmetric_psd(
            covariance.reshape(n_points * n_outputs, n_points * n_outputs),
            name="covariance",
            expected_shape=(n_points * n_outputs, n_points * n_outputs),
        )
        covariance = np.array(flat_covariance.reshape(expected), copy=True)
        covariance.setflags(write=False)

        if self.observation_noise is None:
            noise = np.zeros((n_points, n_outputs, n_outputs), dtype=np.float64)
        else:
            noise = _readonly_float_array(self.observation_noise, name="observation_noise")
            if noise.shape != (n_points, n_outputs, n_outputs):
                raise ValueError(
                    "observation_noise must have shape (n_points, n_outputs, n_outputs)"
                )
            validated_noise = np.empty_like(noise)
            for point, point_noise in enumerate(noise):
                validated_noise[point] = _symmetric_psd(
                    point_noise,
                    name=f"observation_noise[{point}]",
                    expected_shape=(n_outputs, n_outputs),
                )
            noise = validated_noise
        noise = np.array(noise, dtype=np.float64, copy=True)
        noise.setflags(write=False)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "observation_noise", noise)

    @property
    def n_points(self) -> int:
        """Number of finite decision or evaluation points."""

        return int(self.mean.shape[0])

    @property
    def n_outputs(self) -> int:
        """Number of jointly modeled outputs per point."""

        return int(self.mean.shape[1])

    @classmethod
    def from_bayesian_linear(
        cls,
        embeddings: ArrayLike,
        weight_mean: ArrayLike,
        weight_covariance: ArrayLike,
        *,
        observation_noise: ArrayLike | None = None,
    ) -> JointGaussianPosterior:
        """Project a Bayesian linear weight posterior onto fixed embeddings.

        ``embeddings`` has shape ``(n_points, n_features)``, ``weight_mean`` has
        shape ``(n_features, n_outputs)``, and ``weight_covariance[d,a,e,b]``
        retains arbitrary covariance between feature/output weights.  Add a
        column of ones to the embeddings before calling when an intercept is
        required.
        """

        features = np.asarray(embeddings, dtype=np.float64)
        weights = np.asarray(weight_mean, dtype=np.float64)
        if features.ndim != 2 or 0 in features.shape:
            raise ValueError("embeddings must have shape (n_points, n_features)")
        if np.any(~np.isfinite(features)):
            raise ValueError("embeddings must contain only finite values")
        if weights.ndim != 2 or weights.shape[0] != features.shape[1] or weights.shape[1] == 0:
            raise ValueError("weight_mean must have shape (n_features, n_outputs)")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weight_mean must contain only finite values")

        n_features, n_outputs = weights.shape
        raw_weight_covariance = np.asarray(weight_covariance, dtype=np.float64)
        expected = (n_features, n_outputs, n_features, n_outputs)
        if raw_weight_covariance.shape != expected:
            raise ValueError(f"weight_covariance must have shape {expected}")
        flat_weight_covariance = _symmetric_psd(
            raw_weight_covariance.reshape(n_features * n_outputs, -1),
            name="weight_covariance",
            expected_shape=(n_features * n_outputs, n_features * n_outputs),
        )
        validated_weight_covariance = flat_weight_covariance.reshape(expected)

        mean = features @ weights
        covariance = np.einsum(
            "id,daeb,je->iajb",
            features,
            validated_weight_covariance,
            features,
            optimize=True,
        )
        return cls(mean, covariance, observation_noise)

    def sample_functions(self, n_samples: int, *, seed: int) -> FloatArray:
        """Draw coherent latent functions with shape ``(samples, points, outputs)``."""

        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise ValueError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        generator = np.random.Generator(np.random.PCG64(int(seed)))
        normals = generator.standard_normal((n_samples, self.n_points * self.n_outputs))
        return self._samples_from_standard_normals(normals)

    def sample_functions_from_seeds(self, seeds: Sequence[int]) -> FloatArray:
        """Draw one coherent function per stable seed, independent of batching."""

        parsed: list[int] = []
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
                raise ValueError("seeds must contain non-negative integers")
            value = int(seed)
            if value < 0:
                raise ValueError("seeds must contain non-negative integers")
            parsed.append(value)
        if not parsed:
            raise ValueError("seeds cannot be empty")
        width = self.n_points * self.n_outputs
        normals = np.stack(
            [np.random.Generator(np.random.PCG64(seed)).standard_normal(width) for seed in parsed]
        )
        return self._samples_from_standard_normals(normals)

    def _samples_from_standard_normals(self, normals: FloatArray) -> FloatArray:
        """Apply the scale-equivariant covariance factor to standard normals."""

        expected_width = self.n_points * self.n_outputs
        if normals.ndim != 2 or normals.shape[1] != expected_width:
            raise ValueError("standard normals must match the flattened posterior width")
        if np.any(~np.isfinite(normals)):
            raise ValueError("standard normals must contain only finite values")
        flat_covariance = self.covariance.reshape(
            expected_width,
            expected_width,
        )
        square_root = _equilibrated_covariance_factor(flat_covariance)
        flat_mean = self.mean.reshape(-1)
        samples = np.empty_like(normals, dtype=np.float64)
        # Always apply one matrix-vector transform per seed.  A batched GEMM can
        # select shape-dependent kernels and thereby change the exact draw bytes
        # when a rollout batch is reordered or sharded.
        with np.errstate(over="ignore", invalid="ignore"):
            for position, normal in enumerate(normals):
                samples[position] = square_root @ normal + flat_mean
        if np.any(~np.isfinite(samples)):
            raise ValueError("posterior samples are not representable in float64")
        samples = samples.reshape(normals.shape[0], self.n_points, self.n_outputs)
        samples.setflags(write=False)
        return samples

    def condition(
        self,
        point_indices: Sequence[int],
        observed_values: ArrayLike,
        *,
        output_indices: Sequence[int] | None = None,
        rcond: float = 1e-12,
    ) -> JointGaussianPosterior:
        """Condition exactly on one block of noisy finite-point observations.

        Every requested point is observed at the same ``output_indices``.  A
        repeated point denotes an independent replicate with the same stored
        within-point observation-noise covariance.
        """

        points, outputs = self._validated_observation_indices(point_indices, output_indices)
        values = np.asarray(observed_values, dtype=np.float64)
        expected = (len(points), len(outputs))
        if values.shape != expected:
            raise ValueError(f"observed_values must have shape {expected}")
        if np.any(~np.isfinite(values)):
            raise ValueError("observed_values must contain only finite values")

        means, covariance = self.fantasy_moments(
            points,
            values[None, :, :],
            output_indices=outputs,
            rcond=rcond,
        )
        return JointGaussianPosterior(means[0], covariance, self.observation_noise)

    def fantasy_moments(
        self,
        point_indices: Sequence[int],
        fantasy_values: ArrayLike,
        *,
        output_indices: Sequence[int] | None = None,
        rcond: float = 1e-12,
    ) -> tuple[FloatArray, FloatArray]:
        """Return exact means and shared covariance for a block of fantasies.

        ``fantasy_values`` has shape ``(n_fantasies, n_points_observed,
        n_outputs_observed)``.  All fantasies share the same conditioned latent
        covariance; their means have shape ``(n_fantasies, n_points,
        n_outputs)``.
        """

        points, outputs = self._validated_observation_indices(point_indices, output_indices)
        fantasies = np.asarray(fantasy_values, dtype=np.float64)
        expected_tail = (len(points), len(outputs))
        if fantasies.ndim != 3 or fantasies.shape[0] == 0 or fantasies.shape[1:] != expected_tail:
            raise ValueError(
                f"fantasy_values must have shape (n_fantasies, {len(points)}, {len(outputs)})"
            )
        if np.any(~np.isfinite(fantasies)):
            raise ValueError("fantasy_values must contain only finite values")
        if not np.isfinite(rcond) or rcond <= 0 or rcond > MAX_RELATIVE_EIGENVALUE_CUTOFF:
            raise ValueError(
                f"rcond must be finite and lie in (0, {MAX_RELATIVE_EIGENVALUE_CUTOFF}]"
            )

        observation_flat, gain, conditioned_covariance, predictive_covariance = (
            self._conditioning_terms(
                points,
                outputs,
                rcond=rcond,
            )
        )
        flat_mean = self.mean.reshape(-1)
        flat_fantasies = fantasies.reshape(fantasies.shape[0], -1)
        observation_mean = flat_mean[observation_flat]
        difference_scale = np.maximum(
            np.max(np.abs(flat_fantasies), axis=1, keepdims=True),
            np.max(np.abs(observation_mean), keepdims=True),
        )
        safe_difference_scale = np.where(difference_scale > 0.0, difference_scale, 1.0)
        scaled_innovations = (
            flat_fantasies / safe_difference_scale
            - observation_mean[None, :] / safe_difference_scale
        )
        if np.any((flat_fantasies != observation_mean[None, :]) & (scaled_innovations == 0.0)):
            raise ValueError("conditioned innovations are not representable in float64")
        # A degenerate Gaussian assigns zero probability outside the affine
        # support of its predictive covariance.  Validate that fantasies lie
        # in that support instead of silently ignoring impossible components.
        _validate_psd_support(predictive_covariance, scaled_innovations.T, rcond=rcond)
        with np.errstate(over="ignore", invalid="ignore"):
            extended_update = np.asarray(scaled_innovations, dtype=np.longdouble) @ np.asarray(
                gain.T,
                dtype=np.longdouble,
            )
            extended_means = np.asarray(flat_mean[None, :], dtype=np.longdouble) + (
                extended_update * np.asarray(safe_difference_scale, dtype=np.longdouble)
            )
        float64_limit = np.longdouble(np.finfo(np.float64).max)
        if np.any(~np.isfinite(extended_means)) or np.any(np.abs(extended_means) > float64_limit):
            raise ValueError("conditioned means are not representable in float64")
        fantasy_means = np.asarray(extended_means, dtype=np.float64)
        if np.any((extended_means != 0.0) & (fantasy_means == 0.0)):
            raise ValueError("conditioned means are not representable in float64")
        fantasy_means = fantasy_means.reshape(fantasies.shape[0], self.n_points, self.n_outputs)
        for point_position, point in enumerate(points):
            for output_position, output in enumerate(outputs):
                if self.observation_noise[point, output, output] == 0.0:
                    fantasy_means[:, point, output] = fantasies[:, point_position, output_position]
        fantasy_means.setflags(write=False)
        conditioned_covariance.setflags(write=False)
        return fantasy_means, conditioned_covariance

    def _validated_observation_indices(
        self,
        point_indices: Sequence[int],
        output_indices: Sequence[int] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        points = _indices(
            point_indices,
            size=self.n_points,
            name="point_indices",
            unique=False,
        )
        outputs = (
            tuple(range(self.n_outputs))
            if output_indices is None
            else _indices(
                output_indices,
                size=self.n_outputs,
                name="output_indices",
                unique=True,
            )
        )
        return points, outputs

    def _conditioning_terms(
        self,
        points: tuple[int, ...],
        outputs: tuple[int, ...],
        *,
        rcond: float,
    ) -> tuple[NDArray[np.int64], FloatArray, FloatArray, FloatArray]:
        flat_covariance = self.covariance.reshape(
            self.n_points * self.n_outputs,
            self.n_points * self.n_outputs,
        )
        observation_flat = np.asarray(
            [point * self.n_outputs + output for point in points for output in outputs],
            dtype=np.int64,
        )
        latent_observation_covariance = flat_covariance[np.ix_(observation_flat, observation_flat)]
        noise_blocks = [self.observation_noise[point][np.ix_(outputs, outputs)] for point in points]
        noise_covariance = block_diag(*noise_blocks)
        predictive_covariance = latent_observation_covariance + noise_covariance
        predictive_covariance = _checked_symmetric_average(
            predictive_covariance,
            name="predictive observation covariance",
        )
        cross_covariance = flat_covariance[:, observation_flat]
        solved_cross = _psd_solve(
            predictive_covariance,
            cross_covariance.T,
            rcond=rcond,
        )
        gain = solved_cross.T
        exact_observed = tuple(
            dict.fromkeys(
                point * self.n_outputs + output
                for point in points
                for output in outputs
                if self.observation_noise[point, output, output] == 0.0
            )
        )
        conditioned_flat = psd_joseph_conditioned_covariance(
            flat_covariance,
            gain,
            cross_covariance,
            latent_observation_covariance,
            noise_covariance,
            exact_zero_indices=exact_observed,
            collapse_factor_roundoff=True,
        )
        conditioned = conditioned_flat.reshape(self.covariance.shape)
        return (
            observation_flat,
            gain,
            np.asarray(conditioned, dtype=np.float64),
            np.asarray(predictive_covariance, dtype=np.float64),
        )

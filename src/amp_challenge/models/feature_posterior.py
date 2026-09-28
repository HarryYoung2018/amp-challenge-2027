"""Batched Gaussian linear prediction with a shared latent coefficient posterior.

This backend accepts frozen real feature matrices (for example ESM plus spectral
features). It does not fit feature transforms, select likelihoods, or certify
calibration. Those decisions belong to the leakage-safe training pipeline.

For features x, f(x) = x W + (x B) z, with z ~ N(m, (R.T R)^-1).
R is upper triangular. Updates use a QR information square root; candidate
scoring never constructs covariance between every pair in the candidate pool.
Only ``joint`` materializes that covariance, under an explicit shortlist cap.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, qr, solve_triangular

from amp_challenge.models.posterior import JointGaussianPosterior

FloatArray = NDArray[np.float64]
MAX_LATENT_RANK = 1024
MAX_FACTOR_ELEMENTS = 16_777_216
MAX_BATCH_POINTS = 8192
MAX_OBSERVATIONS = 512
MAX_JOINT_POINTS = 256
MAX_OUTPUTS = 16
MAX_PROJECTION_ELEMENTS = 8_388_608
MAX_DRAW_ELEMENTS = 4_194_304


def _array(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    view = np.asarray(value)
    if view.ndim != ndim or view.size > MAX_FACTOR_ELEMENTS:
        raise ValueError(f"{name} dimensions or storage exceed supported bounds")
    if view.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    result = np.array(view, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _integer(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class MarginalBlock:
    """A contiguous input slice with latent mean and within-point covariance."""

    start: int
    mean: FloatArray
    covariance: FloatArray


@dataclass(frozen=True, slots=True)
class FeatureGaussianPosterior:
    """A Gaussian coefficient posterior shared by every evaluated peptide.

    ``coefficient_factor`` has axes (feature, output, latent_rank). It can be
    rank deficient: deterministic predictions and correlated outputs are valid.
    The latent precision itself must be strictly positive definite. Noiseless
    conditioning is intentionally delegated to the existing dense support-aware
    backend; this implementation requires positive-definite measurement noise.
    """

    coefficient_mean: FloatArray
    coefficient_factor: FloatArray
    latent_mean: FloatArray
    upper_precision: FloatArray
    observation_count: int = 0

    def __post_init__(self) -> None:
        # Reject invalid dimensions before detaching/casting potentially large
        # mapped coefficient arrays. Array-like Python lists must be converted
        # once to discover their shape; ndarray/memmap inputs remain views here.
        raw_weights = np.asarray(self.coefficient_mean)
        raw_basis = np.asarray(self.coefficient_factor)
        raw_mean = np.asarray(self.latent_mean)
        raw_root = np.asarray(self.upper_precision)
        if raw_weights.ndim != 2 or raw_basis.ndim != 3 or raw_mean.ndim != 1:
            raise ValueError("coefficient and latent array dimensions are invalid")
        d, outputs = raw_weights.shape
        rank = len(raw_mean)
        if d == 0 or not 1 <= outputs <= MAX_OUTPUTS:
            raise ValueError("feature and output counts are outside supported bounds")
        if not 1 <= rank <= MAX_LATENT_RANK or raw_basis.size > MAX_FACTOR_ELEMENTS:
            raise ValueError("latent factor exceeds the resource bounds")
        if raw_basis.shape != (d, outputs, rank) or raw_root.shape != (rank, rank):
            raise ValueError("coefficient factor and precision dimensions disagree")
        weights = _array(self.coefficient_mean, "coefficient_mean", 2)
        basis = _array(self.coefficient_factor, "coefficient_factor", 3)
        mean = _array(self.latent_mean, "latent_mean", 1)
        root = _array(self.upper_precision, "upper_precision", 2)
        if np.any(np.diag(root) <= 0) or np.any(np.tril(root, -1) != 0):
            raise ValueError("upper_precision must be upper triangular with positive diagonal")
        if type(self.observation_count) is not int or self.observation_count < 0:
            raise ValueError("observation_count must be a nonnegative integer")
        for name, value in (
            ("coefficient_mean", weights),
            ("coefficient_factor", basis),
            ("latent_mean", mean),
            ("upper_precision", root),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def prior(
        cls, coefficient_mean: ArrayLike, coefficient_factor: ArrayLike
    ) -> FeatureGaussianPosterior:
        """Create a prior with unit latent covariance; scaling belongs in B."""

        basis = np.asarray(coefficient_factor)
        if basis.ndim != 3 or not 1 <= basis.shape[2] <= MAX_LATENT_RANK:
            raise ValueError("coefficient_factor must have a supported latent rank")
        rank = basis.shape[2]
        return cls(coefficient_mean, basis, np.zeros(rank), np.eye(rank))

    def _features(self, value: ArrayLike, *, maximum: int | None = None) -> FloatArray:
        features = np.asarray(value)
        if features.ndim != 2 or features.shape[1] != self.coefficient_mean.shape[0]:
            raise ValueError("features must have shape (points, trained_feature_count)")
        if features.dtype.kind not in "fiu":
            raise ValueError("features must contain real numbers")
        if maximum is not None and len(features) > maximum:
            raise ValueError(f"point count exceeds limit {maximum}")
        # A whole pool may be a memory map. Validate each slice at consumption,
        # avoiding an N*D temporary and allowing streaming from disk.
        return features

    def _project(self, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        if (
            max(
                features.size,
                len(features) * self.coefficient_factor.shape[1] * len(self.latent_mean),
            )
            > MAX_PROJECTION_ELEMENTS
        ):
            raise ValueError("feature projection exceeds the intermediate memory limit")
        features = np.asarray(features, dtype=np.float64)
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                h = np.einsum("nd,dor->nor", features, self.coefficient_factor)
                mean = features @ self.coefficient_mean + h @ self.latent_mean
            except FloatingPointError as error:
                raise ValueError("feature projection is not representable") from error
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(mean)):
            raise ValueError("feature projection is not representable")
        return mean, h

    def _predictive_factor(self, h: FloatArray) -> FloatArray:
        flat = h.reshape(-1, h.shape[-1])
        # C = H R^-1, so C C.T = H (R.T R)^-1 H.T.
        factor = solve_triangular(self.upper_precision.T, flat.T, lower=True).T
        if not np.all(np.isfinite(factor)):
            raise ValueError("predictive factor is not representable")
        return factor.reshape(h.shape)

    def iter_marginals(
        self, features: ArrayLike, *, batch_size: int = 512
    ) -> Iterator[MarginalBlock]:
        """Score a pool in bounded slices without an N-by-N covariance matrix.

        Returned covariance excludes observation noise. A caller must add its
        frozen likelihood noise when predicting future measurements.
        """

        _integer(batch_size, "batch_size", MAX_BATCH_POINTS)
        features = self._features(features)
        batch_size = min(
            batch_size,
            MAX_PROJECTION_ELEMENTS
            // max(
                self.coefficient_mean.shape[0],
                self.coefficient_mean.shape[1] * len(self.latent_mean),
            ),
        )
        if batch_size == 0:
            raise ValueError("one feature row exceeds the intermediate memory limit")
        for start in range(0, len(features), batch_size):
            mean, h = self._project(features[start : start + batch_size])
            factor = self._predictive_factor(h)
            with np.errstate(over="ignore", invalid="ignore"):
                covariance = np.einsum("nor,npr->nop", factor, factor)
            if not np.all(np.isfinite(covariance)):
                raise ValueError("marginal covariance is not representable")
            mean.setflags(write=False)
            covariance.setflags(write=False)
            yield MarginalBlock(start, mean, covariance)

    def joint(
        self, features: ArrayLike, *, observation_noise: ArrayLike | None = None
    ) -> JointGaussianPosterior:
        """Materialize at most 256 points for covariance-aware acquisition."""

        features = self._features(features, maximum=MAX_JOINT_POINTS)
        if not len(features):
            raise ValueError("joint shortlist must be nonempty")
        if len(features) * self.coefficient_mean.shape[1] > 1024:
            raise ValueError("joint shortlist exceeds the scalar covariance limit")
        mean, h = self._project(features)
        factor = self._predictive_factor(h).reshape(mean.size, -1)
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = factor @ factor.T
        return JointGaussianPosterior(
            mean, covariance.reshape(*mean.shape, *mean.shape), observation_noise
        )

    def condition(
        self,
        features: ArrayLike,
        output_indices: Sequence[int],
        values: ArrayLike,
        noise_covariance: ArrayLike,
    ) -> FeatureGaussianPosterior:
        """Assimilate scalar observations, including partial endpoint panels.

        Each row is one observed (feature vector, output index, value). Repeated
        feature rows represent multiple endpoints or genuine noisy replicates.
        ``noise_covariance`` is their full positive-definite measurement-noise
        matrix; correlated endpoints at the same peptide can be supplied.
        Identity deduplication, censoring, budget accounting, and receipt checks
        belong to the campaign's data/oracle layer and are not inferred here.
        """

        features = self._features(features, maximum=MAX_OBSERVATIONS)
        count = len(features)
        if not count:
            raise ValueError("observation batch must be nonempty")
        outputs = tuple(output_indices)
        if len(outputs) != count or any(
            type(i) is not int or not 0 <= i < self.coefficient_mean.shape[1] for i in outputs
        ):
            raise ValueError("output_indices must contain one valid integer per observation")
        values = _array(values, "values", 1)
        noise = _array(noise_covariance, "noise_covariance", 2)
        if values.shape != (count,) or noise.shape != (count, count):
            raise ValueError("values and noise must match the observation count")
        if not np.array_equal(noise, noise.T):
            raise ValueError("measurement noise must be symmetric")
        try:
            noise_root = cholesky(noise, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("measurement noise must be positive definite") from error
        mean, all_h = self._project(features)
        indices = np.asarray(outputs)
        h = all_h[np.arange(count), indices]
        with np.errstate(over="ignore", invalid="ignore"):
            residual = values - mean[np.arange(count), indices]
        if not np.all(np.isfinite(residual)):
            raise ValueError("observation residual is not representable")
        whitened_h = solve_triangular(noise_root, h, lower=True)
        whitened_residual = solve_triangular(noise_root, residual, lower=True)
        if not np.all(np.isfinite(whitened_h)) or not np.all(np.isfinite(whitened_residual)):
            raise ValueError("whitened observation is not representable")
        # QR updates the information square root without subtracting nearly
        # equal covariances or constructing the full pool's Gaussian state.
        orthogonal, root = qr(np.vstack((self.upper_precision, whitened_h)), mode="economic")
        # The augmented least-squares residual has zero prior rows. Projecting
        # its RHS avoids forming H.T H or H.T r, whose squared scale can
        # overflow even when the QR solution is representable.
        with np.errstate(over="ignore", invalid="ignore"):
            rhs = orthogonal[len(self.latent_mean) :].T @ whitened_residual
        if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(root)):
            raise ValueError("posterior information update is not representable")
        delta = solve_triangular(root, rhs, lower=False)
        root = np.where(np.diag(root)[:, None] < 0, -root, root)
        with np.errstate(over="ignore", invalid="ignore"):
            updated_mean = self.latent_mean + delta
        return FeatureGaussianPosterior(
            self.coefficient_mean,
            self.coefficient_factor,
            updated_mean,
            root,
            self.observation_count + count,
        )

    def draw_latent(self, *, seed: int, count: int = 1) -> FloatArray:
        """Draw once and reuse across every batch for frozen Thompson functions."""

        _integer(count, "count", 512)
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        standard = np.random.Generator(np.random.PCG64DXSM(seed)).standard_normal(
            (len(self.latent_mean), count)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            draws = self.latent_mean + solve_triangular(self.upper_precision, standard).T
        if not np.all(np.isfinite(draws)):
            raise ValueError("latent draws are not representable")
        draws.setflags(write=False)
        return draws

    def evaluate_draws(self, features: ArrayLike, draws: ArrayLike) -> FloatArray:
        """Evaluate saved latent draws on a bounded candidate slice."""

        features = self._features(features, maximum=MAX_BATCH_POINTS)
        draws = _array(draws, "draws", 2)
        if draws.shape[1] != len(self.latent_mean) or not 1 <= len(draws) <= 512:
            raise ValueError("draws must match the latent rank and sample limit")
        if len(features) * len(draws) * self.coefficient_mean.shape[1] > MAX_DRAW_ELEMENTS:
            raise ValueError("sampled prediction exceeds the output memory limit")
        _, h = self._project(features)
        with np.errstate(over="ignore", invalid="ignore"):
            values = (features @ self.coefficient_mean)[None] + np.einsum("nor,sr->sno", h, draws)
        if not np.all(np.isfinite(values)):
            raise ValueError("sampled prediction is not representable")
        return values

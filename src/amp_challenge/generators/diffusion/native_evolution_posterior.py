"""Frozen charged-only Gaussian seam and receipt-preserving feature cache.

The numerical learner and label-free producer are caller-authenticated assets.
This module does not train a teacher, grant oracle access, or certify calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from amp_challenge.generators.diffusion.native_baseline_operators import sequence_id
from amp_challenge.generators.diffusion.native_endpoint import _json_hash
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import (
    canonical_sequence,
    hash_string,
)
from amp_challenge.models.feature_posterior import FeatureGaussianPosterior


@dataclass(frozen=True, slots=True)
class EvolutionFeatureBinding:
    representation: str
    feature_source_sha256: str
    transform_sha256: str
    provider_sha256: str

    def __post_init__(self):
        if self.representation not in (
            "esm320_plus_normalized_length",
            "esm320_plus_normalized_length_plus_spectral32",
        ) or any(
            not hash_string(value)
            for value in (self.feature_source_sha256, self.transform_sha256, self.provider_sha256)
        ):
            raise ValueError("evolutionary label-free feature binding differs")

    @property
    def width(self):
        return 321 if self.representation == "esm320_plus_normalized_length" else 353

    @property
    def sha256(self):
        return _json_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class EvolutionFeatureBatch:
    sequences: tuple[str, ...]
    features: np.ndarray
    receipt_sha256: str
    binding: EvolutionFeatureBinding

    def __post_init__(self):
        if (
            type(self.sequences) is not tuple
            or not 1 <= len(self.sequences) <= 128
            or len(set(self.sequences)) != len(self.sequences)
            or any(not canonical_sequence(seq) for seq in self.sequences)
        ):
            raise ValueError("feature batch support/order differs")
        if type(self.binding) is not EvolutionFeatureBinding or not hash_string(
            self.receipt_sha256
        ):
            raise ValueError("feature batch source receipt differs")
        raw = np.asarray(self.features)
        if (
            raw.ndim != 2
            or raw.shape != (len(self.sequences), self.binding.width)
            or raw.dtype.kind not in "fiu"
        ):
            raise ValueError("feature dimensions/numerical support differ")
        values = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("feature values must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "features", values)


class EvolutionFeatureProvider(Protocol):
    binding: EvolutionFeatureBinding

    def evaluate(self, sequences: tuple[str, ...]) -> EvolutionFeatureBatch: ...


class EvolutionFeatureCache:
    """Requests count even on failure; cached rows retain original receipt/order."""

    def __init__(self, binding: EvolutionFeatureBinding, *, maximum_requests: int = 128):
        if type(maximum_requests) is not int or not 1 <= maximum_requests <= 1024:
            raise ValueError("feature request bound must be explicitly bounded")
        self.binding = binding
        self.maximum_requests = maximum_requests
        self.rows: dict[str, tuple[np.ndarray, str, int, str]] = {}
        self.requests = 0
        self.wave_requests = 0
        self.events: list[dict] = []

    def preload(self, batch: EvolutionFeatureBatch):
        if type(batch) is not EvolutionFeatureBatch or batch.binding != self.binding:
            raise ValueError("preloaded feature source differs")
        batch.__post_init__()
        for index, seq in enumerate(batch.sequences):
            vector = batch.features[index].copy()
            vector.setflags(write=False)
            entry = (vector, batch.receipt_sha256, index, _json_hash(vector.tolist()))
            if seq in self.rows:
                old = self.rows[seq]
                if not np.array_equal(old[0], vector):
                    raise ValueError("feature cache identity changed between receipts")
            else:
                self.rows[seq] = entry

    def begin_wave(self):
        self.wave_requests = 0
        self.events = []

    def matrix(self, sequences: tuple[str, ...]):
        if not sequences or any(seq not in self.rows for seq in sequences):
            raise ValueError("required parent/terminal feature not in authenticated cache")
        for seq in sequences:
            vector, receipt, row, numerical = self.rows[seq]
            if _json_hash(vector.tolist()) != numerical:
                raise ValueError("cached feature bytes changed")
            self.events.append(
                {
                    "kind": "cache_hit",
                    "sequence_id": sequence_id(seq),
                    "receipt_sha256": receipt,
                    "receipt_row": row,
                    "feature_sha256": numerical,
                }
            )
        return np.stack([self.rows[seq][0] for seq in sequences])

    def ensure(self, sequences, provider: EvolutionFeatureProvider, deadline):
        unseen = tuple(dict.fromkeys(seq for seq in sequences if seq not in self.rows))
        for begin in range(0, len(unseen), 128):
            deadline.check("before_feature_request")
            if self.requests >= self.maximum_requests or self.wave_requests >= 4:
                raise ValueError("feature request budget exhausted without cache fallback")
            if provider.binding != self.binding:
                raise ValueError("feature provider binding differs before request")
            requested = unseen[begin : begin + 128]
            self.requests += 1
            self.wave_requests += 1
            event = {
                "kind": "feature_request",
                "ordinal": self.requests - 1,
                "sequence_ids": [sequence_id(seq) for seq in requested],
                "status": "started",
            }
            self.events.append(event)
            batch = provider.evaluate(requested)
            if (
                provider.binding != self.binding
                or type(batch) is not EvolutionFeatureBatch
                or batch.sequences != requested
            ):
                raise ValueError("feature provider order/binding changed")
            self.preload(batch)
            event.update(status="completed", receipt_sha256=batch.receipt_sha256)
            deadline.check("after_feature_request")


class FrozenEvolutionPosterior:
    """One un-clipped Gaussian scale; snapshot must be fitted from charged data."""

    def __init__(
        self,
        backend: FeatureGaussianPosterior,
        *,
        history_sha256: str,
        context_sha256: str,
        learner_source_sha256: str,
        observation_noise: np.ndarray,
        feature_binding: EvolutionFeatureBinding,
        transform,
    ):
        if type(backend) is not FeatureGaussianPosterior or backend.coefficient_mean.shape != (
            feature_binding.width + 1,
            2,
        ):
            raise ValueError("frozen charged learner feature/output dimensions differ")
        if transform.sha256 != feature_binding.transform_sha256 or not callable(transform.apply):
            raise ValueError("explicit generator-only transform differs")
        if any(
            not hash_string(value)
            for value in (history_sha256, context_sha256, learner_source_sha256)
        ):
            raise ValueError("frozen charged learner source/history differs")
        raw_noise = np.asarray(observation_noise)
        if raw_noise.shape != (2, 2) or raw_noise.dtype.kind not in "fiu":
            raise ValueError("regression residual noise shape/dtype differs")
        noise = np.array(raw_noise, dtype=np.float64, copy=True)
        if (
            noise.shape != (2, 2)
            or not np.isfinite(noise).all()
            or not np.allclose(noise, noise.T, atol=1e-12, rtol=1e-12)
            or np.linalg.eigvalsh(noise).min() < 0
        ):
            raise ValueError("regression residual noise must be symmetric PSD")
        noise.setflags(write=False)
        self.backend, self.feature_binding, self.noise = backend, feature_binding, noise
        self.transform = transform
        self.history_sha256, self.context_sha256 = history_sha256, context_sha256
        self.learner_source_sha256 = learner_source_sha256
        self.numerical_sha256 = self._numerical_hash()
        self.sha256 = _json_hash(
            [
                self.numerical_sha256,
                history_sha256,
                context_sha256,
                learner_source_sha256,
                feature_binding.sha256,
                "unclipped_probability_space_gaussian_model_proxy_not_calibrated_probability",
            ]
        )

    def _numerical_hash(self):
        return _json_hash(
            [
                self.backend.coefficient_mean.tolist(),
                self.backend.coefficient_factor.tolist(),
                self.backend.latent_mean.tolist(),
                self.backend.upper_precision.tolist(),
                self.noise.tolist(),
            ]
        )

    def check(self):
        if (
            self._numerical_hash() != self.numerical_sha256
            or self.transform.sha256 != self.feature_binding.transform_sha256
        ):
            raise ValueError("frozen learner numerical snapshot mutated")

    def _features(self, raw):
        if self.transform.sha256 != self.feature_binding.transform_sha256:
            raise ValueError("generator-only transform changed")
        values = np.asarray(raw)
        if (
            values.ndim != 2
            or not 1 <= len(values) <= 8192
            or values.shape[1] != self.feature_binding.width
            or not np.isfinite(values).all()
        ):
            raise ValueError("raw feature layout differs from transform")
        result = self.transform.apply(values)
        if (
            result.shape != (len(values), self.feature_binding.width + 1)
            or not np.isfinite(result).all()
            or not np.array_equal(result[:, 0], np.ones(len(result)))
        ):
            raise ValueError("transformed features/intercept differ")
        return result

    def means(self, features):
        coefficients = self.backend.coefficient_mean + np.einsum(
            "dor,r->do", self.backend.coefficient_factor, self.backend.latent_mean
        )
        result = self._features(features) @ coefficients
        if not np.isfinite(result).all():
            raise ValueError("frozen posterior point means are nonfinite")
        return result

    def joint(self, features):
        return self.backend.joint(
            self._features(features),
            observation_noise=np.broadcast_to(self.noise, (len(features), 2, 2)),
        )

    def draw(self, features, latent):
        return self.backend.evaluate_draws(self._features(features), np.asarray(latent)[None, :])[0]

    def direction_scores(self, features, latent_draws, directions):
        """Contract frozen coefficients before rows: no per-row rank projection.

        Algebraically the accepted backend's evaluate_draws followed by each
        fixed objective direction. Float64 CPU tests qualify numerical agreement;
        no bitwise cross-device or tied near-boundary ranking claim is made.
        """
        draws, weights = np.asarray(latent_draws), np.asarray(directions)
        if (
            draws.ndim != 2
            or not 1 <= len(draws) <= 40
            or draws.shape[1] != len(self.backend.latent_mean)
            or weights.shape != (len(draws), 2)
            or not np.isfinite(draws).all()
            or not np.isfinite(weights).all()
        ):
            raise ValueError("frozen direction/draw dimensions differ")
        coefficients = self.backend.coefficient_mean @ weights.T + np.einsum(
            "dor,br,bo->db", self.backend.coefficient_factor, draws, weights, optimize=True
        )
        result = self._features(features) @ coefficients
        if not np.isfinite(result).all():
            raise ValueError("frozen Thompson directional values are nonfinite")
        return result

    def pairs(self, child_raw, parent_raw):
        """One batched latent projection per128 pairs; never full-pool covariance.

        Reuses the pinned Gaussian backend's bounded coefficient projection and
        triangular solve, then contracts only each child's four-variable pair.
        """
        if np.shape(child_raw) != np.shape(parent_raw) or not 1 <= len(child_raw) <= 512:
            raise ValueError("paired feature inventory differs")
        means, covariances = [], []
        for start in range(0, len(child_raw), 128):
            raw = np.stack(
                (child_raw[start : start + 128], parent_raw[start : start + 128]), axis=1
            )
            transformed = self._features(raw.reshape(-1, self.feature_binding.width))
            mean, h = self.backend._project(transformed)
            factor = self.backend._predictive_factor(h).reshape(len(raw), 4, -1)
            covariance = np.einsum("nir,njr->nij", factor, factor)
            if not np.isfinite(covariance).all():
                raise ValueError("paired latent covariance is nonfinite")
            means.append(mean.reshape(len(raw), 2, 2))
            covariances.append(covariance)
        return np.concatenate(means), np.concatenate(covariances)

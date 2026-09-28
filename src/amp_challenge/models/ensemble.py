"""Uncertainty-aware aggregation for heterogeneous AMP oracle predictions.

All downstream selection code consumes *utility-oriented* values (larger is
better).  Individual endpoints may still be expressed in their natural units;
``EndpointSpec.direction`` handles the sign conversion exactly once.

The total predictive variance follows the law of total variance: weighted
within-model (aleatoric) variance plus between-model (epistemic/disagreement)
variance.  This is intentionally transparent and auditable.  Correlated model
errors can make disagreement optimistic, so production weights and risk
penalties must be chosen from out-of-fold, homology-clustered validation data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EndpointSpec:
    """Definition of one oracle endpoint.

    Parameters
    ----------
    name:
        Stable endpoint name (for example ``gram_negative_activity``).
    direction:
        ``maximize`` for probabilities/utility and ``minimize`` for quantities
        such as MIC or hemolysis risk.
    objective_weight:
        Relative importance when a scalar portfolio score is needed.
    risk_penalty:
        Multiplicative penalty applied to total predictive standard deviation.
    """

    name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    objective_weight: float = 1.0
    risk_penalty: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("endpoint name cannot be empty")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"unsupported endpoint direction: {self.direction!r}")
        if self.objective_weight < 0 or not np.isfinite(self.objective_weight):
            raise ValueError("objective_weight must be finite and non-negative")
        if self.risk_penalty < 0 or not np.isfinite(self.risk_penalty):
            raise ValueError("risk_penalty must be finite and non-negative")


@dataclass(frozen=True)
class ModelPrediction:
    """Predictions from one independently trained ensemble member.

    ``mean`` and ``std`` have shape ``(n_candidates, n_endpoints)``.  ``weight``
    may be a scalar or one validation-derived weight per endpoint.  Missing
    means are allowed and are ignored per candidate/endpoint; a cell for which
    every model is missing is rejected.
    """

    name: str
    mean: FloatArray
    std: FloatArray | None = None
    weight: float | Sequence[float] | FloatArray = 1.0


@dataclass(frozen=True)
class EnsembleResult:
    """Aggregated predictions in raw and higher-is-better utility space."""

    endpoint_names: tuple[str, ...]
    model_names: tuple[str, ...]
    mean: FloatArray
    utility_mean: FloatArray
    aleatoric_std: FloatArray
    epistemic_std: FloatArray
    total_std: FloatArray
    risk_adjusted_utility: FloatArray
    effective_model_count: NDArray[np.int64]

    @property
    def scalar_utility(self) -> FloatArray:
        """Return a weighted conservative utility for initial ranking."""

        # Objective weights are applied by OracleEnsemble.scalar_utility, where
        # the EndpointSpec objects are available.  The unweighted mean remains
        # a useful and unsurprising default on a standalone result.
        return np.mean(self.risk_adjusted_utility, axis=1)


class OracleEnsemble:
    """Combine diverse oracle outputs while retaining model disagreement."""

    def __init__(self, endpoints: Sequence[EndpointSpec]) -> None:
        self.endpoints = tuple(endpoints)
        if not self.endpoints:
            raise ValueError("at least one endpoint is required")
        names = [endpoint.name for endpoint in self.endpoints]
        if len(names) != len(set(names)):
            raise ValueError("endpoint names must be unique")

    def aggregate(self, predictions: Sequence[ModelPrediction]) -> EnsembleResult:
        """Aggregate ensemble members with per-cell missing-value handling."""

        members = tuple(predictions)
        if not members:
            raise ValueError("at least one model prediction is required")
        if len({member.name for member in members}) != len(members):
            raise ValueError("model names must be unique")

        means = [np.asarray(member.mean, dtype=np.float64) for member in members]
        expected_shape = means[0].shape
        if len(expected_shape) != 2:
            raise ValueError("prediction means must be two-dimensional")
        if expected_shape[1] != len(self.endpoints):
            raise ValueError(f"expected {len(self.endpoints)} endpoints, got {expected_shape[1]}")
        if expected_shape[0] == 0:
            raise ValueError("prediction arrays cannot be empty")
        if any(mean.shape != expected_shape for mean in means):
            raise ValueError("all prediction arrays must have the same shape")

        mean_stack = np.stack(means, axis=0)
        std_stack = np.stack(
            [
                np.zeros(expected_shape, dtype=np.float64)
                if member.std is None
                else self._validated_std(member.std, expected_shape, member.name)
                for member in members
            ],
            axis=0,
        )

        endpoint_weights = np.stack(
            [self._validated_weight(member.weight, member.name) for member in members],
            axis=0,
        )
        available = np.isfinite(mean_stack)
        if np.any(np.isinf(mean_stack)):
            raise ValueError("prediction means may be finite or NaN, but not infinite")
        if np.any(~np.isfinite(std_stack)) or np.any(std_stack < 0):
            raise ValueError("prediction standard deviations must be finite and non-negative")

        weights = endpoint_weights[:, None, :] * available
        denominator = np.sum(weights, axis=0)
        if np.any(denominator <= 0):
            missing = np.argwhere(denominator <= 0)[0]
            raise ValueError(
                "every candidate/endpoint needs at least one positive-weight prediction; "
                f"first missing cell is candidate={missing[0]}, endpoint={missing[1]}"
            )
        normalized = weights / denominator[None, :, :]
        safe_means = np.where(available, mean_stack, 0.0)

        mean = np.sum(normalized * safe_means, axis=0)
        disagreement = safe_means - mean[None, :, :]
        epistemic_var = np.sum(normalized * np.square(disagreement), axis=0)
        aleatoric_var = np.sum(normalized * np.square(std_stack), axis=0)
        epistemic_std = np.sqrt(np.maximum(epistemic_var, 0.0))
        aleatoric_std = np.sqrt(np.maximum(aleatoric_var, 0.0))
        total_std = np.sqrt(np.maximum(epistemic_var + aleatoric_var, 0.0))

        directions = np.asarray(
            [1.0 if endpoint.direction == "maximize" else -1.0 for endpoint in self.endpoints]
        )
        penalties = np.asarray([endpoint.risk_penalty for endpoint in self.endpoints])
        utility_mean = mean * directions[None, :]
        risk_adjusted = utility_mean - penalties[None, :] * total_std

        return EnsembleResult(
            endpoint_names=tuple(endpoint.name for endpoint in self.endpoints),
            model_names=tuple(member.name for member in members),
            mean=mean,
            utility_mean=utility_mean,
            aleatoric_std=aleatoric_std,
            epistemic_std=epistemic_std,
            total_std=total_std,
            risk_adjusted_utility=risk_adjusted,
            effective_model_count=np.sum(available & (weights > 0), axis=0).astype(np.int64),
        )

    def scalar_utility(self, result: EnsembleResult, *, conservative: bool = True) -> FloatArray:
        """Collapse endpoints using declared objective weights.

        Endpoint values should normally be calibrated or normalized before
        calling this method. Final portfolio construction uses posterior
        expected normalized rewards plus a separate reward-protection gate;
        mixed acquisition is reserved for internal label collection.
        """

        if result.endpoint_names != tuple(endpoint.name for endpoint in self.endpoints):
            raise ValueError("result endpoints do not match this ensemble")
        values = result.risk_adjusted_utility if conservative else result.utility_mean
        weights = np.asarray([endpoint.objective_weight for endpoint in self.endpoints])
        if np.sum(weights) <= 0:
            raise ValueError("at least one endpoint objective_weight must be positive")
        return values @ (weights / np.sum(weights))

    def _validated_weight(
        self, weight: float | Sequence[float] | FloatArray, model_name: str
    ) -> FloatArray:
        values = np.asarray(weight, dtype=np.float64)
        if values.ndim == 0:
            values = np.repeat(values, len(self.endpoints))
        if values.shape != (len(self.endpoints),):
            raise ValueError(
                f"model {model_name!r} weight must be scalar or have one value per endpoint"
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"model {model_name!r} weights must be finite and non-negative")
        return values

    @staticmethod
    def _validated_std(std: FloatArray, shape: tuple[int, int], model_name: str) -> FloatArray:
        values = np.asarray(std, dtype=np.float64)
        if values.shape != shape:
            raise ValueError(
                f"model {model_name!r} std shape {values.shape} does not match mean {shape}"
            )
        return values

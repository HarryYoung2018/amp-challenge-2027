"""Dependency-light strain-level activity baselines for Gate-1 evaluation.

The two model families deliberately make different assumptions:

* :class:`DescriptorLogisticOracle` is a global parametric model using
  physicochemical descriptors, amino-acid composition, and additive assay
  context effects.
* :class:`HomologyKnnOracle` is a local non-parametric model that transfers
  labels from the most similar training peptides measured against the same
  strain (then Gram class, if a strain was unseen).

Neither class performs hyperparameter selection.  Benchmark hyperparameters
must be fixed before cross-validation so out-of-fold predictions stay honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.sequences import canonicalize_sequence
from amp_challenge.similarity import global_sequence_identity

FloatArray = NDArray[np.float64]
GramClass = Literal["positive", "negative", "unknown"]


@dataclass(frozen=True, slots=True)
class OracleInput:
    """Features available for one sequence/strain prediction."""

    sequence: str
    strain: str
    gram: GramClass = "unknown"

    def __post_init__(self) -> None:
        canonical = canonicalize_sequence(self.sequence)
        if canonical != self.sequence:
            raise ValueError("OracleInput.sequence must already be canonical")
        if not self.strain.strip():
            raise ValueError("OracleInput.strain cannot be empty")
        if self.gram not in {"positive", "negative", "unknown"}:
            raise ValueError(f"invalid Gram class: {self.gram!r}")


def _validated_labels(labels: NDArray[np.int64] | list[int]) -> NDArray[np.float64]:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("labels must contain only 0 and 1")
    return values


def _smoothed_prevalence(labels: FloatArray, strength: float) -> float:
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("prior_strength must be finite and non-negative")
    return float((np.sum(labels) + 0.5 * strength) / (labels.size + strength))


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


@lru_cache(maxsize=100_000)
def _sequence_features(sequence: str) -> tuple[float, ...]:
    descriptors = compute_descriptors(sequence).as_dict()
    length = len(sequence)
    descriptor_values = (
        float(descriptors["length"]),
        float(descriptors["molecular_weight_da"]),
        float(descriptors["net_charge"]),
        float(descriptors["charge_density"]),
        float(descriptors["isoelectric_point"]),
        float(descriptors["mean_hydrophobicity"]),
        float(descriptors["hydrophobic_moment"]),
        float(descriptors["hydrophobic_fraction"]),
        float(descriptors["aromatic_fraction"]),
        float(descriptors["basic_fraction"]),
        float(descriptors["acidic_fraction"]),
        float(descriptors["shannon_entropy"]),
        float(descriptors["max_residue_fraction"]),
    )
    composition = tuple(sequence.count(residue) / length for residue in STANDARD_AMINO_ACIDS)
    return descriptor_values + composition


class DescriptorLogisticOracle:
    """L2-regularized logistic regression on transparent sequence features."""

    name = "descriptor_logistic"

    def __init__(
        self,
        *,
        l2: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
        prior_strength: float = 2.0,
    ) -> None:
        if not math.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be finite and positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and non-negative")
        self.l2 = float(l2)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.prior_strength = float(prior_strength)
        self._strains: tuple[str, ...] | None = None
        self._mean: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._coefficient: FloatArray | None = None
        self._constant_probability: float | None = None

    def fit(
        self,
        rows: list[OracleInput] | tuple[OracleInput, ...],
        labels: NDArray[np.int64] | list[int],
    ) -> DescriptorLogisticOracle:
        if not rows:
            raise ValueError("at least one training row is required")
        y = _validated_labels(labels)
        if len(rows) != y.size:
            raise ValueError("rows and labels must have equal length")

        self._strains = tuple(sorted({row.strain for row in rows}))
        continuous = np.asarray([_sequence_features(row.sequence) for row in rows])
        self._mean = np.mean(continuous, axis=0)
        self._scale = np.std(continuous, axis=0)
        self._scale[self._scale < 1e-12] = 1.0
        x = self._design_matrix(rows)
        prior = _smoothed_prevalence(y, self.prior_strength)

        if np.all(y == y[0]):
            self._constant_probability = prior
            self._coefficient = None
            return self

        coefficient = np.zeros(x.shape[1], dtype=np.float64)
        coefficient[0] = math.log(prior / (1.0 - prior))
        penalty = np.full(x.shape[1], self.l2, dtype=np.float64)
        penalty[0] = 0.0
        sample_count = y.size

        for _ in range(self.max_iterations):
            probability = _sigmoid(x @ coefficient)
            variance = np.clip(probability * (1.0 - probability), 1e-9, None)
            gradient = (x.T @ (probability - y)) / sample_count + penalty * coefficient
            hessian = (x.T * variance) @ x / sample_count
            hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            coefficient -= step
            if float(np.max(np.abs(step))) <= self.tolerance:
                break

        self._coefficient = coefficient
        self._constant_probability = None
        return self

    def predict_proba(
        self,
        rows: list[OracleInput] | tuple[OracleInput, ...],
    ) -> FloatArray:
        if self._strains is None or self._mean is None or self._scale is None:
            raise RuntimeError("model must be fit before prediction")
        if not rows:
            return np.empty(0, dtype=np.float64)
        if self._constant_probability is not None:
            return np.full(len(rows), self._constant_probability, dtype=np.float64)
        assert self._coefficient is not None
        probability = _sigmoid(self._design_matrix(rows) @ self._coefficient)
        return np.clip(probability, 1e-6, 1.0 - 1e-6)

    def _design_matrix(
        self,
        rows: list[OracleInput] | tuple[OracleInput, ...],
    ) -> FloatArray:
        assert self._strains is not None
        assert self._mean is not None
        assert self._scale is not None
        continuous = np.asarray([_sequence_features(row.sequence) for row in rows])
        standardized = (continuous - self._mean) / self._scale
        strain_index = {strain: index for index, strain in enumerate(self._strains)}
        # One extra strain column is an explicit unseen-category bucket.
        strain = np.zeros((len(rows), len(self._strains) + 1), dtype=np.float64)
        for row_index, row in enumerate(rows):
            strain[row_index, strain_index.get(row.strain, len(self._strains))] = 1.0
        gram = np.zeros((len(rows), 3), dtype=np.float64)
        gram_index = {"positive": 0, "negative": 1, "unknown": 2}
        for row_index, row in enumerate(rows):
            gram[row_index, gram_index[row.gram]] = 1.0
        intercept = np.ones((len(rows), 1), dtype=np.float64)
        return np.concatenate((intercept, standardized, strain, gram), axis=1)


class HomologyKnnOracle:
    """Similarity-weighted kNN with strain-first context matching."""

    name = "homology_knn"

    def __init__(
        self,
        *,
        neighbors: int = 7,
        similarity_power: float = 4.0,
        prior_strength: float = 2.0,
        minimum_weight: float = 1e-6,
    ) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        if not math.isfinite(similarity_power) or similarity_power <= 0:
            raise ValueError("similarity_power must be finite and positive")
        if not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and non-negative")
        if not math.isfinite(minimum_weight) or minimum_weight <= 0:
            raise ValueError("minimum_weight must be finite and positive")
        self.neighbors = int(neighbors)
        self.similarity_power = float(similarity_power)
        self.prior_strength = float(prior_strength)
        self.minimum_weight = float(minimum_weight)
        self._rows: tuple[OracleInput, ...] | None = None
        self._labels: FloatArray | None = None
        self._global_prior: float | None = None

    def fit(
        self,
        rows: list[OracleInput] | tuple[OracleInput, ...],
        labels: NDArray[np.int64] | list[int],
    ) -> HomologyKnnOracle:
        if not rows:
            raise ValueError("at least one training row is required")
        y = _validated_labels(labels)
        if len(rows) != y.size:
            raise ValueError("rows and labels must have equal length")
        self._rows = tuple(rows)
        self._labels = y
        self._global_prior = _smoothed_prevalence(y, self.prior_strength)
        return self

    def predict_proba(
        self,
        rows: list[OracleInput] | tuple[OracleInput, ...],
    ) -> FloatArray:
        if self._rows is None or self._labels is None or self._global_prior is None:
            raise RuntimeError("model must be fit before prediction")
        return np.asarray([self._predict_one(row) for row in rows], dtype=np.float64)

    def _predict_one(self, query: OracleInput) -> float:
        assert self._rows is not None
        assert self._labels is not None
        assert self._global_prior is not None
        candidates = [index for index, row in enumerate(self._rows) if row.strain == query.strain]
        if not candidates:
            candidates = [index for index, row in enumerate(self._rows) if row.gram == query.gram]
        if not candidates:
            candidates = list(range(len(self._rows)))

        prior_labels = self._labels[candidates]
        context_prior = _smoothed_prevalence(prior_labels, self.prior_strength)
        similarities = [
            global_sequence_identity(query.sequence, self._rows[index].sequence)
            for index in candidates
        ]
        ordered = sorted(
            zip(candidates, similarities, strict=True),
            key=lambda pair: (
                -pair[1],
                self._rows[pair[0]].sequence,
                self._rows[pair[0]].strain,
                pair[0],
            ),
        )[: self.neighbors]
        weights = np.asarray(
            [
                max(similarity**self.similarity_power, self.minimum_weight)
                for _, similarity in ordered
            ],
            dtype=np.float64,
        )
        neighbor_labels = np.asarray([self._labels[index] for index, _ in ordered])
        numerator = float(weights @ neighbor_labels) + self.prior_strength * context_prior
        denominator = float(np.sum(weights)) + self.prior_strength
        if denominator <= 0:  # pragma: no cover - guarded by constructor validation
            return self._global_prior
        return float(np.clip(numerator / denominator, 1e-6, 1.0 - 1e-6))

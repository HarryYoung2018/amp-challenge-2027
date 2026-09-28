"""Exact research-only multi-position commits with policy-independent support.

This is NOT the confidence-ranked production sampler. Its action is an
unordered, uniformly selected k-subset of currently masked positions together
with independent residue draws conditioned on the unchanged pre-commit state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def positive_residue_probabilities(logits: NDArray, epsilon: float) -> NDArray[np.float64]:
    """The fixed uniform mixture gives every allowed residue positive support."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite position-by-residue matrix")
    if isinstance(epsilon, bool) or not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    if len(values) == 0:
        return np.empty_like(values)
    shifted = values - np.max(values, axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= softmax.sum(axis=1, keepdims=True)
    return (1.0 - epsilon) * softmax + epsilon / values.shape[1]


@dataclass(frozen=True, slots=True)
class SubsetCommitDraw:
    positions: tuple[int, ...]
    residues: tuple[int, ...]
    log_probability: float


@dataclass(frozen=True, slots=True)
class UniformSubsetCommitKernel:
    """Complete multi-position kernel, not a one-position surrogate factor."""

    masked_positions: tuple[int, ...]
    commit_count: int
    residue_probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        positions = tuple(self.masked_positions)
        if any(type(value) is not int or value < 0 for value in positions):
            raise ValueError("masked positions must be nonnegative Python integers")
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("masked positions must be sorted and unique")
        if type(self.commit_count) is not int or not 0 <= self.commit_count <= len(positions):
            raise ValueError("commit count must lie between zero and the masked count")
        values = np.array(self.residue_probabilities, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[0] != len(positions):
            raise ValueError("residue probability rows must match masked positions")
        # No unobserved neural probabilities are invented for the empty event.
        # Existing valid positive-probability k=0 inputs remain supported too.
        if values.shape[1] == 0 and self.commit_count == 0:
            values.setflags(write=False)
            object.__setattr__(self, "masked_positions", positions)
            object.__setattr__(self, "residue_probabilities", values)
            return
        if values.shape[1] < 2:
            raise ValueError("an active kernel needs residue probabilities")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("residue probabilities must be finite and strictly positive")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("residue probabilities must sum to one")
        values /= values.sum(axis=1, keepdims=True)
        values.setflags(write=False)
        object.__setattr__(self, "masked_positions", positions)
        object.__setattr__(self, "residue_probabilities", values)

    def log_probability(self, positions: tuple[int, ...], residues: tuple[int, ...]) -> float:
        if len(positions) != self.commit_count or len(residues) != self.commit_count:
            raise ValueError("event must contain the scheduled number of commits")
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("event positions must be a sorted unordered subset")
        if any(position not in self.masked_positions for position in positions):
            raise ValueError("event contains a nonmasked position")
        if any(
            type(value) is not int or not 0 <= value < self.residue_probabilities.shape[1]
            for value in residues
        ):
            raise ValueError("event residue is outside the vocabulary")
        value = -math.log(math.comb(len(self.masked_positions), self.commit_count))
        for position, residue in zip(positions, residues, strict=True):
            value += math.log(
                float(self.residue_probabilities[self.masked_positions.index(position), residue])
            )
        return value

    def sample(self, rng: np.random.Generator) -> SubsetCommitDraw:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("sampling requires an explicit NumPy generator")
        selected = tuple(
            sorted(
                int(value)
                for value in rng.choice(
                    self.masked_positions, size=self.commit_count, replace=False
                )
            )
        )
        residues = tuple(
            int(
                rng.choice(
                    self.residue_probabilities.shape[1],
                    p=self.residue_probabilities[self.masked_positions.index(position)],
                )
            )
            for position in selected
        )
        return SubsetCommitDraw(selected, residues, self.log_probability(selected, residues))


def complete_subset_commit_kl(
    numerator: UniformSubsetCommitKernel, denominator: UniformSubsetCommitKernel
) -> float:
    """Exact conditional KL, including ALL k simultaneous residue commits.

    Position-subset terms cancel because BOTH policies choose the same uniform
    subset law. Inclusion probability k/m yields (k/m) * sum_i KL(p_i || q_i).
    This identity is invalid for the existing confidence-ranked sampler.
    """
    if (
        numerator.masked_positions != denominator.masked_positions
        or numerator.commit_count != denominator.commit_count
    ):
        raise ValueError("paired complete kernels must share their subset support")
    if numerator.commit_count == 0:
        return 0.0
    if numerator.residue_probabilities.shape != denominator.residue_probabilities.shape:
        raise ValueError("paired active kernels must share residue vocabulary support")
    p, q = numerator.residue_probabilities, denominator.residue_probabilities
    terms = p * (np.log(p) - np.log(q))
    value = float(numerator.commit_count / len(numerator.masked_positions) * terms.sum())
    tolerance = 64 * np.finfo(np.float64).eps * max(1.0, float(np.abs(terms).sum()))
    if value < -tolerance:
        raise FloatingPointError("negative complete subset KL beyond roundoff")
    return max(0.0, value)

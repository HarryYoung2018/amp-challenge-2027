"""Fixed-band Laplacian log-spectral densities for residue-contact graphs.

This module deliberately accepts contact matrices rather than constructing a
contact predictor.  A caller must freeze and audit that upstream producer.  The
single public calculation hides graph construction, normalized-Laplacian
eigensolves, global-grid smoothing, normalization, and length-bucketed batching.

The result is an eigenvalue-only Gaussian density on a log-energy grid.  Its
windows are inspired by WKS, but it is neither pointwise WKS (which also uses
eigenvectors) nor NetLSD's heat or wave trace.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LaplacianLogSpectralDensityConfig:
    """Frozen numerical and graph contract for a fixed-band density."""

    bands: int = 32
    minimum_grid_eigenvalue: float = 1e-3
    maximum_grid_eigenvalue: float = 2.0
    bandwidth_grid_steps: float = 1.5
    contact_scale: float = 1.0
    eigenvalue_floor: float = 1e-12
    zero_tolerance: float = 1e-10
    probability_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if isinstance(self.bands, bool) or not isinstance(self.bands, int) or self.bands < 2:
            raise ValueError("bands must be an integer of at least two")
        finite_positive = {
            "minimum_grid_eigenvalue": self.minimum_grid_eigenvalue,
            "maximum_grid_eigenvalue": self.maximum_grid_eigenvalue,
            "bandwidth_grid_steps": self.bandwidth_grid_steps,
            "eigenvalue_floor": self.eigenvalue_floor,
            "zero_tolerance": self.zero_tolerance,
            "probability_tolerance": self.probability_tolerance,
        }
        for name, value in finite_positive.items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.contact_scale, bool)
            or not np.isfinite(self.contact_scale)
            or self.contact_scale < 0.0
        ):
            raise ValueError("contact_scale must be finite and non-negative")
        if self.minimum_grid_eigenvalue >= self.maximum_grid_eigenvalue:
            raise ValueError("minimum_grid_eigenvalue must be below maximum_grid_eigenvalue")
        if self.maximum_grid_eigenvalue > 2.0:
            raise ValueError("normalized-Laplacian bands cannot extend above two")
        if self.eigenvalue_floor >= self.minimum_grid_eigenvalue:
            raise ValueError("eigenvalue_floor must be below minimum_grid_eigenvalue")
        if self.zero_tolerance >= self.minimum_grid_eigenvalue:
            raise ValueError("zero_tolerance must be below minimum_grid_eigenvalue")
        if self.probability_tolerance >= 1.0:
            raise ValueError("probability_tolerance must be below one")
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise ValueError("derived log-grid bandwidth must be finite and positive")

    @property
    def energy_grid(self) -> FloatArray:
        """Return the common log-eigenvalue coordinates used by every graph."""

        return np.linspace(
            np.log(self.minimum_grid_eigenvalue),
            np.log(self.maximum_grid_eigenvalue),
            self.bands,
            dtype=np.float64,
        )

    @property
    def bandwidth(self) -> float:
        """Return the Gaussian width in log-eigenvalue units."""

        return float(self.bandwidth_grid_steps * np.diff(self.energy_grid)[0])


_DEFAULT_CONFIG = LaplacianLogSpectralDensityConfig()


def _validated_contacts(value: FloatArray, *, graph_index: int) -> FloatArray:
    contacts = np.asarray(value, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[0] != contacts.shape[1]:
        raise ValueError(f"contact matrix {graph_index} must be square")
    if contacts.shape[0] < 2:
        raise ValueError(f"contact matrix {graph_index} must contain at least two residues")
    if not np.all(np.isfinite(contacts)):
        raise ValueError(f"contact matrix {graph_index} contains a non-finite value")
    return contacts


def _signature_chunk(
    contact_probabilities: FloatArray,
    config: LaplacianLogSpectralDensityConfig,
) -> FloatArray:
    batch, length, _ = contact_probabilities.shape
    if np.any(contact_probabilities < -config.probability_tolerance) or np.any(
        contact_probabilities > 1.0 + config.probability_tolerance
    ):
        raise ValueError("contact probabilities must lie in [0, 1] within tolerance")
    bounded_contacts = np.clip(contact_probabilities, 0.0, 1.0)
    contacts = 0.5 * (bounded_contacts + np.swapaxes(bounded_contacts, 1, 2))

    positions = np.arange(length)
    separation = np.abs(positions[:, None] - positions[None, :])
    with np.errstate(over="ignore", invalid="ignore"):
        adjacency = np.where(
            separation[None, :, :] >= 3,
            config.contact_scale * contacts,
            0.0,
        )
    backbone = np.arange(length - 1)
    adjacency[:, backbone, backbone + 1] = 1.0
    adjacency[:, backbone + 1, backbone] = 1.0
    adjacency[:, positions, positions] = 0.0
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("weighted residue adjacency contains a non-finite value")

    with np.errstate(over="ignore", invalid="ignore"):
        degrees = np.sum(adjacency, axis=2)
    if not np.all(np.isfinite(degrees)):
        raise ValueError("weighted residue degrees contain a non-finite value")
    if np.any(degrees <= 0.0):
        raise ValueError("residue graph contains an isolated node")
    inverse_sqrt_degree = 1.0 / np.sqrt(degrees)
    normalized_adjacency = (
        inverse_sqrt_degree[:, :, None] * adjacency * inverse_sqrt_degree[:, None, :]
    )
    laplacian = np.eye(length, dtype=np.float64)[None, :, :] - normalized_adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)

    zero_counts = np.sum(np.abs(eigenvalues) <= config.zero_tolerance, axis=1)
    if np.any(zero_counts != 1):
        bad = np.flatnonzero(zero_counts != 1).tolist()
        raise ValueError(f"residue graph must have exactly one zero eigenvalue; chunk rows {bad}")
    if np.any(eigenvalues < -config.zero_tolerance):
        raise ValueError("normalized Laplacian has a negative eigenvalue outside tolerance")
    if np.any(eigenvalues > 2.0 + config.zero_tolerance):
        raise ValueError("normalized Laplacian has an eigenvalue above two outside tolerance")

    positive = np.clip(eigenvalues[:, 1:], config.eigenvalue_floor, 2.0)
    offsets = np.log(positive)[:, :, None] - config.energy_grid[None, None, :]
    density = np.exp(-0.5 * np.square(offsets / config.bandwidth)).mean(axis=1)
    mass = np.sum(density, axis=1, keepdims=True)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("spectral density could not be normalized")
    signature = density / mass
    if signature.shape != (batch, config.bands):
        raise RuntimeError("internal log-spectral density shape mismatch")
    return signature


def laplacian_log_spectral_densities(
    contact_probabilities: Sequence[FloatArray],
    *,
    config: LaplacianLogSpectralDensityConfig = _DEFAULT_CONFIG,
    batch_size: int = 4096,
) -> FloatArray:
    """Return fixed-band log-spectral densities for residue-contact predictions.

    Each input is an ``L x L`` contact-probability matrix.  The implementation
    rejects material range violations, clips tolerance-sized excursions,
    symmetrizes, retains contacts only for sequence separation at least three,
    inserts unit-weight backbone edges, and uses the normalized Laplacian.
    Matrices of different lengths are bucketed before batched ``eigvalsh``
    calls, then scattered back into source order.

    The energy grid belongs to ``config`` and is shared by every graph.  A
    per-graph grid would make corresponding output coordinates incomparable.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    matrices = tuple(
        _validated_contacts(value, graph_index=index)
        for index, value in enumerate(contact_probabilities)
    )
    result = np.empty((len(matrices), config.bands), dtype=np.float64)
    if not matrices:
        return result

    length_buckets: dict[int, list[int]] = defaultdict(list)
    for index, contacts in enumerate(matrices):
        length_buckets[contacts.shape[0]].append(index)

    for length in sorted(length_buckets):
        indices = length_buckets[length]
        for start in range(0, len(indices), batch_size):
            chunk_indices = indices[start : start + batch_size]
            chunk = np.stack([matrices[index] for index in chunk_indices])
            result[chunk_indices] = _signature_chunk(chunk, config)
    return result


__all__ = [
    "LaplacianLogSpectralDensityConfig",
    "laplacian_log_spectral_densities",
]

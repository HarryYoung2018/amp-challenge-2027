"""Transparent, inexpensive descriptors for canonical peptide sequences.

These calculations assume an unmodified linear peptide with free termini and
use approximate average masses/pKa values.  They are useful for data audits and
baseline models, not replacements for experimental measurements or a validated
chemistry package.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .constants import (
    ACIDIC_RESIDUES,
    AROMATIC_RESIDUES,
    BASIC_RESIDUES,
    EISENBERG_HYDROPHOBICITY,
    HYDROPHOBIC_RESIDUES,
    RESIDUE_MASSES_DA,
    WATER_MASS_DA,
)
from .sequences import canonicalize_sequence

_POSITIVE_SIDECHAIN_PKA = {"H": 6.0, "K": 10.5, "R": 12.5}
_NEGATIVE_SIDECHAIN_PKA = {"C": 8.3, "D": 3.9, "E": 4.1, "Y": 10.1}
_N_TERMINUS_PKA = 8.0
_C_TERMINUS_PKA = 3.1


@dataclass(frozen=True, slots=True)
class PeptideDescriptors:
    length: int
    molecular_weight_da: float
    net_charge: float
    charge_density: float
    isoelectric_point: float
    mean_hydrophobicity: float
    hydrophobic_moment: float
    hydrophobic_fraction: float
    aromatic_fraction: float
    basic_fraction: float
    acidic_fraction: float
    shannon_entropy: float
    max_residue_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _canonical(sequence: str) -> str:
    return canonicalize_sequence(sequence)


def molecular_weight(sequence: str) -> float:
    """Approximate average molecular weight in Da for a free-termini peptide."""

    canonical = _canonical(sequence)
    return float(sum(RESIDUE_MASSES_DA[residue] for residue in canonical) + WATER_MASS_DA)


def net_charge(sequence: str, *, ph: float = 7.4) -> float:
    """Approximate net charge using independent Henderson-Hasselbalch groups."""

    if not math.isfinite(ph) or not 0.0 <= ph <= 14.0:
        raise ValueError("ph must be finite and between 0 and 14")
    canonical = _canonical(sequence)
    positive = 1.0 / (1.0 + 10.0 ** (ph - _N_TERMINUS_PKA))
    negative = 1.0 / (1.0 + 10.0 ** (_C_TERMINUS_PKA - ph))
    for residue, pka in _POSITIVE_SIDECHAIN_PKA.items():
        positive += canonical.count(residue) / (1.0 + 10.0 ** (ph - pka))
    for residue, pka in _NEGATIVE_SIDECHAIN_PKA.items():
        negative += canonical.count(residue) / (1.0 + 10.0 ** (pka - ph))
    return positive - negative


def isoelectric_point(
    sequence: str,
    *,
    lower_ph: float = 0.0,
    upper_ph: float = 14.0,
    iterations: int = 60,
) -> float:
    """Estimate pI by bisection on the approximate net-charge function."""

    canonical = _canonical(sequence)
    if not lower_ph < upper_ph:
        raise ValueError("lower_ph must be smaller than upper_ph")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    lower = float(lower_ph)
    upper = float(upper_ph)
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        if net_charge(canonical, ph=midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def hydrophobic_moment(sequence: str, *, angle_degrees: float = 100.0) -> float:
    """Return an alpha-helix hydrophobic moment using the Eisenberg scale."""

    canonical = _canonical(sequence)
    if not math.isfinite(angle_degrees):
        raise ValueError("angle_degrees must be finite")
    angles = np.deg2rad(np.arange(len(canonical), dtype=np.float64) * angle_degrees)
    hydrophobicities = np.asarray(
        [EISENBERG_HYDROPHOBICITY[residue] for residue in canonical],
        dtype=np.float64,
    )
    x_component = float(np.sum(hydrophobicities * np.cos(angles)))
    y_component = float(np.sum(hydrophobicities * np.sin(angles)))
    return math.hypot(x_component, y_component) / len(canonical)


def shannon_entropy(sequence: str) -> float:
    """Residue entropy in bits; low values indicate low complexity."""

    canonical = _canonical(sequence)
    counts = np.asarray(
        [canonical.count(residue) for residue in sorted(set(canonical))],
        dtype=np.float64,
    )
    probabilities = counts / len(canonical)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_descriptors(sequence: str, *, ph: float = 7.4) -> PeptideDescriptors:
    """Compute the complete lightweight descriptor bundle."""

    canonical = _canonical(sequence)
    length = len(canonical)
    charge = net_charge(canonical, ph=ph)
    hydrophobicities = [EISENBERG_HYDROPHOBICITY[r] for r in canonical]
    counts = {residue: canonical.count(residue) for residue in set(canonical)}
    return PeptideDescriptors(
        length=length,
        molecular_weight_da=molecular_weight(canonical),
        net_charge=charge,
        charge_density=charge / length,
        isoelectric_point=isoelectric_point(canonical),
        mean_hydrophobicity=float(np.mean(hydrophobicities)),
        hydrophobic_moment=hydrophobic_moment(canonical),
        hydrophobic_fraction=sum(r in HYDROPHOBIC_RESIDUES for r in canonical) / length,
        aromatic_fraction=sum(r in AROMATIC_RESIDUES for r in canonical) / length,
        basic_fraction=sum(r in BASIC_RESIDUES for r in canonical) / length,
        acidic_fraction=sum(r in ACIDIC_RESIDUES for r in canonical) / length,
        shannon_entropy=shannon_entropy(canonical),
        max_residue_fraction=max(counts.values()) / length,
    )


# Readable alias for feature-pipeline code.
describe_sequence = compute_descriptors

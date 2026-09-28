"""Deterministic, competition-valid baseline peptide generator.

This module exists to exercise the complete submission and selection path before
trained models are available.  Its scores are transparent physicochemical
heuristics, not claims of antimicrobial activity or safety.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from amp_challenge.constants import (
    AROMATIC_RESIDUES,
    EISENBERG_HYDROPHOBICITY,
    HYDROPHOBIC_RESIDUES,
    STANDARD_AMINO_ACIDS,
)
from amp_challenge.models import EndpointSpec, ModelPrediction, OracleEnsemble

FloatArray = NDArray[np.float64]

FEATURE_NAMES = (
    "length",
    "net_charge_proxy",
    "charge_density",
    "hydrophobic_fraction",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "aromatic_fraction",
    "proline_fraction",
    "glycine_fraction",
    "entropy_fraction",
    "max_residue_fraction",
    "synthesis_proxy",
)

OBJECTIVE_NAMES = (
    "broad_spectrum",
    "gram_positive",
    "gram_negative",
    "mdr_eskape",
    "selectivity",
)


@dataclass(frozen=True)
class FamilySpec:
    name: str
    probability: float
    mean_length: float
    length_std: float
    min_length: int
    max_length: int
    residue_weights: dict[str, float]
    amphipathic_pattern: bool = False


@dataclass(frozen=True)
class GeneratedPool:
    sequences: tuple[str, ...]
    families: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.sequences) != len(self.families):
            raise ValueError("sequences and families must have the same length")
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("generated sequences must be unique")


def _weights(**values: float) -> dict[str, float]:
    return values


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "amphipathic_alpha",
        0.30,
        22,
        5,
        12,
        36,
        _weights(A=8, F=4, G=5, I=6, K=14, L=11, N=2, Q=3, R=10, S=4, T=2, V=6, W=2, Y=2),
        amphipathic_pattern=True,
    ),
    FamilySpec(
        "balanced_cationic",
        0.25,
        19,
        5,
        10,
        34,
        _weights(
            A=8,
            D=1,
            E=1,
            F=4,
            G=7,
            H=2,
            I=5,
            K=13,
            L=9,
            M=1,
            N=3,
            P=2,
            Q=3,
            R=9,
            S=5,
            T=3,
            V=5,
            W=2,
            Y=2,
        ),
    ),
    FamilySpec(
        "gly_trp_cationic",
        0.15,
        15,
        3,
        9,
        24,
        _weights(A=5, F=3, G=18, H=2, I=3, K=13, L=5, N=2, Q=2, R=13, S=4, T=2, V=3, W=8, Y=2),
    ),
    FamilySpec(
        "short_arginine_rich",
        0.10,
        13,
        2.5,
        8,
        20,
        _weights(A=6, F=4, G=6, I=3, K=12, L=5, N=2, Q=2, R=20, S=5, T=2, V=3, W=4, Y=2),
    ),
    FamilySpec(
        "proline_rich",
        0.10,
        21,
        5,
        12,
        35,
        _weights(A=6, F=2, G=8, H=1, I=3, K=12, L=4, N=3, P=14, Q=4, R=12, S=6, T=3, V=3, W=1, Y=1),
    ),
    FamilySpec(
        "broad_composition",
        0.10,
        27,
        7,
        12,
        45,
        _weights(
            A=8,
            D=2,
            E=2,
            F=4,
            G=7,
            H=2,
            I=5,
            K=10,
            L=8,
            M=1,
            N=4,
            P=3,
            Q=4,
            R=8,
            S=6,
            T=4,
            V=5,
            W=2,
            Y=2,
        ),
    ),
)


def generate_baseline_pool(
    n_sequences: int,
    *,
    seed: int = 42,
    forbidden: Collection[str] = (),
) -> GeneratedPool:
    """Generate unique sequences from several transparent composition priors."""

    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    rng = np.random.default_rng(seed)
    names = np.asarray([family.name for family in FAMILIES], dtype=object)
    probabilities = np.asarray([family.probability for family in FAMILIES], dtype=float)
    probabilities /= probabilities.sum()
    by_name = {family.name: family for family in FAMILIES}
    forbidden_set = {sequence.strip().upper() for sequence in forbidden}
    seen: set[str] = set()
    sequences: list[str] = []
    families: list[str] = []
    max_attempts = max(10_000, n_sequences * 100)

    for _ in range(max_attempts):
        family_name = str(rng.choice(names, p=probabilities))
        family = by_name[family_name]
        length = int(
            np.clip(
                round(float(rng.normal(family.mean_length, family.length_std))),
                family.min_length,
                family.max_length,
            )
        )
        sequence = _sample_sequence(family, length, rng)
        if sequence in seen or sequence in forbidden_set or not _passes_basic_realism(sequence):
            continue
        seen.add(sequence)
        sequences.append(sequence)
        families.append(family.name)
        if len(sequences) == n_sequences:
            return GeneratedPool(tuple(sequences), tuple(families))
    raise RuntimeError(
        f"generated only {len(sequences)} unique sequences after {max_attempts} attempts"
    )


def _sample_sequence(family: FamilySpec, length: int, rng: np.random.Generator) -> str:
    alphabet = np.asarray(STANDARD_AMINO_ACIDS)
    base_probabilities = np.asarray(
        [family.residue_weights.get(residue, 0.0) for residue in alphabet], dtype=float
    )
    base_probabilities /= base_probabilities.sum()
    if not family.amphipathic_pattern:
        return "".join(rng.choice(alphabet, size=length, p=base_probabilities).tolist())

    # Approximate one hydrophobic helical face using the 100-degree residue turn.
    angles = np.deg2rad(np.arange(length) * 100.0)
    hydrophobic_face = np.cos(angles) >= 0.15
    hydrophobic_weights = np.asarray(
        [
            family.residue_weights.get(residue, 0.0) if residue in HYDROPHOBIC_RESIDUES else 0.0
            for residue in alphabet
        ],
        dtype=float,
    )
    polar_weights = np.asarray(
        [
            family.residue_weights.get(residue, 0.0)
            if residue not in HYDROPHOBIC_RESIDUES
            else 0.15 * family.residue_weights.get(residue, 0.0)
            for residue in alphabet
        ],
        dtype=float,
    )
    hydrophobic_weights /= hydrophobic_weights.sum()
    polar_weights /= polar_weights.sum()
    output = np.empty(length, dtype="<U1")
    output[hydrophobic_face] = rng.choice(
        alphabet, size=int(hydrophobic_face.sum()), p=hydrophobic_weights
    )
    output[~hydrophobic_face] = rng.choice(
        alphabet, size=int((~hydrophobic_face).sum()), p=polar_weights
    )
    return "".join(output.tolist())


def _passes_basic_realism(sequence: str) -> bool:
    counts = {residue: sequence.count(residue) for residue in set(sequence)}
    if len(counts) < 4 or max(counts.values()) > max(4, math.ceil(0.35 * len(sequence))):
        return False
    if any(
        sequence[index] == sequence[index - 1] == sequence[index - 2] == sequence[index - 3]
        for index in range(3, len(sequence))
    ):
        return False
    charge = sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")
    hydrophobic_fraction = sum(residue in HYDROPHOBIC_RESIDUES for residue in sequence) / len(
        sequence
    )
    return charge >= 1 and 0.18 <= hydrophobic_fraction <= 0.75


def baseline_feature_matrix(sequences: Sequence[str]) -> FloatArray:
    """Compute inexpensive features used by the runnable baseline and audit."""

    features = np.empty((len(sequences), len(FEATURE_NAMES)), dtype=np.float64)
    max_width = max(map(len, sequences), default=0)
    cosines = np.cos(np.deg2rad(np.arange(max_width) * 100.0))
    sines = np.sin(np.deg2rad(np.arange(max_width) * 100.0))
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        counts = {residue: sequence.count(residue) for residue in set(sequence)}
        charge = (
            sequence.count("K")
            + sequence.count("R")
            + 0.04 * sequence.count("H")
            - sequence.count("D")
            - sequence.count("E")
        )
        hydrophobic_values = np.asarray([EISENBERG_HYDROPHOBICITY[residue] for residue in sequence])
        moment = (
            math.hypot(
                float(hydrophobic_values @ cosines[:length]),
                float(hydrophobic_values @ sines[:length]),
            )
            / length
        )
        probabilities = np.asarray(list(counts.values()), dtype=float) / length
        entropy_fraction = float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(20))
        hydro_fraction = sum(residue in HYDROPHOBIC_RESIDUES for residue in sequence) / length
        aromatic_fraction = sum(residue in AROMATIC_RESIDUES for residue in sequence) / length
        max_fraction = max(counts.values()) / length
        synthesis_penalty = (
            max(0.0, (length - 32) / 18)
            + max(0.0, (abs(charge) - 9) / 8)
            + max(0.0, (hydro_fraction - 0.65) / 0.2)
            + max(0.0, (aromatic_fraction - 0.25) / 0.2)
            + max(0.0, (max_fraction - 0.30) / 0.2)
            + 0.08 * sequence.count("M")
            + 0.12 * sequence.count("C")
        )
        features[row] = (
            length,
            charge,
            charge / length,
            hydro_fraction,
            float(np.mean(hydrophobic_values)),
            moment,
            aromatic_fraction,
            sequence.count("P") / length,
            sequence.count("G") / length,
            entropy_fraction,
            max_fraction,
            math.exp(-synthesis_penalty),
        )
    return features


def baseline_proxy_ensemble(features: FloatArray):
    """Return three heuristic views through the production ensemble contract."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have {len(FEATURE_NAMES)} columns")
    endpoints = [
        EndpointSpec(name, objective_weight=1.0, risk_penalty=1.0) for name in OBJECTIVE_NAMES
    ]
    ensemble = OracleEnsemble(endpoints)
    members: list[ModelPrediction] = []
    for model_index, shift in enumerate((-1.0, 0.0, 1.0)):
        means = _proxy_member(values, shift)
        base_uncertainty = (
            0.05 + 0.12 * (1.0 - values[:, 11]) + 0.10 * np.maximum(0.0, 0.65 - values[:, 9])
        )
        std = np.clip(
            base_uncertainty[:, None] * np.asarray([1.0, 0.9, 1.1, 1.15, 1.2])[None, :],
            0.03,
            0.35,
        )
        members.append(
            ModelPrediction(
                name=("composition", "membrane_balance", "synthesis_conservative")[model_index],
                mean=means,
                std=std,
                weight=np.asarray([1.0, 0.9, 1.1, 1.0, 1.1]),
            )
        )
    return ensemble, ensemble.aggregate(members)


def _proxy_member(features: FloatArray, shift: float) -> FloatArray:
    length = features[:, 0]
    charge_density = features[:, 2]
    hydro = features[:, 3]
    moment = features[:, 5]
    aromatic = features[:, 6]
    proline = features[:, 7]
    entropy = features[:, 9]
    synthesis = features[:, 11]

    charge_good = _gaussian(charge_density, 0.22 + 0.015 * shift, 0.14)
    hydro_good = _gaussian(hydro, 0.46 + 0.015 * shift, 0.17)
    length_good = _gaussian(length, 21.0 + shift, 10.0)
    moment_good = _gaussian(moment, 0.50 + 0.03 * shift, 0.40)
    complexity_good = np.clip((entropy - 0.45) / 0.45, 0.0, 1.0)
    membrane = 0.30 * charge_good + 0.25 * hydro_good + 0.25 * moment_good + 0.20 * length_good
    broad = 0.60 * membrane + 0.20 * complexity_good + 0.20 * synthesis
    gram_positive = (
        0.50 * membrane
        + 0.25 * _gaussian(hydro, 0.52, 0.18)
        + 0.15 * length_good
        + 0.10 * synthesis
    )
    gram_negative = (
        0.45 * membrane
        + 0.30 * _gaussian(charge_density, 0.28, 0.16)
        + 0.15 * moment_good
        + 0.10 * synthesis
    )
    mdr = 0.45 * broad + 0.25 * gram_negative + 0.15 * complexity_good + 0.15 * synthesis
    toxicity_proxy = np.clip(
        0.55 * np.maximum(0.0, (hydro - 0.48) / 0.30)
        + 0.25 * np.maximum(0.0, (aromatic - 0.12) / 0.25)
        + 0.20 * np.maximum(0.0, (charge_density - 0.38) / 0.30),
        0.0,
        1.0,
    )
    # Proline-rich peptides are not assumed helical; retain them as a distinct
    # exploration family without giving the helix moment undue influence.
    selectivity = (
        0.45 * broad
        + 0.35 * (1.0 - toxicity_proxy)
        + 0.20 * synthesis
        + 0.05 * np.minimum(proline / 0.20, 1.0)
    )
    return np.clip(
        np.column_stack([broad, gram_positive, gram_negative, mdr, selectivity]),
        0.0,
        1.0,
    )


def baseline_embeddings(features: FloatArray) -> FloatArray:
    """Standardized descriptor embedding for baseline diversity selection."""

    chosen = np.asarray(features, dtype=np.float64)[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
    scale = np.std(chosen, axis=0)
    return (chosen - np.mean(chosen, axis=0)) / np.where(scale > 1e-12, scale, 1.0)


def novelty_proxy(features: FloatArray) -> FloatArray:
    """Complexity proxy used only before exact organizer-reference screening."""

    values = np.asarray(features, dtype=np.float64)
    return np.clip(
        0.55 * values[:, 9] + 0.30 * (1.0 - values[:, 10]) + 0.15 * values[:, 11],
        0.0,
        1.0,
    )


def coarse_cluster_ids(features: FloatArray, families: Sequence[str]) -> tuple[str, ...]:
    """Create coarse strata for the runnable baseline's cluster-cap demo.

    These are not substitutes for sequence-homology clusters in final models.
    """

    values = np.asarray(features)
    return tuple(
        f"{family}:l{int(values[index, 0] // 4)}:q{int(np.floor(values[index, 2] / 0.08))}:h{int(np.floor(values[index, 3] / 0.10))}"
        for index, family in enumerate(families)
    )


def _gaussian(values: FloatArray, center: float, width: float) -> FloatArray:
    return np.exp(-0.5 * np.square((values - center) / width))

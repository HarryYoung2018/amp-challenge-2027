"""Auditable allocation, paired credit and native operator planning."""

from __future__ import annotations

import math

import numpy as np

from amp_challenge.acquisition.counterfactual import linear_paired_contrast
from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule
from amp_challenge.generators.diffusion.native_endpoint import _seed
from amp_challenge.generators.diffusion.native_evolution_records import EvolutionContrast


def branch_score(branch, *, counterfactual: bool):
    alpha = 1 + branch.useful_descendants
    beta = 1 + branch.charged_descendants - branch.useful_descendants
    mean = alpha / (alpha + beta)
    sd = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
    return (branch.credit if counterfactual else 1.0) * (mean + sd)


def allocate_branches(branches, *, counterfactual: bool):
    if len(branches) != 4 or tuple(branch.branch for branch in branches) != (0, 1, 2, 3):
        raise ValueError("four canonical branches required")
    scores = np.array([branch_score(branch, counterfactual=counterfactual) for branch in branches])
    ideal = 16 * scores / scores.sum()
    floors = np.floor(ideal).astype(int)
    for index in sorted(range(4), key=lambda row: (-(ideal[row] - floors[row]), row))[
        : 16 - int(floors.sum())
    ]:
        floors[index] += 1
    return tuple(map(float, scores)), tuple(int(count + 2) for count in floors)


def paired_credit(belief, *, feasible: bool):
    if belief.n_points != 2 or belief.n_outputs != 2:
        raise ValueError("paired contrast requires child/parent two-output joint")
    return contrast_from_moments(belief.mean, belief.covariance.reshape(4, 4), feasible=feasible)


def contrast_from_moments(mean, covariance, *, feasible: bool):
    moments = linear_paired_contrast(mean[0], mean[1], covariance, (0.5, 0.5))
    weights = np.array([0.5, 0.5])
    absolute_mean = float(mean[0] @ weights)
    variance = max(0.0, float(weights @ covariance[:2, :2] @ weights))
    advantage = moments.mean - math.sqrt(moments.variance)
    absolute_risk = absolute_mean - math.sqrt(variance)
    return EvolutionContrast(
        moments.mean,
        moments.variance,
        advantage,
        absolute_mean,
        absolute_risk,
        feasible,
        bool(feasible and advantage > 0 and absolute_risk >= 0.5),
    )


def operator_plan(unit, parent: str, operator: str, *, seed: int, ordinal: int):
    length_logp = 0.0
    if operator == "full_regeneration":
        index = int(_seed(seed, ordinal, "evolution-full-length", 0).integers(len(unit.sequences)))
        length = len(unit.sequences[index])
        # The sampled event is LENGTH; the all-masked template is a canonical
        # representative, not an extra unaccounted stochastic peptide draw.
        template = min(seq for seq in unit.sequences if len(seq) == length)
        mass = sum(len(seq) == len(template) for seq in unit.sequences) / len(unit.sequences)
        return template, unit.model.config.levels, math.log(mass)
    fractions = {"single_site": 0.0, "quarter_remask": 0.25, "half_remask": 0.5}
    if operator not in fractions:
        raise ValueError("unknown native evolutionary operator")
    desired = max(1, math.ceil(len(parent) * fractions[operator]))
    counts = CosineMaskSchedule().mask_counts(
        np.full(unit.model.config.levels, len(parent)),
        np.arange(1, unit.model.config.levels + 1),
        total_levels=unit.model.config.levels,
    )
    level = int(np.flatnonzero(counts >= desired)[0]) + 1
    if operator == "single_site" and counts[level - 1] != 1:
        raise ValueError("native schedule cannot represent exact one-site operator")
    return parent, level, length_logp


def contrast_credit_value(advantage: float):
    return math.exp(float(np.clip(advantage / 0.1, math.log(0.25), math.log(4))))

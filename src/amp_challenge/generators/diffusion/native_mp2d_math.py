"""Clean-room numerical primitives for the declared MP2D-style adaptation.

Complete native subset probabilities are conditional model event probabilities,
not endpoint or search-selection likelihoods. No oracle or training occurs here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from amp_challenge.generators.diffusion.categorical import CosineMaskSchedule, PeptideVocabulary
from amp_challenge.generators.diffusion.model import MASK_TOKEN_INDEX
from amp_challenge.generators.diffusion.native_endpoint import (
    NATIVE_ENDPOINT_DEFAULTS,
    NativeTransitionState,
    _json_hash,
    _validated_transition_kernels,
)
from amp_challenge.generators.diffusion.subset_kernel import SubsetCommitDraw

CONFIG_SHA256 = "e93ea6425335ffa035cce7802c13f1cbae3afa1860967d8f517e33c2bda33727"
DIRECTIONS = tuple((index / 63, 1 - index / 63) for index in range(64))


def keyed_rng(stream: str, *key: object) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64DXSM(int(_json_hash(["native-mp2d-v1", stream, key])[:32], 16))
    )


def pareto_indices(values, *, deduplicate=True) -> tuple[int, ...]:
    points = np.asarray(values, dtype=np.float64)
    if points.size == 0:
        return ()
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Pareto values must be finite two-objective vectors")
    return tuple(
        index
        for index, value in enumerate(points)
        if not any(
            np.all(other >= value)
            and (
                np.any(other > value)
                or (deduplicate and previous < index and np.array_equal(other, value))
            )
            for previous, other in enumerate(points)
            if previous != index
        )
    )


def vector_ucb(rewards, visits, priors, parent_visits: int) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    visits, priors = np.asarray(visits, dtype=float), np.asarray(priors, dtype=float)
    if (
        rewards.shape != (len(visits), 2)
        or priors.shape != visits.shape
        or not np.isfinite(rewards).all()
        or not np.isfinite(visits).all()
        or not np.isfinite(priors).all()
        or np.any(visits < 0)
        or np.any(priors < 0)
        or np.any(priors > 1)
        or type(parent_visits) is not int
        or parent_visits < 0
    ):
        raise ValueError("invalid vector-UCB statistics")
    q = np.divide(rewards, visits[:, None], out=np.zeros_like(rewards), where=visits[:, None] > 0)
    return q + (0.1 * priors * math.sqrt(parent_visits) / (1 + visits))[:, None]


@dataclass(frozen=True, slots=True)
class AngularDecision:
    retained: tuple[int, ...]
    mode: str
    cosines: tuple[float, ...]
    rejection_fraction: float
    new_ema: float
    new_angle: float


def angular_decision(parent, children, feasible, direction, angle, ema) -> AngularDecision:
    values = np.asarray(children, dtype=float)
    parent, direction = np.asarray(parent, dtype=float), np.asarray(direction, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(values) == 0
        or parent.shape != (2,)
        or direction.shape != (2,)
        or not np.isfinite(values).all()
        or not np.isfinite(parent).all()
        or not np.isfinite(direction).all()
        or np.linalg.norm(direction) == 0
        or len(feasible) != len(values)
        or any(type(value) is not bool for value in feasible)
        or not 15 <= angle <= 75
        or not 0 <= ema <= 1
    ):
        raise ValueError("invalid angular gate inputs")
    delta = values - parent
    norms = np.linalg.norm(delta, axis=1) * np.linalg.norm(direction)
    cosines = np.divide(delta @ direction, norms, out=np.zeros(len(values)), where=norms > 0)
    cosines = np.clip(cosines, -1, 1)
    primary = tuple(
        index
        for index, (cosine, valid) in enumerate(zip(cosines, feasible, strict=True))
        if valid and norms[index] > 0 and cosine >= math.cos(math.radians(angle))
    )
    rejected = 1 - len(primary) / len(values)
    updated_ema = 0.5 * ema + 0.5 * rejected
    updated_angle = float(np.clip(angle * math.exp(updated_ema - 0.3), 15, 75))
    retained = primary or tuple(
        index for index, valid in enumerate(feasible) if valid and cosines[index] > 0
    )
    return AngularDecision(
        retained,
        "angular" if primary else "positive_alignment" if retained else "blocked_parent",
        tuple(map(float, cosines)),
        rejected,
        updated_ema,
        updated_angle,
    )


def archive_rewards(children, archive) -> np.ndarray:
    children = np.asarray(children, dtype=float)
    if children.size == 0:
        return np.empty((0, 2), dtype=float)
    if not len(archive):
        return np.ones_like(children)
    return np.mean(np.asarray(archive)[None, :, :] <= children[:, None, :], axis=1)


def mpi_values(improvements, direction, noise_level: int) -> np.ndarray:
    delta = np.asarray(improvements, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (
        delta.ndim != 2
        or delta.shape[1] != 2
        or not len(delta)
        or not np.isfinite(delta).all()
        or direction.shape != (2,)
        or not np.isfinite(direction).all()
        or type(noise_level) is not int
        or noise_level < 1
    ):
        raise ValueError("MPI needs finite vectors and positive refinement noise")
    ranks = np.empty_like(delta)
    for objective in range(2):
        column = delta[:, objective]
        for row, value in enumerate(column):
            ranks[row, objective] = (
                np.count_nonzero(column < value) + (np.count_nonzero(column == value) + 1) / 2
            )
    rank_term = np.mean(direction * ranks / noise_level, axis=1)
    alignment = delta @ direction

    def zscore(values):
        scale = float(np.std(values))
        return np.zeros_like(values) if scale == 0 else (values - np.mean(values)) / scale

    return zscore(rank_term) + zscore(alignment)


def stable_softmax(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("softmax input must be a finite vector")
    result = np.exp(values - max(values))
    return result / result.sum()


def remask(model, sequence, level, stream, ordinal):
    tokens = PeptideVocabulary().encode((sequence,), max_length=model.config.max_length).tokens[0]
    count = int(
        CosineMaskSchedule().mask_counts(len(sequence), level, total_levels=model.config.levels)[0]
    )
    positions = tuple(
        sorted(
            map(
                int,
                keyed_rng(stream, "remask", ordinal).choice(
                    len(sequence), size=count, replace=False
                ),
            )
        )
    )
    tokens[list(positions)] = MASK_TOKEN_INDEX
    return tokens, positions, -math.log(math.comb(len(sequence), count))


def event_draw(kernel, *, rng=None) -> SubsetCommitDraw:
    """Gumbel categorical exact law, or deterministic complete-event MAP."""
    if not kernel.commit_count:
        return SubsetCommitDraw((), (), 0.0)
    probabilities = kernel.residue_probabilities
    if rng is None:
        indices = sorted(
            range(len(kernel.masked_positions)),
            key=lambda row: (-float(probabilities[row].max()), kernel.masked_positions[row]),
        )[: kernel.commit_count]
        positions = tuple(sorted(kernel.masked_positions[row] for row in indices))
        residues = tuple(
            int(np.argmax(probabilities[kernel.masked_positions.index(pos)])) for pos in positions
        )
    else:
        positions = tuple(
            sorted(
                map(
                    int,
                    rng.choice(kernel.masked_positions, size=kernel.commit_count, replace=False),
                )
            )
        )
        residues = tuple(
            int(
                np.argmax(
                    np.log(probabilities[kernel.masked_positions.index(pos)])
                    + rng.gumbel(size=probabilities.shape[1])
                )
            )
            for pos in positions
        )
    return SubsetCommitDraw(positions, residues, kernel.log_probability(positions, residues))


def commit(tokens, level, draw, mode):
    after = np.array(tokens, copy=True)
    after[list(draw.positions)] = draw.residues
    return after, {
        "level": level,
        "before_sha256": _json_hash(tokens.tolist()),
        "after_sha256": _json_hash(after.tolist()),
        "positions": list(draw.positions),
        "residues": list(draw.residues),
        "base_log_probability": draw.log_probability,
        "mode": mode,
    }


def greedy_completions(rows, *, check=lambda: None):
    """Batched heterogeneous-model native rollouts; caller already binds models.

    rows = (model, copied token array, length, first level). Every level including
    empty events is recorded. All neural calls use accepted bounded primitives.
    """
    working = [np.array(row[1], copy=True) for row in rows]
    traces = [[] for _ in rows]
    for level in range(max((row[3] for row in rows), default=0), 0, -1):
        check()
        groups = {}
        for index, row in enumerate(rows):
            if row[3] >= level:
                groups.setdefault(id(row[0]), []).append(index)
        for indices in groups.values():
            model = rows[indices[0]][0]
            for start in range(0, len(indices), 128):
                chunk = indices[start : start + 128]
                states = tuple(
                    NativeTransitionState(working[index], rows[index][2], level) for index in chunk
                )
                kernels = _validated_transition_kernels(model, states, NATIVE_ENDPOINT_DEFAULTS)
                for index, kernel in zip(chunk, kernels, strict=True):
                    working[index], step = commit(
                        working[index], level, event_draw(kernel), "greedy_map"
                    )
                    traces[index].append(step)
    return tuple(
        (PeptideVocabulary().decode(tokens[None, :])[0], traces[index])
        for index, tokens in enumerate(working)
    )

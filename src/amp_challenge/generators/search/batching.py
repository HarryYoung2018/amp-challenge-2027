"""Deterministic length-bucketed plans for bounded sequence batches."""

from __future__ import annotations

import tomllib
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """Hard sequence-count and padded-token limits for length buckets."""

    max_sequences: int
    max_tokens: int
    length_bucket_boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.max_sequences, name="max_sequences")
        _positive_integer(self.max_tokens, name="max_tokens")
        boundaries = self.length_bucket_boundaries
        if not isinstance(boundaries, tuple) or not boundaries:
            raise ValueError("length_bucket_boundaries must be a non-empty tuple")
        for boundary in boundaries:
            _positive_integer(boundary, name="length bucket boundary")
        if any(left >= right for left, right in pairwise(boundaries)):
            raise ValueError("length_bucket_boundaries must be strictly increasing")


@dataclass(frozen=True, slots=True)
class IndexedBatch:
    """One batch plus the source indices needed to scatter results back."""

    indices: tuple[int, ...]
    padded_width: int
    token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.indices, tuple) or not self.indices:
            raise ValueError("indices must be a non-empty tuple")
        if any(type(index) is not int or index < 0 for index in self.indices):
            raise ValueError("indices must contain non-negative integers")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices must be unique")
        width = _positive_integer(self.padded_width, name="padded_width")
        tokens = _positive_integer(self.token_count, name="token_count")
        if tokens != len(self.indices) * width:
            raise ValueError("token_count must equal len(indices) * padded_width")


@dataclass(frozen=True, slots=True)
class CountBatch:
    """One half-open item range for a fixed-size batched stage."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be a non-negative integer")
        if type(self.stop) is not int or self.stop <= self.start:
            raise ValueError("stop must be an integer greater than start")

    @property
    def size(self) -> int:
        """Return the number of items in the half-open range."""

        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class BatchExecutionPlan:
    """Versioned upper limits for batch-first search and replay execution."""

    profile: str
    rollout_batch_size: int
    proposal_batch_size: int
    surrogate_batch_size: int
    kg_candidate_chunk_size: int
    kg_fantasy_chunk_size: int
    kg_max_joint_size: int
    kg_max_combinations: int
    oracle_batch_size: int
    replay_limits: BatchLimits
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        object.__setattr__(self, "profile", self.profile.strip())
        for name in (
            "rollout_batch_size",
            "proposal_batch_size",
            "surrogate_batch_size",
            "kg_candidate_chunk_size",
            "kg_fantasy_chunk_size",
            "kg_max_joint_size",
            "kg_max_combinations",
            "oracle_batch_size",
            "gradient_accumulation_steps",
        ):
            _positive_integer(getattr(self, name), name=name)
        if not isinstance(self.replay_limits, BatchLimits):
            raise TypeError("replay_limits must be BatchLimits")

    def effective_replay_sequence_batch(
        self,
        *,
        padded_width: int | None = None,
        world_size: int = 1,
    ) -> int:
        """Return the token-aware accumulated data-parallel sequence batch.

        When no width is supplied, use the longest configured bucket so the
        advertised profile capacity is conservative rather than assuming that
        every sequence is short enough to hit the count ceiling.
        """

        workers = _positive_integer(world_size, name="world_size")
        width = (
            self.replay_limits.length_bucket_boundaries[-1]
            if padded_width is None
            else _positive_integer(padded_width, name="padded_width")
        )
        microbatch = self.replay_sequence_microbatch(padded_width=width)
        return microbatch * self.gradient_accumulation_steps * workers

    def replay_sequence_microbatch(self, *, padded_width: int) -> int:
        """Return the per-worker row capacity at one realized padded width."""

        width = _positive_integer(padded_width, name="padded_width")
        if width > self.replay_limits.max_tokens:
            raise ValueError("padded_width exceeds replay max_tokens")
        return min(
            self.replay_limits.max_sequences,
            self.replay_limits.max_tokens // width,
        )

    def effective_replay_token_batch(self, *, world_size: int = 1) -> int:
        """Return padded-token cap times accumulation times workers."""

        workers = _positive_integer(world_size, name="world_size")
        return self.replay_limits.max_tokens * self.gradient_accumulation_steps * workers


_BATCH_EXECUTION_KEYS = {
    "profile",
    "rollout_batch_size",
    "proposal_batch_size",
    "surrogate_batch_size",
    "kg_candidate_chunk_size",
    "kg_fantasy_chunk_size",
    "kg_max_joint_size",
    "kg_max_combinations",
    "oracle_batch_size",
    "replay_max_sequences",
    "replay_max_tokens",
    "replay_length_bucket_boundaries",
    "gradient_accumulation_steps",
}


def batch_execution_plan_from_mapping(values: dict[str, Any]) -> BatchExecutionPlan:
    """Parse one strict ``[batching]`` mapping shared by smoke and cluster plans."""

    unknown = sorted(set(values) - _BATCH_EXECUTION_KEYS)
    missing = sorted(_BATCH_EXECUTION_KEYS - set(values))
    if unknown:
        raise ValueError(f"unknown batching configuration keys: {unknown}")
    if missing:
        raise ValueError(f"missing batching configuration keys: {missing}")
    boundaries = values["replay_length_bucket_boundaries"]
    if not isinstance(boundaries, list):
        raise ValueError("replay_length_bucket_boundaries must be a TOML array")
    return BatchExecutionPlan(
        profile=values["profile"],
        rollout_batch_size=values["rollout_batch_size"],
        proposal_batch_size=values["proposal_batch_size"],
        surrogate_batch_size=values["surrogate_batch_size"],
        kg_candidate_chunk_size=values["kg_candidate_chunk_size"],
        kg_fantasy_chunk_size=values["kg_fantasy_chunk_size"],
        kg_max_joint_size=values["kg_max_joint_size"],
        kg_max_combinations=values["kg_max_combinations"],
        oracle_batch_size=values["oracle_batch_size"],
        replay_limits=BatchLimits(
            max_sequences=values["replay_max_sequences"],
            max_tokens=values["replay_max_tokens"],
            length_bucket_boundaries=tuple(boundaries),
        ),
        gradient_accumulation_steps=values["gradient_accumulation_steps"],
    )


def load_batch_execution_plan(path: Path | str) -> BatchExecutionPlan:
    """Load a standalone, schema-versioned cluster or smoke batch plan."""

    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"schema_version", "batching"}:
        raise ValueError("batch-plan files require exactly schema_version and batching")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("batch-plan schema_version must be 1")
    batching = raw["batching"]
    if not isinstance(batching, dict):
        raise ValueError("batching must be a TOML table")
    return batch_execution_plan_from_mapping(batching)


def plan_count_batches(total_items: int, max_batch_size: int) -> tuple[CountBatch, ...]:
    """Split a count into stable half-open ranges bounded by ``max_batch_size``.

    The planner stores only range metadata: even a very large requested
    workload does not allocate proposal, posterior, or oracle arrays.
    """

    if type(total_items) is not int or total_items < 0:
        raise ValueError("total_items must be a non-negative integer")
    batch_size = _positive_integer(max_batch_size, name="max_batch_size")
    return tuple(
        CountBatch(start=start, stop=min(start + batch_size, total_items))
        for start in range(0, total_items, batch_size)
    )


def plan_length_bucketed_batches(
    lengths: Sequence[int],
    limits: BatchLimits,
) -> tuple[IndexedBatch, ...]:
    """Plan stable batches while respecting count and padded-token caps.

    Each length is assigned to the smallest inclusive bucket boundary that
    covers it. Buckets are visited in boundary order and source order is
    retained within a bucket. The planner pads only to the longest member of a
    realized batch, rather than to the bucket boundary.
    """

    if not isinstance(limits, BatchLimits):
        raise TypeError("limits must be BatchLimits")
    parsed_lengths = tuple(lengths)
    if not parsed_lengths:
        return ()

    buckets: list[list[tuple[int, int]]] = [[] for _boundary in limits.length_bucket_boundaries]
    for index, length in enumerate(parsed_lengths):
        parsed = _positive_integer(length, name=f"lengths[{index}]")
        if parsed > limits.max_tokens:
            raise ValueError(f"lengths[{index}] exceeds max_tokens")
        bucket_index = bisect_left(limits.length_bucket_boundaries, parsed)
        if bucket_index == len(limits.length_bucket_boundaries):
            raise ValueError(f"lengths[{index}] exceeds the final bucket boundary")
        buckets[bucket_index].append((index, parsed))

    planned: list[IndexedBatch] = []
    for bucket in buckets:
        batch_indices: list[int] = []
        padded_width = 0
        for index, length in bucket:
            projected_width = max(padded_width, length)
            projected_size = len(batch_indices) + 1
            exceeds_limit = (
                projected_size > limits.max_sequences
                or projected_size * projected_width > limits.max_tokens
            )
            if batch_indices and exceeds_limit:
                planned.append(
                    IndexedBatch(
                        indices=tuple(batch_indices),
                        padded_width=padded_width,
                        token_count=len(batch_indices) * padded_width,
                    )
                )
                batch_indices = []
                padded_width = 0
                projected_width = length
            batch_indices.append(index)
            padded_width = projected_width
        if batch_indices:
            planned.append(
                IndexedBatch(
                    indices=tuple(batch_indices),
                    padded_width=padded_width,
                    token_count=len(batch_indices) * padded_width,
                )
            )

    flattened = tuple(index for batch in planned for index in batch.indices)
    if len(flattened) != len(parsed_lengths) or set(flattened) != set(range(len(parsed_lengths))):
        raise RuntimeError("batch planner did not assign every source index exactly once")
    return tuple(planned)

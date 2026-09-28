"""Inspect a batch-first search profile without running an experiment.

The command turns requested cluster workload counts into bounded half-open
ranges and explicit replay length buckets. It allocates planning metadata only;
it does not create model tensors, evaluate an oracle, or launch training.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.acquisition import maximum_exhaustive_pool_size
from amp_challenge.generators.search import (
    BatchExecutionPlan,
    load_batch_execution_plan,
    plan_count_batches,
    plan_length_bucketed_batches,
)

_COUNT_STAGES = (
    ("rollouts", "rollouts", "rollout_batch_size"),
    ("proposals", "proposals", "proposal_batch_size"),
    ("surrogate_rows", "surrogate_rows", "surrogate_batch_size"),
    ("kg_candidates", "kg_candidates", "kg_candidate_chunk_size"),
    ("kg_fantasies", "kg_fantasies", "kg_fantasy_chunk_size"),
    ("oracle_evaluations", "oracle_evaluations", "oracle_batch_size"),
)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BatchPlanningRequest:
    """Requested work used to materialize one deterministic planning report."""

    rollouts: int = 0
    proposals: int = 0
    surrogate_rows: int = 0
    kg_candidates: int = 0
    kg_fantasies: int = 0
    oracle_evaluations: int = 0
    sequence_lengths: tuple[int, ...] = ()
    world_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "rollouts",
            "proposals",
            "surrogate_rows",
            "kg_candidates",
            "kg_fantasies",
            "oracle_evaluations",
        ):
            _nonnegative_integer(getattr(self, name), name=name)
        if not isinstance(self.sequence_lengths, tuple):
            raise ValueError("sequence_lengths must be a tuple")
        for index, length in enumerate(self.sequence_lengths):
            _positive_integer(length, name=f"sequence_lengths[{index}]")
        _positive_integer(self.world_size, name="world_size")


def _count_batch_document(total_items: int, max_batch_size: int) -> dict[str, object]:
    batches = plan_count_batches(total_items, max_batch_size)
    return {
        "batch_count": len(batches),
        "batch_size_ceiling": max_batch_size,
        "batches": [
            {
                "batch_index": batch_index,
                "size": batch.size,
                "start": batch.start,
                "stop": batch.stop,
            }
            for batch_index, batch in enumerate(batches)
        ],
        "total_items": total_items,
    }


def build_batch_plan_document(
    execution_plan: BatchExecutionPlan,
    request: BatchPlanningRequest,
) -> dict[str, object]:
    """Build a JSON-ready, deterministic report for one requested workload."""

    if not isinstance(execution_plan, BatchExecutionPlan):
        raise TypeError("execution_plan must be BatchExecutionPlan")
    if not isinstance(request, BatchPlanningRequest):
        raise TypeError("request must be BatchPlanningRequest")

    count_batches = {
        output_name: _count_batch_document(
            getattr(request, request_name),
            getattr(execution_plan, batch_size_name),
        )
        for output_name, request_name, batch_size_name in _COUNT_STAGES
    }
    replay_batches = plan_length_bucketed_batches(
        request.sequence_lengths,
        execution_plan.replay_limits,
    )
    limits = execution_plan.replay_limits
    return {
        "artifact": "amp_search_batch_plan_v1",
        "configured_ceilings": {
            "gradient_accumulation_steps": execution_plan.gradient_accumulation_steps,
            "kg_candidate_chunk_size": execution_plan.kg_candidate_chunk_size,
            "kg_fantasy_chunk_size": execution_plan.kg_fantasy_chunk_size,
            "kg_max_combinations": execution_plan.kg_max_combinations,
            "kg_max_joint_size": execution_plan.kg_max_joint_size,
            "oracle_batch_size": execution_plan.oracle_batch_size,
            "proposal_batch_size": execution_plan.proposal_batch_size,
            "replay_length_bucket_boundaries": list(limits.length_bucket_boundaries),
            "replay_max_sequences": limits.max_sequences,
            "replay_max_tokens": limits.max_tokens,
            "rollout_batch_size": execution_plan.rollout_batch_size,
            "surrogate_batch_size": execution_plan.surrogate_batch_size,
        },
        "count_batches": count_batches,
        "effective_replay_batch": {
            "assumed_padded_width": limits.length_bucket_boundaries[-1],
            "padded_token_ceiling": execution_plan.effective_replay_token_batch(
                world_size=request.world_size
            ),
            "sequence_ceiling": execution_plan.effective_replay_sequence_batch(
                world_size=request.world_size
            ),
            "world_size": request.world_size,
        },
        "kg_exact_joint_search": {
            "combination_guard": execution_plan.kg_max_combinations,
            "joint_batch_size_ceiling": execution_plan.kg_max_joint_size,
            "requested_candidate_count": request.kg_candidates,
            "screened_pool_size_ceiling": maximum_exhaustive_pool_size(
                batch_size=execution_plan.kg_max_joint_size,
                max_combinations=execution_plan.kg_max_combinations,
                candidate_count=request.kg_candidates,
            ),
        },
        "profile": execution_plan.profile,
        "range_convention": "zero_based_half_open",
        "replay": {
            "batch_count": len(replay_batches),
            "batches": [
                {
                    "batch_index": batch_index,
                    "padded_token_count": batch.token_count,
                    "padded_width": batch.padded_width,
                    "sequence_count": len(batch.indices),
                    "source_indices": list(batch.indices),
                }
                for batch_index, batch in enumerate(replay_batches)
            ],
            "sequence_lengths": list(request.sequence_lengths),
            "total_sequences": len(request.sequence_lengths),
        },
        "schema_version": 1,
    }


def _nonnegative_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_argument(value: str) -> int:
    parsed = _nonnegative_argument(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sequence_lengths_argument(value: str) -> tuple[int, ...]:
    fields = value.split(",")
    if not fields or any(not field.strip() for field in fields):
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers")
    try:
        return tuple(_positive_argument(field.strip()) for field in fields)
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of positive integers"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    """Build the ``amp-search-batch-plan`` command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="standalone batching TOML")
    parser.add_argument("--rollouts", type=_nonnegative_argument, default=0)
    parser.add_argument("--proposals", type=_nonnegative_argument, default=0)
    parser.add_argument("--surrogate-rows", type=_nonnegative_argument, default=0)
    parser.add_argument("--kg-candidates", type=_nonnegative_argument, default=0)
    parser.add_argument("--kg-fantasies", type=_nonnegative_argument, default=0)
    parser.add_argument("--oracle-evaluations", type=_nonnegative_argument, default=0)
    parser.add_argument(
        "--sequence-lengths",
        action="append",
        type=_sequence_lengths_argument,
        default=[],
        help="comma-separated replay lengths; repeat to append more lengths",
    )
    parser.add_argument("--world-size", type=_positive_argument, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the requested plan as canonical JSON without starting work."""

    args = build_parser().parse_args(argv)
    sequence_lengths = tuple(
        length for length_group in args.sequence_lengths for length in length_group
    )
    try:
        execution_plan = load_batch_execution_plan(args.config)
        request = BatchPlanningRequest(
            rollouts=args.rollouts,
            proposals=args.proposals,
            surrogate_rows=args.surrogate_rows,
            kg_candidates=args.kg_candidates,
            kg_fantasies=args.kg_fantasies,
            oracle_evaluations=args.oracle_evaluations,
            sequence_lengths=sequence_lengths,
            world_size=args.world_size,
        )
        document = build_batch_plan_document(execution_plan, request)
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"amp-search-batch-plan error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

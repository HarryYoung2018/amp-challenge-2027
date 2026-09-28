"""Frozen orchestration identities for sequential mixed-acquisition v2.

This module contains no assay data, outcome access, fitting, or selection code.
It defines the deterministic rotation and policy-run graph that every producer
phase and the independent verifier must agree on before scientific work starts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

SCHEMA_VERSION = 1
FOLDS = tuple(range(5))
RANDOM_SEEDS = (17, 42, 91, 137, 271)
NO_QUERY = "no_query"
MEAN = "mean"
RANDOM = "random"
MEAN_NINE_DIVERSITY_ONE = "mean_nine_diversity_one"
MEAN_NINE_NOVELTY_ONE = "mean_nine_novelty_one"
MIXED = "mixed_eight_diversity_one_novelty_one"
CEILING = "full_acquisition_fold_ceiling"
POLICY_ORDER = (
    NO_QUERY,
    MEAN,
    RANDOM,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
    CEILING,
)
DETERMINISTIC_POLICIES = (
    NO_QUERY,
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
    CEILING,
)
BUDGETED_POLICIES = (
    MEAN,
    RANDOM,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
)
EXPECTED_SUPPORT_BY_FOLD = (202, 126, 112, 97, 113)
EXPECTED_CONTEXTS_BY_FOLD = (546, 486, 485, 487, 488)
EXPECTED_ROTATIONS = 20
EXPECTED_TRACKS_PER_ROTATION = 11
EXPECTED_POLICY_RUNS = 220
EXPECTED_POOL_CANDIDATES = 2_600
EXPECTED_PREDICTION_POOL_VIEW_ROWS = 2_600
EXPECTED_RANDOM_POOL_VIEW_ROWS = 2_600
EXPECTED_PHYSICAL_POOL_VIEW_ROWS = 5_200
EXPECTED_POOL_COMMITTED_ASSOCIATIONS = 4_400
EXPECTED_UPDATE_STATES = 220
EXPECTED_REFITS = 200
EXPECTED_OUTER_CONTEXT_PREDICTIONS = 109_648
EXPECTED_OUTER_CANDIDATES = 28_600
EXPECTED_OUTER_COMMITTED_ASSOCIATIONS = 2_200


def _fold(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in FOLDS:
        raise ValueError(f"{label} must be an integer in [0, 4]")
    return value


@dataclass(frozen=True, slots=True)
class RotationSpec:
    """One ordered outer/pool rotation and its exact three base folds."""

    outer_fold: int
    pool_fold: int

    def __post_init__(self) -> None:
        outer = _fold(self.outer_fold, label="outer fold")
        pool = _fold(self.pool_fold, label="pool fold")
        if outer == pool:
            raise ValueError("outer and pool folds must differ")

    @property
    def rotation_id(self) -> str:
        return f"outer-{self.outer_fold}.pool-{self.pool_fold}"

    @property
    def base_folds(self) -> tuple[int, int, int]:
        values = tuple(fold for fold in FOLDS if fold not in {self.outer_fold, self.pool_fold})
        if len(values) != 3:
            raise AssertionError("rotation did not produce exactly three base folds")
        return values

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation_id": self.rotation_id,
            "outer_fold": self.outer_fold,
            "acquisition_pool_fold": self.pool_fold,
            "base_folds": list(self.base_folds),
        }


@dataclass(frozen=True, slots=True)
class PolicyRunSpec:
    """One isolated policy-run identity inside a frozen rotation."""

    rotation: RotationSpec
    policy: str
    seed: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.rotation, RotationSpec):
            raise TypeError("policy run rotation must be a RotationSpec")
        if self.policy not in POLICY_ORDER:
            raise ValueError(f"unknown sequential-v2 policy: {self.policy!r}")
        if self.policy == RANDOM:
            if (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, int)
                or self.seed not in RANDOM_SEEDS
            ):
                raise ValueError("random policy seed must be one of the five frozen seeds")
        elif self.seed is not None:
            raise ValueError("only the random policy may carry a seed")

    @property
    def track_id(self) -> str:
        base = f"{self.rotation.rotation_id}.policy-{self.policy}"
        return f"{base}.seed-{self.seed}" if self.seed is not None else base

    @property
    def selection_kind(self) -> str:
        if self.policy == NO_QUERY:
            return "no_query"
        if self.policy == CEILING:
            return "full_acquisition_fold_ceiling"
        return "budgeted"

    @property
    def expected_pool_selection_count(self) -> int:
        if self.policy == NO_QUERY:
            return 0
        if self.policy == CEILING:
            return EXPECTED_SUPPORT_BY_FOLD[self.rotation.pool_fold]
        return 10

    @property
    def expected_outer_selection_count(self) -> int:
        return 10

    @property
    def refit(self) -> bool:
        return self.policy != NO_QUERY

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": self.track_id,
            "rotation_id": self.rotation.rotation_id,
            "policy": self.policy,
            "seed": self.seed,
            "selection_kind": self.selection_kind,
            "expected_pool_selection_count": self.expected_pool_selection_count,
            "expected_outer_selection_count": self.expected_outer_selection_count,
            "refit": self.refit,
        }


def ordered_rotations() -> tuple[RotationSpec, ...]:
    """Return the exact outer-major, pool-minor sequence of 20 rotations."""

    result = tuple(
        RotationSpec(outer_fold=outer, pool_fold=pool)
        for outer in FOLDS
        for pool in FOLDS
        if pool != outer
    )
    if len(result) != EXPECTED_ROTATIONS:
        raise AssertionError("frozen rotation census changed")
    return result


def policy_runs_for_rotation(rotation: RotationSpec) -> tuple[PolicyRunSpec, ...]:
    """Return the exact policy order, expanding random at the frozen seed position."""

    if not isinstance(rotation, RotationSpec):
        raise TypeError("rotation must be a RotationSpec")
    result: list[PolicyRunSpec] = []
    for policy in POLICY_ORDER:
        if policy == RANDOM:
            result.extend(PolicyRunSpec(rotation, policy, seed) for seed in RANDOM_SEEDS)
        else:
            result.append(PolicyRunSpec(rotation, policy, None))
    if len(result) != EXPECTED_TRACKS_PER_ROTATION:
        raise AssertionError("frozen per-rotation policy census changed")
    return tuple(result)


def ordered_policy_runs() -> tuple[PolicyRunSpec, ...]:
    """Return every one of the 220 frozen policy-run identities in canonical order."""

    result = tuple(
        policy_run
        for rotation in ordered_rotations()
        for policy_run in policy_runs_for_rotation(rotation)
    )
    if len(result) != EXPECTED_POLICY_RUNS or len({run.track_id for run in result}) != len(result):
        raise AssertionError("frozen policy-run identities are not exactly 220 unique tracks")
    return result


def rotation_by_id(rotation_id: str) -> RotationSpec:
    """Resolve a canonical rotation ID without accepting aliases."""

    if not isinstance(rotation_id, str):
        raise TypeError("rotation_id must be a string")
    matches = [rotation for rotation in ordered_rotations() if rotation.rotation_id == rotation_id]
    if len(matches) != 1:
        raise ValueError(f"unknown canonical rotation_id: {rotation_id!r}")
    return matches[0]


def policy_run_by_track_id(track_id: str) -> PolicyRunSpec:
    """Resolve a canonical track ID without accepting reordered or aliased fields."""

    if not isinstance(track_id, str):
        raise TypeError("track_id must be a string")
    matches = [run for run in ordered_policy_runs() if run.track_id == track_id]
    if len(matches) != 1:
        raise ValueError(f"unknown canonical track_id: {track_id!r}")
    return matches[0]


def validate_ordered_documents(
    documents: Iterable[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> tuple[Mapping[str, object], ...]:
    """Require exact document keys, values, and order for a frozen protocol table."""

    rows = tuple(documents)
    frozen = tuple(expected)
    if len(rows) != len(frozen):
        raise ValueError(f"{label} census must be exactly {len(frozen)}, got {len(rows)}")
    for index, (actual, wanted) in enumerate(zip(rows, frozen, strict=True)):
        if not isinstance(actual, Mapping):
            raise TypeError(f"{label} row {index} must be a mapping")
        if not _exact_json_equal(dict(actual), dict(wanted)):
            raise ValueError(f"{label} row {index} differs from the frozen protocol")
    return rows


def _exact_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int/float aliases."""

    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        left = actual
        right = expected
        assert isinstance(left, dict) and isinstance(right, dict)
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in right
        )
    if type(expected) is list:
        left = actual
        right = expected
        assert isinstance(left, list) and isinstance(right, list)
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return actual == expected


def validate_rotation_documents(
    documents: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return validate_ordered_documents(
        documents,
        tuple(rotation.document() for rotation in ordered_rotations()),
        label="rotation documents",
    )


def validate_policy_run_documents(
    documents: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return validate_ordered_documents(
        documents,
        tuple(run.document() for run in ordered_policy_runs()),
        label="policy-run documents",
    )


def protocol_census() -> dict[str, object]:
    """Publish all exact graph-wide counts used by launcher barrier checks."""

    rotations = ordered_rotations()
    runs = ordered_policy_runs()
    pool_associations = sum(run.expected_pool_selection_count for run in runs)
    outer_associations = sum(run.expected_outer_selection_count for run in runs)
    outer_candidates = sum(EXPECTED_SUPPORT_BY_FOLD[run.rotation.outer_fold] for run in runs)
    outer_context_predictions = sum(
        EXPECTED_CONTEXTS_BY_FOLD[run.rotation.outer_fold] for run in runs
    )
    census = {
        "schema_version": SCHEMA_VERSION,
        "rotations": len(rotations),
        "policy_runs": len(runs),
        "policy_runs_per_rotation": EXPECTED_TRACKS_PER_ROTATION,
        "random_policy_runs": sum(run.policy == RANDOM for run in runs),
        "deterministic_policy_runs": sum(run.policy != RANDOM for run in runs),
        "base_state_references": sum(not run.refit for run in runs),
        "refits": sum(run.refit for run in runs),
        "update_states": len(runs),
        "pool_candidates": sum(
            EXPECTED_SUPPORT_BY_FOLD[rotation.pool_fold] for rotation in rotations
        ),
        "prediction_bearing_pool_view_rows": EXPECTED_PREDICTION_POOL_VIEW_ROWS,
        "random_minimal_pool_view_rows": EXPECTED_RANDOM_POOL_VIEW_ROWS,
        "physical_pool_candidate_view_rows": EXPECTED_PHYSICAL_POOL_VIEW_ROWS,
        "pool_committed_sequence_associations": pool_associations,
        "outer_context_predictions": outer_context_predictions,
        "outer_candidates": outer_candidates,
        "outer_committed_sequence_associations": outer_associations,
    }
    expected = {
        "rotations": EXPECTED_ROTATIONS,
        "policy_runs": EXPECTED_POLICY_RUNS,
        "policy_runs_per_rotation": EXPECTED_TRACKS_PER_ROTATION,
        "random_policy_runs": 100,
        "deterministic_policy_runs": 120,
        "base_state_references": 20,
        "refits": EXPECTED_REFITS,
        "update_states": EXPECTED_UPDATE_STATES,
        "pool_candidates": EXPECTED_POOL_CANDIDATES,
        "prediction_bearing_pool_view_rows": EXPECTED_PREDICTION_POOL_VIEW_ROWS,
        "random_minimal_pool_view_rows": EXPECTED_RANDOM_POOL_VIEW_ROWS,
        "physical_pool_candidate_view_rows": EXPECTED_PHYSICAL_POOL_VIEW_ROWS,
        "pool_committed_sequence_associations": EXPECTED_POOL_COMMITTED_ASSOCIATIONS,
        "outer_context_predictions": EXPECTED_OUTER_CONTEXT_PREDICTIONS,
        "outer_candidates": EXPECTED_OUTER_CANDIDATES,
        "outer_committed_sequence_associations": EXPECTED_OUTER_COMMITTED_ASSOCIATIONS,
    }
    for key, value in expected.items():
        if census[key] != value:
            raise AssertionError(f"frozen protocol census changed for {key}")
    return census

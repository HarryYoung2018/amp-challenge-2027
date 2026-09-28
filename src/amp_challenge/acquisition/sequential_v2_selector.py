"""Outcome-free deterministic selectors for sequential mixed acquisition v2."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
MEAN = "mean"
MIXED = "mixed_eight_diversity_one_novelty_one"
MEAN_NINE_DIVERSITY_ONE = "mean_nine_diversity_one"
MEAN_NINE_NOVELTY_ONE = "mean_nine_novelty_one"
RANDOM = "random"
OUTER_MEAN = "outer_mean"
POOL_POLICIES = (MEAN, MIXED, MEAN_NINE_DIVERSITY_ONE, MEAN_NINE_NOVELTY_ONE)
_RANDOM_DOMAIN = b"amp_challenge.sequential_mixed_acquisition_v2.random\0"
_COMPONENT_PREFIX = "seqv2-div70:"
_GUARD_ABSOLUTE_TOLERANCE = 1e-12
_MIN_PROBABILITY = 1e-6
_MAX_PROBABILITY = 0.999999


def _exact_mapping(value: object, keys: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _identity(sequence_id: object, sequence: object, *, label: str) -> tuple[str, str]:
    if not isinstance(sequence, str) or canonicalize_sequence(sequence) != sequence:
        raise ValueError(f"{label} sequence must be canonical")
    if not isinstance(sequence_id, str) or sequence_id != canonical_sequence_id(sequence):
        raise ValueError(f"{label} sequence_id must be the canonical lowercase SHA-256")
    return sequence_id, sequence


def _component(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(_COMPONENT_PREFIX)
        or len(value) != len(_COMPONENT_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in value[len(_COMPONENT_PREFIX) :])
    ):
        raise ValueError(
            f"{label} diversity_component_id must be {_COMPONENT_PREFIX} plus lowercase SHA-256"
        )
    return value


def _eligible(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} eligible must be a boolean")
    return value


def _rotation(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} rotation_id must be canonical text")
    parts = value.split(".")
    if len(parts) != 2 or not parts[0].startswith("outer-") or not parts[1].startswith("pool-"):
        raise ValueError(f"{label} rotation_id must have form outer-o.pool-a")
    try:
        outer_fold = int(parts[0].removeprefix("outer-"))
        pool_fold = int(parts[1].removeprefix("pool-"))
    except ValueError as error:
        raise ValueError(f"{label} rotation_id must contain integer folds") from error
    expected = f"outer-{outer_fold}.pool-{pool_fold}"
    if (
        value != expected
        or outer_fold not in range(5)
        or pool_fold not in range(5)
        or outer_fold == pool_fold
    ):
        raise ValueError(f"{label} rotation_id must bind distinct folds in [0, 4]")
    return value


def _probabilities(value: object, *, label: str) -> tuple[float, float, float]:
    raw = _exact_mapping(value, set(OBJECTIVES), label=f"{label} objective_probabilities")
    values: list[float] = []
    for objective in OBJECTIVES:
        item = raw[objective]
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{label} {objective} must be numeric")
        number = float(item)
        if not math.isfinite(number) or not _MIN_PROBABILITY <= number <= _MAX_PROBABILITY:
            raise ValueError(
                f"{label} {objective} must be finite in [{_MIN_PROBABILITY}, {_MAX_PROBABILITY}]"
            )
        values.append(number)
    return values[0], values[1], values[2]


@dataclass(frozen=True, slots=True)
class PoolCandidate:
    """Strict outcome-free view for mean, mixed, and ablation policies."""

    rotation_id: str
    sequence_id: str
    sequence: str
    objective_probabilities: tuple[float, float, float]
    features: tuple[float, ...]
    novelty: float
    diversity_component_id: str
    eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_id", _rotation(self.rotation_id, label="pool candidate"))
        sequence_id, sequence = _identity(self.sequence_id, self.sequence, label="pool candidate")
        probabilities = tuple(self.objective_probabilities)
        if len(probabilities) != 3 or any(
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
            or not _MIN_PROBABILITY <= float(item) <= _MAX_PROBABILITY
            for item in probabilities
        ):
            raise ValueError(
                "pool candidate requires exactly three finite accepted-model probabilities"
            )
        if any(
            isinstance(item, bool) or not isinstance(item, int | float) for item in self.features
        ):
            raise ValueError("pool candidate features must be numeric")
        features = tuple(float(item) for item in self.features)
        if len(features) != 33 or any(not math.isfinite(item) for item in features):
            raise ValueError("pool candidate requires exactly 33 finite features")
        norm = math.sqrt(math.fsum(item * item for item in features))
        if norm != 0.0 and not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("pool candidate features must be unit-normalized or all zero")
        if (
            isinstance(self.novelty, bool)
            or not isinstance(self.novelty, int | float)
            or not math.isfinite(float(self.novelty))
            or not 0.0 <= float(self.novelty) <= 1.0
        ):
            raise ValueError("pool candidate novelty must be finite in [0, 1]")
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "objective_probabilities", tuple(map(float, probabilities)))
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "novelty", float(self.novelty))
        object.__setattr__(
            self,
            "diversity_component_id",
            _component(self.diversity_component_id, label="pool candidate"),
        )
        object.__setattr__(self, "eligible", _eligible(self.eligible, label="pool candidate"))

    @classmethod
    def from_mapping(cls, value: object) -> PoolCandidate:
        raw = _exact_mapping(
            value,
            {
                "rotation_id",
                "sequence_id",
                "sequence",
                "objective_probabilities",
                "features",
                "novelty",
                "diversity_component_id",
                "eligible",
            },
            label="pool candidate",
        )
        sequence_id, sequence = _identity(
            raw["sequence_id"], raw["sequence"], label="pool candidate"
        )
        feature_raw = raw["features"]
        if isinstance(feature_raw, str | bytes) or not isinstance(feature_raw, Sequence):
            raise ValueError("pool candidate features must be a non-empty sequence")
        if any(isinstance(item, bool) or not isinstance(item, int | float) for item in feature_raw):
            raise ValueError("pool candidate features must be numeric, not Boolean")
        features = tuple(float(item) for item in feature_raw)
        if len(features) != 33 or any(not math.isfinite(item) for item in features):
            raise ValueError("pool candidate requires exactly 33 finite features")
        norm = math.sqrt(math.fsum(item * item for item in features))
        if norm != 0.0 and not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("pool candidate features must be unit-normalized or all zero")
        novelty_raw = raw["novelty"]
        if isinstance(novelty_raw, bool) or not isinstance(novelty_raw, int | float):
            raise ValueError("pool candidate novelty must be numeric")
        novelty = float(novelty_raw)
        if not math.isfinite(novelty) or not 0.0 <= novelty <= 1.0:
            raise ValueError("pool candidate novelty must be finite in [0, 1]")
        return cls(
            rotation_id=_rotation(raw["rotation_id"], label="pool candidate"),
            sequence_id=sequence_id,
            sequence=sequence,
            objective_probabilities=_probabilities(raw["objective_probabilities"], label="pool"),
            features=features,
            novelty=novelty,
            diversity_component_id=_component(
                raw["diversity_component_id"], label="pool candidate"
            ),
            eligible=_eligible(raw["eligible"], label="pool candidate"),
        )

    @property
    def scalar_mean(self) -> float:
        return math.fsum(self.objective_probabilities) / 3.0


@dataclass(frozen=True, slots=True)
class RandomPoolCandidate:
    """Minimal random-control view; predictions and features cannot be supplied."""

    rotation_id: str
    sequence_id: str
    sequence: str
    diversity_component_id: str
    eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rotation_id", _rotation(self.rotation_id, label="random pool candidate")
        )
        sequence_id, sequence = _identity(
            self.sequence_id, self.sequence, label="random pool candidate"
        )
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(
            self,
            "diversity_component_id",
            _component(self.diversity_component_id, label="random pool candidate"),
        )
        object.__setattr__(self, "eligible", _eligible(self.eligible, label="random candidate"))

    @classmethod
    def from_mapping(cls, value: object) -> RandomPoolCandidate:
        raw = _exact_mapping(
            value,
            {
                "rotation_id",
                "sequence_id",
                "sequence",
                "diversity_component_id",
                "eligible",
            },
            label="random pool candidate",
        )
        sequence_id, sequence = _identity(
            raw["sequence_id"], raw["sequence"], label="random pool candidate"
        )
        return cls(
            _rotation(raw["rotation_id"], label="random pool candidate"),
            sequence_id,
            sequence,
            _component(raw["diversity_component_id"], label="random pool candidate"),
            _eligible(raw["eligible"], label="random pool candidate"),
        )


@dataclass(frozen=True, slots=True)
class OuterMeanCandidate:
    """Strict untouched-outer-fold view for the next-round mean decision."""

    rotation_id: str
    sequence_id: str
    sequence: str
    objective_probabilities: tuple[float, float, float]
    diversity_component_id: str
    eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rotation_id", _rotation(self.rotation_id, label="outer mean candidate")
        )
        sequence_id, sequence = _identity(
            self.sequence_id, self.sequence, label="outer mean candidate"
        )
        probabilities = tuple(self.objective_probabilities)
        if len(probabilities) != 3 or any(
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
            or not _MIN_PROBABILITY <= float(item) <= _MAX_PROBABILITY
            for item in probabilities
        ):
            raise ValueError(
                "outer mean requires exactly three finite accepted-model probabilities"
            )
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "objective_probabilities", tuple(map(float, probabilities)))
        object.__setattr__(
            self,
            "diversity_component_id",
            _component(self.diversity_component_id, label="outer mean candidate"),
        )
        object.__setattr__(self, "eligible", _eligible(self.eligible, label="outer candidate"))

    @classmethod
    def from_mapping(cls, value: object) -> OuterMeanCandidate:
        raw = _exact_mapping(
            value,
            {
                "rotation_id",
                "sequence_id",
                "sequence",
                "objective_probabilities",
                "diversity_component_id",
                "eligible",
            },
            label="outer mean candidate",
        )
        sequence_id, sequence = _identity(
            raw["sequence_id"], raw["sequence"], label="outer mean candidate"
        )
        return cls(
            _rotation(raw["rotation_id"], label="outer mean candidate"),
            sequence_id,
            sequence,
            _probabilities(raw["objective_probabilities"], label="outer mean"),
            _component(raw["diversity_component_id"], label="outer mean candidate"),
            _eligible(raw["eligible"], label="outer mean candidate"),
        )

    @property
    def scalar_mean(self) -> float:
        return math.fsum(self.objective_probabilities) / 3.0


@dataclass(frozen=True, slots=True)
class PoolPolicy:
    name: str
    exploit_seats: int
    diversity_seats: int
    novelty_seats: int
    batch_size: int = 10
    max_per_component: int = 2
    scalar_loss_max: float = 0.02
    objective_loss_max: float = 0.03

    def __post_init__(self) -> None:
        expected = {
            MEAN: (10, 0, 0),
            MIXED: (8, 1, 1),
            MEAN_NINE_DIVERSITY_ONE: (9, 1, 0),
            MEAN_NINE_NOVELTY_ONE: (9, 0, 1),
        }
        if (
            self.name not in expected
            or (
                self.exploit_seats,
                self.diversity_seats,
                self.novelty_seats,
            )
            != expected[self.name]
        ):
            raise ValueError("pool policy name and frozen seat allocation disagree")
        if self.batch_size != 10 or self.max_per_component != 2:
            raise ValueError("v2 policies require batch_size=10 and max_per_component=2")
        if self.scalar_loss_max != 0.02 or self.objective_loss_max != 0.03:
            raise ValueError("v2 reward-guard thresholds are frozen")


POLICY_BY_NAME = {
    MEAN: PoolPolicy(MEAN, 10, 0, 0),
    MIXED: PoolPolicy(MIXED, 8, 1, 1),
    MEAN_NINE_DIVERSITY_ONE: PoolPolicy(MEAN_NINE_DIVERSITY_ONE, 9, 1, 0),
    MEAN_NINE_NOVELTY_ONE: PoolPolicy(MEAN_NINE_NOVELTY_ONE, 9, 0, 1),
}


@dataclass(frozen=True, slots=True)
class SelectedSeat:
    sequence_id: str
    requested_role: str
    applied_role: str
    acquisition_score: float | int
    scalar_mean: float | None


@dataclass(frozen=True, slots=True)
class RepairStep:
    step_number: int
    requested_role: str
    removed_sequence_id: str
    added_sequence_id: str


@dataclass(frozen=True, slots=True)
class GuardEvaluation:
    stage: str
    sequence_ids: tuple[str, ...]
    objective_means: tuple[float, float, float]
    scalar_mean: float
    scalar_loss: float
    objective_losses: tuple[float, float, float]
    passed: bool


@dataclass(frozen=True, slots=True)
class SelectionResult:
    rotation_id: str
    policy: str
    seats: tuple[SelectedSeat, ...]
    mean_control_sequence_ids: tuple[str, ...]
    requested_sequence_ids: tuple[str, ...]
    requested_complete: bool
    requested_role_counts: Mapping[str, int]
    applied_role_counts: Mapping[str, int]
    component_counts: Mapping[str, int]
    scalar_loss: float | None
    objective_losses: tuple[float, float, float] | None
    repair_trace: tuple[RepairStep, ...]
    guard_evaluations: tuple[GuardEvaluation, ...]
    fallback_reason: str | None

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(seat.sequence_id for seat in self.seats)


def _validate_unique(
    candidates: Sequence[PoolCandidate | RandomPoolCandidate | OuterMeanCandidate],
) -> None:
    if not candidates:
        raise ValueError("candidate view cannot be empty")
    ids = [item.sequence_id for item in candidates]
    sequences = [item.sequence for item in candidates]
    if len(set(ids)) != len(ids) or len(set(sequences)) != len(sequences):
        raise ValueError("candidate sequence IDs and sequences must be unique")


def _midranks(values: Sequence[float]) -> dict[float, float]:
    ordered = sorted(values)
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    result: dict[float, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end] == ordered[start]:
            end += 1
        result[ordered[start]] = (start + end - 1) / (2.0 * (len(ordered) - 1))
        start = end
    return result


def _greedy_mean(
    candidates: Sequence[PoolCandidate | OuterMeanCandidate],
) -> list[PoolCandidate | OuterMeanCandidate]:
    chosen: list[PoolCandidate | OuterMeanCandidate] = []
    counts: Counter[str] = Counter()
    for candidate in sorted(candidates, key=lambda item: (-item.scalar_mean, item.sequence_id)):
        if not candidate.eligible or counts[candidate.diversity_component_id] >= 2:
            continue
        chosen.append(candidate)
        counts[candidate.diversity_component_id] += 1
        if len(chosen) == 10:
            return chosen
    raise ValueError("strict component cap leaves fewer than ten eligible candidates")


def _losses(
    control: Sequence[PoolCandidate], selected: Sequence[PoolCandidate]
) -> tuple[float, tuple[float, float, float]]:
    control_objectives = tuple(
        math.fsum(item.objective_probabilities[column] for item in control) / 10.0
        for column in range(3)
    )
    selected_objectives = tuple(
        math.fsum(item.objective_probabilities[column] for item in selected) / 10.0
        for column in range(3)
    )
    objective_losses = tuple(
        control_objectives[column] - selected_objectives[column] for column in range(3)
    )
    return math.fsum(objective_losses) / 3.0, objective_losses


def _guard_evaluation(
    stage: str,
    control: Sequence[PoolCandidate],
    selected: Sequence[PoolCandidate],
) -> GuardEvaluation:
    objective_means = tuple(
        math.fsum(item.objective_probabilities[column] for item in selected) / 10.0
        for column in range(3)
    )
    scalar_mean = math.fsum(objective_means) / 3.0
    scalar_loss, objective_losses = _losses(control, selected)
    return GuardEvaluation(
        stage=stage,
        sequence_ids=tuple(item.sequence_id for item in selected),
        objective_means=objective_means,
        scalar_mean=scalar_mean,
        scalar_loss=scalar_loss,
        objective_losses=objective_losses,
        passed=_within_guard(scalar_loss, objective_losses),
    )


def _within_guard(loss: float, objective_losses: Sequence[float]) -> bool:
    def within(value: float, threshold: float) -> bool:
        return value <= threshold or math.isclose(
            value, threshold, rel_tol=0.0, abs_tol=_GUARD_ABSOLUTE_TOLERANCE
        )

    return within(loss, 0.02) and all(within(item, 0.03) for item in objective_losses)


def select_pool_policy(
    candidates: Sequence[PoolCandidate],
    policy: str | PoolPolicy,
    *,
    outer_fold: int,
    pool_fold: int,
) -> SelectionResult:
    """Select one frozen pool policy without accepting any outcome-bearing object."""

    items = tuple(candidates)
    if any(not isinstance(item, PoolCandidate) for item in items):
        raise TypeError("pool selection accepts only PoolCandidate instances")
    _validate_unique(items)
    _validate_random_context(outer_fold=outer_fold, pool_fold=pool_fold, seed=17)
    expected_rotation = f"outer-{outer_fold}.pool-{pool_fold}"
    if any(item.rotation_id != expected_rotation for item in items):
        raise ValueError("pool candidate rotation_id disagrees with selector invocation")
    dimensions = {len(item.features) for item in items}
    if len(dimensions) != 1:
        raise ValueError("pool feature dimensions must be identical")
    if isinstance(policy, str) and policy not in POLICY_BY_NAME:
        raise ValueError(f"unsupported v2 pool policy: {policy!r}")
    frozen = POLICY_BY_NAME[policy] if isinstance(policy, str) else policy
    if not isinstance(frozen, PoolPolicy):
        raise TypeError("policy must be a frozen v2 policy name or PoolPolicy")
    eligible = [item for item in items if item.eligible]
    control = [item for item in _greedy_mean(eligible) if isinstance(item, PoolCandidate)]
    requested_roles = (
        ["exploit"] * frozen.exploit_seats
        + ["diversity"] * frozen.diversity_seats
        + ["novelty"] * frozen.novelty_seats
    )
    if frozen.name == MEAN:
        seats = tuple(
            SelectedSeat(item.sequence_id, "exploit", "exploit", item.scalar_mean, item.scalar_mean)
            for item in control
        )
        return _result(frozen.name, seats, items, (), None, control, control)

    selected: list[PoolCandidate] = []
    counts: Counter[str] = Counter()
    for item in control:
        if len(selected) == frozen.exploit_seats:
            break
        selected.append(item)
        counts[item.diversity_component_id] += 1
    scalar_ranks = _midranks([item.scalar_mean for item in eligible])
    novelty_ranks = _midranks([item.novelty for item in eligible])
    floor = statistics.median(item.scalar_mean for item in eligible)
    scores: list[float] = [item.scalar_mean for item in selected]

    for role in requested_roles[len(selected) :]:
        feasible = [
            item
            for item in eligible
            if item not in selected
            and counts[item.diversity_component_id] < 2
            and item.scalar_mean >= floor
        ]
        if not feasible:
            return _fallback(
                frozen,
                requested_roles,
                control,
                items,
                "hard_median_floor_infeasible",
                requested_sequence_ids=tuple(item.sequence_id for item in selected),
            )
        if role == "diversity":

            def score(item: PoolCandidate) -> float:
                distance = min(
                    max(
                        0.0,
                        min(
                            2.0,
                            1.0
                            - math.fsum(
                                a * b for a, b in zip(item.features, prior.features, strict=True)
                            ),
                        ),
                    )
                    for prior in selected
                )
                return 0.5 * (distance / 2.0) + 0.5 * scalar_ranks[item.scalar_mean]
        else:

            def score(item: PoolCandidate) -> float:
                return 0.8 * novelty_ranks[item.novelty] + 0.2 * scalar_ranks[item.scalar_mean]

        chosen = min(feasible, key=lambda item: (-score(item), -item.scalar_mean, item.sequence_id))
        chosen_score = score(chosen)
        selected.append(chosen)
        counts[chosen.diversity_component_id] += 1
        scores.append(chosen_score)

    trace: list[RepairStep] = []
    scalar_loss, objective_losses = _losses(control, selected)
    requested_sequence_ids = tuple(item.sequence_id for item in selected)
    evaluations = [_guard_evaluation("requested", control, selected)]
    removed_ids: set[str] = set()
    for role in ("novelty", "diversity"):
        if _within_guard(scalar_loss, objective_losses):
            break
        positions = [index for index, requested in enumerate(requested_roles) if requested == role]
        if not positions:
            continue
        position = positions[-1]
        removed = selected[position]
        removed_ids.add(removed.sequence_id)
        retained = selected[:position] + selected[position + 1 :]
        retained_counts = Counter(item.diversity_component_id for item in retained)
        replacements = sorted(
            (
                item
                for item in eligible
                if item.sequence_id not in removed_ids
                and item not in retained
                and retained_counts[item.diversity_component_id] < 2
            ),
            key=lambda item: (-item.scalar_mean, item.sequence_id),
        )
        if not replacements:
            return _fallback(
                frozen,
                requested_roles,
                control,
                items,
                "repair_candidate_infeasible",
                trace=tuple(trace),
                requested_sequence_ids=requested_sequence_ids,
                evaluations=tuple(evaluations),
            )
        replacement = replacements[0]
        selected[position] = replacement
        scores[position] = replacement.scalar_mean
        scalar_loss, objective_losses = _losses(control, selected)
        evaluation = _guard_evaluation(f"after_{role}_repair", control, selected)
        evaluations.append(evaluation)
        trace.append(
            RepairStep(
                step_number=len(trace) + 1,
                requested_role=role,
                removed_sequence_id=removed.sequence_id,
                added_sequence_id=replacement.sequence_id,
            )
        )
    if not _within_guard(scalar_loss, objective_losses):
        return _fallback(
            frozen,
            requested_roles,
            control,
            items,
            "reward_guard_failed",
            trace=tuple(trace),
            requested_sequence_ids=requested_sequence_ids,
            evaluations=tuple(evaluations),
        )
    repaired_roles = {step.requested_role for step in trace}
    seats = tuple(
        SelectedSeat(
            item.sequence_id,
            requested_roles[index],
            "exploit" if requested_roles[index] in repaired_roles else requested_roles[index],
            scores[index],
            item.scalar_mean,
        )
        for index, item in enumerate(selected)
    )
    return _result(
        frozen.name,
        seats,
        items,
        tuple(trace),
        None,
        control,
        selected,
        requested_sequence_ids=requested_sequence_ids,
        evaluations=tuple(evaluations),
    )


def _fallback(
    policy: PoolPolicy,
    requested_roles: Sequence[str],
    control: Sequence[PoolCandidate],
    universe: Sequence[PoolCandidate],
    reason: str,
    *,
    trace: tuple[RepairStep, ...] = (),
    requested_sequence_ids: tuple[str, ...] | None = None,
    evaluations: tuple[GuardEvaluation, ...] = (),
) -> SelectionResult:
    seats = tuple(
        SelectedSeat(
            item.sequence_id, requested_roles[index], "exploit", item.scalar_mean, item.scalar_mean
        )
        for index, item in enumerate(control)
    )
    final_evaluation = _guard_evaluation("fallback", control, control)
    return _result(
        policy.name,
        seats,
        universe,
        trace,
        reason,
        control,
        control,
        requested_sequence_ids=requested_sequence_ids,
        evaluations=(*evaluations, final_evaluation),
    )


def _result(
    policy: str,
    seats: tuple[SelectedSeat, ...],
    universe: Sequence[PoolCandidate | OuterMeanCandidate],
    trace: tuple[RepairStep, ...],
    fallback: str | None,
    control: Sequence[PoolCandidate] | None,
    selected: Sequence[PoolCandidate] | None,
    *,
    requested_sequence_ids: tuple[str, ...] | None = None,
    evaluations: tuple[GuardEvaluation, ...] = (),
) -> SelectionResult:
    by_id = {item.sequence_id: item for item in universe}
    losses = (None, None) if control is None or selected is None else _losses(control, selected)
    return SelectionResult(
        rotation_id=universe[0].rotation_id,
        policy=policy,
        seats=seats,
        mean_control_sequence_ids=(
            tuple(seat.sequence_id for seat in seats)
            if control is None
            else tuple(item.sequence_id for item in control)
        ),
        requested_sequence_ids=(
            tuple(seat.sequence_id for seat in seats)
            if requested_sequence_ids is None
            else requested_sequence_ids
        ),
        requested_complete=requested_sequence_ids is None or len(requested_sequence_ids) == 10,
        requested_role_counts=dict(Counter(seat.requested_role for seat in seats)),
        applied_role_counts=dict(Counter(seat.applied_role for seat in seats)),
        component_counts=dict(
            Counter(by_id[seat.sequence_id].diversity_component_id for seat in seats)
        ),
        scalar_loss=losses[0],
        objective_losses=losses[1],
        repair_trace=trace,
        guard_evaluations=evaluations,
        fallback_reason=fallback,
    )


def _validate_random_context(*, outer_fold: int, pool_fold: int, seed: int) -> None:
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (outer_fold, pool_fold, seed)
        )
        or outer_fold not in range(5)
        or pool_fold not in range(5)
        or outer_fold == pool_fold
        or seed not in {17, 42, 91, 137, 271}
    ):
        raise ValueError("random priority requires distinct folds 0..4 and a frozen v2 seed")


def random_priority(*, sequence_id: str, outer_fold: int, pool_fold: int, seed: int) -> int:
    _validate_random_context(outer_fold=outer_fold, pool_fold=pool_fold, seed=seed)
    if (
        not isinstance(sequence_id, str)
        or len(sequence_id) != 64
        or any(character not in "0123456789abcdef" for character in sequence_id)
    ):
        raise ValueError("sequence_id must be a lowercase SHA-256")
    payload = _RANDOM_DOMAIN + f"outer-{outer_fold}.pool-{pool_fold}\0{seed}\0{sequence_id}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def select_random(
    candidates: Sequence[RandomPoolCandidate], *, outer_fold: int, pool_fold: int, seed: int
) -> SelectionResult:
    items = tuple(candidates)
    if any(not isinstance(item, RandomPoolCandidate) for item in items):
        raise TypeError("random selection accepts only RandomPoolCandidate instances")
    _validate_unique(items)
    _validate_random_context(outer_fold=outer_fold, pool_fold=pool_fold, seed=seed)
    expected_rotation = f"outer-{outer_fold}.pool-{pool_fold}"
    if any(item.rotation_id != expected_rotation for item in items):
        raise ValueError("random candidate rotation_id disagrees with selector invocation")
    counts: Counter[str] = Counter()
    chosen: list[RandomPoolCandidate] = []
    priorities: list[int] = []
    for item in sorted(
        (candidate for candidate in items if candidate.eligible),
        key=lambda candidate: (
            random_priority(
                sequence_id=candidate.sequence_id,
                outer_fold=outer_fold,
                pool_fold=pool_fold,
                seed=seed,
            ),
            candidate.sequence_id,
        ),
    ):
        if counts[item.diversity_component_id] >= 2:
            continue
        chosen.append(item)
        counts[item.diversity_component_id] += 1
        priorities.append(
            random_priority(
                sequence_id=item.sequence_id,
                outer_fold=outer_fold,
                pool_fold=pool_fold,
                seed=seed,
            )
        )
        if len(chosen) == 10:
            break
    if len(chosen) != 10:
        raise ValueError("strict component cap leaves fewer than ten random candidates")
    seats = tuple(
        SelectedSeat(item.sequence_id, "random", "random", priority, None)
        for item, priority in zip(chosen, priorities, strict=True)
    )
    return SelectionResult(
        rotation_id=expected_rotation,
        policy=RANDOM,
        seats=seats,
        mean_control_sequence_ids=(),
        requested_sequence_ids=tuple(seat.sequence_id for seat in seats),
        requested_complete=True,
        requested_role_counts={"random": 10},
        applied_role_counts={"random": 10},
        component_counts=dict(counts),
        scalar_loss=None,
        objective_losses=None,
        repair_trace=(),
        guard_evaluations=(),
        fallback_reason=None,
    )


def select_outer_mean(
    candidates: Sequence[OuterMeanCandidate], *, outer_fold: int, pool_fold: int
) -> SelectionResult:
    items = tuple(candidates)
    if any(not isinstance(item, OuterMeanCandidate) for item in items):
        raise TypeError("outer mean selection accepts only OuterMeanCandidate instances")
    _validate_unique(items)
    _validate_random_context(outer_fold=outer_fold, pool_fold=pool_fold, seed=17)
    expected_rotation = f"outer-{outer_fold}.pool-{pool_fold}"
    if any(item.rotation_id != expected_rotation for item in items):
        raise ValueError("outer candidate rotation_id disagrees with selector invocation")
    selected = _greedy_mean(items)
    seats = tuple(
        SelectedSeat(item.sequence_id, "exploit", "exploit", item.scalar_mean, item.scalar_mean)
        for item in selected
    )
    return _result(OUTER_MEAN, seats, items, (), None, None, None)


__all__ = [
    "MEAN",
    "MEAN_NINE_DIVERSITY_ONE",
    "MEAN_NINE_NOVELTY_ONE",
    "MIXED",
    "OBJECTIVES",
    "OUTER_MEAN",
    "POLICY_BY_NAME",
    "POOL_POLICIES",
    "RANDOM",
    "GuardEvaluation",
    "OuterMeanCandidate",
    "PoolCandidate",
    "PoolPolicy",
    "RandomPoolCandidate",
    "RepairStep",
    "SelectedSeat",
    "SelectionResult",
    "random_priority",
    "select_outer_mean",
    "select_pool_policy",
    "select_random",
]

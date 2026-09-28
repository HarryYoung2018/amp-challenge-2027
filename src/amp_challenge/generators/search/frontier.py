"""Pure branch-allocation and progressive-widening calculations."""

from __future__ import annotations

import math
from dataclasses import replace

from amp_challenge.generators.search.records import BranchRecord


def _finite(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _credit_bounds(floor: float, ceiling: float) -> tuple[float, float]:
    floor = _finite(floor, field="floor")
    ceiling = _finite(ceiling, field="ceiling")
    if not 0.0 < floor <= 1.0 <= ceiling:
        raise ValueError("credit bounds must satisfy 0 < floor <= 1 <= ceiling")
    return floor, ceiling


def bounded_credit(credit: float, *, floor: float, ceiling: float) -> float:
    """Clamp a historical credit multiplier to prespecified safe bounds."""

    credit = _finite(credit, field="credit")
    floor, ceiling = _credit_bounds(floor, ceiling)
    return min(ceiling, max(floor, credit))


def bounded_decayed_credit(
    credit: float,
    *,
    decay: float,
    floor: float,
    ceiling: float,
    steps: int = 1,
) -> float:
    """Geometrically decay bounded credit toward the neutral multiplier one."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    decay = _finite(decay, field="decay")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must lie in [0, 1]")
    floor, ceiling = _credit_bounds(floor, ceiling)
    credit = bounded_credit(credit, floor=floor, ceiling=ceiling)
    decayed = 1.0 + (credit - 1.0) * decay**steps
    return bounded_credit(decayed, floor=floor, ceiling=ceiling)


def decay_branch_credit(branch: BranchRecord, *, steps: int = 1) -> BranchRecord:
    """Return a new immutable branch snapshot with decayed credit."""

    if not isinstance(branch, BranchRecord):
        raise TypeError("branch must be a BranchRecord")
    return replace(
        branch,
        credit=bounded_decayed_credit(
            branch.credit,
            decay=branch.credit_decay,
            floor=branch.credit_floor,
            ceiling=branch.credit_ceiling,
            steps=steps,
        ),
    )


def branch_score(branch: BranchRecord, *, exploration_scale: float, credit_mix: float) -> float:
    """Compute the proposal's credit-weighted UCB-style allocation score."""

    if not isinstance(branch, BranchRecord):
        raise TypeError("branch must be a BranchRecord")
    exploration_scale = _finite(exploration_scale, field="exploration_scale")
    if exploration_scale < 0.0:
        raise ValueError("exploration_scale must be non-negative")
    credit_mix = _finite(credit_mix, field="credit_mix")
    if not 0.0 <= credit_mix <= 1.0:
        raise ValueError("credit_mix must lie in [0, 1]")
    credit_multiplier = (1.0 - credit_mix) + credit_mix * branch.credit
    # Hypervolume improvement has a zero outside option.  Rectifying its UCB
    # also prevents a smaller positive credit multiplier from making a
    # negative bound look spuriously better.
    optimistic_yield = max(
        branch.yield_mean + exploration_scale * branch.yield_std,
        0.0,
    )
    return credit_multiplier * optimistic_yield


def progressive_widening_limit(*, visits: int, coefficient: float, exponent: float) -> int:
    """Return floor(c_pw * visits**alpha), the admissible child count."""

    if not isinstance(visits, int) or isinstance(visits, bool) or visits < 0:
        raise ValueError("visits must be a non-negative integer")
    coefficient = _finite(coefficient, field="coefficient")
    if coefficient <= 0.0:
        raise ValueError("coefficient must be positive")
    exponent = _finite(exponent, field="exponent")
    if not 0.0 < exponent < 1.0:
        raise ValueError("progressive widening requires 0 < exponent < 1")
    return math.floor(coefficient * visits**exponent)


def can_progressively_widen(
    *,
    visits: int,
    child_count: int,
    coefficient: float,
    exponent: float,
) -> bool:
    """Whether one more child fits under the progressive-widening budget."""

    if not isinstance(child_count, int) or isinstance(child_count, bool) or child_count < 0:
        raise ValueError("child_count must be a non-negative integer")
    return child_count < progressive_widening_limit(
        visits=visits, coefficient=coefficient, exponent=exponent
    )

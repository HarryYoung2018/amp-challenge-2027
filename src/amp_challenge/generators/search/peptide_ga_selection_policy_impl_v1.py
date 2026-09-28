"""Frozen executable selection policies hashed as one canonical source artifact."""

from __future__ import annotations


def bootstrap_selection_membership(
    *,
    proposal_id: str,
    eligible_proposal_ids: tuple[str, ...],
    selected: bool,
) -> bool:
    """Accept exactly a selected bootstrap proposal named by its eligibility set."""

    if type(proposal_id) is not str:
        raise ValueError("bootstrap proposal ID type differs")
    if type(eligible_proposal_ids) is not tuple or not eligible_proposal_ids:
        raise ValueError("bootstrap eligible inventory differs")
    if any(type(value) is not str for value in eligible_proposal_ids):
        raise ValueError("bootstrap eligible proposal ID type differs")
    if len(set(eligible_proposal_ids)) != len(eligible_proposal_ids):
        raise ValueError("bootstrap eligible proposal IDs are not unique")
    if type(selected) is not bool:
        raise ValueError("bootstrap selection flag type differs")
    return selected and proposal_id in eligible_proposal_ids


def controller_first_available_prefix_positions(
    availability: tuple[bool, ...],
    *,
    seat_count: int,
) -> tuple[int, ...]:
    """Take the first fixed-prefix positions not vetoed by private collisions."""

    if type(availability) is not tuple or len(availability) > 256:
        raise ValueError("controller availability inventory differs")
    if any(type(value) is not bool for value in availability):
        raise ValueError("controller availability flag type differs")
    if type(seat_count) is not int or not 0 <= seat_count <= 14:
        raise ValueError("controller seat count differs")
    positions = tuple(index for index, available in enumerate(availability) if available)
    if len(positions) < seat_count:
        raise ValueError("insufficient available controller prefix positions")
    return positions[:seat_count]


__all__ = [
    "bootstrap_selection_membership",
    "controller_first_available_prefix_positions",
]

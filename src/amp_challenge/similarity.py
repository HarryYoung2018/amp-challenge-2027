"""Lightweight sequence identity and deterministic single-link clustering.

The implementation uses a simple global alignment and quadratic all-pairs
clustering.  It is suitable for tests, audits, and modest peptide collections,
but not for clustering millions of generated candidates.  Production-scale
work should use a pinned external implementation (for example MMseqs2) and audit
that its identity/coverage convention matches the competition definition.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from .sequences import canonicalize_sequence


def _alignment_summary(left: str, right: str) -> tuple[int, int]:
    """Return ``(matches, alignment_length)`` for one global alignment.

    Alignments maximize a score of +1/-1/-1 for match/mismatch/gap.  Ties prefer
    more matches, then a shorter alignment, with diagonal/up/left as the final
    deterministic ordering.  This is intentionally dependency-free and is not
    claimed to reproduce any organizer-side aligner.
    """

    # State is (score, matches, negative_alignment_length).
    previous: list[tuple[int, int, int]] = [(0, 0, 0)]
    for column in range(1, len(right) + 1):
        previous.append((-column, 0, -column))

    for row, left_residue in enumerate(left, start=1):
        current: list[tuple[int, int, int]] = [(-row, 0, -row)]
        for column, right_residue in enumerate(right, start=1):
            diagonal = previous[column - 1]
            is_match = int(left_residue == right_residue)
            diagonal_state = (
                diagonal[0] + (1 if is_match else -1),
                diagonal[1] + is_match,
                diagonal[2] - 1,
            )
            up = previous[column]
            up_state = (up[0] - 1, up[1], up[2] - 1)
            left_state = current[column - 1]
            left_candidate = (
                left_state[0] - 1,
                left_state[1],
                left_state[2] - 1,
            )
            # max() retains the first candidate for a complete tie.
            current.append(max(diagonal_state, up_state, left_candidate))
        previous = current

    _, matches, negative_length = previous[-1]
    return matches, -negative_length


@lru_cache(maxsize=1_000_000)
def _cached_canonical_identity(left: str, right: str) -> float:
    matches, alignment_length = _alignment_summary(left, right)
    return matches / alignment_length


def global_sequence_identity(left: str, right: str) -> float:
    """Compute matches divided by global-alignment length in ``[0, 1]``.

    Length limits are relaxed here so the helper can be used in unit tests and
    pre-filter audits; both strings must still be nonempty and use standard amino
    acids. Canonically ordered pairs are cached because grouped evaluation asks
    for the same peptide pair across many assay contexts and model families.
    """

    left_canonical = canonicalize_sequence(left, min_length=1, max_length=10**9)
    right_canonical = canonicalize_sequence(right, min_length=1, max_length=10**9)
    if right_canonical < left_canonical:
        left_canonical, right_canonical = right_canonical, left_canonical
    return _cached_canonical_identity(left_canonical, right_canonical)


def pairwise_identity_matrix(sequences: Iterable[str]) -> np.ndarray:
    """Return a symmetric all-pairs identity matrix in input order."""

    canonical = [
        canonicalize_sequence(sequence, min_length=1, max_length=10**9) for sequence in sequences
    ]
    matrix = np.eye(len(canonical), dtype=np.float64)
    for left_index, left in enumerate(canonical):
        for right_index in range(left_index + 1, len(canonical)):
            identity = global_sequence_identity(left, canonical[right_index])
            matrix[left_index, right_index] = identity
            matrix[right_index, left_index] = identity
    return matrix


def cluster_sequences(
    sequences: Iterable[str],
    *,
    identity_threshold: float = 0.8,
) -> tuple[tuple[str, ...], ...]:
    """Single-link cluster unique canonical sequences by global identity.

    Components are deterministic and input-order independent.  Single-link
    transitivity means two members of one component need not directly meet the
    threshold; this behavior should be recorded in experiment manifests.
    """

    if not 0.0 <= identity_threshold <= 1.0:
        raise ValueError("identity_threshold must be between 0 and 1")
    canonical = sorted(
        {canonicalize_sequence(sequence, min_length=1, max_length=10**9) for sequence in sequences}
    )
    parent = list(range(len(canonical)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root == right_root:
            return
        # Always attach the larger root index to the smaller one.
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for left_index, left in enumerate(canonical):
        for right_index in range(left_index + 1, len(canonical)):
            right = canonical[right_index]
            # No global alignment can match more than the shorter sequence.
            length_upper_bound = min(len(left), len(right)) / max(len(left), len(right))
            if length_upper_bound < identity_threshold:
                continue
            if global_sequence_identity(left, right) >= identity_threshold:
                union(left_index, right_index)

    components: dict[int, list[str]] = {}
    for index, sequence in enumerate(canonical):
        components.setdefault(find(index), []).append(sequence)
    result = [tuple(sorted(component)) for component in components.values()]
    return tuple(sorted(result, key=lambda component: component[0]))

"""Deterministic homology-clustered dataset splits.

This dependency-light implementation performs quadratic all-pairs global
alignments and single-link clustering.  It is intended for reproducible audits
and modest AMP corpora.  Large datasets should use a pinned scalable clusterer,
then feed its immutable cluster map into the same split-assignment logic.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..sequences import canonical_sequence_id, canonicalize_sequence
from ..similarity import cluster_sequences
from .records import PeptideRecord

DEFAULT_SPLIT_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("train", 0.8),
    ("validation", 0.1),
    ("test", 0.1),
)


@dataclass(frozen=True, slots=True)
class HomologyCluster:
    cluster_id: str
    sequence_ids: tuple[str, ...]
    sequences: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.sequence_ids)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    sequence_id: str
    cluster_id: str
    split: str


@dataclass(frozen=True, slots=True)
class HomologySplit:
    """Immutable cluster definitions and per-sequence assignments."""

    identity_threshold: float
    seed: int
    fractions: tuple[tuple[str, float], ...]
    clusters: tuple[HomologyCluster, ...]
    assignments: tuple[SplitAssignment, ...]

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.fractions)

    def split_for(self, sequence_or_id: str) -> str:
        """Look up an assignment from a canonical sequence or SHA-256 ID."""

        candidate_id = sequence_or_id
        if len(candidate_id) != 64 or any(
            character not in "0123456789abcdef" for character in candidate_id
        ):
            candidate_id = canonical_sequence_id(sequence_or_id)
        for assignment in self.assignments:
            if assignment.sequence_id == candidate_id:
                return assignment.split
        raise KeyError(sequence_or_id)

    def sequence_ids(self, split: str) -> tuple[str, ...]:
        if split not in self.split_names:
            raise KeyError(split)
        return tuple(
            assignment.sequence_id for assignment in self.assignments if assignment.split == split
        )

    def records_by_split(
        self,
        records: Iterable[PeptideRecord],
    ) -> dict[str, tuple[PeptideRecord, ...]]:
        """Return deterministically ordered deduplicated records for each split."""

        records_by_id = {record.sequence_id: record for record in records}
        output: dict[str, tuple[PeptideRecord, ...]] = {}
        for split_name in self.split_names:
            output[split_name] = tuple(
                records_by_id[sequence_id]
                for sequence_id in self.sequence_ids(split_name)
                if sequence_id in records_by_id
            )
        return output


def _normalized_fractions(
    split_fractions: Mapping[str, float] | Iterable[tuple[str, float]] | None,
) -> tuple[tuple[str, float], ...]:
    if split_fractions is None:
        items = list(DEFAULT_SPLIT_FRACTIONS)
    elif isinstance(split_fractions, Mapping):
        items = list(split_fractions.items())
    else:
        items = list(split_fractions)
    if not items:
        raise ValueError("at least one split is required")
    names = [str(name) for name, _ in items]
    if any(not name.strip() for name in names):
        raise ValueError("split names cannot be empty")
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    values = [float(fraction) for _, fraction in items]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("split fractions must be finite and non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("split fractions must have a positive sum")
    return tuple((name, value / total) for name, value in zip(names, values, strict=True))


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _make_clusters(
    sequences: Iterable[str],
    *,
    identity_threshold: float,
) -> tuple[HomologyCluster, ...]:
    components = cluster_sequences(
        sequences,
        identity_threshold=identity_threshold,
    )
    clusters: list[HomologyCluster] = []
    for component in components:
        pairs = sorted((canonical_sequence_id(sequence), sequence) for sequence in component)
        sequence_ids = tuple(sequence_id for sequence_id, _ in pairs)
        ordered_sequences = tuple(sequence for _, sequence in pairs)
        cluster_id = hashlib.sha256("\n".join(sequence_ids).encode("ascii")).hexdigest()
        clusters.append(
            HomologyCluster(
                cluster_id=cluster_id,
                sequence_ids=sequence_ids,
                sequences=ordered_sequences,
            )
        )
    return tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))


def make_homology_split(
    records_or_sequences: Iterable[PeptideRecord | str],
    *,
    identity_threshold: float = 0.8,
    split_fractions: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    seed: int = 0,
) -> HomologySplit:
    """Cluster unique sequences and greedily balance whole clusters across splits.

    The result is independent of input ordering.  Largest clusters are assigned
    first; a SHA-256 digest of the seed resolves otherwise equivalent choices.
    Exact target fractions may be impossible when clusters are large.
    """

    fractions = _normalized_fractions(split_fractions)
    raw_items = tuple(records_or_sequences)
    sequences = sorted(
        {
            canonicalize_sequence(item.sequence if isinstance(item, PeptideRecord) else item)
            for item in raw_items
        }
    )
    clusters = _make_clusters(
        sequences,
        identity_threshold=identity_threshold,
    )
    total_sequences = len(sequences)
    targets = {name: fraction * total_sequences for name, fraction in fractions}
    counts = {name: 0 for name, _ in fractions}
    assigned_cluster: dict[str, str] = {}

    ordered_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -cluster.size,
            _stable_digest(seed, "cluster-order", cluster.cluster_id),
        ),
    )
    for cluster in ordered_clusters:
        candidate_scores: list[tuple[float, str, str]] = []
        for split_name, _ in fractions:
            projected = dict(counts)
            projected[split_name] += cluster.size
            squared_error = sum(
                ((projected[name] - targets[name]) ** 2) / max(targets[name], 1.0)
                for name, _ in fractions
            )
            tie_break = _stable_digest(seed, cluster.cluster_id, split_name)
            candidate_scores.append((squared_error, tie_break, split_name))
        _, _, chosen_split = min(candidate_scores)
        counts[chosen_split] += cluster.size
        assigned_cluster[cluster.cluster_id] = chosen_split

    assignments = tuple(
        sorted(
            (
                SplitAssignment(
                    sequence_id=sequence_id,
                    cluster_id=cluster.cluster_id,
                    split=assigned_cluster[cluster.cluster_id],
                )
                for cluster in clusters
                for sequence_id in cluster.sequence_ids
            ),
            key=lambda assignment: assignment.sequence_id,
        )
    )
    return HomologySplit(
        identity_threshold=identity_threshold,
        seed=int(seed),
        fractions=fractions,
        clusters=clusters,
        assignments=assignments,
    )


# Brief alias used in workflow code.
homology_cluster_split = make_homology_split

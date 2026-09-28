"""Gate-1 strain-level AMP activity benchmark with grouped cross-validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.data.records import CensoredValue, CensorRelation
from amp_challenge.models.oracle_baselines import (
    DescriptorLogisticOracle,
    HomologyKnnOracle,
    OracleInput,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import cluster_sequences, global_sequence_identity

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GramClass = Literal["positive", "negative", "unknown"]


@dataclass(frozen=True, slots=True)
class NormalizedMicObservation:
    row_id: str
    input_line: int
    sequence_id: str
    sequence: str
    strain: str | None
    gram: GramClass
    value: CensoredValue


@dataclass(frozen=True, slots=True)
class ActivityExample:
    example_id: str
    sequence_id: str
    sequence: str
    strain: str
    gram: GramClass
    label: int
    source_row_ids: tuple[str, ...]

    @property
    def model_input(self) -> OracleInput:
        return OracleInput(sequence=self.sequence, strain=self.strain, gram=self.gram)


@dataclass(frozen=True, slots=True)
class LabelAuditRow:
    row_id: str
    input_line: int
    sequence_id: str
    sequence: str
    strain: str | None
    gram: GramClass
    relation: CensorRelation
    lower: float | None
    upper: float | None
    lower_inclusive: bool
    upper_inclusive: bool
    unit: str | None
    individual_label: int | None
    individual_reason: str
    group_status: str
    example_id: str | None


@dataclass(frozen=True, slots=True)
class ActivityDataset:
    examples: tuple[ActivityExample, ...]
    audit_rows: tuple[LabelAuditRow, ...]
    ignored_non_mic_rows: int


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    example_id: str
    sequence_id: str
    cluster_id: str
    fold: int


@dataclass(frozen=True, slots=True)
class _LabeledHomologyCluster:
    cluster_id: str
    sequences: tuple[str, ...]
    examples: tuple[ActivityExample, ...]


@dataclass(frozen=True, slots=True)
class OofPrediction:
    model: str
    example_id: str
    sequence_id: str
    sequence: str
    strain: str
    gram: GramClass
    label: int
    source_observations: int
    fold: int
    cluster_id: str
    max_train_identity: float
    probability: float


@dataclass(frozen=True, slots=True)
class LogisticSettings:
    l2: float
    max_iterations: int
    tolerance: float
    prior_strength: float


@dataclass(frozen=True, slots=True)
class KnnSettings:
    neighbors: int
    similarity_power: float
    prior_strength: float
    minimum_weight: float


@dataclass(frozen=True, slots=True)
class Gate1Config:
    schema_version: int
    activity_threshold_um: float
    homology_identity_threshold: float
    folds: int
    seed: int
    bootstrap_replicates: int
    calibration_bins: int
    similarity_bin_edges: tuple[float, ...]
    logistic: LogisticSettings
    knn: KnnSettings

    @classmethod
    def from_toml(cls, path: str | Path) -> Gate1Config:
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        logistic_raw = raw.get("descriptor_logistic", {})
        knn_raw = raw.get("homology_knn", {})
        config = cls(
            schema_version=int(raw.get("schema_version", 0)),
            activity_threshold_um=float(raw["activity_threshold_um"]),
            homology_identity_threshold=float(raw["homology_identity_threshold"]),
            folds=int(raw["folds"]),
            seed=int(raw["seed"]),
            bootstrap_replicates=int(raw["bootstrap_replicates"]),
            calibration_bins=int(raw["calibration_bins"]),
            similarity_bin_edges=tuple(float(value) for value in raw["similarity_bin_edges"]),
            logistic=LogisticSettings(
                l2=float(logistic_raw["l2"]),
                max_iterations=int(logistic_raw["max_iterations"]),
                tolerance=float(logistic_raw["tolerance"]),
                prior_strength=float(logistic_raw["prior_strength"]),
            ),
            knn=KnnSettings(
                neighbors=int(knn_raw["neighbors"]),
                similarity_power=float(knn_raw["similarity_power"]),
                prior_strength=float(knn_raw["prior_strength"]),
                minimum_weight=float(knn_raw["minimum_weight"]),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported benchmark schema_version: {self.schema_version}")
        if not math.isfinite(self.activity_threshold_um) or self.activity_threshold_um <= 0:
            raise ValueError("activity_threshold_um must be finite and positive")
        if not 0 < self.homology_identity_threshold <= 1:
            raise ValueError("homology_identity_threshold must be in (0, 1]")
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates cannot be negative")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least two")
        if len(self.similarity_bin_edges) < 2:
            raise ValueError("similarity_bin_edges needs at least two edges")
        if self.similarity_bin_edges[0] != 0.0:
            raise ValueError("similarity_bin_edges must start at zero")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.similarity_bin_edges
        ):
            raise ValueError("similarity_bin_edges must be finite values in [0, 1]")
        if any(
            right <= left
            for left, right in zip(
                self.similarity_bin_edges,
                self.similarity_bin_edges[1:],
                strict=False,
            )
        ):
            raise ValueError("similarity_bin_edges must be strictly increasing")
        if self.similarity_bin_edges[-1] < self.homology_identity_threshold:
            raise ValueError("last similarity edge must cover the homology threshold")


def _stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_bytes(path: str | Path) -> tuple[bytes, str]:
    """Read one regular file while rejecting an in-place or path-target change."""

    source = Path(path).resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"expected a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    before_fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_fingerprint = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_fingerprint != after_fingerprint or len(payload) != before.st_size:
        raise ValueError(f"input changed while it was being read: {source}")
    return payload, hashlib.sha256(payload).hexdigest()


def verify_normalized_dataset_summary(
    assays_path: str | Path,
    summary_path: str | Path,
    *,
    sequences_path: str | Path | None = None,
) -> dict[str, object]:
    """Verify the normalized assay checksum and approved-source declaration."""

    summary_file = Path(summary_path)
    with summary_file.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    schema_version = int(summary.get("schema_version", 0))
    if schema_version not in {1, 2}:
        raise ValueError("normalized summary schema_version must be 1 or 2")
    expected_hash = str(summary.get("assays_sha256", ""))
    actual_hash = _file_sha256(assays_path)
    if expected_hash != actual_hash:
        raise ValueError(
            "normalized assay checksum does not match summary: "
            f"expected {expected_hash!r}, got {actual_hash!r}"
        )
    if sequences_path is not None:
        expected_sequences_hash = str(summary.get("sequences_sha256", ""))
        actual_sequences_hash = _file_sha256(sequences_path)
        if expected_sequences_hash != actual_sequences_hash:
            raise ValueError(
                "normalized sequence checksum does not match summary: "
                f"expected {expected_sequences_hash!r}, got {actual_sequences_hash!r}"
            )
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("normalized summary must declare at least one source artifact")
    nonapproved: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            nonapproved.append("<malformed>")
        elif artifact.get("training_status") != "approved":
            nonapproved.append(str(artifact.get("name", "<unnamed>")))
    if nonapproved:
        raise ValueError(f"normalized dataset contains non-approved artifacts: {nonapproved}")
    return cast(dict[str, object], summary)


def read_normalized_sequence_union(path: str | Path) -> tuple[str, ...]:
    """Read the complete normalized modeling sequence universe."""

    sequences_by_id: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for input_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{input_line}: expected a JSON object")
            sequence = canonicalize_sequence(str(row.get("sequence", "")))
            sequence_id = str(row.get("sequence_id", ""))
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(f"{path}:{input_line}: sequence_id does not match sequence")
            if sequence_id in sequences_by_id:
                raise ValueError(f"{path}:{input_line}: duplicate sequence_id")
            sequences_by_id[sequence_id] = sequence
    if not sequences_by_id:
        raise ValueError(f"normalized sequence table is empty: {path}")
    return tuple(sequences_by_id[sequence_id] for sequence_id in sorted(sequences_by_id))


def read_normalized_endpoint_counts(
    path: str | Path,
    *,
    sequence_universe: Iterable[str] | None = None,
) -> dict[str, int]:
    """Count endpoints and optionally bind every assay to the full sequence union."""

    counts: dict[str, int] = defaultdict(int)
    sequences_by_id = (
        None
        if sequence_universe is None
        else {
            canonical_sequence_id(sequence): canonicalize_sequence(sequence)
            for sequence in sequence_universe
        }
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        for input_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{input_line}: expected a JSON object")
            endpoint = str(row.get("endpoint", "")).strip().lower()
            if not endpoint:
                raise ValueError(f"{path}:{input_line}: endpoint is empty")
            if sequences_by_id is not None:
                sequence = canonicalize_sequence(str(row.get("sequence", "")))
                sequence_id = str(row.get("sequence_id", ""))
                if sequence_id != canonical_sequence_id(sequence):
                    raise ValueError(f"{path}:{input_line}: sequence_id does not match sequence")
                if sequences_by_id.get(sequence_id) != sequence:
                    raise ValueError(
                        f"{path}:{input_line}: assay sequence is absent from the sequence table"
                    )
            counts[endpoint] += 1
    if not counts:
        raise ValueError(f"normalized assay table is empty: {path}")
    return dict(sorted(counts.items()))


def verify_sequence_union_declaration(
    summary: Mapping[str, object],
    sequence_universe: Sequence[str],
) -> None:
    """Require the normalized summary to declare the complete sequence-table size."""

    declared = summary.get("unique_sequences")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        raise ValueError("normalized summary must declare a positive unique_sequences count")
    if declared != len(sequence_universe):
        raise ValueError(
            "normalized unique_sequences count mismatch: "
            f"expected {len(sequence_universe)}, got {declared}"
        )


def verify_declared_endpoint_counts(
    summary: Mapping[str, object],
    observed_counts: Mapping[str, int],
) -> None:
    """Bind schema-v2 endpoint declarations to the assay rows used by Gate-1."""

    schema_version = int(summary.get("schema_version", 0))
    declared = summary.get("endpoint_counts")
    if declared is None:
        if schema_version == 2:
            raise ValueError("normalized schema v2 requires endpoint_counts")
        return
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("normalized endpoint_counts must be a non-empty object")

    declared_counts: dict[str, int] = {}
    for raw_endpoint, raw_count in declared.items():
        endpoint = str(raw_endpoint).strip().lower()
        if not endpoint or endpoint in declared_counts:
            raise ValueError("normalized endpoint_counts contains an invalid endpoint key")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(
                f"normalized endpoint_counts[{endpoint!r}] must be a non-negative integer"
            )
        declared_counts[endpoint] = raw_count

    normalized_observed: dict[str, int] = {}
    for raw_endpoint, raw_count in observed_counts.items():
        endpoint = str(raw_endpoint).strip().lower()
        if not endpoint or endpoint in normalized_observed:
            raise ValueError("observed endpoint counts contains an invalid endpoint key")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"observed endpoint count for {endpoint!r} is invalid")
        normalized_observed[endpoint] = raw_count
    if declared_counts != normalized_observed:
        raise ValueError(
            "normalized endpoint count mismatch: "
            f"expected {json.dumps(normalized_observed, sort_keys=True)}, "
            f"got {json.dumps(declared_counts, sort_keys=True)}"
        )
    assay_observations = summary.get("assay_observations")
    if (
        isinstance(assay_observations, bool)
        or not isinstance(assay_observations, int)
        or assay_observations != sum(normalized_observed.values())
    ):
        raise ValueError("normalized assay_observations does not match the assay table")


def read_normalized_mic_observations(
    path: str | Path,
) -> tuple[tuple[NormalizedMicObservation, ...], int]:
    """Read and validate normalized MIC JSONL without imputing intervals."""

    observations: list[NormalizedMicObservation] = []
    ignored_non_mic = 0
    row_id_occurrences: dict[str, int] = defaultdict(int)
    with Path(path).open("r", encoding="utf-8") as handle:
        for input_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("endpoint", "")).strip().lower() != "mic":
                ignored_non_mic += 1
                continue
            sequence = canonicalize_sequence(str(row["sequence"]))
            sequence_id = str(row["sequence_id"])
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(f"{path}:{input_line}: sequence_id does not match sequence")
            relation_raw = str(row["relation"])
            relation = cast(CensorRelation, relation_raw)
            value = CensoredValue(
                relation=relation,
                lower=None if row.get("lower") is None else float(row["lower"]),
                upper=None if row.get("upper") is None else float(row["upper"]),
                lower_inclusive=bool(row["lower_inclusive"]),
                upper_inclusive=bool(row["upper_inclusive"]),
                unit=None if row.get("unit") is None else str(row["unit"]),
                raw=None if row.get("raw_value") is None else str(row["raw_value"]),
                source_unit=(None if row.get("source_unit") is None else str(row["source_unit"])),
            )
            strain_raw = row.get("strain")
            strain = None if strain_raw is None or not str(strain_raw).strip() else str(strain_raw)
            gram_raw = str(row.get("gram", "unknown"))
            if gram_raw not in {"positive", "negative", "unknown"}:
                raise ValueError(f"{path}:{input_line}: invalid Gram class {gram_raw!r}")
            gram = cast(GramClass, gram_raw)
            provenance = row.get("provenance", {})
            base_row_id = _stable_digest(
                "normalized-mic-v1",
                sequence_id,
                strain,
                relation,
                value.lower,
                value.upper,
                json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            )
            occurrence = row_id_occurrences[base_row_id]
            row_id_occurrences[base_row_id] += 1
            # Some approved sources report the same numeric result more than once
            # for distinct assay conditions that their normalized schema cannot yet
            # represent. Preserve each source row for the label audit while keeping
            # IDs unchanged for datasets without such duplicates.
            row_id = (
                base_row_id
                if occurrence == 0
                else _stable_digest("normalized-mic-v1-duplicate", base_row_id, occurrence)
            )
            observations.append(
                NormalizedMicObservation(
                    row_id=row_id,
                    input_line=input_line,
                    sequence_id=sequence_id,
                    sequence=sequence,
                    strain=strain,
                    gram=gram,
                    value=value,
                )
            )
    return tuple(sorted(observations, key=lambda item: item.row_id)), ignored_non_mic


def derive_activity_label(
    value: CensoredValue,
    *,
    threshold_um: float,
) -> tuple[int | None, str]:
    """Return ``MIC <= threshold`` only when the interval proves the class."""

    if not math.isfinite(threshold_um) or threshold_um <= 0:
        raise ValueError("threshold_um must be finite and positive")
    if value.unit != "uM":
        return None, "unsupported_or_non_um_unit"
    if value.relation == "approx":
        return None, "approximate_value_has_undeclared_tolerance"
    if value.upper is not None and value.upper <= threshold_um:
        return 1, "interval_guarantees_mic_at_or_below_threshold"
    if value.lower is not None and (
        value.lower > threshold_um or (value.lower == threshold_um and not value.lower_inclusive)
    ):
        return 0, "interval_guarantees_mic_above_threshold"
    return None, "measurement_interval_crosses_or_touches_decision_boundary"


def build_activity_dataset(
    observations: Iterable[NormalizedMicObservation],
    *,
    threshold_um: float,
    ignored_non_mic_rows: int = 0,
) -> ActivityDataset:
    """Collapse replicates only when every observation proves one same class."""

    grouped: dict[tuple[str, str | None], list[NormalizedMicObservation]] = defaultdict(list)
    individual: dict[str, tuple[int | None, str]] = {}
    for observation in observations:
        grouped[(observation.sequence_id, observation.strain)].append(observation)
        individual[observation.row_id] = derive_activity_label(
            observation.value,
            threshold_um=threshold_um,
        )

    examples: list[ActivityExample] = []
    audit_rows: list[LabelAuditRow] = []
    for group_key in sorted(grouped, key=lambda item: (item[0], item[1] or "")):
        rows = sorted(grouped[group_key], key=lambda item: item.row_id)
        strain = group_key[1]
        labels = [individual[row.row_id][0] for row in rows]
        gram_values = {row.gram for row in rows}
        example: ActivityExample | None = None
        if strain is None:
            group_status = "excluded_missing_strain"
        elif any(label is None for label in labels):
            group_status = "excluded_nonidentifiable_replicate"
        elif len(set(cast(list[int], labels))) != 1:
            group_status = "excluded_conflicting_replicates"
        elif len(gram_values) != 1:
            group_status = "excluded_conflicting_gram_metadata"
        else:
            row = rows[0]
            label = cast(int, labels[0])
            example_id = _stable_digest("activity-example-v1", row.sequence_id, strain)
            example = ActivityExample(
                example_id=example_id,
                sequence_id=row.sequence_id,
                sequence=row.sequence,
                strain=strain,
                gram=row.gram,
                label=label,
                source_row_ids=tuple(item.row_id for item in rows),
            )
            examples.append(example)
            group_status = "included_consistent_identifiable_group"
        for row in rows:
            label, reason = individual[row.row_id]
            audit_rows.append(
                LabelAuditRow(
                    row_id=row.row_id,
                    input_line=row.input_line,
                    sequence_id=row.sequence_id,
                    sequence=row.sequence,
                    strain=row.strain,
                    gram=row.gram,
                    relation=row.value.relation,
                    lower=row.value.lower,
                    upper=row.value.upper,
                    lower_inclusive=row.value.lower_inclusive,
                    upper_inclusive=row.value.upper_inclusive,
                    unit=row.value.unit,
                    individual_label=label,
                    individual_reason=reason,
                    group_status=group_status,
                    example_id=None if example is None else example.example_id,
                )
            )
    return ActivityDataset(
        examples=tuple(sorted(examples, key=lambda item: item.example_id)),
        audit_rows=tuple(sorted(audit_rows, key=lambda item: item.row_id)),
        ignored_non_mic_rows=ignored_non_mic_rows,
    )


def _labeled_homology_clusters(
    examples: Iterable[ActivityExample],
    *,
    identity_threshold: float,
    sequence_universe: Iterable[str] | None,
) -> tuple[tuple[ActivityExample, ...], tuple[_LabeledHomologyCluster, ...], dict[str, int]]:
    items = tuple(sorted(examples, key=lambda item: item.example_id))
    if not items:
        raise ValueError("at least one labeled example is required")
    labeled_sequences = {item.sequence for item in items}
    if sequence_universe is None:
        universe = set(labeled_sequences)
    else:
        universe = {canonicalize_sequence(sequence) for sequence in sequence_universe}
        missing = sorted(labeled_sequences - universe)
        if missing:
            raise ValueError(f"full sequence universe omits {len(missing)} labeled sequence(s)")

    components = cluster_sequences(universe, identity_threshold=identity_threshold)
    examples_by_sequence: dict[str, list[ActivityExample]] = defaultdict(list)
    for item in items:
        examples_by_sequence[item.sequence].append(item)

    clusters: list[_LabeledHomologyCluster] = []
    unmodeled_components = 0
    components_with_unlabeled_sequences = 0
    for component in components:
        component_examples = tuple(
            sorted(
                (item for sequence in component for item in examples_by_sequence[sequence]),
                key=lambda item: item.example_id,
            )
        )
        if not component_examples:
            unmodeled_components += 1
            continue
        if any(sequence not in labeled_sequences for sequence in component):
            components_with_unlabeled_sequences += 1
        sequence_ids = sorted(canonical_sequence_id(sequence) for sequence in component)
        cluster_id = hashlib.sha256("\n".join(sequence_ids).encode("ascii")).hexdigest()
        clusters.append(
            _LabeledHomologyCluster(
                cluster_id=cluster_id,
                sequences=component,
                examples=component_examples,
            )
        )
    stats = {
        "full_sequence_union": len(universe),
        "labeled_sequences": len(labeled_sequences),
        "unlabeled_sequences": len(universe - labeled_sequences),
        "full_union_components": len(components),
        "modeled_components": len(clusters),
        "unmodeled_components": unmodeled_components,
        "modeled_components_with_unlabeled_sequences": components_with_unlabeled_sequences,
    }
    return items, tuple(clusters), stats


def assign_grouped_folds(
    examples: Iterable[ActivityExample],
    *,
    folds: int,
    identity_threshold: float,
    seed: int,
    sequence_universe: Iterable[str] | None = None,
) -> tuple[FoldAssignment, ...]:
    """Balance labels while clustering the complete supplied sequence universe."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    items, clusters, _ = _labeled_homology_clusters(
        examples,
        identity_threshold=identity_threshold,
        sequence_universe=sequence_universe,
    )
    if len(clusters) < folds:
        raise ValueError(
            f"cannot create {folds} nonempty folds from {len(clusters)} labeled homology groups"
        )
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster.examples),
            -abs(2 * sum(item.label for item in cluster.examples) - len(cluster.examples)),
            -len(cluster.sequences),
            _stable_digest(seed, "cluster-order", cluster.cluster_id),
        ),
    )
    totals = {
        "rows": len(items),
        "positive": sum(item.label for item in items),
        "negative": sum(1 - item.label for item in items),
        "sequences": sum(len(cluster.sequences) for cluster in clusters),
    }
    counts = {
        fold: {"rows": 0, "positive": 0, "negative": 0, "sequences": 0} for fold in range(folds)
    }
    assigned: dict[str, int] = {}
    for cluster_index, cluster in enumerate(ordered):
        cluster_rows = len(cluster.examples)
        cluster_positive = sum(item.label for item in cluster.examples)
        if cluster_index < folds:
            candidates = [fold for fold in range(folds) if counts[fold]["rows"] == 0]
        else:
            candidates = list(range(folds))
        scored: list[tuple[float, str, int]] = []
        for candidate in candidates:
            projected = {fold: dict(values) for fold, values in counts.items()}
            projected[candidate]["rows"] += cluster_rows
            projected[candidate]["positive"] += cluster_positive
            projected[candidate]["negative"] += cluster_rows - cluster_positive
            projected[candidate]["sequences"] += len(cluster.sequences)
            score = 0.0
            for metric, total in totals.items():
                target = total / folds
                score += sum(
                    ((projected[fold][metric] - target) ** 2) / max(target, 1.0)
                    for fold in range(folds)
                )
            scored.append(
                (
                    score,
                    _stable_digest(seed, cluster.cluster_id, candidate),
                    candidate,
                )
            )
        _, _, chosen = min(scored)
        counts[chosen]["rows"] += cluster_rows
        counts[chosen]["positive"] += cluster_positive
        counts[chosen]["negative"] += cluster_rows - cluster_positive
        counts[chosen]["sequences"] += len(cluster.sequences)
        assigned[cluster.cluster_id] = chosen

    assignments = [
        FoldAssignment(
            example_id=item.example_id,
            sequence_id=item.sequence_id,
            cluster_id=cluster.cluster_id,
            fold=assigned[cluster.cluster_id],
        )
        for cluster in clusters
        for item in cluster.examples
    ]
    return tuple(sorted(assignments, key=lambda item: item.example_id))


def load_frozen_fold_source(
    *,
    folds_path: str | Path,
    manifest_path: str | Path,
    oof_path: str | Path,
    config_path: str | Path,
    config: Gate1Config,
    current_examples: Iterable[ActivityExample],
) -> tuple[dict[str, FoldAssignment], dict[str, object]]:
    """Verify and load a prior Gate-1 fold map as a content-addressed input."""

    folds_file = Path(folds_path)
    manifest_file = Path(manifest_path)
    oof_file = Path(oof_path)
    folds_payload, folds_sha256 = _stable_file_bytes(folds_file)
    manifest_payload, manifest_sha256 = _stable_file_bytes(manifest_file)
    oof_payload, oof_sha256 = _stable_file_bytes(oof_file)
    folds_document = json.loads(folds_payload)
    manifest = json.loads(manifest_payload)
    if not isinstance(folds_document, dict) or not isinstance(manifest, dict):
        raise ValueError("frozen fold source must contain JSON objects")
    if manifest.get("schema_version") != 1 or manifest.get("benchmark") != "gate1_strain_activity":
        raise ValueError("frozen fold manifest is not a Gate-1 benchmark manifest")
    if (
        manifest.get("fold_policy")
        != "sequence-level single-link homology groups held out together"
    ):
        raise ValueError("frozen fold manifest has an unsupported homology policy")
    if manifest.get("config_sha256") != _file_sha256(config_path):
        raise ValueError("frozen fold manifest does not match the supplied Gate-1 config")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("frozen fold manifest has no artifact map")
    folds_artifact = artifacts.get("folds")
    if not isinstance(folds_artifact, dict) or folds_artifact.get("sha256") != folds_sha256:
        raise ValueError("frozen folds checksum does not match its manifest")
    oof_artifact = artifacts.get("oof")
    if not isinstance(oof_artifact, dict) or oof_artifact.get("sha256") != oof_sha256:
        raise ValueError("frozen OOF checksum does not match its manifest")
    if not math.isclose(
        float(folds_document.get("identity_threshold", -1.0)),
        config.homology_identity_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("frozen fold identity threshold does not match the config")
    if folds_document.get("seed") != config.seed:
        raise ValueError("frozen fold seed does not match the config")

    raw_assignments = folds_document.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("frozen folds must contain a non-empty assignment list")
    assignments: dict[str, FoldAssignment] = {}
    cluster_folds: dict[str, set[int]] = defaultdict(set)
    sequence_folds: dict[str, set[int]] = defaultdict(set)
    for index, raw in enumerate(raw_assignments):
        if not isinstance(raw, dict):
            raise ValueError(f"frozen fold assignment {index} is not an object")
        example_id = str(raw.get("example_id", ""))
        sequence_id = str(raw.get("sequence_id", ""))
        cluster_id = str(raw.get("cluster_id", ""))
        fold = raw.get("fold")
        if not example_id or not sequence_id or not cluster_id:
            raise ValueError(f"frozen fold assignment {index} has an empty identifier")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < config.folds:
            raise ValueError(f"frozen fold assignment {index} has an invalid fold")
        if example_id in assignments:
            raise ValueError(f"duplicate frozen fold example_id: {example_id}")
        assignment = FoldAssignment(
            example_id=example_id,
            sequence_id=sequence_id,
            cluster_id=cluster_id,
            fold=fold,
        )
        assignments[example_id] = assignment
        cluster_folds[cluster_id].add(fold)
        sequence_folds[sequence_id].add(fold)
    if any(len(folds) != 1 for folds in cluster_folds.values()):
        raise ValueError("a frozen homology cluster spans multiple folds")
    if any(len(folds) != 1 for folds in sequence_folds.values()):
        raise ValueError("a frozen sequence spans multiple folds")

    declared_models = manifest.get("models")
    if (
        not isinstance(declared_models, list)
        or not declared_models
        or any(not isinstance(model, str) or not model for model in declared_models)
        or len(set(declared_models)) != len(declared_models)
    ):
        raise ValueError("frozen fold manifest has an invalid model list")
    model_names = set(declared_models)
    source_labels: dict[str, tuple[str, int]] = {}
    models_by_example: dict[str, set[str]] = defaultdict(set)
    with io.StringIO(oof_payload.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model", "example_id", "sequence_id", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("frozen OOF is missing required columns")
        for row_number, row in enumerate(reader, start=2):
            model = str(row["model"])
            example_id = str(row["example_id"])
            sequence_id = str(row["sequence_id"])
            if model not in model_names or not example_id or not sequence_id:
                raise ValueError(f"frozen OOF row {row_number} has invalid identifiers")
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"frozen OOF row {row_number} has an invalid label") from error
            if label not in {0, 1}:
                raise ValueError(f"frozen OOF row {row_number} has an invalid label")
            if model in models_by_example[example_id]:
                raise ValueError(f"duplicate frozen OOF model/example row: {model}/{example_id}")
            models_by_example[example_id].add(model)
            previous = source_labels.setdefault(example_id, (sequence_id, label))
            if previous != (sequence_id, label):
                raise ValueError(f"frozen OOF models disagree for example {example_id}")
    if set(source_labels) != set(assignments):
        raise ValueError("frozen OOF and fold map do not have identical example support")
    if any(models != model_names for models in models_by_example.values()):
        raise ValueError("frozen OOF does not have complete model support for every example")

    current_by_id = {item.example_id: item for item in current_examples}
    missing_current = sorted(set(current_by_id) - set(source_labels))
    if missing_current:
        raise ValueError(f"frozen OOF omits {len(missing_current)} current example(s)")
    changed_labels = [
        example_id
        for example_id, item in current_by_id.items()
        if source_labels[example_id] != (item.sequence_id, item.label)
    ]
    if changed_labels:
        raise ValueError(f"frozen OOF disagrees with {len(changed_labels)} current label(s)")
    source_only = set(source_labels) - set(current_by_id)

    source = {
        "folds_sha256": folds_sha256,
        "manifest_sha256": manifest_sha256,
        "oof_sha256": oof_sha256,
        "source_input_assays_sha256": manifest.get("input_assays_sha256"),
        "source_normalized_summary_sha256": manifest.get("normalized_summary_sha256"),
        "source_examples": len(assignments),
        "source_homology_clusters": len(cluster_folds),
        "current_examples": len(current_by_id),
        "shared_labels_unchanged": len(current_by_id),
        "source_only_examples": len(source_only),
        "source_only_positive_examples": sum(source_labels[item][1] for item in source_only),
        "source_only_negative_examples": sum(1 - source_labels[item][1] for item in source_only),
    }
    return assignments, source


def assess_frozen_fold_compatibility(
    examples: Iterable[ActivityExample],
    *,
    frozen_assignments: Mapping[str, FoldAssignment],
    folds: int,
    identity_threshold: float,
    sequence_universe: Iterable[str],
) -> tuple[tuple[FoldAssignment, ...] | None, dict[str, object]]:
    """Audit prior folds against full-union components before any model fit."""

    items, clusters, union_stats = _labeled_homology_clusters(
        examples,
        identity_threshold=identity_threshold,
        sequence_universe=sequence_universe,
    )
    current_by_id = {item.example_id: item for item in items}
    missing = sorted(set(current_by_id) - set(frozen_assignments))
    if missing:
        raise ValueError(f"frozen fold source omits {len(missing)} current example(s)")
    for example_id, item in current_by_id.items():
        if frozen_assignments[example_id].sequence_id != item.sequence_id:
            raise ValueError(f"frozen fold sequence mismatch for example {example_id}")

    output: list[FoldAssignment] = []
    cross_fold_components: list[dict[str, object]] = []
    merged_source_components = 0
    for cluster in clusters:
        source_rows = [frozen_assignments[item.example_id] for item in cluster.examples]
        source_folds = {item.fold for item in source_rows}
        if len(source_folds) != 1:
            modeled_sequence_ids = {item.sequence_id for item in cluster.examples}
            cross_fold_components.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "folds": sorted(source_folds),
                    "sequence_ids": sorted(
                        canonical_sequence_id(sequence) for sequence in cluster.sequences
                    ),
                    "unmodeled_sequence_ids": sorted(
                        canonical_sequence_id(sequence)
                        for sequence in cluster.sequences
                        if canonical_sequence_id(sequence) not in modeled_sequence_ids
                    ),
                    "source_cluster_ids": sorted({item.cluster_id for item in source_rows}),
                }
            )
            continue
        if len({item.cluster_id for item in source_rows}) > 1:
            merged_source_components += 1
        fold = next(iter(source_folds))
        output.extend(
            FoldAssignment(
                example_id=item.example_id,
                sequence_id=item.sequence_id,
                cluster_id=cluster.cluster_id,
                fold=fold,
            )
            for item in cluster.examples
        )
    source_only_ids = set(frozen_assignments) - set(current_by_id)
    report: dict[str, object] = {
        "schema_version": 1,
        "policy": "full_sequence_union_frozen_fold_compatibility_audit",
        "identity_threshold": identity_threshold,
        **union_stats,
        "reused_examples": len(output),
        "source_only_examples": len(source_only_ids),
        "source_only_sequence_ids": sorted(
            {frozen_assignments[example_id].sequence_id for example_id in source_only_ids}
        ),
        "components_merging_source_clusters": merged_source_components,
        "cross_fold_bridge_components": cross_fold_components,
    }
    if cross_fold_components:
        report["status"] = "incompatible"
        report["reused_examples"] = 0
        report["observed_folds"] = []
        return None, report

    observed_folds = {item.fold for item in output}
    if observed_folds != set(range(folds)):
        raise ValueError(
            f"reused folds must keep every fold nonempty; observed {sorted(observed_folds)}"
        )
    report["status"] = "compatible"
    report["observed_folds"] = sorted(observed_folds)
    return tuple(sorted(output, key=lambda item: item.example_id)), report


def reuse_grouped_folds(
    examples: Iterable[ActivityExample],
    *,
    frozen_assignments: Mapping[str, FoldAssignment],
    folds: int,
    identity_threshold: float,
    sequence_universe: Iterable[str],
) -> tuple[tuple[FoldAssignment, ...], dict[str, object]]:
    """Reuse prior folds, failing closed when the full-union audit rejects them."""

    assignments, report = assess_frozen_fold_compatibility(
        examples,
        frozen_assignments=frozen_assignments,
        folds=folds,
        identity_threshold=identity_threshold,
        sequence_universe=sequence_universe,
    )
    if assignments is None:
        first = cast(list[dict[str, object]], report["cross_fold_bridge_components"])[0]
        raise ValueError(
            "full sequence-union bridge makes frozen folds unsafe: "
            f"component {first['cluster_id']} spans folds {first['folds']}"
        )
    return assignments, report


def resolve_grouped_fold_policy(
    examples: Iterable[ActivityExample],
    *,
    frozen_assignments: Mapping[str, FoldAssignment],
    folds: int,
    identity_threshold: float,
    seed: int,
    sequence_universe: Iterable[str],
    allow_fresh_fallback: bool,
) -> tuple[tuple[FoldAssignment, ...], dict[str, object], str]:
    """Resolve reuse versus an explicitly authorized fresh full-union split."""

    assignments, report = assess_frozen_fold_compatibility(
        examples,
        frozen_assignments=frozen_assignments,
        folds=folds,
        identity_threshold=identity_threshold,
        sequence_universe=sequence_universe,
    )
    if assignments is not None:
        return (
            assignments,
            report,
            "full_sequence_union_bridge_audit_then_reuse_frozen_example_folds",
        )
    if not allow_fresh_fallback:
        raise ValueError(
            "frozen folds are incompatible with the full sequence union; "
            "fresh assignment requires explicit authorization"
        )
    fresh = assign_grouped_folds(
        examples,
        folds=folds,
        identity_threshold=identity_threshold,
        seed=seed,
        sequence_universe=sequence_universe,
    )
    report["fallback"] = "fresh_full_sequence_union_assignment"
    report["fallback_homology_clusters"] = len({item.cluster_id for item in fresh})
    report["fallback_observed_folds"] = sorted({item.fold for item in fresh})
    return (
        fresh,
        report,
        "frozen_source_incompatible_then_full_sequence_union_greedy_balance",
    )


def _maximum_training_identity(sequence: str, training_sequences: tuple[str, ...]) -> float:
    return max(global_sequence_identity(sequence, other) for other in training_sequences)


def make_oof_predictions(
    examples: Iterable[ActivityExample],
    assignments: Iterable[FoldAssignment],
    *,
    config: Gate1Config,
) -> tuple[OofPrediction, ...]:
    """Fit both frozen baselines in each fold and return complete OOF predictions."""

    items = tuple(sorted(examples, key=lambda item: item.example_id))
    by_id = {item.example_id: item for item in items}
    assignment_by_id = {item.example_id: item for item in assignments}
    if set(by_id) != set(assignment_by_id):
        raise ValueError("fold assignments must cover every example exactly once")
    output: list[OofPrediction] = []
    for fold in range(config.folds):
        training = tuple(item for item in items if assignment_by_id[item.example_id].fold != fold)
        testing = tuple(item for item in items if assignment_by_id[item.example_id].fold == fold)
        if not training or not testing:
            raise ValueError(f"fold {fold} has an empty train or test partition")
        training_rows = tuple(item.model_input for item in training)
        testing_rows = tuple(item.model_input for item in testing)
        labels = np.asarray([item.label for item in training], dtype=np.int64)
        models = (
            DescriptorLogisticOracle(
                l2=config.logistic.l2,
                max_iterations=config.logistic.max_iterations,
                tolerance=config.logistic.tolerance,
                prior_strength=config.logistic.prior_strength,
            ),
            HomologyKnnOracle(
                neighbors=config.knn.neighbors,
                similarity_power=config.knn.similarity_power,
                prior_strength=config.knn.prior_strength,
                minimum_weight=config.knn.minimum_weight,
            ),
        )
        model_probabilities: dict[str, FloatArray] = {}
        for model in models:
            model.fit(training_rows, labels)
            model_probabilities[model.name] = model.predict_proba(testing_rows)
        model_probabilities["equal_weight_ensemble"] = np.mean(
            np.stack(tuple(model_probabilities.values()), axis=0),
            axis=0,
        )
        training_sequences = tuple(sorted({item.sequence for item in training}))
        maximum_identity_by_sequence = {
            sequence: _maximum_training_identity(sequence, training_sequences)
            for sequence in sorted({item.sequence for item in testing})
        }
        for test_index, item in enumerate(testing):
            assignment = assignment_by_id[item.example_id]
            maximum_identity = maximum_identity_by_sequence[item.sequence]
            if maximum_identity >= config.homology_identity_threshold:
                raise AssertionError(
                    "homology leakage: test sequence reaches the training identity threshold"
                )
            for model_name, probabilities in model_probabilities.items():
                output.append(
                    OofPrediction(
                        model=model_name,
                        example_id=item.example_id,
                        sequence_id=item.sequence_id,
                        sequence=item.sequence,
                        strain=item.strain,
                        gram=item.gram,
                        label=item.label,
                        source_observations=len(item.source_row_ids),
                        fold=fold,
                        cluster_id=assignment.cluster_id,
                        max_train_identity=maximum_identity,
                        probability=float(probabilities[test_index]),
                    )
                )
    expected = len(items) * 3
    if len(output) != expected:
        raise AssertionError(f"expected {expected} OOF rows, created {len(output)}")
    return tuple(sorted(output, key=lambda item: (item.model, item.example_id)))


def _roc_auc(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: IntArray, probabilities: FloatArray) -> float | None:
    positive_count = int(np.sum(labels))
    if positive_count == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    index = 0
    while index < labels.size:
        end = index + 1
        while end < labels.size and sorted_probability[end] == sorted_probability[index]:
            end += 1
        group = sorted_labels[index:end]
        new_positives = int(np.sum(group))
        true_positive += new_positives
        false_positive += len(group) - new_positives
        precision = true_positive / (true_positive + false_positive)
        average_precision += (new_positives / positive_count) * precision
        index = end
    return float(average_precision)


def binary_metrics(
    labels: IntArray | Sequence[int],
    probabilities: FloatArray | Sequence[float],
    *,
    calibration_bins: int,
) -> dict[str, int | float | None]:
    """Compute discrimination, calibration, and fixed-threshold metrics."""

    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or probability.ndim != 1 or y.size != probability.size or y.size == 0:
        raise ValueError("labels and probabilities must be equal, non-empty vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must contain only 0 and 1")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    prediction = probability >= 0.5
    positive = y == 1
    negative = ~positive
    true_positive_rate = float(np.mean(prediction[positive])) if np.any(positive) else None
    true_negative_rate = float(np.mean(~prediction[negative])) if np.any(negative) else None
    balanced_accuracy = (
        None
        if true_positive_rate is None or true_negative_rate is None
        else (true_positive_rate + true_negative_rate) / 2.0
    )
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    bin_index = np.digitize(probability, edges[1:-1], right=False)
    calibration_error = 0.0
    for index in range(calibration_bins):
        selected = bin_index == index
        if np.any(selected):
            calibration_error += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(y[selected]))
            )
    return {
        "n": int(y.size),
        "positives": int(np.sum(y)),
        "negatives": int(y.size - np.sum(y)),
        "prevalence": float(np.mean(y)),
        "roc_auc": _roc_auc(y, probability),
        "average_precision": _average_precision(y, probability),
        "brier": float(np.mean(np.square(probability - y))),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "balanced_accuracy_at_0_5": balanced_accuracy,
        "sensitivity_at_0_5": true_positive_rate,
        "specificity_at_0_5": true_negative_rate,
        "ece_equal_width": calibration_error,
    }


def _cluster_bootstrap_intervals(
    predictions: Sequence[OofPrediction],
    *,
    calibration_bins: int,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss")
    point = binary_metrics(
        [item.label for item in predictions],
        [item.probability for item in predictions],
        calibration_bins=calibration_bins,
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    if replicates:
        by_cluster: dict[str, list[OofPrediction]] = defaultdict(list)
        for item in predictions:
            by_cluster[item.cluster_id].append(item)
        clusters = tuple(sorted(by_cluster))
        generator = np.random.default_rng(seed)
        for _ in range(replicates):
            sampled = generator.choice(clusters, size=len(clusters), replace=True)
            rows = [row for cluster in sampled for row in by_cluster[str(cluster)]]
            metrics = binary_metrics(
                [item.label for item in rows],
                [item.probability for item in rows],
                calibration_bins=calibration_bins,
            )
            for metric_name in metric_names:
                value = metrics[metric_name]
                if value is not None:
                    values[metric_name].append(float(value))
    output: dict[str, dict[str, float | int | None]] = {}
    for metric_name in metric_names:
        samples = np.asarray(values[metric_name], dtype=np.float64)
        output[metric_name] = {
            "point": cast(float | None, point[metric_name]),
            "lower": None if samples.size == 0 else float(np.quantile(samples, 0.025)),
            "upper": None if samples.size == 0 else float(np.quantile(samples, 0.975)),
            "successful_replicates": int(samples.size),
        }
    return output


def _metric_subset(
    predictions: Sequence[OofPrediction],
    *,
    calibration_bins: int,
) -> dict[str, int | float | None]:
    return binary_metrics(
        [item.label for item in predictions],
        [item.probability for item in predictions],
        calibration_bins=calibration_bins,
    )


def summarize_oof_predictions(
    predictions: Iterable[OofPrediction],
    *,
    config: Gate1Config,
) -> dict[str, object]:
    items = tuple(predictions)
    by_model: dict[str, list[OofPrediction]] = defaultdict(list)
    for item in items:
        by_model[item.model].append(item)
    output: dict[str, object] = {}
    for model_name in sorted(by_model):
        model_rows = tuple(sorted(by_model[model_name], key=lambda item: item.example_id))
        by_fold = {
            str(fold): _metric_subset(
                [item for item in model_rows if item.fold == fold],
                calibration_bins=config.calibration_bins,
            )
            for fold in range(config.folds)
        }
        strains = sorted({item.strain for item in model_rows})
        by_strain = {
            strain: _metric_subset(
                [item for item in model_rows if item.strain == strain],
                calibration_bins=config.calibration_bins,
            )
            for strain in strains
        }
        by_similarity: dict[str, dict[str, int | float | None]] = {}
        for left, right in zip(
            config.similarity_bin_edges,
            config.similarity_bin_edges[1:],
            strict=False,
        ):
            name = f"[{left:.2f},{right:.2f})"
            selected = [
                item
                for item in model_rows
                if left <= item.max_train_identity < right
                or (right == 1.0 and item.max_train_identity == right)
            ]
            if selected:
                by_similarity[name] = _metric_subset(
                    selected,
                    calibration_bins=config.calibration_bins,
                )
        output[model_name] = {
            "overall": _metric_subset(
                model_rows,
                calibration_bins=config.calibration_bins,
            ),
            "homology_cluster_bootstrap_95ci": _cluster_bootstrap_intervals(
                model_rows,
                calibration_bins=config.calibration_bins,
                replicates=config.bootstrap_replicates,
                seed=config.seed,
            ),
            "by_fold": by_fold,
            "by_strain": by_strain,
            "by_max_train_identity": by_similarity,
        }
    member_names = ("descriptor_logistic", "homology_knn")
    if all(name in by_model for name in (*member_names, "equal_weight_ensemble")):
        aligned = {
            name: {item.example_id: item.probability for item in by_model[name]}
            for name in (*member_names, "equal_weight_ensemble")
        }
        example_ids = sorted(aligned[member_names[0]])
        left = np.asarray([aligned[member_names[0]][item] for item in example_ids])
        right = np.asarray([aligned[member_names[1]][item] for item in example_ids])
        correlation = (
            float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else None
        )
        output["model_diversity"] = {
            "member_prediction_pearson": correlation,
            "members": list(member_names),
            "ensemble_policy": "untrained_equal_probability_mean",
        }
    return output


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True, allow_nan=False) + "\n")


def _write_oof_csv(path: Path, predictions: Sequence[OofPrediction]) -> None:
    fieldnames = [
        "model",
        "example_id",
        "sequence_id",
        "sequence",
        "strain",
        "gram",
        "label",
        "source_observations",
        "fold",
        "cluster_id",
        "max_train_identity",
        "probability",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in predictions:
            row = asdict(item)
            row["max_train_identity"] = f"{item.max_train_identity:.12g}"
            row["probability"] = f"{item.probability:.12g}"
            writer.writerow(row)


def _fold_report(
    examples: Sequence[ActivityExample],
    assignments: Sequence[FoldAssignment],
    *,
    config: Gate1Config,
    method: str,
) -> dict[str, object]:
    example_by_id = {item.example_id: item for item in examples}
    folds: dict[str, object] = {}
    for fold in range(config.folds):
        selected = [item for item in assignments if item.fold == fold]
        labels = [example_by_id[item.example_id].label for item in selected]
        folds[str(fold)] = {
            "examples": len(selected),
            "sequences": len({item.sequence_id for item in selected}),
            "homology_clusters": len({item.cluster_id for item in selected}),
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
        }
    return {
        "method": method,
        "identity_threshold": config.homology_identity_threshold,
        "seed": config.seed,
        "folds": folds,
        "assignments": [asdict(item) for item in assignments],
    }


def run_gate1_benchmark(
    *,
    assays_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    normalized_summary_path: str | Path | None = None,
    sequences_path: str | Path | None = None,
    frozen_folds_path: str | Path | None = None,
    frozen_fold_manifest_path: str | Path | None = None,
    frozen_oof_path: str | Path | None = None,
    allow_fresh_fold_fallback: bool = False,
    code_manifest_path: str | Path | None = None,
    normalized_data_manifest_path: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, object]:
    """Execute deterministic grouped CV and write a fully auditable artifact."""

    assays = Path(assays_path)
    config_file = Path(config_path)
    normalized_summary_file = (
        assays.parent / "summary.json"
        if normalized_summary_path is None
        else Path(normalized_summary_path)
    )
    sequences_file = (
        assays.parent / "sequences.jsonl" if sequences_path is None else Path(sequences_path)
    )
    frozen_inputs = (frozen_folds_path, frozen_fold_manifest_path, frozen_oof_path)
    if any(path is None for path in frozen_inputs) and any(
        path is not None for path in frozen_inputs
    ):
        raise ValueError("frozen folds, manifest, and OOF predictions must be supplied together")
    if git_commit is not None and (
        len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("git_commit must be a full lowercase SHA-1 commit ID")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = Gate1Config.from_toml(config_file)
    normalized_summary = verify_normalized_dataset_summary(
        assays,
        normalized_summary_file,
        sequences_path=sequences_file,
    )
    sequence_universe = read_normalized_sequence_union(sequences_file)
    verify_sequence_union_declaration(normalized_summary, sequence_universe)
    endpoint_counts = read_normalized_endpoint_counts(
        assays,
        sequence_universe=sequence_universe,
    )
    observations, ignored = read_normalized_mic_observations(assays)
    if endpoint_counts.get("mic", 0) != len(observations) or (
        sum(endpoint_counts.values()) - endpoint_counts.get("mic", 0) != ignored
    ):
        raise AssertionError("MIC reader counts disagree with the normalized endpoint histogram")
    verify_declared_endpoint_counts(normalized_summary, endpoint_counts)
    dataset = build_activity_dataset(
        observations,
        threshold_um=config.activity_threshold_um,
        ignored_non_mic_rows=ignored,
    )
    fold_source: dict[str, object] | None = None
    if frozen_folds_path is None:
        assignments = assign_grouped_folds(
            dataset.examples,
            folds=config.folds,
            identity_threshold=config.homology_identity_threshold,
            seed=config.seed,
            sequence_universe=sequence_universe,
        )
        split_compatibility: dict[str, object] = {
            "schema_version": 1,
            "status": "fresh_assignment",
            "policy": "cluster_full_sequence_union_then_greedy_multicriterion_balance",
            "identity_threshold": config.homology_identity_threshold,
            "full_sequence_union": len(sequence_universe),
            "labeled_sequences": len({item.sequence_id for item in dataset.examples}),
            "unlabeled_sequences": len(sequence_universe)
            - len({item.sequence_id for item in dataset.examples}),
            "modeled_components": len({item.cluster_id for item in assignments}),
            "observed_folds": sorted({item.fold for item in assignments}),
        }
        fold_method = "full_sequence_union_then_greedy_multicriterion_balance"
    else:
        assert frozen_fold_manifest_path is not None
        assert frozen_oof_path is not None
        frozen_assignments, fold_source = load_frozen_fold_source(
            folds_path=frozen_folds_path,
            manifest_path=frozen_fold_manifest_path,
            oof_path=frozen_oof_path,
            config_path=config_file,
            config=config,
            current_examples=dataset.examples,
        )
        assignments, split_compatibility, fold_method = resolve_grouped_fold_policy(
            dataset.examples,
            frozen_assignments=frozen_assignments,
            folds=config.folds,
            identity_threshold=config.homology_identity_threshold,
            seed=config.seed,
            sequence_universe=sequence_universe,
            allow_fresh_fallback=allow_fresh_fold_fallback,
        )
        split_compatibility["fold_source"] = fold_source
    predictions = make_oof_predictions(dataset.examples, assignments, config=config)
    metrics = summarize_oof_predictions(predictions, config=config)

    label_summary = {
        "input_mic_observations": len(dataset.audit_rows),
        "ignored_non_mic_rows": dataset.ignored_non_mic_rows,
        "included_strain_level_examples": len(dataset.examples),
        "included_source_observations": sum(
            row.group_status == "included_consistent_identifiable_group"
            for row in dataset.audit_rows
        ),
        "excluded_source_observations": sum(
            row.group_status != "included_consistent_identifiable_group"
            for row in dataset.audit_rows
        ),
        "activity_threshold_um": config.activity_threshold_um,
        "positive_examples": sum(item.label for item in dataset.examples),
        "negative_examples": sum(1 - item.label for item in dataset.examples),
    }
    paths = {
        "label_audit": output / "label_audit.jsonl",
        "folds": output / "folds.json",
        "oof": output / "oof_predictions.csv",
        "metrics": output / "metrics.json",
        "split_compatibility": output / "split_compatibility.json",
        "manifest": output / "manifest.json",
    }
    _write_jsonl(paths["label_audit"], (asdict(item) for item in dataset.audit_rows))
    _write_json(
        paths["folds"],
        _fold_report(dataset.examples, assignments, config=config, method=fold_method),
    )
    _write_oof_csv(paths["oof"], predictions)
    _write_json(paths["metrics"], metrics)
    _write_json(paths["split_compatibility"], split_compatibility)
    provenance: dict[str, object] = {}
    for name, path in (
        ("code_manifest", code_manifest_path),
        ("normalized_data_manifest", normalized_data_manifest_path),
    ):
        if path is not None:
            source = Path(path)
            if not source.is_file():
                raise ValueError(f"{name} is not a regular file: {source}")
            provenance[name] = {"filename": source.name, "sha256": _file_sha256(source)}
    if git_commit is not None:
        provenance["git_commit"] = git_commit
    manifest = {
        "schema_version": 1,
        "benchmark": "gate1_strain_activity",
        "input_assays_sha256": _file_sha256(assays),
        "input_sequences_sha256": _file_sha256(sequences_file),
        "normalized_summary_sha256": _file_sha256(normalized_summary_file),
        "normalized_schema_version": normalized_summary["schema_version"],
        "normalized_parser_id": normalized_summary.get("parser_id"),
        "normalized_endpoint_counts": normalized_summary.get("endpoint_counts"),
        "source_artifacts": normalized_summary["artifacts"],
        "config_sha256": _file_sha256(config_file),
        "fold_assignment_policy": fold_method,
        "fold_source": fold_source,
        "provenance": provenance,
        "runtime": {
            "numpy": np.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "label_policy": "MIC <= threshold only when every replicate interval identifies same class",
        "fold_policy": "sequence-level single-link homology groups held out together",
        "models": ["descriptor_logistic", "homology_knn", "equal_weight_ensemble"],
        "label_summary": label_summary,
        "artifacts": {
            name: {"filename": path.name, "sha256": _file_sha256(path)}
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assays", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        type=Path,
        help="defaults to sequences.jsonl beside --assays and defines the full clustering union",
    )
    parser.add_argument(
        "--normalized-summary",
        type=Path,
        help="defaults to summary.json beside --assays",
    )
    parser.add_argument("--frozen-folds", type=Path)
    parser.add_argument("--frozen-fold-manifest", type=Path)
    parser.add_argument("--frozen-oof", type=Path)
    parser.add_argument(
        "--allow-fresh-fold-fallback",
        action="store_true",
        help="after an incompatible frozen-fold audit, authorize a fresh full-union split",
    )
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--normalized-data-manifest", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/oracle_gate1.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_gate1_benchmark(
        assays_path=args.assays,
        config_path=args.config,
        output_dir=args.output_dir,
        normalized_summary_path=args.normalized_summary,
        sequences_path=args.sequences,
        frozen_folds_path=args.frozen_folds,
        frozen_fold_manifest_path=args.frozen_fold_manifest,
        frozen_oof_path=args.frozen_oof,
        allow_fresh_fold_fallback=args.allow_fresh_fold_fallback,
        code_manifest_path=args.code_manifest,
        normalized_data_manifest_path=args.normalized_data_manifest,
        git_commit=args.git_commit,
    )
    print(json.dumps(manifest["label_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

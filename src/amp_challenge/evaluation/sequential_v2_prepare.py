"""Outcome-isolated prepare computation for sequential mixed-acquisition v2.

The caller must construct :class:`PrepareCapability` at the authenticated stage
boundary.  This module can see base labels and label-free acquisition metadata,
but it has no API accepting an acquisition or outer outcome vault.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from amp_challenge.acquisition.sequential_v2_selector import (
    OBJECTIVES,
    PoolCandidate,
    RandomPoolCandidate,
)
from amp_challenge.evaluation.sequential_v2_primitives import (
    NEGATIVE_TARGETS,
    POSITIVE_TARGETS,
    TARGET_GRAM,
    TARGETS,
    ContextRow,
    DescriptorLogisticState,
    DiversityTransform,
    NoveltyEvidence,
    cluster_diversity_components,
    diversity_component_id,
    exact_training_novelty,
    fit_descriptor_logistic,
    fit_diversity_transform,
    predict_target_objectives,
    transform_diversity,
)
from amp_challenge.evaluation.sequential_v2_protocol import RotationSpec
from amp_challenge.evaluation.sequential_v2_staging import LabelFreeContext, PrepareCapability
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_SHA256 = re.compile(r"[0-9a-f]{64}")


def id_stream_sha256(values: Iterable[str]) -> str:
    """Hash a nonempty sorted unique lowercase-SHA stream with one LF per ID."""

    raw = tuple(values)
    if (
        not raw
        or any(not isinstance(value, str) for value in raw)
        or raw != tuple(sorted(set(raw)))
        or any(_SHA256.fullmatch(value) is None for value in raw)
    ):
        raise ValueError("ID stream must be nonempty, sorted, unique lowercase SHA-256 values")
    return hashlib.sha256("".join(f"{value}\n" for value in raw).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class PoolPrediction:
    """Complete seven-target and three-objective prediction for one pool sequence."""

    sequence_id: str
    target_probabilities: tuple[float, ...]
    objective_probabilities: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or _SHA256.fullmatch(self.sequence_id) is None:
            raise ValueError("pool prediction sequence_id must be a lowercase SHA-256")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (*self.target_probabilities, *self.objective_probabilities)
        ):
            raise ValueError("pool probabilities must be real numbers, not aliases")
        target = tuple(float(value) for value in self.target_probabilities)
        objective = tuple(float(value) for value in self.objective_probabilities)
        if len(target) != len(TARGETS) or len(objective) != len(OBJECTIVES):
            raise ValueError("pool prediction dimensions disagree with the frozen panel")
        if any(not math.isfinite(value) or not 1e-6 <= value <= 0.999999 for value in target):
            raise ValueError("target probabilities must lie in the accepted model range")
        if any(not math.isfinite(value) or not 1e-6 <= value <= 0.999999 for value in objective):
            raise ValueError("objective probabilities must lie in the accepted model range")
        object.__setattr__(self, "target_probabilities", target)
        object.__setattr__(self, "objective_probabilities", objective)

    def document(self, *, rotation_id: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rotation_id": rotation_id,
            "sequence_id": self.sequence_id,
            "target_probabilities_hex": {
                target: probability.hex()
                for target, probability in zip(
                    TARGETS,
                    self.target_probabilities,
                    strict=True,
                )
            },
            "objective_probabilities_hex": {
                objective: probability.hex()
                for objective, probability in zip(
                    OBJECTIVES,
                    self.objective_probabilities,
                    strict=True,
                )
            },
        }


@dataclass(frozen=True, slots=True)
class PoolComponent:
    """One complete pool-local 70%-identity component."""

    component_id: str
    sequence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.component_id.startswith("seqv2-div70:")
            or _SHA256.fullmatch(self.component_id.removeprefix("seqv2-div70:")) is None
        ):
            raise ValueError("pool component ID violates the sequential-v2 namespace")
        if (
            not self.sequence_ids
            or any(not isinstance(value, str) for value in self.sequence_ids)
            or self.sequence_ids != tuple(sorted(set(self.sequence_ids)))
            or any(_SHA256.fullmatch(value) is None for value in self.sequence_ids)
        ):
            raise ValueError("pool component members must be sorted unique sequence IDs")

    def document(self, *, spec: RotationSpec) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rotation_id": spec.rotation_id,
            "role": "pool",
            "fold": spec.pool_fold,
            "diversity_component_id": self.component_id,
            "sequence_ids": list(self.sequence_ids),
        }


@dataclass(frozen=True, slots=True)
class PreparedRotation:
    """Complete deterministic outcome-free result of one prepare worker."""

    spec: RotationSpec
    base_model: DescriptorLogisticState
    diversity_transform: DiversityTransform
    base_example_ids: tuple[str, ...]
    base_sequence_ids: tuple[str, ...]
    base_sequences: tuple[str, ...]
    pool_predictions: tuple[PoolPrediction, ...]
    pool_novelty: tuple[NoveltyEvidence, ...]
    pool_components: tuple[PoolComponent, ...]
    pool_candidates: tuple[PoolCandidate, ...]
    random_pool_candidates: tuple[RandomPoolCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RotationSpec):
            raise TypeError("prepared rotation spec must be a RotationSpec")
        if not isinstance(self.base_model, DescriptorLogisticState) or not isinstance(
            self.diversity_transform, DiversityTransform
        ):
            raise TypeError("prepared rotation model and transform states are invalid")
        id_stream_sha256(self.base_example_ids)
        id_stream_sha256(self.base_sequence_ids)
        if (
            not isinstance(self.base_sequences, tuple)
            or not self.base_sequences
            or len(set(self.base_sequences)) != len(self.base_sequences)
        ):
            raise ValueError("base sequences must be a nonempty immutable unique tuple")
        canonical_base = tuple(canonicalize_sequence(value) for value in self.base_sequences)
        if (
            canonical_base != self.base_sequences
            or tuple(canonical_sequence_id(value) for value in canonical_base)
            != self.base_sequence_ids
        ):
            raise ValueError("base sequences differ from the authenticated sequence-ID evidence")
        if self.base_model.training_contexts != len(self.base_example_ids):
            raise ValueError("base model training census differs from base example evidence")
        if self.diversity_transform.fit_sequence_ids != self.base_sequence_ids:
            raise ValueError("diversity transform fit IDs differ from base sequence evidence")
        expected_transform = fit_diversity_transform(self.base_sequences)
        if self.diversity_transform != expected_transform:
            raise ValueError("diversity transform differs from the authenticated base sequences")
        ids = tuple(candidate.sequence_id for candidate in self.pool_candidates)
        if not ids or ids != tuple(sorted(set(ids))):
            raise ValueError("prepared pool candidates must be nonempty, sorted, and unique")
        if tuple(item.sequence_id for item in self.random_pool_candidates) != ids:
            raise ValueError("random and prediction-bearing pool views must align exactly")
        if tuple(item.sequence_id for item in self.pool_predictions) != ids:
            raise ValueError("pool prediction evidence must align to candidate order")
        if tuple(item.sequence_id for item in self.pool_novelty) != ids:
            raise ValueError("pool novelty evidence must align to candidate order")
        if any(not isinstance(item, NoveltyEvidence) for item in self.pool_novelty):
            raise TypeError("pool novelty evidence must contain only NoveltyEvidence instances")
        if any(
            candidate.rotation_id != self.spec.rotation_id or not candidate.eligible
            for candidate in self.pool_candidates
        ):
            raise ValueError("prepared candidates disagree with the rotation or support contract")
        model_targets, model_objectives = predict_target_objectives(
            self.base_model,
            tuple(candidate.sequence for candidate in self.pool_candidates),
        )
        expected_features = transform_diversity(
            self.diversity_transform,
            tuple(candidate.sequence for candidate in self.pool_candidates),
        )
        expected_novelty = exact_training_novelty(
            tuple(candidate.sequence for candidate in self.pool_candidates),
            self.base_sequences,
        )
        if self.pool_novelty != expected_novelty:
            raise ValueError(
                "pool novelty evidence differs from exact authenticated-base Indel maxima"
            )
        expected_component_map, expected_component_members = cluster_diversity_components(
            tuple(candidate.sequence for candidate in self.pool_candidates),
            role="pool",
            fold=self.spec.pool_fold,
        )
        expected_components = tuple(
            sorted(
                (
                    PoolComponent(
                        component_id=expected_component_map[members[0]],
                        sequence_ids=members,
                    )
                    for members in expected_component_members
                ),
                key=lambda item: item.component_id,
            )
        )
        if self.pool_components != expected_components:
            raise ValueError(
                "pool component evidence differs from exact candidate-sequence clustering"
            )
        for candidate, random_candidate, prediction, novelty in zip(
            self.pool_candidates,
            self.random_pool_candidates,
            self.pool_predictions,
            self.pool_novelty,
            strict=True,
        ):
            if (
                random_candidate.rotation_id != self.spec.rotation_id
                or random_candidate.sequence != candidate.sequence
                or random_candidate.diversity_component_id != candidate.diversity_component_id
                or random_candidate.eligible is not True
            ):
                raise ValueError("minimal random view differs from the prediction-bearing view")
            if candidate.objective_probabilities != prediction.objective_probabilities:
                raise ValueError("candidate objectives differ from pool prediction evidence")
            if candidate.novelty != novelty.novelty:
                raise ValueError("candidate novelty differs from pool novelty evidence")
            target_values = np.asarray(prediction.target_probabilities, dtype=np.float64)
            target_index = {target: index for index, target in enumerate(TARGETS)}
            reduced_objectives = (
                float(np.mean(target_values, dtype=np.float64)),
                float(
                    np.mean(
                        target_values[[target_index[target] for target in POSITIVE_TARGETS]],
                        dtype=np.float64,
                    )
                ),
                float(
                    np.mean(
                        target_values[[target_index[target] for target in NEGATIVE_TARGETS]],
                        dtype=np.float64,
                    )
                ),
            )
            if prediction.objective_probabilities != reduced_objectives:
                raise ValueError("pool objective evidence differs from the seven-target means")
        if not np.array_equal(
            model_targets,
            np.asarray(
                [prediction.target_probabilities for prediction in self.pool_predictions],
                dtype=np.float64,
            ),
        ) or not np.array_equal(
            model_objectives,
            np.asarray(
                [prediction.objective_probabilities for prediction in self.pool_predictions],
                dtype=np.float64,
            ),
        ):
            raise ValueError("pool prediction evidence differs from the frozen base model")
        if not np.array_equal(
            expected_features,
            np.asarray(
                [candidate.features for candidate in self.pool_candidates], dtype=np.float64
            ),
        ):
            raise ValueError("candidate features differ from the fold-local transform")
        if self.pool_components != tuple(
            sorted(self.pool_components, key=lambda item: item.component_id)
        ):
            raise ValueError("pool components must be ordered by component ID")
        component_members = tuple(
            sorted(
                sequence_id
                for component in self.pool_components
                for sequence_id in component.sequence_ids
            )
        )
        if component_members != ids:
            raise ValueError("pool components must partition every candidate exactly once")
        component_by_sequence: dict[str, str] = {}
        for component in self.pool_components:
            expected_component_id = diversity_component_id(
                component.sequence_ids,
                role="pool",
                fold=self.spec.pool_fold,
            )
            if component.component_id != expected_component_id:
                raise ValueError("pool component ID differs from its role/fold/content hash")
            for sequence_id in component.sequence_ids:
                component_by_sequence[sequence_id] = component.component_id
        if any(
            candidate.diversity_component_id != component_by_sequence[candidate.sequence_id]
            for candidate in self.pool_candidates
        ):
            raise ValueError("candidate component assignments differ from component evidence")
        if component_by_sequence != expected_component_map:
            raise ValueError(
                "candidate component assignments differ from exact sequence clustering"
            )

    def model_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rotation_id": self.spec.rotation_id,
            "base_folds": list(self.spec.base_folds),
            "training_example_ids_sha256": id_stream_sha256(self.base_example_ids),
            "training_sequence_ids_sha256": id_stream_sha256(self.base_sequence_ids),
            "training_example_count": len(self.base_example_ids),
            "training_sequence_count": len(self.base_sequence_ids),
            "model_state": self.base_model.document(),
        }

    def transform_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rotation_id": self.spec.rotation_id,
            "base_folds": list(self.spec.base_folds),
            "training_sequence_ids_sha256": id_stream_sha256(self.base_sequence_ids),
            "transform_state": self.diversity_transform.document(),
        }

    def prediction_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(
            item.document(rotation_id=self.spec.rotation_id) for item in self.pool_predictions
        )

    def novelty_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "schema_version": 1,
                "rotation_id": self.spec.rotation_id,
                "sequence_id": item.sequence_id,
                "max_similarity_hex": item.max_similarity.hex(),
                "nearest_training_sequence_id": item.nearest_training_sequence_id,
                "novelty_hex": item.novelty.hex(),
            }
            for item in self.pool_novelty
        )

    def component_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(component.document(spec=self.spec) for component in self.pool_components)

    def candidate_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(pool_candidate_document(candidate) for candidate in self.pool_candidates)

    def random_candidate_documents(self) -> tuple[dict[str, object], ...]:
        return tuple(
            random_pool_candidate_document(candidate) for candidate in self.random_pool_candidates
        )


def pool_candidate_document(candidate: PoolCandidate) -> dict[str, object]:
    """Serialize exactly the eight fields accepted by the v2 pool selector."""

    if not isinstance(candidate, PoolCandidate):
        raise TypeError("candidate must be a PoolCandidate")
    return {
        "rotation_id": candidate.rotation_id,
        "sequence_id": candidate.sequence_id,
        "sequence": candidate.sequence,
        "objective_probabilities": {
            name: value
            for name, value in zip(OBJECTIVES, candidate.objective_probabilities, strict=True)
        },
        "features": list(candidate.features),
        "novelty": candidate.novelty,
        "diversity_component_id": candidate.diversity_component_id,
        "eligible": candidate.eligible,
    }


def random_pool_candidate_document(candidate: RandomPoolCandidate) -> dict[str, object]:
    """Serialize the minimal five-field view accepted by the random selector."""

    if not isinstance(candidate, RandomPoolCandidate):
        raise TypeError("candidate must be a RandomPoolCandidate")
    return {
        "rotation_id": candidate.rotation_id,
        "sequence_id": candidate.sequence_id,
        "sequence": candidate.sequence,
        "diversity_component_id": candidate.diversity_component_id,
        "eligible": candidate.eligible,
    }


def _validate_label_free(row: LabelFreeContext, *, spec: RotationSpec) -> None:
    if not isinstance(row, LabelFreeContext):
        raise TypeError("acquisition metadata accepts only LabelFreeContext instances")
    if (
        _SHA256.fullmatch(row.example_id) is None
        or row.assay_context_id != row.example_id
        or _SHA256.fullmatch(row.sequence_id) is None
    ):
        raise ValueError("acquisition metadata contains an invalid identity")
    sequence = canonicalize_sequence(row.sequence)
    if sequence != row.sequence or canonical_sequence_id(sequence) != row.sequence_id:
        raise ValueError("acquisition metadata sequence identity is invalid")
    if row.target not in TARGET_GRAM or row.gram != TARGET_GRAM[row.target]:
        raise ValueError("acquisition metadata target/Gram pair is invalid")
    if isinstance(row.fold, bool) or not isinstance(row.fold, int) or row.fold != spec.pool_fold:
        raise ValueError("acquisition metadata escaped the rotation pool fold")


def _sequence_map(
    rows: Iterable[ContextRow | LabelFreeContext],
) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        previous = result.setdefault(row.sequence_id, row.sequence)
        if previous != row.sequence:
            raise ValueError("one sequence ID maps to inconsistent canonical sequences")
    return dict(sorted(result.items()))


def _eligible_sequence_ids(rows: Iterable[LabelFreeContext]) -> tuple[str, ...]:
    grams: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grams[row.sequence_id].add(row.gram)
    return tuple(
        sorted(
            sequence_id
            for sequence_id, observed in grams.items()
            if observed == {"positive", "negative"}
        )
    )


def _validate_capability(capability: PrepareCapability) -> None:
    if not isinstance(capability, PrepareCapability):
        raise TypeError("prepare accepts only an authenticated PrepareCapability")
    spec = capability.spec
    if not isinstance(spec, RotationSpec):
        raise TypeError("prepare capability has an invalid rotation spec")
    base = tuple(capability.base_contexts)
    acquisition = tuple(capability.acquisition_metadata)
    if not base or not acquisition:
        raise ValueError("prepare capability must contain base and acquisition rows")
    if any(not isinstance(row, ContextRow) for row in base):
        raise TypeError("base capability accepts only ContextRow instances")
    if any(
        _SHA256.fullmatch(row.example_id) is None or row.assay_context_id != row.example_id
        for row in base
    ):
        raise ValueError("base capability contains an invalid example identity")
    if any(row.fold not in spec.base_folds for row in base):
        raise ValueError("base context escaped the rotation's exact three folds")
    if {row.fold for row in base} != set(spec.base_folds):
        raise ValueError("base capability must populate each of the exact three base folds")
    for row in acquisition:
        _validate_label_free(row, spec=spec)
    base_ids = tuple(row.example_id for row in base)
    acquisition_ids = tuple(row.example_id for row in acquisition)
    if base_ids != tuple(sorted(set(base_ids))):
        raise ValueError("base contexts must be sorted by unique example ID")
    if acquisition_ids != tuple(sorted(set(acquisition_ids))):
        raise ValueError("acquisition metadata must be sorted by unique example ID")
    if set(base_ids) & set(acquisition_ids):
        raise ValueError("base and acquisition example capabilities overlap")
    if tuple(capability.allowed_base_example_ids) != base_ids:
        raise ValueError("declared base example capability differs from supplied rows")
    if tuple(capability.allowed_acquisition_metadata_example_ids) != acquisition_ids:
        raise ValueError("declared acquisition metadata capability differs from supplied rows")
    support = tuple(capability.acquisition_support_sequence_ids)
    if support != tuple(sorted(set(support))) or any(
        _SHA256.fullmatch(value) is None for value in support
    ):
        raise ValueError("acquisition support IDs must be sorted unique lowercase SHA-256 values")
    observed_support = _eligible_sequence_ids(acquisition)
    if support != observed_support:
        raise ValueError("acquisition support IDs do not match label-free Gram support")


def prepare_rotation(capability: PrepareCapability) -> PreparedRotation:
    """Fit and serialize one rotation without any pool or outer outcome access."""

    _validate_capability(capability)
    spec = capability.spec
    base_rows = tuple(capability.base_contexts)
    pool_rows = tuple(capability.acquisition_metadata)
    base_sequences = _sequence_map(base_rows)
    pool_sequences = _sequence_map(pool_rows)
    support_ids = tuple(capability.acquisition_support_sequence_ids)
    support_sequences = tuple(pool_sequences[sequence_id] for sequence_id in support_ids)

    model = fit_descriptor_logistic(base_rows)
    transform = fit_diversity_transform(base_sequences.values())
    target_matrix, objective_matrix = predict_target_objectives(model, support_sequences)
    feature_matrix = transform_diversity(transform, support_sequences)
    novelty = exact_training_novelty(support_sequences, base_sequences.values())
    component_by_id, raw_components = cluster_diversity_components(
        support_sequences,
        role="pool",
        fold=spec.pool_fold,
    )

    predictions = tuple(
        PoolPrediction(
            sequence_id=sequence_id,
            target_probabilities=tuple(map(float, target_matrix[index])),
            objective_probabilities=tuple(map(float, objective_matrix[index])),
        )
        for index, sequence_id in enumerate(support_ids)
    )
    novelty_by_id = {item.sequence_id: item for item in novelty}
    ordered_novelty = tuple(novelty_by_id[sequence_id] for sequence_id in support_ids)
    components = tuple(
        sorted(
            (
                PoolComponent(
                    component_id=component_by_id[sequence_ids[0]],
                    sequence_ids=tuple(sequence_ids),
                )
                for sequence_ids in raw_components
            ),
            key=lambda item: item.component_id,
        )
    )
    candidates = tuple(
        PoolCandidate(
            rotation_id=spec.rotation_id,
            sequence_id=sequence_id,
            sequence=pool_sequences[sequence_id],
            objective_probabilities=predictions[index].objective_probabilities,
            features=tuple(map(float, feature_matrix[index])),
            novelty=ordered_novelty[index].novelty,
            diversity_component_id=component_by_id[sequence_id],
            eligible=True,
        )
        for index, sequence_id in enumerate(support_ids)
    )
    random_candidates = tuple(
        RandomPoolCandidate(
            rotation_id=candidate.rotation_id,
            sequence_id=candidate.sequence_id,
            sequence=candidate.sequence,
            diversity_component_id=candidate.diversity_component_id,
            eligible=True,
        )
        for candidate in candidates
    )
    return PreparedRotation(
        spec=spec,
        base_model=model,
        diversity_transform=transform,
        base_example_ids=tuple(row.example_id for row in base_rows),
        base_sequence_ids=tuple(base_sequences),
        base_sequences=tuple(base_sequences.values()),
        pool_predictions=predictions,
        pool_novelty=ordered_novelty,
        pool_components=components,
        pool_candidates=candidates,
        random_pool_candidates=random_candidates,
    )


__all__ = [
    "PoolComponent",
    "PoolPrediction",
    "PreparedRotation",
    "id_stream_sha256",
    "pool_candidate_document",
    "prepare_rotation",
    "random_pool_candidate_document",
]

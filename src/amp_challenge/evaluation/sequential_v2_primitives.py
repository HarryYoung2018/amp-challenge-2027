"""Frozen numerical primitives for sequential mixed-acquisition v2.

The sequential replay deliberately does not reuse the legacy replay engine.
These helpers implement only the producer-side, preregistered v2 mathematics:
one pooled contextual descriptor-logistic model, fold-local transparent
diversity/novelty features, sparse observed-sequence rewards, target-level
metrics, and five-outer-fold paired bootstrap intervals.

The independent verifier must not import this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rapidfuzz.distance import Indel

from amp_challenge.constants import STANDARD_AMINO_ACIDS
from amp_challenge.descriptors import compute_descriptors
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import cluster_sequences

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TARGET_GRAMS: tuple[tuple[str, str], ...] = (
    ("acinetobacter_baumannii", "negative"),
    ("enterococcus_faecalis", "positive"),
    ("enterococcus_faecium", "positive"),
    ("escherichia_coli", "negative"),
    ("klebsiella_pneumoniae", "negative"),
    ("pseudomonas_aeruginosa", "negative"),
    ("staphylococcus_aureus", "positive"),
)
TARGETS: tuple[str, ...] = tuple(target for target, _ in TARGET_GRAMS)
TARGET_GRAM: Mapping[str, str] = dict(TARGET_GRAMS)
POSITIVE_TARGETS: tuple[str, ...] = tuple(
    target for target, gram in TARGET_GRAMS if gram == "positive"
)
NEGATIVE_TARGETS: tuple[str, ...] = tuple(
    target for target, gram in TARGET_GRAMS if gram == "negative"
)
OBJECTIVES: tuple[str, ...] = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "length",
    "molecular_weight_da",
    "net_charge",
    "charge_density",
    "isoelectric_point",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "hydrophobic_fraction",
    "aromatic_fraction",
    "basic_fraction",
    "acidic_fraction",
    "shannon_entropy",
    "max_residue_fraction",
)
FEATURE_NAMES: tuple[str, ...] = DESCRIPTOR_NAMES + tuple(
    f"residue_fraction_{residue}" for residue in STANDARD_AMINO_ACIDS
)
FEATURE_COUNT = 33
LOGISTIC_L2 = 0.10
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1e-9
LOGISTIC_PRIOR_STRENGTH = 2.0
PROBABILITY_CLIP = 1e-6
NLL_CLIP = 1e-15
DIVERSITY_IDENTITY_THRESHOLD = 0.70
DIVERSITY_ALGORITHM = "global_alignment_identity_v1_deterministic_single_link"
DIVERSITY_COMPONENT_DOMAIN = "amp_challenge.sequential_mixed_acquisition_v2.diversity_component"
DIVERSITY_COMPONENT_PREFIX = "seqv2-div70:"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_905


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


@dataclass(frozen=True, slots=True)
class ContextRow:
    """One accepted context example in the frozen union Gate-1 panel."""

    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    target: str
    gram: str
    fold: int
    label: int
    source_observations: int

    def __post_init__(self) -> None:
        sequence = canonicalize_sequence(self.sequence)
        if sequence != self.sequence:
            raise ValueError("context sequence must already be canonical")
        if canonical_sequence_id(sequence) != self.sequence_id:
            raise ValueError("context sequence_id does not match sequence")
        if not self.example_id or not self.assay_context_id:
            raise ValueError("context identifiers must be non-empty")
        if self.target not in TARGET_GRAM:
            raise ValueError(f"unexpected canonical target: {self.target!r}")
        if self.gram != TARGET_GRAM[self.target]:
            raise ValueError("context Gram class disagrees with the frozen target map")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or not 0 <= self.fold < 5:
            raise ValueError("context fold must be an integer in [0, 4]")
        if (
            isinstance(self.label, bool)
            or not isinstance(self.label, int)
            or self.label not in (0, 1)
        ):
            raise ValueError("context label must be integer zero or one")
        if (
            isinstance(self.source_observations, bool)
            or not isinstance(self.source_observations, int)
            or self.source_observations < 1
        ):
            raise ValueError("source_observations must be a positive integer")


@dataclass(frozen=True, slots=True)
class DescriptorLogisticState:
    """Complete deterministic state of one pooled contextual logistic fit."""

    strains: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficient: tuple[float, ...] | None
    constant_probability: float | None
    iterations: int
    training_contexts: int
    positive_contexts: int
    converged: bool = True

    def __post_init__(self) -> None:
        if self.strains != tuple(sorted(set(self.strains))):
            raise ValueError("model strains must be sorted and unique")
        if any(target not in TARGET_GRAM for target in self.strains):
            raise ValueError("model state contains an unexpected strain")
        if len(self.feature_mean) != FEATURE_COUNT or len(self.feature_scale) != FEATURE_COUNT:
            raise ValueError("model feature state must have exactly 33 values")
        if any(not math.isfinite(value) for value in self.feature_mean):
            raise ValueError("model feature means must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.feature_scale):
            raise ValueError("model feature scales must be finite and positive")
        expected_coefficients = 1 + FEATURE_COUNT + len(self.strains) + 1 + 3
        if self.coefficient is None:
            if self.constant_probability is None:
                raise ValueError("constant model state lacks a probability")
        else:
            if self.constant_probability is not None:
                raise ValueError("nonconstant model state has a constant probability")
            if len(self.coefficient) != expected_coefficients:
                raise ValueError("model coefficient dimension is inconsistent with strains")
            if any(not math.isfinite(value) for value in self.coefficient):
                raise ValueError("model coefficients must be finite")
        if self.constant_probability is not None and not 0.0 < self.constant_probability < 1.0:
            raise ValueError("constant probability must be strictly between zero and one")
        if not self.converged:
            raise ValueError("serialized model state must be converged")
        if self.iterations < 0 or self.iterations > LOGISTIC_MAX_ITERATIONS:
            raise ValueError("model iteration count is outside the frozen bounds")
        if self.training_contexts < 1 or not 0 <= self.positive_contexts <= self.training_contexts:
            raise ValueError("model training census is invalid")

    def document(self) -> dict[str, object]:
        """Return a path-free, bit-exact JSON-friendly state document."""

        return {
            "schema_version": 1,
            "artifact": "sequential_v2_pooled_descriptor_logistic_state",
            "converged": self.converged,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean_hex": [value.hex() for value in self.feature_mean],
            "feature_scale_hex": [value.hex() for value in self.feature_scale],
            "strains": list(self.strains),
            "coefficient_hex": (
                None if self.coefficient is None else [value.hex() for value in self.coefficient]
            ),
            "constant_probability_hex": (
                None if self.constant_probability is None else self.constant_probability.hex()
            ),
            "iterations": self.iterations,
            "training_contexts": self.training_contexts,
            "positive_contexts": self.positive_contexts,
            "hyperparameters": {
                "l2_hex": LOGISTIC_L2.hex(),
                "max_iterations": LOGISTIC_MAX_ITERATIONS,
                "prior_strength_hex": LOGISTIC_PRIOR_STRENGTH.hex(),
                "probability_clip_hex": PROBABILITY_CLIP.hex(),
                "tolerance_hex": LOGISTIC_TOLERANCE.hex(),
            },
        }


@dataclass(frozen=True, slots=True)
class DiversityTransform:
    """Equal-unique-sequence fold-local diversity transform."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fit_sequence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != FEATURE_COUNT or len(self.scale) != FEATURE_COUNT:
            raise ValueError("diversity transform must have 33 coordinates")
        if any(not math.isfinite(value) for value in self.mean):
            raise ValueError("diversity means must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scale):
            raise ValueError("diversity scales must be finite and positive")
        if not self.fit_sequence_ids or self.fit_sequence_ids != tuple(
            sorted(set(self.fit_sequence_ids))
        ):
            raise ValueError("fit sequence IDs must be non-empty, sorted, and unique")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact": "sequential_v2_fold_local_diversity_transform",
            "feature_names": list(FEATURE_NAMES),
            "fit_sequence_count": len(self.fit_sequence_ids),
            "fit_sequence_ids_sha256": hashlib.sha256(
                "".join(f"{value}\n" for value in self.fit_sequence_ids).encode("ascii")
            ).hexdigest(),
            "mean_hex": [value.hex() for value in self.mean],
            "scale_hex": [value.hex() for value in self.scale],
            "scale_floor_rule": "population_sd_below_1e-12_replaced_by_one",
            "row_normalization": "euclidean_l2_zero_vector_stays_zero",
        }


@dataclass(frozen=True, slots=True)
class ObservedSequenceOutcome:
    sequence_id: str
    sequence: str
    values: tuple[float | None, float | None, float | None]
    context_counts: tuple[int, int, int]

    @property
    def support_eligible(self) -> bool:
        return all(count > 0 for count in self.context_counts)

    @property
    def scalar_reward(self) -> float:
        if not self.support_eligible or any(value is None for value in self.values):
            raise ValueError("scalar reward requires support for every observed objective")
        return float(sum(float(value) for value in self.values) / len(OBJECTIVES))


@dataclass(frozen=True, slots=True)
class NoveltyEvidence:
    sequence_id: str
    max_similarity: float
    nearest_training_sequence_id: str
    novelty: float


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    point: float
    lower: float
    upper: float
    replicates: int
    seed: int
    resample_indices_sha256: str


def raw_sequence_features(sequence: str) -> FloatArray:
    """Return the frozen 13-descriptor plus 20-composition vector."""

    canonical = canonicalize_sequence(sequence)
    values = compute_descriptors(canonical).as_dict()
    descriptor_values = tuple(float(values[name]) for name in DESCRIPTOR_NAMES)
    composition = tuple(
        canonical.count(residue) / len(canonical) for residue in STANDARD_AMINO_ACIDS
    )
    result = np.asarray(descriptor_values + composition, dtype=np.float64)
    if result.shape != (FEATURE_COUNT,) or not bool(np.all(np.isfinite(result))):
        raise ValueError("sequence descriptor vector is invalid")
    return result


def _sigmoid(values: FloatArray) -> FloatArray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _design_matrix(
    sequences: Sequence[str],
    targets: Sequence[str],
    grams: Sequence[str],
    *,
    strains: Sequence[str],
    mean: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    if not (len(sequences) == len(targets) == len(grams)):
        raise ValueError("model query columns must have equal length")
    if not sequences:
        return np.empty((0, 1 + FEATURE_COUNT + len(strains) + 1 + 3), dtype=np.float64)
    canonical = tuple(canonicalize_sequence(sequence) for sequence in sequences)
    for target, gram in zip(targets, grams, strict=True):
        if target not in TARGET_GRAM or gram != TARGET_GRAM[target]:
            raise ValueError("model query target/Gram pair is outside the frozen panel")
    continuous = np.asarray([raw_sequence_features(sequence) for sequence in canonical])
    standardized = (continuous - mean) / scale
    strain_index = {strain: index for index, strain in enumerate(strains)}
    strain_matrix = np.zeros((len(sequences), len(strains) + 1), dtype=np.float64)
    for row, target in enumerate(targets):
        strain_matrix[row, strain_index.get(target, len(strains))] = 1.0
    gram_matrix = np.zeros((len(sequences), 3), dtype=np.float64)
    gram_index = {"positive": 0, "negative": 1, "unknown": 2}
    for row, gram in enumerate(grams):
        gram_matrix[row, gram_index[gram]] = 1.0
    return np.concatenate(
        (
            np.ones((len(sequences), 1), dtype=np.float64),
            standardized,
            strain_matrix,
            gram_matrix,
        ),
        axis=1,
    )


def fit_descriptor_logistic(contexts: Sequence[ContextRow]) -> DescriptorLogisticState:
    """Fit one pooled contextual descriptor logistic or fail on nonconvergence."""

    unsorted_rows = tuple(contexts)
    if any(not isinstance(row, ContextRow) for row in unsorted_rows):
        raise TypeError("descriptor logistic accepts only ContextRow instances")
    rows = tuple(sorted(unsorted_rows, key=lambda row: row.example_id))
    if not rows:
        raise ValueError("descriptor logistic requires at least one context")
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("descriptor logistic contexts must have unique example IDs")
    strains = tuple(sorted({row.target for row in rows}))
    continuous = np.asarray([raw_sequence_features(row.sequence) for row in rows])
    mean = np.mean(continuous, axis=0, dtype=np.float64)
    scale = np.std(continuous, axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    labels = np.asarray([row.label for row in rows], dtype=np.float64)
    prior = float(
        (np.sum(labels, dtype=np.float64) + 0.5 * LOGISTIC_PRIOR_STRENGTH)
        / (len(rows) + LOGISTIC_PRIOR_STRENGTH)
    )
    if bool(np.all(labels == labels[0])):
        return DescriptorLogisticState(
            strains=strains,
            feature_mean=tuple(map(float, mean)),
            feature_scale=tuple(map(float, scale)),
            coefficient=None,
            constant_probability=prior,
            iterations=0,
            training_contexts=len(rows),
            positive_contexts=int(np.sum(labels)),
        )

    matrix = _design_matrix(
        tuple(row.sequence for row in rows),
        tuple(row.target for row in rows),
        tuple(row.gram for row in rows),
        strains=strains,
        mean=mean,
        scale=scale,
    )
    coefficient = np.zeros(matrix.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(matrix.shape[1], LOGISTIC_L2, dtype=np.float64)
    penalty[0] = 0.0
    converged_at: int | None = None
    for iteration in range(1, LOGISTIC_MAX_ITERATIONS + 1):
        probability = _sigmoid(matrix @ coefficient)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = (matrix.T @ (probability - labels)) / len(rows) + penalty * coefficient
        hessian = (matrix.T * variance) @ matrix / len(rows)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not bool(np.all(np.isfinite(step))):
            raise RuntimeError("descriptor logistic produced a non-finite Newton step")
        coefficient -= step
        if float(np.max(np.abs(step))) <= LOGISTIC_TOLERANCE:
            converged_at = iteration
            break
    if converged_at is None:
        raise RuntimeError("descriptor logistic did not converge within the frozen 100 iterations")
    return DescriptorLogisticState(
        strains=strains,
        feature_mean=tuple(map(float, mean)),
        feature_scale=tuple(map(float, scale)),
        coefficient=tuple(map(float, coefficient)),
        constant_probability=None,
        iterations=converged_at,
        training_contexts=len(rows),
        positive_contexts=int(np.sum(labels)),
    )


def predict_contexts(
    state: DescriptorLogisticState,
    sequences: Sequence[str],
    targets: Sequence[str],
    grams: Sequence[str],
) -> FloatArray:
    """Predict finite raw logistic probabilities for frozen target contexts."""

    if not (len(sequences) == len(targets) == len(grams)):
        raise ValueError("prediction query columns must have equal length")
    if not sequences:
        return np.empty(0, dtype=np.float64)
    canonical = tuple(canonicalize_sequence(sequence) for sequence in sequences)
    for target, gram in zip(targets, grams, strict=True):
        if target not in TARGET_GRAM or gram != TARGET_GRAM[target]:
            raise ValueError("model query target/Gram pair is outside the frozen panel")
    if state.constant_probability is not None:
        result = np.full(len(sequences), state.constant_probability, dtype=np.float64)
    else:
        assert state.coefficient is not None
        matrix = _design_matrix(
            canonical,
            targets,
            grams,
            strains=state.strains,
            mean=np.asarray(state.feature_mean, dtype=np.float64),
            scale=np.asarray(state.feature_scale, dtype=np.float64),
        )
        result = _sigmoid(matrix @ np.asarray(state.coefficient, dtype=np.float64))
    result = np.clip(result, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    if result.shape != (len(sequences),) or not bool(np.all(np.isfinite(result))):
        raise ValueError("descriptor logistic produced invalid probabilities")
    return result


def predict_target_objectives(
    state: DescriptorLogisticState,
    sequences: Sequence[str],
) -> tuple[FloatArray, FloatArray]:
    """Predict the exact seven-target panel and three raw mean objectives."""

    canonical = tuple(canonicalize_sequence(sequence) for sequence in sequences)
    flat_sequences = tuple(sequence for sequence in canonical for _ in TARGETS)
    flat_targets = TARGETS * len(canonical)
    flat_grams = tuple(TARGET_GRAM[target] for _ in canonical for target in TARGETS)
    probabilities = predict_contexts(state, flat_sequences, flat_targets, flat_grams).reshape(
        len(canonical), len(TARGETS)
    )
    target_index = {target: index for index, target in enumerate(TARGETS)}
    positive = np.asarray([target_index[target] for target in POSITIVE_TARGETS], dtype=np.int64)
    negative = np.asarray([target_index[target] for target in NEGATIVE_TARGETS], dtype=np.int64)
    objectives = np.column_stack(
        (
            np.mean(probabilities, axis=1, dtype=np.float64),
            np.mean(probabilities[:, positive], axis=1, dtype=np.float64),
            np.mean(probabilities[:, negative], axis=1, dtype=np.float64),
        )
    )
    return probabilities, objectives


def aggregate_observed_sequence_outcomes(
    contexts: Iterable[ContextRow],
) -> tuple[ObservedSequenceOutcome, ...]:
    """Aggregate sparse context labels without weighting source observations."""

    grouped: dict[str, list[ContextRow]] = defaultdict(list)
    for row in contexts:
        grouped[row.sequence_id].append(row)
    results: list[ObservedSequenceOutcome] = []
    for sequence_id in sorted(grouped):
        rows = grouped[sequence_id]
        sequences = {row.sequence for row in rows}
        if len(sequences) != 1:
            raise ValueError("one sequence ID maps to inconsistent sequences")
        subsets = (
            rows,
            [row for row in rows if row.gram == "positive"],
            [row for row in rows if row.gram == "negative"],
        )
        values = tuple(
            None if not subset else float(np.mean([row.label for row in subset], dtype=np.float64))
            for subset in subsets
        )
        results.append(
            ObservedSequenceOutcome(
                sequence_id=sequence_id,
                sequence=next(iter(sequences)),
                values=values,
                context_counts=tuple(len(subset) for subset in subsets),
            )
        )
    return tuple(results)


def fit_diversity_transform(sequences: Iterable[str]) -> DiversityTransform:
    """Fit equal-weight population statistics over unique base-fold sequences."""

    canonical = tuple(sorted({canonicalize_sequence(sequence) for sequence in sequences}))
    if not canonical:
        raise ValueError("diversity transform requires at least one base sequence")
    matrix = np.asarray([raw_sequence_features(sequence) for sequence in canonical])
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    scale = np.std(matrix, axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    return DiversityTransform(
        mean=tuple(map(float, mean)),
        scale=tuple(map(float, scale)),
        fit_sequence_ids=tuple(sorted(canonical_sequence_id(sequence) for sequence in canonical)),
    )


def transform_diversity(
    transform: DiversityTransform,
    sequences: Sequence[str],
) -> FloatArray:
    """Apply z-scoring and row L2 normalization; zero rows remain zero."""

    canonical = tuple(canonicalize_sequence(sequence) for sequence in sequences)
    if not canonical:
        return np.empty((0, FEATURE_COUNT), dtype=np.float64)
    matrix = np.asarray([raw_sequence_features(sequence) for sequence in canonical])
    standardized = (matrix - np.asarray(transform.mean, dtype=np.float64)) / np.asarray(
        transform.scale, dtype=np.float64
    )
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    result = np.divide(
        standardized,
        norms,
        out=np.zeros_like(standardized, dtype=np.float64),
        where=norms > 0.0,
    )
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("diversity transform produced a non-finite coordinate")
    return result


def exact_training_novelty(
    sequences: Sequence[str],
    training_sequences: Iterable[str],
) -> tuple[NoveltyEvidence, ...]:
    """Compute exact Indel maxima with canonical training-ID tie resolution."""

    queries = tuple(canonicalize_sequence(sequence) for sequence in sequences)
    if len(set(queries)) != len(queries):
        raise ValueError("novelty queries must be unique")
    training = tuple(sorted({canonicalize_sequence(sequence) for sequence in training_sequences}))
    if not training:
        raise ValueError("novelty requires at least one unique training sequence")
    choices = tuple((canonical_sequence_id(sequence), sequence) for sequence in training)
    output: list[NoveltyEvidence] = []
    for query in queries:
        scored = tuple(
            (float(Indel.normalized_similarity(query, sequence)), sequence_id)
            for sequence_id, sequence in choices
        )
        maximum = max(score for score, _ in scored)
        nearest = min(sequence_id for score, sequence_id in scored if score == maximum)
        output.append(
            NoveltyEvidence(
                sequence_id=canonical_sequence_id(query),
                max_similarity=maximum,
                nearest_training_sequence_id=nearest,
                novelty=1.0 - maximum,
            )
        )
    return tuple(output)


def diversity_component_id(sequence_ids: Iterable[str], *, role: str, fold: int) -> str:
    """Return the frozen content-addressed 70%-identity component ID."""

    raw_identifiers = tuple(sequence_ids)
    identifiers = tuple(sorted(raw_identifiers))
    if not identifiers or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in identifiers
    ):
        raise ValueError("component sequence IDs must be canonical lowercase SHA-256 strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("component sequence IDs must be unique")
    if role not in {"pool", "outer"}:
        raise ValueError("component role must be 'pool' or 'outer'")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("component fold must be an integer in [0, 4]")
    payload = json.dumps(
        {
            "algorithm": DIVERSITY_ALGORITHM,
            "domain": DIVERSITY_COMPONENT_DOMAIN,
            "fold": fold,
            "identity_threshold_hex": DIVERSITY_IDENTITY_THRESHOLD.hex(),
            "role": role,
            "schema_version": 1,
            "sequence_ids": list(identifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return DIVERSITY_COMPONENT_PREFIX + hashlib.sha256(payload).hexdigest()


def cluster_diversity_components(
    sequences: Iterable[str],
    *,
    role: str,
    fold: int,
) -> tuple[dict[str, str], tuple[tuple[str, ...], ...]]:
    """Cluster only the supplied support panel under the frozen inclusive rule."""

    canonical = tuple(sorted({canonicalize_sequence(sequence) for sequence in sequences}))
    if not canonical:
        raise ValueError("diversity clustering requires at least one support sequence")
    components = cluster_sequences(canonical, identity_threshold=DIVERSITY_IDENTITY_THRESHOLD)
    mapping: dict[str, str] = {}
    component_ids: list[tuple[str, ...]] = []
    for component in components:
        identifiers = tuple(sorted(canonical_sequence_id(sequence) for sequence in component))
        component_id = diversity_component_id(identifiers, role=role, fold=fold)
        for sequence_id in identifiers:
            if sequence_id in mapping:
                raise AssertionError("sequence entered more than one diversity component")
            mapping[sequence_id] = component_id
        component_ids.append(identifiers)
    if len(mapping) != len(canonical):
        raise AssertionError("diversity clustering did not cover the support panel")
    return dict(sorted(mapping.items())), tuple(sorted(component_ids))


def _roc_auc(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    differences = positives[:, None] - negatives[None, :]
    return float((np.sum(differences > 0) + 0.5 * np.sum(differences == 0)) / differences.size)


def _average_precision(labels: IntArray, probabilities: FloatArray) -> float | None:
    positives = int(np.sum(labels))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    ordered_probabilities = probabilities[order]
    ordered_labels = labels[order]
    true_positive = 0
    false_positive = 0
    result = 0.0
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and ordered_probabilities[end] == ordered_probabilities[start]:
            end += 1
        group = ordered_labels[start:end]
        new_positive = int(np.sum(group))
        true_positive += new_positive
        false_positive += len(group) - new_positive
        result += (new_positive / positives) * (true_positive / (true_positive + false_positive))
        start = end
    return float(result)


def _ece10(labels: IntArray, probabilities: FloatArray) -> float:
    bins = np.minimum(np.floor(probabilities * 10.0).astype(np.int64), 9)
    result = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        count = int(np.sum(selected))
        if count:
            result += (count / len(labels)) * abs(
                float(np.mean(probabilities[selected], dtype=np.float64))
                - float(np.mean(labels[selected], dtype=np.float64))
            )
    return float(result)


def summarize_target_metrics(
    targets: Sequence[str],
    labels: Sequence[int] | IntArray,
    probabilities: Sequence[float] | FloatArray,
) -> dict[str, object]:
    """Return explicit per-target and defined-target macro metrics."""

    if not (len(targets) == len(labels) == len(probabilities)):
        raise ValueError("metric target, label, and probability columns must align")
    raw_label_array = np.asarray(labels)
    raw_probability_array = np.asarray(probabilities)
    if raw_label_array.ndim != 1 or raw_probability_array.ndim != 1 or not len(raw_label_array):
        raise ValueError("metric inputs must be non-empty one-dimensional columns")
    if raw_label_array.dtype.kind not in {"i", "u"}:
        raise ValueError("metric labels must be integer binary values")
    if raw_probability_array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("metric probabilities must be numeric")
    label_array = raw_label_array.astype(np.int64, copy=False)
    probability_array = raw_probability_array.astype(np.float64, copy=False)
    if bool(np.any((label_array != 0) & (label_array != 1))):
        raise ValueError("metric labels must be binary")
    if bool(np.any(~np.isfinite(probability_array))) or bool(
        np.any((probability_array < 0.0) | (probability_array > 1.0))
    ):
        raise ValueError("metric probabilities must be finite and in [0, 1]")
    if any(target not in TARGET_GRAM for target in targets):
        raise ValueError("metric targets must belong to the frozen seven-target panel")

    by_target: dict[str, dict[str, object]] = {}
    metric_names = ("brier", "nll", "roc_auc", "average_precision", "ece_10")
    defined: dict[str, list[str]] = {name: [] for name in metric_names}
    for target in TARGETS:
        mask = np.asarray([value == target for value in targets], dtype=bool)
        target_labels = label_array[mask]
        target_probabilities = probability_array[mask]
        if not len(target_labels):
            values: dict[str, object] = {
                "n": 0,
                "positives": 0,
                "negatives": 0,
                **{name: None for name in metric_names},
                "undefined_reason": "no_outer_contexts",
            }
        else:
            clipped = np.clip(target_probabilities, NLL_CLIP, 1.0 - NLL_CLIP)
            brier = float(np.mean((target_probabilities - target_labels) ** 2, dtype=np.float64))
            nll = float(
                -np.mean(
                    target_labels * np.log(clipped) + (1 - target_labels) * np.log1p(-clipped),
                    dtype=np.float64,
                )
            )
            roc_auc = _roc_auc(target_labels, target_probabilities)
            average_precision = _average_precision(target_labels, target_probabilities)
            ece = _ece10(target_labels, target_probabilities)
            values = {
                "n": len(target_labels),
                "positives": int(np.sum(target_labels)),
                "negatives": int(len(target_labels) - np.sum(target_labels)),
                "brier": brier,
                "nll": nll,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "ece_10": ece,
                "undefined_reason": (
                    None
                    if roc_auc is not None and average_precision is not None
                    else {
                        "roc_auc": (
                            None if roc_auc is not None else "requires_both_binary_classes"
                        ),
                        "average_precision": (
                            None
                            if average_precision is not None
                            else "requires_at_least_one_positive"
                        ),
                    }
                ),
            }
        by_target[target] = values
        for name in metric_names:
            if values[name] is not None:
                defined[name].append(target)

    macro: dict[str, float] = {}
    for name in metric_names:
        names = defined[name]
        if not names:
            raise ValueError(f"no canonical target defines metric {name}")
        macro[name] = float(
            np.mean([float(by_target[target][name]) for target in names], dtype=np.float64)
        )
    return {
        "by_target": by_target,
        "macro": macro,
        "defined_targets": {name: names for name, names in defined.items()},
        "defined_target_counts": {name: len(names) for name, names in defined.items()},
        "target_order": list(TARGETS),
        "nll_probability_clip_hex": NLL_CLIP.hex(),
        "ece_bins": "[0,.1),[.1,.2),...,[.8,.9),[.9,1]",
    }


def observed_batch_reward(
    outcomes: Mapping[str, ObservedSequenceOutcome],
    selected_sequence_ids: Sequence[str],
) -> float:
    """Return equal-candidate mean of equal-three-objective observed rewards."""

    selected = tuple(selected_sequence_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected sequence IDs must be non-empty and unique")
    try:
        values = [outcomes[sequence_id].scalar_reward for sequence_id in selected]
    except KeyError as error:
        raise ValueError(f"selected sequence outcome is missing: {error.args[0]}") from error
    return float(np.mean(values, dtype=np.float64))


def mean_pairwise_feature_cosine_distance(
    features: Sequence[Sequence[float]] | FloatArray,
) -> float:
    """Return the frozen strict-upper-triangle feature cosine distance.

    Similarity involving a zero vector is defined as zero.  Nonzero cosine
    similarities are clipped to ``[-1, 1]`` before subtracting from one, which
    protects the exact distance range from binary64 roundoff.
    """

    raw = np.asarray(features)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ValueError("pairwise feature distance requires at least two nonempty vectors")
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("pairwise feature vectors must be numeric")
    matrix = raw.astype(np.float64, copy=False)
    if bool(np.any(~np.isfinite(matrix))):
        raise ValueError("pairwise feature vectors must be finite")

    norms = np.linalg.norm(matrix, axis=1)
    products = matrix @ matrix.T
    denominators = norms[:, None] * norms[None, :]
    similarities = np.divide(
        products,
        denominators,
        out=np.zeros_like(products, dtype=np.float64),
        where=denominators > 0.0,
    )
    similarities = np.clip(similarities, -1.0, 1.0)
    upper = np.triu_indices(matrix.shape[0], k=1)
    return float(np.mean(1.0 - similarities[upper], dtype=np.float64))


def _outer_fold_array(values: Mapping[int, float], *, label: str) -> FloatArray:
    if (
        not isinstance(values, Mapping)
        or any(isinstance(fold, bool) or not isinstance(fold, int) for fold in values)
        or set(values) != set(range(5))
    ):
        raise ValueError(f"{label} must bind exactly outer folds 0 through 4")
    return np.asarray(
        [_finite_number(values[fold], label=f"{label} fold {fold}") for fold in range(5)],
        dtype=np.float64,
    )


def frozen_bootstrap_indices() -> IntArray:
    """Return the common frozen 10,000-by-5 paired outer-fold resample matrix."""

    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(
        0,
        5,
        size=(BOOTSTRAP_REPLICATES, 5),
        dtype=np.int64,
    )
    indices.setflags(write=False)
    return indices


def bootstrap_indices_sha256(indices: IntArray | None = None) -> str:
    """Hash resample indices as C-order little-endian signed 64-bit bytes."""

    matrix = frozen_bootstrap_indices() if indices is None else np.asarray(indices)
    if matrix.shape != (BOOTSTRAP_REPLICATES, 5) or matrix.dtype != np.dtype(np.int64):
        raise ValueError("bootstrap index matrix shape or dtype changed")
    if bool(np.any((matrix < 0) | (matrix > 4))):
        raise ValueError("bootstrap index matrix contains an invalid outer-fold index")
    return hashlib.sha256(matrix.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def paired_outer_bootstrap(
    candidate: Mapping[int, float],
    control: Mapping[int, float],
) -> PairedBootstrapInterval:
    """Bootstrap paired differences over exactly five complete outer units."""

    left = _outer_fold_array(candidate, label="candidate outer values")
    right = _outer_fold_array(control, label="control outer values")
    differences = left - right
    indices = frozen_bootstrap_indices()
    samples = np.mean(differences[indices], axis=1, dtype=np.float64)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return PairedBootstrapInterval(
        point=float(np.mean(differences, dtype=np.float64)),
        lower=float(lower),
        upper=float(upper),
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        resample_indices_sha256=bootstrap_indices_sha256(indices),
    )


def _rotation_value_map(values: Mapping[str, float], *, label: str) -> dict[str, float]:
    expected = {rotation_id(outer_fold, pool_fold) for outer_fold, pool_fold in ordered_rotations()}
    if not isinstance(values, Mapping) or set(values) != expected:
        raise ValueError(f"{label} must bind exactly the twenty frozen rotations")
    return {key: _finite_number(values[key], label=f"{label} {key}") for key in sorted(expected)}


def aggregate_deterministic_outer_units(values: Mapping[str, float]) -> dict[int, float]:
    """Average four acquisition rotations within each of five outer folds."""

    parsed = _rotation_value_map(values, label="deterministic rotation values")
    return {
        outer_fold: float(
            np.mean(
                [
                    parsed[rotation_id(outer_fold, pool_fold)]
                    for pool_fold in range(5)
                    if pool_fold != outer_fold
                ],
                dtype=np.float64,
            )
        )
        for outer_fold in range(5)
    }


def aggregate_random_outer_units(
    values: Mapping[tuple[str, int], float],
) -> dict[int, float]:
    """Average five seeds per rotation, then four rotations per outer fold."""

    seeds = (17, 42, 91, 137, 271)
    expected = {
        (rotation_id(outer_fold, pool_fold), seed)
        for outer_fold, pool_fold in ordered_rotations()
        for seed in seeds
    }
    if (
        not isinstance(values, Mapping)
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            for key in values
        )
        or set(values) != expected
    ):
        raise ValueError("random values must bind exactly five seeds for twenty rotations")
    rotation_means = {
        key: float(
            np.mean(
                [
                    _finite_number(values[(key, seed)], label=f"random value {key} seed {seed}")
                    for seed in seeds
                ],
                dtype=np.float64,
            )
        )
        for key in sorted({rotation for rotation, _ in expected})
    }
    return aggregate_deterministic_outer_units(rotation_means)


def context_rows_by_fold(contexts: Iterable[ContextRow]) -> dict[int, tuple[ContextRow, ...]]:
    """Return an exact five-fold context partition in canonical example order."""

    grouped: dict[int, list[ContextRow]] = {fold: [] for fold in range(5)}
    seen: set[str] = set()
    for row in contexts:
        if row.example_id in seen:
            raise ValueError("context rows contain a duplicate example ID")
        seen.add(row.example_id)
        grouped[row.fold].append(row)
    if not seen or any(not rows for rows in grouped.values()):
        raise ValueError("context panel must populate every frozen fold")
    return {
        fold: tuple(sorted(rows, key=lambda item: item.example_id))
        for fold, rows in grouped.items()
    }


def rotation_id(outer_fold: int, acquisition_fold: int) -> str:
    """Return the canonical ordered train/pool/test rotation identifier."""

    if (
        isinstance(outer_fold, bool)
        or isinstance(acquisition_fold, bool)
        or not isinstance(outer_fold, int)
        or not isinstance(acquisition_fold, int)
        or not 0 <= outer_fold < 5
        or not 0 <= acquisition_fold < 5
        or outer_fold == acquisition_fold
    ):
        raise ValueError("rotation folds must be distinct integers in [0, 4]")
    return f"outer-{outer_fold}.pool-{acquisition_fold}"


def ordered_rotations() -> tuple[tuple[int, int], ...]:
    return tuple(
        (outer_fold, acquisition_fold)
        for outer_fold in range(5)
        for acquisition_fold in range(5)
        if acquisition_fold != outer_fold
    )

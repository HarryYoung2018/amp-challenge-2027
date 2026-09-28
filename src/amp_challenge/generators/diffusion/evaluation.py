"""Leakage-safe evaluation and count controls for native diffusion v0.

The module is intentionally NumPy-only at the model boundary.  It constructs
the frozen corruption ledger, scores finite categorical predictions, fits the
three train-only count baselines, performs union-component paired bootstrap,
and audits raw candidates without importing an oracle or ensemble.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import process
from rapidfuzz.distance.Indel import normalized_similarity as levenshtein_ratio

from amp_challenge.descriptors import compute_descriptors
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import global_sequence_identity

from .categorical import AbsorbingDiffusion, CosineMaskSchedule, PeptideVocabulary
from .contract import CONFIG_SHA256
from .data import (
    ACCEPTED_TRAINING_PROJECTION_SHA256,
    VALIDATION_FOLD,
    DiffusionCorpusRow,
    NativeDiffusionCorpus,
    TrainingDistribution,
    TrainingRow,
    namespaced_seed,
)
from .sampling import (
    SAMPLING_BATCH_SEQUENCES,
    canonical_length_plan,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EVALUATION_LEVELS = 64
EVALUATION_REPLICATES = 4
EVALUATION_SEED = 20260904
EVALUATION_BATCH_SEQUENCES = 256
EXPECTED_TRAIN_SEQUENCES = 914
EXPECTED_VALIDATION_SEQUENCES = 199
EXPECTED_VALIDATION_HOMOLOGY_COMPONENTS = 126
EXPECTED_VALIDATION_UNION_COMPONENTS = 57
EXPECTED_VALIDATION_CASES = 50_944
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260904
CALIBRATION_BINS = 15
TOP_REFERENCE_RATIO = 0.80
IDENTITY_THRESHOLD = 0.70
GENERATOR_CONTROL_METHODS = (
    "component_weighted_unigram",
    "component_weighted_forward_markov",
)
DESCRIPTOR_FEATURE_NAMES = (
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
DESCRIPTOR_SCALING = "training_component_weighted_population_zscore_floor_1e-12"
DESCRIPTOR_REFERENCE_POPULATION = "training_projection_component_weighted"
DESCRIPTOR_ENERGY_ESTIMATOR = "weighted_v_statistic_euclidean"
DESCRIPTOR_SCALE_FLOOR = 1e-12
DESCRIPTOR_DISTANCE_CHUNK_SIZE = 256
_CLUSTER_PREFILTER_MARGIN = 1e-12

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_CASE_DOMAIN = b"amp-challenge/native-categorical-diffusion/validation-case/v1\0"
_TRAIN_IDS_DOMAIN = b"amp-challenge/native-categorical-diffusion/baseline-train-ids/v1\0"
_SEQUENCE_COLLECTION_DOMAIN = b"amp-challenge/native-categorical-diffusion/sequence-collection/v1\0"
_CONTROL_DRAW_DOMAIN = b"amp-challenge/native-categorical-diffusion/control-draw/v1\0"
_CONTROL_LOGICAL_DOMAIN = b"amp-challenge/native-categorical-diffusion/control-logical/v1\0"
_PROBABILITY_TABLE_DOMAIN = b"amp-challenge/native-categorical-diffusion/count-tables/v1\0"

LogitProvider = Callable[
    [NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64]],
    NDArray[np.floating],
]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _method(value: object) -> str:
    if not isinstance(value, str) or _METHOD_RE.fullmatch(value) is None:
        raise ValueError("method must be a lowercase manifest-safe identifier")
    return value


def _finite(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be a canonical Python float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _bounded_float(value: object, *, label: str, lower: float, upper: float) -> float:
    result = _finite(value, label=label)
    if not lower <= result <= upper:
        raise ValueError(f"{label} must lie in [{lower}, {upper}]")
    return result


def _framed_update(digest, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _case_id(sequence_id: str, level: int, replicate: int, row_seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(_CASE_DOMAIN)
    for value in (
        CONFIG_SHA256.encode("ascii"),
        sequence_id.encode("ascii"),
        level.to_bytes(2, "big"),
        replicate.to_bytes(2, "big"),
        row_seed.to_bytes(8, "big"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """One immutable sequence, level, and stateless corruption replicate."""

    case_id: str
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    level: int
    replicate: int
    row_seed: int
    mask_count: int
    sampling_weight: float

    def __post_init__(self) -> None:
        _sha256(self.case_id, label="case_id")
        _sha256(self.sequence_id, label="sequence_id")
        _sha256(self.homology_component_id, label="homology_component_id")
        _sha256(self.union_component_id, label="union_component_id")
        if type(self.level) is not int or not 1 <= self.level <= EVALUATION_LEVELS:
            raise ValueError("case level must be an integer in 1..64")
        if type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("case replicate must be a non-negative integer")
        if type(self.row_seed) is not int or not 0 <= self.row_seed < 2**64:
            raise ValueError("case row_seed must be an unsigned 64-bit integer")
        if type(self.mask_count) is not int or self.mask_count <= 0:
            raise ValueError("case mask_count must be a positive integer")
        weight = _finite(self.sampling_weight, label="case sampling_weight")
        if weight <= 0.0:
            raise ValueError("case sampling_weight must be positive")
        if self.case_id != _case_id(
            self.sequence_id,
            self.level,
            self.replicate,
            self.row_seed,
        ):
            raise ValueError("case_id does not match the validation case identity")

    def canonical_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "corruption_seed_hex": f"{self.row_seed:016x}",
            "homology_component_id": self.homology_component_id,
            "level": self.level,
            "mask_count": self.mask_count,
            "replicate": self.replicate,
            "sampling_weight": self.sampling_weight,
            "schema_version": 1,
            "sequence_id": self.sequence_id,
            "union_component_id": self.union_component_id,
        }


@dataclass(frozen=True, slots=True)
class ValidationCaseLedger:
    cases: tuple[ValidationCase, ...]
    validation_sequence_count: int
    homology_component_count: int
    union_component_count: int
    levels: int
    replicates: int
    evaluation_seed: int

    def __post_init__(self) -> None:
        if (
            type(self.cases) is not tuple
            or not self.cases
            or any(type(case) is not ValidationCase for case in self.cases)
        ):
            raise ValueError("validation case ledger cannot be empty")
        for label, value in (
            ("validation_sequence_count", self.validation_sequence_count),
            ("homology_component_count", self.homology_component_count),
            ("union_component_count", self.union_component_count),
            ("levels", self.levels),
            ("replicates", self.replicates),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"ledger {label} must be a positive integer")
        if self.levels > EVALUATION_LEVELS:
            raise ValueError("ledger levels cannot exceed 64")
        if type(self.evaluation_seed) is not int or not 0 <= self.evaluation_seed < 2**64:
            raise ValueError("ledger evaluation_seed must be an unsigned 64-bit integer")
        identities = tuple((case.sequence_id, case.level, case.replicate) for case in self.cases)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError(
                "validation cases must be uniquely ordered by sequence, level, replicate"
            )
        expected = self.validation_sequence_count * self.levels * self.replicates
        if len(self.cases) != expected:
            raise ValueError("validation case ledger has an inconsistent case census")
        if any(
            case.level > self.levels or case.replicate >= self.replicates for case in self.cases
        ):
            raise ValueError("validation case exceeds the declared level or replicate range")
        sequence_ids = {case.sequence_id for case in self.cases}
        homology_ids = {case.homology_component_id for case in self.cases}
        union_ids = {case.union_component_id for case in self.cases}
        if (
            len(sequence_ids) != self.validation_sequence_count
            or len(homology_ids) != self.homology_component_count
            or len(union_ids) != self.union_component_count
        ):
            raise ValueError("validation case ledger component census is inconsistent")
        expected_keys = {
            (level, replicate)
            for level in range(1, self.levels + 1)
            for replicate in range(self.replicates)
        }
        by_sequence: dict[str, list[ValidationCase]] = defaultdict(list)
        for case in self.cases:
            by_sequence[case.sequence_id].append(case)
        row_weights: list[float] = []
        for cases in by_sequence.values():
            if {(case.level, case.replicate) for case in cases} != expected_keys:
                raise ValueError("every validation row must have every level and replicate")
            first = cases[0]
            if any(
                case.homology_component_id != first.homology_component_id
                or case.union_component_id != first.union_component_id
                or case.sampling_weight != first.sampling_weight
                for case in cases
            ):
                raise ValueError("validation metadata changes within a ledger row")
            row_weights.append(first.sampling_weight)
        if not math.isclose(math.fsum(row_weights), 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("validation row weights in the ledger must sum to one")

    def canonical_jsonl_bytes(self) -> bytes:
        return b"".join(_canonical_json_bytes(case.canonical_record()) for case in self.cases)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_jsonl_bytes()).hexdigest()


def build_validation_case_ledger(
    source: NativeDiffusionCorpus | Sequence[DiffusionCorpusRow],
    *,
    levels: int = EVALUATION_LEVELS,
    replicates: int = EVALUATION_REPLICATES,
    evaluation_seed: int = EVALUATION_SEED,
    require_locked_census: bool = True,
) -> ValidationCaseLedger:
    """Construct the order-invariant fixed fold-4 corruption ledger."""

    if type(levels) is not int or levels <= 0 or levels > EVALUATION_LEVELS:
        raise ValueError("levels must be an integer in 1..64")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2**64:
        raise ValueError("evaluation_seed must be an unsigned 64-bit integer")
    if not isinstance(require_locked_census, bool):
        raise TypeError("require_locked_census must be boolean")
    if require_locked_census and (
        levels != EVALUATION_LEVELS
        or replicates != EVALUATION_REPLICATES
        or evaluation_seed != EVALUATION_SEED
    ):
        raise ValueError("locked validation requires 64 levels, four replicates, and seed 20260904")

    rows = source.validation_rows if isinstance(source, NativeDiffusionCorpus) else tuple(source)
    if not rows:
        raise ValueError("validation case ledger requires fold-4 rows")
    if any(type(row) is not DiffusionCorpusRow for row in rows):
        raise TypeError("validation ledger requires full DiffusionCorpusRow records")
    rows = tuple(sorted(rows, key=lambda row: row.sequence_id))
    if any(row.role != "validation" or row.fold != VALIDATION_FOLD for row in rows):
        raise ValueError("validation case ledger accepts fold-4 validation rows only")
    if len({row.sequence_id for row in rows}) != len(rows):
        raise ValueError("validation rows must have unique sequence IDs")
    for row in rows:
        if canonical_sequence_id(row.sequence) != row.sequence_id:
            raise ValueError("validation row sequence identity is invalid")
        if (
            type(row.sampling_weight) is not float
            or not math.isfinite(row.sampling_weight)
            or row.sampling_weight <= 0.0
        ):
            raise ValueError("validation sampling weights must be positive and finite")
    if not math.isclose(
        math.fsum(row.sampling_weight for row in rows),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("validation sampling weights must sum to one")

    homology_count = len({row.homology_component_id for row in rows})
    union_count = len({row.union_component_id for row in rows})
    if require_locked_census and (
        len(rows) != EXPECTED_VALIDATION_SEQUENCES
        or homology_count != EXPECTED_VALIDATION_HOMOLOGY_COMPONENTS
        or union_count != EXPECTED_VALIDATION_UNION_COMPONENTS
    ):
        raise ValueError("validation rows do not match the frozen 199/126/57 census")

    schedule = CosineMaskSchedule(offset=0.008)
    cases: list[ValidationCase] = []
    for row in rows:
        for level in range(1, levels + 1):
            mask_count = int(
                schedule.mask_counts(
                    np.asarray([len(row.sequence)], dtype=np.int64),
                    level,
                    total_levels=levels,
                )[0]
            )
            for replicate in range(replicates):
                row_seed = namespaced_seed(
                    evaluation_seed,
                    "validation",
                    CONFIG_SHA256,
                    row.sequence_id,
                    level,
                    replicate,
                )
                cases.append(
                    ValidationCase(
                        case_id=_case_id(row.sequence_id, level, replicate, row_seed),
                        sequence_id=row.sequence_id,
                        homology_component_id=row.homology_component_id,
                        union_component_id=row.union_component_id,
                        level=level,
                        replicate=replicate,
                        row_seed=row_seed,
                        mask_count=mask_count,
                        sampling_weight=row.sampling_weight,
                    )
                )
    ledger = ValidationCaseLedger(
        cases=tuple(cases),
        validation_sequence_count=len(rows),
        homology_component_count=homology_count,
        union_component_count=union_count,
        levels=levels,
        replicates=replicates,
        evaluation_seed=evaluation_seed,
    )
    if require_locked_census and len(ledger.cases) != EXPECTED_VALIDATION_CASES:
        raise RuntimeError("locked validation did not produce exactly 50,944 cases")
    return ledger


@dataclass(frozen=True, slots=True)
class CountBaselineSuite:
    """Three immutable controls estimated exclusively from trainer projections."""

    unigram: NDArray[np.float64]
    forward_markov: NDArray[np.float64]
    reverse_markov: NDArray[np.float64]
    relative_position: NDArray[np.float64]
    effective_count_scale: int
    total_effective_residue_evidence: float
    total_effective_forward_transition_evidence: float
    total_effective_reverse_transition_evidence: float
    training_projection_sha256: str
    training_sequence_ids_sha256: str

    def __post_init__(self) -> None:
        expected_shapes = {
            "unigram": (20,),
            "forward_markov": (20, 20),
            "reverse_markov": (20, 20),
            "relative_position": (5, 10, 20),
        }
        for name, shape in expected_shapes.items():
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind != "f" or raw.shape != shape or np.any(~np.isfinite(raw)):
                raise ValueError(f"{name} baseline probabilities are invalid")
            value = raw.astype(np.float64, copy=True)
            if np.any(value <= 0.0) or not np.allclose(
                np.sum(value, axis=-1), 1.0, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"{name} baseline probabilities must be positive and normalized")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if type(self.effective_count_scale) is not int or self.effective_count_scale <= 0:
            raise ValueError("effective_count_scale must be a positive integer")
        evidence = _finite(
            self.total_effective_residue_evidence,
            label="total_effective_residue_evidence",
        )
        if not math.isclose(
            evidence,
            float(self.effective_count_scale),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("effective residue evidence must equal its declared scale")
        for label in (
            "total_effective_forward_transition_evidence",
            "total_effective_reverse_transition_evidence",
        ):
            transition_evidence = _finite(getattr(self, label), label=label)
            if not math.isclose(
                transition_evidence,
                float(self.effective_count_scale),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError("effective transition evidence must equal its declared scale")
        _sha256(self.training_projection_sha256, label="training_projection_sha256")
        _sha256(self.training_sequence_ids_sha256, label="training_sequence_ids_sha256")

    def canonical_record(self) -> dict[str, object]:
        return {
            "config_sha256": CONFIG_SHA256,
            "effective_count_scale": self.effective_count_scale,
            "names": list(self.names),
            "probability_tables_sha256": self.probability_tables_sha256,
            "schema_version": 1,
            "total_effective_forward_transition_evidence": (
                self.total_effective_forward_transition_evidence
            ),
            "total_effective_residue_evidence": self.total_effective_residue_evidence,
            "total_effective_reverse_transition_evidence": (
                self.total_effective_reverse_transition_evidence
            ),
            "training_projection_sha256": self.training_projection_sha256,
            "training_sequence_ids_sha256": self.training_sequence_ids_sha256,
        }

    @property
    def probability_tables_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_PROBABILITY_TABLE_DOMAIN)
        for name in ("unigram", "forward_markov", "reverse_markov", "relative_position"):
            values = np.asarray(getattr(self, name), dtype="<f8", order="C")
            _framed_update(
                digest,
                _canonical_json_bytes(
                    {
                        "dtype": "little_endian_float64",
                        "name": name,
                        "shape": list(values.shape),
                    }
                ),
            )
            _framed_update(digest, values.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def logical_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_CONTROL_LOGICAL_DOMAIN)
        _framed_update(digest, _canonical_json_bytes(self.canonical_record()))
        return digest.hexdigest()

    @property
    def names(self) -> tuple[str, ...]:
        return (
            "component_weighted_unigram",
            "component_weighted_bidirectional_markov",
            "length_relative_position_frequency",
        )

    def probabilities(
        self,
        name: str,
        corrupted_tokens: object,
        attention_mask: object,
        lengths: object,
    ) -> NDArray[np.float64]:
        """Return finite 20-way probabilities without consulting validation targets."""

        method = _method(name)
        if method not in self.names:
            raise ValueError(f"unknown count baseline: {method}")
        tokens, valid, length_values = _validated_corrupted_batch(
            corrupted_tokens,
            attention_mask,
            lengths,
        )
        batch, width = tokens.shape
        output = np.empty((batch, width, len(ALPHABET)), dtype=np.float64)
        output[:] = self.unigram
        if method == "component_weighted_unigram":
            return output

        if method == "length_relative_position_frequency":
            length_edges = (8, 15, 20, 25, 33, 51)
            for row, length in enumerate(length_values):
                length_bin = int(np.searchsorted(length_edges, int(length), side="right") - 1)
                for position in range(int(length)):
                    position_bin = min(9, (10 * position) // int(length))
                    output[row, position] = self.relative_position[length_bin, position_bin]
            return output

        log_unigram = np.log(self.unigram)
        log_forward = np.log(self.forward_markov)
        log_reverse = np.log(self.reverse_markov)
        for row, length in enumerate(length_values):
            for position in range(int(length)):
                score = log_unigram.copy()
                if position > 0 and 0 <= tokens[row, position - 1] < len(ALPHABET):
                    left = int(tokens[row, position - 1])
                    score += log_forward[left] - log_unigram
                if position + 1 < length and 0 <= tokens[row, position + 1] < len(ALPHABET):
                    right = int(tokens[row, position + 1])
                    score += log_reverse[right] - log_unigram
                score -= np.max(score)
                probabilities = np.exp(score)
                output[row, position] = probabilities / math.fsum(probabilities.tolist())
        return output


def _validated_training_rows(rows: Sequence[TrainingRow]) -> tuple[TrainingRow, ...]:
    source = tuple(rows)
    if not source:
        raise ValueError("count baselines require training rows")
    if any(type(row) is not TrainingRow for row in source):
        raise TypeError("count baselines accept the three-field TrainingRow projection only")
    values = tuple(sorted(source, key=lambda row: row.sequence_id))
    if len({row.sequence_id for row in values}) != len(values):
        raise ValueError("count-baseline training rows must have unique sequence IDs")
    for row in values:
        if canonical_sequence_id(row.sequence) != row.sequence_id:
            raise ValueError("count-baseline training sequence identity is invalid")
        if (
            type(row.sampling_weight) is not float
            or not math.isfinite(row.sampling_weight)
            or row.sampling_weight <= 0.0
        ):
            raise ValueError("count-baseline sampling weights must be positive and finite")
    if not math.isclose(
        math.fsum(row.sampling_weight for row in values),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("count-baseline sampling weights must sum to one")
    return values


def _validated_training_projection(
    training: TrainingDistribution | Sequence[TrainingRow],
    *,
    require_locked_census: bool,
) -> tuple[tuple[TrainingRow, ...], str]:
    if not isinstance(require_locked_census, bool):
        raise TypeError("require_locked_census must be boolean")
    if isinstance(training, TrainingDistribution):
        rows = _validated_training_rows(training.rows)
        if training.probabilities != tuple(row.sampling_weight for row in rows):
            raise ValueError("training distribution probabilities differ from its rows")
    else:
        rows = _validated_training_rows(training)
    projection_payload = b"".join(
        _canonical_json_bytes(
            {
                "sampling_weight": row.sampling_weight,
                "sequence": row.sequence,
                "sequence_id": row.sequence_id,
            }
        )
        for row in rows
    )
    projection_sha256 = hashlib.sha256(projection_payload).hexdigest()
    if require_locked_census and (
        len(rows) != EXPECTED_TRAIN_SEQUENCES
        or projection_sha256 != ACCEPTED_TRAINING_PROJECTION_SHA256
    ):
        raise ValueError("locked controls require the accepted 914-row training projection")
    return rows, projection_sha256


def fit_count_baselines(
    training: TrainingDistribution | Sequence[TrainingRow],
    *,
    require_locked_census: bool = True,
    effective_count_scale: int | None = None,
) -> CountBaselineSuite:
    """Fit the preregistered controls from the three-field training view only."""

    rows, projection_sha256 = _validated_training_projection(
        training,
        require_locked_census=require_locked_census,
    )
    if effective_count_scale is None:
        scale = EXPECTED_TRAIN_SEQUENCES if require_locked_census else len(rows)
    elif isinstance(effective_count_scale, bool) or not isinstance(effective_count_scale, int):
        raise TypeError("effective_count_scale must be an integer")
    else:
        scale = effective_count_scale
    if scale <= 0:
        raise ValueError("effective_count_scale must be positive")
    if require_locked_census and (len(rows) != EXPECTED_TRAIN_SEQUENCES or scale != len(rows)):
        raise ValueError("locked count baselines require exactly 914 training rows and scale 914")

    vocabulary = PeptideVocabulary(ALPHABET)
    unigram_counts = np.zeros(20, dtype=np.float64)
    forward_counts = np.zeros((20, 20), dtype=np.float64)
    reverse_counts = np.zeros((20, 20), dtype=np.float64)
    position_counts = np.zeros((5, 10, 20), dtype=np.float64)
    length_edges = (8, 15, 20, 25, 33, 51)
    for row in rows:
        encoded = vocabulary.encode([row.sequence]).tokens[0, : len(row.sequence)]
        residue_mass = scale * row.sampling_weight / len(row.sequence)
        for position, residue_raw in enumerate(encoded):
            residue = int(residue_raw)
            unigram_counts[residue] += residue_mass
            length_bin = int(np.searchsorted(length_edges, len(row.sequence), side="right") - 1)
            position_bin = min(9, (10 * position) // len(row.sequence))
            position_counts[length_bin, position_bin, residue] += residue_mass
        if len(encoded) > 1:
            transition_mass = scale * row.sampling_weight / (len(encoded) - 1)
            for left_raw, right_raw in pairwise(encoded):
                left, right = int(left_raw), int(right_raw)
                forward_counts[left, right] += transition_mass
                reverse_counts[right, left] += transition_mass

    if not math.isclose(
        float(np.sum(unigram_counts)),
        float(scale),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("effective unigram evidence does not equal its frozen scale")
    pseudocount = 0.5
    unigram = unigram_counts + pseudocount
    unigram /= np.sum(unigram)
    forward_markov = forward_counts + pseudocount
    forward_markov /= np.sum(forward_markov, axis=1, keepdims=True)
    reverse_markov = reverse_counts + pseudocount
    reverse_markov /= np.sum(reverse_markov, axis=1, keepdims=True)
    relative_position = position_counts + 20.0 * unigram[None, None, :]
    relative_position /= np.sum(relative_position, axis=2, keepdims=True)

    digest = hashlib.sha256()
    digest.update(_TRAIN_IDS_DOMAIN)
    for row in rows:
        digest.update(_canonical_json_bytes({"sequence_id": row.sequence_id}))
    return CountBaselineSuite(
        unigram=unigram,
        forward_markov=forward_markov,
        reverse_markov=reverse_markov,
        relative_position=relative_position,
        effective_count_scale=scale,
        total_effective_residue_evidence=float(np.sum(unigram_counts)),
        total_effective_forward_transition_evidence=float(np.sum(forward_counts)),
        total_effective_reverse_transition_evidence=float(np.sum(reverse_counts)),
        training_projection_sha256=projection_sha256,
        training_sequence_ids_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class GeneratorControlCandidate:
    """One unfiltered raw proposal from a train-only count generator."""

    method: str
    control_logical_sha256: str
    training_projection_sha256: str
    ordinal: int
    seed: int
    sequence_id: str
    sequence: str
    length: int

    def __post_init__(self) -> None:
        if _method(self.method) not in GENERATOR_CONTROL_METHODS:
            raise ValueError("unknown count generator control")
        _sha256(self.control_logical_sha256, label="control_logical_sha256")
        _sha256(self.training_projection_sha256, label="training_projection_sha256")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 2**64:
            raise ValueError("control candidate ordinal must be unsigned 64-bit integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("control candidate seed must be unsigned 64-bit integer")
        if type(self.length) is not int or not 8 <= self.length <= 50:
            raise ValueError("control candidate length must be an integer in 8..50")
        if type(self.sequence) is not str or len(self.sequence) != self.length:
            raise ValueError("control candidate sequence and length disagree")
        if set(self.sequence) - set(ALPHABET):
            raise ValueError("control candidate contains a noncanonical residue")
        if canonical_sequence_id(self.sequence) != self.sequence_id:
            raise ValueError("control candidate sequence_id disagrees with its sequence")

    def canonical_record(self) -> dict[str, object]:
        return {
            "control_logical_sha256": self.control_logical_sha256,
            "length": self.length,
            "method": self.method,
            "ordinal": self.ordinal,
            "schema_version": 1,
            "seed": self.seed,
            "sequence": self.sequence,
            "sequence_id": self.sequence_id,
            "training_projection_sha256": self.training_projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class GeneratorControlSamplingResult:
    candidates: tuple[GeneratorControlCandidate, ...]
    length_plan_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.candidates) is not tuple
            or not self.candidates
            or any(type(item) is not GeneratorControlCandidate for item in self.candidates)
        ):
            raise ValueError("control sampling result requires a non-empty candidate tuple")
        ordinals = tuple(item.ordinal for item in self.candidates)
        if ordinals != tuple(sorted(ordinals)) or len(ordinals) != len(set(ordinals)):
            raise ValueError("control candidates must have unique sorted ordinals")
        for attribute in (
            "method",
            "control_logical_sha256",
            "training_projection_sha256",
            "seed",
        ):
            if len({getattr(item, attribute) for item in self.candidates}) != 1:
                raise ValueError(f"control candidates must share one {attribute}")
        _, expected_hash = canonical_length_plan(
            [item.length for item in self.candidates],
            ordinals=ordinals,
            require_locked_count=False,
        )
        if self.length_plan_sha256 != expected_hash:
            raise ValueError("control result length-plan hash disagrees with its candidates")

    def canonical_jsonl_bytes(self) -> bytes:
        return b"".join(_canonical_json_bytes(item.canonical_record()) for item in self.candidates)


def _control_draw_seed(
    method: str,
    seed: int,
    ordinal: int,
    position: int,
) -> int:
    digest = hashlib.sha256()
    digest.update(_CONTROL_DRAW_DOMAIN)
    for payload in (
        method.encode("ascii"),
        seed.to_bytes(8, "big"),
        ordinal.to_bytes(8, "big"),
        position.to_bytes(2, "big"),
    ):
        _framed_update(digest, payload)
    return int.from_bytes(digest.digest()[:8], "big")


def _draw_residue(probabilities: NDArray[np.float64], *, draw_seed: int) -> int:
    if probabilities.shape != (len(ALPHABET),) or np.any(~np.isfinite(probabilities)):
        raise ValueError("control residue probabilities are invalid")
    if np.any(probabilities <= 0.0) or not math.isclose(
        math.fsum(probabilities.tolist()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("control residue probabilities must be positive and normalized")
    uniform = float(np.random.Generator(np.random.PCG64DXSM(draw_seed)).random())
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    return min(
        int(np.searchsorted(cumulative, uniform, side="right")),
        len(ALPHABET) - 1,
    )


def sample_count_generator_control_v0(
    suite: CountBaselineSuite,
    lengths: Sequence[int],
    *,
    method: str,
    seed: int,
    ordinals: Sequence[int] | None = None,
    batch_size: int = SAMPLING_BATCH_SEQUENCES,
    require_locked_count: bool = True,
) -> GeneratorControlSamplingResult:
    """Sample the preregistered unigram or forward-Markov raw control census."""

    if type(suite) is not CountBaselineSuite:
        raise TypeError("suite must be a fitted CountBaselineSuite")
    method_name = _method(method)
    if method_name not in GENERATOR_CONTROL_METHODS:
        raise ValueError("method must name a preregistered count generator control")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("control sampling seed must be an unsigned 64-bit integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    plan, length_plan_sha256 = canonical_length_plan(
        lengths,
        ordinals=ordinals,
        require_locked_count=require_locked_count,
    )
    if require_locked_count and batch_size != SAMPLING_BATCH_SEQUENCES:
        raise ValueError("locked v0 control sampling requires batches of 256 proposals")
    if (
        require_locked_count
        and suite.training_projection_sha256 != ACCEPTED_TRAINING_PROJECTION_SHA256
    ):
        raise ValueError("locked v0 controls require the accepted training projection")

    candidates: list[GeneratorControlCandidate] = []
    logical_sha256 = suite.logical_sha256
    for start in range(0, len(plan), batch_size):
        for ordinal, length in plan[start : start + batch_size]:
            residues: list[int] = []
            for position in range(length):
                probabilities = (
                    suite.unigram
                    if method_name == "component_weighted_unigram" or position == 0
                    else suite.forward_markov[residues[-1]]
                )
                residues.append(
                    _draw_residue(
                        probabilities,
                        draw_seed=_control_draw_seed(
                            method_name,
                            seed,
                            ordinal,
                            position,
                        ),
                    )
                )
            sequence = "".join(ALPHABET[index] for index in residues)
            candidates.append(
                GeneratorControlCandidate(
                    method=method_name,
                    control_logical_sha256=logical_sha256,
                    training_projection_sha256=suite.training_projection_sha256,
                    ordinal=ordinal,
                    seed=seed,
                    sequence_id=canonical_sequence_id(sequence),
                    sequence=sequence,
                    length=length,
                )
            )
    return GeneratorControlSamplingResult(
        candidates=tuple(candidates),
        length_plan_sha256=length_plan_sha256,
    )


def _validated_corrupted_batch(
    corrupted_tokens: object,
    attention_mask: object,
    lengths: object,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64]]:
    raw_tokens = np.asarray(corrupted_tokens)
    raw_mask = np.asarray(attention_mask)
    raw_lengths = np.asarray(lengths)
    if raw_tokens.dtype.kind not in {"i", "u"} or raw_tokens.ndim != 2:
        raise TypeError("corrupted_tokens must be a two-dimensional integer array")
    if raw_mask.dtype.kind != "b" or raw_mask.shape != raw_tokens.shape:
        raise TypeError("attention_mask must be a matching Boolean array")
    if raw_lengths.dtype.kind not in {"i", "u"} or raw_lengths.shape != (len(raw_tokens),):
        raise TypeError("lengths must be an aligned integer vector")
    tokens = raw_tokens.astype(np.int64, copy=False)
    valid = raw_mask.astype(np.bool_, copy=False)
    length_values = raw_lengths.astype(np.int64, copy=False)
    if np.any((length_values < 8) | (length_values > 50)):
        raise ValueError("lengths must lie in 8..50")
    expected_mask = np.arange(tokens.shape[1])[None, :] < length_values[:, None]
    if not np.array_equal(valid, expected_mask):
        raise ValueError("attention_mask must equal the declared length prefix")
    if np.any(valid & ((tokens < 0) | (tokens > 21))):
        raise ValueError("valid positions contain a token outside residues or MASK")
    if np.any(valid & (tokens == 20)) or np.any(~valid & (tokens != 20)):
        raise ValueError("PAD placement does not match attention_mask")
    return tokens, valid, length_values


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    method: str
    case_id: str
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    level: int
    replicate: int
    sampling_weight: float
    masked_tokens: int
    mean_nll: float
    top1_accuracy: float
    top3_accuracy: float
    mean_brier: float
    calibration_counts: tuple[int, ...]
    calibration_confidence_sums: tuple[float, ...]
    calibration_correct_sums: tuple[int, ...]

    def __post_init__(self) -> None:
        _method(self.method)
        _sha256(self.case_id, label="case_id")
        _sha256(self.sequence_id, label="sequence_id")
        _sha256(self.homology_component_id, label="homology_component_id")
        _sha256(self.union_component_id, label="union_component_id")
        if type(self.level) is not int or not 1 <= self.level <= EVALUATION_LEVELS:
            raise ValueError("case metric level must be an integer in 1..64")
        if type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("case metric replicate must be a non-negative integer")
        if type(self.masked_tokens) is not int or self.masked_tokens <= 0:
            raise ValueError("case metric masked_tokens must be a positive integer")
        if _finite(self.sampling_weight, label="case metric sampling_weight") <= 0.0:
            raise ValueError("case metric sampling_weight must be positive")
        if _finite(self.mean_nll, label="case metric mean_nll") < 0.0:
            raise ValueError("case metric mean_nll must be non-negative")
        top1 = _bounded_float(
            self.top1_accuracy,
            label="case metric top1_accuracy",
            lower=0.0,
            upper=1.0,
        )
        top3 = _bounded_float(
            self.top3_accuracy,
            label="case metric top3_accuracy",
            lower=0.0,
            upper=1.0,
        )
        if top3 < top1:
            raise ValueError("case metric top3_accuracy cannot be below top1_accuracy")
        _bounded_float(
            self.mean_brier,
            label="case metric mean_brier",
            lower=0.0,
            upper=2.0,
        )
        if any(
            type(values) is not tuple or len(values) != CALIBRATION_BINS
            for values in (
                self.calibration_counts,
                self.calibration_confidence_sums,
                self.calibration_correct_sums,
            )
        ):
            raise ValueError("case calibration vectors must be 15-element tuples")
        if any(type(value) is not int or value < 0 for value in self.calibration_counts):
            raise ValueError("case calibration counts must be non-negative integers")
        if any(type(value) is not int or value < 0 for value in self.calibration_correct_sums):
            raise ValueError("case calibration correct sums must be non-negative integers")
        for count, confidence, correct in zip(
            self.calibration_counts,
            self.calibration_confidence_sums,
            self.calibration_correct_sums,
            strict=True,
        ):
            if (
                not 0.0
                <= _finite(
                    confidence,
                    label="case calibration confidence sum",
                )
                <= count
            ):
                raise ValueError("case calibration confidence exceeds its bin count")
            if correct > count:
                raise ValueError("case calibration correct sum exceeds its bin count")
        if sum(self.calibration_counts) != self.masked_tokens:
            raise ValueError("case calibration counts do not equal masked_tokens")
        correct_total = sum(self.calibration_correct_sums)
        if not math.isclose(
            self.top1_accuracy,
            correct_total / self.masked_tokens,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("case top1_accuracy disagrees with calibration correctness")

    def canonical_record(self) -> dict[str, object]:
        return {
            "calibration_confidence_sums": list(self.calibration_confidence_sums),
            "calibration_correct_sums": list(self.calibration_correct_sums),
            "calibration_counts": list(self.calibration_counts),
            "case_id": self.case_id,
            "homology_component_id": self.homology_component_id,
            "level": self.level,
            "masked_tokens": self.masked_tokens,
            "mean_brier": self.mean_brier,
            "mean_nll": self.mean_nll,
            "method": self.method,
            "replicate": self.replicate,
            "sampling_weight": self.sampling_weight,
            "schema_version": 1,
            "sequence_id": self.sequence_id,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "union_component_id": self.union_component_id,
        }


@dataclass(frozen=True, slots=True)
class RowMetrics:
    method: str
    sequence_id: str
    homology_component_id: str
    union_component_id: str
    sampling_weight: float
    case_count: int
    masked_tokens: int
    mean_nll: float
    top1_accuracy: float
    top3_accuracy: float
    mean_brier: float
    ece: float

    def __post_init__(self) -> None:
        _method(self.method)
        _sha256(self.sequence_id, label="sequence_id")
        _sha256(self.homology_component_id, label="homology_component_id")
        _sha256(self.union_component_id, label="union_component_id")
        if _finite(self.sampling_weight, label="row metric sampling_weight") <= 0.0:
            raise ValueError("row metric sampling_weight must be positive")
        if type(self.case_count) is not int or self.case_count <= 0:
            raise ValueError("row metric case_count must be a positive integer")
        if type(self.masked_tokens) is not int or self.masked_tokens <= 0:
            raise ValueError("row metric masked_tokens must be a positive integer")
        if _finite(self.mean_nll, label="row metric mean_nll") < 0.0:
            raise ValueError("row metric mean_nll must be non-negative")
        top1 = _bounded_float(
            self.top1_accuracy,
            label="row metric top1_accuracy",
            lower=0.0,
            upper=1.0,
        )
        top3 = _bounded_float(
            self.top3_accuracy,
            label="row metric top3_accuracy",
            lower=0.0,
            upper=1.0,
        )
        if top3 < top1:
            raise ValueError("row metric top3_accuracy cannot be below top1_accuracy")
        _bounded_float(
            self.mean_brier,
            label="row metric mean_brier",
            lower=0.0,
            upper=2.0,
        )
        _bounded_float(self.ece, label="row metric ece", lower=0.0, upper=1.0)

    def canonical_record(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "ece": self.ece,
            "homology_component_id": self.homology_component_id,
            "masked_tokens": self.masked_tokens,
            "mean_brier": self.mean_brier,
            "mean_nll": self.mean_nll,
            "method": self.method,
            "sampling_weight": self.sampling_weight,
            "schema_version": 1,
            "sequence_id": self.sequence_id,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "union_component_id": self.union_component_id,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    method: str
    ledger_sha256: str
    case_metrics: tuple[CaseMetrics, ...]
    row_metrics: tuple[RowMetrics, ...]
    primary_nll: float
    perplexity: float
    top1_accuracy: float
    top3_accuracy: float
    mean_brier: float
    ece: float

    def __post_init__(self) -> None:
        method = _method(self.method)
        _sha256(self.ledger_sha256, label="ledger_sha256")
        if (
            type(self.case_metrics) is not tuple
            or not self.case_metrics
            or any(type(item) is not CaseMetrics for item in self.case_metrics)
        ):
            raise ValueError("evaluation case_metrics must be a non-empty tuple")
        if (
            type(self.row_metrics) is not tuple
            or not self.row_metrics
            or any(type(item) is not RowMetrics for item in self.row_metrics)
        ):
            raise ValueError("evaluation row_metrics must be a non-empty tuple")
        if any(item.method != method for item in (*self.case_metrics, *self.row_metrics)):
            raise ValueError("evaluation records must use the declared method")
        row_ids = tuple(item.sequence_id for item in self.row_metrics)
        if row_ids != tuple(sorted(row_ids)) or len(row_ids) != len(set(row_ids)):
            raise ValueError("evaluation row metrics must have unique sorted sequence IDs")
        if sum(item.case_count for item in self.row_metrics) != len(self.case_metrics):
            raise ValueError("evaluation row and case metric censuses disagree")
        weight_sum = math.fsum(item.sampling_weight for item in self.row_metrics)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("evaluation row weights must sum to one")
        primary_nll = _finite(self.primary_nll, label="primary_nll")
        if primary_nll < 0.0:
            raise ValueError("primary_nll must be non-negative")
        expected_nll = math.fsum(item.sampling_weight * item.mean_nll for item in self.row_metrics)
        if not math.isclose(primary_nll, expected_nll, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("primary_nll disagrees with row metrics")
        perplexity = _finite(self.perplexity, label="perplexity")
        try:
            expected_perplexity = math.exp(primary_nll)
        except OverflowError as error:
            raise ValueError("primary_nll is too large for finite perplexity") from error
        if not math.isclose(perplexity, expected_perplexity, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError("perplexity disagrees with primary_nll")
        for label, value, row_field, upper in (
            ("top1_accuracy", self.top1_accuracy, "top1_accuracy", 1.0),
            ("top3_accuracy", self.top3_accuracy, "top3_accuracy", 1.0),
            ("mean_brier", self.mean_brier, "mean_brier", 2.0),
        ):
            observed = _bounded_float(value, label=label, lower=0.0, upper=upper)
            expected = math.fsum(
                item.sampling_weight * getattr(item, row_field) for item in self.row_metrics
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{label} disagrees with row metrics")
        if self.top3_accuracy < self.top1_accuracy:
            raise ValueError("top3_accuracy cannot be below top1_accuracy")
        _bounded_float(self.ece, label="ece", lower=0.0, upper=1.0)

    def canonical_case_metrics_jsonl_bytes(self) -> bytes:
        return b"".join(
            _canonical_json_bytes(item.canonical_record()) for item in self.case_metrics
        )

    def canonical_row_metrics_jsonl_bytes(self) -> bytes:
        return b"".join(_canonical_json_bytes(item.canonical_record()) for item in self.row_metrics)

    def canonical_summary_record(self) -> dict[str, object]:
        case_bytes = self.canonical_case_metrics_jsonl_bytes()
        row_bytes = self.canonical_row_metrics_jsonl_bytes()
        return {
            "case_count": len(self.case_metrics),
            "case_metrics_sha256": hashlib.sha256(case_bytes).hexdigest(),
            "ece": self.ece,
            "ledger_sha256": self.ledger_sha256,
            "mean_brier": self.mean_brier,
            "method": self.method,
            "perplexity": self.perplexity,
            "primary_nll": self.primary_nll,
            "row_count": len(self.row_metrics),
            "row_metrics_sha256": hashlib.sha256(row_bytes).hexdigest(),
            "schema_version": 1,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
        }


def _logit_probabilities(
    logits: object,
    *,
    expected_shape: tuple[int, int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    raw = np.asarray(logits)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("logit provider must return a real numeric array")
    if raw.shape != expected_shape:
        raise ValueError(f"logit provider returned shape {raw.shape}; expected {expected_shape}")
    values = raw.astype(np.float64, copy=True)
    if np.any(~np.isfinite(values)):
        raise ValueError("logit provider returned non-finite values")
    maxima = np.max(values, axis=-1, keepdims=True)
    shifted = values - maxima
    exponentials = np.exp(shifted)
    totals = np.sum(exponentials, axis=-1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("logit provider produced invalid categorical mass")
    log_probabilities = shifted - np.log(totals)
    return np.exp(log_probabilities), log_probabilities


def _score_case(
    case: ValidationCase,
    clean_tokens: NDArray[np.int64],
    selected_mask: NDArray[np.bool_],
    probabilities: NDArray[np.float64],
    log_probabilities: NDArray[np.float64],
    *,
    method: str,
) -> CaseMetrics:
    positions = np.flatnonzero(selected_mask)
    if len(positions) != case.mask_count:
        raise ValueError("materialized corruption does not match its declared mask count")
    if probabilities.ndim != 2 or probabilities.shape != log_probabilities.shape:
        raise ValueError("case probability and log-probability arrays must align")
    if probabilities.shape[1] != len(ALPHABET) or probabilities.shape[0] != len(clean_tokens):
        raise ValueError("case probabilities have the wrong shape")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("case probabilities must be finite and non-negative")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("case probabilities must sum to one")

    nll_values: list[float] = []
    top1_correct = 0
    top3_correct = 0
    brier_sum = 0.0
    calibration_counts = [0] * CALIBRATION_BINS
    calibration_confidence_sums = [0.0] * CALIBRATION_BINS
    calibration_correct_sums = [0] * CALIBRATION_BINS
    for position_raw in positions:
        position = int(position_raw)
        target = int(clean_tokens[position])
        if not 0 <= target < len(ALPHABET):
            raise ValueError("selected validation target is not a residue")
        distribution = probabilities[position]
        nll = -float(log_probabilities[position, target])
        if not math.isfinite(nll):
            raise ValueError("selected validation target has non-finite NLL")
        nll_values.append(nll)
        ranking = np.argsort(-distribution, kind="stable")
        prediction = int(ranking[0])
        correct = int(prediction == target)
        top1_correct += correct
        top3_correct += int(target in ranking[:3])
        target_vector = np.zeros(len(ALPHABET), dtype=np.float64)
        target_vector[target] = 1.0
        brier_sum += float(np.sum(np.square(distribution - target_vector)))
        confidence = float(distribution[prediction])
        bin_index = min(CALIBRATION_BINS - 1, int(confidence * CALIBRATION_BINS))
        calibration_counts[bin_index] += 1
        calibration_confidence_sums[bin_index] += confidence
        calibration_correct_sums[bin_index] += correct

    count = len(positions)
    return CaseMetrics(
        method=_method(method),
        case_id=case.case_id,
        sequence_id=case.sequence_id,
        homology_component_id=case.homology_component_id,
        union_component_id=case.union_component_id,
        level=case.level,
        replicate=case.replicate,
        sampling_weight=case.sampling_weight,
        masked_tokens=count,
        mean_nll=math.fsum(nll_values) / count,
        top1_accuracy=top1_correct / count,
        top3_accuracy=top3_correct / count,
        mean_brier=brier_sum / count,
        calibration_counts=tuple(calibration_counts),
        calibration_confidence_sums=tuple(calibration_confidence_sums),
        calibration_correct_sums=tuple(calibration_correct_sums),
    )


def _ece(
    counts: Sequence[float],
    confidence_sums: Sequence[float],
    correct_sums: Sequence[float],
) -> float:
    total = math.fsum(counts)
    if total <= 0.0:
        raise ValueError("ECE requires positive calibration mass")
    result = 0.0
    for count, confidence, correct in zip(
        counts,
        confidence_sums,
        correct_sums,
        strict=True,
    ):
        if count > 0.0:
            result += count / total * abs(correct / count - confidence / count)
    return result


def aggregate_case_metrics(
    ledger: ValidationCaseLedger,
    case_metrics: Sequence[CaseMetrics],
) -> EvaluationResult:
    """Aggregate case means using the accepted per-row component weights."""

    metrics = tuple(case_metrics)
    if len(metrics) != len(ledger.cases):
        raise ValueError("case metrics must have exactly one record per ledger case")
    expected_ids = tuple(case.case_id for case in ledger.cases)
    observed_ids = tuple(item.case_id for item in metrics)
    if observed_ids != expected_ids:
        raise ValueError("case metrics must follow the exact validation ledger order")
    for case, item in zip(ledger.cases, metrics, strict=True):
        if (
            item.sequence_id != case.sequence_id
            or item.homology_component_id != case.homology_component_id
            or item.union_component_id != case.union_component_id
            or item.level != case.level
            or item.replicate != case.replicate
            or item.sampling_weight != case.sampling_weight
            or item.masked_tokens != case.mask_count
        ):
            raise ValueError("case metric metadata differs from its validation ledger case")
    methods = {_method(item.method) for item in metrics}
    if len(methods) != 1:
        raise ValueError("all case metrics must use one method")
    method = methods.pop()

    grouped: dict[str, list[CaseMetrics]] = defaultdict(list)
    for item in metrics:
        grouped[item.sequence_id].append(item)
    row_metrics: list[RowMetrics] = []
    global_counts = [0.0] * CALIBRATION_BINS
    global_confidence = [0.0] * CALIBRATION_BINS
    global_correct = [0.0] * CALIBRATION_BINS
    for sequence_id in sorted(grouped):
        items = grouped[sequence_id]
        if len(items) != ledger.levels * ledger.replicates:
            raise ValueError("each validation row must have every level and replicate")
        keys = {(item.level, item.replicate) for item in items}
        expected_keys = {
            (level, replicate)
            for level in range(1, ledger.levels + 1)
            for replicate in range(ledger.replicates)
        }
        if keys != expected_keys:
            raise ValueError("validation row case coverage is incomplete")
        first = items[0]
        if any(
            item.homology_component_id != first.homology_component_id
            or item.union_component_id != first.union_component_id
            or item.sampling_weight != first.sampling_weight
            for item in items
        ):
            raise ValueError("validation row metadata changes across corruption cases")
        case_count = len(items)
        row_counts = [0.0] * CALIBRATION_BINS
        row_confidence = [0.0] * CALIBRATION_BINS
        row_correct = [0.0] * CALIBRATION_BINS
        for item in items:
            token_scale = 1.0 / (case_count * item.masked_tokens)
            for index in range(CALIBRATION_BINS):
                weighted_count = item.calibration_counts[index] * token_scale
                row_counts[index] += weighted_count
                row_confidence[index] += item.calibration_confidence_sums[index] * token_scale
                row_correct[index] += item.calibration_correct_sums[index] * token_scale
        for index in range(CALIBRATION_BINS):
            global_counts[index] += first.sampling_weight * row_counts[index]
            global_confidence[index] += first.sampling_weight * row_confidence[index]
            global_correct[index] += first.sampling_weight * row_correct[index]
        row_metrics.append(
            RowMetrics(
                method=method,
                sequence_id=sequence_id,
                homology_component_id=first.homology_component_id,
                union_component_id=first.union_component_id,
                sampling_weight=first.sampling_weight,
                case_count=case_count,
                masked_tokens=sum(item.masked_tokens for item in items),
                mean_nll=math.fsum(item.mean_nll for item in items) / case_count,
                top1_accuracy=math.fsum(item.top1_accuracy for item in items) / case_count,
                top3_accuracy=math.fsum(item.top3_accuracy for item in items) / case_count,
                mean_brier=math.fsum(item.mean_brier for item in items) / case_count,
                ece=_ece(row_counts, row_confidence, row_correct),
            )
        )

    weight_sum = math.fsum(item.sampling_weight for item in row_metrics)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("row-metric sampling weights must sum to one")
    primary_nll = math.fsum(item.sampling_weight * item.mean_nll for item in row_metrics)
    try:
        perplexity = math.exp(primary_nll)
    except OverflowError as error:
        raise ValueError("primary NLL is too large for finite perplexity") from error
    result = EvaluationResult(
        method=method,
        ledger_sha256=ledger.sha256,
        case_metrics=metrics,
        row_metrics=tuple(row_metrics),
        primary_nll=primary_nll,
        perplexity=perplexity,
        top1_accuracy=math.fsum(item.sampling_weight * item.top1_accuracy for item in row_metrics),
        top3_accuracy=math.fsum(item.sampling_weight * item.top3_accuracy for item in row_metrics),
        mean_brier=math.fsum(item.sampling_weight * item.mean_brier for item in row_metrics),
        ece=_ece(global_counts, global_confidence, global_correct),
    )
    for label, value in result.canonical_summary_record().items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"evaluation summary {label} is non-finite")
    return result


def _evaluation_rows(
    ledger: ValidationCaseLedger,
    validation_rows: Sequence[DiffusionCorpusRow],
) -> dict[str, DiffusionCorpusRow]:
    values = tuple(validation_rows)
    if any(type(row) is not DiffusionCorpusRow for row in values):
        raise TypeError("evaluation requires full DiffusionCorpusRow validation records")
    rows = {row.sequence_id: row for row in values}
    if len(rows) != len(values):
        raise ValueError("validation rows contain duplicate sequence IDs")
    expected_ids = {case.sequence_id for case in ledger.cases}
    if set(rows) != expected_ids:
        raise ValueError("validation rows do not exactly match the case ledger")
    for row in rows.values():
        if row.role != "validation" or row.fold != VALIDATION_FOLD:
            raise ValueError("evaluator accepts fold-4 validation rows only")
        if canonical_sequence_id(row.sequence) != row.sequence_id:
            raise ValueError("validation row sequence identity is invalid")
    return rows


def _materialize_batch(
    cases: Sequence[ValidationCase],
    rows: Mapping[str, DiffusionCorpusRow],
    *,
    total_levels: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    vocabulary = PeptideVocabulary(ALPHABET)
    sequences = [rows[case.sequence_id].sequence for case in cases]
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    encoded = vocabulary.encode(sequences, max_length=50)
    levels = np.asarray([case.level for case in cases], dtype=np.int64)
    diffusion = AbsorbingDiffusion(vocabulary, CosineMaskSchedule(offset=0.008))
    corrupted, selected = diffusion.corrupt_fixed_count(
        encoded.tokens,
        encoded.attention_mask,
        levels,
        total_levels=total_levels,
        row_seeds=[case.row_seed for case in cases],
    )
    observed = np.sum(selected, axis=1, dtype=np.int64)
    expected = np.asarray([case.mask_count for case in cases], dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise RuntimeError("materialized validation masks disagree with the ledger")
    return encoded.tokens, encoded.attention_mask, selected, corrupted, lengths


def _require_locked_evaluation(ledger: ValidationCaseLedger, *, batch_size: int) -> None:
    if batch_size != EVALUATION_BATCH_SEQUENCES:
        raise ValueError("locked v0 evaluation requires batches of 256 cases")
    if (
        ledger.validation_sequence_count != EXPECTED_VALIDATION_SEQUENCES
        or ledger.homology_component_count != EXPECTED_VALIDATION_HOMOLOGY_COMPONENTS
        or ledger.union_component_count != EXPECTED_VALIDATION_UNION_COMPONENTS
        or ledger.levels != EVALUATION_LEVELS
        or ledger.replicates != EVALUATION_REPLICATES
        or ledger.evaluation_seed != EVALUATION_SEED
        or len(ledger.cases) != EXPECTED_VALIDATION_CASES
    ):
        raise ValueError("locked v0 evaluation requires the frozen 199/126/57 case ledger")


def evaluate_logit_provider(
    ledger: ValidationCaseLedger,
    validation_rows: Sequence[DiffusionCorpusRow],
    logit_provider: LogitProvider,
    *,
    method: str,
    batch_size: int = EVALUATION_BATCH_SEQUENCES,
    require_locked_batch_size: bool = True,
) -> EvaluationResult:
    """Evaluate model logits on the ledger without exposing targets to the provider."""

    if not callable(logit_provider):
        raise TypeError("logit_provider must be callable")
    method_name = _method(method)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(require_locked_batch_size, bool):
        raise TypeError("require_locked_batch_size must be boolean")
    if require_locked_batch_size:
        _require_locked_evaluation(ledger, batch_size=batch_size)
    rows = _evaluation_rows(ledger, validation_rows)
    metrics: list[CaseMetrics] = []
    for start in range(0, len(ledger.cases), batch_size):
        cases = ledger.cases[start : start + batch_size]
        clean, valid, selected, corrupted, lengths = _materialize_batch(
            cases,
            rows,
            total_levels=ledger.levels,
        )
        levels = np.asarray([case.level for case in cases], dtype=np.int64)
        provider_inputs = [corrupted.copy(), valid.copy(), levels.copy(), lengths.copy()]
        for value in provider_inputs:
            value.flags.writeable = False
        logits = logit_provider(*provider_inputs)
        probabilities, log_probabilities = _logit_probabilities(
            logits,
            expected_shape=(len(cases), corrupted.shape[1], len(ALPHABET)),
        )
        for index, case in enumerate(cases):
            metrics.append(
                _score_case(
                    case,
                    clean[index],
                    selected[index],
                    probabilities[index],
                    log_probabilities[index],
                    method=method_name,
                )
            )
    return aggregate_case_metrics(ledger, metrics)


def evaluate_count_baseline(
    ledger: ValidationCaseLedger,
    validation_rows: Sequence[DiffusionCorpusRow],
    suite: CountBaselineSuite,
    *,
    method: str,
    batch_size: int = EVALUATION_BATCH_SEQUENCES,
    require_locked_batch_size: bool = True,
) -> EvaluationResult:
    """Evaluate one frozen train-only count control on identical corruptions."""

    method_name = _method(method)
    if method_name not in suite.names:
        raise ValueError(f"unknown count baseline: {method_name}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(require_locked_batch_size, bool):
        raise TypeError("require_locked_batch_size must be boolean")
    if require_locked_batch_size:
        _require_locked_evaluation(ledger, batch_size=batch_size)
        if suite.training_projection_sha256 != ACCEPTED_TRAINING_PROJECTION_SHA256:
            raise ValueError("locked count baseline requires the accepted training projection")
    rows = _evaluation_rows(ledger, validation_rows)
    metrics: list[CaseMetrics] = []
    for start in range(0, len(ledger.cases), batch_size):
        cases = ledger.cases[start : start + batch_size]
        clean, valid, selected, corrupted, lengths = _materialize_batch(
            cases,
            rows,
            total_levels=ledger.levels,
        )
        probabilities = suite.probabilities(method_name, corrupted, valid, lengths)
        log_probabilities = np.log(probabilities)
        for index, case in enumerate(cases):
            metrics.append(
                _score_case(
                    case,
                    clean[index],
                    selected[index],
                    probabilities[index],
                    log_probabilities[index],
                    method=method_name,
                )
            )
    return aggregate_case_metrics(ledger, metrics)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    model_method: str
    baseline_method: str
    unit: str
    unit_count: int
    replicates: int
    seed: int
    point_relative_nll_improvement: float
    lower_95: float
    upper_95: float
    draws_sha256: str

    def __post_init__(self) -> None:
        _method(self.model_method)
        _method(self.baseline_method)
        if self.unit != "union_component_id":
            raise ValueError("bootstrap unit must be union_component_id")
        if type(self.unit_count) is not int or self.unit_count <= 0:
            raise ValueError("bootstrap unit_count must be a positive integer")
        if type(self.replicates) is not int or self.replicates <= 0:
            raise ValueError("bootstrap replicates must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
        point = _finite(
            self.point_relative_nll_improvement,
            label="point_relative_nll_improvement",
        )
        lower = _finite(self.lower_95, label="lower_95")
        upper = _finite(self.upper_95, label="upper_95")
        if point > 1.0 or lower > upper or upper > 1.0 + 1e-12:
            raise ValueError("bootstrap relative-improvement interval is invalid")
        _sha256(self.draws_sha256, label="draws_sha256")

    def canonical_record(self) -> dict[str, object]:
        return {
            "baseline_method": self.baseline_method,
            "draws_sha256": self.draws_sha256,
            "lower_95": self.lower_95,
            "model_method": self.model_method,
            "point_relative_nll_improvement": self.point_relative_nll_improvement,
            "replicates": self.replicates,
            "schema_version": 1,
            "seed": self.seed,
            "unit": self.unit,
            "unit_count": self.unit_count,
            "upper_95": self.upper_95,
        }


def paired_union_component_bootstrap(
    model_rows: Sequence[RowMetrics],
    baseline_rows: Sequence[RowMetrics],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    expected_union_components: int | None = EXPECTED_VALIDATION_UNION_COMPONENTS,
) -> BootstrapResult:
    """Bootstrap paired relative NLL improvement over whole union components."""

    if type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
    if expected_union_components is not None and (
        type(expected_union_components) is not int or expected_union_components <= 0
    ):
        raise ValueError("expected_union_components must be positive or None")

    model_values = tuple(model_rows)
    baseline_values = tuple(baseline_rows)
    if any(type(row) is not RowMetrics for row in (*model_values, *baseline_values)):
        raise TypeError("bootstrap inputs must contain RowMetrics records")
    model = {row.sequence_id: row for row in model_values}
    baseline = {row.sequence_id: row for row in baseline_values}
    if not model or len(model) != len(model_values) or len(baseline) != len(baseline_values):
        raise ValueError("bootstrap rows must be non-empty with unique sequence IDs")
    if set(model) != set(baseline):
        raise ValueError("model and baseline bootstrap rows must align exactly")
    model_methods = {_method(row.method) for row in model.values()}
    baseline_methods = {_method(row.method) for row in baseline.values()}
    if len(model_methods) != 1 or len(baseline_methods) != 1:
        raise ValueError("each bootstrap side must use exactly one method")
    for sequence_id in model:
        left, right = model[sequence_id], baseline[sequence_id]
        if (
            left.union_component_id != right.union_component_id
            or left.homology_component_id != right.homology_component_id
            or left.sampling_weight != right.sampling_weight
        ):
            raise ValueError("paired bootstrap metadata differs between methods")
        for value in (left.mean_nll, right.mean_nll):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("bootstrap NLL values must be non-negative and finite")
        if not math.isfinite(left.sampling_weight) or left.sampling_weight <= 0.0:
            raise ValueError("bootstrap weights must be positive and finite")
    if not math.isclose(
        math.fsum(row.sampling_weight for row in model.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("bootstrap sampling weights must sum to one")

    union_ids = tuple(sorted({row.union_component_id for row in model.values()}))
    if expected_union_components is not None and len(union_ids) != expected_union_components:
        raise ValueError("bootstrap union-component census differs from the frozen contract")
    grouped: dict[str, list[str]] = defaultdict(list)
    for sequence_id, row in model.items():
        grouped[row.union_component_id].append(sequence_id)
    for sequence_ids in grouped.values():
        sequence_ids.sort()

    def estimate(multiplicities: Mapping[str, int] | None = None) -> float:
        model_numerator = 0.0
        baseline_numerator = 0.0
        mass = 0.0
        for union_id in union_ids:
            multiplier = 1 if multiplicities is None else multiplicities.get(union_id, 0)
            if multiplier == 0:
                continue
            for sequence_id in grouped[union_id]:
                model_row = model[sequence_id]
                baseline_row = baseline[sequence_id]
                weight = multiplier * model_row.sampling_weight
                mass += weight
                model_numerator += weight * model_row.mean_nll
                baseline_numerator += weight * baseline_row.mean_nll
        if mass <= 0.0 or baseline_numerator <= 0.0:
            raise RuntimeError("bootstrap replicate has invalid probability mass")
        model_nll = model_numerator / mass
        baseline_nll = baseline_numerator / mass
        return (baseline_nll - model_nll) / baseline_nll

    samples = np.empty(replicates, dtype=np.float64)
    draws_digest = hashlib.sha256()
    draws_digest.update(b"amp-native-diffusion-union-bootstrap-draws-v1\0")
    for replicate in range(replicates):
        rng = np.random.Generator(
            np.random.PCG64DXSM(namespaced_seed(seed, "bootstrap", replicate))
        )
        draws = rng.integers(0, len(union_ids), size=len(union_ids))
        multiplicities: Counter[str] = Counter()
        for slot, draw_raw in enumerate(draws):
            union_id = union_ids[int(draw_raw)]
            multiplicities[union_id] += 1
            draws_digest.update(f"{replicate}\t{slot}\t{union_id}\n".encode("ascii"))
        samples[replicate] = estimate(multiplicities)
    if np.any(~np.isfinite(samples)):
        raise RuntimeError("bootstrap produced a non-finite improvement")
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return BootstrapResult(
        model_method=model_methods.pop(),
        baseline_method=baseline_methods.pop(),
        unit="union_component_id",
        unit_count=len(union_ids),
        replicates=replicates,
        seed=seed,
        point_relative_nll_improvement=estimate(),
        lower_95=float(lower),
        upper_95=float(upper),
        draws_sha256=draws_digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CandidateDiagnostics:
    raw_count: int
    canonical_valid_count: int
    canonical_valid_fraction: float
    unique_valid_count: int
    raw_unique_fraction: float
    exact_train_overlap_count: int
    exact_train_overlap_fraction: float
    exact_reference_overlap_count: int
    common_funnel_count: int
    common_funnel_yield_fraction: float
    top_reference_safe_count: int
    identity_70_cluster_count: int
    hill2_effective_70pct_clusters: float
    largest_70pct_cluster_fraction: float
    nearest_train_indel_ratio_q50: float
    nearest_train_indel_ratio_q90: float
    nearest_train_indel_ratio_q99: float
    residue_entropy_fraction: float
    maximum_residue_fraction: float
    top_reference_ratio_threshold: float
    identity_threshold: float
    training_sequences_sha256: str
    reference_sequences_sha256: str

    def __post_init__(self) -> None:
        count_fields = (
            "raw_count",
            "canonical_valid_count",
            "unique_valid_count",
            "exact_train_overlap_count",
            "exact_reference_overlap_count",
            "common_funnel_count",
            "top_reference_safe_count",
            "identity_70_cluster_count",
        )
        for label in count_fields:
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"candidate diagnostic {label} must be non-negative integer")
        if self.raw_count <= 0:
            raise ValueError("candidate diagnostic raw_count must be positive")
        if not (
            self.exact_train_overlap_count <= self.canonical_valid_count <= self.raw_count
            and self.unique_valid_count <= self.canonical_valid_count
            and self.exact_reference_overlap_count <= self.unique_valid_count
            and self.common_funnel_count + self.exact_reference_overlap_count
            == self.unique_valid_count
            and self.top_reference_safe_count <= self.common_funnel_count
            and self.identity_70_cluster_count <= self.common_funnel_count
        ):
            raise ValueError("candidate diagnostic counts are internally inconsistent")
        for label in (
            "canonical_valid_fraction",
            "raw_unique_fraction",
            "exact_train_overlap_fraction",
            "common_funnel_yield_fraction",
            "largest_70pct_cluster_fraction",
            "nearest_train_indel_ratio_q50",
            "nearest_train_indel_ratio_q90",
            "nearest_train_indel_ratio_q99",
            "residue_entropy_fraction",
            "maximum_residue_fraction",
            "top_reference_ratio_threshold",
            "identity_threshold",
        ):
            _bounded_float(getattr(self, label), label=label, lower=0.0, upper=1.0)
        for label, observed, numerator in (
            (
                "canonical_valid_fraction",
                self.canonical_valid_fraction,
                self.canonical_valid_count,
            ),
            ("raw_unique_fraction", self.raw_unique_fraction, self.unique_valid_count),
            (
                "exact_train_overlap_fraction",
                self.exact_train_overlap_fraction,
                self.exact_train_overlap_count,
            ),
            (
                "common_funnel_yield_fraction",
                self.common_funnel_yield_fraction,
                self.common_funnel_count,
            ),
        ):
            if not math.isclose(
                observed,
                numerator / self.raw_count,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"candidate diagnostic {label} disagrees with its count")
        if not (
            self.nearest_train_indel_ratio_q50
            <= self.nearest_train_indel_ratio_q90
            <= self.nearest_train_indel_ratio_q99
        ):
            raise ValueError("nearest-train ratio quantiles must be monotone")
        effective = _finite(
            self.hill2_effective_70pct_clusters,
            label="hill2_effective_70pct_clusters",
        )
        if self.common_funnel_count == 0:
            if any(
                value != 0
                for value in (
                    self.identity_70_cluster_count,
                    effective,
                    self.largest_70pct_cluster_fraction,
                    self.residue_entropy_fraction,
                    self.maximum_residue_fraction,
                )
            ):
                raise ValueError("empty common funnel must have zero diversity statistics")
        elif not 1.0 <= effective <= self.identity_70_cluster_count + 1e-12:
            raise ValueError("Hill-2 effective clusters are inconsistent with cluster count")
        _sha256(self.training_sequences_sha256, label="training_sequences_sha256")
        _sha256(self.reference_sequences_sha256, label="reference_sequences_sha256")

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }


def _canonical_sequence_set(
    sequences: Sequence[str],
    *,
    label: str,
    challenge_lengths: bool,
    deduplicate: bool = True,
) -> tuple[str, ...]:
    values: list[str] = []
    for sequence in sequences:
        if type(sequence) is not str:
            raise TypeError(f"{label} must contain strings")
        canonical = canonicalize_sequence(
            sequence,
            min_length=8 if challenge_lengths else 1,
            max_length=50 if challenge_lengths else 10**9,
        )
        if canonical != sequence:
            raise ValueError(f"{label} must contain canonical uppercase sequences")
        values.append(canonical)
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return tuple(sorted(set(values) if deduplicate else values))


def _sequence_collection_sha256(values: Sequence[str], *, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(_SEQUENCE_COLLECTION_DOMAIN)
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    for sequence in values:
        digest.update(_canonical_json_bytes({"sequence": sequence}))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DistributionCandidatePool:
    """The frozen deduplicated, canonical, exact-reference-safe comparison pool."""

    sequences: tuple[str, ...]
    raw_count: int
    canonical_valid_count: int
    exact_reference_removed_count: int
    reference_sequences_sha256: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.sequences) is not tuple or not self.sequences:
            raise ValueError("distribution candidate pool cannot be empty")
        if self.sequences != tuple(sorted(set(self.sequences))):
            raise ValueError("distribution candidate pool must be sorted and deduplicated")
        if any(
            type(sequence) is not str or canonicalize_sequence(sequence) != sequence
            for sequence in self.sequences
        ):
            raise ValueError("distribution candidate pool contains a noncanonical sequence")
        for label in (
            "raw_count",
            "canonical_valid_count",
            "exact_reference_removed_count",
        ):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"distribution pool {label} must be non-negative integer")
        if not (
            self.raw_count
            >= self.canonical_valid_count
            >= len(self.sequences) + self.exact_reference_removed_count
        ):
            raise ValueError("distribution candidate pool census is inconsistent")
        _sha256(self.reference_sequences_sha256, label="reference_sequences_sha256")
        _sha256(self.sha256, label="distribution candidate pool sha256")
        expected = _sequence_collection_sha256(self.sequences, label="candidate_pool")
        if self.sha256 != expected:
            raise ValueError("distribution candidate pool SHA-256 disagrees with its sequences")

    def canonical_record(self) -> dict[str, object]:
        return {
            "canonical_valid_count": self.canonical_valid_count,
            "exact_reference_removed_count": self.exact_reference_removed_count,
            "raw_count": self.raw_count,
            "reference_sequences_sha256": self.reference_sequences_sha256,
            "schema_version": 1,
            "sequence_count": len(self.sequences),
            "sha256": self.sha256,
        }


def build_distribution_candidate_pool(
    candidates: Sequence[str],
    *,
    reference_sequences: Sequence[str],
) -> DistributionCandidatePool:
    """Apply only the predeclared distribution-comparison funnel."""

    raw = tuple(candidates)
    if not raw:
        raise ValueError("distribution candidate pool requires raw proposals")
    if any(type(sequence) is not str for sequence in raw):
        raise TypeError("distribution candidate candidates must contain strings")
    references = _canonical_sequence_set(
        reference_sequences,
        label="reference_sequences",
        challenge_lengths=False,
    )
    reference_set = set(references)
    valid: list[str] = []
    for sequence in raw:
        try:
            canonical = canonicalize_sequence(sequence)
        except (TypeError, ValueError):
            continue
        if canonical == sequence:
            valid.append(sequence)
    unique = tuple(sorted(set(valid)))
    safe = tuple(sequence for sequence in unique if sequence not in reference_set)
    if not safe:
        raise ValueError("distribution candidate funnel is empty")
    digest = _sequence_collection_sha256(safe, label="candidate_pool")
    return DistributionCandidatePool(
        sequences=safe,
        raw_count=len(raw),
        canonical_valid_count=len(valid),
        exact_reference_removed_count=len(unique) - len(safe),
        reference_sequences_sha256=_sequence_collection_sha256(
            references,
            label="reference",
        ),
        sha256=digest,
    )


def _max_ratio(sequence: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    match = process.extractOne(
        sequence,
        references,
        scorer=levenshtein_ratio,
    )
    if match is None:  # pragma: no cover - guarded by the nonempty check above
        return 0.0
    return float(match[1])


def _cluster_sequences_exact(
    sequences: Sequence[str],
    *,
    identity_threshold: float,
) -> tuple[tuple[str, ...], ...]:
    """Match the frozen exact clustering with a lossless cheap prefilter.

    Any alignment's matches form a common subsequence.  Consequently, Indel
    similarity (twice the LCS length divided by the two sequence lengths) is
    at least that alignment's matches/alignment-length identity.  A pair whose
    Indel similarity is below the threshold therefore cannot be an edge in the
    exact global-alignment graph.  The small downward margin makes the filter
    conservative at floating-point boundaries; every surviving pair still
    uses the original exact identity implementation and tie rules.
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
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    indel_cutoff = max(0.0, identity_threshold - _CLUSTER_PREFILTER_MARGIN)
    for left_index, left in enumerate(canonical):
        for right_index in range(left_index + 1, len(canonical)):
            right = canonical[right_index]
            length_upper_bound = min(len(left), len(right)) / max(len(left), len(right))
            if length_upper_bound < identity_threshold:
                continue
            if float(levenshtein_ratio(left, right)) < indel_cutoff:
                continue
            if find(left_index) == find(right_index):
                continue
            if global_sequence_identity(left, right) >= identity_threshold:
                union(left_index, right_index)

    components: dict[int, list[str]] = {}
    for index, sequence in enumerate(canonical):
        components.setdefault(find(index), []).append(sequence)
    result = [tuple(sorted(component)) for component in components.values()]
    return tuple(sorted(result, key=lambda component: component[0]))


def candidate_diagnostics(
    candidates: Sequence[str],
    *,
    train_sequences: TrainingDistribution | Sequence[str],
    reference_sequences: Sequence[str],
    top_reference_ratio: float = TOP_REFERENCE_RATIO,
    identity_threshold: float = IDENTITY_THRESHOLD,
    require_locked_protocol: bool = True,
) -> CandidateDiagnostics:
    """Audit raw proposals without deduplication, retry, oracle, or ensemble access."""

    raw = tuple(candidates)
    if not raw:
        raise ValueError("candidate diagnostics require at least one raw proposal")
    if any(type(sequence) is not str for sequence in raw):
        raise TypeError("candidates must contain strings")
    if type(top_reference_ratio) is not float or not 0.0 <= top_reference_ratio <= 1.0:
        raise ValueError("top_reference_ratio must be a float in [0, 1]")
    if type(identity_threshold) is not float or not 0.0 <= identity_threshold <= 1.0:
        raise ValueError("identity_threshold must be a float in [0, 1]")
    if not isinstance(require_locked_protocol, bool):
        raise TypeError("require_locked_protocol must be boolean")
    if require_locked_protocol and (
        top_reference_ratio != TOP_REFERENCE_RATIO or identity_threshold != IDENTITY_THRESHOLD
    ):
        raise ValueError("locked candidate diagnostics require thresholds 0.80 and 0.70")
    if isinstance(train_sequences, TrainingDistribution):
        training_rows, _ = _validated_training_projection(
            train_sequences,
            require_locked_census=require_locked_protocol,
        )
        training = tuple(row.sequence for row in training_rows)
    else:
        if require_locked_protocol:
            raise TypeError("locked candidate diagnostics require a TrainingDistribution")
        training = _canonical_sequence_set(
            train_sequences,
            label="train_sequences",
            challenge_lengths=True,
        )
    references = _canonical_sequence_set(
        reference_sequences,
        label="reference_sequences",
        challenge_lengths=False,
    )
    training_set = set(training)
    reference_set = set(references)

    valid: list[str] = []
    for sequence in raw:
        try:
            canonical = canonicalize_sequence(sequence)
        except (TypeError, ValueError):
            continue
        if canonical == sequence:
            valid.append(sequence)
    unique_valid = tuple(sorted(set(valid)))
    exact_train_count = sum(sequence in training_set for sequence in valid)
    exact_reference_count = sum(sequence in reference_set for sequence in unique_valid)
    common = tuple(sequence for sequence in unique_valid if sequence not in reference_set)

    reference_ratios = {sequence: _max_ratio(sequence, references) for sequence in common}
    top_safe_count = sum(value <= top_reference_ratio for value in reference_ratios.values())
    nearest_train = np.asarray(
        [_max_ratio(sequence, training) for sequence in common],
        dtype=np.float64,
    )
    if len(nearest_train):
        q50, q90, q99 = np.quantile(nearest_train, [0.50, 0.90, 0.99], method="linear")
    else:
        q50 = q90 = q99 = 0.0

    components = (
        _cluster_sequences_exact(common, identity_threshold=identity_threshold) if common else ()
    )
    component_sizes = np.asarray([len(component) for component in components], dtype=np.float64)
    if len(component_sizes):
        proportions = component_sizes / np.sum(component_sizes)
        effective_clusters = float(1.0 / np.sum(np.square(proportions)))
        largest_cluster_fraction = float(np.max(proportions))
    else:
        effective_clusters = 0.0
        largest_cluster_fraction = 0.0

    residue_counts = Counter("".join(common))
    residue_total = sum(residue_counts.values())
    if residue_total:
        residue_probabilities = np.asarray(
            [count / residue_total for count in residue_counts.values()],
            dtype=np.float64,
        )
        entropy_fraction = float(
            -np.sum(residue_probabilities * np.log2(residue_probabilities)) / math.log2(20)
        )
        maximum_residue_fraction = max(residue_counts.values()) / residue_total
    else:
        entropy_fraction = 0.0
        maximum_residue_fraction = 0.0

    raw_count = len(raw)
    result = CandidateDiagnostics(
        raw_count=raw_count,
        canonical_valid_count=len(valid),
        canonical_valid_fraction=len(valid) / raw_count,
        unique_valid_count=len(unique_valid),
        raw_unique_fraction=len(unique_valid) / raw_count,
        exact_train_overlap_count=exact_train_count,
        exact_train_overlap_fraction=exact_train_count / raw_count,
        exact_reference_overlap_count=exact_reference_count,
        common_funnel_count=len(common),
        common_funnel_yield_fraction=len(common) / raw_count,
        top_reference_safe_count=top_safe_count,
        identity_70_cluster_count=len(components),
        hill2_effective_70pct_clusters=effective_clusters,
        largest_70pct_cluster_fraction=largest_cluster_fraction,
        nearest_train_indel_ratio_q50=float(q50),
        nearest_train_indel_ratio_q90=float(q90),
        nearest_train_indel_ratio_q99=float(q99),
        residue_entropy_fraction=entropy_fraction,
        maximum_residue_fraction=maximum_residue_fraction,
        top_reference_ratio_threshold=top_reference_ratio,
        identity_threshold=identity_threshold,
        training_sequences_sha256=_sequence_collection_sha256(
            training,
            label="training",
        ),
        reference_sequences_sha256=_sequence_collection_sha256(
            references,
            label="reference",
        ),
    )
    for label, value in result.canonical_record().items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"candidate diagnostic {label} is non-finite")
    return result


def ngram_jensen_shannon(
    left_sequences: Sequence[str],
    right_sequences: Sequence[str],
    *,
    order: int,
    left_weights: Sequence[float] | None = None,
    right_weights: Sequence[float] | None = None,
) -> float:
    """Return peptide-weighted base-2 Jensen--Shannon n-gram divergence.

    A peptide's probability mass is divided equally among its available
    n-grams, preventing long sequences from receiving unintended extra weight.
    """

    if type(order) is not int or order <= 0:
        raise ValueError("ngram order must be a positive integer")

    def weighted_sequences(
        sequences: Sequence[str],
        weights: Sequence[float] | None,
        *,
        label: str,
    ) -> tuple[tuple[str, float], ...]:
        values = tuple(sequences)
        if not values:
            raise ValueError(f"{label} cannot be empty")
        canonical: list[str] = []
        for sequence in values:
            if type(sequence) is not str:
                raise TypeError(f"{label} must contain strings")
            normalized = canonicalize_sequence(sequence)
            if normalized != sequence:
                raise ValueError(f"{label} must contain canonical uppercase sequences")
            if len(sequence) < order:
                raise ValueError("ngram order exceeds a sequence length")
            canonical.append(sequence)
        if weights is None:
            probabilities = (1.0 / len(canonical),) * len(canonical)
        else:
            probabilities = tuple(weights)
            if len(probabilities) != len(canonical) or any(
                type(weight) is not float or not math.isfinite(weight) or weight <= 0.0
                for weight in probabilities
            ):
                raise ValueError(f"{label} weights must be aligned positive Python floats")
            if not math.isclose(
                math.fsum(probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{label} weights must sum to one")
        return tuple(sorted(zip(canonical, probabilities, strict=True)))

    left = weighted_sequences(left_sequences, left_weights, label="left_sequences")
    right = weighted_sequences(right_sequences, right_weights, label="right_sequences")

    def masses(values: Sequence[tuple[str, float]]) -> dict[str, float]:
        contributions: dict[str, list[float]] = defaultdict(list)
        for sequence, sequence_weight in values:
            available = len(sequence) - order + 1
            contribution = sequence_weight / available
            for index in range(available):
                contributions[sequence[index : index + order]].append(contribution)
        result = {ngram: math.fsum(values) for ngram, values in contributions.items()}
        if not math.isclose(
            math.fsum(result.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("weighted n-gram probability mass does not sum to one")
        return result

    left_counts = masses(left)
    right_counts = masses(right)
    support = tuple(sorted(set(left_counts) | set(right_counts)))
    left_probability = np.asarray(
        [left_counts.get(value, 0.0) for value in support], dtype=np.float64
    )
    right_probability = np.asarray(
        [right_counts.get(value, 0.0) for value in support], dtype=np.float64
    )
    midpoint = 0.5 * (left_probability + right_probability)

    def kl(values: NDArray[np.float64]) -> float:
        selected = values > 0.0
        return float(np.sum(values[selected] * np.log2(values[selected] / midpoint[selected])))

    result = 0.5 * (kl(left_probability) + kl(right_probability))
    if not math.isfinite(result) or not -1e-12 <= result <= 1.0 + 1e-12:
        raise RuntimeError("ngram Jensen--Shannon divergence is invalid")
    return min(max(result, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class NgramDistanceResult:
    order: int
    bits: float
    candidate_count: int
    reference_count: int
    candidate_pool_sha256: str
    training_projection_sha256: str
    candidate_weighting: str
    reference_weighting: str

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order not in {1, 3}:
            raise ValueError("sampling n-gram order must be 1 or 3")
        _bounded_float(self.bits, label="ngram JSD bits", lower=0.0, upper=1.0)
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            raise ValueError("ngram candidate_count must be positive")
        if type(self.reference_count) is not int or self.reference_count <= 0:
            raise ValueError("ngram reference_count must be positive")
        _sha256(self.candidate_pool_sha256, label="candidate_pool_sha256")
        _sha256(self.training_projection_sha256, label="training_projection_sha256")
        if self.candidate_weighting != "equal_weight_divided_by_available_ngrams":
            raise ValueError("unexpected candidate n-gram weighting")
        if self.reference_weighting != "sampling_weight_divided_by_available_ngrams":
            raise ValueError("unexpected reference n-gram weighting")

    def canonical_record(self) -> dict[str, object]:
        return {
            "bits": self.bits,
            "candidate_count": self.candidate_count,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "candidate_weighting": self.candidate_weighting,
            "order": self.order,
            "reference_count": self.reference_count,
            "reference_weighting": self.reference_weighting,
            "schema_version": 1,
            "training_projection_sha256": self.training_projection_sha256,
        }


def sampling_ngram_jensen_shannon(
    pool: DistributionCandidatePool,
    training: TrainingDistribution | Sequence[TrainingRow],
    *,
    order: int,
    require_locked_census: bool = True,
) -> NgramDistanceResult:
    """Compare the frozen candidate funnel with the weighted training reference."""

    if type(pool) is not DistributionCandidatePool:
        raise TypeError("pool must be a DistributionCandidatePool")
    if require_locked_census and order not in {1, 3}:
        raise ValueError("locked sampling comparison permits only 1-mers and 3-mers")
    rows, projection_sha256 = _validated_training_projection(
        training,
        require_locked_census=require_locked_census,
    )
    bits = ngram_jensen_shannon(
        pool.sequences,
        [row.sequence for row in rows],
        order=order,
        right_weights=[row.sampling_weight for row in rows],
    )
    return NgramDistanceResult(
        order=order,
        bits=bits,
        candidate_count=len(pool.sequences),
        reference_count=len(rows),
        candidate_pool_sha256=pool.sha256,
        training_projection_sha256=projection_sha256,
        candidate_weighting="equal_weight_divided_by_available_ngrams",
        reference_weighting="sampling_weight_divided_by_available_ngrams",
    )


def _descriptor_matrix(sequences: Sequence[str]) -> NDArray[np.float64]:
    matrix = np.empty((len(sequences), len(DESCRIPTOR_FEATURE_NAMES)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        values = compute_descriptors(sequence)
        matrix[row] = tuple(float(getattr(values, name)) for name in DESCRIPTOR_FEATURE_NAMES)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("descriptor matrix contains a non-finite value")
    return matrix


def _weighted_feature_moments(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.asarray(
        [
            math.fsum(float(weights[row] * values[row, column]) for row in range(len(values)))
            for column in range(values.shape[1])
        ],
        dtype=np.float64,
    )
    variance = np.asarray(
        [
            math.fsum(
                float(weights[row] * (values[row, column] - mean[column]) ** 2)
                for row in range(len(values))
            )
            for column in range(values.shape[1])
        ],
        dtype=np.float64,
    )
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), DESCRIPTOR_SCALE_FLOOR)
    return mean, scale


def _weighted_pairwise_distance_expectation(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    left_weights: NDArray[np.float64],
    right_weights: NDArray[np.float64],
) -> float:
    contributions: list[float] = []
    for left_start in range(0, len(left), DESCRIPTOR_DISTANCE_CHUNK_SIZE):
        left_stop = min(len(left), left_start + DESCRIPTOR_DISTANCE_CHUNK_SIZE)
        for right_start in range(0, len(right), DESCRIPTOR_DISTANCE_CHUNK_SIZE):
            right_stop = min(len(right), right_start + DESCRIPTOR_DISTANCE_CHUNK_SIZE)
            difference = (
                left[left_start:left_stop, None, :] - right[None, right_start:right_stop, :]
            )
            distances = np.sqrt(np.sum(np.square(difference), axis=2))
            pair_weights = (
                left_weights[left_start:left_stop, None]
                * right_weights[None, right_start:right_stop]
            )
            contributions.append(float(np.sum(distances * pair_weights)))
    result = math.fsum(contributions)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError("pairwise descriptor expectation is invalid")
    return result


@dataclass(frozen=True, slots=True)
class DescriptorEnergyDistanceResult:
    distance: float
    candidate_count: int
    reference_count: int
    candidate_pool_sha256: str
    training_projection_sha256: str
    feature_names: tuple[str, ...]
    training_mean: tuple[float, ...]
    training_scale: tuple[float, ...]
    scale_floor: float
    standardization: str
    reference_population: str
    estimator: str

    def __post_init__(self) -> None:
        if _finite(self.distance, label="descriptor energy distance") < 0.0:
            raise ValueError("descriptor energy distance must be non-negative")
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            raise ValueError("descriptor candidate_count must be positive")
        if type(self.reference_count) is not int or self.reference_count <= 0:
            raise ValueError("descriptor reference_count must be positive")
        _sha256(self.candidate_pool_sha256, label="candidate_pool_sha256")
        _sha256(self.training_projection_sha256, label="training_projection_sha256")
        if self.feature_names != DESCRIPTOR_FEATURE_NAMES:
            raise ValueError("descriptor feature list differs from the frozen 13 features")
        if type(self.training_mean) is not tuple or type(self.training_scale) is not tuple:
            raise TypeError("descriptor moments must be canonical tuples")
        if len(self.training_mean) != len(self.feature_names) or len(self.training_scale) != len(
            self.feature_names
        ):
            raise ValueError("descriptor moments must align with the feature list")
        for value in self.training_mean:
            _finite(value, label="descriptor training mean")
        for value in self.training_scale:
            if _finite(value, label="descriptor training scale") < DESCRIPTOR_SCALE_FLOOR:
                raise ValueError("descriptor scale is below the frozen floor")
        if self.scale_floor != DESCRIPTOR_SCALE_FLOOR:
            raise ValueError("descriptor scale floor differs from 1e-12")
        if self.standardization != DESCRIPTOR_SCALING:
            raise ValueError("descriptor standardization differs from the frozen protocol")
        if self.reference_population != DESCRIPTOR_REFERENCE_POPULATION:
            raise ValueError("descriptor reference population differs from the frozen protocol")
        if self.estimator != DESCRIPTOR_ENERGY_ESTIMATOR:
            raise ValueError("descriptor estimator differs from the frozen protocol")

    def canonical_record(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "distance": self.distance,
            "estimator": self.estimator,
            "feature_names": list(self.feature_names),
            "reference_count": self.reference_count,
            "reference_population": self.reference_population,
            "scale_floor": self.scale_floor,
            "schema_version": 1,
            "standardization": self.standardization,
            "training_mean": list(self.training_mean),
            "training_projection_sha256": self.training_projection_sha256,
            "training_scale": list(self.training_scale),
        }


def descriptor_energy_distance(
    pool: DistributionCandidatePool,
    training: TrainingDistribution | Sequence[TrainingRow],
    *,
    require_locked_census: bool = True,
) -> DescriptorEnergyDistanceResult:
    """Compute the frozen weighted 13-descriptor Euclidean energy distance."""

    if type(pool) is not DistributionCandidatePool:
        raise TypeError("pool must be a DistributionCandidatePool")
    rows, projection_sha256 = _validated_training_projection(
        training,
        require_locked_census=require_locked_census,
    )
    candidate_sequences = tuple(sorted(pool.sequences))
    reference_sequences = tuple(row.sequence for row in rows)
    candidate = _descriptor_matrix(candidate_sequences)
    reference = _descriptor_matrix(reference_sequences)
    reference_weights = np.asarray([row.sampling_weight for row in rows], dtype=np.float64)
    candidate_weights = np.full(len(candidate), 1.0 / len(candidate), dtype=np.float64)
    mean, scale = _weighted_feature_moments(reference, reference_weights)
    candidate_standardized = (candidate - mean) / scale
    reference_standardized = (reference - mean) / scale
    cross = _weighted_pairwise_distance_expectation(
        candidate_standardized,
        reference_standardized,
        candidate_weights,
        reference_weights,
    )
    within_candidate = _weighted_pairwise_distance_expectation(
        candidate_standardized,
        candidate_standardized,
        candidate_weights,
        candidate_weights,
    )
    within_reference = _weighted_pairwise_distance_expectation(
        reference_standardized,
        reference_standardized,
        reference_weights,
        reference_weights,
    )
    raw_distance = 2.0 * cross - within_candidate - within_reference
    if not math.isfinite(raw_distance) or raw_distance < -1e-10:
        raise RuntimeError("descriptor energy distance is materially negative")
    return DescriptorEnergyDistanceResult(
        distance=max(0.0, raw_distance),
        candidate_count=len(candidate),
        reference_count=len(reference),
        candidate_pool_sha256=pool.sha256,
        training_projection_sha256=projection_sha256,
        feature_names=DESCRIPTOR_FEATURE_NAMES,
        training_mean=tuple(float(value) for value in mean),
        training_scale=tuple(float(value) for value in scale),
        scale_floor=DESCRIPTOR_SCALE_FLOOR,
        standardization=DESCRIPTOR_SCALING,
        reference_population=DESCRIPTOR_REFERENCE_POPULATION,
        estimator=DESCRIPTOR_ENERGY_ESTIMATOR,
    )

"""Deterministic, score-free candidate-pool construction for AMP Challenge.

The v1 pool is an input to later ensemble inference and acquisition, not a
scientifically promoted portfolio.  It combines the transparent baseline with
two train-only count controls, records every generation lineage, and opens the
organizer reference only after all raw proposals have been generated.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.generators.baseline import FAMILIES, generate_baseline_pool
from amp_challenge.generators.diffusion.data import (
    ACCEPTED_TRAINING_PROJECTION_SHA256,
    TrainingDistribution,
    load_training_projection,
)
from amp_challenge.generators.diffusion.evaluation import (
    GENERATOR_CONTROL_METHODS,
    fit_count_baselines,
    sample_count_generator_control_v0,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

SCHEMA_VERSION = 1
POOL_NAME = "candidate_pool_v1"
POOL_STATUS = "candidate_input_only_no_scientific_promotion"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MINIMUM_LENGTH = 8
MAXIMUM_LENGTH = 50
BASELINE_GENERATOR = "transparent_physicochemical_baseline_v0"
COUNT_GENERATOR_FAMILY = "train_only_count_control_v0"
ORGANIZER_REFERENCE_ROLE = "compliance_only"
ACCEPTED_TRAINING_PROJECTION_ROWS = 914
ACCEPTED_TRAINING_PROJECTION_SIZE_BYTES = 146_365
ACCEPTED_ORGANIZER_REFERENCE_SHA256 = (
    "cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9"
)
ACCEPTED_ORGANIZER_REFERENCE_RECORDS = 39_448
ACCEPTED_ORGANIZER_REFERENCE_SIZE_BYTES = 4_318_298
PRODUCTION_BASELINE_CANDIDATES = 50_000
PRODUCTION_BASELINE_SEED = 42
PRODUCTION_COUNT_CANDIDATES_PER_METHOD = 32_768
PRODUCTION_COUNT_SEED = 42
PRODUCTION_COUNT_BATCH_SIZE = 256
PRODUCTION_LENGTH_PLAN_NAMESPACE = "candidate_pool_v1_shared_train_only_length_plan"
PRODUCTION_MINIMUM_UNIQUE_CANDIDATES = 50_000
PRODUCTION_CONFIG_SHA256 = "a9b6514f8fe146badac3fcf2502b5aed2676d7a8d74dfd66dd0a0f4910b63163"
PRODUCTION_CONFIG_SIZE_BYTES = 1_146

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_BASELINE_SPEC_DOMAIN = b"amp-challenge/candidate-pool/baseline-spec/v1\0"
_BASELINE_FAMILY_NAMES = frozenset(family.name for family in FAMILIES)
_OUTPUT_NAMES = (
    "candidates.fasta",
    "candidates.jsonl",
    "family_metrics.json",
    "manifest.json",
    "SHA256SUMS",
)
_CANDIDATE_FIELDS = (
    "schema_version",
    "sequence_id",
    "sequence",
    "length",
    "library_eligible",
    "lineages",
)
_LINEAGE_FIELDS = (
    "schema_version",
    "generator_family",
    "generator_variant",
    "logical_sha256",
    "training_projection_sha256",
    "seed",
    "ordinal",
)


@dataclass(frozen=True, slots=True)
class CandidatePoolConfig:
    """Strict, flattened representation of candidate-pool TOML v1."""

    training_projection_sha256: str
    training_projection_rows: int
    require_accepted_training_projection: bool
    organizer_reference_sha256: str
    organizer_reference_records: int
    baseline_candidates: int
    baseline_seed: int
    count_candidates_per_method: int
    count_seed: int
    count_batch_size: int
    length_plan_namespace: str
    minimum_unique_candidates: int


@dataclass(frozen=True, slots=True)
class _Snapshot:
    payload: bytes
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int
    link_count: int
    owner_uid: int
    owner_gid: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class CandidateLineage:
    """One transparent generation route that yielded a candidate sequence."""

    generator_family: str
    generator_variant: str
    logical_sha256: str
    training_projection_sha256: str | None
    seed: int
    ordinal: int

    def __post_init__(self) -> None:
        _manifest_name(self.generator_family, label="lineage generator_family")
        _manifest_name(self.generator_variant, label="lineage generator_variant")
        _sha256(self.logical_sha256, label="lineage logical_sha256")
        if self.training_projection_sha256 is not None:
            _sha256(
                self.training_projection_sha256,
                label="lineage training_projection_sha256",
            )
        _uint64(self.seed, label="lineage seed")
        _uint64(self.ordinal, label="lineage ordinal")
        if self.generator_family == BASELINE_GENERATOR:
            if self.training_projection_sha256 is not None:
                raise ValueError("transparent baseline lineage cannot claim training access")
            if self.generator_variant not in _BASELINE_FAMILY_NAMES:
                raise ValueError("transparent baseline lineage has an unknown family")
        elif self.generator_family == COUNT_GENERATOR_FAMILY:
            if self.generator_variant not in GENERATOR_CONTROL_METHODS:
                raise ValueError("count-control lineage has an unknown generator variant")
            if self.training_projection_sha256 is None:
                raise ValueError("count-control lineage must bind its training projection")
        else:
            raise ValueError("candidate_pool_v1 contains an unknown generator family")

    def sort_key(self) -> tuple[str, str, str, str, int, int]:
        return (
            self.generator_family,
            self.generator_variant,
            self.logical_sha256,
            self.training_projection_sha256 or "",
            self.seed,
            self.ordinal,
        )

    def canonical_record(self) -> dict[str, object]:
        return {
            "generator_family": self.generator_family,
            "generator_variant": self.generator_variant,
            "logical_sha256": self.logical_sha256,
            "ordinal": self.ordinal,
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "training_projection_sha256": self.training_projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class RawCandidateProposal:
    """One unfiltered sequence and its generation lineage."""

    sequence: str
    lineage: CandidateLineage

    def __post_init__(self) -> None:
        if type(self.sequence) is not str:
            raise TypeError("raw proposal sequence must be a string")
        if canonicalize_sequence(self.sequence) != self.sequence:
            raise ValueError("raw proposal sequence must already be canonical")
        if type(self.lineage) is not CandidateLineage:
            raise TypeError("raw proposal lineage must be CandidateLineage")


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One eligible union sequence with all distinct generation lineages."""

    sequence: str
    lineages: tuple[CandidateLineage, ...]

    def __post_init__(self) -> None:
        if type(self.sequence) is not str or canonicalize_sequence(self.sequence) != self.sequence:
            raise ValueError("candidate sequence must be canonical")
        if (
            type(self.lineages) is not tuple
            or not self.lineages
            or any(type(lineage) is not CandidateLineage for lineage in self.lineages)
        ):
            raise ValueError("candidate must retain a non-empty lineage tuple")
        expected = tuple(sorted(set(self.lineages), key=CandidateLineage.sort_key))
        if self.lineages != expected:
            raise ValueError("candidate lineages must be distinct and canonically ordered")

    def canonical_record(self) -> dict[str, object]:
        return {
            "length": len(self.sequence),
            "library_eligible": True,
            "lineages": [lineage.canonical_record() for lineage in self.lineages],
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "sequence_id": canonical_sequence_id(self.sequence),
        }


@dataclass(frozen=True, slots=True)
class CandidateAssembly:
    """Canonical eligible records and their score-free source accounting."""

    records: tuple[CandidateRecord, ...]
    family_metrics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CandidatePoolExecution:
    """Paths and content identities for one completed write-once publication."""

    output_dir: Path
    candidate_count: int
    raw_proposal_count: int
    candidates_jsonl_sha256: str
    candidates_fasta_sha256: str
    family_metrics_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CandidatePoolVerification:
    """Stable content identity returned by either candidate-pool verifier."""

    output_dir: Path
    candidate_count: int
    candidates_jsonl_sha256: str
    candidates_fasta_sha256: str
    family_metrics_sha256: str
    manifest_sha256: str


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_canonical_json(payload: bytes, *, label: str) -> object:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-terminated canonical JSON")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if _canonical_json_bytes(document) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return document


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _manifest_name(value: object, *, label: str) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase manifest-safe identifier")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _uint64(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _exact_mapping(
    value: object,
    *,
    label: str,
    expected_keys: Collection[str],
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    expected = set(expected_keys)
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    candidates = tuple(reversed((absolute, *absolute.parents)))
    for candidate in candidates:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} cannot traverse a symbolic link: {candidate}")


def _snapshot_regular(path: str | Path, *, label: str) -> _Snapshot:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source, label=label)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {source}") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError(f"{label} must be a regular, non-symbolic file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    identities = {
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_uid,
            item.st_gid,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (named_before, opened_before, opened_after, named_after)
    }
    _reject_symlink_chain(source, label=label)
    if len(identities) != 1:
        raise ValueError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != named_before.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return _Snapshot(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        device=named_before.st_dev,
        inode=named_before.st_ino,
        mode=named_before.st_mode,
        link_count=named_before.st_nlink,
        owner_uid=named_before.st_uid,
        owner_gid=named_before.st_gid,
        modified_ns=named_before.st_mtime_ns,
        changed_ns=named_before.st_ctime_ns,
    )


def _require_unchanged(path: str | Path, original: _Snapshot, *, label: str) -> None:
    current = _snapshot_regular(path, label=label)
    if current != original:
        raise ValueError(f"{label} changed after its authenticated snapshot")


def _parse_candidate_pool_config(payload: bytes) -> CandidatePoolConfig:
    try:
        document_raw = tomllib.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("candidate-pool config must be UTF-8") from error
    document = _exact_mapping(
        document_raw,
        label="candidate-pool config",
        expected_keys={
            "schema_version",
            "name",
            "status",
            "alphabet",
            "minimum_length",
            "maximum_length",
            "inputs",
            "generation",
            "filtering",
            "output",
        },
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("candidate-pool schema_version must be 1")
    if document["name"] != POOL_NAME:
        raise ValueError(f"candidate-pool name must be {POOL_NAME!r}")
    if document["status"] != POOL_STATUS:
        raise ValueError(f"candidate-pool status must be {POOL_STATUS!r}")
    if document["alphabet"] != ALPHABET:
        raise ValueError("candidate-pool alphabet differs from the organizer alphabet")
    if type(document["minimum_length"]) is not int or document["minimum_length"] != MINIMUM_LENGTH:
        raise ValueError("candidate-pool minimum_length must be 8")
    if type(document["maximum_length"]) is not int or document["maximum_length"] != MAXIMUM_LENGTH:
        raise ValueError("candidate-pool maximum_length must be 50")

    inputs = _exact_mapping(
        document["inputs"],
        label="[inputs]",
        expected_keys={
            "training_projection_sha256",
            "training_projection_rows",
            "require_accepted_training_projection",
            "organizer_reference_sha256",
            "organizer_reference_records",
            "organizer_reference_role",
        },
    )
    generation = _exact_mapping(
        document["generation"],
        label="[generation]",
        expected_keys={
            "baseline_generator",
            "baseline_candidates",
            "baseline_seed",
            "count_generator_methods",
            "count_candidates_per_method",
            "count_seed",
            "count_batch_size",
            "length_plan_namespace",
        },
    )
    filtering = _exact_mapping(
        document["filtering"],
        label="[filtering]",
        expected_keys={
            "exclude_exact_training_overlap",
            "exclude_exact_organizer_reference_overlap",
        },
    )
    output = _exact_mapping(
        document["output"],
        label="[output]",
        expected_keys={
            "minimum_unique_candidates",
            "candidate_schema_version",
            "write_prediction_fields",
            "write_acquisition_lineage",
        },
    )

    training_sha256 = _sha256(
        inputs["training_projection_sha256"],
        label="training_projection_sha256",
    )
    training_rows = _positive_integer(
        inputs["training_projection_rows"],
        label="training_projection_rows",
    )
    require_accepted = _boolean(
        inputs["require_accepted_training_projection"],
        label="require_accepted_training_projection",
    )
    reference_sha256 = _sha256(
        inputs["organizer_reference_sha256"],
        label="organizer_reference_sha256",
    )
    reference_records = _positive_integer(
        inputs["organizer_reference_records"],
        label="organizer_reference_records",
    )
    if inputs["organizer_reference_role"] != ORGANIZER_REFERENCE_ROLE:
        raise ValueError("organizer reference role must be compliance_only")
    if generation["baseline_generator"] != BASELINE_GENERATOR:
        raise ValueError("candidate_pool_v1 requires the transparent baseline generator")
    methods_raw = generation["count_generator_methods"]
    if not isinstance(methods_raw, list) or any(type(item) is not str for item in methods_raw):
        raise ValueError("count_generator_methods must be a TOML string array")
    methods = tuple(methods_raw)
    if methods != GENERATOR_CONTROL_METHODS:
        raise ValueError("count_generator_methods must contain unigram then forward-Markov control")
    baseline_candidates = _positive_integer(
        generation["baseline_candidates"],
        label="baseline_candidates",
    )
    baseline_seed = _uint64(generation["baseline_seed"], label="baseline_seed")
    count_candidates = _positive_integer(
        generation["count_candidates_per_method"],
        label="count_candidates_per_method",
    )
    count_seed = _uint64(generation["count_seed"], label="count_seed")
    count_batch_size = _positive_integer(
        generation["count_batch_size"],
        label="count_batch_size",
    )
    namespace = _manifest_name(
        generation["length_plan_namespace"],
        label="length_plan_namespace",
    )
    if not _boolean(
        filtering["exclude_exact_training_overlap"],
        label="exclude_exact_training_overlap",
    ):
        raise ValueError("candidate_pool_v1 must exclude exact training overlap")
    if not _boolean(
        filtering["exclude_exact_organizer_reference_overlap"],
        label="exclude_exact_organizer_reference_overlap",
    ):
        raise ValueError("candidate_pool_v1 must exclude exact organizer-reference overlap")
    minimum_unique = _positive_integer(
        output["minimum_unique_candidates"],
        label="minimum_unique_candidates",
    )
    if (
        type(output["candidate_schema_version"]) is not int
        or output["candidate_schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("candidate_schema_version must be 1")
    if _boolean(output["write_prediction_fields"], label="write_prediction_fields"):
        raise ValueError("candidate_pool_v1 cannot write prediction fields")
    if _boolean(output["write_acquisition_lineage"], label="write_acquisition_lineage"):
        raise ValueError("candidate_pool_v1 cannot write acquisition lineage")
    raw_capacity = baseline_candidates + len(methods) * count_candidates
    if raw_capacity < minimum_unique:
        raise ValueError("raw proposal capacity cannot satisfy minimum_unique_candidates")

    if require_accepted:
        expected = {
            "training_projection_sha256": ACCEPTED_TRAINING_PROJECTION_SHA256,
            "training_projection_rows": ACCEPTED_TRAINING_PROJECTION_ROWS,
            "organizer_reference_sha256": ACCEPTED_ORGANIZER_REFERENCE_SHA256,
            "organizer_reference_records": ACCEPTED_ORGANIZER_REFERENCE_RECORDS,
            "baseline_candidates": PRODUCTION_BASELINE_CANDIDATES,
            "baseline_seed": PRODUCTION_BASELINE_SEED,
            "count_candidates_per_method": PRODUCTION_COUNT_CANDIDATES_PER_METHOD,
            "count_seed": PRODUCTION_COUNT_SEED,
            "count_batch_size": PRODUCTION_COUNT_BATCH_SIZE,
            "length_plan_namespace": PRODUCTION_LENGTH_PLAN_NAMESPACE,
            "minimum_unique_candidates": PRODUCTION_MINIMUM_UNIQUE_CANDIDATES,
        }
        observed = {
            "training_projection_sha256": training_sha256,
            "training_projection_rows": training_rows,
            "organizer_reference_sha256": reference_sha256,
            "organizer_reference_records": reference_records,
            "baseline_candidates": baseline_candidates,
            "baseline_seed": baseline_seed,
            "count_candidates_per_method": count_candidates,
            "count_seed": count_seed,
            "count_batch_size": count_batch_size,
            "length_plan_namespace": namespace,
            "minimum_unique_candidates": minimum_unique,
        }
        changed = sorted(key for key, value in observed.items() if value != expected[key])
        if changed:
            raise ValueError(
                "accepted candidate_pool_v1 contract differs for key(s): " + ", ".join(changed)
            )

    return CandidatePoolConfig(
        training_projection_sha256=training_sha256,
        training_projection_rows=training_rows,
        require_accepted_training_projection=require_accepted,
        organizer_reference_sha256=reference_sha256,
        organizer_reference_records=reference_records,
        baseline_candidates=baseline_candidates,
        baseline_seed=baseline_seed,
        count_candidates_per_method=count_candidates,
        count_seed=count_seed,
        count_batch_size=count_batch_size,
        length_plan_namespace=namespace,
        minimum_unique_candidates=minimum_unique,
    )


def load_candidate_pool_config(path: str | Path) -> CandidatePoolConfig:
    """Load and strictly validate the machine-readable candidate-pool contract."""

    return _parse_candidate_pool_config(
        _snapshot_regular(path, label="candidate-pool config").payload
    )


def _baseline_spec_record() -> dict[str, object]:
    return {
        "alphabet": ALPHABET,
        "generator_family": BASELINE_GENERATOR,
        "schema_version": SCHEMA_VERSION,
        "families": [
            {
                "amphipathic_pattern": family.amphipathic_pattern,
                "length_std": float(family.length_std),
                "max_length": family.max_length,
                "mean_length": float(family.mean_length),
                "min_length": family.min_length,
                "name": family.name,
                "probability": float(family.probability),
                "residue_weights": {
                    residue: float(weight)
                    for residue, weight in sorted(family.residue_weights.items())
                },
            }
            for family in FAMILIES
        ],
    }


def baseline_spec_sha256() -> str:
    """Return the semantic identity of the transparent baseline family table."""

    digest = hashlib.sha256()
    digest.update(_BASELINE_SPEC_DOMAIN)
    digest.update(_canonical_json_bytes(_baseline_spec_record()))
    return digest.hexdigest()


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(length): counts[length] for length in sorted(counts)}


def _source_key(lineage: CandidateLineage) -> tuple[str, str, str, str, int]:
    return (
        lineage.generator_family,
        lineage.generator_variant,
        lineage.logical_sha256,
        lineage.training_projection_sha256 or "",
        lineage.seed,
    )


def _training_distributions(
    sequence_weights: Mapping[str, float],
) -> tuple[set[str], tuple[float, ...], tuple[float, ...]]:
    if not sequence_weights:
        raise ValueError("training_sequence_weights cannot be empty")
    training: set[str] = set()
    length_terms: list[list[float]] = [[] for _ in range(MAXIMUM_LENGTH - MINIMUM_LENGTH + 1)]
    residue_terms: list[list[float]] = [[] for _ in ALPHABET]
    residue_index = {residue: index for index, residue in enumerate(ALPHABET)}
    weights: list[float] = []
    for sequence, weight in sequence_weights.items():
        if type(sequence) is not str or canonicalize_sequence(sequence) != sequence:
            raise ValueError("training_sequence_weights contains a noncanonical sequence")
        if type(weight) is not float or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("training sampling weights must be positive finite floats")
        training.add(sequence)
        weights.append(weight)
        length_terms[len(sequence) - MINIMUM_LENGTH].append(weight)
        residue_counts = Counter(sequence)
        for residue, count in residue_counts.items():
            residue_terms[residue_index[residue]].append(weight * count / len(sequence))
    weight_total = math.fsum(weights)
    if not math.isclose(weight_total, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("training sampling weights must sum to one")
    length_mass = tuple(math.fsum(terms) for terms in length_terms)
    residue_mass = tuple(math.fsum(terms) for terms in residue_terms)
    length_total = math.fsum(length_mass)
    residue_total = math.fsum(residue_mass)
    if not math.isclose(length_total, weight_total, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("training length distribution did not retain unit mass")
    if not math.isclose(residue_total, weight_total, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("training residue distribution did not retain unit mass")
    return (
        training,
        tuple(value / length_total for value in length_mass),
        tuple(value / residue_total for value in residue_mass),
    )


def _candidate_distributions(
    sequences: Collection[str],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    ordered = tuple(sorted(set(sequences)))
    if not ordered:
        return None
    length_mass = [0.0] * (MAXIMUM_LENGTH - MINIMUM_LENGTH + 1)
    residue_mass = [0.0] * len(ALPHABET)
    residue_index = {residue: index for index, residue in enumerate(ALPHABET)}
    sequence_mass = 1.0 / len(ordered)
    for sequence in ordered:
        if canonicalize_sequence(sequence) != sequence:
            raise ValueError("candidate distribution contains a noncanonical sequence")
        length_mass[len(sequence) - MINIMUM_LENGTH] += sequence_mass
        residue_weight = sequence_mass / len(sequence)
        for residue in sequence:
            residue_mass[residue_index[residue]] += residue_weight
    length_total = math.fsum(length_mass)
    residue_total = math.fsum(residue_mass)
    return (
        tuple(value / length_total for value in length_mass),
        tuple(value / residue_total for value in residue_mass),
    )


def _jensen_shannon_bits(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Jensen-Shannon distributions must be non-empty and aligned")
    if any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in (*first, *second)
    ):
        raise ValueError("Jensen-Shannon distributions must contain finite probabilities")
    if not math.isclose(math.fsum(first), 1.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        math.fsum(second),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Jensen-Shannon distributions must each sum to one")
    terms: list[float] = []
    for left, right in zip(first, second, strict=True):
        midpoint = 0.5 * (left + right)
        if left > 0.0:
            terms.append(0.5 * left * math.log2(left / midpoint))
        if right > 0.0:
            terms.append(0.5 * right * math.log2(right / midpoint))
    result = math.fsum(terms)
    if not -1e-15 <= result <= 1.0 + 1e-15:
        raise RuntimeError("Jensen-Shannon divergence escaped its base-2 bounds")
    return min(1.0, max(0.0, result))


def _distribution_divergences(
    sequences: Collection[str],
    *,
    training_length: Sequence[float],
    training_residue: Sequence[float],
) -> tuple[float | None, float | None]:
    candidate = _candidate_distributions(sequences)
    if candidate is None:
        return None, None
    candidate_length, candidate_residue = candidate
    return (
        _jensen_shannon_bits(candidate_length, training_length),
        _jensen_shannon_bits(candidate_residue, training_residue),
    )


def assemble_candidate_records(
    proposals: Sequence[RawCandidateProposal],
    *,
    training_sequence_weights: Mapping[str, float],
    organizer_reference_sequences: Collection[str],
    minimum_unique_candidates: int,
    pool_name: str = POOL_NAME,
) -> CandidateAssembly:
    """Filter an unscored raw union and retain every distinct source lineage."""

    _manifest_name(pool_name, label="pool_name")
    minimum = _positive_integer(
        minimum_unique_candidates,
        label="minimum_unique_candidates",
    )
    raw = tuple(proposals)
    if not raw or any(type(proposal) is not RawCandidateProposal for proposal in raw):
        raise ValueError("candidate assembly requires RawCandidateProposal values")
    training, training_length, training_residue = _training_distributions(training_sequence_weights)
    reference = set(organizer_reference_sequences)
    if any(type(sequence) is not str or not sequence for sequence in reference):
        raise ValueError("organizer reference set must contain non-empty sequence strings")

    lineages_by_sequence: dict[str, set[CandidateLineage]] = defaultdict(set)
    source_sequences: dict[tuple[str, str, str, str, int], set[str]] = defaultdict(set)
    source_raw_lengths: dict[tuple[str, str, str, str, int], list[int]] = defaultdict(list)
    source_raw_counts: Counter[tuple[str, str, str, str, int]] = Counter()
    for proposal in raw:
        lineages_by_sequence[proposal.sequence].add(proposal.lineage)
        key = _source_key(proposal.lineage)
        source_sequences[key].add(proposal.sequence)
        source_raw_lengths[key].append(len(proposal.sequence))
        source_raw_counts[key] += 1

    excluded_training = set(lineages_by_sequence) & training
    excluded_reference = set(lineages_by_sequence) & reference
    excluded = excluded_training | excluded_reference
    eligible_sequences = sorted(set(lineages_by_sequence) - excluded)
    records = tuple(
        CandidateRecord(
            sequence=sequence,
            lineages=tuple(sorted(lineages_by_sequence[sequence], key=CandidateLineage.sort_key)),
        )
        for sequence in eligible_sequences
    )
    if len(records) < minimum:
        raise ValueError(
            f"candidate pool produced {len(records)} eligible unique sequences; "
            f"minimum is {minimum}"
        )

    source_metrics: list[dict[str, object]] = []
    for key in sorted(source_sequences):
        family, variant, logical_sha256, training_sha256, seed = key
        unique = source_sequences[key]
        eligible = unique - excluded
        length_divergence, residue_divergence = _distribution_divergences(
            eligible,
            training_length=training_length,
            training_residue=training_residue,
        )
        source_metrics.append(
            {
                "accepted_union_sequences_with_lineage": len(eligible),
                "eligible_length_js_divergence_bits_vs_training": length_divergence,
                "eligible_unique_length_histogram": _histogram(
                    [len(sequence) for sequence in sorted(eligible)]
                ),
                "eligible_unique_sequences": len(eligible),
                "eligible_residue_js_divergence_bits_vs_training": residue_divergence,
                "exact_organizer_reference_overlap_unique_sequences": len(unique & reference),
                "exact_training_overlap_unique_sequences": len(unique & training),
                "generator_family": family,
                "generator_variant": variant,
                "logical_sha256": logical_sha256,
                "raw_length_histogram": _histogram(source_raw_lengths[key]),
                "raw_proposals": source_raw_counts[key],
                "raw_unique_sequences": len(unique),
                "seed": seed,
                "training_projection_sha256": training_sha256 or None,
                "within_source_duplicate_proposals": source_raw_counts[key] - len(unique),
            }
        )

    multi_lineage = sum(len(record.lineages) > 1 for record in records)
    multi_variant = sum(
        len({(item.generator_family, item.generator_variant) for item in record.lineages}) > 1
        for record in records
    )
    multi_family = sum(
        len({item.generator_family for item in record.lineages}) > 1 for record in records
    )
    union_unique = len(lineages_by_sequence)
    union_length_divergence, union_residue_divergence = _distribution_divergences(
        eligible_sequences,
        training_length=training_length,
        training_residue=training_residue,
    )
    family_metrics: dict[str, object] = {
        "distribution_reference": {
            "candidate_weighting": "uniform_over_eligible_unique_sequences",
            "divergence": "jensen_shannon_base_2_bits",
            "residue_estimator": "mean_per_sequence_residue_fraction",
            "training_length_probabilities": {
                str(length): training_length[length - MINIMUM_LENGTH]
                for length in range(MINIMUM_LENGTH, MAXIMUM_LENGTH + 1)
            },
            "training_residue_probabilities": {
                residue: training_residue[index] for index, residue in enumerate(ALPHABET)
            },
            "training_weighting": "component_weighted_sampling_weight",
        },
        "pool_name": pool_name,
        "schema_version": SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "union_metrics": {
            "duplicate_raw_proposals": len(raw) - union_unique,
            "eligible_length_js_divergence_bits_vs_training": union_length_divergence,
            "eligible_unique_candidates": len(records),
            "eligible_unique_length_histogram": _histogram(
                [len(record.sequence) for record in records]
            ),
            "exact_overlap_with_both_unique_sequences": len(excluded_training & excluded_reference),
            "exact_organizer_reference_overlap_unique_sequences": len(excluded_reference),
            "exact_training_overlap_unique_sequences": len(excluded_training),
            "excluded_unique_sequences": len(excluded),
            "eligible_residue_js_divergence_bits_vs_training": union_residue_divergence,
            "minimum_unique_candidates": minimum,
            "minimum_unique_candidates_passed": len(records) >= minimum,
            "multi_generator_family_candidates": multi_family,
            "multi_generator_variant_candidates": multi_variant,
            "multi_lineage_candidates": multi_lineage,
            "raw_proposals": len(raw),
            "raw_unique_sequences_before_filtering": union_unique,
        },
    }
    return CandidateAssembly(records=records, family_metrics=family_metrics)


def _read_reference_snapshot(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_records: int,
) -> tuple[_Snapshot, tuple[str, ...]]:
    snapshot = _snapshot_regular(path, label="organizer reference FASTA")
    if snapshot.sha256 != expected_sha256:
        raise ValueError(
            f"organizer reference FASTA SHA-256 expected {expected_sha256}, got {snapshot.sha256}"
        )
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("organizer reference FASTA must be UTF-8") from error
    sequences: list[str] = []
    active = False
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if active:
                sequences.append("".join(parts))
            active = True
            parts = []
        elif active:
            parts.append(line.upper())
    if active:
        sequences.append("".join(parts))
    if len(sequences) != expected_records:
        raise ValueError(
            f"organizer reference expected {expected_records} records, got {len(sequences)}"
        )
    if any(not sequence or set(sequence) - set(ALPHABET) for sequence in sequences):
        raise ValueError("organizer reference contains a noncanonical amino-acid sequence")
    return snapshot, tuple(sequences)


def _candidate_jsonl_bytes(records: Sequence[CandidateRecord]) -> bytes:
    return b"".join(_canonical_json_bytes(record.canonical_record()) for record in records)


def _candidate_fasta_bytes(records: Sequence[CandidateRecord]) -> bytes:
    lines: list[str] = []
    for index, record in enumerate(records, 1):
        sequence_id = canonical_sequence_id(record.sequence)
        lines.extend(
            (
                f">candidate_{index:06d} sequence_id={sequence_id} lineages={len(record.lineages)}",
                record.sequence,
            )
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _artifact_record(payload: bytes, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if records is not None:
        result["records"] = records
    return result


def _parse_candidate_records(payload: bytes) -> tuple[CandidateRecord, ...]:
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("candidates.jsonl must be non-empty canonical LF-framed JSONL")
    records: list[CandidateRecord] = []
    for number, line in enumerate(payload.splitlines(keepends=True), 1):
        document_raw = _parse_canonical_json(line, label=f"candidate row {number}")
        document = _exact_mapping(
            document_raw,
            label=f"candidate row {number}",
            expected_keys=_CANDIDATE_FIELDS,
        )
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError(f"candidate row {number} schema_version must be 1")
        if type(document["sequence"]) is not str:
            raise ValueError(f"candidate row {number} sequence must be a string")
        sequence = document["sequence"]
        sequence_id = _sha256(
            document["sequence_id"],
            label=f"candidate row {number} sequence_id",
        )
        if type(document["length"]) is not int or document["length"] != len(sequence):
            raise ValueError(f"candidate row {number} length disagrees with its sequence")
        if type(document["library_eligible"]) is not bool or not document["library_eligible"]:
            raise ValueError(f"candidate row {number} must be library eligible")
        lineages_raw = document["lineages"]
        if not isinstance(lineages_raw, list) or not lineages_raw:
            raise ValueError(f"candidate row {number} must contain generation lineages")
        lineages: list[CandidateLineage] = []
        for lineage_number, lineage_raw in enumerate(lineages_raw, 1):
            lineage = _exact_mapping(
                lineage_raw,
                label=f"candidate row {number} lineage {lineage_number}",
                expected_keys=_LINEAGE_FIELDS,
            )
            if type(lineage["schema_version"]) is not int or lineage["schema_version"] != 1:
                raise ValueError("candidate lineage schema_version must be 1")
            lineages.append(
                CandidateLineage(
                    generator_family=lineage["generator_family"],
                    generator_variant=lineage["generator_variant"],
                    logical_sha256=lineage["logical_sha256"],
                    training_projection_sha256=lineage["training_projection_sha256"],
                    seed=lineage["seed"],
                    ordinal=lineage["ordinal"],
                )
            )
        record = CandidateRecord(sequence=sequence, lineages=tuple(lineages))
        if canonical_sequence_id(sequence) != sequence_id:
            raise ValueError(f"candidate row {number} sequence identity is invalid")
        if record.canonical_record() != document:
            raise ValueError(f"candidate row {number} differs from its canonical record")
        records.append(record)
    sequences = tuple(record.sequence for record in records)
    if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
        raise ValueError("candidate rows must be unique and ordered by ascending sequence")
    return tuple(records)


def _probability_vector(
    value: object,
    *,
    keys: Sequence[str],
    label: str,
) -> tuple[float, ...]:
    document = _exact_mapping(value, label=label, expected_keys=keys)
    result: list[float] = []
    for key in keys:
        probability = document[key]
        if type(probability) is not float or not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"{label} contains an invalid probability")
        result.append(probability)
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} probabilities must sum to one")
    return tuple(result)


def _verify_distribution_metrics(
    metrics: Mapping[str, Any],
    records: Sequence[CandidateRecord],
) -> None:
    _exact_mapping(
        metrics,
        label="family_metrics",
        expected_keys={
            "distribution_reference",
            "pool_name",
            "schema_version",
            "source_metrics",
            "union_metrics",
        },
    )
    if (
        metrics.get("pool_name") != POOL_NAME
        or type(metrics.get("schema_version")) is not int
        or metrics.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("family_metrics identity is invalid")
    reference = _exact_mapping(
        metrics.get("distribution_reference"),
        label="family_metrics distribution_reference",
        expected_keys={
            "candidate_weighting",
            "divergence",
            "residue_estimator",
            "training_length_probabilities",
            "training_residue_probabilities",
            "training_weighting",
        },
    )
    if (
        reference["candidate_weighting"] != "uniform_over_eligible_unique_sequences"
        or reference["divergence"] != "jensen_shannon_base_2_bits"
        or reference["residue_estimator"] != "mean_per_sequence_residue_fraction"
        or reference["training_weighting"] != "component_weighted_sampling_weight"
    ):
        raise ValueError("family_metrics distribution contract is invalid")
    training_length = _probability_vector(
        reference["training_length_probabilities"],
        keys=tuple(str(length) for length in range(MINIMUM_LENGTH, MAXIMUM_LENGTH + 1)),
        label="training length probabilities",
    )
    training_residue = _probability_vector(
        reference["training_residue_probabilities"],
        keys=tuple(ALPHABET),
        label="training residue probabilities",
    )
    union = _exact_mapping(
        metrics.get("union_metrics"),
        label="family_metrics union_metrics",
        expected_keys={
            "duplicate_raw_proposals",
            "eligible_length_js_divergence_bits_vs_training",
            "eligible_residue_js_divergence_bits_vs_training",
            "eligible_unique_candidates",
            "eligible_unique_length_histogram",
            "exact_overlap_with_both_unique_sequences",
            "exact_organizer_reference_overlap_unique_sequences",
            "exact_training_overlap_unique_sequences",
            "excluded_unique_sequences",
            "minimum_unique_candidates",
            "minimum_unique_candidates_passed",
            "multi_generator_family_candidates",
            "multi_generator_variant_candidates",
            "multi_lineage_candidates",
            "raw_proposals",
            "raw_unique_sequences_before_filtering",
        },
    )
    observed_length, observed_residue = _distribution_divergences(
        {record.sequence for record in records},
        training_length=training_length,
        training_residue=training_residue,
    )
    if (
        union.get("eligible_length_js_divergence_bits_vs_training") != observed_length
        or union.get("eligible_residue_js_divergence_bits_vs_training") != observed_residue
    ):
        raise ValueError("family_metrics union distribution divergences are not reproducible")
    if union.get("eligible_unique_candidates") != len(records):
        raise ValueError("family_metrics eligible candidate count disagrees with JSONL")
    if union.get("eligible_unique_length_histogram") != _histogram(
        [len(record.sequence) for record in records]
    ):
        raise ValueError("family_metrics union length histogram disagrees with JSONL")
    union_integer_fields = (
        "duplicate_raw_proposals",
        "eligible_unique_candidates",
        "exact_overlap_with_both_unique_sequences",
        "exact_organizer_reference_overlap_unique_sequences",
        "exact_training_overlap_unique_sequences",
        "excluded_unique_sequences",
        "minimum_unique_candidates",
        "multi_generator_family_candidates",
        "multi_generator_variant_candidates",
        "multi_lineage_candidates",
        "raw_proposals",
        "raw_unique_sequences_before_filtering",
    )
    if any(type(union[field]) is not int or union[field] < 0 for field in union_integer_fields):
        raise ValueError("family_metrics union counts are invalid")
    if (
        type(union["minimum_unique_candidates_passed"]) is not bool
        or not union["minimum_unique_candidates_passed"]
    ):
        raise ValueError("family_metrics minimum candidate gate did not pass")
    if (
        union["minimum_unique_candidates"] <= 0
        or len(records) < union["minimum_unique_candidates"]
        or union["raw_proposals"] < union["raw_unique_sequences_before_filtering"]
        or union["duplicate_raw_proposals"]
        != union["raw_proposals"] - union["raw_unique_sequences_before_filtering"]
        or union["raw_unique_sequences_before_filtering"] - union["excluded_unique_sequences"]
        != len(records)
        or union["excluded_unique_sequences"]
        != union["exact_training_overlap_unique_sequences"]
        + union["exact_organizer_reference_overlap_unique_sequences"]
        - union["exact_overlap_with_both_unique_sequences"]
        or union["exact_overlap_with_both_unique_sequences"]
        > min(
            union["exact_training_overlap_unique_sequences"],
            union["exact_organizer_reference_overlap_unique_sequences"],
        )
        or not (
            union["multi_generator_family_candidates"]
            <= union["multi_generator_variant_candidates"]
            <= union["multi_lineage_candidates"]
            <= len(records)
        )
    ):
        raise ValueError("family_metrics union counts are inconsistent")

    sources_raw = metrics.get("source_metrics")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("family_metrics source_metrics must be a non-empty array")
    source_keys: list[tuple[str, str, str, str, int]] = []
    expected_source_fields = {
        "accepted_union_sequences_with_lineage",
        "eligible_length_js_divergence_bits_vs_training",
        "eligible_residue_js_divergence_bits_vs_training",
        "eligible_unique_length_histogram",
        "eligible_unique_sequences",
        "exact_organizer_reference_overlap_unique_sequences",
        "exact_training_overlap_unique_sequences",
        "generator_family",
        "generator_variant",
        "logical_sha256",
        "raw_length_histogram",
        "raw_proposals",
        "raw_unique_sequences",
        "seed",
        "training_projection_sha256",
        "within_source_duplicate_proposals",
    }
    for index, source_raw in enumerate(sources_raw, 1):
        source = _exact_mapping(
            source_raw,
            label=f"family_metrics source {index}",
            expected_keys=expected_source_fields,
        )
        family = _manifest_name(source["generator_family"], label="source generator_family")
        variant = _manifest_name(source["generator_variant"], label="source generator_variant")
        logical_sha256 = _sha256(source["logical_sha256"], label="source logical_sha256")
        training_sha256 = source["training_projection_sha256"]
        if training_sha256 is not None:
            training_sha256 = _sha256(
                training_sha256,
                label="source training_projection_sha256",
            )
        seed = _uint64(source["seed"], label="source seed")
        source_lineage = CandidateLineage(
            generator_family=family,
            generator_variant=variant,
            logical_sha256=logical_sha256,
            training_projection_sha256=training_sha256,
            seed=seed,
            ordinal=0,
        )
        key = _source_key(source_lineage)
        source_keys.append(key)
        eligible = {
            record.sequence
            for record in records
            if any(_source_key(lineage) == key for lineage in record.lineages)
        }
        length_divergence, residue_divergence = _distribution_divergences(
            eligible,
            training_length=training_length,
            training_residue=training_residue,
        )
        if (
            source["eligible_unique_sequences"] != len(eligible)
            or source["accepted_union_sequences_with_lineage"] != len(eligible)
            or source["eligible_unique_length_histogram"]
            != _histogram([len(sequence) for sequence in eligible])
            or source["eligible_length_js_divergence_bits_vs_training"] != length_divergence
            or source["eligible_residue_js_divergence_bits_vs_training"] != residue_divergence
        ):
            raise ValueError(f"family_metrics source {index} eligible metrics are invalid")
        integer_fields = (
            "accepted_union_sequences_with_lineage",
            "eligible_unique_sequences",
            "raw_proposals",
            "raw_unique_sequences",
            "within_source_duplicate_proposals",
            "exact_training_overlap_unique_sequences",
            "exact_organizer_reference_overlap_unique_sequences",
        )
        if any(type(source[field]) is not int or source[field] < 0 for field in integer_fields):
            raise ValueError(f"family_metrics source {index} counts are invalid")
        if (
            source["raw_proposals"] < source["raw_unique_sequences"]
            or source["raw_unique_sequences"] < len(eligible)
            or source["within_source_duplicate_proposals"]
            != source["raw_proposals"] - source["raw_unique_sequences"]
        ):
            raise ValueError(f"family_metrics source {index} raw counts are inconsistent")
        raw_histogram = source["raw_length_histogram"]
        if not isinstance(raw_histogram, dict) or any(
            type(key) is not str
            or not key.isascii()
            or not key.isdigit()
            or key != str(int(key))
            or not MINIMUM_LENGTH <= int(key) <= MAXIMUM_LENGTH
            or type(value) is not int
            or value <= 0
            for key, value in raw_histogram.items()
        ):
            raise ValueError(f"family_metrics source {index} raw histogram is invalid")
        if sum(raw_histogram.values()) != source["raw_proposals"]:
            raise ValueError(f"family_metrics source {index} raw histogram count is invalid")
    if source_keys != sorted(set(source_keys)):
        raise ValueError("family_metrics sources must be unique and canonically ordered")
    candidate_source_keys = {
        _source_key(lineage) for record in records for lineage in record.lineages
    }
    if not candidate_source_keys <= set(source_keys):
        raise ValueError("candidate JSONL contains an unreported generator source")


def _verify_manifest_provenance(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    records: Sequence[CandidateRecord],
) -> None:
    inputs = _exact_mapping(
        manifest.get("inputs"),
        label="candidate-pool manifest inputs",
        expected_keys={"config", "organizer_reference", "training_projection"},
    )
    config_input = _exact_mapping(
        inputs["config"],
        label="candidate-pool config input",
        expected_keys={"sha256", "size_bytes"},
    )
    config_sha256 = _sha256(config_input["sha256"], label="config input sha256")
    config_size = _positive_integer(
        config_input["size_bytes"],
        label="config input size_bytes",
    )
    training = _exact_mapping(
        inputs["training_projection"],
        label="candidate-pool training input",
        expected_keys={
            "accepted_contract_required",
            "accepted_contract_satisfied",
            "fold_4_data_access",
            "records",
            "role",
            "sha256",
            "size_bytes",
            "trainer_visible_fields",
        },
    )
    accepted_required = _boolean(
        training["accepted_contract_required"],
        label="training accepted_contract_required",
    )
    accepted_satisfied = _boolean(
        training["accepted_contract_satisfied"],
        label="training accepted_contract_satisfied",
    )
    training_sha256 = _sha256(training["sha256"], label="training input sha256")
    training_records = _positive_integer(training["records"], label="training input records")
    training_size = _positive_integer(
        training["size_bytes"],
        label="training input size_bytes",
    )
    derived_accepted = (
        training_sha256 == ACCEPTED_TRAINING_PROJECTION_SHA256
        and training_records == ACCEPTED_TRAINING_PROJECTION_ROWS
        and training_size == ACCEPTED_TRAINING_PROJECTION_SIZE_BYTES
    )
    if accepted_satisfied is not derived_accepted:
        raise ValueError("training accepted-contract status is inconsistent")
    if (
        training["fold_4_data_access"] is not False
        or training["role"] != "generation_training_only"
        or training["trainer_visible_fields"] != ["sequence_id", "sequence", "sampling_weight"]
    ):
        raise ValueError("training input role or fold-4 boundary is invalid")

    reference = _exact_mapping(
        inputs["organizer_reference"],
        label="candidate-pool organizer-reference input",
        expected_keys={"records", "role", "sha256", "size_bytes", "unique_sequences"},
    )
    reference_sha256 = _sha256(reference["sha256"], label="organizer reference sha256")
    reference_records = _positive_integer(
        reference["records"],
        label="organizer reference records",
    )
    reference_size = _positive_integer(
        reference["size_bytes"],
        label="organizer reference size_bytes",
    )
    reference_unique = _positive_integer(
        reference["unique_sequences"],
        label="organizer reference unique_sequences",
    )
    if reference["role"] != ORGANIZER_REFERENCE_ROLE or reference_unique > reference_records:
        raise ValueError("organizer reference role or census is invalid")

    generation = _exact_mapping(
        manifest.get("generation"),
        label="candidate-pool manifest generation",
        expected_keys={
            "baseline",
            "count_controls",
            "organizer_reference_read_after_all_raw_proposals",
            "raw_proposals_complete_before_compliance_filtering",
        },
    )
    if (
        generation["organizer_reference_read_after_all_raw_proposals"] is not True
        or generation["raw_proposals_complete_before_compliance_filtering"] is not True
    ):
        raise ValueError("organizer-reference compliance ordering is invalid")
    baseline = _exact_mapping(
        generation["baseline"],
        label="candidate-pool manifest baseline",
        expected_keys={"candidates", "generator_family", "seed", "specification_sha256"},
    )
    baseline_candidates = _positive_integer(
        baseline["candidates"],
        label="baseline candidates",
    )
    baseline_seed = _uint64(baseline["seed"], label="baseline seed")
    baseline_identity = _sha256(
        baseline["specification_sha256"],
        label="baseline specification_sha256",
    )
    if baseline["generator_family"] != BASELINE_GENERATOR:
        raise ValueError("manifest baseline generator family is invalid")
    if baseline_identity != baseline_spec_sha256():
        raise ValueError("manifest baseline specification identity is invalid")

    controls = _exact_mapping(
        generation["count_controls"],
        label="candidate-pool manifest count controls",
        expected_keys={
            "batch_size",
            "candidates_per_method",
            "control_logical_sha256",
            "length_plan_namespace",
            "length_plan_sha256",
            "methods",
            "probability_tables_sha256",
            "seed",
            "shared_length_plan",
            "training_projection_sha256",
        },
    )
    count_batch_size = _positive_integer(controls["batch_size"], label="count batch_size")
    count_candidates = _positive_integer(
        controls["candidates_per_method"],
        label="count candidates_per_method",
    )
    count_logical_sha256 = _sha256(
        controls["control_logical_sha256"],
        label="count control_logical_sha256",
    )
    _sha256(
        controls["length_plan_sha256"],
        label="count length_plan_sha256",
    )
    _sha256(
        controls["probability_tables_sha256"],
        label="count probability_tables_sha256",
    )
    count_seed = _uint64(controls["seed"], label="count seed")
    namespace = _manifest_name(
        controls["length_plan_namespace"],
        label="count length_plan_namespace",
    )
    if (
        controls["methods"] != list(GENERATOR_CONTROL_METHODS)
        or controls["shared_length_plan"] is not True
        or controls["training_projection_sha256"] != training_sha256
    ):
        raise ValueError("count-control method, length-plan, or training binding is invalid")
    production_config = (
        config_sha256 == PRODUCTION_CONFIG_SHA256 and config_size == PRODUCTION_CONFIG_SIZE_BYTES
    )
    if accepted_required and not accepted_satisfied:
        raise ValueError("production candidate pool requires the accepted training projection")
    if accepted_required is not production_config:
        raise ValueError("production config identity and accepted-contract flag disagree")
    if accepted_required:
        production_values = (
            config_sha256 == PRODUCTION_CONFIG_SHA256,
            config_size == PRODUCTION_CONFIG_SIZE_BYTES,
            training_size == ACCEPTED_TRAINING_PROJECTION_SIZE_BYTES,
            reference_sha256 == ACCEPTED_ORGANIZER_REFERENCE_SHA256,
            reference_records == ACCEPTED_ORGANIZER_REFERENCE_RECORDS,
            reference_size == ACCEPTED_ORGANIZER_REFERENCE_SIZE_BYTES,
            baseline_candidates == PRODUCTION_BASELINE_CANDIDATES,
            baseline_seed == PRODUCTION_BASELINE_SEED,
            count_candidates == PRODUCTION_COUNT_CANDIDATES_PER_METHOD,
            count_seed == PRODUCTION_COUNT_SEED,
            count_batch_size == PRODUCTION_COUNT_BATCH_SIZE,
            namespace == PRODUCTION_LENGTH_PLAN_NAMESPACE,
            metrics["union_metrics"]["minimum_unique_candidates"]
            == PRODUCTION_MINIMUM_UNIQUE_CANDIDATES,
        )
        if not all(production_values):
            raise ValueError(
                "production candidate-pool provenance differs from its frozen contract"
            )

    source_metrics = metrics["source_metrics"]
    baseline_raw = 0
    count_raw = {method: 0 for method in GENERATOR_CONTROL_METHODS}
    baseline_variants: set[str] = set()
    for source in source_metrics:
        family = source["generator_family"]
        variant = source["generator_variant"]
        if family == BASELINE_GENERATOR:
            baseline_raw += source["raw_proposals"]
            baseline_variants.add(variant)
            if (
                source["logical_sha256"] != baseline_identity
                or source["training_projection_sha256"] is not None
                or source["seed"] != baseline_seed
            ):
                raise ValueError("baseline source metrics disagree with manifest generation")
        elif family == COUNT_GENERATOR_FAMILY:
            if variant not in count_raw:
                raise ValueError("source metrics contain an unconfigured count-control method")
            count_raw[variant] += source["raw_proposals"]
            if (
                source["logical_sha256"] != count_logical_sha256
                or source["training_projection_sha256"] != training_sha256
                or source["seed"] != count_seed
            ):
                raise ValueError("count source metrics disagree with manifest generation")
        else:
            raise ValueError("source metrics contain an unknown generator family")
    if baseline_raw != baseline_candidates or any(
        count_raw[method] != count_candidates for method in GENERATOR_CONTROL_METHODS
    ):
        raise ValueError("manifest generator census disagrees with family_metrics")
    if accepted_required and baseline_variants != _BASELINE_FAMILY_NAMES:
        raise ValueError("production baseline family coverage is incomplete")
    expected_raw = baseline_candidates + len(GENERATOR_CONTROL_METHODS) * count_candidates
    if metrics["union_metrics"]["raw_proposals"] != expected_raw:
        raise ValueError("manifest raw proposal census disagrees with union metrics")

    baseline_ordinals: set[int] = set()
    count_ordinals = {method: set() for method in GENERATOR_CONTROL_METHODS}
    for record in records:
        for lineage in record.lineages:
            if lineage.generator_family == BASELINE_GENERATOR:
                valid = (
                    lineage.logical_sha256 == baseline_identity
                    and lineage.training_projection_sha256 is None
                    and lineage.seed == baseline_seed
                    and lineage.ordinal < baseline_candidates
                    and lineage.ordinal not in baseline_ordinals
                )
                baseline_ordinals.add(lineage.ordinal)
            else:
                valid = (
                    lineage.generator_family == COUNT_GENERATOR_FAMILY
                    and lineage.generator_variant in GENERATOR_CONTROL_METHODS
                    and lineage.logical_sha256 == count_logical_sha256
                    and lineage.training_projection_sha256 == training_sha256
                    and lineage.seed == count_seed
                    and lineage.ordinal < count_candidates
                    and lineage.ordinal not in count_ordinals[lineage.generator_variant]
                )
                if lineage.generator_variant in count_ordinals:
                    count_ordinals[lineage.generator_variant].add(lineage.ordinal)
            if not valid:
                raise ValueError("candidate lineage disagrees with manifest generation")


def verify_candidate_pool_structure(output_dir: str | Path) -> CandidatePoolVerification:
    """Verify one sealed publication's self-contained structure and cross-links.

    This verifier deliberately makes no claim that the declared inputs generated
    the publication.  Use :func:`verify_candidate_pool_output` when the frozen
    config, training projection, and organizer reference are available.
    """

    output = Path(os.path.abspath(os.fspath(output_dir)))
    _reject_symlink_chain(output, label="candidate-pool output")
    try:
        output_metadata = os.lstat(output)
    except OSError as error:
        raise ValueError(f"cannot inspect candidate-pool output: {output}") from error
    if not stat.S_ISDIR(output_metadata.st_mode):
        raise ValueError("candidate-pool output must be a real directory")
    if stat.S_IMODE(output_metadata.st_mode) != 0o555:
        raise ValueError("candidate-pool output directory mode must be 0555")
    output_identity = (
        output_metadata.st_dev,
        output_metadata.st_ino,
        output_metadata.st_mode,
        output_metadata.st_nlink,
        output_metadata.st_uid,
        output_metadata.st_gid,
        output_metadata.st_size,
        output_metadata.st_mtime_ns,
        output_metadata.st_ctime_ns,
    )
    try:
        with os.scandir(output) as entries:
            names = {entry.name for entry in entries}
    except OSError as error:
        raise ValueError("cannot enumerate candidate-pool output") from error
    if names != set(_OUTPUT_NAMES):
        raise ValueError(
            "candidate-pool output inventory mismatch: "
            f"missing={sorted(set(_OUTPUT_NAMES) - names)}, "
            f"extra={sorted(names - set(_OUTPUT_NAMES))}"
        )

    snapshots: dict[str, _Snapshot] = {}
    for name in _OUTPUT_NAMES:
        path = output / name
        snapshot = _snapshot_regular(path, label=f"candidate-pool artifact {name}")
        if not stat.S_ISREG(snapshot.mode):
            raise ValueError(f"candidate-pool artifact must be a regular file: {name}")
        if stat.S_IMODE(snapshot.mode) != 0o444:
            raise ValueError(f"candidate-pool artifact mode must be 0444: {name}")
        if snapshot.link_count != 1:
            raise ValueError(f"candidate-pool artifact link count must be one: {name}")
        snapshots[name] = snapshot

    records = _parse_candidate_records(snapshots["candidates.jsonl"].payload)
    expected_fasta = _candidate_fasta_bytes(records)
    if snapshots["candidates.fasta"].payload != expected_fasta:
        raise ValueError("candidates.fasta does not exactly mirror candidates.jsonl")
    metrics_raw = _parse_canonical_json(
        snapshots["family_metrics.json"].payload,
        label="family_metrics.json",
    )
    if not isinstance(metrics_raw, dict):
        raise ValueError("family_metrics.json must contain a JSON object")
    _verify_distribution_metrics(metrics_raw, records)
    manifest_raw = _parse_canonical_json(
        snapshots["manifest.json"].payload,
        label="manifest.json",
    )
    if not isinstance(manifest_raw, dict):
        raise ValueError("manifest.json must contain a JSON object")
    _exact_mapping(
        manifest_raw,
        label="candidate-pool manifest",
        expected_keys={
            "artifacts",
            "candidate_contract",
            "counts",
            "generation",
            "inputs",
            "name",
            "schema_version",
            "status",
        },
    )
    if (
        type(manifest_raw.get("schema_version")) is not int
        or manifest_raw.get("schema_version") != SCHEMA_VERSION
        or manifest_raw.get("name") != POOL_NAME
        or manifest_raw.get("status") != POOL_STATUS
    ):
        raise ValueError("candidate-pool manifest identity is invalid")
    contract = _exact_mapping(
        manifest_raw.get("candidate_contract"),
        label="candidate-pool manifest candidate_contract",
        expected_keys={
            "alphabet",
            "candidate_fields",
            "contains_acquisition_lineage",
            "contains_prediction_fields",
            "library_eligible_value",
            "lineage_fields",
            "maximum_length",
            "minimum_length",
            "ordering",
            "schema_version",
        },
    )
    if (
        contract.get("candidate_fields") != list(_CANDIDATE_FIELDS)
        or contract.get("lineage_fields") != list(_LINEAGE_FIELDS)
        or contract.get("contains_prediction_fields") is not False
        or contract.get("contains_acquisition_lineage") is not False
        or contract.get("library_eligible_value") is not True
        or contract.get("alphabet") != ALPHABET
        or type(contract.get("minimum_length")) is not int
        or contract.get("minimum_length") != MINIMUM_LENGTH
        or type(contract.get("maximum_length")) is not int
        or contract.get("maximum_length") != MAXIMUM_LENGTH
        or contract.get("ordering") != "ascending_sequence_lexicographic"
        or type(contract.get("schema_version")) is not int
        or contract.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("candidate-pool manifest candidate contract is invalid")
    if manifest_raw.get("counts") != metrics_raw.get("union_metrics"):
        raise ValueError("candidate-pool manifest counts differ from family_metrics")
    _verify_manifest_provenance(manifest_raw, metrics_raw, records)
    artifacts = _exact_mapping(
        manifest_raw.get("artifacts"),
        label="candidate-pool manifest artifacts",
        expected_keys={"candidates.fasta", "candidates.jsonl", "family_metrics.json"},
    )
    for name in ("candidates.fasta", "candidates.jsonl", "family_metrics.json"):
        expected_keys = {"sha256", "size_bytes"}
        if name != "family_metrics.json":
            expected_keys.add("records")
        artifact = _exact_mapping(
            artifacts[name],
            label=f"manifest artifact {name}",
            expected_keys=expected_keys,
        )
        snapshot = snapshots[name]
        artifact_sha256 = _sha256(
            artifact["sha256"],
            label=f"manifest artifact {name} sha256",
        )
        artifact_size = _positive_integer(
            artifact["size_bytes"],
            label=f"manifest artifact {name} size_bytes",
        )
        if artifact_sha256 != snapshot.sha256 or artifact_size != snapshot.size_bytes:
            raise ValueError(f"manifest artifact identity disagrees for {name}")
        if "records" in artifact:
            artifact_records = _positive_integer(
                artifact["records"],
                label=f"manifest artifact {name} records",
            )
            if artifact_records != len(records):
                raise ValueError(f"manifest record count disagrees for {name}")

    checksum_payloads = {
        name: snapshots[name].payload
        for name in ("candidates.fasta", "candidates.jsonl", "family_metrics.json", "manifest.json")
    }
    expected_sha256sums = "".join(
        f"{hashlib.sha256(checksum_payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(checksum_payloads)
    ).encode("ascii")
    if snapshots["SHA256SUMS"].payload != expected_sha256sums:
        raise ValueError("SHA256SUMS is noncanonical or contains a checksum mismatch")

    for name, snapshot in snapshots.items():
        _require_unchanged(
            output / name,
            snapshot,
            label=f"candidate-pool artifact {name}",
        )
    try:
        output_after = os.lstat(output)
        with os.scandir(output) as entries:
            names_after = {entry.name for entry in entries}
    except OSError as error:
        raise ValueError("candidate-pool output changed while it was verified") from error
    output_identity_after = (
        output_after.st_dev,
        output_after.st_ino,
        output_after.st_mode,
        output_after.st_nlink,
        output_after.st_uid,
        output_after.st_gid,
        output_after.st_size,
        output_after.st_mtime_ns,
        output_after.st_ctime_ns,
    )
    if output_identity_after != output_identity or names_after != names:
        raise ValueError("candidate-pool output changed while it was verified")
    return CandidatePoolVerification(
        output_dir=output,
        candidate_count=len(records),
        candidates_jsonl_sha256=snapshots["candidates.jsonl"].sha256,
        candidates_fasta_sha256=snapshots["candidates.fasta"].sha256,
        family_metrics_sha256=snapshots["family_metrics.json"].sha256,
        manifest_sha256=snapshots["manifest.json"].sha256,
    )


def _stage_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            path.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _renameat2_directory_noreplace(staging: Path, output: Path) -> int:
    """Return zero on atomic no-replace rename, otherwise the Linux errno."""

    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
    except (AttributeError, OSError):
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    return 0 if result == 0 else (ctypes.get_errno() or errno.EIO)


def _same_publication_content(current: _Snapshot, expected: _Snapshot) -> bool:
    """Compare immutable content identity while allowing link/ctime transitions."""

    return (
        current.sha256 == expected.sha256
        and current.size_bytes == expected.size_bytes
        and current.device == expected.device
        and current.inode == expected.inode
        and current.owner_uid == expected.owner_uid
        and current.owner_gid == expected.owner_gid
        and current.modified_ns == expected.modified_ns
    )


def _link_commit_directory_noreplace(
    staging: Path,
    output: Path,
    *,
    expected_snapshots: Mapping[str, _Snapshot],
) -> None:
    """Claim ``output`` exclusively and make ``SHA256SUMS`` readable last.

    Lustre can reject ``renameat2(RENAME_NOREPLACE)`` with ``EINVAL``.  This
    fallback first claims the public name with an exclusive ``mkdir``, links
    the already authenticated immutable payloads, and links ``SHA256SUMS`` in
    mode 000.  It removes the private staging links before freezing the public
    directory, so every committed file has exactly one link.  The final chmod
    of the public checksum marker to 0444 is the commit transition.  Any
    earlier failure leaves a never-reused directory without a readable commit
    marker.
    """

    if set(expected_snapshots) != set(_OUTPUT_NAMES):
        raise ValueError("candidate-pool publication snapshot inventory is invalid")
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace candidate-pool output: {output}") from error
    claim_metadata = os.lstat(output)
    claim_inode = (claim_metadata.st_dev, claim_metadata.st_ino)
    if (
        stat.S_ISLNK(claim_metadata.st_mode)
        or not stat.S_ISDIR(claim_metadata.st_mode)
        or stat.S_IMODE(claim_metadata.st_mode) != 0o700
    ):
        raise RuntimeError("candidate-pool publication name claim is not a real directory")

    marker_name = "SHA256SUMS"
    payload_names = tuple(name for name in _OUTPUT_NAMES if name != marker_name)
    marker_destination = output / marker_name
    try:
        staging_metadata = os.lstat(staging)
        if (
            stat.S_ISLNK(staging_metadata.st_mode)
            or not stat.S_ISDIR(staging_metadata.st_mode)
            or staging_metadata.st_dev != claim_metadata.st_dev
            or stat.S_IMODE(staging_metadata.st_mode) != 0o555
        ):
            raise RuntimeError(
                "candidate-pool staging and publication claim are not real same-device directories"
            )

        for name in payload_names:
            source = staging / name
            expected = expected_snapshots[name]
            _require_unchanged(source, expected, label=f"staged publication artifact {name}")
            os.link(source, output / name, follow_symlinks=False)
            source_after = _snapshot_regular(
                source,
                label=f"linked staged publication artifact {name}",
            )
            destination_after = _snapshot_regular(
                output / name,
                label=f"claimed publication artifact {name}",
            )
            if (
                not _same_publication_content(source_after, expected)
                or not _same_publication_content(destination_after, expected)
                or source_after.link_count != 2
                or destination_after.link_count != 2
                or stat.S_IMODE(source_after.mode) != 0o444
                or stat.S_IMODE(destination_after.mode) != 0o444
            ):
                raise RuntimeError(f"candidate-pool publication link changed for {name}")

        marker_source = staging / marker_name
        marker_expected = expected_snapshots[marker_name]
        _require_unchanged(
            marker_source,
            marker_expected,
            label="staged publication checksum marker",
        )
        os.chmod(marker_source, 0o000)
        os.link(marker_source, marker_destination, follow_symlinks=False)
        marker_source_metadata = os.lstat(marker_source)
        marker_destination_metadata = os.lstat(marker_destination)
        if (
            (marker_source_metadata.st_dev, marker_source_metadata.st_ino)
            != (marker_expected.device, marker_expected.inode)
            or (marker_destination_metadata.st_dev, marker_destination_metadata.st_ino)
            != (marker_expected.device, marker_expected.inode)
            or marker_source_metadata.st_nlink != 2
            or marker_destination_metadata.st_nlink != 2
            or stat.S_IMODE(marker_source_metadata.st_mode) != 0
            or stat.S_IMODE(marker_destination_metadata.st_mode) != 0
        ):
            raise RuntimeError("candidate-pool checksum marker link changed before commit")

        with os.scandir(output) as entries:
            published_names = {entry.name for entry in entries}
        if published_names != set(_OUTPUT_NAMES):
            raise RuntimeError("claimed candidate-pool publication inventory changed")
        _fsync_directory(output)

        # Remove every private source link before commit.  This is deliberately
        # not best-effort: a committed candidate-pool tree must have nlink=1.
        os.chmod(staging, 0o700)
        for name in _OUTPUT_NAMES:
            (staging / name).unlink()
        staging.rmdir()

        for name in payload_names:
            final_snapshot = _snapshot_regular(
                output / name,
                label=f"final claimed publication artifact {name}",
            )
            if (
                not _same_publication_content(final_snapshot, expected_snapshots[name])
                or final_snapshot.link_count != 1
                or stat.S_IMODE(final_snapshot.mode) != 0o444
            ):
                raise RuntimeError(f"candidate-pool final publication changed for {name}")
        final_marker_metadata = os.lstat(marker_destination)
        if (
            not stat.S_ISREG(final_marker_metadata.st_mode)
            or (final_marker_metadata.st_dev, final_marker_metadata.st_ino)
            != (marker_expected.device, marker_expected.inode)
            or final_marker_metadata.st_size != marker_expected.size_bytes
            or final_marker_metadata.st_uid != marker_expected.owner_uid
            or final_marker_metadata.st_gid != marker_expected.owner_gid
            or final_marker_metadata.st_mtime_ns != marker_expected.modified_ns
            or final_marker_metadata.st_nlink != 1
            or stat.S_IMODE(final_marker_metadata.st_mode) != 0
        ):
            raise RuntimeError("candidate-pool checksum marker changed before commit")

        current_claim = os.lstat(output)
        if (
            stat.S_ISLNK(current_claim.st_mode)
            or not stat.S_ISDIR(current_claim.st_mode)
            or (current_claim.st_dev, current_claim.st_ino) != claim_inode
        ):
            raise RuntimeError("candidate-pool publication name claim changed before commit")
        os.chmod(output, 0o555)
        frozen_claim = os.lstat(output)
        if (
            stat.S_ISLNK(frozen_claim.st_mode)
            or (frozen_claim.st_dev, frozen_claim.st_ino) != claim_inode
            or stat.S_IMODE(frozen_claim.st_mode) != 0o555
        ):
            raise RuntimeError("candidate-pool publication name claim did not freeze")
        _fsync_directory(output)
        _fsync_directory(output.parent)

        # Atomic commit point.  No fallible publication step follows this
        # transition: consumers accept only this readable, authenticated marker.
        os.chmod(marker_destination, 0o444)
    except BaseException:
        # Preserve a claimed name as a fail-closed quarantine.  Never delete it
        # or make its marker readable after a failed publication attempt.
        with suppress(OSError):
            current_claim = os.lstat(output)
            if (
                stat.S_ISDIR(current_claim.st_mode)
                and not stat.S_ISLNK(current_claim.st_mode)
                and (current_claim.st_dev, current_claim.st_ino) == claim_inode
            ):
                with suppress(OSError):
                    marker_metadata = os.lstat(marker_destination)
                    if stat.S_ISREG(marker_metadata.st_mode):
                        os.chmod(marker_destination, 0o000)
                os.chmod(output, 0o555)
        raise


def _remove_private_staging(staging: Path, *, expected_inode: tuple[int, int]) -> None:
    """Best-effort removal of only the private staging tree we created."""

    try:
        metadata = os.lstat(staging)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_inode
        ):
            return
        with os.scandir(staging) as entries:
            children = tuple(entries)
        if any(
            entry.name not in _OUTPUT_NAMES
            or entry.is_symlink()
            or not entry.is_file(follow_symlinks=False)
            for entry in children
        ):
            return
        os.chmod(staging, 0o700)
        for entry in children:
            (staging / entry.name).unlink()
        staging.rmdir()
    except OSError:
        return


def _publish_output(
    output_dir: str | Path,
    artifacts: Mapping[str, bytes],
) -> tuple[Path, CandidatePoolVerification]:
    if tuple(artifacts) != _OUTPUT_NAMES:
        raise ValueError("candidate-pool artifact inventory or publication order is invalid")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    parent = output.parent
    _reject_symlink_chain(parent, label="candidate-pool output parent")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(parent, label="candidate-pool output parent")
    if not parent.is_dir():
        raise ValueError("candidate-pool output parent must be a directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse candidate-pool output: {output}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=parent))
    staging_metadata = os.lstat(staging)
    staging_inode = (staging_metadata.st_dev, staging_metadata.st_ino)
    committed = False
    try:
        for name, payload in artifacts.items():
            _stage_file(staging / name, payload)
        _fsync_directory(staging)
        os.chmod(staging, 0o555)
        _fsync_directory(staging)
        staged_verification = verify_candidate_pool_structure(staging)
        staged_snapshots = {
            name: _snapshot_regular(
                staging / name,
                label=f"authenticated staged publication artifact {name}",
            )
            for name in _OUTPUT_NAMES
        }

        error_number = _renameat2_directory_noreplace(staging, output)
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"refusing to replace candidate-pool output: {output}")
        if error_number == 0:
            committed = True
            published_metadata = os.lstat(output)
            if (
                os.path.lexists(staging)
                or stat.S_ISLNK(published_metadata.st_mode)
                or not stat.S_ISDIR(published_metadata.st_mode)
                or (published_metadata.st_dev, published_metadata.st_ino) != staging_inode
            ):
                raise RuntimeError("atomic candidate-pool publication postcondition failed")
            _fsync_directory(parent)
        elif error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            _link_commit_directory_noreplace(
                staging,
                output,
                expected_snapshots=staged_snapshots,
            )
            committed = True
        else:
            raise OSError(
                error_number,
                "candidate-pool publication no-replace commit failed: " + os.strerror(error_number),
                output,
            )
        verification = verify_candidate_pool_structure(output)
        staged_identity = (
            staged_verification.candidate_count,
            staged_verification.candidates_jsonl_sha256,
            staged_verification.candidates_fasta_sha256,
            staged_verification.family_metrics_sha256,
            staged_verification.manifest_sha256,
        )
        published_identity = (
            verification.candidate_count,
            verification.candidates_jsonl_sha256,
            verification.candidates_fasta_sha256,
            verification.family_metrics_sha256,
            verification.manifest_sha256,
        )
        if published_identity != staged_identity:
            raise RuntimeError("candidate-pool bytes changed across atomic publication")
        return output, verification
    finally:
        if not committed:
            _remove_private_staging(staging, expected_inode=staging_inode)


def build_candidate_pool(
    *,
    config_path: str | Path,
    training_projection_path: str | Path,
    organizer_reference_path: str | Path,
    output_dir: str | Path,
) -> CandidatePoolExecution:
    """Generate and publish the deterministic score-free candidate pool."""

    config_snapshot = _snapshot_regular(config_path, label="candidate-pool config")
    config = _parse_candidate_pool_config(config_snapshot.payload)
    requested_output = Path(os.path.abspath(os.fspath(output_dir)))
    _reject_symlink_chain(requested_output, label="candidate-pool output")
    if os.path.lexists(requested_output):
        raise FileExistsError(f"refusing to reuse candidate-pool output: {requested_output}")
    training_snapshot = _snapshot_regular(
        training_projection_path,
        label="training projection",
    )
    if training_snapshot.sha256 != config.training_projection_sha256:
        raise ValueError(
            "training projection SHA-256 expected "
            f"{config.training_projection_sha256}, got {training_snapshot.sha256}"
        )
    training: TrainingDistribution = load_training_projection(
        training_projection_path,
        expected_sha256=config.training_projection_sha256,
        expected_rows=config.training_projection_rows,
    )

    baseline_identity = baseline_spec_sha256()
    baseline = generate_baseline_pool(
        config.baseline_candidates,
        seed=config.baseline_seed,
    )
    proposals: list[RawCandidateProposal] = [
        RawCandidateProposal(
            sequence=sequence,
            lineage=CandidateLineage(
                generator_family=BASELINE_GENERATOR,
                generator_variant=family,
                logical_sha256=baseline_identity,
                training_projection_sha256=None,
                seed=config.baseline_seed,
                ordinal=ordinal,
            ),
        )
        for ordinal, (sequence, family) in enumerate(
            zip(baseline.sequences, baseline.families, strict=True)
        )
    ]

    count_suite = fit_count_baselines(
        training,
        require_locked_census=config.require_accepted_training_projection,
    )
    lengths = training.length_prior.draw(
        root_seed=config.count_seed,
        draw_start=0,
        draw_count=config.count_candidates_per_method,
        namespace=config.length_plan_namespace,
    )
    length_plan_sha256: str | None = None
    for method in GENERATOR_CONTROL_METHODS:
        result = sample_count_generator_control_v0(
            count_suite,
            lengths,
            method=method,
            seed=config.count_seed,
            batch_size=config.count_batch_size,
            require_locked_count=False,
        )
        if length_plan_sha256 is None:
            length_plan_sha256 = result.length_plan_sha256
        elif result.length_plan_sha256 != length_plan_sha256:
            raise RuntimeError("count controls did not share one canonical length plan")
        proposals.extend(
            RawCandidateProposal(
                sequence=candidate.sequence,
                lineage=CandidateLineage(
                    generator_family=COUNT_GENERATOR_FAMILY,
                    generator_variant=candidate.method,
                    logical_sha256=candidate.control_logical_sha256,
                    training_projection_sha256=candidate.training_projection_sha256,
                    seed=candidate.seed,
                    ordinal=candidate.ordinal,
                ),
            )
            for candidate in result.candidates
        )
    if length_plan_sha256 is None:
        raise RuntimeError("candidate_pool_v1 did not construct a count-control length plan")

    # The compliance reference is deliberately unavailable until every proposal
    # sequence, length, family, and ordinal is fixed above.
    reference_snapshot, reference_sequences = _read_reference_snapshot(
        organizer_reference_path,
        expected_sha256=config.organizer_reference_sha256,
        expected_records=config.organizer_reference_records,
    )
    assembly = assemble_candidate_records(
        proposals,
        training_sequence_weights={row.sequence: row.sampling_weight for row in training.rows},
        organizer_reference_sequences=set(reference_sequences),
        minimum_unique_candidates=config.minimum_unique_candidates,
    )

    candidates_jsonl = _candidate_jsonl_bytes(assembly.records)
    candidates_fasta = _candidate_fasta_bytes(assembly.records)
    family_metrics = _canonical_json_bytes(assembly.family_metrics)
    artifacts_without_manifest = {
        "candidates.fasta": _artifact_record(
            candidates_fasta,
            records=len(assembly.records),
        ),
        "candidates.jsonl": _artifact_record(
            candidates_jsonl,
            records=len(assembly.records),
        ),
        "family_metrics.json": _artifact_record(family_metrics),
    }
    manifest_document: dict[str, object] = {
        "artifacts": artifacts_without_manifest,
        "candidate_contract": {
            "alphabet": ALPHABET,
            "candidate_fields": list(_CANDIDATE_FIELDS),
            "contains_acquisition_lineage": False,
            "contains_prediction_fields": False,
            "library_eligible_value": True,
            "lineage_fields": list(_LINEAGE_FIELDS),
            "maximum_length": MAXIMUM_LENGTH,
            "minimum_length": MINIMUM_LENGTH,
            "ordering": "ascending_sequence_lexicographic",
            "schema_version": SCHEMA_VERSION,
        },
        "counts": assembly.family_metrics["union_metrics"],
        "generation": {
            "baseline": {
                "candidates": config.baseline_candidates,
                "generator_family": BASELINE_GENERATOR,
                "seed": config.baseline_seed,
                "specification_sha256": baseline_identity,
            },
            "count_controls": {
                "batch_size": config.count_batch_size,
                "candidates_per_method": config.count_candidates_per_method,
                "control_logical_sha256": count_suite.logical_sha256,
                "length_plan_namespace": config.length_plan_namespace,
                "length_plan_sha256": length_plan_sha256,
                "methods": list(GENERATOR_CONTROL_METHODS),
                "probability_tables_sha256": count_suite.probability_tables_sha256,
                "seed": config.count_seed,
                "shared_length_plan": True,
                "training_projection_sha256": count_suite.training_projection_sha256,
            },
            "organizer_reference_read_after_all_raw_proposals": True,
            "raw_proposals_complete_before_compliance_filtering": True,
        },
        "inputs": {
            "config": {
                "sha256": config_snapshot.sha256,
                "size_bytes": config_snapshot.size_bytes,
            },
            "organizer_reference": {
                "records": config.organizer_reference_records,
                "role": ORGANIZER_REFERENCE_ROLE,
                "sha256": reference_snapshot.sha256,
                "size_bytes": reference_snapshot.size_bytes,
                "unique_sequences": len(set(reference_sequences)),
            },
            "training_projection": {
                "accepted_contract_required": config.require_accepted_training_projection,
                "accepted_contract_satisfied": (
                    training_snapshot.sha256 == ACCEPTED_TRAINING_PROJECTION_SHA256
                    and len(training.rows) == ACCEPTED_TRAINING_PROJECTION_ROWS
                    and training_snapshot.size_bytes == ACCEPTED_TRAINING_PROJECTION_SIZE_BYTES
                ),
                "fold_4_data_access": False,
                "records": len(training.rows),
                "role": "generation_training_only",
                "sha256": training_snapshot.sha256,
                "size_bytes": training_snapshot.size_bytes,
                "trainer_visible_fields": [
                    "sequence_id",
                    "sequence",
                    "sampling_weight",
                ],
            },
        },
        "name": POOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": POOL_STATUS,
    }
    manifest = _canonical_json_bytes(manifest_document)
    payloads = {
        "candidates.fasta": candidates_fasta,
        "candidates.jsonl": candidates_jsonl,
        "family_metrics.json": family_metrics,
        "manifest.json": manifest,
    }
    sha256sums = "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")
    publication = {
        "candidates.fasta": candidates_fasta,
        "candidates.jsonl": candidates_jsonl,
        "family_metrics.json": family_metrics,
        "manifest.json": manifest,
        "SHA256SUMS": sha256sums,
    }

    _require_unchanged(config_path, config_snapshot, label="candidate-pool config")
    _require_unchanged(
        training_projection_path,
        training_snapshot,
        label="training projection",
    )
    _require_unchanged(
        organizer_reference_path,
        reference_snapshot,
        label="organizer reference FASTA",
    )
    output, verification = _publish_output(output_dir, publication)
    return CandidatePoolExecution(
        output_dir=output,
        candidate_count=verification.candidate_count,
        raw_proposal_count=len(proposals),
        candidates_jsonl_sha256=verification.candidates_jsonl_sha256,
        candidates_fasta_sha256=verification.candidates_fasta_sha256,
        family_metrics_sha256=verification.family_metrics_sha256,
        manifest_sha256=verification.manifest_sha256,
    )


def verify_candidate_pool_output(
    output_dir: str | Path,
    *,
    config_path: str | Path,
    training_projection_path: str | Path,
    organizer_reference_path: str | Path,
) -> CandidatePoolVerification:
    """Authenticate inputs and reproduce every byte of one candidate pool.

    Structural checks alone cannot prove compliance exclusions, raw proposal
    accounting, or learned-generator identities.  This verifier snapshots all
    three frozen inputs, performs a fresh deterministic construction in a
    private temporary directory, and requires exact equality of all five
    publication artifacts before returning a final stable structural snapshot.
    """

    observed_before = verify_candidate_pool_structure(output_dir)
    config_snapshot = _snapshot_regular(config_path, label="candidate-pool verifier config")
    training_snapshot = _snapshot_regular(
        training_projection_path,
        label="candidate-pool verifier training projection",
    )
    reference_snapshot: _Snapshot | None = None
    with tempfile.TemporaryDirectory(prefix="amp-candidate-pool-semantic-") as temporary:
        expected_output = Path(temporary) / "expected"
        try:
            build_candidate_pool(
                config_path=config_path,
                training_projection_path=training_projection_path,
                organizer_reference_path=organizer_reference_path,
                output_dir=expected_output,
            )
            # Keep the compliance-only reference outside the learned proposal
            # phase even in the deterministic reconstruction path.
            reference_snapshot = _snapshot_regular(
                organizer_reference_path,
                label="candidate-pool verifier organizer reference",
            )
            for name in _OUTPUT_NAMES:
                observed = _snapshot_regular(
                    observed_before.output_dir / name,
                    label=f"observed candidate-pool artifact {name}",
                )
                expected = _snapshot_regular(
                    expected_output / name,
                    label=f"reproduced candidate-pool artifact {name}",
                )
                if observed.payload != expected.payload or observed.sha256 != expected.sha256:
                    raise ValueError(
                        "candidate-pool publication does not match deterministic "
                        f"input-bound reconstruction: {name}"
                    )
        finally:
            try:
                expected_metadata = os.lstat(expected_output)
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISDIR(expected_metadata.st_mode) and not stat.S_ISLNK(
                    expected_metadata.st_mode
                ):
                    os.chmod(expected_output, 0o700)

    _require_unchanged(
        config_path,
        config_snapshot,
        label="candidate-pool verifier config",
    )
    _require_unchanged(
        training_projection_path,
        training_snapshot,
        label="candidate-pool verifier training projection",
    )
    if reference_snapshot is None:
        raise RuntimeError("semantic verifier did not authenticate the organizer reference")
    _require_unchanged(
        organizer_reference_path,
        reference_snapshot,
        label="candidate-pool verifier organizer reference",
    )
    observed_after = verify_candidate_pool_structure(output_dir)
    before_identity = (
        observed_before.candidate_count,
        observed_before.candidates_jsonl_sha256,
        observed_before.candidates_fasta_sha256,
        observed_before.family_metrics_sha256,
        observed_before.manifest_sha256,
    )
    after_identity = (
        observed_after.candidate_count,
        observed_after.candidates_jsonl_sha256,
        observed_after.candidates_fasta_sha256,
        observed_after.family_metrics_sha256,
        observed_after.manifest_sha256,
    )
    if after_identity != before_identity:
        raise ValueError("candidate-pool publication changed during semantic verification")
    return observed_after


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic, score-free AMP candidate_pool_v1 artifact."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-projection", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        execution = build_candidate_pool(
            config_path=args.config,
            training_projection_path=args.training_projection,
            organizer_reference_path=args.reference,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"candidate_pool_v1 failed: {error}", file=sys.stderr)
        return 2
    print(
        f"published {execution.candidate_count} eligible candidates from "
        f"{execution.raw_proposal_count} raw proposals to {execution.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

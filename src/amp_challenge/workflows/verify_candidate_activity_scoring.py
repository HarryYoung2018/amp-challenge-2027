"""Independently verify the development candidate-activity scoring bundle.

This module deliberately does not import the scoring producer, the baseline
model implementation, or the project's sequence and descriptor helpers.  It
reimplements the small transparent feature pipeline and deployment-time model
application so a producer bug cannot automatically validate itself.

The verifier is intentionally CPU-only and accepts ordinary small fixtures as
well as the immutable production-sized inputs.  It never writes inside the
verified output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SCHEMA_VERSION = 1
_CONFIG_ARTIFACT = "candidate_activity_scoring_development_v1"
_STATE_ARTIFACT = "candidate_activity_descriptor_model_states_v1"
_OOF_REPRODUCTION_ARTIFACT = "candidate_activity_descriptor_oof_reproduction_v1"
_OUTPUT_ARTIFACT = "candidate_activity_scoring_development_v1"
_OUTPUT_STATUS = "development_candidate_scoring_only"
_CONFIG_STATUS = "predeclared_development_candidate_scoring_only"
_OOF_MODEL = "descriptor_logistic"
_PROBABILITY_CALIBRATION = "none_raw_logistic_probability"
_PREDICTION_SCOPE = "declared_seven_target_activity_panel"
_TRAINING_SCOPE = "all_2492_accepted_gate1_contexts_after_recipe_freeze"
_TARGET_AGGREGATION = "equal_target_arithmetic_mean_within_declared_scope_v1"
_EXPECTED_CANDIDATES = 115_536
_EXPECTED_EXAMPLES = 2_492
_EXPECTED_FOLDS = 5
_EXPECTED_BATCH_SIZE = 2_048
_EXPECTED_OOF_ABSOLUTE_TOLERANCE = 1e-12
_FIT_SENSITIVITY = (
    "population_sd_across_five_outer_fold_complement_refits_diagnostic_not_posterior_or_"
    "aleatoric_uncertainty"
)
_OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
_FORBIDDEN_CLAIMS = (
    "aleatoric_uncertainty",
    "calibrated_uncertainty",
    "epistemic_uncertainty",
    "final_ranking",
    "hemolysis_risk",
    "mdr_eskape_activity",
    "out_of_distribution",
    "posterior_draws",
    "production_ensemble",
    "quality_probability",
    "selectivity",
)
_MODEL_WEIGHTS = {
    "apex": 0.0,
    "descriptor_logistic": 1.0,
    "esm_supervised": 0.0,
    "geometry": 0.0,
    "homology_knn": 0.0,
}
_FROZEN_DESCRIPTOR_SETTINGS = {
    "l2": 0.10,
    "max_iterations": 100,
    "tolerance": 1e-9,
    "prior_strength": 2.0,
}
_FROZEN_CANDIDATE_POOL_CONTRACT = {
    "producer_job_id": 225275,
    "git_commit": "50435d805a73ec4756cde1f7db1e6e7a06d30eb4",
    "publication_top_sha256": ("d248fe898629f83b9b0c5fc478a0d859dec46c95a82de663812a4b107eb2174c"),
    "manifest_sha256": "2f6ec8de3d65f9580aef12bbe9bf4b414f13a831e76b8dbc7504c158a3d91990",
    "candidates_sha256": "77315bfbec533f5587123714199c2dd2eb2a51a16d95957ab53927494983dbf2",
    "validation_summary_sha256": (
        "58e93fd08f670b0e8e8118bbd29a1a7c1378081fc17a646e87d3082abfbb135c"
    ),
    "final_publication_check_sha256": (
        "ae8d5d9ecfcfbd3d5007715a52cab0e7f2f598e1f709eda6468ba5385af91296"
    ),
}
_FROZEN_GATE1_CONTRACT = {
    "producer_job_id": 223248,
    "audit_job_id": 223250,
    "git_commit": "0468cc2cbc0b7c3b2a50f7866da1b1083c8ef1a8",
    "publication_top_sha256": ("4e259cfb43033c069598d42fd54fec49a67ba55bfbb5c1e77eeef4887815fe2f"),
    "semantic_top_sha256": ("556e06fd2b1af1e678de88008bc1b87434fb8cf1179f38c897de7ee2c9fd779d"),
    "examples_sha256": "d3eecbf3014fd78cf7021818466893d292315b6e90e77cea85d4e1fbd5bec520",
    "oof_sha256": "11eea907a242606b78261a9b37107647ce393db743be95c146cbbe5caced137c",
    "independent_receipt_sha256": (
        "1f6a6ce811d3fbecdbbbe7463e6052dc7766130372926d29952c70be65743e81"
    ),
}
_STATE_NAMES = (*(f"outer_fold_{fold}" for fold in range(5)), "all_data_deployment")
_OUTPUT_FILES = frozenset(
    {
        "candidate_activity_scores.csv",
        "fold_model_target_probabilities.npy",
        "model_states.json",
        "oof_reproduction.json",
        "manifest.json",
        "SHA256SUMS",
    }
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
_EXAMPLE_FIELDS = (
    "schema_version",
    "example_id",
    "assay_context_id",
    "sequence_id",
    "sequence",
    "canonical_target",
    "gram",
    "label",
    "source_observations",
    "fold",
    "homology_component_id",
    "union_component_id",
)
_OOF_FIELDS = (
    "model",
    "example_id",
    "assay_context_id",
    "sequence_id",
    "sequence",
    "canonical_target",
    "gram",
    "label",
    "source_observations",
    "fold",
    "homology_component_id",
    "union_component_id",
    "max_train_identity",
    "probability",
)
_DESCRIPTOR_SETTINGS = (
    "l2",
    "max_iterations",
    "tolerance",
    "prior_strength",
)
_DESCRIPTOR_NAMES = (
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
_STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_FEATURE_ORDER = _DESCRIPTOR_NAMES + tuple(
    f"composition_{residue}" for residue in _STANDARD_AMINO_ACIDS
)
_AMINO_ACID_SET = frozenset(_STANDARD_AMINO_ACIDS)
_MINIMUM_LENGTH = 8
_MAXIMUM_LENGTH = 50
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_COMPONENT_RE = re.compile(r"[^\x00-\x20\x7f]{1,256}")

_RESIDUE_MASSES_DA = {
    "A": 71.0788,
    "C": 103.1388,
    "D": 115.0886,
    "E": 129.1155,
    "F": 147.1766,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "K": 128.1741,
    "L": 113.1594,
    "M": 131.1926,
    "N": 114.1038,
    "P": 97.1167,
    "Q": 128.1307,
    "R": 156.1875,
    "S": 87.0782,
    "T": 101.1051,
    "V": 99.1326,
    "W": 186.2132,
    "Y": 163.1760,
}
_WATER_MASS_DA = 18.01528
_EISENBERG_HYDROPHOBICITY = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}
_HYDROPHOBIC_RESIDUES = frozenset("ACFILMVWY")
_AROMATIC_RESIDUES = frozenset("FWY")
_BASIC_RESIDUES = frozenset("HKR")
_ACIDIC_RESIDUES = frozenset("DE")
_POSITIVE_SIDECHAIN_PKA = {"H": 6.0, "K": 10.5, "R": 12.5}
_NEGATIVE_SIDECHAIN_PKA = {"C": 8.3, "D": 3.9, "E": 4.1, "Y": 10.1}
_N_TERMINUS_PKA = 8.0
_C_TERMINUS_PKA = 3.1

_GRAM_BY_TARGET = {
    "acinetobacter_baumannii": "negative",
    "escherichia_coli": "negative",
    "klebsiella_pneumoniae": "negative",
    "pseudomonas_aeruginosa": "negative",
    "enterococcus_faecalis": "positive",
    "enterococcus_faecium": "positive",
    "staphylococcus_aureus": "positive",
}
_OBJECTIVE_TARGET_GRAMS = {
    "broad_spectrum_activity": frozenset({"negative", "positive"}),
    "gram_positive_activity": frozenset({"positive"}),
    "gram_negative_activity": frozenset({"negative"}),
}
_FORBIDDEN_SCORE_ALIASES = (
    "std_",
    "_std",
    "uncertainty",
    "variance",
    "posterior",
    "epistemic",
    "aleatoric",
    "confidence_interval",
    "credible_interval",
    "toxicity",
    "hemolysis",
    "selectivity",
)


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Candidate:
    source_ordinal: int
    sequence_id: str
    sequence: str
    length: int
    library_eligible: bool
    generator_families: str
    generator_variants: str


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    canonical_target: str
    gram: str
    label: int
    source_observations: int
    fold: int
    homology_component_id: str
    union_component_id: str


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    sha256: str
    status: str
    automatic_production_eligible: bool
    candidate_jsonl_sha256: str
    expected_candidates: int
    gate1_examples_sha256: str
    expected_gate1_examples: int
    gate1_oof_sha256: str
    expected_gate1_oof_rows: int
    folds: int
    oof_absolute_tolerance: float
    scoring_chunk_size: int
    descriptor_settings: Mapping[str, int | float]
    target_order: tuple[str, ...]
    gram_by_target: Mapping[str, str]
    objectives: tuple[str, ...]
    target_aggregation: str
    calibration_scope: str
    fit_sensitivity_semantics: str
    training_scope: str
    forbidden_claims: tuple[str, ...]
    model_weights: Mapping[str, float]
    candidate_pool: Mapping[str, object]
    gate1: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModelState:
    name: str
    heldout_fold: int | None
    training_examples: int
    training_sequences: int
    training_union_components: int
    training_example_ids_sha256: str
    strains: tuple[str, ...]
    mean: FloatArray
    scale: FloatArray
    coefficient: FloatArray | None
    constant_probability: float | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_mapping(
    value: object,
    *,
    label: str,
    expected_keys: Sequence[str] | frozenset[str] | set[str],
) -> dict[str, object]:
    _require(isinstance(value, dict), f"{label} must be a JSON/TOML object")
    result = cast(dict[str, object], value)
    expected = set(expected_keys)
    actual = set(result)
    _require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )
    return result


def _exact_json_value(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int/float equality aliases."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            _exact_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _json_value(payload: bytes, *, label: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON number {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error


def _json_object(payload: bytes, *, label: str) -> dict[str, object]:
    value = _json_value(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return cast(dict[str, object], value)


def _canonical_json(value: object) -> bytes:
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


def _fingerprint(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in reversed((absolute, *absolute.parents)):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {candidate}") from error
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"{label} cannot traverse a symbolic link: {candidate}",
        )


def _snapshot(path: str | Path, *, label: str) -> Snapshot:
    requested = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(requested, label=label)
    try:
        named_before = os.lstat(requested)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {requested}") from error
    _require(not stat.S_ISLNK(named_before.st_mode), f"{label} must not be a symbolic link")
    _require(stat.S_ISREG(named_before.st_mode), f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {requested}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(requested)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    identities = {
        _fingerprint(item) for item in (named_before, opened_before, opened_after, named_after)
    }
    _require(
        len(identities) == 1,
        f"{label} changed while it was read",
    )
    payload = b"".join(chunks)
    _require(len(payload) == named_before.st_size, f"{label} size changed while it was read")
    _reject_symlink_chain(requested, label=label)
    return Snapshot(
        path=requested,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        fingerprint=_fingerprint(named_before),
    )


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    _reject_symlink_chain(snapshot.path, label=label)
    try:
        metadata = os.lstat(snapshot.path)
    except OSError as error:
        raise ValueError(f"cannot recheck {label}: {snapshot.path}") from error
    _require(
        stat.S_ISREG(metadata.st_mode) and _fingerprint(metadata) == snapshot.fingerprint,
        f"{label} changed during verification",
    )


def _safe_manifest_filename(value: str, *, label: str) -> str:
    _require(value != "", f"{label} contains an empty filename")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and len(pure.parts) == 1
        and pure.parts[0] not in {".", ".."}
        and "\\" not in value
        and "\x00" not in value,
        f"{label} contains an unsafe filename: {value!r}",
    )
    return value


def _parse_checksum_manifest(payload: bytes) -> Mapping[str, str]:
    _require(
        bool(payload) and payload.endswith(b"\n") and b"\r" not in payload,
        "SHA256SUMS must be non-empty LF-framed text",
    )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("SHA256SUMS must be ASCII") from error
    entries: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00-\x1f\x7f]+)", line)
        _require(match is not None, f"SHA256SUMS line {number} is malformed")
        assert match is not None
        digest, raw_name = match.groups()
        name = _safe_manifest_filename(raw_name, label="SHA256SUMS")
        _require(name != "SHA256SUMS", "SHA256SUMS cannot list itself")
        _require(name not in entries, f"SHA256SUMS repeats {name!r}")
        _require(previous is None or previous < name, "SHA256SUMS entries are not sorted")
        entries[name] = digest
        previous = name
    return entries


def _authenticate_output_tree(output_dir: str | Path) -> Mapping[str, Snapshot]:
    output = Path(output_dir)
    _reject_symlink_chain(output, label="candidate scoring output directory")
    try:
        metadata = os.lstat(output)
    except OSError as error:
        raise ValueError(f"cannot inspect candidate scoring output directory: {output}") from error
    _require(
        stat.S_ISDIR(metadata.st_mode),
        "candidate scoring output must be a real directory",
    )
    _require(
        stat.S_IMODE(metadata.st_mode) == 0o555,
        "candidate scoring output directory mode must be 0555",
    )
    try:
        names = {entry.name for entry in os.scandir(output)}
    except OSError as error:
        raise ValueError("cannot enumerate candidate scoring output") from error
    _require(
        names == set(_OUTPUT_FILES),
        "candidate scoring output inventory differs: "
        f"missing={sorted(set(_OUTPUT_FILES) - names)}, "
        f"extra={sorted(names - set(_OUTPUT_FILES))}",
    )
    top = _snapshot(output / "SHA256SUMS", label="candidate scoring SHA256SUMS")
    entries = _parse_checksum_manifest(top.payload)
    expected_entries = set(_OUTPUT_FILES) - {"SHA256SUMS"}
    _require(set(entries) == expected_entries, "SHA256SUMS inventory is not exact")
    snapshots: dict[str, Snapshot] = {"SHA256SUMS": top}
    for name in sorted(_OUTPUT_FILES):
        item_metadata = os.lstat(output / name)
        _require(
            stat.S_ISREG(item_metadata.st_mode) and stat.S_IMODE(item_metadata.st_mode) == 0o444,
            f"candidate scoring artifact {name} must be a regular 0444 file",
        )
    for name in sorted(entries):
        item = _snapshot(output / name, label=f"candidate scoring artifact {name}")
        _require(item.sha256 == entries[name], f"checksum mismatch for output artifact {name}")
        snapshots[name] = item
    return snapshots


def _sha256(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return cast(str, value)


def _positive_int(value: object, *, label: str) -> int:
    _require(type(value) is int and cast(int, value) > 0, f"{label} must be a positive integer")
    return cast(int, value)


def _nonnegative_int(value: object, *, label: str) -> int:
    _require(
        type(value) is int and cast(int, value) >= 0,
        f"{label} must be a non-negative integer",
    )
    return cast(int, value)


def _finite_float(value: object, *, label: str) -> float:
    _require(
        type(value) in {int, float} and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(cast(float, value))
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _csv_float(value: str | None, *, label: str) -> float:
    _require(
        value is not None and value != "" and value.strip() == value,
        f"{label} must be a canonical finite decimal",
    )
    try:
        result = float(cast(str, value))
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    _require(math.isfinite(result), f"{label} must be finite")
    _require(
        value == f"{result:.17g}" and not (result == 0.0 and value.startswith("-")),
        f"{label} must use canonical 17-digit decimal formatting",
    )
    return result


def _csv_int(value: str | None, *, label: str) -> int:
    _require(
        value is not None and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
        f"{label} must be a canonical non-negative integer",
    )
    return int(cast(str, value))


def _canonical_sequence(value: object, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    sequence = cast(str, value)
    try:
        sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be ASCII") from error
    _require(
        _MINIMUM_LENGTH <= len(sequence) <= _MAXIMUM_LENGTH,
        f"{label} length must be in [{_MINIMUM_LENGTH}, {_MAXIMUM_LENGTH}]",
    )
    _require(
        set(sequence) <= _AMINO_ACID_SET,
        f"{label} must contain only the 20 canonical uppercase amino acids",
    )
    return sequence


def _canonical_sequence_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _safe_name(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SAFE_NAME_RE.fullmatch(value) is not None,
        f"{label} is not a safe canonical name",
    )
    return cast(str, value)


def _parse_candidate_jsonl(snapshot: Snapshot) -> tuple[Candidate, ...]:
    payload = snapshot.payload
    _require(
        bool(payload) and payload.endswith(b"\n") and b"\r" not in payload,
        "candidate JSONL must be non-empty canonical LF-framed data",
    )
    candidates: list[Candidate] = []
    seen_ids: dict[str, str] = {}
    seen_sequences: set[str] = set()
    previous_sequence: str | None = None
    for ordinal, line in enumerate(payload.splitlines(keepends=True), 1):
        document = _exact_mapping(
            _json_value(line, label=f"candidate row {ordinal}"),
            label=f"candidate row {ordinal}",
            expected_keys=_CANDIDATE_FIELDS,
        )
        _require(
            document["schema_version"] == _SCHEMA_VERSION
            and type(document["schema_version"]) is int,
            f"candidate row {ordinal} schema_version must be 1",
        )
        sequence = _canonical_sequence(
            document["sequence"], label=f"candidate row {ordinal} sequence"
        )
        sequence_id = _sha256(document["sequence_id"], label=f"candidate row {ordinal} sequence_id")
        _require(
            sequence_id == _canonical_sequence_id(sequence),
            f"candidate row {ordinal} sequence_id is not its canonical hash",
        )
        _require(
            type(document["length"]) is int and document["length"] == len(sequence),
            f"candidate row {ordinal} length disagrees with sequence",
        )
        _require(
            document["library_eligible"] is True,
            f"candidate row {ordinal} must be library eligible",
        )
        lineages = document["lineages"]
        _require(
            isinstance(lineages, list) and bool(lineages),
            f"candidate row {ordinal} must contain lineages",
        )
        lineage_sort_keys: list[tuple[str, str, str, str, int, int]] = []
        families: set[str] = set()
        variants: set[str] = set()
        for lineage_ordinal, raw_lineage in enumerate(cast(list[object], lineages), 1):
            lineage = _exact_mapping(
                raw_lineage,
                label=f"candidate row {ordinal} lineage {lineage_ordinal}",
                expected_keys=_LINEAGE_FIELDS,
            )
            _require(
                lineage["schema_version"] == 1 and type(lineage["schema_version"]) is int,
                "candidate lineage schema_version must be 1",
            )
            family = _safe_name(lineage["generator_family"], label="generator_family")
            variant = _safe_name(lineage["generator_variant"], label="generator_variant")
            logical = _sha256(lineage["logical_sha256"], label="lineage logical_sha256")
            training_raw = lineage["training_projection_sha256"]
            _require(
                training_raw is None
                or (
                    isinstance(training_raw, str) and _SHA256_RE.fullmatch(training_raw) is not None
                ),
                "lineage training_projection_sha256 is invalid",
            )
            seed = _nonnegative_int(lineage["seed"], label="lineage seed")
            lineage_source_ordinal = _nonnegative_int(lineage["ordinal"], label="lineage ordinal")
            _require(
                seed < 2**64 and lineage_source_ordinal < 2**64,
                "candidate lineage seed/ordinal must fit unsigned 64-bit",
            )
            lineage_sort_keys.append(
                (
                    family,
                    variant,
                    logical,
                    cast(str | None, training_raw) or "",
                    seed,
                    lineage_source_ordinal,
                )
            )
            families.add(family)
            variants.add(variant)
        _require(
            lineage_sort_keys == sorted(set(lineage_sort_keys)),
            f"candidate row {ordinal} lineages are not unique canonical order",
        )
        _require(
            _canonical_json(document) == line,
            f"candidate row {ordinal} is not canonical JSON",
        )
        _require(sequence not in seen_sequences, f"duplicate candidate sequence at row {ordinal}")
        previous = seen_ids.get(sequence_id)
        _require(
            previous is None,
            (
                f"duplicate candidate sequence_id at row {ordinal}"
                if previous == sequence
                else f"candidate SHA-256 collision at row {ordinal}"
            ),
        )
        _require(
            previous_sequence is None or previous_sequence < sequence,
            "candidate rows must be strictly ascending by sequence",
        )
        seen_ids[sequence_id] = sequence
        seen_sequences.add(sequence)
        previous_sequence = sequence
        candidates.append(
            Candidate(
                source_ordinal=ordinal,
                sequence_id=sequence_id,
                sequence=sequence,
                length=len(sequence),
                library_eligible=True,
                generator_families="|".join(sorted(families)),
                generator_variants="|".join(sorted(variants)),
            )
        )
    return tuple(candidates)


def _parse_examples(snapshot: Snapshot, *, folds: int) -> tuple[Example, ...]:
    payload = snapshot.payload
    _require(
        bool(payload) and payload.endswith(b"\n") and b"\r" not in payload,
        "examples JSONL must be non-empty canonical LF-framed data",
    )
    examples: list[Example] = []
    seen: set[str] = set()
    previous: str | None = None
    sequence_metadata: dict[str, tuple[str, int]] = {}
    component_folds: dict[tuple[str, str], int] = {}
    folds_seen: set[int] = set()
    targets_seen: set[str] = set()
    for number, line in enumerate(payload.splitlines(keepends=True), 1):
        raw = _exact_mapping(
            _json_value(line, label=f"example row {number}"),
            label=f"example row {number}",
            expected_keys=_EXAMPLE_FIELDS,
        )
        _require(
            raw["schema_version"] == 1 and type(raw["schema_version"]) is int,
            f"example row {number} schema_version must be 1",
        )
        example_id = _sha256(raw["example_id"], label=f"example row {number} example_id")
        sequence = _canonical_sequence(raw["sequence"], label=f"example row {number} sequence")
        sequence_id = _sha256(raw["sequence_id"], label=f"example row {number} sequence_id")
        _require(
            sequence_id == _canonical_sequence_id(sequence),
            f"example row {number} sequence identity is invalid",
        )
        target = _safe_name(raw["canonical_target"], label="canonical_target")
        gram = cast(str, raw["gram"])
        _require(gram in {"negative", "positive"}, f"example row {number} gram is invalid")
        _require(
            _GRAM_BY_TARGET.get(target) == gram,
            f"example row {number} target/Gram mapping is invalid",
        )
        label = raw["label"]
        _require(type(label) is int and label in {0, 1}, f"example row {number} label is invalid")
        fold = raw["fold"]
        _require(
            type(fold) is int and 0 <= cast(int, fold) < folds,
            f"example row {number} fold is invalid",
        )
        source_observations = _positive_int(
            raw["source_observations"], label=f"example row {number} source_observations"
        )
        assay_context_id = _sha256(
            raw["assay_context_id"], label=f"example row {number} assay_context_id"
        )
        _require(
            example_id == assay_context_id,
            f"example row {number} example_id must equal assay_context_id",
        )
        component_values: list[str] = []
        for field in ("homology_component_id", "union_component_id"):
            value = raw[field]
            _require(
                isinstance(value, str) and _COMPONENT_RE.fullmatch(value) is not None,
                f"example row {number} {field} is invalid",
            )
            component = cast(str, value)
            prior_fold = component_folds.setdefault((field, component), cast(int, fold))
            _require(
                prior_fold == fold,
                f"example {field} {component!r} crosses outer folds",
            )
            component_values.append(component)
        homology_component_id, union_component_id = component_values
        prior_sequence = sequence_metadata.setdefault(sequence_id, (sequence, cast(int, fold)))
        _require(
            prior_sequence == (sequence, fold),
            "an example sequence identity changes sequence or outer fold",
        )
        _require(example_id not in seen, f"duplicate example_id at row {number}")
        _require(
            previous is None or previous < example_id,
            "examples must be strictly ascending by example_id",
        )
        _require(_canonical_json(raw) == line, f"example row {number} is not canonical JSON")
        seen.add(example_id)
        previous = example_id
        folds_seen.add(cast(int, fold))
        targets_seen.add(target)
        examples.append(
            Example(
                example_id=example_id,
                assay_context_id=assay_context_id,
                sequence_id=sequence_id,
                sequence=sequence,
                canonical_target=target,
                gram=gram,
                label=cast(int, label),
                source_observations=source_observations,
                fold=cast(int, fold),
                homology_component_id=homology_component_id,
                union_component_id=union_component_id,
            )
        )
    _require(folds_seen == set(range(folds)), "examples do not cover every outer fold")
    _require(targets_seen == set(_GRAM_BY_TARGET), "examples do not cover all seven targets")
    _require({item.label for item in examples} == {0, 1}, "examples must contain both labels")
    for heldout_fold in range(folds):
        _require(
            {item.label for item in examples if item.fold != heldout_fold} == {0, 1},
            f"fold-complement {heldout_fold} must contain both labels",
        )
    return tuple(examples)


def _parse_config(snapshot: Snapshot) -> ScoringConfig:
    _require(
        bool(snapshot.payload)
        and snapshot.payload.endswith(b"\n")
        and not snapshot.payload.endswith(b"\n\n")
        and b"\r" not in snapshot.payload,
        "scoring config must use LF framing with exactly one final LF",
    )
    try:
        raw_value = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"scoring config is invalid: {error}") from error
    raw = _exact_mapping(
        raw_value,
        label="scoring config",
        expected_keys={
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "expected_candidates",
            "expected_examples",
            "folds",
            "oof_absolute_tolerance",
            "batch_size",
            "objectives",
            "target_aggregation",
            "calibration_scope",
            "fit_sensitivity_semantics",
            "training_scope",
            "historical_method_choice_used_fold4",
            "organizer_reference_used_for_model_fit",
            "forbidden_claims",
            "candidate_pool",
            "gate1",
            "descriptor_logistic",
            "model_weights",
            "target",
        },
    )
    _require(
        raw["schema_version"] == 1 and type(raw["schema_version"]) is int,
        "scoring config schema_version must be 1",
    )
    _require(raw["artifact"] == _CONFIG_ARTIFACT, "scoring config artifact is invalid")
    _require(raw["status"] == _CONFIG_STATUS, "scoring config status is invalid")
    _require(
        raw["automatic_production_eligible"] is False,
        "scoring config cannot declare automatic production eligibility",
    )
    _require(
        raw["historical_method_choice_used_fold4"] is True,
        "scoring config must disclose historical fold-4 method-choice use",
    )
    _require(
        raw["organizer_reference_used_for_model_fit"] is False,
        "organizer reference cannot be used for model fitting",
    )
    folds = _positive_int(raw["folds"], label="config folds")
    _require(
        folds == _EXPECTED_FOLDS,
        "scoring config must use exactly five accepted outer folds",
    )
    expected_candidates = _positive_int(
        raw["expected_candidates"], label="config expected_candidates"
    )
    _require(
        expected_candidates == _EXPECTED_CANDIDATES,
        f"config expected_candidates must be the frozen value {_EXPECTED_CANDIDATES}",
    )
    expected_examples = _positive_int(raw["expected_examples"], label="config expected_examples")
    _require(
        expected_examples == _EXPECTED_EXAMPLES,
        f"config expected_examples must be the frozen value {_EXPECTED_EXAMPLES}",
    )
    batch_size = _positive_int(raw["batch_size"], label="config batch_size")
    _require(
        batch_size == _EXPECTED_BATCH_SIZE,
        f"config batch_size must be the frozen value {_EXPECTED_BATCH_SIZE}",
    )
    tolerance = _finite_float(raw["oof_absolute_tolerance"], label="config oof_absolute_tolerance")
    _require(
        tolerance == _EXPECTED_OOF_ABSOLUTE_TOLERANCE,
        "config OOF absolute tolerance differs from the frozen scoring recipe",
    )
    descriptor_raw = _exact_mapping(
        raw["descriptor_logistic"],
        label="config descriptor_logistic",
        expected_keys=_DESCRIPTOR_SETTINGS,
    )
    descriptor_settings: dict[str, int | float] = {
        "l2": _finite_float(descriptor_raw["l2"], label="descriptor l2"),
        "max_iterations": _positive_int(
            descriptor_raw["max_iterations"], label="descriptor max_iterations"
        ),
        "tolerance": _finite_float(descriptor_raw["tolerance"], label="descriptor tolerance"),
        "prior_strength": _finite_float(
            descriptor_raw["prior_strength"], label="descriptor prior_strength"
        ),
    }
    _require(cast(float, descriptor_settings["l2"]) > 0.0, "descriptor l2 must be positive")
    _require(
        cast(float, descriptor_settings["tolerance"]) > 0.0,
        "descriptor tolerance must be positive",
    )
    _require(
        cast(float, descriptor_settings["prior_strength"]) >= 0.0,
        "descriptor prior_strength must be non-negative",
    )
    _require(
        descriptor_settings == _FROZEN_DESCRIPTOR_SETTINGS,
        "descriptor settings differ from the frozen scoring recipe",
    )
    objectives_raw = raw["objectives"]
    _require(
        isinstance(objectives_raw, list) and tuple(objectives_raw) == _OBJECTIVES,
        "scoring config objectives are invalid",
    )
    _require(
        raw["target_aggregation"] == _TARGET_AGGREGATION,
        "scoring config target aggregation policy is invalid",
    )
    _require(
        raw["calibration_scope"] == _PROBABILITY_CALIBRATION,
        "scoring config calibration scope is invalid",
    )
    _require(raw["training_scope"] == _TRAINING_SCOPE, "scoring training scope is invalid")
    _require(
        raw["fit_sensitivity_semantics"] == _FIT_SENSITIVITY,
        "scoring model-fit sensitivity semantics are invalid",
    )
    forbidden_raw = raw["forbidden_claims"]
    _require(
        isinstance(forbidden_raw, list) and tuple(forbidden_raw) == _FORBIDDEN_CLAIMS,
        "scoring forbidden claims are invalid",
    )
    weights_raw = _exact_mapping(
        raw["model_weights"],
        label="config model_weights",
        expected_keys=set(_MODEL_WEIGHTS),
    )
    model_weights = {
        name: _finite_float(weights_raw[name], label=f"config model_weights.{name}")
        for name in sorted(_MODEL_WEIGHTS)
    }
    _require(model_weights == _MODEL_WEIGHTS, "scoring model weights are invalid")

    candidate_pool = _exact_mapping(
        raw["candidate_pool"],
        label="config candidate_pool",
        expected_keys={
            "producer_job_id",
            "git_commit",
            "publication_top_sha256",
            "manifest_sha256",
            "candidates_sha256",
            "validation_summary_sha256",
            "final_publication_check_sha256",
        },
    )
    _positive_int(candidate_pool["producer_job_id"], label="candidate_pool producer_job_id")
    _require(
        isinstance(candidate_pool["git_commit"], str)
        and _GIT_SHA_RE.fullmatch(cast(str, candidate_pool["git_commit"])) is not None,
        "candidate_pool git_commit must be a full lowercase Git SHA",
    )
    for name in (
        "publication_top_sha256",
        "manifest_sha256",
        "candidates_sha256",
        "validation_summary_sha256",
        "final_publication_check_sha256",
    ):
        _sha256(candidate_pool[name], label=f"candidate_pool {name}")
    _require(
        candidate_pool == _FROZEN_CANDIDATE_POOL_CONTRACT,
        "candidate_pool contract differs from the accepted frozen publication",
    )

    gate1 = _exact_mapping(
        raw["gate1"],
        label="config gate1",
        expected_keys={
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "publication_top_sha256",
            "semantic_top_sha256",
            "examples_sha256",
            "oof_sha256",
            "independent_receipt_sha256",
        },
    )
    _positive_int(gate1["producer_job_id"], label="gate1 producer_job_id")
    _positive_int(gate1["audit_job_id"], label="gate1 audit_job_id")
    _require(
        isinstance(gate1["git_commit"], str)
        and _GIT_SHA_RE.fullmatch(cast(str, gate1["git_commit"])) is not None,
        "gate1 git_commit must be a full lowercase Git SHA",
    )
    for name in (
        "publication_top_sha256",
        "semantic_top_sha256",
        "examples_sha256",
        "oof_sha256",
        "independent_receipt_sha256",
    ):
        _sha256(gate1[name], label=f"gate1 {name}")
    _require(
        gate1 == _FROZEN_GATE1_CONTRACT,
        "gate1 contract differs from the accepted frozen publication",
    )

    targets_raw = raw["target"]
    _require(isinstance(targets_raw, list), "config targets must be an array of tables")
    target_order: list[str] = []
    gram_by_target: dict[str, str] = {}
    for index, item in enumerate(cast(list[object], targets_raw)):
        target = _exact_mapping(
            item,
            label=f"config target {index}",
            expected_keys={"name", "gram"},
        )
        name = _safe_name(target["name"], label=f"config target {index} name")
        gram = target["gram"]
        _require(
            isinstance(gram, str) and gram in {"negative", "positive"},
            f"config target {index} gram is invalid",
        )
        _require(name not in gram_by_target, f"duplicate config target: {name}")
        _require(
            _GRAM_BY_TARGET.get(name) == gram,
            f"config target {name} has an invalid Gram assignment",
        )
        target_order.append(name)
        gram_by_target[name] = cast(str, gram)
    _require(
        len(target_order) == 7 and set(target_order) == set(_GRAM_BY_TARGET),
        "config must contain the exact seven-target activity panel",
    )
    _require(
        tuple(target_order) == tuple(sorted(target_order)),
        "config targets must be strictly alphabetical",
    )
    _require(
        sum(gram == "negative" for gram in gram_by_target.values()) == 4
        and sum(gram == "positive" for gram in gram_by_target.values()) == 3,
        "config target Gram census is invalid",
    )
    return ScoringConfig(
        sha256=snapshot.sha256,
        status=cast(str, raw["status"]),
        automatic_production_eligible=False,
        candidate_jsonl_sha256=cast(str, candidate_pool["candidates_sha256"]),
        expected_candidates=expected_candidates,
        gate1_examples_sha256=cast(str, gate1["examples_sha256"]),
        expected_gate1_examples=expected_examples,
        gate1_oof_sha256=cast(str, gate1["oof_sha256"]),
        expected_gate1_oof_rows=expected_examples * 3,
        folds=folds,
        oof_absolute_tolerance=tolerance,
        scoring_chunk_size=batch_size,
        descriptor_settings=descriptor_settings,
        target_order=tuple(target_order),
        gram_by_target=gram_by_target,
        objectives=_OBJECTIVES,
        target_aggregation=_TARGET_AGGREGATION,
        calibration_scope=_PROBABILITY_CALIBRATION,
        fit_sensitivity_semantics=_FIT_SENSITIVITY,
        training_scope=_TRAINING_SCOPE,
        forbidden_claims=_FORBIDDEN_CLAIMS,
        model_weights=model_weights,
        candidate_pool=dict(candidate_pool),
        gate1=dict(gate1),
    )


def _parse_oof(
    snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    expected_rows: int,
) -> Mapping[str, float]:
    _require(
        bool(snapshot.payload)
        and snapshot.payload.endswith(b"\n")
        and not snapshot.payload.endswith(b"\n\n")
        and b"\n\n" not in snapshot.payload
        and b"\r" not in snapshot.payload,
        "OOF CSV must be non-empty LF-framed data",
    )
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("OOF CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require(reader.fieldnames == list(_OOF_FIELDS), "OOF CSV header is invalid")
    by_example = {item.example_id: item for item in examples}
    descriptor: dict[str, float] = {}
    row_count = 0
    seen_pairs: set[tuple[str, str]] = set()
    models_by_example: dict[str, set[str]] = {item.example_id: set() for item in examples}
    accepted_models = {"descriptor_logistic", "equal_weight_ensemble", "homology_knn"}
    previous_pair: tuple[str, str] | None = None
    for row_count, row in enumerate(reader, 1):
        _require(
            None not in row and all(value is not None for value in row.values()),
            f"OOF row {row_count} has an invalid field count",
        )
        model = cast(str, row["model"])
        example_id = cast(str, row["example_id"])
        pair = (model, example_id)
        _require(model in accepted_models, f"OOF row {row_count} has an unknown model")
        _require(
            pair not in seen_pairs and (previous_pair is None or previous_pair < pair),
            "OOF rows must be unique and strictly ordered by model/example_id",
        )
        seen_pairs.add(pair)
        previous_pair = pair
        _require(example_id in by_example, f"OOF row {row_count} has unknown example_id")
        example = by_example[example_id]
        exact_values = {
            "assay_context_id": example.assay_context_id,
            "sequence_id": example.sequence_id,
            "sequence": example.sequence,
            "canonical_target": example.canonical_target,
            "gram": example.gram,
            "label": str(example.label),
            "source_observations": str(example.source_observations),
            "fold": str(example.fold),
            "homology_component_id": example.homology_component_id,
            "union_component_id": example.union_component_id,
        }
        for name, expected in exact_values.items():
            _require(
                row[name] == expected,
                f"OOF row {row_count} {name} disagrees with examples JSONL",
            )
        maximum_identity = _csv_float(
            row["max_train_identity"], label=f"OOF row {row_count} max_train_identity"
        )
        _require(
            0.0 <= maximum_identity <= 1.0,
            f"OOF row {row_count} max_train_identity is outside [0, 1]",
        )
        probability = _csv_float(row["probability"], label=f"OOF row {row_count} probability")
        _require(
            0.0 <= probability <= 1.0,
            f"OOF row {row_count} probability is outside [0, 1]",
        )
        if model == _OOF_MODEL:
            descriptor[example_id] = probability
        models_by_example[example_id].add(model)
    _require(row_count == expected_rows, f"OOF CSV must contain exactly {expected_rows} rows")
    _require(
        all(models == accepted_models for models in models_by_example.values()),
        "OOF CSV must contain exactly three accepted models per example",
    )
    _require(
        set(descriptor) == set(by_example),
        "descriptor OOF rows do not exactly cover the examples",
    )
    return descriptor


def _net_charge(sequence: str, *, ph: float = 7.4) -> float:
    positive = 1.0 / (1.0 + 10.0 ** (ph - _N_TERMINUS_PKA))
    negative = 1.0 / (1.0 + 10.0 ** (_C_TERMINUS_PKA - ph))
    for residue, pka in _POSITIVE_SIDECHAIN_PKA.items():
        positive += sequence.count(residue) / (1.0 + 10.0 ** (ph - pka))
    for residue, pka in _NEGATIVE_SIDECHAIN_PKA.items():
        negative += sequence.count(residue) / (1.0 + 10.0 ** (pka - ph))
    return positive - negative


def _isoelectric_point(sequence: str) -> float:
    lower = 0.0
    upper = 14.0
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        if _net_charge(sequence, ph=midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _sequence_features(sequence: str) -> FloatArray:
    length = len(sequence)
    charge = _net_charge(sequence)
    hydrophobicities = np.asarray(
        [_EISENBERG_HYDROPHOBICITY[residue] for residue in sequence],
        dtype=np.float64,
    )
    angles = np.deg2rad(np.arange(length, dtype=np.float64) * 100.0)
    x_component = float(np.sum(hydrophobicities * np.cos(angles)))
    y_component = float(np.sum(hydrophobicities * np.sin(angles)))
    counts = {residue: sequence.count(residue) for residue in set(sequence)}
    entropy_counts = np.asarray(
        [sequence.count(residue) for residue in sorted(set(sequence))],
        dtype=np.float64,
    )
    entropy_probabilities = entropy_counts / length
    descriptors = (
        float(length),
        float(sum(_RESIDUE_MASSES_DA[residue] for residue in sequence) + _WATER_MASS_DA),
        charge,
        charge / length,
        _isoelectric_point(sequence),
        float(np.mean(hydrophobicities)),
        math.hypot(x_component, y_component) / length,
        sum(residue in _HYDROPHOBIC_RESIDUES for residue in sequence) / length,
        sum(residue in _AROMATIC_RESIDUES for residue in sequence) / length,
        sum(residue in _BASIC_RESIDUES for residue in sequence) / length,
        sum(residue in _ACIDIC_RESIDUES for residue in sequence) / length,
        float(-np.sum(entropy_probabilities * np.log2(entropy_probabilities))),
        max(counts.values()) / length,
    )
    composition = tuple(sequence.count(residue) / length for residue in _STANDARD_AMINO_ACIDS)
    return np.asarray(descriptors + composition, dtype=np.float64)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _apply_state(state: ModelState, *, sequence: str, target: str, gram: str) -> float:
    return _apply_state_features(
        state,
        features=_sequence_features(sequence),
        target=target,
        gram=gram,
    )


def _apply_state_features(
    state: ModelState,
    *,
    features: FloatArray,
    target: str,
    gram: str,
) -> float:
    if state.constant_probability is not None:
        return state.constant_probability
    assert state.coefficient is not None
    continuous = (features - state.mean) / state.scale
    strain = np.zeros(len(state.strains) + 1, dtype=np.float64)
    try:
        strain_index = state.strains.index(target)
    except ValueError:
        strain_index = len(state.strains)
    strain[strain_index] = 1.0
    gram_vector = np.zeros(3, dtype=np.float64)
    gram_vector[{"positive": 0, "negative": 1, "unknown": 2}[gram]] = 1.0
    design = np.concatenate((np.ones(1, dtype=np.float64), continuous, strain, gram_vector))
    probability = _sigmoid(float(design @ state.coefficient))
    return float(np.clip(probability, 1e-6, 1.0 - 1e-6))


def _float_vector(value: object, *, label: str, length: int) -> FloatArray:
    _require(isinstance(value, list), f"{label} must be a JSON array")
    raw = cast(list[object], value)
    _require(len(raw) == length, f"{label} must have length {length}")
    result = np.asarray(
        [_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(raw)],
        dtype=np.float64,
    )
    _require(result.shape == (length,), f"{label} has the wrong shape")
    return result


def _ordered_identifier_digest(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _training_examples_for_state(
    examples: Sequence[Example], *, heldout_fold: int | None
) -> tuple[Example, ...]:
    selected = (
        tuple(examples)
        if heldout_fold is None
        else tuple(item for item in examples if item.fold != heldout_fold)
    )
    _require(bool(selected), "serialized model state has an empty training partition")
    return selected


def _independent_fit_state(
    *,
    training: Sequence[Example],
    strains: Sequence[str],
    mean: FloatArray,
    scale: FloatArray,
    settings: Mapping[str, int | float],
) -> tuple[FloatArray | None, float | None]:
    features = np.asarray([_sequence_features(item.sequence) for item in training])
    standardized = (features - mean) / scale
    strain_index = {strain: index for index, strain in enumerate(strains)}
    strain = np.zeros((len(training), len(strains) + 1), dtype=np.float64)
    gram = np.zeros((len(training), 3), dtype=np.float64)
    gram_index = {"positive": 0, "negative": 1, "unknown": 2}
    for row_index, item in enumerate(training):
        strain[row_index, strain_index.get(item.canonical_target, len(strains))] = 1.0
        gram[row_index, gram_index[item.gram]] = 1.0
    design = np.concatenate(
        (np.ones((len(training), 1), dtype=np.float64), standardized, strain, gram),
        axis=1,
    )
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    prior_strength = float(settings["prior_strength"])
    prior = float((np.sum(labels) + 0.5 * prior_strength) / (labels.size + prior_strength))
    if np.all(labels == labels[0]):
        return None, prior
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    l2 = float(settings["l2"])
    penalty = np.full(coefficient.size, l2, dtype=np.float64)
    penalty[0] = 0.0
    converged = False
    for _ in range(int(settings["max_iterations"])):
        logits = design @ coefficient
        probability = np.empty_like(logits)
        positive = logits >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        probability[~positive] = exponent / (1.0 + exponent)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - labels) / labels.size + penalty * coefficient
        hessian = (design.T * variance) @ design / labels.size
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        _require(np.all(np.isfinite(step)), "independent descriptor fit produced invalid step")
        coefficient -= step
        _require(
            np.all(np.isfinite(coefficient)),
            "independent descriptor fit produced invalid coefficients",
        )
        if float(np.max(np.abs(step))) <= float(settings["tolerance"]):
            converged = True
            break
    _require(converged, "independent descriptor fit did not converge")
    return coefficient, None


def _parse_model_states(
    snapshot: Snapshot,
    *,
    config: ScoringConfig,
    examples: Sequence[Example],
) -> tuple[ModelState, ...]:
    document = _exact_mapping(
        _json_object(snapshot.payload, label="model_states.json"),
        label="model_states.json",
        expected_keys={
            "schema_version",
            "artifact",
            "descriptor_settings",
            "target_order",
            "feature_order",
            "states",
        },
    )
    _require(
        _canonical_json(document) == snapshot.payload,
        "model_states.json is not canonical compact LF-framed JSON",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "model_states.json schema_version must be 1",
    )
    _require(document["artifact"] == _STATE_ARTIFACT, "model state artifact is invalid")
    settings = _exact_mapping(
        document["descriptor_settings"],
        label="model state descriptor_settings",
        expected_keys=_DESCRIPTOR_SETTINGS,
    )
    for name in _DESCRIPTOR_SETTINGS:
        expected = config.descriptor_settings[name]
        actual = settings[name]
        if name == "max_iterations":
            _require(
                type(actual) is int and actual == expected,
                f"model state descriptor setting {name} differs from config",
            )
        else:
            _require(
                _finite_float(actual, label=f"model state descriptor {name}") == float(expected),
                f"model state descriptor setting {name} differs from config",
            )
    target_order = document["target_order"]
    feature_order = document["feature_order"]
    _require(
        target_order == list(config.target_order),
        "model state target order differs from config",
    )
    _require(feature_order == list(_FEATURE_ORDER), "model state feature order is invalid")
    raw_states = document["states"]
    _require(isinstance(raw_states, list), "model state states must be a JSON array")
    _require(len(raw_states) == 6, "model_states.json must contain exactly six states")
    states: list[ModelState] = []
    for index, raw_state in enumerate(cast(list[object], raw_states)):
        state = _exact_mapping(
            raw_state,
            label=f"model state {index}",
            expected_keys={
                "name",
                "heldout_fold",
                "training_examples",
                "training_sequences",
                "training_union_components",
                "training_example_ids_sha256",
                "strains",
                "mean",
                "scale",
                "coefficient",
                "constant_probability",
            },
        )
        expected_name = _STATE_NAMES[index]
        _require(state["name"] == expected_name, f"model state {index} name/order is invalid")
        expected_heldout = index if index < config.folds else None
        _require(
            state["heldout_fold"] == expected_heldout
            and (state["heldout_fold"] is None or type(state["heldout_fold"]) is int),
            f"model state {expected_name} heldout_fold is invalid",
        )
        training = _training_examples_for_state(examples, heldout_fold=expected_heldout)
        training_ids = tuple(item.example_id for item in training)
        expected_counts = (
            len(training),
            len({item.sequence_id for item in training}),
            len({item.union_component_id for item in training}),
        )
        observed_counts = (
            _positive_int(
                state["training_examples"],
                label=f"model state {expected_name} training_examples",
            ),
            _positive_int(
                state["training_sequences"],
                label=f"model state {expected_name} training_sequences",
            ),
            _positive_int(
                state["training_union_components"],
                label=f"model state {expected_name} training_union_components",
            ),
        )
        _require(
            observed_counts == expected_counts,
            f"model state {expected_name} training census is invalid",
        )
        observed_digest = _sha256(
            state["training_example_ids_sha256"],
            label=f"model state {expected_name} training_example_ids_sha256",
        )
        _require(
            observed_digest == _ordered_identifier_digest(training_ids),
            f"model state {expected_name} training example digest is invalid",
        )
        strains_raw = state["strains"]
        _require(
            isinstance(strains_raw, list) and all(isinstance(item, str) for item in strains_raw),
            f"model state {expected_name} strains must be strings",
        )
        strains = tuple(cast(list[str], strains_raw))
        expected_strains = tuple(sorted({item.canonical_target for item in training}))
        _require(
            strains == expected_strains,
            f"model state {expected_name} strains differ from its training support",
        )
        mean = _float_vector(
            state["mean"], label=f"model state {expected_name} mean", length=len(_FEATURE_ORDER)
        )
        scale = _float_vector(
            state["scale"],
            label=f"model state {expected_name} scale",
            length=len(_FEATURE_ORDER),
        )
        _require(np.all(scale > 0.0), f"model state {expected_name} scale must be positive")
        feature_matrix = np.asarray(
            [_sequence_features(item.sequence) for item in training], dtype=np.float64
        )
        expected_mean = np.mean(feature_matrix, axis=0)
        expected_scale = np.std(feature_matrix, axis=0)
        expected_scale[expected_scale < 1e-12] = 1.0
        _require(
            np.allclose(mean, expected_mean, rtol=0.0, atol=1e-12),
            f"model state {expected_name} mean is not reproduced independently",
        )
        _require(
            np.allclose(scale, expected_scale, rtol=0.0, atol=1e-12),
            f"model state {expected_name} scale is not reproduced independently",
        )
        coefficient_raw = state["coefficient"]
        constant_raw = state["constant_probability"]
        expected_coefficient_count = 1 + len(_FEATURE_ORDER) + len(strains) + 1 + 3
        if constant_raw is None:
            coefficient = _float_vector(
                coefficient_raw,
                label=f"model state {expected_name} coefficient",
                length=expected_coefficient_count,
            )
            constant_probability = None
        else:
            _require(
                coefficient_raw is None,
                f"constant model state {expected_name} must not contain coefficients",
            )
            constant_probability = _finite_float(
                constant_raw,
                label=f"model state {expected_name} constant_probability",
            )
            _require(
                0.0 <= constant_probability <= 1.0,
                f"model state {expected_name} constant probability is outside [0, 1]",
            )
            coefficient = None
        reproduced_coefficient, reproduced_constant = _independent_fit_state(
            training=training,
            strains=strains,
            mean=mean,
            scale=scale,
            settings=config.descriptor_settings,
        )
        if coefficient is not None:
            _require(
                reproduced_coefficient is not None
                and np.allclose(coefficient, reproduced_coefficient, rtol=0.0, atol=1e-12),
                f"model state {expected_name} coefficients are not independently reproduced",
            )
        else:
            _require(
                reproduced_constant is not None
                and constant_probability is not None
                and abs(constant_probability - reproduced_constant) <= 1e-15,
                f"model state {expected_name} constant probability is not reproduced",
            )
        states.append(
            ModelState(
                name=expected_name,
                heldout_fold=expected_heldout,
                training_examples=observed_counts[0],
                training_sequences=observed_counts[1],
                training_union_components=observed_counts[2],
                training_example_ids_sha256=observed_digest,
                strains=strains,
                mean=mean,
                scale=scale,
                coefficient=coefficient,
                constant_probability=constant_probability,
            )
        )
    return tuple(states)


def _reproduce_oof(
    *,
    examples: Sequence[Example],
    accepted_probabilities: Mapping[str, float],
    states: Sequence[ModelState],
    tolerance: float,
) -> tuple[float, tuple[dict[str, int | float], ...]]:
    state_by_fold = {
        cast(int, state.heldout_fold): state for state in states if state.heldout_fold is not None
    }
    _require(set(state_by_fold) == set(range(5)), "outer-fold state support is incomplete")
    maximum = 0.0
    by_fold: list[dict[str, int | float]] = []
    for fold in range(5):
        fold_errors: list[float] = []
        for example in examples:
            if example.fold != fold:
                continue
            reproduced = _apply_state(
                state_by_fold[fold],
                sequence=example.sequence,
                target=example.canonical_target,
                gram=example.gram,
            )
            error = abs(reproduced - accepted_probabilities[example.example_id])
            _require(
                error <= tolerance,
                f"outer-fold state {fold} does not reproduce accepted descriptor OOF "
                f"for {example.example_id}: error={error:.17g}",
            )
            fold_errors.append(error)
        _require(bool(fold_errors), f"fold {fold} has no OOF examples")
        fold_maximum = max(fold_errors)
        maximum = max(maximum, fold_maximum)
        by_fold.append(
            {
                "fold": fold,
                "heldout_examples": len(fold_errors),
                "training_examples": sum(item.fold != fold for item in examples),
                "maximum_absolute_error": fold_maximum,
            }
        )
    return maximum, tuple(by_fold)


def _load_fold_tensor(
    snapshot: Snapshot,
    *,
    candidate_count: int,
    folds: int,
    target_count: int,
) -> FloatArray:
    stream = io.BytesIO(snapshot.payload)
    try:
        value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("fold-model probability artifact is not a valid NPY array") from error
    _require(
        stream.tell() == len(snapshot.payload),
        "fold-model probability NPY contains trailing bytes",
    )
    _require(isinstance(value, np.ndarray), "fold-model probability artifact must be an array")
    array = cast(NDArray[Any], value)
    _require(
        array.dtype.str == "<f8",
        "fold-model probability tensor must have exact little-endian float64 dtype",
    )
    _require(array.flags.c_contiguous, "fold-model probability tensor must be C-contiguous")
    expected_shape = (candidate_count, folds, target_count)
    _require(
        array.shape == expected_shape,
        f"fold-model probability tensor shape must be {expected_shape}, got {array.shape}",
    )
    probabilities = np.asarray(array, dtype=np.float64)
    _require(
        np.all(np.isfinite(probabilities)),
        "fold-model probability tensor contains a non-finite value",
    )
    _require(
        np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
        "fold-model probability tensor contains a value outside [0, 1]",
    )
    return probabilities


def _objective_indices(config: ScoringConfig) -> Mapping[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for objective, grams in _OBJECTIVE_TARGET_GRAMS.items():
        indices = tuple(
            index
            for index, target in enumerate(config.target_order)
            if config.gram_by_target[target] in grams
        )
        _require(bool(indices), f"objective {objective} has no target support")
        result[objective] = indices
    return result


def _expected_score_columns(config: ScoringConfig) -> tuple[str, ...]:
    return (
        "source_ordinal",
        "sequence_id",
        "sequence",
        "length",
        "library_eligible",
        "generator_families",
        "generator_variants",
        *(f"probability_{target}" for target in config.target_order),
        *(f"mean_{objective}" for objective in _OBJECTIVES),
        *(f"model_fit_sensitivity_{objective}" for objective in _OBJECTIVES),
        "prediction_scope",
        "training_scope",
        "calibration_scope",
        "model_release_id",
    )


def _validate_no_forbidden_score_aliases(columns: Sequence[str]) -> None:
    expected_sensitivity = {f"model_fit_sensitivity_{objective}" for objective in _OBJECTIVES}
    for column in columns:
        lowered = column.lower()
        if column in expected_sensitivity or column == "calibration_scope":
            continue
        for forbidden in _FORBIDDEN_SCORE_ALIASES:
            _require(
                forbidden not in lowered,
                f"score CSV uses forbidden endpoint or uncertainty alias {column!r}",
            )


def _parse_boolean_csv(value: str | None, *, label: str) -> bool:
    _require(value is not None, f"{label} is missing")
    _require(value in {"true", "false"}, f"{label} must be canonical true or false")
    return value == "true"


def _score_rows_and_reproduce_candidates(
    snapshot: Snapshot,
    *,
    candidates: Sequence[Candidate],
    config: ScoringConfig,
    states: Sequence[ModelState],
    fold_tensor: FloatArray,
    model_release_id: str,
    arithmetic_tolerance: float,
) -> tuple[float, float]:
    _require(
        bool(snapshot.payload)
        and snapshot.payload.endswith(b"\n")
        and not snapshot.payload.endswith(b"\n\n")
        and b"\n\n" not in snapshot.payload
        and b"\r" not in snapshot.payload,
        "candidate score CSV must be non-empty LF-framed data",
    )
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("candidate score CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    expected_columns = _expected_score_columns(config)
    _require(reader.fieldnames == list(expected_columns), "candidate score CSV header is invalid")
    _validate_no_forbidden_score_aliases(expected_columns)
    objectives = _objective_indices(config)
    fold_states = states[: config.folds]
    deployment_state = states[-1]
    maximum_state_error = 0.0
    maximum_arithmetic_error = 0.0
    row_count = 0
    for row_count, row in enumerate(reader, 1):
        _require(None not in row, f"candidate score row {row_count} has extra columns")
        _require(row_count <= len(candidates), "candidate score CSV has extra rows")
        candidate = candidates[row_count - 1]
        exact_values = {
            "source_ordinal": str(candidate.source_ordinal),
            "sequence_id": candidate.sequence_id,
            "sequence": candidate.sequence,
            "length": str(candidate.length),
            "generator_families": candidate.generator_families,
            "generator_variants": candidate.generator_variants,
            "prediction_scope": _PREDICTION_SCOPE,
            "training_scope": config.training_scope,
            "calibration_scope": config.calibration_scope,
            "model_release_id": model_release_id,
        }
        for name, expected in exact_values.items():
            _require(
                row[name] == expected,
                f"candidate score row {row_count} {name} differs from authenticated input",
            )
        _require(
            _parse_boolean_csv(
                row["library_eligible"],
                label=f"candidate score row {row_count} library_eligible",
            )
            is candidate.library_eligible,
            f"candidate score row {row_count} eligibility differs from source",
        )
        features = _sequence_features(candidate.sequence)
        deployment_probabilities = np.empty(len(config.target_order), dtype=np.float64)
        reproduced_folds = np.empty((config.folds, len(config.target_order)), dtype=np.float64)
        for target_index, target in enumerate(config.target_order):
            gram = config.gram_by_target[target]
            deployment_probabilities[target_index] = _apply_state_features(
                deployment_state,
                features=features,
                target=target,
                gram=gram,
            )
            for fold, state in enumerate(fold_states):
                reproduced_folds[fold, target_index] = _apply_state_features(
                    state,
                    features=features,
                    target=target,
                    gram=gram,
                )
        state_error = float(np.max(np.abs(reproduced_folds - fold_tensor[row_count - 1, :, :])))
        maximum_state_error = max(maximum_state_error, state_error)
        _require(
            state_error <= arithmetic_tolerance,
            f"candidate score row {row_count} tensor values do not reproduce from model states",
        )
        csv_probabilities = np.asarray(
            [
                _csv_float(
                    row[f"probability_{target}"],
                    label=f"candidate score row {row_count} probability_{target}",
                )
                for target in config.target_order
            ],
            dtype=np.float64,
        )
        _require(
            np.all((csv_probabilities >= 0.0) & (csv_probabilities <= 1.0)),
            f"candidate score row {row_count} has probability outside [0, 1]",
        )
        probability_error = float(np.max(np.abs(csv_probabilities - deployment_probabilities)))
        maximum_arithmetic_error = max(maximum_arithmetic_error, probability_error)
        _require(
            probability_error <= arithmetic_tolerance,
            f"candidate score row {row_count} probabilities do not reproduce from deployment state",
        )
        for objective, indices in objectives.items():
            deployment_mean = float(np.mean(deployment_probabilities[list(indices)]))
            observed_mean = _csv_float(
                row[f"mean_{objective}"],
                label=f"candidate score row {row_count} mean_{objective}",
            )
            _require(
                0.0 <= observed_mean <= 1.0,
                f"candidate score row {row_count} objective mean is outside [0, 1]",
            )
            mean_error = abs(observed_mean - deployment_mean)
            fold_objectives = np.mean(reproduced_folds[:, list(indices)], axis=1)
            expected_sensitivity = float(np.std(fold_objectives, ddof=0))
            observed_sensitivity = _csv_float(
                row[f"model_fit_sensitivity_{objective}"],
                label=f"candidate score row {row_count} sensitivity_{objective}",
            )
            _require(
                0.0 <= observed_sensitivity <= 0.5,
                f"candidate score row {row_count} model-fit sensitivity is outside [0, 0.5]",
            )
            sensitivity_error = abs(observed_sensitivity - expected_sensitivity)
            maximum_arithmetic_error = max(maximum_arithmetic_error, mean_error, sensitivity_error)
            _require(
                max(mean_error, sensitivity_error) <= arithmetic_tolerance,
                f"candidate score row {row_count} {objective} arithmetic is invalid",
            )
    _require(row_count == len(candidates), "candidate score CSV does not cover every candidate")
    return maximum_state_error, maximum_arithmetic_error


def _verify_oof_reproduction_document(
    snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    folds: int,
    tolerance: float,
    maximum_error: float,
    by_fold: Sequence[Mapping[str, int | float]],
) -> None:
    document = _exact_mapping(
        _json_object(snapshot.payload, label="oof_reproduction.json"),
        label="oof_reproduction.json",
        expected_keys={
            "schema_version",
            "artifact",
            "status",
            "accepted_model",
            "examples",
            "folds",
            "absolute_tolerance",
            "maximum_absolute_error",
            "by_fold",
        },
    )
    _require(
        _canonical_json(document) == snapshot.payload,
        "oof_reproduction.json is not canonical compact LF-framed JSON",
    )
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "OOF reproduction schema_version must be 1",
    )
    _require(
        document["artifact"] == _OOF_REPRODUCTION_ARTIFACT,
        "OOF reproduction artifact is invalid",
    )
    _require(
        document["status"] == "passed_exact_descriptor_oof_reproduction",
        "OOF reproduction did not pass",
    )
    _require(document["accepted_model"] == _OOF_MODEL, "OOF reproduction model is invalid")
    _require(
        document["examples"] == len(examples) and type(document["examples"]) is int,
        "OOF reproduction example count is invalid",
    )
    _require(
        document["folds"] == folds and type(document["folds"]) is int,
        "OOF reproduction fold count is invalid",
    )
    observed_tolerance = _finite_float(
        document["absolute_tolerance"], label="OOF reproduction absolute_tolerance"
    )
    _require(observed_tolerance == tolerance, "OOF reproduction tolerance differs from config")
    observed_maximum = _finite_float(
        document["maximum_absolute_error"],
        label="OOF reproduction maximum_absolute_error",
    )
    _require(
        abs(observed_maximum - maximum_error) <= tolerance,
        "OOF reproduction maximum error is not independently reproduced",
    )
    observed_by_fold = document["by_fold"]
    _require(
        isinstance(observed_by_fold, list) and len(observed_by_fold) == folds,
        "OOF reproduction by_fold must contain exactly five ordered records",
    )
    for fold in range(folds):
        observed = _exact_mapping(
            cast(list[object], observed_by_fold)[fold],
            label=f"OOF reproduction fold {fold}",
            expected_keys={
                "fold",
                "heldout_examples",
                "training_examples",
                "maximum_absolute_error",
            },
        )
        expected = by_fold[fold]
        _require(
            observed["fold"] == fold
            and type(observed["fold"]) is int
            and observed["heldout_examples"] == expected["heldout_examples"]
            and type(observed["heldout_examples"]) is int
            and observed["training_examples"] == expected["training_examples"]
            and type(observed["training_examples"]) is int,
            f"OOF reproduction fold {fold} census is invalid",
        )
        observed_error = _finite_float(
            observed["maximum_absolute_error"],
            label=f"OOF reproduction fold {fold} maximum_absolute_error",
        )
        _require(
            abs(observed_error - float(expected["maximum_absolute_error"])) <= tolerance,
            f"OOF reproduction fold {fold} maximum error is invalid",
        )


def _artifact_record(
    value: object,
    *,
    label: str,
    snapshot: Snapshot,
    expected_rows: int | None = None,
) -> None:
    expected_keys = {"sha256", "size_bytes"}
    if expected_rows is not None:
        expected_keys.add("rows")
    record = _exact_mapping(value, label=label, expected_keys=expected_keys)
    _require(
        _sha256(record["sha256"], label=f"{label} sha256") == snapshot.sha256,
        f"{label} checksum differs from output bytes",
    )
    _require(
        type(record["size_bytes"]) is int and record["size_bytes"] == len(snapshot.payload),
        f"{label} size differs from output bytes",
    )
    if expected_rows is not None:
        _require(
            type(record["rows"]) is int and record["rows"] == expected_rows,
            f"{label} row count is invalid",
        )


def _verify_manifest(
    snapshot: Snapshot,
    *,
    snapshots: Mapping[str, Snapshot],
    config_snapshot: Snapshot,
    candidate_snapshot: Snapshot,
    examples_snapshot: Snapshot,
    oof_snapshot: Snapshot,
    config: ScoringConfig,
    candidates: Sequence[Candidate],
    fold_tensor: FloatArray,
) -> tuple[str, str, float]:
    document = _exact_mapping(
        _json_object(snapshot.payload, label="manifest.json"),
        label="manifest.json",
        expected_keys={
            "schema_version",
            "artifact",
            "status",
            "automatic_production_eligible",
            "probability_calibration",
            "prediction_scope",
            "training_scope",
            "fit_sensitivity_semantics",
            "target_aggregation",
            "model_release_id",
            "git_commit",
            "inputs",
            "targets",
            "models",
            "aggregations",
            "fold_tensor",
            "candidate_csv",
            "claims",
            "artifacts",
        },
    )
    _require(_canonical_json(document) == snapshot.payload, "manifest.json is not canonical")
    _require(
        document["schema_version"] == 1 and type(document["schema_version"]) is int,
        "manifest schema_version must be 1",
    )
    _require(document["artifact"] == _OUTPUT_ARTIFACT, "manifest artifact is invalid")
    _require(document["status"] == _OUTPUT_STATUS, "manifest status is invalid")
    _require(
        document["automatic_production_eligible"] is False,
        "manifest cannot declare automatic production eligibility",
    )
    scalar_policies = {
        "probability_calibration": config.calibration_scope,
        "prediction_scope": _PREDICTION_SCOPE,
        "training_scope": config.training_scope,
        "fit_sensitivity_semantics": config.fit_sensitivity_semantics,
        "target_aggregation": config.target_aggregation,
    }
    for key, expected in scalar_policies.items():
        _require(document[key] == expected, f"manifest {key} is invalid")
    model_release_id = _sha256(document["model_release_id"], label="model_release_id")
    _require(
        model_release_id == snapshots["model_states.json"].sha256,
        "model_release_id must equal the exact model_states.json SHA-256",
    )
    git_commit = document["git_commit"]
    _require(
        isinstance(git_commit, str) and _GIT_SHA_RE.fullmatch(git_commit) is not None,
        "manifest git_commit must be a full lowercase Git SHA",
    )

    inputs = _exact_mapping(
        document["inputs"],
        label="manifest inputs",
        expected_keys={"config", "candidate_pool", "gate1"},
    )
    config_input = _exact_mapping(
        inputs["config"],
        label="manifest config input",
        expected_keys={"sha256", "size_bytes"},
    )
    _require(
        config_input["sha256"] == config_snapshot.sha256
        and config_input["size_bytes"] == len(config_snapshot.payload)
        and type(config_input["size_bytes"]) is int,
        "manifest config input identity is invalid",
    )
    expected_candidate_input = {
        **dict(config.candidate_pool),
        "candidates": len(candidates),
    }
    _require(
        _exact_json_value(inputs["candidate_pool"], expected_candidate_input),
        "manifest candidate-pool provenance differs from config/input",
    )
    expected_gate1_input = {
        **dict(config.gate1),
        "examples": config.expected_gate1_examples,
    }
    _require(
        _exact_json_value(inputs["gate1"], expected_gate1_input),
        "manifest Gate-1 provenance differs from config/input",
    )
    _require(
        config.candidate_jsonl_sha256 == candidate_snapshot.sha256
        and config.gate1_examples_sha256 == examples_snapshot.sha256
        and config.gate1_oof_sha256 == oof_snapshot.sha256,
        "manifest/config external input hash binding is invalid",
    )

    targets_raw = document["targets"]
    _require(
        isinstance(targets_raw, list) and len(targets_raw) == len(config.target_order),
        "manifest target records are invalid",
    )
    expected_targets = [
        {
            "index": index,
            "name": target,
            "gram": config.gram_by_target[target],
            "probability_column": f"probability_{target}",
        }
        for index, target in enumerate(config.target_order)
    ]
    _require(
        _exact_json_value(targets_raw, expected_targets),
        "manifest target records differ from config",
    )

    models = _exact_mapping(
        document["models"],
        label="manifest models",
        expected_keys={
            "family",
            "accepted_oof_model",
            "fold_members",
            "deployment_member",
            "weights",
            "serialized_states_file",
        },
    )
    expected_fold_members = [f"outer_fold_{fold}" for fold in range(config.folds)]
    _require(
        _exact_json_value(
            models,
            {
                "family": _OOF_MODEL,
                "accepted_oof_model": _OOF_MODEL,
                "fold_members": expected_fold_members,
                "deployment_member": "all_data_deployment",
                "weights": dict(config.model_weights),
                "serialized_states_file": "model_states.json",
            },
        ),
        "manifest model declaration is invalid",
    )
    aggregations = _exact_mapping(
        document["aggregations"],
        label="manifest aggregations",
        expected_keys={
            "objectives",
            "target_reduction",
            "deployment_probability_source",
            "model_fit_sensitivity",
        },
    )
    _require(
        _exact_json_value(
            aggregations,
            {
                "objectives": list(config.objectives),
                "target_reduction": config.target_aggregation,
                "deployment_probability_source": "all_data_deployment",
                "model_fit_sensitivity": config.fit_sensitivity_semantics,
            },
        ),
        "manifest aggregation declaration is invalid",
    )

    tensor = _exact_mapping(
        document["fold_tensor"],
        label="manifest fold_tensor",
        expected_keys={
            "filename",
            "axes",
            "shape",
            "dtype",
            "byte_order",
            "memory_order",
            "candidate_axis_order",
            "fold_axis_member_order",
            "target_axis_order",
            "raw_data_sha256",
        },
    )
    raw_data_sha256 = hashlib.sha256(fold_tensor.tobytes(order="C")).hexdigest()
    _require(
        _exact_json_value(
            tensor,
            {
                "filename": "fold_model_target_probabilities.npy",
                "axes": ["candidate", "outer_fold_complement_model", "target"],
                "shape": list(fold_tensor.shape),
                "dtype": "float64",
                "byte_order": "little_endian",
                "memory_order": "C_contiguous",
                "candidate_axis_order": "candidate_jsonl_source_ordinal_ascending",
                "fold_axis_member_order": expected_fold_members,
                "target_axis_order": list(config.target_order),
                "raw_data_sha256": raw_data_sha256,
            },
        ),
        "manifest fold tensor declaration is invalid",
    )
    candidate_csv = _exact_mapping(
        document["candidate_csv"],
        label="manifest candidate_csv",
        expected_keys={"filename", "columns", "rows"},
    )
    _require(
        _exact_json_value(
            candidate_csv,
            {
                "filename": "candidate_activity_scores.csv",
                "columns": list(_expected_score_columns(config)),
                "rows": len(candidates),
            },
        ),
        "manifest candidate CSV declaration is invalid",
    )
    claims = _exact_mapping(
        document["claims"],
        label="manifest claims",
        expected_keys={
            "forbidden",
            "historical_method_choice_used_fold4",
            "organizer_reference_used_for_model_fit",
        },
    )
    _require(
        _exact_json_value(
            claims,
            {
                "forbidden": list(config.forbidden_claims),
                "historical_method_choice_used_fold4": True,
                "organizer_reference_used_for_model_fit": False,
            },
        ),
        "manifest claim boundary is invalid",
    )
    artifacts = _exact_mapping(
        document["artifacts"],
        label="manifest artifacts",
        expected_keys={
            "candidate_activity_scores.csv",
            "fold_model_target_probabilities.npy",
            "model_states.json",
            "oof_reproduction.json",
        },
    )
    for name in (
        "fold_model_target_probabilities.npy",
        "model_states.json",
        "oof_reproduction.json",
    ):
        _artifact_record(
            artifacts[name],
            label=f"manifest artifact {name}",
            snapshot=snapshots[name],
        )
    _artifact_record(
        artifacts["candidate_activity_scores.csv"],
        label="manifest artifact candidate_activity_scores.csv",
        snapshot=snapshots["candidate_activity_scores.csv"],
        expected_rows=len(candidates),
    )
    return model_release_id, cast(str, git_commit), config.oof_absolute_tolerance


def verify_candidate_activity_scoring(
    *,
    candidate_jsonl_path: str | Path,
    config_path: str | Path,
    examples_path: str | Path,
    oof_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, object]:
    """Verify one candidate-activity bundle from independent input files.

    The four supplied input files are authenticated against the preregistered
    hashes and the output manifest.  Callers may therefore extract them from
    already authenticated upstream publications without granting this verifier
    access to the producer's broader directory trees.
    """

    input_snapshots = {
        "candidate_jsonl": _snapshot(candidate_jsonl_path, label="candidate JSONL"),
        "config": _snapshot(config_path, label="scoring config"),
        "examples": _snapshot(examples_path, label="Gate-1 examples"),
        "oof": _snapshot(oof_path, label="Gate-1 OOF predictions"),
    }
    config = _parse_config(input_snapshots["config"])
    _require(
        input_snapshots["candidate_jsonl"].sha256 == config.candidate_jsonl_sha256,
        "candidate JSONL checksum differs from the preregistration",
    )
    _require(
        input_snapshots["examples"].sha256 == config.gate1_examples_sha256,
        "Gate-1 examples checksum differs from the preregistration",
    )
    _require(
        input_snapshots["oof"].sha256 == config.gate1_oof_sha256,
        "Gate-1 OOF checksum differs from the preregistration",
    )
    candidates = _parse_candidate_jsonl(input_snapshots["candidate_jsonl"])
    _require(
        len(candidates) == config.expected_candidates,
        f"expected {config.expected_candidates} candidates, observed {len(candidates)}",
    )
    examples = _parse_examples(input_snapshots["examples"], folds=config.folds)
    _require(
        len(examples) == config.expected_gate1_examples,
        f"expected {config.expected_gate1_examples} examples, observed {len(examples)}",
    )
    accepted_oof = _parse_oof(
        input_snapshots["oof"],
        examples=examples,
        expected_rows=config.expected_gate1_oof_rows,
    )

    output_path = Path(os.path.abspath(os.fspath(output_dir)))
    _reject_symlink_chain(output_path, label="candidate scoring output directory")
    try:
        output_directory_before = os.lstat(output_path)
    except OSError as error:
        raise ValueError(
            f"cannot inspect candidate scoring output directory: {output_path}"
        ) from error
    _require(
        stat.S_ISDIR(output_directory_before.st_mode)
        and stat.S_IMODE(output_directory_before.st_mode) == 0o555,
        "candidate scoring output directory must be a real immutable 0555 directory",
    )
    output_snapshots = _authenticate_output_tree(output_path)
    for name in (
        "candidate_activity_scores.csv",
        "model_states.json",
        "oof_reproduction.json",
        "manifest.json",
    ):
        for private_prefix in (b"/home/", b"/lustre/", b"file://"):
            _require(
                private_prefix not in output_snapshots[name].payload,
                f"output artifact {name} contains a private or mutable path",
            )
    fold_tensor = _load_fold_tensor(
        output_snapshots["fold_model_target_probabilities.npy"],
        candidate_count=len(candidates),
        folds=config.folds,
        target_count=len(config.target_order),
    )
    model_release_id, git_commit, arithmetic_tolerance = _verify_manifest(
        output_snapshots["manifest.json"],
        snapshots=output_snapshots,
        config_snapshot=input_snapshots["config"],
        candidate_snapshot=input_snapshots["candidate_jsonl"],
        examples_snapshot=input_snapshots["examples"],
        oof_snapshot=input_snapshots["oof"],
        config=config,
        candidates=candidates,
        fold_tensor=fold_tensor,
    )
    states = _parse_model_states(
        output_snapshots["model_states.json"],
        config=config,
        examples=examples,
    )
    maximum_oof_error, oof_by_fold = _reproduce_oof(
        examples=examples,
        accepted_probabilities=accepted_oof,
        states=states,
        tolerance=config.oof_absolute_tolerance,
    )
    _verify_oof_reproduction_document(
        output_snapshots["oof_reproduction.json"],
        examples=examples,
        folds=config.folds,
        tolerance=config.oof_absolute_tolerance,
        maximum_error=maximum_oof_error,
        by_fold=oof_by_fold,
    )
    maximum_tensor_state_error, maximum_score_arithmetic_error = (
        _score_rows_and_reproduce_candidates(
            output_snapshots["candidate_activity_scores.csv"],
            candidates=candidates,
            config=config,
            states=states,
            fold_tensor=fold_tensor,
            model_release_id=model_release_id,
            arithmetic_tolerance=arithmetic_tolerance,
        )
    )
    for name, snapshot in {**input_snapshots, **output_snapshots}.items():
        _assert_unchanged(snapshot, label=name)
    _reject_symlink_chain(output_path, label="candidate scoring output directory")
    try:
        output_directory_after = os.lstat(output_path)
    except OSError as error:
        raise ValueError(
            "candidate scoring output directory disappeared during verification"
        ) from error
    _require(
        _fingerprint(output_directory_after) == _fingerprint(output_directory_before)
        and stat.S_ISDIR(output_directory_after.st_mode)
        and stat.S_IMODE(output_directory_after.st_mode) == 0o555,
        "candidate scoring output directory changed during verification",
    )
    _require(
        {entry.name for entry in os.scandir(output_path)} == set(_OUTPUT_FILES),
        "candidate scoring output inventory changed during verification",
    )
    checks = {
        "candidate_identity_and_order_exact": True,
        "candidate_lineage_summaries_exact": True,
        "config_and_input_hashes_authenticated": True,
        "descriptor_features_reimplemented_independently": True,
        "descriptor_states_refit_independently": True,
        "forbidden_endpoint_and_uncertainty_claims_absent": True,
        "manifest_and_checksum_inventory_exact": True,
        "model_release_bound_to_states": True,
        "oof_descriptor_probabilities_reproduced": True,
        "output_bounds_and_finiteness_valid": True,
        "score_objective_arithmetic_reproduced": True,
        "tensor_axes_and_state_predictions_reproduced": True,
    }
    return {
        "schema_version": 1,
        "artifact": "candidate_activity_scoring_development_v1_independent_verification",
        "status": "passed",
        "automatic_production_eligible": False,
        "checks": checks,
        "git_commit": git_commit,
        "model_release_id": model_release_id,
        "input_sha256": {
            name: snapshot.sha256 for name, snapshot in sorted(input_snapshots.items())
        },
        "output_sha256": {
            name: snapshot.sha256 for name, snapshot in sorted(output_snapshots.items())
        },
        "census": {
            "candidates": len(candidates),
            "examples": len(examples),
            "folds": config.folds,
            "targets": len(config.target_order),
            "fold_tensor_shape": list(fold_tensor.shape),
        },
        "maximum_absolute_error": {
            "accepted_descriptor_oof": maximum_oof_error,
            "candidate_fold_tensor_from_states": maximum_tensor_state_error,
            "candidate_score_arithmetic": maximum_score_arithmetic_error,
        },
        "claims": {
            "development_candidate_scoring_only": True,
            "model_fit_sensitivity_is_calibrated_uncertainty": False,
            "production_ensemble": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = verify_candidate_activity_scoring(
            candidate_jsonl_path=args.candidate_jsonl,
            config_path=args.config,
            examples_path=args.examples,
            oof_path=args.oof,
            output_dir=args.output_dir,
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(f"candidate activity scoring verification error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the cluster CLI
    raise SystemExit(main())

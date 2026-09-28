"""Sealed cohort evaluation and decision bundle for native diffusion v0.

This module is the producer-side publication boundary for the preregistered
experiment.  One bundle covers the ordered seed cohort ``[42, 43, 44]``.  It
binds the three primary training bundles and the independently trained seed-42
twin, stores the shared 50,944-case validation panel and per-selected-token
sufficient statistics in deterministic, pickle-free NPZ archives, records all
raw proposals for the diffusion model and two count controls, evaluates every
frozen scientific gate, and publishes an immutable directory with the manifest
as the final commit artifact.

The producer can mark directly observable evidence failures.  Operational
node/job checks and the independent third-node audit may subsequently downgrade
any producer decision to ``invalid_run``; they can never promote it.  A passing
producer decision is deliberately limited to candidate-generator status.

The public publisher always reloads the byte-pinned production TOML.  The
smaller immutable protocol and encoding functions exist so serialization and
gate behavior can be unit-tested without running a model or allocating the
production-size arrays.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from numpy.typing import NDArray

from amp_challenge.sequences import canonical_sequence_id

from .categorical import AbsorbingDiffusion, CosineMaskSchedule, PeptideVocabulary
from .contract import CONFIG_SHA256, NativeDiffusionContract, load_unconditional_v0_contract
from .data import (
    DiffusionCorpusRow,
    NativeDiffusionCorpus,
    TrainingDistribution,
    load_native_diffusion_corpus,
    load_training_projection,
    namespaced_seed,
)
from .evaluation import (
    CALIBRATION_BINS,
    CandidateDiagnostics,
    CaseMetrics,
    CountBaselineSuite,
    DescriptorEnergyDistanceResult,
    EvaluationResult,
    GeneratorControlSamplingResult,
    NgramDistanceResult,
    ValidationCaseLedger,
    aggregate_case_metrics,
    build_distribution_candidate_pool,
    candidate_diagnostics,
    descriptor_energy_distance,
    fit_count_baselines,
    sample_count_generator_control_v0,
    sampling_ngram_jensen_shannon,
)
from .sampling import SamplingResult, canonical_length_plan

EVALUATION_BUNDLE_FILES = (
    "contract.toml",
    "training_bundle.sha256",
    "validation_corruptions.npz",
    "validation_token_stats.npz",
    "validation_metrics.json",
    "baseline_metrics.json",
    "length_plan.json",
    "raw_proposals.fasta",
    "candidate_ledger.jsonl",
    "sampling_metrics.json",
    "manifest.json",
)
EVALUATION_MANIFEST_FIELDS = (
    "schema_version",
    "artifact",
    "config_sha256",
    "git_commit",
    "seeds",
    "training_bundle_sha256",
    "validation",
    "baselines",
    "sampling",
    "gates",
    "decision_status",
    "artifacts",
)
TRAINING_BUNDLE_LABELS = (
    "seed-42-primary",
    "seed-42-twin",
    "seed-43",
    "seed-44",
)
PROPOSAL_METHOD_ORDER = (
    "native_categorical_diffusion",
    "component_weighted_unigram",
    "component_weighted_forward_markov",
)
TOKEN_STATS_METHOD_ORDER = (
    "component_weighted_unigram",
    "component_weighted_bidirectional_markov",
    "length_relative_position_frequency",
    "native_seed_42",
    "native_seed_43",
    "native_seed_44",
)
PRODUCER_CHECK_NAMES = (
    "accepted_input_hashes",
    "clean_synchronized_git",
    "training_bundle_integrity",
    "runtime_environment",
    "deterministic_execution",
    "validation_execution",
    "sampling_execution",
    "finite_values",
)
NPZ_ARCHIVE_FORMAT = "zip_stored_npy_v1_0_dos_epoch_1980_member_order_v1"
CANDIDATE_LEDGER_FIELDS = (
    "schema_version",
    "method",
    "seed",
    "ordinal",
    "sequence_id",
    "sequence",
    "length",
    "generator_binding_kind",
    "generator_binding_sha256",
    "config_sha256",
    "training_projection_sha256",
    "length_plan_sha256",
)
FASTA_HEADER_GRAMMAR = ">{method}|seed={seed}|ordinal={ordinal}|sequence_id={sequence_id}"
BASELINE_METRICS_FIELDS = (
    "schema_version",
    "config_sha256",
    "method_order",
    "methods",
    "strongest_method",
)
VALIDATION_METRICS_FIELDS = (
    "schema_version",
    "config_sha256",
    "seeds",
    "method_order",
    "models",
    "bootstrap",
    "timestep_bins",
)
LENGTH_PLAN_FIELDS = (
    "schema_version",
    "config_sha256",
    "distribution",
    "shared_across_methods",
    "seeds",
)
SAMPLING_METRICS_FIELDS = (
    "schema_version",
    "config_sha256",
    "seeds",
    "method_order",
    "methods",
    "aggregates",
    "gates",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_LABEL_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_NPZ_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_UINT64_LIMIT = 1 << 64
_DOS_EPOCH = (1980, 1, 1, 0, 0, 0)
_BUNDLE_DOMAIN = b"amp-native-diffusion-evaluation-bundle-v1\0"
_BOOTSTRAP_DOMAIN = b"amp-native-diffusion-cohort-bootstrap-draws-v1\0"
_CHECKPOINT_BINDING_DOMAIN = (
    b"amp-challenge/native-categorical-diffusion/checkpoint-contract-binding/v1\0"
)
_PAYLOAD_CONSTRUCTION_TOKEN = object()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_git_commit(value: object) -> str:
    if type(value) is not str or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise ValueError("git_commit must be a lowercase forty-character Git object ID")
    return value


def _finite(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite canonical Python float")
    return value


def _bounded(value: object, *, label: str, lower: float, upper: float) -> float:
    result = _finite(value, label=label)
    if not lower <= result <= upper:
        raise ValueError(f"{label} must lie in [{lower}, {upper}]")
    return result


def _uint64(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value < _UINT64_LIMIT:
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value


def _reject_path_values(value: object, *, label: str = "semantic document") -> None:
    """Reject path objects and path-bearing strings from semantic artifacts."""

    if isinstance(value, os.PathLike):
        raise ValueError(f"{label} contains a path object")
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return
    if type(value) is str:
        if "\x00" in value or "\\" in value or "/" in value:
            raise ValueError(f"{label} contains a path-bearing string")
        return
    if type(value) in {tuple, list}:
        for item in value:
            _reject_path_values(item, label=label)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be strings")
            _reject_path_values(key, label=label)
            _reject_path_values(item, label=label)
        return
    raise TypeError(f"{label} contains a non-canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode one finite, path-free canonical JSON document with a final LF."""

    _reject_path_values(value)
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


def _strict_json_loads(payload: bytes, *, label: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-terminated bytes")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite number {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from error
    if type(parsed) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    if canonical_json_bytes(parsed) != payload:
        raise ValueError(f"{label} is not in canonical JSON byte form")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _framed_update(digest, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@dataclass(frozen=True, slots=True)
class NpzArraySpec:
    """One exact member in a deterministic NPZ archive."""

    name: str
    dtype: str
    dimensions: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NPZ_NAME_RE.fullmatch(self.name) is None:
            raise ValueError("NPZ member name must be a lowercase identifier")
        if type(self.dtype) is not str or self.dtype not in {
            "S64",
            "u1",
            "b1",
            "<u2",
            "<u8",
            "<f8",
        }:
            raise ValueError("NPZ member dtype is not in the frozen portable set")
        if (
            type(self.dimensions) is not tuple
            or not self.dimensions
            or any(
                (type(item) is int and item <= 0)
                or (type(item) is str and item != "N_selected")
                or type(item) not in {int, str}
                for item in self.dimensions
            )
        ):
            raise ValueError("NPZ dimensions must be positive integers or N_selected")

    @property
    def contract_record(self) -> str:
        return "|".join(
            (
                self.name,
                self.dtype,
                ",".join(str(value) for value in self.dimensions),
            )
        )

    @property
    def numpy_dtype(self) -> np.dtype:
        aliases = {"S64": "|S64", "u1": "|u1", "b1": "|b1"}
        return np.dtype(aliases.get(self.dtype, self.dtype))


def parse_npz_schema(values: Sequence[str]) -> tuple[NpzArraySpec, ...]:
    """Parse the compact, ordered schema syntax pinned in the TOML contract."""

    if type(values) is not tuple or not values:
        raise TypeError("NPZ schema must be a non-empty tuple of strings")
    result: list[NpzArraySpec] = []
    for raw in values:
        if type(raw) is not str:
            raise TypeError("NPZ schema entries must be strings")
        parts = raw.split("|")
        if len(parts) != 3:
            raise ValueError(f"invalid NPZ schema entry: {raw!r}")
        dimensions: list[int | str] = []
        for item in parts[2].split(","):
            if item == "N_selected":
                dimensions.append(item)
            elif item.isascii() and item.isdigit() and not item.startswith("0"):
                dimensions.append(int(item))
            else:
                raise ValueError(f"invalid NPZ dimension in {raw!r}")
        result.append(NpzArraySpec(parts[0], parts[1], tuple(dimensions)))
    names = tuple(item.name for item in result)
    if len(names) != len(set(names)):
        raise ValueError("NPZ schema member names must be unique")
    if tuple(item.contract_record for item in result) != tuple(values):
        raise ValueError("NPZ schema is not in canonical compact form")
    return tuple(result)


def deterministic_npz_bytes(
    schema: Sequence[NpzArraySpec],
    arrays: Mapping[str, NDArray[np.generic]],
) -> bytes:
    """Write fixed-order ZIP_STORED NPY-v1.0 members with DOS-epoch metadata.

    Object arrays and implicit dtype/shape coercions are rejected, so NumPy can
    never invoke pickle.  ZIP member permissions, timestamps, creator, flags,
    compression, comments, and ordering are all explicit.
    """

    entries = tuple(schema)
    if not entries or any(type(item) is not NpzArraySpec for item in entries):
        raise TypeError("schema must contain NpzArraySpec values")
    if type(arrays) is not dict:
        raise TypeError("arrays must be a plain dict")
    expected_names = tuple(item.name for item in entries)
    if tuple(arrays) != expected_names:
        raise ValueError("NPZ arrays must follow the exact schema name and member order")
    symbolic_sizes: dict[str, int] = {}
    prepared: list[tuple[NpzArraySpec, NDArray[np.generic]]] = []
    for item in entries:
        raw = arrays[item.name]
        if type(raw) is not np.ndarray:
            raise TypeError(f"NPZ member {item.name} must be an ndarray")
        if raw.dtype != item.numpy_dtype:
            raise TypeError(
                f"NPZ member {item.name} dtype must be {item.numpy_dtype.str}, got {raw.dtype.str}"
            )
        if raw.ndim != len(item.dimensions):
            raise ValueError(f"NPZ member {item.name} has the wrong rank")
        for observed, expected in zip(raw.shape, item.dimensions, strict=True):
            if type(expected) is int and observed != expected:
                raise ValueError(f"NPZ member {item.name} has the wrong shape")
            if type(expected) is str:
                prior = symbolic_sizes.setdefault(expected, observed)
                if observed != prior:
                    raise ValueError(f"NPZ symbolic dimension {expected} is inconsistent")
        if raw.dtype.hasobject:
            raise TypeError("object arrays are forbidden in evaluation NPZ artifacts")
        if raw.dtype.kind == "f" and np.any(~np.isfinite(raw)):
            raise ValueError(f"NPZ member {item.name} contains a non-finite value")
        values = np.ascontiguousarray(raw)
        values.flags.writeable = False
        prepared.append((item, values))

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for item, values in prepared:
            member = io.BytesIO()
            np.lib.format.write_array(
                member,
                values,
                version=(1, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{item.name}.npy", date_time=_DOS_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.internal_attr = 0
            info.flag_bits = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Resolved immutable values used by the bundle core."""

    schema_version: int
    artifact: str
    config_sha256: str
    corpus_sha256: str
    training_projection_sha256: str
    organizer_reference_sha256: str
    organizer_reference_records: int
    training_sequences: int
    seeds: tuple[int, ...]
    validation_sequences: int
    corruption_cases: int
    levels: int
    replicates: int
    width: int
    bootstrap_replicates: int
    bootstrap_seed: int
    evaluation_seed: int
    evaluation_batch_sequences: int
    raw_proposals_per_seed: int
    timestep_bins: tuple[str, ...]
    baseline_methods: tuple[str, ...]
    proposal_methods: tuple[str, ...]
    token_stats_methods: tuple[str, ...]
    training_bundle_labels: tuple[str, ...]
    corruption_schema: tuple[NpzArraySpec, ...]
    token_stats_schema: tuple[NpzArraySpec, ...]
    training_bundle_files: tuple[str, ...]
    training_manifest_fields: tuple[str, ...]
    bundle_files: tuple[str, ...]
    manifest_fields: tuple[str, ...]
    evidence_invalid_status: str
    no_go_status: str
    candidate_status: str
    minimum_mean_relative_nll_improvement: float
    minimum_bootstrap_lower_bound_improvement: float
    minimum_high_noise_relative_nll_improvement: float
    maximum_timestep_bin_relative_nll_regression: float
    maximum_ece: float
    maximum_ece_regression: float
    minimum_canonical_valid_fraction_each_seed: float
    minimum_raw_unique_fraction_each_seed: float
    maximum_exact_train_overlap_fraction_each_seed: float
    minimum_common_funnel_yield_fraction_each_seed: float
    minimum_top_reference_safe_count_each_seed: int
    minimum_hill2_effective_70pct_clusters_each_seed: float
    maximum_largest_70pct_cluster_fraction_each_seed: float
    maximum_common_funnel_yield_deficit_vs_best_control: float
    maximum_ngram_jsd_regression_bits_vs_best_control: float
    maximum_descriptor_energy_distance_ratio_vs_best_control: float
    minimum_improvement_in_3mer_jsd_or_descriptor_distance: float

    def __post_init__(self) -> None:
        for label, value in (
            ("schema_version", self.schema_version),
            ("validation_sequences", self.validation_sequences),
            ("corruption_cases", self.corruption_cases),
            ("levels", self.levels),
            ("replicates", self.replicates),
            ("width", self.width),
            ("bootstrap_replicates", self.bootstrap_replicates),
            ("evaluation_batch_sequences", self.evaluation_batch_sequences),
            ("organizer_reference_records", self.organizer_reference_records),
            ("training_sequences", self.training_sequences),
            ("raw_proposals_per_seed", self.raw_proposals_per_seed),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        _require_sha256(self.config_sha256, label="config_sha256")
        _require_sha256(self.corpus_sha256, label="corpus_sha256")
        _require_sha256(
            self.training_projection_sha256,
            label="training_projection_sha256",
        )
        _require_sha256(
            self.organizer_reference_sha256,
            label="organizer_reference_sha256",
        )
        _uint64(self.bootstrap_seed, label="bootstrap_seed")
        _uint64(self.evaluation_seed, label="evaluation_seed")
        if type(self.artifact) is not str or not self.artifact:
            raise ValueError("artifact must be a non-empty string")
        if (
            type(self.seeds) is not tuple
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be a non-empty unique tuple")
        for seed in self.seeds:
            _uint64(seed, label="seed")
        tuple_fields = (
            self.timestep_bins,
            self.baseline_methods,
            self.proposal_methods,
            self.token_stats_methods,
            self.training_bundle_labels,
            self.corruption_schema,
            self.token_stats_schema,
            self.training_bundle_files,
            self.training_manifest_fields,
            self.bundle_files,
            self.manifest_fields,
        )
        if any(type(value) is not tuple or not value for value in tuple_fields):
            raise ValueError("protocol tuple fields must be non-empty tuples")
        if len(set(self.baseline_methods)) != len(self.baseline_methods):
            raise ValueError("baseline methods must be unique")
        if len(set(self.proposal_methods)) != len(self.proposal_methods):
            raise ValueError("proposal methods must be unique")
        if len(self.token_stats_methods) != len(self.baseline_methods) + len(self.seeds):
            raise ValueError(
                "token-stat method axis must contain baselines then one model per seed"
            )
        if self.corruption_cases != self.validation_sequences * self.levels * self.replicates:
            raise ValueError("corruption-case census is inconsistent")
        if self.bundle_files[-1] != "manifest.json":
            raise ValueError("manifest.json must be the final bundle artifact")
        for label in self.training_bundle_labels:
            if _LABEL_RE.fullmatch(label) is None:
                raise ValueError("training bundle labels must be symbolic and path-free")
        for label in (
            self.evidence_invalid_status,
            self.no_go_status,
            self.candidate_status,
        ):
            if type(label) is not str or not label:
                raise ValueError("decision statuses must be non-empty strings")
        for field_name in (
            "minimum_mean_relative_nll_improvement",
            "minimum_bootstrap_lower_bound_improvement",
            "minimum_high_noise_relative_nll_improvement",
            "maximum_timestep_bin_relative_nll_regression",
            "maximum_ece",
            "maximum_ece_regression",
            "minimum_canonical_valid_fraction_each_seed",
            "minimum_raw_unique_fraction_each_seed",
            "maximum_exact_train_overlap_fraction_each_seed",
            "minimum_common_funnel_yield_fraction_each_seed",
            "minimum_hill2_effective_70pct_clusters_each_seed",
            "maximum_largest_70pct_cluster_fraction_each_seed",
            "maximum_common_funnel_yield_deficit_vs_best_control",
            "maximum_ngram_jsd_regression_bits_vs_best_control",
            "maximum_descriptor_energy_distance_ratio_vs_best_control",
            "minimum_improvement_in_3mer_jsd_or_descriptor_distance",
        ):
            _finite(getattr(self, field_name), label=field_name)
        if (
            type(self.minimum_top_reference_safe_count_each_seed) is not int
            or self.minimum_top_reference_safe_count_each_seed < 0
        ):
            raise ValueError("minimum_top_reference_safe_count_each_seed is invalid")


def protocol_from_contract(contract: NativeDiffusionContract) -> EvaluationProtocol:
    """Resolve and cross-check every bundle-facing field from the strict TOML."""

    if not isinstance(contract, NativeDiffusionContract):
        raise TypeError("contract must be a NativeDiffusionContract")
    artifacts = contract.artifacts
    denoising = contract.denoising_gates
    sampling = contract.sampling_gates
    expected_literals = {
        "evaluation_scope": (artifacts.evaluation_scope, "cohort_seed_order_42_43_44"),
        "npz_archive_format": (artifacts.npz_archive_format, NPZ_ARCHIVE_FORMAT),
        "bootstrap comparator": (
            denoising.bootstrap_lower_bound_comparator,
            "strictly_greater_than",
        ),
        "bootstrap aggregation": (
            denoising.bootstrap_seed_aggregation,
            "arithmetic_mean_relative_improvement",
        ),
        "high-noise aggregation": (
            denoising.high_noise_seed_aggregation,
            "arithmetic_mean_nll",
        ),
        "timestep-bin aggregation": (
            denoising.timestep_bin_seed_aggregation,
            "arithmetic_mean_nll",
        ),
        "ECE seed rule": (denoising.ece_seed_rule, "every_seed"),
        "relative control aggregation": (
            sampling.relative_control_seed_aggregation,
            "median",
        ),
    }
    for label, (observed, expected) in expected_literals.items():
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"{label} differs from the v0 evaluation protocol")
    exact_tuples = {
        "seeds": (artifacts.evaluation_seed_order, contract.training.seeds),
        "training bundle labels": (
            artifacts.training_bundle_binding_labels,
            TRAINING_BUNDLE_LABELS,
        ),
        "proposal methods": (artifacts.proposal_method_order, PROPOSAL_METHOD_ORDER),
        "token-stat methods": (
            artifacts.validation_token_stats_method_order,
            TOKEN_STATS_METHOD_ORDER,
        ),
        "bundle inventory": (artifacts.evaluation_bundle_files, EVALUATION_BUNDLE_FILES),
        "manifest fields": (
            artifacts.evaluation_manifest_fields,
            EVALUATION_MANIFEST_FIELDS,
        ),
    }
    for label, (observed, expected) in exact_tuples.items():
        if observed != expected:
            raise ValueError(f"{label} differs from the frozen evaluation protocol")
    if tuple(contract.baselines.names) != TOKEN_STATS_METHOD_ORDER[:3]:
        raise ValueError("baseline method order differs from the token-stat axis")
    if tuple(contract.baselines.generator_controls) != PROPOSAL_METHOD_ORDER[1:]:
        raise ValueError("generator controls differ from the proposal method axis")
    if not denoising.require_every_seed_beats_strongest_baseline:
        raise ValueError("v0 must require every seed to beat the strongest baseline")
    if denoising.high_noise_timestep_bin != "49_to_64":
        raise ValueError("v0 high-noise timestep bin differs from 49_to_64")
    return EvaluationProtocol(
        schema_version=artifacts.schema_version,
        artifact=contract.artifact,
        config_sha256=contract.config_sha256,
        corpus_sha256=contract.input.corpus_sha256,
        training_projection_sha256=contract.input.training_projection_sha256,
        organizer_reference_sha256=contract.input.organizer_reference_sha256,
        organizer_reference_records=contract.input.organizer_reference_records,
        training_sequences=contract.input.expected_train_sequences,
        seeds=artifacts.evaluation_seed_order,
        validation_sequences=contract.evaluation.expected_validation_sequences,
        corruption_cases=contract.evaluation.expected_corruption_cases,
        levels=contract.evaluation.levels,
        replicates=contract.evaluation.replicates_per_sequence_level,
        width=contract.model.max_length,
        bootstrap_replicates=contract.evaluation.bootstrap_replicates,
        bootstrap_seed=contract.evaluation.bootstrap_seed,
        evaluation_seed=contract.evaluation.evaluation_seed,
        evaluation_batch_sequences=contract.evaluation.batch_sequences,
        raw_proposals_per_seed=contract.sampling.raw_proposals_per_seed,
        timestep_bins=contract.evaluation.timestep_bins,
        baseline_methods=contract.baselines.names,
        proposal_methods=artifacts.proposal_method_order,
        token_stats_methods=artifacts.validation_token_stats_method_order,
        training_bundle_labels=artifacts.training_bundle_binding_labels,
        corruption_schema=parse_npz_schema(artifacts.validation_corruptions_npz_schema),
        token_stats_schema=parse_npz_schema(artifacts.validation_token_stats_npz_schema),
        training_bundle_files=artifacts.training_bundle_files,
        training_manifest_fields=artifacts.training_manifest_fields,
        bundle_files=artifacts.evaluation_bundle_files,
        manifest_fields=artifacts.evaluation_manifest_fields,
        evidence_invalid_status=contract.status.evidence_invalid,
        no_go_status=contract.status.reproducible_no_go,
        candidate_status=contract.status.candidate_generator_only,
        minimum_mean_relative_nll_improvement=(denoising.minimum_mean_relative_nll_improvement),
        minimum_bootstrap_lower_bound_improvement=(
            denoising.minimum_bootstrap_lower_bound_improvement
        ),
        minimum_high_noise_relative_nll_improvement=(
            denoising.minimum_high_noise_relative_nll_improvement
        ),
        maximum_timestep_bin_relative_nll_regression=(
            denoising.maximum_timestep_bin_relative_nll_regression
        ),
        maximum_ece=denoising.maximum_ece,
        maximum_ece_regression=denoising.maximum_ece_regression,
        minimum_canonical_valid_fraction_each_seed=(
            sampling.minimum_canonical_valid_fraction_each_seed
        ),
        minimum_raw_unique_fraction_each_seed=(sampling.minimum_raw_unique_fraction_each_seed),
        maximum_exact_train_overlap_fraction_each_seed=(
            sampling.maximum_exact_train_overlap_fraction_each_seed
        ),
        minimum_common_funnel_yield_fraction_each_seed=(
            sampling.minimum_common_funnel_yield_fraction_each_seed
        ),
        minimum_top_reference_safe_count_each_seed=(
            sampling.minimum_top_reference_safe_count_each_seed
        ),
        minimum_hill2_effective_70pct_clusters_each_seed=(
            sampling.minimum_hill2_effective_70pct_clusters_each_seed
        ),
        maximum_largest_70pct_cluster_fraction_each_seed=(
            sampling.maximum_largest_70pct_cluster_fraction_each_seed
        ),
        maximum_common_funnel_yield_deficit_vs_best_control=(
            sampling.maximum_common_funnel_yield_deficit_vs_best_control
        ),
        maximum_ngram_jsd_regression_bits_vs_best_control=(
            sampling.maximum_ngram_jsd_regression_bits_vs_best_control
        ),
        maximum_descriptor_energy_distance_ratio_vs_best_control=(
            sampling.maximum_descriptor_energy_distance_ratio_vs_best_control
        ),
        minimum_improvement_in_3mer_jsd_or_descriptor_distance=(
            sampling.minimum_improvement_in_3mer_jsd_or_descriptor_distance
        ),
    )


def _bound_checkpoint_sha256(config_sha256: str, model_logical_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(_CHECKPOINT_BINDING_DOMAIN)
    _framed_update(digest, bytes.fromhex(_require_sha256(config_sha256, label="config_sha256")))
    _framed_update(
        digest,
        bytes.fromhex(_require_sha256(model_logical_sha256, label="model logical SHA-256")),
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingManifestRecord:
    """One path-free identity extracted from a verified training bundle."""

    label: str
    seed: int
    manifest_sha256: str
    checkpoint_file_sha256: str
    checkpoint_model_logical_sha256: str
    checkpoint_bound_logical_sha256: str

    def __post_init__(self) -> None:
        if type(self.label) is not str or _LABEL_RE.fullmatch(self.label) is None:
            raise ValueError("training label must be a symbolic path-free identifier")
        _uint64(self.seed, label="training seed")
        for field_name in (
            "manifest_sha256",
            "checkpoint_file_sha256",
            "checkpoint_model_logical_sha256",
            "checkpoint_bound_logical_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)


@dataclass(frozen=True, slots=True)
class TrainingBundleBinding:
    """Canonical four-manifest checksum binding used by the evaluation bundle."""

    records: tuple[TrainingManifestRecord, ...]
    payload: bytes
    sha256: str
    git_commit: str
    config_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.records) is not tuple
            or not self.records
            or any(type(item) is not TrainingManifestRecord for item in self.records)
        ):
            raise ValueError("training binding records must be a non-empty exact tuple")
        if type(self.payload) is not bytes or not self.payload.endswith(b"\n"):
            raise ValueError("training binding payload must be LF-terminated bytes")
        _require_sha256(self.sha256, label="training binding SHA-256")
        _require_git_commit(self.git_commit)
        _require_sha256(self.config_sha256, label="training binding config SHA-256")
        if tuple(item.label for item in self.records) != TRAINING_BUNDLE_LABELS or tuple(
            item.seed for item in self.records
        ) != (42, 42, 43, 44):
            raise ValueError("training binding records differ from the frozen cohort order")
        if any(
            item.checkpoint_bound_logical_sha256
            != _bound_checkpoint_sha256(
                self.config_sha256,
                item.checkpoint_model_logical_sha256,
            )
            for item in self.records
        ):
            raise ValueError("training binding checkpoint logical identity is inconsistent")
        expected = b"".join(
            f"{item.manifest_sha256}  {item.label}\n".encode("ascii") for item in self.records
        )
        if self.payload != expected or hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("training binding payload or digest is inconsistent")
        if self.records[0].manifest_sha256 != self.records[1].manifest_sha256:
            raise ValueError("seed-42 primary and twin training manifests must be byte-identical")

    @property
    def primary_by_seed(self) -> dict[int, TrainingManifestRecord]:
        result: dict[int, TrainingManifestRecord] = {}
        for item in self.records:
            if item.label == "seed-42-twin":
                continue
            if item.seed in result:
                raise RuntimeError("training binding contains duplicate primary seed")
            result[item.seed] = item
        return result


def _training_record_from_manifest(
    *,
    label: str,
    payload: bytes,
    protocol: EvaluationProtocol,
    expected_git_commit: str,
) -> TrainingManifestRecord:
    document = _strict_json_loads(payload, label=f"training manifest {label}")
    if set(document) != set(protocol.training_manifest_fields):
        raise ValueError(f"training manifest {label} has the wrong top-level schema")
    expected_seed = 42 if label.startswith("seed-42-") else int(label.removeprefix("seed-"))
    exact = {
        "schema_version": protocol.schema_version,
        "artifact": protocol.artifact,
        "config_sha256": protocol.config_sha256,
        "git_commit": expected_git_commit,
        "seed": expected_seed,
    }
    for key, expected in exact.items():
        observed = document.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"training manifest {label} field {key} differs from the contract")
    corpus = document.get("corpus")
    corpus_fields = {
        "accepted_parent_sha256",
        "training_projection_sha256",
        "trainer_visible_sequences",
        "trainer_visible_fields",
        "roles",
    }
    if type(corpus) is not dict or set(corpus) != corpus_fields:
        raise ValueError(f"training manifest {label} corpus schema differs from the contract")
    if corpus.get("accepted_parent_sha256") != protocol.corpus_sha256:
        raise ValueError(f"training manifest {label} does not bind the accepted corpus")
    if corpus.get("training_projection_sha256") != protocol.training_projection_sha256:
        raise ValueError(f"training manifest {label} does not bind the training projection")
    if (
        corpus.get("trainer_visible_sequences") != protocol.training_sequences
        or corpus.get("trainer_visible_fields") != ["sequence_id", "sequence", "sampling_weight"]
        or corpus.get("roles") != ["train"]
    ):
        raise ValueError(f"training manifest {label} trainer boundary differs from the contract")
    model = document.get("model")
    model_fields = {
        "config",
        "trainable_parameters",
        "checkpoint_file_sha256",
        "checkpoint_logical_state_sha256",
        "checkpoint_format",
    }
    if type(model) is not dict or set(model) != model_fields:
        raise ValueError(f"training manifest {label} model schema differs from the contract")
    if (
        type(model.get("config")) is not dict
        or type(model.get("trainable_parameters")) is not int
        or model.get("trainable_parameters", 0) <= 0
        or model.get("checkpoint_format") != "safetensors"
    ):
        raise ValueError(f"training manifest {label} does not declare a safetensors checkpoint")
    checkpoint_file = _require_sha256(
        model.get("checkpoint_file_sha256"),
        label=f"training manifest {label} checkpoint file",
    )
    checkpoint_model = _require_sha256(
        model.get("checkpoint_logical_state_sha256"),
        label=f"training manifest {label} checkpoint logical state",
    )
    artifacts = document.get("artifacts")
    expected_artifacts = set(protocol.training_bundle_files) - {"manifest.json"}
    if type(artifacts) is not dict or set(artifacts) != expected_artifacts:
        raise ValueError(f"training manifest {label} artifact inventory differs from the contract")
    for name, digest in artifacts.items():
        _require_sha256(digest, label=f"training manifest {label} artifact {name}")
    if artifacts.get("model_final.safetensors") != checkpoint_file:
        raise ValueError(f"training manifest {label} checkpoint file digest is inconsistent")
    rng = document.get("rng")
    if (
        type(rng) is not dict
        or set(rng) != {"filename", "sha256"}
        or rng.get("filename") != "rng.json"
        or rng.get("sha256") != artifacts.get("rng.json")
    ):
        raise ValueError(f"training manifest {label} RNG binding is inconsistent")
    training = document.get("training")
    training_fields = {
        "steps",
        "batch_sequences",
        "optimizer",
        "parameter_decay_names",
        "parameter_no_decay_names",
        "schedule_sha256",
        "checkpoint_selection",
        "validation_during_training",
        "early_stopping",
        "resume_supported",
        "metrics_sha256",
    }
    if type(training) is not dict or set(training) != training_fields:
        raise ValueError(f"training manifest {label} training schema differs from the contract")
    if (
        type(training.get("steps")) is not int
        or training.get("steps", 0) <= 0
        or type(training.get("batch_sequences")) is not int
        or training.get("batch_sequences", 0) <= 0
        or training.get("optimizer") != "adamw_unfused"
        or type(training.get("parameter_decay_names")) is not list
        or type(training.get("parameter_no_decay_names")) is not list
        or not training.get("parameter_decay_names")
        or not training.get("parameter_no_decay_names")
        or training.get("validation_during_training") is not False
        or training.get("early_stopping") is not False
        or training.get("resume_supported") is not False
        or training.get("metrics_sha256") != artifacts.get("train_metrics.json")
    ):
        raise ValueError(f"training manifest {label} training protocol is inconsistent")
    _require_sha256(
        training.get("schedule_sha256"),
        label=f"training manifest {label} schedule SHA-256",
    )
    if type(training.get("checkpoint_selection")) is not str:
        raise ValueError(f"training manifest {label} checkpoint selection is invalid")
    return TrainingManifestRecord(
        label=label,
        seed=expected_seed,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_file_sha256=checkpoint_file,
        checkpoint_model_logical_sha256=checkpoint_model,
        checkpoint_bound_logical_sha256=_bound_checkpoint_sha256(
            protocol.config_sha256,
            checkpoint_model,
        ),
    )


def training_bundle_binding_from_manifests(
    manifests: Mapping[str, bytes],
    *,
    protocol: EvaluationProtocol,
    git_commit: str,
) -> TrainingBundleBinding:
    """Build the exact four-row binding from canonical manifest bytes."""

    if type(manifests) is not dict:
        raise TypeError("manifests must be a plain dict")
    if tuple(manifests) != protocol.training_bundle_labels:
        raise ValueError("training manifests must follow the exact symbolic label order")
    commit = _require_git_commit(git_commit)
    records = tuple(
        _training_record_from_manifest(
            label=label,
            payload=manifests[label],
            protocol=protocol,
            expected_git_commit=commit,
        )
        for label in protocol.training_bundle_labels
    )
    if tuple(item.seed for item in records) != (42, 42, 43, 44):
        raise ValueError("training manifests do not represent the frozen seed/twin cohort")
    if set(item.seed for item in records) != set(protocol.seeds):
        raise ValueError("training manifests do not cover the protocol seed cohort")
    payload = b"".join(
        f"{item.manifest_sha256}  {item.label}\n".encode("ascii") for item in records
    )
    return TrainingBundleBinding(
        records=records,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        git_commit=commit,
        config_sha256=protocol.config_sha256,
    )


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"path ancestor does not exist: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"path traverses a symbolic link: {current}")
        if current.parent == current:
            return
        current = current.parent


def _read_regular_bytes(path: Path) -> bytes:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(source)
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = source.stat(follow_symlinks=False)
    fingerprints = {
        (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            stat.S_IMODE(item.st_mode),
        )
        for item in (before, opened_before, opened_after, named_after)
    }
    payload = b"".join(chunks)
    if len(fingerprints) != 1 or len(payload) != before.st_size:
        raise ValueError(f"input changed while it was read: {source}")
    return payload


def load_training_bundle_binding(
    bundle_directories: Mapping[str, str | Path],
    *,
    protocol: EvaluationProtocol,
    contract_payload: bytes,
    git_commit: str,
) -> TrainingBundleBinding:
    """Verify four immutable training inventories and return their path-free binding."""

    if type(bundle_directories) is not dict:
        raise TypeError("bundle_directories must be a plain dict")
    if tuple(bundle_directories) != protocol.training_bundle_labels:
        raise ValueError("training bundle paths must follow the exact symbolic label order")
    if hashlib.sha256(contract_payload).hexdigest() != protocol.config_sha256:
        raise ValueError("contract payload differs from the evaluation protocol")
    manifests: dict[str, bytes] = {}
    for label in protocol.training_bundle_labels:
        root = Path(os.path.abspath(os.fspath(bundle_directories[label])))
        _reject_symlink_chain(root)
        metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o555:
            raise ValueError(f"training bundle {label} must be a read-only real directory")
        children = tuple(sorted(root.iterdir(), key=lambda item: item.name))
        if tuple(item.name for item in children) != tuple(sorted(protocol.training_bundle_files)):
            raise ValueError(f"training bundle {label} inventory differs from the contract")
        payloads: dict[str, bytes] = {}
        for child in children:
            child_stat = child.stat(follow_symlinks=False)
            if (
                child.is_symlink()
                or not stat.S_ISREG(child_stat.st_mode)
                or stat.S_IMODE(child_stat.st_mode) != 0o444
            ):
                raise ValueError(f"training bundle {label} contains a mutable or non-file entry")
            payloads[child.name] = _read_regular_bytes(child)
        if payloads["contract.toml"] != contract_payload:
            raise ValueError(f"training bundle {label} embeds a different contract")
        manifest = _strict_json_loads(
            payloads["manifest.json"],
            label=f"training manifest {label}",
        )
        artifact_hashes = manifest.get("artifacts")
        if type(artifact_hashes) is not dict:
            raise ValueError(f"training manifest {label} has no artifact digest map")
        for name in set(protocol.training_bundle_files) - {"manifest.json"}:
            if artifact_hashes.get(name) != hashlib.sha256(payloads[name]).hexdigest():
                raise ValueError(f"training bundle {label} artifact hash mismatch for {name}")
        training = manifest.get("training")
        if type(training) is not dict:
            raise ValueError(f"training manifest {label} has no training record")
        schedule_sha256 = _require_sha256(
            training.get("schedule_sha256"),
            label=f"training manifest {label} schedule SHA-256",
        )
        if payloads["training_schedule.sha256"] != f"{schedule_sha256}\n".encode("ascii"):
            raise ValueError(f"training bundle {label} schedule artifact differs from its manifest")
        final_metadata = root.stat(follow_symlinks=False)
        final_names = tuple(sorted(item.name for item in root.iterdir()))
        if (metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_ctime_ns) != (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        ) or final_names != tuple(sorted(protocol.training_bundle_files)):
            raise ValueError(f"training bundle {label} changed while it was verified")
        manifests[label] = payloads["manifest.json"]
    return training_bundle_binding_from_manifests(
        manifests,
        protocol=protocol,
        git_commit=git_commit,
    )


def _frozen_array(
    value: object,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...],
    label: str,
) -> NDArray[np.generic]:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{label} dtype must be {expected_dtype.str}, got {value.dtype.str}")
    if value.shape != shape:
        raise ValueError(f"{label} shape must be {shape}, got {value.shape}")
    result = np.ascontiguousarray(value).copy()
    if result.dtype.kind == "f" and np.any(~np.isfinite(result)):
        raise ValueError(f"{label} contains a non-finite value")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ValidationCorruptions:
    """Shared row table and all ordered stateless validation corruptions."""

    rows: tuple[DiffusionCorpusRow, ...]
    ledger: ValidationCaseLedger
    clean_tokens: NDArray[np.uint8]
    attention_mask: NDArray[np.bool_]
    corrupted_tokens: NDArray[np.uint8]
    selected_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not tuple
            or not self.rows
            or any(type(item) is not DiffusionCorpusRow for item in self.rows)
        ):
            raise ValueError("validation rows must be a non-empty DiffusionCorpusRow tuple")
        if type(self.ledger) is not ValidationCaseLedger:
            raise TypeError("ledger must be a ValidationCaseLedger")
        row_ids = tuple(item.sequence_id for item in self.rows)
        if row_ids != tuple(sorted(row_ids)) or len(row_ids) != len(set(row_ids)):
            raise ValueError("validation rows must have unique sorted sequence IDs")
        if any(item.role != "validation" or item.fold != 4 for item in self.rows):
            raise ValueError("validation corruptions accept only fold-4 validation rows")
        if set(row_ids) != {item.sequence_id for item in self.ledger.cases}:
            raise ValueError("validation rows and corruption ledger identities differ")
        width = (
            int(np.asarray(self.clean_tokens).shape[1])
            if np.asarray(self.clean_tokens).ndim == 2
            else -1
        )
        rows = len(self.rows)
        cases = len(self.ledger.cases)
        clean = _frozen_array(
            self.clean_tokens,
            dtype=np.dtype("|u1"),
            shape=(rows, width),
            label="clean_tokens",
        )
        attention = _frozen_array(
            self.attention_mask,
            dtype=np.dtype("|b1"),
            shape=(rows, width),
            label="attention_mask",
        )
        corrupted = _frozen_array(
            self.corrupted_tokens,
            dtype=np.dtype("|u1"),
            shape=(cases, width),
            label="corrupted_tokens",
        )
        selected = _frozen_array(
            self.selected_mask,
            dtype=np.dtype("|b1"),
            shape=(cases, width),
            label="selected_mask",
        )
        if width <= 0:
            raise ValueError("validation token width must be positive")
        vocabulary = PeptideVocabulary()
        encoded = vocabulary.encode([item.sequence for item in self.rows], max_length=width)
        if not np.array_equal(clean, encoded.tokens.astype(np.uint8)):
            raise ValueError("clean_tokens do not encode the declared validation sequences")
        if not np.array_equal(attention, encoded.attention_mask):
            raise ValueError("attention_mask does not encode the declared validation lengths")
        row_index = {sequence_id: index for index, sequence_id in enumerate(row_ids)}
        expected_case_order = tuple(
            (row_index[item.sequence_id], item.level, item.replicate) for item in self.ledger.cases
        )
        if expected_case_order != tuple(sorted(expected_case_order)):
            raise ValueError("corruption cases must be ordered by sequence_id, level, replicate")
        for index, case in enumerate(self.ledger.cases):
            source = row_index[case.sequence_id]
            valid = attention[source]
            chosen = selected[index]
            if np.any(chosen & ~valid) or int(np.sum(chosen, dtype=np.int64)) != case.mask_count:
                raise ValueError("selected_mask violates a case mask count or valid prefix")
            if np.any(corrupted[index, chosen] != vocabulary.mask_index):
                raise ValueError("selected positions must contain MASK in corrupted_tokens")
            if not np.array_equal(corrupted[index, ~chosen], clean[source, ~chosen]):
                raise ValueError("corruption modified a non-selected position")
        diffusion = AbsorbingDiffusion(
            vocabulary=vocabulary,
            schedule=CosineMaskSchedule(offset=0.008),
        )
        for start in range(0, cases, 256):
            batch_cases = self.ledger.cases[start : start + 256]
            source_indices = np.asarray(
                [row_index[item.sequence_id] for item in batch_cases], dtype=np.int64
            )
            expected_corrupted, expected_selected = diffusion.corrupt_fixed_count(
                clean[source_indices].astype(np.int64),
                attention[source_indices],
                np.asarray([item.level for item in batch_cases], dtype=np.int64),
                total_levels=self.ledger.levels,
                row_seeds=tuple(item.row_seed for item in batch_cases),
            )
            if not np.array_equal(
                corrupted[start : start + len(batch_cases)],
                expected_corrupted,
            ) or not np.array_equal(
                selected[start : start + len(batch_cases)],
                expected_selected,
            ):
                raise ValueError("validation corruption differs from its stateless row seed")
        for name, value in (
            ("clean_tokens", clean),
            ("attention_mask", attention),
            ("corrupted_tokens", corrupted),
            ("selected_mask", selected),
        ):
            object.__setattr__(self, name, value)

    @property
    def width(self) -> int:
        return int(self.clean_tokens.shape[1])

    @property
    def selected_count(self) -> int:
        return int(np.sum(self.selected_mask, dtype=np.int64))

    def npz_arrays(self) -> dict[str, NDArray[np.generic]]:
        row_lookup = {item.sequence_id: index for index, item in enumerate(self.rows)}
        cases = self.ledger.cases
        arrays: dict[str, NDArray[np.generic]] = {
            "sequence_id": np.asarray(
                [item.sequence_id.encode("ascii") for item in self.rows], dtype="|S64"
            ),
            "homology_component_id": np.asarray(
                [item.homology_component_id.encode("ascii") for item in self.rows], dtype="|S64"
            ),
            "union_component_id": np.asarray(
                [item.union_component_id.encode("ascii") for item in self.rows], dtype="|S64"
            ),
            "length": np.asarray([len(item.sequence) for item in self.rows], dtype="|u1"),
            "sampling_weight": np.asarray(
                [item.sampling_weight for item in self.rows], dtype="<f8"
            ),
            "clean_tokens": np.asarray(self.clean_tokens, dtype="|u1", order="C"),
            "attention_mask": np.asarray(self.attention_mask, dtype="|b1", order="C"),
            "case_id": np.asarray([item.case_id.encode("ascii") for item in cases], dtype="|S64"),
            "row_index": np.asarray([row_lookup[item.sequence_id] for item in cases], dtype="<u2"),
            "level": np.asarray([item.level for item in cases], dtype="|u1"),
            "replicate": np.asarray([item.replicate for item in cases], dtype="|u1"),
            "row_seed": np.asarray([item.row_seed for item in cases], dtype="<u8"),
            "mask_count": np.asarray([item.mask_count for item in cases], dtype="|u1"),
            "corrupted_tokens": np.asarray(self.corrupted_tokens, dtype="|u1", order="C"),
            "selected_mask": np.asarray(self.selected_mask, dtype="|b1", order="C"),
        }
        for value in arrays.values():
            value.flags.writeable = False
        return arrays

    def npz_bytes(self, protocol: EvaluationProtocol) -> bytes:
        if len(self.rows) != protocol.validation_sequences:
            raise ValueError("validation row census differs from the protocol")
        if len(self.ledger.cases) != protocol.corruption_cases:
            raise ValueError("validation case census differs from the protocol")
        if self.ledger.levels != protocol.levels or self.ledger.replicates != protocol.replicates:
            raise ValueError("validation ledger levels or replicates differ from the protocol")
        if self.width != protocol.width:
            raise ValueError("validation array width differs from the protocol")
        return deterministic_npz_bytes(protocol.corruption_schema, self.npz_arrays())


@dataclass(frozen=True, slots=True)
class ValidationTokenStatistics:
    """Per-selected-token sufficient statistics for every frozen method."""

    case_ids: tuple[str, ...]
    case_offsets: NDArray[np.uint64]
    position: NDArray[np.uint8]
    target_token: NDArray[np.uint8]
    methods: tuple[str, ...]
    target_log_probability: NDArray[np.float64]
    top1_confidence: NDArray[np.float64]
    top1_correct: NDArray[np.bool_]
    top3_correct: NDArray[np.bool_]
    multiclass_brier: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            type(self.case_ids) is not tuple
            or not self.case_ids
            or any(
                type(item) is not str or _SHA256_RE.fullmatch(item) is None
                for item in self.case_ids
            )
            or len(set(self.case_ids)) != len(self.case_ids)
        ):
            raise ValueError("token-stat case_ids must be unique lowercase SHA-256 strings")
        if (
            type(self.methods) is not tuple
            or not self.methods
            or len(set(self.methods)) != len(self.methods)
            or any(
                type(item) is not str or _METHOD_RE.fullmatch(item) is None for item in self.methods
            )
        ):
            raise ValueError("token-stat methods must be unique manifest-safe identifiers")
        cases = len(self.case_ids)
        offsets = _frozen_array(
            self.case_offsets,
            dtype=np.dtype("<u8"),
            shape=(cases + 1,),
            label="case_offsets",
        )
        if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError("case_offsets must start at zero and increase strictly")
        selected = int(offsets[-1])
        position = _frozen_array(
            self.position,
            dtype=np.dtype("|u1"),
            shape=(selected,),
            label="position",
        )
        target = _frozen_array(
            self.target_token,
            dtype=np.dtype("|u1"),
            shape=(selected,),
            label="target_token",
        )
        if np.any(target >= 20):
            raise ValueError("target_token contains a special or invalid token")
        shape = (len(self.methods), selected)
        log_probability = _frozen_array(
            self.target_log_probability,
            dtype=np.dtype("<f8"),
            shape=shape,
            label="target_log_probability",
        )
        confidence = _frozen_array(
            self.top1_confidence,
            dtype=np.dtype("<f8"),
            shape=shape,
            label="top1_confidence",
        )
        top1 = _frozen_array(
            self.top1_correct,
            dtype=np.dtype("|b1"),
            shape=shape,
            label="top1_correct",
        )
        top3 = _frozen_array(
            self.top3_correct,
            dtype=np.dtype("|b1"),
            shape=shape,
            label="top3_correct",
        )
        brier = _frozen_array(
            self.multiclass_brier,
            dtype=np.dtype("<f8"),
            shape=shape,
            label="multiclass_brier",
        )
        if np.any(log_probability > 1e-15):
            raise ValueError("target_log_probability cannot exceed log(1)")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("top1_confidence must lie in [0, 1]")
        if np.any(top1 & ~top3):
            raise ValueError("top3 correctness cannot be false when top1 is correct")
        if np.any((brier < 0.0) | (brier > 2.0 + 1e-12)):
            raise ValueError("multiclass_brier must lie in [0, 2]")
        for name, value in (
            ("case_offsets", offsets),
            ("position", position),
            ("target_token", target),
            ("target_log_probability", log_probability),
            ("top1_confidence", confidence),
            ("top1_correct", top1),
            ("top3_correct", top3),
            ("multiclass_brier", brier),
        ):
            object.__setattr__(self, name, value)

    @property
    def selected_count(self) -> int:
        return int(self.case_offsets[-1])

    def validate_against(self, corruptions: ValidationCorruptions) -> None:
        cases = corruptions.ledger.cases
        if self.case_ids != tuple(item.case_id for item in cases):
            raise ValueError("token-stat case IDs differ from the shared corruption ledger")
        expected_counts = np.asarray([item.mask_count for item in cases], dtype="<u8")
        if not np.array_equal(np.diff(self.case_offsets), expected_counts):
            raise ValueError("token-stat case offsets differ from corruption mask counts")
        row_lookup = {item.sequence_id: index for index, item in enumerate(corruptions.rows)}
        cursor = 0
        for case_index, case in enumerate(cases):
            positions = np.flatnonzero(corruptions.selected_mask[case_index]).astype(np.uint8)
            stop = cursor + len(positions)
            if not np.array_equal(self.position[cursor:stop], positions):
                raise ValueError("token-stat positions differ from the selected corruption masks")
            source = row_lookup[case.sequence_id]
            if not np.array_equal(
                self.target_token[cursor:stop],
                corruptions.clean_tokens[source, positions],
            ):
                raise ValueError("token-stat targets differ from clean validation tokens")
            cursor = stop
        if cursor != self.selected_count:
            raise AssertionError("selected-token validation did not consume every value")

    def npz_arrays(self) -> dict[str, NDArray[np.generic]]:
        arrays: dict[str, NDArray[np.generic]] = {
            "case_id": np.asarray([item.encode("ascii") for item in self.case_ids], dtype="|S64"),
            "case_offsets": np.asarray(self.case_offsets, dtype="<u8", order="C"),
            "position": np.asarray(self.position, dtype="|u1", order="C"),
            "target_token": np.asarray(self.target_token, dtype="|u1", order="C"),
            "method": np.asarray([item.encode("ascii") for item in self.methods], dtype="|S64"),
            "target_log_probability": np.asarray(
                self.target_log_probability, dtype="<f8", order="C"
            ),
            "top1_confidence": np.asarray(self.top1_confidence, dtype="<f8", order="C"),
            "top1_correct": np.asarray(self.top1_correct, dtype="|b1", order="C"),
            "top3_correct": np.asarray(self.top3_correct, dtype="|b1", order="C"),
            "multiclass_brier": np.asarray(self.multiclass_brier, dtype="<f8", order="C"),
        }
        for value in arrays.values():
            value.flags.writeable = False
        return arrays

    def npz_bytes(
        self,
        protocol: EvaluationProtocol,
        corruptions: ValidationCorruptions,
    ) -> bytes:
        self.validate_against(corruptions)
        if self.methods != protocol.token_stats_methods:
            raise ValueError("token-stat method axis differs from the protocol")
        if len(self.case_ids) != protocol.corruption_cases:
            raise ValueError("token-stat case census differs from the protocol")
        return deterministic_npz_bytes(protocol.token_stats_schema, self.npz_arrays())


def evaluation_results_from_token_statistics(
    token_statistics: ValidationTokenStatistics,
    corruptions: ValidationCorruptions,
) -> tuple[EvaluationResult, ...]:
    """Recompute all denoising metrics solely from archived sufficient statistics."""

    if type(token_statistics) is not ValidationTokenStatistics:
        raise TypeError("token_statistics must be ValidationTokenStatistics")
    if type(corruptions) is not ValidationCorruptions:
        raise TypeError("corruptions must be ValidationCorruptions")
    token_statistics.validate_against(corruptions)
    results: list[EvaluationResult] = []
    cases = corruptions.ledger.cases
    for method_index, method in enumerate(token_statistics.methods):
        metrics: list[CaseMetrics] = []
        for case_index, case in enumerate(cases):
            start = int(token_statistics.case_offsets[case_index])
            stop = int(token_statistics.case_offsets[case_index + 1])
            count = stop - start
            log_probability = token_statistics.target_log_probability[method_index, start:stop]
            confidence = token_statistics.top1_confidence[method_index, start:stop]
            top1 = token_statistics.top1_correct[method_index, start:stop]
            top3 = token_statistics.top3_correct[method_index, start:stop]
            brier = token_statistics.multiclass_brier[method_index, start:stop]
            target_probability = np.exp(log_probability)
            if np.any(target_probability > confidence + 1e-12):
                raise ValueError("target probability exceeds declared top-1 confidence")
            if np.any(top1 & ~np.isclose(target_probability, confidence, rtol=0.0, atol=1e-12)):
                raise ValueError("correct top-1 predictions must expose target probability")
            calibration_counts = [0] * CALIBRATION_BINS
            calibration_confidence = [0.0] * CALIBRATION_BINS
            calibration_correct = [0] * CALIBRATION_BINS
            for probability, correct in zip(confidence, top1, strict=True):
                probability_value = float(probability)
                bin_index = min(
                    CALIBRATION_BINS - 1,
                    int(probability_value * CALIBRATION_BINS),
                )
                calibration_counts[bin_index] += 1
                calibration_confidence[bin_index] += probability_value
                calibration_correct[bin_index] += int(correct)
            metrics.append(
                CaseMetrics(
                    method=method,
                    case_id=case.case_id,
                    sequence_id=case.sequence_id,
                    homology_component_id=case.homology_component_id,
                    union_component_id=case.union_component_id,
                    level=case.level,
                    replicate=case.replicate,
                    sampling_weight=case.sampling_weight,
                    masked_tokens=count,
                    mean_nll=math.fsum(float(-item) for item in log_probability) / count,
                    top1_accuracy=int(np.sum(top1, dtype=np.int64)) / count,
                    top3_accuracy=int(np.sum(top3, dtype=np.int64)) / count,
                    mean_brier=math.fsum(float(item) for item in brier) / count,
                    calibration_counts=tuple(calibration_counts),
                    calibration_confidence_sums=tuple(calibration_confidence),
                    calibration_correct_sums=tuple(calibration_correct),
                )
            )
        results.append(aggregate_case_metrics(corruptions.ledger, metrics))
    return tuple(results)


def _validate_baseline_token_statistics(
    token_statistics: ValidationTokenStatistics,
    corruptions: ValidationCorruptions,
    *,
    training: TrainingDistribution,
    protocol: EvaluationProtocol,
) -> CountBaselineSuite:
    """Recompute archived count-control token statistics from training only."""

    suite = fit_count_baselines(
        training,
        require_locked_census=False,
        effective_count_scale=protocol.training_sequences,
    )
    row_lookup = {item.sequence_id: index for index, item in enumerate(corruptions.rows)}
    lengths_by_row = np.asarray([len(item.sequence) for item in corruptions.rows], dtype=np.int64)
    cases = corruptions.ledger.cases
    for method_index, method in enumerate(protocol.baseline_methods):
        for batch_start in range(0, len(cases), protocol.evaluation_batch_sequences):
            batch_cases = cases[batch_start : batch_start + protocol.evaluation_batch_sequences]
            row_indices = np.asarray(
                [row_lookup[case.sequence_id] for case in batch_cases],
                dtype=np.int64,
            )
            probabilities = suite.probabilities(
                method,
                corruptions.corrupted_tokens[batch_start : batch_start + len(batch_cases)].astype(
                    np.int64
                ),
                corruptions.attention_mask[row_indices],
                lengths_by_row[row_indices],
            )
            for local_index in range(len(batch_cases)):
                case_index = batch_start + local_index
                start = int(token_statistics.case_offsets[case_index])
                stop = int(token_statistics.case_offsets[case_index + 1])
                positions = token_statistics.position[start:stop].astype(np.int64)
                targets = token_statistics.target_token[start:stop].astype(np.int64)
                selected = probabilities[local_index, positions]
                target_probability = selected[np.arange(len(positions)), targets]
                ranking = np.argsort(-selected, axis=1, kind="stable")
                prediction = ranking[:, 0]
                expected_log_probability = np.log(target_probability)
                expected_confidence = selected[np.arange(len(positions)), prediction]
                expected_top1 = prediction == targets
                expected_top3 = np.any(ranking[:, :3] == targets[:, None], axis=1)
                expected_brier = (
                    np.sum(np.square(selected), axis=1) - 2.0 * target_probability + 1.0
                )
                comparisons = (
                    (
                        token_statistics.target_log_probability[method_index, start:stop],
                        expected_log_probability,
                        "target_log_probability",
                    ),
                    (
                        token_statistics.top1_confidence[method_index, start:stop],
                        expected_confidence,
                        "top1_confidence",
                    ),
                    (
                        token_statistics.multiclass_brier[method_index, start:stop],
                        expected_brier,
                        "multiclass_brier",
                    ),
                )
                for observed, expected, label in comparisons:
                    if not np.allclose(observed, expected, rtol=0.0, atol=1e-14):
                        raise ValueError(
                            f"archived baseline {label} differs from training-only counts"
                        )
                if not np.array_equal(
                    token_statistics.top1_correct[method_index, start:stop],
                    expected_top1,
                ) or not np.array_equal(
                    token_statistics.top3_correct[method_index, start:stop],
                    expected_top3,
                ):
                    raise ValueError("archived baseline ranking statistics differ from counts")
    return suite


@dataclass(frozen=True, slots=True)
class OrganizerReference:
    """Byte identity and parsed sequence population of the post-generation input."""

    sequences: tuple[str, ...]
    file_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.sequences) is not tuple
            or not self.sequences
            or any(type(item) is not str or not item for item in self.sequences)
        ):
            raise ValueError("organizer reference must contain a non-empty sequence tuple")
        _require_sha256(self.file_sha256, label="organizer reference SHA-256")


def load_organizer_reference(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_records: int,
) -> OrganizerReference:
    """Load only the hash-pinned post-generation FASTA using organizer semantics."""

    digest = _require_sha256(expected_sha256, label="expected organizer reference SHA-256")
    if type(expected_records) is not int or expected_records <= 0:
        raise ValueError("expected organizer reference records must be positive")
    payload = _read_regular_bytes(Path(path))
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("organizer reference FASTA SHA-256 differs from the contract")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("organizer reference FASTA is not UTF-8") from error
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
        raise ValueError("organizer reference FASTA record census differs from the contract")
    alphabet = set(PeptideVocabulary().alphabet)
    if any(not item or set(item) - alphabet for item in sequences):
        raise ValueError("organizer reference contains a noncanonical amino-acid sequence")
    return OrganizerReference(sequences=tuple(sequences), file_sha256=digest)


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """One normalized raw candidate-ledger row shared by all three methods."""

    method: str
    seed: int
    ordinal: int
    sequence_id: str
    sequence: str
    length: int
    generator_binding_kind: str
    generator_binding_sha256: str
    config_sha256: str
    training_projection_sha256: str
    length_plan_sha256: str

    def __post_init__(self) -> None:
        if type(self.method) is not str or _METHOD_RE.fullmatch(self.method) is None:
            raise ValueError("proposal method must be a manifest-safe identifier")
        _uint64(self.seed, label="proposal seed")
        _uint64(self.ordinal, label="proposal ordinal")
        if type(self.length) is not int or not 8 <= self.length <= 50:
            raise ValueError("proposal length must be an integer in 8..50")
        if (
            type(self.sequence) is not str
            or len(self.sequence) != self.length
            or set(self.sequence) - set(PeptideVocabulary().alphabet)
        ):
            raise ValueError("proposal sequence is noncanonical or disagrees with length")
        if canonical_sequence_id(self.sequence) != self.sequence_id:
            raise ValueError("proposal sequence identity is inconsistent")
        if self.generator_binding_kind not in {
            "checkpoint_contract_logical_sha256",
            "count_control_logical_sha256",
        }:
            raise ValueError("proposal generator binding kind is invalid")
        for field_name in (
            "generator_binding_sha256",
            "config_sha256",
            "training_projection_sha256",
            "length_plan_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=f"proposal {field_name}")

    def canonical_record(self) -> dict[str, object]:
        result = {
            "schema_version": 1,
            "method": self.method,
            "seed": self.seed,
            "ordinal": self.ordinal,
            "sequence_id": self.sequence_id,
            "sequence": self.sequence,
            "length": self.length,
            "generator_binding_kind": self.generator_binding_kind,
            "generator_binding_sha256": self.generator_binding_sha256,
            "config_sha256": self.config_sha256,
            "training_projection_sha256": self.training_projection_sha256,
            "length_plan_sha256": self.length_plan_sha256,
        }
        if tuple(result) != CANDIDATE_LEDGER_FIELDS:
            raise AssertionError("candidate ledger record schema changed")
        return result


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    """One seed/method raw proposal census with duplicates preserved."""

    method: str
    seed: int
    records: tuple[ProposalRecord, ...]
    length_plan_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.records) is not tuple
            or not self.records
            or any(type(item) is not ProposalRecord for item in self.records)
        ):
            raise ValueError("proposal batch must contain ProposalRecord values")
        _uint64(self.seed, label="proposal batch seed")
        _require_sha256(self.length_plan_sha256, label="proposal batch length plan")
        if any(
            item.method != self.method
            or item.seed != self.seed
            or item.length_plan_sha256 != self.length_plan_sha256
            for item in self.records
        ):
            raise ValueError("proposal batch records disagree with their batch identity")
        ordinals = tuple(item.ordinal for item in self.records)
        if ordinals != tuple(sorted(ordinals)) or len(ordinals) != len(set(ordinals)):
            raise ValueError("proposal batch ordinals must be unique and sorted")
        _, observed_hash = canonical_length_plan(
            [item.length for item in self.records],
            ordinals=ordinals,
            require_locked_count=False,
        )
        if observed_hash != self.length_plan_sha256:
            raise ValueError("proposal records disagree with their length-plan SHA-256")

    @classmethod
    def from_diffusion(
        cls,
        result: SamplingResult,
        *,
        protocol: EvaluationProtocol,
    ) -> ProposalBatch:
        if type(result) is not SamplingResult:
            raise TypeError("diffusion result must be a SamplingResult")
        seed = result.candidates[0].seed
        records = tuple(
            ProposalRecord(
                method="native_categorical_diffusion",
                seed=item.seed,
                ordinal=item.ordinal,
                sequence_id=item.sequence_id,
                sequence=item.sequence,
                length=item.length,
                generator_binding_kind="checkpoint_contract_logical_sha256",
                generator_binding_sha256=item.checkpoint_logical_sha256,
                config_sha256=item.contract_sha256,
                training_projection_sha256=protocol.training_projection_sha256,
                length_plan_sha256=result.length_plan_sha256,
            )
            for item in result.candidates
        )
        return cls(
            method="native_categorical_diffusion",
            seed=seed,
            records=records,
            length_plan_sha256=result.length_plan_sha256,
        )

    @classmethod
    def from_control(
        cls,
        result: GeneratorControlSamplingResult,
        *,
        protocol: EvaluationProtocol,
    ) -> ProposalBatch:
        if type(result) is not GeneratorControlSamplingResult:
            raise TypeError("control result must be a GeneratorControlSamplingResult")
        first = result.candidates[0]
        records = tuple(
            ProposalRecord(
                method=item.method,
                seed=item.seed,
                ordinal=item.ordinal,
                sequence_id=item.sequence_id,
                sequence=item.sequence,
                length=item.length,
                generator_binding_kind="count_control_logical_sha256",
                generator_binding_sha256=item.control_logical_sha256,
                config_sha256=protocol.config_sha256,
                training_projection_sha256=item.training_projection_sha256,
                length_plan_sha256=result.length_plan_sha256,
            )
            for item in result.candidates
        )
        return cls(
            method=first.method,
            seed=first.seed,
            records=records,
            length_plan_sha256=result.length_plan_sha256,
        )


def _training_projection_digest(distribution: TrainingDistribution) -> str:
    payload = b"".join(
        canonical_json_bytes(
            {
                "sampling_weight": item.sampling_weight,
                "sequence": item.sequence,
                "sequence_id": item.sequence_id,
            }
        )
        for item in distribution.rows
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SamplingMethodMetrics:
    method: str
    seed: int
    length_plan_sha256: str
    diagnostics: CandidateDiagnostics
    ngram_1: NgramDistanceResult
    ngram_3: NgramDistanceResult
    descriptor: DescriptorEnergyDistanceResult

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": self.method,
            "seed": self.seed,
            "length_plan_sha256": self.length_plan_sha256,
            "diagnostics": self.diagnostics.canonical_record(),
            "ngram_jsd": {
                "1": self.ngram_1.canonical_record(),
                "3": self.ngram_3.canonical_record(),
            },
            "descriptor_energy_distance": self.descriptor.canonical_record(),
        }


def compute_sampling_metrics(
    batches: Sequence[ProposalBatch],
    *,
    training: TrainingDistribution,
    reference: OrganizerReference,
    protocol: EvaluationProtocol,
    require_locked_census: bool,
) -> tuple[SamplingMethodMetrics, ...]:
    """Recompute every sampling diagnostic and distance from raw candidates."""

    values = tuple(batches)
    expected_keys = tuple(
        (method, seed) for method in protocol.proposal_methods for seed in protocol.seeds
    )
    if tuple((item.method, item.seed) for item in values) != expected_keys:
        raise ValueError("proposal batches must follow the exact method-major cohort order")
    projection_digest = _training_projection_digest(training)
    if projection_digest != protocol.training_projection_sha256:
        raise ValueError("sampling training projection differs from the protocol")
    if reference.file_sha256 != protocol.organizer_reference_sha256:
        raise ValueError("sampling organizer reference differs from the protocol")
    if len(reference.sequences) != protocol.organizer_reference_records:
        raise ValueError("sampling organizer reference census differs from the protocol")
    output: list[SamplingMethodMetrics] = []
    for batch in values:
        if len(batch.records) != protocol.raw_proposals_per_seed:
            raise ValueError("raw proposal census differs from the protocol")
        if any(
            item.config_sha256 != protocol.config_sha256
            or item.training_projection_sha256 != protocol.training_projection_sha256
            for item in batch.records
        ):
            raise ValueError("proposal records do not bind the protocol inputs")
        sequences = tuple(item.sequence for item in batch.records)
        diagnostics = candidate_diagnostics(
            sequences,
            train_sequences=training,
            reference_sequences=reference.sequences,
            require_locked_protocol=require_locked_census,
        )
        pool = build_distribution_candidate_pool(
            sequences,
            reference_sequences=reference.sequences,
        )
        ngram_1 = sampling_ngram_jensen_shannon(
            pool,
            training,
            order=1,
            require_locked_census=require_locked_census,
        )
        ngram_3 = sampling_ngram_jensen_shannon(
            pool,
            training,
            order=3,
            require_locked_census=require_locked_census,
        )
        descriptor = descriptor_energy_distance(
            pool,
            training,
            require_locked_census=require_locked_census,
        )
        output.append(
            SamplingMethodMetrics(
                method=batch.method,
                seed=batch.seed,
                length_plan_sha256=batch.length_plan_sha256,
                diagnostics=diagnostics,
                ngram_1=ngram_1,
                ngram_3=ngram_3,
                descriptor=descriptor,
            )
        )
    return tuple(output)


def _relative_improvement(model_value: float, control_value: float, *, label: str) -> float:
    _finite(model_value, label=f"{label} model value")
    if _finite(control_value, label=f"{label} control value") <= 0.0:
        raise ValueError(f"{label} control value must be positive")
    result = (control_value - model_value) / control_value
    if not math.isfinite(result):
        raise ValueError(f"{label} relative improvement is non-finite")
    return result


def _timestep_levels(name: str, *, maximum: int) -> frozenset[int]:
    if name == "64_fully_masked":
        if maximum != 64:
            raise ValueError("fully masked bin is defined only for the 64-level v0 protocol")
        return frozenset({64})
    match = re.fullmatch(r"([1-9][0-9]*)_to_([1-9][0-9]*)", name)
    if match is None:
        raise ValueError(f"invalid timestep-bin label: {name!r}")
    lower, upper = (int(match.group(1)), int(match.group(2)))
    if not 1 <= lower <= upper <= maximum:
        raise ValueError(f"timestep-bin bounds are outside 1..{maximum}")
    return frozenset(range(lower, upper + 1))


def _subset_component_balanced_nll(
    result: EvaluationResult,
    levels: frozenset[int],
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    weights: dict[str, float] = {}
    for item in result.case_metrics:
        if item.level in levels:
            grouped[item.sequence_id].append(item.mean_nll)
            prior = weights.setdefault(item.sequence_id, item.sampling_weight)
            if prior != item.sampling_weight:
                raise ValueError("case sampling weight changes within a validation row")
    row_ids = tuple(item.sequence_id for item in result.row_metrics)
    if set(grouped) != set(row_ids) or any(not values for values in grouped.values()):
        raise ValueError("timestep bin does not contain every validation sequence")
    value = math.fsum(
        weights[sequence_id] * math.fsum(grouped[sequence_id]) / len(grouped[sequence_id])
        for sequence_id in sorted(grouped)
    )
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("timestep-bin component-balanced NLL is invalid")
    return value


@dataclass(frozen=True, slots=True)
class CohortBootstrapResult:
    baseline_method: str
    model_methods: tuple[str, ...]
    unit: str
    unit_count: int
    replicates: int
    seed: int
    statistic: str
    point_mean_relative_nll_improvement: float
    lower_95: float
    upper_95: float
    draws_sha256: str

    def __post_init__(self) -> None:
        if self.unit != "union_component_id":
            raise ValueError("cohort bootstrap unit must be union_component_id")
        if self.statistic != "arithmetic_mean_relative_improvement_across_seeds":
            raise ValueError("cohort bootstrap statistic differs from the protocol")
        if type(self.unit_count) is not int or self.unit_count <= 0:
            raise ValueError("cohort bootstrap unit count must be positive")
        if type(self.replicates) is not int or self.replicates <= 0:
            raise ValueError("cohort bootstrap replicates must be positive")
        _uint64(self.seed, label="cohort bootstrap seed")
        for field_name in (
            "point_mean_relative_nll_improvement",
            "lower_95",
            "upper_95",
        ):
            _finite(getattr(self, field_name), label=field_name)
        if self.lower_95 > self.upper_95:
            raise ValueError("cohort bootstrap interval bounds are reversed")
        _require_sha256(self.draws_sha256, label="cohort bootstrap draws SHA-256")

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "baseline_method": self.baseline_method,
            "model_methods": list(self.model_methods),
            "unit": self.unit,
            "unit_count": self.unit_count,
            "replicates": self.replicates,
            "seed": self.seed,
            "statistic": self.statistic,
            "point_mean_relative_nll_improvement": (self.point_mean_relative_nll_improvement),
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "draws_sha256": self.draws_sha256,
        }


def cohort_union_component_bootstrap(
    models: Sequence[EvaluationResult],
    baseline: EvaluationResult,
    *,
    replicates: int,
    seed: int,
) -> CohortBootstrapResult:
    """Bootstrap the arithmetic mean of three paired seed improvements."""

    model_values = tuple(models)
    if not model_values or any(type(item) is not EvaluationResult for item in model_values):
        raise TypeError("cohort bootstrap models must be EvaluationResult values")
    if type(baseline) is not EvaluationResult:
        raise TypeError("cohort bootstrap baseline must be an EvaluationResult")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("cohort bootstrap replicates must be positive")
    _uint64(seed, label="cohort bootstrap seed")
    baseline_rows = {item.sequence_id: item for item in baseline.row_metrics}
    model_rows = tuple(
        {item.sequence_id: item for item in result.row_metrics} for result in model_values
    )
    if not baseline_rows or any(set(values) != set(baseline_rows) for values in model_rows):
        raise ValueError("cohort bootstrap row identities do not align")
    for sequence_id, baseline_row in baseline_rows.items():
        for values in model_rows:
            row = values[sequence_id]
            if (
                row.union_component_id != baseline_row.union_component_id
                or row.homology_component_id != baseline_row.homology_component_id
                or row.sampling_weight != baseline_row.sampling_weight
            ):
                raise ValueError("cohort bootstrap paired row metadata differs")
    union_ids = tuple(sorted({item.union_component_id for item in baseline_rows.values()}))
    union_index = {value: index for index, value in enumerate(union_ids)}
    masses = np.zeros(len(union_ids), dtype=np.float64)
    baseline_sums = np.zeros(len(union_ids), dtype=np.float64)
    model_sums = np.zeros((len(model_rows), len(union_ids)), dtype=np.float64)
    for sequence_id in sorted(baseline_rows):
        row = baseline_rows[sequence_id]
        index = union_index[row.union_component_id]
        masses[index] += row.sampling_weight
        baseline_sums[index] += row.sampling_weight * row.mean_nll
        for model_index, values in enumerate(model_rows):
            model_sums[model_index, index] += row.sampling_weight * values[sequence_id].mean_nll

    def estimate(multiplicity: NDArray[np.int64]) -> float:
        mass = math.fsum(float(multiplicity[index] * masses[index]) for index in range(len(masses)))
        baseline_sum = math.fsum(
            float(multiplicity[index] * baseline_sums[index]) for index in range(len(masses))
        )
        if mass <= 0.0 or baseline_sum <= 0.0:
            raise RuntimeError("cohort bootstrap draw has invalid mass or control NLL")
        control_nll = baseline_sum / mass
        improvements = []
        for model_index in range(len(model_rows)):
            model_sum = math.fsum(
                float(multiplicity[index] * model_sums[model_index, index])
                for index in range(len(masses))
            )
            improvements.append((control_nll - model_sum / mass) / control_nll)
        return math.fsum(improvements) / len(improvements)

    samples = np.empty(replicates, dtype=np.float64)
    draws_digest = hashlib.sha256()
    draws_digest.update(_BOOTSTRAP_DOMAIN)
    for replicate in range(replicates):
        rng = np.random.Generator(
            np.random.PCG64DXSM(namespaced_seed(seed, "bootstrap", replicate))
        )
        draws = rng.integers(0, len(union_ids), size=len(union_ids))
        multiplicity = np.bincount(draws, minlength=len(union_ids)).astype(np.int64)
        for slot, draw in enumerate(draws):
            draws_digest.update(f"{replicate}\t{slot}\t{union_ids[int(draw)]}\n".encode("ascii"))
        samples[replicate] = estimate(multiplicity)
    if np.any(~np.isfinite(samples)):
        raise RuntimeError("cohort bootstrap produced a non-finite statistic")
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return CohortBootstrapResult(
        baseline_method=baseline.method,
        model_methods=tuple(item.method for item in model_values),
        unit="union_component_id",
        unit_count=len(union_ids),
        replicates=replicates,
        seed=seed,
        statistic="arithmetic_mean_relative_improvement_across_seeds",
        point_mean_relative_nll_improvement=estimate(np.ones(len(union_ids), dtype=np.int64)),
        lower_95=float(lower),
        upper_95=float(upper),
        draws_sha256=draws_digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ProducerEvidenceChecks:
    """Direct producer checks; any false value has decision precedence."""

    accepted_input_hashes: bool
    clean_synchronized_git: bool
    training_bundle_integrity: bool
    runtime_environment: bool
    deterministic_execution: bool
    validation_execution: bool
    sampling_execution: bool
    finite_values: bool

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not bool for name in PRODUCER_CHECK_NAMES):
            raise TypeError("every producer evidence check must be a canonical bool")

    @property
    def passed(self) -> bool:
        return all(getattr(self, name) for name in PRODUCER_CHECK_NAMES)

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "checks": [
                {"name": name, "passed": getattr(self, name)} for name in PRODUCER_CHECK_NAMES
            ],
        }


@dataclass(frozen=True, slots=True)
class DenoisingGateResult:
    strongest_baseline_method: str
    per_seed: tuple[dict[str, object], ...]
    mean_relative_nll_improvement: float
    bootstrap: CohortBootstrapResult
    timestep_bins: tuple[dict[str, object], ...]
    every_seed_beats: bool
    mean_improvement_passed: bool
    bootstrap_passed: bool
    high_noise_passed: bool
    timestep_regression_passed: bool
    ece_passed: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.every_seed_beats,
                self.mean_improvement_passed,
                self.bootstrap_passed,
                self.high_noise_passed,
                self.timestep_regression_passed,
                self.ece_passed,
            )
        )

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "strongest_baseline_method": self.strongest_baseline_method,
            "per_seed": list(self.per_seed),
            "mean_relative_nll_improvement": self.mean_relative_nll_improvement,
            "bootstrap": self.bootstrap.canonical_record(),
            "timestep_bins": list(self.timestep_bins),
            "checks": {
                "every_seed_beats_strongest_baseline": self.every_seed_beats,
                "minimum_mean_relative_nll_improvement": self.mean_improvement_passed,
                "strict_bootstrap_lower_bound": self.bootstrap_passed,
                "minimum_high_noise_relative_nll_improvement": self.high_noise_passed,
                "maximum_timestep_bin_relative_nll_regression": (self.timestep_regression_passed),
                "every_seed_ece": self.ece_passed,
            },
        }


def evaluate_denoising_gates(
    results: Sequence[EvaluationResult],
    *,
    protocol: EvaluationProtocol,
) -> DenoisingGateResult:
    """Evaluate every preregistered denoising comparison from raw-stat aggregates."""

    values = tuple(results)
    if tuple(item.method for item in values) != protocol.token_stats_methods:
        raise ValueError("denoising results differ from the frozen method axis")
    ledger_hashes = {item.ledger_sha256 for item in values}
    if len(ledger_hashes) != 1:
        raise ValueError("denoising methods do not share one corruption ledger")
    baselines = values[: len(protocol.baseline_methods)]
    models = values[len(protocol.baseline_methods) :]
    if len(models) != len(protocol.seeds):
        raise ValueError("denoising model result count differs from the seed cohort")
    strongest = min(baselines, key=lambda item: (item.primary_nll, item.method))
    per_seed: list[dict[str, object]] = []
    improvements: list[float] = []
    every_seed_beats = True
    every_seed_ece = True
    for seed, model in zip(protocol.seeds, models, strict=True):
        improvement = _relative_improvement(
            model.primary_nll,
            strongest.primary_nll,
            label=f"seed {seed} primary NLL",
        )
        ece_regression = model.ece - strongest.ece
        beats = model.primary_nll < strongest.primary_nll
        ece_absolute_passed = model.ece <= protocol.maximum_ece
        ece_regression_passed = ece_regression <= protocol.maximum_ece_regression
        improvements.append(improvement)
        every_seed_beats &= beats
        every_seed_ece &= ece_absolute_passed and ece_regression_passed
        per_seed.append(
            {
                "seed": seed,
                "method": model.method,
                "primary_nll": model.primary_nll,
                "relative_nll_improvement": improvement,
                "beats_strongest_baseline": beats,
                "ece": model.ece,
                "ece_regression": ece_regression,
                "maximum_ece_passed": ece_absolute_passed,
                "maximum_ece_regression_passed": ece_regression_passed,
            }
        )
    mean_improvement = math.fsum(improvements) / len(improvements)
    bootstrap = cohort_union_component_bootstrap(
        models,
        strongest,
        replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )
    if not math.isclose(
        bootstrap.point_mean_relative_nll_improvement,
        mean_improvement,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("cohort bootstrap point estimate differs from validation summaries")
    bin_records: list[dict[str, object]] = []
    high_noise_passed = False
    regression_passed = True
    for name in protocol.timestep_bins:
        levels = _timestep_levels(name, maximum=protocol.levels)
        baseline_nll = _subset_component_balanced_nll(strongest, levels)
        model_nlls = tuple(_subset_component_balanced_nll(item, levels) for item in models)
        model_mean = math.fsum(model_nlls) / len(model_nlls)
        improvement = _relative_improvement(
            model_mean,
            baseline_nll,
            label=f"timestep bin {name}",
        )
        regression = -improvement
        within_regression = regression <= protocol.maximum_timestep_bin_relative_nll_regression
        regression_passed &= within_regression
        is_high_noise = name == "49_to_64"
        high_pass = (
            improvement >= protocol.minimum_high_noise_relative_nll_improvement
            if is_high_noise
            else None
        )
        if is_high_noise:
            high_noise_passed = bool(high_pass)
        bin_records.append(
            {
                "name": name,
                "baseline_nll": baseline_nll,
                "model_seed_nll": list(model_nlls),
                "model_arithmetic_mean_nll": model_mean,
                "relative_nll_improvement": improvement,
                "relative_nll_regression": regression,
                "maximum_regression_passed": within_regression,
                "high_noise_minimum_passed": high_pass,
            }
        )
    if not any(item["name"] == "49_to_64" for item in bin_records):
        raise ValueError("timestep bins omit the preregistered high-noise bin")
    return DenoisingGateResult(
        strongest_baseline_method=strongest.method,
        per_seed=tuple(per_seed),
        mean_relative_nll_improvement=mean_improvement,
        bootstrap=bootstrap,
        timestep_bins=tuple(bin_records),
        every_seed_beats=every_seed_beats,
        mean_improvement_passed=(
            mean_improvement >= protocol.minimum_mean_relative_nll_improvement
        ),
        bootstrap_passed=(bootstrap.lower_95 > protocol.minimum_bootstrap_lower_bound_improvement),
        high_noise_passed=high_noise_passed,
        timestep_regression_passed=regression_passed,
        ece_passed=every_seed_ece,
    )


@dataclass(frozen=True, slots=True)
class SamplingGateResult:
    per_seed: tuple[dict[str, object], ...]
    method_medians: tuple[dict[str, object], ...]
    relative_comparisons: dict[str, object]
    absolute_passed: bool
    relative_passed: bool

    @property
    def passed(self) -> bool:
        return self.absolute_passed and self.relative_passed

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "absolute_each_seed_passed": self.absolute_passed,
            "relative_median_passed": self.relative_passed,
            "per_seed": list(self.per_seed),
            "method_medians": list(self.method_medians),
            "relative_comparisons": self.relative_comparisons,
        }


def _distance_improvement(candidate: float, control: float) -> float:
    if control == 0.0:
        return 0.0 if candidate == 0.0 else -math.inf
    return (control - candidate) / control


def evaluate_sampling_gates(
    metrics: Sequence[SamplingMethodMetrics],
    *,
    protocol: EvaluationProtocol,
) -> SamplingGateResult:
    """Evaluate per-seed absolute gates and metric-specific best-control medians."""

    values = tuple(metrics)
    expected_keys = tuple(
        (method, seed) for method in protocol.proposal_methods for seed in protocol.seeds
    )
    if tuple((item.method, item.seed) for item in values) != expected_keys:
        raise ValueError("sampling metrics differ from the frozen method-major cohort axis")
    by_key = {(item.method, item.seed): item for item in values}
    native = protocol.proposal_methods[0]
    controls = protocol.proposal_methods[1:]
    per_seed: list[dict[str, object]] = []
    absolute_passed = True
    for seed in protocol.seeds:
        item = by_key[(native, seed)]
        diagnostics = item.diagnostics
        checks = {
            "canonical_valid_fraction": (
                diagnostics.canonical_valid_fraction
                >= protocol.minimum_canonical_valid_fraction_each_seed
            ),
            "raw_unique_fraction": (
                diagnostics.raw_unique_fraction >= protocol.minimum_raw_unique_fraction_each_seed
            ),
            "exact_train_overlap_fraction": (
                diagnostics.exact_train_overlap_fraction
                <= protocol.maximum_exact_train_overlap_fraction_each_seed
            ),
            "common_funnel_yield_fraction": (
                diagnostics.common_funnel_yield_fraction
                >= protocol.minimum_common_funnel_yield_fraction_each_seed
            ),
            "top_reference_safe_count": (
                diagnostics.top_reference_safe_count
                >= protocol.minimum_top_reference_safe_count_each_seed
            ),
            "hill2_effective_70pct_clusters": (
                diagnostics.hill2_effective_70pct_clusters
                >= protocol.minimum_hill2_effective_70pct_clusters_each_seed
            ),
            "largest_70pct_cluster_fraction": (
                diagnostics.largest_70pct_cluster_fraction
                <= protocol.maximum_largest_70pct_cluster_fraction_each_seed
            ),
        }
        passed = all(checks.values())
        absolute_passed &= passed
        per_seed.append(
            {
                "seed": seed,
                "passed": passed,
                "checks": checks,
                "diagnostics": diagnostics.canonical_record(),
            }
        )

    aggregate_fields: tuple[tuple[str, Callable[[SamplingMethodMetrics], float]], ...] = (
        (
            "common_funnel_yield_fraction",
            lambda item: item.diagnostics.common_funnel_yield_fraction,
        ),
        ("ngram_1_jsd_bits", lambda item: item.ngram_1.bits),
        ("ngram_3_jsd_bits", lambda item: item.ngram_3.bits),
        ("descriptor_energy_distance", lambda item: item.descriptor.distance),
    )
    medians: dict[str, dict[str, float]] = {}
    method_records: list[dict[str, object]] = []
    for method in protocol.proposal_methods:
        method_values = [by_key[(method, seed)] for seed in protocol.seeds]
        record: dict[str, float] = {}
        for name, getter in aggregate_fields:
            record[name] = float(median(getter(item) for item in method_values))
        medians[method] = record
        method_records.append({"method": method, **record})

    native_values = medians[native]
    best_yield = max(medians[item]["common_funnel_yield_fraction"] for item in controls)
    best_one = min(medians[item]["ngram_1_jsd_bits"] for item in controls)
    best_three = min(medians[item]["ngram_3_jsd_bits"] for item in controls)
    best_descriptor = min(medians[item]["descriptor_energy_distance"] for item in controls)
    yield_deficit = best_yield - native_values["common_funnel_yield_fraction"]
    one_regression = native_values["ngram_1_jsd_bits"] - best_one
    three_regression = native_values["ngram_3_jsd_bits"] - best_three
    if best_descriptor == 0.0:
        descriptor_ratio = 1.0 if native_values["descriptor_energy_distance"] == 0.0 else math.inf
    else:
        descriptor_ratio = native_values["descriptor_energy_distance"] / best_descriptor
    three_improvement = _distance_improvement(native_values["ngram_3_jsd_bits"], best_three)
    descriptor_improvement = _distance_improvement(
        native_values["descriptor_energy_distance"], best_descriptor
    )
    comparisons = {
        "best_control_common_funnel_yield_fraction": best_yield,
        "native_common_funnel_yield_deficit": yield_deficit,
        "common_funnel_yield_passed": (
            yield_deficit <= protocol.maximum_common_funnel_yield_deficit_vs_best_control
        ),
        "best_control_ngram_1_jsd_bits": best_one,
        "native_ngram_1_jsd_regression_bits": one_regression,
        "ngram_1_jsd_passed": (
            one_regression <= protocol.maximum_ngram_jsd_regression_bits_vs_best_control
        ),
        "best_control_ngram_3_jsd_bits": best_three,
        "native_ngram_3_jsd_regression_bits": three_regression,
        "ngram_3_jsd_passed": (
            three_regression <= protocol.maximum_ngram_jsd_regression_bits_vs_best_control
        ),
        "best_control_descriptor_energy_distance": best_descriptor,
        "native_descriptor_energy_distance_ratio": descriptor_ratio,
        "descriptor_ratio_passed": (
            descriptor_ratio <= protocol.maximum_descriptor_energy_distance_ratio_vs_best_control
        ),
        "native_ngram_3_relative_improvement": three_improvement,
        "native_descriptor_relative_improvement": descriptor_improvement,
        "minimum_one_distance_improvement_passed": (
            max(three_improvement, descriptor_improvement)
            >= protocol.minimum_improvement_in_3mer_jsd_or_descriptor_distance
        ),
    }
    relative_passed = all(
        comparisons[name]
        for name in (
            "common_funnel_yield_passed",
            "ngram_1_jsd_passed",
            "ngram_3_jsd_passed",
            "descriptor_ratio_passed",
            "minimum_one_distance_improvement_passed",
        )
    )
    # Infinite ratios/improvements cannot enter JSON.  Preserve a finite sentinel
    # whose associated Boolean comparison still fails closed.
    if not math.isfinite(descriptor_ratio):
        comparisons["native_descriptor_energy_distance_ratio"] = float(np.finfo(np.float64).max)
    if not math.isfinite(three_improvement):
        comparisons["native_ngram_3_relative_improvement"] = -float(np.finfo(np.float64).max)
    if not math.isfinite(descriptor_improvement):
        comparisons["native_descriptor_relative_improvement"] = -float(np.finfo(np.float64).max)
    return SamplingGateResult(
        per_seed=tuple(per_seed),
        method_medians=tuple(method_records),
        relative_comparisons=comparisons,
        absolute_passed=absolute_passed,
        relative_passed=relative_passed,
    )


@dataclass(frozen=True, slots=True)
class EvaluationDecision:
    status: str
    evidence: ProducerEvidenceChecks
    denoising: DenoisingGateResult
    sampling: SamplingGateResult

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "decision_precedence": [
                "evidence_invalid",
                "reproducible_no_go",
                "candidate_generator_only",
            ],
            "evidence": self.evidence.canonical_record(),
            "denoising": self.denoising.canonical_record(),
            "sampling": self.sampling.canonical_record(),
            "decision_status": self.status,
        }


def decide_evaluation(
    *,
    evidence: ProducerEvidenceChecks,
    denoising: DenoisingGateResult,
    sampling: SamplingGateResult,
    protocol: EvaluationProtocol,
) -> EvaluationDecision:
    """Apply the immutable evidence-invalid -> no-go -> candidate precedence."""

    if type(evidence) is not ProducerEvidenceChecks:
        raise TypeError("evidence must be ProducerEvidenceChecks")
    if type(denoising) is not DenoisingGateResult:
        raise TypeError("denoising must be DenoisingGateResult")
    if type(sampling) is not SamplingGateResult:
        raise TypeError("sampling must be SamplingGateResult")
    if not evidence.passed:
        status = protocol.evidence_invalid_status
    elif not denoising.passed or not sampling.passed:
        status = protocol.no_go_status
    else:
        status = protocol.candidate_status
    return EvaluationDecision(
        status=status,
        evidence=evidence,
        denoising=denoising,
        sampling=sampling,
    )


@dataclass(frozen=True, slots=True)
class EvaluationBundleInputs:
    """Strict raw-evidence boundary; no caller-supplied aggregate can pass a gate."""

    corpus: NativeDiffusionCorpus
    training: TrainingDistribution
    organizer_reference: OrganizerReference
    training_binding: TrainingBundleBinding
    corruptions: ValidationCorruptions
    token_statistics: ValidationTokenStatistics
    proposal_batches: tuple[ProposalBatch, ...]
    producer_checks: ProducerEvidenceChecks

    def validate(self, protocol: EvaluationProtocol) -> None:
        if type(self.corpus) is not NativeDiffusionCorpus:
            raise TypeError(
                "corpus must be a NativeDiffusionCorpus loaded through the strict loader"
            )
        if type(self.training) is not TrainingDistribution:
            raise TypeError("training must be a strict TrainingDistribution projection")
        if type(self.organizer_reference) is not OrganizerReference:
            raise TypeError("organizer_reference must be an OrganizerReference")
        if type(self.training_binding) is not TrainingBundleBinding:
            raise TypeError("training_binding must be a TrainingBundleBinding")
        if type(self.corruptions) is not ValidationCorruptions:
            raise TypeError("corruptions must be ValidationCorruptions")
        if type(self.token_statistics) is not ValidationTokenStatistics:
            raise TypeError("token_statistics must be ValidationTokenStatistics")
        if type(self.producer_checks) is not ProducerEvidenceChecks:
            raise TypeError("producer_checks must be ProducerEvidenceChecks")
        if self.corpus.sha256 != protocol.corpus_sha256:
            raise ValueError("accepted full corpus SHA-256 differs from the protocol")
        if _training_projection_digest(self.training) != protocol.training_projection_sha256:
            raise ValueError("training projection SHA-256 differs from the protocol")
        if self.organizer_reference.file_sha256 != protocol.organizer_reference_sha256:
            raise ValueError("organizer reference SHA-256 differs from the protocol")
        if len(self.organizer_reference.sequences) != protocol.organizer_reference_records:
            raise ValueError("organizer reference record census differs from the protocol")
        expected_projection = tuple(
            (item.sequence_id, item.sequence, item.sampling_weight)
            for item in self.corpus.train_rows
        )
        observed_projection = tuple(
            (item.sequence_id, item.sequence, item.sampling_weight) for item in self.training.rows
        )
        if observed_projection != expected_projection:
            raise ValueError("training projection does not exactly project the accepted corpus")
        length_mass: dict[int, list[float]] = defaultdict(list)
        for item in self.training.rows:
            length_mass[len(item.sequence)].append(item.sampling_weight)
        expected_lengths = tuple(sorted(length_mass))
        raw_length_probabilities = tuple(
            math.fsum(length_mass[length]) for length in expected_lengths
        )
        length_total = math.fsum(raw_length_probabilities)
        expected_length_probabilities = tuple(
            value / length_total for value in raw_length_probabilities
        )
        if (
            self.training.length_prior.lengths != expected_lengths
            or self.training.length_prior.probabilities != expected_length_probabilities
        ):
            raise ValueError("length prior is not exactly derived from the training projection")
        if self.corruptions.rows != self.corpus.validation_rows:
            raise ValueError("validation corruptions do not exactly bind the accepted fold-4 rows")
        ledger = self.corruptions.ledger
        if (
            ledger.evaluation_seed != protocol.evaluation_seed
            or ledger.levels != protocol.levels
            or ledger.replicates != protocol.replicates
            or ledger.validation_sequence_count != protocol.validation_sequences
            or len(ledger.cases) != protocol.corruption_cases
        ):
            raise ValueError("validation corruption ledger parameters differ from the protocol")
        rows_by_id = {item.sequence_id: item for item in self.corruptions.rows}
        schedule = CosineMaskSchedule(offset=0.008)
        mask_counts = {
            (len(row.sequence), level): int(
                schedule.mask_counts(
                    np.asarray([len(row.sequence)], dtype=np.int64),
                    level,
                    total_levels=protocol.levels,
                )[0]
            )
            for row in self.corruptions.rows
            for level in range(1, protocol.levels + 1)
        }
        for case in ledger.cases:
            row = rows_by_id[case.sequence_id]
            expected_seed = namespaced_seed(
                protocol.evaluation_seed,
                "validation",
                CONFIG_SHA256,
                row.sequence_id,
                case.level,
                case.replicate,
            )
            if (
                case.row_seed != expected_seed
                or case.mask_count != mask_counts[(len(row.sequence), case.level)]
                or case.homology_component_id != row.homology_component_id
                or case.union_component_id != row.union_component_id
                or case.sampling_weight != row.sampling_weight
            ):
                raise ValueError(
                    "validation corruption case differs from the accepted row or frozen derivation"
                )
        self.token_statistics.validate_against(self.corruptions)
        if self.training_binding.config_sha256 != protocol.config_sha256:
            raise ValueError("training bundle binding uses a different contract")
        if self.training_binding.git_commit == "":
            raise ValueError("training bundle binding has no Git identity")
        batches = self.proposal_batches
        expected_keys = tuple(
            (method, seed) for method in protocol.proposal_methods for seed in protocol.seeds
        )
        if (
            type(batches) is not tuple
            or any(type(item) is not ProposalBatch for item in batches)
            or tuple((item.method, item.seed) for item in batches) != expected_keys
        ):
            raise ValueError("proposal batches differ from the method-major cohort protocol")
        primary = self.training_binding.primary_by_seed
        if tuple(sorted(primary)) != protocol.seeds:
            raise ValueError("training bundle primaries differ from the protocol seeds")
        for seed in protocol.seeds:
            native = next(
                item
                for item in batches
                if item.method == protocol.proposal_methods[0] and item.seed == seed
            )
            if any(
                record.generator_binding_kind != "checkpoint_contract_logical_sha256"
                or record.generator_binding_sha256 != primary[seed].checkpoint_bound_logical_sha256
                for record in native.records
            ):
                raise ValueError("diffusion proposals do not bind their sealed seed checkpoint")
        control_bindings = {
            record.generator_binding_sha256
            for batch in batches
            if batch.method in protocol.proposal_methods[1:]
            for record in batch.records
        }
        if any(
            record.generator_binding_kind != "count_control_logical_sha256"
            for batch in batches
            if batch.method in protocol.proposal_methods[1:]
            for record in batch.records
        ):
            raise ValueError("count-control proposals use the wrong generator binding kind")
        if len(control_bindings) != 1:
            raise ValueError("count-control proposals do not share one fitted control identity")
        if len(self.training.rows) != protocol.training_sequences:
            raise ValueError("training projection row census differs from the protocol")
        control_suite = _validate_baseline_token_statistics(
            self.token_statistics,
            self.corruptions,
            training=self.training,
            protocol=protocol,
        )
        if control_bindings != {control_suite.logical_sha256}:
            raise ValueError("count-control proposals do not bind the recomputed baseline suite")
        for seed in protocol.seeds:
            seed_batches = [item for item in batches if item.seed == seed]
            if len({item.length_plan_sha256 for item in seed_batches}) != 1:
                raise ValueError("proposal methods do not share the per-seed length plan")
            plans = {
                tuple((record.ordinal, record.length) for record in item.records)
                for item in seed_batches
            }
            if len(plans) != 1:
                raise ValueError(
                    "proposal methods do not share exact per-seed ordinal/length pairs"
                )
            expected_lengths = self.training.length_prior.draw(
                root_seed=seed,
                draw_start=0,
                draw_count=protocol.raw_proposals_per_seed,
                namespace="proposal",
            )
            expected_plan = tuple(enumerate(expected_lengths))
            if plans != {expected_plan}:
                raise ValueError(
                    "proposal ordinal/length plan differs from the train-only frozen prior"
                )
            for batch in seed_batches:
                if batch.method == protocol.proposal_methods[0]:
                    continue
                expected_control = sample_count_generator_control_v0(
                    control_suite,
                    expected_lengths,
                    method=batch.method,
                    seed=seed,
                    ordinals=tuple(range(protocol.raw_proposals_per_seed)),
                    require_locked_count=False,
                )
                if tuple(item.sequence for item in batch.records) != tuple(
                    item.sequence for item in expected_control.candidates
                ):
                    raise ValueError(
                        "count-control proposals differ from deterministic regeneration"
                    )
        if any(len(item.records) != protocol.raw_proposals_per_seed for item in batches):
            raise ValueError("proposal batch census differs from the protocol")


def _proposal_ledger_bytes(
    batches: Sequence[ProposalBatch],
) -> bytes:
    return b"".join(
        canonical_json_bytes(record.canonical_record())
        for batch in batches
        for record in batch.records
    )


def _proposal_fasta_bytes(batches: Sequence[ProposalBatch]) -> bytes:
    output = bytearray()
    for batch in batches:
        for item in batch.records:
            header = FASTA_HEADER_GRAMMAR.format(
                method=item.method,
                seed=item.seed,
                ordinal=item.ordinal,
                sequence_id=item.sequence_id,
            )
            output.extend(header.encode("ascii"))
            output.extend(b"\n")
            output.extend(item.sequence.encode("ascii"))
            output.extend(b"\n")
    if not output:
        raise ValueError("raw proposal FASTA cannot be empty")
    return bytes(output)


def _length_plan_document(
    batches: Sequence[ProposalBatch],
    *,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in protocol.seeds:
        batch = next(
            item
            for item in batches
            if item.method == protocol.proposal_methods[0] and item.seed == seed
        )
        rows.append(
            {
                "seed": seed,
                "count": len(batch.records),
                "length_plan_sha256": batch.length_plan_sha256,
                "items": [
                    {"ordinal": item.ordinal, "length": item.length} for item in batch.records
                ],
            }
        )
    result = {
        "schema_version": protocol.schema_version,
        "config_sha256": protocol.config_sha256,
        "distribution": "train_only_component_weighted_empirical",
        "shared_across_methods": True,
        "seeds": rows,
    }
    if tuple(result) != LENGTH_PLAN_FIELDS:
        raise AssertionError("length-plan document schema changed")
    return result


def _baseline_metrics_document(
    results: Sequence[EvaluationResult],
    denoising: DenoisingGateResult,
    *,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    baselines = tuple(results)[: len(protocol.baseline_methods)]
    result = {
        "schema_version": protocol.schema_version,
        "config_sha256": protocol.config_sha256,
        "method_order": list(protocol.baseline_methods),
        "methods": [item.canonical_summary_record() for item in baselines],
        "strongest_method": denoising.strongest_baseline_method,
    }
    if tuple(result) != BASELINE_METRICS_FIELDS:
        raise AssertionError("baseline metrics document schema changed")
    return result


def _validation_metrics_document(
    results: Sequence[EvaluationResult],
    denoising: DenoisingGateResult,
    binding: TrainingBundleBinding,
    *,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    models = tuple(results)[len(protocol.baseline_methods) :]
    primaries = binding.primary_by_seed
    result = {
        "schema_version": protocol.schema_version,
        "config_sha256": protocol.config_sha256,
        "seeds": list(protocol.seeds),
        "method_order": list(protocol.token_stats_methods),
        "models": [
            {
                "seed": seed,
                "checkpoint_contract_logical_sha256": (
                    primaries[seed].checkpoint_bound_logical_sha256
                ),
                "metrics": model.canonical_summary_record(),
            }
            for seed, model in zip(protocol.seeds, models, strict=True)
        ],
        "bootstrap": denoising.bootstrap.canonical_record(),
        "timestep_bins": list(denoising.timestep_bins),
    }
    if tuple(result) != VALIDATION_METRICS_FIELDS:
        raise AssertionError("validation metrics document schema changed")
    return result


def _sampling_metrics_document(
    metrics: Sequence[SamplingMethodMetrics],
    gates: SamplingGateResult,
    *,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    result = {
        "schema_version": protocol.schema_version,
        "config_sha256": protocol.config_sha256,
        "seeds": list(protocol.seeds),
        "method_order": list(protocol.proposal_methods),
        "methods": [item.canonical_record() for item in metrics],
        "aggregates": list(gates.method_medians),
        "gates": gates.canonical_record(),
    }
    if tuple(result) != SAMPLING_METRICS_FIELDS:
        raise AssertionError("sampling metrics document schema changed")
    return result


@dataclass(frozen=True, slots=True)
class EvaluationBundlePayloads:
    """All deterministic pre-publication bytes plus the computed decision."""

    construction_token: object
    artifacts: tuple[tuple[str, bytes], ...]
    manifest: dict[str, object]
    decision: EvaluationDecision
    sealed_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.construction_token is not _PAYLOAD_CONSTRUCTION_TOKEN:
            raise TypeError("evaluation payloads must be constructed from raw bundle inputs")
        names = tuple(name for name, _ in self.artifacts)
        if names != EVALUATION_BUNDLE_FILES[:-1]:
            raise ValueError("evaluation payload inventory/order differs from the protocol")
        if any(type(payload) is not bytes or not payload for _, payload in self.artifacts):
            raise ValueError("evaluation artifact payloads must be non-empty bytes")
        _require_sha256(self.sealed_manifest_sha256, label="sealed manifest SHA-256")
        if hashlib.sha256(canonical_json_bytes(self.manifest)).hexdigest() != (
            self.sealed_manifest_sha256
        ):
            raise ValueError("evaluation manifest differs from its construction-time seal")

    @property
    def manifest_bytes(self) -> bytes:
        payload = canonical_json_bytes(self.manifest)
        if hashlib.sha256(payload).hexdigest() != self.sealed_manifest_sha256:
            raise ValueError("evaluation manifest mutated after raw-evidence construction")
        return payload

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    @property
    def logical_sha256(self) -> str:
        # Refuse to mint an identity for a document that no longer matches the
        # construction-time raw-evidence seal.
        _ = self.manifest_bytes
        return evaluation_bundle_logical_sha256(self.manifest)


def evaluation_bundle_logical_sha256(manifest: Mapping[str, object]) -> str:
    """Return the path-free domain-separated semantic identity of a bundle."""

    if type(manifest) is not dict:
        raise TypeError("evaluation manifest must be a plain dict")
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256()
    digest.update(_BUNDLE_DOMAIN)
    _framed_update(digest, payload)
    return digest.hexdigest()


def build_evaluation_bundle_payloads(
    inputs: EvaluationBundleInputs,
    *,
    protocol: EvaluationProtocol,
    contract_payload: bytes,
    require_locked_sampling_census: bool,
) -> EvaluationBundlePayloads:
    """Recompute all metrics/gates and construct every canonical bundle byte."""

    if type(inputs) is not EvaluationBundleInputs:
        raise TypeError("inputs must be EvaluationBundleInputs")
    if type(contract_payload) is not bytes or not contract_payload:
        raise ValueError("contract_payload must be non-empty bytes")
    if hashlib.sha256(contract_payload).hexdigest() != protocol.config_sha256:
        raise ValueError("contract payload SHA-256 differs from the evaluation protocol")
    if type(require_locked_sampling_census) is not bool:
        raise TypeError("require_locked_sampling_census must be boolean")
    inputs.validate(protocol)
    corruptions_npz = inputs.corruptions.npz_bytes(protocol)
    token_stats_npz = inputs.token_statistics.npz_bytes(protocol, inputs.corruptions)
    evaluation_results = evaluation_results_from_token_statistics(
        inputs.token_statistics,
        inputs.corruptions,
    )
    denoising = evaluate_denoising_gates(evaluation_results, protocol=protocol)
    sampling_metrics = compute_sampling_metrics(
        inputs.proposal_batches,
        training=inputs.training,
        reference=inputs.organizer_reference,
        protocol=protocol,
        require_locked_census=require_locked_sampling_census,
    )
    sampling = evaluate_sampling_gates(sampling_metrics, protocol=protocol)
    decision = decide_evaluation(
        evidence=inputs.producer_checks,
        denoising=denoising,
        sampling=sampling,
        protocol=protocol,
    )
    baseline_document = _baseline_metrics_document(
        evaluation_results,
        denoising,
        protocol=protocol,
    )
    validation_document = _validation_metrics_document(
        evaluation_results,
        denoising,
        inputs.training_binding,
        protocol=protocol,
    )
    sampling_document = _sampling_metrics_document(
        sampling_metrics,
        sampling,
        protocol=protocol,
    )
    length_document = _length_plan_document(inputs.proposal_batches, protocol=protocol)
    ledger_payload = _proposal_ledger_bytes(inputs.proposal_batches)
    fasta_payload = _proposal_fasta_bytes(inputs.proposal_batches)
    payloads = (
        ("contract.toml", contract_payload),
        ("training_bundle.sha256", inputs.training_binding.payload),
        ("validation_corruptions.npz", corruptions_npz),
        ("validation_token_stats.npz", token_stats_npz),
        ("validation_metrics.json", canonical_json_bytes(validation_document)),
        ("baseline_metrics.json", canonical_json_bytes(baseline_document)),
        ("length_plan.json", canonical_json_bytes(length_document)),
        ("raw_proposals.fasta", fasta_payload),
        ("candidate_ledger.jsonl", ledger_payload),
        ("sampling_metrics.json", canonical_json_bytes(sampling_document)),
    )
    artifact_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads}
    manifest = {
        "schema_version": protocol.schema_version,
        "artifact": protocol.artifact,
        "config_sha256": protocol.config_sha256,
        "git_commit": inputs.training_binding.git_commit,
        "seeds": list(protocol.seeds),
        "training_bundle_sha256": inputs.training_binding.sha256,
        "validation": {
            "accepted_corpus_sha256": protocol.corpus_sha256,
            "validation_fold": 4,
            "validation_sequences": protocol.validation_sequences,
            "corruption_cases": protocol.corruption_cases,
            "selected_tokens": inputs.token_statistics.selected_count,
            "ledger_sha256": inputs.corruptions.ledger.sha256,
            "method_order": list(protocol.token_stats_methods),
            "corruptions_sha256": artifact_hashes["validation_corruptions.npz"],
            "token_stats_sha256": artifact_hashes["validation_token_stats.npz"],
            "metrics_sha256": artifact_hashes["validation_metrics.json"],
        },
        "baselines": {
            "method_order": list(protocol.baseline_methods),
            "strongest_method": denoising.strongest_baseline_method,
            "metrics_sha256": artifact_hashes["baseline_metrics.json"],
        },
        "sampling": {
            "method_order": list(protocol.proposal_methods),
            "raw_proposals_per_method_seed": protocol.raw_proposals_per_seed,
            "total_raw_proposals": sum(len(item.records) for item in inputs.proposal_batches),
            "length_plan_sha256": artifact_hashes["length_plan.json"],
            "fasta_sha256": artifact_hashes["raw_proposals.fasta"],
            "candidate_ledger_sha256": artifact_hashes["candidate_ledger.jsonl"],
            "metrics_sha256": artifact_hashes["sampling_metrics.json"],
            "training_projection_sha256": protocol.training_projection_sha256,
            "organizer_reference_sha256": protocol.organizer_reference_sha256,
            "organizer_reference_records": protocol.organizer_reference_records,
        },
        "gates": decision.canonical_record(),
        "decision_status": decision.status,
        "artifacts": artifact_hashes,
    }
    if tuple(manifest) != protocol.manifest_fields:
        raise AssertionError("evaluation manifest construction order/schema changed")
    canonical_json_bytes(manifest)
    return EvaluationBundlePayloads(
        construction_token=_PAYLOAD_CONSTRUCTION_TOKEN,
        artifacts=payloads,
        manifest=manifest,
        decision=decision,
        sealed_manifest_sha256=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class EvaluationBundleResult:
    """Operational handle returned only after immutable publication succeeds."""

    output_dir: Path
    decision_status: str
    manifest_sha256: str
    logical_sha256: str
    training_bundle_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path) or not self.output_dir.is_absolute():
            raise ValueError("published output_dir must be an absolute Path")
        if type(self.decision_status) is not str or not self.decision_status:
            raise ValueError("decision_status must be non-empty")
        for field_name in (
            "manifest_sha256",
            "logical_sha256",
            "training_bundle_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)


def _validated_new_output(path: str | Path) -> Path:
    output = Path(os.path.abspath(os.fspath(path)))
    if output.name in {"", ".", ".."}:
        raise ValueError("output directory must have a concrete final name")
    _reject_symlink_chain(output.parent)
    parent = output.parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(parent.st_mode):
        raise ValueError("output parent must be an existing real directory")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite or resume evaluation output: {output}")
    return output


def _write_new_bytes(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"artifact {path.name} must be non-empty bytes")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())


def _bundle_identity(root: Path) -> tuple[tuple[str, int, str], ...]:
    result: list[tuple[str, int, str]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        metadata = child.stat(follow_symlinks=False)
        if child.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("evaluation bundle must contain only regular non-symlink files")
        result.append((child.name, metadata.st_size, _file_sha256(child)))
    return tuple(result)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except FileNotFoundError:
        return
    shutil.rmtree(path)


def _prepare_staging(
    staging: Path,
) -> tuple[tuple[tuple[str, int, str], ...], tuple[int, int]]:
    if tuple(sorted(item.name for item in staging.iterdir())) != tuple(
        sorted(EVALUATION_BUNDLE_FILES)
    ):
        raise ValueError("evaluation staging inventory differs from the contract")
    for child in staging.iterdir():
        metadata = child.stat(follow_symlinks=False)
        if child.is_symlink() or stat.S_IMODE(metadata.st_mode) != 0o444:
            raise ValueError("evaluation artifacts must be regular and read-only")
        with child.open("rb") as handle:
            os.fsync(handle.fileno())
    identity = _bundle_identity(staging)
    metadata = staging.stat(follow_symlinks=False)
    root_identity = (metadata.st_dev, metadata.st_ino)
    _fsync_directory(staging)
    os.chmod(staging, 0o555)
    return identity, root_identity


def _renameat2_noreplace(staging: Path, output: Path) -> int:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
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


def _publish_by_links(
    staging: Path,
    output: Path,
    *,
    expected_identity: tuple[tuple[str, int, str], ...],
) -> None:
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        os.chmod(staging, 0o755)
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}") from error
    claim = output.stat(follow_symlinks=False)
    claim_identity = (claim.st_dev, claim.st_ino)
    expected = {name: (size, digest) for name, size, digest in expected_identity}
    try:
        if output.is_symlink() or not stat.S_ISDIR(claim.st_mode):
            raise RuntimeError("evaluation publication claim is not a real directory")
        if staging.stat(follow_symlinks=False).st_dev != claim.st_dev:
            raise RuntimeError("evaluation staging and output are on different filesystems")
        for name in EVALUATION_BUNDLE_FILES[:-1]:
            os.link(staging / name, output / name, follow_symlinks=False)
        # The manifest inode is linked with mode 000 and made visible/read-only
        # only after all other files and the directory entry are durable.
        manifest_source = staging / "manifest.json"
        os.chmod(manifest_source, 0o000)
        os.link(manifest_source, output / "manifest.json", follow_symlinks=False)
        if tuple(sorted(item.name for item in output.iterdir())) != tuple(
            sorted(EVALUATION_BUNDLE_FILES)
        ):
            raise RuntimeError("linked evaluation publication inventory changed")
        for name in EVALUATION_BUNDLE_FILES[:-1]:
            target = output / name
            target_stat = target.stat(follow_symlinks=False)
            size, digest = expected[name]
            if (
                not stat.S_ISREG(target_stat.st_mode)
                or stat.S_IMODE(target_stat.st_mode) != 0o444
                or target_stat.st_size != size
                or _file_sha256(target) != digest
            ):
                raise RuntimeError(f"linked evaluation artifact changed for {name}")
        manifest_target = output / "manifest.json"
        if (
            manifest_target.stat(follow_symlinks=False).st_ino
            != manifest_source.stat(follow_symlinks=False).st_ino
            or stat.S_IMODE(manifest_target.stat(follow_symlinks=False).st_mode) != 0
        ):
            raise RuntimeError("linked evaluation manifest changed before commit")
        current = output.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != claim_identity:
            raise RuntimeError("evaluation publication claim changed")
        _fsync_directory(output)
        os.chmod(output, 0o555)
        _fsync_directory(output.parent)
        # Final commit point for the portable hard-link fallback.
        os.chmod(manifest_target, 0o444)
        with manifest_target.open("rb") as handle:
            os.fsync(handle.fileno())
        if (
            stat.S_IMODE(manifest_target.stat(follow_symlinks=False).st_mode) != 0o444
            or _bundle_identity(output) != expected_identity
        ):
            raise RuntimeError("linked evaluation publication postcondition failed")
        _fsync_directory(output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            current = output.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == claim_identity and not output.is_symlink():
                os.chmod(output, 0o555)
        finally:
            os.chmod(staging, 0o755)
        raise
    try:
        os.chmod(staging, 0o755)
        _remove_staging(staging)
    except OSError:
        pass


def _publish_bundle_noreplace(staging: Path, output: Path) -> None:
    expected_identity, root_identity = _prepare_staging(staging)
    error_number = _renameat2_noreplace(staging, output)
    if error_number == 0:
        observed = output.stat(follow_symlinks=False)
        if (
            staging.exists()
            or output.is_symlink()
            or (observed.st_dev, observed.st_ino) != root_identity
            or stat.S_IMODE(observed.st_mode) != 0o555
            or _bundle_identity(output) != expected_identity
        ):
            raise RuntimeError("atomic evaluation publication postcondition failed")
        _fsync_directory(output.parent)
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        os.chmod(staging, 0o755)
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        _publish_by_links(staging, output, expected_identity=expected_identity)
        return
    os.chmod(staging, 0o755)
    raise OSError(error_number, os.strerror(error_number), str(output))


def publish_evaluation_bundle_payloads(
    payloads: EvaluationBundlePayloads,
    output_dir: str | Path,
    *,
    prepublish_check: Callable[[], None] | None = None,
) -> EvaluationBundleResult:
    """Atomically publish already recomputed bytes without overwrite or resume."""

    if type(payloads) is not EvaluationBundlePayloads:
        raise TypeError("payloads must be EvaluationBundlePayloads")
    if prepublish_check is not None and not callable(prepublish_check):
        raise TypeError("prepublish_check must be callable")
    manifest_payload = payloads.manifest_bytes
    observed_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.artifacts
    }
    if payloads.manifest.get("artifacts") != observed_hashes:
        raise ValueError("evaluation manifest no longer binds its artifact payloads")
    if payloads.manifest.get("decision_status") != payloads.decision.status:
        raise ValueError("evaluation manifest decision changed after raw-evidence construction")
    if tuple(payloads.manifest) != EVALUATION_MANIFEST_FIELDS:
        raise ValueError("evaluation manifest schema changed after construction")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    logical_sha256 = payloads.logical_sha256
    decision_status = payloads.decision.status
    training_bundle_sha256 = _require_sha256(
        payloads.manifest["training_bundle_sha256"],
        label="manifest training_bundle_sha256",
    )
    output = _validated_new_output(output_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    published = False
    try:
        for name, payload in payloads.artifacts:
            _write_new_bytes(staging / name, payload)
        # Intentionally created after every other file.
        _write_new_bytes(staging / "manifest.json", manifest_payload)
        if prepublish_check is not None:
            prepublish_check()
        if payloads.manifest_bytes != manifest_payload:
            raise ValueError("evaluation manifest changed during the prepublication check")
        _publish_bundle_noreplace(staging, output)
        published = True
    finally:
        if not published and staging.exists():
            _remove_staging(staging)
    return EvaluationBundleResult(
        output_dir=output,
        decision_status=decision_status,
        manifest_sha256=manifest_sha256,
        logical_sha256=logical_sha256,
        training_bundle_sha256=training_bundle_sha256,
    )


def publish_unconditional_v0_evaluation_bundle(
    *,
    output_dir: str | Path,
    contract_path: str | Path,
    accepted_corpus_path: str | Path,
    training_projection_path: str | Path,
    organizer_reference_path: str | Path,
    training_bundle_directories: Mapping[str, str | Path],
    corruptions: ValidationCorruptions,
    token_statistics: ValidationTokenStatistics,
    proposal_batches: tuple[ProposalBatch, ...],
    producer_checks: ProducerEvidenceChecks,
    expected_git_commit: str,
    prepublish_check: Callable[[], None] | None = None,
) -> EvaluationBundleResult:
    """Load every pinned input, recompute metrics/gates, and seal the v0 bundle.

    This boundary intentionally starts after model/control execution: callers
    provide only raw corruptions, per-token observations, and raw proposals—not
    aggregate metrics or gate results.  Loading checkpoints and producing those
    observations belongs to the GPU evaluation driver; publication independently
    recomputes every scientific quantity from the raw evidence supplied here.
    """

    commit = _require_git_commit(expected_git_commit)
    contract_source = Path(os.path.abspath(os.fspath(contract_path)))
    contract = load_unconditional_v0_contract(contract_source)
    contract_payload = _read_regular_bytes(contract_source)
    protocol = protocol_from_contract(contract)
    if hashlib.sha256(contract_payload).hexdigest() != protocol.config_sha256:
        raise ValueError("contract changed after strict parsing")
    corpus = load_native_diffusion_corpus(
        accepted_corpus_path,
        expected_sha256=protocol.corpus_sha256,
    )
    training = load_training_projection(
        training_projection_path,
        expected_sha256=protocol.training_projection_sha256,
        expected_rows=contract.input.expected_train_sequences,
    )
    reference = load_organizer_reference(
        organizer_reference_path,
        expected_sha256=protocol.organizer_reference_sha256,
        expected_records=protocol.organizer_reference_records,
    )
    binding = load_training_bundle_binding(
        training_bundle_directories,
        protocol=protocol,
        contract_payload=contract_payload,
        git_commit=commit,
    )
    inputs = EvaluationBundleInputs(
        corpus=corpus,
        training=training,
        organizer_reference=reference,
        training_binding=binding,
        corruptions=corruptions,
        token_statistics=token_statistics,
        proposal_batches=proposal_batches,
        producer_checks=producer_checks,
    )
    payloads = build_evaluation_bundle_payloads(
        inputs,
        protocol=protocol,
        contract_payload=contract_payload,
        require_locked_sampling_census=True,
    )
    return publish_evaluation_bundle_payloads(
        payloads,
        output_dir,
        prepublish_check=prepublish_check,
    )

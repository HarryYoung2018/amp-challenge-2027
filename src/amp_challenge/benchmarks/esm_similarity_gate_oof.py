"""Nested parser-v7 ESM residual benchmark with a frozen similarity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.benchmarks.ensemble_oof import (
    BASE_FOLD_ASSIGNMENT_POLICY,
    BASE_FOLD_POLICY,
    BASE_MODELS,
    BASE_PARSER_ID,
    Metadata,
    _audit_descriptor_fit,
    _partition_evidence,
    _read_base_oof,
    _read_sha256_manifest,
    _require_disjoint_homology,
    _require_manifest_entry,
    _require_two_classes,
    _set_digest,
    _sha256,
    _verify_base_folds,
    _verify_base_manifest,
    _verify_split_compatibility,
)
from amp_challenge.benchmarks.oracle_gate1 import Gate1Config, binary_metrics
from amp_challenge.models.oracle_baselines import DescriptorLogisticOracle
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import pairwise_identity_matrix

FloatArray = NDArray[np.float64]
GramClass = Literal["positive", "negative", "unknown"]

ESM_MODEL = "esm2_t6_8m_mean_logistic"
PROBABILITY_BLEND = "base_esm_probability_half_blend"
UNGATED_MODEL = "nested_ungated_logit_residual"
HARD_GATED_MODEL = "nested_hard_lt_0_4_logit_residual"
FIXED_SOFT_MODEL = "soft_similarity_gate_beta_0_5"
PRIMARY_MODEL = "nested_soft_similarity_gated_logit_residual"
METHODS = (
    *BASE_MODELS,
    ESM_MODEL,
    PROBABILITY_BLEND,
    UNGATED_MODEL,
    HARD_GATED_MODEL,
    FIXED_SOFT_MODEL,
    PRIMARY_MODEL,
)
PROMOTION_REFERENCE = "equal_weight_ensemble"
CONFIG_SHA256 = "fc973f3ec6a4b6b2682b8d892f94f711cfba5bb8b17f2b34bdab3c607177b7ba"
FROZEN_EMBEDDING_DIMENSION = 320
BASE_CONFIG_SHA256 = "b761051d9c46dfc1cf3e58525a89e71659b215fab8004eeaf983fdca0c52af54"
ACCEPTED_BASE_CHECKSUMS_SHA256 = "159be8b11b5814b1af8814877e6f9d4411b7cc2467d5512e577ac43b55ef54c1"
ACCEPTED_EMBEDDING_MATRIX_SHA256 = (
    "fefb3a7123a9fd9f90481862d04da59492b6e5511e6c620b250a4b09da4f3927"
)
ACCEPTED_EMBEDDING_INDEX_SHA256 = "941382918b7c861d1cb32f69af4192e87bdfd90c94ad88197abe125ae80a2b97"
ACCEPTED_EMBEDDING_MANIFEST_SHA256 = (
    "875695820e101aef21d3daf9dc97d63467593048fce1274ebf9ee7f3cf6cd867"
)
ACCEPTED_TRUST_RECEIPT_SHA256 = "ad9655f57cbb4d033c88fe38dfdfeffb56fd24274749c2caded890667e63e850"
ACCEPTED_BASE_EVIDENCE = {
    "examples": 5446,
    "positive_examples": 3843,
    "negative_examples": 1603,
    "unique_sequences": 1084,
    "homology_clusters": 581,
    "sequence_ids_sha256": "de6dbed6ad2a8f5a1ab38c215c92d7fcc3779ded36c7374ddc0ed7b08f28073e",
    "example_label_fold_cluster_sha256": (
        "56983047f070b1fb2cf4b073b4e0a91d52ea720b198931cd283c20c77fbc9b60"
    ),
}
ACCEPTED_EMBEDDING_EVIDENCE = {
    "required_sequences": 1084,
    "embedding_sequences": 1086,
    "required_sequence_ids_sha256": (
        "de6dbed6ad2a8f5a1ab38c215c92d7fcc3779ded36c7374ddc0ed7b08f28073e"
    ),
    "embedding_sequence_ids_sha256": (
        "aa25e92c1e35f763cee5e9a3b3cfdbba2788a5996a522685dccd56af612adc81"
    ),
    "unused_sequence_ids_sha256": (
        "e4217326d5bb38fc9ef1bfd1ff89cf3f82fbc80fb4380ee099368da2de2254eb"
    ),
}
ACCEPTED_UNUSED_SEQUENCE_IDS = (
    "004ac10e14ff6511cce8ff4149dee5eb79094e9b04013a9c0ab2845ff7b88e5b",
    "098401b4a81824aa223e387fc207c9a79c13cf2e60695fd5750f324088c7a873",
)
GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MODEL_RELATIVE_PATH = "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt"
CONTACT_RELATIVE_PATH = "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt"
LOCK_RELATIVE_PATH = "source/uv.lock"
PREDICTION_SCHEMA = (
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
    "gate_policy",
    "gate_value",
    "beta",
    "probability",
)
PRODUCTION_BLOCKERS = (
    "similarity_gate_hypothesis_was_motivated_by_posthoc_v4_slices_with_overlapping_labels",
    "evaluation_labels_previously_inspected_and_resplit",
    "untouched_external_panel_confirmation_required",
    "esm2_ur50d_exact_sequence_membership_not_established",
)


@dataclass(frozen=True, slots=True)
class SimilarityGateConfig:
    path: Path
    methods: tuple[str, ...]
    promotion_reference: str
    promotion_candidate: str
    probability_clip: float
    beta_grid: tuple[float, ...]
    hard_gate_threshold: float
    soft_gate_full_below: float
    soft_gate_zero_at: float
    fixed_soft_beta: float
    selection_objective: str
    selection_tie_break: str
    selection_tie_tolerance: float
    embedding_model: str
    embedding_dimension: int
    representation_layer: int
    context_features: str
    esm_l2: float
    esm_prior_strength: float
    esm_max_iterations: int
    esm_tolerance: float
    folds: int
    homology_identity_threshold: float
    similarity_bin_edges: tuple[float, ...]
    calibration_bins: int
    bootstrap_replicates: int
    seed: int
    expected_input_fasta_sha256: str
    expected_input_fasta_manifest_sha256: str
    expected_model_checkpoint_sha256: str
    expected_contact_regression_sha256: str
    expected_environment_lock_sha256: str
    expected_trust_manifest_sha256: str
    expected_embedding_worker_sha256: str
    expected_embedding_python_version: str
    expected_embedding_torch_version: str
    expected_embedding_fair_esm_version: str
    expected_embedding_numpy_version: str
    expected_embedding_cuda_runtime: str
    expected_embedding_cudnn_version: int
    expected_embedding_device_type: str
    expected_embedding_batch_size: int
    expected_embedding_seed: int
    promotion_auc_delta_lower_minimum: float
    promotion_brier_delta_upper_maximum: float
    promotion_log_loss_delta_upper_maximum: float


@dataclass(frozen=True, slots=True)
class EsmHead:
    outer_fold: int
    inner_fold: int | None
    training_examples: int
    training_sequences: int
    training_positives: int
    training_negatives: int
    training_example_ids_sha256: str
    training_assignment_sha256: str
    iterations: int
    converged: bool
    embedding_mean: tuple[float, ...]
    embedding_scale: tuple[float, ...]
    coefficients: tuple[float, ...]


def _hash(raw: object, name: str) -> str:
    value = str(raw)
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"config {name} must be a lowercase SHA-256")
    return value


def load_config(path: str | Path) -> SimilarityGateConfig:
    config_path = Path(path).resolve(strict=True)
    if _sha256(config_path) != CONFIG_SHA256:
        raise ValueError("ESM similarity-gate config differs from the frozen configuration")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {field for field in SimilarityGateConfig.__dataclass_fields__ if field != "path"}
    allowed.add("schema_version")
    if set(raw) != allowed:
        raise ValueError(
            "ESM similarity-gate config keys differ from the frozen schema: "
            f"{sorted(set(raw) ^ allowed)}"
        )
    if raw["schema_version"] != 1:
        raise ValueError("ESM similarity-gate config schema_version must be 1")
    methods = tuple(str(value) for value in cast(list[object], raw["methods"]))
    beta_grid = tuple(float(value) for value in cast(list[object], raw["beta_grid"]))
    edges = tuple(float(value) for value in cast(list[object], raw["similarity_bin_edges"]))
    if methods != METHODS:
        raise ValueError("methods differ from the frozen nine-method diagnostic panel")
    if (
        str(raw["promotion_reference"]) != PROMOTION_REFERENCE
        or str(raw["promotion_candidate"]) != PRIMARY_MODEL
    ):
        raise ValueError("the frozen reference/sole promotion candidate changed")
    if beta_grid != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("beta_grid differs from the frozen grid")
    exact = {
        "hard_gate_threshold": 0.4,
        "soft_gate_full_below": 0.4,
        "soft_gate_zero_at": 0.6,
        "fixed_soft_beta": 0.5,
        "homology_identity_threshold": 0.8,
    }
    if any(float(raw[name]) != value for name, value in exact.items()):
        raise ValueError("similarity thresholds differ from the frozen gate contract")
    if edges != (0.0, 0.4, 0.6, 0.8):
        raise ValueError("similarity_bin_edges differ from the frozen strata")
    if int(raw["folds"]) != 5 or int(raw["bootstrap_replicates"]) != 500:
        raise ValueError("fold or bootstrap count differs from the frozen contract")
    if str(raw["selection_objective"]) != "homology_cluster_equal_weighted_log_loss":
        raise ValueError("selection_objective differs from the frozen objective")
    if str(raw["selection_tie_break"]) != "lowest_beta":
        raise ValueError("selection_tie_break must be lowest_beta")
    if str(raw["context_features"]) != "gram_one_hot":
        raise ValueError("context_features must be gram_one_hot")
    positive_names = (
        "probability_clip",
        "esm_l2",
        "esm_tolerance",
    )
    if any(not math.isfinite(float(raw[name])) or float(raw[name]) <= 0 for name in positive_names):
        raise ValueError("positive floating-point config fields must be finite and positive")
    if float(raw["probability_clip"]) >= 0.5:
        raise ValueError("probability_clip must be below 0.5")
    if not math.isfinite(float(raw["esm_prior_strength"])) or float(raw["esm_prior_strength"]) < 0:
        raise ValueError("esm_prior_strength must be finite and non-negative")
    integer_positive = (
        "embedding_dimension",
        "esm_max_iterations",
        "calibration_bins",
        "expected_embedding_cudnn_version",
        "expected_embedding_batch_size",
    )
    if any(int(raw[name]) < 1 for name in integer_positive):
        raise ValueError("positive integer config fields must be positive")
    hash_names = (
        "expected_input_fasta_sha256",
        "expected_input_fasta_manifest_sha256",
        "expected_model_checkpoint_sha256",
        "expected_contact_regression_sha256",
        "expected_environment_lock_sha256",
        "expected_trust_manifest_sha256",
        "expected_embedding_worker_sha256",
    )
    values = dict(raw)
    for name in hash_names:
        values[name] = _hash(raw[name], name)
    del values["schema_version"]
    values["path"] = config_path
    values["methods"] = methods
    values["beta_grid"] = beta_grid
    values["similarity_bin_edges"] = edges
    config = SimilarityGateConfig(**values)
    exact_values: dict[str, object] = {
        "probability_clip": 1e-6,
        "selection_tie_tolerance": 1e-12,
        "embedding_model": "esm2_t6_8M_UR50D",
        "embedding_dimension": FROZEN_EMBEDDING_DIMENSION,
        "representation_layer": 6,
        "context_features": "gram_one_hot",
        "esm_l2": 0.1,
        "esm_prior_strength": 2.0,
        "esm_max_iterations": 100,
        "esm_tolerance": 1e-9,
        "folds": 5,
        "homology_identity_threshold": 0.8,
        "similarity_bin_edges": (0.0, 0.4, 0.6, 0.8),
        "calibration_bins": 10,
        "bootstrap_replicates": 500,
        "seed": 20260902,
        "expected_input_fasta_sha256": (
            "1c01e87608042b8b9534c43eed3309e89b7a1bc2b639254ac95fb41ece7a9d7e"
        ),
        "expected_input_fasta_manifest_sha256": (
            "eab831e5de718363ebdaa7952b50e25fcc53a7eee49184ae99f057fc8afb08ec"
        ),
        "expected_model_checkpoint_sha256": (
            "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928"
        ),
        "expected_contact_regression_sha256": (
            "8f7a4557d57713b97ba0e484303007efb7230d25299c0ac47a0a1b12a87bbb9d"
        ),
        "expected_environment_lock_sha256": (
            "aaf37baa3adf5070dd3090c43527daa2696741a7e3057c72c78d51a80af9e234"
        ),
        "expected_trust_manifest_sha256": (
            "5e62db4fd3241e1fa6bbd83cc0f26e299c6d16e9fa1e449304fab2e4c98fd9ff"
        ),
        "expected_embedding_worker_sha256": (
            "34fdb77e87aaa44044960655259cc60fb20b165bfb87c38ba19bee21565f2cba"
        ),
        "expected_embedding_python_version": "3.10.19",
        "expected_embedding_torch_version": "2.5.1+cu121",
        "expected_embedding_fair_esm_version": "2.0.0",
        "expected_embedding_numpy_version": "2.2.6",
        "expected_embedding_cuda_runtime": "12.1",
        "expected_embedding_cudnn_version": 90100,
        "expected_embedding_device_type": "cuda",
        "expected_embedding_batch_size": 128,
        "expected_embedding_seed": 20260902,
        "promotion_auc_delta_lower_minimum": 0.0,
        "promotion_brier_delta_upper_maximum": 0.0,
        "promotion_log_loss_delta_upper_maximum": 0.0,
    }
    changed = [name for name, expected in exact_values.items() if getattr(config, name) != expected]
    if changed:
        raise ValueError(f"frozen scientific constants changed: {changed}")
    return config


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _assignment_digest(rows: Sequence[Metadata]) -> str:
    payload = "".join(
        f"{item.example_id}\t{item.sequence_id}\t{item.label}\t{item.fold}\t{item.cluster_id}\n"
        for item in sorted(rows, key=lambda value: value.example_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _similarity_digest(rows: Sequence[tuple[str, float]]) -> str:
    payload = "".join(
        f"{example_id}\t{float(value).hex()}\n" for example_id, value in sorted(rows)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _float_matrix_digest(matrix: NDArray[np.generic], dtype: str) -> str:
    values = np.ascontiguousarray(matrix, dtype=np.dtype(dtype))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _verify_trust_receipt(path: Path, config: SimilarityGateConfig) -> dict[str, object]:
    payload = path.read_text(encoding="utf-8")
    document = json.loads(payload)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if payload != canonical:
        raise ValueError("embedding trust receipt is not canonical JSON")
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "receipt_type",
        "component",
        "bundle_root",
        "integration",
        "source",
        "trust_manifest",
        "artifacts",
    }:
        raise ValueError("embedding trust receipt keys differ from the required schema")
    root = document.get("bundle_root")
    source = document.get("source")
    trust_manifest = document.get("trust_manifest")
    if (
        document.get("schema_version") != 1
        or document.get("receipt_type") != "ampdiffusion_bundle_verification"
        or document.get("component") != "generation"
        or not isinstance(root, str)
        or not PurePosixPath(root).is_absolute()
        or PurePosixPath(root).as_posix() != root
        or not isinstance(document.get("integration"), str)
        or not document["integration"]
        or not isinstance(source, dict)
        or set(source) != {"repository", "commit", "license"}
        or not isinstance(source.get("repository"), str)
        or not str(source["repository"]).startswith("https://")
        or GIT_SHA1_RE.fullmatch(str(source.get("commit", ""))) is None
        or not isinstance(source.get("license"), str)
        or not source["license"]
        or not isinstance(trust_manifest, dict)
        or set(trust_manifest) != {"filename", "sha256", "size"}
        or trust_manifest.get("filename") != "artifacts.toml"
        or trust_manifest.get("sha256") != config.expected_trust_manifest_sha256
        or type(trust_manifest.get("size")) is not int
        or int(trust_manifest["size"]) <= 0
    ):
        raise ValueError("embedding trust receipt identity/provenance is invalid")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("embedding trust receipt has no artifacts")
    observed: dict[str, tuple[str, str]] = {}
    ordered: list[str] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size", "role"}:
            raise ValueError(f"embedding trust receipt artifact {index} has invalid schema")
        relative = str(item["path"])
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in observed
            or SHA256_RE.fullmatch(str(item["sha256"])) is None
            or type(item["size"]) is not int
            or int(item["size"]) <= 0
            or not isinstance(item["role"], str)
            or not item["role"]
        ):
            raise ValueError(f"embedding trust receipt artifact {index} is invalid")
        ordered.append(relative)
        observed[relative] = (str(item["sha256"]), str(item["role"]))
    if ordered != sorted(ordered):
        raise ValueError("embedding trust receipt artifacts are not path-sorted")
    expected = {
        MODEL_RELATIVE_PATH: (config.expected_model_checkpoint_sha256, "pickle_checkpoint"),
        CONTACT_RELATIVE_PATH: (
            config.expected_contact_regression_sha256,
            "pickle_checkpoint",
        ),
        LOCK_RELATIVE_PATH: (config.expected_environment_lock_sha256, "environment_lock"),
    }
    if any(observed.get(name) != value for name, value in expected.items()):
        raise ValueError("embedding trust receipt does not bind the accepted ESM artifacts")
    return cast(dict[str, object], document)


def _verify_embedding_manifest(
    matrix_path: Path,
    index_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    config: SimilarityGateConfig,
) -> dict[str, object]:
    receipt = _verify_trust_receipt(receipt_path, config)
    document = _load_json(manifest_path, "embedding manifest")
    outputs = document.get("outputs")
    trust = document.get("trust")
    determinism = document.get("determinism")
    runtime = document.get("runtime")
    if (
        document.get("schema_version") != 1
        or document.get("benchmark") != "esm2_embedding_extraction"
        or document.get("model") != config.embedding_model
        or document.get("representation_layer") != config.representation_layer
        or document.get("embedding_dimension") != config.embedding_dimension
        or document.get("pooling")
        != "arithmetic mean over residue representations; BOS/EOS/padding excluded"
        or document.get("dtype") != "float32"
        or document.get("ordering") != "input FASTA order (ascending sequence_id)"
        or document.get("input_fasta_sha256") != config.expected_input_fasta_sha256
        or document.get("input_fasta_manifest_sha256")
        != config.expected_input_fasta_manifest_sha256
        or not isinstance(outputs, dict)
        or set(outputs) != {matrix_path.name, index_path.name}
        or outputs.get(matrix_path.name) != _sha256(matrix_path)
        or outputs.get(index_path.name) != _sha256(index_path)
        or not isinstance(trust, dict)
        or trust.get("trust_manifest_sha256") != config.expected_trust_manifest_sha256
        or trust.get("verification_receipt_sha256") != _sha256(receipt_path)
        or trust.get("bundle_root") != receipt.get("bundle_root")
        or trust.get("integration") != receipt.get("integration")
        or trust.get("source_commit") != cast(dict[str, object], receipt["source"]).get("commit")
        or document.get("worker_sha256") != config.expected_embedding_worker_sha256
        or not isinstance(determinism, dict)
        or determinism.get("seed") != config.expected_embedding_seed
        or determinism.get("batch_size") != config.expected_embedding_batch_size
        or determinism.get("torch_deterministic_algorithms") is not True
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("cudnn_deterministic") is not True
        or determinism.get("tf32") is not False
        or determinism.get("cublas_workspace_config") != ":4096:8"
        or not isinstance(runtime, dict)
        or runtime.get("python") != config.expected_embedding_python_version
        or runtime.get("torch") != config.expected_embedding_torch_version
        or runtime.get("fair_esm") != config.expected_embedding_fair_esm_version
        or runtime.get("numpy") != config.expected_embedding_numpy_version
        or runtime.get("cuda_runtime") != config.expected_embedding_cuda_runtime
        or runtime.get("cudnn") != config.expected_embedding_cudnn_version
        or runtime.get("device_type") != config.expected_embedding_device_type
        or not isinstance(runtime.get("device_name"), str)
        or not runtime["device_name"]
        or not isinstance(runtime.get("device_capability"), list)
        or len(runtime["device_capability"]) != 2
        or not all(type(value) is int and value >= 0 for value in runtime["device_capability"])
    ):
        raise ValueError("embedding manifest/input/output chain mismatch")
    trusted = trust.get("artifacts") if isinstance(trust, dict) else None
    expected_trusted = {
        "model_checkpoint": config.expected_model_checkpoint_sha256,
        "contact_regression_checkpoint": config.expected_contact_regression_sha256,
        "environment_lock": config.expected_environment_lock_sha256,
    }
    if not isinstance(trusted, dict):
        raise ValueError("embedding manifest has no trusted artifact table")
    for name, expected_hash in expected_trusted.items():
        item = trusted.get(name)
        if not isinstance(item, dict) or item.get("sha256") != expected_hash:
            raise ValueError(f"embedding manifest trusted {name} checksum mismatch")
    return document


def _read_embeddings(
    matrix_path: Path,
    index_path: Path,
    *,
    metadata: Mapping[str, Metadata],
    config: SimilarityGateConfig,
) -> tuple[dict[str, FloatArray], dict[str, object]]:
    matrix = np.load(matrix_path, allow_pickle=False)
    if (
        matrix.dtype != np.float32
        or matrix.ndim != 2
        or matrix.shape[1] != config.embedding_dimension
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("embedding matrix has the wrong dtype, shape, or finite-value contract")
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row_index", "sequence_id", "sequence", "length"]:
            raise ValueError("embedding index fields differ from the required schema")
        rows = list(reader)
    if len(rows) != matrix.shape[0]:
        raise ValueError("embedding index and matrix row counts differ")
    sequences: dict[str, str] = {}
    row_by_sequence: dict[str, int] = {}
    ordered_ids: list[str] = []
    for expected_index, row in enumerate(rows):
        try:
            row_index = int(row["row_index"])
            sequence = canonicalize_sequence(row["sequence"])
            sequence_id = row["sequence_id"]
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"embedding index row {expected_index + 2} is invalid") from error
        if (
            row_index != expected_index
            or sequence_id != canonical_sequence_id(sequence)
            or length != len(sequence)
            or sequence_id in sequences
        ):
            raise ValueError("embedding index sequence/row identity is invalid")
        ordered_ids.append(sequence_id)
        sequences[sequence_id] = sequence
        row_by_sequence[sequence_id] = expected_index
    if ordered_ids != sorted(ordered_ids):
        raise ValueError("embedding index is not in ascending sequence_id order")
    required = {item.sequence_id: item.sequence for item in metadata.values()}
    if len(required) != len({item.sequence for item in metadata.values()}):
        raise ValueError("base metadata sequence IDs are not one-to-one with sequences")
    missing = tuple(sorted(set(required) - set(sequences)))
    mismatched = tuple(
        sorted(
            sequence_id
            for sequence_id in set(required) & set(sequences)
            if required[sequence_id] != sequences[sequence_id]
        )
    )
    unused = tuple(sorted(set(sequences) - set(required)))
    if missing or mismatched:
        raise ValueError("accepted embeddings do not fully and exactly cover parser-v7 sequences")
    full_sequence_to_row_payload = "".join(
        f"{sequence_id}\t{row_by_sequence[sequence_id]}\n" for sequence_id in sorted(sequences)
    ).encode("ascii")
    required_sequence_to_row_payload = "".join(
        f"{sequence_id}\t{row_by_sequence[sequence_id]}\n" for sequence_id in sorted(required)
    ).encode("ascii")
    required_matrix = np.asarray(
        [matrix[row_by_sequence[sequence_id]] for sequence_id in sorted(required)],
        dtype=np.float32,
    )
    coverage: dict[str, object] = {
        "schema_version": 1,
        "required_sequences": len(required),
        "embedding_sequences": len(sequences),
        "missing_sequence_ids": list(missing),
        "mismatched_sequence_ids": list(mismatched),
        "unused_sequence_ids": list(unused),
        "required_sequence_ids_sha256": _set_digest(set(required)),
        "embedding_sequence_ids_sha256": _set_digest(set(sequences)),
        "unused_sequence_ids_sha256": _set_digest(set(unused)),
        "full_sequence_to_row_sha256": hashlib.sha256(full_sequence_to_row_payload).hexdigest(),
        "required_sequence_to_row_sha256": hashlib.sha256(
            required_sequence_to_row_payload
        ).hexdigest(),
        "required_vectors_sha256": _float_matrix_digest(required_matrix, "<f4"),
        "vector_digest_encoding": (
            "required vectors ordered by ascending sequence_id and encoded as contiguous "
            "little-endian float32 bytes"
        ),
        "sequence_to_row_digest_encoding": (
            "full or required sequence_id and original zero-based row_index pairs, "
            "tab-separated and ascending by sequence_id, with a terminal LF"
        ),
        "embedding_matrix_sha256": _sha256(matrix_path),
        "embedding_index_sha256": _sha256(index_path),
    }
    expected = {**ACCEPTED_EMBEDDING_EVIDENCE}
    if any(coverage.get(key) != value for key, value in expected.items()):
        raise ValueError("embedding coverage census/digests differ from the accepted v7 audit")
    if unused != ACCEPTED_UNUSED_SEQUENCE_IDS:
        raise ValueError("embedding index has an unexpected parser-v7 superset remainder")
    embeddings = {
        sequence_id: matrix[row_by_sequence[sequence_id]].astype(np.float64)
        for sequence_id in required
    }
    return embeddings, coverage


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _logit(value: float, clip: float) -> float:
    probability = float(np.clip(value, clip, 1.0 - clip))
    return math.log(probability / (1.0 - probability))


def _esm_design(
    rows: Sequence[Metadata],
    embeddings: Mapping[str, FloatArray],
    mean: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    continuous = np.asarray([embeddings[item.sequence_id] for item in rows], dtype=np.float64)
    standardized = (continuous - mean) / scale
    gram = np.zeros((len(rows), 3), dtype=np.float64)
    gram_index = {"positive": 0, "negative": 1, "unknown": 2}
    for index, item in enumerate(rows):
        gram[index, gram_index[item.gram]] = 1.0
    design = np.column_stack((np.ones(len(rows), dtype=np.float64), standardized, gram))
    if not np.all(np.isfinite(design)):
        raise ValueError("ESM head design matrix contains non-finite values")
    return design


def _fit_esm_head(
    training: Sequence[Metadata],
    *,
    outer_fold: int,
    inner_fold: int | None,
    embeddings: Mapping[str, FloatArray],
    config: SimilarityGateConfig,
) -> tuple[EsmHead, FloatArray, FloatArray, FloatArray]:
    _require_two_classes(training, description="ESM head training partition")
    raw = np.asarray([embeddings[item.sequence_id] for item in training], dtype=np.float64)
    if raw.shape != (len(training), config.embedding_dimension) or not np.all(np.isfinite(raw)):
        raise ValueError("ESM head training embeddings are invalid")
    mean = np.mean(raw, axis=0, dtype=np.float64)
    scale = np.std(raw, axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    design = _esm_design(training, embeddings, mean, scale)
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    prior = float(
        (np.sum(labels) + 0.5 * config.esm_prior_strength)
        / (len(labels) + config.esm_prior_strength)
    )
    if not 0 < prior < 1:
        raise ValueError("ESM head training prior is degenerate")
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(design.shape[1], config.esm_l2, dtype=np.float64)
    penalty[0] = 0.0
    converged = False
    iterations = 0
    for iteration in range(1, config.esm_max_iterations + 1):
        probability = _sigmoid(design @ coefficients)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - labels) / len(labels) + penalty * coefficients
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError("ESM head produced a non-finite Newton step")
        coefficients -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("ESM head produced non-finite coefficients")
        if float(np.max(np.abs(step))) <= config.esm_tolerance:
            converged = True
            break
    if not converged:
        scope = f"outer {outer_fold}" + ("" if inner_fold is None else f"/inner {inner_fold}")
        raise ValueError(f"{scope} ESM head exhausted its iteration budget before convergence")
    evidence = _partition_evidence(training)
    model = EsmHead(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        training_examples=len(training),
        training_sequences=len({item.sequence_id for item in training}),
        training_positives=int(np.sum(labels)),
        training_negatives=len(training) - int(np.sum(labels)),
        training_example_ids_sha256=str(evidence["example_ids_sha256"]),
        training_assignment_sha256=str(evidence["example_label_fold_cluster_sha256"]),
        iterations=iterations,
        converged=True,
        embedding_mean=tuple(float(value) for value in mean),
        embedding_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
    )
    return model, mean, scale, coefficients


def _predict_esm(
    rows: Sequence[Metadata],
    embeddings: Mapping[str, FloatArray],
    mean: FloatArray,
    scale: FloatArray,
    coefficients: FloatArray,
    clip: float,
) -> FloatArray:
    values = _sigmoid(_esm_design(rows, embeddings, mean, scale) @ coefficients)
    values = np.clip(values, clip, 1.0 - clip)
    if values.shape != (len(rows),) or not np.all(np.isfinite(values)):
        raise ValueError("ESM head predictions are invalid")
    return values


def _predict_knn_from_identity(
    training: Sequence[Metadata],
    testing: Sequence[Metadata],
    identity: FloatArray,
    sequence_index: Mapping[str, int],
    base_config: Gate1Config,
) -> FloatArray:
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    output: list[float] = []
    for query in testing:
        candidates = [index for index, item in enumerate(training) if item.strain == query.strain]
        if not candidates:
            candidates = [index for index, item in enumerate(training) if item.gram == query.gram]
        if not candidates:
            candidates = list(range(len(training)))
        context_labels = labels[candidates]
        context_prior = float(
            (np.sum(context_labels) + 0.5 * base_config.knn.prior_strength)
            / (len(context_labels) + base_config.knn.prior_strength)
        )
        query_index = sequence_index[query.sequence_id]
        ordered = sorted(
            (
                (index, float(identity[query_index, sequence_index[training[index].sequence_id]]))
                for index in candidates
            ),
            key=lambda pair: (
                -pair[1],
                training[pair[0]].sequence,
                training[pair[0]].strain,
                pair[0],
            ),
        )[: base_config.knn.neighbors]
        weights = np.asarray(
            [
                max(value**base_config.knn.similarity_power, base_config.knn.minimum_weight)
                for _, value in ordered
            ],
            dtype=np.float64,
        )
        neighbor_labels = np.asarray([labels[index] for index, _ in ordered], dtype=np.float64)
        denominator = float(np.sum(weights)) + base_config.knn.prior_strength
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError("homology kNN produced an invalid weight denominator")
        probability = (
            float(weights @ neighbor_labels) + base_config.knn.prior_strength * context_prior
        ) / denominator
        output.append(float(np.clip(probability, 1e-6, 1.0 - 1e-6)))
    values = np.asarray(output, dtype=np.float64)
    if values.shape != (len(testing),) or not np.all(np.isfinite(values)):
        raise ValueError("homology kNN produced invalid predictions")
    return values


def _max_similarity(
    testing: Sequence[Metadata],
    training: Sequence[Metadata],
    identity: FloatArray,
    sequence_index: Mapping[str, int],
) -> FloatArray:
    training_indices = np.asarray(
        sorted({sequence_index[item.sequence_id] for item in training}), dtype=np.int64
    )
    if training_indices.size == 0:
        raise ValueError("cannot compute maximum identity against an empty training set")
    values = np.asarray(
        [np.max(identity[sequence_index[item.sequence_id], training_indices]) for item in testing],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values >= 0.8):
        raise ValueError("split-local maximum identities violate the homology holdout")
    return values


def _soft_gate(similarity: float, config: SimilarityGateConfig) -> float:
    if similarity < config.soft_gate_full_below:
        return 1.0
    if similarity < config.soft_gate_zero_at:
        return (config.soft_gate_zero_at - similarity) / (
            config.soft_gate_zero_at - config.soft_gate_full_below
        )
    return 0.0


def _hard_gate(similarity: float, config: SimilarityGateConfig) -> float:
    return float(similarity < config.hard_gate_threshold)


def _residual_probability(
    base_probability: float,
    esm_probability: float,
    *,
    beta: float,
    gate: float,
    clip: float,
) -> float:
    if not all(math.isfinite(value) for value in (base_probability, esm_probability, beta, gate)):
        raise ValueError("residual blend received a non-finite value")
    if not 0 <= beta <= 1 or not 0 <= gate <= 1:
        raise ValueError("residual beta and gate must be in [0, 1]")
    if beta == 0.0 or gate == 0.0:
        return base_probability
    residual = _logit(esm_probability, clip) - _logit(base_probability, clip)
    value = float(
        _sigmoid(np.asarray([_logit(base_probability, clip) + beta * gate * residual]))[0]
    )
    if not math.isfinite(value):
        raise ValueError("residual blend produced a non-finite probability")
    return float(np.clip(value, clip, 1.0 - clip))


def _cluster_equal_log_loss(rows: Sequence[Mapping[str, object]], beta: float, gate: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        if gate == "ungated":
            gate_value = 1.0
        elif gate == "hard_lt_0_4":
            gate_value = float(item["hard_gate"])
        elif gate == "soft_0_4_to_0_6":
            gate_value = float(item["soft_gate"])
        else:  # pragma: no cover - callers are frozen
            raise ValueError(f"unknown gate {gate!r}")
        probability = _residual_probability(
            float(item["base_probability"]),
            float(item["esm_probability"]),
            beta=beta,
            gate=gate_value,
            clip=float(item["probability_clip"]),
        )
        label = int(item["label"])
        loss = -(label * math.log(probability) + (1 - label) * math.log1p(-probability))
        if not math.isfinite(loss):
            raise ValueError("gate-selection loss is non-finite")
        grouped[str(item["cluster_id"])].append(loss)
    if not grouped:
        raise ValueError("gate selection received no nested rows")
    value = float(np.mean([np.mean(losses) for _, losses in sorted(grouped.items())]))
    if not math.isfinite(value):
        raise ValueError("cluster-equal gate-selection log loss is non-finite")
    return value


def _select_beta(
    rows: Sequence[Mapping[str, object]],
    *,
    outer_fold: int,
    gate: str,
    config: SimilarityGateConfig,
) -> dict[str, object]:
    losses = [
        {
            "beta": beta,
            "homology_cluster_equal_weighted_log_loss": _cluster_equal_log_loss(rows, beta, gate),
        }
        for beta in config.beta_grid
    ]
    minimum_loss = min(float(item["homology_cluster_equal_weighted_log_loss"]) for item in losses)
    selected = next(
        item
        for item in losses
        if float(item["homology_cluster_equal_weighted_log_loss"])
        <= minimum_loss + config.selection_tie_tolerance
    )
    return {
        "outer_fold": outer_fold,
        "gate": gate,
        "objective": config.selection_objective,
        "tie_break": config.selection_tie_break,
        "tie_tolerance": config.selection_tie_tolerance,
        "nested_examples": len(rows),
        "nested_homology_clusters": len({str(item["cluster_id"]) for item in rows}),
        "losses": losses,
        "selected_beta": float(selected["beta"]),
    }


def _split_similarity_evidence(
    *,
    outer_fold: int,
    inner_fold: int | None,
    training: Sequence[Metadata],
    testing: Sequence[Metadata],
    values: FloatArray,
) -> dict[str, object]:
    if len(values) != len(testing):
        raise ValueError("split similarity vector and query rows differ in length")
    return {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "excluded_folds": (
            [outer_fold] if inner_fold is None else sorted((outer_fold, inner_fold))
        ),
        "training_examples": len(training),
        "training_sequences": len({item.sequence_id for item in training}),
        "query_examples": len(testing),
        "query_sequences": len({item.sequence_id for item in testing}),
        "training_example_ids_sha256": _set_digest({item.example_id for item in training}),
        "training_sequence_ids_sha256": _set_digest({item.sequence_id for item in training}),
        "query_example_ids_sha256": _set_digest({item.example_id for item in testing}),
        "query_sequence_ids_sha256": _set_digest({item.sequence_id for item in testing}),
        "maximum_identity_minimum": float(np.min(values)),
        "maximum_identity_maximum": float(np.max(values)),
        "maximum_identity_sha256": _similarity_digest(
            [(item.example_id, float(value)) for item, value in zip(testing, values, strict=True)]
        ),
        "identity_below_homology_threshold": bool(np.all(values < 0.8)),
        "train_query_examples_disjoint": True,
        "train_query_sequences_disjoint": True,
        "train_query_clusters_disjoint": True,
    }


def _nested_predictions(
    metadata: Mapping[str, Metadata],
    *,
    outer_fold: int,
    inner_fold: int,
    embeddings: Mapping[str, FloatArray],
    identity: FloatArray,
    sequence_index: Mapping[str, int],
    base_config: Gate1Config,
    config: SimilarityGateConfig,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    excluded = {outer_fold, inner_fold}
    training = sorted(
        (item for item in metadata.values() if item.fold not in excluded),
        key=lambda item: item.example_id,
    )
    testing = sorted(
        (item for item in metadata.values() if item.fold == inner_fold),
        key=lambda item: item.example_id,
    )
    outer_testing = sorted(
        (item for item in metadata.values() if item.fold == outer_fold),
        key=lambda item: item.example_id,
    )
    if not training or not testing:
        raise ValueError(f"outer {outer_fold}/inner {inner_fold} has an empty partition")
    _require_two_classes(training, description=f"outer {outer_fold}/inner {inner_fold} training")
    _require_two_classes(testing, description=f"outer {outer_fold}/inner {inner_fold} query")
    _require_disjoint_homology(
        training,
        [*testing, *outer_testing],
        description=f"outer {outer_fold}/inner {inner_fold} nested split",
    )
    inputs = tuple(item.model_input for item in training)
    queries = tuple(item.model_input for item in testing)
    labels = np.asarray([item.label for item in training], dtype=np.int64)
    descriptor = DescriptorLogisticOracle(
        l2=base_config.logistic.l2,
        max_iterations=base_config.logistic.max_iterations,
        tolerance=base_config.logistic.tolerance,
        prior_strength=base_config.logistic.prior_strength,
    )
    descriptor.fit(inputs, labels)
    descriptor_probability = descriptor.predict_proba(queries)
    descriptor_evidence = _audit_descriptor_fit(
        descriptor,
        inputs,
        labels.astype(np.float64),
        description=f"outer {outer_fold}/inner {inner_fold}",
    )
    knn_probability = _predict_knn_from_identity(
        training, testing, identity, sequence_index, base_config
    )
    esm_head, mean, scale, coefficients = _fit_esm_head(
        training,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        embeddings=embeddings,
        config=config,
    )
    esm_probability = _predict_esm(
        testing, embeddings, mean, scale, coefficients, config.probability_clip
    )
    similarities = _max_similarity(testing, training, identity, sequence_index)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(testing):
        base_probability = float(0.5 * (descriptor_probability[index] + knn_probability[index]))
        similarity = float(similarities[index])
        rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "example_id": item.example_id,
                "cluster_id": item.cluster_id,
                "label": item.label,
                "base_probability": base_probability,
                "esm_probability": float(esm_probability[index]),
                "similarity": similarity,
                "hard_gate": _hard_gate(similarity, config),
                "soft_gate": _soft_gate(similarity, config),
                "probability_clip": config.probability_clip,
            }
        )
    ledger_payload = "".join(
        "\t".join(
            (
                str(item["outer_fold"]),
                str(item["inner_fold"]),
                str(item["example_id"]),
                float(item["base_probability"]).hex(),
                float(item["esm_probability"]).hex(),
                float(item["similarity"]).hex(),
            )
        )
        + "\n"
        for item in rows
    ).encode("ascii")
    fit_evidence: dict[str, object] = {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "excluded_folds": sorted(excluded),
        "training": _partition_evidence(training),
        "query": _partition_evidence(testing),
        "descriptor_fit": descriptor_evidence,
        "homology_knn_fit": {
            "converged": True,
            "finite_predictions": bool(np.all(np.isfinite(knn_probability))),
            "neighbors": base_config.knn.neighbors,
            "similarity_power": base_config.knn.similarity_power,
            "prior_strength": base_config.knn.prior_strength,
            "minimum_weight": base_config.knn.minimum_weight,
        },
        "esm_fit": asdict(esm_head),
        "prediction_rows": len(rows),
        "prediction_ledger_sha256": hashlib.sha256(ledger_payload).hexdigest(),
    }
    similarity_evidence = _split_similarity_evidence(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        training=training,
        testing=testing,
        values=similarities,
    )
    return rows, fit_evidence, similarity_evidence


def _prediction_row(
    model: str,
    item: Metadata,
    *,
    similarity: float,
    gate_policy: str,
    gate_value: float | str,
    beta: float | str,
    probability: float,
) -> dict[str, object]:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{model} produced an invalid probability")
    return {
        "model": model,
        "example_id": item.example_id,
        "sequence_id": item.sequence_id,
        "sequence": item.sequence,
        "strain": item.strain,
        "gram": item.gram,
        "label": item.label,
        "source_observations": item.source_observations,
        "fold": item.fold,
        "cluster_id": item.cluster_id,
        "max_train_identity": similarity,
        "gate_policy": gate_policy,
        "gate_value": gate_value,
        "beta": beta,
        "probability": probability,
    }


def _make_predictions(
    metadata: Mapping[str, Metadata],
    base_predictions: Mapping[str, Mapping[str, float]],
    *,
    embeddings: Mapping[str, FloatArray],
    identity: FloatArray,
    sequence_index: Mapping[str, int],
    base_config: Gate1Config,
    config: SimilarityGateConfig,
) -> tuple[
    list[dict[str, object]],
    list[EsmHead],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    rows: list[dict[str, object]] = []
    outer_heads: list[EsmHead] = []
    nested_fits: list[dict[str, object]] = []
    nested_similarity: list[dict[str, object]] = []
    outer_similarity: list[dict[str, object]] = []
    gate_selections: list[dict[str, object]] = []
    nested_ledger_rows: list[dict[str, object]] = []
    for outer_fold in range(config.folds):
        nested_rows: list[dict[str, object]] = []
        for inner_fold in range(config.folds):
            if inner_fold == outer_fold:
                continue
            split_rows, fit_evidence, similarity_evidence = _nested_predictions(
                metadata,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                embeddings=embeddings,
                identity=identity,
                sequence_index=sequence_index,
                base_config=base_config,
                config=config,
            )
            nested_rows.extend(split_rows)
            nested_ledger_rows.extend(split_rows)
            nested_fits.append(fit_evidence)
            nested_similarity.append(similarity_evidence)
        selections = {
            gate: _select_beta(
                nested_rows,
                outer_fold=outer_fold,
                gate=gate,
                config=config,
            )
            for gate in ("ungated", "hard_lt_0_4", "soft_0_4_to_0_6")
        }
        gate_selections.append(
            {
                "outer_fold": outer_fold,
                "ungated": selections["ungated"],
                "hard_lt_0_4": selections["hard_lt_0_4"],
                "soft_0_4_to_0_6": selections["soft_0_4_to_0_6"],
            }
        )
        training = sorted(
            (item for item in metadata.values() if item.fold != outer_fold),
            key=lambda item: item.example_id,
        )
        testing = sorted(
            (item for item in metadata.values() if item.fold == outer_fold),
            key=lambda item: item.example_id,
        )
        _require_disjoint_homology(
            training, testing, description=f"outer fold {outer_fold} final split"
        )
        head, mean, scale, coefficients = _fit_esm_head(
            training,
            outer_fold=outer_fold,
            inner_fold=None,
            embeddings=embeddings,
            config=config,
        )
        outer_heads.append(head)
        esm_probability = _predict_esm(
            testing, embeddings, mean, scale, coefficients, config.probability_clip
        )
        similarities = _max_similarity(testing, training, identity, sequence_index)
        if any(
            format(float(value), ".12g") != format(item.max_train_identity, ".12g")
            for item, value in zip(testing, similarities, strict=True)
        ):
            raise ValueError(
                f"outer fold {outer_fold} recomputed identities differ from accepted v7 metadata"
            )
        outer_similarity.append(
            _split_similarity_evidence(
                outer_fold=outer_fold,
                inner_fold=None,
                training=training,
                testing=testing,
                values=similarities,
            )
        )
        selected_ungated = float(selections["ungated"]["selected_beta"])
        selected_hard = float(selections["hard_lt_0_4"]["selected_beta"])
        selected_soft = float(selections["soft_0_4_to_0_6"]["selected_beta"])
        for index, item in enumerate(testing):
            similarity = float(similarities[index])
            if (
                _hard_gate(similarity, config) != _hard_gate(item.max_train_identity, config)
                or (similarity >= config.soft_gate_zero_at)
                != (item.max_train_identity >= config.soft_gate_zero_at)
                or (similarity < config.soft_gate_full_below)
                != (item.max_train_identity < config.soft_gate_full_below)
            ):
                raise ValueError("stored identity rounding changes a frozen gate boundary")
            base_values = {
                model: float(base_predictions[model][item.example_id]) for model in BASE_MODELS
            }
            base = base_values[PROMOTION_REFERENCE]
            esm = float(esm_probability[index])
            hard = _hard_gate(similarity, config)
            soft = _soft_gate(similarity, config)
            probabilities = {
                **base_values,
                ESM_MODEL: esm,
                PROBABILITY_BLEND: 0.5 * (base + esm),
                UNGATED_MODEL: _residual_probability(
                    base,
                    esm,
                    beta=selected_ungated,
                    gate=1.0,
                    clip=config.probability_clip,
                ),
                HARD_GATED_MODEL: _residual_probability(
                    base,
                    esm,
                    beta=selected_hard,
                    gate=hard,
                    clip=config.probability_clip,
                ),
                FIXED_SOFT_MODEL: _residual_probability(
                    base,
                    esm,
                    beta=config.fixed_soft_beta,
                    gate=soft,
                    clip=config.probability_clip,
                ),
                PRIMARY_MODEL: _residual_probability(
                    base,
                    esm,
                    beta=selected_soft,
                    gate=soft,
                    clip=config.probability_clip,
                ),
            }
            annotations: dict[str, tuple[str, float | str, float | str]] = {
                **{model: ("not_applicable", "", "") for model in BASE_MODELS},
                ESM_MODEL: ("not_applicable", "", ""),
                PROBABILITY_BLEND: ("not_applicable", "", ""),
                UNGATED_MODEL: ("ungated", 1.0, selected_ungated),
                HARD_GATED_MODEL: ("hard_lt_0_4", hard, selected_hard),
                FIXED_SOFT_MODEL: (
                    "soft_0_4_to_0_6",
                    soft,
                    config.fixed_soft_beta,
                ),
                PRIMARY_MODEL: ("soft_0_4_to_0_6", soft, selected_soft),
            }
            for model in METHODS:
                gate_name, gate_value, beta = annotations[model]
                rows.append(
                    _prediction_row(
                        model,
                        item,
                        similarity=item.max_train_identity,
                        gate_policy=gate_name,
                        gate_value=gate_value,
                        beta=beta,
                        probability=probabilities[model],
                    )
                )
    rows.sort(key=lambda item: (METHODS.index(str(item["model"])), str(item["example_id"])))
    expected_rows = len(metadata) * len(METHODS)
    if len(rows) != expected_rows or len(nested_ledger_rows) != len(metadata) * (config.folds - 1):
        raise ValueError("outer/nested prediction row contract is incomplete")
    reference = {
        str(item["example_id"]): float(item["probability"])
        for item in rows
        if item["model"] == PROMOTION_REFERENCE
    }
    fallback_contract = (
        (HARD_GATED_MODEL, 0.4),
        (FIXED_SOFT_MODEL, 0.6),
        (PRIMARY_MODEL, 0.6),
    )
    for method, threshold in fallback_contract:
        selected = [
            item
            for item in rows
            if item["model"] == method and float(item["max_train_identity"]) >= threshold
        ]
        if not selected or any(
            float(item["probability"]) != reference[str(item["example_id"])] for item in selected
        ):
            raise ValueError(f"{method} is not exactly equal to base at similarity >= {threshold}")
    nested_payload = "".join(
        "\t".join(
            (
                str(item["outer_fold"]),
                str(item["inner_fold"]),
                str(item["example_id"]),
                float(item["base_probability"]).hex(),
                float(item["esm_probability"]).hex(),
                float(item["similarity"]).hex(),
            )
        )
        + "\n"
        for item in sorted(
            nested_ledger_rows,
            key=lambda value: (
                int(value["outer_fold"]),
                int(value["inner_fold"]),
                str(value["example_id"]),
            ),
        )
    ).encode("ascii")
    nested_contract = {
        "rows": len(nested_ledger_rows),
        "sha256": hashlib.sha256(nested_payload).hexdigest(),
        "encoding": (
            "outer_fold, inner_fold, example_id, base probability, ESM probability, and "
            "split-local similarity; tab-separated, float.hex(), sorted, terminal LF"
        ),
    }
    return (
        rows,
        outer_heads,
        nested_fits,
        nested_similarity,
        outer_similarity,
        gate_selections,
        nested_contract,
    )


def _metric(rows: Sequence[Mapping[str, object]], bins: int) -> dict[str, int | float | None]:
    return binary_metrics(
        [int(item["label"]) for item in rows],
        [float(item["probability"]) for item in rows],
        calibration_bins=bins,
    )


def _summarize(
    rows: Sequence[Mapping[str, object]], config: SimilarityGateConfig
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    if tuple(grouped) != METHODS or any(
        len(grouped[method]) != len(grouped[METHODS[0]]) for method in METHODS
    ):
        raise ValueError("diagnostic methods do not have identical complete support")
    model_metrics: dict[str, object] = {}
    fixed_strata = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8))
    for method in METHODS:
        model_rows = grouped[method]
        by_similarity = {
            f"[{left:.2f},{right:.2f})": _metric(
                [item for item in model_rows if left <= float(item["max_train_identity"]) < right],
                config.calibration_bins,
            )
            for left, right in fixed_strata
            if any(left <= float(item["max_train_identity"]) < right for item in model_rows)
        }
        if set(by_similarity) != {"[0.00,0.40)", "[0.40,0.60)", "[0.60,0.80)"}:
            raise ValueError("accepted panel does not cover every fixed similarity stratum")
        model_metrics[method] = {
            "overall": _metric(model_rows, config.calibration_bins),
            "by_fold": {
                str(fold): _metric(
                    [item for item in model_rows if int(item["fold"]) == fold],
                    config.calibration_bins,
                )
                for fold in range(config.folds)
            },
            "by_gram": {
                gram: _metric(
                    [item for item in model_rows if str(item["gram"]) == gram],
                    config.calibration_bins,
                )
                for gram in sorted({str(item["gram"]) for item in model_rows})
            },
            "by_strain": {
                strain: _metric(
                    [item for item in model_rows if str(item["strain"]) == strain],
                    config.calibration_bins,
                )
                for strain in sorted({str(item["strain"]) for item in model_rows})
            },
            "by_max_train_identity": by_similarity,
        }
    aligned = {
        method: {str(item["example_id"]): item for item in grouped[method]} for method in METHODS
    }
    example_ids = sorted(aligned[PROMOTION_REFERENCE])
    if any(set(aligned[method]) != set(example_ids) for method in METHODS):
        raise ValueError("diagnostic methods have different example IDs")
    clusters: dict[str, list[str]] = defaultdict(list)
    for example_id in example_ids:
        reference = aligned[PROMOTION_REFERENCE][example_id]
        candidate = aligned[PRIMARY_MODEL][example_id]
        if any(
            candidate[field] != reference[field]
            for field in ("label", "fold", "cluster_id", "sequence_id")
        ):
            raise ValueError("primary/reference metadata are not aligned")
        clusters[str(reference["cluster_id"])].append(example_id)
    cluster_ids = tuple(sorted(clusters))
    metric_names = ("roc_auc", "brier", "log_loss")
    samples = {name: [] for name in metric_names}
    generator = np.random.default_rng(config.seed)
    for replicate in range(1, config.bootstrap_replicates + 1):
        sampled_clusters = generator.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_ids = [
            example_id
            for cluster_id in sampled_clusters
            for example_id in clusters[str(cluster_id)]
        ]
        candidate_metric = _metric(
            [aligned[PRIMARY_MODEL][example_id] for example_id in sampled_ids],
            config.calibration_bins,
        )
        reference_metric = _metric(
            [aligned[PROMOTION_REFERENCE][example_id] for example_id in sampled_ids],
            config.calibration_bins,
        )
        for name in metric_names:
            left = candidate_metric[name]
            right = reference_metric[name]
            if (
                left is None
                or right is None
                or not math.isfinite(float(left))
                or not math.isfinite(float(right))
            ):
                raise ValueError(
                    f"paired component bootstrap replicate {replicate} is undefined/non-finite"
                )
            samples[name].append(float(left) - float(right))
    if any(len(values) != config.bootstrap_replicates for values in samples.values()):
        raise ValueError("paired component bootstrap evidence is incomplete")
    candidate_overall = cast(dict[str, object], model_metrics[PRIMARY_MODEL])["overall"]
    reference_overall = cast(dict[str, object], model_metrics[PROMOTION_REFERENCE])["overall"]
    assert isinstance(candidate_overall, dict) and isinstance(reference_overall, dict)
    paired = {
        name: {
            "point": float(candidate_overall[name]) - float(reference_overall[name]),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "successful_replicates": len(values),
        }
        for name, values in samples.items()
    }
    checks = {
        "roc_auc_delta_lower_above_minimum": (
            paired["roc_auc"]["lower"] > config.promotion_auc_delta_lower_minimum
        ),
        "brier_delta_upper_below_maximum": (
            paired["brier"]["upper"] < config.promotion_brier_delta_upper_maximum
        ),
        "log_loss_delta_upper_below_maximum": (
            paired["log_loss"]["upper"] < config.promotion_log_loss_delta_upper_maximum
        ),
    }
    statistical_gate_passed = all(checks.values())
    matrix = np.asarray(
        [
            [float(aligned[method][example_id]["probability"]) for example_id in example_ids]
            for method in METHODS
        ],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(matrix)
    if not np.all(np.isfinite(correlation)):
        raise ValueError("diagnostic method probability correlations are non-finite")
    return {
        "schema_version": 1,
        "models": model_metrics,
        "primary_comparison": {
            "candidate": PRIMARY_MODEL,
            "reference": PROMOTION_REFERENCE,
            "delta_definition": "candidate minus reference",
            "homology_cluster_paired_bootstrap_95ci": paired,
        },
        "promotion_rule": {
            "scope": "statistical_offline_gate_only",
            "reference": PROMOTION_REFERENCE,
            "candidates": [PRIMARY_MODEL],
            "requirements": {
                "roc_auc_delta_bootstrap_95ci_lower_strictly_above": (
                    config.promotion_auc_delta_lower_minimum
                ),
                "brier_delta_bootstrap_95ci_upper_strictly_below": (
                    config.promotion_brier_delta_upper_maximum
                ),
                "log_loss_delta_bootstrap_95ci_upper_strictly_below": (
                    config.promotion_log_loss_delta_upper_maximum
                ),
            },
            "decision": {
                "candidate": PRIMARY_MODEL,
                "passed": statistical_gate_passed,
                "checks": checks,
            },
        },
        "paired_component_bootstrap": {
            "unit": "homology_cluster",
            "requested_replicates": config.bootstrap_replicates,
            "successful_replicates": config.bootstrap_replicates,
            "attempted_replicates": config.bootstrap_replicates,
            "rejected_replicates": 0,
            "draw_policy": "exactly first requested draws; fail if any draw is undefined; no redraw",
            "seed": config.seed,
        },
        "probability_correlation": {
            method: {
                other: float(correlation[index, other_index])
                for other_index, other in enumerate(METHODS)
            }
            for index, method in enumerate(METHODS)
        },
        "statistical_gate_passed": statistical_gate_passed,
    }


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("refusing to serialize a non-finite benchmark value")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_SCHEMA, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized: dict[str, object] = {}
            for field in PREDICTION_SCHEMA:
                value = row[field]
                if field == "max_train_identity":
                    serialized[field] = format(float(value), ".12g")
                elif isinstance(value, float):
                    serialized[field] = repr(value)
                else:
                    serialized[field] = value
            writer.writerow(serialized)


def run_esm_similarity_gate_oof(
    *,
    base_oof_path: str | Path,
    base_manifest_path: str | Path,
    base_folds_path: str | Path,
    base_split_compatibility_path: str | Path,
    base_checksums_path: str | Path,
    base_config_path: str | Path,
    embedding_matrix_path: str | Path,
    embedding_index_path: str | Path,
    embedding_manifest_path: str | Path,
    embedding_trust_receipt_path: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    frozen_input_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the fully nested ESM residual audit and publish one atomic result tree."""

    if GIT_SHA1_RE.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase SHA-1 commit ID")
    base_oof = Path(base_oof_path).resolve(strict=True)
    base_manifest = Path(base_manifest_path).resolve(strict=True)
    base_folds = Path(base_folds_path).resolve(strict=True)
    base_split = Path(base_split_compatibility_path).resolve(strict=True)
    base_checksums = Path(base_checksums_path).resolve(strict=True)
    base_config_path = Path(base_config_path).resolve(strict=True)
    embedding_matrix = Path(embedding_matrix_path).resolve(strict=True)
    embedding_index = Path(embedding_index_path).resolve(strict=True)
    embedding_manifest = Path(embedding_manifest_path).resolve(strict=True)
    embedding_receipt = Path(embedding_trust_receipt_path).resolve(strict=True)
    code_manifest = Path(code_manifest_path).resolve(strict=True)
    frozen_input_manifest = Path(frozen_input_manifest_path).resolve(strict=True)
    config = load_config(config_path)
    final_output = Path(output_dir).absolute()
    if final_output.exists() or final_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing ESM gate output: {final_output}")
    if _sha256(base_checksums) != ACCEPTED_BASE_CHECKSUMS_SHA256:
        raise ValueError("base checksums do not match accepted Gate-1-v7 job 223009")
    accepted_embedding_hashes = {
        embedding_matrix: ACCEPTED_EMBEDDING_MATRIX_SHA256,
        embedding_index: ACCEPTED_EMBEDDING_INDEX_SHA256,
        embedding_manifest: ACCEPTED_EMBEDDING_MANIFEST_SHA256,
        embedding_receipt: ACCEPTED_TRUST_RECEIPT_SHA256,
    }
    if any(_sha256(path) != expected for path, expected in accepted_embedding_hashes.items()):
        raise ValueError("embedding artifacts differ from accepted extraction job 222578")
    code_entries = _read_sha256_manifest(code_manifest, description="ESM gate code manifest")
    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parents[1]
    required_code = [
        (source.relative_to(repository_root).as_posix(), source)
        for source in sorted(package_root.rglob("*.py"))
    ]
    required_code.extend(
        (
            ("configs/benchmarks/oracle_gate1.toml", base_config_path),
            ("configs/evaluation/esm_similarity_gate_oof.toml", config.path),
        )
    )
    for logical_path, source in required_code:
        _require_manifest_entry(
            code_entries,
            logical_path=logical_path,
            source=source,
            manifest_name="ESM gate code manifest",
        )
    frozen_entries = _read_sha256_manifest(
        frozen_input_manifest, description="ESM gate frozen-input manifest"
    )
    for logical_path, source in (
        ("base/oof_predictions.csv", base_oof),
        ("base/manifest.json", base_manifest),
        ("base/folds.json", base_folds),
        ("base/split_compatibility.json", base_split),
        ("base/SHA256SUMS", base_checksums),
        ("embedding/embeddings/embeddings.npy", embedding_matrix),
        ("embedding/embeddings/embedding_index.csv", embedding_index),
        ("embedding/embeddings/embedding_manifest.json", embedding_manifest),
        ("embedding/verification_receipt.json", embedding_receipt),
    ):
        _require_manifest_entry(
            frozen_entries,
            logical_path=logical_path,
            source=source,
            manifest_name="ESM gate frozen-input manifest",
        )
    if _sha256(base_config_path) != BASE_CONFIG_SHA256:
        raise ValueError("Gate-1 config differs from accepted parser-v7 configuration")
    base_config = Gate1Config.from_toml(base_config_path)
    if (
        base_config.folds != config.folds
        or base_config.homology_identity_threshold != config.homology_identity_threshold
        or base_config.similarity_bin_edges != config.similarity_bin_edges
    ):
        raise ValueError("ESM gate and accepted Gate-1 fold contracts differ")
    base_document, _ = _verify_base_manifest(
        base_oof,
        base_manifest,
        base_folds,
        base_split,
        base_checksums,
        base_config_path,
    )
    metadata, base_predictions = _read_base_oof(base_oof)
    _verify_base_folds(base_folds, metadata=metadata, config=base_config, manifest=base_document)
    _verify_split_compatibility(
        base_split, metadata=metadata, config=base_config, manifest=base_document
    )
    base_evidence = _partition_evidence(tuple(metadata.values()))
    if any(base_evidence.get(key) != value for key, value in ACCEPTED_BASE_EVIDENCE.items()):
        raise ValueError("accepted Gate-1 v7 census/digests differ from job 223009")
    label_summary = base_document.get("label_summary")
    if (
        not isinstance(label_summary, dict)
        or label_summary.get("activity_threshold_um") != base_config.activity_threshold_um
        or label_summary.get("included_strain_level_examples") != len(metadata)
        or label_summary.get("positive_examples") != base_evidence["positive_examples"]
        or label_summary.get("negative_examples") != base_evidence["negative_examples"]
    ):
        raise ValueError("base Gate-1 label summary differs from OOF metadata")
    embedding_document = _verify_embedding_manifest(
        embedding_matrix, embedding_index, embedding_manifest, embedding_receipt, config
    )
    embeddings, embedding_coverage = _read_embeddings(
        embedding_matrix,
        embedding_index,
        metadata=metadata,
        config=config,
    )
    if embedding_document.get("records") != embedding_coverage["embedding_sequences"]:
        raise ValueError("embedding manifest record count differs from its index")
    embedding_coverage.update(
        {
            "embedding_manifest_sha256": _sha256(embedding_manifest),
            "embedding_trust_receipt_sha256": _sha256(embedding_receipt),
            "accepted_subset_coverage": True,
        }
    )
    ordered_sequence_ids = sorted(embeddings)
    sequence_index = {sequence_id: index for index, sequence_id in enumerate(ordered_sequence_ids)}
    identity = pairwise_identity_matrix(
        [
            next(item.sequence for item in metadata.values() if item.sequence_id == sequence_id)
            for sequence_id in ordered_sequence_ids
        ]
    )
    if (
        identity.shape != (len(ordered_sequence_ids), len(ordered_sequence_ids))
        or identity.dtype != np.float64
        or not np.all(np.isfinite(identity))
        or not np.array_equal(identity, identity.T)
        or not np.array_equal(np.diag(identity), np.ones(len(identity)))
    ):
        raise ValueError("all-sequence identity matrix is invalid")
    (
        prediction_rows,
        outer_heads,
        nested_fits,
        nested_similarity,
        outer_similarity,
        gate_selections,
        nested_contract,
    ) = _make_predictions(
        metadata,
        base_predictions,
        embeddings=embeddings,
        identity=identity,
        sequence_index=sequence_index,
        base_config=base_config,
        config=config,
    )
    if (
        len(outer_heads) != config.folds
        or len(nested_fits) != config.folds * (config.folds - 1)
        or len(gate_selections) != config.folds
        or any(not item.converged or item.iterations < 1 for item in outer_heads)
        or any(
            not cast(dict[str, object], item["descriptor_fit"]).get("converged")
            or not cast(dict[str, object], item["homology_knn_fit"]).get("finite_predictions")
            or not cast(dict[str, object], item["esm_fit"]).get("converged")
            for item in nested_fits
        )
    ):
        raise ValueError("nested/outer fit evidence is incomplete or non-converged")
    metrics = _summarize(prediction_rows, config)
    statistical_gate_passed = bool(metrics["statistical_gate_passed"])
    base_rows = [item for item in prediction_rows if item["model"] == PROMOTION_REFERENCE]
    base_by_id = {str(item["example_id"]): float(item["probability"]) for item in base_rows}
    fallback_checks: dict[str, object] = {}
    for model, threshold in (
        (HARD_GATED_MODEL, config.hard_gate_threshold),
        (FIXED_SOFT_MODEL, config.soft_gate_zero_at),
        (PRIMARY_MODEL, config.soft_gate_zero_at),
    ):
        model_rows = {
            str(item["example_id"]): item
            for item in prediction_rows
            if item["model"] == model and float(item["max_train_identity"]) >= threshold
        }
        exact = bool(model_rows) and all(
            float(item["probability"]) == base_by_id[example_id]
            for example_id, item in model_rows.items()
        )
        if not exact:
            raise ValueError(f"{model} fallback predictions differ from the accepted base")
        fallback_checks[model] = {
            "similarity_at_or_above": threshold,
            "examples": len(model_rows),
            "exact_base_equal": True,
        }
    similarity_audit = {
        "schema_version": 1,
        "matrix": {
            "sequences": len(ordered_sequence_ids),
            "sequence_ids_sha256": _set_digest(set(ordered_sequence_ids)),
            "dtype": "float64",
            "sha256": _float_matrix_digest(identity, "<f8"),
            "encoding": (
                "ascending sequence_id rows/columns, contiguous little-endian float64 bytes"
            ),
            "symmetric": True,
            "unit_diagonal": True,
        },
        "outer_partitions": outer_similarity,
        "nested_partitions": nested_similarity,
        "outer_stored_identity_match": True,
        "exact_base_fallback": fallback_checks,
        "high_similarity_examples": cast(dict[str, object], fallback_checks[PRIMARY_MODEL])[
            "examples"
        ],
        "high_similarity_exact_base_equal": True,
    }
    fold_models = {
        "schema_version": 1,
        "model": ESM_MODEL,
        "feature_order": [
            "intercept",
            *(f"esm2_mean_{index:03d}" for index in range(config.embedding_dimension)),
            "gram_positive",
            "gram_negative",
            "gram_unknown",
        ],
        "scaling_policy": "embedding mean/std fit only on examples outside the outer fold",
        "outer_esm_fits": [asdict(item) for item in outer_heads],
        "gate_selections": gate_selections,
    }
    nested_document = {
        "schema_version": 1,
        "ordered_outer_inner_splits": len(nested_fits),
        "descriptor_fits": len(nested_fits),
        "homology_knn_fits": len(nested_fits),
        "esm_fits": len(nested_fits),
        "all_fits_converged_and_finite": True,
        "nested_prediction_ledger": nested_contract,
        "splits": nested_fits,
    }
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
    )
    predictions_out = staging / "esm_similarity_gate_oof_predictions.csv"
    coverage_out = staging / "embedding_coverage.json"
    similarity_out = staging / "similarity_audit.json"
    fold_models_out = staging / "fold_models.json"
    nested_out = staging / "nested_fits.json"
    metrics_out = staging / "metrics.json"
    manifest_out = staging / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "esm_similarity_gated_residual_oof",
        "base_oof_sha256": _sha256(base_oof),
        "base_manifest_sha256": _sha256(base_manifest),
        "base_folds_sha256": _sha256(base_folds),
        "base_split_compatibility_sha256": _sha256(base_split),
        "base_checksums_sha256": _sha256(base_checksums),
        "base_config_sha256": _sha256(base_config_path),
        "embedding_matrix_sha256": _sha256(embedding_matrix),
        "embedding_index_sha256": _sha256(embedding_index),
        "embedding_manifest_sha256": _sha256(embedding_manifest),
        "embedding_trust_receipt_sha256": _sha256(embedding_receipt),
        "config_sha256": _sha256(config.path),
        "accepted_upstream_contract": True,
        "normalized_parser_id": BASE_PARSER_ID,
        "fold_policy": BASE_FOLD_POLICY,
        "fold_assignment_policy": BASE_FOLD_ASSIGNMENT_POLICY,
        "methods_predeclared": list(METHODS),
        "promotion_reference": PROMOTION_REFERENCE,
        "promotion_candidates": [PRIMARY_MODEL],
        "base_examples": len(metadata),
        "positive_examples": base_evidence["positive_examples"],
        "negative_examples": base_evidence["negative_examples"],
        "unique_sequences": base_evidence["unique_sequences"],
        "homology_clusters": base_evidence["homology_clusters"],
        "base_contract": base_evidence,
        "embedding_coverage_contract": embedding_coverage,
        "embedding_extraction_contract": {
            "model": config.embedding_model,
            "representation_layer": config.representation_layer,
            "embedding_dimension": config.embedding_dimension,
            "input_fasta_sha256": config.expected_input_fasta_sha256,
            "input_fasta_manifest_sha256": config.expected_input_fasta_manifest_sha256,
            "model_checkpoint_sha256": config.expected_model_checkpoint_sha256,
            "contact_regression_sha256": config.expected_contact_regression_sha256,
            "environment_lock_sha256": config.expected_environment_lock_sha256,
            "trust_manifest_sha256": config.expected_trust_manifest_sha256,
            "embedding_worker_sha256": config.expected_embedding_worker_sha256,
            "runtime": cast(dict[str, object], embedding_document["runtime"]),
            "determinism": cast(dict[str, object], embedding_document["determinism"]),
            "upstream_pretraining": "self-supervised",
            "exact_ur50d_sequence_membership_established": False,
        },
        "gate_contract": {
            "formula": (
                "sigmoid(logit(base) + beta * gate(max_train_identity) * "
                "(logit(esm) - logit(base)))"
            ),
            "beta_grid": list(config.beta_grid),
            "hard_gate": "1 when identity < 0.4, otherwise 0",
            "soft_gate": ("1 below 0.4; (0.6 - identity) / 0.2 on [0.4,0.6); 0 at >=0.6"),
            "selection_objective": config.selection_objective,
            "selection_tie_break": config.selection_tie_break,
            "selection_tie_tolerance": config.selection_tie_tolerance,
            "fixed_soft_beta": config.fixed_soft_beta,
        },
        "nested_fit_contract": {
            "outer_esm_fits": len(outer_heads),
            "ordered_outer_inner_splits": len(nested_fits),
            "nested_descriptor_fits": len(nested_fits),
            "nested_homology_knn_fits": len(nested_fits),
            "nested_esm_fits": len(nested_fits),
            "gate_selections": len(gate_selections),
            "all_numerical_fits_converged_and_finite": True,
        },
        "nested_prediction_ledger": nested_contract,
        "similarity_contract": {
            "identity_matrix_sha256": similarity_audit["matrix"]["sha256"],
            "identity_matrix_dtype": "float64",
            "outer_partitions": len(outer_similarity),
            "ordered_outer_inner_partitions": len(nested_similarity),
            "outer_stored_identity_match": True,
            "exact_base_fallback": fallback_checks,
        },
        "statistical_gate_passed": statistical_gate_passed,
        "production_policy": {
            "production_eligible": False,
            "production_esm_weight": 0.0,
            "statistical_gate_controls_eligibility": False,
            "blockers": list(PRODUCTION_BLOCKERS),
        },
        "upstream_pretraining": (
            "ESM2 UR50D pretraining was self-supervised; exact sequence membership in the "
            "public pretraining corpus is not established"
        ),
        "provenance": {
            "code_manifest": {"filename": code_manifest.name, "sha256": _sha256(code_manifest)},
            "frozen_input_manifest": {
                "filename": frozen_input_manifest.name,
                "sha256": _sha256(frozen_input_manifest),
            },
            "git_commit": git_commit,
            "base_top_checksums": {
                "filename": base_checksums.name,
                "sha256": _sha256(base_checksums),
            },
            "accepted_base_job": "223009",
            "accepted_embedding_artifact_twins": ["222578", "222580"],
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {},
    }
    try:
        _write_predictions(predictions_out, prediction_rows)
        _write_json(coverage_out, embedding_coverage)
        _write_json(similarity_out, similarity_audit)
        _write_json(fold_models_out, fold_models)
        _write_json(nested_out, nested_document)
        _write_json(metrics_out, metrics)
        manifest["outputs"] = {
            predictions_out.name: _sha256(predictions_out),
            coverage_out.name: _sha256(coverage_out),
            similarity_out.name: _sha256(similarity_out),
            fold_models_out.name: _sha256(fold_models_out),
            nested_out.name: _sha256(nested_out),
            metrics_out.name: _sha256(metrics_out),
        }
        _write_json(manifest_out, manifest)
        staging.rename(final_output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-folds", type=Path, required=True)
    parser.add_argument("--base-split-compatibility", type=Path, required=True)
    parser.add_argument("--base-checksums", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--embedding-matrix", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--trust-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--frozen-input-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_esm_similarity_gate_oof(
        base_oof_path=args.base_oof,
        base_manifest_path=args.base_manifest,
        base_folds_path=args.base_folds,
        base_split_compatibility_path=args.base_split_compatibility,
        base_checksums_path=args.base_checksums,
        base_config_path=args.base_config,
        embedding_matrix_path=args.embedding_matrix,
        embedding_index_path=args.embedding_index,
        embedding_manifest_path=args.embedding_manifest,
        embedding_trust_receipt_path=args.trust_receipt,
        config_path=args.config,
        code_manifest_path=args.code_manifest,
        frozen_input_manifest_path=args.frozen_input_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

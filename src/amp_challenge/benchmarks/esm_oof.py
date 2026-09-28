"""Leakage-safe frozen-ESM2 sequence oracle on the Gate-1 homology folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from amp_challenge.benchmarks.oracle_gate1 import OofPrediction, binary_metrics
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

FloatArray = NDArray[np.float64]
GramClass = Literal["positive", "negative", "unknown"]
MODEL_NAME = "esm2_t6_8m_mean_logistic"
THREE_MEMBER_BLEND = "equal_descriptor_knn_esm"
FAMILY_BLEND = "equal_base_family_esm"
BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
NEW_MODELS = (MODEL_NAME, THREE_MEMBER_BLEND, FAMILY_BLEND)
MODEL_RELATIVE_PATH = "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt"
CONTACT_RELATIVE_PATH = "cache/torch/hub/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt"
LOCK_RELATIVE_PATH = "source/uv.lock"


@dataclass(frozen=True, slots=True)
class EsmOofConfig:
    path: Path
    metadata_model: str
    expected_base_oof_sha256: str
    expected_base_manifest_sha256: str
    expected_embedding_model: str
    expected_model_checkpoint_sha256: str
    expected_contact_regression_sha256: str
    expected_environment_lock_sha256: str
    expected_trust_manifest_sha256: str
    expected_embedding_worker_sha256: str
    expected_benchmark_code_sha256: str
    expected_python_version: str
    expected_torch_version: str
    expected_fair_esm_version: str
    expected_numpy_version: str
    expected_cuda_runtime: str
    expected_cudnn_version: int
    expected_device_type: str
    representation_layer: int
    embedding_dimension: int
    embedding_batch_size: int
    folds: int
    homology_identity_threshold: float
    similarity_bin_edges: tuple[float, ...]
    context_features: str
    comparison_base_models: tuple[str, ...]
    promotion_reference: str
    promotion_rule: str
    l2: float
    prior_strength: float
    max_iterations: int
    tolerance: float
    calibration_bins: int
    bootstrap_replicates: int
    seed: int


@dataclass(frozen=True, slots=True)
class Example:
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


@dataclass(frozen=True, slots=True)
class FoldModel:
    heldout_fold: int
    training_examples: int
    training_sequences: int
    positives: int
    negatives: int
    iterations: int
    converged: bool
    embedding_mean: tuple[float, ...]
    embedding_scale: tuple[float, ...]
    coefficients: tuple[float, ...]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(value: object, field: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"ESM OOF config {field} must be a lowercase SHA-256")
    return result


def _positive_float(raw: Mapping[str, object], name: str) -> float:
    value = float(raw[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"ESM OOF config {name} must be finite and positive")
    return value


def load_config(path: str | Path) -> EsmOofConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {
        "schema_version",
        "metadata_model",
        "expected_base_oof_sha256",
        "expected_base_manifest_sha256",
        "expected_embedding_model",
        "expected_model_checkpoint_sha256",
        "expected_contact_regression_sha256",
        "expected_environment_lock_sha256",
        "expected_trust_manifest_sha256",
        "expected_embedding_worker_sha256",
        "expected_benchmark_code_sha256",
        "expected_python_version",
        "expected_torch_version",
        "expected_fair_esm_version",
        "expected_numpy_version",
        "expected_cuda_runtime",
        "expected_cudnn_version",
        "expected_device_type",
        "representation_layer",
        "embedding_dimension",
        "embedding_batch_size",
        "folds",
        "homology_identity_threshold",
        "similarity_bin_edges",
        "context_features",
        "comparison_base_models",
        "promotion_reference",
        "promotion_rule",
        "l2",
        "prior_strength",
        "max_iterations",
        "tolerance",
        "calibration_bins",
        "bootstrap_replicates",
        "seed",
    }
    if set(raw) != allowed:
        raise ValueError(
            "ESM OOF config keys differ from the required schema: "
            f"missing={sorted(allowed - set(raw))}, extra={sorted(set(raw) - allowed)}"
        )
    if raw["schema_version"] != 1:
        raise ValueError("ESM OOF config schema_version must be 1")
    metadata_model = str(raw["metadata_model"]).strip()
    expected_model = str(raw["expected_embedding_model"]).strip()
    context = str(raw["context_features"]).strip()
    if not metadata_model or not expected_model:
        raise ValueError("metadata and embedding model names cannot be empty")
    if context != "gram_one_hot":
        raise ValueError("context_features must be gram_one_hot")
    expected_runtime = {
        name: str(raw[name]).strip()
        for name in (
            "expected_python_version",
            "expected_torch_version",
            "expected_fair_esm_version",
            "expected_numpy_version",
            "expected_cuda_runtime",
            "expected_device_type",
        )
    }
    if any(not value for value in expected_runtime.values()):
        raise ValueError("expected embedding runtime values cannot be empty")
    if expected_runtime["expected_device_type"] != "cuda":
        raise ValueError("expected_device_type must be cuda for the frozen GPU extraction")
    comparison_raw = raw["comparison_base_models"]
    if not isinstance(comparison_raw, list) or tuple(comparison_raw) != BASE_MODELS:
        raise ValueError(f"comparison_base_models must be exactly {list(BASE_MODELS)}")
    promotion_reference = str(raw["promotion_reference"])
    promotion_rule = str(raw["promotion_rule"])
    if promotion_reference != "equal_weight_ensemble":
        raise ValueError("promotion_reference must be equal_weight_ensemble")
    if promotion_rule != "paired_cluster_bootstrap_95ci_all_metrics":
        raise ValueError("promotion_rule must be paired_cluster_bootstrap_95ci_all_metrics")
    dimension = int(raw["embedding_dimension"])
    batch_size = int(raw["embedding_batch_size"])
    folds = int(raw["folds"])
    representation_layer = int(raw["representation_layer"])
    iterations = int(raw["max_iterations"])
    bins = int(raw["calibration_bins"])
    bootstrap = int(raw["bootstrap_replicates"])
    if dimension < 1 or batch_size < 1 or folds < 2 or representation_layer < 0:
        raise ValueError("invalid embedding dimension, fold count, or representation layer")
    if iterations < 1 or bins < 2 or bootstrap < 0:
        raise ValueError("invalid iteration, calibration-bin, or bootstrap count")
    threshold = float(raw["homology_identity_threshold"])
    if not 0 < threshold <= 1:
        raise ValueError("homology_identity_threshold must be in (0, 1]")
    edges_raw = raw["similarity_bin_edges"]
    if not isinstance(edges_raw, list):
        raise ValueError("similarity_bin_edges must be an array")
    edges = tuple(float(value) for value in edges_raw)
    if (
        len(edges) < 2
        or edges[0] != 0.0
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in edges)
        or any(right <= left for left, right in pairwise(edges))
        or edges[-1] < threshold
    ):
        raise ValueError("similarity_bin_edges must increase from 0 through the split threshold")
    seed = int(raw["seed"])
    cudnn_version = int(raw["expected_cudnn_version"])
    if seed < 0 or seed >= 2**32 or cudnn_version < 1:
        raise ValueError("invalid seed or expected cuDNN version")
    config = EsmOofConfig(
        path=config_path,
        metadata_model=metadata_model,
        expected_base_oof_sha256=_hash_value(
            raw["expected_base_oof_sha256"], "expected_base_oof_sha256"
        ),
        expected_base_manifest_sha256=_hash_value(
            raw["expected_base_manifest_sha256"], "expected_base_manifest_sha256"
        ),
        expected_embedding_model=expected_model,
        expected_model_checkpoint_sha256=_hash_value(
            raw["expected_model_checkpoint_sha256"], "expected_model_checkpoint_sha256"
        ),
        expected_contact_regression_sha256=_hash_value(
            raw["expected_contact_regression_sha256"], "expected_contact_regression_sha256"
        ),
        expected_environment_lock_sha256=_hash_value(
            raw["expected_environment_lock_sha256"], "expected_environment_lock_sha256"
        ),
        expected_trust_manifest_sha256=_hash_value(
            raw["expected_trust_manifest_sha256"], "expected_trust_manifest_sha256"
        ),
        expected_embedding_worker_sha256=_hash_value(
            raw["expected_embedding_worker_sha256"], "expected_embedding_worker_sha256"
        ),
        expected_benchmark_code_sha256=_hash_value(
            raw["expected_benchmark_code_sha256"], "expected_benchmark_code_sha256"
        ),
        expected_python_version=expected_runtime["expected_python_version"],
        expected_torch_version=expected_runtime["expected_torch_version"],
        expected_fair_esm_version=expected_runtime["expected_fair_esm_version"],
        expected_numpy_version=expected_runtime["expected_numpy_version"],
        expected_cuda_runtime=expected_runtime["expected_cuda_runtime"],
        expected_cudnn_version=cudnn_version,
        expected_device_type=expected_runtime["expected_device_type"],
        representation_layer=representation_layer,
        embedding_dimension=dimension,
        embedding_batch_size=batch_size,
        folds=folds,
        homology_identity_threshold=threshold,
        similarity_bin_edges=edges,
        context_features=context,
        comparison_base_models=BASE_MODELS,
        promotion_reference=promotion_reference,
        promotion_rule=promotion_rule,
        l2=_positive_float(raw, "l2"),
        prior_strength=_positive_float(raw, "prior_strength"),
        max_iterations=iterations,
        tolerance=_positive_float(raw, "tolerance"),
        calibration_bins=bins,
        bootstrap_replicates=bootstrap,
        seed=seed,
    )
    if _sha256(Path(__file__).resolve()) != config.expected_benchmark_code_sha256:
        raise ValueError("ESM OOF benchmark code checksum differs from the frozen config")
    return config


def _verify_base_manifest(
    oof_path: Path,
    manifest_path: Path,
    config: EsmOofConfig,
) -> dict[str, object]:
    actual_oof_hash = _sha256(oof_path)
    actual_manifest_hash = _sha256(manifest_path)
    if actual_oof_hash != config.expected_base_oof_sha256:
        raise ValueError("base OOF checksum differs from the frozen ESM config")
    if actual_manifest_hash != config.expected_base_manifest_sha256:
        raise ValueError("base manifest checksum differs from the frozen ESM config")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("benchmark") != "gate1_strain_activity":
        raise ValueError("base manifest is not a Gate-1 benchmark manifest")
    if document.get("fold_policy") != (
        "sequence-level single-link homology groups held out together"
    ):
        raise ValueError("base manifest does not declare the required homology fold policy")
    models = document.get("models")
    if not isinstance(models, list) or config.metadata_model not in models:
        raise ValueError("configured metadata model is not declared by the base manifest")
    artifacts = document.get("artifacts")
    artifact = artifacts.get("oof") if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError("base manifest has no OOF artifact")
    if artifact.get("filename") != oof_path.name or artifact.get("sha256") != actual_oof_hash:
        raise ValueError("base manifest does not match the supplied OOF CSV")
    return cast(dict[str, object], document)


def _read_examples(path: Path, config: EsmOofConfig) -> tuple[Example, ...]:
    required = {
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
    }
    examples: list[Example] = []
    observed_examples: set[str] = set()
    sequence_fold: dict[str, int] = {}
    cluster_fold: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"base OOF CSV is missing fields: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            if row["model"] != config.metadata_model:
                continue
            sequence = canonicalize_sequence(row["sequence"])
            sequence_id = row["sequence_id"]
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(f"base OOF row {row_number} sequence_id mismatch")
            example_id = row["example_id"].strip()
            if not example_id or example_id in observed_examples:
                raise ValueError(f"base OOF row {row_number} has duplicate/empty example_id")
            observed_examples.add(example_id)
            gram = row["gram"].strip()
            if gram not in {"positive", "negative", "unknown"}:
                raise ValueError(f"base OOF row {row_number} has invalid Gram class")
            label = int(row["label"])
            fold = int(row["fold"])
            identity = float(row["max_train_identity"])
            observations = int(row["source_observations"])
            cluster_id = row["cluster_id"].strip()
            strain = row["strain"].strip()
            if (
                label not in {0, 1}
                or fold not in range(config.folds)
                or not 0 <= identity < config.homology_identity_threshold
                or observations < 1
                or not cluster_id
                or not strain
            ):
                raise ValueError(f"base OOF row {row_number} has invalid benchmark metadata")
            if sequence_fold.setdefault(sequence_id, fold) != fold:
                raise ValueError("one sequence appears in multiple Gate-1 folds")
            if cluster_fold.setdefault(cluster_id, fold) != fold:
                raise ValueError("one homology cluster appears in multiple Gate-1 folds")
            examples.append(
                Example(
                    example_id=example_id,
                    sequence_id=sequence_id,
                    sequence=sequence,
                    strain=strain,
                    gram=cast(GramClass, gram),
                    label=label,
                    source_observations=observations,
                    fold=fold,
                    cluster_id=cluster_id,
                    max_train_identity=identity,
                )
            )
    if not examples:
        raise ValueError(f"no base OOF rows found for model {config.metadata_model!r}")
    if set(item.fold for item in examples) != set(range(config.folds)):
        raise ValueError("base OOF rows do not cover every configured fold")
    return tuple(sorted(examples, key=lambda item: item.example_id))


def _read_base_predictions(
    path: Path,
    examples: Sequence[Example],
) -> dict[str, tuple[OofPrediction, ...]]:
    """Read the three frozen base families and verify exact metadata alignment."""

    expected = {item.example_id: item for item in examples}
    grouped: dict[str, dict[str, OofPrediction]] = {name: {} for name in BASE_MODELS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "probability" not in reader.fieldnames:
            raise ValueError("base OOF CSV has no probability column")
        for row_number, row in enumerate(reader, start=2):
            model = row["model"]
            if model not in grouped:
                continue
            example_id = row["example_id"]
            example = expected.get(example_id)
            if example is None or example_id in grouped[model]:
                raise ValueError(f"base comparison row {row_number} has unknown/duplicate example")
            probability = float(row["probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"base comparison row {row_number} has invalid probability")
            metadata_matches = (
                row["sequence_id"] == example.sequence_id
                and row["sequence"] == example.sequence
                and row["strain"] == example.strain
                and row["gram"] == example.gram
                and int(row["label"]) == example.label
                and int(row["source_observations"]) == example.source_observations
                and int(row["fold"]) == example.fold
                and row["cluster_id"] == example.cluster_id
                and float(row["max_train_identity"]) == example.max_train_identity
            )
            if not metadata_matches:
                raise ValueError(f"base comparison row {row_number} metadata mismatch")
            grouped[model][example_id] = OofPrediction(
                model=model,
                example_id=example.example_id,
                sequence_id=example.sequence_id,
                sequence=example.sequence,
                strain=example.strain,
                gram=example.gram,
                label=example.label,
                source_observations=example.source_observations,
                fold=example.fold,
                cluster_id=example.cluster_id,
                max_train_identity=example.max_train_identity,
                probability=probability,
            )
    expected_ids = set(expected)
    for model, rows in grouped.items():
        if set(rows) != expected_ids:
            raise ValueError(f"base OOF model {model!r} does not exactly cover all examples")
    descriptor = grouped["descriptor_logistic"]
    knn = grouped["homology_knn"]
    ensemble = grouped["equal_weight_ensemble"]
    for example_id in expected_ids:
        reconstructed = 0.5 * (descriptor[example_id].probability + knn[example_id].probability)
        if not math.isclose(
            ensemble[example_id].probability,
            reconstructed,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("stored equal-weight base ensemble is not the member mean")
    return {model: tuple(rows[item] for item in sorted(rows)) for model, rows in grouped.items()}


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def export_gate1_fasta(
    *,
    base_oof_path: str | Path,
    base_manifest_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Export exactly the unique sequences referenced by the frozen Gate-1 OOF."""

    base_oof = Path(base_oof_path).resolve()
    base_manifest = Path(base_manifest_path).resolve()
    config = load_config(config_path)
    _verify_base_manifest(base_oof, base_manifest, config)
    examples = _read_examples(base_oof, config)
    sequences: dict[str, str] = {}
    for item in examples:
        previous = sequences.setdefault(item.sequence_id, item.sequence)
        if previous != item.sequence:  # pragma: no cover - SHA collision defense
            raise ValueError("one sequence_id maps to different canonical sequences")
    records = tuple(sorted(sequences.items()))

    output = Path(output_dir).resolve()
    fasta_path = output / "gate1_sequences.fasta"
    manifest_path = output / "fasta_manifest.json"
    if any(path.exists() for path in (fasta_path, manifest_path)):
        raise FileExistsError(f"refusing to overwrite Gate-1 FASTA outputs in {output}")
    output.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("x", encoding="ascii", newline="\n") as handle:
        for sequence_id, sequence in records:
            handle.write(f">sequence_id={sequence_id}\n{sequence}\n")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "gate1_esm2_embedding_input",
        "base_oof_sha256": _sha256(base_oof),
        "base_manifest_sha256": _sha256(base_manifest),
        "config_sha256": _sha256(config.path),
        "metadata_model": config.metadata_model,
        "records": len(records),
        "ordering": "ascending sequence_id",
        "artifact": {"filename": fasta_path.name, "sha256": _sha256(fasta_path)},
    }
    _write_json(manifest_path, manifest)
    return manifest


def _read_strict_fasta(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or len(lines) % 2:
        raise ValueError("Gate-1 FASTA must contain non-empty two-line records")
    records: list[tuple[str, str]] = []
    for offset in range(0, len(lines), 2):
        match = re.fullmatch(r">sequence_id=([0-9a-f]{64})", lines[offset])
        if match is None:
            raise ValueError(f"invalid Gate-1 FASTA header at line {offset + 1}")
        sequence = canonicalize_sequence(lines[offset + 1])
        sequence_id = match.group(1)
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"Gate-1 FASTA sequence ID mismatch at line {offset + 1}")
        records.append((sequence_id, sequence))
    if len(set(records)) != len(records):
        raise ValueError("Gate-1 FASTA contains duplicate records")
    return tuple(records)


def _verify_fasta_chain(
    *,
    fasta: Path,
    manifest_path: Path,
    base_oof: Path,
    base_manifest: Path,
    config: EsmOofConfig,
    examples: Sequence[Example],
) -> dict[str, object]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = document.get("artifact")
    if (
        document.get("schema_version") != 1
        or document.get("purpose") != "gate1_esm2_embedding_input"
        or document.get("base_oof_sha256") != _sha256(base_oof)
        or document.get("base_manifest_sha256") != _sha256(base_manifest)
        or document.get("config_sha256") != _sha256(config.path)
        or document.get("metadata_model") != config.metadata_model
        or not isinstance(artifact, dict)
        or artifact.get("filename") != fasta.name
        or artifact.get("sha256") != _sha256(fasta)
    ):
        raise ValueError("Gate-1 FASTA manifest/input chain mismatch")
    records = _read_strict_fasta(fasta)
    expected = {(item.sequence_id, item.sequence) for item in examples}
    if set(records) != expected or len(records) != len(expected):
        raise ValueError("Gate-1 FASTA does not exactly cover the base OOF sequences")
    if document.get("records") != len(records):
        raise ValueError("Gate-1 FASTA manifest record count mismatch")
    if [item[0] for item in records] != sorted(item[0] for item in records):
        raise ValueError("Gate-1 FASTA is not ordered by sequence_id")
    return cast(dict[str, object], document)


def _trust_artifact_hash(document: Mapping[str, object], name: str) -> str:
    trust = document.get("trust")
    artifacts = trust.get("artifacts") if isinstance(trust, dict) else None
    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError(f"embedding manifest has no trusted {name}")
    return str(artifact.get("sha256", ""))


def _verify_trust_receipt(path: Path, config: EsmOofConfig) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if text != canonical:
        raise ValueError("embedding trust receipt is not canonical JSON")
    required = {
        "schema_version",
        "receipt_type",
        "component",
        "bundle_root",
        "integration",
        "source",
        "trust_manifest",
        "artifacts",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("embedding trust receipt keys differ from the required schema")
    bundle_root = document["bundle_root"]
    if (
        document["schema_version"] != 1
        or document["receipt_type"] != "ampdiffusion_bundle_verification"
        or document["component"] != "generation"
        or not isinstance(bundle_root, str)
        or not PurePosixPath(bundle_root).is_absolute()
        or PurePosixPath(bundle_root).as_posix() != bundle_root
        or not isinstance(document["integration"], str)
        or not document["integration"]
    ):
        raise ValueError("embedding trust receipt identity is invalid")
    source = document["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"repository", "commit", "license"}
        or not isinstance(source["repository"], str)
        or not source["repository"].startswith("https://")
        or not isinstance(source["commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is None
        or not isinstance(source["license"], str)
        or not source["license"]
    ):
        raise ValueError("embedding trust receipt source metadata is invalid")
    manifest = document["trust_manifest"]
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"filename", "sha256", "size"}
        or manifest["filename"] != "artifacts.toml"
        or manifest["sha256"] != config.expected_trust_manifest_sha256
        or type(manifest["size"]) is not int
        or manifest["size"] <= 0
    ):
        raise ValueError("embedding trust receipt does not match the frozen trust manifest")
    raw_artifacts = document["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("embedding trust receipt has no artifacts")
    artifacts: dict[str, dict[str, object]] = {}
    ordered_paths: list[str] = []
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size", "role"}:
            raise ValueError(f"embedding trust receipt artifact {index} has an invalid schema")
        relative_text = raw["path"]
        digest = raw["sha256"]
        size = raw["size"]
        role = raw["role"]
        if not isinstance(relative_text, str) or "\\" in relative_text:
            raise ValueError(f"embedding trust receipt artifact {index} has an invalid path")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative_text in artifacts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size <= 0
            or not isinstance(role, str)
            or not role
        ):
            raise ValueError(f"embedding trust receipt artifact {index} has invalid metadata")
        artifacts[relative_text] = raw
        ordered_paths.append(relative_text)
    if ordered_paths != sorted(ordered_paths):
        raise ValueError("embedding trust receipt artifacts are not path-sorted")
    expected = {
        MODEL_RELATIVE_PATH: (config.expected_model_checkpoint_sha256, "pickle_checkpoint"),
        CONTACT_RELATIVE_PATH: (
            config.expected_contact_regression_sha256,
            "pickle_checkpoint",
        ),
        LOCK_RELATIVE_PATH: (config.expected_environment_lock_sha256, "environment_lock"),
    }
    for relative_text, (digest, role) in expected.items():
        artifact = artifacts.get(relative_text)
        if artifact is None or artifact["sha256"] != digest or artifact["role"] != role:
            raise ValueError(f"embedding trust receipt mismatch for {relative_text}")
    return cast(dict[str, object], document)


def _verify_embedding_manifest(
    *,
    matrix_path: Path,
    index_path: Path,
    manifest_path: Path,
    trust_receipt_path: Path,
    fasta: Path,
    fasta_manifest: Path,
    config: EsmOofConfig,
) -> dict[str, object]:
    receipt = _verify_trust_receipt(trust_receipt_path, config)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = document.get("outputs")
    trust = document.get("trust")
    determinism = document.get("determinism")
    runtime = document.get("runtime")
    worker_sha = str(document.get("worker_sha256", ""))
    if (
        document.get("schema_version") != 1
        or document.get("benchmark") != "esm2_embedding_extraction"
        or document.get("model") != config.expected_embedding_model
        or document.get("representation_layer") != config.representation_layer
        or document.get("embedding_dimension") != config.embedding_dimension
        or document.get("pooling")
        != "arithmetic mean over residue representations; BOS/EOS/padding excluded"
        or document.get("dtype") != "float32"
        or document.get("ordering") != "input FASTA order (ascending sequence_id)"
        or document.get("input_fasta_sha256") != _sha256(fasta)
        or document.get("input_fasta_manifest_sha256") != _sha256(fasta_manifest)
        or not isinstance(outputs, dict)
        or outputs.get(matrix_path.name) != _sha256(matrix_path)
        or outputs.get(index_path.name) != _sha256(index_path)
        or not isinstance(trust, dict)
        or trust.get("trust_manifest_sha256") != config.expected_trust_manifest_sha256
        or trust.get("verification_receipt_sha256") != _sha256(trust_receipt_path)
        or trust.get("bundle_root") != receipt.get("bundle_root")
        or not isinstance(determinism, dict)
        or determinism.get("seed") != config.seed
        or determinism.get("batch_size") != config.embedding_batch_size
        or determinism.get("torch_deterministic_algorithms") is not True
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("cudnn_deterministic") is not True
        or determinism.get("tf32") is not False
        or determinism.get("cublas_workspace_config") != ":4096:8"
        or worker_sha != config.expected_embedding_worker_sha256
        or not isinstance(runtime, dict)
        or runtime.get("python") != config.expected_python_version
        or runtime.get("torch") != config.expected_torch_version
        or runtime.get("fair_esm") != config.expected_fair_esm_version
        or runtime.get("numpy") != config.expected_numpy_version
        or runtime.get("cuda_runtime") != config.expected_cuda_runtime
        or runtime.get("cudnn") != config.expected_cudnn_version
        or runtime.get("device_type") != config.expected_device_type
        or not isinstance(runtime.get("device_name"), str)
        or not runtime.get("device_name")
        or not isinstance(runtime.get("device_capability"), list)
        or len(runtime["device_capability"]) != 2
        or not all(type(value) is int and value >= 0 for value in runtime["device_capability"])
    ):
        raise ValueError("embedding manifest/input/output chain mismatch")
    expected_trust_hashes = {
        "model_checkpoint": config.expected_model_checkpoint_sha256,
        "contact_regression_checkpoint": config.expected_contact_regression_sha256,
        "environment_lock": config.expected_environment_lock_sha256,
    }
    for name, expected in expected_trust_hashes.items():
        if _trust_artifact_hash(document, name) != expected:
            raise ValueError(f"embedding manifest trusted {name} checksum mismatch")
    return cast(dict[str, object], document)


def _read_embeddings(
    matrix_path: Path,
    index_path: Path,
    *,
    dimension: int,
    examples: Sequence[Example],
) -> dict[str, FloatArray]:
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError("embedding matrix has the wrong dtype or shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding matrix contains non-finite values")
    embeddings: dict[str, FloatArray] = {}
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != (
            "row_index",
            "sequence_id",
            "sequence",
            "length",
        ):
            raise ValueError("embedding index fields differ from the required schema")
        rows = list(reader)
    if len(rows) != matrix.shape[0]:
        raise ValueError("embedding index and matrix row counts differ")
    indexed_sequences: dict[str, str] = {}
    for expected_index, row in enumerate(rows):
        if int(row["row_index"]) != expected_index:
            raise ValueError("embedding index row numbers are not contiguous")
        sequence = canonicalize_sequence(row["sequence"])
        sequence_id = row["sequence_id"]
        if sequence_id != canonical_sequence_id(sequence) or int(row["length"]) != len(sequence):
            raise ValueError("embedding index sequence ID/length mismatch")
        if sequence_id in embeddings:
            raise ValueError("embedding index contains a duplicate sequence")
        indexed_sequences[sequence_id] = sequence
        embeddings[sequence_id] = matrix[expected_index].astype(np.float64)
    expected_sequences = {item.sequence_id: item.sequence for item in examples}
    if indexed_sequences != expected_sequences:
        raise ValueError("embedding index does not exactly join to the base OOF sequences")
    return embeddings


def _sigmoid(values: FloatArray) -> FloatArray:
    output = np.empty_like(values)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output


def _design_matrix(
    examples: Sequence[Example],
    embeddings: Mapping[str, FloatArray],
    mean: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    continuous = np.asarray([embeddings[item.sequence_id] for item in examples])
    standardized = (continuous - mean) / scale
    gram = np.zeros((len(examples), 3), dtype=np.float64)
    gram_index = {"positive": 0, "negative": 1, "unknown": 2}
    for index, item in enumerate(examples):
        gram[index, gram_index[item.gram]] = 1.0
    return np.column_stack((np.ones(len(examples)), standardized, gram))


def _fit_fold(
    training: Sequence[Example],
    *,
    heldout_fold: int,
    embeddings: Mapping[str, FloatArray],
    config: EsmOofConfig,
) -> tuple[FoldModel, FloatArray, FloatArray, FloatArray]:
    raw = np.asarray([embeddings[item.sequence_id] for item in training])
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale[scale < 1e-12] = 1.0
    design = _design_matrix(training, embeddings, mean, scale)
    labels = np.asarray([item.label for item in training], dtype=np.float64)
    prior = float(
        (np.sum(labels) + 0.5 * config.prior_strength) / (len(labels) + config.prior_strength)
    )
    prior = float(np.clip(prior, 1e-6, 1.0 - 1e-6))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    iterations = 0
    converged = True
    if np.any(labels == 0) and np.any(labels == 1):
        penalty = np.full(design.shape[1], config.l2, dtype=np.float64)
        penalty[0] = 0.0
        converged = False
        for iteration in range(1, config.max_iterations + 1):
            probabilities = _sigmoid(design @ coefficients)
            variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
            gradient = design.T @ (probabilities - labels) / len(labels)
            gradient += penalty * coefficients
            hessian = (design.T * variance) @ design / len(labels)
            hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            coefficients -= step
            iterations = iteration
            if float(np.max(np.abs(step))) <= config.tolerance:
                converged = True
                break
    model = FoldModel(
        heldout_fold=heldout_fold,
        training_examples=len(training),
        training_sequences=len({item.sequence_id for item in training}),
        positives=int(np.sum(labels)),
        negatives=int(len(labels) - np.sum(labels)),
        iterations=iterations,
        converged=converged,
        embedding_mean=tuple(float(value) for value in mean),
        embedding_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
    )
    return model, mean, scale, coefficients


def _make_predictions(
    examples: Sequence[Example],
    embeddings: Mapping[str, FloatArray],
    config: EsmOofConfig,
) -> tuple[tuple[OofPrediction, ...], tuple[FoldModel, ...]]:
    predictions: list[OofPrediction] = []
    models: list[FoldModel] = []
    for fold in range(config.folds):
        training = tuple(item for item in examples if item.fold != fold)
        testing = tuple(item for item in examples if item.fold == fold)
        if not training or not testing:
            raise ValueError(f"fold {fold} leaves an empty train or test partition")
        model, mean, scale, coefficients = _fit_fold(
            training,
            heldout_fold=fold,
            embeddings=embeddings,
            config=config,
        )
        models.append(model)
        probability = np.clip(
            _sigmoid(_design_matrix(testing, embeddings, mean, scale) @ coefficients),
            1e-6,
            1.0 - 1e-6,
        )
        for item, value in zip(testing, probability, strict=True):
            predictions.append(
                OofPrediction(
                    model=MODEL_NAME,
                    example_id=item.example_id,
                    sequence_id=item.sequence_id,
                    sequence=item.sequence,
                    strain=item.strain,
                    gram=item.gram,
                    label=item.label,
                    source_observations=item.source_observations,
                    fold=item.fold,
                    cluster_id=item.cluster_id,
                    max_train_identity=item.max_train_identity,
                    probability=float(value),
                )
            )
    if len(predictions) != len(examples):
        raise AssertionError("ESM OOF prediction count differs from the base examples")
    return (
        tuple(sorted(predictions, key=lambda item: item.example_id)),
        tuple(sorted(models, key=lambda item: item.heldout_fold)),
    )


def _fixed_oof_blends(
    esm_rows: Sequence[OofPrediction],
    base_rows: Mapping[str, Sequence[OofPrediction]],
) -> dict[str, tuple[OofPrediction, ...]]:
    """Apply fixed arithmetic blends only after every member is out-of-fold."""

    aligned = {model: {item.example_id: item for item in rows} for model, rows in base_rows.items()}
    output: dict[str, list[OofPrediction]] = {
        THREE_MEMBER_BLEND: [],
        FAMILY_BLEND: [],
    }
    for esm in esm_rows:
        descriptor = aligned["descriptor_logistic"][esm.example_id].probability
        knn = aligned["homology_knn"][esm.example_id].probability
        base_equal = aligned["equal_weight_ensemble"][esm.example_id].probability
        output[THREE_MEMBER_BLEND].append(
            replace(
                esm,
                model=THREE_MEMBER_BLEND,
                probability=float((descriptor + knn + esm.probability) / 3.0),
            )
        )
        output[FAMILY_BLEND].append(
            replace(
                esm,
                model=FAMILY_BLEND,
                probability=float(0.5 * base_equal + 0.5 * esm.probability),
            )
        )
    return {
        model: tuple(sorted(rows, key=lambda item: item.example_id))
        for model, rows in output.items()
    }


def _metric_subset(
    rows: Sequence[OofPrediction], config: EsmOofConfig
) -> dict[str, int | float | None]:
    return binary_metrics(
        [item.label for item in rows],
        [item.probability for item in rows],
        calibration_bins=config.calibration_bins,
    )


def _cluster_bootstrap(
    rows: Sequence[OofPrediction], config: EsmOofConfig
) -> dict[str, dict[str, float | int | None]]:
    names = ("roc_auc", "average_precision", "brier", "log_loss")
    point = _metric_subset(rows, config)
    samples: dict[str, list[float]] = {name: [] for name in names}
    by_cluster: dict[str, list[OofPrediction]] = defaultdict(list)
    for row in rows:
        by_cluster[row.cluster_id].append(row)
    clusters = tuple(sorted(by_cluster))
    generator = np.random.default_rng(config.seed)
    for _ in range(config.bootstrap_replicates):
        selected = [
            row
            for cluster in generator.choice(clusters, size=len(clusters), replace=True)
            for row in by_cluster[str(cluster)]
        ]
        values = _metric_subset(selected, config)
        for name in names:
            if values[name] is not None:
                samples[name].append(float(values[name]))
    return {
        name: {
            "point": point[name],
            "lower": None if not samples[name] else float(np.quantile(samples[name], 0.025)),
            "upper": None if not samples[name] else float(np.quantile(samples[name], 0.975)),
            "successful_replicates": len(samples[name]),
        }
        for name in names
    }


def _model_metrics(rows: Sequence[OofPrediction], config: EsmOofConfig) -> dict[str, object]:
    by_similarity: dict[str, object] = {}
    for left, right in zip(
        config.similarity_bin_edges, config.similarity_bin_edges[1:], strict=False
    ):
        selected = [
            item
            for item in rows
            if left <= item.max_train_identity < right
            or (right == 1.0 and item.max_train_identity == right)
        ]
        if selected:
            by_similarity[f"[{left:.2f},{right:.2f})"] = _metric_subset(selected, config)
    return {
        "overall": _metric_subset(rows, config),
        "homology_cluster_bootstrap_95ci": _cluster_bootstrap(rows, config),
        "by_fold": {
            str(fold): _metric_subset([item for item in rows if item.fold == fold], config)
            for fold in range(config.folds)
        },
        "by_strain": {
            strain: _metric_subset([item for item in rows if item.strain == strain], config)
            for strain in sorted({item.strain for item in rows})
        },
        "by_max_train_identity": by_similarity,
    }


def _paired_cluster_deltas(
    candidate_rows: Sequence[OofPrediction],
    reference_rows: Sequence[OofPrediction],
    config: EsmOofConfig,
) -> dict[str, object]:
    candidate = {item.example_id: item for item in candidate_rows}
    reference = {item.example_id: item for item in reference_rows}
    if set(candidate) != set(reference):
        raise ValueError("paired comparison models do not have identical example support")
    example_ids = sorted(candidate)
    for example_id in example_ids:
        left = candidate[example_id]
        right = reference[example_id]
        if (
            left.label != right.label
            or left.fold != right.fold
            or left.cluster_id != right.cluster_id
        ):
            raise ValueError("paired comparison metadata are not aligned")
    point_candidate = _metric_subset([candidate[item] for item in example_ids], config)
    point_reference = _metric_subset([reference[item] for item in example_ids], config)
    metric_names = ("roc_auc", "brier", "log_loss")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for example_id in example_ids:
        by_cluster[candidate[example_id].cluster_id].append(example_id)
    clusters = tuple(sorted(by_cluster))
    generator = np.random.default_rng(config.seed)
    for _ in range(config.bootstrap_replicates):
        sampled_ids = [
            example_id
            for cluster in generator.choice(clusters, size=len(clusters), replace=True)
            for example_id in by_cluster[str(cluster)]
        ]
        candidate_metrics = _metric_subset([candidate[item] for item in sampled_ids], config)
        reference_metrics = _metric_subset([reference[item] for item in sampled_ids], config)
        for name in metric_names:
            left = candidate_metrics[name]
            right = reference_metrics[name]
            if left is not None and right is not None:
                samples[name].append(float(left) - float(right))
    return {
        "candidate": candidate_rows[0].model,
        "reference": reference_rows[0].model,
        "delta_definition": "candidate minus reference",
        "homology_cluster_paired_bootstrap_95ci": {
            name: {
                "point": (
                    None
                    if point_candidate[name] is None or point_reference[name] is None
                    else float(point_candidate[name]) - float(point_reference[name])
                ),
                "lower": (None if not samples[name] else float(np.quantile(samples[name], 0.025))),
                "upper": (None if not samples[name] else float(np.quantile(samples[name], 0.975))),
                "successful_replicates": len(samples[name]),
            }
            for name in metric_names
        },
    }


def _comparisons(
    candidates: Mapping[str, Sequence[OofPrediction]],
    base: Mapping[str, Sequence[OofPrediction]],
    config: EsmOofConfig,
) -> dict[str, object]:
    # Full stored-base point metrics make every same-support reference visible;
    # paired intervals use the predeclared promotion reference to control cost.
    base_metrics = {name: _metric_subset(tuple(rows), config) for name, rows in base.items()}
    reference = base[config.promotion_reference]
    paired = {
        name: _paired_cluster_deltas(tuple(rows), tuple(reference), config)
        for name, rows in candidates.items()
    }
    promotion: dict[str, object] = {}
    for name, comparison in paired.items():
        intervals = cast(
            dict[str, dict[str, float | int | None]],
            comparison["homology_cluster_paired_bootstrap_95ci"],
        )
        auc_lower = intervals["roc_auc"]["lower"]
        brier_upper = intervals["brier"]["upper"]
        logloss_upper = intervals["log_loss"]["upper"]
        passes = (
            auc_lower is not None
            and brier_upper is not None
            and logloss_upper is not None
            and float(auc_lower) > 0.0
            and float(brier_upper) < 0.0
            and float(logloss_upper) < 0.0
        )
        promotion[name] = {
            "promoted": passes,
            "requirements": {
                "roc_auc_delta_95ci_lower_gt": 0.0,
                "brier_delta_95ci_upper_lt": 0.0,
                "log_loss_delta_95ci_upper_lt": 0.0,
            },
        }
    return {
        "same_support_examples": len(reference),
        "stored_base_point_metrics": base_metrics,
        "paired_deltas_vs_equal_weight_ensemble": paired,
        "promotion_rule": {
            "name": config.promotion_rule,
            "reference": config.promotion_reference,
            "decision": promotion,
            "interpretation": (
                "promote only when all three paired homology-cluster 95% intervals are "
                "strictly beneficial"
            ),
        },
    }


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _write_oof(path: Path, rows: Sequence[OofPrediction]) -> None:
    fields = (
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
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            row = asdict(item)
            row["max_train_identity"] = f"{item.max_train_identity:.12g}"
            row["probability"] = f"{item.probability:.12g}"
            writer.writerow(row)


def run_esm_oof(
    *,
    base_oof_path: str | Path,
    base_manifest_path: str | Path,
    fasta_path: str | Path,
    fasta_manifest_path: str | Path,
    embedding_matrix_path: str | Path,
    embedding_index_path: str | Path,
    embedding_manifest_path: str | Path,
    trust_receipt_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Fit fold-local logistic heads and write auditable ESM2 OOF predictions."""

    base_oof = Path(base_oof_path).resolve()
    base_manifest = Path(base_manifest_path).resolve()
    fasta = Path(fasta_path).resolve()
    fasta_manifest = Path(fasta_manifest_path).resolve()
    matrix = Path(embedding_matrix_path).resolve()
    index = Path(embedding_index_path).resolve()
    embedding_manifest = Path(embedding_manifest_path).resolve()
    trust_receipt = Path(trust_receipt_path).resolve()
    config = load_config(config_path)
    base_document = _verify_base_manifest(base_oof, base_manifest, config)
    examples = _read_examples(base_oof, config)
    _verify_fasta_chain(
        fasta=fasta,
        manifest_path=fasta_manifest,
        base_oof=base_oof,
        base_manifest=base_manifest,
        config=config,
        examples=examples,
    )
    embedding_document = _verify_embedding_manifest(
        matrix_path=matrix,
        index_path=index,
        manifest_path=embedding_manifest,
        trust_receipt_path=trust_receipt,
        fasta=fasta,
        fasta_manifest=fasta_manifest,
        config=config,
    )
    if embedding_document.get("records") != len({item.sequence_id for item in examples}):
        raise ValueError("embedding manifest record count differs from the base OOF")
    embeddings = _read_embeddings(
        matrix,
        index,
        dimension=config.embedding_dimension,
        examples=examples,
    )
    esm_predictions, fold_models = _make_predictions(examples, embeddings, config)
    unconverged = [item.heldout_fold for item in fold_models if not item.converged]
    if unconverged:
        raise RuntimeError(
            "ESM logistic head failed the configured convergence tolerance in fold(s): "
            f"{unconverged}"
        )
    base_predictions = _read_base_predictions(base_oof, examples)
    blends = _fixed_oof_blends(esm_predictions, base_predictions)
    candidate_predictions: dict[str, tuple[OofPrediction, ...]] = {
        MODEL_NAME: esm_predictions,
        **blends,
    }
    predictions = tuple(
        sorted(
            (row for rows in candidate_predictions.values() for row in rows),
            key=lambda item: (item.model, item.example_id),
        )
    )
    metrics = {
        name: _model_metrics(rows, config) for name, rows in sorted(candidate_predictions.items())
    }
    comparisons = _comparisons(candidate_predictions, base_predictions, config)

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty ESM OOF output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    predictions_out = output / "esm_oof_predictions.csv"
    models_out = output / "fold_models.json"
    metrics_out = output / "metrics.json"
    comparisons_out = output / "comparisons.json"
    manifest_out = output / "manifest.json"
    _write_oof(predictions_out, predictions)
    _write_json(
        models_out,
        {
            "model": MODEL_NAME,
            "feature_order": [
                "intercept",
                *(f"esm2_mean_{index:03d}" for index in range(config.embedding_dimension)),
                "gram_positive",
                "gram_negative",
                "gram_unknown",
            ],
            "scaling_policy": "embedding mean/std fit on non-heldout examples only",
            "fold_models": [asdict(item) for item in fold_models],
        },
    )
    _write_json(metrics_out, metrics)
    _write_json(comparisons_out, comparisons)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "esm2_gate1_homology_oof",
        "model": MODEL_NAME,
        "base_oof_sha256": _sha256(base_oof),
        "base_manifest_sha256": _sha256(base_manifest),
        "base_label_summary": base_document.get("label_summary"),
        "fasta_sha256": _sha256(fasta),
        "fasta_manifest_sha256": _sha256(fasta_manifest),
        "embedding_matrix_sha256": _sha256(matrix),
        "embedding_index_sha256": _sha256(index),
        "embedding_manifest_sha256": _sha256(embedding_manifest),
        "trust_receipt_sha256": _sha256(trust_receipt),
        "config_sha256": _sha256(config.path),
        "benchmark_code_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "fold_policy": "reuse frozen base Gate-1 homology folds without reassignment",
        "training_policy": (
            "fixed ESM2 embeddings; fold-local embedding scaling and L2 logistic fit on "
            "non-heldout folds only"
        ),
        "blend_policy": (
            "fixed arithmetic combinations of already-OOF predictions; no learned stacker"
        ),
        "upstream_pretraining": (
            "ESM2 UR50D self-supervised pretraining is label-independent, but exact sequence "
            "membership in its public corpus is not asserted"
        ),
        "examples": len(examples),
        "unique_sequences": len(embeddings),
        "folds": config.folds,
        "models": list(NEW_MODELS),
        "stored_comparison_models": list(BASE_MODELS),
        "all_fold_models_converged": all(item.converged for item in fold_models),
        "outputs": {
            predictions_out.name: _sha256(predictions_out),
            models_out.name: _sha256(models_out),
            metrics_out.name: _sha256(metrics_out),
            comparisons_out.name: _sha256(comparisons_out),
        },
    }
    _write_json(manifest_out, manifest)
    return manifest


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export frozen Gate-1 sequences for ESM2")
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models/esm2_oof.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def export_main(argv: Sequence[str] | None = None) -> int:
    args = build_export_parser().parse_args(argv)
    result = export_gate1_fasta(
        base_oof_path=args.base_oof,
        base_manifest_path=args.base_manifest,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--fasta-manifest", type=Path, required=True)
    parser.add_argument("--embedding-matrix", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--trust-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models/esm2_oof.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_esm_oof(
        base_oof_path=args.base_oof,
        base_manifest_path=args.base_manifest,
        fasta_path=args.fasta,
        fasta_manifest_path=args.fasta_manifest,
        embedding_matrix_path=args.embedding_matrix,
        embedding_index_path=args.embedding_index,
        embedding_manifest_path=args.embedding_manifest,
        trust_receipt_path=args.trust_receipt,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

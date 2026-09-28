"""Build a union-v1 activity-acquisition ledger from accepted OOF evidence.

This adapter is intentionally separate from :mod:`oracle_ledger`.  It consumes
the exact accepted homology-and-study-union Gate-1 artifact and the exact
accepted union-panel ESM2 embeddings, verifies both independent acceptance
receipts, and emits one sequence-level row only when all three replay
objectives have measured support.

The descriptor-logistic family supplies the objective means.  The ``std_*``
columns are the population standard deviation of the descriptor and homology
kNN family aggregates (for two values, half their absolute difference).  They
are an uncalibrated family-disagreement proxy, not posterior or aleatoric
uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import stat
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence
from amp_challenge.similarity import cluster_sequences

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIVERSITY_ALGORITHM = "global_alignment_identity_single_link_v1"
_ARTIFACT = "union_gate1_activity_replay_ledger_v1"
_MEAN_MODEL = "descriptor_logistic"
_DISAGREEMENT_MODEL = "homology_knn"
_ENSEMBLE_CHECK_MODEL = "equal_weight_ensemble"
_MODELS = (_MEAN_MODEL, _DISAGREEMENT_MODEL, _ENSEMBLE_CHECK_MODEL)
_OBJECTIVES = ("broad_spectrum", "gram_positive", "gram_negative")
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
_EMBEDDING_INDEX_FIELDS = ("row_index", "sequence_id", "sequence", "length")


@dataclass(frozen=True, slots=True)
class Gate1EvidenceConfig:
    producer_job_id: int
    audit_job_id: int
    git_commit: str
    publication_top_sha256: str
    semantic_top_sha256: str
    oof_sha256: str
    independent_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class EmbeddingEvidenceConfig:
    producer_job_id: int
    audit_job_id: int
    git_commit: str
    publication_top_sha256: str
    semantic_top_sha256: str
    index_sha256: str
    manifest_sha256: str
    matrix_sha256: str
    tensor_data_sha256: str
    sequence_ids_sha256: str
    independent_receipt_sha256: str
    records: int
    dimensions: int


@dataclass(frozen=True, slots=True)
class UnionOracleLedgerConfig:
    path: Path
    minimum_observations: int
    homology_identity_threshold: float
    diversity_identity_threshold: float
    expected_folds: int
    expected_oof_rows: int
    expected_examples: int
    expected_sequences: int
    expected_candidates: int
    expected_candidate_contexts: int
    expected_candidate_source_observations: int
    expected_candidate_ids_sha256: str
    expected_unique_target_counts_2_to_7: tuple[int, ...]
    expected_examples_by_fold: tuple[int, ...]
    expected_candidates_by_fold: tuple[int, ...]
    canonical_target_grams: Mapping[str, str]
    gate1: Gate1EvidenceConfig
    embeddings: EmbeddingEvidenceConfig


@dataclass(frozen=True, slots=True)
class UnionOracleLedgerExecution:
    ledger_path: Path
    diversity_components_path: Path
    summary_path: Path
    candidate_count: int
    excluded_sequence_count: int
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class _Example:
    example_id: str
    sequence_id: str
    sequence: str
    canonical_target: str
    gram: str
    label: int
    source_observations: int
    fold: int
    homology_component_id: str
    union_component_id: str
    max_train_identity: float
    predictions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _EmbeddingIndexRow:
    row_index: int
    sequence_id: str
    sequence: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    while True:
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link: {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _snapshot_regular(path: str | Path, *, label: str) -> _Snapshot:
    candidate = Path(path)
    _reject_symlink_chain(candidate, label=label)
    metadata = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {candidate}")
    payload = candidate.read_bytes()
    if _fingerprint(candidate) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while it was read: {candidate}")
    return _Snapshot(
        path=candidate.resolve(strict=True),
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=(
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ),
    )


def _assert_unchanged(snapshot: _Snapshot, *, label: str) -> None:
    if (
        _fingerprint(snapshot.path) != snapshot.fingerprint
        or _sha256_file(snapshot.path) != snapshot.sha256
    ):
        raise RuntimeError(f"{label} changed after validation: {snapshot.path}")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"union replay ledger config {field!r} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"union replay ledger config {field!r} must be a full Git commit")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"union replay ledger config {field!r} must be a positive integer")
    return value


def _count_vector(value: object, *, field: str, size: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(
            f"union replay ledger config {field!r} must contain {size} positive integers"
        )
    return tuple(value)


def _nonnegative_count_vector(value: object, *, field: str, size: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(
            f"union replay ledger config {field!r} must contain {size} non-negative integers"
        )
    return tuple(value)


def _load_evidence_config(
    value: object,
    *,
    field: str,
    embedding: bool,
) -> Gate1EvidenceConfig | EmbeddingEvidenceConfig:
    if not isinstance(value, dict):
        raise ValueError(f"union replay ledger config [{field}] must be a table")
    common = {
        "producer_job_id",
        "audit_job_id",
        "git_commit",
        "publication_top_sha256",
        "semantic_top_sha256",
        "independent_receipt_sha256",
    }
    embedding_only = {
        "index_sha256",
        "manifest_sha256",
        "matrix_sha256",
        "tensor_data_sha256",
        "sequence_ids_sha256",
        "records",
        "dimensions",
    }
    gate_only = {"oof_sha256"}
    expected = common | (embedding_only if embedding else gate_only)
    if set(value) != expected:
        raise ValueError(
            f"union replay ledger config [{field}] keys must be exactly {sorted(expected)}"
        )
    common_values = {
        "producer_job_id": _positive_integer(
            value["producer_job_id"], field=f"{field}.producer_job_id"
        ),
        "audit_job_id": _positive_integer(value["audit_job_id"], field=f"{field}.audit_job_id"),
        "git_commit": _require_commit(value["git_commit"], field=f"{field}.git_commit"),
        "publication_top_sha256": _require_sha256(
            value["publication_top_sha256"], field=f"{field}.publication_top_sha256"
        ),
        "semantic_top_sha256": _require_sha256(
            value["semantic_top_sha256"], field=f"{field}.semantic_top_sha256"
        ),
        "independent_receipt_sha256": _require_sha256(
            value["independent_receipt_sha256"],
            field=f"{field}.independent_receipt_sha256",
        ),
    }
    if not embedding:
        return Gate1EvidenceConfig(
            **common_values,
            oof_sha256=_require_sha256(value["oof_sha256"], field=f"{field}.oof_sha256"),
        )
    return EmbeddingEvidenceConfig(
        **common_values,
        index_sha256=_require_sha256(value["index_sha256"], field=f"{field}.index_sha256"),
        manifest_sha256=_require_sha256(value["manifest_sha256"], field=f"{field}.manifest_sha256"),
        matrix_sha256=_require_sha256(value["matrix_sha256"], field=f"{field}.matrix_sha256"),
        tensor_data_sha256=_require_sha256(
            value["tensor_data_sha256"], field=f"{field}.tensor_data_sha256"
        ),
        sequence_ids_sha256=_require_sha256(
            value["sequence_ids_sha256"], field=f"{field}.sequence_ids_sha256"
        ),
        records=_positive_integer(value["records"], field=f"{field}.records"),
        dimensions=_positive_integer(value["dimensions"], field=f"{field}.dimensions"),
    )


def _load_union_oracle_ledger_config_snapshot(
    path: str | Path,
) -> tuple[UnionOracleLedgerConfig, _Snapshot]:
    snapshot = _snapshot_regular(path, label="union replay ledger config")
    try:
        document = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("union replay ledger config is not valid UTF-8 TOML") from error
    config_path = snapshot.path
    expected_keys = {
        "schema_version",
        "artifact",
        "mean_model",
        "disagreement_model",
        "ensemble_check_model",
        "objectives",
        "minimum_observations",
        "homology_identity_threshold",
        "diversity_identity_threshold",
        "diversity_algorithm",
        "expected_folds",
        "expected_oof_rows",
        "expected_examples",
        "expected_sequences",
        "expected_candidates",
        "expected_candidate_contexts",
        "expected_candidate_source_observations",
        "expected_candidate_ids_sha256",
        "expected_unique_target_counts_2_to_7",
        "expected_examples_by_fold",
        "expected_candidates_by_fold",
        "canonical_target_grams",
        "gate1",
        "embeddings",
    }
    if set(document) != expected_keys:
        raise ValueError(f"union replay ledger config keys must be exactly {sorted(expected_keys)}")
    if document["schema_version"] != 1 or document["artifact"] != _ARTIFACT:
        raise ValueError("union replay ledger config schema/artifact is not union-v1")
    if (
        document["mean_model"] != _MEAN_MODEL
        or document["disagreement_model"] != _DISAGREEMENT_MODEL
        or document["ensemble_check_model"] != _ENSEMBLE_CHECK_MODEL
        or document["objectives"] != list(_OBJECTIVES)
    ):
        raise ValueError("union replay ledger model/objective contract changed")
    if document["diversity_algorithm"] != _DIVERSITY_ALGORITHM:
        raise ValueError("union replay ledger diversity algorithm changed")

    folds = _positive_integer(document["expected_folds"], field="expected_folds")
    minimum = _positive_integer(document["minimum_observations"], field="minimum_observations")
    homology_threshold = document["homology_identity_threshold"]
    diversity_threshold = document["diversity_identity_threshold"]
    for name, value in (
        ("homology_identity_threshold", homology_threshold),
        ("diversity_identity_threshold", diversity_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"union replay ledger config {name!r} must be in (0, 1]")
    if float(homology_threshold) != 0.80:
        raise ValueError("union-v1 replay homology_identity_threshold must be exactly 0.80")
    if float(diversity_threshold) != 0.70:
        raise ValueError("union-v1 replay diversity_identity_threshold must be exactly 0.70")

    target_grams = document["canonical_target_grams"]
    if (
        not isinstance(target_grams, dict)
        or not target_grams
        or any(not isinstance(key, str) or not key for key in target_grams)
        or any(value not in {"positive", "negative"} for value in target_grams.values())
    ):
        raise ValueError("[canonical_target_grams] must map targets to positive/negative")

    gate1 = _load_evidence_config(document["gate1"], field="gate1", embedding=False)
    embeddings = _load_evidence_config(document["embeddings"], field="embeddings", embedding=True)
    assert isinstance(gate1, Gate1EvidenceConfig)
    assert isinstance(embeddings, EmbeddingEvidenceConfig)
    config = UnionOracleLedgerConfig(
        path=config_path,
        minimum_observations=minimum,
        homology_identity_threshold=float(homology_threshold),
        diversity_identity_threshold=float(diversity_threshold),
        expected_folds=folds,
        expected_oof_rows=_positive_integer(
            document["expected_oof_rows"], field="expected_oof_rows"
        ),
        expected_examples=_positive_integer(
            document["expected_examples"], field="expected_examples"
        ),
        expected_sequences=_positive_integer(
            document["expected_sequences"], field="expected_sequences"
        ),
        expected_candidates=_positive_integer(
            document["expected_candidates"], field="expected_candidates"
        ),
        expected_candidate_contexts=_positive_integer(
            document["expected_candidate_contexts"], field="expected_candidate_contexts"
        ),
        expected_candidate_source_observations=_positive_integer(
            document["expected_candidate_source_observations"],
            field="expected_candidate_source_observations",
        ),
        expected_candidate_ids_sha256=_require_sha256(
            document["expected_candidate_ids_sha256"], field="expected_candidate_ids_sha256"
        ),
        expected_unique_target_counts_2_to_7=_nonnegative_count_vector(
            document["expected_unique_target_counts_2_to_7"],
            field="expected_unique_target_counts_2_to_7",
            size=6,
        ),
        expected_examples_by_fold=_count_vector(
            document["expected_examples_by_fold"], field="expected_examples_by_fold", size=folds
        ),
        expected_candidates_by_fold=_count_vector(
            document["expected_candidates_by_fold"],
            field="expected_candidates_by_fold",
            size=folds,
        ),
        canonical_target_grams=dict(sorted(target_grams.items())),
        gate1=gate1,
        embeddings=embeddings,
    )
    return config, snapshot


def load_union_oracle_ledger_config(path: str | Path) -> UnionOracleLedgerConfig:
    """Load the strict union-v1 ledger and accepted-evidence contract."""

    config, _ = _load_union_oracle_ledger_config_snapshot(path)
    return config


def _validate_twin_location(path: str | Path, *, job_id: int, label: str) -> Path:
    candidate = Path(path)
    _reject_symlink_chain(candidate, label=label)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir() or resolved.name not in {"0", "1"}:
        raise ValueError(f"{label} must be one producer twin directory named 0 or 1")
    if resolved.parent.name != str(job_id):
        raise ValueError(f"{label} is not nested under accepted producer job {job_id}")
    return resolved


def _validate_receipt_location(
    path: str | Path,
    *,
    producer_job_id: int,
    audit_job_id: int,
    label: str,
) -> _Snapshot:
    candidate = Path(path)
    if candidate.name != f"independent-verification-{audit_job_id}.json":
        raise ValueError(f"{label} filename does not bind audit job {audit_job_id}")
    if candidate.parent.name != str(producer_job_id):
        raise ValueError(f"{label} is not nested under producer job {producer_job_id}")
    return _snapshot_regular(candidate, label=label)


def _parse_manifest(snapshot: _Snapshot, *, label: str) -> Mapping[str, str]:
    payload = snapshot.payload
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty canonical LF-terminated text")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must contain ASCII only") from error
    entries: dict[str, str] = {}
    previous = ""
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        if match is None:
            raise ValueError(f"{label} contains a malformed checksum line")
        digest, raw_path = match.groups()
        logical = PurePosixPath(raw_path)
        if logical.is_absolute() or ".." in logical.parts or raw_path in entries:
            raise ValueError(f"{label} contains an unsafe or duplicate path: {raw_path}")
        if previous and raw_path <= previous:
            raise ValueError(f"{label} paths must be strictly sorted")
        entries[raw_path] = digest
        previous = raw_path
    return entries


def _expect_hash(observed: str, expected: str, *, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")


def _verify_gate1_inputs(
    twin_path: str | Path,
    receipt_path: str | Path,
    *,
    config: UnionOracleLedgerConfig,
) -> tuple[_Snapshot, _Snapshot, tuple[_Snapshot, ...]]:
    root = _validate_twin_location(
        twin_path, job_id=config.gate1.producer_job_id, label="Gate-1 union twin"
    )
    top = _snapshot_regular(root / "SHA256SUMS", label="Gate-1 publication manifest")
    semantic = _snapshot_regular(root / "gate1" / "SHA256SUMS", label="Gate-1 semantic manifest")
    oof = _snapshot_regular(root / "gate1" / "oof_predictions.csv", label="Gate-1 OOF CSV")
    receipt = _validate_receipt_location(
        receipt_path,
        producer_job_id=config.gate1.producer_job_id,
        audit_job_id=config.gate1.audit_job_id,
        label="Gate-1 independent receipt",
    )
    for observed, expected, label in (
        (top.sha256, config.gate1.publication_top_sha256, "Gate-1 publication manifest"),
        (semantic.sha256, config.gate1.semantic_top_sha256, "Gate-1 semantic manifest"),
        (oof.sha256, config.gate1.oof_sha256, "Gate-1 OOF CSV"),
        (receipt.sha256, config.gate1.independent_receipt_sha256, "Gate-1 receipt"),
    ):
        _expect_hash(observed, expected, label=label)
    top_entries = _parse_manifest(top, label="Gate-1 publication manifest")
    semantic_entries = _parse_manifest(semantic, label="Gate-1 semantic manifest")
    if top_entries.get("gate1/SHA256SUMS") != semantic.sha256:
        raise ValueError("Gate-1 publication manifest does not bind the semantic manifest")
    if top_entries.get("gate1/oof_predictions.csv") != oof.sha256:
        raise ValueError("Gate-1 publication manifest does not bind the OOF CSV")
    if semantic_entries.get("oof_predictions.csv") != oof.sha256:
        raise ValueError("Gate-1 semantic manifest does not bind the OOF CSV")
    _validate_gate1_receipt(receipt.payload, config=config)
    return oof, receipt, (top, semantic, oof, receipt)


def _verify_embedding_inputs(
    twin_path: str | Path,
    receipt_path: str | Path,
    *,
    config: UnionOracleLedgerConfig,
) -> tuple[_Snapshot, _Snapshot, _Snapshot, tuple[_Snapshot, ...]]:
    root = _validate_twin_location(
        twin_path,
        job_id=config.embeddings.producer_job_id,
        label="ESM2 union embedding twin",
    )
    top = _snapshot_regular(root / "SHA256SUMS", label="embedding publication manifest")
    semantic = _snapshot_regular(
        root / "embeddings" / "SHA256SUMS", label="embedding semantic manifest"
    )
    index = _snapshot_regular(root / "embeddings" / "embedding_index.csv", label="embedding index")
    manifest = _snapshot_regular(
        root / "embeddings" / "embedding_manifest.json", label="embedding manifest"
    )
    matrix = _snapshot_regular(root / "embeddings" / "embeddings.npy", label="embedding matrix")
    receipt = _validate_receipt_location(
        receipt_path,
        producer_job_id=config.embeddings.producer_job_id,
        audit_job_id=config.embeddings.audit_job_id,
        label="embedding independent receipt",
    )
    for observed, expected, label in (
        (top.sha256, config.embeddings.publication_top_sha256, "embedding publication manifest"),
        (semantic.sha256, config.embeddings.semantic_top_sha256, "embedding semantic manifest"),
        (index.sha256, config.embeddings.index_sha256, "embedding index"),
        (manifest.sha256, config.embeddings.manifest_sha256, "embedding manifest"),
        (matrix.sha256, config.embeddings.matrix_sha256, "embedding matrix"),
        (receipt.sha256, config.embeddings.independent_receipt_sha256, "embedding receipt"),
    ):
        _expect_hash(observed, expected, label=label)
    top_entries = _parse_manifest(top, label="embedding publication manifest")
    semantic_entries = _parse_manifest(semantic, label="embedding semantic manifest")
    expected_top = {
        "embeddings/SHA256SUMS": semantic.sha256,
        "embeddings/embedding_index.csv": index.sha256,
        "embeddings/embedding_manifest.json": manifest.sha256,
        "embeddings/embeddings.npy": matrix.sha256,
    }
    expected_semantic = {
        "embedding_index.csv": index.sha256,
        "embedding_manifest.json": manifest.sha256,
        "embeddings.npy": matrix.sha256,
    }
    if any(top_entries.get(name) != digest for name, digest in expected_top.items()):
        raise ValueError("embedding publication manifest does not bind every required artifact")
    if any(semantic_entries.get(name) != digest for name, digest in expected_semantic.items()):
        raise ValueError("embedding semantic manifest does not bind every required artifact")
    _validate_embedding_receipt(receipt.payload, config=config)
    return index, matrix, receipt, (top, semantic, index, manifest, matrix, receipt)


def _json_mapping(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _all_true_checks(value: object, *, label: str) -> None:
    if not isinstance(value, dict) or not value or any(item is not True for item in value.values()):
        raise ValueError(f"{label} does not record a non-empty all-true check set")


def _validate_gate1_receipt(payload: bytes, *, config: UnionOracleLedgerConfig) -> None:
    receipt = _json_mapping(payload, label="Gate-1 independent receipt")
    artifacts = receipt.get("artifact_sha256")
    census = receipt.get("census")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact")
        != "gate1_context_activity_homology_study_union_v1_independent_verification"
        or receipt.get("status") != "passed"
        or receipt.get("git_commit") != config.gate1.git_commit
        or receipt.get("publication_top_manifest_sha256") != config.gate1.publication_top_sha256
        or receipt.get("gate1_top_manifest_sha256") != config.gate1.semantic_top_sha256
        or not isinstance(artifacts, dict)
        or artifacts.get("oof_predictions.csv") != config.gate1.oof_sha256
        or not isinstance(census, dict)
        or census.get("context_examples") != config.expected_examples
        or census.get("modeled_sequences") != config.expected_sequences
    ):
        raise ValueError("Gate-1 independent receipt does not match the frozen union-v1 contract")
    _all_true_checks(receipt.get("checks"), label="Gate-1 independent receipt")


def _validate_embedding_receipt(payload: bytes, *, config: UnionOracleLedgerConfig) -> None:
    receipt = _json_mapping(payload, label="embedding independent receipt")
    artifacts = receipt.get("artifact_sha256")
    acceptance = receipt.get("acceptance_scope")
    tensor = receipt.get("tensor")
    panel = receipt.get("panel")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact") != "gate1_union_esm2_embeddings_v1_independent_verification"
        or receipt.get("status") != "accepted_for_downstream_embedding_feature_input_only"
        or receipt.get("git_commit") != config.embeddings.git_commit
        or not isinstance(artifacts, dict)
        or artifacts.get("SHA256SUMS") != config.embeddings.publication_top_sha256
        or artifacts.get("embeddings/SHA256SUMS") != config.embeddings.semantic_top_sha256
        or artifacts.get("embeddings/embedding_index.csv") != config.embeddings.index_sha256
        or artifacts.get("embeddings/embedding_manifest.json") != config.embeddings.manifest_sha256
        or artifacts.get("embeddings/embeddings.npy") != config.embeddings.matrix_sha256
        or not isinstance(acceptance, dict)
        or acceptance.get("embedding_feature_input_eligible") is not True
        or acceptance.get("embeddings_verified") is not True
        or acceptance.get("full_embedding_matrix_independently_recomputed") is not True
        or not isinstance(tensor, dict)
        or tensor.get("dtype") != "float32"
        or tensor.get("layout") != "C_contiguous"
        or tensor.get("shape") != [config.embeddings.records, config.embeddings.dimensions]
        or tensor.get("tensor_data_sha256") != config.embeddings.tensor_data_sha256
        or not isinstance(panel, dict)
        or panel.get("sequence_ids_sha256") != config.embeddings.sequence_ids_sha256
    ):
        raise ValueError(
            "embedding independent receipt does not match the frozen union-v1 contract"
        )
    _all_true_checks(receipt.get("checks"), label="embedding independent receipt")


def _finite(value: str | None, *, field: str, row: int) -> float:
    try:
        parsed = float(value or "")
    except ValueError as error:
        raise ValueError(f"OOF row {row} has invalid {field!r}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"OOF row {row} has non-finite {field!r}")
    return parsed


def _csv_reader(payload: bytes, *, label: str) -> csv.DictReader:
    if (
        not payload
        or not payload.endswith(b"\n")
        or b"\r" in payload
        or payload.startswith(b"\xef\xbb\xbf")
    ):
        raise ValueError(f"{label} must be canonical UTF-8 with LF termination and no BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be valid UTF-8") from error
    return csv.DictReader(io.StringIO(text, newline=""))


def _read_examples(payload: bytes, config: UnionOracleLedgerConfig) -> tuple[_Example, ...]:
    reader = _csv_reader(payload, label="Gate-1 OOF CSV")
    if tuple(reader.fieldnames or ()) != _OOF_FIELDS:
        raise ValueError("Gate-1 OOF CSV does not have the exact union-v1 schema")
    metadata: dict[str, tuple[object, ...]] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    row_count = 0
    for row_number, row in enumerate(reader, start=2):
        row_count += 1
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"OOF row {row_number} does not match the exact CSV width")
        model = row["model"]
        if model not in _MODELS:
            raise ValueError(f"OOF row {row_number} has unexpected model {model!r}")
        example_id = row["example_id"]
        if _SHA256.fullmatch(example_id) is None or row["assay_context_id"] != example_id:
            raise ValueError(f"OOF row {row_number} has invalid example/assay-context identity")
        try:
            sequence = canonicalize_sequence(row["sequence"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"OOF row {row_number} has invalid sequence: {error}") from error
        sequence_id = row["sequence_id"]
        if sequence_id != canonical_sequence_id(sequence):
            raise ValueError(f"OOF row {row_number} sequence_id does not match sequence")
        target = row["canonical_target"]
        gram = row["gram"]
        if config.canonical_target_grams.get(target) != gram:
            raise ValueError(f"OOF row {row_number} has an unknown or Gram-inconsistent target")
        label_raw = row["label"]
        if label_raw not in {"0", "1"}:
            raise ValueError(f"OOF row {row_number} label must be 0 or 1")
        try:
            source_observations = int(row["source_observations"])
            fold = int(row["fold"])
        except ValueError as error:
            raise ValueError(f"OOF row {row_number} has invalid integer metadata") from error
        if source_observations <= 0 or not 0 <= fold < config.expected_folds:
            raise ValueError(f"OOF row {row_number} has out-of-range integer metadata")
        homology_component = row["homology_component_id"]
        union_component = row["union_component_id"]
        if (
            _SHA256.fullmatch(homology_component) is None
            or _SHA256.fullmatch(union_component) is None
        ):
            raise ValueError(f"OOF row {row_number} has invalid component identity")
        max_identity = _finite(
            row["max_train_identity"], field="max_train_identity", row=row_number
        )
        probability = _finite(row["probability"], field="probability", row=row_number)
        if not 0 <= max_identity < config.homology_identity_threshold or not 0 <= probability <= 1:
            raise ValueError(f"OOF row {row_number} violates probability/identity bounds")
        current = (
            sequence_id,
            sequence,
            target,
            gram,
            int(label_raw),
            source_observations,
            fold,
            homology_component,
            union_component,
            max_identity,
        )
        previous = metadata.setdefault(example_id, current)
        if previous != current:
            raise ValueError(f"OOF example {example_id} has inconsistent model metadata")
        if model in predictions[example_id]:
            raise ValueError(f"OOF example {example_id} duplicates model {model}")
        predictions[example_id][model] = probability

    if row_count != config.expected_oof_rows:
        raise ValueError(
            f"Gate-1 OOF row census changed: expected {config.expected_oof_rows}, got {row_count}"
        )
    if len(metadata) != config.expected_examples:
        raise ValueError("Gate-1 OOF example census changed")
    examples: list[_Example] = []
    for example_id in sorted(metadata):
        if set(predictions[example_id]) != set(_MODELS):
            raise ValueError(f"OOF example {example_id} lacks exact three-model support")
        descriptor = predictions[example_id][_MEAN_MODEL]
        knn = predictions[example_id][_DISAGREEMENT_MODEL]
        ensemble = predictions[example_id][_ENSEMBLE_CHECK_MODEL]
        if not math.isclose(ensemble, 0.5 * (descriptor + knn), rel_tol=0.0, abs_tol=2e-15):
            raise ValueError(f"OOF example {example_id} has an inconsistent equal-weight ensemble")
        (
            sequence_id,
            sequence,
            target,
            gram,
            label,
            source_observations,
            fold,
            homology_component,
            union_component,
            max_identity,
        ) = metadata[example_id]
        examples.append(
            _Example(
                example_id=example_id,
                sequence_id=str(sequence_id),
                sequence=str(sequence),
                canonical_target=str(target),
                gram=str(gram),
                label=int(label),
                source_observations=int(source_observations),
                fold=int(fold),
                homology_component_id=str(homology_component),
                union_component_id=str(union_component),
                max_train_identity=float(max_identity),
                predictions=dict(predictions[example_id]),
            )
        )
    examples_by_fold = Counter(item.fold for item in examples)
    if tuple(examples_by_fold[index] for index in range(config.expected_folds)) != (
        config.expected_examples_by_fold
    ):
        raise ValueError("Gate-1 OOF per-fold example census changed")
    if len({item.sequence_id for item in examples}) != config.expected_sequences:
        raise ValueError("Gate-1 OOF modeled-sequence census changed")
    if {item.canonical_target for item in examples} != set(config.canonical_target_grams):
        raise ValueError("Gate-1 OOF canonical-target census changed")
    return tuple(examples)


def _read_embedding_index(
    payload: bytes, *, config: UnionOracleLedgerConfig
) -> tuple[_EmbeddingIndexRow, ...]:
    reader = _csv_reader(payload, label="embedding index")
    if tuple(reader.fieldnames or ()) != _EMBEDDING_INDEX_FIELDS:
        raise ValueError("embedding index does not have the exact accepted schema")
    rows: list[_EmbeddingIndexRow] = []
    previous_sequence_id = ""
    for csv_row, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"embedding index row {csv_row} has the wrong width")
        try:
            row_index = int(row["row_index"])
            length = int(row["length"])
        except ValueError as error:
            raise ValueError(f"embedding index row {csv_row} has invalid integers") from error
        if row_index != len(rows):
            raise ValueError("embedding index row_index is not contiguous from zero")
        try:
            sequence = canonicalize_sequence(row["sequence"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"embedding index row {csv_row} has invalid sequence: {error}"
            ) from error
        sequence_id = row["sequence_id"]
        if sequence_id != canonical_sequence_id(sequence) or length != len(sequence):
            raise ValueError(f"embedding index row {csv_row} sequence identity/length mismatch")
        if previous_sequence_id and sequence_id <= previous_sequence_id:
            raise ValueError("embedding index must be strictly ordered by sequence_id")
        previous_sequence_id = sequence_id
        rows.append(
            _EmbeddingIndexRow(row_index=row_index, sequence_id=sequence_id, sequence=sequence)
        )
    if len(rows) != config.embeddings.records:
        raise ValueError("embedding index record census changed")
    sequence_ids_sha256 = _sha256_bytes(
        "".join(f"{item.sequence_id}\n" for item in rows).encode("ascii")
    )
    if sequence_ids_sha256 != config.embeddings.sequence_ids_sha256:
        raise ValueError("embedding index ordered sequence-ID digest changed")
    return tuple(rows)


def _read_embedding_matrix(payload: bytes, *, config: UnionOracleLedgerConfig) -> np.ndarray:
    try:
        matrix = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("embedding matrix is not a safe NumPy array") from error
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (config.embeddings.records, config.embeddings.dimensions)
        or matrix.dtype != np.dtype("<f4")
        or not matrix.flags.c_contiguous
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("embedding matrix shape/dtype/layout/finiteness contract changed")
    if _sha256_bytes(matrix.tobytes(order="C")) != config.embeddings.tensor_data_sha256:
        raise ValueError("embedding matrix tensor-data SHA-256 changed")
    matrix.setflags(write=False)
    return matrix


def _diversity_component_id(sequences: Sequence[str], *, threshold: float) -> str:
    sequence_ids = sorted(canonical_sequence_id(sequence) for sequence in sequences)
    payload = _canonical_json(
        {
            "algorithm": _DIVERSITY_ALGORITHM,
            "domain": "amp_challenge.union_v1.replay.diversity_cluster",
            "identity_threshold_hex": threshold.hex(),
            "schema_version": 1,
            "sequence_ids": sequence_ids,
        }
    ).encode("ascii")
    return "div70:" + hashlib.sha256(payload).hexdigest()


def _build_diversity_components(
    sequences: Sequence[str], *, threshold: float, fold: int
) -> tuple[Mapping[str, str], tuple[Mapping[str, object], ...]]:
    components = cluster_sequences(sequences, identity_threshold=threshold)
    by_sequence: dict[str, str] = {}
    rows: list[Mapping[str, object]] = []
    for component in components:
        component_id = _diversity_component_id(component, threshold=threshold)
        sequence_ids = sorted(canonical_sequence_id(sequence) for sequence in component)
        for sequence in component:
            if sequence in by_sequence:
                raise RuntimeError("diversity clustering assigned one sequence more than once")
            by_sequence[sequence] = component_id
        rows.append(
            {
                "diversity_cluster_id_70": component_id,
                "identity_algorithm": _DIVERSITY_ALGORITHM,
                "identity_threshold": threshold,
                "prediction_fold": fold,
                "replay_round": f"fold-{fold}",
                "sequence_count": len(component),
                "sequence_ids": sequence_ids,
            }
        )
    if set(by_sequence) != set(sequences):
        raise RuntimeError("diversity clustering did not cover the replay-fold candidate pool")
    return by_sequence, tuple(rows)


def _objective_examples(examples: Sequence[_Example], objective: str) -> list[_Example]:
    if objective == "broad_spectrum":
        return list(examples)
    gram = "positive" if objective == "gram_positive" else "negative"
    return [item for item in examples if item.gram == gram]


def _format_csv_value(value: object) -> object:
    return format(value, ".17g") if isinstance(value, float) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_csv_value(row[field]) for field in fields})


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_union_oracle_replay_ledger(
    gate1_twin: str | Path,
    gate1_independent_receipt: str | Path,
    embedding_twin: str | Path,
    embedding_independent_receipt: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> UnionOracleLedgerExecution:
    """Create the non-start-aware three-objective union-v1 replay ledger."""

    config, config_snapshot = _load_union_oracle_ledger_config_snapshot(config_path)
    gate_oof, gate_receipt, gate_snapshots = _verify_gate1_inputs(
        gate1_twin, gate1_independent_receipt, config=config
    )
    embedding_index_snapshot, embedding_matrix_snapshot, embedding_receipt, embedding_snapshots = (
        _verify_embedding_inputs(embedding_twin, embedding_independent_receipt, config=config)
    )
    examples = _read_examples(gate_oof.payload, config)
    index_rows = _read_embedding_index(embedding_index_snapshot.payload, config=config)
    matrix = _read_embedding_matrix(embedding_matrix_snapshot.payload, config=config)

    by_sequence: dict[str, list[_Example]] = defaultdict(list)
    for example in examples:
        by_sequence[example.sequence_id].append(example)
    oof_sequence_map = {
        sequence_id: sequence_examples[0].sequence
        for sequence_id, sequence_examples in by_sequence.items()
    }
    embedding_sequence_map = {item.sequence_id: item.sequence for item in index_rows}
    if oof_sequence_map != embedding_sequence_map:
        missing = sorted(set(oof_sequence_map) - set(embedding_sequence_map))
        extra = sorted(set(embedding_sequence_map) - set(oof_sequence_map))
        mismatched = sorted(
            sequence_id
            for sequence_id in set(oof_sequence_map) & set(embedding_sequence_map)
            if oof_sequence_map[sequence_id] != embedding_sequence_map[sequence_id]
        )
        raise ValueError(
            "accepted OOF and ESM2 index sequence joins differ: "
            f"missing={missing[:3]} extra={extra[:3]} mismatched={mismatched[:3]}"
        )
    embedding_row_by_id = {item.sequence_id: item.row_index for item in index_rows}
    rows: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    support: dict[str, list[int]] = {objective: [] for objective in _OBJECTIVES}
    candidate_contexts = 0
    candidate_source_observations = 0
    candidate_unique_target_counts: Counter[int] = Counter()
    for sequence_id in sorted(by_sequence):
        sequence_examples = by_sequence[sequence_id]
        sequence = sequence_examples[0].sequence
        folds = {item.fold for item in sequence_examples}
        homology_components = {item.homology_component_id for item in sequence_examples}
        union_components = {item.union_component_id for item in sequence_examples}
        maximum_identities = {item.max_train_identity for item in sequence_examples}
        sequences = {item.sequence for item in sequence_examples}
        if any(
            len(values) != 1
            for values in (
                folds,
                homology_components,
                union_components,
                maximum_identities,
                sequences,
            )
        ):
            raise ValueError(f"sequence {sequence_id} has inconsistent accepted OOF metadata")

        objective_values: dict[str, tuple[float, float, float, int]] = {}
        missing_objectives: list[str] = []
        for objective in _OBJECTIVES:
            selected = _objective_examples(sequence_examples, objective)
            if len(selected) < config.minimum_observations:
                missing_objectives.append(objective)
                continue
            descriptor_mean = float(
                np.mean([item.predictions[_MEAN_MODEL] for item in selected], dtype=np.float64)
            )
            knn_mean = float(
                np.mean(
                    [item.predictions[_DISAGREEMENT_MODEL] for item in selected], dtype=np.float64
                )
            )
            family_disagreement_sd = 0.5 * abs(descriptor_mean - knn_mean)
            objective_values[objective] = (
                descriptor_mean,
                family_disagreement_sd,
                float(np.mean([item.label for item in selected], dtype=np.float64)),
                len(selected),
            )
        if missing_objectives:
            exclusions["missing:" + ",".join(missing_objectives)] += 1
            continue

        fold = next(iter(folds))
        embedding_row = embedding_row_by_id[sequence_id]
        row: dict[str, object] = {
            "sequence_id": sequence_id,
            "sequence": sequence,
            **{f"mean_{objective}": objective_values[objective][0] for objective in _OBJECTIVES},
            **{f"std_{objective}": objective_values[objective][1] for objective in _OBJECTIVES},
            "novelty": 1.0 - next(iter(maximum_identities)),
        }
        for column in range(config.embeddings.dimensions):
            row[f"embedding_{column:03d}"] = float(matrix[embedding_row, column])
        row.update(
            {
                "homology_component_id": next(iter(homology_components)),
                "union_component_id": next(iter(union_components)),
                "esm_source_row_index": embedding_row,
                "eligible": "true",
                "replay_round": f"fold-{fold}",
                "prediction_scope": "out_of_fold",
                "prediction_fold": fold,
                "outcome_fold": fold,
            }
        )
        for objective in _OBJECTIVES:
            row[f"outcome_{objective}"] = objective_values[objective][2]
            row[f"n_{objective}"] = objective_values[objective][3]
            support[objective].append(objective_values[objective][3])
        rows.append(row)
        candidate_contexts += len(sequence_examples)
        candidate_source_observations += sum(item.source_observations for item in sequence_examples)
        unique_target_count = len({item.canonical_target for item in sequence_examples})
        candidate_unique_target_counts[unique_target_count] += 1

    rows.sort(key=lambda row: str(row["sequence"]))
    if len(rows) != config.expected_candidates:
        raise ValueError(
            f"union replay candidate census changed: expected {config.expected_candidates}, "
            f"got {len(rows)}"
        )
    candidates_by_fold = Counter(int(row["prediction_fold"]) for row in rows)
    if tuple(candidates_by_fold[index] for index in range(config.expected_folds)) != (
        config.expected_candidates_by_fold
    ):
        raise ValueError("union replay per-fold candidate census changed")
    candidate_ids_sha256 = _sha256_bytes(
        "".join(
            f"{row['sequence_id']}\n"
            for row in sorted(rows, key=lambda row: str(row["sequence_id"]))
        ).encode("ascii")
    )
    if candidate_ids_sha256 != config.expected_candidate_ids_sha256:
        raise ValueError("union replay sorted candidate sequence-ID digest changed")
    if candidate_contexts != config.expected_candidate_contexts:
        raise ValueError("union replay observed-context census changed")
    if candidate_source_observations != config.expected_candidate_source_observations:
        raise ValueError("union replay contributing source-observation census changed")
    observed_unique_target_counts = tuple(
        candidate_unique_target_counts[count] for count in range(2, 8)
    )
    if observed_unique_target_counts != config.expected_unique_target_counts_2_to_7:
        raise ValueError("union replay sparse unique-target distribution changed")
    if sum(candidate_unique_target_counts.values()) != sum(observed_unique_target_counts):
        raise ValueError("union replay candidate has a unique-target count outside 2..7")

    diversity_by_fold_sequence: dict[tuple[int, str], str] = {}
    diversity_rows_list: list[Mapping[str, object]] = []
    for fold in range(config.expected_folds):
        fold_sequences = tuple(
            str(row["sequence"]) for row in rows if int(row["prediction_fold"]) == fold
        )
        fold_mapping, fold_components = _build_diversity_components(
            fold_sequences,
            threshold=config.diversity_identity_threshold,
            fold=fold,
        )
        diversity_by_fold_sequence.update(
            {(fold, sequence): component for sequence, component in fold_mapping.items()}
        )
        diversity_rows_list.extend(fold_components)
    diversity_rows = tuple(
        sorted(
            diversity_rows_list,
            key=lambda row: (int(row["prediction_fold"]), str(row["diversity_cluster_id_70"])),
        )
    )
    for row in rows:
        diversity_component = diversity_by_fold_sequence[
            (int(row["prediction_fold"]), str(row["sequence"]))
        ]
        row["cluster_id"] = diversity_component
        row["diversity_cluster_id_70"] = diversity_component

    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty/non-directory output: {output}")
    for snapshot in (config_snapshot, *gate_snapshots, *embedding_snapshots):
        _assert_unchanged(snapshot, label="accepted replay input")
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "candidate_ledger.csv"
    diversity_path = output / "diversity_components.jsonl"
    summary_path = output / "ledger_summary.json"
    embedding_fields = [f"embedding_{column:03d}" for column in range(config.embeddings.dimensions)]
    fields = [
        "sequence_id",
        "sequence",
        *(f"mean_{objective}" for objective in _OBJECTIVES),
        *(f"std_{objective}" for objective in _OBJECTIVES),
        "novelty",
        *embedding_fields,
        "cluster_id",
        "diversity_cluster_id_70",
        "homology_component_id",
        "union_component_id",
        "esm_source_row_index",
        "eligible",
        "replay_round",
        "prediction_scope",
        "prediction_fold",
        "outcome_fold",
        *(f"outcome_{objective}" for objective in _OBJECTIVES),
        *(f"n_{objective}" for objective in _OBJECTIVES),
    ]
    _write_csv(ledger_path, rows, fields)
    diversity_path.write_text(
        "".join(_canonical_json(row) + "\n" for row in diversity_rows), encoding="utf-8"
    )
    component_sizes = [int(row["sequence_count"]) for row in diversity_rows]
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact": _ARTIFACT,
        "status": "built_for_union_v1_development_replay",
        "config_sha256": config_snapshot.sha256,
        "input_sha256": {
            "gate1_oof_predictions": gate_oof.sha256,
            "gate1_independent_receipt": gate_receipt.sha256,
            "esm2_embedding_index": embedding_index_snapshot.sha256,
            "esm2_embedding_matrix": embedding_matrix_snapshot.sha256,
            "esm2_independent_receipt": embedding_receipt.sha256,
        },
        "aggregation_unit": (
            "one accepted assay_context_id; source_observations is audit metadata and does not "
            "multiply context weight"
        ),
        "objective_mean": "descriptor_logistic context-probability arithmetic mean",
        "objective_std": (
            "population SD across descriptor_logistic and homology_knn context-aggregate "
            "means; uncalibrated family disagreement, not posterior or aleatoric SD"
        ),
        "objectives": list(_OBJECTIVES),
        "input_examples": len(examples),
        "input_sequences": len(by_sequence),
        "candidate_sequences": len(rows),
        "candidate_sequence_ids_sha256": candidate_ids_sha256,
        "candidate_observed_contexts": candidate_contexts,
        "candidate_source_observations": candidate_source_observations,
        "candidate_unique_target_counts": {
            str(count): candidate_unique_target_counts[count] for count in range(2, 8)
        },
        "excluded_sequences": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "candidates_by_fold": {
            str(index): candidates_by_fold[index] for index in range(config.expected_folds)
        },
        "objective_observation_support": {
            objective: {
                "minimum": min(values),
                "median": float(np.median(values)),
                "maximum": max(values),
            }
            for objective, values in support.items()
        },
        "embedding": {
            "model": "esm2_t6_8M_UR50D",
            "dimensions": config.embeddings.dimensions,
            "join": "exact full-panel sequence_id_and_sequence",
            "fallback": "none",
        },
        "diversity_clustering": {
            "algorithm": _DIVERSITY_ALGORITHM,
            "identity_threshold": config.diversity_identity_threshold,
            "scope": "support_eligible_candidates_clustered_separately_within_each_replay_fold",
            "single_link_bridges_through_ineligible_or_other_fold_sequences": False,
            "components": len(diversity_rows),
            "largest_component": max(component_sizes),
            "candidate_cluster_field": "cluster_id",
            "explicit_audit_field": "diversity_cluster_id_70",
            "upstream_split_components_are_not_diversity_clusters": True,
        },
        "artifacts": {
            "candidate_ledger.csv": _sha256_file(ledger_path),
            "diversity_components.jsonl": _sha256_file(diversity_path),
        },
        "outcome_limit": (
            "observed-context activity-fraction proxies over sparse accepted target coverage; "
            "not full-panel broad-spectrum outcomes"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for snapshot in (config_snapshot, *gate_snapshots, *embedding_snapshots):
        _assert_unchanged(snapshot, label="accepted replay input")
    return UnionOracleLedgerExecution(
        ledger_path=ledger_path,
        diversity_components_path=diversity_path,
        summary_path=summary_path,
        candidate_count=len(rows),
        excluded_sequence_count=sum(exclusions.values()),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-twin", type=Path, required=True)
    parser.add_argument("--gate1-independent-receipt", type=Path, required=True)
    parser.add_argument("--embedding-twin", type=Path, required=True)
    parser.add_argument("--embedding-independent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = build_union_oracle_replay_ledger(
            args.gate1_twin,
            args.gate1_independent_receipt,
            args.embedding_twin,
            args.embedding_independent_receipt,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"AMP union replay-ledger error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidates": execution.candidate_count,
                "excluded_sequences": execution.excluded_sequence_count,
                "ledger": str(execution.ledger_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

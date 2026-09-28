"""Independently reconstruct and verify frozen union-v1 replay twins.

This verifier intentionally imports none of the production ledger, replay,
acquisition, or sequence-similarity implementations.  It is bounded to the
accepted union-v1 OOF/ESM evidence and its frozen non-start-aware replay.  It
emits a path-free scientific receipt and a path-free operational receipt; only
the latter records the Slurm job and node identities needed for attestation.
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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
_ARTIFACT = "union_gate1_activity_replay_bundle_v1"
_INDEPENDENT_ARTIFACT = "union_gate1_activity_replay_v1_independent_verification"
_OPERATIONAL_ARTIFACT = "union_gate1_activity_replay_v1_operational_verification"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_OBJECTIVES = ("broad_spectrum", "gram_positive", "gram_negative")
_STRATEGY_ORDER = (
    "exploit",
    "pareto",
    "diversity",
    "novelty",
    "uncertainty",
    "start_ucb",
    "ucb",
    "random",
)
_POLICIES = ("mixed", "lcb", "mean", "random")
_SEEDS = (17, 42, 91, 137, 271)
_GROUPS = tuple(f"fold-{index}" for index in range(5))
_FOLD_COUNTS = (202, 126, 112, 97, 113)
_CANDIDATE_ID_SHA256 = "990ef4b248f7d95a9864fef7da5bc648a5e6468c1adc74ba87c3af38dedb21c9"
_DIVERSITY_ALGORITHM = "global_alignment_identity_single_link_v1"
_TARGET_GRAMS = {
    "acinetobacter_baumannii": "negative",
    "enterococcus_faecalis": "positive",
    "enterococcus_faecium": "positive",
    "escherichia_coli": "negative",
    "klebsiella_pneumoniae": "negative",
    "pseudomonas_aeruginosa": "negative",
    "staphylococcus_aureus": "positive",
}
_SELECTION_CONFIG = "configs/acquisition/replay_union_v1_mixed.toml"
_LEDGER_CONFIG = "configs/evaluation/oracle_activity_ledger_union_v1.toml"
_REPLAY_CONFIG = "configs/evaluation/oracle_activity_replay_union_v1.toml"
_UPSTREAM_FILE_MODES = {
    "embeddings/SHA256SUMS": 0o400,
    "embeddings/embedding_index.csv": 0o400,
    "embeddings/embedding_manifest.json": 0o400,
    "embeddings/embeddings.npy": 0o400,
    "embeddings/independent-receipt.json": 0o400,
    "embeddings/semantic-SHA256SUMS": 0o400,
    "gate1/SHA256SUMS": 0o400,
    "gate1/independent-receipt.json": 0o444,
    "gate1/oof_predictions.csv": 0o444,
    "gate1/semantic-SHA256SUMS": 0o444,
}

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
_CODE_PATHS = (
    "cluster/slurm/audit_union_activity_replay_v1_twins.sbatch",
    "cluster/slurm/run_union_activity_replay_v1_twins.sbatch",
    "cluster/slurm/union_activity_replay_v1_hardened.sh",
    "cluster/slurm/validate_union_activity_replay_v1.sbatch",
    _SELECTION_CONFIG,
    _LEDGER_CONFIG,
    _REPLAY_CONFIG,
    "pyproject.toml",
    "src/amp_challenge/__init__.py",
    "src/amp_challenge/acquisition/__init__.py",
    "src/amp_challenge/acquisition/mixer.py",
    "src/amp_challenge/constants.py",
    "src/amp_challenge/evaluation/__init__.py",
    "src/amp_challenge/evaluation/replay.py",
    "src/amp_challenge/evaluation/union_oracle_ledger.py",
    "src/amp_challenge/evaluation/union_replay_bundle.py",
    "src/amp_challenge/evaluation/verify_union_replay_bundle.py",
    "src/amp_challenge/sequences.py",
    "src/amp_challenge/similarity.py",
    "src/amp_challenge/workflows/__init__.py",
    "src/amp_challenge/workflows/select.py",
    "uv.lock",
)
_BUNDLE_FILES = (
    "CODE_SHA256SUMS",
    "FROZEN_INPUT_SHA256SUMS",
    "SHA256SUMS",
    "ledger/candidate_ledger.csv",
    "ledger/diversity_components.jsonl",
    "ledger/ledger_summary.json",
    "manifest.json",
    "replay/runs.csv",
    "replay/selections.csv",
    "replay/summary.json",
)


class VerificationError(ValueError):
    """Raised when any frozen evidence or reconstruction check fails."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    sequence_id: str
    sequence: str
    target: str
    gram: str
    label: int
    observations: int
    fold: int
    homology_id: str
    union_id: str
    max_train_identity: float
    predictions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Candidate:
    sequence_id: str
    sequence: str
    means: tuple[float, float, float]
    stds: tuple[float, float, float]
    outcomes: tuple[float, float, float]
    support: tuple[int, int, int]
    novelty: float
    embedding: np.ndarray
    esm_row: int
    homology_id: str
    union_id: str
    fold: int
    cluster_id: str


@dataclass(frozen=True, slots=True)
class Selection:
    indices: tuple[int, ...]
    reasons: tuple[str, ...]
    conservative: tuple[float, ...]
    acquisition: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VerificationExecution:
    receipt_dir: Path
    independent_receipt: Path
    operational_receipt: Path
    independent_sha256: str
    operational_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> tuple[int, int, int, int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_mode,
        metadata.st_ctime_ns,
    )


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    candidate = path.absolute()
    while True:
        _require(not candidate.is_symlink(), f"{label} traverses a symbolic link")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _snapshot(
    path: str | Path,
    *,
    label: str,
    required_mode: int | None = None,
) -> Snapshot:
    candidate = Path(path)
    _reject_symlink_chain(candidate, label=label)
    metadata = candidate.stat(follow_symlinks=False)
    _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    if required_mode is not None:
        _require(
            stat.S_IMODE(metadata.st_mode) == required_mode,
            f"{label} is not mode {required_mode:04o}",
        )
    payload = candidate.read_bytes()
    fingerprint = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_mode,
        metadata.st_ctime_ns,
    )
    _require(_fingerprint(candidate) == fingerprint, f"{label} changed while being read")
    return Snapshot(
        path=candidate.resolve(strict=True),
        payload=payload,
        sha256=_sha256_bytes(payload),
        fingerprint=fingerprint,
    )


def _unchanged(snapshot: Snapshot, *, label: str) -> None:
    _require(
        _fingerprint(snapshot.path) == snapshot.fingerprint
        and _sha256_file(snapshot.path) == snapshot.sha256,
        f"{label} changed during independent verification",
    )


def _json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{label} is not valid UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _parse_sha256sums(payload: bytes, *, label: str) -> Mapping[str, str]:
    _require(
        payload and payload.endswith(b"\n") and b"\r" not in payload,
        f"{label} is not canonical text",
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not ASCII") from error
    result: dict[str, str] = {}
    previous = ""
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        _require(match is not None, f"{label} has a malformed line")
        digest, raw_path = match.groups()
        logical = PurePosixPath(raw_path)
        _require(
            not logical.is_absolute()
            and ".." not in logical.parts
            and raw_path not in result
            and (not previous or raw_path > previous),
            f"{label} has an unsafe, duplicate, or unsorted path",
        )
        result[raw_path] = digest
        previous = raw_path
    _require(bool(result), f"{label} is empty")
    return result


def _canonical_sequence(raw: str) -> str:
    _require(isinstance(raw, str), "sequence is not text")
    sequence = "".join(raw.split()).upper()
    _require(8 <= len(sequence) <= 50, "sequence length is outside 8..50")
    _require(not (set(sequence) - _ALPHABET), "sequence uses a non-standard residue")
    return sequence


def _sequence_id(sequence: str) -> str:
    return hashlib.sha256(_canonical_sequence(sequence).encode("ascii")).hexdigest()


def _csv_reader(payload: bytes, *, label: str) -> csv.DictReader:
    _require(
        payload
        and payload.endswith(b"\n")
        and b"\r" not in payload
        and not payload.startswith(b"\xef\xbb\xbf"),
        f"{label} is not canonical LF-terminated UTF-8",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    return csv.DictReader(io.StringIO(text, newline=""))


def _finite(raw: str, *, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} is not numeric") from error
    _require(math.isfinite(value), f"{label} is not finite")
    return value


def _load_documents(
    snapshots: Mapping[str, Snapshot],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    logical_paths = (_SELECTION_CONFIG, _LEDGER_CONFIG, _REPLAY_CONFIG)
    documents: list[Mapping[str, Any]] = []
    for logical, label in zip(logical_paths, ("selection", "ledger", "replay"), strict=True):
        try:
            value = tomllib.loads(snapshots[logical].payload.decode("utf-8"))
        except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise VerificationError(f"{label} config is invalid TOML") from error
        _require(isinstance(value, dict), f"{label} config is not a table")
        documents.append(value)
    selection, ledger, replay = documents
    _require(
        set(selection)
        == {
            "objectives",
            "batch_size",
            "objective_weights",
            "risk_beta",
            "ucb_beta",
            "quality_floor_quantile",
            "diversity_quality_weight",
            "max_per_cluster",
            "strict_cluster_cap",
            "seed",
            "strategy_mix",
            "specialist_quotas",
        },
        "selection config root keys differ from the frozen contract",
    )
    _require(
        set(ledger)
        == {
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
        },
        "ledger config root keys differ from the frozen contract",
    )
    _require(
        set(replay)
        == {
            "schema_version",
            "selection_config",
            "objectives",
            "policies",
            "seeds",
            "group_column",
            "prediction_scope_column",
            "required_prediction_scope",
            "prediction_fold_column",
            "outcome_fold_column",
            "outcome_thresholds",
        },
        "replay config root keys differ from the frozen contract",
    )
    gate_evidence = ledger.get("gate1")
    embedding_evidence = ledger.get("embeddings")
    _require(
        isinstance(gate_evidence, dict)
        and set(gate_evidence)
        == {
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "publication_top_sha256",
            "semantic_top_sha256",
            "oof_sha256",
            "independent_receipt_sha256",
        }
        and isinstance(embedding_evidence, dict)
        and set(embedding_evidence)
        == {
            "producer_job_id",
            "audit_job_id",
            "git_commit",
            "publication_top_sha256",
            "semantic_top_sha256",
            "index_sha256",
            "manifest_sha256",
            "matrix_sha256",
            "tensor_data_sha256",
            "sequence_ids_sha256",
            "independent_receipt_sha256",
            "records",
            "dimensions",
        },
        "ledger evidence subtable keys differ from the frozen contract",
    )
    _require(
        ledger.get("schema_version") == 1
        and ledger.get("artifact") == "union_gate1_activity_replay_ledger_v1"
        and ledger.get("mean_model") == "descriptor_logistic"
        and ledger.get("disagreement_model") == "homology_knn"
        and ledger.get("ensemble_check_model") == "equal_weight_ensemble"
        and tuple(ledger.get("objectives", ())) == _OBJECTIVES
        and ledger.get("minimum_observations") == 1
        and ledger.get("homology_identity_threshold") == 0.80
        and ledger.get("diversity_identity_threshold") == 0.70
        and ledger.get("diversity_algorithm") == _DIVERSITY_ALGORITHM
        and ledger.get("expected_folds") == 5
        and ledger.get("expected_oof_rows") == 7476
        and ledger.get("expected_examples") == 2492
        and ledger.get("expected_sequences") == 952
        and ledger.get("expected_candidates") == 650
        and ledger.get("expected_candidate_contexts") == 1967
        and ledger.get("expected_candidate_source_observations") == 2062
        and tuple(ledger.get("expected_unique_target_counts_2_to_7", ()))
        == (364, 213, 56, 16, 0, 1)
        and tuple(ledger.get("expected_examples_by_fold", ())) == (546, 486, 485, 487, 488)
        and tuple(ledger.get("expected_candidates_by_fold", ())) == _FOLD_COUNTS
        and ledger.get("expected_candidate_ids_sha256") == _CANDIDATE_ID_SHA256
        and ledger.get("canonical_target_grams") == _TARGET_GRAMS
        and embedding_evidence.get("records") == 952
        and embedding_evidence.get("dimensions") == 320,
        "ledger config differs from the frozen union-v1 contract",
    )
    _require(
        replay.get("schema_version") == 1
        and tuple(replay.get("objectives", ())) == _OBJECTIVES
        and tuple(replay.get("policies", ())) == _POLICIES
        and tuple(replay.get("seeds", ())) == _SEEDS
        and replay.get("group_column") == "replay_round"
        and replay.get("prediction_scope_column") == "prediction_scope"
        and replay.get("required_prediction_scope") == "out_of_fold"
        and replay.get("prediction_fold_column") == "prediction_fold"
        and replay.get("outcome_fold_column") == "outcome_fold"
        and replay.get("outcome_thresholds") == {name: 0.5 for name in _OBJECTIVES},
        "replay config differs from the frozen union-v1 contract",
    )
    _require(
        tuple(selection.get("objectives", ())) == _OBJECTIVES
        and selection.get("batch_size") == 10
        and tuple(selection.get("objective_weights", ())) == (1.0, 1.0, 1.0)
        and selection.get("risk_beta") == 1.0
        and selection.get("ucb_beta") == 1.0
        and selection.get("quality_floor_quantile") == 0.20
        and selection.get("diversity_quality_weight") == 0.35
        and selection.get("max_per_cluster") == 2
        and selection.get("strict_cluster_cap") is True
        and selection.get("seed") == 42
        and selection.get("strategy_mix")
        == {
            "exploit": 0.50,
            "diversity": 0.20,
            "pareto": 0.10,
            "uncertainty": 0.10,
            "random": 0.10,
        }
        and selection.get("specialist_quotas") == {}
        and "max_per_start" not in selection
        and "rollouts_per_start" not in selection,
        "selection config differs from the frozen non-start-aware max-per-cluster=2 policy",
    )
    _require(
        replay.get("selection_config") == "../acquisition/replay_union_v1_mixed.toml",
        "replay config selects a different policy",
    )
    return selection, ledger, replay


def _all_true(value: object, *, label: str) -> None:
    _require(
        isinstance(value, dict) and bool(value) and all(item is True for item in value.values()),
        f"{label} does not contain a non-empty all-true check set",
    )


def _verify_gate_receipt(snapshot: Snapshot, ledger: Mapping[str, Any]) -> None:
    evidence = ledger.get("gate1")
    _require(isinstance(evidence, dict), "ledger config Gate-1 evidence is absent")
    receipt = _json_object(snapshot.payload, label="Gate-1 independent receipt")
    artifacts = receipt.get("artifact_sha256")
    census = receipt.get("census")
    _require(
        snapshot.sha256 == evidence.get("independent_receipt_sha256")
        and receipt.get("schema_version") == 1
        and receipt.get("artifact")
        == "gate1_context_activity_homology_study_union_v1_independent_verification"
        and receipt.get("status") == "passed"
        and receipt.get("git_commit") == evidence.get("git_commit")
        and receipt.get("publication_top_manifest_sha256") == evidence.get("publication_top_sha256")
        and receipt.get("gate1_top_manifest_sha256") == evidence.get("semantic_top_sha256")
        and isinstance(artifacts, dict)
        and artifacts.get("oof_predictions.csv") == evidence.get("oof_sha256")
        and isinstance(census, dict)
        and census.get("context_examples") == 2492
        and census.get("modeled_sequences") == 952,
        "Gate-1 independent receipt differs from the accepted contract",
    )
    _all_true(receipt.get("checks"), label="Gate-1 independent receipt")


def _verify_embedding_receipt(snapshot: Snapshot, ledger: Mapping[str, Any]) -> None:
    evidence = ledger.get("embeddings")
    _require(isinstance(evidence, dict), "ledger config embedding evidence is absent")
    receipt = _json_object(snapshot.payload, label="embedding independent receipt")
    artifacts = receipt.get("artifact_sha256")
    acceptance = receipt.get("acceptance_scope")
    tensor = receipt.get("tensor")
    panel = receipt.get("panel")
    _require(
        snapshot.sha256 == evidence.get("independent_receipt_sha256")
        and receipt.get("schema_version") == 1
        and receipt.get("artifact") == "gate1_union_esm2_embeddings_v1_independent_verification"
        and receipt.get("status") == "accepted_for_downstream_embedding_feature_input_only"
        and receipt.get("git_commit") == evidence.get("git_commit")
        and isinstance(artifacts, dict)
        and artifacts.get("SHA256SUMS") == evidence.get("publication_top_sha256")
        and artifacts.get("embeddings/SHA256SUMS") == evidence.get("semantic_top_sha256")
        and artifacts.get("embeddings/embedding_index.csv") == evidence.get("index_sha256")
        and artifacts.get("embeddings/embedding_manifest.json") == evidence.get("manifest_sha256")
        and artifacts.get("embeddings/embeddings.npy") == evidence.get("matrix_sha256")
        and isinstance(acceptance, dict)
        and acceptance.get("embedding_feature_input_eligible") is True
        and acceptance.get("embeddings_verified") is True
        and acceptance.get("full_embedding_matrix_independently_recomputed") is True
        and isinstance(tensor, dict)
        and tensor.get("dtype") == "float32"
        and tensor.get("layout") == "C_contiguous"
        and tensor.get("shape") == [952, 320]
        and tensor.get("tensor_data_sha256") == evidence.get("tensor_data_sha256")
        and isinstance(panel, dict)
        and panel.get("sequence_ids_sha256") == evidence.get("sequence_ids_sha256"),
        "embedding independent receipt differs from the accepted contract",
    )
    _all_true(receipt.get("checks"), label="embedding independent receipt")


def _require_two_twins(
    paths: Sequence[str | Path], *, job_id: int, label: str
) -> tuple[Path, Path]:
    _require(len(paths) == 2, f"{label} requires exactly two twins")
    resolved: list[Path] = []
    for expected_name, raw in enumerate(paths):
        candidate = Path(raw)
        _reject_symlink_chain(candidate, label=label)
        path = candidate.resolve(strict=True)
        _require(
            path.is_dir()
            and not path.is_symlink()
            and path.name == str(expected_name)
            and path.parent.name == str(job_id),
            f"{label} {expected_name} is not under the accepted producer job",
        )
        resolved.append(path)
    return resolved[0], resolved[1]


def _verify_upstream_inputs(
    gate_twins: Sequence[str | Path],
    gate_receipt_path: str | Path,
    embedding_twins: Sequence[str | Path],
    embedding_receipt_path: str | Path,
    *,
    ledger: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]], tuple[Snapshot, ...]]:
    gate_config = ledger.get("gate1")
    embedding_config = ledger.get("embeddings")
    _require(isinstance(gate_config, dict), "Gate-1 evidence config is missing")
    _require(isinstance(embedding_config, dict), "embedding evidence config is missing")
    _require(
        gate_config.get("producer_job_id") == 223248
        and gate_config.get("audit_job_id") == 223250
        and embedding_config.get("producer_job_id") == 223330
        and embedding_config.get("audit_job_id") == 223334,
        "accepted upstream job identities changed",
    )
    gates = _require_two_twins(gate_twins, job_id=223248, label="Gate-1 input")
    embeddings = _require_two_twins(
        embedding_twins,
        job_id=223330,
        label="embedding input",
    )
    gate_receipt = _snapshot(
        gate_receipt_path,
        label="Gate-1 receipt",
        required_mode=_UPSTREAM_FILE_MODES["gate1/independent-receipt.json"],
    )
    embedding_receipt = _snapshot(
        embedding_receipt_path,
        label="embedding receipt",
        required_mode=_UPSTREAM_FILE_MODES["embeddings/independent-receipt.json"],
    )
    _require(
        gate_receipt.path.name == "independent-verification-223250.json"
        and gate_receipt.path.parent.name == "223248",
        "Gate-1 receipt path does not bind its accepted audit",
    )
    _require(
        embedding_receipt.path.name == "independent-verification-223334.json"
        and embedding_receipt.path.parent.name == "223330",
        "embedding receipt path does not bind its accepted audit",
    )
    _verify_gate_receipt(gate_receipt, ledger)
    _verify_embedding_receipt(embedding_receipt, ledger)

    paired: list[Mapping[str, Snapshot]] = []
    all_snapshots: list[Snapshot] = [gate_receipt, embedding_receipt]
    for index in range(2):
        gate_top = _snapshot(
            gates[index] / "SHA256SUMS",
            label="Gate-1 top manifest",
            required_mode=_UPSTREAM_FILE_MODES["gate1/SHA256SUMS"],
        )
        gate_semantic = _snapshot(
            gates[index] / "gate1" / "SHA256SUMS",
            label="Gate-1 semantic manifest",
            required_mode=_UPSTREAM_FILE_MODES["gate1/semantic-SHA256SUMS"],
        )
        oof = _snapshot(
            gates[index] / "gate1" / "oof_predictions.csv",
            label="Gate-1 OOF predictions",
            required_mode=_UPSTREAM_FILE_MODES["gate1/oof_predictions.csv"],
        )
        embedding_top = _snapshot(
            embeddings[index] / "SHA256SUMS",
            label="embedding top manifest",
            required_mode=_UPSTREAM_FILE_MODES["embeddings/SHA256SUMS"],
        )
        embedding_semantic = _snapshot(
            embeddings[index] / "embeddings" / "SHA256SUMS",
            label="embedding semantic manifest",
            required_mode=_UPSTREAM_FILE_MODES["embeddings/semantic-SHA256SUMS"],
        )
        embedding_index = _snapshot(
            embeddings[index] / "embeddings" / "embedding_index.csv",
            label="embedding index",
            required_mode=_UPSTREAM_FILE_MODES["embeddings/embedding_index.csv"],
        )
        embedding_manifest = _snapshot(
            embeddings[index] / "embeddings" / "embedding_manifest.json",
            label="embedding manifest",
            required_mode=_UPSTREAM_FILE_MODES["embeddings/embedding_manifest.json"],
        )
        embedding_matrix = _snapshot(
            embeddings[index] / "embeddings" / "embeddings.npy",
            label="embedding matrix",
            required_mode=_UPSTREAM_FILE_MODES["embeddings/embeddings.npy"],
        )
        _require(
            gate_top.sha256 == gate_config.get("publication_top_sha256")
            and gate_semantic.sha256 == gate_config.get("semantic_top_sha256")
            and oof.sha256 == gate_config.get("oof_sha256")
            and embedding_top.sha256 == embedding_config.get("publication_top_sha256")
            and embedding_semantic.sha256 == embedding_config.get("semantic_top_sha256")
            and embedding_index.sha256 == embedding_config.get("index_sha256")
            and embedding_manifest.sha256 == embedding_config.get("manifest_sha256")
            and embedding_matrix.sha256 == embedding_config.get("matrix_sha256"),
            "accepted upstream artifact hash changed",
        )
        gate_top_entries = _parse_sha256sums(gate_top.payload, label="Gate-1 top manifest")
        gate_semantic_entries = _parse_sha256sums(
            gate_semantic.payload,
            label="Gate-1 semantic manifest",
        )
        _require(
            gate_top_entries.get("gate1/SHA256SUMS") == gate_semantic.sha256
            and gate_top_entries.get("gate1/oof_predictions.csv") == oof.sha256
            and gate_semantic_entries.get("oof_predictions.csv") == oof.sha256,
            "Gate-1 manifests do not bind OOF evidence",
        )
        embedding_top_entries = _parse_sha256sums(
            embedding_top.payload,
            label="embedding top manifest",
        )
        embedding_semantic_entries = _parse_sha256sums(
            embedding_semantic.payload,
            label="embedding semantic manifest",
        )
        expected_top = {
            "embeddings/SHA256SUMS": embedding_semantic.sha256,
            "embeddings/embedding_index.csv": embedding_index.sha256,
            "embeddings/embedding_manifest.json": embedding_manifest.sha256,
            "embeddings/embeddings.npy": embedding_matrix.sha256,
        }
        expected_semantic = {
            "embedding_index.csv": embedding_index.sha256,
            "embedding_manifest.json": embedding_manifest.sha256,
            "embeddings.npy": embedding_matrix.sha256,
        }
        _require(
            all(embedding_top_entries.get(name) == digest for name, digest in expected_top.items())
            and all(
                embedding_semantic_entries.get(name) == digest
                for name, digest in expected_semantic.items()
            ),
            "embedding manifests do not bind raw index/matrix evidence",
        )
        current = {
            "embeddings/SHA256SUMS": embedding_top,
            "embeddings/embedding_index.csv": embedding_index,
            "embeddings/embedding_manifest.json": embedding_manifest,
            "embeddings/embeddings.npy": embedding_matrix,
            "embeddings/independent-receipt.json": embedding_receipt,
            "embeddings/semantic-SHA256SUMS": embedding_semantic,
            "gate1/SHA256SUMS": gate_top,
            "gate1/independent-receipt.json": gate_receipt,
            "gate1/oof_predictions.csv": oof,
            "gate1/semantic-SHA256SUMS": gate_semantic,
        }
        paired.append(current)
        all_snapshots.extend(
            (
                gate_top,
                gate_semantic,
                oof,
                embedding_top,
                embedding_semantic,
                embedding_index,
                embedding_manifest,
                embedding_matrix,
            )
        )
    for logical in paired[0]:
        _require(
            paired[0][logical].payload == paired[1][logical].payload,
            f"accepted upstream twins differ at {logical}",
        )
    return (paired[0], paired[1]), tuple(all_snapshots)


def _verify_bundle_tree(
    root: Path,
    *,
    code_snapshots: Mapping[str, Snapshot],
    git_commit: str,
    frozen_inputs: Mapping[str, Snapshot],
) -> tuple[Mapping[str, Snapshot], tuple[Snapshot, ...]]:
    _reject_symlink_chain(root, label="replay bundle")
    resolved = root.resolve(strict=True)
    _require(resolved.is_dir() and not resolved.is_symlink(), "replay bundle is not a directory")
    for path in resolved.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        _require(
            not path.is_symlink()
            and (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)),
            "replay bundle contains a symbolic or non-regular entry",
        )
    observed_files = sorted(
        path.relative_to(resolved).as_posix() for path in resolved.rglob("*") if path.is_file()
    )
    observed_dirs = sorted(
        path.relative_to(resolved).as_posix() for path in resolved.rglob("*") if path.is_dir()
    )
    _require(observed_files == list(_BUNDLE_FILES), "replay bundle file inventory changed")
    _require(observed_dirs == ["ledger", "replay"], "replay bundle directory inventory changed")
    _require(stat.S_IMODE(resolved.stat().st_mode) == 0o555, "replay bundle root is not mode 0555")
    snapshots: dict[str, Snapshot] = {}
    for relative in _BUNDLE_FILES:
        snapshots[relative] = _snapshot(
            resolved / relative,
            label=f"bundle artifact {relative}",
            required_mode=0o444,
        )
    for directory in (resolved / "ledger", resolved / "replay"):
        _require(stat.S_IMODE(directory.stat().st_mode) == 0o555, "bundle subdirectory is not 0555")

    top = _parse_sha256sums(snapshots["SHA256SUMS"].payload, label="bundle SHA256SUMS")
    expected_top = {
        relative: snapshots[relative].sha256
        for relative in _BUNDLE_FILES
        if relative != "SHA256SUMS"
    }
    _require(top == expected_top, "bundle top manifest does not bind the exact nine payloads")

    expected_code = {relative: code_snapshots[relative].sha256 for relative in _CODE_PATHS}
    observed_code = _parse_sha256sums(
        snapshots["CODE_SHA256SUMS"].payload,
        label="bundle CODE_SHA256SUMS",
    )
    _require(observed_code == expected_code, "bundle code manifest differs from reviewed code")
    expected_inputs = {name: snapshot.sha256 for name, snapshot in frozen_inputs.items()}
    observed_inputs = _parse_sha256sums(
        snapshots["FROZEN_INPUT_SHA256SUMS"].payload,
        label="bundle FROZEN_INPUT_SHA256SUMS",
    )
    _require(observed_inputs == expected_inputs, "bundle frozen-input manifest differs")
    manifest = _json_object(snapshots["manifest.json"].payload, label="bundle manifest")
    expected_manifest = {
        "artifact": _ARTIFACT,
        "automatic_production_eligible": False,
        "candidate_count": 650,
        "code_sha256sums_sha256": snapshots["CODE_SHA256SUMS"].sha256,
        "decision_scope": "development_activity_acquisition_replay_only",
        "embedding_dimensions": 320,
        "frozen_input_sha256sums_sha256": snapshots["FROZEN_INPUT_SHA256SUMS"].sha256,
        "git_commit": git_commit,
        "manual_review_required": True,
        "max_per_diversity_cluster": 2,
        "replay": {
            "run_count": 100,
            "selection_count": 1000,
            "summary_sha256": snapshots["replay/summary.json"].sha256,
        },
        "schema_version": 1,
        "status": "completed_not_automatically_promoted",
    }
    _require(
        manifest == expected_manifest
        and snapshots["manifest.json"].payload
        == _canonical_json_bytes(expected_manifest, pretty=True),
        "bundle manifest violates the fixed non-promotion contract",
    )
    forbidden = (str(resolved), "/lustre/", "SLURM_JOB_ID", "node_name", '"job_id"')
    for relative, snapshot in snapshots.items():
        if relative.endswith((".json", ".jsonl", ".csv")):
            text = snapshot.payload.decode("utf-8")
            _require(
                not any(value in text for value in forbidden),
                f"bundle payload {relative} leaks dynamic execution metadata",
            )
    return snapshots, tuple(snapshots.values())


def _verify_bundles(
    bundle_paths: Sequence[str | Path],
    *,
    code_snapshots: Mapping[str, Snapshot],
    git_commit: str,
    upstream: tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
) -> tuple[
    tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
    tuple[Snapshot, ...],
]:
    _require(len(bundle_paths) == 2, "exactly two replay bundles are required")
    checked: list[Mapping[str, Snapshot]] = []
    all_snapshots: list[Snapshot] = []
    for index, raw in enumerate(bundle_paths):
        root = Path(raw)
        _require(root.name == str(index), "replay bundle arguments must be ordered twin 0, twin 1")
        snapshots, owned = _verify_bundle_tree(
            root,
            code_snapshots=code_snapshots,
            git_commit=git_commit,
            frozen_inputs=upstream[index],
        )
        checked.append(snapshots)
        all_snapshots.extend(owned)
    for relative in _BUNDLE_FILES:
        _require(
            checked[0][relative].payload == checked[1][relative].payload,
            f"replay twins differ at {relative}",
        )
    return (checked[0], checked[1]), tuple(all_snapshots)


def _read_examples(snapshot: Snapshot, ledger: Mapping[str, Any]) -> tuple[Example, ...]:
    targets = ledger.get("canonical_target_grams")
    _require(isinstance(targets, dict) and len(targets) == 7, "target/Gram contract changed")
    reader = _csv_reader(snapshot.payload, label="Gate-1 OOF predictions")
    _require(tuple(reader.fieldnames or ()) == _OOF_FIELDS, "Gate-1 OOF schema changed")
    metadata: dict[str, tuple[object, ...]] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    row_count = 0
    for row_number, row in enumerate(reader, start=2):
        row_count += 1
        _require(
            None not in row and all(value is not None for value in row.values()),
            f"OOF row {row_number} has the wrong width",
        )
        model = row["model"]
        example_id = row["example_id"]
        _require(model in _MODELS, f"OOF row {row_number} has an unknown model")
        _require(
            _SHA256.fullmatch(example_id) is not None and row["assay_context_id"] == example_id,
            f"OOF row {row_number} has invalid example identity",
        )
        sequence = _canonical_sequence(row["sequence"])
        sequence_id = row["sequence_id"]
        _require(
            sequence == row["sequence"] and sequence_id == _sequence_id(sequence),
            f"OOF row {row_number} has invalid sequence identity",
        )
        target = row["canonical_target"]
        gram = row["gram"]
        _require(
            targets.get(target) == gram, f"OOF row {row_number} has inconsistent Gram metadata"
        )
        _require(row["label"] in {"0", "1"}, f"OOF row {row_number} has invalid label")
        try:
            observations = int(row["source_observations"])
            fold = int(row["fold"])
        except ValueError as error:
            raise VerificationError(f"OOF row {row_number} has invalid integer metadata") from error
        homology_id = row["homology_component_id"]
        union_id = row["union_component_id"]
        identity = _finite(row["max_train_identity"], label="OOF max_train_identity")
        probability = _finite(row["probability"], label="OOF probability")
        _require(
            observations > 0
            and fold in range(5)
            and _SHA256.fullmatch(homology_id) is not None
            and _SHA256.fullmatch(union_id) is not None
            and 0 <= identity < 0.80
            and 0 <= probability <= 1,
            f"OOF row {row_number} violates frozen value bounds",
        )
        current = (
            sequence_id,
            sequence,
            target,
            gram,
            int(row["label"]),
            observations,
            fold,
            homology_id,
            union_id,
            identity,
        )
        previous = metadata.setdefault(example_id, current)
        _require(previous == current, f"OOF example {example_id} has inconsistent model metadata")
        _require(
            model not in predictions[example_id], f"OOF example {example_id} duplicates a model"
        )
        predictions[example_id][model] = probability
    _require(row_count == 7476 and len(metadata) == 2492, "Gate-1 OOF census changed")

    examples: list[Example] = []
    for example_id in sorted(metadata):
        model_values = predictions[example_id]
        _require(set(model_values) == set(_MODELS), "OOF example lacks exact three-model support")
        _require(
            math.isclose(
                model_values["equal_weight_ensemble"],
                0.5 * (model_values["descriptor_logistic"] + model_values["homology_knn"]),
                rel_tol=0.0,
                abs_tol=2e-15,
            ),
            "OOF equal-weight ensemble is inconsistent",
        )
        (
            sequence_id,
            sequence,
            target,
            gram,
            label,
            observations,
            fold,
            homology_id,
            union_id,
            identity,
        ) = metadata[example_id]
        examples.append(
            Example(
                example_id=example_id,
                sequence_id=str(sequence_id),
                sequence=str(sequence),
                target=str(target),
                gram=str(gram),
                label=int(label),
                observations=int(observations),
                fold=int(fold),
                homology_id=str(homology_id),
                union_id=str(union_id),
                max_train_identity=float(identity),
                predictions=dict(model_values),
            )
        )
    examples_by_fold = Counter(example.fold for example in examples)
    _require(
        tuple(examples_by_fold[index] for index in range(5)) == (546, 486, 485, 487, 488),
        "Gate-1 example fold census changed",
    )
    _require(
        len({example.sequence_id for example in examples}) == 952, "OOF sequence census changed"
    )
    _require({example.target for example in examples} == set(targets), "OOF target census changed")
    for field in ("homology_id", "union_id"):
        fold_by_component: dict[str, int] = {}
        for example in examples:
            component = str(getattr(example, field))
            previous = fold_by_component.setdefault(component, example.fold)
            _require(previous == example.fold, f"OOF {field} crosses held-out folds")
    return tuple(examples)


def _read_embeddings(
    index_snapshot: Snapshot,
    matrix_snapshot: Snapshot,
    ledger: Mapping[str, Any],
) -> tuple[tuple[tuple[str, str], ...], np.ndarray]:
    config = ledger.get("embeddings")
    _require(isinstance(config, dict), "embedding config is missing")
    reader = _csv_reader(index_snapshot.payload, label="embedding index")
    _require(
        tuple(reader.fieldnames or ()) == _EMBEDDING_INDEX_FIELDS, "embedding index schema changed"
    )
    index: list[tuple[str, str]] = []
    previous_id = ""
    for row_number, row in enumerate(reader, start=2):
        _require(
            None not in row and all(value is not None for value in row.values()),
            f"embedding index row {row_number} has the wrong width",
        )
        try:
            row_index = int(row["row_index"])
            length = int(row["length"])
        except ValueError as error:
            raise VerificationError("embedding index has invalid integer metadata") from error
        sequence = _canonical_sequence(row["sequence"])
        sequence_id = row["sequence_id"]
        _require(
            row_index == len(index)
            and sequence == row["sequence"]
            and sequence_id == _sequence_id(sequence)
            and length == len(sequence)
            and (not previous_id or sequence_id > previous_id),
            f"embedding index row {row_number} violates identity/order",
        )
        index.append((sequence_id, sequence))
        previous_id = sequence_id
    _require(len(index) == 952, "embedding index census changed")
    ordered_id_sha = _sha256_bytes(
        "".join(f"{sequence_id}\n" for sequence_id, _ in index).encode("ascii")
    )
    _require(
        ordered_id_sha == config.get("sequence_ids_sha256"), "embedding ordered-ID hash changed"
    )
    try:
        matrix = np.load(io.BytesIO(matrix_snapshot.payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise VerificationError("embedding matrix is not a safe NumPy array") from error
    _require(
        isinstance(matrix, np.ndarray)
        and matrix.shape == (952, 320)
        and matrix.dtype == np.dtype("<f4")
        and matrix.flags.c_contiguous
        and np.all(np.isfinite(matrix)),
        "embedding matrix shape/dtype/layout/finiteness changed",
    )
    _require(
        _sha256_bytes(matrix.tobytes(order="C")) == config.get("tensor_data_sha256"),
        "embedding tensor-data hash changed",
    )
    matrix.setflags(write=False)
    return tuple(index), matrix


def _alignment_identity(left: str, right: str) -> float:
    previous: list[tuple[int, int, int]] = [(0, 0, 0)]
    for column in range(1, len(right) + 1):
        previous.append((-column, 0, -column))
    for row_index, left_residue in enumerate(left, start=1):
        current: list[tuple[int, int, int]] = [(-row_index, 0, -row_index)]
        for column, right_residue in enumerate(right, start=1):
            diagonal = previous[column - 1]
            matched = int(left_residue == right_residue)
            diagonal_state = (
                diagonal[0] + (1 if matched else -1),
                diagonal[1] + matched,
                diagonal[2] - 1,
            )
            up = previous[column]
            up_state = (up[0] - 1, up[1], up[2] - 1)
            prior = current[column - 1]
            left_state = (prior[0] - 1, prior[1], prior[2] - 1)
            current.append(max(diagonal_state, up_state, left_state))
        previous = current
    _, matches, negative_length = previous[-1]
    return matches / -negative_length


def _cluster_sequences(sequences: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    values = sorted(set(sequences))
    _require(len(values) == len(sequences), "diversity clustering input contains duplicates")
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for left_index, left in enumerate(values):
        for right_index in range(left_index + 1, len(values)):
            right = values[right_index]
            if min(len(left), len(right)) / max(len(left), len(right)) < 0.70:
                continue
            if _alignment_identity(left, right) >= 0.70:
                union(left_index, right_index)
    components: dict[int, list[str]] = defaultdict(list)
    for index, sequence in enumerate(values):
        components[find(index)].append(sequence)
    return tuple(
        sorted(
            (tuple(sorted(component)) for component in components.values()), key=lambda row: row[0]
        )
    )


def _component_id(sequences: Sequence[str]) -> str:
    sequence_ids = sorted(_sequence_id(sequence) for sequence in sequences)
    payload = json.dumps(
        {
            "algorithm": _DIVERSITY_ALGORITHM,
            "domain": "amp_challenge.union_v1.replay.diversity_cluster",
            "identity_threshold_hex": (0.70).hex(),
            "schema_version": 1,
            "sequence_ids": sequence_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "div70:" + _sha256_bytes(payload)


def _objective_examples(examples: Sequence[Example], objective: str) -> tuple[Example, ...]:
    if objective == "broad_spectrum":
        return tuple(examples)
    gram = "positive" if objective == "gram_positive" else "negative"
    return tuple(example for example in examples if example.gram == gram)


def _reconstruct_candidates(
    examples: Sequence[Example],
    embedding_index: Sequence[tuple[str, str]],
    matrix: np.ndarray,
) -> tuple[tuple[Candidate, ...], bytes, bytes]:
    by_sequence: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_sequence[example.sequence_id].append(example)
    oof_map = {
        sequence_id: sequence_examples[0].sequence
        for sequence_id, sequence_examples in by_sequence.items()
    }
    embedding_map = {
        sequence_id: (row_index, sequence)
        for row_index, (sequence_id, sequence) in enumerate(embedding_index)
    }
    _require(
        oof_map == {sequence_id: sequence for sequence_id, (_, sequence) in embedding_map.items()},
        "OOF and ESM panels do not join exactly by sequence ID and sequence",
    )

    partial: list[dict[str, object]] = []
    contexts = 0
    source_observations = 0
    target_counts: Counter[int] = Counter()
    exclusions: Counter[str] = Counter()
    support: dict[str, list[int]] = {objective: [] for objective in _OBJECTIVES}
    for sequence_id in sorted(by_sequence):
        sequence_examples = by_sequence[sequence_id]
        metadata_sets = (
            {item.sequence for item in sequence_examples},
            {item.fold for item in sequence_examples},
            {item.homology_id for item in sequence_examples},
            {item.union_id for item in sequence_examples},
            {item.max_train_identity for item in sequence_examples},
        )
        _require(
            all(len(values) == 1 for values in metadata_sets),
            "OOF sequence metadata is inconsistent",
        )
        objective_values: list[tuple[float, float, float, int]] = []
        missing: list[str] = []
        for objective in _OBJECTIVES:
            selected = _objective_examples(sequence_examples, objective)
            if not selected:
                missing.append(objective)
                continue
            descriptor = float(
                np.mean(
                    [item.predictions["descriptor_logistic"] for item in selected],
                    dtype=np.float64,
                )
            )
            knn = float(
                np.mean(
                    [item.predictions["homology_knn"] for item in selected],
                    dtype=np.float64,
                )
            )
            objective_values.append(
                (
                    descriptor,
                    0.5 * abs(descriptor - knn),
                    float(np.mean([item.label for item in selected], dtype=np.float64)),
                    len(selected),
                )
            )
        if missing:
            exclusions["missing:" + ",".join(missing)] += 1
            continue
        row_index, embedding_sequence = embedding_map[sequence_id]
        sequence = sequence_examples[0].sequence
        _require(sequence == embedding_sequence, "ESM sequence join changed")
        row = {
            "sequence_id": sequence_id,
            "sequence": sequence,
            "means": tuple(value[0] for value in objective_values),
            "stds": tuple(value[1] for value in objective_values),
            "outcomes": tuple(value[2] for value in objective_values),
            "support": tuple(value[3] for value in objective_values),
            "novelty": 1.0 - sequence_examples[0].max_train_identity,
            "embedding": matrix[row_index],
            "esm_row": row_index,
            "homology_id": sequence_examples[0].homology_id,
            "union_id": sequence_examples[0].union_id,
            "fold": sequence_examples[0].fold,
        }
        partial.append(row)
        contexts += len(sequence_examples)
        source_observations += sum(item.observations for item in sequence_examples)
        target_counts[len({item.target for item in sequence_examples})] += 1
        for objective, values in zip(_OBJECTIVES, objective_values, strict=True):
            support[objective].append(values[3])
    partial.sort(key=lambda row: str(row["sequence"]))
    _require(
        len(partial) == 650
        and contexts == 1967
        and source_observations == 2062
        and tuple(target_counts[index] for index in range(2, 8)) == (364, 213, 56, 16, 0, 1)
        and sum(target_counts.values()) == 650,
        "independently reconstructed candidate/support census changed",
    )
    candidate_ids_sha = _sha256_bytes(
        "".join(
            f"{row['sequence_id']}\n"
            for row in sorted(partial, key=lambda row: str(row["sequence_id"]))
        ).encode("ascii")
    )
    _require(candidate_ids_sha == _CANDIDATE_ID_SHA256, "reconstructed candidate-ID hash changed")
    fold_counts = Counter(int(row["fold"]) for row in partial)
    _require(
        tuple(fold_counts[index] for index in range(5)) == _FOLD_COUNTS,
        "candidate fold census changed",
    )

    cluster_by_fold_sequence: dict[tuple[int, str], str] = {}
    diversity_rows: list[dict[str, object]] = []
    for fold in range(5):
        fold_sequences = tuple(str(row["sequence"]) for row in partial if row["fold"] == fold)
        for component in _cluster_sequences(fold_sequences):
            cluster = _component_id(component)
            ids = sorted(_sequence_id(sequence) for sequence in component)
            for sequence in component:
                cluster_by_fold_sequence[(fold, sequence)] = cluster
            diversity_rows.append(
                {
                    "diversity_cluster_id_70": cluster,
                    "identity_algorithm": _DIVERSITY_ALGORITHM,
                    "identity_threshold": 0.70,
                    "prediction_fold": fold,
                    "replay_round": f"fold-{fold}",
                    "sequence_count": len(component),
                    "sequence_ids": ids,
                }
            )
    diversity_rows.sort(
        key=lambda row: (int(row["prediction_fold"]), str(row["diversity_cluster_id_70"]))
    )
    diversity_payload = b"".join(_canonical_json_bytes(row) for row in diversity_rows)

    candidates = tuple(
        Candidate(
            sequence_id=str(row["sequence_id"]),
            sequence=str(row["sequence"]),
            means=tuple(row["means"]),
            stds=tuple(row["stds"]),
            outcomes=tuple(row["outcomes"]),
            support=tuple(row["support"]),
            novelty=float(row["novelty"]),
            embedding=np.asarray(row["embedding"]),
            esm_row=int(row["esm_row"]),
            homology_id=str(row["homology_id"]),
            union_id=str(row["union_id"]),
            fold=int(row["fold"]),
            cluster_id=cluster_by_fold_sequence[(int(row["fold"]), str(row["sequence"]))],
        )
        for row in partial
    )
    embedding_fields = tuple(f"embedding_{column:03d}" for column in range(320))
    fields = (
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
    )
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for candidate in candidates:
        row: dict[str, object] = {
            "sequence_id": candidate.sequence_id,
            "sequence": candidate.sequence,
            **{
                f"mean_{objective}": candidate.means[index]
                for index, objective in enumerate(_OBJECTIVES)
            },
            **{
                f"std_{objective}": candidate.stds[index]
                for index, objective in enumerate(_OBJECTIVES)
            },
            "novelty": candidate.novelty,
            **{
                name: float(candidate.embedding[index])
                for index, name in enumerate(embedding_fields)
            },
            "cluster_id": candidate.cluster_id,
            "diversity_cluster_id_70": candidate.cluster_id,
            "homology_component_id": candidate.homology_id,
            "union_component_id": candidate.union_id,
            "esm_source_row_index": candidate.esm_row,
            "eligible": "true",
            "replay_round": f"fold-{candidate.fold}",
            "prediction_scope": "out_of_fold",
            "prediction_fold": candidate.fold,
            "outcome_fold": candidate.fold,
            **{
                f"outcome_{objective}": candidate.outcomes[index]
                for index, objective in enumerate(_OBJECTIVES)
            },
            **{
                f"n_{objective}": candidate.support[index]
                for index, objective in enumerate(_OBJECTIVES)
            },
        }
        writer.writerow(
            {
                name: format(value, ".17g") if isinstance(value, float) else value
                for name, value in row.items()
            }
        )
    return candidates, target.getvalue().encode("utf-8"), diversity_payload


def _rank_scale(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    _require(vector.ndim == 1 and np.all(np.isfinite(vector)), "rank input is invalid")
    if len(vector) == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(vector, kind="stable")
    result = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and vector[order[end]] == vector[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) / (len(vector) - 1)
        start = end
    return result


def _column_rank_scale(values: np.ndarray) -> np.ndarray:
    return np.column_stack([_rank_scale(values[:, column]) for column in range(values.shape[1])])


def _minimum_cosine_distance(embeddings: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    if selected:
        similarities = normalized @ normalized[np.asarray(selected)].T
        return np.clip(1.0 - np.max(similarities, axis=1), 0.0, 2.0) / 2.0
    centroid = np.mean(normalized, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return np.ones(len(values), dtype=np.float64)
    similarities = normalized @ (centroid / centroid_norm)
    return np.clip(1.0 - similarities, 0.0, 2.0) / 2.0


def _allocate_quotas(weights: Mapping[str, float], total: int) -> Mapping[str, int]:
    positive = {name: value for name, value in weights.items() if value > 0}
    scale = total / sum(positive.values())
    exact = {name: value * scale for name, value in positive.items()}
    quotas = {name: int(np.floor(value)) for name, value in exact.items()}
    missing = total - sum(quotas.values())
    order = sorted(
        positive,
        key=lambda name: (-(exact[name] - quotas[name]), _STRATEGY_ORDER.index(name)),
    )
    for name in order[:missing]:
        quotas[name] += 1
    return quotas


def _choose(
    scores: np.ndarray,
    *,
    selected: Sequence[int],
    clusters: Sequence[str],
    cluster_counts: Counter[str],
    quality_floor: float | None,
    conservative: np.ndarray,
) -> int:
    selected_set = set(selected)
    order = np.lexsort((np.arange(len(scores)), -scores))
    for enforce_floor in (True, False):
        for raw_index in order:
            index = int(raw_index)
            if index in selected_set or cluster_counts[clusters[index]] >= 2:
                continue
            if enforce_floor and quality_floor is not None and conservative[index] < quality_floor:
                continue
            return index
    raise VerificationError("independent selector cannot fill the strict cluster-capped batch")


def _select_policy(candidates: Sequence[Candidate], *, policy: str, seed: int) -> Selection:
    means = np.asarray([candidate.means for candidate in candidates], dtype=np.float64)
    stds = np.asarray([candidate.stds for candidate in candidates], dtype=np.float64)
    embeddings: np.ndarray | None = np.asarray(
        [candidate.embedding for candidate in candidates],
        dtype=np.float64,
    )
    if policy == "mixed":
        strategy_mix = {
            "exploit": 0.50,
            "diversity": 0.20,
            "pareto": 0.10,
            "uncertainty": 0.10,
            "random": 0.10,
        }
        risk_beta = 1.0
        quality_quantile = 0.20
    elif policy == "lcb":
        strategy_mix = {"exploit": 1.0}
        risk_beta = 1.0
        quality_quantile = 0.20
    elif policy == "mean":
        strategy_mix = {"exploit": 1.0}
        risk_beta = 0.0
        quality_quantile = 0.20
    elif policy == "random":
        strategy_mix = {"random": 1.0}
        risk_beta = 0.0
        quality_quantile = 0.0
        means = np.zeros_like(means)
        stds = np.zeros_like(stds)
        embeddings = None
    else:
        raise VerificationError(f"unsupported frozen policy {policy}")

    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    mean_rank = _column_rank_scale(means)
    std_rank = _column_rank_scale(stds)
    conservative = (mean_rank - risk_beta * std_rank) @ weights
    uncertainty = std_rank @ weights + 0.15 * conservative
    rng = np.random.default_rng(seed)
    random_scores = rng.random(len(candidates))
    quality_floor = float(np.quantile(conservative, quality_quantile))
    pending = dict(_allocate_quotas(strategy_mix, 10))
    selected: list[int] = []
    reasons: list[str] = []
    acquisition: list[float] = []
    clusters = tuple(candidate.cluster_id for candidate in candidates)
    cluster_counts: Counter[str] = Counter()
    while sum(pending.values()) > 0:
        progressed = False
        for strategy in _STRATEGY_ORDER:
            if pending.get(strategy, 0) <= 0:
                continue
            if strategy == "exploit":
                scores = conservative
            elif strategy == "pareto":
                direction = rng.dirichlet(np.maximum(weights, 1e-6))
                scores = (mean_rank - 0.25 * risk_beta * std_rank) @ direction
            elif strategy == "diversity":
                _require(embeddings is not None, "diversity policy lacks ESM vectors")
                distance = _minimum_cosine_distance(embeddings, selected)
                scores = 0.65 * distance + 0.35 * _rank_scale(conservative)
            elif strategy == "uncertainty":
                scores = uncertainty
            elif strategy == "random":
                scores = random_scores + 0.10 * conservative
            else:
                raise VerificationError("unexpected strategy in frozen non-start-aware replay")
            floor = None if strategy == "exploit" else quality_floor
            index = _choose(
                scores,
                selected=selected,
                clusters=clusters,
                cluster_counts=cluster_counts,
                quality_floor=floor,
                conservative=conservative,
            )
            selected.append(index)
            reasons.append(strategy)
            acquisition.append(float(scores[index]))
            cluster_counts[clusters[index]] += 1
            pending[strategy] -= 1
            progressed = True
        _require(progressed, "independent strategy allocation stalled")
    ranked = sorted(
        range(len(selected)),
        key=lambda position: (-conservative[selected[position]], selected[position]),
    )
    return Selection(
        indices=tuple(selected[position] for position in ranked),
        reasons=tuple(reasons[position] for position in ranked),
        conservative=tuple(float(conservative[selected[position]]) for position in ranked),
        acquisition=tuple(acquisition[position] for position in ranked),
    )


def _policy_seed(seed: int, group: str, policy: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{group}\0{policy}".encode()).digest()[:4], "big")


def _oracle_rewards(rewards: np.ndarray, clusters: Sequence[str], batch_size: int) -> np.ndarray:
    order = sorted(range(len(rewards)), key=lambda index: (-rewards[index], index))
    selected: list[float] = []
    counts: Counter[str] = Counter()
    for index in order:
        if counts[clusters[index]] >= 2:
            continue
        counts[clusters[index]] += 1
        selected.append(float(rewards[index]))
        if len(selected) == batch_size:
            return np.asarray(selected, dtype=np.float64)
    raise VerificationError("independent regret oracle cannot fill the cluster-capped batch")


def _pairwise_distance(embeddings: np.ndarray) -> float:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings, dtype=np.float64),
        where=norms > 0,
    )
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    upper = np.triu_indices(len(embeddings), k=1)
    return float(np.mean(1.0 - similarities[upper]))


def _metric_row(
    candidates: Sequence[Candidate],
    selection: Selection,
    *,
    policy: str,
    group: str,
    seed: int,
) -> Mapping[str, object]:
    outcomes = np.asarray([candidate.outcomes for candidate in candidates], dtype=np.float64)
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    scalar = outcomes @ weights
    selected = np.asarray(selection.indices, dtype=np.int64)
    selected_outcomes = outcomes[selected]
    selected_scalar = scalar[selected]
    clusters = tuple(candidate.cluster_id for candidate in candidates)
    optimum = _oracle_rewards(scalar, clusters, len(selected))
    hits = selected_outcomes >= 0.5
    embeddings = np.asarray([candidates[index].embedding for index in selected], dtype=np.float64)
    row: dict[str, object] = {
        "policy": policy,
        "group": group,
        "seed": seed,
        "candidate_count": len(candidates),
        "eligible_count": len(candidates),
        "batch_size": len(selected),
        "reward_mean": float(np.mean(selected_scalar)),
        "reward_sum": float(np.sum(selected_scalar)),
        "best_reward": float(np.max(selected_scalar)),
        "cumulative_regret": float(max(0.0, np.sum(optimum) - np.sum(selected_scalar))),
        "simple_regret": float(max(0.0, np.max(scalar) - np.max(selected_scalar))),
        "hit_rate_any": float(np.mean(np.any(hits, axis=1))),
        "hit_rate_all": float(np.mean(np.all(hits, axis=1))),
        "category_coverage": float(np.mean(np.any(hits, axis=0))),
        "unique_clusters": len({clusters[index] for index in selected}),
        "mean_pairwise_cosine_distance": _pairwise_distance(embeddings),
        "strategy_counts": json.dumps(
            dict(sorted(Counter(selection.reasons).items())), sort_keys=True
        ),
    }
    for index, objective in enumerate(_OBJECTIVES):
        row[f"mean_outcome_{objective}"] = float(np.mean(selected_outcomes[:, index]))
        row[f"hit_count_{objective}"] = int(np.count_nonzero(hits[:, index]))
    return row


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: ""
                if row.get(name) is None
                else f"{row[name]:.12g}"
                if isinstance(row.get(name), float)
                else row.get(name)
                for name in fields
            }
        )
    return target.getvalue().encode("utf-8")


def _aggregate_runs(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    metrics = (
        "reward_mean",
        "reward_sum",
        "best_reward",
        "cumulative_regret",
        "simple_regret",
        "hit_rate_any",
        "hit_rate_all",
        "category_coverage",
        "unique_clusters",
        "mean_pairwise_cosine_distance",
        *(f"mean_outcome_{objective}" for objective in _OBJECTIVES),
    )
    summary: dict[str, object] = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        subset = [row for row in rows if row["policy"] == policy]
        values: dict[str, object] = {"evaluations": len(subset)}
        for metric in metrics:
            observed = [float(row[metric]) for row in subset if row.get(metric) is not None]
            values[metric] = {
                "mean": float(np.mean(observed)),
                "std": float(np.std(observed)),
                "min": float(np.min(observed)),
                "max": float(np.max(observed)),
            }
        values["mean_outcomes"] = {
            objective: values.pop(f"mean_outcome_{objective}") for objective in _OBJECTIVES
        }
        summary[policy] = values
    return summary


def _paired_improvements(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    directions = {
        "reward_mean": 1.0,
        "best_reward": 1.0,
        "hit_rate_any": 1.0,
        "hit_rate_all": 1.0,
        "category_coverage": 1.0,
        "mean_pairwise_cosine_distance": 1.0,
        "cumulative_regret": -1.0,
        "simple_regret": -1.0,
    }
    means: dict[tuple[str, str, str], float] = {}
    for policy in _POLICIES:
        for group in _GROUPS:
            subset = [row for row in rows if row["policy"] == policy and row["group"] == group]
            for metric in directions:
                values = [float(row[metric]) for row in subset if row.get(metric) is not None]
                means[(policy, group, metric)] = float(np.mean(values))
    result: dict[str, object] = {}
    for baseline in ("lcb", "mean", "random"):
        metric_rows: dict[str, object] = {}
        for metric, direction in directions.items():
            by_group = {
                group: direction
                * (means[("mixed", group, metric)] - means[(baseline, group, metric)])
                for group in _GROUPS
            }
            values = list(by_group.values())
            metric_rows[metric] = {
                "positive_means_mixed_is_better": True,
                "mean_improvement": float(np.mean(values)),
                "median_improvement": float(np.median(values)),
                "improved_groups": sum(value > 1e-12 for value in values),
                "tied_groups": sum(abs(value) <= 1e-12 for value in values),
                "worsened_groups": sum(value < -1e-12 for value in values),
                "groups": len(values),
                "by_group": dict(sorted(by_group.items())),
            }
        result[f"mixed_vs_{baseline}"] = metric_rows
    return result


def _reconstruct_replay(
    candidates: Sequence[Candidate],
    *,
    ledger_sha256: str,
    replay_config_sha256: str,
    selection_config_sha256: str,
) -> tuple[bytes, bytes, bytes]:
    run_rows: list[Mapping[str, object]] = []
    selection_rows: list[Mapping[str, object]] = []
    for group in _GROUPS:
        fold = int(group.removeprefix("fold-"))
        group_candidates = tuple(candidate for candidate in candidates if candidate.fold == fold)
        _require(len(group_candidates) == _FOLD_COUNTS[fold], "replay group census changed")
        for configured_seed in _SEEDS:
            for policy in _POLICIES:
                result = _select_policy(
                    group_candidates,
                    policy=policy,
                    seed=_policy_seed(configured_seed, group, policy),
                )
                run_rows.append(
                    _metric_row(
                        group_candidates,
                        result,
                        policy=policy,
                        group=group,
                        seed=configured_seed,
                    )
                )
                for rank, local_index in enumerate(result.indices, start=1):
                    candidate = group_candidates[local_index]
                    hit_vector = tuple(value >= 0.5 for value in candidate.outcomes)
                    row: dict[str, object] = {
                        "policy": policy,
                        "group": group,
                        "seed": configured_seed,
                        "rank": rank,
                        "sequence_id": candidate.sequence_id,
                        "sequence": candidate.sequence,
                        "reason": result.reasons[rank - 1],
                        "conservative_score": result.conservative[rank - 1],
                        "acquisition_score": result.acquisition[rank - 1],
                        "scalar_outcome": float(np.asarray(candidate.outcomes) @ np.full(3, 1 / 3)),
                        "hit_any": bool(any(hit_vector)),
                        "hit_all": bool(all(hit_vector)),
                        "cluster_id": candidate.cluster_id,
                    }
                    for index, objective in enumerate(_OBJECTIVES):
                        row[f"outcome_{objective}"] = candidate.outcomes[index]
                        row[f"hit_{objective}"] = hit_vector[index]
                    selection_rows.append(row)
    _require(len(run_rows) == 100 and len(selection_rows) == 1000, "replay output census changed")
    run_fields = (
        "policy",
        "group",
        "seed",
        "candidate_count",
        "eligible_count",
        "batch_size",
        "reward_mean",
        "reward_sum",
        "best_reward",
        "cumulative_regret",
        "simple_regret",
        "hit_rate_any",
        "hit_rate_all",
        "category_coverage",
        "unique_clusters",
        "mean_pairwise_cosine_distance",
        *(f"mean_outcome_{objective}" for objective in _OBJECTIVES),
        *(f"hit_count_{objective}" for objective in _OBJECTIVES),
        "strategy_counts",
    )
    selection_fields = (
        "policy",
        "group",
        "seed",
        "rank",
        "sequence_id",
        "sequence",
        "reason",
        "conservative_score",
        "acquisition_score",
        "scalar_outcome",
        "hit_any",
        "hit_all",
        "cluster_id",
        *(f"outcome_{objective}" for objective in _OBJECTIVES),
        *(f"hit_{objective}" for objective in _OBJECTIVES),
    )
    runs_payload = _csv_bytes(run_fields, run_rows)
    selections_payload = _csv_bytes(selection_fields, selection_rows)
    summary = {
        "schema_version": 1,
        "warning": (
            "Replay validity depends on the ledger's predictions genuinely being produced "
            "without fitting the corresponding outcome fold."
        ),
        "ledger_sha256": ledger_sha256,
        "replay_config_sha256": replay_config_sha256,
        "selection_config_sha256": selection_config_sha256,
        "objectives": list(_OBJECTIVES),
        "policies": list(_POLICIES),
        "seeds": list(_SEEDS),
        "groups": list(_GROUPS),
        "run_count": 100,
        "selection_count": 1000,
        "runs_sha256": _sha256_bytes(runs_payload),
        "selections_sha256": _sha256_bytes(selections_payload),
        "policy_metrics": _aggregate_runs(run_rows),
        "paired_group_improvements": _paired_improvements(run_rows),
    }
    return runs_payload, selections_payload, _canonical_json_bytes(summary, pretty=True)


def _reconstruct_ledger_summary(
    *,
    examples: Sequence[Example],
    candidates: Sequence[Candidate],
    diversity_payload: bytes,
    ledger_payload: bytes,
    ledger_config_sha256: str,
    upstream: Mapping[str, Snapshot],
) -> bytes:
    diversity_rows = [json.loads(line) for line in diversity_payload.decode("utf-8").splitlines()]
    component_sizes = [int(row["sequence_count"]) for row in diversity_rows]
    fold_counts = Counter(candidate.fold for candidate in candidates)
    by_sequence: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_sequence[example.sequence_id].append(example)
    exclusions: Counter[str] = Counter()
    for sequence_examples in by_sequence.values():
        missing = [
            objective
            for objective in _OBJECTIVES
            if not _objective_examples(sequence_examples, objective)
        ]
        if missing:
            exclusions["missing:" + ",".join(missing)] += 1
    target_counts = Counter(
        len({example.target for example in by_sequence[candidate.sequence_id]})
        for candidate in candidates
    )
    support = {
        objective: [candidate.support[index] for candidate in candidates]
        for index, objective in enumerate(_OBJECTIVES)
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact": "union_gate1_activity_replay_ledger_v1",
        "status": "built_for_union_v1_development_replay",
        "config_sha256": ledger_config_sha256,
        "input_sha256": {
            "gate1_oof_predictions": upstream["gate1/oof_predictions.csv"].sha256,
            "gate1_independent_receipt": upstream["gate1/independent-receipt.json"].sha256,
            "esm2_embedding_index": upstream["embeddings/embedding_index.csv"].sha256,
            "esm2_embedding_matrix": upstream["embeddings/embeddings.npy"].sha256,
            "esm2_independent_receipt": upstream["embeddings/independent-receipt.json"].sha256,
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
        "candidate_sequences": len(candidates),
        "candidate_sequence_ids_sha256": _CANDIDATE_ID_SHA256,
        "candidate_observed_contexts": sum(
            len(by_sequence[candidate.sequence_id]) for candidate in candidates
        ),
        "candidate_source_observations": sum(
            example.observations
            for candidate in candidates
            for example in by_sequence[candidate.sequence_id]
        ),
        "candidate_unique_target_counts": {
            str(count): target_counts[count] for count in range(2, 8)
        },
        "excluded_sequences": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "candidates_by_fold": {str(index): fold_counts[index] for index in range(5)},
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
            "dimensions": 320,
            "join": "exact full-panel sequence_id_and_sequence",
            "fallback": "none",
        },
        "diversity_clustering": {
            "algorithm": _DIVERSITY_ALGORITHM,
            "identity_threshold": 0.70,
            "scope": "support_eligible_candidates_clustered_separately_within_each_replay_fold",
            "single_link_bridges_through_ineligible_or_other_fold_sequences": False,
            "components": len(diversity_rows),
            "largest_component": max(component_sizes),
            "candidate_cluster_field": "cluster_id",
            "explicit_audit_field": "diversity_cluster_id_70",
            "upstream_split_components_are_not_diversity_clusters": True,
        },
        "artifacts": {
            "candidate_ledger.csv": _sha256_bytes(ledger_payload),
            "diversity_components.jsonl": _sha256_bytes(diversity_payload),
        },
        "outcome_limit": (
            "observed-context activity-fraction proxies over sparse accepted target coverage; "
            "not full-panel broad-spectrum outcomes"
        ),
    }
    return _canonical_json_bytes(summary, pretty=True)


def _tree_sha256(root: Path) -> str:
    lines = "".join(
        f"{stat.S_IMODE((root / relative).stat().st_mode):03o} "
        f"{_sha256_file(root / relative)} {relative}\n"
        for relative in _BUNDLE_FILES
    )
    return _sha256_bytes(lines.encode("ascii"))


def _load_runtime_preflight(path: str | Path) -> tuple[Snapshot, Mapping[str, Any]]:
    snapshot = _snapshot(
        path,
        label="operational runtime preflight",
        required_mode=0o444,
    )
    value = _json_object(snapshot.payload, label="operational runtime preflight")
    audit = value.get("audit")
    checks = value.get("checks")
    expected_resources = {
        "account": "bio",
        "cpus_per_task": 4,
        "gpus": 0,
        "memory_per_node_mib": 16384,
        "nodes": 1,
        "partition": "standard",
        "tasks": 1,
    }
    expected_environment = {
        "fresh_job_scoped": True,
        "install": "uv_sync_locked_offline_no_editable_refreshed_project",
        "scope": "core",
    }
    _require(
        snapshot.payload == _canonical_json_bytes(value, pretty=True)
        and set(value) == {"audit", "checks", "schema_version"}
        and value.get("schema_version") == 1
        and isinstance(audit, dict)
        and set(audit) == {"environment", "job_id", "node_name", "resources"}
        and isinstance(audit.get("job_id"), int)
        and audit["job_id"] > 0
        and isinstance(audit.get("node_name"), str)
        and re.fullmatch(r"[A-Za-z0-9._-]+", audit["node_name"]) is not None
        and audit.get("resources") == expected_resources
        and audit.get("environment") == expected_environment
        and checks
        == {
            "audit_runtime_on_slurm_compute": True,
            "fresh_locked_offline_no_editable_refreshed_project_environment_synced": True,
            "producer_resources_verified_from_slurm_accounting": True,
            "repository_clean_synchronized_exact_tree_verified": True,
            "spooled_launcher_attested": True,
        },
        "operational runtime preflight is not the exact resource/runtime attestation",
    )
    return snapshot, value


def _verify_producer_handshake(
    bundle_paths: Sequence[str | Path],
    bundles: tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
    *,
    config_snapshots: Mapping[str, Snapshot],
    git_commit: str,
    upstream: tuple[Mapping[str, Snapshot], Mapping[str, Snapshot]],
) -> tuple[Mapping[str, object], tuple[Snapshot, ...]]:
    roots = tuple(Path(path).resolve(strict=True) for path in bundle_paths)
    _require(roots[0].parent == roots[1].parent, "replay bundles do not share one producer root")
    producer_root = roots[0].parent
    _require(
        producer_root.name.isdigit() and int(producer_root.name) > 0,
        "producer root does not bind a numeric Slurm job",
    )
    producer_job_id = int(producer_root.name)
    receipt_root = producer_root / "node-receipts"
    _reject_symlink_chain(receipt_root, label="producer node-receipts")
    _require(
        receipt_root.is_dir()
        and stat.S_IMODE(receipt_root.stat().st_mode) == 0o555
        and stat.S_IMODE(producer_root.stat().st_mode) == 0o555,
        "producer publication or handshake directory is not sealed 0555",
    )
    observed_root_dirs = sorted(path.name for path in producer_root.iterdir() if path.is_dir())
    observed_root_files = sorted(path.name for path in producer_root.iterdir() if path.is_file())
    for path in producer_root.iterdir():
        metadata = path.stat(follow_symlinks=False)
        _require(
            not path.is_symlink()
            and (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)),
            "producer root contains a symbolic or non-regular entry",
        )
    _require(
        observed_root_dirs == ["0", "1", "node-receipts"] and not observed_root_files,
        "producer root inventory changed",
    )
    expected_names = [
        "0.ack",
        "0.receipt",
        "0.result",
        "1.ack",
        "1.receipt",
        "1.result",
    ]
    _require(
        sorted(path.name for path in receipt_root.iterdir()) == expected_names,
        "producer handshake inventory changed",
    )
    for path in receipt_root.iterdir():
        metadata = path.stat(follow_symlinks=False)
        _require(
            not path.is_symlink() and stat.S_ISREG(metadata.st_mode),
            "producer handshake contains a symbolic or non-regular entry",
        )
    resources = {
        "account": "bio",
        "cpus_per_task": 4,
        "gpus_per_task": 0,
        "launch": "one_srun_exact",
        "memory_per_node_mib": 16384,
        "nodes": 2,
        "partition": "standard",
        "tasks": 2,
        "tasks_per_node": 1,
    }
    environment = {
        "fresh_job_scoped": True,
        "install": "uv_sync_locked_offline_no_editable_refreshed_project",
        "scope": "core",
        "workers": "uv_run_locked_no_sync",
    }
    receipts: dict[int, Snapshot] = {}
    acks: dict[int, Snapshot] = {}
    results: dict[int, Snapshot] = {}
    documents: dict[int, Mapping[str, Any]] = {}
    result_documents: dict[int, Mapping[str, Any]] = {}
    config_sha256 = {
        "ledger": config_snapshots[_LEDGER_CONFIG].sha256,
        "replay": config_snapshots[_REPLAY_CONFIG].sha256,
        "selection": config_snapshots[_SELECTION_CONFIG].sha256,
    }
    for task in range(2):
        receipt = _snapshot(
            receipt_root / f"{task}.receipt",
            label=f"producer task {task} receipt",
            required_mode=0o444,
        )
        ack = _snapshot(
            receipt_root / f"{task}.ack",
            label=f"producer task {task} acknowledgement",
            required_mode=0o444,
        )
        result = _snapshot(
            receipt_root / f"{task}.result",
            label=f"producer task {task} result",
            required_mode=0o444,
        )
        document = _json_object(receipt.payload, label=f"producer task {task} receipt")
        frozen_hashes = {name: snapshot.sha256 for name, snapshot in upstream[task].items()}
        _require(
            receipt.payload == _canonical_json_bytes(document, pretty=True)
            and document
            == {
                "artifact": "union_gate1_activity_replay_v1_node_receipt",
                "config_sha256": config_sha256,
                "environment": environment,
                "frozen_input_sha256": frozen_hashes,
                "git_commit": git_commit,
                "input_twin_slot": task,
                "node_name": document.get("node_name"),
                "producer_job_id": producer_job_id,
                "resources": resources,
                "schema_version": 1,
                "task_id": task,
            }
            and isinstance(document.get("node_name"), str)
            and re.fullmatch(r"[A-Za-z0-9._-]+", str(document["node_name"])) is not None,
            f"producer task {task} receipt is inconsistent",
        )
        receipts[task] = receipt
        acks[task] = ack
        results[task] = result
        documents[task] = document
        result_document = _json_object(result.payload, label=f"producer task {task} result")
        semantic_hashes = {
            relative: bundles[task][relative].sha256
            for relative in (
                "ledger/candidate_ledger.csv",
                "ledger/diversity_components.jsonl",
                "ledger/ledger_summary.json",
                "replay/runs.csv",
                "replay/selections.csv",
                "replay/summary.json",
            )
        }
        _require(
            result.payload == _canonical_json_bytes(result_document, pretty=True)
            and result_document
            == {
                "artifact": "union_gate1_activity_replay_v1_node_result",
                "bundle_top_manifest_sha256": bundles[task]["SHA256SUMS"].sha256,
                "bundle_tree_sha256": _tree_sha256(roots[task]),
                "candidate_count": 650,
                "producer_job_id": producer_job_id,
                "replay_run_count": 100,
                "replay_selection_count": 1000,
                "schema_version": 1,
                "semantic_sha256": semantic_hashes,
                "status": "completed_not_automatically_promoted",
                "task_id": task,
            },
            f"producer task {task} result does not bind its completed semantic artifacts",
        )
        result_documents[task] = result_document
    _require(
        documents[0]["node_name"] != documents[1]["node_name"],
        "producer tasks did not run on distinct nodes",
    )
    for task in range(2):
        sibling = 1 - task
        ack_document = _json_object(acks[task].payload, label=f"producer task {task} ack")
        _require(
            acks[task].payload == _canonical_json_bytes(ack_document, pretty=True)
            and ack_document
            == {
                "artifact": "union_gate1_activity_replay_v1_node_ack",
                "observed_sibling_receipt_sha256": receipts[sibling].sha256,
                "producer_job_id": producer_job_id,
                "schema_version": 1,
                "task_id": task,
            },
            f"producer task {task} acknowledgement does not bind its sibling",
        )
    return (
        {
            "job_id": producer_job_id,
            "environment": environment,
            "nodes": [documents[0]["node_name"], documents[1]["node_name"]],
            "requested_resources": resources,
            "handshake": {
                str(task): {
                    "ack_sha256": acks[task].sha256,
                    "receipt_sha256": receipts[task].sha256,
                    "result_sha256": results[task].sha256,
                    "bundle_tree_sha256": result_documents[task]["bundle_tree_sha256"],
                }
                for task in range(2)
            },
        },
        tuple(
            snapshot
            for task in range(2)
            for snapshot in (receipts[task], acks[task], results[task])
        ),
    )


def _write_receipt(path: Path, value: Mapping[str, object]) -> Snapshot:
    payload = _canonical_json_bytes(value, pretty=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
    os.chmod(path, 0o444, follow_symlinks=False)
    return _snapshot(path, label="verification receipt", required_mode=0o444)


def verify_union_replay_bundles(
    gate_twins: Sequence[str | Path],
    gate_independent_receipt: str | Path,
    embedding_twins: Sequence[str | Path],
    embedding_independent_receipt: str | Path,
    bundle_twins: Sequence[str | Path],
    *,
    repository_root: str | Path,
    git_commit: str,
    operational_preflight: str | Path,
    receipt_dir: str | Path,
) -> VerificationExecution:
    """Reconstruct both scientific stages and publish two immutable receipts."""

    _require(_COMMIT.fullmatch(git_commit) is not None, "git_commit must be full lowercase hex")
    repository = Path(repository_root)
    _reject_symlink_chain(repository, label="repository")
    repository = repository.resolve(strict=True)
    _require(repository.is_dir(), "repository root is not a directory")
    output = Path(receipt_dir)
    _reject_symlink_chain(output, label="receipt output")
    _require(not os.path.lexists(output), "refusing to replace receipt output")

    code_snapshots = {
        relative: _snapshot(repository / relative, label=f"reviewed code {relative}")
        for relative in _CODE_PATHS
    }
    _, ledger_document, _ = _load_documents(code_snapshots)
    upstream, upstream_snapshots = _verify_upstream_inputs(
        gate_twins,
        gate_independent_receipt,
        embedding_twins,
        embedding_independent_receipt,
        ledger=ledger_document,
    )
    bundles, bundle_snapshots = _verify_bundles(
        bundle_twins,
        code_snapshots=code_snapshots,
        git_commit=git_commit,
        upstream=upstream,
    )
    bundle = bundles[0]
    preflight, preflight_document = _load_runtime_preflight(operational_preflight)
    production, handshake_snapshots = _verify_producer_handshake(
        bundle_twins,
        bundles,
        config_snapshots=code_snapshots,
        git_commit=git_commit,
        upstream=upstream,
    )
    audit = preflight_document["audit"]
    _require(isinstance(audit, dict), "runtime preflight audit record is invalid")
    _require(
        audit["job_id"] != production["job_id"] and audit["node_name"] not in production["nodes"],
        "audit must have a different job ID and physical node from both producers",
    )

    examples = _read_examples(upstream[0]["gate1/oof_predictions.csv"], ledger_document)
    embedding_index, embedding_matrix = _read_embeddings(
        upstream[0]["embeddings/embedding_index.csv"],
        upstream[0]["embeddings/embeddings.npy"],
        ledger_document,
    )
    candidates, ledger_payload, diversity_payload = _reconstruct_candidates(
        examples,
        embedding_index,
        embedding_matrix,
    )
    _require(
        bundle["ledger/candidate_ledger.csv"].payload == ledger_payload,
        "candidate ledger bytes differ from independent raw-input reconstruction",
    )
    _require(
        bundle["ledger/diversity_components.jsonl"].payload == diversity_payload,
        "diversity component bytes differ from independent 70%-identity reconstruction",
    )
    config_hashes = {
        relative: code_snapshots[relative].sha256
        for relative in (_SELECTION_CONFIG, _LEDGER_CONFIG, _REPLAY_CONFIG)
    }
    ledger_summary_payload = _reconstruct_ledger_summary(
        examples=examples,
        candidates=candidates,
        diversity_payload=diversity_payload,
        ledger_payload=ledger_payload,
        ledger_config_sha256=config_hashes[_LEDGER_CONFIG],
        upstream=upstream[0],
    )
    _require(
        bundle["ledger/ledger_summary.json"].payload == ledger_summary_payload,
        "ledger summary bytes differ from independent reconstruction",
    )

    runs_payload, selections_payload, summary_payload = _reconstruct_replay(
        candidates,
        ledger_sha256=_sha256_bytes(ledger_payload),
        replay_config_sha256=config_hashes[_REPLAY_CONFIG],
        selection_config_sha256=config_hashes[_SELECTION_CONFIG],
    )
    _require(bundle["replay/runs.csv"].payload == runs_payload, "replay run metrics differ")
    _require(
        bundle["replay/selections.csv"].payload == selections_payload,
        "replay selections differ from independent policy execution",
    )
    _require(bundle["replay/summary.json"].payload == summary_payload, "replay summary differs")

    replay_summary = _json_object(summary_payload, label="reconstructed replay summary")
    independent_checks = {
        "accepted_embedding_matrix_and_index_hashes_verified": True,
        "accepted_gate1_oof_and_receipt_hashes_verified": True,
        "candidate_650_cohort_reconstructed": True,
        "candidate_context_aggregation_reconstructed": True,
        "diversity_70_percent_components_reconstructed_per_fold": True,
        "esm_320_feature_use_reconstructed_exactly": True,
        "frozen_max_per_cluster_two_enforced": True,
        "all_six_semantic_artifacts_reconstructed_byte_for_byte": True,
        "mixed_lcb_mean_random_policies_reimplemented": True,
        "replay_metrics_and_summary_reconstructed": True,
        "replay_selections_reconstructed": True,
    }
    independent_receipt_value: dict[str, object] = {
        "artifact": _INDEPENDENT_ARTIFACT,
        "artifact_sha256": {
            relative: bundle[relative].sha256
            for relative in (
                "ledger/candidate_ledger.csv",
                "ledger/diversity_components.jsonl",
                "ledger/ledger_summary.json",
                "replay/runs.csv",
                "replay/selections.csv",
                "replay/summary.json",
            )
        },
        "automatic_production_eligible": False,
        "checks": independent_checks,
        "census": {
            "candidate_sequences": 650,
            "embedding_dimensions": 320,
            "fold_candidate_counts": list(_FOLD_COUNTS),
            "replay_runs": 100,
            "replay_selections": 1000,
        },
        "decision_scope": "development_activity_acquisition_replay_only",
        "git_commit": git_commit,
        "limitations": [
            "activity-fraction outcomes cover only each sequence's sparse observed target contexts",
            "family disagreement is not calibrated posterior or aleatoric uncertainty",
            "policy promotion requires manual scientific review and prospective evidence",
        ],
        "paired_group_improvements": replay_summary["paired_group_improvements"],
        "schema_version": 1,
        "status": "passed_development_evidence_only",
    }
    _all_true(independent_checks, label="constructed independent receipt")

    output.mkdir(parents=True, mode=0o700)
    independent_path = output / "independent-verification.json"
    independent_snapshot = _write_receipt(independent_path, independent_receipt_value)
    operational_checks = {
        "audit_job_and_node_distinct_from_production": True,
        "audit_requested_resources_verified": True,
        "bundle_code_manifest_matches_reviewed_tree": True,
        "bundle_frozen_input_manifest_matches_accepted_evidence": True,
        "bundle_inventory_exactly_ten_files": True,
        "bundle_payloads_mode_0444_and_directories_0555": True,
        "bundle_twins_byte_identical": True,
        "dynamic_paths_job_ids_and_nodes_absent_from_bundle": True,
        "fresh_locked_offline_no_editable_refreshed_project_environments_verified": True,
        "independent_receipt_published_without_replacement": True,
        "producer_node_receipt_ack_and_result_hashes_verified": True,
        "producer_requested_resources_and_topology_verified": True,
        "runtime_preflight_exact_commit_tree_and_spooled_launcher_verified": True,
        "top_manifest_binds_exactly_nine_payloads": True,
    }
    operational_receipt_value: dict[str, object] = {
        "artifact": _OPERATIONAL_ARTIFACT,
        "automatic_production_eligible": False,
        "bundle_identity": {
            "code_sha256sums_sha256": bundle["CODE_SHA256SUMS"].sha256,
            "frozen_input_sha256sums_sha256": bundle["FROZEN_INPUT_SHA256SUMS"].sha256,
            "top_manifest_sha256": bundle["SHA256SUMS"].sha256,
        },
        "checks": operational_checks,
        "git_commit": git_commit,
        "independent_receipt_sha256": independent_snapshot.sha256,
        "runtime": {
            "audit": audit,
            "production": production,
        },
        "runtime_preflight_sha256": preflight.sha256,
        "schema_version": 1,
        "status": "passed_development_evidence_only",
    }
    _all_true(operational_checks, label="constructed operational receipt")
    operational_path = output / "operational-receipt.json"
    operational_snapshot = _write_receipt(operational_path, operational_receipt_value)
    os.chmod(output, 0o555, follow_symlinks=False)

    all_snapshots = (
        *code_snapshots.values(),
        *upstream_snapshots,
        *bundle_snapshots,
        *handshake_snapshots,
        preflight,
    )
    for snapshot in all_snapshots:
        _unchanged(snapshot, label="verification input")
    _unchanged(independent_snapshot, label="independent receipt")
    _unchanged(operational_snapshot, label="operational receipt")
    final_bundles, _ = _verify_bundles(
        bundle_twins,
        code_snapshots=code_snapshots,
        git_commit=git_commit,
        upstream=upstream,
    )
    final_production, _ = _verify_producer_handshake(
        bundle_twins,
        final_bundles,
        config_snapshots=code_snapshots,
        git_commit=git_commit,
        upstream=upstream,
    )
    _require(final_production == production, "producer topology/handshake changed after audit")
    for path in output.iterdir():
        metadata = path.stat(follow_symlinks=False)
        _require(
            not path.is_symlink()
            and stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o444,
            "receipt output contains a symbolic, non-regular, or writable entry",
        )
    _require(
        sorted(path.name for path in output.iterdir())
        == ["independent-verification.json", "operational-receipt.json"]
        and stat.S_IMODE(output.stat().st_mode) == 0o555,
        "receipt publication inventory or mode changed",
    )
    for snapshot in all_snapshots:
        _unchanged(snapshot, label="verification input after final inventory check")
    _unchanged(independent_snapshot, label="independent receipt after final inventory check")
    _unchanged(operational_snapshot, label="operational receipt after final inventory check")
    return VerificationExecution(
        receipt_dir=output.resolve(strict=True),
        independent_receipt=independent_snapshot.path,
        operational_receipt=operational_snapshot.path,
        independent_sha256=independent_snapshot.sha256,
        operational_sha256=operational_snapshot.sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-twin", action="append", type=Path, required=True)
    parser.add_argument("--gate1-independent-receipt", type=Path, required=True)
    parser.add_argument("--embedding-twin", action="append", type=Path, required=True)
    parser.add_argument("--embedding-independent-receipt", type=Path, required=True)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--operational-preflight", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        execution = verify_union_replay_bundles(
            arguments.gate1_twin,
            arguments.gate1_independent_receipt,
            arguments.embedding_twin,
            arguments.embedding_independent_receipt,
            arguments.bundle,
            repository_root=arguments.repository_root,
            git_commit=arguments.git_commit,
            operational_preflight=arguments.operational_preflight,
            receipt_dir=arguments.receipt_dir,
        )
    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"AMP union replay verification error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "automatic_production_eligible": False,
                "independent_receipt_sha256": execution.independent_sha256,
                "operational_receipt_sha256": execution.operational_sha256,
                "status": "passed_development_evidence_only",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

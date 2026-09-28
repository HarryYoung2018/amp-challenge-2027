"""Independently reconstruct and verify exact-union ESM FASTA export twins.

This module deliberately imports none of the FASTA exporter.  It reparses the
accepted Gate-1 evidence, reconstructs the exact sequence panel and every
semantic export artifact, verifies the production twin/overlap attestations,
and emits a path-free receipt.  A passing receipt accepts only the FASTA as an
input to the declared embedding extraction.  It is not evidence that an
embedding, prediction, model, or performance result exists or is valid.
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
import subprocess
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_NODE_RE = re.compile(r"[A-Za-z0-9._-]+")
_ABSOLUTE_PATH_RE = re.compile(rb"/(?:lustre|tmp)/|/home/[^/]+/")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MIN_LENGTH = 8
_MAX_LENGTH = 50

_BASE_ARTIFACT = "gate1_context_activity_homology_study_union_v1"
_BASE_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_BASE_FOLD_ARTIFACT = "gate1_context_union_fold_reuse"
_BASE_RECEIPT_ARTIFACT = "gate1_union_accepted_split_consumption_receipt"
_BASE_AUDIT_ARTIFACT = "gate1_context_activity_homology_study_union_v1_independent_verification"
_BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_FOLD_POLICY = "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
_LABEL_POLICY = (
    "group every ledger row by assay_context_id before eligibility filtering; retain only "
    "fully bacterial_mic16-eligible contexts with unanimous label, source Gram, and "
    "canonical target"
)
_FORBIDDEN_MODEL_FEATURES = (
    "provenance_id",
    "source_record_id",
    "source_record_key",
    "study_keys",
    "study_status",
    "homology_component_id",
    "union_component_id",
    "fold",
)
_OUTPUT_ARTIFACT = "gate1_union_esm2_exact_panel_fasta_v1"
_COVERAGE_ARTIFACT = "gate1_union_esm2_exact_panel_coverage_v1"
_VERIFICATION_ARTIFACT = "gate1_union_esm2_exact_panel_fasta_v1_independent_verification"
_ACCEPTANCE_STATUS = "accepted_for_embedding_input_only"

_EXPECTED_GATE1_PUBLICATION_SHA256 = (
    "4e259cfb43033c069598d42fd54fec49a67ba55bfbb5c1e77eeef4887815fe2f"
)
_EXPECTED_GATE1_TOP_SHA256 = "556e06fd2b1af1e678de88008bc1b87434fb8cf1179f38c897de7ee2c9fd779d"
_EXPECTED_GATE1_INDEPENDENT_RECEIPT_SHA256 = (
    "1f6a6ce811d3fbecdbbbe7463e6052dc7766130372926d29952c70be65743e81"
)
_EXPECTED_SEQUENCE_IDS_SHA256 = "45a704812a51d51876b8e32e86c8890db5b7d6aa52de1b1f35634e797acd3f03"

_LOGICAL_CONFIG_PATH = "configs/models/esm2_union_v1.toml"
_PRODUCER_MODULE_PATH = "src/amp_challenge/benchmarks/export_union_esm_fasta.py"
_VERIFIER_MODULE_PATH = "src/amp_challenge/benchmarks/verify_union_esm_fasta.py"
_CODE_FIXED_PATHS = frozenset(
    {
        "cluster/slurm/export_union_esm_fasta_v1_twins.sbatch",
        "cluster/validate_union_esm_fasta_output.sh",
        _LOGICAL_CONFIG_PATH,
        "pyproject.toml",
        "uv.lock",
    }
)
_EXPORT_PUBLICATION_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "panel/SHA256SUMS",
        "panel/coverage_receipt.json",
        "panel/manifest.json",
        "panel/union_esm_sequences.fasta",
    }
)
_PANEL_FILES = frozenset(
    {
        "SHA256SUMS",
        "coverage_receipt.json",
        "manifest.json",
        "union_esm_sequences.fasta",
    }
)
_GATE1_PUBLICATION_FILES = frozenset(
    {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "SHA256SUMS",
        "gate1/SHA256SUMS",
        "gate1/context_audit.jsonl",
        "gate1/examples.jsonl",
        "gate1/folds.json",
        "gate1/manifest.json",
        "gate1/metrics.json",
        "gate1/oof_predictions.csv",
        "gate1/split_receipt.json",
    }
)
_GATE1_FILES = frozenset(
    {
        "SHA256SUMS",
        "context_audit.jsonl",
        "examples.jsonl",
        "folds.json",
        "manifest.json",
        "metrics.json",
        "oof_predictions.csv",
        "split_receipt.json",
    }
)
_EXAMPLE_FIELDS = frozenset(
    {
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
    }
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
_FOLD_ASSIGNMENT_FIELDS = frozenset(
    {"example_id", "sequence_id", "homology_component_id", "union_component_id", "fold"}
)
_CLASS_METRICS = (
    "gram_negative_mic16_negative",
    "gram_negative_mic16_positive",
    "gram_positive_mic16_negative",
    "gram_positive_mic16_positive",
)
_SPLIT_RECEIPT_INVARIANTS = frozenset(
    {
        "accepted_independent_verification_passed",
        "accepted_sequence_assignments_reused_exactly",
        "accepted_split_hash_chain_valid",
        "all_contexts_grouped_before_eligibility_filtering",
        "component_assignment_join_exact",
        "cross_fold_identity_strictly_below_threshold",
        "fold_label_census_exact",
        "model_feature_allowlist_exact",
    }
)
_GATE1_INDEPENDENT_CHECKS = frozenset(
    {
        "accepted_split_receipt_chain_valid",
        "all_semantic_artifacts_exact",
        "assignment_and_component_join_recomputed",
        "canonical_serialization_verified",
        "context_partition_recomputed",
        "cross_fold_identity_recomputed",
        "descriptor_logistic_oof_recomputed",
        "ensemble_mean_exact",
        "executing_source_bound_to_repository",
        "homology_knn_oof_recomputed",
        "input_twins_and_frozen_manifests_valid",
        "metrics_and_union_bootstrap_recomputed",
        "production_overlap_handshake_valid",
        "publication_top_manifests_valid",
        "repository_commit_and_cleanliness_verified",
        "scratch_and_absolute_paths_absent",
        "twins_byte_identical",
    }
)
_VERIFICATION_CHECKS = frozenset(
    {
        "accepted_gate1_evidence_chain_valid",
        "accepted_gate1_twins_immutable_and_identical",
        "all_contexts_preserved_before_unique_sequence_projection",
        "all_semantic_artifacts_independently_reconstructed",
        "base_probabilities_not_consumed",
        "canonical_serialization_verified",
        "code_and_config_bound_to_synchronized_commit",
        "exact_sorted_fasta_reconstructed",
        "export_frozen_input_manifests_reconstructed",
        "export_twins_immutable_and_identical",
        "no_embedding_or_model_evidence_claimed",
        "path_free_artifacts_and_receipt",
        "production_overlap_handshake_valid",
        "union_components_fold_disjoint",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "gate1_publication_top_sha256",
        "gate1_top_sha256",
        "gate1_examples_sha256",
        "gate1_folds_sha256",
        "gate1_oof_sha256",
        "gate1_manifest_sha256",
        "gate1_split_receipt_sha256",
        "gate1_independent_receipt_sha256",
        "gate1_config_sha256",
        "gate1_code_manifest_sha256",
        "gate1_frozen_input_manifest_sha256",
        "gate1_git_commit",
        "expected_examples",
        "expected_source_observations",
        "expected_sequences",
        "expected_positive_examples",
        "expected_negative_examples",
        "expected_examples_by_fold",
        "expected_positive_examples_by_fold",
        "expected_negative_examples_by_fold",
        "folds",
        "homology_identity_threshold",
        "embedding_model",
        "representation_layer",
        "embedding_dimension",
        "embedding_batch_size",
        "embedding_seed",
        "model_checkpoint_sha256",
        "contact_regression_sha256",
        "environment_lock_sha256",
        "trust_manifest_sha256",
        "embedding_worker_sha256",
        "python_version",
        "torch_version",
        "fair_esm_version",
        "numpy_version",
        "cuda_runtime",
        "cudnn_version",
        "device_type",
        "historical_design_hint_status",
        "historical_design_hint_input_fasta_sha256",
        "historical_design_hint_input_fasta_manifest_sha256",
        "historical_design_hint_embedding_records",
        "historical_design_hint_extra_embedding_sequence_ids",
        "historical_design_hint_extra_embedding_sequence_ids_sha256",
        "historical_design_hint_missing_union_sequence_ids",
        "historical_design_hint_missing_union_sequence_ids_sha256",
    }
)


class VerificationError(ValueError):
    """Raised when an independent union-ESM verification invariant fails."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    payload: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class UnionEsmContract:
    path: Path
    sha256: str
    gate1_publication_top_sha256: str
    gate1_top_sha256: str
    gate1_examples_sha256: str
    gate1_folds_sha256: str
    gate1_oof_sha256: str
    gate1_manifest_sha256: str
    gate1_split_receipt_sha256: str
    gate1_independent_receipt_sha256: str
    gate1_config_sha256: str
    gate1_code_manifest_sha256: str
    gate1_frozen_input_manifest_sha256: str
    gate1_git_commit: str
    expected_examples: int
    expected_source_observations: int
    expected_sequences: int
    expected_positive_examples: int
    expected_negative_examples: int
    expected_examples_by_fold: tuple[int, ...]
    expected_positive_examples_by_fold: tuple[int, ...]
    expected_negative_examples_by_fold: tuple[int, ...]
    folds: int
    homology_identity_threshold: float
    embedding_model: str
    representation_layer: int
    embedding_dimension: int
    embedding_batch_size: int
    embedding_seed: int
    model_checkpoint_sha256: str
    contact_regression_sha256: str
    environment_lock_sha256: str
    trust_manifest_sha256: str
    embedding_worker_sha256: str
    python_version: str
    torch_version: str
    fair_esm_version: str
    numpy_version: str
    cuda_runtime: str
    cudnn_version: int
    device_type: str
    historical_design_hint_status: str
    historical_design_hint_input_fasta_sha256: str
    historical_design_hint_input_fasta_manifest_sha256: str
    historical_design_hint_embedding_records: int
    historical_design_hint_extra_embedding_sequence_ids: tuple[str, ...]
    historical_design_hint_extra_embedding_sequence_ids_sha256: str
    historical_design_hint_missing_union_sequence_ids: tuple[str, ...]
    historical_design_hint_missing_union_sequence_ids_sha256: str


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _require_no_symlink(path: Path, *, label: str, ancestors: bool = False) -> None:
    candidate = path.absolute()
    targets = [candidate, *candidate.parents] if ancestors else [candidate]
    for target in targets:
        _require(not target.is_symlink(), f"{label} must not traverse a symbolic link")


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    requested = Path(path)
    _require_no_symlink(requested, label=label, ancestors=True)
    resolved = requested.resolve(strict=True)
    _require(resolved.is_dir() and not resolved.is_symlink(), f"{label} is not a real directory")
    return resolved


def _read_snapshot(path: Path, *, label: str) -> Snapshot:
    _require_no_symlink(path, label=label, ancestors=True)
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    _require(
        _fingerprint(before) == _fingerprint(after) and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return Snapshot(resolved, payload, _sha256(payload), _fingerprint(after))


def _assert_unchanged(snapshot: Snapshot, *, label: str) -> None:
    try:
        before = snapshot.path.stat()
    except FileNotFoundError as error:
        raise VerificationError(f"{label} disappeared during verification") from error
    payload = snapshot.path.read_bytes()
    after = snapshot.path.stat()
    _require(
        stat.S_ISREG(before.st_mode)
        and _fingerprint(before) == snapshot.fingerprint
        and _fingerprint(after) == snapshot.fingerprint
        and _sha256(payload) == snapshot.sha256,
        f"{label} changed during verification",
    )


def _reject_constant(value: str) -> object:
    raise VerificationError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _loads_json(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, VerificationError) as error:
        raise VerificationError(f"{label} is not strict UTF-8 JSON") from error


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(payload: bytes, *, label: str, canonical: bool = True) -> dict[str, object]:
    _require(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") and b"\r" not in payload,
        f"{label} must have exactly one final LF",
    )
    value = _loads_json(payload, label=label)
    _require(isinstance(value, dict), f"{label} must contain one object")
    result = cast(dict[str, object], value)
    if canonical:
        _require(payload == _pretty_json_bytes(result), f"{label} is not canonical pretty JSON")
    return result


def _jsonl_rows(payload: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} LF framing")
    rows: list[dict[str, object]] = []
    for number, raw in enumerate(payload[:-1].split(b"\n"), start=1):
        _require(raw != b"", f"{label} row {number} is blank")
        value = _loads_json(raw, label=f"{label} row {number}")
        _require(isinstance(value, dict), f"{label} row {number} is not an object")
        row = cast(dict[str, object], value)
        canonical = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        _require(raw == canonical, f"{label} row {number} is not canonical JSONL")
        rows.append(row)
    return tuple(rows)


def _safe_manifest_name(value: str, *, label: str) -> str:
    pure = PurePosixPath(value)
    _require(
        value != ""
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "." not in pure.parts
        and "\\" not in value,
        f"{label} contains unsafe path {value!r}",
    )
    return value


def _parse_sha_manifest(payload: bytes, *, label: str) -> dict[str, str]:
    _require(payload and payload.endswith(b"\n") and b"\r" not in payload, f"{label} LF framing")
    try:
        lines = payload[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not UTF-8") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        _require(match is not None, f"{label} row {number} is malformed")
        assert match is not None
        digest, mode, raw_name = match.groups()
        _require(mode == " ", f"{label} row {number} is not text-mode syntax")
        name = _safe_manifest_name(raw_name, label=label)
        _require(name not in result, f"{label} repeats {name!r}")
        if previous is not None:
            _require(name > previous, f"{label} paths are not strictly sorted")
        result[name] = digest
        previous = name
    return result


def _sha_manifest_bytes(entries: Mapping[str, str]) -> bytes:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("utf-8")


def _tree_inventory(root: Path) -> frozenset[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "verified tree contains a symbolic link")
        if path.is_dir():
            continue
        _require(path.is_file(), "verified tree contains a non-regular entry")
        result.add(path.relative_to(root).as_posix())
    return frozenset(result)


def _verify_immutable_tree(root: Path, *, label: str) -> None:
    for path in (root, *root.rglob("*")):
        _require(not path.is_symlink(), f"{label} contains a symbolic link")
        _require(path.stat().st_mode & 0o222 == 0, f"{label} contains a writable entry")


def _verify_manifest_tree(
    root: Path,
    *,
    expected_top_sha256: str | None,
    expected_inventory: frozenset[str],
    label: str,
) -> tuple[Snapshot, dict[str, str]]:
    top = _read_snapshot(root / "SHA256SUMS", label=f"{label} top manifest")
    if expected_top_sha256 is not None:
        _require(top.sha256 == expected_top_sha256, f"{label} top checksum changed")
    entries = _parse_sha_manifest(top.payload, label=f"{label} top manifest")
    inventory = _tree_inventory(root)
    _require(inventory == expected_inventory, f"{label} inventory differs from contract")
    _require(set(entries) == set(expected_inventory) - {"SHA256SUMS"}, f"{label} coverage gap")
    for name, digest in entries.items():
        observed = _read_snapshot(root / name, label=f"{label} {name}")
        _require(observed.sha256 == digest, f"{label} checksum mismatch for {name}")
    return top, entries


def _verify_tree_bytes(
    left: Path,
    right: Path,
    *,
    expected_inventory: frozenset[str],
    label: str,
) -> None:
    _require(_tree_inventory(left) == expected_inventory, f"left {label} inventory changed")
    _require(_tree_inventory(right) == expected_inventory, f"right {label} inventory changed")
    for name in sorted(expected_inventory):
        left_payload = _read_snapshot(left / name, label=f"left {label} {name}").payload
        right_payload = _read_snapshot(right / name, label=f"right {label} {name}").payload
        _require(left_payload == right_payload, f"{label} twins differ at {name}")


def _exact_fields(value: Mapping[str, object], expected: Iterable[str], *, label: str) -> None:
    required = set(expected)
    _require(
        set(value) == required,
        f"{label} schema mismatch: missing={sorted(required - set(value))}, "
        f"extra={sorted(set(value) - required)}",
    )


def _sha_field(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"bad {label}")
    return cast(str, value)


def _string(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and value != "" and value == value.strip(), f"bad {label}")
    return cast(str, value)


def _positive_int(value: object, *, label: str) -> int:
    _require(type(value) is int and cast(int, value) > 0, f"{label} must be a positive integer")
    return cast(int, value)


def _count_vector(value: object, *, folds: int, label: str) -> tuple[int, ...]:
    _require(isinstance(value, list) and len(value) == folds, f"bad {label} vector")
    items = cast(list[object], value)
    _require(
        all(type(item) is int and cast(int, item) >= 0 for item in items),
        f"bad {label} vector",
    )
    return tuple(cast(list[int], items))


def _id_pair(value: object, *, label: str) -> tuple[str, str]:
    _require(isinstance(value, list) and len(value) == 2, f"bad {label}")
    items = cast(list[object], value)
    _require(
        all(isinstance(item, str) and _SHA256_RE.fullmatch(item) is not None for item in items),
        f"bad {label}",
    )
    result = tuple(cast(list[str], items))
    _require(result == tuple(sorted(set(result))), f"bad {label} ordering")
    return cast(tuple[str, str], result)


def _set_digest(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    payload = b"" if not ordered else ("\n".join(ordered) + "\n").encode("ascii")
    return _sha256(payload)


def _parse_config(snapshot: Snapshot) -> UnionEsmContract:
    try:
        raw = tomllib.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError("union ESM config is not valid UTF-8 TOML") from error
    _exact_fields(raw, _CONFIG_FIELDS, label="union ESM config")
    _require(raw["schema_version"] == 1 and type(raw["schema_version"]) is int, "bad schema")
    folds = _positive_int(raw["folds"], label="folds")
    totals = _count_vector(raw["expected_examples_by_fold"], folds=folds, label="examples")
    positives = _count_vector(
        raw["expected_positive_examples_by_fold"], folds=folds, label="positive examples"
    )
    negatives = _count_vector(
        raw["expected_negative_examples_by_fold"], folds=folds, label="negative examples"
    )
    expected_examples = _positive_int(raw["expected_examples"], label="expected examples")
    expected_positives = _positive_int(
        raw["expected_positive_examples"], label="expected positive examples"
    )
    expected_negatives = _positive_int(
        raw["expected_negative_examples"], label="expected negative examples"
    )
    _require(sum(totals) == expected_examples, "configured example total is inconsistent")
    _require(sum(positives) == expected_positives, "configured positive total is inconsistent")
    _require(sum(negatives) == expected_negatives, "configured negative total is inconsistent")
    _require(
        expected_examples == expected_positives + expected_negatives
        and all(
            total == pos + neg for total, pos, neg in zip(totals, positives, negatives, strict=True)
        ),
        "configured label totals are inconsistent",
    )
    threshold = raw["homology_identity_threshold"]
    _require(
        type(threshold) in {int, float}
        and math.isfinite(float(cast(int | float, threshold)))
        and float(cast(int | float, threshold)) == 0.8,
        "homology identity threshold must be 0.8",
    )
    git_commit = _string(raw["gate1_git_commit"], label="Gate-1 Git commit")
    _require(_GIT_RE.fullmatch(git_commit) is not None, "bad Gate-1 Git commit")
    extra_ids = _id_pair(
        raw["historical_design_hint_extra_embedding_sequence_ids"],
        label="historical design-hint extra IDs",
    )
    missing_ids = _id_pair(
        raw["historical_design_hint_missing_union_sequence_ids"],
        label="historical design-hint missing IDs",
    )
    hashes = {
        name: _sha_field(raw[name], label=name)
        for name in (
            "gate1_publication_top_sha256",
            "gate1_top_sha256",
            "gate1_examples_sha256",
            "gate1_folds_sha256",
            "gate1_oof_sha256",
            "gate1_manifest_sha256",
            "gate1_split_receipt_sha256",
            "gate1_independent_receipt_sha256",
            "gate1_config_sha256",
            "gate1_code_manifest_sha256",
            "gate1_frozen_input_manifest_sha256",
            "model_checkpoint_sha256",
            "contact_regression_sha256",
            "environment_lock_sha256",
            "trust_manifest_sha256",
            "embedding_worker_sha256",
            "historical_design_hint_input_fasta_sha256",
            "historical_design_hint_input_fasta_manifest_sha256",
            "historical_design_hint_extra_embedding_sequence_ids_sha256",
            "historical_design_hint_missing_union_sequence_ids_sha256",
        )
    }
    _require(
        raw["historical_design_hint_status"] == "unverified_historical_design_hint_not_evidence",
        "historical design hint is not explicitly unverified",
    )
    _require(
        _set_digest(extra_ids)
        == hashes["historical_design_hint_extra_embedding_sequence_ids_sha256"],
        "historical design-hint extra-ID digest mismatch",
    )
    _require(
        _set_digest(missing_ids)
        == hashes["historical_design_hint_missing_union_sequence_ids_sha256"],
        "historical design-hint missing-ID digest mismatch",
    )
    return UnionEsmContract(
        path=snapshot.path,
        sha256=snapshot.sha256,
        gate1_publication_top_sha256=hashes["gate1_publication_top_sha256"],
        gate1_top_sha256=hashes["gate1_top_sha256"],
        gate1_examples_sha256=hashes["gate1_examples_sha256"],
        gate1_folds_sha256=hashes["gate1_folds_sha256"],
        gate1_oof_sha256=hashes["gate1_oof_sha256"],
        gate1_manifest_sha256=hashes["gate1_manifest_sha256"],
        gate1_split_receipt_sha256=hashes["gate1_split_receipt_sha256"],
        gate1_independent_receipt_sha256=hashes["gate1_independent_receipt_sha256"],
        gate1_config_sha256=hashes["gate1_config_sha256"],
        gate1_code_manifest_sha256=hashes["gate1_code_manifest_sha256"],
        gate1_frozen_input_manifest_sha256=hashes["gate1_frozen_input_manifest_sha256"],
        gate1_git_commit=git_commit,
        expected_examples=expected_examples,
        expected_source_observations=_positive_int(
            raw["expected_source_observations"], label="expected source observations"
        ),
        expected_sequences=_positive_int(raw["expected_sequences"], label="expected sequences"),
        expected_positive_examples=expected_positives,
        expected_negative_examples=expected_negatives,
        expected_examples_by_fold=totals,
        expected_positive_examples_by_fold=positives,
        expected_negative_examples_by_fold=negatives,
        folds=folds,
        homology_identity_threshold=float(cast(int | float, threshold)),
        embedding_model=_string(raw["embedding_model"], label="embedding model"),
        representation_layer=_positive_int(
            raw["representation_layer"], label="representation layer"
        ),
        embedding_dimension=_positive_int(raw["embedding_dimension"], label="embedding dimension"),
        embedding_batch_size=_positive_int(
            raw["embedding_batch_size"], label="embedding batch size"
        ),
        embedding_seed=_positive_int(raw["embedding_seed"], label="embedding seed"),
        model_checkpoint_sha256=hashes["model_checkpoint_sha256"],
        contact_regression_sha256=hashes["contact_regression_sha256"],
        environment_lock_sha256=hashes["environment_lock_sha256"],
        trust_manifest_sha256=hashes["trust_manifest_sha256"],
        embedding_worker_sha256=hashes["embedding_worker_sha256"],
        python_version=_string(raw["python_version"], label="Python version"),
        torch_version=_string(raw["torch_version"], label="Torch version"),
        fair_esm_version=_string(raw["fair_esm_version"], label="fair-esm version"),
        numpy_version=_string(raw["numpy_version"], label="NumPy version"),
        cuda_runtime=_string(raw["cuda_runtime"], label="CUDA runtime"),
        cudnn_version=_positive_int(raw["cudnn_version"], label="cuDNN version"),
        device_type=_string(raw["device_type"], label="device type"),
        historical_design_hint_status=cast(str, raw["historical_design_hint_status"]),
        historical_design_hint_input_fasta_sha256=hashes[
            "historical_design_hint_input_fasta_sha256"
        ],
        historical_design_hint_input_fasta_manifest_sha256=hashes[
            "historical_design_hint_input_fasta_manifest_sha256"
        ],
        historical_design_hint_embedding_records=_positive_int(
            raw["historical_design_hint_embedding_records"],
            label="historical design-hint record count",
        ),
        historical_design_hint_extra_embedding_sequence_ids=extra_ids,
        historical_design_hint_extra_embedding_sequence_ids_sha256=hashes[
            "historical_design_hint_extra_embedding_sequence_ids_sha256"
        ],
        historical_design_hint_missing_union_sequence_ids=missing_ids,
        historical_design_hint_missing_union_sequence_ids_sha256=hashes[
            "historical_design_hint_missing_union_sequence_ids_sha256"
        ],
    )


def load_config(path: str | Path) -> UnionEsmContract:
    """Load and independently validate the union ESM export contract."""

    return _parse_config(_read_snapshot(Path(path), label="union ESM config"))


def _canonical_sequence(value: object, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} sequence is not a string")
    sequence = cast(str, value)
    try:
        sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise VerificationError(f"{label} sequence is not ASCII") from error
    _require(
        sequence == sequence.strip().upper()
        and _MIN_LENGTH <= len(sequence) <= _MAX_LENGTH
        and set(sequence) <= _AMINO_ACIDS,
        f"{label} sequence is not canonical",
    )
    return sequence


def _read_examples(snapshot: Snapshot, *, config: UnionEsmContract) -> tuple[Example, ...]:
    rows = _jsonl_rows(snapshot.payload, label="Gate-1 examples")
    _require(len(rows) == config.expected_examples, "Gate-1 example census changed")
    examples: list[Example] = []
    prior_id: str | None = None
    sequence_metadata: dict[str, tuple[str, int, str, str]] = {}
    union_folds: dict[str, set[int]] = defaultdict(set)
    homology_locations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for number, row in enumerate(rows, start=1):
        label = f"Gate-1 example {number}"
        _exact_fields(row, _EXAMPLE_FIELDS, label=label)
        _require(row["schema_version"] == 1 and type(row["schema_version"]) is int, "bad schema")
        example_id = _sha_field(row["example_id"], label=f"{label} example_id")
        context_id = _sha_field(row["assay_context_id"], label=f"{label} context_id")
        _require(example_id == context_id, f"{label} example/context IDs differ")
        if prior_id is not None:
            _require(example_id > prior_id, "Gate-1 examples are not strictly sorted")
        sequence = _canonical_sequence(row["sequence"], label=label)
        sequence_id = _sha_field(row["sequence_id"], label=f"{label} sequence_id")
        _require(sequence_id == _sha256(sequence.encode("ascii")), f"{label} hash mismatch")
        target = _string(row["canonical_target"], label=f"{label} target")
        gram = row["gram"]
        binary_label = row["label"]
        observations = row["source_observations"]
        fold = row["fold"]
        _require(gram in {"negative", "positive"}, f"{label} Gram class is invalid")
        _require(type(binary_label) is int and binary_label in {0, 1}, f"{label} label invalid")
        _require(
            type(observations) is int and cast(int, observations) > 0, f"{label} count invalid"
        )
        _require(
            type(fold) is int and 0 <= cast(int, fold) < config.folds,
            f"{label} fold invalid",
        )
        homology_id = _sha_field(row["homology_component_id"], label=f"{label} homology component")
        union_id = _sha_field(row["union_component_id"], label=f"{label} union component")
        metadata = (sequence, cast(int, fold), homology_id, union_id)
        prior = sequence_metadata.setdefault(sequence_id, metadata)
        _require(prior == metadata, "one sequence has inconsistent fold/component metadata")
        union_folds[union_id].add(cast(int, fold))
        homology_locations[homology_id].add((union_id, cast(int, fold)))
        examples.append(
            Example(
                example_id=example_id,
                assay_context_id=context_id,
                sequence_id=sequence_id,
                sequence=sequence,
                canonical_target=target,
                gram=cast(str, gram),
                label=cast(int, binary_label),
                source_observations=cast(int, observations),
                fold=cast(int, fold),
                homology_component_id=homology_id,
                union_component_id=union_id,
            )
        )
        prior_id = example_id
    _require(all(len(value) == 1 for value in union_folds.values()), "union component spans folds")
    _require(
        all(len(value) == 1 for value in homology_locations.values()),
        "homology component spans union components or folds",
    )
    _require(len(sequence_metadata) == config.expected_sequences, "modeled sequence count changed")
    _require(
        sum(item.source_observations for item in examples) == config.expected_source_observations,
        "source-observation count changed",
    )
    positive = sum(item.label for item in examples)
    _require(
        positive == config.expected_positive_examples
        and len(examples) - positive == config.expected_negative_examples,
        "Gate-1 label census changed",
    )
    for fold_index in range(config.folds):
        selected = [item for item in examples if item.fold == fold_index]
        fold_positive = sum(item.label for item in selected)
        _require(
            len(selected) == config.expected_examples_by_fold[fold_index]
            and fold_positive == config.expected_positive_examples_by_fold[fold_index]
            and len(selected) - fold_positive
            == config.expected_negative_examples_by_fold[fold_index],
            f"Gate-1 fold {fold_index} census changed",
        )
    return tuple(examples)


def _metadata_tuple(item: Example) -> tuple[str, ...]:
    return (
        item.example_id,
        item.assay_context_id,
        item.sequence_id,
        item.sequence,
        item.canonical_target,
        item.gram,
        str(item.label),
        str(item.source_observations),
        str(item.fold),
        item.homology_component_id,
        item.union_component_id,
    )


def _verify_oof_metadata(
    snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    config: UnionEsmContract,
) -> dict[str, float]:
    payload = snapshot.payload
    _require(
        payload.endswith(b"\n")
        and not payload.endswith(b"\n\n")
        and b"\r" not in payload
        and b"\x00" not in payload,
        "Gate-1 OOF CSV framing changed",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("Gate-1 OOF is not UTF-8") from error
    expected = {item.example_id: item for item in examples}
    support: dict[str, set[str]] = defaultdict(set)
    identities: dict[str, tuple[str, float]] = {}
    previous: tuple[str, str] | None = None
    rows = 0
    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require(reader.fieldnames == list(_OOF_FIELDS), "Gate-1 OOF schema changed")
    for number, row in enumerate(reader, start=2):
        rows += 1
        _require(
            None not in row and all(value is not None for value in row.values()), "bad OOF row"
        )
        model = cast(str, row["model"])
        example_id = cast(str, row["example_id"])
        _require(model in _BASE_MODELS and example_id in expected, f"unknown OOF row {number}")
        order = (model, example_id)
        if previous is not None:
            _require(order > previous, "Gate-1 OOF rows are not strictly model/example sorted")
        _require(model not in support[example_id], f"duplicate OOF row {number}")
        observed_metadata = tuple(cast(str, row[field]) for field in _OOF_FIELDS[1:12])
        _require(
            observed_metadata == _metadata_tuple(expected[example_id]),
            f"Gate-1 OOF row {number} metadata differs from examples",
        )
        identity_text = cast(str, row["max_train_identity"])
        try:
            identity = float(identity_text)
        except ValueError as error:
            raise VerificationError(f"Gate-1 OOF row {number} identity is invalid") from error
        _require(
            math.isfinite(identity) and 0 <= identity < config.homology_identity_threshold,
            f"Gate-1 OOF row {number} violates identity threshold",
        )
        prior = identities.setdefault(example_id, (identity_text, identity))
        _require(prior[0] == identity_text, "OOF model rows disagree on identity text")
        # The probability cell is deliberately opaque: the accepted OOF file
        # is hash-bound, but predictions cannot influence FASTA membership.
        support[example_id].add(model)
        previous = order
    _require(rows == len(examples) * len(_BASE_MODELS), "Gate-1 OOF row census changed")
    _require(set(support) == set(expected), "Gate-1 OOF context support is incomplete")
    _require(
        all(models == set(_BASE_MODELS) for models in support.values()),
        "Gate-1 OOF model support is incomplete",
    )
    return {key: value[1] for key, value in identities.items()}


def _fold_summary(examples: Sequence[Example], fold: int) -> dict[str, int]:
    selected = [item for item in examples if item.fold == fold]
    class_counts = {
        metric: sum(
            item.gram == metric.split("_")[1] and item.label == int(metric.endswith("positive"))
            for item in selected
        )
        for metric in _CLASS_METRICS
    }
    return {
        "examples": len(selected),
        "source_observations": sum(item.source_observations for item in selected),
        "sequences": len({item.sequence_id for item in selected}),
        "homology_components": len({item.homology_component_id for item in selected}),
        "union_components": len({item.union_component_id for item in selected}),
        "positives": sum(item.label for item in selected),
        "negatives": sum(1 - item.label for item in selected),
        **class_counts,
    }


def _verify_folds(
    snapshot: Snapshot,
    *,
    examples: Sequence[Example],
    identities: Mapping[str, float],
    config: UnionEsmContract,
) -> None:
    document = _json_object(snapshot.payload, label="Gate-1 folds")
    _exact_fields(
        document,
        {
            "schema_version",
            "artifact",
            "assignment_policy",
            "identity_threshold",
            "maximum_cross_fold_identity",
            "folds",
            "canonical_targets_by_fold",
            "assignments",
        },
        label="Gate-1 folds",
    )
    maximum = document["maximum_cross_fold_identity"]
    _require(
        document["schema_version"] == 1
        and document["artifact"] == _BASE_FOLD_ARTIFACT
        and document["assignment_policy"] == _FOLD_POLICY
        and document["identity_threshold"] == config.homology_identity_threshold
        and type(maximum) in {int, float}
        and 0 <= float(cast(int | float, maximum)) < config.homology_identity_threshold,
        "Gate-1 fold policy/identity contract changed",
    )
    assignments = document["assignments"]
    _require(isinstance(assignments, list) and len(assignments) == len(examples), "bad folds")
    observed: dict[str, tuple[str, str, str, int]] = {}
    for number, raw in enumerate(cast(list[object], assignments), start=1):
        _require(isinstance(raw, dict), f"Gate-1 assignment {number} is not an object")
        item = cast(dict[str, object], raw)
        _exact_fields(item, _FOLD_ASSIGNMENT_FIELDS, label=f"Gate-1 assignment {number}")
        example_id = _sha_field(item["example_id"], label="assignment example_id")
        _require(example_id not in observed, "duplicate Gate-1 fold assignment")
        fold = item["fold"]
        _require(type(fold) is int, "Gate-1 assignment fold is invalid")
        observed[example_id] = (
            _sha_field(item["sequence_id"], label="assignment sequence_id"),
            _sha_field(item["homology_component_id"], label="assignment homology ID"),
            _sha_field(item["union_component_id"], label="assignment union ID"),
            cast(int, fold),
        )
    expected = {
        item.example_id: (
            item.sequence_id,
            item.homology_component_id,
            item.union_component_id,
            item.fold,
        )
        for item in examples
    }
    _require(observed == expected and set(identities) == set(expected), "fold join changed")
    summaries = {str(fold): _fold_summary(examples, fold) for fold in range(config.folds)}
    targets = {
        str(fold): dict(
            sorted(Counter(item.canonical_target for item in examples if item.fold == fold).items())
        )
        for fold in range(config.folds)
    }
    _require(
        math.isclose(
            float(cast(int | float, maximum)),
            max(identities.values()),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "Gate-1 maximum identity differs from OOF metadata",
    )
    _require(document["folds"] == summaries, "Gate-1 fold summary changed")
    _require(document["canonical_targets_by_fold"] == targets, "Gate-1 target summary changed")


def _artifact_binding(
    artifacts: object,
    *,
    key: str,
    filename: str,
    digest: str,
) -> None:
    _require(isinstance(artifacts, dict), "Gate-1 artifacts field is not an object")
    value = cast(dict[str, object], artifacts).get(key)
    _require(isinstance(value, dict), f"Gate-1 manifest omits {key}")
    item = cast(dict[str, object], value)
    _require(
        set(item) == {"filename", "sha256", "role"}
        and item.get("filename") == filename
        and item.get("sha256") == digest
        and isinstance(item.get("role"), str)
        and bool(item.get("role")),
        f"Gate-1 manifest does not bind {filename}",
    )


def _verify_gate1_documents(
    *,
    snapshots: Mapping[str, Snapshot],
    config: UnionEsmContract,
    actual_handshake: Mapping[str, object],
) -> None:
    manifest = _json_object(snapshots["manifest"].payload, label="Gate-1 manifest")
    _exact_fields(
        manifest,
        {
            "schema_version",
            "artifact",
            "status",
            "config_sha256",
            "git_commit",
            "input_sha256",
            "accepted_split",
            "code_attestation",
            "policies",
            "models",
            "label_summary",
            "identity_audit",
            "runtime",
            "artifacts",
        },
        label="Gate-1 manifest",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["artifact"] == _BASE_ARTIFACT
        and manifest["status"] == _BASE_STATUS
        and manifest["config_sha256"] == config.gate1_config_sha256
        and manifest["git_commit"] == config.gate1_git_commit
        and manifest["models"] == list(_BASE_MODELS),
        "Gate-1 manifest identity changed",
    )
    policies = manifest["policies"]
    _require(isinstance(policies, dict), "Gate-1 policies are invalid")
    policy = cast(dict[str, object], policies)
    _require(
        policy.get("label") == _LABEL_POLICY
        and policy.get("fold_assignment") == _FOLD_POLICY
        and policy.get("bootstrap_unit") == "union_component_id"
        and policy.get("ensemble") == "untrained_equal_probability_mean"
        and policy.get("model_features") == ["sequence", "canonical_target", "gram"]
        and policy.get("forbidden_model_features") == list(_FORBIDDEN_MODEL_FEATURES)
        and policy.get("context_feature") == "canonical_target",
        "Gate-1 leakage/split policy changed",
    )
    code = manifest["code_attestation"]
    _require(isinstance(code, dict), "Gate-1 code attestation is invalid")
    code_object = cast(dict[str, object], code)
    code_invariants = code_object.get("invariants")
    _require(
        code_object.get("code_manifest_sha256") == config.gate1_code_manifest_sha256
        and isinstance(code_invariants, dict)
        and bool(code_invariants)
        and all(value is True for value in cast(dict[str, object], code_invariants).values()),
        "Gate-1 code attestation changed",
    )
    for key, filename in (
        ("examples", "examples.jsonl"),
        ("folds", "folds.json"),
        ("oof", "oof_predictions.csv"),
        ("split_receipt", "split_receipt.json"),
    ):
        _artifact_binding(
            manifest["artifacts"],
            key=key,
            filename=filename,
            digest=snapshots[key].sha256,
        )
    labels = manifest["label_summary"]
    _require(isinstance(labels, dict), "Gate-1 label summary is invalid")
    label_object = cast(dict[str, object], labels)
    _require(
        label_object.get("context_examples") == config.expected_examples
        and label_object.get("source_observations") == config.expected_source_observations
        and label_object.get("modeled_sequences") == config.expected_sequences
        and label_object.get("positive_examples") == config.expected_positive_examples
        and label_object.get("negative_examples") == config.expected_negative_examples,
        "Gate-1 label summary changed",
    )

    receipt = _json_object(snapshots["split_receipt"].payload, label="Gate-1 split receipt")
    _exact_fields(
        receipt,
        {
            "schema_version",
            "artifact",
            "status",
            "assignment_policy",
            "input_sha256",
            "accepted_split",
            "code_attestation",
            "census",
            "identity",
            "invariants",
        },
        label="Gate-1 split receipt",
    )
    _require(
        receipt["schema_version"] == 1
        and receipt["artifact"] == _BASE_RECEIPT_ARTIFACT
        and receipt["status"] == "passed"
        and receipt["assignment_policy"] == _FOLD_POLICY
        and receipt["input_sha256"] == manifest["input_sha256"]
        and receipt["accepted_split"] == manifest["accepted_split"]
        and receipt["code_attestation"] == manifest["code_attestation"],
        "Gate-1 split receipt identity changed",
    )
    census = receipt["census"]
    _require(isinstance(census, dict), "Gate-1 split census is invalid")
    census_object = cast(dict[str, object], census)
    expected_census = {
        "context_examples": config.expected_examples,
        "source_observations": config.expected_source_observations,
        "modeled_sequences": config.expected_sequences,
        "positives": config.expected_positive_examples,
        "negatives": config.expected_negative_examples,
        "examples_by_fold": list(config.expected_examples_by_fold),
        "positive_examples_by_fold": list(config.expected_positive_examples_by_fold),
        "negative_examples_by_fold": list(config.expected_negative_examples_by_fold),
    }
    _require(
        all(census_object.get(key) == value for key, value in expected_census.items()),
        "Gate-1 split census changed",
    )
    invariants = receipt["invariants"]
    _require(
        isinstance(invariants, dict)
        and set(invariants) == set(_SPLIT_RECEIPT_INVARIANTS)
        and all(value is True for value in cast(dict[str, object], invariants).values()),
        "Gate-1 split receipt invariants changed",
    )

    independent = _json_object(
        snapshots["independent_receipt"].payload,
        label="Gate-1 independent receipt",
    )
    _exact_fields(
        independent,
        {
            "schema_version",
            "artifact",
            "status",
            "checks",
            "git_commit",
            "config_sha256",
            "publication_top_manifest_sha256",
            "gate1_top_manifest_sha256",
            "code_manifest_sha256",
            "frozen_input_manifest_sha256",
            "input_sha256",
            "artifact_sha256",
            "census",
            "identity",
            "overall_metrics",
            "union_component_bootstrap_95ci",
            "production_handshake",
        },
        label="Gate-1 independent receipt",
    )
    _require(
        independent["schema_version"] == 1
        and independent["artifact"] == _BASE_AUDIT_ARTIFACT
        and independent["status"] == "passed"
        and independent["git_commit"] == config.gate1_git_commit
        and independent["config_sha256"] == config.gate1_config_sha256
        and independent["publication_top_manifest_sha256"] == config.gate1_publication_top_sha256
        and independent["gate1_top_manifest_sha256"] == config.gate1_top_sha256
        and independent["code_manifest_sha256"] == config.gate1_code_manifest_sha256
        and independent["frozen_input_manifest_sha256"] == config.gate1_frozen_input_manifest_sha256
        and independent["production_handshake"] == actual_handshake,
        "Gate-1 independent receipt chain changed",
    )
    checks = independent["checks"]
    _require(
        isinstance(checks, dict)
        and set(checks) == set(_GATE1_INDEPENDENT_CHECKS)
        and all(value is True for value in cast(dict[str, object], checks).values()),
        "Gate-1 independent checks changed",
    )
    expected_artifacts = {
        "context_audit.jsonl": snapshots["context_audit"].sha256,
        "examples.jsonl": snapshots["examples"].sha256,
        "folds.json": snapshots["folds"].sha256,
        "manifest.json": snapshots["manifest"].sha256,
        "metrics.json": snapshots["metrics"].sha256,
        "oof_predictions.csv": snapshots["oof"].sha256,
        "split_receipt.json": snapshots["split_receipt"].sha256,
    }
    _require(
        independent["artifact_sha256"] == expected_artifacts,
        "Gate-1 independent artifact hashes changed",
    )
    independent_census = independent["census"]
    _require(isinstance(independent_census, dict), "Gate-1 independent census is invalid")
    independent_counts = cast(dict[str, object], independent_census)
    _require(
        independent_counts.get("context_examples") == config.expected_examples
        and independent_counts.get("source_observations") == config.expected_source_observations
        and independent_counts.get("modeled_sequences") == config.expected_sequences
        and independent_counts.get("positives") == config.expected_positive_examples
        and independent_counts.get("negatives") == config.expected_negative_examples
        and independent_counts.get("examples_by_fold") == list(config.expected_examples_by_fold)
        and independent_counts.get("positive_examples_by_fold")
        == list(config.expected_positive_examples_by_fold)
        and independent_counts.get("negative_examples_by_fold")
        == list(config.expected_negative_examples_by_fold),
        "Gate-1 independent census changed",
    )


def _verify_handshake(twin_root: Path, *, label: str) -> dict[str, object]:
    receipt_root = twin_root / "node-receipts"
    _require(
        receipt_root.is_dir() and not receipt_root.is_symlink(),
        f"{label} node-receipts directory missing",
    )
    _require(
        _tree_inventory(receipt_root) == frozenset({"0.receipt", "1.receipt", "0.ack", "1.ack"}),
        f"{label} handshake inventory changed",
    )
    _require(receipt_root.stat().st_mode & 0o222 == 0, f"{label} handshake directory writable")
    job_id = twin_root.name
    _require(job_id.isdigit(), f"{label} root basename is not a Slurm array job ID")
    receipt_payloads: dict[int, bytes] = {}
    node_names: dict[int, str] = {}
    snapshots: list[Snapshot] = []
    for task in (0, 1):
        snapshot = _read_snapshot(
            receipt_root / f"{task}.receipt", label=f"{label} task {task} receipt"
        )
        snapshots.append(snapshot)
        _require(snapshot.path.stat().st_mode & 0o222 == 0, f"{label} receipt writable")
        _require(
            snapshot.payload.endswith(b"\n") and b"\r" not in snapshot.payload,
            f"{label} receipt framing changed",
        )
        try:
            lines = snapshot.payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise VerificationError(f"{label} receipt is not UTF-8") from error
        _require(
            lines[:2] == [f"array_job_id={job_id}", f"array_task_id={task}"]
            and len(lines) == 3
            and lines[2].startswith("node_name="),
            f"{label} task {task} receipt is invalid",
        )
        node_name = lines[2].removeprefix("node_name=")
        _require(_SAFE_NODE_RE.fullmatch(node_name) is not None, f"{label} node name unsafe")
        receipt_payloads[task] = snapshot.payload
        node_names[task] = node_name
    _require(node_names[0] != node_names[1], f"{label} twins did not overlap on distinct nodes")
    receipt_hashes = {str(task): _sha256(payload) for task, payload in receipt_payloads.items()}
    ack_hashes: dict[str, str] = {}
    for task in (0, 1):
        snapshot = _read_snapshot(
            receipt_root / f"{task}.ack", label=f"{label} task {task} acknowledgement"
        )
        snapshots.append(snapshot)
        _require(snapshot.path.stat().st_mode & 0o222 == 0, f"{label} acknowledgement writable")
        expected = f"observed_sibling_receipt_sha256={receipt_hashes[str(1 - task)]}\n".encode(
            "ascii"
        )
        _require(snapshot.payload == expected, f"{label} acknowledgement {task} is invalid")
        ack_hashes[str(task)] = snapshot.sha256
    for snapshot in snapshots:
        _assert_unchanged(snapshot, label=f"{label} handshake entry")
    return {
        "distinct_nodes": True,
        "bidirectional_acknowledgement": True,
        "receipt_sha256": receipt_hashes,
        "acknowledgement_sha256": ack_hashes,
    }


def _git(repository: Path, arguments: Sequence[str], *, label: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise VerificationError(f"Git check failed for {label}") from error


def _committed_blob(repository: Path, commit: str, logical_path: str) -> bytes:
    kind = _git(repository, ["cat-file", "-t", f"{commit}:{logical_path}"], label=logical_path)
    _require(kind == b"blob\n", f"committed path is not a blob: {logical_path}")
    return _git(repository, ["cat-file", "blob", f"{commit}:{logical_path}"], label=logical_path)


def _verify_repository(repository: Path, expected_commit: str) -> None:
    _require(_GIT_RE.fullmatch(expected_commit) is not None, "expected Git commit is invalid")
    observed = _git(repository, ["rev-parse", "--verify", "HEAD^{commit}"], label="HEAD")
    _require(observed.decode().strip() == expected_commit, "repository HEAD differs from expected")
    _git(repository, ["diff", "--no-ext-diff", "--quiet", "--exit-code", "--"], label="tree")
    _git(
        repository,
        ["diff", "--cached", "--no-ext-diff", "--quiet", "--exit-code", "--"],
        label="index",
    )
    _require(
        _git(repository, ["ls-files", "--others", "--exclude-standard", "-z"], label="untracked")
        == b"",
        "repository contains untracked files",
    )
    origin = _git(
        repository,
        ["rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
        label="origin/main",
    )
    _require(origin.decode().strip() == expected_commit, "expected commit is not on origin/main")


def _verify_execution_source(repository: Path, expected_commit: str) -> None:
    requested = Path(__file__)
    _require_no_symlink(requested, label="executing verifier", ancestors=True)
    executing = requested.resolve(strict=True)
    expected = (repository / _VERIFIER_MODULE_PATH).resolve(strict=True)
    _require(executing == expected, "executing verifier is a stale installation")
    source = _read_snapshot(expected, label="verifier source")
    _require(
        source.payload == _committed_blob(repository, expected_commit, _VERIFIER_MODULE_PATH),
        "executing verifier differs from its committed blob",
    )


def _verify_code_manifest(
    snapshot: Snapshot,
    *,
    repository: Path,
    expected_commit: str,
    config: UnionEsmContract,
) -> dict[str, object]:
    entries = _parse_sha_manifest(snapshot.payload, label="export code manifest")
    source_root = repository / "src/amp_challenge"
    _require(source_root.is_dir() and not source_root.is_symlink(), "Python source tree missing")
    source_paths: set[str] = set()
    for source in source_root.rglob("*.py"):
        _require(not source.is_symlink(), "Python source tree contains a symbolic link")
        if source.is_file():
            source_paths.add(source.relative_to(repository).as_posix())
    expected_paths = set(_CODE_FIXED_PATHS) | source_paths
    _require(set(entries) == expected_paths, "export code-manifest inventory changed")
    _require(_PRODUCER_MODULE_PATH in entries, "export code manifest omits producer")
    _require(_VERIFIER_MODULE_PATH in entries, "export code manifest omits verifier")
    for logical_path in sorted(expected_paths):
        source = _read_snapshot(repository / logical_path, label=f"code entry {logical_path}")
        _require(source.sha256 == entries[logical_path], f"code hash mismatch: {logical_path}")
        _require(
            source.payload == _committed_blob(repository, expected_commit, logical_path),
            f"code entry differs from committed blob: {logical_path}",
        )
    _require(entries[_LOGICAL_CONFIG_PATH] == config.sha256, "code manifest config link changed")
    return {
        "schema_version": 1,
        "code_manifest_sha256": snapshot.sha256,
        "inventory_entries": len(entries),
        "executing_module": {
            "logical_path": _PRODUCER_MODULE_PATH,
            "sha256": entries[_PRODUCER_MODULE_PATH],
        },
        "logical_config": {
            "logical_path": _LOGICAL_CONFIG_PATH,
            "sha256": config.sha256,
        },
        "launcher_git_cleanliness_policy": (
            "clean committed Git state is enforced by the production Slurm launcher; "
            "the exporter verifies bytes and runs no Git subprocess"
        ),
        "invariants": {
            "code_manifest_inventory_exact": True,
            "every_repository_code_entry_hash_verified": True,
            "executing_module_bound_to_manifest": True,
            "supplied_config_bound_to_logical_manifest_entry": True,
        },
    }


def _prefixed_hashes(root: Path, prefix: str) -> dict[str, str]:
    return {
        f"{prefix}/{name}": _read_snapshot(root / name, label=f"frozen input {name}").sha256
        for name in sorted(_tree_inventory(root))
    }


def _verify_frozen_manifest(
    snapshot: Snapshot,
    *,
    gate1_run: Path,
    gate1_independent_receipt: Snapshot,
) -> None:
    entries = _parse_sha_manifest(snapshot.payload, label="export frozen-input manifest")
    expected = _prefixed_hashes(gate1_run, "gate1")
    expected["gate1-independent-receipt.json"] = gate1_independent_receipt.sha256
    _require(entries == dict(sorted(expected.items())), "export frozen-input manifest changed")


def _assignment_digest(examples: Sequence[Example]) -> str:
    payload = "".join(
        "\t".join(
            (
                item.example_id,
                item.assay_context_id,
                item.sequence_id,
                item.canonical_target,
                str(item.label),
                str(item.fold),
                item.homology_component_id,
                item.union_component_id,
            )
        )
        + "\n"
        for item in sorted(examples, key=lambda row: row.example_id)
    ).encode("utf-8")
    return _sha256(payload)


def _reconstruct_panel(
    examples: Sequence[Example],
    *,
    config: UnionEsmContract,
) -> tuple[tuple[tuple[str, str], ...], dict[str, object]]:
    sequences: dict[str, str] = {}
    union_folds: dict[str, set[int]] = defaultdict(set)
    for item in examples:
        prior = sequences.setdefault(item.sequence_id, item.sequence)
        _require(prior == item.sequence, "one sequence ID maps to multiple sequences")
        union_folds[item.union_component_id].add(item.fold)
    _require(
        all(len(folds) == 1 for folds in union_folds.values()),
        "union component spans folds",
    )
    records = tuple(sorted(sequences.items()))
    _require(len(records) == config.expected_sequences, "reconstructed sequence census changed")
    panel = {
        "context_examples": len(examples),
        "source_observations": sum(item.source_observations for item in examples),
        "unique_sequences": len(records),
        "positive_examples": sum(item.label for item in examples),
        "negative_examples": sum(1 - item.label for item in examples),
        "homology_components": len({item.homology_component_id for item in examples}),
        "union_components": len({item.union_component_id for item in examples}),
        "sequence_ids_sha256": _set_digest(sequences),
        "example_ids_sha256": _set_digest(item.example_id for item in examples),
        "example_assignment_sha256": _assignment_digest(examples),
        "examples_by_fold": list(config.expected_examples_by_fold),
        "positive_examples_by_fold": list(config.expected_positive_examples_by_fold),
        "negative_examples_by_fold": list(config.expected_negative_examples_by_fold),
    }
    return records, panel


def _historical_design_hint(
    config: UnionEsmContract,
    *,
    sequence_ids: set[str],
) -> dict[str, object]:
    configured_missing = set(config.historical_design_hint_missing_union_sequence_ids)
    _require(
        configured_missing <= sequence_ids,
        "configured design-hint union IDs are absent from accepted panel",
    )
    return {
        "status": config.historical_design_hint_status,
        "source_status": "preliminary_observation_not_consumed_or_verified_by_this_protocol",
        "configured_historical_input_fasta_sha256": (
            config.historical_design_hint_input_fasta_sha256
        ),
        "configured_historical_input_fasta_manifest_sha256": (
            config.historical_design_hint_input_fasta_manifest_sha256
        ),
        "configured_historical_embedding_records": (
            config.historical_design_hint_embedding_records
        ),
        "configured_old_cohort_extra_embedding_records": {
            "sequence_ids": list(config.historical_design_hint_extra_embedding_sequence_ids),
            "sequence_ids_sha256": (
                config.historical_design_hint_extra_embedding_sequence_ids_sha256
            ),
        },
        "configured_union_sequences_reported_missing_from_historical_embeddings": {
            "sequence_ids": list(config.historical_design_hint_missing_union_sequence_ids),
            "sequence_ids_sha256": (
                config.historical_design_hint_missing_union_sequence_ids_sha256
            ),
            "count": len(config.historical_design_hint_missing_union_sequence_ids),
            "all_present_in_exported_union_panel": True,
        },
        "all_union_sequences_exported_regardless_of_design_hint": True,
    }


def _expected_output_payloads(
    *,
    examples: Sequence[Example],
    config: UnionEsmContract,
    input_sha256: Mapping[str, str],
    code_attestation: Mapping[str, object],
    git_commit: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    records, panel = _reconstruct_panel(examples, config=config)
    fasta = b"".join(
        f">sequence_id={sequence_id}\n{sequence}\n".encode("ascii")
        for sequence_id, sequence in records
    )
    design_hint = _historical_design_hint(
        config,
        sequence_ids={sequence_id for sequence_id, _ in records},
    )
    coverage = {
        "schema_version": 1,
        "artifact": _COVERAGE_ARTIFACT,
        "status": "passed",
        "input_sha256": dict(input_sha256),
        "digest_encoding": (
            "sorted unique lowercase SHA-256 identifier strings encoded as ASCII, "
            "joined by LF with a terminal LF; empty set is empty bytes"
        ),
        "assignment_digest_encoding": (
            "example_id, assay_context_id, sequence_id, canonical_target, label, fold, "
            "homology_component_id, union_component_id encoded as tab-separated UTF-8 "
            "records sorted by example_id with a terminal LF"
        ),
        "panel": panel,
        "export": {
            "records": len(records),
            "fasta_sha256": _sha256(fasta),
            "ordering": "ascending sequence_id",
            "header_format": ">sequence_id=<lowercase_sha256>",
            "line_format": "exactly two LF-terminated ASCII lines per record",
            "missing_sequences": 0,
            "missing_sequence_ids": [],
            "missing_sequence_ids_sha256": _set_digest(()),
            "extra_sequences": 0,
            "extra_sequence_ids": [],
            "extra_sequence_ids_sha256": _set_digest(()),
        },
        "unverified_historical_design_hint_not_evidence": design_hint,
        "invariants": {
            "accepted_gate1_publication_chain_valid": True,
            "accepted_gate1_independent_receipt_valid": True,
            "all_oof_models_have_identical_context_metadata": True,
            "base_probabilities_not_consumed": True,
            "context_panel_not_collapsed_before_sequence_export": True,
            "exact_modeled_sequence_set_exported": True,
            "configured_design_hint_ids_not_used_as_exclusions": True,
            "configured_design_hint_missing_union_ids_present_in_export": True,
            "union_components_fold_disjoint": True,
        },
    }
    coverage_payload = _pretty_json_bytes(coverage)
    manifest = {
        "schema_version": 1,
        "artifact": _OUTPUT_ARTIFACT,
        "status": "candidate_pending_independent_verification",
        "production_eligible": False,
        "production_ineligibility_reason": (
            "independent_export_reconstruction_receipt_not_yet_available"
        ),
        "git_commit": git_commit,
        "config_sha256": config.sha256,
        "input_sha256": dict(input_sha256),
        "code_attestation": dict(code_attestation),
        "base_contract": {
            "artifact": _BASE_ARTIFACT,
            "git_commit": config.gate1_git_commit,
            "fold_assignment_policy": _FOLD_POLICY,
            "context_examples": config.expected_examples,
            "source_observations": config.expected_source_observations,
            "modeled_sequences": config.expected_sequences,
        },
        "extraction_contract": {
            "embedding_model": config.embedding_model,
            "representation_layer": config.representation_layer,
            "embedding_dimension": config.embedding_dimension,
            "embedding_batch_size": config.embedding_batch_size,
            "embedding_seed": config.embedding_seed,
            "model_checkpoint_sha256": config.model_checkpoint_sha256,
            "contact_regression_sha256": config.contact_regression_sha256,
            "environment_lock_sha256": config.environment_lock_sha256,
            "trust_manifest_sha256": config.trust_manifest_sha256,
            "embedding_worker_sha256": config.embedding_worker_sha256,
            "runtime": {
                "python": config.python_version,
                "torch": config.torch_version,
                "fair_esm": config.fair_esm_version,
                "numpy": config.numpy_version,
                "cuda_runtime": config.cuda_runtime,
                "cudnn_version": config.cudnn_version,
                "device_type": config.device_type,
            },
        },
        "panel": panel,
        "unverified_historical_design_hint_not_evidence": design_hint,
        "artifacts": {
            "fasta": {
                "filename": "union_esm_sequences.fasta",
                "sha256": _sha256(fasta),
                "records": len(records),
            },
            "coverage_receipt": {
                "filename": "coverage_receipt.json",
                "sha256": _sha256(coverage_payload),
            },
        },
        "invariants": {
            "accepted_union_gate1_only": True,
            "code_and_config_attested": True,
            "deterministic_sequence_id_order": True,
            "exact_configured_sequence_candidate_panel": len(records) == config.expected_sequences,
            "independent_verification_required_before_production": True,
            "path_free_semantic_artifacts": True,
        },
    }
    semantic = {
        "coverage_receipt.json": coverage_payload,
        "manifest.json": _pretty_json_bytes(manifest),
        "union_esm_sequences.fasta": fasta,
    }
    semantic["SHA256SUMS"] = _sha_manifest_bytes(
        {name: _sha256(payload) for name, payload in semantic.items()}
    )
    return semantic, panel


def _require_exact_payload(actual: bytes, expected: bytes, *, label: str) -> None:
    _require(actual == expected, f"{label} differs from independent reconstruction")


def _scan_path_free(payloads: Iterable[bytes], prefixes: Sequence[bytes]) -> None:
    for payload in payloads:
        _require(_ABSOLUTE_PATH_RE.search(payload) is None, "artifact exposes an absolute path")
        _require(
            not any(prefix in payload for prefix in prefixes), "artifact exposes forbidden path"
        )


def _gate1_snapshots(run: Path, independent_receipt: Snapshot) -> dict[str, Snapshot]:
    return {
        "publication_top": _read_snapshot(run / "SHA256SUMS", label="Gate-1 publication top"),
        "gate1_top": _read_snapshot(run / "gate1/SHA256SUMS", label="Gate-1 semantic top"),
        "context_audit": _read_snapshot(
            run / "gate1/context_audit.jsonl", label="Gate-1 context audit"
        ),
        "examples": _read_snapshot(run / "gate1/examples.jsonl", label="Gate-1 examples"),
        "folds": _read_snapshot(run / "gate1/folds.json", label="Gate-1 folds"),
        "manifest": _read_snapshot(run / "gate1/manifest.json", label="Gate-1 manifest"),
        "metrics": _read_snapshot(run / "gate1/metrics.json", label="Gate-1 metrics"),
        "oof": _read_snapshot(run / "gate1/oof_predictions.csv", label="Gate-1 OOF"),
        "split_receipt": _read_snapshot(
            run / "gate1/split_receipt.json", label="Gate-1 split receipt"
        ),
        "code_manifest": _read_snapshot(run / "CODE_SHA256SUMS", label="Gate-1 code manifest"),
        "frozen_manifest": _read_snapshot(
            run / "FROZEN_INPUT_SHA256SUMS", label="Gate-1 frozen manifest"
        ),
        "independent_receipt": independent_receipt,
    }


def _verify_gate1_chain(
    gate1_root: Path,
    *,
    independent_receipt: Snapshot,
    config: UnionEsmContract,
) -> tuple[dict[str, Snapshot], dict[str, object]]:
    immediate = {path.name: path for path in gate1_root.iterdir()}
    _require(set(immediate) == {"0", "1", "node-receipts"}, "Gate-1 job-root inventory changed")
    _require(
        all(path.is_dir() and not path.is_symlink() for path in immediate.values()),
        "Gate-1 job-root entries are not real directories",
    )
    runs = (gate1_root / "0", gate1_root / "1")
    handshake = _verify_handshake(gate1_root, label="Gate-1")
    for index, run in enumerate(runs):
        _verify_manifest_tree(
            run,
            expected_top_sha256=_EXPECTED_GATE1_PUBLICATION_SHA256,
            expected_inventory=_GATE1_PUBLICATION_FILES,
            label=f"Gate-1 publication twin {index}",
        )
        _verify_manifest_tree(
            run / "gate1",
            expected_top_sha256=_EXPECTED_GATE1_TOP_SHA256,
            expected_inventory=_GATE1_FILES,
            label=f"Gate-1 semantic twin {index}",
        )
        _verify_immutable_tree(run, label=f"Gate-1 twin {index}")
    _verify_tree_bytes(
        runs[0],
        runs[1],
        expected_inventory=_GATE1_PUBLICATION_FILES,
        label="Gate-1",
    )
    snapshots = _gate1_snapshots(runs[0], independent_receipt)
    expected_hashes = {
        "publication_top": config.gate1_publication_top_sha256,
        "gate1_top": config.gate1_top_sha256,
        "examples": config.gate1_examples_sha256,
        "folds": config.gate1_folds_sha256,
        "oof": config.gate1_oof_sha256,
        "manifest": config.gate1_manifest_sha256,
        "split_receipt": config.gate1_split_receipt_sha256,
        "independent_receipt": config.gate1_independent_receipt_sha256,
        "code_manifest": config.gate1_code_manifest_sha256,
        "frozen_manifest": config.gate1_frozen_input_manifest_sha256,
    }
    for key, digest in expected_hashes.items():
        _require(snapshots[key].sha256 == digest, f"accepted Gate-1 {key} hash changed")
    _require(
        config.gate1_publication_top_sha256 == _EXPECTED_GATE1_PUBLICATION_SHA256
        and config.gate1_top_sha256 == _EXPECTED_GATE1_TOP_SHA256
        and config.gate1_independent_receipt_sha256 == _EXPECTED_GATE1_INDEPENDENT_RECEIPT_SHA256,
        "config does not select accepted Gate-1 evidence",
    )
    _verify_gate1_documents(snapshots=snapshots, config=config, actual_handshake=handshake)
    examples = _read_examples(snapshots["examples"], config=config)
    identities = _verify_oof_metadata(snapshots["oof"], examples=examples, config=config)
    _verify_folds(snapshots["folds"], examples=examples, identities=identities, config=config)
    return snapshots, handshake


def verify_union_esm_fasta_twins(
    *,
    twin_root: str | Path,
    gate1_twin_root: str | Path,
    gate1_independent_receipt: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    forbidden_prefixes: Sequence[str] = (),
) -> dict[str, object]:
    """Return a path-free exact-panel acceptance receipt for two export twins."""

    twins = _resolved_directory(twin_root, label="union ESM export twin root")
    gate1_twins = _resolved_directory(gate1_twin_root, label="accepted Gate-1 twin root")
    repository = _resolved_directory(repo_root, label="repository root")
    config_requested = Path(config_path)
    _require_no_symlink(config_requested, label="union ESM config", ancestors=True)
    config_file = config_requested.resolve(strict=True)
    expected_config = (repository / _LOGICAL_CONFIG_PATH).resolve(strict=True)
    _require(config_file == expected_config, "verifier received the wrong logical config")
    config_snapshot = _read_snapshot(config_file, label="union ESM config")
    config = _parse_config(config_snapshot)
    independent = _read_snapshot(
        Path(gate1_independent_receipt), label="accepted Gate-1 independent receipt"
    )
    _require(
        independent.path.stat().st_mode & 0o222 == 0,
        "accepted Gate-1 independent receipt remains writable",
    )
    _require(
        independent.sha256 == _EXPECTED_GATE1_INDEPENDENT_RECEIPT_SHA256,
        "Gate-1 independent receipt is not the accepted content address",
    )
    _require(_GIT_RE.fullmatch(expected_git_commit) is not None, "bad expected Git commit")
    _verify_repository(repository, expected_git_commit)
    _verify_execution_source(repository, expected_git_commit)

    gate1_snapshots, gate1_handshake = _verify_gate1_chain(
        gate1_twins,
        independent_receipt=independent,
        config=config,
    )
    immediate = {path.name: path for path in twins.iterdir()}
    _require(set(immediate) == {"0", "1", "node-receipts"}, "export job-root inventory changed")
    _require(
        all(path.is_dir() and not path.is_symlink() for path in immediate.values()),
        "export job-root entries are not real directories",
    )
    runs = (twins / "0", twins / "1")
    export_handshake = _verify_handshake(twins, label="union ESM export")
    top_records: list[tuple[Snapshot, dict[str, str], Snapshot, dict[str, str]]] = []
    for index, run in enumerate(runs):
        top, entries = _verify_manifest_tree(
            run,
            expected_top_sha256=None,
            expected_inventory=_EXPORT_PUBLICATION_FILES,
            label=f"union ESM publication twin {index}",
        )
        panel_top, panel_entries = _verify_manifest_tree(
            run / "panel",
            expected_top_sha256=None,
            expected_inventory=_PANEL_FILES,
            label=f"union ESM panel twin {index}",
        )
        _verify_immutable_tree(run, label=f"union ESM export twin {index}")
        top_records.append((top, entries, panel_top, panel_entries))
    _verify_tree_bytes(
        runs[0],
        runs[1],
        expected_inventory=_EXPORT_PUBLICATION_FILES,
        label="union ESM export",
    )
    _require(
        top_records[0][1] == top_records[1][1]
        and top_records[0][2].sha256 == top_records[1][2].sha256
        and top_records[0][3] == top_records[1][3],
        "export twin manifest records differ",
    )

    code_manifest = _read_snapshot(runs[0] / "CODE_SHA256SUMS", label="export code manifest")
    frozen_manifest = _read_snapshot(
        runs[0] / "FROZEN_INPUT_SHA256SUMS", label="export frozen-input manifest"
    )
    code_attestation = _verify_code_manifest(
        code_manifest,
        repository=repository,
        expected_commit=expected_git_commit,
        config=config,
    )
    _verify_frozen_manifest(
        frozen_manifest,
        gate1_run=gate1_twins / "0",
        gate1_independent_receipt=independent,
    )
    examples = _read_examples(gate1_snapshots["examples"], config=config)
    identities = _verify_oof_metadata(gate1_snapshots["oof"], examples=examples, config=config)
    _verify_folds(
        gate1_snapshots["folds"],
        examples=examples,
        identities=identities,
        config=config,
    )
    input_hashes = {
        "gate1_publication_top_manifest": gate1_snapshots["publication_top"].sha256,
        "gate1_top_manifest": gate1_snapshots["gate1_top"].sha256,
        "gate1_examples": gate1_snapshots["examples"].sha256,
        "gate1_folds": gate1_snapshots["folds"].sha256,
        "gate1_oof": gate1_snapshots["oof"].sha256,
        "gate1_manifest": gate1_snapshots["manifest"].sha256,
        "gate1_split_receipt": gate1_snapshots["split_receipt"].sha256,
        "gate1_independent_receipt": independent.sha256,
        "config": config.sha256,
        "code_manifest": code_manifest.sha256,
    }
    expected_payloads, panel = _expected_output_payloads(
        examples=examples,
        config=config,
        input_sha256=input_hashes,
        code_attestation=code_attestation,
        git_commit=expected_git_commit,
    )
    output_snapshots: dict[str, Snapshot] = {}
    for filename, expected_payload in expected_payloads.items():
        snapshot = _read_snapshot(runs[0] / "panel" / filename, label=f"panel {filename}")
        _require_exact_payload(snapshot.payload, expected_payload, label=filename)
        output_snapshots[filename] = snapshot
    expected_publication_top = _sha_manifest_bytes(
        {
            "CODE_SHA256SUMS": code_manifest.sha256,
            "FROZEN_INPUT_SHA256SUMS": frozen_manifest.sha256,
            "panel/SHA256SUMS": output_snapshots["SHA256SUMS"].sha256,
            "panel/coverage_receipt.json": output_snapshots["coverage_receipt.json"].sha256,
            "panel/manifest.json": output_snapshots["manifest.json"].sha256,
            "panel/union_esm_sequences.fasta": output_snapshots["union_esm_sequences.fasta"].sha256,
        }
    )
    _require_exact_payload(
        top_records[0][0].payload,
        expected_publication_top,
        label="publication SHA256SUMS",
    )
    _require(
        panel["sequence_ids_sha256"] == _EXPECTED_SEQUENCE_IDS_SHA256,
        "reconstructed sequence-ID set is not the accepted 952-sequence panel",
    )

    forbidden = [b"/lustre/scratch/users/"]
    for prefix in forbidden_prefixes:
        _require(isinstance(prefix, str) and prefix != "", "forbidden prefix is empty")
        forbidden.append(prefix.encode("utf-8"))
    publication_payloads = [
        _read_snapshot(runs[0] / name, label=f"path scan {name}").payload
        for name in sorted(_EXPORT_PUBLICATION_FILES)
    ]
    _scan_path_free(publication_payloads, forbidden)

    for snapshot in (
        config_snapshot,
        independent,
        code_manifest,
        frozen_manifest,
        top_records[0][0],
        top_records[0][2],
        *gate1_snapshots.values(),
        *output_snapshots.values(),
    ):
        _assert_unchanged(snapshot, label=snapshot.path.name)
    _verify_repository(repository, expected_git_commit)
    _verify_execution_source(repository, expected_git_commit)
    _verify_tree_bytes(
        runs[0],
        runs[1],
        expected_inventory=_EXPORT_PUBLICATION_FILES,
        label="union ESM export",
    )
    _verify_tree_bytes(
        gate1_twins / "0",
        gate1_twins / "1",
        expected_inventory=_GATE1_PUBLICATION_FILES,
        label="Gate-1",
    )

    extraction_contract = {
        "status": "declared_contract_hash_bound_but_extraction_not_executed_or_verified",
        "embedding_model": config.embedding_model,
        "representation_layer": config.representation_layer,
        "embedding_dimension": config.embedding_dimension,
        "embedding_batch_size": config.embedding_batch_size,
        "embedding_seed": config.embedding_seed,
        "model_checkpoint_sha256": config.model_checkpoint_sha256,
        "contact_regression_sha256": config.contact_regression_sha256,
        "environment_lock_sha256": config.environment_lock_sha256,
        "trust_manifest_sha256": config.trust_manifest_sha256,
        "embedding_worker_sha256": config.embedding_worker_sha256,
    }
    verifier_source = _read_snapshot(
        repository / _VERIFIER_MODULE_PATH,
        label="independent verifier source",
    )
    producer_code_entries = _parse_sha_manifest(
        code_manifest.payload,
        label="export code manifest",
    )
    _require(
        producer_code_entries[_VERIFIER_MODULE_PATH] == verifier_source.sha256,
        "producer code manifest does not bind the independent verifier source",
    )
    _assert_unchanged(verifier_source, label="independent verifier source")
    receipt = {
        "schema_version": 1,
        "artifact": _VERIFICATION_ARTIFACT,
        "status": _ACCEPTANCE_STATUS,
        "acceptance_scope": {
            "embedding_input_eligible": True,
            "accepted_use": "exact_input_to_the_declared_fresh_embedding_extraction_only",
            "embedding_extraction_executed": False,
            "embeddings_verified": False,
            "model_predictions_verified": False,
            "model_performance_evidence": False,
            "downstream_evidence_requirement": (
                "fresh extraction and every downstream model artifact require separate "
                "independent verification"
            ),
        },
        "checks": {name: True for name in sorted(_VERIFICATION_CHECKS)},
        "git_commit": expected_git_commit,
        "config_sha256": config.sha256,
        "publication_top_manifest_sha256": top_records[0][0].sha256,
        "panel_top_manifest_sha256": output_snapshots["SHA256SUMS"].sha256,
        "code_manifest_sha256": code_manifest.sha256,
        "frozen_input_manifest_sha256": frozen_manifest.sha256,
        "verifier_attestation": {
            "logical_path": _VERIFIER_MODULE_PATH,
            "sha256": verifier_source.sha256,
            "git_commit": expected_git_commit,
            "producer_code_manifest_sha256": code_manifest.sha256,
            "included_in_producer_code_manifest": True,
            "executing_source_matches_committed_blob": True,
        },
        "input_sha256": {
            **input_hashes,
            "export_frozen_input_manifest": frozen_manifest.sha256,
        },
        "artifact_sha256": {
            "SHA256SUMS": top_records[0][0].sha256,
            "panel/SHA256SUMS": output_snapshots["SHA256SUMS"].sha256,
            "panel/coverage_receipt.json": output_snapshots["coverage_receipt.json"].sha256,
            "panel/manifest.json": output_snapshots["manifest.json"].sha256,
            "panel/union_esm_sequences.fasta": output_snapshots["union_esm_sequences.fasta"].sha256,
        },
        "panel": panel,
        "declared_extraction_contract_not_execution_evidence": extraction_contract,
        "accepted_gate1": {
            "publication_top_manifest_sha256": gate1_snapshots["publication_top"].sha256,
            "gate1_top_manifest_sha256": gate1_snapshots["gate1_top"].sha256,
            "independent_receipt_sha256": independent.sha256,
            "git_commit": config.gate1_git_commit,
        },
        "production_handshake": export_handshake,
        "accepted_gate1_production_handshake": gate1_handshake,
    }
    payload = _pretty_json_bytes(receipt)
    _scan_path_free([payload], forbidden)
    return receipt


def _write_receipt(
    path: Path,
    receipt: Mapping[str, object],
    *,
    protected_roots: Sequence[Path],
) -> None:
    requested = Path(path)
    _require(requested.name not in {"", ".", ".."}, "verification receipt has no filename")
    _require_no_symlink(requested.parent, label="receipt parent", ancestors=True)
    _require(not os.path.lexists(requested), f"refusing to overwrite receipt: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    _require(not os.path.lexists(target), f"refusing to overwrite receipt: {target}")
    for root in protected_roots:
        _require(
            target != root and not target.is_relative_to(root),
            "receipt must be outside verified trees",
        )
    payload = _pretty_json_bytes(receipt)
    _scan_path_free([payload], [b"/lustre/scratch/users/"])
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o444)
        os.link(staging, target)
        staging.unlink()
    finally:
        if staging.exists():
            staging.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-root", type=Path, required=True)
    parser.add_argument("--gate1-twin-root", type=Path, required=True)
    parser.add_argument("--gate1-independent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--forbidden-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protected = [
        _resolved_directory(args.twin_root, label="union ESM export twin root"),
        _resolved_directory(args.gate1_twin_root, label="accepted Gate-1 twin root"),
        _resolved_directory(args.repo_root, label="repository root"),
    ]
    output = Path(args.output)
    prospective = output.resolve(strict=False)
    for root in protected:
        _require(
            prospective != root and not prospective.is_relative_to(root),
            "receipt must be outside verified trees",
        )
    receipt = verify_union_esm_fasta_twins(
        twin_root=args.twin_root,
        gate1_twin_root=args.gate1_twin_root,
        gate1_independent_receipt=args.gate1_independent_receipt,
        config_path=args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        forbidden_prefixes=args.forbidden_prefix,
    )
    _write_receipt(output, receipt, protected_roots=protected)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the Slurm CLI
    raise SystemExit(main())

"""Compare fixed base/APEX families with a nested homology-fold OOF stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import stat
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from amp_challenge.benchmarks.apex_oof import (
    ApexOofConfig,
)
from amp_challenge.benchmarks.apex_oof import (
    load_config as load_apex_oof_config,
)
from amp_challenge.benchmarks.oracle_gate1 import Gate1Config, binary_metrics
from amp_challenge.models.oracle_baselines import (
    DescriptorLogisticOracle,
    HomologyKnnOracle,
    OracleInput,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

GramClass = Literal["positive", "negative", "unknown"]
BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
APEX_MODEL = "apex_member_mean"
EQUAL_FAMILY_MODEL = "equal_family_blend"
STACK_MODEL = "nested_fold_local_logistic_stack"
DECLARED_METHODS = (*BASE_MODELS, APEX_MODEL, EQUAL_FAMILY_MODEL, STACK_MODEL)
STACK_FEATURES = ("descriptor_logistic", "homology_knn", APEX_MODEL)
PROMOTION_REFERENCE = "equal_weight_ensemble"
PROMOTION_CANDIDATES = (EQUAL_FAMILY_MODEL, STACK_MODEL)
STACK_EVALUATION_STATUS = "nested_outer_inner_cross_fit"
BASE_FOLD_POLICY = "sequence-level single-link homology groups held out together"
BASE_FOLD_ASSIGNMENT_POLICY = "full_sequence_union_bridge_audit_then_reuse_frozen_example_folds"
BASE_PARSER_ID = "amp_challenge.data.dramp:v7"
APEX_FOLD_POLICY = "reuse base Gate-1 homology folds; calibrate on non-heldout folds only"
APEX_TRAINING_INDEPENDENCE = "not established; OOF applies to calibration only"
ENSEMBLE_FOLD_POLICY = (
    "outer homology fold held out; layer-one meta-training features regenerated with "
    "both outer and row-specific inner folds excluded"
)
SIMILARITY_STRATA = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8))
BASE_CONFIG_SHA256 = "b761051d9c46dfc1cf3e58525a89e71659b215fab8004eeaf983fdca0c52af54"
APEX_CONFIG_SHA256 = "8dd0f32d6798cedbd8cecb64bf9a204928154a089871b738b3ccbedc7517ac17"
ENSEMBLE_CONFIG_SHA256 = "c944bd13f7a0cf3db68e04940bf1314b9f62c81240a3ee9967d38344d952b84e"
ACCEPTED_BASE_CHECKSUMS_SHA256 = "159be8b11b5814b1af8814877e6f9d4411b7cc2467d5512e577ac43b55ef54c1"
ACCEPTED_APEX_CHECKSUMS_SHA256 = "869388147ba17f5afd54e94461a957fdbe02f92efe7c1bbb24697d5ce33696ea"
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
ACCEPTED_SUPPORT_EVIDENCE = {
    "examples": 2592,
    "positive_examples": 1792,
    "negative_examples": 800,
    "unique_sequences": 988,
    "homology_clusters": 517,
    "sequence_ids_sha256": "316c6f6cff91f5ad2914202de9cc8e8b748574214641088960383c745cfb2bda",
    "example_ids_sha256": "f916912e8e7bba0c2ce7f2450b870d09b20680cdeb829911bbd25b5444d2fb55",
    "example_label_fold_cluster_sha256": (
        "1ae915a77226695265b8f7eaa6f51d31cae4c913fd44f94afe9847b153b054e3"
    ),
}
ACCEPTED_SUPPORT_BY_FOLD = {
    "0": {"examples": 535, "positive_examples": 357, "negative_examples": 178},
    "1": {"examples": 591, "positive_examples": 395, "negative_examples": 196},
    "2": {"examples": 468, "positive_examples": 346, "negative_examples": 122},
    "3": {"examples": 497, "positive_examples": 357, "negative_examples": 140},
    "4": {"examples": 501, "positive_examples": 337, "negative_examples": 164},
}
ACCEPTED_SUPPORT_BY_TARGET = {
    "acinetobacter_baumannii": {"examples": 13, "positive_examples": 11, "negative_examples": 2},
    "enterococcus_faecalis": {"examples": 99, "positive_examples": 58, "negative_examples": 41},
    "enterococcus_faecium": {"examples": 8, "positive_examples": 6, "negative_examples": 2},
    "escherichia_coli": {"examples": 1011, "positive_examples": 707, "negative_examples": 304},
    "klebsiella_pneumoniae": {"examples": 75, "positive_examples": 50, "negative_examples": 25},
    "pseudomonas_aeruginosa": {"examples": 508, "positive_examples": 356, "negative_examples": 152},
    "staphylococcus_aureus": {"examples": 878, "positive_examples": 604, "negative_examples": 274},
}
ACCEPTED_SUPPORT_BY_SIMILARITY = {
    "[0.00,0.40)": 192,
    "[0.40,0.60)": 1026,
    "[0.60,0.80)": 1374,
}
ACCEPTED_NESTED_FEATURE_ROWS = 10_368
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}")
SET_DIGEST_ENCODING = (
    "sorted unique identifier strings encoded as ASCII, joined by LF with a terminal LF; "
    "empty set is empty bytes"
)
ASSIGNMENT_DIGEST_ENCODING = (
    "example_id, sequence_id, label, fold, cluster_id encoded as tab-separated UTF-8 "
    "records, sorted by example_id, with a terminal LF"
)
NESTED_FEATURE_DIGEST_ENCODING = (
    "outer_fold, inner_fold, example_id, then three logit features encoded as tab-separated "
    "ASCII; floats use Python float.hex(); records sorted by outer_fold, inner_fold, example_id "
    "with a terminal LF"
)
PRODUCTION_BLOCKERS = (
    "apex_upstream_training_independence_not_established",
    "evaluation_labels_previously_inspected_and_resplit",
)
BASE_OOF_SCHEMA = (
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
APEX_OOF_SCHEMA = (
    "model",
    "member_id",
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
    "target_rule",
    "apex_endpoints",
    "activity_signal",
    "probability",
    "member_probability_std",
)


@dataclass(frozen=True, slots=True)
class EnsembleOofConfig:
    path: Path
    methods: tuple[str, ...]
    stack_features: tuple[str, ...]
    stack_input_transform: str
    stack_evaluation_status: str
    probability_clip: float
    stack_l2: float
    stack_prior_strength: float
    stack_max_iterations: int
    stack_tolerance: float
    calibration_bins: int
    bootstrap_replicates: int
    seed: int
    promotion_reference: str
    promotion_candidates: tuple[str, ...]
    promotion_auc_delta_lower_minimum: float
    promotion_brier_delta_upper_maximum: float
    promotion_log_loss_delta_upper_maximum: float


@dataclass(frozen=True, slots=True)
class Metadata:
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

    @property
    def model_input(self) -> OracleInput:
        return OracleInput(sequence=self.sequence, strain=self.strain, gram=self.gram)


@dataclass(frozen=True, slots=True)
class ApexPrediction:
    metadata: Metadata
    target_rule: str
    probability: float
    signals: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class AlignedExample:
    metadata: Metadata
    target_rule: str
    probabilities: Mapping[str, float]
    apex_signals: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class NestedApexCalibrator:
    outer_fold: int
    inner_fold: int
    member_id: str
    training_examples: int
    training_positives: int
    training_negatives: int
    training_example_ids_sha256: str
    training_assignment_sha256: str
    iterations: int
    converged: bool
    signal_mean: float
    signal_scale: float
    intercept: float
    slope: float

    def predict(self, signal: float) -> float:
        logit = self.intercept + self.slope * ((signal - self.signal_mean) / self.signal_scale)
        return float(np.clip(_sigmoid(logit), 1e-6, 1.0 - 1e-6))


@dataclass(frozen=True, slots=True)
class StackCalibrator:
    heldout_fold: int
    training_examples: int
    training_positives: int
    training_negatives: int
    training_example_ids_sha256: str
    training_assignment_sha256: str
    heldout_examples: int
    heldout_positives: int
    heldout_negatives: int
    heldout_example_ids_sha256: str
    heldout_assignment_sha256: str
    iterations: int
    converged: bool
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]

    def predict(self, features: Sequence[float]) -> float:
        values = np.asarray(features, dtype=np.float64)
        means = np.asarray(self.feature_means, dtype=np.float64)
        scales = np.asarray(self.feature_scales, dtype=np.float64)
        logit = self.intercept + float(
            np.dot(np.asarray(self.coefficients, dtype=np.float64), (values - means) / scales)
        )
        return float(np.clip(_sigmoid(logit), 1e-6, 1.0 - 1e-6))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_file(path: str | Path, *, description: str) -> bytes:
    source = Path(path).resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{description} is not a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    before_fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_fingerprint = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_fingerprint != after_fingerprint or len(payload) != before.st_size:
        raise ValueError(f"{description} changed while it was being read: {source}")
    return payload


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(_read_regular_file(path, description=description))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _load_json_array(path: Path, description: str) -> list[object]:
    try:
        value = json.loads(_read_regular_file(path, description=description))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if not isinstance(value, list):
        raise ValueError(f"{description} must contain a JSON array")
    return value


def _read_sha256_manifest(path: Path, *, description: str) -> dict[str, str]:
    try:
        text = _read_regular_file(path, description=description).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} must be UTF-8 text") from error
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise ValueError(f"{description} line {line_number} is not a SHA-256 manifest entry")
        digest, _, logical_path = match.groups()
        if logical_path in entries:
            raise ValueError(f"{description} contains duplicate path {logical_path!r}")
        entries[logical_path] = digest
    if not entries:
        raise ValueError(f"{description} must contain at least one entry")
    return entries


def _require_manifest_entry(
    entries: Mapping[str, str],
    *,
    logical_path: str,
    source: Path,
    manifest_name: str,
) -> None:
    if entries.get(logical_path) != _sha256(source):
        raise ValueError(f"{manifest_name} does not attest {source.name!r} at {logical_path!r}")


def _set_digest(values: Sequence[str] | set[str]) -> str:
    ordered = sorted(set(values))
    payload = b"" if not ordered else ("\n".join(ordered) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _assignment_digest(rows: Sequence[Metadata]) -> str:
    payload = "".join(
        f"{item.example_id}\t{item.sequence_id}\t{item.label}\t{item.fold}\t{item.cluster_id}\n"
        for item in sorted(rows, key=lambda value: value.example_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_evidence(rows: Sequence[Metadata]) -> dict[str, object]:
    ordered = tuple(sorted(rows, key=lambda item: item.example_id))
    if len({item.example_id for item in ordered}) != len(ordered):
        raise ValueError("partition evidence contains duplicate example IDs")
    positives = sum(item.label for item in ordered)
    return {
        "examples": len(ordered),
        "positive_examples": positives,
        "negative_examples": len(ordered) - positives,
        "unique_sequences": len({item.sequence_id for item in ordered}),
        "homology_clusters": len({item.cluster_id for item in ordered}),
        "folds": sorted({item.fold for item in ordered}),
        "example_ids_sha256": _set_digest({item.example_id for item in ordered}),
        "sequence_ids_sha256": _set_digest({item.sequence_id for item in ordered}),
        "example_label_fold_cluster_sha256": _assignment_digest(ordered),
    }


def _nested_feature_digest(
    records: Sequence[tuple[int, int, str, tuple[float, ...]]],
) -> str:
    ordered = sorted(records, key=lambda item: (item[0], item[1], item[2]))
    payload = "".join(
        "\t".join(
            (
                str(outer_fold),
                str(inner_fold),
                example_id,
                *(float(value).hex() for value in features),
            )
        )
        + "\n"
        for outer_fold, inner_fold, example_id, features in ordered
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_two_classes(rows: Sequence[Metadata], *, description: str) -> None:
    labels = {item.label for item in rows}
    if labels != {0, 1}:
        raise ValueError(f"{description} does not contain both label classes")


def _require_disjoint_homology(
    training: Sequence[Metadata],
    testing: Sequence[Metadata],
    *,
    description: str,
) -> None:
    train_examples = {item.example_id for item in training}
    test_examples = {item.example_id for item in testing}
    train_sequences = {item.sequence_id for item in training}
    test_sequences = {item.sequence_id for item in testing}
    train_clusters = {item.cluster_id for item in training}
    test_clusters = {item.cluster_id for item in testing}
    if train_examples & test_examples:
        raise ValueError(f"{description} leaks example IDs across train/test")
    if train_sequences & test_sequences:
        raise ValueError(f"{description} leaks sequences across train/test")
    if train_clusters & test_clusters:
        raise ValueError(f"{description} leaks homology components across train/test")


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"ensemble OOF config {field!r} must be a non-empty string array")
    output = tuple(item.strip() for item in value)
    if any(not item for item in output) or len(output) != len(set(output)):
        raise ValueError(f"ensemble OOF config {field!r} must contain unique non-empty values")
    return output


def load_config(path: str | Path) -> EnsembleOofConfig:
    config_path = Path(path).resolve()
    if _sha256(config_path) != ENSEMBLE_CONFIG_SHA256:
        raise ValueError("ensemble OOF config differs from the frozen scientific configuration")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {
        "schema_version",
        "methods",
        "stack_features",
        "stack_input_transform",
        "stack_evaluation_status",
        "probability_clip",
        "stack_l2",
        "stack_prior_strength",
        "stack_max_iterations",
        "stack_tolerance",
        "calibration_bins",
        "bootstrap_replicates",
        "seed",
        "promotion_reference",
        "promotion_candidates",
        "promotion_auc_delta_lower_minimum",
        "promotion_brier_delta_upper_maximum",
        "promotion_log_loss_delta_upper_maximum",
    }
    if set(raw) != allowed:
        raise ValueError(
            f"ensemble OOF config keys differ from the frozen schema: {sorted(set(raw) ^ allowed)}"
        )
    if raw["schema_version"] != 1:
        raise ValueError("ensemble OOF config schema_version must be 1")
    methods = _string_tuple(raw["methods"], "methods")
    features = _string_tuple(raw["stack_features"], "stack_features")
    if methods != DECLARED_METHODS:
        raise ValueError(f"methods must equal the predeclared comparison {DECLARED_METHODS!r}")
    if features != STACK_FEATURES:
        raise ValueError(f"stack_features must equal the predeclared inputs {STACK_FEATURES!r}")
    transform = str(raw["stack_input_transform"])
    if transform != "logit":
        raise ValueError("stack_input_transform must be the predeclared value 'logit'")
    stack_status = str(raw["stack_evaluation_status"])
    if stack_status != STACK_EVALUATION_STATUS:
        raise ValueError(
            f"stack_evaluation_status must preserve the value {STACK_EVALUATION_STATUS!r}"
        )
    reference = str(raw["promotion_reference"])
    candidates = _string_tuple(raw["promotion_candidates"], "promotion_candidates")
    if reference != PROMOTION_REFERENCE:
        raise ValueError(
            f"promotion_reference must be the predeclared value {PROMOTION_REFERENCE!r}"
        )
    if candidates != PROMOTION_CANDIDATES:
        raise ValueError(
            f"promotion_candidates must equal the predeclared values {PROMOTION_CANDIDATES!r}"
        )

    def positive_float(name: str) -> float:
        value = float(raw[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"ensemble OOF config {name} must be finite and positive")
        return value

    probability_clip = positive_float("probability_clip")
    if probability_clip >= 0.5:
        raise ValueError("probability_clip must be below 0.5")
    prior = float(raw["stack_prior_strength"])
    iterations = int(raw["stack_max_iterations"])
    bins = int(raw["calibration_bins"])
    bootstrap = int(raw["bootstrap_replicates"])
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("stack_prior_strength must be finite and non-negative")
    if iterations < 1 or bins < 2 or bootstrap != 500:
        raise ValueError("invalid stack iteration, calibration-bin, or bootstrap count")
    promotion_thresholds = (
        float(raw["promotion_auc_delta_lower_minimum"]),
        float(raw["promotion_brier_delta_upper_maximum"]),
        float(raw["promotion_log_loss_delta_upper_maximum"]),
    )
    if any(not math.isfinite(value) for value in promotion_thresholds):
        raise ValueError("promotion thresholds must be finite")
    return EnsembleOofConfig(
        path=config_path,
        methods=methods,
        stack_features=features,
        stack_input_transform=transform,
        stack_evaluation_status=stack_status,
        probability_clip=probability_clip,
        stack_l2=positive_float("stack_l2"),
        stack_prior_strength=prior,
        stack_max_iterations=iterations,
        stack_tolerance=positive_float("stack_tolerance"),
        calibration_bins=bins,
        bootstrap_replicates=bootstrap,
        seed=int(raw["seed"]),
        promotion_reference=reference,
        promotion_candidates=candidates,
        promotion_auc_delta_lower_minimum=promotion_thresholds[0],
        promotion_brier_delta_upper_maximum=promotion_thresholds[1],
        promotion_log_loss_delta_upper_maximum=promotion_thresholds[2],
    )


def _verify_base_manifest(
    oof_path: Path,
    manifest_path: Path,
    folds_path: Path,
    split_path: Path,
    checksums_path: Path,
    config_path: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    checksum_entries = _read_sha256_manifest(
        checksums_path,
        description="base Gate-1 top checksum manifest",
    )
    for logical_path, source in (
        ("oof_predictions.csv", oof_path),
        ("manifest.json", manifest_path),
        ("folds.json", folds_path),
        ("split_compatibility.json", split_path),
    ):
        _require_manifest_entry(
            checksum_entries,
            logical_path=logical_path,
            source=source,
            manifest_name="base Gate-1 top checksum manifest",
        )
    document = _load_json(manifest_path, "base manifest")
    if document.get("schema_version") != 1 or document.get("benchmark") != "gate1_strain_activity":
        raise ValueError("base manifest is not a Gate-1 benchmark manifest")
    if document.get("normalized_schema_version") != 2:
        raise ValueError("base manifest is not normalized schema v2")
    if document.get("normalized_parser_id") != BASE_PARSER_ID:
        raise ValueError("base manifest is not bound to the DRAMP parser-v7 contract")
    if document.get("fold_policy") != BASE_FOLD_POLICY:
        raise ValueError("base manifest does not declare the required homology fold policy")
    if document.get("fold_assignment_policy") != BASE_FOLD_ASSIGNMENT_POLICY:
        raise ValueError("base manifest does not declare the accepted fold assignment policy")
    if tuple(document.get("models", ())) != BASE_MODELS:
        raise ValueError("base manifest model list differs from the frozen ensemble inputs")
    if _sha256(config_path) != BASE_CONFIG_SHA256:
        raise ValueError("Gate-1 config differs from the accepted parser-v7 configuration")
    if document.get("config_sha256") != BASE_CONFIG_SHA256:
        raise ValueError("base manifest does not bind the accepted Gate-1 config")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("base manifest has no artifact table")
    required_artifacts = {
        "oof": (oof_path, "oof_predictions.csv"),
        "folds": (folds_path, "folds.json"),
        "split_compatibility": (split_path, "split_compatibility.json"),
    }
    for key, (source, expected_filename) in required_artifacts.items():
        entry = artifacts.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"base manifest has no {key!r} artifact")
        if entry.get("filename") != expected_filename or entry.get("sha256") != _sha256(source):
            raise ValueError(f"base manifest does not match the supplied {key} artifact")
    for key, entry in artifacts.items():
        if (
            not isinstance(key, str)
            or not isinstance(entry, dict)
            or not isinstance(entry.get("filename"), str)
            or not isinstance(entry.get("sha256"), str)
            or SHA256_RE.fullmatch(str(entry["sha256"])) is None
            or checksum_entries.get(str(entry["filename"])) != entry["sha256"]
        ):
            raise ValueError("base manifest artifact table differs from its top checksums")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("base manifest has no provenance object")
    code_manifest = provenance.get("code_manifest")
    normalized_manifest = provenance.get("normalized_data_manifest")
    git_commit = provenance.get("git_commit")
    if (
        not isinstance(code_manifest, dict)
        or code_manifest.get("filename") != "CODE_SHA256SUMS"
        or checksum_entries.get("CODE_SHA256SUMS") != code_manifest.get("sha256")
        or not isinstance(normalized_manifest, dict)
        or normalized_manifest.get("filename") != "SHA256SUMS"
        or SHA256_RE.fullmatch(str(normalized_manifest.get("sha256", ""))) is None
        or not isinstance(git_commit, str)
        or GIT_SHA1_RE.fullmatch(git_commit) is None
    ):
        raise ValueError("base manifest provenance is incomplete or inconsistent")
    return document, checksum_entries


def _verify_apex_manifest(
    predictions_path: Path,
    manifest_path: Path,
    coverage_path: Path,
    calibrators_path: Path,
    checksums_path: Path,
    *,
    base_oof_path: Path,
    base_manifest_path: Path,
    base_folds_path: Path,
    base_config_path: Path,
    apex_config_path: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    checksum_entries = _read_sha256_manifest(
        checksums_path,
        description="APEX-v7 top checksum manifest",
    )
    for logical_path, source in (
        ("apex_oof_predictions.csv", predictions_path),
        ("manifest.json", manifest_path),
        ("sequence_coverage_receipt.json", coverage_path),
        ("calibrators.json", calibrators_path),
    ):
        _require_manifest_entry(
            checksum_entries,
            logical_path=logical_path,
            source=source,
            manifest_name="APEX-v7 top checksum manifest",
        )
    document = _load_json(manifest_path, "APEX OOF manifest")
    if (
        document.get("schema_version") != 1
        or document.get("benchmark") != "apex_homology_oof_calibration"
    ):
        raise ValueError("APEX manifest is not an APEX homology OOF calibration manifest")
    if document.get("fold_policy") != APEX_FOLD_POLICY:
        raise ValueError(
            "APEX manifest does not declare the required fold-local calibration policy"
        )
    if document.get("calibration_policy") != (
        "per-member logistic calibration of log10 activity signal"
    ):
        raise ValueError("APEX manifest has the wrong calibration policy")
    if document.get("upstream_training_independence") != APEX_TRAINING_INDEPENDENCE:
        raise ValueError("APEX manifest must preserve the upstream training-independence caveat")
    if document.get("base_oof_sha256") != _sha256(base_oof_path):
        raise ValueError("APEX manifest was calibrated against a different base OOF CSV")
    if document.get("base_manifest_sha256") != _sha256(base_manifest_path):
        raise ValueError("APEX manifest was calibrated against a different base manifest")
    if document.get("base_folds_sha256") != _sha256(base_folds_path):
        raise ValueError("APEX manifest was calibrated against different base folds")
    if document.get("base_config_sha256") != _sha256(base_config_path):
        raise ValueError("APEX manifest was calibrated with a different Gate-1 config")
    if _sha256(apex_config_path) != APEX_CONFIG_SHA256:
        raise ValueError("APEX config differs from the accepted APEX-v7 configuration")
    if document.get("config_sha256") != APEX_CONFIG_SHA256:
        raise ValueError("APEX manifest does not bind the accepted APEX config")
    outputs = document.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("APEX manifest has no output table")
    if outputs.get("apex_oof_predictions.csv") != _sha256(predictions_path):
        raise ValueError("APEX manifest does not match the supplied OOF predictions")
    if outputs.get("sequence_coverage_receipt.json") != _sha256(coverage_path):
        raise ValueError("APEX manifest does not match the supplied coverage receipt")
    if outputs.get("calibrators.json") != _sha256(calibrators_path):
        raise ValueError("APEX manifest does not match the supplied calibrator evidence")
    if set(outputs) != {
        "apex_oof_predictions.csv",
        "calibrators.json",
        "metrics.json",
        "sequence_coverage_receipt.json",
    }:
        raise ValueError("APEX manifest output inventory differs from schema v1")
    for filename, digest in outputs.items():
        if (
            not isinstance(filename, str)
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or checksum_entries.get(filename) != digest
        ):
            raise ValueError("APEX manifest output table differs from its top checksums")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("APEX manifest has no provenance object")
    code_manifest = provenance.get("code_manifest")
    frozen_manifest = provenance.get("frozen_input_manifest")
    git_commit = provenance.get("git_commit")
    if (
        not isinstance(code_manifest, dict)
        or code_manifest.get("filename") != "CODE_SHA256SUMS"
        or checksum_entries.get("CODE_SHA256SUMS") != code_manifest.get("sha256")
        or not isinstance(frozen_manifest, dict)
        or frozen_manifest.get("filename") != "FROZEN_INPUT_SHA256SUMS"
        or checksum_entries.get("FROZEN_INPUT_SHA256SUMS") != frozen_manifest.get("sha256")
        or not isinstance(git_commit, str)
        or GIT_SHA1_RE.fullmatch(git_commit) is None
    ):
        raise ValueError("APEX manifest provenance is incomplete or inconsistent")
    return document, checksum_entries


def _metadata(row: Mapping[str, str], *, row_number: int, source: str) -> Metadata:
    required = {
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
    if not required <= set(row):
        raise ValueError(f"{source} CSV is missing metadata fields: {sorted(required - set(row))}")
    sequence = canonicalize_sequence(row["sequence"])
    if row["sequence_id"] != canonical_sequence_id(sequence):
        raise ValueError(f"{source} row {row_number} sequence_id mismatch")
    label = int(row["label"])
    fold = int(row["fold"])
    observations = int(row["source_observations"])
    identity = float(row["max_train_identity"])
    if (
        label not in {0, 1}
        or fold < 0
        or observations < 1
        or not math.isfinite(identity)
        or not 0 <= identity < 0.8
    ):
        raise ValueError(f"{source} row {row_number} has invalid metadata")
    example_id = row["example_id"].strip()
    strain = row["strain"].strip()
    gram = row["gram"].strip()
    cluster = row["cluster_id"].strip()
    if not example_id or not strain or not gram or not cluster:
        raise ValueError(f"{source} row {row_number} has empty metadata")
    if gram not in {"positive", "negative", "unknown"}:
        raise ValueError(f"{source} row {row_number} has invalid Gram class")
    return Metadata(
        example_id=example_id,
        sequence_id=row["sequence_id"],
        sequence=sequence,
        strain=strain,
        gram=cast(GramClass, gram),
        label=label,
        source_observations=observations,
        fold=fold,
        cluster_id=cluster,
        max_train_identity=identity,
    )


def _probability(row: Mapping[str, str], *, row_number: int, source: str) -> float:
    value = float(row["probability"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{source} row {row_number} has invalid probability")
    return value


def _read_base_oof(path: Path) -> tuple[dict[str, Metadata], dict[str, dict[str, float]]]:
    metadata: dict[str, Metadata] = {}
    predictions: dict[str, dict[str, float]] = {model: {} for model in BASE_MODELS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(BASE_OOF_SCHEMA):
            raise ValueError("base OOF CSV does not have the frozen Gate-1 schema")
        for row_number, row in enumerate(reader, start=2):
            model = row["model"]
            if model not in predictions:
                raise ValueError(f"base OOF row {row_number} has undeclared model {model!r}")
            item = _metadata(row, row_number=row_number, source="base OOF")
            existing = metadata.setdefault(item.example_id, item)
            if existing != item:
                raise ValueError(f"base OOF metadata mismatch for example {item.example_id}")
            if item.example_id in predictions[model]:
                raise ValueError(f"base OOF duplicates {model}/{item.example_id}")
            predictions[model][item.example_id] = _probability(
                row, row_number=row_number, source="base OOF"
            )
    if not metadata:
        raise ValueError("base OOF CSV is empty")
    expected = set(metadata)
    for model, values in predictions.items():
        if set(values) != expected:
            raise ValueError(f"base OOF model {model!r} does not cover the same examples")
    cluster_folds: dict[str, int] = {}
    sequence_folds: dict[str, int] = {}
    sequence_clusters: dict[str, str] = {}
    for item in metadata.values():
        previous = cluster_folds.setdefault(item.cluster_id, item.fold)
        if previous != item.fold:
            raise ValueError(f"homology cluster {item.cluster_id!r} spans multiple folds")
        previous_sequence_fold = sequence_folds.setdefault(item.sequence_id, item.fold)
        if previous_sequence_fold != item.fold:
            raise ValueError(f"sequence {item.sequence_id!r} spans multiple folds")
        previous_sequence_cluster = sequence_clusters.setdefault(item.sequence_id, item.cluster_id)
        if previous_sequence_cluster != item.cluster_id:
            raise ValueError(f"sequence {item.sequence_id!r} spans multiple homology clusters")
    for example_id in expected:
        expected_mean = (
            predictions["descriptor_logistic"][example_id] + predictions["homology_knn"][example_id]
        ) / 2.0
        observed = predictions["equal_weight_ensemble"][example_id]
        if not math.isclose(observed, expected_mean, rel_tol=0.0, abs_tol=2e-11):
            raise ValueError(f"base equal ensemble arithmetic mismatch for {example_id}")
    return metadata, predictions


def _verify_base_folds(
    path: Path,
    *,
    metadata: Mapping[str, Metadata],
    config: Gate1Config,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    document = _load_json(path, "base Gate-1 folds")
    if set(document) != {"assignments", "folds", "identity_threshold", "method", "seed"}:
        raise ValueError("base Gate-1 folds artifact has an unexpected schema")
    if document.get("method") != BASE_FOLD_ASSIGNMENT_POLICY or document.get(
        "method"
    ) != manifest.get("fold_assignment_policy"):
        raise ValueError("base Gate-1 folds artifact has the wrong assignment policy")
    if (
        document.get("identity_threshold") != 0.8
        or document.get("identity_threshold") != config.homology_identity_threshold
    ):
        raise ValueError("base Gate-1 folds artifact has the wrong identity threshold")
    if document.get("seed") != config.seed:
        raise ValueError("base Gate-1 folds artifact has the wrong seed")
    assignments = document.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(metadata):
        raise ValueError("base Gate-1 fold assignments do not cover every example")
    observed: dict[str, tuple[str, int, str]] = {}
    for index, value in enumerate(assignments):
        if not isinstance(value, dict) or set(value) != {
            "cluster_id",
            "example_id",
            "fold",
            "sequence_id",
        }:
            raise ValueError(f"base Gate-1 fold assignment {index} has an invalid schema")
        example_id = value.get("example_id")
        sequence_id = value.get("sequence_id")
        fold = value.get("fold")
        cluster_id = value.get("cluster_id")
        if (
            not isinstance(example_id, str)
            or not example_id
            or example_id in observed
            or not isinstance(sequence_id, str)
            or SHA256_RE.fullmatch(sequence_id) is None
            or not isinstance(fold, int)
            or isinstance(fold, bool)
            or not isinstance(cluster_id, str)
            or not cluster_id
        ):
            raise ValueError(f"base Gate-1 fold assignment {index} is invalid")
        observed[example_id] = (sequence_id, fold, cluster_id)
    expected = {
        item.example_id: (item.sequence_id, item.fold, item.cluster_id)
        for item in metadata.values()
    }
    if observed != expected:
        raise ValueError("base Gate-1 fold assignments differ from OOF metadata")
    observed_folds = {item.fold for item in metadata.values()}
    if observed_folds != set(range(config.folds)):
        raise ValueError("base Gate-1 OOF does not cover every configured fold")
    summary: dict[str, dict[str, int]] = {}
    for fold in sorted(observed_folds):
        selected = [item for item in metadata.values() if item.fold == fold]
        _require_two_classes(selected, description=f"base Gate-1 fold {fold}")
        positives = sum(item.label for item in selected)
        summary[str(fold)] = {
            "examples": len(selected),
            "homology_clusters": len({item.cluster_id for item in selected}),
            "negatives": len(selected) - positives,
            "positives": positives,
            "sequences": len({item.sequence_id for item in selected}),
        }
    if document.get("folds") != summary:
        raise ValueError("base Gate-1 fold summary differs from OOF metadata")
    return document


def _verify_split_compatibility(
    path: Path,
    *,
    metadata: Mapping[str, Metadata],
    config: Gate1Config,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    document = _load_json(path, "base split compatibility receipt")
    required = {
        "components_merging_source_clusters",
        "cross_fold_bridge_components",
        "fold_source",
        "full_sequence_union",
        "full_union_components",
        "identity_threshold",
        "labeled_sequences",
        "modeled_components",
        "modeled_components_with_unlabeled_sequences",
        "observed_folds",
        "policy",
        "reused_examples",
        "schema_version",
        "source_only_examples",
        "source_only_sequence_ids",
        "status",
        "unlabeled_sequences",
        "unmodeled_components",
    }
    if set(document) != required:
        raise ValueError("base split compatibility receipt has an unexpected schema")
    if (
        document.get("schema_version") != 1
        or document.get("policy") != "full_sequence_union_frozen_fold_compatibility_audit"
        or document.get("status") != "compatible"
        or document.get("identity_threshold") != config.homology_identity_threshold
        or document.get("observed_folds") != list(range(config.folds))
        or document.get("cross_fold_bridge_components") != []
        or document.get("components_merging_source_clusters") != 0
        or document.get("source_only_examples") != 0
        or document.get("source_only_sequence_ids") != []
        or document.get("reused_examples") != len(metadata)
        or document.get("labeled_sequences")
        != len({item.sequence_id for item in metadata.values()})
        or document.get("modeled_components")
        != len({item.cluster_id for item in metadata.values()})
    ):
        raise ValueError("base split compatibility receipt does not prove a safe fold reuse")
    if document.get("fold_source") != manifest.get("fold_source"):
        raise ValueError("base split compatibility fold source differs from its manifest")
    integer_fields = (
        "full_sequence_union",
        "full_union_components",
        "labeled_sequences",
        "modeled_components",
        "modeled_components_with_unlabeled_sequences",
        "reused_examples",
        "source_only_examples",
        "unlabeled_sequences",
        "unmodeled_components",
    )
    if any(
        not isinstance(document.get(field), int)
        or isinstance(document.get(field), bool)
        or int(document[field]) < 0
        for field in integer_fields
    ):
        raise ValueError("base split compatibility receipt has invalid census values")
    if (
        document["full_sequence_union"]
        != document["labeled_sequences"] + document["unlabeled_sequences"]
        or document["full_union_components"]
        != document["modeled_components"] + document["unmodeled_components"]
        or document["modeled_components_with_unlabeled_sequences"] > document["modeled_components"]
    ):
        raise ValueError("base split compatibility census is internally inconsistent")
    fold_source = document.get("fold_source")
    if not isinstance(fold_source, dict):
        raise ValueError("base split compatibility receipt has no fold source")
    for field in (
        "folds_sha256",
        "manifest_sha256",
        "oof_sha256",
        "source_input_assays_sha256",
        "source_normalized_summary_sha256",
    ):
        if SHA256_RE.fullmatch(str(fold_source.get(field, ""))) is None:
            raise ValueError("base split compatibility fold source has an invalid digest")
    if (
        fold_source.get("current_examples") != len(metadata)
        or fold_source.get("shared_labels_unchanged") != len(metadata)
        or fold_source.get("source_only_examples") != 0
    ):
        raise ValueError("base split compatibility label bridge is incomplete")
    return document


def _apex_members(document: Mapping[str, object], config: ApexOofConfig) -> tuple[str, ...]:
    raw = document.get("members")
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise ValueError("APEX OOF manifest has an invalid member list")
    members = tuple(raw)
    checkpoints = document.get("member_checkpoints")
    if (
        len(members) != len(set(members))
        or len(members) != config.expected_member_count
        or config.expected_member_count != 8
        or not isinstance(checkpoints, list)
        or len(checkpoints) != len(members)
    ):
        raise ValueError("APEX OOF member list differs from its frozen config")
    checkpoint_members: list[str] = []
    for item in checkpoints:
        if (
            not isinstance(item, dict)
            or set(item) != {"checkpoint_sha256", "member_id"}
            or not isinstance(item.get("member_id"), str)
            or SHA256_RE.fullmatch(str(item.get("checkpoint_sha256", ""))) is None
        ):
            raise ValueError("APEX OOF manifest has an invalid member checkpoint entry")
        checkpoint_members.append(str(item["member_id"]))
    if tuple(checkpoint_members) != members:
        raise ValueError("APEX OOF checkpoint order differs from its member list")
    return members


def _read_apex_oof(
    path: Path,
    members: Sequence[str],
    config: ApexOofConfig,
) -> dict[str, ApexPrediction]:
    mean_rows: dict[str, tuple[Metadata, str, float, float, float]] = {}
    member_metadata: dict[tuple[str, str], tuple[Metadata, str]] = {}
    signals: dict[str, dict[str, float]] = defaultdict(dict)
    member_probabilities: dict[str, dict[str, float]] = defaultdict(dict)
    expected_members = set(members)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(APEX_OOF_SCHEMA):
            raise ValueError("APEX OOF CSV does not have the frozen schema")
        for row_number, row in enumerate(reader, start=2):
            item = _metadata(row, row_number=row_number, source="APEX OOF")
            target = row["target_rule"].strip()
            endpoints = row["apex_endpoints"].strip()
            if not target or not endpoints:
                raise ValueError(f"APEX OOF row {row_number} has empty target metadata")
            matches = [rule for rule in config.targets if rule.matches(item.strain)]
            if len(matches) != 1:
                raise ValueError(
                    f"APEX OOF row {row_number} does not map to exactly one frozen target"
                )
            expected_target = matches[0]
            if target != expected_target.name or endpoints != ";".join(expected_target.endpoints):
                raise ValueError(f"APEX OOF row {row_number} target/endpoints mismatch")
            probability = _probability(row, row_number=row_number, source="APEX OOF")
            signal = float(row["activity_signal"])
            if not math.isfinite(signal):
                raise ValueError(f"APEX OOF row {row_number} has a non-finite activity signal")
            model = row["model"]
            if model == APEX_MODEL:
                member_std = float(row["member_probability_std"])
                if (
                    row["member_id"]
                    or item.example_id in mean_rows
                    or not math.isfinite(member_std)
                    or member_std < 0
                ):
                    raise ValueError(
                        f"APEX OOF duplicates or mislabels {APEX_MODEL}/{item.example_id}"
                    )
                mean_rows[item.example_id] = (
                    item,
                    f"{target}\x1f{endpoints}",
                    signal,
                    probability,
                    member_std,
                )
                continue
            member = row["member_id"]
            if member not in expected_members or model != f"apex_member::{member}":
                raise ValueError(f"APEX OOF row {row_number} has an unexpected model/member")
            if row["member_probability_std"]:
                raise ValueError(f"APEX OOF row {row_number} has member-level ensemble spread")
            key = (item.example_id, member)
            if key in member_metadata:
                raise ValueError(f"APEX OOF duplicates member {member}/{item.example_id}")
            member_metadata[key] = (item, f"{target}\x1f{endpoints}")
            signals[item.example_id][member] = signal
            member_probabilities[item.example_id][member] = probability
    if not mean_rows:
        raise ValueError(f"APEX OOF CSV contains no {APEX_MODEL!r} rows")
    predictions: dict[str, ApexPrediction] = {}
    for example_id, (
        metadata,
        target_and_endpoints,
        mean_signal,
        probability,
        member_std,
    ) in mean_rows.items():
        if set(signals[example_id]) != expected_members:
            raise ValueError(f"APEX OOF example {example_id} does not have every raw member signal")
        for member in members:
            if member_metadata[(example_id, member)] != (metadata, target_and_endpoints):
                raise ValueError(f"APEX OOF member metadata mismatch for {member}/{example_id}")
        ordered_probabilities = [member_probabilities[example_id][member] for member in members]
        ordered_signals = [signals[example_id][member] for member in members]
        expected_mean = float(np.mean(ordered_probabilities))
        expected_signal = float(np.mean(ordered_signals))
        expected_std = float(np.std(ordered_probabilities))
        if not math.isclose(probability, expected_mean, rel_tol=0.0, abs_tol=2e-11):
            raise ValueError(f"APEX member-mean arithmetic mismatch for {example_id}")
        if not math.isclose(mean_signal, expected_signal, rel_tol=0.0, abs_tol=2e-11):
            raise ValueError(f"APEX member-signal arithmetic mismatch for {example_id}")
        if not math.isclose(member_std, expected_std, rel_tol=0.0, abs_tol=2e-11):
            raise ValueError(f"APEX member-spread arithmetic mismatch for {example_id}")
        target, _ = target_and_endpoints.split("\x1f", maxsplit=1)
        predictions[example_id] = ApexPrediction(
            metadata=metadata,
            target_rule=target,
            probability=probability,
            signals=signals[example_id],
        )
    if set(signals) != set(mean_rows):
        raise ValueError("APEX OOF member examples differ from member-mean examples")
    return predictions


def _verify_apex_coverage(
    path: Path,
    *,
    metadata: Mapping[str, Metadata],
    apex: Mapping[str, ApexPrediction],
    members: Sequence[str],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    document = _load_json(path, "APEX-v7 sequence coverage receipt")
    if set(document) != {
        "apex",
        "assignment_digest_encoding",
        "base",
        "coverage",
        "excluded_examples",
        "exclusion_reasons",
        "schema_version",
        "set_digest_encoding",
        "supported",
    }:
        raise ValueError("APEX-v7 coverage receipt has an unexpected schema")
    if (
        document.get("schema_version") != 1
        or document.get("set_digest_encoding") != SET_DIGEST_ENCODING
        or document.get("assignment_digest_encoding") != ASSIGNMENT_DIGEST_ENCODING
    ):
        raise ValueError("APEX-v7 coverage receipt has the wrong digest contract")
    base_evidence = _partition_evidence(tuple(metadata.values()))
    apex_evidence = _partition_evidence(tuple(item.metadata for item in apex.values()))
    base = document.get("base")
    supported = document.get("supported")
    if not isinstance(base, dict) or not isinstance(supported, dict):
        raise ValueError("APEX-v7 coverage receipt is missing base/supported evidence")
    expected_base = {
        "examples": base_evidence["examples"],
        "positive_examples": base_evidence["positive_examples"],
        "negative_examples": base_evidence["negative_examples"],
        "unique_sequences": base_evidence["unique_sequences"],
        "homology_clusters": base_evidence["homology_clusters"],
        "sequence_ids_sha256": base_evidence["sequence_ids_sha256"],
        "example_label_fold_cluster_sha256": base_evidence["example_label_fold_cluster_sha256"],
    }
    expected_supported = {
        "examples": apex_evidence["examples"],
        "positive_examples": apex_evidence["positive_examples"],
        "negative_examples": apex_evidence["negative_examples"],
        "unique_sequences": apex_evidence["unique_sequences"],
        "homology_clusters": apex_evidence["homology_clusters"],
        "sequence_ids_sha256": apex_evidence["sequence_ids_sha256"],
        "example_ids_sha256": apex_evidence["example_ids_sha256"],
        "example_label_fold_cluster_sha256": apex_evidence["example_label_fold_cluster_sha256"],
    }
    if base != expected_base:
        raise ValueError("APEX-v7 coverage base evidence differs from Gate-1")
    if supported != expected_supported:
        raise ValueError("APEX-v7 coverage supported evidence differs from APEX OOF rows")
    if (
        document.get("excluded_examples") != len(metadata) - len(apex)
        or document.get("excluded_examples") != manifest.get("excluded_examples")
        or document.get("exclusion_reasons") != manifest.get("exclusion_reasons")
    ):
        raise ValueError("APEX-v7 exclusion evidence differs from its manifest")
    coverage = document.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("APEX-v7 coverage receipt has no coverage object")
    base_sequence_ids = {item.sequence_id for item in metadata.values()}
    missing_ids = coverage.get("missing_sequence_ids")
    extra_ids = coverage.get("extra_sequence_ids")
    if (
        coverage.get("all_base_sequences_covered") is not True
        or coverage.get("covered_base_sequences") != len(base_sequence_ids)
        or coverage.get("covered_sequence_ids_sha256") != _set_digest(base_sequence_ids)
        or coverage.get("missing_sequences") != 0
        or missing_ids != []
        or coverage.get("missing_sequence_ids_sha256") != _set_digest(set())
        or not isinstance(extra_ids, list)
        or not all(isinstance(value, str) and SHA256_RE.fullmatch(value) for value in extra_ids)
        or len(extra_ids) != len(set(extra_ids))
        or coverage.get("extra_sequences") != len(extra_ids)
        or coverage.get("extra_sequence_ids_sha256") != _set_digest(set(extra_ids))
    ):
        raise ValueError("APEX-v7 sequence coverage evidence is incomplete")
    raw_apex = document.get("apex")
    if not isinstance(raw_apex, dict):
        raise ValueError("APEX-v7 coverage receipt has no raw APEX census")
    raw_sequences = raw_apex.get("unique_sequences")
    extra_sequence_ids = set(cast(list[str], extra_ids))
    if (
        raw_apex.get("members") != len(members)
        or not isinstance(raw_sequences, int)
        or isinstance(raw_sequences, bool)
        or raw_sequences < len(base_sequence_ids)
        or raw_apex.get("rows") != raw_sequences * len(members)
        or base_sequence_ids & extra_sequence_ids
        or raw_sequences != len(base_sequence_ids | extra_sequence_ids)
        or raw_apex.get("sequence_ids_sha256")
        != _set_digest(base_sequence_ids | extra_sequence_ids)
    ):
        raise ValueError("APEX-v7 raw prediction census is invalid")
    exclusion_reasons = document.get("exclusion_reasons")
    if (
        not isinstance(exclusion_reasons, dict)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for key, value in exclusion_reasons.items()
        )
        or sum(cast(dict[str, int], exclusion_reasons).values()) != document["excluded_examples"]
    ):
        raise ValueError("APEX-v7 exclusion census is invalid")
    return document


def _verify_apex_calibrators(
    path: Path,
    *,
    aligned: Sequence[AlignedExample],
    members: Sequence[str],
    folds: Sequence[int],
    config: ApexOofConfig,
) -> list[dict[str, object]]:
    raw = _load_json_array(path, "APEX-v7 calibrators")
    expected_keys = {
        "converged",
        "fold",
        "intercept",
        "iterations",
        "member_id",
        "positives",
        "signal_mean",
        "signal_scale",
        "slope",
        "training_examples",
    }
    expected_pairs = {(fold, member) for fold in folds for member in members}
    observed: dict[tuple[int, str], dict[str, object]] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(f"APEX-v7 calibrator {index} has an unexpected schema")
        fold = value.get("fold")
        member = value.get("member_id")
        iterations = value.get("iterations")
        if (
            not isinstance(fold, int)
            or isinstance(fold, bool)
            or not isinstance(member, str)
            or (fold, member) in observed
            or not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations < 1
            or iterations > config.calibration_max_iterations
            or value.get("converged") is not True
        ):
            raise ValueError(f"APEX-v7 calibrator {index} has invalid identity/convergence")
        numeric = (
            value.get("signal_mean"),
            value.get("signal_scale"),
            value.get("intercept"),
            value.get("slope"),
        )
        if (
            any(
                not isinstance(item, int | float)
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                for item in numeric
            )
            or float(value["signal_scale"]) <= 0
        ):
            raise ValueError(f"APEX-v7 calibrator {index} has non-finite parameters")
        training = [item.metadata for item in aligned if item.metadata.fold != fold]
        _require_two_classes(training, description=f"APEX-v7 fold {fold} calibration training")
        positives = sum(item.label for item in training)
        if value.get("training_examples") != len(training) or value.get("positives") != positives:
            raise ValueError(f"APEX-v7 calibrator {index} has the wrong class census")
        observed[(fold, member)] = value
    if set(observed) != expected_pairs:
        raise ValueError("APEX-v7 calibrators do not cover every fold/member pair")
    return [observed[key] for key in sorted(observed)]


def _align(
    metadata: Mapping[str, Metadata],
    base: Mapping[str, Mapping[str, float]],
    apex: Mapping[str, ApexPrediction],
) -> tuple[AlignedExample, ...]:
    unknown = set(apex) - set(metadata)
    if unknown:
        raise ValueError(f"APEX OOF contains examples absent from base OOF: {sorted(unknown)[:3]}")
    output: list[AlignedExample] = []
    for example_id in sorted(apex):
        item = apex[example_id]
        if item.metadata != metadata[example_id]:
            raise ValueError(f"APEX/base metadata mismatch for example {example_id}")
        probabilities = {model: float(base[model][example_id]) for model in BASE_MODELS}
        probabilities[APEX_MODEL] = item.probability
        probabilities[EQUAL_FAMILY_MODEL] = (
            probabilities["equal_weight_ensemble"] + probabilities[APEX_MODEL]
        ) / 2.0
        output.append(
            AlignedExample(
                metadata=item.metadata,
                target_rule=item.target_rule,
                probabilities=probabilities,
                apex_signals=item.signals,
            )
        )
    if len({item.metadata.fold for item in output}) < 2:
        raise ValueError("supported APEX subset must contain at least two folds")
    return tuple(output)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _transform_probabilities(
    probabilities: Mapping[str, float], config: EnsembleOofConfig
) -> tuple[float, ...]:
    values: list[float] = []
    for name in config.stack_features:
        probability = float(
            np.clip(probabilities[name], config.probability_clip, 1 - config.probability_clip)
        )
        values.append(math.log(probability / (1.0 - probability)))
    return tuple(values)


def _stack_inputs(item: AlignedExample, config: EnsembleOofConfig) -> tuple[float, ...]:
    return _transform_probabilities(item.probabilities, config)


def _audit_descriptor_fit(
    model: DescriptorLogisticOracle,
    training_inputs: Sequence[OracleInput],
    labels: np.ndarray,
    *,
    description: str,
) -> dict[str, object]:
    fitted_coefficient = model._coefficient
    if fitted_coefficient is None or not np.all(np.isfinite(fitted_coefficient)):
        raise ValueError(f"{description} has missing or non-finite descriptor coefficients")
    design = model._design_matrix(tuple(training_inputs))
    if not np.all(np.isfinite(design)):
        raise ValueError(f"{description} has non-finite descriptor features")
    prior = float(
        (np.sum(labels) + 0.5 * model.prior_strength) / (len(labels) + model.prior_strength)
    )
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(coefficient.size, model.l2, dtype=np.float64)
    penalty[0] = 0.0
    converged = False
    iterations = 0
    for iteration in range(1, model.max_iterations + 1):
        logits = design @ coefficient
        probability = np.empty_like(logits)
        positive = logits >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        probability[~positive] = exponent / (1.0 + exponent)
        variance = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - labels) / len(labels) + penalty * coefficient
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError(f"{description} has a non-finite descriptor convergence step")
        coefficient -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficient)):
            raise ValueError(f"{description} has non-finite descriptor coefficients")
        if float(np.max(np.abs(step))) <= model.tolerance:
            converged = True
            break
    if not converged:
        raise ValueError(
            f"{description} descriptor logistic exhausted its iteration budget before convergence"
        )
    if not np.array_equal(coefficient, fitted_coefficient):
        raise ValueError(f"{description} descriptor audit differs from the fitted model")
    return {
        "converged": True,
        "iterations": iterations,
        "coefficient_count": int(coefficient.size),
        "parameters_finite": True,
    }


def _fit_nested_apex_calibrator(
    rows: Sequence[AlignedExample],
    *,
    outer_fold: int,
    inner_fold: int,
    member_id: str,
    config: ApexOofConfig,
) -> NestedApexCalibrator:
    training = [item for item in rows if item.metadata.fold not in {outer_fold, inner_fold}]
    if not training:
        raise ValueError(
            f"outer fold {outer_fold}, inner fold {inner_fold} leaves no APEX calibration rows"
        )
    training_metadata = [item.metadata for item in training]
    _require_two_classes(
        training_metadata,
        description=f"outer fold {outer_fold}, inner fold {inner_fold} APEX calibration",
    )
    signal = np.asarray([item.apex_signals[member_id] for item in training], dtype=np.float64)
    labels = np.asarray([item.metadata.label for item in training], dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"outer fold {outer_fold}, inner fold {inner_fold}, member {member_id!r} "
            "has non-finite calibration signals"
        )
    signal_mean = float(np.mean(signal))
    signal_scale = float(np.std(signal))
    if signal_scale < 1e-12:
        signal_scale = 1.0
    standardized = (signal - signal_mean) / signal_scale
    prior = float(
        (np.sum(labels) + 0.5 * config.calibration_prior_strength)
        / (len(labels) + config.calibration_prior_strength)
    )
    if not 0 < prior < 1:
        raise ValueError("nested APEX calibration has a degenerate prior")
    coefficient = np.asarray([math.log(prior / (1.0 - prior)), 0.0], dtype=np.float64)
    design = np.column_stack((np.ones(len(labels)), standardized))
    penalty = np.asarray([0.0, config.calibration_l2], dtype=np.float64)
    iterations = 0
    converged = False
    for iteration in range(1, config.calibration_max_iterations + 1):
        logits = design @ coefficient
        probabilities = np.asarray([_sigmoid(float(value)) for value in logits])
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (probabilities - labels) / len(labels) + penalty * coefficient
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError("nested APEX calibration produced a non-finite Newton step")
        coefficient -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficient)):
            raise ValueError("nested APEX calibration produced non-finite coefficients")
        if float(np.max(np.abs(step))) <= config.calibration_tolerance:
            converged = True
            break
    if not converged:
        raise ValueError(
            f"outer fold {outer_fold}, inner fold {inner_fold}, member {member_id!r} "
            "APEX calibration did not converge"
        )
    evidence = _partition_evidence(training_metadata)
    return NestedApexCalibrator(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        member_id=member_id,
        training_examples=len(training),
        training_positives=int(np.sum(labels)),
        training_negatives=len(training) - int(np.sum(labels)),
        training_example_ids_sha256=str(evidence["example_ids_sha256"]),
        training_assignment_sha256=str(evidence["example_label_fold_cluster_sha256"]),
        iterations=iterations,
        converged=converged,
        signal_mean=signal_mean,
        signal_scale=signal_scale,
        intercept=float(coefficient[0]),
        slope=float(coefficient[1]),
    )


def _nested_layer_one_features(
    all_metadata: Mapping[str, Metadata],
    aligned: Sequence[AlignedExample],
    *,
    outer_fold: int,
    inner_fold: int,
    base_config: Gate1Config,
    apex_config: ApexOofConfig,
    members: Sequence[str],
    ensemble_config: EnsembleOofConfig,
) -> tuple[dict[str, tuple[float, ...]], dict[str, object]]:
    excluded = {outer_fold, inner_fold}
    base_training = sorted(
        (item for item in all_metadata.values() if item.fold not in excluded),
        key=lambda item: item.example_id,
    )
    testing = sorted(
        (item for item in aligned if item.metadata.fold == inner_fold),
        key=lambda item: item.metadata.example_id,
    )
    outer_testing = sorted(
        (item for item in aligned if item.metadata.fold == outer_fold),
        key=lambda item: item.metadata.example_id,
    )
    apex_training = sorted(
        (item for item in aligned if item.metadata.fold not in excluded),
        key=lambda item: item.metadata.example_id,
    )
    if not base_training or not testing:
        raise ValueError(
            f"outer fold {outer_fold}, inner fold {inner_fold} has an empty nested train/test split"
        )
    base_training_metadata = list(base_training)
    apex_training_metadata = [item.metadata for item in apex_training]
    inner_test_metadata = [item.metadata for item in testing]
    excluded_test_metadata = [
        *(item.metadata for item in testing),
        *(item.metadata for item in outer_testing),
    ]
    _require_two_classes(
        base_training_metadata,
        description=f"outer {outer_fold}/inner {inner_fold} base training",
    )
    _require_two_classes(
        apex_training_metadata,
        description=f"outer {outer_fold}/inner {inner_fold} APEX training",
    )
    _require_two_classes(
        inner_test_metadata,
        description=f"outer {outer_fold}/inner {inner_fold} inner test",
    )
    _require_disjoint_homology(
        base_training_metadata,
        excluded_test_metadata,
        description=f"outer {outer_fold}/inner {inner_fold} base split",
    )
    _require_disjoint_homology(
        apex_training_metadata,
        excluded_test_metadata,
        description=f"outer {outer_fold}/inner {inner_fold} APEX split",
    )
    training_inputs = tuple(item.model_input for item in base_training)
    testing_inputs = tuple(item.metadata.model_input for item in testing)
    labels = np.asarray([item.label for item in base_training], dtype=np.int64)
    base_models = (
        DescriptorLogisticOracle(
            l2=base_config.logistic.l2,
            max_iterations=base_config.logistic.max_iterations,
            tolerance=base_config.logistic.tolerance,
            prior_strength=base_config.logistic.prior_strength,
        ),
        HomologyKnnOracle(
            neighbors=base_config.knn.neighbors,
            similarity_power=base_config.knn.similarity_power,
            prior_strength=base_config.knn.prior_strength,
            minimum_weight=base_config.knn.minimum_weight,
        ),
    )
    base_probabilities: dict[str, np.ndarray] = {}
    descriptor_fit_evidence: dict[str, object] | None = None
    for model in base_models:
        model.fit(training_inputs, labels)
        probability = model.predict_proba(testing_inputs)
        if probability.shape != (len(testing),) or not np.all(np.isfinite(probability)):
            raise ValueError(
                f"outer {outer_fold}/inner {inner_fold} {model.name} predictions are invalid"
            )
        base_probabilities[model.name] = probability
        if isinstance(model, DescriptorLogisticOracle):
            descriptor_fit_evidence = _audit_descriptor_fit(
                model,
                training_inputs,
                labels.astype(np.float64),
                description=f"outer {outer_fold}/inner {inner_fold}",
            )
    if descriptor_fit_evidence is None:
        raise AssertionError("descriptor convergence evidence was not generated")

    apex_calibrators = [
        _fit_nested_apex_calibrator(
            aligned,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            member_id=member,
            config=apex_config,
        )
        for member in members
    ]
    features: dict[str, tuple[float, ...]] = {}
    for index, item in enumerate(testing):
        probabilities = {
            "descriptor_logistic": float(base_probabilities["descriptor_logistic"][index]),
            "homology_knn": float(base_probabilities["homology_knn"][index]),
            APEX_MODEL: float(
                np.mean(
                    [
                        calibrator.predict(item.apex_signals[calibrator.member_id])
                        for calibrator in apex_calibrators
                    ]
                )
            ),
        }
        features[item.metadata.example_id] = _transform_probabilities(
            probabilities, ensemble_config
        )
    evidence: dict[str, object] = {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "excluded_folds": sorted(excluded),
        "base_training_examples": len(base_training),
        "base_training_positives": int(np.sum(labels)),
        "base_training_negatives": len(base_training) - int(np.sum(labels)),
        "apex_calibration_examples": len(apex_training),
        "apex_calibration_positives": sum(item.metadata.label for item in apex_training),
        "apex_calibration_negatives": len(apex_training)
        - sum(item.metadata.label for item in apex_training),
        "inner_test_examples": len(testing),
        "inner_test_positives": sum(item.metadata.label for item in testing),
        "inner_test_negatives": len(testing) - sum(item.metadata.label for item in testing),
        "base_training": _partition_evidence(base_training_metadata),
        "apex_calibration_training": _partition_evidence(apex_training_metadata),
        "inner_test": _partition_evidence(inner_test_metadata),
        "inner_feature_rows_sha256": _nested_feature_digest(
            [(outer_fold, inner_fold, example_id, value) for example_id, value in features.items()]
        ),
        "base_descriptor_fit": descriptor_fit_evidence,
        "base_homology_knn_fit": {"finite_predictions": True},
        "apex_member_calibrators": [asdict(item) for item in apex_calibrators],
    }
    return features, evidence


def _fit_stack(
    rows: Sequence[AlignedExample],
    *,
    heldout_fold: int,
    nested_features: Mapping[str, Sequence[float]],
    config: EnsembleOofConfig,
) -> StackCalibrator:
    training = [item for item in rows if item.metadata.fold != heldout_fold]
    heldout = [item for item in rows if item.metadata.fold == heldout_fold]
    if not training:
        raise ValueError(f"fold {heldout_fold} leaves no stack-training examples")
    if not heldout:
        raise ValueError(f"fold {heldout_fold} has no held-out stack examples")
    training_metadata = [item.metadata for item in training]
    heldout_metadata = [item.metadata for item in heldout]
    _require_two_classes(training_metadata, description=f"fold {heldout_fold} stack training")
    _require_two_classes(heldout_metadata, description=f"fold {heldout_fold} stack test")
    _require_disjoint_homology(
        training_metadata,
        heldout_metadata,
        description=f"fold {heldout_fold} stack split",
    )
    expected = {item.metadata.example_id for item in training}
    if set(nested_features) != expected:
        raise ValueError(
            f"nested layer-one features do not cover outer fold {heldout_fold} training"
        )
    features = np.asarray(
        [nested_features[item.metadata.example_id] for item in training], dtype=np.float64
    )
    if features.shape != (len(training), len(config.stack_features)) or not np.all(
        np.isfinite(features)
    ):
        raise ValueError(f"fold {heldout_fold} has invalid nested stack features")
    labels = np.asarray([item.metadata.label for item in training], dtype=np.float64)
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    prior = float(
        (np.sum(labels) + 0.5 * config.stack_prior_strength)
        / (len(labels) + config.stack_prior_strength)
    )
    if not 0 < prior < 1:
        raise ValueError(f"fold {heldout_fold} stack training has a degenerate prior")
    coefficients = np.zeros(features.shape[1] + 1, dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    design = np.column_stack((np.ones(len(labels)), standardized))
    penalty = np.asarray([0.0, *([config.stack_l2] * features.shape[1])])
    iterations = 0
    converged = False
    for iteration in range(1, config.stack_max_iterations + 1):
        logits = design @ coefficients
        probabilities = np.asarray([_sigmoid(float(value)) for value in logits])
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (probabilities - labels) / len(labels) + penalty * coefficients
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError(f"fold {heldout_fold} stack produced a non-finite Newton step")
        coefficients -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(f"fold {heldout_fold} stack produced non-finite coefficients")
        if float(np.max(np.abs(step))) <= config.stack_tolerance:
            converged = True
            break
    if not converged:
        raise ValueError(f"fold {heldout_fold} stack did not converge")
    training_evidence = _partition_evidence(training_metadata)
    heldout_evidence = _partition_evidence(heldout_metadata)
    return StackCalibrator(
        heldout_fold=heldout_fold,
        training_examples=len(training),
        training_positives=int(np.sum(labels)),
        training_negatives=len(training) - int(np.sum(labels)),
        training_example_ids_sha256=str(training_evidence["example_ids_sha256"]),
        training_assignment_sha256=str(training_evidence["example_label_fold_cluster_sha256"]),
        heldout_examples=len(heldout),
        heldout_positives=sum(item.metadata.label for item in heldout),
        heldout_negatives=len(heldout) - sum(item.metadata.label for item in heldout),
        heldout_example_ids_sha256=str(heldout_evidence["example_ids_sha256"]),
        heldout_assignment_sha256=str(heldout_evidence["example_label_fold_cluster_sha256"]),
        iterations=iterations,
        converged=converged,
        feature_names=config.stack_features,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        intercept=float(coefficients[0]),
        coefficients=tuple(float(value) for value in coefficients[1:]),
    )


def _make_prediction_rows(
    aligned: Sequence[AlignedExample],
    *,
    all_metadata: Mapping[str, Metadata],
    base_config: Gate1Config,
    apex_config: ApexOofConfig,
    members: Sequence[str],
    config: EnsembleOofConfig,
) -> tuple[
    list[dict[str, object]],
    list[StackCalibrator],
    list[dict[str, object]],
    dict[str, object],
]:
    folds = sorted({item.metadata.fold for item in aligned})
    calibrators: list[StackCalibrator] = []
    nested_evidence: list[dict[str, object]] = []
    nested_feature_records: list[tuple[int, int, str, tuple[float, ...]]] = []
    for outer_fold in folds:
        nested_features: dict[str, tuple[float, ...]] = {}
        for inner_fold in folds:
            if inner_fold == outer_fold:
                continue
            inner_features, evidence = _nested_layer_one_features(
                all_metadata,
                aligned,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                base_config=base_config,
                apex_config=apex_config,
                members=members,
                ensemble_config=config,
            )
            overlap = set(nested_features) & set(inner_features)
            if overlap:
                raise AssertionError(f"nested feature rows repeated across inner folds: {overlap}")
            nested_features.update(inner_features)
            nested_feature_records.extend(
                (outer_fold, inner_fold, example_id, values)
                for example_id, values in inner_features.items()
            )
            nested_evidence.append(evidence)
        calibrators.append(
            _fit_stack(
                aligned,
                heldout_fold=outer_fold,
                nested_features=nested_features,
                config=config,
            )
        )
    calibrator_by_fold = {item.heldout_fold: item for item in calibrators}
    rows: list[dict[str, object]] = []
    for item in aligned:
        probabilities = dict(item.probabilities)
        probabilities[STACK_MODEL] = calibrator_by_fold[item.metadata.fold].predict(
            _stack_inputs(item, config)
        )
        for method in config.methods:
            rows.append(
                {
                    "model": method,
                    **asdict(item.metadata),
                    "target_rule": item.target_rule,
                    "probability": probabilities[method],
                }
            )
    expected_feature_rows = len(aligned) * (len(folds) - 1)
    if (
        len(nested_feature_records) != expected_feature_rows
        or len({(item[0], item[2]) for item in nested_feature_records}) != expected_feature_rows
    ):
        raise ValueError("nested layer-one feature ledger is incomplete or duplicated")
    nested_feature_contract = {
        "rows": len(nested_feature_records),
        "features_per_row": len(config.stack_features),
        "feature_names": list(config.stack_features),
        "digest_encoding": NESTED_FEATURE_DIGEST_ENCODING,
        "rows_sha256": _nested_feature_digest(nested_feature_records),
    }
    return (
        sorted(rows, key=lambda row: (str(row["model"]), str(row["example_id"]))),
        calibrators,
        nested_evidence,
        nested_feature_contract,
    )


def _metric(rows: Sequence[Mapping[str, object]], bins: int) -> dict[str, int | float | None]:
    return binary_metrics(
        [int(item["label"]) for item in rows],
        [float(item["probability"]) for item in rows],
        calibration_bins=bins,
    )


def _summarize(
    rows: Sequence[Mapping[str, object]], config: EnsembleOofConfig
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    methods: dict[str, object] = {}
    for method in config.methods:
        model_rows = grouped[method]
        methods[method] = {
            "overall": _metric(model_rows, config.calibration_bins),
            "by_fold": {
                str(fold): _metric(
                    [item for item in model_rows if int(item["fold"]) == fold],
                    config.calibration_bins,
                )
                for fold in sorted({int(item["fold"]) for item in model_rows})
            },
            "by_target": {
                target: _metric(
                    [item for item in model_rows if str(item["target_rule"]) == target],
                    config.calibration_bins,
                )
                for target in sorted({str(item["target_rule"]) for item in model_rows})
            },
            "by_max_train_identity": {
                f"[{left:.2f},{right:.2f})": _metric(
                    [
                        item
                        for item in model_rows
                        if left <= float(item["max_train_identity"]) < right
                    ],
                    config.calibration_bins,
                )
                for left, right in SIMILARITY_STRATA
                if any(left <= float(item["max_train_identity"]) < right for item in model_rows)
            },
        }
        if set(methods[method]["by_max_train_identity"]) != {  # type: ignore[index]
            "[0.00,0.40)",
            "[0.40,0.60)",
            "[0.60,0.80)",
        }:
            raise ValueError("supported panel does not cover every fixed similarity stratum")

    aligned_by_method = {
        method: {str(item["example_id"]): item for item in grouped[method]}
        for method in config.methods
    }
    example_ids = sorted(aligned_by_method[config.methods[0]])
    cluster_examples: dict[str, list[str]] = defaultdict(list)
    for example_id in example_ids:
        cluster_examples[
            str(aligned_by_method[config.methods[0]][example_id]["cluster_id"])
        ].append(example_id)
    clusters = tuple(sorted(cluster_examples))
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss")
    bootstrap: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in metric_names} for method in config.methods
    }
    deltas: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in metric_names}
        for method in config.methods
        if method != config.promotion_reference
    }
    generator = np.random.default_rng(config.seed)
    successful_replicates = 0
    attempted_replicates = 0
    for _ in range(config.bootstrap_replicates):
        attempted_replicates += 1
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        sampled_ids = [example for cluster in sampled for example in cluster_examples[str(cluster)]]
        replicate: dict[str, dict[str, int | float | None]] = {}
        for method in config.methods:
            replicate[method] = _metric(
                [aligned_by_method[method][example] for example in sampled_ids],
                config.calibration_bins,
            )
        if any(
            replicate[method][metric] is None or not math.isfinite(float(replicate[method][metric]))
            for method in config.methods
            for metric in metric_names
        ):
            raise ValueError(
                f"paired component bootstrap replicate {attempted_replicates} is undefined/non-finite"
            )
        successful_replicates += 1
        for method in config.methods:
            for metric in metric_names:
                bootstrap[method][metric].append(float(replicate[method][metric]))
        baseline = replicate[config.promotion_reference]
        for method in deltas:
            for metric in metric_names:
                value = replicate[method][metric]
                reference = baseline[metric]
                assert value is not None and reference is not None
                deltas[method][metric].append(float(value) - float(reference))

    if any(
        len(values) != config.bootstrap_replicates
        for method_values in bootstrap.values()
        for values in method_values.values()
    ) or any(
        len(values) != config.bootstrap_replicates
        for method_values in deltas.values()
        for values in method_values.values()
    ):
        raise ValueError("paired component bootstrap evidence is incomplete")

    for method in config.methods:
        method_metrics = methods[method]
        assert isinstance(method_metrics, dict)
        method_metrics["homology_cluster_bootstrap_95ci"] = {
            metric: {
                "lower": None if not values else float(np.quantile(values, 0.025)),
                "upper": None if not values else float(np.quantile(values, 0.975)),
                "successful_replicates": len(values),
            }
            for metric, values in bootstrap[method].items()
        }
    point_baseline = methods[config.promotion_reference]
    assert isinstance(point_baseline, dict)
    baseline_overall = point_baseline["overall"]
    assert isinstance(baseline_overall, dict)
    paired: dict[str, object] = {}
    for method, metric_values in deltas.items():
        method_entry = methods[method]
        assert isinstance(method_entry, dict)
        method_overall = method_entry["overall"]
        assert isinstance(method_overall, dict)
        paired[method] = {
            metric: {
                "point": (
                    None
                    if method_overall[metric] is None or baseline_overall[metric] is None
                    else float(method_overall[metric]) - float(baseline_overall[metric])
                ),
                "lower": None if not values else float(np.quantile(values, 0.025)),
                "upper": None if not values else float(np.quantile(values, 0.975)),
                "successful_replicates": len(values),
            }
            for metric, values in metric_values.items()
        }
    promotion_decisions: dict[str, object] = {}
    for candidate in config.promotion_candidates:
        comparison = paired[candidate]
        assert isinstance(comparison, dict)
        auc = comparison["roc_auc"]
        brier = comparison["brier"]
        log_loss = comparison["log_loss"]
        assert isinstance(auc, dict) and isinstance(brier, dict) and isinstance(log_loss, dict)
        checks = {
            "roc_auc_delta_lower_above_minimum": (
                auc["lower"] is not None
                and float(auc["lower"]) > config.promotion_auc_delta_lower_minimum
            ),
            "brier_delta_upper_below_maximum": (
                brier["upper"] is not None
                and float(brier["upper"]) < config.promotion_brier_delta_upper_maximum
            ),
            "log_loss_delta_upper_below_maximum": (
                log_loss["upper"] is not None
                and float(log_loss["upper"]) < config.promotion_log_loss_delta_upper_maximum
            ),
        }
        promotion_decisions[candidate] = {
            "promoted": all(checks.values()),
            "passed": all(checks.values()),
            "checks": checks,
        }
    matrix = np.asarray(
        [
            [float(aligned_by_method[method][example]["probability"]) for example in example_ids]
            for method in config.methods
        ],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.corrcoef(matrix)
    if not np.all(np.isfinite(correlations)):
        raise ValueError("ensemble method probability correlations are non-finite")
    return {
        "methods": methods,
        f"paired_deltas_vs_{config.promotion_reference}": paired,
        "promotion_rule": {
            "scope": "statistical_offline_gate_only",
            "reference": config.promotion_reference,
            "candidates": list(config.promotion_candidates),
            "requirements": {
                "roc_auc_delta_bootstrap_95ci_lower_strictly_above": config.promotion_auc_delta_lower_minimum,
                "brier_delta_bootstrap_95ci_upper_strictly_below": config.promotion_brier_delta_upper_maximum,
                "log_loss_delta_bootstrap_95ci_upper_strictly_below": config.promotion_log_loss_delta_upper_maximum,
            },
            "decisions": promotion_decisions,
        },
        "paired_component_bootstrap": {
            "unit": "homology_cluster",
            "requested_successful_replicates": config.bootstrap_replicates,
            "successful_replicates": successful_replicates,
            "attempted_replicates": attempted_replicates,
            "rejected_replicates": attempted_replicates - successful_replicates,
            "seed": config.seed,
        },
        "probability_correlation": {
            method: {
                other: float(correlations[index, other_index])
                for other_index, other in enumerate(config.methods)
            }
            for index, method in enumerate(config.methods)
        },
    }


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("refusing to serialize a non-finite ensemble value")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _format(value: object) -> object:
    return repr(value) if isinstance(value, float) else value


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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
        "target_rule",
        "probability",
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row[field]) for field in fields})


def run_ensemble_oof(
    *,
    base_oof_path: str | Path,
    base_manifest_path: str | Path,
    base_folds_path: str | Path,
    base_split_compatibility_path: str | Path,
    base_checksums_path: str | Path,
    base_config_path: str | Path,
    apex_oof_path: str | Path,
    apex_manifest_path: str | Path,
    apex_coverage_path: str | Path,
    apex_calibrators_path: str | Path,
    apex_checksums_path: str | Path,
    apex_config_path: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    frozen_input_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Align the supported subset and write true nested-stack comparison evidence."""

    base_oof = Path(base_oof_path).resolve()
    base_manifest = Path(base_manifest_path).resolve()
    base_folds = Path(base_folds_path).resolve()
    base_split = Path(base_split_compatibility_path).resolve()
    base_checksums = Path(base_checksums_path).resolve()
    base_config_path = Path(base_config_path).resolve()
    apex_oof = Path(apex_oof_path).resolve()
    apex_manifest = Path(apex_manifest_path).resolve()
    apex_coverage = Path(apex_coverage_path).resolve()
    apex_calibrators = Path(apex_calibrators_path).resolve()
    apex_checksums = Path(apex_checksums_path).resolve()
    apex_config_path = Path(apex_config_path).resolve()
    code_manifest = Path(code_manifest_path).resolve()
    frozen_input_manifest = Path(frozen_input_manifest_path).resolve()
    if GIT_SHA1_RE.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase SHA-1 commit ID")
    config = load_config(config_path)
    code_entries = _read_sha256_manifest(code_manifest, description="ensemble code manifest")
    frozen_entries = _read_sha256_manifest(
        frozen_input_manifest,
        description="ensemble frozen-input manifest",
    )
    base_checksums_sha256 = _sha256(base_checksums)
    apex_checksums_sha256 = _sha256(apex_checksums)
    if base_checksums_sha256 != ACCEPTED_BASE_CHECKSUMS_SHA256:
        raise ValueError("base checksums do not match accepted Gate-1-v7 job 223009")
    if apex_checksums_sha256 != ACCEPTED_APEX_CHECKSUMS_SHA256:
        raise ValueError("APEX checksums do not match accepted APEX-v7 job 223027")
    for logical_path, source in (
        ("src/amp_challenge/benchmarks/ensemble_oof.py", Path(__file__).resolve()),
        ("configs/benchmarks/oracle_gate1.toml", base_config_path),
        ("configs/models/apex_oof.toml", apex_config_path),
        ("configs/evaluation/ensemble_oof.toml", config.path),
    ):
        _require_manifest_entry(
            code_entries,
            logical_path=logical_path,
            source=source,
            manifest_name="ensemble code manifest",
        )
    for logical_path, source in (
        ("base/oof_predictions.csv", base_oof),
        ("base/manifest.json", base_manifest),
        ("base/folds.json", base_folds),
        ("base/split_compatibility.json", base_split),
        ("base/SHA256SUMS", base_checksums),
        ("apex/apex_oof_predictions.csv", apex_oof),
        ("apex/manifest.json", apex_manifest),
        ("apex/sequence_coverage_receipt.json", apex_coverage),
        ("apex/calibrators.json", apex_calibrators),
        ("apex/SHA256SUMS", apex_checksums),
    ):
        _require_manifest_entry(
            frozen_entries,
            logical_path=logical_path,
            source=source,
            manifest_name="ensemble frozen-input manifest",
        )

    base_config = Gate1Config.from_toml(base_config_path)
    apex_config = load_apex_oof_config(apex_config_path)
    if (
        _sha256(base_config_path) != BASE_CONFIG_SHA256
        or base_config.homology_identity_threshold != 0.8
        or base_config.folds != 5
        or base_config.similarity_bin_edges != (0.0, 0.4, 0.6, 0.8)
    ):
        raise ValueError("Gate-1 config differs from the accepted parser-v7 fold contract")
    if _sha256(apex_config_path) != APEX_CONFIG_SHA256:
        raise ValueError("APEX config differs from the accepted APEX-v7 contract")
    base_document, _ = _verify_base_manifest(
        base_oof,
        base_manifest,
        base_folds,
        base_split,
        base_checksums,
        base_config_path,
    )
    apex_document, _ = _verify_apex_manifest(
        apex_oof,
        apex_manifest,
        apex_coverage,
        apex_calibrators,
        apex_checksums,
        base_oof_path=base_oof,
        base_manifest_path=base_manifest,
        base_folds_path=base_folds,
        base_config_path=base_config_path,
        apex_config_path=apex_config_path,
    )
    members = _apex_members(apex_document, apex_config)
    metadata, base_predictions = _read_base_oof(base_oof)
    _verify_base_folds(
        base_folds,
        metadata=metadata,
        config=base_config,
        manifest=base_document,
    )
    _verify_split_compatibility(
        base_split,
        metadata=metadata,
        config=base_config,
        manifest=base_document,
    )
    observed_folds = set(range(base_config.folds))
    base_evidence = _partition_evidence(tuple(metadata.values()))
    label_summary = base_document.get("label_summary")
    if (
        not isinstance(label_summary, dict)
        or label_summary.get("activity_threshold_um") != base_config.activity_threshold_um
        or label_summary.get("included_strain_level_examples") != len(metadata)
        or label_summary.get("positive_examples") != base_evidence["positive_examples"]
        or label_summary.get("negative_examples") != base_evidence["negative_examples"]
    ):
        raise ValueError("base Gate-1 label summary differs from its OOF metadata")
    if not math.isclose(
        base_config.activity_threshold_um,
        apex_config.activity_threshold_um,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("base and APEX activity thresholds differ")
    apex_predictions = _read_apex_oof(apex_oof, members, apex_config)
    if apex_document.get("input_examples") != len(metadata):
        raise ValueError("APEX manifest input-example count differs from the base OOF")
    if apex_document.get("supported_examples") != len(apex_predictions):
        raise ValueError("APEX manifest supported-example count differs from its prediction rows")
    aligned = _align(metadata, base_predictions, apex_predictions)
    if {item.metadata.fold for item in aligned} != observed_folds:
        raise ValueError("APEX-supported examples must cover every configured fold")
    for fold in sorted(observed_folds):
        _require_two_classes(
            [item.metadata for item in aligned if item.metadata.fold == fold],
            description=f"APEX-supported fold {fold}",
        )
    _verify_apex_coverage(
        apex_coverage,
        metadata=metadata,
        apex=apex_predictions,
        members=members,
        manifest=apex_document,
    )
    _verify_apex_calibrators(
        apex_calibrators,
        aligned=aligned,
        members=members,
        folds=sorted(observed_folds),
        config=apex_config,
    )
    supported_evidence = _partition_evidence([item.metadata for item in aligned])
    support_by_fold: dict[str, dict[str, int]] = {}
    for fold in sorted(observed_folds):
        selected = [item for item in aligned if item.metadata.fold == fold]
        positives = sum(item.metadata.label for item in selected)
        support_by_fold[str(fold)] = {
            "examples": len(selected),
            "positive_examples": positives,
            "negative_examples": len(selected) - positives,
        }
    support_by_target: dict[str, dict[str, int]] = {}
    for target in sorted({item.target_rule for item in aligned}):
        selected = [item for item in aligned if item.target_rule == target]
        positives = sum(item.metadata.label for item in selected)
        support_by_target[target] = {
            "examples": len(selected),
            "positive_examples": positives,
            "negative_examples": len(selected) - positives,
        }
    support_by_similarity = {
        f"[{left:.2f},{right:.2f})": sum(
            left <= item.metadata.max_train_identity < right for item in aligned
        )
        for left, right in SIMILARITY_STRATA
    }
    if any(base_evidence[key] != value for key, value in ACCEPTED_BASE_EVIDENCE.items()):
        raise ValueError("accepted Gate-1 v7 census/digests differ from job 223009")
    if any(supported_evidence[key] != value for key, value in ACCEPTED_SUPPORT_EVIDENCE.items()):
        raise ValueError("accepted APEX-v7 support census/digests differ from job 223027")
    if support_by_fold != ACCEPTED_SUPPORT_BY_FOLD:
        raise ValueError("accepted APEX-v7 fold census differs from job 223027")
    if support_by_target != ACCEPTED_SUPPORT_BY_TARGET:
        raise ValueError("accepted APEX-v7 target census differs from job 223027")
    if support_by_similarity != ACCEPTED_SUPPORT_BY_SIMILARITY:
        raise ValueError("accepted APEX-v7 similarity census differs from job 223027")
    expected_base_contract = {
        "benchmark": "gate1_strain_activity",
        "configured_folds": base_config.folds,
        "fold_assignment_policy": BASE_FOLD_ASSIGNMENT_POLICY,
        "fold_policy": BASE_FOLD_POLICY,
        "homology_clusters": base_evidence["homology_clusters"],
        "identity_threshold": base_config.homology_identity_threshold,
        "input_examples": base_evidence["examples"],
        "negative_examples": base_evidence["negative_examples"],
        "normalized_parser_id": BASE_PARSER_ID,
        "observed_folds": sorted(observed_folds),
        "positive_examples": base_evidence["positive_examples"],
        "supported_examples": supported_evidence["examples"],
        "supported_homology_clusters": supported_evidence["homology_clusters"],
        "supported_negative_examples": supported_evidence["negative_examples"],
        "supported_positive_examples": supported_evidence["positive_examples"],
        "supported_unique_sequences": supported_evidence["unique_sequences"],
        "unique_sequences": base_evidence["unique_sequences"],
    }
    if apex_document.get("base_contract") != expected_base_contract:
        raise ValueError("APEX-v7 base contract differs from the accepted Gate-1 evidence")
    rows, calibrators, nested_evidence, nested_feature_contract = _make_prediction_rows(
        aligned,
        all_metadata=metadata,
        base_config=base_config,
        apex_config=apex_config,
        members=members,
        config=config,
    )
    if (
        len(calibrators) != base_config.folds
        or len(nested_evidence) != base_config.folds * (base_config.folds - 1)
        or any(not item.converged or item.iterations < 1 for item in calibrators)
        or any(
            len(item["apex_member_calibrators"]) != len(members)
            or not all(
                value["converged"] and value["iterations"] >= 1
                for value in item["apex_member_calibrators"]
            )
            for item in nested_evidence
        )
    ):
        raise ValueError("nested ensemble fit evidence is incomplete or non-converged")
    if nested_feature_contract["rows"] != ACCEPTED_NESTED_FEATURE_ROWS:
        raise ValueError("accepted nested layer-one feature ledger must contain 10,368 rows")
    metrics = _summarize(rows, config)
    promotion = metrics["promotion_rule"]
    assert isinstance(promotion, dict)
    decisions = promotion["decisions"]
    assert isinstance(decisions, dict)
    statistical_gate_passed = any(
        isinstance(value, dict) and value.get("passed") is True for value in decisions.values()
    )

    final_output = Path(output_dir).absolute()
    if final_output.exists() or final_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing ensemble OOF output: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.staging-",
            dir=final_output.parent,
        )
    )
    predictions_out = output / "ensemble_oof_predictions.csv"
    calibrators_out = output / "stack_calibrators.json"
    nested_evidence_out = output / "nested_layer1_fits.json"
    metrics_out = output / "metrics.json"
    manifest_out = output / "manifest.json"
    folds = sorted({item.metadata.fold for item in aligned})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "base_apex_nested_oof_ensemble",
        "base_oof_sha256": _sha256(base_oof),
        "base_manifest_sha256": _sha256(base_manifest),
        "base_folds_sha256": _sha256(base_folds),
        "base_split_compatibility_sha256": _sha256(base_split),
        "base_checksums_sha256": _sha256(base_checksums),
        "base_config_sha256": _sha256(base_config_path),
        "apex_oof_predictions_sha256": _sha256(apex_oof),
        "apex_oof_manifest_sha256": _sha256(apex_manifest),
        "apex_coverage_sha256": _sha256(apex_coverage),
        "apex_calibrators_sha256": _sha256(apex_calibrators),
        "apex_checksums_sha256": _sha256(apex_checksums),
        "apex_config_sha256": _sha256(apex_config_path),
        "config_sha256": _sha256(config.path),
        "methods_predeclared": list(config.methods),
        "stack_features_predeclared": list(config.stack_features),
        "fold_policy": ENSEMBLE_FOLD_POLICY,
        "stack_input_transform": config.stack_input_transform,
        "stack_evaluation_status": config.stack_evaluation_status,
        "promotion_reference": config.promotion_reference,
        "promotion_candidates": list(config.promotion_candidates),
        "base_examples": len(metadata),
        "supported_examples": len(aligned),
        "coverage_fraction": len(aligned) / len(metadata),
        "supported_homology_clusters": len({item.metadata.cluster_id for item in aligned}),
        "supported_examples_by_fold": {
            str(fold): sum(item.metadata.fold == fold for item in aligned) for fold in folds
        },
        "apex_upstream_training_independence": apex_document["upstream_training_independence"],
        "support_contract": supported_evidence,
        "support_by_fold": support_by_fold,
        "support_by_target": support_by_target,
        "support_by_max_train_identity": support_by_similarity,
        "accepted_upstream_contract": True,
        "digest_encodings": {
            "set": SET_DIGEST_ENCODING,
            "assignment": ASSIGNMENT_DIGEST_ENCODING,
        },
        "nested_fit_contract": {
            "outer_stack_fits": len(calibrators),
            "ordered_outer_inner_splits": len(nested_evidence),
            "nested_apex_member_fits": len(nested_evidence) * len(members),
            "all_numerical_fits_converged": True,
            "descriptor_fits": len(nested_evidence),
        },
        "nested_layer1_feature_contract": nested_feature_contract,
        "statistical_gate_passed": statistical_gate_passed,
        "production_policy": {
            "production_eligible": False,
            "production_apex_weight": 0.0,
            "statistical_gate_controls_eligibility": False,
            "blockers": list(PRODUCTION_BLOCKERS),
        },
        "fold_policy_checks": {
            "base_manifest_declares_homology_group_holdout": True,
            "apex_manifest_declares_fold_local_calibration": True,
            "supported_clusters_each_map_to_one_fold": True,
            "nested_base_fit_excludes_outer_and_inner_folds": True,
            "nested_apex_calibration_excludes_outer_and_inner_folds": True,
            "meta_fit_excludes_outer_fold": True,
        },
        "interpretation_caveat": (
            "APEX checkpoint training membership is not established; this benchmark proves "
            "nested calibration and stack fitting, not upstream checkpoint independence"
        ),
        "limitations": [
            "accepted_gate1_oof_descriptor_fit_iterations_not_recorded; "
            "all_20_nested_descriptor_refits_are_replayed_and_convergence_audited"
        ],
        "base_source_artifacts": base_document.get("source_artifacts", []),
        "provenance": {
            "code_manifest": {
                "filename": code_manifest.name,
                "sha256": _sha256(code_manifest),
            },
            "frozen_input_manifest": {
                "filename": frozen_input_manifest.name,
                "sha256": _sha256(frozen_input_manifest),
            },
            "git_commit": git_commit,
            "base_top_checksums": {
                "filename": base_checksums.name,
                "sha256": _sha256(base_checksums),
            },
            "apex_top_checksums": {
                "filename": apex_checksums.name,
                "sha256": _sha256(apex_checksums),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {},
    }
    try:
        _write_predictions(predictions_out, rows)
        _write_json(calibrators_out, [asdict(item) for item in calibrators])
        _write_json(nested_evidence_out, nested_evidence)
        _write_json(metrics_out, metrics)
        manifest["outputs"] = {
            "ensemble_oof_predictions.csv": _sha256(predictions_out),
            "stack_calibrators.json": _sha256(calibrators_out),
            "nested_layer1_fits.json": _sha256(nested_evidence_out),
            "metrics.json": _sha256(metrics_out),
        }
        _write_json(manifest_out, manifest)
        output.rename(final_output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
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
    parser.add_argument("--apex-oof", type=Path, required=True)
    parser.add_argument("--apex-manifest", type=Path, required=True)
    parser.add_argument("--apex-coverage", type=Path, required=True)
    parser.add_argument("--apex-calibrators", type=Path, required=True)
    parser.add_argument("--apex-checksums", type=Path, required=True)
    parser.add_argument("--apex-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--frozen-input-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_ensemble_oof(
        base_oof_path=args.base_oof,
        base_manifest_path=args.base_manifest,
        base_folds_path=args.base_folds,
        base_split_compatibility_path=args.base_split_compatibility,
        base_checksums_path=args.base_checksums,
        base_config_path=args.base_config,
        apex_oof_path=args.apex_oof,
        apex_manifest_path=args.apex_manifest,
        apex_coverage_path=args.apex_coverage,
        apex_calibrators_path=args.apex_calibrators,
        apex_checksums_path=args.apex_checksums,
        apex_config_path=args.apex_config,
        config_path=args.config,
        code_manifest_path=args.code_manifest,
        frozen_input_manifest_path=args.frozen_input_manifest,
        git_commit=args.git_commit,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

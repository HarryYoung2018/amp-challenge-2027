"""Calibrate frozen APEX members strictly within homology-held-out folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from amp_challenge.benchmarks.oracle_gate1 import Gate1Config, binary_metrics
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_BASE_MODELS = ("descriptor_logistic", "homology_knn", "equal_weight_ensemble")
_BASE_FOLD_POLICY = "sequence-level single-link homology groups held out together"
_BASE_FOLD_ASSIGNMENT_POLICY = "full_sequence_union_bridge_audit_then_reuse_frozen_example_folds"
_BASE_PARSER_ID = "amp_challenge.data.dramp:v7"
_CALIBRATION_FOLD_POLICY = "reuse base Gate-1 homology folds; calibrate on non-heldout folds only"
_SIMILARITY_STRATA = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8))
_SET_DIGEST_ENCODING = (
    "sorted unique identifier strings encoded as ASCII, joined by LF with a terminal LF; "
    "empty set is empty bytes"
)
_ASSIGNMENT_DIGEST_ENCODING = (
    "example_id, sequence_id, label, fold, cluster_id encoded as tab-separated UTF-8 "
    "records, sorted by example_id, with a terminal LF"
)
_APEX_ENDPOINTS = (
    ("A. baumannii ATCC 19606", "apex_mic_um__a_baumannii_atcc_19606"),
    ("E. coli ATCC 11775", "apex_mic_um__e_coli_atcc_11775"),
    ("E. coli AIC221", "apex_mic_um__e_coli_aic221"),
    ("E. coli AIC222", "apex_mic_um__e_coli_aic222"),
    ("K. pneumoniae ATCC 13883", "apex_mic_um__k_pneumoniae_atcc_13883"),
    ("P. aeruginosa PA01", "apex_mic_um__p_aeruginosa_pa01"),
    ("P. aeruginosa PA14", "apex_mic_um__p_aeruginosa_pa14"),
    ("S. aureus ATCC 12600", "apex_mic_um__s_aureus_atcc_12600"),
    (
        "S. aureus (ATCC BAA-1556) - MRSA",
        "apex_mic_um__s_aureus_atcc_baa_1556_mrsa",
    ),
    (
        "vancomycin-resistant E. faecalis ATCC 700802",
        "apex_mic_um__vancomycin_resistant_e_faecalis_atcc_700802",
    ),
    (
        "vancomycin-resistant E. faecium ATCC 700221",
        "apex_mic_um__vancomycin_resistant_e_faecium_atcc_700221",
    ),
)


@dataclass(frozen=True, slots=True)
class TargetRule:
    name: str
    strain_regex: str
    endpoints: tuple[str, ...]

    def matches(self, strain: str) -> bool:
        return re.search(self.strain_regex, strain) is not None


@dataclass(frozen=True, slots=True)
class ApexOofConfig:
    path: Path
    metadata_model: str
    expected_member_count: int
    calibration_l2: float
    calibration_prior_strength: float
    calibration_max_iterations: int
    calibration_tolerance: float
    calibration_bins: int
    bootstrap_replicates: int
    seed: int
    activity_threshold_um: float
    targets: tuple[TargetRule, ...]


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    sequence_id: str
    sequence: str
    strain: str
    gram: str
    label: int
    source_observations: int
    fold: int
    cluster_id: str
    max_train_identity: float


@dataclass(frozen=True, slots=True)
class SupportedExample:
    example: Example
    target: TargetRule
    signals: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Calibrator:
    fold: int
    member_id: str
    training_examples: int
    positives: int
    iterations: int
    converged: bool
    signal_mean: float
    signal_scale: float
    intercept: float
    slope: float

    def predict(self, signal: float) -> float:
        logit = self.intercept + self.slope * ((signal - self.signal_mean) / self.signal_scale)
        return float(np.clip(_sigmoid_scalar(logit), 1e-6, 1.0 - 1e-6))


@dataclass(frozen=True, slots=True)
class ApexPredictions:
    members: tuple[str, ...]
    member_hashes: Mapping[str, str]
    values: Mapping[str, Mapping[str, Mapping[str, float]]]
    sequences: Mapping[str, str]
    row_count: int


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_file(path: str | Path, *, name: str) -> bytes:
    source = Path(path).resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} is not a regular file: {source}")
    payload = source.read_bytes()
    after = source.stat()
    fingerprint_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    fingerprint_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if fingerprint_before != fingerprint_after or len(payload) != before.st_size:
        raise ValueError(f"{name} changed while it was being read: {source}")
    return payload


def _read_json_object(path: str | Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(_read_regular_file(path, name=name))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _verify_sha256_manifest(path: Path, *, name: str) -> dict[str, str]:
    try:
        text = _read_regular_file(path, name=name).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8 text") from error
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise ValueError(f"{name} line {line_number} is not a SHA-256 manifest entry")
        digest, _, filename = match.groups()
        if filename in entries:
            raise ValueError(f"{name} contains duplicate path {filename!r}")
        entries[filename] = digest
    if not entries:
        raise ValueError(f"{name} must contain at least one SHA-256 entry")
    return entries


def _require_manifest_digest(
    entries: Mapping[str, str],
    source: Path,
    *,
    entry: str,
    manifest_name: str,
    artifact_name: str,
) -> None:
    digest = _sha256(source)
    if entries.get(entry) != digest:
        raise ValueError(
            f"{manifest_name} does not attest {artifact_name} at expected entry {entry!r}"
        )


def _sequence_set_digest(sequence_ids: Sequence[str] | set[str]) -> str:
    ordered = sorted(set(sequence_ids))
    payload = b"" if not ordered else ("\n".join(ordered) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _assignment_digest(examples: Sequence[Example]) -> str:
    payload = "".join(
        f"{item.example_id}\t{item.sequence_id}\t{item.label}\t{item.fold}\t{item.cluster_id}\n"
        for item in sorted(examples, key=lambda value: value.example_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_artifact_entry(
    artifacts: object,
    *,
    key: str,
    path: Path,
    manifest_name: str,
) -> None:
    if not isinstance(artifacts, dict):
        raise ValueError(f"{manifest_name} has no artifact table")
    entry = artifacts.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"{manifest_name} has no {key!r} artifact")
    if entry.get("filename") != path.name or entry.get("sha256") != _sha256(path):
        raise ValueError(f"{manifest_name} does not match the supplied {key} artifact")


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"APEX OOF config {field!r} must be a non-empty string")
    return value.strip()


def load_config(path: str | Path) -> ApexOofConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {
        "schema_version",
        "metadata_model",
        "expected_member_count",
        "calibration_l2",
        "calibration_prior_strength",
        "calibration_max_iterations",
        "calibration_tolerance",
        "calibration_bins",
        "bootstrap_replicates",
        "seed",
        "activity_threshold_um",
        "target",
    }
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"unexpected APEX OOF config key(s): {sorted(extra)}")
    if raw.get("schema_version") != 1:
        raise ValueError("APEX OOF config schema_version must be 1")
    targets_raw = raw.get("target")
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("APEX OOF config requires at least one [[target]]")
    targets: list[TargetRule] = []
    endpoint_owner: dict[str, str] = {}
    for index, item in enumerate(targets_raw):
        if not isinstance(item, dict) or set(item) != {"name", "strain_regex", "endpoints"}:
            raise ValueError(f"invalid target table at index {index}")
        name = _require_string(item["name"], f"target[{index}].name")
        pattern = _require_string(item["strain_regex"], f"target[{index}].strain_regex")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"invalid target regex {name!r}: {error}") from error
        endpoints_raw = item["endpoints"]
        if (
            not isinstance(endpoints_raw, list)
            or not endpoints_raw
            or not all(isinstance(value, str) and value.strip() for value in endpoints_raw)
        ):
            raise ValueError(f"target {name!r} endpoints must be a non-empty string array")
        endpoints = tuple(value.strip() for value in endpoints_raw)
        if len(endpoints) != len(set(endpoints)):
            raise ValueError(f"target {name!r} endpoints must be unique")
        for endpoint in endpoints:
            previous = endpoint_owner.setdefault(endpoint, name)
            if previous != name:
                raise ValueError(f"APEX endpoint {endpoint!r} belongs to multiple targets")
        targets.append(TargetRule(name=name, strain_regex=pattern, endpoints=endpoints))
    names = [item.name for item in targets]
    if len(names) != len(set(names)):
        raise ValueError("APEX target names must be unique")

    def positive_float(name: str) -> float:
        value = float(raw[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"APEX OOF config {name} must be finite and positive")
        return value

    expected_members = int(raw["expected_member_count"])
    max_iterations = int(raw["calibration_max_iterations"])
    bins = int(raw["calibration_bins"])
    bootstrap = int(raw["bootstrap_replicates"])
    if expected_members < 2 or max_iterations < 1 or bins < 2 or bootstrap < 0:
        raise ValueError("invalid APEX member/calibration/bootstrap count")
    prior = float(raw["calibration_prior_strength"])
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("calibration_prior_strength must be finite and non-negative")
    return ApexOofConfig(
        path=config_path,
        metadata_model=_require_string(raw["metadata_model"], "metadata_model"),
        expected_member_count=expected_members,
        calibration_l2=positive_float("calibration_l2"),
        calibration_prior_strength=prior,
        calibration_max_iterations=max_iterations,
        calibration_tolerance=positive_float("calibration_tolerance"),
        calibration_bins=bins,
        bootstrap_replicates=bootstrap,
        seed=int(raw["seed"]),
        activity_threshold_um=positive_float("activity_threshold_um"),
        targets=tuple(targets),
    )


def _verify_base_manifest(
    oof_path: Path,
    folds_path: Path,
    config_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    document = _read_json_object(manifest_path, name="base OOF manifest")
    if document.get("schema_version") != 1 or document.get("benchmark") != "gate1_strain_activity":
        raise ValueError("base OOF manifest is not a Gate-1 benchmark manifest")
    if document.get("normalized_schema_version") != 2:
        raise ValueError("base OOF manifest is not bound to normalized schema v2")
    if document.get("normalized_parser_id") != _BASE_PARSER_ID:
        raise ValueError("base OOF manifest is not bound to the DRAMP v7 parser")
    if document.get("config_sha256") != _sha256(config_path):
        raise ValueError("base OOF manifest does not match the supplied Gate-1 config")
    if document.get("models") != list(_BASE_MODELS):
        raise ValueError("base OOF manifest does not declare the frozen Gate-1 model set")
    if document.get("fold_policy") != _BASE_FOLD_POLICY:
        raise ValueError("base OOF manifest does not declare the required fold policy")
    if document.get("fold_assignment_policy") != _BASE_FOLD_ASSIGNMENT_POLICY:
        raise ValueError("base OOF manifest does not declare the accepted fold assignment policy")
    artifacts = document.get("artifacts")
    _verify_artifact_entry(
        artifacts,
        key="oof",
        path=oof_path,
        manifest_name="base OOF manifest",
    )
    _verify_artifact_entry(
        artifacts,
        key="folds",
        path=folds_path,
        manifest_name="base OOF manifest",
    )
    return document


def _read_examples(path: Path, metadata_model: str) -> tuple[Example, ...]:
    required = (
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
    by_model: dict[str, dict[str, Example]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(required):
            raise ValueError("base OOF CSV does not have the frozen Gate-1 prediction schema")
        for row_number, row in enumerate(reader, start=2):
            model = row["model"]
            if model not in _BASE_MODELS:
                raise ValueError(f"base OOF row {row_number} has an unexpected model")
            sequence = canonicalize_sequence(row["sequence"])
            sequence_id = row["sequence_id"]
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(f"base OOF row {row_number} sequence_id mismatch")
            example_id = row["example_id"].strip()
            if not example_id or example_id in by_model[model]:
                raise ValueError(f"base OOF row {row_number} has duplicate/empty example_id")
            label = int(row["label"])
            fold = int(row["fold"])
            identity = float(row["max_train_identity"])
            source_observations = int(row["source_observations"])
            probability = float(row["probability"])
            if (
                label not in {0, 1}
                or fold < 0
                or not math.isfinite(identity)
                or not 0 <= identity <= 1
                or source_observations < 1
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError(f"base OOF row {row_number} has invalid label/fold/identity")
            strain = row["strain"].strip()
            cluster = row["cluster_id"].strip()
            gram = row["gram"].strip()
            if not strain or not cluster or gram not in {"positive", "negative", "unknown"}:
                raise ValueError(f"base OOF row {row_number} has empty strain/cluster")
            by_model[model][example_id] = Example(
                example_id=example_id,
                sequence_id=sequence_id,
                sequence=sequence,
                strain=strain,
                gram=gram,
                label=label,
                source_observations=source_observations,
                fold=fold,
                cluster_id=cluster,
                max_train_identity=identity,
            )
    if set(by_model) != set(_BASE_MODELS):
        raise ValueError("base OOF CSV does not contain every frozen Gate-1 model")
    examples_by_id = by_model[metadata_model]
    for model in _BASE_MODELS:
        if by_model[model] != examples_by_id:
            raise ValueError(f"base OOF metadata/support differs for model {model!r}")
    examples = list(examples_by_id.values())
    if not examples:
        raise ValueError(f"base OOF CSV has no rows for metadata model {metadata_model!r}")
    if len({item.fold for item in examples}) < 2:
        raise ValueError("base OOF metadata must contain at least two folds")
    return tuple(sorted(examples, key=lambda item: item.example_id))


def _verify_base_folds(
    path: Path,
    *,
    examples: Sequence[Example],
    config: Gate1Config,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    document = _read_json_object(path, name="base Gate-1 folds")
    if set(document) != {"assignments", "folds", "identity_threshold", "method", "seed"}:
        raise ValueError("base Gate-1 folds artifact has an unexpected schema")
    if document.get("method") != _BASE_FOLD_ASSIGNMENT_POLICY or document.get(
        "method"
    ) != manifest.get("fold_assignment_policy"):
        raise ValueError("base Gate-1 folds artifact has the wrong assignment policy")
    if document.get("identity_threshold") != config.homology_identity_threshold:
        raise ValueError("base Gate-1 folds identity threshold differs from its config")
    if document.get("seed") != config.seed:
        raise ValueError("base Gate-1 folds seed differs from its config")

    assignments = document.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(examples):
        raise ValueError("base Gate-1 folds assignments do not cover every example")
    observed_assignments: dict[str, tuple[str, int, str]] = {}
    for index, item in enumerate(assignments):
        if not isinstance(item, dict) or set(item) != {
            "cluster_id",
            "example_id",
            "fold",
            "sequence_id",
        }:
            raise ValueError(f"base Gate-1 fold assignment {index} is invalid")
        example_id = item.get("example_id")
        sequence_id = item.get("sequence_id")
        cluster_id = item.get("cluster_id")
        fold = item.get("fold")
        if (
            not isinstance(example_id, str)
            or not isinstance(sequence_id, str)
            or _SHA256.fullmatch(sequence_id) is None
            or not isinstance(cluster_id, str)
            or not cluster_id
            or not isinstance(fold, int)
            or isinstance(fold, bool)
            or example_id in observed_assignments
        ):
            raise ValueError(f"base Gate-1 fold assignment {index} is invalid")
        observed_assignments[example_id] = (sequence_id, fold, cluster_id)
    expected_assignments = {
        item.example_id: (item.sequence_id, item.fold, item.cluster_id) for item in examples
    }
    if observed_assignments != expected_assignments:
        raise ValueError("base Gate-1 folds assignments differ from the supplied OOF metadata")

    expected_folds = set(range(config.folds))
    observed_folds = {item.fold for item in examples}
    if observed_folds != expected_folds:
        raise ValueError("base Gate-1 OOF does not cover every configured fold exactly")
    cluster_folds: dict[str, set[int]] = defaultdict(set)
    sequence_assignments: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for item in examples:
        cluster_folds[item.cluster_id].add(item.fold)
        sequence_assignments[item.sequence_id].add((item.fold, item.cluster_id))
        if not item.max_train_identity < config.homology_identity_threshold:
            raise ValueError("base Gate-1 OOF violates the held-out homology threshold")
    if any(len(folds) != 1 for folds in cluster_folds.values()):
        raise ValueError("base Gate-1 homology component appears in multiple folds")
    if any(len(values) != 1 for values in sequence_assignments.values()):
        raise ValueError("base Gate-1 sequence appears in multiple fold/component assignments")

    expected_summary: dict[str, dict[str, int]] = {}
    for fold in sorted(observed_folds):
        selected = [item for item in examples if item.fold == fold]
        positives = sum(item.label for item in selected)
        negatives = len(selected) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"base Gate-1 fold {fold} does not contain both label classes")
        expected_summary[str(fold)] = {
            "examples": len(selected),
            "homology_clusters": len({item.cluster_id for item in selected}),
            "negatives": negatives,
            "positives": positives,
            "sequences": len({item.sequence_id for item in selected}),
        }
    if document.get("folds") != expected_summary:
        raise ValueError("base Gate-1 fold summary differs from the supplied OOF metadata")
    return document


def _verify_apex_manifest(
    predictions_path: Path,
    reconciliation_path: Path,
    manifest_path: Path,
    *,
    config: ApexOofConfig,
) -> dict[str, object]:
    document = _read_json_object(manifest_path, name="APEX member manifest")
    if document.get("format_version") != 1 or document.get("mode") != "apex_member_introspection":
        raise ValueError("APEX manifest is not a member-introspection manifest")
    if config.expected_member_count != 8 or document.get("member_count") != 8:
        raise ValueError("APEX OOF requires the frozen eight-member ensemble")
    source_commit = document.get("source_commit")
    if not isinstance(source_commit, str) or _GIT_SHA1.fullmatch(source_commit) is None:
        raise ValueError("APEX manifest source_commit is not a full lowercase Git SHA-1")
    endpoints = document.get("endpoints")
    configured_endpoints = [endpoint for target in config.targets for endpoint in target.endpoints]
    expected_labels = [label for label, _ in _APEX_ENDPOINTS]
    expected_columns = [column for _, column in _APEX_ENDPOINTS]
    if endpoints != expected_labels or configured_endpoints != expected_columns:
        raise ValueError("APEX manifest endpoint inventory differs from the frozen config")
    outputs = document.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("APEX manifest has no output table")
    if outputs.get("apex_member_predictions.csv") != _sha256(predictions_path):
        raise ValueError("APEX manifest does not match the supplied member predictions")
    if outputs.get("apex_member_reconciliation.csv") != _sha256(reconciliation_path):
        raise ValueError("APEX manifest does not match the supplied reconciliation CSV")
    reconciliation = document.get("reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("all_passed") is not True:
        raise ValueError("APEX manifest does not declare a passing member reconciliation")
    return document


def _read_apex_predictions(
    path: Path,
    *,
    manifest: Mapping[str, object],
    config: ApexOofConfig,
) -> ApexPredictions:
    manifest_members = manifest.get("members")
    if (
        not isinstance(manifest_members, list)
        or len(manifest_members) != config.expected_member_count
    ):
        raise ValueError("APEX manifest member count does not match config")
    member_hashes: dict[str, str] = {}
    member_order: list[str] = []
    for item in manifest_members:
        if not isinstance(item, dict):
            raise ValueError("APEX manifest member entry is invalid")
        member_id = item.get("member_id")
        checkpoint_hash = item.get("checkpoint_sha256")
        if (
            not isinstance(member_id, str)
            or not member_id
            or not isinstance(checkpoint_hash, str)
            or _SHA256.fullmatch(checkpoint_hash) is None
            or member_id in member_hashes
        ):
            raise ValueError("APEX manifest contains an invalid/duplicate member")
        member_hashes[member_id] = checkpoint_hash
        member_order.append(member_id)
    endpoint_columns = tuple(endpoint for target in config.targets for endpoint in target.endpoints)
    expected_schema = (
        "sequence_id",
        "sequence",
        "model_family",
        "model_version",
        "member_id",
        "member_checkpoint_sha256",
        "apex_member_mean_mic_um",
        *endpoint_columns,
    )
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    sequences: dict[str, str] = {}
    row_count = 0
    source_commit = str(manifest["source_commit"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_schema):
            raise ValueError("APEX prediction CSV does not have the frozen member schema")
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            sequence = canonicalize_sequence(row["sequence"])
            sequence_id = row["sequence_id"]
            if sequence_id != canonical_sequence_id(sequence):
                raise ValueError(f"APEX row {row_number} sequence_id mismatch")
            if row["model_family"] != "apex_pathogen_member":
                raise ValueError(f"APEX row {row_number} model family mismatch")
            if row["model_version"] != source_commit:
                raise ValueError(f"APEX row {row_number} model version mismatch")
            member_id = row["member_id"]
            if member_hashes.get(member_id) != row["member_checkpoint_sha256"]:
                raise ValueError(f"APEX row {row_number} member identity mismatch")
            if member_id in values[sequence_id]:
                raise ValueError(f"APEX row {row_number} duplicates sequence/member")
            endpoint_values = {endpoint: float(row[endpoint]) for endpoint in endpoint_columns}
            if any(not math.isfinite(value) or value <= 0 for value in endpoint_values.values()):
                raise ValueError(f"APEX row {row_number} has non-positive/non-finite MIC")
            broad_mean = float(row["apex_member_mean_mic_um"])
            expected_broad_mean = math.fsum(endpoint_values.values()) / len(endpoint_values)
            if (
                not math.isfinite(broad_mean)
                or broad_mean <= 0
                or broad_mean != expected_broad_mean
            ):
                raise ValueError(f"APEX row {row_number} has an invalid broad MIC mean")
            previous = sequences.setdefault(sequence_id, sequence)
            if previous != sequence:
                raise ValueError(f"APEX row {row_number} sequence identity collision")
            values[sequence_id][member_id] = endpoint_values
    if not values:
        raise ValueError("APEX member prediction CSV is empty")
    expected_members = set(member_order)
    for sequence_id, members in values.items():
        if set(members) != expected_members:
            raise ValueError(f"APEX sequence {sequence_id!r} does not have every member")
    if manifest.get("n_sequences") != len(values):
        raise ValueError("APEX manifest sequence count does not match predictions")
    if row_count != len(values) * len(member_order):
        raise ValueError("APEX member prediction row count is not rectangular")
    return ApexPredictions(
        members=tuple(member_order),
        member_hashes=member_hashes,
        values=dict(values),
        sequences=sequences,
        row_count=row_count,
    )


def _verify_apex_reconciliation(
    path: Path,
    *,
    manifest: Mapping[str, object],
    predictions: ApexPredictions,
) -> None:
    expected_schema = (
        "sequence_id",
        "sequence",
        "endpoint",
        "member_count",
        "introspected_mean_mic_um",
        "fidelity_mean_mic_um",
        "absolute_error",
        "relative_error",
        "allowed_error",
        "passed",
    )
    expected_endpoints = {label for label, _ in _APEX_ENDPOINTS} | {"broad_mean"}
    endpoint_columns = dict(_APEX_ENDPOINTS)
    summary = manifest.get("reconciliation")
    assert isinstance(summary, dict)
    absolute_tolerance = summary.get("absolute_tolerance")
    relative_tolerance = summary.get("relative_tolerance")
    if (
        not isinstance(absolute_tolerance, int | float)
        or isinstance(absolute_tolerance, bool)
        or not isinstance(relative_tolerance, int | float)
        or isinstance(relative_tolerance, bool)
        or not math.isfinite(float(absolute_tolerance))
        or not math.isfinite(float(relative_tolerance))
        or float(absolute_tolerance) < 0
        or float(relative_tolerance) < 0
    ):
        raise ValueError("APEX reconciliation manifest has invalid tolerances")
    seen: set[tuple[str, str]] = set()
    max_absolute_error = 0.0
    max_relative_error = 0.0
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_schema):
            raise ValueError("APEX reconciliation CSV does not have the frozen schema")
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            sequence = canonicalize_sequence(row["sequence"])
            sequence_id = row["sequence_id"]
            if (
                sequence_id != canonical_sequence_id(sequence)
                or predictions.sequences.get(sequence_id) != sequence
            ):
                raise ValueError(f"APEX reconciliation row {row_number} sequence mismatch")
            endpoint = row["endpoint"]
            key = (sequence_id, endpoint)
            if endpoint not in expected_endpoints or key in seen:
                raise ValueError(f"APEX reconciliation row {row_number} endpoint mismatch")
            seen.add(key)
            try:
                member_count = int(row["member_count"])
                introspected = float(row["introspected_mean_mic_um"])
                fidelity = float(row["fidelity_mean_mic_um"])
                absolute_error = float(row["absolute_error"])
                relative_error = float(row["relative_error"])
                allowed_error = float(row["allowed_error"])
            except ValueError as error:
                raise ValueError(f"APEX reconciliation row {row_number} is not numeric") from error
            numeric = (introspected, fidelity, absolute_error, relative_error, allowed_error)
            if endpoint == "broad_mean":
                member_broad_means = [
                    math.fsum(predictions.values[sequence_id][member].values())
                    / len(_APEX_ENDPOINTS)
                    for member in predictions.members
                ]
                expected_introspected = math.fsum(member_broad_means) / len(member_broad_means)
            else:
                column = endpoint_columns[endpoint]
                member_values = [
                    predictions.values[sequence_id][member][column]
                    for member in predictions.members
                ]
                expected_introspected = math.fsum(member_values) / len(member_values)
            expected_absolute_error = abs(introspected - fidelity)
            expected_relative_error = expected_absolute_error / abs(fidelity)
            expected_allowed_error = float(absolute_tolerance) + float(relative_tolerance) * abs(
                fidelity
            )
            if (
                member_count != len(predictions.members)
                or not all(math.isfinite(value) for value in numeric)
                or introspected <= 0
                or fidelity <= 0
                or absolute_error < 0
                or relative_error < 0
                or allowed_error < 0
                or not math.isclose(
                    introspected,
                    expected_introspected,
                    rel_tol=1e-15,
                    abs_tol=1e-12,
                )
                or absolute_error != expected_absolute_error
                or relative_error != expected_relative_error
                or allowed_error != expected_allowed_error
                or absolute_error > allowed_error
                or row["passed"] != "true"
            ):
                raise ValueError(f"APEX reconciliation row {row_number} did not pass")
            max_absolute_error = max(max_absolute_error, absolute_error)
            max_relative_error = max(max_relative_error, relative_error)
    expected_pairs = {
        (sequence_id, endpoint)
        for sequence_id in predictions.sequences
        for endpoint in expected_endpoints
    }
    if seen != expected_pairs:
        raise ValueError("APEX reconciliation does not cover every sequence/endpoint pair")
    if (
        summary.get("all_passed") is not True
        or summary.get("comparison_count") != row_count
        or summary.get("max_absolute_error") != max_absolute_error
        or summary.get("max_relative_error") != max_relative_error
    ):
        raise ValueError("APEX reconciliation summary differs from its CSV")


def _matching_targets(strain: str, rules: Sequence[TargetRule]) -> tuple[TargetRule, ...]:
    return tuple(rule for rule in rules if rule.matches(strain))


def _support_examples(
    examples: Sequence[Example],
    *,
    rules: Sequence[TargetRule],
    apex: Mapping[str, Mapping[str, Mapping[str, float]]],
    members: Sequence[str],
    activity_threshold_um: float,
) -> tuple[tuple[SupportedExample, ...], Counter[str]]:
    supported: list[SupportedExample] = []
    exclusions: Counter[str] = Counter()
    threshold_log = math.log10(activity_threshold_um)
    for example in examples:
        matches = _matching_targets(example.strain, rules)
        if not matches:
            exclusions["unsupported_target"] += 1
            continue
        if len(matches) > 1:
            # APEX has no defensible way to assign one group-level MIC label to
            # one of several named species. Quarantine rather than guessing.
            exclusions["ambiguous_multiple_supported_targets"] += 1
            continue
        target = matches[0]
        if example.sequence_id not in apex:
            raise ValueError(
                f"APEX predictions are missing base OOF sequence {example.sequence_id!r}"
            )
        signals = {
            member: threshold_log
            - float(
                np.mean(
                    [math.log10(apex[example.sequence_id][member][e]) for e in target.endpoints]
                )
            )
            for member in members
        }
        supported.append(SupportedExample(example=example, target=target, signals=signals))
    if not supported:
        raise ValueError("no base OOF example maps to a configured APEX target")
    return tuple(supported), exclusions


def _verify_supported_fold_contract(
    supported: Sequence[SupportedExample],
    *,
    configured_folds: int,
) -> None:
    expected_folds = set(range(configured_folds))
    observed_folds = {item.example.fold for item in supported}
    if observed_folds != expected_folds:
        raise ValueError("APEX-supported examples do not cover every configured Gate-1 fold")
    for heldout_fold in sorted(expected_folds):
        heldout_labels = {
            item.example.label for item in supported if item.example.fold == heldout_fold
        }
        if heldout_labels != {0, 1}:
            raise ValueError(
                f"APEX-supported fold {heldout_fold} does not contain both label classes"
            )
        training_labels = {
            item.example.label for item in supported if item.example.fold != heldout_fold
        }
        if training_labels != {0, 1}:
            raise ValueError(
                f"fold {heldout_fold} APEX calibration training complement does not contain "
                "both label classes"
            )


def _sequence_coverage_receipt(
    examples: Sequence[Example],
    *,
    predictions: ApexPredictions,
    supported: Sequence[SupportedExample],
    exclusions: Counter[str],
) -> dict[str, object]:
    base_ids = {item.sequence_id for item in examples}
    apex_ids = set(predictions.sequences)
    supported_ids = {item.example.sequence_id for item in supported}
    missing_ids = sorted(base_ids - apex_ids)
    extra_ids = sorted(apex_ids - base_ids)
    covered_ids = base_ids & apex_ids
    if missing_ids:
        raise ValueError(
            f"APEX predictions do not cover {len(missing_ids)} base Gate-1 sequence(s)"
        )
    supported_positives = sum(item.example.label for item in supported)
    return {
        "schema_version": 1,
        "set_digest_encoding": _SET_DIGEST_ENCODING,
        "assignment_digest_encoding": _ASSIGNMENT_DIGEST_ENCODING,
        "base": {
            "examples": len(examples),
            "positive_examples": sum(item.label for item in examples),
            "negative_examples": sum(1 - item.label for item in examples),
            "unique_sequences": len(base_ids),
            "homology_clusters": len({item.cluster_id for item in examples}),
            "sequence_ids_sha256": _sequence_set_digest(base_ids),
            "example_label_fold_cluster_sha256": _assignment_digest(examples),
        },
        "apex": {
            "rows": predictions.row_count,
            "members": len(predictions.members),
            "unique_sequences": len(apex_ids),
            "sequence_ids_sha256": _sequence_set_digest(apex_ids),
        },
        "coverage": {
            "all_base_sequences_covered": not missing_ids,
            "covered_base_sequences": len(covered_ids),
            "covered_sequence_ids_sha256": _sequence_set_digest(covered_ids),
            "missing_sequences": len(missing_ids),
            "missing_sequence_ids": missing_ids,
            "missing_sequence_ids_sha256": _sequence_set_digest(missing_ids),
            "extra_sequences": len(extra_ids),
            "extra_sequence_ids": extra_ids,
            "extra_sequence_ids_sha256": _sequence_set_digest(extra_ids),
        },
        "supported": {
            "examples": len(supported),
            "positive_examples": supported_positives,
            "negative_examples": len(supported) - supported_positives,
            "unique_sequences": len(supported_ids),
            "homology_clusters": len({item.example.cluster_id for item in supported}),
            "sequence_ids_sha256": _sequence_set_digest(supported_ids),
            "example_ids_sha256": _sequence_set_digest(
                {item.example.example_id for item in supported}
            ),
            "example_label_fold_cluster_sha256": _assignment_digest(
                [item.example for item in supported]
            ),
        },
        "excluded_examples": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
    }


def _sigmoid_scalar(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _fit_calibrator(
    rows: Sequence[SupportedExample],
    *,
    heldout_fold: int,
    member_id: str,
    config: ApexOofConfig,
) -> Calibrator:
    training = [row for row in rows if row.example.fold != heldout_fold]
    if not training:
        raise ValueError(f"fold {heldout_fold} leaves no calibration training examples")
    signal = np.asarray([row.signals[member_id] for row in training], dtype=np.float64)
    labels = np.asarray([row.example.label for row in training], dtype=np.float64)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError(
            f"fold {heldout_fold} member {member_id!r} calibration training data must "
            "contain both label classes"
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
        raise ValueError(
            f"fold {heldout_fold} member {member_id!r} has a degenerate calibration prior"
        )
    coefficient = np.asarray([math.log(prior / (1.0 - prior)), 0.0], dtype=np.float64)
    iterations = 0
    converged = False
    design = np.column_stack((np.ones(len(labels)), standardized))
    penalty = np.asarray([0.0, config.calibration_l2], dtype=np.float64)
    for iteration in range(1, config.calibration_max_iterations + 1):
        logits = design @ coefficient
        probabilities = np.asarray([_sigmoid_scalar(float(value)) for value in logits])
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = design.T @ (probabilities - labels) / len(labels) + penalty * coefficient
        hessian = (design.T * variance) @ design / len(labels)
        hessian.flat[:: hessian.shape[0] + 1] += penalty + 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise ValueError(f"fold {heldout_fold} member {member_id!r} produced a non-finite step")
        coefficient -= step
        iterations = iteration
        if not np.all(np.isfinite(coefficient)):
            raise ValueError(
                f"fold {heldout_fold} member {member_id!r} produced non-finite coefficients"
            )
        if float(np.max(np.abs(step))) <= config.calibration_tolerance:
            converged = True
            break
    if not converged:
        raise ValueError(
            f"fold {heldout_fold} member {member_id!r} calibration did not converge "
            f"within {config.calibration_max_iterations} iterations"
        )
    return Calibrator(
        fold=heldout_fold,
        member_id=member_id,
        training_examples=len(training),
        positives=int(np.sum(labels)),
        iterations=iterations,
        converged=converged,
        signal_mean=signal_mean,
        signal_scale=signal_scale,
        intercept=float(coefficient[0]),
        slope=float(coefficient[1]),
    )


def _prediction_rows(
    supported: Sequence[SupportedExample],
    members: Sequence[str],
    config: ApexOofConfig,
) -> tuple[list[dict[str, object]], list[Calibrator]]:
    calibrators: list[Calibrator] = []
    by_key: dict[tuple[int, str], Calibrator] = {}
    folds = sorted({row.example.fold for row in supported})
    for fold in folds:
        for member in members:
            fitted = _fit_calibrator(
                supported,
                heldout_fold=fold,
                member_id=member,
                config=config,
            )
            calibrators.append(fitted)
            by_key[(fold, member)] = fitted
    rows: list[dict[str, object]] = []
    for item in sorted(supported, key=lambda value: value.example.example_id):
        example = item.example
        member_probabilities: list[float] = []
        for member in members:
            calibrator = by_key[(example.fold, member)]
            probability = calibrator.predict(item.signals[member])
            member_probabilities.append(probability)
            rows.append(
                {
                    **asdict(example),
                    "model": f"apex_member::{member}",
                    "member_id": member,
                    "target_rule": item.target.name,
                    "apex_endpoints": ";".join(item.target.endpoints),
                    "activity_signal": item.signals[member],
                    "probability": probability,
                    "member_probability_std": "",
                }
            )
        rows.append(
            {
                **asdict(example),
                "model": "apex_member_mean",
                "member_id": "",
                "target_rule": item.target.name,
                "apex_endpoints": ";".join(item.target.endpoints),
                "activity_signal": float(np.mean(list(item.signals.values()))),
                "probability": float(np.mean(member_probabilities)),
                "member_probability_std": float(np.std(member_probabilities)),
            }
        )
    return rows, calibrators


def _metrics(rows: Sequence[Mapping[str, object]], config: ApexOofConfig) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    output: dict[str, object] = {}
    for model, model_rows in sorted(grouped.items()):
        metric = lambda selected: binary_metrics(  # noqa: E731
            [int(row["label"]) for row in selected],
            [float(row["probability"]) for row in selected],
            calibration_bins=config.calibration_bins,
        )
        output[model] = {
            "overall": metric(model_rows),
            "by_fold": {
                str(fold): metric([row for row in model_rows if int(row["fold"]) == fold])
                for fold in sorted({int(row["fold"]) for row in model_rows})
            },
            "by_target": {
                target: metric([row for row in model_rows if row["target_rule"] == target])
                for target in sorted({str(row["target_rule"]) for row in model_rows})
            },
            "by_max_train_identity": {
                f"[{left:.2f},{right:.2f})": metric(
                    [row for row in model_rows if left <= float(row["max_train_identity"]) < right]
                )
                for left, right in _SIMILARITY_STRATA
                if any(left <= float(row["max_train_identity"]) < right for row in model_rows)
            },
        }
    ensemble = grouped["apex_member_mean"]
    by_cluster: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in ensemble:
        by_cluster[str(row["cluster_id"])].append(row)
    bootstrap: dict[str, list[float]] = {
        name: [] for name in ("roc_auc", "average_precision", "brier", "log_loss")
    }
    clusters = tuple(sorted(by_cluster))
    generator = np.random.default_rng(config.seed)
    for _ in range(config.bootstrap_replicates):
        selected = [
            row
            for cluster in generator.choice(clusters, size=len(clusters), replace=True)
            for row in by_cluster[str(cluster)]
        ]
        values = binary_metrics(
            [int(row["label"]) for row in selected],
            [float(row["probability"]) for row in selected],
            calibration_bins=config.calibration_bins,
        )
        for name in bootstrap:
            value = values[name]
            if value is not None:
                bootstrap[name].append(float(value))
    output["apex_member_mean"]["homology_cluster_bootstrap_95ci"] = {  # type: ignore[index]
        name: {
            "lower": None if not values else float(np.quantile(values, 0.025)),
            "upper": None if not values else float(np.quantile(values, 0.975)),
            "successful_replicates": len(values),
        }
        for name, values in bootstrap.items()
    }
    member_models = [name for name in grouped if name.startswith("apex_member::")]
    aligned = {
        name: {str(row["example_id"]): float(row["probability"]) for row in grouped[name]}
        for name in member_models
    }
    example_ids = sorted(next(iter(aligned.values())))
    matrix = np.asarray(
        [[aligned[name][example] for example in example_ids] for name in member_models]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.corrcoef(matrix)
    if not np.all(np.isfinite(correlations)):
        raise ValueError("APEX member probability correlations are non-finite")
    off_diagonal = correlations[np.triu_indices(len(member_models), 1)]
    output["model_diversity"] = {
        "members": member_models,
        "pairwise_probability_correlation_minimum": float(np.min(off_diagonal)),
        "pairwise_probability_correlation_median": float(np.median(off_diagonal)),
        "pairwise_probability_correlation_maximum": float(np.max(off_diagonal)),
    }
    return output


def _format(value: object) -> object:
    return format(value, ".12g") if isinstance(value, float) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row[field]) for field in fields})


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _json_ready(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def run_apex_oof(
    *,
    base_oof_path: str | Path,
    base_manifest_path: str | Path,
    base_folds_path: str | Path,
    base_config_path: str | Path,
    apex_predictions_path: str | Path,
    apex_manifest_path: str | Path,
    apex_reconciliation_path: str | Path,
    config_path: str | Path,
    code_manifest_path: str | Path,
    frozen_input_manifest_path: str | Path,
    git_commit: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Fit fold-local APEX calibration and write deterministic OOF evidence."""

    base_oof = Path(base_oof_path).resolve()
    base_manifest = Path(base_manifest_path).resolve()
    base_folds = Path(base_folds_path).resolve()
    base_config = Path(base_config_path).resolve()
    apex_predictions = Path(apex_predictions_path).resolve()
    apex_manifest = Path(apex_manifest_path).resolve()
    apex_reconciliation = Path(apex_reconciliation_path).resolve()
    code_manifest = Path(code_manifest_path).resolve()
    frozen_input_manifest = Path(frozen_input_manifest_path).resolve()
    if _GIT_SHA1.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase SHA-1 commit ID")
    config = load_config(config_path)
    base_gate_config = Gate1Config.from_toml(base_config)
    if base_gate_config.homology_identity_threshold != 0.8:
        raise ValueError("APEX OOF requires the frozen Gate-1 0.80 homology threshold")
    if base_gate_config.similarity_bin_edges != (0.0, 0.4, 0.6, 0.8):
        raise ValueError("APEX OOF requires the frozen Gate-1 similarity strata")
    if config.metadata_model != "descriptor_logistic" or config.expected_member_count != 8:
        raise ValueError("APEX OOF requires the frozen descriptor metadata and eight members")

    code_entries = _verify_sha256_manifest(code_manifest, name="code manifest")
    frozen_entries = _verify_sha256_manifest(
        frozen_input_manifest,
        name="frozen input manifest",
    )
    for source, entry, name in (
        (
            Path(__file__).resolve(),
            "src/amp_challenge/benchmarks/apex_oof.py",
            "APEX OOF implementation",
        ),
        (base_config, "configs/benchmarks/oracle_gate1.toml", "base Gate-1 config"),
        (config.path, "configs/models/apex_oof.toml", "APEX OOF config"),
    ):
        _require_manifest_digest(
            code_entries,
            source,
            entry=entry,
            manifest_name="code manifest",
            artifact_name=name,
        )
    for source, entry, name in (
        (base_oof, "base/oof_predictions.csv", "base OOF predictions"),
        (base_manifest, "base/manifest.json", "base OOF manifest"),
        (base_folds, "base/folds.json", "base folds"),
        (
            apex_predictions,
            "apex/apex_member_predictions.csv",
            "APEX member predictions",
        ),
        (apex_manifest, "apex/apex_member_run_manifest.json", "APEX member manifest"),
        (
            apex_reconciliation,
            "apex/apex_member_reconciliation.csv",
            "APEX member reconciliation",
        ),
    ):
        _require_manifest_digest(
            frozen_entries,
            source,
            entry=entry,
            manifest_name="frozen input manifest",
            artifact_name=name,
        )

    base_document = _verify_base_manifest(base_oof, base_folds, base_config, base_manifest)
    label_summary = base_document.get("label_summary")
    if (
        not isinstance(label_summary, dict)
        or label_summary.get("activity_threshold_um") != base_gate_config.activity_threshold_um
        or base_gate_config.activity_threshold_um != config.activity_threshold_um
    ):
        raise ValueError("APEX calibration threshold does not match the base OOF label contract")
    examples = _read_examples(base_oof, config.metadata_model)
    _verify_base_folds(
        base_folds,
        examples=examples,
        config=base_gate_config,
        manifest=base_document,
    )
    positives = sum(item.label for item in examples)
    if (
        label_summary.get("included_strain_level_examples") != len(examples)
        or label_summary.get("positive_examples") != positives
        or label_summary.get("negative_examples") != len(examples) - positives
    ):
        raise ValueError("base Gate-1 label summary differs from its OOF metadata")

    apex_document = _verify_apex_manifest(
        apex_predictions,
        apex_reconciliation,
        apex_manifest,
        config=config,
    )
    apex_input = _read_apex_predictions(
        apex_predictions,
        manifest=apex_document,
        config=config,
    )
    _verify_apex_reconciliation(
        apex_reconciliation,
        manifest=apex_document,
        predictions=apex_input,
    )
    missing_base_sequences = {item.sequence_id for item in examples} - set(apex_input.sequences)
    if missing_base_sequences:
        raise ValueError(
            "APEX predictions do not cover every base Gate-1 sequence: "
            f"{sorted(missing_base_sequences)[:3]}"
        )
    supported, exclusions = _support_examples(
        examples,
        rules=config.targets,
        apex=apex_input.values,
        members=apex_input.members,
        activity_threshold_um=config.activity_threshold_um,
    )
    _verify_supported_fold_contract(
        supported,
        configured_folds=base_gate_config.folds,
    )
    coverage_receipt = _sequence_coverage_receipt(
        examples,
        predictions=apex_input,
        supported=supported,
        exclusions=exclusions,
    )
    prediction_rows, calibrators = _prediction_rows(supported, apex_input.members, config)
    metrics = _metrics(prediction_rows, config)

    requested_output = Path(output_dir)
    if requested_output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite symlinked APEX OOF output: {requested_output}"
        )
    final_output = requested_output.resolve()
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite existing APEX OOF output: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{final_output.name}.staging-",
        dir=final_output.parent,
    ) as staging_name:
        output = Path(staging_name)
        predictions_out = output / "apex_oof_predictions.csv"
        calibrators_out = output / "calibrators.json"
        metrics_out = output / "metrics.json"
        coverage_out = output / "sequence_coverage_receipt.json"
        manifest_out = output / "manifest.json"
        fields = (
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
        prediction_rows.sort(key=lambda row: (str(row["model"]), str(row["example_id"])))
        _write_csv(predictions_out, prediction_rows, fields)
        _write_json(calibrators_out, [asdict(item) for item in calibrators])
        _write_json(metrics_out, metrics)
        _write_json(coverage_out, coverage_receipt)
        supported_positives = sum(item.example.label for item in supported)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "benchmark": "apex_homology_oof_calibration",
            "base_oof_sha256": _sha256(base_oof),
            "base_manifest_sha256": _sha256(base_manifest),
            "base_folds_sha256": _sha256(base_folds),
            "base_config_sha256": _sha256(base_config),
            "apex_member_predictions_sha256": _sha256(apex_predictions),
            "apex_member_manifest_sha256": _sha256(apex_manifest),
            "apex_member_reconciliation_sha256": _sha256(apex_reconciliation),
            "config_sha256": _sha256(config.path),
            "fold_policy": _CALIBRATION_FOLD_POLICY,
            "calibration_policy": "per-member logistic calibration of log10 activity signal",
            "upstream_training_independence": ("not established; OOF applies to calibration only"),
            "base_contract": {
                "benchmark": base_document["benchmark"],
                "normalized_parser_id": base_document["normalized_parser_id"],
                "fold_policy": base_document["fold_policy"],
                "fold_assignment_policy": base_document["fold_assignment_policy"],
                "identity_threshold": base_gate_config.homology_identity_threshold,
                "configured_folds": base_gate_config.folds,
                "observed_folds": sorted({item.fold for item in examples}),
                "input_examples": len(examples),
                "positive_examples": positives,
                "negative_examples": len(examples) - positives,
                "unique_sequences": len({item.sequence_id for item in examples}),
                "homology_clusters": len({item.cluster_id for item in examples}),
                "supported_examples": len(supported),
                "supported_positive_examples": supported_positives,
                "supported_negative_examples": len(supported) - supported_positives,
                "supported_unique_sequences": len({item.example.sequence_id for item in supported}),
                "supported_homology_clusters": len({item.example.cluster_id for item in supported}),
            },
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
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "input_examples": len(examples),
            "supported_examples": len(supported),
            "excluded_examples": sum(exclusions.values()),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "members": list(apex_input.members),
            "member_checkpoints": [
                {"member_id": member, "checkpoint_sha256": apex_input.member_hashes[member]}
                for member in apex_input.members
            ],
            "outputs": {
                "apex_oof_predictions.csv": _sha256(predictions_out),
                "calibrators.json": _sha256(calibrators_out),
                "metrics.json": _sha256(metrics_out),
                "sequence_coverage_receipt.json": _sha256(coverage_out),
            },
        }
        _write_json(manifest_out, manifest)
        output.rename(final_output)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-folds", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--apex-predictions", type=Path, required=True)
    parser.add_argument("--apex-manifest", type=Path, required=True)
    parser.add_argument("--apex-reconciliation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models/apex_oof.toml"))
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--frozen-input-manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_apex_oof(
        base_oof_path=args.base_oof,
        base_manifest_path=args.base_manifest,
        base_folds_path=args.base_folds,
        base_config_path=args.base_config,
        apex_predictions_path=args.apex_predictions,
        apex_manifest_path=args.apex_manifest,
        apex_reconciliation_path=args.apex_reconciliation,
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

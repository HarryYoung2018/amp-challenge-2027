"""Independently verify sealed sequential-v2 finalization publications.

This audit implementation deliberately imports none of the sequential-v2
producer, selector, metric, reduction, seal, or supervisor modules.  It
authenticates one or two finalization closures under paired campaign/stage roots,
derives the 927 direct finalization predecessors from the independently read
global indexes, reconstructs all 220 redacted rotation metrics from the
authenticated outcome/prediction/reveal/view/selection leaves, and then
reimplements every reduction through the seven ``finalize/global`` payloads.

The evidence boundary is intentionally precise: the verifier recomputes the
final metrics and outer-mean selections, but it does not refit the 220 update
models or independently rerun the earlier pool acquisition policies.  Those
earlier decisions are authenticated through their externally pinned global
barriers and exact leaf/index bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
VERIFICATION_ARTIFACT = "sequential_v2_finalize_v1_finalize_only_partial_independent_verification"
FINALIZE_ARTIFACT = "sequential_v2_finalize_campaign_v1"
FINALIZE_SUMMARY_ARTIFACT = "sequential_v2_finalize_summary_v1"
PROMOTION_DECISION_ARTIFACT = "sequential_v2_promotion_decision_v1"

FINALIZE_PAYLOAD_PATHS = (
    "outer-fold-units.jsonl",
    "paired-comparisons.jsonl",
    "policy-point-estimates.jsonl",
    "policy-rotation-metrics.jsonl",
    "promotion-decision.json",
    "rotation-metrics.jsonl",
    "summary.json",
)

TARGET_GRAMS = (
    ("acinetobacter_baumannii", "negative"),
    ("enterococcus_faecalis", "positive"),
    ("enterococcus_faecium", "positive"),
    ("escherichia_coli", "negative"),
    ("klebsiella_pneumoniae", "negative"),
    ("pseudomonas_aeruginosa", "negative"),
    ("staphylococcus_aureus", "positive"),
)
TARGETS = tuple(target for target, _gram in TARGET_GRAMS)
TARGET_GRAM = dict(TARGET_GRAMS)
OBJECTIVES = (
    "broad_spectrum_activity",
    "gram_positive_activity",
    "gram_negative_activity",
)
POLICY_ORDER = (
    "no_query",
    "mean",
    "random",
    "mean_nine_diversity_one",
    "mean_nine_novelty_one",
    "mixed_eight_diversity_one_novelty_one",
    "full_acquisition_fold_ceiling",
)
RANDOM_SEEDS = (17, 42, 91, 137, 271)
GUARDED_POLICIES = frozenset(
    {
        "mean_nine_diversity_one",
        "mean_nine_novelty_one",
        "mixed_eight_diversity_one_novelty_one",
    }
)
EXPECTED_CONTEXTS_BY_FOLD = (546, 486, 485, 487, 488)
EXPECTED_SUPPORT_BY_FOLD = (202, 126, 112, 97, 113)
AGGREGATE_METRIC_NAMES = (
    "macro_brier",
    "macro_negative_log_likelihood",
    "macro_roc_auc",
    "macro_average_precision",
    "macro_ece_10",
    "next_round_outer_top10_mean_reward",
    "queried_outcome_mean_reward",
    "queried_unique_diversity_components",
    "queried_mean_pairwise_feature_cosine_distance",
    "revealed_context_count",
)
COMPARISON_SPECS = (
    ("mixed_minus_mean_macro_brier", "mean", "macro_brier", "lower"),
    (
        "mixed_minus_mean_macro_negative_log_likelihood",
        "mean",
        "macro_negative_log_likelihood",
        "lower",
    ),
    (
        "mixed_minus_mean_next_round_outer_top10_mean_reward",
        "mean",
        "next_round_outer_top10_mean_reward",
        "higher",
    ),
    (
        "mixed_minus_mean_queried_mean_pairwise_feature_cosine_distance",
        "mean",
        "queried_mean_pairwise_feature_cosine_distance",
        "higher",
    ),
    (
        "mixed_minus_mean_queried_unique_diversity_components",
        "mean",
        "queried_unique_diversity_components",
        "higher",
    ),
    ("mixed_minus_no_query_macro_brier", "no_query", "macro_brier", "lower"),
    ("mixed_minus_random_macro_brier", "random", "macro_brier", "lower"),
)
COMPARISON_ORDER = tuple(item[0] for item in COMPARISON_SPECS)
PROMOTION_ITEM_ORDER = (
    "brier_vs_mean",
    "nll_vs_mean",
    "outer_reward_vs_mean",
    "queried_diversity_vs_mean",
    "brier_vs_no_query_and_random",
    "guarded_selection_validity",
)

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_905
BOOTSTRAP_INDICES_SHA256 = "5c3c5db8a262e5b29cd65b5334049e81f2659d47817d8815cf0720ed5114a2b3"

_ACCEPTED_GATE1_PRODUCER_JOB_ID = 223248
_ACCEPTED_GATE1_AUDIT_JOB_ID = 223250
_ACCEPTED_GATE1_PUBLICATION_TOP_SHA256 = (
    "4e259cfb43033c069598d42fd54fec49a67ba55bfbb5c1e77eeef4887815fe2f"
)
_ACCEPTED_GATE1_SEMANTIC_TOP_SHA256 = (
    "556e06fd2b1af1e678de88008bc1b87434fb8cf1179f38c897de7ee2c9fd779d"
)
_ACCEPTED_GATE1_INDEPENDENT_RECEIPT_SHA256 = (
    "1f6a6ce811d3fbecdbbbe7463e6052dc7766130372926d29952c70be65743e81"
)
_ACCEPTED_GATE1_SOURCE_FILES = (
    (
        "CODE_SHA256SUMS",
        "f770a656c4f300855c129ebc044cc753c3bd119cb0476ef14eae277f5b104e72",
        "0444",
    ),
    (
        "FROZEN_INPUT_SHA256SUMS",
        "229e10ad276430136671cb7831f44e50c5a6d6a6f09ef3f16e58129771f53e60",
        "0444",
    ),
    ("SHA256SUMS", _ACCEPTED_GATE1_PUBLICATION_TOP_SHA256, "0400"),
    ("gate1/SHA256SUMS", _ACCEPTED_GATE1_SEMANTIC_TOP_SHA256, "0444"),
    (
        "gate1/context_audit.jsonl",
        "9886928e195543eb5725bcb1c1b35b9c3d29eec555e339b0be28a12028813900",
        "0444",
    ),
    (
        "gate1/examples.jsonl",
        "d3eecbf3014fd78cf7021818466893d292315b6e90e77cea85d4e1fbd5bec520",
        "0444",
    ),
    (
        "gate1/folds.json",
        "37e035d488634e50ad95dfefe01d235474867aeb96b8205b543d22c25fdc9988",
        "0444",
    ),
    (
        "gate1/manifest.json",
        "b1cf4a2ad396d26776a824d4ec3cc077ba45bf9d2d47e00f8f2f482053b1f78e",
        "0444",
    ),
    (
        "gate1/metrics.json",
        "e06b1893572309b2175fa4c9c1743af2fbb2d4006cdae9a7c452fcd4dba2563b",
        "0444",
    ),
    (
        "gate1/oof_predictions.csv",
        "11eea907a242606b78261a9b37107647ce393db743be95c146cbbe5caced137c",
        "0444",
    ),
    (
        "gate1/split_receipt.json",
        "63cf02d87cd4a9b9ec1afe9c40c2e0914fc75515a14a21e71f1e50dce0fd01a7",
        "0444",
    ),
)
_ACCEPTED_GATE1_SOURCE_PREDECESSORS = {
    "source/gate1-publication-top": _ACCEPTED_GATE1_PUBLICATION_TOP_SHA256,
    "source/gate1-semantic-top": _ACCEPTED_GATE1_SEMANTIC_TOP_SHA256,
    "source/gate1-independent-receipt": _ACCEPTED_GATE1_INDEPENDENT_RECEIPT_SHA256,
}
NLL_CLIP = 1e-15
PROBABILITY_CLIP = 1e-6
MAX_FILE_BYTES = 128 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_COMPONENT = re.compile(r"seqv2-div70:[0-9a-f]{64}\Z")
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_RECEIPT = "receipt.json"
_MANIFEST = "SHA256SUMS"


class VerificationError(ValueError):
    """Raised when an independent finalization check fails closed."""


@dataclass(frozen=True, slots=True)
class PublicationIdentity:
    git_commit: str
    code_manifest_sha256: str
    config_sha256: str
    lock_sha256: str

    def __post_init__(self) -> None:
        if type(self.git_commit) is not str or _GIT_COMMIT.fullmatch(self.git_commit) is None:
            raise VerificationError("git_commit must be forty lowercase hexadecimal characters")
        for label, value in (
            ("code manifest", self.code_manifest_sha256),
            ("config", self.config_sha256),
            ("lock", self.lock_sha256),
        ):
            _sha256(value, label=label)

    def metadata(self, *, phase: str, scope_id: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": phase,
            "scope_id": scope_id,
            "git_commit": self.git_commit,
            "code_manifest_sha256": self.code_manifest_sha256,
            "config_sha256": self.config_sha256,
            "lock_sha256": self.lock_sha256,
        }

    def document(self) -> dict[str, str]:
        return {
            "git_commit": self.git_commit,
            "code_manifest_sha256": self.code_manifest_sha256,
            "config_sha256": self.config_sha256,
            "lock_sha256": self.lock_sha256,
        }


@dataclass(frozen=True, slots=True)
class GlobalAuthorities:
    protocol: str
    stage: str
    prepare: str
    select: str
    reveal: str
    update: str
    outer_select: str

    def __post_init__(self) -> None:
        for name in (
            "protocol",
            "stage",
            "prepare",
            "select",
            "reveal",
            "update",
            "outer_select",
        ):
            _sha256(getattr(self, name), label=f"expected {name} global seal")

    def document(self) -> dict[str, str]:
        return {
            "protocol_seal_sha256": self.protocol,
            "stage_global_seal_sha256": self.stage,
            "prepare_global_seal_sha256": self.prepare,
            "selection_global_seal_sha256": self.select,
            "reveal_global_seal_sha256": self.reveal,
            "update_global_seal_sha256": self.update,
            "outer_selection_global_seal_sha256": self.outer_select,
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    payload: bytes
    sha256: str
    fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PhaseTreeSnapshot:
    root_path: Path
    root_fingerprint: tuple[int, ...]
    file_fingerprints: tuple[tuple[str, tuple[int, ...]], ...]
    marker_path: Path
    marker_sha256: str
    marker_fingerprint: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PhaseEvidence:
    artifact: str
    seal_sha256: str
    payloads: Mapping[str, bytes]
    payload_sha256: Mapping[str, str]
    predecessors: Mapping[str, str]
    metadata: Mapping[str, object]
    file_payloads: Mapping[str, bytes]
    tree_snapshot: _PhaseTreeSnapshot


@dataclass(frozen=True, slots=True)
class _CampaignResult:
    finalize: _PhaseEvidence
    reconstructed_payloads: tuple[tuple[str, bytes], ...]
    phase_evidence: tuple[_PhaseEvidence, ...]
    promising: bool


@dataclass(frozen=True, slots=True)
class VerificationExecution:
    receipt_dir: Path
    receipt_path: Path
    receipt_sha256: str
    finalize_global_seal_sha256: str
    publication_count: int
    promising_for_prospective_followup: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise VerificationError(f"{label} must be one lowercase SHA-256")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _reject_duplicates(label: str):
    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(f"{label} duplicates key {key!r}")
            result[key] = value
        return result

    return hook


def _reject_constant(label: str):
    def reject(value: str) -> object:
        raise VerificationError(f"{label} contains invalid constant {value}")

    return reject


def _strict_json(payload: bytes, *, label: str) -> object:
    _require(
        type(payload) is bytes and payload.endswith(b"\n") and b"\r" not in payload,
        f"{label} must be LF-terminated canonical JSON",
    )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicates(label),
            parse_constant=_reject_constant(label),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{label} is not strict UTF-8 JSON") from error
    _require(_canonical_json_bytes(value) == payload, f"{label} is not canonical compact JSON")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json(payload, label=label)
    _require(type(value) is dict, f"{label} must be a JSON object")
    return value  # type: ignore[return-value]


def _strict_jsonl(
    payload: bytes,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not payload:
        _require(allow_empty, f"{label} must be nonempty canonical JSON Lines")
        return ()
    _require(payload.endswith(b"\n") and b"\r" not in payload, f"{label} is not canonical JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        value = _strict_json(line, label=f"{label} row {index}")
        _require(type(value) is dict, f"{label} row {index} must be an object")
        rows.append(value)  # type: ignore[arg-type]
    _require(_canonical_jsonl_bytes(rows) == payload, f"{label} does not round-trip canonically")
    return tuple(rows)


def _exact_object(
    value: object, fields: set[str] | frozenset[str], *, label: str
) -> dict[str, Any]:
    _require(
        type(value) is dict and set(value) == fields and all(type(key) is str for key in value),
        f"{label} must contain exactly {sorted(fields)}",
    )
    return value  # type: ignore[return-value]


def _safe_relative_path(value: object, *, label: str) -> str:
    _require(type(value) is str and value != "", f"{label} must be nonempty text")
    raw = value  # type: ignore[assignment]
    path = PurePosixPath(raw)
    _require(
        not path.is_absolute()
        and path.as_posix() == raw
        and all(
            part not in {"", ".", ".."} and _SAFE_PATH_COMPONENT.fullmatch(part)
            for part in path.parts
        ),
        f"{label} is not a safe canonical relative path",
    )
    return raw


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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
    candidate = path.absolute()
    while True:
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise VerificationError(f"{label} traverses symbolic link {candidate}")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _snapshot_file(path: Path, *, label: str, max_bytes: int = MAX_FILE_BYTES) -> _FileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"cannot safely open {label}: {path}") from error
    payload = bytearray()
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o444
            and before.st_nlink == 1,
            f"{label} must be a one-link mode-0444 regular file",
        )
        _require(before.st_size <= max_bytes, f"{label} exceeds its byte bound")
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            digest.update(chunk)
            _require(len(payload) <= max_bytes, f"{label} grew beyond its byte bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    _require(
        len(payload) == before.st_size
        and _fingerprint(before) == _fingerprint(after) == _fingerprint(observed),
        f"{label} changed while read",
    )
    return _FileSnapshot(bytes(payload), digest.hexdigest(), _fingerprint(after))


def _parse_manifest(payload: bytes) -> dict[str, str]:
    _require(
        payload and payload.endswith(b"\n") and b"\r" not in payload,
        "manifest is not canonical LF text",
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise VerificationError("manifest is not ASCII") from error
    result: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"manifest line {number} is malformed")
        assert match is not None
        digest, raw_path = match.groups()
        path = _safe_relative_path(raw_path, label=f"manifest path at line {number}")
        _require(path != _MANIFEST and path not in result, f"manifest path {path!r} is invalid")
        result[path] = digest
    rebuilt = "".join(f"{result[path]}  {path}\n" for path in sorted(result)).encode("ascii")
    _require(rebuilt == payload, "manifest entries are not canonically ordered")
    return result


def _verify_phase(
    root: Path,
    *,
    expected_artifact: str,
    expected_payload_paths: Sequence[str],
    expected_seal_sha256: str | None = None,
    expected_predecessors: Mapping[str, str] | None = None,
    expected_metadata: Mapping[str, object] | None = None,
) -> _PhaseEvidence:
    requested = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_chain(requested, label="phase root")
    root_metadata = os.lstat(requested)
    _require(
        stat.S_ISDIR(root_metadata.st_mode)
        and not stat.S_ISLNK(root_metadata.st_mode)
        and stat.S_IMODE(root_metadata.st_mode) == 0o555,
        f"phase root must be a real mode-0555 directory: {requested}",
    )
    expected_files = tuple(sorted((*expected_payload_paths, _RECEIPT, _MANIFEST)))
    with os.scandir(requested) as iterator:
        entries = tuple(sorted(entry.name for entry in iterator))
    _require(entries == expected_files, f"phase inventory differs at {requested}")

    snapshots = {
        name: _snapshot_file(requested / name, label=f"phase file {name}")
        for name in expected_files
    }
    manifest_entries = _parse_manifest(snapshots[_MANIFEST].payload)
    _require(
        tuple(sorted((*manifest_entries, _MANIFEST))) == expected_files,
        "phase manifest does not bind the exact file inventory",
    )
    for name, digest in manifest_entries.items():
        _require(snapshots[name].sha256 == digest, f"phase checksum differs for {name}")

    receipt = _strict_json_object(snapshots[_RECEIPT].payload, label="phase receipt")
    receipt = _exact_object(
        receipt,
        {"artifact", "metadata", "payloads", "predecessor_seals", "schema_version", "status"},
        label="phase receipt",
    )
    _require(
        receipt["schema_version"] == 1
        and type(receipt["schema_version"]) is int
        and receipt["status"] == "sealed"
        and type(receipt["status"]) is str
        and receipt["artifact"] == expected_artifact
        and type(receipt["artifact"]) is str,
        "phase receipt identity/version/status changed",
    )
    payload_digests = _exact_object(
        receipt["payloads"], set(expected_payload_paths), label="phase receipt payload map"
    )
    predecessor_raw = receipt["predecessor_seals"]
    _require(type(predecessor_raw) is dict, "phase predecessor seals must be an object")
    predecessors: dict[str, str] = {}
    for raw_path, raw_digest in predecessor_raw.items():
        path = _safe_relative_path(raw_path, label="phase predecessor path")
        _require(path not in predecessors, "phase predecessor path is duplicated")
        predecessors[path] = _sha256(raw_digest, label=f"phase predecessor {path}")
    for path in expected_payload_paths:
        _require(
            _sha256(payload_digests[path], label=f"phase receipt payload {path}")
            == snapshots[path].sha256,
            f"phase receipt payload digest differs for {path}",
        )
    _require(
        manifest_entries[_RECEIPT] == snapshots[_RECEIPT].sha256,
        "manifest receipt digest differs",
    )
    if expected_seal_sha256 is not None:
        _require(
            snapshots[_MANIFEST].sha256
            == _sha256(expected_seal_sha256, label="expected phase seal"),
            "phase seal differs from external authority",
        )
    if expected_predecessors is not None:
        normalized_expected = {
            _safe_relative_path(path, label="expected predecessor path"): _sha256(
                digest, label=f"expected predecessor {path}"
            )
            for path, digest in expected_predecessors.items()
        }
        _require(predecessors == normalized_expected, "phase predecessor closure differs")
    metadata = receipt["metadata"]
    _require(type(metadata) is dict, "phase receipt metadata must be an object")
    if expected_metadata is not None:
        _require(
            _canonical_json_bytes(metadata) == _canonical_json_bytes(dict(expected_metadata)),
            "phase publication identity differs",
        )

    root_after = os.lstat(requested)
    with os.scandir(requested) as iterator:
        entries_after = tuple(sorted(entry.name for entry in iterator))
    _require(
        _fingerprint(root_metadata) == _fingerprint(root_after) and entries_after == expected_files,
        f"phase tree changed while authenticated: {requested}",
    )
    tree_snapshot = _PhaseTreeSnapshot(
        root_path=requested,
        root_fingerprint=_fingerprint(root_after),
        file_fingerprints=tuple((name, snapshots[name].fingerprint) for name in expected_files),
        marker_path=requested / _MANIFEST,
        marker_sha256=snapshots[_MANIFEST].sha256,
        marker_fingerprint=snapshots[_MANIFEST].fingerprint,
    )

    return _PhaseEvidence(
        artifact=expected_artifact,
        seal_sha256=snapshots[_MANIFEST].sha256,
        payloads={path: snapshots[path].payload for path in expected_payload_paths},
        payload_sha256={path: snapshots[path].sha256 for path in expected_payload_paths},
        predecessors=dict(sorted(predecessors.items())),
        metadata=metadata,
        file_payloads={name: snapshots[name].payload for name in expected_files},
        tree_snapshot=tree_snapshot,
    )


def _assert_phase_unchanged(phase: _PhaseEvidence) -> None:
    evidence = phase.tree_snapshot
    expected_names = tuple(name for name, _fingerprint_value in evidence.file_fingerprints)
    _require(
        set(phase.file_payloads) == set(expected_names),
        "authenticated phase evidence inventory differs from its tree snapshot",
    )
    _reject_symlink_chain(evidence.root_path, label="authenticated phase root")
    try:
        root_before = os.lstat(evidence.root_path)
        with os.scandir(evidence.root_path) as iterator:
            inventory_before = tuple(sorted(entry.name for entry in iterator))
        observed_snapshots = tuple(
            (
                name,
                _snapshot_file(
                    evidence.root_path / name,
                    label=f"authenticated phase file {name}",
                    max_bytes=1024 * 1024 if name == _MANIFEST else MAX_FILE_BYTES,
                ),
            )
            for name, _fingerprint_value in evidence.file_fingerprints
        )
        with os.scandir(evidence.root_path) as iterator:
            inventory_after = tuple(sorted(entry.name for entry in iterator))
        root_after = os.lstat(evidence.root_path)
    except OSError as error:
        raise VerificationError(
            f"authenticated phase tree disappeared: {evidence.root_path}"
        ) from error
    observed_files = tuple((name, snapshot.fingerprint) for name, snapshot in observed_snapshots)
    observed_payloads = tuple((name, snapshot.payload) for name, snapshot in observed_snapshots)
    expected_payloads = tuple((name, phase.file_payloads[name]) for name in expected_names)
    _require(
        _fingerprint(root_before) == _fingerprint(root_after) == evidence.root_fingerprint
        and inventory_before == inventory_after == expected_names
        and observed_files == evidence.file_fingerprints
        and observed_payloads == expected_payloads,
        f"authenticated phase tree changed after capture: {evidence.root_path}",
    )
    snapshot = _snapshot_file(
        evidence.marker_path,
        label="authenticated phase marker",
        max_bytes=1024 * 1024,
    )
    _require(
        snapshot.sha256 == evidence.marker_sha256
        and snapshot.fingerprint == evidence.marker_fingerprint,
        f"authenticated marker changed after capture: {evidence.marker_path}",
    )


def _rotation_document(outer_fold: int, pool_fold: int) -> dict[str, object]:
    _require(
        type(outer_fold) is int
        and type(pool_fold) is int
        and outer_fold in range(5)
        and pool_fold in range(5)
        and outer_fold != pool_fold,
        "invalid frozen rotation",
    )
    return {
        "schema_version": 1,
        "rotation_id": f"outer-{outer_fold}.pool-{pool_fold}",
        "outer_fold": outer_fold,
        "acquisition_pool_fold": pool_fold,
        "base_folds": [fold for fold in range(5) if fold not in {outer_fold, pool_fold}],
    }


def _ordered_rotations() -> tuple[dict[str, object], ...]:
    return tuple(
        _rotation_document(outer, pool) for outer in range(5) for pool in range(5) if outer != pool
    )


def _run_document(
    rotation: Mapping[str, object], policy: str, seed: int | None
) -> dict[str, object]:
    rotation_id = rotation["rotation_id"]
    pool_fold = rotation["acquisition_pool_fold"]
    _require(type(rotation_id) is str and type(pool_fold) is int, "rotation document is malformed")
    if policy == "random":
        _require(type(seed) is int and seed in RANDOM_SEEDS, "random seed is not frozen")
    else:
        _require(seed is None, "only random policy may carry a seed")
    track_base = f"{rotation_id}.policy-{policy}"
    track_id = f"{track_base}.seed-{seed}" if seed is not None else track_base
    selection_kind = (
        "no_query"
        if policy == "no_query"
        else "full_acquisition_fold_ceiling"
        if policy == "full_acquisition_fold_ceiling"
        else "budgeted"
    )
    selected_count = (
        0
        if policy == "no_query"
        else EXPECTED_SUPPORT_BY_FOLD[pool_fold]
        if policy == "full_acquisition_fold_ceiling"
        else 10
    )
    return {
        "schema_version": 1,
        "track_id": track_id,
        "rotation_id": rotation_id,
        "policy": policy,
        "seed": seed,
        "selection_kind": selection_kind,
        "expected_pool_selection_count": selected_count,
        "expected_outer_selection_count": 10,
        "refit": policy != "no_query",
    }


def _ordered_runs() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for rotation in _ordered_rotations():
        for policy in POLICY_ORDER:
            seeds: Sequence[int | None] = RANDOM_SEEDS if policy == "random" else (None,)
            rows.extend(_run_document(rotation, policy, seed) for seed in seeds)
    _require(len(rows) == 220, "frozen run census changed")
    return tuple(rows)


def _ordered_id_stream_sha256(values: Sequence[str], *, allow_empty: bool = False) -> str:
    identifiers = tuple(values)
    _require(allow_empty or identifiers, "identifier stream must be nonempty")
    _require(
        len(identifiers) == len(set(identifiers))
        and all(
            type(value) is str and "\n" not in value and "\r" not in value for value in identifiers
        ),
        "identifier stream is invalid",
    )
    return _sha256_bytes("".join(f"{value}\n" for value in identifiers).encode("ascii"))


def _canonical_hex(value: object, *, label: str) -> float:
    _require(type(value) is str, f"{label} must be canonical binary64 hexadecimal text")
    try:
        parsed = float.fromhex(value)  # type: ignore[arg-type]
    except ValueError as error:
        raise VerificationError(f"{label} is not binary64 hexadecimal text") from error
    _require(
        math.isfinite(parsed) and parsed.hex() == value,
        f"{label} is not canonical finite binary64 text",
    )
    return parsed


def _finite_number(value: object, *, label: str) -> float:
    _require(
        type(value) in {int, float} and not isinstance(value, bool), f"{label} must be numeric"
    )
    parsed = float(value)
    _require(math.isfinite(parsed), f"{label} must be finite")
    return parsed


def _mean(values: Sequence[float], *, label: str) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    _require(
        array.ndim == 1 and len(array) > 0 and bool(np.all(np.isfinite(array))),
        f"{label} is invalid",
    )
    return float(np.mean(array, dtype=np.float64))


def _expect_run_document(value: object, expected: Mapping[str, object], *, label: str) -> None:
    _require(
        type(value) is dict
        and _canonical_json_bytes(value) == _canonical_json_bytes(dict(expected)),
        f"{label} differs from frozen run identity",
    )


def _expect_flat_run_fields(
    row: Mapping[str, object], run: Mapping[str, object], *, label: str
) -> None:
    for field in ("track_id", "rotation_id", "policy", "seed", "selection_kind"):
        _require(
            field in row and type(row[field]) is type(run[field]) and row[field] == run[field],
            f"{label} {field} differs from frozen run identity",
        )


def _payload_digest_map(
    value: object, expected_paths: Sequence[str], *, label: str
) -> dict[str, str]:
    raw = _exact_object(value, set(expected_paths), label=label)
    return {path: _sha256(raw[path], label=f"{label} {path}") for path in expected_paths}


def _global_phase_specs() -> dict[str, tuple[str, tuple[str, ...], str]]:
    return {
        "protocol": (
            "sequential_v2_protocol_v1",
            ("policy-runs.jsonl", "protocol-census.json", "rotations.jsonl"),
            "protocol",
        ),
        "prepare": (
            "sequential_v2_prepare_campaign_barrier_v1",
            ("prepare-index.jsonl", "prepare-summary.json"),
            "prepare",
        ),
        "select": (
            "sequential_v2_pool_commitment_campaign_barrier_v1",
            ("campaign-summary.json", "commitment-index.jsonl", "rotation-index.jsonl"),
            "select",
        ),
        "reveal": (
            "sequential_v2_pool_reveal_campaign_barrier_v1",
            ("reveal-index.jsonl", "reveal-summary.json"),
            "reveal",
        ),
        "update": (
            "sequential_v2_update_campaign_barrier_v1",
            ("update-index.jsonl", "update-summary.json"),
            "update",
        ),
        "outer-select": (
            "sequential_v2_outer_selection_campaign_barrier_v1",
            ("outer-selection-index.jsonl", "outer-selection-summary.json"),
            "outer-select",
        ),
    }


def _verify_protocol_payloads(phase: _PhaseEvidence) -> None:
    _strict_jsonl(phase.payloads["rotations.jsonl"], label="protocol rotations")
    _strict_jsonl(phase.payloads["policy-runs.jsonl"], label="protocol policy runs")
    expected_rotations = _ordered_rotations()
    expected_runs = _ordered_runs()
    _require(
        phase.payloads["rotations.jsonl"] == _canonical_jsonl_bytes(expected_rotations),
        "protocol rotations differ from frozen graph",
    )
    _require(
        phase.payloads["policy-runs.jsonl"] == _canonical_jsonl_bytes(expected_runs),
        "protocol policy runs differ from frozen graph",
    )
    _strict_json_object(phase.payloads["protocol-census.json"], label="protocol census")
    expected_census = {
        "schema_version": 1,
        "rotations": 20,
        "policy_runs": 220,
        "policy_runs_per_rotation": 11,
        "random_policy_runs": 100,
        "deterministic_policy_runs": 120,
        "base_state_references": 20,
        "refits": 200,
        "update_states": 220,
        "pool_candidates": 2600,
        "prediction_bearing_pool_view_rows": 2600,
        "random_minimal_pool_view_rows": 2600,
        "physical_pool_candidate_view_rows": 5200,
        "pool_committed_sequence_associations": 4400,
        "outer_context_predictions": 109648,
        "outer_candidates": 28600,
        "outer_committed_sequence_associations": 2200,
    }
    _require(
        phase.payloads["protocol-census.json"] == _canonical_json_bytes(expected_census),
        "protocol census differs from frozen graph",
    )
    _require(phase.predecessors == {}, "protocol phase must have no predecessors")


@dataclass(frozen=True, slots=True)
class _IndexedAuthorities:
    stage_outer: Mapping[str, str]
    prediction_view: Mapping[str, str]
    commitments: Mapping[str, tuple[str, str]]
    reveal: Mapping[str, str]
    outer_evidence: Mapping[str, str]
    outer_view: Mapping[str, str]
    outer_selection: Mapping[str, str]


def _verify_stage_source_anchors(phase: _PhaseEvidence) -> None:
    document = _strict_json_object(
        phase.payloads["source-anchors.json"],
        label="stage source anchors",
    )
    anchor_fields = {
        "schema_version",
        "artifact",
        "publication_top_sha256",
        "semantic_top_sha256",
        "independent_receipt_sha256",
        "gate1_authentication_evidence",
    }
    anchors = _exact_object(document, anchor_fields, label="stage source anchors")
    _require(
        anchors["schema_version"] == 1
        and type(anchors["schema_version"]) is int
        and anchors["artifact"] == "sequential_v2_stage_source_anchors",
        "stage source-anchor identity changed",
    )
    _require(
        _sha256(anchors["publication_top_sha256"], label="Gate-1 publication top")
        == _ACCEPTED_GATE1_PUBLICATION_TOP_SHA256
        and _sha256(anchors["semantic_top_sha256"], label="Gate-1 semantic top")
        == _ACCEPTED_GATE1_SEMANTIC_TOP_SHA256
        and _sha256(anchors["independent_receipt_sha256"], label="Gate-1 audit receipt")
        == _ACCEPTED_GATE1_INDEPENDENT_RECEIPT_SHA256,
        "stage source anchors differ from the frozen accepted Gate-1 pins",
    )

    raw_evidence = anchors["gate1_authentication_evidence"]
    evidence_fields = {
        "schema_version",
        "artifact",
        "producer_job_id",
        "audit_job_id",
        "twins_byte_identical",
        "independent_receipt_sha256",
        "files",
    }
    evidence = _exact_object(
        raw_evidence,
        evidence_fields,
        label="stage Gate-1 authentication evidence",
    )
    _require(
        evidence["schema_version"] == 1
        and type(evidence["schema_version"]) is int
        and evidence["artifact"] == "sequential_v2_authenticated_gate1_source"
        and evidence["producer_job_id"] == _ACCEPTED_GATE1_PRODUCER_JOB_ID
        and type(evidence["producer_job_id"]) is int
        and evidence["audit_job_id"] == _ACCEPTED_GATE1_AUDIT_JOB_ID
        and type(evidence["audit_job_id"]) is int
        and evidence["twins_byte_identical"] is True
        and evidence["independent_receipt_sha256"] == _ACCEPTED_GATE1_INDEPENDENT_RECEIPT_SHA256,
        "stage Gate-1 authentication evidence differs from the frozen job IDs or receipt",
    )

    expected_files = {
        logical: (digest, mode) for logical, digest, mode in _ACCEPTED_GATE1_SOURCE_FILES
    }
    raw_files = _exact_object(
        evidence["files"],
        set(expected_files),
        label="stage Gate-1 authenticated file inventory",
    )
    for logical, (expected_digest, expected_mode) in expected_files.items():
        item = _exact_object(
            raw_files[logical],
            {"mode", "sha256", "size_bytes"},
            label=f"stage Gate-1 file evidence {logical}",
        )
        _require(
            _sha256(item["sha256"], label=f"stage Gate-1 file {logical}") == expected_digest
            and item["mode"] == expected_mode
            and type(item["mode"]) is str
            and type(item["size_bytes"]) is int
            and item["size_bytes"] > 0,
            f"stage Gate-1 file {logical} differs from the frozen accepted evidence",
        )
    _require(
        raw_files["SHA256SUMS"]["sha256"] == anchors["publication_top_sha256"]
        and raw_files["gate1/SHA256SUMS"]["sha256"] == anchors["semantic_top_sha256"],
        "stage Gate-1 top anchors differ from their authenticated file evidence",
    )
    for path, digest in _ACCEPTED_GATE1_SOURCE_PREDECESSORS.items():
        _require(
            phase.predecessors.get(path) == digest,
            f"stage global predecessor {path} differs from the frozen Gate-1 pin",
        )


def _stage_index_authority(phase: _PhaseEvidence) -> dict[str, str]:
    rows = _strict_jsonl(phase.payloads["capability-index.jsonl"], label="stage capability index")
    roles = (
        ("prepare-capability", "sequential_v2_stage_prepare_capability"),
        ("pool-outcome-vault", "sequential_v2_stage_pool_outcome_vault"),
        ("outer-metadata", "sequential_v2_stage_outer_metadata"),
        ("outer-outcome-vault", "sequential_v2_stage_outer_outcome_vault"),
    )
    expected_positions = tuple(
        (rotation, role, artifact) for rotation in _ordered_rotations() for role, artifact in roles
    )
    _require(len(rows) == 80, "stage capability index must contain exactly 80 rows")
    outer: dict[str, str] = {}
    indexed: dict[str, str] = {}
    fields = {
        "schema_version",
        "rotation_id",
        "role",
        "relative_path",
        "leaf_artifact",
        "leaf_seal_sha256",
        "payload_paths",
        "row_counts",
        "id_streams",
    }
    for index, (raw, (rotation, role, artifact)) in enumerate(
        zip(rows, expected_positions, strict=True)
    ):
        row = _exact_object(raw, fields, label=f"stage index row {index}")
        rotation_id = rotation["rotation_id"]
        expected_path = f"rotations/{rotation_id}/{role}"
        _require(
            row["schema_version"] == 1
            and type(row["schema_version"]) is int
            and row["rotation_id"] == rotation_id
            and row["role"] == role
            and row["relative_path"] == expected_path
            and row["leaf_artifact"] == artifact,
            f"stage index row {index} identity/order changed",
        )
        digest = _sha256(row["leaf_seal_sha256"], label=f"stage index leaf {index}")
        indexed[expected_path] = digest
        if role == "outer-outcome-vault":
            outer[str(rotation_id)] = digest
    for path, digest in indexed.items():
        _require(
            phase.predecessors.get(path) == digest,
            f"stage global does not bind indexed leaf {path}",
        )
    _require(len(phase.predecessors) == 83, "stage global predecessor census changed")
    return outer


def _prepare_index_authority(phase: _PhaseEvidence) -> dict[str, str]:
    rows = _strict_jsonl(phase.payloads["prepare-index.jsonl"], label="prepare campaign index")
    roles = (
        ("evidence", "sequential_v2_prepare_evidence_v1"),
        ("base-update", "sequential_v2_prepare_base_update_v1"),
        ("prediction-view", "sequential_v2_prepare_prediction_view_v1"),
        ("random-minimal-view", "sequential_v2_prepare_random_minimal_view_v1"),
    )
    expected_positions = tuple(
        (rotation, role, artifact) for rotation in _ordered_rotations() for role, artifact in roles
    )
    _require(len(rows) == 80, "prepare campaign index must contain exactly 80 rows")
    prediction: dict[str, str] = {}
    indexed: dict[str, str] = {}
    fields = {
        "schema_version",
        "rotation_id",
        "leaf_role",
        "relative_path",
        "artifact",
        "phase_seal_sha256",
        "payload_sha256",
        "base_model_state_sha256",
        "candidate_count",
        "candidate_ids_sha256",
    }
    for index, (raw, (rotation, role, artifact)) in enumerate(
        zip(rows, expected_positions, strict=True)
    ):
        row = _exact_object(raw, fields, label=f"prepare index row {index}")
        rotation_id = rotation["rotation_id"]
        expected_path = f"prepare/rotations/{rotation_id}/{role}"
        _require(
            row["schema_version"] == 1
            and type(row["schema_version"]) is int
            and row["rotation_id"] == rotation_id
            and row["leaf_role"] == role
            and row["relative_path"] == expected_path
            and row["artifact"] == artifact,
            f"prepare index row {index} identity/order changed",
        )
        digest = _sha256(row["phase_seal_sha256"], label=f"prepare index leaf {index}")
        indexed[expected_path] = digest
        if role == "prediction-view":
            _require(
                row["candidate_count"]
                == EXPECTED_SUPPORT_BY_FOLD[rotation["acquisition_pool_fold"]]
                and type(row["candidate_count"]) is int,
                "prediction-view index census changed",
            )
            prediction[str(rotation_id)] = digest
    expected_predecessors = {"protocol/SHA256SUMS": phase.predecessors.get("protocol/SHA256SUMS")}
    expected_predecessors.update({f"{path}/SHA256SUMS": digest for path, digest in indexed.items()})
    _require(
        phase.predecessors == expected_predecessors,
        "prepare global index/predecessor closure differs",
    )
    return prediction


def _select_index_authority(phase: _PhaseEvidence) -> dict[str, tuple[str, str]]:
    rows = _strict_jsonl(phase.payloads["commitment-index.jsonl"], label="select commitment index")
    runs = _ordered_runs()
    _require(len(rows) == 220, "select commitment index must contain exactly 220 rows")
    result: dict[str, tuple[str, str]] = {}
    indexed: dict[str, str] = {}
    fields = {
        "schema_version",
        "track_id",
        "rotation_id",
        "policy",
        "seed",
        "selection_kind",
        "relative_path",
        "leaf_artifact",
        "leaf_seal_sha256",
        "commitment_payload_sha256",
        "selected_sequence_count",
        "selected_sequence_ids_sha256",
        "selector_kind",
        "selector_output_seal_sha256",
        "input_view",
    }
    for index, (raw, run) in enumerate(zip(rows, runs, strict=True)):
        row = _exact_object(raw, fields, label=f"select commitment index row {index}")
        _expect_flat_run_fields(row, run, label=f"select commitment index row {index}")
        track_id = str(run["track_id"])
        rotation_id = str(run["rotation_id"])
        expected_path = f"select/rotations/{rotation_id}/commitments/{track_id}"
        _require(
            row["schema_version"] == 1
            and type(row["schema_version"]) is int
            and row["relative_path"] == expected_path
            and row["leaf_artifact"] == "sequential_v2_pool_commitment_v1"
            and row["selected_sequence_count"] == run["expected_pool_selection_count"]
            and type(row["selected_sequence_count"]) is int,
            f"select commitment index row {index} changed",
        )
        leaf = _sha256(row["leaf_seal_sha256"], label=f"select commitment leaf {index}")
        payload = _sha256(
            row["commitment_payload_sha256"], label=f"select commitment payload {index}"
        )
        result[track_id] = (leaf, payload)
        indexed[expected_path] = leaf
    rotation_rows = _strict_jsonl(
        phase.payloads["rotation-index.jsonl"], label="select rotation index"
    )
    _require(len(rotation_rows) == 20, "select rotation index must contain exactly 20 rows")
    for index, (row, rotation) in enumerate(zip(rotation_rows, _ordered_rotations(), strict=True)):
        _require(
            row.get("rotation_id") == rotation["rotation_id"],
            f"select rotation index row {index} order changed",
        )
        path = row.get("relative_path")
        digest = row.get("phase_seal_sha256")
        _require(type(path) is str, f"select rotation index row {index} lacks path")
        indexed[path] = _sha256(digest, label=f"select rotation leaf {index}")
    expected = {
        "protocol/SHA256SUMS": phase.predecessors.get("protocol/SHA256SUMS"),
        "prepare/global/SHA256SUMS": phase.predecessors.get("prepare/global/SHA256SUMS"),
    }
    expected.update({f"{path}/SHA256SUMS": digest for path, digest in indexed.items()})
    _require(phase.predecessors == expected, "select global index/predecessor closure differs")
    return result


def _reveal_index_authority(
    phase: _PhaseEvidence,
    commitments: Mapping[str, tuple[str, str]],
) -> dict[str, str]:
    rows = _strict_jsonl(phase.payloads["reveal-index.jsonl"], label="reveal campaign index")
    runs = _ordered_runs()
    _require(len(rows) == 220, "reveal index must contain exactly 220 rows")
    result: dict[str, str] = {}
    for index, (row, run) in enumerate(zip(rows, runs, strict=True)):
        _expect_flat_run_fields(row, run, label=f"reveal index row {index}")
        track_id = str(run["track_id"])
        expected_path = f"reveal/tracks/{track_id}"
        _require(
            row.get("schema_version") == 1
            and type(row.get("schema_version")) is int
            and row.get("relative_path") == expected_path
            and row.get("leaf_artifact") == "sequential_v2_pool_reveal_v1"
            and row.get("selected_sequence_count") == run["expected_pool_selection_count"],
            f"reveal index row {index} changed",
        )
        _require(
            row.get("commitment_leaf_seal_sha256") == commitments[track_id][0],
            f"reveal index row {index} is not bound to select-global commitment",
        )
        result[track_id] = _sha256(row.get("leaf_seal_sha256"), label=f"reveal index leaf {index}")
    expected = {
        "protocol/SHA256SUMS": phase.predecessors.get("protocol/SHA256SUMS"),
        "stage/global/SHA256SUMS": phase.predecessors.get("stage/global/SHA256SUMS"),
        "select/global/SHA256SUMS": phase.predecessors.get("select/global/SHA256SUMS"),
        **{
            f"reveal/tracks/{run['track_id']}/SHA256SUMS": result[str(run["track_id"])]
            for run in runs
        },
    }
    _require(phase.predecessors == expected, "reveal global index/predecessor closure differs")
    return result


def _update_index_authority(phase: _PhaseEvidence) -> tuple[dict[str, str], dict[str, str]]:
    rows = _strict_jsonl(phase.payloads["update-index.jsonl"], label="update campaign index")
    runs = _ordered_runs()
    rotations = _ordered_rotations()
    _require(len(rows) == 680, "update index must contain exactly 680 rows")
    expected_roles = (
        *("state" for _ in runs),
        *("outer_components" for _ in rotations),
        *("outer_view" for _ in runs),
        *("outer_evidence" for _ in runs),
    )
    indexed: dict[str, str] = {}
    views: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for index, (row, role) in enumerate(zip(rows, expected_roles, strict=True)):
        _require(row.get("index_role") == role, f"update index role/order changed at row {index}")
        path = row.get("relative_path")
        _require(type(path) is str, f"update index row {index} lacks relative path")
        digest = _sha256(row.get("leaf_seal_sha256"), label=f"update index leaf {index}")
        indexed[path] = digest
        if role in {"state", "outer_view", "outer_evidence"}:
            run_index = (
                index if role == "state" else index - 240 if role == "outer_view" else index - 460
            )
            run = runs[run_index]
            _expect_run_document(row.get("run"), run, label=f"update {role} index row {index}")
            track_id = str(run["track_id"])
            suffix = role.replace("outer_", "outer-")
            _require(path == f"update/tracks/{track_id}/{suffix}", f"update {role} path changed")
            if role == "outer_view":
                views[track_id] = digest
            elif role == "outer_evidence":
                evidence[track_id] = digest
        else:
            rotation = rotations[index - 220]
            _require(
                row.get("rotation") == rotation,
                f"update component rotation order changed at {index}",
            )
    expected = {
        "protocol/SHA256SUMS": phase.predecessors.get("protocol/SHA256SUMS"),
        "stage/global/SHA256SUMS": phase.predecessors.get("stage/global/SHA256SUMS"),
        "prepare/global/SHA256SUMS": phase.predecessors.get("prepare/global/SHA256SUMS"),
        "reveal/global/SHA256SUMS": phase.predecessors.get("reveal/global/SHA256SUMS"),
        **{f"{path}/SHA256SUMS": digest for path, digest in indexed.items()},
    }
    _require(phase.predecessors == expected, "update global index/predecessor closure differs")
    return evidence, views


def _outer_selection_index_authority(
    phase: _PhaseEvidence,
    outer_views: Mapping[str, str],
) -> dict[str, str]:
    rows = _strict_jsonl(
        phase.payloads["outer-selection-index.jsonl"], label="outer-selection campaign index"
    )
    runs = _ordered_runs()
    _require(len(rows) == 220, "outer-selection index must contain exactly 220 rows")
    result: dict[str, str] = {}
    for index, (row, run) in enumerate(zip(rows, runs, strict=True)):
        _expect_run_document(row.get("run"), run, label=f"outer-selection index row {index}")
        track_id = str(run["track_id"])
        path = f"outer-select/tracks/{track_id}"
        _require(
            row.get("schema_version") == 1
            and type(row.get("schema_version")) is int
            and row.get("relative_path") == path
            and row.get("leaf_artifact") == "sequential_v2_outer_selection_commitment_v1"
            and row.get("selected_sequence_count") == 10,
            f"outer-selection index row {index} changed",
        )
        _require(
            row.get("outer_view_leaf_seal_sha256") == outer_views[track_id],
            f"outer-selection index row {index} is not bound to update outer view",
        )
        result[track_id] = _sha256(
            row.get("leaf_seal_sha256"), label=f"outer-selection index leaf {index}"
        )
    expected = {
        "protocol/SHA256SUMS": phase.predecessors.get("protocol/SHA256SUMS"),
        "update/global/SHA256SUMS": phase.predecessors.get("update/global/SHA256SUMS"),
        **{
            f"outer-select/tracks/{run['track_id']}/SHA256SUMS": result[str(run["track_id"])]
            for run in runs
        },
    }
    _require(
        phase.predecessors == expected, "outer-selection global index/predecessor closure differs"
    )
    return result


def _verify_global_anchors(
    phases: Mapping[str, _PhaseEvidence], authorities: GlobalAuthorities
) -> None:
    expected = authorities.document()
    anchor_checks = {
        "prepare": {"protocol/SHA256SUMS": expected["protocol_seal_sha256"]},
        "select": {
            "protocol/SHA256SUMS": expected["protocol_seal_sha256"],
            "prepare/global/SHA256SUMS": expected["prepare_global_seal_sha256"],
        },
        "reveal": {
            "protocol/SHA256SUMS": expected["protocol_seal_sha256"],
            "stage/global/SHA256SUMS": expected["stage_global_seal_sha256"],
            "select/global/SHA256SUMS": expected["selection_global_seal_sha256"],
        },
        "update": {
            "protocol/SHA256SUMS": expected["protocol_seal_sha256"],
            "stage/global/SHA256SUMS": expected["stage_global_seal_sha256"],
            "prepare/global/SHA256SUMS": expected["prepare_global_seal_sha256"],
            "reveal/global/SHA256SUMS": expected["reveal_global_seal_sha256"],
        },
        "outer-select": {
            "protocol/SHA256SUMS": expected["protocol_seal_sha256"],
            "update/global/SHA256SUMS": expected["update_global_seal_sha256"],
        },
    }
    for phase_name, anchors in anchor_checks.items():
        for path, digest in anchors.items():
            _require(
                phases[phase_name].predecessors.get(path) == digest,
                f"{phase_name} global anchor differs at {path}",
            )


@dataclass(frozen=True, slots=True)
class _Context:
    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    target: str
    gram: str
    fold: int
    label: int
    source_observations: int


@dataclass(frozen=True, slots=True)
class _PoolCandidate:
    sequence_id: str
    features: tuple[float, ...]
    component: str


@dataclass(frozen=True, slots=True)
class _OuterCandidate:
    sequence_id: str
    probabilities: tuple[float, float, float]
    component: str

    @property
    def scalar_mean(self) -> float:
        return math.fsum(self.probabilities) / 3.0


def _canonical_sequence(value: object, sequence_id: object, *, label: str) -> tuple[str, str]:
    _require(type(value) is str, f"{label} sequence must be text")
    sequence = value  # type: ignore[assignment]
    _require(
        8 <= len(sequence) <= 50
        and set(sequence).issubset(_AMINO_ACIDS)
        and sequence == sequence.upper(),
        f"{label} sequence is not canonical",
    )
    expected_id = _sha256_bytes(sequence.encode("ascii"))
    _require(
        sequence_id == expected_id and type(sequence_id) is str, f"{label} sequence ID differs"
    )
    return expected_id, sequence


def _context_rows(payload: bytes, *, fold: int, label: str) -> tuple[_Context, ...]:
    fields = {
        "example_id",
        "assay_context_id",
        "sequence_id",
        "sequence",
        "target",
        "gram",
        "fold",
        "label",
        "source_observations",
    }
    result: list[_Context] = []
    for index, raw in enumerate(_strict_jsonl(payload, label=label, allow_empty=True)):
        row = _exact_object(raw, fields, label=f"{label} row {index}")
        sequence_id, sequence = _canonical_sequence(
            row["sequence"], row["sequence_id"], label=f"{label} row {index}"
        )
        example_id = _sha256(row["example_id"], label=f"{label} example ID {index}")
        target = row["target"]
        _require(
            type(target) is str
            and target in TARGET_GRAM
            and row["gram"] == TARGET_GRAM[target]
            and type(row["gram"]) is str
            and row["assay_context_id"] == example_id
            and type(row["assay_context_id"]) is str
            and row["fold"] == fold
            and type(row["fold"]) is int
            and type(row["label"]) is int
            and row["label"] in {0, 1}
            and type(row["source_observations"]) is int
            and row["source_observations"] >= 1,
            f"{label} row {index} has invalid context semantics",
        )
        result.append(
            _Context(
                example_id,
                str(row["assay_context_id"]),
                sequence_id,
                sequence,
                target,
                str(row["gram"]),
                int(row["fold"]),
                int(row["label"]),
                int(row["source_observations"]),
            )
        )
    ids = tuple(row.example_id for row in result)
    _require(ids == tuple(sorted(set(ids))), f"{label} must use ascending unique example IDs")
    return tuple(result)


def _outer_predictions(
    payload: bytes,
    *,
    run: Mapping[str, object],
    outcomes: Sequence[_Context],
) -> tuple[float, ...]:
    rows = _strict_jsonl(payload, label="outer context predictions")
    _require(len(rows) == len(outcomes), "outer prediction/outcome census differs")
    fields = {
        "schema_version",
        "track_id",
        "rotation_id",
        "example_id",
        "sequence_id",
        "target",
        "gram",
        "fold",
        "probability_hex",
    }
    result: list[float] = []
    for index, (raw, outcome) in enumerate(zip(rows, outcomes, strict=True)):
        row = _exact_object(raw, fields, label=f"outer context prediction {index}")
        expected_identity = (
            outcome.example_id,
            outcome.sequence_id,
            outcome.target,
            outcome.gram,
            outcome.fold,
        )
        observed_identity = tuple(
            row[field] for field in ("example_id", "sequence_id", "target", "gram", "fold")
        )
        _require(
            row["schema_version"] == 1
            and type(row["schema_version"]) is int
            and all(
                type(row[field]) is str
                for field in (
                    "track_id",
                    "rotation_id",
                    "example_id",
                    "sequence_id",
                    "target",
                    "gram",
                )
            )
            and type(row["fold"]) is int
            and row["track_id"] == run["track_id"]
            and row["rotation_id"] == run["rotation_id"]
            and observed_identity == expected_identity,
            f"outer context prediction {index} identity differs exactly",
        )
        probability = _canonical_hex(
            row["probability_hex"], label=f"outer context probability {index}"
        )
        _require(
            PROBABILITY_CLIP <= probability <= 1.0 - PROBABILITY_CLIP,
            f"outer context probability {index} is outside model bounds",
        )
        result.append(probability)
    return tuple(result)


def _selected_ids(payload: bytes, *, expected_count: int, label: str) -> tuple[str, ...]:
    rows = _strict_jsonl(payload, label=label, allow_empty=expected_count == 0)
    identifiers: list[str] = []
    for index, raw in enumerate(rows):
        row = _exact_object(raw, {"sequence_id"}, label=f"{label} row {index}")
        identifiers.append(_sha256(row["sequence_id"], label=f"{label} sequence ID {index}"))
    result = tuple(identifiers)
    _require(
        len(result) == expected_count and len(result) == len(set(result)),
        f"{label} count or uniqueness changed",
    )
    return result


def _probability_object(value: object, *, label: str) -> tuple[float, float, float]:
    row = _exact_object(value, set(OBJECTIVES), label=label)
    values = tuple(_finite_number(row[name], label=f"{label} {name}") for name in OBJECTIVES)
    _require(
        all(PROBABILITY_CLIP <= item <= 1.0 - PROBABILITY_CLIP for item in values),
        f"{label} probabilities are outside model bounds",
    )
    return values  # type: ignore[return-value]


def _pool_candidates(
    payload: bytes, *, rotation: Mapping[str, object]
) -> tuple[_PoolCandidate, ...]:
    rows = _strict_jsonl(payload, label="prediction-view candidates")
    pool_fold = rotation["acquisition_pool_fold"]
    _require(type(pool_fold) is int, "rotation pool fold is invalid")
    _require(
        len(rows) == EXPECTED_SUPPORT_BY_FOLD[pool_fold],
        "prediction-view candidate census changed",
    )
    fields = {
        "rotation_id",
        "sequence_id",
        "sequence",
        "objective_probabilities",
        "features",
        "novelty",
        "diversity_component_id",
        "eligible",
    }
    result: list[_PoolCandidate] = []
    for index, raw in enumerate(rows):
        row = _exact_object(raw, fields, label=f"prediction-view candidate {index}")
        sequence_id, _sequence = _canonical_sequence(
            row["sequence"], row["sequence_id"], label=f"prediction-view candidate {index}"
        )
        _probability_object(
            row["objective_probabilities"], label=f"prediction-view candidate {index} objectives"
        )
        features_raw = row["features"]
        _require(
            type(features_raw) is list and len(features_raw) == 33, "candidate features changed"
        )
        features = tuple(
            _finite_number(value, label=f"candidate {index} feature") for value in features_raw
        )
        norm = math.sqrt(math.fsum(value * value for value in features))
        _require(
            norm == 0.0 or math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12),
            f"candidate {index} features are not normalized",
        )
        novelty = _finite_number(row["novelty"], label=f"candidate {index} novelty")
        component = row["diversity_component_id"]
        _require(
            row["rotation_id"] == rotation["rotation_id"]
            and row["eligible"] is True
            and 0.0 <= novelty <= 1.0
            and type(component) is str
            and _COMPONENT.fullmatch(component) is not None,
            f"prediction-view candidate {index} is invalid",
        )
        result.append(_PoolCandidate(sequence_id, features, component))
    ids = tuple(item.sequence_id for item in result)
    _require(ids == tuple(sorted(set(ids))), "prediction-view candidates are not canonical order")
    return tuple(result)


def _outer_candidates(payload: bytes, *, run: Mapping[str, object]) -> tuple[_OuterCandidate, ...]:
    rows = _strict_jsonl(payload, label="outer-view candidates")
    outer_fold = int(str(run["rotation_id"]).split(".", 1)[0].split("-", 1)[1])
    _require(len(rows) == EXPECTED_SUPPORT_BY_FOLD[outer_fold], "outer candidate census changed")
    fields = {
        "rotation_id",
        "sequence_id",
        "sequence",
        "objective_probabilities",
        "diversity_component_id",
        "eligible",
    }
    result: list[_OuterCandidate] = []
    for index, raw in enumerate(rows):
        row = _exact_object(raw, fields, label=f"outer candidate {index}")
        sequence_id, _sequence = _canonical_sequence(
            row["sequence"], row["sequence_id"], label=f"outer candidate {index}"
        )
        probabilities = _probability_object(
            row["objective_probabilities"], label=f"outer candidate {index} objectives"
        )
        component = row["diversity_component_id"]
        _require(
            row["rotation_id"] == run["rotation_id"]
            and row["eligible"] is True
            and type(component) is str
            and _COMPONENT.fullmatch(component) is not None,
            f"outer candidate {index} is invalid",
        )
        result.append(_OuterCandidate(sequence_id, probabilities, component))
    ids = tuple(item.sequence_id for item in result)
    _require(ids == tuple(sorted(set(ids))), "outer-view candidates are not canonical order")
    return tuple(result)


def _verify_outer_sequence_evidence(
    payload: bytes,
    *,
    run: Mapping[str, object],
    candidates: Sequence[_OuterCandidate],
) -> None:
    rows = _strict_jsonl(payload, label="outer-sequence predictions")
    _require(
        len(rows) == len(candidates),
        "outer sequence-evidence/view candidate censuses differ",
    )
    fields = {
        "schema_version",
        "track_id",
        "rotation_id",
        "sequence_id",
        "target_probabilities_hex",
        "objective_probabilities_hex",
    }
    positive_indices = np.asarray(
        [index for index, target in enumerate(TARGETS) if TARGET_GRAM[target] == "positive"],
        dtype=np.int64,
    )
    negative_indices = np.asarray(
        [index for index, target in enumerate(TARGETS) if TARGET_GRAM[target] == "negative"],
        dtype=np.int64,
    )
    for index, (raw, candidate) in enumerate(zip(rows, candidates, strict=True)):
        row = _exact_object(raw, fields, label=f"outer-sequence prediction {index}")
        _require(
            row["schema_version"] == 1
            and type(row["schema_version"]) is int
            and row["track_id"] == run["track_id"]
            and type(row["track_id"]) is str
            and row["rotation_id"] == run["rotation_id"]
            and type(row["rotation_id"]) is str
            and row["sequence_id"] == candidate.sequence_id
            and type(row["sequence_id"]) is str,
            f"outer-sequence prediction {index} identity differs from its view row",
        )
        target_raw = _exact_object(
            row["target_probabilities_hex"],
            set(TARGETS),
            label=f"outer-sequence prediction {index} target probabilities",
        )
        objective_raw = _exact_object(
            row["objective_probabilities_hex"],
            set(OBJECTIVES),
            label=f"outer-sequence prediction {index} objective probabilities",
        )
        target_values = tuple(
            _canonical_hex(
                target_raw[target],
                label=f"outer-sequence prediction {index} target {target}",
            )
            for target in TARGETS
        )
        objective_values = tuple(
            _canonical_hex(
                objective_raw[objective],
                label=f"outer-sequence prediction {index} objective {objective}",
            )
            for objective in OBJECTIVES
        )
        _require(
            all(
                PROBABILITY_CLIP <= probability <= 1.0 - PROBABILITY_CLIP
                for probability in (*target_values, *objective_values)
            ),
            f"outer-sequence prediction {index} lies outside model bounds",
        )
        target_array = np.asarray(target_values, dtype=np.float64)
        independently_reduced = (
            float(np.mean(target_array, dtype=np.float64)),
            float(np.mean(target_array[positive_indices], dtype=np.float64)),
            float(np.mean(target_array[negative_indices], dtype=np.float64)),
        )
        _require(
            objective_values == independently_reduced,
            f"outer-sequence prediction {index} objectives differ from seven targets",
        )
        _require(
            candidate.probabilities == objective_values,
            f"outer-view candidate {index} probabilities differ from sequence evidence",
        )


def _rederive_outer_selection(candidates: Sequence[_OuterCandidate]) -> tuple[str, ...]:
    selected: list[str] = []
    counts: Counter[str] = Counter()
    for candidate in sorted(candidates, key=lambda item: (-item.scalar_mean, item.sequence_id)):
        if counts[candidate.component] >= 2:
            continue
        selected.append(candidate.sequence_id)
        counts[candidate.component] += 1
        if len(selected) == 10:
            return tuple(selected)
    raise VerificationError("strict component cap leaves fewer than ten outer candidates")


def _guard_status(
    commitment: Mapping[str, object],
    *,
    run: Mapping[str, object],
    selected: Sequence[str],
) -> str:
    result = commitment["selection_result"]
    policy = run["policy"]
    if policy in {"no_query", "full_acquisition_fold_ceiling"}:
        _require(result is None, "no-query/ceiling commitment invents selector result")
        return "not_applicable"
    _require(type(result) is dict, "budgeted commitment lacks selection result")
    selection = result  # type: ignore[assignment]
    seats = selection.get("seats")
    _require(
        type(seats) is list and all(type(seat) is dict for seat in seats),
        "commitment selection-result seats must be mappings",
    )
    _require(
        selection.get("rotation_id") == run["rotation_id"]
        and selection.get("policy") == policy
        and tuple(seat.get("sequence_id") for seat in seats) == tuple(selected),
        "commitment selection result differs from selected IDs",
    )
    evaluations = selection.get("guard_evaluations")
    fallback = selection.get("fallback_reason")
    if policy not in GUARDED_POLICIES:
        _require(evaluations == [] and fallback is None, "unguarded policy carries guard state")
        return "not_applicable"
    _require(type(evaluations) is list and evaluations, "guarded policy lacks final evaluation")
    final = evaluations[-1]
    _require(
        type(final) is dict
        and final.get("passed") is True
        and final.get("sequence_ids") == list(selected),
        "guarded policy lacks a passing final evaluation",
    )
    if fallback is None:
        _require(final.get("stage") != "fallback", "nonfallback guard ends at fallback")
        return "passed"
    _require(
        type(fallback) is str
        and fallback
        and final.get("stage") == "fallback"
        and selection.get("mean_control_sequence_ids") == list(selected),
        "guard fallback is not exact mean fallback",
    )
    return "exact_mean_fallback"


def _commitment_selection(
    payload: bytes,
    *,
    run: Mapping[str, object],
    selected_payload: bytes,
    expected_commitment_sha256: str,
) -> tuple[tuple[str, ...], str]:
    _require(
        _sha256_bytes(payload) == expected_commitment_sha256,
        "reveal commitment copy differs from select-global index",
    )
    commitment = _strict_json_object(payload, label="pool commitment")
    fields = {
        "schema_version",
        "track_id",
        "rotation_id",
        "policy",
        "seed",
        "selection_kind",
        "input_view",
        "selected_sequence_ids",
        "selected_sequence_count",
        "selected_sequence_ids_sha256",
        "selection_result",
    }
    commitment = _exact_object(commitment, fields, label="pool commitment")
    _expect_flat_run_fields(commitment, run, label="pool commitment")
    expected_count = int(run["expected_pool_selection_count"])
    raw_ids = commitment["selected_sequence_ids"]
    _require(type(raw_ids) is list, "pool commitment selected IDs must be an array")
    selected = tuple(_sha256(value, label="pool commitment selected ID") for value in raw_ids)
    _require(
        len(selected) == expected_count
        and len(selected) == len(set(selected))
        and commitment["selected_sequence_count"] == expected_count
        and type(commitment["selected_sequence_count"]) is int
        and commitment["selected_sequence_ids_sha256"]
        == _ordered_id_stream_sha256(selected, allow_empty=expected_count == 0),
        "pool commitment selected identity/census changed",
    )
    from_payload = _selected_ids(
        selected_payload,
        expected_count=expected_count,
        label="reveal selected sequence IDs",
    )
    _require(selected == from_payload, "reveal selected IDs differ from commitment")
    return selected, _guard_status(commitment, run=run, selected=selected)


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    differences = positives[:, None] - negatives[None, :]
    return float((np.sum(differences > 0) + 0.5 * np.sum(differences == 0)) / differences.size)


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(np.sum(labels))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    ordered_probabilities = probabilities[order]
    ordered_labels = labels[order]
    true_positive = 0
    false_positive = 0
    result = 0.0
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and ordered_probabilities[end] == ordered_probabilities[start]:
            end += 1
        group = ordered_labels[start:end]
        new_positive = int(np.sum(group))
        true_positive += new_positive
        false_positive += len(group) - new_positive
        result += (new_positive / positives) * (true_positive / (true_positive + false_positive))
        start = end
    return float(result)


def _ece_10(labels: np.ndarray, probabilities: np.ndarray) -> float:
    bins = np.minimum(np.floor(probabilities * 10.0).astype(np.int64), 9)
    result = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        count = int(np.sum(selected))
        if count:
            result += (count / len(labels)) * abs(
                float(np.mean(probabilities[selected], dtype=np.float64))
                - float(np.mean(labels[selected], dtype=np.float64))
            )
    return float(result)


def _target_metric_document(
    contexts: Sequence[_Context], probabilities: Sequence[float]
) -> tuple[dict[str, object], dict[str, float]]:
    _require(len(contexts) == len(probabilities) and contexts, "target metric columns differ")
    labels = np.asarray([row.label for row in contexts], dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    targets = tuple(row.target for row in contexts)
    metric_names = ("brier", "nll", "roc_auc", "average_precision", "ece_10")
    by_target: dict[str, dict[str, object]] = {}
    defined: dict[str, list[str]] = {name: [] for name in metric_names}
    for target in TARGETS:
        mask = np.asarray([value == target for value in targets], dtype=bool)
        target_labels = labels[mask]
        target_probabilities = probability_array[mask]
        if not len(target_labels):
            values: dict[str, object] = {
                "n": 0,
                "positives": 0,
                "negatives": 0,
                **{name: None for name in metric_names},
                "undefined_reason": "no_outer_contexts",
            }
        else:
            clipped = np.clip(target_probabilities, NLL_CLIP, 1.0 - NLL_CLIP)
            brier = float(np.mean((target_probabilities - target_labels) ** 2, dtype=np.float64))
            nll = float(
                -np.mean(
                    target_labels * np.log(clipped) + (1 - target_labels) * np.log1p(-clipped),
                    dtype=np.float64,
                )
            )
            roc_auc = _roc_auc(target_labels, target_probabilities)
            average_precision = _average_precision(target_labels, target_probabilities)
            ece = _ece_10(target_labels, target_probabilities)
            values = {
                "n": len(target_labels),
                "positives": int(np.sum(target_labels)),
                "negatives": int(len(target_labels) - np.sum(target_labels)),
                "brier": brier,
                "nll": nll,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "ece_10": ece,
                "undefined_reason": (
                    None
                    if roc_auc is not None and average_precision is not None
                    else {
                        "roc_auc": (
                            None if roc_auc is not None else "requires_both_binary_classes"
                        ),
                        "average_precision": (
                            None
                            if average_precision is not None
                            else "requires_at_least_one_positive"
                        ),
                    }
                ),
            }
        by_target[target] = values
        for name in metric_names:
            if values[name] is not None:
                defined[name].append(target)
    macro = {
        name: _mean(
            [float(by_target[target][name]) for target in defined[name]],
            label=f"macro {name}",
        )
        for name in metric_names
    }
    document = {
        "target_order": list(TARGETS),
        "by_target": {
            target: {
                "n": values["n"],
                "positives": values["positives"],
                "negatives": values["negatives"],
                **{
                    name: None if values[name] is None else float(values[name]).hex()
                    for name in metric_names
                },
                "undefined_reason": values["undefined_reason"],
            }
            for target, values in by_target.items()
        },
        "macro": {name: value.hex() for name, value in macro.items()},
        "defined_targets": defined,
        "defined_target_counts": {name: len(defined[name]) for name in metric_names},
        "nll_probability_clip_hex": NLL_CLIP.hex(),
        "ece_bins": "[0,.1),[.1,.2),...,[.8,.9),[.9,1]",
    }
    aggregate = {
        "macro_brier": macro["brier"],
        "macro_negative_log_likelihood": macro["nll"],
        "macro_roc_auc": macro["roc_auc"],
        "macro_average_precision": macro["average_precision"],
        "macro_ece_10": macro["ece_10"],
    }
    return document, aggregate


def _observed_sequence_rewards(contexts: Sequence[_Context]) -> dict[str, float]:
    grouped: dict[str, list[_Context]] = defaultdict(list)
    for row in contexts:
        grouped[row.sequence_id].append(row)
    result: dict[str, float] = {}
    for sequence_id, rows in grouped.items():
        subsets = (
            rows,
            [row for row in rows if row.gram == "positive"],
            [row for row in rows if row.gram == "negative"],
        )
        _require(
            all(subset for subset in subsets), f"sequence {sequence_id} lacks objective support"
        )
        values = tuple(
            float(np.mean([row.label for row in subset], dtype=np.float64)) for subset in subsets
        )
        result[sequence_id] = float(sum(values) / len(OBJECTIVES))
    return result


def _batch_reward(contexts: Sequence[_Context], selected: Sequence[str], *, label: str) -> float:
    _require(selected and len(selected) == len(set(selected)), f"{label} selected IDs are invalid")
    rewards = _observed_sequence_rewards(contexts)
    _require(
        all(sequence_id in rewards for sequence_id in selected),
        f"{label} selected outcome is missing",
    )
    return float(np.mean([rewards[sequence_id] for sequence_id in selected], dtype=np.float64))


def _feature_distance(candidates: Sequence[_PoolCandidate]) -> float:
    matrix = np.asarray([item.features for item in candidates], dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] >= 2, "pairwise feature matrix is invalid")
    norms = np.linalg.norm(matrix, axis=1)
    products = matrix @ matrix.T
    denominators = norms[:, None] * norms[None, :]
    similarities = np.divide(
        products,
        denominators,
        out=np.zeros_like(products, dtype=np.float64),
        where=denominators > 0.0,
    )
    similarities = np.clip(similarities, -1.0, 1.0)
    upper = np.triu_indices(matrix.shape[0], k=1)
    return float(np.mean(1.0 - similarities[upper], dtype=np.float64))


def _rotation_metric_document(
    *,
    run: Mapping[str, object],
    outer_outcomes: Sequence[_Context],
    outer_probabilities: Sequence[float],
    pool_contexts: Sequence[_Context],
    pool_selected: Sequence[str],
    pool_candidates: Sequence[_PoolCandidate],
    guard_status: str,
    outer_selected: Sequence[str],
    outer_candidates: Sequence[_OuterCandidate],
) -> tuple[dict[str, object], dict[str, float | None]]:
    target_metrics, aggregate = _target_metric_document(outer_outcomes, outer_probabilities)
    pool_by_id = {item.sequence_id: item for item in pool_candidates}
    outer_by_id = {item.sequence_id: item for item in outer_candidates}
    _require(len(pool_by_id) == len(pool_candidates), "prediction-view IDs are duplicated")
    _require(len(outer_by_id) == len(outer_candidates), "outer-view IDs are duplicated")
    _require(
        all(item in pool_by_id for item in pool_selected),
        "pool selected ID is absent from prediction view",
    )
    _require(
        all(item in outer_by_id for item in outer_selected),
        "outer selected ID is absent from outer view",
    )

    if run["policy"] == "no_query":
        _require(not pool_contexts and not pool_selected, "no-query acquisition is not empty")
        queried_reward = None
        queried_components = 0
        queried_distance = None
        null_reason: str | None = "no_query_has_empty_acquisition"
    else:
        represented = {row.sequence_id for row in pool_contexts}
        _require(represented == set(pool_selected), "pool reveal does not represent selected IDs")
        queried_reward = _batch_reward(pool_contexts, pool_selected, label="queried")
        selected_candidates = tuple(pool_by_id[item] for item in pool_selected)
        queried_components = len({item.component for item in selected_candidates})
        queried_distance = _feature_distance(selected_candidates)
        null_reason = None

    outer_reward = _batch_reward(outer_outcomes, outer_selected, label="outer")
    per_target = {target: sum(row.target == target for row in pool_contexts) for target in TARGETS}
    component_counts = Counter(outer_by_id[item].component for item in outer_selected)
    unique_outer = len(component_counts)
    max_outer = max(component_counts.values())
    _require(unique_outer >= 5 and max_outer <= 2, "outer selection violates component cap")

    aggregate.update(
        {
            "next_round_outer_top10_mean_reward": outer_reward,
            "queried_outcome_mean_reward": queried_reward,
            "queried_unique_diversity_components": float(queried_components),
            "queried_mean_pairwise_feature_cosine_distance": queried_distance,
            "revealed_context_count": float(len(pool_contexts)),
        }
    )
    document = {
        "schema_version": 1,
        "run": dict(run),
        "target_metrics": target_metrics,
        "next_round_outer_top10_mean_reward_hex": outer_reward.hex(),
        "queried_outcome_mean_reward_hex": (
            None if queried_reward is None else queried_reward.hex()
        ),
        "queried_unique_diversity_components": queried_components,
        "queried_mean_pairwise_feature_cosine_distance_hex": (
            None if queried_distance is None else queried_distance.hex()
        ),
        "queried_metric_null_reason": null_reason,
        "revealed_context_count": len(pool_contexts),
        "per_target_revealed_context_counts": per_target,
        "guard_status": guard_status,
        "outer_selected_unique_component_count": unique_outer,
        "outer_selected_max_component_occupancy": max_outer,
    }
    return document, aggregate


def _aggregate_metrics(
    rows: Sequence[Mapping[str, float | None]], *, label: str
) -> dict[str, float | None]:
    _require(rows, f"{label} requires at least one metric row")
    result: dict[str, float | None] = {}
    for name in AGGREGATE_METRIC_NAMES:
        column = tuple(row[name] for row in rows)
        if all(value is None for value in column):
            result[name] = None
        else:
            _require(
                all(value is not None for value in column), f"{label} {name} mixes null and finite"
            )
            result[name] = _mean(
                [float(value) for value in column if value is not None],
                label=f"{label} {name}",
            )
    return result


def _metric_document(metrics: Mapping[str, float | None]) -> dict[str, object]:
    _require(set(metrics) == set(AGGREGATE_METRIC_NAMES), "aggregate metric inventory changed")
    return {
        f"{name}_hex": None if metrics[name] is None else float(metrics[name]).hex()
        for name in AGGREGATE_METRIC_NAMES
    }


def _bootstrap() -> tuple[np.ndarray, str]:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5), dtype=np.int64)
    digest = _sha256_bytes(indices.astype("<i8", copy=False).tobytes(order="C"))
    _require(digest == BOOTSTRAP_INDICES_SHA256, "independent bootstrap index digest changed")
    return indices, digest


def _paired_comparison_document(
    *,
    comparison_id: str,
    control_policy: str,
    metric: str,
    direction: str,
    by_policy_fold: Mapping[tuple[str, int], Mapping[str, float | None]],
) -> dict[str, object]:
    candidate_policy = "mixed_eight_diversity_one_novelty_one"
    candidate = np.asarray(
        [by_policy_fold[(candidate_policy, fold)][metric] for fold in range(5)],
        dtype=np.float64,
    )
    control = np.asarray(
        [by_policy_fold[(control_policy, fold)][metric] for fold in range(5)],
        dtype=np.float64,
    )
    _require(
        candidate.shape == (5,)
        and control.shape == (5,)
        and bool(np.all(np.isfinite(candidate)))
        and bool(np.all(np.isfinite(control))),
        f"comparison {comparison_id} consumes null/nonfinite metrics",
    )
    differences = candidate - control
    indices, digest = _bootstrap()
    samples = np.mean(differences[indices], axis=1, dtype=np.float64)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    improved = int(np.sum(differences < 0.0 if direction == "lower" else differences > 0.0))
    return {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "candidate_policy": candidate_policy,
        "control_policy": control_policy,
        "metric": metric,
        "difference": "candidate_minus_control",
        "outer_fold_differences_hex": [float(value).hex() for value in differences],
        "point_hex": float(np.mean(differences, dtype=np.float64)).hex(),
        "lower_hex": float(lower).hex(),
        "upper_hex": float(upper).hex(),
        "improved_outer_fold_count": improved,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resample_indices_sha256": digest,
    }


def _float_requirement(
    requirement_id: str, operator: str, observed: float, threshold: float
) -> dict[str, object]:
    operations = {
        "lt": observed < threshold,
        "le": observed <= threshold,
        "gt": observed > threshold,
        "ge": observed >= threshold,
    }
    _require(operator in operations, "unknown float promotion operator")
    return {
        "requirement_id": requirement_id,
        "operator": operator,
        "observed_hex": observed.hex(),
        "observed_integer": None,
        "threshold_hex": threshold.hex(),
        "threshold_integer": None,
        "numeric_representation": "binary64_hex",
        "passed": operations[operator],
    }


def _integer_requirement(
    requirement_id: str, operator: str, observed: int, threshold: int
) -> dict[str, object]:
    operations = {"ge": observed >= threshold, "eq": observed == threshold}
    _require(operator in operations, "unknown integer promotion operator")
    return {
        "requirement_id": requirement_id,
        "operator": operator,
        "observed_hex": None,
        "observed_integer": observed,
        "threshold_hex": None,
        "threshold_integer": threshold,
        "numeric_representation": "integer",
        "passed": operations[operator],
    }


def _promotion_decision(
    comparisons: Sequence[Mapping[str, object]],
    *,
    guarded_track_count: int,
    valid_guarded_track_count: int,
) -> dict[str, object]:
    _require(
        tuple(row["comparison_id"] for row in comparisons) == COMPARISON_ORDER,
        "comparison order changed before promotion",
    )
    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def value(comparison: str, field: str) -> float:
        return _canonical_hex(by_id[comparison][field], label=f"promotion {comparison} {field}")

    def count(comparison: str) -> int:
        observed = by_id[comparison]["improved_outer_fold_count"]
        _require(type(observed) is int, "promotion comparison count is not an integer")
        return observed  # type: ignore[return-value]

    brier = COMPARISON_ORDER[0]
    nll = COMPARISON_ORDER[1]
    reward = COMPARISON_ORDER[2]
    distance = COMPARISON_ORDER[3]
    components = COMPARISON_ORDER[4]
    no_query = COMPARISON_ORDER[5]
    random = COMPARISON_ORDER[6]
    item_requirements = (
        (
            "brier_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_brier_point_strictly_below_zero",
                    "lt",
                    value(brier, "point_hex"),
                    0.0,
                ),
                _integer_requirement(
                    "mixed_brier_improved_outer_folds_at_least_three",
                    "ge",
                    count(brier),
                    3,
                ),
                _float_requirement(
                    "mixed_minus_mean_brier_ci_upper_at_most_0_005",
                    "le",
                    value(brier, "upper_hex"),
                    0.005,
                ),
            ),
        ),
        (
            "nll_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_nll_ci_upper_at_most_0_01",
                    "le",
                    value(nll, "upper_hex"),
                    0.01,
                ),
            ),
        ),
        (
            "outer_reward_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_outer_reward_ci_lower_at_least_minus_0_01",
                    "ge",
                    value(reward, "lower_hex"),
                    -0.01,
                ),
            ),
        ),
        (
            "queried_diversity_vs_mean",
            (
                _float_requirement(
                    "mixed_minus_mean_feature_distance_point_strictly_above_zero",
                    "gt",
                    value(distance, "point_hex"),
                    0.0,
                ),
                _float_requirement(
                    "mixed_minus_mean_unique_components_point_at_least_zero",
                    "ge",
                    value(components, "point_hex"),
                    0.0,
                ),
            ),
        ),
        (
            "brier_vs_no_query_and_random",
            (
                _float_requirement(
                    "mixed_minus_no_query_brier_point_strictly_below_zero",
                    "lt",
                    value(no_query, "point_hex"),
                    0.0,
                ),
                _float_requirement(
                    "mixed_minus_random_brier_point_strictly_below_zero",
                    "lt",
                    value(random, "point_hex"),
                    0.0,
                ),
            ),
        ),
        (
            "guarded_selection_validity",
            (
                _integer_requirement(
                    "guarded_track_count_equals_sixty", "eq", guarded_track_count, 60
                ),
                _integer_requirement(
                    "valid_guarded_track_count_equals_sixty",
                    "eq",
                    valid_guarded_track_count,
                    60,
                ),
            ),
        ),
    )
    _require(
        tuple(item[0] for item in item_requirements) == PROMOTION_ITEM_ORDER,
        "promotion item order changed",
    )
    items = [
        {
            "item_id": item_id,
            "requirements": list(requirements),
            "passed": all(bool(requirement["passed"]) for requirement in requirements),
        }
        for item_id, requirements in item_requirements
    ]
    passed = all(bool(item["passed"]) for item in items)
    return {
        "schema_version": 1,
        "artifact": PROMOTION_DECISION_ARTIFACT,
        "items": items,
        "all_items_required": True,
        "promising_for_prospective_followup": passed,
        "disposition": (
            "promising_for_new_prospective_or_chronological_evaluation" if passed else "v2_no_go"
        ),
        "authorizes_final_50000_library": False,
        "authorizes_final_top_100": False,
        "authorizes_uncertainty_quota": False,
    }


def _reconstruct_finalize_payloads(
    rotation_rows: Sequence[tuple[Mapping[str, object], Mapping[str, float | None]]],
) -> tuple[tuple[tuple[str, bytes], ...], bool]:
    runs = _ordered_runs()
    rotations = _ordered_rotations()
    _require(len(rotation_rows) == 220, "rotation metric reconstruction census changed")
    documents = tuple(dict(row[0]) for row in rotation_rows)
    metric_rows = tuple(dict(row[1]) for row in rotation_rows)

    by_track = {
        str(run["track_id"]): metrics for run, metrics in zip(runs, metric_rows, strict=True)
    }
    policy_rotation_rows: list[dict[str, object]] = []
    policy_rotation_metrics: dict[tuple[str, str], dict[str, float | None]] = {}
    for rotation in rotations:
        rotation_id = str(rotation["rotation_id"])
        for policy in POLICY_ORDER:
            contributing = tuple(
                run for run in runs if run["rotation_id"] == rotation_id and run["policy"] == policy
            )
            expected_count = 5 if policy == "random" else 1
            _require(
                len(contributing) == expected_count, "policy-rotation contribution count changed"
            )
            metrics = _aggregate_metrics(
                [by_track[str(run["track_id"])] for run in contributing],
                label=f"policy rotation {rotation_id} {policy}",
            )
            policy_rotation_metrics[(policy, rotation_id)] = metrics
            track_ids = tuple(str(run["track_id"]) for run in contributing)
            policy_rotation_rows.append(
                {
                    "schema_version": 1,
                    "rotation": dict(rotation),
                    "policy": policy,
                    "contributing_track_count": len(contributing),
                    "contributing_track_ids_sha256": _ordered_id_stream_sha256(track_ids),
                    "metrics": _metric_document(metrics),
                }
            )

    outer_rows: list[dict[str, object]] = []
    outer_metrics: dict[tuple[str, int], dict[str, float | None]] = {}
    for policy in POLICY_ORDER:
        for outer_fold in range(5):
            contributing_rotations = tuple(
                rotation for rotation in rotations if rotation["outer_fold"] == outer_fold
            )
            metrics = _aggregate_metrics(
                [
                    policy_rotation_metrics[(policy, str(rotation["rotation_id"]))]
                    for rotation in contributing_rotations
                ],
                label=f"outer fold {outer_fold} {policy}",
            )
            outer_metrics[(policy, outer_fold)] = metrics
            rotation_ids = tuple(
                str(rotation["rotation_id"]) for rotation in contributing_rotations
            )
            outer_rows.append(
                {
                    "schema_version": 1,
                    "policy": policy,
                    "outer_fold": outer_fold,
                    "contributing_rotation_count": 4,
                    "contributing_rotation_ids_sha256": _ordered_id_stream_sha256(rotation_ids),
                    "metrics": _metric_document(metrics),
                }
            )

    point_rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        metrics = _aggregate_metrics(
            [outer_metrics[(policy, fold)] for fold in range(5)],
            label=f"policy point estimate {policy}",
        )
        point_rows.append(
            {
                "schema_version": 1,
                "policy": policy,
                "outer_fold_unit_count": 5,
                "metrics": _metric_document(metrics),
            }
        )

    comparisons = tuple(
        _paired_comparison_document(
            comparison_id=comparison_id,
            control_policy=control,
            metric=metric,
            direction=direction,
            by_policy_fold=outer_metrics,
        )
        for comparison_id, control, metric, direction in COMPARISON_SPECS
    )
    guarded = tuple(
        document for document in documents if document["run"]["policy"] in GUARDED_POLICIES
    )
    valid_guarded = tuple(
        document
        for document in guarded
        if document["guard_status"] in {"passed", "exact_mean_fallback"}
    )
    decision = _promotion_decision(
        comparisons,
        guarded_track_count=len(guarded),
        valid_guarded_track_count=len(valid_guarded),
    )
    rotation_payload = _canonical_jsonl_bytes(documents)
    policy_rotation_payload = _canonical_jsonl_bytes(policy_rotation_rows)
    outer_payload = _canonical_jsonl_bytes(outer_rows)
    point_payload = _canonical_jsonl_bytes(point_rows)
    comparison_payload = _canonical_jsonl_bytes(comparisons)
    decision_payload = _canonical_json_bytes(decision)
    summary = {
        "schema_version": 1,
        "artifact": FINALIZE_SUMMARY_ARTIFACT,
        "rotation_metric_count": 220,
        "policy_rotation_metric_count": 140,
        "outer_fold_unit_count": 35,
        "policy_point_estimate_count": 7,
        "paired_comparison_count": 7,
        "guarded_track_count": len(guarded),
        "valid_guarded_track_count": len(valid_guarded),
        "bootstrap_indices_sha256": BOOTSTRAP_INDICES_SHA256,
        "rotation_metrics_sha256": _sha256_bytes(rotation_payload),
        "policy_rotation_metrics_sha256": _sha256_bytes(policy_rotation_payload),
        "outer_fold_units_sha256": _sha256_bytes(outer_payload),
        "policy_point_estimates_sha256": _sha256_bytes(point_payload),
        "paired_comparisons_sha256": _sha256_bytes(comparison_payload),
        "promotion_decision_sha256": _sha256_bytes(decision_payload),
        "promising_for_prospective_followup": decision["promising_for_prospective_followup"],
    }
    payloads = (
        ("outer-fold-units.jsonl", outer_payload),
        ("paired-comparisons.jsonl", comparison_payload),
        ("policy-point-estimates.jsonl", point_payload),
        ("policy-rotation-metrics.jsonl", policy_rotation_payload),
        ("promotion-decision.json", decision_payload),
        ("rotation-metrics.jsonl", rotation_payload),
        ("summary.json", _canonical_json_bytes(summary)),
    )
    return payloads, bool(decision["promising_for_prospective_followup"])


def _verify_campaign_pair(
    run_root: Path,
    stage_root: Path,
    *,
    identity: PublicationIdentity,
    authorities: GlobalAuthorities,
    expected_finalize_global_sha256: str | None,
) -> _CampaignResult:
    run_path = Path(os.path.abspath(os.fspath(run_root)))
    stage_path = Path(os.path.abspath(os.fspath(stage_root)))
    _require(run_path != stage_path, "run and stage roots must be distinct")
    _reject_symlink_chain(run_path, label="run root")
    _reject_symlink_chain(stage_path, label="stage root")

    stage = _verify_phase(
        stage_path / "global",
        expected_artifact="sequential_v2_trusted_stage",
        expected_payload_paths=(
            "capability-index.jsonl",
            "policy-runs.jsonl",
            "protocol-census.json",
            "rotations.jsonl",
            "source-anchors.json",
            "stage-summary.json",
        ),
        expected_seal_sha256=authorities.stage,
    )
    global_phases: dict[str, _PhaseEvidence] = {}
    digest_by_phase = {
        "protocol": authorities.protocol,
        "prepare": authorities.prepare,
        "select": authorities.select,
        "reveal": authorities.reveal,
        "update": authorities.update,
        "outer-select": authorities.outer_select,
    }
    for name, (artifact, payloads, metadata_phase) in _global_phase_specs().items():
        root = run_path / name / "global" if name != "protocol" else run_path / "protocol"
        global_phases[name] = _verify_phase(
            root,
            expected_artifact=artifact,
            expected_payload_paths=payloads,
            expected_seal_sha256=digest_by_phase[name],
            expected_metadata=identity.metadata(phase=metadata_phase, scope_id="global"),
        )
    _verify_protocol_payloads(global_phases["protocol"])
    for path in ("policy-runs.jsonl", "protocol-census.json", "rotations.jsonl"):
        _require(
            stage.payloads[path] == global_phases["protocol"].payloads[path],
            f"stage and run protocol payloads differ at {path}",
        )
    _verify_stage_source_anchors(stage)
    _verify_global_anchors(global_phases, authorities)

    stage_outer = _stage_index_authority(stage)
    prediction_view = _prepare_index_authority(global_phases["prepare"])
    commitments = _select_index_authority(global_phases["select"])
    reveal = _reveal_index_authority(global_phases["reveal"], commitments)
    outer_evidence, outer_view = _update_index_authority(global_phases["update"])
    outer_selection = _outer_selection_index_authority(global_phases["outer-select"], outer_view)
    indexed = _IndexedAuthorities(
        stage_outer,
        prediction_view,
        commitments,
        reveal,
        outer_evidence,
        outer_view,
        outer_selection,
    )

    phase_evidence = [stage, *global_phases.values()]
    rotation_outcomes: dict[str, tuple[_Context, ...]] = {}
    rotation_candidates: dict[str, tuple[_PoolCandidate, ...]] = {}
    outer_fold_context_bytes: dict[int, bytes] = {}
    predecessors = {
        "protocol/SHA256SUMS": authorities.protocol,
        "stage/global/SHA256SUMS": authorities.stage,
        "prepare/global/SHA256SUMS": authorities.prepare,
        "select/global/SHA256SUMS": authorities.select,
        "reveal/global/SHA256SUMS": authorities.reveal,
        "update/global/SHA256SUMS": authorities.update,
        "outer-select/global/SHA256SUMS": authorities.outer_select,
    }
    rotation_by_id = {str(row["rotation_id"]): row for row in _ordered_rotations()}
    for rotation_id, rotation in rotation_by_id.items():
        outer_root = stage_path / "rotations" / rotation_id / "outer-outcome-vault"
        outer_phase = _verify_phase(
            outer_root,
            expected_artifact="sequential_v2_stage_outer_outcome_vault",
            expected_payload_paths=("capability.json", "contexts.jsonl"),
            expected_seal_sha256=indexed.stage_outer[rotation_id],
        )
        prediction_root = run_path / "prepare" / "rotations" / rotation_id / "prediction-view"
        prediction_phase = _verify_phase(
            prediction_root,
            expected_artifact="sequential_v2_prepare_prediction_view_v1",
            expected_payload_paths=("candidates.jsonl", "view-summary.json"),
            expected_seal_sha256=indexed.prediction_view[rotation_id],
            expected_metadata=identity.metadata(phase="prepare", scope_id=rotation_id),
        )
        outer_fold = int(rotation["outer_fold"])
        context_bytes = outer_phase.payloads["contexts.jsonl"]
        prior = outer_fold_context_bytes.setdefault(outer_fold, context_bytes)
        _require(prior == context_bytes, "same outer-fold outcome contexts differ across rotations")
        outcomes = _context_rows(
            context_bytes,
            fold=outer_fold,
            label=f"outer outcome {rotation_id}",
        )
        _require(
            len(outcomes) == EXPECTED_CONTEXTS_BY_FOLD[outer_fold],
            f"outer outcome {rotation_id} census changed",
        )
        rotation_outcomes[rotation_id] = outcomes
        rotation_candidates[rotation_id] = _pool_candidates(
            prediction_phase.payloads["candidates.jsonl"], rotation=rotation
        )
        predecessors[f"stage/rotations/{rotation_id}/outer-outcome-vault/SHA256SUMS"] = (
            outer_phase.seal_sha256
        )
        predecessors[f"prepare/rotations/{rotation_id}/prediction-view/SHA256SUMS"] = (
            prediction_phase.seal_sha256
        )
        phase_evidence.extend((outer_phase, prediction_phase))

    reduced: list[tuple[Mapping[str, object], Mapping[str, float | None]]] = []
    for run_index, run in enumerate(_ordered_runs()):
        track_id = str(run["track_id"])
        rotation_id = str(run["rotation_id"])
        reveal_phase = _verify_phase(
            run_path / "reveal" / "tracks" / track_id,
            expected_artifact="sequential_v2_pool_reveal_v1",
            expected_payload_paths=(
                "commitment.json",
                "contexts.jsonl",
                "reveal-summary.json",
                "selected-sequence-ids.jsonl",
            ),
            expected_seal_sha256=indexed.reveal[track_id],
            expected_metadata=identity.metadata(phase="reveal", scope_id=track_id),
        )
        evidence_phase = _verify_phase(
            run_path / "update" / "tracks" / track_id / "outer-evidence",
            expected_artifact="sequential_v2_update_outer_evidence_v1",
            expected_payload_paths=(
                "outer-context-predictions.jsonl",
                "outer-evidence-summary.json",
                "outer-sequence-predictions.jsonl",
            ),
            expected_seal_sha256=indexed.outer_evidence[track_id],
            expected_metadata=identity.metadata(phase="update", scope_id=track_id),
        )
        view_phase = _verify_phase(
            run_path / "update" / "tracks" / track_id / "outer-view",
            expected_artifact="sequential_v2_update_outer_view_v1",
            expected_payload_paths=("candidates.jsonl", "view-summary.json"),
            expected_seal_sha256=indexed.outer_view[track_id],
            expected_metadata=identity.metadata(phase="update", scope_id=track_id),
        )
        selection_phase = _verify_phase(
            run_path / "outer-select" / "tracks" / track_id,
            expected_artifact="sequential_v2_outer_selection_commitment_v1",
            expected_payload_paths=(
                "selected-sequence-ids.jsonl",
                "selection-result.json",
                "selection-summary.json",
            ),
            expected_seal_sha256=indexed.outer_selection[track_id],
            expected_metadata=identity.metadata(phase="outer-select", scope_id=track_id),
        )
        pool_selected, guard_status = _commitment_selection(
            reveal_phase.payloads["commitment.json"],
            run=run,
            selected_payload=reveal_phase.payloads["selected-sequence-ids.jsonl"],
            expected_commitment_sha256=indexed.commitments[track_id][1],
        )
        pool_fold = int(str(rotation_id).split(".pool-", 1)[1])
        pool_contexts = _context_rows(
            reveal_phase.payloads["contexts.jsonl"],
            fold=pool_fold,
            label=f"pool reveal {track_id}",
        )
        outer_outcomes = rotation_outcomes[rotation_id]
        outer_probabilities = _outer_predictions(
            evidence_phase.payloads["outer-context-predictions.jsonl"],
            run=run,
            outcomes=outer_outcomes,
        )
        outer_candidates = _outer_candidates(view_phase.payloads["candidates.jsonl"], run=run)
        _verify_outer_sequence_evidence(
            evidence_phase.payloads["outer-sequence-predictions.jsonl"],
            run=run,
            candidates=outer_candidates,
        )
        rederived_outer = _rederive_outer_selection(outer_candidates)
        published_outer = _selected_ids(
            selection_phase.payloads["selected-sequence-ids.jsonl"],
            expected_count=10,
            label=f"outer selected IDs {track_id}",
        )
        _require(
            published_outer == rederived_outer,
            f"outer selection {track_id} differs from independent mean rederivation",
        )
        selection_summary = _strict_json_object(
            selection_phase.payloads["selection-summary.json"],
            label=f"outer selection summary {track_id}",
        )
        candidate_by_id = {item.sequence_id: item for item in outer_candidates}
        component_counts = Counter(candidate_by_id[item].component for item in published_outer)
        _require(
            type(selection_summary.get("selected_unique_component_count")) is int
            and selection_summary.get("selected_unique_component_count") == len(component_counts)
            and type(selection_summary.get("selected_max_component_occupancy")) is int
            and selection_summary.get("selected_max_component_occupancy")
            == max(component_counts.values())
            and type(selection_summary.get("selected_sequence_count")) is int
            and selection_summary.get("selected_sequence_count") == 10,
            f"outer selection summary {track_id} differs from rederived census",
        )
        reduced.append(
            _rotation_metric_document(
                run=run,
                outer_outcomes=outer_outcomes,
                outer_probabilities=outer_probabilities,
                pool_contexts=pool_contexts,
                pool_selected=pool_selected,
                pool_candidates=rotation_candidates[rotation_id],
                guard_status=guard_status,
                outer_selected=published_outer,
                outer_candidates=outer_candidates,
            )
        )
        predecessors[f"reveal/tracks/{track_id}/SHA256SUMS"] = reveal_phase.seal_sha256
        predecessors[f"update/tracks/{track_id}/outer-evidence/SHA256SUMS"] = (
            evidence_phase.seal_sha256
        )
        predecessors[f"update/tracks/{track_id}/outer-view/SHA256SUMS"] = view_phase.seal_sha256
        predecessors[f"outer-select/tracks/{track_id}/SHA256SUMS"] = selection_phase.seal_sha256
        phase_evidence.extend((reveal_phase, evidence_phase, view_phase, selection_phase))
        _require(len(reduced) == run_index + 1, "rotation metric reconstruction order changed")

    _require(len(predecessors) == 927, "finalize predecessor census differs from 927")
    reconstructed, promising = _reconstruct_finalize_payloads(reduced)
    finalize_phase = _verify_phase(
        run_path / "finalize" / "global",
        expected_artifact=FINALIZE_ARTIFACT,
        expected_payload_paths=FINALIZE_PAYLOAD_PATHS,
        expected_seal_sha256=expected_finalize_global_sha256,
        expected_predecessors=predecessors,
        expected_metadata=identity.metadata(phase="finalize", scope_id="global"),
    )
    for path, expected_payload in reconstructed:
        _require(
            finalize_phase.payloads[path] == expected_payload,
            f"finalize payload {path} differs from independent reconstruction",
        )
    phase_evidence.append(finalize_phase)
    _require(len(phase_evidence) == 928, "authenticated phase evidence census changed")
    return _CampaignResult(
        finalize_phase,
        reconstructed,
        tuple(phase_evidence),
        promising,
    )


def _root_identity(path: Path, *, label: str) -> tuple[int, int]:
    requested = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_chain(requested, label=label)
    try:
        metadata = os.lstat(requested)
    except OSError as error:
        raise VerificationError(f"cannot inspect {label}: {requested}") from error
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be a real directory: {requested}",
    )
    return metadata.st_dev, metadata.st_ino


def _publish_receipt(receipt_dir: Path, payload: bytes) -> tuple[Path, str]:
    requested = Path(os.path.abspath(os.fspath(receipt_dir)))
    _require(requested.name not in {"", ".", ".."}, "receipt directory is invalid")
    _reject_symlink_chain(requested.parent, label="receipt parent")
    try:
        parent_metadata = os.lstat(requested.parent)
    except OSError as error:
        raise VerificationError(f"cannot inspect receipt parent: {requested.parent}") from error
    _require(
        stat.S_ISDIR(parent_metadata.st_mode) and not stat.S_ISLNK(parent_metadata.st_mode),
        f"receipt parent must be a real directory: {requested.parent}",
    )
    try:
        os.mkdir(requested, mode=0o700)
    except FileExistsError as error:
        raise VerificationError(f"receipt directory already exists: {requested}") from error
    except OSError as error:
        raise VerificationError(f"cannot create receipt directory: {requested}") from error

    receipt_path = requested / "independent-verification.json"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(receipt_path, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "receipt write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except OSError as error:
        raise VerificationError(f"cannot publish independent receipt: {receipt_path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    directory_descriptor = -1
    try:
        directory_descriptor = os.open(
            requested,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(directory_descriptor)
        os.fchmod(directory_descriptor, 0o555)
        os.fsync(directory_descriptor)
    except OSError as error:
        raise VerificationError(f"cannot seal receipt directory: {requested}") from error
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)

    directory_metadata = os.lstat(requested)
    _require(
        stat.S_ISDIR(directory_metadata.st_mode)
        and not stat.S_ISLNK(directory_metadata.st_mode)
        and stat.S_IMODE(directory_metadata.st_mode) == 0o555,
        "independent receipt directory did not seal mode 0555",
    )
    with os.scandir(requested) as iterator:
        inventory = tuple(sorted(entry.name for entry in iterator))
    _require(
        inventory == ("independent-verification.json",),
        "independent receipt directory inventory changed during publication",
    )
    snapshot = _snapshot_file(receipt_path, label="published independent receipt")
    _require(snapshot.payload == payload, "published independent receipt bytes changed")
    return receipt_path, snapshot.sha256


def verify_sequential_v2_finalize(
    *,
    run_twins: Sequence[str | Path],
    stage_twins: Sequence[str | Path],
    publication_identity: PublicationIdentity,
    global_authorities: GlobalAuthorities,
    receipt_dir: str | Path,
    expected_finalize_global_sha256: str | None = None,
) -> VerificationExecution:
    """Verify one or two finalization publications and emit a partial receipt.

    This is deliberately a finalize-only verifier.  It independently rebuilds
    final metrics and reductions from authenticated predecessor leaves, but it
    does not refit update models or replay acquisition-policy selections.
    """

    run_paths = tuple(Path(path) for path in run_twins)
    stage_paths = tuple(Path(path) for path in stage_twins)
    _require(
        len(run_paths) == len(stage_paths) and len(run_paths) in {1, 2},
        "provide one or two occurrence-paired --run-twin/--stage-twin roots",
    )
    if expected_finalize_global_sha256 is not None:
        _sha256(expected_finalize_global_sha256, label="expected finalize global seal")

    run_identities = tuple(
        _root_identity(path, label=f"run twin {index}") for index, path in enumerate(run_paths)
    )
    stage_identities = tuple(
        _root_identity(path, label=f"stage twin {index}") for index, path in enumerate(stage_paths)
    )
    if len(run_paths) == 2:
        _require(
            len(set(run_identities)) == 2,
            "the two run twins must be distinct directory objects",
        )
        _require(
            len(set(stage_identities)) == 2,
            "the two stage twins must be distinct directory objects",
        )

    results = tuple(
        _verify_campaign_pair(
            run_root=run_path,
            stage_root=stage_path,
            identity=publication_identity,
            authorities=global_authorities,
            expected_finalize_global_sha256=expected_finalize_global_sha256,
        )
        for run_path, stage_path in zip(run_paths, stage_paths, strict=True)
    )
    reference = results[0]
    twin_byte_identity_required = len(results) == 2
    if twin_byte_identity_required:
        peer = results[1]
        _require(
            reference.finalize.file_payloads == peer.finalize.file_payloads,
            "the two finalize/global publications are not byte-identical",
        )
        _require(
            reference.reconstructed_payloads == peer.reconstructed_payloads,
            "the two independent finalize reconstructions are not byte-identical",
        )
        _require(
            reference.promising == peer.promising,
            "the two independently reconstructed prospective decisions differ",
        )

    for result in results:
        for phase in result.phase_evidence:
            _assert_phase_unchanged(phase)

    payload_sha256 = {
        path: _sha256_bytes(payload) for path, payload in reference.reconstructed_payloads
    }
    receipt = {
        "artifact": VERIFICATION_ARTIFACT,
        "automatic_benchmark_acceptance": False,
        "checks": {
            "accepted_gate1_source_anchors_match_frozen_pins": True,
            "all_220_rotation_metrics_reconstructed_from_authenticated_finalize_inputs": True,
            "all_seven_finalize_payloads_reconstructed_byte_for_byte": True,
            "all_consumed_leaf_phase_manifests_verified": True,
            "authenticated_phase_trees_rechecked_after_reconstruction": True,
            "exact_927_predecessor_closure_verified": True,
            "global_indices_bind_consumed_leaves": True,
            "outer_mean_selections_independently_rederived": True,
            "outer_sequence_evidence_matches_selector_view": True,
            "publication_equality_requirement_satisfied": True,
            "upstream_global_authorities_verified": True,
        },
        "counts": {
            "authenticated_leaf_phases_per_publication": 920,
            "authenticated_markers_per_publication": 928,
            "direct_finalize_predecessors": 927,
            "finalize_payloads": 7,
            "outer_fold_unit_rows": 35,
            "paired_comparisons": 7,
            "policy_point_estimates": 7,
            "policy_rotation_metric_rows": 140,
            "rotation_metric_rows": 220,
        },
        "excluded_verifier_item_3_satisfied": False,
        "external_global_authorities": global_authorities.document(),
        "finalize_global_seal_sha256": reference.finalize.seal_sha256,
        "full_prepare_through_decision_reconstruction": False,
        "limitations": [
            (
                "220 update-state leaf digests were globally bound, but their phase files were "
                "not opened and their model refits were not independently rerun"
            ),
            "pool acquisition-policy selections were authenticated but not independently replayed",
            (
                "pool-outcome-vault projections and full guarded-selection numerics were "
                "authenticated upstream but not independently rebuilt"
            ),
            (
                "outer sequence-evidence IDs and objective probabilities were cross-checked to "
                "selector views, but component assignments, component-source leaves, and view "
                "generation were not independently rebuilt"
            ),
            (
                "nonmetric payloads, including selection-result and selection-summary documents, "
                "and nonessential index attestation fields were seal-bound and authenticated but "
                "not fully semantically reconstructed"
            ),
        ],
        "payload_sha256": payload_sha256,
        "publication_count": len(results),
        "publication_identity": publication_identity.document(),
        "reconstructed_finalize_disposition": {
            "promising_for_prospective_followup": reference.promising,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed_finalize_only_partial_verification",
        "twin_byte_identity": {
            "required": twin_byte_identity_required,
            "verified": twin_byte_identity_required,
        },
        "verification_scope": "finalize_only_from_authenticated_predecessor_leaves",
    }
    receipt_payload = _canonical_json_bytes(receipt)
    published_path, receipt_sha256 = _publish_receipt(Path(receipt_dir), receipt_payload)
    return VerificationExecution(
        receipt_dir=published_path.parent,
        receipt_path=published_path,
        receipt_sha256=receipt_sha256,
        finalize_global_seal_sha256=reference.finalize.seal_sha256,
        publication_count=len(results),
        promising_for_prospective_followup=reference.promising,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently reconstruct sequential-v2 finalize/global publications. "
            "This finalize-only partial verifier does not rerun model refits or pool acquisition."
        )
    )
    parser.add_argument(
        "--run-twin",
        action="append",
        required=True,
        help="campaign root; repeat once for each occurrence-paired twin",
    )
    parser.add_argument(
        "--stage-twin",
        action="append",
        required=True,
        help="stage root; repeat once for each occurrence-paired twin",
    )
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-stage-global-sha256", required=True)
    parser.add_argument("--expected-prepare-global-sha256", required=True)
    parser.add_argument("--expected-select-global-sha256", required=True)
    parser.add_argument("--expected-reveal-global-sha256", required=True)
    parser.add_argument("--expected-update-global-sha256", required=True)
    parser.add_argument("--expected-outer-select-global-sha256", required=True)
    parser.add_argument("--expected-finalize-global-sha256")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--code-manifest-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--receipt-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execution = verify_sequential_v2_finalize(
            run_twins=tuple(args.run_twin),
            stage_twins=tuple(args.stage_twin),
            publication_identity=PublicationIdentity(
                git_commit=args.git_commit,
                code_manifest_sha256=args.code_manifest_sha256,
                config_sha256=args.config_sha256,
                lock_sha256=args.lock_sha256,
            ),
            global_authorities=GlobalAuthorities(
                protocol=args.expected_protocol_sha256,
                stage=args.expected_stage_global_sha256,
                prepare=args.expected_prepare_global_sha256,
                select=args.expected_select_global_sha256,
                reveal=args.expected_reveal_global_sha256,
                update=args.expected_update_global_sha256,
                outer_select=args.expected_outer_select_global_sha256,
            ),
            receipt_dir=args.receipt_dir,
            expected_finalize_global_sha256=args.expected_finalize_global_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"independent sequential-v2 finalize verification failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": VERIFICATION_ARTIFACT,
                "finalize_global_seal_sha256": execution.finalize_global_seal_sha256,
                "publication_count": execution.publication_count,
                "receipt_path": str(execution.receipt_path),
                "receipt_sha256": execution.receipt_sha256,
                "status": "passed_finalize_only_partial_verification",
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "GlobalAuthorities",
    "PublicationIdentity",
    "VerificationError",
    "VerificationExecution",
    "build_parser",
    "main",
    "verify_sequential_v2_finalize",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

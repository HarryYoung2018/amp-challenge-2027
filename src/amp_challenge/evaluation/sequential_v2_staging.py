"""Authenticate and partition the frozen Gate-1 source for sequential replay v2.

This module is the read boundary for the accepted context-level Gate-1 evidence.
It deliberately does not expose the historical OOF probabilities: sequential v2
refits its descriptor model from the authenticated context examples.  A later CLI
may serialize the returned capability slices, but selectors must never receive an
``AuthenticatedGate1Source`` or an outcome-vault object directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from amp_challenge.evaluation.sequential_v2_primitives import TARGET_GRAM, ContextRow
from amp_challenge.evaluation.sequential_v2_protocol import RotationSpec
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ASSIGNMENT_POLICY = (
    "reuse_accepted_homology_study_union_sequence_assignments_without_reassignment_v1"
)
_GATE1_ARTIFACT = "gate1_context_activity_homology_study_union_v1"
_GATE1_STATUS = "development_evidence_not_an_untouched_evaluation_panel"
_RECEIPT_ARTIFACT = "gate1_context_activity_homology_study_union_v1_independent_verification"
_SPLIT_RECEIPT_ARTIFACT = "gate1_union_accepted_split_consumption_receipt"
_FOLDS_ARTIFACT = "gate1_context_union_fold_reuse"

_SEMANTIC_FILES = (
    "context_audit.jsonl",
    "examples.jsonl",
    "folds.json",
    "manifest.json",
    "metrics.json",
    "oof_predictions.csv",
    "split_receipt.json",
)
_ROOT_FILES = (
    "CODE_SHA256SUMS",
    "FROZEN_INPUT_SHA256SUMS",
    "SHA256SUMS",
)
_EXAMPLE_FIELDS = frozenset(
    {
        "assay_context_id",
        "canonical_target",
        "example_id",
        "fold",
        "gram",
        "homology_component_id",
        "label",
        "schema_version",
        "sequence",
        "sequence_id",
        "source_observations",
        "union_component_id",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "example_id",
        "fold",
        "homology_component_id",
        "sequence_id",
        "union_component_id",
    }
)


@dataclass(frozen=True, slots=True)
class Gate1SourceContract:
    """Content and census contract for one immutable Gate-1 publication."""

    producer_job_id: int
    audit_job_id: int
    git_commit: str
    config_sha256: str
    publication_top_sha256: str
    semantic_top_sha256: str
    code_manifest_sha256: str
    frozen_input_manifest_sha256: str
    independent_receipt_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    expected_contexts: int
    expected_sequences: int
    expected_examples_by_fold: tuple[int, int, int, int, int]
    expected_sequences_by_fold: tuple[int, int, int, int, int]
    expected_negatives: int
    expected_positives: int
    expected_source_observations: int
    expected_panel_homology_components: int
    expected_panel_union_components: int
    expected_split_homology_components: int
    expected_split_union_components: int
    expected_support_sequences: int
    expected_support_by_fold: tuple[int, int, int, int, int]
    expected_support_ids_sha256: str
    expected_support_contexts: int
    expected_support_source_observations: int
    expected_support_unique_target_counts_2_to_7: tuple[int, int, int, int, int, int]
    maximum_cross_fold_identity: float

    def __post_init__(self) -> None:
        if self.producer_job_id <= 0 or self.audit_job_id <= 0:
            raise ValueError("Gate-1 job IDs must be positive")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None:
            raise ValueError("Gate-1 git_commit must be a full lowercase Git SHA")
        digests = (
            self.config_sha256,
            self.publication_top_sha256,
            self.semantic_top_sha256,
            self.code_manifest_sha256,
            self.frozen_input_manifest_sha256,
            self.independent_receipt_sha256,
            self.expected_support_ids_sha256,
            *(digest for _, digest in self.artifact_sha256),
        )
        if any(_SHA256.fullmatch(value) is None for value in digests):
            raise ValueError("Gate-1 contract contains an invalid SHA-256 digest")
        artifact_names = tuple(name for name, _ in self.artifact_sha256)
        if artifact_names != _SEMANTIC_FILES:
            raise ValueError("Gate-1 contract semantic artifact inventory is not exact")
        if any(value < 0 for value in self.expected_examples_by_fold):
            raise ValueError("Gate-1 fold censuses cannot be negative")
        if sum(self.expected_examples_by_fold) != self.expected_contexts:
            raise ValueError("Gate-1 per-fold context census does not sum to its total")
        if sum(self.expected_sequences_by_fold) != self.expected_sequences:
            raise ValueError("Gate-1 per-fold sequence census does not sum to its total")
        if sum(self.expected_support_by_fold) != self.expected_support_sequences:
            raise ValueError("Gate-1 per-fold support census does not sum to its total")

    @property
    def artifact_hashes(self) -> Mapping[str, str]:
        return dict(self.artifact_sha256)


ACCEPTED_GATE1_SOURCE_CONTRACT = Gate1SourceContract(
    producer_job_id=223248,
    audit_job_id=223250,
    git_commit="0468cc2cbc0b7c3b2a50f7866da1b1083c8ef1a8",
    config_sha256="e4e420501efc1bf89c2b1fbd15db5adc1340d3c42963c005bf1eced23c9d0410",
    publication_top_sha256="4e259cfb43033c069598d42fd54fec49a67ba55bfbb5c1e77eeef4887815fe2f",
    semantic_top_sha256="556e06fd2b1af1e678de88008bc1b87434fb8cf1179f38c897de7ee2c9fd779d",
    code_manifest_sha256="f770a656c4f300855c129ebc044cc753c3bd119cb0476ef14eae277f5b104e72",
    frozen_input_manifest_sha256=(
        "229e10ad276430136671cb7831f44e50c5a6d6a6f09ef3f16e58129771f53e60"
    ),
    independent_receipt_sha256=("1f6a6ce811d3fbecdbbbe7463e6052dc7766130372926d29952c70be65743e81"),
    artifact_sha256=(
        ("context_audit.jsonl", "9886928e195543eb5725bcb1c1b35b9c3d29eec555e339b0be28a12028813900"),
        ("examples.jsonl", "d3eecbf3014fd78cf7021818466893d292315b6e90e77cea85d4e1fbd5bec520"),
        ("folds.json", "37e035d488634e50ad95dfefe01d235474867aeb96b8205b543d22c25fdc9988"),
        ("manifest.json", "b1cf4a2ad396d26776a824d4ec3cc077ba45bf9d2d47e00f8f2f482053b1f78e"),
        ("metrics.json", "e06b1893572309b2175fa4c9c1743af2fbb2d4006cdae9a7c452fcd4dba2563b"),
        (
            "oof_predictions.csv",
            "11eea907a242606b78261a9b37107647ce393db743be95c146cbbe5caced137c",
        ),
        ("split_receipt.json", "63cf02d87cd4a9b9ec1afe9c40c2e0914fc75515a14a21e71f1e50dce0fd01a7"),
    ),
    expected_contexts=2492,
    expected_sequences=952,
    expected_examples_by_fold=(546, 486, 485, 487, 488),
    expected_sequences_by_fold=(271, 159, 176, 170, 176),
    expected_negatives=761,
    expected_positives=1731,
    expected_source_observations=2592,
    expected_panel_homology_components=485,
    expected_panel_union_components=213,
    expected_split_homology_components=597,
    expected_split_union_components=278,
    expected_support_sequences=650,
    expected_support_by_fold=(202, 126, 112, 97, 113),
    expected_support_ids_sha256=(
        "990ef4b248f7d95a9864fef7da5bc648a5e6468c1adc74ba87c3af38dedb21c9"
    ),
    expected_support_contexts=1967,
    expected_support_source_observations=2062,
    expected_support_unique_target_counts_2_to_7=(364, 213, 56, 16, 0, 1),
    maximum_cross_fold_identity=0.7777777777777778,
)


@dataclass(frozen=True, slots=True)
class LabelFreeContext:
    """Context metadata with all outcome and observation-count fields removed."""

    example_id: str
    assay_context_id: str
    sequence_id: str
    sequence: str
    target: str
    gram: str
    fold: int


@dataclass(frozen=True, slots=True)
class Gate1Panel:
    """Validated contexts plus deterministic fold/support indexes."""

    contexts: tuple[ContextRow, ...]
    contexts_by_fold: tuple[tuple[ContextRow, ...], ...]
    support_sequence_ids: tuple[str, ...]
    support_sequence_ids_by_fold: tuple[tuple[str, ...], ...]
    support_sequence_ids_sha256: str


@dataclass(frozen=True, slots=True)
class RotationCapabilityIndex:
    """Outcome-free exact example-ID capabilities for one rotation."""

    spec: RotationSpec
    base_example_ids: tuple[str, ...]
    acquisition_example_ids: tuple[str, ...]
    outer_example_ids: tuple[str, ...]
    acquisition_support_sequence_ids: tuple[str, ...]
    outer_support_sequence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrepareCapability:
    """The only source material a prepare worker needs for one rotation."""

    spec: RotationSpec
    base_contexts: tuple[ContextRow, ...]
    acquisition_metadata: tuple[LabelFreeContext, ...]
    acquisition_support_sequence_ids: tuple[str, ...]
    allowed_base_example_ids: tuple[str, ...]
    allowed_acquisition_metadata_example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoolOutcomeVault:
    """All and only acquisition-fold outcomes for a rotation's reveal custodian."""

    spec: RotationSpec
    contexts: tuple[ContextRow, ...]
    allowed_example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OuterMetadataCapability:
    """Label-free outer-fold contexts made available only after model update."""

    spec: RotationSpec
    contexts: tuple[LabelFreeContext, ...]
    allowed_example_ids: tuple[str, ...]
    support_sequence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OuterOutcomeVault:
    """Outer outcomes reserved exclusively for the finalization custodian."""

    spec: RotationSpec
    contexts: tuple[ContextRow, ...]
    allowed_example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceFileEvidence:
    logical_path: str
    sha256: str
    size_bytes: int
    mode: str


@dataclass(frozen=True, slots=True)
class Gate1AuthenticationEvidence:
    producer_job_id: int
    audit_job_id: int
    files: tuple[SourceFileEvidence, ...]
    independent_receipt_sha256: str
    twins_byte_identical: bool = True

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact": "sequential_v2_authenticated_gate1_source",
            "producer_job_id": self.producer_job_id,
            "audit_job_id": self.audit_job_id,
            "twins_byte_identical": self.twins_byte_identical,
            "independent_receipt_sha256": self.independent_receipt_sha256,
            "files": {
                item.logical_path: {
                    "mode": item.mode,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            },
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedGate1Source:
    panel: Gate1Panel
    evidence: Gate1AuthenticationEvidence


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    mode: int
    fingerprint: tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _DirectorySnapshot:
    path: Path
    entries: tuple[str, ...]
    fingerprint: tuple[int, int, int, int, int, int, int, int, int]


def _fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
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


def _reject_symlink_chain(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in reversed((absolute, *absolute.parents)):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect {label}: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} cannot traverse a symbolic link: {candidate}")
    return absolute


def _require_directory(path: Path, *, label: str, mode: int) -> Path:
    absolute = _reject_symlink_chain(path, label=label)
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {absolute}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symbolic directory")
    observed_mode = stat.S_IMODE(metadata.st_mode)
    if observed_mode != mode:
        raise ValueError(f"{label} must be mode {mode:04o}, observed {observed_mode:04o}")
    return absolute


def _snapshot_directory(path: Path, *, label: str, mode: int) -> _DirectorySnapshot:
    absolute = _require_directory(path, label=label, mode=mode)
    before = os.lstat(absolute)
    try:
        entries = tuple(sorted(entry.name for entry in os.scandir(absolute)))
    except OSError as error:
        raise ValueError(f"cannot inventory {label}: {absolute}") from error
    after = os.lstat(absolute)
    if _fingerprint(before) != _fingerprint(after):
        raise ValueError(f"{label} changed while its inventory was read")
    return _DirectorySnapshot(
        path=absolute,
        entries=entries,
        fingerprint=_fingerprint(before),
    )


def _assert_directory_unchanged(snapshot: _DirectorySnapshot, *, label: str) -> None:
    _reject_symlink_chain(snapshot.path, label=label)
    try:
        metadata = os.lstat(snapshot.path)
        entries = tuple(sorted(entry.name for entry in os.scandir(snapshot.path)))
    except OSError as error:
        raise ValueError(f"{label} changed after authentication") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or _fingerprint(metadata) != snapshot.fingerprint
        or entries != snapshot.entries
    ):
        raise ValueError(f"{label} changed after authentication")


def _snapshot_regular(path: Path, *, label: str, mode: int) -> _Snapshot:
    source = _reject_symlink_chain(path, label=label)
    try:
        named_before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {source}") from error
    observed_mode = stat.S_IMODE(named_before.st_mode)
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError(f"{label} must be a regular non-symbolic file")
    if observed_mode != mode:
        raise ValueError(f"{label} must be mode {mode:04o}, observed {observed_mode:04o}")
    if named_before.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"cannot open {label}: {source}") from error
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    identities = {
        _fingerprint(item) for item in (named_before, opened_before, opened_after, named_after)
    }
    _reject_symlink_chain(source, label=label)
    if len(identities) != 1:
        raise ValueError(f"{label} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != named_before.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return _Snapshot(
        path=source,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        mode=observed_mode,
        fingerprint=_fingerprint(named_before),
    )


def _assert_unchanged(snapshot: _Snapshot, *, label: str) -> None:
    _reject_symlink_chain(snapshot.path, label=label)
    try:
        metadata = os.lstat(snapshot.path)
    except OSError as error:
        raise ValueError(f"{label} disappeared after authentication") from error
    if not stat.S_ISREG(metadata.st_mode) or _fingerprint(metadata) != snapshot.fingerprint:
        raise ValueError(f"{label} changed after authentication")


def _strict_json(payload: bytes, *, label: str) -> object:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must use LF framing with exactly one final LF")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value in {label}: {value}")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json(payload, label=label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _canonical_json_bytes(value: object) -> bytes:
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


def _canonical_jsonl(payload: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty canonical LF-framed JSONL")
    rows: list[Mapping[str, Any]] = []
    for number, line in enumerate(payload.splitlines(keepends=True), start=1):
        value = _json_object(line, label=f"{label} row {number}")
        if _canonical_json_bytes(value) != line:
            raise ValueError(f"{label} row {number} is not canonical compact JSON")
        rows.append(value)
    return tuple(rows)


def _parse_sha256sums(payload: bytes, *, label: str) -> Mapping[str, str]:
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n") or b"\r" in payload:
        raise ValueError(f"{label} must be non-empty LF-framed checksum text")
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must contain ASCII only") from error
    result: dict[str, str] = {}
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        if match is None:
            raise ValueError(f"{label} line {number} is not a strict SHA-256 entry")
        digest, name = match.groups()
        logical = PurePosixPath(name)
        if logical.is_absolute() or ".." in logical.parts or name in {"", "."}:
            raise ValueError(f"{label} line {number} has an unsafe path")
        if previous is not None and name <= previous:
            raise ValueError(f"{label} paths must be unique and strictly sorted")
        result[name] = digest
        previous = name
    return result


def _require_exact_fields(
    value: object, *, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    document = cast(dict[str, Any], value)
    missing = expected - set(document)
    extra = set(document) - expected
    if missing or extra:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return document


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(cast(str, value)) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _parse_examples(payload: bytes) -> tuple[tuple[ContextRow, str, str], ...]:
    rows = _canonical_jsonl(payload, label="Gate-1 examples")
    parsed: list[tuple[ContextRow, str, str]] = []
    previous_id: str | None = None
    seen_ids: set[str] = set()
    for number, raw in enumerate(rows, start=1):
        row = _require_exact_fields(raw, expected=_EXAMPLE_FIELDS, label=f"Gate-1 example {number}")
        if row["schema_version"] != 1 or isinstance(row["schema_version"], bool):
            raise ValueError(f"Gate-1 example {number} schema_version must be integer one")
        example_id = _require_sha256(row["example_id"], label=f"Gate-1 example {number} ID")
        assay_id = _require_sha256(
            row["assay_context_id"], label=f"Gate-1 example {number} assay context ID"
        )
        if assay_id != example_id:
            raise ValueError("Gate-1 example_id must exactly equal assay_context_id")
        if example_id in seen_ids or (previous_id is not None and example_id <= previous_id):
            raise ValueError("Gate-1 examples must be unique and strictly ordered by example_id")
        if type(row["sequence"]) is not str:
            raise ValueError(f"Gate-1 example {number} sequence must be a string")
        sequence = cast(str, row["sequence"])
        try:
            canonical = canonicalize_sequence(sequence)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Gate-1 example {number} has an invalid sequence") from error
        if canonical != sequence:
            raise ValueError(f"Gate-1 example {number} sequence is not canonical")
        sequence_id = _require_sha256(
            row["sequence_id"], label=f"Gate-1 example {number} sequence ID"
        )
        if canonical_sequence_id(sequence) != sequence_id:
            raise ValueError(f"Gate-1 example {number} sequence ID does not match sequence")
        target = row["canonical_target"]
        gram = row["gram"]
        if type(target) is not str or target not in TARGET_GRAM or gram != TARGET_GRAM[target]:
            raise ValueError(f"Gate-1 example {number} target/Gram mapping is invalid")
        label_value = _require_int(row["label"], label=f"Gate-1 example {number} label")
        if label_value not in {0, 1}:
            raise ValueError(f"Gate-1 example {number} label must be zero or one")
        fold = _require_int(row["fold"], label=f"Gate-1 example {number} fold")
        if fold >= 5:
            raise ValueError(f"Gate-1 example {number} fold must be in [0, 4]")
        observations = _require_int(
            row["source_observations"],
            label=f"Gate-1 example {number} source observations",
            minimum=1,
        )
        homology_id = _require_sha256(
            row["homology_component_id"],
            label=f"Gate-1 example {number} homology component",
        )
        union_id = _require_sha256(
            row["union_component_id"], label=f"Gate-1 example {number} union component"
        )
        parsed.append(
            (
                ContextRow(
                    example_id=example_id,
                    assay_context_id=assay_id,
                    sequence_id=sequence_id,
                    sequence=sequence,
                    target=cast(str, target),
                    gram=cast(str, gram),
                    fold=fold,
                    label=label_value,
                    source_observations=observations,
                ),
                homology_id,
                union_id,
            )
        )
        previous_id = example_id
        seen_ids.add(example_id)
    return tuple(parsed)


def _validate_fold_join(
    payload: bytes,
    *,
    parsed: Sequence[tuple[ContextRow, str, str]],
    contract: Gate1SourceContract,
) -> None:
    document = _json_object(payload, label="Gate-1 folds")
    if (
        document.get("schema_version") != 1
        or document.get("artifact") != _FOLDS_ARTIFACT
        or document.get("assignment_policy") != _ASSIGNMENT_POLICY
        or document.get("identity_threshold") != 0.8
        or document.get("maximum_cross_fold_identity") != contract.maximum_cross_fold_identity
    ):
        raise ValueError("Gate-1 folds identity contract is invalid")
    assignments = document.get("assignments")
    if type(assignments) is not list or len(assignments) != len(parsed):
        raise ValueError("Gate-1 folds assignments do not match the context census")
    for number, (raw, expected) in enumerate(zip(assignments, parsed, strict=True), start=1):
        assignment = _require_exact_fields(
            raw, expected=_ASSIGNMENT_FIELDS, label=f"Gate-1 fold assignment {number}"
        )
        assignment_example_id = _require_sha256(
            assignment["example_id"], label=f"Gate-1 fold assignment {number} example ID"
        )
        assignment_sequence_id = _require_sha256(
            assignment["sequence_id"], label=f"Gate-1 fold assignment {number} sequence ID"
        )
        assignment_homology_id = _require_sha256(
            assignment["homology_component_id"],
            label=f"Gate-1 fold assignment {number} homology component",
        )
        assignment_union_id = _require_sha256(
            assignment["union_component_id"],
            label=f"Gate-1 fold assignment {number} union component",
        )
        assignment_fold = _require_int(
            assignment["fold"], label=f"Gate-1 fold assignment {number} fold"
        )
        if assignment_fold >= 5:
            raise ValueError(f"Gate-1 fold assignment {number} fold must be in [0, 4]")
        context, homology_id, union_id = expected
        expected_assignment = {
            "example_id": context.example_id,
            "fold": context.fold,
            "homology_component_id": homology_id,
            "sequence_id": context.sequence_id,
            "union_component_id": union_id,
        }
        validated_assignment = {
            "example_id": assignment_example_id,
            "fold": assignment_fold,
            "homology_component_id": assignment_homology_id,
            "sequence_id": assignment_sequence_id,
            "union_component_id": assignment_union_id,
        }
        if validated_assignment != expected_assignment:
            raise ValueError(f"Gate-1 fold assignment {number} differs from examples JSONL")


def _support_ids_sha256(ids: Iterable[str]) -> str:
    ordered = tuple(sorted(ids))
    return hashlib.sha256("".join(f"{value}\n" for value in ordered).encode("ascii")).hexdigest()


def parse_gate1_panel(
    examples_payload: bytes,
    folds_payload: bytes,
    *,
    contract: Gate1SourceContract = ACCEPTED_GATE1_SOURCE_CONTRACT,
) -> Gate1Panel:
    """Parse authenticated example/fold bytes and recompute every v2 support invariant."""

    parsed = _parse_examples(examples_payload)
    if len(parsed) != contract.expected_contexts:
        raise ValueError(
            f"Gate-1 expected {contract.expected_contexts} contexts, observed {len(parsed)}"
        )
    _validate_fold_join(folds_payload, parsed=parsed, contract=contract)
    contexts = tuple(item[0] for item in parsed)
    if {row.target for row in contexts} != set(TARGET_GRAM):
        raise ValueError("Gate-1 contexts do not cover the exact seven-target panel")
    labels = Counter(row.label for row in contexts)
    if (labels[0], labels[1]) != (contract.expected_negatives, contract.expected_positives):
        raise ValueError("Gate-1 label census changed")
    if sum(row.source_observations for row in contexts) != contract.expected_source_observations:
        raise ValueError("Gate-1 source-observation census changed")

    by_fold_lists: list[list[ContextRow]] = [[] for _ in range(5)]
    by_sequence: dict[str, list[tuple[ContextRow, str, str]]] = defaultdict(list)
    component_fold: dict[tuple[str, str], int] = {}
    for context, homology_id, union_id in parsed:
        by_fold_lists[context.fold].append(context)
        by_sequence[context.sequence_id].append((context, homology_id, union_id))
        for kind, component_id in (("homology", homology_id), ("union", union_id)):
            previous = component_fold.setdefault((kind, component_id), context.fold)
            if previous != context.fold:
                raise ValueError(f"Gate-1 {kind} component crosses folds")
    examples_by_fold = tuple(len(rows) for rows in by_fold_lists)
    if examples_by_fold != contract.expected_examples_by_fold:
        raise ValueError("Gate-1 per-fold context census changed")
    if len(by_sequence) != contract.expected_sequences:
        raise ValueError("Gate-1 modeled-sequence census changed")
    if len({item[1] for item in parsed}) != contract.expected_panel_homology_components:
        raise ValueError("Gate-1 homology-component census changed")
    if len({item[2] for item in parsed}) != contract.expected_panel_union_components:
        raise ValueError("Gate-1 union-component census changed")

    sequence_counts = Counter()
    support_by_fold_lists: list[list[str]] = [[] for _ in range(5)]
    support_ids: list[str] = []
    support_contexts = 0
    support_observations = 0
    support_target_counts: Counter[int] = Counter()
    for sequence_id in sorted(by_sequence):
        rows = by_sequence[sequence_id]
        metadata = {
            (row.sequence, row.fold, homology_id, union_id) for row, homology_id, union_id in rows
        }
        if len(metadata) != 1:
            raise ValueError(f"Gate-1 sequence {sequence_id} changes sequence or fold metadata")
        fold = rows[0][0].fold
        sequence_counts[fold] += 1
        grams = {row.gram for row, _, _ in rows}
        if grams == {"positive", "negative"}:
            support_ids.append(sequence_id)
            support_by_fold_lists[fold].append(sequence_id)
            support_contexts += len(rows)
            support_observations += sum(row.source_observations for row, _, _ in rows)
            support_target_counts[len({row.target for row, _, _ in rows})] += 1
    if tuple(sequence_counts[fold] for fold in range(5)) != contract.expected_sequences_by_fold:
        raise ValueError("Gate-1 per-fold modeled-sequence census changed")
    if len(support_ids) != contract.expected_support_sequences:
        raise ValueError("Gate-1 support-eligible sequence census changed")
    support_by_fold = tuple(tuple(sorted(values)) for values in support_by_fold_lists)
    if tuple(len(values) for values in support_by_fold) != contract.expected_support_by_fold:
        raise ValueError("Gate-1 per-fold support census changed")
    support_digest = _support_ids_sha256(support_ids)
    if support_digest != contract.expected_support_ids_sha256:
        raise ValueError("Gate-1 sorted support sequence-ID digest changed")
    if support_contexts != contract.expected_support_contexts:
        raise ValueError("Gate-1 support-context census changed")
    if support_observations != contract.expected_support_source_observations:
        raise ValueError("Gate-1 support source-observation census changed")
    target_distribution = tuple(support_target_counts[count] for count in range(2, 8))
    if target_distribution != contract.expected_support_unique_target_counts_2_to_7:
        raise ValueError("Gate-1 support unique-target distribution changed")
    if sum(support_target_counts.values()) != sum(target_distribution):
        raise ValueError("Gate-1 support sequence has a unique-target count outside 2..7")
    return Gate1Panel(
        contexts=contexts,
        contexts_by_fold=tuple(tuple(rows) for rows in by_fold_lists),
        support_sequence_ids=tuple(sorted(support_ids)),
        support_sequence_ids_by_fold=support_by_fold,
        support_sequence_ids_sha256=support_digest,
    )


def label_free_contexts(contexts: Iterable[ContextRow]) -> tuple[LabelFreeContext, ...]:
    """Remove labels and source-observation counts from contexts in canonical order."""

    return tuple(
        LabelFreeContext(
            example_id=row.example_id,
            assay_context_id=row.assay_context_id,
            sequence_id=row.sequence_id,
            sequence=row.sequence,
            target=row.target,
            gram=row.gram,
            fold=row.fold,
        )
        for row in sorted(contexts, key=lambda item: item.example_id)
    )


def rotation_capability_index(panel: Gate1Panel, *, spec: RotationSpec) -> RotationCapabilityIndex:
    """Return disjoint, complete, label-free example-ID capabilities."""

    base_rows = tuple(
        sorted(
            (row for fold in spec.base_folds for row in panel.contexts_by_fold[fold]),
            key=lambda row: row.example_id,
        )
    )
    base_ids = tuple(row.example_id for row in base_rows)
    acquisition_ids = tuple(row.example_id for row in panel.contexts_by_fold[spec.pool_fold])
    outer_ids = tuple(row.example_id for row in panel.contexts_by_fold[spec.outer_fold])
    id_sets = tuple(map(set, (base_ids, acquisition_ids, outer_ids)))
    if any(id_sets[left] & id_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("rotation role example-ID capabilities overlap")
    if set().union(*id_sets) != {row.example_id for row in panel.contexts}:
        raise ValueError("rotation role example-ID capabilities do not cover the panel")
    return RotationCapabilityIndex(
        spec=spec,
        base_example_ids=base_ids,
        acquisition_example_ids=acquisition_ids,
        outer_example_ids=outer_ids,
        acquisition_support_sequence_ids=panel.support_sequence_ids_by_fold[spec.pool_fold],
        outer_support_sequence_ids=panel.support_sequence_ids_by_fold[spec.outer_fold],
    )


def prepare_capability(panel: Gate1Panel, *, spec: RotationSpec) -> PrepareCapability:
    """Return three-base-fold labels and acquisition-fold metadata, never other labels."""

    index = rotation_capability_index(panel, spec=spec)
    base_contexts = tuple(
        sorted(
            (row for fold in spec.base_folds for row in panel.contexts_by_fold[fold]),
            key=lambda row: row.example_id,
        )
    )
    acquisition_metadata = label_free_contexts(panel.contexts_by_fold[spec.pool_fold])
    if tuple(row.example_id for row in base_contexts) != index.base_example_ids:
        raise ValueError("prepare base capability differs from its authenticated ID index")
    if tuple(row.example_id for row in acquisition_metadata) != index.acquisition_example_ids:
        raise ValueError("prepare pool metadata differs from its authenticated ID index")
    return PrepareCapability(
        spec=spec,
        base_contexts=base_contexts,
        acquisition_metadata=acquisition_metadata,
        acquisition_support_sequence_ids=index.acquisition_support_sequence_ids,
        allowed_base_example_ids=index.base_example_ids,
        allowed_acquisition_metadata_example_ids=index.acquisition_example_ids,
    )


def pool_outcome_vault(panel: Gate1Panel, *, spec: RotationSpec) -> PoolOutcomeVault:
    """Return the pool-label capability that only a reveal custodian may receive."""

    index = rotation_capability_index(panel, spec=spec)
    contexts = panel.contexts_by_fold[spec.pool_fold]
    if tuple(row.example_id for row in contexts) != index.acquisition_example_ids:
        raise ValueError("pool outcome vault differs from its authenticated ID index")
    return PoolOutcomeVault(
        spec=spec, contexts=contexts, allowed_example_ids=index.acquisition_example_ids
    )


def outer_metadata_capability(panel: Gate1Panel, *, spec: RotationSpec) -> OuterMetadataCapability:
    """Return outer label-free metadata for post-update prediction and selection."""

    index = rotation_capability_index(panel, spec=spec)
    contexts = label_free_contexts(panel.contexts_by_fold[spec.outer_fold])
    if tuple(row.example_id for row in contexts) != index.outer_example_ids:
        raise ValueError("outer metadata capability differs from its authenticated ID index")
    return OuterMetadataCapability(
        spec=spec,
        contexts=contexts,
        allowed_example_ids=index.outer_example_ids,
        support_sequence_ids=index.outer_support_sequence_ids,
    )


def outer_outcome_vault(panel: Gate1Panel, *, spec: RotationSpec) -> OuterOutcomeVault:
    """Return the outer-label capability that only finalization may receive."""

    index = rotation_capability_index(panel, spec=spec)
    contexts = panel.contexts_by_fold[spec.outer_fold]
    if tuple(row.example_id for row in contexts) != index.outer_example_ids:
        raise ValueError("outer outcome vault differs from its authenticated ID index")
    return OuterOutcomeVault(
        spec=spec, contexts=contexts, allowed_example_ids=index.outer_example_ids
    )


def _validate_manifest(
    payload: bytes, *, semantic_hashes: Mapping[str, str], contract: Gate1SourceContract
) -> None:
    document = _json_object(payload, label="Gate-1 manifest")
    if (
        document.get("schema_version") != 1
        or document.get("artifact") != _GATE1_ARTIFACT
        or document.get("status") != _GATE1_STATUS
        or document.get("git_commit") != contract.git_commit
        or document.get("config_sha256") != contract.config_sha256
        or document.get("models")
        != ["descriptor_logistic", "homology_knn", "equal_weight_ensemble"]
    ):
        raise ValueError("Gate-1 manifest identity/status is invalid")
    artifacts = document.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("Gate-1 manifest artifacts must be an object")
    expected_keys = {
        "context_audit": "context_audit.jsonl",
        "examples": "examples.jsonl",
        "folds": "folds.json",
        "metrics": "metrics.json",
        "oof": "oof_predictions.csv",
        "split_receipt": "split_receipt.json",
    }
    if set(artifacts) != set(expected_keys):
        raise ValueError("Gate-1 manifest artifact inventory is not exact")
    for key, filename in expected_keys.items():
        record = artifacts[key]
        if (
            type(record) is not dict
            or record.get("filename") != filename
            or record.get("sha256") != semantic_hashes[filename]
        ):
            raise ValueError(f"Gate-1 manifest does not bind {filename}")


def _validate_split_receipt(payload: bytes, *, contract: Gate1SourceContract) -> None:
    document = _json_object(payload, label="Gate-1 split receipt")
    census = document.get("census")
    identity = document.get("identity")
    invariants = document.get("invariants")
    code = document.get("code_attestation")
    if (
        document.get("schema_version") != 1
        or document.get("artifact") != _SPLIT_RECEIPT_ARTIFACT
        or document.get("status") != "passed"
        or document.get("assignment_policy") != _ASSIGNMENT_POLICY
        or type(census) is not dict
        or census.get("context_examples") != contract.expected_contexts
        or census.get("modeled_sequences") != contract.expected_sequences
        or census.get("examples_by_fold") != list(contract.expected_examples_by_fold)
        or census.get("negatives") != contract.expected_negatives
        or census.get("positives") != contract.expected_positives
        or census.get("source_observations") != contract.expected_source_observations
        or census.get("homology_components") != contract.expected_split_homology_components
        or census.get("union_components") != contract.expected_split_union_components
        or type(identity) is not dict
        or identity.get("threshold") != 0.8
        or identity.get("maximum_cross_fold_identity") != contract.maximum_cross_fold_identity
        or type(invariants) is not dict
        or not invariants
        or any(value is not True for value in invariants.values())
        or type(code) is not dict
        or code.get("code_manifest_sha256") != contract.code_manifest_sha256
    ):
        raise ValueError("Gate-1 split receipt differs from the accepted contract")


def _validate_independent_receipt(payload: bytes, *, contract: Gate1SourceContract) -> None:
    document = _json_object(payload, label="Gate-1 independent receipt")
    artifacts = document.get("artifact_sha256")
    census = document.get("census")
    identity = document.get("identity")
    checks = document.get("checks")
    handshake = document.get("production_handshake")
    if (
        document.get("schema_version") != 1
        or document.get("artifact") != _RECEIPT_ARTIFACT
        or document.get("status") != "passed"
        or document.get("git_commit") != contract.git_commit
        or document.get("config_sha256") != contract.config_sha256
        or document.get("code_manifest_sha256") != contract.code_manifest_sha256
        or document.get("frozen_input_manifest_sha256") != contract.frozen_input_manifest_sha256
        or document.get("publication_top_manifest_sha256") != contract.publication_top_sha256
        or document.get("gate1_top_manifest_sha256") != contract.semantic_top_sha256
        or artifacts != dict(contract.artifact_sha256)
        or type(census) is not dict
        or census.get("context_examples") != contract.expected_contexts
        or census.get("modeled_sequences") != contract.expected_sequences
        or census.get("examples_by_fold") != list(contract.expected_examples_by_fold)
        or census.get("negatives") != contract.expected_negatives
        or census.get("positives") != contract.expected_positives
        or census.get("source_observations") != contract.expected_source_observations
        or type(identity) is not dict
        or identity.get("threshold") != 0.8
        or identity.get("maximum_cross_fold_identity") != contract.maximum_cross_fold_identity
        or type(checks) is not dict
        or not checks
        or any(value is not True for value in checks.values())
        or type(handshake) is not dict
        or handshake.get("bidirectional_acknowledgement") is not True
        or handshake.get("distinct_nodes") is not True
    ):
        raise ValueError("Gate-1 independent receipt differs from the accepted contract")
    for field in ("acknowledgement_sha256", "receipt_sha256"):
        values = handshake.get(field)
        if type(values) is not dict or set(values) != {"0", "1"}:
            raise ValueError(f"Gate-1 independent receipt has an invalid {field} map")
        for twin, digest in values.items():
            _require_sha256(digest, label=f"Gate-1 independent receipt {field}.{twin}")


def _authenticate_twin(
    root: Path, *, expected_name: str, contract: Gate1SourceContract
) -> tuple[Mapping[str, _Snapshot], _Snapshot, tuple[_DirectorySnapshot, ...]]:
    root_directory = _snapshot_directory(
        root,
        label=f"Gate-1 twin {expected_name}",
        mode=0o500,
    )
    root = root_directory.path
    if root.name != expected_name or root.parent.name != str(contract.producer_job_id):
        raise ValueError(
            f"Gate-1 twin {expected_name} must be directly below job {contract.producer_job_id}"
        )
    observed_root = set(root_directory.entries)
    if observed_root != {*_ROOT_FILES, "gate1"}:
        raise ValueError(f"Gate-1 twin {expected_name} root inventory is not exact")
    gate_directory = _snapshot_directory(
        root / "gate1",
        label=f"Gate-1 twin {expected_name} semantic",
        mode=0o500,
    )
    gate = gate_directory.path
    observed_gate = set(gate_directory.entries)
    if observed_gate != {"SHA256SUMS", *_SEMANTIC_FILES}:
        raise ValueError(f"Gate-1 twin {expected_name} semantic inventory is not exact")

    snapshots: dict[str, _Snapshot] = {}
    snapshots["SHA256SUMS"] = _snapshot_regular(
        root / "SHA256SUMS", label=f"Gate-1 twin {expected_name} top manifest", mode=0o400
    )
    for name in ("CODE_SHA256SUMS", "FROZEN_INPUT_SHA256SUMS"):
        snapshots[name] = _snapshot_regular(
            root / name, label=f"Gate-1 twin {expected_name} {name}", mode=0o444
        )
    snapshots["gate1/SHA256SUMS"] = _snapshot_regular(
        gate / "SHA256SUMS",
        label=f"Gate-1 twin {expected_name} semantic manifest",
        mode=0o444,
    )
    for name in _SEMANTIC_FILES:
        snapshots[f"gate1/{name}"] = _snapshot_regular(
            gate / name, label=f"Gate-1 twin {expected_name} gate1/{name}", mode=0o444
        )

    if snapshots["SHA256SUMS"].sha256 != contract.publication_top_sha256:
        raise ValueError(f"Gate-1 twin {expected_name} publication manifest hash changed")
    if snapshots["CODE_SHA256SUMS"].sha256 != contract.code_manifest_sha256:
        raise ValueError(f"Gate-1 twin {expected_name} code manifest hash changed")
    if snapshots["FROZEN_INPUT_SHA256SUMS"].sha256 != contract.frozen_input_manifest_sha256:
        raise ValueError(f"Gate-1 twin {expected_name} frozen-input manifest hash changed")
    semantic = snapshots["gate1/SHA256SUMS"]
    if semantic.sha256 != contract.semantic_top_sha256:
        raise ValueError(f"Gate-1 twin {expected_name} semantic manifest hash changed")

    top_entries = _parse_sha256sums(
        snapshots["SHA256SUMS"].payload, label=f"Gate-1 twin {expected_name} top manifest"
    )
    expected_top_names = {
        "CODE_SHA256SUMS",
        "FROZEN_INPUT_SHA256SUMS",
        "gate1/SHA256SUMS",
        *(f"gate1/{name}" for name in _SEMANTIC_FILES),
    }
    if set(top_entries) != expected_top_names:
        raise ValueError(f"Gate-1 twin {expected_name} top checksum inventory is not exact")
    semantic_entries = _parse_sha256sums(
        semantic.payload, label=f"Gate-1 twin {expected_name} semantic manifest"
    )
    if set(semantic_entries) != set(_SEMANTIC_FILES):
        raise ValueError(f"Gate-1 twin {expected_name} semantic checksum inventory is not exact")
    for logical, snapshot in snapshots.items():
        if logical == "SHA256SUMS":
            continue
        if top_entries.get(logical) != snapshot.sha256:
            raise ValueError(f"Gate-1 twin {expected_name} top manifest does not bind {logical}")
    for name, expected_hash in contract.artifact_sha256:
        snapshot = snapshots[f"gate1/{name}"]
        if semantic_entries.get(name) != snapshot.sha256 or snapshot.sha256 != expected_hash:
            raise ValueError(f"Gate-1 twin {expected_name} semantic artifact hash changed: {name}")
    _validate_manifest(
        snapshots["gate1/manifest.json"].payload,
        semantic_hashes=semantic_entries,
        contract=contract,
    )
    _validate_split_receipt(snapshots["gate1/split_receipt.json"].payload, contract=contract)
    _assert_directory_unchanged(root_directory, label=f"Gate-1 twin {expected_name}")
    _assert_directory_unchanged(
        gate_directory,
        label=f"Gate-1 twin {expected_name} semantic",
    )
    return snapshots, snapshots["gate1/examples.jsonl"], (root_directory, gate_directory)


def _authenticate_gate1_source(
    twin_roots: Sequence[str | Path],
    independent_receipt_path: str | Path,
    *,
    primary_twin_slot: int,
    contract: Gate1SourceContract,
) -> AuthenticatedGate1Source:
    if type(primary_twin_slot) is not int or primary_twin_slot not in (0, 1):
        raise ValueError("Gate-1 primary twin slot must be integer zero or one")
    if len(twin_roots) != 2:
        raise ValueError("Gate-1 authentication requires exactly two ordered twins")
    zero_requested = Path(os.path.abspath(os.fspath(twin_roots[0])))
    one_requested = Path(os.path.abspath(os.fspath(twin_roots[1])))
    if zero_requested.parent != one_requested.parent:
        raise ValueError("Gate-1 twins must share the same accepted producer-job parent")
    zero, zero_examples, zero_directories = _authenticate_twin(
        zero_requested,
        expected_name="0",
        contract=contract,
    )
    one, one_examples, one_directories = _authenticate_twin(
        one_requested,
        expected_name="1",
        contract=contract,
    )
    if set(zero) != set(one):
        raise ValueError("Gate-1 twins have different authenticated inventories")
    for logical in sorted(zero):
        if zero[logical].payload != one[logical].payload:
            raise ValueError(f"Gate-1 twins differ at {logical}")

    receipt_path = Path(os.path.abspath(os.fspath(independent_receipt_path)))
    if (
        receipt_path.name != f"independent-verification-{contract.audit_job_id}.json"
        or receipt_path.parent.name != str(contract.producer_job_id)
    ):
        raise ValueError("Gate-1 independent receipt path does not bind producer/audit job IDs")
    receipt = _snapshot_regular(receipt_path, label="Gate-1 independent receipt", mode=0o444)
    if receipt.sha256 != contract.independent_receipt_sha256:
        raise ValueError("Gate-1 independent receipt hash changed")
    _validate_independent_receipt(receipt.payload, contract=contract)

    primary = (zero, one)[primary_twin_slot]
    primary_examples = (zero_examples, one_examples)[primary_twin_slot]
    panel = parse_gate1_panel(
        primary_examples.payload,
        primary["gate1/folds.json"].payload,
        contract=contract,
    )
    for logical, snapshot in (*zero.items(), *one.items()):
        _assert_unchanged(snapshot, label=f"Gate-1 authenticated source {logical}")
    for index, directories in enumerate((zero_directories, one_directories)):
        for directory in directories:
            _assert_directory_unchanged(
                directory,
                label=f"Gate-1 authenticated twin {index} directory",
            )
    _assert_unchanged(receipt, label="Gate-1 independent receipt")
    files = tuple(
        SourceFileEvidence(
            logical_path=logical,
            sha256=primary[logical].sha256,
            size_bytes=primary[logical].size_bytes,
            mode=f"{primary[logical].mode:04o}",
        )
        for logical in sorted(zero)
    )
    return AuthenticatedGate1Source(
        panel=panel,
        evidence=Gate1AuthenticationEvidence(
            producer_job_id=contract.producer_job_id,
            audit_job_id=contract.audit_job_id,
            files=files,
            independent_receipt_sha256=receipt.sha256,
        ),
    )


def authenticate_accepted_gate1_source(
    twin_roots: Sequence[str | Path],
    independent_receipt_path: str | Path,
    *,
    primary_twin_slot: int,
) -> AuthenticatedGate1Source:
    """Authenticate both accepted twins and parse one explicitly selected source."""

    return _authenticate_gate1_source(
        twin_roots,
        independent_receipt_path,
        primary_twin_slot=primary_twin_slot,
        contract=ACCEPTED_GATE1_SOURCE_CONTRACT,
    )


__all__ = [
    "ACCEPTED_GATE1_SOURCE_CONTRACT",
    "AuthenticatedGate1Source",
    "Gate1AuthenticationEvidence",
    "Gate1Panel",
    "Gate1SourceContract",
    "LabelFreeContext",
    "OuterMetadataCapability",
    "OuterOutcomeVault",
    "PoolOutcomeVault",
    "PrepareCapability",
    "RotationCapabilityIndex",
    "RotationSpec",
    "SourceFileEvidence",
    "authenticate_accepted_gate1_source",
    "label_free_contexts",
    "outer_metadata_capability",
    "outer_outcome_vault",
    "parse_gate1_panel",
    "pool_outcome_vault",
    "prepare_capability",
    "rotation_capability_index",
]

"""Publish and consume the role-isolated trusted stage for sequential replay v2.

The stage custodian is the only process allowed to hold the complete labeled
Gate-1 panel.  It authenticates that panel, emits four physically separate
sealed capabilities for each frozen rotation, seals an outcome-free global
index over all 80 leaves, and exits.  Later workers authenticate the global
index and open only the one role leaf that they are authorized to consume.

The final directory is assembled in a fresh private sibling.  Publication uses
an atomic no-replace directory rename when the filesystem supports it.  The
Lustre-safe fallback first claims the destination exclusively while it is not a
valid stage, moves the already verified trees without retaining aliases, and
makes ``global/SHA256SUMS`` readable only as the final commit transition.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from amp_challenge.evaluation.sequential_v2_primitives import TARGET_GRAM, ContextRow
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    protocol_census,
    rotation_by_id,
    validate_policy_run_documents,
    validate_rotation_documents,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    MANIFEST_NAME,
    RECEIPT_NAME,
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    checksum_manifest_bytes,
    publish_phase,
    sha256_bytes,
    verify_phase,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_staging import (
    ACCEPTED_GATE1_SOURCE_CONTRACT,
    AuthenticatedGate1Source,
    Gate1AuthenticationEvidence,
    Gate1Panel,
    LabelFreeContext,
    OuterMetadataCapability,
    OuterOutcomeVault,
    PoolOutcomeVault,
    PrepareCapability,
    SourceFileEvidence,
    authenticate_accepted_gate1_source,
    outer_metadata_capability,
    outer_outcome_vault,
    pool_outcome_vault,
    prepare_capability,
)
from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

SCHEMA_VERSION = 1
STAGE_ARTIFACT = "sequential_v2_trusted_stage"
SOURCE_ANCHORS_ARTIFACT = "sequential_v2_stage_source_anchors"
STAGE_SUMMARY_ARTIFACT = "sequential_v2_trusted_stage_summary"
LEAF_CAPABILITY_ARTIFACT = "sequential_v2_stage_leaf_capability"

GLOBAL_DIRECTORY = "global"
ROTATIONS_DIRECTORY = "rotations"
GLOBAL_PAYLOAD_PATHS = (
    "capability-index.jsonl",
    "policy-runs.jsonl",
    "protocol-census.json",
    "rotations.jsonl",
    "source-anchors.json",
    "stage-summary.json",
)

PREPARE_ROLE = "prepare-capability"
POOL_OUTCOME_ROLE = "pool-outcome-vault"
OUTER_METADATA_ROLE = "outer-metadata"
OUTER_OUTCOME_ROLE = "outer-outcome-vault"
ROLE_ORDER = (
    PREPARE_ROLE,
    POOL_OUTCOME_ROLE,
    OUTER_METADATA_ROLE,
    OUTER_OUTCOME_ROLE,
)
LEAF_ARTIFACTS: Mapping[str, str] = {
    PREPARE_ROLE: "sequential_v2_stage_prepare_capability",
    POOL_OUTCOME_ROLE: "sequential_v2_stage_pool_outcome_vault",
    OUTER_METADATA_ROLE: "sequential_v2_stage_outer_metadata",
    OUTER_OUTCOME_ROLE: "sequential_v2_stage_outer_outcome_vault",
}
LEAF_DATA_PAYLOAD_PATHS: Mapping[str, tuple[str, ...]] = {
    PREPARE_ROLE: (
        "base-contexts.jsonl",
        "pool-metadata.jsonl",
        "pool-support-sequence-ids.jsonl",
    ),
    POOL_OUTCOME_ROLE: ("contexts.jsonl",),
    OUTER_METADATA_ROLE: ("contexts.jsonl", "support-sequence-ids.jsonl"),
    OUTER_OUTCOME_ROLE: ("contexts.jsonl",),
}
LEAF_PAYLOAD_PATHS: Mapping[str, tuple[str, ...]] = {
    role: ("capability.json", *paths) for role, paths in LEAF_DATA_PAYLOAD_PATHS.items()
}
LEAF_ID_STREAM_NAMES: Mapping[str, tuple[str, ...]] = {
    PREPARE_ROLE: (
        "base_example_ids",
        "pool_metadata_example_ids",
        "pool_support_sequence_ids",
    ),
    POOL_OUTCOME_ROLE: ("pool_outcome_example_ids",),
    OUTER_METADATA_ROLE: (
        "outer_metadata_example_ids",
        "outer_support_sequence_ids",
    ),
    OUTER_OUTCOME_ROLE: ("outer_outcome_example_ids",),
}

SOURCE_PUBLICATION_PREDECESSOR = "source/gate1-publication-top"
SOURCE_SEMANTIC_PREDECESSOR = "source/gate1-semantic-top"
SOURCE_RECEIPT_PREDECESSOR = "source/gate1-independent-receipt"

_CONTEXT_FIELDS = frozenset(
    {
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
)
_METADATA_FIELDS = frozenset(_CONTEXT_FIELDS - {"label", "source_observations"})
_SUPPORT_FIELDS = frozenset({"sequence_id"})
_INDEX_FIELDS = frozenset(
    {
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
)
_CAPABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "rotation_id",
        "role",
        "outer_fold",
        "acquisition_pool_fold",
        "base_folds",
        "data_payload_paths",
        "row_counts",
        "id_streams",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "source_anchors_sha256",
        "capability_index_sha256",
        "rotation_count",
        "leaf_count",
        "panel_context_count",
        "panel_example_ids_sha256",
        "panel_sequence_count",
        "panel_sequence_ids_sha256",
        "contexts_by_fold",
        "sequences_by_fold",
        "support_sequence_count",
        "support_sequence_ids_sha256",
        "support_sequences_by_fold",
        "role_leaf_counts",
        "role_association_censuses",
    }
)
_SOURCE_ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "publication_top_sha256",
        "semantic_top_sha256",
        "independent_receipt_sha256",
        "gate1_authentication_evidence",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_RENAME_NOREPLACE = 1


@dataclass(frozen=True, slots=True)
class IdStreamBinding:
    """Count and canonical LF-stream digest for one sorted unique ID set."""

    count: int
    sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("ID-stream count must be a positive integer")
        _require_sha256(self.sha256, label="ID-stream digest")

    def document(self) -> dict[str, object]:
        return {"count": self.count, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class StageLeafIndex:
    """Outcome-free authenticated location and census for one role leaf."""

    spec: RotationSpec
    role: str
    relative_path: str
    leaf_artifact: str
    leaf_seal_sha256: str
    payload_paths: tuple[str, ...]
    row_counts: tuple[tuple[str, int], ...]
    id_streams: tuple[tuple[str, IdStreamBinding], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RotationSpec) or self.role not in ROLE_ORDER:
            raise ValueError("stage leaf has an invalid rotation or role")
        if self.relative_path != leaf_relative_path(self.spec, self.role):
            raise ValueError("stage leaf relative path differs from its role identity")
        if self.leaf_artifact != LEAF_ARTIFACTS[self.role]:
            raise ValueError("stage leaf artifact differs from its role")
        _require_sha256(self.leaf_seal_sha256, label="stage leaf seal")
        if self.payload_paths != LEAF_PAYLOAD_PATHS[self.role]:
            raise ValueError("stage leaf payload inventory differs from its role")
        expected_rows = LEAF_DATA_PAYLOAD_PATHS[self.role]
        if tuple(path for path, _ in self.row_counts) != expected_rows:
            raise ValueError("stage leaf row-count inventory differs from its role")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for _, count in self.row_counts
        ):
            raise ValueError("stage leaf row counts must be positive integers")
        if tuple(name for name, _ in self.id_streams) != LEAF_ID_STREAM_NAMES[self.role]:
            raise ValueError("stage leaf ID-stream inventory differs from its role")
        expected_counts = _stream_row_count_bindings(self.role, dict(self.row_counts))
        if any(binding.count != expected_counts[name] for name, binding in self.id_streams):
            raise ValueError("stage leaf ID-stream count differs from its row census")

    @property
    def row_count_map(self) -> Mapping[str, int]:
        return dict(self.row_counts)

    @property
    def id_stream_map(self) -> Mapping[str, IdStreamBinding]:
        return dict(self.id_streams)

    def capability_document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": LEAF_CAPABILITY_ARTIFACT,
            "rotation_id": self.spec.rotation_id,
            "role": self.role,
            "outer_fold": self.spec.outer_fold,
            "acquisition_pool_fold": self.spec.pool_fold,
            "base_folds": list(self.spec.base_folds),
            "data_payload_paths": list(LEAF_DATA_PAYLOAD_PATHS[self.role]),
            "row_counts": dict(self.row_counts),
            "id_streams": {name: binding.document() for name, binding in self.id_streams},
        }

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rotation_id": self.spec.rotation_id,
            "role": self.role,
            "relative_path": self.relative_path,
            "leaf_artifact": self.leaf_artifact,
            "leaf_seal_sha256": self.leaf_seal_sha256,
            "payload_paths": list(self.payload_paths),
            "row_counts": dict(self.row_counts),
            "id_streams": {name: binding.document() for name, binding in self.id_streams},
        }


@dataclass(frozen=True, slots=True)
class StageManifestCapability:
    """Rootless label-free stage-global bytes, not authority by itself.

    A downstream worker must pass the capsule through
    :func:`verify_stage_manifest_capability` with the controller-authoritative
    expected stage-global digest before using its index.  Keeping only the
    captured ``PhaseSeal`` avoids a second, caller-controlled copy of any
    decoded index or source binding.
    """

    seal: PhaseSeal

    def __post_init__(self) -> None:
        if type(self.seal) is not PhaseSeal:
            raise TypeError("stage manifest capability must contain an exact PhaseSeal")
        _decode_stage_manifest_seal(
            self.seal,
            expected_global_seal_sha256=self.seal.seal_sha256,
        )

    @property
    def global_seal_sha256(self) -> str:
        return self.seal.seal_sha256

    @property
    def source_anchors_sha256(self) -> str:
        return self._decoded().source_anchors_sha256

    @property
    def source_predecessors(self) -> tuple[tuple[str, str], ...]:
        return self._decoded().source_predecessors

    @property
    def leaves(self) -> tuple[StageLeafIndex, ...]:
        return self._decoded().leaves

    @property
    def stage_summary_json(self) -> bytes:
        return self._decoded().stage_summary_json

    def leaf(self, *, spec: RotationSpec, role: str) -> StageLeafIndex:
        """Return one exact leaf identity from the sealed global index."""

        frozen = _require_frozen_rotation_boundary(spec, label="stage capability lookup")
        if type(role) is not str or role not in ROLE_ORDER:
            raise ValueError("stage capability lookup role is invalid")
        matches = tuple(
            entry for entry in self.leaves if entry.spec == frozen and entry.role == role
        )
        if len(matches) != 1:
            raise ValueError("stage capability lacks one exact requested role leaf")
        return matches[0]

    def _decoded(self) -> _DecodedStageManifest:
        # This self-digest check proves only structural consistency.  Public
        # consumers re-run the decoder under an externally supplied digest.
        return _decode_stage_manifest_seal(
            self.seal,
            expected_global_seal_sha256=self.seal.seal_sha256,
        )


@dataclass(frozen=True, slots=True)
class TrustedStage:
    """Outcome-free handle returned after authenticating the stage graph."""

    root: Path
    global_seal: PhaseSeal
    global_seal_sha256: str
    source_anchors_sha256: str
    source_predecessors: tuple[tuple[str, str], ...]
    leaves: tuple[StageLeafIndex, ...]
    stage_summary_json: bytes

    def __post_init__(self) -> None:
        if type(self.global_seal) is not PhaseSeal:
            raise TypeError("trusted stage must retain an exact rootless global PhaseSeal")
        _require_sha256(self.global_seal_sha256, label="global stage seal")
        verify_phase_capability(
            self.global_seal,
            expected_artifact=STAGE_ARTIFACT,
            expected_payload_paths=GLOBAL_PAYLOAD_PATHS,
            expected_seal_sha256=self.global_seal_sha256,
        )
        _require_sha256(self.source_anchors_sha256, label="source-anchor document")
        if len(self.leaves) != EXPECTED_ROTATIONS * len(ROLE_ORDER):
            raise ValueError("trusted stage must bind exactly 80 role leaves")

    def leaf(self, *, spec: RotationSpec, role: str) -> StageLeafIndex:
        matches = tuple(entry for entry in self.leaves if entry.spec == spec and entry.role == role)
        if len(matches) != 1:
            raise ValueError("trusted stage lacks one exact requested role leaf")
        return matches[0]

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": STAGE_ARTIFACT,
            "root": str(self.root),
            "global_seal_sha256": self.global_seal_sha256,
            "source_anchors_sha256": self.source_anchors_sha256,
            "leaf_count": len(self.leaves),
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedLeafCapsule:
    """Rootless authenticated bytes for exactly one role leaf."""

    entry: StageLeafIndex
    seal: PhaseSeal
    source_anchors_sha256: str
    source_predecessors: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entry, StageLeafIndex) or not isinstance(self.seal, PhaseSeal):
            raise TypeError("authenticated leaf capsule has invalid components")
        anchors_sha256 = _require_sha256(
            self.source_anchors_sha256,
            label="capsule source-anchor document",
        )
        expected_predecessors = tuple(sorted(self.source_predecessors))
        if (
            expected_predecessors != self.source_predecessors
            or len(dict(expected_predecessors)) != len(expected_predecessors)
            or any(
                not isinstance(path, str)
                or _require_sha256(digest, label=f"capsule predecessor {path}") != digest
                for path, digest in expected_predecessors
            )
        ):
            raise ValueError("authenticated leaf capsule predecessors are not canonical")
        expected_paths = tuple(sorted(self.entry.payload_paths))
        payload_hashes = self.seal.payload_sha256
        payload_bytes = self.seal.payload_bytes
        if (
            self.seal.artifact != self.entry.leaf_artifact
            or self.seal.seal_sha256 != self.entry.leaf_seal_sha256
            or tuple(path for path, _digest in payload_hashes) != expected_paths
            or tuple(path for path, _payload in payload_bytes) != expected_paths
            or len(dict(payload_hashes)) != len(payload_hashes)
            or len(dict(payload_bytes)) != len(payload_bytes)
            or self.seal.predecessor_seals != expected_predecessors
            or self.seal.files != tuple(sorted((*expected_paths, RECEIPT_NAME, MANIFEST_NAME)))
            or self.seal.metadata_json
            != canonical_json_bytes(_leaf_metadata(self.entry, anchors_sha256))
        ):
            raise ValueError("authenticated leaf capsule differs from its sealed index entry")
        hash_map = dict(payload_hashes)
        if any(sha256_bytes(payload) != hash_map[path] for path, payload in payload_bytes):
            raise ValueError("authenticated leaf capsule payload bytes differ from their hashes")
        metadata = _strict_json_object(
            self.seal.metadata_json,
            label="authenticated leaf capsule metadata",
        )
        receipt_bytes = canonical_json_bytes(
            {
                "artifact": self.seal.artifact,
                "metadata": metadata,
                "payloads": hash_map,
                "predecessor_seals": dict(expected_predecessors),
                "schema_version": 1,
                "status": "sealed",
            }
        )
        if sha256_bytes(receipt_bytes) != self.seal.receipt_sha256:
            raise ValueError("authenticated leaf capsule receipt identity is invalid")
        manifest_bytes = checksum_manifest_bytes(
            {**hash_map, RECEIPT_NAME: self.seal.receipt_sha256}
        )
        if sha256_bytes(manifest_bytes) != self.seal.seal_sha256:
            raise ValueError("authenticated leaf capsule seal identity is invalid")


@dataclass(frozen=True, slots=True)
class _LeafMaterial:
    spec: RotationSpec
    role: str
    payloads: Mapping[str, bytes]
    row_counts: tuple[tuple[str, int], ...]
    id_streams: tuple[tuple[str, IdStreamBinding], ...]


@dataclass(frozen=True, slots=True)
class _LoadedLeaf:
    entry: StageLeafIndex
    base_contexts: tuple[ContextRow, ...] = ()
    metadata_contexts: tuple[LabelFreeContext, ...] = ()
    outcome_contexts: tuple[ContextRow, ...] = ()
    support_sequence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DecodedStageManifest:
    seal: PhaseSeal
    source_anchors_json: bytes
    source_anchors_sha256: str
    source_predecessors: tuple[tuple[str, str], ...]
    leaves: tuple[StageLeafIndex, ...]
    stage_summary_json: bytes


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_frozen_rotation_boundary(value: object, *, label: str) -> RotationSpec:
    """Return the canonical rotation after rejecting subclass-based aliases."""

    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} fold identities must be exact integers")
    canonical = rotation_by_id(value.rotation_id)
    if (
        value.outer_fold != canonical.outer_fold
        or value.pool_fold != canonical.pool_fold
        or value.base_folds != canonical.base_folds
    ):
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return canonical


def _require_exact_stage_leaf_index(value: object, *, label: str) -> StageLeafIndex:
    """Reject custom objects anywhere in a caller-supplied stage index row."""

    if type(value) is not StageLeafIndex:
        raise TypeError(f"{label} must be an exact StageLeafIndex")
    _require_frozen_rotation_boundary(value.spec, label=f"{label} rotation")
    for field in ("role", "relative_path", "leaf_artifact", "leaf_seal_sha256"):
        if type(getattr(value, field)) is not str:
            raise TypeError(f"{label} identity fields must use exact strings")
    _require_sha256(value.leaf_seal_sha256, label=f"{label} leaf seal")
    if type(value.payload_paths) is not tuple or any(
        type(path) is not str for path in value.payload_paths
    ):
        raise TypeError(f"{label} payload paths must be an exact string tuple")
    if type(value.row_counts) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not int
        for item in value.row_counts
    ):
        raise TypeError(f"{label} row counts must use exact immutable scalar types")
    if type(value.id_streams) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not IdStreamBinding
        or type(item[1].count) is not int
        or type(item[1].sha256) is not str
        for item in value.id_streams
    ):
        raise TypeError(f"{label} ID streams must use exact immutable scalar types")
    return value


def _require_exact_fields(
    document: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    if set(document) != expected:
        raise ValueError(f"{label} has an unexpected schema")


def _id_stream_sha256(values: Iterable[str], *, label: str) -> str:
    ids = tuple(values)
    if (
        not ids
        or ids != tuple(sorted(set(ids)))
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in ids)
    ):
        raise ValueError(f"{label} must be nonempty sorted unique lowercase SHA-256 IDs")
    return sha256_bytes("".join(f"{value}\n" for value in ids).encode("ascii"))


def leaf_relative_path(spec: RotationSpec, role: str) -> str:
    """Return the only allowed aggregate-relative path for one role leaf."""

    if not isinstance(spec, RotationSpec):
        raise TypeError("stage leaf spec must be a RotationSpec")
    if role not in ROLE_ORDER:
        raise ValueError(f"unknown stage role: {role!r}")
    return f"{ROTATIONS_DIRECTORY}/{spec.rotation_id}/{role}"


def _stream_row_count_bindings(role: str, rows: Mapping[str, int]) -> Mapping[str, int]:
    if role == PREPARE_ROLE:
        return {
            "base_example_ids": rows["base-contexts.jsonl"],
            "pool_metadata_example_ids": rows["pool-metadata.jsonl"],
            "pool_support_sequence_ids": rows["pool-support-sequence-ids.jsonl"],
        }
    if role == POOL_OUTCOME_ROLE:
        return {"pool_outcome_example_ids": rows["contexts.jsonl"]}
    if role == OUTER_METADATA_ROLE:
        return {
            "outer_metadata_example_ids": rows["contexts.jsonl"],
            "outer_support_sequence_ids": rows["support-sequence-ids.jsonl"],
        }
    if role == OUTER_OUTCOME_ROLE:
        return {"outer_outcome_example_ids": rows["contexts.jsonl"]}
    raise ValueError(f"unknown stage role: {role!r}")


def _context_document(row: ContextRow) -> dict[str, object]:
    if not isinstance(row, ContextRow):
        raise TypeError("stage outcome rows must be ContextRow instances")
    return {
        "example_id": row.example_id,
        "assay_context_id": row.assay_context_id,
        "sequence_id": row.sequence_id,
        "sequence": row.sequence,
        "target": row.target,
        "gram": row.gram,
        "fold": row.fold,
        "label": row.label,
        "source_observations": row.source_observations,
    }


def _metadata_document(row: LabelFreeContext) -> dict[str, object]:
    if not isinstance(row, LabelFreeContext):
        raise TypeError("stage metadata rows must be LabelFreeContext instances")
    return {
        "example_id": row.example_id,
        "assay_context_id": row.assay_context_id,
        "sequence_id": row.sequence_id,
        "sequence": row.sequence,
        "target": row.target,
        "gram": row.gram,
        "fold": row.fold,
    }


def _support_documents(ids: Iterable[str]) -> tuple[dict[str, object], ...]:
    values = tuple(ids)
    _id_stream_sha256(values, label="support sequence IDs")
    return tuple({"sequence_id": value} for value in values)


def _metadata_projection(row: ContextRow) -> LabelFreeContext:
    return LabelFreeContext(
        example_id=row.example_id,
        assay_context_id=row.assay_context_id,
        sequence_id=row.sequence_id,
        sequence=row.sequence,
        target=row.target,
        gram=row.gram,
        fold=row.fold,
    )


def _strict_json(payload: bytes, *, label: str) -> object:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be LF-terminated canonical JSON")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate object key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical compact JSON")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json(payload, label=label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not payload:
        raise ValueError(f"{label} must be nonempty canonical JSON Lines")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        value = _strict_json(line, label=f"{label} row {index}")
        if type(value) is not dict:
            raise ValueError(f"{label} row {index} must be a JSON object")
        rows.append(value)
    return tuple(rows)


def _parse_context_rows(payload: bytes, *, label: str) -> tuple[ContextRow, ...]:
    parsed: list[ContextRow] = []
    for index, document in enumerate(_strict_jsonl(payload, label=label)):
        _require_exact_fields(document, _CONTEXT_FIELDS, label=f"{label} row {index}")
        values = {field: document[field] for field in _CONTEXT_FIELDS}
        if any(
            not isinstance(values[field], str)
            for field in (
                "example_id",
                "assay_context_id",
                "sequence_id",
                "sequence",
                "target",
                "gram",
            )
        ):
            raise ValueError(f"{label} row {index} contains a non-string identity")
        row = ContextRow(**values)  # type: ignore[arg-type]
        if (
            _SHA256.fullmatch(row.example_id) is None
            or _SHA256.fullmatch(row.sequence_id) is None
            or row.assay_context_id != row.example_id
        ):
            raise ValueError(f"{label} row {index} has an invalid canonical identity")
        parsed.append(row)
    result = tuple(parsed)
    ids = tuple(row.example_id for row in result)
    if ids != tuple(sorted(set(ids))):
        raise ValueError(f"{label} must be ordered by unique ascending example ID")
    return result


def _parse_metadata_rows(payload: bytes, *, label: str) -> tuple[LabelFreeContext, ...]:
    parsed: list[LabelFreeContext] = []
    for index, document in enumerate(_strict_jsonl(payload, label=label)):
        _require_exact_fields(document, _METADATA_FIELDS, label=f"{label} row {index}")
        if any(
            not isinstance(document[field], str)
            for field in (
                "example_id",
                "assay_context_id",
                "sequence_id",
                "sequence",
                "target",
                "gram",
            )
        ):
            raise ValueError(f"{label} row {index} contains a non-string identity")
        fold = document["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
            raise ValueError(f"{label} row {index} has an invalid fold")
        sequence = document["sequence"]
        sequence_id = document["sequence_id"]
        target = document["target"]
        gram = document["gram"]
        if (
            canonicalize_sequence(sequence) != sequence
            or canonical_sequence_id(sequence) != sequence_id
            or _SHA256.fullmatch(sequence_id) is None
            or document["assay_context_id"] != document["example_id"]
            or _SHA256.fullmatch(document["example_id"]) is None
            or target not in TARGET_GRAM
            or TARGET_GRAM[target] != gram
        ):
            raise ValueError(f"{label} row {index} has an invalid canonical identity")
        parsed.append(
            LabelFreeContext(
                example_id=document["example_id"],
                assay_context_id=document["assay_context_id"],
                sequence_id=sequence_id,
                sequence=sequence,
                target=target,
                gram=gram,
                fold=fold,
            )
        )
    result = tuple(parsed)
    ids = tuple(row.example_id for row in result)
    if ids != tuple(sorted(set(ids))):
        raise ValueError(f"{label} must be ordered by unique ascending example ID")
    return result


def _parse_support_ids(payload: bytes, *, label: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, document in enumerate(_strict_jsonl(payload, label=label)):
        _require_exact_fields(document, _SUPPORT_FIELDS, label=f"{label} row {index}")
        value = document["sequence_id"]
        if not isinstance(value, str):
            raise ValueError(f"{label} row {index} sequence_id must be a string")
        values.append(value)
    result = tuple(values)
    _id_stream_sha256(result, label=label)
    return result


def _source_file_map(evidence: Gate1AuthenticationEvidence) -> Mapping[str, SourceFileEvidence]:
    if not isinstance(evidence, Gate1AuthenticationEvidence):
        raise TypeError("source evidence must be Gate1AuthenticationEvidence")
    if evidence.twins_byte_identical is not True:
        raise ValueError("stage source twins must be authenticated byte-identical")
    files: dict[str, SourceFileEvidence] = {}
    for item in evidence.files:
        if not isinstance(item, SourceFileEvidence) or item.logical_path in files:
            raise ValueError("source evidence contains invalid or duplicate file entries")
        _require_sha256(item.sha256, label=f"source file {item.logical_path}")
        if (
            not item.logical_path
            or item.logical_path.startswith("/")
            or ".." in Path(item.logical_path).parts
            or not re.fullmatch(r"0[0-7]{3}", item.mode)
            or isinstance(item.size_bytes, bool)
            or not isinstance(item.size_bytes, int)
            or item.size_bytes < 0
        ):
            raise ValueError("source evidence contains an invalid file identity")
        files[item.logical_path] = item
    if tuple(files) != tuple(sorted(files)):
        raise ValueError("source evidence files must be in ascending logical-path order")
    for required in ("SHA256SUMS", "gate1/SHA256SUMS"):
        if required not in files:
            raise ValueError(f"source evidence lacks required anchor {required}")
    _require_sha256(evidence.independent_receipt_sha256, label="independent receipt")
    _require_int(evidence.producer_job_id, label="producer job ID", minimum=1)
    _require_int(evidence.audit_job_id, label="audit job ID", minimum=1)
    return files


def source_anchors_document(evidence: Gate1AuthenticationEvidence) -> dict[str, object]:
    """Return the canonical path-free source anchors bound by the global seal."""

    files = _source_file_map(evidence)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": SOURCE_ANCHORS_ARTIFACT,
        "publication_top_sha256": files["SHA256SUMS"].sha256,
        "semantic_top_sha256": files["gate1/SHA256SUMS"].sha256,
        "independent_receipt_sha256": evidence.independent_receipt_sha256,
        "gate1_authentication_evidence": evidence.document(),
    }


def _validate_source_anchors(document: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    _require_exact_fields(document, _SOURCE_ANCHOR_FIELDS, label="stage source anchors")
    if (
        document["schema_version"] != SCHEMA_VERSION
        or type(document["schema_version"]) is not int
        or document["artifact"] != SOURCE_ANCHORS_ARTIFACT
    ):
        raise ValueError("stage source-anchor identity changed")
    publication = _require_sha256(document["publication_top_sha256"], label="publication top")
    semantic = _require_sha256(document["semantic_top_sha256"], label="semantic top")
    receipt = _require_sha256(document["independent_receipt_sha256"], label="independent receipt")
    raw_evidence = document["gate1_authentication_evidence"]
    if type(raw_evidence) is not dict:
        raise ValueError("stage source anchors lack Gate-1 authentication evidence")
    expected_evidence_fields = {
        "schema_version",
        "artifact",
        "producer_job_id",
        "audit_job_id",
        "twins_byte_identical",
        "independent_receipt_sha256",
        "files",
    }
    if set(raw_evidence) != expected_evidence_fields:
        raise ValueError("Gate-1 authentication evidence has an unexpected schema")
    if (
        raw_evidence["schema_version"] != 1
        or type(raw_evidence["schema_version"]) is not int
        or raw_evidence["artifact"] != "sequential_v2_authenticated_gate1_source"
        or raw_evidence["twins_byte_identical"] is not True
        or raw_evidence["independent_receipt_sha256"] != receipt
    ):
        raise ValueError("Gate-1 authentication evidence identity changed")
    _require_int(raw_evidence["producer_job_id"], label="producer job ID", minimum=1)
    _require_int(raw_evidence["audit_job_id"], label="audit job ID", minimum=1)
    raw_files = raw_evidence["files"]
    if type(raw_files) is not dict or not raw_files:
        raise ValueError("Gate-1 authentication evidence files must be a nonempty object")
    for logical, item in raw_files.items():
        if not isinstance(logical, str) or type(item) is not dict:
            raise ValueError("Gate-1 source file evidence is malformed")
        if set(item) != {"mode", "sha256", "size_bytes"}:
            raise ValueError("Gate-1 source file evidence has an unexpected schema")
        _require_sha256(item["sha256"], label=f"source file {logical}")
        _require_int(item["size_bytes"], label=f"source file {logical} size")
        if not isinstance(item["mode"], str) or re.fullmatch(r"0[0-7]{3}", item["mode"]) is None:
            raise ValueError("Gate-1 source file evidence has an invalid mode")
    if (
        "SHA256SUMS" not in raw_files
        or "gate1/SHA256SUMS" not in raw_files
        or raw_files["SHA256SUMS"]["sha256"] != publication
        or raw_files["gate1/SHA256SUMS"]["sha256"] != semantic
    ):
        raise ValueError("Gate-1 source anchors differ from their authenticated files")
    source_predecessors = (
        (SOURCE_PUBLICATION_PREDECESSOR, publication),
        (SOURCE_SEMANTIC_PREDECESSOR, semantic),
        (SOURCE_RECEIPT_PREDECESSOR, receipt),
    )
    return tuple(sorted(source_predecessors))


def _validate_panel(panel: Gate1Panel) -> tuple[tuple[ContextRow, ...], tuple[str, ...]]:
    if not isinstance(panel, Gate1Panel):
        raise TypeError("trusted stage source panel must be a Gate1Panel")
    contexts = tuple(panel.contexts)
    if not contexts or any(not isinstance(row, ContextRow) for row in contexts):
        raise ValueError("trusted stage panel contexts are empty or invalid")
    example_ids = tuple(row.example_id for row in contexts)
    _id_stream_sha256(example_ids, label="panel example IDs")
    if len(panel.contexts_by_fold) != 5:
        raise ValueError("trusted stage panel must have exactly five context folds")
    expected_by_fold = tuple(
        tuple(row for row in contexts if row.fold == fold) for fold in range(5)
    )
    if tuple(tuple(rows) for rows in panel.contexts_by_fold) != expected_by_fold:
        raise ValueError("trusted stage panel fold indexes differ from its contexts")
    sequence_folds: dict[str, int] = {}
    grams: dict[str, set[str]] = defaultdict(set)
    for row in contexts:
        if row.assay_context_id != row.example_id or _SHA256.fullmatch(row.example_id) is None:
            raise ValueError("trusted stage panel contains an invalid example identity")
        previous = sequence_folds.setdefault(row.sequence_id, row.fold)
        if previous != row.fold:
            raise ValueError("trusted stage panel assigns one sequence to multiple folds")
        grams[row.sequence_id].add(row.gram)
    support = tuple(
        sorted(
            sequence_id
            for sequence_id, observed in grams.items()
            if {"positive", "negative"}.issubset(observed)
        )
    )
    if support != tuple(panel.support_sequence_ids):
        raise ValueError("trusted stage panel support IDs differ from contextual support")
    if _id_stream_sha256(support, label="panel support IDs") != panel.support_sequence_ids_sha256:
        raise ValueError("trusted stage panel support digest changed")
    expected_support_by_fold = tuple(
        tuple(sequence_id for sequence_id in support if sequence_folds[sequence_id] == fold)
        for fold in range(5)
    )
    if tuple(tuple(ids) for ids in panel.support_sequence_ids_by_fold) != expected_support_by_fold:
        raise ValueError("trusted stage panel support fold indexes changed")
    return contexts, support


def _leaf_material(panel: Gate1Panel, *, spec: RotationSpec, role: str) -> _LeafMaterial:
    if role == PREPARE_ROLE:
        capability = prepare_capability(panel, spec=spec)
        data = {
            "base-contexts.jsonl": canonical_jsonl_bytes(
                _context_document(row) for row in capability.base_contexts
            ),
            "pool-metadata.jsonl": canonical_jsonl_bytes(
                _metadata_document(row) for row in capability.acquisition_metadata
            ),
            "pool-support-sequence-ids.jsonl": canonical_jsonl_bytes(
                _support_documents(capability.acquisition_support_sequence_ids)
            ),
        }
        streams = (
            (
                "base_example_ids",
                tuple(row.example_id for row in capability.base_contexts),
            ),
            (
                "pool_metadata_example_ids",
                tuple(row.example_id for row in capability.acquisition_metadata),
            ),
            ("pool_support_sequence_ids", capability.acquisition_support_sequence_ids),
        )
    elif role == POOL_OUTCOME_ROLE:
        vault = pool_outcome_vault(panel, spec=spec)
        data = {
            "contexts.jsonl": canonical_jsonl_bytes(
                _context_document(row) for row in vault.contexts
            )
        }
        streams = (("pool_outcome_example_ids", vault.allowed_example_ids),)
    elif role == OUTER_METADATA_ROLE:
        capability = outer_metadata_capability(panel, spec=spec)
        data = {
            "contexts.jsonl": canonical_jsonl_bytes(
                _metadata_document(row) for row in capability.contexts
            ),
            "support-sequence-ids.jsonl": canonical_jsonl_bytes(
                _support_documents(capability.support_sequence_ids)
            ),
        }
        streams = (
            ("outer_metadata_example_ids", capability.allowed_example_ids),
            ("outer_support_sequence_ids", capability.support_sequence_ids),
        )
    elif role == OUTER_OUTCOME_ROLE:
        vault = outer_outcome_vault(panel, spec=spec)
        data = {
            "contexts.jsonl": canonical_jsonl_bytes(
                _context_document(row) for row in vault.contexts
            )
        }
        streams = (("outer_outcome_example_ids", vault.allowed_example_ids),)
    else:
        raise ValueError(f"unknown stage role: {role!r}")
    row_counts = tuple(
        (path, len(_strict_jsonl(data[path], label=f"staged {path}")))
        for path in LEAF_DATA_PAYLOAD_PATHS[role]
    )
    id_streams = tuple(
        (
            name,
            IdStreamBinding(
                count=len(ids),
                sha256=_id_stream_sha256(ids, label=f"{spec.rotation_id} {name}"),
            ),
        )
        for name, ids in streams
    )
    provisional = StageLeafIndex(
        spec=spec,
        role=role,
        relative_path=leaf_relative_path(spec, role),
        leaf_artifact=LEAF_ARTIFACTS[role],
        leaf_seal_sha256="0" * 64,
        payload_paths=LEAF_PAYLOAD_PATHS[role],
        row_counts=row_counts,
        id_streams=id_streams,
    )
    return _LeafMaterial(
        spec=spec,
        role=role,
        payloads={
            "capability.json": canonical_json_bytes(provisional.capability_document()),
            **data,
        },
        row_counts=row_counts,
        id_streams=id_streams,
    )


def _leaf_metadata(entry: StageLeafIndex, source_anchors_sha256: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rotation_id": entry.spec.rotation_id,
        "role": entry.role,
        "source_anchors_sha256": source_anchors_sha256,
    }


def _entry_from_material(material: _LeafMaterial, seal: PhaseSeal) -> StageLeafIndex:
    return StageLeafIndex(
        spec=material.spec,
        role=material.role,
        relative_path=leaf_relative_path(material.spec, material.role),
        leaf_artifact=LEAF_ARTIFACTS[material.role],
        leaf_seal_sha256=seal.seal_sha256,
        payload_paths=LEAF_PAYLOAD_PATHS[material.role],
        row_counts=material.row_counts,
        id_streams=material.id_streams,
    )


def _summary_document(
    contexts: Sequence[ContextRow],
    support_sequence_ids: Sequence[str],
    entries: Sequence[StageLeafIndex],
    *,
    source_anchors_sha256: str,
    capability_index_sha256: str,
) -> dict[str, object]:
    rows = tuple(contexts)
    support = tuple(support_sequence_ids)
    example_ids = tuple(row.example_id for row in rows)
    sequence_ids = tuple(sorted({row.sequence_id for row in rows}))
    sequence_fold: dict[str, int] = {}
    for row in rows:
        previous = sequence_fold.setdefault(row.sequence_id, row.fold)
        if previous != row.fold:
            raise ValueError("stage summary found a sequence assigned to multiple folds")
    associations: Counter[str] = Counter()
    role_leaves: Counter[str] = Counter()
    for entry in entries:
        role_leaves[entry.role] += 1
        for name, binding in entry.id_streams:
            associations[name] += binding.count
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": STAGE_SUMMARY_ARTIFACT,
        "source_anchors_sha256": source_anchors_sha256,
        "capability_index_sha256": capability_index_sha256,
        "rotation_count": EXPECTED_ROTATIONS,
        "leaf_count": len(entries),
        "panel_context_count": len(rows),
        "panel_example_ids_sha256": _id_stream_sha256(example_ids, label="stage panel example IDs"),
        "panel_sequence_count": len(sequence_ids),
        "panel_sequence_ids_sha256": _id_stream_sha256(
            sequence_ids, label="stage panel sequence IDs"
        ),
        "contexts_by_fold": [sum(row.fold == fold for row in rows) for fold in range(5)],
        "sequences_by_fold": [
            sum(sequence_fold[sequence_id] == fold for sequence_id in sequence_ids)
            for fold in range(5)
        ],
        "support_sequence_count": len(support),
        "support_sequence_ids_sha256": _id_stream_sha256(
            support, label="stage support sequence IDs"
        ),
        "support_sequences_by_fold": [
            sum(sequence_fold[sequence_id] == fold for sequence_id in support) for fold in range(5)
        ],
        "role_leaf_counts": {role: role_leaves[role] for role in ROLE_ORDER},
        "role_association_censuses": {
            name: associations[name]
            for name in (
                "base_example_ids",
                "pool_metadata_example_ids",
                "pool_support_sequence_ids",
                "pool_outcome_example_ids",
                "outer_metadata_example_ids",
                "outer_support_sequence_ids",
                "outer_outcome_example_ids",
            )
        },
    }


def _parse_id_streams(
    raw: object, *, role: str, row_counts: Mapping[str, int], label: str
) -> tuple[tuple[str, IdStreamBinding], ...]:
    if type(raw) is not dict or tuple(raw) != LEAF_ID_STREAM_NAMES[role]:
        raise ValueError(f"{label} ID-stream inventory or order changed")
    expected_counts = _stream_row_count_bindings(role, row_counts)
    result: list[tuple[str, IdStreamBinding]] = []
    for name in LEAF_ID_STREAM_NAMES[role]:
        item = raw[name]
        if type(item) is not dict or set(item) != {"count", "sha256"}:
            raise ValueError(f"{label} ID stream {name} has an unexpected schema")
        binding = IdStreamBinding(
            count=_require_int(item["count"], label=f"{label} {name} count", minimum=1),
            sha256=_require_sha256(item["sha256"], label=f"{label} {name}"),
        )
        if binding.count != expected_counts[name]:
            raise ValueError(f"{label} ID stream {name} differs from its row census")
        result.append((name, binding))
    return tuple(result)


def _parse_leaf_index(payload: bytes) -> tuple[StageLeafIndex, ...]:
    documents = _strict_jsonl(payload, label="stage capability index")
    expected_order = tuple((spec, role) for spec in ordered_rotations() for role in ROLE_ORDER)
    if len(documents) != len(expected_order):
        raise ValueError("stage capability index must contain exactly 80 rows")
    result: list[StageLeafIndex] = []
    for index, (document, (expected_spec, expected_role)) in enumerate(
        zip(documents, expected_order, strict=True)
    ):
        label = f"stage capability index row {index}"
        _require_exact_fields(document, _INDEX_FIELDS, label=label)
        if (
            document["schema_version"] != SCHEMA_VERSION
            or type(document["schema_version"]) is not int
        ):
            raise ValueError(f"{label} version changed")
        if document["rotation_id"] != expected_spec.rotation_id:
            raise ValueError(f"{label} differs from frozen rotation order")
        spec = rotation_by_id(document["rotation_id"])
        role = document["role"]
        if role != expected_role:
            raise ValueError(f"{label} differs from frozen role order")
        if (
            type(document["payload_paths"]) is not list
            or tuple(document["payload_paths"]) != LEAF_PAYLOAD_PATHS[role]
        ):
            raise ValueError(f"{label} payload inventory changed")
        raw_counts = document["row_counts"]
        if type(raw_counts) is not dict or tuple(raw_counts) != LEAF_DATA_PAYLOAD_PATHS[role]:
            raise ValueError(f"{label} row-count inventory or order changed")
        row_counts = tuple(
            (
                path,
                _require_int(raw_counts[path], label=f"{label} {path} count", minimum=1),
            )
            for path in LEAF_DATA_PAYLOAD_PATHS[role]
        )
        id_streams = _parse_id_streams(
            document["id_streams"],
            role=role,
            row_counts=dict(row_counts),
            label=label,
        )
        result.append(
            StageLeafIndex(
                spec=spec,
                role=role,
                relative_path=document["relative_path"],
                leaf_artifact=document["leaf_artifact"],
                leaf_seal_sha256=document["leaf_seal_sha256"],
                payload_paths=tuple(document["payload_paths"]),
                row_counts=row_counts,
                id_streams=id_streams,
            )
        )
    if len({entry.relative_path for entry in result}) != len(result):
        raise ValueError("stage capability index contains duplicate leaf paths")
    return tuple(result)


def _parse_capability(payload: bytes, *, entry: StageLeafIndex) -> None:
    document = _strict_json_object(payload, label=f"{entry.relative_path} capability")
    _require_exact_fields(document, _CAPABILITY_FIELDS, label="stage leaf capability")
    if canonical_json_bytes(document) != canonical_json_bytes(entry.capability_document()):
        raise ValueError("stage leaf capability differs from the global capability index")


def _read_bound_payload(root: Path, relative: str, expected_sha256: str | None) -> bytes:
    path = root / relative
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open sealed payload {relative}") from error
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
        ):
            raise ValueError(f"sealed payload {relative} must be a one-link mode-0444 file")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = os.lstat(path)

    def fingerprint(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(named):
        raise RuntimeError(f"sealed payload {relative} changed while it was read")
    if expected_sha256 is not None and digest.hexdigest() != _require_sha256(
        expected_sha256, label=f"expected {relative}"
    ):
        raise ValueError(f"sealed payload checksum differs: {relative}")
    return b"".join(chunks)


def _verify_stage_leaf_with_bindings(
    root: Path,
    *,
    expected_leaf: StageLeafIndex,
    source_predecessors: Mapping[str, str],
    source_anchors_sha256: str,
) -> PhaseSeal:
    seal = verify_phase(
        root,
        expected_artifact=expected_leaf.leaf_artifact,
        expected_payload_paths=expected_leaf.payload_paths,
        expected_predecessor_seals=source_predecessors,
        expected_seal_sha256=expected_leaf.leaf_seal_sha256,
    )
    expected_metadata = canonical_json_bytes(_leaf_metadata(expected_leaf, source_anchors_sha256))
    if seal.metadata_json != expected_metadata:
        raise ValueError("stage leaf receipt metadata differs from its capability index")
    return seal


def authenticate_stage_leaf_for_controller(
    root: str | Path,
    *,
    expected_leaf: StageLeafIndex,
    expected_source_anchors: Mapping[str, object],
) -> AuthenticatedLeafCapsule:
    """Controller-only path access; return a rootless, one-role byte capsule."""

    if not isinstance(expected_leaf, StageLeafIndex):
        raise TypeError("expected_leaf must be a StageLeafIndex from the sealed global index")
    anchor_bytes = canonical_json_bytes(expected_source_anchors)
    anchor_document = _strict_json_object(anchor_bytes, label="expected source anchors")
    predecessors = dict(_validate_source_anchors(anchor_document))
    seal = _verify_stage_leaf_with_bindings(
        Path(os.path.abspath(os.fspath(root))),
        expected_leaf=expected_leaf,
        source_predecessors=predecessors,
        source_anchors_sha256=sha256_bytes(anchor_bytes),
    )
    return AuthenticatedLeafCapsule(
        entry=expected_leaf,
        seal=seal,
        source_anchors_sha256=sha256_bytes(anchor_bytes),
        source_predecessors=tuple(sorted(predecessors.items())),
    )


def _load_leaf(stage: TrustedStage, entry: StageLeafIndex) -> _LoadedLeaf:
    seal = _verify_stage_leaf_with_bindings(
        stage.root / entry.relative_path,
        expected_leaf=entry,
        source_predecessors=dict(stage.source_predecessors),
        source_anchors_sha256=stage.source_anchors_sha256,
    )
    return _decode_leaf(entry, seal)


def _decode_leaf(entry: StageLeafIndex, seal: PhaseSeal) -> _LoadedLeaf:
    payloads = {path: seal.read_payload_bytes(path) for path in entry.payload_paths}
    _parse_capability(payloads["capability.json"], entry=entry)
    if entry.role == PREPARE_ROLE:
        base = _parse_context_rows(payloads["base-contexts.jsonl"], label="prepare base contexts")
        metadata = _parse_metadata_rows(
            payloads["pool-metadata.jsonl"], label="prepare pool metadata"
        )
        support = _parse_support_ids(
            payloads["pool-support-sequence-ids.jsonl"],
            label="prepare pool support sequence IDs",
        )
        loaded = _LoadedLeaf(
            entry=entry,
            base_contexts=base,
            metadata_contexts=metadata,
            support_sequence_ids=support,
        )
    elif entry.role == POOL_OUTCOME_ROLE:
        loaded = _LoadedLeaf(
            entry=entry,
            outcome_contexts=_parse_context_rows(
                payloads["contexts.jsonl"], label="pool outcome contexts"
            ),
        )
    elif entry.role == OUTER_METADATA_ROLE:
        loaded = _LoadedLeaf(
            entry=entry,
            metadata_contexts=_parse_metadata_rows(
                payloads["contexts.jsonl"], label="outer metadata contexts"
            ),
            support_sequence_ids=_parse_support_ids(
                payloads["support-sequence-ids.jsonl"],
                label="outer support sequence IDs",
            ),
        )
    elif entry.role == OUTER_OUTCOME_ROLE:
        loaded = _LoadedLeaf(
            entry=entry,
            outcome_contexts=_parse_context_rows(
                payloads["contexts.jsonl"], label="outer outcome contexts"
            ),
        )
    else:
        raise AssertionError("unreachable stage role")
    _validate_loaded_leaf(loaded)
    return loaded


def _validate_loaded_leaf(loaded: _LoadedLeaf) -> None:
    entry = loaded.entry
    counts = entry.row_count_map
    streams = entry.id_stream_map
    if entry.role == PREPARE_ROLE:
        base_ids = tuple(row.example_id for row in loaded.base_contexts)
        metadata_ids = tuple(row.example_id for row in loaded.metadata_contexts)
        if (
            len(base_ids) != counts["base-contexts.jsonl"]
            or len(metadata_ids) != counts["pool-metadata.jsonl"]
            or len(loaded.support_sequence_ids) != counts["pool-support-sequence-ids.jsonl"]
            or {row.fold for row in loaded.base_contexts} != set(entry.spec.base_folds)
            or any(row.fold not in entry.spec.base_folds for row in loaded.base_contexts)
            or any(row.fold != entry.spec.pool_fold for row in loaded.metadata_contexts)
            or set(base_ids) & set(metadata_ids)
        ):
            raise ValueError("prepare stage leaf has the wrong role, census, or fold partition")
        values = {
            "base_example_ids": base_ids,
            "pool_metadata_example_ids": metadata_ids,
            "pool_support_sequence_ids": loaded.support_sequence_ids,
        }
    elif entry.role == POOL_OUTCOME_ROLE:
        ids = tuple(row.example_id for row in loaded.outcome_contexts)
        if len(ids) != counts["contexts.jsonl"] or any(
            row.fold != entry.spec.pool_fold for row in loaded.outcome_contexts
        ):
            raise ValueError("pool outcome stage leaf has the wrong role, census, or fold")
        values = {"pool_outcome_example_ids": ids}
    elif entry.role == OUTER_METADATA_ROLE:
        ids = tuple(row.example_id for row in loaded.metadata_contexts)
        if (
            len(ids) != counts["contexts.jsonl"]
            or len(loaded.support_sequence_ids) != counts["support-sequence-ids.jsonl"]
            or any(row.fold != entry.spec.outer_fold for row in loaded.metadata_contexts)
        ):
            raise ValueError("outer metadata stage leaf has the wrong role, census, or fold")
        values = {
            "outer_metadata_example_ids": ids,
            "outer_support_sequence_ids": loaded.support_sequence_ids,
        }
    elif entry.role == OUTER_OUTCOME_ROLE:
        ids = tuple(row.example_id for row in loaded.outcome_contexts)
        if len(ids) != counts["contexts.jsonl"] or any(
            row.fold != entry.spec.outer_fold for row in loaded.outcome_contexts
        ):
            raise ValueError("outer outcome stage leaf has the wrong role, census, or fold")
        values = {"outer_outcome_example_ids": ids}
    else:
        raise AssertionError("unreachable stage role")
    for name, ids in values.items():
        binding = streams[name]
        if len(ids) != binding.count or _id_stream_sha256(ids, label=name) != binding.sha256:
            raise ValueError(f"stage leaf ID stream differs from its index: {name}")


def _assert_real_directory(path: Path, *, mode: int, label: str) -> None:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise ValueError(f"{label} must be a non-symbolic mode-{mode:04o} directory")


def _assert_aggregate_topology(root: Path, *, root_mode: int = 0o555) -> None:
    _assert_real_directory(root, mode=root_mode, label="trusted stage root")
    entries = tuple(sorted(entry.name for entry in os.scandir(root)))
    if entries != (GLOBAL_DIRECTORY, ROTATIONS_DIRECTORY):
        raise ValueError("trusted stage root has an unexpected inventory")
    rotations_root = root / ROTATIONS_DIRECTORY
    _assert_real_directory(rotations_root, mode=0o555, label="trusted stage rotations")
    expected_rotations = tuple(sorted(spec.rotation_id for spec in ordered_rotations()))
    observed_rotations = tuple(sorted(entry.name for entry in os.scandir(rotations_root)))
    if observed_rotations != expected_rotations:
        raise ValueError("trusted stage rotation-directory inventory changed")
    for spec in ordered_rotations():
        rotation_root = rotations_root / spec.rotation_id
        _assert_real_directory(rotation_root, mode=0o555, label="trusted stage rotation")
        observed_roles = tuple(sorted(entry.name for entry in os.scandir(rotation_root)))
        if observed_roles != tuple(sorted(ROLE_ORDER)):
            raise ValueError("trusted stage rotation role inventory changed")
        for role in ROLE_ORDER:
            _assert_real_directory(
                rotation_root / role,
                mode=0o555,
                label="trusted stage role leaf",
            )
    _assert_real_directory(root / GLOBAL_DIRECTORY, mode=0o555, label="trusted stage global")


def _validate_summary_shape(
    document: Mapping[str, object],
    *,
    entries: Sequence[StageLeafIndex],
    source_anchors_sha256: str,
    capability_index_sha256: str,
) -> None:
    _require_exact_fields(document, _SUMMARY_FIELDS, label="trusted stage summary")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != STAGE_SUMMARY_ARTIFACT
        or document["source_anchors_sha256"] != source_anchors_sha256
        or document["capability_index_sha256"] != capability_index_sha256
        or document["rotation_count"] != EXPECTED_ROTATIONS
        or type(document["rotation_count"]) is not int
        or document["leaf_count"] != EXPECTED_ROTATIONS * len(ROLE_ORDER)
        or type(document["leaf_count"]) is not int
    ):
        raise ValueError("trusted stage summary identity or global census changed")
    for key in ("contexts_by_fold", "sequences_by_fold", "support_sequences_by_fold"):
        value = document[key]
        if (
            type(value) is not list
            or len(value) != 5
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value
            )
        ):
            raise ValueError(f"trusted stage summary {key} is invalid")
    for key in (
        "panel_context_count",
        "panel_sequence_count",
        "support_sequence_count",
    ):
        _require_int(document[key], label=f"trusted stage summary {key}", minimum=1)
    for key in (
        "panel_example_ids_sha256",
        "panel_sequence_ids_sha256",
        "support_sequence_ids_sha256",
    ):
        _require_sha256(document[key], label=f"trusted stage summary {key}")
    expected_role_counts = {role: EXPECTED_ROTATIONS for role in ROLE_ORDER}
    if document["role_leaf_counts"] != expected_role_counts:
        raise ValueError("trusted stage summary role-leaf census changed")
    associations: Counter[str] = Counter()
    for entry in entries:
        for name, binding in entry.id_streams:
            associations[name] += binding.count
    expected_associations = {
        name: associations[name]
        for name in (
            "base_example_ids",
            "pool_metadata_example_ids",
            "pool_support_sequence_ids",
            "pool_outcome_example_ids",
            "outer_metadata_example_ids",
            "outer_support_sequence_ids",
            "outer_outcome_example_ids",
        )
    }
    if document["role_association_censuses"] != expected_associations:
        raise ValueError("trusted stage summary role-association census changed")
    if (
        sum(document["contexts_by_fold"]) != document["panel_context_count"]
        or sum(document["sequences_by_fold"]) != document["panel_sequence_count"]
        or sum(document["support_sequences_by_fold"]) != document["support_sequence_count"]
    ):
        raise ValueError("trusted stage summary fold censuses do not sum to their totals")


def _require_boundary_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256 string")
    return value


def _decode_stage_manifest_seal(
    seal: PhaseSeal,
    *,
    expected_global_seal_sha256: str,
) -> _DecodedStageManifest:
    """Reconstruct the complete label-free stage-global graph from captured bytes."""

    expected_seal = _require_boundary_sha256(
        expected_global_seal_sha256,
        label="expected stage-global seal",
    )
    authenticated = verify_phase_capability(
        seal,
        expected_artifact=STAGE_ARTIFACT,
        expected_payload_paths=GLOBAL_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal,
    )
    payloads = {path: authenticated.read_payload_bytes(path) for path in GLOBAL_PAYLOAD_PATHS}
    source_anchor_bytes = payloads["source-anchors.json"]
    source_anchor_document = _strict_json_object(
        source_anchor_bytes,
        label="stage capability source anchors",
    )
    source_predecessors = _validate_source_anchors(source_anchor_document)
    source_anchors_sha256 = sha256_bytes(source_anchor_bytes)
    leaves = _parse_leaf_index(payloads["capability-index.jsonl"])
    predecessors = {
        **dict(source_predecessors),
        **{entry.relative_path: entry.leaf_seal_sha256 for entry in leaves},
    }
    verified = verify_phase_capability(
        authenticated,
        expected_artifact=STAGE_ARTIFACT,
        expected_payload_paths=GLOBAL_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=expected_seal,
    )
    rotations = _strict_jsonl(payloads["rotations.jsonl"], label="stage capability rotations")
    policies = _strict_jsonl(payloads["policy-runs.jsonl"], label="stage capability policy runs")
    validate_rotation_documents(rotations)
    validate_policy_run_documents(policies)
    census = _strict_json_object(
        payloads["protocol-census.json"],
        label="stage capability protocol census",
    )
    if canonical_json_bytes(census) != canonical_json_bytes(protocol_census()):
        raise ValueError("stage capability protocol census differs from the frozen graph")
    summary_payload = payloads["stage-summary.json"]
    summary = _strict_json_object(summary_payload, label="stage capability summary")
    _validate_summary_shape(
        summary,
        entries=leaves,
        source_anchors_sha256=source_anchors_sha256,
        capability_index_sha256=sha256_bytes(payloads["capability-index.jsonl"]),
    )
    expected_metadata = canonical_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "rotation_count": EXPECTED_ROTATIONS,
            "leaf_count": len(leaves),
            "source_anchors_sha256": source_anchors_sha256,
        }
    )
    if verified.metadata_json != expected_metadata:
        raise ValueError("stage capability global receipt metadata changed")
    return _DecodedStageManifest(
        seal=verified,
        source_anchors_json=source_anchor_bytes,
        source_anchors_sha256=source_anchors_sha256,
        source_predecessors=source_predecessors,
        leaves=leaves,
        stage_summary_json=summary_payload,
    )


def verify_stage_manifest_capability(
    seal: PhaseSeal,
    *,
    expected_global_seal_sha256: str,
) -> StageManifestCapability:
    """Authenticate a pathless stage-global capability under external authority."""

    decoded = _decode_stage_manifest_seal(
        seal,
        expected_global_seal_sha256=expected_global_seal_sha256,
    )
    return StageManifestCapability(seal=decoded.seal)


def authenticate_stage_manifest_for_controller(
    stage: TrustedStage,
    *,
    expected_global_seal_sha256: str,
) -> StageManifestCapability:
    """Strip a verified controller handle to its already captured global bytes."""

    if type(stage) is not TrustedStage:
        raise TypeError("stage manifest capture requires an exact TrustedStage")
    expected = _require_boundary_sha256(
        expected_global_seal_sha256,
        label="expected stage-global seal",
    )
    if stage.global_seal_sha256 != expected or stage.global_seal.seal_sha256 != expected:
        raise ValueError("trusted stage differs from the expected stage-global seal")
    return verify_stage_manifest_capability(
        stage.global_seal,
        expected_global_seal_sha256=expected,
    )


def verify_stage_manifest(
    root: str | Path,
    *,
    expected_source_anchors: Mapping[str, object],
    expected_global_seal_sha256: str,
) -> TrustedStage:
    """Authenticate the outcome-free global graph without opening any role payload."""

    requested = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_chain(requested, label="trusted stage root")
    _assert_aggregate_topology(requested)
    if not isinstance(expected_source_anchors, Mapping):
        raise TypeError("expected source anchors must be a mapping")
    expected_anchor_bytes = canonical_json_bytes(expected_source_anchors)
    expected_anchor_document = _strict_json_object(
        expected_anchor_bytes, label="expected source anchors"
    )
    source_predecessors = _validate_source_anchors(expected_anchor_document)
    source_anchors_sha256 = sha256_bytes(expected_anchor_bytes)

    global_root = requested / GLOBAL_DIRECTORY
    preliminary_index = _read_bound_payload(global_root, "capability-index.jsonl", None)
    entries = _parse_leaf_index(preliminary_index)
    predecessors = {
        **dict(source_predecessors),
        **{entry.relative_path: entry.leaf_seal_sha256 for entry in entries},
    }
    global_seal = verify_phase(
        global_root,
        expected_artifact=STAGE_ARTIFACT,
        expected_payload_paths=GLOBAL_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessors,
        expected_seal_sha256=expected_global_seal_sha256,
    )
    decoded = _decode_stage_manifest_seal(
        global_seal,
        expected_global_seal_sha256=expected_global_seal_sha256,
    )
    if decoded.seal.read_payload_bytes("capability-index.jsonl") != preliminary_index:
        raise RuntimeError("stage capability index changed while the global phase was verified")
    if decoded.source_anchors_json != expected_anchor_bytes:
        raise ValueError("trusted stage source anchors differ from the expected source")
    if (
        decoded.source_anchors_sha256 != source_anchors_sha256
        or decoded.source_predecessors != source_predecessors
    ):
        raise ValueError("trusted stage source authority changed during capture")
    return TrustedStage(
        root=requested.resolve(strict=True),
        global_seal=decoded.seal,
        global_seal_sha256=decoded.seal.seal_sha256,
        source_anchors_sha256=decoded.source_anchors_sha256,
        source_predecessors=decoded.source_predecessors,
        leaves=decoded.leaves,
        stage_summary_json=decoded.stage_summary_json,
    )


def _validate_loaded_stage(stage: TrustedStage, loaded: Sequence[_LoadedLeaf]) -> None:
    if len(loaded) != EXPECTED_ROTATIONS * len(ROLE_ORDER):
        raise ValueError("trusted stage semantic audit did not load exactly 80 leaves")
    by_key = {(item.entry.spec, item.entry.role): item for item in loaded}
    if len(by_key) != len(loaded):
        raise ValueError("trusted stage semantic audit found duplicate role leaves")
    canonical_rows: dict[str, ContextRow] = {}
    role_occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    support_occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    for item in loaded:
        if item.entry.role == PREPARE_ROLE:
            for row in item.base_contexts:
                role_occurrences["base_example_ids"][row.example_id] += 1
                previous = canonical_rows.setdefault(row.example_id, row)
                if previous != row:
                    raise ValueError("stage repeats one labeled context with different content")
            for row in item.metadata_contexts:
                role_occurrences["pool_metadata_example_ids"][row.example_id] += 1
            for sequence_id in item.support_sequence_ids:
                support_occurrences["pool_support_sequence_ids"][sequence_id] += 1
        elif item.entry.role == POOL_OUTCOME_ROLE:
            for row in item.outcome_contexts:
                role_occurrences["pool_outcome_example_ids"][row.example_id] += 1
                previous = canonical_rows.setdefault(row.example_id, row)
                if previous != row:
                    raise ValueError("stage repeats one labeled context with different content")
        elif item.entry.role == OUTER_METADATA_ROLE:
            for row in item.metadata_contexts:
                role_occurrences["outer_metadata_example_ids"][row.example_id] += 1
            for sequence_id in item.support_sequence_ids:
                support_occurrences["outer_support_sequence_ids"][sequence_id] += 1
        else:
            for row in item.outcome_contexts:
                role_occurrences["outer_outcome_example_ids"][row.example_id] += 1
                previous = canonical_rows.setdefault(row.example_id, row)
                if previous != row:
                    raise ValueError("stage repeats one labeled context with different content")

    contexts = tuple(sorted(canonical_rows.values(), key=lambda row: row.example_id))
    if not contexts:
        raise ValueError("trusted stage semantic audit found no contexts")
    for spec in ordered_rotations():
        prepare = by_key[(spec, PREPARE_ROLE)]
        pool = by_key[(spec, POOL_OUTCOME_ROLE)]
        outer_metadata = by_key[(spec, OUTER_METADATA_ROLE)]
        outer = by_key[(spec, OUTER_OUTCOME_ROLE)]
        if tuple(row.example_id for row in prepare.metadata_contexts) != tuple(
            row.example_id for row in pool.outcome_contexts
        ):
            raise ValueError("stage pool metadata and outcome vault IDs differ")
        if tuple(row.example_id for row in outer_metadata.metadata_contexts) != tuple(
            row.example_id for row in outer.outcome_contexts
        ):
            raise ValueError("stage outer metadata and outcome vault IDs differ")
        pool_support = _support_from_contexts(pool.outcome_contexts)
        outer_support = _support_from_contexts(outer.outcome_contexts)
        if prepare.support_sequence_ids != pool_support:
            raise ValueError("stage pool support IDs differ from its outcome contexts")
        if outer_metadata.support_sequence_ids != outer_support:
            raise ValueError("stage outer support IDs differ from its outcome contexts")
        role_ids = (
            {row.example_id for row in prepare.base_contexts},
            {row.example_id for row in pool.outcome_contexts},
            {row.example_id for row in outer.outcome_contexts},
        )
        if any(
            role_ids[left] & role_ids[right] for left, right in ((0, 1), (0, 2), (1, 2))
        ) or set().union(*role_ids) != set(canonical_rows):
            raise ValueError("stage rotation roles do not exactly partition the panel")

    for item in loaded:
        for row in item.metadata_contexts:
            expected = canonical_rows.get(row.example_id)
            if expected is None or row != _metadata_projection(expected):
                raise ValueError("stage label-free metadata differs from its source context")
    expected_context_occurrences = {
        "base_example_ids": 12,
        "pool_metadata_example_ids": 4,
        "pool_outcome_example_ids": 4,
        "outer_metadata_example_ids": 4,
        "outer_outcome_example_ids": 4,
    }
    for role, expected_count in expected_context_occurrences.items():
        if set(role_occurrences[role]) != set(canonical_rows) or set(
            role_occurrences[role].values()
        ) != {expected_count}:
            raise ValueError("stage global context role-association census changed")
    support = _support_from_contexts(contexts)
    for role in ("pool_support_sequence_ids", "outer_support_sequence_ids"):
        if set(support_occurrences[role]) != set(support) or set(
            support_occurrences[role].values()
        ) != {4}:
            raise ValueError("stage global support role-association census changed")
    summary = _strict_json_object(stage.stage_summary_json, label="trusted stage summary")
    index_bytes = canonical_jsonl_bytes(entry.document() for entry in stage.leaves)
    expected_summary = _summary_document(
        contexts,
        support,
        stage.leaves,
        source_anchors_sha256=stage.source_anchors_sha256,
        capability_index_sha256=sha256_bytes(index_bytes),
    )
    if canonical_json_bytes(summary) != canonical_json_bytes(expected_summary):
        raise ValueError("trusted stage summary differs from reconstructed role leaves")


def _support_from_contexts(rows: Iterable[ContextRow]) -> tuple[str, ...]:
    grams: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grams[row.sequence_id].add(row.gram)
    result = tuple(
        sorted(
            sequence_id
            for sequence_id, observed in grams.items()
            if {"positive", "negative"}.issubset(observed)
        )
    )
    _id_stream_sha256(result, label="context-derived support IDs")
    return result


def verify_trusted_stage(
    root: str | Path,
    *,
    expected_source_anchors: Mapping[str, object],
    expected_global_seal_sha256: str,
) -> TrustedStage:
    """Fully authenticate all 80 leaves and their cross-role semantics."""

    stage = verify_stage_manifest(
        root,
        expected_source_anchors=expected_source_anchors,
        expected_global_seal_sha256=expected_global_seal_sha256,
    )
    loaded = tuple(_load_leaf(stage, entry) for entry in stage.leaves)
    _validate_loaded_stage(stage, loaded)
    return stage


def _decode_capsule(
    capsule: AuthenticatedLeafCapsule,
    *,
    role: str,
) -> _LoadedLeaf:
    if not isinstance(capsule, AuthenticatedLeafCapsule):
        raise TypeError("role materialization requires an AuthenticatedLeafCapsule")
    if capsule.entry.role != role:
        raise ValueError(f"leaf capsule has role {capsule.entry.role!r}, expected {role!r}")
    return _decode_leaf(capsule.entry, capsule.seal)


def prepare_capability_from_capsule(
    capsule: AuthenticatedLeafCapsule,
) -> PrepareCapability:
    """Materialize prepare data from rootless authenticated bytes only."""

    expected_leaf = capsule.entry
    if expected_leaf.role != PREPARE_ROLE:
        raise ValueError("prepare loader requires a prepare-capability leaf")
    loaded = _decode_capsule(capsule, role=PREPARE_ROLE)
    return PrepareCapability(
        spec=expected_leaf.spec,
        base_contexts=loaded.base_contexts,
        acquisition_metadata=loaded.metadata_contexts,
        acquisition_support_sequence_ids=loaded.support_sequence_ids,
        allowed_base_example_ids=tuple(row.example_id for row in loaded.base_contexts),
        allowed_acquisition_metadata_example_ids=tuple(
            row.example_id for row in loaded.metadata_contexts
        ),
    )


def pool_outcome_vault_from_capsule(
    capsule: AuthenticatedLeafCapsule,
) -> PoolOutcomeVault:
    """Materialize a pool vault from rootless authenticated bytes only."""

    expected_leaf = capsule.entry
    if expected_leaf.role != POOL_OUTCOME_ROLE:
        raise ValueError("pool outcome loader requires a pool-outcome-vault leaf")
    loaded = _decode_capsule(capsule, role=POOL_OUTCOME_ROLE)
    ids = tuple(row.example_id for row in loaded.outcome_contexts)
    return PoolOutcomeVault(
        spec=expected_leaf.spec,
        contexts=loaded.outcome_contexts,
        allowed_example_ids=ids,
    )


def _require_exact_pool_outcome_vault(
    vault: object,
    *,
    spec: RotationSpec,
) -> PoolOutcomeVault:
    if type(vault) is not PoolOutcomeVault:
        raise TypeError("pool outcome materialization must return an exact PoolOutcomeVault")
    canonical_spec = _require_frozen_rotation_boundary(
        vault.spec,
        label="pool outcome vault rotation",
    )
    if canonical_spec != spec:
        raise ValueError("pool outcome vault has the wrong rotation")
    if type(vault.contexts) is not tuple or any(
        type(row) is not ContextRow for row in vault.contexts
    ):
        raise TypeError("pool outcome vault contexts must be an exact ContextRow tuple")
    for row in vault.contexts:
        if any(
            type(getattr(row, field)) is not str
            for field in (
                "example_id",
                "assay_context_id",
                "sequence_id",
                "sequence",
                "target",
                "gram",
            )
        ) or any(
            type(getattr(row, field)) is not int
            for field in ("fold", "label", "source_observations")
        ):
            raise TypeError("pool outcome vault rows must use exact scalar field types")
    if type(vault.allowed_example_ids) is not tuple or any(
        type(value) is not str for value in vault.allowed_example_ids
    ):
        raise TypeError("pool outcome vault allowed IDs must be an exact string tuple")
    ids = tuple(row.example_id for row in vault.contexts)
    if (
        not ids
        or ids != tuple(sorted(set(ids)))
        or vault.allowed_example_ids != ids
        or any(row.fold != spec.pool_fold for row in vault.contexts)
    ):
        raise ValueError("pool outcome vault differs from its exact acquisition fold")
    support = _support_from_contexts(vault.contexts)
    if len(support) != EXPECTED_SUPPORT_BY_FOLD[spec.pool_fold]:
        raise ValueError("pool outcome vault support census differs from the frozen protocol")
    return vault


def pool_outcome_vault_from_stage_capabilities(
    stage_capability: StageManifestCapability,
    capsule: AuthenticatedLeafCapsule,
    *,
    spec: RotationSpec,
    expected_stage_global_seal_sha256: str,
) -> PoolOutcomeVault:
    """Open one pool vault only after re-anchoring it through stage-global."""

    if type(stage_capability) is not StageManifestCapability:
        raise TypeError("pool outcome access requires an exact StageManifestCapability")
    canonical_spec = _require_frozen_rotation_boundary(
        spec,
        label="pool outcome access rotation",
    )
    decoded_stage = _decode_stage_manifest_seal(
        stage_capability.seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    matches = tuple(
        entry
        for entry in decoded_stage.leaves
        if entry.spec == canonical_spec and entry.role == POOL_OUTCOME_ROLE
    )
    if len(matches) != 1:
        raise ValueError("stage-global index lacks one exact pool-outcome leaf")
    expected_entry = matches[0]

    if type(capsule) is not AuthenticatedLeafCapsule:
        raise TypeError("pool outcome access requires one exact AuthenticatedLeafCapsule")
    supplied_entry = _require_exact_stage_leaf_index(
        capsule.entry,
        label="pool outcome capsule index row",
    )
    if canonical_json_bytes(supplied_entry.document()) != canonical_json_bytes(
        expected_entry.document()
    ):
        raise ValueError("pool outcome capsule is not the leaf indexed by stage-global")
    if type(capsule.source_anchors_sha256) is not str or (
        capsule.source_anchors_sha256 != decoded_stage.source_anchors_sha256
    ):
        raise ValueError("pool outcome capsule has the wrong source-anchor identity")
    if type(capsule.source_predecessors) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in capsule.source_predecessors
    ):
        raise TypeError("pool outcome capsule source predecessors must use exact immutable types")
    expected_source_predecessors = tuple(sorted(decoded_stage.source_predecessors))
    if capsule.source_predecessors != expected_source_predecessors:
        raise ValueError("pool outcome capsule has the wrong source predecessors")
    if type(capsule.seal) is not PhaseSeal:
        raise TypeError("pool outcome capsule must contain an exact PhaseSeal")
    verified_leaf = verify_phase_capability(
        capsule.seal,
        expected_artifact=expected_entry.leaf_artifact,
        expected_payload_paths=expected_entry.payload_paths,
        expected_predecessor_seals=dict(decoded_stage.source_predecessors),
        expected_seal_sha256=expected_entry.leaf_seal_sha256,
    )
    reconstructed = AuthenticatedLeafCapsule(
        entry=expected_entry,
        seal=verified_leaf,
        source_anchors_sha256=decoded_stage.source_anchors_sha256,
        source_predecessors=expected_source_predecessors,
    )
    decoded = pool_outcome_vault_from_capsule(reconstructed)
    _require_exact_pool_outcome_vault(decoded, spec=canonical_spec)
    return PoolOutcomeVault(
        spec=canonical_spec,
        contexts=tuple(decoded.contexts),
        allowed_example_ids=tuple(decoded.allowed_example_ids),
    )


def outer_metadata_capability_from_capsule(
    capsule: AuthenticatedLeafCapsule,
) -> OuterMetadataCapability:
    """Materialize outer metadata from rootless authenticated bytes only."""

    expected_leaf = capsule.entry
    if expected_leaf.role != OUTER_METADATA_ROLE:
        raise ValueError("outer metadata loader requires an outer-metadata leaf")
    loaded = _decode_capsule(capsule, role=OUTER_METADATA_ROLE)
    ids = tuple(row.example_id for row in loaded.metadata_contexts)
    return OuterMetadataCapability(
        spec=expected_leaf.spec,
        contexts=loaded.metadata_contexts,
        allowed_example_ids=ids,
        support_sequence_ids=loaded.support_sequence_ids,
    )


def _require_exact_outer_metadata_capability(
    capability: object,
    *,
    spec: RotationSpec,
) -> OuterMetadataCapability:
    if type(capability) is not OuterMetadataCapability:
        raise TypeError(
            "outer metadata materialization must return an exact OuterMetadataCapability"
        )
    canonical_spec = _require_frozen_rotation_boundary(
        capability.spec,
        label="outer metadata capability rotation",
    )
    if canonical_spec != spec:
        raise ValueError("outer metadata capability has the wrong rotation")
    if type(capability.contexts) is not tuple or any(
        type(row) is not LabelFreeContext for row in capability.contexts
    ):
        raise TypeError("outer metadata contexts must be an exact LabelFreeContext tuple")
    for row in capability.contexts:
        if (
            any(
                type(getattr(row, field)) is not str
                for field in (
                    "example_id",
                    "assay_context_id",
                    "sequence_id",
                    "sequence",
                    "target",
                    "gram",
                )
            )
            or type(row.fold) is not int
        ):
            raise TypeError("outer metadata rows must use exact scalar field types")
        if (
            _SHA256.fullmatch(row.example_id) is None
            or row.assay_context_id != row.example_id
            or _SHA256.fullmatch(row.sequence_id) is None
            or canonicalize_sequence(row.sequence) != row.sequence
            or canonical_sequence_id(row.sequence) != row.sequence_id
            or row.target not in TARGET_GRAM
            or row.gram != TARGET_GRAM[row.target]
        ):
            raise ValueError("outer metadata row has an invalid canonical identity")
    if type(capability.allowed_example_ids) is not tuple or any(
        type(value) is not str for value in capability.allowed_example_ids
    ):
        raise TypeError("outer metadata allowed IDs must be an exact string tuple")
    if type(capability.support_sequence_ids) is not tuple or any(
        type(value) is not str for value in capability.support_sequence_ids
    ):
        raise TypeError("outer metadata support IDs must be an exact string tuple")
    ids = tuple(row.example_id for row in capability.contexts)
    support = capability.support_sequence_ids
    sequence_by_id: dict[str, str] = {}
    for row in capability.contexts:
        previous = sequence_by_id.setdefault(row.sequence_id, row.sequence)
        if previous != row.sequence:
            raise ValueError("outer metadata maps one sequence ID to multiple sequences")
    if (
        not ids
        or ids != tuple(sorted(set(ids)))
        or capability.allowed_example_ids != ids
        or any(row.fold != spec.outer_fold for row in capability.contexts)
        or support != tuple(sorted(set(support)))
        or any(_SHA256.fullmatch(value) is None for value in support)
        or not set(support).issubset(sequence_by_id)
    ):
        raise ValueError("outer metadata differs from its exact outer fold")
    if len(support) != EXPECTED_SUPPORT_BY_FOLD[spec.outer_fold]:
        raise ValueError("outer metadata support census differs from the frozen protocol")
    return capability


def outer_metadata_capability_from_stage_capabilities(
    stage_capability: StageManifestCapability,
    capsule: AuthenticatedLeafCapsule,
    *,
    spec: RotationSpec,
    expected_stage_global_seal_sha256: str,
) -> OuterMetadataCapability:
    """Open one label-free outer view only after re-anchoring it through stage-global."""

    if type(stage_capability) is not StageManifestCapability:
        raise TypeError("outer metadata access requires an exact StageManifestCapability")
    canonical_spec = _require_frozen_rotation_boundary(
        spec,
        label="outer metadata access rotation",
    )
    decoded_stage = _decode_stage_manifest_seal(
        stage_capability.seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    matches = tuple(
        entry
        for entry in decoded_stage.leaves
        if entry.spec == canonical_spec and entry.role == OUTER_METADATA_ROLE
    )
    if len(matches) != 1:
        raise ValueError("stage-global index lacks one exact outer-metadata leaf")
    expected_entry = matches[0]

    if type(capsule) is not AuthenticatedLeafCapsule:
        raise TypeError("outer metadata access requires one exact AuthenticatedLeafCapsule")
    supplied_entry = _require_exact_stage_leaf_index(
        capsule.entry,
        label="outer metadata capsule index row",
    )
    if canonical_json_bytes(supplied_entry.document()) != canonical_json_bytes(
        expected_entry.document()
    ):
        raise ValueError("outer metadata capsule is not the leaf indexed by stage-global")
    if type(capsule.source_anchors_sha256) is not str or (
        capsule.source_anchors_sha256 != decoded_stage.source_anchors_sha256
    ):
        raise ValueError("outer metadata capsule has the wrong source-anchor identity")
    if type(capsule.source_predecessors) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in capsule.source_predecessors
    ):
        raise TypeError("outer metadata capsule source predecessors must use exact immutable types")
    expected_source_predecessors = tuple(sorted(decoded_stage.source_predecessors))
    if capsule.source_predecessors != expected_source_predecessors:
        raise ValueError("outer metadata capsule has the wrong source predecessors")
    if type(capsule.seal) is not PhaseSeal:
        raise TypeError("outer metadata capsule must contain an exact PhaseSeal")
    verified_leaf = verify_phase_capability(
        capsule.seal,
        expected_artifact=expected_entry.leaf_artifact,
        expected_payload_paths=expected_entry.payload_paths,
        expected_predecessor_seals=dict(decoded_stage.source_predecessors),
        expected_seal_sha256=expected_entry.leaf_seal_sha256,
    )
    reconstructed = AuthenticatedLeafCapsule(
        entry=expected_entry,
        seal=verified_leaf,
        source_anchors_sha256=decoded_stage.source_anchors_sha256,
        source_predecessors=expected_source_predecessors,
    )
    decoded = outer_metadata_capability_from_capsule(reconstructed)
    _require_exact_outer_metadata_capability(decoded, spec=canonical_spec)
    return OuterMetadataCapability(
        spec=canonical_spec,
        contexts=tuple(decoded.contexts),
        allowed_example_ids=tuple(decoded.allowed_example_ids),
        support_sequence_ids=tuple(decoded.support_sequence_ids),
    )


def outer_outcome_vault_from_capsule(
    capsule: AuthenticatedLeafCapsule,
) -> OuterOutcomeVault:
    """Materialize an outer vault from rootless authenticated bytes only."""

    expected_leaf = capsule.entry
    if expected_leaf.role != OUTER_OUTCOME_ROLE:
        raise ValueError("outer outcome loader requires an outer-outcome-vault leaf")
    loaded = _decode_capsule(capsule, role=OUTER_OUTCOME_ROLE)
    ids = tuple(row.example_id for row in loaded.outcome_contexts)
    return OuterOutcomeVault(
        spec=expected_leaf.spec,
        contexts=loaded.outcome_contexts,
        allowed_example_ids=ids,
    )


def _require_exact_outer_outcome_vault(
    vault: object,
    *,
    spec: RotationSpec,
    expected_entry: StageLeafIndex,
) -> OuterOutcomeVault:
    """Recheck finalization-only outcomes against their stage-global index row."""

    if type(vault) is not OuterOutcomeVault:
        raise TypeError("outer outcome materialization must return an exact OuterOutcomeVault")
    canonical_spec = _require_frozen_rotation_boundary(
        vault.spec,
        label="outer outcome vault rotation",
    )
    if canonical_spec != spec:
        raise ValueError("outer outcome vault has the wrong rotation")
    indexed = _require_exact_stage_leaf_index(
        expected_entry,
        label="outer outcome stage-global index row",
    )
    if indexed.role != OUTER_OUTCOME_ROLE or indexed.spec != spec:
        raise ValueError("outer outcome stage-global index row has the wrong role or rotation")
    if type(vault.contexts) is not tuple or any(
        type(row) is not ContextRow for row in vault.contexts
    ):
        raise TypeError("outer outcome vault contexts must be an exact ContextRow tuple")
    for row in vault.contexts:
        if any(
            type(getattr(row, field)) is not str
            for field in (
                "example_id",
                "assay_context_id",
                "sequence_id",
                "sequence",
                "target",
                "gram",
            )
        ) or any(
            type(getattr(row, field)) is not int
            for field in ("fold", "label", "source_observations")
        ):
            raise TypeError("outer outcome vault rows must use exact scalar field types")
        if (
            _SHA256.fullmatch(row.example_id) is None
            or row.assay_context_id != row.example_id
            or _SHA256.fullmatch(row.sequence_id) is None
            or canonicalize_sequence(row.sequence) != row.sequence
            or canonical_sequence_id(row.sequence) != row.sequence_id
            or row.target not in TARGET_GRAM
            or row.gram != TARGET_GRAM[row.target]
            or row.label not in (0, 1)
            or row.source_observations < 1
        ):
            raise ValueError("outer outcome vault row has an invalid canonical identity")
    if type(vault.allowed_example_ids) is not tuple or any(
        type(value) is not str for value in vault.allowed_example_ids
    ):
        raise TypeError("outer outcome vault allowed IDs must be an exact string tuple")
    ids = tuple(row.example_id for row in vault.contexts)
    expected_count = indexed.row_count_map["contexts.jsonl"]
    expected_stream = indexed.id_stream_map["outer_outcome_example_ids"]
    if (
        not ids
        or ids != tuple(sorted(set(ids)))
        or vault.allowed_example_ids != ids
        or any(row.fold != spec.outer_fold for row in vault.contexts)
        or len(ids) != expected_count
        or expected_stream.count != expected_count
        or _id_stream_sha256(ids, label="outer outcome vault example IDs") != expected_stream.sha256
    ):
        raise ValueError("outer outcome vault differs from its indexed outer-fold census")
    support = _support_from_contexts(vault.contexts)
    if len(support) != EXPECTED_SUPPORT_BY_FOLD[spec.outer_fold]:
        raise ValueError("outer outcome vault support census differs from the frozen protocol")
    return vault


def outer_outcome_vault_from_stage_capabilities(
    stage_capability: StageManifestCapability,
    capsule: AuthenticatedLeafCapsule,
    *,
    spec: RotationSpec,
    expected_stage_global_seal_sha256: str,
) -> OuterOutcomeVault:
    """Open one outer vault only after re-anchoring it through stage-global."""

    if type(stage_capability) is not StageManifestCapability:
        raise TypeError("outer outcome access requires an exact StageManifestCapability")
    canonical_spec = _require_frozen_rotation_boundary(
        spec,
        label="outer outcome access rotation",
    )
    decoded_stage = _decode_stage_manifest_seal(
        stage_capability.seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    matches = tuple(
        entry
        for entry in decoded_stage.leaves
        if entry.spec == canonical_spec and entry.role == OUTER_OUTCOME_ROLE
    )
    if len(matches) != 1:
        raise ValueError("stage-global index lacks one exact outer-outcome leaf")
    expected_entry = matches[0]

    if type(capsule) is not AuthenticatedLeafCapsule:
        raise TypeError("outer outcome access requires one exact AuthenticatedLeafCapsule")
    supplied_entry = _require_exact_stage_leaf_index(
        capsule.entry,
        label="outer outcome capsule index row",
    )
    if canonical_json_bytes(supplied_entry.document()) != canonical_json_bytes(
        expected_entry.document()
    ):
        raise ValueError("outer outcome capsule is not the leaf indexed by stage-global")
    if type(capsule.source_anchors_sha256) is not str or (
        capsule.source_anchors_sha256 != decoded_stage.source_anchors_sha256
    ):
        raise ValueError("outer outcome capsule has the wrong source-anchor identity")
    if type(capsule.source_predecessors) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in capsule.source_predecessors
    ):
        raise TypeError("outer outcome capsule source predecessors must use exact immutable types")
    expected_source_predecessors = tuple(sorted(decoded_stage.source_predecessors))
    if capsule.source_predecessors != expected_source_predecessors:
        raise ValueError("outer outcome capsule has the wrong source predecessors")
    if type(capsule.seal) is not PhaseSeal:
        raise TypeError("outer outcome capsule must contain an exact PhaseSeal")
    verified_leaf = verify_phase_capability(
        capsule.seal,
        expected_artifact=expected_entry.leaf_artifact,
        expected_payload_paths=expected_entry.payload_paths,
        expected_predecessor_seals=dict(decoded_stage.source_predecessors),
        expected_seal_sha256=expected_entry.leaf_seal_sha256,
    )
    reconstructed = AuthenticatedLeafCapsule(
        entry=expected_entry,
        seal=verified_leaf,
        source_anchors_sha256=decoded_stage.source_anchors_sha256,
        source_predecessors=expected_source_predecessors,
    )
    decoded = outer_outcome_vault_from_capsule(reconstructed)
    _require_exact_outer_outcome_vault(
        decoded,
        spec=canonical_spec,
        expected_entry=expected_entry,
    )
    return OuterOutcomeVault(
        spec=canonical_spec,
        contexts=tuple(decoded.contexts),
        allowed_example_ids=tuple(decoded.allowed_example_ids),
    )


def load_prepare_capability(
    capsule: AuthenticatedLeafCapsule,
) -> PrepareCapability:
    """Worker-safe loader accepting no filesystem path or stage handle."""

    return prepare_capability_from_capsule(capsule)


def load_pool_outcome_vault(
    capsule: AuthenticatedLeafCapsule,
) -> PoolOutcomeVault:
    """Custodian-safe loader accepting no filesystem path or stage handle."""

    return pool_outcome_vault_from_capsule(capsule)


def load_outer_metadata_capability(
    capsule: AuthenticatedLeafCapsule,
) -> OuterMetadataCapability:
    """Worker-safe loader accepting no filesystem path or stage handle."""

    return outer_metadata_capability_from_capsule(capsule)


def load_outer_outcome_vault(
    capsule: AuthenticatedLeafCapsule,
) -> OuterOutcomeVault:
    """Custodian-safe loader accepting no filesystem path or stage handle."""

    return outer_outcome_vault_from_capsule(capsule)


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        with suppress(FileNotFoundError):
            if stat.S_ISLNK(os.lstat(candidate).st_mode):
                raise ValueError(f"{label} must not traverse a symbolic link: {candidate}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _try_rename_directory_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> bool:
    """Rename sibling directories without resolving their parent pathname."""

    try:
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = library.renameat2
    except (AttributeError, OSError):
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to replace trusted stage: {destination_name}")
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        return False
    raise OSError(
        error_number,
        f"atomic trusted-stage publication failed: {os.strerror(error_number)}",
        destination_name,
    )


def _directory_entry_still_names_open_directory(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
) -> bool:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(opened.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and (opened.st_dev, opened.st_ino) == identity
        and (named.st_dev, named.st_ino) == identity
    )


def _path_still_names_open_directory(
    path: Path,
    descriptor: int,
    identity: tuple[int, int],
) -> bool:
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISDIR(opened.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and (opened.st_dev, opened.st_ino) == identity
        and (named.st_dev, named.st_ino) == identity
    )


def _remove_private_tree(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
) -> None:
    """Remove only the private inode retained by the publisher."""

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != identity:
            return

        def remove_contents(directory_descriptor: int) -> None:
            os.fchmod(directory_descriptor, 0o700)
            for name in tuple(os.listdir(directory_descriptor)):
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                    child_descriptor = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        child_opened = os.fstat(child_descriptor)
                        if (child_opened.st_dev, child_opened.st_ino) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        ):
                            raise RuntimeError("private cleanup child identity changed")
                        remove_contents(child_descriptor)
                    finally:
                        os.close(child_descriptor)
                    current = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (current.st_dev, current.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise RuntimeError("private cleanup child name was replaced")
                    os.rmdir(name, dir_fd=directory_descriptor)
                else:
                    os.unlink(name, dir_fd=directory_descriptor)

        remove_contents(descriptor)
        if _directory_entry_still_names_open_directory(
            parent_descriptor,
            name,
            descriptor,
            identity,
        ):
            os.rmdir(name, dir_fd=parent_descriptor)
    except (OSError, RuntimeError):
        # A failed private cleanup is quarantined through the retained inode;
        # never follow or mutate a replacement pathname.
        with suppress(OSError):
            os.fchmod(descriptor, 0o000)


def _claim_still_names_open_directory(
    parent_descriptor: int,
    destination_name: str,
    descriptor: int,
    identity: tuple[int, int],
) -> bool:
    return _directory_entry_still_names_open_directory(
        parent_descriptor,
        destination_name,
        descriptor,
        identity,
    )


def _marker_still_named_by_global_directory(
    global_descriptor: int,
    marker_descriptor: int,
    marker_identity: tuple[int, int],
) -> bool:
    try:
        opened = os.fstat(marker_descriptor)
        named = os.stat(
            MANIFEST_NAME,
            dir_fd=global_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(named.st_mode)
        and (opened.st_dev, opened.st_ino) == marker_identity
        and (named.st_dev, named.st_ino) == marker_identity
        and opened.st_nlink == 1
        and named.st_nlink == 1
    )


def _snapshot_descriptor_tree(
    descriptor: int,
    *,
    label: str,
    retained_files: Mapping[str, int] | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Hash an exact tree without resolving any pathname above its root fd."""

    retained = {} if retained_files is None else dict(retained_files)
    records: list[tuple[object, ...]] = []

    def visit(directory_descriptor: int, prefix: str) -> None:
        directory_before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise RuntimeError(f"{label} contains a non-directory container")
        names = tuple(sorted(os.listdir(directory_descriptor)))
        records.append(
            (
                prefix,
                "directory",
                stat.S_IMODE(directory_before.st_mode),
                directory_before.st_dev,
                directory_before.st_ino,
                directory_before.st_nlink,
                names,
            )
        )
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            named_before = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(named_before.st_mode):
                raise RuntimeError(f"{label} contains a symbolic link: {relative}")
            identity = (named_before.st_dev, named_before.st_ino)
            if stat.S_ISDIR(named_before.st_mode):
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_descriptor,
                )
                try:
                    child_opened = os.fstat(child_descriptor)
                    if (child_opened.st_dev, child_opened.st_ino) != identity:
                        raise RuntimeError(f"{label} directory identity changed: {relative}")
                    visit(child_descriptor, relative)
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(named_before.st_mode):
                retained_descriptor = retained.get(relative)
                close_file = retained_descriptor is None
                file_descriptor = (
                    os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_descriptor,
                    )
                    if retained_descriptor is None
                    else retained_descriptor
                )
                try:
                    opened_before = os.fstat(file_descriptor)
                    if (
                        not stat.S_ISREG(opened_before.st_mode)
                        or (opened_before.st_dev, opened_before.st_ino) != identity
                        or opened_before.st_nlink != 1
                    ):
                        raise RuntimeError(f"{label} file identity changed: {relative}")
                    digest = hashlib.sha256()
                    os.lseek(file_descriptor, 0, os.SEEK_SET)
                    while chunk := os.read(file_descriptor, 1024 * 1024):
                        digest.update(chunk)
                    opened_after = os.fstat(file_descriptor)
                    named_after = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        (opened_after.st_dev, opened_after.st_ino) != identity
                        or (named_after.st_dev, named_after.st_ino) != identity
                        or opened_after.st_size != opened_before.st_size
                        or named_after.st_size != opened_before.st_size
                        or stat.S_IMODE(opened_after.st_mode) != stat.S_IMODE(opened_before.st_mode)
                        or stat.S_IMODE(named_after.st_mode) != stat.S_IMODE(opened_before.st_mode)
                        or opened_after.st_nlink != 1
                        or named_after.st_nlink != 1
                    ):
                        raise RuntimeError(f"{label} file changed while read: {relative}")
                    records.append(
                        (
                            relative,
                            "file",
                            stat.S_IMODE(opened_after.st_mode),
                            opened_after.st_dev,
                            opened_after.st_ino,
                            opened_after.st_nlink,
                            opened_after.st_size,
                            digest.hexdigest(),
                        )
                    )
                finally:
                    if close_file:
                        os.close(file_descriptor)
            else:
                raise RuntimeError(f"{label} contains a special file: {relative}")
            named_after = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (named_after.st_dev, named_after.st_ino) != identity:
                raise RuntimeError(f"{label} entry name changed: {relative}")
        directory_after = os.fstat(directory_descriptor)
        if (
            (directory_after.st_dev, directory_after.st_ino)
            != (directory_before.st_dev, directory_before.st_ino)
            or stat.S_IMODE(directory_after.st_mode) != stat.S_IMODE(directory_before.st_mode)
            or directory_after.st_nlink != directory_before.st_nlink
        ):
            raise RuntimeError(f"{label} directory changed while scanned: {prefix or '.'}")

    visit(descriptor, "")
    return tuple(records)


def _child_still_names_open_directory(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
    *,
    mode: int,
) -> bool:
    return (
        _directory_entry_still_names_open_directory(
            parent_descriptor,
            name,
            descriptor,
            identity,
        )
        and stat.S_IMODE(os.fstat(descriptor).st_mode) == mode
    )


def _quarantine_failed_aggregate_claim(
    *,
    claim_descriptor: int,
    claim_identity: tuple[int, int],
    global_descriptor: int | None,
    marker_descriptor: int | None,
    marker_identity: tuple[int, int] | None,
) -> None:
    """Make only retained failed-claim inodes inaccessible and invalid."""

    opened_claim = os.fstat(claim_descriptor)
    if (
        not stat.S_ISDIR(opened_claim.st_mode)
        or (opened_claim.st_dev, opened_claim.st_ino) != claim_identity
    ):
        return
    if (
        global_descriptor is not None
        and marker_descriptor is not None
        and marker_identity is not None
        and _marker_still_named_by_global_directory(
            global_descriptor,
            marker_descriptor,
            marker_identity,
        )
    ):
        with suppress(OSError):
            os.fchmod(marker_descriptor, 0o000)
    with suppress(OSError):
        os.fchmod(claim_descriptor, 0o000)


def _assert_fallback_inventory(
    *,
    rotations_descriptor: int,
    global_descriptor: int,
    marker_descriptor: int,
    expected_global_seal_sha256: str,
    expected_rotations_snapshot: tuple[tuple[object, ...], ...],
    expected_global_snapshot: tuple[tuple[object, ...], ...],
) -> None:
    rotations_snapshot = _snapshot_descriptor_tree(
        rotations_descriptor,
        label="moved rotations tree",
    )
    global_snapshot = _snapshot_descriptor_tree(
        global_descriptor,
        label="moved global tree",
        retained_files={MANIFEST_NAME: marker_descriptor},
    )
    if rotations_snapshot != expected_rotations_snapshot:
        raise RuntimeError("moved rotations tree differs from the verified source")
    if global_snapshot != expected_global_snapshot:
        raise RuntimeError("moved global tree differs from the verified source")
    os.lseek(marker_descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(marker_descriptor, 1024 * 1024):
        chunks.append(chunk)
    payload = b"".join(chunks)
    if sha256_bytes(payload) != expected_global_seal_sha256:
        raise RuntimeError("uncommitted trusted-stage marker bytes changed")


def _publish_tree_noreplace(
    staging: Path,
    destination: Path,
    *,
    global_seal_sha256: str,
    parent_descriptor: int,
    parent_identity: tuple[int, int],
    staging_descriptor: int,
    staging_identity: tuple[int, int],
) -> None:
    if not _path_still_names_open_directory(
        destination.parent,
        parent_descriptor,
        parent_identity,
    ) or not _directory_entry_still_names_open_directory(
        parent_descriptor,
        staging.name,
        staging_descriptor,
        staging_identity,
    ):
        raise RuntimeError("trusted-stage parent or private staging identity changed")
    if _try_rename_directory_noreplace(
        parent_descriptor,
        staging.name,
        destination.name,
    ):
        if not _path_still_names_open_directory(
            destination.parent,
            parent_descriptor,
            parent_identity,
        ) or not _directory_entry_still_names_open_directory(
            parent_descriptor,
            destination.name,
            staging_descriptor,
            staging_identity,
        ):
            raise RuntimeError("atomically published trusted-stage name was replaced")
        # The atomic rename is the commit point.  A later durability hint must
        # not turn a successful publication into a reported failure while the
        # committed destination remains visible.
        with suppress(OSError):
            os.fsync(parent_descriptor)
        return

    claim_descriptor: int | None = None
    claim_identity: tuple[int, int] | None = None
    marker_descriptor: int | None = None
    marker_identity: tuple[int, int] | None = None
    rotations_descriptor: int | None = None
    global_descriptor: int | None = None
    rotations_identity: tuple[int, int] | None = None
    global_identity: tuple[int, int] | None = None
    committed = False
    try:
        os.mkdir(destination.name, 0o700, dir_fd=parent_descriptor)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace trusted stage: {destination}") from error
    try:
        claim = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        claim_identity = (claim.st_dev, claim.st_ino)
        claim_descriptor = os.open(
            destination.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        os.fchmod(claim_descriptor, 0o700)
        if not _claim_still_names_open_directory(
            parent_descriptor,
            destination.name,
            claim_descriptor,
            claim_identity,
        ):
            raise RuntimeError("trusted-stage claim changed immediately after creation")

        os.fchmod(staging_descriptor, 0o700)
        rotations_descriptor = os.open(
            ROTATIONS_DIRECTORY,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=staging_descriptor,
        )
        global_descriptor = os.open(
            GLOBAL_DIRECTORY,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=staging_descriptor,
        )
        rotations_before = os.fstat(rotations_descriptor)
        global_before = os.fstat(global_descriptor)
        rotations_identity = (rotations_before.st_dev, rotations_before.st_ino)
        global_identity = (global_before.st_dev, global_before.st_ino)
        if not _child_still_names_open_directory(
            staging_descriptor,
            ROTATIONS_DIRECTORY,
            rotations_descriptor,
            rotations_identity,
            mode=0o555,
        ) or not _child_still_names_open_directory(
            staging_descriptor,
            GLOBAL_DIRECTORY,
            global_descriptor,
            global_identity,
            mode=0o555,
        ):
            raise RuntimeError("verified stage containers changed before publication")
        marker_descriptor = os.open(
            MANIFEST_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=global_descriptor,
        )
        marker_before = os.fstat(marker_descriptor)
        marker_identity = (marker_before.st_dev, marker_before.st_ino)
        if (
            not stat.S_ISREG(marker_before.st_mode)
            or stat.S_IMODE(marker_before.st_mode) != 0o444
            or marker_before.st_nlink != 1
        ):
            raise RuntimeError("verified global stage marker changed before publication")
        os.lseek(marker_descriptor, 0, os.SEEK_SET)
        marker_chunks: list[bytes] = []
        while chunk := os.read(marker_descriptor, 1024 * 1024):
            marker_chunks.append(chunk)
        if sha256_bytes(b"".join(marker_chunks)) != global_seal_sha256:
            raise RuntimeError("verified global stage marker digest changed before publication")
        os.fchmod(marker_descriptor, 0o000)
        expected_rotations_snapshot = _snapshot_descriptor_tree(
            rotations_descriptor,
            label="verified rotations tree",
        )
        expected_global_snapshot = _snapshot_descriptor_tree(
            global_descriptor,
            label="verified global tree",
            retained_files={MANIFEST_NAME: marker_descriptor},
        )
        # Some NFS/Lustre servers require write permission on the moved
        # directory itself, in addition to both parents.  Only the two already
        # verified aggregate containers are thawed; every sealed leaf remains
        # read-only throughout the move.
        os.fchmod(rotations_descriptor, 0o700)
        os.fchmod(global_descriptor, 0o700)
        os.rename(
            ROTATIONS_DIRECTORY,
            ROTATIONS_DIRECTORY,
            src_dir_fd=staging_descriptor,
            dst_dir_fd=claim_descriptor,
        )
        os.rename(
            GLOBAL_DIRECTORY,
            GLOBAL_DIRECTORY,
            src_dir_fd=staging_descriptor,
            dst_dir_fd=claim_descriptor,
        )
        if (
            not _child_still_names_open_directory(
                claim_descriptor,
                ROTATIONS_DIRECTORY,
                rotations_descriptor,
                rotations_identity,
                mode=0o700,
            )
            or not _child_still_names_open_directory(
                claim_descriptor,
                GLOBAL_DIRECTORY,
                global_descriptor,
                global_identity,
                mode=0o700,
            )
            or not _marker_still_named_by_global_directory(
                global_descriptor,
                marker_descriptor,
                marker_identity,
            )
        ):
            raise RuntimeError("trusted-stage container identity changed during publication")
        os.fchmod(rotations_descriptor, 0o555)
        os.fchmod(global_descriptor, 0o555)
        _assert_fallback_inventory(
            rotations_descriptor=rotations_descriptor,
            global_descriptor=global_descriptor,
            marker_descriptor=marker_descriptor,
            expected_global_seal_sha256=global_seal_sha256,
            expected_rotations_snapshot=expected_rotations_snapshot,
            expected_global_snapshot=expected_global_snapshot,
        )
        if os.listdir(staging_descriptor):
            raise RuntimeError("private stage retained aliases after aggregate moves")
        if not _directory_entry_still_names_open_directory(
            parent_descriptor,
            staging.name,
            staging_descriptor,
            staging_identity,
        ):
            raise RuntimeError("private stage name changed before alias removal")
        os.rmdir(staging.name, dir_fd=parent_descriptor)
        os.fsync(rotations_descriptor)
        os.fsync(global_descriptor)
        os.fchmod(claim_descriptor, 0o555)
        os.fsync(claim_descriptor)
        os.fsync(parent_descriptor)
        _assert_fallback_inventory(
            rotations_descriptor=rotations_descriptor,
            global_descriptor=global_descriptor,
            marker_descriptor=marker_descriptor,
            expected_global_seal_sha256=global_seal_sha256,
            expected_rotations_snapshot=expected_rotations_snapshot,
            expected_global_snapshot=expected_global_snapshot,
        )
        if (
            not _path_still_names_open_directory(
                destination.parent,
                parent_descriptor,
                parent_identity,
            )
            or not _claim_still_names_open_directory(
                parent_descriptor,
                destination.name,
                claim_descriptor,
                claim_identity,
            )
            or stat.S_IMODE(os.fstat(claim_descriptor).st_mode) != 0o555
            or tuple(sorted(os.listdir(claim_descriptor)))
            != (GLOBAL_DIRECTORY, ROTATIONS_DIRECTORY)
            or not _child_still_names_open_directory(
                claim_descriptor,
                ROTATIONS_DIRECTORY,
                rotations_descriptor,
                rotations_identity,
                mode=0o555,
            )
            or not _child_still_names_open_directory(
                claim_descriptor,
                GLOBAL_DIRECTORY,
                global_descriptor,
                global_identity,
                mode=0o555,
            )
            or not _marker_still_named_by_global_directory(
                global_descriptor,
                marker_descriptor,
                marker_identity,
            )
            or stat.S_IMODE(os.fstat(marker_descriptor).st_mode) != 0o000
        ):
            raise RuntimeError("trusted-stage claim or marker changed before commit")

        # No fallible operation follows this one-way commit transition.
        os.fchmod(marker_descriptor, 0o444)
        committed = True
    finally:
        if not committed and claim_descriptor is not None and claim_identity is not None:
            _quarantine_failed_aggregate_claim(
                claim_descriptor=claim_descriptor,
                claim_identity=claim_identity,
                global_descriptor=global_descriptor,
                marker_descriptor=marker_descriptor,
                marker_identity=marker_identity,
            )
        for descriptor in (
            rotations_descriptor,
            global_descriptor,
            marker_descriptor,
            claim_descriptor,
        ):
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)


def _publish_leaf(
    root: Path,
    material: _LeafMaterial,
    *,
    source_predecessors: Mapping[str, str],
    source_anchors_sha256: str,
) -> StageLeafIndex:
    provisional = StageLeafIndex(
        spec=material.spec,
        role=material.role,
        relative_path=leaf_relative_path(material.spec, material.role),
        leaf_artifact=LEAF_ARTIFACTS[material.role],
        leaf_seal_sha256="0" * 64,
        payload_paths=LEAF_PAYLOAD_PATHS[material.role],
        row_counts=material.row_counts,
        id_streams=material.id_streams,
    )
    seal = publish_phase(
        root / provisional.relative_path,
        artifact=provisional.leaf_artifact,
        payloads=material.payloads,
        predecessor_seals=source_predecessors,
        metadata=_leaf_metadata(provisional, source_anchors_sha256),
    )
    return _entry_from_material(material, seal)


def _create_private_stage(
    parent: Path,
    *,
    parent_descriptor: int,
    destination_name: str,
) -> tuple[Path, int, tuple[int, int]]:
    prefix = f".{destination_name}.stage."
    for _attempt in range(128):
        name = f"{prefix}{secrets.token_hex(12)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
            os.fchmod(descriptor, 0o700)
            opened = os.fstat(descriptor)
            identity = (opened.st_dev, opened.st_ino)
            if not _directory_entry_still_names_open_directory(
                parent_descriptor,
                name,
                descriptor,
                identity,
            ):
                raise RuntimeError("private stage identity changed during creation")
            return parent / name, descriptor, identity
        except BaseException:
            if descriptor is not None:
                with suppress(OSError):
                    os.fchmod(descriptor, 0o000)
                with suppress(OSError):
                    os.close(descriptor)
            # Never remove this pathname after an identity failure: another
            # same-UID actor may have rebound it.  The retained owned inode is
            # quarantined above and contains no stage data at this point.
            raise
    raise FileExistsError("could not claim a fresh private trusted-stage sibling")


def _require_exact_accepted_source(source: AuthenticatedGate1Source) -> None:
    """Fail closed unless an authenticated source matches the frozen contract."""

    if not isinstance(source, AuthenticatedGate1Source):
        raise TypeError("stage publication requires an AuthenticatedGate1Source")
    contract = ACCEPTED_GATE1_SOURCE_CONTRACT
    evidence = source.evidence
    expected_hashes = {
        "CODE_SHA256SUMS": contract.code_manifest_sha256,
        "FROZEN_INPUT_SHA256SUMS": contract.frozen_input_manifest_sha256,
        "SHA256SUMS": contract.publication_top_sha256,
        "gate1/SHA256SUMS": contract.semantic_top_sha256,
        **{f"gate1/{name}": digest for name, digest in contract.artifact_sha256},
    }
    expected_paths = tuple(sorted(expected_hashes))
    if (
        evidence.producer_job_id != contract.producer_job_id
        or evidence.audit_job_id != contract.audit_job_id
        or evidence.twins_byte_identical is not True
        or evidence.independent_receipt_sha256 != contract.independent_receipt_sha256
        or tuple(item.logical_path for item in evidence.files) != expected_paths
    ):
        raise ValueError("trusted-stage source differs from the exact accepted Gate-1 evidence")
    for item in evidence.files:
        required_mode = "0400" if item.logical_path == "SHA256SUMS" else "0444"
        if (
            item.sha256 != expected_hashes[item.logical_path]
            or item.mode != required_mode
            or isinstance(item.size_bytes, bool)
            or not isinstance(item.size_bytes, int)
            or item.size_bytes <= 0
        ):
            raise ValueError(
                "trusted-stage source file evidence differs from the accepted contract"
            )

    panel = source.panel
    contexts = panel.contexts
    sequences_by_fold = tuple(
        len({row.sequence_id for row in contexts if row.fold == fold}) for fold in range(5)
    )
    label_counts = Counter(row.label for row in contexts)
    support = set(panel.support_sequence_ids)
    support_contexts = tuple(row for row in contexts if row.sequence_id in support)
    support_target_counts = Counter(
        len({row.target for row in support_contexts if row.sequence_id == sequence_id})
        for sequence_id in support
    )
    if (
        len(contexts) != contract.expected_contexts
        or len({row.sequence_id for row in contexts}) != contract.expected_sequences
        or tuple(len(rows) for rows in panel.contexts_by_fold) != contract.expected_examples_by_fold
        or sequences_by_fold != contract.expected_sequences_by_fold
        or label_counts != Counter({0: contract.expected_negatives, 1: contract.expected_positives})
        or sum(row.source_observations for row in contexts) != contract.expected_source_observations
        or len(panel.support_sequence_ids) != contract.expected_support_sequences
        or tuple(len(ids) for ids in panel.support_sequence_ids_by_fold)
        != contract.expected_support_by_fold
        or panel.support_sequence_ids_sha256 != contract.expected_support_ids_sha256
        or len(support_contexts) != contract.expected_support_contexts
        or sum(row.source_observations for row in support_contexts)
        != contract.expected_support_source_observations
        or tuple(support_target_counts[count] for count in range(2, 8))
        != contract.expected_support_unique_target_counts_2_to_7
    ):
        raise ValueError("trusted-stage panel differs from the exact accepted Gate-1 census")


def _publish_trusted_stage_unchecked(
    source: AuthenticatedGate1Source,
    destination: str | Path,
) -> TrustedStage:
    """Private generic builder used only by synthetic adversarial tests."""

    if not isinstance(source, AuthenticatedGate1Source):
        raise TypeError("stage publication requires an AuthenticatedGate1Source")
    contexts, support = _validate_panel(source.panel)
    anchors = source_anchors_document(source.evidence)
    anchor_bytes = canonical_json_bytes(anchors)
    anchors_sha256 = sha256_bytes(anchor_bytes)
    source_predecessors = dict(_validate_source_anchors(anchors))

    final = Path(os.path.abspath(os.fspath(destination)))
    if _PATH_COMPONENT.fullmatch(final.name) is None:
        raise ValueError("trusted-stage destination basename is unsafe")
    parent = final.parent
    _reject_symlink_chain(parent, label="trusted-stage destination parent")
    parent_metadata = os.lstat(parent)
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("trusted-stage destination parent must be a real existing directory")
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    parent_opened = os.fstat(parent_descriptor)
    parent_identity = (parent_opened.st_dev, parent_opened.st_ino)
    if not _path_still_names_open_directory(parent, parent_descriptor, parent_identity):
        os.close(parent_descriptor)
        raise RuntimeError("trusted-stage destination parent changed while it was opened")
    try:
        os.stat(final.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except BaseException:
        os.close(parent_descriptor)
        raise
    else:
        os.close(parent_descriptor)
        raise FileExistsError(f"refusing to reuse trusted-stage destination: {final}")
    try:
        staging, staging_descriptor, staging_identity = _create_private_stage(
            parent,
            parent_descriptor=parent_descriptor,
            destination_name=final.name,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    published = False
    try:
        # PhaseBuilder currently accepts a path rather than a root descriptor.
        # The parent is therefore required to be a trusted same-UID boundary;
        # retained descriptors and identity checks bracket all construction,
        # while every publication mutation below is descriptor-relative.
        if not _path_still_names_open_directory(
            parent,
            parent_descriptor,
            parent_identity,
        ) or not _path_still_names_open_directory(
            staging,
            staging_descriptor,
            staging_identity,
        ):
            raise RuntimeError("private trusted-stage path changed before construction")
        rotations_root = staging / ROTATIONS_DIRECTORY
        rotations_root.mkdir(mode=0o700)
        for spec in ordered_rotations():
            rotation_root = rotations_root / spec.rotation_id
            rotation_root.mkdir(mode=0o700)

        entries: list[StageLeafIndex] = []
        for spec in ordered_rotations():
            for role in ROLE_ORDER:
                material = _leaf_material(source.panel, spec=spec, role=role)
                entries.append(
                    _publish_leaf(
                        staging,
                        material,
                        source_predecessors=source_predecessors,
                        source_anchors_sha256=anchors_sha256,
                    )
                )
        index_bytes = canonical_jsonl_bytes(entry.document() for entry in entries)
        global_payloads = {
            "capability-index.jsonl": index_bytes,
            "policy-runs.jsonl": canonical_jsonl_bytes(
                run.document() for run in ordered_policy_runs()
            ),
            "protocol-census.json": canonical_json_bytes(protocol_census()),
            "rotations.jsonl": canonical_jsonl_bytes(
                rotation.document() for rotation in ordered_rotations()
            ),
            "source-anchors.json": anchor_bytes,
            "stage-summary.json": canonical_json_bytes(
                _summary_document(
                    contexts,
                    support,
                    entries,
                    source_anchors_sha256=anchors_sha256,
                    capability_index_sha256=sha256_bytes(index_bytes),
                )
            ),
        }
        global_predecessors = {
            **source_predecessors,
            **{entry.relative_path: entry.leaf_seal_sha256 for entry in entries},
        }
        global_seal = publish_phase(
            staging / GLOBAL_DIRECTORY,
            artifact=STAGE_ARTIFACT,
            payloads=global_payloads,
            predecessor_seals=global_predecessors,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "rotation_count": EXPECTED_ROTATIONS,
                "leaf_count": len(entries),
                "source_anchors_sha256": anchors_sha256,
            },
        )

        for spec in ordered_rotations():
            os.chmod(rotations_root / spec.rotation_id, 0o555)
            _fsync_directory(rotations_root / spec.rotation_id)
        os.chmod(rotations_root, 0o555)
        _fsync_directory(rotations_root)
        os.chmod(staging, 0o555)
        _fsync_directory(staging)

        verified = verify_trusted_stage(
            staging,
            expected_source_anchors=anchors,
            expected_global_seal_sha256=global_seal.seal_sha256,
        )
        _publish_tree_noreplace(
            staging,
            final,
            global_seal_sha256=verified.global_seal_sha256,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            staging_descriptor=staging_descriptor,
            staging_identity=staging_identity,
        )
        published = True
        published_stage = verify_trusted_stage(
            final,
            expected_source_anchors=anchors,
            expected_global_seal_sha256=verified.global_seal_sha256,
        )
        if published_stage != replace(verified, root=published_stage.root):
            raise RuntimeError("trusted stage identity changed across publication")
        return published_stage
    finally:
        if not published:
            _remove_private_tree(
                parent_descriptor,
                staging.name,
                staging_descriptor,
                staging_identity,
            )
        with suppress(OSError):
            os.close(staging_descriptor)
        with suppress(OSError):
            os.close(parent_descriptor)


def publish_trusted_stage(
    source: AuthenticatedGate1Source,
    destination: str | Path,
) -> TrustedStage:
    """Publish only the exact accepted Gate-1 source as a trusted stage."""

    _require_exact_accepted_source(source)
    return _publish_trusted_stage_unchecked(source, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m amp_challenge.evaluation.sequential_v2_stage",
        description="Publish the authenticated sequential-v2 trusted stage and exit.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="authenticate Gate-1 and publish role stores")
    stage.add_argument(
        "--twin-root",
        action="append",
        required=True,
        help="accepted Gate-1 twin root; provide exactly twice in 0,1 order",
    )
    stage.add_argument(
        "--primary-twin-slot",
        required=True,
        type=int,
        choices=(0, 1),
        help="ordered accepted Gate-1 twin whose authenticated bytes are parsed",
    )
    stage.add_argument("--independent-receipt", required=True, type=Path)
    stage.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run only the trusted stage custodian; this command never launches workers."""

    arguments = _parser().parse_args(argv)
    if arguments.command != "stage":
        raise AssertionError("unreachable sequential-v2 stage command")
    if len(arguments.twin_root) != 2:
        raise ValueError("stage requires exactly two --twin-root arguments")
    source = authenticate_accepted_gate1_source(
        tuple(Path(path) for path in arguments.twin_root),
        arguments.independent_receipt,
        primary_twin_slot=arguments.primary_twin_slot,
    )
    result = publish_trusted_stage(source, arguments.destination)
    print(canonical_json_bytes(result.document()).decode("utf-8"), end="")
    return 0


__all__ = [
    "GLOBAL_DIRECTORY",
    "GLOBAL_PAYLOAD_PATHS",
    "LEAF_ARTIFACTS",
    "LEAF_CAPABILITY_ARTIFACT",
    "LEAF_DATA_PAYLOAD_PATHS",
    "LEAF_PAYLOAD_PATHS",
    "OUTER_METADATA_ROLE",
    "OUTER_OUTCOME_ROLE",
    "POOL_OUTCOME_ROLE",
    "PREPARE_ROLE",
    "ROLE_ORDER",
    "SOURCE_ANCHORS_ARTIFACT",
    "STAGE_ARTIFACT",
    "STAGE_SUMMARY_ARTIFACT",
    "AuthenticatedLeafCapsule",
    "IdStreamBinding",
    "StageLeafIndex",
    "StageManifestCapability",
    "TrustedStage",
    "authenticate_stage_leaf_for_controller",
    "authenticate_stage_manifest_for_controller",
    "leaf_relative_path",
    "load_outer_metadata_capability",
    "load_outer_outcome_vault",
    "load_pool_outcome_vault",
    "load_prepare_capability",
    "main",
    "outer_metadata_capability_from_capsule",
    "outer_metadata_capability_from_stage_capabilities",
    "outer_outcome_vault_from_capsule",
    "outer_outcome_vault_from_stage_capabilities",
    "pool_outcome_vault_from_capsule",
    "pool_outcome_vault_from_stage_capabilities",
    "prepare_capability_from_capsule",
    "publish_trusted_stage",
    "source_anchors_document",
    "verify_stage_manifest",
    "verify_stage_manifest_capability",
    "verify_trusted_stage",
]


if __name__ == "__main__":
    raise SystemExit(main())

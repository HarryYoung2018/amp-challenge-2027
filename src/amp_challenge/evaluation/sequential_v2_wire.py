"""Canonical pathless wire values for sequential-v2 fresh executables.

The sealed evaluation APIs deliberately use in-memory :class:`PhaseSeal`
objects after a controller has authenticated filesystem paths and captured the
exact bytes.  Fresh workers cannot receive those Python objects directly.  This
module provides a small, explicit JSON/base64 transport for the already
captured bytes; the routing envelope never adds a filesystem root or
source-leaf path.  Captured metadata and payload bytes remain intentionally
opaque here.  Their phase-specific validators, rather than this transport, are
responsible for any stronger content-level disclosure policy.

Wire bytes are not authority on their own.  A worker reconstructs and
self-verifies every phase capability, then each role-specific API reanchors it
to controller-authoritative digests carried over the supervised input channel.
No pickle or dynamic type dispatch is used.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass

from amp_challenge.evaluation.sequential_v2_commitments import (
    CEILING_SELECTOR_ARTIFACT,
    POOL_COMMITMENT_ARTIFACT,
    POOL_COMMITMENT_PAYLOAD_PATHS,
    PREDICTION_SELECTOR_ARTIFACT,
    RANDOM_SELECTOR_ARTIFACT,
    ROTATION_INDEX_ARTIFACT,
    ROTATION_INDEX_PAYLOAD_PATHS,
    SELECTOR_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    CAMPAIGN_PAYLOAD_PATHS,
    PREDICTION_VIEW_ARTIFACT,
    PREDICTION_VIEW_PAYLOAD_PATHS,
    PREPARE_CAMPAIGN_ARTIFACT,
    PROTOCOL_ARTIFACT,
    PROTOCOL_PAYLOAD_PATHS,
    RANDOM_MINIMAL_VIEW_ARTIFACT,
    RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS,
    PrepareRotationAttestation,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    prepare_rotation_attestation_from_document,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_runs_for_rotation,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    validate_relative_path,
    verify_phase_capability,
)

SCHEMA_VERSION = 1
PHASE_CAPABILITY_WIRE_ARTIFACT = "sequential_v2_phase_capability_wire_v1"
PROTOCOL_WORKER_REQUEST_ARTIFACT = "sequential_v2_protocol_worker_request_v1"
PREPARE_WORKER_REQUEST_ARTIFACT = "sequential_v2_prepare_worker_request_v1"
PREPARE_BARRIER_WORKER_REQUEST_ARTIFACT = "sequential_v2_prepare_barrier_worker_request_v1"
SELECT_PREDICTION_WORKER_REQUEST_ARTIFACT = "sequential_v2_select_prediction_worker_request_v1"
SELECT_RANDOM_WORKER_REQUEST_ARTIFACT = "sequential_v2_select_random_worker_request_v1"
SELECT_CEILING_WORKER_REQUEST_ARTIFACT = "sequential_v2_select_ceiling_worker_request_v1"
SELECT_ROTATION_WORKER_REQUEST_ARTIFACT = "sequential_v2_select_rotation_worker_request_v1"
SELECT_BARRIER_WORKER_REQUEST_ARTIFACT = "sequential_v2_select_barrier_worker_request_v1"
SELECT_ROTATION_ATTESTATION_ARTIFACT = "sequential_v2_select_rotation_attestation_v1"
PHASE_PUBLICATION_ATTESTATION_ARTIFACT = "sequential_v2_phase_publication_attestation_v1"

PROTOCOL_WORKER_ROLE = "protocol"
PREPARE_WORKER_ROLE = "prepare"
PREPARE_BARRIER_WORKER_ROLE = "prepare-barrier"
SELECT_PREDICTION_WORKER_ROLE = "select-prediction"
SELECT_RANDOM_WORKER_ROLE = "select-random"
SELECT_CEILING_WORKER_ROLE = "select-ceiling"
SELECT_ROTATION_WORKER_ROLE = "select-rotation"
SELECT_BARRIER_WORKER_ROLE = "select-barrier"

_PHASE_PUBLICATION_WORKER_ROLES = frozenset(
    {
        PROTOCOL_WORKER_ROLE,
        PREPARE_BARRIER_WORKER_ROLE,
        SELECT_PREDICTION_WORKER_ROLE,
        SELECT_RANDOM_WORKER_ROLE,
        SELECT_CEILING_WORKER_ROLE,
        SELECT_ROTATION_WORKER_ROLE,
        SELECT_BARRIER_WORKER_ROLE,
        "reveal-barrier",
        "update-barrier",
        "outer-select-barrier",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PHASE_ARTIFACT = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_MAX_CAPTURED_FIELD_BYTES = 128 * 1024 * 1024
_MAX_CAPTURED_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_CAPTURED_BASE64_CHARACTERS = 192 * 1024 * 1024
_MAX_PHASE_PAYLOAD_COUNT = 1_024
_MAX_PHASE_PREDECESSOR_COUNT = 4_096
_MAX_PHASE_STRUCTURAL_CHARACTERS = 16 * 1024 * 1024
_MAX_REQUEST_CAPTURED_BYTES = 128 * 1024 * 1024
_MAX_REQUEST_BASE64_CHARACTERS = 192 * 1024 * 1024
_MAX_WIRE_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_WIRE_RESULT_BYTES = 1 * 1024 * 1024
_PHASE_CAPABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "phase_artifact",
        "seal_sha256",
        "receipt_sha256",
        "predecessor_seals",
        "payload_sha256",
        "payload_base64",
        "files",
        "metadata_base64",
    }
)
_PUBLICATION_IDENTITY_FIELDS = frozenset(
    {"git_commit", "code_manifest_sha256", "config_sha256", "lock_sha256"}
)
_FORBIDDEN_ROUTING_FIELDS = frozenset(
    {
        "absolute_path",
        "cwd",
        "destination",
        "destination_path",
        "input_path",
        "input_root",
        "leaf_path",
        "output_path",
        "output_root",
        "root",
        "source_dir",
        "source_directory",
        "source_path",
        "source_root",
        "stage_dir",
        "stage_directory",
        "stage_path",
        "stage_root",
        "working_directory",
    }
)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _exact_object(value: object, *, fields: frozenset[str], label: str) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(key) is not str for key in value)
    ):
        raise ValueError(f"{label} must contain its exact field set")
    return value


@dataclass(frozen=True, slots=True)
class _CaptureMeasurements:
    decoded_bytes: int
    base64_characters: int


def _base64_length(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


def _bounded_canonical_bytes(
    value: object,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    payload = canonical_json_bytes(value)
    if len(payload) > maximum_bytes:
        raise ValueError(f"{label} exceeds its encoded byte bound")
    return payload


def _bounded_structural_characters(values: tuple[str, ...], *, label: str) -> None:
    total = 0
    for value in values:
        if type(value) is not str:
            raise ValueError(f"{label} structural values must be exact strings")
        total += len(value)
        if total > _MAX_PHASE_STRUCTURAL_CHARACTERS:
            raise ValueError(f"{label} exceeds its structural character bound")


def _phase_seal_capture_measurements(
    seal: PhaseSeal,
    *,
    label: str,
) -> _CaptureMeasurements:
    """Bound one in-memory seal before allocating any base64 representation."""

    if type(seal) is not PhaseSeal:
        raise TypeError(f"{label} must be an exact PhaseSeal")
    if (
        type(seal.predecessor_seals) is not tuple
        or len(seal.predecessor_seals) > _MAX_PHASE_PREDECESSOR_COUNT
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in seal.predecessor_seals
        )
    ):
        raise ValueError(f"{label} predecessor inventory is not bounded exact pairs")
    if (
        type(seal.payload_sha256) is not tuple
        or len(seal.payload_sha256) > _MAX_PHASE_PAYLOAD_COUNT
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in seal.payload_sha256
        )
    ):
        raise ValueError(f"{label} payload-hash inventory is not bounded exact pairs")
    if (
        type(seal.payload_bytes) is not tuple
        or len(seal.payload_bytes) > _MAX_PHASE_PAYLOAD_COUNT
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not bytes
            for item in seal.payload_bytes
        )
    ):
        raise ValueError(f"{label} captured payload inventory is not bounded exact pairs")
    if (
        type(seal.files) is not tuple
        or len(seal.files) > _MAX_PHASE_PAYLOAD_COUNT + 2
        or any(type(item) is not str for item in seal.files)
    ):
        raise ValueError(f"{label} file inventory is not a bounded exact string tuple")
    if type(seal.metadata_json) is not bytes:
        raise ValueError(f"{label} metadata must be exact bytes")

    captured = (*tuple(payload for _path, payload in seal.payload_bytes), seal.metadata_json)
    if any(len(payload) > _MAX_CAPTURED_FIELD_BYTES for payload in captured):
        raise ValueError(f"{label} contains an oversized captured field")
    decoded_bytes = sum(len(payload) for payload in captured)
    base64_characters = sum(_base64_length(len(payload)) for payload in captured)
    if decoded_bytes > _MAX_CAPTURED_TOTAL_BYTES:
        raise ValueError(f"{label} exceeds its aggregate captured-byte bound")
    if base64_characters > _MAX_CAPTURED_BASE64_CHARACTERS:
        raise ValueError(f"{label} exceeds its aggregate base64 bound")
    _bounded_structural_characters(
        (
            seal.artifact,
            seal.seal_sha256,
            seal.receipt_sha256,
            *(value for item in seal.predecessor_seals for value in item),
            *(value for item in seal.payload_sha256 for value in item),
            *(path for path, _payload in seal.payload_bytes),
            *seal.files,
        ),
        label=label,
    )
    return _CaptureMeasurements(
        decoded_bytes=decoded_bytes,
        base64_characters=base64_characters,
    )


def _base64_decoded_size(value: object, *, label: str) -> int:
    """Return decoded size after allocation-free structural base64 checks."""

    if type(value) is not str or len(value) > _base64_length(_MAX_CAPTURED_FIELD_BYTES):
        raise ValueError(f"{label} is not a bounded base64 string")
    if not value.isascii() or len(value) % 4 != 0:
        raise ValueError(f"{label} is not canonical base64")
    padding = 2 if value.endswith("==") else 1 if value.endswith("=") else 0
    unpadded = len(value) - padding
    if "=" in value[:unpadded]:
        raise ValueError(f"{label} is not canonical base64")
    decoded_size = (len(value) // 4) * 3 - padding
    if decoded_size > _MAX_CAPTURED_FIELD_BYTES:
        raise ValueError(f"{label} is not a bounded base64 string")
    return decoded_size


def _phase_document_capture_measurements(
    value: object,
    *,
    label: str,
) -> _CaptureMeasurements:
    """Preflight one parsed capability before allocating decoded byte fields."""

    document = _exact_object(value, fields=_PHASE_CAPABILITY_FIELDS, label=label)
    predecessor_raw = document["predecessor_seals"]
    payload_sha_raw = document["payload_sha256"]
    payload_base64_raw = document["payload_base64"]
    files_raw = document["files"]
    if (
        type(predecessor_raw) is not dict
        or len(predecessor_raw) > _MAX_PHASE_PREDECESSOR_COUNT
        or type(payload_sha_raw) is not dict
        or len(payload_sha_raw) > _MAX_PHASE_PAYLOAD_COUNT
        or type(payload_base64_raw) is not dict
        or len(payload_base64_raw) > _MAX_PHASE_PAYLOAD_COUNT
        or type(files_raw) is not list
        or len(files_raw) > _MAX_PHASE_PAYLOAD_COUNT + 2
    ):
        raise ValueError(f"{label} contains an unbounded capability inventory")
    if (
        any(type(key) is not str for key in predecessor_raw)
        or any(type(key) is not str for key in payload_sha_raw)
        or any(type(key) is not str for key in payload_base64_raw)
        or any(type(item) is not str for item in files_raw)
    ):
        raise ValueError(f"{label} capability inventories require exact strings")
    if set(payload_base64_raw) != set(payload_sha_raw):
        raise ValueError(f"{label} captured bytes do not match the payload inventory")

    metadata = document["metadata_base64"]
    decoded_bytes = _base64_decoded_size(metadata, label=f"{label} metadata")
    base64_characters = len(metadata)
    for path in payload_base64_raw:
        encoded = payload_base64_raw[path]
        decoded_bytes += _base64_decoded_size(
            encoded,
            label=f"{label} payload bytes {path}",
        )
        base64_characters += len(encoded)
        if decoded_bytes > _MAX_CAPTURED_TOTAL_BYTES:
            raise ValueError(f"{label} exceeds its aggregate captured-byte bound")
        if base64_characters > _MAX_CAPTURED_BASE64_CHARACTERS:
            raise ValueError(f"{label} exceeds its aggregate base64 bound")
    _bounded_structural_characters(
        (
            *(key for key in document),
            *(key for key in predecessor_raw),
            *(value for value in predecessor_raw.values() if type(value) is str),
            *(key for key in payload_sha_raw),
            *(value for value in payload_sha_raw.values() if type(value) is str),
            *(key for key in payload_base64_raw),
            *files_raw,
        ),
        label=label,
    )
    return _CaptureMeasurements(
        decoded_bytes=decoded_bytes,
        base64_characters=base64_characters,
    )


def _require_request_capture_bounds(
    measurements: tuple[_CaptureMeasurements, ...],
    *,
    label: str,
) -> None:
    decoded_bytes = sum(item.decoded_bytes for item in measurements)
    base64_characters = sum(item.base64_characters for item in measurements)
    if decoded_bytes > _MAX_REQUEST_CAPTURED_BYTES:
        raise ValueError(f"{label} exceeds its aggregate captured-byte bound")
    if base64_characters > _MAX_REQUEST_BASE64_CHARACTERS:
        raise ValueError(f"{label} exceeds its aggregate base64 bound")


def strict_canonical_json_object(
    payload: bytes,
    *,
    label: str,
    maximum_bytes: int = _MAX_WIRE_REQUEST_BYTES,
) -> dict[str, object]:
    """Decode one exact canonical LF-terminated JSON object without duplicates."""

    if type(payload) is not bytes or not payload:
        raise TypeError(f"{label} must be nonempty exact bytes")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("canonical JSON byte bound must be one positive exact integer")
    if len(payload) > maximum_bytes:
        raise ValueError(f"{label} exceeds its encoded byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains invalid JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not one canonical LF-terminated JSON object")
    return value


def _base64_bytes(value: bytes) -> str:
    if type(value) is not bytes or len(value) > _MAX_CAPTURED_FIELD_BYTES:
        raise ValueError("captured capability field is not bounded exact bytes")
    return base64.b64encode(value).decode("ascii")


def _bytes_from_base64(value: object, *, label: str) -> bytes:
    if type(value) is not str or len(value) > _base64_length(_MAX_CAPTURED_FIELD_BYTES):
        raise ValueError(f"{label} is not a bounded base64 string")
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical base64") from error
    if len(payload) > _MAX_CAPTURED_FIELD_BYTES or _base64_bytes(payload) != value:
        raise ValueError(f"{label} is not canonical bounded base64")
    return payload


def phase_seal_document(seal: PhaseSeal) -> dict[str, object]:
    """Serialize one self-verifying rootless phase capability."""

    _phase_seal_capture_measurements(seal, label="wire phase capability")
    verified = verify_phase_capability(seal, expected_seal_sha256=seal.seal_sha256)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": PHASE_CAPABILITY_WIRE_ARTIFACT,
        "phase_artifact": verified.artifact,
        "seal_sha256": verified.seal_sha256,
        "receipt_sha256": verified.receipt_sha256,
        "predecessor_seals": dict(verified.predecessor_seals),
        "payload_sha256": dict(verified.payload_sha256),
        "payload_base64": {
            path: _base64_bytes(payload) for path, payload in verified.payload_bytes
        },
        "files": list(verified.files),
        "metadata_base64": _base64_bytes(verified.metadata_json),
    }


def phase_seal_from_document(value: object) -> PhaseSeal:
    """Strictly reconstruct and authenticate one pathless phase capability."""

    document = _exact_object(
        value,
        fields=_PHASE_CAPABILITY_FIELDS,
        label="phase capability wire document",
    )
    _phase_document_capture_measurements(
        document,
        label="phase capability wire document",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != PHASE_CAPABILITY_WIRE_ARTIFACT
        or type(document["phase_artifact"]) is not str
        or not document["phase_artifact"]
    ):
        raise ValueError("phase capability wire identity is invalid")
    predecessor_raw = document["predecessor_seals"]
    payload_sha_raw = document["payload_sha256"]
    payload_base64_raw = document["payload_base64"]
    if any(
        type(item) is not dict for item in (predecessor_raw, payload_sha_raw, payload_base64_raw)
    ):
        raise ValueError("phase capability wire mappings must be exact JSON objects")
    predecessors = tuple(
        sorted(
            (
                key,
                _sha256(digest, label=f"phase capability predecessor {key}"),
            )
            for key, digest in predecessor_raw.items()
            if type(key) is str
        )
    )
    if len(predecessors) != len(predecessor_raw):
        raise ValueError("phase capability predecessor keys must be exact strings")
    payload_sha256 = tuple(
        sorted(
            (
                key,
                _sha256(digest, label=f"phase capability payload {key}"),
            )
            for key, digest in payload_sha_raw.items()
            if type(key) is str
        )
    )
    if len(payload_sha256) != len(payload_sha_raw):
        raise ValueError("phase capability payload-hash keys must be exact strings")
    if set(payload_base64_raw) != set(payload_sha_raw) or any(
        type(key) is not str for key in payload_base64_raw
    ):
        raise ValueError("phase capability captured bytes do not match its payload inventory")
    payload_bytes = tuple(
        (
            path,
            _bytes_from_base64(
                payload_base64_raw[path],
                label=f"phase capability payload bytes {path}",
            ),
        )
        for path, _digest in payload_sha256
    )
    files_raw = document["files"]
    if type(files_raw) is not list or any(type(item) is not str for item in files_raw):
        raise ValueError("phase capability file inventory must be an exact string list")
    seal = PhaseSeal(
        artifact=document["phase_artifact"],
        seal_sha256=_sha256(document["seal_sha256"], label="phase capability seal"),
        receipt_sha256=_sha256(
            document["receipt_sha256"],
            label="phase capability receipt",
        ),
        predecessor_seals=predecessors,
        payload_sha256=payload_sha256,
        payload_bytes=payload_bytes,
        files=tuple(files_raw),
        metadata_json=_bytes_from_base64(
            document["metadata_base64"],
            label="phase capability metadata",
        ),
    )
    return verify_phase_capability(seal, expected_seal_sha256=seal.seal_sha256)


def publication_identity_document(
    identity: SequentialV2PublicationIdentity,
) -> dict[str, object]:
    if type(identity) is not SequentialV2PublicationIdentity:
        raise TypeError("wire publication identity must be exact")
    return {
        "git_commit": identity.git_commit,
        "code_manifest_sha256": identity.code_manifest_sha256,
        "config_sha256": identity.config_sha256,
        "lock_sha256": identity.lock_sha256,
    }


def publication_identity_from_document(value: object) -> SequentialV2PublicationIdentity:
    document = _exact_object(
        value,
        fields=_PUBLICATION_IDENTITY_FIELDS,
        label="publication identity",
    )
    if any(type(document[key]) is not str for key in _PUBLICATION_IDENTITY_FIELDS):
        raise ValueError("publication identity fields must be exact strings")
    if _GIT_COMMIT.fullmatch(document["git_commit"]) is None:
        raise ValueError("publication identity git commit is invalid")
    return SequentialV2PublicationIdentity(
        git_commit=document["git_commit"],
        code_manifest_sha256=document["code_manifest_sha256"],
        config_sha256=document["config_sha256"],
        lock_sha256=document["lock_sha256"],
    )


def _frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} folds must be exact integers")
    canonical = rotation_by_id(value.rotation_id)
    if canonical != value:
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _verify_request_phase(
    seal: object,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    expected_artifact: str,
    expected_payload_paths: tuple[str, ...],
    phase: str,
    scope_id: str,
    label: str,
    expected_seal_sha256: str | None = None,
) -> PhaseSeal:
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError(f"{label} requires an exact publication identity")
    if type(seal) is not PhaseSeal:
        raise TypeError(f"{label} must be an exact PhaseSeal")
    verified = verify_phase_capability(
        seal,
        expected_artifact=expected_artifact,
        expected_payload_paths=expected_payload_paths,
        expected_seal_sha256=expected_seal_sha256,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase=phase,
        scope_id=scope_id,
    )
    return verified


def _validate_select_common(
    *,
    spec: RotationSpec | None,
    publication_identity: object,
    protocol_capability: object,
    prepare_campaign_seal: object,
    expected_prepare_campaign_seal_sha256: object,
    label: str,
) -> None:
    if spec is not None:
        _frozen_rotation(spec, label=f"{label} rotation")
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError(f"{label} requires an exact publication identity")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError(f"{label} requires an exact protocol capability")
    if type(protocol_capability.seal) is not PhaseSeal:
        raise TypeError(f"{label} protocol capability must contain an exact PhaseSeal")
    expected_campaign = _sha256(
        expected_prepare_campaign_seal_sha256,
        label=f"{label} expected prepare campaign seal",
    )
    if type(prepare_campaign_seal) is not PhaseSeal:
        raise TypeError(f"{label} prepare campaign must be an exact PhaseSeal")
    if prepare_campaign_seal.seal_sha256 != expected_campaign:
        raise ValueError(f"{label} prepare campaign differs from external authority")
    _verify_request_phase(
        protocol_capability.seal,
        publication_identity=publication_identity,
        expected_artifact=PROTOCOL_ARTIFACT,
        expected_payload_paths=PROTOCOL_PAYLOAD_PATHS,
        phase="protocol",
        scope_id="global",
        label=f"{label} protocol capability",
    )
    _verify_request_phase(
        prepare_campaign_seal,
        publication_identity=publication_identity,
        expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
        phase="prepare",
        scope_id="global",
        label=f"{label} prepare campaign capability",
        expected_seal_sha256=expected_campaign,
    )


def _select_common_capabilities(
    *,
    protocol_capability: ProtocolCapability,
    prepare_campaign_seal: PhaseSeal,
    additional: tuple[PhaseSeal, ...],
    label: str,
) -> None:
    _require_request_capture_bounds(
        tuple(
            _phase_seal_capture_measurements(seal, label=f"{label} capability")
            for seal in (
                protocol_capability.seal,
                prepare_campaign_seal,
                *additional,
            )
        ),
        label=label,
    )


def _preflight_select_request_capabilities(
    document: dict[str, object],
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    _require_request_capture_bounds(
        tuple(
            _phase_document_capture_measurements(
                document[field],
                label=f"{label} {field}",
            )
            for field in fields
        ),
        label=label,
    )


@dataclass(frozen=True, slots=True)
class ProtocolWorkerRequest:
    """The complete outcome-free authority for one protocol custodian."""

    publication_identity: SequentialV2PublicationIdentity

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("protocol worker request requires an exact publication identity")

    def canonical_bytes(self) -> bytes:
        return _bounded_canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": PROTOCOL_WORKER_REQUEST_ARTIFACT,
                "publication_identity": publication_identity_document(self.publication_identity),
            },
            label="protocol worker request",
            maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
        )


def protocol_worker_request_from_bytes(payload: bytes) -> ProtocolWorkerRequest:
    document = _exact_object(
        strict_canonical_json_object(payload, label="protocol worker request"),
        fields=frozenset({"schema_version", "artifact", "publication_identity"}),
        label="protocol worker request",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != PROTOCOL_WORKER_REQUEST_ARTIFACT
    ):
        raise ValueError("protocol worker request identity is invalid")
    result = ProtocolWorkerRequest(
        publication_identity=publication_identity_from_document(document["publication_identity"])
    )
    if result.canonical_bytes() != payload:
        raise ValueError("protocol worker request changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class PrepareWorkerRequest:
    """One rotation's only allowed prepare-worker capabilities."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    stage_global_seal: PhaseSeal
    expected_stage_global_seal_sha256: str
    source_prepare_leaf_seal: PhaseSeal

    def __post_init__(self) -> None:
        if type(self.spec) is not RotationSpec or self.spec != rotation_by_id(
            self.spec.rotation_id
        ):
            raise ValueError("prepare worker request has a non-frozen rotation")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("prepare worker request requires an exact publication identity")
        if type(self.protocol_capability) is not ProtocolCapability:
            raise TypeError("prepare worker request requires an exact protocol capability")
        if (
            type(self.stage_global_seal) is not PhaseSeal
            or type(self.source_prepare_leaf_seal) is not PhaseSeal
        ):
            raise TypeError("prepare worker request requires exact phase capabilities")
        expected = _sha256(
            self.expected_stage_global_seal_sha256,
            label="prepare worker expected stage-global seal",
        )
        if self.stage_global_seal.seal_sha256 != expected:
            raise ValueError("prepare worker stage-global capability differs from authority")
        verify_phase_capability(
            self.protocol_capability.seal,
            expected_seal_sha256=self.protocol_capability.seal.seal_sha256,
        )
        verify_phase_capability(self.stage_global_seal, expected_seal_sha256=expected)
        verify_phase_capability(
            self.source_prepare_leaf_seal,
            expected_seal_sha256=self.source_prepare_leaf_seal.seal_sha256,
        )

    def canonical_bytes(self) -> bytes:
        _require_request_capture_bounds(
            tuple(
                _phase_seal_capture_measurements(seal, label="prepare worker capability")
                for seal in (
                    self.protocol_capability.seal,
                    self.stage_global_seal,
                    self.source_prepare_leaf_seal,
                )
            ),
            label="prepare worker request",
        )
        return _bounded_canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": PREPARE_WORKER_REQUEST_ARTIFACT,
                "rotation_id": self.spec.rotation_id,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "stage_global_capability": phase_seal_document(self.stage_global_seal),
                "expected_stage_global_seal_sha256": (self.expected_stage_global_seal_sha256),
                "source_prepare_leaf_capability": phase_seal_document(
                    self.source_prepare_leaf_seal
                ),
            },
            label="prepare worker request",
            maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
        )


def prepare_worker_request_from_bytes(payload: bytes) -> PrepareWorkerRequest:
    document = _exact_object(
        strict_canonical_json_object(payload, label="prepare worker request"),
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "rotation_id",
                "publication_identity",
                "protocol_capability",
                "stage_global_capability",
                "expected_stage_global_seal_sha256",
                "source_prepare_leaf_capability",
            }
        ),
        label="prepare worker request",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != PREPARE_WORKER_REQUEST_ARTIFACT
        or type(document["rotation_id"]) is not str
    ):
        raise ValueError("prepare worker request identity is invalid")
    _require_request_capture_bounds(
        tuple(
            _phase_document_capture_measurements(
                document[field],
                label=f"prepare worker {field}",
            )
            for field in (
                "protocol_capability",
                "stage_global_capability",
                "source_prepare_leaf_capability",
            )
        ),
        label="prepare worker request",
    )
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = PrepareWorkerRequest(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        stage_global_seal=phase_seal_from_document(document["stage_global_capability"]),
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        source_prepare_leaf_seal=phase_seal_from_document(
            document["source_prepare_leaf_capability"]
        ),
    )
    if result.canonical_bytes() != payload:
        raise ValueError("prepare worker request changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class PrepareBarrierWorkerRequest:
    """Only the safe inputs allowed inside the prepare-global custodian."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    attestations: tuple[PrepareRotationAttestation, ...]

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("prepare barrier requires an exact publication identity")
        if type(self.protocol_capability) is not ProtocolCapability:
            raise TypeError("prepare barrier requires an exact protocol capability")
        expected_rotations = ordered_rotations()
        if (
            type(self.attestations) is not tuple
            or len(self.attestations) != len(expected_rotations)
            or any(type(item) is not PrepareRotationAttestation for item in self.attestations)
            or tuple(item.spec for item in self.attestations) != expected_rotations
            or any(
                item.publication_identity != self.publication_identity for item in self.attestations
            )
        ):
            raise ValueError("prepare barrier requires twenty ordered exact attestations")

    def canonical_bytes(self) -> bytes:
        _require_request_capture_bounds(
            (
                _phase_seal_capture_measurements(
                    self.protocol_capability.seal,
                    label="prepare barrier protocol capability",
                ),
            ),
            label="prepare barrier worker request",
        )
        return _bounded_canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": PREPARE_BARRIER_WORKER_REQUEST_ARTIFACT,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "prepare_rotation_attestations": [item.document() for item in self.attestations],
            },
            label="prepare barrier worker request",
            maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
        )


def prepare_barrier_worker_request_from_bytes(
    payload: bytes,
) -> PrepareBarrierWorkerRequest:
    document = _exact_object(
        strict_canonical_json_object(payload, label="prepare barrier worker request"),
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "publication_identity",
                "protocol_capability",
                "prepare_rotation_attestations",
            }
        ),
        label="prepare barrier worker request",
    )
    attestations_raw = document["prepare_rotation_attestations"]
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != PREPARE_BARRIER_WORKER_REQUEST_ARTIFACT
        or type(attestations_raw) is not list
    ):
        raise ValueError("prepare barrier worker request identity is invalid")
    _require_request_capture_bounds(
        (
            _phase_document_capture_measurements(
                document["protocol_capability"],
                label="prepare barrier protocol capability",
            ),
        ),
        label="prepare barrier worker request",
    )
    result = PrepareBarrierWorkerRequest(
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        attestations=tuple(
            prepare_rotation_attestation_from_document(item) for item in attestations_raw
        ),
    )
    if result.canonical_bytes() != payload:
        raise ValueError("prepare barrier request changed during typed reconstruction")
    return result


def _select_leaf_worker_request_bytes(
    *,
    artifact: str,
    label: str,
    spec: RotationSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign_seal: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    view_field: str,
    view_seal: PhaseSeal,
) -> bytes:
    _select_common_capabilities(
        protocol_capability=protocol_capability,
        prepare_campaign_seal=prepare_campaign_seal,
        additional=(view_seal,),
        label=label,
    )
    return _bounded_canonical_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": artifact,
            "rotation_id": spec.rotation_id,
            "publication_identity": publication_identity_document(publication_identity),
            "protocol_capability": phase_seal_document(protocol_capability.seal),
            "prepare_campaign_capability": phase_seal_document(prepare_campaign_seal),
            "expected_prepare_campaign_seal_sha256": (expected_prepare_campaign_seal_sha256),
            view_field: phase_seal_document(view_seal),
        },
        label=label,
        maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
    )


def _select_leaf_worker_request_document(
    payload: bytes,
    *,
    artifact: str,
    label: str,
    view_field: str,
) -> dict[str, object]:
    document = _exact_object(
        strict_canonical_json_object(payload, label=label),
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "rotation_id",
                "publication_identity",
                "protocol_capability",
                "prepare_campaign_capability",
                "expected_prepare_campaign_seal_sha256",
                view_field,
            }
        ),
        label=label,
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != artifact
        or type(document["rotation_id"]) is not str
    ):
        raise ValueError(f"{label} identity is invalid")
    _preflight_select_request_capabilities(
        document,
        fields=(
            "protocol_capability",
            "prepare_campaign_capability",
            view_field,
        ),
        label=label,
    )
    return document


@dataclass(frozen=True, slots=True)
class SelectPredictionWorkerRequest:
    """Only the prediction view and global authorities for one selector."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    prediction_view_seal: PhaseSeal

    def __post_init__(self) -> None:
        label = "select prediction worker request"
        _validate_select_common(
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            label=label,
        )
        _verify_request_phase(
            self.prediction_view_seal,
            publication_identity=self.publication_identity,
            expected_artifact=PREDICTION_VIEW_ARTIFACT,
            expected_payload_paths=PREDICTION_VIEW_PAYLOAD_PATHS,
            phase="prepare",
            scope_id=self.spec.rotation_id,
            label=f"{label} prediction view",
        )

    def canonical_bytes(self) -> bytes:
        return _select_leaf_worker_request_bytes(
            artifact=SELECT_PREDICTION_WORKER_REQUEST_ARTIFACT,
            label="select prediction worker request",
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            view_field="prediction_view_capability",
            view_seal=self.prediction_view_seal,
        )


def select_prediction_worker_request_from_bytes(
    payload: bytes,
) -> SelectPredictionWorkerRequest:
    label = "select prediction worker request"
    document = _select_leaf_worker_request_document(
        payload,
        artifact=SELECT_PREDICTION_WORKER_REQUEST_ARTIFACT,
        label=label,
        view_field="prediction_view_capability",
    )
    result = SelectPredictionWorkerRequest(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        prediction_view_seal=phase_seal_from_document(document["prediction_view_capability"]),
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class SelectRandomWorkerRequest:
    """Only the stripped random view and global authorities for one selector."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    random_minimal_view_seal: PhaseSeal

    def __post_init__(self) -> None:
        label = "select random worker request"
        _validate_select_common(
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            label=label,
        )
        _verify_request_phase(
            self.random_minimal_view_seal,
            publication_identity=self.publication_identity,
            expected_artifact=RANDOM_MINIMAL_VIEW_ARTIFACT,
            expected_payload_paths=RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS,
            phase="prepare",
            scope_id=self.spec.rotation_id,
            label=f"{label} random-minimal view",
        )

    def canonical_bytes(self) -> bytes:
        return _select_leaf_worker_request_bytes(
            artifact=SELECT_RANDOM_WORKER_REQUEST_ARTIFACT,
            label="select random worker request",
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            view_field="random_minimal_view_capability",
            view_seal=self.random_minimal_view_seal,
        )


def select_random_worker_request_from_bytes(payload: bytes) -> SelectRandomWorkerRequest:
    label = "select random worker request"
    document = _select_leaf_worker_request_document(
        payload,
        artifact=SELECT_RANDOM_WORKER_REQUEST_ARTIFACT,
        label=label,
        view_field="random_minimal_view_capability",
    )
    result = SelectRandomWorkerRequest(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        random_minimal_view_seal=phase_seal_from_document(
            document["random_minimal_view_capability"]
        ),
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class SelectCeilingWorkerRequest:
    """A fresh ceiling selector with only the stripped random-minimal view."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    random_minimal_view_seal: PhaseSeal

    def __post_init__(self) -> None:
        label = "select ceiling worker request"
        _validate_select_common(
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            label=label,
        )
        _verify_request_phase(
            self.random_minimal_view_seal,
            publication_identity=self.publication_identity,
            expected_artifact=RANDOM_MINIMAL_VIEW_ARTIFACT,
            expected_payload_paths=RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS,
            phase="prepare",
            scope_id=self.spec.rotation_id,
            label=f"{label} random-minimal view",
        )

    def canonical_bytes(self) -> bytes:
        return _select_leaf_worker_request_bytes(
            artifact=SELECT_CEILING_WORKER_REQUEST_ARTIFACT,
            label="select ceiling worker request",
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            view_field="random_minimal_view_capability",
            view_seal=self.random_minimal_view_seal,
        )


def select_ceiling_worker_request_from_bytes(payload: bytes) -> SelectCeilingWorkerRequest:
    label = "select ceiling worker request"
    document = _select_leaf_worker_request_document(
        payload,
        artifact=SELECT_CEILING_WORKER_REQUEST_ARTIFACT,
        label=label,
        view_field="random_minimal_view_capability",
    )
    result = SelectCeilingWorkerRequest(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        random_minimal_view_seal=phase_seal_from_document(
            document["random_minimal_view_capability"]
        ),
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class SelectRotationWorkerRequest:
    """Three selector capabilities required to assemble one sealed rotation."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    prediction_selector_seal: PhaseSeal
    expected_prediction_selector_seal_sha256: str
    random_selector_seal: PhaseSeal
    expected_random_selector_seal_sha256: str
    ceiling_selector_seal: PhaseSeal
    expected_ceiling_selector_seal_sha256: str

    def __post_init__(self) -> None:
        label = "select rotation worker request"
        _validate_select_common(
            spec=self.spec,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            label=label,
        )
        selectors = (
            (
                "prediction",
                self.prediction_selector_seal,
                self.expected_prediction_selector_seal_sha256,
                PREDICTION_SELECTOR_ARTIFACT,
            ),
            (
                "random",
                self.random_selector_seal,
                self.expected_random_selector_seal_sha256,
                RANDOM_SELECTOR_ARTIFACT,
            ),
            (
                "ceiling",
                self.ceiling_selector_seal,
                self.expected_ceiling_selector_seal_sha256,
                CEILING_SELECTOR_ARTIFACT,
            ),
        )
        for kind, seal, expected_raw, artifact in selectors:
            expected = _sha256(
                expected_raw,
                label=f"{label} expected {kind} selector seal",
            )
            if type(seal) is not PhaseSeal or seal.seal_sha256 != expected:
                raise ValueError(f"{label} {kind} selector differs from external authority")
            _verify_request_phase(
                seal,
                publication_identity=self.publication_identity,
                expected_artifact=artifact,
                expected_payload_paths=SELECTOR_PAYLOAD_PATHS,
                phase="select",
                scope_id=self.spec.rotation_id,
                label=f"{label} {kind} selector",
                expected_seal_sha256=expected,
            )

    def canonical_bytes(self) -> bytes:
        label = "select rotation worker request"
        _select_common_capabilities(
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            additional=(
                self.prediction_selector_seal,
                self.random_selector_seal,
                self.ceiling_selector_seal,
            ),
            label=label,
        )
        return _bounded_canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": SELECT_ROTATION_WORKER_REQUEST_ARTIFACT,
                "rotation_id": self.spec.rotation_id,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "prepare_campaign_capability": phase_seal_document(self.prepare_campaign_seal),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "prediction_selector_capability": phase_seal_document(
                    self.prediction_selector_seal
                ),
                "expected_prediction_selector_seal_sha256": (
                    self.expected_prediction_selector_seal_sha256
                ),
                "random_selector_capability": phase_seal_document(self.random_selector_seal),
                "expected_random_selector_seal_sha256": (self.expected_random_selector_seal_sha256),
                "ceiling_selector_capability": phase_seal_document(self.ceiling_selector_seal),
                "expected_ceiling_selector_seal_sha256": (
                    self.expected_ceiling_selector_seal_sha256
                ),
            },
            label=label,
            maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
        )


def select_rotation_worker_request_from_bytes(payload: bytes) -> SelectRotationWorkerRequest:
    label = "select rotation worker request"
    capability_fields = (
        "protocol_capability",
        "prepare_campaign_capability",
        "prediction_selector_capability",
        "random_selector_capability",
        "ceiling_selector_capability",
    )
    document = _exact_object(
        strict_canonical_json_object(payload, label=label),
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "rotation_id",
                "publication_identity",
                *capability_fields,
                "expected_prepare_campaign_seal_sha256",
                "expected_prediction_selector_seal_sha256",
                "expected_random_selector_seal_sha256",
                "expected_ceiling_selector_seal_sha256",
            }
        ),
        label=label,
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != SELECT_ROTATION_WORKER_REQUEST_ARTIFACT
        or type(document["rotation_id"]) is not str
    ):
        raise ValueError(f"{label} identity is invalid")
    _preflight_select_request_capabilities(
        document,
        fields=capability_fields,
        label=label,
    )
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = SelectRotationWorkerRequest(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        prediction_selector_seal=phase_seal_from_document(
            document["prediction_selector_capability"]
        ),
        expected_prediction_selector_seal_sha256=document[
            "expected_prediction_selector_seal_sha256"
        ],
        random_selector_seal=phase_seal_from_document(document["random_selector_capability"]),
        expected_random_selector_seal_sha256=document["expected_random_selector_seal_sha256"],
        ceiling_selector_seal=phase_seal_from_document(document["ceiling_selector_capability"]),
        expected_ceiling_selector_seal_sha256=document["expected_ceiling_selector_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class SelectBarrierWorkerRequest:
    """The ordered 20-index/220-leaf authority for the select-global barrier."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    rotation_index_seals: tuple[PhaseSeal, ...]
    expected_rotation_index_seal_sha256s: tuple[str, ...]
    commitment_leaf_seals: tuple[PhaseSeal, ...]

    def __post_init__(self) -> None:
        label = "select barrier worker request"
        _validate_select_common(
            spec=None,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            label=label,
        )
        rotations = ordered_rotations()
        runs = ordered_policy_runs()
        if (
            type(self.rotation_index_seals) is not tuple
            or len(self.rotation_index_seals) != len(rotations)
            or any(type(item) is not PhaseSeal for item in self.rotation_index_seals)
        ):
            raise ValueError(f"{label} requires twenty ordered exact rotation indexes")
        if (
            type(self.expected_rotation_index_seal_sha256s) is not tuple
            or len(self.expected_rotation_index_seal_sha256s) != len(rotations)
            or any(type(item) is not str for item in self.expected_rotation_index_seal_sha256s)
        ):
            raise ValueError(f"{label} requires twenty ordered expected index seals")
        if (
            type(self.commitment_leaf_seals) is not tuple
            or len(self.commitment_leaf_seals) != len(runs)
            or any(type(item) is not PhaseSeal for item in self.commitment_leaf_seals)
        ):
            raise ValueError(f"{label} requires 220 ordered exact commitment leaves")
        for spec, seal, expected_raw in zip(
            rotations,
            self.rotation_index_seals,
            self.expected_rotation_index_seal_sha256s,
            strict=True,
        ):
            expected = _sha256(
                expected_raw,
                label=f"{label} expected rotation index seal",
            )
            if seal.seal_sha256 != expected:
                raise ValueError(f"{label} rotation index differs from external authority")
            _verify_request_phase(
                seal,
                publication_identity=self.publication_identity,
                expected_artifact=ROTATION_INDEX_ARTIFACT,
                expected_payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
                phase="select",
                scope_id=spec.rotation_id,
                label=f"{label} rotation index {spec.rotation_id}",
                expected_seal_sha256=expected,
            )
        for run, seal in zip(runs, self.commitment_leaf_seals, strict=True):
            _verify_request_phase(
                seal,
                publication_identity=self.publication_identity,
                expected_artifact=POOL_COMMITMENT_ARTIFACT,
                expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
                phase="select",
                scope_id=run.track_id,
                label=f"{label} commitment leaf {run.track_id}",
            )
        all_seals = (*self.rotation_index_seals, *self.commitment_leaf_seals)
        if len({seal.seal_sha256 for seal in all_seals}) != len(all_seals):
            raise ValueError(f"{label} phase seals must be unique")

    def canonical_bytes(self) -> bytes:
        label = "select barrier worker request"
        _select_common_capabilities(
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            additional=(*self.rotation_index_seals, *self.commitment_leaf_seals),
            label=label,
        )
        return _bounded_canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": SELECT_BARRIER_WORKER_REQUEST_ARTIFACT,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "prepare_campaign_capability": phase_seal_document(self.prepare_campaign_seal),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "rotation_index_capabilities": [
                    phase_seal_document(seal) for seal in self.rotation_index_seals
                ],
                "expected_rotation_index_seal_sha256s": list(
                    self.expected_rotation_index_seal_sha256s
                ),
                "commitment_leaf_capabilities": [
                    phase_seal_document(seal) for seal in self.commitment_leaf_seals
                ],
            },
            label=label,
            maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
        )


def select_barrier_worker_request_from_bytes(payload: bytes) -> SelectBarrierWorkerRequest:
    label = "select barrier worker request"
    document = _exact_object(
        strict_canonical_json_object(payload, label=label),
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "publication_identity",
                "protocol_capability",
                "prepare_campaign_capability",
                "expected_prepare_campaign_seal_sha256",
                "rotation_index_capabilities",
                "expected_rotation_index_seal_sha256s",
                "commitment_leaf_capabilities",
            }
        ),
        label=label,
    )
    rotation_raw = document["rotation_index_capabilities"]
    expected_rotation_raw = document["expected_rotation_index_seal_sha256s"]
    commitment_raw = document["commitment_leaf_capabilities"]
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != SELECT_BARRIER_WORKER_REQUEST_ARTIFACT
        or type(rotation_raw) is not list
        or len(rotation_raw) != len(ordered_rotations())
        or type(expected_rotation_raw) is not list
        or len(expected_rotation_raw) != len(ordered_rotations())
        or any(type(item) is not str for item in expected_rotation_raw)
        or type(commitment_raw) is not list
        or len(commitment_raw) != len(ordered_policy_runs())
    ):
        raise ValueError(f"{label} identity or ordered census is invalid")
    phase_documents = (
        document["protocol_capability"],
        document["prepare_campaign_capability"],
        *rotation_raw,
        *commitment_raw,
    )
    _require_request_capture_bounds(
        tuple(
            _phase_document_capture_measurements(item, label=f"{label} capability")
            for item in phase_documents
        ),
        label=label,
    )
    result = SelectBarrierWorkerRequest(
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        rotation_index_seals=tuple(phase_seal_from_document(item) for item in rotation_raw),
        expected_rotation_index_seal_sha256s=tuple(expected_rotation_raw),
        commitment_leaf_seals=tuple(phase_seal_from_document(item) for item in commitment_raw),
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class PhasePublicationAttestation:
    """Payload-free result for a protocol or barrier publication worker."""

    worker_role: str
    phase_artifact: str
    phase_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.worker_role) is not str
            or self.worker_role not in _PHASE_PUBLICATION_WORKER_ROLES
        ):
            raise ValueError("phase publication attestation worker role is invalid")
        if (
            type(self.phase_artifact) is not str
            or _PHASE_ARTIFACT.fullmatch(self.phase_artifact) is None
        ):
            raise ValueError("phase publication attestation artifact is invalid")
        _sha256(self.phase_seal_sha256, label="phase publication attestation seal")
        if (
            type(self.payload_sha256) is not tuple
            or len(self.payload_sha256) > _MAX_PHASE_PAYLOAD_COUNT
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                for item in self.payload_sha256
            )
            or self.payload_sha256 != tuple(sorted(self.payload_sha256))
            or len(dict(self.payload_sha256)) != len(self.payload_sha256)
        ):
            raise ValueError("phase publication attestation payload map is invalid")
        for path, digest in self.payload_sha256:
            validate_relative_path(path)
            _sha256(digest, label=f"phase publication attestation payload {path}")
        _bounded_structural_characters(
            (
                self.worker_role,
                self.phase_artifact,
                self.phase_seal_sha256,
                *(value for item in self.payload_sha256 for value in item),
            ),
            label="phase publication attestation",
        )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": PHASE_PUBLICATION_ATTESTATION_ARTIFACT,
            "worker_role": self.worker_role,
            "phase_artifact": self.phase_artifact,
            "phase_seal_sha256": self.phase_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
        }

    def canonical_bytes(self) -> bytes:
        return _bounded_canonical_bytes(
            self.document(),
            label="phase publication attestation",
            maximum_bytes=_MAX_WIRE_RESULT_BYTES,
        )

    @classmethod
    def from_seal(cls, *, worker_role: str, seal: PhaseSeal) -> PhasePublicationAttestation:
        _phase_seal_capture_measurements(seal, label="attested phase capability")
        verified = verify_phase_capability(seal, expected_seal_sha256=seal.seal_sha256)
        return cls(
            worker_role=worker_role,
            phase_artifact=verified.artifact,
            phase_seal_sha256=verified.seal_sha256,
            payload_sha256=verified.payload_sha256,
        )


def _phase_publication_attestation_from_document(
    value: object,
) -> PhasePublicationAttestation:
    document = _exact_object(
        value,
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "worker_role",
                "phase_artifact",
                "phase_seal_sha256",
                "payload_sha256",
            }
        ),
        label="phase publication attestation",
    )
    payload_raw = document["payload_sha256"]
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != PHASE_PUBLICATION_ATTESTATION_ARTIFACT
        or type(document["worker_role"]) is not str
        or type(document["phase_artifact"]) is not str
        or type(payload_raw) is not dict
        or any(type(key) is not str for key in payload_raw)
    ):
        raise ValueError("phase publication attestation identity is invalid")
    result = PhasePublicationAttestation(
        worker_role=document["worker_role"],
        phase_artifact=document["phase_artifact"],
        phase_seal_sha256=document["phase_seal_sha256"],
        payload_sha256=tuple(sorted(payload_raw.items())),
    )
    if result.document() != document:
        raise ValueError("phase publication attestation changed during reconstruction")
    return result


def phase_publication_attestation_from_bytes(
    payload: bytes,
) -> PhasePublicationAttestation:
    document = strict_canonical_json_object(
        payload,
        label="phase publication attestation",
        maximum_bytes=_MAX_WIRE_RESULT_BYTES,
    )
    result = _phase_publication_attestation_from_document(document)
    if result.canonical_bytes() != payload:
        raise ValueError("phase publication attestation changed during reconstruction")
    return result


def _has_exact_attested_payloads(
    publication: PhasePublicationAttestation,
    expected_paths: tuple[str, ...],
) -> bool:
    return tuple(path for path, _digest in publication.payload_sha256) == tuple(
        sorted(expected_paths)
    )


@dataclass(frozen=True, slots=True)
class SelectCommitmentLeafAttestation:
    """One ordered track identity paired with its payload-free publication result."""

    track_id: str
    publication: PhasePublicationAttestation

    def __post_init__(self) -> None:
        if type(self.track_id) is not str or not self.track_id:
            raise ValueError("select commitment leaf attestation track ID is invalid")
        if type(self.publication) is not PhasePublicationAttestation:
            raise TypeError("select commitment leaf attestation publication must be exact")
        if (
            self.publication.worker_role != SELECT_ROTATION_WORKER_ROLE
            or self.publication.phase_artifact != POOL_COMMITMENT_ARTIFACT
            or not _has_exact_attested_payloads(
                self.publication,
                POOL_COMMITMENT_PAYLOAD_PATHS,
            )
        ):
            raise ValueError("select commitment leaf publication shape is invalid")

    def document(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "publication": self.publication.document(),
        }


def _select_commitment_leaf_attestation_from_document(
    value: object,
) -> SelectCommitmentLeafAttestation:
    document = _exact_object(
        value,
        fields=frozenset({"track_id", "publication"}),
        label="select commitment leaf attestation",
    )
    result = SelectCommitmentLeafAttestation(
        track_id=document["track_id"],
        publication=_phase_publication_attestation_from_document(document["publication"]),
    )
    if result.document() != document:
        raise ValueError("select commitment leaf attestation changed during reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class SelectRotationAttestation:
    """Bounded ordered results from one isolated rotation assembler."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    commitment_leaves: tuple[SelectCommitmentLeafAttestation, ...]
    rotation_index: PhasePublicationAttestation

    def __post_init__(self) -> None:
        _frozen_rotation(self.spec, label="select rotation attestation rotation")
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("select rotation attestation requires an exact publication identity")
        expected_runs = policy_runs_for_rotation(self.spec)
        if (
            type(self.commitment_leaves) is not tuple
            or len(self.commitment_leaves) != len(expected_runs)
            or any(
                type(item) is not SelectCommitmentLeafAttestation for item in self.commitment_leaves
            )
            or tuple(item.track_id for item in self.commitment_leaves)
            != tuple(run.track_id for run in expected_runs)
        ):
            raise ValueError(
                "select rotation attestation requires eleven frozen ordered commitment leaves"
            )
        if type(self.rotation_index) is not PhasePublicationAttestation:
            raise TypeError("select rotation index attestation must be exact")
        if (
            self.rotation_index.worker_role != SELECT_ROTATION_WORKER_ROLE
            or self.rotation_index.phase_artifact != ROTATION_INDEX_ARTIFACT
            or not _has_exact_attested_payloads(
                self.rotation_index,
                ROTATION_INDEX_PAYLOAD_PATHS,
            )
        ):
            raise ValueError("select rotation index publication shape is invalid")
        digests = (
            *(item.publication.phase_seal_sha256 for item in self.commitment_leaves),
            self.rotation_index.phase_seal_sha256,
        )
        if len(set(digests)) != len(digests):
            raise ValueError("select rotation attestation phase seals must be unique")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": SELECT_ROTATION_ATTESTATION_ARTIFACT,
            "rotation_id": self.spec.rotation_id,
            "publication_identity": publication_identity_document(self.publication_identity),
            "commitment_leaves": [item.document() for item in self.commitment_leaves],
            "rotation_index": self.rotation_index.document(),
        }

    def canonical_bytes(self) -> bytes:
        return _bounded_canonical_bytes(
            self.document(),
            label="select rotation attestation",
            maximum_bytes=_MAX_WIRE_RESULT_BYTES,
        )

    @classmethod
    def from_seals(
        cls,
        *,
        spec: RotationSpec,
        publication_identity: SequentialV2PublicationIdentity,
        commitment_leaf_seals: tuple[PhaseSeal, ...],
        rotation_index_seal: PhaseSeal,
    ) -> SelectRotationAttestation:
        frozen_spec = _frozen_rotation(spec, label="select rotation attestation rotation")
        runs = policy_runs_for_rotation(frozen_spec)
        if (
            type(commitment_leaf_seals) is not tuple
            or len(commitment_leaf_seals) != len(runs)
            or any(type(item) is not PhaseSeal for item in commitment_leaf_seals)
        ):
            raise ValueError("select rotation result requires eleven exact commitment seals")
        for run, seal in zip(runs, commitment_leaf_seals, strict=True):
            _verify_request_phase(
                seal,
                publication_identity=publication_identity,
                expected_artifact=POOL_COMMITMENT_ARTIFACT,
                expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
                phase="select",
                scope_id=run.track_id,
                label=f"select rotation result commitment {run.track_id}",
            )
        _verify_request_phase(
            rotation_index_seal,
            publication_identity=publication_identity,
            expected_artifact=ROTATION_INDEX_ARTIFACT,
            expected_payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
            phase="select",
            scope_id=frozen_spec.rotation_id,
            label="select rotation result index",
        )
        return cls(
            spec=frozen_spec,
            publication_identity=publication_identity,
            commitment_leaves=tuple(
                SelectCommitmentLeafAttestation(
                    track_id=run.track_id,
                    publication=PhasePublicationAttestation.from_seal(
                        worker_role=SELECT_ROTATION_WORKER_ROLE,
                        seal=seal,
                    ),
                )
                for run, seal in zip(runs, commitment_leaf_seals, strict=True)
            ),
            rotation_index=PhasePublicationAttestation.from_seal(
                worker_role=SELECT_ROTATION_WORKER_ROLE,
                seal=rotation_index_seal,
            ),
        )


def select_rotation_attestation_from_document(value: object) -> SelectRotationAttestation:
    document = _exact_object(
        value,
        fields=frozenset(
            {
                "schema_version",
                "artifact",
                "rotation_id",
                "publication_identity",
                "commitment_leaves",
                "rotation_index",
            }
        ),
        label="select rotation attestation",
    )
    leaves_raw = document["commitment_leaves"]
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != SELECT_ROTATION_ATTESTATION_ARTIFACT
        or type(document["rotation_id"]) is not str
        or type(leaves_raw) is not list
        or len(leaves_raw) != len(policy_runs_for_rotation(rotation_by_id(document["rotation_id"])))
    ):
        raise ValueError("select rotation attestation identity or leaf census is invalid")
    result = SelectRotationAttestation(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        commitment_leaves=tuple(
            _select_commitment_leaf_attestation_from_document(item) for item in leaves_raw
        ),
        rotation_index=_phase_publication_attestation_from_document(document["rotation_index"]),
    )
    if result.document() != document:
        raise ValueError("select rotation attestation changed during reconstruction")
    return result


def select_rotation_attestation_from_bytes(payload: bytes) -> SelectRotationAttestation:
    document = strict_canonical_json_object(
        payload,
        label="select rotation attestation",
        maximum_bytes=_MAX_WIRE_RESULT_BYTES,
    )
    result = select_rotation_attestation_from_document(document)
    if result.canonical_bytes() != payload:
        raise ValueError("select rotation attestation changed during typed reconstruction")
    return result


def assert_wire_document_has_no_source_path_fields(value: object) -> None:
    """Reject reserved routing fields that could reveal controller paths.

    This is a structural envelope check, not a content sanitizer.  In
    particular, authenticated ``*_base64`` fields remain opaque and must be
    governed by their phase-specific semantic validators.
    """

    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("wire document keys must be strings")
            lowered = key.lower().replace("-", "_")
            if lowered in _FORBIDDEN_ROUTING_FIELDS:
                raise ValueError(f"wire document contains forbidden path field {key!r}")
            assert_wire_document_has_no_source_path_fields(item)
    elif type(value) is list:
        for item in value:
            assert_wire_document_has_no_source_path_fields(item)
    elif value is None or type(value) in {str, int, bool}:
        return
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("wire document contains a non-finite float")
    else:
        raise TypeError(f"wire document contains unsupported type {type(value).__name__}")


__all__ = [
    "PHASE_CAPABILITY_WIRE_ARTIFACT",
    "PHASE_PUBLICATION_ATTESTATION_ARTIFACT",
    "PREPARE_BARRIER_WORKER_REQUEST_ARTIFACT",
    "PREPARE_BARRIER_WORKER_ROLE",
    "PREPARE_WORKER_REQUEST_ARTIFACT",
    "PREPARE_WORKER_ROLE",
    "PROTOCOL_WORKER_REQUEST_ARTIFACT",
    "PROTOCOL_WORKER_ROLE",
    "SELECT_BARRIER_WORKER_REQUEST_ARTIFACT",
    "SELECT_BARRIER_WORKER_ROLE",
    "SELECT_CEILING_WORKER_REQUEST_ARTIFACT",
    "SELECT_CEILING_WORKER_ROLE",
    "SELECT_PREDICTION_WORKER_REQUEST_ARTIFACT",
    "SELECT_PREDICTION_WORKER_ROLE",
    "SELECT_RANDOM_WORKER_REQUEST_ARTIFACT",
    "SELECT_RANDOM_WORKER_ROLE",
    "SELECT_ROTATION_ATTESTATION_ARTIFACT",
    "SELECT_ROTATION_WORKER_REQUEST_ARTIFACT",
    "SELECT_ROTATION_WORKER_ROLE",
    "PhasePublicationAttestation",
    "PrepareBarrierWorkerRequest",
    "PrepareWorkerRequest",
    "ProtocolWorkerRequest",
    "SelectBarrierWorkerRequest",
    "SelectCeilingWorkerRequest",
    "SelectCommitmentLeafAttestation",
    "SelectPredictionWorkerRequest",
    "SelectRandomWorkerRequest",
    "SelectRotationAttestation",
    "SelectRotationWorkerRequest",
    "assert_wire_document_has_no_source_path_fields",
    "phase_publication_attestation_from_bytes",
    "phase_seal_document",
    "phase_seal_from_document",
    "prepare_barrier_worker_request_from_bytes",
    "prepare_worker_request_from_bytes",
    "protocol_worker_request_from_bytes",
    "publication_identity_document",
    "publication_identity_from_document",
    "select_barrier_worker_request_from_bytes",
    "select_ceiling_worker_request_from_bytes",
    "select_prediction_worker_request_from_bytes",
    "select_random_worker_request_from_bytes",
    "select_rotation_attestation_from_bytes",
    "select_rotation_attestation_from_document",
    "select_rotation_worker_request_from_bytes",
    "strict_canonical_json_object",
]

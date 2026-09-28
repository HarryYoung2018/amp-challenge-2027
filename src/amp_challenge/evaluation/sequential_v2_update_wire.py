"""Canonical pathless requests for fresh-executable sequential-v2 UPDATE workers.

This module is deliberately transport-only.  It captures the exact rootless
capabilities accepted by the public update publishers, rejects every extra
authority, and preserves controller-owned digest snapshots across the exec
boundary.  The worker still performs the full semantic authorization in the
update-state, outer-projection, and update-campaign modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from amp_challenge.evaluation import sequential_v2_wire as _core_wire
from amp_challenge.evaluation.sequential_v2_commitments import (
    CAMPAIGN_BARRIER_ARTIFACT as SELECT_CAMPAIGN_ARTIFACT,
)
from amp_challenge.evaluation.sequential_v2_commitments import (
    CAMPAIGN_BARRIER_PAYLOAD_PATHS as SELECT_CAMPAIGN_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    BASE_UPDATE_ARTIFACT,
    BASE_UPDATE_PAYLOAD_PATHS,
    PREPARE_CAMPAIGN_ARTIFACT,
    PROTOCOL_ARTIFACT,
    PROTOCOL_PAYLOAD_PATHS,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    CAMPAIGN_PAYLOAD_PATHS as PREPARE_CAMPAIGN_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_POLICY_RUNS,
    EXPECTED_ROTATIONS,
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_run_by_track_id,
    policy_runs_for_rotation,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    POOL_REVEAL_ARTIFACT,
    POOL_REVEAL_PAYLOAD_PATHS,
    REVEAL_CAMPAIGN_ARTIFACT,
    REVEAL_CAMPAIGN_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_seals import PhaseSeal, canonical_json_bytes
from amp_challenge.evaluation.sequential_v2_stage import (
    GLOBAL_PAYLOAD_PATHS as STAGE_GLOBAL_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    LEAF_ARTIFACTS,
    LEAF_DATA_PAYLOAD_PATHS,
    LEAF_ID_STREAM_NAMES,
    LEAF_PAYLOAD_PATHS,
    OUTER_METADATA_ROLE,
    STAGE_ARTIFACT,
    AuthenticatedLeafCapsule,
    IdStreamBinding,
    StageLeafIndex,
    leaf_relative_path,
)
from amp_challenge.evaluation.sequential_v2_update import (
    OUTER_COMPONENT_PAYLOAD_PATHS,
    OUTER_COMPONENTS_ARTIFACT,
    UPDATE_STATE_ARTIFACT,
    UPDATE_STATE_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_update_outer import (
    OuterComponentAttestation,
    OuterProjectionAttestation,
    outer_component_attestation_from_document,
    outer_projection_attestation_from_document,
)
from amp_challenge.evaluation.sequential_v2_update_state import (
    UpdateStateAttestation,
    update_state_attestation_from_document,
)
from amp_challenge.evaluation.sequential_v2_wire import (
    assert_wire_document_has_no_source_path_fields,
    phase_seal_document,
    phase_seal_from_document,
    publication_identity_document,
    publication_identity_from_document,
    strict_canonical_json_object,
)

SCHEMA_VERSION = 1

UPDATE_STATE_WORKER_REQUEST_ARTIFACT = "sequential_v2_update_state_worker_request_v1"
UPDATE_COMPONENT_WORKER_REQUEST_ARTIFACT = "sequential_v2_update_component_worker_request_v1"
UPDATE_PROJECTION_WORKER_REQUEST_ARTIFACT = "sequential_v2_update_projection_worker_request_v1"
UPDATE_BARRIER_WORKER_REQUEST_ARTIFACT = "sequential_v2_update_barrier_worker_request_v1"
OUTER_METADATA_CAPSULE_WIRE_ARTIFACT = "sequential_v2_outer_metadata_capsule_wire_v1"

UPDATE_STATE_WORKER_ROLE = "update-state"
UPDATE_COMPONENT_WORKER_ROLE = "update-component"
UPDATE_PROJECTION_WORKER_ROLE = "update-projection"
UPDATE_BARRIER_WORKER_ROLE = "update-barrier"

EXPECTED_UPDATE_LEAVES = 680
_MAX_WIRE_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_REQUEST_CAPTURED_BYTES = 128 * 1024 * 1024
_MAX_REQUEST_BASE64_CHARACTERS = 192 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPSULE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "entry",
        "leaf_capability",
        "source_anchors_sha256",
        "source_predecessors",
    }
)
_STAGE_LEAF_FIELDS = frozenset(
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


def _frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} folds must be exact integers")
    canonical = rotation_by_id(value.rotation_id)
    if canonical != value:
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    canonical = policy_run_by_track_id(value.track_id)
    if canonical != value:
        raise ValueError(f"{label} differs from the frozen policy registry")
    return value


def _require_capture_bounds(measurements: tuple[object, ...], *, label: str) -> None:
    decoded = sum(item.decoded_bytes for item in measurements)  # type: ignore[attr-defined]
    encoded = sum(item.base64_characters for item in measurements)  # type: ignore[attr-defined]
    if decoded > _MAX_REQUEST_CAPTURED_BYTES:
        raise ValueError(f"{label} exceeds its aggregate captured-byte bound")
    if encoded > _MAX_REQUEST_BASE64_CHARACTERS:
        raise ValueError(f"{label} exceeds its aggregate base64 bound")


def _preflight_seals(seals: tuple[PhaseSeal, ...], *, label: str) -> None:
    _require_capture_bounds(
        tuple(
            _core_wire._phase_seal_capture_measurements(
                seal,
                label=f"{label} capability {index}",
            )
            for index, seal in enumerate(seals)
        ),
        label=label,
    )


def _preflight_phase_documents(values: tuple[object, ...], *, label: str) -> None:
    _require_capture_bounds(
        tuple(
            _core_wire._phase_document_capture_measurements(
                value,
                label=f"{label} capability {index}",
            )
            for index, value in enumerate(values)
        ),
        label=label,
    )


def _canonical_request_bytes(document: dict[str, object], *, label: str) -> bytes:
    assert_wire_document_has_no_source_path_fields(document)
    payload = canonical_json_bytes(document)
    if len(payload) > _MAX_WIRE_REQUEST_BYTES:
        raise ValueError(f"{label} exceeds its encoded byte bound")
    return payload


def _request_document(payload: bytes, *, fields: frozenset[str], label: str) -> dict[str, object]:
    document = _exact_object(
        strict_canonical_json_object(
            payload,
            label=label,
            maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
        ),
        fields=fields,
        label=label,
    )
    assert_wire_document_has_no_source_path_fields(document)
    return document


def _validate_request_identity(
    document: dict[str, object],
    *,
    artifact: str,
    label: str,
) -> None:
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != artifact
    ):
        raise ValueError(f"{label} identity is invalid")


def _verify_phase_shape(
    seal: object,
    *,
    artifact: str,
    payload_paths: tuple[str, ...],
    label: str,
    expected_seal_sha256: str | None = None,
    publication_identity: SequentialV2PublicationIdentity | None = None,
    phase: str | None = None,
    scope_id: str | None = None,
) -> PhaseSeal:
    if type(seal) is not PhaseSeal:
        raise TypeError(f"{label} must be an exact PhaseSeal")
    expected = (
        seal.seal_sha256
        if expected_seal_sha256 is None
        else _sha256(expected_seal_sha256, label=f"{label} expected seal")
    )
    if seal.seal_sha256 != expected:
        raise ValueError(f"{label} differs from external authority")
    verified = _core_wire.verify_phase_capability(
        seal,
        expected_artifact=artifact,
        expected_payload_paths=payload_paths,
        expected_seal_sha256=expected,
    )
    if publication_identity is not None:
        if phase is None or scope_id is None:
            raise AssertionError("phase metadata validation requires a complete identity")
        publication_identity.verify_metadata(
            verified.metadata_json,
            phase=phase,
            scope_id=scope_id,
        )
    return verified


def _validate_protocol(
    capability: object,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    label: str,
) -> ProtocolCapability:
    if type(capability) is not ProtocolCapability:
        raise TypeError(f"{label} requires an exact ProtocolCapability")
    _verify_phase_shape(
        capability.seal,
        artifact=PROTOCOL_ARTIFACT,
        payload_paths=PROTOCOL_PAYLOAD_PATHS,
        label=f"{label} protocol capability",
        publication_identity=publication_identity,
        phase="protocol",
        scope_id="global",
    )
    return verify_protocol_capability(
        capability.seal,
        publication_identity=publication_identity,
    )


def _validate_identity(value: object, *, label: str) -> SequentialV2PublicationIdentity:
    if type(value) is not SequentialV2PublicationIdentity:
        raise TypeError(f"{label} requires an exact publication identity")
    publication_identity_document(value)
    return value


def _validate_stage_global(
    seal: object,
    *,
    expected_seal_sha256: str,
    label: str,
) -> PhaseSeal:
    return _verify_phase_shape(
        seal,
        artifact=STAGE_ARTIFACT,
        payload_paths=STAGE_GLOBAL_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal_sha256,
        label=label,
    )


def _validate_prepare_global(
    seal: object,
    *,
    expected_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    label: str,
) -> PhaseSeal:
    return _verify_phase_shape(
        seal,
        artifact=PREPARE_CAMPAIGN_ARTIFACT,
        payload_paths=PREPARE_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal_sha256,
        publication_identity=publication_identity,
        phase="prepare",
        scope_id="global",
        label=label,
    )


def _validate_select_global(
    seal: object,
    *,
    expected_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    label: str,
) -> PhaseSeal:
    return _verify_phase_shape(
        seal,
        artifact=SELECT_CAMPAIGN_ARTIFACT,
        payload_paths=SELECT_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal_sha256,
        publication_identity=publication_identity,
        phase="select",
        scope_id="global",
        label=label,
    )


def _validate_reveal_global(
    seal: object,
    *,
    expected_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    label: str,
) -> PhaseSeal:
    return _verify_phase_shape(
        seal,
        artifact=REVEAL_CAMPAIGN_ARTIFACT,
        payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_seal_sha256,
        publication_identity=publication_identity,
        phase="reveal",
        scope_id="global",
        label=label,
    )


def _validate_all_globals(
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign_seal: PhaseSeal,
    stage_manifest_seal: PhaseSeal,
    selection_barrier: PhaseSeal,
    reveal_campaign_seal: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    label: str,
) -> None:
    _validate_identity(publication_identity, label=label)
    _validate_protocol(
        protocol_capability,
        publication_identity=publication_identity,
        label=label,
    )
    _validate_prepare_global(
        prepare_campaign_seal,
        expected_seal_sha256=expected_prepare_campaign_seal_sha256,
        publication_identity=publication_identity,
        label=f"{label} prepare-global capability",
    )
    _validate_stage_global(
        stage_manifest_seal,
        expected_seal_sha256=expected_stage_global_seal_sha256,
        label=f"{label} stage-global capability",
    )
    _validate_select_global(
        selection_barrier,
        expected_seal_sha256=expected_selection_barrier_seal_sha256,
        publication_identity=publication_identity,
        label=f"{label} select-global capability",
    )
    _validate_reveal_global(
        reveal_campaign_seal,
        expected_seal_sha256=expected_reveal_campaign_seal_sha256,
        publication_identity=publication_identity,
        label=f"{label} reveal-global capability",
    )


def _stage_leaf_from_document(value: object, *, label: str) -> StageLeafIndex:
    document = _exact_object(value, fields=_STAGE_LEAF_FIELDS, label=label)
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["rotation_id"]) is not str
        or document["role"] != OUTER_METADATA_ROLE
    ):
        raise ValueError(f"{label} identity is invalid")
    spec = rotation_by_id(document["rotation_id"])
    role = OUTER_METADATA_ROLE
    payload_paths = document["payload_paths"]
    row_counts = document["row_counts"]
    id_streams = document["id_streams"]
    if (
        type(document["relative_path"]) is not str
        or document["relative_path"] != leaf_relative_path(spec, role)
        or type(document["leaf_artifact"]) is not str
        or document["leaf_artifact"] != LEAF_ARTIFACTS[role]
        or type(payload_paths) is not list
        or tuple(payload_paths) != LEAF_PAYLOAD_PATHS[role]
        or any(type(path) is not str for path in payload_paths)
        or type(row_counts) is not dict
        or tuple(row_counts) != LEAF_DATA_PAYLOAD_PATHS[role]
        or type(id_streams) is not dict
        or tuple(id_streams) != LEAF_ID_STREAM_NAMES[role]
    ):
        raise ValueError(f"{label} differs from the exact outer-metadata role")
    parsed_counts: list[tuple[str, int]] = []
    for path in LEAF_DATA_PAYLOAD_PATHS[role]:
        count = row_counts[path]
        if type(count) is not int or count <= 0:
            raise ValueError(f"{label} row count must be a positive exact integer")
        parsed_counts.append((path, count))
    parsed_streams: list[tuple[str, IdStreamBinding]] = []
    for name in LEAF_ID_STREAM_NAMES[role]:
        raw = _exact_object(
            id_streams[name],
            fields=frozenset({"count", "sha256"}),
            label=f"{label} ID stream {name}",
        )
        if type(raw["count"]) is not int or raw["count"] <= 0:
            raise ValueError(f"{label} ID stream count must be a positive exact integer")
        parsed_streams.append(
            (
                name,
                IdStreamBinding(
                    count=raw["count"],
                    sha256=_sha256(raw["sha256"], label=f"{label} ID stream {name}"),
                ),
            )
        )
    result = StageLeafIndex(
        spec=spec,
        role=role,
        relative_path=document["relative_path"],
        leaf_artifact=document["leaf_artifact"],
        leaf_seal_sha256=_sha256(document["leaf_seal_sha256"], label=f"{label} leaf seal"),
        payload_paths=tuple(payload_paths),
        row_counts=tuple(parsed_counts),
        id_streams=tuple(parsed_streams),
    )
    if result.document() != document:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


def _capsule_document(capsule: AuthenticatedLeafCapsule) -> dict[str, object]:
    if type(capsule) is not AuthenticatedLeafCapsule:
        raise TypeError("outer-metadata capsule must be exact")
    if type(capsule.entry) is not StageLeafIndex or capsule.entry.role != OUTER_METADATA_ROLE:
        raise ValueError("outer-metadata capsule has the wrong indexed role")
    if type(capsule.source_predecessors) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in capsule.source_predecessors
    ):
        raise TypeError("outer-metadata capsule predecessors must be exact immutable pairs")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": OUTER_METADATA_CAPSULE_WIRE_ARTIFACT,
        "entry": capsule.entry.document(),
        "leaf_capability": phase_seal_document(capsule.seal),
        "source_anchors_sha256": _sha256(
            capsule.source_anchors_sha256,
            label="outer-metadata capsule source anchors",
        ),
        "source_predecessors": dict(capsule.source_predecessors),
    }


def _capsule_phase_document(value: object, *, label: str) -> object:
    document = _exact_object(value, fields=_CAPSULE_FIELDS, label=label)
    return document["leaf_capability"]


def _capsule_from_document(value: object, *, label: str) -> AuthenticatedLeafCapsule:
    document = _exact_object(value, fields=_CAPSULE_FIELDS, label=label)
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != OUTER_METADATA_CAPSULE_WIRE_ARTIFACT
    ):
        raise ValueError(f"{label} identity is invalid")
    predecessors = document["source_predecessors"]
    if type(predecessors) is not dict or any(
        type(path) is not str or type(digest) is not str for path, digest in predecessors.items()
    ):
        raise ValueError(f"{label} predecessors must be one exact string mapping")
    result = AuthenticatedLeafCapsule(
        entry=_stage_leaf_from_document(document["entry"], label=f"{label} entry"),
        seal=phase_seal_from_document(document["leaf_capability"]),
        source_anchors_sha256=_sha256(
            document["source_anchors_sha256"],
            label=f"{label} source anchors",
        ),
        source_predecessors=tuple(
            sorted(
                (
                    path,
                    _sha256(digest, label=f"{label} predecessor {path}"),
                )
                for path, digest in predecessors.items()
            )
        ),
    )
    if _capsule_document(result) != document:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


def _validate_capsule(
    value: object,
    *,
    spec: RotationSpec,
    label: str,
) -> AuthenticatedLeafCapsule:
    if type(value) is not AuthenticatedLeafCapsule:
        raise TypeError(f"{label} requires one exact AuthenticatedLeafCapsule")
    if (
        type(value.entry) is not StageLeafIndex
        or type(value.seal) is not PhaseSeal
        or type(value.source_anchors_sha256) is not str
        or type(value.source_predecessors) is not tuple
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in value.source_predecessors
        )
    ):
        raise TypeError(f"{label} capsule fields must use exact immutable types")
    if value.entry.spec != spec or value.entry.role != OUTER_METADATA_ROLE:
        raise ValueError(f"{label} capsule belongs to another rotation or role")
    reconstructed = AuthenticatedLeafCapsule(
        entry=value.entry,
        seal=value.seal,
        source_anchors_sha256=value.source_anchors_sha256,
        source_predecessors=value.source_predecessors,
    )
    if reconstructed != value:
        raise ValueError(f"{label} capsule changed during strict reconstruction")
    return value


def _strict_state_attestation(value: object, *, label: str) -> UpdateStateAttestation:
    if type(value) is not UpdateStateAttestation:
        raise TypeError(f"{label} must be an exact UpdateStateAttestation")
    reconstructed = update_state_attestation_from_document(value.document())
    if reconstructed.canonical_bytes() != value.canonical_bytes():
        raise ValueError(f"{label} changed during strict reconstruction")
    return value


def _strict_component_attestation(value: object, *, label: str) -> OuterComponentAttestation:
    if type(value) is not OuterComponentAttestation:
        raise TypeError(f"{label} must be an exact OuterComponentAttestation")
    reconstructed = outer_component_attestation_from_document(value.document())
    if reconstructed.canonical_bytes() != value.canonical_bytes():
        raise ValueError(f"{label} changed during strict reconstruction")
    return value


def _strict_projection_attestation(value: object, *, label: str) -> OuterProjectionAttestation:
    if type(value) is not OuterProjectionAttestation:
        raise TypeError(f"{label} must be an exact OuterProjectionAttestation")
    reconstructed = outer_projection_attestation_from_document(value.document())
    if reconstructed.canonical_bytes() != value.canonical_bytes():
        raise ValueError(f"{label} changed during strict reconstruction")
    return value


@dataclass(frozen=True, slots=True)
class UpdateStateWorkerRequest:
    """The only two label-bearing leaves allowed in one state worker."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    base_update_seal: PhaseSeal
    reveal_campaign_seal: PhaseSeal
    reveal_seal: PhaseSeal
    stage_manifest_seal: PhaseSeal
    selection_barrier: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        label = "update-state worker request"
        run = _frozen_run(self.run, label=f"{label} run")
        _validate_all_globals(
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            stage_manifest_seal=self.stage_manifest_seal,
            selection_barrier=self.selection_barrier,
            reveal_campaign_seal=self.reveal_campaign_seal,
            expected_prepare_campaign_seal_sha256=self.expected_prepare_campaign_seal_sha256,
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            label=label,
        )
        _verify_phase_shape(
            self.base_update_seal,
            artifact=BASE_UPDATE_ARTIFACT,
            payload_paths=BASE_UPDATE_PAYLOAD_PATHS,
            publication_identity=self.publication_identity,
            phase="prepare",
            scope_id=run.rotation.rotation_id,
            label=f"{label} base-update capability",
        )
        _verify_phase_shape(
            self.reveal_seal,
            artifact=POOL_REVEAL_ARTIFACT,
            payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
            publication_identity=self.publication_identity,
            phase="reveal",
            scope_id=run.track_id,
            label=f"{label} reveal capability",
        )

    def canonical_bytes(self) -> bytes:
        label = "update-state worker request"
        seals = (
            self.protocol_capability.seal,
            self.prepare_campaign_seal,
            self.base_update_seal,
            self.reveal_campaign_seal,
            self.reveal_seal,
            self.stage_manifest_seal,
            self.selection_barrier,
        )
        _preflight_seals(seals, label=label)
        return _canonical_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": UPDATE_STATE_WORKER_REQUEST_ARTIFACT,
                "track_id": self.run.track_id,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "prepare_campaign_capability": phase_seal_document(self.prepare_campaign_seal),
                "base_update_capability": phase_seal_document(self.base_update_seal),
                "reveal_campaign_capability": phase_seal_document(self.reveal_campaign_seal),
                "reveal_capability": phase_seal_document(self.reveal_seal),
                "stage_manifest_capability": phase_seal_document(self.stage_manifest_seal),
                "selection_barrier_capability": phase_seal_document(self.selection_barrier),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
                "expected_selection_barrier_seal_sha256": (
                    self.expected_selection_barrier_seal_sha256
                ),
                "expected_reveal_campaign_seal_sha256": (self.expected_reveal_campaign_seal_sha256),
            },
            label=label,
        )


_STATE_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "track_id",
        "publication_identity",
        "protocol_capability",
        "prepare_campaign_capability",
        "base_update_capability",
        "reveal_campaign_capability",
        "reveal_capability",
        "stage_manifest_capability",
        "selection_barrier_capability",
        "expected_prepare_campaign_seal_sha256",
        "expected_stage_global_seal_sha256",
        "expected_selection_barrier_seal_sha256",
        "expected_reveal_campaign_seal_sha256",
    }
)


def update_state_worker_request_from_bytes(payload: bytes) -> UpdateStateWorkerRequest:
    label = "update-state worker request"
    document = _request_document(payload, fields=_STATE_REQUEST_FIELDS, label=label)
    _validate_request_identity(document, artifact=UPDATE_STATE_WORKER_REQUEST_ARTIFACT, label=label)
    if type(document["track_id"]) is not str:
        raise ValueError(f"{label} track ID must be exact text")
    phase_fields = (
        "protocol_capability",
        "prepare_campaign_capability",
        "base_update_capability",
        "reveal_campaign_capability",
        "reveal_capability",
        "stage_manifest_capability",
        "selection_barrier_capability",
    )
    _preflight_phase_documents(tuple(document[field] for field in phase_fields), label=label)
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = UpdateStateWorkerRequest(
        run=policy_run_by_track_id(document["track_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        base_update_seal=phase_seal_from_document(document["base_update_capability"]),
        reveal_campaign_seal=phase_seal_from_document(document["reveal_campaign_capability"]),
        reveal_seal=phase_seal_from_document(document["reveal_capability"]),
        stage_manifest_seal=phase_seal_from_document(document["stage_manifest_capability"]),
        selection_barrier=phase_seal_from_document(document["selection_barrier_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        expected_selection_barrier_seal_sha256=document["expected_selection_barrier_seal_sha256"],
        expected_reveal_campaign_seal_sha256=document["expected_reveal_campaign_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class UpdateComponentWorkerRequest:
    """Eleven payload-free state attestations plus one metadata capsule."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    stage_manifest_seal: PhaseSeal
    outer_metadata_capsule: AuthenticatedLeafCapsule
    state_attestations: tuple[UpdateStateAttestation, ...]
    expected_state_leaf_seal_sha256s: tuple[str, ...]
    expected_stage_global_seal_sha256: str

    def __post_init__(self) -> None:
        label = "update-component worker request"
        spec = _frozen_rotation(self.spec, label=f"{label} rotation")
        identity = _validate_identity(self.publication_identity, label=label)
        protocol = _validate_protocol(
            self.protocol_capability,
            publication_identity=identity,
            label=label,
        )
        _validate_stage_global(
            self.stage_manifest_seal,
            expected_seal_sha256=self.expected_stage_global_seal_sha256,
            label=f"{label} stage-global capability",
        )
        _validate_capsule(self.outer_metadata_capsule, spec=spec, label=label)
        expected_runs = policy_runs_for_rotation(spec)
        if (
            type(self.state_attestations) is not tuple
            or len(self.state_attestations) != len(expected_runs)
            or any(type(item) is not UpdateStateAttestation for item in self.state_attestations)
            or tuple(item.run for item in self.state_attestations) != expected_runs
            or type(self.expected_state_leaf_seal_sha256s) is not tuple
            or len(self.expected_state_leaf_seal_sha256s) != len(expected_runs)
        ):
            raise ValueError(f"{label} requires eleven state attestations in frozen order")
        expected = tuple(
            _sha256(value, label=f"{label} expected state leaf {index}")
            for index, value in enumerate(self.expected_state_leaf_seal_sha256s)
        )
        if len(set(expected)) != len(expected):
            raise ValueError(f"{label} expected state leaf digests must be distinct")
        for index, (item, digest) in enumerate(zip(self.state_attestations, expected, strict=True)):
            state = _strict_state_attestation(item, label=f"{label} state {index}")
            if (
                state.publication_identity != identity
                or state.protocol_seal_sha256 != protocol.seal.seal_sha256
                or state.state_leaf_seal_sha256 != digest
            ):
                raise ValueError(f"{label} state attestation differs from controller authority")
        if (
            len({item.prepare_global_seal_sha256 for item in self.state_attestations}) != 1
            or len({item.reveal_global_seal_sha256 for item in self.state_attestations}) != 1
        ):
            raise ValueError(f"{label} states do not share prepare/reveal authority")

    def canonical_bytes(self) -> bytes:
        label = "update-component worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.stage_manifest_seal,
                self.outer_metadata_capsule.seal,
            ),
            label=label,
        )
        return _canonical_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": UPDATE_COMPONENT_WORKER_REQUEST_ARTIFACT,
                "rotation_id": self.spec.rotation_id,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "stage_manifest_capability": phase_seal_document(self.stage_manifest_seal),
                "outer_metadata_capsule": _capsule_document(self.outer_metadata_capsule),
                "update_state_attestations": [item.document() for item in self.state_attestations],
                "expected_state_leaf_seal_sha256s": list(self.expected_state_leaf_seal_sha256s),
                "expected_stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
            },
            label=label,
        )


_COMPONENT_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "rotation_id",
        "publication_identity",
        "protocol_capability",
        "stage_manifest_capability",
        "outer_metadata_capsule",
        "update_state_attestations",
        "expected_state_leaf_seal_sha256s",
        "expected_stage_global_seal_sha256",
    }
)


def update_component_worker_request_from_bytes(payload: bytes) -> UpdateComponentWorkerRequest:
    label = "update-component worker request"
    document = _request_document(payload, fields=_COMPONENT_REQUEST_FIELDS, label=label)
    _validate_request_identity(
        document,
        artifact=UPDATE_COMPONENT_WORKER_REQUEST_ARTIFACT,
        label=label,
    )
    attestations = document["update_state_attestations"]
    expected = document["expected_state_leaf_seal_sha256s"]
    if (
        type(document["rotation_id"]) is not str
        or type(attestations) is not list
        or len(attestations)
        != len(policy_runs_for_rotation(rotation_by_id(document["rotation_id"])))
        or type(expected) is not list
        or len(expected) != len(attestations)
    ):
        raise ValueError(f"{label} census or identity is invalid")
    capsule_phase = _capsule_phase_document(
        document["outer_metadata_capsule"],
        label=f"{label} outer-metadata capsule",
    )
    _preflight_phase_documents(
        (
            document["protocol_capability"],
            document["stage_manifest_capability"],
            capsule_phase,
        ),
        label=label,
    )
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = UpdateComponentWorkerRequest(
        spec=rotation_by_id(document["rotation_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        stage_manifest_seal=phase_seal_from_document(document["stage_manifest_capability"]),
        outer_metadata_capsule=_capsule_from_document(
            document["outer_metadata_capsule"],
            label=f"{label} outer-metadata capsule",
        ),
        state_attestations=tuple(
            update_state_attestation_from_document(item) for item in attestations
        ),
        expected_state_leaf_seal_sha256s=tuple(expected),
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class UpdateProjectionWorkerRequest:
    """One state, one rotation component, and one label-free metadata capsule."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    reveal_campaign_seal: PhaseSeal
    stage_manifest_seal: PhaseSeal
    selection_barrier: PhaseSeal
    state_seal: PhaseSeal
    state_attestation: UpdateStateAttestation
    component_seal: PhaseSeal
    component_attestation: OuterComponentAttestation
    outer_metadata_capsule: AuthenticatedLeafCapsule
    expected_prepare_campaign_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str
    expected_state_leaf_seal_sha256: str
    expected_component_leaf_seal_sha256: str

    def __post_init__(self) -> None:
        label = "update-projection worker request"
        run = _frozen_run(self.run, label=f"{label} run")
        _validate_all_globals(
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            stage_manifest_seal=self.stage_manifest_seal,
            selection_barrier=self.selection_barrier,
            reveal_campaign_seal=self.reveal_campaign_seal,
            expected_prepare_campaign_seal_sha256=self.expected_prepare_campaign_seal_sha256,
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            label=label,
        )
        state_expected = _sha256(
            self.expected_state_leaf_seal_sha256,
            label=f"{label} expected state leaf",
        )
        component_expected = _sha256(
            self.expected_component_leaf_seal_sha256,
            label=f"{label} expected component leaf",
        )
        _verify_phase_shape(
            self.state_seal,
            artifact=UPDATE_STATE_ARTIFACT,
            payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
            expected_seal_sha256=state_expected,
            publication_identity=self.publication_identity,
            phase="update",
            scope_id=run.track_id,
            label=f"{label} state capability",
        )
        _verify_phase_shape(
            self.component_seal,
            artifact=OUTER_COMPONENTS_ARTIFACT,
            payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            expected_seal_sha256=component_expected,
            publication_identity=self.publication_identity,
            phase="update",
            scope_id=run.rotation.rotation_id,
            label=f"{label} component capability",
        )
        state = _strict_state_attestation(
            self.state_attestation,
            label=f"{label} state attestation",
        )
        component = _strict_component_attestation(
            self.component_attestation,
            label=f"{label} component attestation",
        )
        capsule = _validate_capsule(
            self.outer_metadata_capsule,
            spec=run.rotation,
            label=label,
        )
        if (
            state.run != run
            or state.publication_identity != self.publication_identity
            or state.protocol_seal_sha256 != self.protocol_capability.seal.seal_sha256
            or state.prepare_global_seal_sha256 != self.expected_prepare_campaign_seal_sha256
            or state.reveal_global_seal_sha256 != self.expected_reveal_campaign_seal_sha256
            or state.state_leaf_seal_sha256 != state_expected
            or component.spec != run.rotation
            or component.publication_identity != self.publication_identity
            or component.protocol_seal_sha256 != self.protocol_capability.seal.seal_sha256
            or component.stage_global_seal_sha256 != self.expected_stage_global_seal_sha256
            or component.component_leaf_seal_sha256 != component_expected
            or component.outer_metadata_leaf_seal_sha256 != capsule.entry.leaf_seal_sha256
        ):
            raise ValueError(f"{label} inputs differ from controller authority")

    def canonical_bytes(self) -> bytes:
        label = "update-projection worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.prepare_campaign_seal,
                self.reveal_campaign_seal,
                self.stage_manifest_seal,
                self.selection_barrier,
                self.state_seal,
                self.component_seal,
                self.outer_metadata_capsule.seal,
            ),
            label=label,
        )
        return _canonical_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": UPDATE_PROJECTION_WORKER_REQUEST_ARTIFACT,
                "track_id": self.run.track_id,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "prepare_campaign_capability": phase_seal_document(self.prepare_campaign_seal),
                "reveal_campaign_capability": phase_seal_document(self.reveal_campaign_seal),
                "stage_manifest_capability": phase_seal_document(self.stage_manifest_seal),
                "selection_barrier_capability": phase_seal_document(self.selection_barrier),
                "state_capability": phase_seal_document(self.state_seal),
                "state_attestation": self.state_attestation.document(),
                "component_capability": phase_seal_document(self.component_seal),
                "component_attestation": self.component_attestation.document(),
                "outer_metadata_capsule": _capsule_document(self.outer_metadata_capsule),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
                "expected_selection_barrier_seal_sha256": (
                    self.expected_selection_barrier_seal_sha256
                ),
                "expected_reveal_campaign_seal_sha256": (self.expected_reveal_campaign_seal_sha256),
                "expected_state_leaf_seal_sha256": self.expected_state_leaf_seal_sha256,
                "expected_component_leaf_seal_sha256": (self.expected_component_leaf_seal_sha256),
            },
            label=label,
        )


_PROJECTION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "track_id",
        "publication_identity",
        "protocol_capability",
        "prepare_campaign_capability",
        "reveal_campaign_capability",
        "stage_manifest_capability",
        "selection_barrier_capability",
        "state_capability",
        "state_attestation",
        "component_capability",
        "component_attestation",
        "outer_metadata_capsule",
        "expected_prepare_campaign_seal_sha256",
        "expected_stage_global_seal_sha256",
        "expected_selection_barrier_seal_sha256",
        "expected_reveal_campaign_seal_sha256",
        "expected_state_leaf_seal_sha256",
        "expected_component_leaf_seal_sha256",
    }
)


def update_projection_worker_request_from_bytes(payload: bytes) -> UpdateProjectionWorkerRequest:
    label = "update-projection worker request"
    document = _request_document(payload, fields=_PROJECTION_REQUEST_FIELDS, label=label)
    _validate_request_identity(
        document,
        artifact=UPDATE_PROJECTION_WORKER_REQUEST_ARTIFACT,
        label=label,
    )
    if type(document["track_id"]) is not str:
        raise ValueError(f"{label} track ID must be exact text")
    capsule_phase = _capsule_phase_document(
        document["outer_metadata_capsule"],
        label=f"{label} outer-metadata capsule",
    )
    phase_fields = (
        "protocol_capability",
        "prepare_campaign_capability",
        "reveal_campaign_capability",
        "stage_manifest_capability",
        "selection_barrier_capability",
        "state_capability",
        "component_capability",
    )
    _preflight_phase_documents(
        (*tuple(document[field] for field in phase_fields), capsule_phase),
        label=label,
    )
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = UpdateProjectionWorkerRequest(
        run=policy_run_by_track_id(document["track_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        reveal_campaign_seal=phase_seal_from_document(document["reveal_campaign_capability"]),
        stage_manifest_seal=phase_seal_from_document(document["stage_manifest_capability"]),
        selection_barrier=phase_seal_from_document(document["selection_barrier_capability"]),
        state_seal=phase_seal_from_document(document["state_capability"]),
        state_attestation=update_state_attestation_from_document(document["state_attestation"]),
        component_seal=phase_seal_from_document(document["component_capability"]),
        component_attestation=outer_component_attestation_from_document(
            document["component_attestation"]
        ),
        outer_metadata_capsule=_capsule_from_document(
            document["outer_metadata_capsule"],
            label=f"{label} outer-metadata capsule",
        ),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        expected_selection_barrier_seal_sha256=document["expected_selection_barrier_seal_sha256"],
        expected_reveal_campaign_seal_sha256=document["expected_reveal_campaign_seal_sha256"],
        expected_state_leaf_seal_sha256=document["expected_state_leaf_seal_sha256"],
        expected_component_leaf_seal_sha256=document["expected_component_leaf_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class UpdateBarrierWorkerRequest:
    """Only payload-free UPDATE attestations and controller digest snapshots."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign_seal: PhaseSeal
    reveal_campaign_seal: PhaseSeal
    stage_manifest_seal: PhaseSeal
    selection_barrier: PhaseSeal
    state_attestations: tuple[UpdateStateAttestation, ...]
    component_attestations: tuple[OuterComponentAttestation, ...]
    projection_attestations: tuple[OuterProjectionAttestation, ...]
    expected_update_leaf_seal_sha256s: tuple[str, ...]
    expected_prepare_campaign_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        label = "update-barrier worker request"
        _validate_all_globals(
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign_seal,
            stage_manifest_seal=self.stage_manifest_seal,
            selection_barrier=self.selection_barrier,
            reveal_campaign_seal=self.reveal_campaign_seal,
            expected_prepare_campaign_seal_sha256=self.expected_prepare_campaign_seal_sha256,
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            label=label,
        )
        runs = ordered_policy_runs()
        rotations = ordered_rotations()
        if (
            type(self.state_attestations) is not tuple
            or len(self.state_attestations) != EXPECTED_POLICY_RUNS
            or any(type(item) is not UpdateStateAttestation for item in self.state_attestations)
            or tuple(item.run for item in self.state_attestations) != runs
            or type(self.component_attestations) is not tuple
            or len(self.component_attestations) != EXPECTED_ROTATIONS
            or any(
                type(item) is not OuterComponentAttestation for item in self.component_attestations
            )
            or tuple(item.spec for item in self.component_attestations) != rotations
            or type(self.projection_attestations) is not tuple
            or len(self.projection_attestations) != EXPECTED_POLICY_RUNS
            or any(
                type(item) is not OuterProjectionAttestation
                for item in self.projection_attestations
            )
            or tuple(item.run for item in self.projection_attestations) != runs
        ):
            raise ValueError(f"{label} attestation census or frozen order changed")
        if (
            type(self.expected_update_leaf_seal_sha256s) is not tuple
            or len(self.expected_update_leaf_seal_sha256s) != EXPECTED_UPDATE_LEAVES
        ):
            raise ValueError(f"{label} requires exactly 680 ordered leaf digests")
        expected = tuple(
            _sha256(value, label=f"{label} expected leaf {index}")
            for index, value in enumerate(self.expected_update_leaf_seal_sha256s)
        )
        if len(set(expected)) != EXPECTED_UPDATE_LEAVES:
            raise ValueError(f"{label} expected leaf digests must be entirely distinct")
        state_expected = expected[:EXPECTED_POLICY_RUNS]
        component_stop = EXPECTED_POLICY_RUNS + EXPECTED_ROTATIONS
        component_expected = expected[EXPECTED_POLICY_RUNS:component_stop]
        view_stop = component_stop + EXPECTED_POLICY_RUNS
        view_expected = expected[component_stop:view_stop]
        evidence_expected = expected[view_stop:]
        states: dict[PolicyRunSpec, UpdateStateAttestation] = {}
        for index, (item, digest) in enumerate(
            zip(self.state_attestations, state_expected, strict=True)
        ):
            state = _strict_state_attestation(item, label=f"{label} state {index}")
            if (
                state.publication_identity != self.publication_identity
                or state.protocol_seal_sha256 != self.protocol_capability.seal.seal_sha256
                or state.prepare_global_seal_sha256 != self.expected_prepare_campaign_seal_sha256
                or state.reveal_global_seal_sha256 != self.expected_reveal_campaign_seal_sha256
                or state.state_leaf_seal_sha256 != digest
            ):
                raise ValueError(f"{label} state differs from controller/global authority")
            states[state.run] = state
        components: dict[RotationSpec, OuterComponentAttestation] = {}
        for index, (item, digest) in enumerate(
            zip(self.component_attestations, component_expected, strict=True)
        ):
            component = _strict_component_attestation(
                item,
                label=f"{label} component {index}",
            )
            expected_states = tuple(
                states[run].state_leaf_seal_sha256
                for run in policy_runs_for_rotation(component.spec)
            )
            if (
                component.publication_identity != self.publication_identity
                or component.protocol_seal_sha256 != self.protocol_capability.seal.seal_sha256
                or component.stage_global_seal_sha256 != self.expected_stage_global_seal_sha256
                or component.state_leaf_seal_sha256s != expected_states
                or component.component_leaf_seal_sha256 != digest
            ):
                raise ValueError(f"{label} component differs from controller/global authority")
            components[component.spec] = component
        actual_views: list[str] = []
        actual_evidence: list[str] = []
        for index, item in enumerate(self.projection_attestations):
            projection = _strict_projection_attestation(
                item,
                label=f"{label} projection {index}",
            )
            state = states[projection.run]
            component = components[projection.run.rotation]
            if (
                projection.publication_identity != self.publication_identity
                or projection.protocol_seal_sha256 != self.protocol_capability.seal.seal_sha256
                or projection.stage_global_seal_sha256 != self.expected_stage_global_seal_sha256
                or projection.state_leaf_seal_sha256 != state.state_leaf_seal_sha256
                or projection.updated_model_payload_sha256
                != dict(state.payload_sha256)["updated-model.json"]
                or projection.outer_component_leaf_seal_sha256
                != component.component_leaf_seal_sha256
                or projection.outer_components_payload_sha256
                != dict(component.payload_sha256)["outer-components.jsonl"]
                or projection.outer_metadata_leaf_seal_sha256
                != component.outer_metadata_leaf_seal_sha256
                or projection.outer_view_leaf_seal_sha256 != view_expected[index]
                or projection.outer_evidence_leaf_seal_sha256 != evidence_expected[index]
            ):
                raise ValueError(f"{label} projection differs from controller/global authority")
            actual_views.append(projection.outer_view_leaf_seal_sha256)
            actual_evidence.append(projection.outer_evidence_leaf_seal_sha256)
        actual = (
            *(item.state_leaf_seal_sha256 for item in self.state_attestations),
            *(item.component_leaf_seal_sha256 for item in self.component_attestations),
            *actual_views,
            *actual_evidence,
        )
        if actual != expected:
            raise ValueError(f"{label} leaves differ from controller digest order")

    def canonical_bytes(self) -> bytes:
        label = "update-barrier worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.prepare_campaign_seal,
                self.reveal_campaign_seal,
                self.stage_manifest_seal,
                self.selection_barrier,
            ),
            label=label,
        )
        return _canonical_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": UPDATE_BARRIER_WORKER_REQUEST_ARTIFACT,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "prepare_campaign_capability": phase_seal_document(self.prepare_campaign_seal),
                "reveal_campaign_capability": phase_seal_document(self.reveal_campaign_seal),
                "stage_manifest_capability": phase_seal_document(self.stage_manifest_seal),
                "selection_barrier_capability": phase_seal_document(self.selection_barrier),
                "update_state_attestations": [item.document() for item in self.state_attestations],
                "outer_component_attestations": [
                    item.document() for item in self.component_attestations
                ],
                "outer_projection_attestations": [
                    item.document() for item in self.projection_attestations
                ],
                "expected_update_leaf_seal_sha256s": list(self.expected_update_leaf_seal_sha256s),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
                "expected_selection_barrier_seal_sha256": (
                    self.expected_selection_barrier_seal_sha256
                ),
                "expected_reveal_campaign_seal_sha256": (self.expected_reveal_campaign_seal_sha256),
            },
            label=label,
        )


_BARRIER_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "publication_identity",
        "protocol_capability",
        "prepare_campaign_capability",
        "reveal_campaign_capability",
        "stage_manifest_capability",
        "selection_barrier_capability",
        "update_state_attestations",
        "outer_component_attestations",
        "outer_projection_attestations",
        "expected_update_leaf_seal_sha256s",
        "expected_prepare_campaign_seal_sha256",
        "expected_stage_global_seal_sha256",
        "expected_selection_barrier_seal_sha256",
        "expected_reveal_campaign_seal_sha256",
    }
)


def update_barrier_worker_request_from_bytes(payload: bytes) -> UpdateBarrierWorkerRequest:
    label = "update-barrier worker request"
    document = _request_document(payload, fields=_BARRIER_REQUEST_FIELDS, label=label)
    _validate_request_identity(
        document,
        artifact=UPDATE_BARRIER_WORKER_REQUEST_ARTIFACT,
        label=label,
    )
    states = document["update_state_attestations"]
    components = document["outer_component_attestations"]
    projections = document["outer_projection_attestations"]
    expected = document["expected_update_leaf_seal_sha256s"]
    if (
        type(states) is not list
        or len(states) != EXPECTED_POLICY_RUNS
        or type(components) is not list
        or len(components) != EXPECTED_ROTATIONS
        or type(projections) is not list
        or len(projections) != EXPECTED_POLICY_RUNS
        or type(expected) is not list
        or len(expected) != EXPECTED_UPDATE_LEAVES
    ):
        raise ValueError(f"{label} census is invalid")
    phase_fields = (
        "protocol_capability",
        "prepare_campaign_capability",
        "reveal_campaign_capability",
        "stage_manifest_capability",
        "selection_barrier_capability",
    )
    _preflight_phase_documents(tuple(document[field] for field in phase_fields), label=label)
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = UpdateBarrierWorkerRequest(
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        prepare_campaign_seal=phase_seal_from_document(document["prepare_campaign_capability"]),
        reveal_campaign_seal=phase_seal_from_document(document["reveal_campaign_capability"]),
        stage_manifest_seal=phase_seal_from_document(document["stage_manifest_capability"]),
        selection_barrier=phase_seal_from_document(document["selection_barrier_capability"]),
        state_attestations=tuple(update_state_attestation_from_document(item) for item in states),
        component_attestations=tuple(
            outer_component_attestation_from_document(item) for item in components
        ),
        projection_attestations=tuple(
            outer_projection_attestation_from_document(item) for item in projections
        ),
        expected_update_leaf_seal_sha256s=tuple(expected),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        expected_selection_barrier_seal_sha256=document["expected_selection_barrier_seal_sha256"],
        expected_reveal_campaign_seal_sha256=document["expected_reveal_campaign_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


__all__ = [
    "EXPECTED_UPDATE_LEAVES",
    "OUTER_METADATA_CAPSULE_WIRE_ARTIFACT",
    "UPDATE_BARRIER_WORKER_REQUEST_ARTIFACT",
    "UPDATE_BARRIER_WORKER_ROLE",
    "UPDATE_COMPONENT_WORKER_REQUEST_ARTIFACT",
    "UPDATE_COMPONENT_WORKER_ROLE",
    "UPDATE_PROJECTION_WORKER_REQUEST_ARTIFACT",
    "UPDATE_PROJECTION_WORKER_ROLE",
    "UPDATE_STATE_WORKER_REQUEST_ARTIFACT",
    "UPDATE_STATE_WORKER_ROLE",
    "UpdateBarrierWorkerRequest",
    "UpdateComponentWorkerRequest",
    "UpdateProjectionWorkerRequest",
    "UpdateStateWorkerRequest",
    "update_barrier_worker_request_from_bytes",
    "update_component_worker_request_from_bytes",
    "update_projection_worker_request_from_bytes",
    "update_state_worker_request_from_bytes",
]

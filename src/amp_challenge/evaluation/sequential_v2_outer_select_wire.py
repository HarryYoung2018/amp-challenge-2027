"""Canonical pathless requests for isolated sequential-v2 OUTER-SELECT workers.

The track request carries the one label-free outer-view capability needed for
one frozen policy run.  The barrier request deliberately replaces all leaf and
view capabilities with canonical payload-free attestations and a separately
observed physical-marker digest sequence.  Neither request can carry a source
path, stage capability, outcome vault, evidence leaf, model, or sibling view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from amp_challenge.evaluation import sequential_v2_wire as _core_wire
from amp_challenge.evaluation.sequential_v2_outer_select import (
    OuterSelectionAttestation,
    _snapshot_update_outer_view_rows,
    outer_selection_attestation_from_document,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    PROTOCOL_ARTIFACT,
    PROTOCOL_PAYLOAD_PATHS,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_POLICY_RUNS,
    PolicyRunSpec,
    ordered_policy_runs,
    policy_run_by_track_id,
)
from amp_challenge.evaluation.sequential_v2_seals import PhaseSeal, canonical_json_bytes
from amp_challenge.evaluation.sequential_v2_update_campaign import (
    UpdateCampaignCapability,
    outer_view_from_campaign,
    verify_update_campaign_capability,
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

OUTER_SELECT_TRACK_WORKER_REQUEST_ARTIFACT = "sequential_v2_outer_select_track_worker_request_v1"
OUTER_SELECT_BARRIER_WORKER_REQUEST_ARTIFACT = (
    "sequential_v2_outer_select_barrier_worker_request_v1"
)
OUTER_SELECT_TRACK_WORKER_ROLE = "outer-select-track"
OUTER_SELECT_BARRIER_WORKER_ROLE = "outer-select-barrier"

_MAX_WIRE_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_REQUEST_CAPTURED_BYTES = 128 * 1024 * 1024
_MAX_REQUEST_BASE64_CHARACTERS = 192 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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


def _frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    canonical = policy_run_by_track_id(value.track_id)
    if canonical != value:
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _identity(
    value: object,
    *,
    label: str,
) -> SequentialV2PublicationIdentity:
    if type(value) is not SequentialV2PublicationIdentity:
        raise TypeError(f"{label} requires an exact publication identity")
    publication_identity_document(value)
    return value


def _protocol(
    value: object,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    label: str,
) -> ProtocolCapability:
    if type(value) is not ProtocolCapability or type(value.seal) is not PhaseSeal:
        raise TypeError(f"{label} requires an exact protocol capability")
    verified = _core_wire.verify_phase_capability(
        value.seal,
        expected_artifact=PROTOCOL_ARTIFACT,
        expected_payload_paths=PROTOCOL_PAYLOAD_PATHS,
        expected_seal_sha256=value.seal.seal_sha256,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="protocol",
        scope_id="global",
    )
    return verify_protocol_capability(
        verified,
        publication_identity=publication_identity,
    )


def _external_digests(
    *,
    expected_stage_global_seal_sha256: object,
    expected_prepare_campaign_seal_sha256: object,
    expected_reveal_campaign_seal_sha256: object,
    expected_update_campaign_seal_sha256: object,
    label: str,
) -> tuple[str, str, str, str]:
    return (
        _sha256(
            expected_stage_global_seal_sha256,
            label=f"{label} expected stage-global seal",
        ),
        _sha256(
            expected_prepare_campaign_seal_sha256,
            label=f"{label} expected prepare-global seal",
        ),
        _sha256(
            expected_reveal_campaign_seal_sha256,
            label=f"{label} expected reveal-global seal",
        ),
        _sha256(
            expected_update_campaign_seal_sha256,
            label=f"{label} expected update-global seal",
        ),
    )


def _validated_campaign(
    seal: object,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    label: str,
) -> UpdateCampaignCapability:
    if type(seal) is not PhaseSeal:
        raise TypeError(f"{label} update campaign must be an exact PhaseSeal")
    if seal.seal_sha256 != expected_update_campaign_seal_sha256:
        raise ValueError(f"{label} update campaign differs from external authority")
    return verify_update_campaign_capability(
        UpdateCampaignCapability(seal, publication_identity),
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )


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
        or type(document["artifact"]) is not str
        or document["artifact"] != artifact
    ):
        raise ValueError(f"{label} identity is invalid")


@dataclass(frozen=True, slots=True)
class OuterSelectTrackWorkerRequest:
    """One track's sole label-free outer-view selection authority."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    update_campaign_seal: PhaseSeal
    outer_view_seal: PhaseSeal
    expected_stage_global_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str
    expected_update_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        label = "outer-select track worker request"
        run = _frozen_run(self.run, label=f"{label} run")
        identity = _identity(self.publication_identity, label=label)
        protocol = _protocol(
            self.protocol_capability,
            publication_identity=identity,
            label=label,
        )
        stage, prepare, reveal, update = _external_digests(
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=self.expected_update_campaign_seal_sha256,
            label=label,
        )
        campaign = _validated_campaign(
            self.update_campaign_seal,
            publication_identity=identity,
            protocol_capability=protocol,
            expected_stage_global_seal_sha256=stage,
            expected_prepare_campaign_seal_sha256=prepare,
            expected_reveal_campaign_seal_sha256=reveal,
            expected_update_campaign_seal_sha256=update,
            label=label,
        )
        if type(self.outer_view_seal) is not PhaseSeal:
            raise TypeError(f"{label} outer view must be an exact PhaseSeal")
        outer_view_from_campaign(
            campaign,
            run=run,
            outer_view_seal=self.outer_view_seal,
            publication_identity=identity,
            protocol_capability=protocol,
            expected_stage_global_seal_sha256=stage,
            expected_prepare_campaign_seal_sha256=prepare,
            expected_reveal_campaign_seal_sha256=reveal,
            expected_update_campaign_seal_sha256=update,
        )

    def canonical_bytes(self) -> bytes:
        label = "outer-select track worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.update_campaign_seal,
                self.outer_view_seal,
            ),
            label=label,
        )
        return _canonical_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": OUTER_SELECT_TRACK_WORKER_REQUEST_ARTIFACT,
                "track_id": self.run.track_id,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "update_campaign_capability": phase_seal_document(self.update_campaign_seal),
                "outer_view_capability": phase_seal_document(self.outer_view_seal),
                "expected_stage_global_seal_sha256": (self.expected_stage_global_seal_sha256),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_reveal_campaign_seal_sha256": (self.expected_reveal_campaign_seal_sha256),
                "expected_update_campaign_seal_sha256": (self.expected_update_campaign_seal_sha256),
            },
            label=label,
        )


_TRACK_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "track_id",
        "publication_identity",
        "protocol_capability",
        "update_campaign_capability",
        "outer_view_capability",
        "expected_stage_global_seal_sha256",
        "expected_prepare_campaign_seal_sha256",
        "expected_reveal_campaign_seal_sha256",
        "expected_update_campaign_seal_sha256",
    }
)


def outer_select_track_worker_request_from_bytes(payload: bytes) -> OuterSelectTrackWorkerRequest:
    label = "outer-select track worker request"
    document = _request_document(payload, fields=_TRACK_REQUEST_FIELDS, label=label)
    _validate_request_identity(
        document,
        artifact=OUTER_SELECT_TRACK_WORKER_REQUEST_ARTIFACT,
        label=label,
    )
    if type(document["track_id"]) is not str:
        raise ValueError(f"{label} track ID must be exact text")
    phase_fields = (
        "protocol_capability",
        "update_campaign_capability",
        "outer_view_capability",
    )
    _preflight_phase_documents(tuple(document[field] for field in phase_fields), label=label)
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = OuterSelectTrackWorkerRequest(
        run=policy_run_by_track_id(document["track_id"]),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        update_campaign_seal=phase_seal_from_document(document["update_campaign_capability"]),
        outer_view_seal=phase_seal_from_document(document["outer_view_capability"]),
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        expected_prepare_campaign_seal_sha256=(document["expected_prepare_campaign_seal_sha256"]),
        expected_reveal_campaign_seal_sha256=document["expected_reveal_campaign_seal_sha256"],
        expected_update_campaign_seal_sha256=document["expected_update_campaign_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class OuterSelectBarrierWorkerRequest:
    """Payload-free 220-track authority for the outer-select global barrier."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    update_campaign_seal: PhaseSeal
    selection_attestations: tuple[OuterSelectionAttestation, ...]
    expected_outer_selection_leaf_seal_sha256s: tuple[str, ...]
    expected_stage_global_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str
    expected_update_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        label = "outer-select barrier worker request"
        identity = _identity(self.publication_identity, label=label)
        protocol = _protocol(
            self.protocol_capability,
            publication_identity=identity,
            label=label,
        )
        stage, prepare, reveal, update = _external_digests(
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=self.expected_update_campaign_seal_sha256,
            label=label,
        )
        campaign = _validated_campaign(
            self.update_campaign_seal,
            publication_identity=identity,
            protocol_capability=protocol,
            expected_stage_global_seal_sha256=stage,
            expected_prepare_campaign_seal_sha256=prepare,
            expected_reveal_campaign_seal_sha256=reveal,
            expected_update_campaign_seal_sha256=update,
            label=label,
        )
        runs = ordered_policy_runs()
        update_outer_view_rows = _snapshot_update_outer_view_rows(campaign)
        if (
            type(self.selection_attestations) is not tuple
            or len(self.selection_attestations) != EXPECTED_POLICY_RUNS
            or any(
                type(item) is not OuterSelectionAttestation for item in self.selection_attestations
            )
            or tuple(item.run for item in self.selection_attestations) != runs
        ):
            raise ValueError(f"{label} attestation census or frozen order changed")
        if (
            type(self.expected_outer_selection_leaf_seal_sha256s) is not tuple
            or len(self.expected_outer_selection_leaf_seal_sha256s) != EXPECTED_POLICY_RUNS
        ):
            raise ValueError(f"{label} requires exactly 220 ordered leaf digests")
        expected = tuple(
            _sha256(value, label=f"{label} expected leaf {index}")
            for index, value in enumerate(self.expected_outer_selection_leaf_seal_sha256s)
        )
        if len(set(expected)) != EXPECTED_POLICY_RUNS:
            raise ValueError(f"{label} expected leaf digests must be entirely distinct")
        for index, (attestation, expected_leaf) in enumerate(
            zip(self.selection_attestations, expected, strict=True)
        ):
            exact = outer_selection_attestation_from_document(attestation.document())
            row = update_outer_view_rows[index]
            if (
                exact != attestation
                or exact.publication_identity != identity
                or exact.protocol_seal_sha256 != protocol.seal.seal_sha256
                or exact.update_global_seal_sha256 != update
                or exact.outer_view_leaf_seal_sha256 != row.leaf_seal_sha256
                or exact.outer_view_payload_sha256 != row.payload_sha256
                or exact.outer_candidate_count != row.candidate_count
                or exact.outer_candidate_ids_sha256 != row.candidate_ids_sha256
                or exact.selection_leaf_seal_sha256 != expected_leaf
                or exact.selected_sequence_count != runs[index].expected_outer_selection_count
            ):
                raise ValueError(
                    f"{label} attestation {index} differs from controller/global authority"
                )

    def canonical_bytes(self) -> bytes:
        label = "outer-select barrier worker request"
        _preflight_seals(
            (self.protocol_capability.seal, self.update_campaign_seal),
            label=label,
        )
        return _canonical_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": OUTER_SELECT_BARRIER_WORKER_REQUEST_ARTIFACT,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "update_campaign_capability": phase_seal_document(self.update_campaign_seal),
                "outer_selection_attestations": [
                    item.document() for item in self.selection_attestations
                ],
                "expected_outer_selection_leaf_seal_sha256s": list(
                    self.expected_outer_selection_leaf_seal_sha256s
                ),
                "expected_stage_global_seal_sha256": (self.expected_stage_global_seal_sha256),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_reveal_campaign_seal_sha256": (self.expected_reveal_campaign_seal_sha256),
                "expected_update_campaign_seal_sha256": (self.expected_update_campaign_seal_sha256),
            },
            label=label,
        )


_BARRIER_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "publication_identity",
        "protocol_capability",
        "update_campaign_capability",
        "outer_selection_attestations",
        "expected_outer_selection_leaf_seal_sha256s",
        "expected_stage_global_seal_sha256",
        "expected_prepare_campaign_seal_sha256",
        "expected_reveal_campaign_seal_sha256",
        "expected_update_campaign_seal_sha256",
    }
)


def outer_select_barrier_worker_request_from_bytes(
    payload: bytes,
) -> OuterSelectBarrierWorkerRequest:
    label = "outer-select barrier worker request"
    document = _request_document(payload, fields=_BARRIER_REQUEST_FIELDS, label=label)
    _validate_request_identity(
        document,
        artifact=OUTER_SELECT_BARRIER_WORKER_REQUEST_ARTIFACT,
        label=label,
    )
    attestations = document["outer_selection_attestations"]
    expected = document["expected_outer_selection_leaf_seal_sha256s"]
    if (
        type(attestations) is not list
        or len(attestations) != EXPECTED_POLICY_RUNS
        or type(expected) is not list
        or len(expected) != EXPECTED_POLICY_RUNS
    ):
        raise ValueError(f"{label} census is invalid")
    _preflight_phase_documents(
        (
            document["protocol_capability"],
            document["update_campaign_capability"],
        ),
        label=label,
    )
    protocol_seal = phase_seal_from_document(document["protocol_capability"])
    result = OuterSelectBarrierWorkerRequest(
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(protocol_seal),
        update_campaign_seal=phase_seal_from_document(document["update_campaign_capability"]),
        selection_attestations=tuple(
            outer_selection_attestation_from_document(item) for item in attestations
        ),
        expected_outer_selection_leaf_seal_sha256s=tuple(expected),
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        expected_prepare_campaign_seal_sha256=(document["expected_prepare_campaign_seal_sha256"]),
        expected_reveal_campaign_seal_sha256=document["expected_reveal_campaign_seal_sha256"],
        expected_update_campaign_seal_sha256=document["expected_update_campaign_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


__all__ = [
    "OUTER_SELECT_BARRIER_WORKER_REQUEST_ARTIFACT",
    "OUTER_SELECT_BARRIER_WORKER_ROLE",
    "OUTER_SELECT_TRACK_WORKER_REQUEST_ARTIFACT",
    "OUTER_SELECT_TRACK_WORKER_ROLE",
    "OuterSelectBarrierWorkerRequest",
    "OuterSelectTrackWorkerRequest",
    "outer_select_barrier_worker_request_from_bytes",
    "outer_select_track_worker_request_from_bytes",
]

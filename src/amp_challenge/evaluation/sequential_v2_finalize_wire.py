"""Canonical pathless request for the isolated sequential-v2 finalizer.

The request carries every global exactly once and only the indexed leaves
needed to compute the frozen metrics.  Construction and decoding authenticate
all label-free campaign barriers and exact leaf order without semantically
opening an outer-outcome payload.  The finalizer must still rederive every
outer selection before it materializes those outcome leaves.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from amp_challenge.evaluation import sequential_v2_wire as _core_wire
from amp_challenge.evaluation.sequential_v2_commitments import (
    verify_pool_commitment_campaign_barrier_capability,
)
from amp_challenge.evaluation.sequential_v2_outer_select import (
    OuterSelectionCampaignCapability,
    verify_outer_selection_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    PREDICTION_VIEW_ROLE,
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_POLICY_RUNS,
    EXPECTED_ROTATIONS,
    ordered_policy_runs,
    ordered_rotations,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    RevealCampaignCapability,
    verify_reveal_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    OUTER_OUTCOME_ROLE,
    StageManifestCapability,
    verify_stage_manifest_capability,
)
from amp_challenge.evaluation.sequential_v2_update_campaign import (
    UpdateCampaignCapability,
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
FINALIZE_WORKER_ROLE = "finalize"
FINALIZE_WORKER_REQUEST_ARTIFACT = "sequential_v2_finalize_worker_request_v1"
FINALIZE_REQUEST_CAPACITY_MEASUREMENT_ARTIFACT = (
    "sequential_v2_finalize_request_capacity_measurement_v1"
)

MAX_REQUEST_CAPTURED_BYTES = 128 * 1024 * 1024
MAX_REQUEST_BASE64_CHARACTERS = 192 * 1024 * 1024
MAX_WIRE_REQUEST_BYTES = 256 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GLOBAL_FIELDS = (
    "protocol_capability",
    "stage_manifest_capability",
    "prepare_campaign_capability",
    "selection_campaign_capability",
    "reveal_campaign_capability",
    "update_campaign_capability",
    "outer_selection_campaign_capability",
)
_ROTATION_LEAF_FIELDS = (
    "outer_outcome_vault_capabilities",
    "prediction_view_capabilities",
)
_TRACK_LEAF_FIELDS = (
    "reveal_leaf_capabilities",
    "outer_evidence_capabilities",
    "outer_view_capabilities",
    "outer_selection_leaf_capabilities",
)
FINALIZE_REQUEST_CAPABILITY_COUNT = (
    len(_GLOBAL_FIELDS)
    + EXPECTED_ROTATIONS * len(_ROTATION_LEAF_FIELDS)
    + EXPECTED_POLICY_RUNS * len(_TRACK_LEAF_FIELDS)
)
FINALIZE_REQUEST_PREDECESSOR_COUNT = FINALIZE_REQUEST_CAPABILITY_COUNT
_EXPECTED_DIGEST_FIELDS = (
    "expected_protocol_seal_sha256",
    "expected_stage_global_seal_sha256",
    "expected_prepare_campaign_seal_sha256",
    "expected_selection_campaign_seal_sha256",
    "expected_reveal_campaign_seal_sha256",
    "expected_update_campaign_seal_sha256",
    "expected_outer_selection_campaign_seal_sha256",
)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "publication_identity",
        *_GLOBAL_FIELDS,
        *_ROTATION_LEAF_FIELDS,
        *_TRACK_LEAF_FIELDS,
        *_EXPECTED_DIGEST_FIELDS,
    }
)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _exact_seals(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[PhaseSeal, ...]:
    if (
        type(value) is not tuple
        or len(value) != count
        or any(type(item) is not PhaseSeal for item in value)
    ):
        raise TypeError(f"{label} must contain exactly {count} PhaseSeal values")
    seals = value
    if len({seal.seal_sha256 for seal in seals}) != count:
        raise ValueError(f"{label} leaf seals must be distinct")
    return seals


def _require_identity(
    value: object,
) -> SequentialV2PublicationIdentity:
    if type(value) is not SequentialV2PublicationIdentity:
        raise TypeError("finalize request requires an exact publication identity")
    publication_identity_document(value)
    return value


@dataclass(frozen=True, slots=True)
class _AuthenticatedFinalizeGlobals:
    protocol: ProtocolCapability
    stage: StageManifestCapability
    prepare: PrepareCampaignCapability
    reveal: RevealCampaignCapability
    update: UpdateCampaignCapability
    outer_selection: OuterSelectionCampaignCapability


def _authenticate_globals(request: FinalizeWorkerRequest) -> _AuthenticatedFinalizeGlobals:
    identity = _require_identity(request.publication_identity)
    protocol_digest = _sha256(
        request.expected_protocol_seal_sha256,
        label="expected finalize protocol seal",
    )
    if (
        type(request.protocol_capability) is not ProtocolCapability
        or request.protocol_capability.seal.seal_sha256 != protocol_digest
    ):
        raise ValueError("finalize protocol capability differs from external authority")
    protocol = verify_protocol_capability(
        request.protocol_capability.seal,
        publication_identity=identity,
    )

    stage_digest = _sha256(
        request.expected_stage_global_seal_sha256,
        label="expected finalize stage-global seal",
    )
    if type(request.stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("finalize request requires an exact stage manifest capability")
    stage = verify_stage_manifest_capability(
        request.stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_digest,
    )

    prepare_digest = _sha256(
        request.expected_prepare_campaign_seal_sha256,
        label="expected finalize prepare-global seal",
    )
    if type(request.prepare_campaign) is not PrepareCampaignCapability:
        raise TypeError("finalize request requires an exact prepare campaign capability")
    prepare = verify_prepare_campaign_capability(
        request.prepare_campaign,
        publication_identity=identity,
        expected_campaign_seal_sha256=prepare_digest,
        expected_protocol_seal_sha256=protocol_digest,
    )

    selection_digest = _sha256(
        request.expected_selection_campaign_seal_sha256,
        label="expected finalize selection-global seal",
    )
    if type(request.selection_campaign_seal) is not PhaseSeal:
        raise TypeError("finalize request requires an exact selection campaign seal")
    verify_pool_commitment_campaign_barrier_capability(
        request.selection_campaign_seal,
        protocol_capability=protocol,
        prepare_campaign=prepare,
        expected_prepare_campaign_seal_sha256=prepare_digest,
        publication_identity=identity,
        expected_seal_sha256=selection_digest,
    )

    reveal_digest = _sha256(
        request.expected_reveal_campaign_seal_sha256,
        label="expected finalize reveal-global seal",
    )
    if type(request.reveal_campaign) is not RevealCampaignCapability:
        raise TypeError("finalize request requires an exact reveal campaign capability")
    reveal = verify_reveal_campaign_capability(
        request.reveal_campaign,
        publication_identity=identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage,
        selection_barrier=request.selection_campaign_seal,
        expected_prepare_campaign_seal_sha256=prepare_digest,
        expected_stage_global_seal_sha256=stage_digest,
        expected_selection_barrier_seal_sha256=selection_digest,
        expected_reveal_campaign_seal_sha256=reveal_digest,
    )

    update_digest = _sha256(
        request.expected_update_campaign_seal_sha256,
        label="expected finalize update-global seal",
    )
    if type(request.update_campaign) is not UpdateCampaignCapability:
        raise TypeError("finalize request requires an exact update campaign capability")
    update = verify_update_campaign_capability(
        request.update_campaign,
        publication_identity=identity,
        protocol_capability=protocol,
        expected_stage_global_seal_sha256=stage_digest,
        expected_prepare_campaign_seal_sha256=prepare_digest,
        expected_reveal_campaign_seal_sha256=reveal_digest,
        expected_update_campaign_seal_sha256=update_digest,
    )

    outer_digest = _sha256(
        request.expected_outer_selection_campaign_seal_sha256,
        label="expected finalize outer-selection-global seal",
    )
    if type(request.outer_selection_campaign) is not OuterSelectionCampaignCapability:
        raise TypeError("finalize request requires an exact outer-selection campaign capability")
    outer_selection = verify_outer_selection_campaign_capability(
        request.outer_selection_campaign,
        publication_identity=identity,
        protocol_capability=protocol,
        update_campaign=update,
        expected_stage_global_seal_sha256=stage_digest,
        expected_prepare_campaign_seal_sha256=prepare_digest,
        expected_reveal_campaign_seal_sha256=reveal_digest,
        expected_update_campaign_seal_sha256=update_digest,
        expected_outer_selection_campaign_seal_sha256=outer_digest,
    )
    return _AuthenticatedFinalizeGlobals(
        protocol=protocol,
        stage=stage,
        prepare=prepare,
        reveal=reveal,
        update=update,
        outer_selection=outer_selection,
    )


def _require_indexed_leaf_order(
    request: FinalizeWorkerRequest,
    globals_: _AuthenticatedFinalizeGlobals,
) -> None:
    outcomes = _exact_seals(
        request.outer_outcome_vault_seals,
        count=EXPECTED_ROTATIONS,
        label="finalize outer-outcome capabilities",
    )
    prediction_views = _exact_seals(
        request.prediction_view_seals,
        count=EXPECTED_ROTATIONS,
        label="finalize prediction-view capabilities",
    )
    for spec, outcome, prediction_view in zip(
        ordered_rotations(), outcomes, prediction_views, strict=True
    ):
        outcome_entry = globals_.stage.leaf(spec=spec, role=OUTER_OUTCOME_ROLE)
        verify_phase_capability(
            outcome,
            expected_artifact=outcome_entry.leaf_artifact,
            expected_payload_paths=outcome_entry.payload_paths,
            expected_predecessor_seals=dict(globals_.stage.source_predecessors),
            expected_seal_sha256=outcome_entry.leaf_seal_sha256,
        )
        expected_prediction = globals_.prepare.leaf_seal_sha256(
            spec=spec,
            role=PREDICTION_VIEW_ROLE,
        )
        try:
            verify_phase_capability(
                prediction_view,
                expected_seal_sha256=expected_prediction,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "finalize prediction views differ from frozen rotation order"
            ) from error

    reveals = _exact_seals(
        request.reveal_leaf_seals,
        count=EXPECTED_POLICY_RUNS,
        label="finalize reveal capabilities",
    )
    evidence = _exact_seals(
        request.outer_evidence_seals,
        count=EXPECTED_POLICY_RUNS,
        label="finalize outer-evidence capabilities",
    )
    views = _exact_seals(
        request.outer_view_seals,
        count=EXPECTED_POLICY_RUNS,
        label="finalize outer-view capabilities",
    )
    selections = _exact_seals(
        request.outer_selection_leaf_seals,
        count=EXPECTED_POLICY_RUNS,
        label="finalize outer-selection capabilities",
    )
    for run, reveal, outer_evidence, outer_view, selection in zip(
        ordered_policy_runs(), reveals, evidence, views, selections, strict=True
    ):
        try:
            verify_phase_capability(
                reveal,
                expected_seal_sha256=globals_.reveal.index_row(run=run).leaf_seal_sha256,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("finalize reveals differ from frozen track order") from error
        evidence_row = globals_.update.outer_evidence_row(run=run)
        try:
            verify_phase_capability(
                outer_evidence,
                expected_seal_sha256=evidence_row.leaf_seal_sha256,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("finalize outer evidence differs from frozen track order") from error
        try:
            verify_phase_capability(
                outer_view,
                expected_seal_sha256=(globals_.update.outer_view_row(run=run).leaf_seal_sha256),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("finalize outer views differ from frozen track order") from error
        try:
            verify_phase_capability(
                selection,
                expected_seal_sha256=(globals_.outer_selection.index_row(run=run).leaf_seal_sha256),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("finalize outer selections differ from frozen track order") from error


@dataclass(frozen=True, slots=True)
class FinalizeWorkerRequest:
    """Complete rootless authority for one post-barrier finalizer process."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    stage_manifest_capability: StageManifestCapability
    prepare_campaign: PrepareCampaignCapability
    selection_campaign_seal: PhaseSeal
    reveal_campaign: RevealCampaignCapability
    update_campaign: UpdateCampaignCapability
    outer_selection_campaign: OuterSelectionCampaignCapability
    outer_outcome_vault_seals: tuple[PhaseSeal, ...]
    prediction_view_seals: tuple[PhaseSeal, ...]
    reveal_leaf_seals: tuple[PhaseSeal, ...]
    outer_evidence_seals: tuple[PhaseSeal, ...]
    outer_view_seals: tuple[PhaseSeal, ...]
    outer_selection_leaf_seals: tuple[PhaseSeal, ...]
    expected_protocol_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    expected_selection_campaign_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str
    expected_update_campaign_seal_sha256: str
    expected_outer_selection_campaign_seal_sha256: str

    def __post_init__(self) -> None:
        globals_ = _authenticate_globals(self)
        _require_indexed_leaf_order(self, globals_)

    def all_phase_seals(self) -> tuple[PhaseSeal, ...]:
        """Return every captured capability exactly once in wire order."""

        return (
            self.protocol_capability.seal,
            self.stage_manifest_capability.seal,
            self.prepare_campaign.seal,
            self.selection_campaign_seal,
            self.reveal_campaign.seal,
            self.update_campaign.seal,
            self.outer_selection_campaign.seal,
            *self.outer_outcome_vault_seals,
            *self.prediction_view_seals,
            *self.reveal_leaf_seals,
            *self.outer_evidence_seals,
            *self.outer_view_seals,
            *self.outer_selection_leaf_seals,
        )

    def canonical_bytes(self) -> bytes:
        seals = self.all_phase_seals()
        measurements = tuple(
            _core_wire._phase_seal_capture_measurements(
                seal,
                label=f"finalize request capability {index}",
            )
            for index, seal in enumerate(seals)
        )
        _require_capture_bounds(measurements, label="finalize worker request")
        document = {
            "schema_version": SCHEMA_VERSION,
            "artifact": FINALIZE_WORKER_REQUEST_ARTIFACT,
            "publication_identity": publication_identity_document(self.publication_identity),
            "protocol_capability": phase_seal_document(self.protocol_capability.seal),
            "stage_manifest_capability": phase_seal_document(self.stage_manifest_capability.seal),
            "prepare_campaign_capability": phase_seal_document(self.prepare_campaign.seal),
            "selection_campaign_capability": phase_seal_document(self.selection_campaign_seal),
            "reveal_campaign_capability": phase_seal_document(self.reveal_campaign.seal),
            "update_campaign_capability": phase_seal_document(self.update_campaign.seal),
            "outer_selection_campaign_capability": phase_seal_document(
                self.outer_selection_campaign.seal
            ),
            "outer_outcome_vault_capabilities": [
                phase_seal_document(seal) for seal in self.outer_outcome_vault_seals
            ],
            "prediction_view_capabilities": [
                phase_seal_document(seal) for seal in self.prediction_view_seals
            ],
            "reveal_leaf_capabilities": [
                phase_seal_document(seal) for seal in self.reveal_leaf_seals
            ],
            "outer_evidence_capabilities": [
                phase_seal_document(seal) for seal in self.outer_evidence_seals
            ],
            "outer_view_capabilities": [
                phase_seal_document(seal) for seal in self.outer_view_seals
            ],
            "outer_selection_leaf_capabilities": [
                phase_seal_document(seal) for seal in self.outer_selection_leaf_seals
            ],
            **{field: getattr(self, field) for field in _EXPECTED_DIGEST_FIELDS},
        }
        assert_wire_document_has_no_source_path_fields(document)
        payload = canonical_json_bytes(document)
        if len(payload) > MAX_WIRE_REQUEST_BYTES:
            raise ValueError("finalize worker request exceeds its encoded byte bound")
        return payload


def _require_capture_bounds(measurements: tuple[object, ...], *, label: str) -> None:
    decoded = sum(item.decoded_bytes for item in measurements)  # type: ignore[attr-defined]
    encoded = sum(item.base64_characters for item in measurements)  # type: ignore[attr-defined]
    if decoded > MAX_REQUEST_CAPTURED_BYTES:
        raise ValueError(f"{label} exceeds its aggregate captured-byte bound")
    if encoded > MAX_REQUEST_BASE64_CHARACTERS:
        raise ValueError(f"{label} exceeds its aggregate base64 bound")


def _phase_array(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[object, ...]:
    if (
        type(value) is not list
        or len(value) != count
        or any(type(item) is not dict for item in value)
    ):
        raise ValueError(f"{label} must contain exactly {count} phase documents")
    return tuple(value)


def finalize_worker_request_from_bytes(payload: bytes) -> FinalizeWorkerRequest:
    """Strictly reconstruct and externally reauthenticate one finalizer request."""

    document = strict_canonical_json_object(
        payload,
        label="finalize worker request",
        maximum_bytes=MAX_WIRE_REQUEST_BYTES,
    )
    if type(document) is not dict or set(document) != _REQUEST_FIELDS:
        raise ValueError("finalize worker request must contain its exact field set")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or document["artifact"] != FINALIZE_WORKER_REQUEST_ARTIFACT
    ):
        raise ValueError("finalize worker request identity is invalid")
    assert_wire_document_has_no_source_path_fields(document)

    rotation_arrays = {
        field: _phase_array(
            document[field],
            count=EXPECTED_ROTATIONS,
            label=f"finalize request {field}",
        )
        for field in _ROTATION_LEAF_FIELDS
    }
    track_arrays = {
        field: _phase_array(
            document[field],
            count=EXPECTED_POLICY_RUNS,
            label=f"finalize request {field}",
        )
        for field in _TRACK_LEAF_FIELDS
    }
    phase_documents = (
        *(document[field] for field in _GLOBAL_FIELDS),
        *(item for field in _ROTATION_LEAF_FIELDS for item in rotation_arrays[field]),
        *(item for field in _TRACK_LEAF_FIELDS for item in track_arrays[field]),
    )
    measurements = tuple(
        _core_wire._phase_document_capture_measurements(
            item,
            label=f"finalize request capability {index}",
        )
        for index, item in enumerate(phase_documents)
    )
    _require_capture_bounds(measurements, label="finalize worker request")

    identity = publication_identity_from_document(document["publication_identity"])
    result = FinalizeWorkerRequest(
        publication_identity=identity,
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        stage_manifest_capability=StageManifestCapability(
            phase_seal_from_document(document["stage_manifest_capability"])
        ),
        prepare_campaign=PrepareCampaignCapability(
            phase_seal_from_document(document["prepare_campaign_capability"]), identity
        ),
        selection_campaign_seal=phase_seal_from_document(document["selection_campaign_capability"]),
        reveal_campaign=RevealCampaignCapability(
            phase_seal_from_document(document["reveal_campaign_capability"]), identity
        ),
        update_campaign=UpdateCampaignCapability(
            phase_seal_from_document(document["update_campaign_capability"]), identity
        ),
        outer_selection_campaign=OuterSelectionCampaignCapability(
            phase_seal_from_document(document["outer_selection_campaign_capability"]), identity
        ),
        outer_outcome_vault_seals=tuple(
            phase_seal_from_document(item)
            for item in rotation_arrays["outer_outcome_vault_capabilities"]
        ),
        prediction_view_seals=tuple(
            phase_seal_from_document(item)
            for item in rotation_arrays["prediction_view_capabilities"]
        ),
        reveal_leaf_seals=tuple(
            phase_seal_from_document(item) for item in track_arrays["reveal_leaf_capabilities"]
        ),
        outer_evidence_seals=tuple(
            phase_seal_from_document(item) for item in track_arrays["outer_evidence_capabilities"]
        ),
        outer_view_seals=tuple(
            phase_seal_from_document(item) for item in track_arrays["outer_view_capabilities"]
        ),
        outer_selection_leaf_seals=tuple(
            phase_seal_from_document(item)
            for item in track_arrays["outer_selection_leaf_capabilities"]
        ),
        **{field: document[field] for field in _EXPECTED_DIGEST_FIELDS},
    )
    if result.canonical_bytes() != payload:
        raise ValueError("finalize worker request changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class FinalizeRequestCapacityMeasurement:
    """Authenticated size census for one exact full-campaign request."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    stage_global_seal_sha256: str
    prepare_global_seal_sha256: str
    selection_global_seal_sha256: str
    reveal_global_seal_sha256: str
    update_global_seal_sha256: str
    outer_selection_global_seal_sha256: str
    phase_capability_count: int
    predecessor_count: int
    captured_bytes: int
    base64_characters: int
    wire_bytes: int
    request_sha256: str

    def __post_init__(self) -> None:
        _require_identity(self.publication_identity)
        for name in (
            "protocol_seal_sha256",
            "stage_global_seal_sha256",
            "prepare_global_seal_sha256",
            "selection_global_seal_sha256",
            "reveal_global_seal_sha256",
            "update_global_seal_sha256",
            "outer_selection_global_seal_sha256",
            "request_sha256",
        ):
            _sha256(getattr(self, name), label=f"finalize capacity measurement {name}")
        expected_counts = {
            "phase_capability_count": FINALIZE_REQUEST_CAPABILITY_COUNT,
            "predecessor_count": FINALIZE_REQUEST_PREDECESSOR_COUNT,
        }
        for name, expected in expected_counts.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"finalize capacity measurement {name} must equal {expected}")
        bounds = {
            "captured_bytes": MAX_REQUEST_CAPTURED_BYTES,
            "base64_characters": MAX_REQUEST_BASE64_CHARACTERS,
            "wire_bytes": MAX_WIRE_REQUEST_BYTES,
        }
        for name, limit in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < limit:
                raise ValueError(
                    f"finalize capacity measurement {name} must be a nonnegative "
                    f"integer strictly below {limit}"
                )

    def document(self) -> dict[str, object]:
        """Return the canonical, request-free measurement document."""

        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": FINALIZE_REQUEST_CAPACITY_MEASUREMENT_ARTIFACT,
            "publication_identity": publication_identity_document(self.publication_identity),
            "global_authorities": {
                "protocol_seal_sha256": self.protocol_seal_sha256,
                "stage_global_seal_sha256": self.stage_global_seal_sha256,
                "prepare_global_seal_sha256": self.prepare_global_seal_sha256,
                "selection_global_seal_sha256": self.selection_global_seal_sha256,
                "reveal_global_seal_sha256": self.reveal_global_seal_sha256,
                "update_global_seal_sha256": self.update_global_seal_sha256,
                "outer_selection_global_seal_sha256": (self.outer_selection_global_seal_sha256),
            },
            "phase_capability_count": self.phase_capability_count,
            "predecessor_count": self.predecessor_count,
            "observed": {
                "captured_bytes": self.captured_bytes,
                "base64_characters": self.base64_characters,
                "wire_bytes": self.wire_bytes,
            },
            "request_sha256": self.request_sha256,
        }


def encode_and_measure_finalize_worker_request(
    request: FinalizeWorkerRequest,
) -> tuple[bytes, FinalizeRequestCapacityMeasurement]:
    """Encode, round-trip, and size one authentic full-census request.

    Capacity acceptance deliberately requires strict headroom below every
    transport bound.  The ordinary wire codec remains inclusive at the exact
    bounds so its compatibility contract is unchanged.
    """

    if type(request) is not FinalizeWorkerRequest:
        raise TypeError("finalize capacity measurement requires an exact worker request")
    seals = request.all_phase_seals()
    if len(seals) != FINALIZE_REQUEST_CAPABILITY_COUNT:
        raise AssertionError("finalize capacity capability census changed")
    measurements = tuple(
        _core_wire._phase_seal_capture_measurements(
            seal,
            label=f"finalize capacity capability {index}",
        )
        for index, seal in enumerate(seals)
    )
    captured_bytes = sum(item.decoded_bytes for item in measurements)
    base64_characters = sum(item.base64_characters for item in measurements)
    if captured_bytes >= MAX_REQUEST_CAPTURED_BYTES:
        raise ValueError(
            "finalize capacity captured bytes require strict headroom below the wire bound"
        )
    if base64_characters >= MAX_REQUEST_BASE64_CHARACTERS:
        raise ValueError(
            "finalize capacity base64 characters require strict headroom below the wire bound"
        )

    payload = request.canonical_bytes()
    if len(payload) >= MAX_WIRE_REQUEST_BYTES:
        raise ValueError(
            "finalize capacity wire bytes require strict headroom below the wire bound"
        )
    if finalize_worker_request_from_bytes(payload) != request:
        raise ValueError("finalize capacity request changed during typed reconstruction")

    return payload, FinalizeRequestCapacityMeasurement(
        publication_identity=request.publication_identity,
        protocol_seal_sha256=request.expected_protocol_seal_sha256,
        stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        prepare_global_seal_sha256=request.expected_prepare_campaign_seal_sha256,
        selection_global_seal_sha256=request.expected_selection_campaign_seal_sha256,
        reveal_global_seal_sha256=request.expected_reveal_campaign_seal_sha256,
        update_global_seal_sha256=request.expected_update_campaign_seal_sha256,
        outer_selection_global_seal_sha256=(request.expected_outer_selection_campaign_seal_sha256),
        phase_capability_count=len(seals),
        predecessor_count=FINALIZE_REQUEST_PREDECESSOR_COUNT,
        captured_bytes=captured_bytes,
        base64_characters=base64_characters,
        wire_bytes=len(payload),
        request_sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "FINALIZE_REQUEST_CAPABILITY_COUNT",
    "FINALIZE_REQUEST_CAPACITY_MEASUREMENT_ARTIFACT",
    "FINALIZE_REQUEST_PREDECESSOR_COUNT",
    "FINALIZE_WORKER_REQUEST_ARTIFACT",
    "FINALIZE_WORKER_ROLE",
    "FinalizeRequestCapacityMeasurement",
    "FinalizeWorkerRequest",
    "encode_and_measure_finalize_worker_request",
    "finalize_worker_request_from_bytes",
]

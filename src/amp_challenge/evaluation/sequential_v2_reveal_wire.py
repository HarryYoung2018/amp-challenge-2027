"""Strict pathless wire requests for sequential-v2 REVEAL workers.

The controller captures and authenticates filesystem-backed phase outputs
before constructing these values.  Requests carry only rootless
``PhaseSeal`` capabilities and independently captured digest anchors; they
never carry controller or source paths.  The no-query and nonempty leaf
schemas are deliberately separate so a no-query worker cannot be granted a
stage or outcome-vault capability accidentally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from amp_challenge.evaluation import sequential_v2_reveal as reveal_core
from amp_challenge.evaluation.sequential_v2_commitments import (
    pool_commitment_capability_from_seals,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    ProtocolCapability,
    SequentialV2PublicationIdentity,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    NO_QUERY,
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    policy_run_by_track_id,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    RevealLeafAttestation,
    reveal_leaf_attestation_from_document,
)
from amp_challenge.evaluation.sequential_v2_seals import PhaseSeal, canonical_json_bytes
from amp_challenge.evaluation.sequential_v2_stage import (
    POOL_OUTCOME_ROLE,
    AuthenticatedLeafCapsule,
    verify_stage_manifest_capability,
)
from amp_challenge.evaluation.sequential_v2_wire import (
    _phase_document_capture_measurements,
    _phase_seal_capture_measurements,
    assert_wire_document_has_no_source_path_fields,
    phase_seal_document,
    phase_seal_from_document,
    publication_identity_document,
    publication_identity_from_document,
    strict_canonical_json_object,
)

SCHEMA_VERSION = 1

REVEAL_NO_QUERY_WORKER_REQUEST_ARTIFACT = "sequential_v2_reveal_no_query_worker_request_v1"
REVEAL_NONEMPTY_WORKER_REQUEST_ARTIFACT = "sequential_v2_reveal_nonempty_worker_request_v1"
REVEAL_BARRIER_WORKER_REQUEST_ARTIFACT = "sequential_v2_reveal_barrier_worker_request_v1"

REVEAL_NO_QUERY_WORKER_ROLE = "reveal-no-query"
REVEAL_NONEMPTY_WORKER_ROLE = "reveal-nonempty"
REVEAL_BARRIER_WORKER_ROLE = "reveal-barrier"

_MAX_WIRE_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_REQUEST_CAPTURED_BYTES = 128 * 1024 * 1024
_MAX_REQUEST_BASE64_CHARACTERS = 192 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "track_id",
        "rotation_id",
        "policy",
        "seed",
        "selection_kind",
        "expected_pool_selection_count",
        "expected_outer_selection_count",
        "refit",
    }
)
_NO_QUERY_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "run",
        "publication_identity",
        "protocol_capability",
        "selection_barrier_capability",
        "commitment_leaf_capability",
        "expected_prepare_campaign_seal_sha256",
        "expected_selection_barrier_seal_sha256",
    }
)
_NONEMPTY_REQUEST_FIELDS = frozenset(
    {
        *_NO_QUERY_REQUEST_FIELDS,
        "stage_global_capability",
        "expected_stage_global_seal_sha256",
        "pool_outcome_vault_capability",
    }
)
_BARRIER_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "publication_identity",
        "protocol_capability",
        "stage_global_capability",
        "selection_barrier_capability",
        "expected_prepare_campaign_seal_sha256",
        "expected_stage_global_seal_sha256",
        "expected_selection_barrier_seal_sha256",
        "reveal_leaf_attestations",
        "expected_reveal_leaf_seal_sha256s",
    }
)


def _exact_object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(key) is not str for key in value)
    ):
        raise ValueError(f"{label} must contain its exact field set")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec or type(value.rotation) is not RotationSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    if (
        type(value.rotation.outer_fold) is not int
        or type(value.rotation.pool_fold) is not int
        or type(value.policy) is not str
        or (value.seed is not None and type(value.seed) is not int)
    ):
        raise TypeError(f"{label} fields must use exact scalar types")
    canonical = policy_run_by_track_id(value.track_id)
    if value != canonical:
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _run_from_document(value: object, *, label: str) -> PolicyRunSpec:
    document = _exact_object(value, fields=_RUN_FIELDS, label=label)
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["track_id"]) is not str
    ):
        raise ValueError(f"{label} identity is invalid")
    run = policy_run_by_track_id(document["track_id"])
    if canonical_json_bytes(document) != canonical_json_bytes(run.document()):
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return _frozen_run(run, label=label)


def _bounded_request_bytes(document: dict[str, object], *, label: str) -> bytes:
    assert_wire_document_has_no_source_path_fields(document)
    payload = canonical_json_bytes(document)
    if len(payload) > _MAX_WIRE_REQUEST_BYTES:
        raise ValueError(f"{label} exceeds its encoded byte bound")
    return payload


def _require_capture_bounds(measurements: tuple[object, ...], *, label: str) -> None:
    decoded_bytes = sum(item.decoded_bytes for item in measurements)
    base64_characters = sum(item.base64_characters for item in measurements)
    if decoded_bytes > _MAX_REQUEST_CAPTURED_BYTES:
        raise ValueError(f"{label} exceeds its aggregate captured-byte bound")
    if base64_characters > _MAX_REQUEST_BASE64_CHARACTERS:
        raise ValueError(f"{label} exceeds its aggregate base64 bound")


def _preflight_seals(seals: tuple[PhaseSeal, ...], *, label: str) -> None:
    _require_capture_bounds(
        tuple(
            _phase_seal_capture_measurements(seal, label=f"{label} capability") for seal in seals
        ),
        label=label,
    )


def _preflight_documents(
    document: dict[str, object],
    *,
    capability_fields: tuple[str, ...],
    label: str,
) -> None:
    _require_capture_bounds(
        tuple(
            _phase_document_capture_measurements(
                document[field],
                label=f"{label} {field}",
            )
            for field in capability_fields
        ),
        label=label,
    )


def _decode_request_document(
    payload: bytes,
    *,
    artifact: str,
    fields: frozenset[str],
    capability_fields: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    document = strict_canonical_json_object(
        payload,
        label=label,
        maximum_bytes=_MAX_WIRE_REQUEST_BYTES,
    )
    assert_wire_document_has_no_source_path_fields(document)
    document = _exact_object(document, fields=fields, label=label)
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SCHEMA_VERSION
        or type(document["artifact"]) is not str
        or document["artifact"] != artifact
    ):
        raise ValueError(f"{label} identity is invalid")
    _preflight_documents(
        document,
        capability_fields=capability_fields,
        label=label,
    )
    return document


def _validate_leaf_authorities(
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    selection_barrier_seal: PhaseSeal,
    commitment_leaf_seal: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
) -> None:
    pool_commitment_capability_from_seals(
        commitment_leaf_seal,
        selection_barrier_seal,
        run=run,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        publication_identity=publication_identity,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
    )


def _validate_nonempty_vault_authority(
    *,
    run: PolicyRunSpec,
    stage_global_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    pool_outcome_vault_seal: PhaseSeal,
) -> None:
    stage = verify_stage_manifest_capability(
        stage_global_seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    entry = stage.leaf(spec=run.rotation, role=POOL_OUTCOME_ROLE)
    AuthenticatedLeafCapsule(
        entry=entry,
        seal=pool_outcome_vault_seal,
        source_anchors_sha256=stage.source_anchors_sha256,
        source_predecessors=tuple(sorted(stage.source_predecessors)),
    )


def _validate_common_leaf_request(
    *,
    run: object,
    publication_identity: object,
    protocol_capability: object,
    selection_barrier_seal: object,
    commitment_leaf_seal: object,
    expected_prepare_campaign_seal_sha256: object,
    expected_selection_barrier_seal_sha256: object,
    label: str,
) -> PolicyRunSpec:
    frozen_run = _frozen_run(run, label=f"{label} run")
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError(f"{label} requires an exact publication identity")
    if (
        type(protocol_capability) is not ProtocolCapability
        or type(protocol_capability.seal) is not PhaseSeal
    ):
        raise TypeError(f"{label} requires an exact protocol capability")
    if type(selection_barrier_seal) is not PhaseSeal:
        raise TypeError(f"{label} selection barrier must be an exact PhaseSeal")
    if type(commitment_leaf_seal) is not PhaseSeal:
        raise TypeError(f"{label} commitment leaf must be an exact PhaseSeal")
    prepare_digest = _sha256(
        expected_prepare_campaign_seal_sha256,
        label=f"{label} expected prepare campaign seal",
    )
    selection_digest = _sha256(
        expected_selection_barrier_seal_sha256,
        label=f"{label} expected selection barrier seal",
    )
    if selection_barrier_seal.seal_sha256 != selection_digest:
        raise ValueError(f"{label} selection barrier differs from external authority")
    _validate_leaf_authorities(
        run=frozen_run,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        selection_barrier_seal=selection_barrier_seal,
        commitment_leaf_seal=commitment_leaf_seal,
        expected_prepare_campaign_seal_sha256=prepare_digest,
        expected_selection_barrier_seal_sha256=selection_digest,
    )
    return frozen_run


@dataclass(frozen=True, slots=True)
class RevealNoQueryWorkerRequest:
    """The exact three-predecessor authority for one no-query reveal."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    selection_barrier_seal: PhaseSeal
    commitment_leaf_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    expected_selection_barrier_seal_sha256: str

    def __post_init__(self) -> None:
        run = _validate_common_leaf_request(
            run=self.run,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            selection_barrier_seal=self.selection_barrier_seal,
            commitment_leaf_seal=self.commitment_leaf_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            label="reveal no-query worker request",
        )
        if run.policy != NO_QUERY:
            raise ValueError("reveal no-query worker request requires the no-query policy")

    def canonical_bytes(self) -> bytes:
        label = "reveal no-query worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.selection_barrier_seal,
                self.commitment_leaf_seal,
            ),
            label=label,
        )
        return _bounded_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": REVEAL_NO_QUERY_WORKER_REQUEST_ARTIFACT,
                "run": self.run.document(),
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "selection_barrier_capability": phase_seal_document(self.selection_barrier_seal),
                "commitment_leaf_capability": phase_seal_document(self.commitment_leaf_seal),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_selection_barrier_seal_sha256": (
                    self.expected_selection_barrier_seal_sha256
                ),
            },
            label=label,
        )


def reveal_no_query_worker_request_from_bytes(
    payload: bytes,
) -> RevealNoQueryWorkerRequest:
    label = "reveal no-query worker request"
    document = _decode_request_document(
        payload,
        artifact=REVEAL_NO_QUERY_WORKER_REQUEST_ARTIFACT,
        fields=_NO_QUERY_REQUEST_FIELDS,
        capability_fields=(
            "protocol_capability",
            "selection_barrier_capability",
            "commitment_leaf_capability",
        ),
        label=label,
    )
    result = RevealNoQueryWorkerRequest(
        run=_run_from_document(document["run"], label=f"{label} run"),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        selection_barrier_seal=phase_seal_from_document(document["selection_barrier_capability"]),
        commitment_leaf_seal=phase_seal_from_document(document["commitment_leaf_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        expected_selection_barrier_seal_sha256=document["expected_selection_barrier_seal_sha256"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class RevealNonemptyWorkerRequest:
    """The exact five-predecessor authority for one nonempty reveal."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    selection_barrier_seal: PhaseSeal
    commitment_leaf_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    stage_global_seal: PhaseSeal
    expected_stage_global_seal_sha256: str
    pool_outcome_vault_seal: PhaseSeal

    def __post_init__(self) -> None:
        label = "reveal nonempty worker request"
        run = _validate_common_leaf_request(
            run=self.run,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            selection_barrier_seal=self.selection_barrier_seal,
            commitment_leaf_seal=self.commitment_leaf_seal,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            label=label,
        )
        if run.policy == NO_QUERY:
            raise ValueError("reveal nonempty worker request rejects the no-query policy")
        if type(self.stage_global_seal) is not PhaseSeal:
            raise TypeError(f"{label} stage global must be an exact PhaseSeal")
        if type(self.pool_outcome_vault_seal) is not PhaseSeal:
            raise TypeError(f"{label} pool vault must be an exact PhaseSeal")
        stage_digest = _sha256(
            self.expected_stage_global_seal_sha256,
            label=f"{label} expected stage-global seal",
        )
        if self.stage_global_seal.seal_sha256 != stage_digest:
            raise ValueError(f"{label} stage global differs from external authority")
        _validate_nonempty_vault_authority(
            run=run,
            stage_global_seal=self.stage_global_seal,
            expected_stage_global_seal_sha256=stage_digest,
            pool_outcome_vault_seal=self.pool_outcome_vault_seal,
        )

    def canonical_bytes(self) -> bytes:
        label = "reveal nonempty worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.selection_barrier_seal,
                self.commitment_leaf_seal,
                self.stage_global_seal,
                self.pool_outcome_vault_seal,
            ),
            label=label,
        )
        return _bounded_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": REVEAL_NONEMPTY_WORKER_REQUEST_ARTIFACT,
                "run": self.run.document(),
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "selection_barrier_capability": phase_seal_document(self.selection_barrier_seal),
                "commitment_leaf_capability": phase_seal_document(self.commitment_leaf_seal),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_selection_barrier_seal_sha256": (
                    self.expected_selection_barrier_seal_sha256
                ),
                "stage_global_capability": phase_seal_document(self.stage_global_seal),
                "expected_stage_global_seal_sha256": (self.expected_stage_global_seal_sha256),
                "pool_outcome_vault_capability": phase_seal_document(self.pool_outcome_vault_seal),
            },
            label=label,
        )


def reveal_nonempty_worker_request_from_bytes(
    payload: bytes,
) -> RevealNonemptyWorkerRequest:
    label = "reveal nonempty worker request"
    document = _decode_request_document(
        payload,
        artifact=REVEAL_NONEMPTY_WORKER_REQUEST_ARTIFACT,
        fields=_NONEMPTY_REQUEST_FIELDS,
        capability_fields=(
            "protocol_capability",
            "selection_barrier_capability",
            "commitment_leaf_capability",
            "stage_global_capability",
            "pool_outcome_vault_capability",
        ),
        label=label,
    )
    result = RevealNonemptyWorkerRequest(
        run=_run_from_document(document["run"], label=f"{label} run"),
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        selection_barrier_seal=phase_seal_from_document(document["selection_barrier_capability"]),
        commitment_leaf_seal=phase_seal_from_document(document["commitment_leaf_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        expected_selection_barrier_seal_sha256=document["expected_selection_barrier_seal_sha256"],
        stage_global_seal=phase_seal_from_document(document["stage_global_capability"]),
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        pool_outcome_vault_seal=phase_seal_from_document(document["pool_outcome_vault_capability"]),
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


def _validate_barrier_authorities(
    *,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_global_seal: PhaseSeal,
    selection_barrier_seal: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    attestations: tuple[RevealLeafAttestation, ...],
    expected_reveal_leaf_seal_sha256s: tuple[str, ...],
) -> None:
    stage = verify_stage_manifest_capability(
        stage_global_seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    reveal_core._validate_reveal_attestations(
        attestations,
        expected_reveal_leaf_seal_sha256s=expected_reveal_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage,
        selection_barrier=selection_barrier_seal,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
    )


@dataclass(frozen=True, slots=True)
class RevealBarrierWorkerRequest:
    """The label-free ordered authority for the reveal-global custodian."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    stage_global_seal: PhaseSeal
    selection_barrier_seal: PhaseSeal
    expected_prepare_campaign_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    attestations: tuple[RevealLeafAttestation, ...]
    expected_reveal_leaf_seal_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        label = "reveal barrier worker request"
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError(f"{label} requires an exact publication identity")
        if (
            type(self.protocol_capability) is not ProtocolCapability
            or type(self.protocol_capability.seal) is not PhaseSeal
        ):
            raise TypeError(f"{label} requires an exact protocol capability")
        if type(self.stage_global_seal) is not PhaseSeal:
            raise TypeError(f"{label} stage global must be an exact PhaseSeal")
        if type(self.selection_barrier_seal) is not PhaseSeal:
            raise TypeError(f"{label} selection barrier must be an exact PhaseSeal")
        prepare_digest = _sha256(
            self.expected_prepare_campaign_seal_sha256,
            label=f"{label} expected prepare campaign seal",
        )
        stage_digest = _sha256(
            self.expected_stage_global_seal_sha256,
            label=f"{label} expected stage-global seal",
        )
        selection_digest = _sha256(
            self.expected_selection_barrier_seal_sha256,
            label=f"{label} expected selection barrier seal",
        )
        if self.stage_global_seal.seal_sha256 != stage_digest:
            raise ValueError(f"{label} stage global differs from external authority")
        if self.selection_barrier_seal.seal_sha256 != selection_digest:
            raise ValueError(f"{label} selection barrier differs from external authority")

        runs = ordered_policy_runs()
        if (
            type(self.attestations) is not tuple
            or len(self.attestations) != len(runs)
            or any(type(item) is not RevealLeafAttestation for item in self.attestations)
            or tuple(item.run for item in self.attestations) != runs
        ):
            raise ValueError(f"{label} requires 220 exact attestations in frozen order")
        if type(self.expected_reveal_leaf_seal_sha256s) is not tuple or len(
            self.expected_reveal_leaf_seal_sha256s
        ) != len(runs):
            raise ValueError(f"{label} requires 220 ordered expected reveal leaf seals")
        expected_leaf_digests = tuple(
            _sha256(value, label=f"{label} expected reveal leaf seal {index}")
            for index, value in enumerate(self.expected_reveal_leaf_seal_sha256s)
        )
        if len(set(expected_leaf_digests)) != len(runs):
            raise ValueError(f"{label} expected reveal leaf seals must be distinct")
        if tuple(item.reveal_leaf_seal_sha256 for item in self.attestations) != (
            expected_leaf_digests
        ):
            raise ValueError(f"{label} attestations differ from external leaf authority")
        if any(
            item.publication_identity != self.publication_identity
            or item.protocol_seal_sha256 != self.protocol_capability.seal.seal_sha256
            or item.select_global_seal_sha256 != selection_digest
            or (
                item.stage_global_seal_sha256
                != (None if item.run.policy == NO_QUERY else stage_digest)
            )
            for item in self.attestations
        ):
            raise ValueError(f"{label} attestations differ from global authorities")

        _validate_barrier_authorities(
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            stage_global_seal=self.stage_global_seal,
            selection_barrier_seal=self.selection_barrier_seal,
            expected_prepare_campaign_seal_sha256=prepare_digest,
            expected_stage_global_seal_sha256=stage_digest,
            expected_selection_barrier_seal_sha256=selection_digest,
            attestations=self.attestations,
            expected_reveal_leaf_seal_sha256s=expected_leaf_digests,
        )

    def canonical_bytes(self) -> bytes:
        label = "reveal barrier worker request"
        _preflight_seals(
            (
                self.protocol_capability.seal,
                self.stage_global_seal,
                self.selection_barrier_seal,
            ),
            label=label,
        )
        return _bounded_request_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": REVEAL_BARRIER_WORKER_REQUEST_ARTIFACT,
                "publication_identity": publication_identity_document(self.publication_identity),
                "protocol_capability": phase_seal_document(self.protocol_capability.seal),
                "stage_global_capability": phase_seal_document(self.stage_global_seal),
                "selection_barrier_capability": phase_seal_document(self.selection_barrier_seal),
                "expected_prepare_campaign_seal_sha256": (
                    self.expected_prepare_campaign_seal_sha256
                ),
                "expected_stage_global_seal_sha256": (self.expected_stage_global_seal_sha256),
                "expected_selection_barrier_seal_sha256": (
                    self.expected_selection_barrier_seal_sha256
                ),
                "reveal_leaf_attestations": [item.document() for item in self.attestations],
                "expected_reveal_leaf_seal_sha256s": list(self.expected_reveal_leaf_seal_sha256s),
            },
            label=label,
        )


def reveal_barrier_worker_request_from_bytes(
    payload: bytes,
) -> RevealBarrierWorkerRequest:
    label = "reveal barrier worker request"
    document = _decode_request_document(
        payload,
        artifact=REVEAL_BARRIER_WORKER_REQUEST_ARTIFACT,
        fields=_BARRIER_REQUEST_FIELDS,
        capability_fields=(
            "protocol_capability",
            "stage_global_capability",
            "selection_barrier_capability",
        ),
        label=label,
    )
    attestations_raw = document["reveal_leaf_attestations"]
    expected_leaf_raw = document["expected_reveal_leaf_seal_sha256s"]
    if (
        type(attestations_raw) is not list
        or len(attestations_raw) != len(ordered_policy_runs())
        or type(expected_leaf_raw) is not list
        or len(expected_leaf_raw) != len(ordered_policy_runs())
        or any(type(item) is not str for item in expected_leaf_raw)
    ):
        raise ValueError(f"{label} ordered census or types are invalid")
    result = RevealBarrierWorkerRequest(
        publication_identity=publication_identity_from_document(document["publication_identity"]),
        protocol_capability=ProtocolCapability(
            phase_seal_from_document(document["protocol_capability"])
        ),
        stage_global_seal=phase_seal_from_document(document["stage_global_capability"]),
        selection_barrier_seal=phase_seal_from_document(document["selection_barrier_capability"]),
        expected_prepare_campaign_seal_sha256=document["expected_prepare_campaign_seal_sha256"],
        expected_stage_global_seal_sha256=document["expected_stage_global_seal_sha256"],
        expected_selection_barrier_seal_sha256=document["expected_selection_barrier_seal_sha256"],
        attestations=tuple(
            reveal_leaf_attestation_from_document(item) for item in attestations_raw
        ),
        expected_reveal_leaf_seal_sha256s=tuple(expected_leaf_raw),
    )
    if result.canonical_bytes() != payload:
        raise ValueError(f"{label} changed during typed reconstruction")
    return result


__all__ = [
    "REVEAL_BARRIER_WORKER_REQUEST_ARTIFACT",
    "REVEAL_BARRIER_WORKER_ROLE",
    "REVEAL_NONEMPTY_WORKER_REQUEST_ARTIFACT",
    "REVEAL_NONEMPTY_WORKER_ROLE",
    "REVEAL_NO_QUERY_WORKER_REQUEST_ARTIFACT",
    "REVEAL_NO_QUERY_WORKER_ROLE",
    "RevealBarrierWorkerRequest",
    "RevealNoQueryWorkerRequest",
    "RevealNonemptyWorkerRequest",
    "reveal_barrier_worker_request_from_bytes",
    "reveal_no_query_worker_request_from_bytes",
    "reveal_nonempty_worker_request_from_bytes",
]

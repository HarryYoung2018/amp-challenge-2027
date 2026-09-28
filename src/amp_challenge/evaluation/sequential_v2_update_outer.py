"""Sealed outer-component and projection boundaries for sequential v2.

The logical update phase deliberately uses three process classes.  State
workers are the only processes that see labels.  A component worker receives
only eleven payload-free state attestations before it may open one rotation's
outer metadata, and a projection worker receives only one authenticated model
state, one component leaf, and the same label-free metadata.

This module implements the latter two boundaries.  The worker-facing public
functions always authenticate rootless phase capabilities and independently
supplied controller digests before reading their payloads.  Attestations are
safe, canonical IPC records; constructing one is not a substitute for the
supervisor-controlled fresh-exec result channel described by the protocol.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amp_challenge.acquisition.sequential_v2_selector import OBJECTIVES, OuterMeanCandidate
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_primitives import PROBABILITY_CLIP, TARGETS
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_CONTEXTS_BY_FOLD,
    EXPECTED_POLICY_RUNS,
    EXPECTED_ROTATIONS,
    EXPECTED_SUPPORT_BY_FOLD,
    EXPECTED_TRACKS_PER_ROTATION,
    PolicyRunSpec,
    RotationSpec,
    policy_run_by_track_id,
    policy_runs_for_rotation,
    rotation_by_id,
)
from amp_challenge.evaluation.sequential_v2_reveal import RevealCampaignCapability
from amp_challenge.evaluation.sequential_v2_seals import (
    RECEIPT_NAME,
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    checksum_manifest_bytes,
    publish_phase,
    sha256_bytes,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    OUTER_METADATA_ROLE,
    AuthenticatedLeafCapsule,
    StageManifestCapability,
    outer_metadata_capability_from_stage_capabilities,
    verify_stage_manifest_capability,
)
from amp_challenge.evaluation.sequential_v2_staging import OuterMetadataCapability
from amp_challenge.evaluation.sequential_v2_update import (
    OUTER_COMPONENT_PAYLOAD_PATHS,
    OUTER_COMPONENTS_ARTIFACT,
    OUTER_EVIDENCE_ARTIFACT,
    OUTER_EVIDENCE_PAYLOAD_PATHS,
    OUTER_VIEW_ARTIFACT,
    OUTER_VIEW_PAYLOAD_PATHS,
    UPDATE_STATE_ARTIFACT,
    UPDATE_STATE_PAYLOAD_PATHS,
    OuterComponent,
    OuterComponentSet,
    OuterContextPrediction,
    OuterProjection,
    OuterSequencePrediction,
    UpdateStateCapability,
    build_outer_component_set,
    build_outer_projection,
    id_stream_sha256,
    outer_mean_candidate_document,
)

SCHEMA_VERSION = 1

OUTER_COMPONENT_SUMMARY_ARTIFACT = "sequential_v2_update_outer_components_summary_v1"
OUTER_COMPONENT_ATTESTATION_ARTIFACT = "sequential_v2_update_outer_components_attestation_v1"
OUTER_VIEW_SUMMARY_ARTIFACT = "sequential_v2_update_outer_view_summary_v1"
OUTER_EVIDENCE_SUMMARY_ARTIFACT = "sequential_v2_update_outer_evidence_summary_v1"
OUTER_PROJECTION_ATTESTATION_ARTIFACT = "sequential_v2_update_outer_projection_attestation_v1"

EXPECTED_COMPONENT_LEAVES = EXPECTED_ROTATIONS
EXPECTED_PROJECTION_WORKERS = EXPECTED_POLICY_RUNS
EXPECTED_COMPONENT_MEMBERSHIPS = 2_600
EXPECTED_OUTER_CONTEXT_PREDICTIONS = 109_648
EXPECTED_OUTER_SEQUENCE_PREDICTIONS = 28_600

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_ID = re.compile(r"seqv2-div70:[0-9a-f]{64}\Z")
_OUTER_VIEW_FIELDS = (
    "rotation_id",
    "sequence_id",
    "sequence",
    "objective_probabilities",
    "diversity_component_id",
    "eligible",
)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _exact_object(
    value: object,
    fields: set[str] | frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} keys must be exact text")
    return value


def _strict_json(payload: bytes, *, label: str) -> object:
    if type(payload) is not bytes or not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{label} must be LF-terminated canonical JSON")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains invalid constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical compact JSON")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json(payload, label=label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _strict_jsonl(payload: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if type(payload) is not bytes or not payload:
        raise ValueError(f"{label} must be nonempty canonical JSON Lines")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        value = _strict_json(line, label=f"{label} row {index}")
        if type(value) is not dict:
            raise ValueError(f"{label} row {index} must be a JSON object")
        rows.append(value)
    return tuple(rows)


def _canonical_hex(value: object, *, label: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical binary64 hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not a binary64 hex string") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{label} is not canonical finite binary64 hex")
    if not PROBABILITY_CLIP <= parsed <= 1.0 - PROBABILITY_CLIP:
        raise ValueError(f"{label} is outside the accepted probability range")
    return parsed


def _require_frozen_rotation(value: object, *, label: str) -> RotationSpec:
    if type(value) is not RotationSpec:
        raise TypeError(f"{label} must be an exact RotationSpec")
    if type(value.outer_fold) is not int or type(value.pool_fold) is not int:
        raise TypeError(f"{label} folds must be exact integers")
    if value != rotation_by_id(value.rotation_id):
        raise ValueError(f"{label} differs from the frozen rotation registry")
    return value


def _require_frozen_run(value: object, *, label: str) -> PolicyRunSpec:
    if type(value) is not PolicyRunSpec:
        raise TypeError(f"{label} must be an exact PolicyRunSpec")
    _require_frozen_rotation(value.rotation, label=f"{label} rotation")
    if type(value.policy) is not str or (value.seed is not None and type(value.seed) is not int):
        raise TypeError(f"{label} policy and seed must use exact scalar types")
    if value != policy_run_by_track_id(value.track_id):
        raise ValueError(f"{label} differs from the frozen policy-run registry")
    return value


def _identity_document(value: SequentialV2PublicationIdentity) -> dict[str, object]:
    if type(value) is not SequentialV2PublicationIdentity:
        raise TypeError("outer update boundary requires an exact publication identity")
    return {
        "git_commit": value.git_commit,
        "code_manifest_sha256": value.code_manifest_sha256,
        "config_sha256": value.config_sha256,
        "lock_sha256": value.lock_sha256,
    }


def _identity_from_document(value: object) -> SequentialV2PublicationIdentity:
    raw = _exact_object(
        value,
        {"git_commit", "code_manifest_sha256", "config_sha256", "lock_sha256"},
        label="outer update publication identity",
    )
    if any(type(raw[key]) is not str for key in raw):
        raise ValueError("outer update publication identity fields must be exact text")
    return SequentialV2PublicationIdentity(
        git_commit=raw["git_commit"],
        code_manifest_sha256=raw["code_manifest_sha256"],
        config_sha256=raw["config_sha256"],
        lock_sha256=raw["lock_sha256"],
    )


def _rotation_from_document(value: object) -> RotationSpec:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "rotation_id",
            "outer_fold",
            "acquisition_pool_fold",
            "base_folds",
        },
        label="outer update rotation",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["rotation_id"]) is not str
        or type(raw["outer_fold"]) is not int
        or type(raw["acquisition_pool_fold"]) is not int
        or type(raw["base_folds"]) is not list
        or any(type(item) is not int for item in raw["base_folds"])
    ):
        raise ValueError("outer update rotation uses non-exact field types")
    spec = RotationSpec(raw["outer_fold"], raw["acquisition_pool_fold"])
    if canonical_json_bytes(raw) != canonical_json_bytes(spec.document()):
        raise ValueError("outer update rotation differs from the frozen document")
    return spec


def _run_from_document(value: object) -> PolicyRunSpec:
    raw = _exact_object(
        value,
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
        },
        label="outer update policy run",
    )
    if type(raw["track_id"]) is not str:
        raise ValueError("outer update policy run track ID must be exact text")
    run = policy_run_by_track_id(raw["track_id"])
    if canonical_json_bytes(raw) != canonical_json_bytes(run.document()):
        raise ValueError("outer update policy run differs from the frozen document")
    return run


def _payload_digest_inventory(
    value: object,
    *,
    expected_paths: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in value
    ):
        raise ValueError(f"{label} must be an exact immutable digest map")
    if tuple(path for path, _digest in value) != expected_paths or len(dict(value)) != len(value):
        raise ValueError(f"{label} has the wrong payload inventory")
    for path, digest in value:
        _sha256(digest, label=f"{label} {path}")
    return value


def _payload_digest(seal: PhaseSeal, path: str, *, label: str) -> str:
    matches = tuple(digest for current, digest in seal.payload_sha256 if current == path)
    if len(matches) != 1:
        raise ValueError(f"{label} lacks one exact payload digest for {path}")
    return _sha256(matches[0], label=f"{label} payload {path}")


def _reconstructed_leaf_seal_sha256(
    *,
    artifact: str,
    publication_identity: SequentialV2PublicationIdentity,
    scope_id: str,
    payload_sha256: tuple[tuple[str, str], ...],
    predecessor_seals: Mapping[str, str],
) -> str:
    payloads = dict(payload_sha256)
    receipt = canonical_json_bytes(
        {
            "artifact": artifact,
            "metadata": publication_identity.metadata(phase="update", scope_id=scope_id),
            "payloads": payloads,
            "predecessor_seals": dict(predecessor_seals),
            "schema_version": SCHEMA_VERSION,
            "status": "sealed",
        }
    )
    return sha256_bytes(checksum_manifest_bytes({**payloads, RECEIPT_NAME: sha256_bytes(receipt)}))


def outer_component_relative_path(spec: RotationSpec) -> str:
    frozen = _require_frozen_rotation(spec, label="outer component path rotation")
    return f"update/rotations/{frozen.rotation_id}/outer-components"


def outer_view_relative_path(run: PolicyRunSpec) -> str:
    frozen = _require_frozen_run(run, label="outer view path run")
    return f"update/tracks/{frozen.track_id}/outer-view"


def outer_evidence_relative_path(run: PolicyRunSpec) -> str:
    frozen = _require_frozen_run(run, label="outer evidence path run")
    return f"update/tracks/{frozen.track_id}/outer-evidence"


def _component_predecessors(
    *,
    spec: RotationSpec,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    state_leaf_seal_sha256s: tuple[str, ...],
) -> dict[str, str]:
    frozen = _require_frozen_rotation(spec, label="outer component predecessor rotation")
    expected_runs = policy_runs_for_rotation(frozen)
    if type(state_leaf_seal_sha256s) is not tuple or len(state_leaf_seal_sha256s) != len(
        expected_runs
    ):
        raise ValueError("outer component requires eleven ordered state-leaf digests")
    state_digests = tuple(
        _sha256(value, label=f"outer component state leaf {index}")
        for index, value in enumerate(state_leaf_seal_sha256s)
    )
    if len(set(state_digests)) != EXPECTED_TRACKS_PER_ROTATION:
        raise ValueError("outer component state-leaf digests must be distinct")
    result = {
        "protocol/SHA256SUMS": _sha256(protocol_seal_sha256, label="component protocol seal"),
        "stage/global/SHA256SUMS": _sha256(
            stage_global_seal_sha256,
            label="component stage-global seal",
        ),
        f"stage/rotations/{frozen.rotation_id}/outer-metadata/SHA256SUMS": _sha256(
            outer_metadata_leaf_seal_sha256,
            label="component outer-metadata leaf seal",
        ),
    }
    result.update(
        {
            f"update/tracks/{run.track_id}/state/SHA256SUMS": digest
            for run, digest in zip(expected_runs, state_digests, strict=True)
        }
    )
    if len(result) != 14:
        raise AssertionError("outer component predecessor census changed")
    return result


def _component_summary_document(
    *,
    spec: RotationSpec,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    state_leaf_seal_sha256s: tuple[str, ...],
    candidate_count: int,
    candidate_ids_sha256: str,
    component_count: int,
    component_membership_count: int,
    outer_components_payload_sha256: str,
) -> dict[str, object]:
    frozen = _require_frozen_rotation(spec, label="outer component summary rotation")
    state_digests = tuple(
        _sha256(value, label=f"outer component summary state leaf {index}")
        for index, value in enumerate(state_leaf_seal_sha256s)
    )
    if len(state_digests) != EXPECTED_TRACKS_PER_ROTATION:
        raise ValueError("outer component summary requires eleven state leaves")
    candidate_total = _exact_int(
        candidate_count,
        label="outer component candidate count",
        minimum=1,
    )
    if candidate_total != EXPECTED_SUPPORT_BY_FOLD[frozen.outer_fold]:
        raise ValueError("outer component candidate count differs from frozen outer support")
    components = _exact_int(component_count, label="outer component count", minimum=1)
    memberships = _exact_int(
        component_membership_count,
        label="outer component membership count",
        minimum=1,
    )
    if memberships != candidate_total or components > memberships:
        raise ValueError("outer component census is inconsistent")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": OUTER_COMPONENT_SUMMARY_ARTIFACT,
        "rotation": frozen.document(),
        "protocol_seal_sha256": _sha256(
            protocol_seal_sha256,
            label="outer component summary protocol seal",
        ),
        "stage_global_seal_sha256": _sha256(
            stage_global_seal_sha256,
            label="outer component summary stage-global seal",
        ),
        "outer_metadata_leaf_seal_sha256": _sha256(
            outer_metadata_leaf_seal_sha256,
            label="outer component summary metadata seal",
        ),
        "state_leaf_count": len(state_digests),
        "state_leaf_seal_sha256s": list(state_digests),
        "candidate_count": candidate_total,
        "candidate_ids_sha256": _sha256(
            candidate_ids_sha256,
            label="outer component summary candidate IDs",
        ),
        "component_count": components,
        "component_membership_count": memberships,
        "outer_components_payload_sha256": _sha256(
            outer_components_payload_sha256,
            label="outer component summary payload",
        ),
    }


@dataclass(frozen=True, slots=True)
class OuterComponentAttestation:
    """Canonical payload-free result from one isolated component worker."""

    spec: RotationSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    stage_global_seal_sha256: str
    outer_metadata_leaf_seal_sha256: str
    state_leaf_seal_sha256s: tuple[str, ...]
    component_leaf_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    candidate_count: int
    candidate_ids_sha256: str
    component_count: int
    component_membership_count: int

    def __post_init__(self) -> None:
        spec = _require_frozen_rotation(self.spec, label="outer component attestation rotation")
        _identity_document(self.publication_identity)
        _sha256(self.protocol_seal_sha256, label="component attestation protocol seal")
        _sha256(self.stage_global_seal_sha256, label="component attestation stage seal")
        _sha256(
            self.outer_metadata_leaf_seal_sha256,
            label="component attestation metadata seal",
        )
        state_digests = tuple(
            _sha256(value, label=f"component attestation state leaf {index}")
            for index, value in enumerate(self.state_leaf_seal_sha256s)
        )
        if (
            type(self.state_leaf_seal_sha256s) is not tuple
            or len(state_digests) != EXPECTED_TRACKS_PER_ROTATION
            or len(set(state_digests)) != EXPECTED_TRACKS_PER_ROTATION
        ):
            raise ValueError("component attestation requires eleven distinct ordered state seals")
        leaf_digest = _sha256(
            self.component_leaf_seal_sha256,
            label="component attestation leaf seal",
        )
        payloads = _payload_digest_inventory(
            self.payload_sha256,
            expected_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            label="component attestation payload digests",
        )
        expected_summary = _component_summary_document(
            spec=spec,
            protocol_seal_sha256=self.protocol_seal_sha256,
            stage_global_seal_sha256=self.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
            state_leaf_seal_sha256s=state_digests,
            candidate_count=self.candidate_count,
            candidate_ids_sha256=self.candidate_ids_sha256,
            component_count=self.component_count,
            component_membership_count=self.component_membership_count,
            outer_components_payload_sha256=dict(payloads)["outer-components.jsonl"],
        )
        if dict(payloads)["components-summary.json"] != sha256_bytes(
            canonical_json_bytes(expected_summary)
        ):
            raise ValueError("component attestation census differs from its summary digest")
        expected_leaf = _reconstructed_leaf_seal_sha256(
            artifact=OUTER_COMPONENTS_ARTIFACT,
            publication_identity=self.publication_identity,
            scope_id=spec.rotation_id,
            payload_sha256=payloads,
            predecessor_seals=_component_predecessors(
                spec=spec,
                protocol_seal_sha256=self.protocol_seal_sha256,
                stage_global_seal_sha256=self.stage_global_seal_sha256,
                outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
                state_leaf_seal_sha256s=state_digests,
            ),
        )
        if leaf_digest != expected_leaf:
            raise ValueError("component attestation does not reconstruct its authoritative leaf")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": OUTER_COMPONENT_ATTESTATION_ARTIFACT,
            "rotation": self.spec.document(),
            "publication_identity": _identity_document(self.publication_identity),
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "stage_global_seal_sha256": self.stage_global_seal_sha256,
            "outer_metadata_leaf_seal_sha256": self.outer_metadata_leaf_seal_sha256,
            "state_leaf_seal_sha256s": list(self.state_leaf_seal_sha256s),
            "component_leaf_seal_sha256": self.component_leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "candidate_count": self.candidate_count,
            "candidate_ids_sha256": self.candidate_ids_sha256,
            "component_count": self.component_count,
            "component_membership_count": self.component_membership_count,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())

    def component_index_document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "outer_components",
            "rotation": self.spec.document(),
            "relative_path": outer_component_relative_path(self.spec),
            "leaf_artifact": OUTER_COMPONENTS_ARTIFACT,
            "leaf_seal_sha256": self.component_leaf_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "candidate_count": self.candidate_count,
            "candidate_ids_sha256": self.candidate_ids_sha256,
            "component_count": self.component_count,
            "component_membership_count": self.component_membership_count,
        }


def outer_component_attestation_from_document(value: object) -> OuterComponentAttestation:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "rotation",
            "publication_identity",
            "protocol_seal_sha256",
            "stage_global_seal_sha256",
            "outer_metadata_leaf_seal_sha256",
            "state_leaf_seal_sha256s",
            "component_leaf_seal_sha256",
            "payload_sha256",
            "candidate_count",
            "candidate_ids_sha256",
            "component_count",
            "component_membership_count",
        },
        label="outer component attestation",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["artifact"]) is not str
        or raw["artifact"] != OUTER_COMPONENT_ATTESTATION_ARTIFACT
        or type(raw["state_leaf_seal_sha256s"]) is not list
        or any(type(item) is not str for item in raw["state_leaf_seal_sha256s"])
    ):
        raise ValueError("outer component attestation identity or state-seal array changed")
    payload_raw = _exact_object(
        raw["payload_sha256"],
        set(OUTER_COMPONENT_PAYLOAD_PATHS),
        label="outer component attestation payload digests",
    )
    if any(
        type(path) is not str or type(digest) is not str for path, digest in payload_raw.items()
    ):
        raise ValueError("outer component attestation payload map must contain exact text")
    attestation = OuterComponentAttestation(
        spec=_rotation_from_document(raw["rotation"]),
        publication_identity=_identity_from_document(raw["publication_identity"]),
        protocol_seal_sha256=_sha256(raw["protocol_seal_sha256"], label="protocol seal"),
        stage_global_seal_sha256=_sha256(
            raw["stage_global_seal_sha256"],
            label="stage-global seal",
        ),
        outer_metadata_leaf_seal_sha256=_sha256(
            raw["outer_metadata_leaf_seal_sha256"],
            label="outer-metadata leaf seal",
        ),
        state_leaf_seal_sha256s=tuple(raw["state_leaf_seal_sha256s"]),
        component_leaf_seal_sha256=_sha256(
            raw["component_leaf_seal_sha256"],
            label="component leaf seal",
        ),
        payload_sha256=tuple((path, payload_raw[path]) for path in OUTER_COMPONENT_PAYLOAD_PATHS),
        candidate_count=_exact_int(raw["candidate_count"], label="candidate count", minimum=1),
        candidate_ids_sha256=_sha256(raw["candidate_ids_sha256"], label="candidate IDs"),
        component_count=_exact_int(raw["component_count"], label="component count", minimum=1),
        component_membership_count=_exact_int(
            raw["component_membership_count"],
            label="component membership count",
            minimum=1,
        ),
    )
    if canonical_json_bytes(attestation.document()) != canonical_json_bytes(raw):
        raise ValueError("outer component attestation does not round-trip exactly")
    return attestation


def outer_component_attestation_from_bytes(payload: bytes) -> OuterComponentAttestation:
    return outer_component_attestation_from_document(
        _strict_json_object(payload, label="outer component attestation")
    )


@dataclass(frozen=True, slots=True)
class OuterComponentCapability:
    """One decoded component leaf after all external authorities were checked."""

    spec: RotationSpec
    component_set: OuterComponentSet
    state_leaf_seal_sha256s: tuple[str, ...]
    outer_metadata_leaf_seal_sha256: str
    component_leaf_seal_sha256: str
    outer_components_payload_sha256: str

    def __post_init__(self) -> None:
        spec = _require_frozen_rotation(self.spec, label="outer component capability rotation")
        if type(self.component_set) is not OuterComponentSet or self.component_set.spec != spec:
            raise ValueError("outer component capability has the wrong component set")
        _component_predecessors(
            spec=spec,
            protocol_seal_sha256="0" * 64,
            stage_global_seal_sha256="1" * 64,
            outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
            state_leaf_seal_sha256s=self.state_leaf_seal_sha256s,
        )
        _sha256(self.component_leaf_seal_sha256, label="component capability leaf seal")
        _sha256(
            self.outer_components_payload_sha256,
            label="component capability payload",
        )


def _validate_component_state_attestations(
    *,
    spec: RotationSpec,
    state_attestations: tuple[object, ...],
    expected_state_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    # Local import avoids an import cycle while the state boundary imports the
    # numerical update core shared by this module.
    from amp_challenge.evaluation.sequential_v2_update_state import (
        UpdateStateAttestation,
        update_state_attestation_from_document,
    )

    frozen = _require_frozen_rotation(spec, label="component state barrier rotation")
    _identity_document(publication_identity)
    protocol_digest = _sha256(protocol_seal_sha256, label="component state protocol seal")
    expected_runs = policy_runs_for_rotation(frozen)
    if type(state_attestations) is not tuple or type(expected_state_leaf_seal_sha256s) is not tuple:
        raise TypeError("component state attestations and digests must be exact tuples")
    if (
        len(state_attestations) != EXPECTED_TRACKS_PER_ROTATION
        or any(type(item) is not UpdateStateAttestation for item in state_attestations)
        or tuple(item.run for item in state_attestations) != expected_runs
        or len(expected_state_leaf_seal_sha256s) != EXPECTED_TRACKS_PER_ROTATION
    ):
        raise ValueError("component worker requires eleven state attestations in frozen order")
    expected_digests = tuple(
        _sha256(value, label=f"expected component state leaf {index}")
        for index, value in enumerate(expected_state_leaf_seal_sha256s)
    )
    if len(set(expected_digests)) != EXPECTED_TRACKS_PER_ROTATION:
        raise ValueError("component state digest sequence must contain eleven distinct values")
    for index, (item, expected_digest) in enumerate(
        zip(state_attestations, expected_digests, strict=True)
    ):
        reconstructed = update_state_attestation_from_document(item.document())
        if reconstructed.canonical_bytes() != item.canonical_bytes():
            raise ValueError(f"state attestation {index} changed during strict reconstruction")
        if (
            item.publication_identity != publication_identity
            or item.protocol_seal_sha256 != protocol_digest
            or item.state_leaf_seal_sha256 != expected_digest
        ):
            raise ValueError("component state attestation differs from controller authority")
    if (
        len({item.prepare_global_seal_sha256 for item in state_attestations}) != 1
        or len({item.reveal_global_seal_sha256 for item in state_attestations}) != 1
    ):
        raise ValueError("rotation state attestations do not share prepare/reveal authorities")
    return state_attestations, expected_digests


def _component_worker_authorities(
    *,
    spec: RotationSpec,
    state_attestations: tuple[object, ...],
    expected_state_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_stage_global_seal_sha256: str,
) -> tuple[
    ProtocolCapability, StageManifestCapability, OuterMetadataCapability, tuple[str, ...], str
]:
    frozen = _require_frozen_rotation(spec, label="component worker rotation")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("component worker requires an exact ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    _items, state_digests = _validate_component_state_attestations(
        spec=frozen,
        state_attestations=state_attestations,
        expected_state_leaf_seal_sha256s=expected_state_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
    )
    # No metadata leaf is opened until all eleven state attestations above have
    # been strictly reconstructed and matched to an independent digest vector.
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("component worker requires an exact StageManifestCapability")
    stage_digest = _sha256(
        expected_stage_global_seal_sha256,
        label="expected component stage-global seal",
    )
    stage = verify_stage_manifest_capability(
        stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_digest,
    )
    metadata_entry = stage.leaf(spec=frozen, role=OUTER_METADATA_ROLE)
    metadata = outer_metadata_capability_from_stage_capabilities(
        stage,
        outer_metadata_capsule,
        spec=frozen,
        expected_stage_global_seal_sha256=stage_digest,
    )
    return protocol, stage, metadata, state_digests, metadata_entry.leaf_seal_sha256


def _component_attestation_from_verified_leaf(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    state_leaf_seal_sha256s: tuple[str, ...],
    component_set: OuterComponentSet,
) -> OuterComponentAttestation:
    return OuterComponentAttestation(
        spec=spec,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        outer_metadata_leaf_seal_sha256=outer_metadata_leaf_seal_sha256,
        state_leaf_seal_sha256s=state_leaf_seal_sha256s,
        component_leaf_seal_sha256=seal.seal_sha256,
        payload_sha256=seal.payload_sha256,
        candidate_count=len(component_set.support_sequence_ids),
        candidate_ids_sha256=id_stream_sha256(component_set.support_sequence_ids),
        component_count=len(component_set.components),
        component_membership_count=sum(
            len(component.sequence_ids) for component in component_set.components
        ),
    )


def _decode_component_rows(
    payload: bytes,
    *,
    spec: RotationSpec,
) -> tuple[OuterComponent, ...]:
    rows = _strict_jsonl(payload, label="outer component rows")
    components: list[OuterComponent] = []
    for index, raw in enumerate(rows):
        row = _exact_object(
            raw,
            {
                "schema_version",
                "rotation_id",
                "role",
                "fold",
                "diversity_component_id",
                "sequence_ids",
            },
            label=f"outer component row {index}",
        )
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != SCHEMA_VERSION
            or type(row["rotation_id"]) is not str
            or row["rotation_id"] != spec.rotation_id
            or type(row["role"]) is not str
            or row["role"] != "outer"
            or type(row["fold"]) is not int
            or row["fold"] != spec.outer_fold
            or type(row["diversity_component_id"]) is not str
            or _COMPONENT_ID.fullmatch(row["diversity_component_id"]) is None
            or type(row["sequence_ids"]) is not list
            or any(type(item) is not str for item in row["sequence_ids"])
        ):
            raise ValueError("outer component row identity or field types changed")
        component = OuterComponent(
            component_id=row["diversity_component_id"],
            sequence_ids=tuple(row["sequence_ids"]),
        )
        if canonical_json_bytes(component.document(spec=spec)) != canonical_json_bytes(row):
            raise ValueError("outer component row does not round-trip exactly")
        components.append(component)
    result = tuple(components)
    if result != tuple(sorted(result, key=lambda item: item.component_id)):
        raise ValueError("outer component rows are not in ascending component-ID order")
    return result


def _decode_outer_component_leaf(
    seal: PhaseSeal,
    *,
    attestation: OuterComponentAttestation,
    outer_metadata: OuterMetadataCapability,
    expected_component_leaf_seal_sha256: str,
) -> OuterComponentCapability:
    if type(attestation) is not OuterComponentAttestation:
        raise TypeError("component decoder requires an exact OuterComponentAttestation")
    strict = outer_component_attestation_from_document(attestation.document())
    if strict.canonical_bytes() != attestation.canonical_bytes():
        raise ValueError("component attestation changed during strict reconstruction")
    expected_leaf = _sha256(
        expected_component_leaf_seal_sha256,
        label="expected component leaf seal",
    )
    if attestation.component_leaf_seal_sha256 != expected_leaf:
        raise ValueError("component attestation differs from controller leaf authority")
    verified = verify_phase_capability(
        seal,
        expected_artifact=OUTER_COMPONENTS_ARTIFACT,
        expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
        expected_predecessor_seals=_component_predecessors(
            spec=attestation.spec,
            protocol_seal_sha256=attestation.protocol_seal_sha256,
            stage_global_seal_sha256=attestation.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=attestation.outer_metadata_leaf_seal_sha256,
            state_leaf_seal_sha256s=attestation.state_leaf_seal_sha256s,
        ),
        expected_seal_sha256=expected_leaf,
    )
    attestation.publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id=attestation.spec.rotation_id,
    )
    if verified.payload_sha256 != attestation.payload_sha256:
        raise ValueError("component leaf payload digests differ from its attestation")
    # All capability, digest, predecessor, and receipt checks above precede the
    # first component payload read.
    component_payload = verified.read_payload_bytes("outer-components.jsonl")
    components = _decode_component_rows(component_payload, spec=attestation.spec)
    component_set = OuterComponentSet(
        spec=attestation.spec,
        support_sequence_ids=outer_metadata.support_sequence_ids,
        components=components,
    )
    expected_set = build_outer_component_set(outer_metadata)
    if component_set != expected_set or component_payload != canonical_jsonl_bytes(
        expected_set.documents()
    ):
        raise ValueError("outer component leaf differs from exact metadata clustering")
    expected_summary = _component_summary_document(
        spec=attestation.spec,
        protocol_seal_sha256=attestation.protocol_seal_sha256,
        stage_global_seal_sha256=attestation.stage_global_seal_sha256,
        outer_metadata_leaf_seal_sha256=attestation.outer_metadata_leaf_seal_sha256,
        state_leaf_seal_sha256s=attestation.state_leaf_seal_sha256s,
        candidate_count=len(component_set.support_sequence_ids),
        candidate_ids_sha256=id_stream_sha256(component_set.support_sequence_ids),
        component_count=len(component_set.components),
        component_membership_count=sum(
            len(component.sequence_ids) for component in component_set.components
        ),
        outer_components_payload_sha256=sha256_bytes(component_payload),
    )
    summary_payload = verified.read_payload_bytes("components-summary.json")
    _strict_json_object(summary_payload, label="outer component summary")
    if summary_payload != canonical_json_bytes(expected_summary):
        raise ValueError("outer component summary differs from its authenticated contents")
    return OuterComponentCapability(
        spec=attestation.spec,
        component_set=component_set,
        state_leaf_seal_sha256s=attestation.state_leaf_seal_sha256s,
        outer_metadata_leaf_seal_sha256=attestation.outer_metadata_leaf_seal_sha256,
        component_leaf_seal_sha256=verified.seal_sha256,
        outer_components_payload_sha256=sha256_bytes(component_payload),
    )


def publish_outer_component(
    destination: str | Path,
    *,
    spec: RotationSpec,
    state_attestations: tuple[object, ...],
    expected_state_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_stage_global_seal_sha256: str,
) -> OuterComponentAttestation:
    """Publish one component leaf after the exact eleven-state temporal barrier."""

    protocol, stage, metadata, state_digests, metadata_leaf_digest = _component_worker_authorities(
        spec=spec,
        state_attestations=state_attestations,
        expected_state_leaf_seal_sha256s=expected_state_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    component_set = build_outer_component_set(metadata)
    component_payload = canonical_jsonl_bytes(component_set.documents())
    summary = _component_summary_document(
        spec=component_set.spec,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        outer_metadata_leaf_seal_sha256=metadata_leaf_digest,
        state_leaf_seal_sha256s=state_digests,
        candidate_count=len(component_set.support_sequence_ids),
        candidate_ids_sha256=id_stream_sha256(component_set.support_sequence_ids),
        component_count=len(component_set.components),
        component_membership_count=sum(
            len(component.sequence_ids) for component in component_set.components
        ),
        outer_components_payload_sha256=sha256_bytes(component_payload),
    )
    seal = publish_phase(
        destination,
        artifact=OUTER_COMPONENTS_ARTIFACT,
        payloads={
            "components-summary.json": canonical_json_bytes(summary),
            "outer-components.jsonl": component_payload,
        },
        predecessor_seals=_component_predecessors(
            spec=component_set.spec,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            stage_global_seal_sha256=stage.seal.seal_sha256,
            outer_metadata_leaf_seal_sha256=metadata_leaf_digest,
            state_leaf_seal_sha256s=state_digests,
        ),
        metadata=publication_identity.metadata(
            phase="update",
            scope_id=component_set.spec.rotation_id,
        ),
    )
    attestation = _component_attestation_from_verified_leaf(
        seal,
        spec=component_set.spec,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        outer_metadata_leaf_seal_sha256=metadata_leaf_digest,
        state_leaf_seal_sha256s=state_digests,
        component_set=component_set,
    )
    _decode_outer_component_leaf(
        seal,
        attestation=attestation,
        outer_metadata=metadata,
        expected_component_leaf_seal_sha256=seal.seal_sha256,
    )
    return attestation


def verify_outer_component_phase_capability(
    seal: PhaseSeal,
    *,
    spec: RotationSpec,
    state_attestations: tuple[object, ...],
    expected_state_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_stage_global_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
) -> OuterComponentAttestation:
    """Reverify one component worker leaf and release only its safe attestation."""

    protocol, stage, metadata, state_digests, metadata_leaf_digest = _component_worker_authorities(
        spec=spec,
        state_attestations=state_attestations,
        expected_state_leaf_seal_sha256s=expected_state_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    expected_leaf = _sha256(
        expected_component_leaf_seal_sha256,
        label="expected verified component leaf",
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=OUTER_COMPONENTS_ARTIFACT,
        expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
        expected_predecessor_seals=_component_predecessors(
            spec=spec,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            stage_global_seal_sha256=stage.seal.seal_sha256,
            outer_metadata_leaf_seal_sha256=metadata_leaf_digest,
            state_leaf_seal_sha256s=state_digests,
        ),
        expected_seal_sha256=expected_leaf,
    )
    publication_identity.verify_metadata(
        verified.metadata_json,
        phase="update",
        scope_id=spec.rotation_id,
    )
    component_payload = verified.read_payload_bytes("outer-components.jsonl")
    components = _decode_component_rows(component_payload, spec=spec)
    component_set = OuterComponentSet(
        spec=spec,
        support_sequence_ids=metadata.support_sequence_ids,
        components=components,
    )
    attestation = _component_attestation_from_verified_leaf(
        verified,
        spec=spec,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        outer_metadata_leaf_seal_sha256=metadata_leaf_digest,
        state_leaf_seal_sha256s=state_digests,
        component_set=component_set,
    )
    _decode_outer_component_leaf(
        verified,
        attestation=attestation,
        outer_metadata=metadata,
        expected_component_leaf_seal_sha256=expected_leaf,
    )
    return attestation


def outer_component_from_authorities(
    seal: PhaseSeal,
    *,
    attestation: OuterComponentAttestation,
    spec: RotationSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_stage_global_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
) -> tuple[OuterComponentCapability, OuterMetadataCapability]:
    """Authorize one component leaf for a projection without any model payload."""

    frozen = _require_frozen_rotation(spec, label="component capability lookup rotation")
    if type(attestation) is not OuterComponentAttestation or attestation.spec != frozen:
        raise ValueError("component capability attestation belongs to another rotation")
    strict_attestation = outer_component_attestation_from_document(attestation.document())
    if strict_attestation.canonical_bytes() != attestation.canonical_bytes():
        raise ValueError("component capability attestation changed during reconstruction")
    expected_component = _sha256(
        expected_component_leaf_seal_sha256,
        label="expected component capability leaf seal",
    )
    if attestation.component_leaf_seal_sha256 != expected_component:
        raise ValueError("component attestation differs from controller leaf authority")
    if attestation.publication_identity != publication_identity:
        raise ValueError("component capability has the wrong publication identity")
    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("component capability requires an exact ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    stage_digest = _sha256(
        expected_stage_global_seal_sha256,
        label="expected component capability stage-global seal",
    )
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("component capability requires an exact StageManifestCapability")
    stage = verify_stage_manifest_capability(
        stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_digest,
    )
    metadata_entry = stage.leaf(spec=frozen, role=OUTER_METADATA_ROLE)
    if (
        attestation.protocol_seal_sha256 != protocol.seal.seal_sha256
        or attestation.stage_global_seal_sha256 != stage_digest
        or attestation.outer_metadata_leaf_seal_sha256 != metadata_entry.leaf_seal_sha256
    ):
        raise ValueError("component attestation differs from protocol/stage authority")
    # Authenticate the component envelope and exact digest before opening either
    # the component payload or the metadata payload.
    verify_phase_capability(
        seal,
        expected_artifact=OUTER_COMPONENTS_ARTIFACT,
        expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
        expected_predecessor_seals=_component_predecessors(
            spec=frozen,
            protocol_seal_sha256=attestation.protocol_seal_sha256,
            stage_global_seal_sha256=attestation.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=attestation.outer_metadata_leaf_seal_sha256,
            state_leaf_seal_sha256s=attestation.state_leaf_seal_sha256s,
        ),
        expected_seal_sha256=expected_component,
    )
    metadata = outer_metadata_capability_from_stage_capabilities(
        stage,
        outer_metadata_capsule,
        spec=frozen,
        expected_stage_global_seal_sha256=stage_digest,
    )
    capability = _decode_outer_component_leaf(
        seal,
        attestation=attestation,
        outer_metadata=metadata,
        expected_component_leaf_seal_sha256=expected_component,
    )
    return capability, metadata


def _view_predecessors(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    component_leaf_seal_sha256: str,
    state_leaf_seal_sha256: str,
) -> dict[str, str]:
    frozen = _require_frozen_run(run, label="outer-view predecessor run")
    result = {
        "protocol/SHA256SUMS": _sha256(protocol_seal_sha256, label="view protocol seal"),
        "stage/global/SHA256SUMS": _sha256(
            stage_global_seal_sha256,
            label="view stage-global seal",
        ),
        f"stage/rotations/{frozen.rotation.rotation_id}/outer-metadata/SHA256SUMS": _sha256(
            outer_metadata_leaf_seal_sha256,
            label="view outer-metadata leaf seal",
        ),
        f"update/rotations/{frozen.rotation.rotation_id}/outer-components/SHA256SUMS": _sha256(
            component_leaf_seal_sha256,
            label="view component leaf seal",
        ),
        f"update/tracks/{frozen.track_id}/state/SHA256SUMS": _sha256(
            state_leaf_seal_sha256,
            label="view state leaf seal",
        ),
    }
    if len(result) != 5:
        raise AssertionError("outer-view predecessor census changed")
    return result


def _evidence_predecessors(
    *,
    run: PolicyRunSpec,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    component_leaf_seal_sha256: str,
    state_leaf_seal_sha256: str,
    outer_view_leaf_seal_sha256: str,
) -> dict[str, str]:
    frozen = _require_frozen_run(run, label="outer-evidence predecessor run")
    result = _view_predecessors(
        run=frozen,
        protocol_seal_sha256=protocol_seal_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        outer_metadata_leaf_seal_sha256=outer_metadata_leaf_seal_sha256,
        component_leaf_seal_sha256=component_leaf_seal_sha256,
        state_leaf_seal_sha256=state_leaf_seal_sha256,
    )
    result[f"{outer_view_relative_path(frozen)}/SHA256SUMS"] = _sha256(
        outer_view_leaf_seal_sha256,
        label="evidence outer-view leaf seal",
    )
    if len(result) != 6:
        raise AssertionError("outer-evidence predecessor census changed")
    return result


def _view_summary_document(
    *,
    run: PolicyRunSpec,
    state_leaf_seal_sha256: str,
    outer_component_leaf_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    candidate_count: int,
    candidate_ids_sha256: str,
    candidate_payload_sha256: str,
) -> dict[str, object]:
    frozen = _require_frozen_run(run, label="outer-view summary run")
    count = _exact_int(candidate_count, label="outer-view candidate count", minimum=1)
    if count != EXPECTED_SUPPORT_BY_FOLD[frozen.rotation.outer_fold]:
        raise ValueError("outer-view candidate count differs from frozen outer support")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": OUTER_VIEW_SUMMARY_ARTIFACT,
        "run": frozen.document(),
        "view_kind": "outer_mean",
        "state_leaf_seal_sha256": _sha256(
            state_leaf_seal_sha256,
            label="outer-view summary state seal",
        ),
        "outer_component_leaf_seal_sha256": _sha256(
            outer_component_leaf_seal_sha256,
            label="outer-view summary component seal",
        ),
        "outer_metadata_leaf_seal_sha256": _sha256(
            outer_metadata_leaf_seal_sha256,
            label="outer-view summary metadata seal",
        ),
        "candidate_count": count,
        "candidate_ids_sha256": _sha256(
            candidate_ids_sha256,
            label="outer-view summary candidate IDs",
        ),
        "candidate_payload_sha256": _sha256(
            candidate_payload_sha256,
            label="outer-view summary candidate payload",
        ),
        "field_names": list(_OUTER_VIEW_FIELDS),
    }


def _evidence_summary_document(
    *,
    run: PolicyRunSpec,
    state_leaf_seal_sha256: str,
    updated_model_payload_sha256: str,
    stage_global_seal_sha256: str,
    outer_metadata_leaf_seal_sha256: str,
    outer_component_leaf_seal_sha256: str,
    outer_components_payload_sha256: str,
    outer_view_leaf_seal_sha256: str,
    outer_view_candidates_payload_sha256: str,
    outer_context_count: int,
    outer_example_ids_sha256: str,
    outer_context_predictions_payload_sha256: str,
    outer_sequence_prediction_count: int,
    outer_candidate_ids_sha256: str,
    outer_sequence_predictions_payload_sha256: str,
    outer_component_count: int,
    outer_component_membership_count: int,
) -> dict[str, object]:
    frozen = _require_frozen_run(run, label="outer-evidence summary run")
    context_count = _exact_int(
        outer_context_count,
        label="outer-evidence context count",
        minimum=1,
    )
    candidate_count = _exact_int(
        outer_sequence_prediction_count,
        label="outer-evidence sequence count",
        minimum=1,
    )
    component_count = _exact_int(
        outer_component_count,
        label="outer-evidence component count",
        minimum=1,
    )
    membership_count = _exact_int(
        outer_component_membership_count,
        label="outer-evidence membership count",
        minimum=1,
    )
    if (
        context_count != EXPECTED_CONTEXTS_BY_FOLD[frozen.rotation.outer_fold]
        or candidate_count != EXPECTED_SUPPORT_BY_FOLD[frozen.rotation.outer_fold]
        or membership_count != candidate_count
        or component_count > membership_count
    ):
        raise ValueError("outer-evidence census differs from the frozen outer fold")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": OUTER_EVIDENCE_SUMMARY_ARTIFACT,
        "run": frozen.document(),
        "state_leaf_seal_sha256": _sha256(
            state_leaf_seal_sha256,
            label="outer-evidence summary state seal",
        ),
        "updated_model_payload_sha256": _sha256(
            updated_model_payload_sha256,
            label="outer-evidence summary model payload",
        ),
        "stage_global_seal_sha256": _sha256(
            stage_global_seal_sha256,
            label="outer-evidence summary stage seal",
        ),
        "outer_metadata_leaf_seal_sha256": _sha256(
            outer_metadata_leaf_seal_sha256,
            label="outer-evidence summary metadata seal",
        ),
        "outer_component_leaf_seal_sha256": _sha256(
            outer_component_leaf_seal_sha256,
            label="outer-evidence summary component seal",
        ),
        "outer_components_payload_sha256": _sha256(
            outer_components_payload_sha256,
            label="outer-evidence summary components payload",
        ),
        "outer_view_leaf_seal_sha256": _sha256(
            outer_view_leaf_seal_sha256,
            label="outer-evidence summary view seal",
        ),
        "outer_view_candidates_payload_sha256": _sha256(
            outer_view_candidates_payload_sha256,
            label="outer-evidence summary view payload",
        ),
        "outer_context_count": context_count,
        "outer_example_ids_sha256": _sha256(
            outer_example_ids_sha256,
            label="outer-evidence summary example IDs",
        ),
        "outer_context_predictions_payload_sha256": _sha256(
            outer_context_predictions_payload_sha256,
            label="outer-evidence summary context payload",
        ),
        "outer_sequence_prediction_count": candidate_count,
        "outer_candidate_ids_sha256": _sha256(
            outer_candidate_ids_sha256,
            label="outer-evidence summary candidate IDs",
        ),
        "outer_sequence_predictions_payload_sha256": _sha256(
            outer_sequence_predictions_payload_sha256,
            label="outer-evidence summary sequence payload",
        ),
        "outer_component_count": component_count,
        "outer_component_membership_count": membership_count,
    }


@dataclass(frozen=True, slots=True)
class OuterProjectionAttestation:
    """One payload-free result binding a projection worker's two leaves."""

    run: PolicyRunSpec
    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    stage_global_seal_sha256: str
    outer_metadata_leaf_seal_sha256: str
    state_leaf_seal_sha256: str
    updated_model_payload_sha256: str
    outer_component_leaf_seal_sha256: str
    outer_components_payload_sha256: str
    outer_view_leaf_seal_sha256: str
    outer_view_payload_sha256: tuple[tuple[str, str], ...]
    outer_evidence_leaf_seal_sha256: str
    outer_evidence_payload_sha256: tuple[tuple[str, str], ...]
    outer_context_count: int
    outer_example_ids_sha256: str
    outer_candidate_count: int
    outer_candidate_ids_sha256: str
    outer_component_count: int
    outer_component_membership_count: int

    def __post_init__(self) -> None:
        run = _require_frozen_run(self.run, label="outer projection attestation run")
        _identity_document(self.publication_identity)
        for label, value in (
            ("protocol seal", self.protocol_seal_sha256),
            ("stage-global seal", self.stage_global_seal_sha256),
            ("outer-metadata leaf seal", self.outer_metadata_leaf_seal_sha256),
            ("state leaf seal", self.state_leaf_seal_sha256),
            ("updated model payload", self.updated_model_payload_sha256),
            ("outer-component leaf seal", self.outer_component_leaf_seal_sha256),
            ("outer-components payload", self.outer_components_payload_sha256),
            ("outer-view leaf seal", self.outer_view_leaf_seal_sha256),
            ("outer-evidence leaf seal", self.outer_evidence_leaf_seal_sha256),
            ("outer example IDs", self.outer_example_ids_sha256),
            ("outer candidate IDs", self.outer_candidate_ids_sha256),
        ):
            _sha256(value, label=f"outer projection attestation {label}")
        if (
            len(
                {
                    self.state_leaf_seal_sha256,
                    self.outer_component_leaf_seal_sha256,
                    self.outer_view_leaf_seal_sha256,
                    self.outer_evidence_leaf_seal_sha256,
                }
            )
            != 4
        ):
            raise ValueError("projection source and output leaf seals must be distinct")
        view_payloads = _payload_digest_inventory(
            self.outer_view_payload_sha256,
            expected_paths=OUTER_VIEW_PAYLOAD_PATHS,
            label="outer projection view payload digests",
        )
        evidence_payloads = _payload_digest_inventory(
            self.outer_evidence_payload_sha256,
            expected_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
            label="outer projection evidence payload digests",
        )
        context_count = _exact_int(
            self.outer_context_count,
            label="outer projection context count",
            minimum=1,
        )
        candidate_count = _exact_int(
            self.outer_candidate_count,
            label="outer projection candidate count",
            minimum=1,
        )
        component_count = _exact_int(
            self.outer_component_count,
            label="outer projection component count",
            minimum=1,
        )
        membership_count = _exact_int(
            self.outer_component_membership_count,
            label="outer projection membership count",
            minimum=1,
        )
        if (
            context_count != EXPECTED_CONTEXTS_BY_FOLD[run.rotation.outer_fold]
            or candidate_count != EXPECTED_SUPPORT_BY_FOLD[run.rotation.outer_fold]
            or membership_count != candidate_count
            or component_count > candidate_count
        ):
            raise ValueError("outer projection attestation census changed")

        expected_view_summary = _view_summary_document(
            run=run,
            state_leaf_seal_sha256=self.state_leaf_seal_sha256,
            outer_component_leaf_seal_sha256=self.outer_component_leaf_seal_sha256,
            outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
            candidate_count=candidate_count,
            candidate_ids_sha256=self.outer_candidate_ids_sha256,
            candidate_payload_sha256=dict(view_payloads)["candidates.jsonl"],
        )
        if dict(view_payloads)["view-summary.json"] != sha256_bytes(
            canonical_json_bytes(expected_view_summary)
        ):
            raise ValueError("projection attestation differs from its view summary digest")
        expected_view_seal = _reconstructed_leaf_seal_sha256(
            artifact=OUTER_VIEW_ARTIFACT,
            publication_identity=self.publication_identity,
            scope_id=run.track_id,
            payload_sha256=view_payloads,
            predecessor_seals=_view_predecessors(
                run=run,
                protocol_seal_sha256=self.protocol_seal_sha256,
                stage_global_seal_sha256=self.stage_global_seal_sha256,
                outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
                component_leaf_seal_sha256=self.outer_component_leaf_seal_sha256,
                state_leaf_seal_sha256=self.state_leaf_seal_sha256,
            ),
        )
        if expected_view_seal != self.outer_view_leaf_seal_sha256:
            raise ValueError("projection attestation does not reconstruct its view leaf")

        expected_evidence_summary = _evidence_summary_document(
            run=run,
            state_leaf_seal_sha256=self.state_leaf_seal_sha256,
            updated_model_payload_sha256=self.updated_model_payload_sha256,
            stage_global_seal_sha256=self.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
            outer_component_leaf_seal_sha256=self.outer_component_leaf_seal_sha256,
            outer_components_payload_sha256=self.outer_components_payload_sha256,
            outer_view_leaf_seal_sha256=self.outer_view_leaf_seal_sha256,
            outer_view_candidates_payload_sha256=dict(view_payloads)["candidates.jsonl"],
            outer_context_count=context_count,
            outer_example_ids_sha256=self.outer_example_ids_sha256,
            outer_context_predictions_payload_sha256=dict(evidence_payloads)[
                "outer-context-predictions.jsonl"
            ],
            outer_sequence_prediction_count=candidate_count,
            outer_candidate_ids_sha256=self.outer_candidate_ids_sha256,
            outer_sequence_predictions_payload_sha256=dict(evidence_payloads)[
                "outer-sequence-predictions.jsonl"
            ],
            outer_component_count=component_count,
            outer_component_membership_count=membership_count,
        )
        if dict(evidence_payloads)["outer-evidence-summary.json"] != sha256_bytes(
            canonical_json_bytes(expected_evidence_summary)
        ):
            raise ValueError("projection attestation differs from its evidence summary digest")
        expected_evidence_seal = _reconstructed_leaf_seal_sha256(
            artifact=OUTER_EVIDENCE_ARTIFACT,
            publication_identity=self.publication_identity,
            scope_id=run.track_id,
            payload_sha256=evidence_payloads,
            predecessor_seals=_evidence_predecessors(
                run=run,
                protocol_seal_sha256=self.protocol_seal_sha256,
                stage_global_seal_sha256=self.stage_global_seal_sha256,
                outer_metadata_leaf_seal_sha256=self.outer_metadata_leaf_seal_sha256,
                component_leaf_seal_sha256=self.outer_component_leaf_seal_sha256,
                state_leaf_seal_sha256=self.state_leaf_seal_sha256,
                outer_view_leaf_seal_sha256=self.outer_view_leaf_seal_sha256,
            ),
        )
        if expected_evidence_seal != self.outer_evidence_leaf_seal_sha256:
            raise ValueError("projection attestation does not reconstruct its evidence leaf")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": OUTER_PROJECTION_ATTESTATION_ARTIFACT,
            "run": self.run.document(),
            "publication_identity": _identity_document(self.publication_identity),
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "stage_global_seal_sha256": self.stage_global_seal_sha256,
            "outer_metadata_leaf_seal_sha256": self.outer_metadata_leaf_seal_sha256,
            "state_leaf_seal_sha256": self.state_leaf_seal_sha256,
            "updated_model_payload_sha256": self.updated_model_payload_sha256,
            "outer_component_leaf_seal_sha256": self.outer_component_leaf_seal_sha256,
            "outer_components_payload_sha256": self.outer_components_payload_sha256,
            "outer_view_leaf_seal_sha256": self.outer_view_leaf_seal_sha256,
            "outer_view_payload_sha256": dict(self.outer_view_payload_sha256),
            "outer_evidence_leaf_seal_sha256": self.outer_evidence_leaf_seal_sha256,
            "outer_evidence_payload_sha256": dict(self.outer_evidence_payload_sha256),
            "outer_context_count": self.outer_context_count,
            "outer_example_ids_sha256": self.outer_example_ids_sha256,
            "outer_candidate_count": self.outer_candidate_count,
            "outer_candidate_ids_sha256": self.outer_candidate_ids_sha256,
            "outer_component_count": self.outer_component_count,
            "outer_component_membership_count": self.outer_component_membership_count,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())

    def view_index_document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "outer_view",
            "run": self.run.document(),
            "relative_path": outer_view_relative_path(self.run),
            "leaf_artifact": OUTER_VIEW_ARTIFACT,
            "leaf_seal_sha256": self.outer_view_leaf_seal_sha256,
            "payload_sha256": dict(self.outer_view_payload_sha256),
            "state_leaf_seal_sha256": self.state_leaf_seal_sha256,
            "outer_component_leaf_seal_sha256": self.outer_component_leaf_seal_sha256,
            "candidate_count": self.outer_candidate_count,
            "candidate_ids_sha256": self.outer_candidate_ids_sha256,
        }

    def evidence_index_document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "index_role": "outer_evidence",
            "run": self.run.document(),
            "relative_path": outer_evidence_relative_path(self.run),
            "leaf_artifact": OUTER_EVIDENCE_ARTIFACT,
            "leaf_seal_sha256": self.outer_evidence_leaf_seal_sha256,
            "payload_sha256": dict(self.outer_evidence_payload_sha256),
            "state_leaf_seal_sha256": self.state_leaf_seal_sha256,
            "outer_component_leaf_seal_sha256": self.outer_component_leaf_seal_sha256,
            "outer_view_leaf_seal_sha256": self.outer_view_leaf_seal_sha256,
            "outer_context_count": self.outer_context_count,
            "outer_example_ids_sha256": self.outer_example_ids_sha256,
            "outer_sequence_prediction_count": self.outer_candidate_count,
            "outer_candidate_ids_sha256": self.outer_candidate_ids_sha256,
            "outer_component_count": self.outer_component_count,
        }


def outer_projection_attestation_from_document(value: object) -> OuterProjectionAttestation:
    raw = _exact_object(
        value,
        {
            "schema_version",
            "artifact",
            "run",
            "publication_identity",
            "protocol_seal_sha256",
            "stage_global_seal_sha256",
            "outer_metadata_leaf_seal_sha256",
            "state_leaf_seal_sha256",
            "updated_model_payload_sha256",
            "outer_component_leaf_seal_sha256",
            "outer_components_payload_sha256",
            "outer_view_leaf_seal_sha256",
            "outer_view_payload_sha256",
            "outer_evidence_leaf_seal_sha256",
            "outer_evidence_payload_sha256",
            "outer_context_count",
            "outer_example_ids_sha256",
            "outer_candidate_count",
            "outer_candidate_ids_sha256",
            "outer_component_count",
            "outer_component_membership_count",
        },
        label="outer projection attestation",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or type(raw["artifact"]) is not str
        or raw["artifact"] != OUTER_PROJECTION_ATTESTATION_ARTIFACT
    ):
        raise ValueError("outer projection attestation identity changed")
    view_payload_raw = _exact_object(
        raw["outer_view_payload_sha256"],
        set(OUTER_VIEW_PAYLOAD_PATHS),
        label="outer projection view payload digests",
    )
    evidence_payload_raw = _exact_object(
        raw["outer_evidence_payload_sha256"],
        set(OUTER_EVIDENCE_PAYLOAD_PATHS),
        label="outer projection evidence payload digests",
    )
    if any(
        type(path) is not str or type(digest) is not str
        for values in (view_payload_raw, evidence_payload_raw)
        for path, digest in values.items()
    ):
        raise ValueError("outer projection payload maps must contain exact text")
    attestation = OuterProjectionAttestation(
        run=_run_from_document(raw["run"]),
        publication_identity=_identity_from_document(raw["publication_identity"]),
        protocol_seal_sha256=_sha256(raw["protocol_seal_sha256"], label="protocol seal"),
        stage_global_seal_sha256=_sha256(
            raw["stage_global_seal_sha256"],
            label="stage-global seal",
        ),
        outer_metadata_leaf_seal_sha256=_sha256(
            raw["outer_metadata_leaf_seal_sha256"],
            label="outer-metadata leaf seal",
        ),
        state_leaf_seal_sha256=_sha256(raw["state_leaf_seal_sha256"], label="state leaf seal"),
        updated_model_payload_sha256=_sha256(
            raw["updated_model_payload_sha256"],
            label="updated model payload",
        ),
        outer_component_leaf_seal_sha256=_sha256(
            raw["outer_component_leaf_seal_sha256"],
            label="outer-component leaf seal",
        ),
        outer_components_payload_sha256=_sha256(
            raw["outer_components_payload_sha256"],
            label="outer-components payload",
        ),
        outer_view_leaf_seal_sha256=_sha256(
            raw["outer_view_leaf_seal_sha256"],
            label="outer-view leaf seal",
        ),
        outer_view_payload_sha256=tuple(
            (path, view_payload_raw[path]) for path in OUTER_VIEW_PAYLOAD_PATHS
        ),
        outer_evidence_leaf_seal_sha256=_sha256(
            raw["outer_evidence_leaf_seal_sha256"],
            label="outer-evidence leaf seal",
        ),
        outer_evidence_payload_sha256=tuple(
            (path, evidence_payload_raw[path]) for path in OUTER_EVIDENCE_PAYLOAD_PATHS
        ),
        outer_context_count=_exact_int(
            raw["outer_context_count"],
            label="outer context count",
            minimum=1,
        ),
        outer_example_ids_sha256=_sha256(raw["outer_example_ids_sha256"], label="outer IDs"),
        outer_candidate_count=_exact_int(
            raw["outer_candidate_count"],
            label="outer candidate count",
            minimum=1,
        ),
        outer_candidate_ids_sha256=_sha256(
            raw["outer_candidate_ids_sha256"],
            label="outer candidate IDs",
        ),
        outer_component_count=_exact_int(
            raw["outer_component_count"],
            label="outer component count",
            minimum=1,
        ),
        outer_component_membership_count=_exact_int(
            raw["outer_component_membership_count"],
            label="outer component membership count",
            minimum=1,
        ),
    )
    if canonical_json_bytes(attestation.document()) != canonical_json_bytes(raw):
        raise ValueError("outer projection attestation does not round-trip exactly")
    return attestation


def outer_projection_attestation_from_bytes(payload: bytes) -> OuterProjectionAttestation:
    return outer_projection_attestation_from_document(
        _strict_json_object(payload, label="outer projection attestation")
    )


@dataclass(frozen=True, slots=True)
class OuterProjectionCapabilities:
    """The two rootless leaves held only by one projection worker/verifier."""

    run: PolicyRunSpec
    outer_view: PhaseSeal
    outer_evidence: PhaseSeal

    def __post_init__(self) -> None:
        _require_frozen_run(self.run, label="outer projection capabilities run")
        if type(self.outer_view) is not PhaseSeal or type(self.outer_evidence) is not PhaseSeal:
            raise TypeError("outer projection capabilities require exact PhaseSeal values")
        if self.outer_view.seal_sha256 == self.outer_evidence.seal_sha256:
            raise ValueError("outer projection view and evidence leaves must be distinct")


def _state_predecessors_from_attestation(attestation: object) -> dict[str, str]:
    from amp_challenge.evaluation.sequential_v2_update_state import UpdateStateAttestation

    if type(attestation) is not UpdateStateAttestation:
        raise TypeError("projection state envelope requires an exact UpdateStateAttestation")
    run = attestation.run
    return {
        "protocol/SHA256SUMS": attestation.protocol_seal_sha256,
        "prepare/global/SHA256SUMS": attestation.prepare_global_seal_sha256,
        f"prepare/rotations/{run.rotation.rotation_id}/base-update/SHA256SUMS": (
            attestation.base_update_leaf_seal_sha256
        ),
        "reveal/global/SHA256SUMS": attestation.reveal_global_seal_sha256,
        f"reveal/tracks/{run.track_id}/SHA256SUMS": attestation.reveal_leaf_seal_sha256,
    }


def _preauthenticate_projection_envelopes(
    *,
    run: PolicyRunSpec,
    state_seal: PhaseSeal,
    state_attestation: object,
    component_seal: PhaseSeal,
    component_attestation: OuterComponentAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest_capability: StageManifestCapability,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_stage_global_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
) -> tuple[ProtocolCapability, StageManifestCapability]:
    from amp_challenge.evaluation.sequential_v2_update_state import (
        UpdateStateAttestation,
        update_state_attestation_from_document,
    )

    frozen = _require_frozen_run(run, label="projection envelope run")
    if type(state_attestation) is not UpdateStateAttestation:
        raise TypeError("projection requires an exact UpdateStateAttestation")
    strict_state = update_state_attestation_from_document(state_attestation.document())
    if strict_state.canonical_bytes() != state_attestation.canonical_bytes():
        raise ValueError("projection state attestation changed during strict reconstruction")
    expected_state = _sha256(
        expected_state_leaf_seal_sha256,
        label="expected projection state leaf",
    )
    if (
        state_attestation.run != frozen
        or state_attestation.publication_identity != publication_identity
        or state_attestation.state_leaf_seal_sha256 != expected_state
    ):
        raise ValueError("projection state attestation differs from controller authority")
    verify_phase_capability(
        state_seal,
        expected_artifact=UPDATE_STATE_ARTIFACT,
        expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
        expected_predecessor_seals=_state_predecessors_from_attestation(state_attestation),
        expected_seal_sha256=expected_state,
    )
    publication_identity.verify_metadata(
        state_seal.metadata_json,
        phase="update",
        scope_id=frozen.track_id,
    )
    if state_seal.payload_sha256 != state_attestation.payload_sha256:
        raise ValueError("projection state payload digests differ from attestation")

    if (
        type(component_attestation) is not OuterComponentAttestation
        or component_attestation.spec != frozen.rotation
        or component_attestation.publication_identity != publication_identity
    ):
        raise ValueError("projection component attestation belongs to another authority")
    strict_component = outer_component_attestation_from_document(component_attestation.document())
    if strict_component.canonical_bytes() != component_attestation.canonical_bytes():
        raise ValueError("projection component attestation changed during strict reconstruction")
    expected_component = _sha256(
        expected_component_leaf_seal_sha256,
        label="expected projection component leaf",
    )
    if component_attestation.component_leaf_seal_sha256 != expected_component:
        raise ValueError("projection component attestation differs from controller authority")
    rotation_runs = policy_runs_for_rotation(frozen.rotation)
    try:
        state_index = rotation_runs.index(frozen)
    except ValueError as error:  # pragma: no cover - guarded by the frozen registry
        raise AssertionError("frozen run is absent from its rotation") from error
    if component_attestation.state_leaf_seal_sha256s[state_index] != expected_state:
        raise ValueError("projection state is absent from the component temporal barrier")
    verify_phase_capability(
        component_seal,
        expected_artifact=OUTER_COMPONENTS_ARTIFACT,
        expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
        expected_predecessor_seals=_component_predecessors(
            spec=frozen.rotation,
            protocol_seal_sha256=component_attestation.protocol_seal_sha256,
            stage_global_seal_sha256=component_attestation.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=(component_attestation.outer_metadata_leaf_seal_sha256),
            state_leaf_seal_sha256s=component_attestation.state_leaf_seal_sha256s,
        ),
        expected_seal_sha256=expected_component,
    )
    publication_identity.verify_metadata(
        component_seal.metadata_json,
        phase="update",
        scope_id=frozen.rotation.rotation_id,
    )
    if component_seal.payload_sha256 != component_attestation.payload_sha256:
        raise ValueError("projection component payload digests differ from attestation")

    if type(protocol_capability) is not ProtocolCapability:
        raise TypeError("projection requires an exact ProtocolCapability")
    protocol = verify_protocol_capability(
        protocol_capability.seal,
        publication_identity=publication_identity,
    )
    stage_digest = _sha256(
        expected_stage_global_seal_sha256,
        label="expected projection stage-global seal",
    )
    if type(stage_manifest_capability) is not StageManifestCapability:
        raise TypeError("projection requires an exact StageManifestCapability")
    stage = verify_stage_manifest_capability(
        stage_manifest_capability.seal,
        expected_global_seal_sha256=stage_digest,
    )
    metadata_entry = stage.leaf(spec=frozen.rotation, role=OUTER_METADATA_ROLE)
    if (
        state_attestation.protocol_seal_sha256 != protocol.seal.seal_sha256
        or component_attestation.protocol_seal_sha256 != protocol.seal.seal_sha256
        or component_attestation.stage_global_seal_sha256 != stage_digest
        or component_attestation.outer_metadata_leaf_seal_sha256 != metadata_entry.leaf_seal_sha256
    ):
        raise ValueError("projection envelopes disagree with protocol/stage authority")
    if type(outer_metadata_capsule) is not AuthenticatedLeafCapsule:
        raise TypeError("projection requires one exact outer-metadata capsule")
    if (
        outer_metadata_capsule.entry != metadata_entry
        or outer_metadata_capsule.source_anchors_sha256 != stage.source_anchors_sha256
        or outer_metadata_capsule.source_predecessors != tuple(sorted(stage.source_predecessors))
    ):
        raise ValueError("projection outer-metadata capsule differs from stage-global index")
    verify_phase_capability(
        outer_metadata_capsule.seal,
        expected_artifact=metadata_entry.leaf_artifact,
        expected_payload_paths=metadata_entry.payload_paths,
        expected_predecessor_seals=dict(stage.source_predecessors),
        expected_seal_sha256=metadata_entry.leaf_seal_sha256,
    )
    return protocol, stage


def _projection_worker_inputs(
    *,
    run: PolicyRunSpec,
    state_seal: PhaseSeal,
    state_attestation: object,
    component_seal: PhaseSeal,
    component_attestation: OuterComponentAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
) -> tuple[
    ProtocolCapability,
    StageManifestCapability,
    UpdateStateCapability,
    OuterComponentCapability,
    OuterMetadataCapability,
]:
    """Authenticate all three projection inputs before opening any leaf payload."""

    from amp_challenge.evaluation.sequential_v2_update_state import (
        UpdateStateAttestation,
        update_state_from_authorities,
    )

    frozen = _require_frozen_run(run, label="projection worker input run")
    protocol, stage = _preauthenticate_projection_envelopes(
        run=frozen,
        state_seal=state_seal,
        state_attestation=state_attestation,
        component_seal=component_seal,
        component_attestation=component_attestation,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
        expected_component_leaf_seal_sha256=expected_component_leaf_seal_sha256,
    )
    if type(state_attestation) is not UpdateStateAttestation:
        raise TypeError("projection worker requires an exact UpdateStateAttestation")

    # The preauthentication call above has already verified the state,
    # component, and metadata envelopes.  These bridges may now decode only
    # the matching model, component table, and label-free metadata payloads.
    state = update_state_from_authorities(
        state_seal,
        attestation=state_attestation,
        run=frozen,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(expected_reveal_campaign_seal_sha256),
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
    )
    component, metadata = outer_component_from_authorities(
        component_seal,
        attestation=component_attestation,
        spec=frozen.rotation,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_capability=stage_manifest_capability,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_component_leaf_seal_sha256=expected_component_leaf_seal_sha256,
    )
    model_digest = _payload_digest(
        state_seal,
        "updated-model.json",
        label="projection state leaf",
    )
    if (
        state.run != frozen
        or state.state_leaf_seal_sha256 != state_attestation.state_leaf_seal_sha256
        or state.updated_model_payload_sha256 != model_digest
        or state.updated_model_payload_sha256
        != dict(state_attestation.payload_sha256)["updated-model.json"]
        or component.spec != frozen.rotation
        or component.component_leaf_seal_sha256 != component_attestation.component_leaf_seal_sha256
        or component.outer_components_payload_sha256
        != dict(component_attestation.payload_sha256)["outer-components.jsonl"]
        or component.outer_metadata_leaf_seal_sha256
        != component_attestation.outer_metadata_leaf_seal_sha256
        or metadata.spec != frozen.rotation
    ):
        raise ValueError("projection materialized inputs differ from their attestations")
    return protocol, stage, state, component, metadata


def _projection_payloads(
    projection: OuterProjection,
    *,
    state: UpdateStateCapability,
    component: OuterComponentCapability,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    if type(projection) is not OuterProjection:
        raise TypeError("projection payload builder requires an exact OuterProjection")
    if type(state) is not UpdateStateCapability or state.run != projection.run:
        raise ValueError("projection payload builder received the wrong update state")
    if (
        type(component) is not OuterComponentCapability
        or component.spec != projection.run.rotation
        or component.component_set != projection.component_set
    ):
        raise ValueError("projection payload builder received the wrong component set")

    candidate_payload = canonical_jsonl_bytes(projection.candidate_documents())
    context_payload = canonical_jsonl_bytes(projection.context_documents())
    sequence_payload = canonical_jsonl_bytes(projection.sequence_documents())
    candidate_ids = tuple(item.sequence_id for item in projection.candidates)
    context_ids = tuple(item.context.example_id for item in projection.context_predictions)
    memberships = sum(len(item.sequence_ids) for item in projection.component_set.components)
    view_payloads = {
        "candidates.jsonl": candidate_payload,
        "view-summary.json": canonical_json_bytes(
            _view_summary_document(
                run=projection.run,
                state_leaf_seal_sha256=state.state_leaf_seal_sha256,
                outer_component_leaf_seal_sha256=component.component_leaf_seal_sha256,
                outer_metadata_leaf_seal_sha256=(component.outer_metadata_leaf_seal_sha256),
                candidate_count=len(candidate_ids),
                candidate_ids_sha256=id_stream_sha256(candidate_ids),
                candidate_payload_sha256=sha256_bytes(candidate_payload),
            )
        ),
    }
    # The evidence summary is completed only after the view has been sealed,
    # because the view digest is an evidence predecessor as well as a payload
    # identity.  The three payloads returned here therefore contain a sentinel
    # summary which the publisher replaces after publishing the view.
    evidence_payloads = {
        "outer-context-predictions.jsonl": context_payload,
        "outer-evidence-summary.json": b"",
        "outer-sequence-predictions.jsonl": sequence_payload,
    }
    _sha256(protocol_seal_sha256, label="projection payload protocol seal")
    _sha256(stage_global_seal_sha256, label="projection payload stage seal")
    if (
        len(context_ids) != EXPECTED_CONTEXTS_BY_FOLD[projection.run.rotation.outer_fold]
        or len(candidate_ids) != EXPECTED_SUPPORT_BY_FOLD[projection.run.rotation.outer_fold]
        or memberships != len(candidate_ids)
    ):
        raise ValueError("projection payload census changed")
    return view_payloads, evidence_payloads


def _complete_evidence_payloads(
    evidence_payloads: Mapping[str, bytes],
    *,
    projection: OuterProjection,
    state: UpdateStateCapability,
    component: OuterComponentCapability,
    stage_global_seal_sha256: str,
    outer_view_seal: PhaseSeal,
) -> dict[str, bytes]:
    if type(evidence_payloads) is not dict or set(evidence_payloads) != set(
        OUTER_EVIDENCE_PAYLOAD_PATHS
    ):
        raise ValueError("projection evidence payload inventory changed")
    if type(outer_view_seal) is not PhaseSeal:
        raise TypeError("projection evidence requires an exact outer-view PhaseSeal")
    context_payload = evidence_payloads["outer-context-predictions.jsonl"]
    sequence_payload = evidence_payloads["outer-sequence-predictions.jsonl"]
    context_ids = tuple(item.context.example_id for item in projection.context_predictions)
    candidate_ids = tuple(item.sequence_id for item in projection.sequence_predictions)
    memberships = sum(len(item.sequence_ids) for item in projection.component_set.components)
    summary = _evidence_summary_document(
        run=projection.run,
        state_leaf_seal_sha256=state.state_leaf_seal_sha256,
        updated_model_payload_sha256=state.updated_model_payload_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        outer_metadata_leaf_seal_sha256=component.outer_metadata_leaf_seal_sha256,
        outer_component_leaf_seal_sha256=component.component_leaf_seal_sha256,
        outer_components_payload_sha256=component.outer_components_payload_sha256,
        outer_view_leaf_seal_sha256=outer_view_seal.seal_sha256,
        outer_view_candidates_payload_sha256=_payload_digest(
            outer_view_seal,
            "candidates.jsonl",
            label="projection outer-view leaf",
        ),
        outer_context_count=len(context_ids),
        outer_example_ids_sha256=id_stream_sha256(context_ids),
        outer_context_predictions_payload_sha256=sha256_bytes(context_payload),
        outer_sequence_prediction_count=len(candidate_ids),
        outer_candidate_ids_sha256=id_stream_sha256(candidate_ids),
        outer_sequence_predictions_payload_sha256=sha256_bytes(sequence_payload),
        outer_component_count=len(projection.component_set.components),
        outer_component_membership_count=memberships,
    )
    return {
        "outer-context-predictions.jsonl": context_payload,
        "outer-evidence-summary.json": canonical_json_bytes(summary),
        "outer-sequence-predictions.jsonl": sequence_payload,
    }


def _decode_outer_candidates(
    payload: bytes,
    *,
    run: PolicyRunSpec,
) -> tuple[OuterMeanCandidate, ...]:
    rows = _strict_jsonl(payload, label="outer-view candidates")
    candidates: list[OuterMeanCandidate] = []
    for index, raw in enumerate(rows):
        candidate = OuterMeanCandidate.from_mapping(raw)
        if (
            candidate.rotation_id != run.rotation.rotation_id
            or candidate.eligible is not True
            or canonical_json_bytes(outer_mean_candidate_document(candidate))
            != canonical_json_bytes(raw)
        ):
            raise ValueError(f"outer-view candidate {index} differs from its exact schema")
        candidates.append(candidate)
    result = tuple(candidates)
    identifiers = tuple(item.sequence_id for item in result)
    if identifiers != tuple(sorted(set(identifiers))):
        raise ValueError("outer-view candidates are not in ascending unique sequence-ID order")
    return result


def _decode_outer_context_predictions(
    payload: bytes,
    *,
    run: PolicyRunSpec,
    outer_metadata: OuterMetadataCapability,
) -> tuple[OuterContextPrediction, ...]:
    rows = _strict_jsonl(payload, label="outer-context predictions")
    if len(rows) != len(outer_metadata.contexts):
        raise ValueError("outer-context prediction census differs from metadata")
    predictions: list[OuterContextPrediction] = []
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
    for index, (raw, context) in enumerate(zip(rows, outer_metadata.contexts, strict=True)):
        row = _exact_object(raw, fields, label=f"outer-context prediction {index}")
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != SCHEMA_VERSION
            or type(row["track_id"]) is not str
            or row["track_id"] != run.track_id
            or type(row["rotation_id"]) is not str
            or row["rotation_id"] != run.rotation.rotation_id
            or type(row["example_id"]) is not str
            or row["example_id"] != context.example_id
            or type(row["sequence_id"]) is not str
            or row["sequence_id"] != context.sequence_id
            or type(row["target"]) is not str
            or row["target"] != context.target
            or type(row["gram"]) is not str
            or row["gram"] != context.gram
            or type(row["fold"]) is not int
            or row["fold"] != context.fold
        ):
            raise ValueError("outer-context prediction differs from authenticated metadata")
        prediction = OuterContextPrediction(
            run=run,
            context=context,
            probability=_canonical_hex(
                row["probability_hex"],
                label=f"outer-context prediction {index} probability",
            ),
        )
        if canonical_json_bytes(prediction.document()) != canonical_json_bytes(row):
            raise ValueError("outer-context prediction does not round-trip exactly")
        predictions.append(prediction)
    return tuple(predictions)


def _decode_outer_sequence_predictions(
    payload: bytes,
    *,
    run: PolicyRunSpec,
    support_sequence_ids: tuple[str, ...],
) -> tuple[OuterSequencePrediction, ...]:
    rows = _strict_jsonl(payload, label="outer-sequence predictions")
    if len(rows) != len(support_sequence_ids):
        raise ValueError("outer-sequence prediction census differs from support")
    predictions: list[OuterSequencePrediction] = []
    fields = {
        "schema_version",
        "track_id",
        "rotation_id",
        "sequence_id",
        "target_probabilities_hex",
        "objective_probabilities_hex",
    }
    for index, (raw, sequence_id) in enumerate(zip(rows, support_sequence_ids, strict=True)):
        row = _exact_object(raw, fields, label=f"outer-sequence prediction {index}")
        if (
            type(row["schema_version"]) is not int
            or row["schema_version"] != SCHEMA_VERSION
            or type(row["track_id"]) is not str
            or row["track_id"] != run.track_id
            or type(row["rotation_id"]) is not str
            or row["rotation_id"] != run.rotation.rotation_id
            or type(row["sequence_id"]) is not str
            or row["sequence_id"] != sequence_id
        ):
            raise ValueError("outer-sequence prediction has the wrong frozen identity")
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
        prediction = OuterSequencePrediction(
            run=run,
            sequence_id=sequence_id,
            target_probabilities=tuple(
                _canonical_hex(
                    target_raw[target],
                    label=f"outer-sequence prediction {index} target {target}",
                )
                for target in TARGETS
            ),
            objective_probabilities=tuple(
                _canonical_hex(
                    objective_raw[objective],
                    label=f"outer-sequence prediction {index} objective {objective}",
                )
                for objective in OBJECTIVES
            ),
        )
        if canonical_json_bytes(prediction.document()) != canonical_json_bytes(row):
            raise ValueError("outer-sequence prediction does not round-trip exactly")
        predictions.append(prediction)
    return tuple(predictions)


def _projection_attestation_from_verified_leaves(
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    state: UpdateStateCapability,
    component: OuterComponentCapability,
    projection: OuterProjection,
    outer_view_seal: PhaseSeal,
    outer_evidence_seal: PhaseSeal,
) -> OuterProjectionAttestation:
    if type(outer_view_seal) is not PhaseSeal or type(outer_evidence_seal) is not PhaseSeal:
        raise TypeError("projection attestation requires two exact PhaseSeal values")
    context_ids = tuple(item.context.example_id for item in projection.context_predictions)
    candidate_ids = tuple(item.sequence_id for item in projection.sequence_predictions)
    memberships = sum(len(item.sequence_ids) for item in projection.component_set.components)
    return OuterProjectionAttestation(
        run=run,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol_seal_sha256,
        stage_global_seal_sha256=stage_global_seal_sha256,
        outer_metadata_leaf_seal_sha256=component.outer_metadata_leaf_seal_sha256,
        state_leaf_seal_sha256=state.state_leaf_seal_sha256,
        updated_model_payload_sha256=state.updated_model_payload_sha256,
        outer_component_leaf_seal_sha256=component.component_leaf_seal_sha256,
        outer_components_payload_sha256=component.outer_components_payload_sha256,
        outer_view_leaf_seal_sha256=outer_view_seal.seal_sha256,
        outer_view_payload_sha256=outer_view_seal.payload_sha256,
        outer_evidence_leaf_seal_sha256=outer_evidence_seal.seal_sha256,
        outer_evidence_payload_sha256=outer_evidence_seal.payload_sha256,
        outer_context_count=len(context_ids),
        outer_example_ids_sha256=id_stream_sha256(context_ids),
        outer_candidate_count=len(candidate_ids),
        outer_candidate_ids_sha256=id_stream_sha256(candidate_ids),
        outer_component_count=len(projection.component_set.components),
        outer_component_membership_count=memberships,
    )


def _decode_outer_projection_leaves(
    capabilities: OuterProjectionCapabilities,
    *,
    attestation: OuterProjectionAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    state: UpdateStateCapability,
    component: OuterComponentCapability,
    outer_metadata: OuterMetadataCapability,
    expected_outer_view_leaf_seal_sha256: str,
    expected_outer_evidence_leaf_seal_sha256: str,
) -> OuterProjection:
    if type(capabilities) is not OuterProjectionCapabilities:
        raise TypeError("projection decoder requires exact paired capabilities")
    if type(attestation) is not OuterProjectionAttestation:
        raise TypeError("projection decoder requires an exact attestation")
    strict = outer_projection_attestation_from_document(attestation.document())
    if strict.canonical_bytes() != attestation.canonical_bytes():
        raise ValueError("projection attestation changed during strict reconstruction")
    if (
        capabilities.run != attestation.run
        or attestation.publication_identity != publication_identity
        or attestation.protocol_seal_sha256
        != _sha256(
            protocol_seal_sha256,
            label="authenticated projection protocol seal",
        )
        or attestation.stage_global_seal_sha256
        != _sha256(
            stage_global_seal_sha256,
            label="authenticated projection stage-global seal",
        )
        or state.run != attestation.run
        or state.state_leaf_seal_sha256 != attestation.state_leaf_seal_sha256
        or state.updated_model_payload_sha256 != attestation.updated_model_payload_sha256
        or component.spec != attestation.run.rotation
        or component.component_leaf_seal_sha256 != attestation.outer_component_leaf_seal_sha256
        or component.outer_components_payload_sha256 != attestation.outer_components_payload_sha256
        or component.outer_metadata_leaf_seal_sha256 != attestation.outer_metadata_leaf_seal_sha256
        or outer_metadata.spec != attestation.run.rotation
    ):
        raise ValueError("projection attestation differs from materialized input authority")

    expected_view = _sha256(
        expected_outer_view_leaf_seal_sha256,
        label="expected outer-view leaf seal",
    )
    expected_evidence = _sha256(
        expected_outer_evidence_leaf_seal_sha256,
        label="expected outer-evidence leaf seal",
    )
    if (
        attestation.outer_view_leaf_seal_sha256 != expected_view
        or attestation.outer_evidence_leaf_seal_sha256 != expected_evidence
    ):
        raise ValueError("projection attestation differs from controller leaf authorities")

    # Authenticate both output envelopes, including the evidence -> view edge,
    # before reading either candidate or evidence payload.
    view = verify_phase_capability(
        capabilities.outer_view,
        expected_artifact=OUTER_VIEW_ARTIFACT,
        expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
        expected_predecessor_seals=_view_predecessors(
            run=attestation.run,
            protocol_seal_sha256=attestation.protocol_seal_sha256,
            stage_global_seal_sha256=attestation.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=(attestation.outer_metadata_leaf_seal_sha256),
            component_leaf_seal_sha256=attestation.outer_component_leaf_seal_sha256,
            state_leaf_seal_sha256=attestation.state_leaf_seal_sha256,
        ),
        expected_seal_sha256=expected_view,
    )
    evidence = verify_phase_capability(
        capabilities.outer_evidence,
        expected_artifact=OUTER_EVIDENCE_ARTIFACT,
        expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
        expected_predecessor_seals=_evidence_predecessors(
            run=attestation.run,
            protocol_seal_sha256=attestation.protocol_seal_sha256,
            stage_global_seal_sha256=attestation.stage_global_seal_sha256,
            outer_metadata_leaf_seal_sha256=(attestation.outer_metadata_leaf_seal_sha256),
            component_leaf_seal_sha256=attestation.outer_component_leaf_seal_sha256,
            state_leaf_seal_sha256=attestation.state_leaf_seal_sha256,
            outer_view_leaf_seal_sha256=expected_view,
        ),
        expected_seal_sha256=expected_evidence,
    )
    publication_identity.verify_metadata(
        view.metadata_json,
        phase="update",
        scope_id=attestation.run.track_id,
    )
    publication_identity.verify_metadata(
        evidence.metadata_json,
        phase="update",
        scope_id=attestation.run.track_id,
    )
    if (
        view.payload_sha256 != attestation.outer_view_payload_sha256
        or evidence.payload_sha256 != attestation.outer_evidence_payload_sha256
    ):
        raise ValueError("projection leaf payload digests differ from its attestation")

    candidate_payload = view.read_payload_bytes("candidates.jsonl")
    context_payload = evidence.read_payload_bytes("outer-context-predictions.jsonl")
    sequence_payload = evidence.read_payload_bytes("outer-sequence-predictions.jsonl")
    candidates = _decode_outer_candidates(candidate_payload, run=attestation.run)
    context_predictions = _decode_outer_context_predictions(
        context_payload,
        run=attestation.run,
        outer_metadata=outer_metadata,
    )
    sequence_predictions = _decode_outer_sequence_predictions(
        sequence_payload,
        run=attestation.run,
        support_sequence_ids=component.component_set.support_sequence_ids,
    )
    projection = OuterProjection(
        run=attestation.run,
        component_set=component.component_set,
        context_predictions=context_predictions,
        sequence_predictions=sequence_predictions,
        candidates=candidates,
    )
    expected_projection = build_outer_projection(
        run=attestation.run,
        state=state,
        outer_metadata=outer_metadata,
        component_set=component.component_set,
    )
    if (
        projection != expected_projection
        or candidate_payload != canonical_jsonl_bytes(expected_projection.candidate_documents())
        or context_payload != canonical_jsonl_bytes(expected_projection.context_documents())
        or sequence_payload != canonical_jsonl_bytes(expected_projection.sequence_documents())
    ):
        raise ValueError("projection leaves differ from the exact authenticated computation")

    expected_view_summary = _view_summary_document(
        run=attestation.run,
        state_leaf_seal_sha256=attestation.state_leaf_seal_sha256,
        outer_component_leaf_seal_sha256=attestation.outer_component_leaf_seal_sha256,
        outer_metadata_leaf_seal_sha256=attestation.outer_metadata_leaf_seal_sha256,
        candidate_count=len(candidates),
        candidate_ids_sha256=id_stream_sha256(tuple(item.sequence_id for item in candidates)),
        candidate_payload_sha256=sha256_bytes(candidate_payload),
    )
    view_summary_payload = view.read_payload_bytes("view-summary.json")
    _strict_json_object(view_summary_payload, label="outer-view summary")
    if view_summary_payload != canonical_json_bytes(expected_view_summary):
        raise ValueError("outer-view summary differs from its authenticated contents")

    expected_evidence_summary = _evidence_summary_document(
        run=attestation.run,
        state_leaf_seal_sha256=attestation.state_leaf_seal_sha256,
        updated_model_payload_sha256=attestation.updated_model_payload_sha256,
        stage_global_seal_sha256=attestation.stage_global_seal_sha256,
        outer_metadata_leaf_seal_sha256=attestation.outer_metadata_leaf_seal_sha256,
        outer_component_leaf_seal_sha256=attestation.outer_component_leaf_seal_sha256,
        outer_components_payload_sha256=attestation.outer_components_payload_sha256,
        outer_view_leaf_seal_sha256=attestation.outer_view_leaf_seal_sha256,
        outer_view_candidates_payload_sha256=sha256_bytes(candidate_payload),
        outer_context_count=len(context_predictions),
        outer_example_ids_sha256=id_stream_sha256(
            tuple(item.context.example_id for item in context_predictions)
        ),
        outer_context_predictions_payload_sha256=sha256_bytes(context_payload),
        outer_sequence_prediction_count=len(sequence_predictions),
        outer_candidate_ids_sha256=id_stream_sha256(
            tuple(item.sequence_id for item in sequence_predictions)
        ),
        outer_sequence_predictions_payload_sha256=sha256_bytes(sequence_payload),
        outer_component_count=len(component.component_set.components),
        outer_component_membership_count=sum(
            len(item.sequence_ids) for item in component.component_set.components
        ),
    )
    evidence_summary_payload = evidence.read_payload_bytes("outer-evidence-summary.json")
    _strict_json_object(evidence_summary_payload, label="outer-evidence summary")
    if evidence_summary_payload != canonical_json_bytes(expected_evidence_summary):
        raise ValueError("outer-evidence summary differs from its authenticated contents")
    return projection


def publish_outer_projection(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    state_seal: PhaseSeal,
    state_attestation: object,
    component_seal: PhaseSeal,
    component_attestation: OuterComponentAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
) -> OuterProjectionAttestation:
    """Publish the view first, then its evidence leaf, for one policy run."""

    frozen = _require_frozen_run(run, label="outer projection publisher run")
    protocol, stage, state, component, metadata = _projection_worker_inputs(
        run=frozen,
        state_seal=state_seal,
        state_attestation=state_attestation,
        component_seal=component_seal,
        component_attestation=component_attestation,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(expected_reveal_campaign_seal_sha256),
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
        expected_component_leaf_seal_sha256=expected_component_leaf_seal_sha256,
    )
    projection = build_outer_projection(
        run=frozen,
        state=state,
        outer_metadata=metadata,
        component_set=component.component_set,
    )
    view_payloads, incomplete_evidence_payloads = _projection_payloads(
        projection,
        state=state,
        component=component,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
    )
    root = Path(destination)
    if not root.is_dir():
        raise ValueError("outer projection destination must be an existing directory")
    metadata_json = publication_identity.metadata(
        phase="update",
        scope_id=frozen.track_id,
    )
    view = publish_phase(
        root / "outer-view",
        artifact=OUTER_VIEW_ARTIFACT,
        payloads=view_payloads,
        predecessor_seals=_view_predecessors(
            run=frozen,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            stage_global_seal_sha256=stage.seal.seal_sha256,
            outer_metadata_leaf_seal_sha256=component.outer_metadata_leaf_seal_sha256,
            component_leaf_seal_sha256=component.component_leaf_seal_sha256,
            state_leaf_seal_sha256=state.state_leaf_seal_sha256,
        ),
        metadata=metadata_json,
    )
    evidence_payloads = _complete_evidence_payloads(
        incomplete_evidence_payloads,
        projection=projection,
        state=state,
        component=component,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        outer_view_seal=view,
    )
    evidence = publish_phase(
        root / "outer-evidence",
        artifact=OUTER_EVIDENCE_ARTIFACT,
        payloads=evidence_payloads,
        predecessor_seals=_evidence_predecessors(
            run=frozen,
            protocol_seal_sha256=protocol.seal.seal_sha256,
            stage_global_seal_sha256=stage.seal.seal_sha256,
            outer_metadata_leaf_seal_sha256=component.outer_metadata_leaf_seal_sha256,
            component_leaf_seal_sha256=component.component_leaf_seal_sha256,
            state_leaf_seal_sha256=state.state_leaf_seal_sha256,
            outer_view_leaf_seal_sha256=view.seal_sha256,
        ),
        metadata=metadata_json,
    )
    capabilities = OuterProjectionCapabilities(
        run=frozen,
        outer_view=view,
        outer_evidence=evidence,
    )
    attestation = _projection_attestation_from_verified_leaves(
        run=frozen,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        state=state,
        component=component,
        projection=projection,
        outer_view_seal=view,
        outer_evidence_seal=evidence,
    )
    _decode_outer_projection_leaves(
        capabilities,
        attestation=attestation,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        state=state,
        component=component,
        outer_metadata=metadata,
        expected_outer_view_leaf_seal_sha256=view.seal_sha256,
        expected_outer_evidence_leaf_seal_sha256=evidence.seal_sha256,
    )
    return attestation


def verify_outer_projection_phase_capabilities(
    capabilities: OuterProjectionCapabilities,
    *,
    state_seal: PhaseSeal,
    state_attestation: object,
    component_seal: PhaseSeal,
    component_attestation: OuterComponentAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
    expected_outer_view_leaf_seal_sha256: str,
    expected_outer_evidence_leaf_seal_sha256: str,
) -> OuterProjectionAttestation:
    """Reverify paired projection leaves and release only their attestation."""

    if type(capabilities) is not OuterProjectionCapabilities:
        raise TypeError("projection verifier requires exact paired capabilities")
    frozen = _require_frozen_run(capabilities.run, label="outer projection verifier run")
    protocol, stage, state, component, metadata = _projection_worker_inputs(
        run=frozen,
        state_seal=state_seal,
        state_attestation=state_attestation,
        component_seal=component_seal,
        component_attestation=component_attestation,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(expected_reveal_campaign_seal_sha256),
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
        expected_component_leaf_seal_sha256=expected_component_leaf_seal_sha256,
    )
    projection = build_outer_projection(
        run=frozen,
        state=state,
        outer_metadata=metadata,
        component_set=component.component_set,
    )
    attestation = _projection_attestation_from_verified_leaves(
        run=frozen,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        state=state,
        component=component,
        projection=projection,
        outer_view_seal=capabilities.outer_view,
        outer_evidence_seal=capabilities.outer_evidence,
    )
    _decode_outer_projection_leaves(
        capabilities,
        attestation=attestation,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        state=state,
        component=component,
        outer_metadata=metadata,
        expected_outer_view_leaf_seal_sha256=(expected_outer_view_leaf_seal_sha256),
        expected_outer_evidence_leaf_seal_sha256=(expected_outer_evidence_leaf_seal_sha256),
    )
    return attestation


def outer_projection_from_authorities(
    capabilities: OuterProjectionCapabilities,
    *,
    attestation: OuterProjectionAttestation,
    state_seal: PhaseSeal,
    state_attestation: object,
    component_seal: PhaseSeal,
    component_attestation: OuterComponentAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest_capability: StageManifestCapability,
    selection_barrier: PhaseSeal,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
    expected_outer_view_leaf_seal_sha256: str,
    expected_outer_evidence_leaf_seal_sha256: str,
) -> OuterProjection:
    """Materialize paired leaves only after every external authority matches."""

    if type(capabilities) is not OuterProjectionCapabilities:
        raise TypeError("projection materializer requires exact paired capabilities")
    if type(attestation) is not OuterProjectionAttestation:
        raise TypeError("projection materializer requires an exact attestation")
    strict = outer_projection_attestation_from_document(attestation.document())
    expected_view = _sha256(
        expected_outer_view_leaf_seal_sha256,
        label="expected materialized outer-view leaf",
    )
    expected_evidence = _sha256(
        expected_outer_evidence_leaf_seal_sha256,
        label="expected materialized outer-evidence leaf",
    )
    if (
        strict.canonical_bytes() != attestation.canonical_bytes()
        or attestation.run != capabilities.run
        or attestation.publication_identity != publication_identity
        or attestation.outer_view_leaf_seal_sha256 != expected_view
        or attestation.outer_evidence_leaf_seal_sha256 != expected_evidence
    ):
        raise ValueError("projection attestation differs from controller authority")
    protocol, stage, state, component, metadata = _projection_worker_inputs(
        run=capabilities.run,
        state_seal=state_seal,
        state_attestation=state_attestation,
        component_seal=component_seal,
        component_attestation=component_attestation,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        stage_manifest_capability=stage_manifest_capability,
        selection_barrier=selection_barrier,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(expected_reveal_campaign_seal_sha256),
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
        expected_component_leaf_seal_sha256=expected_component_leaf_seal_sha256,
    )
    return _decode_outer_projection_leaves(
        capabilities,
        attestation=attestation,
        publication_identity=publication_identity,
        protocol_seal_sha256=protocol.seal.seal_sha256,
        stage_global_seal_sha256=stage.seal.seal_sha256,
        state=state,
        component=component,
        outer_metadata=metadata,
        expected_outer_view_leaf_seal_sha256=expected_view,
        expected_outer_evidence_leaf_seal_sha256=expected_evidence,
    )


__all__ = [
    "EXPECTED_COMPONENT_LEAVES",
    "EXPECTED_COMPONENT_MEMBERSHIPS",
    "EXPECTED_OUTER_CONTEXT_PREDICTIONS",
    "EXPECTED_OUTER_SEQUENCE_PREDICTIONS",
    "EXPECTED_PROJECTION_WORKERS",
    "OUTER_COMPONENT_ATTESTATION_ARTIFACT",
    "OUTER_COMPONENT_SUMMARY_ARTIFACT",
    "OUTER_EVIDENCE_SUMMARY_ARTIFACT",
    "OUTER_PROJECTION_ATTESTATION_ARTIFACT",
    "OUTER_VIEW_SUMMARY_ARTIFACT",
    "OuterComponentAttestation",
    "OuterComponentCapability",
    "OuterProjectionAttestation",
    "OuterProjectionCapabilities",
    "outer_component_attestation_from_bytes",
    "outer_component_attestation_from_document",
    "outer_component_from_authorities",
    "outer_component_relative_path",
    "outer_evidence_relative_path",
    "outer_projection_attestation_from_bytes",
    "outer_projection_attestation_from_document",
    "outer_projection_from_authorities",
    "outer_view_relative_path",
    "publish_outer_component",
    "publish_outer_projection",
    "verify_outer_component_phase_capability",
    "verify_outer_projection_phase_capabilities",
]

"""Isolated execution and supervision for sequential-v2 finalization.

The controller proves the complete outer-selection barrier before it captures
an outer-outcome leaf.  One pathless request is then consumed by one fresh
worker, which rederives every outer selection before decoding any outcome and
publishes the exact seven-payload ``finalize/global`` phase.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_commitments import (
    CAMPAIGN_BARRIER_ARTIFACT,
    CAMPAIGN_BARRIER_PAYLOAD_PATHS,
    verify_pool_commitment_campaign_barrier_capability,
)
from amp_challenge.evaluation.sequential_v2_finalize import (
    FINALIZE_ARTIFACT,
    FINALIZE_PAYLOAD_PATHS,
    FINALIZE_SUMMARY_ARTIFACT,
    FinalizationTrackInput,
    PoolSelectionEvidence,
    finalize_campaign,
)
from amp_challenge.evaluation.sequential_v2_finalize_wire import (
    FINALIZE_WORKER_ROLE,
    FinalizeRequestCapacityMeasurement,
    FinalizeWorkerRequest,
    encode_and_measure_finalize_worker_request,
    finalize_worker_request_from_bytes,
)
from amp_challenge.evaluation.sequential_v2_outer_select import (
    OUTER_SELECTION_ARTIFACT,
    OUTER_SELECTION_CAMPAIGN_ARTIFACT,
    OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
    OUTER_SELECTION_PAYLOAD_PATHS,
    OuterSelectionCampaignCapability,
    outer_selection_from_campaign,
    outer_selection_relative_path,
    verify_outer_selection_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    CAMPAIGN_PAYLOAD_PATHS,
    PREDICTION_VIEW_ARTIFACT,
    PREDICTION_VIEW_PAYLOAD_PATHS,
    PREDICTION_VIEW_ROLE,
    PREPARE_CAMPAIGN_ARTIFACT,
    PROTOCOL_ARTIFACT,
    PROTOCOL_PAYLOAD_PATHS,
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    prediction_view_from_campaign,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_primitives import bootstrap_indices_sha256
from amp_challenge.evaluation.sequential_v2_protocol import (
    EXPECTED_POLICY_RUNS,
    EXPECTED_ROTATIONS,
    ordered_policy_runs,
    ordered_rotations,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    POOL_REVEAL_ARTIFACT,
    POOL_REVEAL_PAYLOAD_PATHS,
    REVEAL_CAMPAIGN_ARTIFACT,
    REVEAL_CAMPAIGN_PAYLOAD_PATHS,
    RevealCampaignCapability,
    pool_reveal_relative_path,
    pool_reveal_with_commitment_from_campaign,
    verify_reveal_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    publish_phase,
    relocate_sealed_phase_noreplace,
    sha256_bytes,
    verify_phase,
    verify_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    GLOBAL_PAYLOAD_PATHS,
    OUTER_OUTCOME_ROLE,
    AuthenticatedLeafCapsule,
    authenticate_stage_leaf_for_controller,
    authenticate_stage_manifest_for_controller,
    outer_outcome_vault_from_stage_capabilities,
    verify_stage_manifest,
)
from amp_challenge.evaluation.sequential_v2_supervisor import (
    _DEFAULT_WORKER_TIMEOUT_SECONDS,
    _create_private_directory,
    _FreshWorkerResult,
    _launch_fresh_worker,
    _observe_phase_marker_sha256,
    _release_successful_outbox,
    _require_unrelated_same_filesystem,
    _trusted_directory,
    _write_all,
)
from amp_challenge.evaluation.sequential_v2_update import (
    OUTER_EVIDENCE_ARTIFACT,
    OUTER_EVIDENCE_PAYLOAD_PATHS,
    OUTER_VIEW_ARTIFACT,
    OUTER_VIEW_PAYLOAD_PATHS,
    UPDATE_CAMPAIGN_ARTIFACT,
    UPDATE_CAMPAIGN_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_update_campaign import (
    UpdateCampaignCapability,
    outer_context_predictions_from_campaign,
    verify_update_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_wire import (
    publication_identity_document,
    publication_identity_from_document,
    strict_canonical_json_object,
)

SCHEMA_VERSION = 1
FINALIZE_ATTESTATION_ARTIFACT = "sequential_v2_finalize_attestation_v1"
FINALIZE_SUPERVISION_ARTIFACT = "sequential_v2_finalize_supervision_v1"
FINALIZE_PREDECESSOR_COUNT = 927
FINALIZE_DIRECT_LEAF_COUNT = 920
FINALIZE_WORKER_COUNT = 1
_MAX_ATTESTATION_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PORTABLE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_PRE_FINALIZE_RUN_INVENTORY = (
    "outer-select",
    "prepare",
    "protocol",
    "reveal",
    "select",
    "update",
)
_COMPLETE_RUN_INVENTORY = ("finalize", *_PRE_FINALIZE_RUN_INVENTORY)

_CAMPAIGN_DOCUMENT = {
    "schema_version": SCHEMA_VERSION,
    "scope_id": "global",
    "rotation_count": EXPECTED_ROTATIONS,
    "policy_run_count": EXPECTED_POLICY_RUNS,
}
_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "campaign",
        "publication_identity",
        "protocol_seal_sha256",
        "stage_global_seal_sha256",
        "prepare_global_seal_sha256",
        "selection_global_seal_sha256",
        "reveal_global_seal_sha256",
        "update_global_seal_sha256",
        "outer_selection_global_seal_sha256",
        "finalize_global_seal_sha256",
        "payload_sha256",
        "predecessor_count",
        "rotation_metric_count",
        "policy_rotation_metric_count",
        "outer_fold_unit_count",
        "policy_point_estimate_count",
        "paired_comparison_count",
        "promising_for_prospective_followup",
    }
)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _payload_digests(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in value
    ):
        raise TypeError("finalize payload digests must be an exact immutable map")
    if tuple(path for path, _digest in value) != FINALIZE_PAYLOAD_PATHS:
        raise ValueError("finalize payload digest inventory or order changed")
    for path, digest in value:
        _sha256(digest, label=f"finalize payload {path}")
    return value


@dataclass(frozen=True, slots=True)
class FinalizeAttestation:
    """Bounded, payload-free result emitted by the one finalizer worker."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_seal_sha256: str
    stage_global_seal_sha256: str
    prepare_global_seal_sha256: str
    selection_global_seal_sha256: str
    reveal_global_seal_sha256: str
    update_global_seal_sha256: str
    outer_selection_global_seal_sha256: str
    finalize_global_seal_sha256: str
    payload_sha256: tuple[tuple[str, str], ...]
    predecessor_count: int
    rotation_metric_count: int
    policy_rotation_metric_count: int
    outer_fold_unit_count: int
    policy_point_estimate_count: int
    paired_comparison_count: int
    promising_for_prospective_followup: bool

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("finalize attestation requires an exact publication identity")
        for name in (
            "protocol_seal_sha256",
            "stage_global_seal_sha256",
            "prepare_global_seal_sha256",
            "selection_global_seal_sha256",
            "reveal_global_seal_sha256",
            "update_global_seal_sha256",
            "outer_selection_global_seal_sha256",
            "finalize_global_seal_sha256",
        ):
            _sha256(getattr(self, name), label=f"finalize attestation {name}")
        _payload_digests(self.payload_sha256)
        expected_counts = {
            "predecessor_count": FINALIZE_PREDECESSOR_COUNT,
            "rotation_metric_count": 220,
            "policy_rotation_metric_count": 140,
            "outer_fold_unit_count": 35,
            "policy_point_estimate_count": 7,
            "paired_comparison_count": 7,
        }
        for name, expected in expected_counts.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"finalize attestation {name} must equal {expected}")
        if type(self.promising_for_prospective_followup) is not bool:
            raise TypeError("finalize attestation decision must be an exact boolean")
        _verify_attestation_summary_binding(self)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": FINALIZE_ATTESTATION_ARTIFACT,
            "campaign": dict(_CAMPAIGN_DOCUMENT),
            "publication_identity": publication_identity_document(self.publication_identity),
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "stage_global_seal_sha256": self.stage_global_seal_sha256,
            "prepare_global_seal_sha256": self.prepare_global_seal_sha256,
            "selection_global_seal_sha256": self.selection_global_seal_sha256,
            "reveal_global_seal_sha256": self.reveal_global_seal_sha256,
            "update_global_seal_sha256": self.update_global_seal_sha256,
            "outer_selection_global_seal_sha256": (self.outer_selection_global_seal_sha256),
            "finalize_global_seal_sha256": self.finalize_global_seal_sha256,
            "payload_sha256": dict(self.payload_sha256),
            "predecessor_count": self.predecessor_count,
            "rotation_metric_count": self.rotation_metric_count,
            "policy_rotation_metric_count": self.policy_rotation_metric_count,
            "outer_fold_unit_count": self.outer_fold_unit_count,
            "policy_point_estimate_count": self.policy_point_estimate_count,
            "paired_comparison_count": self.paired_comparison_count,
            "promising_for_prospective_followup": self.promising_for_prospective_followup,
        }

    def canonical_bytes(self) -> bytes:
        payload = canonical_json_bytes(self.document())
        if len(payload) > _MAX_ATTESTATION_BYTES:
            raise ValueError("finalize attestation exceeds its byte bound")
        return payload


def _expected_summary_document(attestation: FinalizeAttestation) -> dict[str, object]:
    digests = dict(attestation.payload_sha256)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": FINALIZE_SUMMARY_ARTIFACT,
        "rotation_metric_count": attestation.rotation_metric_count,
        "policy_rotation_metric_count": attestation.policy_rotation_metric_count,
        "outer_fold_unit_count": attestation.outer_fold_unit_count,
        "policy_point_estimate_count": attestation.policy_point_estimate_count,
        "paired_comparison_count": attestation.paired_comparison_count,
        "guarded_track_count": 60,
        "valid_guarded_track_count": 60,
        "bootstrap_indices_sha256": bootstrap_indices_sha256(),
        "rotation_metrics_sha256": digests["rotation-metrics.jsonl"],
        "policy_rotation_metrics_sha256": digests["policy-rotation-metrics.jsonl"],
        "outer_fold_units_sha256": digests["outer-fold-units.jsonl"],
        "policy_point_estimates_sha256": digests["policy-point-estimates.jsonl"],
        "paired_comparisons_sha256": digests["paired-comparisons.jsonl"],
        "promotion_decision_sha256": digests["promotion-decision.json"],
        "promising_for_prospective_followup": (attestation.promising_for_prospective_followup),
    }


def _verify_attestation_summary_binding(attestation: FinalizeAttestation) -> None:
    expected = canonical_json_bytes(_expected_summary_document(attestation))
    if dict(attestation.payload_sha256)["summary.json"] != sha256_bytes(expected):
        raise ValueError("finalize attestation summary digest differs from its safe fields")


def finalize_attestation_from_bytes(payload: bytes) -> FinalizeAttestation:
    """Strictly decode one canonical, payload-free finalizer result."""

    raw = strict_canonical_json_object(
        payload,
        label="finalize attestation",
        maximum_bytes=_MAX_ATTESTATION_BYTES,
    )
    if type(raw) is not dict or set(raw) != _ATTESTATION_FIELDS:
        raise ValueError("finalize attestation must contain its exact field set")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
        or raw["artifact"] != FINALIZE_ATTESTATION_ARTIFACT
        or raw["campaign"] != _CAMPAIGN_DOCUMENT
    ):
        raise ValueError("finalize attestation identity or campaign changed")
    payload_map = raw["payload_sha256"]
    if type(payload_map) is not dict or tuple(payload_map) != FINALIZE_PAYLOAD_PATHS:
        raise ValueError("finalize attestation payload digest map changed")
    result = FinalizeAttestation(
        publication_identity=publication_identity_from_document(raw["publication_identity"]),
        protocol_seal_sha256=_sha256(
            raw["protocol_seal_sha256"], label="finalize attestation protocol"
        ),
        stage_global_seal_sha256=_sha256(
            raw["stage_global_seal_sha256"], label="finalize attestation stage"
        ),
        prepare_global_seal_sha256=_sha256(
            raw["prepare_global_seal_sha256"], label="finalize attestation prepare"
        ),
        selection_global_seal_sha256=_sha256(
            raw["selection_global_seal_sha256"], label="finalize attestation selection"
        ),
        reveal_global_seal_sha256=_sha256(
            raw["reveal_global_seal_sha256"], label="finalize attestation reveal"
        ),
        update_global_seal_sha256=_sha256(
            raw["update_global_seal_sha256"], label="finalize attestation update"
        ),
        outer_selection_global_seal_sha256=_sha256(
            raw["outer_selection_global_seal_sha256"],
            label="finalize attestation outer selection",
        ),
        finalize_global_seal_sha256=_sha256(
            raw["finalize_global_seal_sha256"], label="finalize attestation final phase"
        ),
        payload_sha256=tuple(
            (path, _sha256(payload_map[path], label=path)) for path in payload_map
        ),
        predecessor_count=raw["predecessor_count"],
        rotation_metric_count=raw["rotation_metric_count"],
        policy_rotation_metric_count=raw["policy_rotation_metric_count"],
        outer_fold_unit_count=raw["outer_fold_unit_count"],
        policy_point_estimate_count=raw["policy_point_estimate_count"],
        paired_comparison_count=raw["paired_comparison_count"],
        promising_for_prospective_followup=raw["promising_for_prospective_followup"],
    )
    if result.canonical_bytes() != payload:
        raise ValueError("finalize attestation changed during typed reconstruction")
    return result


def finalize_predecessors(request: FinalizeWorkerRequest) -> dict[str, str]:
    """Construct the exact lexicographically normalized 927-edge closure."""

    if type(request) is not FinalizeWorkerRequest:
        raise TypeError("finalize predecessors require an exact worker request")
    predecessors = {
        "protocol/SHA256SUMS": request.protocol_capability.seal.seal_sha256,
        "stage/global/SHA256SUMS": request.stage_manifest_capability.seal.seal_sha256,
        "prepare/global/SHA256SUMS": request.prepare_campaign.seal.seal_sha256,
        "select/global/SHA256SUMS": request.selection_campaign_seal.seal_sha256,
        "reveal/global/SHA256SUMS": request.reveal_campaign.seal.seal_sha256,
        "update/global/SHA256SUMS": request.update_campaign.seal.seal_sha256,
        "outer-select/global/SHA256SUMS": request.outer_selection_campaign.seal.seal_sha256,
    }
    for spec, outcome, prediction in zip(
        ordered_rotations(),
        request.outer_outcome_vault_seals,
        request.prediction_view_seals,
        strict=True,
    ):
        predecessors[f"stage/rotations/{spec.rotation_id}/{OUTER_OUTCOME_ROLE}/SHA256SUMS"] = (
            outcome.seal_sha256
        )
        predecessors[f"prepare/rotations/{spec.rotation_id}/prediction-view/SHA256SUMS"] = (
            prediction.seal_sha256
        )
    for run, reveal, evidence, view, selection in zip(
        ordered_policy_runs(),
        request.reveal_leaf_seals,
        request.outer_evidence_seals,
        request.outer_view_seals,
        request.outer_selection_leaf_seals,
        strict=True,
    ):
        predecessors[f"{pool_reveal_relative_path(run)}/SHA256SUMS"] = reveal.seal_sha256
        predecessors[f"update/tracks/{run.track_id}/outer-evidence/SHA256SUMS"] = (
            evidence.seal_sha256
        )
        predecessors[f"update/tracks/{run.track_id}/outer-view/SHA256SUMS"] = view.seal_sha256
        predecessors[f"{outer_selection_relative_path(run)}/SHA256SUMS"] = selection.seal_sha256
    if len(predecessors) != FINALIZE_PREDECESSOR_COUNT:
        raise AssertionError("finalize predecessor census changed")
    return dict(sorted(predecessors.items()))


def run_finalize_worker_request(request_payload: bytes, output: Path) -> bytes:
    """Reauthenticate, reduce, and publish one complete rootless request."""

    request = finalize_worker_request_from_bytes(request_payload)
    runs = ordered_policy_runs()
    rotations = ordered_rotations()

    # This complete rederivation intentionally precedes the first semantic
    # outer-outcome decode below.
    selected_outer = tuple(
        outer_selection_from_campaign(
            request.outer_selection_campaign,
            run=run,
            selection_seal=selection_seal,
            update_campaign=request.update_campaign,
            outer_view_seal=outer_view_seal,
            publication_identity=request.publication_identity,
            protocol_capability=request.protocol_capability,
            expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
            expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
            expected_update_campaign_seal_sha256=(request.expected_update_campaign_seal_sha256),
            expected_outer_selection_campaign_seal_sha256=(
                request.expected_outer_selection_campaign_seal_sha256
            ),
        )
        for run, outer_view_seal, selection_seal in zip(
            runs,
            request.outer_view_seals,
            request.outer_selection_leaf_seals,
            strict=True,
        )
    )

    context_payload_by_outer_fold: dict[int, bytes] = {}
    for spec, seal in zip(rotations, request.outer_outcome_vault_seals, strict=True):
        payload = seal.read_payload_bytes("contexts.jsonl")
        prior = context_payload_by_outer_fold.setdefault(spec.outer_fold, payload)
        if prior != payload:
            raise ValueError("same outer-fold outcome contexts differ across acquisition rotations")

    outer_vaults = []
    for spec, seal in zip(rotations, request.outer_outcome_vault_seals, strict=True):
        entry = request.stage_manifest_capability.leaf(spec=spec, role=OUTER_OUTCOME_ROLE)
        capsule = AuthenticatedLeafCapsule(
            entry=entry,
            seal=seal,
            source_anchors_sha256=request.stage_manifest_capability.source_anchors_sha256,
            source_predecessors=request.stage_manifest_capability.source_predecessors,
        )
        outer_vaults.append(
            outer_outcome_vault_from_stage_capabilities(
                request.stage_manifest_capability,
                capsule,
                spec=spec,
                expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
            )
        )
    vault_by_rotation = dict(zip(rotations, outer_vaults, strict=True))

    prediction_views = tuple(
        prediction_view_from_campaign(
            request.prepare_campaign,
            spec=spec,
            prediction_view_seal=seal,
            publication_identity=request.publication_identity,
            protocol_capability=request.protocol_capability,
            expected_campaign_seal_sha256=request.expected_prepare_campaign_seal_sha256,
        )
        for spec, seal in zip(rotations, request.prediction_view_seals, strict=True)
    )
    view_by_rotation = dict(zip(rotations, prediction_views, strict=True))

    inputs: list[FinalizationTrackInput] = []
    for run, reveal_seal, evidence_seal, selected in zip(
        runs,
        request.reveal_leaf_seals,
        request.outer_evidence_seals,
        selected_outer,
        strict=True,
    ):
        vault = vault_by_rotation[run.rotation]
        reveal = pool_reveal_with_commitment_from_campaign(
            request.reveal_campaign,
            run=run,
            reveal_seal=reveal_seal,
            publication_identity=request.publication_identity,
            protocol_capability=request.protocol_capability,
            stage_manifest_capability=request.stage_manifest_capability,
            selection_barrier=request.selection_campaign_seal,
            expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
            expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(
                request.expected_selection_campaign_seal_sha256
            ),
            expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
        )
        predictions = outer_context_predictions_from_campaign(
            request.update_campaign,
            run=run,
            outer_evidence_seal=evidence_seal,
            outer_outcome_vault=vault,
            publication_identity=request.publication_identity,
            protocol_capability=request.protocol_capability,
            expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
            expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
            expected_update_campaign_seal_sha256=(request.expected_update_campaign_seal_sha256),
        )
        inputs.append(
            FinalizationTrackInput(
                run=run,
                outer_context_predictions=predictions,
                outer_outcomes=vault.contexts,
                pool_selection=PoolSelectionEvidence.from_commitment(reveal.commitment),
                pool_revealed_contexts=reveal.reveal.contexts,
                prediction_view=view_by_rotation[run.rotation],
                outer_selected_sequence_ids=selected.selected_sequence_ids,
                outer_selected_unique_component_count=(selected.selected_unique_component_count),
                outer_selected_max_component_occupancy=(selected.selected_max_component_occupancy),
            )
        )
    payloads = finalize_campaign(tuple(inputs))
    predecessor_seals = finalize_predecessors(request)
    seal = publish_phase(
        output / "global",
        artifact=FINALIZE_ARTIFACT,
        payloads=dict(payloads.payloads),
        predecessor_seals=predecessor_seals,
        metadata=request.publication_identity.metadata(phase="finalize", scope_id="global"),
    )
    verified = verify_phase_capability(
        seal,
        expected_artifact=FINALIZE_ARTIFACT,
        expected_payload_paths=FINALIZE_PAYLOAD_PATHS,
        expected_predecessor_seals=predecessor_seals,
        expected_seal_sha256=seal.seal_sha256,
    )
    request.publication_identity.verify_metadata(
        verified.metadata_json,
        phase="finalize",
        scope_id="global",
    )
    summary = strict_canonical_json_object(
        payloads.read("summary.json"),
        label="finalize summary",
    )
    attestation = FinalizeAttestation(
        publication_identity=request.publication_identity,
        protocol_seal_sha256=request.expected_protocol_seal_sha256,
        stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        prepare_global_seal_sha256=request.expected_prepare_campaign_seal_sha256,
        selection_global_seal_sha256=request.expected_selection_campaign_seal_sha256,
        reveal_global_seal_sha256=request.expected_reveal_campaign_seal_sha256,
        update_global_seal_sha256=request.expected_update_campaign_seal_sha256,
        outer_selection_global_seal_sha256=(request.expected_outer_selection_campaign_seal_sha256),
        finalize_global_seal_sha256=verified.seal_sha256,
        payload_sha256=verified.payload_sha256,
        predecessor_count=len(predecessor_seals),
        rotation_metric_count=summary["rotation_metric_count"],
        policy_rotation_metric_count=summary["policy_rotation_metric_count"],
        outer_fold_unit_count=summary["outer_fold_unit_count"],
        policy_point_estimate_count=summary["policy_point_estimate_count"],
        paired_comparison_count=summary["paired_comparison_count"],
        promising_for_prospective_followup=summary["promising_for_prospective_followup"],
    )
    return attestation.canonical_bytes()


@dataclass(frozen=True, slots=True)
class _MarkerAuthority:
    root: Path
    payload_paths: tuple[str, ...]
    seal_sha256: str


@dataclass(frozen=True, slots=True)
class _FinalizeRequestCapture:
    """One authenticated request plus the controller's pre-worker snapshots."""

    request: FinalizeWorkerRequest
    stage_root: Path
    run_root: Path
    outer_root: Path
    tracks_root: Path
    marker_authorities: tuple[_MarkerAuthority, ...]
    observed_markers: tuple[str, ...]
    observed_outer_selection_markers: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.request) is not FinalizeWorkerRequest:
            raise TypeError("finalize capture requires an exact worker request")
        if (
            type(self.marker_authorities) is not tuple
            or len(self.marker_authorities) != FINALIZE_PREDECESSOR_COUNT
            or any(type(item) is not _MarkerAuthority for item in self.marker_authorities)
            or type(self.observed_markers) is not tuple
            or len(self.observed_markers) != FINALIZE_PREDECESSOR_COUNT
        ):
            raise ValueError("finalize capture requires exactly 927 marker authorities")
        if (
            type(self.observed_outer_selection_markers) is not tuple
            or len(self.observed_outer_selection_markers) != EXPECTED_POLICY_RUNS
        ):
            raise ValueError("finalize capture requires the complete outer-selection snapshot")


@dataclass(frozen=True, slots=True)
class _PublishedFinalizeRequest:
    path: Path
    parent_identity: tuple[int, int]
    file_identity: tuple[int, int]
    payload_sha256: str


def _observe_markers(values: tuple[_MarkerAuthority, ...]) -> tuple[str, ...]:
    return tuple(
        _observe_phase_marker_sha256(
            item.root,
            expected_payload_paths=item.payload_paths,
            expected_seal_sha256=item.seal_sha256,
        )
        for item in values
    )


def _require_exact_directory_names(
    root: Path,
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    if tuple(sorted(entry.name for entry in os.scandir(root))) != tuple(sorted(names)):
        raise RuntimeError(f"{label} differs from the frozen finalization graph")


def _validate_finalize_capture_arguments(
    *,
    stage_root: str | Path,
    expected_source_anchors: Mapping[str, object],
    run_root: str | Path,
    expected_protocol_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> tuple[Path, Path]:
    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("finalization requires an exact publication identity")
    if not isinstance(expected_source_anchors, Mapping):
        raise TypeError("finalization source anchors must be a mapping")
    authorities = (
        expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256,
    )
    for index, digest in enumerate(authorities):
        _sha256(digest, label=f"finalization global authority {index}")

    final_root = _trusted_directory(
        run_root,
        label="sequential-v2 run root",
        require_empty=False,
        exact_mode=0o700,
    )
    stage_path = _trusted_directory(
        stage_root,
        label="trusted stage root",
        require_empty=False,
        exact_mode=0o555,
    )
    if (
        stage_path == final_root
        or stage_path in final_root.parents
        or final_root in stage_path.parents
    ):
        raise ValueError("stage and run roots must be disjoint trees")
    return stage_path, final_root


def _require_worker_identity(result: _FreshWorkerResult) -> None:
    if (
        result.role != FINALIZE_WORKER_ROLE
        or type(result.process_pid) is not int
        or result.process_pid <= 0
    ):
        raise ValueError("finalize worker identity differs from its launched role")


def _require_output_inventory(result: _FreshWorkerResult) -> None:
    if tuple(sorted(entry.name for entry in os.scandir(result.output))) != ("global",):
        raise RuntimeError("finalize worker outbox differs from its exact role")


def _attestation_matches_request(
    attestation: FinalizeAttestation,
    request: FinalizeWorkerRequest,
) -> None:
    expected = (
        request.publication_identity,
        request.expected_protocol_seal_sha256,
        request.expected_stage_global_seal_sha256,
        request.expected_prepare_campaign_seal_sha256,
        request.expected_selection_campaign_seal_sha256,
        request.expected_reveal_campaign_seal_sha256,
        request.expected_update_campaign_seal_sha256,
        request.expected_outer_selection_campaign_seal_sha256,
    )
    actual = (
        attestation.publication_identity,
        attestation.protocol_seal_sha256,
        attestation.stage_global_seal_sha256,
        attestation.prepare_global_seal_sha256,
        attestation.selection_global_seal_sha256,
        attestation.reveal_global_seal_sha256,
        attestation.update_global_seal_sha256,
        attestation.outer_selection_global_seal_sha256,
    )
    if actual != expected:
        raise ValueError("finalize attestation differs from controller authority")


@dataclass(frozen=True, slots=True)
class FinalizeSupervisionResult:
    """Verified release from exactly one complete-campaign finalizer."""

    publication_identity: SequentialV2PublicationIdentity
    attestation: FinalizeAttestation
    input_marker_count: int
    worker_process_count: int

    def __post_init__(self) -> None:
        if (
            type(self.publication_identity) is not SequentialV2PublicationIdentity
            or type(self.attestation) is not FinalizeAttestation
            or self.attestation.publication_identity != self.publication_identity
        ):
            raise ValueError("finalize supervision identity is invalid")
        if type(self.input_marker_count) is not int or self.input_marker_count != 927:
            raise ValueError("finalize supervision requires exactly 927 input markers")
        if type(self.worker_process_count) is not int or self.worker_process_count != 1:
            raise ValueError("finalize supervision requires exactly one worker")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": FINALIZE_SUPERVISION_ARTIFACT,
            "finalize_global_seal_sha256": self.attestation.finalize_global_seal_sha256,
            "input_marker_count": self.input_marker_count,
            "worker_process_count": self.worker_process_count,
            "promising_for_prospective_followup": (
                self.attestation.promising_for_prospective_followup
            ),
        }


def _capture_global_capabilities(
    run_root: Path,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    expected_protocol_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
    outer_global_seal: PhaseSeal,
) -> tuple[
    ProtocolCapability,
    PrepareCampaignCapability,
    PhaseSeal,
    PhaseSeal,
    UpdateCampaignCapability,
    OuterSelectionCampaignCapability,
]:
    protocol_seal = verify_phase(
        run_root / "protocol",
        expected_artifact=PROTOCOL_ARTIFACT,
        expected_payload_paths=PROTOCOL_PAYLOAD_PATHS,
        expected_predecessor_seals={},
        expected_seal_sha256=expected_protocol_seal_sha256,
    )
    protocol = verify_protocol_capability(
        protocol_seal,
        publication_identity=publication_identity,
    )
    prepare_seal = verify_phase(
        run_root / "prepare" / "global",
        expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    prepare = verify_prepare_campaign_capability(
        PrepareCampaignCapability(prepare_seal, publication_identity),
        publication_identity=publication_identity,
        expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
    )
    selection = verify_phase(
        run_root / "select" / "global",
        expected_artifact=CAMPAIGN_BARRIER_ARTIFACT,
        expected_payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
        expected_seal_sha256=expected_selection_campaign_seal_sha256,
    )
    verify_pool_commitment_campaign_barrier_capability(
        selection,
        protocol_capability=protocol,
        prepare_campaign=prepare,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        publication_identity=publication_identity,
        expected_seal_sha256=expected_selection_campaign_seal_sha256,
    )
    update_seal = verify_phase(
        run_root / "update" / "global",
        expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_update_campaign_seal_sha256,
    )
    update = verify_update_campaign_capability(
        UpdateCampaignCapability(update_seal, publication_identity),
        publication_identity=publication_identity,
        protocol_capability=protocol,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    outer = verify_outer_selection_campaign_capability(
        OuterSelectionCampaignCapability(outer_global_seal, publication_identity),
        publication_identity=publication_identity,
        protocol_capability=protocol,
        update_campaign=update,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
    )
    return protocol, prepare, selection, update_seal, update, outer


def _capture_finalize_campaign_request(
    *,
    stage_root: Path,
    expected_source_anchors: Mapping[str, object],
    run_root: Path,
    expected_protocol_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> _FinalizeRequestCapture:
    """Capture the exact authentic 927-input request without launching a worker."""

    _require_exact_directory_names(
        run_root,
        _PRE_FINALIZE_RUN_INVENTORY,
        label="pre-finalize run-root inventory",
    )
    outer_root = _trusted_directory(
        run_root / "outer-select",
        label="outer-selection root",
        require_empty=False,
        exact_mode=0o700,
    )
    tracks_root = _trusted_directory(
        outer_root / "tracks",
        label="outer-selection tracks root",
        require_empty=False,
        exact_mode=0o700,
    )
    _require_exact_directory_names(
        outer_root,
        ("global", "tracks"),
        label="outer-selection root inventory",
    )
    runs = ordered_policy_runs()
    rotations = ordered_rotations()
    _require_exact_directory_names(
        tracks_root,
        tuple(run.track_id for run in runs),
        label="outer-selection 220-leaf inventory",
    )

    # Physical outer-select/global is the first phase opened here. No stage
    # outcome capsule is authenticated until the 220-marker snapshot below.
    outer_global_seal = verify_phase(
        outer_root / "global",
        expected_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
        expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_outer_selection_campaign_seal_sha256,
    )
    protocol, prepare, selection, _update_seal, update, outer = _capture_global_capabilities(
        run_root,
        publication_identity=publication_identity,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
        outer_global_seal=outer_global_seal,
    )
    outer_rows = tuple(outer.index_row(run=run) for run in runs)
    prelaunch_outer_markers = tuple(
        _observe_phase_marker_sha256(
            run_root / outer_selection_relative_path(run),
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=row.leaf_seal_sha256,
        )
        for run, row in zip(runs, outer_rows, strict=True)
    )

    # The start barrier has now passed. Only here may the controller capture
    # stage/global and its twenty outer-outcome capsules.
    stage = verify_stage_manifest(
        stage_root,
        expected_source_anchors=expected_source_anchors,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    stage_capability = authenticate_stage_manifest_for_controller(
        stage,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    reveal_seal = verify_phase(
        run_root / "reveal" / "global",
        expected_artifact=REVEAL_CAMPAIGN_ARTIFACT,
        expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    reveal = verify_reveal_campaign_capability(
        RevealCampaignCapability(reveal_seal, publication_identity),
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage_capability,
        selection_barrier=selection,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )

    outcome_seals: list[PhaseSeal] = []
    prediction_seals: list[PhaseSeal] = []
    marker_authorities: list[_MarkerAuthority] = [
        _MarkerAuthority(
            stage_root / "global", GLOBAL_PAYLOAD_PATHS, expected_stage_global_seal_sha256
        ),
        _MarkerAuthority(
            run_root / "protocol", PROTOCOL_PAYLOAD_PATHS, expected_protocol_seal_sha256
        ),
        _MarkerAuthority(
            run_root / "prepare" / "global",
            CAMPAIGN_PAYLOAD_PATHS,
            expected_prepare_campaign_seal_sha256,
        ),
        _MarkerAuthority(
            run_root / "select" / "global",
            CAMPAIGN_BARRIER_PAYLOAD_PATHS,
            expected_selection_campaign_seal_sha256,
        ),
        _MarkerAuthority(
            run_root / "reveal" / "global",
            REVEAL_CAMPAIGN_PAYLOAD_PATHS,
            expected_reveal_campaign_seal_sha256,
        ),
        _MarkerAuthority(
            run_root / "update" / "global",
            UPDATE_CAMPAIGN_PAYLOAD_PATHS,
            expected_update_campaign_seal_sha256,
        ),
        _MarkerAuthority(
            outer_root / "global",
            OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
            expected_outer_selection_campaign_seal_sha256,
        ),
    ]
    for spec in rotations:
        entry = stage_capability.leaf(spec=spec, role=OUTER_OUTCOME_ROLE)
        capsule = authenticate_stage_leaf_for_controller(
            stage_root / entry.relative_path,
            expected_leaf=entry,
            expected_source_anchors=expected_source_anchors,
        )
        outcome_seals.append(capsule.seal)
        outcome_root = stage_root / entry.relative_path
        marker_authorities.append(
            _MarkerAuthority(outcome_root, entry.payload_paths, entry.leaf_seal_sha256)
        )
        prediction_digest = prepare.leaf_seal_sha256(spec=spec, role=PREDICTION_VIEW_ROLE)
        prediction_root = run_root / "prepare" / "rotations" / spec.rotation_id / "prediction-view"
        prediction_seal = verify_phase(
            prediction_root,
            expected_artifact=PREDICTION_VIEW_ARTIFACT,
            expected_payload_paths=PREDICTION_VIEW_PAYLOAD_PATHS,
            expected_seal_sha256=prediction_digest,
        )
        prediction_seals.append(prediction_seal)
        marker_authorities.append(
            _MarkerAuthority(prediction_root, PREDICTION_VIEW_PAYLOAD_PATHS, prediction_digest)
        )

    reveal_leaf_seals: list[PhaseSeal] = []
    evidence_seals: list[PhaseSeal] = []
    view_seals: list[PhaseSeal] = []
    selection_seals: list[PhaseSeal] = []
    for run, outer_row in zip(runs, outer_rows, strict=True):
        reveal_row = reveal.index_row(run=run)
        reveal_root = run_root / pool_reveal_relative_path(run)
        reveal_leaf = verify_phase(
            reveal_root,
            expected_artifact=POOL_REVEAL_ARTIFACT,
            expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
            expected_seal_sha256=reveal_row.leaf_seal_sha256,
        )
        evidence_row = update.outer_evidence_row(run=run)
        evidence_root = run_root / evidence_row.relative_path
        evidence = verify_phase(
            evidence_root,
            expected_artifact=OUTER_EVIDENCE_ARTIFACT,
            expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
            expected_seal_sha256=evidence_row.leaf_seal_sha256,
        )
        view_row = update.outer_view_row(run=run)
        view_root = run_root / view_row.relative_path
        view = verify_phase(
            view_root,
            expected_artifact=OUTER_VIEW_ARTIFACT,
            expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
            expected_seal_sha256=view_row.leaf_seal_sha256,
        )
        selection_root = run_root / outer_selection_relative_path(run)
        selected = verify_phase(
            selection_root,
            expected_artifact=OUTER_SELECTION_ARTIFACT,
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=outer_row.leaf_seal_sha256,
        )
        reveal_leaf_seals.append(reveal_leaf)
        evidence_seals.append(evidence)
        view_seals.append(view)
        selection_seals.append(selected)
        marker_authorities.extend(
            (
                _MarkerAuthority(
                    reveal_root, POOL_REVEAL_PAYLOAD_PATHS, reveal_row.leaf_seal_sha256
                ),
                _MarkerAuthority(
                    evidence_root,
                    OUTER_EVIDENCE_PAYLOAD_PATHS,
                    evidence_row.leaf_seal_sha256,
                ),
                _MarkerAuthority(view_root, OUTER_VIEW_PAYLOAD_PATHS, view_row.leaf_seal_sha256),
                _MarkerAuthority(
                    selection_root, OUTER_SELECTION_PAYLOAD_PATHS, outer_row.leaf_seal_sha256
                ),
            )
        )
    markers = tuple(marker_authorities)
    if len(markers) != FINALIZE_PREDECESSOR_COUNT:
        raise AssertionError("finalize controller marker census changed")
    observed_markers = _observe_markers(markers)

    request = FinalizeWorkerRequest(
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage_capability,
        prepare_campaign=prepare,
        selection_campaign_seal=selection,
        reveal_campaign=reveal,
        update_campaign=update,
        outer_selection_campaign=outer,
        outer_outcome_vault_seals=tuple(outcome_seals),
        prediction_view_seals=tuple(prediction_seals),
        reveal_leaf_seals=tuple(reveal_leaf_seals),
        outer_evidence_seals=tuple(evidence_seals),
        outer_view_seals=tuple(view_seals),
        outer_selection_leaf_seals=tuple(selection_seals),
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
    )
    return _FinalizeRequestCapture(
        request=request,
        stage_root=stage_root,
        run_root=run_root,
        outer_root=outer_root,
        tracks_root=tracks_root,
        marker_authorities=markers,
        observed_markers=observed_markers,
        observed_outer_selection_markers=prelaunch_outer_markers,
    )


def _reobserve_finalize_request_capture(
    capture: _FinalizeRequestCapture,
    *,
    marker_change_message: str,
    outer_change_message: str,
    inventory_label_prefix: str,
    expected_run_inventory: tuple[str, ...],
) -> None:
    observed_markers = _observe_markers(capture.marker_authorities)
    if observed_markers != capture.observed_markers:
        raise RuntimeError(marker_change_message)
    runs = ordered_policy_runs()
    observed_outer = tuple(
        _observe_phase_marker_sha256(
            capture.run_root / outer_selection_relative_path(run),
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=digest,
        )
        for run, digest in zip(
            runs,
            capture.observed_outer_selection_markers,
            strict=True,
        )
    )
    if observed_outer != capture.observed_outer_selection_markers:
        raise RuntimeError(outer_change_message)
    _require_exact_directory_names(
        capture.tracks_root,
        tuple(run.track_id for run in runs),
        label=f"{inventory_label_prefix} outer-selection leaf inventory",
    )
    _require_exact_directory_names(
        capture.outer_root,
        ("global", "tracks"),
        label=f"{inventory_label_prefix} outer-selection root inventory",
    )
    _require_exact_directory_names(
        capture.run_root,
        expected_run_inventory,
        label=f"{inventory_label_prefix} run-root inventory",
    )


_REQUEST_METADATA_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(metadata, field) for field in _REQUEST_METADATA_FIELDS)


def _require_private_parent_metadata(
    metadata: os.stat_result,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    identity = (metadata.st_dev, metadata.st_ino)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise RuntimeError("finalize request parent changed or is not an exact private directory")
    return identity


def _require_published_request_metadata(
    metadata: os.stat_result,
    *,
    expected_identity: tuple[int, int] | None,
    expected_size: int,
) -> tuple[int, int]:
    identity = (metadata.st_dev, metadata.st_ino)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_size != expected_size
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise RuntimeError("published finalize request has an unsafe file identity")
    return identity


def _request_output_target(value: str | Path) -> tuple[Path, Path]:
    requested = Path(os.path.abspath(os.fspath(value)))
    if _PORTABLE_PATH_COMPONENT.fullmatch(requested.name) is None:
        raise ValueError("finalize request output must use one portable path component")
    parent = _trusted_directory(
        requested.parent,
        label="finalize request output parent",
        require_empty=False,
        exact_mode=0o700,
    )
    return parent / requested.name, parent


def _readback_finalize_request(
    parent_descriptor: int,
    name: str,
    *,
    payload: bytes,
    expected_file_identity: tuple[int, int],
) -> tuple[os.stat_result, str]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        _require_published_request_metadata(
            before,
            expected_identity=expected_file_identity,
            expected_size=len(payload),
        )
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            if offset + len(chunk) > len(payload) or chunk != payload[offset : offset + len(chunk)]:
                raise RuntimeError("published finalize request differs from canonical bytes")
            digest.update(chunk)
            offset += len(chunk)
        if offset != len(payload):
            raise RuntimeError("published finalize request was truncated during readback")
        after = os.fstat(descriptor)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise RuntimeError("published finalize request changed during readback")
    finally:
        os.close(descriptor)
    named_after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    _require_published_request_metadata(
        named_after,
        expected_identity=expected_file_identity,
        expected_size=len(payload),
    )
    if _metadata_fingerprint(after) != _metadata_fingerprint(named_after):
        raise RuntimeError("published finalize request identity changed after readback")
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("published finalize request digest differs from canonical bytes")
    return after, observed_sha256


def _unlink_created_finalize_request(
    parent_descriptor: int,
    name: str,
    *,
    expected_file_identity: tuple[int, int],
) -> None:
    metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (metadata.st_dev, metadata.st_ino) != expected_file_identity:
        raise RuntimeError("refusing to unlink a replaced finalize request output")
    os.unlink(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _publish_finalize_request(
    output: Path,
    parent: Path,
    payload: bytes,
) -> _PublishedFinalizeRequest:
    """Durably publish one no-replace, read-back-verified canonical request."""

    trusted_parent = _trusted_directory(
        parent,
        label="finalize request output parent during publication",
        require_empty=False,
        exact_mode=0o700,
    )
    if trusted_parent != parent or output.parent != parent:
        raise RuntimeError("finalize request output parent changed before publication")
    named_parent_before = os.lstat(parent)
    parent_identity = _require_private_parent_metadata(named_parent_before)
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    created_identity: tuple[int, int] | None = None
    try:
        opened_parent = os.fstat(parent_descriptor)
        _require_private_parent_metadata(
            opened_parent,
            expected_identity=parent_identity,
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(output.name, flags, 0o400, dir_fd=parent_descriptor)
        created = os.fstat(descriptor)
        created_identity = (created.st_dev, created.st_ino)
        os.fchmod(descriptor, 0o400)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        written = os.fstat(descriptor)
        _require_published_request_metadata(
            written,
            expected_identity=created_identity,
            expected_size=len(payload),
        )
        completed_descriptor = descriptor
        descriptor = -1
        os.close(completed_descriptor)

        if created_identity is None:
            raise AssertionError("finalize request publication lost its created identity")
        readback, payload_sha256 = _readback_finalize_request(
            parent_descriptor,
            output.name,
            payload=payload,
            expected_file_identity=created_identity,
        )
        if _metadata_fingerprint(written) != _metadata_fingerprint(readback):
            raise RuntimeError("published finalize request changed before readback")
        os.fsync(parent_descriptor)
        named_parent_after = os.lstat(parent)
        opened_parent_after = os.fstat(parent_descriptor)
        _require_private_parent_metadata(
            named_parent_after,
            expected_identity=parent_identity,
        )
        _require_private_parent_metadata(
            opened_parent_after,
            expected_identity=parent_identity,
        )
        if (named_parent_after.st_dev, named_parent_after.st_ino) != (
            opened_parent_after.st_dev,
            opened_parent_after.st_ino,
        ):
            raise RuntimeError("finalize request parent identity changed during publication")
        return _PublishedFinalizeRequest(
            path=output,
            parent_identity=parent_identity,
            file_identity=created_identity,
            payload_sha256=payload_sha256,
        )
    except BaseException:
        if descriptor >= 0 or created_identity is not None:
            try:
                if created_identity is None:
                    # The exclusive create has already installed a pathname. Keep
                    # its descriptor open and retry identity capture so an early
                    # fstat failure cannot strand an output that blocks every
                    # later no-replace attempt. ``os.stat(fd)`` is an independent
                    # CPython entry point for the same descriptor-pinned identity.
                    try:
                        created = os.fstat(descriptor)
                    except BaseException:
                        created = os.stat(descriptor)
                    created_identity = (created.st_dev, created.st_ino)
                with suppress(FileNotFoundError):
                    _unlink_created_finalize_request(
                        parent_descriptor,
                        output.name,
                        expected_file_identity=created_identity,
                    )
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"failed finalize request publication could not be rolled back: {output}"
                ) from rollback_error
        raise
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)


def _rollback_published_finalize_request(publication: _PublishedFinalizeRequest) -> None:
    parent = _trusted_directory(
        publication.path.parent,
        label="finalize request output parent during rollback",
        require_empty=False,
        exact_mode=0o700,
    )
    named_parent = os.lstat(parent)
    _require_private_parent_metadata(
        named_parent,
        expected_identity=publication.parent_identity,
    )
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _require_private_parent_metadata(
            os.fstat(parent_descriptor),
            expected_identity=publication.parent_identity,
        )
        _unlink_created_finalize_request(
            parent_descriptor,
            publication.path.name,
            expected_file_identity=publication.file_identity,
        )
    finally:
        os.close(parent_descriptor)


def _reobserve_published_finalize_request(
    publication: _PublishedFinalizeRequest,
    payload: bytes,
) -> None:
    parent = _trusted_directory(
        publication.path.parent,
        label="finalize request output parent during reobservation",
        require_empty=False,
        exact_mode=0o700,
    )
    named_parent = os.lstat(parent)
    _require_private_parent_metadata(
        named_parent,
        expected_identity=publication.parent_identity,
    )
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _require_private_parent_metadata(
            os.fstat(parent_descriptor),
            expected_identity=publication.parent_identity,
        )
        _metadata, observed_sha256 = _readback_finalize_request(
            parent_descriptor,
            publication.path.name,
            payload=payload,
            expected_file_identity=publication.file_identity,
        )
        if observed_sha256 != publication.payload_sha256:
            raise RuntimeError("published finalize request digest changed after input checks")
        os.fsync(parent_descriptor)
        _require_private_parent_metadata(
            os.lstat(parent),
            expected_identity=publication.parent_identity,
        )
    finally:
        os.close(parent_descriptor)


def measure_finalize_campaign_request(
    *,
    stage_root: str | Path,
    expected_source_anchors: Mapping[str, object],
    run_root: str | Path,
    request_output: str | Path,
    expected_protocol_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
) -> FinalizeRequestCapacityMeasurement:
    """Authenticate, measure, and publish the exact finalizer request without running it."""

    stage_path, final_root = _validate_finalize_capture_arguments(
        stage_root=stage_root,
        expected_source_anchors=expected_source_anchors,
        run_root=run_root,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
        publication_identity=publication_identity,
    )
    output, output_parent = _request_output_target(request_output)
    roots = (stage_path, final_root, output_parent)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ValueError("stage, run, and request-output roots must be disjoint trees")

    capture = _capture_finalize_campaign_request(
        stage_root=stage_path,
        expected_source_anchors=expected_source_anchors,
        run_root=final_root,
        publication_identity=publication_identity,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
    )
    payload, measurement = encode_and_measure_finalize_worker_request(capture.request)
    if payload != capture.request.canonical_bytes():
        raise RuntimeError("measured finalize request differs from canonical request bytes")
    publication = _publish_finalize_request(output, output_parent, payload)
    try:
        if publication.payload_sha256 != measurement.request_sha256:
            raise RuntimeError("published finalize request differs from its measured digest")
        _reobserve_finalize_request_capture(
            capture,
            marker_change_message="finalize inputs changed while the request was measured",
            outer_change_message=(
                "outer-selection start barrier changed while the request was measured"
            ),
            inventory_label_prefix="post-measurement",
            expected_run_inventory=_PRE_FINALIZE_RUN_INVENTORY,
        )
        _reobserve_published_finalize_request(publication, payload)
    except BaseException:
        try:
            _rollback_published_finalize_request(publication)
        except BaseException as rollback_error:
            raise RuntimeError(
                "finalize request measurement failed and its published request could not "
                f"be rolled back safely: {publication.path}"
            ) from rollback_error
        raise
    return measurement


def supervise_finalize_campaign(
    *,
    stage_root: str | Path,
    expected_source_anchors: Mapping[str, object],
    run_root: str | Path,
    worker_scratch_root: str | Path,
    expected_protocol_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    expected_outer_selection_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> FinalizeSupervisionResult:
    """Run the one finalizer after proving the complete 220-track barrier."""

    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("finalization requires an exact publication identity")
    if not isinstance(expected_source_anchors, Mapping):
        raise TypeError("finalization source anchors must be a mapping")
    if type(timeout_seconds) is not float or not 0.0 < timeout_seconds <= 24 * 60 * 60:
        raise ValueError("finalize timeout must be one bounded positive float")
    stage_path, final_root = _validate_finalize_capture_arguments(
        stage_root=stage_root,
        expected_source_anchors=expected_source_anchors,
        run_root=run_root,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
        publication_identity=publication_identity,
    )
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    roots = (stage_path, final_root, scratch)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ValueError("stage, run, and worker scratch roots must be disjoint trees")
    _require_unrelated_same_filesystem(final_root, scratch)
    capture = _capture_finalize_campaign_request(
        stage_root=stage_path,
        expected_source_anchors=expected_source_anchors,
        run_root=final_root,
        publication_identity=publication_identity,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_campaign_seal_sha256=expected_selection_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        expected_outer_selection_campaign_seal_sha256=(
            expected_outer_selection_campaign_seal_sha256
        ),
    )
    request = capture.request
    request_payload = request.canonical_bytes()
    finalize_root = _create_private_directory(final_root, "finalize")
    result = _launch_fresh_worker(
        FINALIZE_WORKER_ROLE,
        request_payload,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    destination = finalize_root / "global"
    relocated = False
    try:
        _require_worker_identity(result)
        attestation = finalize_attestation_from_bytes(result.payload)
        _attestation_matches_request(attestation, request)
        _require_output_inventory(result)
        predecessors = finalize_predecessors(request)
        source = result.output / "global"
        _observe_phase_marker_sha256(
            source,
            expected_payload_paths=FINALIZE_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.finalize_global_seal_sha256,
        )
        candidate = verify_phase(
            source,
            expected_artifact=FINALIZE_ARTIFACT,
            expected_payload_paths=FINALIZE_PAYLOAD_PATHS,
            expected_predecessor_seals=predecessors,
            expected_seal_sha256=attestation.finalize_global_seal_sha256,
        )
        publication_identity.verify_metadata(
            candidate.metadata_json,
            phase="finalize",
            scope_id="global",
        )
        if candidate.payload_sha256 != attestation.payload_sha256:
            raise ValueError("finalize phase payload digests differ from attestation")
        if candidate.read_payload_bytes("summary.json") != canonical_json_bytes(
            _expected_summary_document(attestation)
        ):
            raise ValueError("finalize summary differs from payload-free attestation")

        _reobserve_finalize_request_capture(
            capture,
            marker_change_message="finalize inputs changed while the worker ran",
            outer_change_message=("outer-selection start barrier changed during finalization"),
            inventory_label_prefix="post-finalize",
            expected_run_inventory=_COMPLETE_RUN_INVENTORY,
        )

        relocate_sealed_phase_noreplace(
            source,
            destination,
            expected_seal_sha256=attestation.finalize_global_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        relocated = True
        final_seal = verify_phase(
            destination,
            expected_artifact=FINALIZE_ARTIFACT,
            expected_payload_paths=FINALIZE_PAYLOAD_PATHS,
            expected_predecessor_seals=predecessors,
            expected_seal_sha256=attestation.finalize_global_seal_sha256,
        )
        publication_identity.verify_metadata(
            final_seal.metadata_json,
            phase="finalize",
            scope_id="global",
        )
        if final_seal.payload_sha256 != attestation.payload_sha256:
            raise RuntimeError("finalize phase changed during no-replace relocation")
        _require_exact_directory_names(
            finalize_root,
            ("global",),
            label="completed finalize root inventory",
        )
        _require_exact_directory_names(
            final_root,
            _COMPLETE_RUN_INVENTORY,
            label="completed run-root inventory",
        )
        _release_successful_outbox(result)
    except BaseException as error:
        if relocated:
            raise RuntimeError(
                "finalize post-release verification failed after finalize/global was "
                f"published at {destination}; abandon this run; diagnostic outbox state "
                f"remains at {result.outbox}"
            ) from error
        raise RuntimeError(
            f"finalize result was not accepted; outbox retained at {result.outbox}"
        ) from error
    return FinalizeSupervisionResult(
        publication_identity=publication_identity,
        attestation=attestation,
        input_marker_count=len(capture.marker_authorities),
        worker_process_count=FINALIZE_WORKER_COUNT,
    )


__all__ = [
    "FINALIZE_ATTESTATION_ARTIFACT",
    "FINALIZE_DIRECT_LEAF_COUNT",
    "FINALIZE_PREDECESSOR_COUNT",
    "FINALIZE_SUPERVISION_ARTIFACT",
    "FinalizeAttestation",
    "FinalizeSupervisionResult",
    "finalize_attestation_from_bytes",
    "finalize_predecessors",
    "measure_finalize_campaign_request",
    "run_finalize_worker_request",
    "supervise_finalize_campaign",
]

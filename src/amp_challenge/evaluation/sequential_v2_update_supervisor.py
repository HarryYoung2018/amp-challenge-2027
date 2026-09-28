"""Fresh-executable supervision for the sealed sequential-v2 UPDATE graph.

The controller launches exactly 220 state workers, twenty component workers,
220 paired projection workers, and one update-global worker.  Worker requests
contain rootless capabilities only.  Results are accepted only after their
role, process identity, payload-free attestation, private-outbox inventory,
and checksum marker agree with the controller's independently held authority.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_commitments import (
    CAMPAIGN_BARRIER_ARTIFACT,
    CAMPAIGN_BARRIER_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    BASE_UPDATE_ARTIFACT,
    BASE_UPDATE_PAYLOAD_PATHS,
    BASE_UPDATE_ROLE,
    CAMPAIGN_PAYLOAD_PATHS,
    PREPARE_CAMPAIGN_ARTIFACT,
    PROTOCOL_ARTIFACT,
    PROTOCOL_PAYLOAD_PATHS,
    PrepareCampaignCapability,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_runs_for_rotation,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    POOL_REVEAL_ARTIFACT,
    POOL_REVEAL_PAYLOAD_PATHS,
    REVEAL_CAMPAIGN_ARTIFACT,
    REVEAL_CAMPAIGN_PAYLOAD_PATHS,
    RevealCampaignCapability,
    pool_reveal_relative_path,
    verify_reveal_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_jsonl_bytes,
    relocate_sealed_phase_noreplace,
    verify_phase,
)
from amp_challenge.evaluation.sequential_v2_stage import (
    OUTER_METADATA_ROLE,
    AuthenticatedLeafCapsule,
    StageManifestCapability,
    authenticate_stage_leaf_for_controller,
    authenticate_stage_manifest_for_controller,
    verify_stage_manifest,
    verify_stage_manifest_capability,
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
    _validate_phase_attestation,
)
from amp_challenge.evaluation.sequential_v2_update import (
    OUTER_COMPONENT_PAYLOAD_PATHS,
    OUTER_COMPONENTS_ARTIFACT,
    OUTER_EVIDENCE_PAYLOAD_PATHS,
    OUTER_VIEW_PAYLOAD_PATHS,
    UPDATE_CAMPAIGN_ARTIFACT,
    UPDATE_CAMPAIGN_PAYLOAD_PATHS,
    UPDATE_STATE_ARTIFACT,
    UPDATE_STATE_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_update_campaign import (
    UpdateCampaignCapability,
    verify_update_campaign_barrier,
)
from amp_challenge.evaluation.sequential_v2_update_outer import (
    OuterComponentAttestation,
    OuterProjectionAttestation,
    outer_component_attestation_from_bytes,
    outer_component_relative_path,
    outer_evidence_relative_path,
    outer_projection_attestation_from_bytes,
    outer_view_relative_path,
)
from amp_challenge.evaluation.sequential_v2_update_state import (
    UpdateStateAttestation,
    update_state_attestation_from_bytes,
    update_state_relative_path,
)
from amp_challenge.evaluation.sequential_v2_update_wire import (
    EXPECTED_UPDATE_LEAVES,
    UPDATE_BARRIER_WORKER_ROLE,
    UPDATE_COMPONENT_WORKER_ROLE,
    UPDATE_PROJECTION_WORKER_ROLE,
    UPDATE_STATE_WORKER_ROLE,
    UpdateBarrierWorkerRequest,
    UpdateComponentWorkerRequest,
    UpdateProjectionWorkerRequest,
    UpdateStateWorkerRequest,
)
from amp_challenge.evaluation.sequential_v2_wire import (
    phase_publication_attestation_from_bytes,
)

SCHEMA_VERSION = 1
UPDATE_SUPERVISION_ARTIFACT = "sequential_v2_update_supervision_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UPDATE_WORKER_PROCESS_COUNT = 461


def _fresh_destination(
    destination: str | Path,
    *,
    label: str,
    expected_name: str,
) -> tuple[Path, Path]:
    requested = Path(destination)
    parent = _trusted_directory(
        requested.parent,
        label=f"{label} destination parent",
        require_empty=False,
        exact_mode=0o700,
    )
    final = parent / requested.name
    if os.path.lexists(final):
        raise FileExistsError(f"{label} destination must be fresh and never replace an entry")
    if final.name != expected_name:
        raise ValueError(f"{label} destination must end in {expected_name!r}")
    return parent, final


def _launcher_scratch(destination_parent: Path, worker_scratch_root: str | Path) -> Path:
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(destination_parent, scratch)
    return scratch


def _require_worker_identity(result: _FreshWorkerResult, *, role: str) -> None:
    if result.role != role or type(result.process_pid) is not int or result.process_pid <= 0:
        raise ValueError("update worker process identity differs from its launched role")


def _require_output_inventory(result: _FreshWorkerResult, expected_names: tuple[str, ...]) -> None:
    actual = tuple(sorted(entry.name for entry in os.scandir(result.output)))
    if actual != tuple(sorted(expected_names)):
        raise RuntimeError("update worker outbox inventory differs from its exact role")


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class UpdateSupervisionResult:
    """Authenticated update-global release from exactly 461 worker leaders."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    stage_manifest: StageManifestCapability
    prepare_campaign: PrepareCampaignCapability
    selection_barrier: PhaseSeal
    reveal_campaign: RevealCampaignCapability
    expected_protocol_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str
    state_attestations: tuple[UpdateStateAttestation, ...]
    component_attestations: tuple[OuterComponentAttestation, ...]
    projection_attestations: tuple[OuterProjectionAttestation, ...]
    observed_update_leaf_seal_sha256s: tuple[str, ...]
    update_campaign: UpdateCampaignCapability
    worker_process_count: int

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("update supervision requires an exact publication identity")
        expected = (
            self.expected_protocol_seal_sha256,
            self.expected_prepare_campaign_seal_sha256,
            self.expected_stage_global_seal_sha256,
            self.expected_selection_barrier_seal_sha256,
            self.expected_reveal_campaign_seal_sha256,
        )
        for index, value in enumerate(expected):
            _require_sha256(value, label=f"update supervision upstream authority {index}")
        if (
            type(self.protocol_capability) is not ProtocolCapability
            or self.protocol_capability.seal.seal_sha256 != self.expected_protocol_seal_sha256
            or type(self.stage_manifest) is not StageManifestCapability
            or self.stage_manifest.seal.seal_sha256 != self.expected_stage_global_seal_sha256
            or type(self.prepare_campaign) is not PrepareCampaignCapability
            or self.prepare_campaign.seal.seal_sha256 != self.expected_prepare_campaign_seal_sha256
            or type(self.selection_barrier) is not PhaseSeal
            or self.selection_barrier.seal_sha256 != self.expected_selection_barrier_seal_sha256
            or type(self.reveal_campaign) is not RevealCampaignCapability
            or self.reveal_campaign.seal.seal_sha256 != self.expected_reveal_campaign_seal_sha256
        ):
            raise ValueError("update supervision globals differ from external authorities")
        if type(self.update_campaign) is not UpdateCampaignCapability:
            raise TypeError("update supervision requires an exact update campaign capability")
        if type(self.worker_process_count) is not int or self.worker_process_count != (
            _UPDATE_WORKER_PROCESS_COUNT
        ):
            raise ValueError("update supervision must contain exactly 461 worker leaders")

        # The barrier request's constructor is also the canonical immutable
        # census/order/authority validator for all 680 worker leaf results.
        UpdateBarrierWorkerRequest(
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            prepare_campaign_seal=self.prepare_campaign.seal,
            reveal_campaign_seal=self.reveal_campaign.seal,
            stage_manifest_seal=self.stage_manifest.seal,
            selection_barrier=self.selection_barrier,
            state_attestations=self.state_attestations,
            component_attestations=self.component_attestations,
            projection_attestations=self.projection_attestations,
            expected_update_leaf_seal_sha256s=self.observed_update_leaf_seal_sha256s,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=(self.expected_reveal_campaign_seal_sha256),
        )
        verify_update_campaign_barrier(
            self.update_campaign.seal,
            state_attestations=self.state_attestations,
            component_attestations=self.component_attestations,
            projection_attestations=self.projection_attestations,
            expected_update_leaf_seal_sha256s=self.observed_update_leaf_seal_sha256s,
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            stage_manifest_capability=self.stage_manifest,
            prepare_campaign=self.prepare_campaign,
            reveal_campaign=self.reveal_campaign,
            selection_barrier=self.selection_barrier,
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=(self.expected_reveal_campaign_seal_sha256),
            expected_update_campaign_seal_sha256=self.update_campaign.seal.seal_sha256,
        )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": UPDATE_SUPERVISION_ARTIFACT,
            "protocol_seal_sha256": self.expected_protocol_seal_sha256,
            "prepare_global_seal_sha256": self.expected_prepare_campaign_seal_sha256,
            "stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
            "select_global_seal_sha256": self.expected_selection_barrier_seal_sha256,
            "reveal_global_seal_sha256": self.expected_reveal_campaign_seal_sha256,
            "update_global_seal_sha256": self.update_campaign.seal.seal_sha256,
            "state_worker_count": len(self.state_attestations),
            "component_worker_count": len(self.component_attestations),
            "projection_worker_count": len(self.projection_attestations),
            "update_leaf_count": len(self.observed_update_leaf_seal_sha256s),
            "barrier_worker_count": 1,
            "worker_process_count": self.worker_process_count,
        }


def launch_update_state_worker(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    base_update_seal: PhaseSeal,
    reveal_campaign: RevealCampaignCapability,
    reveal_seal: PhaseSeal,
    stage_manifest: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> UpdateStateAttestation:
    """Publish one state leaf through its sole two-label fresh worker."""

    parent, final = _fresh_destination(
        destination,
        label="update-state",
        expected_name="state",
    )
    scratch = _launcher_scratch(parent, worker_scratch_root)
    request = UpdateStateWorkerRequest(
        run=run,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign_seal=prepare_campaign.seal,
        base_update_seal=base_update_seal,
        reveal_campaign_seal=reveal_campaign.seal,
        reveal_seal=reveal_seal,
        stage_manifest_seal=stage_manifest.seal,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    reveal_row = reveal_campaign.index_row(run=run)
    expected_base = prepare_campaign.leaf_seal_sha256(spec=run.rotation, role=BASE_UPDATE_ROLE)
    if (
        base_update_seal.seal_sha256 != expected_base
        or reveal_seal.seal_sha256 != reveal_row.leaf_seal_sha256
    ):
        raise ValueError("update-state sources differ from their campaign indexes")
    result = _launch_fresh_worker(
        UPDATE_STATE_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        _require_worker_identity(result, role=UPDATE_STATE_WORKER_ROLE)
        attestation = update_state_attestation_from_bytes(result.payload)
        base_payloads = dict(base_update_seal.payload_sha256)
        reveal_payloads = dict(reveal_seal.payload_sha256)
        if (
            attestation.run != run
            or attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
            or attestation.prepare_global_seal_sha256 != expected_prepare_campaign_seal_sha256
            or attestation.base_update_leaf_seal_sha256 != expected_base
            or attestation.base_contexts_payload_sha256 != base_payloads["base-contexts.jsonl"]
            or attestation.base_model_payload_sha256 != base_payloads["base-model.json"]
            or attestation.base_model_state_sha256
            != prepare_campaign.base_model_state_sha256(spec=run.rotation)
            or attestation.reveal_global_seal_sha256 != expected_reveal_campaign_seal_sha256
            or attestation.reveal_leaf_seal_sha256 != reveal_row.leaf_seal_sha256
            or attestation.reveal_contexts_payload_sha256 != reveal_payloads["contexts.jsonl"]
            or attestation.revealed_context_count != reveal_row.revealed_context_count
            or attestation.revealed_example_ids_sha256 != reveal_row.revealed_example_ids_sha256
        ):
            raise ValueError("update-state attestation differs from launched authority")
        _require_output_inventory(result, ("state",))
        _observe_phase_marker_sha256(
            result.output / "state",
            expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.state_leaf_seal_sha256,
        )
        relocate_sealed_phase_noreplace(
            result.output / "state",
            final,
            expected_seal_sha256=attestation.state_leaf_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        _observe_phase_marker_sha256(
            final,
            expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.state_leaf_seal_sha256,
        )
        _release_successful_outbox(result)
        return attestation
    except BaseException as error:
        raise RuntimeError(
            f"update-state result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_update_component_worker(
    destination: str | Path,
    *,
    spec: RotationSpec,
    state_attestations: tuple[UpdateStateAttestation, ...],
    expected_state_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest: StageManifestCapability,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_stage_global_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> OuterComponentAttestation:
    """Publish one rotation component leaf after its eleven state markers."""

    parent, final = _fresh_destination(
        destination,
        label="update component",
        expected_name="outer-components",
    )
    scratch = _launcher_scratch(parent, worker_scratch_root)
    request = UpdateComponentWorkerRequest(
        spec=spec,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_manifest_seal=stage_manifest.seal,
        outer_metadata_capsule=outer_metadata_capsule,
        state_attestations=state_attestations,
        expected_state_leaf_seal_sha256s=expected_state_leaf_seal_sha256s,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    result = _launch_fresh_worker(
        UPDATE_COMPONENT_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        _require_worker_identity(result, role=UPDATE_COMPONENT_WORKER_ROLE)
        attestation = outer_component_attestation_from_bytes(result.payload)
        candidate_binding = outer_metadata_capsule.entry.id_stream_map["outer_support_sequence_ids"]
        if (
            attestation.spec != spec
            or attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
            or attestation.stage_global_seal_sha256 != expected_stage_global_seal_sha256
            or attestation.outer_metadata_leaf_seal_sha256
            != outer_metadata_capsule.entry.leaf_seal_sha256
            or attestation.state_leaf_seal_sha256s != expected_state_leaf_seal_sha256s
            or attestation.candidate_count != candidate_binding.count
            or attestation.candidate_ids_sha256 != candidate_binding.sha256
        ):
            raise ValueError("update component attestation differs from launched authority")
        _require_output_inventory(result, ("component",))
        _observe_phase_marker_sha256(
            result.output / "component",
            expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.component_leaf_seal_sha256,
        )
        relocate_sealed_phase_noreplace(
            result.output / "component",
            final,
            expected_seal_sha256=attestation.component_leaf_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        _observe_phase_marker_sha256(
            final,
            expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.component_leaf_seal_sha256,
        )
        _release_successful_outbox(result)
        return attestation
    except BaseException as error:
        raise RuntimeError(
            f"update component result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_update_projection_worker(
    view_destination: str | Path,
    evidence_destination: str | Path,
    *,
    run: PolicyRunSpec,
    state_seal: PhaseSeal,
    state_attestation: UpdateStateAttestation,
    component_seal: PhaseSeal,
    component_attestation: OuterComponentAttestation,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    stage_manifest: StageManifestCapability,
    selection_barrier: PhaseSeal,
    outer_metadata_capsule: AuthenticatedLeafCapsule,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_state_leaf_seal_sha256: str,
    expected_component_leaf_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> OuterProjectionAttestation:
    """Publish one paired view/evidence result through one fresh worker."""

    view_parent, final_view = _fresh_destination(
        view_destination,
        label="update outer-view",
        expected_name="outer-view",
    )
    evidence_parent, final_evidence = _fresh_destination(
        evidence_destination,
        label="update outer-evidence",
        expected_name="outer-evidence",
    )
    if view_parent != evidence_parent or final_view == final_evidence:
        raise ValueError("paired projection destinations must be distinct siblings")
    scratch = _launcher_scratch(view_parent, worker_scratch_root)
    request = UpdateProjectionWorkerRequest(
        run=run,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign_seal=prepare_campaign.seal,
        reveal_campaign_seal=reveal_campaign.seal,
        stage_manifest_seal=stage_manifest.seal,
        selection_barrier=selection_barrier,
        state_seal=state_seal,
        state_attestation=state_attestation,
        component_seal=component_seal,
        component_attestation=component_attestation,
        outer_metadata_capsule=outer_metadata_capsule,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_state_leaf_seal_sha256=expected_state_leaf_seal_sha256,
        expected_component_leaf_seal_sha256=expected_component_leaf_seal_sha256,
    )
    result = _launch_fresh_worker(
        UPDATE_PROJECTION_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        _require_worker_identity(result, role=UPDATE_PROJECTION_WORKER_ROLE)
        attestation = outer_projection_attestation_from_bytes(result.payload)
        metadata_entry = outer_metadata_capsule.entry
        outer_examples = metadata_entry.id_stream_map["outer_metadata_example_ids"]
        outer_candidates = metadata_entry.id_stream_map["outer_support_sequence_ids"]
        if (
            attestation.run != run
            or attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
            or attestation.stage_global_seal_sha256 != expected_stage_global_seal_sha256
            or attestation.outer_metadata_leaf_seal_sha256 != metadata_entry.leaf_seal_sha256
            or attestation.state_leaf_seal_sha256 != expected_state_leaf_seal_sha256
            or attestation.state_leaf_seal_sha256 != state_attestation.state_leaf_seal_sha256
            or attestation.updated_model_payload_sha256
            != dict(state_attestation.payload_sha256)["updated-model.json"]
            or attestation.outer_component_leaf_seal_sha256 != expected_component_leaf_seal_sha256
            or attestation.outer_component_leaf_seal_sha256
            != component_attestation.component_leaf_seal_sha256
            or attestation.outer_components_payload_sha256
            != dict(component_attestation.payload_sha256)["outer-components.jsonl"]
            or attestation.outer_context_count != outer_examples.count
            or attestation.outer_example_ids_sha256 != outer_examples.sha256
            or attestation.outer_candidate_count != outer_candidates.count
            or attestation.outer_candidate_count != component_attestation.candidate_count
            or attestation.outer_candidate_ids_sha256 != outer_candidates.sha256
            or attestation.outer_candidate_ids_sha256 != component_attestation.candidate_ids_sha256
            or attestation.outer_component_count != component_attestation.component_count
            or attestation.outer_component_membership_count
            != component_attestation.component_membership_count
        ):
            raise ValueError("update projection attestation differs from launched authority")
        _require_output_inventory(result, ("outer-view", "outer-evidence"))

        # Both private markers are captured before either phase is made visible.
        # Publication and final observation remain view-first because evidence
        # has the view as a semantic predecessor.
        _observe_phase_marker_sha256(
            result.output / "outer-view",
            expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.outer_view_leaf_seal_sha256,
        )
        _observe_phase_marker_sha256(
            result.output / "outer-evidence",
            expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.outer_evidence_leaf_seal_sha256,
        )
        relocate_sealed_phase_noreplace(
            result.output / "outer-view",
            final_view,
            expected_seal_sha256=attestation.outer_view_leaf_seal_sha256,
            expected_payload_sha256=dict(attestation.outer_view_payload_sha256),
        )
        relocate_sealed_phase_noreplace(
            result.output / "outer-evidence",
            final_evidence,
            expected_seal_sha256=attestation.outer_evidence_leaf_seal_sha256,
            expected_payload_sha256=dict(attestation.outer_evidence_payload_sha256),
        )
        _observe_phase_marker_sha256(
            final_view,
            expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.outer_view_leaf_seal_sha256,
        )
        _observe_phase_marker_sha256(
            final_evidence,
            expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.outer_evidence_leaf_seal_sha256,
        )
        _release_successful_outbox(result)
        return attestation
    except BaseException as error:
        raise RuntimeError(
            f"update projection result was not accepted; outbox retained at {result.outbox}"
        ) from error


def _expected_update_index_bytes(
    state_attestations: tuple[UpdateStateAttestation, ...],
    component_attestations: tuple[OuterComponentAttestation, ...],
    projection_attestations: tuple[OuterProjectionAttestation, ...],
) -> bytes:
    documents = (
        *(item.index_document() for item in state_attestations),
        *(item.component_index_document() for item in component_attestations),
        *(item.view_index_document() for item in projection_attestations),
        *(item.evidence_index_document() for item in projection_attestations),
    )
    return canonical_jsonl_bytes(documents)


def launch_update_barrier_worker(
    destination: str | Path,
    *,
    state_attestations: tuple[UpdateStateAttestation, ...],
    component_attestations: tuple[OuterComponentAttestation, ...],
    projection_attestations: tuple[OuterProjectionAttestation, ...],
    expected_update_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest: StageManifestCapability,
    prepare_campaign: PrepareCampaignCapability,
    reveal_campaign: RevealCampaignCapability,
    selection_barrier: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> UpdateCampaignCapability:
    """Publish and fully reconstruct update/global through one fresh worker."""

    parent, final = _fresh_destination(
        destination,
        label="update-global",
        expected_name="global",
    )
    scratch = _launcher_scratch(parent, worker_scratch_root)
    request = UpdateBarrierWorkerRequest(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign_seal=prepare_campaign.seal,
        reveal_campaign_seal=reveal_campaign.seal,
        stage_manifest_seal=stage_manifest.seal,
        selection_barrier=selection_barrier,
        state_attestations=state_attestations,
        component_attestations=component_attestations,
        projection_attestations=projection_attestations,
        expected_update_leaf_seal_sha256s=expected_update_leaf_seal_sha256s,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    expected_index = _expected_update_index_bytes(
        state_attestations,
        component_attestations,
        projection_attestations,
    )
    result = _launch_fresh_worker(
        UPDATE_BARRIER_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        _require_worker_identity(result, role=UPDATE_BARRIER_WORKER_ROLE)
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=UPDATE_BARRIER_WORKER_ROLE,
            phase_artifact=UPDATE_CAMPAIGN_ARTIFACT,
            payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        )
        _require_output_inventory(result, ("global",))
        source = result.output / "global"
        _observe_phase_marker_sha256(
            source,
            expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )

        # update/global is label-free, so authenticate its exact 680-row index
        # and 684-predecessor graph before committing it to the final DAG.
        candidate_seal = verify_phase(
            source,
            expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
            expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        if candidate_seal.read_payload_bytes("update-index.jsonl") != expected_index:
            raise ValueError("update-global index differs from exact worker results")
        verify_update_campaign_barrier(
            candidate_seal,
            state_attestations=state_attestations,
            component_attestations=component_attestations,
            projection_attestations=projection_attestations,
            expected_update_leaf_seal_sha256s=expected_update_leaf_seal_sha256s,
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            stage_manifest_capability=stage_manifest,
            prepare_campaign=prepare_campaign,
            reveal_campaign=reveal_campaign,
            selection_barrier=selection_barrier,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
            expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=attestation.phase_seal_sha256,
        )
        relocate_sealed_phase_noreplace(
            source,
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        _observe_phase_marker_sha256(
            final,
            expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        final_seal = verify_phase(
            final,
            expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
            expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        if final_seal.read_payload_bytes("update-index.jsonl") != expected_index:
            raise ValueError("published update-global index changed during relocation")
        campaign = verify_update_campaign_barrier(
            final_seal,
            state_attestations=state_attestations,
            component_attestations=component_attestations,
            projection_attestations=projection_attestations,
            expected_update_leaf_seal_sha256s=expected_update_leaf_seal_sha256s,
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            stage_manifest_capability=stage_manifest,
            prepare_campaign=prepare_campaign,
            reveal_campaign=reveal_campaign,
            selection_barrier=selection_barrier,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
            expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=attestation.phase_seal_sha256,
        )
        _release_successful_outbox(result)
        return campaign
    except BaseException as error:
        raise RuntimeError(
            f"update-global result was not accepted; outbox retained at {result.outbox}"
        ) from error


def _require_exact_directory_names(
    root: Path,
    expected_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    if tuple(sorted(entry.name for entry in os.scandir(root))) != tuple(sorted(expected_names)):
        raise RuntimeError(f"{label} differs from the frozen update graph")


def _observe_update_leaf_snapshot(
    *,
    tracks_root: Path,
    rotations_root: Path,
    state_authorities: tuple[str, ...],
    component_authorities: tuple[str, ...],
    view_authorities: tuple[str, ...],
    evidence_authorities: tuple[str, ...],
) -> tuple[str, ...]:
    """Observe all final UPDATE markers in the one frozen barrier order."""

    runs = ordered_policy_runs()
    rotations = ordered_rotations()
    if (
        type(state_authorities) is not tuple
        or len(state_authorities) != len(runs)
        or type(component_authorities) is not tuple
        or len(component_authorities) != len(rotations)
        or type(view_authorities) is not tuple
        or len(view_authorities) != len(runs)
        or type(evidence_authorities) is not tuple
        or len(evidence_authorities) != len(runs)
    ):
        raise ValueError("update marker snapshot authority census changed")
    return (
        *(
            _observe_phase_marker_sha256(
                tracks_root / run.track_id / "state",
                expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
                expected_seal_sha256=expected,
            )
            for run, expected in zip(runs, state_authorities, strict=True)
        ),
        *(
            _observe_phase_marker_sha256(
                rotations_root / spec.rotation_id / "outer-components",
                expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
                expected_seal_sha256=expected,
            )
            for spec, expected in zip(rotations, component_authorities, strict=True)
        ),
        *(
            _observe_phase_marker_sha256(
                tracks_root / run.track_id / "outer-view",
                expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
                expected_seal_sha256=expected,
            )
            for run, expected in zip(runs, view_authorities, strict=True)
        ),
        *(
            _observe_phase_marker_sha256(
                tracks_root / run.track_id / "outer-evidence",
                expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
                expected_seal_sha256=expected,
            )
            for run, expected in zip(runs, evidence_authorities, strict=True)
        ),
    )


def supervise_update_campaign(
    *,
    stage_root: str | Path,
    expected_source_anchors: Mapping[str, object],
    run_root: str | Path,
    worker_scratch_root: str | Path,
    expected_protocol_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> UpdateSupervisionResult:
    """Run the frozen 220-state/20-component/220-projection UPDATE DAG."""

    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("update campaign requires an exact publication identity")
    if type(timeout_seconds) is not float or not 0.0 < timeout_seconds <= 24 * 60 * 60:
        raise ValueError("update campaign timeout must be a bounded positive float")
    external_digests = (
        expected_protocol_seal_sha256,
        expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256,
    )
    for index, value in enumerate(external_digests):
        _require_sha256(value, label=f"update campaign external authority {index}")
    if not isinstance(expected_source_anchors, Mapping):
        raise TypeError("expected source anchors must be a mapping")

    final_root = _trusted_directory(
        run_root,
        label="sequential-v2 run root",
        require_empty=False,
        exact_mode=0o700,
    )
    _require_exact_directory_names(
        final_root,
        ("prepare", "protocol", "reveal", "select"),
        label="update input run-root inventory",
    )
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    stage_path = _trusted_directory(
        stage_root,
        label="trusted stage root",
        require_empty=False,
        exact_mode=0o555,
    )
    roots = (stage_path, final_root, scratch)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ValueError("stage, final run, and worker scratch roots must be disjoint trees")

    # Exhaust every physical upstream-global and external digest authority
    # before creating the update namespace or releasing any worker request.
    stage = verify_stage_manifest(
        stage_path,
        expected_source_anchors=expected_source_anchors,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    stage_manifest = authenticate_stage_manifest_for_controller(
        stage,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    verify_stage_manifest_capability(
        stage_manifest.seal,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    protocol_seal = verify_phase(
        final_root / "protocol",
        expected_artifact=PROTOCOL_ARTIFACT,
        expected_payload_paths=PROTOCOL_PAYLOAD_PATHS,
        expected_predecessor_seals={},
        expected_seal_sha256=expected_protocol_seal_sha256,
    )
    protocol = verify_protocol_capability(
        protocol_seal,
        publication_identity=publication_identity,
    )
    if protocol.seal.seal_sha256 != expected_protocol_seal_sha256:
        raise ValueError("physical protocol differs from its external authority")
    prepare_seal = verify_phase(
        final_root / "prepare" / "global",
        expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    prepare_campaign = verify_prepare_campaign_capability(
        PrepareCampaignCapability(prepare_seal, publication_identity),
        publication_identity=publication_identity,
        expected_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
    )
    selection_barrier = verify_phase(
        final_root / "select" / "global",
        expected_artifact=CAMPAIGN_BARRIER_ARTIFACT,
        expected_payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
        expected_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    reveal_seal = verify_phase(
        final_root / "reveal" / "global",
        expected_artifact=REVEAL_CAMPAIGN_ARTIFACT,
        expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_reveal_campaign_seal_sha256,
    )
    reveal_campaign = verify_reveal_campaign_capability(
        RevealCampaignCapability(reveal_seal, publication_identity),
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest_capability=stage_manifest,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
    )

    update_root = _create_private_directory(final_root, "update")
    tracks_root = _create_private_directory(update_root, "tracks")
    rotations_root = _create_private_directory(update_root, "rotations")

    state_by_run: dict[PolicyRunSpec, UpdateStateAttestation] = {}
    state_seal_by_run: dict[PolicyRunSpec, PhaseSeal] = {}
    observed_state_by_run: dict[PolicyRunSpec, str] = {}
    component_by_rotation: dict[RotationSpec, OuterComponentAttestation] = {}
    observed_component_by_rotation: dict[RotationSpec, str] = {}
    projection_by_run: dict[PolicyRunSpec, OuterProjectionAttestation] = {}
    observed_view_by_run: dict[PolicyRunSpec, str] = {}
    observed_evidence_by_run: dict[PolicyRunSpec, str] = {}

    for spec in ordered_rotations():
        rotation_directory = _create_private_directory(rotations_root, spec.rotation_id)
        runs = policy_runs_for_rotation(spec)
        if len(runs) != 11:
            raise AssertionError("frozen update rotation must contain eleven policy runs")

        expected_base_update = prepare_campaign.leaf_seal_sha256(
            spec=spec,
            role=BASE_UPDATE_ROLE,
        )
        base_update_seal = verify_phase(
            final_root / "prepare" / "rotations" / spec.rotation_id / "base-update",
            expected_artifact=BASE_UPDATE_ARTIFACT,
            expected_payload_paths=BASE_UPDATE_PAYLOAD_PATHS,
            expected_seal_sha256=expected_base_update,
        )
        for run in runs:
            track_directory = _create_private_directory(tracks_root, run.track_id)
            reveal_row = reveal_campaign.index_row(run=run)
            selected_reveal_seal = verify_phase(
                final_root / pool_reveal_relative_path(run),
                expected_artifact=POOL_REVEAL_ARTIFACT,
                expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
                expected_seal_sha256=reveal_row.leaf_seal_sha256,
            )
            attestation = launch_update_state_worker(
                final_root / update_state_relative_path(run),
                run=run,
                publication_identity=publication_identity,
                protocol_capability=protocol,
                prepare_campaign=prepare_campaign,
                base_update_seal=base_update_seal,
                reveal_campaign=reveal_campaign,
                reveal_seal=selected_reveal_seal,
                stage_manifest=stage_manifest,
                selection_barrier=selection_barrier,
                expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
                expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
                expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
                expected_reveal_campaign_seal_sha256=(expected_reveal_campaign_seal_sha256),
                worker_scratch_root=scratch,
                timeout_seconds=timeout_seconds,
            )
            observed = _observe_phase_marker_sha256(
                track_directory / "state",
                expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
                expected_seal_sha256=attestation.state_leaf_seal_sha256,
            )
            state_seal = verify_phase(
                track_directory / "state",
                expected_artifact=UPDATE_STATE_ARTIFACT,
                expected_payload_paths=UPDATE_STATE_PAYLOAD_PATHS,
                expected_seal_sha256=observed,
            )
            state_by_run[run] = attestation
            state_seal_by_run[run] = state_seal
            observed_state_by_run[run] = observed
            del selected_reveal_seal

        # Component workers receive only payload-free state attestations; the
        # label-bearing base leaf is dropped before metadata is captured.
        del base_update_seal
        metadata_entry = stage_manifest.leaf(spec=spec, role=OUTER_METADATA_ROLE)
        outer_metadata_capsule = authenticate_stage_leaf_for_controller(
            stage_path / metadata_entry.relative_path,
            expected_leaf=metadata_entry,
            expected_source_anchors=expected_source_anchors,
        )
        rotation_states = tuple(state_by_run[run] for run in runs)
        rotation_state_digests = tuple(observed_state_by_run[run] for run in runs)
        component_attestation = launch_update_component_worker(
            final_root / outer_component_relative_path(spec),
            spec=spec,
            state_attestations=rotation_states,
            expected_state_leaf_seal_sha256s=rotation_state_digests,
            publication_identity=publication_identity,
            protocol_capability=protocol,
            stage_manifest=stage_manifest,
            outer_metadata_capsule=outer_metadata_capsule,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            worker_scratch_root=scratch,
            timeout_seconds=timeout_seconds,
        )
        observed_component = _observe_phase_marker_sha256(
            rotation_directory / "outer-components",
            expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            expected_seal_sha256=component_attestation.component_leaf_seal_sha256,
        )
        component_seal = verify_phase(
            rotation_directory / "outer-components",
            expected_artifact=OUTER_COMPONENTS_ARTIFACT,
            expected_payload_paths=OUTER_COMPONENT_PAYLOAD_PATHS,
            expected_seal_sha256=observed_component,
        )
        component_by_rotation[spec] = component_attestation
        observed_component_by_rotation[spec] = observed_component

        for run in runs:
            state_attestation = state_by_run[run]
            projection_attestation = launch_update_projection_worker(
                final_root / outer_view_relative_path(run),
                final_root / outer_evidence_relative_path(run),
                run=run,
                state_seal=state_seal_by_run[run],
                state_attestation=state_attestation,
                component_seal=component_seal,
                component_attestation=component_attestation,
                publication_identity=publication_identity,
                protocol_capability=protocol,
                prepare_campaign=prepare_campaign,
                reveal_campaign=reveal_campaign,
                stage_manifest=stage_manifest,
                selection_barrier=selection_barrier,
                outer_metadata_capsule=outer_metadata_capsule,
                expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
                expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
                expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
                expected_reveal_campaign_seal_sha256=(expected_reveal_campaign_seal_sha256),
                expected_state_leaf_seal_sha256=observed_state_by_run[run],
                expected_component_leaf_seal_sha256=observed_component,
                worker_scratch_root=scratch,
                timeout_seconds=timeout_seconds,
            )
            observed_view = _observe_phase_marker_sha256(
                tracks_root / run.track_id / "outer-view",
                expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
                expected_seal_sha256=projection_attestation.outer_view_leaf_seal_sha256,
            )
            observed_evidence = _observe_phase_marker_sha256(
                tracks_root / run.track_id / "outer-evidence",
                expected_payload_paths=OUTER_EVIDENCE_PAYLOAD_PATHS,
                expected_seal_sha256=(projection_attestation.outer_evidence_leaf_seal_sha256),
            )
            projection_by_run[run] = projection_attestation
            observed_view_by_run[run] = observed_view
            observed_evidence_by_run[run] = observed_evidence
        del outer_metadata_capsule

    ordered_runs = ordered_policy_runs()
    ordered_specs = ordered_rotations()
    if (
        tuple(state_by_run) != ordered_runs
        or tuple(component_by_rotation) != ordered_specs
        or tuple(projection_by_run) != ordered_runs
    ):
        raise RuntimeError("update execution order differs from the frozen campaign")
    state_attestations = tuple(state_by_run[run] for run in ordered_runs)
    component_attestations = tuple(component_by_rotation[spec] for spec in ordered_specs)
    projection_attestations = tuple(projection_by_run[run] for run in ordered_runs)
    expected_track_ids = tuple(run.track_id for run in ordered_runs)
    expected_rotation_ids = tuple(spec.rotation_id for spec in ordered_specs)
    _require_exact_directory_names(
        tracks_root,
        expected_track_ids,
        label="update track inventory",
    )
    _require_exact_directory_names(
        rotations_root,
        expected_rotation_ids,
        label="update rotation inventory",
    )
    for run in ordered_runs:
        _require_exact_directory_names(
            tracks_root / run.track_id,
            ("state", "outer-view", "outer-evidence"),
            label=f"update track {run.track_id!r} inventory",
        )
    for spec in ordered_specs:
        _require_exact_directory_names(
            rotations_root / spec.rotation_id,
            ("outer-components",),
            label=f"update rotation {spec.rotation_id!r} inventory",
        )
    _require_exact_directory_names(
        update_root,
        ("tracks", "rotations"),
        label="pre-barrier update root inventory",
    )

    # Re-observe every final checksum marker as one uninterrupted canonical
    # snapshot immediately before constructing the barrier request.  Earlier
    # observations establish temporal release; only this four-vector snapshot
    # is the controller authority supplied to update/global.
    observed_update_leaf_seal_sha256s = _observe_update_leaf_snapshot(
        tracks_root=tracks_root,
        rotations_root=rotations_root,
        state_authorities=tuple(observed_state_by_run[run] for run in ordered_runs),
        component_authorities=tuple(observed_component_by_rotation[spec] for spec in ordered_specs),
        view_authorities=tuple(observed_view_by_run[run] for run in ordered_runs),
        evidence_authorities=tuple(observed_evidence_by_run[run] for run in ordered_runs),
    )
    if len(observed_update_leaf_seal_sha256s) != EXPECTED_UPDATE_LEAVES:
        raise AssertionError("frozen update campaign must contain exactly 680 leaves")

    update_campaign = launch_update_barrier_worker(
        update_root / "global",
        state_attestations=state_attestations,
        component_attestations=component_attestations,
        projection_attestations=projection_attestations,
        expected_update_leaf_seal_sha256s=observed_update_leaf_seal_sha256s,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest=stage_manifest,
        prepare_campaign=prepare_campaign,
        reveal_campaign=reveal_campaign,
        selection_barrier=selection_barrier,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    _observe_phase_marker_sha256(
        update_root / "global",
        expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=update_campaign.seal.seal_sha256,
    )
    state_stop = len(ordered_runs)
    component_stop = state_stop + len(ordered_specs)
    view_stop = component_stop + len(ordered_runs)
    post_barrier_leaf_seal_sha256s = _observe_update_leaf_snapshot(
        tracks_root=tracks_root,
        rotations_root=rotations_root,
        state_authorities=observed_update_leaf_seal_sha256s[:state_stop],
        component_authorities=observed_update_leaf_seal_sha256s[state_stop:component_stop],
        view_authorities=observed_update_leaf_seal_sha256s[component_stop:view_stop],
        evidence_authorities=observed_update_leaf_seal_sha256s[view_stop:],
    )
    if post_barrier_leaf_seal_sha256s != observed_update_leaf_seal_sha256s:
        raise RuntimeError("update leaves changed across update-global publication")
    _require_exact_directory_names(
        tracks_root,
        expected_track_ids,
        label="post-barrier update track inventory",
    )
    _require_exact_directory_names(
        rotations_root,
        expected_rotation_ids,
        label="post-barrier update rotation inventory",
    )
    for run in ordered_runs:
        _require_exact_directory_names(
            tracks_root / run.track_id,
            ("state", "outer-view", "outer-evidence"),
            label=f"post-barrier update track {run.track_id!r} inventory",
        )
    for spec in ordered_specs:
        _require_exact_directory_names(
            rotations_root / spec.rotation_id,
            ("outer-components",),
            label=f"post-barrier update rotation {spec.rotation_id!r} inventory",
        )
    _require_exact_directory_names(
        update_root,
        ("global", "tracks", "rotations"),
        label="completed update root inventory",
    )
    return UpdateSupervisionResult(
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest=stage_manifest,
        prepare_campaign=prepare_campaign,
        selection_barrier=selection_barrier,
        reveal_campaign=reveal_campaign,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        state_attestations=state_attestations,
        component_attestations=component_attestations,
        projection_attestations=projection_attestations,
        observed_update_leaf_seal_sha256s=observed_update_leaf_seal_sha256s,
        update_campaign=update_campaign,
        worker_process_count=_UPDATE_WORKER_PROCESS_COUNT,
    )


__all__ = [
    "UPDATE_SUPERVISION_ARTIFACT",
    "UpdateSupervisionResult",
    "launch_update_barrier_worker",
    "launch_update_component_worker",
    "launch_update_projection_worker",
    "launch_update_state_worker",
    "supervise_update_campaign",
]

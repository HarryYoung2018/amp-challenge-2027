"""Fresh-executable supervision for sequential-v2 OUTER-SELECT.

Each of the 220 track workers receives one exact, label-free outer-view
capability.  The barrier receives only payload-free attestations and an
independently observed ordered marker snapshot.  No track worker learns a
source path or a sibling view, and the barrier never receives a leaf/view
``PhaseSeal``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_outer_select import (
    OUTER_SELECTION_CAMPAIGN_ARTIFACT,
    OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
    OUTER_SELECTION_PAYLOAD_PATHS,
    OuterSelectionAttestation,
    OuterSelectionCampaignCapability,
    _snapshot_update_outer_view_rows,
    outer_selection_attestation_from_bytes,
    outer_selection_relative_path,
    verify_outer_selection_campaign_barrier,
)
from amp_challenge.evaluation.sequential_v2_outer_select_wire import (
    OUTER_SELECT_BARRIER_WORKER_ROLE,
    OUTER_SELECT_TRACK_WORKER_ROLE,
    OuterSelectBarrierWorkerRequest,
    OuterSelectTrackWorkerRequest,
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
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_jsonl_bytes,
    relocate_sealed_phase_noreplace,
    verify_phase,
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
    OUTER_VIEW_ARTIFACT,
    OUTER_VIEW_PAYLOAD_PATHS,
    UPDATE_CAMPAIGN_ARTIFACT,
    UPDATE_CAMPAIGN_PAYLOAD_PATHS,
)
from amp_challenge.evaluation.sequential_v2_update_campaign import (
    UpdateCampaignCapability,
    UpdateOuterViewIndexRow,
    verify_update_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_wire import (
    phase_publication_attestation_from_bytes,
)

SCHEMA_VERSION = 1
OUTER_SELECTION_SUPERVISION_ARTIFACT = "sequential_v2_outer_selection_supervision_v1"
_OUTER_SELECTION_WORKER_PROCESS_COUNT = 221
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


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
        raise ValueError(f"{label} destination has the wrong canonical suffix")
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
        raise ValueError("outer-select worker process identity differs from its launched role")


def _require_output_inventory(result: _FreshWorkerResult, expected_names: tuple[str, ...]) -> None:
    actual = tuple(sorted(entry.name for entry in os.scandir(result.output)))
    if actual != tuple(sorted(expected_names)):
        raise RuntimeError("outer-select worker outbox inventory differs from its exact role")


def _require_exact_directory_names(
    root: Path,
    expected_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    if tuple(sorted(entry.name for entry in os.scandir(root))) != tuple(sorted(expected_names)):
        raise RuntimeError(f"{label} differs from the frozen outer-selection graph")


def _observe_selection_snapshot(
    tracks_root: Path,
    *,
    expected_leaf_seal_sha256s: tuple[str, ...],
) -> tuple[str, ...]:
    runs = ordered_policy_runs()
    if (
        type(expected_leaf_seal_sha256s) is not tuple
        or len(expected_leaf_seal_sha256s) != EXPECTED_POLICY_RUNS
    ):
        raise ValueError("outer-selection marker snapshot requires 220 ordered authorities")
    return tuple(
        _observe_phase_marker_sha256(
            tracks_root / run.track_id,
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=expected,
        )
        for run, expected in zip(runs, expected_leaf_seal_sha256s, strict=True)
    )


@dataclass(frozen=True, slots=True)
class OuterSelectionSupervisionResult:
    """Authenticated outer-select/global release from 221 worker leaders."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    update_campaign: UpdateCampaignCapability
    expected_protocol_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    expected_reveal_campaign_seal_sha256: str
    expected_update_campaign_seal_sha256: str
    selection_attestations: tuple[OuterSelectionAttestation, ...]
    observed_outer_selection_leaf_seal_sha256s: tuple[str, ...]
    outer_selection_campaign: OuterSelectionCampaignCapability
    worker_process_count: int

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("outer-selection supervision requires an exact publication identity")
        external = (
            self.expected_protocol_seal_sha256,
            self.expected_stage_global_seal_sha256,
            self.expected_prepare_campaign_seal_sha256,
            self.expected_reveal_campaign_seal_sha256,
            self.expected_update_campaign_seal_sha256,
        )
        for index, value in enumerate(external):
            _require_sha256(value, label=f"outer-selection external authority {index}")
        if (
            type(self.protocol_capability) is not ProtocolCapability
            or self.protocol_capability.seal.seal_sha256 != self.expected_protocol_seal_sha256
            or type(self.update_campaign) is not UpdateCampaignCapability
            or self.update_campaign.seal.seal_sha256 != self.expected_update_campaign_seal_sha256
            or type(self.outer_selection_campaign) is not OuterSelectionCampaignCapability
        ):
            raise ValueError("outer-selection supervision globals differ from authority")
        if type(self.worker_process_count) is not int or self.worker_process_count != (
            _OUTER_SELECTION_WORKER_PROCESS_COUNT
        ):
            raise ValueError("outer-selection supervision requires exactly 221 worker leaders")
        OuterSelectBarrierWorkerRequest(
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            update_campaign_seal=self.update_campaign.seal,
            selection_attestations=self.selection_attestations,
            expected_outer_selection_leaf_seal_sha256s=(
                self.observed_outer_selection_leaf_seal_sha256s
            ),
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=self.expected_update_campaign_seal_sha256,
        )
        verify_outer_selection_campaign_barrier(
            self.outer_selection_campaign.seal,
            selection_attestations=self.selection_attestations,
            expected_outer_selection_leaf_seal_sha256s=(
                self.observed_outer_selection_leaf_seal_sha256s
            ),
            publication_identity=self.publication_identity,
            protocol_capability=self.protocol_capability,
            update_campaign=self.update_campaign,
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_reveal_campaign_seal_sha256=self.expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=self.expected_update_campaign_seal_sha256,
            expected_outer_selection_campaign_seal_sha256=(
                self.outer_selection_campaign.seal.seal_sha256
            ),
        )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": OUTER_SELECTION_SUPERVISION_ARTIFACT,
            "protocol_seal_sha256": self.expected_protocol_seal_sha256,
            "stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
            "prepare_global_seal_sha256": self.expected_prepare_campaign_seal_sha256,
            "reveal_global_seal_sha256": self.expected_reveal_campaign_seal_sha256,
            "update_global_seal_sha256": self.expected_update_campaign_seal_sha256,
            "outer_selection_global_seal_sha256": (self.outer_selection_campaign.seal.seal_sha256),
            "track_worker_count": len(self.selection_attestations),
            "selection_leaf_count": len(self.observed_outer_selection_leaf_seal_sha256s),
            "barrier_worker_count": 1,
            "worker_process_count": self.worker_process_count,
        }


def launch_outer_select_track_worker(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    outer_view_seal: PhaseSeal,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> OuterSelectionAttestation:
    """Publish one exact track commitment through a fresh worker."""

    row = update_campaign.outer_view_row(run=run)
    return _launch_outer_select_track_worker_with_row(
        destination,
        run=run,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign=update_campaign,
        outer_view_seal=outer_view_seal,
        expected_outer_view_row=row,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        worker_scratch_root=worker_scratch_root,
        timeout_seconds=timeout_seconds,
    )


def _launch_outer_select_track_worker_with_row(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    outer_view_seal: PhaseSeal,
    expected_outer_view_row: UpdateOuterViewIndexRow,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float,
) -> OuterSelectionAttestation:
    """Launch one track using a controller-authenticated campaign-row snapshot."""

    if (
        type(expected_outer_view_row) is not UpdateOuterViewIndexRow
        or expected_outer_view_row.run != run
    ):
        raise ValueError("outer-select snapshot row differs from the requested track")
    parent, final = _fresh_destination(
        destination,
        label="outer-select track",
        expected_name=run.track_id,
    )
    scratch = _launcher_scratch(parent, worker_scratch_root)
    request = OuterSelectTrackWorkerRequest(
        run=run,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign_seal=update_campaign.seal,
        outer_view_seal=outer_view_seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    row = expected_outer_view_row
    if outer_view_seal.seal_sha256 != row.leaf_seal_sha256:
        raise ValueError("outer-select view differs from update-global index")
    result = _launch_fresh_worker(
        OUTER_SELECT_TRACK_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        _require_worker_identity(result, role=OUTER_SELECT_TRACK_WORKER_ROLE)
        attestation = outer_selection_attestation_from_bytes(result.payload)
        if (
            attestation.run != run
            or attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
            or attestation.update_global_seal_sha256 != expected_update_campaign_seal_sha256
            or attestation.outer_view_leaf_seal_sha256 != row.leaf_seal_sha256
            or attestation.outer_view_payload_sha256 != row.payload_sha256
            or attestation.outer_candidate_count != row.candidate_count
            or attestation.outer_candidate_ids_sha256 != row.candidate_ids_sha256
            or attestation.selected_sequence_count != run.expected_outer_selection_count
        ):
            raise ValueError("outer-select attestation differs from launched authority")
        _require_output_inventory(result, ("commitment",))
        source = result.output / "commitment"
        _observe_phase_marker_sha256(
            source,
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.selection_leaf_seal_sha256,
        )
        relocate_sealed_phase_noreplace(
            source,
            final,
            expected_seal_sha256=attestation.selection_leaf_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        _observe_phase_marker_sha256(
            final,
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.selection_leaf_seal_sha256,
        )
        _release_successful_outbox(result)
        return attestation
    except BaseException as error:
        raise RuntimeError(
            f"outer-select track result was not accepted; outbox retained at {result.outbox}"
        ) from error


def _expected_outer_selection_index_bytes(
    attestations: tuple[OuterSelectionAttestation, ...],
) -> bytes:
    return canonical_jsonl_bytes(item.index_document() for item in attestations)


def launch_outer_select_barrier_worker(
    destination: str | Path,
    *,
    selection_attestations: tuple[OuterSelectionAttestation, ...],
    expected_outer_selection_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    update_campaign: UpdateCampaignCapability,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> OuterSelectionCampaignCapability:
    """Publish and reconstruct outer-select/global through one fresh worker."""

    parent, final = _fresh_destination(
        destination,
        label="outer-select global",
        expected_name="global",
    )
    scratch = _launcher_scratch(parent, worker_scratch_root)
    request = OuterSelectBarrierWorkerRequest(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        update_campaign_seal=update_campaign.seal,
        selection_attestations=selection_attestations,
        expected_outer_selection_leaf_seal_sha256s=(expected_outer_selection_leaf_seal_sha256s),
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )
    expected_index = _expected_outer_selection_index_bytes(selection_attestations)
    result = _launch_fresh_worker(
        OUTER_SELECT_BARRIER_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        _require_worker_identity(result, role=OUTER_SELECT_BARRIER_WORKER_ROLE)
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=OUTER_SELECT_BARRIER_WORKER_ROLE,
            phase_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
            payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
        )
        _require_output_inventory(result, ("global",))
        source = result.output / "global"
        _observe_phase_marker_sha256(
            source,
            expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        candidate_seal = verify_phase(
            source,
            expected_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
            expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        index_path = OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS[0]
        if candidate_seal.read_payload_bytes(index_path) != expected_index:
            raise ValueError("outer-select global index differs from exact track results")
        verify_outer_selection_campaign_barrier(
            candidate_seal,
            selection_attestations=selection_attestations,
            expected_outer_selection_leaf_seal_sha256s=(expected_outer_selection_leaf_seal_sha256s),
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            update_campaign=update_campaign,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
            expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
            expected_outer_selection_campaign_seal_sha256=(attestation.phase_seal_sha256),
        )
        relocate_sealed_phase_noreplace(
            source,
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        _observe_phase_marker_sha256(
            final,
            expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        final_seal = verify_phase(
            final,
            expected_artifact=OUTER_SELECTION_CAMPAIGN_ARTIFACT,
            expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        if final_seal.read_payload_bytes(index_path) != expected_index:
            raise ValueError("published outer-select global index changed during relocation")
        campaign = verify_outer_selection_campaign_barrier(
            final_seal,
            selection_attestations=selection_attestations,
            expected_outer_selection_leaf_seal_sha256s=(expected_outer_selection_leaf_seal_sha256s),
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            update_campaign=update_campaign,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
            expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
            expected_outer_selection_campaign_seal_sha256=(attestation.phase_seal_sha256),
        )
        _release_successful_outbox(result)
        return campaign
    except BaseException as error:
        raise RuntimeError(
            f"outer-select global result was not accepted; outbox retained at {result.outbox}"
        ) from error


def supervise_outer_selection_campaign(
    *,
    run_root: str | Path,
    worker_scratch_root: str | Path,
    expected_protocol_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_reveal_campaign_seal_sha256: str,
    expected_update_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> OuterSelectionSupervisionResult:
    """Run all 220 track workers and the outer-selection global barrier."""

    if type(publication_identity) is not SequentialV2PublicationIdentity:
        raise TypeError("outer-selection campaign requires an exact publication identity")
    if type(timeout_seconds) is not float or not 0.0 < timeout_seconds <= 24 * 60 * 60:
        raise ValueError("outer-selection timeout must be a bounded positive float")
    external = (
        expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256,
    )
    for index, value in enumerate(external):
        _require_sha256(value, label=f"outer-selection campaign authority {index}")
    final_root = _trusted_directory(
        run_root,
        label="sequential-v2 run root",
        require_empty=False,
        exact_mode=0o700,
    )
    _require_exact_directory_names(
        final_root,
        ("prepare", "protocol", "reveal", "select", "update"),
        label="outer-selection input run-root inventory",
    )
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    if final_root == scratch or final_root in scratch.parents or scratch in final_root.parents:
        raise ValueError("final run and worker scratch roots must be disjoint trees")

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
    update_seal = verify_phase(
        final_root / "update" / "global",
        expected_artifact=UPDATE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=UPDATE_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_update_campaign_seal_sha256,
    )
    update_campaign = verify_update_campaign_capability(
        UpdateCampaignCapability(update_seal, publication_identity),
        publication_identity=publication_identity,
        protocol_capability=protocol,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
    )

    outer_select_root = _create_private_directory(final_root, "outer-select")
    tracks_root = _create_private_directory(outer_select_root, "tracks")
    selection_by_run: dict[PolicyRunSpec, OuterSelectionAttestation] = {}
    early_marker_by_run: dict[PolicyRunSpec, str] = {}
    runs = ordered_policy_runs()
    update_outer_view_rows = _snapshot_update_outer_view_rows(update_campaign)
    for run, row in zip(runs, update_outer_view_rows, strict=True):
        outer_view_seal = verify_phase(
            final_root / row.relative_path,
            expected_artifact=OUTER_VIEW_ARTIFACT,
            expected_payload_paths=OUTER_VIEW_PAYLOAD_PATHS,
            expected_seal_sha256=row.leaf_seal_sha256,
        )
        attestation = _launch_outer_select_track_worker_with_row(
            final_root / outer_selection_relative_path(run),
            run=run,
            publication_identity=publication_identity,
            protocol_capability=protocol,
            update_campaign=update_campaign,
            outer_view_seal=outer_view_seal,
            expected_outer_view_row=row,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
            expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
            expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
            worker_scratch_root=scratch,
            timeout_seconds=timeout_seconds,
        )
        early_marker = _observe_phase_marker_sha256(
            final_root / outer_selection_relative_path(run),
            expected_payload_paths=OUTER_SELECTION_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.selection_leaf_seal_sha256,
        )
        selection_by_run[run] = attestation
        early_marker_by_run[run] = early_marker
        del outer_view_seal

    if tuple(selection_by_run) != runs:
        raise RuntimeError("outer-selection execution order differs from frozen tracks")
    attestations = tuple(selection_by_run[run] for run in runs)
    expected_track_ids = tuple(run.track_id for run in runs)
    _require_exact_directory_names(
        tracks_root,
        expected_track_ids,
        label="outer-selection track inventory",
    )
    _require_exact_directory_names(
        outer_select_root,
        ("tracks",),
        label="pre-barrier outer-selection root inventory",
    )
    pre_barrier_markers = _observe_selection_snapshot(
        tracks_root,
        expected_leaf_seal_sha256s=tuple(early_marker_by_run[run] for run in runs),
    )
    campaign = launch_outer_select_barrier_worker(
        outer_select_root / "global",
        selection_attestations=attestations,
        expected_outer_selection_leaf_seal_sha256s=pre_barrier_markers,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        update_campaign=update_campaign,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    _observe_phase_marker_sha256(
        outer_select_root / "global",
        expected_payload_paths=OUTER_SELECTION_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=campaign.seal.seal_sha256,
    )
    post_barrier_markers = _observe_selection_snapshot(
        tracks_root,
        expected_leaf_seal_sha256s=pre_barrier_markers,
    )
    if post_barrier_markers != pre_barrier_markers:
        raise RuntimeError("outer-selection leaves changed across global publication")
    _require_exact_directory_names(
        tracks_root,
        expected_track_ids,
        label="post-barrier outer-selection track inventory",
    )
    _require_exact_directory_names(
        outer_select_root,
        ("global", "tracks"),
        label="completed outer-selection root inventory",
    )
    return OuterSelectionSupervisionResult(
        publication_identity=publication_identity,
        protocol_capability=protocol,
        update_campaign=update_campaign,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_reveal_campaign_seal_sha256=expected_reveal_campaign_seal_sha256,
        expected_update_campaign_seal_sha256=expected_update_campaign_seal_sha256,
        selection_attestations=attestations,
        observed_outer_selection_leaf_seal_sha256s=pre_barrier_markers,
        outer_selection_campaign=campaign,
        worker_process_count=_OUTER_SELECTION_WORKER_PROCESS_COUNT,
    )


__all__ = [
    "OUTER_SELECTION_SUPERVISION_ARTIFACT",
    "OuterSelectionSupervisionResult",
    "launch_outer_select_barrier_worker",
    "launch_outer_select_track_worker",
    "supervise_outer_selection_campaign",
]

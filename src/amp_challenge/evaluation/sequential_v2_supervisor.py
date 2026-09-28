"""Fresh-executable orchestration through sequential-v2 finalization.

This is the first executable slice after the trusted stage custodian exits.  A
controller that never decodes Gate-1 data launches, in order:

* one outcome-free protocol custodian;
* twenty fresh prepare workers, one per frozen rotation;
* one outcome-free prepare-global custodian;
* sixty least-authority selector custodians;
* twenty outcome-free rotation assemblers; and
* one outcome-free select-global custodian;
* 220 least-authority reveal workers; and
* one outcome-free reveal-global custodian;
* 220 isolated update-state workers;
* twenty label-free outer-component workers;
* 220 label-free outer-projection workers; and
* one outcome-free update-global custodian;
* 220 one-view outer-selection workers; and
* one payload-free outer-selection barrier custodian; and
* one complete-campaign finalization custodian after that barrier.

Every worker starts in an unrelated private courier/outbox directory.  It gets
no original stage path, final DAG path, inherited input descriptor, standard
output, or standard error.  Its only inherited nonstandard descriptor is an
exclusive framed result pipe.  After exit, the controller strictly parses the
payload-free attestation, independently observes every readable checksum
marker, and publishes the exact sealed files into the final DAG without
replacement.

Twin production and the independent verifier remain outside this supervisor
slice.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import select
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from amp_challenge.evaluation.sequential_v2_commitments import (
    CAMPAIGN_BARRIER_ARTIFACT,
    CAMPAIGN_BARRIER_PAYLOAD_PATHS,
    CEILING_SELECTOR_ARTIFACT,
    POOL_COMMITMENT_ARTIFACT,
    POOL_COMMITMENT_PAYLOAD_PATHS,
    PREDICTION_SELECTOR_ARTIFACT,
    RANDOM_SELECTOR_ARTIFACT,
    ROTATION_INDEX_ARTIFACT,
    ROTATION_INDEX_PAYLOAD_PATHS,
    SELECTOR_PAYLOAD_PATHS,
    CommitmentIndexRow,
    decode_pool_commitment,
    pool_commitment_capability_from_seals,
    pool_commitment_relative_path,
    rotation_commitment_index_relative_path,
    verify_pool_commitment_campaign_barrier_capability,
    verify_pool_commitment_campaign_barrier_for_reveal,
    verify_pool_commitment_phase_capability,
    verify_rotation_commitment_index_capability,
    verify_selector_commitment_phase_capability,
)
from amp_challenge.evaluation.sequential_v2_finalize_wire import FINALIZE_WORKER_ROLE
from amp_challenge.evaluation.sequential_v2_outer_select_wire import (
    OUTER_SELECT_BARRIER_WORKER_ROLE,
    OUTER_SELECT_TRACK_WORKER_ROLE,
)
from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
    CAMPAIGN_PAYLOAD_PATHS,
    PREDICTION_VIEW_ARTIFACT,
    PREDICTION_VIEW_PAYLOAD_PATHS,
    PREDICTION_VIEW_ROLE,
    PREPARE_CAMPAIGN_ARTIFACT,
    PROTOCOL_ARTIFACT,
    PROTOCOL_PAYLOAD_PATHS,
    RANDOM_MINIMAL_VIEW_ARTIFACT,
    RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS,
    RANDOM_MINIMAL_VIEW_ROLE,
    PrepareCampaignCapability,
    PrepareRotationAttestation,
    ProtocolCapability,
    SequentialV2PublicationIdentity,
    prepare_rotation_attestation_from_bytes,
    verify_prepare_campaign_barrier,
    verify_prepare_campaign_capability,
    verify_protocol_capability,
)
from amp_challenge.evaluation.sequential_v2_protocol import (
    CEILING,
    EXPECTED_POLICY_RUNS,
    EXPECTED_ROTATIONS,
    MEAN,
    MEAN_NINE_DIVERSITY_ONE,
    MEAN_NINE_NOVELTY_ONE,
    MIXED,
    NO_QUERY,
    RANDOM,
    PolicyRunSpec,
    RotationSpec,
    ordered_policy_runs,
    ordered_rotations,
    policy_runs_for_rotation,
)
from amp_challenge.evaluation.sequential_v2_reveal import (
    POOL_REVEAL_PAYLOAD_PATHS,
    REVEAL_CAMPAIGN_ARTIFACT,
    REVEAL_CAMPAIGN_PAYLOAD_PATHS,
    RevealCampaignCapability,
    RevealLeafAttestation,
    pool_reveal_relative_path,
    reveal_campaign_capability_from_seal,
    reveal_leaf_attestation_from_bytes,
    verify_reveal_campaign_capability,
)
from amp_challenge.evaluation.sequential_v2_reveal_wire import (
    REVEAL_BARRIER_WORKER_ROLE,
    REVEAL_NO_QUERY_WORKER_ROLE,
    REVEAL_NONEMPTY_WORKER_ROLE,
    RevealBarrierWorkerRequest,
    RevealNonemptyWorkerRequest,
    RevealNoQueryWorkerRequest,
)
from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    relocate_sealed_phase_noreplace,
    verify_phase,
)
from amp_challenge.evaluation.sequential_v2_select import ordered_id_stream_sha256
from amp_challenge.evaluation.sequential_v2_stage import (
    POOL_OUTCOME_ROLE,
    PREPARE_ROLE,
    StageManifestCapability,
    authenticate_stage_leaf_for_controller,
    authenticate_stage_manifest_for_controller,
    verify_stage_manifest,
)
from amp_challenge.evaluation.sequential_v2_update_wire import (
    UPDATE_BARRIER_WORKER_ROLE,
    UPDATE_COMPONENT_WORKER_ROLE,
    UPDATE_PROJECTION_WORKER_ROLE,
    UPDATE_STATE_WORKER_ROLE,
)
from amp_challenge.evaluation.sequential_v2_wire import (
    PREPARE_BARRIER_WORKER_ROLE,
    PREPARE_WORKER_ROLE,
    PROTOCOL_WORKER_ROLE,
    SELECT_BARRIER_WORKER_ROLE,
    SELECT_CEILING_WORKER_ROLE,
    SELECT_PREDICTION_WORKER_ROLE,
    SELECT_RANDOM_WORKER_ROLE,
    SELECT_ROTATION_WORKER_ROLE,
    PhasePublicationAttestation,
    PrepareBarrierWorkerRequest,
    PrepareWorkerRequest,
    ProtocolWorkerRequest,
    SelectBarrierWorkerRequest,
    SelectCeilingWorkerRequest,
    SelectPredictionWorkerRequest,
    SelectRandomWorkerRequest,
    SelectRotationAttestation,
    SelectRotationWorkerRequest,
    assert_wire_document_has_no_source_path_fields,
    phase_publication_attestation_from_bytes,
    select_rotation_attestation_from_bytes,
    strict_canonical_json_object,
)

SCHEMA_VERSION = 1
PREPARE_SUPERVISION_ARTIFACT = "sequential_v2_prepare_supervision_v1"
SELECT_SUPERVISION_ARTIFACT = "sequential_v2_select_supervision_v1"
REVEAL_SUPERVISION_ARTIFACT = "sequential_v2_reveal_supervision_v1"
_WORKER_MODULE = "amp_challenge.evaluation.sequential_v2_worker"
_REQUEST_NAME = "request.json"
_OUTPUT_NAME = "output"
_FRAME_BYTES = 8
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_RESULT_BYTES = 1 * 1024 * 1024
_MAX_SOURCE_ANCHORS_BYTES = 16 * 1024 * 1024
_DEFAULT_WORKER_TIMEOUT_SECONDS = 30 * 60.0
_POLL_SECONDS = 0.2
_TERMINATION_GRACE_SECONDS = 5.0
_WORKER_ROLES = frozenset(
    {
        PROTOCOL_WORKER_ROLE,
        PREPARE_WORKER_ROLE,
        PREPARE_BARRIER_WORKER_ROLE,
        SELECT_PREDICTION_WORKER_ROLE,
        SELECT_RANDOM_WORKER_ROLE,
        SELECT_CEILING_WORKER_ROLE,
        SELECT_ROTATION_WORKER_ROLE,
        SELECT_BARRIER_WORKER_ROLE,
        REVEAL_NO_QUERY_WORKER_ROLE,
        REVEAL_NONEMPTY_WORKER_ROLE,
        REVEAL_BARRIER_WORKER_ROLE,
        UPDATE_STATE_WORKER_ROLE,
        UPDATE_COMPONENT_WORKER_ROLE,
        UPDATE_PROJECTION_WORKER_ROLE,
        UPDATE_BARRIER_WORKER_ROLE,
        OUTER_SELECT_TRACK_WORKER_ROLE,
        OUTER_SELECT_BARRIER_WORKER_ROLE,
        FINALIZE_WORKER_ROLE,
    }
)
_COMMON_FAILURE_CHECKPOINTS = frozenset(
    {
        "descriptor-inventory",
        "private-courier",
        "request-authentication",
        "result-publication",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FAILURE_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_FAILURE_ERROR_CODE = re.compile(r"[a-z0-9][a-z0-9_-]{0,159}\Z")


@dataclass(frozen=True, slots=True)
class _FreshWorkerResult:
    role: str
    process_pid: int
    payload: bytes
    outbox: Path
    output: Path


@dataclass(frozen=True, slots=True)
class PrepareSupervisionResult:
    """Verified prepare-global release from the exact fresh-process campaign."""

    protocol_capability: ProtocolCapability
    rotation_attestations: tuple[PrepareRotationAttestation, ...]
    prepare_campaign: PrepareCampaignCapability
    worker_process_count: int

    def __post_init__(self) -> None:
        if type(self.protocol_capability) is not ProtocolCapability:
            raise TypeError("prepare supervision result requires an exact protocol capability")
        if (
            type(self.rotation_attestations) is not tuple
            or len(self.rotation_attestations) != EXPECTED_ROTATIONS
            or any(
                type(item) is not PrepareRotationAttestation for item in self.rotation_attestations
            )
            or tuple(item.spec for item in self.rotation_attestations) != ordered_rotations()
        ):
            raise ValueError("prepare supervision result has the wrong rotation attestations")
        if type(self.prepare_campaign) is not PrepareCampaignCapability:
            raise TypeError("prepare supervision result requires an exact campaign capability")
        if type(self.worker_process_count) is not int or self.worker_process_count != 22:
            raise ValueError(
                "prepare supervision must contain exactly twenty-two launched worker leaders"
            )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": PREPARE_SUPERVISION_ARTIFACT,
            "protocol_seal_sha256": self.protocol_capability.seal.seal_sha256,
            "prepare_global_seal_sha256": self.prepare_campaign.seal.seal_sha256,
            "rotation_worker_count": len(self.rotation_attestations),
            "worker_process_count": self.worker_process_count,
        }


@dataclass(frozen=True, slots=True)
class SelectSupervisionResult:
    """Verified select-global release from exactly eighty-one fresh workers."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    prepare_campaign: PrepareCampaignCapability
    expected_protocol_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    selector_input_view_seals: tuple[PhaseSeal, ...]
    selector_seals: tuple[PhaseSeal, ...]
    commitment_leaf_seals: tuple[PhaseSeal, ...]
    rotation_index_seals: tuple[PhaseSeal, ...]
    selection_barrier: PhaseSeal
    worker_process_count: int

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("select supervision requires an exact publication identity")
        if type(self.protocol_capability) is not ProtocolCapability:
            raise TypeError("select supervision requires an exact protocol capability")
        if type(self.prepare_campaign) is not PrepareCampaignCapability:
            raise TypeError("select supervision requires an exact prepare capability")
        if self.protocol_capability.seal.seal_sha256 != self.expected_protocol_seal_sha256:
            raise ValueError("select supervision protocol differs from external authority")
        protocol = verify_protocol_capability(
            self.protocol_capability.seal,
            publication_identity=self.publication_identity,
        )
        campaign = verify_prepare_campaign_capability(
            self.prepare_campaign,
            publication_identity=self.publication_identity,
            expected_campaign_seal_sha256=self.expected_prepare_campaign_seal_sha256,
            expected_protocol_seal_sha256=self.expected_protocol_seal_sha256,
        )
        if (
            type(self.selector_input_view_seals) is not tuple
            or len(self.selector_input_view_seals) != EXPECTED_ROTATIONS * 3
            or any(type(item) is not PhaseSeal for item in self.selector_input_view_seals)
        ):
            raise ValueError("select supervision requires sixty ordered selector input views")
        if (
            type(self.selector_seals) is not tuple
            or len(self.selector_seals) != EXPECTED_ROTATIONS * 3
            or any(type(item) is not PhaseSeal for item in self.selector_seals)
            or tuple(item.artifact for item in self.selector_seals)
            != (
                PREDICTION_SELECTOR_ARTIFACT,
                RANDOM_SELECTOR_ARTIFACT,
                CEILING_SELECTOR_ARTIFACT,
            )
            * EXPECTED_ROTATIONS
        ):
            raise ValueError("select supervision requires sixty exact selector seals")
        if (
            type(self.commitment_leaf_seals) is not tuple
            or len(self.commitment_leaf_seals) != EXPECTED_POLICY_RUNS
            or any(type(item) is not PhaseSeal for item in self.commitment_leaf_seals)
            or any(item.artifact != POOL_COMMITMENT_ARTIFACT for item in self.commitment_leaf_seals)
        ):
            raise ValueError("select supervision requires 220 exact commitment seals")
        if (
            type(self.rotation_index_seals) is not tuple
            or len(self.rotation_index_seals) != EXPECTED_ROTATIONS
            or any(type(item) is not PhaseSeal for item in self.rotation_index_seals)
            or any(item.artifact != ROTATION_INDEX_ARTIFACT for item in self.rotation_index_seals)
        ):
            raise ValueError("select supervision requires twenty exact rotation indices")
        if (
            type(self.selection_barrier) is not PhaseSeal
            or self.selection_barrier.artifact != CAMPAIGN_BARRIER_ARTIFACT
        ):
            raise TypeError("select supervision requires an exact selection barrier")
        if type(self.worker_process_count) is not int or self.worker_process_count != 81:
            raise ValueError("select supervision must contain exactly eighty-one worker leaders")

        output_seals = (
            *self.selector_seals,
            *self.commitment_leaf_seals,
            *self.rotation_index_seals,
            self.selection_barrier,
        )
        if len({seal.seal_sha256 for seal in output_seals}) != len(output_seals):
            raise ValueError("select supervision output phase seals must be unique")

        selector_by_rotation: dict[str, dict[str, PhaseSeal]] = {}
        selector_offset = 0
        for spec in ordered_rotations():
            current: dict[str, PhaseSeal] = {}
            for kind in ("prediction", "random", "ceiling"):
                selector = self.selector_seals[selector_offset]
                input_view = self.selector_input_view_seals[selector_offset]
                selector_offset += 1
                verify_selector_commitment_phase_capability(
                    selector,
                    spec=spec,
                    selector_kind=kind,
                    input_view_seal=input_view,
                    protocol_capability=protocol,
                    prepare_campaign=campaign,
                    expected_prepare_campaign_seal_sha256=(
                        self.expected_prepare_campaign_seal_sha256
                    ),
                    publication_identity=self.publication_identity,
                    expected_seal_sha256=selector.seal_sha256,
                )
                current[kind] = selector
            selector_by_rotation[spec.rotation_id] = current

        all_rows: list[CommitmentIndexRow] = []
        leaf_offset = 0
        for spec, rotation_seal in zip(ordered_rotations(), self.rotation_index_seals, strict=True):
            rows = verify_rotation_commitment_index_capability(
                rotation_seal,
                spec=spec,
                protocol_capability=protocol,
                prepare_campaign=campaign,
                expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
                publication_identity=self.publication_identity,
                expected_seal_sha256=rotation_seal.seal_sha256,
            )
            leaves = self.commitment_leaf_seals[leaf_offset : leaf_offset + len(rows)]
            leaf_offset += len(rows)
            selectors = selector_by_rotation[spec.rotation_id]
            for row, leaf in zip(rows, leaves, strict=True):
                selector_kind = row.selector_kind
                selector = None if selector_kind is None else selectors[selector_kind]
                verify_pool_commitment_phase_capability(
                    leaf,
                    run=row.run,
                    protocol_capability=protocol,
                    prepare_campaign=campaign,
                    expected_prepare_campaign_seal_sha256=(
                        self.expected_prepare_campaign_seal_sha256
                    ),
                    selector_output_seal=selector,
                    expected_selector_output_seal_sha256=(
                        None if selector is None else selector.seal_sha256
                    ),
                    publication_identity=self.publication_identity,
                    expected_seal_sha256=leaf.seal_sha256,
                )
            _require_commitment_rows_bind_leaves(
                rows,
                leaves,
                protocol_seal_sha256=self.expected_protocol_seal_sha256,
                prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            )
            if any(
                row.selector_output_seal_sha256
                != (None if row.selector_kind is None else selectors[row.selector_kind].seal_sha256)
                for row in rows
            ):
                raise ValueError("select supervision rotation differs from its selector outputs")
            all_rows.extend(rows)
        if leaf_offset != EXPECTED_POLICY_RUNS:
            raise AssertionError("select supervision leaf partition changed")

        barrier_rows = verify_pool_commitment_campaign_barrier_capability(
            self.selection_barrier,
            protocol_capability=protocol,
            prepare_campaign=campaign,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.selection_barrier.seal_sha256,
        )
        if barrier_rows != tuple(all_rows):
            raise ValueError("select supervision barrier differs from its exact rotation rows")
        expected_predecessors = _select_global_predecessors(
            protocol_seal_sha256=self.expected_protocol_seal_sha256,
            prepare_campaign_seal_sha256=self.expected_prepare_campaign_seal_sha256,
            rotation_index_seal_sha256s=tuple(
                seal.seal_sha256 for seal in self.rotation_index_seals
            ),
            commitment_leaf_seals=self.commitment_leaf_seals,
        )
        if dict(self.selection_barrier.predecessor_seals) != expected_predecessors:
            raise ValueError("select supervision barrier differs from its exact output graph")

    def document(self) -> dict[str, object]:
        predecessors = dict(self.selection_barrier.predecessor_seals)
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": SELECT_SUPERVISION_ARTIFACT,
            "protocol_seal_sha256": predecessors["protocol/SHA256SUMS"],
            "prepare_global_seal_sha256": predecessors["prepare/global/SHA256SUMS"],
            "select_global_seal_sha256": self.selection_barrier.seal_sha256,
            "selector_worker_count": len(self.selector_seals),
            "rotation_worker_count": len(self.rotation_index_seals),
            "commitment_leaf_count": len(self.commitment_leaf_seals),
            "worker_process_count": self.worker_process_count,
        }


@dataclass(frozen=True, slots=True)
class RevealSupervisionResult:
    """Verified reveal-global release from exactly 221 fresh workers."""

    publication_identity: SequentialV2PublicationIdentity
    protocol_capability: ProtocolCapability
    stage_manifest: StageManifestCapability
    selection_barrier: PhaseSeal
    expected_protocol_seal_sha256: str
    expected_prepare_campaign_seal_sha256: str
    expected_stage_global_seal_sha256: str
    expected_selection_barrier_seal_sha256: str
    reveal_attestations: tuple[RevealLeafAttestation, ...]
    observed_reveal_leaf_seal_sha256s: tuple[str, ...]
    reveal_campaign: RevealCampaignCapability
    worker_process_count: int

    def __post_init__(self) -> None:
        if type(self.publication_identity) is not SequentialV2PublicationIdentity:
            raise TypeError("reveal supervision requires an exact publication identity")
        if type(self.protocol_capability) is not ProtocolCapability:
            raise TypeError("reveal supervision requires an exact protocol capability")
        if type(self.stage_manifest) is not StageManifestCapability:
            raise TypeError("reveal supervision requires an exact stage capability")
        if type(self.selection_barrier) is not PhaseSeal:
            raise TypeError("reveal supervision requires an exact select-global phase")
        expected_values = (
            self.expected_protocol_seal_sha256,
            self.expected_prepare_campaign_seal_sha256,
            self.expected_stage_global_seal_sha256,
            self.expected_selection_barrier_seal_sha256,
        )
        if any(
            type(value) is not str or _SHA256.fullmatch(value) is None for value in expected_values
        ):
            raise ValueError("reveal supervision external authorities must be lowercase SHA-256s")
        if (
            self.protocol_capability.seal.seal_sha256 != self.expected_protocol_seal_sha256
            or self.stage_manifest.seal.seal_sha256 != self.expected_stage_global_seal_sha256
            or self.selection_barrier.seal_sha256 != self.expected_selection_barrier_seal_sha256
        ):
            raise ValueError("reveal supervision globals differ from external authorities")
        protocol = verify_protocol_capability(
            self.protocol_capability.seal,
            publication_identity=self.publication_identity,
        )
        commitments = verify_pool_commitment_campaign_barrier_for_reveal(
            self.selection_barrier,
            protocol_capability=protocol,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            publication_identity=self.publication_identity,
            expected_seal_sha256=self.expected_selection_barrier_seal_sha256,
        )
        runs = ordered_policy_runs()
        if (
            type(self.reveal_attestations) is not tuple
            or len(self.reveal_attestations) != EXPECTED_POLICY_RUNS
            or any(type(item) is not RevealLeafAttestation for item in self.reveal_attestations)
            or tuple(item.run for item in self.reveal_attestations) != runs
        ):
            raise ValueError("reveal supervision requires 220 exact ordered attestations")
        if (
            type(self.observed_reveal_leaf_seal_sha256s) is not tuple
            or len(self.observed_reveal_leaf_seal_sha256s) != EXPECTED_POLICY_RUNS
            or any(
                type(value) is not str or _SHA256.fullmatch(value) is None
                for value in self.observed_reveal_leaf_seal_sha256s
            )
            or len(set(self.observed_reveal_leaf_seal_sha256s)) != EXPECTED_POLICY_RUNS
        ):
            raise ValueError("reveal supervision requires 220 distinct observed leaf seals")
        if tuple(item.reveal_leaf_seal_sha256 for item in self.reveal_attestations) != (
            self.observed_reveal_leaf_seal_sha256s
        ):
            raise ValueError("reveal supervision attestations differ from physical leaf markers")
        for item, commitment, observed in zip(
            self.reveal_attestations,
            commitments,
            self.observed_reveal_leaf_seal_sha256s,
            strict=True,
        ):
            if (
                item.publication_identity != self.publication_identity
                or item.protocol_seal_sha256 != self.expected_protocol_seal_sha256
                or item.select_global_seal_sha256 != self.expected_selection_barrier_seal_sha256
                or item.commitment_leaf_seal_sha256 != commitment.leaf_seal_sha256
                or item.reveal_leaf_seal_sha256 != observed
                or dict(item.payload_sha256)["commitment.json"]
                != commitment.commitment_payload_sha256
                or item.selected_sequence_count != commitment.selected_sequence_count
                or item.selected_sequence_ids_sha256 != commitment.selected_sequence_ids_sha256
            ):
                raise ValueError("reveal supervision leaf differs from global authority")
            if item.run.policy == NO_QUERY:
                if (
                    item.stage_global_seal_sha256 is not None
                    or item.pool_outcome_vault_seal_sha256 is not None
                ):
                    raise ValueError("reveal supervision no-query leaf binds stage authority")
            else:
                expected_vault = self.stage_manifest.leaf(
                    spec=item.run.rotation,
                    role=POOL_OUTCOME_ROLE,
                )
                if (
                    item.stage_global_seal_sha256 != self.expected_stage_global_seal_sha256
                    or item.pool_outcome_vault_seal_sha256 != expected_vault.leaf_seal_sha256
                ):
                    raise ValueError("reveal supervision leaf differs from pool-vault authority")
        if type(self.reveal_campaign) is not RevealCampaignCapability:
            raise TypeError("reveal supervision requires an exact campaign capability")
        campaign = verify_reveal_campaign_capability(
            self.reveal_campaign,
            publication_identity=self.publication_identity,
            protocol_capability=protocol,
            stage_manifest_capability=self.stage_manifest,
            selection_barrier=self.selection_barrier,
            expected_prepare_campaign_seal_sha256=(self.expected_prepare_campaign_seal_sha256),
            expected_stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(self.expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=(self.reveal_campaign.seal.seal_sha256),
        )
        expected_index = canonical_jsonl_bytes(
            item.index_document() for item in self.reveal_attestations
        )
        if campaign.seal.read_payload_bytes("reveal-index.jsonl") != expected_index:
            raise ValueError("reveal supervision campaign index differs from exact leaf results")
        expected_predecessors = _reveal_global_predecessors(
            protocol_seal_sha256=self.expected_protocol_seal_sha256,
            stage_global_seal_sha256=self.expected_stage_global_seal_sha256,
            selection_barrier_seal_sha256=self.expected_selection_barrier_seal_sha256,
            reveal_leaf_seal_sha256s=self.observed_reveal_leaf_seal_sha256s,
        )
        if dict(campaign.seal.predecessor_seals) != expected_predecessors:
            raise ValueError("reveal supervision campaign differs from its exact output graph")
        if type(self.worker_process_count) is not int or self.worker_process_count != 221:
            raise ValueError("reveal supervision must contain exactly 221 worker leaders")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": REVEAL_SUPERVISION_ARTIFACT,
            "protocol_seal_sha256": self.expected_protocol_seal_sha256,
            "prepare_global_seal_sha256": self.expected_prepare_campaign_seal_sha256,
            "stage_global_seal_sha256": self.expected_stage_global_seal_sha256,
            "select_global_seal_sha256": self.expected_selection_barrier_seal_sha256,
            "reveal_global_seal_sha256": self.reveal_campaign.seal.seal_sha256,
            "leaf_worker_count": len(self.reveal_attestations),
            "barrier_worker_count": 1,
            "worker_process_count": self.worker_process_count,
        }


def _trusted_directory(
    value: str | Path,
    *,
    label: str,
    require_empty: bool,
    exact_mode: int | None = None,
) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    candidate = path
    while True:
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not traverse a symbolic link")
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    metadata = os.lstat(path)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or mode & 0o022
        or (exact_mode is not None and mode != exact_mode)
    ):
        raise ValueError(
            f"{label} must be a current-user-owned real directory without group/world write"
        )
    if require_empty and tuple(os.scandir(path)):
        raise ValueError(f"{label} must be empty and never reused")
    return path.resolve(strict=True)


def _observe_phase_marker_sha256(
    phase: str | Path,
    *,
    expected_payload_paths: tuple[str, ...],
    expected_seal_sha256: str,
) -> str:
    """Observe one final checksum marker without opening any phase payload."""

    if (
        type(expected_payload_paths) is not tuple
        or not expected_payload_paths
        or any(type(path) is not str or not path or "/" in path for path in expected_payload_paths)
        or len(set(expected_payload_paths)) != len(expected_payload_paths)
    ):
        raise ValueError("marker observation requires exact flat payload paths")
    if type(expected_seal_sha256) is not str or _SHA256.fullmatch(expected_seal_sha256) is None:
        raise ValueError("marker observation requires a lowercase SHA-256 authority")
    root = _trusted_directory(
        phase,
        label="published phase for marker observation",
        require_empty=False,
        exact_mode=0o555,
    )
    expected_names = tuple(sorted((*expected_payload_paths, "receipt.json", "SHA256SUMS")))
    named_root_before = os.lstat(root)
    directory_descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    marker_descriptor = -1
    try:
        directory_before = os.fstat(directory_descriptor)
        if (
            (named_root_before.st_dev, named_root_before.st_ino)
            != (directory_before.st_dev, directory_before.st_ino)
            or not stat.S_ISDIR(directory_before.st_mode)
            or directory_before.st_uid != os.geteuid()
            or stat.S_IMODE(directory_before.st_mode) != 0o555
        ):
            raise RuntimeError("published phase changed before marker observation")

        def inventory() -> dict[str, tuple[int, ...]]:
            names = tuple(sorted(os.listdir(directory_descriptor)))
            if names != expected_names:
                raise ValueError("published phase has the wrong marker-observation inventory")
            fingerprints: dict[str, tuple[int, ...]] = {}
            for name in names:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o444
                ):
                    raise ValueError("published phase contains an unsafe marker-observation entry")
                fingerprints[name] = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            return fingerprints

        inventory_before = inventory()
        marker_descriptor = os.open(
            "SHA256SUMS",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        marker_before = os.fstat(marker_descriptor)
        if (
            not stat.S_ISREG(marker_before.st_mode)
            or marker_before.st_uid != os.geteuid()
            or marker_before.st_nlink != 1
            or stat.S_IMODE(marker_before.st_mode) != 0o444
            or not 0 < marker_before.st_size <= _MAX_SOURCE_ANCHORS_BYTES
        ):
            raise ValueError("published phase checksum marker is unsafe")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(marker_descriptor, 1024 * 1024):
            total += len(chunk)
            if total > _MAX_SOURCE_ANCHORS_BYTES:
                raise ValueError("published phase checksum marker exceeds its byte bound")
            digest.update(chunk)
        marker_after = os.fstat(marker_descriptor)
        named_marker = os.stat(
            "SHA256SUMS",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        inventory_after = inventory()
        directory_after = os.fstat(directory_descriptor)
        named_root_after = os.lstat(root)
        if (
            (
                marker_before.st_dev,
                marker_before.st_ino,
                marker_before.st_size,
                marker_before.st_mtime_ns,
            )
            != (
                marker_after.st_dev,
                marker_after.st_ino,
                marker_after.st_size,
                marker_after.st_mtime_ns,
            )
            or (marker_after.st_dev, marker_after.st_ino)
            != (named_marker.st_dev, named_marker.st_ino)
            or inventory_before != inventory_after
            or (
                directory_before.st_dev,
                directory_before.st_ino,
                directory_before.st_mode,
                directory_before.st_uid,
                directory_before.st_gid,
                directory_before.st_nlink,
                directory_before.st_mtime_ns,
                directory_before.st_ctime_ns,
            )
            != (
                directory_after.st_dev,
                directory_after.st_ino,
                directory_after.st_mode,
                directory_after.st_uid,
                directory_after.st_gid,
                directory_after.st_nlink,
                directory_after.st_mtime_ns,
                directory_after.st_ctime_ns,
            )
            or (directory_after.st_dev, directory_after.st_ino)
            != (named_root_after.st_dev, named_root_after.st_ino)
            or not stat.S_ISDIR(named_root_after.st_mode)
            or named_root_after.st_uid != os.geteuid()
            or stat.S_IMODE(named_root_after.st_mode) != 0o555
        ):
            raise RuntimeError("published phase changed while its marker was observed")
    finally:
        if marker_descriptor >= 0:
            os.close(marker_descriptor)
        os.close(directory_descriptor)
    observed = digest.hexdigest()
    if observed != expected_seal_sha256:
        raise ValueError("published phase marker differs from its external authority")
    return observed


def _create_private_directory(parent: Path, name: str) -> Path:
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError("private directory name is unsafe")
    destination = parent / name
    os.mkdir(destination, 0o700)
    metadata = os.lstat(destination)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RuntimeError("created private directory has an unsafe identity")
    return destination


def _child_environment(outbox: Path) -> dict[str, str]:
    environment = {
        "BLIS_NUM_THREADS": "1",
        "HOME": os.fspath(outbox),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OMP_THREAD_LIMIT": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "42",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TMPDIR": os.fspath(outbox),
        "TZ": "UTC",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    forbidden = sorted(
        name
        for name in environment
        if name.startswith(("AMP_", "GIT_", "SLURM_"))
        or name
        in {
            "CONDA_PREFIX",
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }
    )
    if forbidden:
        raise RuntimeError("fresh worker environment retained forbidden authority")
    return environment


def _require_unrelated_same_filesystem(
    destination_parent: Path,
    worker_scratch_root: Path,
) -> None:
    if (
        destination_parent == worker_scratch_root
        or destination_parent in worker_scratch_root.parents
        or worker_scratch_root in destination_parent.parents
    ):
        raise ValueError("final destination and worker scratch must be disjoint trees")
    if os.lstat(destination_parent).st_dev != os.lstat(worker_scratch_root).st_dev:
        raise ValueError("final destination and worker scratch must share one filesystem")


def _write_request(outbox: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_REQUEST_BYTES:
        raise ValueError("fresh worker request is not bounded exact bytes")
    document = strict_canonical_json_object(payload, label="fresh worker request")
    assert_wire_document_has_no_source_path_fields(document)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(outbox / _REQUEST_NAME, flags, 0o400)
    try:
        os.fchmod(descriptor, 0o400)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("fresh worker request made no write progress")
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError("fresh worker request has an unsafe file identity")
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write exact bytes even when the underlying descriptor short-writes."""

    if type(descriptor) is not int or type(payload) is not bytes:
        raise TypeError("descriptor output requires an exact integer and bytes")
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0 or written > len(remaining):
            raise OSError("descriptor output made invalid write progress")
        remaining = remaining[written:]


def _worker_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_worker_group_exit(process_group: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _worker_group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return True


def _terminate_worker(process: subprocess.Popen[bytes]) -> None:
    process.poll()
    if not _worker_group_exists(process.pid):
        if process.returncode is None:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.returncode is None:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return
    if process.returncode is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    if _wait_for_worker_group_exit(
        process.pid,
        timeout=_TERMINATION_GRACE_SECONDS,
    ):
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.returncode is None:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    if not _wait_for_worker_group_exit(
        process.pid,
        timeout=_TERMINATION_GRACE_SECONDS,
    ):
        raise RuntimeError("fresh worker process group survived SIGKILL")


def _read_framed_result(
    descriptor: int,
    *,
    process: subprocess.Popen[bytes],
    deadline: float,
    expected_role: str,
) -> bytes:
    if expected_role not in _WORKER_ROLES:
        raise ValueError("expected fresh-worker role is invalid")
    os.set_blocking(descriptor, False)
    framed = bytearray()
    expected_total: int | None = None
    eof = False

    def read_available() -> None:
        nonlocal eof, expected_total
        while True:
            try:
                chunk = os.read(descriptor, 65_536)
            except BlockingIOError:
                return
            if not chunk:
                eof = True
                return
            framed.extend(chunk)
            if len(framed) > _FRAME_BYTES + _MAX_RESULT_BYTES:
                _terminate_worker(process)
                raise RuntimeError("fresh worker result exceeded its byte bound")
            if len(framed) >= _FRAME_BYTES and expected_total is None:
                declared = int.from_bytes(framed[:_FRAME_BYTES], "big")
                if not 0 < declared <= _MAX_RESULT_BYTES:
                    _terminate_worker(process)
                    raise RuntimeError("fresh worker result declared an invalid length")
                expected_total = _FRAME_BYTES + declared
            if expected_total is not None and len(framed) > expected_total:
                _terminate_worker(process)
                raise RuntimeError("fresh worker result contains trailing bytes")

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            _terminate_worker(process)
            raise TimeoutError("fresh worker exceeded its deadline")
        if eof:
            try:
                process.wait(timeout=min(_POLL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                continue
            break
        readable, _writable, _exceptional = select.select(
            (descriptor,),
            (),
            (),
            min(_POLL_SECONDS, remaining),
        )
        if readable:
            read_available()
        status = process.poll()
        if status is not None and not eof:
            # Close the race where the leader exits immediately after the
            # readiness sample.  A normal closed writer drains to EOF here;
            # EAGAIN means a descendant or escaped process retained the pipe.
            read_available()
            if not eof:
                _terminate_worker(process)
                raise RuntimeError("fresh worker exited before closing its result channel")
        if eof and status is not None:
            break
    if expected_total is None or len(framed) != expected_total:
        raise RuntimeError("fresh worker result is missing, truncated, or unframed")
    payload = bytes(framed[_FRAME_BYTES:])
    if _worker_group_exists(process.pid):
        _terminate_worker(process)
        raise RuntimeError("fresh worker left a surviving process-group member")
    if process.returncode != 0:
        digest = hashlib.sha256(payload).hexdigest()
        try:
            failure = strict_canonical_json_object(payload, label="fresh worker failure")
        except (TypeError, ValueError):
            raise RuntimeError(
                f"fresh worker exited with status {process.returncode}; "
                f"opaque result sha256={digest}"
            ) from None
        safe_fields = {
            "schema_version",
            "artifact",
            "worker_role",
            "checkpoint",
            "error_type",
            "error_code",
        }
        if (
            set(failure) == safe_fields
            and type(failure["schema_version"]) is int
            and failure["schema_version"] == 1
            and failure["artifact"] == "sequential_v2_fresh_worker_failure_v1"
            and all(
                type(failure[key]) is str
                for key in ("worker_role", "checkpoint", "error_type", "error_code")
            )
            and failure["worker_role"] == expected_role
            and failure["checkpoint"] in {*_COMMON_FAILURE_CHECKPOINTS, expected_role}
            and _FAILURE_ERROR_TYPE.fullmatch(failure["error_type"]) is not None
            and _FAILURE_ERROR_CODE.fullmatch(failure["error_code"]) is not None
        ):
            raise RuntimeError(
                f"fresh worker exited with status {process.returncode} at "
                f"{failure['checkpoint']}; failure sha256={digest}"
            )
        raise RuntimeError(
            f"fresh worker exited with status {process.returncode}; "
            f"unrecognized result sha256={digest}"
        )
    return payload


def _fresh_worker_command(role: str, result_fd: int) -> tuple[str, ...]:
    # Bind code to this supervisor's checkout without giving the courier a
    # PYTHONPATH or inheriting the caller's scientific/environment authority.
    source_root = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
        f"runpy.run_module({_WORKER_MODULE!r}, run_name='__main__', alter_sys=True)"
    )
    return (
        sys.executable,
        "-c",
        bootstrap,
        os.fspath(source_root),
        role,
        "--result-fd",
        str(result_fd),
    )


def _launch_fresh_worker(
    role: str,
    request_payload: bytes,
    *,
    worker_scratch_root: Path,
    timeout_seconds: float,
) -> _FreshWorkerResult:
    if role not in _WORKER_ROLES:
        raise ValueError("fresh worker role is invalid")
    if type(timeout_seconds) is not float or not 0.0 < timeout_seconds <= 24 * 60 * 60:
        raise ValueError("fresh worker timeout must be a bounded positive float")
    outbox = Path(tempfile.mkdtemp(prefix=f".seqv2-{role}.", dir=worker_scratch_root))
    os.chmod(outbox, 0o700)
    output = _create_private_directory(outbox, _OUTPUT_NAME)
    try:
        _write_request(outbox, request_payload)
        read_descriptor = -1
        write_descriptor = -1
        null_input = -1
        null_output = -1
        null_error = -1
        process: subprocess.Popen[bytes] | None = None
        try:
            read_descriptor, write_descriptor = os.pipe()
            null_input = os.open(
                os.devnull,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            null_output = os.open(
                os.devnull,
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            null_error = os.open(
                os.devnull,
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            command = _fresh_worker_command(role, write_descriptor)
            environment = _child_environment(outbox)
            process = subprocess.Popen(
                command,
                stdin=null_input,
                stdout=null_output,
                stderr=null_error,
                cwd=outbox,
                env=environment,
                close_fds=True,
                start_new_session=True,
                pass_fds=(write_descriptor,),
            )
            os.close(null_input)
            null_input = -1
            os.close(null_output)
            null_output = -1
            os.close(null_error)
            null_error = -1
            os.close(write_descriptor)
            write_descriptor = -1
            payload = _read_framed_result(
                read_descriptor,
                process=process,
                deadline=time.monotonic() + timeout_seconds,
                expected_role=role,
            )
        except BaseException:
            if process is not None:
                _terminate_worker(process)
            raise
        finally:
            for descriptor in (
                read_descriptor,
                write_descriptor,
                null_input,
                null_output,
                null_error,
            ):
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
        if os.path.lexists(outbox / _REQUEST_NAME):
            raise RuntimeError("fresh worker did not close and remove its request")
        if tuple(sorted(entry.name for entry in os.scandir(outbox))) != (_OUTPUT_NAME,):
            raise RuntimeError("fresh worker wrote outside its sole output directory")
        return _FreshWorkerResult(
            role=role,
            process_pid=process.pid,
            payload=payload,
            outbox=outbox,
            output=output,
        )
    except BaseException as error:
        # Preserve every failed courier/outbox for forensic inspection.  It is
        # private and source-root-free; a failed final run must be abandoned.
        raise RuntimeError(
            f"{role} fresh worker failed; private outbox retained at {outbox}"
        ) from error


def _release_successful_outbox(result: _FreshWorkerResult) -> None:
    if tuple(os.scandir(result.output)):
        raise RuntimeError("cannot release a fresh worker outbox before every phase moved")
    os.rmdir(result.output)
    if tuple(os.scandir(result.outbox)):
        raise RuntimeError("successful fresh worker outbox retained an unexpected entry")
    os.rmdir(result.outbox)


def _validate_phase_attestation(
    attestation: PhasePublicationAttestation,
    *,
    worker_role: str,
    phase_artifact: str,
    payload_paths: tuple[str, ...],
) -> None:
    if (
        attestation.worker_role != worker_role
        or attestation.phase_artifact != phase_artifact
        or tuple(path for path, _digest in attestation.payload_sha256)
        != tuple(sorted(payload_paths))
    ):
        raise ValueError("fresh worker phase attestation differs from its exact role")


def launch_protocol_worker(
    destination: str | Path,
    *,
    publication_identity: SequentialV2PublicationIdentity,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> ProtocolCapability:
    """Publish and verify protocol through one fresh outcome-free custodian."""

    parent = _trusted_directory(
        Path(destination).parent,
        label="protocol destination parent",
        require_empty=True,
    )
    final = parent / Path(destination).name
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(parent, scratch)
    request = ProtocolWorkerRequest(publication_identity).canonical_bytes()
    result = _launch_fresh_worker(
        PROTOCOL_WORKER_ROLE,
        request,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=PROTOCOL_WORKER_ROLE,
            phase_artifact=PROTOCOL_ARTIFACT,
            payload_paths=PROTOCOL_PAYLOAD_PATHS,
        )
        source = result.output / "protocol"
        if tuple(entry.name for entry in os.scandir(result.output)) != ("protocol",):
            raise RuntimeError("protocol worker outbox inventory is invalid")
        relocate_sealed_phase_noreplace(
            source,
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        seal = verify_phase(
            final,
            expected_artifact=PROTOCOL_ARTIFACT,
            expected_payload_paths=PROTOCOL_PAYLOAD_PATHS,
            expected_predecessor_seals={},
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        capability = verify_protocol_capability(
            seal,
            publication_identity=publication_identity,
        )
        _release_successful_outbox(result)
        return capability
    except BaseException as error:
        raise RuntimeError(
            f"protocol worker result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_prepare_rotation_worker(
    destination: str | Path,
    *,
    spec: RotationSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest: StageManifestCapability,
    expected_stage_global_seal_sha256: str,
    source_prepare_leaf_seal: PhaseSeal,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> PrepareRotationAttestation:
    """Publish one rotation's four leaves through one exclusive fresh worker."""

    final_parent = _trusted_directory(
        destination,
        label="prepare rotation destination",
        require_empty=True,
        exact_mode=0o700,
    )
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    request_value = PrepareWorkerRequest(
        spec=spec,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_global_seal=stage_manifest.seal,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        source_prepare_leaf_seal=source_prepare_leaf_seal,
    )
    result = _launch_fresh_worker(
        PREPARE_WORKER_ROLE,
        request_value.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        attestation = prepare_rotation_attestation_from_bytes(result.payload)
        if (
            attestation.spec != spec
            or attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
            or attestation.stage_global_seal_sha256 != expected_stage_global_seal_sha256
            or attestation.source_prepare_leaf_seal_sha256 != source_prepare_leaf_seal.seal_sha256
        ):
            raise ValueError("prepare worker attestation differs from its launched authority")
        expected_suffixes = tuple(
            PurePosixPath(leaf.relative_path).name for leaf in attestation.leaves
        )
        if tuple(sorted(entry.name for entry in os.scandir(result.output))) != tuple(
            sorted(expected_suffixes)
        ):
            raise RuntimeError("prepare worker outbox does not contain its exact four leaves")
        for leaf in attestation.leaves:
            suffix = PurePosixPath(leaf.relative_path).name
            relocate_sealed_phase_noreplace(
                result.output / suffix,
                final_parent / suffix,
                expected_seal_sha256=leaf.phase_seal_sha256,
                expected_payload_sha256=dict(leaf.payload_sha256),
            )
        _release_successful_outbox(result)
        return attestation
    except BaseException as error:
        raise RuntimeError(
            f"prepare worker result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_prepare_barrier_worker(
    destination: str | Path,
    *,
    attestations: tuple[PrepareRotationAttestation, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> PrepareCampaignCapability:
    """Publish and verify prepare/global through a fresh payload-free custodian."""

    final_parent = _trusted_directory(
        Path(destination).parent,
        label="prepare-global destination parent",
        require_empty=False,
        exact_mode=0o700,
    )
    final = final_parent / Path(destination).name
    if os.path.lexists(final):
        raise FileExistsError("prepare-global destination must be fresh")
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    request = PrepareBarrierWorkerRequest(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        attestations=attestations,
    ).canonical_bytes()
    result = _launch_fresh_worker(
        PREPARE_BARRIER_WORKER_ROLE,
        request,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=PREPARE_BARRIER_WORKER_ROLE,
            phase_artifact=PREPARE_CAMPAIGN_ARTIFACT,
            payload_paths=CAMPAIGN_PAYLOAD_PATHS,
        )
        if tuple(entry.name for entry in os.scandir(result.output)) != ("global",):
            raise RuntimeError("prepare barrier worker outbox inventory is invalid")
        relocate_sealed_phase_noreplace(
            result.output / "global",
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        seal = verify_phase(
            final,
            expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
            expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        campaign = verify_prepare_campaign_barrier(
            seal,
            attestations=attestations,
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
        )
        _release_successful_outbox(result)
        return campaign
    except BaseException as error:
        raise RuntimeError(
            f"prepare barrier result was not accepted; outbox retained at {result.outbox}"
        ) from error


def _selector_kind_for_run(run: PolicyRunSpec) -> str | None:
    if type(run) is not PolicyRunSpec:
        raise TypeError("selector lookup requires an exact policy run")
    if run.policy == NO_QUERY:
        return None
    if run.policy in {
        MEAN,
        MEAN_NINE_DIVERSITY_ONE,
        MEAN_NINE_NOVELTY_ONE,
        MIXED,
    }:
        return "prediction"
    if run.policy == RANDOM:
        return "random"
    if run.policy == CEILING:
        return "ceiling"
    raise AssertionError("frozen policy run has no selector kind")


def _selector_worker_contract(selector_kind: str) -> tuple[str, str]:
    try:
        return {
            "prediction": (
                SELECT_PREDICTION_WORKER_ROLE,
                PREDICTION_SELECTOR_ARTIFACT,
            ),
            "random": (SELECT_RANDOM_WORKER_ROLE, RANDOM_SELECTOR_ARTIFACT),
            "ceiling": (SELECT_CEILING_WORKER_ROLE, CEILING_SELECTOR_ARTIFACT),
        }[selector_kind]
    except (KeyError, TypeError) as error:
        raise ValueError("selector worker kind is invalid") from error


def _require_commitment_rows_bind_leaves(
    rows: Sequence[CommitmentIndexRow],
    leaves: Sequence[PhaseSeal],
    *,
    protocol_seal_sha256: str,
    prepare_campaign_seal_sha256: str,
) -> None:
    """Bind index claims to exact leaf payloads and predecessor capabilities."""

    captured_rows = tuple(rows)
    captured_leaves = tuple(leaves)
    if (
        len(captured_rows) != len(captured_leaves)
        or any(type(row) is not CommitmentIndexRow for row in captured_rows)
        or any(type(leaf) is not PhaseSeal for leaf in captured_leaves)
    ):
        raise ValueError("commitment row and leaf censuses differ")
    for row, leaf in zip(captured_rows, captured_leaves, strict=True):
        commitment = decode_pool_commitment(leaf.read_payload_bytes("commitment.json"))
        payload_digest = dict(leaf.payload_sha256).get("commitment.json")
        if (
            row.run != commitment.run
            or row.leaf_seal_sha256 != leaf.seal_sha256
            or row.commitment_payload_sha256 != payload_digest
            or row.selected_sequence_count != len(commitment.selected_sequence_ids)
            or row.selected_sequence_ids_sha256
            != ordered_id_stream_sha256(commitment.selected_sequence_ids)
            or row.input_view != commitment.input_view
        ):
            raise ValueError("commitment index row differs from its exact leaf semantics")
        expected_predecessors = {
            "protocol/SHA256SUMS": protocol_seal_sha256,
            "prepare/global/SHA256SUMS": prepare_campaign_seal_sha256,
        }
        if row.selector_kind is not None:
            view_role = (
                "prediction-view" if row.selector_kind == "prediction" else "random-minimal-view"
            )
            expected_predecessors[
                f"prepare/rotations/{row.run.rotation.rotation_id}/{view_role}/SHA256SUMS"
            ] = row.input_view.phase_seal_sha256
            expected_predecessors[
                "select/selector-outputs/"
                f"{row.run.rotation.rotation_id}/{row.selector_kind}/SHA256SUMS"
            ] = row.selector_output_seal_sha256
        if dict(leaf.predecessor_seals) != expected_predecessors:
            raise ValueError("commitment leaf differs from its exact index authorities")


def _select_global_predecessors(
    *,
    protocol_seal_sha256: str,
    prepare_campaign_seal_sha256: str,
    rotation_index_seal_sha256s: Sequence[str],
    commitment_leaf_seals: Sequence[PhaseSeal],
) -> dict[str, str]:
    rotations = ordered_rotations()
    runs = ordered_policy_runs()
    rotation_digests = tuple(rotation_index_seal_sha256s)
    leaves = tuple(commitment_leaf_seals)
    if len(rotation_digests) != len(rotations) or len(leaves) != len(runs):
        raise ValueError("select-global predecessor census changed")
    result = {
        "protocol/SHA256SUMS": protocol_seal_sha256,
        "prepare/global/SHA256SUMS": prepare_campaign_seal_sha256,
        **{
            f"{rotation_commitment_index_relative_path(spec)}/SHA256SUMS": digest
            for spec, digest in zip(rotations, rotation_digests, strict=True)
        },
        **{
            f"{pool_commitment_relative_path(run)}/SHA256SUMS": leaf.seal_sha256
            for run, leaf in zip(runs, leaves, strict=True)
        },
    }
    if len(result) != 242:
        raise AssertionError("select-global predecessor census changed")
    return result


def _reveal_global_predecessors(
    *,
    protocol_seal_sha256: str,
    stage_global_seal_sha256: str,
    selection_barrier_seal_sha256: str,
    reveal_leaf_seal_sha256s: Sequence[str],
) -> dict[str, str]:
    runs = ordered_policy_runs()
    digests = tuple(reveal_leaf_seal_sha256s)
    if (
        len(digests) != len(runs)
        or any(type(value) is not str or _SHA256.fullmatch(value) is None for value in digests)
        or len(set(digests)) != len(runs)
    ):
        raise ValueError("reveal-global predecessor leaf authority is invalid")
    result = {
        "protocol/SHA256SUMS": protocol_seal_sha256,
        "stage/global/SHA256SUMS": stage_global_seal_sha256,
        "select/global/SHA256SUMS": selection_barrier_seal_sha256,
        **{
            f"{pool_reveal_relative_path(run)}/SHA256SUMS": digest
            for run, digest in zip(runs, digests, strict=True)
        },
    }
    if len(result) != 223:
        raise AssertionError("reveal-global predecessor census changed")
    return result


def launch_select_selector_worker(
    destination: str | Path,
    *,
    spec: RotationSpec,
    selector_kind: str,
    input_view_seal: PhaseSeal,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> PhaseSeal:
    """Publish and verify one least-authority selector in a fresh process."""

    worker_role, phase_artifact = _selector_worker_contract(selector_kind)
    final_parent = _trusted_directory(
        Path(destination).parent,
        label="selector destination parent",
        require_empty=False,
        exact_mode=0o700,
    )
    final = final_parent / Path(destination).name
    if os.path.lexists(final):
        raise FileExistsError("selector destination must be fresh")
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    common = {
        "spec": spec,
        "publication_identity": publication_identity,
        "protocol_capability": protocol_capability,
        "prepare_campaign_seal": prepare_campaign.seal,
        "expected_prepare_campaign_seal_sha256": (expected_prepare_campaign_seal_sha256),
    }
    if selector_kind == "prediction":
        request = SelectPredictionWorkerRequest(
            **common,
            prediction_view_seal=input_view_seal,
        )
    elif selector_kind == "random":
        request = SelectRandomWorkerRequest(
            **common,
            random_minimal_view_seal=input_view_seal,
        )
    elif selector_kind == "ceiling":
        request = SelectCeilingWorkerRequest(
            **common,
            random_minimal_view_seal=input_view_seal,
        )
    else:
        raise AssertionError("unreachable selector kind")
    result = _launch_fresh_worker(
        worker_role,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=worker_role,
            phase_artifact=phase_artifact,
            payload_paths=SELECTOR_PAYLOAD_PATHS,
        )
        if tuple(entry.name for entry in os.scandir(result.output)) != ("selector",):
            raise RuntimeError("selector worker outbox inventory is invalid")
        relocate_sealed_phase_noreplace(
            result.output / "selector",
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        seal = verify_phase(
            final,
            expected_artifact=phase_artifact,
            expected_payload_paths=SELECTOR_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        verify_selector_commitment_phase_capability(
            seal,
            spec=spec,
            selector_kind=selector_kind,
            input_view_seal=input_view_seal,
            protocol_capability=protocol_capability,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            publication_identity=publication_identity,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        _release_successful_outbox(result)
        return seal
    except BaseException as error:
        raise RuntimeError(
            f"{selector_kind} selector result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_select_rotation_worker(
    destination: str | Path,
    *,
    spec: RotationSpec,
    selector_phase_seals: Mapping[str, PhaseSeal],
    expected_selector_seal_sha256_by_kind: Mapping[str, str],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> tuple[tuple[PhaseSeal, ...], PhaseSeal]:
    """Publish one rotation's 11 commitments and index in a fresh assembler."""

    if not isinstance(selector_phase_seals, Mapping):
        raise ValueError("rotation worker requires exactly three selector capabilities")
    selectors = dict(selector_phase_seals)
    if set(selectors) != {"prediction", "random", "ceiling"}:
        raise ValueError("rotation worker requires exactly three selector capabilities")
    if any(type(seal) is not PhaseSeal for seal in selectors.values()):
        raise TypeError("rotation worker selector capabilities must be exact PhaseSeals")
    if not isinstance(expected_selector_seal_sha256_by_kind, Mapping):
        raise ValueError("rotation worker requires three external selector digest authorities")
    expected_selectors = dict(expected_selector_seal_sha256_by_kind)
    if set(expected_selectors) != {"prediction", "random", "ceiling"} or any(
        type(digest) is not str for digest in expected_selectors.values()
    ):
        raise ValueError("rotation worker requires three external selector digest authorities")
    final_parent = _trusted_directory(
        destination,
        label="select rotation destination",
        require_empty=True,
        exact_mode=0o700,
    )
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    request = SelectRotationWorkerRequest(
        spec=spec,
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign_seal=prepare_campaign.seal,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        prediction_selector_seal=selectors["prediction"],
        expected_prediction_selector_seal_sha256=expected_selectors["prediction"],
        random_selector_seal=selectors["random"],
        expected_random_selector_seal_sha256=expected_selectors["random"],
        ceiling_selector_seal=selectors["ceiling"],
        expected_ceiling_selector_seal_sha256=expected_selectors["ceiling"],
    )
    result = _launch_fresh_worker(
        SELECT_ROTATION_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        attestation = select_rotation_attestation_from_bytes(result.payload)
        if (
            type(attestation) is not SelectRotationAttestation
            or attestation.spec != spec
            or attestation.publication_identity != publication_identity
        ):
            raise ValueError("rotation attestation differs from launched authority")
        runs = policy_runs_for_rotation(spec)
        if tuple(item.track_id for item in attestation.commitment_leaves) != tuple(
            run.track_id for run in runs
        ):
            raise ValueError("rotation attestation track order changed")
        if tuple(sorted(entry.name for entry in os.scandir(result.output))) != (
            "commitments",
            "global",
        ):
            raise RuntimeError("rotation worker outbox inventory is invalid")
        source_commitments = _trusted_directory(
            result.output / "commitments",
            label="rotation worker commitment outbox",
            require_empty=False,
            exact_mode=0o700,
        )
        expected_track_ids = tuple(run.track_id for run in runs)
        if tuple(sorted(entry.name for entry in os.scandir(source_commitments))) != tuple(
            sorted(expected_track_ids)
        ):
            raise RuntimeError("rotation worker commitment inventory is invalid")
        for item in attestation.commitment_leaves:
            _validate_phase_attestation(
                item.publication,
                worker_role=SELECT_ROTATION_WORKER_ROLE,
                phase_artifact=POOL_COMMITMENT_ARTIFACT,
                payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
            )
        _validate_phase_attestation(
            attestation.rotation_index,
            worker_role=SELECT_ROTATION_WORKER_ROLE,
            phase_artifact=ROTATION_INDEX_ARTIFACT,
            payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
        )

        final_commitments = _create_private_directory(final_parent, "commitments")
        leaf_seals: list[PhaseSeal] = []
        for run, item in zip(runs, attestation.commitment_leaves, strict=True):
            publication = item.publication
            relocate_sealed_phase_noreplace(
                source_commitments / run.track_id,
                final_commitments / run.track_id,
                expected_seal_sha256=publication.phase_seal_sha256,
                expected_payload_sha256=dict(publication.payload_sha256),
            )
            seal = verify_phase(
                final_commitments / run.track_id,
                expected_artifact=POOL_COMMITMENT_ARTIFACT,
                expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
                expected_seal_sha256=publication.phase_seal_sha256,
            )
            selector_kind = _selector_kind_for_run(run)
            selector_seal = None if selector_kind is None else selectors[selector_kind]
            verify_pool_commitment_phase_capability(
                seal,
                run=run,
                protocol_capability=protocol_capability,
                prepare_campaign=prepare_campaign,
                expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
                selector_output_seal=selector_seal,
                expected_selector_output_seal_sha256=(
                    None if selector_kind is None else expected_selectors[selector_kind]
                ),
                publication_identity=publication_identity,
                expected_seal_sha256=publication.phase_seal_sha256,
            )
            leaf_seals.append(seal)

        rotation_publication = attestation.rotation_index
        relocate_sealed_phase_noreplace(
            result.output / "global",
            final_parent / "global",
            expected_seal_sha256=rotation_publication.phase_seal_sha256,
            expected_payload_sha256=dict(rotation_publication.payload_sha256),
        )
        rotation_seal = verify_phase(
            final_parent / "global",
            expected_artifact=ROTATION_INDEX_ARTIFACT,
            expected_payload_paths=ROTATION_INDEX_PAYLOAD_PATHS,
            expected_seal_sha256=rotation_publication.phase_seal_sha256,
        )
        rows = verify_rotation_commitment_index_capability(
            rotation_seal,
            spec=spec,
            protocol_capability=protocol_capability,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            publication_identity=publication_identity,
            expected_seal_sha256=rotation_publication.phase_seal_sha256,
        )
        _require_commitment_rows_bind_leaves(
            rows,
            leaf_seals,
            protocol_seal_sha256=protocol_capability.seal.seal_sha256,
            prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        )
        if any(
            row.selector_output_seal_sha256
            != (None if row.selector_kind is None else expected_selectors[row.selector_kind])
            for row in rows
        ):
            raise ValueError("rotation index does not bind the authorized selector outputs")
        os.rmdir(source_commitments)
        _release_successful_outbox(result)
        return tuple(leaf_seals), rotation_seal
    except BaseException as error:
        raise RuntimeError(
            f"select rotation result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_select_barrier_worker(
    destination: str | Path,
    *,
    rotation_index_seals: tuple[PhaseSeal, ...],
    expected_rotation_index_seal_sha256s: tuple[str, ...],
    commitment_leaf_seals: tuple[PhaseSeal, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    prepare_campaign: PrepareCampaignCapability,
    expected_prepare_campaign_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> PhaseSeal:
    """Publish and verify select/global in one fresh outcome-free assembler."""

    final_parent = _trusted_directory(
        Path(destination).parent,
        label="select-global destination parent",
        require_empty=False,
        exact_mode=0o700,
    )
    final = final_parent / Path(destination).name
    if os.path.lexists(final):
        raise FileExistsError("select-global destination must be fresh")
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    request = SelectBarrierWorkerRequest(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        prepare_campaign_seal=prepare_campaign.seal,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        rotation_index_seals=rotation_index_seals,
        expected_rotation_index_seal_sha256s=expected_rotation_index_seal_sha256s,
        commitment_leaf_seals=commitment_leaf_seals,
    )
    expected_rows = tuple(
        row
        for spec, rotation_seal, expected_digest in zip(
            ordered_rotations(),
            request.rotation_index_seals,
            request.expected_rotation_index_seal_sha256s,
            strict=True,
        )
        for row in verify_rotation_commitment_index_capability(
            rotation_seal,
            spec=spec,
            protocol_capability=protocol_capability,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            publication_identity=publication_identity,
            expected_seal_sha256=expected_digest,
        )
    )
    _require_commitment_rows_bind_leaves(
        expected_rows,
        request.commitment_leaf_seals,
        protocol_seal_sha256=protocol_capability.seal.seal_sha256,
        prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    result = _launch_fresh_worker(
        SELECT_BARRIER_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=SELECT_BARRIER_WORKER_ROLE,
            phase_artifact=CAMPAIGN_BARRIER_ARTIFACT,
            payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
        )
        if tuple(entry.name for entry in os.scandir(result.output)) != ("global",):
            raise RuntimeError("select barrier worker outbox inventory is invalid")
        relocate_sealed_phase_noreplace(
            result.output / "global",
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        seal = verify_phase(
            final,
            expected_artifact=CAMPAIGN_BARRIER_ARTIFACT,
            expected_payload_paths=CAMPAIGN_BARRIER_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        rows = verify_pool_commitment_campaign_barrier_capability(
            seal,
            protocol_capability=protocol_capability,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            publication_identity=publication_identity,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        if rows != expected_rows:
            raise ValueError("select-global rows differ from the exact authorized index rows")
        expected_predecessors = _select_global_predecessors(
            protocol_seal_sha256=protocol_capability.seal.seal_sha256,
            prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
            rotation_index_seal_sha256s=(request.expected_rotation_index_seal_sha256s),
            commitment_leaf_seals=request.commitment_leaf_seals,
        )
        if dict(seal.predecessor_seals) != expected_predecessors:
            raise ValueError("select-global does not bind the exact authorized input graph")
        _release_successful_outbox(result)
        return seal
    except BaseException as error:
        raise RuntimeError(
            f"select barrier result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_reveal_leaf_worker(
    destination: str | Path,
    *,
    run: PolicyRunSpec,
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    selection_barrier: PhaseSeal,
    commitment_leaf_seal: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    worker_scratch_root: str | Path,
    stage_manifest: StageManifestCapability | None = None,
    expected_stage_global_seal_sha256: str | None = None,
    pool_outcome_vault_seal: PhaseSeal | None = None,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> RevealLeafAttestation:
    """Publish one exact pool reveal in a fresh least-authority process."""

    if type(run) is not PolicyRunSpec or run not in ordered_policy_runs():
        raise ValueError("reveal worker requires one exact frozen policy run")
    if run.policy == NO_QUERY:
        if (
            stage_manifest is not None
            or expected_stage_global_seal_sha256 is not None
            or pool_outcome_vault_seal is not None
        ):
            raise ValueError("no-query reveal worker cannot receive stage or vault authority")
    elif (
        type(stage_manifest) is not StageManifestCapability
        or type(pool_outcome_vault_seal) is not PhaseSeal
        or type(expected_stage_global_seal_sha256) is not str
    ):
        raise ValueError("nonempty reveal worker requires exact stage and vault authority")
    final_parent = _trusted_directory(
        Path(destination).parent,
        label="reveal leaf destination parent",
        require_empty=False,
        exact_mode=0o700,
    )
    final = final_parent / Path(destination).name
    if os.path.lexists(final):
        raise FileExistsError("reveal leaf destination must be fresh")
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    commitment = pool_commitment_capability_from_seals(
        commitment_leaf_seal,
        selection_barrier,
        run=run,
        protocol_capability=protocol_capability,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        publication_identity=publication_identity,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
    )
    common = {
        "run": run,
        "publication_identity": publication_identity,
        "protocol_capability": protocol_capability,
        "selection_barrier_seal": selection_barrier,
        "commitment_leaf_seal": commitment_leaf_seal,
        "expected_prepare_campaign_seal_sha256": (expected_prepare_campaign_seal_sha256),
        "expected_selection_barrier_seal_sha256": (expected_selection_barrier_seal_sha256),
    }
    if run.policy == NO_QUERY:
        role = REVEAL_NO_QUERY_WORKER_ROLE
        request: RevealNoQueryWorkerRequest | RevealNonemptyWorkerRequest = (
            RevealNoQueryWorkerRequest(**common)
        )
        expected_stage = None
        expected_vault = None
    else:
        assert type(stage_manifest) is StageManifestCapability
        assert type(pool_outcome_vault_seal) is PhaseSeal
        assert type(expected_stage_global_seal_sha256) is str
        if (
            _SHA256.fullmatch(expected_stage_global_seal_sha256) is None
            or stage_manifest.seal.seal_sha256 != expected_stage_global_seal_sha256
        ):
            raise ValueError("nonempty reveal stage differs from external authority")
        vault_entry = stage_manifest.leaf(spec=run.rotation, role=POOL_OUTCOME_ROLE)
        if pool_outcome_vault_seal.seal_sha256 != vault_entry.leaf_seal_sha256:
            raise ValueError("nonempty reveal vault differs from stage-global index")
        role = REVEAL_NONEMPTY_WORKER_ROLE
        request = RevealNonemptyWorkerRequest(
            **common,
            stage_global_seal=stage_manifest.seal,
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            pool_outcome_vault_seal=pool_outcome_vault_seal,
        )
        expected_stage = expected_stage_global_seal_sha256
        expected_vault = pool_outcome_vault_seal.seal_sha256
    result = _launch_fresh_worker(
        role,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        if result.role != role or type(result.process_pid) is not int or result.process_pid <= 0:
            raise ValueError("reveal worker process identity differs from launched role")
        attestation = reveal_leaf_attestation_from_bytes(result.payload)
        commitment_payload_sha256 = dict(commitment_leaf_seal.payload_sha256)["commitment.json"]
        selected_ids = commitment.commitment.selected_sequence_ids
        selected_ids_payload_sha256 = hashlib.sha256(
            canonical_jsonl_bytes({"sequence_id": value} for value in selected_ids)
        ).hexdigest()
        if (
            attestation.run != run
            or attestation.publication_identity != publication_identity
            or attestation.protocol_seal_sha256 != protocol_capability.seal.seal_sha256
            or attestation.select_global_seal_sha256 != expected_selection_barrier_seal_sha256
            or attestation.stage_global_seal_sha256 != expected_stage
            or attestation.commitment_leaf_seal_sha256 != commitment_leaf_seal.seal_sha256
            or attestation.pool_outcome_vault_seal_sha256 != expected_vault
            or dict(attestation.payload_sha256)["commitment.json"] != commitment_payload_sha256
            or dict(attestation.payload_sha256)["selected-sequence-ids.jsonl"]
            != selected_ids_payload_sha256
            or attestation.selected_sequence_count != len(selected_ids)
            or attestation.selected_sequence_ids_sha256 != ordered_id_stream_sha256(selected_ids)
        ):
            raise ValueError("reveal worker attestation differs from launched authority")
        if tuple(entry.name for entry in os.scandir(result.output)) != ("reveal",):
            raise RuntimeError("reveal worker outbox inventory is invalid")
        relocate_sealed_phase_noreplace(
            result.output / "reveal",
            final,
            expected_seal_sha256=attestation.reveal_leaf_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        _observe_phase_marker_sha256(
            final,
            expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.reveal_leaf_seal_sha256,
        )
        _release_successful_outbox(result)
        return attestation
    except BaseException as error:
        raise RuntimeError(
            f"reveal leaf result was not accepted; outbox retained at {result.outbox}"
        ) from error


def launch_reveal_barrier_worker(
    destination: str | Path,
    *,
    attestations: tuple[RevealLeafAttestation, ...],
    expected_reveal_leaf_seal_sha256s: tuple[str, ...],
    publication_identity: SequentialV2PublicationIdentity,
    protocol_capability: ProtocolCapability,
    stage_manifest: StageManifestCapability,
    selection_barrier: PhaseSeal,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    worker_scratch_root: str | Path,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> RevealCampaignCapability:
    """Publish reveal/global in one fresh label-free custodian."""

    final_parent = _trusted_directory(
        Path(destination).parent,
        label="reveal-global destination parent",
        require_empty=False,
        exact_mode=0o700,
    )
    final = final_parent / Path(destination).name
    if os.path.lexists(final):
        raise FileExistsError("reveal-global destination must be fresh")
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_parent, scratch)
    request = RevealBarrierWorkerRequest(
        publication_identity=publication_identity,
        protocol_capability=protocol_capability,
        stage_global_seal=stage_manifest.seal,
        selection_barrier_seal=selection_barrier,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        attestations=attestations,
        expected_reveal_leaf_seal_sha256s=expected_reveal_leaf_seal_sha256s,
    )
    expected_index = canonical_jsonl_bytes(item.index_document() for item in attestations)
    expected_predecessors = _reveal_global_predecessors(
        protocol_seal_sha256=protocol_capability.seal.seal_sha256,
        stage_global_seal_sha256=expected_stage_global_seal_sha256,
        selection_barrier_seal_sha256=expected_selection_barrier_seal_sha256,
        reveal_leaf_seal_sha256s=expected_reveal_leaf_seal_sha256s,
    )
    result = _launch_fresh_worker(
        REVEAL_BARRIER_WORKER_ROLE,
        request.canonical_bytes(),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    try:
        if (
            result.role != REVEAL_BARRIER_WORKER_ROLE
            or type(result.process_pid) is not int
            or result.process_pid <= 0
        ):
            raise ValueError("reveal barrier process identity differs from launched role")
        attestation = phase_publication_attestation_from_bytes(result.payload)
        _validate_phase_attestation(
            attestation,
            worker_role=REVEAL_BARRIER_WORKER_ROLE,
            phase_artifact=REVEAL_CAMPAIGN_ARTIFACT,
            payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        )
        if tuple(entry.name for entry in os.scandir(result.output)) != ("global",):
            raise RuntimeError("reveal barrier worker outbox inventory is invalid")
        relocate_sealed_phase_noreplace(
            result.output / "global",
            final,
            expected_seal_sha256=attestation.phase_seal_sha256,
            expected_payload_sha256=dict(attestation.payload_sha256),
        )
        seal = verify_phase(
            final,
            expected_artifact=REVEAL_CAMPAIGN_ARTIFACT,
            expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
            expected_predecessor_seals=expected_predecessors,
            expected_seal_sha256=attestation.phase_seal_sha256,
        )
        if seal.read_payload_bytes("reveal-index.jsonl") != expected_index:
            raise ValueError("reveal-global index differs from exact leaf results")
        campaign = reveal_campaign_capability_from_seal(
            seal,
            publication_identity=publication_identity,
            protocol_capability=protocol_capability,
            stage_manifest_capability=stage_manifest,
            selection_barrier=selection_barrier,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
            expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
            expected_reveal_campaign_seal_sha256=attestation.phase_seal_sha256,
        )
        _release_successful_outbox(result)
        return campaign
    except BaseException as error:
        raise RuntimeError(
            f"reveal barrier result was not accepted; outbox retained at {result.outbox}"
        ) from error


def supervise_prepare_campaign(
    *,
    stage_root: str | Path,
    expected_source_anchors: Mapping[str, object],
    expected_stage_global_seal_sha256: str,
    run_root: str | Path,
    worker_scratch_root: str | Path,
    publication_identity: SequentialV2PublicationIdentity,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> PrepareSupervisionResult:
    """Run the exact protocol-through-prepare process DAG after stage exit."""

    final_root = _trusted_directory(
        run_root,
        label="sequential-v2 run root",
        require_empty=True,
        exact_mode=0o700,
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
    if not isinstance(expected_source_anchors, Mapping):
        raise TypeError("expected source anchors must be a mapping")

    # This controller-only call authenticates the outcome-free graph and never
    # invokes verify_trusted_stage, which would capture all eighty role leaves.
    stage = verify_stage_manifest(
        stage_path,
        expected_source_anchors=expected_source_anchors,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    stage_manifest = authenticate_stage_manifest_for_controller(
        stage,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    protocol = launch_protocol_worker(
        final_root / "protocol",
        publication_identity=publication_identity,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )

    prepare_root = _create_private_directory(final_root, "prepare")
    rotations_root = _create_private_directory(prepare_root, "rotations")
    attestations: list[PrepareRotationAttestation] = []
    for spec in ordered_rotations():
        destination = _create_private_directory(rotations_root, spec.rotation_id)
        expected_leaf = stage_manifest.leaf(spec=spec, role=PREPARE_ROLE)
        source_capsule = authenticate_stage_leaf_for_controller(
            stage_path / expected_leaf.relative_path,
            expected_leaf=expected_leaf,
            expected_source_anchors=expected_source_anchors,
        )
        attestations.append(
            launch_prepare_rotation_worker(
                destination,
                spec=spec,
                publication_identity=publication_identity,
                protocol_capability=protocol,
                stage_manifest=stage_manifest,
                expected_stage_global_seal_sha256=(expected_stage_global_seal_sha256),
                source_prepare_leaf_seal=source_capsule.seal,
                worker_scratch_root=scratch,
                timeout_seconds=timeout_seconds,
            )
        )
        del source_capsule
    ordered_attestations = tuple(attestations)
    campaign = launch_prepare_barrier_worker(
        prepare_root / "global",
        attestations=ordered_attestations,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    return PrepareSupervisionResult(
        protocol_capability=protocol,
        rotation_attestations=ordered_attestations,
        prepare_campaign=campaign,
        worker_process_count=22,
    )


def _capture_prepare_selector_view(
    run_root: Path,
    *,
    spec: RotationSpec,
    view_role: str,
    prepare_campaign: PrepareCampaignCapability,
) -> PhaseSeal:
    if view_role == PREDICTION_VIEW_ROLE:
        suffix = "prediction-view"
        artifact = PREDICTION_VIEW_ARTIFACT
        payload_paths = PREDICTION_VIEW_PAYLOAD_PATHS
    elif view_role == RANDOM_MINIMAL_VIEW_ROLE:
        suffix = "random-minimal-view"
        artifact = RANDOM_MINIMAL_VIEW_ARTIFACT
        payload_paths = RANDOM_MINIMAL_VIEW_PAYLOAD_PATHS
    else:
        raise ValueError("selector view role is invalid")
    return verify_phase(
        run_root / "prepare" / "rotations" / spec.rotation_id / suffix,
        expected_artifact=artifact,
        expected_payload_paths=payload_paths,
        expected_seal_sha256=prepare_campaign.leaf_seal_sha256(
            spec=spec,
            role=view_role,
        ),
    )


def supervise_select_campaign(
    *,
    run_root: str | Path,
    worker_scratch_root: str | Path,
    expected_protocol_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> SelectSupervisionResult:
    """Run the exact 60-selector, 20-rotation, select-global process DAG."""

    final_root = _trusted_directory(
        run_root,
        label="sequential-v2 run root",
        require_empty=False,
        exact_mode=0o700,
    )
    if tuple(sorted(entry.name for entry in os.scandir(final_root))) != (
        "prepare",
        "protocol",
    ):
        raise ValueError("select requires an exact unused protocol/prepare run root")
    scratch = _trusted_directory(
        worker_scratch_root,
        label="worker scratch root",
        require_empty=False,
    )
    _require_unrelated_same_filesystem(final_root, scratch)
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

    select_root = _create_private_directory(final_root, "select")
    selector_outputs_root = _create_private_directory(select_root, "selector-outputs")
    selector_by_rotation: dict[str, dict[str, PhaseSeal]] = {}
    selector_digest_by_rotation: dict[str, dict[str, str]] = {}
    selector_input_view_seals: list[PhaseSeal] = []
    selector_seals: list[PhaseSeal] = []
    for spec in ordered_rotations():
        rotation_selectors = _create_private_directory(
            selector_outputs_root,
            spec.rotation_id,
        )
        by_kind: dict[str, PhaseSeal] = {}

        prediction_view = _capture_prepare_selector_view(
            final_root,
            spec=spec,
            view_role=PREDICTION_VIEW_ROLE,
            prepare_campaign=prepare_campaign,
        )
        by_kind["prediction"] = launch_select_selector_worker(
            rotation_selectors / "prediction",
            spec=spec,
            selector_kind="prediction",
            input_view_seal=prediction_view,
            publication_identity=publication_identity,
            protocol_capability=protocol,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            worker_scratch_root=scratch,
            timeout_seconds=timeout_seconds,
        )
        selector_input_view_seals.append(prediction_view)
        del prediction_view

        random_view = _capture_prepare_selector_view(
            final_root,
            spec=spec,
            view_role=RANDOM_MINIMAL_VIEW_ROLE,
            prepare_campaign=prepare_campaign,
        )
        by_kind["random"] = launch_select_selector_worker(
            rotation_selectors / "random",
            spec=spec,
            selector_kind="random",
            input_view_seal=random_view,
            publication_identity=publication_identity,
            protocol_capability=protocol,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            worker_scratch_root=scratch,
            timeout_seconds=timeout_seconds,
        )
        selector_input_view_seals.append(random_view)
        del random_view

        ceiling_view = _capture_prepare_selector_view(
            final_root,
            spec=spec,
            view_role=RANDOM_MINIMAL_VIEW_ROLE,
            prepare_campaign=prepare_campaign,
        )
        by_kind["ceiling"] = launch_select_selector_worker(
            rotation_selectors / "ceiling",
            spec=spec,
            selector_kind="ceiling",
            input_view_seal=ceiling_view,
            publication_identity=publication_identity,
            protocol_capability=protocol,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            worker_scratch_root=scratch,
            timeout_seconds=timeout_seconds,
        )
        selector_input_view_seals.append(ceiling_view)
        del ceiling_view
        selector_by_rotation[spec.rotation_id] = by_kind
        selector_digest_by_rotation[spec.rotation_id] = {
            kind: by_kind[kind].seal_sha256 for kind in ("prediction", "random", "ceiling")
        }
        selector_seals.extend(by_kind[kind] for kind in ("prediction", "random", "ceiling"))

    # This directory is deliberately absent until all sixty selector outputs
    # have independently verified, making the frozen barrier observable.
    rotations_root = _create_private_directory(select_root, "rotations")
    commitment_leaf_seals: list[PhaseSeal] = []
    rotation_index_seals: list[PhaseSeal] = []
    rotation_index_digests: list[str] = []
    for spec in ordered_rotations():
        rotation_destination = _create_private_directory(
            rotations_root,
            spec.rotation_id,
        )
        leaves, index = launch_select_rotation_worker(
            rotation_destination,
            spec=spec,
            selector_phase_seals=selector_by_rotation[spec.rotation_id],
            expected_selector_seal_sha256_by_kind=(selector_digest_by_rotation[spec.rotation_id]),
            publication_identity=publication_identity,
            protocol_capability=protocol,
            prepare_campaign=prepare_campaign,
            expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
            worker_scratch_root=scratch,
            timeout_seconds=timeout_seconds,
        )
        commitment_leaf_seals.extend(leaves)
        rotation_index_seals.append(index)
        rotation_index_digests.append(index.seal_sha256)

    ordered_commitment_seals = tuple(commitment_leaf_seals)
    ordered_rotation_seals = tuple(rotation_index_seals)
    if len(ordered_commitment_seals) != len(ordered_policy_runs()):
        raise AssertionError("select supervisor commitment census changed")
    barrier = launch_select_barrier_worker(
        select_root / "global",
        rotation_index_seals=ordered_rotation_seals,
        expected_rotation_index_seal_sha256s=tuple(rotation_index_digests),
        commitment_leaf_seals=ordered_commitment_seals,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        prepare_campaign=prepare_campaign,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    return SelectSupervisionResult(
        publication_identity=publication_identity,
        protocol_capability=protocol,
        prepare_campaign=prepare_campaign,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        selector_input_view_seals=tuple(selector_input_view_seals),
        selector_seals=tuple(selector_seals),
        commitment_leaf_seals=ordered_commitment_seals,
        rotation_index_seals=ordered_rotation_seals,
        selection_barrier=barrier,
        worker_process_count=81,
    )


def supervise_reveal_campaign(
    *,
    stage_root: str | Path,
    expected_source_anchors: Mapping[str, object],
    run_root: str | Path,
    worker_scratch_root: str | Path,
    expected_protocol_seal_sha256: str,
    expected_prepare_campaign_seal_sha256: str,
    expected_stage_global_seal_sha256: str,
    expected_selection_barrier_seal_sha256: str,
    publication_identity: SequentialV2PublicationIdentity,
    timeout_seconds: float = _DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> RevealSupervisionResult:
    """Run the exact 220-leaf then reveal-global fresh-process DAG."""

    final_root = _trusted_directory(
        run_root,
        label="sequential-v2 run root",
        require_empty=False,
        exact_mode=0o700,
    )
    if tuple(sorted(entry.name for entry in os.scandir(final_root))) != (
        "prepare",
        "protocol",
        "select",
    ):
        raise ValueError("reveal requires an exact unused protocol/prepare/select run root")
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
    if not isinstance(expected_source_anchors, Mapping):
        raise TypeError("expected source anchors must be a mapping")

    stage = verify_stage_manifest(
        stage_path,
        expected_source_anchors=expected_source_anchors,
        expected_global_seal_sha256=expected_stage_global_seal_sha256,
    )
    stage_manifest = authenticate_stage_manifest_for_controller(
        stage,
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
    prepare_seal = verify_phase(
        final_root / "prepare" / "global",
        expected_artifact=PREPARE_CAMPAIGN_ARTIFACT,
        expected_payload_paths=CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=expected_prepare_campaign_seal_sha256,
    )
    verify_prepare_campaign_capability(
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
    commitment_rows = verify_pool_commitment_campaign_barrier_for_reveal(
        selection_barrier,
        protocol_capability=protocol,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        publication_identity=publication_identity,
        expected_seal_sha256=expected_selection_barrier_seal_sha256,
    )
    if tuple(item.run for item in commitment_rows) != ordered_policy_runs():
        raise ValueError("select-global commitment index differs from frozen reveal order")

    reveal_root = _create_private_directory(final_root, "reveal")
    tracks_root = _create_private_directory(reveal_root, "tracks")
    attestations: list[RevealLeafAttestation] = []
    for run, row in zip(ordered_policy_runs(), commitment_rows, strict=True):
        commitment_leaf = verify_phase(
            final_root / pool_commitment_relative_path(run),
            expected_artifact=POOL_COMMITMENT_ARTIFACT,
            expected_payload_paths=POOL_COMMITMENT_PAYLOAD_PATHS,
            expected_seal_sha256=row.leaf_seal_sha256,
        )
        if run.policy == NO_QUERY:
            attestation = launch_reveal_leaf_worker(
                tracks_root / run.track_id,
                run=run,
                publication_identity=publication_identity,
                protocol_capability=protocol,
                selection_barrier=selection_barrier,
                commitment_leaf_seal=commitment_leaf,
                expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
                expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
                worker_scratch_root=scratch,
                timeout_seconds=timeout_seconds,
            )
        else:
            expected_vault = stage_manifest.leaf(
                spec=run.rotation,
                role=POOL_OUTCOME_ROLE,
            )
            pool_capsule = authenticate_stage_leaf_for_controller(
                stage_path / expected_vault.relative_path,
                expected_leaf=expected_vault,
                expected_source_anchors=expected_source_anchors,
            )
            attestation = launch_reveal_leaf_worker(
                tracks_root / run.track_id,
                run=run,
                publication_identity=publication_identity,
                protocol_capability=protocol,
                selection_barrier=selection_barrier,
                commitment_leaf_seal=commitment_leaf,
                expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
                expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
                worker_scratch_root=scratch,
                stage_manifest=stage_manifest,
                expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
                pool_outcome_vault_seal=pool_capsule.seal,
                timeout_seconds=timeout_seconds,
            )
            del pool_capsule
        attestations.append(attestation)
        del commitment_leaf
    expected_track_ids = tuple(sorted(run.track_id for run in ordered_policy_runs()))
    if tuple(sorted(entry.name for entry in os.scandir(tracks_root))) != expected_track_ids:
        raise RuntimeError("reveal track inventory differs from the frozen campaign")

    ordered_attestations = tuple(attestations)
    # Take one contiguous controller snapshot only after the complete leaf set
    # exists; these canonical-path observations, rather than worker claims or
    # earlier per-leaf observations, are the barrier's digest authority.
    ordered_observed_seals = tuple(
        _observe_phase_marker_sha256(
            tracks_root / run.track_id,
            expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
            expected_seal_sha256=attestation.reveal_leaf_seal_sha256,
        )
        for run, attestation in zip(
            ordered_policy_runs(),
            ordered_attestations,
            strict=True,
        )
    )
    campaign = launch_reveal_barrier_worker(
        reveal_root / "global",
        attestations=ordered_attestations,
        expected_reveal_leaf_seal_sha256s=ordered_observed_seals,
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest=stage_manifest,
        selection_barrier=selection_barrier,
        expected_prepare_campaign_seal_sha256=expected_prepare_campaign_seal_sha256,
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        worker_scratch_root=scratch,
        timeout_seconds=timeout_seconds,
    )
    _observe_phase_marker_sha256(
        reveal_root / "global",
        expected_payload_paths=REVEAL_CAMPAIGN_PAYLOAD_PATHS,
        expected_seal_sha256=campaign.seal.seal_sha256,
    )
    post_barrier_observed_seals = tuple(
        _observe_phase_marker_sha256(
            tracks_root / run.track_id,
            expected_payload_paths=POOL_REVEAL_PAYLOAD_PATHS,
            expected_seal_sha256=expected_seal,
        )
        for run, expected_seal in zip(
            ordered_policy_runs(),
            ordered_observed_seals,
            strict=True,
        )
    )
    if post_barrier_observed_seals != ordered_observed_seals:
        raise RuntimeError("reveal leaves changed while reveal-global was published")
    if tuple(sorted(entry.name for entry in os.scandir(tracks_root))) != expected_track_ids:
        raise RuntimeError("reveal track inventory changed while reveal-global was published")
    if tuple(sorted(entry.name for entry in os.scandir(reveal_root))) != ("global", "tracks"):
        raise RuntimeError("reveal root inventory differs from the completed campaign")
    return RevealSupervisionResult(
        publication_identity=publication_identity,
        protocol_capability=protocol,
        stage_manifest=stage_manifest,
        selection_barrier=selection_barrier,
        expected_protocol_seal_sha256=expected_protocol_seal_sha256,
        expected_prepare_campaign_seal_sha256=(expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(expected_selection_barrier_seal_sha256),
        reveal_attestations=ordered_attestations,
        observed_reveal_leaf_seal_sha256s=ordered_observed_seals,
        reveal_campaign=campaign,
        worker_process_count=221,
    )


def _read_canonical_source_anchors(path: Path) -> dict[str, object]:
    requested = Path(os.path.abspath(os.fspath(path)))
    parent = _trusted_directory(
        requested.parent,
        label="source-anchors authority parent",
        require_empty=False,
    )
    requested = parent / requested.name
    named_before = os.lstat(requested)
    if (
        stat.S_ISLNK(named_before.st_mode)
        or not stat.S_ISREG(named_before.st_mode)
        or named_before.st_uid != os.geteuid()
        or named_before.st_nlink != 1
        or stat.S_IMODE(named_before.st_mode) & 0o022
        or named_before.st_size <= 0
        or named_before.st_size > _MAX_SOURCE_ANCHORS_BYTES
    ):
        raise ValueError(
            "source-anchors authority must be one bounded owned non-writable regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(requested, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or (before.st_dev, before.st_ino) != (named_before.st_dev, named_before.st_ino)
        ):
            raise ValueError("source-anchors authority must be one bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(
            descriptor,
            min(1024 * 1024, _MAX_SOURCE_ANCHORS_BYTES + 1 - total),
        ):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_SOURCE_ANCHORS_BYTES:
                raise ValueError("source-anchors authority exceeds its byte bound")
        after = os.fstat(descriptor)
        named_after = os.lstat(requested)
        fingerprint_fields = (
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
        if tuple(getattr(before, field) for field in fingerprint_fields) != tuple(
            getattr(after, field) for field in fingerprint_fields
        ) or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino):
            raise RuntimeError("source-anchors authority changed while it was read")
    finally:
        os.close(descriptor)
    return strict_canonical_json_object(
        b"".join(chunks),
        label="source-anchors authority",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amp-run-sequential-v2",
        description="Run fresh-executable sequential-v2 campaign boundaries.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="run protocol, twenty prepare workers, and prepare-global",
    )
    prepare.add_argument("--stage-root", required=True, type=Path)
    prepare.add_argument("--source-anchors-json", required=True, type=Path)
    prepare.add_argument("--expected-stage-global-sha256", required=True)
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--worker-scratch-root", required=True, type=Path)
    prepare.add_argument("--git-commit", required=True)
    prepare.add_argument("--code-manifest-sha256", required=True)
    prepare.add_argument("--config-sha256", required=True)
    prepare.add_argument("--lock-sha256", required=True)
    prepare.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    select_command = commands.add_parser(
        "select",
        help="run 60 selectors, 20 rotation assemblers, and select-global",
    )
    select_command.add_argument("--run-root", required=True, type=Path)
    select_command.add_argument("--worker-scratch-root", required=True, type=Path)
    select_command.add_argument("--expected-protocol-sha256", required=True)
    select_command.add_argument("--expected-prepare-global-sha256", required=True)
    select_command.add_argument("--git-commit", required=True)
    select_command.add_argument("--code-manifest-sha256", required=True)
    select_command.add_argument("--config-sha256", required=True)
    select_command.add_argument("--lock-sha256", required=True)
    select_command.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    reveal_command = commands.add_parser(
        "reveal",
        help="run 220 isolated reveals and reveal-global",
    )
    reveal_command.add_argument("--stage-root", required=True, type=Path)
    reveal_command.add_argument("--source-anchors-json", required=True, type=Path)
    reveal_command.add_argument("--run-root", required=True, type=Path)
    reveal_command.add_argument("--worker-scratch-root", required=True, type=Path)
    reveal_command.add_argument("--expected-protocol-sha256", required=True)
    reveal_command.add_argument("--expected-prepare-global-sha256", required=True)
    reveal_command.add_argument("--expected-stage-global-sha256", required=True)
    reveal_command.add_argument("--expected-select-global-sha256", required=True)
    reveal_command.add_argument("--git-commit", required=True)
    reveal_command.add_argument("--code-manifest-sha256", required=True)
    reveal_command.add_argument("--config-sha256", required=True)
    reveal_command.add_argument("--lock-sha256", required=True)
    reveal_command.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    update_command = commands.add_parser(
        "update",
        help="run 220 states, 20 components, 220 projections, and update-global",
    )
    update_command.add_argument("--stage-root", required=True, type=Path)
    update_command.add_argument("--source-anchors-json", required=True, type=Path)
    update_command.add_argument("--run-root", required=True, type=Path)
    update_command.add_argument("--worker-scratch-root", required=True, type=Path)
    update_command.add_argument("--expected-protocol-sha256", required=True)
    update_command.add_argument("--expected-prepare-global-sha256", required=True)
    update_command.add_argument("--expected-stage-global-sha256", required=True)
    update_command.add_argument("--expected-select-global-sha256", required=True)
    update_command.add_argument("--expected-reveal-global-sha256", required=True)
    update_command.add_argument("--git-commit", required=True)
    update_command.add_argument("--code-manifest-sha256", required=True)
    update_command.add_argument("--config-sha256", required=True)
    update_command.add_argument("--lock-sha256", required=True)
    update_command.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    outer_select_command = commands.add_parser(
        "outer-select",
        help="run 220 one-view selectors and the outer-selection barrier",
    )
    outer_select_command.add_argument("--run-root", required=True, type=Path)
    outer_select_command.add_argument("--worker-scratch-root", required=True, type=Path)
    outer_select_command.add_argument("--expected-protocol-sha256", required=True)
    outer_select_command.add_argument("--expected-stage-global-sha256", required=True)
    outer_select_command.add_argument("--expected-prepare-global-sha256", required=True)
    outer_select_command.add_argument("--expected-reveal-global-sha256", required=True)
    outer_select_command.add_argument("--expected-update-global-sha256", required=True)
    outer_select_command.add_argument("--git-commit", required=True)
    outer_select_command.add_argument("--code-manifest-sha256", required=True)
    outer_select_command.add_argument("--config-sha256", required=True)
    outer_select_command.add_argument("--lock-sha256", required=True)
    outer_select_command.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    measure_finalize_command = commands.add_parser(
        "measure-finalize-request",
        help="authenticate, measure, and publish the finalizer request without running it",
    )
    measure_finalize_command.add_argument("--stage-root", required=True, type=Path)
    measure_finalize_command.add_argument("--source-anchors-json", required=True, type=Path)
    measure_finalize_command.add_argument("--run-root", required=True, type=Path)
    measure_finalize_command.add_argument("--expected-protocol-sha256", required=True)
    measure_finalize_command.add_argument("--expected-stage-global-sha256", required=True)
    measure_finalize_command.add_argument("--expected-prepare-global-sha256", required=True)
    measure_finalize_command.add_argument("--expected-select-global-sha256", required=True)
    measure_finalize_command.add_argument("--expected-reveal-global-sha256", required=True)
    measure_finalize_command.add_argument("--expected-update-global-sha256", required=True)
    measure_finalize_command.add_argument("--expected-outer-select-global-sha256", required=True)
    measure_finalize_command.add_argument("--git-commit", required=True)
    measure_finalize_command.add_argument("--code-manifest-sha256", required=True)
    measure_finalize_command.add_argument("--config-sha256", required=True)
    measure_finalize_command.add_argument("--lock-sha256", required=True)
    measure_finalize_command.add_argument("--request-output", required=True, type=Path)
    finalize_command = commands.add_parser(
        "finalize",
        help="run the one post-outer-selection finalization worker",
    )
    finalize_command.add_argument("--stage-root", required=True, type=Path)
    finalize_command.add_argument("--source-anchors-json", required=True, type=Path)
    finalize_command.add_argument("--run-root", required=True, type=Path)
    finalize_command.add_argument("--worker-scratch-root", required=True, type=Path)
    finalize_command.add_argument("--expected-protocol-sha256", required=True)
    finalize_command.add_argument("--expected-stage-global-sha256", required=True)
    finalize_command.add_argument("--expected-prepare-global-sha256", required=True)
    finalize_command.add_argument("--expected-select-global-sha256", required=True)
    finalize_command.add_argument("--expected-reveal-global-sha256", required=True)
    finalize_command.add_argument("--expected-update-global-sha256", required=True)
    finalize_command.add_argument("--expected-outer-select-global-sha256", required=True)
    finalize_command.add_argument("--git-commit", required=True)
    finalize_command.add_argument("--code-manifest-sha256", required=True)
    finalize_command.add_argument("--config-sha256", required=True)
    finalize_command.add_argument("--lock-sha256", required=True)
    finalize_command.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one implemented fresh-executable campaign boundary."""

    arguments = _parser().parse_args(argv)
    identity = SequentialV2PublicationIdentity(
        git_commit=arguments.git_commit,
        code_manifest_sha256=arguments.code_manifest_sha256,
        config_sha256=arguments.config_sha256,
        lock_sha256=arguments.lock_sha256,
    )
    if arguments.command == "prepare":
        result = supervise_prepare_campaign(
            stage_root=arguments.stage_root,
            expected_source_anchors=_read_canonical_source_anchors(arguments.source_anchors_json),
            expected_stage_global_seal_sha256=(arguments.expected_stage_global_sha256),
            run_root=arguments.run_root,
            worker_scratch_root=arguments.worker_scratch_root,
            publication_identity=identity,
            timeout_seconds=float(arguments.worker_timeout_seconds),
        )
    elif arguments.command == "select":
        result = supervise_select_campaign(
            run_root=arguments.run_root,
            worker_scratch_root=arguments.worker_scratch_root,
            expected_protocol_seal_sha256=arguments.expected_protocol_sha256,
            expected_prepare_campaign_seal_sha256=(arguments.expected_prepare_global_sha256),
            publication_identity=identity,
            timeout_seconds=float(arguments.worker_timeout_seconds),
        )
    elif arguments.command == "reveal":
        result = supervise_reveal_campaign(
            stage_root=arguments.stage_root,
            expected_source_anchors=_read_canonical_source_anchors(arguments.source_anchors_json),
            run_root=arguments.run_root,
            worker_scratch_root=arguments.worker_scratch_root,
            expected_protocol_seal_sha256=arguments.expected_protocol_sha256,
            expected_prepare_campaign_seal_sha256=(arguments.expected_prepare_global_sha256),
            expected_stage_global_seal_sha256=arguments.expected_stage_global_sha256,
            expected_selection_barrier_seal_sha256=(arguments.expected_select_global_sha256),
            publication_identity=identity,
            timeout_seconds=float(arguments.worker_timeout_seconds),
        )
    elif arguments.command == "update":
        from amp_challenge.evaluation.sequential_v2_update_supervisor import (
            supervise_update_campaign,
        )

        result = supervise_update_campaign(
            stage_root=arguments.stage_root,
            expected_source_anchors=_read_canonical_source_anchors(arguments.source_anchors_json),
            run_root=arguments.run_root,
            worker_scratch_root=arguments.worker_scratch_root,
            expected_protocol_seal_sha256=arguments.expected_protocol_sha256,
            expected_prepare_campaign_seal_sha256=(arguments.expected_prepare_global_sha256),
            expected_stage_global_seal_sha256=arguments.expected_stage_global_sha256,
            expected_selection_barrier_seal_sha256=(arguments.expected_select_global_sha256),
            expected_reveal_campaign_seal_sha256=(arguments.expected_reveal_global_sha256),
            publication_identity=identity,
            timeout_seconds=float(arguments.worker_timeout_seconds),
        )
    elif arguments.command == "outer-select":
        from amp_challenge.evaluation.sequential_v2_outer_select_supervisor import (
            supervise_outer_selection_campaign,
        )

        result = supervise_outer_selection_campaign(
            run_root=arguments.run_root,
            worker_scratch_root=arguments.worker_scratch_root,
            expected_protocol_seal_sha256=arguments.expected_protocol_sha256,
            expected_stage_global_seal_sha256=arguments.expected_stage_global_sha256,
            expected_prepare_campaign_seal_sha256=(arguments.expected_prepare_global_sha256),
            expected_reveal_campaign_seal_sha256=(arguments.expected_reveal_global_sha256),
            expected_update_campaign_seal_sha256=(arguments.expected_update_global_sha256),
            publication_identity=identity,
            timeout_seconds=float(arguments.worker_timeout_seconds),
        )
    elif arguments.command == "measure-finalize-request":
        from amp_challenge.evaluation.sequential_v2_finalize_supervisor import (
            measure_finalize_campaign_request,
        )

        result = measure_finalize_campaign_request(
            stage_root=arguments.stage_root,
            expected_source_anchors=_read_canonical_source_anchors(arguments.source_anchors_json),
            run_root=arguments.run_root,
            request_output=arguments.request_output,
            expected_protocol_seal_sha256=arguments.expected_protocol_sha256,
            expected_stage_global_seal_sha256=arguments.expected_stage_global_sha256,
            expected_prepare_campaign_seal_sha256=(arguments.expected_prepare_global_sha256),
            expected_selection_campaign_seal_sha256=(arguments.expected_select_global_sha256),
            expected_reveal_campaign_seal_sha256=(arguments.expected_reveal_global_sha256),
            expected_update_campaign_seal_sha256=(arguments.expected_update_global_sha256),
            expected_outer_selection_campaign_seal_sha256=(
                arguments.expected_outer_select_global_sha256
            ),
            publication_identity=identity,
        )
    elif arguments.command == "finalize":
        from amp_challenge.evaluation.sequential_v2_finalize_supervisor import (
            supervise_finalize_campaign,
        )

        result = supervise_finalize_campaign(
            stage_root=arguments.stage_root,
            expected_source_anchors=_read_canonical_source_anchors(arguments.source_anchors_json),
            run_root=arguments.run_root,
            worker_scratch_root=arguments.worker_scratch_root,
            expected_protocol_seal_sha256=arguments.expected_protocol_sha256,
            expected_stage_global_seal_sha256=arguments.expected_stage_global_sha256,
            expected_prepare_campaign_seal_sha256=(arguments.expected_prepare_global_sha256),
            expected_selection_campaign_seal_sha256=(arguments.expected_select_global_sha256),
            expected_reveal_campaign_seal_sha256=(arguments.expected_reveal_global_sha256),
            expected_update_campaign_seal_sha256=(arguments.expected_update_global_sha256),
            expected_outer_selection_campaign_seal_sha256=(
                arguments.expected_outer_select_global_sha256
            ),
            publication_identity=identity,
            timeout_seconds=float(arguments.worker_timeout_seconds),
        )
    else:
        raise AssertionError("unreachable sequential-v2 supervisor command")
    _write_all(sys.stdout.fileno(), canonical_json_bytes(result.document()))
    return 0


__all__ = [
    "PREPARE_SUPERVISION_ARTIFACT",
    "REVEAL_SUPERVISION_ARTIFACT",
    "SELECT_SUPERVISION_ARTIFACT",
    "PrepareSupervisionResult",
    "RevealSupervisionResult",
    "SelectSupervisionResult",
    "launch_prepare_barrier_worker",
    "launch_prepare_rotation_worker",
    "launch_protocol_worker",
    "launch_reveal_barrier_worker",
    "launch_reveal_leaf_worker",
    "launch_select_barrier_worker",
    "launch_select_rotation_worker",
    "launch_select_selector_worker",
    "main",
    "supervise_prepare_campaign",
    "supervise_reveal_campaign",
    "supervise_select_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())

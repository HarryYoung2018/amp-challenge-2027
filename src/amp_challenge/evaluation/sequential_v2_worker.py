"""Fresh-executable custodians through the sequential-v2 finalize boundary.

Each invocation starts in a private courier/outbox directory containing only a
mode-0400 canonical request and an empty mode-0700 ``output`` directory.  The
request carries rootless captured capabilities, never an original source path.
The worker closes and unlinks that request before computation, publishes only
inside its isolated outbox, and writes one framed canonical attestation to the
sole inherited output descriptor.  Standard input, output, and error are
discarded by the supervisor.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

_REQUEST_NAME = "request.json"
_OUTPUT_NAME = "output"
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_RESULT_BYTES = 1 * 1024 * 1024
_FRAME_BYTES = 8
_FAILURE_ARTIFACT = "sequential_v2_fresh_worker_failure_v1"
PROTOCOL_WORKER_ROLE = "protocol"
PREPARE_WORKER_ROLE = "prepare"
PREPARE_BARRIER_WORKER_ROLE = "prepare-barrier"
SELECT_PREDICTION_WORKER_ROLE = "select-prediction"
SELECT_RANDOM_WORKER_ROLE = "select-random"
SELECT_CEILING_WORKER_ROLE = "select-ceiling"
SELECT_ROTATION_WORKER_ROLE = "select-rotation"
SELECT_BARRIER_WORKER_ROLE = "select-barrier"
REVEAL_NO_QUERY_WORKER_ROLE = "reveal-no-query"
REVEAL_NONEMPTY_WORKER_ROLE = "reveal-nonempty"
REVEAL_BARRIER_WORKER_ROLE = "reveal-barrier"
UPDATE_STATE_WORKER_ROLE = "update-state"
UPDATE_COMPONENT_WORKER_ROLE = "update-component"
UPDATE_PROJECTION_WORKER_ROLE = "update-projection"
UPDATE_BARRIER_WORKER_ROLE = "update-barrier"
OUTER_SELECT_TRACK_WORKER_ROLE = "outer-select-track"
OUTER_SELECT_BARRIER_WORKER_ROLE = "outer-select-barrier"
FINALIZE_WORKER_ROLE = "finalize"


class _BoundaryViolation(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def _assert_initial_descriptor_inventory(result_descriptor: int) -> None:
    """Require standard /dev/null streams plus the sole output-only pipe."""

    null_metadata = os.stat(os.devnull)
    null_identity = (
        null_metadata.st_dev,
        null_metadata.st_ino,
        null_metadata.st_rdev,
    )
    expected_access = (os.O_RDONLY, os.O_WRONLY, os.O_WRONLY)
    for descriptor, access in zip((0, 1, 2), expected_access, strict=True):
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISCHR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino, metadata.st_rdev) != null_identity
            or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != access
        ):
            raise _BoundaryViolation("standard-stream-not-devnull")
    result_metadata = os.fstat(result_descriptor)
    result_status_flags = fcntl.fcntl(result_descriptor, fcntl.F_GETFL)
    result_descriptor_flags = fcntl.fcntl(result_descriptor, fcntl.F_GETFD)
    if (
        not stat.S_ISFIFO(result_metadata.st_mode)
        or result_status_flags & os.O_ACCMODE != os.O_WRONLY
        or result_descriptor_flags & fcntl.FD_CLOEXEC == 0
    ):
        raise _BoundaryViolation("result-descriptor-not-output-only-pipe")
    try:
        names = os.listdir("/proc/self/fd")
    except OSError as error:
        raise _BoundaryViolation("proc-descriptor-audit-unavailable") from error
    allowed = {0, 1, 2, result_descriptor}
    for name in names:
        if not name.isascii() or not name.isdecimal():
            raise _BoundaryViolation("malformed-descriptor-inventory")
        descriptor = int(name)
        if descriptor in allowed:
            continue
        try:
            metadata = os.fstat(descriptor)
        except OSError:
            # CPython may expose the transient directory descriptor used by
            # os.listdir; it is already closed by the time we inspect it.
            continue
        if stat.S_ISCHR(metadata.st_mode):
            kind = "character"
        elif stat.S_ISFIFO(metadata.st_mode):
            kind = "pipe"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "regular"
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        else:
            kind = "other"
        raise _BoundaryViolation(f"unauthorized-inherited-descriptor-{descriptor}-{kind}")


def _private_current_directory() -> Path:
    current = Path.cwd()
    metadata = os.lstat(current)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("worker current directory must be a current-user-owned mode-0700 tree")
    output = current / _OUTPUT_NAME
    output_metadata = os.lstat(output)
    if (
        stat.S_ISLNK(output_metadata.st_mode)
        or not stat.S_ISDIR(output_metadata.st_mode)
        or output_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(output_metadata.st_mode) != 0o700
        or tuple(os.scandir(output))
    ):
        raise ValueError("worker output directory must be an empty owned mode-0700 tree")
    if tuple(sorted(entry.name for entry in os.scandir(current))) != (
        _OUTPUT_NAME,
        _REQUEST_NAME,
    ):
        raise ValueError("worker courier directory has an unexpected inventory")
    return current


def _read_and_remove_request(current: Path) -> bytes:
    request = current / _REQUEST_NAME
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(request, flags)
    chunks: list[bytes] = []
    total = 0
    try:
        before = os.fstat(descriptor)
        named = os.lstat(request)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
            or before.st_size <= 0
            or before.st_size > _MAX_REQUEST_BYTES
        ):
            raise ValueError("worker request must be one bounded owned mode-0400 regular file")
        while chunk := os.read(descriptor, min(1024 * 1024, _MAX_REQUEST_BYTES + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_REQUEST_BYTES:
                raise ValueError("worker request exceeds its byte bound")
        after = os.fstat(descriptor)
        named_after = os.lstat(request)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino):
            raise RuntimeError("worker request changed while it was read")
    finally:
        os.close(descriptor)
    os.unlink(request)
    directory_descriptor = os.open(
        current,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    payload = b"".join(chunks)
    if not payload:
        raise ValueError("worker request is empty")
    return payload


def _write_result(descriptor: int, payload: bytes) -> None:
    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("worker result descriptor is unsafe")
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_RESULT_BYTES:
        raise ValueError("worker result is not bounded exact bytes")
    os.set_inheritable(descriptor, False)
    framed = len(payload).to_bytes(_FRAME_BYTES, "big") + payload
    view = memoryview(framed)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("worker result descriptor made no write progress")
        written += count
    os.close(descriptor)


def _run_protocol(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        publish_protocol_capability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
        protocol_worker_request_from_bytes,
    )

    request = protocol_worker_request_from_bytes(request_payload)
    capability = publish_protocol_capability(
        output / "protocol",
        publication_identity=request.publication_identity,
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=PROTOCOL_WORKER_ROLE,
        seal=capability.seal,
    ).canonical_bytes()


def _run_prepare(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_prepare import prepare_rotation
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        publish_prepared_rotation_artifacts,
    )
    from amp_challenge.evaluation.sequential_v2_stage import (
        PREPARE_ROLE,
        AuthenticatedLeafCapsule,
        prepare_capability_from_capsule,
        verify_stage_manifest_capability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        prepare_worker_request_from_bytes,
    )

    request = prepare_worker_request_from_bytes(request_payload)
    stage = verify_stage_manifest_capability(
        request.stage_global_seal,
        expected_global_seal_sha256=request.expected_stage_global_seal_sha256,
    )
    entry = stage.leaf(spec=request.spec, role=PREPARE_ROLE)
    source_capsule = AuthenticatedLeafCapsule(
        entry=entry,
        seal=request.source_prepare_leaf_seal,
        source_anchors_sha256=stage.source_anchors_sha256,
        source_predecessors=tuple(sorted(stage.source_predecessors)),
    )
    capability = prepare_capability_from_capsule(source_capsule)
    prepared = prepare_rotation(capability)
    attestation = publish_prepared_rotation_artifacts(
        output,
        prepared=prepared,
        capability=capability,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        stage_global_seal=request.stage_global_seal,
        expected_stage_global_seal_sha256=(request.expected_stage_global_seal_sha256),
        source_prepare_leaf_seal=request.source_prepare_leaf_seal,
    )
    return attestation.canonical_bytes()


def _run_prepare_barrier(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        publish_prepare_campaign_barrier,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
        prepare_barrier_worker_request_from_bytes,
    )

    request = prepare_barrier_worker_request_from_bytes(request_payload)
    campaign = publish_prepare_campaign_barrier(
        output / "global",
        attestations=request.attestations,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=PREPARE_BARRIER_WORKER_ROLE,
        seal=campaign.seal,
    ).canonical_bytes()


def _run_select_prediction(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        publish_prediction_selector_commitments,
    )
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
        select_prediction_worker_request_from_bytes,
    )

    request = select_prediction_worker_request_from_bytes(request_payload)
    seal = publish_prediction_selector_commitments(
        output / "selector",
        spec=request.spec,
        prediction_view_seal=request.prediction_view_seal,
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=SELECT_PREDICTION_WORKER_ROLE,
        seal=seal,
    ).canonical_bytes()


def _run_select_random(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        publish_random_selector_commitments,
    )
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
        select_random_worker_request_from_bytes,
    )

    request = select_random_worker_request_from_bytes(request_payload)
    seal = publish_random_selector_commitments(
        output / "selector",
        spec=request.spec,
        random_minimal_view_seal=request.random_minimal_view_seal,
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=SELECT_RANDOM_WORKER_ROLE,
        seal=seal,
    ).canonical_bytes()


def _run_select_ceiling(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        publish_ceiling_selector_commitment,
    )
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
        select_ceiling_worker_request_from_bytes,
    )

    request = select_ceiling_worker_request_from_bytes(request_payload)
    seal = publish_ceiling_selector_commitment(
        output / "selector",
        spec=request.spec,
        random_minimal_view_seal=request.random_minimal_view_seal,
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=SELECT_CEILING_WORKER_ROLE,
        seal=seal,
    ).canonical_bytes()


def _run_select_rotation(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        publish_rotation_commitment_bundle,
    )
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        SelectRotationAttestation,
        select_rotation_worker_request_from_bytes,
    )

    request = select_rotation_worker_request_from_bytes(request_payload)
    selector_seals = {
        "prediction": request.prediction_selector_seal,
        "random": request.random_selector_seal,
        "ceiling": request.ceiling_selector_seal,
    }
    bundle = publish_rotation_commitment_bundle(
        output,
        spec=request.spec,
        selector_phase_seals=selector_seals,
        expected_selector_seal_sha256_by_kind={
            "prediction": request.expected_prediction_selector_seal_sha256,
            "random": request.expected_random_selector_seal_sha256,
            "ceiling": request.expected_ceiling_selector_seal_sha256,
        },
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
    )
    return SelectRotationAttestation.from_seals(
        spec=request.spec,
        publication_identity=request.publication_identity,
        commitment_leaf_seals=bundle.commitment_leaf_seals,
        rotation_index_seal=bundle.rotation_index_seal,
    ).canonical_bytes()


def _run_select_barrier(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        publish_pool_commitment_campaign_barrier,
    )
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
        select_barrier_worker_request_from_bytes,
    )

    request = select_barrier_worker_request_from_bytes(request_payload)
    seal = publish_pool_commitment_campaign_barrier(
        output / "global",
        rotation_index_seals=request.rotation_index_seals,
        commitment_leaf_seals=request.commitment_leaf_seals,
        expected_rotation_index_seal_sha256s=(request.expected_rotation_index_seal_sha256s),
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=SELECT_BARRIER_WORKER_ROLE,
        seal=seal,
    ).canonical_bytes()


def _run_reveal_no_query(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        pool_commitment_capability_from_seals,
    )
    from amp_challenge.evaluation.sequential_v2_reveal import (
        publish_no_query_pool_reveal,
    )
    from amp_challenge.evaluation.sequential_v2_reveal_wire import (
        reveal_no_query_worker_request_from_bytes,
    )

    request = reveal_no_query_worker_request_from_bytes(request_payload)
    commitment = pool_commitment_capability_from_seals(
        request.commitment_leaf_seal,
        request.selection_barrier_seal,
        run=request.run,
        protocol_capability=request.protocol_capability,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
    )
    attestation = publish_no_query_pool_reveal(
        output / "reveal",
        run=request.run,
        commitment_capability=commitment,
        protocol_capability=request.protocol_capability,
        publication_identity=request.publication_identity,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
    )
    return attestation.canonical_bytes()


def _run_reveal_nonempty(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_commitments import (
        pool_commitment_capability_from_seals,
    )
    from amp_challenge.evaluation.sequential_v2_reveal import (
        publish_nonempty_pool_reveal,
    )
    from amp_challenge.evaluation.sequential_v2_reveal_wire import (
        reveal_nonempty_worker_request_from_bytes,
    )
    from amp_challenge.evaluation.sequential_v2_stage import (
        POOL_OUTCOME_ROLE,
        AuthenticatedLeafCapsule,
        verify_stage_manifest_capability,
    )

    request = reveal_nonempty_worker_request_from_bytes(request_payload)
    commitment = pool_commitment_capability_from_seals(
        request.commitment_leaf_seal,
        request.selection_barrier_seal,
        run=request.run,
        protocol_capability=request.protocol_capability,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        publication_identity=request.publication_identity,
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
    )
    stage = verify_stage_manifest_capability(
        request.stage_global_seal,
        expected_global_seal_sha256=request.expected_stage_global_seal_sha256,
    )
    entry = stage.leaf(spec=request.run.rotation, role=POOL_OUTCOME_ROLE)
    pool_capsule = AuthenticatedLeafCapsule(
        entry=entry,
        seal=request.pool_outcome_vault_seal,
        source_anchors_sha256=stage.source_anchors_sha256,
        source_predecessors=tuple(sorted(stage.source_predecessors)),
    )
    attestation = publish_nonempty_pool_reveal(
        output / "reveal",
        run=request.run,
        commitment_capability=commitment,
        protocol_capability=request.protocol_capability,
        publication_identity=request.publication_identity,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
        stage_manifest_capability=stage,
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        pool_outcome_vault_capsule=pool_capsule,
    )
    return attestation.canonical_bytes()


def _run_reveal_barrier(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_reveal import (
        publish_pool_reveal_campaign_barrier,
    )
    from amp_challenge.evaluation.sequential_v2_reveal_wire import (
        reveal_barrier_worker_request_from_bytes,
    )
    from amp_challenge.evaluation.sequential_v2_stage import (
        verify_stage_manifest_capability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
    )

    request = reveal_barrier_worker_request_from_bytes(request_payload)
    stage = verify_stage_manifest_capability(
        request.stage_global_seal,
        expected_global_seal_sha256=request.expected_stage_global_seal_sha256,
    )
    campaign = publish_pool_reveal_campaign_barrier(
        output / "global",
        attestations=request.attestations,
        expected_reveal_leaf_seal_sha256s=(request.expected_reveal_leaf_seal_sha256s),
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        stage_manifest_capability=stage,
        selection_barrier=request.selection_barrier_seal,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=REVEAL_BARRIER_WORKER_ROLE,
        seal=campaign.seal,
    ).canonical_bytes()


def _run_update_state(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_reveal import (
        RevealCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_stage import (
        StageManifestCapability,
    )
    from amp_challenge.evaluation.sequential_v2_update_state import (
        publish_update_state,
    )
    from amp_challenge.evaluation.sequential_v2_update_wire import (
        update_state_worker_request_from_bytes,
    )

    request = update_state_worker_request_from_bytes(request_payload)
    attestation = publish_update_state(
        output / "state",
        run=request.run,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        base_update_seal=request.base_update_seal,
        reveal_campaign=RevealCampaignCapability(
            request.reveal_campaign_seal,
            request.publication_identity,
        ),
        reveal_seal=request.reveal_seal,
        stage_manifest_capability=StageManifestCapability(request.stage_manifest_seal),
        selection_barrier=request.selection_barrier,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
    )
    return attestation.canonical_bytes()


def _run_update_component(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_stage import (
        StageManifestCapability,
    )
    from amp_challenge.evaluation.sequential_v2_update_outer import (
        publish_outer_component,
    )
    from amp_challenge.evaluation.sequential_v2_update_wire import (
        update_component_worker_request_from_bytes,
    )

    request = update_component_worker_request_from_bytes(request_payload)
    attestation = publish_outer_component(
        output / "component",
        spec=request.spec,
        state_attestations=request.state_attestations,
        expected_state_leaf_seal_sha256s=request.expected_state_leaf_seal_sha256s,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        stage_manifest_capability=StageManifestCapability(request.stage_manifest_seal),
        outer_metadata_capsule=request.outer_metadata_capsule,
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
    )
    return attestation.canonical_bytes()


def _run_update_projection(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_reveal import (
        RevealCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_stage import (
        StageManifestCapability,
    )
    from amp_challenge.evaluation.sequential_v2_update_outer import (
        publish_outer_projection,
    )
    from amp_challenge.evaluation.sequential_v2_update_wire import (
        update_projection_worker_request_from_bytes,
    )

    request = update_projection_worker_request_from_bytes(request_payload)
    attestation = publish_outer_projection(
        output,
        run=request.run,
        state_seal=request.state_seal,
        state_attestation=request.state_attestation,
        component_seal=request.component_seal,
        component_attestation=request.component_attestation,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        reveal_campaign=RevealCampaignCapability(
            request.reveal_campaign_seal,
            request.publication_identity,
        ),
        stage_manifest_capability=StageManifestCapability(request.stage_manifest_seal),
        selection_barrier=request.selection_barrier,
        outer_metadata_capsule=request.outer_metadata_capsule,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
        expected_state_leaf_seal_sha256=request.expected_state_leaf_seal_sha256,
        expected_component_leaf_seal_sha256=request.expected_component_leaf_seal_sha256,
    )
    return attestation.canonical_bytes()


def _run_update_barrier(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_prepare_artifacts import (
        PrepareCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_reveal import (
        RevealCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_stage import (
        StageManifestCapability,
    )
    from amp_challenge.evaluation.sequential_v2_update_campaign import (
        publish_update_campaign_barrier,
    )
    from amp_challenge.evaluation.sequential_v2_update_wire import (
        update_barrier_worker_request_from_bytes,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
    )

    request = update_barrier_worker_request_from_bytes(request_payload)
    campaign = publish_update_campaign_barrier(
        output / "global",
        state_attestations=request.state_attestations,
        component_attestations=request.component_attestations,
        projection_attestations=request.projection_attestations,
        expected_update_leaf_seal_sha256s=request.expected_update_leaf_seal_sha256s,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        stage_manifest_capability=StageManifestCapability(request.stage_manifest_seal),
        prepare_campaign=PrepareCampaignCapability(
            request.prepare_campaign_seal,
            request.publication_identity,
        ),
        reveal_campaign=RevealCampaignCapability(
            request.reveal_campaign_seal,
            request.publication_identity,
        ),
        selection_barrier=request.selection_barrier,
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_selection_barrier_seal_sha256=(request.expected_selection_barrier_seal_sha256),
        expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=UPDATE_BARRIER_WORKER_ROLE,
        seal=campaign.seal,
    ).canonical_bytes()


def _run_outer_select_track(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_outer_select import (
        publish_outer_selection_commitment,
    )
    from amp_challenge.evaluation.sequential_v2_outer_select_wire import (
        outer_select_track_worker_request_from_bytes,
    )
    from amp_challenge.evaluation.sequential_v2_update_campaign import (
        UpdateCampaignCapability,
    )

    request = outer_select_track_worker_request_from_bytes(request_payload)
    attestation = publish_outer_selection_commitment(
        output / "commitment",
        run=request.run,
        update_campaign=UpdateCampaignCapability(
            request.update_campaign_seal,
            request.publication_identity,
        ),
        outer_view_seal=request.outer_view_seal,
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
        expected_update_campaign_seal_sha256=(request.expected_update_campaign_seal_sha256),
    )
    return attestation.canonical_bytes()


def _run_outer_select_barrier(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_outer_select import (
        publish_outer_selection_campaign_barrier,
    )
    from amp_challenge.evaluation.sequential_v2_outer_select_wire import (
        outer_select_barrier_worker_request_from_bytes,
    )
    from amp_challenge.evaluation.sequential_v2_update_campaign import (
        UpdateCampaignCapability,
    )
    from amp_challenge.evaluation.sequential_v2_wire import (
        PhasePublicationAttestation,
    )

    request = outer_select_barrier_worker_request_from_bytes(request_payload)
    campaign = publish_outer_selection_campaign_barrier(
        output / "global",
        selection_attestations=request.selection_attestations,
        expected_outer_selection_leaf_seal_sha256s=(
            request.expected_outer_selection_leaf_seal_sha256s
        ),
        publication_identity=request.publication_identity,
        protocol_capability=request.protocol_capability,
        update_campaign=UpdateCampaignCapability(
            request.update_campaign_seal,
            request.publication_identity,
        ),
        expected_stage_global_seal_sha256=request.expected_stage_global_seal_sha256,
        expected_prepare_campaign_seal_sha256=(request.expected_prepare_campaign_seal_sha256),
        expected_reveal_campaign_seal_sha256=(request.expected_reveal_campaign_seal_sha256),
        expected_update_campaign_seal_sha256=(request.expected_update_campaign_seal_sha256),
    )
    return PhasePublicationAttestation.from_seal(
        worker_role=OUTER_SELECT_BARRIER_WORKER_ROLE,
        seal=campaign.seal,
    ).canonical_bytes()


def _run_finalize(request_payload: bytes, output: Path) -> bytes:
    from amp_challenge.evaluation.sequential_v2_finalize_supervisor import (
        run_finalize_worker_request,
    )

    return run_finalize_worker_request(request_payload, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m amp_challenge.evaluation.sequential_v2_worker",
        description="Run one path-isolated sequential-v2 custodian.",
    )
    parser.add_argument(
        "role",
        choices=(
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
        ),
    )
    parser.add_argument("--result-fd", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute exactly one worker role and emit only on its private result FD."""

    arguments = _parser().parse_args(argv)
    if (
        type(arguments.result_fd) is not str
        or not arguments.result_fd.isascii()
        or not arguments.result_fd.isdecimal()
    ):
        raise ValueError("worker result descriptor must be canonical decimal")
    result_descriptor = int(arguments.result_fd)
    if result_descriptor < 3:
        raise ValueError("worker result descriptor is unsafe")
    os.set_inheritable(result_descriptor, False)
    checkpoint = "descriptor-inventory"
    try:
        _assert_initial_descriptor_inventory(result_descriptor)
        checkpoint = "private-courier"
        os.umask(0o077)
        current = _private_current_directory()
        checkpoint = "request-authentication"
        request = _read_and_remove_request(current)
        output = current / _OUTPUT_NAME
        checkpoint = arguments.role
        if arguments.role == PROTOCOL_WORKER_ROLE:
            result = _run_protocol(request, output)
        elif arguments.role == PREPARE_WORKER_ROLE:
            result = _run_prepare(request, output)
        elif arguments.role == PREPARE_BARRIER_WORKER_ROLE:
            result = _run_prepare_barrier(request, output)
        elif arguments.role == SELECT_PREDICTION_WORKER_ROLE:
            result = _run_select_prediction(request, output)
        elif arguments.role == SELECT_RANDOM_WORKER_ROLE:
            result = _run_select_random(request, output)
        elif arguments.role == SELECT_CEILING_WORKER_ROLE:
            result = _run_select_ceiling(request, output)
        elif arguments.role == SELECT_ROTATION_WORKER_ROLE:
            result = _run_select_rotation(request, output)
        elif arguments.role == SELECT_BARRIER_WORKER_ROLE:
            result = _run_select_barrier(request, output)
        elif arguments.role == REVEAL_NO_QUERY_WORKER_ROLE:
            result = _run_reveal_no_query(request, output)
        elif arguments.role == REVEAL_NONEMPTY_WORKER_ROLE:
            result = _run_reveal_nonempty(request, output)
        elif arguments.role == REVEAL_BARRIER_WORKER_ROLE:
            result = _run_reveal_barrier(request, output)
        elif arguments.role == UPDATE_STATE_WORKER_ROLE:
            result = _run_update_state(request, output)
        elif arguments.role == UPDATE_COMPONENT_WORKER_ROLE:
            result = _run_update_component(request, output)
        elif arguments.role == UPDATE_PROJECTION_WORKER_ROLE:
            result = _run_update_projection(request, output)
        elif arguments.role == UPDATE_BARRIER_WORKER_ROLE:
            result = _run_update_barrier(request, output)
        elif arguments.role == OUTER_SELECT_TRACK_WORKER_ROLE:
            result = _run_outer_select_track(request, output)
        elif arguments.role == OUTER_SELECT_BARRIER_WORKER_ROLE:
            result = _run_outer_select_barrier(request, output)
        elif arguments.role == FINALIZE_WORKER_ROLE:
            result = _run_finalize(request, output)
        else:
            raise AssertionError("unreachable sequential-v2 worker role")
        checkpoint = "result-publication"
        _write_result(result_descriptor, result)
        return 0
    except BaseException as error:
        from amp_challenge.evaluation.sequential_v2_seals import canonical_json_bytes

        failure = canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact": _FAILURE_ARTIFACT,
                "worker_role": arguments.role,
                "checkpoint": checkpoint,
                "error_type": type(error).__name__,
                "error_code": (
                    error.safe_code if type(error) is _BoundaryViolation else "worker-exception"
                ),
            }
        )
        try:
            _write_result(result_descriptor, failure)
        except BaseException:
            with suppress(OSError):
                os.close(result_descriptor)
        return 70


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())

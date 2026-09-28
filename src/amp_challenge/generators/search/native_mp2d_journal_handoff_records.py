"""Bounded MP2D handoff bytes and publication, never dispatch authority.

Copied native payloads retain their encoding; copied journal events have no new
fields. A completed phase is historical evidence, not a live permission token.
"""

from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseBuilder,
    PhaseSeal,
    canonical_json_bytes,
    checksum_manifest_bytes,
    sha256_bytes,
    verify_phase,
)
from amp_challenge.generators.diffusion.native_mp2d_driver import (
    NativeMP2DPreparation,
    NativeMP2DSeats,
)
from amp_challenge.generators.search import durable_dispatch_journal_records as journal_records
from amp_challenge.generators.search.campaign_ledger import OracleQueryIdentity
from amp_challenge.representations.run_feature_cache_records import canonical as native_canonical
from amp_challenge.representations.run_feature_cache_verify import inspect_feature_tree

MAX_NATIVE_BYTES = 128 * 1024**2
MAX_PHASE_BYTES = 512 * 1024**2
MAX_HANDOFF_BYTES = 512 * 1024**2
MAX_RUN_BYTES = 5 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
FAILURE_RESERVE_BYTES = 4096
INTENT_ARTIFACT = "native_mp2d_journal_handoff_intent_v1"
COMPLETION_ARTIFACT = "native_mp2d_journal_handoff_completion_v1"
FLAGS = {
    "scientific_evidence_accepted": False,
    "production_eligible": False,
    "campaign_eligible": False,
}
require = journal_records.require


@dataclass(frozen=True, slots=True)
class MP2DJournalHandoffResult:
    round_index: int
    status: str
    preparation_sha256: str
    seats_sha256: str | None
    previous_checkpoint: journal_records.JournalCheckpoint
    checkpoint: journal_records.JournalCheckpoint | None
    intent_seal: PhaseSeal | None
    completion_seal: PhaseSeal | None
    record_payload: bytes

    @property
    def sha256(self):
        return sha256_bytes(b"amp/mp2d-journal-handoff/result/v1\0" + self.record_payload)


def encode_document(document):
    require(type(document) is dict, "handoff document must be an ordinary dictionary")
    require(
        all(document.get(name) is False for name in FLAGS),
        "handoff metadata cannot confer eligibility",
    )
    payload = canonical_json_bytes(document)
    require(len(payload) <= MAX_NATIVE_BYTES, "handoff document exceeds its byte admission")
    return payload


def native_document(result, exact_type):
    require(
        exact_type in (NativeMP2DPreparation, NativeMP2DSeats) and type(result) is exact_type,
        "handoff native result exact type differs",
    )
    payload = result.record_payload
    require(
        type(payload) is bytes and 0 < len(payload) <= MAX_NATIVE_BYTES,
        "handoff native payload byte admission differs",
    )
    document = json.loads(payload)
    require(
        type(document) is dict and native_canonical(document) == payload,
        "handoff native payload is not canonical",
    )
    require(
        all(document.get(name) is False for name in FLAGS),
        "handoff native payload eligibility differs",
    )
    return document


def planned_wave(binding, checkpoint, round_index, sequences, query_ids, replicate_ids):
    """Exact 14+2 wire construction, without history/authentication or I/O."""
    require(type(binding) is journal_records.JournalBinding, "handoff journal binding differs")
    require(
        type(checkpoint) is journal_records.JournalCheckpoint,
        "handoff predecessor checkpoint differs",
    )
    binding.__post_init__()
    checkpoint.__post_init__()
    require(
        type(round_index) is int
        and 1 <= round_index <= 28
        and checkpoint.genesis_sha256 == binding.sha256
        and checkpoint.charged_count == 64 + 16 * (round_index - 1),
        "handoff round/charge/genesis differs",
    )
    require(
        all(
            type(value) is tuple and len(value) == 14
            for value in (sequences, query_ids, replicate_ids)
        )
        and len(set(sequences)) == len(set(query_ids)) == 14
        and all(journal_records.identifier(value) for value in query_ids)
        and all(type(value) is int and value >= 0 for value in replicate_ids),
        "handoff needs fourteen unique ordered requests and supplied replicate IDs",
    )
    template = binding.initial_requests[0].identity
    requests = tuple(
        journal_records.JournalRequest(
            query_id,
            sequence,
            OracleQueryIdentity(
                sha256_bytes(sequence.encode("ascii")),
                template.oracle_contract_sha256,
                template.evaluator_sha256,
                template.checkpoint_sha256,
                template.endpoint_context_sha256,
                template.transform_sha256,
                replicate_id,
            ),
        )
        for sequence, query_id, replicate_id in zip(
            sequences, query_ids, replicate_ids, strict=True
        )
    )
    for request in requests:
        binding.validate_request(request)
    event = journal_records.event_record(
        binding.sha256,
        checkpoint.event_count,
        checkpoint.head_sha256,
        "wave_seal",
        {
            "wave_index": round_index,
            "requests": [row.document() for row in (*requests, *binding.reserves[round_index - 1])],
        },
    )
    following = journal_records.JournalCheckpoint(
        binding.sha256, checkpoint.event_count + 1, event.sha256, checkpoint.charged_count
    )
    return requests, event, following


def directory(path):
    require(isinstance(path, Path), "handoff root must be a Path")
    require(path.is_absolute() and path == path.resolve(strict=True), "handoff root is aliased")
    for ancestor in (path, *path.parents):
        require(stat.S_ISDIR(ancestor.lstat().st_mode), "handoff root crosses a non-directory")
    return path


def _admit_tree(root, maximum):
    """Quota-only admission, never content authentication or a retained inventory."""
    directory(root)
    total, metadata_bytes, pending = 0, 2, [root]
    while pending:
        parent = pending.pop()
        require(stat.S_ISDIR(parent.lstat().st_mode), "handoff traversal directory changed")
        with os.scandir(parent) as children:
            for child in children:
                metadata = child.stat(follow_symlinks=False)
                path = Path(child.path)
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    record = {"path": relative, "type": "directory"}
                    pending.append(path)
                else:
                    require(
                        stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                        "handoff evidence contains a link or nonregular file",
                    )
                    total += metadata.st_size
                    require(total <= maximum, "handoff evidence exceeds byte admission")
                    record = {
                        "path": relative,
                        "type": "file",
                        "bytes": metadata.st_size,
                        "sha256": "0" * 64,
                    }
                # Exactly the existing cache inventory's ASCII-escaped JSON+LF
                # accounting. Digest spelling affects length, not authentication.
                metadata_bytes += len(native_canonical(record))
                require(
                    metadata_bytes <= MAX_METADATA_BYTES,
                    "handoff inventory metadata exceeds byte admission",
                )
    return total


def quota_bytes(handoff_root, run_root):
    """Fresh safe size/representation admission without reading file contents."""
    require(handoff_root.parent == run_root, "handoff root is not a run-root child")
    return _admit_tree(handoff_root, MAX_HANDOFF_BYTES), _admit_tree(run_root, MAX_RUN_BYTES)


def inventory(handoff_root, run_root):
    quota_bytes(handoff_root, run_root)
    own = inspect_feature_tree(handoff_root)
    shared = inspect_feature_tree(run_root)
    require(
        own["regular_bytes"] <= MAX_HANDOFF_BYTES and shared["regular_bytes"] <= MAX_RUN_BYTES,
        "handoff/shared output budget exceeded",
    )
    return own, shared


def _phase_bytes(artifact, payloads, predecessors, metadata):
    receipt = canonical_json_bytes(
        {
            "artifact": artifact,
            "metadata": metadata,
            "payloads": {name: sha256_bytes(payload) for name, payload in payloads.items()},
            "predecessor_seals": predecessors,
            "schema_version": 1,
            "status": "sealed",
        }
    )
    marker = checksum_manifest_bytes(
        {
            **{name: sha256_bytes(payload) for name, payload in payloads.items()},
            "receipt.json": sha256_bytes(receipt),
        }
    )
    return sum(map(len, payloads.values())) + len(receipt) + len(marker)


def _read_phase(destination, expected_seal, handoff_root):
    """Exact phase readback shared by the single-phase and final-batch paths."""
    require(type(expected_seal) is PhaseSeal, "handoff expected phase type differs")
    require(destination.parent == handoff_root, "handoff phase escaped its fixed root")
    _admit_tree(destination, MAX_PHASE_BYTES)
    result = verify_phase(
        destination,
        expected_artifact=expected_seal.artifact,
        expected_payload_paths=tuple(name for name, _ in expected_seal.payload_sha256),
        expected_predecessor_seals=dict(expected_seal.predecessor_seals),
        expected_seal_sha256=expected_seal.seal_sha256,
    )
    require(result == expected_seal, "handoff phase bytes changed")
    return result


def readback_phase(destination, *, expected_seal, handoff_root, run_root, deadline):
    """Callback-free structural readback; never numerical or oracle replay."""
    result = _read_phase(destination, expected_seal, handoff_root)
    inventory(handoff_root, run_root)
    if time.monotonic() >= deadline:
        raise TimeoutError("handoff original deadline expired during phase readback")
    return result


def readback_phases(phases, *, handoff_root, run_root, deadline):
    """Reverify every held phase, then inventory the shared roots once, without callbacks."""
    require(type(phases) is tuple and phases, "handoff final phase batch is empty or mutable")
    require(len({path for path, _ in phases}) == len(phases), "handoff phase batch repeats a path")
    for destination, expected_seal in phases:
        _read_phase(destination, expected_seal, handoff_root)
        if time.monotonic() >= deadline:
            raise TimeoutError("handoff original deadline expired during phase readback")
    inventory(handoff_root, run_root)
    if time.monotonic() >= deadline:
        raise TimeoutError("handoff original deadline expired during phase readback")


def _publish(
    destination,
    *,
    artifact,
    payloads,
    predecessors,
    metadata,
    handoff_root,
    run_root,
    checkpoint,
    deadline,
):
    require(destination.parent == handoff_root, "handoff publication escaped its fixed root")
    require(
        all(type(raw) is bytes and 0 < len(raw) <= MAX_NATIVE_BYTES for raw in payloads.values()),
        "handoff publication payload admission differs",
    )
    require(
        all(metadata.get(name) is False for name in FLAGS),
        "handoff publication eligibility differs",
    )
    size = _phase_bytes(artifact, payloads, predecessors, metadata)
    require(size <= MAX_PHASE_BYTES, "handoff phase byte admission exhausted")
    builder = PhaseBuilder(
        destination, artifact=artifact, predecessor_seals=predecessors, metadata=metadata
    )
    written = 0

    def admission(stage):
        checkpoint(stage)
        # Re-admit after the last supplied callback, immediately before copying.
        # Already written payloads are counted by inventory, not counted twice.
        own, shared = quota_bytes(handoff_root, run_root)
        remaining = 2 * size - written
        require(
            own + remaining <= MAX_HANDOFF_BYTES - FAILURE_RESERVE_BYTES
            and shared + remaining <= MAX_RUN_BYTES - FAILURE_RESERVE_BYTES,
            "handoff publication/alias byte admission exhausted",
        )
        if builder._staging is not None:
            directory(builder._staging)
            metadata = builder._staging.stat()
            require(
                (metadata.st_dev, metadata.st_ino) == builder._staging_identity,
                "handoff staging identity changed",
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("handoff original deadline expired before publication work")

    try:
        admission("before_phase_publication")
        builder.__enter__()
        for name, payload in payloads.items():
            admission("before_phase_payload_" + name)
            builder.write_bytes(name, payload)
            written += len(payload)
            checkpoint("after_phase_payload_" + name)
        admission("before_phase_commit")
        result = builder.publish(expected_payload_paths=tuple(payloads))
        checkpoint("after_phase_publication")
        return readback_phase(
            destination,
            expected_seal=result,
            handoff_root=handoff_root,
            run_root=run_root,
            deadline=deadline,
        )
    except BaseException as error:
        # No __exit__: preserve owned staging/aliases which still exist. The
        # unchanged writer may already have removed its own incomplete payload.
        staging = builder._staging
        error.add_note(f"handoff destination: {destination}; known staging: {staging}")
        raise


def publish_intent(
    destination,
    *,
    preparation,
    seats,
    planned_event,
    document,
    predecessors,
    handoff_root,
    run_root,
    checkpoint,
    deadline,
):
    native_document(preparation, NativeMP2DPreparation)
    native_document(seats, NativeMP2DSeats)
    require(
        type(planned_event) is journal_records.JournalEvent, "handoff planned event type differs"
    )
    planned_event.__post_init__()
    return _publish(
        destination,
        artifact=INTENT_ARTIFACT,
        payloads={
            "preparation.json": preparation.record_payload,
            "seats.json": seats.record_payload,
            "planned_event.json": planned_event.payload,
            "handoff.json": encode_document(document),
        },
        predecessors=predecessors,
        metadata={"round_index": preparation.round_index, **FLAGS},
        handoff_root=handoff_root,
        run_root=run_root,
        checkpoint=checkpoint,
        deadline=deadline,
    )


def publish_completion(
    destination,
    *,
    document,
    intent_seal_sha256,
    journal_phase_sha256,
    handoff_root,
    run_root,
    checkpoint,
    deadline,
):
    return _publish(
        destination,
        artifact=COMPLETION_ARTIFACT,
        payloads={"completion.json": encode_document(document)},
        predecessors={
            "handoff_intent": intent_seal_sha256,
            "journal_wave_phase": journal_phase_sha256,
        },
        metadata={"round_index": document["round_index"], **FLAGS},
        handoff_root=handoff_root,
        run_root=run_root,
        checkpoint=checkpoint,
        deadline=deadline,
    )

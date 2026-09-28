"""Independent saved-artifact/cache reconstruction, without producer or model imports.

Authored separately from the bridge. Existing safe peptide-array derivation is
shared explicitly; this is not an independent ESM implementation, external
timing authority, scientific result or permission to adopt historical features.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import Path

import numpy as np

from amp_challenge.representations.candidate_features import CandidateFeatureRequest
from amp_challenge.representations.fixed_shape_esm import FIXED_LAYOUT, FIXED_LAYOUT_SHA256
from amp_challenge.representations.peptide_esm import (
    ARRAY_NAMES,
    CONFIG,
    canonical_json,
    digest,
    file_digest,
    load_features,
    read_sequences,
)
from amp_challenge.representations.run_feature_cache_records import (
    ALIASES,
    CONTRACT_PATH,
    CONTRACT_SHA256,
    FULL_ARMS,
    FeatureAssemblyBinding,
    FeatureCacheReceipt,
    FeatureCounters,
    FeatureIntent,
    FeatureRunBinding,
    PrivateFeatureRelease,
    check_tr2_capacity,
    check_tr2_capacity_request,
    legacy_history_document,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SESSION_BYTES = 2 * 1024**3
_RUN_BYTES = 5 * 1024**3
_JSON_BYTES = 1024**2
_OPERATION_BYTES = 16 * 1024**2


def _hash(value: object) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError("expected an explicit SHA256")
    return value


def _number(value: object, *, minimum: float = 0.0, maximum: float = math.inf) -> float:
    if type(value) not in (int, float):
        raise ValueError("clock/count is not a built-in finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("clock/count overflow") from error
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError("clock/count outside its declared bound")
    return result


def _count(value: object, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid exact integer count")
    return value


def _read(path: Path, maximum: int, expected_sha256: str | None = None) -> bytes:
    """Bound reads before allocation and reject links/special files.

    Same-account cooperative reconstruction is the scope; this is not a claim
    to lock a concurrently hostile filesystem or prevent every rename race.
    """
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
        raise ValueError("artifact is linked, nonregular or oversized")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("artifact changed before read")
        payload = stream.read(maximum + 1)
    after = path.lstat()
    if len(payload) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("artifact changed during read")
    if expected_sha256 is not None and digest(payload) != _hash(expected_sha256):
        raise ValueError("externally expected artifact digest differs")
    return payload


def _document(path: Path, expected_sha256: str | None = None) -> tuple[dict, bytes]:
    payload = _read(path, _JSON_BYTES, expected_sha256)
    value = json.loads(payload)
    if type(value) is not dict or canonical_json(value) != payload:
        raise ValueError("artifact is not a canonical JSON object")
    return value, payload


def _stream_hash(path: Path, maximum: int, expected_sha256: str | None = None) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
        raise ValueError("streamed artifact is linked, nonregular or oversized")
    result = file_digest(path)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or (expected_sha256 is not None and result != _hash(expected_sha256)):
        raise ValueError("streamed artifact changed or its pin differs")
    return result


def _exact_entries(root: Path, expected: set[str]) -> None:
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("expected an unlinked artifact directory")
    seen = set()
    with os.scandir(root) as iterator:
        for entry in iterator:
            if entry.name not in expected or entry.name in seen:
                raise ValueError("artifact directory has an unexpected entry")
            seen.add(entry.name)
    if seen != expected:
        raise ValueError("artifact directory is missing an expected entry")


def inspect_feature_tree(root: Path, *, session: bool = False) -> dict:
    """Read-only regular-file inventory; no new run-wide inode limit is imposed."""
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("artifact root is not a real directory")
    entries, total, metadata_bytes = [], 0, 2
    pending = [root]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as iterator:
            for entry in iterator:
                path = Path(entry.path)
                metadata = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    record = {"path": relative, "type": "directory"}
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    total += metadata.st_size
                    if total > (_SESSION_BYTES if session else _RUN_BYTES):
                        raise ValueError("retained regular bytes exceed the declared cap")
                    record = {
                        "path": relative,
                        "type": "file",
                        "bytes": metadata.st_size,
                        "sha256": _stream_hash(path, _SESSION_BYTES if session else _RUN_BYTES),
                    }
                else:
                    raise ValueError("artifact tree contains a link or nonregular entry")
                metadata_bytes += len(canonical_json(record))
                if metadata_bytes > _OPERATION_BYTES:
                    raise ValueError(
                        "inventory metadata is unsupported by the prospective representation bound"
                    )
                entries.append(record)
                if session and len(entries) > 10000:
                    raise ValueError("session descendant-entry cap exceeded")
    entries.sort(key=lambda entry: entry["path"])
    return {"entries": entries, "regular_bytes": total, "descendant_entries": len(entries)}


def _session_record(root: Path, expected: dict) -> tuple[dict, str]:
    session, payload = _document(root / "session.json")
    if canonical_json(session) != canonical_json(expected):
        raise ValueError("saved session differs from the externally supplied record")
    profile = check_tr2_capacity(session.get("tr2_grouped_capacity"))
    if set(session) - ({"tr2_grouped_capacity"} if profile else set()) != {
        "artifact",
        "run_id",
        "session_id",
        "source",
        "job_id",
        "maximum_requests",
        "timeout_seconds",
        "deadline_monotonic",
        "maximum_output_bytes",
        "feature_layout",
        "feature_layout_sha256",
    }:
        raise ValueError("fixed session schema differs")
    if (
        session["artifact"] != "warm_candidate_feature_session_v1"
        or not _count(session["maximum_requests"], 281 if profile else 128) >= 1
        or (profile is not None and session["maximum_requests"] != 281)
        or session["maximum_output_bytes"] != _SESSION_BYTES
        or canonical_json(session["feature_layout"]) != canonical_json(FIXED_LAYOUT)
        or session["feature_layout_sha256"] != FIXED_LAYOUT_SHA256
        or type(session["job_id"]) is not str
        or not session["job_id"].isascii()
        or not session["job_id"].isdigit()
    ):
        raise ValueError("fixed session identity/resource/layout mismatch")
    _number(session["timeout_seconds"], minimum=0.0, maximum=7200.0)
    if session["timeout_seconds"] <= 0:
        raise ValueError("session timeout must be positive")
    _number(session["deadline_monotonic"])
    CandidateFeatureRequest(session["run_id"], session["session_id"], ("ACDEFGHI",))
    ready, _ = _document(root / "ready.json")
    if ready != {"artifact": "warm_candidate_feature_ready_v1", "session_sha256": digest(payload)}:
        raise ValueError("saved startup binding differs")
    return session, digest(payload)


def load_saved_feature_batch(
    session_root: Path,
    *,
    expected_session: dict,
    ordinal: int,
    expected_previous_head: str,
    expected_response_sha256: str,
    expected_runtime: dict,
    expected_model_sha256: str,
) -> tuple[CandidateFeatureRequest, dict, dict, dict]:
    """Check one saved physical batch and return its original safe arrays.

    The caller authenticates the session/runtime/response pins. This function
    never requests features or opens a live session, and does not read a clock.
    A completed batch can be retained within an ultimately failed session; only
    the ledger/session verifier decides whether a cache transition was accepted.
    """
    profile = check_tr2_capacity(expected_session.get("tr2_grouped_capacity"))
    ordinal = _count(ordinal, 280 if profile else 127)
    original_session, original_runtime = expected_session, expected_runtime
    session_expectation = canonical_json(expected_session)
    runtime_expectation = canonical_json(expected_runtime)
    expected_session = json.loads(session_expectation)
    expected_runtime = json.loads(runtime_expectation)
    _hash(expected_previous_head)
    _hash(expected_response_sha256)
    _hash(expected_model_sha256)
    session, session_sha = _session_record(session_root, expected_session)
    if ordinal >= session["maximum_requests"]:
        raise ValueError("batch ordinal exceeds original session quota")
    batch = session_root / f"batch-{ordinal:04d}"
    _exact_entries(
        batch,
        {
            "request.json",
            "command.json",
            "response.json",
            "invocation.json",
            "features",
        },
    )
    request_bytes = _read(batch / "request.json", 64 * 1024)
    request_sha = digest(request_bytes)
    request = CandidateFeatureRequest.from_bytes(request_bytes, request_sha)
    check_tr2_capacity_request(profile, ordinal, request.sequences)
    if request.run_id != session["run_id"]:
        raise ValueError("request is from a different run")
    command = {
        "operation": "features",
        "ordinal": ordinal,
        "previous_sha256": expected_previous_head,
        "request_sha256": request_sha,
    }
    command_bytes = _read(batch / "command.json", 8192)
    if command_bytes != canonical_json(command):
        raise ValueError("saved command predecessor/ordinal/request differs")
    response, response_bytes = _document(batch / "response.json", expected_response_sha256)
    manifest_sha = _hash(response.get("manifest_sha256"))
    if response_bytes != canonical_json({**command, "manifest_sha256": manifest_sha}):
        raise ValueError("saved response does not bind the exact command")
    output = batch / "features"
    if not stat.S_ISDIR(output.lstat().st_mode):
        raise ValueError("feature output is not a real directory")
    inventory = {f"{name}.npy" for name in ARRAY_NAMES} | {
        "sequences.jsonl",
        "manifest.json",
        "COMPLETE",
    }
    _exact_entries(output, inventory)
    total = 0
    for name in inventory:
        metadata = (output / name).lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("feature payload is linked or nonregular")
        total += metadata.st_size
    if total > CONFIG["maximum_output_bytes"]:
        raise ValueError("saved feature payload exceeds its unchanged array cap")
    declared, manifest_bytes = _document(output / "manifest.json", manifest_sha)
    sequence_payload = _read(output / "sequences.jsonl", 64 * 1024)
    if sequence_payload != request.sequence_payload:
        raise ValueError("saved feature sequence identities/order differ")
    # Shape and row bounds are checked BEFORE shared np.load or a float copy.
    lengths = [len(sequence) for sequence in request.sequences]
    n, residues, contacts = len(lengths), sum(lengths), sum(v * v for v in lengths)
    shapes = {
        "lengths": ([n], "<i8"),
        "token_offsets": ([n + 1], "<i8"),
        "contact_offsets": ([n + 1], "<i8"),
        "residue_tokens": ([residues, 320], "<f4"),
        "contacts": ([contacts], "<f4"),
        "esm_mean": ([n, 320], "<f4"),
        "esm_length": ([n, 321], "<f4"),
        "spectral": ([n, 32], "<f8"),
        "esm_length_spectral": ([n, 353], "<f8"),
    }
    if set(declared.get("arrays", {})) != {f"{name}.npy" for name in shapes}:
        raise ValueError("saved array manifest inventory differs")
    for name, (shape, dtype) in shapes.items():
        pin = declared["arrays"][f"{name}.npy"]
        byte_count = _count(pin.get("bytes"), CONFIG["maximum_output_bytes"])
        if pin.get("shape") != shape or pin.get("dtype") != dtype:
            raise ValueError("saved array declared shape/type is not a bounded peptide array")
        data = _read(output / f"{name}.npy", byte_count, _hash(pin.get("sha256")))
        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            header_shape, fortran, header_dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            header_shape, fortran, header_dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unexpected bounded NPY version")
        if (
            list(header_shape) != shape
            or fortran
            or header_dtype.str != dtype
            or (len(data) - stream.tell() != math.prod(shape) * header_dtype.itemsize)
        ):
            raise ValueError("actual NPY header differs from bounded array shape/type")
    rows, arrays, manifest = load_features(output, manifest_sha)
    expected_identity = {
        "candidate_request_sha256": request_sha,
        "run_id": request.run_id,
        "batch_id": request.batch_id,
        "source": session["source"],
        "job_id": session["job_id"],
        "runtime": expected_runtime,
        "frozen_model_sha256": expected_model_sha256,
        "feature_layout": FIXED_LAYOUT,
        "feature_layout_sha256": FIXED_LAYOUT_SHA256,
        "warm_session": {
            "session_sha256": session_sha,
            "ordinal": ordinal,
            "previous_sha256": expected_previous_head,
        },
        "input_scope": "generated_peptides_label_free_not_namespace_membership_or_oracle_truth",
    }
    if rows != read_sequences(request.sequence_payload) or any(
        canonical_json(manifest["identity"].get(key)) != canonical_json(value)
        for key, value in expected_identity.items()
    ):
        raise ValueError("saved feature source/runtime/model/session identity differs")
    invocation, invocation_bytes = _document(batch / "invocation.json")
    elapsed = _number(invocation.get("elapsed_seconds"), maximum=120.0)
    expected_invocation = {
        "artifact": "warm_candidate_feature_invocation_v1",
        "response_sha256": expected_response_sha256,
        "elapsed_seconds": invocation["elapsed_seconds"],
        "rows": len(rows),
        "oracle_calls": 0,
        "production_input_eligible": False,
        "scientific_evidence_accepted": False,
        "timing_scope": "local_measurement_not_independent_external_timing_authority",
    }
    if canonical_json(expected_invocation) != invocation_bytes or (
        type(invocation.get("rows")) is not int
        or type(invocation.get("oracle_calls")) is not int
        or invocation["production_input_eligible"] is not False
        or invocation["scientific_evidence_accepted"] is not False
    ):
        raise ValueError("saved invocation schema/authority/timing differs")
    if (
        _read(output / "manifest.json", _JSON_BYTES) != manifest_bytes
        or _read(batch / "response.json", _JSON_BYTES) != response_bytes
        or _read(batch / "request.json", 64 * 1024) != request_bytes
    ):
        raise ValueError("saved batch changed during reconstruction")
    if (
        canonical_json(original_session) != session_expectation
        or canonical_json(original_runtime) != runtime_expectation
    ):
        raise ValueError("caller session/runtime expectations changed during reconstruction")
    for value in arrays.values():
        value.flags.writeable = False
    summary = {
        "batch_relative_root": f"batch-{ordinal:04d}",
        "ordinal": ordinal,
        "row_count": len(rows),
        "sequence_ids": [row["sequence_id"] for row in rows],
        "session_sha256": session_sha,
        "request_sha256": request_sha,
        "command_sha256": digest(command_bytes),
        "response_sha256": expected_response_sha256,
        "invocation_sha256": digest(invocation_bytes),
        "manifest_sha256": manifest_sha,
        "previous_response_head": expected_previous_head,
        "elapsed_seconds": elapsed,
    }
    return request, arrays, manifest, summary


def verify_saved_feature_session(
    session_root: Path,
    *,
    expected_session: dict,
    expected_complete_sha256: str,
    expected_runtime: dict,
    expected_model_sha256: str,
    expected_batches: tuple[dict, ...],
) -> dict:
    """Stream a pinned closed session, retaining hashes/order but no batch arrays."""
    original_expectations = (expected_session, expected_runtime, expected_batches)
    expectation_bytes = canonical_json(original_expectations)
    frozen = json.loads(expectation_bytes)
    expected_session, expected_runtime, expected_batches = frozen[0], frozen[1], tuple(frozen[2])
    complete, complete_bytes = _document(session_root / "COMPLETE.json", expected_complete_sha256)
    if (session_root / "FAILED.json").exists():
        raise ValueError("COMPLETE does not override retained FAILED evidence")
    session, session_sha = _session_record(session_root, expected_session)
    n = _count(complete.get("completed_requests"), session["maximum_requests"])
    if type(expected_batches) is not tuple or len(expected_batches) != n:
        raise ValueError("external saved-batch inventory differs")
    if set(complete) != {
        "artifact",
        "session_sha256",
        "completed_requests",
        "invocation_sha256s",
        "previous_sha256",
        "startup_seconds",
        "elapsed_seconds",
        "retained_bytes_before_receipt",
        "stderr_sha256",
        "oracle_calls",
        "production_input_eligible",
        "scientific_evidence_accepted",
    } or (
        complete["artifact"] != "warm_candidate_feature_session_complete_v1"
        or complete["session_sha256"] != session_sha
        or type(complete["oracle_calls"]) is not int
        or complete["oracle_calls"] != 0
        or complete["production_input_eligible"] is not False
        or complete["scientific_evidence_accepted"] is not False
        or type(complete["invocation_sha256s"]) is not list
        or len(complete["invocation_sha256s"]) != n
    ):
        raise ValueError("saved COMPLETE schema/authority/chain differs")
    elapsed = _number(complete["elapsed_seconds"], maximum=session["timeout_seconds"])
    startup = _number(complete["startup_seconds"], maximum=min(120.0, elapsed))
    if elapsed >= session["timeout_seconds"]:
        raise ValueError("recorded closed session exceeds original allowance")
    _stream_hash(session_root / "stderr.log", _SESSION_BYTES, complete["stderr_sha256"])
    _exact_entries(
        session_root,
        {
            "session.json",
            "ready.json",
            "stderr.log",
            "official-source",
            "COMPLETE.json",
        }
        | {f"batch-{i:04d}" for i in range(n)},
    )
    before = inspect_feature_tree(session_root, session=True)
    source_files = expected_runtime.get("esm_source_sha256")
    if type(source_files) is not dict or digest(
        canonical_json(source_files)
    ) != expected_runtime.get("esm_source_inventory_sha256"):
        raise ValueError("saved official source inventory is unbound")
    prefix = "official-source/esm/"
    found = {
        row["path"][len(prefix) :]: row["sha256"]
        for row in before["entries"]
        if row["type"] == "file" and row["path"].startswith(prefix) and row["path"].endswith(".py")
    }
    if found != source_files:
        raise ValueError("saved official model code differs from the pinned runtime")
    if before["regular_bytes"] - len(complete_bytes) != _count(
        complete["retained_bytes_before_receipt"], _SESSION_BYTES
    ):
        raise ValueError("session final marker is absent from retained-byte reconstruction")
    head, summaries, batch_ids = session_sha, [], set()
    for ordinal, expected in enumerate(expected_batches):
        response, response_bytes = _document(
            session_root / f"batch-{ordinal:04d}" / "response.json"
        )
        if response.get("manifest_sha256") != expected.get("manifest_sha256"):
            raise ValueError("saved manifest differs from external historical receipt")
        request, arrays, manifest, summary = load_saved_feature_batch(
            session_root,
            expected_session=session,
            ordinal=ordinal,
            expected_previous_head=head,
            expected_response_sha256=digest(response_bytes),
            expected_runtime=expected_runtime,
            expected_model_sha256=expected_model_sha256,
        )
        if request.batch_id in batch_ids:
            raise ValueError("repeated physical batch identity")
        batch_ids.add(request.batch_id)
        if summary["invocation_sha256"] != complete["invocation_sha256s"][ordinal] or any(
            summary[key] != expected[key] for key in ("manifest_sha256", "invocation_sha256")
        ):
            raise ValueError("saved invocation lacks its original COMPLETE/external pin")
        if (
            expected.get("rows") != len(request.sequences)
            or expected.get("elapsed_seconds") != summary["elapsed_seconds"]
            or (
                expected.get("node") != manifest["identity"]["node"]
                or expected.get("frozen_model_sha256") != expected_model_sha256
            )
        ):
            raise ValueError("historical batch job/model/timing/rows differ")
        summary["sequences"] = list(request.sequences)
        summary["array_sha256s"] = {
            name: digest(arrays[name].tobytes(order="C")) for name in ARRAY_NAMES
        }
        summary["array_shapes"] = {name: list(arrays[name].shape) for name in ARRAY_NAMES}
        summaries.append(summary)
        head = summary["response_sha256"]
        del arrays, manifest
    if (
        head != complete["previous_sha256"]
        or inspect_feature_tree(session_root, session=True) != before
        or canonical_json(original_expectations) != expectation_bytes
    ):
        raise ValueError("session chain or saved artifacts changed during audit")
    return {
        "session_sha256": session_sha,
        "complete_sha256": expected_complete_sha256,
        "job_id": session["job_id"],
        "historical_source": session["source"],
        "startup_seconds": startup,
        "elapsed_seconds": elapsed,
        "regular_bytes_including_complete": before["regular_bytes"],
        "descendant_entries": before["descendant_entries"],
        "batches": summaries,
        "previous_response_head": head,
    }


def _equal(actual: object, expected: object, message: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(message)


def _raw_sha(row: np.ndarray, width: int) -> str:
    if row.shape != (width,) or row.dtype.kind not in "fiu" or not np.isfinite(row).all():
        raise ValueError("raw row shape/type/finiteness differs")
    return digest(
        b"amp/run-feature-row/v1\0"
        + canonical_json({"dtype": "<f8", "width": width})
        + np.asarray(row, dtype="<f8", order="C").tobytes()
    )


def _acquired_origins(
    request, arrays, operation: int, ordinal: int, manifest_sha: str
) -> list[dict]:
    result = []
    for index, sequence in enumerate(request.sequences):
        origin = {
            "sequence": sequence,
            "sequence_id": digest(sequence.encode("ascii")),
            "acquisition_operation": operation,
            "batch_ordinal": ordinal,
            "row": index,
            "manifest_sha256": manifest_sha,
            "raw_sha256s": {
                name: _raw_sha(arrays[name][index], width) for name, (_, width) in ALIASES.items()
            },
        }
        result.append({**origin, "origin_sha256": digest(canonical_json(origin))})
    return result


def _public_rows(
    sequences: tuple[str, ...], representation: str, origins: list[dict], intent: dict
) -> dict:
    alias, width = ALIASES[representation]
    return {
        "artifact": "run_feature_requested_rows_v1",
        "sequence_ids": [digest(sequence.encode("ascii")) for sequence in sequences],
        "representation": alias,
        "width": width,
        "row_sha256s": [origin["raw_sha256s"][representation] for origin in origins],
        "objective_context_sha256": intent["objective_context_sha256"],
        "history_sha256": intent["history_sha256"],
        "oracle_calls": 0,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }


def _assembly_rows(
    binding: FeatureAssemblyBinding, run: FeatureRunBinding
) -> tuple[list[dict], dict]:
    if type(binding) is not FeatureAssemblyBinding:
        raise ValueError("external assembly binding must have the exact record type")
    document = binding.document()
    history = legacy_history_document(document["raw_history"])
    if set(history) != {
        "run_id",
        "seed",
        "round_index",
        "objective_context_sha256",
        "oracle_bundle_sha256",
        "previous_wave_head_sha256",
        "observations",
        "receipt_sha256",
    }:
        raise ValueError("assembly history schema differs")
    round_index = _count(history["round_index"], 29)
    if (
        round_index < 1
        or history["run_id"] != run.run_id
        or type(history["seed"]) is not int
        or history["seed"] != run.seed
        or history["objective_context_sha256"] != run.objective_context_sha256
    ):
        raise ValueError("assembly run/seed/round/context differs")
    for key in ("oracle_bundle_sha256", "previous_wave_head_sha256", "receipt_sha256"):
        _hash(history[key])
    observations = history["observations"]
    if type(observations) is not list or len(observations) != 64 + 16 * (round_index - 1):
        raise ValueError("assembly charged denominator differs")
    successful, query_ids, sequences = set(), set(), set()
    for index, row in enumerate(observations):
        if (
            type(row) is not dict
            or set(row)
            != {
                "charge_index",
                "query_id",
                "sequence",
                "response_receipt_sha256",
                "status",
                "objectives",
            }
            or type(row["charge_index"]) is not int
            or row["charge_index"] != index
        ):
            raise ValueError("assembly does not retain the exact contiguous charged ledger")
        query = row["query_id"]
        if (
            type(query) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", query) is None
            or query in query_ids
            or row["sequence"] in sequences
        ):
            raise ValueError("assembly query/sequence identity repeats or is invalid")
        CandidateFeatureRequest("assembly-audit", "row", (row["sequence"],))
        _hash(row["response_receipt_sha256"])
        query_ids.add(query)
        sequences.add(row["sequence"])
        if row["status"] == "successful":
            if (
                type(row["objectives"]) is not list
                or len(row["objectives"]) != 2
                or any(
                    type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
                    for value in row["objectives"]
                )
            ):
                raise ValueError("successful charged objective schema differs")
            successful.add(query)
        elif (
            row["status"] not in ("failed", "missing", "censored", "partial", "timed_out")
            or row["objectives"] is not None
        ):
            raise ValueError("unsuccessful charge exposes objectives or changes status")
    eligible = set(binding.eligible_query_ids)
    if not eligible <= successful:
        raise ValueError("external eligible subset contains a non-successful or absent charge")
    return [row for row in observations if row["query_id"] in eligible], history


def _relative_path(root: Path, relative: object) -> Path:
    if (
        type(relative) is not str
        or not relative
        or relative == "."
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or str(Path(relative)) != relative
    ):
        raise ValueError("audit path is not a canonical run-relative path")
    return root / relative


def _source_snapshot(binding: FeatureRunBinding) -> dict:
    repository = Path(binding.repository)
    documents = (
        json.loads(binding.warm_source_payload)["files"],
        json.loads(binding.implementation_source_payload),
    )
    expected = {}
    for mapping in documents:
        for name, pin in mapping.items():
            if name in expected and expected[name] != pin:
                raise ValueError("source inventories disagree")
            expected[name] = pin
    if expected.get(CONTRACT_PATH, CONTRACT_SHA256) != CONTRACT_SHA256:
        raise ValueError("source contract digest differs")
    expected[CONTRACT_PATH] = CONTRACT_SHA256
    for name, pin in expected.items():
        _stream_hash(_relative_path(repository, name), _RUN_BYTES, pin)
    return expected


def _inventory_record(
    value: dict, binding: FeatureRunBinding, *, pending_path=None, pending_bytes=0
) -> dict:
    if type(value) is not dict or set(value) != {
        "entries",
        "regular_bytes",
        "session_bytes",
        "session_entries",
        "pending_final_path",
        "pending_final_bytes",
    }:
        raise ValueError("private inventory schema differs")
    if (
        value["pending_final_path"] != pending_path
        or type(value["pending_final_bytes"]) is not int
        or value["pending_final_bytes"] != pending_bytes
    ):
        raise ValueError("pending self-receipt identity/size differs")
    entries = value["entries"]
    if type(entries) is not list or len(canonical_json(entries)) > _OPERATION_BYTES:
        raise ValueError("inventory metadata representation bound differs")
    by_path, regular, session_bytes, session_entries = {}, 0, 0, 0
    prefix = binding.session_root.relative_to(Path(binding.run_root)).as_posix() + "/"
    for entry in entries:
        if type(entry) is not dict or entry.get("type") not in ("file", "directory"):
            raise ValueError("inventory entry type differs")
        expected_keys = {"path", "type"} | (
            {"bytes", "sha256"} if entry["type"] == "file" else set()
        )
        if set(entry) != expected_keys:
            raise ValueError("inventory entry schema differs")
        path = entry["path"]
        _relative_path(Path(binding.run_root), path)
        if path in by_path or path == pending_path:
            raise ValueError("inventory repeats a path or includes its own self hash")
        by_path[path] = entry
        if path.startswith(prefix):
            session_entries += 1
        if entry["type"] == "file":
            size = _count(entry["bytes"], _RUN_BYTES)
            _hash(entry["sha256"])
            regular += size
            if path.startswith(prefix):
                session_bytes += size
    if list(by_path) != sorted(by_path):
        raise ValueError("inventory ordering differs")
    if pending_path is not None:
        pending = _relative_path(Path(binding.run_root), pending_path)
        if pending.parent != binding.audit_root:
            raise ValueError("pending final receipt is not a direct private audit artifact")
    expected = {
        "regular_bytes": regular,
        "session_bytes": session_bytes,
        "session_entries": session_entries,
    }
    if any(type(value[key]) is not int or value[key] != count for key, count in expected.items()):
        raise ValueError("inventory byte/entry arithmetic differs")
    if (
        regular + pending_bytes > _RUN_BYTES
        or session_bytes > _SESSION_BYTES
        or session_entries > 10000
    ):
        raise ValueError("inventory exceeds original run/session resource caps")
    return by_path


def _inventory_files(root: Path, entries: dict, cache: dict) -> None:
    for name, entry in entries.items():
        path = _relative_path(root, name)
        if entry["type"] == "directory":
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("recorded directory has changed type")
            continue
        key = (name, entry["bytes"], entry["sha256"])
        if key in cache:
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size < entry["bytes"]:
            raise ValueError("retained file was removed, shortened, linked or changed type")
        remaining, hasher = entry["bytes"], hashlib.sha256()
        with path.open("rb") as stream:
            while remaining:
                chunk = stream.read(min(1024**2, remaining))
                if not chunk:
                    raise ValueError("retained file ended before its recorded prefix")
                remaining -= len(chunk)
                hasher.update(chunk)
        if hasher.hexdigest() != entry["sha256"]:
            raise ValueError("recorded file prefix no longer reconstructs")
        cache[key] = True


def _counter_limits(counters: dict, binding: FeatureRunBinding, *, failed: bool) -> None:
    for key, value in counters.items():
        if key != "candidate_wave_counts":
            _count(value, 2**63 - 1)
    waves = counters["candidate_wave_counts"]
    if type(waves) is not list or len(waves) != 28:
        raise ValueError("candidate per-wave inventory differs")
    candidate_run, candidate_wave = binding.candidate_limits
    if (
        any(type(v) is not int or v < 0 for v in waves)
        or sum(waves) != counters["candidate_opportunities"]
    ):
        raise ValueError("candidate wave/run counters do not reconcile")
    if (
        counters["logical_opportunities"]
        != counters["candidate_opportunities"] + counters["auxiliary_opportunities"]
        or counters["physical_attempts"]
        != counters["physical_completed"] + counters["physical_failed"]
    ):
        raise ValueError("logical/physical accounting does not reconcile")
    if (
        counters["padded_rows_dispatched"] != counters["physical_attempts"] * 128
        or not counters["physical_attempts"]
        <= counters["real_rows_dispatched"]
        <= 128 * counters["physical_attempts"]
    ):
        raise ValueError("real/padded dispatch accounting differs")
    if (
        counters["physical_attempts"] > binding.admission_limit
        or counters["acquired_origins"] > binding.origin_limit
    ):
        raise ValueError("physical admission/acquired-origin caps were exceeded")
    # The first valid but exhausted logical opportunity is paid and retained in
    # a failed record. It cannot authorize a physical attempt or another call.
    excess = 1 if failed else 0
    if counters["candidate_opportunities"] > candidate_run + excess or any(
        v > candidate_wave + excess for v in waves
    ):
        raise ValueError("consumer opportunity quota was reset or expanded")
    if binding.arm_id in FULL_ARMS and counters["full_accounted_requests"] > 128 + excess:
        raise ValueError("full-method auxiliary/provider accounting exceeds its bound")


def _timing_record(document: dict, binding: FeatureRunBinding, previous_time: float) -> float:
    timings = document["timings"]
    if type(timings) is not list:
        raise ValueError("operation is missing its original clock observations")
    if not timings:
        if (
            document["status"] == "failed"
            and document["batch"] is None
            and document["staging"]["started"] is None
        ):
            return previous_time
        raise ValueError("paid or accepted operation lacks recorded entry timing")
    phases = (
        "entry",
        "source",
        "session_start",
        "session_ready",
        "dispatch_prepare",
        "before_request",
        "after_request",
        "reconstruct",
        "serialize",
        "published",
        "final",
    )
    deadline = binding.original_deadline
    intent = document["intent"]
    if intent is not None:
        deadline = min(deadline, _number(intent["original_effective_deadline"]))
    start = None
    for timing in timings:
        if (
            type(timing) is not dict
            or set(timing) != {"phase", "monotonic"}
            or timing["phase"] not in phases
        ):
            raise ValueError("operation clock phase schema differs")
        value = _number(timing["monotonic"])
        if value < max(previous_time, binding.original_epoch):
            raise ValueError("operation clock moved before its original epoch/predecessor")
        if start is None:
            if timing["phase"] != "entry":
                raise ValueError("operation clock lacks the entry observation")
            start = value
        previous_time = value
        if document["status"] == "completed" and value >= deadline:
            raise ValueError("completed operation has a post-deadline clock observation")
    return previous_time


def _checked_session_pin(
    document: dict | None, binding: FeatureRunBinding, previous: dict | None, *, failed=False
) -> dict | None:
    if document is None:
        if previous is not None and not failed:
            raise ValueError("operation erased its original session")
        return previous
    if type(document) is not dict or set(document) != {
        "config",
        "config_sha256",
        "ready_sha256",
        "complete_sha256",
        "failed_sha256",
    }:
        raise ValueError("operation session-pin schema differs")
    config = document["config"]
    _equal(
        config.get("tr2_grouped_capacity"),
        binding.tr2_grouped_capacity,
        "session capacity profile differs from binding",
    )
    if config.get("maximum_requests") != binding.admission_limit:
        raise ValueError("session physical admission differs from binding")
    _equal(config.get("source"), json.loads(binding.warm_source_payload), "session source changed")
    if (
        config.get("run_id") != binding.run_id
        or config.get("session_id") != binding.session_id
        or config.get("job_id") != binding.job_id
    ):
        raise ValueError("session run/job identity differs")
    payload = _read(binding.session_root / "session.json", _JSON_BYTES, document["config_sha256"])
    _equal(config, json.loads(payload), "session record changed")
    _number(config["deadline_monotonic"])
    if _number(config["timeout_seconds"], maximum=7200) <= 0:
        raise ValueError("relative constructor timeout differs")
    if previous is not None:
        for key in ("config", "config_sha256", "ready_sha256"):
            _equal(document[key], previous[key], "session identity restarted or changed")
    for key, name in (
        ("ready_sha256", "ready.json"),
        ("complete_sha256", "COMPLETE.json"),
        ("failed_sha256", "FAILED.json"),
    ):
        if document[key] is not None:
            _read(binding.session_root / name, _JSON_BYTES, document[key])
    if document["ready_sha256"] is not None:
        _session_record(binding.session_root, config)
    return document


def verify_feature_cache_ledger(
    receipts: tuple[FeatureCacheReceipt, ...],
    *,
    expected_binding: FeatureRunBinding,
    expected_intents: tuple[FeatureIntent, ...],
    expected_releases: tuple[PrivateFeatureRelease, ...],
    expected_final_head: str,
    expected_assemblies: tuple[FeatureAssemblyBinding, ...] = (),
    expected_post_ledger_inventory: tuple[dict, ...] = (),
) -> dict:
    """Reconstruct exact externally bound feature-only operations, offline.

    Never calls the bridge, a consumer, a worker, a model or an oracle. Saved
    physical artifacts, request-return reports and accepted cache transitions
    are distinct. Local clock/dispatch/return/failure claims are not independent
    execution evidence. Offline verification consumes its own bounded audit
    allocation; it is not free work within a scientific controller clock.
    Optional post-ledger inventory is permitted only after a failed ledger.
    The caller independently authenticates its semantics and ordering (for
    example, a subsequent journal stop); these pins alone are not authority.
    No ledger-prefix entry may change and every additional entry must match.
    """
    if (
        type(expected_binding) is not FeatureRunBinding
        or type(receipts) is not tuple
        or not receipts
    ):
        raise ValueError("exact binding and nonempty immutable receipt sequence required")
    expected_binding.__post_init__()
    for values, kind in (
        (expected_intents, FeatureIntent),
        (expected_releases, PrivateFeatureRelease),
        (expected_assemblies, FeatureAssemblyBinding),
    ):
        if type(values) is not tuple or any(type(value) is not kind for value in values):
            raise ValueError("external expectations require exact immutable record tuples")
    if type(expected_post_ledger_inventory) is not tuple or any(
        type(row) is not dict or type(row.get("path")) is not str
        for row in expected_post_ledger_inventory
    ):
        raise ValueError("post-ledger inventory requires exact external entry pins")
    post_ledger_bytes = canonical_json(expected_post_ledger_inventory)
    if len(post_ledger_bytes) > _OPERATION_BYTES:
        raise ValueError("post-ledger inventory exceeds metadata bound")
    binding_bytes = canonical_json(expected_binding.document())
    expected_bytes = canonical_json(
        [
            [value.document() for value in expected_intents],
            [value.document() for value in expected_releases],
            [value.document() for value in expected_assemblies],
        ]
    )
    original_seals = tuple(receipt.sha256 for receipt in receipts)
    _hash(expected_final_head)
    source_pins = _source_snapshot(expected_binding)
    root = Path(expected_binding.run_root)
    initial_inventory = inspect_feature_tree(root)
    counters = json.loads(canonical_json(FeatureCounters().document()))
    evidence_head = accepted_head = expected_binding.sha256
    release_head = expected_binding.sha256
    visible, private, acquired = {}, {}, {}
    session_pin, physical_head, physical_ordinal = None, None, 0
    intent_index = release_index = 0
    history_assemblies = {value.sha256: value for value in expected_assemblies}
    child_groups, assembled, summaries, file_cache, released_ids = {}, [], [], {}, set()
    last_time, terminal = float(expected_binding.original_epoch), False
    partial_limits, batch_audits, invocation_pins = [], [], []
    last_after = None
    for ordinal, receipt in enumerate(receipts):
        if (
            type(receipt) is not FeatureCacheReceipt
            or len(receipt.payload) > _OPERATION_BYTES
            or terminal
        ):
            raise ValueError("receipt type/size differs or work followed a terminal operation")
        document = receipt.document()
        completed = document["status"] == "completed"
        kind, intent = document["kind"], document["intent"]
        if any(
            type(document[name]) is not list
            for name in ("sequences", "row_origins", "scatter", "timings")
        ):
            raise ValueError("ordered operation fields must be exact JSON lists")
        staging = document["staging"]
        if type(staging) is not dict or set(staging) != {"started", "dispatch"}:
            raise ValueError("nonaccepting admission staging schema differs")
        _equal(document["binding"], json.loads(binding_bytes), "receipt run/source binding differs")
        if (
            document["operation_ordinal"] != ordinal
            or document["previous_evidence_head"] != evidence_head
            or document["previous_accepted_head"] != accepted_head
        ):
            raise ValueError("operation/evidence/accepted predecessor chain differs")
        if (ordinal == 0) != (kind == "open"):
            raise ValueError("one run-owned session must open first and cannot restart")
        _equal(document["before_counters"], counters, "operation erased prior paid work")
        if document["inventory_before"] is None:
            if (
                completed
                or document["staging"]["started"] is not None
                or document["batch"] is not None
            ):
                raise ValueError("accepted/started physical work erased its before inventory")
            before = {}
            partial_limits.append(
                {
                    "operation_ordinal": ordinal,
                    "unreconstructed": "pre-operation inventory not retained before failure",
                }
            )
        else:
            before = _inventory_record(document["inventory_before"], expected_binding)
        suffix = "" if completed else ".failure"
        pending_path = (
            expected_binding.audit_root.relative_to(root).as_posix()
            + f"/operation-{ordinal:06d}{suffix}.json"
        )
        after = _inventory_record(
            document["inventory_after"],
            expected_binding,
            pending_path=pending_path,
            pending_bytes=len(receipt.payload),
        )
        if not set(before) <= set(after):
            raise ValueError("operation removed retained artifacts")
        if last_after is not None and before and not set(last_after) <= set(before):
            raise ValueError("inter-operation inventory discarded prior evidence")
        _inventory_files(root, before, file_cache)
        _inventory_files(root, after, file_cache)
        _read(_relative_path(root, pending_path), _OPERATION_BYTES, receipt.sha256)
        last_time = _timing_record(document, expected_binding, last_time)
        if not document["timings"]:
            partial_limits.append(
                {
                    "operation_ordinal": ordinal,
                    "unreconstructed": "no valid clock observation before failure",
                }
            )
        if kind == "open":
            if intent is not None:
                raise ValueError("opening a session cannot invent a consumer intent")
        elif intent is not None:
            if intent_index >= len(expected_intents):
                raise ValueError("operation has an unrequested external intent")
            _equal(
                intent,
                expected_intents[intent_index].document(),
                "external intent/subset/deadline differs",
            )
            intent_index += 1
            if (
                intent["objective_context_sha256"] != expected_binding.objective_context_sha256
                or not expected_binding.original_epoch
                < intent["original_effective_deadline"]
                <= expected_binding.original_deadline
            ):
                raise ValueError("intent changed original context or enlarged its clock")
            if kind != "assemble" and intent["expected_previous_head"] != accepted_head:
                raise ValueError("intent does not bind the actual accepted predecessor")
        elif completed:
            raise ValueError("completed consumer operation lacks an authenticated intent")
        if staging["started"] is not None:
            pin = staging["started"]
            expected_path = (
                expected_binding.audit_root.relative_to(root).as_posix()
                + f"/operation-{ordinal:06d}.started.json"
            )
            if set(pin) != {"path", "sha256"} or pin["path"] != expected_path:
                raise ValueError("STARTED marker path/schema differs")
            started, _ = _document(_relative_path(root, pin["path"]), pin["sha256"])
            expected_started = {
                "artifact": "run_feature_operation_started_v1",
                "binding_sha256": expected_binding.sha256,
                "operation_ordinal": ordinal,
                "kind": kind,
                "intent": intent,
                "parent_assembly": document["parent_assembly"],
                "sequences": document["sequences"],
                "representation": document["representation"],
                "before_counters": counters,
                "previous_evidence_head": evidence_head,
                "previous_accepted_head": accepted_head,
                "monotonic": document["timings"][0]["monotonic"],
            }
            _equal(started, expected_started, "STARTED marker rewrote admission inputs")
        elif completed:
            raise ValueError("completed operation lacks its STARTED evidence")
        next_counters = json.loads(canonical_json(counters))
        if kind in ("raw", "preload") and intent is not None:
            next_counters["logical_opportunities"] += 1
            if intent["purpose"] == "candidate":
                if intent["round_index"] == 29:
                    raise ValueError("terminal round cannot spend candidate opportunities")
                next_counters["candidate_opportunities"] += 1
                next_counters["candidate_wave_counts"][intent["round_index"] - 1] += 1
                if expected_binding.arm_id in FULL_ARMS:
                    next_counters["full_accounted_requests"] += 1
            else:
                next_counters["auxiliary_opportunities"] += 1
        session_pin = _checked_session_pin(
            document["session"], expected_binding, session_pin, failed=not completed
        )
        if completed and session_pin is None:
            raise ValueError("accepted operation lacks its original saved session")
        if completed and session_pin["ready_sha256"] is None:
            raise ValueError("accepted operation lacks its original startup receipt")
        if session_pin is not None and physical_head is None:
            physical_head = session_pin["config_sha256"]
        sequences = tuple(document["sequences"])
        representation = document["representation"]
        parent = document["parent_assembly"]
        if parent is not None:
            if (
                kind != "raw"
                or set(parent) != {"binding_sha256", "intent"}
                or parent["binding_sha256"] not in history_assemblies
            ):
                raise ValueError("child operation has an unbound assembly parent")
            assembly_binding = history_assemblies[parent["binding_sha256"]]
            selected, history = _assembly_rows(assembly_binding, expected_binding)
            if (
                intent is None
                or intent["history_sha256"] != assembly_binding.history_sha256
                or intent["round_index"] != history["round_index"]
            ):
                raise ValueError("child assembly history/round differs")
            group_key = digest(canonical_json(parent))
            if group_key not in child_groups:
                next_counters["assembly_opportunities"] += 1
            group = child_groups.setdefault(
                group_key, {"parent": parent, "receipts": [], "rows": []}
            )
            _equal(group["parent"], parent, "assembly parent changed between feature-only children")
            if (
                not group["receipts"]
                and parent["intent"]["expected_previous_head"] != accepted_head
            ):
                raise ValueError("assembly original predecessor was reset")
            if (
                parent["intent"]["original_effective_deadline"]
                != intent["original_effective_deadline"]
                or parent["intent"]["history_sha256"] != intent["history_sha256"]
            ):
                raise ValueError("child reset its parent's original clock/history")
            if any(sequence not in {row["sequence"] for row in selected} for sequence in sequences):
                raise ValueError("assembly child requests an ineligible/unrevealed sequence")
        requested_origins, new_origins, dispatch_request = [], [], None
        target_cache = private if kind == "preload" else visible
        if kind in ("raw", "preload") and intent is not None:
            if (
                type(document["sequences"]) is not list
                or not 1 <= len(sequences) <= 128
                or (
                    (
                        kind == "raw"
                        and (type(representation) is not str or representation not in ALIASES)
                    )
                    or (kind == "preload" and representation is not None)
                )
            ):
                raise ValueError("raw requested rows/representation differ")
            for sequence in sequences:
                CandidateFeatureRequest("ledger-audit", "row", (sequence,))
            if kind == "preload" and (
                len(set(sequences)) != len(sequences) or intent["purpose"] != "private_preload"
            ):
                raise ValueError("private preload requires its declared unique vault request")
            if kind == "raw" and intent["purpose"] not in (
                "candidate",
                "charged",
                "initial",
                "terminal",
            ):
                raise ValueError("raw feature purpose differs")
            profile = expected_binding.tr2_grouped_capacity
            if profile and document["status"] == "completed":
                if kind == "preload":
                    check_tr2_capacity_request(profile, 0, sequences)
                    if private or physical_ordinal != 0:
                        raise ValueError("TR2 capacity repeated/missing first preload")
                elif (
                    len(private) != 120
                    or not set(profile["shared_sequence_ids"][:64]) <= set(visible)
                    or (intent["purpose"] == "candidate" and len(sequences) > 80)
                ):
                    raise ValueError("TR2 capacity initial release/grouped request differs")
            misses = tuple(
                sequence
                for sequence in dict.fromkeys(sequences)
                if digest(sequence.encode("ascii")) not in target_cache
            )
        else:
            misses = ()
        batch = document["batch"]
        if batch is not None:
            if (
                kind not in ("raw", "preload")
                or intent is None
                or not misses
                or session_pin is None
            ):
                raise ValueError("physical batch has no legitimate visible/private miss")
            if set(batch) != {
                "root",
                "ordinal",
                "expected_previous_head",
                "request_sha256",
                "command_sha256",
                "response_sha256",
                "invocation_sha256",
                "manifest_sha256",
                "dispatched",
                "returned",
            }:
                raise ValueError("physical batch outcome schema differs")
            if (
                type(batch["dispatched"]) is not bool
                or type(batch["returned"]) is not bool
                or (batch["returned"] and not batch["dispatched"])
            ):
                raise ValueError("physical dispatch/return type or order differs")
            expected_batch_path = (
                expected_binding.session_root.relative_to(root).as_posix()
                + f"/batch-{physical_ordinal:04d}"
            )
            if (
                batch["root"] != expected_batch_path
                or type(batch["ordinal"]) is not int
                or batch["ordinal"] != physical_ordinal
                or batch["expected_previous_head"] != physical_head
            ):
                raise ValueError("physical request reset ordinal/head or source session")
            if batch["dispatched"]:
                profile = expected_binding.tr2_grouped_capacity
                if profile:
                    check_tr2_capacity_request(profile, physical_ordinal, misses)
                    if (
                        intent["purpose"]
                        != ("private_preload" if physical_ordinal == 0 else "candidate")
                        or next_counters["acquired_origins"] + len(misses)
                        > expected_binding.origin_limit
                    ):
                        raise ValueError("TR2 capacity acquisition setup/origin bound differs")
                candidate_run_limit, candidate_wave_limit = expected_binding.candidate_limits
                projected_full = next_counters["full_accounted_requests"] + (
                    intent["purpose"] != "candidate"
                )
                if (
                    next_counters["physical_attempts"] >= expected_binding.admission_limit
                    or next_counters["candidate_opportunities"] > candidate_run_limit
                    or any(
                        value > candidate_wave_limit
                        for value in next_counters["candidate_wave_counts"]
                    )
                    or (expected_binding.arm_id in FULL_ARMS and projected_full > 128)
                ):
                    raise ValueError("a failed admission gate still dispatched physical work")
                if staging["dispatch"] is None:
                    raise ValueError("physical attempt lacks its predispatch evidence")
            if staging["dispatch"] is not None:
                pin = staging["dispatch"]
                expected_path = (
                    expected_binding.audit_root.relative_to(root).as_posix()
                    + f"/operation-{ordinal:06d}.dispatch.json"
                )
                if set(pin) != {"path", "sha256"} or pin["path"] != expected_path:
                    raise ValueError("dispatch marker path/schema differs")
                dispatch, _ = _document(_relative_path(root, pin["path"]), pin["sha256"])
                if type(dispatch.get("request")) is not dict:
                    raise ValueError("dispatch request is missing")
                request_bytes = canonical_json(dispatch["request"])
                dispatch_request = CandidateFeatureRequest.from_bytes(
                    request_bytes, batch["request_sha256"]
                )
                if (
                    dispatch_request.sequences != misses
                    or dispatch_request.run_id != expected_binding.run_id
                ):
                    raise ValueError(
                        "physical admission used private hits, reordered misses or repeated a row"
                    )
                admission_counters = json.loads(canonical_json(next_counters))
                admission_counters["physical_attempts"] += 1
                admission_counters["real_rows_dispatched"] += len(misses)
                admission_counters["padded_rows_dispatched"] += 128
                if expected_binding.arm_id in FULL_ARMS and intent["purpose"] != "candidate":
                    admission_counters["full_accounted_requests"] += 1
                monotonic = _number(dispatch.get("monotonic"))
                if not document["timings"][0]["monotonic"] <= monotonic <= last_time:
                    raise ValueError("dispatch clock lies outside its recorded operation")
                expected_dispatch = {
                    "artifact": "run_feature_dispatch_v1",
                    "binding_sha256": expected_binding.sha256,
                    "operation_ordinal": ordinal,
                    "batch_ordinal": physical_ordinal,
                    "request": dispatch["request"],
                    "request_sha256": batch["request_sha256"],
                    "session_sha256": session_pin["config_sha256"],
                    "expected_previous_head": physical_head,
                    "counters_after_admission": admission_counters,
                    "monotonic": dispatch["monotonic"],
                }
                _equal(
                    dispatch,
                    expected_dispatch,
                    "dispatch inputs/counters were resealed inconsistently",
                )
            else:
                dispatch_request = CandidateFeatureRequest(
                    expected_binding.run_id, f"feature-{physical_ordinal:04d}", misses
                )
                if digest(dispatch_request.payload) != batch["request_sha256"]:
                    raise ValueError("unadmitted planned request is not reconstructible")
            if batch["dispatched"]:
                next_counters = admission_counters
                next_counters["physical_completed" if batch["returned"] else "physical_failed"] += 1
            elif completed:
                raise ValueError("nonadmitted physical work claimed acceptance")
            available_pins = {}
            for key, name in (
                ("request_sha256", "request.json"),
                ("command_sha256", "command.json"),
                ("response_sha256", "response.json"),
                ("invocation_sha256", "invocation.json"),
                ("manifest_sha256", "features/manifest.json"),
            ):
                path = batch["root"] + "/" + name
                saved = after.get(path)
                if saved is not None:
                    available_pins[key] = saved["sha256"]
                    if batch[key] is not None and batch[key] != saved["sha256"]:
                        raise ValueError("retained partial artifact differs from its recorded pin")
                    payload = _read(_relative_path(root, path), _JSON_BYTES, saved["sha256"])
                    if key == "request_sha256" and payload != dispatch_request.payload:
                        raise ValueError("retained partial request differs from planned rows")
                    if key in ("command_sha256", "response_sha256"):
                        expected_wire = {
                            "operation": "features",
                            "ordinal": physical_ordinal,
                            "previous_sha256": physical_head,
                            "request_sha256": batch["request_sha256"],
                        }
                        if key == "response_sha256":
                            response_value = json.loads(payload)
                            expected_wire["manifest_sha256"] = _hash(
                                response_value.get("manifest_sha256")
                            )
                        if payload != canonical_json(expected_wire):
                            raise ValueError("retained partial request/response chain differs")
                elif batch[key] is not None and key != "request_sha256":
                    raise ValueError("partial receipt names an artifact that was not retained")
            if len(available_pins) == 5:
                if not batch["dispatched"]:
                    raise ValueError(
                        "completed physical artifacts cannot erase the invocation attempt"
                    )
                request, arrays, manifest, summary = load_saved_feature_batch(
                    expected_binding.session_root,
                    expected_session=session_pin["config"],
                    ordinal=physical_ordinal,
                    expected_previous_head=physical_head,
                    expected_response_sha256=available_pins["response_sha256"],
                    expected_runtime=json.loads(expected_binding.runtime_payload),
                    expected_model_sha256=expected_binding.model_sha256,
                )
                if dispatch_request is None or request.payload != dispatch_request.payload:
                    raise ValueError("returned physical request differs from its admission")
                for key in (
                    "request_sha256",
                    "command_sha256",
                    "response_sha256",
                    "invocation_sha256",
                    "manifest_sha256",
                ):
                    if available_pins[key] != summary[key]:
                        raise ValueError("physical outcome pins differ from saved reconstruction")
                new_origins = _acquired_origins(
                    request, arrays, ordinal, physical_ordinal, summary["manifest_sha256"]
                )
                if batch["returned"]:
                    physical_head, physical_ordinal = (
                        summary["response_sha256"],
                        physical_ordinal + 1,
                    )
                    invocation_pins.append(summary["invocation_sha256"])
                del arrays, manifest
            if completed and (not batch["returned"] or len(available_pins) != 5):
                raise ValueError(
                    "a nonreturned or incompletely reconstructed request cannot commit cached rows"
                )
            batch_audits.append(
                {
                    "operation_ordinal": ordinal,
                    "dispatched_local_report": batch["dispatched"],
                    "returned_local_report": batch["returned"],
                    "retained_artifact_pins": available_pins,
                    "complete_saved_arrays_reconstructed": len(available_pins) == 5,
                    "accepted_cache_transition": completed,
                }
            )
            if len(available_pins) != 5:
                partial_limits.append(
                    {
                        "operation_ordinal": ordinal,
                        "unreconstructed": "incomplete physical artifact prefix; paid dispatch/return reports retained separately",
                    }
                )
        elif staging["dispatch"] is not None or (completed and misses):
            raise ValueError("unexplained physical dispatch or missing acquisition")
        pending_cache = dict(target_cache)
        for origin in new_origins:
            pending_cache[origin["sequence_id"]] = origin
        if (
            kind in ("raw", "preload")
            and intent is not None
            and all(digest(sequence.encode("ascii")) in pending_cache for sequence in sequences)
        ):
            requested_origins = [
                pending_cache[digest(sequence.encode("ascii"))] for sequence in sequences
            ]
        if kind == "release" and document["release"] is None and not completed:
            if release_index >= len(expected_releases):
                raise ValueError("failed release lacks its external attempted-release expectation")
            release_index += 1
            partial_limits.append(
                {
                    "operation_ordinal": ordinal,
                    "unreconstructed": "release authority was not retained before failure; no release accepted",
                }
            )
        elif kind == "release":
            if release_index >= len(expected_releases) or intent is None:
                raise ValueError("private release lacks its external exact subset")
            release = expected_releases[release_index]
            release_index += 1
            _equal(
                document["release"],
                release.document(),
                "release source/history/exact subset differs",
            )
            if (
                release.run_id != expected_binding.run_id
                or release.history_sha256 != intent["history_sha256"]
                or release.objective_context_sha256 != intent["objective_context_sha256"]
                or release.expected_previous_release_head != release_head
            ):
                raise ValueError("release context/history/predecessor differs")
            if any(key not in private for key in release.revealed_sequence_ids):
                raise ValueError("release names a row absent from the private vault")
            for key in release.revealed_sequence_ids:
                if key in visible and visible[key]["raw_sha256s"] != private[key]["raw_sha256s"]:
                    raise ValueError("released private row disagrees with first visible features")
                requested_origins.append(visible.get(key, private[key]))
            if set(release.revealed_sequence_ids) & released_ids:
                raise ValueError("private release reused an already revealed subset")
            if sequences or representation is not None:
                raise ValueError("controller-only release exposed a public feature result")
            requested_origins = []
        elif document["release"] is not None:
            raise ValueError("nonrelease operation injected a private release")
        if kind == "assemble" and document["assembly"] is None and not completed:
            if intent is not None and not any(
                canonical_json(group["parent"]["intent"]) == canonical_json(intent)
                for group in child_groups.values()
            ):
                next_counters["assembly_opportunities"] += 1
            partial_limits.append(
                {
                    "operation_ordinal": ordinal,
                    "unreconstructed": "assembly inputs not retained before failure; no learner input accepted",
                }
            )
        elif kind == "assemble":
            assembly = document["assembly"]
            if (
                intent is None
                or type(assembly) is not dict
                or set(assembly)
                != {
                    "binding",
                    "selected_query_ids",
                    "feature_sequence_ids",
                    "child_receipt_sha256s",
                }
            ):
                raise ValueError("charged assembly is missing its external binding")
            key = digest(canonical_json(assembly["binding"]))
            if key not in history_assemblies:
                raise ValueError("assembly supplied an unrequested eligibility/history binding")
            selected, history = _assembly_rows(history_assemblies[key], expected_binding)
            _equal(
                assembly["binding"],
                history_assemblies[key].document(),
                "assembly input binding differs",
            )
            if (
                intent["history_sha256"] != history_assemblies[key].history_sha256
                or intent["round_index"] != history["round_index"]
            ):
                raise ValueError("assembly history/context round differs")
            group = child_groups.get(
                digest(canonical_json({"binding_sha256": key, "intent": intent}))
            )
            if group is None:
                next_counters["assembly_opportunities"] += 1
                if intent["expected_previous_head"] != accepted_head:
                    raise ValueError("zero-child assembly changed its predecessor")
                child_seals = []
            else:
                _equal(
                    group["parent"]["intent"], intent, "assembly changed its original parent intent"
                )
                child_seals = group["receipts"]
            expected_sequence_ids = [digest(row["sequence"].encode("ascii")) for row in selected]
            _equal(
                assembly["selected_query_ids"],
                [row["query_id"] for row in selected],
                "assembly query order or eligibility changed",
            )
            _equal(
                assembly["feature_sequence_ids"],
                expected_sequence_ids,
                "assembly feature order changed",
            )
            _equal(
                assembly["child_receipt_sha256s"],
                child_seals,
                "assembly hid/reordered feature-only child work",
            )
            if sequences != tuple(row["sequence"] for row in selected):
                raise ValueError("assembly changed charged raw history ordering")
            if completed and any(key not in visible for key in expected_sequence_ids):
                raise ValueError("completed assembly has missing feature rows")
            if all(key in visible for key in expected_sequence_ids):
                requested_origins = [visible[key] for key in expected_sequence_ids]
            assembled.append(key)
        elif document["assembly"] is not None:
            raise ValueError("nonassembly operation injected learner inputs")
        if kind in ("open", "close") and (
            sequences or representation is not None or batch is not None
        ):
            raise ValueError("lifecycle operation smuggles feature rows")
        unique_origins, scatter, lookup = [], [], {}
        for origin in requested_origins:
            key = origin["origin_sha256"]
            if key not in lookup:
                lookup[key] = len(unique_origins)
                unique_origins.append(origin)
            scatter.append(lookup[key])
        if kind == "preload":
            scatter = []
        if completed or document["row_origins"] or document["scatter"]:
            _equal(
                document["row_origins"],
                unique_origins,
                "row original acquisition/offset/hash differs",
            )
            _equal(document["scatter"], scatter, "ordered duplicate scatter differs")
        if kind in ("raw", "assemble") and (completed or document["public_payload"] is not None):
            if representation not in ALIASES:
                raise ValueError("public raw representation differs")
            public = _public_rows(sequences, representation, requested_origins, intent)
            _equal(
                document["public_payload"],
                public,
                "public output leaked private provenance or changed requested rows",
            )
            if document["public_sha256"] != digest(canonical_json(public)):
                raise ValueError("public semantic receipt digest differs")
        elif document["public_payload"] is not None or document["public_sha256"] is not None:
            raise ValueError("private/lifecycle operation exposed a consumer result")
        if completed:
            if document["failure"] is not None:
                raise ValueError("accepted operation also declares failure")
            if kind in ("raw", "preload"):
                target_cache.update(pending_cache)
                for origin in new_origins:
                    acquired[origin["origin_sha256"]] = origin
                if batch is None:
                    next_counters["cache_only_opportunities"] += 1
                if parent is not None:
                    child_groups[digest(canonical_json(parent))]["receipts"].append(receipt.sha256)
            if kind == "release":
                for key in release.revealed_sequence_ids:
                    visible.setdefault(key, private[key])
                next_counters["released_rows"] += len(release.revealed_sequence_ids)
                released_ids.update(release.revealed_sequence_ids)
                release_head = release.sha256
            next_counters["acquired_origins"] = len(acquired)
            next_counters["visible_rows"] = len(visible)
            next_counters["private_rows"] = len(private)
            accepted_head = receipt.sha256
        else:
            if document["failure"] is None:
                raise ValueError("failed operation lacks retained diagnostic evidence")
            terminal = True
        if kind == "close":
            terminal = True
            if completed:
                if (
                    session_pin is None
                    or session_pin["complete_sha256"] is None
                    or session_pin["failed_sha256"] is not None
                    or (expected_binding.session_root / "FAILED.json").exists()
                ):
                    raise ValueError("closed acceptance lacks original COMPLETE or ignores FAILED")
                complete, _ = _document(
                    expected_binding.session_root / "COMPLETE.json", session_pin["complete_sha256"]
                )
                if (
                    set(complete)
                    != {
                        "artifact",
                        "session_sha256",
                        "completed_requests",
                        "invocation_sha256s",
                        "previous_sha256",
                        "startup_seconds",
                        "elapsed_seconds",
                        "retained_bytes_before_receipt",
                        "stderr_sha256",
                        "oracle_calls",
                        "production_input_eligible",
                        "scientific_evidence_accepted",
                    }
                    or complete.get("artifact") != "warm_candidate_feature_session_complete_v1"
                    or type(complete.get("completed_requests")) is not int
                    or complete.get("completed_requests") != physical_ordinal
                    or complete.get("previous_sha256") != physical_head
                    or complete.get("session_sha256") != session_pin["config_sha256"]
                ):
                    raise ValueError("closed session omits physical requests")
                _equal(
                    complete["invocation_sha256s"],
                    invocation_pins,
                    "close erased ordered invocation evidence",
                )
                if (
                    complete["production_input_eligible"] is not False
                    or complete["scientific_evidence_accepted"] is not False
                    or type(complete["oracle_calls"]) is not int
                    or complete["oracle_calls"] != 0
                ):
                    raise ValueError("close introduced authority or changed oracle-call count")
                elapsed = _number(
                    complete["elapsed_seconds"], maximum=session_pin["config"]["timeout_seconds"]
                )
                _number(complete["startup_seconds"], maximum=min(120.0, elapsed))
                if elapsed >= session_pin["config"]["timeout_seconds"]:
                    raise ValueError("close exceeded the original internal session allowance")
                _stream_hash(
                    expected_binding.session_root / "stderr.log",
                    _SESSION_BYTES,
                    complete["stderr_sha256"],
                )
                complete_path = (
                    expected_binding.session_root.relative_to(root).as_posix() + "/COMPLETE.json"
                )
                if document["inventory_after"]["session_bytes"] - after[complete_path][
                    "bytes"
                ] != _count(complete["retained_bytes_before_receipt"], _SESSION_BYTES):
                    raise ValueError("close failed to charge the original COMPLETE marker bytes")
        _counter_limits(next_counters, expected_binding, failed=not completed)
        _equal(
            document["after_counters"],
            next_counters,
            "paid/accepted cache counters do not independently reconstruct",
        )
        counters, evidence_head = next_counters, receipt.sha256
        last_after = dict(after)
        last_after[pending_path] = {
            "path": pending_path,
            "type": "file",
            "bytes": len(receipt.payload),
            "sha256": receipt.sha256,
        }
        summaries.append(
            {
                "operation_ordinal": ordinal,
                "kind": kind,
                "status": document["status"],
                "sha256": receipt.sha256,
                "accepted_head": accepted_head,
            }
        )
    if (
        intent_index != len(expected_intents)
        or release_index != len(expected_releases)
        or evidence_head != expected_final_head
    ):
        raise ValueError("external operation/release/final-head inventory differs")
    if last_after is None:
        raise ValueError("empty reconstruction")
    final_inventory = inspect_feature_tree(root)
    _equal(final_inventory, initial_inventory, "saved run artifacts changed during audit")
    appended = {row["path"]: row for row in expected_post_ledger_inventory}
    if expected_post_ledger_inventory and (
        summaries[-1]["status"] != "failed"
        or len(appended) != len(expected_post_ledger_inventory)
        or set(appended) & set(last_after)
    ):
        raise ValueError("post-ledger inventory must append unique paths after failure")
    _equal(
        sorted(
            [*last_after.values(), *expected_post_ledger_inventory], key=lambda row: row["path"]
        ),
        final_inventory["entries"],
        "last post-publication inventory omits retained artifacts",
    )
    if expected_post_ledger_inventory:
        prefix = expected_binding.session_root.relative_to(root).as_posix() + "/"
        session_entries = [
            row for row in final_inventory["entries"] if row["path"].startswith(prefix)
        ]
        _inventory_record(
            {
                "entries": final_inventory["entries"],
                "regular_bytes": final_inventory["regular_bytes"],
                "session_bytes": sum(row.get("bytes", 0) for row in session_entries),
                "session_entries": len(session_entries),
                "pending_final_path": None,
                "pending_final_bytes": 0,
            },
            expected_binding,
        )
    if canonical_json(expected_post_ledger_inventory) != post_ledger_bytes:
        raise ValueError("post-ledger external pins changed during audit")
    _equal(_source_snapshot(expected_binding), source_pins, "source bytes changed during audit")
    if (
        canonical_json(expected_binding.document()) != binding_bytes
        or canonical_json(
            [
                [value.document() for value in expected_intents],
                [value.document() for value in expected_releases],
                [value.document() for value in expected_assemblies],
            ]
        )
        != expected_bytes
        or tuple(receipt.sha256 for receipt in receipts) != original_seals
    ):
        raise ValueError("external expectations/receipt seals changed during reconstruction")
    for receipt in receipts:
        receipt.__post_init__()
    return {
        "artifact": "run_feature_cache_independent_reconstruction_v1",
        **(
            {
                "post_ledger_inventory": list(expected_post_ledger_inventory),
                "post_ledger_semantics_authenticated_by_this_reader": False,
            }
            if expected_post_ledger_inventory
            else {}
        ),
        "binding_sha256": expected_binding.sha256,
        "final_evidence_head": evidence_head,
        "final_accepted_head": accepted_head,
        "counters": counters,
        "operations": summaries,
        "physical_batch_audits": batch_audits,
        "partial_reconstruction_limits": partial_limits,
        "status": summaries[-1]["status"],
        "closed": summaries[-1]["kind"] == "close" and summaries[-1]["status"] == "completed",
        "request_return_and_dispatch_are_local_reports": True,
        "failed_semantic_payload_is_unreturned_staging_only": True,
        "failure_cause_or_external_timing_authenticated": False,
        "shared_safe_feature_derivation": True,
        "offline_audit_is_not_free_inline_work": True,
        "oracle_calls": 0,
        "scientific_evidence_accepted": False,
        "production_eligible": False,
    }

"""Independently reconstruct a dispatch journal, without producer transitions.

The generic phase capability checker is shared publication machinery, not an
independent filesystem durability implementation. Receipt authority is supplied
by the caller. A reconstructed report never authenticates that caller, elapsed
time, actual transport execution, or an unadopted/ambiguous continuation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import (
    PhaseSeal,
    canonical_json_bytes,
    checksum_manifest_bytes,
    verify_phase_capability,
)
from amp_challenge.generators.search.durable_dispatch_journal_records import (
    CONTRACT_PATH,
    CONTRACT_SHA256,
    EVENT_ARTIFACT,
    EVENT_DATA_FIELDS,
    EVENT_PAYLOAD,
    GENESIS_ARTIFACT,
    GENESIS_NAME,
    GENESIS_PAYLOAD,
    LOCK_NAME,
    MAX_EVENT_BYTES,
    MAX_EVENTS,
    MAX_JOURNAL_BYTES,
    PENDING_NAME,
    CallbackPin,
    DispatchRequest,
    JournalBinding,
    JournalCheckpoint,
    JournalEvent,
    JournalReport,
    OutstandingDispatch,
    binding_from_bytes,
    canonical,
    digest,
    document,
    exact_fields,
    identifier,
    pin,
    receipt_bytes,
    receipt_from_hex,
    request_from_document,
    require,
    source_inventory_sha256,
)
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import ChargedObservation
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

_EVENT_NAME = re.compile(r"event-([0-9]{6})\Z")
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_REGULAR = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_STATUS = {
    "succeeded": "successful",
    "failed": "failed",
    "missing": "missing",
    "censored": "censored",
    "partial": "partial",
    "timeout": "timed_out",
}


def _checkpoint_bytes(value):
    require(type(value) is JournalCheckpoint, "checkpoint exact type differs")
    value.__post_init__()
    return canonical(asdict(value))


def _fingerprint(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def _directory_path(path):
    require(
        isinstance(path, Path) and path.is_absolute() and path == Path(os.path.abspath(path)),
        "directory path is not canonical absolute",
    )
    descriptor = os.open(path.anchor, _DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, _DIRECTORY, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _directory_paths(paths):
    """Open shared prefixes once, for this callback-free inspection only.

    Every edge uses O_NOFOLLOW. On exit, recheck every live directory entry
    against its still-open parent, including the root, so an ancestor rename
    cannot turn an old descriptor into evidence about a replacement path.
    No handle survives this context or is reused across an external callback.
    """

    def identity(metadata):
        return metadata.st_dev, metadata.st_ino, metadata.st_mode

    with ExitStack() as stack:
        opened = {}
        for path in paths:
            require(
                isinstance(path, Path)
                and path.is_absolute()
                and path == Path(os.path.abspath(path)),
                "directory path is not canonical absolute",
            )
            for part in (*reversed(path.parents), path):
                if part in opened:
                    continue
                parent = None if part == part.parent else opened[part.parent][0]
                descriptor = os.open(
                    str(part) if parent is None else part.name, _DIRECTORY, dir_fd=parent
                )
                stack.callback(os.close, descriptor)
                opened[part] = descriptor, identity(os.fstat(descriptor))
        yield {path: opened[path][0] for path in paths}
        for path, (descriptor, original) in reversed(tuple(opened.items())):
            parent = None if path == path.parent else opened[path.parent][0]
            current = os.stat(
                str(path) if parent is None else path.name, dir_fd=parent, follow_symlinks=False
            )
            require(
                identity(os.fstat(descriptor)) == original and identity(current) == original,
                "source directory ancestry changed during replay",
            )


@contextmanager
def _child_directory(parent, name):
    original = os.stat(name, dir_fd=parent, follow_symlinks=False)
    require(stat.S_ISDIR(original.st_mode), "journal entry is not a real directory")
    descriptor = os.open(name, _DIRECTORY, dir_fd=parent)
    try:
        require(
            _fingerprint(os.fstat(descriptor)) == _fingerprint(original),
            "directory changed while opening",
        )
        yield descriptor
        require(
            _fingerprint(os.fstat(descriptor)) == _fingerprint(original)
            and _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
            == _fingerprint(original),
            "directory changed during capture",
        )
    finally:
        os.close(descriptor)


def _regular(parent, name, *, capture=False, maximum=None):
    original = os.stat(name, dir_fd=parent, follow_symlinks=False)
    require(stat.S_ISREG(original.st_mode), "journal/source file is not regular")
    if maximum is not None:
        require(original.st_size <= maximum, "file exceeds byte admission before reading")
    descriptor = os.open(name, _REGULAR, dir_fd=parent)
    try:
        require(
            _fingerprint(os.fstat(descriptor)) == _fingerprint(original),
            "file changed while opening",
        )
        hashed = hashlib.sha256()
        chunks = [] if capture else None
        total = 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            total += len(chunk)
            require(total <= original.st_size, "file grew while reading")
            hashed.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        require(
            total == original.st_size
            and _fingerprint(os.fstat(descriptor)) == _fingerprint(original)
            and _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
            == _fingerprint(original),
            "file changed during capture",
        )
        return original, hashed.hexdigest(), None if chunks is None else b"".join(chunks)
    finally:
        os.close(descriptor)


class _Sources:
    def __init__(self, expected):
        self.expected = expected
        self.sha256 = source_inventory_sha256(expected)
        self.root = Path(__file__).absolute().parents[4]
        self.frozen = []
        for relative, expected_digest in expected:
            path = self.root / relative
            with _directory_path(path.parent) as parent:
                metadata, actual, _ = _regular(parent, path.name)
            require(actual == expected_digest, "loaded implementation source bytes differ")
            if relative == CONTRACT_PATH:
                require(actual == CONTRACT_SHA256, "prospective contract bytes differ")
            self.frozen.append((path, _fingerprint(metadata)))

    def check(self, *, hashes=False):
        require(
            source_inventory_sha256(self.expected) == self.sha256,
            "external source inventory changed",
        )
        # Both traversals are fresh within this single callback-free check.
        # Sharing ancestors reduces repeated path opens, not file inspection.
        paths = tuple(dict.fromkeys(path.parent for path, _ in self.frozen))
        with _directory_paths(paths) as parents:
            directories = {path: _fingerprint(os.fstat(fd)) for path, fd in parents.items()}
            for (relative, expected_digest), (path, frozen) in zip(
                self.expected, self.frozen, strict=True
            ):
                require(path == self.root / relative, "source path binding changed")
                parent = parents[path.parent]
                if hashes:
                    metadata, actual, _ = _regular(parent, path.name)
                    require(actual == expected_digest, "implementation changed after replay")
                else:
                    metadata = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                require(_fingerprint(metadata) == frozen, "source identity changed during replay")
            with _directory_paths(paths) as current:
                for path, parent in parents.items():
                    require(
                        _fingerprint(os.fstat(parent)) == directories[path]
                        and _fingerprint(os.fstat(current[path])) == directories[path],
                        "source directory path changed during replay",
                    )


class _Authority:
    def __init__(self, binding, checkpoint, authenticator, sources):
        require(
            type(binding) is JournalBinding
            and type(checkpoint) is JournalCheckpoint
            and type(authenticator) is CallbackPin,
            "external authority records differ",
        )
        binding.__post_init__()
        checkpoint.__post_init__()
        authenticator.__post_init__()
        require(
            checkpoint.genesis_sha256 == binding.sha256
            and authenticator.source_sha256 == binding.authenticator_sha256
            and sources.sha256 == binding.implementation_sha256,
            "external authority/source bindings disagree",
        )
        self.binding = binding
        self.binding_bytes = canonical(binding.document())
        # Replay-local identity only. guard() still validates and compares the
        # complete live binding before/after every external callback.
        self.genesis_sha256 = digest(b"amp/durable-dispatch/genesis/v1\0" + self.binding_bytes)
        self.checkpoint = checkpoint
        self.checkpoint_bytes = _checkpoint_bytes(checkpoint)
        self.callback = authenticator
        self.target = authenticator.target
        self.source = authenticator.source_sha256
        self.sources = sources
        self.guard()

    def guard(self):
        # This getter may execute external code. All checks following it are
        # bounded structural/filesystem checks, not another provider invocation.
        observed = getattr(self.target, "source_sha256", None)
        require(
            type(self.callback) is CallbackPin
            and self.callback.target is self.target
            and type(self.callback.source_sha256) is str
            and self.callback.source_sha256 == self.source
            and type(observed) is str
            and observed == self.source,
            "authenticator callable/source drift",
        )
        require(
            canonical(self.binding.document()) == self.binding_bytes
            and _checkpoint_bytes(self.checkpoint) == self.checkpoint_bytes,
            "external binding/checkpoint drift",
        )
        self.sources.check()

    def authenticate(self, kind, raw, expected):
        receipt_bytes(raw)
        expected_bytes = canonical(expected)
        require(len(expected_bytes) <= MAX_EVENT_BYTES, "receipt expectation is too large")
        self.guard()
        result = self.target(kind, raw, expected_bytes)
        self.guard()
        return document(result)


def _names(descriptor, allowed=None, maximum=None):
    found = []
    with os.scandir(descriptor) as entries:
        for entry in entries:
            if allowed is not None:
                require(entry.name in allowed, "unexpected publication entry")
            found.append(entry.name)
            if maximum is not None:
                require(len(found) <= maximum, "publication entry inventory exceeds bounds")
    return tuple(sorted(found))


class _Tree:
    def __init__(self, root, trusted_parent, parent, descriptor):
        self.root = root
        self.trusted_parent = trusted_parent
        self.parent = parent
        self.descriptor = descriptor
        self.parent_identity = _fingerprint(os.fstat(parent))[:2]
        self.root_identity = _fingerprint(os.fstat(descriptor))

    def check_path(self):
        with _directory_path(self.trusted_parent) as reopened:
            require(
                _fingerprint(os.fstat(reopened))[:2] == self.parent_identity,
                "trusted parent was replaced",
            )
        require(
            _fingerprint(os.fstat(self.descriptor)) == self.root_identity
            and _fingerprint(os.stat(self.root.name, dir_fd=self.parent, follow_symlinks=False))
            == self.root_identity,
            "journal root changed during replay",
        )

    def snapshot(self):
        self.check_path()
        hashed = hashlib.sha256()
        total = 0
        entries_count = 0

        def flat_phase(parent, name, relative, *, pending=False):
            nonlocal total, entries_count
            payload = GENESIS_PAYLOAD if name == GENESIS_NAME else EVENT_PAYLOAD
            with _child_directory(parent, name) as phase:
                hashed.update(canonical([relative, _fingerprint(os.fstat(phase))]))
                for filename in _names(
                    phase, allowed={payload, "receipt.json", "SHA256SUMS"}, maximum=3
                ):
                    metadata = os.stat(filename, dir_fd=phase, follow_symlinks=False)
                    require(stat.S_ISREG(metadata.st_mode), "nested/special journal phase entry")
                    entries_count += 1
                    require(
                        total + metadata.st_size <= MAX_JOURNAL_BYTES,
                        "journal total byte limit exceeded before reading",
                    )
                    total += metadata.st_size
                    hashed.update(canonical([f"{relative}/{filename}", _fingerprint(metadata)]))
                    if metadata.st_mode & 0o444 == 0:
                        # An unreadable publication marker/remnant is never a
                        # committed capability. Its bytes are charged by stat,
                        # not represented as authenticated captured contents.
                        require(
                            pending or filename == "SHA256SUMS",
                            "unreadable non-marker publication payload",
                        )
                        hashed.update(b"unreadable-nonauthorizing-remnant\0")
                    else:
                        _, file_digest, _ = _regular(phase, filename)
                        hashed.update(file_digest.encode("ascii"))

        # The complete namespace is finite by the declared event-name law.
        # Reject unexpected entries before accumulating or recursing; all real
        # publications, including retained staging files, are flat three-file
        # phases. No recursive arbitrary directory walk or inode allowance.
        phase_names = {GENESIS_NAME, *(f"event-{index:06d}" for index in range(MAX_EVENTS))}
        root_names = _names(
            self.descriptor, allowed=phase_names | {PENDING_NAME, LOCK_NAME}, maximum=MAX_EVENTS + 3
        )
        for name in root_names:
            entries_count += 1
            if name == LOCK_NAME:
                metadata, file_digest, _ = _regular(self.descriptor, name, maximum=0)
                hashed.update(canonical([name, _fingerprint(metadata), file_digest]))
            elif name == PENDING_NAME:
                with _child_directory(self.descriptor, name) as pending:
                    hashed.update(canonical([name, _fingerprint(os.fstat(pending))]))
                    for child in _names(pending, allowed=phase_names, maximum=MAX_EVENTS + 1):
                        entries_count += 1
                        flat_phase(pending, child, f"{name}/{child}", pending=True)
            else:
                flat_phase(self.descriptor, name, name)
        self.check_path()
        return total, entries_count, hashed.hexdigest()

    def layout(self):
        allowed = {GENESIS_NAME, PENDING_NAME, LOCK_NAME}
        names = _names(self.descriptor, maximum=MAX_EVENTS + 3)
        indices = []
        for name in names:
            if name not in allowed:
                match = _EVENT_NAME.fullmatch(name)
                require(
                    match is not None and int(match[1]) < MAX_EVENTS,
                    "unexpected journal root entry",
                )
                indices.append(int(match[1]))
        require(all(name in names for name in allowed), "journal root inventory is incomplete")
        require(indices == list(range(len(indices))), "journal event names are not contiguous")
        lock, _, _ = _regular(self.descriptor, LOCK_NAME, maximum=0)
        require(lock.st_size == 0, "journal writer lock is not empty")
        with (
            _child_directory(self.descriptor, PENDING_NAME) as pending,
            os.scandir(pending) as entries,
        ):
            has_pending = next(entries, None) is not None
        return len(indices), has_pending

    def phase(self, name, payload_name, artifact, metadata, predecessors):
        with _child_directory(self.descriptor, name) as phase:
            wanted = {payload_name, "receipt.json", "SHA256SUMS"}
            names = _names(phase, allowed=wanted, maximum=3)
            if names != tuple(sorted(wanted)):
                return None
            marker = os.stat("SHA256SUMS", dir_fd=phase, follow_symlinks=False)
            require(stat.S_ISREG(marker.st_mode), "publication marker is not regular")
            if marker.st_mode & 0o444 == 0:
                return None
            captured = {}
            for filename in names:
                original, actual, payload = _regular(
                    phase, filename, capture=True, maximum=MAX_EVENT_BYTES
                )
                require(
                    stat.S_IMODE(original.st_mode) == 0o444 and original.st_nlink == 1,
                    "committed phase file mode/link count differs",
                )
                captured[filename] = (actual, payload)
            require(
                stat.S_IMODE(os.fstat(phase).st_mode) == 0o555,
                "committed phase directory mode differs",
            )
            receipt = {
                "artifact": artifact,
                "metadata": metadata,
                "payloads": {payload_name: captured[payload_name][0]},
                "predecessor_seals": predecessors,
                "schema_version": 1,
                "status": "sealed",
            }
            require(
                captured["receipt.json"][1] == canonical_json_bytes(receipt),
                "publication receipt metadata/payload differs",
            )
            manifest = checksum_manifest_bytes(
                {
                    payload_name: captured[payload_name][0],
                    "receipt.json": captured["receipt.json"][0],
                }
            )
            require(captured["SHA256SUMS"][1] == manifest, "publication manifest differs")
            seal = PhaseSeal(
                artifact,
                captured["SHA256SUMS"][0],
                captured["receipt.json"][0],
                tuple(sorted(predecessors.items())),
                ((payload_name, captured[payload_name][0]),),
                ((payload_name, captured[payload_name][1]),),
                names,
                canonical_json_bytes(metadata),
            )
            verify_phase_capability(
                seal,
                expected_artifact=artifact,
                expected_payload_paths=(payload_name,),
                expected_predecessor_seals=predecessors,
            )
            return captured[payload_name][1]


def _terminal(status, objectives):
    require(type(status) is str and status in _STATUS, "authenticated terminal status differs")
    if status == "succeeded":
        require(
            type(objectives) is list
            and len(objectives) == 2
            and all(
                type(value) is float and math.isfinite(value) and 0 <= value <= 1
                for value in objectives
            ),
            "authenticated objectives differ",
        )
        return tuple(objectives)
    require(objectives is None, "non-successful response has objective values")
    return None


class _Replay:
    def __init__(self, authority):
        self.authority = authority
        self.binding = authority.binding
        self.genesis_sha256 = authority.genesis_sha256
        self.head = self.genesis_sha256
        self.events = []
        self.observations = []
        self.intents = {}
        self.external_ids = set()
        self.imported = False
        self.stopped = False
        self.wave = 0
        self.wave_head = None
        self.wave_requests = ()
        self.next_seat = 0
        all_reserved = (
            *self.binding.initial_requests,
            *(row for wave in self.binding.reserves for row in wave),
        )
        self.query_ids = {row.query_id for row in all_reserved}
        self.sequences = {row.sequence for row in all_reserved}
        self.identity_keys = {row.identity.key for row in all_reserved}

    @property
    def charges(self):
        return (64 if self.imported else 0) + len(self.intents)

    def checkpoint(self):
        return JournalCheckpoint(self.genesis_sha256, len(self.events), self.head, self.charges)

    def _initial(self, data, event):
        require(not self.events and not self.imported, "initial import is not the first event")
        source_raw = receipt_from_hex(data["source_receipt_hex"])
        copy_raw = receipt_from_hex(data["copy_receipt_hex"])
        binding = self.binding
        require(
            digest(source_raw) == binding.initial_source_receipt_sha256
            and digest(copy_raw) == binding.initial_copy_receipt_sha256,
            "initial raw source/copy receipts differ from genesis",
        )
        expected = {
            "kind": "initial_source",
            "source_run_id": binding.initial_source_run_id,
            "seed": binding.seed,
            "objective_context_sha256": binding.objective_context_sha256,
            "oracle_bundle_sha256": binding.oracle_bundle_sha256,
            "requests": [request.document() for request in binding.initial_requests],
        }
        source = self.authority.authenticate("initial_source", source_raw, expected)
        exact_fields(source, ("rows",))
        require(
            type(source["rows"]) is list and len(source["rows"]) == 64,
            "authenticated initial source is not 64 rows",
        )
        require(
            canonical(source) == canonical(data["source_document"]),
            "retained initial source semantics differ from external authentication",
        )
        rows = []
        identifiers = set()
        for index, (request, row) in enumerate(
            zip(binding.initial_requests, source["rows"], strict=True)
        ):
            exact_fields(
                row,
                (
                    "request_sha256",
                    "external_submission_id",
                    "status",
                    "objectives",
                    "response_receipt_sha256",
                ),
            )
            external_id = row["external_submission_id"]
            require(
                row["request_sha256"] == request.sha256
                and identifier(external_id)
                and external_id not in identifiers
                and pin(row["response_receipt_sha256"]),
                "initial source request/receipt/external identity differs",
            )
            identifiers.add(external_id)
            values = _terminal(row["status"], row["objectives"])
            rows.append(
                ChargedObservation(
                    index,
                    request.query_id,
                    request.sequence,
                    row["response_receipt_sha256"],
                    _STATUS[row["status"]],
                    values,
                )
            )
        rows_sha256 = digest(canonical(source["rows"]))
        expected.update(
            {
                "kind": "initial_copy",
                "run_id": binding.run_id,
                "rows_sha256": rows_sha256,
                "source_receipt_sha256": binding.initial_source_receipt_sha256,
            }
        )
        copied = self.authority.authenticate("initial_copy", copy_raw, expected)
        wanted = {
            "run_id": binding.run_id,
            "seed": binding.seed,
            "rows_sha256": rows_sha256,
            "source_receipt_sha256": binding.initial_source_receipt_sha256,
        }
        require(
            canonical(copied) == canonical(wanted)
            and canonical(data["copy_document"]) == canonical(wanted),
            "initial same-seed copy binding differs",
        )
        self.observations = rows
        self.external_ids = identifiers
        self.imported = True
        self.wave_head = event.sha256

    def _wave(self, data, event):
        wave = data["wave_index"]
        require(
            type(wave) is int and wave == self.wave + 1 and wave <= 28,
            "adaptive waves are not contiguous",
        )
        require(
            len(self.observations) == 64 + 16 * (wave - 1)
            and self.charges == len(self.observations),
            "new wave precedes authenticated completion of prior charges",
        )
        require(
            type(data["requests"]) is list and len(data["requests"]) == 16,
            "request wave does not have 14 plus 2 seats",
        )
        requests = tuple(request_from_document(row) for row in data["requests"])
        for request in requests:
            self.binding.validate_request(request)
        require(
            canonical([row.document() for row in requests[14:]])
            == canonical([row.document() for row in self.binding.reserves[wave - 1]]),
            "reserve seats differ from the private frozen schedule",
        )
        for request in requests[:14]:
            require(
                request.query_id not in self.query_ids
                and request.sequence not in self.sequences
                and request.identity.key not in self.identity_keys,
                "method request collides with prior or reserved inventory",
            )
            self.query_ids.add(request.query_id)
            self.sequences.add(request.sequence)
            self.identity_keys.add(request.identity.key)
        self.wave = wave
        self.wave_head = event.sha256
        self.wave_requests = requests
        self.next_seat = 0

    def _intent(self, data, event):
        require(self.wave > 0 and self.next_seat < 16, "intent has no available sealed seat")
        request = request_from_document(data["request"])
        wanted = {
            "wave_index": self.wave,
            "seat_index": self.next_seat,
            "charge_index": self.charges,
            "request": self.wave_requests[self.next_seat].document(),
        }
        require(canonical(data) == canonical(wanted), "intent is not the exact next sealed seat")
        dispatch = DispatchRequest(
            self.genesis_sha256, event.sha256, self.wave, self.next_seat, self.charges, request
        )
        # Independently derive the token rather than call the producer/records helper.
        token = digest(
            b"amp/durable-dispatch/token/v1\0"
            + canonical(
                {
                    "genesis_sha256": self.genesis_sha256,
                    "intent_sha256": event.sha256,
                }
            )
        )
        require(dispatch.token_sha256 == token, "dispatch token domain differs")
        self.intents[event.sha256] = {
            "request": dispatch,
            "token": token,
            "external_id": None,
            "terminal": False,
            "fault": False,
        }
        self.next_seat += 1

    def _response(self, kind, data):
        require(
            pin(data["intent_sha256"]) and data["intent_sha256"] in self.intents,
            "receipt refers to an unknown durable intent",
        )
        entry = self.intents[data["intent_sha256"]]
        request = entry["request"]
        expected_request = {
            "genesis_sha256": self.genesis_sha256,
            "intent_sha256": request.intent_sha256,
            "wave_index": request.wave_index,
            "seat_index": request.seat_index,
            "charge_index": request.charge_index,
            "request": request.request.document(),
            "token_sha256": entry["token"],
        }
        expected = {
            "kind": kind,
            "run_id": self.binding.run_id,
            "request": expected_request,
            "objective_context_sha256": self.binding.objective_context_sha256,
            "oracle_bundle_sha256": self.binding.oracle_bundle_sha256,
            "transport_sha256": self.binding.transport_sha256,
            "timing_context_sha256": self.binding.timing_context_sha256,
        }
        if kind == "terminal_response":
            require(
                entry["external_id"] is not None
                and not entry["terminal"]
                and request.charge_index == len(self.observations),
                "terminal response is unacknowledged, repeated or out of charged order",
            )
            expected["external_submission_id"] = entry["external_id"]
        else:
            require(entry["external_id"] is None, "intent already has its sole external ID")
        raw = receipt_from_hex(data["receipt_hex"])
        actual = self.authority.authenticate(kind, raw, expected)
        require(
            canonical(actual) == canonical(data["authenticated"]),
            "saved receipt semantics differ from external authentication",
        )
        if kind == "submission_ack":
            exact_fields(actual, ("external_submission_id",))
            external_id = actual["external_submission_id"]
            require(
                identifier(external_id) and external_id not in self.external_ids,
                "external submission ID repeats within the run",
            )
            self.external_ids.add(external_id)
            entry["external_id"] = external_id
        else:
            exact_fields(actual, ("external_submission_id", "status", "objectives"))
            require(
                actual["external_submission_id"] == entry["external_id"],
                "terminal external ID changed",
            )
            values = _terminal(actual["status"], actual["objectives"])
            self.observations.append(
                ChargedObservation(
                    request.charge_index,
                    request.request.query_id,
                    request.request.sequence,
                    digest(raw),
                    _STATUS[actual["status"]],
                    values,
                )
            )
            entry["terminal"] = True

    def consume(self, event):
        raw = event.document()
        require(not self.stopped, "event follows terminal stop")
        require(
            raw["genesis_sha256"] == self.genesis_sha256
            and raw["ordinal"] == len(self.events)
            and raw["previous_event_sha256"] == self.head,
            "event chain binding differs",
        )
        kind, data = raw["kind"], raw["data"]
        exact_fields(data, EVENT_DATA_FIELDS[kind])
        require(
            kind in ("initial_import", "stop") or self.imported,
            "adaptive event precedes initial import",
        )
        if kind == "initial_import":
            self._initial(data, event)
        elif kind == "wave_seal":
            self._wave(data, event)
        elif kind == "dispatch_intent":
            self._intent(data, event)
        elif kind in ("submission_ack", "terminal_response"):
            self._response(kind, data)
        elif kind == "dispatch_fault":
            require(
                pin(data["intent_sha256"]) and data["intent_sha256"] in self.intents,
                "fault refers to an unknown intent",
            )
            entry = self.intents[data["intent_sha256"]]
            require(
                not entry["fault"]
                and type(data["stage"]) is str
                and data["stage"] in {"permission", "transport", "acknowledgement"}
                and type(data["error_type"]) is str
                and 1 <= len(data["error_type"]) <= 128
                and type(data["detail"]) is str
                and len(data["detail"]) <= 1024,
                "fault is repeated or malformed",
            )
            entry["fault"] = True
        else:
            require(
                type(data["reason"]) is str
                and data["reason"] in {"completed", "external_stop", "integrity_failure"}
                and type(data["detail"]) is str
                and len(data["detail"]) <= 1024,
                "stop evidence is malformed",
            )
            require(
                data["reason"] != "completed" or self.charges == len(self.observations) == 512,
                "completed stop lacks 512 authenticated terminal charges",
            )
            self.stopped = True
        self.head = event.sha256
        self.events.append(event.sha256)

    def report(self, *, ambiguous, journal_bytes, extension):
        checkpoint = self.checkpoint()
        observations = tuple(self.observations)
        outstanding = tuple(
            OutstandingDispatch(entry["request"], entry["external_id"])
            for entry in self.intents.values()
            if not entry["terminal"]
        )
        history = None
        if self.imported and not ambiguous:
            history = VerifiedHistorySnapshot(
                self.binding.run_id,
                self.binding.seed,
                self.wave + 1,
                self.binding.objective_context_sha256,
                self.binding.oracle_bundle_sha256,
                self.wave_head,
                observations,
                self.head,
            )
        return JournalReport(
            self.binding,
            checkpoint,
            history,
            observations,
            outstanding,
            len(self.intents),
            self.stopped,
            extension,
            ambiguous,
            journal_bytes,
            tuple(self.events),
        )


def _verify_dispatch_journal(
    root: Path,
    *,
    trusted_parent: Path,
    expected_binding: JournalBinding,
    expected_checkpoint: JournalCheckpoint,
    expected_source_inventory: tuple[tuple[str, str], ...],
    authenticator: CallbackPin,
    capture_initial: bool,
) -> tuple[JournalReport, tuple[bytes, bytes] | None]:
    """Shared independent replay; optional captures never bypass its final guards."""
    require(type(capture_initial) is bool, "initial capture option differs")
    require(
        isinstance(root, Path)
        and isinstance(trusted_parent, Path)
        and root.is_absolute()
        and root.parent == trusted_parent,
        "journal must be an immediate child of the trusted absolute parent",
    )
    sources = _Sources(expected_source_inventory)
    authority = _Authority(expected_binding, expected_checkpoint, authenticator, sources)
    with (
        _directory_path(trusted_parent) as parent,
        _child_directory(parent, root.name) as descriptor,
    ):
        tree = _Tree(root, trusted_parent, parent, descriptor)
        count, ambiguous = tree.layout()
        before = tree.snapshot()
        binding_bytes = tree.phase(
            GENESIS_NAME,
            GENESIS_PAYLOAD,
            GENESIS_ARTIFACT,
            {"genesis_sha256": authority.genesis_sha256},
            {},
        )
        require(binding_bytes is not None, "genesis has no authenticated publication")
        observed_binding = binding_from_bytes(binding_bytes)
        require(
            binding_bytes == authority.binding_bytes
            and observed_binding.sha256 == authority.genesis_sha256,
            "journal genesis differs from externally expected binding",
        )
        state = _Replay(authority)
        initial_payload = None
        checkpoint_seen = expected_checkpoint.event_count == 0
        if checkpoint_seen:
            require(state.checkpoint() == expected_checkpoint, "empty checkpoint differs")
        for ordinal in range(count):
            name = f"event-{ordinal:06d}"
            # Preliminary bounded payload capture obtains only the domain
            # digest needed by generic publication metadata, not authority.
            with _child_directory(descriptor, name) as phase:
                names = _names(
                    phase, allowed={EVENT_PAYLOAD, "receipt.json", "SHA256SUMS"}, maximum=3
                )
                if EVENT_PAYLOAD not in names:
                    require(ordinal == count - 1, "incomplete publication is not the final tail")
                    ambiguous = True
                    break
                _, _, provisional = _regular(
                    phase, EVENT_PAYLOAD, capture=True, maximum=MAX_EVENT_BYTES
                )
            provisional_digest = digest(b"amp/durable-dispatch/event/v1\0" + provisional)
            payload = tree.phase(
                name,
                EVENT_PAYLOAD,
                EVENT_ARTIFACT,
                {
                    "genesis_sha256": authority.genesis_sha256,
                    "event_sha256": provisional_digest,
                    "ordinal": ordinal,
                },
                {"previous_event": state.head},
            )
            if payload is None:
                require(ordinal == count - 1, "incomplete publication is not the final tail")
                ambiguous = True
                break
            require(payload == provisional, "event changed between capture and publication check")
            state.consume(JournalEvent(payload))
            if capture_initial and ordinal == 0:
                initial_payload = payload
            if len(state.events) == expected_checkpoint.event_count:
                require(
                    state.checkpoint() == expected_checkpoint,
                    "externally retained checkpoint differs from reconstructed prefix",
                )
                checkpoint_seen = True
        require(checkpoint_seen, "rollback: externally held checkpoint is absent")
        authority.guard()
        sources.check(hashes=True)
        require(tree.snapshot() == before, "journal changed during receipt reconstruction")
        result = state.report(
            ambiguous=ambiguous,
            journal_bytes=before[0],
            extension=len(state.events) > expected_checkpoint.event_count,
        )
        captured_payloads = None
        if capture_initial:
            require(
                result.checkpoint == expected_checkpoint
                and result.checkpoint.event_count == 1
                and result.checkpoint.charged_count == 64
                and len(result.event_sha256s) == 1
                and len(result.observations) == 64
                and result.adaptive_attempts == 0
                and not result.outstanding
                and not result.stopped
                and not result.ambiguous_tail
                and not result.extension_requires_adoption
                and result.history is not None,
                "journal is not a clean sole initial import",
            )
            require(
                type(binding_bytes) is bytes
                and 0 < len(binding_bytes) <= MAX_EVENT_BYTES
                and type(initial_payload) is bytes
                and 0 < len(initial_payload) <= MAX_EVENT_BYTES,
                "initial capture payloads are unavailable or oversized",
            )
            require(
                JournalEvent(initial_payload).document()["kind"] == "initial_import",
                "journal is not a clean sole initial import",
            )
            captured_payloads = (binding_bytes, initial_payload)
        returned = (result, captured_payloads)
        # No external provider callbacks after this structural/readback tail.
        require(
            canonical(expected_binding.document()) == authority.binding_bytes
            and _checkpoint_bytes(expected_checkpoint) == authority.checkpoint_bytes
            and authenticator.target is authority.target
            and authenticator.source_sha256 == authority.source,
            "external inputs changed during final report construction",
        )
        sources.check()
        tree.check_path()
        return returned


def verify_dispatch_journal(
    root: Path,
    *,
    trusted_parent: Path,
    expected_binding: JournalBinding,
    expected_checkpoint: JournalCheckpoint,
    expected_source_inventory: tuple[tuple[str, str], ...],
    authenticator: CallbackPin,
) -> JournalReport:
    """Re-authenticate one bounded journal; never call transport or reset a quota.

    A complete extra suffix is observed, not adopted. An ambiguous tail retains
    only the verified prefix's known charge counts; unknown additional work is
    not refunded. Any report with either flag is unusable for continuation.
    Offline reconstruction consumes its caller's separately bounded allocation.
    """
    report, _ = _verify_dispatch_journal(
        root,
        trusted_parent=trusted_parent,
        expected_binding=expected_binding,
        expected_checkpoint=expected_checkpoint,
        expected_source_inventory=expected_source_inventory,
        authenticator=authenticator,
        capture_initial=False,
    )
    return report


def verify_initial_dispatch_journal(
    root: Path,
    *,
    trusted_parent: Path,
    expected_binding: JournalBinding,
    expected_checkpoint: JournalCheckpoint,
    expected_source_inventory: tuple[tuple[str, str], ...],
    authenticator: CallbackPin,
) -> tuple[bytes, bytes]:
    """Return already-verified genesis/initial-event bytes, never release authority.

    Reconstruction is relative to the caller's external expectations. The
    returned tuple is constructible, not an authenticity receipt or evidence
    that physical oracle work happened. A composed consumer must invoke this
    function itself before claiming fresh reconstruction. No second read or
    provider invocation occurs after the shared verifier's final checks.
    Genesis includes private reserve identities: these payloads belong only
    at the controller/audit boundary, never in method or learner inputs.
    """
    require(
        type(expected_checkpoint) is JournalCheckpoint,
        "initial checkpoint must be an exact JournalCheckpoint",
    )
    expected_checkpoint.__post_init__()
    require(
        expected_checkpoint.event_count == 1 and expected_checkpoint.charged_count == 64,
        "initial checkpoint must contain exactly one event and 64 charges",
    )
    _, captured_payloads = _verify_dispatch_journal(
        root,
        trusted_parent=trusted_parent,
        expected_binding=expected_binding,
        expected_checkpoint=expected_checkpoint,
        expected_source_inventory=expected_source_inventory,
        authenticator=authenticator,
        capture_initial=True,
    )
    require(captured_payloads is not None, "initial capture payloads are unavailable")
    return captured_payloads


class JournalHistoryCallback:
    """Fixed adopted checkpoint and request-wave view; not a mutable head proxy."""

    def __init__(
        self,
        root: Path,
        *,
        trusted_parent: Path,
        expected_binding: JournalBinding,
        expected_checkpoint: JournalCheckpoint,
        expected_source_inventory: tuple[tuple[str, str], ...],
        authenticator: CallbackPin,
        expected_round_index: int,
        expected_wave_head_sha256: str,
    ):
        require(
            type(expected_round_index) is int
            and 1 <= expected_round_index <= 29
            and pin(expected_wave_head_sha256),
            "callback round/wave binding differs",
        )
        self._root = root
        self._arguments = {
            "trusted_parent": trusted_parent,
            "expected_binding": expected_binding,
            "expected_checkpoint": expected_checkpoint,
            "expected_source_inventory": expected_source_inventory,
            "authenticator": authenticator,
        }
        self._round = expected_round_index
        self._head = expected_wave_head_sha256
        self._provider = expected_binding.provider_sha256
        self._binding = canonical(expected_binding.document())
        self._checkpoint = _checkpoint_bytes(expected_checkpoint)
        self._auth_target = authenticator.target
        self._auth_source = authenticator.source_sha256
        self._validate(verify_dispatch_journal(self._root, **self._arguments))

    @property
    def provider_sha256(self):
        return self._provider

    def _validate(self, report):
        require(
            not report.extension_requires_adoption
            and not report.ambiguous_tail
            and report.history is not None,
            "callback lacks an exactly adopted clean history",
        )
        require(
            report.history.round_index == self._round
            and report.history.previous_wave_head_sha256 == self._head,
            "callback is bound to another request wave",
        )
        return report.history

    def __call__(self, expected_previous_head_sha256, expected_charge_count):
        require(
            type(expected_charge_count) is int
            and pin(expected_previous_head_sha256)
            and expected_charge_count == 64 + 16 * (self._round - 1)
            and expected_previous_head_sha256 == self._head,
            "caller history wave head/count differs",
        )
        binding = self._arguments["expected_binding"]
        checkpoint = self._arguments["expected_checkpoint"]
        authenticator = self._arguments["authenticator"]
        require(
            canonical(binding.document()) == self._binding
            and _checkpoint_bytes(checkpoint) == self._checkpoint
            and authenticator.target is self._auth_target
            and authenticator.source_sha256 == self._auth_source,
            "callback captured authority changed",
        )
        history = self._validate(verify_dispatch_journal(self._root, **self._arguments))
        require(
            binding.provider_sha256 == self._provider
            and canonical(binding.document()) == self._binding
            and _checkpoint_bytes(checkpoint) == self._checkpoint
            and authenticator.target is self._auth_target
            and authenticator.source_sha256 == self._auth_source,
            "callback authority changed during reconstruction",
        )
        return history

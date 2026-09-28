"""Single-arm durable write-ahead dispatch boundary, never an oracle or runner.

Normal writes retain an independently initialized, single-writer state. Recovery
uses the separately authored checker. An intent is irreversibly charged before
permission or transport; this module never dispatches an intent loaded from disk.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import (
    canonical_json_bytes,
    checksum_manifest_bytes,
    relocate_phase_capability_noreplace_at,
    verify_phase,
)
from amp_challenge.generators.search import durable_dispatch_journal_records as r

_REPOSITORY = Path(__file__).resolve().parents[4]


class DispatchUnresolved(RuntimeError):
    """A durable attempt remains spent without an accepted acknowledgement."""


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


def _directory(path):
    path = Path(path)
    r.require(
        path.is_absolute() and path == path.resolve(strict=True),
        "journal path must be canonical, existing and absolute",
    )
    for part in (path, *path.parents):
        r.require(stat.S_ISDIR(os.lstat(part).st_mode), "journal path crosses a non-directory")
    return path


def _source_check(repository, inventory):
    r.source_inventory_sha256(inventory)
    for name, expected in inventory:
        if name == r.CONTRACT_PATH:
            r.require(expected == r.CONTRACT_SHA256, "frozen prospective contract digest differs")
        path = repository / name
        r.require(path == path.resolve(strict=True), "source path alias differs")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(descriptor)
            r.require(
                stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "source is not a single-linked regular file",
            )
            hashed = hashlib.sha256()
            while chunk := os.read(descriptor, 65536):
                hashed.update(chunk)
            r.require(
                _fingerprint(os.fstat(descriptor)) == _fingerprint(before)
                and _fingerprint(os.stat(path, follow_symlinks=False)) == _fingerprint(before)
                and path == path.resolve(strict=True)
                and hashed.hexdigest() == expected,
                "implementation source bytes drifted",
            )
        finally:
            os.close(descriptor)


def _write_retained(directory_descriptor, name, payload):
    """Synchronize one exclusive stage file; never unlink partial bytes on failure."""
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset : offset + 65536])
            r.require(written > 0, "journal write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _content_digest(phase_descriptor, name, before):
    """Hash one admitted file freshly; metadata alone is not a content pin."""
    r.require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and stat.S_IMODE(before.st_mode) == 0o444
        and 0 < before.st_size <= r.MAX_EVENT_BYTES,
        "immutable phase file differs",
    )
    fingerprint = _fingerprint(before)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
        dir_fd=phase_descriptor,
    )
    try:
        r.require(
            _fingerprint(os.fstat(descriptor)) == fingerprint,
            "immutable phase file changed before content read",
        )
        hashed, count = hashlib.sha256(), 0
        while chunk := os.read(descriptor, min(65536, before.st_size + 1 - count)):
            count += len(chunk)
            r.require(count <= before.st_size, "immutable phase file grew during content read")
            hashed.update(chunk)
        r.require(count == before.st_size, "immutable phase file truncated during content read")
        r.require(
            _fingerprint(os.fstat(descriptor)) == fingerprint
            and _fingerprint(os.stat(name, dir_fd=phase_descriptor, follow_symlinks=False))
            == fingerprint,
            "immutable phase file changed during content read",
        )
        return hashed.hexdigest()
    finally:
        os.close(descriptor)


class DurableDispatchJournal:
    """Use create/open; callers own authenticators, permission and transport.

    `close` releases local descriptors without creating a scientific stop. `stop`
    durably forbids further mutation. Neither operation resets a charge or clock.
    Returned checkpoints require external retention; they are not issuer authority.
    """

    def __init__(self):
        raise TypeError("use DurableDispatchJournal.create or .open")

    @classmethod
    def create(
        cls,
        root,
        *,
        trusted_parent,
        binding,
        expected_source_inventory,
        authenticator,
        transport,
        permission,
        repository,
    ):
        self = cls._prepare(
            root,
            trusted_parent,
            binding,
            expected_source_inventory,
            authenticator,
            transport,
            permission,
            repository,
            create=True,
        )
        try:
            self._publish(
                r.GENESIS_NAME,
                r.GENESIS_PAYLOAD,
                r.canonical(binding.document()),
                r.GENESIS_ARTIFACT,
                {"genesis_sha256": binding.sha256},
                {},
            )
            self._checkpoint = r.JournalCheckpoint(binding.sha256, 0, binding.sha256, 0)
            self._guard()
            return self
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(
        cls,
        root,
        *,
        trusted_parent,
        binding,
        expected_checkpoint,
        expected_source_inventory,
        authenticator,
        transport,
        permission,
        repository,
    ):
        self = cls._prepare(
            root,
            trusted_parent,
            binding,
            expected_source_inventory,
            authenticator,
            transport,
            permission,
            repository,
            create=False,
        )
        try:
            r.require(
                type(expected_checkpoint) is r.JournalCheckpoint,
                "exact caller checkpoint required",
            )
            expected_checkpoint.__post_init__()
            r.require(
                expected_checkpoint.genesis_sha256 == binding.sha256,
                "journal genesis differs from caller checkpoint or binding",
            )
            genesis_descriptor = os.open(
                r.GENESIS_NAME,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=self._root_fd,
            )
            try:
                genesis = os.stat(
                    r.GENESIS_PAYLOAD, dir_fd=genesis_descriptor, follow_symlinks=False
                )
                r.require(
                    _content_digest(genesis_descriptor, r.GENESIS_PAYLOAD, genesis)
                    == r.digest(self._binding_bytes),
                    "journal genesis content differs from caller binding",
                )
            finally:
                os.close(genesis_descriptor)
            self._checkpoint = expected_checkpoint
            # Private observations only: independent replay must authenticate this
            # exact prefix, and subsequent captures must still match these bytes.
            self._capture(provisional=True)
            from amp_challenge.generators.search.durable_dispatch_journal_verify import (
                verify_dispatch_journal,
            )

            report = verify_dispatch_journal(
                self.root,
                trusted_parent=self._parent,
                expected_binding=binding,
                expected_checkpoint=expected_checkpoint,
                expected_source_inventory=expected_source_inventory,
                authenticator=authenticator,
            )
            r.require(type(report) is r.JournalReport, "independent report type differs")
            report.__post_init__()
            r.require(
                not report.extension_requires_adoption
                and not report.ambiguous_tail
                and report.checkpoint == expected_checkpoint,
                "resume requires explicit adoption of an exact complete journal tip",
            )
            self._checkpoint = report.checkpoint
            self._stopped = report.stopped
            for index, expected in enumerate(report.event_sha256s):
                event = r.JournalEvent(self._read(r.event_name(index) + "/" + r.EVENT_PAYLOAD))
                r.require(
                    event.sha256 == expected, "event changed after independent reconstruction"
                )
                self._remember(event)
            self._capture()
            self._guard(allow_stopped=True)
            return self
        except BaseException:
            self.close()
            raise

    @classmethod
    def _prepare(
        cls,
        root,
        trusted_parent,
        binding,
        inventory,
        authenticator,
        transport,
        permission,
        repository,
        *,
        create,
    ):
        r.require(type(binding) is r.JournalBinding, "exact journal binding required")
        binding.__post_init__()
        r.require(
            r.source_inventory_sha256(inventory) == binding.implementation_sha256,
            "caller implementation inventory differs from genesis",
        )
        for callback, expected in (
            (authenticator, binding.authenticator_sha256),
            (transport, binding.transport_sha256),
            (permission, binding.permission_sha256),
        ):
            r.require(
                type(callback) is r.CallbackPin and callback.source_sha256 == expected,
                "caller callback/source expectation differs",
            )
            callback.__post_init__()
        parent = _directory(trusted_parent)
        root = Path(root)
        r.require(
            root.is_absolute() and root.parent == parent and r.identifier(root.name),
            "journal root must be one canonical child of its trusted parent",
        )
        repository = _directory(repository)
        r.require(repository == _REPOSITORY, "source inventory root differs from loaded producer")
        _source_check(repository, inventory)
        self = object.__new__(cls)
        self.root, self._parent, self._repository = root, parent, repository
        self._binding, self._binding_bytes = binding, r.canonical(binding.document())
        self._inventory = inventory
        self._callbacks = (authenticator, transport, permission)
        self._targets = tuple(row.target for row in self._callbacks)
        self._source_pins = tuple(row.source_sha256 for row in self._callbacks)
        self._parent_fd = self._root_fd = self._pending_fd = self._lock_fd = -1
        self._closed = self._poisoned = self._stopped = self._busy = False
        self._watch = {}
        self._content_watch = {}
        self._pending_content = ()
        self._bytes = 0
        self._checkpoint = r.JournalCheckpoint(binding.sha256, 0, binding.sha256, 0)
        self._requests = ()
        self._wave_index = 0
        self._terminal_count = 0
        self._intents = {}
        self._external_ids = set()
        self._acks = {}
        self._faults = set()
        self._terminal_intents = set()
        try:
            self._parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if create:
                os.mkdir(root.name, 0o700, dir_fd=self._parent_fd)
                os.fsync(self._parent_fd)
            self._root_fd = os.open(
                root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._parent_fd
            )
            self._root_identity = _fingerprint(os.fstat(self._root_fd))[:2]
            if create:
                os.mkdir(r.PENDING_NAME, 0o700, dir_fd=self._root_fd)
            self._pending_fd = os.open(
                r.PENDING_NAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._root_fd
            )
            self._pending_identity = _fingerprint(os.fstat(self._pending_fd))[:2]
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            self._lock_fd = os.open(r.LOCK_NAME, flags, 0o600, dir_fd=self._root_fd)
            lock = os.fstat(self._lock_fd)
            r.require(
                stat.S_ISREG(lock.st_mode) and lock.st_nlink == 1 and lock.st_size == 0,
                "writer lock file differs",
            )
            self._lock_fingerprint = _fingerprint(lock)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if create:
                os.fsync(self._lock_fd)
                os.fsync(self._root_fd)
            self._path_guard()
            self._callback_guard()
            return self
        except BaseException:
            self.close()
            raise

    def _path_guard(self):
        r.require(not self._closed, "journal descriptors are closed")
        r.require(
            _directory(self._parent) == self._parent
            and _fingerprint(
                os.stat(self.root.name, dir_fd=self._parent_fd, follow_symlinks=False)
            )[:2]
            == self._root_identity
            and _fingerprint(os.lstat(self.root))[:2] == self._root_identity,
            "journal path changed",
        )
        r.require(
            _fingerprint(os.stat(r.PENDING_NAME, dir_fd=self._root_fd, follow_symlinks=False))[:2]
            == self._pending_identity,
            "journal staging directory changed",
        )
        r.require(
            _fingerprint(os.stat(r.LOCK_NAME, dir_fd=self._root_fd, follow_symlinks=False))
            == _fingerprint(os.fstat(self._lock_fd))
            == self._lock_fingerprint,
            "journal writer lock changed",
        )

    def _callback_guard(self):
        r.require(
            type(self._binding) is r.JournalBinding
            and r.canonical(self._binding.document()) == self._binding_bytes,
            "caller binding drifted",
        )
        r.require(
            type(self._callbacks) is tuple and len(self._callbacks) == 3, "callback tuple drifted"
        )
        for index, callback in enumerate(self._callbacks):
            r.require(
                type(callback) is r.CallbackPin
                and callback.target is self._targets[index]
                and callback.source_sha256 == self._source_pins[index],
                "exact caller callback identity drifted",
            )
            actual = getattr(callback.target, "source_sha256", None)
            r.require(
                type(actual) is str and actual == self._source_pins[index],
                "live callback source identity drifted",
            )
        _source_check(self._repository, self._inventory)
        r.require(
            r.source_inventory_sha256(self._inventory) == self._binding.implementation_sha256,
            "caller source inventory drifted",
        )
        r.require(
            type(self._callbacks) is tuple
            and len(self._callbacks) == 3
            and all(
                type(callback) is r.CallbackPin
                and callback.target is self._targets[index]
                and callback.source_sha256 == self._source_pins[index]
                for index, callback in enumerate(self._callbacks)
            ),
            "callback identity changed during a source accessor",
        )
        r.require(
            type(self._binding) is r.JournalBinding
            and r.canonical(self._binding.document()) == self._binding_bytes,
            "caller binding changed during a source accessor",
        )

    def _read(self, relative):
        phase, name = relative.split("/")
        phase_descriptor = os.open(
            phase, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._root_fd
        )
        descriptor = -1
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=phase_descriptor)
            before = os.fstat(descriptor)
            r.require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and 0 < before.st_size <= r.MAX_EVENT_BYTES,
                "journal record read bound differs",
            )
            data = bytearray()
            while chunk := os.read(descriptor, min(65536, r.MAX_EVENT_BYTES + 1 - len(data))):
                data.extend(chunk)
                r.require(len(data) <= r.MAX_EVENT_BYTES, "journal record grew beyond bound")
            r.require(
                _fingerprint(before) == _fingerprint(os.fstat(descriptor)),
                "journal record changed while reading",
            )
            return bytes(data)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(phase_descriptor)

    def _pending_empty(self):
        with os.scandir(self._pending_fd) as entries:
            r.require(next(entries, None) is None, "ambiguous pending publication remains")

    def _capture(self, *, provisional=False):
        """Freshly capture metadata and bytes, without replaying receipt semantics."""
        r.require(type(provisional) is bool, "provisional capture flag differs")
        pending = self._pending_content
        r.require(
            type(pending) is tuple
            and len(pending) in (0, 3)
            and all(
                type(row) is tuple and len(row) == 2 and type(row[0]) is str and r.pin(row[1])
                for row in pending
            ),
            "publication content handoff differs",
        )
        expected_new = dict(pending)
        r.require(len(expected_new) == len(pending), "publication content handoff repeats a key")
        if provisional:
            r.require(
                not self._watch and not self._content_watch and not pending and self._bytes == 0,
                "provisional capture cannot replace an existing watch",
            )
        self._path_guard()
        self._pending_empty()
        phases = (
            r.GENESIS_NAME,
            *(r.event_name(index) for index in range(self._checkpoint.event_count)),
        )
        names = {*phases, r.PENDING_NAME, r.LOCK_NAME}
        with os.scandir(self._root_fd) as entries:
            actual = set()
            for entry in entries:
                r.require(
                    len(actual) < r.MAX_EVENTS + 3 and entry.name not in actual,
                    "journal directory inventory exceeds bound or repeats a name",
                )
                actual.add(entry.name)
        r.require(actual == names, "unexpected/missing journal entry")
        genesis_files = {*r.PHASE_FILES, r.GENESIS_PAYLOAD}
        event_files = {*r.PHASE_FILES, r.EVENT_PAYLOAD}
        watch, content, size = {}, {}, 0
        for phase in phases:
            names = genesis_files if phase == r.GENESIS_NAME else event_files
            prefix = phase + "/"
            descriptor = os.open(
                phase, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._root_fd
            )
            try:
                directory = os.fstat(descriptor)
                directory_fingerprint = _fingerprint(directory)
                r.require(
                    stat.S_IMODE(directory.st_mode) == 0o555, "immutable phase directory differs"
                )
                watch[prefix] = directory_fingerprint
                with os.scandir(descriptor) as entries:
                    actual = set()
                    for entry in entries:
                        name = entry.name
                        r.require(
                            len(actual) < 3 and name in names and name not in actual,
                            "unexpected or duplicate immutable phase child",
                        )
                        actual.add(name)
                        value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                        r.require(
                            stat.S_ISREG(value.st_mode)
                            and value.st_nlink == 1
                            and stat.S_IMODE(value.st_mode) == 0o444
                            and 0 < value.st_size <= r.MAX_EVENT_BYTES,
                            "immutable phase file differs",
                        )
                        watch[prefix + name] = _fingerprint(value)
                        size += value.st_size
                        r.require(size <= r.MAX_JOURNAL_BYTES, "journal byte cap exceeded")
                        content[prefix + name] = _content_digest(descriptor, name, value)
                r.require(actual == names, "immutable phase payload inventory differs")
                r.require(
                    _fingerprint(os.fstat(descriptor)) == directory_fingerprint
                    and _fingerprint(os.stat(phase, dir_fd=self._root_fd, follow_symlinks=False))
                    == directory_fingerprint,
                    "phase directory changed during inventory",
                )
            finally:
                os.close(descriptor)
        for name, fingerprint in self._watch.items():
            r.require(watch.get(name) == fingerprint, "accepted immutable journal file drifted")
        for name, digest in self._content_watch.items():
            r.require(content.get(name) == digest, "accepted immutable journal content drifted")
        if not provisional:
            r.require(
                content.keys() - self._content_watch.keys() == expected_new.keys()
                and all(content.get(name) == digest for name, digest in pending),
                "new journal content differs from publication pins",
            )
        self._path_guard()
        r.require(self._pending_content is pending, "publication content handoff changed")
        self._watch, self._content_watch, self._bytes = watch, content, size
        self._pending_content = ()

    def _guard(self, *, allow_stopped=False):
        r.require(
            not self._closed and not self._poisoned and (allow_stopped or not self._stopped),
            "journal is stopped, closed or ambiguous",
        )
        self._callback_guard()
        self._capture()

    def _publish(self, name, payload_name, payload, artifact, metadata, predecessors):
        r.require(not self._pending_content, "previous publication content remains unaccepted")
        r.require(
            type(payload) is bytes and len(payload) <= r.MAX_EVENT_BYTES,
            "publication payload exceeds bound",
        )
        receipt = canonical_json_bytes(
            {
                "artifact": artifact,
                "metadata": metadata,
                "payloads": {payload_name: r.digest(payload)},
                "predecessor_seals": predecessors,
                "schema_version": 1,
                "status": "sealed",
            }
        )
        marker = checksum_manifest_bytes(
            {payload_name: r.digest(payload), "receipt.json": r.digest(receipt)}
        )
        phase_bytes = len(payload) + len(receipt) + len(marker)
        # A failed relocation can retain both the complete source and destination aliases.
        r.require(
            self._bytes + 2 * phase_bytes <= r.MAX_JOURNAL_BYTES,
            "publication would exceed journal byte admission",
        )
        self._path_guard()
        self._pending_empty()
        self._pending_content = tuple(
            (name + "/" + child, r.digest(raw))
            for child, raw in (
                (payload_name, payload),
                ("receipt.json", receipt),
                ("SHA256SUMS", marker),
            )
        )
        stage_descriptor = -1
        try:
            os.mkdir(name, 0o700, dir_fd=self._pending_fd)
            os.fsync(self._pending_fd)
            stage_descriptor = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._pending_fd
            )
            _write_retained(stage_descriptor, payload_name, payload)
            _write_retained(stage_descriptor, "receipt.json", receipt)
            _write_retained(stage_descriptor, "SHA256SUMS", marker)
            os.fchmod(stage_descriptor, 0o555)
            os.fsync(stage_descriptor)
            os.fsync(self._pending_fd)
            staged = verify_phase(
                self.root / r.PENDING_NAME / name,
                expected_artifact=artifact,
                expected_payload_paths=(payload_name,),
                expected_predecessor_seals=predecessors,
                expected_seal_sha256=r.digest(marker),
            )
            self._path_guard()
            relocate_phase_capability_noreplace_at(
                self._pending_fd,
                name,
                self._root_fd,
                name,
                expected=staged,
            )
            os.fsync(self._root_fd)
            self._path_guard()
            verified = verify_phase(
                self.root / name,
                expected_artifact=artifact,
                expected_payload_paths=(payload_name,),
                expected_predecessor_seals=predecessors,
                expected_seal_sha256=staged.seal_sha256,
            )
            r.require(
                verified.metadata_json == staged.metadata_json, "published phase metadata drifted"
            )
        except BaseException:
            self._poisoned = True
            raise
        finally:
            if stage_descriptor >= 0:
                os.close(stage_descriptor)

    def _append(self, kind, data):
        r.require(not self._stopped and not self._poisoned, "cannot append after terminal boundary")
        event = r.event_record(
            self._binding.sha256,
            self._checkpoint.event_count,
            self._checkpoint.head_sha256,
            kind,
            data,
        )
        r.exact_fields(data, r.EVENT_DATA_FIELDS[kind])
        raw = event.document()
        self._publish(
            r.event_name(raw["ordinal"]),
            r.EVENT_PAYLOAD,
            event.payload,
            r.EVENT_ARTIFACT,
            {
                "genesis_sha256": self._binding.sha256,
                "event_sha256": event.sha256,
                "ordinal": raw["ordinal"],
            },
            {"previous_event": raw["previous_event_sha256"]},
        )
        charged = self._checkpoint.charged_count
        if kind == "initial_import":
            charged += 64
        elif kind == "dispatch_intent":
            charged += 1
        self._checkpoint = r.JournalCheckpoint(
            self._binding.sha256, raw["ordinal"] + 1, event.sha256, charged
        )
        self._remember(event)
        self._capture()
        return event

    def _remember(self, event):
        """Incremental bookkeeping after publication or fresh independent recovery."""
        raw = event.document()
        kind, data = raw["kind"], raw["data"]
        if kind == "initial_import":
            self._terminal_count = 64
            self._external_ids.update(
                row["external_submission_id"] for row in data["source_document"]["rows"]
            )
        elif kind == "wave_seal":
            self._wave_index = data["wave_index"]
            self._requests = tuple(r.request_from_document(row) for row in data["requests"])
        elif kind == "dispatch_intent":
            self._intents[event.sha256] = r.DispatchRequest(
                self._binding.sha256,
                event.sha256,
                data["wave_index"],
                data["seat_index"],
                data["charge_index"],
                r.request_from_document(data["request"]),
            )
        elif kind == "submission_ack":
            external = data["authenticated"]["external_submission_id"]
            self._acks[data["intent_sha256"]] = external
            self._external_ids.add(external)
        elif kind == "terminal_response":
            self._terminal_count += 1
            self._terminal_intents.add(data["intent_sha256"])
        elif kind == "dispatch_fault":
            self._faults.add(data["intent_sha256"])
        elif kind == "stop":
            self._stopped = True

    def _enter(self):
        r.require(
            not self._closed and not self._stopped and not self._poisoned and not self._busy,
            "journal operation is terminal, ambiguous or reentrant",
        )
        self._busy = True
        try:
            self._guard()
        except BaseException:
            self._busy = False
            raise

    def _authenticate(self, kind, receipt, expected):
        receipt = r.receipt_bytes(receipt)
        r.document(expected)
        before = self._checkpoint
        self._guard()
        result = self._targets[0](kind, receipt, expected)
        self._guard()
        r.require(self._checkpoint == before, "checkpoint drifted during authentication")
        return r.document(result)

    @property
    def checkpoint(self):
        """Locally known durable minimum; external adoption/replay remains mandatory."""
        return self._checkpoint

    def import_initial(self, source_receipt, copy_receipt):
        self._enter()
        try:
            r.require(self._checkpoint.event_count == 0, "initial block cannot be imported twice")
            r.require(
                r.digest(r.receipt_bytes(source_receipt))
                == self._binding.initial_source_receipt_sha256
                and r.digest(r.receipt_bytes(copy_receipt))
                == self._binding.initial_copy_receipt_sha256,
                "initial raw receipt pin differs",
            )
            source = self._authenticate(
                "initial_source",
                source_receipt,
                r.initial_expected(self._binding, "initial_source"),
            )
            r.exact_fields(source, ("rows",))
            r.require(
                type(source["rows"]) is list and len(source["rows"]) == 64,
                "initial source must authenticate exactly 64 rows",
            )
            external = set()
            for request, row in zip(self._binding.initial_requests, source["rows"], strict=True):
                r.exact_fields(row, r.INITIAL_ROW_FIELDS)
                r.require(
                    row["request_sha256"] == request.sha256
                    and r.identifier(row["external_submission_id"])
                    and row["external_submission_id"] not in external
                    and r.pin(row["response_receipt_sha256"]),
                    "initial row provenance differs",
                )
                r.terminal_values(row["status"], row["objectives"])
                external.add(row["external_submission_id"])
            rows_sha256 = r.digest(r.canonical(source["rows"]))
            copy = self._authenticate(
                "initial_copy",
                copy_receipt,
                r.initial_expected(self._binding, "initial_copy", rows_sha256=rows_sha256),
            )
            r.exact_fields(copy, r.COPY_FIELDS)
            r.require(
                copy
                == {
                    "source_receipt_sha256": self._binding.initial_source_receipt_sha256,
                    "rows_sha256": rows_sha256,
                    "run_id": self._binding.run_id,
                    "seed": self._binding.seed,
                }
                and type(copy["seed"]) is int,
                "initial copy does not bind exact source rows/run/seed",
            )
            self._append(
                "initial_import",
                {
                    "source_receipt_hex": source_receipt.hex(),
                    "copy_receipt_hex": copy_receipt.hex(),
                    "source_document": source,
                    "copy_document": copy,
                },
            )
            self._guard()
            return self._checkpoint
        finally:
            self._busy = False

    def seal_wave(self, method_requests):
        self._enter()
        try:
            wave = self._wave_index + 1
            r.require(
                1 <= wave <= 28
                and self._terminal_count == 64 + 16 * (wave - 1)
                and self._checkpoint.charged_count == self._terminal_count,
                "cannot seal wave before exact prior complete history",
            )
            r.require(
                type(method_requests) is tuple and len(method_requests) == 14,
                "wave requires exactly 14 caller-supplied method seats",
            )
            reserved = tuple(row for pair in self._binding.reserves for row in pair)
            previous = (
                *self._binding.initial_requests,
                *(row.request for row in self._intents.values()),
            )
            forbidden = (*previous, *reserved)
            query_ids = {row.query_id for row in forbidden}
            sequences = {row.sequence for row in forbidden}
            identities = {row.identity.key for row in forbidden}
            for request in method_requests:
                self._binding.validate_request(request)
                r.require(
                    request.query_id not in query_ids
                    and request.sequence not in sequences
                    and request.identity.key not in identities,
                    "method/reserve/charge collision",
                )
                query_ids.add(request.query_id)
                sequences.add(request.sequence)
                identities.add(request.identity.key)
            requests = (*method_requests, *self._binding.reserves[wave - 1])
            self._append(
                "wave_seal", {"wave_index": wave, "requests": [row.document() for row in requests]}
            )
            self._guard()
            return self._checkpoint
        finally:
            self._busy = False

    def _accept_ack(self, intent_sha256, receipt):
        r.require(
            intent_sha256 in self._intents and intent_sha256 not in self._acks,
            "acknowledgement requires an unacknowledged durable intent",
        )
        request = self._intents[intent_sha256]
        ack = self._authenticate(
            "submission_ack", receipt, r.dispatch_expected(self._binding, request, "submission_ack")
        )
        r.exact_fields(ack, ("external_submission_id",))
        r.require(
            r.identifier(ack["external_submission_id"])
            and ack["external_submission_id"] not in self._external_ids,
            "external submission ID is invalid or already used in this run",
        )
        self._append(
            "submission_ack",
            {
                "intent_sha256": intent_sha256,
                "receipt_hex": r.receipt_bytes(receipt).hex(),
                "authenticated": ack,
            },
        )

    def dispatch_next(self):
        self._enter()
        try:
            seat = self._checkpoint.charged_count - (64 + 16 * (self._wave_index - 1))
            r.require(
                self._wave_index >= 1 and 0 <= seat < 16 and self._checkpoint.charged_count < 512,
                "no fresh sealed dispatch seat",
            )
            event = self._append(
                "dispatch_intent",
                {
                    "wave_index": self._wave_index,
                    "seat_index": seat,
                    "charge_index": self._checkpoint.charged_count,
                    "request": self._requests[seat].document(),
                },
            )
            request = self._intents[event.sha256]
            frozen = r.canonical(request.document())
            checkpoint = self._checkpoint
            stage = "permission"
            try:
                self._guard()
                allowed = self._targets[2](request)
                self._guard()
                r.require(
                    allowed is True, "original permission/deadline checkpoint denied dispatch"
                )
                r.require(
                    self._checkpoint == checkpoint and r.canonical(request.document()) == frozen,
                    "dispatch changed during original permission checkpoint",
                )
                stage = "transport"
                receipt = self._targets[1](request)
                self._guard()
                r.require(
                    self._checkpoint == checkpoint and r.canonical(request.document()) == frozen,
                    "dispatch changed during transport",
                )
                stage = "acknowledgement"
                self._accept_ack(event.sha256, receipt)
                self._guard()
                return self._checkpoint
            except BaseException as error:
                if not self._poisoned and event.sha256 not in self._faults:
                    try:
                        self._guard()
                        self._append(
                            "dispatch_fault",
                            {
                                "intent_sha256": event.sha256,
                                "stage": stage,
                                "error_type": type(error).__name__[:128],
                                "detail": "local failure evidence; external cause not independently proved",
                            },
                        )
                    except BaseException:
                        pass  # Durable intent remains the conservative minimum; never resubmit.
                if event.sha256 in self._acks:
                    raise RuntimeError(
                        "post-acknowledgement failure; charge and accepted ID remain"
                    ) from error
                raise DispatchUnresolved(
                    "durable dispatch remains spent; reconcile the same token"
                ) from error
        finally:
            self._busy = False

    def acknowledge(self, intent_sha256, receipt):
        """Accept externally obtained same-token evidence; never call a transport."""
        self._enter()
        try:
            self._accept_ack(intent_sha256, receipt)
            self._guard()
            return self._checkpoint
        finally:
            self._busy = False

    def record_terminal(self, intent_sha256, receipt):
        self._enter()
        try:
            r.require(
                intent_sha256 in self._acks and intent_sha256 not in self._terminal_intents,
                "terminal requires sole acknowledged, nonterminal intent",
            )
            request = self._intents[intent_sha256]
            r.require(
                request.charge_index == self._terminal_count,
                "terminal responses must follow charged order",
            )
            terminal = self._authenticate(
                "terminal_response",
                receipt,
                r.dispatch_expected(
                    self._binding,
                    request,
                    "terminal_response",
                    external_submission_id=self._acks[intent_sha256],
                ),
            )
            r.exact_fields(terminal, ("external_submission_id", "status", "objectives"))
            r.require(
                terminal["external_submission_id"] == self._acks[intent_sha256],
                "terminal changed the sole external submission ID",
            )
            r.terminal_values(terminal["status"], terminal["objectives"])
            self._append(
                "terminal_response",
                {
                    "intent_sha256": intent_sha256,
                    "receipt_hex": r.receipt_bytes(receipt).hex(),
                    "authenticated": terminal,
                },
            )
            self._guard()
            return self._checkpoint
        finally:
            self._busy = False

    def stop(self, reason="external_stop", detail="outer controller stopped this journal"):
        self._enter()
        try:
            r.require(
                type(reason) is str
                and reason in ("completed", "external_stop", "integrity_failure")
                and type(detail) is str
                and len(detail) <= 1024,
                "stop evidence differs",
            )
            if reason == "completed":
                r.require(
                    self._terminal_count == self._checkpoint.charged_count == 512,
                    "completion requires all 512 authenticated terminal rows",
                )
            self._append("stop", {"reason": reason, "detail": detail})
            self._guard(allow_stopped=True)
            return self._checkpoint
        finally:
            self._busy = False

    def close(self):
        if not self._closed:
            self._closed = True
            for name in ("_lock_fd", "_pending_fd", "_root_fd", "_parent_fd"):
                descriptor = getattr(self, name, -1)
                if descriptor >= 0:
                    os.close(descriptor)
                    setattr(self, name, -1)

    def __enter__(self):
        r.require(not self._closed, "journal is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

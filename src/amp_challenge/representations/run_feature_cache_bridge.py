"""One real fixed-shape session, feature-only private/public caches and paid ledger.

No model fitting, oracle access, restart, campaign scheduling or timing authority.
The controller supplies authenticated external pins and hard interruption. A
failed multi-chunk assembly retains completed feature children, never learner input.
"""

from __future__ import annotations

import json
import os
import stat
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from amp_challenge.generators.search.peptide_ga_tunable_v2_records import ChargedObservation
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot
from amp_challenge.representations.candidate_features import (
    CandidateFeatureRequest,
    _exclusive_write,
)
from amp_challenge.representations.peptide_esm import file_digest
from amp_challenge.representations.run_feature_cache_records import (
    ALIASES,
    BINDING_BYTES,
    CONTRACT_SHA256,
    FULL_ARMS,
    LAYOUT_SHA256,
    METADATA_BYTES,
    RUN_BYTES,
    SESSION_BYTES,
    SESSION_ENTRIES,
    FeatureAssemblyBinding,
    FeatureCacheReceipt,
    FeatureCounters,
    FeatureIntent,
    FeatureRunBinding,
    PrivateFeatureRelease,
    canonical,
    check_tr2_capacity_request,
    finite_clock,
    json_object,
    pin,
    public_document,
    representation_checked,
    require,
    sequences_checked,
    sha256,
)
from amp_challenge.representations.run_feature_cache_verify import (
    inspect_feature_tree,
    load_saved_feature_batch,
)
from amp_challenge.representations.warm_candidate_features import (
    WarmCandidateFeatureSession,
    warm_source_identity,
)


def row_sha256(row):
    values = np.asarray(row, dtype="<f8", order="C")
    require(
        values.ndim == 1 and len(values) in (321, 353) and np.isfinite(values).all(),
        "feature row numerical shape/support differs",
    )
    return sha256(
        b"amp/run-feature-row/v1\0"
        + canonical({"dtype": "<f8", "width": len(values)})
        + values.tobytes()
    )


def _metadata_bytes(path):
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and 0 < before.st_size <= BINDING_BYTES,
        "feature metadata file is unsafe or oversized",
    )
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        require(
            (before.st_dev, before.st_ino, before.st_size)
            == (opened.st_dev, opened.st_ino, opened.st_size),
            "feature metadata changed before read",
        )
        payload = stream.read(BINDING_BYTES + 1)
    after = path.lstat()
    require(
        len(payload) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "feature metadata changed during bounded read",
    )
    return payload


@dataclass(frozen=True, slots=True)
class FeatureRows:
    """Requested rows only. No private origins, cache-hit flags or shared heads."""

    sequences: tuple[str, ...]
    representation: str
    matrix: np.ndarray
    receipt_payload: bytes
    receipt_sha256: str

    def __post_init__(self):
        sequences_checked(self.sequences, empty=True, maximum=512)
        _, width = representation_checked(self.representation)
        require(
            type(self.receipt_payload) is bytes
            and pin(self.receipt_sha256)
            and sha256(self.receipt_payload) == self.receipt_sha256,
            "requested feature receipt differs",
        )
        document = json_object(self.receipt_payload)
        require(
            type(self.matrix) is np.ndarray
            and self.matrix.shape == (len(self.sequences), width)
            and self.matrix.dtype.kind in "iuf",
            "requested feature matrix shape or real dtype differs",
        )
        values = np.array(self.matrix, dtype=np.float64, copy=True)
        require(
            values.shape == (len(self.sequences), width) and np.isfinite(values).all(),
            "requested feature matrix differs",
        )
        wanted = public_document(
            self.sequences,
            self.representation,
            tuple(row_sha256(row) for row in values),
            context_sha256=document["objective_context_sha256"],
            history_sha256=document["history_sha256"],
        )
        require(canonical(wanted) == self.receipt_payload, "requested feature row seal differs")
        values.setflags(write=False)
        object.__setattr__(self, "matrix", values)


@dataclass(frozen=True, slots=True)
class _Row:
    raw321: np.ndarray
    raw353: np.ndarray
    origin_payload: bytes

    def check(self):
        origin = json_object(self.origin_payload)
        unsealed = {key: value for key, value in origin.items() if key != "origin_sha256"}
        require(sha256(canonical(unsealed)) == origin["origin_sha256"], "feature origin seal drift")
        for name, values in (("esm_length", self.raw321), ("esm_length_spectral", self.raw353)):
            require(
                type(values) is np.ndarray
                and not values.flags.writeable
                and row_sha256(values) == origin["raw_sha256s"][name],
                "cached raw feature drift",
            )
        require(np.array_equal(self.raw321, self.raw353[:321]), "raw feature alias values differ")
        return origin

    def values(self, representation):
        self.check()
        return self.raw321 if representation == "esm_length" else self.raw353


class VisibleFeaturePort:
    """Capability-narrow dataflow view, not a same-account security sandbox."""

    __slots__ = ("__request",)

    def __init__(self, request):
        self.__request = request

    def raw_rows(self, sequences, representation, intent):
        return self.__request(sequences, representation, intent)


class ScopedVisibleFeaturePort:
    """Fixed public wave binding; private ledger-head updates remain controller-owned."""

    __slots__ = ("__abort", "__binding_payload", "__check", "__request")

    def __init__(self, request, check, abort, binding_payload):
        self.__request, self.__check = request, check
        self.__abort = abort
        json_object(binding_payload)
        self.__binding_payload = binding_payload

    @property
    def binding_payload(self):
        return self.__binding_payload

    def raw_rows(self, sequences, representation):
        return self.__request(sequences, representation)

    def check(self):
        self.__check()

    def abort(self, error):
        self.__abort(error)


class RunFeatureBridge:
    def __init__(self, binding: FeatureRunBinding, *, monotonic=time.monotonic):
        require(type(binding) is FeatureRunBinding, "exact run-feature binding required")
        binding.__post_init__()
        self.binding = binding
        self.__binding_payload = canonical(binding.document())
        self.__visible = {}
        self.__private = {}
        self.__receipts = ()
        self.__counters = FeatureCounters()
        self.__accepted_head = binding.sha256
        self.__evidence_head = binding.sha256
        self.__release_head = binding.sha256
        self.__released = frozenset()
        self.__session = None
        self.__session_payload = None
        self.__clock = monotonic
        self.__last_clock = binding.original_epoch
        self.__terminal = False
        self.last_failure = None
        self.__state_identity = self._state_identity()
        run_root = Path(binding.run_root)
        require(
            run_root.is_dir() and run_root.resolve(strict=True) == run_root,
            "feature run root must already be a real controller-owned directory",
        )
        require(
            not binding.session_root.exists() and not binding.audit_root.exists(),
            "feature session/audit roots must be exclusive and unused",
        )
        require(
            not any(
                path.name.startswith(("feature-session-", "feature-audit-"))
                for path in run_root.iterdir()
            ),
            "one-session bridge cannot reset a run's earlier feature session",
        )
        binding.audit_root.mkdir(mode=0o700)
        self._execute("open", None, (), None, self._open)

    @property
    def accepted_head(self):
        self._guard()
        return self.__accepted_head

    @property
    def evidence_head(self):
        return self.__evidence_head

    @property
    def counters(self):
        return self.__counters

    @property
    def receipts(self):
        """Controller-private audit accessor; never handed to consumer views."""
        return self.__receipts

    def consumer(self):
        self._guard()
        return VisibleFeaturePort(self.raw_rows)

    def scoped_consumer(self, intent):
        self._guard()
        self._valid_intent(intent)
        original = canonical(intent.document())
        ordinal = [intent.logical_ordinal]
        public = {
            key: value
            for key, value in intent.document().items()
            if key != "expected_previous_head"
        }
        public["feature_source_sha256"] = sha256(
            canonical(
                {
                    "warm_source": json_object(self.binding.warm_source_payload),
                    "runtime_sha256": sha256(self.binding.runtime_payload),
                    "model_sha256": self.binding.model_sha256,
                    "layout_sha256": LAYOUT_SHA256,
                }
            )
        )
        public["provider_sha256"] = sha256(self.binding.implementation_source_payload)

        def check():
            try:
                self._guard()
                require(canonical(intent.document()) == original, "feature scoped intent drifted")
                tx = self._new_tx("raw", intent, (), None, None)
                self._sources()
                self._clock(tx, "final")
                self._guard()
                require(
                    canonical(intent.document()) == original,
                    "feature scoped intent changed at clock",
                )
            except BaseException as error:
                self.abort(error, intent=intent)
                raise

        def request(sequences, representation):
            try:
                require(canonical(intent.document()) == original, "feature scoped intent drifted")
            except BaseException as error:
                self.abort(error, intent=intent)
                raise
            current = replace(
                intent, expected_previous_head=self.__accepted_head, logical_ordinal=ordinal[0]
            )
            ordinal[0] += 1
            return self.raw_rows(sequences, representation, current)

        return ScopedVisibleFeaturePort(
            request, check, lambda error: self.abort(error, intent=intent), canonical(public)
        )

    def abort(self, error, *, intent=None):
        """Controller/view failure: terminate without a new feature opportunity.

        A failed close record is evidence only, not a successful close or a
        replacement feature call. Completed child features remain auditable.
        """
        if self.__terminal:
            return
        closing = None
        with suppress(BaseException):
            if type(intent) is FeatureIntent:
                intent.__post_init__()
                closing = replace(
                    intent, purpose="close", expected_previous_head=self.__accepted_head
                )

        def fail(tx, visible, private):
            del tx, visible, private
            raise error

        with suppress(BaseException):
            self._execute("close", closing, (), None, fail)
        self.__terminal = True
        if self.last_failure is None:
            self.last_failure = {
                "failure": f"{type(error).__name__}: {error}",
                "accepted_head": self.__accepted_head,
            }

    def _state_identity(self):
        require(
            type(self.__counters) is FeatureCounters and type(self.__receipts) is tuple,
            "feature committed state types drifted",
        )
        self.__counters.__post_init__()
        inventory = {}
        for role, mapping in (("visible", self.__visible), ("private", self.__private)):
            require(type(mapping) is dict, "feature committed cache type drifted")
            inventory[role] = []
            for sequence, row in sorted(mapping.items()):
                require(type(row) is _Row, "feature committed row type drifted")
                origin = row.check()
                require(origin["sequence"] == sequence, "feature committed row identity drifted")
                inventory[role].append(origin)
        receipts = []
        for receipt in self.__receipts:
            require(type(receipt) is FeatureCacheReceipt, "feature committed receipt type drifted")
            # Full schema/canonical validation happens at seal and before
            # acceptance. The immutable payload's freshly verified digest is
            # part of the committed state identity below, so replacing even a
            # consistently resealed receipt still changes that identity. Re-
            # parsing all historical inventory JSON adds no mutation detection.
            receipt.__post_init__()
            receipts.append(receipt.sha256)
        return sha256(
            canonical(
                {
                    "counters": self.__counters.document(),
                    "rows": inventory,
                    "receipts": receipts,
                    "accepted": self.__accepted_head,
                    "evidence": self.__evidence_head,
                    "release": self.__release_head,
                    "released": sorted(self.__released),
                }
            )
        )

    def _guard(self):
        require(
            type(self.binding) is FeatureRunBinding
            and canonical(self.binding.document()) == self.__binding_payload,
            "feature external binding drifted",
        )
        require(not self.__terminal, "feature bridge is permanently stopped")
        require(self._state_identity() == self.__state_identity, "feature accepted state drifted")

    def _sources(self):
        binding = self.binding
        require(
            warm_source_identity(
                Path(binding.repository),
                binding.expected_commit,
                *([binding.tr2_grouped_capacity] if binding.tr2_grouped_capacity else []),
            )
            == json_object(binding.warm_source_payload),
            "feature frozen source changed",
        )
        for relative, expected in json_object(binding.implementation_source_payload).items():
            require(
                file_digest(Path(binding.repository) / relative) == expected,
                "feature consumed implementation source changed",
            )

    def _clock(self, tx, phase):
        now = finite_clock(self.__clock())
        require(
            now >= self.__last_clock and now >= self.binding.original_epoch,
            "feature original monotonic epoch moved backwards",
        )
        deadline = self.binding.original_deadline
        if tx["intent"] is not None:
            deadline = min(deadline, tx["intent"]["original_effective_deadline"])
        require(deadline - now <= 7200, "feature original remaining allowance differs")
        self.__last_clock = now
        tx["timings"].append({"phase": phase, "monotonic": now})
        if now >= deadline:
            raise TimeoutError("feature original deadline exceeded")
        return now

    def _inventory(self, *, pending_path=None, pending_bytes=0):
        inventory = inspect_feature_tree(Path(self.binding.run_root))
        prefix = self.binding.session_root.name + "/"
        entries = inventory["entries"]
        session_entries = [row for row in entries if row["path"].startswith(prefix)]
        session_bytes = sum(row.get("bytes", 0) for row in session_entries)
        require(
            session_bytes <= SESSION_BYTES and len(session_entries) <= SESSION_ENTRIES,
            "feature session final artifact bounds exceeded",
        )
        require(
            inventory["regular_bytes"] + pending_bytes <= RUN_BYTES,
            "feature aggregate output budget exceeded",
        )
        result = {
            "entries": entries,
            "regular_bytes": inventory["regular_bytes"],
            "session_bytes": session_bytes,
            "session_entries": len(session_entries),
            "pending_final_path": pending_path,
            "pending_final_bytes": pending_bytes,
        }
        require(len(canonical(result)) <= METADATA_BYTES, "feature inventory metadata unsupported")
        return result

    def _session_record(self):
        root = self.binding.session_root
        config_path = root / "session.json"
        if not config_path.exists():
            return None
        payload = _metadata_bytes(config_path)
        config = json_object(payload)
        require(
            config["run_id"] == self.binding.run_id
            and config["session_id"] == self.binding.session_id
            and config["source"] == json_object(self.binding.warm_source_payload)
            and config["job_id"] == self.binding.job_id
            and config.get("feature_layout_sha256") == LAYOUT_SHA256
            and config.get("tr2_grouped_capacity") == self.binding.tr2_grouped_capacity
            and config.get("maximum_requests") == self.binding.admission_limit,
            "feature live session identity differs",
        )
        if self.__session_payload is not None:
            require(payload == self.__session_payload, "feature original session config changed")
        return {
            "config": config,
            "config_sha256": sha256(payload),
            "ready_sha256": file_digest(root / "ready.json")
            if (root / "ready.json").is_file()
            else None,
            "complete_sha256": file_digest(root / "COMPLETE.json")
            if (root / "COMPLETE.json").is_file()
            else None,
            "failed_sha256": file_digest(root / "FAILED.json")
            if (root / "FAILED.json").is_file()
            else None,
        }

    def _session_memory(self, tx):
        session = self.__session
        require(type(session) is WarmCandidateFeatureSession, "feature live session type drifted")
        config = json_object(self.__session_payload)
        require(
            type(session.config) is dict
            and canonical(session.config) == self.__session_payload
            and session.config_sha256 == sha256(self.__session_payload)
            and type(session.source) is dict
            and canonical(session.source) == self.binding.warm_source_payload
            and session.expected_commit == self.binding.expected_commit
            and session.repository == Path(self.binding.repository)
            and session.bundle == Path(self.binding.bundle)
            and session.output_root == self.binding.session_root
            and type(session.job_id) is str
            and session.job_id == self.binding.job_id,
            "feature live session immutable binding drifted",
        )
        require(
            type(session.deadline) in (int, float)
            and finite_clock(session.deadline) == config["deadline_monotonic"],
            "feature live session original deadline drifted",
        )
        invocations = []
        last_head = session.config_sha256
        for record in (*[receipt.document() for receipt in self.__receipts], tx):
            batch = record["batch"]
            if batch is not None and batch["returned"]:
                invocations.append(batch["invocation_sha256"])
                last_head = batch["response_sha256"]
        require(
            type(session.ordinal) is int
            and session.ordinal == len(invocations)
            and type(session.head) is str
            and session.head == last_head
            and type(session.invocation_sha256s) is list
            and session.invocation_sha256s == invocations
            and type(session.batch_ids) is set
            and session.batch_ids == {f"feature-{index:04d}" for index in range(len(invocations))}
            and type(session.closed) is bool
            and session.closed == (tx["kind"] == "close"),
            "feature live session committed response inventory drifted",
        )

    def _new_tx(self, kind, intent, sequences, representation, parent_assembly):
        intent_document = None
        with suppress(BaseException):
            if type(intent) is FeatureIntent:
                intent_document = intent.document()
        return {
            "artifact": "run_feature_cache_operation_v1",
            "contract_sha256": CONTRACT_SHA256,
            "binding": json_object(self.__binding_payload),
            "operation_ordinal": len(self.__receipts),
            "kind": kind,
            "status": "failed",
            "previous_evidence_head": self.__evidence_head,
            "previous_accepted_head": self.__accepted_head,
            "intent": intent_document,
            "parent_assembly": parent_assembly,
            "sequences": list(sequences),
            "representation": representation,
            "staging": {"started": None, "dispatch": None},
            "batch": None,
            "session": None,
            "release": None,
            "assembly": None,
            "row_origins": [],
            "scatter": [],
            "public_payload": None,
            "public_sha256": None,
            "before_counters": self.__counters.document(),
            "after_counters": self.__counters.document(),
            "inventory_before": None,
            "inventory_after": None,
            "timings": [],
            "failure": None,
            "oracle_calls": 0,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
        }

    def _write_marker(self, tx, suffix, document):
        path = self.binding.audit_root / f"operation-{tx['operation_ordinal']:06d}.{suffix}.json"
        payload = canonical(document)
        require(len(payload) <= METADATA_BYTES, "feature staging metadata unsupported")
        require(
            self._inventory()["regular_bytes"] + len(payload) <= RUN_BYTES,
            "feature staging publication exceeds run allowance",
        )
        _exclusive_write(path, payload)
        return {
            "path": path.relative_to(self.binding.run_root).as_posix(),
            "sha256": sha256(payload),
        }

    def _valid_intent(self, intent, *, allow_parent=False):
        require(type(intent) is FeatureIntent, "exact externally bound feature intent required")
        intent.__post_init__()
        require(
            intent.objective_context_sha256 == self.binding.objective_context_sha256
            and self.binding.original_epoch
            < intent.original_effective_deadline
            <= self.binding.original_deadline,
            "feature context/original deadline differs",
        )
        require(
            allow_parent or intent.expected_previous_head == self.__accepted_head,
            "feature expected accepted predecessor differs",
        )

    def _charge_logical(self, tx):
        counters = tx["after_counters"]
        if tx["parent_assembly"] is not None and not any(
            canonical(receipt.document()["parent_assembly"]) == canonical(tx["parent_assembly"])
            for receipt in self.__receipts
        ):
            counters["assembly_opportunities"] += 1
        if tx["kind"] in ("raw", "preload"):
            candidate = tx["intent"]["purpose"] == "candidate"
            counters["logical_opportunities"] += 1
            counters["candidate_opportunities" if candidate else "auxiliary_opportunities"] += 1
            if candidate:
                wave = tx["intent"]["round_index"]
                require(wave <= 28, "terminal round has no candidate opportunity")
                counters["candidate_wave_counts"] = list(counters["candidate_wave_counts"])
                counters["candidate_wave_counts"][wave - 1] += 1
                if self.binding.arm_id in FULL_ARMS:
                    counters["full_accounted_requests"] += 1
                total, per_wave = self.binding.candidate_limits
                require(
                    counters["candidate_opportunities"] <= total
                    and counters["candidate_wave_counts"][wave - 1] <= per_wave,
                    "feature existing candidate opportunity cap exceeded",
                )
        elif tx["kind"] == "assemble" and not tx["assembly"]["child_receipt_sha256s"]:
            counters["assembly_opportunities"] += 1
        require(
            self.binding.arm_id not in FULL_ARMS or counters["full_accounted_requests"] <= 128,
            "feature full-method run request cap exceeded",
        )

    @staticmethod
    def _counters_from(document):
        fields = dict(document)
        fields["candidate_wave_counts"] = tuple(fields["candidate_wave_counts"])
        result = FeatureCounters(**fields)
        result.__post_init__()
        return result

    def _publish(self, tx, *, failure=False):
        suffix = ".failure" if failure else ""
        final = self.binding.audit_root / f"operation-{tx['operation_ordinal']:06d}{suffix}.json"
        relative = final.relative_to(self.binding.run_root).as_posix()
        tx["inventory_after"] = self._inventory(pending_path=relative)
        receipt = None
        for _ in range(8):
            receipt = FeatureCacheReceipt.seal(tx)
            size = len(receipt.payload)
            if size == tx["inventory_after"]["pending_final_bytes"]:
                break
            tx["inventory_after"]["pending_final_bytes"] = size
        else:
            raise ValueError("feature pending receipt size failed to stabilize")
        require(
            tx["inventory_after"]["regular_bytes"] + len(receipt.payload) <= RUN_BYTES,
            "feature final receipt exceeds aggregate run allowance",
        )
        _exclusive_write(final, receipt.payload)
        actual = self._inventory()
        expected = list(tx["inventory_after"]["entries"])
        expected.append(
            {
                "path": relative,
                "type": "file",
                "bytes": len(receipt.payload),
                "sha256": receipt.sha256,
            }
        )
        require(
            actual["entries"] == sorted(expected, key=lambda row: row["path"]),
            "feature post-publication inventory differs",
        )
        require(
            actual["regular_bytes"]
            == tx["inventory_after"]["regular_bytes"] + len(receipt.payload),
            "feature actual final byte accounting differs",
        )
        return receipt

    def _execute(
        self,
        kind,
        intent,
        sequences,
        representation,
        action,
        *,
        parent_assembly=None,
        allow_parent=False,
        final_guard=None,
        assembly=None,
    ):
        # Terminal reuse is not a new failed attempt and must not append files,
        # re-close the worker, or supersede a previously accepted COMPLETE.
        require(not self.__terminal, "feature bridge is permanently stopped")
        tx = self._new_tx(kind, intent, sequences, representation, parent_assembly)
        tx["assembly"] = assembly
        staged_visible, staged_private = dict(self.__visible), dict(self.__private)
        result = None
        try:
            self._guard()
            if kind != "open":
                self._valid_intent(intent, allow_parent=allow_parent)
            self._charge_logical(tx)
            self._clock(tx, "entry")
            self._guard()
            self._sources()
            self._clock(tx, "source")
            tx["inventory_before"] = self._inventory()
            marker = {
                key: tx[key]
                for key in (
                    "operation_ordinal",
                    "kind",
                    "intent",
                    "parent_assembly",
                    "sequences",
                    "representation",
                    "before_counters",
                    "previous_evidence_head",
                    "previous_accepted_head",
                )
            }
            marker.update(
                artifact="run_feature_operation_started_v1",
                binding_sha256=self.binding.sha256,
                monotonic=tx["timings"][0]["monotonic"],
            )
            tx["staging"]["started"] = self._write_marker(tx, "started", marker)
            result = action(tx, staged_visible, staged_private)
            tx["session"] = self._session_record()
            self._session_memory(tx)
            tx["after_counters"]["visible_rows"] = len(staged_visible)
            tx["after_counters"]["private_rows"] = len(staged_private)
            tx["status"] = "completed"
            self._clock(tx, "serialize")
            self._guard()
            self._sources()
            receipt = self._publish(tx)
            sealed_payload = receipt.payload
            sealed_timings = receipt.document()["timings"]
            self._sources()
            # Last external clock first, then only callback-free identity checks.
            self._clock(tx, "final")
            self._guard()
            self._session_memory(tx)
            require(
                (type(intent) is FeatureIntent and intent.document() == tx["intent"])
                or (kind == "open" and intent is None),
                "feature final intent drifted",
            )
            receipt.document()
            require(
                type(receipt.payload) is bytes
                and receipt.payload == sealed_payload
                and canonical(tx | {"timings": sealed_timings}) == sealed_payload,
                "feature receipt changed after seal",
            )
            if final_guard is not None:
                final_guard()
            if result is not None:
                require(type(result) is FeatureRows, "feature returned record type differs")
                result.__post_init__()
                require(
                    result.sequences == tuple(tx["sequences"])
                    and type(result.representation) is str
                    and result.representation == tx["representation"]
                    and result.receipt_payload == canonical(tx["public_payload"])
                    and result.receipt_sha256 == tx["public_sha256"],
                    "feature returned result differs from original sealed operation",
                )
                require(
                    all(
                        np.array_equal(
                            result.matrix[index], staged_visible[seq].values(result.representation)
                        )
                        for index, seq in enumerate(result.sequences)
                    ),
                    "feature returned rows differ from staged accepted cache",
                )
            self.__visible, self.__private = staged_visible, staged_private
            self.__counters = self._counters_from(tx["after_counters"])
            self.__receipts += (receipt,)
            self.__accepted_head = self.__evidence_head = receipt.sha256
            if tx["release"] is not None:
                self.__release_head = sha256(canonical(tx["release"]))
                self.__released |= frozenset(tx["release"]["revealed_sequence_ids"])
            self.__state_identity = self._state_identity()
            return result if result is not None else receipt
        except BaseException as error:
            self.__terminal = True
            if self.__session is not None:
                with suppress(BaseException):
                    self.__session.__exit__(type(error), error, None)
            tx["status"] = "failed"
            tx["failure"] = f"{type(error).__name__}: {error}"
            for key in (
                "acquired_origins",
                "visible_rows",
                "private_rows",
                "released_rows",
                "cache_only_opportunities",
            ):
                tx["after_counters"][key] = tx["before_counters"][key]
            self.last_failure = tx
            with suppress(BaseException):
                tx["session"] = self._session_record()
                # A late failure keeps its earlier complete publication, never
                # overwrites it. The separate failure evidence dominates it.
                failure_receipt = self._publish(tx, failure=True)
                self.__receipts += (failure_receipt,)
                self.__evidence_head = failure_receipt.sha256
            self.__counters = self._counters_from(tx["after_counters"])
            raise

    def _open(self, tx, visible, private):
        del visible, private
        now = self._clock(tx, "session_start")
        binding = self.binding
        self.__session = WarmCandidateFeatureSession(
            run_id=binding.run_id,
            session_id=binding.session_id,
            repository=Path(binding.repository),
            expected_commit=binding.expected_commit,
            bundle=Path(binding.bundle),
            output_root=binding.session_root,
            timeout_seconds=binding.original_deadline - now,
            maximum_requests=binding.admission_limit,
            **(
                {"tr2_grouped_capacity": binding.tr2_grouped_capacity}
                if binding.tr2_grouped_capacity
                else {}
            ),
            fixed_shape=True,
        )
        self.__session_payload = _metadata_bytes(binding.session_root / "session.json")
        self._clock(tx, "session_ready")

    def _acquire(self, tx, sequences):
        counters = tx["after_counters"]
        require(
            counters["physical_attempts"] < self.binding.admission_limit,
            "feature one-session/arm physical cap exhausted",
        )
        if self.binding.arm_id in FULL_ARMS and tx["intent"]["purpose"] != "candidate":
            require(
                counters["full_accounted_requests"] < 128, "feature auxiliary run cap exhausted"
            )
        profile = self.binding.tr2_grouped_capacity
        check_tr2_capacity_request(profile, counters["physical_attempts"], sequences)
        if profile:
            require(
                tx["intent"]["purpose"]
                == ("private_preload" if counters["physical_attempts"] == 0 else "candidate"),
                "TR2 capacity permits preload/candidate acquisitions only",
            )
            require(
                counters["acquired_origins"] + len(sequences) <= self.binding.origin_limit,
                "feature acquired-origin bound exhausted",
            )
        ordinal = self.__session.ordinal
        request = CandidateFeatureRequest(self.binding.run_id, f"feature-{ordinal:04d}", sequences)
        previous = self.__session.head
        next_counters = dict(counters)
        next_counters["physical_attempts"] += 1
        next_counters["real_rows_dispatched"] += len(sequences)
        next_counters["padded_rows_dispatched"] += 128
        if self.binding.arm_id in FULL_ARMS and tx["intent"]["purpose"] != "candidate":
            next_counters["full_accounted_requests"] += 1
        tx["batch"] = {
            "root": (self.binding.session_root / f"batch-{ordinal:04d}")
            .relative_to(self.binding.run_root)
            .as_posix(),
            "ordinal": ordinal,
            "expected_previous_head": previous,
            "request_sha256": sha256(request.payload),
            "command_sha256": None,
            "response_sha256": None,
            "invocation_sha256": None,
            "manifest_sha256": None,
            "dispatched": False,
            "returned": False,
        }
        stamp = self._clock(tx, "dispatch_prepare")
        marker = {
            "artifact": "run_feature_dispatch_v1",
            "binding_sha256": self.binding.sha256,
            "operation_ordinal": tx["operation_ordinal"],
            "batch_ordinal": ordinal,
            "request": json.loads(request.payload),
            "request_sha256": sha256(request.payload),
            "session_sha256": self.__session.config_sha256,
            "expected_previous_head": previous,
            "counters_after_admission": next_counters,
            "monotonic": stamp,
        }
        tx["staging"]["dispatch"] = self._write_marker(tx, "dispatch", marker)
        self._clock(tx, "before_request")
        self._guard()
        counters.update(next_counters)
        tx["batch"]["dispatched"] = True
        try:
            invocation = self.__session.request(request)
        except BaseException:
            counters["physical_failed"] += 1
            raise
        counters["physical_completed"] += 1
        tx["batch"]["returned"] = True
        self._clock(tx, "after_request")
        batch_root = self.binding.session_root / f"batch-{ordinal:04d}"
        for field, relative in (
            ("command_sha256", "command.json"),
            ("response_sha256", "response.json"),
            ("invocation_sha256", "invocation.json"),
            ("manifest_sha256", "features/manifest.json"),
        ):
            tx["batch"][field] = file_digest(batch_root / relative)
        require(
            invocation["response_sha256"] == tx["batch"]["response_sha256"],
            "feature saved response differs from returned invocation",
        )
        checked, arrays, _, _ = load_saved_feature_batch(
            self.binding.session_root,
            expected_session=json_object(self.__session_payload),
            ordinal=ordinal,
            expected_previous_head=previous,
            expected_response_sha256=tx["batch"]["response_sha256"],
            expected_runtime=json_object(self.binding.runtime_payload),
            expected_model_sha256=self.binding.model_sha256,
        )
        require(
            checked.payload == request.payload
            and self.__session.ordinal == ordinal + 1
            and self.__session.head == tx["batch"]["response_sha256"],
            "feature returned session/request drift",
        )
        self._clock(tx, "reconstruct")
        rows = {}
        for index, sequence in enumerate(sequences):
            raw321 = np.array(arrays["esm_length"][index], dtype=np.float64, copy=True)
            raw353 = np.array(arrays["esm_length_spectral"][index], dtype=np.float64, copy=True)
            raw321.setflags(write=False)
            raw353.setflags(write=False)
            origin = {
                "sequence": sequence,
                "sequence_id": sha256(sequence.encode("ascii")),
                "acquisition_operation": tx["operation_ordinal"],
                "batch_ordinal": ordinal,
                "row": index,
                "manifest_sha256": tx["batch"]["manifest_sha256"],
                "raw_sha256s": {
                    "esm_length": row_sha256(raw321),
                    "esm_length_spectral": row_sha256(raw353),
                },
            }
            origin["origin_sha256"] = sha256(canonical(origin))
            rows[sequence] = _Row(raw321, raw353, canonical(origin))
            rows[sequence].check()
        counters["acquired_origins"] += len(rows)
        require(
            counters["acquired_origins"] <= self.binding.origin_limit,
            "feature acquired origin bound exceeded",
        )
        return rows

    @staticmethod
    def _result(tx, sequences, representation, rows):
        unique = tuple(dict.fromkeys(sequences))
        tx["row_origins"] = [rows[sequence].check() for sequence in unique]
        indices = {sequence: index for index, sequence in enumerate(unique)}
        tx["scatter"] = [indices[sequence] for sequence in sequences]
        _, width = representation_checked(representation)
        matrix = (
            np.stack([rows[sequence].values(representation) for sequence in sequences])
            if sequences
            else np.empty((0, width))
        )
        public = public_document(
            sequences,
            representation,
            tuple(row_sha256(row) for row in matrix),
            context_sha256=tx["intent"]["objective_context_sha256"],
            history_sha256=tx["intent"]["history_sha256"],
        )
        tx["public_payload"] = public
        tx["public_sha256"] = sha256(canonical(public))
        return FeatureRows(
            sequences, representation, matrix, canonical(public), tx["public_sha256"]
        )

    def raw_rows(self, sequences, representation, intent, *, parent_assembly=None):
        try:
            return self._raw_rows(
                sequences, representation, intent, parent_assembly=parent_assembly
            )
        except BaseException as error:
            self.abort(error, intent=intent)
            raise

    def _raw_rows(self, sequences, representation, intent, *, parent_assembly=None):
        sequences_checked(sequences)
        representation_checked(representation)
        require(type(intent) is FeatureIntent, "raw feature intent type differs")
        intent.__post_init__()
        require(
            intent.purpose in ("candidate", "charged", "initial", "terminal"),
            "raw feature purpose differs",
        )
        require(
            intent.purpose != "candidate" or intent.round_index <= 28,
            "candidate feature round is terminal, not an admitted opportunity",
        )

        def action(tx, visible, private):
            profile = self.binding.tr2_grouped_capacity
            if profile:
                require(
                    len(private) == 120 and all(value in visible for value in tuple(private)[:64]),
                    "TR2 capacity requires shared preload and released initial rows",
                )
                require(
                    intent.purpose != "candidate" or len(sequences) <= 80,
                    "TR2 grouped candidate request exceeds80 rows",
                )
            misses = tuple(dict.fromkeys(seq for seq in sequences if seq not in visible))
            if misses:
                require(
                    not profile or intent.purpose == "candidate",
                    "TR2 capacity charged fits must reuse cache",
                )
                visible.update(self._acquire(tx, misses))
            else:
                tx["after_counters"]["cache_only_opportunities"] += 1
            return self._result(tx, sequences, representation, visible)

        return self._execute(
            "raw", intent, sequences, representation, action, parent_assembly=parent_assembly
        )

    def preload_private(self, sequences, intent):
        try:
            return self._preload_private(sequences, intent)
        except BaseException as error:
            self.abort(error, intent=intent)
            raise

    def _preload_private(self, sequences, intent):
        sequences_checked(sequences, unique=True)
        require(type(intent) is FeatureIntent, "private preload intent type differs")
        intent.__post_init__()
        require(intent.purpose == "private_preload", "private preload purpose differs")
        if self.binding.tr2_grouped_capacity:
            require(
                not self.__private and self.__counters.physical_attempts == 0,
                "TR2 capacity requires one initial shared preload",
            )
            check_tr2_capacity_request(self.binding.tr2_grouped_capacity, 0, sequences)

        def action(tx, visible, private):
            del visible
            misses = tuple(seq for seq in sequences if seq not in private)
            if misses:
                private.update(self._acquire(tx, misses))
            else:
                tx["after_counters"]["cache_only_opportunities"] += 1
            tx["row_origins"] = [private[seq].check() for seq in sequences]

        return self._execute("preload", intent, sequences, None, action)

    def release_private(self, release, intent, *, expected_release):
        try:
            return self._release_private(release, intent, expected_release=expected_release)
        except BaseException as error:
            self.abort(error, intent=intent)
            raise

    def _release_private(self, release, intent, *, expected_release):
        require(
            type(release) is PrivateFeatureRelease
            and type(expected_release) is PrivateFeatureRelease,
            "exact private release authority required",
        )
        require(
            release.document() == expected_release.document(), "external private release differs"
        )
        original = canonical(release.document())

        def action(tx, visible, private):
            require(
                intent.purpose == "release"
                and release.run_id == self.binding.run_id
                and release.history_sha256 == intent.history_sha256
                and release.objective_context_sha256 == intent.objective_context_sha256
                and release.expected_previous_release_head == self.__release_head,
                "private release history/context/head differs",
            )
            indexed = {sha256(seq.encode("ascii")): seq for seq in private}
            require(
                set(release.revealed_sequence_ids) <= set(indexed),
                "private release subset not in vault",
            )
            require(
                not set(release.revealed_sequence_ids) & self.__released,
                "private release repeats an already released subset",
            )
            for sequence_id in release.revealed_sequence_ids:
                seq = indexed[sequence_id]
                if seq in visible:
                    require(
                        all(
                            np.array_equal(visible[seq].values(name), private[seq].values(name))
                            for name in ALIASES
                        ),
                        "private release disagrees with first visible feature",
                    )
                else:
                    visible[seq] = private[seq]
            tx["release"] = release.document()
            tx["after_counters"]["released_rows"] += len(release.revealed_sequence_ids)
            require(
                canonical(release.document()) == original
                and canonical(expected_release.document()) == original,
                "private release authority drifted",
            )

        def final_guard():
            require(
                canonical(release.document()) == original
                and canonical(expected_release.document()) == original,
                "private release authority changed at final clock",
            )

        return self._execute("release", intent, (), None, action, final_guard=final_guard)

    def assemble_rows(self, authority, representation, intent, *, expected_authority):
        try:
            return self._assemble_rows(
                authority, representation, intent, expected_authority=expected_authority
            )
        except BaseException as error:
            # Failed children already sealed their paid failure and terminalized.
            # A parent-only validation failure must also prohibit any future use.
            self.abort(error, intent=intent)
            raise

    def _assemble_rows(self, authority, representation, intent, *, expected_authority):
        """Feature-only children then exact assembly; never performs a learner fit."""
        require(
            type(authority) is FeatureAssemblyBinding
            and type(expected_authority) is FeatureAssemblyBinding,
            "exact external charged feature assembly authority required",
        )
        original_authority = canonical(authority.document())
        require(
            canonical(expected_authority.document()) == original_authority,
            "external charged assembly authority differs",
        )
        original_intent = canonical(intent.document())
        self._guard()
        self._valid_intent(intent)
        representation_checked(representation)
        require(
            intent.purpose in ("charged", "initial", "terminal"), "charged assembly purpose differs"
        )
        history_fields = json_object(authority.raw_history_payload)
        observations = []
        for row in history_fields["observations"]:
            require(type(row) is dict, "charged assembly row JSON type differs")
            row = dict(row)
            if row["status"] == "successful":
                require(
                    type(row["objectives"]) is list and len(row["objectives"]) == 2,
                    "charged assembly objective JSON schema differs",
                )
                row["objectives"] = tuple(row["objectives"])
            observations.append(ChargedObservation(**row))
        history_fields["observations"] = tuple(observations)
        history = VerifiedHistorySnapshot(**history_fields)
        history.__post_init__()
        require(
            history.complete
            and history.sha256 == authority.history_sha256 == intent.history_sha256
            and history.run_id == self.binding.run_id
            and history.seed == self.binding.seed
            and history.objective_context_sha256 == self.binding.objective_context_sha256
            and history.round_index == intent.round_index,
            "charged assembly raw history differs",
        )
        eligible = set(authority.eligible_query_ids)
        require(
            eligible
            <= {row.query_id for row in history.observations if row.status == "successful"},
            "charged assembly eligibility must contain successful revealed IDs only",
        )
        selected = tuple(row for row in history.observations if row.query_id in eligible)
        wanted = tuple(row.sequence for row in selected)
        missing = tuple(seq for seq in wanted if seq not in self.__visible)
        require(
            not self.binding.tr2_grouped_capacity or not missing,
            "TR2 capacity charged fits must reuse cache",
        )
        parent = {"binding_sha256": authority.sha256, "intent": intent.document()}
        child_heads = []
        for index in range(0, len(missing), 128):
            require(
                canonical(authority.document()) == original_authority
                and canonical(expected_authority.document()) == original_authority
                and canonical(intent.document()) == original_intent,
                "charged parent assembly changed between chunks",
            )
            child_intent = replace(
                intent,
                expected_previous_head=self.__accepted_head,
                logical_ordinal=intent.logical_ordinal * 4 + index // 128,
            )
            self.raw_rows(
                missing[index : index + 128], representation, child_intent, parent_assembly=parent
            )
            child_heads.append(self.__accepted_head)
        assembly = {
            "binding": authority.document(),
            "selected_query_ids": [row.query_id for row in selected],
            "feature_sequence_ids": [sha256(seq.encode("ascii")) for seq in wanted],
            "child_receipt_sha256s": child_heads,
        }

        def final_guard():
            require(
                canonical(authority.document()) == original_authority
                and canonical(expected_authority.document()) == original_authority
                and canonical(intent.document()) == original_intent,
                "charged parent assembly authority changed at final check",
            )

        def action(tx, visible, private):
            del private
            final_guard()
            require(all(seq in visible for seq in wanted), "charged assembly has missing features")
            return self._result(tx, wanted, representation, visible)

        return self._execute(
            "assemble",
            intent,
            wanted,
            representation,
            action,
            allow_parent=True,
            final_guard=final_guard,
            assembly=assembly,
        )

    def close(self, intent):
        def action(tx, visible, private):
            del visible, private
            require(intent.purpose == "close", "feature close purpose differs")
            self.__session.close()
            require(
                self.__session.closed
                and (self.binding.session_root / "COMPLETE.json").is_file()
                and not (self.binding.session_root / "FAILED.json").exists(),
                "feature close was not successful",
            )

        result = self._execute("close", intent, (), None, action)
        self.__terminal = True
        return result

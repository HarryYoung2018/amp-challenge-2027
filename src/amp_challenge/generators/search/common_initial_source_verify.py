"""Fresh bounded source evidence inspection and independent receipt reconstruction."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from amp_challenge.evaluation.sequential_v2_seals import verify_phase
from amp_challenge.generators.search import common_initial_source_records as r
from amp_challenge.generators.search import durable_dispatch_journal_records as j
from amp_challenge.generators.search.durable_dispatch_journal_verify import (
    _directory_path,
    _regular,
)


def read_file(path, maximum):
    with _directory_path(path.parent) as parent:
        stat, digest, payload = _regular(parent, path.name, capture=True, maximum=maximum)
    return (
        (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_mtime_ns, stat.st_ctime_ns),
        digest,
        payload,
    )


def callback_identity(pin):
    j.require(type(pin) is j.CallbackPin, "source callback pin type differs")
    pin.__post_init__()
    value = inspect.getattr_static(pin.target, "source_sha256", None)
    j.require(type(value) is str and value == pin.source_sha256, "source callback identity differs")


class Evidence:
    """Re-read retained bytes across callbacks; never cache permission or authentication."""

    def __init__(self, root, plan, expected_plan_sha256, repository, sources, callbacks):
        j.require(
            type(plan) is r.SourcePlan
            and j.pin(expected_plan_sha256)
            and plan.sha256 == expected_plan_sha256,
            "source plan external pin differs",
        )
        j.require(
            type(sources) is tuple
            and len(sources) == len(r.SOURCE_FILES)
            and all(
                type(row) is tuple and len(row) == 2 and type(row[0]) is str and j.pin(row[1])
                for row in sources
            )
            and tuple(row[0] for row in sources) == r.SOURCE_FILES,
            "source inventory differs",
        )
        self.root, self.plan, self.plan_pin = root, plan, expected_plan_sha256
        self.repository, self.sources = repository, sources
        self.callbacks = tuple((pin, pin.target, pin.source_sha256) for pin in callbacks)
        self.phases = []
        self.arrivals = {}
        self.root_identity = self.directory_identity(root)
        self.arrivals_identity = self.directory_identity(root / "arrivals")
        self.check()

    @staticmethod
    def directory_identity(path):
        with _directory_path(path) as descriptor:
            value = os.fstat(descriptor)
        return value.st_dev, value.st_ino

    def check(self):
        j.require(
            type(self.plan) is r.SourcePlan and self.plan.sha256 == self.plan_pin,
            "source plan changed",
        )
        for pin, target, digest in self.callbacks:
            j.require(
                pin.target is target
                and type(pin.source_sha256) is str
                and pin.source_sha256 == digest,
                "source callback pin changed",
            )
            callback_identity(pin)
        loaded_repository = Path(__file__).resolve().parents[4]
        for name, expected in self.sources:
            j.require(
                read_file(self.repository / name, 4 * 1024 * 1024)[1] == expected,
                f"source bytes changed: {name}",
            )
            if self.repository != loaded_repository:
                j.require(
                    read_file(loaded_repository / name, 4 * 1024 * 1024)[1] == expected,
                    f"loaded source differs from the supplied repository: {name}",
                )
        j.require(
            self.directory_identity(self.root) == self.root_identity
            and self.directory_identity(self.root / "arrivals") == self.arrivals_identity,
            "source root identity changed",
        )
        with _directory_path(self.root) as descriptor:
            names = set(os.listdir(descriptor))
        j.require(
            names == {"arrivals", *(f"phase-{index:06d}" for index in range(len(self.phases)))},
            "source root inventory changed",
        )
        previous = {}
        for index, (expected, identity) in enumerate(self.phases):
            path = self.root / f"phase-{index:06d}"
            j.require(self.directory_identity(path) == identity, "source phase identity changed")
            seal = verify_phase(
                path,
                expected_artifact=r.ARTIFACT,
                expected_payload_paths=("event.json",),
                expected_predecessor_seals=previous,
                expected_seal_sha256=expected,
            )
            previous = {"previous": seal.seal_sha256}
        with _directory_path(self.root / "arrivals") as descriptor:
            j.require(
                set(os.listdir(descriptor)) == set(self.arrivals),
                "source arrivals inventory changed",
            )
        for name, snapshot in self.arrivals.items():
            j.require(
                read_file(self.root / "arrivals" / name, j.MAX_RECEIPT_BYTES)[:2] == snapshot,
                "retained source receipt changed",
            )

    def receipt(self, name):
        j.require(type(name) is str and name in self.arrivals, "source arrival missing")
        return read_file(self.root / "arrivals" / name, j.MAX_RECEIPT_BYTES)[2]


def verify_common_initial_source(
    root,
    *,
    expected_plan,
    expected_plan_sha256,
    expected_head_sha256,
    repository,
    expected_sources,
    authenticator,
):
    """Accept only complete source blocks; failures remain evidence, never ready copies.

    Reconstruct order, raw authentication, terminal rows and all 13 copy receipts
    without importing or calling the source producer. External expected pins are
    mandatory. Source issuing alone earns zero journal import charges.
    """
    j.require(isinstance(root, Path) and j.pin(expected_head_sha256), "reader root/head differs")
    # Admission precedes generic phase parsing and receipt authentication.
    with _directory_path(root) as descriptor:
        names = os.listdir(descriptor)
    phase_names = sorted(name for name in names if name.startswith("phase-"))
    j.require(
        1 <= len(phase_names) <= r.MAX_PHASES
        and set(names) == {"arrivals", *phase_names}
        and phase_names == [f"phase-{index:06d}" for index in range(len(phase_names))],
        "source phase inventory incomplete or unbounded",
    )
    events, phase_pins = [], []
    previous = {}
    for name in phase_names:
        path = root / name
        read_file(path / "event.json", r.MAX_EVENT_BYTES)
        read_file(path / "receipt.json", 4096)
        read_file(path / "SHA256SUMS", 4096)
        seal = verify_phase(
            path,
            expected_artifact=r.ARTIFACT,
            expected_payload_paths=("event.json",),
            expected_predecessor_seals=previous,
        )
        events.append(
            j.document(
                seal.read_payload_bytes("event.json"),
                fields=("index", "kind", "data"),
                maximum=r.MAX_EVENT_BYTES,
            )
        )
        phase_pins.append((seal.seal_sha256, Evidence.directory_identity(path)))
        previous = {"previous": seal.seal_sha256}
    j.require(phase_pins[-1][0] == expected_head_sha256, "source head differs from external pin")
    with _directory_path(root / "arrivals") as descriptor:
        arrival_names = sorted(os.listdir(descriptor))
    j.require(
        len(arrival_names) == 142
        and arrival_names == [f"{index:06d}.receipt" for index in range(142)],
        "source ready arrival count differs",
    )
    arrivals = {
        name: read_file(root / "arrivals" / name, j.MAX_RECEIPT_BYTES)[:2] for name in arrival_names
    }
    # Build the same evidence guard around the already captured tree. Its
    # constructor expects a new empty tree, so initialize only after admission.
    evidence = object.__new__(Evidence)
    evidence.root, evidence.plan, evidence.plan_pin = root, expected_plan, expected_plan_sha256
    evidence.repository, evidence.sources = repository, expected_sources
    evidence.callbacks = ((authenticator, authenticator.target, authenticator.source_sha256),)
    evidence.phases, evidence.arrivals = phase_pins, arrivals
    evidence.root_identity = Evidence.directory_identity(root)
    evidence.arrivals_identity = Evidence.directory_identity(root / "arrivals")
    j.require(
        type(expected_plan) is r.SourcePlan
        and j.pin(expected_plan_sha256)
        and expected_plan.sha256 == expected_plan_sha256,
        "reader source plan pin differs",
    )
    j.require(
        type(expected_sources) is tuple
        and len(expected_sources) == len(r.SOURCE_FILES)
        and all(
            type(row) is tuple and len(row) == 2 and type(row[0]) is str and j.pin(row[1])
            for row in expected_sources
        )
        and tuple(row[0] for row in expected_sources) == r.SOURCE_FILES,
        "reader sources differ",
    )
    evidence.check()
    for index, event in enumerate(events):
        j.require(
            type(event["index"]) is int and event["index"] == index, "source phase index differs"
        )
    cursor = 0
    used_arrivals = []

    def event(kind):
        nonlocal cursor
        j.require(
            cursor < len(events) and events[cursor]["kind"] == kind, "source phase order differs"
        )
        value = events[cursor]["data"]
        cursor += 1
        return value

    def authenticated(data, kind, expected):
        j.exact_fields(data, ("arrival", "receipt_sha256", "semantic"))
        j.require(
            data["arrival"] == f"{len(used_arrivals):06d}.receipt", "source arrival order differs"
        )
        raw = evidence.receipt(data["arrival"])
        j.require(j.digest(raw) == data["receipt_sha256"], "source receipt pin differs")
        evidence.check()
        semantic = authenticator.target(kind, raw, expected)
        evidence.check()
        j.require(
            type(semantic) is bytes and semantic == j.canonical(data["semantic"]),
            "source receipt authentication differs",
        )
        used_arrivals.append(data["arrival"])
        return data["semantic"], raw

    start = event("start")
    j.exact_fields(
        start,
        (
            "plan",
            "plan_sha256",
            "original_epoch",
            "original_deadline",
            "clock_epoch_id",
            "sources",
            "callbacks",
        ),
    )
    j.require(
        j.canonical(start["plan"]) == j.canonical(expected_plan.document())
        and start["plan_sha256"] == expected_plan_sha256
        and j.canonical(start["sources"]) == j.canonical(expected_sources),
        "source start binding differs",
    )
    r.timing(start["original_epoch"], start["original_deadline"], start["clock_epoch_id"])
    j.exact_fields(
        start["callbacks"], ("permission", "transport", "collector", "issuer", "authenticator")
    )
    j.require(
        all(j.pin(value) for value in start["callbacks"].values())
        and start["callbacks"]["authenticator"] == authenticator.source_sha256,
        "source start callback pins differ",
    )
    rows, external_ids = [], set()
    for index in range(64):
        dispatch = r.SourceDispatch(
            expected_plan,
            index,
            start["original_epoch"],
            start["original_deadline"],
            start["clock_epoch_id"],
        )
        j.require(
            j.canonical(event("intent")) == j.canonical(dispatch.document()),
            "source intent differs",
        )
        ack, _ = authenticated(
            event("ack"), "common_initial_ack", r.dispatch_expected(dispatch, "common_initial_ack")
        )
        j.exact_fields(ack, ("external_submission_id",))
        external_id = ack["external_submission_id"]
        j.require(
            j.identifier(external_id) and external_id not in external_ids,
            "source external ID duplicated/invalid",
        )
        external_ids.add(external_id)
        terminal, raw = authenticated(
            event("terminal"),
            "common_initial_terminal",
            r.dispatch_expected(dispatch, "common_initial_terminal", external_id),
        )
        j.exact_fields(terminal, ("external_submission_id", "status", "objectives"))
        j.require(
            type(terminal["external_submission_id"]) is str
            and terminal["external_submission_id"] == external_id,
            "source terminal acknowledgement differs",
        )
        j.terminal_values(terminal["status"], terminal["objectives"])
        rows.append(
            {
                "request_sha256": expected_plan.initial_requests[index].sha256,
                "external_submission_id": external_id,
                "status": terminal["status"],
                "objectives": terminal["objectives"],
                "response_receipt_sha256": j.digest(raw),
            }
        )
    aggregate, source_raw = authenticated(
        event("source"), "initial_source", r.initial_expected(expected_plan, "initial_source")
    )
    j.require(
        j.canonical(aggregate) == j.canonical({"rows": rows}),
        "aggregate source rows differ from raw terminals",
    )
    rows_sha = j.digest(j.canonical(rows))
    source_sha = j.digest(source_raw)
    copies = []
    for config, run_id in expected_plan.destinations:
        semantic = {
            "source_receipt_sha256": source_sha,
            "rows_sha256": rows_sha,
            "run_id": run_id,
            "seed": expected_plan.seed,
        }
        copy, raw = authenticated(
            event("copy"),
            "initial_copy",
            r.initial_expected(
                expected_plan,
                "initial_copy",
                run_id=run_id,
                rows_sha256=rows_sha,
                source_receipt_sha256=source_sha,
            ),
        )
        j.require(j.canonical(copy) == j.canonical(semantic), "source copy semantics differ")
        copies.append(
            {"configuration_id": config, "run_id": run_id, "receipt_sha256": j.digest(raw)}
        )
    ready = event("completed")
    expected_ready = {
        "status": "ready",
        "source_plan_sha256": expected_plan_sha256,
        "intents": 64,
        "transport_invocations": 64,
        "acknowledgements": 64,
        "terminal_rows": 64,
        "copy_receipts": 13,
        "logical_journal_imports": 0,
        "source_receipt_sha256": source_sha,
        "rows_sha256": rows_sha,
        "copies": copies,
        **r.FLAGS,
    }
    j.require(
        j.canonical(ready) == j.canonical(expected_ready)
        and cursor == len(events)
        and set(used_arrivals) == set(arrival_names),
        "source completion accounting differs",
    )
    evidence.check()
    return {
        **expected_ready,
        "head_sha256": expected_head_sha256,
        "source_receipt_hex": source_raw.hex(),
        "rows": rows,
        "copy_receipt_hex": [evidence.receipt(name).hex() for name in used_arrivals[-13:]],
        "external_timing_verified": False,
    }

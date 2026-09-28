"""Execute exactly 64 shared source queries before issuing 13 destination receipts."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import BuiltinFunctionType

from amp_challenge.evaluation.sequential_v2_seals import PhaseBuilder
from amp_challenge.generators.search import common_initial_source_records as r
from amp_challenge.generators.search import durable_dispatch_journal_records as j
from amp_challenge.generators.search.common_initial_source_verify import Evidence, read_file
from amp_challenge.generators.search.durable_dispatch_journal_verify import _directory_path


@dataclass(frozen=True, slots=True)
class SourceInputs:
    plan: r.SourcePlan
    expected_plan_sha256: str
    repository: Path
    expected_sources: tuple[tuple[str, str], ...]
    permission: j.CallbackPin
    transport: j.CallbackPin
    collector: j.CallbackPin
    issuer: j.CallbackPin
    authenticator: j.CallbackPin


def _write(path, payload):
    """Exclusive durable arrival/failure; never overwrite an earlier execution."""
    with _directory_path(path.parent) as parent:
        descriptor = os.open(
            path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.fsync(parent)


def execute_common_initial_source(
    inputs,
    *,
    output_root,
    original_epoch,
    original_deadline,
    clock_epoch_id,
    monotonic=time.monotonic,
):
    j.require(type(inputs) is SourceInputs, "source inputs exact type differs")
    r.timing(original_epoch, original_deadline, clock_epoch_id)
    j.require(
        isinstance(output_root, Path) and output_root.is_absolute(), "source output path differs"
    )
    root = output_root / "source"
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    (root / "arrivals").mkdir(mode=0o700)
    counts = {
        "intents": 0,
        "transport_invocations": 0,
        "acknowledgements": 0,
        "terminal_rows": 0,
        "copy_receipts": 0,
    }
    stage = "admission"
    evidence = None
    callback_names = ("permission", "transport", "collector", "issuer", "authenticator")
    pins = tuple(getattr(inputs, name) for name in callback_names)
    original_inputs = tuple(getattr(inputs, name) for name in SourceInputs.__dataclass_fields__)

    def guard():
        j.require(
            all(
                getattr(inputs, name) is value
                for name, value in zip(
                    SourceInputs.__dataclass_fields__, original_inputs, strict=True
                )
            ),
            "source input capability changed",
        )
        evidence.check()
        observed = monotonic()
        if not (type(monotonic) is BuiltinFunctionType and monotonic is time.monotonic):
            evidence.check()
        j.require(
            type(observed) is float and math.isfinite(observed) and observed >= original_epoch,
            "source observed clock differs",
        )
        if observed >= original_deadline:
            raise TimeoutError("original source deadline exceeded")

    def invoke(pin, *args, arrival=False, transport=False):
        guard()
        if transport:
            counts["transport_invocations"] += 1
        result = pin.target(*args)
        if arrival:
            # A late raw acknowledgement/response is evidence even though it
            # must not be authenticated into an accepted terminal row.
            raw = j.receipt_bytes(result)
            name = f"{len(evidence.arrivals):06d}.receipt"
            j.require(len(evidence.arrivals) < 142, "source arrival cap exceeded")
            j.require(
                Evidence.directory_identity(root) == evidence.root_identity
                and Evidence.directory_identity(root / "arrivals") == evidence.arrivals_identity,
                "source arrival parent identity changed",
            )
            _write(root / "arrivals" / name, raw)
            evidence.arrivals[name] = read_file(root / "arrivals" / name, j.MAX_RECEIPT_BYTES)[:2]
            guard()
            return name, raw
        guard()
        return result

    def publish(kind, data):
        index = len(evidence.phases)
        j.require(index < r.MAX_PHASES, "source phase cap exceeded")
        path = root / f"phase-{index:06d}"
        predecessors = {} if not evidence.phases else {"previous": evidence.phases[-1][0]}
        payload = j.canonical({"index": index, "kind": kind, "data": data})
        j.document(payload, maximum=r.MAX_EVENT_BYTES)
        with PhaseBuilder(path, artifact=r.ARTIFACT, predecessor_seals=predecessors) as builder:
            builder.write_bytes("event.json", payload)
            seal = builder.publish(expected_payload_paths=("event.json",))
        evidence.phases.append((seal.seal_sha256, Evidence.directory_identity(path)))
        count_key = {
            "intent": "intents",
            "ack": "acknowledgements",
            "terminal": "terminal_rows",
            "copy": "copy_receipts",
        }.get(kind)
        if count_key is not None:
            counts[count_key] += 1
        guard()
        return seal

    def authenticate(kind, expected, arrival):
        name, raw = arrival
        semantic = invoke(inputs.authenticator, kind, raw, expected)
        document = j.document(semantic, maximum=j.MAX_RECEIPT_BYTES)
        return {"arrival": name, "receipt_sha256": j.digest(raw), "semantic": document}

    try:
        evidence = Evidence(
            root,
            inputs.plan,
            inputs.expected_plan_sha256,
            inputs.repository,
            inputs.expected_sources,
            pins,
        )
        guard()
        publish(
            "start",
            {
                "plan": inputs.plan.document(),
                "plan_sha256": inputs.expected_plan_sha256,
                "original_epoch": original_epoch,
                "original_deadline": original_deadline,
                "clock_epoch_id": clock_epoch_id,
                "sources": inputs.expected_sources,
                "callbacks": {
                    name: pin.source_sha256 for name, pin in zip(callback_names, pins, strict=True)
                },
            },
        )
        rows, external_ids = [], set()
        for index, request in enumerate(inputs.plan.initial_requests):
            dispatch = r.SourceDispatch(
                inputs.plan, index, original_epoch, original_deadline, clock_epoch_id
            )
            stage = f"intent-{index}"
            publish("intent", dispatch.document())
            stage = f"permission-{index}"
            j.require(
                invoke(inputs.permission, dispatch) is True,
                "source permission denied after durable intent",
            )
            stage = f"transport-{index}"
            # Count actual local invocations, including calls that lose their
            # acknowledgement. No implicit retry or physical-count inference.
            arrival = invoke(inputs.transport, dispatch, arrival=True, transport=True)
            stage = f"ack-{index}"
            ack = authenticate(
                "common_initial_ack", r.dispatch_expected(dispatch, "common_initial_ack"), arrival
            )
            j.exact_fields(ack["semantic"], ("external_submission_id",))
            external_id = ack["semantic"]["external_submission_id"]
            j.require(
                j.identifier(external_id) and external_id not in external_ids,
                "source external ID duplicated or invalid",
            )
            external_ids.add(external_id)
            publish("ack", ack)
            stage = f"collect-{index}"
            arrival = invoke(
                inputs.collector, dispatch, external_id, original_deadline, arrival=True
            )
            stage = f"terminal-{index}"
            terminal = authenticate(
                "common_initial_terminal",
                r.dispatch_expected(dispatch, "common_initial_terminal", external_id),
                arrival,
            )
            semantic = terminal["semantic"]
            j.exact_fields(semantic, ("external_submission_id", "status", "objectives"))
            j.require(
                type(semantic["external_submission_id"]) is str
                and semantic["external_submission_id"] == external_id,
                "source terminal external ID differs",
            )
            j.terminal_values(semantic["status"], semantic["objectives"])
            publish("terminal", terminal)
            rows.append(
                {
                    "request_sha256": request.sha256,
                    "external_submission_id": external_id,
                    "status": semantic["status"],
                    "objectives": semantic["objectives"],
                    "response_receipt_sha256": terminal["receipt_sha256"],
                }
            )
        stage = "source-aggregate"
        expected = r.initial_expected(inputs.plan, "initial_source")
        requested = j.canonical({"rows": rows})
        arrival = invoke(inputs.issuer, "initial_source", expected, requested, arrival=True)
        source = authenticate("initial_source", expected, arrival)
        j.require(j.canonical(source["semantic"]) == requested, "aggregate source rows differ")
        publish("source", source)
        rows_sha = j.digest(j.canonical(rows))
        source_sha = source["receipt_sha256"]
        copies = []
        for config, run_id in inputs.plan.destinations:
            stage = f"copy-{config}"
            expected = r.initial_expected(
                inputs.plan,
                "initial_copy",
                run_id=run_id,
                rows_sha256=rows_sha,
                source_receipt_sha256=source_sha,
            )
            requested = j.canonical(
                {
                    "source_receipt_sha256": source_sha,
                    "rows_sha256": rows_sha,
                    "run_id": run_id,
                    "seed": inputs.plan.seed,
                }
            )
            arrival = invoke(inputs.issuer, "initial_copy", expected, requested, arrival=True)
            copy = authenticate("initial_copy", expected, arrival)
            j.require(
                j.canonical(copy["semantic"]) == requested, "source copy rows/destination differs"
            )
            publish("copy", copy)
            copies.append(
                {
                    "configuration_id": config,
                    "run_id": run_id,
                    "receipt_sha256": copy["receipt_sha256"],
                }
            )
        stage = "completion"
        ready = {
            "status": "ready",
            "source_plan_sha256": inputs.expected_plan_sha256,
            **counts,
            "logical_journal_imports": 0,
            "source_receipt_sha256": source_sha,
            "rows_sha256": rows_sha,
            "copies": copies,
            **r.FLAGS,
        }
        seal = publish("completed", ready)
        return {**ready, "head_sha256": seal.seal_sha256}
    except BaseException as error:
        # Counts are producer observations, not an authentication of an external
        # service's once-only ledger. Raw late arrivals remain under arrivals/.
        _write(
            root / "failure.json",
            j.canonical(
                {
                    "status": "failed",
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "message": str(error)[:4096],
                    **counts,
                    "logical_journal_imports": 0,
                    "original_epoch": original_epoch,
                    "original_deadline": original_deadline,
                    "clock_epoch_id": clock_epoch_id,
                    "retained_phases": 0 if evidence is None else len(evidence.phases),
                    "retained_arrivals": 0 if evidence is None else len(evidence.arrivals),
                    "head_sha256": None
                    if evidence is None or not evidence.phases
                    else evidence.phases[-1][0],
                    **r.FLAGS,
                }
            ),
        )
        raise

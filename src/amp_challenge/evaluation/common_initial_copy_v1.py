"""Pure, non-authorizing comparison of one frozen 13-arm initial-data block.

Inputs are constructible bytes, not authentication or reconstruction receipts.
Opaque receipt equality cannot establish their semantics or physical oracle
work. A separate adapter must itself reconstruct current journals and retain
qualified upstream evidence. Never pass private genesis bytes to a learner.
"""

from __future__ import annotations

import json
import re

from amp_challenge.evaluation.evolutionary_kl_successor_protocol_v2 import (
    CONFIGURATION_IDS_V2,
    FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
    SCREEN_SEEDS_V2,
)
from amp_challenge.evaluation.evolutionary_kl_successor_runtime_v1 import (
    FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
    DigestSlot,
    ScreenRunPlan,
    build_logical_schedule,
)
from amp_challenge.generators.search.durable_dispatch_journal_records import (
    COPY_FIELDS,
    EVENT_DATA_FIELDS,
    INITIAL_ROW_FIELDS,
    MAX_EVENT_BYTES,
    QUERY_FIELDS,
    JournalEvent,
    binding_from_bytes,
    canonical,
    digest,
    exact_fields,
    identifier,
    pin,
    receipt_from_hex,
    request_from_document,
    require,
    terminal_values,
)

__all__ = ("compare_common_initial_seed",)

_MAX_EXPECTED_BYTES = 2097152
_MAX_PLAN_BYTES = 16384
_MAX_TOTAL_BYTES = 29573120
_MAX_OUTPUT_BYTES = 32768
_MAX_DEPTH = 12
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_ARTIFACT = "evolutionary_kl_common_initial_copy_expectations_v1"
_OUTPUT_ARTIFACT = "evolutionary_kl_common_initial_copy_consistency_v1"
_EXPECTED_FIELDS = (
    "artifact",
    "schema_version",
    "protocol_sha256",
    "runtime_sha256",
    "seed",
    "source_commit",
    "journal_implementation_sha256",
    "input_evidence_class",
    "source",
    "arms",
)
_SOURCE_FIELDS = (
    "run_id",
    "objective_context_sha256",
    "oracle_bundle_sha256",
    "requests",
    "receipt_hex",
    "document",
)
_ARM_FIELDS = (
    "configuration_id",
    "run_plan_sha256",
    "genesis_sha256",
    "initial_event_sha256",
    "copy_receipt_hex",
    "authenticator_sha256",
)
_PLAN_FIELDS = (
    "artifact",
    "artifact_digests",
    "authorization",
    "configuration_id",
    "evidence_class",
    "protocol_sha256",
    "run_id",
    "runtime_sha256",
    "schema_version",
    "seed",
    "topology_sha256",
)
_FALSE_FLAGS = (
    "execution_authorized",
    "oracle_calls_authorized",
    "scientific_evidence_accepted",
    "production_eligible",
    "external_issuer_authenticity_established",
    "release_authorized",
    "fresh_journal_reconstruction_established",
    "physical_oracle_work_established",
    "upstream_evidence_qualification_established",
)


def _admit_depth(payload: bytes) -> None:
    # This is only a nesting admission scan, not a JSON parser. String contents
    # and escaped quotes cannot create false container nesting. Invalid syntax
    # is handled by the bounded canonical parser at the appropriate later stage.
    depth = 0
    quoted = escaped = False
    for value in payload:
        if quoted:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                quoted = False
        elif value == 34:
            quoted = True
        elif value in (91, 123):
            depth += 1
            require(depth <= _MAX_DEPTH, "invalid_input: JSON nesting exceeds admission")
        elif value in (93, 125):
            depth = max(0, depth - 1)


def _admit(expected, members) -> None:
    require(
        type(expected) is bytes and 0 < len(expected) <= _MAX_EXPECTED_BYTES,
        "invalid_input: expected byte admission differs",
    )
    require(
        type(members) is tuple and len(members) == len(CONFIGURATION_IDS_V2),
        "invalid_input: member inventory differs",
    )
    payloads = [expected]
    total = len(expected)
    for member in members:
        require(
            type(member) is tuple and len(member) == 2,
            "invalid_input: member shape differs",
        )
        plan, capture = member
        require(
            type(plan) is bytes and 0 < len(plan) <= _MAX_PLAN_BYTES,
            "invalid_input: plan byte admission differs",
        )
        payloads.append(plan)
        total += len(plan)
        if capture is not None:
            require(
                type(capture) is tuple and len(capture) == 2,
                "invalid_input: capture shape differs",
            )
            for payload in capture:
                require(
                    type(payload) is bytes and 0 < len(payload) <= MAX_EVENT_BYTES,
                    "invalid_input: capture byte admission differs",
                )
                total += len(payload)
                payloads.append(payload)
    require(total <= _MAX_TOTAL_BYTES, "invalid_input: aggregate byte admission differs")
    for payload in payloads:
        _admit_depth(payload)


def _document(payload: bytes, *, newline: bool = False):
    try:
        value = json.loads(payload)
        encoded = canonical(value) + (b"\n" if newline else b"")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("bounded canonical JSON differs") from error
    require(type(value) is dict and encoded == payload, "canonical object bytes differ")
    return value


def _requests(source):
    raw = source["requests"]
    require(type(raw) is list and len(raw) == 64, "source request inventory differs")
    requests = tuple(request_from_document(row) for row in raw)
    require(
        len({row.query_id for row in requests}) == 64
        and len({row.sequence for row in requests}) == 64
        and len({row.identity.key for row in requests}) == 64,
        "source request identities repeat",
    )
    reference = requests[0].identity
    for request in requests:
        require(
            request.identity.endpoint_context_sha256 == source["objective_context_sha256"]
            and all(
                getattr(request.identity, field) == getattr(reference, field)
                for field in QUERY_FIELDS[1:-1]
            ),
            "source request context differs",
        )
    return requests


def _rows(source_document, requests) -> bytes:
    exact_fields(source_document, ("rows",))
    rows = source_document["rows"]
    require(type(rows) is list and len(rows) == 64, "source row inventory differs")
    external_ids = set()
    for request, row in zip(requests, rows, strict=True):
        exact_fields(row, INITIAL_ROW_FIELDS)
        external_id = row["external_submission_id"]
        require(
            row["request_sha256"] == request.sha256
            and identifier(external_id)
            and external_id not in external_ids
            and pin(row["response_receipt_sha256"]),
            "source row identity differs",
        )
        external_ids.add(external_id)
        terminal_values(row["status"], row["objectives"])
    return canonical(rows)


def _expectations(payload: bytes):
    value = _document(payload)
    exact_fields(value, _EXPECTED_FIELDS)
    require(
        value["artifact"] == _EXPECTED_ARTIFACT
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["protocol_sha256"] == FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256
        and value["runtime_sha256"] == FROZEN_SUCCESSOR_RUNTIME_V1_SHA256
        and type(value["seed"]) is int
        and value["seed"] in SCREEN_SEEDS_V2
        and type(value["source_commit"]) is str
        and _COMMIT.fullmatch(value["source_commit"]) is not None
        and pin(value["journal_implementation_sha256"])
        and type(value["input_evidence_class"]) is str
        and value["input_evidence_class"] in ("synthetic_fixture", "external_unqualified"),
        "expectation header differs",
    )
    source = value["source"]
    exact_fields(source, _SOURCE_FIELDS)
    require(
        identifier(source["run_id"])
        and pin(source["objective_context_sha256"])
        and pin(source["oracle_bundle_sha256"]),
        "expected source binding differs",
    )
    requests = _requests(source)
    source_raw = receipt_from_hex(source["receipt_hex"])
    source_rows = _rows(source["document"], requests)
    arms = value["arms"]
    require(
        type(arms) is list and len(arms) == len(CONFIGURATION_IDS_V2),
        "expected arm inventory differs",
    )
    copy_receipts = []
    for configuration_id, arm in zip(CONFIGURATION_IDS_V2, arms, strict=True):
        exact_fields(arm, _ARM_FIELDS)
        require(
            arm["configuration_id"] == configuration_id
            and all(
                pin(arm[key])
                for key in (
                    "run_plan_sha256",
                    "genesis_sha256",
                    "initial_event_sha256",
                    "authenticator_sha256",
                )
            ),
            "expected arm mapping differs",
        )
        copy_receipts.append(receipt_from_hex(arm["copy_receipt_hex"]))
    return value, requests, source_raw, source_rows, tuple(copy_receipts)


def _plan(payload: bytes, expected_arm, seed: int) -> ScreenRunPlan:
    value = _document(payload, newline=True)
    exact_fields(value, _PLAN_FIELDS)
    slots = value["artifact_digests"]
    require(type(slots) is list, "plan artifact inventory differs")
    for slot in slots:
        exact_fields(slot, ("role", "status", "sha256"))
    plan = ScreenRunPlan(
        run_id=value["run_id"],
        configuration_id=value["configuration_id"],
        seed=value["seed"],
        logical_schedule=build_logical_schedule(),
        topology_sha256=value["topology_sha256"],
        artifact_digests=tuple(DigestSlot(**slot) for slot in slots),
    )
    require(
        canonical(plan.document()) + b"\n" == payload
        and plan.configuration_id == expected_arm["configuration_id"]
        and plan.seed == seed
        and plan.plan_sha256 == expected_arm["run_plan_sha256"],
        "plan roundtrip or external mapping differs",
    )
    return plan


def _member_status(
    plan_payload, capture, *, expected, arm, requests, source_raw, source_rows, copy_raw
):
    try:
        plan = _plan(plan_payload, arm, expected["seed"])
    except ValueError:
        return "plan_mismatch", None
    missing_roles = [slot.role for slot in plan.artifact_digests if slot.is_blocking]
    if capture is None:
        return "missing_capture", missing_roles
    try:
        # Enforce bounded canonical parsing before the existing pure wire parsers.
        _document(capture[0])
        _document(capture[1])
        binding = binding_from_bytes(capture[0])
        event = JournalEvent(capture[1])
        event_document = event.document()
        data = event_document["data"]
        exact_fields(data, EVENT_DATA_FIELDS["initial_import"])
    except ValueError:
        return "malformed_capture", missing_roles
    source = expected["source"]
    if not (
        binding.run_id == plan.run_id
        and binding.arm_id == plan.configuration_id
        and binding.seed == plan.seed
        and binding.initial_source_run_id == source["run_id"]
        and binding.objective_context_sha256 == source["objective_context_sha256"]
        and binding.oracle_bundle_sha256 == source["oracle_bundle_sha256"]
        and binding.implementation_sha256 == expected["journal_implementation_sha256"]
        and binding.authenticator_sha256 == arm["authenticator_sha256"]
        and canonical([row.document() for row in binding.initial_requests])
        == canonical(source["requests"])
    ):
        return "identity_mismatch", missing_roles
    if not (
        binding.sha256 == arm["genesis_sha256"]
        and event.sha256 == arm["initial_event_sha256"]
        and event_document["ordinal"] == 0
        and event_document["kind"] == "initial_import"
        and event_document["genesis_sha256"] == binding.sha256
        and event_document["previous_event_sha256"] == binding.sha256
    ):
        return "checkpoint_mismatch", missing_roles
    try:
        retained_source = receipt_from_hex(data["source_receipt_hex"])
        require(
            retained_source == source_raw
            and digest(retained_source) == binding.initial_source_receipt_sha256
            and _rows(data["source_document"], requests) == source_rows
            and canonical(data["source_document"]) == canonical(source["document"]),
            "source content differs",
        )
    except ValueError:
        return "source_mismatch", missing_roles
    try:
        retained_copy = receipt_from_hex(data["copy_receipt_hex"])
        exact_fields(data["copy_document"], COPY_FIELDS)
        require(
            retained_copy == copy_raw
            and digest(retained_copy) == binding.initial_copy_receipt_sha256
            and canonical(data["copy_document"])
            == canonical(
                {
                    "source_receipt_sha256": digest(source_raw),
                    "rows_sha256": digest(source_rows),
                    "run_id": plan.run_id,
                    "seed": plan.seed,
                }
            ),
            "copy content differs",
        )
    except ValueError:
        return "copy_mismatch", missing_roles
    return "consistent", missing_roles


def compare_common_initial_seed(
    *,
    expected: bytes,
    members: tuple[tuple[bytes, tuple[bytes, bytes] | None], ...],
) -> bytes:
    """Compare all 13 initial copies; authorize nothing and perform no I/O.

    Expectations must be held independently by the caller, not derived from
    the candidate captures being checked. Their classification, authenticity,
    physical origin and qualification are not established by this function.
    Missing and inconsistent admitted members remain in the complete output.
    """
    _admit(expected, members)
    try:
        wanted, requests, source_raw, source_rows, copies = _expectations(expected)
    except ValueError as error:
        raise ValueError("invalid_expected: canonical schema or source mapping differs") from error
    arms = []
    for expected_arm, member, copy_raw in zip(wanted["arms"], members, copies, strict=True):
        status, missing_roles = _member_status(
            *member,
            expected=wanted,
            arm=expected_arm,
            requests=requests,
            source_raw=source_raw,
            source_rows=source_rows,
            copy_raw=copy_raw,
        )
        configuration_id = expected_arm["configuration_id"]
        arms.append(
            {
                "configuration_id": configuration_id,
                "run_id": f"screen.{configuration_id}.seed-{wanted['seed']}",
                "expected_plan_sha256": expected_arm["run_plan_sha256"],
                "expected_genesis_sha256": expected_arm["genesis_sha256"],
                "expected_initial_event_sha256": expected_arm["initial_event_sha256"],
                "status": status,
                "plan_missing_roles": missing_roles,
            }
        )
    matched = sum(arm["status"] == "consistent" for arm in arms)
    result = {
        "artifact": _OUTPUT_ARTIFACT,
        "schema_version": 1,
        "protocol_sha256": FROZEN_SUCCESSOR_PROTOCOL_V2_SHA256,
        "runtime_sha256": FROZEN_SUCCESSOR_RUNTIME_V1_SHA256,
        "seed": wanted["seed"],
        "source_commit": wanted["source_commit"],
        "journal_implementation_sha256": wanted["journal_implementation_sha256"],
        "expected_sha256": digest(expected),
        "input_evidence_class": wanted["input_evidence_class"],
        "evidence_class": "engineering_consistency_only",
        "source_receipt_sha256": digest(source_raw),
        "source_rows_sha256": digest(source_rows),
        "source_requests_sha256": digest(canonical(wanted["source"]["requests"])),
        "status": "consistent"
        if matched == len(CONFIGURATION_IDS_V2)
        else "incomplete_or_inconsistent",
        "accounting": {
            "basis": "required_protocol_identity_not_observed_work",
            "required_source_submissions": 64,
            "required_logical_initial_charges": 832,
            "comparison_physical_submissions": 0,
            "matched_arms": matched,
            "matched_logical_initial_charges": 64 * matched,
        },
        "arms": arms,
        **dict.fromkeys(_FALSE_FLAGS, False),
    }
    payload = canonical(result)
    require(len(payload) <= _MAX_OUTPUT_BYTES, "comparison output exceeds admission")
    return payload

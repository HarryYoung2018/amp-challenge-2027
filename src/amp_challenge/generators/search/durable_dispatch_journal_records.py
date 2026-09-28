"""Pure wire records for the prospective durable dispatch journal.

These records establish structure/content identity, never receipt authenticity.
The producer and independent checker share this module, not replay transitions.
Canonical JSON is ASCII, sorted, compact, without a newline. Event/genesis/token
hash domains are distinct. No function here opens a path or calls a provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from typing import Protocol

from amp_challenge.generators.search.campaign_ledger import OracleQueryIdentity
from amp_challenge.generators.search.peptide_ga_tunable_v2_records import ChargedObservation
from amp_challenge.generators.search.verified_charged_history import VerifiedHistorySnapshot

CONTRACT_PATH = "configs/search/durable_dispatch_journal_v1.toml"
CONTRACT_SHA256 = "82e4a0708e487bac2c4c716dbbaa66bc8ad16b7053b0f96541a055db6626adda"
PROTOCOL_SHA256 = "579d071fac1d70bf514761ff98e6d2019730fbdad484e548f9b672fda1a001af"
MAX_EVENTS = 2048
MAX_EVENT_BYTES = 1048576
MAX_RECEIPT_BYTES = 65536
MAX_JOURNAL_BYTES = 268435456
GENESIS_NAME = "genesis"
GENESIS_PAYLOAD = "binding.json"
EVENT_PAYLOAD = "event.json"
PENDING_NAME = ".pending"
LOCK_NAME = "writer.lock"
GENESIS_ARTIFACT = "durable_dispatch_journal_genesis_v1"
EVENT_ARTIFACT = "durable_dispatch_journal_event_v1"
PHASE_FILES = ("SHA256SUMS", "receipt.json")
# Disk layout: genesis/{binding.json,receipt.json,SHA256SUMS}, followed by
# event-NNNNNN/{event.json,receipt.json,SHA256SUMS}; writer.lock is empty and
# .pending/ contains no entries on a clean boundary. Pending remnants are retained
# and block writing/handoff, never treated as another committed event.
# Generic phase metadata is {genesis_sha256} for genesis, and
# {genesis_sha256,event_sha256,ordinal} for an event. Genesis predecessors are {};
# event predecessors are {"previous_event": event.previous_event_sha256}.
# Checkpoint/event/wave heads use the domain-separated event digest, NOT the
# generic publication SHA256SUMS digest. Both must be independently verified.
KINDS = (
    "initial_import",
    "wave_seal",
    "dispatch_intent",
    "submission_ack",
    "terminal_response",
    "dispatch_fault",
    "stop",
)
STATUSES = ("succeeded", "failed", "missing", "censored", "partial", "timeout")
STATUS_TO_HISTORY = {
    "succeeded": "successful",
    "failed": "failed",
    "missing": "missing",
    "censored": "censored",
    "partial": "partial",
    "timeout": "timed_out",
}
QUERY_FIELDS = (
    "canonical_sequence_id",
    "oracle_contract_sha256",
    "evaluator_sha256",
    "checkpoint_sha256",
    "endpoint_context_sha256",
    "transform_sha256",
    "replicate_id",
)
SOURCE_FILES = frozenset(
    {
        CONTRACT_PATH,
        "src/amp_challenge/generators/search/durable_dispatch_journal_records.py",
        "src/amp_challenge/generators/search/durable_dispatch_journal.py",
        "src/amp_challenge/generators/search/durable_dispatch_journal_verify.py",
        "src/amp_challenge/evaluation/sequential_v2_seals.py",
        "src/amp_challenge/evaluation/evolutionary_kl_protocol.py",
        "src/amp_challenge/generators/search/campaign_ledger.py",
        "src/amp_challenge/generators/search/batching.py",
        "src/amp_challenge/generators/search/ledger.py",
        "src/amp_challenge/generators/search/records.py",
        "src/amp_challenge/generators/search/peptide_ga_records.py",
        "src/amp_challenge/generators/search/peptide_ga_tunable_v2_records.py",
        "src/amp_challenge/generators/search/verified_charged_history.py",
    }
)
_PIN = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{8,50}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(value):
    return type(value) is str and _PIN.fullmatch(value) is not None


def identifier(value):
    return type(value) is str and _ID.fullmatch(value) is not None


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest(payload):
    require(type(payload) is bytes, "digest input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def document(payload, *, fields=None, maximum=MAX_EVENT_BYTES):
    require(
        type(payload) is bytes and 0 < len(payload) <= maximum,
        "canonical record exceeds its byte admission",
    )
    value = json.loads(payload)
    require(
        type(value) is dict and canonical(value) == payload,
        "record is not an exact canonical object",
    )
    if fields is not None:
        require(set(value) == set(fields), "record field inventory differs")
    return value


def exact_fields(value, fields):
    require(type(value) is dict and set(value) == set(fields), "record fields differ")


def receipt_bytes(value):
    require(
        type(value) is bytes and 0 < len(value) <= MAX_RECEIPT_BYTES,
        "external receipt byte admission differs",
    )
    return value


def receipt_from_hex(value):
    require(
        type(value) is str and 0 < len(value) <= 2 * MAX_RECEIPT_BYTES and len(value) % 2 == 0,
        "receipt hex admission differs",
    )
    result = bytes.fromhex(value)
    require(result.hex() == value, "receipt hex is not canonical lowercase")
    return receipt_bytes(result)


def source_inventory_sha256(inventory):
    require(
        type(inventory) is tuple and len(inventory) == len(SOURCE_FILES),
        "source inventory cardinality differs",
    )
    require(
        all(
            type(row) is tuple and len(row) == 2 and type(row[0]) is str and pin(row[1])
            for row in inventory
        ),
        "source inventory shape differs",
    )
    require(
        tuple(sorted(inventory)) == inventory and {row[0] for row in inventory} == SOURCE_FILES,
        "source inventory keys/order differ",
    )
    return digest(b"amp/durable-dispatch/source/v1\0" + canonical(inventory))


@dataclass(frozen=True, slots=True)
class JournalRequest:
    query_id: str
    sequence: str
    identity: OracleQueryIdentity

    def __post_init__(self):
        require(identifier(self.query_id), "request query ID differs")
        require(
            type(self.sequence) is str and _SEQUENCE.fullmatch(self.sequence) is not None,
            "request sequence differs",
        )
        require(type(self.identity) is OracleQueryIdentity, "request identity type differs")
        self.identity.__post_init__()
        require(
            self.identity.canonical_sequence_id == digest(self.sequence.encode("ascii")),
            "request identity does not bind canonical sequence",
        )

    def document(self):
        self.__post_init__()
        return asdict(self)

    @property
    def sha256(self):
        return digest(b"amp/durable-dispatch/request/v1\0" + canonical(self.document()))


def request_from_document(value):
    exact_fields(value, ("query_id", "sequence", "identity"))
    exact_fields(value["identity"], QUERY_FIELDS)
    return JournalRequest(
        value["query_id"], value["sequence"], OracleQueryIdentity(**value["identity"])
    )


@dataclass(frozen=True, slots=True)
class JournalBinding:
    run_id: str
    arm_id: str
    seed: int
    objective_context_sha256: str
    oracle_bundle_sha256: str
    implementation_sha256: str
    transport_sha256: str
    authenticator_sha256: str
    permission_sha256: str
    timing_context_sha256: str
    initial_source_run_id: str
    initial_source_receipt_sha256: str
    initial_copy_receipt_sha256: str
    initial_requests: tuple[JournalRequest, ...]
    reserves: tuple[tuple[JournalRequest, JournalRequest], ...]

    def __post_init__(self):
        require(
            all(
                identifier(value)
                for value in (self.run_id, self.arm_id, self.initial_source_run_id)
            ),
            "binding identifier differs",
        )
        require(type(self.seed) is int and 0 <= self.seed < 2**63, "binding seed differs")
        require(
            all(
                pin(getattr(self, name))
                for name in (
                    "objective_context_sha256",
                    "oracle_bundle_sha256",
                    "implementation_sha256",
                    "transport_sha256",
                    "authenticator_sha256",
                    "permission_sha256",
                    "timing_context_sha256",
                    "initial_source_receipt_sha256",
                    "initial_copy_receipt_sha256",
                )
            ),
            "binding digest differs",
        )
        require(
            type(self.initial_requests) is tuple and len(self.initial_requests) == 64,
            "initial request block must contain exactly 64 rows",
        )
        require(
            type(self.reserves) is tuple
            and len(self.reserves) == 28
            and all(type(wave) is tuple and len(wave) == 2 for wave in self.reserves),
            "private reserve schedule must be exact 28 by 2",
        )
        rows = (*self.initial_requests, *(row for wave in self.reserves for row in wave))
        for row in rows:
            require(type(row) is JournalRequest, "binding request type differs")
            row.__post_init__()
            self.validate_request(row)
        require(
            len({row.query_id for row in rows}) == len(rows)
            and len({row.sequence for row in rows}) == len(rows)
            and len({row.identity.key for row in rows}) == len(rows),
            "initial/reserve request inventories overlap",
        )

    def validate_request(self, request):
        require(type(request) is JournalRequest, "request exact type differs")
        request.__post_init__()
        expected = self.initial_requests[0].identity
        require(
            request.identity.endpoint_context_sha256 == self.objective_context_sha256,
            "request objective context differs",
        )
        require(
            all(
                getattr(request.identity, field) == getattr(expected, field)
                for field in QUERY_FIELDS[1:-1]
            ),
            "request oracle identity differs",
        )

    def document(self):
        self.__post_init__()
        # Full validation above establishes exact str/int leaves and exact
        # request/identity types. Build fresh containers directly; generic
        # recursive deepcopy adds no isolation for those immutable leaves.
        # Nothing is cached: every guard still validates and serializes live
        # values, including all 64 initial requests and 56 private reserves.
        if type(self) is JournalBinding:
            binding = {
                field.name: getattr(self, field.name)
                for field in fields(JournalBinding)
                if field.name not in ("initial_requests", "reserves")
            }

            def request(row):
                return {
                    "query_id": row.query_id,
                    "sequence": row.sequence,
                    "identity": {name: getattr(row.identity, name) for name in QUERY_FIELDS},
                }

            binding["initial_requests"] = tuple(request(row) for row in self.initial_requests)
            binding["reserves"] = tuple(
                tuple(request(row) for row in wave) for wave in self.reserves
            )
        else:
            binding = asdict(self)
        return {
            "artifact": GENESIS_ARTIFACT,
            "schema_version": 1,
            "contract_sha256": CONTRACT_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "scientific_evidence_accepted": False,
            "production_eligible": False,
            "binding": binding,
        }

    @property
    def sha256(self):
        return digest(b"amp/durable-dispatch/genesis/v1\0" + canonical(self.document()))

    @property
    def provider_sha256(self):
        return digest(
            b"amp/durable-dispatch/history-provider/v1\0"
            + self.implementation_sha256.encode("ascii")
        )


def binding_from_bytes(payload):
    value = document(
        payload,
        fields=(
            "artifact",
            "schema_version",
            "contract_sha256",
            "protocol_sha256",
            "scientific_evidence_accepted",
            "production_eligible",
            "binding",
        ),
    )
    require(
        value["artifact"] == GENESIS_ARTIFACT
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["contract_sha256"] == CONTRACT_SHA256
        and value["protocol_sha256"] == PROTOCOL_SHA256
        and value["scientific_evidence_accepted"] is False
        and value["production_eligible"] is False,
        "genesis header differs",
    )
    raw = value["binding"]
    exact_fields(raw, JournalBinding.__dataclass_fields__)
    require(
        type(raw["initial_requests"]) is list
        and len(raw["initial_requests"]) == 64
        and type(raw["reserves"]) is list
        and len(raw["reserves"]) == 28,
        "genesis schedule shape differs",
    )
    require(
        all(type(wave) is list and len(wave) == 2 for wave in raw["reserves"]),
        "genesis reserve shape differs",
    )
    result = JournalBinding(
        **{
            **raw,
            "initial_requests": tuple(
                request_from_document(row) for row in raw["initial_requests"]
            ),
            "reserves": tuple(
                tuple(request_from_document(row) for row in wave) for wave in raw["reserves"]
            ),
        }
    )
    require(canonical(result.document()) == payload, "genesis did not roundtrip exactly")
    return result


@dataclass(frozen=True, slots=True)
class JournalEvent:
    payload: bytes

    def __post_init__(self):
        raw = document(
            self.payload,
            fields=(
                "artifact",
                "schema_version",
                "genesis_sha256",
                "ordinal",
                "previous_event_sha256",
                "kind",
                "data",
            ),
        )
        require(
            raw["artifact"] == EVENT_ARTIFACT
            and type(raw["schema_version"]) is int
            and raw["schema_version"] == 1,
            "event header differs",
        )
        require(
            pin(raw["genesis_sha256"]) and pin(raw["previous_event_sha256"]),
            "event predecessor/genesis digest differs",
        )
        require(
            type(raw["ordinal"]) is int and 0 <= raw["ordinal"] < MAX_EVENTS,
            "event ordinal differs",
        )
        require(
            type(raw["kind"]) is str and raw["kind"] in KINDS and type(raw["data"]) is dict,
            "event kind/payload differs",
        )

    def document(self):
        self.__post_init__()
        return document(self.payload)

    @property
    def sha256(self):
        self.__post_init__()
        return digest(b"amp/durable-dispatch/event/v1\0" + self.payload)


def event_record(genesis_sha256, ordinal, previous_event_sha256, kind, data):
    return JournalEvent(
        canonical(
            {
                "artifact": EVENT_ARTIFACT,
                "schema_version": 1,
                "genesis_sha256": genesis_sha256,
                "ordinal": ordinal,
                "previous_event_sha256": previous_event_sha256,
                "kind": kind,
                "data": data,
            }
        )
    )


def event_name(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < MAX_EVENTS, "event name ordinal differs")
    return f"event-{ordinal:06d}"


def intent_token(genesis_sha256, intent_sha256):
    require(pin(genesis_sha256) and pin(intent_sha256), "intent token bindings differ")
    return digest(
        b"amp/durable-dispatch/token/v1\0"
        + canonical(
            {
                "genesis_sha256": genesis_sha256,
                "intent_sha256": intent_sha256,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    genesis_sha256: str
    intent_sha256: str
    wave_index: int
    seat_index: int
    charge_index: int
    request: JournalRequest

    def __post_init__(self):
        require(pin(self.genesis_sha256) and pin(self.intent_sha256), "dispatch pins differ")
        require(
            type(self.wave_index) is int
            and 1 <= self.wave_index <= 28
            and type(self.seat_index) is int
            and 0 <= self.seat_index < 16
            and type(self.charge_index) is int
            and self.charge_index == 64 + 16 * (self.wave_index - 1) + self.seat_index,
            "dispatch seat/charge differs",
        )
        require(type(self.request) is JournalRequest, "dispatch request type differs")
        self.request.__post_init__()

    @property
    def token_sha256(self):
        return intent_token(self.genesis_sha256, self.intent_sha256)

    def document(self):
        self.__post_init__()
        return {**asdict(self), "token_sha256": self.token_sha256}


@dataclass(frozen=True, slots=True)
class JournalCheckpoint:
    genesis_sha256: str
    event_count: int
    head_sha256: str
    charged_count: int

    def __post_init__(self):
        require(pin(self.genesis_sha256) and pin(self.head_sha256), "checkpoint digest differs")
        require(
            type(self.event_count) is int and 0 <= self.event_count <= MAX_EVENTS,
            "checkpoint event count differs",
        )
        require(
            type(self.charged_count) is int
            and (self.charged_count == 0 or 64 <= self.charged_count <= 512),
            "checkpoint charged count differs",
        )
        if self.event_count == 0:
            require(
                self.head_sha256 == self.genesis_sha256 and self.charged_count == 0,
                "empty checkpoint differs",
            )


@dataclass(frozen=True, slots=True)
class CallbackPin:
    """Externally held callable identity; no property lookup or invocation here."""

    target: object
    source_sha256: str

    def __post_init__(self):
        require(callable(self.target) and pin(self.source_sha256), "callback pin differs")


class ReceiptAuthenticator(Protocol):
    source_sha256: str

    def __call__(self, kind: str, receipt: bytes, expected: bytes) -> bytes:
        """Authenticate externally, returning canonical semantic bytes, not a Boolean."""


class DispatchTransport(Protocol):
    source_sha256: str

    def __call__(self, request: DispatchRequest) -> bytes:
        """Check original permission/deadline immediately before I/O; return raw ack bytes."""


class DispatchPermission(Protocol):
    source_sha256: str

    def __call__(self, request: DispatchRequest) -> bool:
        """Check original externally owned deadline and permission after intent fsync."""


def initial_expected(binding, kind, *, rows_sha256=None):
    """Exact authenticator expectation for aggregate source or same-seed copy receipt."""
    require(
        type(binding) is JournalBinding and kind in ("initial_source", "initial_copy"),
        "initial authentication kind differs",
    )
    binding.__post_init__()
    expected = {
        "kind": kind,
        "source_run_id": binding.initial_source_run_id,
        "seed": binding.seed,
        "objective_context_sha256": binding.objective_context_sha256,
        "oracle_bundle_sha256": binding.oracle_bundle_sha256,
        "requests": [row.document() for row in binding.initial_requests],
    }
    if kind == "initial_copy":
        require(pin(rows_sha256), "initial copy rows digest differs")
        expected.update(
            {
                "run_id": binding.run_id,
                "rows_sha256": rows_sha256,
                "source_receipt_sha256": binding.initial_source_receipt_sha256,
            }
        )
    return canonical(expected)


def dispatch_expected(binding, request, kind, *, external_submission_id=None):
    require(
        type(binding) is JournalBinding and type(request) is DispatchRequest,
        "dispatch authentication records differ",
    )
    binding.__post_init__()
    request.__post_init__()
    require(
        request.genesis_sha256 == binding.sha256
        and kind in ("submission_ack", "terminal_response"),
        "dispatch authority differs",
    )
    expected = {
        "kind": kind,
        "run_id": binding.run_id,
        "request": request.document(),
        "objective_context_sha256": binding.objective_context_sha256,
        "oracle_bundle_sha256": binding.oracle_bundle_sha256,
        "transport_sha256": binding.transport_sha256,
        "timing_context_sha256": binding.timing_context_sha256,
    }
    if kind == "terminal_response":
        require(identifier(external_submission_id), "terminal external ID differs")
        expected["external_submission_id"] = external_submission_id
    return canonical(expected)


def terminal_values(status, objectives):
    """Shape-only conversion; independent replay still authenticates each raw receipt."""
    require(type(status) is str and status in STATUSES, "terminal status differs")
    if status != "succeeded":
        require(objectives is None, "non-success terminal exposes objectives")
        return None
    require(
        type(objectives) is list
        and len(objectives) == 2
        and all(
            type(value) is float and math.isfinite(value) and 0 <= value <= 1
            for value in objectives
        ),
        "terminal objectives differ",
    )
    return tuple(objectives)


# Exact authenticated semantic documents (the raw receipt may have its own wire format):
# initial_source -> {"rows": [64 rows below]}; each row has request_sha256,
# external_submission_id, status, objectives, response_receipt_sha256. The source
# authenticator must establish these claims against initial_expected bytes.
# Original per-row raw receipts remain in the externally authenticated source
# ledger; this journal retains the aggregate source/copy receipt bytes, not
# fabricated individual receipt bodies or a claim of independently replaying them.
# initial_copy -> {"source_receipt_sha256": PIN, "rows_sha256": PIN, "run_id": ID,
#                  "seed": INT}; full canonical source rows are copied unchanged.
# submission_ack -> {"external_submission_id": ID}; expected bytes bind token/request/run.
# terminal_response -> {"external_submission_id": ID, "status": STATUS,
#                       "objectives": [FLOAT,FLOAT] or null}.
# Every event stores the authenticated document and original receipt hex separately.
INITIAL_ROW_FIELDS = frozenset(
    {"request_sha256", "external_submission_id", "status", "objectives", "response_receipt_sha256"}
)
COPY_FIELDS = frozenset({"source_receipt_sha256", "rows_sha256", "run_id", "seed"})
EVENT_DATA_FIELDS = {
    "initial_import": frozenset(
        {"source_receipt_hex", "copy_receipt_hex", "source_document", "copy_document"}
    ),
    "wave_seal": frozenset({"wave_index", "requests"}),
    "dispatch_intent": frozenset({"wave_index", "seat_index", "charge_index", "request"}),
    "submission_ack": frozenset({"intent_sha256", "receipt_hex", "authenticated"}),
    "terminal_response": frozenset({"intent_sha256", "receipt_hex", "authenticated"}),
    "dispatch_fault": frozenset({"intent_sha256", "stage", "error_type", "detail"}),
    "stop": frozenset({"reason", "detail"}),
}


@dataclass(frozen=True, slots=True)
class OutstandingDispatch:
    request: DispatchRequest
    external_submission_id: str | None

    def __post_init__(self):
        require(type(self.request) is DispatchRequest, "outstanding request differs")
        self.request.__post_init__()
        require(
            self.external_submission_id is None or identifier(self.external_submission_id),
            "outstanding external ID differs",
        )


@dataclass(frozen=True, slots=True)
class JournalReport:
    """Independent consistency report, not a clock/oracle/rollback authority.

    `history` is absent before initial import or on an ambiguous publication tail.
    Snapshot receipt_sha256 is the reconstructed current journal event head; its
    previous_wave_head_sha256 is instead the immutable initial/wave-seal event
    digest. Additional suffix adoption and unresolved charges are explicit.
    """

    binding: JournalBinding
    checkpoint: JournalCheckpoint
    history: VerifiedHistorySnapshot | None
    observations: tuple[ChargedObservation, ...]
    outstanding: tuple[OutstandingDispatch, ...]
    adaptive_attempts: int
    stopped: bool
    extension_requires_adoption: bool
    ambiguous_tail: bool
    journal_bytes: int
    event_sha256s: tuple[str, ...]

    def __post_init__(self):
        require(
            type(self.binding) is JournalBinding and type(self.checkpoint) is JournalCheckpoint,
            "report binding/checkpoint type differs",
        )
        self.binding.__post_init__()
        self.checkpoint.__post_init__()
        require(self.checkpoint.genesis_sha256 == self.binding.sha256, "report genesis differs")
        require(
            type(self.observations) is tuple
            and len(self.observations) <= 512
            and all(type(row) is ChargedObservation for row in self.observations),
            "report observations differ",
        )
        for row in self.observations:
            row.__post_init__()
        require(
            type(self.outstanding) is tuple
            and len(self.outstanding) <= 448
            and all(type(row) is OutstandingDispatch for row in self.outstanding),
            "report outstanding rows differ",
        )
        for row in self.outstanding:
            row.__post_init__()
        require(
            type(self.adaptive_attempts) is int and 0 <= self.adaptive_attempts <= 448,
            "report attempt count differs",
        )
        require(
            all(
                type(value) is bool
                for value in (self.stopped, self.extension_requires_adoption, self.ambiguous_tail)
            ),
            "report flags differ",
        )
        require(
            type(self.journal_bytes) is int and 0 <= self.journal_bytes <= MAX_JOURNAL_BYTES,
            "report bytes differ",
        )
        require(
            type(self.event_sha256s) is tuple
            and len(self.event_sha256s) == self.checkpoint.event_count
            and all(pin(value) for value in self.event_sha256s)
            and len(set(self.event_sha256s)) == len(self.event_sha256s),
            "report event inventory differs",
        )
        require(
            self.checkpoint.head_sha256
            == (self.event_sha256s[-1] if self.event_sha256s else self.binding.sha256),
            "report checkpoint is not its event tip",
        )
        require(
            self.checkpoint.charged_count == len(self.observations) + len(self.outstanding),
            "report omits or fabricates charged work",
        )
        if self.checkpoint.charged_count == 0:
            require(
                self.adaptive_attempts == 0 and not self.observations and not self.outstanding,
                "unimported report has charged work",
            )
        else:
            require(
                self.checkpoint.charged_count == 64 + self.adaptive_attempts
                and len(self.observations) >= 64,
                "report initial/adaptive arithmetic differs",
            )
            require(
                tuple((row.query_id, row.sequence) for row in self.observations[:64])
                == tuple((row.query_id, row.sequence) for row in self.binding.initial_requests),
                "report initial row identities differ",
            )
        require(
            tuple(row.charge_index for row in self.observations)
            == tuple(range(len(self.observations))),
            "report terminal prefix is not contiguous",
        )
        require(
            tuple(row.request.charge_index for row in self.outstanding)
            == tuple(range(len(self.observations), self.checkpoint.charged_count)),
            "report outstanding charge range differs",
        )
        for row in self.outstanding:
            require(
                row.request.genesis_sha256 == self.binding.sha256,
                "report outstanding genesis differs",
            )
            self.binding.validate_request(row.request.request)
        known = [(row.query_id, row.sequence) for row in self.observations]
        known.extend(
            (row.request.request.query_id, row.request.request.sequence) for row in self.outstanding
        )
        require(
            len({row[0] for row in known}) == len(known)
            and len({row[1] for row in known}) == len(known),
            "report known charge identities overlap",
        )
        require(
            len({row.request.request.identity.key for row in self.outstanding})
            == len(self.outstanding),
            "report outstanding query identities overlap",
        )
        require(
            self.history is None or type(self.history) is VerifiedHistorySnapshot,
            "report history type differs",
        )
        require(
            (self.history is None) == (self.ambiguous_tail or self.checkpoint.charged_count == 0),
            "report history presence differs from initial import/tail state",
        )
        if self.history is not None:
            self.history.__post_init__()
            require(
                not self.ambiguous_tail and self.history.observations == self.observations,
                "report history is not its complete authenticated prefix",
            )
            require(
                (
                    self.history.run_id,
                    self.history.seed,
                    self.history.objective_context_sha256,
                    self.history.oracle_bundle_sha256,
                    self.history.receipt_sha256,
                )
                == (
                    self.binding.run_id,
                    self.binding.seed,
                    self.binding.objective_context_sha256,
                    self.binding.oracle_bundle_sha256,
                    self.checkpoint.head_sha256,
                ),
                "report history run/context/event receipt differs",
            )
            require(
                self.history.previous_wave_head_sha256 in self.event_sha256s,
                "report history wave head is not a known event",
            )

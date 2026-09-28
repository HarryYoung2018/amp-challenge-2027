"""Execution-disabled state machine for a future evolutionary/KL campaign.

This module is deliberately an engineering boundary rather than a campaign
runner.  It contains no network, oracle, model, GPU, filesystem publication, or
production-adapter implementation.  Exact fake ports exercise request sealing,
logical-call accounting, timing chains, wave archival, and terminal decisions.
A later independently reviewed successor must replace the fake boundary and add
durable phase publication without weakening these transitions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from amp_challenge.evaluation.evolutionary_kl_protocol import (
    CONFIRMATION_METHOD_IDS,
    CONFIRMATION_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    SCREEN_CONFIGURATION_IDS,
    SCREEN_SEEDS,
)
from amp_challenge.evaluation.sequential_v2_seals import canonical_json_bytes, sha256_bytes
from amp_challenge.generators.search.campaign_ledger import OracleQueryIdentity

CONTROLLER_V2_REGISTRY_SHA256 = "7bf1afa4282c90f71d6607a086e771da925f83482d7687ab0e1a1244439caaac"
ENGINEERING_EVIDENCE = "engineering_fixture_only_not_scientific_evidence"
INITIAL_CALLS = 64
ADAPTIVE_WAVES = 28
CALLS_PER_WAVE = 16
METHOD_SEATS_PER_WAVE = 14
RESERVE_SEATS_PER_WAVE = 2
TOTAL_UNIQUE_CALLS = 512
METHOD_PROPOSAL_ATTEMPT_CAP = 65_536
SCIENTIFIC_DEADLINE_NS = 7_200_000_000_000
SEALING_ALLOWANCE_NS = 900_000_000_000
MINIMUM_TERMINAL_ELIGIBLE = 100
REQUIRED_ASSET_ROLES = (
    "calibrated_joint_posterior",
    "checkpoint_aggregation",
    "endpoint_context",
    "external_timing_issuer",
    "generated_sequence_contacts",
    "generator_oracle_provenance",
    "homology_exclusion",
    "method_adapter",
    "native_sampler",
    "oracle_checkpoint",
    "oracle_contract",
    "oracle_evaluator",
    "oracle_transform",
    "telemetry_pipeline",
    "terminal_contract",
)
TERMINAL_ORACLE_STATUSES = (
    "succeeded",
    "failed",
    "missing",
    "censored",
    "partial",
    "timeout",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ROLE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class CampaignControllerV2Error(ValueError):
    """Raised when an engineering transition would weaken the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignControllerV2Error(message)


def _sha256(value: object, *, label: str) -> str:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None, f"{label} is invalid")
    assert isinstance(value, str)
    return value


def _identifier(value: object, *, label: str) -> str:
    _require(
        type(value) is str and _IDENTIFIER.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    assert isinstance(value, str)
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a non-negative integer")
    assert isinstance(value, int)
    return value


def _finite_float(value: object, *, label: str) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    assert isinstance(value, float)
    return value


def _hash_document(domain: bytes, document: object) -> str:
    return sha256_bytes(domain + canonical_json_bytes(document))


def _query_document(identity: OracleQueryIdentity) -> dict[str, object]:
    _require(type(identity) is OracleQueryIdentity, "query identity must have its exact v1 type")
    identity.__post_init__()
    return identity.document()


@dataclass(frozen=True, slots=True)
class AssetDigestBindingV2:
    """One fake-only asset identity and its separately named receipt digest."""

    role: str
    payload_sha256: str
    receipt_sha256: str
    evidence_class: Literal["engineering_fixture_only_not_scientific_evidence"] = (
        ENGINEERING_EVIDENCE
    )
    independently_verified: Literal[False] = False
    production_input_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        _require(
            type(self.role) is str and _ROLE.fullmatch(self.role) is not None, "asset role invalid"
        )
        _sha256(self.payload_sha256, label=f"{self.role} payload")
        _sha256(self.receipt_sha256, label=f"{self.role} receipt")
        _require(self.evidence_class == ENGINEERING_EVIDENCE, "v2 asset must remain a fixture")
        _require(
            self.independently_verified is False, "v2 asset cannot claim independent acceptance"
        )
        _require(
            self.production_input_eligible is False, "v2 asset cannot claim production eligibility"
        )

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "evidence_class": self.evidence_class,
            "independently_verified": self.independently_verified,
            "payload_sha256": self.payload_sha256,
            "production_input_eligible": self.production_input_eligible,
            "receipt_sha256": self.receipt_sha256,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class CampaignAuthorityV2:
    """Path-free identity for one strictly non-authorizing engineering campaign."""

    campaign_id: str
    phase: Literal["screen", "confirmation"]
    configuration_id: str
    seed: int
    protocol_sha256: str
    successor_registry_sha256: str
    source_commit: str
    source_tree_sha256: str
    environment_lock_sha256: str
    assets: tuple[AssetDigestBindingV2, ...]
    initial_queries: tuple[OracleQueryIdentity, ...]
    common_reserve_by_wave: tuple[tuple[OracleQueryIdentity, OracleQueryIdentity], ...]
    overflow_queries: tuple[OracleQueryIdentity, ...]

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, label="campaign ID")
        _require(self.phase in {"screen", "confirmation"}, "campaign phase is invalid")
        configurations = (
            SCREEN_CONFIGURATION_IDS if self.phase == "screen" else CONFIRMATION_METHOD_IDS
        )
        seeds = SCREEN_SEEDS if self.phase == "screen" else CONFIRMATION_SEEDS
        _require(self.configuration_id in configurations, "configuration is not frozen for phase")
        _require(type(self.seed) is int and self.seed in seeds, "seed is not frozen for phase")
        _require(self.protocol_sha256 == FROZEN_PROTOCOL_SHA256, "protocol digest differs")
        _require(
            self.successor_registry_sha256 == CONTROLLER_V2_REGISTRY_SHA256,
            "successor registry digest differs",
        )
        _require(
            type(self.source_commit) is str and _GIT_SHA1.fullmatch(self.source_commit) is not None,
            "source commit must be a full lowercase Git SHA-1",
        )
        _sha256(self.source_tree_sha256, label="source tree")
        _sha256(self.environment_lock_sha256, label="environment lock")
        self._validate_assets()
        self._validate_schedules()

    def _validate_assets(self) -> None:
        _require(type(self.assets) is tuple, "asset bindings must be an exact tuple")
        _require(
            all(type(asset) is AssetDigestBindingV2 for asset in self.assets),
            "asset bindings must have exact v2 types",
        )
        roles = tuple(asset.role for asset in self.assets)
        _require(roles == REQUIRED_ASSET_ROLES, "asset roles must equal the frozen sorted role set")
        payloads = tuple(asset.payload_sha256 for asset in self.assets)
        receipts = tuple(asset.receipt_sha256 for asset in self.assets)
        _require(len(payloads) == len(set(payloads)), "asset payload digests must be unique")
        _require(len(receipts) == len(set(receipts)), "asset receipt digests must be unique")
        for asset in self.assets:
            asset.__post_init__()

    def _validate_schedules(self) -> None:
        _require(
            type(self.initial_queries) is tuple and len(self.initial_queries) == INITIAL_CALLS,
            "initial schedule must contain exactly 64 queries",
        )
        _require(
            type(self.common_reserve_by_wave) is tuple
            and len(self.common_reserve_by_wave) == ADAPTIVE_WAVES,
            "common reserve schedule must contain exactly 28 waves",
        )
        reserve: list[OracleQueryIdentity] = []
        for pair in self.common_reserve_by_wave:
            _require(
                type(pair) is tuple
                and len(pair) == RESERVE_SEATS_PER_WAVE
                and all(type(identity) is OracleQueryIdentity for identity in pair),
                "each common reserve wave must contain exactly two exact query identities",
            )
            reserve.extend(pair)
        _require(
            type(self.overflow_queries) is tuple
            and bool(self.overflow_queries)
            and all(type(identity) is OracleQueryIdentity for identity in self.overflow_queries),
            "overflow reserve must be a non-empty exact tuple of query identities",
        )
        frozen = (*self.initial_queries, *reserve, *self.overflow_queries)
        for identity in frozen:
            self.validate_query_identity(identity)
        keys = tuple(identity.key for identity in frozen)
        _require(
            len(keys) == len(set(keys)),
            "initial, common-reserve, and overflow query identities must be pairwise disjoint",
        )

    @property
    def asset_map(self) -> dict[str, AssetDigestBindingV2]:
        self.__post_init__()
        return {asset.role: asset for asset in self.assets}

    def validate_query_identity(self, identity: OracleQueryIdentity) -> None:
        _query_document(identity)
        assets = {asset.role: asset for asset in self.assets}
        expected = {
            "checkpoint_sha256": assets["oracle_checkpoint"].payload_sha256,
            "endpoint_context_sha256": assets["endpoint_context"].payload_sha256,
            "evaluator_sha256": assets["oracle_evaluator"].payload_sha256,
            "oracle_contract_sha256": assets["oracle_contract"].payload_sha256,
            "transform_sha256": assets["oracle_transform"].payload_sha256,
        }
        for name, digest in expected.items():
            _require(
                getattr(identity, name) == digest, f"query identity {name} is not authority-bound"
            )

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_campaign_authority_v2_engineering_fixture",
            "assets": [asset.document() for asset in self.assets],
            "authorization": {
                "automatic_production_eligible": False,
                "biological_superiority_claim_allowed": False,
                "execution_authorized": False,
                "network_oracle_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "campaign_id": self.campaign_id,
            "common_reserve_by_wave": [
                [_query_document(identity) for identity in pair]
                for pair in self.common_reserve_by_wave
            ],
            "configuration_id": self.configuration_id,
            "environment_lock_sha256": self.environment_lock_sha256,
            "evidence_class": ENGINEERING_EVIDENCE,
            "initial_queries": [_query_document(identity) for identity in self.initial_queries],
            "limits": {
                "adaptive_waves": ADAPTIVE_WAVES,
                "calls_per_wave": CALLS_PER_WAVE,
                "initial_calls": INITIAL_CALLS,
                "method_proposal_attempt_cap": METHOD_PROPOSAL_ATTEMPT_CAP,
                "method_seats_per_wave": METHOD_SEATS_PER_WAVE,
                "minimum_terminal_eligible": MINIMUM_TERMINAL_ELIGIBLE,
                "reserve_seats_per_wave": RESERVE_SEATS_PER_WAVE,
                "scientific_deadline_ns": SCIENTIFIC_DEADLINE_NS,
                "sealing_allowance_ns": SEALING_ALLOWANCE_NS,
                "total_unique_calls": TOTAL_UNIQUE_CALLS,
            },
            "overflow_queries": [_query_document(identity) for identity in self.overflow_queries],
            "phase": self.phase,
            "protocol_sha256": self.protocol_sha256,
            "schema_version": 2,
            "seed": self.seed,
            "source_commit": self.source_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "status": "execution_disabled_typed_transition_validator_only",
            "successor_registry_sha256": self.successor_registry_sha256,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/campaign-authority/v2\0", self.document())


RequestPhase = Literal["initial", "adaptive"]
RequestSource = Literal["initial", "method", "overflow", "common_reserve"]


@dataclass(frozen=True, slots=True)
class OracleRequestV2:
    """One logical request whose identity and seat are fixed before submission."""

    authority_sha256: str
    phase: RequestPhase
    wave_index: int
    seat_position: int
    source: RequestSource
    identity: OracleQueryIdentity
    hard_valid: Literal[True] = True

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="request authority")
        _require(self.phase in {"initial", "adaptive"}, "request phase is invalid")
        _nonnegative_integer(self.seat_position, label="request seat position")
        _query_document(self.identity)
        _require(self.hard_valid is True, "sealed oracle request must be hard-valid")
        if self.phase == "initial":
            _require(self.wave_index == -1, "initial request wave index must be -1")
            _require(0 <= self.seat_position < INITIAL_CALLS, "initial request seat is invalid")
            _require(self.source == "initial", "initial request source differs")
        else:
            _require(
                type(self.wave_index) is int and 0 <= self.wave_index < ADAPTIVE_WAVES,
                "adaptive request wave index is invalid",
            )
            _require(0 <= self.seat_position < CALLS_PER_WAVE, "adaptive request seat is invalid")
            if self.seat_position < METHOD_SEATS_PER_WAVE:
                _require(
                    self.source in {"method", "overflow"},
                    "method-controlled seat has an invalid source",
                )
            else:
                _require(
                    self.source == "common_reserve",
                    "common-reserve seat source differs",
                )

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "authority_sha256": self.authority_sha256,
            "hard_valid": self.hard_valid,
            "identity": _query_document(self.identity),
            "phase": self.phase,
            "schema_version": 2,
            "seat_position": self.seat_position,
            "source": self.source,
            "wave_index": self.wave_index,
        }

    @property
    def request_id(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/oracle-request/v2\0", self.document())


@dataclass(frozen=True, slots=True)
class RequestWaveSealV2:
    """Immutable request plan that must exist before any external submission."""

    authority_sha256: str
    phase: RequestPhase
    wave_index: int
    previous_archive_sha256: str | None
    requests: tuple[OracleRequestV2, ...]

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="request-wave authority")
        _require(self.phase in {"initial", "adaptive"}, "request-wave phase is invalid")
        if self.previous_archive_sha256 is not None:
            _sha256(self.previous_archive_sha256, label="previous archive")
        if self.phase == "initial":
            _require(self.wave_index == -1, "initial request-wave index must be -1")
            _require(self.previous_archive_sha256 is None, "initial request wave has predecessor")
        else:
            _require(
                type(self.wave_index) is int and 0 <= self.wave_index < ADAPTIVE_WAVES,
                "adaptive request-wave index is invalid",
            )
            _require(
                self.previous_archive_sha256 is not None,
                "adaptive request wave must follow an archived predecessor",
            )
        _require(type(self.requests) is tuple, "request wave must contain an exact tuple")
        expected_size = INITIAL_CALLS if self.phase == "initial" else CALLS_PER_WAVE
        expected_wave = -1 if self.phase == "initial" else self.wave_index
        _require(len(self.requests) == expected_size, "request wave size differs")
        _require(
            tuple(request.seat_position for request in self.requests)
            == tuple(range(expected_size)),
            "request wave seats must be contiguous and ordered",
        )
        for request in self.requests:
            _require(type(request) is OracleRequestV2, "request wave contains a non-v2 request")
            request.__post_init__()
            _require(request.authority_sha256 == self.authority_sha256, "request authority differs")
            _require(request.phase == self.phase, "request phase differs within wave")
            _require(request.wave_index == expected_wave, "request wave index differs")
        keys = tuple(request.identity.key for request in self.requests)
        _require(len(keys) == len(set(keys)), "request wave contains duplicate logical identities")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_request_wave_seal_v2",
            "authority_sha256": self.authority_sha256,
            "evidence_class": ENGINEERING_EVIDENCE,
            "phase": self.phase,
            "previous_archive_sha256": self.previous_archive_sha256,
            "requests": [request.document() for request in self.requests],
            "schema_version": 2,
            "scientific_evidence_accepted": False,
            "wave_index": self.wave_index,
        }

    @property
    def seal_sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/request-wave-seal/v2\0", self.document())


@dataclass(frozen=True, slots=True)
class SubmissionReceiptV2:
    """Fake external acknowledgement for exactly one already-sealed request."""

    authority_sha256: str
    request_seal_sha256: str
    request_id: str
    identity_key: str
    external_submission_id: str
    transport_sha256: str
    evidence_class: Literal["engineering_fixture_only_not_scientific_evidence"] = (
        ENGINEERING_EVIDENCE
    )

    def __post_init__(self) -> None:
        for label, digest in (
            ("submission authority", self.authority_sha256),
            ("request seal", self.request_seal_sha256),
            ("request ID", self.request_id),
            ("query identity key", self.identity_key),
            ("transport", self.transport_sha256),
        ):
            _sha256(digest, label=label)
        _identifier(self.external_submission_id, label="external submission ID")
        _require(self.evidence_class == ENGINEERING_EVIDENCE, "submission receipt is not a fixture")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_fake_submission_receipt_v2",
            "authority_sha256": self.authority_sha256,
            "evidence_class": self.evidence_class,
            "execution_authorized": False,
            "external_submission_id": self.external_submission_id,
            "identity_key": self.identity_key,
            "request_id": self.request_id,
            "request_seal_sha256": self.request_seal_sha256,
            "schema_version": 2,
            "transport_sha256": self.transport_sha256,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/fake-submission-receipt/v2\0", self.document())


OracleTerminalStatus = Literal["succeeded", "failed", "missing", "censored", "partial", "timeout"]


@dataclass(frozen=True, slots=True)
class OracleTerminalResultV2:
    """Fixed fake-oracle terminal result; every status consumes its submitted call."""

    status: OracleTerminalStatus
    gram_positive_activity: float | None
    gram_negative_activity: float | None
    constraints: tuple[tuple[str, bool], ...]
    detail: str | None

    def __post_init__(self) -> None:
        _require(self.status in TERMINAL_ORACLE_STATUSES, "oracle terminal status is invalid")
        _require(type(self.constraints) is tuple, "oracle constraints must be an exact tuple")
        if self.status == "succeeded":
            _finite_float(self.gram_positive_activity, label="Gram-positive activity")
            _finite_float(self.gram_negative_activity, label="Gram-negative activity")
            _require(bool(self.constraints), "successful result must contain every constraint")
            names: list[str] = []
            for row in self.constraints:
                _require(
                    type(row) is tuple
                    and len(row) == 2
                    and type(row[0]) is str
                    and _ROLE.fullmatch(row[0]) is not None
                    and type(row[1]) is bool,
                    "oracle constraint row is invalid",
                )
                names.append(row[0])
            _require(
                tuple(names) == tuple(sorted(set(names))), "constraints must be sorted and unique"
            )
            _require(self.detail is None, "successful result cannot carry failure detail")
        else:
            _require(
                self.gram_positive_activity is None and self.gram_negative_activity is None,
                "non-success result cannot carry objective values",
            )
            _require(not self.constraints, "non-success result cannot carry constraint values")
            _identifier(self.detail, label="oracle failure detail")

    @property
    def eligible(self) -> bool:
        self.__post_init__()
        return self.status == "succeeded" and all(value for _name, value in self.constraints)

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "constraints": [list(row) for row in self.constraints],
            "detail": self.detail,
            "gram_negative_activity": self.gram_negative_activity,
            "gram_positive_activity": self.gram_positive_activity,
            "status": self.status,
        }


PollStatus = Literal["pending", "succeeded", "failed", "missing", "censored", "partial", "timeout"]


@dataclass(frozen=True, slots=True)
class PollReceiptV2:
    """One poll of the sole external submission ID assigned to a request."""

    authority_sha256: str
    request_seal_sha256: str
    request_id: str
    identity_key: str
    external_submission_id: str
    transport_sha256: str
    poll_ordinal: int
    status: PollStatus
    terminal_result: OracleTerminalResultV2 | None
    evidence_class: Literal["engineering_fixture_only_not_scientific_evidence"] = (
        ENGINEERING_EVIDENCE
    )

    def __post_init__(self) -> None:
        for label, digest in (
            ("poll authority", self.authority_sha256),
            ("request seal", self.request_seal_sha256),
            ("request ID", self.request_id),
            ("query identity key", self.identity_key),
            ("transport", self.transport_sha256),
        ):
            _sha256(digest, label=label)
        _identifier(self.external_submission_id, label="external submission ID")
        _nonnegative_integer(self.poll_ordinal, label="poll ordinal")
        _require(self.status in {"pending", *TERMINAL_ORACLE_STATUSES}, "poll status is invalid")
        if self.status == "pending":
            _require(self.terminal_result is None, "pending poll cannot carry a terminal result")
        else:
            _require(
                type(self.terminal_result) is OracleTerminalResultV2,
                "terminal poll requires an exact result",
            )
            assert self.terminal_result is not None
            self.terminal_result.__post_init__()
            _require(self.terminal_result.status == self.status, "poll and result statuses differ")
        _require(self.evidence_class == ENGINEERING_EVIDENCE, "poll receipt is not a fixture")

    @property
    def is_terminal(self) -> bool:
        return self.status != "pending"

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_fake_poll_receipt_v2",
            "authority_sha256": self.authority_sha256,
            "evidence_class": self.evidence_class,
            "execution_authorized": False,
            "external_submission_id": self.external_submission_id,
            "identity_key": self.identity_key,
            "poll_ordinal": self.poll_ordinal,
            "request_id": self.request_id,
            "request_seal_sha256": self.request_seal_sha256,
            "schema_version": 2,
            "status": self.status,
            "terminal_result": (
                None if self.terminal_result is None else self.terminal_result.document()
            ),
            "transport_sha256": self.transport_sha256,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/fake-poll-receipt/v2\0", self.document())


TimingStream = Literal["scientific", "sealing"]


@dataclass(frozen=True, slots=True)
class TimingReceiptV2:
    """Fake-only cumulative receipt in one of two independently chained clocks."""

    authority_sha256: str
    issuer_sha256: str
    stream: TimingStream
    ordinal: int
    predecessor_sha256: str | None
    cumulative_elapsed_ns: int
    cumulative_wall_elapsed_ns: int
    segment_id: str
    evidence_class: Literal["engineering_fixture_only_not_scientific_evidence"] = (
        ENGINEERING_EVIDENCE
    )

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="timing authority")
        _sha256(self.issuer_sha256, label="timing issuer")
        _require(self.stream in {"scientific", "sealing"}, "timing stream is invalid")
        _nonnegative_integer(self.ordinal, label="timing ordinal")
        if self.predecessor_sha256 is not None:
            _sha256(self.predecessor_sha256, label="timing predecessor")
        _nonnegative_integer(self.cumulative_elapsed_ns, label="cumulative elapsed ns")
        _nonnegative_integer(
            self.cumulative_wall_elapsed_ns,
            label="cumulative wall elapsed ns",
        )
        _identifier(self.segment_id, label="timing segment ID")
        _require(self.evidence_class == ENGINEERING_EVIDENCE, "timing receipt is not a fixture")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_fake_timing_receipt_v2",
            "authority_sha256": self.authority_sha256,
            "cumulative_monotonic_elapsed_ns": self.cumulative_elapsed_ns,
            "cumulative_wall_elapsed_ns": self.cumulative_wall_elapsed_ns,
            "evidence_class": self.evidence_class,
            "execution_authorized": False,
            "issuer_sha256": self.issuer_sha256,
            "ordinal": self.ordinal,
            "predecessor_sha256": self.predecessor_sha256,
            "schema_version": 2,
            "segment_id": self.segment_id,
            "stream": self.stream,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/fake-timing-receipt/v2\0", self.document())


@dataclass(frozen=True, slots=True)
class ArchivedWaveV2:
    """Complete evidence wave; every sealed request has one terminal poll."""

    authority_sha256: str
    request_seal: RequestWaveSealV2
    previous_archive_sha256: str | None
    submissions: tuple[SubmissionReceiptV2, ...]
    polls: tuple[PollReceiptV2, ...]
    scientific_elapsed_ns: int
    scientific_wall_elapsed_ns: int
    scientific_timing_head_sha256: str | None
    sealing_elapsed_ns: int
    sealing_wall_elapsed_ns: int
    sealing_timing_receipt_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="archive authority")
        _require(type(self.request_seal) is RequestWaveSealV2, "archive request seal type differs")
        self.request_seal.__post_init__()
        _require(
            self.request_seal.authority_sha256 == self.authority_sha256,
            "archive request authority differs",
        )
        _require(
            self.request_seal.previous_archive_sha256 == self.previous_archive_sha256,
            "archive predecessor differs from request seal",
        )
        if self.previous_archive_sha256 is not None:
            _sha256(self.previous_archive_sha256, label="archive predecessor")
        _require(type(self.submissions) is tuple, "archive submissions must be an exact tuple")
        _require(type(self.polls) is tuple, "archive polls must be an exact tuple")
        requests = self.request_seal.requests
        expected_ids = tuple(request.request_id for request in requests)
        _require(
            tuple(receipt.request_id for receipt in self.submissions) == expected_ids,
            "archive must contain one ordered submission per request",
        )
        external_ids: set[str] = set()
        for request, submission in zip(requests, self.submissions, strict=True):
            _require(type(submission) is SubmissionReceiptV2, "archive submission type differs")
            submission.__post_init__()
            _require(
                submission.authority_sha256 == self.authority_sha256, "submission authority differs"
            )
            _require(
                submission.request_seal_sha256 == self.request_seal.seal_sha256,
                "submission request seal differs",
            )
            _require(submission.identity_key == request.identity.key, "submission identity differs")
            _require(
                submission.external_submission_id not in external_ids,
                "external submission ID is reused within an archive",
            )
            external_ids.add(submission.external_submission_id)
        poll_ids = tuple(receipt.request_id for receipt in self.polls)
        _require(set(poll_ids) == set(expected_ids), "archive is missing a request poll chain")
        request_positions = {
            request_id: position for position, request_id in enumerate(expected_ids)
        }
        poll_order = tuple(
            (request_positions[receipt.request_id], receipt.poll_ordinal) for receipt in self.polls
        )
        _require(
            poll_order == tuple(sorted(poll_order)),
            "archive polls must follow sealed request order",
        )
        for request_id in expected_ids:
            chain = tuple(receipt for receipt in self.polls if receipt.request_id == request_id)
            _require(bool(chain), "archive poll chain is empty")
            _require(
                tuple(receipt.poll_ordinal for receipt in chain) == tuple(range(len(chain))),
                "archive poll ordinals are not contiguous",
            )
            _require(chain[-1].is_terminal, "archive request does not have a terminal poll")
            _require(
                sum(receipt.is_terminal for receipt in chain) == 1,
                "archive request has multiple terminal polls",
            )
            submission = self.submissions[expected_ids.index(request_id)]
            for receipt in chain:
                _require(type(receipt) is PollReceiptV2, "archive poll type differs")
                receipt.__post_init__()
                _require(
                    receipt.authority_sha256 == self.authority_sha256, "poll authority differs"
                )
                _require(
                    receipt.request_seal_sha256 == self.request_seal.seal_sha256,
                    "poll request seal differs",
                )
                _require(
                    receipt.identity_key == submission.identity_key,
                    "poll identity differs from its submission",
                )
                _require(
                    receipt.transport_sha256 == submission.transport_sha256,
                    "poll transport differs from its submission",
                )
                _require(
                    receipt.external_submission_id == submission.external_submission_id,
                    "poll changed the sole external submission ID",
                )
        _nonnegative_integer(self.scientific_elapsed_ns, label="archive scientific elapsed ns")
        _nonnegative_integer(
            self.scientific_wall_elapsed_ns,
            label="archive scientific wall elapsed ns",
        )
        _require(
            self.scientific_elapsed_ns <= SCIENTIFIC_DEADLINE_NS
            and self.scientific_wall_elapsed_ns <= SCIENTIFIC_DEADLINE_NS,
            "archive scientific deadline exceeded",
        )
        if self.request_seal.phase == "initial":
            _require(
                self.scientific_elapsed_ns == 0
                and self.scientific_wall_elapsed_ns == 0
                and self.scientific_timing_head_sha256 is None,
                "initial archive must precede the scientific clock",
            )
        else:
            _sha256(self.scientific_timing_head_sha256, label="scientific timing head")
        _nonnegative_integer(self.sealing_elapsed_ns, label="archive sealing elapsed ns")
        _nonnegative_integer(
            self.sealing_wall_elapsed_ns,
            label="archive sealing wall elapsed ns",
        )
        _require(
            self.sealing_elapsed_ns <= SEALING_ALLOWANCE_NS
            and self.sealing_wall_elapsed_ns <= SEALING_ALLOWANCE_NS,
            "archive sealing allowance exceeded",
        )
        _sha256(self.sealing_timing_receipt_sha256, label="archive sealing timing receipt")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_complete_wave_archive_v2",
            "authority_sha256": self.authority_sha256,
            "authorization": {
                "external_acceptance_authorized": False,
                "execution_authorized": False,
                "resume_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "evidence_class": ENGINEERING_EVIDENCE,
            "polls": [receipt.document() for receipt in self.polls],
            "previous_archive_sha256": self.previous_archive_sha256,
            "request_seal": self.request_seal.document(),
            "request_seal_sha256": self.request_seal.seal_sha256,
            "schema_version": 2,
            "scientific_elapsed_ns": self.scientific_elapsed_ns,
            "scientific_wall_elapsed_ns": self.scientific_wall_elapsed_ns,
            "scientific_timing_head_sha256": self.scientific_timing_head_sha256,
            "sealing_elapsed_ns": self.sealing_elapsed_ns,
            "sealing_wall_elapsed_ns": self.sealing_wall_elapsed_ns,
            "sealing_timing_receipt_sha256": self.sealing_timing_receipt_sha256,
            "submissions": [receipt.document() for receipt in self.submissions],
        }

    @property
    def archive_sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/complete-wave-archive/v2\0", self.document())


@dataclass(frozen=True, slots=True)
class DiscardedWaveV2:
    """Non-evidence record of an unarchived wave whose submissions remain charged."""

    authority_sha256: str
    request_seal: RequestWaveSealV2
    submissions: tuple[SubmissionReceiptV2, ...]
    polls: tuple[PollReceiptV2, ...]
    reason: Literal["scientific_deadline"]
    scientific_elapsed_ns: int
    scientific_wall_elapsed_ns: int
    scientific_timing_receipt_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="discard authority")
        _require(type(self.request_seal) is RequestWaveSealV2, "discard request seal type differs")
        self.request_seal.__post_init__()
        _require(self.request_seal.phase == "adaptive", "initial schedule cannot be discarded")
        _require(
            self.request_seal.authority_sha256 == self.authority_sha256,
            "discard request authority differs",
        )
        _require(type(self.submissions) is tuple, "discard submissions must be an exact tuple")
        _require(type(self.polls) is tuple, "discard polls must be an exact tuple")
        request_by_id = {request.request_id: request for request in self.request_seal.requests}
        expected_request_ids = tuple(request_by_id)
        _require(
            tuple(submission.request_id for submission in self.submissions)
            == expected_request_ids[: len(self.submissions)],
            "discard submissions must be a sealed-order prefix",
        )
        seen_requests: set[str] = set()
        external_ids: set[str] = set()
        for submission in self.submissions:
            _require(type(submission) is SubmissionReceiptV2, "discard submission type differs")
            submission.__post_init__()
            request = request_by_id.get(submission.request_id)
            _require(request is not None, "discard submission is outside the request seal")
            assert request is not None
            _require(submission.request_id not in seen_requests, "discard has duplicate submission")
            _require(
                submission.authority_sha256 == self.authority_sha256, "discard authority differs"
            )
            _require(
                submission.request_seal_sha256 == self.request_seal.seal_sha256,
                "discard submission request seal differs",
            )
            _require(
                submission.external_submission_id not in external_ids,
                "discard reuses an external submission ID",
            )
            _require(submission.identity_key == request.identity.key, "discard identity differs")
            seen_requests.add(submission.request_id)
            external_ids.add(submission.external_submission_id)
        submission_by_request = {receipt.request_id: receipt for receipt in self.submissions}
        poll_chains: dict[str, list[PollReceiptV2]] = {
            request_id: [] for request_id in seen_requests
        }
        for poll in self.polls:
            _require(type(poll) is PollReceiptV2, "discard poll type differs")
            poll.__post_init__()
            _require(poll.request_id in seen_requests, "discard poll lacks a charged submission")
            submission = submission_by_request[poll.request_id]
            request = request_by_id[poll.request_id]
            _require(
                poll.authority_sha256 == self.authority_sha256, "discard poll authority differs"
            )
            _require(
                poll.request_seal_sha256 == self.request_seal.seal_sha256,
                "discard poll request seal differs",
            )
            _require(poll.identity_key == request.identity.key, "discard poll identity differs")
            _require(
                poll.transport_sha256 == submission.transport_sha256,
                "discard poll transport differs from its submission",
            )
            _require(
                poll.external_submission_id == submission.external_submission_id,
                "discard poll changed the sole external submission ID",
            )
            poll_chains[poll.request_id].append(poll)
        for chain in poll_chains.values():
            _require(
                tuple(receipt.poll_ordinal for receipt in chain) == tuple(range(len(chain))),
                "discard poll ordinals are not contiguous",
            )
            _require(
                sum(receipt.is_terminal for receipt in chain) <= 1,
                "discard contains multiple terminal polls for one request",
            )
            if any(receipt.is_terminal for receipt in chain):
                _require(chain[-1].is_terminal, "discard polls continue after a terminal result")
        request_positions = {
            request_id: position for position, request_id in enumerate(expected_request_ids)
        }
        poll_order = tuple(
            (request_positions[poll.request_id], poll.poll_ordinal) for poll in self.polls
        )
        _require(
            poll_order == tuple(sorted(poll_order)),
            "discard polls must follow sealed request order",
        )
        _require(self.reason == "scientific_deadline", "discard reason differs")
        _nonnegative_integer(self.scientific_elapsed_ns, label="discard scientific elapsed ns")
        _nonnegative_integer(
            self.scientific_wall_elapsed_ns,
            label="discard scientific wall elapsed ns",
        )
        _require(
            max(self.scientific_elapsed_ns, self.scientific_wall_elapsed_ns)
            == SCIENTIFIC_DEADLINE_NS,
            "deadline discard must bind the exact scientific ceiling",
        )
        _require(
            0 <= self.scientific_wall_elapsed_ns <= SCIENTIFIC_DEADLINE_NS,
            "discard scientific wall elapsed is invalid",
        )
        _sha256(self.scientific_timing_receipt_sha256, label="discard timing receipt")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "artifact": "evolutionary_kl_discarded_unarchived_wave_v2",
            "authority_sha256": self.authority_sha256,
            "authorization": {
                "external_acceptance_authorized": False,
                "execution_authorized": False,
                "resume_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "charged_submission_count": len(self.submissions),
            "evidence_class": ENGINEERING_EVIDENCE,
            "enters_scientific_archive": False,
            "polls": [receipt.document() for receipt in self.polls],
            "reason": self.reason,
            "request_seal": self.request_seal.document(),
            "request_seal_sha256": self.request_seal.seal_sha256,
            "schema_version": 2,
            "scientific_elapsed_ns": self.scientific_elapsed_ns,
            "scientific_wall_elapsed_ns": self.scientific_wall_elapsed_ns,
            "scientific_timing_receipt_sha256": self.scientific_timing_receipt_sha256,
            "submissions": [receipt.document() for receipt in self.submissions],
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/discarded-wave/v2\0", self.document())


@dataclass(frozen=True, slots=True)
class PosteriorCandidateV2:
    """Terminal posterior means for one archived successful query identity."""

    identity_key: str
    canonical_sequence_id: str
    gram_positive_mean: float
    gram_negative_mean: float

    def __post_init__(self) -> None:
        _sha256(self.identity_key, label="posterior identity key")
        _sha256(self.canonical_sequence_id, label="posterior canonical sequence ID")
        _finite_float(self.gram_positive_mean, label="posterior Gram-positive mean")
        _finite_float(self.gram_negative_mean, label="posterior Gram-negative mean")

    @property
    def utility(self) -> float:
        self.__post_init__()
        return 0.5 * (self.gram_positive_mean + self.gram_negative_mean)

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "canonical_sequence_id": self.canonical_sequence_id,
            "gram_negative_mean": self.gram_negative_mean,
            "gram_positive_mean": self.gram_positive_mean,
            "identity_key": self.identity_key,
        }


@dataclass(frozen=True, slots=True)
class PosteriorSnapshotV2:
    """Complete fake posterior table for all archived successful requests."""

    authority_sha256: str
    archive_head_sha256: str
    posterior_model_sha256: str
    candidates: tuple[PosteriorCandidateV2, ...]
    evidence_class: Literal["engineering_fixture_only_not_scientific_evidence"] = (
        ENGINEERING_EVIDENCE
    )

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="posterior authority")
        _sha256(self.archive_head_sha256, label="posterior archive head")
        _sha256(self.posterior_model_sha256, label="posterior model")
        _require(type(self.candidates) is tuple, "posterior candidates must be an exact tuple")
        _require(
            all(type(candidate) is PosteriorCandidateV2 for candidate in self.candidates),
            "posterior candidates must have exact v2 types",
        )
        keys = tuple(candidate.identity_key for candidate in self.candidates)
        _require(keys == tuple(sorted(set(keys))), "posterior rows must be sorted and unique")
        for candidate in self.candidates:
            candidate.__post_init__()
        _require(self.evidence_class == ENGINEERING_EVIDENCE, "posterior snapshot is not a fixture")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "archive_head_sha256": self.archive_head_sha256,
            "artifact": "evolutionary_kl_fake_terminal_posterior_v2",
            "authority_sha256": self.authority_sha256,
            "candidates": [candidate.document() for candidate in self.candidates],
            "evidence_class": self.evidence_class,
            "posterior_model_sha256": self.posterior_model_sha256,
            "schema_version": 2,
            "scientific_evidence_accepted": False,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/fake-terminal-posterior/v2\0", self.document())


StopReason = Literal["budget_complete", "scientific_deadline", "overflow_exhausted"]


@dataclass(frozen=True, slots=True)
class TerminalDecisionV2:
    """Recomputed terminal decision with an explicit below-100 abstention."""

    authority_sha256: str
    archive_head_sha256: str
    posterior_snapshot_sha256: str
    stop_reason: StopReason
    eligible_candidate_count: int
    archived_call_count: int
    charged_call_count: int
    discarded_charged_call_count: int
    selected_identity_key: str | None
    selected_canonical_sequence_id: str | None
    posterior_mean_utility: float
    abstention_reason: str | None

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="terminal authority")
        _sha256(self.archive_head_sha256, label="terminal archive head")
        _sha256(self.posterior_snapshot_sha256, label="terminal posterior")
        _require(
            self.stop_reason in {"budget_complete", "scientific_deadline", "overflow_exhausted"},
            "terminal stop reason is invalid",
        )
        for name in (
            "eligible_candidate_count",
            "archived_call_count",
            "charged_call_count",
            "discarded_charged_call_count",
        ):
            _nonnegative_integer(getattr(self, name), label=f"terminal {name}")
        _require(
            self.archived_call_count + self.discarded_charged_call_count == self.charged_call_count,
            "terminal charged denominator does not reconcile",
        )
        _require(
            self.eligible_candidate_count <= self.archived_call_count,
            "terminal eligible count exceeds archived calls",
        )
        _require(self.charged_call_count <= TOTAL_UNIQUE_CALLS, "terminal call ceiling exceeded")
        _finite_float(self.posterior_mean_utility, label="terminal posterior-mean utility")
        if self.eligible_candidate_count < MINIMUM_TERMINAL_ELIGIBLE:
            _require(
                self.selected_identity_key is None
                and self.selected_canonical_sequence_id is None
                and self.abstention_reason == "fewer_than_100_eligible_candidates"
                and self.posterior_mean_utility == 0.0,
                "below-threshold terminal decision must explicitly abstain",
            )
        else:
            _sha256(self.selected_identity_key, label="selected identity key")
            _sha256(
                self.selected_canonical_sequence_id,
                label="selected canonical sequence ID",
            )
            _require(self.abstention_reason is None, "selected terminal decision also abstains")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "archive_head_sha256": self.archive_head_sha256,
            "archived_call_count": self.archived_call_count,
            "artifact": "evolutionary_kl_terminal_decision_v2_engineering_fixture",
            "authority_sha256": self.authority_sha256,
            "authorization": {
                "automatic_production_eligible": False,
                "biological_superiority_claim_allowed": False,
                "execution_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "abstention_reason": self.abstention_reason,
            "charged_call_count": self.charged_call_count,
            "discarded_charged_call_count": self.discarded_charged_call_count,
            "eligible_candidate_count": self.eligible_candidate_count,
            "evidence_class": ENGINEERING_EVIDENCE,
            "posterior_mean_utility": self.posterior_mean_utility,
            "posterior_snapshot_sha256": self.posterior_snapshot_sha256,
            "schema_version": 2,
            "selected_canonical_sequence_id": self.selected_canonical_sequence_id,
            "selected_identity_key": self.selected_identity_key,
            "stop_reason": self.stop_reason,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/terminal-decision/v2\0", self.document())


@runtime_checkable
class CampaignOraclePortV2(Protocol):
    """Structural description of the deliberately non-migratable fake seam."""

    authority_sha256: str
    transport_sha256: str
    execution_authorized: Literal[False]
    network_access_enabled: Literal[False]

    def submit(
        self, request_seal: RequestWaveSealV2, request: OracleRequestV2
    ) -> SubmissionReceiptV2: ...

    def poll(self, submission: SubmissionReceiptV2) -> PollReceiptV2: ...


@runtime_checkable
class CampaignTimingPortV2(Protocol):
    """Structural description of the fake cumulative timing issuer."""

    authority_sha256: str
    issuer_sha256: str
    execution_authorized: Literal[False]

    def issue(
        self,
        stream: TimingStream,
        *,
        increment_ns: int,
        wall_increment_ns: int | None = None,
    ) -> TimingReceiptV2: ...


@runtime_checkable
class CampaignPosteriorPortV2(Protocol):
    """Structural description of the fake terminal posterior producer."""

    authority_sha256: str
    posterior_model_sha256: str
    execution_authorized: Literal[False]

    def snapshot(
        self,
        *,
        archive_head_sha256: str,
        candidates: tuple[tuple[str, str], ...],
    ) -> PosteriorSnapshotV2: ...


class EngineeringFakeOraclePortV2:
    """Deterministic in-memory oracle double with no network-capable surface."""

    __slots__ = (
        "_authority_sha256",
        "_pending_polls_before_terminal",
        "_poll_counts",
        "_responses",
        "_submitted",
        "_terminal_emitted",
        "_transport_sha256",
    )

    execution_authorized: Literal[False] = False
    network_access_enabled: Literal[False] = False

    def __init__(
        self,
        authority: CampaignAuthorityV2,
        *,
        responses: tuple[tuple[str, OracleTerminalResultV2], ...],
        pending_polls_before_terminal: int = 0,
    ) -> None:
        _require(type(authority) is CampaignAuthorityV2, "fake oracle requires exact authority")
        authority.__post_init__()
        _require(type(responses) is tuple, "fake oracle responses must be an exact tuple")
        keys: list[str] = []
        response_map: dict[str, OracleTerminalResultV2] = {}
        for row in responses:
            _require(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is OracleTerminalResultV2,
                "fake oracle response row is invalid",
            )
            _sha256(row[0], label="fake oracle response identity")
            row[1].__post_init__()
            keys.append(row[0])
            response_map[row[0]] = row[1]
        _require(tuple(keys) == tuple(sorted(set(keys))), "fake oracle responses are not sorted")
        _nonnegative_integer(pending_polls_before_terminal, label="pending fake poll count")
        self._authority_sha256 = authority.sha256
        self._transport_sha256 = authority.asset_map["oracle_evaluator"].payload_sha256
        self._responses = response_map
        self._pending_polls_before_terminal = pending_polls_before_terminal
        self._submitted: dict[str, SubmissionReceiptV2] = {}
        self._poll_counts: dict[str, int] = {}
        self._terminal_emitted: set[str] = set()

    @property
    def authority_sha256(self) -> str:
        return self._authority_sha256

    @property
    def transport_sha256(self) -> str:
        return self._transport_sha256

    def submit(
        self, request_seal: RequestWaveSealV2, request: OracleRequestV2
    ) -> SubmissionReceiptV2:
        _require(type(request_seal) is RequestWaveSealV2, "fake submit requires request seal")
        _require(type(request) is OracleRequestV2, "fake submit requires exact request")
        request_seal.__post_init__()
        request.__post_init__()
        _require(
            request_seal.authority_sha256 == self.authority_sha256, "fake oracle authority differs"
        )
        _require(request in request_seal.requests, "fake submit request is outside its seal")
        _require(request.request_id not in self._submitted, "fake oracle refuses resubmission")
        external_digest = _hash_document(
            b"amp/evolutionary-kl/fake-external-submission/v2\0",
            {
                "authority_sha256": self.authority_sha256,
                "request_id": request.request_id,
                "request_seal_sha256": request_seal.seal_sha256,
            },
        )
        receipt = SubmissionReceiptV2(
            authority_sha256=self.authority_sha256,
            request_seal_sha256=request_seal.seal_sha256,
            request_id=request.request_id,
            identity_key=request.identity.key,
            external_submission_id=f"fake-{external_digest[:32]}",
            transport_sha256=self.transport_sha256,
        )
        self._submitted[request.request_id] = receipt
        self._poll_counts[request.request_id] = 0
        return receipt

    def poll(self, submission: SubmissionReceiptV2) -> PollReceiptV2:
        _require(type(submission) is SubmissionReceiptV2, "fake poll requires submission receipt")
        submission.__post_init__()
        expected = self._submitted.get(submission.request_id)
        _require(expected == submission, "fake poll does not match the sole recorded submission")
        _require(
            submission.request_id not in self._terminal_emitted,
            "fake oracle refuses polling after a terminal result",
        )
        ordinal = self._poll_counts[submission.request_id]
        if ordinal < self._pending_polls_before_terminal:
            status: PollStatus = "pending"
            result = None
        else:
            result = self._responses.get(submission.identity_key)
            _require(result is not None, "fake oracle lacks a frozen terminal response")
            assert result is not None
            status = result.status
            self._terminal_emitted.add(submission.request_id)
        receipt = PollReceiptV2(
            authority_sha256=self.authority_sha256,
            request_seal_sha256=submission.request_seal_sha256,
            request_id=submission.request_id,
            identity_key=submission.identity_key,
            external_submission_id=submission.external_submission_id,
            transport_sha256=self.transport_sha256,
            poll_ordinal=ordinal,
            status=status,
            terminal_result=result,
        )
        self._poll_counts[submission.request_id] = ordinal + 1
        return receipt


class EngineeringFakeTimingPortV2:
    """Deterministic two-stream cumulative clock double, never a trusted issuer."""

    __slots__ = (
        "_authority_sha256",
        "_cumulative",
        "_cumulative_wall",
        "_heads",
        "_issuer_sha256",
        "_ordinals",
    )

    execution_authorized: Literal[False] = False

    def __init__(self, authority: CampaignAuthorityV2) -> None:
        _require(
            type(authority) is CampaignAuthorityV2, "fake timing port requires exact authority"
        )
        authority.__post_init__()
        self._authority_sha256 = authority.sha256
        self._issuer_sha256 = authority.asset_map["external_timing_issuer"].payload_sha256
        self._cumulative: dict[TimingStream, int] = {"scientific": 0, "sealing": 0}
        self._cumulative_wall: dict[TimingStream, int] = {"scientific": 0, "sealing": 0}
        self._heads: dict[TimingStream, str | None] = {"scientific": None, "sealing": None}
        self._ordinals: dict[TimingStream, int] = {"scientific": 0, "sealing": 0}

    @property
    def authority_sha256(self) -> str:
        return self._authority_sha256

    @property
    def issuer_sha256(self) -> str:
        return self._issuer_sha256

    def issue(
        self,
        stream: TimingStream,
        *,
        increment_ns: int,
        wall_increment_ns: int | None = None,
    ) -> TimingReceiptV2:
        _require(stream in {"scientific", "sealing"}, "fake timing stream is invalid")
        _require(
            type(increment_ns) is int and increment_ns > 0,
            "fake timing increment must be a positive integer",
        )
        if wall_increment_ns is None:
            wall_increment_ns = increment_ns
        _require(
            type(wall_increment_ns) is int and wall_increment_ns > 0,
            "fake wall timing increment must be a positive integer",
        )
        ordinal = self._ordinals[stream]
        receipt = TimingReceiptV2(
            authority_sha256=self.authority_sha256,
            issuer_sha256=self.issuer_sha256,
            stream=stream,
            ordinal=ordinal,
            predecessor_sha256=self._heads[stream],
            cumulative_elapsed_ns=self._cumulative[stream] + increment_ns,
            cumulative_wall_elapsed_ns=(self._cumulative_wall[stream] + wall_increment_ns),
            segment_id=f"fake-{stream}-{ordinal:06d}",
        )
        self._cumulative[stream] = receipt.cumulative_elapsed_ns
        self._cumulative_wall[stream] = receipt.cumulative_wall_elapsed_ns
        self._heads[stream] = receipt.sha256
        self._ordinals[stream] = ordinal + 1
        return receipt


class EngineeringFakePosteriorPortV2:
    """Frozen lookup-table posterior double; it performs no fit or inference."""

    __slots__ = ("_authority_sha256", "_means", "_posterior_model_sha256")

    execution_authorized: Literal[False] = False

    def __init__(
        self,
        authority: CampaignAuthorityV2,
        *,
        means: tuple[tuple[str, float, float], ...],
    ) -> None:
        _require(type(authority) is CampaignAuthorityV2, "fake posterior requires exact authority")
        authority.__post_init__()
        _require(type(means) is tuple, "fake posterior means must be an exact tuple")
        keys: list[str] = []
        lookup: dict[str, tuple[float, float]] = {}
        for row in means:
            _require(
                type(row) is tuple and len(row) == 3 and type(row[0]) is str,
                "fake posterior mean row is invalid",
            )
            key = _sha256(row[0], label="fake posterior identity")
            positive = _finite_float(row[1], label="fake posterior Gram-positive mean")
            negative = _finite_float(row[2], label="fake posterior Gram-negative mean")
            keys.append(key)
            lookup[key] = (positive, negative)
        _require(tuple(keys) == tuple(sorted(set(keys))), "fake posterior means are not sorted")
        self._authority_sha256 = authority.sha256
        self._posterior_model_sha256 = authority.asset_map[
            "calibrated_joint_posterior"
        ].payload_sha256
        self._means = lookup

    @property
    def authority_sha256(self) -> str:
        return self._authority_sha256

    @property
    def posterior_model_sha256(self) -> str:
        return self._posterior_model_sha256

    def snapshot(
        self,
        *,
        archive_head_sha256: str,
        candidates: tuple[tuple[str, str], ...],
    ) -> PosteriorSnapshotV2:
        _sha256(archive_head_sha256, label="fake posterior archive head")
        _require(type(candidates) is tuple, "fake posterior candidates must be an exact tuple")
        rows: list[PosteriorCandidateV2] = []
        for row in candidates:
            _require(
                type(row) is tuple
                and len(row) == 2
                and type(row[0]) is str
                and type(row[1]) is str,
                "fake posterior candidate input is invalid",
            )
            key = _sha256(row[0], label="fake posterior candidate identity")
            sequence_id = _sha256(row[1], label="fake posterior candidate sequence")
            _require(key in self._means, "fake posterior lacks a frozen candidate mean")
            positive, negative = self._means[key]
            rows.append(
                PosteriorCandidateV2(
                    identity_key=key,
                    canonical_sequence_id=sequence_id,
                    gram_positive_mean=positive,
                    gram_negative_mean=negative,
                )
            )
        rows.sort(key=lambda candidate: candidate.identity_key)
        return PosteriorSnapshotV2(
            authority_sha256=self.authority_sha256,
            archive_head_sha256=archive_head_sha256,
            posterior_model_sha256=self.posterior_model_sha256,
            candidates=tuple(rows),
        )


class _ActiveWaveV2:
    __slots__ = ("polls", "request_seal", "submissions", "terminal_request_ids")

    def __init__(self, request_seal: RequestWaveSealV2) -> None:
        self.request_seal = request_seal
        self.submissions: dict[str, SubmissionReceiptV2] = {}
        self.polls: dict[str, list[PollReceiptV2]] = {}
        self.terminal_request_ids: set[str] = set()


@dataclass(frozen=True, slots=True)
class CampaignControllerSnapshotV2:
    """Small immutable accounting view with no authority to resume or execute."""

    authority_sha256: str
    state: Literal["initial", "adaptive", "stopped", "terminal"]
    stop_reason: StopReason | None
    active_request_seal_sha256: str | None
    complete_archive_count: int
    adaptive_archive_count: int
    charged_call_count: int
    archived_call_count: int
    unarchived_charged_call_count: int
    scientific_elapsed_ns: int
    scientific_wall_elapsed_ns: int
    sealing_elapsed_ns: int
    sealing_wall_elapsed_ns: int
    overflow_cursor: int
    terminal_decision_sha256: str | None

    def __post_init__(self) -> None:
        _sha256(self.authority_sha256, label="controller snapshot authority")
        _require(self.state in {"initial", "adaptive", "stopped", "terminal"}, "state invalid")
        if self.stop_reason is not None:
            _require(
                self.stop_reason
                in {"budget_complete", "scientific_deadline", "overflow_exhausted"},
                "snapshot stop reason is invalid",
            )
        if self.active_request_seal_sha256 is not None:
            _sha256(self.active_request_seal_sha256, label="active request seal")
        for name in (
            "complete_archive_count",
            "adaptive_archive_count",
            "charged_call_count",
            "archived_call_count",
            "unarchived_charged_call_count",
            "scientific_elapsed_ns",
            "scientific_wall_elapsed_ns",
            "sealing_elapsed_ns",
            "sealing_wall_elapsed_ns",
            "overflow_cursor",
        ):
            _nonnegative_integer(getattr(self, name), label=f"controller snapshot {name}")
        _require(
            self.archived_call_count + self.unarchived_charged_call_count
            == self.charged_call_count,
            "controller snapshot charged denominator does not reconcile",
        )
        if self.terminal_decision_sha256 is not None:
            _sha256(self.terminal_decision_sha256, label="terminal decision")

    def document(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "active_request_seal_sha256": self.active_request_seal_sha256,
            "adaptive_archive_count": self.adaptive_archive_count,
            "archived_call_count": self.archived_call_count,
            "artifact": "evolutionary_kl_campaign_controller_snapshot_v2",
            "authority_sha256": self.authority_sha256,
            "authorization": {
                "execution_authorized": False,
                "network_oracle_authorized": False,
                "scientific_evidence_accepted": False,
            },
            "charged_call_count": self.charged_call_count,
            "complete_archive_count": self.complete_archive_count,
            "unarchived_charged_call_count": self.unarchived_charged_call_count,
            "evidence_class": ENGINEERING_EVIDENCE,
            "overflow_cursor": self.overflow_cursor,
            "schema_version": 2,
            "scientific_elapsed_ns": self.scientific_elapsed_ns,
            "scientific_wall_elapsed_ns": self.scientific_wall_elapsed_ns,
            "sealing_elapsed_ns": self.sealing_elapsed_ns,
            "sealing_wall_elapsed_ns": self.sealing_wall_elapsed_ns,
            "state": self.state,
            "stop_reason": self.stop_reason,
            "terminal_decision_sha256": self.terminal_decision_sha256,
        }

    @property
    def sha256(self) -> str:
        return _hash_document(b"amp/evolutionary-kl/controller-snapshot/v2\0", self.document())


class CampaignControllerV2:
    """Pure engineering transition validator driven only by exact fake ports."""

    __slots__ = (
        "_active",
        "_adaptive_archive_count",
        "_archived_request_ids",
        "_archives",
        "_asset_payloads",
        "_authority",
        "_authority_sha256",
        "_charged_identity_keys",
        "_discarded",
        "_external_submission_ids",
        "_overflow_cursor",
        "_scientific_elapsed_ns",
        "_scientific_timing_head",
        "_scientific_timing_ordinal",
        "_scientific_wall_elapsed_ns",
        "_sealing_elapsed_ns",
        "_sealing_timing_head",
        "_sealing_timing_ordinal",
        "_sealing_wall_elapsed_ns",
        "_state",
        "_stop_reason",
        "_terminal_decision",
    )

    execution_authorized: Literal[False] = False
    network_oracle_authorized: Literal[False] = False
    scientific_evidence_accepted: Literal[False] = False
    automatic_production_eligible: Literal[False] = False

    def __init__(self, authority: CampaignAuthorityV2) -> None:
        _require(type(authority) is CampaignAuthorityV2, "controller requires exact v2 authority")
        authority.__post_init__()
        self._authority = authority
        self._authority_sha256 = authority.sha256
        self._asset_payloads = {
            role: asset.payload_sha256 for role, asset in authority.asset_map.items()
        }
        self._state: Literal["initial", "adaptive", "stopped", "terminal"] = "initial"
        self._stop_reason: StopReason | None = None
        self._active: _ActiveWaveV2 | None = None
        self._archives: list[ArchivedWaveV2] = []
        self._discarded: list[DiscardedWaveV2] = []
        self._adaptive_archive_count = 0
        self._archived_request_ids: set[str] = set()
        self._charged_identity_keys: set[str] = set()
        self._external_submission_ids: set[str] = set()
        self._overflow_cursor = 0
        self._scientific_elapsed_ns = 0
        self._scientific_wall_elapsed_ns = 0
        self._scientific_timing_head: str | None = None
        self._scientific_timing_ordinal = 0
        self._sealing_elapsed_ns = 0
        self._sealing_wall_elapsed_ns = 0
        self._sealing_timing_head: str | None = None
        self._sealing_timing_ordinal = 0
        self._terminal_decision: TerminalDecisionV2 | None = None

    @property
    def authority(self) -> CampaignAuthorityV2:
        return self._authority

    @property
    def active_request_seal(self) -> RequestWaveSealV2 | None:
        return None if self._active is None else self._active.request_seal

    @property
    def archives(self) -> tuple[ArchivedWaveV2, ...]:
        return tuple(self._archives)

    @property
    def discarded_waves(self) -> tuple[DiscardedWaveV2, ...]:
        return tuple(self._discarded)

    @property
    def terminal_decision(self) -> TerminalDecisionV2 | None:
        return self._terminal_decision

    @property
    def archive_head_sha256(self) -> str | None:
        return None if not self._archives else self._archives[-1].archive_sha256

    def snapshot(self) -> CampaignControllerSnapshotV2:
        archived_count = sum(len(archive.request_seal.requests) for archive in self._archives)
        discarded_charged = len(self._charged_identity_keys) - archived_count
        _require(discarded_charged >= 0, "archived calls exceed charged calls")
        return CampaignControllerSnapshotV2(
            authority_sha256=self._authority_sha256,
            state=self._state,
            stop_reason=self._stop_reason,
            active_request_seal_sha256=(
                None if self._active is None else self._active.request_seal.seal_sha256
            ),
            complete_archive_count=len(self._archives),
            adaptive_archive_count=self._adaptive_archive_count,
            charged_call_count=len(self._charged_identity_keys),
            archived_call_count=archived_count,
            unarchived_charged_call_count=discarded_charged,
            scientific_elapsed_ns=self._scientific_elapsed_ns,
            scientific_wall_elapsed_ns=self._scientific_wall_elapsed_ns,
            sealing_elapsed_ns=self._sealing_elapsed_ns,
            sealing_wall_elapsed_ns=self._sealing_wall_elapsed_ns,
            overflow_cursor=self._overflow_cursor,
            terminal_decision_sha256=(
                None if self._terminal_decision is None else self._terminal_decision.sha256
            ),
        )

    def _expected_previous_archive(self) -> str | None:
        return self.archive_head_sha256

    def seal_initial_requests(self) -> RequestWaveSealV2:
        _require(self._state == "initial", "initial requests can only be sealed in initial state")
        _require(self._active is None and not self._archives, "initial request seal already exists")
        requests = tuple(
            OracleRequestV2(
                authority_sha256=self._authority_sha256,
                phase="initial",
                wave_index=-1,
                seat_position=position,
                source="initial",
                identity=identity,
            )
            for position, identity in enumerate(self.authority.initial_queries)
        )
        request_seal = RequestWaveSealV2(
            authority_sha256=self._authority_sha256,
            phase="initial",
            wave_index=-1,
            previous_archive_sha256=None,
            requests=requests,
        )
        self._active = _ActiveWaveV2(request_seal)
        return request_seal

    def _validate_timing_receipt(
        self,
        receipt: TimingReceiptV2,
        *,
        stream: TimingStream,
        deadline_transition: bool = False,
    ) -> None:
        _require(type(receipt) is TimingReceiptV2, "timing receipt must have its exact v2 type")
        receipt.__post_init__()
        _require(receipt.authority_sha256 == self._authority_sha256, "timing authority differs")
        expected_issuer = self._asset_payloads["external_timing_issuer"]
        _require(receipt.issuer_sha256 == expected_issuer, "timing issuer is not authority-bound")
        _require(receipt.stream == stream, "timing receipt stream differs")
        if stream == "scientific":
            ordinal = self._scientific_timing_ordinal
            predecessor = self._scientific_timing_head
            elapsed = self._scientific_elapsed_ns
            wall_elapsed = self._scientific_wall_elapsed_ns
            limit = SCIENTIFIC_DEADLINE_NS
        else:
            ordinal = self._sealing_timing_ordinal
            predecessor = self._sealing_timing_head
            elapsed = self._sealing_elapsed_ns
            wall_elapsed = self._sealing_wall_elapsed_ns
            limit = SEALING_ALLOWANCE_NS
        _require(receipt.ordinal == ordinal, "timing receipt ordinal is not contiguous")
        _require(receipt.predecessor_sha256 == predecessor, "timing predecessor differs")
        _require(receipt.cumulative_elapsed_ns > elapsed, "timing receipt did not advance")
        _require(
            receipt.cumulative_wall_elapsed_ns > wall_elapsed,
            "wall timing receipt did not advance",
        )
        _require(
            receipt.cumulative_elapsed_ns <= limit and receipt.cumulative_wall_elapsed_ns <= limit,
            "timing allowance exceeded",
        )
        if stream == "scientific":
            if deadline_transition:
                _require(
                    max(
                        receipt.cumulative_elapsed_ns,
                        receipt.cumulative_wall_elapsed_ns,
                    )
                    == SCIENTIFIC_DEADLINE_NS,
                    "deadline transition requires the exact scientific ceiling",
                )
            else:
                _require(
                    receipt.cumulative_elapsed_ns < SCIENTIFIC_DEADLINE_NS
                    and receipt.cumulative_wall_elapsed_ns < SCIENTIFIC_DEADLINE_NS,
                    "scientific action cannot begin at the deadline",
                )

    def _commit_timing_receipt(self, receipt: TimingReceiptV2) -> None:
        if receipt.stream == "scientific":
            self._scientific_elapsed_ns = receipt.cumulative_elapsed_ns
            self._scientific_wall_elapsed_ns = receipt.cumulative_wall_elapsed_ns
            self._scientific_timing_head = receipt.sha256
            self._scientific_timing_ordinal += 1
        else:
            self._sealing_elapsed_ns = receipt.cumulative_elapsed_ns
            self._sealing_wall_elapsed_ns = receipt.cumulative_wall_elapsed_ns
            self._sealing_timing_head = receipt.sha256
            self._sealing_timing_ordinal += 1

    def seal_adaptive_requests(
        self,
        method_candidates: tuple[OracleQueryIdentity, ...],
        *,
        timing_receipt: TimingReceiptV2,
    ) -> RequestWaveSealV2 | None:
        """Freeze one 14+2 request wave or stop on frozen-overflow exhaustion."""

        _require(self._state == "adaptive", "adaptive requests require adaptive state")
        _require(self._active is None, "a request wave is already active")
        _require(
            self._adaptive_archive_count < ADAPTIVE_WAVES,
            "all 28 adaptive waves are already archived",
        )
        _require(type(method_candidates) is tuple, "method candidates must be an exact tuple")
        _require(
            len(method_candidates) <= METHOD_PROPOSAL_ATTEMPT_CAP,
            "method proposal attempt cap exceeded",
        )
        scheduled_reserve_keys = {
            identity.key for pair in self.authority.common_reserve_by_wave for identity in pair
        }
        for identity in method_candidates:
            self.authority.validate_query_identity(identity)
            _require(
                identity.key not in scheduled_reserve_keys,
                "method candidate collides with the pre-frozen common reserve",
            )

        selected: list[tuple[OracleQueryIdentity, Literal["method", "overflow"]]] = []
        selected_keys: set[str] = set()
        for identity in method_candidates:
            if identity.key in self._charged_identity_keys or identity.key in selected_keys:
                continue
            selected.append((identity, "method"))
            selected_keys.add(identity.key)
            if len(selected) == METHOD_SEATS_PER_WAVE:
                break

        overflow_cursor = self._overflow_cursor
        while len(selected) < METHOD_SEATS_PER_WAVE and overflow_cursor < len(
            self.authority.overflow_queries
        ):
            identity = self.authority.overflow_queries[overflow_cursor]
            overflow_cursor += 1
            if identity.key in self._charged_identity_keys or identity.key in selected_keys:
                continue
            selected.append((identity, "overflow"))
            selected_keys.add(identity.key)

        self._validate_timing_receipt(timing_receipt, stream="scientific")
        if len(selected) != METHOD_SEATS_PER_WAVE:
            self._commit_timing_receipt(timing_receipt)
            self._overflow_cursor = overflow_cursor
            self._state = "stopped"
            self._stop_reason = "overflow_exhausted"
            return None

        wave_index = self._adaptive_archive_count
        common_reserve = self.authority.common_reserve_by_wave[wave_index]
        _require(
            all(identity.key not in self._charged_identity_keys for identity in common_reserve),
            "common reserve identity was already submitted in this run",
        )
        _require(
            all(identity.key not in selected_keys for identity in common_reserve),
            "method seats collide with this wave's common reserve",
        )
        requests: list[OracleRequestV2] = []
        for position, (identity, source) in enumerate(selected):
            requests.append(
                OracleRequestV2(
                    authority_sha256=self._authority_sha256,
                    phase="adaptive",
                    wave_index=wave_index,
                    seat_position=position,
                    source=source,
                    identity=identity,
                )
            )
        for reserve_position, identity in enumerate(common_reserve, start=METHOD_SEATS_PER_WAVE):
            requests.append(
                OracleRequestV2(
                    authority_sha256=self._authority_sha256,
                    phase="adaptive",
                    wave_index=wave_index,
                    seat_position=reserve_position,
                    source="common_reserve",
                    identity=identity,
                )
            )
        request_seal = RequestWaveSealV2(
            authority_sha256=self._authority_sha256,
            phase="adaptive",
            wave_index=wave_index,
            previous_archive_sha256=self._expected_previous_archive(),
            requests=tuple(requests),
        )
        self._commit_timing_receipt(timing_receipt)
        self._overflow_cursor = overflow_cursor
        self._active = _ActiveWaveV2(request_seal)
        return request_seal

    def _active_request(self, request_id: str) -> OracleRequestV2:
        _sha256(request_id, label="active request ID")
        _require(self._active is not None, "no request wave is active")
        assert self._active is not None
        matches = tuple(
            request
            for request in self._active.request_seal.requests
            if request.request_id == request_id
        )
        _require(len(matches) == 1, "request is not in the active request seal")
        return matches[0]

    def _validate_action_timing(self, timing_receipt: TimingReceiptV2 | None) -> None:
        assert self._active is not None
        if self._active.request_seal.phase == "initial":
            _require(timing_receipt is None, "initial calls occur before the scientific clock")
        else:
            _require(
                type(timing_receipt) is TimingReceiptV2,
                "adaptive transport action requires an exact scientific timing receipt",
            )
            assert timing_receipt is not None
            self._validate_timing_receipt(timing_receipt, stream="scientific")

    def _commit_action_timing(self, timing_receipt: TimingReceiptV2 | None) -> None:
        if timing_receipt is not None:
            self._commit_timing_receipt(timing_receipt)

    def submit_fake(
        self,
        port: EngineeringFakeOraclePortV2,
        request_id: str,
        *,
        timing_receipt: TimingReceiptV2 | None = None,
    ) -> SubmissionReceiptV2:
        """Submit through the one exact in-memory fake type; no arbitrary port is accepted."""

        _require(
            type(port) is EngineeringFakeOraclePortV2,
            "controller only accepts the exact non-network fake oracle port",
        )
        _require(
            port.execution_authorized is False and port.network_access_enabled is False,
            "fake oracle authority flags changed",
        )
        _require(port.authority_sha256 == self._authority_sha256, "fake oracle authority differs")
        expected_transport = self._asset_payloads["oracle_evaluator"]
        _require(port.transport_sha256 == expected_transport, "fake oracle transport differs")
        request = self._active_request(request_id)
        assert self._active is not None
        _require(request_id not in self._active.submissions, "request has already been submitted")
        next_position = len(self._active.submissions)
        _require(
            next_position < len(self._active.request_seal.requests),
            "all sealed requests are already submitted",
        )
        expected_request = self._active.request_seal.requests[next_position]
        _require(
            request_id == expected_request.request_id,
            "submissions must follow sealed request seat order",
        )
        _require(
            request.identity.key not in self._charged_identity_keys,
            "logical oracle identity has already been charged",
        )
        _require(
            len(self._charged_identity_keys) < TOTAL_UNIQUE_CALLS,
            "unique logical-call ceiling exceeded",
        )
        self._validate_action_timing(timing_receipt)
        receipt = port.submit(self._active.request_seal, request)
        _require(type(receipt) is SubmissionReceiptV2, "fake oracle returned a non-v2 submission")
        receipt.__post_init__()
        _require(receipt.authority_sha256 == self._authority_sha256, "submission authority differs")
        _require(
            receipt.request_seal_sha256 == self._active.request_seal.seal_sha256,
            "submission occurred before or outside the active request seal",
        )
        _require(receipt.request_id == request.request_id, "submission request ID differs")
        _require(receipt.identity_key == request.identity.key, "submission identity differs")
        _require(receipt.transport_sha256 == expected_transport, "submission transport differs")
        _require(
            receipt.external_submission_id not in self._external_submission_ids,
            "external submission ID is not unique",
        )
        self._commit_action_timing(timing_receipt)
        self._active.submissions[request_id] = receipt
        self._active.polls[request_id] = []
        self._charged_identity_keys.add(request.identity.key)
        self._external_submission_ids.add(receipt.external_submission_id)
        return receipt

    def poll_fake(
        self,
        port: EngineeringFakeOraclePortV2,
        request_id: str,
        *,
        timing_receipt: TimingReceiptV2 | None = None,
    ) -> PollReceiptV2:
        """Poll the sole recorded external ID; there is intentionally no retry-submit API."""

        _require(
            type(port) is EngineeringFakeOraclePortV2,
            "controller only accepts the exact non-network fake oracle port",
        )
        _require(
            port.execution_authorized is False and port.network_access_enabled is False,
            "fake oracle authority flags changed",
        )
        _require(port.authority_sha256 == self._authority_sha256, "fake oracle authority differs")
        expected_transport = self._asset_payloads["oracle_evaluator"]
        _require(port.transport_sha256 == expected_transport, "fake oracle transport differs")
        request = self._active_request(request_id)
        assert self._active is not None
        submission = self._active.submissions.get(request_id)
        _require(submission is not None, "request must be submitted once before polling")
        assert submission is not None
        _require(request_id not in self._active.terminal_request_ids, "request is already terminal")
        ordered_submitted = tuple(
            sealed.request_id
            for sealed in self._active.request_seal.requests
            if sealed.request_id in self._active.submissions
        )
        next_terminal_request = next(
            submitted
            for submitted in ordered_submitted
            if submitted not in self._active.terminal_request_ids
        )
        _require(
            request_id == next_terminal_request,
            "polls must finish charged requests in sealed seat order",
        )
        self._validate_action_timing(timing_receipt)
        receipt = port.poll(submission)
        _require(type(receipt) is PollReceiptV2, "fake oracle returned a non-v2 poll")
        receipt.__post_init__()
        _require(receipt.authority_sha256 == self._authority_sha256, "poll authority differs")
        _require(
            receipt.request_seal_sha256 == self._active.request_seal.seal_sha256,
            "poll request seal differs",
        )
        _require(receipt.request_id == request_id, "poll request ID differs")
        _require(receipt.identity_key == request.identity.key, "poll identity differs")
        _require(
            receipt.transport_sha256 == expected_transport
            and receipt.transport_sha256 == submission.transport_sha256,
            "poll transport differs from its submission or authority-bound transport",
        )
        _require(
            receipt.external_submission_id == submission.external_submission_id,
            "poll changed the sole external submission ID",
        )
        expected_ordinal = len(self._active.polls[request_id])
        _require(receipt.poll_ordinal == expected_ordinal, "poll ordinal is not contiguous")
        self._commit_action_timing(timing_receipt)
        self._active.polls[request_id].append(receipt)
        if receipt.is_terminal:
            self._active.terminal_request_ids.add(request_id)
        return receipt

    def _ordered_active_receipts(
        self,
    ) -> tuple[tuple[SubmissionReceiptV2, ...], tuple[PollReceiptV2, ...]]:
        _require(self._active is not None, "no request wave is active")
        assert self._active is not None
        submissions: list[SubmissionReceiptV2] = []
        polls: list[PollReceiptV2] = []
        for request in self._active.request_seal.requests:
            submission = self._active.submissions.get(request.request_id)
            if submission is not None:
                submissions.append(submission)
                polls.extend(self._active.polls[request.request_id])
        return tuple(submissions), tuple(polls)

    def archive_completed_wave(self, *, sealing_receipt: TimingReceiptV2) -> ArchivedWaveV2:
        """Archive only a complete 64-call initial or complete 16-call adaptive wave."""

        _require(self._active is not None, "no request wave is active")
        assert self._active is not None
        request_ids = tuple(request.request_id for request in self._active.request_seal.requests)
        _require(
            set(self._active.submissions) == set(request_ids),
            "complete archive requires every sealed request to be submitted",
        )
        _require(
            self._active.terminal_request_ids == set(request_ids),
            "complete archive requires one terminal result for every request",
        )
        self._validate_timing_receipt(sealing_receipt, stream="sealing")
        submissions, polls = self._ordered_active_receipts()
        archive = ArchivedWaveV2(
            authority_sha256=self._authority_sha256,
            request_seal=self._active.request_seal,
            previous_archive_sha256=self._expected_previous_archive(),
            submissions=submissions,
            polls=polls,
            scientific_elapsed_ns=self._scientific_elapsed_ns,
            scientific_wall_elapsed_ns=self._scientific_wall_elapsed_ns,
            scientific_timing_head_sha256=self._scientific_timing_head,
            sealing_elapsed_ns=sealing_receipt.cumulative_elapsed_ns,
            sealing_wall_elapsed_ns=sealing_receipt.cumulative_wall_elapsed_ns,
            sealing_timing_receipt_sha256=sealing_receipt.sha256,
        )
        self._commit_timing_receipt(sealing_receipt)
        self._archives.append(archive)
        self._archived_request_ids.update(request_ids)
        phase = self._active.request_seal.phase
        self._active = None
        if phase == "initial":
            _require(
                len(self._charged_identity_keys) == INITIAL_CALLS, "initial call count differs"
            )
            self._state = "adaptive"
        else:
            self._adaptive_archive_count += 1
            if self._adaptive_archive_count == ADAPTIVE_WAVES:
                _require(
                    len(self._charged_identity_keys) == TOTAL_UNIQUE_CALLS,
                    "28 complete waves must total exactly 512 unique calls",
                )
                self._state = "stopped"
                self._stop_reason = "budget_complete"
        return archive

    def reach_scientific_deadline(self, *, timing_receipt: TimingReceiptV2) -> None:
        """Stop at exactly 7200 s and retain, but do not evidence, the active partial wave."""

        _require(self._state == "adaptive", "scientific deadline requires adaptive state")
        self._validate_timing_receipt(
            timing_receipt,
            stream="scientific",
            deadline_transition=True,
        )
        discarded: DiscardedWaveV2 | None = None
        if self._active is not None:
            submissions, polls = self._ordered_active_receipts()
            discarded = DiscardedWaveV2(
                authority_sha256=self._authority_sha256,
                request_seal=self._active.request_seal,
                submissions=submissions,
                polls=polls,
                reason="scientific_deadline",
                scientific_elapsed_ns=timing_receipt.cumulative_elapsed_ns,
                scientific_wall_elapsed_ns=timing_receipt.cumulative_wall_elapsed_ns,
                scientific_timing_receipt_sha256=timing_receipt.sha256,
            )
        self._commit_timing_receipt(timing_receipt)
        if discarded is not None:
            self._discarded.append(discarded)
        self._active = None
        self._state = "stopped"
        self._stop_reason = "scientific_deadline"

    def _archived_successes(
        self,
    ) -> dict[str, tuple[OracleRequestV2, OracleTerminalResultV2]]:
        successes: dict[str, tuple[OracleRequestV2, OracleTerminalResultV2]] = {}
        for archive in self._archives:
            request_by_id = {
                request.request_id: request for request in archive.request_seal.requests
            }
            for poll in archive.polls:
                if not poll.is_terminal or poll.status != "succeeded":
                    continue
                result = poll.terminal_result
                assert result is not None
                request = request_by_id[poll.request_id]
                _require(request.hard_valid is True, "archived success is not hard-valid")
                _require(
                    request.identity.key not in successes,
                    "successful logical identity appears more than once",
                )
                successes[request.identity.key] = (request, result)
        return successes

    def _validate_terminal_snapshot(
        self,
        snapshot: PosteriorSnapshotV2,
        successes: dict[str, tuple[OracleRequestV2, OracleTerminalResultV2]],
    ) -> None:
        _require(type(snapshot) is PosteriorSnapshotV2, "terminal posterior type differs")
        snapshot.__post_init__()
        _require(snapshot.authority_sha256 == self._authority_sha256, "posterior authority differs")
        _require(
            snapshot.archive_head_sha256 == self.archive_head_sha256,
            "posterior is not bound to the final complete archive",
        )
        expected_model = self._asset_payloads["calibrated_joint_posterior"]
        _require(snapshot.posterior_model_sha256 == expected_model, "posterior model differs")
        expected_keys = tuple(sorted(successes))
        observed_keys = tuple(candidate.identity_key for candidate in snapshot.candidates)
        _require(
            observed_keys == expected_keys,
            "posterior must contain every and only archived successful query identity",
        )
        for candidate in snapshot.candidates:
            request, _result = successes[candidate.identity_key]
            _require(
                candidate.canonical_sequence_id == request.identity.canonical_sequence_id,
                "posterior candidate sequence differs from archived request",
            )

    def finalize_with_fake_posterior(
        self, port: EngineeringFakePosteriorPortV2
    ) -> TerminalDecisionV2:
        """Compute the only terminal rule from a complete exact fake posterior table."""

        _require(self._state == "stopped", "terminal decision requires a stopped controller")
        _require(self._active is None, "terminal decision cannot include an active wave")
        _require(self._terminal_decision is None, "terminal decision already exists")
        _require(bool(self._archives), "terminal decision requires the archived initial design")
        _require(
            type(port) is EngineeringFakePosteriorPortV2,
            "controller only accepts the exact fake posterior port",
        )
        _require(port.execution_authorized is False, "fake posterior authority flag changed")
        _require(
            port.authority_sha256 == self._authority_sha256, "fake posterior authority differs"
        )
        expected_model = self._asset_payloads["calibrated_joint_posterior"]
        _require(port.posterior_model_sha256 == expected_model, "fake posterior model differs")
        successes = self._archived_successes()
        candidate_inputs = tuple(
            sorted(
                (
                    identity_key,
                    request.identity.canonical_sequence_id,
                )
                for identity_key, (request, _result) in successes.items()
            )
        )
        archive_head = self.archive_head_sha256
        assert archive_head is not None
        snapshot = port.snapshot(
            archive_head_sha256=archive_head,
            candidates=candidate_inputs,
        )
        self._validate_terminal_snapshot(snapshot, successes)
        row_by_key = {candidate.identity_key: candidate for candidate in snapshot.candidates}
        eligible_rows = tuple(
            row_by_key[identity_key]
            for identity_key, (_request, result) in successes.items()
            if result.eligible
        )
        if len(eligible_rows) < MINIMUM_TERMINAL_ELIGIBLE:
            selected_key = None
            selected_sequence = None
            utility = 0.0
            abstention_reason = "fewer_than_100_eligible_candidates"
        else:
            selected = min(
                eligible_rows,
                key=lambda candidate: (
                    -candidate.utility,
                    candidate.canonical_sequence_id,
                    candidate.identity_key,
                ),
            )
            selected_key = selected.identity_key
            selected_sequence = selected.canonical_sequence_id
            utility = selected.utility
            abstention_reason = None
        stop_reason = self._stop_reason
        _require(stop_reason is not None, "stopped controller lacks an exact stop reason")
        assert stop_reason is not None
        terminal = TerminalDecisionV2(
            authority_sha256=self._authority_sha256,
            archive_head_sha256=archive_head,
            posterior_snapshot_sha256=snapshot.sha256,
            stop_reason=stop_reason,
            eligible_candidate_count=len(eligible_rows),
            archived_call_count=len(self._archived_request_ids),
            charged_call_count=len(self._charged_identity_keys),
            discarded_charged_call_count=(
                len(self._charged_identity_keys) - len(self._archived_request_ids)
            ),
            selected_identity_key=selected_key,
            selected_canonical_sequence_id=selected_sequence,
            posterior_mean_utility=utility,
            abstention_reason=abstention_reason,
        )
        self._terminal_decision = terminal
        self._state = "terminal"
        return terminal

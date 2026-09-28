"""Fail-closed engineering adapter for the categorical-diffusion post-hoc arm.

Nothing in this module can construct or run a sampler.  The frozen v1 policy
inventory is empty.  The useful part of the module is deliberately pure: it
records a fixed sampling grid, resolves a pool without labels, ranks that pool
once from a structurally bound but externally unauthenticated initial-64
posterior snapshot claim, and freezes the 392 method seats used by the common
14+2 controller.

Organizer-reference data is intentionally absent from every API below.  It can
only be attached later by :func:`build_unauthenticated_compliance_claim`, whose
output is a separate, non-authoritative artifact and cannot be fed back into
this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from amp_challenge.sequences import canonical_sequence_id, canonicalize_sequence

FROZEN_ADAPTER_CONFIG_SHA256 = "d230c740ff5c3636a790eea2262bb09129c7197119c6c8a6f165366d68c226e4"
ADAPTER_ARTIFACT = "categorical_diffusion_posthoc_engineering_adapter_v1"
GRID_SLOTS = 2_048
ATTEMPTS_PER_SLOT = 8
ATTEMPT_COUNT = GRID_SLOTS * ATTEMPTS_PER_SLOT
METHOD_SEATS = 392
POLICY_MEMBERS = 10
WAVES = 28
METHOD_SEATS_PER_WAVE = 14
RESERVE_SEATS_PER_WAVE = 2
INITIAL_RESPONSE_COUNT = 64
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_RAW_OUTPUT_CHARACTERS = 256
MAX_COMPLIANCE_REASON_CHARACTERS = 256
MAX_COMPLIANCE_CLAIM_BYTES = 2 * 1024 * 1024
MAX_ADAPTER_CONFIG_BYTES = 64 * 1024
MAX_POLICY_IDENTITY_CHARACTERS = 256
MAX_SIGNED_63 = (1 << 63) - 1
MAX_PATH_CHARACTERS = 4_096
MAX_RAW_OUTPUT_UTF8_BYTES = 256
MAX_ATTEMPT_JSONL_LINE_BYTES = 2 * 1024
MAX_TERMINAL_STATUS_CHARACTERS = 32
MAX_CONTROLLER_SOURCE_CHARACTERS = 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_POLICY_TOKENS = (
    "fixture",
    "no_go",
    "no-go",
    "incident",
    "invalid",
    "pilot",
    "runtime_hash",
)


class PosthocContractError(ValueError):
    """Raised when immutable engineering data violates the adapter contract."""


class PosthocAdapterBlockedError(RuntimeError):
    """Raised by every public attempt to construct the absent real sampler."""


@dataclass(frozen=True, slots=True)
class AdapterContract:
    """The relevant fail-closed fields of the frozen adapter declaration."""

    config_sha256: str
    accepted_policy_sets: tuple[str, ...]
    execution_authorized: bool
    oracle_calls_authorized: bool
    scientific_evidence_accepted: bool
    production_input_eligible: bool
    automatic_generator_mixture_eligible: bool

    def __post_init__(self) -> None:
        if (
            type(self.config_sha256) is not str
            or len(self.config_sha256) != 64
            or _SHA256_RE.fullmatch(self.config_sha256) is None
            or self.config_sha256 != FROZEN_ADAPTER_CONFIG_SHA256
            or type(self.accepted_policy_sets) is not tuple
            or len(self.accepted_policy_sets) != 0
        ):
            raise PosthocContractError("adapter contract identity/inventory invariant violated")
        for value in (
            self.execution_authorized,
            self.oracle_calls_authorized,
            self.scientific_evidence_accepted,
            self.production_input_eligible,
            self.automatic_generator_mixture_eligible,
        ):
            if value is not False:
                raise PosthocContractError("adapter contract authority flags must remain false")


@dataclass(frozen=True, slots=True)
class SamplerPreflight:
    """A public, path-free record explaining why sampler construction stops."""

    status: str
    blockers: tuple[str, ...]
    accepted_policy_count: int
    asset_paths_resolved: bool = False
    execution_authorized: bool = False
    oracle_calls_authorized: bool = False
    scientific_evidence_accepted: bool = False
    production_input_eligible: bool = False
    external_inputs_authenticated: bool = False

    def __post_init__(self) -> None:
        expected_blockers = (
            "no_independently_accepted_ten_policy_set",
            "no_accepted_native_sampler_trajectory_receipt",
            "no_accepted_common_initial_64_posterior_snapshot",
        )
        if (
            type(self.status) is not str
            or self.status != "blocked_no_accepted_policy_or_campaign_assets"
            or type(self.blockers) is not tuple
            or len(self.blockers) != len(expected_blockers)
        ):
            raise PosthocContractError("sampler preflight identity/blocker invariant violated")
        for blocker, expected in zip(self.blockers, expected_blockers, strict=True):
            if type(blocker) is not str or blocker != expected:
                raise PosthocContractError("sampler preflight blocker invariant violated")
        if type(self.accepted_policy_count) is not int or self.accepted_policy_count != 0:
            raise PosthocContractError("sampler preflight accepted policy count must remain zero")
        for value in (
            self.asset_paths_resolved,
            self.execution_authorized,
            self.oracle_calls_authorized,
            self.scientific_evidence_accepted,
            self.production_input_eligible,
            self.external_inputs_authenticated,
        ):
            if value is not False:
                raise PosthocContractError("sampler preflight authority flags must remain false")


@dataclass(frozen=True, slots=True, init=False)
class AcceptedNativeSamplerOutput:
    """Opaque future protocol; v1 cannot construct it or invent trajectories."""

    policy_set_sha256: str
    policy_member_index: int
    sequence: str
    sample_seed: int
    sample_counter: int
    trajectory_receipt_sha256: str

    def __init__(self, *_: object, **__: object) -> None:
        raise PosthocAdapterBlockedError(
            "v1 has no accepted native sampler; a new contract with an authenticated "
            "trajectory receipt is required"
        )


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    slot: int
    attempt: int
    source_ordinal: int
    request_sha256: str
    policy_member_index: int
    policy_member_sha256: str
    sample_seed: int
    sample_counter: int
    trajectory_receipt_sha256: str
    raw_sequence: str
    execution_authorized: bool = False
    oracle_calls_authorized: bool = False
    scientific_evidence_accepted: bool = False
    production_input_eligible: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedSlot:
    slot: int
    selected_source_ordinal: int | None
    sequence: str | None
    sequence_id: str | None
    rejection: str | None


@dataclass(frozen=True, slots=True)
class SealedPool:
    request_sha256: str
    initial64_snapshot_sha256: str
    attempts: tuple[CandidateAttempt, ...]
    slots: tuple[ResolvedSlot, ...]
    exact_training_overlap_ids: tuple[str, ...]
    accepted_training_homology_ids: tuple[str, ...]
    seal_sha256: str

    @property
    def candidates(self) -> tuple[ResolvedSlot, ...]:
        return tuple(row for row in self.slots if row.sequence_id is not None)


@dataclass(frozen=True, slots=True)
class Initial64PosteriorSnapshot:
    """A structural snapshot claim; v1 does not authenticate its external digest."""

    snapshot_sha256: str
    initial_archive_sha256: str
    posterior_model_sha256: str
    response_count: int
    adaptive_response_count: int
    joint_chance_threshold: float
    covariance_dimension: int
    covariance_rank: int
    affine_support_validated: bool
    execution_authorized: bool = False
    oracle_calls_authorized: bool = False
    scientific_evidence_accepted: bool = False
    production_input_eligible: bool = False


@dataclass(frozen=True, slots=True)
class CandidateScore:
    sequence_id: str
    source_ordinal: int
    snapshot_sha256: str
    gram_positive_mean: float
    gram_negative_mean: float
    joint_chance_probability: float


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    rank: int
    sequence_id: str
    source_ordinal: int
    chance_feasible: bool
    mean_utility: float
    joint_chance_probability: float


@dataclass(frozen=True, slots=True)
class SealedRanking:
    pool_seal_sha256: str
    snapshot_sha256: str
    scores: tuple[CandidateScore, ...]
    ranked: tuple[RankedCandidate, ...]
    method_stream: tuple[str, ...]
    seal_sha256: str


@dataclass(frozen=True, slots=True)
class ControllerSeat:
    wave: int
    seat: int
    source: str
    sequence_id: str


@dataclass(frozen=True, slots=True)
class ChargedCallOutcome:
    logical_call: int
    sequence_id: str
    terminal_status: str
    charged: bool = True
    replacement_allowed: bool = False


@dataclass(frozen=True, slots=True)
class ComplianceEntry:
    sequence_id: str
    organizer_reference_receipt_sha256: str
    compliant: bool
    reason: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _require_sha(value: str, field: str) -> None:
    if type(value) is not str or len(value) != 64 or _SHA256_RE.fullmatch(value) is None:
        raise PosthocContractError(f"{field} must be a lowercase SHA-256")


def _require_builtin_sequence(value: object, field: str) -> None:
    if type(value) not in (list, tuple):
        raise PosthocContractError(f"{field} must be an exact built-in list or tuple")


def _require_false_flags(row: object) -> None:
    for name in (
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
    ):
        if not hasattr(row, name) or getattr(row, name) is not False:
            raise PosthocContractError(f"{name} must remain false")


def _validate_controller_seat_fields(row: ControllerSeat) -> None:
    if type(row) is not ControllerSeat:
        raise PosthocContractError("controller records must have the exact ControllerSeat type")
    if type(row.wave) is not int or type(row.seat) is not int:
        raise PosthocContractError("controller wave and seat must be exact integers")
    if type(row.source) is not str or len(row.source) > MAX_CONTROLLER_SOURCE_CHARACTERS:
        raise PosthocContractError("controller source must be a bounded exact string")
    _require_sha(row.sequence_id, "controller sequence_id")


def _read_bounded_contract(path: Path) -> bytes:
    """Capture one stable, single-link config without following links or blocking."""

    if type(path) is not type(Path()):
        raise PosthocContractError("adapter config path must have the exact platform Path type")
    raw_path = os.fspath(path)
    if (
        type(raw_path) is not str
        or not raw_path
        or len(raw_path) > MAX_PATH_CHARACTERS
        or "\x00" in raw_path
    ):
        raise PosthocContractError("adapter config path exceeds its bounded text contract")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise PosthocContractError("safe no-follow nonblocking config opens are unavailable")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(raw_path, flags)
    except OSError as exc:
        raise PosthocContractError("cannot safely open adapter config") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PosthocContractError("adapter config must be a single-link regular file")
        if before.st_size > MAX_ADAPTER_CONFIG_BYTES:
            raise PosthocContractError("adapter config exceeds its byte cap")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, MAX_ADAPTER_CONFIG_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_ADAPTER_CONFIG_BYTES:
                raise PosthocContractError("adapter config grew beyond its byte cap")
        after = os.fstat(descriptor)
        try:
            entry_after = os.stat(raw_path, follow_symlinks=False)
        except OSError as exc:
            raise PosthocContractError("adapter config changed during capture") from exc
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            total != before.st_size
            or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
            or any(getattr(after, field) != getattr(entry_after, field) for field in stable_fields)
        ):
            raise PosthocContractError("adapter config changed during capture")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_adapter_contract(path: Path) -> AdapterContract:
    """Authenticate and load the permanently blocked v1 declaration."""

    payload = _read_bounded_contract(path)
    if _sha256_bytes(payload) != FROZEN_ADAPTER_CONFIG_SHA256:
        raise PosthocContractError("adapter config hash mismatch")
    raw = tomllib.loads(payload.decode("utf-8"))
    false_fields = (
        "execution_authorized",
        "oracle_calls_authorized",
        "scientific_evidence_accepted",
        "production_input_eligible",
        "automatic_generator_mixture_eligible",
        "biological_superiority_claim_allowed",
        "external_request_identity_authenticated",
        "external_policy_member_identities_authenticated",
        "external_trajectory_receipts_authenticated",
        "external_initial64_snapshot_authenticated",
        "external_common_reserve_inventory_authenticated",
        "charged_call_evidence_present",
    )
    if raw.get("artifact") != ADAPTER_ARTIFACT or any(
        raw.get(k) is not False for k in false_fields
    ):
        raise PosthocContractError("adapter authorization invariant violated")
    if raw.get("accepted_policy_sets") != []:
        raise PosthocContractError("v1 accepted policy inventory must be empty")
    if raw.get("identity", {}).get("accepted_policy_count") != 0:
        raise PosthocContractError("accepted policy count must be zero")
    compliance = raw.get("compliance")
    generation = raw.get("generation")
    resources = raw.get("resources")
    scoring = raw.get("scoring")
    if (
        raw.get("charged_call_output") != "unauthenticated_structural_claim_only"
        or type(compliance) is not dict
        or compliance.get("organizer_reference_role")
        != "unauthenticated_post_generation_structural_claim_only"
        or compliance.get("compliance_claim_may_veto_later_shadow_or_production") is not False
        or compliance.get("maximum_reason_characters") != MAX_COMPLIANCE_REASON_CHARACTERS
        or compliance.get("maximum_claim_bytes") != MAX_COMPLIANCE_CLAIM_BYTES
        or type(generation) is not dict
        or generation.get("maximum_raw_output_utf8_bytes") != MAX_RAW_OUTPUT_UTF8_BYTES
        or generation.get("maximum_attempt_jsonl_line_bytes") != MAX_ATTEMPT_JSONL_LINE_BYTES
        or type(scoring) is not dict
        or scoring.get("degenerate_joint_support")
        != "unauthenticated_structural_affine_support_claim_only"
        or type(resources) is not dict
        or resources.get("maximum_adapter_config_bytes") != MAX_ADAPTER_CONFIG_BYTES
        or resources.get("maximum_path_characters") != MAX_PATH_CHARACTERS
        or resources.get("maximum_nonnegative_signed63") != MAX_SIGNED_63
    ):
        raise PosthocContractError("unauthenticated structural-claim boundary changed")
    return AdapterContract(
        config_sha256=FROZEN_ADAPTER_CONFIG_SHA256,
        accepted_policy_sets=(),
        execution_authorized=False,
        oracle_calls_authorized=False,
        scientific_evidence_accepted=False,
        production_input_eligible=False,
        automatic_generator_mixture_eligible=False,
    )


def preflight_real_sampler(*, policy_identity: str, asset_root: object) -> NoReturn:
    """Reject all v1 sampler requests before resolving ``asset_root``.

    ``asset_root`` deliberately has type ``object``: neither ``os.fspath`` nor a
    filesystem API is called.  Identity denials are reported before the empty
    accepted-policy denial, but both happen before any path resolution.
    """

    del asset_root
    if (
        type(policy_identity) is not str
        or not policy_identity
        or len(policy_identity) > MAX_POLICY_IDENTITY_CHARACTERS
    ):
        raise PosthocAdapterBlockedError(
            "a bounded exact-string policy identity is required; no path was resolved"
        )
    lowered = policy_identity.casefold()
    if any(token in lowered for token in _FORBIDDEN_POLICY_TOKENS):
        raise PosthocAdapterBlockedError(
            "fixture/no-go/incident/invalid policy identities are rejected; no path was resolved"
        )
    raise PosthocAdapterBlockedError(
        "accepted policy inventory is empty in v1; no path was resolved and a new contract is required"
    )


def sampler_preflight_record() -> SamplerPreflight:
    return SamplerPreflight(
        status="blocked_no_accepted_policy_or_campaign_assets",
        blockers=(
            "no_independently_accepted_ten_policy_set",
            "no_accepted_native_sampler_trajectory_receipt",
            "no_accepted_common_initial_64_posterior_snapshot",
        ),
        accepted_policy_count=0,
    )


def _validate_raw_output(raw: object) -> str:
    if type(raw) is not str or len(raw) > MAX_RAW_OUTPUT_CHARACTERS:
        raise PosthocContractError(
            f"every raw sampler output must be a string of at most {MAX_RAW_OUTPUT_CHARACTERS} characters"
        )
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PosthocContractError("raw sampler output is not valid UTF-8 text") from exc
    if len(encoded) > MAX_RAW_OUTPUT_UTF8_BYTES:
        raise PosthocContractError("raw sampler output exceeds its UTF-8 byte cap")
    return raw


def _validate_attempt_line_bytes(row: CandidateAttempt) -> None:
    if len(_canonical_json_bytes(asdict(row))) > MAX_ATTEMPT_JSONL_LINE_BYTES:
        raise PosthocContractError("attempt record exceeds the verifier JSONL line cap")


def build_attempt_grid(
    *,
    request_sha256: str,
    policy_member_sha256s: Sequence[str],
    sample_seed: int,
    sequences: Sequence[str],
    trajectory_receipt_sha256s: Sequence[str],
) -> tuple[CandidateAttempt, ...]:
    """Record the exact 2,048 by 8 output grid without interpreting trajectories."""

    _require_sha(request_sha256, "request_sha256")
    _require_builtin_sequence(policy_member_sha256s, "policy_member_sha256s")
    _require_builtin_sequence(sequences, "sequences")
    _require_builtin_sequence(trajectory_receipt_sha256s, "trajectory_receipt_sha256s")
    if type(sample_seed) is not int or not 0 <= sample_seed <= MAX_SIGNED_63:
        raise PosthocContractError("sample_seed must be a nonnegative signed-63 integer")
    if len(policy_member_sha256s) != POLICY_MEMBERS:
        raise PosthocContractError("exactly ten policy member identities are required")
    for member_sha in policy_member_sha256s:
        _require_sha(member_sha, "policy member identity")
    if len(set(policy_member_sha256s)) != POLICY_MEMBERS:
        raise PosthocContractError("policy member identities must be unique")
    if len(sequences) != ATTEMPT_COUNT or len(trajectory_receipt_sha256s) != ATTEMPT_COUNT:
        raise PosthocContractError(f"expected exactly {ATTEMPT_COUNT} sampler outputs")
    for raw in sequences:
        _validate_raw_output(raw)
    for receipt_sha in trajectory_receipt_sha256s:
        _require_sha(receipt_sha, "trajectory receipt identity")
    rows: list[CandidateAttempt] = []
    for ordinal, raw in enumerate(sequences):
        row = CandidateAttempt(
            slot=ordinal // ATTEMPTS_PER_SLOT,
            attempt=ordinal % ATTEMPTS_PER_SLOT,
            source_ordinal=ordinal,
            request_sha256=request_sha256,
            policy_member_index=ordinal % POLICY_MEMBERS,
            policy_member_sha256=policy_member_sha256s[ordinal % POLICY_MEMBERS],
            sample_seed=sample_seed,
            sample_counter=ordinal,
            trajectory_receipt_sha256=trajectory_receipt_sha256s[ordinal],
            raw_sequence=raw,
        )
        _validate_attempt_line_bytes(row)
        rows.append(row)
    return tuple(rows)


def _validate_id_set(values: Iterable[str], field: str) -> tuple[str, ...]:
    _require_builtin_sequence(values, field)
    if len(values) > ATTEMPT_COUNT:
        raise PosthocContractError(f"{field} exceeds the bounded inventory limit")
    materialized: list[str] = []
    for index, value in enumerate(values):
        if index >= ATTEMPT_COUNT:
            raise PosthocContractError(f"{field} exceeds the bounded inventory limit")
        _require_sha(value, field)
        materialized.append(value)
    try:
        return tuple(sorted(set(materialized)))
    except (TypeError, ValueError) as exc:
        raise PosthocContractError(f"{field} cannot be collected and sorted safely") from exc


def _pool_core_payload(pool: SealedPool | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(pool, SealedPool):
        return {
            "request_sha256": pool.request_sha256,
            "initial64_snapshot_sha256": pool.initial64_snapshot_sha256,
            "attempts": [asdict(row) for row in pool.attempts],
            "slots": [asdict(row) for row in pool.slots],
            "exact_training_overlap_ids": list(pool.exact_training_overlap_ids),
            "accepted_training_homology_ids": list(pool.accepted_training_homology_ids),
        }
    return pool


def seal_pool_once(
    *,
    attempts: Sequence[CandidateAttempt],
    initial64_snapshot_sha256: str,
    exact_training_overlap_ids: Iterable[str],
    accepted_training_homology_ids: Iterable[str],
) -> SealedPool:
    """Resolve every slot once; rejected attempts are charged and never replaced."""

    _require_builtin_sequence(attempts, "attempts")
    _require_sha(initial64_snapshot_sha256, "initial64_snapshot_sha256")
    if len(attempts) != ATTEMPT_COUNT:
        raise PosthocContractError("attempt grid is incomplete")
    for row in attempts:
        if type(row) is not CandidateAttempt:
            raise PosthocContractError("attempt records must have the exact CandidateAttempt type")
    request_sha = attempts[0].request_sha256
    _require_sha(request_sha, "request_sha256")
    if (
        type(attempts[0].sample_seed) is not int
        or not 0 <= attempts[0].sample_seed <= MAX_SIGNED_63
    ):
        raise PosthocContractError("sample seed must be a nonnegative signed-63 integer")
    training = _validate_id_set(exact_training_overlap_ids, "training sequence id")
    homology = _validate_id_set(accepted_training_homology_ids, "homology sequence id")
    training_set = set(training)
    homology_set = set(homology)
    excluded = training_set | homology_set
    accepted: set[str] = set()
    policy_members: dict[int, str] = {}
    trajectory_receipts: set[str] = set()
    sample_seed = attempts[0].sample_seed
    slots: list[ResolvedSlot] = []
    for ordinal, row in enumerate(attempts):
        _validate_raw_output(row.raw_sequence)
        _require_sha(row.request_sha256, "request_sha256")
        _require_sha(row.policy_member_sha256, "policy member identity")
        _require_sha(row.trajectory_receipt_sha256, "trajectory receipt identity")
        if (
            type(row.source_ordinal) is not int
            or type(row.slot) is not int
            or type(row.attempt) is not int
            or type(row.policy_member_index) is not int
            or type(row.sample_seed) is not int
            or type(row.sample_counter) is not int
            or row.source_ordinal != ordinal
            or row.slot != ordinal // 8
            or row.attempt != ordinal % 8
        ):
            raise PosthocContractError("attempt grid order/ordinal was changed")
        if row.request_sha256 != request_sha:
            raise PosthocContractError("attempt request identity mismatch")
        if (
            row.policy_member_index != ordinal % POLICY_MEMBERS
            or not 0 <= row.sample_seed <= MAX_SIGNED_63
            or row.sample_counter != ordinal
        ):
            raise PosthocContractError("policy member, seed, or sample counter was changed")
        if (
            policy_members.setdefault(row.policy_member_index, row.policy_member_sha256)
            != row.policy_member_sha256
        ):
            raise PosthocContractError("policy member identity changed within the fixed schedule")
        if row.sample_seed != sample_seed:
            raise PosthocContractError("sample seed changed within the fixed grid")
        if row.trajectory_receipt_sha256 in trajectory_receipts:
            raise PosthocContractError("trajectory receipt must be unique per sampler output")
        trajectory_receipts.add(row.trajectory_receipt_sha256)
        _require_false_flags(row)
        _validate_attempt_line_bytes(row)
    if (
        set(policy_members) != set(range(POLICY_MEMBERS))
        or len(set(policy_members.values())) != POLICY_MEMBERS
    ):
        raise PosthocContractError("attempt grid does not bind exactly ten distinct policy members")
    for slot in range(GRID_SLOTS):
        winner: tuple[int, str, str] | None = None
        last_reason = "attempt_budget_exhausted"
        for row in attempts[slot * 8 : (slot + 1) * 8]:
            try:
                sequence = canonicalize_sequence(row.raw_sequence, min_length=8, max_length=50)
            except (TypeError, ValueError):
                last_reason = "canonical_support_rejection"
                continue
            sequence_id = canonical_sequence_id(sequence)
            if sequence_id in excluded:
                last_reason = (
                    "exact_generator_training_overlap"
                    if sequence_id in training_set
                    else "accepted_training_homology_exclusion"
                )
                continue
            if sequence_id in accepted:
                last_reason = "duplicate_of_previously_accepted_sequence"
                continue
            winner = (row.source_ordinal, sequence, sequence_id)
            accepted.add(sequence_id)
            break
        if winner is None:
            slots.append(ResolvedSlot(slot, None, None, None, last_reason))
        else:
            source_ordinal, sequence, sequence_id = winner
            slots.append(ResolvedSlot(slot, source_ordinal, sequence, sequence_id, None))
    provisional = SealedPool(
        request_sha256=request_sha,
        initial64_snapshot_sha256=initial64_snapshot_sha256,
        attempts=tuple(attempts),
        slots=tuple(slots),
        exact_training_overlap_ids=training,
        accepted_training_homology_ids=homology,
        seal_sha256="",
    )
    seal = _sha256_bytes(_canonical_json_bytes(_pool_core_payload(provisional)))
    return SealedPool(
        request_sha256=provisional.request_sha256,
        initial64_snapshot_sha256=provisional.initial64_snapshot_sha256,
        attempts=provisional.attempts,
        slots=provisional.slots,
        exact_training_overlap_ids=provisional.exact_training_overlap_ids,
        accepted_training_homology_ids=provisional.accepted_training_homology_ids,
        seal_sha256=seal,
    )


def validate_initial64_snapshot(snapshot: Initial64PosteriorSnapshot) -> None:
    if type(snapshot) is not Initial64PosteriorSnapshot:
        raise PosthocContractError("snapshot must have the exact Initial64PosteriorSnapshot type")
    for field in ("snapshot_sha256", "initial_archive_sha256", "posterior_model_sha256"):
        _require_sha(getattr(snapshot, field), field)
    _require_false_flags(snapshot)
    if (
        type(snapshot.response_count) is not int
        or type(snapshot.adaptive_response_count) is not int
        or snapshot.response_count != INITIAL_RESPONSE_COUNT
        or snapshot.adaptive_response_count != 0
    ):
        raise PosthocContractError("snapshot must structurally claim only the common initial 64")
    if not _is_finite_number(snapshot.joint_chance_threshold):
        raise PosthocContractError("joint chance threshold must be a finite numeric value")
    if not 0 <= snapshot.joint_chance_threshold <= 1:
        raise PosthocContractError("joint chance threshold must be finite and in [0, 1]")
    if (
        type(snapshot.covariance_dimension) is not int
        or not 1 <= snapshot.covariance_dimension <= MAX_SIGNED_63
    ):
        raise PosthocContractError("covariance dimension must be a positive signed-63 integer")
    if (
        type(snapshot.covariance_rank) is not int
        or not 0 <= snapshot.covariance_rank <= MAX_SIGNED_63
        or snapshot.covariance_rank > snapshot.covariance_dimension
    ):
        raise PosthocContractError("covariance rank is invalid")
    if snapshot.affine_support_validated is not True:
        raise PosthocContractError(
            "joint covariance affine support validation is not structurally claimed"
        )


def _is_finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    if type(value) is int and not -MAX_SIGNED_63 <= value <= MAX_SIGNED_63:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _finite_midpoint(first: int | float, second: int | float) -> float:
    """Return an overflow-safe mean for two already finite values."""

    try:
        midpoint = float(first) * 0.5 + float(second) * 0.5
    except (OverflowError, ValueError) as exc:
        raise PosthocContractError("derived mean utility is nonfinite") from exc
    if not math.isfinite(midpoint):
        raise PosthocContractError("derived mean utility is nonfinite")
    return midpoint


def rank_pool_once(
    *, pool: SealedPool, snapshot: Initial64PosteriorSnapshot, scores: Sequence[CandidateScore]
) -> SealedRanking:
    """Freeze a stable post-hoc ranking; adaptive outcomes are not an input."""

    _require_builtin_sequence(scores, "scores")
    validate_initial64_snapshot(snapshot)
    validate_sealed_pool(pool)
    if pool.initial64_snapshot_sha256 != snapshot.snapshot_sha256:
        raise PosthocContractError("pool and posterior snapshot identity mismatch")
    candidate_by_id = {row.sequence_id: row for row in pool.candidates}
    if len(candidate_by_id) != len(pool.candidates):
        raise PosthocContractError("resolved pool contains duplicate identities")
    if len(scores) != len(candidate_by_id):
        raise PosthocContractError("score count must equal the bounded resolved pool")
    score_by_id: dict[str, CandidateScore] = {}
    for score in scores:
        if type(score) is not CandidateScore:
            raise PosthocContractError("scores must have the exact CandidateScore type")
        _require_sha(score.sequence_id, "score sequence_id")
        _require_sha(score.snapshot_sha256, "score snapshot_sha256")
        if type(score.source_ordinal) is not int:
            raise PosthocContractError("score source ordinal was changed")
        values = (
            score.gram_positive_mean,
            score.gram_negative_mean,
            score.joint_chance_probability,
        )
        if not all(_is_finite_number(value) for value in values):
            raise PosthocContractError("nonfinite scorer input")
        if not 0 <= score.joint_chance_probability <= 1:
            raise PosthocContractError("joint chance probability must be in [0, 1]")
        if score.sequence_id not in candidate_by_id or score.sequence_id in score_by_id:
            raise PosthocContractError("scores must map one-to-one onto the resolved pool")
        expected = candidate_by_id[score.sequence_id]
        if score.source_ordinal != expected.selected_source_ordinal:
            raise PosthocContractError("score source ordinal was changed")
        if score.snapshot_sha256 != snapshot.snapshot_sha256:
            raise PosthocContractError("score was not produced from the sealed initial-64 snapshot")
        score_by_id[score.sequence_id] = score
    if set(score_by_id) != set(candidate_by_id):
        raise PosthocContractError("every resolved candidate must be scored exactly once")
    canonical_scores = tuple(sorted(scores, key=lambda row: (row.source_ordinal, row.sequence_id)))
    utilities = {
        row.sequence_id: _finite_midpoint(row.gram_positive_mean, row.gram_negative_mean)
        for row in canonical_scores
    }
    ordered = sorted(
        canonical_scores,
        key=lambda row: (
            -(row.joint_chance_probability >= snapshot.joint_chance_threshold),
            -utilities[row.sequence_id],
            row.sequence_id,
            row.source_ordinal,
        ),
    )
    ranked = tuple(
        RankedCandidate(
            rank=index,
            sequence_id=row.sequence_id,
            source_ordinal=row.source_ordinal,
            chance_feasible=row.joint_chance_probability >= snapshot.joint_chance_threshold,
            mean_utility=utilities[row.sequence_id],
            joint_chance_probability=row.joint_chance_probability,
        )
        for index, row in enumerate(ordered, start=1)
    )
    feasible = tuple(row.sequence_id for row in ranked if row.chance_feasible)
    if len(feasible) < METHOD_SEATS:
        raise PosthocContractError(
            "fewer than 392 chance-feasible candidates; controller overflow required"
        )
    provisional = {
        "pool_seal_sha256": pool.seal_sha256,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "scores": [asdict(row) for row in canonical_scores],
        "ranked": [asdict(row) for row in ranked],
        "method_stream": list(feasible[:METHOD_SEATS]),
    }
    seal = _sha256_bytes(_canonical_json_bytes(provisional))
    return SealedRanking(
        pool_seal_sha256=pool.seal_sha256,
        snapshot_sha256=snapshot.snapshot_sha256,
        scores=canonical_scores,
        ranked=ranked,
        method_stream=feasible[:METHOD_SEATS],
        seal_sha256=seal,
    )


def validate_sealed_pool(pool: SealedPool) -> None:
    """Reconstruct a pool record so callers cannot inject a forged seal."""

    if type(pool) is not SealedPool:
        raise PosthocContractError("pool must have the exact SealedPool type")
    if (
        type(pool.attempts) is not tuple
        or type(pool.slots) is not tuple
        or type(pool.exact_training_overlap_ids) is not tuple
        or type(pool.accepted_training_homology_ids) is not tuple
        or len(pool.attempts) != ATTEMPT_COUNT
        or len(pool.slots) != GRID_SLOTS
    ):
        raise PosthocContractError("pool containers violate their exact bounded contract")
    rebuilt = seal_pool_once(
        attempts=pool.attempts,
        initial64_snapshot_sha256=pool.initial64_snapshot_sha256,
        exact_training_overlap_ids=pool.exact_training_overlap_ids,
        accepted_training_homology_ids=pool.accepted_training_homology_ids,
    )
    if _canonical_json_bytes(asdict(rebuilt)) != _canonical_json_bytes(asdict(pool)):
        raise PosthocContractError("pool contents or seal do not reconstruct")


def validate_sealed_ranking(
    ranking: SealedRanking, *, pool: SealedPool, snapshot: Initial64PosteriorSnapshot
) -> None:
    """Reconstruct a ranking before any controller or bundle consumer uses it."""

    if type(ranking) is not SealedRanking:
        raise PosthocContractError("ranking must have the exact SealedRanking type")
    if (
        type(ranking.scores) is not tuple
        or type(ranking.ranked) is not tuple
        or type(ranking.method_stream) is not tuple
        or len(ranking.scores) > GRID_SLOTS
        or len(ranking.ranked) > GRID_SLOTS
        or len(ranking.method_stream) != METHOD_SEATS
    ):
        raise PosthocContractError("ranking containers violate their exact bounded contract")
    rebuilt = rank_pool_once(pool=pool, snapshot=snapshot, scores=ranking.scores)
    if _canonical_json_bytes(asdict(rebuilt)) != _canonical_json_bytes(asdict(ranking)):
        raise PosthocContractError("ranking contents or seal do not reconstruct")


def freeze_controller_composition(
    ranking: SealedRanking,
    *,
    pool: SealedPool,
    snapshot: Initial64PosteriorSnapshot,
    common_reserve_ids: Sequence[Sequence[str]],
) -> tuple[ControllerSeat, ...]:
    """Map the immutable 392-method stream into the common 28-wave 14+2 boundary."""

    _require_builtin_sequence(common_reserve_ids, "common_reserve_ids")
    validate_sealed_ranking(ranking, pool=pool, snapshot=snapshot)
    if len(ranking.method_stream) != METHOD_SEATS or len(common_reserve_ids) != WAVES:
        raise PosthocContractError("controller requires 392 method IDs and 28 reserve pairs")
    seats: list[ControllerSeat] = []
    seen: set[str] = set()
    for wave in range(WAVES):
        _require_builtin_sequence(common_reserve_ids[wave], f"common reserve wave {wave}")
        if len(common_reserve_ids[wave]) != RESERVE_SEATS_PER_WAVE:
            raise PosthocContractError("each controller wave requires exactly two common reserves")
        reserve = tuple(common_reserve_ids[wave])
        method = ranking.method_stream[wave * 14 : (wave + 1) * 14]
        for seat, sequence_id in enumerate((*method, *reserve)):
            _require_sha(sequence_id, "controller sequence_id")
            if sequence_id in seen:
                raise PosthocContractError("controller identities must be globally unique")
            seen.add(sequence_id)
            seats.append(
                ControllerSeat(
                    wave=wave,
                    seat=seat,
                    source="categorical_diffusion_posthoc" if seat < 14 else "common_reserve",
                    sequence_id=sequence_id,
                )
            )
    return tuple(seats)


def build_unauthenticated_charged_call_claim(
    *, controller: Sequence[ControllerSeat], outcomes: Sequence[ChargedCallOutcome]
) -> bytes:
    """Build a non-authoritative structural claim about 448 adaptive calls."""

    _require_builtin_sequence(controller, "controller")
    _require_builtin_sequence(outcomes, "outcomes")
    if len(controller) != 448 or len(outcomes) != 448:
        raise PosthocContractError("charged-call ledger requires exactly 448 adaptive calls")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for ordinal, (seat, outcome) in enumerate(zip(controller, outcomes, strict=True)):
        if type(seat) is not ControllerSeat or type(outcome) is not ChargedCallOutcome:
            raise PosthocContractError("charged ledger record type mismatch")
        _validate_controller_seat_fields(seat)
        wave, seat_index = divmod(ordinal, 16)
        expected_source = "categorical_diffusion_posthoc" if seat_index < 14 else "common_reserve"
        if (
            type(seat.wave) is not int
            or type(seat.seat) is not int
            or seat.wave != wave
            or seat.seat != seat_index
            or seat.source != expected_source
            or seat.sequence_id in seen
        ):
            raise PosthocContractError("charged ledger controller mapping is invalid")
        seen.add(seat.sequence_id)
        _require_sha(outcome.sequence_id, "charged outcome sequence_id")
        if (
            type(outcome.terminal_status) is not str
            or len(outcome.terminal_status) > MAX_TERMINAL_STATUS_CHARACTERS
        ):
            raise PosthocContractError("charged outcome terminal status is invalid")
        if (
            type(outcome.logical_call) is not int
            or outcome.logical_call != INITIAL_RESPONSE_COUNT + ordinal
            or outcome.sequence_id != seat.sequence_id
            or outcome.terminal_status not in {"success", "submitted_failure"}
            or outcome.charged is not True
            or outcome.replacement_allowed is not False
        ):
            raise PosthocContractError("call identity/order/status/charging invariant violated")
        rows.append(asdict(outcome))
    return _canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact": "categorical_diffusion_posthoc_unauthenticated_charged_call_claim_v1",
            "status": "structural_claim_only",
            "claimed_outcomes": rows,
            "charged_call_evidence_present": False,
            "controller_authority_accepted": False,
            "external_inputs_authenticated": False,
            "execution_authorized": False,
            "oracle_calls_authorized": False,
            "scientific_evidence_accepted": False,
            "production_input_eligible": False,
        }
    )


def build_unauthenticated_compliance_claim(
    *,
    ranking: SealedRanking,
    pool: SealedPool,
    snapshot: Initial64PosteriorSnapshot,
    organizer_reference_receipt_sha256: str,
    entries: Sequence[ComplianceEntry],
) -> bytes:
    """Build a bounded, non-authoritative post-generation compliance claim."""

    validate_sealed_ranking(ranking, pool=pool, snapshot=snapshot)
    _require_sha(organizer_reference_receipt_sha256, "organizer_reference_receipt_sha256")
    _require_builtin_sequence(entries, "compliance entries")
    if len(entries) != len(ranking.ranked):
        raise PosthocContractError("compliance entry count must equal the sealed ranked inventory")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    reason_bytes = 0
    for entry in entries:
        if type(entry) is not ComplianceEntry:
            raise PosthocContractError(
                "compliance entries must have the exact ComplianceEntry type"
            )
        _require_sha(entry.sequence_id, "compliance sequence_id")
        _require_sha(
            entry.organizer_reference_receipt_sha256,
            "compliance organizer_reference_receipt_sha256",
        )
        if entry.organizer_reference_receipt_sha256 != organizer_reference_receipt_sha256:
            raise PosthocContractError("organizer receipt mismatch")
        if entry.sequence_id in seen:
            raise PosthocContractError("duplicate compliance identity")
        if (
            type(entry.compliant) is not bool
            or type(entry.reason) is not str
            or not entry.reason
            or len(entry.reason) > MAX_COMPLIANCE_REASON_CHARACTERS
        ):
            raise PosthocContractError("compliance decision fields are invalid")
        reason_bytes += len(entry.reason.encode("utf-8"))
        if reason_bytes > MAX_COMPLIANCE_CLAIM_BYTES:
            raise PosthocContractError("compliance reasons exceed the total byte cap")
        seen.add(entry.sequence_id)
        normalized.append(asdict(entry))
    intended = tuple(row.sequence_id for row in ranking.ranked)
    if tuple(sorted(seen)) != tuple(sorted(intended)) or len(seen) != len(intended):
        raise PosthocContractError("compliance claim must cover the exact sealed ranked inventory")
    body = {
        "schema_version": 1,
        "artifact": "categorical_diffusion_posthoc_unauthenticated_compliance_claim_v1",
        "status": "structural_claim_only",
        "claimed_ranking_sha256": ranking.seal_sha256,
        "claimed_sequence_ids_sha256": _sha256_bytes(_canonical_json_bytes(list(intended))),
        "claimed_organizer_reference_receipt_sha256": organizer_reference_receipt_sha256,
        "claimed_entries": normalized,
        "organizer_reference_authenticated": False,
        "organizer_reference_evidence_present": False,
        "external_inputs_authenticated": False,
        "may_veto_later_shadow_or_production": False,
        "may_enter_generation_fill_fit_score_rank_select_or_stop": False,
        "execution_authorized": False,
        "oracle_calls_authorized": False,
        "scientific_evidence_accepted": False,
        "production_input_eligible": False,
    }
    payload = _canonical_json_bytes(body)
    if len(payload) > MAX_COMPLIANCE_CLAIM_BYTES:
        raise PosthocContractError("compliance claim exceeds the total byte cap")
    return payload


def build_engineering_bundle(
    *,
    contract_bytes: bytes,
    pool: SealedPool,
    snapshot: Initial64PosteriorSnapshot,
    ranking: SealedRanking,
    controller: Sequence[ControllerSeat],
) -> Mapping[str, bytes]:
    """Create canonical, content-addressed bytes for independent reconstruction."""

    if type(contract_bytes) is not bytes:
        raise PosthocContractError("contract_bytes must have exact bytes type")
    if len(contract_bytes) > MAX_ADAPTER_CONFIG_BYTES:
        raise PosthocContractError("contract_bytes exceeds the adapter config byte cap")
    _require_builtin_sequence(controller, "controller")
    if _sha256_bytes(contract_bytes) != FROZEN_ADAPTER_CONFIG_SHA256:
        raise PosthocContractError("wrong adapter contract bytes")
    validate_initial64_snapshot(snapshot)
    validate_sealed_ranking(ranking, pool=pool, snapshot=snapshot)
    if len(controller) != WAVES * (METHOD_SEATS_PER_WAVE + RESERVE_SEATS_PER_WAVE):
        raise PosthocContractError("controller composition is incomplete")
    seen_controller: set[str] = set()
    expected_method: list[str] = []
    for ordinal, row in enumerate(controller):
        _validate_controller_seat_fields(row)
        wave, seat = divmod(ordinal, 16)
        expected_source = "categorical_diffusion_posthoc" if seat < 14 else "common_reserve"
        if (
            type(row.wave) is not int
            or type(row.seat) is not int
            or row.wave != wave
            or row.seat != seat
            or row.source != expected_source
        ):
            raise PosthocContractError("controller 14+2 seat mapping was changed")
        if row.sequence_id in seen_controller:
            raise PosthocContractError("controller identities must be globally unique")
        seen_controller.add(row.sequence_id)
        if row.source == "categorical_diffusion_posthoc":
            expected_method.append(row.sequence_id)
    if tuple(expected_method) != ranking.method_stream:
        raise PosthocContractError("controller does not contain the immutable method stream")
    context = {
        "request_sha256": pool.request_sha256,
        "initial64_snapshot_sha256": pool.initial64_snapshot_sha256,
        "exact_training_overlap_ids": list(pool.exact_training_overlap_ids),
        "accepted_training_homology_ids": list(pool.accepted_training_homology_ids),
        "pool_seal_sha256": pool.seal_sha256,
    }
    payloads: dict[str, bytes] = {
        "adapter-contract.toml": contract_bytes,
        "pool-context.json": _canonical_json_bytes(context),
        "attempts.jsonl": _canonical_jsonl_bytes(asdict(row) for row in pool.attempts),
        "resolved-pool.jsonl": _canonical_jsonl_bytes(asdict(row) for row in pool.slots),
        "posterior-snapshot.json": _canonical_json_bytes(asdict(snapshot)),
        "scores.jsonl": _canonical_jsonl_bytes(asdict(row) for row in ranking.scores),
        "ranking.jsonl": _canonical_jsonl_bytes(asdict(row) for row in ranking.ranked),
        "controller-composition.jsonl": _canonical_jsonl_bytes(asdict(row) for row in controller),
    }
    manifest = {
        "schema_version": 1,
        "artifact": "categorical_diffusion_posthoc_engineering_bundle_v1",
        "status": "blocked_engineering_only",
        "evidence_class": "structural_only_external_inputs_unauthenticated",
        "pool_seal_sha256": pool.seal_sha256,
        "ranking_seal_sha256": ranking.seal_sha256,
        "method_stream": list(ranking.method_stream),
        "execution_authorized": False,
        "oracle_calls_authorized": False,
        "scientific_evidence_accepted": False,
        "production_input_eligible": False,
        "organizer_reference_present": False,
        "external_request_policy_trajectory_and_snapshot_authenticated": False,
        "common_reserve_authority_accepted": False,
        "charged_call_evidence_present": False,
        "payload_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
    }
    payloads["manifest.json"] = _canonical_json_bytes(manifest)
    payloads["SHA256SUMS"] = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")
    return MappingProxyType(payloads)
